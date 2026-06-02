"""CLI entry: Hydra-configured entry point with environment setup."""

import os
from copy import deepcopy
from omegaconf import OmegaConf
from contextlib import nullcontext

import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

import nexus_align.models  # noqa: F401  # registers model factories on import
import nexus_align.datasets  # noqa: F401  # registers dataset factories on import
import nexus_align.algorithms  # noqa: F401  # registers algorithm factories on import
from nexus_align.registry import registry
from nexus_align.engine.setup import with_env_setup
from nexus_align.engine.distributed import all_reduce


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current (DDP-wrapped) model."""
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        ema_params[name.replace("module.", "")].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad on all parameters of a model."""
    for p in model.parameters():
        p.requires_grad = flag


def amp_autocast(dtype):
    """CUDA autocast context for the given dtype, or a no-op when dtype is None."""
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
@with_env_setup
def main(cfg, device):
    rank = cfg.common.rank
    world_size = cfg.common.world_size

    # --------------------------------------------------------------------------------
    # 1. Prepare dataset
    # --------------------------------------------------------------------------------
    train_batch_size = cfg.algorithm.train.train_batch_size
    grad_accu_step = cfg.algorithm.train.grad_accu_step
    sample_ratio = cfg.data.sample_ratio
    drop_last = cfg.data.drop_last
    cache_dir = cfg.data.cache_dir

    train_dataset = registry.get("dataset", cfg.data.name)(
        cfg.data.lmdb_path,
        flip_prob=cfg.data.flip_prob,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size // world_size,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    coef = world_size * grad_accu_step
    total_train_batch_size = train_batch_size * coef
    info = ["\n📚 Training data:"]
    info += [f"    dataset: {cfg.data.name}"]
    info += [f"    sample count: {len(train_dataset)}"]
    info += [f"    batchsize: {train_batch_size}"]
    info += [f"    total batchsize: {total_train_batch_size}"]
    if sample_ratio != 1.:
        info += [f"    sample_ratio: {sample_ratio}"]
    if drop_last:
        info += ["    drop_last: true"]
    if cache_dir:
        info += [f"    cache_dir: {cache_dir}"]
    print("\n".join(info))
    print("✅ Prepared training dataset")

    # --------------------------------------------------------------------------------
    # 2. Prepare models
    # --------------------------------------------------------------------------------
    model = registry.get("model", cfg.model.name)(
        input_size=cfg.model.resolution // 8,
        num_classes=cfg.model.num_classes,
        use_cfg=cfg.model.cfg_prob > 0,
    ).to(device)
    print("✅ Prepared model")
    
    # --------------------------------------------------------------------------------
    # 3. Prepare algorithms
    # --------------------------------------------------------------------------------
    loss_fn = registry.get("algorithm", cfg.algorithm.name)(
        label_dropout_prob=cfg.model.cfg_prob,
        **cfg.algorithm.loss,
    )
    print("✅ Prepared algorithm")

    # --------------------------------------------------------------------------------
    # 4. Prepare running
    # --------------------------------------------------------------------------------
    train_cfg = cfg.algorithm.train

    if train_cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=train_cfg.adam_weight_decay,
        eps=train_cfg.adam_epsilon,
    )

    # Wrap in DDP (broadcasts rank-0 weights so all ranks share one starting point),
    # then snapshot EMA so every rank starts from the same parameters.
    model.train()  # enables label-embedding dropout for classifier-free guidance
    model = DDP(model, device_ids=[device.index])
    ema = deepcopy(model.module).to(device)
    requires_grad(ema, False)
    ema.eval()

    # Mixed precision: bf16/no need no GradScaler; fp16 does.
    amp_dtype = {"no": None, "fp16": torch.float16, "bf16": torch.bfloat16}[train_cfg.mixed_precision]
    scaler = torch.amp.GradScaler("cuda", enabled=(train_cfg.mixed_precision == "fp16"))

    steps_per_epoch = max(len(dataloader) // grad_accu_step, 1)
    max_train_steps = train_cfg.epochs * steps_per_epoch

    # VAE latent normalization (trick from IMM).
    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(device)

    ckpt_dir = os.path.join(cfg.log.log_dir, "checkpoints")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Resume from a checkpoint if requested.
    global_step = 0
    if train_cfg.resume_step > 0:
        ckpt_path = os.path.join(ckpt_dir, f"{train_cfg.resume_step:07d}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["opt"])
        global_step = ckpt["steps"]
        print(f"Loaded checkpoint from {ckpt_path} (step={global_step})")

    start_epoch = global_step // steps_per_epoch
    print(f"steps_per_epoch={steps_per_epoch}, max_train_steps={max_train_steps}")
    print("✅ Prepared running")

    # --------------------------------------------------------------------------------
    # 5. Run
    # --------------------------------------------------------------------------------
    for epoch in range(start_epoch, train_cfg.epochs):
        model.train()
        sampler.set_epoch(epoch)
        data_iter = iter(dataloader)

        # In a resumed epoch, skip the batches already consumed.
        steps_done = global_step % steps_per_epoch if epoch == start_epoch else 0
        for _ in range(steps_done * grad_accu_step):
            next(data_iter, None)

        for _ in range(steps_done, steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for micro_step in range(grad_accu_step):
                moments, labels = next(data_iter)
                moments = moments.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.no_grad():
                    x = DiagonalGaussianDistribution(moments).sample()
                    x = x * latents_scale + latents_bias

                # Sync gradients only on the last micro-step.
                sync_ctx = nullcontext() if micro_step == grad_accu_step - 1 else model.no_sync()
                with sync_ctx:
                    with amp_autocast(amp_dtype):
                        loss, loss_ref = loss_fn(model, x, dict(y=labels))
                        loss_mean = loss.mean()
                        loss_mean_ref = loss_ref.mean()
                    scaler.scale(loss_mean / grad_accu_step).backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema, model, decay=train_cfg.ema_decay)
            global_step += 1

            if global_step % 100 == 0:
                loss_v = all_reduce(loss_mean.detach(), "mean").item()
                loss_ref_v = all_reduce(loss_mean_ref.detach(), "mean").item()
                grad_v = all_reduce(grad_norm.detach(), "mean").item()
                print(f"Step {global_step}: loss = {loss_v:.4f}, loss_ref = {loss_ref_v:.4f}, grad_norm = {grad_v:.4f}")

            should_save = (
                (global_step % train_cfg.checkpointing_steps == 0)
                or global_step >= max_train_steps
            )
            if should_save:
                dist.barrier()
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "config": OmegaConf.to_container(cfg, resolve=True),
                        "steps": global_step,
                    }
                    ckpt_path = os.path.join(ckpt_dir, f"{global_step:07d}.pt")
                    torch.save(checkpoint, ckpt_path)
                    print(f"Saved checkpoint to {ckpt_path}")
                dist.barrier()

            if global_step >= max_train_steps:
                break

        print(f"Completed epoch {epoch + 1}/{train_cfg.epochs}")
        if global_step >= max_train_steps:
            break

    dist.barrier()
    print("✅ Training completed")


if __name__ == "__main__":
    main()
