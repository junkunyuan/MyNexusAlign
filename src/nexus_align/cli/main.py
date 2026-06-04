"""CLI entry: Hydra-configured entry point with environment setup."""

import os
from omegaconf import OmegaConf

import hydra
import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

import nexus_align.models  # noqa: F401  # registers model factories on import
import nexus_align.datasets  # noqa: F401  # registers dataset factories on import
import nexus_align.algorithms  # noqa: F401  # registers algorithm factories on import
from nexus_align.registry import registry
from nexus_align.engine.setup import with_env_setup
from nexus_align.engine.distributed import all_reduce
from nexus_align.datasets.dist_dataloader import build_dataloader


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
@with_env_setup
def main(cfg, device):
    rank = cfg.common.rank

    # 1. Prepare dataset
    train_dataset = registry.get("dataset", cfg.data.name)(cfg)
    dataloader = build_dataloader(cfg, train_dataset, mode="train")
    print("✅ Prepared training dataset")

    # 2. Prepare models (FSDP-wrapped with an EMA copy; see BaseModel)
    model_wrapper = registry.get("model", cfg.model.name)(cfg, device)
    model = model_wrapper.model
    print("✅ Prepared model")

    # 3. Prepare algorithms
    loss_fn = registry.get("algorithm", cfg.algorithm.name)(cfg)
    print("✅ Prepared algorithm")

    # --------------------------------------------------------------------------------
    # 4. Prepare running
    # --------------------------------------------------------------------------------
    train_cfg = cfg.algorithm.train
    grad_accu_step = train_cfg.grad_accu_step

    # NOTE: the optimizer must be built on the FSDP-wrapped parameters.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
        weight_decay=train_cfg.adam_weight_decay,
        eps=train_cfg.adam_epsilon,
    )

    # Mixed precision is handled by FSDP (see BaseModel); only fp16 needs a scaler.
    scaler = ShardedGradScaler(enabled=(train_cfg.mixed_precision == "fp16"))

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
        model_wrapper.load_state_dict(ckpt)
        model_wrapper.load_optim_state_dict(optimizer, ckpt["opt"])
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
        dataloader.sampler.set_epoch(epoch)
        data_iter = iter(dataloader)

        # In a resumed epoch, skip the batches already consumed.
        steps_done = global_step % steps_per_epoch if epoch == start_epoch else 0
        for _ in range(steps_done * grad_accu_step):
            next(data_iter, None)

        for _ in range(steps_done, steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accu_step):
                moments, labels = next(data_iter)
                moments = moments.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.no_grad():
                    x = DiagonalGaussianDistribution(moments).sample()
                    x = x * latents_scale + latents_bias

                # FSDP reduce-scatters and accumulates sharded grads each micro-step
                # (no_sync() is incompatible with use_orig_params=True).
                loss, loss_ref = loss_fn(model, x, dict(y=labels))
                loss_mean = loss.mean()
                loss_mean_ref = loss_ref.mean()
                scaler.scale(loss_mean / grad_accu_step).backward()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = model.clip_grad_norm_(train_cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            model_wrapper.ema_step(decay=train_cfg.ema_decay)
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
                # Gather full state dicts on all ranks; only rank 0 saves.
                state_dict = model_wrapper.state_dict()
                optim_state_dict = model_wrapper.optim_state_dict(optimizer)
                if rank == 0:
                    checkpoint = {
                        "model": state_dict["model"],
                        "ema": state_dict["ema"],
                        "opt": optim_state_dict,
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
