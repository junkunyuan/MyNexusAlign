"""CLI entry: Hydra-configured entry point with environment setup."""

import os

import hydra
import torch

import nexus_align.models  # noqa: F401  # registers model factories on import
import nexus_align.datasets  # noqa: F401  # registers dataset factories on import
from nexus_align.registry import registry
from nexus_align.engine.setup import with_env_setup


@hydra.main(
    config_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs")),
    config_name="main",
    version_base="1.3",
)
@with_env_setup
def main(cfg, env):
    world_size, rank, device = env.world_size, env.rank, env.device
    cfg_dict = env.cfg_dict

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

    # --------------------------------------------------------------------------------
    # 4. Prepare running
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # 5. Run
    # --------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
