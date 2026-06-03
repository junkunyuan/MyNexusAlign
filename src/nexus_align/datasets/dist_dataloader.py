"""Dataloader construction: distributed dataloader builder."""

import torch


def build_dataloader(cfg, dataset, mode: str = "train") -> torch.utils.data.DataLoader:
    """
    Build a distributed dataloader (sampler available as .sampler) and print a data summary.

    mode ("train"/"valid"/"eval") selects the cfg.algorithm.{mode} section:
    {mode}_batch_size (required), plus shuffle/drop_last (default false) and
    grad_accu_step (default 1).
    """
    if mode not in ("train", "valid", "eval"):
        raise ValueError(f"❌ Unknown dataloader mode: {mode}")
    rank = cfg.common.rank
    world_size = cfg.common.world_size
    mode_cfg = cfg.algorithm[mode]
    batch_size = mode_cfg[f"{mode}_batch_size"]
    batch_size_per_rank = batch_size // world_size
    shuffle = mode_cfg.get("shuffle", False)
    drop_last = mode_cfg.get("drop_last", False)
    grad_accu_step = mode_cfg.get("grad_accu_step", 1)
    sample_ratio = cfg.data.sample_ratio
    cache_dir = cfg.data.cache_dir

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_rank,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=drop_last
    )

    # Print dataset info
    effective_batch_size = batch_size_per_rank * world_size * grad_accu_step
    title = {"train": "Training", "valid": "Validation", "eval": "Evaluation"}[mode]
    info = [f"\n📚 {title} data:"]
    info += [f"    dataset: {cfg.data.name}"]
    info += [f"    sample_count: {len(dataset)}"]
    info += [f"    batch_size_per_rank: {batch_size_per_rank}"]
    info += [f"    effective_batch_size: {effective_batch_size}"]
    info += [f"    batch_count: {len(dataloader)}"]
    if sample_ratio != 1.:
        info += [f"    sample_ratio: {sample_ratio}"]
    if drop_last:
        info += ["    drop_last: true"]
    if cache_dir:
        info += [f"    cache_dir: {cache_dir}"]
    print("\n".join(info))

    return dataloader
