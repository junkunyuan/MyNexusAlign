"""Environment setup: distributed init, logging, seed, and config for train/eval."""

import os
from datetime import datetime

import yaml
from omegaconf import OmegaConf, open_dict

import torch
import torch.distributed as dist

from nexus_align.engine.logger import init_log
from nexus_align.engine.tracker import init_wandb
from nexus_align.engine.distributed import init_dist_env
from nexus_align.utils.seed import set_seed, set_deterministic


def prepare_env(cfg) -> None:
    """Prepare common environment: distributed init, logging, seed, config."""
    # Initialize distributed environment
    world_size, rank = init_dist_env()

    # Update configs
    with open_dict(cfg):
        if cfg.log.get(timestamp, True):
            ts = [datetime.now().strftime("%Y%m%d-%H%M%S")]
            dist.broadcast_object_list(ts, src=0)
            cfg.log.exp_info = f"{cfg.log.exp_info}_{ts[0]}"
        cfg.common.project_path = os.getcwd()
        cfg.log.log_dir = os.path.abspath(os.path.join(cfg.log.log_dir, cfg.log.exp_info))
        cfg.common.world_size = world_size
        cfg.common.rank = rank

    # Initialize logger and wandb
    init_log(
        exp_info=cfg.log.exp_info,
        debug_console=cfg.common.debug,
        exp_dir=os.path.join(cfg.log.log_dir, "logs"),
    )
    wandb_init = init_wandb(
        entity=cfg.log.wandb.entity,
        project=cfg.log.wandb.project,
        name=cfg.log.wandb.name,
        wandb_offline=cfg.log.wandb.wandb_offline or cfg.common.debug,
        log_dir=cfg.log.log_dir,
    )
    with open_dict(cfg):
        cfg.log.wandb.wandb_init = wandb_init

    # Fix seed
    seed = cfg.common.seed
    if cfg.common.exp_mode == "train":
        seed = cfg.common.seed + rank
        cfg.common.seed = seed
    set_seed(seed)
    set_deterministic(cfg.common.deterministic)

    # Allow TF32 on matmul and cudnn for faster training on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Save configs
    if rank == 0:
        config_save_path = os.path.join(cfg.log.log_dir, "config.yaml")
        container = OmegaConf.to_container(cfg, resolve=True)
        with open(config_save_path, "w", encoding="utf-8") as f:
            yaml.dump(container, f, sort_keys=True, allow_unicode=True, indent=2)
        print(f"💾 Saved configs to <{config_save_path}>")


def with_env_setup(main_fn):
    """Prepare environment before running."""

    def wrapper(cfg):
        prepare_env(cfg)
        return main_fn(cfg)

    return wrapper
