"""Environment setup: distributed init, logging, seed, and config for train/eval."""

import os
import pyinstrument
from typing import Optional
from dataclasses import dataclass
from omegaconf import OmegaConf, open_dict

import torch

from nexus_align.engine.logger import init_log
from nexus_align.engine.tracker import init_wandb
from nexus_align.engine.distributed import init_dist_env
from nexus_align.utils.seed import set_seed, set_deterministic


@dataclass
class EnvContext:
    """Context returned after environment preparation."""

    world_size: int
    rank: int
    device: torch.device
    cfg_dict: dict
    seed: int
    profiler: Optional[pyinstrument.Profiler]


def prepare_env(cfg) -> EnvContext:
    """Prepare common environment: distributed init, logging, seed, config."""
    # Initialize distributed environment
    world_size, rank, device = init_dist_env()

    # For time cost analysis
    profiler = None
    if rank == 0 and cfg.common.time_cost_analysis:
        profiler = pyinstrument.Profiler()
        profiler.start()

    # Update configs
    with open_dict(cfg):
        base_log_dir = cfg.log.log_dir
        log_dir = os.path.join(base_log_dir, cfg.log.exp_info)
        cfg.common.project_path = os.getcwd().split(log_dir)[0]
        cfg.log.log_dir = os.path.join(cfg.common.project_path, log_dir)  # use abs path

    # Initialize logger and wandb
    init_log(
        exp_info=cfg.log.exp_info,
        debug_console=cfg.common.debug_console,
        exp_dir=os.path.join(cfg.log.log_dir, "logs"),
    )
    wandb_init = init_wandb(
        entity=cfg.log.wandb.entity,
        project=cfg.log.wandb.project,
        name=cfg.log.wandb.name,
        wandb_offline=cfg.log.wandb.wandb_offline or cfg.common.debug,
    )
    with open_dict(cfg):
        cfg.log.wandb.wandb_init = wandb_init

    # Fix seed
    seed = cfg.common.seed
    if cfg.exp_mode == "train":
        seed = cfg.common.seed + rank
        cfg.common.seed = seed
    set_seed(seed)
    set_deterministic(cfg.common.deterministic)

    # Save configs
    if rank == 0:
        config_save_path = os.path.join(cfg.log.log_dir, "config.yaml")
        OmegaConf.save(config=cfg, resolve=True, f=config_save_path)
        print(f"💾 Saved configs to <{config_save_path}>")

    cfg_dict = OmegaConf.to_container(cfg)

    return EnvContext(
        world_size=world_size,
        rank=rank,
        device=device,
        cfg_dict=cfg_dict,
        seed=seed,
        profiler=profiler,
    )


def with_env_setup(main_fn):
    """Decorator that runs prepare_env first and passes EnvContext.

    The decorated function must accept (cfg, env: EnvContext).
    """

    def wrapper(cfg):
        env = prepare_env(cfg)
        return main_fn(cfg, env)

    return wrapper
