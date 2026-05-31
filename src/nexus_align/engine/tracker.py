"""Metrics tracking: TensorBoard and Wandb scalar logging (rank 0 only)."""

import os
import json

import wandb
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from nexus_align.engine.meter import WindowMeter


def init_wandb(entity: str, project: str, name: str, wandb_offline: bool) -> bool:
    """Initialize Weights & Biases (https://wandb.ai/site).

    Tries online mode first, then falls back to offline. Only rank 0 runs init.

    Args:
        entity: Weights & Biases entity.
        project: Weights & Biases project.
        name: Weights & Biases run name.
        wandb_offline: If True, wandb is initialized in offline mode.

    Returns:
        True if wandb was initialized successfully, False otherwise.
    """
    wandb_init = False
    if dist.get_rank() == 0:
        init_params = {"entity": entity, "project": project, "name": name}

        if not wandb_offline:
            try:
                wandb.init(mode="online", **init_params)
                wandb_init = True
                print("✅ Wandb initialized (online mode)")
            except Exception as e:
                print(f"⚠️ Wandb online init failed:\n{e}")

        if not wandb_init:
            try:
                wandb.init(mode="offline", **init_params)
                wandb_init = True
                print("✅ Wandb initialized (offline mode)")
            except Exception as e:
                print(f"⚠️ Wandb offline init failed:\n{e}")
                print("⚠️ Training will continue without wandb logging")
    dist.barrier()

    return wandb_init


class TrainingTracker:
    """Track training metrics to TensorBoard and/or Wandb (rank 0 only)."""

    def __init__(self, log_dir: str, wandb_init: bool) -> None:
        self.log_dir = log_dir
        self.wandb_init = wandb_init
        self.tb_writer = None
        self._meters = None  # optional; set via set_meters() for log_all_meters
        self.save_result_path = os.path.join(log_dir, "results.jsonl")

    def set_meters(self, meters) -> None:
        """Set the meters instance used by log_all_meters (optional)."""
        self._meters = meters

    def open_writer(self) -> None:
        """Open metric writer."""
        if dist.get_rank() == 0:
            try:
                tb_dir = os.path.join(self.log_dir, "tensorboard")
                os.makedirs(tb_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                print("✅ TensorBoard used")
            except Exception as e:
                print(f"⚠️ TensorBoard init failed: {e}")
            if self.wandb_init:
                print("✅ Wandb used")

    def close_writer(self) -> None:
        """Close metric writer."""
        if self.tb_writer is not None:
            self.tb_writer.close()
            self.tb_writer = None
        if self.wandb_init:
            wandb.finish()

    def track(
        self, 
        name: str, 
        value: float, 
        total_step: int | None = None,
        epoch: int | None = None
    ) -> None:
        """Track one metric."""
        log_xs = {"total_step":total_step, "epoch": epoch}
        for log_x_name, log_x in log_xs.items(): 
            if log_x != None:
                log_key = f"{log_x_name}/{name}"
                if self.tb_writer is not None:    
                    self.tb_writer.add_scalar(
                        log_key, 
                        scalar_value=value, 
                        global_step=log_x
                    )
                if self.wandb_init:
                    wandb.log({log_key: value}, step=log_x)

    def track_all(
        self, 
        exp_info: dict, 
        total_step: int | None = None,
        epoch: int | None = None
    ) -> None:
        """Track all metrics."""
        for n, v in exp_info.items():
            self.track(n, v, total_step=total_step, epoch=epoch)
        if meters is not None and dist.get_rank() == 0:
            print(meters.info())
            self._append_meters_to_jsonl(meters)

    def save_metrics(self, exp_info: dict) -> None:
        """Save the latest results to a JSONL file."""
        try:
            with open(self.save_result_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(exp_info, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"⚠️ Failed to add results to JSONL: {e}")
