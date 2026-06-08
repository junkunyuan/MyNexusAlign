"""Metric tracking: scalar logging to Tensorboard and wandb."""

import os
import json

import wandb
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def init_wandb(entity: str, project: str, name: str, wandb_offline: bool, log_dir: str) -> bool:
    """Initialize wandb (Weights & Biases: https://wandb.ai/site).

    Tries online mode first, then falls back to offline. Only rank 0 runs init.

    Args:
        entity: wandb entity.
        project: wandb project.
        name: wandb run name.
        wandb_offline: If True, wandb is initialized in offline mode.
        log_dir: Directory where wandb files are saved.

    Returns:
        True if wandb was initialized successfully, False otherwise.
    """
    wandb_init = False
    if dist.get_rank() == 0:
        init_params = {"entity": entity, "project": project, "name": name, "dir": log_dir}

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
                print("⚠️ Continue without wandb logging")
    dist.barrier()

    return wandb_init


class TrainingTracker:
    """Track training metrics to TensorBoard and/or Wandb (rank 0 only)."""

    def __init__(self, log_dir: str, wandb_init: bool) -> None:
        self.log_dir = log_dir
        self.wandb_init = wandb_init
        self.tb_writer = None
        self.save_result_path = os.path.join(log_dir, "results.jsonl")

    def open_writer(self) -> None:
        """Open the TensorBoard writer (rank 0 only)."""
        if dist.get_rank() != 0:
            return
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
        """Close the TensorBoard writer and finish wandb."""
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
        epoch: int | None = None,
    ) -> None:
        """Track one metric against the total_step and/or epoch x-axis."""
        x_axes = {"total_step": total_step, "epoch": epoch}
        for axis, x in x_axes.items():
            if x is None:
                continue
            key = f"{axis}/{name}"
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(key, scalar_value=value, global_step=x)
            if self.wandb_init:
                wandb.log({key: value}, step=x)

    def track_all(
        self,
        metrics: dict,
        total_step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Track every metric in the dict against the total_step and/or epoch x-axis."""
        for name, value in metrics.items():
            self.track(name, value, total_step=total_step, epoch=epoch)

    def save_metrics(self, metrics: dict) -> None:
        """Append the latest metrics to a JSONL file."""
        if dist.get_rank() != 0:
            return
        try:
            with open(self.save_result_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"⚠️ Failed to append metrics to JSONL: {e}")
