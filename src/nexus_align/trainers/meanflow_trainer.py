"""MeanFlow trainer: FSDP training on VAE latents with mixed precision, EMA, and checkpointing."""

import os
from contextlib import nullcontext
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from nexus_align.engine.distributed import all_reduce
from nexus_align.trainers.base_trainer import BaseTrainer


class MeanFlowTrainer(BaseTrainer):
    """Train a MeanFlow model: sample VAE latents, regress the MeanFlow loss."""

    def __init__(self, cfg, train_dataloader, model, algorithm) -> None:
        super().__init__(cfg, train_dataloader, model, algorithm)
        train_cfg = self.train_cfg
        device = model.device

        # NOTE: the optimizer must be built on the FSDP-wrapped parameters.
        self.optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=train_cfg.learning_rate,
            betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
            weight_decay=train_cfg.adam_weight_decay,
            eps=train_cfg.adam_epsilon,
        )

        # Mixed precision is handled by FSDP (see BaseModel); only fp16 needs a scaler.
        self.scaler = ShardedGradScaler(enabled=(train_cfg.mixed_precision == "fp16"))

        # VAE latent normalization (trick from IMM).
        self.latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
        self.latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(device)

        self.ckpt_dir = os.path.join(cfg.log.log_dir, "checkpoints")
        if cfg.common.rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def train_mode(self):
        self.model.model.train()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def forward(self, data):
        device = self.model.device
        moments, labels = data
        moments = moments.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            x = DiagonalGaussianDistribution(moments).sample()
            x = x * self.latents_scale + self.latents_bias

        loss, loss_ref = self.algorithm(self.model.model, x, dict(y=labels))
        loss_mean = loss.mean()
        self._loss = loss_mean.detach()
        self._loss_ref = loss_ref.mean().detach()
        return {"loss": loss_mean}

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def no_sync(self):
        # FSDP reduce-scatters and accumulates sharded grads each micro-step
        # (no_sync() is incompatible with use_orig_params=True).
        return nullcontext()

    def clip_grad(self):
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
        grad_norm = self.model.model.clip_grad_norm_(self.train_cfg.max_grad_norm)
        self._grad_norm = grad_norm.detach()

    def optimizer_step(self):
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def ema_step(self):
        self.model.ema_step(decay=self.train_cfg.ema_decay)

    def load_checkpoint(self):
        if self.train_cfg.resume_step <= 0:
            return
        ckpt_path = os.path.join(self.ckpt_dir, f"{self.train_cfg.resume_step:07d}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.model.load_optim_state_dict(self.optimizer, ckpt["opt"])
        self.total_step = ckpt["steps"]
        print(f"Loaded checkpoint from {ckpt_path} (step={self.total_step})")

    def save_checkpoint(self):
        should_save = (
            (self.total_step % self.train_cfg.checkpointing_steps == 0)
            or self.total_step >= self.max_total_steps
        )
        if not should_save:
            return
        dist.barrier()
        # Gather full state dicts on all ranks; only rank 0 saves.
        state_dict = self.model.state_dict()
        optim_state_dict = self.model.optim_state_dict(self.optimizer)
        if self.cfg.common.rank == 0:
            checkpoint = {
                "model": state_dict["model"],
                "ema": state_dict["ema"],
                "opt": optim_state_dict,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "steps": self.total_step,
            }
            ckpt_path = os.path.join(self.ckpt_dir, f"{self.total_step:07d}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        dist.barrier()

    def update_log(self):
        if self.total_step % 100 != 0:
            return
        loss = all_reduce(self._loss, "mean").item()
        loss_ref = all_reduce(self._loss_ref, "mean").item()
        grad_norm = all_reduce(self._grad_norm, "mean").item()
        print(f"Step {self.total_step}: loss = {loss:.4f}, loss_ref = {loss_ref:.4f}, grad_norm = {grad_norm:.4f}")
