"""Base trainer: abstract interface and the unified training loop."""

from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from nexus_align.models.base_model import BaseModel
from nexus_align.algorithms.base_algorithm import BaseAlgorithm
from nexus_align.datasets.dist_dataloader import build_dataloader


class BaseTrainer(ABC):
    """
    Base trainer: run() drives the training loop; subclasses fill in the steps.
    """

    def __init__(
        self,
        cfg,
        train_dataset: Dataset = None,
        model: BaseModel = None,
        algorithm: BaseAlgorithm = None,
    ) -> None:
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.model = model
        self.algorithm = algorithm
        self.optimizer = None
        self.lr_scheduler = None

        self.train_cfg = cfg.algorithm.train
        self.grad_accum_steps = self.train_cfg.grad_accu_step
        self.max_epochs = self.train_cfg.epochs

        self.train_dataloader = self.get_train_dataloader()
        self.steps_per_epoch = max(len(self.train_dataloader) // self.grad_accum_steps, 1)
        self.max_total_steps = self.max_epochs * self.steps_per_epoch
        self.total_step = 0

    def get_train_dataloader(self) -> DataLoader:
        """Build the training dataloader from the train dataset."""
        return build_dataloader(self.cfg, self.train_dataset, mode="train")

    @abstractmethod
    def train_mode(self):
        """
        Switch to the training mode.
        """
        ...

    @abstractmethod
    def zero_grad(self):
        """
        Clean model gradients.
        """
        ...

    @abstractmethod
    def forward(self, data):
        """
        Forward data to the model. Return a dict with at least "loss".
        """
        ...

    @abstractmethod
    def backward(self, loss):
        """
        Backward the loss.
        """
        ...

    @abstractmethod
    def no_sync(self):
        """
        Stop gradient synchronization.
        Return a context manager that disables gradient sync, e.g., `model.no_sync()`.
        """
        ...

    @abstractmethod
    def clip_grad(self):
        """
        Clip gradients.
        """
        ...

    @abstractmethod
    def optimizer_step(self):
        """
        Optimizer updates one step.
        """
        ...

    @abstractmethod
    def lr_scheduler_step(self):
        """
        LR scheduler updates one step.
        """
        ...

    @abstractmethod
    def ema_step(self):
        """
        EMA model updates one step.
        """
        ...

    @abstractmethod
    def load_checkpoint(self):
        """
        Resume model/optimizer states and total_step from a checkpoint.
        """
        ...

    @abstractmethod
    def save_checkpoint(self):
        """
        Save a checkpoint (the subclass decides whether this step needs one).
        """
        ...

    @abstractmethod
    def update_log(self):
        """
        Update logs.
        """
        ...

    def run(self) -> None:
        """
        Run the training loop.
        """
        assert self.train_dataloader is not None

        self.load_checkpoint()
        start_epoch = self.total_step // self.steps_per_epoch
        print(f"steps_per_epoch={self.steps_per_epoch}, max_total_steps={self.max_total_steps}")

        for epoch in range(start_epoch, self.max_epochs):
            print(f"\n{'=' * 80}\n🚀 Epoch {epoch + 1}/{self.max_epochs}\n{'=' * 80}")
            reached_max_steps = False
            self.train_mode()
            self.train_dataloader.sampler.set_epoch(epoch)
            data_iter = iter(self.train_dataloader)

            # In a resumed epoch, skip the batches already consumed.
            steps_done = self.total_step % self.steps_per_epoch if epoch == start_epoch else 0
            for _ in range(steps_done * self.grad_accum_steps):
                next(data_iter, None)

            for _ in range(steps_done, self.steps_per_epoch):
                # 1. Forward and backward with gradient accumulation
                self.zero_grad()
                for micro_step in range(self.grad_accum_steps):
                    data = next(data_iter)

                    # Only synchronize gradients on the last micro-step for efficiency
                    sync_context = (
                        nullcontext()
                        if micro_step == self.grad_accum_steps - 1
                        else self.no_sync()
                    )
                    with sync_context:
                        forward_results = self.forward(data)
                        self.backward(forward_results["loss"] / self.grad_accum_steps)

                # 2. Clip gradients
                self.clip_grad()

                # 3. Optimizer step
                self.optimizer_step()

                # 4. LR scheduler step
                self.lr_scheduler_step()

                # 5. Update EMA
                self.ema_step()
                self.total_step += 1

                # 6. Update logs
                self.update_log()

                # 7. Save checkpoints
                self.save_checkpoint()

                if self.total_step >= self.max_total_steps:
                    reached_max_steps = True
                    break

            if reached_max_steps:
                break
            print(f"Completed epoch {epoch + 1}/{self.max_epochs}")

        dist.barrier()
