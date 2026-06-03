"""Base trainer: abstract interface for trainers."""

from abc import ABC, abstractmethod
from contextlib import nullcontext

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from nexus_align.models.base_model import BaseModel
from nexus_align.algorithms.base_algorithm import BaseAlgorithm


class BaseTrainer(ABC):
    """
    Base trainer.
    """

    def __init__(
        self,
        train_dataloader: DataLoader = None,
        valid_dataloader: DataLoader = None,
        eval_dataloader: DataLoader = None,
        model: BaseModel = None,
        algorithm: BaseAlgorithm = None,
        optimizer: Optimizer = None,
        lr_scheduler: LRScheduler = None
    ) -> None:
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.algorithm = algorithm
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.tracker = None  # TODO
        self.checkpoint_manager = None  # TODO

    @abstractmethod
    def train_mode(self):
        """
        Switch to the training mode.
        """
        ...
    
    @abstractmethod
    def eval_mode(self):
        """
        Switch to the evaluation mode.
        """
        ...
    
    @abstractmethod
    def zero_grad(self):
        """
        Clean model gradients.
        """
        ...

    @abstractmethod
    def validate_model(self):
        """
        Validate the model.

        Call self.eval_mode() before validating the model.
        """
        ...
    
    @abstractmethod
    def evaluate_model(self):
        """
        Evaluate the model.

        Call self.eval_mode() before evaluating the model.
        """
        ...
    
    @abstractmethod
    def forward(self, data):
        """
        Forward data to the model.
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
    def update_log(self):
        """
        Update logs.
        """
        ...

    def run(self) -> None:
        """
        Run the loop.
        """
        assert self.train_dataloader is not None or self.eval_dataloader is not None

        # Start evaluating
        if self.train_dataloader is None and self.eval_dataloader is not None:
            self.evaluate_model()
            return

        # Start training
        if self.train_dataloader is not None:
            for epoch in range(self.tracker.max_epochs):
                # Skip the resumed epoch
                if epoch < self.tracker.epoch:
                    continue
                
                # Start a new epoch
                print(f"\n{'=' * 80}\n🚀 Epoch {epoch}/{self.tracker.max_epochs}\n{'=' * 80}")
                self.train_dataloader.sampler.set_epoch(epoch)
                self.tracker.start("epoch")
                num_steps_per_epoch = len(self.train_dataloader) // self.tracker.grad_accum_steps
                train_data_iterator = iter(self.train_dataloader)
                for step in range(num_steps_per_epoch):
                    # Skip the resumed step
                    if step < self.tracker.step:
                        for _ in range(self.tracker.grad_accum_steps):
                            next(train_data_iterator)
                        continue
                    # Exit if reaching the total steps
                    if self.tracker.total_step >= self.tracker.max_total_step:
                        return
                    
                    # Start a new step
                    epoch_f = f"Epoch {self.tracker.epoch}/{self.tracker.max_epochs}"
                    step_f = f"Step {self.tracker.step}/{num_steps_per_epoch}"
                    total_step_f = f"Total step {self.tracker.total_step}/{self.tracker.max_total_step}"
                    print(f"\n{'=' * 80}\n🚀  {epoch_f}  {step_f}  {total_step_f}\n{'=' * 80}")
                    
                    # Start validating
                    is_in_val_step = self.tracker.total_step in self.tracker.validate_steps
                    is_in_val_interval = self.tracker.total_step % self.tracker.validate_interval == 0
                    is_init_val_needed = self.tracker.init_validate
                    if is_in_val_step or is_in_val_interval or is_init_val_needed:
                        self.validate_model()
                        self.tracker.init_validate = False

                    # Start evaluating
                    is_in_eval_step = self.tracker.total_step in self.tracker.evaluate_steps
                    is_in_eval_interval = self.tracker.total_step % self.tracker.evaluate_interval == 0
                    is_init_eval_needed = self.tracker.init_evaluate
                    if is_in_eval_step or is_in_eval_interval or is_init_eval_needed:
                        self.evaluate_model()
                        self.tracker.init_evaluate = False

                    self.tracker.start("step")
                    self.train_mode()
                    self.zero_grad()
                    for micro_step in range(self.tracker.grad_accum_steps):
                        # 1. Get data
                        data = next(train_data_iterator)

                        # Only synchronize gradients on the last micro-step for efficiency
                        sync_context = (
                            nullcontext()
                            if micro_step == self.tracker.grad_accum_steps - 1
                            else self.no_sync()
                        )
                        with sync_context:
                            # 2. Model forward
                            forward_results = self.forward(data)

                            # 3. Model backward
                            (forward_results["loss"] / self.tracker.grad_accum_steps).backward()

                    # 4. Clip gradients
                    self.clip_grad()

                    # 5. Optimizer step
                    self.optimizer_step()

                    # 6. LR scheduler step
                    self.lr_scheduler_step()

                    # 7. Update EMA
                    self.ema_step()

                    # 8. Save checkpoints
                    # NOTE: checkpoint_manager will decide whether to save based on train_state
                    self.checkpoint_manager.save(
                        model={
                            "model": self.model,
                            "ema": self.ema,
                            "optimizer": self.optimizer,
                            "lr_scheduler": self.lr_scheduler
                        },
                        train_state={
                            "epoch": self.tracker.epoch,
                            "step": self.tracker.step,
                            "total_step": self.tracker.total_step,
                        }
                    )

                    # 9. Update logs
                    self.update_log()

                    self.tracker.end("step")
                
                self.tracker.end("epoch")
                self.update_log()
