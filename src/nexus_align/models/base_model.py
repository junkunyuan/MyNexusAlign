"""Base model: generic FSDP wrapping and EMA maintenance for models."""

from copy import deepcopy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    CPUOffload,
)

from nexus_align.engine.distributed import reduce_scalar
from nexus_align.utils.dtype import DTYPE_MAP

SHARDING_STRATEGY_MAP = {
    "no_shard": ShardingStrategy.NO_SHARD,
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD
}


class BaseModel(ABC):
    """
    Base model: builds a model, wraps it with FSDP, and maintains an EMA copy.

    Subclasses implement build_model() and wrap_modules(); __init__ then exposes
    self.model (FSDP-wrapped, trainable) and self.ema (FSDP-wrapped, frozen).
    Both share the same sharding, so EMA updates run directly on local shards.
    """

    def __init__(self, cfg_model) -> None:
        self.cfg_model = cfg_model
        self.device = torch.device("cuda", torch.cuda.current_device())

        # Build model and ema
        model = self.build_model()
        model_dtype = DTYPE_MAP[cfg_model.get("dtype", "fp32")]
        model = model.to(model_dtype)
        ema = deepcopy(model)
        ema.requires_grad_(False)

        self.model = self.fsdp_wrap(model, model_name="model")
        self.ema = self.fsdp_wrap(ema, model_name="ema")
        self.model.train()
        self.ema.eval()

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model module (before FSDP wrapping)."""
        ...

    @abstractmethod
    def wrap_modules(self) -> tuple:
        """Return the module classes wrapped as FSDP units."""
        ...

    def fsdp_wrap(self, module: nn.Module, model_name: str = "model") -> FSDP:
        """Wrap a module with torch FSDP (Fully Sharded Data Parallel)."""
        cfg_model_fsdp = self.cfg_model.fsdp
        fsdp_kwargs = {}

        # FSDP strategy
        strategy = cfg_model_fsdp.get("strategy", "no_shard")
        fsdp_kwargs["sharding_strategy"] = SHARDING_STRATEGY_MAP[strategy]

        # CPU offload
        cpu_offload = cfg_model_fsdp.get("cpu_offload", False)
        fsdp_kwargs["cpu_offload"] = CPUOffload(True) if cpu_offload else None

        # Device ID
        fsdp_kwargs["device_id"] = torch.cuda.current_device()

        # Wrap policy
        wrap_modules = self.wrap_modules()
        def auto_wrap_policy(module, recurse, nonwrapped_numel):
            if recurse:
                return True
            if wrap_modules is None:
                return True
            return any(isinstance(module, m) for m in wrap_modules)
        fsdp_kwargs["auto_wrap_policy"] = auto_wrap_policy

        # Mixed precision of model, buffer, and reduce
        param_dtype = DTYPE_MAP[cfg_model_fsdp.get("param_dtype", "fp32")]
        buffer_dtype = DTYPE_MAP[cfg_model_fsdp.get("buffer_dtype", param_dtype)]
        reduce_dtype = DTYPE_MAP[cfg_model_fsdp.get("reduce_dtype", "bf16")]
        fsdp_kwargs["mixed_precision"] = MixedPrecision(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype
        )

        module = FSDP(
            module,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            sharding_strategy=SHARDING_STRATEGY_MAP[strategy],
            cpu_offload=CPUOffload(True) if cpu_offload else None,
            device_id=self.device,
            sync_module_states=True,  # ranks seed differently; broadcast rank-0 init
            use_orig_params=True,  # flexible, e.g., with torch.compile
        )

        total_params = reduce_scalar(sum(p.numel() for p in module.parameters()), "sum") / 1e9
        info = [f"📦 Wrapped {model_name} with FSDP:"]
        info += [f"    Total params: {total_params:.4f} B"]
        info += [f"    Strategy: {strategy}"]
        info += [f"    Wrapped modules: {', '.join(m.__name__ for m in wrap_modules)}"]
        info += [f"    Param/Reduce/Buffer dtype: {param_dtype} / {reduce_dtype} / {buffer_dtype}"]
        info += [f"    CPU offload: {cpu_offload}"]
        print("\n".join(info))

        return module

    @torch.no_grad()
    def ema_step(self, decay: float) -> None:
        """Step EMA params toward model params (both share the same sharding)."""
        for ema_p, p in zip(self.ema.parameters(), self.model.parameters()):
            ema_p.mul_(decay).add_(p.data, alpha=1 - decay)

    def state_dict(self) -> dict:
        """Full model/EMA state dicts, gathered to CPU on rank 0 (call on all ranks)."""
        full_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_config):
            model_sd = self.model.state_dict()
        with FSDP.state_dict_type(self.ema, StateDictType.FULL_STATE_DICT, full_config):
            ema_sd = self.ema.state_dict()
        return {"model": model_sd, "ema": ema_sd}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load full model/EMA state dicts (call on all ranks)."""
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            self.model.load_state_dict(state_dict["model"])
        with FSDP.state_dict_type(self.ema, StateDictType.FULL_STATE_DICT):
            self.ema.load_state_dict(state_dict["ema"])

    def optim_state_dict(self, optimizer) -> dict:
        """Full optimizer state dict, gathered on rank 0 (call on all ranks)."""
        return FSDP.optim_state_dict(self.model, optimizer)

    def load_optim_state_dict(self, optimizer, optim_state_dict: dict) -> None:
        """Load a full optimizer state dict into the sharded optimizer (call on all ranks)."""
        optim_state_dict = FSDP.optim_state_dict_to_load(self.model, optimizer, optim_state_dict)
        optimizer.load_state_dict(optim_state_dict)
