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

SHARDING_STRATEGY_MAP = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
}

PARAM_DTYPE_MAP = {
    "no": None,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
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

        model = self.build_model()
        model_dtype = PARAM_DTYPE_MAP[cfg_model.get("dtype", "fp32")]
        if model_dtype is not None:
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
        fsdp_cfg = self.cfg_model.fsdp
        strategy = fsdp_cfg.get("strategy", "full_shard")
        if strategy not in SHARDING_STRATEGY_MAP:
            raise ValueError(f"❌ Invalid FSDP sharding strategy: {strategy}")
        cpu_offload = fsdp_cfg.get("cpu_offload", False)

        wrap_modules = self.wrap_modules()

        def auto_wrap_policy(module, recurse, nonwrapped_numel):
            if recurse:
                return True
            return any(isinstance(module, m) for m in wrap_modules)

        # param_dtype is the compute dtype; reduce defaults to fp32, buffer follows param.
        param_dtype = PARAM_DTYPE_MAP[fsdp_cfg.get("param_dtype", "bf16")]
        reduce_dtype = buffer_dtype = None
        if param_dtype is None:
            mixed_precision = None
        else:
            reduce_dtype = PARAM_DTYPE_MAP[fsdp_cfg.get("reduce_dtype", "fp32")]
            buffer_key = fsdp_cfg.get("buffer_dtype", None)
            buffer_dtype = PARAM_DTYPE_MAP[buffer_key] if buffer_key is not None else param_dtype
            mixed_precision = MixedPrecision(
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
            use_orig_params=True,  # allow mixed requires_grad (e.g. frozen pos_embed)
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
