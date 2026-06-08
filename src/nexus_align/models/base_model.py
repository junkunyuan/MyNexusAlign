"""Base model: generic FSDP2 sharding and EMA maintenance for models."""

from copy import deepcopy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    OffloadPolicy,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from nexus_align.utils.dtype import DTYPE_MAP

# FSDP2 maps the sharding strategy to reshard_after_forward:
# full_shard reshards params after forward; shard_grad_op keeps them for backward.
RESHARD_AFTER_FORWARD = {"full_shard": True, "shard_grad_op": False}


class BaseModel(ABC):
    """
    Base model: builds a model, shards it with FSDP2, and maintains an EMA copy.

    Subclasses implement build_model() and wrap_modules(); __init__ then exposes
    self.model (sharded, trainable) and self.ema (sharded, frozen). Both share the
    same mesh and sharding, so EMA updates run directly on local DTensor shards.
    """

    def __init__(self, cfg_model) -> None:
        self.cfg_model = cfg_model
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.mesh = init_device_mesh("cuda", (dist.get_world_size(),))

        model = self.build_model()
        model_dtype = DTYPE_MAP[cfg_model.get("dtype", "fp32")]
        if model_dtype is not None:
            model = model.to(model_dtype)
        model = model.to(self.device)
        # Ranks seed differently; broadcast rank-0 weights so all ranks start identical.
        for tensor in list(model.parameters()) + list(model.buffers()):
            dist.broadcast(tensor.data, src=0)

        ema = deepcopy(model)
        ema.requires_grad_(False)

        self.model = self.fsdp_wrap(model, model_name="model")
        self.ema = self.fsdp_wrap(ema, model_name="ema")
        self.model.train()
        self.ema.eval()

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model module (before FSDP sharding)."""
        ...

    @abstractmethod
    def wrap_modules(self) -> tuple:
        """Return the module classes sharded as individual FSDP units."""
        ...

    def fsdp_wrap(self, module: nn.Module, model_name: str = "model") -> nn.Module:
        """Shard a module in place with FSDP2 (fully_shard): each wrapped unit, then the root."""
        fsdp_cfg = self.cfg_model.fsdp
        strategy = fsdp_cfg.get("strategy", "full_shard")
        if strategy not in RESHARD_AFTER_FORWARD:
            raise ValueError(f"❌ Invalid FSDP strategy: {strategy}")

        # param_dtype is the compute dtype; reduce defaults to fp32 (None disables mixed precision).
        param_dtype = DTYPE_MAP[fsdp_cfg.get("param_dtype", "bf16")]
        reduce_dtype = DTYPE_MAP[fsdp_cfg.get("reduce_dtype", "fp32")] if param_dtype else None
        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)

        cpu_offload = fsdp_cfg.get("cpu_offload", False)
        fsdp_kwargs = dict(
            mesh=self.mesh,
            reshard_after_forward=RESHARD_AFTER_FORWARD[strategy],
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy() if cpu_offload else OffloadPolicy(),
        )

        wrap_modules = self.wrap_modules()
        for submodule in module.modules():
            if isinstance(submodule, wrap_modules):
                fully_shard(submodule, **fsdp_kwargs)
        fully_shard(module, **fsdp_kwargs)

        total_params = sum(p.numel() for p in module.parameters()) / 1e9  # DTensor.numel is global
        info = [f"📦 Sharded {model_name} with FSDP2:"]
        info += [f"    Total params: {total_params:.4f} B"]
        info += [f"    Strategy: {strategy}"]
        info += [f"    Wrapped modules: {', '.join(m.__name__ for m in wrap_modules)}"]
        info += [f"    Param/Reduce dtype: {param_dtype} / {reduce_dtype}"]
        info += [f"    CPU offload: {cpu_offload}"]
        print("\n".join(info))

        return module

    @torch.no_grad()
    def ema_step(self, decay: float) -> None:
        """Step EMA params toward model params (both share the same sharding)."""
        for ema_p, p in zip(self.ema.parameters(), self.model.parameters()):
            ema_p.mul_(decay).add_(p.data, alpha=1 - decay)

    def state_dict(self) -> dict:
        """Full model/EMA state dicts, gathered to CPU (call on all ranks)."""
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return {
            "model": get_model_state_dict(self.model, options=options),
            "ema": get_model_state_dict(self.ema, options=options),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load full model/EMA state dicts, broadcasting from rank 0 (call on all ranks)."""
        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_model_state_dict(self.model, state_dict["model"], options=options)
        set_model_state_dict(self.ema, state_dict["ema"], options=options)

    def optim_state_dict(self, optimizer) -> dict:
        """Full optimizer state dict, gathered to CPU (call on all ranks)."""
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        return get_optimizer_state_dict(self.model, optimizer, options=options)

    def load_optim_state_dict(self, optimizer, optim_state_dict: dict) -> None:
        """Load a full optimizer state dict, broadcasting from rank 0 (call on all ranks)."""
        options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
        set_optimizer_state_dict(self.model, optimizer, optim_state_dict, options=options)
