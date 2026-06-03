"""Distributed: environment init and collective communication helpers."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

# Generous collective timeout: in-runtime latent preprocessing can take many minutes
# and desynchronizes ranks, so the first training collective must tolerate the wait.
_PG_TIMEOUT = timedelta(hours=2)


def init_dist_env() -> tuple[int, int, torch.device]:
    """
    Initialize the distributed process group.
    Returns (world_size, rank, device).
    """
    if all(var in os.environ for var in ["WORLD_SIZE", "RANK", "LOCAL_RANK"]):
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        device = torch.device(f"cuda:{local_rank}")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=device,
            timeout=_PG_TIMEOUT,
        )

        torch.cuda.set_device(local_rank)

        prefix = "🌐 Distributed environment initialization:  "
        print(f"{prefix}world_size {world_size}  rank {rank}  device {device}")
        dist.barrier()
    else:
        dist_safe_exit(1, message="❌ Initialize distributed environment failed")

    return world_size, rank, device


def dist_safe_exit(exit_code: int = 0, message: str = "") -> None:
    """Destroy the process group (if initialized) and exit the process."""
    message = f"\n{message}" if message else message
    print(f"🛑 Exiting with code {exit_code}{message}")
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
            print("✅ Destroyed process group")
        except Exception as e:
            print(f"⚠️ Destroy process group failed: {e}")
    os._exit(exit_code)


def all_reduce(
    x: int | float | list | torch.Tensor,
    op: str = "sum",
) -> int | float | list | torch.Tensor:
    """All-reduce a Python scalar, list, or tensor across distributed ranks.

    It performs an in-place ``torch.distributed.all_reduce`` on an internal 
    tensor copy and returns the reduced result without modifying the input.

    Supported operations are ``sum``, ``mean``, ``max``, and ``min``.
    ``mean`` is implemented as ``SUM / world_size`` for backend compatible.

    Notes:
        * All ranks must call it with the same input type, shape, and dtype.
        * ``x`` will be returned on their original data type and device.
        * int inputs with ``op="mean"`` return floating-point results.

    Args:
        x: Input. Support: ``int``, ``float``, ``list``, and ``torch.Tensor``.
        op: Reduction operation. Support: ``sum``, ``mean``, ``max``, and ``min``.

    Returns:
        The reduced value, converted back to the original data type and device.
    """
    if op not in {"sum", "mean", "max", "min"}:
        raise ValueError(f"❌ Invalid all_reduce op: {op}")

    if not isinstance(x, (int, float, list, torch.Tensor)):
        raise TypeError(f"❌ Unsupported input type: {type(x)}")

    reduce_op = {
        "sum": dist.ReduceOp.SUM,
        "mean": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }[op]

    world_size = dist.get_world_size()

    is_tensor = isinstance(x, torch.Tensor)
    ori_device = x.device if is_tensor else None
    ori_dtype = x.dtype if is_tensor else None

    if is_tensor:
        tensor = x.detach().clone().cuda()
    else:
        tensor = torch.as_tensor(x, device="cuda")

    if op == "mean" and not tensor.is_floating_point():
        tensor = tensor.float()

    dist.all_reduce(tensor, op=reduce_op)

    if op == "mean":
        tensor = tensor / world_size

    if is_tensor:
        if op == "mean" and not ori_dtype.is_floating_point:
            return tensor.to(device=ori_device)
        return tensor.to(device=ori_device, dtype=ori_dtype)

    if isinstance(x, int):
        return float(tensor.item()) if op == "mean" else int(tensor.item())

    if isinstance(x, float):
        return float(tensor.item())

    return tensor.tolist()
