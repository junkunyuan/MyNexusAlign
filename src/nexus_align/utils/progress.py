"""Progress display: tqdm bar wrapper for distributed runs."""

import torch.distributed as dist
from tqdm import tqdm


class TqdmBar:
    """Progress bar shown only on the chosen rank (rank="all" shows on every rank)."""

    def __init__(
        self,
        total: int,
        desc: str,
        unit: str,
        colour: str = "blue",
        rank: int | str = 0,
    ) -> None:
        fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        show = rank == "all" or not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == rank
        self.pbar = tqdm(total=total, desc=desc, unit=unit, bar_format=fmt, colour=colour) if show else None

    def update(self, step: int) -> None:
        if self.pbar is not None:
            self.pbar.update(step)

    def close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
