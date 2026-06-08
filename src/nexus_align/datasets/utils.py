"""Dataset utils: generic helpers shared across datasets."""

import random
import hashlib
from typing import TypeVar

T = TypeVar("T")


def sample_items(items: list[T], sample_ratio: float | int | None = None) -> list[T]:
    """
    Randomly subsample items.

    If sample_ratio<=0, keeps all.
    If sample_ratio<=1, sample the specific fraction.
    Else, sample the specific count.
    """
    n = len(items)
    if not isinstance(sample_ratio, (int, float)) or sample_ratio <= 0 or n == 0:
        return items
    k = min(int(n * sample_ratio if sample_ratio < 1.1 else sample_ratio), n)
    return random.sample(items, k)


def compute_md5(
    image_bytes: bytes | None = None,
    text: str | None = None,
    label: int | None = None,
) -> str:
    """Hash available content into a stable hex md5 identifier."""
    h = hashlib.md5()
    if image_bytes is not None:
        h.update(image_bytes)
    if text is not None:
        h.update(text.encode("utf-8"))
    if label is not None:
        h.update(str(label).encode("utf-8"))
    return h.hexdigest()
