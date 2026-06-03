"""Base dataset: generic image-text dataset."""

import hashlib
import random
from typing import Any, TypeVar
from abc import ABC, abstractmethod
from collections.abc import Callable
from string import ascii_letters, digits, punctuation, whitespace

import torch
from torch.utils.data import Dataset

T = TypeVar("T")

_ENGLISH_CHARS = set(ascii_letters + digits + punctuation + whitespace)


def is_pure_english(text: str) -> bool:
    """Return True if every character is ASCII English/punctuation/whitespace."""
    return all(c in _ENGLISH_CHARS for c in text)


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


class BaseTextImageDataset(Dataset, ABC):
    """
    Base image+text+label dataset.
    """

    def __init__(
        self,
        image_transform: Callable | None = None,
        text_transform: Callable | None = None,
        sample_ratio: float | int | None = None,
        dedup: bool = False,
    ) -> None:
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.sample_ratio = sample_ratio
        self.dedup = dedup
        self._indices = None

    @abstractmethod
    def get_raw(self, index: int) -> dict[str, Any]:
        """Return a raw sample dict."""
        ...

    def build_indices(self, num_items: int) -> None:
        """Build the active index from num_items, dropping duplicate uids then sampling."""
        if self.dedup:
            seen = set()
            indices = []
            for i in range(num_items):
                uid = self.get_uid(self.get_raw(i))
                if uid not in seen:
                    seen.add(uid)
                    indices.append(i)
        else:
            indices = list(range(num_items))
        self._indices = sample_items(indices, self.sample_ratio)

    def __len__(self) -> int:
        return len(self._indices)

    @staticmethod
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

    def get_uid(self, raw: dict[str, Any]) -> str:
        """Content uid of a raw sample dict (image bytes + text + label)."""
        image, image_bytes = raw.get("image"), raw.get("image_bytes")
        if image_bytes is None and image is not None:
            image_bytes = image.tobytes()
        return self.compute_md5(image_bytes=image_bytes, text=raw.get("text"), label=raw.get("label"))

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Get image, text, label, and uid
        raw = self.get_raw(self._indices[index])
        image, text, label = raw.get("image"), raw.get("text"), raw.get("label")
        uid = self.get_uid(raw)

        # Data transforms
        if image is not None and self.image_transform is not None:
            image = self.image_transform(image)
        if text is not None and self.text_transform is not None:
            text = self.text_transform(text)

        return {"uid": uid, "image": image, "text": text, "label": label}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate sample dicts: stack tensor images/labels, list the rest."""
        def stack(key: str) -> Any:
            vals = [b[key] for b in batch]
            if vals and torch.is_tensor(vals[0]):
                return torch.stack(vals)
            if key == "label" and all(v is not None for v in vals):
                return torch.tensor(vals)
            return vals

        return {
            "uid": [b["uid"] for b in batch],
            "image": stack("image"),
            "text": [b["text"] for b in batch],
            "label": stack("label"),
        }
