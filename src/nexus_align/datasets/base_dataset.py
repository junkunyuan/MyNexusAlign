"""Base dataset: generic dataset."""

from typing import Any
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

import torch
from torch.utils.data import Dataset

from .utils import sample_items, compute_md5


class BaseTextImageDataset(Dataset, ABC):
    """
    Abstract base for image + text + label datasets.

    Subclasses implement get_raw to return one raw sample.
    
    Optionally deduplicate (by md5 uid over image bytes + text + label).
    Apply random sampling by sample_ratio when calling build_indices.
    Apply image/text transforms in __getitem__.
    Provide collate_fn to batch samples.
    """

    def __init__(
        self,
        image_transform: Callable | None = None,
        text_transform: Callable | None = None,
        sample_ratio: float | int | None = None,
        deduplicate: bool = False,
    ) -> None:
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.sample_ratio = sample_ratio
        self.deduplicate = deduplicate
        self._indices = None  # built by build_indices; maps dataset indices to raw sample indices

    @abstractmethod
    def get_raw(self, index: int) -> dict[str, Any]:
        """Return a raw sample dict with "image", "text", "label"."""
        ...

    def build_indices(self, num_items: int, uids: Iterable[Any] | None = None) -> None:
        """Build the active index: drop duplicate uids then sample.

        uids: a precomputed uid per item; if None, derive from get_raw (raw mode).
        """
        if self.deduplicate:
            if uids is None:
                uids = (self.get_uid(self.get_raw(i)) for i in range(num_items))
            seen, indices = set(), []
            for i, uid in enumerate(uids):
                if uid not in seen:
                    seen.add(uid)
                    indices.append(i)
        else:
            indices = list(range(num_items))
        self._indices = sample_items(indices, self.sample_ratio)

    def __len__(self) -> int:
        return len(self._indices)

    def get_uid(self, raw: dict[str, Any]) -> str:
        """Calculate uid of a raw sample dict (image bytes + text + label)."""
        image, image_bytes = raw.get("image"), raw.get("image_bytes")
        if image_bytes is None and image is not None:
            image_bytes = image.tobytes()
        uid = compute_md5(
            image_bytes=image_bytes,
            text=raw.get("text"),
            label=raw.get("label")
        )
        return uid

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
