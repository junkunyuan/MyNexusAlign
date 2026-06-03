"""ImageNet-1K dataset: parquet reader and cached VAE-latent loader."""

import bisect
import glob
import importlib.util
import io
import json
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors import safe_open

from nexus_align.datasets.base_dataset import BaseTextImageDataset
from nexus_align.utils.progress import TqdmBar

NUM_CLASSES = 1000


def load_class_text(root: str) -> list[str] | None:
    """Load index->class-text list from the dataset's classes.py, or None."""
    path = os.path.join(root, "classes.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("imagenet_classes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.IMAGENET2012_CLASSES.values())


def split_files(root: str, split: str) -> list[str]:
    """Sorted parquet shard paths for a split (train/validation/test)."""
    return sorted(glob.glob(os.path.join(root, "data", f"{split}-*.parquet")))


def shard_num_samples(path: str) -> int:
    """Number of samples stored in a latent shard file."""
    with safe_open(path, framework="pt") as f:
        return f.get_slice("label").get_shape()[0]


def _wait_for(condition: Callable[[], bool], timeout: float = 7200.0, interval: float = 2.0) -> None:
    """Poll until condition() is true, raising after timeout seconds."""
    start = time.monotonic()
    while not condition():
        if time.monotonic() - start > timeout:
            raise RuntimeError("❌ Timed out waiting for latent cache build coordination")
        time.sleep(interval)


class ImageNet1K(BaseTextImageDataset):
    """ImageNet-1K dataset."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 256,
        flip_prob: float = 0.5,
        cache_dir: str | None = None,
        image_transform: Callable | None = None,
        text_transform: Callable | None = None,
        sample_ratio: float | int | None = None,
        dedup: bool = False,
        vae: str = "stabilityai/sd-vae-ft-ema",
        read_batch: int = 256,
        vae_batch: int = 64,
        shard_size: int = 8192,
        preprocess_workers: int = 8,
    ) -> None:
        super().__init__(image_transform, text_transform, sample_ratio, dedup)
        self.root = root
        self.split = split
        self.img_size = img_size
        self.flip_prob = flip_prob
        self.cache_dir = cache_dir
        self.vae = vae
        self.read_batch = read_batch
        self.vae_batch = vae_batch
        self.shard_size = shard_size
        self.preprocess_workers = preprocess_workers

        if cache_dir is not None:
            self._init_latent_mode()
        else:
            self._init_raw_mode()

    # --------------------------------------------------------------------------------
    # Raw Model: Load and process data from parquets
    # --------------------------------------------------------------------------------
    def _init_raw_mode(self) -> None:
        self.mode = "raw"
        self.files = split_files(self.root, self.split)
        if not self.files:
            raise FileNotFoundError(f"❌ No {self.split} parquet under {self.root}/data")
        self.class_text = load_class_text(self.root)

        # Global index -> (file, row_group) via row-group boundaries (metadata only).
        self._rg_starts: list[int] = []
        self._rg_loc: list[tuple[int, int]] = []
        start = 0
        for fi, f in enumerate(self.files):
            meta = pq.ParquetFile(f).metadata
            for rg in range(meta.num_row_groups):
                self._rg_starts.append(start)
                self._rg_loc.append((fi, rg))
                start += meta.row_group(rg).num_rows

        # Per-worker lazy caches (kept empty here so they survive DataLoader fork).
        self._pf_cache: dict[int, pq.ParquetFile] = {}
        self._rg_key: tuple[int, int] | None = None
        self._rg_table: pa.Table | None = None
        self.build_indices(start)

    def _row_group(self, file_idx: int, rg_idx: int) -> pa.Table:
        """Read a parquet row group, caching the most recent one per worker."""
        key = (file_idx, rg_idx)
        if self._rg_key != key:
            pf = self._pf_cache.get(file_idx)
            if pf is None:
                pf = pq.ParquetFile(self.files[file_idx])
                self._pf_cache[file_idx] = pf
            self._rg_table = pf.read_row_group(rg_idx, columns=["image", "label"])
            self._rg_key = key
        return self._rg_table

    def get_raw(self, index: int) -> dict[str, Any]:
        """Load raw data in raw mode."""
        if self.mode != "raw":
            raise RuntimeError("❌ get_raw is only available in raw mode (cache_dir unset)")
        rg = bisect.bisect_right(self._rg_starts, index) - 1
        file_idx, rg_idx = self._rg_loc[rg]
        local = index - self._rg_starts[rg]

        table = self._row_group(file_idx, rg_idx)
        img_struct = table.column("image")[local].as_py()
        label = table.column("label")[local].as_py()
        image_bytes = img_struct["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = self.class_text[label] if self.class_text else None
        return {"image": image, "image_bytes": image_bytes, "text": text, "label": label}

    # --------------------------------------------------------------------------------
    # Latent Model: Load data from safetensors
    # --------------------------------------------------------------------------------
    def _init_latent_mode(self) -> None:
        self.mode = "latent"
        # (Re)build on the fly whenever this node's cache is missing or incomplete. The
        # build uses only node-local (filesystem) coordination -- no cross-node
        # collectives -- so nodes in different cache states never block each other.
        if self._load_manifest() is None:
            self._build_cache()
        manifest = self._load_manifest()
        if manifest is None:
            raise RuntimeError(f"❌ Latent cache build incomplete at {self.cache_dir}")

        self.shards: list[tuple[str, int]] = []
        start = 0
        for s in manifest["shards"]:
            path = os.path.join(self.cache_dir, s["file"])
            self.shards.append((path, start))
            start += s["num_samples"]
        self._shard_starts = [shard_start for _, shard_start in self.shards]
        self._handles: dict[str, Any] = {}  # per-worker lazy safe_open handles
        self.build_indices(start)

    def _load_manifest(self) -> dict[str, Any] | None:
        """Return the manifest if a complete cache exists for this split, else None."""
        manifest_path = os.path.join(self.cache_dir, f"manifest-{self.split}.json")
        if not os.path.exists(manifest_path):
            return None
        with open(manifest_path) as f:
            manifest = json.load(f)
        all_present = all(os.path.exists(os.path.join(self.cache_dir, s["file"]))
                          for s in manifest["shards"])
        return manifest if all_present else None

    def _build_cache(self) -> None:
        """Encode the parquet data to fp16 VAE latents on this node's local disk.

        Files are split across node-local ranks (LOCAL_RANK / LOCAL_WORLD_SIZE) so each
        node builds a complete cache; decoding runs in parallel workers overlapped with
        GPU encoding. Coordination is purely node-local via marker files (no NCCL
        collectives), so nodes never block one another. Local rank 0 wipes any stale
        cache, then collects all ranks' shards and writes the manifest.
        """
        from diffusers.models import AutoencoderKL
        from safetensors.torch import save_file

        rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ.get("LOCAL_WORLD_SIZE") or torch.cuda.device_count() or 1)
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", "0")
        device = (torch.device(f"cuda:{torch.cuda.current_device()}")
                  if torch.cuda.is_available() else torch.device("cpu"))
        cache = self.cache_dir
        os.makedirs(cache, exist_ok=True)

        manifest_path = os.path.join(cache, f"manifest-{self.split}.json")
        clean_flag = os.path.join(cache, f".clean-{self.split}-{run_id}")
        done_flag = os.path.join(cache, f".done-{self.split}-{run_id}-{rank:03d}")
        done_glob = os.path.join(cache, f".done-{self.split}-{run_id}-*")

        # Local rank 0 wipes any stale shards/manifest/markers, then signals; the others
        # wait so their writes are never deleted.
        if rank == 0:
            print(f"⚙️  Building latent cache at {cache} ...")
            for pat in (f"latents-{self.split}-*.safetensors", f".done-{self.split}-*", f".clean-{self.split}-*"):
                for p in glob.glob(os.path.join(cache, pat)):
                    os.remove(p)
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
            open(clean_flag, "w").close()
        else:
            _wait_for(lambda: os.path.exists(clean_flag))

        vae = AutoencoderKL.from_pretrained(self.vae).to(device).eval()
        files = split_files(self.root, self.split)[rank::world]
        stream = _ParquetImageStream(files, self.img_size, load_class_text(self.root), self.read_batch)
        loader = torch.utils.data.DataLoader(
            stream, batch_size=self.vae_batch, num_workers=self.preprocess_workers, pin_memory=True,
        )

        # One progress bar per node (driven by local rank 0, tracking its own samples).
        bar = None
        if rank == 0:
            total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
            bar = TqdmBar(total, "Encoding latents", "img", rank="all")

        out_m, out_f, out_l, out_u = [], [], [], []
        state = {"count": 0, "shard": 0}

        def write_shard() -> None:
            if not out_m:
                return
            tensors = {
                "moments": torch.cat(out_m).contiguous(),
                "moments_flip": torch.cat(out_f).contiguous(),
                "label": torch.cat(out_l).contiguous(),
                "uid": torch.cat(out_u).contiguous(),
            }
            path = os.path.join(cache, f"latents-{self.split}-{rank:03d}-{state['shard']:05d}.safetensors")
            save_file(tensors, path, metadata={"img_size": str(self.img_size), "vae": self.vae})
            out_m.clear(); out_f.clear(); out_l.clear(); out_u.clear()
            state["count"] = 0
            state["shard"] += 1

        for imgs, labels, uids in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.no_grad():
                out_m.append(vae.encode(imgs).latent_dist.parameters.cpu().half())
                out_f.append(vae.encode(imgs.flip(dims=[3])).latent_dist.parameters.cpu().half())
            out_l.append(labels.to(torch.int16))
            out_u.append(uids.to(torch.uint8))
            state["count"] += imgs.shape[0]
            if bar is not None:
                bar.update(imgs.shape[0])
            if state["count"] >= self.shard_size:
                write_shard()
        write_shard()
        if bar is not None:
            bar.close()
        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Mark this rank done; local rank 0 waits for all ranks, then writes the manifest.
        open(done_flag, "w").close()
        if rank == 0:
            _wait_for(lambda: len(glob.glob(done_glob)) >= world)
            _write_manifest(cache, self.split, self.img_size, self.vae)
            for p in glob.glob(done_glob):
                os.remove(p)
            os.remove(clean_flag)
        else:
            _wait_for(lambda: os.path.exists(manifest_path))

    def _get_latent(self, index: int) -> tuple[torch.Tensor, int]:
        index = self._indices[index]  # map filtered index back to the raw sample
        si = bisect.bisect_right(self._shard_starts, index) - 1
        path, start = self.shards[si]
        local = index - start

        f = self._handles.get(path)
        if f is None:
            f = safe_open(path, framework="pt")
            self._handles[path] = f

        label = int(f.get_slice("label")[local].item())
        use_flip = torch.rand(1).item() < self.flip_prob
        name = "moments_flip" if use_flip else "moments"
        moments = f.get_slice(name)[local]
        return moments.float(), label

    # --------------------------------------------------------------------------------
    # Common Model: Load data from parquets or safetensors
    # --------------------------------------------------------------------------------
    def __getitem__(self, index: int) -> dict[str, Any] | tuple[torch.Tensor, int]:
        if self.mode == "latent":
            return self._get_latent(index)
        return super().__getitem__(index)


# --------------------------------------------------------------------------------
# Preprocessing helpers: image transform, manifest writer, parallel decode stream
# --------------------------------------------------------------------------------

def _center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """Center-crop a PIL image to a square of image_size (DiT/MeanFlow style)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size])


def _build_transform(img_size: int) -> Callable:
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Lambda(lambda img: _center_crop_arr(img, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _write_manifest(cache_dir: str, split: str, img_size: int, vae_name: str) -> None:
    """Scan written shards and atomically write manifest-{split}.json."""
    shard_files = sorted(glob.glob(os.path.join(cache_dir, f"latents-{split}-*.safetensors")))
    shards: list[dict[str, Any]] = []
    total = 0
    for path in shard_files:
        n = shard_num_samples(path)
        shards.append({"file": os.path.basename(path), "num_samples": n})
        total += n
    manifest = {
        "dataset": "imagenet-1k", "split": split, "img_size": img_size,
        "vae": vae_name, "num_classes": NUM_CLASSES, "num_samples": total,
        "latent_shape": [8, img_size // 8, img_size // 8], "shards": shards,
    }
    path = os.path.join(cache_dir, f"manifest-{split}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, path)
    print(f"✅ Wrote manifest: {total} samples across {len(shards)} shards -> {path}")


class _ParquetImageStream(torch.utils.data.IterableDataset):
    """Stream (image_tensor, label, uid) from parquet; files are split across workers.

    Each worker reads whole files sequentially (good parquet locality) and does the
    JPEG decode + transform, so decoding runs in parallel and overlaps GPU encoding.
    """

    def __init__(self, files: list[str], img_size: int, class_text: list[str] | None,
                 read_batch: int) -> None:
        self.files = files
        self.img_size = img_size
        self.class_text = class_text
        self.read_batch = read_batch

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        files = self.files if info is None else self.files[info.id::info.num_workers]
        transform = _build_transform(self.img_size)
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=self.read_batch, columns=["image", "label"]):
                for img_struct, label in zip(batch.column("image").to_pylist(),
                                             batch.column("label").to_pylist()):
                    image_bytes = img_struct["bytes"]
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    text = self.class_text[label] if self.class_text else None
                    uid = BaseTextImageDataset.compute_md5(image_bytes=image_bytes, text=text, label=label)
                    uid_arr = np.frombuffer(bytes.fromhex(uid), dtype=np.uint8).copy()
                    yield transform(image), label, uid_arr


