"""ImageNet-1K dataset: parquet reader and cached VAE-latent loader."""

import argparse
import bisect
import glob
import importlib.util
import io
import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from safetensors import safe_open

from nexus_align.datasets.base_dataset import BaseTextImageDataset

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
        # Empty cache -> build it on the fly (collective across ranks); a partial
        # cache is left for _verify_cache to reject with a detailed error.
        if self._cache_is_empty():
            self._run_preprocess()
        manifest = self._verify_cache()

        self.shards: list[tuple[str, int]] = []
        start = 0
        for s in manifest["shards"]:
            path = os.path.join(self.cache_dir, s["file"])
            self.shards.append((path, start))
            start += s["num_samples"]
        self._shard_starts = [shard_start for _, shard_start in self.shards]
        self._handles: dict[str, Any] = {}  # per-worker lazy safe_open handles
        self.build_indices(start)

    def _cache_is_empty(self) -> bool:
        """True if no manifest and no shard files exist for this split."""
        manifest_path = os.path.join(self.cache_dir, f"manifest-{self.split}.json")
        shard_files = glob.glob(os.path.join(self.cache_dir, f"latents-{self.split}-*.safetensors"))
        return not os.path.exists(manifest_path) and not shard_files

    def _run_preprocess(self) -> None:
        """Build the latent cache in-place, reusing the current distributed context."""
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            rank, world = dist.get_rank(), dist.get_world_size()
        else:
            rank, world = 0, 1
        device = (torch.device(f"cuda:{torch.cuda.current_device()}")
                  if torch.cuda.is_available() else torch.device("cpu"))
        if rank == 0:
            print(f"⚙️  Empty cache, building latents at {self.cache_dir} ...")
        build_latent_cache(self.root, self.split, self.img_size, self.cache_dir,
                           self.vae, self.read_batch, self.vae_batch, self.shard_size,
                           rank, world, device)

    def _verify_cache(self) -> dict[str, Any]:
        """Load and validate the latent manifest, raising a detailed error if not ready."""
        manifest_path = os.path.join(self.cache_dir, f"manifest-{self.split}.json")
        shard_files = sorted(glob.glob(
            os.path.join(self.cache_dir, f"latents-{self.split}-*.safetensors")))

        # Raise error if empty cache dir or incomplete cache data
        if not os.path.exists(manifest_path):
            expected = sum(pq.ParquetFile(f).metadata.num_rows
                           for f in split_files(self.root, self.split))
            if not os.path.isdir(self.cache_dir) or not os.listdir(self.cache_dir):
                raise RuntimeError(
                    f"❌ Cache dir is empty: {self.cache_dir}. "
                    f"Run preprocessing first.")
            cached = sum(shard_num_samples(f) for f in shard_files)
            raise RuntimeError(
                f"❌ Incomplete cache in {self.cache_dir}: expected {expected} samples, "
                f"found {len(shard_files)} shards with {cached} samples "
                f"(missing {expected - cached}). Re-run preprocessing.")

        with open(manifest_path) as f:
            manifest = json.load(f)
        missing = [s["file"] for s in manifest["shards"]
                   if not os.path.exists(os.path.join(self.cache_dir, s["file"]))]
        if missing:
            shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            raise RuntimeError(
                f"❌ Manifest lists {len(manifest['shards'])} shards, "
                f"{len(missing)} missing: {shown}")
        return manifest

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


# ==============================================================================
# Preprocessing: encode raw images to VAE latents and write the cache (torchrun).
# ==============================================================================

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


def build_latent_cache(
    root: str,
    split: str,
    img_size: int,
    cache_dir: str,
    vae_name: str,
    read_batch: int,
    vae_batch: int,
    shard_size: int,
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    """Encode this rank's parquet shards to fp16 VAE latents and save them.

    Uses the caller's distributed context (does not init/destroy the group); rank 0
    writes the manifest last and a barrier guards loading right after.
    """
    import torch.distributed as dist
    from diffusers.models import AutoencoderKL
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from safetensors.torch import save_file

    distributed = dist.is_available() and dist.is_initialized()
    vae = AutoencoderKL.from_pretrained(vae_name).to(device).eval()
    transform = _build_transform(img_size)
    class_text = load_class_text(root)
    os.makedirs(cache_dir, exist_ok=True)

    my_files = split_files(root, split)[rank::world]

    # Pending encode batch and output buffers (one shard flushed at shard_size).
    img_batch: list[torch.Tensor] = []
    lab_batch: list[int] = []
    uid_batch: list[np.ndarray] = []
    out_m, out_f, out_l, out_u = [], [], [], []
    state = {"count": 0, "shard": 0}

    @torch.no_grad()
    def flush_encode() -> None:
        if not img_batch:
            return
        x = torch.stack(img_batch).to(device)
        moments = DiagonalGaussianDistribution(vae._encode(x)).parameters
        flip = DiagonalGaussianDistribution(vae._encode(x.flip(dims=[3]))).parameters
        out_m.append(moments.cpu().half())
        out_f.append(flip.cpu().half())
        out_l.append(torch.tensor(lab_batch, dtype=torch.int16))
        out_u.append(np.stack(uid_batch))
        state["count"] += len(img_batch)
        img_batch.clear(); lab_batch.clear(); uid_batch.clear()

    def write_shard() -> None:
        if not out_m:
            return
        tensors = {
            "moments": torch.cat(out_m).contiguous(),
            "moments_flip": torch.cat(out_f).contiguous(),
            "label": torch.cat(out_l).contiguous(),
            "uid": torch.from_numpy(np.concatenate(out_u)).contiguous(),
        }
        path = os.path.join(cache_dir, f"latents-{split}-{rank:03d}-{state['shard']:05d}.safetensors")
        save_file(tensors, path, metadata={"img_size": str(img_size), "vae": vae_name})
        print(f"[rank {rank}] wrote {tensors['label'].shape[0]} samples -> {os.path.basename(path)}")
        out_m.clear(); out_f.clear(); out_l.clear(); out_u.clear()
        state["count"] = 0
        state["shard"] += 1

    for f in my_files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=read_batch, columns=["image", "label"]):
            for img_struct, label in zip(batch.column("image").to_pylist(),
                                         batch.column("label").to_pylist()):
                image_bytes = img_struct["bytes"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                text = class_text[label] if class_text else None
                uid = BaseTextImageDataset.compute_md5(image_bytes=image_bytes, text=text, label=label)
                img_batch.append(transform(image))
                lab_batch.append(label)
                uid_batch.append(np.frombuffer(bytes.fromhex(uid), dtype=np.uint8))
                if len(img_batch) >= vae_batch:
                    flush_encode()
                    if state["count"] >= shard_size:
                        write_shard()
    flush_encode()
    write_shard()

    if distributed:
        dist.barrier()  # all shards written before the manifest scan
    if rank == 0:
        _write_manifest(cache_dir, split, img_size, vae_name)
    if distributed:
        dist.barrier()  # manifest visible to every rank before loading

    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preprocess(args: argparse.Namespace) -> None:
    """Standalone entry: init the process group, build the cache, tear it down."""
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    build_latent_cache(args.root, args.split, args.img_size, args.cache_dir,
                       args.vae, args.read_batch, args.vae_batch, args.shard_size,
                       rank, world, device)
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode ImageNet-1K to VAE latents")
    p.add_argument("--root", required=True, help="Dataset root (contains data/ and classes.py)")
    p.add_argument("--cache_dir", required=True, help="Output dir for latent shards + manifest")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-ema")
    p.add_argument("--read_batch", type=int, default=256, help="Parquet read batch size")
    p.add_argument("--vae_batch", type=int, default=64, help="VAE encode batch size")
    p.add_argument("--shard_size", type=int, default=8192, help="Samples per output shard")
    return p.parse_args()


if __name__ == "__main__":
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun, e.g. torchrun --nproc_per_node=8 -m "
                           "nexus_align.datasets.imagenet_1k --root ... --cache_dir ...")
    preprocess(parse_args())
