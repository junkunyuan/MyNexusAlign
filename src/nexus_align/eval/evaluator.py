"""Evaluators: pluggable model evaluation for in-training and standalone runs."""

import glob
import json
import math
import os
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL

from nexus_align.eval.sampler import meanflow_sampler

# VAE latent normalization (must match training in cli/main.py).
LATENTS_SCALE = 0.18215
LATENTS_BIAS = 0.0


class BaseEvaluator(ABC):
    """
    Base evaluator: evaluate a model and return a metrics dict on rank 0.

    Options are read from cfg.eval; outputs go to <sample_dir>/<tag>/.
    Subclass and register under "evaluator" to add new eval methods.
    """

    def __init__(self, cfg, device: torch.device) -> None:
        self.cfg = cfg
        self.eval_cfg = cfg.eval
        self.device = device
        self.sample_dir = self.eval_cfg.get("sample_dir") or os.path.join(cfg.log.log_dir, "eval")

    @abstractmethod
    def evaluate(self, model, tag: str) -> dict | None:
        """Evaluate model; tag names the output folder (e.g. the training step)."""
        ...


class FidEvaluator(BaseEvaluator):
    """Sample images with the model on all ranks, then compute FID/IS on rank 0."""

    def __init__(self, cfg, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._vae = None  # loaded lazily on first evaluate()

    @property
    def vae(self):
        if self._vae is None:
            self._vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device)
        return self._vae

    @torch.no_grad()
    def evaluate(self, model, tag: str) -> dict | None:
        e = self.eval_cfg
        world_size, rank = self.cfg.common.world_size, self.cfg.common.rank
        assert e.cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"

        out_dir = os.path.join(self.sample_dir, tag)
        img_folder = os.path.join(out_dir, "img_dir")
        metrics_file = os.path.join(out_dir, "metrics.json")

        # metrics.json is the "already evaluated" marker (same convention as eval_all).
        if os.path.exists(metrics_file):
            if rank == 0:
                print(f"[skip] {tag} already evaluated: {metrics_file}")
            with open(metrics_file) as f:
                return json.load(f)

        if rank == 0:
            os.makedirs(img_folder, exist_ok=True)
            print(f"🔍 Evaluating with {e.num_steps}-step sampling, saving .png samples at {out_dir}")
        dist.barrier()

        n = e.per_proc_batch_size
        global_batch_size = n * world_size
        total_samples = int(math.ceil(e.num_fid_samples / global_batch_size) * global_batch_size)
        iterations = total_samples // world_size // n
        latent_size = self.cfg.model.resolution // 8

        # Dedicated per-rank generator: reproducible eval that leaves training RNG untouched.
        gen = torch.Generator(self.device)
        gen.manual_seed(e.global_seed * world_size + rank)

        # Skip sampling if a previous run already produced every image.
        existing = len(glob.glob(os.path.join(img_folder, "*.png")))
        if existing >= total_samples:
            if rank == 0:
                print(f"[skip-sample] {existing} images already exist in {img_folder}")
        else:
            if rank == 0 and existing > 0:
                print(f"[resample] only {existing}/{total_samples} found — resampling all")
            pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
            total = 0
            for _ in pbar:
                z = torch.randn(n, model.in_channels, latent_size, latent_size,
                                device=self.device, generator=gen)
                y = torch.randint(0, self.cfg.model.num_classes, (n,),
                                  device=self.device, generator=gen)
                samples = meanflow_sampler(
                    model=model, latents=z, y=y,
                    cfg_scale=e.cfg_scale, num_steps=e.num_steps,
                ).to(torch.float32)
                samples = self.vae.decode((samples - LATENTS_BIAS) / LATENTS_SCALE).sample
                samples = (samples + 1) / 2.0
                samples = torch.clamp(255.0 * samples, 0, 255)
                samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
                for i, sample in enumerate(samples):
                    index = i * world_size + rank + total
                    Image.fromarray(sample).save(f"{img_folder}/{index:06d}.png")
                total += global_batch_size

        dist.barrier()

        metrics = None
        if rank == 0 and e.get("compute_metrics", True):
            print("Computing evaluation metrics...")
            if self.cfg.model.resolution != 256:
                raise NotImplementedError
            assert e.fid_statistics_file and os.path.exists(e.fid_statistics_file), \
                f"FID stats file not found: {e.fid_statistics_file}"

            # Imported lazily: torch_fidelity is only required when metrics are computed.
            from nexus_align.eval.metrics import compute_metrics_with_cached_stats
            metrics = compute_metrics_with_cached_stats(
                img_folder=img_folder,
                fid_stats_file=e.fid_statistics_file,
                device=self.device,
            )
            fid = metrics.get("frechet_inception_distance")
            is_mean = metrics.get("inception_score_mean")
            is_std = metrics.get("inception_score_std")

            print("\n===== Evaluation Results =====")
            if fid is not None:
                print(f"FID: {fid:.2f}")
            if is_mean is not None:
                print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")

            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved to {metrics_file}")

        dist.barrier()
        return metrics
