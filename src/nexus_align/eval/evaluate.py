"""Single-checkpoint evaluator: DDP sampling then FID/IS on rank 0."""

import argparse
import glob
import json
import math
import os

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf

import nexus_align.models  # noqa: F401  # registers model factories on import
from nexus_align.models.meanflow_sit import MeanFlowSiT_models
from nexus_align.registry import registry
from nexus_align.engine.distributed import init_dist_env
from nexus_align.eval.sampler import meanflow_sampler
from nexus_align.eval.metrics import compute_metrics_with_cached_stats

# VAE latent normalization (must match training in cli/main.py).
LATENTS_SCALE = 0.18215
LATENTS_BIAS = 0.0


def sample_folder_name(args):
    """Folder uniquely identifying this (ckpt, resolution, cfg, steps, seed) run."""
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    return (
        f"meanflow-{model_string_name}-{ckpt_string_name}-size-{args.resolution}-"
        f"cfg-{args.cfg_scale}-steps-{args.num_steps}-seed-{args.global_seed}"
    )


def main(args):
    """Sample images from one checkpoint, then compute FID/IS on rank 0."""
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Evaluation with DDP requires at least one GPU"
    torch.set_grad_enabled(False)

    world_size, rank, device = init_dist_env()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Build model from the registry and load the EMA weights.
    latent_size = args.resolution // 8
    # cfg_prob > 0 enables CFG (the model factory reads cfg.model.cfg_prob).
    cfg = OmegaConf.create({
        "model": {
            "resolution": args.resolution,
            "num_classes": args.num_classes,
            "cfg_prob": 1.0,
        }
    })
    model = registry.get("model", args.model)(cfg).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)["ema"]
    model.load_state_dict(state_dict)
    model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"

    eval_fid_dir = os.path.join(args.sample_dir, sample_folder_name(args))
    img_folder = os.path.join(eval_fid_dir, "img_dir")
    if rank == 0:
        os.makedirs(img_folder, exist_ok=True)
        print(f"Saving .png samples at {eval_fid_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Using {args.num_steps}-step sampling")

    samples_per_gpu = total_samples // world_size
    assert samples_per_gpu % n == 0, "samples_per_gpu must be divisible by per-GPU batch size"
    iterations = samples_per_gpu // n

    # Skip sampling if a previous run already produced every image.
    existing = len(glob.glob(os.path.join(img_folder, "*.png")))
    if existing >= total_samples:
        if rank == 0:
            print(f"[skip-sample] {existing} images already exist in {img_folder}")
    else:
        if rank == 0 and existing > 0:
            print(f"[resample] only {existing}/{total_samples} found — resampling all")
        pbar = tqdm(range(iterations)) if rank == 0 else range(iterations)
        latents_scale = torch.full((1, 4, 1, 1), LATENTS_SCALE, device=device)
        latents_bias = torch.full((1, 4, 1, 1), LATENTS_BIAS, device=device)
        total = 0
        for _ in pbar:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)
            samples = meanflow_sampler(
                model=model, latents=z, y=y,
                cfg_scale=args.cfg_scale, num_steps=args.num_steps,
            ).to(torch.float32)
            samples = vae.decode((samples - latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.0
            samples = torch.clamp(255.0 * samples, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
            for i, sample in enumerate(samples):
                index = i * world_size + rank + total
                Image.fromarray(sample).save(f"{img_folder}/{index:06d}.png")
            total += global_batch_size

    dist.barrier()

    if rank == 0 and args.compute_metrics:
        print("Computing evaluation metrics...")
        if args.resolution != 256:
            raise NotImplementedError
        assert args.fid_statistics_file and os.path.exists(args.fid_statistics_file), \
            f"FID stats file not found: {args.fid_statistics_file}"

        metrics_dict = compute_metrics_with_cached_stats(
            img_folder=img_folder,
            fid_stats_file=args.fid_statistics_file,
            device=device,
        )
        fid = metrics_dict.get("frechet_inception_distance")
        is_mean = metrics_dict.get("inception_score_mean")
        is_std = metrics_dict.get("inception_score_std")

        print("\n===== Evaluation Results =====")
        if fid is not None:
            print(f"FID: {fid:.2f}")
        if is_mean is not None:
            print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")

        metrics_file = os.path.join(eval_fid_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved to {metrics_file}")

    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--global-seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, required=True, help="Path to a MeanFlow checkpoint.")
    p.add_argument("--sample-dir", type=str, default="samples")
    p.add_argument("--model", type=str, choices=list(MeanFlowSiT_models.keys()), default="MeanFlowSiT-L/2")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    p.add_argument("--per-proc-batch-size", type=int, default=32)
    p.add_argument("--num-fid-samples", type=int, default=50_000)
    p.add_argument("--num-steps", type=int, default=1, help="Number of sampling steps")
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--compute-metrics", action="store_true", help="Compute FID and IS after sampling")
    p.add_argument("--fid-statistics-file", type=str, default="", help="Path to FID statistics .npz")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
