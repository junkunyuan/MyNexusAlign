# Evaluation

One-click FID/IS evaluation over every checkpoint of an experiment.

## Quick start

```bash
EXP_NAME=meanflow_l_2 bash eval.sh
```

This scans `logs/<EXP_NAME>/checkpoints/*.pt`, evaluates each checkpoint that
has not been evaluated yet, and refreshes a line chart after every new result.

Outputs land in `logs/<EXP_NAME>/eval/`:
- per-checkpoint samples + `metrics.json` (the "already evaluated" marker)
- `eval_summary.json` — aggregated FID/IS per step
- `eval_curve.png` — FID (↓) and Inception Score (↑) vs. training step

Re-running skips checkpoints whose `metrics.json` already exists, so it is safe
to launch repeatedly while training continues.

## Common overrides

```bash
NPROC=4 bash eval.sh                 # number of GPUs
bash eval.sh --min-step 50000        # only ckpts >= step 50000
bash eval.sh --dry-run               # print the plan, run nothing
NUM_FID_SAMPLES=5000 bash eval.sh    # quick smoke eval
```

Env vars: `EXP_NAME`, `OUTPUT_DIR` (default `logs`), `MODEL` (`SiT-L/2`),
`RESOLUTION`, `CFG_SCALE`, `NUM_STEPS`, `NUM_FID_SAMPLES`, `PER_PROC_BATCH`,
`NPROC`, `FID_STATS`. Any extra flags are forwarded to `eval_all`.

> Note: `meanflow_l_2` is trained with CFG baked in (`cfg-omega=0.2`), so the
> eval `CFG_SCALE` must stay at `1.0`.

## Internals

- `src/nexus_align/eval/eval_all.py` — sweep orchestrator (scan, skip, plot).
- `src/nexus_align/eval/evaluate.py` — single-checkpoint DDP sampler + metrics
  (the `torchrun` target).
- `src/nexus_align/eval/sampler.py` — MeanFlow sampler.
- `src/nexus_align/eval/metrics.py` — FID vs. cached stats + Inception Score.

Reference FID statistics (`fid_stats/adm_in256_stats.npz`) ship with the repo.
The VAE and Inception weights are pulled from HDFS into the local HF/torch
caches on first run.
