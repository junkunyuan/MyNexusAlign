#!/bin/bash
# One-click: evaluate every checkpoint of an experiment, skip those already done,
# and auto-refresh the FID/IS line chart after each new ckpt.
#
# Usage:
#   bash eval.sh                       # use defaults
#   NPROC=4 bash eval.sh               # override # of GPUs
#   EXP_NAME=meanflow_l_2 bash eval.sh # pick the experiment to evaluate
#   bash eval.sh --min-step 50000      # only evaluate ckpts >= step 50000
#   bash eval.sh --dry-run             # print plan, don't run
#
# Pre-downloaded weights (VAE + Inception + FID stats) live under data_and_model/
# and are loaded directly from there, so eval runs fully offline.
set -euo pipefail

EXP_NAME=${EXP_NAME:-default_20260607-213925}
OUTPUT_DIR=${OUTPUT_DIR:-logs}
MODEL=${MODEL:-MeanFlowSiT-L/2}
RESOLUTION=${RESOLUTION:-256}
# meanflow_l_2 trains with CFG (cfg-omega=0.2) -> eval cfg-scale must be 1.0.
CFG_SCALE=${CFG_SCALE:-1.0}
NUM_STEPS=${NUM_STEPS:-1}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-50000}
PER_PROC_BATCH=${PER_PROC_BATCH:-32}
NPROC=${NPROC:-${ARNOLD_WORKER_GPU:-8}}

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

# Unified entry point for all pre-downloaded data and model weights (HF naming).
DATA_AND_MODEL_DIR=${DATA_AND_MODEL_DIR:-$(pwd)/data_and_model}
FID_STATS=${FID_STATS:-$DATA_AND_MODEL_DIR/fid_stats/adm_in256_stats.npz}

# Force offline mode so HF/transformers never reach the network even if available.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m nexus_align.eval.eval_all \
    --exp-name "$EXP_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --resolution "$RESOLUTION" \
    --cfg-scale "$CFG_SCALE" \
    --num-steps "$NUM_STEPS" \
    --num-fid-samples "$NUM_FID_SAMPLES" \
    --per-proc-batch-size "$PER_PROC_BATCH" \
    --nproc-per-node "$NPROC" \
    --fid-statistics-file "$FID_STATS" \
    --data-and-model-dir "$DATA_AND_MODEL_DIR" \
    "$@"
