#!/bin/bash
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
cd /opt/tiger

# ---- Data ---------------------------------------------------------------------
LOCAL_DST="/opt/tiger/MeanFlow/data_and_model/imagenet_train_latents.lmdb"
HDFS_SRC=${SG}junkun/data_and_model/open_source/ILSVRC/imagenet-1k/data_MeanFlow/imagenet_train_latents.lmdb

if [[ -e "$LOCAL_DST" ]]; then
  echo "ImageNet 已存在,跳过下载: $LOCAL_DST"
else
  mkdir -p /opt/tiger/MeanFlow/data_and_model
  hdfs dfs -get -t 1024 "$HDFS_SRC" "$LOCAL_DST"
  echo "完成ImageNet拷贝"
fi

# ---- Deps ---------------------------------------------------------------------
pip install lmdb hydra-core
pip install pyinstrument

# ---- Distributed env ----------------------------------------------------------
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"   # absolute; no trailing ":"
export NCCL_DEBUG=WARN
export WANDB_DIR="$REPO/logs"                              # keep wandb out of /opt/tiger

NNODES=$ARNOLD_NUM
NODE_RANK=$ARNOLD_ID
NPROC_PER_NODE=$ARNOLD_WORKER_GPU
MASTER_ADDRESS=$ARNOLD_WORKER_0_HOST
MASTER_PORT=10861   # PORT1（PORT0 被 sshd 占用）

torchrun \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDRESS} \
    --master_port ${MASTER_PORT} \
    "$REPO/src/nexus_align/cli/main.py" \
    hydra.job.chdir=false \
    log.log_dir="$REPO/logs" \
    data.lmdb_path="$LOCAL_DST" \
    "$@"
