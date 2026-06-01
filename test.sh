#!/bin/bash
LOCAL_DST="/opt/tiger/MeanFlow/data_and_model/imagenet_train_latents.lmdb"
HDFS_SRC=${SG}junkun/data_and_model/open_source/ILSVRC/imagenet-1k/data_MeanFlow/imagenet_train_latents.lmdb
LOCAL_SRC=/mnt/hdfs/sg/junkun/data_and_model/open_source/ILSVRC/imagenet-1k/data_MeanFlow/imagenet_train_latents.lmdb

if [[ -e "$LOCAL_DST" ]]; then
  echo "ImageNet 已存在,跳过下载: $LOCAL_DST"
else
  mkdir -p /opt/tiger/MeanFlow/data_and_model
  hdfs dfs -get -t 1024 "$HDFS_SRC" "$LOCAL_DST"
  echo "完成ImageNet拷贝"
fi

pip install lmdb
pip install hydra-core

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

export NCCL_DEBUG=WARN

NNODES=$ARNOLD_NUM
NODE_RANK=$ARNOLD_ID
NPROC_PER_NODE=$ARNOLD_WORKER_GPU
MASTER_ADDRESS=$ARNOLD_WORKER_0_HOST
MASTER_PORT="${ARNOLD_WORKER_0_PORT##*,}"   # 用 PORT1（PORT0 被 sshd 占用）
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

torchrun \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --rdzv_backend c10d \
    --rdzv_endpoint "[${MASTER_ADDRESS}]:${MASTER_PORT}" \
    --rdzv_id exp \
    src/nexus_align/cli/main.py "$@"
