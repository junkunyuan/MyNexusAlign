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

set -e
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

export NCCL_DEBUG=WARN

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --standalone \
    src/nexus_align/cli/main.py "$@"
