#!/bin/bash
set -e

source /opt/tiger/junkun.yuan/junkun_tools/merlin/ENV.sh

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export NCCL_DEBUG=WARN

mkdir -p data_and_model
LOCAL_DST="data_and_model/ILSVRC"
HDFS_SRC=${SG}junkun/data_and_model/open_source/ILSVRC

if [[ -e "$LOCAL_DST" ]]; then
  echo "ImageNet 已存在,跳过下载: $LOCAL_DST"
else
  mkdir -p data_and_model
  hdfs dfs -get -t 1024 "$HDFS_SRC" "$LOCAL_DST"
  echo "完成ImageNet拷贝"
fi


pip install lmdb hydra-core
pip install pyinstrument

# fuser -k -9 ${MASTER_PORT}/tcp 2>/dev/null || true
# pkill -9 -f "src/nexus_align/cli/main.py" 2>/dev/null || true
# pkill -9 -f "torchrun.*--master_port ${MASTER_PORT}" 2>/dev/null || true

torchrun \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDRESS} \
    --master_port ${MASTER_PORT} \
    src/nexus_align/cli/main.py \
    data=imagenet_1k \
