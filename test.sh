#!/bin/bash
set -e

LOCAL_DST="/opt/tiger/imagenet_train_latents.lmdb"
HDFS_SRC=hdfs://harunasg/home/byte_icvg_aigc_cp/user/video/junkun/data_and_model/open_source/ILSVRC/imagenet-1k/data_MeanFlow/imagenet_train_latents.lmdb

if [[ -e "$LOCAL_DST" ]]; then
  echo "ImageNet 已存在,跳过下载: $LOCAL_DST"
else
  mkdir -p /opt/tiger/MeanFlow/data_and_model
  hdfs dfs -get -t 1024 "$HDFS_SRC" "$LOCAL_DST"
  echo "完成ImageNet拷贝"
fi

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

pip install lmdb hydra-core
pip install pyinstrument

export NCCL_DEBUG=WARN

source /opt/tiger/junkun_tools/merlin/ENV.sh

# fuser -k -9 ${MASTER_PORT}/tcp 2>/dev/null || true
# pkill -9 -f "src/nexus_align/cli/main.py" 2>/dev/null || true
# pkill -9 -f "torchrun.*--master_port ${MASTER_PORT}" 2>/dev/null || true
if fuser "${MASTER_PORT}"/tcp >/dev/null 2>&1; then
    echo "⚠️  [警告] 端口 ${MASTER_PORT} 没清干净,仍被占用!torchrun 很可能报 EADDRINUSE。"
    echo "⚠️  请在所有节点手动清理:pkill -9 -f 'src/nexus_align/cli/main.py'; pkill -9 -f torchrun; fuser -k -9 ${MASTER_PORT}/tcp"
fi

torchrun \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDRESS} \
    --master_port ${MASTER_PORT} \
    src/nexus_align/cli/main.py \
