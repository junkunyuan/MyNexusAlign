#!/bin/bash
set -e

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

pip install lmdb hydra-core
pip install pyinstrument

export NCCL_DEBUG=WARN

source /opt/tiger/junkun.yuan/junkun_tools/merlin/ENV.sh

# The VAE latent cache is built automatically on first run: if cfg.data.cache_dir is
# empty, ImageNet1K preprocesses the parquet data in-place, then loads the latents.

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
    data=imagenet_1k \
