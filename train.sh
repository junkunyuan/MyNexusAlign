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
  pip install lmdb hydra-core
  pip install pyinstrument
else
  mkdir -p data_and_model
  hdfs dfs -get -t 1024 "$HDFS_SRC" "$LOCAL_DST"
  echo "完成ImageNet拷贝"
fi

VAE_DST="data_and_model/stabilityai/sd-vae-ft-ema"
VAE_SRC=${SG}junkun/data_and_model/open_source/sd-vae-ft-ema.zip
if [[ -e "$VAE_DST/config.json" ]]; then
  echo "VAE 已存在,跳过下载: $VAE_DST"
else
  hdfs dfs -get "$VAE_SRC" data_and_model/sd-vae-ft-ema.zip
  unzip -q data_and_model/sd-vae-ft-ema.zip -d data_and_model/_vae_tmp
  mkdir -p "$VAE_DST"
  cp -rL data_and_model/_vae_tmp/models--stabilityai--sd-vae-ft-ema/snapshots/*/* "$VAE_DST"/
  rm -rf data_and_model/_vae_tmp data_and_model/sd-vae-ft-ema.zip
  echo "完成VAE拷贝"
fi

torchrun \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --nproc_per_node ${NPROC_PER_NODE} \
    --master_addr ${MASTER_ADDRESS} \
    --master_port ${MASTER_PORT3} \
    src/nexus_align/cli/main.py \
    data=imagenet_1k
