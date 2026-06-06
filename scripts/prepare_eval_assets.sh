#!/bin/bash
# Prepare offline eval model caches (VAE + Inception); source me before eval.
#
# Pre-downloaded weights are pulled from HDFS on first use and cached locally;
# subsequent runs skip the download/extract step. Override the source with
# HDFS_MODELS_DIR=hdfs://harunasg/... if running on the SG cluster.

HDFS_MODELS_DIR=${HDFS_MODELS_DIR:-${FR}junkun/data_and_model/open_source}

VAE_DIR="$HOME/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-ema"
INCEPTION_FILE="$HOME/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"

# VAE: ready iff the snapshots dir contains at least one safetensors file
if compgen -G "$VAE_DIR/snapshots/*/diffusion_pytorch_model.safetensors" > /dev/null; then
    echo "[skip] sd-vae-ft-ema already cached at $VAE_DIR"
else
    echo "[get]  sd-vae-ft-ema from $HDFS_MODELS_DIR/sd-vae-ft-ema.zip"
    mkdir -p "$HOME/.cache/huggingface/hub"
    hdfs dfs -get -t 32 "$HDFS_MODELS_DIR/sd-vae-ft-ema.zip" /tmp/sd-vae-ft-ema.zip
    unzip -q -o /tmp/sd-vae-ft-ema.zip -d "$HOME/.cache/huggingface/hub"
    rm -f /tmp/sd-vae-ft-ema.zip
    echo "[done] sd-vae-ft-ema -> $VAE_DIR"
fi

# Inception (torch_fidelity): single .pth file
if [[ -s "$INCEPTION_FILE" ]]; then
    echo "[skip] inception weights already cached at $INCEPTION_FILE"
else
    echo "[get]  inception weights from $HDFS_MODELS_DIR/weights-inception-2015-12-05-6726825d.pth"
    mkdir -p "$(dirname "$INCEPTION_FILE")"
    hdfs dfs -get -t 32 \
        "$HDFS_MODELS_DIR/weights-inception-2015-12-05-6726825d.pth" \
        "$INCEPTION_FILE"
    echo "[done] inception -> $INCEPTION_FILE"
fi

# Force offline mode so HF/transformers never reach the network even if available.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
