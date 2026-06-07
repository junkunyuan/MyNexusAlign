#!/bin/bash
# Prepare offline eval model caches (VAE + Inception); source me before eval.
#
# Pre-downloaded weights live under data_and_model/ and are linked into the local
# HF/torch caches on first use, so eval runs offline. Override the source root
# with DM=/path/to/data_and_model if your checkpoints live elsewhere.

DM=${DM:-$(pwd)/data_and_model}

VAE_DIR="$HOME/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-ema"
INCEPTION_FILE="$HOME/.cache/torch/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"

# VAE: link the local snapshot into the HF hub cache for offline from_pretrained.
if compgen -G "$VAE_DIR/snapshots/*/diffusion_pytorch_model.safetensors" > /dev/null; then
    echo "[skip] sd-vae-ft-ema already cached at $VAE_DIR"
else
    echo "[link] sd-vae-ft-ema from $DM/stabilityai/sd-vae-ft-ema"
    mkdir -p "$VAE_DIR/snapshots/local" "$VAE_DIR/refs"
    ln -sfn "$DM/stabilityai/sd-vae-ft-ema"/* "$VAE_DIR/snapshots/local/"
    echo local > "$VAE_DIR/refs/main"
fi

# Inception (torch_fidelity): link the single .pth into the torch hub cache.
if [[ -s "$INCEPTION_FILE" ]]; then
    echo "[skip] inception weights already cached at $INCEPTION_FILE"
else
    echo "[link] inception weights from $DM/torch_fidelity"
    mkdir -p "$(dirname "$INCEPTION_FILE")"
    ln -sfn "$DM/torch_fidelity/weights-inception-2015-12-05-6726825d.pth" "$INCEPTION_FILE"
fi

# Force offline mode so HF/transformers never reach the network even if available.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
