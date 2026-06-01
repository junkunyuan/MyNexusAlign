set -e
cd "$(dirname "$0")"

export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

export NCCL_DEBUG=WARN

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --standalone \
    src/nexus_align/cli/main.py "$@"
