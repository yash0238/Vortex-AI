#!/bin/bash
# DDP Launch Script for VORTEX-AI on 2x T4 GPUs
# Usage: ./scripts/train_ddp.sh [config_path] [additional_args...]

set -e

CONFIG="${1:-training/config.yaml}"
shift || true

# DDP environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export PYTHONFAULTHANDLER=1

# Use both GPUs (0 and 1)
GPUS=2

echo "========================================="
echo "VORTEX-AI DDP Training Launch"
echo "Config: $CONFIG"
echo "GPUs: $GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "========================================="

# Launch with torchrun
torchrun \
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_vortex.py \
    --config "$CONFIG" \
    "$@"