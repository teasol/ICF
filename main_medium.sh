#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
exec "${PROJECT_ROOT}/scripts/launch_interactive_training.sh" \
    medium "${CONFIG:-configs/train_medium.yaml}"
