#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/test.yaml}"
SEED="${SEED:-42}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
# RTX 2080 Ti (Turing) supports fast FP16, but not native BF16.
PRECISION="${PRECISION:-16-mixed}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/predictions/ici_predictions_$(date +%Y%m%d_%H%M%S).pt}"

cd "${PROJECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

if [[ -z "${CHECKPOINT}" ]]; then
    echo "An architecture-v16 checkpoint path is required." >&2
    echo "Usage: ./test.sh checkpoints/path/to/model.ckpt" >&2
    exit 2
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    echo "Usage: ./test.sh [checkpoints/path/to/model.ckpt]" >&2
    exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 2
fi

mkdir -p "$(dirname -- "${OUTPUT}")"

echo "Host: $(hostname)"
echo "GPU(s): ${CUDA_DEVICES}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Precision: ${PRECISION}"
echo "Output: ${OUTPUT}"

if ! nvidia-smi --query-gpu=index,name,memory.used,memory.total \
    --format=csv,noheader; then
    echo "NVIDIA GPU is not accessible from this interactive shell." >&2
    exit 3
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" scripts/test.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --output "${OUTPUT}" \
    --accelerator gpu \
    --devices 1 \
    --precision "${PRECISION}"

echo "ICI five-fold in-context validation and external inference completed."
