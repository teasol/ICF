#!/usr/bin/env bash

#SBATCH --job-name=ici-icl-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Submit this script from the repository root. Slurm preserves that directory
# in SLURM_SUBMIT_DIR even when it executes a spooled copy of this file.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT="${1:-${CHECKPOINT:-}}"
CONFIG="${CONFIG:-configs/test.yaml}"
SEED="${SEED:-42}"
PRECISION="${PRECISION:-16-mixed}"
OUTPUT="${OUTPUT:-predictions/ici_predictions_${SLURM_JOB_ID}.pt}"

cd "${PROJECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

if [[ -z "${CHECKPOINT}" ]]; then
    echo "An architecture-v11 checkpoint is required." >&2
    echo "Usage: sbatch test_slurm.sh checkpoints/path/to/model.ckpt" >&2
    exit 2
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi

mkdir -p "$(dirname -- "${OUTPUT}")"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Precision: ${PRECISION}"
echo "Output: ${OUTPUT}"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

srun --ntasks=1 "${PYTHON_BIN}" scripts/test.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --output "${OUTPUT}" \
    --accelerator gpu \
    --devices 1 \
    --precision "${PRECISION}"

echo "ICI five-fold in-context validation and external inference completed."
