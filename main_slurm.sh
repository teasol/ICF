#!/usr/bin/env bash

#SBATCH --job-name=icf-synthetic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Submit this script from the repository root. Slurm preserves that directory
# in SLURM_SUBMIT_DIR even when it executes a spooled copy of this file.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CONFIG="${CONFIG:-configs/train_synthetic.yaml}"
SEED="${SEED:-42}"
NPROC_PER_NODE=8

RUN_TIME="$(date +%Y%m%d_%H%M)"
RUN_ID="${SLURM_JOB_ID:-local}"
RUN_NAME="synthetic_${RUN_TIME}_${RUN_ID}"
RUN_GROUP="synthetic_${RUN_TIME}"
CHECKPOINT_DIR="checkpoints/${RUN_TIME}_${RUN_ID}/synthetic"

cd "${PROJECT_ROOT}"
mkdir -p "${CHECKPOINT_DIR}"

if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
    echo "torchrun executable not found: ${TORCHRUN_BIN}" >&2
    exit 2
fi

# P2P/NVLink has previously caused NCCL hangs on this cluster.
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-assigned by Slurm}"
echo "Config: ${CONFIG}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"

srun --ntasks=1 "${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/train.py \
    --config "${CONFIG}" \
    --seed "${SEED}" \
    --run-name "${RUN_NAME}" \
    --run-group "${RUN_GROUP}" \
    --checkpoint-dir "${CHECKPOINT_DIR}"

echo "Synthetic pretraining completed successfully."
