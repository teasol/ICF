#!/usr/bin/env bash
#SBATCH --job-name=ru101margin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# RU-101: 7-branch per-branch margins for Primary 7, with provenance.
# Unblocks the (a) branch-level diagnosis of the 3/7 worsening that C-20260919-2 held.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. scripts/node_env.sh

export OFFICIAL=data/repro_labels_folds/official
export FEATURES=data/repro_labels_folds/features
CONFIG=configs/baseline/v121_7branch_active.yaml
OUT=predictions/margins_7b

echo "=== RU-101 margins ==="
echo "host=$(hostname) node=${SLURM_JOB_NODELIST:-?} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
"$ICF_PYTHON" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: cuda unavailable - exclude this node and resubmit.')
"
SECONDS=0
"$ICF_PYTHON" scripts/analysis/dump_branch_margins.py \
  --config "$CONFIG" --out "$OUT" --folds 50
rc=$?
echo "=== RU-101 elapsed=${SECONDS}s rc=${rc} out=${OUT} ==="
ls -la "$OUT" 2>/dev/null || true
exit "$rc"
