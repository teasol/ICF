#!/usr/bin/env bash
#SBATCH --job-name=fcprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# FC (focus-contrast) probe: compute FC margins and merge with official margins for gate-1 screening.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. scripts/node_env.sh
export OFFICIAL=data/repro_labels_folds/official
export FEATURES=data/repro_labels_folds/features
echo "host=$(hostname) node=${SLURM_JOB_NODELIST:-?} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
"$ICF_PYTHON" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: cuda unavailable - exclude this node and resubmit.')
"
SECONDS=0
"$ICF_PYTHON" scripts/analysis/fc_probe.py \
  --config configs/baseline/v121_7branch_active.yaml \
  --official predictions/margins_7b --out predictions/fc_gate \
  --taus 0.5,1.0,2.0,4.0 --dim 32 --lambda 1.0
rc=$?
echo "=== FC probe elapsed=${SECONDS}s rc=${rc} ==="
exit "$rc"
