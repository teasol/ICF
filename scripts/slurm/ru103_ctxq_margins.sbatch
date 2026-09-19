#!/usr/bin/env bash
#SBATCH --job-name=ru103ctxq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# RU-103: dump context self-margins + query margins for the TGW transfer test.
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
"$ICF_PYTHON" scripts/analysis/dump_context_query_margins.py \
  --config configs/baseline/v121_7branch_active.yaml --out predictions/ctxq_7b --folds 50
rc=$?
echo "=== RU-103 elapsed=${SECONDS}s rc=${rc} ==="
exit "$rc"
