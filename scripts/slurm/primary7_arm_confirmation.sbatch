#!/usr/bin/env bash
#SBATCH --job-name=armconf
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# RU-100: 5-branch 대 7-branch Primary 7 paired 재확증 (provenance-fixed).
# 과제 하나가 array 원소 하나. 그 원소가 두 arm 을 같은 노드·GPU·세션에서 순차 실행한다
# (기준선 재사용 금지 - 세션 간 드리프트를 Δ에 섞지 않는다).
# 출력을 삭제된 parity_* 이름으로 되살려 scripts/analysis/parity_paired_delta.py 가 그대로 돈다.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. scripts/node_env.sh

TASKS=(
  cptac_lscc/ARID1A_mutation
  cptac_lscc/Histologic_Grade
  cptac_lscc/KEAP1_mutation
  cptac_luad/KRAS_mutation
  cptac_pda/SMAD4_mutation
  ucla_lung/progression_regression
  cptac_ccrcc/PBRM1_mutation
)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
NAME="$(echo "$TASK" | tr '/' '_')"
mkdir -p predictions

echo "=== RU-100 ${TASK} node=$(hostname) job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_TASK_ID:-?} ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "nvidia-smi 없음"
echo "commit=$(git rev-parse HEAD)"
"$ICF_PYTHON" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: cuda unavailable - exclude this node and resubmit.')
"

for arm in active 7branch_active; do
  cfg="configs/baseline/v121_${arm}.yaml"
  out="predictions/parity_${NAME}_v121_${arm}.pt"
  SECONDS=0
  echo "--- arm=${arm} config=$cfg sha=$(sha256sum "$cfg" | cut -c1-16) ---"
  "$ICF_PYTHON" scripts/evaluate_pure.py \
    --config "$cfg" \
    --features data/repro_labels_folds/features \
    --official-folds "data/repro_labels_folds/official/$TASK" \
    --official-nfolds 50 --device cuda:0 --output "$out"
  rc=$?
  echo "=== ${TASK} arm=${arm} elapsed=${SECONDS}s rc=${rc} out=${out} ==="
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo "=== RU-100 ${TASK} done ==="
