#!/usr/bin/env bash
#SBATCH --job-name=ru100cost
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
# RU-100: 7-branch 단가를 전용 GPU 에서 오염 없이 잰다 (동시 실행의 경합 제거).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. scripts/node_env.sh
mkdir -p scratch/ru100
echo "host=$(hostname) node=${SLURM_JOB_NODELIST:-?} commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
SECONDS=0
"$ICF_PYTHON" scripts/evaluate_pure.py \
  --config configs/baseline/v121_7branch_active.yaml \
  --features data/repro_labels_folds/features \
  --official-folds data/repro_labels_folds/official/cptac_pda/SMAD4_mutation \
  --official-nfolds 10 --device cuda:0 --output scratch/ru100/7br_smad4_10.pt
echo "SEVENBR_ELAPSED_10FOLD_SECONDS=${SECONDS}"
echo "SEVENBR_SECONDS_PER_FOLD=$(echo "scale=3; ${SECONDS}/10" | bc)"
