#!/usr/bin/env bash
#SBATCH --job-name=xrepro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
# RU-98: 같은 코드·설정·데이터를 다른 장비에서 돌렸을 때 과제별 AUROC이 재현되는가.
# 과제 하나가 array 원소 하나다. 배치는 Slurm에 맡기고, 어느 노드였는지는 로그가 남긴다.
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
CONFIG=configs/baseline/v121_active.yaml
OUTDIR=predictions/xmachine; mkdir -p "$OUTDIR"

# provenance -- 이 값들이 없으면 결과를 어느 기계에 귀속시킬 수 없다.
echo "=== RU-98 provenance ==="
echo "task=$TASK"
echo "host=$(hostname)"
echo "slurm_node=${SLURM_JOB_NODELIST:-?} job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_TASK_ID:-?}"
echo "cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "nvidia-smi 없음"
echo "commit=$(git rev-parse HEAD)"
echo "config_sha256=$(sha256sum $CONFIG | cut -c1-16)"
"$ICF_PYTHON" -c "import torch;print('torch='+torch.__version__, 'cuda_available='+str(torch.cuda.is_available()))"
echo "========================"

# 드라이버가 낮으면 여기서 크게 실패한다. 조용히 CPU로 떨어져 며칠 뒤에 알아차리는 일이 없도록.
"$ICF_PYTHON" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: torch.cuda.is_available()=False — 이 노드의 드라이버가 cu130(>=580)에 못 미친다. 이 노드를 --exclude 하고 재제출하라.')
" || exit 3

SECONDS=0
"$ICF_PYTHON" scripts/evaluate_pure.py \
  --config "$CONFIG" \
  --features data/repro_labels_folds/features \
  --official-folds "data/repro_labels_folds/official/$TASK" \
  --official-nfolds 50 \
  --device cuda:0 \
  --output "$OUTDIR/$(echo "$TASK" | tr '/' '_').pt"
RC=$?
echo "=== elapsed=${SECONDS}s rc=$RC task=$TASK node=$(hostname) ==="
exit $RC
