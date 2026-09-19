#!/usr/bin/env bash
#SBATCH --job-name=ru99
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
# RU-99: fold 적합 디바이스 상주화의 수치 동등성·비용 실측.
# 같은 코드로 --cpu-bags(변경 전 동작)와 상주(변경 후)를 같은 노드·과제·fold로 돌려
# AUROC와 초/fold를 직접 비교한다. src/는 건드리지 않았고 호출부만 바뀌었다.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
. scripts/node_env.sh

CONFIG=configs/baseline/v121_active.yaml
TASK=cptac_pda/SMAD4_mutation
OUTDIR=scratch/ru99; mkdir -p "$OUTDIR"

echo "=== RU-99 provenance ==="
echo "host=$(hostname) job=${SLURM_JOB_ID:-?} node=${SLURM_JOB_NODELIST:-?} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "nvidia-smi 없음"
echo "commit=$(git rev-parse HEAD)"
echo "config_sha256=$(sha256sum $CONFIG | cut -c1-16)"
echo "src_changed=$(git diff --name-only HEAD -- src/ | wc -l)"
"$ICF_PYTHON" -c "import torch;print('torch='+torch.__version__,'cuda='+str(torch.cuda.is_available()))"
echo "========================"

"$ICF_PYTHON" -c "
import sys, torch
if not torch.cuda.is_available():
    sys.exit('FATAL: cuda unavailable — driver < cu130. exclude this node and resubmit.')
"

run_one () {
  local label="$1"; shift
  local out="$OUTDIR/smad4_${label}.pt"
  SECONDS=0
  "$ICF_PYTHON" scripts/evaluate_pure.py \
    --config "$CONFIG" \
    --features data/repro_labels_folds/features \
    --official-folds "data/repro_labels_folds/official/$TASK" \
    --official-nfolds 10 --device cuda:0 --output "$out" "$@"
  local rc=$?
  echo "=== ${label} elapsed=${SECONDS}s rc=${rc} out=${out} ==="
  echo "${label}=${SECONDS}" >> "$OUTDIR/timing.txt"
  return $rc
}

rm -f "$OUTDIR/timing.txt"
run_one before --cpu-bags || exit $?
run_one after  || exit $?

# 전후 AUROC 대조와 초/fold 요약
"$ICF_PYTHON" - <<'PY'
import torch, json
from pathlib import Path
out = Path("scratch/ru99")
res = {}
for label in ("before", "after"):
    d = torch.load(out / f"smad4_{label}.pt", weights_only=False)
    res[label] = d
    print(f"{label}: fold-mean {d['fold_auroc_mean']:.6f}  n={len(d['fold_aurocs'])}  "
          f"folds={d['fold_indices']}")
delta = res["after"]["fold_aurocs"]
deltas = [a - b for a, b in zip(res["after"]["fold_aurocs"], res["before"]["fold_aurocs"])]
print(f"Δ fold-mean = {res['after']['fold_auroc_mean'] - res['before']['fold_auroc_mean']:+.8f}")
print(f"max |Δfold AUROC| = {max(abs(x) for x in deltas):.3e}")
print("timing.txt:", (out / "timing.txt").read_text())
PY
echo "=== RU-99 done node=$(hostname) ==="
