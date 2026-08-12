#!/usr/bin/env bash
# Evaluate the v36 Q1 / v37 arms on the default task (bc_therapy/er_status,
# official 50-fold, docs §64). Each arm is scored with ITS OWN training config:
# all four arms are rare-free, so the v34 config would inject an untrained
# rare branch (§62-7 warning in the v36 configs).
#
# Usage: bash scripts/eval_v36_v37_arms.sh <gpu> <arm> [<arm> ...]
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON_BIN:-/home/aibio_3/miniconda3/envs/BagPFN/bin/python}"
TASK_DIR="/NHNHOME/BASE/kimds/Data/PathoBench/official/bc_therapy/er_status"
FEATURES="/NHNHOME/BASE/kimds/Data/PathoBench/features"
LOGDIR="logs/official50"
mkdir -p "$LOGDIR" predictions

GPU="$1"; shift

for arm in "$@"; do
  case "$arm" in
    v36_baseline)
      CKPT="checkpoints/20260808_144323/v36_q1_baseline/epoch=049-val_ce_loss=0.3402.ckpt"
      CFG="configs/archive/v35_v39_pre_cvonly/train_v36_q1_baseline_1536.yaml" ;;
    v36_structured)
      CKPT="checkpoints/20260808_144330/v36_q1_structured/epoch=049-val_ce_loss=0.3405.ckpt"
      CFG="configs/archive/v35_v39_pre_cvonly/train_v36_q1_structured_1536.yaml" ;;
    v37_baseline)
      CKPT="checkpoints/20260808_180137/v37_baseline/epoch=132-val_ce_loss=0.3354.ckpt"
      CFG="configs/archive/v35_v39_pre_cvonly/train_v37_baseline_1536.yaml" ;;
    v37_context_adaptive)
      CKPT="checkpoints/20260808_180145/v37_context_adaptive/epoch=088-val_ce_loss=0.3372.ckpt"
      CFG="configs/archive/v35_v39_pre_cvonly/train_v37_context_adaptive_1536.yaml" ;;
    *) echo "unknown arm: $arm"; exit 2 ;;
  esac

  OUT="predictions/pathobench_bc_therapy_er_status_${arm}_official50_bf16.pt"
  LOG="$LOGDIR/er_status_${arm}_official50_bf16.log"
  echo "=== START ${arm} on GPU ${GPU} $(date) ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/test_pathobench.py \
      --checkpoint "$CKPT" \
      --config "$CFG" \
      --official-folds "$TASK_DIR" \
      --features "$FEATURES" \
      --input-dim 1536 \
      --precision bf16-mixed \
      --output "$OUT" > "$LOG" 2>&1
  echo "=== END ${arm} rc=$? $(date) ==="
done
