#!/usr/bin/env bash
# Evaluate the v38 ridge-ablation set (docs SS67) on the default task
# (bc_therapy/er_status, official 50-fold, SS64: bf16 + per-fold context caching,
# ~50 s/arm on one GPU).
#
# Each arm is scored with ITS OWN training config. Two reasons this is not
# optional: the arms are rare-free (SS61), and an ablated ridge's parameters
# never receive gradient, so re-enabling either branch at eval time injects an
# untrained branch into the logits.
#
# The best checkpoint is discovered by val_ce_loss in the filename rather than
# hardcoded, since the best epoch differs per arm.
#
# Usage: bash scripts/eval_v40_ridge_ablation.sh [gpu]
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PYTHON_BIN:-/home/aibio_3/miniconda3/envs/BagPFN/bin/python}"
TASK_DIR="/NHNHOME/BASE/kimds/Data/PathoBench/official/bc_therapy/er_status"
FEATURES="/NHNHOME/BASE/kimds/Data/PathoBench/features"
GPU="${1:-0}"
mkdir -p logs/official50 predictions

for arm in cv_only; do
  # Newest run directory for this arm, then its lowest-val_ce checkpoint.
  run_dir=$(ls -dt checkpoints/*/v40_${arm} 2>/dev/null | head -1)
  if [[ -z "$run_dir" ]]; then
    echo "!! no checkpoint dir for v40_${arm} -- skipping"
    continue
  fi
  ckpt=$(ls "$run_dir"/epoch=*val_ce_loss=*.ckpt 2>/dev/null \
         | sed -E 's/.*val_ce_loss=([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
  if [[ -z "$ckpt" ]]; then
    echo "!! no epoch checkpoint in $run_dir -- skipping"
    continue
  fi

  out="predictions/pathobench_bc_therapy_er_status_v40_${arm}_official50_bf16.pt"
  log="logs/official50/er_status_v40_${arm}_official50_bf16.log"
  echo "=== START v40_${arm} $(date)"
  echo "    ckpt: $ckpt"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/test_pathobench.py \
      --checkpoint "$ckpt" \
      --config "configs/archive/v40_v45_cvonly/train_v40_cv_only_1536.yaml" \
      --official-folds "$TASK_DIR" \
      --features "$FEATURES" \
      --input-dim 1536 \
      --precision bf16-mixed \
      --output "$out" > "$log" 2>&1
  echo "=== END v40_${arm} rc=$? $(date)"
  grep -E "fold-mean" "$log" || tail -3 "$log"
done
