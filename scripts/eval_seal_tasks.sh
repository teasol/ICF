#!/usr/bin/env bash
# Evaluate one checkpoint on the SEAL-comparable official tasks (docs SS70-6).
#
# Which tasks: docs/seal_univ2_baseline_17tasks.csv marks 10 rows `in_seal=yes`
# -- the only ones with a published SEAL ABMIL/MeanMIL number on the SAME cohort
# and the same 50-fold protocol. er_status is already done, so this covers the
# remaining 9. The other 7 rows of that CSV have no SEAL counterpart and are not
# part of this comparison.
#
# Everything so far has been er_status alone; a single task cannot support a
# "beats supervised SEAL" claim. This is that check.
#
# Usage: bash scripts/eval_seal_tasks.sh <gpu> <ckpt> <config> <tag> <task>...
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"   # docs SS164: node paths in one place
PY="$PYTHON_BIN"


GPU="$1"; CKPT="$2"; CONFIG="$3"; TAG="$4"; shift 4
mkdir -p logs/official50 predictions
overall_rc=0
for task in "$@"; do
  name="${task//\//_}"
  out="predictions/pathobench_${name}_${TAG}_official50_bf16.pt"
  log="logs/official50/${name}_${TAG}.log"
  echo "=== START ${task} $(date +%H:%M:%S)"
  ckpt_args=()
  if [ -n "$CKPT" ]; then
    ckpt_args=(--checkpoint "$CKPT")
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/test_pathobench.py \
    "${ckpt_args[@]}" --config "$CONFIG" \
    --official-folds "$OFFICIAL/$task" --features "$FEATURES" \
    --input-dim 1536 --precision bf16-mixed --output "$out" > "$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then overall_rc="$rc"; fi
  res=$(grep -aoE "fold-mean AUROC: [0-9.]+ ± [0-9.]+   pooled AUROC: [0-9.]+" "$log" | tail -1)
  echo "=== END   ${task} rc=$rc $(date +%H:%M:%S)  ${res:-$(tail -2 "$log")}"
done
exit "$overall_rc"
