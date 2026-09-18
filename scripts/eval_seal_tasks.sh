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
  # Do not substitute silently. Until 2026-09-18 this block replaced any config
  # carrying a `model:` block with the official 7-branch config. Every arm runner
  # passed a *training* config, so all of them -- eval_v120, eval_v121,
  # eval_ct_alone, eval_ds_aug, run_v120_clean_loo_experiments,
  # run_v121_salience_anchor -- silently converged on the same 7-branch setup
  # while printing banners claiming otherwise. Arm comparisons built on those
  # runs compared identical configurations. See D-053.
  cfg="${CONFIG:-}"
  if [ -z "$cfg" ]; then
    echo "ERROR: no config given. Pass an explicit evaluation config." >&2
    echo "       (the implicit 7-branch default was removed -- D-053)" >&2
    exit 3
  fi
  if [ ! -f "$cfg" ]; then
    echo "ERROR: config not found: $cfg" >&2; exit 3
  fi
  if grep -q "^model:" "$cfg" 2>/dev/null; then
    echo "ERROR: $cfg is a training config (has a 'model:' block), not an" >&2
    echo "       evaluation config. Pass configs/baseline/*.yaml instead." >&2
    exit 3
  fi
  # effective config, recorded per task: requested == effective by construction now
  echo "=== CONFIG ${cfg}  sha256=$(sha256sum "$cfg" | cut -c1-16)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/evaluate_pure.py \
    --config "$cfg" \
    --official-folds "$OFFICIAL/$task" --features "$FEATURES" \
    --output "$out" > "$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then overall_rc="$rc"; fi
  res=$(grep -aoE "fold-mean AUROC: [0-9.]+ ± [0-9.]+   pooled AUROC: [0-9.]+" "$log" | tail -1)
  echo "=== END   ${task} rc=$rc $(date +%H:%M:%S)  ${res:-$(tail -2 "$log")}"
done
exit "$overall_rc"
