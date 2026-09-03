#!/usr/bin/env bash
# Orchestrator for DS Context Sub-bag Data Augmentation Experiments (Cases A, B, C)
# Usage: bash scripts/run_ds_aug_experiments.sh [case_a|case_b|case_c|all]
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

MODE="${1:-all}"

run_case() {
  local tag="$1"
  local s="$2"
  local frac="$3"
  echo "================================================================================"
  echo ">>> LAUNCHING CASE: ${tag} (S=${s}, fraction=${frac}) on Primary 7 (5 GPUs) <<<"
  echo "================================================================================"

  bash scripts/eval_ds_aug.sh 0 "${tag}" "$s" "$frac" cptac_lscc/ARID1A_mutation ucla_lung/progression_regression &
  local pid0=$!
  bash scripts/eval_ds_aug.sh 1 "${tag}" "$s" "$frac" cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation &
  local pid1=$!
  bash scripts/eval_ds_aug.sh 2 "${tag}" "$s" "$frac" cptac_lscc/KEAP1_mutation &
  local pid2=$!
  bash scripts/eval_ds_aug.sh 3 "${tag}" "$s" "$frac" cptac_luad/KRAS_mutation &
  local pid3=$!
  bash scripts/eval_ds_aug.sh 4 "${tag}" "$s" "$frac" cptac_pda/SMAD4_mutation &
  local pid4=$!

  wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4"
  echo ">>> FINISHED CASE: ${tag} <<<"
}

if [ "$MODE" = "case_a" ] || [ "$MODE" = "all" ]; then
  run_case "ds_aug_s5_f07" 5 0.7
fi

if [ "$MODE" = "case_b" ] || [ "$MODE" = "all" ]; then
  run_case "ds_aug_s10_f05" 10 0.5
fi

if [ "$MODE" = "case_c" ] || [ "$MODE" = "all" ]; then
  run_case "ds_aug_s100_f02" 100 0.2
fi

if [ "$MODE" = "query_tta" ] || [ "$MODE" = "all_query" ]; then
  export ICF_DS_AUG_MODE=query
  run_case "ds_query_tta_s5_f07" 5 0.7
fi

echo "================================================================================"
echo ">>> All requested augmentation runs complete. Summarizing results... <<<"
echo "================================================================================"
.venv/bin/python scripts/analysis/parse_ds_aug_results.py
