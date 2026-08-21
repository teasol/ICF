#!/usr/bin/env bash
# Evaluate the SS182 DD candidate on all 17 official tasks, one worker per GPU.
# Deterministic arm: no seed repetition is needed (docs SS139, SS151, SS182).
#
# Usage: bash scripts/run_dd_ordered_typicality_eval.sh <out_dir> [tag]
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"

OUT="${1:?usage: run_dd_ordered_typicality_eval.sh <out_dir> [tag]}"
TAG="${2:-dd_ordered_typicality_k1}"
mkdir -p "$OUT"

TASKS=(
  bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status
  cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation
  cptac_luad/EGFR_mutation cptac_luad/STK11_mutation cptac_luad/TP53_mutation
  cptac_ccrcc/BAP1_mutation cptac_ccrcc/VHL_mutation
  cptac_lscc/ARID1A_mutation cptac_lscc/Histologic_Grade cptac_lscc/KEAP1_mutation
  cptac_luad/KRAS_mutation cptac_pda/SMAD4_mutation
  ucla_lung/progression_regression cptac_ccrcc/PBRM1_mutation
)

echo "${#TASKS[@]} deterministic jobs over $NGPU GPUs; tag=$TAG"
for ((worker=0; worker<NGPU; worker++)); do
  gpu="$((worker + GPU_OFFSET))"
  (
    for ((index=worker; index<${#TASKS[@]}; index+=NGPU)); do
      task="${TASKS[$index]}"
      name="${task//\//_}"
      task_log="$OUT/${name}.runner.log"
      official_log="logs/official50/${name}_${TAG}.log"
      if grep -aq 'fold-mean AUROC:' "$official_log" 2>/dev/null; then
        echo "SKIP gpu=$gpu task=$task (complete)"
        continue
      fi
      echo "START gpu=$gpu task=$task $(date --iso-8601=seconds)"
      bash scripts/eval_dd_ordered_typicality.sh \
        "$gpu" "$TAG" "$task" > "$task_log" 2>&1
      rc=$?
      result="$(grep -ao 'fold-mean AUROC: [0-9.]*' "$official_log" 2>/dev/null | tail -1)"
      echo "END gpu=$gpu task=$task rc=$rc ${result:-FAILED} $(date --iso-8601=seconds)"
    done
  ) &
done
wait
echo "EVALUATION DONE $(date --iso-8601=seconds)"
