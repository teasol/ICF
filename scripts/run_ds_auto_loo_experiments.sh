#!/usr/bin/env bash
# Orchestrator for DS In-Episode LOO Dual Selection Experiments across Primary 7 (5 GPUs)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

TAG="ds_auto_loo_s5_f07"
export ICF_DS_AUG_MODE=auto_loo
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo ">>> LAUNCHING DS IN-EPISODE LOO DUAL SELECTION (S=5, fraction=0.7) on Primary 7 <<<"
echo "================================================================================"

bash scripts/eval_ds_aug.sh 0 "${TAG}" 5 0.7 cptac_lscc/ARID1A_mutation ucla_lung/progression_regression &
pid0=$!
bash scripts/eval_ds_aug.sh 1 "${TAG}" 5 0.7 cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation &
pid1=$!
bash scripts/eval_ds_aug.sh 2 "${TAG}" 5 0.7 cptac_lscc/KEAP1_mutation &
pid2=$!
bash scripts/eval_ds_aug.sh 3 "${TAG}" 5 0.7 cptac_luad/KRAS_mutation &
pid3=$!
bash scripts/eval_ds_aug.sh 4 "${TAG}" 5 0.7 cptac_pda/SMAD4_mutation &
pid4=$!

wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4"
echo ">>> FINISHED DS AUTO LOO RUNS <<<"

echo "================================================================================"
echo ">>> Summarizing results... <<<"
echo "================================================================================"
.venv/bin/python scripts/analysis/parse_ds_aug_results.py
