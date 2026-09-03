#!/usr/bin/env bash
# 5-GPU Parallel Orchestrator for DS Salience-Guided Anchor Subsampling across Primary 7
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

TAG="ds_salience_anchor_s5_f07_a15"
export ICF_DS_AUG_MODE=salience_anchor
export ICF_DS_ANCHOR_FRACTION=0.15
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo ">>> LAUNCHING DS SALIENCE ANCHOR SUBSAMPLING (alpha=0.15, S=5, f=0.7) on Primary 7 <<<"
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
echo ">>> FINISHED DS SALIENCE ANCHOR RUNS <<<"
