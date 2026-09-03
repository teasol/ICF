#!/usr/bin/env bash
# 5-GPU Parallel Orchestrator for v121 5-Branch Ensemble with DS Salience-Guided Anchor Subsampling
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
. "$(dirname "${BASH_SOURCE[0]}")/lib/arms.sh"

TAG="v121_salience_anchor_s5_f07_a15"
CONFIG="configs/archive/v94_v102_cell_value/train_v98_p1_reverse_1536_1gpu.yaml"

icf_arm_v121
export ICF_DS_AUG_MODE="salience_anchor"
export ICF_DS_ANCHOR_FRACTION="0.15"
export ICF_DS_AUG_S="5"
export ICF_DS_AUG_FRACTION="0.70"
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo ">>> LAUNCHING v121 + DS SALIENCE ANCHOR SUBSAMPLING on Primary 7 (5 GPUs) <<<"
echo "================================================================================"

bash scripts/eval_seal_tasks.sh 0 "" "$CONFIG" "${TAG}" cptac_lscc/ARID1A_mutation ucla_lung/progression_regression &
pid0=$!
bash scripts/eval_seal_tasks.sh 1 "" "$CONFIG" "${TAG}" cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation &
pid1=$!
bash scripts/eval_seal_tasks.sh 2 "" "$CONFIG" "${TAG}" cptac_lscc/KEAP1_mutation &
pid2=$!
bash scripts/eval_seal_tasks.sh 3 "" "$CONFIG" "${TAG}" cptac_luad/KRAS_mutation &
pid3=$!
bash scripts/eval_seal_tasks.sh 4 "" "$CONFIG" "${TAG}" cptac_pda/SMAD4_mutation &
pid4=$!

wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4"
echo ">>> FINISHED v121 + DS SALIENCE ANCHOR RUNS <<<"
