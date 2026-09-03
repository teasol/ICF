#!/usr/bin/env bash
# Orchestrator for v120 6-Branch Context LOO Stacking (NO subsampling) across Primary 7 (5 GPUs)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
. "$(dirname "${BASH_SOURCE[0]}")/lib/arms.sh"

TAG="v120_clean_context_loo"
CONFIG="configs/archive/v94_v102_cell_value/train_v98_p1_reverse_1536_1gpu.yaml"

icf_arm_v120
export ICF_AGGREGATION="context_loo"
export ICF_DS_AUG_MODE="none"
export PYTHONUNBUFFERED=1

echo "================================================================================"
echo ">>> LAUNCHING v120 CLEAN CONTEXT LOO STACKING (NO SUBSAMPLING) on Primary 7 <<<"
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
echo ">>> FINISHED v120 CLEAN CONTEXT LOO RUNS <<<"
