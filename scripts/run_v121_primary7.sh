#!/usr/bin/env bash
# 5-GPU Parallel Orchestrator for v121 5-Branch Fast Baseline (Mute CT) across Primary 7
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

TAG="${1:-v121_baseline}"

echo "================================================================================"
echo ">>> LAUNCHING v121 5-BRANCH FAST BASELINE (CT=OFF) on Primary 7 (5 GPUs) <<<"
echo "================================================================================"

bash scripts/eval_v121.sh 0 "${TAG}" cptac_lscc/ARID1A_mutation ucla_lung/progression_regression &
pid0=$!
bash scripts/eval_v121.sh 1 "${TAG}" cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation &
pid1=$!
bash scripts/eval_v121.sh 2 "${TAG}" cptac_lscc/KEAP1_mutation &
pid2=$!
bash scripts/eval_v121.sh 3 "${TAG}" cptac_luad/KRAS_mutation &
pid3=$!
bash scripts/eval_v121.sh 4 "${TAG}" cptac_pda/SMAD4_mutation &
pid4=$!

wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4"
echo ">>> FINISHED v121 RUNS <<<"
