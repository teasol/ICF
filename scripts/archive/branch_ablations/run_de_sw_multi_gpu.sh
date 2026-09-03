#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

mkdir -p logs/official50

MODE="${1:?usage: run_de_sw_multi_gpu.sh <mode> <tag>}"
TAG="${2:?usage: run_de_sw_multi_gpu.sh <mode> <tag>}"

echo "=== Launching $MODE Evaluation across 4 GPUs on Primary 7 Tasks (TAG: $TAG) ==="

# GPU 0: cptac_lscc/ARID1A_mutation & cptac_lscc/Histologic_Grade
bash scripts/eval_de_sw.sh "$MODE" 0 "$TAG" \
  cptac_lscc/ARID1A_mutation \
  cptac_lscc/Histologic_Grade > "logs/${TAG}_gpu0.log" 2>&1 &
PID0=$!

# GPU 1: cptac_lscc/KEAP1_mutation & cptac_luad/KRAS_mutation
bash scripts/eval_de_sw.sh "$MODE" 1 "$TAG" \
  cptac_lscc/KEAP1_mutation \
  cptac_luad/KRAS_mutation > "logs/${TAG}_gpu1.log" 2>&1 &
PID1=$!

# GPU 2: cptac_pda/SMAD4_mutation & ucla_lung/progression_regression
bash scripts/eval_de_sw.sh "$MODE" 2 "$TAG" \
  cptac_pda/SMAD4_mutation \
  ucla_lung/progression_regression > "logs/${TAG}_gpu2.log" 2>&1 &
PID2=$!

# GPU 3: cptac_ccrcc/PBRM1_mutation
bash scripts/eval_de_sw.sh "$MODE" 3 "$TAG" \
  cptac_ccrcc/PBRM1_mutation > "logs/${TAG}_gpu3.log" 2>&1 &
PID3=$!

echo "Workers launched: GPU0=$PID0, GPU1=$PID1, GPU2=$PID2, GPU3=$PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "All multi-GPU workers finished successfully for TAG: $TAG!"
