#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

mkdir -p logs/official50

TAG="${1:-ds_w1_primary7}"

echo "=== Launching DS Branch Evaluation across 4 GPUs on Primary 7 Tasks (TAG: $TAG) ==="

# GPU 0: cptac_lscc/ARID1A_mutation & cptac_lscc/Histologic_Grade
bash scripts/eval_ds.sh 0 "$TAG" \
  cptac_lscc/ARID1A_mutation \
  cptac_lscc/Histologic_Grade > logs/ds_p7_gpu0.log 2>&1 &
PID0=$!

# GPU 1: cptac_lscc/KEAP1_mutation & cptac_luad/KRAS_mutation
bash scripts/eval_ds.sh 1 "$TAG" \
  cptac_lscc/KEAP1_mutation \
  cptac_luad/KRAS_mutation > logs/ds_p7_gpu1.log 2>&1 &
PID1=$!

# GPU 2: cptac_pda/SMAD4_mutation & ucla_lung/progression_regression
bash scripts/eval_ds.sh 2 "$TAG" \
  cptac_pda/SMAD4_mutation \
  ucla_lung/progression_regression > logs/ds_p7_gpu2.log 2>&1 &
PID2=$!

# GPU 3: cptac_ccrcc/PBRM1_mutation
bash scripts/eval_ds.sh 3 "$TAG" \
  cptac_ccrcc/PBRM1_mutation > logs/ds_p7_gpu3.log 2>&1 &
PID3=$!

echo "DS multi-GPU workers launched: GPU0=$PID0, GPU1=$PID1, GPU2=$PID2, GPU3=$PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "All DS multi-GPU workers finished successfully!"
