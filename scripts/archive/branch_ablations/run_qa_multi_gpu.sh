#!/usr/bin/env bash
# run_qa_multi_gpu.sh: 4-GPU evaluation of QA Branch on Primary 7 Tasks
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TAG="${1:-qa_w1_primary7}"

echo "=== Launching QA Branch (v119) Evaluation across 4 GPUs on Primary 7 Tasks (TAG: $TAG) ==="

mkdir -p logs/official50

# GPU 0: LSCC ARID1A, Histologic Grade
bash scripts/eval_v119.sh 0 "$TAG" \
  cptac_lscc/ARID1A_mutation \
  cptac_lscc/Histologic_Grade > logs/qa_gpu0.log 2>&1 &
PID0=$!

# GPU 1: LSCC KEAP1, LUAD KRAS
bash scripts/eval_v119.sh 1 "$TAG" \
  cptac_lscc/KEAP1_mutation \
  cptac_luad/KRAS_mutation > logs/qa_gpu1.log 2>&1 &
PID1=$!

# GPU 2: PDA SMAD4, UCLA Lung Progression
bash scripts/eval_v119.sh 2 "$TAG" \
  cptac_pda/SMAD4_mutation \
  ucla_lung/progression_regression > logs/qa_gpu2.log 2>&1 &
PID2=$!

# GPU 3: CCRCC PBRM1
bash scripts/eval_v119.sh 3 "$TAG" \
  cptac_ccrcc/PBRM1_mutation > logs/qa_gpu3.log 2>&1 &
PID3=$!

echo "QA 4-GPU workers launched: GPU0=$PID0, GPU1=$PID1, GPU2=$PID2, GPU3=$PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "All QA 4-GPU workers completed successfully!"
