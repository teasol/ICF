#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

mkdir -p logs

TAG="${1:-v116_branch_logits}"

echo "=== Launching v116 Baseline Evaluation with 5-Branch Logit Recording (TAG: $TAG) across 4 GPUs ==="

# GPU 0: LSCC ARID1A & Histologic Grade (2 tasks)
bash scripts/eval_v116.sh 0 "$TAG" \
  cptac_lscc/ARID1A_mutation \
  cptac_lscc/Histologic_Grade > logs/v116_logits_gpu0.log 2>&1 &
PID0=$!

# GPU 1: LSCC KEAP1 & LUAD KRAS (2 tasks)
bash scripts/eval_v116.sh 1 "$TAG" \
  cptac_lscc/KEAP1_mutation \
  cptac_luad/KRAS_mutation > logs/v116_logits_gpu1.log 2>&1 &
PID1=$!

# GPU 2: PDA SMAD4 & UCLA Lung (2 tasks)
bash scripts/eval_v116.sh 2 "$TAG" \
  cptac_pda/SMAD4_mutation \
  ucla_lung/progression_regression > logs/v116_logits_gpu2.log 2>&1 &
PID2=$!

# GPU 3: CCRCC PBRM1 (1 task)
bash scripts/eval_v116.sh 3 "$TAG" \
  cptac_ccrcc/PBRM1_mutation > logs/v116_logits_gpu3.log 2>&1 &
PID3=$!

echo "v116 multi-GPU workers launched: GPU0=$PID0, GPU1=$PID1, GPU2=$PID2, GPU3=$PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "All v116 multi-GPU workers finished successfully!"
