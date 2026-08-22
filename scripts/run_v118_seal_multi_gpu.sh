#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

mkdir -p logs/official50

TAG="${1:-v118_seal10}"

echo "=== Launching v118 Baseline Evaluation across 4 GPUs on SEAL 10 Tasks (TAG: $TAG) ==="

# GPU 0: BC Therapy (3 tasks: ER, Grade, HER2)
bash scripts/eval_v118.sh 0 "$TAG" \
  bc_therapy/er_status \
  bc_therapy/grade \
  bc_therapy/her2_status > logs/v118_seal_gpu0.log 2>&1 &
PID0=$!

# GPU 1: CPTAC BRCA (2 tasks: PIK3CA, TP53)
bash scripts/eval_v118.sh 1 "$TAG" \
  cptac_brca/PIK3CA_mutation \
  cptac_brca/TP53_mutation > logs/v118_seal_gpu1.log 2>&1 &
PID1=$!

# GPU 2: CPTAC LUAD (3 tasks: EGFR, STK11, TP53)
bash scripts/eval_v118.sh 2 "$TAG" \
  cptac_luad/EGFR_mutation \
  cptac_luad/STK11_mutation \
  cptac_luad/TP53_mutation > logs/v118_seal_gpu2.log 2>&1 &
PID2=$!

# GPU 3: CPTAC CCRCC (2 tasks: BAP1, VHL)
bash scripts/eval_v118.sh 3 "$TAG" \
  cptac_ccrcc/BAP1_mutation \
  cptac_ccrcc/VHL_mutation > logs/v118_seal_gpu3.log 2>&1 &
PID3=$!

echo "v118 SEAL 10 multi-GPU workers launched: GPU0=$PID0, GPU1=$PID1, GPU2=$PID2, GPU3=$PID3"
wait $PID0 $PID1 $PID2 $PID3
echo "All v118 SEAL 10 multi-GPU workers finished successfully!"
