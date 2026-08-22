#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

export ICF_FIXED_HEAD_BM_WEIGHT=1.0
export ICF_BM_DIM=32
export ICF_BM_LAMBDA=1.0

# GPU 1: LSCC 3 tasks
bash scripts/eval_v114.sh 1 bm_w1_primary7 \
  cptac_lscc/ARID1A_mutation \
  cptac_lscc/Histologic_Grade \
  cptac_lscc/KEAP1_mutation > logs/bm_gpu1.log 2>&1 &
PID1=$!

# GPU 2: LUAD & PDA (2 tasks)
bash scripts/eval_v114.sh 2 bm_w1_primary7 \
  cptac_luad/KRAS_mutation \
  cptac_pda/SMAD4_mutation > logs/bm_gpu2.log 2>&1 &
PID2=$!

# GPU 3: UCLA & CCRCC (2 tasks)
bash scripts/eval_v114.sh 3 bm_w1_primary7 \
  cptac_ccrcc/PBRM1_mutation \
  ucla_lung/progression_regression > logs/bm_gpu3.log 2>&1 &
PID3=$!

echo "BM multi-GPU workers launched: GPU1=$PID1, GPU2=$PID2, GPU3=$PID3"
wait $PID1 $PID2 $PID3
echo "All BM multi-GPU workers finished successfully!"
