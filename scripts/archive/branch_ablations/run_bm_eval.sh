#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
export ICF_FIXED_HEAD_BM_WEIGHT=1.0
export ICF_BM_DIM=32
export ICF_BM_LAMBDA=1.0
exec bash scripts/eval_v114.sh 1 bm_w1_primary7 \
  cptac_lscc/ARID1A_mutation \
  cptac_lscc/Histologic_Grade \
  cptac_lscc/KEAP1_mutation \
  cptac_luad/KRAS_mutation \
  cptac_pda/SMAD4_mutation \
  ucla_lung/progression_regression \
  cptac_ccrcc/PBRM1_mutation
