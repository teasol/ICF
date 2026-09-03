#!/usr/bin/env bash
# Standalone evaluation runner for CT (Cell-Type Abundance) branch alone.
# Usage: bash scripts/eval_ct_alone.sh <gpu> <tag> [task]...
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
. "$(dirname "${BASH_SOURCE[0]}")/lib/arms.sh"

GPU="${1:?usage: eval_ct_alone.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_ct_alone.sh <gpu> <tag> [task]...}"
shift 2

if [ "$#" -eq 0 ]; then
  set -- cptac_lscc/ARID1A_mutation \
    cptac_lscc/Histologic_Grade \
    cptac_lscc/KEAP1_mutation \
    cptac_luad/KRAS_mutation \
    cptac_pda/SMAD4_mutation \
    ucla_lung/progression_regression \
    cptac_ccrcc/PBRM1_mutation
fi

icf_arm_v120

export ICF_FIXED_HEAD_CV_WEIGHT=0.0
export ICF_FIXED_HEAD_DD_WEIGHT=0.0
export ICF_FIXED_HEAD_CT_WEIGHT=1.0
export ICF_FIXED_HEAD_BM_WEIGHT=0.0
export ICF_FIXED_HEAD_BD_WEIGHT=0.0
export ICF_FIXED_HEAD_QA_WEIGHT=0.0
export ICF_FIXED_HEAD_DS_WEIGHT=0.0
export ICF_AGGREGATION=soft_voting

CKPT="${ICF_V120_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V120_CONFIG:-$ICF_CONFIG}"

echo "CT Alone Standalone Runner: CT=1.0, all other branches=0.0, AGG=soft_voting"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
