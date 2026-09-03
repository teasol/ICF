#!/usr/bin/env bash
# Direct In-Context Patch Likelihood Ratio + Top-K MIL Extreme Pooling (LR) evaluation runner.
#
# Usage: bash scripts/eval_lr.sh <gpu> <tag> [task]...   (default: Primary 7 tasks)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V120_CKPT:-${ICF_V119_CKPT:-${ICF_V118_CKPT:-$ICF_CKPT}}}"
CONFIG="${ICF_V120_CONFIG:-${ICF_V119_CONFIG:-${ICF_V118_CONFIG:-$ICF_CONFIG}}}"
GPU="${1:?usage: eval_lr.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_lr.sh <gpu> <tag> [task]...}"
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

export ICF_COVARIANCE_BASIS=pca_within
export ICF_FIXED_HEAD=1
export ICF_SKETCH_DIM=256
export ICF_CT_PCA_DIM=32
export ICF_CT_READOUT=ridge
export ICF_CT_CELLS="${ICF_CT_CELLS:-0.125}"
export ICF_CT_CELLS_SCALE="${ICF_CT_CELLS_SCALE:-own}"
export ICF_CT_CELLS_MIN="${ICF_CT_CELLS_MIN:-64}"
export ICF_CT_ABUNDANCE_CELLS="${ICF_CT_ABUNDANCE_CELLS:-match}"
export ICF_CT_SAMPLING=random
export ICF_CT_SAMPLING_SEED="${ICF_CT_SAMPLING_SEED:-0}"
export ICF_CT_TOKENS=256
export ICF_CT_TOKENIZER=kmeans_plusplus
export ICF_CT_KMEANS_MAX_ITER="${ICF_CT_KMEANS_MAX_ITER:-8}"
export ICF_CT_DISTANCE_KERNEL=gemm
export ICF_CV_BLOCKS=offdiag

# Baseline branch weights (v120)
export ICF_FIXED_HEAD_CV_WEIGHT="${ICF_FIXED_HEAD_CV_WEIGHT:-1.0}"
export ICF_FIXED_HEAD_DD_WEIGHT=0.0
export ICF_FIXED_HEAD_CT_WEIGHT="${ICF_FIXED_HEAD_CT_WEIGHT:-1.0}"
export ICF_FIXED_HEAD_BM_WEIGHT="${ICF_FIXED_HEAD_BM_WEIGHT:-1.0}"
export ICF_BM_DIM=32
export ICF_BM_LAMBDA=1.0
export ICF_FIXED_HEAD_BD_WEIGHT="${ICF_FIXED_HEAD_BD_WEIGHT:-1.0}"
export ICF_BD_DIM=256
export ICF_BD_METRIC=entropy
export ICF_BD_READOUT=ordered_typicality
export ICF_BD_SEPARATION_FLOOR=1.0
export ICF_FIXED_HEAD_QA_WEIGHT="${ICF_FIXED_HEAD_QA_WEIGHT:-1.0}"
export ICF_QA_DIM=32
export ICF_QA_LAMBDA=1.0
export ICF_FIXED_HEAD_DS_WEIGHT="${ICF_FIXED_HEAD_DS_WEIGHT:-1.0}"
export ICF_DS_DIM=32
export ICF_DS_LAMBDA=1.0
export ICF_DS_TEMPERATURE=1.0
export ICF_DS_TOKENS=256

# LR Branch settings
export ICF_FIXED_HEAD_LR_WEIGHT="${ICF_FIXED_HEAD_LR_WEIGHT:-1.0}"
export ICF_LR_DIM="${ICF_LR_DIM:-32}"
export ICF_LR_LAMBDA="${ICF_LR_LAMBDA:-1.0}"
export ICF_LR_TAU="${ICF_LR_TAU:-5.0}"
export ICF_LR_TOPK_FRACTION="${ICF_LR_TOPK_FRACTION:-0.05}"
export ICF_LR_TOPK_MIN="${ICF_LR_TOPK_MIN:-4}"
export ICF_LR_TOPK_MAX="${ICF_LR_TOPK_MAX:-64}"
export ICF_LR_PATCHES_PER_CTX="${ICF_LR_PATCHES_PER_CTX:-64}"

export ICF_AGGREGATION="${ICF_AGGREGATION:-trimmed_mean}"
export ICF_KRR_KERNEL=linear

echo "LR Run: CV=$ICF_FIXED_HEAD_CV_WEIGHT CT=$ICF_FIXED_HEAD_CT_WEIGHT BM=$ICF_FIXED_HEAD_BM_WEIGHT BD=$ICF_FIXED_HEAD_BD_WEIGHT QA=$ICF_FIXED_HEAD_QA_WEIGHT DS=$ICF_FIXED_HEAD_DS_WEIGHT LR=$ICF_FIXED_HEAD_LR_WEIGHT AGG=$ICF_AGGREGATION"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
