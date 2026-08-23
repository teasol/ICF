#!/usr/bin/env bash
# In-Context Fisher Discriminant Subspace (Option 3) runner for PathoBench.
#
# Usage: bash scripts/eval_fisher.sh <gpu> <tag> [task]...   (default: Primary 7 tasks)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V120_CKPT:-${ICF_V119_CKPT:-${ICF_V118_CKPT:-$ICF_CKPT}}}"
CONFIG="${ICF_V120_CONFIG:-${ICF_V119_CONFIG:-${ICF_V118_CONFIG:-$ICF_CONFIG}}}"
GPU="${1:?usage: eval_fisher.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_fisher.sh <gpu> <tag> [task]...}"
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

# Set In-Context Fisher Basis
export ICF_COVARIANCE_BASIS=fisher
export ICF_FISHER_SHRINKAGE="${ICF_FISHER_SHRINKAGE:-0.1}"

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

# Baseline 6-Branch weights (v120)
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
export ICF_FIXED_HEAD_LR_WEIGHT=0.0

export ICF_AGGREGATION="${ICF_AGGREGATION:-trimmed_mean}"
export ICF_KRR_KERNEL=linear

echo "Fisher Run: BASIS=$ICF_COVARIANCE_BASIS SHRINKAGE=$ICF_FISHER_SHRINKAGE AGG=$ICF_AGGREGATION"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
