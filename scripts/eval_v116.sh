#!/usr/bin/env bash
# v116 = v115 (CV=1.0, DD=1.0, CT=1.0, BM=1.0) + BD (Bag Dispersion Spectral Entropy, w_BD=1.0).
#
# Primary 7-Task Benchmark:
#   v114 Baseline: 0.6051
#   v115 Baseline: 0.6094 (+0.0043 vs v114)
#   v116 Baseline (BD w=1.0): 0.6119 (+0.0025 vs v115, +0.0068 vs v114)
#
# Usage: bash scripts/eval_v116.sh <gpu> <tag> [task]...   (default: Primary 7 tasks)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V116_CKPT:-${ICF_V115_CKPT:-${ICF_V114_CKPT:-$ICF_CKPT}}}"
CONFIG="${ICF_V116_CONFIG:-${ICF_V115_CONFIG:-${ICF_V114_CONFIG:-$ICF_CONFIG}}}"
GPU="${1:?usage: eval_v116.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v116.sh <gpu> <tag> [task]...}"
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
export ICF_DD_ORDERED_TYPICALITY=1
export ICF_DD_SEPARATION_FLOOR=1.0
export ICF_FIXED_HEAD_CV_WEIGHT=1.0
export ICF_FIXED_HEAD_DD_WEIGHT=1.0
export ICF_FIXED_HEAD_CT_WEIGHT=1.0
export ICF_FIXED_HEAD_BM_WEIGHT=1.0
export ICF_BM_DIM=32
export ICF_BM_LAMBDA=1.0
export ICF_FIXED_HEAD_BD_WEIGHT=1.0
export ICF_BD_DIM=256
export ICF_BD_METRIC=entropy
export ICF_BD_READOUT=ordered_typicality
export ICF_BD_SEPARATION_FLOOR=1.0

echo "v116: CV=offdiag w=1.0 DD=ordered_typicality(kappa=1) w=1.0 CT=pca32/random-frac=${ICF_CT_CELLS}/${ICF_CT_CELLS_SCALE}/min=${ICF_CT_CELLS_MIN}/kmeans++/K256/match-abundance/ridge w=1.0 BM=dim32/lambda=1.0 w=1.0 BD=entropy/dim256/ordered_typicality w=1.0"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
