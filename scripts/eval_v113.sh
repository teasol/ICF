#!/usr/bin/env bash
# Score the v113 CT redesign on top of the v112 head/DD/CV stack.
#
# v112 kept full-cell hierarchical 2-means (SS181). That path OOMs on A5000
# LUAD because it cats every tile. The rejected historical alternative was a
# FIXED random-512 draw, which ignored bag size and therefore mixed dictionary
# geometry with an arbitrary abundance sample (SS165-167).
#
# v113 keeps the same cells *proportion* of each bag (default 1/8 of that
# bag's own n, floor 64) and rebuilds tokens with seeded k-means++ + Lloyd.
# Abundance matches the dictionary sample. Override the budget with
# ICF_CT_CELLS=0.125 | own:0.125 | median:0.125 | 64 | all.
#
# Usage: bash scripts/eval_v113.sh <gpu> <tag> [task]...   (default: SEAL 10)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V113_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V113_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_v113.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v113.sh <gpu> <tag> [task]...}"
shift 2

if [ "$#" -eq 0 ]; then
  set -- bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status \
    cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation \
    cptac_luad/EGFR_mutation cptac_luad/STK11_mutation cptac_luad/TP53_mutation \
    cptac_ccrcc/BAP1_mutation cptac_ccrcc/VHL_mutation
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
export ICF_FIXED_HEAD_CT_WEIGHT=0.7
export ICF_CV_BLOCKS=offdiag
export ICF_DD_ORDERED_TYPICALITY=1
export ICF_DD_SEPARATION_FLOOR=1.0
export ICF_FIXED_HEAD_DD_WEIGHT=1
echo "v113: CV=offdiag DD=ordered_typicality(kappa=1,w=1) CT=pca32/random-frac=${ICF_CT_CELLS}/${ICF_CT_CELLS_SCALE}/min=${ICF_CT_CELLS_MIN}/kmeans++/K256/match-abundance/ridge w=0.7"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
