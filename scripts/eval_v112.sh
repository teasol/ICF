#!/usr/bin/env bash
# Score the active zero-parameter v112 baseline (docs SS183).
# Identical to v111's CT/CV configuration; the only change is DD's readout --
# bounded ordered-coordinate x nearest-class typicality (kappa=1) at unit
# fixed-head weight, replacing the unbounded distance readout at weight 0.343.
#
# Usage: bash scripts/eval_v112.sh <gpu> <tag> [task]...   (default: SEAL 10)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V112_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V112_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_v112.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v112.sh <gpu> <tag> [task]...}"
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
export ICF_CT_CELLS=all
export ICF_CT_ABUNDANCE_CELLS=all
export ICF_CT_SAMPLING=even
export ICF_CT_TOKENS=256
export ICF_CT_TOKENIZER=hierarchical_2means
export ICF_CT_DISTANCE_KERNEL=gemm
export ICF_CT_TREE_REDUCTION=segment
export ICF_FIXED_HEAD_CT_WEIGHT=0.7
export ICF_CV_BLOCKS=offdiag
export ICF_DD_ORDERED_TYPICALITY=1
export ICF_DD_SEPARATION_FLOOR=1.0
export ICF_FIXED_HEAD_DD_WEIGHT=1
echo "v112: CV=offdiag DD=ordered_typicality(kappa=1,w=1) CT=pca32/full-cell/hierarchical-K256/full-abundance/ridge w=0.7"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
