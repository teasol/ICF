#!/usr/bin/env bash
# Score the active zero-parameter v111 baseline (docs SS181).
# CT uses every cell for both its hierarchical PCA/2-means dictionary and final
# abundance, eliminating storage-order selection bias and sampling randomness.
#
# Usage: bash scripts/eval_v111.sh <gpu> <tag> [task]...   (default: SEAL 10)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V111_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V111_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_v111.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v111.sh <gpu> <tag> [task]...}"
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
echo "v111: CV=offdiag DD=full CT=pca32/full-cell/hierarchical-K256/full-abundance/ridge w=0.7"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
