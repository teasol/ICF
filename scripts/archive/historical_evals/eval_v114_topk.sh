#!/usr/bin/env bash
# v114 + top-k mean abundance pooling (SS189). Deterministic, 0-param diagnostic
# over the CELL dimension (orthogonal to G0's kernel readout over the token
# dimension): instead of the mean abundance, each token keeps the average of its
# most similar `fraction` of cells (floor `min`). Everything else is bit-identical
# to v114; the CT readout stays `ridge`.
#
# Usage: bash scripts/eval_v114_topk.sh <gpu> <tag> [task]...
#   ICF_CT_ABUNDANCE_TOPK_FRACTION  fraction of cells per token (default 0.1)
#   ICF_CT_ABUNDANCE_TOPK_MIN       floor cell count per token  (default 1)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V114_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V114_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_v114_topk.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v114_topk.sh <gpu> <tag> [task]...}"
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
export ICF_CT_ABUNDANCE_POOLING=topk
export ICF_CT_ABUNDANCE_TOPK_FRACTION="${ICF_CT_ABUNDANCE_TOPK_FRACTION:-0.1}"
export ICF_CT_ABUNDANCE_TOPK_MIN="${ICF_CT_ABUNDANCE_TOPK_MIN:-1}"
export ICF_CV_BLOCKS=offdiag
export ICF_DD_ORDERED_TYPICALITY=1
export ICF_DD_SEPARATION_FLOOR=1.0
export ICF_FIXED_HEAD_CV_WEIGHT=1.0
export ICF_FIXED_HEAD_DD_WEIGHT=1.0
export ICF_FIXED_HEAD_CT_WEIGHT=1.0
echo "v114+topk: pooling=topk fraction=${ICF_CT_ABUNDANCE_TOPK_FRACTION} min=${ICF_CT_ABUNDANCE_TOPK_MIN} (rest = v114)"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
