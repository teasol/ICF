#!/usr/bin/env bash
# Evaluate the SS182 DD candidate on top of the deterministic v111 baseline.
# The baseline stays reproducible: this runner alone enables the new readout.
#
# Usage: bash scripts/eval_dd_ordered_typicality.sh <gpu> <tag> [task]...
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V111_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V111_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_dd_ordered_typicality.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_dd_ordered_typicality.sh <gpu> <tag> [task]...}"
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
# ordered_typicality's margin is bounded to [-1,1] by construction, not the old
# distance-based DD's magnitude that 0.343 was fitted against (SS151/153) --
# reusing that constant here is unjustified, so take DD at unit weight instead.
export ICF_FIXED_HEAD_DD_WEIGHT=1
export ICF_CV_BLOCKS=offdiag
export ICF_DD_ORDERED_TYPICALITY=1
export ICF_DD_SEPARATION_FLOOR="${ICF_DD_SEPARATION_FLOOR:-1.0}"
echo "DD arm: v111 + ordered*typicality floor=${ICF_DD_SEPARATION_FLOOR}"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
