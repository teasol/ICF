#!/usr/bin/env bash
# v114 + G0 kernel-ridge CT readout (SS188). Deterministic, 0-param diagnostic:
# does the abundance -> label relation have non-linear structure the linear
# ridge misses? Everything else is bit-identical to v114 (eval_v114.sh); only
# the CT readout changes from `ridge` to `kernel_ridge` with the chosen kernel.
#
# Usage: bash scripts/eval_v114_kernel.sh <gpu> <tag> <kernel> [task]...
#   kernel in {linear, rbf, poly}
#     linear = control, reproduces v114's primal ridge exactly.
#   ICF_CT_KERNEL_GAMMA / ICF_CT_KERNEL_DEGREE / ICF_CT_KERNEL_COEF0 override the
#   kernel hyperparameters (defaults: gamma=1/dims, degree=2, coef0=1.0).
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V114_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V114_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_v114_kernel.sh <gpu> <tag> <kernel> [task]...}"
TAG="${2:?usage: eval_v114_kernel.sh <gpu> <tag> <kernel> [task]...}"
KERNEL="${3:?usage: eval_v114_kernel.sh <gpu> <tag> <kernel> [task]...}"
shift 3

case "$KERNEL" in
  linear|rbf|poly) ;;
  *) echo "kernel must be linear|rbf|poly, got '$KERNEL'" >&2; exit 2 ;;
esac

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
export ICF_CT_READOUT=kernel_ridge
export ICF_CT_KERNEL="$KERNEL"
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
echo "v114+G0 kernel_ridge: kernel=${KERNEL} gamma=${ICF_CT_KERNEL_GAMMA:-auto} degree=${ICF_CT_KERNEL_DEGREE:-2} coef0=${ICF_CT_KERNEL_COEF0:-1.0} (rest = v114)"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
