#!/usr/bin/env bash
# Score the ACTIVE configuration, v107 (docs SS139, SS142). Zero learned parameters.
#
# v107 = v106 (within-slide PCA projection + constant 3-coefficient head) with
# K = 256 instead of 128. It is defined entirely by three environment variables
# on top of the ordinary SEAL runner, and this script exists so that definition
# lives in exactly one place instead of being retyped per session:
#
#   ICF_COVARIANCE_BASIS=pca_within   P := top-K eigenvectors of the fold's
#                                     context cells, each bag centred on its own
#                                     mean (drops the between-slide term, SS139-4)
#   ICF_FIXED_HEAD=1                  head := 1.442*(CV1-CV0) - 0.343*(D1-D0)
#                                             + 0.286*(q1-q0)   (SS137-3)
#   ICF_SKETCH_DIM=256                K, free to set only because the projection
#                                     is no longer learned (SS142)
#
# The checkpoint is still passed because the runner builds the model from it,
# but NOTHING learned survives the three overrides: P and the head are replaced,
# and the encoder does not enter the margin. Any v98 seed gives the same number.
# There is no seed to average: SS139 measured seed std 0.00000.
#
# Expected: SEAL 10-task macro 0.6945 (K=128 gave 0.6864).
#
# Usage: bash scripts/eval_v107.sh <gpu> <tag> [task]...   (default: the SEAL 10)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"   # docs SS164
CKPT="${ICF_V107_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V107_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_v107.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v107.sh <gpu> <tag> [task]...}"
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
echo "v107: pca_within + fixed head + K=$ICF_SKETCH_DIM  (ckpt $(basename "$CKPT"))"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
