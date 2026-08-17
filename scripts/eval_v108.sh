#!/usr/bin/env bash
# Score the ACTIVE configuration, v108 (docs SS152). Zero learned parameters.
#
# v108 = v107 (within-slide PCA K=256 + constant 3-coefficient head) with the CT
# branch changed in two ways that only work TOGETHER (SS150):
#
#   ICF_CT_PCA_DIM=32    cell-token distances are measured in the leading 32 PCA
#                        directions instead of raw 1,536, where squared Euclidean
#                        concentrates (rel_std 0.229 -> 0.368, SS149-2)
#   ICF_CT_READOUT=ridge all 16 abundance dims go through a class-balanced ridge
#                        instead of reading two extreme coordinates
#
# Neither alone is worth having: singles are +0.0019 and +0.0008 over 17 tasks,
# the pair is +0.0037 and is the only CT variant positive in BOTH task groups.
#
# The basis is the one the CV branch already built for the fold, so there is no
# extra eigh -- and it is also why the gain is capped, since CT then lives inside
# a subspace CV already covers (SS149-4).
#
# CT weight stays at 0.286. SS151 swept it: the 17-task mean peaks near 0.5-0.7 but
# per-task agreement falls monotonically (11/17 at 0.286-0.4 down to 7/17 at 1.0),
# because the weight is a gain on a branch that is right on some tasks and wrong on
# others. Not promoted.
#
# Expected: SEAL 10-task macro 0.6967 (v107 gave 0.6945).
# Deterministic -- there is no seed to average (SS139).
#
# Usage: bash scripts/eval_v108.sh <gpu> <tag> [task]...   (default: the SEAL 10)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

CKPT="${ICF_V108_CKPT:-$(ls checkpoints/20260815_113422/v98_p1_reverse_seed42/periodic-epoch=049*.ckpt)}"
CONFIG="${ICF_V108_CONFIG:-configs/train_v98_p1_reverse_1536_1gpu.yaml}"
GPU="${1:?usage: eval_v108.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v108.sh <gpu> <tag> [task]...}"
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
echo "v108: pca_within K=256 + fixed head + CT(pca32, ridge)  (ckpt $(basename "$CKPT"))"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
