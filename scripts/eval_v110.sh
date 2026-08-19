#!/usr/bin/env bash
# Historical v110 replay. The active baseline is v111 (docs SS181).
#
# v110 = v109 with CT's cluster count raised 16 -> 32 (SS161). v109 itself was v108
# plus two changes, one per branch:
#
#   CT: ICF_CT_KMEANS=30 + ICF_FIXED_HEAD_CT_WEIGHT=0.7
#       Farthest-point sampling was using ~1.9 of its 16 tokens -- it optimises
#       "as far apart as possible" and so lands on outlier cells no cell is
#       nearest to (SS157-2). 30 Lloyd iterations from those same points bring the
#       effective token count to ~13 and lift CT-only by +0.037. The branch then
#       earns more of the head: on FPS tokens raising the weight collapsed sign
#       agreement 11/17 -> 7/17, on k-means tokens it HOLDS at 11/17 (SS157-5).
#
#   CV: ICF_CV_BLOCKS=offdiag
#       Drops the 1,536-d raw bag mean (which adds nothing -- removing it is
#       +0.0019) and the 256 diagonal entries (actively harmful -- removing them is
#       +0.0052 at 13/17). CV's performance comes from the CORRELATION structure
#       between PCA directions, not the per-direction spectrum (SS156).
#       ⚠️ DD still receives the FULL triangle; the mask reaches only the CV ridge.
#
# The two live in different branches and measured additive: +0.0070 and +0.0030
# alone, +0.0096 together at 13/17 against a predicted 0.6624 vs measured 0.6621.
#
# ⚠️ 32 clusters, NOT more cells. The full-cell variant was measured at 16, 32, 64
# and 128 tokens and lost in every row (-0.0015 to -0.0031), so the 64-cell sample
# stays: k-means centres sit in dense regions, whose shares 64 cells already
# estimate well, while extra cells mainly add clusters on rare populations with
# noisy per-bag shares (SS159, SS160-3).
#
# Expected: SEAL 10-task macro 0.7070 (v109 gave 0.7027). Held-out 7: 0.6103.
# Deterministic -- no seed to average (SS139). Report sign agreement, never t
# (SS151-1).
#
# Usage: bash scripts/eval_v110.sh <gpu> <tag> [task]...   (default: the SEAL 10)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"   # docs SS164
CKPT="${ICF_V110_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V110_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_v110.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v110.sh <gpu> <tag> [task]...}"
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
export ICF_CT_KMEANS=30
export ICF_CT_CELLS=64
export ICF_CT_SAMPLING=even
export ICF_CT_SAMPLING_SEED=0
export ICF_CT_TOKENIZER=fps_lloyd
export ICF_CT_DISTANCE_KERNEL=broadcast
export ICF_FIXED_HEAD_CT_WEIGHT=0.7
export ICF_CT_TOKENS=32
export ICF_CV_BLOCKS=offdiag
echo "v110: CV=offdiag DD=full CT=pca32/even64/FPS+Lloyd30/K32/ridge w=0.7"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
