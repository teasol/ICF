#!/usr/bin/env bash
# §218 shape-family screening: BS (log total variance) and SH (per-dim skew+kurtosis).
#
# BS occupies the axis BD's entropy normalises away (total variance).
# SH is location- and scale-invariant by construction, so it cannot restate the
# Location family (CV/BM/QA/DS) or BS.
# ICF_SHAPE_SCREEN_ONLY=1 records both margins WITHOUT letting them enter the
# ensemble, so the |r| <= 0.6 screen is measured before any performance number.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

TAG="${1:-v121_shape_screen}"
export ICF_FIXED_HEAD_BS_WEIGHT="${ICF_FIXED_HEAD_BS_WEIGHT:-1.0}"
export ICF_FIXED_HEAD_SH_WEIGHT="${ICF_FIXED_HEAD_SH_WEIGHT:-1.0}"
export ICF_BS_DIM="${ICF_BS_DIM:-256}"
export ICF_SH_DIM="${ICF_SH_DIM:-32}"
export ICF_SHAPE_SCREEN_ONLY="${ICF_SHAPE_SCREEN_ONLY:-1}"

echo ">>> §218 SHAPE SCREEN | BS_dim=${ICF_BS_DIM} SH_dim=${ICF_SH_DIM} screen_only=${ICF_SHAPE_SCREEN_ONLY} tag=${TAG}"

bash scripts/eval_v121.sh 0 "${TAG}" cptac_lscc/ARID1A_mutation ucla_lung/progression_regression &
pid0=$!
bash scripts/eval_v121.sh 1 "${TAG}" cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation &
pid1=$!
bash scripts/eval_v121.sh 2 "${TAG}" cptac_lscc/KEAP1_mutation &
pid2=$!
bash scripts/eval_v121.sh 3 "${TAG}" cptac_luad/KRAS_mutation &
pid3=$!
bash scripts/eval_v121.sh 4 "${TAG}" cptac_pda/SMAD4_mutation &
pid4=$!
wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4"
echo ">>> FINISHED SHAPE SCREEN <<<"
