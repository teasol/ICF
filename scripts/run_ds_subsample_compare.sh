#!/usr/bin/env bash
# §222: run DS with salience-anchor subsampling while ALSO recording the full-bag
# arm and the per-draw subsampled margins, so the label-free draw-stability
# selector can be tested offline against both arms.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
TAG="${1:-v121_ds_subsample_compare}"
export ICF_DS_AUG_MODE=salience_anchor
export ICF_DS_ANCHOR_FRACTION="${ICF_DS_ANCHOR_FRACTION:-0.15}"
export ICF_DS_AUG_S="${ICF_DS_AUG_S:-5}"
export ICF_DS_AUG_FRACTION="${ICF_DS_AUG_FRACTION:-0.7}"
export ICF_DS_SUBSAMPLE_COMPARE=1
echo ">>> §222 DS SUBSAMPLE COMPARE | S=${ICF_DS_AUG_S} f=${ICF_DS_AUG_FRACTION} a=${ICF_DS_ANCHOR_FRACTION} tag=${TAG}"
bash scripts/eval_v121.sh 0 "${TAG}" cptac_lscc/ARID1A_mutation ucla_lung/progression_regression & p0=$!
bash scripts/eval_v121.sh 1 "${TAG}" cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation & p1=$!
bash scripts/eval_v121.sh 2 "${TAG}" cptac_lscc/KEAP1_mutation & p2=$!
bash scripts/eval_v121.sh 3 "${TAG}" cptac_luad/KRAS_mutation & p3=$!
bash scripts/eval_v121.sh 4 "${TAG}" cptac_pda/SMAD4_mutation & p4=$!
wait "$p0" "$p1" "$p2" "$p3" "$p4"
echo ">>> FINISHED DS SUBSAMPLE COMPARE <<<"
