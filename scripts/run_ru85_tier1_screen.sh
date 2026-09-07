#!/usr/bin/env bash
# RU-85: label-free gate 1 screen for the Tier 1 candidates AKS, MDX and LID.
#
# ICF_SHAPE_SCREEN_ONLY=1 records every candidate margin WITHOUT letting it enter
# the ensemble, so the |r| <= 0.6 screen is measured before any performance number
# and the fold-mean AUROC must match the baseline to four decimals.
# SH weight stays on because gate 1 must screen against the adopted SH and SHJ.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

TAG="${1:-ru85_tier1_screen}"
shift || true
export ICF_FIXED_HEAD_SH_WEIGHT="${ICF_FIXED_HEAD_SH_WEIGHT:-1.0}"
export ICF_FIXED_HEAD_BS_WEIGHT="${ICF_FIXED_HEAD_BS_WEIGHT:-0.0}"
export ICF_SH_DIM="${ICF_SH_DIM:-32}"
export ICF_SHAPE_SCREEN_ONLY=1
export ICF_TIER1="${ICF_TIER1:-aks,mdx,lid}"

TASKS=("$@")
if [ "${#TASKS[@]}" -eq 0 ]; then
  TASKS=(cptac_lscc/ARID1A_mutation ucla_lung/progression_regression
         cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation
         cptac_lscc/KEAP1_mutation cptac_luad/KRAS_mutation cptac_pda/SMAD4_mutation)
fi

echo ">>> RU-85 TIER1 SCREEN | tier1=${ICF_TIER1} screen_only=1 tag=${TAG} tasks=${#TASKS[@]}"
pids=()
gpu=0
for task in "${TASKS[@]}"; do
  bash scripts/eval_v121.sh "$gpu" "${TAG}" "$task" &
  pids+=("$!")
  gpu=$(( (gpu + 1) % 8 ))
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
echo ">>> FINISHED RU-85 TIER1 SCREEN (rc=${rc}) <<<"
exit "$rc"
