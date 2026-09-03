#!/usr/bin/env bash
# §218 screening run: v121 5-branch + RM (Residual Bag-Mean over PCA dims 33-256).
#
# RM reads basis[:, 32:256] -- the tail of the same K=256 PCA basis that BM/QA/DS
# only ever read the leading 32 columns of. ICF_RM_SCREEN_ONLY=1 records the RM
# margin WITHOUT letting it enter the ensemble, so the |r| screen mandated by
# docs/agent_handoff.md is measured before any performance claim.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

TAG="${1:-v121_rm_screen}"
export ICF_FIXED_HEAD_RM_WEIGHT="${ICF_FIXED_HEAD_RM_WEIGHT:-1.0}"
export ICF_RM_START="${ICF_RM_START:-32}"
export ICF_RM_DIM="${ICF_RM_DIM:-224}"
export ICF_RM_LAMBDA="${ICF_RM_LAMBDA:-1.0}"
export ICF_RM_SCREEN_ONLY="${ICF_RM_SCREEN_ONLY:-1}"

echo ">>> §218 RM SCREEN | start=${ICF_RM_START} dim=${ICF_RM_DIM} screen_only=${ICF_RM_SCREEN_ONLY} tag=${TAG}"

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
echo ">>> FINISHED RM SCREEN RUNS <<<"
