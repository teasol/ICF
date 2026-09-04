#!/usr/bin/env bash
# §221: record context-LOO margins for every branch that supports them, plus the
# context labels needed to score them. ICF_SAVE_CONTEXT_LOO=1 computes LOO WITHOUT
# switching the aggregation, so the ensemble output is unchanged.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
TAG="${1:-v121_loo_capacity}"
export ICF_FIXED_HEAD_SH_WEIGHT=1.0 ICF_FIXED_HEAD_BS_WEIGHT=0.0
export ICF_SHAPE_SCREEN_ONLY=1 ICF_SAVE_CONTEXT_LOO=1
echo ">>> §221 LOO CAPACITY | tag=${TAG}"
bash scripts/eval_v121.sh 0 "${TAG}" cptac_lscc/ARID1A_mutation ucla_lung/progression_regression & p0=$!
bash scripts/eval_v121.sh 1 "${TAG}" cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation & p1=$!
bash scripts/eval_v121.sh 2 "${TAG}" cptac_lscc/KEAP1_mutation & p2=$!
bash scripts/eval_v121.sh 3 "${TAG}" cptac_luad/KRAS_mutation & p3=$!
bash scripts/eval_v121.sh 4 "${TAG}" cptac_pda/SMAD4_mutation & p4=$!
wait "$p0" "$p1" "$p2" "$p3" "$p4"
echo ">>> FINISHED LOO CAPACITY <<<"
