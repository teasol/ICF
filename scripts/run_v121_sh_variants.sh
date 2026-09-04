#!/usr/bin/env bash
# §219: SH strengthening. Emits seven variants as separate margins in ONE pass.
#   sh    moments (skew+kurt) dim32   -- reproduces §218
#   shs   skewness only     dim32     -- Phase 1 decomposition
#   shk   kurtosis only     dim32     -- Phase 1 decomposition
#   sh2   moments           dim256    -- pre-declared 32-vs-256 test
#   shr   robust (Bowley + Moors) dim32
#   shr2  robust            dim256
#   shj   joint whitened-radius shape (multivariate, not marginal)
# All are location- and scale-invariant by construction, as SH is.
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
TAG="${1:-v121_sh_variants}"
export ICF_FIXED_HEAD_SH_WEIGHT=1.0
export ICF_FIXED_HEAD_BS_WEIGHT=0.0     # BS rejected in §218
export ICF_SH_DIM="${ICF_SH_DIM:-32}"
export ICF_SH_WIDE="${ICF_SH_WIDE:-256}"
export ICF_SHAPE_SCREEN_ONLY=1
echo ">>> §219 SH VARIANTS | narrow=${ICF_SH_DIM} wide=${ICF_SH_WIDE} tag=${TAG}"
bash scripts/eval_v121.sh 0 "${TAG}" cptac_lscc/ARID1A_mutation ucla_lung/progression_regression &
p0=$!
bash scripts/eval_v121.sh 1 "${TAG}" cptac_lscc/Histologic_Grade cptac_ccrcc/PBRM1_mutation &
p1=$!
bash scripts/eval_v121.sh 2 "${TAG}" cptac_lscc/KEAP1_mutation & p2=$!
bash scripts/eval_v121.sh 3 "${TAG}" cptac_luad/KRAS_mutation & p3=$!
bash scripts/eval_v121.sh 4 "${TAG}" cptac_pda/SMAD4_mutation & p4=$!
wait "$p0" "$p1" "$p2" "$p3" "$p4"
echo ">>> FINISHED SH VARIANTS <<<"
