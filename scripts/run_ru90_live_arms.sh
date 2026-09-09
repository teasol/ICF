#!/usr/bin/env bash
# RU-90 Phase 3 -- layer-2 wiring equivalence. Runs the +SH, +SJ and +SH+SJ arms
# through the LIVE path (ICF_SHAPE_SCREEN_ONLY=0, so the shape margins really do
# enter the trimmed-mean pool) on ONE task, so their per-slide probabilities can
# be compared against an offline re-aggregation of the Phase 1 margins.
#
# The BASE arm needs no run here: Phase 1 is screen-only, so its own stored
# probabilities already are the 5-branch BASE arm.
#
# One task is enough because equivalence is exact arithmetic, not a statistical
# claim -- and 3 arms x 7 tasks would cost ~3 GPU-h against RU-90's 2.0 cap.
# BS stays at weight 0: it is gate-(2) rejected and is not an arm of this RU.
#
# Usage: bash scripts/run_ru90_live_arms.sh [task] [tag_prefix]
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
. scripts/lib/free_gpus.sh

TASK="${1:-cptac_lscc/ARID1A_mutation}"
PREFIX="${2:-ru90_live}"

mapfile -t GPUS < <(icf_free_gpus)
if [ "${#GPUS[@]}" -lt 1 ]; then
  echo "!!! no idle GPU -- RU-90 kill condition (3)." >&2
  exit 1
fi
echo ">>> RU-90 PHASE 3 | task=${TASK} | idle GPUs: ${GPUS[*]}"

# arm_name  sh_weight  sj_weight
ARMS=("sh 1.0 0.0" "sj 0.0 1.0" "shsj 1.0 1.0")

pids=()
i=0
for arm in "${ARMS[@]}"; do
  read -r name w_sh w_sj <<<"$arm"
  g="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  i=$(( i + 1 ))
  echo "    GPU ${g}: arm=${name} SH=${w_sh} SJ=${w_sj}"
  (
    export ICF_SHAPE_SCREEN_ONLY=0
    export ICF_FIXED_HEAD_BS_WEIGHT=0.0
    export ICF_FIXED_HEAD_SH_WEIGHT="$w_sh"
    export ICF_FIXED_HEAD_SJ_WEIGHT="$w_sj"
    export ICF_BS_DIM=256 ICF_BS_LAMBDA=1.0
    export ICF_SH_DIM=32 ICF_SH_WIDE=256 ICF_SH_LAMBDA=1.0
    export ICF_SH_VARIANTS=sh,sj
    bash scripts/eval_v121.sh "$g" "${PREFIX}_${name}" "$TASK"
  ) &
  pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo ">>> RU-90 PHASE 3 FINISHED rc=${rc} <<<"
exit "$rc"
