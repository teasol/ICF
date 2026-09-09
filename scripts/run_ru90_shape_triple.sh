#!/usr/bin/env bash
# RU-90 Phase 1 -- screen BS, SH and SJ in ONE run so every shape margin comes
# from a single code generation, a single basis and a single fold set.
#
# Why a fresh run is needed: no stored tag holds all three margins, and the two
# that hold m_sh disagree. In v121_shape_screen (§218) and v121_sh_variants
# (§219) the five base branches and the labels are bit-identical (max|diff| = 0)
# but m_sh differs by up to 6.5e-2, even though both run scripts set
# ICF_SH_DIM=32 and ICF_SH_WIDE=256 -- the stored margins are products of
# different code generations. Mechanism unidentified.
#
# ICF_SHAPE_SCREEN_ONLY=1 keeps BS/SH/SJ out of every aggregation pool, so this
# run's own probabilities ARE the official 5-branch BASE arm. That makes the
# run serve double duty: it supplies RU-90's layer-1 fidelity check (macro
# 0.6171, SMAD4 0.4421, PBRM1 0.5553) and the BASE arm of layer 2 at no extra
# cost, while the +SH / +SJ / +SH+SJ arms come from offline re-aggregation.
#
# BS is screened for its margin only. It is gate-(2) rejected (31/50, p=0.059,
# archive.md:1443-1452) and by user decision of 2026-09-10 it is NOT an arm of
# this RU; no BS configuration may be put forward for promotion.
#
# Usage: bash scripts/run_ru90_shape_triple.sh [tag]
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
. scripts/lib/free_gpus.sh

TAG="${1:-ru90_shape_triple}"

export ICF_FIXED_HEAD_BS_WEIGHT="${ICF_FIXED_HEAD_BS_WEIGHT:-1.0}"
export ICF_FIXED_HEAD_SH_WEIGHT="${ICF_FIXED_HEAD_SH_WEIGHT:-1.0}"
export ICF_FIXED_HEAD_SJ_WEIGHT="${ICF_FIXED_HEAD_SJ_WEIGHT:-1.0}"
export ICF_BS_DIM="${ICF_BS_DIM:-256}"
export ICF_BS_LAMBDA="${ICF_BS_LAMBDA:-1.0}"
export ICF_SH_DIM="${ICF_SH_DIM:-32}"
export ICF_SH_WIDE="${ICF_SH_WIDE:-256}"
export ICF_SH_LAMBDA="${ICF_SH_LAMBDA:-1.0}"
# Only the two adopted shape margins need a ridge solve; the §219 screen-only
# variants (shs/shk/sh2/shr/shr2) are not part of this RU.
export ICF_SH_VARIANTS="${ICF_SH_VARIANTS:-sh,sj}"
export ICF_SHAPE_SCREEN_ONLY=1

TASKS=(
  cptac_lscc/ARID1A_mutation
  cptac_lscc/Histologic_Grade
  cptac_lscc/KEAP1_mutation
  cptac_luad/KRAS_mutation
  cptac_pda/SMAD4_mutation
  ucla_lung/progression_regression
  cptac_ccrcc/PBRM1_mutation
)

mapfile -t GPUS < <(icf_free_gpus)
if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "!!! no idle GPU -- RU-90 kill condition (3). Refusing to evict a running job." >&2
  exit 1
fi
echo ">>> RU-90 PHASE 1 | tag=${TAG} | idle GPUs: ${GPUS[*]} | BS/SH/SJ screen-only"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# Round-robin the tasks over the idle devices, one eval_v121.sh per device.
declare -A PLAN=()
for i in "${!TASKS[@]}"; do
  g="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  PLAN[$g]="${PLAN[$g]:-} ${TASKS[$i]}"
done

pids=()
for g in "${!PLAN[@]}"; do
  echo "    GPU ${g}:${PLAN[$g]}"
  # shellcheck disable=SC2086
  bash scripts/eval_v121.sh "$g" "$TAG" ${PLAN[$g]} &
  pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo ">>> RU-90 PHASE 1 FINISHED rc=${rc} <<<"
exit "$rc"
