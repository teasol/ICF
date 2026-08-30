#!/usr/bin/env bash
# v114 = v113 + fixed-head branch weights unified to 1.0 (SS186).
#
# v113 kept the legacy per-branch weights (CV=1.442 fitted against the old
# 8-head decomposition SS137-3, DD=1.0, CT=0.7 arbitrary). Setting all three
# to 1.0 measured SEAL 10 macro 0.70509 vs v113's 0.70394 (+0.00115, SS186) --
# most of the gain came from the two low-signal ccrcc tasks (BAP1, VHL), which
# SS118 weighs more heavily than the offsetting brca PIK3CA dip. Promoted as
# active baseline (SS187) despite being a deterministic arm with no t/p gate.
#
# Usage: bash scripts/eval_v114.sh <gpu> <tag> [task]...   (default: SEAL 10)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V114_CKPT:-$ICF_CKPT}"
CONFIG="${ICF_V114_CONFIG:-$ICF_CONFIG}"
GPU="${1:?usage: eval_v114.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v114.sh <gpu> <tag> [task]...}"
shift 2

if [ "$#" -eq 0 ]; then
  set -- bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status \
    cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation \
    cptac_luad/EGFR_mutation cptac_luad/STK11_mutation cptac_luad/TP53_mutation \
    cptac_ccrcc/BAP1_mutation cptac_ccrcc/VHL_mutation
fi

. "$(dirname "${BASH_SOURCE[0]}")/lib/arms.sh"
icf_arm_v114
echo "v114: CV=offdiag w=1.0 DD=ordered_typicality(kappa=1) w=1.0 CT=pca32/random-frac=${ICF_CT_CELLS}/${ICF_CT_CELLS_SCALE}/min=${ICF_CT_CELLS_MIN}/kmeans++/K256/match-abundance/ridge w=1.0"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
