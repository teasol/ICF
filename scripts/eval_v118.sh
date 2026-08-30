#!/usr/bin/env bash
# v118 = v117 + Soft Voting (4-branch Probability Averaging via Sigmoid: CV, CT, BM, BD, w_DD=0.0).
#
# Primary 7-Task Benchmark:
#   v114 Baseline: 0.6051
#   v115 Baseline: 0.6094 (+0.0043 vs v114)
#   v116 Baseline: 0.6119 (+0.0025 vs v115)
#   v117 (No-DD Linear): 0.6191 (+0.0072 vs v116)
#   v118 Baseline (No-DD Soft Voting): 0.6205 (+0.0086 vs v116, +0.0154 vs v114, 6/7 tasks won)
#
# Usage: bash scripts/eval_v118.sh <gpu> <tag> [task]...   (default: Primary 7 tasks)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V118_CKPT:-${ICF_V116_CKPT:-$ICF_CKPT}}"
CONFIG="${ICF_V118_CONFIG:-${ICF_V116_CONFIG:-$ICF_CONFIG}}"
GPU="${1:?usage: eval_v118.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v118.sh <gpu> <tag> [task]...}"
shift 2

if [ "$#" -eq 0 ]; then
  set -- cptac_lscc/ARID1A_mutation \
    cptac_lscc/Histologic_Grade \
    cptac_lscc/KEAP1_mutation \
    cptac_luad/KRAS_mutation \
    cptac_pda/SMAD4_mutation \
    ucla_lung/progression_regression \
    cptac_ccrcc/PBRM1_mutation
fi

. "$(dirname "${BASH_SOURCE[0]}")/lib/arms.sh"
icf_arm_v118

echo "v118: CV=offdiag w=1.0 DD=off w=0.0 CT=pca32/random-frac=0.125/kmeans++/K256/ridge w=1.0 BM=dim32/lambda=1.0 w=1.0 BD=entropy/dim256 w=1.0 AGG=soft_voting"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
