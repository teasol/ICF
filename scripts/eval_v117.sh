#!/usr/bin/env bash
# v117 = v116 minus DD (Linear 4-branch: CV=1.0, DD=0.0, CT=1.0, BM=1.0, BD=1.0).
#
# Primary 7-Task Benchmark:
#   v116 Baseline: 0.6119
#   v117 (No-DD Linear): 0.6191 (+0.0072 vs v116, 5/7 tasks won)
#
# Usage: bash scripts/eval_v117.sh <gpu> <tag> [task]...   (default: Primary 7 tasks)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V117_CKPT:-${ICF_V116_CKPT:-$ICF_CKPT}}"
CONFIG="${ICF_V117_CONFIG:-${ICF_V116_CONFIG:-$ICF_CONFIG}}"
GPU="${1:?usage: eval_v117.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v117.sh <gpu> <tag> [task]...}"
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
icf_arm_v117

echo "v117: CV=offdiag w=1.0 DD=off w=0.0 CT=pca32/random-frac=0.125/kmeans++/K256/ridge w=1.0 BM=dim32/lambda=1.0 w=1.0 BD=entropy/dim256 w=1.0 AGG=linear"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
