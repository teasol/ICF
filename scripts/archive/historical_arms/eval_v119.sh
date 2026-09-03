#!/usr/bin/env bash
# v119 = 5-Branch (CV + CT + BM + BD + QA) + Trimmed Mean Voting.
#
# Branches:
#   - CV: off-diagonal covariance ridge (w=1.0)
#   - DD: OFF (w=0.0)
#   - CT: PCA-32 K256 soft abundance ridge (w=1.0)
#   - BM: PCA-32 projected bag-mean ridge (w=1.0)
#   - BD: Spectral Entropy ordered-typicality (w=1.0)
#   - QA: PCA-32 Quantile & Extremum Evidence ridge (w=1.0, quantiles: 0.05, 0.10, 0.90, 0.95)
#   - Aggregation: Trimmed Mean Voting (drops min & max probability per slide, averages remaining 3)
#
# Usage: bash scripts/eval_v119.sh <gpu> <tag> [task]...   (default: Primary 7 tasks)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V119_CKPT:-${ICF_V118_CKPT:-$ICF_CKPT}}"
CONFIG="${ICF_V119_CONFIG:-${ICF_V118_CONFIG:-$ICF_CONFIG}}"
GPU="${1:?usage: eval_v119.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v119.sh <gpu> <tag> [task]...}"
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
icf_arm_v119

echo "v119: CV=offdiag w=1.0 DD=off w=0.0 CT=pca32/ridge w=1.0 BM=dim32 w=1.0 BD=entropy/dim256 w=1.0 QA=dim32/ridge w=1.0 AGG=trimmed_mean"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"


