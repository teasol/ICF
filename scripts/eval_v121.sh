#!/usr/bin/env bash
# v121 = 5-Branch Fast Baseline (CV + BM + BD + QA + DS, CT=OFF) + Trimmed Mean Voting.
#
# Branches:
#   - CV: off-diagonal covariance ridge (w=1.0)
#   - DD: OFF (w=0.0)
#   - CT: OFF (w=0.0) -> Bypasses K-Means in 0 ms!
#   - BM: PCA-32 projected bag-mean ridge (w=1.0)
#   - BD: Spectral Entropy ordered-typicality (w=1.0)
#   - QA: PCA-32 Quantile & Extremum Evidence ridge (w=1.0, quantiles: 0.05, 0.10, 0.90, 0.95)
#   - DS: PCA-32 Salience Denoised Bag-Mean ridge (w=1.0, temp=1.0, tokens=256)
#   - Aggregation: Trimmed Mean Voting (drops min & max probability per slide, averages remaining 3)
#
# Usage: bash scripts/eval_v121.sh <gpu> <tag> [task]...   (default: Primary 7 tasks)
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT=""
CONFIG="configs/archive/v94_v102_cell_value/train_v98_p1_reverse_1536_1gpu.yaml"
GPU="${1:?usage: eval_v121.sh <gpu> <tag> [task]...}"
TAG="${2:?usage: eval_v121.sh <gpu> <tag> [task]...}"
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
icf_arm_v121

echo "v121: CV=offdiag w=1.0 DD=off w=0.0 CT=off w=0.0 BM=dim32 w=1.0 BD=entropy/dim256 w=1.0 QA=dim32/ridge w=1.0 DS=dim32/ridge w=1.0 AGG=trimmed_mean"
exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
