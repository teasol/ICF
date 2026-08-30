#!/usr/bin/env bash
# DE (Dual Extreme MIL) and SW (Sliced Wasserstein) evaluation runner.
#
# Usage: bash scripts/eval_de_sw.sh <mode> <gpu> <tag> [task]...
# Modes:
#   - standalone_de      : only DE=1.0, all others 0.0
#   - standalone_sw      : only SW=1.0, all others 0.0
#   - ensemble_7branch_de: v120 (CV+CT+BM+BD+QA+DS) + DE=1.0, Trimmed Mean
#   - ensemble_7branch_sw: v120 (CV+CT+BM+BD+QA+DS) + SW=1.0, Trimmed Mean
#   - ensemble_8branch   : v120 (CV+CT+BM+BD+QA+DS) + DE=1.0 + SW=1.0, Trimmed Mean
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
CKPT="${ICF_V119_CKPT:-${ICF_V118_CKPT:-$ICF_CKPT}}"
CONFIG="${ICF_V119_CONFIG:-${ICF_V118_CONFIG:-$ICF_CONFIG}}"

MODE="${1:?usage: eval_de_sw.sh <mode> <gpu> <tag> [task]...}"
GPU="${2:?usage: eval_de_sw.sh <mode> <gpu> <tag> [task]...}"
TAG="${3:?usage: eval_de_sw.sh <mode> <gpu> <tag> [task]...}"
shift 3

if [ "$#" -eq 0 ]; then
  set -- cptac_lscc/ARID1A_mutation \
    cptac_lscc/Histologic_Grade \
    cptac_lscc/KEAP1_mutation \
    cptac_luad/KRAS_mutation \
    cptac_pda/SMAD4_mutation \
    ucla_lung/progression_regression \
    cptac_ccrcc/PBRM1_mutation
fi

export ICF_COVARIANCE_BASIS=pca_within
export ICF_FIXED_HEAD=1
export ICF_SKETCH_DIM=256

# Common Head params
export ICF_CT_PCA_DIM=32
export ICF_CT_READOUT=ridge
export ICF_CT_CELLS="${ICF_CT_CELLS:-0.125}"
export ICF_CT_CELLS_SCALE="${ICF_CT_CELLS_SCALE:-own}"
export ICF_CT_CELLS_MIN="${ICF_CT_CELLS_MIN:-64}"
export ICF_CT_ABUNDANCE_CELLS="${ICF_CT_ABUNDANCE_CELLS:-match}"
export ICF_CT_SAMPLING=random
export ICF_CT_SAMPLING_SEED="${ICF_CT_SAMPLING_SEED:-0}"
export ICF_CT_TOKENS=256
export ICF_CT_TOKENIZER=kmeans_plusplus
export ICF_CT_KMEANS_MAX_ITER="${ICF_CT_KMEANS_MAX_ITER:-8}"
export ICF_CT_DISTANCE_KERNEL=gemm
export ICF_CV_BLOCKS=offdiag
export ICF_BM_DIM=32
export ICF_BM_LAMBDA=1.0
export ICF_BD_DIM=256
export ICF_BD_METRIC=entropy
export ICF_BD_READOUT=ordered_typicality
export ICF_BD_SEPARATION_FLOOR=1.0
export ICF_QA_DIM=32
export ICF_QA_LAMBDA=1.0
export ICF_DS_DIM=32
export ICF_DS_LAMBDA=1.0
export ICF_DS_TEMPERATURE=1.0
export ICF_DS_TOKENS=256

# DE params
export ICF_DE_DIM=32
export ICF_DE_LAMBDA=1.0
export ICF_DE_TOPK_FRACTION=0.05
export ICF_DE_TOPK_MIN=4
export ICF_DE_TOPK_MAX=64

# SW params
export ICF_SW_DIM=32
export ICF_SW_LAMBDA=1.0
export ICF_SW_NUM_SLICES=32
export ICF_SW_NUM_QUANTILES=32

case "$MODE" in
  standalone_de)
    export ICF_FIXED_HEAD_CV_WEIGHT=0.0
    export ICF_FIXED_HEAD_DD_WEIGHT=0.0
    export ICF_FIXED_HEAD_CT_WEIGHT=0.0
    export ICF_FIXED_HEAD_BM_WEIGHT=0.0
    export ICF_FIXED_HEAD_BD_WEIGHT=0.0
    export ICF_FIXED_HEAD_QA_WEIGHT=0.0
    export ICF_FIXED_HEAD_DS_WEIGHT=0.0
    export ICF_FIXED_HEAD_LR_WEIGHT=0.0
    export ICF_FIXED_HEAD_DE_WEIGHT=1.0
    export ICF_FIXED_HEAD_SW_WEIGHT=0.0
    export ICF_AGGREGATION=linear
    echo "=== Mode: Standalone DE (Dual Extreme Instance MIL) ==="
    ;;
  standalone_sw)
    export ICF_FIXED_HEAD_CV_WEIGHT=0.0
    export ICF_FIXED_HEAD_DD_WEIGHT=0.0
    export ICF_FIXED_HEAD_CT_WEIGHT=0.0
    export ICF_FIXED_HEAD_BM_WEIGHT=0.0
    export ICF_FIXED_HEAD_BD_WEIGHT=0.0
    export ICF_FIXED_HEAD_QA_WEIGHT=0.0
    export ICF_FIXED_HEAD_DS_WEIGHT=0.0
    export ICF_FIXED_HEAD_LR_WEIGHT=0.0
    export ICF_FIXED_HEAD_DE_WEIGHT=0.0
    export ICF_FIXED_HEAD_SW_WEIGHT=1.0
    export ICF_AGGREGATION=linear
    echo "=== Mode: Standalone SW (Sliced Wasserstein Distribution) ==="
    ;;
  ensemble_7branch_de)
    export ICF_FIXED_HEAD_CV_WEIGHT=1.0
    export ICF_FIXED_HEAD_DD_WEIGHT=0.0
    export ICF_FIXED_HEAD_CT_WEIGHT=1.0
    export ICF_FIXED_HEAD_BM_WEIGHT=1.0
    export ICF_FIXED_HEAD_BD_WEIGHT=1.0
    export ICF_FIXED_HEAD_QA_WEIGHT=1.0
    export ICF_FIXED_HEAD_DS_WEIGHT=1.0
    export ICF_FIXED_HEAD_LR_WEIGHT=0.0
    export ICF_FIXED_HEAD_DE_WEIGHT=1.0
    export ICF_FIXED_HEAD_SW_WEIGHT=0.0
    export ICF_AGGREGATION=trimmed_mean
    echo "=== Mode: 7-Branch Ensemble (v120 + DE) Trimmed Mean ==="
    ;;
  ensemble_7branch_sw)
    export ICF_FIXED_HEAD_CV_WEIGHT=1.0
    export ICF_FIXED_HEAD_DD_WEIGHT=0.0
    export ICF_FIXED_HEAD_CT_WEIGHT=1.0
    export ICF_FIXED_HEAD_BM_WEIGHT=1.0
    export ICF_FIXED_HEAD_BD_WEIGHT=1.0
    export ICF_FIXED_HEAD_QA_WEIGHT=1.0
    export ICF_FIXED_HEAD_DS_WEIGHT=1.0
    export ICF_FIXED_HEAD_LR_WEIGHT=0.0
    export ICF_FIXED_HEAD_DE_WEIGHT=0.0
    export ICF_FIXED_HEAD_SW_WEIGHT=1.0
    export ICF_AGGREGATION=trimmed_mean
    echo "=== Mode: 7-Branch Ensemble (v120 + SW) Trimmed Mean ==="
    ;;
  ensemble_8branch)
    export ICF_FIXED_HEAD_CV_WEIGHT=1.0
    export ICF_FIXED_HEAD_DD_WEIGHT=0.0
    export ICF_FIXED_HEAD_CT_WEIGHT=1.0
    export ICF_FIXED_HEAD_BM_WEIGHT=1.0
    export ICF_FIXED_HEAD_BD_WEIGHT=1.0
    export ICF_FIXED_HEAD_QA_WEIGHT=1.0
    export ICF_FIXED_HEAD_DS_WEIGHT=1.0
    export ICF_FIXED_HEAD_LR_WEIGHT=0.0
    export ICF_FIXED_HEAD_DE_WEIGHT=1.0
    export ICF_FIXED_HEAD_SW_WEIGHT=1.0
    export ICF_AGGREGATION=trimmed_mean
    echo "=== Mode: 8-Branch Ensemble (v120 + DE + SW) Trimmed Mean ==="
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac

exec bash scripts/eval_seal_tasks.sh "$GPU" "$CKPT" "$CONFIG" "$TAG" "$@"
