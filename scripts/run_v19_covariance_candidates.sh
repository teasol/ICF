#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

BATCH_TIME="${ICF_CANDIDATE_BATCH_TIME:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG_DIR="${ICF_LOG_ROOT:-${PROJECT_ROOT}/logs}/v19_covariance_candidates_${BATCH_TIME}"
mkdir -p "${MASTER_LOG_DIR}"

if [[ "${ICF_CANDIDATE_WORKER:-0}" != "1" && "${ICF_FOREGROUND:-0}" != "1" ]]; then
    nohup setsid env \
        ICF_CANDIDATE_WORKER=1 \
        ICF_CANDIDATE_BATCH_TIME="${BATCH_TIME}" \
        ICF_LOG_ROOT="${ICF_LOG_ROOT:-${PROJECT_ROOT}/logs}" \
        ICF_CHECKPOINT_ROOT="${ICF_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}" \
        CUDA_DEVICES="${CUDA_DEVICES:-0}" \
        TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}" \
        "$0" "$@" >"${MASTER_LOG_DIR}/runner.out" 2>&1 < /dev/null &
    echo "Detached v19 covariance candidate sequence started."
    echo "PID: $!"
    echo "Runner log: ${MASTER_LOG_DIR}/runner.out"
    exit 0
fi

declare -A CONFIGS=(
    [raw]="configs/train_v19_cov_candidate_0_raw.yaml"
    [correlation]="configs/train_v19_cov_candidate_1_correlation.yaml"
    [shrinkage]="configs/train_v19_cov_candidate_2_shrinkage.yaml"
    [logcorr]="configs/train_v19_cov_candidate_3_logcorr.yaml"
    [raw_logcorr]="configs/train_v19_cov_candidate_4_raw_logcorr.yaml"
)
DEFAULT_CANDIDATES=(raw correlation shrinkage logcorr raw_logcorr)

if [[ $# -eq 0 || "${1:-}" == "all" ]]; then
    CANDIDATES=("${DEFAULT_CANDIDATES[@]}")
else
    CANDIDATES=("$@")
fi

for candidate in "${CANDIDATES[@]}"; do
    if [[ -z "${CONFIGS[${candidate}]:-}" ]]; then
        echo "Unknown candidate: ${candidate}" >&2
        echo "Valid candidates: ${DEFAULT_CANDIDATES[*]}" >&2
        exit 2
    fi
done

for candidate in "${CANDIDATES[@]}"; do
    run_kind="v19_covariance_${candidate}"
    run_time="${BATCH_TIME}_${candidate}"
    config="${CONFIGS[${candidate}]}"
    echo "Starting ${candidate}: ${config}"
    ICF_FOREGROUND=1 ICF_RUN_TIME="${run_time}" \
    CUDA_DEVICES="${CUDA_DEVICES:-0}" NPROC_PER_NODE=1 CKPT_PATH= \
    TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}" \
        scripts/launch_interactive_training.sh "${run_kind}" "${config}"

    checkpoint_dir="${ICF_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}/${run_time}/${run_kind}"
    best="$({ find "${checkpoint_dir}" -maxdepth 1 -type f \
        -name 'epoch=*-val_ce_loss=*.ckpt' -printf '%f\n'; } \
        | sed -n 's/.*val_ce_loss=\([0-9.]*\)\.ckpt/\1 &/p' \
        | sort -n | head -n 1)"
    echo "Completed ${candidate}; best ${best:-unavailable}"
done

echo "All requested v19 covariance candidates completed."
