#!/usr/bin/env bash

# Sequential pipeline launcher for BagPFN experiments.
# Automatically runs multiple experiments sequentially in background mode (nohup setsid).
#
# Usage:
#   scripts/run_sequential_pipeline.sh <pipeline_name> <run1_name>:<config1> <run2_name>:<config2> ...

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <pipeline_name> <run1_name>:<config1> [<run2_name>:<config2> ...]" >&2
    exit 2
fi

PIPELINE_NAME="$1"
shift

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RUN_TIME="${ICF_RUN_TIME:-$(date +%Y%m%d_%H%M%S)}"
MASTER_LOG_DIR="${PROJECT_ROOT}/logs/pipeline_${PIPELINE_NAME}_${RUN_TIME}"

mkdir -p "${MASTER_LOG_DIR}"

PIPELINE_LOG="${MASTER_LOG_DIR}/pipeline.log"
PID_FILE="${MASTER_LOG_DIR}/pipeline.pid"

if [[ "${ICF_PIPELINE_WORKER:-0}" != "1" ]]; then
    export ICF_PIPELINE_WORKER=1
    nohup setsid stdbuf -oL -eL env "ICF_PIPELINE_WORKER=1" "CUDA_DEVICES=${CUDA_DEVICES}" "NPROC_PER_NODE=${NPROC_PER_NODE}" "TORCHRUN_BIN=${TORCHRUN_BIN}" \
        "$0" "${PIPELINE_NAME}" "$@" >"${PIPELINE_LOG}" 2>&1 < /dev/null &
    PIPELINE_PID=$!
    printf '%s\n' "${PIPELINE_PID}" >"${PID_FILE}"
    echo "Detached sequential pipeline '${PIPELINE_NAME}' started."
    echo "Pipeline PID: ${PIPELINE_PID}"
    echo "Pipeline Log: ${PIPELINE_LOG}"
    exit 0
fi

trap '' HUP

echo "=========================================================================="
echo "Starting Sequential Pipeline: ${PIPELINE_NAME}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================================="

STEP_NUM=1
TOTAL_STEPS=$#

for SPEC in "$@"; do
    RUN_NAME="${SPEC%%:*}"
    CONFIG="${SPEC#*:}"

    echo ""
    echo "--------------------------------------------------------------------------"
    echo "[Step ${STEP_NUM}/${TOTAL_STEPS}] Launching ${RUN_NAME} (${CONFIG})"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------------------------"

    # Run training in foreground within this pipeline worker
    ICF_FOREGROUND=1 \
    CUDA_DEVICES="${CUDA_DEVICES}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    TORCHRUN_BIN="${TORCHRUN_BIN}" \
    "${PROJECT_ROOT}/scripts/launch_interactive_training.sh" "${RUN_NAME}" "${CONFIG}"

    EXIT_CODE=$?
    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "[ERROR] Step ${STEP_NUM} (${RUN_NAME}) failed with exit code ${EXIT_CODE}."
        echo "Aborting remaining pipeline steps."
        exit ${EXIT_CODE}
    fi

    echo "[SUCCESS] Step ${STEP_NUM} (${RUN_NAME}) completed successfully."
    STEP_NUM=$((STEP_NUM + 1))
done

echo ""
echo "=========================================================================="
echo "Pipeline '${PIPELINE_NAME}' Completed All Steps Successfully!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================================="
