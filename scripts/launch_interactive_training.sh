#!/usr/bin/env bash

# Launch an interactive-server training run in a new session. The default
# process is detached from the caller's terminal, so closing SSH/VS Code does
# not deliver SIGHUP to torchrun or its workers.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <run-kind> <config>" >&2
    exit 2
fi

RUN_KIND="$1"
DEFAULT_CONFIG="$2"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
SEED="${SEED:-42}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Avoid allocator fragmentation from changing synthetic episode shapes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# This A6000 host has an unstable NCCL peer-to-peer/NVLink path. Force NCCL to
# use its non-P2P transport so the first DDP collective cannot hang.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

CKPT_PATH="${CKPT_PATH:-}"
RUN_TIME="${ICF_RUN_TIME:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${ICF_LOG_ROOT:-${PROJECT_ROOT}/logs}"
CHECKPOINT_ROOT="${ICF_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
if [[ "${LOG_ROOT}" != /* ]]; then
    LOG_ROOT="${PROJECT_ROOT}/${LOG_ROOT}"
fi
if [[ "${CHECKPOINT_ROOT}" != /* ]]; then
    CHECKPOINT_ROOT="${PROJECT_ROOT}/${CHECKPOINT_ROOT}"
fi
LOG_DIR="${LOG_ROOT}/${RUN_TIME}"
LOG_FILE="${LOG_DIR}/${RUN_KIND}.out"
LAUNCH_LOG="${LOG_DIR}/${RUN_KIND}_launcher.out"
PID_FILE="${LOG_DIR}/${RUN_KIND}.pid"
CHECKPOINT_DIR="${ICF_CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${RUN_TIME}/${RUN_KIND}}"
if [[ "${CHECKPOINT_DIR}" != /* ]]; then
    CHECKPOINT_DIR="${PROJECT_ROOT}/${CHECKPOINT_DIR}"
fi
RUN_NAME="${RUN_KIND}_${RUN_TIME}"
RUN_GROUP="${RUN_KIND}_${RUN_TIME}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

if [[ "${ICF_DETACHED_WORKER:-0}" != "1" && "${ICF_FOREGROUND:-0}" != "1" ]]; then
    DETACHED_ENV=(
        "ICF_DETACHED_WORKER=1"
        "ICF_RUN_TIME=${RUN_TIME}"
        "ICF_LOG_ROOT=${LOG_ROOT}"
        "ICF_CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
        "ICF_CHECKPOINT_DIR=${CHECKPOINT_DIR}"
        "TORCHRUN_BIN=${TORCHRUN_BIN}"
        "CONFIG=${CONFIG}"
        "SEED=${SEED}"
        "CUDA_DEVICES=${CUDA_DEVICES}"
        "NPROC_PER_NODE=${NPROC_PER_NODE}"
        "CKPT_PATH=${CKPT_PATH}"
    )
    nohup setsid env "${DETACHED_ENV[@]}" "$0" "${RUN_KIND}" "${DEFAULT_CONFIG}" \
        >"${LAUNCH_LOG}" 2>&1 < /dev/null &
    LAUNCHER_PID=$!
    printf '%s\n' "${LAUNCHER_PID}" >"${PID_FILE}"
    echo "Detached ${RUN_KIND} training started."
    echo "PID: ${LAUNCHER_PID}"
    echo "Training log: ${LOG_FILE}"
    echo "Launcher log: ${LAUNCH_LOG}"
    echo "Checkpoint directory: ${CHECKPOINT_DIR}"
    exit 0
fi

if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
    echo "torchrun executable not found: ${TORCHRUN_BIN}" >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Training config not found: ${CONFIG}" >&2
    exit 2
fi

# Defense in depth for foreground children of the detached worker. setsid
# removes the controlling terminal and this prevents an inherited HUP from
# terminating torchrun if an outer shell exits unexpectedly.
if [[ "${ICF_DETACHED_WORKER:-0}" == "1" ]]; then
    trap '' HUP
fi

TRAIN_ARGS=(
    --config "${CONFIG}"
    --seed "${SEED}"
    --run-name "${RUN_NAME}"
    --run-group "${RUN_GROUP}"
    --checkpoint-dir "${CHECKPOINT_DIR}"
)
if [[ -n "${CKPT_PATH}" ]]; then
    TRAIN_ARGS+=(--ckpt-path "${CKPT_PATH}")
fi

echo "Starting ${RUN_KIND} training on GPUs ${CUDA_DEVICES}."
echo "Training log: ${LOG_FILE}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
if [[ -n "${CKPT_PATH}" ]]; then
    echo "Resuming from checkpoint: ${CKPT_PATH}"
fi

if [[ -z "${WANDB_MODE:-}" ]]; then
    NETRC_FILE="${NETRC:-${PROJECT_ROOT}/../.netrc}"
    if [[ ! -r "${NETRC_FILE}" ]]; then
        NETRC_FILE="${HOME}/.netrc"
    fi
    if [[ -n "${WANDB_API_KEY:-}" ]] || \
        { [[ -r "${NETRC_FILE}" ]] && grep -q 'machine api\.wandb\.ai' "${NETRC_FILE}"; }; then
        export WANDB_MODE=online
        export NETRC="${NETRC_FILE}"
    else
        export WANDB_MODE=offline
    fi
fi

set +e
PYTHONUNBUFFERED=1 \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${TORCHRUN_BIN}" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/train.py "${TRAIN_ARGS[@]}" \
    >"${LOG_FILE}" 2>&1
EXIT_STATUS=$?
set -e

if [[ ${EXIT_STATUS} -eq 0 ]]; then
    echo "${RUN_KIND} training completed successfully."
else
    echo "${RUN_KIND} training exited with status ${EXIT_STATUS}." >&2
fi
exit "${EXIT_STATUS}"
