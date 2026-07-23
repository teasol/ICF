#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <a|b|b2|c|c0|c1|c2|c3|c4|c4-n|c4-d|c5|d0|d1|d2|d3|d4>" >&2
    exit 2
fi

case "$1" in
    c4-n) stage="c4-n"; config_stage="c4_n" ;;
    c4-d) stage="c4-d"; config_stage="c4_d" ;;
    a|b|b2|c|c0|c1|c2|c3|c4|c5|d0|d1|d2|d3|d4) stage="$1"; config_stage="$1" ;;
    *) echo "Unknown learnability stage: $1" >&2; exit 2 ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

run_kind="learnability_${stage}"
config="configs/train_learnability_${config_stage}.yaml"
CUDA_DEVICES="${CUDA_DEVICES:-0}" \
NPROC_PER_NODE=1 \
CKPT_PATH= \
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}" \
    scripts/launch_interactive_training.sh "${run_kind}" "${config}"
