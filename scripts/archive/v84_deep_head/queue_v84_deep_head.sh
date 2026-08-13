#!/usr/bin/env bash

# ARCHIVED (docs SS110, 2026-08-13): already ran once (checkpoints/20260813_163412,
# tags v84_deep_head_seed4{2..5}_ep49) and v84 was rejected against both v82 and
# v83 (4/4 seeds, |t|>3.6 either way). Kept only as reproduction evidence -- the
# relation-head-depth axis is closed, don't relaunch this to explore it further.
#
# Launch the v84 deep-head 4-seed batch once the v83 linear-head batch is done.
#
# Both arms need all four of GPUs 0-3, so they cannot overlap. This waits on the
# artifact each run must produce -- the epoch-49 periodic checkpoint for every
# seed -- rather than on process state.
#
# WHY NOT pgrep (SS66-5, measured): `while pgrep -f "scripts/train.py"` never
# terminates, because this script's own bash process carries that pattern in its
# command line and matches itself. Waiting on the checkpoints also survives a
# launcher wrapper exiting before its torchrun child.
#
# Usage: scripts/archive/v84_deep_head/queue_v84_deep_head.sh <v83_checkpoint_run_dir>
#   e.g. scripts/archive/v84_deep_head/queue_v84_deep_head.sh checkpoints/20260813_111630

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <v83_checkpoint_run_dir>" >&2
    exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

V83_DIR="$1"
SEEDS=(42 43 44 45)
CONFIG="configs/archive/v84_deep_head/train_v84_deep_head_1536_1gpu.yaml"
# 50 epochs is about 28 min; bail out well past that rather than hang forever.
DEADLINE=$(( $(date +%s) + 7200 ))

echo "[queue] waiting for v83 epoch-49 checkpoints under ${V83_DIR}"
while true; do
    done_count=0
    for seed in "${SEEDS[@]}"; do
        if compgen -G "${V83_DIR}/v83_linear_head_seed${seed}/periodic-epoch=049-*.ckpt" > /dev/null; then
            done_count=$((done_count + 1))
        fi
    done
    if [[ ${done_count} -eq ${#SEEDS[@]} ]]; then
        echo "[queue] all ${done_count} v83 seeds reached epoch 49"
        break
    fi
    if [[ $(date +%s) -gt ${DEADLINE} ]]; then
        echo "[queue] TIMEOUT: only ${done_count}/${#SEEDS[@]} v83 seeds finished; not launching v84" >&2
        exit 1
    fi
    sleep 60
done

# The checkpoint appears at the end of epoch 49, but the process still has to
# tear down and release its GPU memory.
sleep 120

echo "[queue] launching v84 deep-head 4-seed batch"
for i in 0 1 2 3; do
    seed=$((42 + i))
    SEED="${seed}" CUDA_DEVICES="${i}" NPROC_PER_NODE=1 \
    TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun \
    NETRC=/NHNHOME/BASE/kimds/.netrc \
    bash scripts/launch_interactive_training.sh \
        "v84_deep_head_seed${seed}" "${CONFIG}"
done
echo "[queue] v84 launched"
