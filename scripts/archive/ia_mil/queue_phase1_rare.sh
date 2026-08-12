#!/usr/bin/env bash
# Sequential queue for the Phase 1 rare discriminator trainings.
# GPU is shared (single B200); 3 parallel runs OOM'd (2 already used ~168 GiB),
# so run one at a time. Each launched training is awaited (train.py actually
# spawned, then exited) before the next one is launched.
#
# FIX (2026-08-03): launch_interactive_training.sh backgrounds a detached worker
# and returns immediately, so a bare `pgrep` right after launch races (train.py
# not yet spawned) and spuriously reported "GPU free" -> next run launched
# concurrently -> CUDA OOM. We now poll for train.py to appear (grace period)
# and then block until it exits before launching the next run.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun
NETRC=/NHNHOME/BASE/kimds/.netrc
QUEUE_LOG="${ICF_QUEUE_LOG:-/tmp/phase1_queue.log}"
# Max seconds to wait for a launched training to actually spawn scripts/train.py.
TRAIN_SPAWN_TIMEOUT="${TRAIN_SPAWN_TIMEOUT:-180}"

echo "[queue $(date +%H:%M:%S)] started" | tee -a "$QUEUE_LOG"

wait_gpu_free() {
  echo "[queue $(date +%H:%M:%S)] waiting for GPU to be free..." | tee -a "$QUEUE_LOG"
  while pgrep -f "scripts/train.py" > /dev/null 2>&1; do
    sleep 60
  done
  echo "[queue $(date +%H:%M:%S)] GPU free." | tee -a "$QUEUE_LOG"
}

# Blocks until the just-launched training has (1) appeared as scripts/train.py
# and (2) finished. Without step (1), the immediate pgrep after launch races
# with the detached worker spawning train.py and can launch two runs at once.
wait_launched_training_done() {
  local name="$1"
  local waited=0
  echo "[queue $(date +%H:%M:%S)] waiting for $name to spawn train.py..." | tee -a "$QUEUE_LOG"
  while ! pgrep -f "scripts/train.py" > /dev/null 2>&1; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= TRAIN_SPAWN_TIMEOUT )); then
      echo "[queue $(date +%H:%M:%S)] WARN: $name showed no train.py within ${TRAIN_SPAWN_TIMEOUT}s (check its launch log)" | tee -a "$QUEUE_LOG"
      wait_gpu_free   # final safety: never proceed while any training is running
      return 0
    fi
  done
  echo "[queue $(date +%H:%M:%S)] $name training started; waiting for completion..." | tee -a "$QUEUE_LOG"
  while pgrep -f "scripts/train.py" > /dev/null 2>&1; do
    sleep 60
  done
  echo "[queue $(date +%H:%M:%S)] $name training finished." | tee -a "$QUEUE_LOG"
}

run_seq() {
  local name="$1" cfg="$2"
  wait_gpu_free
  echo "[queue $(date +%H:%M:%S)] launching $name ($cfg)" | tee -a "$QUEUE_LOG"
  CUDA_DEVICES=0 NPROC_PER_NODE=1 TORCHRUN_BIN="$TORCHRUN_BIN" NETRC="$NETRC" \
    scripts/launch_interactive_training.sh "$name" "$cfg" \
    || { echo "[queue] launch FAILED for $name" | tee -a "$QUEUE_LOG"; exit 1; }
  wait_launched_training_done "$name"
}

# Run selector: pass run names as args to run only those (in order); with no
# args all defaults run. e.g. `queue_phase1_rare.sh v24_musklike_easy_rare_mil`
# trains only the MIL discriminator without re-running the baseline.
declare -A RUN_CONFIGS=(
  [v24_musklike_easy_rare_baseline]=configs/archive/ia_mil/train_v24_musklike_easy_rare_baseline.yaml
  [v24_musklike_easy_rare_mil]=configs/archive/ia_mil/train_v24_musklike_easy_rare_mil.yaml
)
if [[ $# -gt 0 ]]; then
  RUNS=("$@")
else
  RUNS=(v24_musklike_easy_rare_baseline v24_musklike_easy_rare_mil)
fi

for run in "${RUNS[@]}"; do
  cfg="${RUN_CONFIGS[$run]:-}"
  if [[ -z "$cfg" ]]; then
    echo "[queue $(date +%H:%M:%S)] ERROR: unknown run '$run' (valid: ${!RUN_CONFIGS[*]})" | tee -a "$QUEUE_LOG"
    exit 1
  fi
  run_seq "$run" "$cfg"
done
echo "[queue $(date +%H:%M:%S)] ALL DONE" | tee -a "$QUEUE_LOG"
