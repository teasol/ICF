#!/usr/bin/env bash
# Sequential S1 pipeline for v30 B1 (pool-standardized bag representation).
#
# For each run: train 50 epochs -> 1,000-episode synthetic eval -> Musk zero-shot
# eval (default bag_view preprocessing; the new representation is baked into the
# config, so --preprocess must stay at its default here).
#
# Sequential because the GPU is a single B200 and parallel runs have OOM'd before
# (docs/history/archive.md SS24: 3 concurrent runs needed ~168 GiB). The spawn-race
# fix from scripts/archive/ia_mil/queue_phase1_rare.sh is carried over: the
# launcher backgrounds a detached worker and returns immediately, so a bare pgrep
# right after launching races with train.py appearing and would spuriously report
# "GPU free", launching two runs at once.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python
TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun
NETRC=/NHNHOME/kimds/.netrc
QUEUE_LOG="${ICF_QUEUE_LOG:-${PROJECT_ROOT}/logs/queue_v30_poolz.log}"
TRAIN_SPAWN_TIMEOUT="${TRAIN_SPAWN_TIMEOUT:-300}"
VAL_EPISODES="${VAL_EPISODES:-1000}"

mkdir -p "$(dirname "$QUEUE_LOG")" predictions
log() { echo "[queue $(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$QUEUE_LOG"; }

declare -A RUN_CONFIGS=(
  [v30_musklike_easy_poolz]=configs/train_v30_musklike_easy_poolz.yaml
  [v30_musklike_easy_poolz_l2]=configs/train_v30_musklike_easy_poolz_l2.yaml
)
DEFAULT_ORDER=(v30_musklike_easy_poolz v30_musklike_easy_poolz_l2)

# Real training processes only.
#
# FIX (2026-08-04): the inherited `pgrep -f "scripts/train.py"` matches ANY process
# whose command line contains that string -- including an interactive shell running
# `pgrep -af scripts/train.py` to check on the queue. That made the queue believe a
# training was live and block forever in wait_gpu_free. Filtering on /proc/<pid>/comm
# keeps only actual python/torchrun processes, so shells cannot be mistaken for runs.
training_pids() {
  local pid comm
  for pid in $(pgrep -f "scripts/train\.py" 2>/dev/null || true); do
    [[ "$pid" == "$$" ]] && continue
    comm="$(cat "/proc/$pid/comm" 2>/dev/null || true)"
    case "$comm" in
      python*|torchrun*|pt_main_thread*) echo "$pid" ;;
    esac
  done
}

training_running() { [[ -n "$(training_pids)" ]]; }

wait_gpu_free() {
  log "waiting for GPU to be free..."
  while training_running; do sleep 60; done
  log "GPU free."
}

# Blocks until the just-launched training has (1) appeared as scripts/train.py
# and (2) finished. Step (1) is what prevents the spawn race described above.
wait_launched_training_done() {
  local name="$1" waited=0
  log "waiting for $name to spawn its trainer..."
  while ! training_running; do
    sleep 5; waited=$((waited + 5))
    if (( waited >= TRAIN_SPAWN_TIMEOUT )); then
      log "WARN: $name spawned no trainer within ${TRAIN_SPAWN_TIMEOUT}s (check its launch log)"
      wait_gpu_free
      return 0
    fi
  done
  log "$name training started (pids: $(training_pids | tr '\n' ' ')); waiting for completion..."
  while training_running; do sleep 60; done
  log "$name training finished."
}

# Newest checkpoint dir for this experiment name, then the lowest val_ce_loss file.
best_checkpoint() {
  local name="$1"
  local dir
  dir="$(find checkpoints -mindepth 2 -maxdepth 2 -type d -name "$name" -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -1 | cut -d' ' -f2-)"
  [[ -z "$dir" ]] && return 1
  find "$dir" -name 'epoch=*val_ce_loss=*.ckpt' 2>/dev/null \
    | sed -E 's/.*val_ce_loss=([0-9.]+)\.ckpt$/\1 &/' | sort -n | head -1 | cut -d' ' -f2-
}

evaluate() {
  local name="$1" cfg="$2" ckpt="$3"
  log "$name: synthetic eval (${VAL_EPISODES} episodes)"
  if "$PYTHON_BIN" scripts/evaluate_synthetic.py \
        --checkpoint "$ckpt" --config "$cfg" \
        --output "predictions/synthetic_${name}_${VAL_EPISODES}ep.pt" \
        --val-episodes "$VAL_EPISODES" >> "$QUEUE_LOG" 2>&1; then
    log "$name: synthetic eval OK -> predictions/synthetic_${name}_${VAL_EPISODES}ep.pt"
  else
    log "$name: synthetic eval FAILED (continuing to Musk)"
  fi

  log "$name: Musk zero-shot eval"
  if "$PYTHON_BIN" scripts/test_musk.py \
        --checkpoint "$ckpt" --config "$cfg" \
        --output "predictions/musk_${name}.pt" >> "$QUEUE_LOG" 2>&1; then
    log "$name: Musk eval OK -> predictions/musk_${name}.pt"
  else
    log "$name: Musk eval FAILED"
  fi

  log "$name: cardinality-stratified Musk report"
  "$PYTHON_BIN" scripts/diagnose_musk_cardinality.py --report stratified --bootstrap 1000 \
    >> "$QUEUE_LOG" 2>&1 || log "$name: stratified report FAILED"
}

finish_one() {
  local name="$1" cfg="${RUN_CONFIGS[$1]}"
  local ckpt
  if ! ckpt="$(best_checkpoint "$name")" || [[ -z "$ckpt" ]]; then
    log "$name: no checkpoint found -> skipping eval (training likely died; check logs/)"
    return 1
  fi
  log "$name: best checkpoint = $ckpt"
  evaluate "$name" "$cfg" "$ckpt"
}

run_one() {
  local name="$1" cfg="${RUN_CONFIGS[$1]}"
  wait_gpu_free
  log "launching $name ($cfg)"
  CUDA_DEVICES=0 NPROC_PER_NODE=1 TORCHRUN_BIN="$TORCHRUN_BIN" NETRC="$NETRC" \
    scripts/launch_interactive_training.sh "$name" "$cfg" \
    || { log "launch FAILED for $name"; return 1; }
  wait_launched_training_done "$name"
  finish_one "$name"
}

# Adopt a training that is ALREADY in flight (launcher detaches, so an orchestrator
# restart must not relaunch it): wait for it to finish, then evaluate.
attach_one() {
  local name="$1"
  log "attaching to in-flight $name (pids: $(training_pids | tr '\n' ' ')); waiting for completion..."
  while training_running; do sleep 60; done
  log "$name training finished."
  finish_one "$name"
}

if [[ $# -gt 0 ]]; then RUNS=("$@"); else RUNS=("${DEFAULT_ORDER[@]}"); fi
# ICF_ATTACH_FIRST=1 means the first queued run is already training; adopt it
# instead of launching a duplicate.
ATTACH_FIRST="${ICF_ATTACH_FIRST:-0}"
log "started; queue = ${RUNS[*]} (attach_first=${ATTACH_FIRST})"
first=1
for run in "${RUNS[@]}"; do
  if [[ -z "${RUN_CONFIGS[$run]:-}" ]]; then
    log "ERROR: unknown run '$run' (valid: ${!RUN_CONFIGS[*]})"; exit 1
  fi
  if [[ "$first" == 1 && "$ATTACH_FIRST" == 1 ]]; then
    attach_one "$run" || log "$run: pipeline reported a failure; continuing with the next run"
  else
    run_one "$run" || log "$run: pipeline reported a failure; continuing with the next run"
  fi
  first=0
done
log "ALL DONE"
