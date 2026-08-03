#!/usr/bin/env bash
# Sequential queue for the Phase 1 rare discriminator trainings.
# GPU is shared (single B200); 3 parallel runs OOM'd (2 already used ~168 GiB),
# so run one at a time. Waits for any running scripts/train.py to finish before
# launching the next.
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun
NETRC=/NHNHOME/kimds/.netrc
QUEUE_LOG="${ICF_QUEUE_LOG:-/tmp/phase1_queue.log}"

echo "[queue $(date +%H:%M:%S)] started" | tee -a "$QUEUE_LOG"

wait_gpu_free() {
  echo "[queue $(date +%H:%M:%S)] waiting for GPU to be free..." | tee -a "$QUEUE_LOG"
  while pgrep -f "scripts/train.py" > /dev/null 2>&1; do
    sleep 60
  done
  echo "[queue $(date +%H:%M:%S)] GPU free." | tee -a "$QUEUE_LOG"
}

run_seq() {
  local name="$1" cfg="$2"
  wait_gpu_free
  echo "[queue $(date +%H:%M:%S)] launching $name ($cfg)" | tee -a "$QUEUE_LOG"
  CUDA_DEVICES=0 NPROC_PER_NODE=1 TORCHRUN_BIN="$TORCHRUN_BIN" NETRC="$NETRC" \
    scripts/launch_interactive_training.sh "$name" "$cfg" \
    || { echo "[queue] launch FAILED for $name" | tee -a "$QUEUE_LOG"; exit 1; }
}

run_seq v24_musklike_easy_rare_baseline configs/train_v24_musklike_easy_rare_baseline.yaml
run_seq v24_musklike_easy_rare_mil configs/train_v24_musklike_easy_rare_mil.yaml
echo "[queue $(date +%H:%M:%S)] ALL DONE" | tee -a "$QUEUE_LOG"
