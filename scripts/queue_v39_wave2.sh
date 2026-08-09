#!/usr/bin/env bash
# Launch v39 wave 2 (no_covariance) once wave 1 (baseline, no_abundance) has
# released its GPU. Only two arms fit at a time (~98 GB each of 183 GB).
#
# Two traps this avoids (docs SS66-5, both hit in the previous session):
#   1. The launcher wrapper exits BEFORE its torchrun child, so waiting on the
#      wrapper PID fires while the GPU is still held. Wait on the launcher log's
#      terminal line instead.
#   2. `pgrep -f "scripts/train.py"` matches THIS script's own command line
#      (the pattern appears in it), so the loop never ends. The pattern below is
#      built at runtime so it cannot appear literally in this file, and `pgrep`
#      is given -x-free explicit exclusion of our own PID.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WAVE1_LOGS=(
  "logs/20260809_094656/v39_baseline_launcher.out"
  "logs/20260809_094703/v39_no_abundance_launcher.out"
)

# Assembled at runtime: never appears literally, so it cannot self-match.
TRAIN_PATTERN="train"".py --config"

while true; do
  done_count=0
  for log in "${WAVE1_LOGS[@]}"; do
    grep -qE "training (completed successfully|exited with status)" "$log" 2>/dev/null \
      && done_count=$((done_count + 1))
  done
  live=$(pgrep -f "$TRAIN_PATTERN" 2>/dev/null | grep -cv "^$$\$" || true)
  [[ "$done_count" -eq 2 && "${live:-0}" -eq 0 ]] && break
  sleep 60
done

echo "wave 1 finished at $(date); starting wave 2"
CUDA_DEVICES=0 NPROC_PER_NODE=1 \
TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun \
NETRC=/NHNHOME/BASE/kimds/.netrc \
scripts/launch_interactive_training.sh v39_no_covariance \
  configs/train_v39_no_covariance_1536.yaml
