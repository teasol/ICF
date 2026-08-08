#!/usr/bin/env bash
# Launch v38 ridge-ablation wave 2 (abundance, covariance) once wave 1
# (control, global) has released both GPUs. Only two arms fit at a time
# (~98 GB each of 183 GB), so the four-arm set runs as two waves.
#
# Waits on the launcher logs rather than on PIDs: the launcher wrapper exits
# before its torchrun child does, so a PID check can fire while the GPU is
# still held.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

WAVE1_LOGS=(
  "logs/20260809_025544/v38_control_launcher.out"
  "logs/20260809_025553/v38_global_launcher.out"
)

while true; do
  done_count=0
  for log in "${WAVE1_LOGS[@]}"; do
    grep -qE "training (completed successfully|exited with status)" "$log" 2>/dev/null && done_count=$((done_count + 1))
  done
  # Belt and braces: also require that no training process is left holding a GPU.
  live=$(pgrep -fc "scripts/train.py" || true)
  [[ "$done_count" -eq 2 && "$live" -eq 0 ]] && break
  sleep 60
done

echo "wave 1 finished at $(date); starting wave 2"
for pair in "0 abundance" "1 covariance"; do
  read -r gpu arm <<< "$pair"
  CUDA_DEVICES="$gpu" NPROC_PER_NODE=1 \
  TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun \
  NETRC=/NHNHOME/BASE/kimds/.netrc \
  scripts/launch_interactive_training.sh "v38_${arm}" \
    "configs/train_v38_ridge_ablation_${arm}_1536.yaml"
done
