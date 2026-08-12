#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CKPT_DIR="checkpoints/20260810_195208/v62_linear16_cv1_k128_capfirst"
CONFIG="configs/archive/v62_v68_hybrid/train_v62_linear_hybrid_cv1_1536.yaml"
WATCH_LOG="logs/official50/watcher_v62_periodic.log"

mkdir -p logs/official50 predictions

is_complete() {
  local tag="$1"
  local count
  count=$(find predictions -maxdepth 1 -type f \
    -name "pathobench_*_${tag}_official50_bf16.pt" | wc -l)
  [[ "$count" -eq 10 ]]
}

evaluate_checkpoint() {
  local checkpoint="$1"
  local zero_epoch actual_epoch tag
  if [[ "$(basename "$checkpoint")" =~ periodic-epoch=([0-9]+)- ]]; then
    zero_epoch=$((10#${BASH_REMATCH[1]}))
  else
    return 0
  fi
  actual_epoch=$((zero_epoch + 1))
  tag=$(printf 'v62_e%03d' "$actual_epoch")

  # Epoch 10 was launched manually before this watcher.
  if is_complete "$tag"; then
    printf '%s SKIP complete %s\n' "$(date '+%F %T')" "$tag" >> "$WATCH_LOG"
    return 0
  fi

  printf '%s START %s %s\n' "$(date '+%F %T')" "$tag" "$checkpoint" >> "$WATCH_LOG"
  bash scripts/eval_seal_tasks.sh 4 "$checkpoint" "$CONFIG" "$tag" \
    bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status \
    cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation \
    >> "$WATCH_LOG" 2>&1 &
  local gpu4_pid=$!
  bash scripts/eval_seal_tasks.sh 5 "$checkpoint" "$CONFIG" "$tag" \
    cptac_luad/EGFR_mutation cptac_luad/STK11_mutation \
    cptac_luad/TP53_mutation cptac_ccrcc/BAP1_mutation \
    cptac_ccrcc/VHL_mutation >> "$WATCH_LOG" 2>&1 &
  local gpu5_pid=$!

  local rc4=0 rc5=0
  wait "$gpu4_pid" || rc4=$?
  wait "$gpu5_pid" || rc5=$?
  printf '%s END %s rc4=%d rc5=%d complete=%s\n' \
    "$(date '+%F %T')" "$tag" "$rc4" "$rc5" \
    "$(is_complete "$tag" && echo yes || echo no)" >> "$WATCH_LOG"
}

printf '%s WATCHER start\n' "$(date '+%F %T')" >> "$WATCH_LOG"
while true; do
  while IFS= read -r checkpoint; do
    evaluate_checkpoint "$checkpoint"
  done < <(find "$CKPT_DIR" -maxdepth 1 -type f -name 'periodic-*.ckpt' | sort)

  if is_complete v62_e100; then
    printf '%s WATCHER complete\n' "$(date '+%F %T')" >> "$WATCH_LOG"
    break
  fi
  sleep 15
done
