#!/usr/bin/env bash
# Run one council round in the foreground, then check the supervisor.
#
# Why this exists. Rounds must be launched by the orchestrator, not by the
# supervisor -- a queue of pre-written cards cannot respond to what the round
# before it found, so the council would converge on agreeing with its own
# earlier framing. But that only works if the orchestrator actually comes back
# when the round ends. On 2026-09-18 it did not: the round finished at 09:12
# and nothing ran until 09:40, because the orchestrator had replaced its
# per-round completion watch with an hourly wake-up.
#
# The fix is to make waking structural rather than remembered. This script
# BLOCKS until the round finishes, so launching it as a background task makes
# the task-completion notification the round-end wake-up. There is no separate
# watcher to arm and therefore none to forget.
#
#   (launch as a background task, never with nohup/&)
#   bash scripts/ops/run_round.sh talks/council/C-YYYYMMDD-N_card.json
#
# Exit: the round's own exit code (0 ok · 1 card rejected · 2 round void)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
CARD="${1:?usage: run_round.sh <card.json>}"
ROUND="$(basename "$CARD" _card.json)"

echo "=== 회차 ${ROUND} 시작 $(date '+%F %T') ==="
.venv/bin/python scripts/council.py run "$CARD"
RC=$?
echo "=== 회차 ${ROUND} 종료 rc=${RC} $(date '+%F %T') ==="

# Required at the end of every round: a dead supervisor is invisible otherwise.
bash scripts/ops/check_supervisor.sh || true

REPORT="talks/council/${ROUND}/report.md"
if [ -f "$REPORT" ]; then
  echo "--- 기권 좌석 ---"
  sed -n '/^## 기권한 좌석/,/^## /p' "$REPORT" | grep '^- ' || echo "  없음"
  echo "--- 큐 대기 ---"
  ls talks/ops/queue/ 2>/dev/null | wc -l
fi
exit "$RC"
