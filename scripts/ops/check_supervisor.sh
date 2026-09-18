#!/usr/bin/env bash
# Is the supervisor alive, and if not, bring it back.
#
# "Who watches the watcher" is a real gap: if the supervisor dies, state.md
# simply stops changing, and a stale file reads exactly like a quiet node. The
# heartbeat is written every tick so staleness is measurable, not inferred.
#
# Liveness is checked through a pid file, never `pgrep -f`. A `-f` pattern
# matches any command line containing it -- including this script's own, since
# the script necessarily names the process it manages. That self-match has
# killed three shells in this repository's history, so the pattern approach is
# not used here at all.
#
# Call at the end of every council round and every orchestrator turn. Cheap
# (no model call, no GPU) and idempotent.
#
# Exit: 0 alive · 1 was dead and restarted · 2 restart failed
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
PIDFILE="talks/ops/supervisor.pid"
HEARTBEAT="talks/ops/heartbeat"
LOG="logs/supervisor.log"
MAX_STALE="${ICF_SUPERVISOR_MAX_STALE:-180}"   # 3 ticks at the default 60s

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

stale_seconds() {
  [ -f "$HEARTBEAT" ] || { echo 999999; return; }
  echo $(( $(date +%s) - $(date -d "$(cat "$HEARTBEAT")" +%s 2>/dev/null || echo 0) ))
}

STALE="$(stale_seconds)"
if running && [ "$STALE" -le "$MAX_STALE" ]; then
  echo "감독자 정상 · pid $(cat "$PIDFILE") · 심박 ${STALE}초 전"
  exit 0
fi

if running; then
  # The process exists but stopped ticking: wedged, not working.
  echo "감독자 pid $(cat "$PIDFILE") 살아 있으나 심박 ${STALE}초 정지 — 교체한다" >&2
  kill "$(cat "$PIDFILE")" 2>/dev/null
  sleep 2
else
  echo "감독자 미가동 (심박 ${STALE}초 전) — 기동한다" >&2
fi

mkdir -p logs
setsid nohup .venv/bin/python scripts/ops/supervisor.py >> "$LOG" 2>&1 &
sleep 8
if running; then
  echo "감독자 재기동 완료 · pid $(cat "$PIDFILE")"
  exit 1
fi
echo "감독자 재기동 실패 — ${LOG} 확인" >&2
exit 2
