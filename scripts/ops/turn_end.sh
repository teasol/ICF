#!/usr/bin/env bash
# Refuse to end an orchestrator turn with an idle node.
#
# The failure this catches has now happened twice in one day: a round finishes,
# the orchestrator reports on it, and the turn ends with nothing running and
# nothing queued. The node then sits until a human types something. Both times
# the orchestrator had said it would continue, and both times the intent was
# not backed by anything the machine could act on.
#
# Intent is not a mechanism. This is: run it before ending a turn, and it says
# what is missing.
#
# Exit: 0 everything armed · 1 node will go idle
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
RC=0

bash scripts/ops/check_supervisor.sh || RC=1

QUEUED=$(ls talks/ops/queue/*.json 2>/dev/null | wc -l)
COUNCIL=$(pgrep -f "counci[l].py run" >/dev/null && echo yes || echo no)
# Not a pattern list: every new experiment script would have to be added to it,
# and a script missing from it makes a busy node read as idle -- which is how a
# dispatch lands on top of a running experiment. gpu_busy.py asks the driver and
# /proc instead. branch_redundancy.py was the first to fall through the list.
EXPCOUNT=$(.venv/bin/python scripts/ops/gpu_busy.py --gpu 4 2>/dev/null | grep -oE '^[0-9]+' || echo 0)
EXPERIMENT=$([ "${EXPCOUNT:-0}" -gt 0 ] && echo yes || echo no)
CHORES=$(pgrep -f "lit_dige[s]t.py|seat_thinkin[g].py|routine_[o]pen_questions.py|run_test[s].sh" \
         | wc -l)

echo "회차 실행: ${COUNCIL} · 실험 실행: ${EXPERIMENT}(${EXPCOUNT}) · 잡무 실행: ${CHORES}건 · 큐: ${QUEUED}건"

# A recurring template due soon counts as armed: the supervisor will queue it.
RECUR=$(ls talks/ops/recurring/*.json 2>/dev/null | wc -l)
if [ "$COUNCIL" = "no" ] && [ "$EXPERIMENT" = "no" ] && [ "$CHORES" -eq 0 ] \
   && [ "$QUEUED" -eq 0 ] && [ "$RECUR" -eq 0 ]; then
  echo "!! 노드가 유휴로 남는다. 회차를 띄우거나 큐를 채운 뒤 턴을 끝내라." >&2
  RC=1
fi

if [ "$COUNCIL" = "no" ] && [ "$EXPERIMENT" = "no" ]; then
  echo "   (참고) 회차도 실험도 돌지 않는다. 주기 잡무만으로는 연구가 진전되지 않는다." >&2
fi

# Chore failures used to vanish: the card moved to done/ whether the command
# worked or not. The supervisor now logs non-zero exits; surface them here so
# a turn cannot end while failures sit unread.
FAILLOG="talks/ops/failures.jsonl"
if [ -s "$FAILLOG" ]; then
  NFAIL=$(wc -l < "$FAILLOG")
  echo "잡무 실패 누적: ${NFAIL}건 · 최근:"
  tail -3 "$FAILLOG" | .venv/bin/python -c "
import json,sys
for line in sys.stdin:
    r = json.loads(line)
    print(f\"   {r['t']}  {r['label']}  rc={r['rc']}  {r['log']}\")
"
fi

# The supervisor holds its code in memory. If supervisor.py is newer than the
# running process, the fix just written is not the code that is running.
if [ -f talks/ops/supervisor.pid ]; then
  SPID=$(cat talks/ops/supervisor.pid)
  if [ -d "/proc/$SPID" ] \
     && [ scripts/ops/supervisor.py -nt "/proc/$SPID" ]; then
    echo "!! supervisor.py가 실행 중인 감독자보다 새롭다. 재시작이 필요하다." >&2
    RC=1
  fi
fi
exit "$RC"
