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
EXPERIMENT=$(pgrep -f "run_gf_standal[o]ne.py|build_gf_[b]ackground_gmm.py|evaluate_[p]ure.py" \
             >/dev/null && echo yes || echo no)
CHORES=$(pgrep -f "lit_dige[s]t.py|seat_thinkin[g].py|routine_[o]pen_questions.py|run_test[s].sh" \
         | wc -l)

echo "회차 실행: ${COUNCIL} · 실험 실행: ${EXPERIMENT} · 잡무 실행: ${CHORES}건 · 큐: ${QUEUED}건"

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
exit "$RC"
