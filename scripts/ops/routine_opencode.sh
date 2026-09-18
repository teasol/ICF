#!/usr/bin/env bash
# Hand one narrow, mechanical edit to the local model, in a sandbox it cannot escape.
#
# Letting an agent write to the repository during idle slots is not a chore in
# the usual sense: chores produce lists nobody has to read, but edits have to be
# reviewed. Two guards make it safe enough to run unattended.
#
#   1. A separate git worktree. The main checkout is being read by council
#      rounds and experiments while this runs; editing it underneath them would
#      corrupt a round that is already in flight.
#   2. A protected set. Norm and decision documents are reverted after the run
#      even if the model touched them. Those files are changed by user decision
#      and recorded by hand, never by a background edit.
#
# Nothing is pushed and main is never checked out here. The result is a branch
# plus a diffstat for the orchestrator to review.
#
#   bash scripts/ops/routine_opencode.sh talks/ops/tasks/<task>.md
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
TASK="${1:?usage: routine_opencode.sh <task.md>}"
NAME="$(basename "$TASK" .md)"
BRANCH="chore/${NAME}-$(date +%m%d-%H%M)"
TREE="${ROOT}/../ICF.worktrees/chore-${NAME}"
MODEL="${ICF_OPENCODE_MODEL:-qwen-gpu6/qwen3.8-27b}"

# Changed only by user decision and recorded by hand. A background edit here
# would rewrite the project's canon without anyone deciding anything.
PROTECTED=(AGENTS.md docs/PROJECT.md docs/closed_axes.md docs/history/archive.md)

[ -f "$TASK" ] || { echo "작업 파일 없음: $TASK" >&2; exit 2; }

rm -rf "$TREE"
git worktree prune
git worktree add -b "$BRANCH" "$TREE" HEAD >/dev/null 2>&1 || {
  echo "worktree 생성 실패" >&2; exit 2; }
echo "=== ${NAME} · 브랜치 ${BRANCH} · 모델 ${MODEL} ==="

( cd "$TREE" && timeout 3000 opencode run --model "$MODEL" "$(cat "$ROOT/$TASK")" ) \
  2>&1 | tail -25
RC=$?

cd "$TREE"
for f in "${PROTECTED[@]}"; do
  if ! git diff --quiet -- "$f" 2>/dev/null; then
    echo "!! 보호 대상 ${f} 이 수정되어 되돌린다"
    git checkout -- "$f"
  fi
done

if git diff --quiet && [ -z "$(git status --porcelain)" ]; then
  echo "변경 없음 — 브랜치를 제거한다"
  cd "$ROOT"; git worktree remove --force "$TREE"; git branch -D "$BRANCH" >/dev/null 2>&1
  exit 0
fi

git add -A
git commit -q -m "chore(${NAME}): opencode 자동 편집 (미검토)

작업 지시: ${TASK}
모델: ${MODEL}
이 커밋은 검토되지 않았다. main에 병합하기 전에 오케스트레이터가 확인한다."
echo "--- 변경 요약 ---"
git diff --stat HEAD~1
echo "--- 검토 대기: 브랜치 ${BRANCH}, worktree ${TREE} ---"
exit "$RC"
