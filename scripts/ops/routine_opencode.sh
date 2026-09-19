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
# 2026-09-19: the model argument used to be an OpenCode *provider* string
# naming an external service. With no project config OpenCode resolved it over
# HTTPS; copying llm.local.json did not change that, because nothing in
# OpenCode reads that file. The run below writes opencode.json pointing at the
# local vLLM and fails closed if that server is not serving the model. There is
# no fallback path.
#
#   bash scripts/ops/routine_opencode.sh talks/ops/tasks/<task>.md
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
TASK="${1:?usage: routine_opencode.sh <task.md>}"
[ -f "$TASK" ] || { echo "작업 파일 없음: $TASK" >&2; exit 2; }
TASK_PATH="$(realpath "$TASK")"
NAME="$(basename "$TASK_PATH" .md)"
BRANCH="chore/${NAME}-$(date +%m%d-%H%M)"
TREE="${ROOT}/../ICF.worktrees/chore-${NAME}"

# Changed only by user decision and recorded by hand. A background edit here
# would rewrite the project's canon without anyone deciding anything.
PROTECTED=(AGENTS.md docs/PROJECT.md docs/closed_axes.md docs/history/archive.md)

cleanup_tree() {
  git worktree remove --force "$TREE" >/dev/null 2>&1
  git branch -D "$BRANCH" >/dev/null 2>&1
}

rm -rf "$TREE"
git worktree prune
git worktree add -b "$BRANCH" "$TREE" HEAD >/dev/null 2>&1 || {
  echo "worktree 생성 실패" >&2; exit 2; }

# git-ignore 되는 기계별 override 는 새 worktree 로 따라오지 않는다. LLM 주소가 그 파일에만
# 있는 기계(Slurm 로그인 노드)에서는 위임받은 모델이 자기 서버에 못 닿아 "No route to host"
# 를 보고 그것을 결과에 적는다. 있으면 복사한다.
[ -f "$ROOT/talks/ops/llm.local.json" ] && cp "$ROOT/talks/ops/llm.local.json" "$TREE/talks/ops/"

# Write the OpenCode project config that actually points at the local server,
# and print the provider/model string it defines. No external provider remains
# on this path: the string is `icf-local/<model>`.
MODEL="$(.venv/bin/python scripts/ops/opencode_config.py --write "$TREE")" || {
  echo "로컬 OpenCode 설정 생성 실패" >&2; cleanup_tree; exit 2; }

# Fail closed. If the local endpoint is not up and serving the model, stop --
# do not let OpenCode reach for a provider over the internet.
.venv/bin/python -c '
import sys; sys.path.insert(0, "scripts/ops")
import llm_env
print("엔드포인트 " + llm_env.base_url() + " · 모델 " + llm_env.require_local_model())
' || {
  echo "로컬 모델 확인 실패 — 외부 provider 로 대체하지 않는다" >&2
  cleanup_tree; exit 2; }
echo "=== ${NAME} · 브랜치 ${BRANCH} · 모델 ${MODEL} ==="

( cd "$TREE" && timeout ${ICF_OPENCODE_TIMEOUT:-5400} opencode run --model "$MODEL" "$(cat "$TASK_PATH")" ) \
  2>&1 | tail -25
RC=$?

# OpenCode가 비정상 종료했는데 변경이 없으면 성공으로 덮어쓰지 않는다. 이전 구현은
# 아래의 "변경 없음" 분기에서 0을 반환해 wrapper가 실패한 작업을 completed로 기록했다.
if [ "$RC" -ne 0 ]; then
  echo "OpenCode 실패(rc=$RC) — 작업을 완료로 기록하지 않는다" >&2
  cd "$ROOT"; cleanup_tree
  exit "$RC"
fi

cd "$TREE"
for f in "${PROTECTED[@]}"; do
  if ! git diff --quiet -- "$f" 2>/dev/null; then
    echo "!! 보호 대상 ${f} 이 수정되어 되돌린다"
    git checkout -- "$f"
  fi
done

if git diff --quiet && [ -z "$(git status --porcelain)" ]; then
  echo "변경 없음 — 브랜치를 제거한다"
  cd "$ROOT"; cleanup_tree
  exit 0
fi

git add -A
git commit -q -m "chore(${NAME}): opencode 자동 편집 (미검토)

작업 지시: ${TASK_PATH}
모델: ${MODEL} (로컬 vLLM)
이 커밋은 검토되지 않았다. main에 병합하기 전에 오케스트레이터가 확인한다."
echo "--- 변경 요약 ---"
git diff --stat HEAD~1
echo "--- 검토 대기: 브랜치 ${BRANCH}, worktree ${TREE} ---"
exit "$RC"
