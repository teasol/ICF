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
# OpenCode's built-in `deepseek/*` provider is the public HTTPS service. Merely
# copying llm.local.json into the worktree does not make OpenCode use our vLLM
# server. Build an explicit OpenAI-compatible provider from the same config the
# other ops scripts read, and fail before creating work if that endpoint is not
# reachable. This prevents a healthy public API call from masquerading as local
# DeepSeek utilisation.
LLM_CONFIG="$ROOT/talks/ops/llm.local.json"
[ -f "$LLM_CONFIG" ] || LLM_CONFIG="$ROOT/talks/ops/llm.json"
readarray -t LLM_VALUES < <("$ROOT/.venv/bin/python" - "$LLM_CONFIG" <<'PY'
import json, sys
from urllib.parse import urlsplit, urlunsplit

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
endpoint = cfg["endpoints"][0]
parts = urlsplit(endpoint)
path = parts.path
for suffix in ("/chat/completions", "/completions"):
    if path.endswith(suffix):
        path = path[:-len(suffix)]
        break
print(urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", "")))
print(cfg["model"])
PY
)
BASE_URL="${LLM_VALUES[0]:?LLM base URL missing}"
SERVER_MODEL="${LLM_VALUES[1]:?LLM model missing}"
MODEL="${ICF_OPENCODE_MODEL:-icf-vllm/$SERVER_MODEL}"
export OPENCODE_CONFIG_CONTENT="$("$ROOT/.venv/bin/python" - "$BASE_URL" "$SERVER_MODEL" <<'PY'
import json, sys

base_url, model = sys.argv[1:]
print(json.dumps({
    "model": f"icf-vllm/{model}",
    "provider": {
        "icf-vllm": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "ICF local vLLM",
            "options": {"baseURL": base_url, "apiKey": "none"},
            "models": {
                model: {
                    "id": model,
                    "name": f"{model} (local)",
                    "reasoning": True,
                    "tool_call": True,
                    "temperature": True,
                    "limit": {"context": 262144, "output": 32000},
                }
            },
        }
    },
}))
PY
)"

"$ROOT/.venv/bin/python" - "$BASE_URL" <<'PY'
import json, sys, urllib.request

url = sys.argv[1].removesuffix("/v1") + "/v1/models"
with urllib.request.urlopen(url, timeout=5) as response:
    json.load(response)
print(f"local-vllm-ok {url}")
PY

# Changed only by user decision and recorded by hand. A background edit here
# would rewrite the project's canon without anyone deciding anything.
PROTECTED=(AGENTS.md docs/PROJECT.md docs/closed_axes.md docs/history/archive.md)

[ -f "$TASK" ] || { echo "작업 파일 없음: $TASK" >&2; exit 2; }

rm -rf "$TREE"
git worktree prune
git worktree add -b "$BRANCH" "$TREE" HEAD >/dev/null 2>&1 || {
  echo "worktree 생성 실패" >&2; exit 2; }
echo "=== ${NAME} · 브랜치 ${BRANCH} · 모델 ${MODEL} · ${BASE_URL} ==="

# git-ignore 되는 기계별 override 는 새 worktree 로 따라오지 않는다. LLM 주소가 그 파일에만
# 있는 기계(Slurm 로그인 노드)에서는 위임받은 모델이 자기 서버에 못 닿아 "No route to host"
# 를 보고 그것을 결과에 적는다. 있으면 복사한다.
[ -f "$ROOT/talks/ops/llm.local.json" ] && cp "$ROOT/talks/ops/llm.local.json" "$TREE/talks/ops/"

( cd "$TREE" && timeout ${ICF_OPENCODE_TIMEOUT:-5400} opencode run --model "$MODEL" "$(cat "$ROOT/$TASK")" ) \
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
