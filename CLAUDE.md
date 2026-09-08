# Claude Code Project Guidelines

@AGENTS.md

## Research Multi-Agent Roster

| Name | Role | Model | Effort | Tools / Agent Definition |
|---|---|---|---|---|
| **Orca** (`orca`) | Main Agent (핸들) | `claude-opus-5` | `high` | `.claude/agents/orca.md` |
| **Owl** (`owl`) | Idea Agent (엑셀) | `claude-sonnet-5` | `medium` | `.claude/agents/owl.md` |
| **Lime** (`lime`) | Coding Agent (브레이크) | `claude-sonnet-5` | `medium` | `.claude/agents/lime.md` |

- 기본 인터랙티브 세션은 **Orca (Main Agent, `claude-opus-5`, effort: `high`)**로 실행됩니다 (`.claude/settings.json`).
- Orca는 Claude Code의 내장 `Agent` 도구를 통해 Owl과 Lime을 즉시 호출할 수 있습니다:
  - `Agent(agent="owl", prompt="...")`
  - `Agent(agent="lime", prompt="...")`
- 또는 셸 래퍼 스크립트를 사용할 수 있습니다:
  - `bash scripts/call_agent.sh owl "<prompt>"`
  - `bash scripts/call_agent.sh lime "<prompt>"`

## Mandatory Signatures
모든 산출물(보고서, 제안서, 결정 문서, 댓글, 커밋)은 본인의 역할과 모델 정보를 서명해야 합니다:
- Orca: `[작성자: Orca / Main Agent / claude-opus-5 (effort: high) · YYYY-MM-DD HH:MM KST]`
- Owl: `[작성자: Owl / Idea Agent / claude-sonnet-5 (effort: medium) · YYYY-MM-DD HH:MM KST]`
- Lime: `[작성자: Lime / Coding Agent / claude-sonnet-5 (effort: medium) · YYYY-MM-DD HH:MM KST]`

## GitHub Authority
- 모든 에이전트는 `~/.gittoken_icf`를 통해 GitHub 저장소 권한을 완전히 보유합니다.
- 저장소 `.git/config`에 `credential.https://github.com.helper = store --file /home/kimds/.gittoken_icf`가 설정되어 있어 비대화형 `git fetch`, `git push`가 가능합니다.

## Shortcuts
- `/handoff-update` : 세션 종료 및 인수인계 문서 갱신
- `/resume-handoff` : 최신 상태 브리핑 및 세션 재개
