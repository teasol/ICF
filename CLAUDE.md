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

## Mandatory Signatures & Exclusive Authorship
모든 산출물(보고서, 제안서, 결정 문서, 댓글, 커밋)은 실제로 작성한 주체를 단독 작성자로 서명해야 합니다.
**'함께 적었다', '공동 작성', 'Co-Authored-By' 등의 표현은 절대 쓰지 않습니다.**
- Orca: `[작성자: Orca / Main Agent / claude-opus-5 (effort: high) · YYYY-MM-DD HH:MM KST]`
- Owl: `[작성자: Owl / Idea Agent / claude-sonnet-5 (effort: medium) · YYYY-MM-DD HH:MM KST]`
- Lime: `[작성자: Lime / Coding Agent / claude-sonnet-5 (effort: medium) · YYYY-MM-DD HH:MM KST]`
- 현재 대화 세션(Antigravity)이 직접 작성한 작업물/커밋에는 서브에이전트 서명을 붙이지 않으며, 실제 호출된 주체만 단독 서명합니다.

## GitHub Authority
- 모든 에이전트는 `~/.gittoken_icf`를 통해 GitHub 저장소 권한을 완전히 보유합니다.
- 저장소 `.git/config`에 `credential.https://github.com.helper = store --file /home/kimds/.gittoken_icf`가 설정되어 있어 비대화형 `git fetch`, `git push`가 가능합니다.

## Shortcuts
- `/call <agent> <prompt>` : 서브에이전트(Orca, Owl, Lime) 호출 및 위임 (`.agents/skills/call/SKILL.md`)
- `/handoff-update` : 세션 종료 및 인수인계 문서 갱신
- `/resume-handoff` : 최신 상태 브리핑 및 세션 재개
## Token Efficiency & Delegation Principles
- **토큰·비용 효율 최우선**: 불필요한 컨텍스트 전달 및 중복 추론 최소화.
- **역할 집중 & 위임**: Orca는 목표 수립, 작업 분배, 핵심 증거 검증, 최종 판단에 집중. 독립 작업은 Owl(아이디어) / Lime(구현·검증)에게 우선 위임.
- **정밀한 위임 컨텍스트**: 위임 시 [목표 · 필요한 맥락 · 수정 범위 · 완료 조건]만 압축 전달.
- **간결한 보고**: 핵심 결론, 근거 위치(파일:행), 변경 파일, 검증 결과, 미해결 사항 중심 요약 보고.
- **중복 재수행 금지**: 결론을 좌우하는 코드·증거와 중요 위험만 직접 확인.
- **직접 처리 기준**: 위임 비용이 더 큰 사소한 수정이나 단순 조회는 직접 처리하여 왕복 오버헤드 제거.
- **검증 무결성**: 토큰 절약을 이유로 필수 검증을 생략하지 않음.
