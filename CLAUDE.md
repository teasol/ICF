# Claude Code Project Guidelines

@AGENTS.md

역할 정의·권한·연구 행동 원칙·서명 규칙의 정본은 위 `AGENTS.md`와 `docs/agent_handoff.md`입니다.
이 문서는 그 내용을 반복하지 않고, **Claude Code 도구 사용에 특화된 사항만** 담습니다.

## Research Multi-Agent Roster

| 이름 | 역할 | 배속 | 비고 |
|---|---|---|---|
| **Orca** (`orca`) | Reasoning Agent | Codex `gpt-6-astra` (effort `medium`) 영속 세션이 기본. 특정 세션에 고정되지 않으며, 컨텍스트 리셋 등 필요 시 Claude Code 로컬 서브에이전트(`.claude/agents/orca.md`, `claude-opus-5`)로도 수행 | `Agent(agent="orca", ...)` 또는 `bash scripts/call_agent.sh orca "..."` |
| **Owl** (`owl`) | Idea Agent | `claude-sonnet-5` (effort `medium`) | `Agent(agent="owl", ...)` 또는 `bash scripts/call_agent.sh owl "..."` |
| **Lime** (`lime`) | Coding Agent | `claude-sonnet-5` (effort `medium`) | `Agent(agent="lime", ...)` 또는 `bash scripts/call_agent.sh lime "..."` |
| Platform Agent | 운영·실행 | 고정 코드네임·모델 없음 | 필요할 때 지정하는 세션(현재 대화 세션 포함) |
| Document Agent | 문서 관리 | 고정 코드네임·모델 없음 | 필요할 때 지정하는 세션 |

실제 산출물에는 위 기본값이 아니라 **그 실행에서 실제로 쓰인 모델과 추론 강도**를 서명에 기록합니다
(`AGENTS.md` §4, `docs/agent_handoff.md` §0.3).

## 서명 — 절대 규칙 하나만 강조

**`Co-Authored-By` 등 공동 작성 표기는 어떤 산출물·커밋에도 쓰지 않습니다.** 실제로 작성한 주체를
단독 서명합니다. 형식과 예외는 `AGENTS.md` §4를 따릅니다.

## GitHub Authority

- 모든 에이전트는 `~/.gittoken_icf`를 통해 GitHub 저장소 권한을 완전히 보유합니다.
- 저장소 `.git/config`에 `credential.https://github.com.helper = store --file /home/kimds/.gittoken_icf`가 설정되어 있어 비대화형 `git fetch`, `git push`가 가능합니다.

## Shortcuts

- `/call <agent> <prompt>` : 서브에이전트(Orca, Owl, Lime) 호출 및 위임 (`.agents/skills/call/SKILL.md`)
- `/handoff-update` : 세션 종료 및 인수인계 문서 갱신
- `/resume-handoff` : 최신 상태 브리핑 및 세션 재개

## 연구 행동 원칙 및 위임 원칙

전문은 `AGENTS.md` §3(모든 역할의 연구 행동 원칙)을 따릅니다. Claude Code 세션에 특히 관련된 요지:
토큰·비용 효율을 위해 독립 작업은 Owl(아이디어)·Lime(구현·검증)에게 위임하고, 위임 시 [목표·필요한
맥락·수정 범위·완료 조건]만 압축 전달합니다. 위임 비용이 더 큰 사소한 조회·수정은 직접 처리하되,
회귀 테스트·실측 검증 등 필수 검증은 토큰 절약을 이유로 생략하지 않습니다.
