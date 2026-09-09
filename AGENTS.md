# Agent Instructions & Handoff Protocol

이 저장소는 멀티 에이전트(Antigravity, Claude Code, GPT Codex, OpenCode)와 멀티 디바이스 환경 간 작업 인수인계를 위해 **Universal Handoff Protocol**을 따릅니다.

## 0. 먼저 읽는 것 — 보편 연구 규범

**연구·실험·분석에 착수하기 전에
[`/home/kimds/agent_rules/research_protocol.md`](/home/kimds/agent_rules/research_protocol.md)(v3.0)를
읽습니다.** 모든 프로젝트 공용 정본이며, 이 저장소의 규범은 그것을 **상속하고 좁힌 것**입니다.

그다음 [`docs/agent_handoff.md`](docs/agent_handoff.md)의 **맨 위 절(이 저장소의 적용 규칙
R1~R7)** 과 [§0 목표 중심 연구 원칙](docs/agent_handoff.md#0-목표-중심-연구-원칙)을 읽습니다.

요지 네 가지:
- **RU 유형을 먼저 정합니다** (탐색 / 진단 / 확증 / 재현·계측). **탐색은 허용되며**,
  실패는 탐색을 한 것이 아니라 **탐색을 확증이라고 부른 것**입니다.
- **연구 방향 없는 수치 최적화(실패 양식 B)를 차단하고 창의적 구조 탐색(보편 규범 §2.3)을 실천합니다.**
  '점수를 높인다'만으로 연구 질문을 대신하지 않으며, 불확실한 새로운 접근에도 최소 구현 프로브와 중단 조건을 배정해 검증 기회를 보장합니다.
- **성능 관측을 기전의 입증으로 바꾸어 쓰지 않습니다.** 기전이 불명확하면 `기전 미확인`을 적습니다.
- **증거 판정**(지지/반박/판별 불가/실행 무효)과 **운영 결정**(채택/추가 진단/보류/종료)을
  분리해 적습니다.

원칙의 정본은 위 두 문서이며, 현재 목표의 완료 조건은 `docs/PROJECT.md`, 진행 상태·후속 행동은 `docs/current_status.md`에 기록합니다.

## 0.1 연구 에이전트 계층 및 서명 규범 (Research Multi-Agent Hierarchy & Roster)

연구 계층의 정본은 [`/home/kimds/agent_rules/research_hierarchy.md`](/home/kimds/agent_rules/research_hierarchy.md)입니다.
ICF 프로젝트에는 다음 세 에이전트가 배속되어 역할을 엄격히 분담합니다:

| 역할 (Role) | 코드네임 | 모델 | 추론 강도 | 주 책임 | 금지 사항 ("하지 않는다") |
|---|---|---|---|---|---|
| **Main Agent** (핸들) | **Orca** (`orca`) | `claude-opus-5` | `high` | 방향 수립, 자원 배분, 증거 판정, 사용자 소통, 서브에이전트(Owl, Lime) 호출 및 총괄 | 혼자 아이디어를 만들지 않는다. 혼자 구현의 옳고 그름을 판단하지 않는다. |
| **Idea Agent** (엑셀) | **Owl** (`owl`) | `claude-sonnet-5` | `medium` | 전제 의심, 새 가능성·대안 가설·반증 조건 제안, 제안서(`docs/proposals/`) 직접 작성 | 자기 아이디어를 스스로 죽이지 않는다. 채택·예산·판정하지 않는다. 코드 작성·실행 금지. 정본 문서 수정 금지. |
| **Coding Agent** (브레이크) | **Lime** (`lime`) | `claude-sonnet-5` | `medium` | 사양대로 구현, 실제 실행 검증, 비용 및 제약 실측, 제약 보고 ("안 된다"는 수정 요청) | 아이디어를 만들지 않는다. 후보를 임의로 기각하지 않는다. 조용히 고치지 않는다. 기준 사후 변경 금지. 추정을 실측에 놓지 않는다. GPU 폴링 금지. |

### 상호작용 및 호출 규칙
1. **판정은 Main(Orca)만 내린다.** Idea(Owl)의 확신도 Coding(Lime)의 실패도 그 자체로는 결론이 아닙니다.
2. **에이전트 호출 및 위임 (`/call` 스킬, `.agents/skills/call/SKILL.md`)**:
   - 호출 구문 (대소문자 무관): `/call <agent>`, `to <agent>`, `<agent>에게`, `<agent> 호출`
   - 위 구문이 입력되면 현재 세션의 어시스턴트는 임의로 답변을 가로채지 않고, 반드시 `bash scripts/call_agent.sh <agent_lower> "<prompt>"`를 실행하여 실제 에이전트의 출력을 그대로 전달합니다.
   - Claude Code 내부: 내장 `Agent` 도구 (`Agent(agent="owl", ...)` / `Agent(agent="lime", ...)`)
   - CLI 래퍼: `bash scripts/call_agent.sh owl "<prompt>"` / `bash scripts/call_agent.sh lime "<prompt>"` / `bash scripts/call_agent.sh orca "<prompt>"`
3. **역할은 별도의 에이전트 문맥으로 분리한다.** 한 세션에서 이름만 바꾸어 번갈아 수행하지 않습니다.
4. **동의는 검증이 아니다.** 셋이 같은 말을 해도 확인된 것이 아니며 독립된 대조·재계산·반례만이 확인입니다.

### 산출물 필수 서명 규범 (Signature Rule)
모든 에이전트는 자신이 작성하는 모든 작업물(제안서, 검증 보고, 결정 문서, 댓글, 커밋 등)에 **이름, 모델, 추론 강도, 작성일시**를 반드시 명기해야 합니다:
- **Orca**: `[작성자: Orca / Main Agent / claude-opus-5 (effort: high) · YYYY-MM-DD HH:MM KST]`
- **Owl**: `[작성자: Owl / Idea Agent / claude-sonnet-5 (effort: medium) · YYYY-MM-DD HH:MM KST]`
- **Lime**: `[작성자: Lime / Coding Agent / claude-sonnet-5 (effort: medium) · YYYY-MM-DD HH:MM KST]`
- **작성 주체 단독 명기 (Co-Author 표기 절대 금지)**:
  - '함께 적었다', '공동 작성', 'Co-Authored-By' 등의 모호한 표현은 절대 사용하지 않습니다.
  - 모든 작업물과 커밋에는 실제로 작성한 주체를 단독 작성자(`작성자: ...`)로 명확히 적습니다.
  - 현재 대화 세션(Antigravity)이 직접 작성한 내용·커밋에는 서브에이전트 이름을 붙이지 않습니다.
  - 실제 호출되어 작업을 수행한 에이전트(Orca, Owl, Lime)가 있을 때만 해당 에이전트를 단독 작성자로 명시합니다.

### GitHub 접근 권한 (GitHub Authority)
- 모든 에이전트는 `~/.gittoken_icf`에 저장된 GitHub Personal Access Token을 통해 저장소 푸시/풀 권한을 완전히 보유합니다.
- 저장소 로컬 설정(`.git/config`)에 `credential.https://github.com.helper = store --file /home/kimds/.gittoken_icf`가 구성되어 있어 비대화형 실행이 보장됩니다.


### 토큰·비용 효율 및 위임 원칙 (Token Efficiency & Delegation Principles)
- **토큰·비용 효율을 우선한다.** 불필요한 컨텍스트 비대화와 중복 추론을 방지합니다.
- **Orca의 집중**: Orca는 목표 설정, 작업 분배, 핵심 증거 확인과 최종 판단에 집중합니다.
- **우선 위임**: 독립적으로 맡길 수 있는 작업은 Idea Agent(`owl`), Coding Agent(`lime`)에게 우선 위임합니다.
- **정밀한 컨텍스트 전달**: 위임할 때는 목표·필요한 맥락·수정 범위·완료 조건만 압축 전달합니다. 전체 대화나 문서를 불필요하게 반복 복사하지 않습니다.
- **간결한 보고 수신**: 결과는 핵심 결론, 근거 위치(파일:행), 변경 파일, 검증 결과, 미해결 사항 중심으로 짧게 보고받습니다.
- **중복 재수행 방지**: Orca는 위임한 작업을 전부 재수행하지 않습니다. 결론을 좌우하는 핵심 코드·증거와 중요한 위험만 직접 확인합니다.
- **직접 처리 기준**: 간단한 답변이나 작은 수정처럼 위임 비용이 더 큰 작업은 Orca가 직접 처리합니다. 불필요한 다중 검토와 반복 호출을 피합니다.
- **검증 무결성 유지**: 토큰 절약을 이유로 필수 검증(회귀 테스트, 실측 계측)을 생략하거나, 승인된 작업을 미완료로 끝내지 않습니다.

## 1. 세션 시작 (Resume Handoff)
사용자가 "이어서 시작하자", "핸드오프 받아줘", "resume", "어디까지 했지" 등으로 작업을 시작할 때:
1. `git fetch origin`을 수행하여 원격 최신 변경사항을 확인하고 안전하게 동기화합니다.
2. [`docs/PROJECT.md`](docs/PROJECT.md) → [`docs/current_status.md`](docs/current_status.md) →
   [`docs/agent_handoff.md`](docs/agent_handoff.md) 순으로 읽습니다. 문서 지도는
   [`docs/README.md`](docs/README.md)에 있습니다.
3. 현재 실행 환경(호스트명, GPU, 드라이버, 파이썬 환경)을 **실측**하여 문서 기록과 다르면
   사용자에게 고지합니다. 문서에 적힌 과거 GPU 수·노드명을 그대로 신뢰하지 않습니다.
4. 파일 전체를 나열하지 말고, **[현재 목표 / 상태 (CLEAN or WIP) / 블로커 / 바로 실행할 다음 명령어]**
   4가지를 5줄 내외로 요약 브리핑합니다.

## 2. 작업 진행 (RU 단위)
진행 단위는 커밋이 아니라 **연구 단위(RU)**입니다. 절차의 정본은
[`docs/agent_handoff.md` §1](docs/agent_handoff.md)입니다.
1. **연구 실험인지 먼저 판별해 밝힙니다.** 문서 편집·환경 점검 등 면제와 카드 초안 절차는
   `docs/agent_handoff.md` §1.0을 따릅니다. 연구 실험은 **착수 전** `python3 scripts/docs/ru.py open --title "<제목>"`으로 카드를 만들고
   질문·가설·판정 기준·예산·**중단 조건**·결과별 후속 행동 여섯 필드를 채웁니다.
   하나라도 비어 있으면 착수하지 않습니다.
2. 제안이나 후보를 심사하기 전에 [`docs/closed_axes.md`](docs/closed_axes.md)의 **경계**를
   확인합니다. PSW·TGW / FC / P3-LIMIT-CURVE / LSAK는 비저촉으로 확정됐습니다(`D-022`).
   새로운 경계 판단이 갈리면 임의로 한쪽을 채택하지 않고 같은 문서 §3에 올립니다.
3. **종료 시** `python3 scripts/docs/ru.py close --id RU-xx`로 총람에 append하고,
   규범·기준·구성이 바뀌었다면 [`docs/history/archive.md`](docs/history/archive.md)의
   **결정 이력** 절에 레코드를, RU 종료만으로는 레코드를 만들지 않고,
   축을 닫거나 열었다면 `closed_axes.md`에 반영합니다.
4. 커밋 메시지 본문에 RU ID를 적습니다 (`feat(sh): ... (RU-79)`).

## 3. 세션 종료 (Handoff Update)
사용자가 "퇴근", "자리 옮길게", "핸드오프 업데이트", "세션 마무리" 등을 요청할 때:
1. `git status`와 `git diff`를 확인하고 변경 사항을 정리합니다.
2. `docs/current_status.md`의 헤더(호스트·환경·Status)를 **실측값으로** 갱신하고,
   `Immediate Next Command`에 다음 세션에서 바로 실행할 명령어를 명시합니다.
3. **`current_status.md`는 100줄을 넘기지 않습니다.** 종료된 절은
   `docs/history/archive.md`로 옮깁니다 (결정 이력도 같은 파일 말미에 있습니다).
4. **정정은 본문을 덮어씁니다.** 낡은 서술에 주석을 덧붙여 누적하지 않고, 시점 한정 사실과
   번복 이력은 `docs/history/archive.md`의 결정 이력에만 남깁니다.
   **living 문서의 각 항목은 자기완결적이어야 합니다** — 링크를 타고 들어가지 않아도
   무엇을 어떻게 할지 알 수 있게 씁니다. 이력 ID는 출처 표기로만 씁니다.
5. `bash scripts/run_tests.sh`로 회귀 스위트(문서 정합성 테스트 포함)를 통과시킨 뒤 커밋합니다.
6. 대용량 가중치(`*.pt`, `*.ckpt` 등)나 비밀키(`.env`)가 스테이징되지 않도록 주의하며,
   코드 커밋과 문서 커밋을 분리하여 원격 저장소에 `push`합니다.

## 4. 효율성 및 안전 제약
- 저장소 전체를 무차별 검색하지 않고 필요한 파일만 읽습니다.
- 대용량 로그나 체크포인트 파일을 깃에 커밋하지 않습니다.
- 테스트되지 않은 미완성 코드를 푸시할 때는 반드시 `wip: ...` 접두사를 붙입니다.
- **GPU 실행을 폴링으로 기다리지 않습니다.** `scripts/run_and_wait.sh`를 백그라운드로 한 번만
  실행합니다 ([`docs/agent_handoff.md` §6](docs/agent_handoff.md)).
