# Agent Instructions & Handoff Protocol

이 저장소는 멀티 에이전트(Antigravity, Claude Code, GPT Codex, OpenCode)와 멀티 디바이스 환경 간 작업 인수인계를 위해 **Universal Handoff Protocol**을 따릅니다.

## 0. 먼저 읽는 것 — 보편 연구 규범

**연구·실험·분석에 착수하기 전에
[`/home/kimds/agent_rules/research_protocol.md`](/home/kimds/agent_rules/research_protocol.md)(v3.0)를
읽습니다.** 모든 프로젝트 공용 정본이며, 이 저장소의 규범은 그것을 **상속하고 좁힌 것**입니다.

그다음 [`docs/agent_handoff.md`](docs/agent_handoff.md)의 **맨 위 절(이 저장소의 적용 규칙
R1~R7)** 과 [§0 목표 중심 연구 원칙](docs/agent_handoff.md#0-목표-중심-연구-원칙)을 읽습니다.

요지 세 가지:
- **RU 유형을 먼저 정합니다** (탐색 / 진단 / 확증 / 재현·계측). **탐색은 허용되며**,
  실패는 탐색을 한 것이 아니라 **탐색을 확증이라고 부른 것**입니다.
- **성능 관측을 기전의 입증으로 바꾸어 쓰지 않습니다.** 기전이 불명확하면 `기전 미확인`을 적습니다.
- **증거 판정**(지지/반박/판별 불가/실행 무효)과 **운영 결정**(채택/추가 진단/보류/종료)을
  분리해 적습니다.

원칙의 정본은 위 두 문서이며, 현재 목표의 완료 조건은 `docs/PROJECT.md`, 진행 상태·후속 행동은 `docs/current_status.md`에 기록합니다.

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
   규범·기준·구성이 바뀌었다면 [`docs/decisions.md`](docs/decisions.md)에 결정 레코드를,
   축을 닫거나 열었다면 `closed_axes.md`에 반영합니다.
4. 커밋 메시지 본문에 RU ID를 적습니다 (`feat(sh): ... (RU-79)`).

## 3. 세션 종료 (Handoff Update)
사용자가 "퇴근", "자리 옮길게", "핸드오프 업데이트", "세션 마무리" 등을 요청할 때:
1. `git status`와 `git diff`를 확인하고 변경 사항을 정리합니다.
2. `docs/current_status.md`의 헤더(호스트·환경·Status)를 **실측값으로** 갱신하고,
   `Immediate Next Command`에 다음 세션에서 바로 실행할 명령어를 명시합니다.
3. **`current_status.md`는 80줄을 넘기지 않습니다.** 종료된 절은
   `docs/history/archive.md`로, 결정은 `docs/decisions.md`로 옮깁니다.
4. **정정은 본문을 덮어씁니다.** 낡은 서술에 주석을 덧붙여 누적하지 않고, 시점 한정 사실과
   번복 이력은 `docs/decisions.md`에만 남깁니다.
5. `bash scripts/run_tests.sh`로 회귀 스위트(문서 정합성 테스트 포함)를 통과시킨 뒤 커밋합니다.
6. 대용량 가중치(`*.pt`, `*.ckpt` 등)나 비밀키(`.env`)가 스테이징되지 않도록 주의하며,
   코드 커밋과 문서 커밋을 분리하여 원격 저장소에 `push`합니다.

## 4. 효율성 및 안전 제약
- 저장소 전체를 무차별 검색하지 않고 필요한 파일만 읽습니다.
- 대용량 로그나 체크포인트 파일을 깃에 커밋하지 않습니다.
- 테스트되지 않은 미완성 코드를 푸시할 때는 반드시 `wip: ...` 접두사를 붙입니다.
- **GPU 실행을 폴링으로 기다리지 않습니다.** `scripts/run_and_wait.sh`를 백그라운드로 한 번만
  실행합니다 ([`docs/agent_handoff.md` §6](docs/agent_handoff.md)).
