# Agent Instructions & Handoff Protocol

이 저장소는 멀티 에이전트(Antigravity, Claude Code, GPT Codex, OpenCode)와 멀티 디바이스 환경 간 작업 인수인계를 위해 **Universal Handoff Protocol**을 따릅니다.

## 1. 세션 시작 (Resume Handoff)
사용자가 "이어서 시작하자", "핸드오프 받아줘", "resume", "어디까지 했지" 등으로 작업을 시작할 때:
1. `git fetch origin`을 수행하여 원격 최신 변경사항을 확인하고 안전하게 동기화합니다.
2. `docs/agent_handoff.md` (불변 설계/아키텍처) 및 `docs/current_status.md` (현재 작업 상태)를 확인합니다.
3. 현재 실행 환경(호스트명, GPU, conda 환경)이 문서에 기록된 환경과 다른 경우(예: GPU 메모리 차이, 패키지 환경 불일치) 사용자에게 고지합니다.
4. 파일 전체를 나열하지 말고, **[현재 단계 / 상태 (CLEAN or WIP) / 블로커 / 바로 실행할 다음 명령어]** 4가지 핵심을 5줄 내외로 요약 브리핑합니다.

## 2. 세션 종료 (Handoff Update)
사용자가 "퇴근", "자리 옮길게", "핸드오프 업데이트", "세션 마무리" 등을 요청할 때:
1. `git status`와 `git diff`를 확인하고 변경 사항을 정리합니다.
2. `docs/current_status.md`의 헤더(호스트명, Conda 환경, GPU, Status: CLEAN/WIP)를 갱신하고 새 날짜 섹션을 추가합니다.
3. 반드시 **`Immediate Next Command`**에 다음 세션에서 바로 실행해야 할 명령어를 1줄로 명시합니다.
4. 해결된 오래된 섹션은 `docs/history/archive.md`로 이동하여 `current_status.md`를 150줄 이내로 유지합니다.
5. 대용량 가중치 파일(`*.pt`, `*.ckpt` 등)이나 비밀키(`.env`)가 스테이징되지 않도록 주의하며, 코드 커밋과 문서 커밋을 단계별로 분리하여 원격 저장소에 `push`합니다.

## 3. 효율성 및 안전 제약
- 저장소 전체를 무차별 검색하지 않고 필요한 파일만 읽습니다.
- 대용량 로그나 체크포인트 파일을 깃에 커밋하지 않습니다.
- 테스트되지 않은 미완성 코드를 푸시할 때는 반드시 `wip: ...` 접두사를 붙여 다음 에이전트가 인지할 수 있게 합니다.
