# 딥시크-로그인 서버 운영 구조 진단 및 재구축

사용자 지시: **현재 딥시크 서버 토론 구조와 로그인 서버 실험 구조에서 기존 서브에이전트
구조가 가지는 문제점을 파악하고, 전체 운영 구조를 재구축하라.** 단순 보고로 끝내지 말고,
확인된 운영 결함을 이 worktree에서 구현·테스트까지 완료하라.

## 현재 토폴로지

- 이 저장소는 `nexgem` Slurm 로그인 노드에 있다. 로그인 노드에는 GPU가 없고 실험은
  `sbatch`로 계산 노드에 제출한다.
- 딥시크 vLLM 서버는 별도 NHN `NEXGEM` 호스트의 GPU 4\~7에서 돈다. 로그인 노드는 SSH
  터널의 `192.168.100.100:8000`을 통해 접근한다.
- 로컬 모델 설정 정본은 `talks/ops/llm.local.json` 또는 `talks/ops/llm.json`이며,
  `scripts/ops/llm_env.py`가 읽는다.
- 토론·잡무·위임의 핵심 코드는 `scripts/ops/`, 상태와 작업 명세는 `talks/ops/`에 있다.
- 이 작업은 격리된 `chore/deepseek_ops_rebuild-*` 브랜치와 worktree에서 수행된다. `main`을
  직접 수정하거나 push하지 마라.

## 이미 실측된 결함 — 반드시 재확인할 것

1. `talks/ops/tasks/*.md`는 모니터에만 보이고 실제 supervisor queue로 들어가는 연결이 없다.
2. task 착수 판정이 현재 `chore/<name>-*` 브랜치 존재만 보므로, 병합 뒤 브랜치를 지우면
   완료 작업이 다시 `미착수`로 보인다.
3. `routine_opencode.sh`의 `deepseek/deepseek-v4.1-flash`는 로컬 vLLM이 아니라 외부 HTTPS
   provider를 사용했다. `llm.local.json` 복사만으로 OpenCode provider는 바뀌지 않는다.
4. supervisor가 살아 있고 recurring 작업이 등록돼 있어도, 대부분의 시간 vLLM 요청은 0건이다.
   주기 CPU 검사 존재와 딥시크 작업 공급을 혼동한다.
5. 실패 카드는 `done/`으로 이동하므로 자동 재시도는 막히지만, pending/running/completed/failed
   상태가 하나의 명시적 상태 머신으로 관리되지 않는다.
6. 로그인 노드의 실험, Slurm 작업, 원격 vLLM 요청, 로컬 CPU 잡무를 하나의 `busy` 불리언으로
   섞어 자원별 유휴·충돌을 정확히 표현하지 못한다.

## 재구축 목표

현재 구현을 읽고 위 결함이 실제인지 코드·로그로 확인한 뒤, 다음을 만족하는 최소한의 운영
구조로 재구축하라.

1. **명시적 작업 수명주기**: one-shot 위임 작업을 `pending / running / completed / failed`로
   구분하고, 완료는 삭제 가능한 브랜치가 아니라 커밋 또는 영속 상태 기록으로 판별한다.
2. **로컬 서버 강제**: 모든 자동 위임은 `llm_env`가 정한 로컬 OpenAI-compatible endpoint를
   사용한다. 실행 전에 `/v1/models`를 확인하고 실제 선택된 provider·base URL·model을 로그에
   남긴다. 외부 provider로 조용히 fallback하지 않는다.
3. **자원별 스케줄링**: `remote_llm`, `slurm`, `login_cpu`, `exclusive_gpu`를 구분한다. 서로 다른
   자원 작업은 병행할 수 있고, 같은 격리 worktree나 같은 배타 자원을 충돌시키지 않는다.
4. **진실한 모니터**: vLLM 실측 running/waiting/token rate, 작업별 상태·최근 오류·산출 branch,
   supervisor heartbeat를 보여준다. 모르는 값은 0이 아니라 `모름`으로 둔다.
5. **복구 가능성**: supervisor 재시작 뒤 running 상태를 잃거나 같은 작업을 중복 발사하지 않는다.
   실패 자동 무한 재시도는 금지하고, 명시적 retry 절차를 제공한다.
6. **토론과 실행의 경계**: 협의체 회차는 supervisor가 자동 선행 발사하지 않는다. 결과를 읽은
   뒤 다음 질문을 정해야 한다는 현행 앵커링 방지 원칙을 보존한다.
7. **작업 공급 부족 가시화**: 딥시크가 비었는데 실행 가능한 task가 없으면 정상처럼 숨기지 말고
   `work-starved` 상태와 원인을 모니터·상태 출력에 명시한다.

## 변경 범위

- 수정 가능: `scripts/ops/`, `tests/test_ops_*`, `tests/test_queue_monitor.py`, `talks/ops/`의
  운영 스키마·템플릿·운영 설명.
- 보호 대상: `AGENTS.md`, `docs/PROJECT.md`, `docs/closed_axes.md`, `docs/history/archive.md`는
  수정하지 마라.
- 연구 모델 수치 경로인 `src/models/`, 평가 알고리즘, benchmark 결과는 수정하지 마라.
- 새 외부 Python 의존성을 추가하지 마라. 표준 라이브러리를 우선 사용하라.

## 완료 조건

1. 현행 결함과 새 구조의 상태 전이를 짧은 보고서 `talks/ops/reports/deepseek_ops_rebuild.md`에
   기록한다. 실측과 설계 판단을 구분한다.
2. queue → running → completed 및 queue → running → failed 전이를 임시 디렉터리에서 검증하는
   단위 테스트를 추가한다.
3. 병합 완료된 기존 작업은 재발사하지 않고, 신규 미완료 작업은 정확히 한 번 큐에 들어가는
   테스트를 추가한다.
4. OpenCode가 로컬 base URL과 모델을 사용하도록 만든 설정을 테스트하고, 외부 provider 문자열이
   실행 경로에 남아 있지 않음을 확인한다.
5. 관련 ops 테스트와 `git diff --check`를 실행한다. 저장소 전체 회귀의 기존 `RU-91`\~`96`
   공백 실패는 별도 기존 결함으로 보고하고, 새 실패와 구분한다.
6. 마지막 출력에 변경 파일, 테스트 결과, 남은 위험, 생성한 브랜치명을 적는다.

빠르게 구현하되 확인하지 않은 사실을 검증됐다고 쓰지 마라. 중대한 선택은 보고서에 대안과
이유를 남기고, 보호 문서나 연구 판정은 건드리지 마라.
