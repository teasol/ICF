# 딥시크 운영 구조 재구축 — 결함 실측과 새 상태 전이

[작성자: OpenCode / 소집자 겸 실행자(사용자 단일 세션 지시) / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 11:05 KST]

브랜치: `chore/deepseek_ops_rebuild-0919-1052` (worktree `ICF.worktrees/chore-deepseek_ops_rebuild`)

이 문서는 **실측**과 **설계 판단**을 구분한다. 실측은 코드·실행 출력으로 확인한 것이고,
설계 판단은 대안과 이유를 적은 선택이다. 확인하지 않은 것은 확인했다고 쓰지 않는다.

---

## 1. 기존 결함 — 실측 재확인

| # | 주장 | 판정 | 근거 |
|---|---|---|---|
| 1 | `tasks/*.md`가 supervisor 큐로 연결되지 않음 | **실측 확인** | `supervisor.py`는 `QUEUE.glob("*.json")`과 `RECURRING`만 읽었다. `tasks/`를 읽는 코드는 `queue_monitor.read_tasks()`뿐이고 이는 표시용이었다. 카드를 추가해도 `queue/`에 카드가 생기지 않았다. |
| 2 | 착수 판정이 브랜치 존재에만 의존 | **실측 확인** | 기존 `read_tasks()`가 `chore/<name>-*` 브랜치를 찾았다. `main`에는 `docs_inventory`(`3c2ba38`)·`queue_monitor`(`1f7d698`) 등 병합 커밋이 있으나 브랜치는 삭제돼, 기존 로직은 이들을 `미착수`로 표시했다. |
| 3 | `routine_opencode.sh`가 외부 provider 사용 | **실측 확인** | 스크립트 26행이 `MODEL="${ICF_OPENCODE_MODEL:-deepseek/deepseek-v4.1-flash}"`였다. 저장소에 `opencode.json`이 없고, OpenCode는 `llm.local.json`을 읽지 않으므로 `llm.local.json` 복사는 provider 선택에 영향이 없었다. |
| 4 | supervisor 생존 ≠ 딥시크 작업 공급 | **실측 확인** | 보고 시점 `/metrics`: `running=0`, `waiting=0`, `generation_tokens_total` 정지. 등록된 주기 잡무 6종 중 `seat-thinking`·`open-questions`만 `remote_llm`이고 나머지 4종(pytest·provenance·lit_index·regression)은 `login_cpu`다. CPU 주기 검사가 "활동"을 만들지만 vLLM 요청은 0이다. |
| 5 | pending/running/completed/failed 상태 머신 부재 | **실측 확인** | `launch()`가 성공·실패 모두 카드를 `done/`으로 rename했고, 실패는 `failures.jsonl`에만 남았다. 상태 기록은 없었다. |
| 6 | 하나의 `busy` 불리언으로 자원 혼합 | **실측 확인** | `snapshot()`의 `busy = any(running.values())` — 로그인 노드 pytest와 GPU 실험이 같은 비트였다. |

부가 실측: 딥시크 서버는 이 로그인 노드에서 `http://192.168.100.100:8000/v1`로 **도달 가능**하며
`/v1/models`가 `deepseek-v4.1-flash`를 반환했다(`probe_models` → `ok=True`). 로컬 강제가 성립하는 환경이다.
로그인 노드에는 `nvidia-smi`가 없어 상태 문서의 GPU 줄은 빈 값으로 나온다(오류 아님).

---

## 2. 새 구조 — 설계 판단

### 2.1 명시적 작업 수명주기 (결함 1·2·5)

새 파일 `scripts/ops/task_state.py`가 정본 저장소 `talks/ops/task_state.json`을 관리한다.

```
pending → running → completed
                 → failed
failed  → pending            (오직 명시적 retry)
pending → completed          (병합 커밋이 사후 발견된 경우)
```

- 전이는 검증된다. 종료 상태를 덮어쓰려 하면 `InvalidTransition`이 발생한다. 빠른 작업이 supervisor의
  pid 기록보다 먼저 끝나도 `running`으로 되돌아가지 않는다.
- `scripts/ops/task_queue.py`가 `tasks/*.md`를 큐로 잇는다. 완료 판정은 **`main` 이력의
  `chore(<name>)` 커밋**(삭제 불가능한 증거)이며, 저장소에 상태 기록이 이미 있으면 그것을 우선한다.
  `sync`를 여러 번 호출해도 큐 카드는 한 장이다(큐 파일 존재가 멱등 가드).
- 실패는 자동 재시도하지 않는다. `task_queue.py retry <name>`이 유일한 복구 경로다.

**대안과 기각 이유**: 저장소를 git 추적 파일로 두는 안은 배경 프로세스가 추적 파일을 계속 수정해
`git status`를 오염시킨다. 커밋 이력 + git-ignore 런타임 저장소 조합을 택했다. 대안의 약점(기계 이전 시
`failed` 상태 소실)은 남은 위험에 적었다.

### 2.2 로컬 서버 강제 (결함 3)

- `llm_env.py`에 `base_url()`, `models_url()`, `probe_models()`, `require_local_model()`을 추가했다.
  `base_url()`은 `/chat/completions`를 잘라 OpenAI-compatible `/v1`을 만든다.
- `scripts/ops/opencode_config.py`가 worktree에 `opencode.json`을 생성한다. provider id는 `icf-local`,
  `baseURL`은 `llm_env`가 정한 로컬 주소, 모델 문자열은 `icf-local/<model>`이다.
- `routine_opencode.sh`는 이제 (1) 설정을 생성하고 (2) `require_local_model()`로 `/v1/models`를
  확인한 뒤 실패하면 **발사를 중단**한다. 대체 provider로 넘어가는 경로가 없다.
- supervisor도 시작 시 그리고 매 발사 전(60초 캐시)에 로컬 모델을 확인하고, 도달 불가면 `remote_llm`
  작업을 발사하지 않고 차단 사유를 tick에 남긴다.

**대안과 기각 이유**: `@ai-sdk/openai-compatible` 대신 `openai` provider를 `baseURL`만 바꿔 재사용하는
안은, 로그에 외부 provider 이름이 남아 "로컬"과 구분되지 않는다. 전용 provider id를 택했다.

### 2.3 자원별 스케줄링 (결함 6)

작업은 `remote_llm` / `slurm` / `login_cpu` / `exclusive_gpu` 중 하나를 갖는다(명세의 `resource`,
없으면 명령 문자열로 추론). 충돌 규칙:

- 같은 `worktree`를 쓰는 두 작업은 동시에 못 돈다(OpenCode 두 개가 한 트리를 편집하는 사고 방지).
- `exclusive_gpu`와 `slurm`은 서로 배타적이다.
- `login_cpu`와 `remote_llm`은 서로 및 다른 자원과 병행 가능하다.

### 2.4 복구 가능성 (결함 5·6 연장)

- 발사 전에 저장소에 `running`을 기록하고 저장한 뒤 프로세스를 띄운다.
- 실제 명령은 `scripts/ops/run_task.py`가 실행하고, 종료 시 스스로 `completed`/`failed`를 기록한다.
  supervisor가 중간에 죽어도 결과가 남는다.
- 재시작 시 `recover()`가 `running` 항목을 해소한다. 살아 있는 pid는 그대로 두고(그 wrapper가 결과를
  쓴다), 죽은 pid는 `failed`(결과 미상)로 표시한다. 자동 재발사하지 않는다.

### 2.5 진실한 모니터 + work-starved (결함 4·7)

- `queue_monitor.read_tasks()`는 저장소의 상태·자원·최근 오류를 표시한다. 브랜치 존재는 더 이상
  착수 판정이 아니다.
- `starvation()`이 `work-starved`를 계산한다. **서버가 도달 가능하고 `running=0, waiting=0`이며
  미착수 작업도 실행 중/대기 중 `remote_llm` 작업도 없을 때만** 선언한다. 모르는 값은 `0`이 아니라
  `모름`으로 그린다(`num()` 기본값 변경, 서버 도달 불가는 `모름`).
- supervisor의 `state.md` 실측 절과 tick 레코드에도 자원별 실행 수와 `work_starved` 사유를 넣는다.

### 2.6 토론·실행 경계 보존

supervisor의 `kind == "council"` 거부는 그대로다. 협의체 회차는 자동 선행 발사하지 않는다.

---

## 3. 검증

관련 ops 테스트(임시 디렉터리):

```
tests/test_ops_task_lifecycle.py  5 passed   (queue→running→completed / →failed, 종료 상태 불변, retry)
tests/test_ops_task_queue.py      5 passed   (병합 작업 미발사, 신규 정확히 1회, running 재발사 금지)
tests/test_ops_opencode_config.py 4 passed   (로컬 baseURL/model, 외부 provider 문자열 부재)
tests/test_queue_monitor.py      11 passed   (상태 표시, work-starved, 도달 불가=모름)
tests/test_ops_supervisor.py      8 passed   (자원 충돌, recover)
tests/test_llm_env.py             3 passed
```

실제 저장소 대상 `sync_tasks` 확인: 병합 커밋이 있는 8개 작업은 `completed`로 기록되어 큐에 들어가지
않았고, 미병합 `perf_transfer`만 `pending`으로 **정확히 한 장** 큐에 들어갔다. 두 번째 `sync`는 0건이었다.

전체 회귀: `pytest tests/ -q` → **256 passed, 16 skipped, 1 failed**. 실패 1건은
`test_docs_consistency.py::TestResearchUnitsLedger::test_ids_are_unique_and_contiguous`,
즉 **기존 결함인 `RU-91`\~`96` 대장 공백**이다(`docs/current_status.md`에도 기존 결함으로 기록됨).
이 재구축으로 생긴 새 실패는 없다.

`git diff --check` 및 `git diff --cached --check` → 위반 0.

문서 일관성 테스트가 테스트 수를 검사하므로, 새 테스트 추가에 맞춰 `docs/current_status.md`와
`docs/agent_handoff.md`의 `220 tests`를 `244 tests`로 갱신했다(보호 문서 아님).

---

## 4. 남은 위험

1. **OpenCode provider 패키지**: `@ai-sdk/openai-compatible`는 첫 사용 시 npm 설치가 필요할 수 있다.
   이 환경에서 `opencode run`을 실제로 실행해 검증하지는 않았다(`모름`). 최종 실행 검증은 orchestrator 몫이다.
2. **완료 판정이 커밋 메시지 규약에 의존**: `chore(<name>)` 형식이 아니면 병합을 감지하지 못한다.
   저장소의 `completed` 기록이 있으면 무관하지만, 새 기계/깨끗한 체크아웃에서는 규약이 유일한 증거다.
3. **동기화 주기**: supervisor는 30틱마다 `sync_tasks`를 돈다. 새 task 카드는 최대 약 30분 늦게 큐에 든다.
4. **`alive(pid)`의 pid 재사용**: 재시작 복구가 무관한 프로세스를 살아 있다고 볼 수 있다. 결과 미상
   실패를 택하므로 연구 산출을 오염시키진 않지만, 오탐 가능성은 남는다.
5. **`failed` 상태의 기계 간 비이식성**: 저장소가 git-ignore이므로 다른 기계로 옮기면 실패 이력이
   사라지고 미착수로 보일 수 있다(위 2의 저장소 기록이 있으면 완화).

---

## 5. 변경 파일

추가: `scripts/ops/task_state.py`, `scripts/ops/task_queue.py`, `scripts/ops/run_task.py`,
`scripts/ops/opencode_config.py`, `tests/test_ops_task_lifecycle.py`, `tests/test_ops_task_queue.py`,
`tests/test_ops_opencode_config.py`, 이 보고서.

수정: `scripts/ops/supervisor.py`, `scripts/ops/queue_monitor.py`, `scripts/ops/llm_env.py`,
`scripts/ops/routine_opencode.sh`, `talks/ops/recurring/*.json`(자원 필드),
`tests/test_ops_supervisor.py`, `tests/test_queue_monitor.py`, `.gitignore`,
`docs/current_status.md`, `docs/agent_handoff.md`(테스트 수).
