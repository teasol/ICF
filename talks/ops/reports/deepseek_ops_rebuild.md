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

---

## 6. 후속 요구 통합 (2026-09-19)

> 이 절은 위 1\~5절이 확정한 상태 머신·자원 스케줄링·수명주기 정본을 **보존한 채**
> 사용자가 추가로 요구한 모니터 표시·수집 경로를 얹은 결과다. 기존 전이는 건드리지
> 않았다.

### 6.1 오늘 생성 토큰 (M 단위) — 누적 카드 제거

- `vllm:generation_tokens_total`은 **프로세스 누적**이라 그대로 쓰면 "오늘"이 아니다.
  자정 기준선과 마지막 counter를 `talks/ops/token_baseline.json`(git-ignore)에 영속
  저장하고 **누적 차분**으로 오늘치를 만든다. 순수 함수 `advance_today`가 자정
  rollover와 counter reset을 처리하고, `format_tokens_m`이 M 단위 문자열을 만든다.
- **첫 관측이 자정 이후면 전체인 척하지 않는다.** 시작 시각을 기록하고
  `partial=True`로 표시하며, 화면에 "자정\~관측 시작 구간 미포함 · 관측 시작 ..."을
  적는다. 자정 정각부터 관측한 경우에만 `partial=False`다.
- **서버 재시작**으로 counter가 줄면 재기록한다. 리셋 이전에 측정해 둔 양은
  `accumulated`에 남기고 리셋 이후 counter를 더하므로, 오늘치가 음수로 가지 않는다.
- 누적 프롬프트 토큰(`vllm:prompt_tokens_total`)과 누적 생성 토큰은 **API dict와
  화면 양쪽에서 제거**했다(`build_server_payload`가 일부러 넣지 않는다).

### 6.2 GPU 4-7 전력·가동률 — 캐시 수집 경로

- 모니터는 `nexgem` 로그인 노드에서 돌고 GPU는 원격 NHN `NEXGEM`(GPU `4-7`)에 있다.
  이 기계에는 `nvidia-smi`가 없다. 브라우저가 2초마다 폴링하므로 **요청 경로에서
  SSH/nvidia-smi를 실행하지 않는다.**
- `scripts/ops/gpu_telemetry.py`가 TTL(기본 15초)마다 한 번 `ssh nhn`으로 원격
  `nvidia-smi`를 실행해 `talks/ops/gpu_telemetry_cache.json`에 원자적으로 쓴다.
  모니터는 데몬 스레드로 이 수집기만 돌리고, 요청 경로는 **캐시 파일만 읽는다**
  (`gpu_state` → `telemetry_state`). 설정은 `talks/ops/gpu.json`(추적)이,
  기계별 override는 `gpu.local.json`(git-ignore)이 정한다(D-053).
- **수집 실패·부분 누락·캐시 없음은 0이 아니라 `모름`/도달 불가로 그린다.** 파싱
  실패(`[N/A]`)와 장치 누락은 `None`이고, 실제 유휴의 `0 W`/`0 %`만 0으로 남는다.
  화면은 출처·수집 시각과 `(캐시 오래됨)` 표시를 함께 낸다.
- **이 환경에서 실측 확인**: `nvidia-smi --query-gpu=index,power.draw,power.limit,utilization.gpu`
  로 GPU 4\~7 각각 `~240 W`, 한도 `1000 W`, 가동률 `0 %`가 반환됐다(`ok=true`,
  `missing=[]`). 즉 현재 SSH 경로로 정확한 수치를 얻는다.
- **SSH가 막힐 때의 exporter 요구를 명시**했다(코드 docstring과 화면 안내). 대안은
  NEXGEM에 `nvidia_gpu_exporter`(`nvidia_smi_power_draw_watts`,
  `nvidia_smi_power_limit_watts`, `nvidia_smi_utilization_gpu_ratio`) 또는
  `dcgm-exporter`(`DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_POWER_MGMT_LIMIT`,
  `DCGM_FI_DEV_GPU_UTIL`)를 세우고 Prometheus scrape 뒤 `gpu.json`에
  `mode: "http"`, `exporter_url`을 적는 것이다. **HTTP 수집 모드는 아직 구현하지
  않았다** — 그 전까지는 도달 불가로 표시한다.

### 6.3 할 일과 완료 분리

- 화면은 task 파일 목록이 아니라 **명시적 수명주기 정본(`task_state.json`)에서 읽은
  실제 `running`/`pending`만** "딥시크가 할 일"로 보여준다. 완료는 별도 "최근 완료"
  섹션이다. 두 목록은 겹치지 않는다(`read_work` / `read_completed`).
- 각 항목은 카드에서 뽑은 제목과 1\~2문장 요약(모델 요약이 아니라 첫 문단),
  상태, 상태 진입 시각(`state_since`)을 표시한다. 상태를 task 파일에서 다시
  추론하지 않으므로 이중 상태가 없다.

### 6.4 검증

```
tests/test_queue_monitor.py   30 passed   (오늘 토큰 rollover/reset/M, 누적 필드 부재,
                                           GPU 정상/부분 누락/도달 불가, work/completed 분리)
```

관련 ops 회귀: `test_ops_supervisor`·`test_ops_task_lifecycle`·`test_ops_task_queue`·
`test_ops_opencode_config`·`test_llm_env` → 25 passed. 상태 머신 전이는 그대로다.

### 6.5 남은 위험 (후속)

1. **자정 rollover의 실측 미확인**: 순수 함수로만 검증했고, 실제 자정을 넘겨
   모니터를 이어 돌려 관측하지는 않았다(`모름`). 부분 집계 표기는 그 불확실성을
   숨기지 않기 위한 것이다.
2. **SSH 경로의 가용성 의존**: `ssh nhn`(alias 설정·키·원격 `nvidia-smi`)이 살아야
   수치가 나온다. 터널/SSH가 끊기면 정확한 값이 아니라 도달 불가가 뜬다 — 0을
   만들지 않으므로 오해는 없지만, 수집 스레드가 없으면 화면은 계속 도달 불가다.
3. **exporter HTTP 모드 미구현**: 위 6.2에 적은 설정을 요구하며, 구현 전까지는
   SSH 불가 시 자동 대체 경로가 없다.
4. **모니터 실브라우저 검증 미실행**: HTTP 렌더를 띄워 눈으로 확인하지 않았다.
   순수 함수·문자열 수준(PAGE에 누적 카드 문자열 부재)까지만 시험했다.

## 7. 변경 파일 (후속)

추가: `scripts/ops/gpu_telemetry.py`, `talks/ops/gpu.json`, 이 절.

수정: `scripts/ops/queue_monitor.py`(오늘 토큰·GPU 캐시 표시·work/completed 분리·
누적 카드 제거·수집 스레드), `tests/test_queue_monitor.py`(+19 tests),
`.gitignore`(`gpu.local.json`, `gpu_telemetry_cache.json`, `token_baseline.json`),
`docs/current_status.md`, `docs/agent_handoff.md`(테스트 수 `263 tests`).
