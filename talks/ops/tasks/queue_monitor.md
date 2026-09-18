딥시크 서버의 **큐를 실시간으로 볼 수 있는 모니터 페이지**를 만들어라.
사용자 요청이다: "딥시크 서버가 처리해야 하고, 처리하고 있는 큐를 실시간으로 확인할 수 있는
모니터 페이지."

## 만들 것

`scripts/ops/queue_monitor.py` — 표준 라이브러리만 쓰는 단일 파일 HTTP 서버.
`--port`(기본 8899)와 `--host`(기본 `0.0.0.0`)를 받는다. 실행하면 브라우저로 열 수 있는
한 페이지를 내놓고, 그 페이지가 **2초마다 스스로 갱신**한다(`fetch`로 같은 서버의
`/api/state` JSON을 폴링. 페이지 전체 새로고침 말고 값만 갈아끼워라).

의존성을 새로 추가하지 마라. `http.server`, `json`, `urllib.request`, `pathlib`, `subprocess`면
충분하다. Flask·FastAPI·React 쓰지 마라.

## 보여줄 것 — 아래 출처만 쓴다. 지어내지 마라

### 1. 서버가 지금 처리 중인 것 (vLLM 실시간)

엔드포인트 주소는 **직접 적지 말고** `scripts/ops/llm_env.py`의 `llm_env.endpoints()`에서 읽어라.
거기서 나온 `.../v1/chat/completions`의 호스트·포트에 `/metrics`를 붙이면 Prometheus 텍스트가
나온다. 실측으로 확인된 이름들이다:

```
vllm:num_requests_running{engine="0",model_name="deepseek-v4.1-flash"} 0.0
vllm:num_requests_waiting{...} 0.0
vllm:num_requests_waiting_by_reason{...,reason="capacity"} 0.0
vllm:gpu_cache_usage_perc{...}
vllm:generation_tokens_total{...} 1.000648e+06
vllm:prompt_tokens_total{...}
```

표시: **지금 돌고 있는 요청 수**, **대기 중인 요청 수**, KV 캐시 사용률, 누적 토큰.
누적 토큰은 직전 폴링과의 차이로 **초당 생성 토큰**을 함께 보여라 — 서버가 노는지 아닌지는
그 값이 가장 잘 말한다.

서버에 닿지 않으면 숫자를 0으로 채우지 말고 **"서버 도달 불가"라고 크게 표시**하라.
0과 "모름"은 다르다.

### 2. 프로젝트가 딥시크에 시킬 것 (대기 큐)

- `talks/ops/recurring/*.json` — 주기 잡무. 각 파일은
  `{"kind","label","cmd","every_minutes","parallel"}` 형식이다.
  `talks/ops/ticks.jsonl`(있으면)에서 각 label의 마지막 실행 시각을 읽어
  **다음 실행까지 남은 분**을 계산해 보여라. 이미 지났으면 "지연 N분"으로 표시한다.
- `talks/ops/tasks/*.md` — 위임 작업 명세. 각각에 대해 같은 이름의 브랜치
  (`chore/<name>-*`)가 git에 이미 있는지 보고 **미착수 / 착수됨**을 구분하라.

### 3. 방금 끝난 것과 실패한 것

- `talks/ops/ticks.jsonl` 마지막 20줄 — 최근 실행 이력.
- `talks/ops/failures.jsonl`(있으면) 마지막 10줄 — **실패는 눈에 띄게(붉게) 표시**하라.
  이 저장소는 실패가 조용히 사라져서 여러 번 당했다.
- `talks/ops/heartbeat` 파일의 수정 시각 — 감독자가 살아 있는지. 5분 넘게 갱신이 없으면
  **"감독자 정지 의심"**으로 표시하라.

위 파일들은 **없을 수 있다**. 없으면 죽지 말고 그 구역에 "기록 없음"이라고 적어라.

## 하지 말 것

- `pgrep -f`로 자기 프로세스를 찾지 마라. 이 저장소에서 셸이 세 번 죽었다.
- 값을 추정해서 채우지 마라. 못 읽으면 "모름"이다.
- 기존 파일을 고치지 마라. 새 파일 `scripts/ops/queue_monitor.py` 하나와,
  그 동작을 확인하는 `tests/test_queue_monitor.py`만 추가한다.

## 검증

`tests/test_queue_monitor.py`를 함께 써라. unittest로 작성한다(이 저장소는 unittest로 돈다).
최소한 이 둘은 확인해야 한다.

1. vLLM Prometheus 텍스트 **샘플 문자열**을 넣으면 running/waiting/토큰 수를 올바로 뽑는다.
   (실제 서버에 붙지 마라 — 파서만 시험한다.)
2. `talks/ops/` 하위 파일이 **하나도 없는** 임시 디렉토리를 가리켜도 예외 없이
   "기록 없음" 상태를 만들어낸다.

다 만들었으면 `.venv/bin/python -m unittest tests.test_queue_monitor` 가 통과하는지 직접 돌려
결과를 보고하라.

## 마지막에 적을 것

`scripts/ops/queue_monitor.py` 파일 맨 위 docstring에 **어떻게 띄우는지 한 줄**과
**어떤 출처를 읽는지** 적어라.
