# 세션 인계 — 2026-09-19 06:40 KST · 실험을 Slurm으로 옮긴다

정본은 [`current_status.md`](current_status.md)다. 이 파일은 **환경이 바뀌는 이번 전환에
필요한 것만** 담는다.

## 바뀌는 것

| | 지금까지 | 앞으로 |
|:---|:---|:---|
| 실험 | 이 기계 GPU 4 직접 실행 | **NEXGEM 로그인 노드에서 Slurm 제출** |
| LLM 서버 | 같은 기계 `127.0.0.1:8000` | **그대로 둔다.** 네트워크로 접속 |

**deepseek 서버는 건드리지 않는다.** hostname `NEXGEM`, GPU 4\~7 텐서 병렬,
`0.0.0.0:8000`에 바인딩돼 있어 이미 외부에서 접속된다. 확인함:

```
curl -s http://10.34.5.16:8000/v1/models          # 200
"17 곱하기 23" -> "391"                            # 계산 정상
```

주소는 `talks/ops/llm.json`에 있고 git에 들어 있다. **코드를 고치지 않고 체크아웃만 하면
맞는 주소를 얻는다.** 대체 주소 `100.134.17.129`도 같은 호스트다.

`127.0.0.1`을 쓰면 안 된다 — Slurm 계산 노드에서 그 주소는 자기 자신이고 거기엔 아무것도
없다.

## 먼저 확인할 것 (이 기계에서는 확인할 수 없었다)

- **`slurm_rules.md`가 이 기계에 없다.** `CLAUDE.md`가 Slurm 규칙의 정본이라고 지목하는
  문서인데 접근 가능한 경로에 없었다. **로그인 노드에서 먼저 읽어라.**
- 이 기계에는 `sbatch`·`squeue`·`sinfo`가 **설치돼 있지 않다.** 파티션 이름, 계정, 큐 정책,
  GPU 요청 방식은 **모른다.** 지어내지 말고 로그인 노드에서 확인하라.
- 데이터 경로가 로그인 노드에서도 같은지 확인하라. 이 기계 기준:
  - `OFFICIAL=/NHNHOME/BASE/kimds/Data/PathoBench/official`
  - `FEATURES=/NHNHOME/BASE/kimds/Data/PathoBench/features`
  - `scripts/node_env.sh`가 경로를 탐색하므로 그 스크립트를 먼저 source 하라.

## 즉시 쓸 수 있는 자산

### 저장된 branch 마진 — 집계 후보는 GPU 0회

`predictions/margins_7b/` (1.3 MB, 7과제 × 50 fold, 공식 7-branch).
**git에 들어가지 않는다**(`predictions/`가 ignore 대상) — 필요하면 이 기계에서 복사하거나
`scripts/analysis/dump_branch_margins.py`로 다시 만든다(`1.94` GPU-h).

이 마진으로 `scripts/analysis/eval_aggregations.py`가 **GPU 없이** 집계 후보를 채점한다.
검증됨: 재집계한 macro `0.6226`이 공식 `0.6227`과 일치한다.

### 수치 지문 — 환경이 바뀌면 반드시 돌릴 것

```bash
.venv/bin/python scripts/analysis/numeric_fingerprint.py --check talks/reports/numeric_baseline.json
```

**기계가 바뀌면 값이 달라질 수 있다.** CPU 해시는
`5branch 799473738cbfece6` · `7branch 905bc41e112bae02` · `ct_on 37a52f0409367cc3`,
CUDA 해시는 `872019b9f37a1075` · `ecbb81f53d40a150` · `a8e2e101f03d237f`다.
**달라지면 그 기계의 산출물을 이 기계의 산출물과 비트 비교할 수 없다** — 무효가 아니라
비교 불가이며, 그 사실을 기록한 뒤 진행하라.

## 미해결 — 성능 문제 하나가 크다

**fold 적합 `20.0`초 중 GPU가 일하는 시간은 절반 미만이다.** 실측: 프로세스 CPU `590~761%`,
같은 시각 GPU 가동률 `42 44 50 67 0 0 0 0 0 0`(0.5초 간격).

원인은 fold마다 context 슬라이드를 호스트에서 디바이스로 다시 보내는 것으로 보이고,
로컬 모델이 **디바이스 상주로 `20.5`초 → `0.59`초(35배)** 를 실측했다. 적용하지 못한 이유는
둘이다.

1. `bm` 브랜치가 `2.4e-7 ~ 4.8e-7` 달라진다(감산 순서). AUROC에 영향이 있는지는 **미측정**.
2. 이 기계 GPU 4에 여유가 1 GB뿐이다 — deepseek가 168 GB를 쓴다.

**Slurm 노드에서는 2번 제약이 사라진다.** 전용 GPU를 받으므로 디바이스 상주가 가능하다.
가장 먼저 할 값어치가 있는 일이다: 35배면 후보 하나가 `1.94` GPU-h에서 **3분대**가 된다.
적용 전에 한 과제로 fold AUROC가 바뀌는지 먼저 재라.

작업 지시서는 [`talks/ops/tasks/perf_transfer.md`](../talks/ops/tasks/perf_transfer.md)에 있다.

## 어제 끝난 것

- **집계 축 종료.** 후보 넷 전부 승격 임계 미달
  ([결과](../talks/reports/2026-09-19_aggregation_result.md), 사전 등록 `19cf06b`).
- **`D-053`**: 여섯 실행 러너가 전부 같은 7-branch 설정으로 수렴해 있었다. 과거 arm 간 차이
  보고는 적합 변동이다.
- **`D-054`**: 반복 없음(`R=1`). 결정론적 후보는 정확, 확률적 후보는 seed 사전 고정.
- 유효 랭크 `3.52/7` 실측 재확인, 중복은 `BM`·`QA`·`DS`에 집중(`1.29/3`).

## 사용자 결정 대기

1. **예산 상한** — `4` GPU-h로는 2 arm paired가 97%다. Slurm으로 옮기면 이 제약이 달라질 수 있다.
2. **`closed_axes` G(AKS 등방성)·H(LID 부분추출)** 판정.
3. 다중 후보 비교 보정 규칙.

## 걸려 넘어지기 쉬운 곳

- 회차는 `run_round.sh`로만 띄운다(블로킹이라 완료가 곧 알림이다). **실험도 같은 방식으로
  띄워라** — 어젯밤 덤프를 그 밖에서 띄워 `00:51`에 끝난 것을 `05:53`에 알았고 **5시간을
  날렸다.**
- 앞선 회차 보고서를 좌석 입력에 넣지 않는다(검증기가 거부한다).
- `pgrep -f`로 자기 프로세스를 찾지 않는다. 이 저장소에서 셸이 세 번 죽었다.
- `test_docs_consistency.py`의 `RU-91~96` 대장 공백 1건 실패는 **기존 결함**이다.

[작성자: Claude Code / 소집자 / claude-opus-5 · 2026-09-19 06:40 KST]
