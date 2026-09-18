# Current Status

> **"지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.** 수치는 [`PROJECT.md`](PROJECT.md),
> 닫힌 축은 [`closed_axes.md`](closed_axes.md), 결정 이력은 [archive](history/archive.md),
> 협의체 규범은 [`AGENTS.md`](../AGENTS.md)가 정본이다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-19 06:40 KST |
| **Status** | **WIP** · 집계 축 종료(후보 4건 전부 미달). **실험을 NEXGEM 로그인 노드 Slurm으로 이관 중** |
| **Host / Node** | `NEXGEM` · NVIDIA B200 8장(각 183,359 MiB) · `sbatch`·`squeue` 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.3 · PyTorch 2.14.0+cu130 · `pytest` 9.1.1 |
| **GPU 배정** | **GPU 4\~7 = deepseek 서버**(텐서 병렬, 168 GB). **실험은 Slurm으로 제출한다.** GPU 0-3은 타 사용자 |
| **Active Job** | 없음. deepseek 서버가 GPU 4\~7에서 `0.0.0.0:8000` 대기 중(네트워크 접속 가능) |
| **회귀 테스트** | `bash scripts/run_tests.sh` 213 tests. `test_docs_consistency` RU 번호 1건 실패는 기존 결함(아래) |

---

## 오늘 확정된 것

### 여섯 러너가 전부 같은 설정으로 수렴해 있었다 (`D-053`)

`eval_v121.sh`가 **학습 설정**을 넘기고 `eval_seal_tasks.sh`가 `model:` 블록을 보고 7-branch로
**조용히 교체**했다. 나머지 다섯도 같다. **arm 이름이 다른 실행들이 서로 다른 구성을 돌린 적이
없다.** `ICF_*`도 `src/`에서 아무도 읽지 않아 두 경로가 동시에 끊겨 있었다 — 과거 arm 간 차이
보고는 전부 적합 변동이다. 폴백은 이제 **큰 소리로 실패**한다.
상세: [`RU-90 보고서`](../talks/reports/2026-09-18_ru90_baseline_regen.md) §9\~§10.

### 5-branch 대 7-branch — 판별 불가 (처음으로 실제로 나눠 돌렸다)

Primary 7 macro 차이는 **`+0.0057`**이고, [`PROJECT.md`](PROJECT.md)의 공식 기준선·계보값·
승격 근거가 **전부 재현된다.** 값이 틀린 적은 없었고 문제는 귀속 불가였다. 그러나 과제
군집(`df=6`) 95% CI가 **`[-0.0115, +0.0228]`로 0을 포함**하고 **7개 중 3개 과제가 악화**된다.
과제 간 sd `0.0185`가 평균 효과의 3배다.
`evaluate_pure.py`는 결정론적이라 적합 분산은 없다.
상세: [`talks/reports/2026-09-18_arm_parity.md`](../talks/reports/2026-09-18_arm_parity.md).

### 5-branch 앵커는 재현되지 않고, 검사가 요구 정밀도를 못 낸다

SMAD4·PBRM1 앵커가 **부호가 반대인 `5e-4`** 만큼 어긋난다. 결정론적 실행이므로 기록 이후
실제로 무언가 바뀌었고(원인 미확인), `PROJECT.md` §3.4의 4자리 요구는 **구성을 안 바꿔도
못 맞춘다.**

### fold 적합 1회 = `20.0`초 — 그중 GPU가 일하는 시간은 절반 미만이다

한 arm Primary 7 `1.94` GPU-hour, 2 arm paired `3.89`(예산 4h의 97%).
**다만 이 값의 절반 이상이 호스트↔디바이스 전송이다** — 디바이스 상주로 `0.59`초/fold(35배)가
실측됐으나 GPU 메모리 부족으로 이 기계에서는 적용 못 했다. Slurm 노드에서는 가능하다.
([비용](../talks/reports/2026-09-18_foldfit_cost.md) · [인계](SESSION_HANDOFF.md))

### 교차 query 누수는 없다

`tests/test_query_independence.py` — query를 50배로 교체·제거·순서변경해도 나머지 margin
변화 `1e-5` 미만.

## 협의체 — Phase B(자유 대화)와 Phase Q(문서 질의) 신설

규범은 [`AGENTS.md`](../AGENTS.md) §2. 요지: 자유 대화는 허용하되 **대화록은 넘어가지 않고
`가설:` 한 줄들만 넘어간다**(추출은 정규식, 모델 아님). 사실 질문은 동료가 아니라 **문서가
인용으로 답하고 코드가 그 인용을 대조**한다. 회차마다 `metrics.json`을 코드가 쓴다 —
**이견 턴이 계속 `0`이면 그 채널은 메아리방이므로 폐기한다.**

회차는 `run_round.sh`로만 띄우고(블록됨), 턴 종료 전 `turn_end.sh`를 반드시 실행한다.
**Phase B는 A/B에서 이득을 보이지 못해 기본 꺼짐이다**
([결과](../talks/reports/2026-09-18_phase_b_ab.md)).

### 실험 실행 위치가 바뀐다 (2026-09-19)

실험은 이 기계가 아니라 **NEXGEM 로그인 노드에서 Slurm으로 제출**한다. LLM 서버는 이 기계에
그대로 두고 네트워크로 쓴다 — 주소는 `talks/ops/llm.json`(`10.34.5.16:8000`)에 있고 git에
들어 있다. `127.0.0.1`을 쓰면 계산 노드 자신을 가리켜 아무것도 없다.
전환 절차와 확인 못 한 것들은 [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md)에 있다.

### Immediate Next Command

```bash
# 로그인 노드에서. 먼저 slurm_rules.md 를 읽는다 (이 저장소에는 없다)
cd <repo>/ICF && . scripts/node_env.sh
.venv/bin/python scripts/analysis/numeric_fingerprint.py --check talks/reports/numeric_baseline.json
# 기계가 바뀌면 해시가 달라질 수 있다. 달라지면 비트 비교 불가로 기록하고 진행한다.

# 가장 값어치 있는 첫 작업: 디바이스 상주 35배를 Slurm 전용 GPU에서 검증
#   talks/ops/tasks/perf_transfer.md · 적용 전 한 과제로 fold AUROC 불변 확인
```

---

## 미해결 (Open Issues)

- **fold 적합의 절반 이상이 GPU 밖이다.** 디바이스 상주 35배가 실측됐으나 이 기계에서는
  GPU 메모리가 없어 적용 못 했다. Slurm 노드의 첫 과제다([인계](SESSION_HANDOFF.md)).
- **적합 분산 포함 승격 절차가 미승인**이고 `R`도 미정이다(`D-054`는 `R=1`만 정했다).
- **4 GPU-hour로는 확증이 불가능하다.** 예산 증액이냐 설계 축소냐는 사용자 판단 사안이다.
- **앵커 재현 실패 원인 미확인**(코드 변경인지 데이터 변경인지).
- **PCA·subsample 사양이 문서에 없다.** 회차 16이 v1 안을 냈으나 기록되지 않았다.
- **RU 번호 공백**: `RU-90` → `RU-97`. `RU-91`\~`96`이 대장에도 열린 카드에도 없어
  `test_docs_consistency.py` 1건이 실패한다. 없는 기록을 지어낼 수 없어 미수정으로 둔다.
- 모든 후보 판정은 `SEAL 10` hold-out 미검증 상태를 유지한다.

[작성자: Claude Code / 소집자 / claude-opus-5 · 2026-09-19 06:40 KST]
