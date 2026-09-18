# Current Status

> **"지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.** 수치는 [`PROJECT.md`](PROJECT.md),
> 닫힌 축은 [`closed_axes.md`](closed_axes.md), 결정 이력은 [archive](history/archive.md),
> 협의체 규범은 [`AGENTS.md`](../AGENTS.md)가 정본이다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-18 14:20 KST |
| **Status** | **WIP** · 계측 오진 정정 완료(`D-053`). 협의체 Phase B·Q 신설, A/B 대조 진행 중 |
| **Host / Node** | `NEXGEM` · NVIDIA B200 8장(각 183,359 MiB) · `sbatch`·`squeue` 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.3 · PyTorch 2.14.0+cu130 · `pytest` 9.1.1 |
| **GPU 배정** | **GPU 4 = 실험 전용.** GPU 5·6·7 = 로컬 Qwen 서버(포트 8003/8001/8000). GPU 0-3은 타 사용자 |
| **Active Job** | **없음.** GPU 5·6·7의 Qwen 서버 3대가 모두 죽어 있다(아래) |
| **회귀 테스트** | `bash scripts/run_tests.sh` 200 tests. `test_docs_consistency` RU 번호 1건 실패는 기존 결함(아래) |

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
군집(`df=6`) 95% CI가 **`[-0.0114, +0.0228]`로 0을 포함**하고 **7개 중 3개 과제가 악화**된다.
과제 간 sd `0.0185`가 평균 효과의 3배다.
`evaluate_pure.py`는 결정론적이라 적합 분산은 없다.
상세: [`talks/reports/2026-09-18_arm_parity.md`](../talks/reports/2026-09-18_arm_parity.md).

### 5-branch 앵커는 재현되지 않고, 검사가 요구 정밀도를 못 낸다

SMAD4·PBRM1 앵커가 **부호가 반대인 `5e-4`** 만큼 어긋난다. 실행은 결정론적이므로 기록 이후
실제로 무언가 바뀌었다(원인 미확인). `PROJECT.md` §3.4가 요구하는 4자리를 **구성을 바꾸지
않아도 못 맞춘다.**

### fold 적합 1회 = `20.0`초 · 확증 프로브는 예산에 안 들어간다

한 arm Primary 7 `1.94` GPU-hour, **2 arm paired `3.89`(예산 4h의 97%)**, `R=4` 2×2는 `31.1`로
7.8배다. 앞선 회차들의 `R=8`·`R=16` 논의는 비용을 모르는 상태의 논의였다.
([비용 보고서](../talks/reports/2026-09-18_foldfit_cost.md))

### 교차 query 누수는 없다 · `tests/test_query_independence.py`

query를 50배로 교체·제거·순서변경해도 나머지 margin 변화 `1e-5` 미만.
## 협의체 — Phase B(자유 대화)와 Phase Q(문서 질의) 신설

규범은 [`AGENTS.md`](../AGENTS.md) §2. 요지: 자유 대화는 허용하되 **대화록은 넘어가지 않고
`가설:` 한 줄들만 넘어간다**(추출은 정규식, 모델 아님). 사실 질문은 동료가 아니라 **문서가
인용으로 답하고 코드가 그 인용을 대조**한다. 회차마다 `metrics.json`을 코드가 쓴다 —
**이견 턴이 계속 `0`이면 그 채널은 메아리방이므로 폐기한다.**

회차는 `run_round.sh`로만 띄우고(블록됨), 턴 종료 전 `turn_end.sh`를 반드시 실행한다.

### A/B 대조 진행 상황

17(Phase B) 대 18(없음)은 **무효**다 — 브레인스토밍 좌석이 제안 원형 임무를 받아 제안서를
썼다. 덕분에 "Phase 1을 두 번 돌리면?"이 측정됐다: 중복도 `0.197` 대 `0.1878`, 지지 `8` 대
`8`, 호출 `19` 대 `13` — **아무것도 사지 않는다.** 19(수정판)는 제안서화 `0`건, 선언된 이견
`6`턴, 질문 `11`건으로 수정이 작동했다. **20(대조군)은 Qwen 서버가 죽어 막혔다.**

### Immediate Next Command

```bash
# 1) Qwen 서버 3대 재기동 — 이것 없이는 회차를 소집할 수 없다
cd /NHNHOME/WORKSPACE/kimds/LLM
GPU=7 PORT=8000 ./start_server_vllm.sh &   # 포트 8000
GPU=6 PORT=8001 ./start_server_vllm.sh &   # 포트 8001
GPU=5 PORT=8003 ./start_server_vllm.sh &   # 포트 8003
# 2) 회차 20(A/B 대조군) — 19와 짝이며 아직 못 돌렸다
cd /NHNHOME/WORKSPACE/kimds/ICF && bash scripts/ops/turn_end.sh
bash scripts/ops/run_round.sh talks/council/C-20260918-20_card.json
# 이후 두 metrics.json 비교: talks/council/C-20260918-{19,20}/metrics.json
```

---

## 미해결 (Open Issues)

- **Qwen 서버 3대가 죽어 있다.** 로그가 정상 동작 중 끊겼고 오류 흔적이 없다 — 외부에서
  종료된 것으로 보인다. 원인 미확인. 재기동 명령은 위에 있다.
- **fold 수준 짝지은 Δ 미산출.** `predictions/parity_*.pt`로 가능하나, 과제 군집 CI가 이미
  0을 포함하므로 우선순위는 낮다.
- **4 GPU-hour로는 확증이 불가능하다.** 예산 증액이냐 설계 축소냐는 사용자 판단 사안이다.
- **적합 분산 포함 승격 절차가 미승인**이고 `R`도 미정이다. `R=4`조차 예산의 7.8배다.
- **앵커 재현 실패 원인 미확인**(코드 변경인지 데이터 변경인지).
- **PCA·subsample 사양이 문서에 없다.** 회차 16이 v1 안을 냈으나 기록되지 않았다.
- **RU 번호 공백**: `RU-90` → `RU-97`. `RU-91`\~`96`이 대장에도 열린 카드에도 없어
  `test_docs_consistency.py` 1건이 실패한다. 없는 기록을 지어낼 수 없어 미수정으로 둔다.
- 모든 후보 판정은 `SEAL 10` hold-out 미검증 상태를 유지한다.

[작성자: Claude Code / 소집자 / claude-opus-5 · 2026-09-18 16:45 KST]
