# Current Status

> **"지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.** 수치는 [`PROJECT.md`](PROJECT.md),
> 닫힌 축은 [`closed_axes.md`](closed_axes.md), 결정 이력은 [archive](history/archive.md),
> 협의체 규범은 [`AGENTS.md`](../AGENTS.md)가 정본이다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-19 11:20 KST |
| **Status** | **WIP** · RU-98 실행 7/7 완료, 전부 표시 정밀도 Δ=`0.0000`. Reasoning 판정·종료 대기 |
| **Host / Node** | **`nexgem`(소문자) = Slurm 로그인 노드.** GPU 없음, 20 CPU · 93 GB. 계산은 전부 `sbatch`/`srun`. 대문자 `NEXGEM`(NHN, B200)은 **다른 기계**다 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · `pytest` 9.1.1. `scripts/node_env.sh`가 `ICF_DATA_ROOT=data/repro_labels_folds`로 해소 |
| **GPU 배정** | Slurm `batch` 파티션 `gnode1\~6`. `gnode5`(A6000, 드라이버 595.91.07) 실측 동작. 딥시크는 NHN `NEXGEM` GPU 4\~7에 그대로 |
| **LLM 접속** | `talks/ops/llm.json`의 주소는 이 기계에서 안 닿는다. `ssh nhn` 터널 + git-ignore된 `llm.local.json` 필요 — [`SESSION_HANDOFF.md`](SESSION_HANDOFF.md) §1 |
| **Active Job** | RU-98 array `156357` 완료. tmux `queue_monitor` → `http://100.65.212.1:8899`; `supervisor` 복구됨 |
| **회귀 테스트** | ops 대상 회귀 57건 통과. 전체 회귀는 변경 전 `275 passed, 16 skipped, 1 failed`, 현재 294건 수집; 실패 1건은 기존 RU 번호 공백(아래) |

---

## 오늘 확정된 것

### 여섯 러너가 전부 같은 설정으로 수렴해 있었다 (`D-053`)

`eval_v121.sh`가 **학습 설정**을 넘기고 `eval_seal_tasks.sh`가 `model:` 블록을 보고 7-branch로
**조용히 교체**했다. 나머지 다섯도 같다. **arm 이름이 다른 실행들이 서로 다른 구성을 돌린 적이
없다.** `ICF_*`도 `src/`에서 아무도 읽지 않아 두 경로가 동시에 끊겨 있었다 — 과거 arm 간 차이
보고는 전부 적합 변동이다. 폴백은 이제 **큰 소리로 실패**한다.
상세: [`RU-90 보고서`](../talks/reports/2026-09-18_ru90_baseline_regen.md) §9\~§10.

### 5-branch 대 7-branch — 판별 불가 · 앵커 재현 실패 · fold 비용

상세는 [`arm_parity`](../talks/reports/2026-09-18_arm_parity.md)와
[`foldfit_cost`](../talks/reports/2026-09-18_foldfit_cost.md)가 정본이다. 요지:
macro 차이 `+0.0057`이나 과제군집 95% CI `[-0.0115, +0.0228]`이 0을 포함하고 7개 중
3개가 악화된다. 5-branch 앵커는 `5e-4` 어긋난다(RU-98이 **장비는 원인이 아님**을 보였다).
fold 적합의 절반 이상이 GPU 밖이며 디바이스 상주로 35배가 실측됐다 — 전용 GPU가 생긴
지금 적용 가능하나 **RU-98이 끝나기 전엔 안 된다**(수치를 바꾼다).

### RU-98 교차 기계 재현 실행 완료

Primary 7 전 과제가 Slurm A5000/A6000에서 완료됐고 B200 참조값과 소수 4자리까지 모두 같았다.
사전 기준에 대입하면 장비를 건너 4자리 결론이 유지되며, 기존 앵커 어긋남은 장비 탓이 아니다.
실행 보고는 [`cross_machine_repro`](../talks/reports/2026-09-19_cross_machine_repro.md), 공식 판정과
RU 종료는 Reasoning 대기다.

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
# 0. 터널이 살아 있는지부터 (죽어 있으면 딥시크·잡무가 전부 조용히 실패한다)
curl -s -m 5 http://192.168.100.100:8000/v1/models >/dev/null && echo tunnel-ok || \
  ssh -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -L 127.0.0.1:8000:127.0.0.1:8000 -L 192.168.100.100:8000:127.0.0.1:8000 nhn

# 1. RU-98 실행 보고를 Reasoning에 전달해 판정·종료
bash scripts/call_agent.sh orca "RU-98 판정·종료: docs/ru/RU-98.json과 talks/reports/2026-09-19_cross_machine_repro.md 검토"

# 2. 운영 상태 확인
tmux list-sessions; squeue -u kimds
```

---

## 미해결 (Open Issues)

- **fold 적합의 절반 이상이 GPU 밖이다.** 디바이스 상주 35배가 실측됐으나 이 기계에서는
  GPU 메모리가 없어 적용 못 했다. Slurm 노드의 첫 과제다([인계](SESSION_HANDOFF.md)).
- **적합 분산 포함 승격 절차가 미승인**이고 `R`도 미정이다(`D-054`는 `R=1`만 정했다).
- **지문 기준선을 기계별로 나눌지 미정.** 기계가 바뀌면 해시가 달라진다(집계 margin 절대
  `1.5e-06`) — 무효가 아니라 비트 비교 불가다. 별도로 **결론 재현 허용 오차**를 판정 규칙에
  넣을지도 사용자 판단 사안이다(RU-98 결과가 입력).
- **재구축 운영 구조의 장기 실측이 남았다.** 명시적 상태 머신·자원별 스케줄링·로컬 vLLM 강제는 구현됐으나, 재시작 복구와 자정 토큰 rollover는 단위 테스트만 통과했다.
- **4 GPU-hour로는 확증이 불가능하다.** 예산 증액이냐 설계 축소냐는 사용자 판단 사안이다.
- **앵커 재현 실패 원인 미확인.** RU-98이 **장비는 원인이 아님**을 보였다(`SMAD4` `0.4426`,
  `PBRM1` `0.5546` 4자리 재현). 코드인지 데이터인지는 여전히 미확인이다.
- **PCA·subsample 사양이 문서에 없다.** 회차 16이 v1 안을 냈으나 기록되지 않았다.
- **RU 번호 공백**: `RU-90` → `RU-97`. `RU-91`\~`96`이 대장에도 열린 카드에도 없어
  `test_docs_consistency.py` 1건이 실패한다. 없는 기록을 지어낼 수 없어 미수정으로 둔다.
- 모든 후보 판정은 `SEAL 10` hold-out 미검증 상태를 유지한다.

[작성자: Claude Code / 소집자 / claude-opus-5 · 2026-09-19 06:40 KST]
[작성자: Claude Code / Platform Agent / claude-opus-5 (effort: 미확인) · 2026-09-19 09:40 KST]
[작성자: OpenAI Codex / Reasoning Agent / GPT-5 (effort: 미확인) · 2026-09-19 11:20 KST]
