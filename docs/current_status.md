# Current Status

> **"지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.** 수치는 [`PROJECT.md`](PROJECT.md),
> 닫힌 축은 [`closed_axes.md`](closed_axes.md), 결정 이력은 [archive](history/archive.md),
> 협의체 규범은 [`AGENTS.md`](../AGENTS.md)가 정본이다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-19 15:03 KST |
| **Status** | **D-042 verified** · RU-100 재확증 · RU-101/102/103 형상 브랜치 진단(TGW context 신호 퇴화 → 판별 불가) · 오염 검사 L1 · 자동 잡무 중단(`D-056`) · `SEAL 10` 보류(`D-057`) |
| **Host / Node** | **`nexgem`(소문자) = Slurm 로그인 노드.** GPU 없음, 20 CPU · 93 GB. 계산은 전부 `sbatch`/`srun`. 대문자 `NEXGEM`(NHN, B200)은 **다른 기계**다 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · `pytest` 9.1.1. `scripts/node_env.sh`가 `ICF_DATA_ROOT=data/repro_labels_folds`로 해소 |
| **GPU 배정** | Slurm `batch` 파티션 `gnode1\~6`. `gnode5`(A6000, 드라이버 595.91.07) 실측 동작. 딥시크는 NHN `NEXGEM` GPU 4\~7에 그대로 |
| **LLM 접속** | NEXGEM이 tailscale로 **직결** — `llm.local.json`의 `100.97.255.47:8000`. 옛 `ssh nhn` 터널은 불필요. ssh도 `100.97.255.47:22`(`NEXGEM_key`) |
| **Active Job** | 없음. 자동 잡무 중단·감독자 정지(`D-056`). `queue_monitor` → `http://100.65.212.1:8899`(표시 정지) |
| **회귀 테스트** | 전체 회귀 **295 tests 통과**(16 skipped). `RU-91`\~`96` 총람 공백을 실측 보고서·`D-043`·`D-044`에서 복원(사전 등록 필드는 `미보존`으로 명시)해 `test_docs_consistency.py` 포함 전부 초록 |

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
macro 차이 `+0.0057`이고 과제군집 95% CI `[-0.0115, +0.0228]`이 0을 포함, 7개 중 3개가 악화된다.
5-branch 앵커는 `5e-4` 어긋난다(RU-98이 **장비는 원인이 아님**을 보였다). fold 상주화는 RU-99
채택(`12.8→7.1`초/fold). 협의체 C-20260919-1의 규정대로 **RU-100**이 두 arm을 같은 잡·세션에서
재실행해 `+0.0057`(§4 충족)·CI·악화 `3/7`을 재현하고 provenance와 함께 **D-042를 verified**로 올렸다.

### RU-98 교차 기계 재현 종료 — 지지

Primary 7 전 과제가 Slurm A5000/A6000에서 완료, B200 참조와 소수 4자리 모두 동일. 사전 기준
(1) 충족으로 **4자리 결론이 장비를 건너 유지**되며 **앵커 `5e-4` 어긋남은 장비 탓이 아님**으로
판정·종료(대장 append). 실행 보고 [`cross_machine_repro`](../talks/reports/2026-09-19_cross_machine_repro.md).
한계: 하드웨어 쌍 하나·5-branch만, 기계 A 원시 예측 부재로 정확 |Δ|는 `<0.0001` 상계뿐.

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
# 0. NEXGEM vLLM 도달 확인 (tailscale 직결)
curl -s -m 5 http://100.97.255.47:8000/v1/models >/dev/null && echo llm-ok || echo llm-DOWN
# 1. (Slurm 풀리면) provenance-fixed 전후 검정 — SMAD4 fold 10, 상주화 ON/OFF
# 2. 운영: sbatch 로 실험, 회차는 `scripts/council.py run` 직접 (감독자 정지, D-056)
```

**오케스트레이션은 이 세션이 직접 한다(`D-056`).** 주기 잡무 6건은
`talks/ops/recurring/disabled/`로 옮겼고 감독자는 정지 상태다. `queue_monitor`는 떠 있으나
`state.md`가 갱신되지 않아 표시가 멈춘다.

---

## 미해결 (Open Issues)

- **판단 대기 항목 정리됨(`D-057`).** CI 하한 게이트 미도입(점추정 유지, `§4`) · 지문 기준선은
  호스트·디바이스별 분리 · 앵커 `5e-4` 원인 규명 보류 · 확률적 `R>1`·fold 재분할은 트리거 대기.
- **`SEAL 10` 개봉 보류(사용자, `D-057`).** 성능이 충분히 오른 후보가 없어 열지 않는다. 후보가
  서면 그때 다시 정하며, 개봉은 독립 최종 검증 가치를 소멸시킨다(`D-047`).
- **오염 검사는 L1 귀속 검사로 재설계(C-20260919-2).** `check_artifacts.py`가 provenance를 선언 config와 대조한다. 시점 간 드리프트 검출은 상실됐고 공백으로 명시한다(L2는 현행 closed-form 전용).
- **PCA·subsample 사양이 문서에 없다.** 회차 16이 v1 안을 냈으나 기록되지 않았다.
- **재구축 운영 구조의 장기 실측이 남았다.** 자정 토큰 rollover·재시작 복구는 단위 테스트만 통과했다.
- **7-branch `3/7` 악화 기전 부분 확인(RU-101).** SJ 단독 AUROC이 악화 3과제에서 ≈`0.49`(무정보)·나머지 `0.56`. RU-102 oracle 상한 Δ `+0.0053`(임계 초과, SE 미만). RU-103: context-only 신호(context 자기 AUROC)가 in-sample이라 퇴화 → **판별 불가**. `TGW`(P4)는 유효 신호 재설계가 선행 조건.
- **앵커 `5e-4` 원인 미확인.** RU-98이 장비를 배제했고, 규명 RU 신설은 `D-057`로 보류했다.

[작성자: Claude Code / 소집자 / claude-opus-5 · 2026-09-19 06:40 KST]
[작성자: Claude Code / Platform Agent / claude-opus-5 (effort: 미확인) · 2026-09-19 09:40 KST]
[작성자: OpenAI Codex / Reasoning Agent / GPT-5 (effort: 미확인) · 2026-09-19 11:20 KST]
[작성자: opencode / Platform Agent / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 14:27 KST]
[작성자: opencode / 종결권자(사용자 위임) / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 15:03 KST]
[작성자: opencode / Platform Agent / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 16:40 KST]
[작성자: opencode / 소집자·오케스트레이터 / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 17:45 KST]
[작성자: opencode / Platform Agent / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 18:00 KST]
[작성자: opencode / 소집자·오케스트레이터 / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 18:10 KST]
[작성자: opencode / Platform Agent / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 19:16 KST]
