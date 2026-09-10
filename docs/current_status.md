# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다.
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-10 11:15 (KST) |
| **Status** | CLEAN — 사용자 승격 기준 명확화(`D-041`) 전 문서 정합화 완료 (진행 중 RU 없음) |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장(23 GB) · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음. GPU 1·5는 **타 프로세스 점유 중**(16.1 GB / 15.3 GB)이므로 실행 전 `scripts/lib/free_gpus.sh`의 `icf_free_gpus`로 유휴 장치를 확인한다 |
| **회귀 테스트** | 153 tests · `OK` (2026-09-10) |

---

## 2026-09-10 — 사용자 승격 기준 명확화(`D-041`) 전 문서 정합화 완료

- **사용자 확정 의도**: 승격의 개선량 기준은 **Primary 7 평균 AUROC의 절대 증가량 +0.003 이상(= +0.3%p)**이다.
  **95% 신뢰구간 하한에 +0.3%p를 요구하는 기준이 아니다.** 상대 증가율 0.3%도 아니다.
- **계산**: 각 과제의 고정 50 fold에서 후보−기준선 AUROC를 짝지어 평균하고, 과제별 평균 Δ 7개를 동일 비중으로 평균한다.
  `Δ_macro = (1/7) Σ_task [(1/50) Σ_fold (AUC_candidate − AUC_baseline)] ≥ 0.003`.
  모든 과제의 개별 개선이나 `sign agreement ≥ 5/7`을 추가 필수 조건으로 요구하지 않는다.
- **직관 확인 예시**: [공식 비교 기준선](PROJECT.md) 대비 후보 +0.3%p 이상이면 개선량 기준 충족.
  RU-90은 약 0.6227, 대응 Δ `+0.562964%p`로 **이 개선량 기준을 충족**한다.
  (독립적 성능 확증·hold-out 검증·공식 구성 교체 완료는 별개 절차다).
- **불확실성의 역할**: CI·SE와 과제별 악화는 투명하게 별도로 보고한다. CI 하한 `> +0.3%p` 또는 `> 0`을
  사용자 승인 없이 승격 필수 조건으로 덧붙이지 않는다. hold-out·선택 이력·비교 조건에 관한 기존 규칙은 별도다.
- **전 문서 정합화 완료**: `PROJECT.md §4·§4.1`, `agent_handoff.md R5·R6`, `history/archive.md D-018·D-021·D-041`,
  `research_directions.md`, RU-89·90 기록 전반에 정합화를 완료했다.
  과거 원자료·당시 판정은 보존하고, 의도 정정 이력(`D-041`)을 남겼다.

## 이전 실행 상태 — D-040 배선 정정 및 RU-90 종료

- BS·SH·SJ의 live 집계 누락을 수정했고 BASE 비트 불변·live/오프라인 등가성·세대 간 Δ 재현을 확인했다.
- RU-89의 당시 `판별 불가`·`보류` 기록은 남아 있으며, RU-90에서 live 재현성과 개선량 충족을 확인했다.
- BS는 게이트 ② 기각 상태이며 RU-90의 arm에서 제외됐다. 상세 증거는 [RU-90](history/research_units_all.md#ru-90-live-path-equivalence-of-the-corrected-bsshsj-wiring-and-single-generation-re-measurement-of-the-shsj-arms)과 [D-040](history/archive.md)에 있다.

## 실행 환경 — 현재 접속 호스트

기존 환경 기록은 `nexgem-s1` 직접 실행 유형(`D-030`), 시스템 Python 3.8.10이다.
`nexgem` 제출 시 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)이 적용된다.

### Immediate Next Command

```bash
bash scripts/run_tests.sh   # 전 문서 정합화 회귀 테스트 검증
```

---

## 미해결 (Open Issues)

**사용자 판단 대기**
- **다음 연구 방향** — §226 Tier 1 3건이 전부 종료됐다. 큐의 `P1-B`(CA-R1 계열 5건)가 다음
  후보군이며 착수 순서·예산이 미결이다([`research_directions.md` §0](research_directions.md)).
- **경계 판정 2건** — `closed_axes.md` §3-I(컨텍스트 라벨 통계 대 `CA-06`)·§3-J(다중 강도 브랜치
  대 `P2-SELECTOR-CEILING`·`CA-09`). 판정 전에는 큐 `B2`·`B3`을 착수하지 않는다.
- **게이트 ② 통계량** — 게이트 ②는 단측 `AUROC > 0.5`라 '정보량'이 아니라 '**양의 방향** 정보량'을
  잰다. 부호 반전 후보는 정보를 담고도 구조적으로 탈락한다(RU-86 관측). 양측 `|AUROC−0.5|`로
  바꾸면 사후 부호 선택의 문이 열리므로 **사용자 판정이 필요하다**(큐 `B5`).
- **재개 축 활용 여부** — `CA-04`(해소 항목 ②·③ 남음)·`CA-02`(재현 선행). 착수 시점 미결이다.

**연구상의 교착**
- **과제 특화 이득을 활용할 선택 신호가 없다.** 이득은 실재하나 라벨 없이 과제를 판별할 수단이
  없다 — §221(SJ/구 SHJ)·§225(subsampling)가 같은 벽이다 ([`closed_axes.md` `CA-R1`](closed_axes.md)).
- **모든 판정이 `hold-out 미검증`** 이다 ([`PROJECT.md` §3.1](PROJECT.md)).
- **v115~v120 6브랜치 조합 검증은 남은 과제다.** 과거 기록을 보존하고 위 사용자 개선량 기준과 불확실성 보고를 구분해 문서를 정합화했다(`D-041`).

**기술 부채**
- §226 Tier 1 3건(`AKS`·`MDX`·`LID`)이 **동일 에이전트 한 배치 산출**이며 독립 비교군이 없다.
- §226 적대적 검증 2건 미실행 (`research_directions.md`에 `미검증` 명시).
- **bf16 정밀도 부채는 RU-87에서 종료됐다.** 정밀도 차는 두 arm의 공통 성분이라 **짝지은 Δ에서
  상쇄된다.** 다만 상쇄는 fold-mean에서만 일어나므로 **단일 fold 수치를 근거로 쓰는 서술은 약
  `0.6%p` 잡음을 안는다.** 수치와 경위는 결정 이력 `D-037`.
- **집계 로직이 두 곳에 따로 구현돼 있다** — `voting.py`(파이프라인, 로짓 반환)와
  `test_pathobench.py` 인라인 분기(평가, 확률 반환). `D-040` 배선 누락의 근본 원인이며 순서·조건
  통일과 일치 회귀 테스트로 **드리프트는 차단했으나 중복은 남아 있다**
  ([`current_architecture.md` §4.1](current_architecture.md)).
- **`ICF_SHAPE_SCREEN_ONLY=0`의 실측 검증은 `trimmed_mean` 1종에만 있다.** 나머지 4개 집계
  (`soft_voting`·`hard_gated`·`adaptive_trimmed`·`context_loo`)에서 형상 계열이 pool에 들어가는
  것은 회귀 테스트로만 고정됐고 평가 실행으로는 확인되지 않았다 (RU-90 한계 ①).
- **AUROC 추정기가 두 개다.** `test_pathobench` 내부 계산과 `branch_diagnostics.auroc`가 동일 확률
  벡터에서 `+8.55e-05`의 고정 차를 낸다. 짝지은 Δ에서 상쇄되지만 **절대 AUROC를 두 출처에서 섞어
  인용하면 소수 4자리에서 어긋난다.** 어느 쪽이 옳은지는 판정하지 않았다 (RU-90 한계 ③).
- `history/archive.md`에 **§199·§200 절 번호가 각각 중복** ([`closed_axes.md` §4](closed_axes.md)).

_기존 실행·환경 기록 작성자: Orca / Main Agent / claude-opus-5 (effort: high) · 2026-09-10 02:05 KST; D-041 전 문서 정합화: 2026-09-10 11:15 KST._
