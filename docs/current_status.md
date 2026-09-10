# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다.
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-10 (KST) |
| **Status** | CLEAN — **`SH`·`SJ` 승격 완료, 공식 구성을 7-branch로 고정**(`D-042`) (진행 중 RU 없음) |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장(23 GB) · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음. GPU 1·5는 **타 프로세스 점유 중**(16.1 GB / 15.3 GB)이므로 실행 전 `scripts/lib/free_gpus.sh`의 `icf_free_gpus`로 유휴 장치를 확인한다 |
| **회귀 테스트** | 153 tests · `OK` (2026-09-10) |

---

## 2026-09-10 — `SH`·`SJ` 승격 및 최종 아키텍처 고정

**공식 비교 기준을 5-branch(`CV,BM,BD,QA,DS`)에서 7-branch(`CV,BM,BD,QA,DS,SH,SJ`)로 교체했다.**
집계는 Trimmed Mean 그대로다. 새 기준선 macro·유효 랭크·불확실성 수치는
[`PROJECT.md` §3.2](PROJECT.md)에만 선언하며 여기에 복사하지 않는다 (출처: 결정 `D-042`).

- **승격 판정 방법**: 각 과제의 고정 50 fold에서 후보−기준선 AUROC를 짝지어 평균하고 과제별 평균 Δ
  7개를 동일 비중 평균한 값이 확정 개선량 기준을 넘는지로 판정했다. 넘었고, 랭크 효율도 상승했다.
  수치는 Main Agent가 `ru90_shape_triple` 태그에서 직접 재현했다.
- **과거 재현성 보존**: 분석 코드의 `BRANCHES`는 7-branch가 되었고, 직전 5-branch 집합은
  `BRANCHES_V121_5` 상수로 분리했다. 과거 RU 재현 스크립트는 후자를 쓰므로 옛 수치가 그대로 나온다.
- **반드시 함께 읽을 것 — 확증이 아니다.** 과제 군집 95% 신뢰구간이 **0을 포함**하므로 개선량 기준은
  충족했으나 **개선의 부호를 확정하지 못했다.** 세 과제(`Histologic_Grade`·`progression_regression`·
  `PBRM1`)는 형상 계열 투입으로 **일관되게 악화**되며 **기전 미확인**이다. 모든 판정은
  **`hold-out 미검증`** 이다. 구간·SE·악화 폭은 [`PROJECT.md` §3.2](PROJECT.md)에 있다.
- **기준 변경의 투명성 — 결과를 보고 완화한 것이 아니다.** RU-89는 같은 브랜치 집합을 `판별 불가`로
  종료했다. 당시 적용한 조건은 신뢰구간 **하한**이 개선량 기준을 넘을 것이었는데, 이 조건은 2026-09-06
  기록 과정에서 사용자 의도와 다르게 들어간 오기였음이 감사로 확인되어 정정됐다. 확정된 기준은
  **평균 개선량의 점추정**이다. 본 승격은 정정된 기준을 **같은 원자료에 다시 적용한 재판정**이며,
  RU-89·RU-90의 당시 판정 기록과 원자료는 그대로 보존한다 (출처: `D-041`·`D-042`).
- **개정 범위**: `PROJECT.md` §3.2·§3.3·§3.4·§5 · `current_architecture.md` §2.8 ·
  `branch_diagnostics.py`(`BRANCHES` 교체 + `BRANCHES_V121_5` 신설) · `branch_screen.py`(`ADOPTED` 비움).
  오염 검사 앵커는 **5-branch 기저 앵커로 의도적으로 유지**했다 — 검사 목적이 "스크리닝이 기저
  앙상블을 건드리지 않았음"의 확인이라 값을 갱신하면 과거 실행과 대조할 수 없기 때문이다.

## 실행 환경 — 현재 접속 호스트

기존 환경 기록은 `nexgem-s1` 직접 실행 유형(`D-030`), 시스템 Python 3.8.10이다.
`nexgem` 제출 시 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)이 적용된다.

### Immediate Next Command

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/analysis/branch_screen.py --tag ru90_shape_triple --candidate m_bs
# 새 공식 기준집합(BRANCHES 7개, adopted 없음)에서 게이트 ①이 도는지 확인
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

_작성자: Orca / Main Agent / claude-opus-5 (effort: high) · 2026-09-10 KST (D-042 승격 및 아키텍처 고정)._
