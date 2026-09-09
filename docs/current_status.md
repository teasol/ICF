# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-10 02:05 (KST) |
| **Status** | CLEAN — D-040 형상 계열 배선 정정 완료, RU-90 종료 (진행 중 RU 없음) |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장(23 GB) · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음. GPU 1·5는 **타 프로세스 점유 중**(16.1 GB / 15.3 GB)이므로 실행 전 `scripts/lib/free_gpus.sh`의 `icf_free_gpus`로 유휴 장치를 확인한다 |
| **회귀 테스트** | 153 tests · `OK` (2026-09-10) |

---

## 2026-09-10 — 형상 계열 배선 정정과 RU-90 종료 (`D-040`)

- **사용자 보고가 확증됐다.** `scripts/test_pathobench.py`의 다섯 집계 분기가 BS·SH·SJ를 pool에
  넣지 않아, 세 브랜치는 **공식 평가 경로에서 최종 확률에 도달한 적이 없다.** 세 마진은
  `logits`에만 가산됐고 `logits`는 모든 브랜치 가중치가 0일 때만 도달하는 폴백에서만 읽힌다.
  `cv_weight` 기본 1.442(v121 arm 1.0)이라 그 폴백은 도달 불가다. 보고에 없던 결함 2건도
  확인했다 — `sj_weight`가 마진 계산 게이트에서 빠져 SJ 단독 활성이 불가능했고, 파이프라인
  경로(`voting.py`)에는 BS가 전무했다.
- **수정**: 참여 조건·순서를 양쪽 경로에서 한 곳으로 통일(`cv…sw, sj, sh, bs`),
  `ICF_SHAPE_SCREEN_ONLY`의 지배 대상을 `logits` 가산에서 **pool 참여**로 이전(기본 `"1"` 유지 →
  기존 스크리닝 실행 비트 불변), BS를 `src/models/branches/bs.py`로 승격. 계약의 정본은
  [`current_architecture.md` §3.1](current_architecture.md)이다.
- **`RU-90`이 수정을 실측 검증했다** (재현·계측, 3층 기준을 결과 전에 고정). **세 층 전부 통과**:
  정정 후 5-branch BASE가 공식 `v121_baseline`과 7과제 전부 `max|diff| = 0.000e+00`으로 비트
  동일하고([`PROJECT.md` §3.2](PROJECT.md)의 기준선 macro와 [§3.4](PROJECT.md)의 오염 검사 상수를
  소수 4자리로 재현), live 경로와 오프라인 재집계가
  `1.788e-07` 안에서 일치하며(per-fold AUROC는 `0.000e+00`), `+SH`/`+SJ`/`+SH+SJ`의 대응 Δ가
  RU-89와 `0.01`/`0.00`/`0.04 %p` 차로 재현된다. 예산 `1.385 GPU-h`(상한 2.0).
- **RU-89의 판정은 바뀌지 않는다** — `판별 불가`·`보류` 유지. 과거 형상 계열 수치는 전부 오프라인
  재집계 산출이어서 유효했고, 문제는 live 재현 불가였다. **수치 정정 없음.**
- **BS는 게이트 ② 기각 상태를 유지한다**(31/50, `p = 0.059`). 사용자 결정(2026-09-10)에 따라
  RU-90의 arm에서 제외했고 마진만 저장했다(`m_bs` 50/50 전 과제). 어떤 구성에서도 승격 심사에
  상정하지 않는다.

## 진행 중 RU

없음. 다음 RU의 질문·대조·예산은 사용자와 확정한다.

## 실행 환경 — 현재 접속 호스트

현재는 `nexgem-s1`이다. `.venv` 버전은 헤더와 일치하며 시스템 `python3`는 3.8.10이다.
`source scripts/node_env.sh`는 `.venv`와 GPU 8장을 탐지하고, GPU 0~7 CUDA matmul 정상 동작을
2026-09-07 확인했다. `sinfo`·`squeue`가 없어 **이 호스트는 직접 실행 유형**이다(`D-030`).
`nexgem`을 통해 제출할 때만 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)이 적용된다.

### Immediate Next Command

```bash
sed -n '1,60p' docs/research_directions.md   # 큐 P1-B(CA-R1 5건)·B5 착수 순서 확정
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
- **v115~v120 6브랜치 조합이 미검증이다.** 과거 기록은 그대로 보존하고, 현행 승격 규칙
  (대응 per-fold Δ의 과제 군집 95% 구간 하한 > `δ_min`)으로 현행 조합을 검증하는 것은 남은 과제다. **절대 macro는 과제 모집단 성능으로 주장 불가**
  (군집 SE `4.34%p`, [`PROJECT.md` §4.1](PROJECT.md)) — 대응 비교만 정밀하다.

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

_by Orca / Main Agent / claude-opus-5 (effort: high) on nexgem-s1 at 2026-09-10 02:05 KST_
