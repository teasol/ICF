# Current Status

> **이 파일은 "지금 무엇을 하고 있는가"만 담는다. 100줄을 넘기지 않는다.**
> 수치·기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [결정 이력](history/archive.md)가 정본이다. **여기에 복사하지 않는다.**
> 종료된 RU의 상세는 [`history/research_units_all.md`](history/research_units_all.md)로 이관한다.

| 항목 | 값 |
|:---|:---|
| **Last Updated** | 2026-09-09 16:52 (KST) |
| **Status** | CLEAN — D-039 SH 정식 통합·집계 리팩터링 완료 (RU 없음) |
| **Host / Node** | `nexgem-s1` · RTX A5000 8장 · driver 580.126.09 · Slurm 명령 없음 |
| **Environment** | uv venv `.venv` · Python 3.12.11 · PyTorch 2.14.0+cu130 · Lightning 2.6.5 |
| **Active Job** | 없음 |
| **회귀 테스트** | 121 tests · `OK` (2026-09-09) |

---

## 2026-09-09 — SH 브랜치 정식 통합 및 집계 계층 리팩터링 (D-039)

- **SH를 `src/models/branches/sh.py`로 이관**했다. §222 때 SJ(구 SHJ)만 이관되고 SH는
  `scripts/test_pathobench.py` 모놀리스에 남아 있던 기술 부채를 해소했다. `weight_sh`
  (기본 0.0)·`sh_dim`·`sh_wide`·`sh_lambda`를 `TrainingFreeConfig`에 신설해 SH가
  환경변수로만 조작되던 구조를 끝냈다. 기본 구성은 비트 단위 불변이다.
- **결함 수정**: `context_loo_stacking`의 branch_pool에 SJ가 전달되지 않아
  `context_loo_*` 집계에서는 `weight_sj`를 켜도 SJ가 조용히 누락됐다. SJ·SH를 pool에
  연결했고 회귀 핀 테스트를 추가했다 (`tests/test_sh_branch.py`).
- **중복 제거**: voting.py 5개 집계 함수에 수동 반복되던 SJ fallback(`m_shj`)과
  브랜치별 on/off 분기를 헬퍼로 통일했다. SHJ alias(`shj.py`, `weight_shj`)는
  `D-038` 영구 적용에 따라 유지한다.
- 회귀 테스트 121 tests `OK` (신설 8건 포함). RU-86·87·88 종료 요약은
  [결정·이력](history/archive.md) 말미로 이관했다.
- **레거시 정리 (사용자 지시)**: `tests/history/`(46파일, 스위트 미수집)·
  `tests/fixtures/`·`scripts/archive/`(93파일)를 삭제했다. 고아 모듈
  `models/mla.py`·`utils/schedulers.py`와 하위 호환 facade
  (`datasets/synthetic_data.py`·`models/ct_readout.py`)도 제거했고, facade
  참조 10곳을 실제 패키지(`datasets/synthetic`·`models/ct`)로 재지정했다.
  `set_transformer_ridge.py`·`src/modules/`·`baseline.py`는 **보존한다** —
  공식 평가 경로(`eval_v121.sh` → `test_pathobench.py`)가 lineage 모델의
  신규(head-less) 인스턴스로 v121 마진을 계산하므로 live 의존이다.

## 진행 중 RU

없음. 다음 RU의 질문·대조·예산은 사용자와 확정한다.

## 실행 환경 — 현재 접속 호스트

현재는 `nexgem-s1`이다. `.venv` 버전은 헤더와 일치하며 시스템 `python3`는 3.8.10이다.
`source scripts/node_env.sh`는 `.venv`와 GPU 8장을 탐지하고, GPU 0~7 CUDA matmul 정상 동작을
2026-09-07 확인했다. `sinfo`·`squeue`가 없어 **이 호스트는 직접 실행 유형**이다(`D-030`).
`nexgem`을 통해 제출할 때만 공용 [Slurm 규칙](/home/kimds/agent_rules/slurm_rules.md)이 적용된다.

### Immediate Next Command

```bash
cat docs/reports/RU-88_car1_fingerprint_panel.md
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
- **bf16 정밀도 부채는 실측으로 종료됐다 (RU-87).** 정밀도 간 차이는 절대 AUROC의 과제별
  fold-mean에서 최대 `0.0837%p`, 대응 Δ의 fold-mean에서 `0.0674%p`이며, 두 arm의 공통 성분으로
  **짝지은 Δ에서 상쇄된다** — `δ_min = 0.3%p`는 잡음 아래가 아니다. 다만 상쇄는 fold-mean에서만
  일어나므로 **단일 fold 수치를 근거로 쓰는 서술은 약 0.6%p 잡음을 안는다**(per-fold 95백분위
  `0.638%p`). 폐기된 `최대 0.5%p` 표기의 경위는 결정 이력 `D-037`.
- `adaptive_trimmed`의 `adaptive_tau`는 무효 인자(죽은 코드).
- `history/archive.md`에 **§199·§200 절 번호가 각각 중복** ([`closed_axes.md` §4](closed_axes.md)).

_by Claude Opus 5 (Main) on nexgem-s1 at 2026-09-08_

_by GLM-5.3-Flash on nexgem-s1 at 2026-09-09 16:52_
