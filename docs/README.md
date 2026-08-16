# Documentation map

**Last updated**: `2026-08-16`
**Active 구성**: **v107 — 학습 파라미터 0, K=256** (§142, 사용자 결정). within-slide PCA 사영 +
고정 3상수 head, 정식 경로 SEAL macro **0.6945**, **seed std 0.00000**. 실행: `bash scripts/eval_v107.sh <gpu> <tag>`.
v106(K=128, 0.6864)과 직전 baseline v98(8 seed 0.6852)은 historical.
⚠️ v98 8 seed 평균 대비 +0.0093이지만 **v98 상위 두 시드와 4-seed 앙상블(0.6951)에는 아직 진다** —
"v98을 이겼다"고 쓰지 말 것. 실행 중: 없음.

> **새 세션이 먼저 알아야 할 3가지 (2026-08-15, 상세는 `current_status.md` 최상단)**
> 1. **데이터 분포 축은 닫혔다 (§129)** — 합성 에피소드를 실제 UNI2 통계에 맞추는 것은 도움이 안 될
>    뿐 아니라 **닫은 격차가 많을수록 단조로 나빠진다**(v102 t=−2.70 기각, v100 t=−3.59 기각).
>    이 축에서 새 arm을 설계하지 말 것.
> 0. **학습 파라미터 0 구성이 확정됐다 (§139)** — 사영은 fold context의 within-slide PCA, head는
>    라벨 반대칭이 강제하는 상수 3개(CV:DD:CT = 1 : −0.238 : +0.199). 재현:
>    `ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1`.
> 2. **문제는 편향이 아니라 분산일 수 있다 (§130·§131)** — **4 seed의 최소 검출 효과가 0.0121**인데
>    15개 arm이 쫓던 효과는 전부 ±0.005 안쪽이었다. **"미판정"은 "효과 없음"이 아니라 "측정 불가"다.**
>    시드 앙상블은 학습 비용 0으로 +0.004~0.006을 주고 **독립 3회 재현**됐다.
> 3. **판정은 게이트가 자동으로 하지 않는다 (§118)** — 최종 승격/기각은 macro + task 10개 전부의
>    성능대별 패턴 + arm 간 일관성을 종합한 **사용자 판단**이고, 보고 형식이 정해져 있다(§118-3).

문서는 **새 대화 세션으로 접속하는 Agent가 최우선으로 읽는 Living 문서 5개와 현행 proposal 1개(`docs/` 루트)**, **과거 기록/딥다이브 분석서([`docs/history.md`](history.md))**로 이원화하여 관리합니다.

---

## 1. 새 세션 접속 Agent가 최우선으로 정독하는 Living 문서 (Docs Root)

사용자가 매번 새 채팅 세션으로 접속할 때, 새로 시작한 Agent는 아래 `docs/` 최상위 루트의 Living md 파일 5개와 현재 `architecture_*_proposal.md` 1개를 우선 정독하고 **Git commit log/diff**를 조회하여 작업 맥락을 동기화합니다:

1. [`agent_handoff.md`](agent_handoff.md): 새 세션 Agent 초기화 수칙, Git 기반 워크플로우, 실행 환경, 타임아웃, 테스트 검증, Docs/Config 정리 규칙
2. [`current_status.md`](current_status.md): 개발 현황, 최신 실증 수치, 판정 프로토콜(§107-3 게이트 + **§118 사용자 종합 판단**), 열린 과제, Next Action Plan (SSOT). §2~§97 본문은 `history.md` §20–§23으로 아카이빙되고 스텁+포인터만 남았다(§101)
3. [`current_architecture.md`](current_architecture.md): 활성 **v83 linear head** relation 구조와 역사적 CV-only/Encoder+Ridge 비교, 데이터·학습 계약
4. [`current_experiments.md`](current_experiments.md): 실험 전략과 검정력, 평가 프로토콜, Stage 1~3 실행 명령어 및 실증 수치
5. [`README.md`](README.md): 문서 맵 및 갱신/아카이빙 가이드라인
6. `architecture_*_proposal.md`: 현재 활성 개선 proposal. 완료·폐기 시 핵심 결론을 `history.md`에 기록하고 원문은 git 이력에 보존. (**2026-08-09 현재 활성 proposal 없음**)

---

## 2. 과거 기록 및 딥다이브 분석서 (History & Deep-Dive Archives)

[`docs/history.md`](history.md)는 과거 설계·딥다이브 분석·실험 판단 근거의 **통합·요약본**입니다. `docs/history/` 폴더의 개별 문서(딥다이브 분석서, 옛 아키텍처 설계안, 폐기된 proposal, 과거 세션 아카이브)를 2026-08-09에 한 파일로 통합했고, 원문은 git 이력에 보존됩니다. 현재 실행 지침이 아니며 `docs/` 루트를 깨끗하게 유지합니다.

주요 내용:

- **§1** 버전 관리 정책(정수 `architecture_version`, semver 폐기, v18→v22 미적용 수정)
- **§2** 평가 방법론 불변식(paired bootstrap CI, 합성 지표 불신, SEAL 10개 macro 평균 판정)
- **§3** bf16-mixed/수치 안전 계약
- **§4~§17** 아키텍처 진화 연대기 — v18 learnability ladder · v20 Candidate A/B · v21 retrieval 제거 · v22~v24 bag-collapse · 0.70 정보 상한 · v26/v27/v29 폐기 · v30(B1+B2) · Musk 0.95 로드맵 · v31~v33 CCER/DR-CCER/MR-BagPFN 폐기 · v34 large-context · v35 스트리밍 · v36/v37 기각 · CV-only 전환 · v41 sketch 기하
- **§18** 운영·인프라 교훈(NCCL P2P, launcher/큐 함정, purge 교훈)
- **§19** 다시 열면 안 되는 결론 / 재개 시 전제 (quick reference)
- **§20~§23** (2026-08-12 추가) CV-2 손잡이 소진과 계보 B 일반화 실패 · 합성 데이터 축(per-bag cardinality, XOR, manifold) · canonical CV/DD/CT와 v70~v77 계보 · v77 파생 arm 전수 기각과 판정 프로토콜 전환(eigen 미분 금지 포함)
- 각 항목에 원본 파일명·작성 시점을 출처로 병기

폐기 architecture/연구 진단 테스트의 archive 정책은 [`../tests/history/README.md`](../tests/history/README.md)를 참고합니다.

---

## 3. 갱신 및 관리 수칙 (Maintenance Rules)

- **Living 문서 유지**: `docs/` 최상위 루트에는 5개의 Living 문서와 현행 proposal 1개만 유지합니다.
- **Git 커밋 동기화**: 세션 핸드오프 시 작업을 남김없이 커밋하고 커밋 내역/diff를 `agent_handoff.md` 및 `current_status.md`에 반영합니다.
- **아카이빙 규칙**: 특정 버전 딥다이브 보고서나 계획 문서는 완료 시 **핵심 결론(ADR·트레이드오프·레슨)을 `docs/history.md`의 해당 시기 절에 추가**하고, 개별 원문 파일은 새로 만들지 않습니다(원문은 git 이력에 보존). `docs/` 루트를 단순하고 가독성 높게 유지합니다.
- **Config 루트 관리**: `configs/` 루트에는 **현재 활성 파이프라인의 entry point만** 둔다
  (상세는 [`agent_handoff.md`](agent_handoff.md) §7). **2026-08-15 기준 루트 구성**:

  | config | 역할 |
  |---|---|
  | `train_v83_linear_head_1536_1gpu.yaml` | **canonical baseline** (§109). 모든 arm의 비교 대상 |
  | `train_v82_medium_classsep_1536{,_1gpu}.yaml` | 직전 baseline(historical) |
  | `train_v77_hard_orthogonal_1536{,_1gpu}.yaml` | historical control |
  | `train_v86_noise / v87_rare` | §105-6 데이터 스윕 재검증 — 둘 다 null |
  | `train_v89_episode_shape / v91_cell_axis / v92_big_bags / v93_cell_axis_clean` | §115 에피소드 **모양** 축 — 전부 닫힘/기각 |
  | `train_v88_population_attention` | 레이블 조건 분기 — 기각(§114) |
  | `train_v90_class_prior` | 클래스 비율 축 — 기각(§118) |
  | `train_v94~v97, v98_p1_reverse, v99_p2_norm, v100_nuisance_min, v101_donor_shift_zero, v102_tail_bagshared` | §123 cell **값 분포** 축 — 전부 null~기각(§129) |

  ⚠️ v94/v95/v96/v97은 **1 seed 스크리닝 전용**이다(헤더에 명시). §3-1 판정표에 넣지 말 것.
  §107-6(fixed P × Medium, v85)은 한 번도 실행되지 않은 채 **취소**됐고 config도 삭제됐다.
  종결된 arm 72개는 `configs/archive/` 아래 시대별
  폴더(`v34_largectx/`, `v40_v45_cvonly/`, `v50_v54_encoder/`, `v57_v61_data_arms/`,
  `v62_v68_hybrid/`, `v69_v76_relation/`, `v77_pop_residual/`, `v78_dd_gradient/`,
  `v79_dual_projection/`, `v80_v82_seed_batch/`, `v84_deep_head/`)로 이관하고 전부 `base_config` 없는
  자체 포함형으로 보관한다. ICI의 fold/seed는 config가 아니라 `--cv`/`--seed`로 주입하므로
  fold별 config를 만들지 않는다.
- **단일 출처화**: 동일한 수치나 진행 상태를 중복해 기록하지 않으며, 상태는 `current_status.md`에만 기록합니다.
- **자율 연속 실행과 추적성**: `current_status.md`에 다음 Action과 판정 기준이 명확하면 재확인 없이 실행합니다. 각 논리 단위의 결과·명령·로그/산출물 경로·판단·후속 Action을 SSOT에 기록하고 Git 커밋하여 다른 작업공간이 즉시 이어받을 수 있게 합니다.


---

## 2026-08-09 재편 (CV-only 전환)

- `current_architecture.md` / `current_experiments.md`를 **CV-only(v40~) 기준으로 전면
  재작성**했다. 이전 판(v34~v39 6-분기)의 핵심은 `history.md` §12~§16에 요약.
- v36 Q1 / v37 proposal은 **둘 다 기각**(§65). 현재 활성 proposal은 **없다** — 다음 노선(learnable 사영)이 정해지면 새로 작성한다.
- 판정 기준이 **SEAL 10개 task macro 평균**으로 바뀌었다(§71-4). `seal_univ2_baseline_17tasks.csv`의
  `in_seal=yes` 행이 대상이고, 러너는 `scripts/eval_seal_tasks.sh`다.
- `docs/history/` 폴더의 개별 문서를 **`docs/history.md` 단일 파일로 통합**하고 폴더를 제거했다(원문은 git 이력 보존).
