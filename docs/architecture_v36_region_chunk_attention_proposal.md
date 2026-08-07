# v36 Proposal: Zero-Init Region-Level Chunk Attention (ABMIL-gap bet)

**작성일**: 2026-08-08

**상태**: 제안 — v36은 rev.2 §8이 지목한 **"ABMIL 격차에 대한 올바른 베팅"** 아키텍처다.
zero-shot 유지(사용자 확정, §60) 하에서 지도 ABMIL을 이기기 위한 **선택적 region 집계**를
추가한다. 구현은 ① §3 chunk 단위(수치 무변화) 선행, ② 동일 게이트 통과 후에만 진행한다.

**기준선**: v35 Phase 0 (`train_v35_phase0_largebag_1536.yaml`, **rare-free** after §61) /
v34-1536 base 아키텍처 (`BaseModel` → `StructuredPopulationMetaClassifier`)

---

## 0. 한 줄 요약

> 우리는 이미 **MeanMIL 수준, ABMIL 미달**이다(§60 실측). ABMIL과 MeanMIL을 가르는 것은
> **task별로 학습된 선택적(region-level) attention**이다. 평균(mean pooling)이 버리는
> **region 간 이질성**(grade류 task 신호)을 보존하는 **permutation-invariant attention head**를
> **zero-init residual**로 추가하면, 초기화 직후에는 현재와 **수치 동일**(회귀 불가)하고
> 학습으로 region 선택을 얻어 ABMIL 격차를 메운다. **단독 arm + 동일 게이트**.

---

## 1. 동기 — ABMIL 격차 (실측, §60)

공식 50-fold macro-AUC (SEAL과 동일 프로토콜):

| task | v35 (zero-shot) | SEAL ABMIL (지도) | SEAL MeanMIL (지도) | v35 vs ABMIL | v35 vs MeanMIL |
|---|---|---|---|---|---|
| LUAD EGFR | 0.7889±0.092 | 0.830±0.089 | 0.777±0.099 | **−0.041** | **+0.012** |
| BRCA PIK3CA | 0.5743±0.109 | 0.595±0.103 | 0.544±0.120 | **−0.021** | **+0.030** |

- v35는 두 task 모두 **지도 MeanMIL과 동급~우위**, **지도 ABMIL에는 −0.02~−0.04 뒤처짐**.
- 목표: zero-shot 유지 상태에서 이 **ABMIL 격차를 절반 이상 축소**(EGFR −0.041 → −0.02 이내)하고,
  최종적으로 ABMIL을 이기는 것.

---

## 2. 진단 — 왜 MeanMIL 수준에 머무는가

1. **현재 집계는 평균 계열**: bag → 구조적 token → meta-classifier 입력은 `project_structured_tokens`
   (또는 mean pool)로 **bag 전체의 요약 token**을 만든다. 이는 region 간 정보를 이미 압축/평균한다.
   rev.2 §2.10: "chunk token 평균은 region-level mean pooling" — 즉 우리 위치는 MeanMIL과 같은
   **정보 손실 구조**다.
2. **선택적 기제가 없음**: 원래의 유일한 선택 기제(rare branch)는 **P0-b로 기여 ≈ 0 확인**됐고(§61,
   |Δpooled| 0.0009) 제거했다. 즉 현재 모델에는 "어느 부분을 보고 판단할지"를 학습하는 메커니즘이 없다.
3. **ABMIL의 이점**: 지도 ABMIL은 train fold 라벨로 **어느 region(타일 집단)이 task 판별에 중요한지**를
   학습한다. zero-shot에선 이 선택성이 결여되어 MeanMIL 수준에 머문다.

→ **결론**: ABMIL 격차는 데이터(대형 bag 등)가 아니라 **아키텍처(선택적 region 집계)** 문제다.
데이터 arm(v35)은 이미 실측으로 marginal(§60)이므로, **아키텍처 구현을 먼저** 하는 것이 옳다(사용자 결정).

---

## 3. 설계 — zero-init region-level chunk attention (§8 구체화)

### 3.1 region token (chunk 단위 bag 요약)

- bag의 타일을 **sequential chunk**(`chunk_cells`, 기본 2048)로 분할한다. h5 타일 순서는 대체로
  공간 순서이므로 각 chunk = **공간적으로 인접한 region 요약**(bag당 `N/2048 ≈ 15`개).
- 각 region에 대해 현재 bag-view 계열의 **경량 요약**을 만들면 된다 (기존 `_bag_view`의 통계를
  chunk 단위로 재사용, 또는 소형 region encoder). region 수는 bag당 수십 개로 **저비용**.

### 3.2 permutation-invariant attention (region-level ABMIL 유사물)

- region token set 위에 **permutation-invariant attention**을 씌운다
  (SetCrossAttention 유사 구조 또는 소형 self-attention).
- 이것이 **평균이 버리는 region 간 이질성**을 보존한다 — grade류 task(이질성 신호)에서 기대 효과.
- 출력은 region-aggregated bag token으로, meta-classifier 입력 경로(`project_structured_tokens`와
  유사한 위치)에 들어간다.

### 3.3 zero-init residual (회귀 불가 보장)

- 저장소 관행(§59 "byte-identical", §56 "forward 동치", v31 zero-init output head)에 맞춰
  **zero-init residual**로 넣는다:
  - 초기화 직후 출력 = 정확 **가중평균(§3.1 region 요약의 mean)**과 동일 → 기존 모델과 **수치 동일**.
  - v34/v35 **weight-only 초기화와 호환** → 기존 ckpt를 그대로 쓸 수 있음.
  - 회귀 시 **즉시 원복 가능** (residual을 끄면 기존과 동일).

### 3.4 rare 제거와의 관계

- §61에서 rare 제거 완료. v36의 region attention은 **rare와 독립** — rare가 하던 "선택성"을
  대체·확장하는 것이 아니라, **새로운 region-level 선택**을 추가하는 것. P0-b가 rare≈0임을
  보였으므로 region attention의 기여를 깨끗하게 측정할 수 있다.

---

## 4. 구현 계획

| 단계 | 내용 | 게이트/검증 | 위험 |
|---|---|---|---|
| **0 (선행)** | rev.2 §3 **chunk 단위(bag 내부) 정확 스트리밍** — bag 단위는 완료, chunk 단위 미구현 | §7-1 수치 동일(‖Δ‖∞<1e-4), 41 tests | 낮음 (수치 무변화) |
| **1** | region token 생성 (chunk 단위 bag 요약) | region 수=1일 때 기존과 동일 | 낮음 |
| **2** | zero-init attention head 구현 | **init 시 기존과 수치 동일** (§6-1) | 중 |
| **3** | **단독 arm 학습** (v35 데이터, rare-free, episode-matched) | v35 대비 paired 50-fold | 중 (훈련 비용) |
| **4** | 공식 50-fold 평가 (3+ task) + SEAL 재비교 | §5 게이트 판정 | — |

- **단독 arm 원칙**: v36 head를 v35 구성에 **하나만** 얹는다 (데이터·다른 아키텍처 변경 없음).
  원인 분리를 위해 회귀 시 한 번에 한 요소만 되돌린다 (§42 arm C 교훈).

---

## 5. 게이트/판정 기준 (동일 게이트)

1. **P0 (무료, 학습 0)**:
   - ⓐ zero-init head가 **init에서 기존과 수치 동일** (‖Δ‖∞ < 1e-4, 동일 ckpt).
   - ⓑ attention이 실제로 region을 선택하는지: attention weight 분포/엔트로피 진단 + region 셔플
     ablation (region 순서를 섞어 결과가 변하는지).
2. **본 게이트 (paired 공식 50-fold, task ≥ 3)**:
   - v35(rare-free) 대비 **pooled AUROC 평균 +0.005 이상**.
   - SEAL **MeanMIL 대비 우위 유지** + **ABMIL 격차 절반 이상 축소** (EGFR −0.041 → −0.02 이내).
3. 미달 시: **v36 미채택, 재현 코드만 보존** (v31/v32/v33/v35 데이터 arm과 동일 규율).

---

## 6. 검증 계획

1. **init 수치 동일성**: zero-init head on/off → final logit ‖Δ‖∞ < 1e-4 (동일 ckpt, weight-only).
2. **region 경계 불변성**: `chunk_cells` {2048, 1024, 512}를 바꿔도 결과 동일 (= §4.3 P0-c, region
   경계가 결과를 바꾸면 구현 버그).
3. **선택성 진단**: attention weight의 region 분포 — **grade류/이질성 task**에서 특정 region에
   집중되는지, EGFR/PIK3CA에서의 분포와 대조.
4. **50-fold**: EGFR / STK11 / Histologic_Grade 등 3+ task, SEAL 재비교 (macro-AUC, §53 프로토콜).
5. **Musk 재확인**: region attention이 소형 bag(≤34)에 회귀를 일으키지 않는지 (Musk 0.95 목표).

---

## 7. 리스크 / 오픈 문제

- **region 내 정보 손실**: chunk 안에서 다시 평균하면 region token 자체가 정보를 잃을 수 있음 —
  경량 region encoder(선택) 또는 충분통계 축약(§3)으로 완화. 검증 항목 1·2가 이를 감지.
- **zero-init attention의 학습 속도**: zero-init 잔차는 초기 학습에서 천천히 깨어날 수 있음 —
  residual scale warm-up 또는 초기 attention temperature 검토.
- **훈련 비용**: region attention은 bag당 ~15 token → **저비용** (기존 bag token 수준).
- **§3 chunk 스트리밍 미구현이 선행 필수**: 기준선(v35 rare-free 50-fold)을 오염시키지 않으려면
  먼저 수치 무변화로 끝내야 한다 (rev.2 §8 "§3이 선행되어야 이 실험의 기준선이 깨끗해진다").
- **ABMIL은 지도 학습이라는 본질적 이점**: zero-shot이 ABMIL을 완전히 이기는 것은 task가 어려울수록
  (ABMIL이 랜덤에 가까운 VHL/PIK3CA) 유리하다. 강한 task(STK11 0.908)에서 ABMIL 초월은 어려울 수
  있음 — 목표는 **전체 평균 우위 + 약한 task에서 우위**로 설정.

---

## 8. 참고

- rev.2 §8 (docs/history/architecture_v35_tokenonly_chunked_query_proposal.md) — "ABMIL 격차에 대한
  올바른 베팅", §2.10 방향성 진단.
- current_status.md §60 (v35 50-fold 완료 + SEAL 비교), §61 (P0-b + rare 제거).
- SEAL baseline: docs/seal_univ2_baseline_17tasks.csv (VHL 0.538 / PIK3CA 0.595 등 최약 task 포함).
- 커밋 `5869535` — query-count-invariant margin; `dc7ea59` — §60 50-fold 결과; `d6e3ba6` — P0-b + rare 제거.
