# Current development status & multi-location sync SSOT

**Last updated**: `2026-07-29 17:05:00 KST`
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)
**Architecture Version**: `22` (`architecture_version = 22`)
**Purpose**: 연구실 / 집 / 노트북 3개 작업 환경 간 대화 기록 비동기화 문제를 해결하기 위한 Single Source of Truth (SSOT) living document.

---

## 1. 멀티 작업공간 (연구실/집/노트북) 바톤 터치 지침

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 원칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 새로 접속한 Agent는 **`docs/` 최상위 루트의 Living md 파일 5개(`agent_handoff.md`, `current_status.md`, `current_architecture.md`, `current_experiments.md`, `README.md`)만 최우선으로 정독**하여 전체 개발 맥락과 프로젝트 규칙을 파악합니다.
> 3. 터미널 조회가 필요한 명령어는 NVML/쉘 hang 방지를 위해 **반드시 `timeout 3s ps aux | grep python`과 같이 타임아웃**을 적용합니다.
> 4. 코드 변경 시 unittest 통과 필수:
>    `timeout 1500s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"`

---

## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture v22)

* **Python Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python`
* **Torchrun Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun`
* **Target Hardware**: NVIDIA B200 GPU 1장 (`CUDA_VISIBLE_DEVICES=0`, 183GB VRAM)
* **Precision Policy**: `bf16-mixed`
* **핵심 수학 기술 4종** (v19부터 이어져 v22에서도 그대로 유지):
  1. **Z-Score Bag Studentization**: Donor Centroid/Std 기반 세포 표현 스케일 정규화.
  2. **Top-1% Sparse Evidence Module**: 배경세포에 희석되는 희귀 반응 신호 핀포인트 추출.
  3. **Covariance Subspace Shrinkage** (`subspace_shrinkage: 0.25`): 노이즈 축 whitening 방어 및 NaN 예방.
  4. **Auxiliary Pairwise Ranking Loss** (`weight: 0.10`): CE 0.685 부근 gradient 소멸 탈출.
* **Batched Multi-Episode Forward**: `forward_episode_batch` 및 `BaseModel.forward`의 4D 분기가 `[episodes, bags, cells, dim]`을 한 optimizer step에 처리 (v22에서도 유지). 검증: `tests/test_batched_episode_forward.py`.
* **Retrieval 없음**: v22는 context 축소(retrieval) 계층이 **없습니다**. 에피소드의 전체 context bag이 그대로 aggregator에 들어갑니다. 제거 근거는 §4 참고.

---

## 3. 실험 현황

### ✅ v22 기준선 (2026-07-29 확정) — **버그 수정 반영본이 공식 기준선**

| 항목 | 값 |
|---|---|
| Config | `configs/train_v22_medium.yaml` (20 epoch, 512 steps/epoch = **10,240 steps**, Phase 1과 동일) |
| 코드 상태 | **Cholesky backward + rank-local 수정 반영 후** (커밋 `be36c59`) |
| Best `val_ce_loss` | **0.5946** (epoch 13) |
| 합성 val AUROC | **0.7466**, 95% CI [0.716, 0.776] (episode cluster bootstrap, 104 episodes / 1,698 query) |
| 합성 val Log Loss | 0.5943 |
| Task별 AUROC | composition 0.8022 / combined 0.8170 / interaction 0.7453 / state 0.6595 / **covariance 0.6122 (최난이도)** |
| 체크포인트 | `checkpoints/20260729_160643/v22_medium_fixed/epoch=013-val_ce_loss=0.5946.ckpt` |
| 예측 파일 | `predictions/synthetic_v22_baseline_fixed.pt` |
| 로그 | `logs/20260729_160643/v22_medium_fixed.out` |

**버그 수정 전후 비교** (`scripts/compare_predictions.py`, paired cluster bootstrap):

| | 수정 전 (`100611`) | **수정 후 (`160643`, 공식)** |
|---|---:|---:|
| AUROC | 0.7463 [0.715, 0.775] | **0.7466 [0.716, 0.775]** |
| `val_ce_loss` | 0.5930 | 0.5946 |
| paired 승률 | \- | **0.42 (구분 불가)** |

> [!NOTE]
> **두 수치는 통계적으로 구분되지 않습니다** (승률 0.42, CI 거의 동일). Cholesky 수정은 성능을 올리는 변경이 아니라 **near-singular 상황에서 backward가 non-finite로 터지는 것을 막는 보험**입니다. 평균적인 run에서는 차이가 안 나는 것이 정상이며, 그래도 반영한 이유는 언제 터질지 모르는 잠재 위험을 남겨둘 이유가 없기 때문입니다.
> 이전 수정 전 기준선(`0.7463` / `checkpoints/20260729_100611/`)은 참고용으로만 남기고, **앞으로의 비교는 수정 후 기준선을 씁니다.**

**v21 Phase 1과 사실상 동일**: `val_ce_loss` 0.5946 vs 0.5921 (차이 0.0025), val_loss 궤적도 0.667 → 0.650 vs 0.667 → 0.644로 겹칩니다. Phase 1도 full-context였으므로 **retrieval 제거가 합성 사전학습 성능을 전혀 훼손하지 않았다는 확인**입니다.

**앞으로 아키텍처 변경은 이 수치와 비교합니다** (§5 전략에 따라 ICI가 아니라 여기서 판단). 비교 시 `scripts/compare_predictions.py`로 위 예측 파일과 paired cluster bootstrap을 돌릴 것.

### 🔴 진단 결과: covariance 분기가 자기가 가진 정보를 못 쓰고 있음 (2026-07-29)

`scripts/diagnose_oracle_covariance_upper_bound.py`를 v22 config로 실행해 **descriptor × relation 조합별 상한**을 측정했습니다. 아래는 최초 측정(기본 val split, covariance 18 episodes)이며, 표본이 작아 T1-0에서 재확인했습니다(바로 아래).

| descriptor | relation | AUROC | 비고 |
|---|---|---:|---|
| `latent_dispersion` | `standardized_distance` | 1.0000 | ⚠ **oracle** (ground-truth latent 사용, 달성 불가) |
| **`observed_covariance`** | **`prototype_cosine`** | **0.8908** | ✅ **관측만으로 달성 가능한 최고치** |
| `observed_covariance` | `multiscale_rbf` | 0.8742 | 관측 가능 |
| `observed_covariance` | `standardized_distance` | 0.8546 | 관측 가능 |
| `observed_local_distance` | `multiscale_rbf` | 0.5959 | 관측 가능 |
| `observed_spectral` / `observed_local_anisotropy` | (전부) | 0.48~0.56 | 거의 무신호 |
| **학습된 v22 모델** | — | **0.6122** | **관측 상한 대비 −0.2786** |

> [!IMPORTANT]
> **covariance는 "본질적으로 어려운 과제"가 아닙니다.** 관측 가능한 공분산 스케치에 단순 prototype-cosine만 태워도 **0.89**가 나오는데, 학습된 모델은 **0.61**입니다. 즉 모델이 **이미 계산해 두고 있는 공분산 정보를 제대로 활용하지 못하고 있습니다.** 이것이 현재 가장 큰 개선 여지입니다.
>
> 처음에는 task별 AUROC 순서(composition 0.80 > state 0.66 > covariance 0.61)가 생성기의 effect scale 순서(1.40 > 0.72 > 0.55)와 정확히 일치해서 **난이도 아티팩트로 보였지만**, 이 진단이 그 해석을 뒤집었습니다. 난이도가 낮아서가 아니라 아키텍처가 못 쓰는 것입니다.

#### ✅ T1-0 확인 완료 (2026-07-29): 206 episodes에서 상한 유지

18 episodes는 표본이 부족해 재확인이 필요했습니다. `--val-episodes 1000`(covariance 206 episodes, 11배)으로 재실행한 결과:

| descriptor | relation | 18 eps | **206 eps** | 95% CI (206) |
|---|---|---:|---:|---|
| **`observed_covariance`** | **`prototype_cosine`** | 0.8908 | **0.8931** | **[0.876, 0.910]** |
| `observed_covariance` | `multiscale_rbf` | 0.8742 | 0.8903 | [0.873, 0.908] |
| `observed_covariance` | `standardized_distance` | 0.8546 | 0.8334 | [0.812, 0.856] |
| `observed_spectral` | (최고) | 0.5609 | 0.5787 | [0.557, 0.600] |
| `observed_local_*` | (최고) | 0.5959 | 0.5450 | [0.522, 0.569] |
| **학습된 v22 모델** | — | 0.6122 | 0.6122 | — |

> [!IMPORTANT]
> **상한이 유지됩니다: 0.8931, CI [0.876, 0.910].** 모델의 0.6122는 이 구간 **한참 아래**로, 겹치지 않습니다.
> **→ Tier 1은 진행할 가치가 있습니다. 헤드룸 약 +0.28은 실재합니다.**
>
> 추가로 드러난 것: 공분산 신호는 **`observed_covariance` 스케치에만** 있습니다. `spectral`(0.58), `local_distance`/`local_anisotropy`(0.53~0.55)는 전부 무작위에 가깝습니다. 즉 **어떤 descriptor를 쓰느냐가 결정적이고, 모델은 이미 올바른 것(`_covariance_sketch`)을 계산하고 있습니다.** 문제는 계산이 아니라 **활용**입니다.
>
> relation 측면에서는 `prototype_cosine`(0.8931)과 `multiscale_rbf`(0.8903)가 사실상 동률이고 `standardized_distance`(0.8334)만 뒤집니다. 현재 모델은 `mode: learned_head`를 씁니다 — T1-1(b) 가설의 직접적 근거입니다.

### v21 이하 과거 수치 (참고용)

> [!CAUTION]
> **아래 수치들은 v21 이하에서 측정된 것이며 v22 코드로 재현되지 않습니다.**
> v22는 `architecture_version`이 22이므로 **기존 체크포인트는 전부 로드 불가**입니다 (`ModelInterface.on_load_checkpoint` 버전 게이트가 거부).

| Phase | 설명 | 지표 | 비고 |
|---|---|---|---|
| Phase 1 (v21) | Medium 합성 사전학습, full context | `val_ce_loss: 0.5921` | 20 epoch, 10,240 steps — v22 기준선이 재현함 |
| Phase 2 (v21) | Hard 합성 사전학습, full context | `val_ce_loss: 0.6845` | 50 epoch |
| Phase 4 (v21) | ICI 5-fold, Naive retrieval | AUROC 0.5524 / LL 0.7288 | 95% CI [0.424, 0.677] |
| Phase 6b (v21) | ICI 5-fold, Signal-Aware retrieval | AUROC 0.5481 / LL 0.8672 | 95% CI [0.421, 0.674] |
| **Phase 6c (v21)** | **ICI 5-fold, retrieval 없음** | **AUROC 0.5454 / LL 0.7921 / Acc 0.6092** | 95% CI [0.419, 0.674] — v22 기본 구성에 해당 |

**핵심**: 위 세 ICI 구성의 **95% 신뢰구간이 전부 0.5(무작위)를 포함하고 서로 거의 완전히 겹칩니다.** paired bootstrap 승률도 0.52~0.55로 동전 던지기 수준입니다. 즉 n=87 코호트에서 이 차이들은 **검출 불가능한 노이즈**입니다. 상세 근거: [`history/v21_retrieval_investigation.md`](history/v21_retrieval_investigation.md) §4-⑧.

---

## 4. v22 결정: retrieval 완전 제거 (2026-07-29)

### 제거 근거 (3대 가설 검증 결과)

1. **retrieval은 ICI에서 이득이 없음**: retrieval을 끈 Phase 6c가 켠 Phase 6b와 AUROC 동일(0.5454 vs 0.5481)한데 Log Loss(0.7921 vs 0.8672)와 Accuracy(0.6092 vs 0.5747)는 오히려 **더 좋음**. ICI는 fold당 context가 ~69명뿐이라 24명 선별은 가용 labeled context의 65%를 버리는 것.
2. **구현이 문서화된 설계와 달랐음**: 설계는 "query 1명당 24명 맞춤 선별"이었으나, 실제로는 두 구현 모두 query 전체에 **공용 context 1세트**를 적용 (외부 collator는 첫 query만, 모델 내부는 query 평균 사용). 공용 context는 각 query의 개별 top-24와 평균 61.1%만 겹쳤고 반응자(y=1)에서 ~50%로 최악.
3. **Phase 5 사전학습은 Phase 1의 1/4만 학습됨**: `episode_batch_size` 8→32로 올리면서 `max_epochs: 20`을 그대로 둬 optimizer step이 10,240 → 2,560으로 감소. 게다가 validation이 retrieval 없이 수행되어 `epoch=014` 체크포인트가 학습 모드와 다른 기준으로 선택됨.

무엇보다 **위 차이들이 통계적으로 구분 불가능**하다는 점이 결정적이었습니다. 검출력 없는 지표 위에 복잡한 계층을 유지할 이유가 없다고 판단해 제거했습니다.

### 제거 범위

| 대상 | 조치 |
|---|---|
| `BaseModel.extract_bag_features` / `retrieve_context_indices` / `_retrieve_context_indices_impl` / `retrieve_context_indices_per_query` | 삭제 (303줄) |
| `BaseModel.forward(retrieval_k=...)` 파라미터 | 삭제 |
| `RetrievalEvaluationEpisodeCollator` / `RetrievalSyntheticTrainingEpisodeCollator` / `SignalAwarePretrainEpisodeCollator` | 삭제 |
| `ModelInterface`의 `retrieval_k` 배선 4곳 + `_build_model` pop-list 항목 | 삭제 |
| `scripts/test.py --retrieval-k` | 삭제 |
| retrieval 계열 config 12개, launch 스크립트 3개, VRAM 벤치 2개 | `configs/archive/v21_retrieval/`, `scripts/archive/v21_retrieval/`로 이관 |
| `tests/test_feature_retrieval.py`, `tests/test_large_context_pretrain.py` | 삭제 → `tests/test_batched_episode_forward.py`로 대체 |
| `architecture_version` | 21 → **22** |

**유지된 것**: v21 aggregator/meta-classifier 전부, 4대 수학 기술, batched multi-episode forward, 세션 중 고친 DataLoader CUDA/pin_memory 수정.

**복구 지점**: retrieval 최종 상태는 git tag **`v21-retrieval-final`** 로 보존되어 있습니다 (`git show v21-retrieval-final`).

### v20 롤백이 불가능했던 이유

사용자 요청은 "v20으로 롤백"이었으나 조사 결과:
* **v20은 코드로 존재하지 않습니다.** 이 브랜치 히스토리는 v18 → v19 → **v21**이며 (`ecf6199`가 19에서 21로 직접 점프), `main`은 아직 v18입니다. `configs/archive/v20/*.yaml`는 v19 코드 위에서 돌던 **설정 파일 시리즈**일 뿐입니다.
* **v21 ≈ v19 + retrieval + 사소한 2줄**. `ecf6199`가 baseline.py에서 제거한 실질 코드는 4줄뿐이고, "v21 4대 개혁"으로 문서화됐던 기술들은 이미 v19에 있었습니다.
* 따라서 retrieval 제거는 버전 롤백이 아니라 **덧붙은 계층의 절제**로 처리하는 것이 맞았고, 사용자 승인 하에 v22 신규 버전으로 진행했습니다.

---

## 5. 실험 전략 (2026-07-29 확정)

> [!IMPORTANT]
> **합성 데이터로 모든 결정을 내리고, ICI 실데이터는 최종 테스트에만 씁니다.**

| 단계 | 데이터 | 도구 | 반복 |
|---|---|---|---|
| 아키텍처 탐색·튜닝 | 합성 val | `scripts/evaluate_synthetic.py` (AUROC + episode cluster CI) | 자유롭게 |
| 최종 확인 | ICI 5 seed × 5 fold + 외부 코호트 26명 | `scripts/launch_ici_protocol.sh` → `scripts/evaluate_protocol.py` | **후보 확정 후 1회** |

**근거 (실측)**:

| | 표본 | AUROC | 95% CI | CI 폭 |
|---|---|---:|---|---:|
| ICI 5-fold (v21 Phase 6c) | 87명 | 0.5454 | [0.422, 0.664] | **0.242** |
| 합성 val (v22 기준선) | 104 episodes / 1,698 query | 0.7466 | [0.716, 0.776] | **0.060** |

합성 구간이 약 **4배 좁고**(그것도 episode cluster bootstrap이라는 보수적 계산으로), 신호도 훨씬 강합니다(0.75 vs 0.55). 합성은 `episodes_per_epoch`로 더 좁힐 수 있지만 ICI는 87명이 상한입니다. v21의 실패는 **검출력 없는 지표 위에서 아키텍처를 반복 비교한 것**이었고, ICI를 반복해서 보면 그 87명에 과적합됩니다.

**지켜야 할 선**: ICI 결과를 보고 아키텍처를 다시 고치기 시작하면 ICI는 더 이상 테스트 세트가 아닙니다. 또한 ICI에서 ±0.13 이내 변동은 그 자체로 아무 근거가 되지 않습니다.

**합성 평가의 CI는 query가 아니라 episode 단위(cluster bootstrap)로 계산합니다.** 한 episode의 query들은 context set과 생성 파라미터를 공유해 독립이 아니며, query 단위 재표집은 구간을 실제보다 좁게 만듭니다. 실질 표본 크기는 **episode 수**입니다.

---

## 6. 다음 작업 세션 Action Plan — 구조적 변경 및 실험 목록

> [!IMPORTANT]
> 모든 판단은 **합성 val**에서 하고 ICI는 손대지 않습니다 (§5). 후보마다
> `scripts/evaluate_synthetic.py` → `scripts/compare_predictions.py`로 기준선
> (`predictions/synthetic_v22_baseline_fixed.pt`, AUROC 0.7466 [0.716, 0.776])과
> paired cluster bootstrap 비교할 것.

### Tier 1 — covariance 분기 (근거가 가장 강함, 헤드룸 +0.28)

**T1-0. ✅ 완료 (2026-07-29) — 상한 확정, Tier 1 진행 승인.**
206 episodes에서 `observed_covariance + prototype_cosine` = **0.8931 [0.876, 0.910]**. 모델 0.6122는 이 CI 아래로 겹치지 않습니다. 헤드룸 +0.28 실재 확인. 상세는 §3 참고.
재현: `python scripts/diagnose_oracle_covariance_upper_bound.py --config configs/train_v22_medium.yaml --val-episodes 1000`

**T1-1. 왜 못 쓰는지 국소화 — 학습 없이 가능.**
`observed_covariance + prototype_cosine`이 0.89인데 모델이 0.61이라면 원인은 셋 중 하나입니다:
- (a) **fusion에서 희석**: covariance 항이 `covariance_residual_scale`(학습된 sigmoid) × `covariance_ridge_scale`로 두 번 감쇠되고, CSP head는 고정 0.50입니다. 학습된 scale의 **실제 수렴값을 먼저 찍어보세요** — 0에 가깝다면 분기가 사실상 꺼진 것입니다.
- (b) **learned head < prototype cosine** — **T1-0이 이 가설을 강화했습니다.** 진단에서 `prototype_cosine` 0.8931 / `multiscale_rbf` 0.8903이 최고인데 현재 모델은 `covariance_relation.mode: learned_head`(2-layer MLP)를 씁니다. `mode`를 prototype 계열로 바꿔 A/B가 가장 값싼 검증입니다.
- (c) **descriptor 손실**: `aggregator_covariance_sketch_dim: 64` 압축 문제. **단, T1-0에서 진단이 쓴 것도 같은 `agg._covariance_sketch`였고 0.89가 나왔으므로 이 가설의 우선순위는 낮아졌습니다.** 신호는 스케치 안에 있습니다.

  T1-0의 부수 소득: 공분산 신호는 **`observed_covariance`에만** 있고 `spectral`(0.58)·`local_*`(0.55)는 무작위 수준입니다. 대체 descriptor를 찾는 방향은 가망 없으니 시도하지 마세요.
`scripts/diagnose_covariance_relations.py`, `diagnose_covariance_subspace.py`, `diagnose_v19_branches.py`(분기별 AUROC 분해)가 이미 있으니 학습 없이 (a)~(c)를 가릅니다.

**T1-2. 원인별 구조 변경** (T1-1 결과에 따라 하나만 선택):
- (a)였다면 → `meta_covariance_residual_scale` 하한 도입 또는 covariance 항을 fusion 이전 단계로 이동
- (b)였다면 → `covariance_relation.mode`를 prototype cosine 계열로 교체 (진단에서 이미 최고 성능)
- (c)였다면 → `aggregator_covariance_sketch_dim` 증대 (64 → 128/256) 또는 압축 방식 변경

### Tier 2 — state 분기 (0.6595, 근거 중간)

**T2-1. state에도 동일한 상한 진단이 없습니다.** covariance에는 `diagnose_oracle_covariance_upper_bound.py`가 있지만 state용은 없습니다. **동형 도구를 만들어 state의 관측 상한을 먼저 재보세요.** 상한이 0.70 근처면 지금이 거의 최선이고, 0.85면 covariance와 같은 종류의 미활용 문제입니다. 도구 없이 아키텍처부터 건드리면 Tier 1에서 피한 실수를 반복하게 됩니다.

### Tier 3 — 방법론 (성능이 아니라 판단 신뢰도)

**T3-1. task별 effect scale 정규화 실험.**
현재 task별 AUROC는 생성기 effect scale(composition 1.40 / state 0.72 / covariance 0.55)에 오염되어 **아키텍처의 상대적 강약을 직접 비교할 수 없습니다.** 모든 task의 effect scale을 동일하게 맞춘 진단용 데이터 config를 만들면 "어느 메커니즘에 실제로 약한가"를 처음으로 공정하게 볼 수 있습니다. 학습된 모델을 그 데이터로 평가만 하면 되므로 재학습 불필요.

**T3-2. 합성 val 검정력 확보.**
현재 104 episodes → CI 폭 0.060. Tier 1/2에서 기대하는 개선폭이 그보다 작다면 검출이 안 됩니다. `val_dataset_kwargs.episodes_per_epoch`를 늘려 CI를 좁히세요 (episode 수가 실질 표본 크기, §5).

**T3-3. Stage 2 (Hard) 기준선.**
`configs/train_v22_hard_realworld.yaml` 아직 v22로 미실행 (v21 Phase 2 참고값 `val_ce_loss 0.6845`). Tier 1 변경이 medium에서만 좋고 hard에서 무너지지 않는지 확인할 대조군이 필요합니다.

### 하지 말 것

- **ICI 실행 금지** — 후보 확정 전까지. §5 참고, 지금 돌리면 테스트 세트 조기 소진.
- **근거 없이 covariance 아키텍처부터 뜯기 금지** — T1-0을 건너뛰면 18 episode 노이즈를 쫓게 됩니다.
- **effect scale 정규화 없이 task별 AUROC로 우열 판단 금지** (T3-1).

---

## 7. 평가 프로토콜 보강 (2026-07-29)

v21 조사의 결론("모든 비교가 노이즈였다")에 대응해 평가 체계를 다시 만들었습니다.

### ① 검정력 분석 — 가장 중요한 결과

`scripts/power_analysis.py` (baseline AUROC 0.55, 모델 간 상관 ρ=0.7 — 실측 Phase 6b vs 6c Pearson ρ=0.737 기반):

| 실제 AUROC 향상 | 검출 확률 |
|---:|---:|
| +0.02 | 15% |
| +0.05 | 26% |
| +0.10 | 66% |
| **+0.15** | **92%** |

> **n=87에서는 +0.13~0.15 미만의 개선을 검출할 수 없습니다.** v21이 쫓던 0.004~0.04 차이는 검출 확률 15~26%였습니다. 이것이 그 실험들이 결론에 도달하지 못한 근본 이유입니다.

### ② 발견: 자원의 4/5를 안 쓰고 있었음

- **seed partition 5개**(`SEED42/1234/2026/271828/314159`)가 디스크에 있었고 각각 87명을 5-fold로 덮는 **독립 분할**입니다 (CV0 기준 seed 간 val donor 겹침 1~5/18). **v21 실험은 전부 SEED42 하나만 사용.**
- **외부 코호트** `data/ICI_GSE285888_scConcept_512.pt` (26명, R 15 / NR 11)도 이미 존재하고 `ICIDataset(state='external')`로 로드 가능하나 **한 번도 평가하지 않았음.**

### ③ 구축한 것

| 스크립트 | 역할 |
|---|---|
| `scripts/power_analysis.py` | 실험 전 검출 가능 효과 크기 확인 |
| `scripts/launch_ici_protocol.sh` | 5 seed × 5 fold sweep (seed 내 fold 병렬, seed 간 순차), manifest 기록 |
| `scripts/evaluate_protocol.py` | per-seed / across-seed SD / pooled bootstrap CI / 외부 코호트를 구분해 보고 |
| `scripts/test.py` | 모든 AUROC에 bootstrap 95% CI **자동 부착** |
| `scripts/compare_predictions.py` | 두 run의 CI + paired bootstrap 승률 |

부수 정리: `--cv`를 `launch_interactive_training.sh`에 추가해 fold를 주입식으로 바꿨고, per-fold config 5개를 `train_v22_ici_finetune.yaml` 하나로 통합했습니다.

### ④ 반드시 구분할 것: seed를 늘려도 CI는 안 줄어듦

- **across-seed SD**: partition/학습 재현성. seed를 늘리면 평균의 표준오차가 줄어듦.
- **pooled bootstrap CI**: 코호트 표본 오차. **seed를 아무리 늘려도 줄어들지 않음** — 같은 87명을 재사용하기 때문. 사람을 더 모아야만 좁아집니다.

이 전제는 `tests/test_evaluation_protocol.py::test_smaller_cohort_gives_a_wider_interval`로 테스트에 고정해 두었습니다.

---

## 8. Source of Truth 파일

- Backbone & Version: `src/models/baseline.py` (`architecture_version = 22`)
- Data Interface & Collators: `src/modules/data_interface.py`
- Loss & Metrics: `src/modules/model_interface.py`
- 통계 비교 도구: `scripts/compare_predictions.py` (§6 참고)
- 검증 스위트: `tests/test_base_model.py`, `tests/test_model_interface.py`, `tests/test_batched_episode_forward.py`, `tests/test_evaluation_protocol.py`
- 평가 프로토콜: `scripts/power_analysis.py`, `scripts/launch_ici_protocol.sh`, `scripts/evaluate_protocol.py`, `scripts/evaluate_synthetic.py`
- **공용 평가 지표 구현**: `src/utils/metrics.py` (rank 기반 AUROC, cluster bootstrap) — 모든 평가 스크립트가 이 하나를 사용하므로 지표가 스크립트마다 어긋날 수 없음
