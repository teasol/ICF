# Current architecture

**Last updated**: `2026-07-28 16:05:00 KST`  
**Code baseline**: Architecture Version `21` (`architecture_version = 21`)

이 문서는 현재 production config 및 코드베이스가 실제로 사용하는 Architecture v21 모델 구조, 수학적 계약 및 40-token Signal-Aware Retrieval 레이어를 명시적으로 설명합니다. 최신 개발 상태는 [`current_status.md`](current_status.md), 실험 프로토콜은 [`current_experiments.md`](current_experiments.md)를 참고합니다.

---

## 1. 입출력 텐서 계약 (Input / Output Specification)

한 episode는 Class-Balanced 24-Donor Retrieval (또는 Model-Level 40-token Signal-Aware Retrieval)로 선별된 context bags ($K=24$, Top-12 NR + Top-12 R)와 query bags로 구성됩니다. 각 instance (단일 세포)의 특징 차원은 $D = 512$입니다.

```text
input:
  outer episodes E        (훈련 시 E=8 또는 E=16, 추론 시 E=1)
  context instances       [E, context_bags=24, instances=1000, 512]
  context labels          [E, context_bags=24] (Y \in {0, 1})
  query instances         [E, query_bags=1, instances=1000, 512]

output:
  query logits            [E, query_bags=1, num_classes=2]
```

Query label은 representation, normalization, class memory, ridge 또는 covariance subspace fitting 경로에 절대로 흐르지 않습니다.

---

## 2. Architecture v21 4대 수학적 개혁 명세

### ① Z-Score Bag Studentization (분산 스케일 정규화)
각 Donor Bag $i$의 세포 표현 $x_{i, j} \in \mathbb{R}^{512}$에 대해 Donor Centroid $\mu_i$와 Standard Deviation $S_i$를 구하여 스튜던트화 정규화를 적용함:

$$\mu_i = \frac{1}{N_i} \sum_{j=1}^{N_i} x_{i, j}, \quad S_i = \sqrt{\frac{1}{N_i} \sum_{j=1}^{N_i} (x_{i, j} - \mu_i)^2 + 1e-6}$$

$$\tilde{x}_{i, j} = \frac{x_{i, j} - \mu_i}{S_i + 1e-5}$$

### ② Top-k Sparse Evidence Tokenization (Sub-1% 희귀세포 핀포인트 추출)
97%+ 비반응 배경세포에 의해 0.5%~3% 희귀 세포 신호가 희석되는 현상을 방지함:
- Bag 중심 $\mu_i$로부터 이상 거리(Outlier Distance) $d_{i, j} = \|\tilde{x}_{i, j}\|_2$ 산출.
- 상위 1% ($k = \max(1, \lceil 0.01 \cdot N_i \rceil)$) 세포 인스턴스 $X_i^{sparse} \in \mathbb{R}^{k \times 512}$ 선별.
- Class Memories $M_c \in \mathbb{R}^{8 \times 512}$ ($c \in \{0, 1\}$)와 Direct Cross-Attention 수행 후 residual logit 결합.

### ③ Covariance Subspace Shrinkage (노이즈 방어)
- SNR-Adaptive Covariance Subspace Fitting의 Shrinkage 파라미터를 **`subspace_shrinkage: 0.25`**로 정밀화.
- Whitening 연산 시 NaN 발생 시 안전하게 Identity Fallback 구동.
- 백색화 변환 거리 vector $[d_0, d_1, d_0 - d_1, \text{sep}]$를 2-layer MLP Learned Head로 연결.

### ④ Auxiliary Pairwise Ranking Loss (`weight: 0.10`)
- Cross-Entropy 0.685 부근 Gradient 소멸 극복:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + 0.10 \cdot \mathcal{L}_{Ranking}$$

$$\mathcal{L}_{Ranking} = \max(0, \gamma - (P_1(Q^+) - P_1(Q^-)))$$

---

## 3. 모델 내장 40-token Aggregator Summary 재사용 Signal-Aware Retrieval Layer

### ① 40-token Bag Summary ($[bags, 40, 512]$)
BagPFN Aggregator(`StructuredEpisodePopulationAggregator`)는 각 Bag을 분류기 입력용으로 이미 **40개의 512-dim 토큰**으로 요약합니다 (`StructuredPopulationMetaClassifier._all_structured_tokens`):
1. **1 global token**: bag 전체의 centered-spread scale 요약 (`global_summary`)
2. **36 slot tokens**: population slot 12개 × center/spread/rare 3종 통계 (`slot_statistic_count = 3`, `representation["slots"]`)
3. **3 tail tokens**: Top-1% / 5% / 15% 희귀 세포 반응 신호 (`representation["tails"]`)

Retrieval은 이 **이미 계산되어 있는 요약을 그대로 재사용**하며, 별도의 손으로 압축한 feature vector를 새로 만들지 않습니다.

### ② Model-Level Retrieval Method
- `extract_bag_features(x)`: `self.aggregator(x, ...)`로 얻은 `representation`을 `self.meta_classifier._all_structured_tokens(representation)`에 그대로 통과시켜 `[bags, 40, 512]`를 반환.
  - **Anchor 안정성**: Population anchor(슬롯 중심)는 그 호출에 함께 들어온 bag 집합(`context_mask`)에 의존하므로, `chunk_size`로 나눠 여러 번 호출하면 anchor가 청크마다 달라져 같은 bag의 descriptor가 흔들리는 문제가 있었음. `extract_bag_features`는 이제 anchor를 항상 전체 `x`에서 **한 번만** 계산(`self.aggregator._context_anchors`)하고, `chunk_size`는 오직 `_forward_dense` 호출을 나누는 메모리 최적화로만 사용함 — chunked/dense 결과가 최대 절대오차 4.5e-8로 사실상 동일함을 확인.
- `retrieve_context_indices(x, y, mask_index, retrieval_k)`: `extract_bag_features`의 `[bags,40,512]`를 `[bags, 40×512]`로 flatten한 뒤 Query $Z_Q$와 Context donor $Z_{C_i}$ 간 Cosine Similarity를 계산하여, 클래스 균형(Class-Balanced) Top-12 NR + Top-12 R ($K=24$) donor 동적 추출.
- `forward(..., retrieval_k=24)`: 단일-episode(3D 또는 4D) 순전파 내에 Signal-Aware Retrieval을 직접 내장.
- **⚠ `forward_episode_batch`는 retrieval을 모름**: 대형 candidate pool 사전학습(Phase 5)에 쓰이는 `BaseModel.forward_episode_batch`는 `retrieval_k` 파라미터가 없고 입력 bag 전체를 그대로 dense aggregator forward에 통과시킴. 따라서 이 경로를 쓰는 학습 루프(`ModelInterface.training_step`의 4D 분기)는 **`forward_episode_batch` 호출 전에 반드시 `self.model.retrieve_context_indices(...)`를 먼저 호출**해 candidate pool을 `retrieval_k + query_count`로 줄여야 함 — 그렇지 않으면 retrieval 없이 전체 pool이 그대로 forward되어 대형 episode에서 OOM 발생. `retrieval_k`/`retrieval_chunk_size`는 `ModelInterface` hparams(`model_kwargs`)로 전달되고 `_build_model`에서 pop되어 `BaseModel` 생성자에는 전달되지 않음.
- **수정 이력**: 최초 구현은 `extract_bag_features`가 실제로 1024-dim(`global_summary`+`tails` 평균 concat)을 반환하던 갭이 있었고, 이후 두 차례(density/tail/covariance/scale 압축 → 슬롯 3-stat 압축)의 hand-crafted feature 시도는 모두 "aggregator가 이미 만든 요약을 재사용"하는 것이 아니라 별도의 새 표현을 만드는 방향이라는 지적에 따라 폐기됨. `training_step`이 `retrieve_context_indices`를 호출하지 않아 이 전체 설계가 실제 학습에서는 죽어있던 문제도 함께 발견/수정됨. 자세한 이력은 [`current_status.md`](current_status.md) §4-②, §4-④ 참고.

---

## 4. Main Classification Branches & Logit Fusion

```text
final_logits = base_logits
             + sparse_evidence_scale * sparse_memory_logits
             + covariance_residual_scale * covariance_ridge
             + residual_scale * covariance_relation_learned_head
```

* `sparse_evidence_scale`: 0.10
* `covariance_residual_scale`: 0.25
* `residual_scale` (CSP Head): 0.50

---

## 5. 모델 스펙 파라미터 (Model Parameter Specs)

```text
Total Trainable Parameters : 6,614,248 (약 6.6M)
Token Dimension           : 512
Aggregator Output Tokens  : 40 tokens (1 global + 36 slot (12 slots x center/spread/rare) + 3 tail)
Meta Hidden Dimension     : 256
Attention Heads           : 8
Set Layers                : 1
Ridge Dimension           : 64
Precision                 : bf16-mixed
Architecture Version      : 21
```

---

## 6. Source of Truth 파일

- Backbone & Version: `src/models/baseline.py` (`architecture_version = 21`)
- Data Interface & Collators: `src/modules/data_interface.py`
- Multi-task Loss & Metrics: `src/modules/model_interface.py`
- Resolved Production Entry: `configs/train_v21_medium.yaml`
- Architecture Verification Suites: `tests/test_base_model.py`, `tests/test_feature_retrieval.py`
