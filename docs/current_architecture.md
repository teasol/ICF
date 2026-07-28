# Current architecture

**Last updated**: `2026-07-28 13:10:00 KST`  
**Code baseline**: Architecture Version `21` (`architecture_version = 21`)

이 문서는 현재 production config 및 코드베이스가 실제로 사용하는 Architecture v21 모델 구조, 수학적 계약 및 40차원 Signal-Aware Retrieval 레이어를 명시적으로 설명합니다. 최신 개발 상태는 [`current_status.md`](current_status.md), 실험 프로토콜은 [`current_experiments.md`](current_experiments.md)를 참고합니다.

---

## 1. 입출력 텐서 계약 (Input / Output Specification)

한 episode는 Class-Balanced 24-Donor Retrieval (또는 Model-Level 40-dim Signal-Aware Retrieval)로 선별된 context bags ($K=24$, Top-12 NR + Top-12 R)와 query bags로 구성됩니다. 각 instance (단일 세포)의 특징 차원은 $D = 512$입니다.

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

## 3. 모델 내장 40차원 Feature 기반 Signal-Aware Retrieval Layer

### ① 40차원 Bag Feature Extractor ($Z \in \mathbb{R}^{40}$)
BagPFN 아키텍처 내부 Aggregator는 Bag 표현을 다음 4가지 축으로 분해하여 40차원 특징 벡터 $Z$로 압축합니다:
1. **12-dim Density Slots**: 세포 밀도 및 sub-population 분포
2. **Tail Evidence Features**: Top-1% / 5% / 15% 희귀 세포 반응 신호
3. **Subspace Covariance Sketch**: 세포 간 상관관계 및 covariance 구조
4. **Centered-Spread Scale**: 세포 확산 척도

### ② Model-Level Retrieval Method
- `extract_bag_features(x)`: 각 Donor Bag의 40차원 $Z$ 특징 벡터 추출.
- `retrieve_context_indices(x, y, mask_index, retrieval_k)`: Query $Z_Q$와 Context donor $Z_{C_i}$ 간의 Cosine Similarity를 계산하여, 클래스 균형(Class-Balanced) Top-12 NR + Top-12 R ($K=24$) donor 동적 추출.
- `forward(..., retrieval_k=24)`: 모델 순전파 내에 Signal-Aware Retrieval을 직접 내장.

> [!WARNING]
> **⚠ 구현 갭 (2026-07-28 세션 확인)**: 현재 `src/models/baseline.py`의 `extract_bag_features` 실제 구현은 위 40-dim 설계(density slots / tail evidence / covariance sketch / scale)를 따르지 않고, `global_summary`(512-dim)와 `tails` 평균(512-dim)을 concat한 **1024-dim** 벡터를 반환함. 설계된 진짜 40-dim descriptor는 미구현 상태이며, 자세한 내용과 조치 필요 사항은 [`current_status.md`](current_status.md) §4-③ 참고.

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
Aggregator Output Tokens  : 40 tokens (1 global + 12 slots + 8 density slots + 12 tail slots + 7 branch tokens)
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
