# Current architecture

**Last updated**: `2026-07-29 10:10:00 KST`  
**Code baseline**: Architecture Version `22` (`architecture_version = 22`)

이 문서는 현재 production config 및 코드베이스가 실제로 사용하는 Architecture v22 모델 구조와 수학적 계약을 설명합니다. v22는 v21에서 **retrieval 계층을 완전히 제거**한 버전이며, aggregator/meta-classifier 본체는 v21과 동일합니다. 최신 개발 상태는 [`current_status.md`](current_status.md), 실험 프로토콜은 [`current_experiments.md`](current_experiments.md)를 참고합니다.

---

## 1. 입출력 텐서 계약 (Input / Output Specification)

한 episode는 해당 에피소드의 **전체 context bags**와 query bags로 구성됩니다. v22에는 context를 K개로 줄이는 retrieval 계층이 없습니다. 각 instance (단일 세포)의 특징 차원은 $D = 512$입니다.

```text
input:
  outer episodes E        (합성 학습 시 E=8, 추론 시 E=1)
  context instances       [E, context_bags, instances, 512]
                          합성: context_bags = num_bags - queries (num_bags 60~100)
                          ICI : context_bags ~= 69 (fold train cohort 전체)
  context labels          [E, context_bags] (Y \in {0, 1})
  query instances         [E, query_bags, instances, 512]

output:
  query logits            [E, query_bags, num_classes=2]
```

Query label은 representation, normalization, class memory, ridge 또는 covariance subspace fitting 경로에 절대로 흐르지 않습니다.

---

## 2. 4대 수학적 핵심 기술 명세 (v19부터 유지)

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

## 3. Context 구성: retrieval 없음 (v22)

v22는 에피소드의 모든 context bag을 그대로 aggregator에 통과시킵니다. v21에 있던 `extract_bag_features` / `retrieve_context_indices` (40-token Signal-Aware Retrieval)와 외부 retrieval collator 3종은 모두 삭제되었습니다.

**제거 근거 요약** (상세: [`current_status.md`](current_status.md) §4):
- ICI 실데이터에서 retrieval을 켠 구성과 끈 구성의 AUROC가 동일했고(0.5481 vs 0.5454), calibration과 accuracy는 끈 쪽이 더 좋았음.
- ICI는 fold당 context가 ~69명뿐이라 K=24 선별은 가용 labeled context의 65%를 버리는 것.
- 무엇보다 n=87에서 이 차이들의 95% CI가 전부 겹치고 0.5를 포함해 **통계적으로 구분 불가능**했음.

aggregator가 각 bag을 40개 토큰(1 global + 12 slots × 3 통계 + 3 tail)으로 요약하는 구조 자체는 **분류기 입력으로 그대로 유지**됩니다 (`StructuredPopulationMetaClassifier._all_structured_tokens`). v21에서 이 40-token 요약을 retrieval 점수 계산에 재사용하던 경로만 사라졌습니다.

**Batched Multi-Episode Forward는 유지**: `BaseModel.forward_episode_batch`와 `forward`의 4D 분기가 `[episodes, bags, cells, dim]`을 한 optimizer step에 처리합니다. 검증: `tests/test_batched_episode_forward.py`.

**복구 지점**: retrieval 최종 구현은 git tag `v21-retrieval-final`에 보존되어 있습니다.

---

## 4. Main Classification Branches & Logit Fusion

실제 코드(`StructuredPopulationMetaClassifier`) 기준 2단 합성입니다.

**1단 — 증거 분기 융합** (`_fuse_evidence`, baseline.py:2518):

```text
logits = global_shape_logits
       + population_scale * population_logits
       + rare_scale     * rare_logits
       + fusion_scale   * interaction(global, population, rare)
```

`interaction`은 세 분기 logit의 곱·차 특징을 `fusion_scorer` MLP에 통과시킨 값입니다.

**2단 — 공분산 항 가산** (baseline.py:2868~2871):

```text
final_logits = logits
             + covariance_residual_scale * (covariance_ridge_scale * covariance_ridge_logits)
             + covariance_relation_residual_scale * covariance_relation_logits
```

> [!IMPORTANT]
> **위 scale 중 상당수는 고정 상수가 아니라 학습됩니다.** `population_scale` / `rare_scale` / `fusion_scale` / `covariance_residual_scale`은 학습 파라미터의 `sigmoid`이고, `covariance_ridge_scale`은 `exp` 후 `[0.1, 100]`으로 clamp됩니다. config의 값들은 이 학습 스케일의 **하한/상한 또는 초기 기준**을 정할 뿐입니다.

config에서 직접 지정하는 고정값:

| config 키 | 값 | 역할 |
|---|---:|---|
| `meta_population_residual_scale` | 0.25 | population 분기 scale 상한 |
| `meta_minimum_population_residual_scale` | 0.10 | 동 하한 |
| `meta_tail_residual_scale` | 0.10 | rare/tail 분기 scale 상한 |
| `meta_minimum_tail_residual_scale` | 0.05 | 동 하한 |
| `meta_fusion_residual_scale` | 0.10 | interaction 항 초기 scale |
| `meta_covariance_residual_scale` | 0.25 | covariance ridge 잔차 기준 |
| `covariance_relation.residual_scale` | 0.50 | CSP learned head 잔차 (고정 상수) |

> [!NOTE]
> 이전 문서는 이 식을 `base_logits + sparse_evidence_scale * sparse_memory_logits + ...`로 적고 `sparse_evidence_scale: 0.10`을 명시했으나, **`sparse_evidence_scale`이라는 파라미터는 코드베이스에 존재하지 않습니다** (2026-07-29 확인). 위 내용으로 교체했습니다.

---

## 5. 모델 스펙 파라미터 (Model Parameter Specs)

```text
Total Trainable Parameters : 6,566,811 (약 6.57M)
Token Dimension           : 512
Aggregator Output Tokens  : 40 tokens (1 global + 36 slot (12 slots x center/spread/rare) + 3 tail)
Context Retrieval         : none (v22 removed it; full context per episode)
Meta Hidden Dimension     : 256
Attention Heads           : 8
Set Layers                : 1
Ridge Dimension           : 64
Precision                 : bf16-mixed
Architecture Version      : 22
```

파라미터 수는 `configs/train_v22_medium.yaml`의 `model` 섹션 기준 실측값입니다 (2026-07-29). v21 태그(`v21-retrieval-final`)에서 동일 kwargs로 측정해도 **6,566,811로 같습니다** — retrieval은 학습 파라미터가 없는 순수 연산 계층이었으므로 제거해도 가중치 수는 변하지 않습니다. (이전 문서에 적혀 있던 `6,614,248`은 근거를 찾을 수 없는 오래된 값이라 실측값으로 교체했습니다.)

---

## 6. Source of Truth 파일

- Backbone & Version: `src/models/baseline.py` (`architecture_version = 22`)
- Data Interface & Collators: `src/modules/data_interface.py`
- Multi-task Loss & Metrics: `src/modules/model_interface.py`
- Resolved Production Entry: `configs/train_v22_medium.yaml`
- Architecture Verification Suites: `tests/test_base_model.py`, `tests/test_batched_episode_forward.py`
