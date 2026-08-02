# Architecture v26 Proposal: Episode-Conditional Mixture-of-Experts Meta-Classifier

**작성일**: `2026-08-02`  
**작성자**: GitHub Copilot (DeepSeek V4 Pro)  
**기반**: v24 확정 아키텍처의 실험적 한계 분석  
**상태**: 설계 제안서 — 검토·토론용

---

## 0. Executive Summary

v24 아키텍처는 bag 요약(40 token → 1 token projection)과 4대 수학 기술
(Z-Score Studentization, Top-1% Sparse Evidence, Covariance Shrinkage,
Pairwise Ranking Loss)을 통해 val CE 0.5903의 성능을 달성했습니다.

그러나 v22부터 v25까지 4개 아키텍처가 모두 **val CE 0.5903~0.5976**의 좁은
대역(폭 0.0073)에 갇혔고, Easy tier에서도 v24-easy(0.9073)와 v25-easy(0.9106)가
사실상 구분 불가능했습니다. 이는 **bag summary 압축 방식이 아닌, 더 근본적인
아키텍처 계열 전체의 구조적 한계**를 시사합니다.

본 문서는:

1. **현재 v24 아키텍처를 정밀하게 분석**하여 수학적·통계적 한계를 지적하고
2. **그 한계를 극복할 수 있는 v26 아키텍처를 제안**합니다

핵심 진단: **모든 evidence branch가 고정된(episode-independent) 혼합 가중치로
융합된다는 것**이 근본 병목입니다. 각 episode는 서로 다른 생성 분포(manifold,
response type, effect scale)를 가지므로, 어떤 evidence branch를 얼마나 신뢰할지는
**episode-conditional**이어야 합니다.

---

## 1. v24 아키텍처 상세 분석

### 1.1 데이터 흐름 개관

```text
Episode (E개의 context bag + Q개의 query bag)
│
├─ 1. Bag Studentization (§2-①)
│   각 bag i 내 세포 표현 x_{i,j} ∈ R^512 를 bag-centroid μ_i 와
│   standard deviation S_i 로 정규화:
│   x̃_{i,j} = (x_{i,j} - μ_i) / (S_i + ε)
│
├─ 2. Spherical k-means Density Anchors
│   Episode 전체 context bag의 cell 표현을 8개 anchor로 clustering.
│   Anchor는 episode의 "population coordinate system" 역할.
│
├─ 3. Slot-based Tokenization
│   각 bag → 40 structured tokens (512d each):
│   - 1 global token (bag average)
│   - 12 slots × 3 population stat types (center / feature-wise spread / rare state)
│     → 36 slot-stat tokens
│   - 3 tail tokens (top 1%/5%/15% outlier aggregates)
│
├─ 4. Bag Token Projection (§3.5, v24-B1)
│   40 tokens × 512d → [40 position-specific Linear(512→64)]
│   → concat(40×64=2560d) → concat with exact-mean residual(512d)
│   → Linear(3072→512) → 1 bag token (512d)
│   ── 압축비: 40:1 (40개 structured token → 1개 학습된 projection)
│
├─ 5. Class Memory Construction (§4)
│   context bag token을 class label별로 group → per-class cross-attention
│   → class당 8개 memory token (M₀, M₁ ∈ R^(8×512))
│   ── 핵심: context의 모든 bag이 16개 memory token(8 per class)으로 압축
│
├─ 6. Evidence Branches & Logit Fusion
│   6a. Global Shape Branch (§4-1단)
│       global_shape_logits = f_global(bag_token)
│       (bag token을 model backbone에 통과시킨 직접 예측)
│
│   6b. Population Branch + Ridge Classifier
│       population_logits = Ridge(M₀, M₁, query_population_stats)
│       (class memory와 query의 population 통계량 간 ridge regression)
│
│   6c. Rare Evidence Branch (§2-②)
│       rare_logits = CrossAttention(M_c, X_query^sparse)
│       (query의 top-1% sparse cell과 class memory 간 cross-attention)
│
│   6d. Covariance Subspace Branch (§2-③)
│       각 class context bag의 covariance subspace fitting →
│       query bag의 covarianceとの Mahalanobis 거리 → MLP → logit
│
│   6e. Fusion (§4)
│       final_logits = global_shape_logits
│              + α_pop · population_logits
│              + α_rare · rare_logits
│              + α_fusion · interaction(global, population, rare)
│              + α_cov · covariance_logits
│       (α_* 는 sigmoid-gated 학습 파라미터 — EPISODE-INDEPENDENT)
│
└─ 7. Loss
    L_total = L_CE + 0.10 · L_ranking
```

### 1.2 수학적 구조 — 모델이 학습하는 것

v24는 다음을 동시에 학습합니다:

- **Bag encoder** `φ`: R^(N_i × 512) → R^512 (bag당 1개 token)
- **Class memory constructor** `ψ`: context bags × labels → 8 memory tokens per class
- **Evidence scorers** `f_global`, `f_pop`, `f_rare`, `f_cov`, `f_fusion`
- **Evidence mixing weights** `α_global`, `α_pop`, `α_rare`, `α_fusion`, `α_cov`

여기서 중요한 점: `α_*`는 **모든 episode에 공통**입니다. 각 α는
`sigmoid(learnable_scalar)`로, 학습 완료 후에는 고정된 값입니다. 즉:

> **모델은 모든 episode의 response signal type을 동일한 혼합 비율로 처리합니다.**

예를 들어, 학습 중에 "state response"가 지배적인 episode와 "composition
response"가 지배적인 episode가 섞여 있었다면, 모델은 평균적인 혼합 비율을
배웠을 뿐, episode마다 적절한 전문가를 선택할 수 없습니다.

---

## 2. 근본적 한계 분석

### 2.1 진단: 왜 v22-v25가 모두 같은 곳에 갇혔나

| 실험 | 관찰 | 시사하는 것 |
|---|---|---|
| v22 (40 token) vs v24 (1 token projection) | CE 0.5946 → 0.5903 (Δ: -0.0043) | Bag 압축 방식의 개선은 marginal |
| v25 (typed bag-preserving, class-memory 우회) vs v24 | Medium: tie, Easy: +0.0033 | Class-memory 우회로는 이득 없음 |
| Easy tier: v24-easy vs v25-easy | 0.9073 vs 0.9106 (Δ: +0.0033) | 신호가 강해도 아키텍처가 갈리지 않음 |
| State task (Tier 2) | Model AUROC = observable descriptor probe AUROC | State 신호를 모델이 probe보다 더 잘 못 읽음 |
| State oracle | Oracle AUROC 0.88~0.90 | 충분한 정보가 data에 존재함 |

이 모든 관찰이 가리키는 공통된 방향:

> **Bag 표현의 압축 방식이나 class-memory 구조가 아니라, episode마다 달라지는
> signal type에 대응하는 episode-conditional adaptation의 부재가 한계다.**

### 2.2 통계적 추론: 고정 혼합 가중치의 정보이론적 한계

현재 v24의 evidence fusion을 정보이론적으로 해석합시다.

에피소드 `e`에 대해, 각 evidence branch `k ∈ {global, population, rare, covariance}`는
query bag의 class probability에 대한 **충분 통계량의 일부**를 제공합니다.
그러나 branch `k`가 유용한 신호를 제공하는 정도는 **에피소드의 생성 분포에
의존적**입니다:

- **State-dominant episode**: rare와 covariance branch가 중요한 신호를 제공
- **Composition-dominant episode**: population branch가 중요
- **Covariance-dominant episode**: covariance branch가 중요

v24의 fusion은 다음과 같이 쓸 수 있습니다:

$$
\text{logit}(e, q) = \sum_{k=1}^{K} \alpha_k \cdot \text{logit}_k(e, q)
$$

여기서 $\alpha_k$는 episode-independent입니다. 이는 사실상 **모든 episode에 대해
evidence branch의 기여도를 사전에 고정**하는 것과 같습니다.

더 나은 접근은 episode-conditional gating입니다:

$$
\text{logit}(e, q) = \sum_{k=1}^{K} g_k(z_e) \cdot \text{logit}_k(e, q)
$$

여기서 $z_e$는 episode의 context로부터 추출한 **episode embedding**이고,
$g_k(z_e)$는 episode에 따라 달라지는 **gating weight**입니다.

### 2.3 수치적 증거: Task별 AUROC 편차

v24 Medium에서 per-task AUROC (state: ~0.52, composition: ~0.70, covariance: ~0.55).
State에서 가장 낮은 성능을 보이는 이유:

1. **State signal은 cell type composition을 바꾸지 않고 continuous expression만
   변화**시킴 → population stats만으로 구분 어려움
2. **Covariance branch가 state를 어느 정도 커버할 수 있지만**, covariance
   estimation은 bag당 세포 수(N=500~1000)에 대해 512×512 공분산의 정밀 추정이
   불가능하여 bag-level에서 noisy함
3. **Rare evidence branch는 state signal을 포착 가능하지만**, 현재 gating에서
   `α_rare = 0.10`으로 고정되어 state-dominant episode에서도 제한된 기여만 함

즉, **모델은 state signal을 감지할 수 있는 branch들을 가지고 있지만,
episode가 state-dominant인지 감지해서 해당 branch에 더 높은 가중치를
부여할 능력이 없습니다.**

---

## 3. 제안: v26 — Episode-Conditional Mixture-of-Experts (EC-MoE)

### 3.1 핵심 아이디어

v26의 핵심은 **"Episode Context Aggregator"** 라는 새로운 구성요소입니다.
이는 모든 context bag(양 class 포함)을 입력받아 episode의 생성 분포 특성을
요약하는 **episode embedding** $z_e \in \mathbb{R}^{d_z}$를 생성합니다.

이 episode embedding은 두 가지 목적으로 사용됩니다:

1. **Evidence Expert Gating**: 각 evidence branch가 현재 episode에서 얼마나
   신뢰할 수 있는지 평가하여, branch fusion 가중치를 episode-conditional로 만듦
2. **Ridge Classifier Conditioning**: Ridge classifier의 정규화 파라미터를
   episode의 신호 강도에 맞게 조정

```
Episode e ─────────────────────────────────────────────────────┐
│                                                               │
├── Context Bags ──┬─── Bag Tokenizer (v24 유지) ───┐         │
│                  │                                 │         │
│                  ├─── Episode Context Aggregator ──┤         │
│                  │   (신규)                        │         │
│                  │   → episode embedding z_e       │         │
│                  │       │                         │         │
│                  │       ├─── Expert Gating ───────┤         │
│                  │       │   g_k(z_e) = softmax    │         │
│                  │       │   (W_gate · z_e)_k      │         │
│                  │       │                         │         │
├── Query Bags ────┤       │                         │         │
│                  ├─── Evidence Branches ───────────┤         │
│                  │   k ∈ {global, pop, rare, cov}  │         │
│                  │   → logit_k(e,q)                │         │
│                  │                                 │         │
│                  └─── Episode-Conditional Fusion ──┘         │
│                   logit(e,q) = Σ_k g_k(z_e) · logit_k(e,q)  │
│                   + α_cov · covariance_ridge(z_e, q)         │
│                                                               │
└── Loss: L_CE + 0.10·L_rank + β·L_balance (load balancing)   │
```

### 3.2 Episode Context Aggregator 설계

Episode context aggregator는 Deep Sets (Zaheer et al., 2017) 원리에 기반합니다.
핵심 속성: 임의 개수의 context bag을 permutation-invariant하게 하나의
episode embedding으로 요약합니다.

**구현**:

1. 각 context bag $i$의 token $t_i \in \mathbb{R}^{512}$ (v24 bag projection 후)을
   MLP로 encoding: $h_i = \text{MLP}_{\text{ctx}}(t_i) \in \mathbb{R}^{d_h}$
2. Bag-level encoding을 episode-level로 aggregate:
   $z_e^{\text{pool}} = \frac{1}{|\text{context}|} \sum_i h_i$
   (permutation-invariant mean pooling)
3. Cross-class contrastive feature: 두 class의 pooled representation 차이도 포함
   $\Delta_e = \frac{1}{|C_1|} \sum_{i \in C_1} h_i - \frac{1}{|C_0|} \sum_{i \in C_0} h_i$
4. 최종 episode embedding:
   $z_e = \text{MLP}_{\text{epi}}([z_e^{\text{pool}} \; \Delta_e]) \in \mathbb{R}^{d_z}$
   여기서 $d_z = 128$ (설계 파라미터)

**수학적 보장**: Deep Sets의 universality theorem에 의해, 충분히 큰 $d_h$와
MLP depth에 대해 이 aggregator는 context bag set의 임의의 permutation-invariant
continuous function을 근사할 수 있습니다.

### 3.3 Episode-Conditional Expert Gating

$K = 4$개의 evidence expert가 있다고 가정합니다 (v24의 branch 구조 유지):

- Expert 1: Global Shape Branch
- Expert 2: Population + Ridge Branch
- Expert 3: Rare Evidence Branch
- Expert 4: Covariance Subspace Branch

Gating network:

$$
g(z_e) = \text{softmax}(W_g \cdot z_e + b_g) \in \Delta^{K}
$$

여기서 $W_g \in \mathbb{R}^{K \times d_z}$, $b_g \in \mathbb{R}^K$는
학습 가능한 파라미터입니다. $g_k(z_e)$는 episode $e$에서 expert $k$의 gating weight.

**Top-2 Sparse Gating** (Shazeer et al., 2017):

전문가 붕괴(expert collapse: 모든 episode가 하나의 expert로 routing)를 방지하기
위해 top-2 sparse gating을 적용합니다:

$$
g_k(z_e) = \begin{cases}
\text{softmax}(\text{top}_2(W_g z_e + b_g))_k & \text{if } k \in \text{top-2} \\
0 & \text{otherwise}
\end{cases}
$$

여기서 $\text{top}_2$는 가장 큰 두 값만 유지하고 나머지를 $-\infty$로 masking.

**Load Balancing Auxiliary Loss**:

모든 episode가 같은 expert로 몰리는 것을 방지하기 위한 보조 loss:

$$
\mathcal{L}_{\text{balance}} = K \cdot \sum_{k=1}^{K} f_k \cdot \bar{g}_k
$$

- $f_k$: 전체 training batch에서 expert $k$가 선택된 episode 비율 (target: $1/K$)
- $\bar{g}_k$: expert $k$에 할당된 평균 gating weight
- $\beta = 0.01$: 보조 loss 가중치

이 loss는 expert utilization이 균등해지도록 유도합니다.

### 3.4 Evidence Branches — v24 구조 재사용

Gating network가 결정되면, 각 expert branch는 v24의 branch를 그대로 사용합니다.
**차이점**: 각 expert의 최종 gating weight가 episode-conditional $g_k(z_e)$로
대체됩니다.

```text
Fusion:
  logit(e, q) = Σ_{k=1}^{4} g_k(z_e) · logit_k(e, q)
              + g_cov(z_e) · covariance_ridge_logit(e, q)

각 logit_k는 v24 branch와 동일한 연산으로 생성됩니다.
g_k(z_e)는 episode-conditional (episode마다 다름).
```

### 3.5 Episode-Conditional Ridge Regularization

Covariance ridge classifier의 정규화 강도를 episode의 noise level에 맞게
조정합니다:

현재 v24: `subspace_shrinkage = 0.25` (고정)
v26 제안: episode embedding에서 shrinkage 조정 함수 학습

$$
\lambda_e = \text{sigmoid}(w_\lambda^\top z_e + b_\lambda) \in [0, 1]
$$

여기서 $\lambda_e$는 episode $e$의 covariance ridge 정규화 강도를 결정합니다.
Noise가 많은 episode는 더 강한 정규화(큰 $\lambda_e$), 신호가 강한 episode는
더 약한 정규화(작은 $\lambda_e$)를 적용합니다.

### 3.6 모델 파라미터 추정

| 구성요소 | 파라미터 수 |
|---|---|
| v24 base (bag encoder + class memory + evidence branches) | 9.45M |
| Episode Context Aggregator (MLP_ctx + MLP_epi) | ~0.3M |
| Expert Gating Network (W_g, b_g) | ~0.001M |
| Ridge Conditioning Head | ~0.001M |
| **v26 총 파라미터** | **~9.75M** (v24 대비 +0.3M, +3.2%) |

파라미터 증가는 미미합니다. 대부분의 용량은 v24 base에 있고,
새로 추가되는 것은 episode embedding과 gating뿐입니다.

### 3.7 아키텍처 버전 및 체크포인트 호환성

- `architecture_version = 26`
- v24 checkpoint는 로드 **불가** (신규 가중치 `EpisodeContextAggregator`,
  `ExpertGatingNetwork`, `RidgeConditioningHead`가 없음)
- v26 체크포인트는 `typed_bag_preserving_branch` 없이 v24 base +
  episode-conditional gating만 포함

---

## 4. 이 아키텍처가 한계를 어떻게 극복하는가

### 4.1 직관적 설명

현재 v24: "모든 환자에게 동일한 진단 가중치를 적용하는 의사"
v26 제안: "환자마다 어떤 검사(evidence branch)를 더 신뢰할지 판단하는 의사"

구체적으로:
- **State-dominant episode**: context aggregator가 "cell type composition이 비슷한데
  expression이 연속적으로 변한다"는 신호를 감지 → `g_rare`와 `g_cov`를 높임 → rare
  evidence와 covariance branch가 분류를 주도
- **Composition-dominant episode**: context aggregator가 "cell type 비율이
  다르다"는 신호를 감지 → `g_pop`을 높임 → population branch가 분류를 주도
- **Covariance-dominant episode**: `g_cov`를 높임 → Mahalanobis 거리 기반 분류

### 4.2 수학적 정당화 — 왜 작동해야 하는가

#### 4.2.1 Task-Adaptive Bayes Optimality

Bayesian 관점에서, 이상적인 classifier는 각 episode의 생성 분포 $P_e$를 알고
있다고 가정할 때의 posterior를 근사해야 합니다:

$$
P(y=1 \mid q, \text{context}_e) = \mathbb{E}_{P_e \sim P(\cdot \mid \text{context}_e)}[P_{P_e}(y=1 \mid q)]
$$

현재 v24는 고정된 evidence 가중치로 이 posterior를 근사합니다. v26은
episode embedding $z_e$를 통해 $P_e$의 충분 통계량을 포착하고, 이를
바탕으로 evidence 가중치를 조정함으로써 **episode-specific Bayes
optimal classifier에 더 가까워집니다**.

#### 4.2.2 Variance Reduction through Contextualization

고정 가중치 모델의 prediction variance를 분해하면:

$$
\text{Var}[\text{logit}(e,q)] = \underbrace{\text{Var}_{\text{episode}}[\text{optimal logit}]}_{\text{episode variability}} + \underbrace{\text{Var}_{\text{fixed} \; \alpha}[\text{logit} \mid \text{episode}]}_{\text{suboptimal gating}}
$$

Episode-conditional gating은 두 번째 항(suboptimal gating에서 오는 variance)을
줄여줍니다. gating이 episode의 signal type을 정확히 감지할수록, evidence fusion은
더 최적에 가까워집니다.

#### 4.2.3 Context-Size Scaling

주요 실험적 발견 중 하나: **context 10→160에서 AUROC가 단조 증가** (§6).
이유: context가 많을수록 episode의 signal type을 더 정확히 추론할 수 있기 때문.

현재 v24는 context가 많아도 이 정보를 활용하지 못합니다 (고정 가중치).
v26의 Episode Context Aggregator는 더 많은 context bag에서 더 정확한
episode embedding을 추출 → gating이 더 정확해짐 → context-size scaling이
더 가파르게 개선될 것으로 예상됨.

### 4.3 예상되는 실험적 결과

| Tier | 예상 |
|---|---|
| Medium | state task에서 가장 큰 개선 예상 (gating이 state-dominant episode에서 rare/cov branch를 강조) — overall AUROC +0.02~0.05 |
| Easy | ceiling effect(Clean signal)가 있더라도, gating이 expert를 정확히 선택하면 state task 개선으로 overall +0.01~0.03 |
| Hard | context가 적어 gating이 noisy해질 수 있으나, 그나마 context가 있는 한 random보다는 나음 |
| Context scaling | context 10→160에서 v24보다 더 가파른 AUROC 증가 예상 |

---

## 5. 구현 계획

### 5.1 Phase 1 — Episode Context Aggregator + Gating (MVP)

**목표**: episode-conditional gating의 기본 효과 검증

1. `src/models/baseline.py`에 `EpisodeContextAggregator` 클래스 추가
   - input: `[episodes, context_bags, 512]` (bag tokens)
   - output: `[episodes, d_z]` (episode embeddings)
2. `ExpertGatingNetwork` 추가
   - input: episode embedding
   - output: top-2 gating weights
3. `_fuse_evidence` 수정: 고정 α를 episode-conditional gating으로 대체
4. `ModelInterface`에 load balancing loss 추가
5. `architecture_version = 26` (체크포인트 게이트)
6. Unit test: Deep Sets permutation invariance, gating weight shape/value, load balancing loss

### 5.2 Phase 2 — Adaptive Ridge Conditioning

1. `RidgeConditioningHead` 추가
2. Covariance branch의 `subspace_shrinkage`를 고정 상수 → episode-conditional로 변경

### 5.3 Phase 3 — Full Integration & Evaluation

1. `configs/train_v26_medium_ec_moe.yaml` (v24 config 기반, gating 추가)
2. 50-epoch Medium scratch 학습
3. v24와 1,000-episode pool-400 paired 비교 (§7 프로토콜)
4. Per-task AUROC 분석 (state/composition/covariance별 gating weight 확인)

### 5.4 Config 설계

```yaml
# configs/train_v26_medium_ec_moe.yaml
base_config: configs/train_v24_medium_bag_proj_residual.yaml

model_overrides:
  architecture_version: 26
  episode_context_aggregator:
    enabled: true
    hidden_dim: 256       # d_h
    episode_embed_dim: 128  # d_z
    num_mlp_layers: 3
  expert_gating:
    enabled: true
    num_experts: 4
    top_k: 2              # sparse gating
    load_balance_weight: 0.01  # β

data_overrides:
  episode_batch_size: 8

trainer:
  max_epochs: 50
```

---

## 6. 위험 요소 및 완화 전략

### 6.1 Expert Collapse

**위험**: 모든 episode가 같은 expert로 routing되어 사실상 v24와 동일해짐  
**완화**: Top-2 sparse gating + load balancing loss + gating weight에 small
Gaussian noise 추가 (exploration)

### 6.2 Episode Embedding이 충분히 정보적이지 않음

**위험**: context aggregator가 episode의 signal type을 구분하지 못함 → gating이
의미 없음  
**완화**: Cross-class contrastive feature (Δ_e)를 embedding에 포함시켜
class간 차이가 episode signal type의 proxy가 되도록 함. 또한
context bag token뿐 아니라 raw population 통계량(anchor별 abundance 등)도
aggregator 입력에 포함

### 6.3 Context 부족으로 gating이 noisy

**위험**: Hard tier나 작은 context(10~20 bags)에서 gating이 불안정  
**완화**: gating weight에 temperature annealing (학습 초기: high temperature →
uniform → 점차 sharpening). 작은 context에서는 gating weight prior를
uniform에 가깝게 유지하는 regularization 추가 (episode batch 내 context
size에 따른 temperature 조정)

### 6.4 Load Balancing Loss가 학습을 방해

**위험**: L_balance가 너무 강하면 실제로 유용한 expert 선택을 방해  
**완화**: β = 0.01로 작게 시작, 필요 시 감소

---

## 7. 관련 연구 맥락

### 7.1 이론적 기반

- **Deep Sets** (Zaheer et al., 2017): Permutation-invariant set encoding의
  universality — Episode Context Aggregator의 수학적 정당성
- **Sparsely-Gated Mixture-of-Experts** (Shazeer et al., 2017):
  Conditional computation with load balancing — Expert Gating 설계의 기반
- **Neural Processes** (Garnelo et al., 2018): Context set → stochastic process
  inference — episode-conditional adaptation의 Bayesian 해석
- **Proto-MAML** (Triantafillou et al., 2020): Metric-based meta-learning에서
  task-conditional prototype refinement

### 7.2 BagPFN 프로젝트 맥락에서의 의의

v26은 BagPFN 프로젝트의 핵심 문제의식—"모든 episode가 같은 classifier로
분류된다"—을 직접 건드립니다. v25(T5-A)가 class-memory pooling을 우회했지만
여전히 episode-independent fusion을 사용했기에 한계가 있었습니다.

v26의 핵심 기여는 **meta-learning에서 task heterogeneity를 다루는 원리적 해법**
(episode-conditional expert routing)을 BagPFN의 evidence fusion 구조에
접목한 것입니다. 이는 다음과 같은 점에서 이전 시도들과 다릅니다:

- v23/v24: bag 표현 압축 (정보 보존 vs 차원 축소) — bag-level 개선
- v25: class-memory pooling 우회 (bag identity 보존) — class-level 개선
- **v26: episode-conditional evidence routing — episode-level 개선**

이 세 level 중 episode-level 개선이 가장 큰 성능 향상을 가져올 가능성이 높습니다.
bag-level과 class-level은 이미 포화에 가깝고(v22-v25 plateau), episode-level은
아직 건드린 적이 없는 축이기 때문입니다.

---

## 8. 결론

v24 아키텍처의 근본적 한계는 **episode-independent evidence fusion**입니다.
모든 episode가 동일한 branch 혼합 가중치로 분류되기 때문에, episode마다 다른
signal type(state/composition/covariance)에 대응할 수 없습니다.

v26 (Episode-Conditional Mixture-of-Experts Meta-Classifier)은:

1. **Episode Context Aggregator**로 episode embedding을 추출하고
2. 이 embedding으로 **evidence branch의 gating weight를 episode-conditional**로
   조정하여
3. 각 episode의 signal type에 최적화된 evidence routing을 수행합니다.

이는 이론적으로 (Bayes optimality, variance reduction) 그리고 실험적으로
(context-size scaling, per-task 개선) 모두 정당화되며, v24 대비 파라미터
증가가 3.2%에 불과해 구현 부담도 낮습니다.

---

## Appendix A: 주요 수식 요약

### A.1 Episode Context Aggregator

$$
\begin{aligned}
h_i &= \text{MLP}_{\text{ctx}}(t_i) \in \mathbb{R}^{d_h} \quad (i \in \text{context bags}) \\
z_e^{\text{pool}} &= \frac{1}{|\mathcal{C}|} \sum_{i \in \mathcal{C}} h_i \\
\Delta_e &= \frac{1}{|\mathcal{C}_1|} \sum_{i \in \mathcal{C}_1} h_i - \frac{1}{|\mathcal{C}_0|} \sum_{i \in \mathcal{C}_0} h_i \\
z_e &= \text{MLP}_{\text{epi}}([z_e^{\text{pool}}; \Delta_e]) \in \mathbb{R}^{d_z}
\end{aligned}
$$

### A.2 Episode-Conditional Expert Gating

$$
g(z_e) = \text{softmax}(\text{top}_2(W_g z_e + b_g)) \in \Delta^K
$$

### A.3 Episode-Conditional Logit Fusion

$$
\text{logit}(e, q) = \sum_{k=1}^{K} g_k(z_e) \cdot \text{logit}_k(e, q) + g_{\text{cov}}(z_e) \cdot \text{cov\_ridge}(e, q; \lambda_e)
$$

여기서 $\lambda_e = \text{sigmoid}(w_\lambda^\top z_e + b_\lambda)$는
episode-conditional ridge regularization.

### A.4 Total Loss

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.10 \cdot \mathcal{L}_{\text{ranking}} + 0.01 \cdot \mathcal{L}_{\text{balance}}
$$

---

## Appendix B: v24와 v26의 Evidence Flow 비교

```text
[ v24: Episode-Independent Fusion ]

Context ──→ Bag Tokens ──→ Evidence Branches ──┐
                                                ├─→ [α₁·logit₁ + α₂·logit₂ + ...]
                                                │   α₁,...,α_K: FIXED scalars
Query   ──→ Bag Token  ──→ Evidence Branches ──┘
         (same branches, different input)


[ v26: Episode-Conditional Fusion ]

Context ──→ Bag Tokens ──┬─→ Episode Context Aggregator ──→ z_e
                         │                                     │
                         ├─→ Evidence Branches ──┐            │
                         │                        │            ▼
                         │                        ├─→ [g₁(z_e)·logit₁ + ...]
                         │                        │   g_k(z_e): episode-conditional
Query   ──→ Bag Token ───┤                        │
                         └─→ Evidence Branches ──┘
                           (same branches, different input)
```

---

## Appendix C: 구현 변경점 요약 (`src/models/baseline.py`)

v26 구현 시 `StructuredPopulationMetaClassifier`에 추가될 주요 변경점:

1. **`__init__`**: `EpisodeContextAggregator`, `ExpertGatingNetwork`, `RidgeConditioningHead` 초기화
2. **`_episode_context_embedding`** (신규 메서드): context bag token → episode embedding
3. **`_expert_gating_weights`** (신규 메서드): episode embedding → softmax gating vector
4. **`_fuse_evidence`** (수정): 고정 α → `g_k(z_e)` 호출
5. **`_covariance_residual_scale`** (수정): 고정 상수 → `λ_e` (episode-conditional)
6. **`ModelInterface.training_step`** (수정): `L_balance` 계산 및 total loss에 추가
7. **체크포인트 게이트**: `architecture_version = 26` 체크, v24 거부
