# Architecture v27 Proposal: Adaptive Covariance-Aware In-Context Attention & Continuous Soft Routing (AC-ICAR)

> [!IMPORTANT]
> **미구현 폐기 (archived 2026-08-02).** `docs/architecture_v28_proposal.md` §3에서
> 비판 완료: 핵심 전제(40→1 압축이 정보를 파괴)는 v22(40 token, CE 0.5946) vs
> v24(1 token, CE 0.5903)로 이미 반증됐고, 제안된 Riemannian branch(§3.4)는
> 실측 결과 step time 7~15배, 인접 고유값 간격 `2.97e-07`로 `bf16-mixed`에서
> backward 불안정 위험이 있어 실행 어려움. Routing 부분은 §6.1의 E2(oracle
> gating 상한 delta 0.0000)로도 별도 반증됨. 원문은 그대로 보존.

**작성일**: `2026-08-02`  
**작성자**: Antigravity AI (Pair Programming with User)  
**기반**: v24 확정 아키텍처 및 v26 Proposal 비판적 진단  
**상태**: 설계 제안서 (Architecture Proposal)

---

## 0. Executive Summary

v24 아키텍처는 residual + bottleneck bag projection(40 token → 1 token)과 4대 수학 기술(Studentization, Top-1% Sparse Evidence, Covariance Shrinkage, Pairwise Ranking Loss)을 바탕으로 val CE 0.5903을 달성하며 v24로 최종 확정되었습니다.

그러나 v22부터 v25까지 모든 아키텍처 변형이 **val CE 0.5903~0.5976** 대역에 정체되었으며, 최근 제안된 v26 Proposal (DeepSeek V4 Pro 작성, Deep Sets 기반 Episode Embedding + Top-2 MoE Gating) 역시 다음 3가지 근본적 한계를 내포하고 있습니다:
1. **노이즈 취약성**: Donor shift 및 background noise가 상존하는 40~80개 bag level에서 단순 mean pooling 기반 episode embedding ($z_e$)을 추출하므로 Gating Network가 노이즈에 심각하게 오염됨.
2. **Hard Top-2 Routing 부작용**: Meta-learning (small episode batch size) 환경에서 Discontinuous Top-2 routing은 gradient variance를 극도로 높여 Expert Collapse 또는 Uniform Gating 묶임 현상을 초래함.
3. **Bag Representation 병목의 방치**: Bag당 1개 token으로 강제 압축하는 v24-B1 projection 구조를 그대로 유지함으로써, 하위 branch가 다룰 수 있는 표현 역량(Representation Upper Bound)이 사전에 제한됨.

본 문서에서는 이 모든 한계를 극복하는 **Architecture v27: AC-ICAR (Adaptive Covariance-Aware In-Context Attention & Continuous Soft Routing)**를 제안합니다.

### 핵심 혁신 3가지:
1. **Multi-Resolution Dual-Path Bag Tokenizer**: Bag당 1개 token 병목 압축을 폐기하고, 16개의 고해상도 Structured Token System (Density Anchors 8 + Outliers 4 + Covariance Basis 4)으로 다중 해상도 표현 보존.
2. **Continuous Cross-Attentive Soft Routing**: Episode Context Set 전체 표현과 Evidence Branch 간 Cross-Attention 기반의 Smooth Temperature Softmax Routing 적용.
3. **Riemannian Subspace Alignment Branch**: Context와 Query Bag Covariance간 Riemannian Log-Euclidean Metric Tensor를 직접 계산하여 State 및 Covariance 신호를 오라클에 가깝게 포착.

---

## 1. v24 및 v26 Proposal 비판적 진단 (Critical Review)

### 1.1 v24 실증 데이터 재해석 (T3-1 & T2-2 진단 결과)

v26 Proposal은 State task AUROC(~0.52)가 낮은 이유를 "고정 가중치 $\alpha_*$가 state-dominant episode를 감지하지 못해서"로 진단했습니다. 하지만 이는 이전 실증 진단 데이터와 명백히 모순됩니다:

| 진단 항목 | 실증 데이터 | 시사점 |
|---|---|---|
| **Effect Scale 통제 (T3-1)** | Effect scale 0.7 통일 시 **Covariance(0.6594) = Composition(0.6488)** | Covariance/State 열위는 고정 가중치가 아닌 생성기 scale 차이 아티팩트 |
| **State Upper Bound (T2-2)** | Raw observable mean probe(0.5478) vs Oracle(0.8819) | 연속 표현 이동(State)은 Bag 요약 단계에서 이미 대규모 정보 손실 발생 |
| **Context Scaling (T4)** | Context 10→160 bag 증가 시 AUROC 0.5084→0.5737 상승 | 모델은 context 수용량이 늘어날수록 in-context adaptation을 잘 수행함 |

즉, **Gating의 부재가 병목이 아니라, Bag 요약 시 연속 변량(State/Covariance) 신호가 찌그러지는 것**이 진짜 원인입니다.

### 1.2 v26 Proposal (EC-MoE)의 3대 구조적 허점

```text
[ v26 Proposal (EC-MoE) 의 문제 구조 ]

Context Bags ──→ Bag Projection (40→1 token) ──→ Deep Sets Mean Pooling ──→ z_e (128d)
                     │ (이미 정보 파괴됨)            │ (Donor Shift 노이즈 오염)
                     ▼                               ▼
                 Evidence Branches ───────────→ Top-2 Hard MoE Routing (Discontinuous Gradient)
```

1. **Information Bottleneck & Noise Propagation**:
   - Bag당 40개 token을 1개 token으로 압축한 뒤, 이를 다시 N개 bag에 대해 단순 평균을 내어 $z_e \in \mathbb{R}^{128}$를 만듭니다.
   - Donor shift(0.35~0.70)나 Rare fraction(0.5~3%)이 작동하는 Hard regime에서 $z_e$는 response signal type이 아닌 **Donor ID와 Nuisance Noise**만을 인코딩하게 됩니다.
2. **Top-2 Hard Routing & Training Instability**:
   - PFN meta-training은 episode batch size가 2~8로 매우 작고 10,000~20,000 step 내외로 수렴합니다.
   - 이러한 조건에서 Top-2 Hard Routing은 Router Gradient Variance를 극대화시켜 특정 Expert로 몰리는 Gating Collapse를 유발하거나, Load Balance Loss ($\mathcal{L}_{\text{balance}}$)에 의해 가중치가 Uniform(0.25)하게 묶여 고정 가중치와 다름없어집니다.
3. **Bag Representation Capacity 무시**:
   - Bag 요약 40:1 압축 병목을 그대로 두어, 아무리 상위에서 Gating을 잘 해줘도 각 branch가 전달받는 정보 자체가 왜곡되어 있습니다.

---

## 2. Architecture v27: AC-ICAR 상세 설계

```text
[ Architecture v27 (AC-ICAR) Data Flow ]

Episode (Context Bags {C_i} + Query Bags {Q_j})
│
├─ 1. Multi-Resolution Dual-Path Bag Tokenizer (§3.1)
│   각 bag → 16 Structured Tokens (512d each):
│   - 8 Global Density Anchors (Population coordinate)
│   - 4 Rare Outlier Aggregates (Top 1%/5%/15% + Tail)
│   - 4 Covariance Subspace Dynamic Basis Tokens (Log-Euclidean Key Vectors)
│
├─ 2. Riemannian Subspace Alignment Branch (§3.2)
│   Context Bag Covariance S_c 와 Query Bag Covariance S_q 간
│   Log-Euclidean Metric Tensor d_riemann(S_c, S_q) 계산 → Direct Metric Logit
│
├─ 3. Continuous Cross-Attentive Soft Gating Transformer (§3.3)
│   - Context Token Matrix E_context ∈ R^(N_ctx × 16 × 512) 를 Key/Value로 활용
│   - Learnable Branch Query Matrix Q_branch ∈ R^(K × 512)
│   - Cross-Attention Score → Softmax Temperature (τ) → Smooth Gating Vector g(E_context)
│
├─ 4. Multi-Branch Logit Fusion
│   logit(e, q) = Σ_{k=1}^{K} g_k(E_context) · logit_k(e, q) + g_riemann(E_context) · logit_riemann(e, q)
│
└─ 5. Smooth Loss Target
    L_total = L_CE + 0.10 · L_ranking + 0.05 · L_gating_entropy
```

### 2.1 Multi-Resolution Dual-Path Bag Tokenizer

v24의 1-token projection 대신, 각 Bag을 **16개의 고해상도 Token**으로 변환합니다:

1. **Global Density Anchors ($T_{\text{pop}} \in \mathbb{R}^{8 \times 512}$)**:
   - Episode 전체 세포에 대한 Spherical k-means centroid 8개를 anchor로 사용.
   - Bag $i$의 세포들을 8개 anchor 영역별로 soft-assignment하여 centroid 표현 구성.
2. **Rare Outlier Aggregates ($T_{\text{rare}} \in \mathbb{R}^{4 \times 512}$)**:
   - Density score 기준 하위 Top 1%, 5%, 15% 세포 및 tail residual의 weighted mean.
3. **Covariance Subspace Dynamic Basis Tokens ($T_{\text{cov}} \in \mathbb{R}^{4 \times 512}$)**:
   - Bag $i$의 Covariance Matrix $\Sigma_i$의 Top-4 Eigenvectors $v_1, v_2, v_3, v_4$와 Eigenvalues $\lambda_k$의 결합 표현:
     $$t_{\text{cov}, k} = \sqrt{\lambda_k} \cdot v_k \in \mathbb{R}^{512}$$

### 2.2 Riemannian Subspace Alignment Branch

Covariance/State 반응 신호는 Euclidean 공간이 아닌 Symmetric Positive Definite (SPD) Manifold 위에 존재합니다. v27은 Context와 Query 간 Riemannian Metric을 직접 계산합니다:

1. Bag Covariance $S_i = \Sigma_i + \epsilon I$ (Shrunk Covariance).
2. Matrix Logarithm 변환: $\tilde{S}_i = \log(S_i) = V \log(\Lambda) V^\top$.
3. Class $c$ Context Bags의 Average Riemannian Matrix:
   $$\bar{S}_c = \exp\left( \frac{1}{|C_c|} \sum_{i \in C_c} \log(S_i) \right)$$
4. Query Bag $q$와의 Riemannian Distance:
   $$d_{\text{riemann}}(q, c) = \|\log(S_q) - \log(\bar{S}_c)\|_F$$
5. Logit 반환:
   $$\text{logit}_{\text{riemann}}(q) = \text{MLP}(d_{\text{riemann}}(q, 0) - d_{\text{riemann}}(q, 1))$$

### 2.3 Continuous Cross-Attentive Soft Gating Transformer

Hard Top-2 Routing 대신, Context Set 전체를 Cross-Attention으로 쿼리하여 지능적이고 연속적인 Soft Routing 가중치를 생성합니다:

$$E_{\text{context}} \in \mathbb{R}^{(N_{\text{ctx}} \cdot 16) \times 512}$$

1. Branch별 Query Vector $Q_{\text{branch}} \in \mathbb{R}^{K \times 512}$ ($K=5$: Global, Pop-Ridge, Rare, Cov-Ridge, Riemannian).
2. Cross-Attention Score:
   $$A_{k, i} = \frac{Q_{\text{branch}, k} \cdot (E_{\text{context}, i} W_K)^\top}{\sqrt{d}}$$
3. Episode-level Pooling & Softmax Routing:
   $$u_k = \text{MLP}\left( \text{MeanPool}_i(A_{k, i}) \right)$$
   $$g_k(E_{\text{context}}) = \frac{\exp(u_k / \tau)}{\sum_{j=1}^{K} \exp(u_j / \tau)}$$

여기서 $\tau = 0.5$ (Temperature Parameter)로 설정하여 뾰족하되 미분 가능한 Smooth Gating을 보장합니다.

---

## 3. 이 아키텍처가 한계를 어떻게 극복하는가

| 구분 | v24 확정 아키텍처 | v26 Proposal (EC-MoE) | **v27 AC-ICAR (본 제안)** |
|---|---|---|---|
| **Bag Representation** | 40→1 token projection (정보 파괴) | 40→1 token projection 유지 | **16 Multi-Resolution Tokens (정보 보존)** |
| **Episode Context Encoding** | 없음 (고정 가중치) | Deep Sets Mean Pooling ($z_e$, 노이즈 취약) | **Cross-Attention Context Set Querying** |
| **Routing Mechanism** | 고정 가중치 $\alpha_*$ | Top-2 Hard MoE (불연속, Collapse 위험) | **Continuous Soft Temperature Softmax** |
| **State/Covariance Control** | Covariance Subspace MLP | Adaptive $\lambda_e$ Scalar Tuning | **Riemannian Log-Euclidean Metric Branch** |
| **Gradient Flow** | Smooth | Discontinuous (Hard Top-2) | **Fully Differentiable Smooth Gradient** |

---

## 4. 구현 및 Config 설계

### 4.1 신규 클래스 구현 (`src/models/baseline.py`)

1. `MultiResolutionBagTokenizer`: 16-token 생성 모듈.
2. `RiemannianSubspaceBranch`: Log-Euclidean Metric 계산 모듈.
3. `CrossAttentiveSoftGating`: Context Cross-Attention 기반 Softmax Router.

### 4.2 Config 명세 (`configs/train_v27_medium_ac_icar.yaml`)

```yaml
base_config: configs/train_v24_medium_bag_proj_residual.yaml

model_overrides:
  architecture_version: 27
  bag_tokenizer:
    multi_resolution: true
    num_density_anchors: 8
    num_rare_outliers: 4
    num_covariance_basis: 4
  riemannian_branch:
    enabled: true
    shrinkage: 0.20
  soft_gating:
    enabled: true
    num_branches: 5
    temperature: 0.5
    entropy_penalty_weight: 0.05

trainer:
  max_epochs: 50
```

---

## 5. 실험 검증 프로토콜 (Verification Plan)

### 🧪 Experiment 1: Scale-Matched Task-Disentangled Evaluation
- **조건**: `--val-episodes 1000`, `--effect-scale 0.7` 통일 (T3-1 프로토콜).
- **목표**: State, Composition, Covariance 3대 task에서 paired bootstrap CI 검증.
- **승격 기준**: Overall AUROC **+0.03 이상** 또는 State AUROC **+0.05 이상**.

### 🧪 Experiment 2: Context Scaling & Noise Robustness Audit
- **조건**: Context Bag 수 `[10, 20, 40, 80, 160]` scaling curve 측정.
- **목표**: Donor Shift(0.70) 조건에서 v24 대비 AUROC 상승 곡선의 기울기 비교.

### 🧪 Experiment 3: Routing Mutual Information & Entropy Audit
- **지표**: Ground-truth task type $T$와 Soft Gating Vector $\mathbf{g}$ 간 상호정보량 $I(\mathbf{g}; T)$ 측정.
- **목표**: Gating이 static bias에 갇히지 않고 실제 signal type에 따라 가중치를 유연하게 할당함을 실증.

---

## 6. 결론

v27 AC-ICAR 아키텍처는 v24의 1-token 압축 병목과 v26 Proposal의 노이즈 취약성/Hard Routing 불연속성을 동시에 해결하는 원리적 해법입니다. 

Multi-Resolution Tokenizer로 세포 표현의 다중 해상도를 보존하고, Continuous Cross-Attention Soft Routing 및 Riemannian Log-Euclidean Metric을 도입함으로써 BagPFN의 performance plateau를 돌파할 준비가 완료되었습니다.
