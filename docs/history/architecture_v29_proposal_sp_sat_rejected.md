# Architecture v29 Proposal: Slot-Parallel Self-Attentive 40-Token Meta-Classifier (SP-SAT)

> [!IMPORTANT]
> **미구현, 다른 설계로 대체 (archived 2026-08-02).** 대화 중 비판적 분석 결과:
> 핵심 전제("40토큰 보존이 이득")가 v22(40 token, CE 0.5946) < v24(1 token,
> CE 0.5903)로 이미 반증됐고, 이 문서가 재사용하는 population-slot 분할 자체의
> 지도학습 상한이 E7 실측(purity 0.335, 진행/폐기 게이트 사이)으로 불확실함이
> 확인됨. 대신 사용자가 제안한 **CLS-token cross-attention pooling**(raw cell
> 전체를 직접 보는, 분할에 의존하지 않는 별도 아이디어)을 `architecture_version=26`
> (`configs/train_v26_medium_cls_token_pool.yaml`)으로 실제 구현·학습 진행.
> 원문은 그대로 보존.

**작성일**: `2026-08-02`  
**작성자**: Antigravity AI (Pair Programming with User)  
**기반**: 사용자 제안 아이디어 (40-Token 구조 보존 및 Slot-Parallel Class Memory)  
**상태**: 설계 제안서 (Architecture Proposal)

---

## 0. Executive Summary

v24 아키텍처는 40개의 structured token(1 global + 36 slots + 3 tails)을 **Bag당 1개의 projected token으로 강제 압축**함으로써 연산 효율성을 얻었으나, 이 과정에서 각 슬롯 고유의 공간적·통계적 신호(세포 군집 중심, 퍼짐, 극단치)가 대량 손실된다는 한계가 지적되었습니다.

제안된 **Architecture v29: SP-SAT (Slot-Parallel Self-Attentive 40-Token Meta-Classifier)**는 Bag 표현을 1개 토큰으로 압축하지 않고 **40개 토큰 구조를 끝까지 유지**하면서, 40개 슬롯을 병렬 차원(Slot Dimension)으로 활용하는 혁신적인 메타 러닝 아키텍처입니다.

### 핵심 아이디어 3가지:
1. **Intra-Bag Self-Attention Encoder**: Bag 내부 40개 토큰 간 상호작용을 파악하여 각 슬롯의 표현력을 정교화 ($\mathbb{R}^{40 \times D} \rightarrow \mathbb{R}^{40 \times D}$).
2. **Slot-Parallel Class Memory ($M_0, M_1 \in \mathbb{R}^{40 \times D}$)**: 40개 슬롯 위치를 병렬 배치 차원으로 처리하여, 각 슬롯 위치별 전용 Class 0 / Class 1 메모리 토큰을 독립 생성.
3. **Slot-Wise Multi-Branch Scoring**: Query Bag의 40개 토큰 각각을 대응되는 Class Memory와 매칭하고, 슬롯별 중요도(Slot Importance) 기반 가중합으로 최종 `[Q, 2]` Logits 생성.

---

## 1. 문제 의식 및 동기 (Motivation)

### 1.1 v24 (1-Token Projection)의 한계
* **정보 압축 병목 (40:1 Bottleneck)**:
  v24는 40개 토큰($40 \times 512 = 20,480$차원)을 Bottleneck + Residual Mean 과정을 거쳐 $512$차원 1개 토큰으로 압축합니다.
  이로 인해 **특정 세포 군집(Slot)에만 존재하는 특이 반응 신호**나 **희귀 세포(Tail) 신호**가 평균 표현 속에 묻히는 정보 파괴 현상이 발생합니다.

### 1.2 v29의 접근법 (40-Token 구조 보존)
* 40개 토큰의 구조적 독립성을 보존하여, **"어느 슬롯 위치에서 Class 0과 Class 1의 차이가 발생하는가?"**를 슬롯 단위로 정밀하게 추적합니다.
* 40개 슬롯을 독립된 병렬 차원(Slot Dimension)으로 다룸으로써 메타 메모리의 표현 역량(Memory Capacity)을 40배 확장합니다.

---

## 2. 입출력 텐서 및 변수 정의 (Notation)

| 기호 | 의미 | 대표 수치 (기본 설정) |
|---|---|---|
| $E$ | 에피소드 수 (Outer Episodes) | $E = 1$ (추론) 또는 $E = 8$ (학습) |
| $C$ | Context Bag 개수 ($C_0 + C_1$) | $C \approx 60 \sim 100$ |
| $Q$ | Query Bag 개수 | $Q \approx 10 \sim 20$ |
| $N$ | Bag 1개 내부의 세포(Instance) 수 | $N \approx 500 \sim 1,000$ |
| $S$ | Bag당 Structured Token 개수 | **$S = 40$** (1 Global + 36 Slots + 3 Tails) |
| $D$ | Feature / Token Dimension | $D = 512$ (입력) / $D_{hidden} = 256$ (메타 연산) |

---

## 3. Architecture v29 (SP-SAT) 데이터 흐름 및 수학적 명세

```text
Bag Input (N cells x 512d)
       │
       ▼  (Structured Population Aggregator)
Bag Tokens X_i ∈ R^(40 x 512)
       │
       ▼  (1단계: Intra-Bag Self-Attention Encoder)
Refined Bag Tokens X̃_i ∈ R^(40 x 512)
       │
  ┌────┴───────────────────────────┐
  ▼ (Context Bags)                  ▼ (Query Bags)
Split by Label (Y=0, Y=1)        Query Tokens X̃_q ∈ R^(40 x 512)
  │                                 │
  ▼ (2단계: Slot-Parallel)         │
M₀ ∈ R^(40 x 256)                   │
M₁ ∈ R^(40 x 256)                   │
  │                                 │
  └────────────────┬────────────────┘
                   ▼ (3단계: Slot-Wise Matching & Scoring)
         Attended_0 ∈ R^(40 x 256)
         Attended_1 ∈ R^(40 x 256)
                   │
                   ▼ (Scorer MLP & Slot Importance Weighting)
         Query Logits ∈ R^(Q x 2)
```

---

### 3.1 1단계: Intra-Bag Self-Attention Encoder

각 Bag $i$에서 추출된 40개 structured token $X_i \in \mathbb{R}^{40 \times D}$는 Bag 내부의 전역 평균(Global), 슬롯 중심/퍼짐(Slots), 극단치(Tails) 정보를 가지고 있습니다.

이 40개 토큰 간의 맥락적 연결을 강화하기 위해 **Bag-level Self-Attention Transformer**를 통과시킵니다:

$$\tilde{X}_i = \text{LayerNorm}(X_i + \text{MultiHeadSelfAttention}(X_i, X_i, X_i))$$
$$\tilde{X}_i = \text{LayerNorm}(\tilde{X}_i + \text{FFN}(\tilde{X}_i)) \in \mathbb{R}^{40 \times D}$$

* **효과**: Global Mean 토큰이 특정 Slot 토큰의 정보를 참조하고, Outlier Tail 토큰이 관련 Slot 토큰과 정보를 교환하여 **각 슬롯 토큰의 표현력이 획기적으로 정교해집니다.**

---

### 3.2 2단계: Slot-Parallel Class Memory 구축 ($M_0, M_1 \in \mathbb{R}^{40 \times D_{hidden}}$)

Context Bag들을 라벨 $Y \in \{0, 1\}$에 따라 나누어 두 그룹의 텐서를 만듭니다:
* **Class 0 Context**: $\tilde{X}_{C0} \in \mathbb{R}^{C_0 \times 40 \times D}$
* **Class 1 Context**: $\tilde{X}_{C1} \in \mathbb{R}^{C_1 \times 40 \times D}$

40개의 슬롯 위치 각각에 대응하는 학습 가능한 메모리 시드 $\text{Seeds} \in \mathbb{R}^{40 \times D_{hidden}}$를 배치하고, **40개 슬롯을 병렬(Parallel Slot Axis)로 처리**하여 Class Memory를 만듭니다.

#### 슬롯 $k$ ($k=1 \dots 40$) 단위 연산:
* **Class 0 Memory ($M_0$)**:
  * Query: $\text{Seed}_k \in \mathbb{R}^{1 \times D_{hidden}}$
  * Key/Value: $\tilde{X}_{C0}[:, k, :] \in \mathbb{R}^{C_0 \times D_{hidden}}$
  $$M_{0, k} = \text{CrossAttention}\left(Q=\text{Seed}_k, K=\tilde{X}_{C0}[:, k, :], V=\tilde{X}_{C0}[:, k, :]\right) \in \mathbb{R}^{1 \times D_{hidden}}$$
* **Class 1 Memory ($M_1$)**:
  * Query: $\text{Seed}_k \in \mathbb{R}^{1 \times D_{hidden}}$
  * Key/Value: $\tilde{X}_{C1}[:, k, :] \in \mathbb{R}^{C_1 \times D_{hidden}}$
  $$M_{1, k} = \text{CrossAttention}\left(Q=\text{Seed}_k, K=\tilde{X}_{C1}[:, k, :], V=\tilde{X}_{C1}[:, k, :]\right) \in \mathbb{R}^{1 \times D_{hidden}}$$

40개 슬롯 연산 결과를 Stack하면 최종 클래스 메모리가 완성됩니다:
$$\mathbf{M_0 = \text{Stack}([M_{0, 1}, M_{0, 2}, \dots, M_{0, 40}]) \in \mathbb{R}^{40 \times D_{hidden}}}$$
$$\mathbf{M_1 = \text{Stack}([M_{1, 1}, M_{1, 2}, \dots, M_{1, 40}]) \in \mathbb{R}^{40 \times D_{hidden}}}$$

---

### 3.3 3단계: Slot-Wise Query Matching & Scorer Pooling

라벨을 모르는 Query Bag $q$의 토큰 $\tilde{X}_q \in \mathbb{R}^{40 \times D_{hidden}}$와 구축된 $M_0, M_1$을 슬롯별로 비교합니다.

#### 1) Query Cross-Attention
각 슬롯 위치 $k$에 대해:
* **Class 0 Attended**:
  $$\text{Attended}_{0, k} = \text{CrossAttention}\left(Q=\tilde{X}_{q, k}, K=M_{0, k}, V=M_{0, k}\right) \in \mathbb{R}^{1 \times D_{hidden}}$$
* **Class 1 Attended**:
  $$\text{Attended}_{1, k} = \text{CrossAttention}\left(Q=\tilde{X}_{q, k}, K=M_{1, k}, V=M_{1, k}\right) \in \mathbb{R}^{1 \times D_{hidden}}$$

#### 2) Relation Feature & Scorer
각 슬롯 $k$별로 상호작용 특징을 결합합니다:
$$\text{Relation}_{c, k} = \left[ \tilde{X}_{q, k} \;;\; \text{Attended}_{c, k} \;;\; (\tilde{X}_{q, k} - \text{Attended}_{c, k}) \;;\; (\tilde{X}_{q, k} \odot \text{Attended}_{c, k}) \right] \in \mathbb{R}^{4D_{hidden}}$$

슬롯 점수 채점기 MLP ($\text{Scorer}: 4D_{hidden} \rightarrow 1$)를 통과시켜 슬롯별 점수를 계산합니다:
$$\text{RawScore}_{c, k} = \text{Scorer}(\text{Relation}_{c, k}) \in \mathbb{R}^1$$

#### 3) Slot Importance Weighting & Final Logits
40개 슬롯 중 분류에 중요한 슬롯에 더 큰 가중치를 부여하기 위해 **Slot Importance Softmax Weight ($w_k$)**를 계산하여 가중합합니다:
$$w = \text{Softmax}\left( \frac{\text{SlotImportanceMLP}(\tilde{X}_q)}{\tau} \right) \in \mathbb{R}^{40}$$

$$\text{Score}_0(q) = \sum_{k=1}^{40} w_k \cdot \text{RawScore}_{0, k}$$
$$\text{Score}_1(q) = \sum_{k=1}^{40} w_k \cdot \text{RawScore}_{1, k}$$

최종 Logits:
$$\mathbf{\text{Query Logits}(q) = \left[ \text{Score}_0(q), \; \text{Score}_1(q) \right] \in \mathbb{R}^2}$$

---

## 4. PyTorch 의사 코드 (PyTorch Pseudocode)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SlotParallelSATMetaClassifier(nn.Module):
    def __init__(self, token_dim=512, hidden_dim=256, num_slots=40, num_classes=2):
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        
        # 1. Intra-Bag Self-Attention Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=8, batch_first=True)
        self.intra_bag_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.proj_in = nn.Linear(token_dim, hidden_dim)
        
        # 2. Memory Seeds (40 slots x hidden_dim)
        self.memory_seeds = nn.Parameter(torch.randn(num_slots, hidden_dim) / (hidden_dim ** 0.5))
        self.memory_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # 3. Query Cross-Attention & Scorer
        self.query_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.slot_importance = nn.Linear(hidden_dim, 1)

    def forward(self, context_tokens, context_labels, query_tokens):
        """
        context_tokens: [C, 40, 512]
        context_labels: [C] (0 or 1)
        query_tokens:   [Q, 40, 512]
        """
        # Step 1: Intra-Bag Self-Attention
        C, S, D = context_tokens.shape
        Q = query_tokens.shape[0]
        
        ref_context = self.proj_in(self.intra_bag_transformer(context_tokens)) # [C, 40, 256]
        ref_query   = self.proj_in(self.intra_bag_transformer(query_tokens))   # [Q, 40, 256]
        
        # Step 2: Slot-Parallel Class Memory Construction
        # Memory shape: [2, 40, 256]
        memories = []
        for c in range(2):
            c_bags = ref_context[context_labels == c] # [C_c, 40, 256]
            # Transpose for slot-parallel batching: [40, C_c, 256]
            c_bags_slot_first = c_bags.transpose(0, 1)
            seeds_slot_first = self.memory_seeds.unsqueeze(1) # [40, 1, 256]
            
            # Cross-Attention over C_c bags for each slot in parallel
            # Query: [40, 1, 256], Key/Value: [40, C_c, 256]
            m_c, _ = self.memory_cross_attn(seeds_slot_first, c_bags_slot_first, c_bags_slot_first)
            memories.append(m_c.squeeze(1)) # [40, 256]
            
        M0, M1 = memories[0], memories[1] # Each [40, 256]
        
        # Step 3: Query Matching & Scoring
        query_slot_first = ref_query.transpose(0, 1) # [40, Q, 256]
        
        logits_list = []
        for M_c in [M0, M1]:
            M_c_expanded = M_c.unsqueeze(1).expand(-1, Q, -1) # [40, Q, 256]
            # Attended shape: [40, Q, 256]
            attended, _ = self.query_cross_attn(query_slot_first, M_c_expanded, M_c_expanded)
            
            # Transpose back: [Q, 40, 256]
            att_q = attended.transpose(0, 1)
            
            # Relation features: [Q, 40, 4 * 256]
            rel = torch.cat([ref_query, att_q, ref_query - att_q, ref_query * att_q], dim=-1)
            raw_scores = self.scorer(rel).squeeze(-1) # [Q, 40]
            
            # Slot Importance Weighting
            w = F.softmax(self.slot_importance(ref_query).squeeze(-1), dim=-1) # [Q, 40]
            final_score = (raw_scores * w).sum(dim=-1, keepdim=True) # [Q, 1]
            logits_list.append(final_score)
            
        logits = torch.cat(logits_list, dim=-1) # [Q, 2]
        return logits
```

---

## 5. 기존 아키텍처와의 비교 및 기대 효과

| 항목 | **v24 (Baseline)** | **v26 (EC-MoE)** | **v27 (AC-ICAR)** | **v29 (SP-SAT, 본 제안)** |
|---|---|---|---|---|
| **Bag Token 수** | 1 Token (압축) | 1 Token (압축) | 16 Tokens | **40 Tokens (손실 없음)** |
| **Intra-Bag 연산** | Residual Bottleneck | Residual Bottleneck | Dual-Path Split | **Intra-Bag Self-Attention** |
| **Class Memory 형태** | `[2, 8, 256]` | `[2, 8, 256]` | `[2, 16, 256]` | **`[2, 40, 256]` (슬롯 병렬)** |
| **Matching 단위** | Bag 1개 단위 | Expert Gating | Smooth Routing | **40개 슬롯 위치별 1:1 병렬 매칭** |
| **정보 파괴 여부** | 높음 (40:1) | 높음 (40:1) | 중간 (40:16) | **없음 (보존율 100%)** |

### 💡 핵심 기대 효과
1. **슬롯별 세밀한 해상도 보존**: Global 평균에 묻혔던 36개 Subgroup Slot 및 3개 Outlier Tail의 독립적 신호가 살아남아, 미세한 세포 구성 변화(State/Composition)를 정확히 판별함.
2. **연산 병렬성 (Slot Parallelism)**: 40개 슬롯을 PyTorch의 배치/시퀀스 차원으로 처리하므로, 40배 늘어난 토큰 수 대비 연산 속도 저하가 최소화됨.
3. **직관적인 해석 가능성 (Explainability)**: 분류 후 40개 슬롯 중 어떤 슬롯($w_k$)이 이번 Query 판단에 가장 기여했는지 슬롯 단위 중요도를 직접 시각화할 수 있음.

---
