# Current architecture

**Last updated**: `2026-08-08`
**Code baseline**: Architecture Version `34` — **v34-1536 (PathoBench 보고용 확정 모델, 2026-08-07)**.
v30 = 확정 baseline (합성/Musk), v34 = v30 B1(`poolz_l2`)·B2(cardinality) 계승 + large-context(1536-d) +
MLA 효율화. v35 = 데이터 단독 arm(rare-free, §61). 활성 개선안: **Q1 structured population**(§62, 별도 proposal).

> [!IMPORTANT]
> 차원 표기 — 차원은 대문자, 실제 토큰은 소문자.

| 기호 | 의미 | 값 |
|---|---|---|
| `E` | number of episodes in one batch (`episode_batch_size`) | **v34 4** / v35 1 / 추론 1 |
| `B` | number of bags in one episode (`num_bags`, 에피소드마다 추첨) | 학습 **[60, 100]** / 추론 = 코호트 전체 (C+Q) |
| `N` | number of instances in one bag (inference에서 bag마다 상이) | 학습 v34 [1, 8192] / v35 [1, 16384] |
| `I` | dimension of one instance | **1536** (UNIv2) |
| `C` | number of context bags in one episode, C+Q=B | |
| `Q` | number of query bags in one episode, C+Q=B | |
| `C0`, `C1` | number of context bags labeled 0 / 1, C0+C1=C | |
| `M` | number of memory tokens per class (총 2M) | 8 |
| `D` | dimension of hidden layers | 256 |
| `R` | ridge dimension | 64 |
| `T` | number of structured tokens per bag | 40 |
| `K` | number of tail fractions | 3 ([0.01, 0.05, 0.15]) |
| `F` | number of rare fractions | 4 ([0.01, 0.05, 0.10, 0.20]) |

v24/v30 문서는 [`history/`](history/) 및 git 기록에 보존. 최신 개발 상태는 [`current_status.md`](current_status.md) §59~§62.

---

## 1. 입출력 텐서 계약 (Input / Output Specification)

에피소드 = 해당 에피소드의 **전체 context bags + query bags**. v22부터 context를 K개로 줄이는 retrieval 계층은 없음. instance = 단일 세포, 특징 차원 $I = 1536$.

```text
input:
  context instances  [E, C, N, I]     (합성 학습 E=8, 추론 E=1; inference에서 N은 bag마다 상이)
  context labels     [E, C]           (Y \in {0, 1})
  query instances    [E, Q, N, I]
output:
  query logits       [E, Q, 2]
```

query label은 representation / normalization / class memory / ridge / covariance 경로에 절대 흐르지 않는다.

---

## 2. 전처리 & 표현 (v30 B1/B2 계승)

### ① Context-Pool Standardized Representation (`poolz_l2`, v30 B1)

per-bag centering(legacy, v24)을 **context-pool 대각 표준화**로 대체 (`bag_representation: poolz_l2`):

$$\mu_{ctx} = \frac{\sum_{i \in ctx} \sum_{j} x_{i,j}}{\sum_{i \in ctx} N_i}, \qquad
\sigma_{ctx} = \sqrt{\frac{\sum_{i \in ctx, j} (x_{i,j} - \mu_{ctx})^2}{\sum_{i \in ctx} N_i}}$$

$$\tilde{x}_{i,j} = \frac{(x_{i,j} - \mu_{ctx})/\sigma_{ctx}}{\|(x_{i,j} - \mu_{ctx})/\sigma_{ctx}\|_2}$$

- pool 통계는 **context bag 전체 세포**에서 cell-count 가중으로 계산 (실 bag 크기 편차 최대 1000배 → per-bag 평균의 평균은 오가중). **query 누출 없음** (`_context_pool_stats`).
- `poolz_l2`는 magnitude를 유계화해 bag 크기 편향을 줄인다 (corr(prob, log n) +0.327→+0.059). `poolz`(L2 없음)는 크기 교란이 커 음성 (§28).
- legacy per-bag centering은 rank $(N_i-1)$ 사영이라 소형 bag에서 파괴적 ($N_i=1$ → 0벡터) — Musk(median 12) 병목의 실체 (§26).

### ② Cardinality-faithful 에피소드 샘플링 (v30 B2)

| 버전 | num_cells | 비고 |
|---|---|---|
| v30 | [1, 1024] log-uniform | Musk 밴드 보존 (median 34) |
| v34-1536 | [1, 8192] log-uniform | PathoBench 보고용 |
| v35 | [1, 16384] + power 1.5 | 데이터 단독 arm (rare-free) |

B1·B2는 상호 필수 — B2만(legacy 뷰) 적용 시 n=1 bag이 0벡터 → NaN, B1만(S1) 적용 시 구간 교환으로 음성. 4D batched 경로 때문에 **에피소드 간** 크기 변동만 가능.

### ③ Covariance Subspace Shrinkage (노이즈 방어)
- SNR-adaptive covariance subspace fitting, `subspace_shrinkage: 0.25`. whitening NaN 시 Identity fallback.
- 거리 벡터 $[d_0, d_1, d_0 - d_1, \text{sep}]$ → 2-layer MLP learned head.

### ④ Auxiliary Pairwise Ranking Loss (`weight: 0.10`)

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + 0.10 \cdot \mathcal{L}_{Ranking}, \quad
\mathcal{L}_{Ranking} = \max(0, \gamma - (P_1(Q^+) - P_1(Q^-)))$$

---

## 3. Tokenization (bag → T=40 tokens, 전부 I-dim)

```text
1 global_summary : per-bag spread (표준편차) — bag mean이 아님 (mean은 centering에만 사용)
36 slot tokens   : 12 slots × (center, spread, rare)
                   anchor 12개 = 에피소드 context에서 1회 계산, 에피소드 내 **모든 bag(context+query) 공유**
                   (구성 방식은 아래 "anchor 구성" 참조 — density 8은 k-means 계열, rare 4는 greedy)
                   assignment = softmax(sim(cell, anchor)/temp), temp=0.1 (slot 축 softmax → cell 순서 불변)
                   center = slot_mean + σ·enc(cat(anchor, slot_mean−anchor, metadata))
                   spread = slot_std  + σ·enc(cat(anchor, slot_std,      metadata))
                   rare   = rare_state+ σ·enc(cat(anchor, rare_state−anchor, metadata))
                   rare_state = assignment×slot-mean distance top-k softmax 가중평균 (fraction 0.05)
                   metadata = (log proportion, dispersion) — enc 입력 특징 (token 아님)
3 tail tokens    : fractions [0.01, 0.05, 0.15]
                   novelty = 1 − nearest-anchor similarity로 top-k cells 선별,
                   (cell − nearest anchor) 편차를 공유 tail encoder로 인코딩 후 mean-pool
```

- slot index는 **에피소드 간(cross-episode)** 안정적 의미가 없음 → slot-index embedding 없음 (위치별 bottleneck이 대체).
  단 **에피소드 안에서는 anchor가 모든 bag에 공유**되므로 slot `i`는 context·query 전 bag에 대해 동일한 anchor다
  (= bag 간 정렬은 성립). class memory·population routing이 slot 수준 비교를 할 수 있는 근거가 이것이다.
  ※ `_typed_bag_tokens` 코드 주석의 "no stable cross-bag identity"는 정확히는 cross-**episode**를 뜻한다.
- v34/v35는 `bag_representation: poolz_l2` 뷰에서 이 40 token을 생성 (aggregator는 per-bag, slot/tail 통계).

### anchor 구성 (`_population_candidates` → `_select_anchors`)

```text
① 후보 풀 (bag 크기 불변) — _population_candidates, context bag마다 독립
   k = min(context_samples_per_bag=32, N_i)
   directions = _candidate_directions[:k]        (고정 random 방향 buffer)
   weights    = softmax(10 · normalize(cell) @ directionsᵀ, dim=cell축)
   candidates = normalize(weightsᵀ @ normalize(cells))        -> (k, I)
   ⇒ bag의 cell 수와 무관하게 **bag당 정확히 32개**(N_i<32면 N_i개)의 soft 후보.
     cell 순서 불변. 에피소드 후보 풀 = 32 × C개.

② density anchor 8개 — _select_anchors, 후보 풀 위에서 (원본 cell 전체가 아님)
   centrality = cos(candidate, 후보 풀 평균방향)
   seed = centrality 내림차순 상위 85% 구간(density_limit)을 균등 분위수로 8개 추출
          (outlier에서 시작하지 않기 위한 결정적·순서 불변 시드)
   refine = softmax(sim / density_temperature=0.15) 가중평균 → normalize, 4회 반복
   ⇒ **후보 풀에 대한 soft spherical k-means** (density_refinement_steps=4)

③ rare anchor 4개 (= num_slots − num_density_slots) — k-means가 아님
   residual  = 1 − max_s cos(candidate, density_s)
   diversity = 이미 뽑힌 rare anchor들과의 (1 − cos) 최솟값
   4회 반복: argmax(residual × diversity) 선택 후 제외      (greedy farthest-point)

anchors = cat(density 8, rare 4)                              -> (12, I)
```

> [!IMPORTANT]
> **bag 크기 불변 anchor 계약 (활성)**: `_population_candidates`가 bag 크기와 무관하게 bag당 `k=32`를
> 반환하므로, 대형 bag(30k 타일)이 후보 풀을 지배하지 않는다. 이것이 ⓐ §59.1에서 chunk 분할 시
> anchor 오염이 문제로 지목된 이유이자, ⓑ bag 단위 스트리밍이 anchor를 **bit-identical**하게 유지할 수
> 있는 이유다(`_select_anchors`가 받는 후보 텐서가 경로와 무관하게 동일). **단 `N_i < 32`인 소형 bag은
> 후보가 `N_i`개로 줄어든다** — Musk(median 12)가 여기 해당하며, 대형/소형 혼합 분포에서 후보 풀 구성이
> bag 크기에 따라 달라지는 유일한 지점이다.

## 4. Aggregation — 40→1 Projection (v24-B1 계승)

```text
mean_token = arithmetic mean(40, I) -> (1, I)
40 projection = position-specific Linear(I -> 64) × 40개   -> (64×40) = 2560
concat [projected(2560), mean_token(I)]                     -> 2560 + I = 4096
Linear(4096 -> I)                                          -> bag token (1, I)   [token 공간 I]
```

- 이 1-token이 `global_shape_classifier`, `_class_memories`(context), `_population_memory_logits`(query)에 흐른다.
- **§62 Q1**: 이 40→1 압축은 라벨 정보 전에 task-무관 고정 선형 사상으로 token을 1개로 죽이고, 그 결과 population routing softmax가 길이 1 축에 걸려 무력 (probe +0.16, §62-4). Q1은 population 경로에서 40 token을 유지한다 → [`architecture_v36_q1_structured_population_proposal.md`](architecture_v36_q1_structured_population_proposal.md).

---

## 5. Architecture Pipeline (E=1) — 전체 분기

```text
1. Input: (B, N, I)
2. Tokenization: (B, 40, I)
3. Aggregation (projected): (B, 1, I)
4. Split: c_bags = (C, 1, I) = (C, I), q_bags = (Q, 1, I) = (Q, I)
```

### Global Branch (projected token)

```text
G-1. Encoding: c = MLP_R(Norm(c_bags)) = (C, R=64),  q = MLP_R(Norm(q_bags)) = (Q, R)
G-2. Ridge: class-balanced closed-form ridge (center/rms → gram + λI → solve),
            c 라벨로 계수 fit → q logits (Q, 2), ridge_scale 배
G-3. Attention residual: set-encoder + cross-attn(클래스 그룹) → (Q, 2), residual_scale 배
G-4. global_shape_logits = ridge_scale·ridge + attn_residual_scale·attention  (Q, 2)
```

### Context Bag Line (class memories, projected token)

```text
C-1. Class split: c0 = c_bags[label==0] = (C0, I), c1 = (C1, I)
C-2. Encoding I->D: c0 = Proj_D(Norm(c0)) = (C0, D), c1 = (C1, D)   [공유 projection]
C-3. Memory attention: m = (M, D) 공유 learnable seeds
     m0 = Attn(m, c0, c0) = (M, D),  m1 = Attn(m, c1, c1) = (M, D)
C-4. Memory self-attention: m0 = SelfAttn(m0, m0, m0) = (M, D), m1 동일
```

### Query Bag Line (population memory attention, projected — 현행)

```text
Q-1. q = Proj_D(Norm(q_bags)) = (Q, 1, D)
Q-2. token_weights = softmax(slot_importance(q)/T_temp) = (Q, 1)      [축 길이 1 → 무력]
Q-3. q0 = Attn(q, m0, m0) = (Q, 1, D), q1 = Attn(q, m1, m1) = (Q, 1, D)
Q-4. r0 = cat[q, q0, q−q0, q⊙q0] = (Q, 4D) → Scorer → s0 (Q, 1);  r1 동일 → s1 (Q, 1)
Q-5. population_attention_logits = [s0, s1] = (Q, 2)
```

### Abundance (population 분기의 절반)

```text
P-1. metadata = slot_metadata = (C, 24)   [12 slots × (log proportion, dispersion)]
P-2. abundance_ridge_logits = ridge(metadata, labels, query) → (Q, 2), abundance_scale 배

⇒ population_logits = abundance_scale·abundance_ridge + attention_scale·population_attention  (Q, 2)
```

### Rare Branch (raw 세포 직접)

```text
R-1. query raw instances (N_i, I) → instance_input_norm → Proj_D → (N_i, D)
R-2. ĥ = L2normalize(Proj_D(...)),  m̂_c = L2normalize(m_c)          [둘 다 단위구로]
     τ = exp(rare_similarity_log_scale).clamp(0.1, 50)              [학습되는 온도, init exp(log 5)=5]
     evidence[c, n] = logsumexp_m( τ · ⟨ĥ_n, m̂_{c,m}⟩ ) − log(M)   [클래스별·cell별, M=8]
       → M개 memory token에 대한 smooth-max. −log(M)은 memory 개수 정규화.
R-3. F=4 fractions [0.01, 0.05, 0.10, 0.20]별로 evidence를 **cell 축 top-k 평균**
     (k = ceil(fraction·N_i), 최소 1) → fraction_scores (Q, 2, 4)
     → rare_evidence_head MLP → rare_logits (Q, 2), tail_scale 배
     ※ v34 = ON (default), v35 = OFF (meta_enable_rare_evidence: false, §61 —
       코드 삭제가 아니라 `force_rare_logits_zero`로 rare_logits=0, ckpt 호환·가역)
     ※ 이 분기만 유일하게 **raw cell을 직접** 소비한다 (나머지는 전부 40 token 경유).
```

### Covariance Branch

```text
CV-1. covariance ridge: covariance_sketch (C, 2080) → ridge → (Q, 2), cov_ridge_scale 배
      ※ 2080 = d(d+1)/2, d = aggregator_covariance_sketch_dim = 64.
        sketch는 (centered cells @ P) 의 d×d 상관행렬(shrinkage 0.25)의 **상삼각 벡터화**이며,
        64는 사영 차원이지 토큰 길이가 아니다. covariance_matrix는 (C, 64, 64).
CV-2. covariance relation: covariance_matrix → subspace
        (class delta + pooled whitening + top-1 eigen filter) → variance-log features (C, 1)
        → class prototype 비교 (learned_head MLP: [d0, d1, d0−d1, sep]) → (Q, 2), residual 0.5 배
```

### Fusion (최종 logits)

```text
F-1. evidence = stack[global_shape, population, rare]              (Q, 2, 3)
F-2. interaction = fusion_scorer(cat(evidence, pair_products, |pair_diffs|))  (Q, 2)
F-3. final = global_shape
           + population_scale·population_logits
           + tail_scale·rare_logits
           + fusion_scale·interaction
           + cov_res_scale·covariance_logits
           + cov_rel_res_scale·covariance_relation_logits          (Q, 2)
     ※ population/tail/fusion/cov_res scale = 학습 파라미터의 (floored) sigmoid.
     ※ routing의 class 조건부화는 공유 함수 f(q, m_c) 형태로만 (label-permutation equivariance 유지).
```

---

## 5b. 실행 경로 — **train과 eval은 서로 다른 코드 경로다**

같은 수학을 서로 다른 두 구현이 계산한다. 어느 쪽을 보고 있는지 모르면 진단이 틀린다.

| | **train (dense / batched)** | **eval (ragged / per-bag)** |
|---|---|---|
| 진입점 | `forward_episode_batch` → `_forward_dense` | `BaseModel.forward` (list-of-bags) |
| 입력 | 패딩된 `[E, B, N_max, I]` + cell_mask | bag마다 다른 길이의 텐서 리스트 |
| anchor 후보 | `_population_candidates_batched` (한 번의 masked softmax) | `_population_candidates` per-bag 루프 |
| population 분기 | `_population_memory_logits_batched` | `_population_memory_logits` |
| bag 스트리밍 | 없음 (dense) | `stream_eval_bags=True` **기본 on** |
| query view | 전 bag | **query bag만** `_bag_view` 생성 |

- **경로 선택은 `self.training`이 결정**한다 — `_context_anchors`는 `self.training`일 때만 batched
  후보를 쓴다. batched 경로는 `[C, N_max, I]` 패딩을 만들어 full-tile 슬라이드에서 OOM이 나므로
  **eval은 항상 per-bag 루프**다.
- **수치 계약**: batched 후보는 **모든 bag이 ≥32 cell일 때 정확**하고, 아니면 per-bag 루프로
  자동 폴백한다(`k < context_samples_per_bag` 분기). 스트리밍은 9개 representation key
  `‖Δ‖∞ < 1e-4`, **anchors bit-identical**, 실데이터 AUROC 동일이며 peak VRAM 40,990 → 18,930 MiB.
  A/B는 `BAGPFN_DISABLE_BAG_STREAMING=1` (§59.3).
- **train/eval 통계 일치**: `_context_pool_stats`는 `unbiased=False`(Bessel 보정 없음)를 쓴다 —
  보정을 켜면 train(dense)과 eval(ragged)에서 0.25% 불일치가 생긴다.
- **쿼리 독립성**: `_covariance_relation_scores`가 query 축 margin RMS로 정규화하므로, 한 forward에
  여러 query를 넣으면 서로 결합된다. 그래서 공식 평가는 **query당 1회 forward**가 기본이고
  `--batch-queries`는 **별도의 미검증 프로토콜**이다(확률 0.01~0.05 이동, 커밋 `5869535`).
- **폴드 단위 캐싱 (§62)**: pool 통계·anchor가 context 전용이라, 한 폴드의 표현을 1회 패스로 전부
  계산해도 query별 패스와 **‖Δ‖∞ = 0.000e+00**(실측)이다. 현행 eval은 query마다 context를
  재인코딩하므로 약 50× 낭비한다 (EGFR 폴드당 16,306 → 324 bag-인코딩).

---

## 6. 학습 목적 (Loss)

$$\mathcal{L} = \mathcal{L}_{CE} + 0.10 \cdot \mathcal{L}_{Ranking}
+ w_{sparsity} \cdot \text{sparsity} + w_{balance} \cdot \text{balance}$$

- $w_{sparsity} = w_{balance} = 0.0$ (v34/v35), `routing_temperature = 0.5`.
- CE 0.685 부근 gradient 소멸 탈출용 ranking loss (weight 0.10).

## 7. 모델 스펙 (v34-1536)

```text
Instance Dimension (I)         : 1536
Hidden Dimension (D)           : 256
Memory tokens per class (M)    : 8  (총 2M = 16)
Structured tokens per bag (T)  : 40  (1 global + 12×3 slots + 3 tails)
Aggregation                    : 40→1 (position bottleneck 64 + residual exact mean)
Ridge Dimension (R)            : 64
Attention Heads                : 8
Set Layers                     : 1
Precision (학습)                : **bf16-mixed — 예외 없는 필수 계약 (2026-08-08 강제)**.
                                 ridge/relation solve 내부는 fp32로 승격.
                                 `configs/trainer/` **모든 group**에 고정되어 있고
                                 `tests/test_precision_contract.py`가 활성 train config 전부 +
                                 trainer group 전부를 검사한다 (group을 바꿔 우회 불가).
                                 ⚠️ 아래 "확정 수치"의 v34/v35 ckpt는 이 강제 이전에 **fp32 폴백**
                                 (해당 group에 precision 미설정 → Lightning 32-true)으로 학습됐다.
                                 재실행은 bf16-mixed로 돌아가며 그 ckpt를 재현하지 않는다
                                 (역사적 재현은 `trainer_overrides.precision: 32-true`).
Precision (평가)                : **fp32**. `scripts/test_pathobench.py`는 Lightning trainer 없이
                                 모델을 직접 빌드하므로 위 계약이 적용되지 않는다. 보고된 AUROC가
                                 전부 이 경로에서 나왔으므로 **바꾸면 모든 수치가 이동**한다.
Bag Representation             : poolz_l2
Routing Temperature            : 0.5 (sparsity/balance = 0.0)
Covariance relation            : enabled / learned_head / subspace / rank 1 / whiten / shrinkage 0.25 / residual 0.5
Rare branch                    : v34 ON (default) / v35 OFF (meta_enable_rare_evidence: false)
Episode batch (E)              : 4 (v34, configs/data/default.yaml) / 1 (v35)
Bags per episode (B)           : [60, 100] 추첨
Anchor 후보                     : context bag당 32개 (bag 크기 불변; N_i<32면 N_i)
학습 예산                       : 1024 ep/epoch × 50 epoch = 51,200 episodes (v34), batch 4
확정 수치                       : best val_ce 0.4419 (checkpoints/20260806_215800/.../epoch=048-...),
                                 fp32 폴백으로 학습됨 (위 Precision 주석 참조)
```

## 8. Source of Truth 파일

- Backbone & Version: `src/models/baseline.py` (코드 `architecture_version`는 projection 경로에서 24 유지; v34/v35는 config/실험 단위)
- Resolved Production Entry: `configs/train_v34_phase0_largectx_1536.yaml` (자체 포함형, PathoBench 보고용)
- Data Interface & Collators: `src/modules/data_interface.py`
- Multi-task Loss & Metrics: `src/modules/model_interface.py`
- Architecture Verification Suites: `tests/` (41 tests)
- 활성 개선안: `docs/architecture_v36_q1_structured_population_proposal.md`
