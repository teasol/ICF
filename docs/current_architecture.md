# Current architecture

**Last updated**: `2026-08-01 13:20:00 KST`  
**Code baseline**: Architecture Version `24` (`architecture_version = 24`) — **v24-B1 (residual + bottleneck bag projection) 확정 (2026-08-01)**

이 문서는 현재 production config(`configs/train_v24_medium_bag_proj_residual.yaml`)가 사용하는 Architecture v24 모델 구조와 수학적 계약을 설명합니다. v24는 v22(retrieval 없는 base, §2~§4는 v24에서도 그대로 유지)에 **bag 내부 40-token 구조화 요약을 1개 학습된 projection token으로 압축**하는 §3.5를 추가한 버전입니다. `project_structured_tokens=false`로 두면 코드는 그대로 v22 동작으로 되돌아갑니다 (§3.5 참고). 최신 개발 상태는 [`current_status.md`](current_status.md) §3 "최종 결정", 실험 프로토콜은 [`current_experiments.md`](current_experiments.md)를 참고합니다.

> [!IMPORTANT]
> **v24는 label 기반 class-memory 압축을 바꾸지 않았습니다.** `_class_memories`는 여전히 context bag을 `context_labels`로 나눠 class당 8개 memory token으로 pooling합니다 (§4, `src/models/baseline.py:2446`). v24가 바꾼 것은 그 이전 단계, 즉 **bag 하나를 몇 개의 토큰으로 요약하는지**(§3.5)뿐입니다.

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

### ① Bag Centering + Per-Cell L2 Projection (구 "Z-Score Bag Studentization")

> [!IMPORTANT]
> **2026-08-04 코드 대조로 정정됨.** 이 절은 오래도록 per-feature 스튜던트화
> ($\tilde{x} = (x-\mu_i)/(S_i+10^{-5})$)라고 기술해 왔으나, **코드는 $S_i$로 나누지 않습니다.**
> 실제 구현(`_bag_view`, `src/models/baseline.py:618-644`)은 per-bag centering 후 **per-cell L2**로
> 사영합니다. $S_i$는 나눗셈에 쓰이지 않고 `global_summary` 토큰으로 **출력**됩니다.
> 참고로 문서가 기술했던 그 공식(진단 프로브의 `zscore` 변형)은 실측에서 **최하위**였습니다
> (합성 Medium 0.5054, [`history/archive.md`](history/archive.md) §19). 상세: `current_status.md` §26.

각 Donor Bag $i$의 세포 표현 $x_{i, j} \in \mathbb{R}^{512}$에 대해 Donor Centroid $\mu_i$와
per-feature Standard Deviation $S_i$를 구합니다:

$$\mu_i = \frac{1}{N_i} \sum_{j=1}^{N_i} x_{i, j}, \quad S_i = \sqrt{\frac{1}{N_i} \sum_{j=1}^{N_i} (x_{i, j} - \mu_i)^2 + 1e-6}$$

실제 적용되는 변환은 **centering + per-cell L2**입니다 (`bag_centered_representation=True`,
`bag_centered_l2_normalize=True`가 기본):

$$\delta_{i,j} = x_{i,j} - \mu_i, \qquad \tilde{x}_{i, j} = \frac{\delta_{i,j}}{\|\delta_{i,j}\|_2 + 10^{-6}}$$

$S_i$는 **`global_summary` 토큰**으로 aggregator에 전달되며(나눗셈에 사용되지 않음), centering 전
편차 $\delta_{i,j}$는 공분산 스케치 경로(`_covariance_sketch`, `_slot_covariance_sketch`)의 입력으로
그대로 쓰입니다 — 즉 크기(magnitude) 정보는 L2로 지워지지 않고 이 두 경로로 공급됩니다.

> **알려진 한계 (2026-08-04)**: per-bag centering은 $d$차원 공간을 **rank $(N_i-1)$** 부분공간으로
> 사영합니다. $N_i$가 작으면 파괴적입니다 — $N_i = 1$이면 $\delta = 0$이므로 bag 전체가 0벡터가
> 됩니다. 실데이터(Musk median 12 instances)에서 이것이 주 병목으로 측정되었습니다:
> `current_status.md` §26 및 [`history/musk_transfer_diagnosis_v30_proposal.md`](history/musk_transfer_diagnosis_v30_proposal.md).

### ② Top-k Sparse Evidence Tokenization (Sub-1% 희귀세포 핀포인트 추출)
97%+ 비반응 배경세포에 의해 0.5%~3% 희귀 세포 신호가 희석되는 현상을 방지함:
- 상위 1% ($k = \max(1, \lceil 0.01 \cdot N_i \rceil)$) 세포 인스턴스 $X_i^{sparse} \in \mathbb{R}^{k \times 512}$ 선별.
- Class Memories $M_c \in \mathbb{R}^{8 \times 512}$ ($c \in \{0, 1\}$)와 Direct Cross-Attention 수행 후 residual logit 결합.

> [!IMPORTANT]
> **2026-08-04 정정**: 이 절은 선별 기준을 "Bag 중심 $\mu_i$로부터의 이상 거리
> $d_{i,j} = \|\tilde{x}_{i,j}\|_2$"라고 기술해 왔으나, ①의 per-cell L2 때문에 코드에서
> $\|\tilde{x}_{i,j}\|_2$는 **모든 세포에 대해 항등적으로 1.0**이며 순위를 만들 수 없습니다.
> 실제 코드는 context anchor에 대한 **novelty**(`novelty = 1 - nearest_similarity`,
> `baseline.py:1074-1075`, `1289-1290`)로 tail을 정렬합니다.

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

## 3.5. Bag Structured-Token Projection (v24, residual + bottleneck)

v22/v23은 각 bag을 40개 structured token(1 global + 12 slots×3 + 3 tail)째로 분류기에 전달하거나(v22), 40개를 단순 arithmetic mean해 1개 token으로 줄였습니다(v23-A0, `mean_pool_structured_tokens: true`). **v24는 이 40개를 학습된 linear projection으로 압축해 1개 token으로 만듭니다** (`project_structured_tokens: true`). 확정된 v24는 아래 두 옵션을 모두 켠 residual + bottleneck variant(구 v24-B1)입니다.

1. **Position-specific bottleneck** (`projection_bottleneck_dim: 64`): 40개 token 각각에 **전용** `Linear(512 → 64)`를 적용합니다(토큰 위치마다 별도 가중치, 총 40개). 그 결과를 concat하면 `40 × 64 = 2560`차원.
2. **Residual exact-mean shortcut** (`projection_residual_mean: true`): 같은 40개 token의 정확한 arithmetic mean(512차원, v23-A0과 동일한 신호)을 위 2560차원 뒤에 concat합니다 → `2560 + 512 = 3072`차원.
3. 최종 `Linear(3072 → 512)` 하나로 bag당 512차원 token 1개를 만듭니다.

```text
40 structured tokens (40 x 512)
  -> [40개 위치별 전용 Linear(512, 64)]   (v24-B0에서 도입)
  -> concat                                (40 x 64 = 2560)
  -> concat with exact mean(40 tokens)     (2560 + 512 = 3072)   <- residual shortcut (v24-B1)
  -> Linear(3072, 512)
  -> 1 token (512-d) per bag
```

이 1-token 표현은 v22의 40-token 표현을 대체하여 `_population_tokens`(direct global/query branch)와 `_class_memories`(context bag → class-memory 압축 입력, §4)에 그대로 흘러갑니다. **`_class_memories`의 label 기반 grouping 로직 자체는 바뀌지 않습니다** — 압축되는 대상이 40 token에서 1 token으로 줄었을 뿐입니다.

구현: `StructuredPopulationMetaClassifier._projected_bag_tokens` (`src/models/baseline.py:2395`), 생성자 분기 (`src/models/baseline.py:1751-1782`).

`architecture_version`은 `project_structured_tokens=true`일 때 무조건 `24`입니다 (`mean_pool_structured_tokens`이면 `23`, 둘 다 false면 `22`; `src/models/baseline.py:3386`). bottleneck/residual 유무는 `architecture_version`에 반영되지 않고 **state_dict 형태**로만 구분됩니다 — v24-A0/B0/B1 체크포인트를 서로 다른 config로 로드하면 shape mismatch로 실패합니다. 확정된 production config는 `configs/train_v24_medium_bag_proj_residual.yaml` (`projection_bottleneck_dim: 64` + `projection_residual_mean: true`) 하나뿐입니다.

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
Total Trainable Parameters : 9,450,000 (약 9.45M; v22 6.57M + residual bottleneck projection ~2.88M)
Token Dimension           : 512
Aggregator Output Tokens  : 1 token per bag (40 structured token을 §3.5 residual+bottleneck projection으로 압축)
Context Retrieval         : none (v22에서 제거, v24도 유지)
Meta Hidden Dimension     : 256
Attention Heads           : 8
Set Layers                : 1
Ridge Dimension           : 64
Precision                 : bf16-mixed
Architecture Version      : 24
```

v22 파라미터 수(6,566,811, 2026-07-29 실측)는 `configs/train_v22_medium.yaml` 기준이며 현재는 폐기된 참고값입니다. v24 파라미터 수(9.45M)는 [`current_status.md`](current_status.md) §3 v24-B1 완료 기록의 실측값입니다.

---

## 6. Source of Truth 파일

- Backbone & Version: `src/models/baseline.py` (`architecture_version = 24`)
- Data Interface & Collators: `src/modules/data_interface.py`
- Multi-task Loss & Metrics: `src/modules/model_interface.py`
- Resolved Production Entry: `configs/train_v24_medium_bag_proj_residual.yaml`
- Architecture Verification Suites: `tests/test_base_model.py`, `tests/test_batched_episode_forward.py`, `tests/test_model_interface.py::test_bottleneck_projection_with_residual_mean`
- 폐기된 v22 production entry(참조용): `configs/train_v22_medium.yaml`, `configs/train_v22_hard_realworld.yaml` 등 (ICI 파이프라인이 아직 참조 — §7 미완 항목)
