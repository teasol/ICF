# v35 Proposal: Token-Only Query Path + Chunked Bag Aggregation

## Rare branch 제거와 context·query 공통 chunk 처리를 통한 대형 bag 학습

**작성일**: 2026-08-07

**상태**: 제안 — 구현 전 (설계 확정 대기)

**기준선**: v34 Phase 0 (`train_v34_phase0_largectx_1536.yaml`, PathoBench reporting model)
- 모델: v22 architecture, `input_dim=1536`, `poolz_l2` bag view, MLA slot affinity, subspace covariance relation (`learned_head`, rank 1)
- 커밋 `5869535`: query-count-invariant margin (`tanh(margin)`) — query batch 정규화 제거 완료
- 데이터: `num_bags [60,100]`, `num_cells [1,8192]` log-uniform, `targets [5,12]`, 1536-d
- 최고 ckpt: `20260806_215800/.../epoch=048-val_ce_loss=0.4419.ckpt`

**예상 수정 범위**:
- [`src/models/baseline.py`](../../src/models/baseline.py) — rare branch 제거, `_fuse_evidence` 단순화, chunk token 집계
- [`src/datasets/synthetic_data.py`](../../src/datasets/synthetic_data.py) — log-power, context/query 별 cell 범위, chunk-as-pseudo-bag
- [`configs/train_v35_*.yaml`](../../configs/), [`configs/data/default.yaml`](../../configs/data/default.yaml), [`configs/model/default.yaml`](../../configs/model/default.yaml)
- [`scripts/test_pathobench.py`](../../scripts/test_pathobench.py) — eval chunk 일관성
- 테스트 (`tests/`) — 32개 기존 테스트 갱신 + 신규 집계/결정성 테스트

---

## 0. 제안 요약

v34에서 배운 두 가지 사실이 이 제안의 출발점이다.

1. **분류 헤드는 가변 길이 query sequence를 보지 않는다.** 모든 bag(context/query)은 aggregator에서 고정 크기 token set(`global_summary` 1 + `slots` 12 + `tails` + covariance sketch)으로 압축되고, meta-classifier는 그 token만 class memory와 비교한다. **query cell 수가 분류기에 들어가는 유일한 경로는 `_rare_instance_logits`(raw cell) 하나뿐**이다.
2. **context cell 상한 8192는 VRAM 가드일 뿐 구조적 제약이 아니다.** 실제 PathoBench slide는 context·query 모두 15k~30k이므로, 현재 훈련은 **context와 query 둘 다** 평가와 불일치한다.

따라서 v35는:

- **결정 1**: raw cell을 소비하는 유일한 branch인 `_rare_instance_logits`를 제거 → query 경로를 100% 고정-token화.
- **결정 2**: 모든 bag(초과 크기)을 **chunk(≤2048 cell)로 분할해 aggregator를 돌린 뒤, chunk token을 원래 bag 단위로 집계**하는 공통 경로 도입 → context·query 모두 실제 slide 크기까지 훈련 가능.
- **데이터**: log-uniform 약화(power)로 대형 bag 밀도 증가, context `[1, 30000]` / query `[3000, 50000]` 분리.

핵심 가설:

> raw-cell branch가 없고 bag→고정 token 경로만 남으면, chunk 단위 token 집계가 bag 크기와 무관하게 성립하고, context·query 모두 실제 slide 크기로 훈련할 수 있어 train/eval 불일치가 사라진다.

---

## 1. 현재 구조에서 query token 수가 실제로 중요한 지점

### 1.1 bag → 고정 token 경로 (모든 bag 공통)

[`StructuredEpisodePopulationAggregator`](../../src/models/baseline.py) `forward`(~1623) / `_forward_dense`(~1242)가 bag 하나를 아래 고정 크기 representation으로 압축한다.

| representation key | 크기 | 용도 |
|---|---|---|
| `global_summary` | 1 token | centered spread, global-shape branch |
| `slots` | `num_slots`=12 | population memory / abundance ridge |
| `tails` | `len(tail_fractions)` token | `_all_structured_tokens` 구성요소 |
| `slot_metadata` | 고정 | abundance ridge |
| `covariance_sketch` | 64 | covariance ridge |
| `slot_covariance_sketch` / `reliability` | 고정 | slot covariance relation |
| `covariance_matrix` | `[1,1]` or projected | subspace covariance relation (v34 `emit_covariance_matrix` 경유) |
| `cls_token` | 1 | v34 off |

→ **cell 수와 무관하게 token 수 불변.** cell 수는 ① per-cell 연산량, ② mean/cov/slot 추정 품질에만 영향.

### 1.2 meta-classifier가 query에 접근하는 경로

[`StructuredPopulationMetaClassifier`](../../src/models/baseline.py) `forward`(~4163) / `forward_batched`(~3951):

| branch | query 입력 | 크기 의존 |
|---|---|---|
| `global_shape_classifier` | projected structured tokens | 고정 |
| `_population_memory_logits` (~3402) | `_population_tokens` (slots 등) | 고정 |
| `_abundance_ridge_logits` (~3567) | `slot_metadata` / `covariance_sketch` | 고정 |
| `_covariance_relation_scores` (~2894) | `covariance_matrix` (subspace) | 고정 |
| **`_rare_instance_logits` (~3438)** | **`query_instances` (raw cell)** | **O(N)** ← 제거 |

`use_instance_attention_mil`은 v34 off → raw cell 소비자는 `_rare_instance_logits`뿐이다.

### 1.3 "query token 수가 중요하다"는 전제의 실체

- **compute/memory**: aggregator의 per-cell 패스(`_bag_view`, slot assignment, `_population_candidates`, rare encoder) — O(N)
- **추정 품질**: mean/cov/slot이 cell이 많을수록 정확 — 통계적 효과
- **rare/tail 통계**: `topk(fraction)` 극단값이 N에 의존 — 구조적

rare 제거 시 **1.3의 3번째가 사라지고** 1·2만 남는데, 1·2는 chunk로 해결된다.

---

## 2. 결정 1 — Rare-instance branch 제거

### 2.1 제거 대상 (정확한 지점)

| 항목 | 위치 | 내용 |
|---|---|---|
| 호출부 | `forward` ~4090, `forward_batched` ~4090-4095 | `tail_logits = self._rare_instance_logits(...)` 제거 |
| fusion | `_fuse_evidence` ~3527 | 3-증거(global/population/rare) → 2-증거(global/population)로 단순화 |
| 파라미터 | `__init__` ~2784-2815 | `instance_input_norm`, `instance_input_projection`, `rare_evidence_head`, `rare_similarity_log_scale`, `tail_residual_logit`, `minimum_tail_residual_scale`, `rare_evidence_fractions` (~2476) |
| 인자 | `forward`/`forward_batched` 시그니처 | `query_instances`, `query_cell_mask` 제거 |
| aux | `return_auxiliary` dict | `tail_logits`, `tail_weights`, `rare_fraction_scores`, `rare_counts`, `tail_residual_scale` 제거 |
| batched twin | `_rare_instance_logits_batched` ~3792 | 함께 제거 |
| config | `configs/model/default.yaml` | `meta_rare_evidence_fractions`, `meta_tail_residual_scale`, `meta_minimum_tail_residual_scale` 제거 |

### 2.2 유지 (혼동 방지)

- aggregator의 **`tails` 구조적 token** (고정 크기 per-bag 요약, `_all_structured_tokens`의 일부) — **유지**. raw cell을 소비하는 게 아니라 per-bag 고정 요약이라 고정-token 설계와 정합.
- `aggregator_slot_rare_fraction` (slot anchoring) — 유지.
- `aggregator_tail_fractions` — 유지 (tails token 생성용).

### 2.3 효과

- query 경로 100% 고정-token → 결정 2의 chunk token 집계가 구조적으로 정당.
- 50k query에서 per-cell 인스턴스 인코더(`instance_input_projection` 등) 연산 제거 → 큰 연산 절감.
- `BaseModel.forward`(~4751)의 query용 중복 `_bag_view` 계산 제거 가능.

### 2.4 리스크

- tail evidence(잔여 0.10 + fusion 상호작용) 제거 → global/population/covariance branch가 부담. 구조 변경이므로 **전체 재평가 필요**.
- **Ablation knob 고려**: `meta_enable_rare_evidence` config(default false)로 두어 tail branch를 비활성화만 하고 파라미터는 유지 → 성능 회귀 시 즉시 원복 가능.
- public 시그니처 변경 → 호출부(테스트, `model_interface.py`, eval 스크립트) 갱신.

---

## 3. 결정 2 — Chunk-as-pseudo-bag + Token 집계 (context/query 공통)

### 3.1 개념

```
원본 bag (최대 50k cells)
  → 데이터 경계에서 분할: ceil(n / CHUNK_CELLS)개 pseudo-bag (≤2048, 결정적 순차)
  → 각 pseudo-bag에 aggregator → 고정 token set (dense batched, padding ≤2048)
  → 원본 bag 단위로 chunk token 집계 → 단일 token set
  → meta-classifier 1회 (기존 그대로)
```

- **chunk 분할은 데이터 경계(dataset)에서 수행** — 배치 padding이 항상 `CHUNK_CELLS`로 상한되어 VRAM 제어.
- **token 집계는 모델 경계(aggregator 후)에서 수행** — meta-classifier가 보는 것은 원본 bag 단위 token.
- **context·query 동일 경로** — 별도 query 전용 코드 없음.

### 3.2 집계 방식 (token별)

| token | 집계 | 비고 |
|---|---|---|
| `global_summary` | **count 가중 평균** | 1차 모멘트 정확 |
| `covariance_matrix` | **count 가중 평균 + between-chunk 보정** `Σ_c n_c(Σ_c + δ_cδ_cᵀ)/N`, `δ_c = μ_c − μ_pooled` | **원본 bag 공분산과 수학적으로 동일** (subspace branch에 정확) |
| `covariance_sketch`, `slot_covariance_sketch` | count 가중 평균 | 근사, 분산 감소 |
| `slot_covariance_reliability` | count 가중 평균 | scale 유지 |
| `slots`, `tails` | slot 정렬 평균 | soft ensemble (추후 attention 확장 가능) |
| `cls_token` (v34 off) | count 가중 평균 | — |

### 3.3 anchor / pool stats 처리

- `_context_pool_stats`(~726): 전체 context cell의 streaming mean/std — **이미 합계 reduction이라 chunk와 무관하게 안전**.
- `_context_anchors`(~1047): context pseudo-bag들의 `_population_candidates`(~977, 방향 soft pooling)의 **합집합**에서 기존처럼 선택. pseudo-bag 수가 늘어도 후보 풀이 유지되므로 안전.

### 3.4 결정성

- chunk 분할: 기본 **순차**(재현 가능). 훈련 augmentation용 `chunk_shuffle`은 **시드 고정 randperm** (epoch/batch 시드로 재현).
- eval(`test_pathobench.py`)도 동일한 chunk 경로를 타야 **train/eval 일관성** 유지 (slide 15k~30k도 chunk됨).

### 3.5 훈련 cardinality 혼합

- 평가는 큰 slide를 항상 chunk하므로, 훈련도 **일부 에피소드는 원본 소형 bag(≤CHUNK_CELLS) 직접 처리, 일부는 대형 bag을 chunk 처리**하는 혼합으로 학습 → chunk 집계를 모델이 학습.
- 실제로는 bag size 분포(log-power + role별 범위)가 이미 자연히 혼합을 만든다.

---

## 4. 데이터 재설계

### 4.1 `src/datasets/synthetic_data.py` 변경

| 항목 | 현재 | v35 |
|---|---|---|
| log-uniform 약화 | `fraction = rand()` (power=1) | **`num_cells_log_uniform_power`** 추가: `fraction = rand() ** (1/power)` (power 1.5~2.0) |
| context cell 범위 | `num_cells [1, 8192]` | `context_num_cells [1, 30000]` |
| query cell 범위 | (동일) | **`query_num_cells [3000, 50000]`** |
| chunk | 없음 | **`chunk_cells 2048`** (pseudo-bag 분할) |

- `sample_num_cells`(~755): power 인자 추가.
- `sample_episode`(~306): `num_cells_per_bag`(line ~337-340)를 **context/query 별 범위로 분리** — dataset이 query 개수(`training_query_range`, ~1062)를 이미 알므로, query 위치에 `query_num_cells`, 나머지에 `context_num_cells` 적용.
- **query 위치 정렬 필요**: dataset이 지정한 query bag 위치와 `model_interface._sample_training_queries`의 mask_index가 일치해야 함 (구현 시 검증 항목).
- `query_num_cells` 하한을 context보다 높게 (query ≥ context 유지) — 근거 §5.

### 4.2 config

`configs/data/default.yaml` (v35 값):

```yaml
num_bags: [40, 80]          # 총 셀수 균형 (대형 bag 증가 대응)
context_num_cells: [1, 30000]
query_num_cells: [3000, 50000]
num_cells_log_uniform_power: 1.5
chunk_cells: 2048
episode_batch_size: 2        # 총 셀수 예산으로 2~4 결정
```

### 4.3 연산/VRAM 예산 (B200, fp32, 1536-d)

- chunk(2048) tensor: `2048×1536×4 = 12.6 MB`.
- **pseudo-bag 배치 padding ≤ 2048** → peak VRAM이 bag 크기와 무관, **chunk sub-batch(64~128)로 상한 제어**.
- 전형적 에피소드: context 50 bag 평균 ~6k = 300k cells + query 8 bag 평균 ~25k = 200k → ~500k cells/episode.
- `episode_batch_size 2` → ~1M cells/batch (현재 worst case 3.3M보다 낮음). 시간은 총 셀수에 비례(불가피).

---

## 5. query ≥ context 비대칭 유지 근거

- **context는 40~80개 bag** → class memory/prototype이 여러 bag에 평균되어 **per-bag 추정 오차가 상쇄**.
- **query는 단일 bag 하나가 결정적 증거** → per-bag 추정 품질이 더 중요.
- 따라서 **둘 다 실제 slide 크기(15k~30k)까지 키우되, query 하한을 context보다 높게** 유지.

---

## 6. 구현 계획 (순서)

| 단계 | 내용 | 검증 |
|---|---|---|
| 1 | rare branch 제거 (`baseline.py`, config, aux, 시그니처) | 32 tests 갱신·통과, smoke forward |
| 2 | dataset: log-power + context/query cell 범위 + query 위치 정렬 | generator 분포 단위 테스트 |
| 3 | dataset: chunk-as-pseudo-bag (≤2048, 결정적) + chunk group metadata | padding ≤ 2048 확인 |
| 4 | model: chunk token 집계 (원본 bag 단위) — `forward`/`forward_episode_batch` | ① 1-chunk direct == 집계 결과, ② 결정성 |
| 5 | `test_pathobench.py` eval chunk 일관성 | 공식 50-fold 결정성 재현 |
| 6 | `train_v35_*.yaml` (100 epochs, 2-GPU DDP, LR 스케줄 결정) | smoke + val 곡선 |

## 7. 검증 계획

1. **단위**: ① chunk 집계 불변성 (1-chunk direct == aggregated), ② 결정성(같은 입력 → 같은 logit), ③ rare 제거(aux 키 부재, 시그니처 변경), ④ dataset 분포(대형 bag 밀도, query ≥ context).
2. **Smoke**: 50k query 에피소드 1~2개 → peak VRAM / step time.
3. **평가**: 공식 50-fold rerun으로 **v34(fixed) vs v35** 비교. (case-leak 없음 — slide-level 분할 유지)
4. **LR 스케줄러**: 이전 곡선(val_ce best 0.4419 @ ep48, ~ep24 수렴) 기반 — 옵션 A: patience 10→5, cooldown 5→3 / 옵션 B: cosine. 별도 결정.

## 8. 오픈 문제

- `meta_enable_rare_evidence` ablation knob을 만들지 여부 (§2.4).
- slot/tail 집계: 평균 vs attention — 평균으로 시작, 성능 보고 확장.
- context도 chunk 집계로 원본 bag 단위 token을 만들지, 아니면 pseudo-bag을 그대로 "bag"으로 쓸지 — **원본 bag 단위 권장**(meta-classifier의 bag=slide 의미 유지).
- `num_bags`/`episode_batch_size` 최종값은 smoke 예산으로 결정.

## 9. 참고

- model fix 커밋 `5869535` (tanh margin, query-count-invariant) — v35가 전제하는 multi-query/chunk 동일성의 기반.
- §57 case-leakage 진단(`fc5e90e`) — 공식 50-fold 기준.
- `scripts/test_pathobench.py` `--batch-queries`(실험적, 기본 off) — chunk batching과 연계 가능.
