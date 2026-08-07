# v35 Proposal (rev.2): Exact Streaming Bag Aggregation

## 원안(rev.1) 비판적 재검토 결과 — "rare 제거 + chunk token 평균"에서 "정확 스트리밍 축약"으로 전환

**작성일**: 2026-08-07 (rev.1) / **개정**: 2026-08-07 (rev.2, 코드 검증 기반 재설계)

**상태**: 제안 — **원안 rev.1의 3개 결정 중 2개는 폐기 권고**, 1개는 무손실 형태로 재설계. P0 게이트 통과 전 구현 금지.

**기준선**: v34 Phase 0 (`train_v34_phase0_largectx_1536.yaml`, PathoBench reporting model)
- 모델: v22 architecture, `input_dim=1536`, `poolz_l2` bag view, MLA slot affinity, subspace covariance relation (`learned_head`, rank 1)
- 커밋 `5869535`: query-count-invariant margin — **재검증 완료(본 문서 §1.1)**
- 데이터: `num_bags [60,100]`, `num_cells [1,8192]` log-uniform(에피소드 단위 1회 추첨), `targets [5,12]`
- 최고 ckpt: `20260806_215800/.../epoch=048-val_ce_loss=0.4419.ckpt`

---

## 0. 개정 요약 — 무엇이 틀렸고 무엇이 바뀌었나

| 원안 rev.1 | 판정 | rev.2 |
|---|---|---|
| **결정 1**: rare-instance branch 제거로 query 경로 100% 고정-token화 | ⛔ **폐기 권고** | 불필요(§3의 정확 축약은 rare branch가 있어도 성립) + 위험(ABMIL-유사 선택 기제를 유일하게 담당) + 근거 미측정. **무료 eval ablation으로 기여도부터 측정**(§4.2) |
| **결정 2**: chunk token을 count 가중 **평균**으로 집계 | ⚠️ **재설계** | 완성된 token을 평균하면 비선형 통계가 깨짐(§2.1~2.3). → **충분통계(sufficient statistics) 누적 + online softmax + top-k merge로 수치 동일(exact) 축약**(§3) |
| **데이터**: `query_num_cells [3000,50000]`, context/query 분리 | ⛔ **폐기 권고** | ① Musk 소형 bag 학습이 사라져 확정 목표(Musk 0.95)와 충돌, ② 폐기된 B2b(per-bag cardinality) 재도입, ③ query 위치를 dataset이 알 수 없어 **구현 불가**, ④ 에피소드 예산 ~5–7× 붕괴 (§2.4~2.7) |
| 검증: "1-chunk direct == 집계" | ⚠️ 약함 | 실제 30k 슬라이드에서 **chunked == unchunked (‖Δ‖∞ < 1e-4)**, v34 ckpt 그대로 (§7) |
| 동기: "train/eval bag 크기 불일치" | ⚠️ **직접 반증 있음** | context 2000 tile cap → pooled AUROC **−0.0019** (LUAD EGFR, §2.8). 크기 확대의 기대이익이 낮음 → **P0 게이트 필수**(§4) |

**rev.2의 한 줄 요약**:

> chunk 축약을 **근사(평균)가 아니라 정확(충분통계)**으로 만들면, ① v34 checkpoint가 그대로 유효하고 ② rare branch를 지울 이유가 없어지며 ③ eval OOM·병렬성 문제가 즉시 해결되고 ④ "훈련 크기를 키운다"는 고위험 결정은 무료 P0 실험 뒤로 미룰 수 있다.

---

## 1. 코드 검증: 원안의 주장별 판정

`src/models/baseline.py`를 직접 읽어 확인한 결과다. 줄 번호는 현재 HEAD(`0487b6d`) 기준.

### 1.1 ✅ 사실로 확인된 주장

| 주장 | 확인 |
|---|---|
| raw cell을 소비하는 branch는 `_rare_instance_logits`뿐 | ✅ `_rare_instance_logits`(3438)만 `query_instances` 사용. `use_instance_attention_mil` off. 다른 branch(`global_shape`, `_population_memory_logits` 3402, `_abundance_ridge_logits` 3567, `_covariance_subspace_features` 3159)는 모두 고정 크기 token만 소비 |
| aggregator token 수는 cell 수와 무관 | ✅ `_validate_representation`(3228)의 9개 key 모두 고정 shape |
| `_context_pool_stats`는 chunk에 안전 | ✅ (713) context bag들의 **모든 cell을 concat**한 뒤 mean/std — bag 경계와 무관. chunk로 쪼개도 union이 같으면 값이 동일. **단, 모든 chunk가 통계에 참여해야 함** → chunk sub-batching 시 별도 선행 pass 필요(§3.3) |
| 커밋 `5869535`가 query 결합을 제거 | ✅ **본 세션에서 재검증**: 동일 context에 무관한 query를 함께 배칭 → q1 logit 변화 `3.8e-6`(fp noise), q2 **내용**만 바꿨을 때 `0.0`. 수정 전 같은 테스트는 `0.113 logit / 0.050 prob` 변화였음. `_covariance_relation_scores`(3015-3023)·`_slot_covariance_relation_scores`(3123-3148)가 `tanh(margin)`으로 per-query 독립 |

→ 원안 §1·§2.1의 **구조 분석 자체는 정확**하다. 문제는 그 분석에서 도출한 결론이다.

### 1.2 ⛔ 반증된 주장

| 주장 | 반증 |
|---|---|
| §3.2 `global_summary` → "count 가중 평균 / **1차 모멘트 정확**" | **틀림.** `global_summary`는 1차 모멘트가 아니다. config `global_summary: centered_spread` + `_bag_view`(789): `global_spread = sqrt(mean((x−μ_bag)²) + 1e-6)` — **bag 자기 평균에 대한 2차 중심모멘트의 sqrt(=표준편차)**. chunk별 std의 count 가중 평균은 pooled std와 다르다(between-chunk 항 누락 + sqrt는 비선형) |
| §3.2 `covariance_matrix` → 보정식으로 "**수학적으로 동일**" | **구현 불가.** 보정식 `δ_c = μ_c − μ_pooled`가 **chunk 평균 μ_c**를 요구하는데, `poolz_l2` 경로에서 `_bag_view`는 `bag_mean`을 **반환하지 않고 버린다**(826-830: `standardized, global_spread, centered_delta`만 반환). representation dict에도 chunk 평균·cell 수가 없다. 즉 원안의 유일한 "정확" 항목조차 representation 확장 없이는 계산할 수 없다 |
| §3.2 `covariance_sketch` → "근사, **분산 감소**" | **편향이지 분산 감소가 아니다.** config `aggregator_covariance_mode: correlation` → `_covariance_sketch`(850)는 2차 모멘트를 **대각으로 정규화(rsqrt)** 한 correlation을 낸다. `log_correlation` 모드는 추가로 **행렬 로그(eigh)**. chunk별 correlation의 평균 ≠ pooled correlation (비affine 변환) |
| §3.2 `slots`/`tails` → "slot 정렬 **평균**" | ① slot은 **에피소드 전역 anchor**로 정의되므로 "정렬"은 애초에 자동이다(오해). ② 더 중요한 건 가중치: `slot_mean = (aᵀx)/mass`, `dispersion = Σa(1−sim)/mass`는 **slot mass 가중** 통계다. cell count 가중(또는 단순) 평균은 slot이 chunk 간 불균등 분포일 때 편향된다. ③ `slot_metadata`는 §3.2 표에 **아예 빠져 있고**, 그 내용은 `(log proportion, dispersion)`이라 log를 평균하면 Jensen 편향이 붙는다 |
| §3.3 `_context_anchors`는 "안전" | **가장 심각한 오류.** `_population_candidates`(977)는 bag 크기와 무관하게 **bag당 정확히 `context_samples_per_bag=32`개** 후보를 낸다(soft direction pooling). 따라서 50k bag을 25 chunk로 쪼개면 그 bag의 후보가 32 → **800개(25×)**가 된다. anchor는 이 후보 풀 위에서 k-means 유사 refinement(1106-1116)와 argmax 기반 farthest-point 선택(1128-1136)으로 정해지므로 **풀 구성 변화에 직접 반응**한다. 즉 "대형 bag이 anchor를 지배"하게 되고, anchor는 에피소드 전역이라 **모든 bag의 모든 token이 바뀐다**. "후보 풀이 유지되므로 안전"은 사실과 반대다 |
| §4.1 query 위치 정렬은 "구현 시 검증 항목" | **설계 모순.** query 위치는 dataset이 정하지 않는다. `src/modules/model_interface.py` `_sample_training_queries`(440)가 **훈련 스텝에서 무작위로** 뽑고(535: `torch.randint`), 개수도 `training_targets_per_episode [5,12]`에서 매번 달라지며(500-536), 클래스당 1개는 query 금지로 보호한다(541-556). dataset은 어느 bag이 query가 될지 **알 수 없다** → `query_num_cells`를 "query 위치에 적용"하는 것은 불가능. §5의 query≥context 비대칭 전체가 이 위에 서 있다 |

### 1.3 원안이 놓친 구조적 사실 (rev.2의 출발점)

`assignment = softmax(similarity / τ, dim=-1)`(1753) — softmax가 **slot 축**이다. 즉 **각 cell의 slot 할당은 다른 cell과 독립**이다. 따라서 slot 관련 모든 통계는 cell에 대한 **순수 합(sum)**이다:

```
mass_s = Σ_n a_ns            slot_mean_s = (Σ_n a_ns x_n) / mass_s
second_s = (Σ_n a_ns x_n²)/mass_s    dispersion_s = (Σ_n a_ns (1−sim_ns)) / mass_s
```

`_population_candidates`(998)의 `weights = softmax(scores·10, dim=0)`은 **cell 축 softmax 가중 평균**이므로 online-softmax(running max/denominator/numerator)로 **정확히** 스트리밍된다(Flash-Attention과 동일한 축약).

`tails`(1841-1861)·`rare_state`(1809-1816)의 `topk`는 **분산 top-k merge**로 정확하다(각 chunk가 자기 top-K를 내고 합집합에서 다시 top-K).

→ **aggregator의 모든 token은 cell에 대한 합·softmax가중평균·top-k의 조합이며, 세 연산 모두 정확히 chunk 축약 가능하다.** 근사할 필요가 전혀 없다.

---

## 2. 치명적 결함 (원안 채택 시 발생하는 것)

### 2.1 결함 A — 완성 token 평균은 비선형 통계를 깨뜨린다
`global_spread`(sqrt), `covariance_sketch`(rsqrt 정규화, 선택적 행렬 로그), `slot_std`(sqrt), `slot_covariance`(대각 정규화 + `diagonal.log()`), `slot_metadata`(log proportion) — 전부 비선형. chunk 평균은 **편향 추정자**이며 편향의 크기는 chunk 수(=bag 크기)에 의존한다. 즉 원안은 "크기 불일치를 없앤다"면서 **크기 의존 편향을 새로 도입**한다.

### 2.2 결함 B — bag 전역 정규화자가 chunk 지역값으로 대체된다
`_slot_covariance_sketch`(908)는 `bag_rms`(bag 전체 centered_delta의 RMS)로 나눈 뒤 통계를 낸다. chunk별로 계산하면 각 chunk가 **서로 다른 스케일**로 정규화된 descriptor를 내고, 그것을 평균하면 스케일이 섞인다. `centered_delta` 자체도 `bag_mean` 기준이라 같은 문제를 갖는다. → **정확 축약은 전역 정규화자를 먼저 확정하는 다단 pass가 필수**(§3.3). 원안의 1-pass + 평균 구조로는 원리적으로 해결되지 않는다.

### 2.3 결함 C — anchor 오염 (§1.2) → 에피소드 전역 회귀
후보 풀이 대형 bag 쪽으로 25× 기울면 anchor·slot 정의가 바뀌고, 이는 **context/query 모든 bag의 모든 token**에 전파된다. v34 checkpoint는 이 anchor 분포에서 학습된 것이라 checkpoint 재사용이 무의미해지고, 회귀가 나도 "chunk 때문인지 anchor 때문인지" 분리할 수 없다.

### 2.4 결함 D — `query_num_cells [3000, 50000]`은 확정된 사용자 목표를 깬다
`sample_num_cells`(755) docstring이 log-uniform을 쓰는 이유를 명시한다: *"real bags are heavy-tailed (Musk2 spans 1..1044 with a median of 12), so a plain uniform draw would put half its mass above 512 and the model would still almost never meet a small bag."*
하한 3000은 **소형 bag 학습을 완전히 제거**한다. Musk zero-shot(현 0.854)은 확정 baseline 지표이고 **"Musk 목표는 0.95 유지"는 2026-08-05 확정된 사용자 결정**이다. §42는 n≤4 구간(0.800→0.725)까지 추적하고 있다. 원안 데이터는 이 지표를 구조적으로 파괴한다.

### 2.5 결함 E — 폐기된 B2b를 이름만 바꿔 재도입
context/query별 다른 cell 수는 **bag마다 다른 cardinality**를 요구한다. 현재 기본 경로는 에피소드당 `num_cells`를 **1회** 추첨하고(`per_bag_cardinality: false`), per-bag 추첨은 v33 **arm C(B2b)** 로 이미 실험되어 **폐기**되었다: legacy overall `0.8100 [0.798, 0.822]` vs v30 `0.8512` → **회귀 0.0412, gate 미달**, 결론은 *"과소학습 편향 가설 기각, B2b 데이터 자체가 회귀 원인"*(current_status.md §4·§42). 원안은 이 결론을 언급조차 하지 않는다.

### 2.6 결함 F — query 크기 비대칭은 구현 불가 + 근거도 반대
구현 불가는 §1.2 마지막 항. 근거도 약하다: 실제 eval에서 query 슬라이드와 context 슬라이드는 **같은 분포**다(본 세션 실측: h5 400개 표본, median 7,511 / IQR [2,949, 15,039] / max 38,560 tiles). 훈련에서만 query를 크게 만들면 **반대 방향의 새 불일치**를 만든다.

### 2.7 결함 G — 에피소드 예산 붕괴가 정량화되지 않음
비교는 **기대 cell 수/에피소드**로 해야 한다.
- v34: log-uniform `[1,8192]`, `E[n] = (B−1)/ln B = 8191/9.011 ≈ 909` cells/bag → 80 bag ≈ **73k cells/episode**
- 원안 §4.3 자체 추정: **~500k cells/episode** → **약 6.8×**

원안 §4.3의 "현재 worst case 3.3M보다 낮음"은 *v35 전형값* vs *v34 최악값* 비교로, 사과-오렌지다. 동일 wall-clock에서 에피소드 예산은 **51,200 → ~7,500 (약 1/7)** 로 줄어든다. 그리고 arm C의 교훈이 정확히 "에피소드 수 매칭 없이는 판정 불가"(§42-43, 8×A6000로 재실행한 이유)였다. 원안은 이 예산 문제를 "시간은 총 셀수에 비례(불가피)" 한 줄로 처리한다.

### 2.8 결함 H — 동기(크기 불일치)에 직접 반증이 존재
본 세션 실측, 동일 v34 ckpt·공식 50-fold·LUAD EGFR:

| 설정 | fold-mean | pooled |
|---|---:|---:|
| full-context (전체 타일) | 0.7769 ± 0.0891 | **0.7714** |
| **context 2,000 tile cap** | 0.7744 ± 0.0883 | **0.7695** |

**Δpooled = −0.0019.** context cell을 15배 이상 버려도 성능이 사실상 불변이다. 게다가 이론적으로도 이 방향이 맞다: aggregator token은 전부 표본통계(평균·공분산·softmax가중평균·분위수)이므로 **표본 크기에 대해 일치추정량(consistent)** 이다 — bag이 커지면 기대값이 아니라 **분산만** 줄어든다. 따라서 "2k로 훈련 / 30k로 평가"는 원안이 말하는 구조적 불일치가 아니라 **같은 추정자의 표본 크기 차이**다.

→ 크기 확대의 기대이익은 낮다. 최소 **P0 게이트**(§4) 없이 착수하면 v31·v32·v33에 이어 네 번째 gate 미달 반복이 된다.

### 2.9 결함 I — 2인자 동시 변경 (교란)
원안은 rare 제거 + chunk + 데이터 재설계를 한 번에 하고 v34와 비교한다. 회귀 시 원인 분리가 불가능하다. §42에서 이미 같은 실수의 비용(과소학습 가설을 기각하기 위한 추가 150-epoch 재실행)을 치렀다.

### 2.10 결함 J — 방향성: ABMIL이 아니라 MeanMIL 쪽으로 간다
chunk token 평균은 **region-level mean pooling**이다. 본 세션 SEAL 비교에서 우리 위치는:

| task | 우리(v34, 공식 50-fold) | SEAL MeanMIL | SEAL ABMIL |
|---|---:|---:|---:|
| LUAD EGFR | 0.771 | 0.777 | 0.830 |
| LUAD STK11 | 0.828 | 0.873 | 0.908 |

우리는 이미 **MeanMIL 수준, ABMIL 미달**이다. 원안은 ① 집계를 명시적 mean pooling으로 바꾸고 ② **유일한 선택적(selective) 기제인 rare/tail branch를 삭제**한다. 즉 ABMIL과 MeanMIL을 가르는 성분을 지우고 평균 쪽으로 이동한다. 정확히 반대 방향이다.

---

## 3. rev.2 재설계 — 정확 스트리밍 축약 (Exact Streaming Aggregation)

### 3.1 원칙

> **완성된 token을 평균하지 말고, 충분통계를 누적한 뒤 비선형 변환을 bag 수준에서 1회 적용한다.**

이는 저장소의 확립된 관행과 정확히 일치한다 — v34의 MLA 저랭크 affinity("None이면 full-dim dot과 **byte-identical**"), slot_std 분산 트릭("default 경로 byte-identical"), 배치 population candidates("**수치 동일**"), §56 폐기분기 제거("forward 동치, diff 0"). rev.1은 이 관행을 이유 없이 깬다.

### 3.2 누적자 (모두 cell에 대한 합 — 정확)

bag `b`, chunk `c`, `P = _covariance_projection`(고정), `v` = `_bag_view` 출력(standardized), `A` = anchors, `a` = assignment.

| 대상 token | 누적자 | bag 수준 마감 | 정확성 |
|---|---|---|---|
| `global_summary` | `n`, `Σx`, `Σx⊙x` | `sqrt(Σx²/n − μ² + 1e-6)` | **exact** |
| `covariance_sketch`, `covariance_matrix` | `Σz`, `Σz zᵀ` (`z = xP`) | `Σ(z−z̄)(z−z̄)ᵀ = Σzzᵀ − (Σz)(Σz)ᵀ/n` → **correlation/log 변환은 여기서 1회** | **exact** (P가 선형이라 `P(x−μ)=Px−Pμ`) |
| `slots` center/spread, `slot_metadata` | slot별 `m_s=Σa_s`, `Σa_s v`, `Σa_s v⊙v`, `Σa_s(1−sim_s)` | `slot_mean=Σa_s v/m_s`, `slot_std=sqrt(Σa_s v²/m_s − slot_mean² +1e-6)`, `proportion=m_s/n`(**log는 마감 후**), `dispersion=Σa_s(1−sim)/m_s` | **exact** (§1.3: assignment는 cell별 독립) |
| `slot_covariance_sketch`/`reliability` | `bag_rms`(pass 1에서 확정) 고정 후 slot별 `Σa_s w`, `Σa_s w wᵀ` | 정규화·`diagonal.log()`를 bag 수준에서 1회 | **exact** |
| anchors용 `_population_candidates` | online softmax: running `max`, `Σexp(·)`, `Σexp(·)·v` | `V/Z` → `F.normalize` | **exact** (cell축 softmax가중평균) |
| `tails` | chunk별 novelty top-`K` (K는 **bag 전체 n** 기준 `ceil(fraction·n)`) | 합집합에서 다시 top-`K` → `shared_tail_encoder(dev).mean()` | **exact** (분산 top-k merge) |
| `rare_state` | slot별 rare_score top-`K` (slot_mean 확정 후) | 합집합 top-`K` → softmax 가중 | **exact** (pass 순서 §3.3) |

`aggregator_min_tail_instances: 1`, `absolute_tail_ks` 미사용(폐기 분기)이라 tail 쪽 예외 처리는 단순하다.

### 3.3 pass 순서 (전역 정규화자 때문에 필수)

```
pass 0  context 전체 cell → pool_mean/pool_std        (_context_pool_stats와 동일 값)
pass 1  bag별 online-softmax 후보 32개 → anchors      (standardized cell 필요 → pass 0 이후)
pass 2  bag별 slot 충분통계 + 공분산 누적 + global_spread 누적 + tail top-K
pass 3  bag별 rare top-K (slot_mean 필요 → pass 2 이후)
```

- **메모리**: peak가 `O(chunk)` — bag 크기와 무관. eval OOM 문제의 근본 해결.
- **연산**: cell을 3~4회 통과 → **eval 연산 3~4×**. 대신 worker 병렬도를 2 → 8~10으로 올릴 수 있다(현재 workers>2가 OOM으로 불가). 순 효과는 **smoke로 측정**할 사항이며 가정하지 않는다.
- **훈련**: autograd에서 다중 pass는 chunk 단위 **gradient checkpointing**(backward에서 chunk 재계산)으로 처리. 활성값 메모리가 `O(chunk)`가 되는 것이 대형 bag 훈련의 실제 enabler다.
- **수치**: 합산 순서가 달라지므로 bit-identical은 아니다. 축약 누적자는 **float64**로 두고 허용오차 `1e-4`(§7)를 요구한다. (`_context_pool_stats`가 `unbiased=False`로 train/eval 0.25% 불일치를 막은 것과 같은 급의 주의.)

### 3.4 이 설계가 자동으로 해결하는 것

| 원안의 문제 | rev.2에서 |
|---|---|
| anchor 오염(§2.3) | 후보를 **원래 bag 단위**로 online-softmax 축약 → 후보 수가 bag당 32 유지 → anchor **불변** |
| 비선형 편향(§2.1), 전역 정규화자(§2.2) | 충분통계 누적 + 마감 1회 → 정의상 소멸 |
| rare branch 제거 필요성(결정 1) | tail/rare top-k도 정확 축약되므로 **삭제할 이유가 없다** → 결정 1 폐기 |
| v34 checkpoint 무효화 | 수치가 같으므로 **기존 ckpt로 즉시 검증·즉시 이득**(eval 전용으로도 배포 가능) |
| 2인자 교란(§2.9) | 1단계가 "수치 무변화" 리팩터링 → 이후 실험의 깨끗한 기준선 |

---

## 4. P0 게이트 — 훈련 코드를 건드리기 전에 (전부 무료: 기존 ckpt, 학습 없음)

`v31/v32/v33`이 모두 사전 게이트로 폐기됐다. v35도 같은 규율을 적용한다.

### 4.1 P0-a: query 크기 민감도 (크기 확대의 기대이익 측정)
`--max-tiles {1000, 2000, 4000, 8000, None}` × context 고정, 공식 50-fold, task 3개(EGFR/STK11 + Histologic_Grade). `--context-max-tiles`로 context를 분리 고정해 query 효과만 본다.
- **판정**: 2k → full에서 pooled AUROC 이득이 **+0.005 미만**이면 (§2.8의 context 결과 −0.0019가 시사하는 바) **"대형 bag 훈련" 프로그램 전체를 폐기**하고 §3만 eval 최적화로 채택.
- 이득이 +0.005 이상인 task가 있으면 그 task 특성(타일 수, 이질성)을 기록해 §5의 분포 설계 근거로 쓴다.

### 4.2 P0-b: rare branch 기여도 (결정 1의 근거)
eval 시 `rare_logits = 0` 강제 플래그를 추가해(파라미터 변경 없음) 완료된 9개 task 재채점.
`meta_tail_residual_scale: 0.10`, `meta_minimum_tail_residual_scale: 0.05` 라 floor가 있어 이 branch는 항상 기여하도록 강제되어 있고, `_fuse_evidence`(3527)의 pair-product/difference 상호작용에도 들어간다.
- **판정**: |Δpooled| 평균 < 0.003 이면 제거가 안전(별도 실험으로 승격 가능). 그보다 크면 **결정 1은 영구 폐기**.
- 비용: 학습 0, 재추론만. 원안 §2.4가 "리스크"로 남긴 것을 숫자로 끝낸다.

### 4.3 P0-c: chunk 경계 민감도
`--context-max-tiles`가 무작위 부표본인 것과 달리, 실제 h5 타일 순서는 대체로 공간 순서다. sequential chunk = **공간적으로 인접한 region**이 된다. 정확 축약(§3)에서는 chunk 경계가 결과에 영향을 주지 않아야 하므로, 이는 §7의 불변성 테스트로 흡수된다. (chunk 경계가 결과를 바꾸면 구현 버그다.)

---

## 5. 데이터 — P0-a 통과 시에만, 그리고 수정된 형태로

원안의 `context_num_cells`/`query_num_cells` 분리는 폐기(§2.4~2.6). 대신:

| 항목 | 현재 | rev.2 |
|---|---|---|
| 범위 | `num_cells [1, 8192]` | `num_cells [1, 32768]` (**하한 1 유지** — Musk 소형 bag 보존) |
| 분포 | log-uniform (power 1) | `num_cells_log_uniform_power: 1.5` — `fraction = U^(1/power)` |
| cardinality | 에피소드당 1회 추첨 | **유지** (per-bag = 폐기된 B2b, §2.5) |
| context/query 분리 | 없음 | **도입하지 않음** (구현 불가 + eval 분포는 대칭, §2.6) |

`power=1.5`, 범위 `[1, 32768]`의 정량 결과 (`ln 32768 = 10.397`):

| 구간 | 확률 | 근거 |
|---|---:|---|
| `n ≤ 34` (Musk 최약 구간) | **19.8%** | `(ln34/10.397)^1.5 = 0.339^1.5` |
| `n ≥ 8192` (현 상한 초과) | **19.3%** | `1 − 0.867^1.5` |
| median | ~700 | `0.5^{2/3}·10.397` 지수 |
| `E[n]` | **4,487** | 대형 bag 확대의 실제 비용 = v34의 `909` 대비 **4.94×** |

→ 소형 bag 학습을 20% 유지하면서 대형 bag 노출을 확보한다. 동일 wall-clock 기준 에피소드 예산은 `51,200 / 4.94 ≈ 10,400`으로 줄어든다. 따라서 **v34와 같은 51,200 episode를 쓰려면 wall-clock을 ~5× 늘려야 하고, wall-clock을 고정하면 episode가 1/5이 된다 — 둘 중 하나를 명시적으로 선택하고 비교는 반드시 episode-matched로 한다**(arm C 교훈, §42-43). 원안 §4.3의 `~500k cells/episode`(v34 대비 6.8×)는 이보다 더 비싸다.

---

## 6. 구현 계획 (수정)

| 단계 | 내용 | 게이트/검증 | 위험 |
|---|---|---|---|
| **1** | §3 정확 스트리밍 축약 (`chunk_cells` 기본 2048, 기본 **off**=기존 경로) | §7-1 수치 동일성, 32 tests | 낮음 (수치 무변화) |
| **2** | eval 경로에 적용 (`test_pathobench.py`, `run_official_folds_parallel.py`) | 완료된 9개 task 재현 (‖Δpooled‖ < 1e-3), workers·wall-clock 실측 | 낮음 |
| **3** | **P0-a / P0-b** 실행 (§4) | 판정 기준 §4.1·§4.2 | 없음 (학습 0) |
| **4** | *P0-a 통과 시에만*: §5 데이터 변경 **단독** arm + episode-matched 학습 | v34 대비 +0.005 (공식 50-fold, task ≥3) | 중 |
| **5** | *P0-b 통과 시에만*: rare 제거를 **단독** arm으로 (config `meta_enable_rare_evidence`) | 동일 게이트 | 중 |
| **6** | (선택, §8) zero-init chunk-attention head | 동일 게이트 | 중 |

**단계 3에서 P0-a가 미달이면 단계 4~5를 실행하지 않고 v35는 "eval 최적화"로 종결한다.** 이것이 rev.2의 핵심 규율이다.

## 7. 검증 계획 (수정)

1. **수치 동일성 (rev.1의 "1-chunk == 집계"를 대체)**: 실제 PathoBench 슬라이드(30k+ 타일) 및 합성 bag에 대해
   `chunked(chunk=2048, 1024, 512) == unchunked` → 9개 representation key 전부 `‖Δ‖∞ < 1e-4`, 최종 logit `< 1e-4`. **v34 ckpt 그대로.** chunk 크기를 바꿔도 결과가 같아야 한다(= chunk 경계 불변성, §4.3).
2. **anchor 불변성**: 동일 에피소드에서 chunk on/off 시 `_context_anchors` 출력이 `‖Δ‖∞ < 1e-5`. (§2.3의 결함이 재발하지 않음을 고정)
3. **메모리/속도**: chunk on에서 peak VRAM이 bag 크기와 무관함(30k vs 8k 슬라이드 비교), fold당 시간·최대 worker 수 실측.
4. **회귀**: 완료된 9개 공식 50-fold task를 chunk 경로로 재실행 → 기존 값 재현.
5. **단위 테스트 신규**: ① online-softmax 축약 == 직접 softmax, ② 분산 top-k merge == 전역 top-k, ③ 충분통계 마감 == 직접 계산(각 token별), ④ float64 누적자 사용 확인.
6. **소형 bag 보존**(단계 4 진입 시): `n ≤ 34` 에피소드 비율이 §5 표와 일치(±2%), Musk zero-shot 재측정.

## 8. 장기 방향 — ABMIL 격차에 대한 올바른 베팅 (§2.10의 대안)

원안이 지우려던 것(선택적 기제)을 오히려 **추가**하는 방향이 데이터와 부합한다.

- chunk token은 sequential 분할 시 **공간적으로 인접한 region 요약**이다(h5 타일 순서). bag당 `N/2048 ≈ 15` 개 → region set이 작다.
- 여기에 **permutation-invariant attention** 을 씌우면 region-level ABMIL의 유사물이 되고, 평균이 버리는 **region 간 이질성**(grade류 task의 신호)이 보존된다.
- 저장소 관행에 맞춰 **zero-init residual**로 넣는다: 초기화 직후 출력이 정확 가중평균(§3)과 동일 → v34 weight-only 초기화와 호환되고 회귀 시 즉시 원복 가능. (v31이 zero-init output head로 "초기 logits 정확히 동일"을 확보한 것과 같은 패턴.)
- 단, 이것도 **단독 arm + 같은 게이트**로만 진행한다. §3(수치 무변화)이 선행되어야 이 실험의 기준선이 깨끗해진다.

## 9. 오픈 문제 (수정)

- `rare_state`의 pass 3를 없애기 위해 chunk별 top-M(M > K)을 캐시하는 절충 — 메모리 vs pass 수. 구현 시 측정.
- 단계 4 진입 시 `num_bags`(현 `[60,100]`)를 줄여 총 cell을 상쇄할지 — arm C가 bag 수 변경도 회귀 원인 후보였으므로 **가급적 고정**.
- 문서 위치: agent_handoff §6.1은 **활성** proposal을 `docs/` 최상위에 두도록 정한다. v35가 활성이면 이 파일은 `docs/architecture_v35_*.md`로 옮기는 것이 규칙에 맞다(현재 `history/`).
- `--batch-queries`(커밋 `5869535`)는 §1.1에서 query 독립성이 확인됐으므로, chunk 경로와 결합하면 fold당 context 인코딩 1회로 추가 이득이 가능하다. 단 공식 보고 전 별도 동일성 검증 필요.

## 10. 참고

- 커밋 `5869535` — query-count-invariant margin. **본 문서 §1.1에서 재검증**(수정 전 0.050 prob 결합 → 수정 후 0.0).
- `fc5e90e` / current_status.md §57 — case leakage 진단, 공식 50-fold가 정직한 기준.
- current_status.md §4·§42-43 — arm C(B2b) 폐기, 회귀 0.0412, episode-matching의 중요성.
- current_status.md §29 — Musk 0.854, 소형 bag 구간 추적. 사용자 확정: Musk 목표 0.95.
- 본 세션 실측 — context 2,000 cap Δpooled −0.0019 (EGFR), 슬라이드 타일 분포 median 7,511 / 48%가 8,192 초과, SEAL 대비 위치(EGFR 0.771 vs ABMIL 0.830 / MeanMIL 0.777).
