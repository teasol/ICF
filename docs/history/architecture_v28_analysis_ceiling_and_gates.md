# v26 / v27 제안서 비판적 분석과, 측정으로 좁혀진 다음 경로

> [!NOTE]
> **Archived 2026-08-02** (`docs/` 최상위는 5개 living 문서만 유지하는 원칙,
> `agent_handoff.md` §6). 이 문서의 모든 실측 데이터·F1~F11 증거 카탈로그·v26/v27
> 비판 요지·경로 A/B 설계·code reference는 `current_status.md` §16으로 그대로
> 옮겨 적었습니다 — 최신 상태는 여기가 아니라 그쪽을 확인할 것. 원문은 provenance
> 보존을 위해 그대로 둡니다.

**작성일**: `2026-08-02`
**작성자**: Claude (코드·실측 기반 재검토)
**대상**: `architecture_v26_proposal.md`(EC-MoE), `architecture_v27_proposal.md`(AC-ICAR)
**상태**: 분석 + 제안 — 검토·토론용 → **§6 사전등록 게이트 E2/E7/A4 실행 완료 (결과는 §6.1)**
**신규 측정**: 학습 없는 감사 3종 (1,000 / 1,000 / 400 episodes) — §4, 사전등록 게이트 3종 (1,000 / 1,000 / 400 episodes) — §6.1

---

## 0. Executive Summary

세 문장 요약:

1. **v26과 v27은 서로 다른 답을 내놓지만 같은 잘못된 변수를 공격합니다.** 둘 다
   "bag 요약 위의 융합/토큰 구성"을 바꾸는 제안이고, `current_status.md` §3 T2-2는
   그 층에 헤드룸이 없음을 이미 측정해 두었습니다. v27은 추가로 실행 불가한
   연산(§3.4 실측)을 요구합니다.
2. **이번 세션 신규 측정은 T2-2의 결론을 5개 task 전체로 확장합니다.** 학습 파라미터
   0개의 closed-form ridge가 v24의 slot별 충분통계 `(logit π, μ, log σ²)`를 받으면
   overall `0.700`, task별로 학습된 9.45M v24와 **±0.02 안에서 일치**합니다.
   즉 v22~v25가 val CE 0.0073 폭에 갇힌 것은 아키텍처 탐색 실패가 아니라
   **"비지도 population slot으로 요약한 bag"이라는 특징 집합의 정보 상한**입니다.
3. **그 상한은 0.70이고, 올바른 분할을 주면 0.93입니다.** 그리고 분할을 관측값·bag
   라벨로 찾는 경로는 이제 **세 번 독립적으로 닫혔습니다**(§4.5). 남은 것은
   privileged supervision 하나이며, 이것이 유일하게 알려진 고수익 경로입니다.

> [!IMPORTANT]
> **이 문서는 신규 아키텍처를 밀지 않습니다.** 초안에서 제안했던 "slot 통계를
> identity-aligned ridge에 연결" (v28-A)은 **본인의 1,000-episode 측정으로 반증됐습니다**
> (§4.3). 40-episode 예비 측정에서 크게 보였던 이득은 표본 잡음이었습니다.
> 남은 제안은 §5의 두 갈래뿐이고, 둘 다 사전 게이트가 붙어 있습니다.

> [!IMPORTANT]
> **2026-08-02 후속: §6 게이트 E2/E7/A4를 전부 실행했습니다 (§6.1).** 결론만 먼저 —
> **E2 FAIL**(oracle gating조차 delta 0.0000, v26/v27의 gating 전제 완전 반증),
> **E7 INCONCLUSIVE**(지도 purity 0.335, 진행/폐기 게이트 사이),
> **A4 약함**(split-context 효과가 정식 400-episode에서는 ctx40 `+0.0035`로 줄어들고
> ctx160에서는 사실상 0). 경로 A/B 둘 다 "확실한 승격" 신호가 아니라 "아키텍처
> 계열 전체의 상한이 낮다"는 가설을 오히려 강화합니다.

---

## 1. 이 저장소의 실측 증거 (두 제안이 통과해야 하는 사실들)

| # | 사실 | 출처 |
|---|---|---|
| F1 | v22(40 token) `0.5946` vs **v24(1 token) `0.5903`** — 압축이 더 좋았다 | §3 |
| F2 | v25(bag-preserving, +4.4M params) Medium 동률, Easy Δ`+0.0033` | §11 |
| F3 | v22~v25 val CE 전부 `0.5903~0.5976` (폭 0.0073) | §11 |
| F4 | state: 모델 `0.6217` = **모델 입력 토큰 probe `0.6210`** | §3 T2-2 |
| F5 | state observable raw mean `0.5478`; **oracle mask `0.8819~0.9013`** | §3 T2-2 |
| F6 | effect scale 통일 시 covariance(0.6594) ≈ composition(0.6488), **state만 전 구간 최하위** | §3 T3-1 |
| F7 | context 40→300에서 v24 AUROC `0.6774→0.8036` (**+0.126**) | §11 |
| F8 | 세포 선택 점수 4종 전부 AUROC ~0.50; slot capture 0.155, fragmentation 0.963 | §3 T1-A/T1-B |
| F9 | 이득 곡선 선형: purity 0.11→1.00에서 covariance 0.517→0.888 | §3 T1-C 1 |
| F10 | bag 라벨 기반 **세포** 선택 purity 0.128 (무작위 0.110) → Tier 1 종료 | §3 T1-C 2 |
| F11 | v25가 작은 context(40)에서 유의하게 우세, 큰 context(300)에서 열세 | §11 |

F4가 결정적입니다. 모델이 실제로 받는 토큰에 ridge probe를 붙인 값이 모델 성능과
같다면, **그 토큰 위에서 융합·routing·attention을 어떻게 바꿔도 개선되지 않습니다.**
v26의 메커니즘 전체와 v27의 routing 부분은 이 측정으로 사전에 상한이 잠겨 있습니다.
그리고 §4.3은 이 결론을 5개 task 전체로 확장합니다.

또한 두 제안 모두 저장소 자체 규칙 **"상한 측정 없이 아키텍처부터 뜯기 금지"**(§6
하지 말 것)를 위반합니다. 두 문서의 공통 전제는 **학습 없이 2시간에 검증 가능**합니다(§6 E2).

---

## 2. v26 (EC-MoE) 비판

### 2.1 동기로 든 수치가 잘못됐다

§2.3은 "v24 Medium per-task AUROC (state ~0.52, composition ~0.70, covariance ~0.55)"를
근거로 씁니다. Medium 1,000-episode 실측은 **state 0.6215 / composition 0.7729 /
covariance 0.6216**입니다. 0.52/0.55는 **Hard tier 값**(state 0.5167, covariance 0.5103)이며
Medium 논증에 섞여 들어왔습니다.

더 중요한 것은 F6입니다. per-task 격차 자체가 생성기 effect scale 아티팩트로 판정된
뒤였고, 저장소는 "effect scale 정규화 없이 task별 AUROC로 우열 판단 금지"를 명문화해
두었습니다. §2.3 전체가 그 금지 사항 위에 서 있습니다.

### 2.2 softmax simplex가 additive residual 구조를 깨뜨린다 (설계 버그)

v24의 실제 융합 (`baseline.py:2766-2773`):

```
logits = global_shape_logits                        # 계수 1.0, 정규화 안 됨
       + population_scale * population_logits       # sigmoid, 상한 0.25
       + rare_scale      * rare_logits              # sigmoid, 상한 0.10
       + fusion_scale    * interaction              # sigmoid, 상한 0.10
```

v26(§3.4)과 v27(§2 step 4)은 이것을 `Σ_k g_k · logit_k`, `g ∈ Δ^K`로 바꿉니다.
simplex 제약 `Σ g = 1`은 **지배 항인 global의 계수를 1.0에서 0.2~0.5로 축소**합니다.
logit 크기가 줄면 CE는 즉시 나빠지고 backbone이 스케일을 재조정할 때까지 학습이
후퇴합니다. 게이트를 쓰려면 simplex가 아니라 **branch별 독립 sigmoid `g_k ∈ [0, c_k]`**
여야 하고 base는 게이팅에서 빼야 합니다.

### 2.3 Top-2 sparse는 이 구조에서 순손실이다

Sparse MoE의 존재 이유는 **연산 절감**입니다. 여기서 gating은 branch logit을 가중하기만
하므로 top-2를 하든 안 하든 **네 branch 전부 계산됩니다**. 절감은 0이고 불연속 gradient만
추가됩니다. v27이 이 점을 지적한 것은 옳습니다.

### 2.4 Load balancing target `1/K`가 잘못된 목표다

`L_balance = K Σ f_k ḡ_k`(Switch Transformer)는 **token 단위 routing**에서 수천 token으로
`f_k`를 추정하는 설정용입니다. 여기서 routing 단위는 **episode**이고
`episode_batch_size=8`이므로 step당 표본 8개로 `f_k`를 추정합니다 — 추정 불가입니다.
게다가 branch는 task와 1:1이 아니므로 **균등 사용이 최적이라는 근거가 없습니다.**

### 2.5 `z_e`는 기존 경로와 정보가 중복된다

`z_e^pool`은 context bag token 평균, `Δ_e`는 class간 차이입니다. v24는 이미 context bag
token을 class별로 묶어 memory token을 만들고(`_class_memories`) 그 위에서
class-balanced ridge를 풉니다(`RidgeResidualMetaClassifier`). `Δ_e`는 그 ridge가 이미
추정하는 1차 판별 방향의 저해상도 버전입니다.

### 2.6 routing 기계는 이미 코드에 있고, v24에서는 no-op다

`model_interface.py:530-550`에 `routing_entropy`와 `_routing_balance_loss`가 이미 구현되어
있고 config는 `routing_sparsity_weight: 0.0`, `routing_balance_weight: 0.0`입니다.
`meta_routing_temperature: 0.5` 소프트 routing도 `_population_memory_logits`
(`baseline.py:2616-2620`)에 존재합니다.

> [!IMPORTANT]
> **v24에서 이 routing은 no-op입니다.** `project_structured_tokens: true`이면
> `_population_tokens`가 bag당 **토큰 1개**를 반환하므로(`baseline.py:2570-2580`),
> `softmax(importance_logits)`는 원소 1개에 대한 softmax = 항상 1.0입니다.
> **v24는 v22에 있던 slot 단위 population 선택 능력을 없앤 버전**이고, 그래도 성능이
> 같았습니다(F1). "population 선택 경로가 애초에 신호를 거의 못 나르고 있었다"는
> 강한 증거이며, 그 경로 위에 더 정교한 routing을 얹는 v26/v27의 전제를 약화시킵니다.

---

## 3. v27 (AC-ICAR) 비판

### 3.1 핵심 전제가 이미 세 번 반증됐다

§0의 3번 한계와 §3 표 첫 행 전체가 "40→1 token 압축이 정보를 파괴한다"에 의존합니다.
그 실험은 이미 세 번 돌았습니다:

| 실험 | 결과 | v27 전제와의 관계 |
|---|---|---|
| v22 40 token vs v24 1 token | `0.5946` vs **`0.5903`** | 압축이 **더 좋았다** |
| v25 typed bag-preserving (+4.4M) | Medium 동률 | 보존해도 이득 없음 |
| Easy tier (신호 강함) | `0.9073` vs `0.9106` | 신호가 커도 안 갈림 |

정보가 파괴되고 있었다면 40 token을 그대로 쓴 v22가 이겼어야 합니다.

### 3.2 제안된 16 token은 v22 40 token의 부분집합에 가깝다

v27의 16 = density anchor 8 + outlier aggregate 4 + covariance basis 4.
v22/v24의 40 = global 1 + 12 slot × (center/spread/rare) + tail 3.
density anchor는 slot center의 8개 버전(12→8 **감소**), outlier aggregate는 tail과 거의
동일, 그리고 **slot spread(분산) 채널 12개가 사라집니다.** covariance basis만 새롭지만
그 정보는 이미 `slot_covariance_sketch`(16-d)와 `covariance_sketch`(64-d)로 계산되고
있습니다. 순효과는 v22보다 정보가 적은 tokenizer입니다.

### 3.3 eigenvector token은 잘 정의된 함수가 아니다

`t_cov,k = √λ_k · v_k`에서 `v_k`는 **부호가 임의**이고(±v_k 둘 다 고유벡터), 고유값이
근접하면 **부분공간 내 회전도 임의**입니다. 이 token은 공분산의 함수가 아니라 LAPACK
구현의 함수이며, 학습 신호에 부호 잡음이 그대로 들어갑니다. 불변량을 쓰려면
`√λ_k v_k v_kᵀ` 같은 부호 불변 형태나 스펙트럼만 써야 합니다 — 코드가 이미 하는 방식
(`_slot_covariance_sketch`의 `spectral` descriptor).

### 3.4 Riemannian branch는 수식·수치·비용 세 층에서 실행 불가

**수식**: `S̄_c = exp(mean_i log S_i)` 직후 `d = ‖log S_q − log S̄_c‖_F`를 계산합니다.
`log(exp(·))`는 항등이므로 `exp`는 **죽은 연산**입니다. 해롭진 않지만 검산되지 않았다는 신호입니다.

**수치 (실측)**: Medium 한 bag(750 cells, 512-d, 중심화·정규화)의 표본 공분산

| 항목 | 값 |
|---|---|
| condition number | `1.10e+02` |
| **인접 고유값 최소 간격** | **`2.97e-07`** |
| v27 shrinkage 0.20 적용 후 condition | `1.51e+01` |
| v27 shrinkage 0.20 적용 후 **인접 간격** | **`2.97e-07` (불변)** |

`eigh` backward는 `1/(λ_i − λ_j)` 항을 포함합니다. 간격 `3e-7`이면 gradient에 `~3e6`
배율이 실리고, 이 저장소가 `bf16-mixed`를 강제하는 이유(공분산 역행렬 fp16 오버플로/NaN,
`agent_handoff.md` §3-4)와 정면충돌합니다. **shrinkage는 condition number만 고치고
eigen-gap은 전혀 고치지 못합니다** — uniform shift는 간격을 보존하기 때문입니다.

**비용 (실측, B200)**: 서로 다른 512×512 64개 batched `eigh` = `0.199 s`.
step당 8 episodes × ~80 bags = 640 bags → **`1.99 s/step` (forward만, backward 제외)**.
v24 실측 step time `0.135~0.273 s` 대비 **7~15배**. 50 epoch × 512 step이면
eigendecomposition만으로 수십 시간입니다(v24는 2.5시간). 참고로 코드베이스가 실제 쓰는
64-d sketch에서는 같은 연산이 `0.007 s/step`입니다(`_covariance_projection`, 512→64 random
projection). **v27은 코드베이스가 의도적으로 피해 둔 연산을 정면으로 되살립니다.**

### 3.5 whole-bag covariance는 해상도가 틀렸다

생성기의 covariance 효과는 `effect_mask`(responsive component, 전체 세포의 **4~18%**)의
세포들에 대해 **latent 방향 1개**의 분산을 `exp(s·c)`로 바꿉니다
(`synthetic_data.py:467-495`). bag 전체 공분산에 미치는 영향은 `O(f·(e^{sc}−1))`이고,
같은 bag에는 `donor_component_shift`(0.12)와 mixture fraction 잡음(logit scale 0.65)이
함께 들어 있습니다. **component 단위 공분산**이 필요한데 v27은 bag 단위로 재고,
component 단위는 이미 `slot_covariance_sketch`로 존재합니다.

또 F6에 따라 covariance는 애초에 약점이 아닙니다. 가장 비싼 신규 branch를 약점 아닌
task에 배정하는 자원 배분입니다.

### 3.6 T2-2를 오독했다

§1.1 표는 "Raw observable mean probe(0.5478) vs Oracle(0.8819)"를 근거로 "Bag 요약
단계에서 이미 대규모 정보 손실"이라 결론합니다. 같은 표에서 **모델 입력 토큰 probe는
0.6210, 모델은 0.6217**입니다. bag 요약은 raw mean보다 **좋고**, 0.88과의 격차는 요약
손실이 아니라 **`responsive_instance_mask`(정답 세포 선택)** 때문입니다. v27은 그 오라클
격차를 요약 방식 탓으로 돌린 뒤, 세포 선택을 개선하지 않는 처방(비지도 anchor 8개)을
내립니다.

### 3.7 기타

- config가 `bag_tokenizer:`/`riemannian_branch:`/`soft_gating:` 중첩 dict를 쓰지만
  코드베이스 규약은 flat `aggregator_*`/`meta_*`입니다(중첩은 `covariance_relation`만).
- 파라미터/연산 예산 추정이 없습니다. §3.4에 따르면 실험 경제성이 완전히 달라집니다.
- `L_gating_entropy` 0.05의 부호(엔트로피 최대화/최소화)가 명시되지 않았습니다.
- §2.3 fusion은 v26과 동일한 simplex 문제를 가집니다(§2.2).

### 3.8 v27이 옳게 지적한 것

- 작은 episode batch에서 hard top-2 routing은 gradient variance 문제를 만든다 — 맞습니다.
- mean-pooled episode embedding은 donor shift 잡음에 취약하다 — 방향은 맞습니다.
- v26이 bag 표현을 그대로 둔 것을 지적한 것 — 지적 자체는 타당하나 F1~F3이 그 축을 닫습니다.

---

## 4. 신규 측정: 상한이 어디이고 무엇이 그것을 막는가

### 4.1 생성 과정의 충분통계

`synthetic_data.py:281-606` 기준 한 episode:

- bag `i`의 라벨은 연속 점수의 부호: `y_i = 1[s_i > 0]`, `s_i = ±(0.08 + 0.8|z|)`
- episode마다 latent 32-d에 mixture component `num_shared+1`개 (`num_shared ∈ [4,10]`)
- 반응은 **오직 하나의 component**(`effect_component_index`)에만 걸리고 채널은 셋:
  composition = 그 component의 **abundance** 이동, state = 방향 `d`로 **평균 이동**,
  covariance = 방향 `v`로 **분산 배율** `exp(s_i·c)`
- nuisance: `donor_shift` 0.35(bag 전체), `donor_component_shift` 0.12(bag×component),
  mixture logit 잡음 0.65/0.70, observation noise 0.01
- 관측은 **episode마다 새로 뽑히는 random 3-layer GELU MLP**(32→96→96→512) 후 L2 정규화

따라서 bag-level 충분통계는 **component별 `(log π_ik, μ_ik, log σ²_ik)`**이고,
generator가 계산해 두는 oracle 특징이 정확히 이것입니다:

```python
# synthetic_data.py:550-563
oracle_population_features = cat((fraction_logit, population_mean, population_variance))
```

**T2-2의 `0.9013`은 이 특징에 context-label ridge를 붙인 값입니다.**

### 4.2 v24는 이 통계를 이미 계산한다

`baseline.py:_forward_dense`가 bag마다 slot별로:

| 코드 | 수량 | 충분통계 대응 |
|---|---|---|
| `proportion` (`:899`) | slot별 soft abundance | `π_ik` |
| `slot_mean` (`:900-902`) | slot별 가중 평균 | `μ_ik` |
| `slot_std` (`:904-911`) | slot별 feature-wise 표준편차 | `σ_ik` |
| `slot_covariance_sketch` (`:918`) | slot별 16-d 공분산 | 보너스 |

**전부 있습니다.** 다만 identity-aligned in-context ridge(`_abundance_ridge_logits`)는
`slot_metadata` 2채널(log π, dispersion)만 받고(`baseline.py:3345`), `slot_mean`/`slot_std`는
512-d 토큰 → 1-token projection → 라벨별 class-memory pooling을 거치며 slot identity가
소멸합니다. 초안은 이것을 "경로 손실"로 보고 v28-A를 제안했습니다. §4.3이 그것을 반증합니다.

### 4.3 감사 결과 (1,000 episodes, 학습 파라미터 0개)

`diagnose_state_upper_bound.py`와 **동일한** episode stream(`diagnostic_episode`,
val seed 50042)과 **동일한** context/query 분할(`query_index` = 공식 평가 collator와 같은
규칙)을 사용했습니다. 따라서 T2-2의 `0.6217`/`0.9013`과 직접 비교 가능합니다.

각 variant는 bag-centred 세포를 episode PCA-32로 투영해 slot별
`(logit π, mean(32), log var(32))`를 만들고, context bag에 class-balanced dual ridge(λ=10)를
풀어 query bag을 예측합니다. **학습되는 파라미터는 없습니다.**

| variant | 차원 | ALL [95% CI] | composition | state | covariance | interaction | combined |
|---|---:|---|---:|---:|---:|---:|---:|
| bag_global (K=1, 분할 없음) | 32 | 0.6299 [0.619,0.640] | 0.7005 | 0.5538 | 0.5438 | 0.5715 | 0.7341 |
| v24 분할, **π만** (≈현 ridge 입력) | 12 | 0.6240 [0.613,0.635] | 0.6611 | 0.5636 | 0.5410 | 0.5878 | 0.7245 |
| v24 분할, π+log σ² | 396 | 0.6569 [0.645,0.667] | 0.7315 | 0.5762 | 0.5593 | 0.6076 | 0.7656 |
| **v24 분할, π+μ** | 396 | **0.7001 [0.687,0.711]** | 0.7901 | 0.6186 | 0.5885 | 0.6441 | 0.8060 |
| v24 분할, π+μ+log σ² | 780 | 0.6947 [0.682,0.706] | 0.7837 | 0.6155 | 0.5842 | 0.6411 | 0.8016 |
| 동, 방향만 (반경 폐기) | 780 | 0.6972 [0.685,0.708] | 0.7864 | 0.6188 | 0.5854 | 0.6425 | 0.8037 |
| PCA k-means K=12 | 780 | 0.6908 [0.678,0.702] | 0.7730 | 0.5903 | 0.5886 | 0.6419 | 0.8061 |
| PCA k-means K=48 | 3120 | 0.6768 [0.664,0.689] | 0.7522 | 0.5804 | 0.5682 | 0.6227 | 0.8063 |
| **oracle 2-slot (responsive/배경)** | 130 | **0.9346 [0.929,0.940]** | 0.9374 | **0.9698** | 0.7585 | 0.9678 | 0.9790 |
| *참조: 학습된 v24/v22 모델* | 9.45M | *0.7078*\* | *0.7729*\* | *0.6215* | *0.6216* | *0.6628*\* | *0.8201*\* |

\* `evaluate_synthetic.py` 스트림. state 0.6215/0.6217은 `diagnose_state_upper_bound.py`가
동일 스트림·동일 분할로 측정한 값이므로 직접 비교 가능합니다.

> [!IMPORTANT]
> **핵심 결론 — 학습 파라미터 0개의 closed-form ridge가 학습된 9.45M 모델을 task별로
> ±0.02 안에서 재현합니다.**
> overall 0.700 vs 0.708, state 0.619 vs 0.622, composition 0.790 vs 0.773,
> interaction 0.644 vs 0.663, combined 0.802 vs 0.820, covariance 0.589 vs 0.622.
>
> 이것은 T2-2(F4)를 **5개 task 전체와 slot 통계 계열 전체로 확장한** 결과입니다.
> 따라서 **v22~v25의 0.0073 정체는 아키텍처 탐색 실패가 아니라 "비지도 population slot으로
> 요약한 bag"이라는 특징 집합의 정보 상한(≈0.70)입니다.** 그 상한 아래에서 토큰 구성·융합·
> routing을 바꾸는 모든 제안(v23~v27 전부, 그리고 이 문서 초안의 v28-A)은 구조적으로
> ±0.02 안에서 움직입니다.
>
> **정정**: 초안은 40-episode 예비 측정(state 0.767)을 근거로 "경로 손실이 크다"고
> 주장했습니다. 1,000-episode에서 state는 0.6155~0.6188이며 모델과 동률입니다.
> 예비값은 표본 잡음이었습니다(task당 ~8 episode). §3 T3-2의 경고("104개로 판정 금지")가
> 그대로 적용됩니다.

부수 결과:

- **π만(12차원)으로도 0.624** — 즉 v24의 현 `_abundance_ridge_logits`는 이미 상한의
  대부분을 담고 있습니다. π+μ로 넓히면 `+0.076`이지만, **전체 모델은 이미 0.708**이므로
  그 정보를 신경 경로로 회수하고 있습니다. 경로 손실 가설은 기각입니다.
- **반경 채널은 무관** (방향만 0.6972 vs 전체 0.6947, CI 겹침). 초안의 §5.1 가설 기각.
- **K를 늘리면 나빠짐** (K=48이 K=12보다 `−0.014`). 분할 세분화는 답이 아닙니다.
- **covariance만 모델이 probe보다 나음** (0.622 vs 0.589) → v24의 전용 covariance branch는
  실제로 일하고 있습니다. 어떤 재설계에서도 유지해야 합니다.

### 4.4 분할 품질 (1,000 episodes)

| 분할 | purity | capture | fragmentation | base rate |
|---|---:|---:|---:|---:|
| v24 (hard argmax, K=12) | 0.2260 | 0.1502 | 0.9679 | 0.1539 |
| PCA-32 k-means K=12 | 0.2374 | 0.1359 | 0.9785 | 0.1539 |
| PCA-32 k-means K=48 | **0.3477** | **0.0503** | 0.9742 | 0.1539 |

v24 soft assignment의 정규화 엔트로피는 **0.5404** — 균등(1.0)이 아니고 적당히 집중돼
있습니다. 즉 문제는 "할당이 뭉개져 있다"가 아니라 **뭉치는 축이 responsive component가
아니다**입니다(capture 0.150 vs base rate 0.154). T1-B의 capture 0.155와 정합합니다.

### 4.5 T1-C 2 재검정 — 분할 발견 경로는 **세 번째로** 닫혔다 (400 episodes)

T1-C 2는 **전역 선형** 규칙 2종(whitened mean difference, 대각 log-variance ratio)으로
**세포**를 골랐고 purity 0.128로 종료됐습니다. 선형 점수는 "component k에 속함"을 표현할
수 없으므로(component는 서로 다른 위치의 점군이지 half-space 순서가 아님), 훨씬 강한
**component 단위 다변량** 규칙으로 재검정했습니다: 분할 K개 중 in-context ridge가 가장
크게 의존하는 slot을 responsive component 예측값으로 씁니다. 사전 게이트는 원안 그대로
purity ≥ 0.30 진행 / ≤ 0.15 종료.

| K | purity(선택) | capture(선택) | **purity(oracle 최적 slot)** | ridge가 최적 slot을 고른 비율 | base |
|---:|---:|---:|---:|---:|---:|
| 12 | 0.1922 | 0.0990 | 0.2374 | 0.2425 | 0.1556 |
| 24 | 0.2175 | 0.0550 | 0.2875 | 0.1950 | 0.1556 |
| 48 | 0.2296 | 0.0271 | **0.3457** | 0.1625 | 0.1556 |

task별 purity(선택) vs base rate:

| task | K12 | K24 | K48 | base |
|---|---:|---:|---:|---:|
| state | 0.1451 | 0.1499 | 0.1381 | 0.1209 |
| composition | 0.2238 | 0.2226 | 0.2146 | 0.1991 |
| covariance | 0.1359 | 0.1548 | 0.1626 | 0.1161 |
| interaction | 0.1717 | 0.1966 | 0.1970 | 0.1512 |
| **combined** | 0.2771 | 0.3583 | **0.4312** | 0.1827 |

> [!IMPORTANT]
> **게이트 미달.** state/composition/covariance는 base rate 대비 사실상 무작위입니다
> (state 0.145 vs 0.121). 더 결정적으로 **oracle이 최적 slot을 골라줘도 purity 상한이
> K=48에서 0.346**입니다 — 실패 원인은 고르는 규칙이 아니라 **이 분할 계열 자체가
> responsive component를 담아내지 못한다**는 것입니다. `select_accuracy` 0.16~0.24라
> influence 기반 정제도 대부분 틀린 slot을 정제하게 됩니다.
>
> 예외는 **combined**(purity 0.431 vs base 0.183 @ K=48) — 세 채널이 동시에 걸려 신호가
> 가장 강한 task에서만 선택이 작동합니다. T1-C 1의 선형 이득 곡선, Easy tier 결과와 정합.

---

## 5. 남은 두 경로

§4는 "관측값 + bag 라벨"만으로는 상한이 `≈0.70`이고, 올바른 분할이 있으면 `0.93`이라는
것을 보여줍니다. 그 분할을 찾는 관측 경로는 닫혔습니다. 따라서 남은 것은 둘뿐입니다.

### 5.1 경로 A (저위험, 즉시 실행 가능): context/label 효율

F7이 지금까지 측정된 **유일하게 큰 레버**입니다: context 40→300에서 `+0.126`.
지금까지의 모든 아키텍처 델타(≤0.005)보다 25배 큽니다. 그리고 실제 배포점인 ICI는
fold당 context ~69명이므로 **작은 context 구간이 정확히 관심 영역**입니다.
F11(v25가 ctx40에서 유의하게 우세, ctx300에서 열세)은 작은 context가 큰 context와
다른 최적점을 가진다는 직접적 증거입니다.

구체 후보 (모두 아키텍처 재설계 아님):

1. **in-context 추정 분산 감소** — `_abundance_ridge_logits`/`RidgeResidual`의 shrinkage를
   feature block별로 학습 (현재 λ는 전역 스칼라 1개). context가 작을 때 자동으로 강한
   정규화가 걸리도록. §4.3이 π+μ 확장의 상한을 이미 알려 주므로 목표치가 명확합니다.
2. **context 크기 조건화** — bag 수를 명시 입력으로 주고 shrinkage/온도를 조건화.
   현재 모델은 context 40과 300을 구분하는 신호를 받지 않습니다.
3. **작은-context 특화 학습 분포** — mixed-context FT는 이미 시도됐고 ctx40/80에서
   `+0.005~0.007`뿐이었습니다(§6). 반대로 40~80에 **집중**한 분포로 학습하면
   ICI 구간이 개선되는지. 판정은 ctx40/80 AUROC.
4. **bag 분할 증강** — 각 context bag을 반으로 쪼개 ridge의 유효 n을 2배로.
   같은 라벨·같은 donor shift를 공유하므로 통계는 노이지해지지만 추정 분산은 줄어듭니다.
   **재학습 없이 평가만으로 검증 가능** (기존 checkpoint로 split-context 평가).

우선순위: **4 → 1 → 2 → 3** (재학습 없는 것부터).

### 5.2 경로 B (고위험·고수익): privileged supervision으로 분할을 가르치기

남은 `0.70 → 0.93` 격차에 도달하는 **유일하게 알려진 경로**입니다.

합성 generator는 `responsive_instance_mask`(세포별)와 `oracle_response_abundance`(bag별)를
이미 만듭니다. 후자는 dataset 출력으로 plumbing까지 되어 있고
(`return_oracle_diagnostics`, `synthetic_data.py:1152-1157`) 현재는 **진단 전용**입니다
(`model_interface.py:646` "oracle diagnostics never participate in optimization").

제안: **분할 모듈만** 보조 손실로 meta-training합니다.

$$\mathcal{L} = \mathcal{L}_{CE} + 0.10\,\mathcal{L}_{rank} + \lambda_1 \underbrace{\text{CE}\bigl(a_{ink},\ \text{mask}_{in}\bigr)}_{\text{세포→slot 정렬}} + \lambda_2 \underbrace{\|\hat{\pi}_{ik^*} - \pi^{oracle}_i\|^2}_{\text{abundance 회귀}}$$

- 분할은 미분 가능해야 하므로 현재의 `_context_anchors`(고정 random 방향 버퍼 +
  argmax k-means, `baseline.py:501-503`, `:737-817` — **학습 파라미터가 전혀 관여하지 않고
  gradient도 흐르지 않음**)를 학습 가능한 soft assignment로 교체해야 합니다.
  생성 과정과 동형인 형태: episode 공유 prototype `m_k` + bag별 mixing weight `π_ik`,
  soft E-M 2~3 step.
- **추론 시 어떤 오라클도 쓰지 않습니다.** 분할 네트워크의 입력은 관측값 + context
  라벨뿐이고, 오라클은 meta-training에서 그 모듈을 형성하는 데만 쓰입니다.
- 저장소 규칙(§6 "오라클 기반 상한을 목표치로 삼기 금지")은 **평가 목표**에 관한 것이라
  충돌하지 않지만, **명시적 승인이 필요한 방법론적 선택**입니다.
- **위험**: (a) "responsive component가 어떻게 생겼는지"의 사전지식이 합성에 과적합될 수
  있음 — ICI 전이 실패 위험. (b) §4.5가 보여준 대로 관측값만으로는 이 신호가 거의
  없으므로, 지도 학습을 해도 학습이 안 될 수 있음. (c) 세포별 mask는 현재
  `diagnostic_episode`에만 있어 collator plumbing이 필요.
- **위험 (b)를 먼저 싸게 검정할 수 있습니다** — §6 E7.

**구현 필수 사항 (분할을 건드릴 때)**: 현재 `slot_std` 계산은
`difference = instances[:, :, None, :] - slot_mean[:, None, :, :]`로
`[bags, cells, K, 512]`를 물리적으로 만듭니다(`baseline.py:903`).
`E_k[x²] − (E_k[x])²` 항등식(`einsum("bnk,bnd->bkd")` 2회)으로 바꾸면 `O(B·N·D + B·K·D)`가
되어 K 확장이 실질적으로 무료입니다.

### 5.3 하지 말 것

- v26/v27 계열(융합 routing, 토큰 구성 변경, whole-bag Riemannian) — §4.3이 상한을 잠금.
- 분할을 비지도로 개선하려는 시도 — §4.4/§4.5가 세 번째로 닫음.
- 40-episode 규모 측정으로 판정 — 이 문서 초안이 그 함정에 빠졌습니다.

---

## 6. 실험 계획 (사전 등록 게이트)

> 판정 기준은 저장소 표준: **overall +0.03 이상 또는 target task +0.05 이상**,
> 1,000 episode, episode cluster CI + paired bootstrap (§6/§7).

| ID | 무엇을 | 비용 | 사전 게이트 |
|---|---|---|---|
| **E0** | §4.3/§4.4/§4.5 감사 | 학습 0 | ✅ 완료 |
| **E2** | **v26/v27 공통 전제 검증**: 학습된 v24 고정, episode마다 5개 fusion scalar를 정답에 대해 직접 최적화한 **oracle gating 상한** | 학습 0, ~2h | `+0.03` 미달 → **MoE/soft-routing 계열 전체 폐기 확정** |
| **E2b** | fusion scale 상한 해제(`meta_population_residual_scale` 0.25→1.0 등) 짧은 FT | ~1h | CE 개선 없으면 "고정 가중치가 병목" 가설 추가 반증 |
| **E7** | **경로 B 사전 검정 (학습 없음)**: 세포 라벨을 준 지도 학습이 **episode 내에서** responsive component를 얼마나 찾는가. `diagnose_cell_selection.py --probe`의 held-out LDA(0.697)를 slot/component 단위로 확장 | 학습 0, ~1h | 지도 상한 purity ≥ 0.50 → 경로 B 진행 / < 0.30 → **경로 B도 종결, 이 계열 상한 0.70 확정** |
| **A4** | 경로 A-4: split-context 평가 (재학습 없음) | 평가만, ~1h | ctx40/80 `+0.01` 이상이면 학습 단계로 승격 |
| **A1** | 경로 A-1: block별 shrinkage 학습 | ~3h | ctx40/80 개선 + overall `+0.03` |
| **B1** | 경로 B: 미분 가능 분할 + privileged 보조 손실, with/without ablation | ~4h ×2 | with가 purity를 실제로 올리고 overall `+0.03` |
| **E6** | context-size curve 40/80/160/300 재측정 | 평가만 | **ctx40/80 개선이 1차 성공 기준** (ICI ~69) |

**순서: E2 → E7 → A4 → (A1 | B1)**

- **E2를 먼저**: 두 제안서를 학습 없이 2시간에 정식 종결하거나 되살립니다.
- **E7을 그다음**: 경로 B의 유일한 미지수(지도 신호가 있으면 배울 수 있는가)를
  학습 없이 검정합니다. 실패하면 이 아키텍처 계열의 상한이 0.70임이 확정되고,
  프로젝트는 "다른 데이터/다른 문제 정의"로 이동해야 합니다 — 그것도 유효한 결론입니다.

---

## 6.1 게이트 실행 결과 (2026-08-02, E2/E7/A4 완료)

### E2 — oracle gating 상한: **FAIL**

`v24` 체크포인트(`epoch=041-val_ce_loss=0.5903.ckpt`)를 고정하고, 6개 evidence
branch(`global_shape_logits`, `population_logits`, `tail_logits`, 역산한
`interaction`, `covariance_logits`, `covariance_relation_logits`)를 episode마다
자유롭게 재가중하는 **가장 느슨한 oracle**(context 라벨이 아니라 held-out query
라벨에 대한 leave-one-out ridge shrinkage, `λ→∞`에서 정확히 baseline로 수렴하는
것으로 정합성 검증 완료)로 측정했습니다. 1,000 episodes, 999 사용, 16,317 query:

| ridge λ | oracle AUROC | delta vs baseline | P(oracle > baseline) |
|---:|---:|---:|---:|
| 1 | 0.6250 | −0.0840 | 0.000 |
| 10 | 0.6695 | −0.0395 | 0.000 |
| 100 | 0.7002 | −0.0088 | 0.000 |
| 1,000 | 0.7084 | −0.0006 | 0.027 |
| **10,000 (best)** | **0.7090** | **+0.0000** | 0.277 |

baseline (v24 고정 가중치 fusion) = **0.7090 [0.697, 0.720]**. **최적 λ가 사실상
무한대**(=재가중 안 함)이고, 그 지점에서 oracle이 baseline과 소수점 4자리까지
정확히 일치합니다. task별로도 전부 delta ±0.0002 이내
(composition/state/interaction/combined 0.0000, covariance +0.0002).

> [!IMPORTANT]
> **query 라벨을 훔쳐보는 oracle조차 v24의 고정 가중치를 단 한 곳에서도 이기지
> 못했습니다.** 이는 "episode마다 5개 fusion scalar를 다르게 주면 더 잘한다"는
> v26·v27의 공통 전제를 실측으로 반증한 것이며, +0.03 문턱에 한참 못 미치는 정도가
> 아니라 **개선의 방향 자체가 존재하지 않습니다**. §6/§7 판정: **v26(EC-MoE) 폐기
> 확정, v27(AC-ICAR)의 routing/gating 부분 폐기 확정** (§3.4의 실행 불가능성과
> 별개로, 이 부분은 실행 가능하다 해도 이득이 없음).

### E7 — 지도 component selection 상한: **INCONCLUSIVE**

동일 v24 분할(`_context_anchors`)에 대해 세포 단위 held-out Fisher 판별
(bag을 정확히 반으로 쪼개 fit/held로 나누고, 각 bag당 4회 독립 분할 평균)을
1,000 episodes, 74,472 bag 단위로 측정했습니다 (purity는 bag-level bootstrap 95% CI):

| task | held-out AUROC | purity [95% CI] | base rate | bags |
|---|---:|---|---:|---:|
| **state** | 0.7463 | **0.3339** [0.331, 0.337] | 0.121 | 14,117 |
| interaction | 0.7350 | 0.3385 [0.335, 0.341] | 0.142 | 15,500 |
| combined | 0.7304 | 0.3792 [0.375, 0.384] | 0.221 | 14,522 |
| composition | 0.6943 | 0.3598 [0.355, 0.364] | 0.226 | 13,964 |
| covariance | 0.6989 | 0.2726 [0.270, 0.275] | 0.116 | 16,369 |
| **ALL** | **0.7207** | **0.3351** | 0.163 | 74,472 |

> [!IMPORTANT]
> **사전 게이트(purity ≥ 0.50 진행 / < 0.30 폐기)의 정확히 사이입니다.** T1-C 2
> (bag 라벨만으로, 전역 선형 규칙, purity 0.128)보다는 명백히 높지만 — **세포
> 라벨을 직접 쓰는 지도학습**을 썼는데도 — path B가 필요로 하는 0.50에는
> 못 미칩니다. "진행하지 마라"로 확정 짓기엔 근거가 약하고, "진행하라"로 보기엔
> 상한이 너무 낮습니다. §8에 사용자 판단 항목으로 남깁니다.
>
> **task별로 갈립니다**: covariance는 **0.2726으로 단독으로는 폐기 문턱 아래**입니다
> (F6/§4.3과 정합 — covariance는 이미 전용 branch가 다루고 있고 cell-selection이
> 아닌 dispersion 신호이므로, 애초에 이 진단이 잡을 대상이 아님). state/interaction은
> 0.33~0.35로 gate 중간, composition/combined은 0.36~0.38로 상대적으로 높습니다.
> 만약 path B를 진행한다면 **task 전체가 아니라 composition/combined 위주로
> 범위를 좁히는 것**이 더 근거 있는 선택입니다.

### A4 — split-context 재평가: **약한 효과, context 커지면 소멸**

`configs/train_v24_medium_bag_proj_residual.yaml` + 확정 체크포인트, context
40/80/160 (pool 340, 400 episodes). Context bag을 정확히 반으로 쪼개 라벨은
유지한 채 ridge 유효 표본만 2배로 늘리는 조작이며, **재학습 없이 동일
checkpoint**로 baseline과 paired 비교했습니다.

| context | baseline | split | delta | P(split > baseline) |
|---:|---:|---:|---:|---:|
| 40 | 0.6749 | 0.6784 | **+0.0035** | 0.950 |
| 80 | 0.7179 | 0.7192 | +0.0013 | 0.781 |
| 160 | 0.7713 | 0.7712 | −0.0001 | 0.476 (노이즈) |

ctx40 task별: composition +0.0036 / state +0.0043 / covariance +0.0025 /
interaction +0.0000 / combined +0.0055 — 전부 작지만 방향은 일관되게 양수.
ctx160에서는 state −0.0051 / covariance −0.0057로 **부호가 뒤집힙니다**.

> [!NOTE]
> 첫 40-episode 예비 측정(+0.0124/+0.0117, P≈0.97)은 §3 T3-2가 경고한 그대로
> **표본 잡음**이었습니다. 정식 400-episode 결과는 방향은 맞지만(작은 context에서
> 양수) 크기가 승격 기준(+0.03/+0.05)에 한참 못 미치고, ICI 관련 구간(ctx40/80)
> 밖(ctx160)에서는 사라지거나 역전됩니다.
>
> **판정**: A4는 "학습된 shrinkage가 도움될 가능성이 있다"는 방향성 힌트로는
> 유효하지만, **A1(block별 shrinkage 학습)에 3시간을 투자할 만큼 강한 근거는
> 아닙니다.** 승격도 폐기도 아닌 "낮은 확신의 계속 관찰" 상태로 남깁니다.

### 종합

세 게이트 중 어느 것도 "명확한 승격"을 주지 않았습니다: E2는 명확한 폐기,
E7과 A4는 애매하거나 약합니다. 이것은 우연이 아니라 **§4.3이 이미 측정한 상한
(비지도 slot 통계 기반 ≈0.70)이 이 세 실험 전부가 건드리는 표면(fusion 가중치,
분할 품질, context 표본 효율) 아래에 실제로 깔려 있다는 일관된 신호**로
읽힙니다. §8 사용자 판단 항목을 이 결과에 맞춰 갱신했습니다.

---

## 7. 반증 조건

1. ~~E2가 `+0.03`을 넘음 → 병목은 정말 fusion routing이었고 v26이 옳다~~ —
   **해당 없음. E2 delta는 정확히 0.0000으로 이 조건은 발생하지 않았다.**
   v26/v27의 gating 전제는 반증 확정.
2. **E7의 지도 상한이 높은데 B1이 실패** → 문제는 특징도 분할도 아니라
   **meta-training/optimizer**다. §6 T4의 3번째 가설로 이동.
   (E7이 0.335로 애매하게 나왔으므로 이 조건 자체가 아직 판정 불가 — B1을
   돌려야 구분됨)
3. **A4/A1이 ctx300만 개선하고 ctx40/80은 그대로** → 경로 A는 ICI에 무용하다.
   **A4 실측은 오히려 반대 패턴**: ctx40에서 가장 크고(+0.0035) ctx160에서
   소멸(-0.0001)한다. 즉 "ICI에 무용"은 아니지만 "크기가 너무 작다"는 별도
   문제로 A1을 여전히 정당화하지 못한다.
4. **B1이 합성에서 크게 이기고 ICI에서 무너짐** → privileged 사전지식의 과적합.
   ICI는 1회용이므로 이 위험은 사전에 감수 여부를 결정해야 한다(§8-2).

---

## 8. 사용자 판단이 필요한 항목 (2026-08-02 E2/E7/A4 결과 반영 갱신)

1. ~~E2/E2b를 먼저 돌려 v26·v27을 정식 종결할지~~ → **완료. v26 폐기 확정,
   v27의 routing 부분 폐기 확정** (§6.1). E2b(fusion scale 상한 해제 FT)는
   이제 우선순위가 낮음 — E2가 이미 "고정 가중치가 최적"이라는 결론을 강하게
   지지하므로, 상한을 풀어도 달라질 것으로 기대하기 어렵습니다. **생략 권고.**
2. **경로 B의 privileged supervision 사용 여부.** E7이 애매하게 나왔으므로
   여전히 사용자 판단이 필요합니다 — 다만 이제 세 갈래로 좁혀졌습니다:
   (a) 전체 task B1 진행, (b) composition/combined만 좁혀서 B1 진행
   (§6.1 E7의 task별 결과 참고), (c) B1 자체를 보류하고 T4/다른 방향으로 이동.
3. **경로 A(A1, 학습된 shrinkage) 투자 여부.** A4가 방향은 맞지만
   (ctx40 `+0.0035`, P=0.950) 크기가 작고 ctx160에서 소멸합니다. 3시간을
   투자할 근거로는 약합니다 — **이번 실측 기준으로는 A1도 보류 권고**하되,
   최종 결정은 사용자 몫입니다.
4. **T4 attribution(§6, current_status.md)과의 우선순위** — 세 게이트가
   전부 약하거나 불리하게 나온 지금, **T4(Medium→Hard 붕괴 원인 규명)로
   우선순위를 옮기는 것이 상대적으로 더 근거 있는 선택**일 수 있습니다.
5. **평가 예산** — task별 `+0.05` 미만을 노리려면 §3 T3-2에 따라 val episode
   **2,000개 이상**이 필요합니다. E7의 task별 차이(covariance 0.27 vs
   composition 0.36)가 실제 차이인지 표본 크기 아티팩트인지도 이 규모에서만
   확인 가능합니다.

---

## Appendix A: 근거 코드 위치

| 주장 | 위치 |
|---|---|
| 분할이 학습되지 않음 (고정 random 버퍼 + argmax) | `baseline.py:501-503`, `:737-745`, `:747-817` |
| slot별 (abundance, mean, std) 이미 계산됨 | `baseline.py:898-915` |
| identity-aligned ridge가 metadata 2채널만 받음 | `baseline.py:915`, `:2774-2830`, `:3345-3348` |
| slot 경로가 방향만 받음 (반경 폐기) | `baseline.py:547-564` |
| v24에서 slot routing이 no-op | `baseline.py:2570-2580`, `:2614-2620` |
| additive residual 융합 (계수 1.0 base) | `baseline.py:2766-2773` |
| routing entropy/balance loss 존재, weight 0 | `model_interface.py:530-550`, `train_v22_medium.yaml` |
| `[B,N,K,D]` 물리 생성 → K 확장 병목 | `baseline.py:903` |
| oracle 특징 = (fraction, mean, variance) | `synthetic_data.py:550-563` |
| oracle abundance plumbing 존재, 최적화 미참여 | `synthetic_data.py:1152-1157`, `model_interface.py:646` |
| covariance는 64-d random projection sketch | `baseline.py:512-518`, `:566-600` |
| 반응 효과는 component 1개에만 적용 | `synthetic_data.py:458-511` |
| 관측 manifold가 episode마다 새 random MLP | `synthetic_data.py:754-767`, `manifold_mode` 기본값 `nonlinear` |

## Appendix B: 재현

이번 세션의 감사 스크립트 2개는 scratchpad에 있으며 위 수치를 그대로 재현합니다.
채택 시 `scripts/`로 이관하고 unittest를 붙여야 합니다.

```
probe_partition2.py   # §4.3 / §4.4  (1,000 ep, 116.7 s + bootstrap)
probe_selection.py    # §4.5         (400 ep, 408.3 s)
```

산출물: `audit_1000_v2.json`, `selection_400.json`.
