# Current architecture — 활성 relation 계보와 역사적 비교군 (2026-08-12)

> [!IMPORTANT]
> **활성 baseline은 v76** (`CovarianceMeanLearnablePDDCTMLPModel`, SEAL 0.6748)이고,
> 동일 구조의 Hard `[0.2,0.8]` 단일-seed 후보는 **0.6873**이다.
> centered cells를 learnable orthogonal P(1536×128)에 사영해 만든 covariance를 CV와 DD가
> 공유하고, CT와 함께 12개 relation feature를 만들어 12→32→1 MLP가 읽는다.
> 학습 파라미터는 P 196,608개 + head 449개 = **197,057개**다. P는 CV ridge 경로로만
> 학습되며 DD/CT는 training-free다. 현재 실행 중인 opt-in ridge-calibration arm은
> `ridge_log_lambda`, `ridge_log_scale` 두 스칼라를 추가해 **197,059개**를 학습한다.

리포에는 활성 relation 계보와 역사적 비교 계보 두 개가 있다. relation 계보와 Encoder 계보는
`src/models/set_transformer_ridge.py`에 있고, 역사적 CV-only는 `src/models/baseline.py`에 있다.
공통 핵심 유틸리티는 episode-local ridge의 `solve_ridge_system`이다.

| | Active. Relation v76 | A. CV-only | B. Encoder+Ridge |
|---|---|---|---|
| 파일 | `set_transformer_ridge.py` | `baseline.py` | `set_transformer_ridge.py` |
| 클래스 | `CovarianceMeanLearnablePDDCTMLPModel` | `BaseModel` | `SetTransformerRidgeModel` |
| bag 기술자 | learnable P covariance + raw mean | fixed P covariance | learned Transformer |
| readout | CV/DD/CT 12→32→1 | CV-1 ridge + CV-2 | episode-local ridge |
| 학습 파라미터 | 197,057 (calibration arm 197,059) | 229 | 5,010,946 |
| SEAL 10개 최고 | Hard 후보 0.6873 | **0.6940** (v41_K128) | 0.6619 / 0.6526 |
| 상태 | **활성** | 역사적 전체 최고 | 기각 |

이전 세대(v34~v39의 6-분기)는 **소스에서 삭제**됐다(§73). 필요하면 git `8caa96c`.

---

# Active. v76 learnable-P CV+DD+CT relation model

## Active-1. Forward 경로

```text
bag cells x_b (N_b × 1536)
  ├─ raw mean μ_b ----------------------------------------------------┐
  └─ centered x_b − μ_b                                              │
       → P_eff = thin_QR(P), P ∈ R^(1536×128)                        │
       → covariance upper triangle (8,256)                           │
       └──────────────────────┬───────────────────────────────────────┘
                              → canonical descriptor (9,792)

context descriptor + support label
  → class-balanced episode-local dual ridge → CV0, CV1, margin, SEP_CV
  → generalized dispersion direction       → D0, D1, margin, SEP_DD

raw support/query cells
  → bag당 최대 64 cell → farthest-point 후보 16개
  → support label로 class-discriminative token 2개 대칭 선택
  → query abundance 관계                     → q0, q1, margin, SEP_CT

[CV 4, DD 4, CT 4] = 12 features
  → Linear(12,32) → GELU → Linear(32,1)
  → symmetric binary logits [-margin/2, +margin/2]
```

CV ridge coefficient는 real/synthetic episode의 support label로 매번 다시 푼다. 학습 checkpoint가
고정 classifier weight를 저장하는 구조가 아니다. raw mean과 covariance block은 context-only로
각각 center/scalar-RMS 정규화한다.

## Active-2. Gradient와 학습 계약

기본 v76에서 학습되는 것은 다음뿐이다.

| 파라미터 | 개수 | gradient 경로 |
|---|---:|---|
| `_covariance_projection` P | 196,608 | CV ridge solve를 통과 |
| `cv_dd_ct_head` | 449 | 12개 relation feature |
| **합계** | **197,057** | |

- 매 forward에서 thin QR로 `P_effᵀP_eff=I`를 보장해 임의 scale/conditioning 변화를 차단한다.
- DD는 현재 P로 만든 covariance를 읽지만 DD→P gradient는 차단된다. CT 선택/특징도 training-free다.
- 기본 `ridge_log_lambda=log(1)`, `ridge_log_scale=log(2)`는 v74/v76에서 동결된다.
- `train_ridge_calibration: true`는 이 두 scalar만 동결 해제한다. 이 opt-in arm은 197,059개이며
  기존 checkpoint/config의 학습 계약은 바꾸지 않는다.

## Active-3. 현재 synthetic data 계약

Hard 실험의 공통 조건은 다음과 같다.

| 항목 | 값 |
|---|---|
| bags/episode | 60–100 |
| raw cells/bag | 256–8,192, bag별 독립 log-uniform(power 2.0) |
| 실제 training cap | 4,096; 초과 시 매 step random subsample |
| latent/output | 32 / 1,536 |
| shared populations | 4–10, fraction 0.82–0.96 |
| response tasks | composition/state/covariance/interaction/combined 동일 확률 |
| ClassSep | **Hard `[0.2,0.8]`** |
| observation noise | 0.005 |
| training | DDP4 GPU 0–3, bf16, 50 epochs |

`manifold_mode`은 실험 축이다.

- `orthogonal`: episode마다 fresh isometric linear map. Hard 0.6873으로 현재 데이터 후보 최고.
- `mlp_bank`: 고정 3-layer MLP를 bank ID seed로 재생성. M=128/512/1024/2048/4096 결과는
  0.6697/0.6726/**0.6779**/0.6751/0.6649.
- `mixed_linear_mlp_bank`: episode마다 mapping 하나를 선택한다. 50% fresh linear + 50% MLP-1024는
  synthetic val CE 0.2218로 좋아졌지만 SEAL은 0.6755로 하락해 기각했다.

중요한 역사적 차이: v40/v41 데이터는 episode-level cardinality `[1,16384]`, cap 없이 fresh
3-layer nonlinear MLP를 썼다. 현재 Hard는 per-bag cardinality와 4,096 cap을 쓰므로 v41과 v76의
성능 차이를 아키텍처만의 차이로 해석하면 안 된다.

## Active-4. 현재 실험과 판정

- 활성 실행: Hard orthogonal + learned ridge λ/logit scale, runner PID/PGID `1952961`.
- control: 동일 Hard orthogonal v76 **0.6873**.
- 판정: validation-best 하나의 공식 SEAL 10-task macro와 task별 regression. synthetic val 지표는
  checkpoint 선택에만 사용한다.
- GPU 정책: ICF는 GPU 0–3만 사용하고 4–7은 사용하지 않는다.

## Active-5. Large-bag ragged fine-tuning 계약

`data.ragged_training: true`는 `episode_batch_size=1`에서만 허용되며, training collator가
list-of-bags를 padding tensor로 바꾸지 않고 그대로 모델의 ragged forward에 전달한다. 기본값은
false이므로 기존 dense masked training 계약은 유지된다. 현재 large-bag arm은 bag별
`[2048,16384]`, `per_bag_max_cells: 16384`를 사용해 4,096 cap을 제거한다.

이 arm은 scratch가 아니라 동일 Hard orthogonal v76 best checkpoint
`checkpoints/20260812_v76_classsep_sweep/hard/epoch=048-val_ce_loss=0.1697.ckpt`를
`--init-checkpoint`로 weight-only load한다. optimizer/scheduler/epoch state는 새로 시작한다.

---

# A. CV-only (`BaseModel`)

## A-0. 한 줄 요약

> **세포 집단의 2차 통계(공분산)를 고정 사영으로 요약하고, 그 위에서 closed-form
> ridge를 에피소드마다 새로 풀어 분류한다.** 사영도 회귀계수도 학습되지 않는다.

## A-1. 전체 경로

```text
입력: bag b의 세포들  x_b  (N_b × 1536)

[1] 중심화        centered_b = x_b − mean_cells(x_b)          파라미터 없음
[2] 사영          proj_b = centered_b @ P                      P: (1536 × K) 고정 buffer
[3] 2차 모멘트    C_b = proj_bᵀ proj_b / N_b
[4] 상관행렬화    R_b = D^(−1/2) C_b D^(−1/2)
[5] shrinkage     R_b ← 0.9·R_b + 0.1·I
[6] 상삼각 벡터화 sketch_b = vec_triu(R_b)                     K(K+1)/2 = 8,256 (K=128)

분기 CV-1 ─ sketch + context 라벨 → closed-form dual ridge → (Q, 2)
분기 CV-2 ─ covariance_matrix + 라벨 → subspace → prototype → head → (Q, 2)

최종:  logits = σ(cov_residual_logit)·(ridge_scale·CV-1) + 0.5·CV-2
```

## A-2. 사영 P — 학습되지 않는다

```python
P = QR( sin(a·d·k) + cos(b·(d+1)·k) ).Q     # register_buffer(persistent=False)
```

> [!IMPORTANT]
> **`a = 0.85π/K`로 대역폭을 고정해야 K 스윕이 공정하다**(§69-4). `a`를 고정한 채 K를
> 바꾸면 대역폭이 함께 변해 차원 효과가 가려진다. 0.85는 가드밴드 — `a·K = π`면
> `sin(π·d) = 0`(정수 d)이라 그 열의 sin 항이 소멸한다.

**P는 데이터도 라벨도 보지 않는다.** §69에서 label-free 선택 8종(랜덤/PCA/Sobol QMC/
앨리어싱/사다리 간격/대역폭/위상)을 전부 시험했고 **모두 0.68 ± 0.03**이었다.
라벨을 보는 사영이 미시험 축이었고, 그것이 계보 B다.

## A-3. CV-1 — closed-form dual ridge

- **매 에피소드·매 폴드 새로 푼다.** 학습으로 얻어 고정하는 W가 없다.
- 학습 파라미터는 `covariance_ridge_log_lambda`, `_log_scale` **스칼라 2개**뿐.
- 단독 AUROC **0.9052** (합성 200 에피소드, 전체 0.9199) — 성능 대부분이 여기서 나온다.

> [!IMPORTANT]
> **dual인 이유는 "bag 수 ≪ 특징 수"이지 instance와 무관하다.** 설계행렬은
> (bag 수 × 8,256)이고 bag은 60~133개다. instance(N)와 임베딩 차원(1536)은 이미 공분산으로
> 요약돼 사라진 뒤라, "N > d니 kernel 불필요" 논리는 여기 적용되지 않는다.
> 실측(§78): dual 0.33 ms vs primal 9.8 ms — **30배**.

## A-4. CV-2 — subspace + prototype (유일한 학습 모듈)

```text
[1] delta = mean(M | y=1) − mean(M | y=0)
[2] whitening (shrinkage 0.25) → operator
[3] |고윳값| 상위 rank개 방향 선택
[4] 그 방향의 로그분산 → [5] context 통계로 z-score (query도 context 통계 사용)
[6] prototype = 클래스별 z 평균 → [7] 거리 → [8] head
```

**head 모드 2가지** (`covariance_relation.mode`):

| 모드 | head 입력 | 라벨 대칭성 | SEAL 10개 |
|---|---|---|---|
| `learned_head` (기본) | 스칼라 4개 (rank 축을 평균으로 소거) | **깨짐 (4.4e-2)** | 0.6940 |
| `paired_head` (§77) | 차원마다 1행 (rank 축 유지) | **정확 (구성으로 보장)** | 0.6937 |

```
paired_head:  margin = mean_r [ h(e0_r, e1_r, sep_r) − h(e1_r, e0_r, sep_r) ]
```
라벨 교환이 두 호출을 맞바꾸므로 부호만 뒤집힌다 — 학습이 아니라 구성으로 보장된다.
head가 차원 간 공유되어 **파라미터 shape이 `subspace_rank`와 무관**하다(ckpt 호환 유지).

> [!WARNING]
> `paired_head`의 출력단 bias는 `h(a,b,s) − h(b,a,s)`에서 상쇄돼 **gradient를 받지
> 않는다**. 반대칭성의 대가이며 정상이다(테스트가 이 사실을 명시적으로 고정).

## A-5. 무엇이 학습되는가 — 229개

| 위치 | 파라미터 | 개수 |
|---|---|---|
| CV-1 | `covariance_ridge_log_lambda`, `_log_scale` | 2 |
| 융합 | `covariance_residual_logit` | 1 |
| CV-2 | `covariance_relation_head` MLP | 226 |

## A-6. 계약

> [!IMPORTANT]
> **죽은 key는 zeros가 아니라 부재다.** `_validate_representation`이
> `{covariance_sketch, covariance_matrix}`만 허용하고, 빠지거나 남으면 `ValueError`다.
> 새 소비처에서 `KeyError`가 나면 **그게 정상 동작**이니 0으로 채우지 말고 분기를 가드할 것.

---

# B. Encoder+Ridge (`SetTransformerRidgeModel`)

## B-0. 왜 만들었나

§69가 label-free 사영 8종을 전부 시험해 0.68 천장을 확인했다. **라벨을 보는 사영**이
유일한 미시험 축이었다. 여기서는 기술자가 학습되고, gradient가 **ridge solve를 통과해**
인코더에 도달한다 — 실제 readout에 맞춰 표현이 최적화된다.

## B-1. 경로

```text
세포 (N × 1536)
  → 상한 8,192로 무작위 추출 (초과 시에만, 호출마다 새로)
  → 사영 (N × 512) + LayerNorm
  → summary token 32개를 앞에 붙임 (CLS 유사)
  → Encoder 2층: 세포끼리 self-attention + summary token도 서로 attend
  → summary token 32개만 취해 flatten = 32 × 512 = 16,384차원

context 기술자 + 라벨 → CV-1과 동일 레시피의 dual ridge → query logits
```

ridge 레시피(context 전용 표준화, 클래스 균형 가중, 가중 중심화 intercept, dual)를
**CV-1과 동일하게** 맞췄다. 차이가 readout이 아니라 **기술자**의 차이가 되도록.

## B-2. attention 백엔드 — cuDNN

> [!IMPORTANT]
> B200(sm_100)에서 **cuDNN이 flash보다 빠르다.** `nn.TransformerEncoderLayer`를 쓰지
> 않고 층을 직접 쓴 유일한 이유가 백엔드를 명시하기 위해서다.

| token_dim 512, fwd+bwd | FLASH | **CUDNN** | MEM_EFF |
|---|---|---|---|
| bag 85 × 2,836셀 | 17.7 ms | **6.5 ms** | 43.6 ms |
| bag 100 × 8,192셀 | 145.3 ms | **55.6 ms** | 387.1 ms |

모델이 실제로 cuDNN을 타는지 확인됨: 기본 42.9 ms == CUDNN 강제, FLASH 강제 63.9 ms.

**FlashAttention-3/4는 쓸 수 없다** — PyPI `flash-attn`은 2.8.3(FA2 계열)이 최신이고,
FA3는 Hopper(sm_90) 타깃인데 이 장비는 sm_100이다. FA4가 배포되면 바꿀 지점은
`set_transformer_ridge.py`의 백엔드 우선순위 목록 한 곳이다.

## B-3. 비용

| 구성 | step (fwd+bwd+opt) | peak VRAM |
|---|---|---|
| bag 84 × 2,836셀 (중앙값) | 45.7 ms | 10.4 GiB |
| bag 100 × 8,192셀 | 193.6 ms | 34.9 GiB |
| bag 100 × 16,384셀 (상한으로 잘림) | 195.1 ms | 44.1 GiB |

epoch 56~69초 (CV-only는 33초).

## B-4. ⚠️ 이전 판본의 설계 오류 (반복 금지)

v50~v52는 세포끼리 attend하지 않는 **inducing-point** 인코더였다. 근거는 "셀-셀
attention은 에피소드당 2.7e10 쌍이라 불가"였는데, **쌍의 개수를 실행 불가로 번역한 것이
오류**다. flash 계열 커널은 attention 행렬을 만들지 않아 메모리가 O(N)이고, 최악 구성도
**3.13 GiB**였다.

그 오판이 bag 기술자를 **256개 숫자**로 좁혔고(sketch는 8,256), v51/v52의 패배
(0.6047 / 0.6619)를 낳았을 것으로 봤다.

⚠️ **그러나 재설계도 SEAL을 올리지 못했다**(§79-6). 세포 간 attention과 16,384차원
기술자로 합성 val_auroc는 0.784 → 0.849로 올랐는데 SEAL은 **0.6619 → 0.6526**으로
내려갔고, 합성에서 가장 좋았던 v54가 SEAL에서 가장 나빴다(0.6219).
**문제는 용량이나 구조가 아니라 일반화다.** 더 키우기 전에 "왜 합성만 좋아지는가"를
답해야 한다.

---

## C. 합성 cardinality와 padding 계약

- `configs/data/default.yaml`은 `per_bag_cardinality: true`: 한 episode 안에서도 각 bag이
  `[1,8192]`(arm override 시 `[1,16384]`)에서 독립적으로 cell 수를 뽑는다.
- training collator는 4,096개를 초과한 bag을 매번 `randperm`으로 4,096개까지 subsample한 뒤
  batch 최대 길이(상한 4,096)까지 zero-padding하고 `cell_mask`/`bag_mask`를 반환한다.
  batch 크기 1도 반드시 이 dense masked 경로를 탄다. validation/test는 결정성을 위해 생성된
  cell 순서의 앞 4,096개를 사용하고 ragged 평가 경로를 유지한다.
- 모델은 padding 값을 데이터로 해석하면 안 된다. 모든 mean/covariance/attention은
  `cell_mask`로 제외하고, padded bag은 `bag_mask`로 context/query에서 제외한다.
- 변경 전과 학습 데이터 분포가 다르므로 재학습 결과를 기존 arm의 연장으로 비교하지 않는다.

## D. 주요 손잡이

| key | 기본 | 계보 | 의미 |
|---|---|---|---|
| `aggregator_covariance_sketch_dim` (K) | 64 | A | 사영 차원. sketch = K(K+1)/2 |
| `aggregator_covariance_matrix_dim` | 32 | A | CV-2가 보는 차원. `null` = K 연동 |
| `aggregator_covariance_slopes` | `null` | A | `[a, b]`. `null` = 역사적 (0.019, 0.011) |
| `covariance_relation.mode` | `learned_head` | A | `paired_head`도 가능 |
| `covariance_relation.subspace_rank` | 1 | A | CV-2 부분공간 차원 |
| `covariance_relation.margin_activation` | `tanh` | A | `identity`는 기각됨(§76) |
| `num_summary_tokens` | 32 | B | 기술자 = S × token_dim |
| `max_cells` | 8192 | B | bag당 세포 상한 |
| `token_dim` / `num_layers` | 512 / 2 | B | |

## E. Source of Truth

- 모델 A: `src/models/baseline.py` (2,224줄)
- 모델 B: `src/models/set_transformer_ridge.py`
- 평가: `scripts/eval_seal_tasks.sh` → `scripts/test_pathobench.py`
- 테스트: `tests/test_ridge_ablation.py`, `test_cvonly_golden.py`,
  `test_paired_relation_head.py`, `test_set_transformer_ridge.py`,
  `test_training_uses_dense_path.py`, `test_per_bag_cardinality_padding.py`,
  `test_config_numeric_types.py`

## F. Canonical CV branch 계약 (2026-08-11, §86)

> **앞으로 “CV branch”는 covariance 단독이 아니라
> fixed covariance upper triangle + raw pre-centering bag mean이다.**

    raw cells ─┬─ bag mean (중심화 전) ─────────────────────────── 1,536-d
               └─ bag 중심화 → fixed P → covariance upper triangle ─ 8,256-d
                                                                   (K=128)
    CV = concat(covariance, raw mean)                              = 9,792-d

covariance와 mean은 ridge 직전 context-only center/scalar-RMS로 각각 독립 정규화한다.
cell padding은 둘 다에서 제외한다. ICI 512-d 입력에서는 P=512×128이고 CV는
8,256+512=8,768차원이다.

PathoBench 무학습 SEAL 10-task에서 covariance-only 0.6630 → CV 0.6667(+0.0037),
ICI 5-seed 평균에서 0.5381 → 0.5449(+0.0068)로 두 도메인 모두 방향이 양수였다.
ICI CI는 0.5를 포함하므로 실세계 통과 주장은 하지 않는다.

- canonical: CovarianceSetTransformerRidgeModel v46, STCVLPRidgeModel v47.
- historical replay: LegacyCovarianceSetTransformerRidgeModel v42,
  LegacySTCVLPRidgeModel v43.
- CovarianceOnlyRidgeModel v44는 §86 ablation control이며 canonical CV가 아니다.

## G. Dispersion Distance와 learned relation head (2026-08-11, §87)

### G-1. DD는 무엇을 측정하는가

DD(Dispersion Distance)는 canonical CV와 같은 centered projected covariance matrix
`C_b ∈ R^(K×K)`를 사용하지만 upper triangle 전체에 ridge를 푸는 대신, support label로
episode-specific한 rank-1 방향을 해석적으로 만든다.

    C̄_c = mean(C_b | y_b=c)              Δ = C̄_1 - C̄_0
    C_pool = mean_b C_b                   τ = tr(C_pool)/K
    S = 0.75 C_pool + 0.25 τ I            W = S^(-1/2)
    A = W Δ W                             A u = λu
    f = W u,  where |λ| is maximal

`f ∈ R^K`는 전체 pooled dispersion으로 whitening한 뒤 두 클래스의 within-bag variance
차이가 가장 큰 cell projection 방향이다. 원래 1,536-d embedding 공간에서는 `P f`다.
각 bag은 이 방향의 log variance scalar로 줄어든다.

    z_b = log(fᵀ C_b f) = log Var((X_b - mean(X_b)) P f)

context-only 통계로 z-score한 뒤 class prototype과 class별 bag-to-bag dispersion을 구한다.

    p_c = mean(z_b | y_b=c)
    v_c = mean((z_b-p_c)^2 | y_b=c)
    D_c(q) = (z_q-p_c)^2 / (v_c+ε)

`D_c`는 query의 분산이 class c의 평균 분산에서 얼마나 떨어졌는지를 그 클래스의
bag-to-bag variance 단위로 잰 standardized squared distance다. 학습 파라미터는 없다.

### G-2. training-free probability와 실패한 고정 결합

반대쪽 distance가 클수록 해당 class evidence가 커지게 한다.

    p_DD(q) = [(D_1+ε)/(D_0+D_1+2ε), (D_0+ε)/(D_0+D_1+2ε)]

SEAL 10-task macro는 DD-only 0.5862, `0.5 p_CV + 0.5 p_DD` 0.6441로 약했다.
`(p_CV + 0.1 p_DD)/1.1`은 0.6688로 canonical CV 0.6667보다 +0.0021이었으나,
고정 가중치는 task별 신뢰도 차이를 처리하지 못했다.

### G-3. v70 CV+DD relation MLP

v70은 feature extractor와 ridge/DD 계산을 모두 고정하고 마지막 321-parameter MLP만 학습한다.

    [CV0, CV1, CV1-CV0, SEP_CV, D0, D1, D1-D0, SEP_DD]
        -> Linear(8,32) -> GELU -> Linear(32,1) -> margin m
    model logits = [-m/2, +m/2], so p(y=1)=sigmoid(m)

`SEP_CV`는 context-only normalized canonical-CV descriptor에서 class centroid 사이 RMS 거리,
`SEP_DD=|p_1-p_0|`는 rank-1 log-dispersion prototype separation이다. 둘 다 episode별
confidence이고 query label은 보지 않는다. CV ridge lambda=1/logit scale=2와 DD shrinkage
0.25는 고정이다.

v70 architecture는 CovarianceMeanDDMLPModel v49다. v71 ablation은 DD의 방향·거리·separation을
모두 제거하고 `[CV0,CV1,CV1-CV0,SEP_CV] -> 4→32→1`만 학습하는
CovarianceMeanCVMLPModel v50이다.

## H. 활성 baseline v74: Composition Token branch

CT는 평균이나 covariance가 아니라 **판별적인 cell-state의 bag-level abundance**를 측정한다.
각 bag에서 최대 64 cells를 균등 샘플링하고 support 전체 통계로 각 원본 coordinate를
표준화한다. 1,536차원은 projection 없이 유지한다. support cells에서 label-free
farthest-point selection으로 후보 token 16개를 만든 뒤 soft assignment abundance
`h_b ∈ R^16`을 계산한다.

    S_j = (mean(h_bj | y=0) - mean(h_bj | y=1)) / SE_j
    j0 = argmax S_j
    j1 = argmin S_j

label은 후보 생성에는 쓰지 않고 class-discriminative token 선택에만 쓴다. query bag은
선택된 token의 standardized abundance `q0,q1`을 읽는다. label swap은 j0/j1을 교환하며
query label은 어떤 단계에서도 보지 않는다.

    [CV0,CV1,CV1-CV0,SEP_CV,
     D0,D1,D1-D0,SEP_DD,
     q0,q1,q0-q1,SEP_CT]
        -> Linear(12,32) -> GELU -> Linear(32,1)

v74는 `CovarianceMeanDDCTMLPModel` architecture v52, trainable 449 parameters다.
CV/DD/CT는 모두 frozen 또는 training-free이고 relation head만 synthetic task로 학습한다.
공식 SEAL 10-task macro 0.6731로 v70보다 +0.0016, 6/10 task 상승하여 활성 baseline이다.

v73의 full-dimensional Magnitude Distance는 Woodbury로 shrinkage Fisher direction을 정확히
계산했지만 macro 0.6473으로 실패했다. raw bag mean은 canonical CV에 이미 포함되므로
Magnitude를 활성 architecture에 넣지 않는다.
