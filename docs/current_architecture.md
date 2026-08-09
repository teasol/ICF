# Current architecture — CV-only (v40~v42)

**적용 범위**: `meta_covariance_only: true` 계열. 현행 최고는 **v41_K128**
(`configs/train_v41_cvonly_K128_1536.yaml`).
**이전 세대**(v34~v39, 6-분기)는 [`history.md`](history.md).

---

## 0. 한 줄 요약

> **세포 집단의 2차 통계(공분산)를 저차원으로 사영해 요약하고, 그 요약 위에서
> closed-form ridge를 에피소드마다 새로 풀어 분류한다.** 학습되는 것은 사영도 회귀계수도
> 아니고 CV-2의 작은 MLP와 스칼라 몇 개뿐이다.

이전 세대의 6개 evidence 분기 중 **4개를 삭제했는데 성능이 동일**했다(§68,
fold-paired −0.0005). 삭제된 것: global_shape(G), population(P-2 + Q-5), rare(R),
fusion interaction. 남은 것은 covariance 2개다.

---

## 1. 전체 경로

```text
입력: bag b의 세포들  x_b  (N_b × 1536)          N_b는 bag마다 다름 (ragged)

[1] 중심화        centered_b = x_b − mean_cells(x_b)              파라미터 없음
[2] 사영          proj_b = centered_b @ P                          P: (1536 × K) 고정 buffer
[3] 2차 모멘트    C_b = proj_bᵀ proj_b / N_b                       (K × K)
[4] 상관행렬화    R_b = D^(−1/2) C_b D^(−1/2),  D = diag(C_b)
[5] shrinkage     R_b ← 0.9·R_b + 0.1·I
[6] 상삼각 벡터화 sketch_b = vec_triu(R_b)                         (K(K+1)/2,)

                  그리고 같은 centered_b 에서
[3'] M_b = proj_bᵀ proj_b / N_b  (앞 D_cv 열만)                    covariance_matrix (D_cv × D_cv)

분기 CV-1 ─ sketch (context+query) + context 라벨 → closed-form ridge → (Q, 2)
분기 CV-2 ─ M (context+query) + context 라벨 → subspace → prototype → MLP → (Q, 2)

최종:  logits = σ(cov_residual_logit)·(ridge_scale·CV-1)
              + 0.5·CV-2
```

`_fuse_evidence`가 CV-only 모드에서 global_shape/population/rare/interaction을 **계산 없이
0으로 반환**하고, aggregator는 slot 파이프라인 자체를 건너뛴다(§4).

---

## 2. 사영 P — 학습되지 않는다

```python
P = QR( sin(a·d·k) + cos(b·(d+1)·k) ).Q        # (1536 × K), register_buffer(persistent=False)
```
- `d` = 임베딩 채널 1..1536 (샘플축), `k` = 사영 방향 1..K (**주파수 배수 겸용**)
- `a`, `b` = 주파수 사다리 간격. 열 k는 채널축을 따라 `a·k` rad/step으로 진동하므로
  **`a·K`가 사용 대역폭**이다.
- **`persistent=False`** — ckpt에 저장되지 않고 공식으로 재생성된다. 랜덤 사영이면 98K~197K
  float를 매 ckpt에 저장하거나 DDP 랭크 간 seed를 맞춰야 하는데 그걸 피한다.

> [!IMPORTANT]
> **`a = 0.85π/K`로 대역폭을 고정해야 K 스윕이 공정하다** (§69-4). `a`를 고정한 채 K를 바꾸면
> 대역폭이 함께 변해 차원 효과가 가려진다. 0.85는 가드밴드 — `a·K = π`면 `sin(π·d) = 0`
> (정수 d)이라 그 열의 sin 항이 항등적으로 소멸한다.
> config: `aggregator_covariance_slopes: [a, b]` (기본 `null` = 역사적 `(0.019, 0.011)`).

**P는 데이터도 라벨도 보지 않는다.** §69에서 label-free 선택 방법 8종(랜덤/PCA/Sobol QMC/
앨리어싱/사다리 간격/대역폭/위상)을 전부 시험했으나 **모두 0.68 ± 0.03**이었다. 라벨을 보는
사영(learnable)이 아직 시험되지 않은 유일한 축이다.

---

## 3. 분기 CV-1 — closed-form ridge (학습 안 됨)

```text
입력: context sketch (C, K(K+1)/2) + context 라벨 (C,) + query sketch (Q, ...)

  중심화·rms 정규화 (context 통계만)
  클래스 균형 가중치  w_i = 1 / |class(i)|
  가중 중심화로 intercept 제거
  gram = X Xᵀ                      (C × C, **dual/kernel 형태**)
  coefficients = solve(gram + λI, Y)
  → query logits (Q, 2)
```

- **매 에피소드·매 폴드마다 새로 푼다.** 학습으로 얻어 고정하는 W가 없다.
- 학습되는 파라미터는 **`covariance_ridge_log_lambda`, `covariance_ridge_log_scale` 스칼라 2개**뿐.
- dual이라 푸는 계의 크기가 feature 수가 아니라 **context bag 수**(er_status 기준 133)다.
- 단독 AUROC **0.9052** (합성 200 에피소드, 전체 모델 0.9199) — 성능의 대부분이 여기서 나온다.

---

## 4. 분기 CV-2 — subspace + prototype (유일한 학습 모듈)

```text
[1] 클래스별 평균 공분산의 차이     delta = mean(M | y=1) − mean(M | y=0)
[2] whitening (shrinkage 0.25)     W = pooled^(−1/2),  operator = W·delta·W
[3] |고윳값| 상위 rank개 방향 선택   filters = W · eigenvectors[:, top-rank]
[4] 그 방향의 로그분산              f_b = log( filtersᵀ M_b filters )     (rank,)
[5] context 통계로 z-score          z_b = (f_b − center)/scale
                                     ⚠️ query도 **context의** center/scale을 쓴다 (누출 방지)
[6] prototype = 클래스별 z 평균      p_0, p_1
[7] d_c = ‖z_q − p_c‖² / dispersion_c,  sep = ‖p_1 − p_0‖
[8] logits = MLP([d_0, d_1, d_0−d_1, sep])        ← 학습되는 유일한 모듈
```

- [1]~[3]도 **매 에피소드 새로 계산**한다(학습 안 됨). 공분산판 LDA에 가깝다.
- **`subspace_rank`가 정보량을 결정한다**: 기본 1이면 K×K 성분이 **스칼라 1개**로 압축된다.
- ⚠️ **rank를 올려도 파라미터 shape은 불변**이다(실측: rank 1/2/4 모두 43.199M, state_dict 동일).
  [7]의 `.square().mean(dim=-1)`이 rank 축을 평균내 없애 MLP 입력이 항상 4개이기 때문이다.
  rank는 그 4개 값이 **어떻게 계산되는지**만 바꾼다.
- 단독 AUROC **0.8867** — 스칼라 1개로 CV-1(2,080개 사용)에 거의 맞먹는다.

---

## 5. 무엇이 학습되는가 (전체)

| 위치 | 파라미터 | 크기 |
|---|---|---|
| CV-1 | `covariance_ridge_log_lambda`, `_log_scale` | 스칼라 2 |
| 융합 | `covariance_residual_logit` | 스칼라 1 |
| CV-2 | `covariance_relation_head` MLP | (32×4) + … |
| (미사용) | 삭제된 분기의 파라미터 — CV-only에서 gradient 없음 | — |

**사영 P와 ridge 계수는 학습되지 않는다.** 그래서 합성 val AUROC가 ep0 0.8885 =
ep49 0.8882로 평평한데 CE만 개선된다(§69-6) — 학습의 상당 부분이 **logit 스케일 보정**이다.
다만 er_status는 ep0 0.6617 → ep49 0.6989로 오르므로 **50 epoch은 필요하다.**

---

## 6. CV-only 구현 계약

> [!IMPORTANT]
> **죽은 key는 zeros가 아니라 부재다.** `_validate_representation`이 CV-only 전용의 더 작은
> 계약(`covariance_sketch`, `covariance_matrix`만)을 강제하고, 빠지거나 남으면 `ValueError`,
> 실수로 읽으면 `KeyError`다. 새 소비처에서 이 모드가 KeyError를 내면 **그게 정상 동작**이니
> 0으로 채우지 말고 분기를 가드할 것. 이 설계가 실제로 소비처 3곳을 잡았다(§68-4).
>
> **`meta_bag_aggregation`(v37 계열)은 CV-only에서 무의미하다** — bag token 소비처가 전부
> 없어 weight maker가 계산조차 되지 않는다. CV-only는 v36~v39 계보와 무관한 모델이다.

**연산 skip** (출력만 0으로 만드는 게 아니라 계산 자체를 건너뜀):
context pool 통계, 전 셀 poolz_l2 표준화, per-episode anchors(top-k), slot assignment/
MLA affinity/encoder, tails, slot_metadata, global_summary, raw-stat, class memories,
population attention, rare. 근거는 **`centered_delta`가 pool 통계에 의존하지 않는다**는 것.

| | 전 분기 | CV-only |
|---|---|---|
| 훈련 `forward_episode_batch` | 16.91 ms | **2.85 ms** |
| 평가 ragged forward | 19.20 ms | 9.39 ms |
| peak VRAM (60bags×16384cells) | 50,527 MiB | **14,720 MiB** |
| epoch 시간 | 98s | **60s** |

⚠️ dense/ragged 두 경로가 동일해야 한다.
`tests/test_ridge_ablation.py::TestCovarianceOnly`가 전 분기 모델의 covariance 항과의
**등가성**까지 고정한다(첫 구현이 2.4e-2 어긋난 것을 이 테스트가 잡았다).

---

## 7. 주요 config 손잡이

| key | 기본 | 의미 |
|---|---|---|
| `meta_covariance_only` | `false` | CV-1+CV-2만 남김 |
| `aggregator_covariance_sketch_dim` (K) | 64 | 사영 차원. sketch 길이 = K(K+1)/2 |
| `aggregator_covariance_matrix_dim` | 32 | CV-2가 보는 차원. **`null` = K 연동** |
| `aggregator_covariance_slopes` | `null` | `[a, b]`. `null` = 역사적 (0.019, 0.011) |
| `covariance_relation.subspace_rank` | 1 | CV-2 부분공간 차원 (중첩 dict → `model_overrides`) |
| `aggregator_covariance_shrinkage` | 0.1 | 상관행렬 shrinkage |
| `covariance_relation.subspace_shrinkage` | 0.25 | CV-2 whitening shrinkage |

---

## 8. Source of Truth

- 모델: `src/models/baseline.py` — `StructuredEpisodePopulationAggregator`(사영·sketch),
  `StructuredPopulationMetaClassifier._covariance_only_forward`(CV-1·CV-2·융합)
- 진단: `scripts/diagnose_branch_contributions.py`(분기별 단독 AUROC),
  `scripts/diagnose_covariance_sketch.py`(기저 진단)
- 평가: `scripts/eval_seal_tasks.sh` (SEAL 10개 task)
- 테스트: `tests/test_ridge_ablation.py`, `tests/test_covariance_sketch_knobs.py`
