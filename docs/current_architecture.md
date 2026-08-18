# Current architecture (2026-08-18)

**활성 구성 v110 — 학습 파라미터 0, 완전 결정론적.** 아래 §0이 현행 명세이고, 그 뒤의
`Historical-*` / `A` / `B` 절은 **학습을 포함하던 직전 계보(v83~v98)** 의 명세로 참조용이다.
그 절들의 gradient·학습 모듈·checkpoint 계약은 **v110에 적용되지 않는다.**

---

# §0. v110 명세 (활성)

## 0-1. 한 눈에

```
입력    fold의 context bag(라벨 있음) + query bag(라벨 없음). bag = [cells, 1536] UNI2 타일.
        전 과정에서 query는 기저·토큰·정규화 통계에 절대 들어가지 않는다.

기저 B  context cell을 bag별 자기 평균으로 센터링해 풀링한 공분산의 상위 256 고유벡터
        (1536×256). between-slide 항을 버린다 — 그게 nuisance이기 때문(§139-4, §123-4).

CV      descriptor = triu(BᵀC_bag B)의 **비대각 32,640차원만**
        → 클래스 균형 dual ridge (λ=1) → logit 2개
        ⚠️ 대각 256차원과 raw bag mean 1,536차원은 **뺐다** (§156)

DD      같은 triangle(**전체**, 대각 포함)에서 K×K 공분산을 재구성 →
        rank-1 분산 방향 1개 → log(uᵀC_bu)의 클래스별 1-D 가우시안 →
        정규화 제곱 **거리** 2개 (logit이 아니다 → head 계수가 음수)

CT      bag당 64 cell 등간격 → B의 상위 **32 PCA 방향**으로 사영 → context로 좌표 표준화 →
        farthest-point 32개를 초기값으로 **k-means(Lloyd 30회)** → soft assignment →
        bag별 32차원 abundance → 클래스 균형 ridge → logit 2개

head    margin = 1.442·(CV1−CV0) − 0.343·(D1−D0) + 0.7·(CT1−CT0)
        logits = (−margin/2, +margin/2)
```

**실행**: `bash scripts/eval_v110.sh <gpu> <tag> [tasks...]`
**구현**: `src/models/training_free.py` (파라미터 0). 정식 채점은 환경변수로 기존 경로를 써도 된다 —
`tests/test_training_free.py`가 두 경로의 등가성을 고정한다.

## 0-2. 수치

| | SEAL 10 | 홀드아웃 7 | seed std |
|---|---:|---:|---:|
| **v110** | **0.7070** | **0.6103** | **0.00000** |
| v109 | 0.7027 | 0.6042 | 0.00000 |
| v108 | 0.6967 | 0.5893 | 0.00000 |
| v107 | 0.6945 | 0.5836 | 0.00000 |
| v106 | 0.6864 | 0.5767 | 0.00000 |
| 참고: ABMIL(지도학습) | 0.727 | — | — |

⚠️ **결정론적이므로 t·p·CI를 쓰지 않는다**(§151-1). 판정은 **부호 일치 수**와
**독립 집단 재현**(SEAL 10 / 홀드아웃 7)으로 한다.

## 0-3. 각 상수가 왜 그 값인가

| 값 | 근거 |
|---|---|
| 기저 = within-slide PCA | pooled 대비 +0.0020. between-slide는 ICC 31.6% nuisance (§139-4, §123-4) |
| K = 256 | 64~768 스윕의 plateau 중 가장 싼 지점. 홀드아웃에서 재현되는 유일한 값 (§142) |
| CV = 비대각만 | mean은 무용(+0.0019), 대각은 유해(+0.0052, 13/17) (§156) |
| DD = rank-1 | r>1은 이득 없음, K는 128에서 포화, \|t\| 게이트·selector 모두 패배 (§145~§147) |
| CT = 32 PCA 차원 | raw 1536은 거리 집중(rel_std 0.229 vs 0.368) (§149) |
| CT = k-means 30회 | FPS는 32개 중 사실상 2개만 사용 (§157) |
| CT = 32 token | 16·32·64·128 중 정점, 15/17 (§160) |
| CT = 64 cell | 전체 cell은 **모든** token 수에서 손해 (§159, §160-3) |
| head = 3 상수 | 라벨 반대칭이 SEP 가중 0과 쌍의 등가·반대를 강제 (§137-3) |
| CT weight 0.7 | k-means token에서만 성립. FPS token에서는 부호가 무너졌다 (§157-5, §151-2) |

## 0-4. ⚠️ 구조적 제약 — 건드리기 전에 알아야 할 것

1. **DD는 CV descriptor의 triangle을 읽는다.** descriptor를 전역으로 마스킹하면 CV를 좁히는 게
   아니라 **DD를 부순다**. CV 마스킹은 `_normalize_descriptors` 안에서만 걸어야 한다 (§156-1).
2. **DD의 K는 CV의 K를 넘을 수 없다** — 부분블록을 읽기 때문이다 (§145-1).
3. **CT와 CV가 같은 기저를 공유한다.** CT 개선이 macro에 잘 안 닿는 상한 요인이다 (§149-4).
4. **에피소드마다 상수를 더하는 변경은 fold-mean AUROC를 움직일 수 없다** (§154-5).
5. **cell↔token 거리는 cell 축으로 chunk된다**(2²⁷ 원소). chunking은 원소별 산술을 안 바꾸므로
   정확하고, v110 설정은 단일 chunk에 들어간다 (§160-1).

## 0-5. 닫힌 축 (재시도 금지)

| 축 | 결과 |
|---|---|
| 합성 데이터 분포 | 격차를 닫을수록 단조로 나빠진다 (§129) |
| DD 전반 | K·rank·게이트·selector 네 갈래 모두 (§145~§147) |
| CT cell 샘플링 | 전체 cell이 4개 token 수 전부에서 손해 (§159, §160) |
| CT two-token readout | ridge로 대체됨. 단 raw 1536에서만 "무관"이었다 (§148·§150) |

## 0-6. 열려 있는 갈래

- **CV 비대각 32,640차원의 가중** — 지금은 무가중 ridge다. 학습을 넣는다면 여기다 (§156-6).
- **CT에 CV와 다른 부분공간** — 대각/스펙트럼 쪽을 주면 §149-4의 중복을 깰 수 있다.
- **DD의 코호트 의존성** — DD는 SEAL에서 도움, 홀드아웃에서 해롭다. 에피소드별 게이트가
  가능한지 (§153).
- **λ 재확인** — CT ridge가 16→32차원으로 커졌는데 λ=1 고정이다 (§156-5, §160-5).

---

> [!IMPORTANT]
> **활성 baseline은 v83 linear head의 4 seed × `epoch 49` checkpoint**
> (`CovarianceMeanLearnablePDDCTMLPModel`, SEAL **0.6880** = 1-GPU 4 seed 평균, §109)다 —
> `checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/`, tags `v83_linear_head_seed4{2..5}_ep49`,
> config `configs/train_v83_linear_head_1536_1gpu.yaml`.
>
> **⚠️ 이 승격은 §107-3 판정 게이트를 충족하지 못한 상태에서의 사용자 결정이다(§108→§109).**
> v82 대비 seed-paired Δ +0.0045, t≈1.15 — seed 44가 부호 반전(3/4 양수)이라 4/4 부호 일치도
> `|t|≥2.5`도 아니다. 인용할 때 "미판정 상태에서 승격됐다"고 밝힐 것.
>
> **⚠️ 모델은 v82/v77과 거의 같다 — 유일한 차이는 relation head 구조다.** `ct_head_hidden_dims: []`로
> hidden layer와 GELU를 없애 head가 `12→32→1`(GELU)에서 bare **`Linear(12,1)`**로 줄었다.
> 클래스·텐서 구조(head 제외)·`architecture_version=54`는 그대로이지만 **head shape가 달라
> v82/v77 checkpoint는 strict-load되지 않는다** — 처음부터 학습했다. 이 문서의 모델 명세(head
> 입력 12개 feature 등)는 v83에도 그대로 적용되고, head 자체의 구조만 아래처럼 다르다.
>
> **⚠️ 0.6880 < 이전 0.6880(v77 DDP4)은 숫자만 같은 별개 레짐의 값이다** — 착시로 "제자리로
> 돌아왔다"고 읽지 말 것. 판정 레짐 자체(1-GPU 4 seed)는 §107에서 DDP4 1 seed로부터 전환된 그대로
> 유지된다. 1-GPU는 같은 config에서 DDP4보다 −0.0098이고, 같은 레짐에서 v77 Hard는 0.6781,
> v82 Medium(직전 baseline)은 **0.6835**다.
> 옛 DDP4 baseline은 `v77_hard_ep49` 0.6880
> (`checkpoints/20260812_v76_classsep_sweep/hard/periodic-epoch=049-val_ce_loss=0.1717.ckpt`,
> val-best epoch 48은 0.6873)이며 **역사적 기록**이다.
> 채점은 계속 **epoch 49 고정**이다(§104).
> centered cells를 learnable orthogonal P(1536×128)에 사영해 만든 covariance를 CV와 DD가
> 공유하고, CT와 함께 12개 relation feature를 만들어 이제는 **bare `Linear(12,1)`**이 읽는다
> (v82/v77은 `12→32→1` GELU였다).
> 학습 파라미터는 P 196,608개 + head 13개 = **196,621개**(v82/v77은 head 449개, 합계 197,057개).
> 기본 v77/v82에서 P는 CV ridge 경로로만 학습되며 DD/CT는 training-free다 — v83도 동일. ridge-calibration
> arm은 두 스칼라를 추가했으나 SEAL 0.6840으로 baseline을 넘지 못했다(v82 기준 값, historical).
>
> **v78 (`train_dd_projection`) — 기각, 그리고 방향까지 확정됐다.** weight 0/0.02/1.0에서
> 0.6873/0.6869/**0.6826**으로 **단조 악화**하고 무가중은 CI가 0을 제외한다(§103, G-5).
> DD는 P를 실제로 움직이지만 그 방향이 해롭다. **되살리지 말 것.**
>
> **v79 (`DualProjectionCVDDCTMLPModel`, version 56) — 기각.** 중재 대신 **분리**를 택했으나
> Δ **−0.0105** [−0.0137, −0.0074]로 세 arm 중 가장 나빴다. 상세와 두 진단은 **Active-6**.
> ⚠️ v78·v79가 함께 말하는 것: **headroom은 CV/DD·사영 배선에 없다**(Active-6 마지막 절).
>
> DD의 rank-1 방향은 **어느 arm에서도 미분하지 않는다** — 이유와 우회 방법은 **G-4**.

리포에는 활성 relation 계보와 역사적 비교 계보 두 개가 있다. relation 계보와 Encoder 계보는
`src/models/set_transformer_ridge.py`에 있고, 역사적 CV-only는 `src/models/baseline.py`에 있다.
공통 핵심 유틸리티는 episode-local ridge의 `solve_ridge_system`이다.

| | Active. Relation v83 | A. CV-only | B. Encoder+Ridge |
|---|---|---|---|
| 파일 | `set_transformer_ridge.py` | `baseline.py` | `set_transformer_ridge.py` |
| 클래스 | `CovarianceMeanLearnablePDDCTMLPModel` | `BaseModel` | `SetTransformerRidgeModel` |
| bag 기술자 | learnable P covariance + raw mean | fixed P covariance | learned Transformer |
| readout | CV/DD/CT 12→1 (bare linear, §109) | CV-1 ridge + CV-2 | episode-local ridge |
| 학습 파라미터 | 196,621 (v82/v77은 197,057) | 229 | 5,010,946 |
| SEAL 10개 최고 | **0.6880** (1-GPU 4 seed, §109) | **0.6940** (v41_K128, DDP4 1 seed) | 0.6619 / 0.6526 |
| 상태 | **활성 baseline** (§108·§109, 게이트 미달 상태의 사용자 결정) | 역사적 전체 최고 — 레짐이 달라 직접 비교 불가 | 기각 |

이전 세대(v34~v39의 6-분기)는 **소스에서 삭제**됐다(§73). 필요하면 git `8caa96c`.

---

# Historical. v83 linear-head learnable-P CV+DD+CT relation model

⚠️ **v110에 적용되지 않는다.** 아래는 학습을 포함하던 직전 계보의 명세다 — gradient 계약,
학습 모듈, checkpoint 호환성은 전부 그 시절 것이다. 현행 명세는 §0.

v83 승격은 §109에서 사용자 결정으로 이뤄졌으며 **§107-3 판정 게이트(4/4 시드 부호 일치 +
|t|≥2.5)를 충족하지 못한 상태의 승격**이다(§108: v82 대비 Δ+0.0045, t≈1.15) — 인용 시 이 점을
함께 밝힐 것. v82에서 바뀐 것은 relation head 구조 하나뿐이다 — `ct_head_hidden_dims: []`로
hidden layer와 GELU를 없애 head가 `12→32→1`(GELU)에서 bare `Linear(12,1)`로 줄었다
(trainable 197,057 → 196,621). 텐서 구조(head 제외)는 v76 이후 동일하므로 모델 클래스의
내부 `architecture_version=54`는 유지하지만, **head shape가 달라 v82/v77 checkpoint는
strict-load되지 않는다**(`tests/test_relation_head_depth.py`가 이 실패를 pin한다) — 처음부터
학습했다. canonical config는 `configs/train_v83_linear_head_1536_1gpu.yaml`(self-contained),
canonical checkpoint는 **4 seed × epoch 49** `checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/`다.
직전 baseline v82의 canonical checkpoint는 `checkpoints/20260813_v82_medium_seeds/seed4{2..5}/`
(1-GPU 4 seed 0.6835, historical)이고, 그 이전 v77의 canonical checkpoint는
`checkpoints/20260812_v76_classsep_sweep/hard/periodic-epoch=049-val_ce_loss=0.1717.ckpt`다.
과거 `PopulationTokenResidualModel`은 이제 **retired provisional v77-pop-residual**로만 부르며,
그 클래스의 내부 version 55 역시 과거 checkpoint replay를 위해 유지한다.

## Historical-1. Forward 경로

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

## Historical-2. Gradient와 학습 계약

활성 baseline v83에서 학습되는 것은 다음뿐이다.

| 파라미터 | 개수 | gradient 경로 |
|---|---:|---|
| `_covariance_projection` P | 196,608 | CV ridge solve를 통과 |
| `cv_dd_ct_head` (bare `Linear(12,1)`) | 13 | 12개 relation feature. ⚠️ **§138-4에서 상수 3개로 대체 가능함이 확정됐다**(정식 경로 Δ−0.0003) — 라벨 반대칭이 `w(SEP)=0`·`bias=0`을 강제하고 차분 feature는 쌍의 선형결합이라 선형 head에 표현력을 안 더한다. `margin = 1.442·(CV1−CV0) − 0.343·(D1−D0) + 0.286·(q1−q0)`, 8 seed std 0.027/0.008/0.012. 재현: `ICF_FIXED_HEAD=1` |
| **합계** | **196,621** | |

v82/v77(head `12→32→1`, GELU)은 head가 449개라 **합계 197,057**이었다 — 두 세대의 유일한 차이는
head 파라미터 수뿐이고 P·gradient 경로는 동일하다.

- 매 forward에서 thin QR로 `P_effᵀP_eff=I`를 보장해 임의 scale/conditioning 변화를 차단한다.
- DD는 현재 P로 만든 covariance를 읽지만 기본 v77에서 DD→P gradient는 차단된다. CT 선택/특징은
  어느 arm에서든 training-free다.
- **v78 opt-in (`train_dd_projection`) — 기각됨.** DD의 이차형식만 그래프에 남겨 P가 DD 목적도
  반영하게 하는 플래그다. 파라미터 0개 추가·shape 보존이라 version 54와 strict-load 호환은
  양방향 유지되지만, weight 스윕이 **단조 악화**로 끝났다(G-5). 되살리지 말 것.
  rank-1 방향 `f`를 왜 미분할 수 없는지는 **G-4**.
- 기본 `ridge_log_lambda=log(1)`, `ridge_log_scale=log(2)`는 v77에서 동결된다.
- `train_ridge_calibration: true`는 이 두 scalar만 동결 해제한다. 이 opt-in arm은 197,059개이며
  기존 checkpoint/config의 학습 계약은 바꾸지 않는다.

## Historical-3. 현재 synthetic data 계약

활성 baseline(v83 linear head, 데이터 계약은 v82 Medium과 동일) 실험의 공통 조건은 다음과 같다.

| 항목 | 값 |
|---|---|
| bags/episode | 60–100 |
| raw cells/bag | 256–8,192, bag별 독립 log-uniform(power 2.0) |
| 실제 training cap | 4,096; 초과 시 매 step random subsample |
| latent/output | 32 / 1,536 |
| shared populations | 4–10, fraction 0.82–0.96 |
| response tasks | composition/state/covariance/interaction/combined 동일 확률 |
| ClassSep | **Medium `[0.5,1.4]`** (v82, §107). Hard `[0.2,0.8]`은 직전 baseline |
| observation noise | 0.005 — ⚠️ **정규화지 nuisance가 아니다**. 끄면 합성 val_ce 0.152→0.099로 쉬워지고 SEAL은 −0.0076(v100, §127-3) |
| donor shift | 0.35 — bag마다 latent 벡터 하나를 그 bag 전 cell에 더하는 강체 이동. 축 전체가 노이즈 안에서 평평(§128-2) |
| training | **1-GPU × 4 seed(42–45)**, GPU 0–3에 하나씩, bf16, 50 epochs (§107) |

### Historical-3z. ⚠️ 서브샘플링 계약 — 학습과 평가가 다르다 (§138-1)

**평가**: relation 계보(v83~)는 **tile 서브샘플링을 하지 않는다.** `eval_seal_tasks.sh`가
`--max-tiles`를 안 넘기므로 스크립트 상한은 `None`이고, 모델의 `max_cells: 8192`는
**`BagTokenEncoder.forward`에만** 있어(`_subsample`) descriptor 경로(`_covariance_descriptors` /
`_bag_means`)는 그것을 타지 않는다. 실증: 같은 설정 재실행이 저장값과 소수 넷째 자리까지 동일하다
(`_subsample`이 발동했다면 per-call randperm으로 값이 달라진다). `ct_cells_per_bag: 64`는
farthest-point라 결정론적이다.

**학습**: 상한이 셋이고 서로 독립이다.

| 위치 | knob | 값 | 성격 |
|---|---|---|---|
| 데이터셋 | `per_bag_max_cells` | 4096 | 에피소드 추출 시 무작위 subsample |
| collator | `padding_max_cells` | 4096 | dense padding 전 무작위 subsample (§120-1) |
| 모델(encoder) | `max_cells` | 8192 | **relation 계보 미발동** |

⚠️ **결과적으로 학습은 bag당 ≤4,096 cell, 평가는 full tile(중앙값 4,988~7,736, 최대 35,107)을
본다.** 의도된 설계가 아니라 계보가 진화하며 남은 비대칭이고, §123-3의 "cell 축이 실제보다 짧다"의
정확한 출처다. **이 비대칭 자체는 아직 arm으로 검정된 적이 없다.**

### Historical-3a. 이후 추가된 생성기 knob (전부 기본값 inert)

| knob | 기본 | 하는 일 | 판정 |
|---|---|---|---|
| `class_prior` | `None` | 에피소드마다 `p~U(lo,hi)` 뽑아 `Bernoulli(p)`. `None`이면 기존 `torch.randint` 스트림과 **bit-identical** | **기각** (v90, §118) |
| `spectral_tail_{dim,decay,scale}` | 1024 / 0.5 / **0.0** | manifold map 뒤에 `k**-decay` 감쇠 공분산의 nuisance를 더해 스펙트럼 꼬리를 만든다. `scale=0`이면 완전 inert | **기각** (v95/v102, §129) |
| `spectral_tail_bag_fraction` | **0.0** | 위 꼬리를 bag 공유분/cell 개별분으로 분할(`√(1−f)`/`√f`, cell별 총분산 보존). 스펙트럼과 응집도를 **하나의 기전**으로 동시에 만든다 | **기각** (v102, §129) |
| `data.padding_max_cells` | **4096** | dense 학습 collator의 per-bag 상한. §120-1에서 하드코딩 상수를 노출한 것 | 중립(인프라) |

⚠️ **bag 크기를 키우는 arm은 cell 상한 3개를 전부 올려야 한다** — `per_bag_max_cells`(데이터),
`max_cells`(모델 subsample), `padding_max_cells`(collator). 하나라도 빠지면 **에러 없이 조용히
잘린다**(§120-1). 테스트: `tests/test_padding_max_cells.py`, `tests/test_spectral_tail.py`,
`tests/test_class_prior.py`.

⚠️⚠️ **§129 결론: 이 데이터 계약을 실제 UNI2 통계에 "맞추려는" 변경은 하지 말 것.**
격차는 §123이 정확히 실측했으나(스펙트럼 r90 47 vs 585, 응집도 cosine 0.130 vs 0.351,
norm sd 0 vs 2.56), **닫을수록 단조로 나빠진다.** 재현 진단: `scripts/diagnose_synthetic_vs_real.py`.

`manifold_mode`은 실험 축이다.

- `orthogonal`: episode마다 fresh isometric linear map. **활성 baseline이 쓰는 값**이다 —
  v83(linear head) Medium epoch 49 = **0.6880**(1-GPU 4 seed, §109). 직전 v82(GELU head)는
  같은 데이터로 **0.6835**(§107). Hard DDP4 1 seed는 0.6880이었다(§104, 숫자만 같은 별개 레짐).
- `mlp_bank`: 고정 3-layer MLP를 bank ID seed로 재생성. **epoch 49 통일 후**
  M=128/512/1024/2048/4096 = 0.6734/0.6730/**0.6780**/**0.6776**/0.6678 —
  1024·2048이 고원이고 4096에서 −0.0102 하강한다(§105-5). 재채점 전 값은
  0.6697/0.6726/0.6779/0.6751/0.6649였고 2048·4096이 epoch 28·27에서 채점된 탓이었다.
- `nonlinear`: episode마다 fresh MLP (= bank size 무한). **Hard에서 기각** — `mlp_num_layers: 2`
  (`[32→96→1536]`, GELU 1개, 가장 얕은 진짜 MLP)로 4 seed 평균 **0.6722**, Δ −0.0158 (§104-6).
  ClassSep baseline 시절 v72(3-layer)의 0.6709와 방향이 같다. ⚠️ `mlp_num_layers: 1`은
  GELU가 하나도 없어(`_map_to_manifold`가 마지막 층 활성을 건너뛴다) MLP가 아니라
  비-isometric 선형사상이다 — arm으로 쓰지 말 것.
- `mixed_linear_mlp_bank`: episode마다 mapping 하나를 선택한다. 50% fresh linear + 50% MLP-1024는
  synthetic val CE 0.2218로 좋아졌지만 SEAL은 0.6755로 하락해 기각했다.

중요한 역사적 차이: v40/v41 데이터는 episode-level cardinality `[1,16384]`, cap 없이 fresh
3-layer nonlinear MLP를 썼다. 현재 Hard는 per-bag cardinality와 4,096 cap을 쓰므로 v41과 v76의
성능 차이를 아키텍처만의 차이로 해석하면 안 된다.

## Historical-4. 현재 실험과 판정

- 활성 실행: 없음.
- **활성 baseline = v98 (`donor_shift_scale` 0.15), 1-GPU 8 seed 평균 0.6852** (§131).
  ⚠️ v83(4 seed 0.6880)과 절대 수치를 빼지 말 것 — 서로 다른 시드 집합이다. 새 arm은 v98의 같은
  시드와 seed-paired로 비교한다.
- ⚠️⚠️ **4 seed의 최소 검출 효과는 0.0121이다(§131-2).** 게이트 |t|≥2.5는 df=3에서 p=0.088.
  **"미판정"을 "효과 없음"으로 쓰지 말 것.**
- ⚠️ **판정 절차가 §118에서 바뀌었다**: §107-3 게이트(4/4 부호 일치 + |t|≥2.5)는 계속 계산·보고하되
  **통과 여부가 승격/기각을 자동 결정하지 않는다.** 최종 판정은 macro + **task 10개 전부**를
  baseline 성능대와 함께 본 패턴 + 다른 arm과의 일관성을 종합한 **사용자 판단**이다.
  보고 형식: (i) arm이 뭘 테스트하는지 → (ii) task 10개 전부 표 → (iii) macro Δ+t (§118-3).
- ⚠️ **1 seed 스크리닝은 배제에만 쓴다** — v94가 seed42에서 +0.0022였으나 4 seed로는 −0.0043이었다
  (§125-1).
- **분산 감소가 현재 최대 레버다 (§130)**: 시드 앙상블이 학습 비용 0으로 +0.0058(v83)~+0.0071(v98).
  model soup(가중치 평균)은 +0.0014로 실패했고, arm 다양성은 기여하지 않는다.
- **v78·v79 모두 기각.** weight 0/0.02/1.0에서 0.6873/0.6869/0.6826(G-5), v79 분리는 0.6768
  (Active-6). 세 방식 모두 지고 건드린 정도가 클수록 더 졌다 — 이 축은 소진으로 본다.
- active baseline: **v83 linear head orthogonal, 1-GPU 4 seed 평균 epoch 49 = 0.6880** (§109).
  ⚠️ **§107-3 게이트를 충족하지 못한 상태의 사용자 결정 승격이다** — v82 대비 Δ+0.0045, t≈1.15,
  3/4 시드 양수(§108). 직전 v82 Medium(GELU head)은 같은 레짐에서 0.6835, v77 Hard는 0.6781,
  DDP4 1 seed로는 0.6880이었다 — **레짐이 다른 숫자를 빼지 말 것**(§107-1). v83의 0.6880과 v77
  DDP4의 0.6880은 숫자만 같은 별개 값이다.
- learned ridge λ/logit scale은 0.6840으로 기각했다. ⚠️ Δ −0.0033은 seed std 0.0051 미만이라
  **§104-4에서 "판정 불가"로 내려갔다**.
- v80 shallow infinite MLP manifold는 4 seed 평균 0.6722로 기각. ⚠️ Δ는 −0.0158이 아니라
  **−0.0059**다(t=−2.7) — 기존 값은 1-GPU arm을 DDP4 baseline과 비교한 탓에 부풀려졌다(§106-4).
- **learnable P vs fixed P는 미판정이다**(§106-3): 같은 Hard·같은 레짐·4 seed에서 +0.0048, t=1.5,
  seed 하나는 부호 반전. v81(`CovarianceMeanDDCTMLPModel`, 449 파라미터)이 그 fixed-P arm이며
  초기 시점에 v77과 수치적으로 동일함을 확인했다(P 차이 3.6e-07).
- 판정: **epoch 49 고정** checkpoint의 공식 SEAL 10-task macro. **task별 regression은 판정
  근거로 쓰지 않는다**(시드만 달라도 task 6개 CI가 0을 제외한다, §104-5). synthetic val 지표는
  checkpoint 선택에만 사용한다. macro 비교는 점추정 차이가 아니라 **fold-paired Δ + bootstrap
  CI**로 한다 — `scripts/compare_arms_paired.py` (§99).
- GPU 정책: ICF는 GPU 0–3만 사용하고 4–7은 사용하지 않는다.

## Historical-5. Large-bag ragged fine-tuning 계약

`data.ragged_training: true`는 `episode_batch_size=1`에서만 허용되며, training collator가
list-of-bags를 padding tensor로 바꾸지 않고 그대로 모델의 ragged forward에 전달한다. 기본값은
false이므로 기존 dense masked training 계약은 유지된다. 현재 large-bag arm은 bag별
`[2048,16384]`, `per_bag_max_cells: 16384`를 사용해 4,096 cap을 제거한다.
대형 CUDA generator buffer와 ragged forward의 중첩을 피하기 위해 이 arm은
`cuda_prefetch: false`를 사용한다.

이 arm은 scratch가 아니라 동일 Hard orthogonal v77 best checkpoint
`checkpoints/20260812_v76_classsep_sweep/hard/epoch=048-val_ce_loss=0.1697.ckpt`를
`--init-checkpoint`로 weight-only load한다. optimizer/scheduler/epoch state는 새로 시작한다.
epoch 34 best의 공식 SEAL macro는 **0.6885**로 baseline 대비 +0.0012에 그쳐, large-ragged
파생 실험으로 유지하고 canonical baseline으로 승격하지 않는다.

## Historical-6. v79 dual projection — **기각** (`DualProjectionCVDDCTMLPModel`, version 56)

v77은 learnable P 하나를 CV와 DD가 공유하므로 subspace가 CV의 readout에 맞춰 최적화되고 DD는
그것을 물려받는다. v78은 그 갈등을 **gradient weight로 중재**하려다 단조 악화로 기각됐다(G-5).
v79는 중재하지 않고 **공유를 끊는다**. branch 4개가 각각 4개 feature를 낸다.

```text
                                       gradient
CV(learnable P)  CV0,CV1,CV1-CV0,SEP_CV   → P (유일한 경로)
CV(fixed P)      CV0,CV1,CV1-CV0,SEP_CV   없음 (training-free)
DD(fixed P)      D0,D1,D1-D0,SEP_DD       없음 — CV의 subspace를 더 이상 타지 않는다
CT               q0,q1,q0-q1,SEP_CT       없음 (사영 자체가 없음)
                                    = 16 → 32 → 1
```

fixed-P CV를 **독립 evidence block으로 남기는 것**이 설계의 두 번째 요점이다. fixed P는 단순히
옛 기본값이 아니라 **v41_K128이 0.6940을 낸 기저**이고 그것이 여전히 역사적 전체 최고다. head가
학습된 subspace와 고정 subspace를 **저울질**하게 하는 것이지, 학습된 것이 고정된 것을 대체하게
하는 것이 아니다.

| 항목 | 값 |
|---|---|
| descriptor | `[cov_learnable 8,256, mean 1,536, cov_fixed 8,256]` = **18,048** |
| 정규화 | 세 block 각각 독립 context-only center/scalar-RMS (canonical CV 계약) |
| raw bag mean | 사영과 무관하므로 두 CV branch가 **공유** |
| trainable | P 196,608 + head 577 = **197,185** |
| `architecture_version` | **56** — v77 ckpt와 strict-load **비호환** |
| config / runner | `train_v79_dual_projection_1536.yaml` / `run_v79_dual_projection.py` |

- fixed 사영은 `_fixed_covariance_projection` **buffer**(`persistent=False`)다. `super().__init__`
  직후 learnable Parameter가 아직 sin/cos 기저와 같으므로 그 시점에 snapshot한다 — 재생성 불필요,
  ckpt 용량 증가 없음.
- `train_dd_projection`은 **ValueError로 거부**한다. DD가 buffer를 읽으므로 그 플래그는 조용한
  no-op이 될 뿐이다(G-4의 "조용한 실패"를 만들지 않기 위한 가드).
- 구조적 주장 두 개를 테스트가 고정한다: ⓐ P를 흔들어도 `cov_fixed`·`mean` block이 **정확히
  불변**, ⓑ v79의 fixed block이 독립 `CovarianceMeanDDCTMLPModel`의 CV와 **수치적으로 동일**
  (재파라미터화가 아니라 진짜 v41-스타일 CV).
- 판정은 v77 대비 fold-paired Δ + CI(§99). tag `v79_dual_projection_best`.

**결과 — 기각 (§103-4).** SEAL macro **0.6768**, fold-paired Δ **−0.0105** [−0.0137, −0.0074],
상승 3/10. PIK3CA −0.0518, VHL 0.4166(랜덤에서 더 멀어짐), er_status만 +0.0168.

두 진단이 원인을 좁힌다.

- **과소학습이 아니다 — 반대다.** v79 best val_ce **0.1687**로 v77의 0.1697보다 **좋다**. 합성
  val이 개선되는 동안 SEAL이 떨어졌다. 이 리포의 대표 병리 세 번째 재현이다(v54 §79-6,
  mixed manifold §94, v79). 새 16-input head 때문에 초반 val_ce가 높았던 것은 수렴 문제가 아니었다.
- **head가 선택하지 않고 분산시켰다.** 학습된 head 1층의 block별 column norm share는
  CV(learnable) 31.4% / CV(fixed) 26.7% / DD(fixed) 25.5% / CT 16.5%로 거의 균등하다. 두 CV
  branch는 **mean block을 공유하고 covariance 정보도 중복**되는데 head가 16개 상관 입력에 weight를
  퍼뜨렸다. ⚠️ column norm은 거친 대리 지표다(feature 스케일이 달라 곧 기여도가 아니다).

> [!IMPORTANT]
> **CV/DD·사영 배선 축은 소진으로 본다 (§103-5).** v78 balanced → 무가중 → v79가
> **−0.0004 → −0.0047 → −0.0105**로 단조 악화한다. gradient 개방·무가중·완전 분리 세 방식 모두
> 지고 **건드린 정도가 클수록 더 졌다.** 이 축에서 새 arm을 설계하지 말 것. 남은 레버는 task-side
> (VHL 랜덤 이하, BAP1 large-bag 붕괴, branch reliability feature)와 **미측정 seed 노이즈**다.

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
  ⚠️ **이 `[1,8192]`은 group default이고 활성 v77/v78 arm은 `[256,8192]`
  (`num_cells_log_uniform_power: 2.0`)로 override한다** — Active-3 표가 활성 값이다.
  실측 확인은 `merge_train_config(Path("configs/train_v77_hard_orthogonal_1536.yaml"))`.
- training collator는 4,096개를 초과한 bag을 매번 `randperm`으로 4,096개까지 subsample한 뒤
  batch 최대 길이(상한 4,096)까지 zero-padding하고 `cell_mask`/`bag_mask`를 반환한다.
  batch 크기 1도 반드시 이 dense masked 경로를 탄다. validation/test는 결정성을 위해 생성된
  cell 순서의 앞 4,096개를 사용하고 ragged 평가 경로를 유지한다.
- 모델은 padding 값을 데이터로 해석하면 안 된다. 모든 mean/covariance/attention은
  `cell_mask`로 제외하고, padded bag은 `bag_mask`로 context/query에서 제외한다.
- 변경 전과 학습 데이터 분포가 다르므로 재학습 결과를 기존 arm의 연장으로 비교하지 않는다.

## D. 주요 손잡이

활성 계보(v77/v78)의 손잡이가 먼저다. `A`/`B`는 역사적 비교군 전용이다.

| key | 기본 | 계보 | 의미 |
|---|---|---|---|
| `covariance_sketch_dim` (K) | 128 | **Active** | learnable P의 열 수. descriptor = K(K+1)/2 + 1,536 |
| `train_dd_projection` | `false` | **Active** | v78. ⚠️ **기각됨**(G-5) — 켜지 말 것. v79에서는 ValueError로 거부된다 |
| `dd_projection_gradient_weight` | `1.0` | **Active** | v78 전용. 무가중이면 DD가 CV의 52배로 P를 지배한다(G-5) |
| `dual_head_hidden_dim` | 32 | **v79** | 16→hidden→1 relation head의 폭 |
| `ct_head_hidden_dims` | `[]` | **Active (§109)** | relation head 은닉층 폭 리스트. `[]` = bare `Linear(12,1)`(v83, 활성 baseline). v82/v77은 `[32]`(GELU 포함, 197,057 파라미터) — §108이 이 둘을 비교했다(미판정, 사용자 결정으로 승격). `[32, 32]`(v84, 198,113 파라미터)는 §110에서 양쪽 baseline 기준 모두 기각됐다 — 이 이상 깊게 가지 말 것 |
| `train_ridge_calibration` | `false` | **Active** | `ridge_log_lambda`/`ridge_log_scale` 동결 해제. v82/v77 head(449) 기준 197,057 → 197,059. SEAL 0.6840으로 기각(historical, v83 head로는 재측정 안 됨) |
| `dd_shrinkage` | 0.25 | **Active** | DD whitening의 shrinkage. ⚠️ backward의 고윳값 **간격은 바꾸지 않는다** |
| `ct_num_tokens` / `ct_cells_per_bag` | 16 / 64 | **Active** | CT 후보 token 수 / bag당 샘플 cell 수 |
| `class_separation` | `[0.5, 1.4]` | **Active** | 합성 난이도. **Medium이 Hard `[0.2,0.8]`보다 +0.0053 낫고**(4 seed, 4/4 양수, t=3.0, §106-2) **§107에서 baseline이 됐다.** 조이는 것 자체는 `[1.0,2.0]` 대비 +0.011~+0.015로 유효 |
| `manifold_mode` | `orthogonal` | **Active** | `mlp_bank`·`mixed_linear_mlp_bank`·`nonlinear` 전부 기각(Active-3, §104) |
| `mlp_num_layers` | 3 | **Active** | `orthogonal`에서는 미사용. `nonlinear`/`mlp_bank`에서만 유효하고 **weight 행렬 개수**다(1은 활성 없음 = MLP 아님) |
| `data.ragged_training` | `false` | **Active** | `episode_batch_size=1` 전용. Active-5 참조 |
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

- **활성 baseline (v83)**: `src/models/set_transformer_ridge.py` —
  `CovarianceMeanLearnablePDDCTMLPModel` (v77·v82와 동일한 클래스. v83은 `ct_head_hidden_dims: []`
  로 head 구조만 바꾼다 — §109, §107-3 게이트 미달 상태의 사용자 결정).
  DD 방향 격리는 `_dd_direction`(G-4), v78의 gradient 균형은 `_ScaleGradient`(G-5, 기각).
- **직전 baseline (v82, historical)**: 같은 클래스, `ct_head_hidden_dims: [32]`(GELU 포함,
  449 파라미터). v77에서 `class_separation`만 바꾼 데이터 단독 변경이었다.
- **기각 (v79)**: 같은 파일 — `DualProjectionCVDDCTMLPModel`. fixed 사영은
  `_fixed_covariance_projection` buffer, 두 CV branch는 `_cv_branch_features`가 공유한다.
- 역사적 모델 A: `src/models/baseline.py` (2,317줄)
- 역사적 모델 B: `src/models/set_transformer_ridge.py` — `SetTransformerRidgeModel`
- 평가: `scripts/eval_seal_tasks.sh` → `scripts/test_pathobench.py`
- **arm 비교(필수)**: arm과 baseline(v83)을 각각 **1-GPU 4 seed**로 돌려 **seed-paired Δ + t**로
  판정한다(§107-3). 시드별 fold-paired CI는 `scripts/compare_arms_paired.py`로 뽑아 보조 근거로
  쓴다. 점추정 macro끼리 빼서 판정하지 않는다(§99).
- canonical config/checkpoint: `configs/train_v83_linear_head_1536_1gpu.yaml` /
  `checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/` (epoch 49, 4 seed 평균 **0.6880**,
  §109). 직전 baseline: `configs/train_v82_medium_classsep_1536_1gpu.yaml` /
  `checkpoints/20260813_v82_medium_seeds/seed4{2..5}/` (0.6835, historical).
- 테스트: `tests/test_set_transformer_ridge.py`(v78 계약 5개 포함 — forward 동일성,
  `_dd_direction`이 autograd 밖, P gradient 도달, weight 선형성, strict-load 양방향),
  `test_ridge_ablation.py`, `test_cvonly_golden.py`, `test_paired_relation_head.py`,
  `test_relation_head_depth.py`(v83 — head shape 불일치 시 strict-load가 시끄럽게 실패하는지 pin),
  `test_training_uses_dense_path.py`, `test_per_bag_cardinality_padding.py`,
  `test_config_numeric_types.py`, `test_precision_contract.py`
- ⚠️ 기존 실패 1건: `tests/test_mlp_manifold_bank.py`가 BagPFN env에 없는 `pytest`를 import한다.

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

## G. Dispersion Distance (DD) — 명세·미분 불가성·사영 축

> [!IMPORTANT]
> **DD를 건드리기 전에 G-4와 G-5를 먼저 읽을 것.** DD의 rank-1 방향은 **미분할 수 없고**
> (G-4), DD의 gradient를 P에 흘리는 것은 **실측으로 해롭다**(G-5). 두 사실 모두 `Δ≈0`으로
> 조용히 실패할 수 있는 형태다.

### G-0. DD가 읽는 covariance는 어느 P에서 오는가 (arm마다 다르다)

DD는 자기 사영을 갖지 않고 **CV가 만든 covariance를 재사용**한다. 따라서 "DD의 subspace"는
그 arm이 CV에 무엇을 쓰는지에 따라 달라진다. 이 구분이 v77~v79의 차이 전부다.

| arm | DD가 읽는 covariance | DD→P gradient |
|---|---|---|
| v74 | **fixed** P | 학습 P 자체가 없음 |
| v77 / v82 / **v83 (활성 baseline)** | **CV가 학습한 P** | 차단 (`no_grad`) |
| v78 (기각) | CV가 학습한 P | **개방**(이차형식만) + weight — G-5 |
| v79 (기각) | **fixed** P — CV의 learnable P와 분리 | 구조적으로 불가(buffer를 읽음) |

⚠️ v77/v82/v83에서 DD는 **자기 목적으로 최적화되지 않은 subspace를 물려받는다.** 이것이 v78·v79가
겨냥한 문제다.

### G-1. DD는 무엇을 측정하는가

DD(Dispersion Distance)는 canonical CV와 같은 centered projected covariance matrix
`C_b ∈ R^(K×K)`를 사용하지만(**어느 P에서 온 `C_b`인지는 G-0**) upper triangle 전체에 ridge를
푸는 대신, support label로 episode-specific한 rank-1 방향을 해석적으로 만든다.

    C̄_c = mean(C_b | y_b=c)              Δ = C̄_1 - C̄_0
    C_pool = mean_b C_b                   τ = tr(C_pool)/K
    S = 0.75 C_pool + 0.25 τ I            W = S^(-1/2)
    A = W Δ W                             A u = λu
    f = W u,  where |λ| is maximal

`f ∈ R^K`는 전체 pooled dispersion으로 whitening한 뒤 두 클래스의 within-bag variance
차이가 가장 큰 cell projection 방향이다. 원래 1,536-d embedding 공간에서는 **`P f`**이며,
여기서 `P`는 **그 arm이 CV에 쓰는 사영**이다(G-0) — v77은 학습된 P, v79는 fixed P다.
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

### G-4. ⚠️ DD의 rank-1 방향은 어느 arm에서도 미분하지 않는다

`_dd_direction`(`set_transformer_ridge.py`)이 G-1의 방향 계산 전체를 `no_grad`에 가둔다.
**이건 보수적 선택이 아니라 미분이 성립하지 않기 때문이다.** 두 가지가 독립적으로 깨진다.

1. **`eigh` backward의 `1/(λ_i − λ_j)`.** G-1에는 eigh가 두 번 있다(`S^(-1/2)` whitening과
   `A = WΔW`의 고유분해). 고유벡터 항의 gradient가 고윳값 간격의 역수를 포함하므로 근축퇴에서
   발산한다. ⚠️ **`S = 0.75 C_pool + 0.25 τ I`의 shrinkage가 이걸 막아주지 못한다** —
   항등행렬 배수는 모든 고윳값을 **같은 양만큼 밀어 간격을 그대로 둔다**. `clamp_min`도
   forward의 `rsqrt`를 지킬 뿐이다. K=128 pooled covariance는 스펙트럼 꼬리에 촘촘한 군집이
   사실상 항상 있다.
2. **hard argmax.** `f = W u where |λ| is maximal`의 선택 자체에 gradient가 없고, 상위 2개
   `|λ|`가 교차하면 `f`가 **점프**한다. 1번을 고쳐도 남는다.

**그리고 이 실패는 조용하다.** `nonfinite_gradient_policy: zero`가 non-finite gradient를 0으로
치환하므로 학습이 완주하고 SEAL 수치도 나온다 — §66의 함정("Δ≈0이 가설 기각인지 경로
미개방인지 구분 불가")이 그대로 재현된다. **DD 경로를 손대면 P의 gradient가 finite·nonzero이고
control과 다른지를 테스트로 단정할 것**(`tests/test_set_transformer_ridge.py`가 고정한다).

미분 가능한 판본이 필요하면 eigh를 우회한다 — whitening은 **Newton–Schulz**, 최대 `|λ|`
고유벡터는 **`A²`에 대한 k회 전개 power iteration**(`A²`의 최대 고유벡터가 `A`의 `argmax|λ|`와
일치하므로 부호 불명확 문제도 없다). 둘 다 matmul·normalize뿐이라 축퇴에서도 backward가 정의된다.
**아직 구현하지 않았다.**

### G-5. DD gradient를 P에 흘리면 해롭다 (v78, 실측 기각)

v78은 방향을 고정한 채 **이차형식 `z_b = log(fᵀ C_b f)`만** 그래프에 남겨 P가 DD 목적도
반영하게 했다(`train_dd_projection`). 무가중이면 DD가 P의 gradient를 CV의 **52배**(median,
6 에피소드 range 21–103)로 지배하고 방향이 거의 직교(`cos = −0.068`)하므로
`dd_projection_gradient_weight`로 균형을 맞춘다. weight를 스윕한 결과는 **단조**다.

| weight | SEAL macro | fold-paired Δ vs v77 | 95% CI | 판정 |
|---:|---:|---:|---|---|
| 0 (v77) | 0.6873 | — | — | baseline |
| 0.02 (≈1/52) | 0.6869 | −0.0004 | [−0.0021, +0.0013] | 구별 불가 |
| 1.0 (무가중) | 0.6826 | −0.0047 | [−0.0082, −0.0013] | **CI 0 제외 — 유의하게 나쁨** |

두 arm 직접 비교도 −0.0043 [−0.0075, −0.0013]로 CI가 0을 제외한다. 즉 **DD는 P를 실제로
움직이며, 그 방향은 전체 readout에 해롭고, 발언권을 줄수록 단조롭게 나빠진다.** 기제가 P에
도달함을 테스트로 단정하고 크기도 맞춰둔 상태였으므로 이것은 **가설 기각**이다(경로 미개방이
아니다). ⚠️ 무가중 arm은 er_status만 +0.0277(44/50)로 크게 올리고 PIK3CA −0.0349, grade
−0.0093, STK11 −0.0079를 떨어뜨린다 — §71이 경고한 "er_status 단독으로 보면 오판" 패턴이다.

**따라서 `train_dd_projection`은 되살리지 말 것.** v79는 중재가 아니라 **분리**를 택했다
(Active-6).

### G-6. 아직 측정되지 않은 것

- **학습된 P에서의 DD-only 성능.** G-2의 DD-only 0.5862는 **fixed P** 수치다(§87). v77의 학습된
  P로 DD-only를 채점하면 "DD가 CV의 학습된 subspace에서 손해를 보는가"를 **학습 없이** 직접
  확인할 수 있다(DD는 학습 파라미터가 없다). v79의 전제를 값싸게 검증하는 진단이며 미실행이다.
- **DD 전용 learnable 사영.** DD가 자기 사영을 갖고 G-4의 우회 구현으로 학습하는 판본. v79는
  DD를 fixed로 되돌리기만 했고 DD 전용 학습 사영은 시험하지 않았다.

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
