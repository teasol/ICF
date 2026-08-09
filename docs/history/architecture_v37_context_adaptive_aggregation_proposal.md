# v37 Proposal: Context-Adaptive Bag Aggregation — 40→1 압축을 에피소드가 결정한다

**작성일**: `2026-08-08`
**상태**: **제안 — 설계 확정, §9의 결정 2건 후 구현 착수.**
**기준선**: v36 Q1 control arm (`train_v36_q1_baseline_1536.yaml`, rare-free·bf16·scratch)
**기본 평가**: bc_therapy/er_status 공식 50-fold (§64, 1 GPU 단일 워커 50초)

---

## 0. 한 줄 요약

> 현행 `_projected_bag_tokens`(40→1)는 학습이 끝나면 **고정**이고 **에피소드·라벨과 무관**하다.
> v37은 이 압축의 가중치 `w`(40-dim)를 **에피소드의 context 전체를 보고 그때그때 만들어낸다**.
> 압축 자체를 in-context로 결정하는 것 — PFN의 문제의식과 같은 방향이다.

---

## 1. 설계

### 1.1 전체 흐름

```text
입력:  구조 token — context (C, 40, I),  query (Q, 40, I)        # I = 1536, 40 = 1 global + 12×3 slot + 3 tail

[0] 공유 사영 + type embedding
    h[b,k,:] = Proj_D( tok[b,k,:] ) + type_emb[k]                # (·, 40, D),  D = 256

[1] weight_maker_token — position마다 bag 축을 집계 (context만)
    for k in 0..39:
        wmt[k] = SetPool_k( { h[b,k,:] : b ∈ context } )          # (C, D) -> (D,)
    weight_maker_token = stack(wmt)                                # (40, D)

[2] w = softmax( WeightMakerMLP( Pool_k(weight_maker_token) ) )    # (D,) -> (40,) -> softmax

[3] 같은 w 를 context·query에 공통 적용
    bag_token[b] = Σ_k w[k] · h[b,k,:]                             # context (C, D), query (Q, D)
```

`SetPool_k`는 **하나의 공유 transformer encoder**를 40개 position에 각각 적용하고 bag 축을 대칭
pooling한다(§1.3). position 구분은 `type_emb`가 담당한다(§1.2).

### 1.2 type embedding — position 정체성 (사용자 결정: positional encoding처럼 가산)

40개 position은 **의미가 고정**돼 있다(`current_architecture.md` §3):

```text
index 0                      : global_summary
index 1 + 3i, 2 + 3i, 3 + 3i : slot i 의 (center, spread, rare)      i = 0..11
index 37, 38, 39             : tail (fraction 0.01, 0.05, 0.15)
```

slot **index** `i`는 에피소드 간 안정적 의미가 없지만(§63), **token type은 완전히 안정**하다.
따라서 index별 40개 임베딩 대신 **합성형(compositional)** 을 권장한다 — slot 간에 파라미터를
공유해 통계 효율이 높고, `num_slots`를 바꿔도 살아남는다:

```text
type_emb[k] = TOKEN_TYPE[ type(k) ]            # 5개: global / slot_center / slot_spread / slot_rare / tail
            + SLOT_CLASS[ class(k) ]           # 2개: density(i < num_density_slots) / rare_slot   (slot token만)
            + TAIL_FRACTION[ frac(k) ]         # 3개: 0.01 / 0.05 / 0.15                            (tail token만)
```

- 파라미터 `(5 + 2 + 3) × D = 2,560`. 무시 가능.
- **선례**: 폐기된 `_typed_bag_tokens`(v25)가 이미 "token-type + tail-fraction identity embedding"을
  구현했고, slot-index embedding은 **의도적으로 넣지 않았다**. 같은 판단을 따른다.
- **대안(단순형)**: index별 40개 임베딩(`40 × D = 10,240`). 구현은 더 간단하지만 slot 간 공유가
  없다. 합성형이 미달하면 폴백.
- ⚠️ **slot-index embedding은 넣지 않는다.** 에피소드마다 slot `i`가 다른 것을 가리키므로
  equivariance는 아니어도 **일반화를 해친다**.

> **왜 필요한가**: 공유 encoder를 position별 slice에 적용하면 입력 데이터가 달라 출력도 달라지지만,
> encoder는 **자기가 몇 번 position을 보고 있는지 모른다**. type embedding이 그 정체성을 준다.
> 이것이 "encoder 40개"(31.6M)를 "공유 encoder 1개 + 임베딩"(0.80M)으로 **40배 줄이면서도**
> position별 처리를 가능하게 하는 장치다.

### 1.3 `SetPool_k` — bag 축 집계

```text
enc  = TransformerEncoderLayer(d_model=D, nhead=8, ff=4D, batch_first=True)   # 공유 1개
seed = nn.Parameter(1, D)                                                     # 학습 seed 1개

SetPool_k(X):            # X = (C, D)
    Z = enc(X)                                   # (C, D)   bag 축 self-attention, 위치 인코딩 없음
    return CrossAttn(seed, Z, Z)                 # (1, D) -> (D,)   대칭 pooling (PMA)
```

- **위치 인코딩 없음** + **대칭 pooling** → bag 순서 불변(§2-1).
- `memory_seeds` + `memory_cross_attention` 패턴을 그대로 재사용한다(`_class_memories`에 선례).
- **대안**: `Z.mean(dim=0)`. 더 단순하고 집합 크기에 더 안정적이다(§2-4 참조).

### 1.4 `WeightMakerMLP`

```text
Pool_k : (40, D) -> (D,)          # position 축 평균
MLP    : Linear(D, D) -> GELU -> Linear(D, 40)     # 마지막 층 zero-init
w      = softmax(MLP(·))                           # (40,)
```

- **zero-init 마지막 층 → 초기 `w`가 정확히 uniform → 초기 bag token이 "40 token 평균"**.
  해석 가능한 기준점이자, 학습 초기에 무작위 압축으로 시작하지 않는 안전장치다.
- position 축 평균이 정체성을 지우는 것처럼 보이지만, **출력 unit `k`가 position `k`에 대응**하는
  position별 파라미터이므로 정체성은 출력층에 있다.
- **대안(고용량)**: `(40, D)`를 flatten → `Linear(40D → 40)` (0.41M). 에피소드 정보를 더 많이
  통과시킨다. 원안(`D→40`)은 **에피소드 → D 벡터 → 40 가중치**라는 좁은 채널이다.

---

## 2. 불변식 (구현 시 강제)

1. **bag 축 permutation invariance** — `enc`에 bag 위치 인코딩 금지, pooling은 대칭.
2. **query 무누출** — `w`는 **context bag만** 보고 만든다. `_context_pool_stats`처럼
   `context_mask`로 명시 분리할 것. 전체 bag을 넣으면 **조용한 누출**이다.
3. **label-permutation equivariance** — 원안은 `w`가 라벨과 무관하므로 자명하게 충족.
   (§9-2에서 라벨 조건화를 채택하면 `w`가 클래스 스왑에 **불변**이어야 한다.)
4. **집합 크기 강건성** — 훈련 `num_bags [60,100]` vs 평가 er_status **133** / EGFR **262**.
   **2~4배 이동**한다. self-attention softmax는 집합 크기에 민감하다 — 이 리포가 B2 cardinality로
   이미 겪은 유형이다. P0에서 `w` 이동폭을 측정하고(§7-1ⓔ), 크면 mean-pool(§1.3 대안)로 전환한다.
5. **cell 축 불변** — 40 token이 이미 cell 순서 불변이라 자동 충족.

---

## 3. 비용 (실측 기반)

현재 모델 **41.67M**, 교체 대상 `bag_token_*` 사영 스택 **10.23M (24.5%)**.

| 구성요소 | 파라미터 |
|---|---|
| `Proj_D` (I→D, 공유) | 0.39M |
| type embedding (합성형) | 0.003M |
| 공유 transformer encoder 1개 (D=256, ff=4D) | 0.79M |
| PMA seed + cross-attention | 0.26M |
| `WeightMakerMLP` (D→D→40) | 0.08M |
| **소계** | **1.52M** |
| 제거: `bag_token_*` | **−10.23M** |
| **총계** | **41.67 − 10.23 + 1.52 ≈ 33.0M (−21%)** |

- 참고: position별 encoder 40개면 **+31.6M → 63.5M (+52%)**. type embedding이 이를 회피한다.
- ⚠️ **encoder를 I=1536에서 돌리면 안 된다** — 40개면 **1,133M**, 현 모델의 **27배**.
  `Proj_D`가 [1] 앞에 오는 것이 필수 전제다.

---

## 4. 구현 체크리스트

### 4.1 구조적 난점 — `w`는 에피소드 단위라 서명이 바뀐다

현행 `_projected_bag_tokens(tokens)`는 **bag별 무상태 함수**라 3곳에서 독립 호출된다:
`forward`(global tokens), `_class_memories`, `_population_tokens`.
v37의 `w`는 **에피소드 단위**이므로 한 번 계산해 세 곳에 **주입**해야 한다.

```text
StructuredPopulationMetaClassifier.forward:
    w = self._context_aggregation_weights(context)          # context만 사용
    context_bag_tokens = self._bag_tokens(context, w)
    query_bag_tokens   = self._bag_tokens(query,   w)
    → global_shape_classifier / _class_memories / _population_tokens 에 전달
```

### 4.2 변경 목록

| # | 위치 | 변경 |
|---|---|---|
| 1 | `StructuredPopulationMetaClassifier.__init__` | `Proj_D`, type embedding 테이블 3개, 공유 encoder, PMA seed, `WeightMakerMLP` 생성. `meta_bag_aggregation: projected \| context_adaptive` 플래그 |
| 2 | 신규 `_token_type_ids()` | index → (type, slot_class, tail_fraction) 매핑. `num_slots`/`num_density_slots`/`tail_fractions`에서 유도, buffer로 등록 |
| 3 | 신규 `_context_aggregation_weights(context)` | §1.1 [0]~[2]. **context만** 받는다 |
| 4 | 신규 `_bag_tokens(representation, w)` | §1.1 [3] |
| 5 | `forward` | `w` 1회 계산 → 세 소비처에 주입 |
| 6 | `_class_memories`, `_population_tokens` | `w`(또는 계산된 bag token)를 인자로 받도록 |
| 7 | **`forward_batched` + `_class_memories_batched` + `_population_memory_logits_batched`** | **동일 변경을 4D 경로에도** |
| 8 | `configs/model/default.yaml` | `meta_bag_aggregation: projected` 기본값 명시 |
| 9 | `configs/train_v37_*.yaml` | v36 control 복제 + 플래그 |

> [!IMPORTANT]
> **7번이 이 제안의 최대 위험 지점이다.** v36에서 이미 확인했듯 4D(훈련) 경로는 로직을 인라인
> 복제하고 있고, 파일 주석이 직접 경고한다 — *"drifting one copy and not the others is exactly how
> the cls token was first missed here"*. v37은 v36보다 손댈 곳이 **3배 많다.**
> 두 경로 동치 테스트를 **먼저** 작성할 것.

### 4.3 테스트 (신규 `tests/test_context_adaptive_aggregation.py`)

1. **초기 동치**: `WeightMakerMLP` zero-init → `w = uniform` → bag token == 40 token 평균 (`‖Δ‖∞ < 1e-6`).
2. **bag 순서 불변**: context bag을 셔플해도 `w`가 `‖Δ‖∞ = 0`.
3. **query 무누출**: query bag의 내용을 바꿔도 `w`가 **정확히 불변** (불변식 2의 직접 검사).
4. **label-permutation equivariance**: 라벨 스왑 시 최종 logits가 스왑에 일관.
5. **두 경로 동치**: dense(4D) vs ragged(3D) 최종 logits `‖Δ‖∞ < 1e-4` — 4.2-7 누락 검출용.
6. **집합 크기 민감도**: 동일 분포에서 `C ∈ {60, 100, 133, 262}`로 `w`를 뽑아 이동폭 기록
   (회귀 방지가 아니라 **관측치 고정**; 불변식 4).
7. **type embedding 매핑**: `_token_type_ids()`가 `num_slots=12/24`에서 올바른 layout을 내는지.
8. **기본값 회귀**: `projected` 모드에서 기존 동작 유지 (v36 control ckpt strict 로드 + 수치 동일).

---

## 5. 정직한 위험

### 5.1 용량 트레이드오프 (최대 위험)

| | 현행 | v37 |
|---|---|---|
| 압축 형태 | `40×I → I` **전 선형사상** (10.23M) | **스칼라 40개**의 볼록결합 |
| 차원 간 혼합 | 가능 | **불가** (token 내부 차원을 섞지 못함) |
| 적응성 | 없음 | 에피소드 적응 |

§62-4의 **+0.16**을 만든 probe는 폴드마다 **61,440차원 per-feature ridge**였다 — 스칼라 40개는
그보다 훨씬 낮은 용량이다. **v37이 +0.16을 회수한다는 보장이 없고, 용량 감소로 회귀할 수 있다.**

완화 후보(§9-1과 연동):
- **잔차형**: 현행 사영을 남기고 v37 가중합을 **더한다** → 회귀 위험 최소, 파라미터 순증.
- **그룹형**: `w`를 `(40, G)`로 → 차원 혼합 일부 회복.
- **순수 교체**(원안): 가장 깨끗하지만 위험이 가장 크다.

### 5.2 `global_shape_classifier`의 입력이 I=1536 → D=256으로 줄어든다

bag token이 D가 되면 **주 logit 경로**(`final = global_shape + scaled others`)의 입력 표현이
6배 좁아진다. 다른 소비처는 어차피 내부에서 `I→D`로 떨어뜨리므로 영향이 작지만,
global 분기만은 실질 변화다. §9-1의 ⓑ(가중합을 `I` token에 적용)를 택하면 이 위험이 사라진다.

### 5.3 `w`가 라벨을 보지 못한다

원안의 `enc`는 context token만 보고 **라벨은 보지 않는다.** §62-2의 진단이
"**라벨 정보가 들어오기 전에** 압축한다"였으므로, v37은 압축을 *에피소드 적응형*으로 만들 뿐
여전히 *라벨 무관*이라 **진단에 절반만 답한다.** → §9-2.

---

## 6. v36 Q1과의 관계 — 다른 소비처

| 소비처 | v36 Q1 (`structured`) | v37 |
|---|---|---|
| `_population_memory_logits` (query) | **40 token 통과** (압축 미사용) | 적응형 1 token |
| `_class_memories` (context) | 불변(고정 압축) | **적응형 1 token** |
| `global_shape_classifier` | 불변(고정 압축) | **적응형 1 token** |

- Q1이 채택돼도 v37은 나머지 두 소비처에서 의미가 있다. **자연스럽게 공존**한다
  (Q1 모드에서 population은 압축을 아예 안 쓰므로 충돌 없음).
- 단 **동시 도입은 2인자**다. v36 결과를 보고 순서를 정한다.
- **기준선은 v36 control arm**(rare-free·bf16·scratch)을 그대로 쓴다 — 이미 학습돼 있어
  추가 비용이 없고, 정밀도·rare 설정이 일치한다.

---

## 7. 게이트

**P0 (무료)**
1. §4.3 테스트 8종 통과. 특히 ⓐ 초기 uniform 동치, ⓑ query 무누출, ⓒ 두 경로 동치,
   ⓔ `C ∈ {60,100,133,262}`에서 `w` 이동폭 측정.
2. `projected` 기본값에서 v36 control 수치 재현.

**본 게이트**
3. **단독 arm 학습** (rare-free·bf16·scratch·episode-matched) — 실측 **약 2시간**(1 GPU).
4. **paired 공식 50-fold, task ≥3** (er_status 기본 + EGFR + STK11).
   **v36 control 대비 pooled AUROC 평균 +0.005 이상.**
   평가는 **그 arm의 훈련 config로** (rare 함정 — v36 제안서 §4.1).
5. SEAL MeanMIL 대비 우위 + ABMIL 격차 축소. Musk 재확인(소형 bag 회귀 없는지).
6. 미달 시 **미채택, 재현 코드만 보존**.

---

## 8. 후속 arm 후보 (한 번에 하나)

| | 내용 |
|---|---|
| ⓐ | **라벨 조건화** (§9-2) — `w`를 task 적응형으로 |
| ⓑ | `WeightMakerMLP`를 flatten형(`40D→40`)으로 — 에피소드 정보 채널 확대 |
| ⓒ | `w`를 `(40, G)` 그룹형으로 — 차원 혼합 일부 회복 |
| ⓓ | `SetPool_k`를 mean-pool로 — 집합 크기 강건성(불변식 4가 문제될 때) |
| ⓔ | v36 Q1과 합성 |

---

## 9. 착수 전 결정 2건

### 9.1 bag token을 `D`로 내릴 것인가

- **ⓐ 명세대로**: 가중합을 `h`(D-dim)에 적용 → bag token `(·, D)`. 하류 4개 모듈
  (`memory_input_*`, `slot_input_*`, `slot_importance`, `global_shape_classifier`)이 따라 바뀐다.
  파라미터가 줄고 설계가 일관되지만 **§5.2 위험**이 있다.
- **ⓑ 보수형**: `w`는 `D` 공간에서 만들되 **가중합은 원래 `I` token에 적용** → bag token `(·, I)`,
  **하류 전부 무변경**. 단독 인자에 가장 가깝고 §5.2가 사라진다.

→ **ⓑ로 시작을 권합니다.** 바꾸는 것이 "압축 가중치를 어떻게 정하느냐" 하나로 좁혀지고,
통과 후 ⓐ를 별도 arm으로 두면 됩니다.

### 9.2 `enc` 입력에 context 라벨을 넣을 것인가

```text
h[b,k,:] = Proj_D(tok[b,k,:]) + type_emb[k] + label_emb[y_b]     # b ∈ context, 2 × D = 512 파라미터
```

- **찬성**: `w`가 task 적응형이 되어 §62-2 진단에 온전히 답한다. 라벨 없는 `w`는
  "이 에피소드는 분산이 크다" 수준의 적응이 한계이고, 라벨을 보면 "이 task에서는 tail이
  판별적이다"를 배울 수 있다. **누출 없음** — query 라벨 미사용 + `w`는 모든 bag에 공통.
- **반대**: 2인자가 된다. equivariance를 위해 `w`가 클래스 스왑에 **불변**임을 별도로 보장해야 한다.

→ **1단계는 라벨 없이(원안), ⓐ를 지정 후속 레버로** 두는 것을 권합니다. 다만 §5.3대로
**1단계만으로는 진단에 절반만 답한다**는 점은 게이트 해석 시 감안해야 합니다.

---

## 10. 참고

- `current_status.md` §62(P0-slots probe, +0.16), §63(아키텍처 명세·bf16), §64(평가 bf16·캐싱·er_status 기본)
- `current_architecture.md` §3(40 token 구성·anchor), §4(현행 40→1 사영), §5b(train/eval 경로 차이)
- v36 Q1 제안서 — §4.1 rare 평가 함정, §5.1 인라인 복제 경고
- 선례: `_typed_bag_tokens`(v25, 폐기) — token-type + tail-fraction embedding, slot-index 제외
