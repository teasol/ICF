# v36(Q1) Proposal: Structured Population Routing — 40→1 압축 해제

**작성일**: `2026-08-08` (rev.2 — 코드 대조 검증 + 구현 준비 완료)
**상태**: **구현 준비 완료, 착수 대기.** §4의 결정 3건이 확정되면 §5 체크리스트대로 바로 진행 가능.
**기준선**: v35 ckpt `checkpoints/20260807_224559/v35_largebag/epoch=048-val_ce_loss=0.3469.ckpt`
**기본 평가**: bc_therapy/er_status 공식 50-fold (§64, 1 GPU 단일 워커 **50초**)

---

## 0. 한 줄 요약 (+ 범위 보정)

> bag의 구조 token 40개를 **라벨 정보가 들어오기 전에** task-무관 고정 선형사상으로 1개로 압축하는
> 바람에, 이미 구현돼 있는 ABMIL형 선택 기제(`_population_memory_logits`의 routing softmax)가
> **길이 1 축에 걸려 무력**하다. Q1은 **population 분기에 한해** 40 token을 그대로 통과시켜 그
> 선택 기제를 되살린다.

> [!IMPORTANT]
> **"aggregate를 없앤다"가 아니다.** `_projected_bag_tokens`(40→1)는 그대로 남고,
> **`global_shape_classifier`와 `_class_memories`는 계속 압축 token을 쓴다.**
> 바뀌는 것은 **query의 population 분기 입력 하나뿐**이다. 압축을 전면 제거하는 것은 v22 아키텍처로
> 되돌아가는 별개의 큰 변경이며, arm C 교훈("한 번에 하나")상 Q1에 포함하지 않는다.

---

## 1. 근거 (실측)

- **§62-4 P0-slots probe** (fold-paired, 공식 50-fold, 학습 0):
  `all@12`(40 token) vs `projected@12`(1 token) ridge →
  **EGFR +0.1597 [+0.1357, +0.1840]**, **STK11 +0.1577 [+0.1313, +0.1841]**.
  PIK3CA는 전 구간 랜덤(0.50)이라 판정 불가.
- **§62-2 실측**: v35 config로 빌드 시 `population_slot_weights` shape **(Q, 1), 값 전부 1.0**.
- **압축기 품질 문제가 아님**: 랜덤 초기화 압축(`projected_random@8`) 0.6496 vs 학습된 압축 0.6802
  (STK11) — 둘 다 token 집합보다 0.15~0.19 낮다. 원인은 **40→1 압축 자체**.

> [!WARNING]
> **+0.16을 그대로 회수한다는 보장은 없다.** probe의 판독기는 **폴드마다 context 라벨로 새로
> 적합한 ridge**(= task 적응적)인 반면, 모델의 `slot_importance`는 **query token만 보는 task-무관
> MLP**다. Q1이 회수하는 것은 "40 token을 보는 것"이지 "task 적응적으로 가중하는 것"이 아니다.
> 후자는 후속 arm ⓑ(§9). 따라서 본 게이트는 낙관치가 아닌 **v35 대비 +0.005**를 유지한다.

---

## 2. 변경 범위 — 정확히 무엇이 바뀌는가

| 소비처 | 입력 | Q1에서 |
|---|---|---|
| `global_shape_classifier` | `_projected_bag_tokens` (1 token) | **불변** |
| `_class_memories` (context) | `_projected_bag_tokens` (1 token) | **불변** (§4.2 결정 대상) |
| **`_population_memory_logits` (query)** | `_population_tokens` → 1 token | **→ 40 token** |
| `_abundance_ridge_logits` | `slot_metadata` (12×2) | 불변 |
| rare / covariance / fusion | — | 불변 |

### 2.1 config 플래그

```yaml
model_kwargs:
  meta_population_token_mode: projected | structured   # 기본 projected = 현행
```

- `projected`(기본): `_population_tokens` = `_projected_bag_tokens(...).unsqueeze(-2)` → `(Q, 1, I)`
- `structured`(Q1): `_population_tokens` = `_all_structured_tokens(...)` → `(Q, 40, I)`
- ⚠️ 기존 `else` 분기(v22, `project`·`mean_pool` 둘 다 off)는 `_flatten_slot_tokens` =
  **slot token 36개만** 반환한다(global_summary·tails 제외). `structured` 모드의 **40개와 다르다** —
  새 모드로 만들어야 하며 v22 경로를 재사용하면 안 된다.

### 2.2 structured 모드 파이프라인 (E=1)

```text
Q-1'. q = _all_structured_tokens(query)             = (Q, 40, I)     [40→1 우회]
Q-2'. q = slot_input_projection(slot_input_norm(q)) = (Q, 40, D)
Q-3'. w = softmax(slot_importance(q) / 0.5, dim=-1) = (Q, 40)        [선택 기제 복원]
Q-4'. q_c = Attn(q, m_c, m_c)                       = (Q, 40, D)     c ∈ {0,1}
Q-5'. r_c = Scorer(cat[q, q_c, q−q_c, q⊙q_c])       = (Q, 40)
Q-6'. population_attention_logits[:, c] = (r_c * w).sum(dim=-1)      → (Q, 2)
```

---

## 3. 코드로 검증된 사실 (구현이 성립하는 근거)

1. **파라미터 0 증가 · ckpt strict 로드**. population 분기의 모든 파라미터가 token 개수가 아니라
   `token_dim`/`hidden_dim`으로만 크기가 정해진다:
   `slot_input_norm` `LayerNorm(token_dim)` / `slot_input_projection` `Linear(token_dim→hidden)` /
   `population_cross_attention` `MHA(hidden, heads)` / `slot_relation_scorer(hidden, …)` /
   `slot_importance(token_dim, …)`. → **shape 보존**이므로 zero-init gate를 쓰지 않는다(§62-6).
2. **두 경로 모두 T=40에서 그대로 동작**한다(형상 검토 완료):
   - ragged: `token_weights` softmax `dim=-1` → `(Q,40)`; cross-attn은 query `(Q,40,D)` / kv `(Q,M,D)`,
     `batch_first=True` → OK.
   - dense: `episodes, queries, slots, _ = query_tokens.shape` → `slots=40`,
     `flat_query.reshape(E*Q, 40, D)`, `memory.repeat_interleave(queries)` → OK.
3. **routing 붕괴 진단이 이미 훈련에 계측돼 있다.** `model_interface.py`가 매 스텝
   `routing_entropy`(= −Σ w log w)와 `_routing_balance_loss`(= KL(mean usage‖uniform))를 계산해
   `terms`에 남긴다. 가중치는 0.0이라 **손실에는 기여하지 않지만 로깅은 된다.**
   현재 T=1이라 둘 다 자명하게 0이고, structured 모드에서 비로소 의미를 갖는다.
   → **별도 진단 코드 없이 훈련 로그만 봐도 붕괴를 감지**할 수 있다.
   (`_routing_balance_loss`는 `[query, slot]` 2D를 요구하는데, 4D 훈련 경로도
   `training_step`이 `value[episode]`로 에피소드별 slice를 넘기므로 2D가 보장된다.)
4. **연산 비용은 무시 가능**. 훈련 스텝당 population 토큰이 `E×Q×40` (v35: 1×~12×40 ≈ 480),
   `relation_features`가 `(E,Q,40,4D)`. cell 축과 무관하므로 VRAM 가드 재추정 불필요.

---

## 4. ⚠️ 착수 전 결정 필요 3건

### 4.1 [필수] rare 설정 — **평가 config 불일치 함정**

**현황**: v35 **ckpt는 rare ON으로 학습**됐고(§60, 런 시점이 §61보다 앞섬),
v35 **config는 §61에서 rare-free로 바뀌었다**(`meta_enable_rare_evidence: false`).
§61이 예고한 "rare-free v35 arm 학습"은 **아직 실행되지 않았다.**

**함정**: `meta_enable_rare_evidence: false`로 학습하면 `_fuse_evidence`가
`rare_logits.new_zeros(...)`로 **텐서를 교체**해 그래프를 끊으므로 **rare 분기 파라미터에 gradient가
전혀 흐르지 않는다 → 초기화 상태로 남는다.** 이 ckpt를 rare가 켜진 config
(`train_v34_phase0_largectx_1536.yaml`)로 평가하면 **랜덤 초기화 분기가 logits에 주입**된다.
지금까지 이 사고가 안 난 이유는 v35 ckpt(rare ON)와 v34 eval config(rare ON)가 우연히 일치했기 때문이다.

**규칙 (신설 권고)**: **평가는 그 arm의 훈련 config로 한다.** `test_pathobench.py --config`에
훈련 config를 그대로 넘기면 된다(official-folds 모드에서 data 섹션은 쓰이지 않는다).

**결정 필요 — 어느 쪽으로 갈지**:

| 안 | 구성 | 요인 수 | 비용 |
|---|---|---|---|
| **A (권장)** | rare-free 기준선 arm + rare-free Q1 arm **둘 다 새로 학습** | **1인자** (population mode) | 학습 2회 (~2시간) |
| B | Q1을 rare **ON**으로 학습해 기존 v35 ckpt와 비교 | 1인자지만 §61 결정 되돌림 | 학습 1회 |
| C | rare-free Q1을 rare-ON v35 ckpt와 비교 | **2인자** (population + rare) | 학습 1회 |

→ **A 권장.** 학습 1회가 **약 1시간**(§7)이라 2회도 부담이 아니고, §61의 미결 과제
("rare-free v35 arm 학습")를 동시에 해소한다. C는 arm C 교훈에 정면으로 어긋난다.

### 4.2 [선택] `_class_memories`도 40 token으로 갈 것인가

- **Q1-minimal(기본)**: context는 압축 token 유지. query 40 token이 **압축된 context**로 만든
  memory에 attend → 비대칭.
- **대칭안**: `_class_memories`도 `_all_structured_tokens`를 쓰면 memory cross-attention의 key/value가
  `C×40` token이 된다(er_status: 133×40 = 5,320 — MHA 8 seed 기준 저렴).
- probe(+0.16)는 **양쪽 다 40 token**을 본 설정이므로, minimal이 미달하면 여기가 다음 레버다.
- → **minimal로 시작**하되, 두 번째 arm 후보로 문서에 남긴다.

### 4.3 [필수] 정밀도 교란 — 기준선을 bf16으로 다시 만들 것

§63–§64로 학습·평가 모두 bf16-mixed가 강제됐다. 기존 v35 ckpt는 **fp32 학습본**이므로,
그대로 비교하면 (population mode) + (precision) **2인자**가 된다.
**§4.1에서 A안을 택하면 기준선 arm도 bf16으로 새로 학습되므로 이 교란은 자동 해소된다.**
(B/C안을 택하면 교란으로 명시하고 진행해야 한다.)

---

## 5. 구현 체크리스트

### 5.1 코드

| # | 파일 | 변경 |
|---|---|---|
| 1 | `src/models/baseline.py` `StructuredPopulationMetaClassifier.__init__` | `meta_population_token_mode: str = "projected"` 인자 추가 + 값 검증(`{"projected","structured"}`) |
| 2 | 〃 `_population_tokens` (≈3469) | `structured`면 `_all_structured_tokens(representation)` 반환 (unsqueeze 없음) |
| 3 | 〃 `_population_memory_logits_batched` (≈3857) | **로직이 인라인 복제**돼 있음 — 같은 분기를 `_all_structured_tokens_batched(query)`로 추가 |
| 4 | 〃 `BaseModel.__init__` (≈4480) | 인자 통과 |
| 5 | `configs/model/default.yaml` | `meta_population_token_mode: projected` 명시(기본값 가시화) |
| 6 | `configs/train_v36_q1_structured_1536.yaml` (신규) | v35 config 복제 + `meta_population_token_mode: structured` |
| 7 | `configs/train_v36_q1_baseline_1536.yaml` (신규, §4.1-A) | v35 config 그대로(=rare-free 기준선 arm) |

> [!IMPORTANT]
> **3번을 빠뜨리면 훈련만 조용히 예전 동작으로 남는다.** `baseline.py`의 주석이 이 중복을 직접
> 경고한다 — *"drifting one copy and not the others is exactly how the cls token was first missed here"*.

### 5.2 테스트 (신규 `tests/test_population_token_mode.py`)

1. **기본값 회귀**: `projected` 모드에서 `population_slot_weights` shape `(Q,1)`·값 1.0 (현행 고정).
2. **모드 동작**: `structured`에서 shape `(Q,40)`, `sum=1`, **비퇴화**(엔트로피 > 0).
3. **두 경로 동치**: 같은 에피소드를 dense(4D `forward_episode_batch`)와 ragged(3D `forward`)로 통과 →
   `population_attention_logits` `‖Δ‖∞ < 1e-4` **(structured 모드에서)**. ← 5.1-3 누락 검출용.
4. **파라미터 불변**: 두 모드의 `state_dict` 키·shape 완전 동일, v35 ckpt **strict 로드 성공**.
5. **토큰 수**: `structured` 토큰 수 == `structured_tokens_per_bag`(40)이고 v22 경로(36)와 다름.

기존 스위트 **51 tests** 통과 유지 (§64 기준; 제안서 rev.1의 "41 tests"는 stale).

---

## 6. P0 게이트 (학습 0)

1. `projected` 기본값에서 **기존과 수치 동일** — v35 ckpt로 er_status 50-fold 재실행 시
   macro `0.6975` / pooled `0.6925` 재현 (§64 기준선). **50초**.
2. `structured`에서 **routing 비퇴화** 확인 — 엔트로피와 top-1 가중치 분포.
   `ln 40 = 3.689`가 균등 상한이다. **한 token으로 붕괴(엔트로피 → 0)하면 40→1 병목을 다른 경로로
   재현**하는 것이므로 후속 arm ⓐ로 넘긴다.
3. 51 tests + §5.2 신규 테스트 통과.

※ 학습 전 `structured` ckpt가 없으므로 2번은 **v35 ckpt에 모드만 켜서** 본다. 이 상태의 AUROC는
T=1로 학습된 분기를 T=40에 넣은 것이라 **의미 없는 수치**이고, 볼 것은 **가중치 분포뿐**이다.

---

## 7. 실행 계획 & 실제 비용

| 단계 | 내용 | 비용 (실측 기반) |
|---|---|---|
| 1 | 코드 + 테스트 (§5) | — |
| 2 | P0 게이트 (§6) | 50초 × 2 + 테스트 65초 |
| 3 | **기준선 arm 학습** (rare-free, bf16, scratch) | **~1시간** |
| 4 | **Q1 arm 학습** (동일 + structured, scratch, episode-matched) | **~1시간** |
| 5 | paired 50-fold ≥3 task | task당 **50초** |

**학습 비용 실측**: v35 런(`logs/20260807_224559`)은 2×B200에서 **512 step/epoch, ~6.5 it/s ≈ 79초/epoch**
→ 50 epoch ≈ **65분**. 제안서 rev.1과 §62가 전제하던 "학습은 비싸다"는 과장이었다.
**평가 비용**도 §64 캐싱으로 EGFR 2.6 GPU-시간 → **50초 수준**이다. 두 병목이 모두 사라졌으므로
**arm 2개를 나란히 돌리는 것이 정석**이다(8×B200 중 2장 사용 중, 여유 있음).

- **warm start 금지**: T=1로 학습된 분기를 T=40 입력에 넣는 것은 분포 외이고 `_fuse_evidence`
  스케일까지 흔든다. **scratch + episode-matched**(51,200 episodes)로 간다(§42–43 arm C 교훈).
- ⚠️ 평가 전 `/tmp/pathobench_official_workers/`의 fp32 시절 캐시 정리 (§64-3).

---

## 8. 본 게이트 / 판정

1. **paired 공식 50-fold, task ≥3** (er_status 기본 + EGFR + STK11 권장 —
   PIK3CA는 §62-4에서 전 구간 랜덤이라 판정력이 없다).
2. **기준선 arm 대비 pooled AUROC 평균 +0.005 이상.**
3. SEAL **MeanMIL 대비 우위** + **ABMIL 격차 축소**. 현재 er_status는 ABMIL −0.020 / MeanMIL −0.015.
4. **Musk 재확인** — 소형 bag(median 12) 회귀 없는지 (확정 목표 0.95).
5. 미달 시 **미채택, 재현 코드만 보존** (v31/v32/v33/v35 데이터 arm과 동일 규율).

---

## 9. 후속 단독 arm 후보 (한 번에 하나)

| | 내용 | 근거 |
|---|---|---|
| ⓐ | `routing_sparsity_weight`/`routing_balance_weight` 복원 | 둘 다 0.0 — routing 붕괴 시 1순위 |
| ⓑ | `slot_importance`를 **class_memory 조건부**로 (공유 함수 `f(q, m_c)`) | 현재 가중치는 task 무관, relation만 class 의존 — **ABMIL과의 진짜 차이가 여기 남아 있다** |
| ⓒ | **token-type embedding** (global / slot-center / slot-spread / slot-rare / tail×3) | slot **index**는 에피소드 간 불안정하지만 **type은 안정**하다. 폐기된 `_typed_bag_tokens`(v25)에 이미 구현체가 있다 |
| ⓓ | `_class_memories`도 40 token (§4.2) | probe는 양쪽 40 token 설정 |
| ⓔ | `num_slots` 재검토 | §62-5에서 task별 부호 불일치 |
| ⓕ | IA-MIL (`use_instance_attention_mil`) | §24 기각이 §31 측정 6에서 무효화됨 |

**label-permutation equivariance 불변식** (ⓑ 진행 시 필수): 모든 클래스 의존성은 라벨 유래 객체
(클래스 그룹, memory `m_c`)를 통해서만 들어오고 파라미터는 클래스 간 공유한다.
**per-class 저장 파라미터·embedding 금지.**

---

## 10. 참고

- `current_status.md` §62(재정의 + P0-slots probe), §63(아키텍처 명세 + bf16 강제), §64(평가 bf16 + 캐싱 + er_status 기본)
- `current_architecture.md` §3(40 token 구성·anchor), §4(40→1 사영), §5(분기 전체), §5b(train/eval 경로 차이)
- probe: `scripts/probe_slot_headroom.py`, `scripts/summarize_slot_headroom.py`
- 반증된 구 proposal(region chunk-attention): 폐기·삭제 2026-08-08, git 이력 보존 (§62-1)
