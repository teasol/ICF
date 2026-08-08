# v36(Q1) Proposal: Structured Population Routing — 40→1 압축 해제 (slot 기반 재정의)

**작성일**: `2026-08-08`
**상태**: 제안 — §62 사용자 결정으로 재정의된 v36. 기존 region chunk-attention 제안(폐기·삭제, 2026-08-08)은
§62-1에서 핵심 전제 3건이 반증되어 이 제안이 **좌표 없는 slot 기반**으로 대체한다.

**기준선**: v35 (`configs/train_v35_phase0_largebag_1536.yaml`, rare-free after §61) /
v34-1536 아키텍처 (`BaseModel` → `StructuredPopulationMetaClassifier`)

---

## 0. 한 줄 요약

40개의 구조 token(global/slot/tail)을 task-무관 선형 사상으로 1개로 압축하는 `_projected_bag_tokens`가
**라벨 정보가 들어오기 전에** 정보를 버리고(+0.16, §62-4), 그 결과 population routing softmax가
**길이 1 축**에 걸려 무력해진다(§62-2). Q1은 population 경로에서 40 token을 그대로 통과시켜
**구조 token 타입에 대한 선택 기제(routing)를 복원**한다. 파라미터 0 증가, ckpt 호환, config 플래그로 가역.

## 1. 동기 & 근거 (실측)

- **§62-4 P0-slots probe** (fold-paired, 공식 50-fold): `all@12`(40 token) vs `projected@12`(1 token)
  ridge → **EGFR +0.1597 / STK11 +0.1577** (95% CI가 0에서 멀리). PIK3CA는 전 구간 랜덤(0.50)이라 판정 불가.
- **§62-2**: `_projected_bag_tokens` = 위치별 `Linear(1536→64)` 40개 + concat(2560) + exact-mean residual(1536)
  → `Linear(4096→1536)`. mean pooling이 아닌 **위치별 병목 + concat**, 라벨·task 무관.
  → `_population_memory_logits`(baseline.py:3509)의 routing softmax가 길이 1 축 →
  `population_slot_weights` shape (Q,1), 값 전부 1.0. ABMIL형 선택 기제가 구현돼 있으나 **무력**.
- **주의**: probe의 +0.16 조합은 **클래스 조건부**(context 라벨로 적합한 ridge)이다. 모델의 `slot_importance`는
  **task 무관**(query token만 봄)이라 Q1이 +0.16을 온전히 회수한다는 보장은 없다 → 본 게이트는 v35 대비 +0.005 유지.

## 2. 설계 — config 플래그 `meta_population_token_mode`

- `projected` (기본, 현행): `_population_tokens` = `_projected_bag_tokens` → (Q, 1, I)
- `structured` (Q1): `_population_tokens` = `_all_structured_tokens` → (Q, 40, I)
- **변경 범위**: `_population_memory_logits`(eval/ragged) + `_population_memory_logits_batched`(훈련/dense)
  **두 경로 모두** (파일 주석: "drifting one copy ... is exactly how the cls token was first missed here").
- **class_memories / global_shape는 projected 유지** (Q1-minimal, 원인 분리 — arm C 교훈: "한 번에 하나").
- **파라미터 0 증가**: population 분기의 모든 파라미터가 token 개수가 아니라 `token_dim`/`hidden_dim`으로만
  크기가 정해짐 → ckpt strict 로드, shape 보존. zero-init gate 사용 안 함 (§62-6.3).

### Structured 모드 파이프라인 (차원=대문자, 토큰=소문자; E=1, I/D 구분)

```text
[Query Bag Line — structured]
Q-1'. q = _all_structured_tokens = (Q, 40, I)                        [40→1 우회]
Q-2'. q = Proj_D(Norm(q)) = (Q, 40, D)
Q-3'. token_weights = softmax(slot_importance(q)/T_temp) = (Q, 40)   [선택 기제 복원]
Q-4'. q0 = Attn(q, m0, m0) = (Q, 40, D), q1 = Attn(q, m1, m1) = (Q, 40, D)
Q-5'. r0 = cat[q, q0, q−q0, q⊙q0] = (Q, 40, 4D) → Scorer → s0 (Q, 40)
Q-6'. population_attention_logits[c] = (s_c · token_weights).sum(dim=-1) → (Q, 2)
```

(나머지 global/abundance/rare/covariance/fusion 분기는 [`current_architecture.md`](current_architecture.md) §5와 동일)

## 3. 설계 결정 & 하드 제약

### 3.1 순서: attend-then-route vs route-then-attend

- **attend-then-route (코드 형태, Q1-minimal)**: 40 token 각각이 memory에 attend → per-token relation → routing 가중합.
  혼합 대상이 **클래스 조건부** per-token 판정.
- **route-then-attend (대안)**: routing으로 먼저 q 가중합 → 단일 readout. ABMIL과 구조 유사, 더 저렴.
- Q1-minimal은 코드 형태 유지. 순서는 1~2 fold 실측으로 판정 (후속 arm).

### 3.2 label-permutation equivariance (구조 보장 유지)

- **불변식**: 모든 클래스 의존성은 "라벨 유래 객체"(클래스 그룹 c0/c1, memory m_c)를 통해서만 들어온다.
  파라미터는 전부 클래스 간 공유. **per-class 저장 파라미터/embedding 금지** (equivariance 파괴).
- routing을 class 조건부로 할 경우 반드시 **공유 함수 f(q, m_c)** 형태 — memory는 라벨 유래라 스왑 시 반전,
  공유 함수 + 출력 슬롯 스왑이 일관되면 equivariance 유지 (증명: §62-6 및 설계 논의).

### 3.3 routing 붕괴 리스크

- `routing_temperature: 0.5` + sparsity/balance 둘 다 0.0 → softmax가 **한 token으로 붕괴**하면
  40→1 병목을 다른 경로로 재현. 1~2 fold entropy 진단으로 검출. 붕괴 시 후속 arm(ⓐ).

## 4. 게이트 / 판정

1. **P0 (무료)**: ① 두 경로(dense/ragged) 동치 ‖Δ‖∞<1e-4, ② 파라미터 diff = 0, ③ ckpt strict 로드, ④ 41 tests.
2. **routing entropy·선택성 진단 (1~2 fold, 50-fold 전)** — §62-6.4 (50-fold는 비쌈: EGFR ≈ 2.6 GPU-시간).
3. 통과 시 **scratch 학습** (episode-matched vs v35) — **warm start 금지** (§62-7.3: T=1로 학습된 분기를
   T=40 입력에 넣는 것은 OOD, `_fuse_evidence` 스케일까지 흔들림).
4. **본 게이트**: 공식 paired 50-fold ≥3 task vs v35 — pooled AUROC 평균 **+0.005 이상** + ABMIL 격차 절반 축소.

## 5. 리스크 / 열린 과제

- `slot_importance`가 task 무관 → +0.16 온전 회수 미보장. 후속 단독 arm (한 번에 하나):
  ⓐ `routing_sparsity_weight`/`routing_balance_weight` 복원,
  ⓑ `slot_importance`를 class_memory 조건부로 (공유 함수 f(q, m_c), equivariance 유지),
  ⓒ num_slots 재검토 (§62-5 부호 불일치),
  ⓓ IA-MIL (§31 측정 6).
- **class_memories를 40-token으로 바꿀지는 Q1-minimal에서 제외** — probe는 양쪽 다 40-token을 썼으므로
  minimal이 미달 시 다음 레버.
- (판단 필요) 폴드 단위 representation 캐싱 eval (bit-identical ~50×, §62-7.4).

## 6. 참고

- [`current_status.md`](current_status.md) §62 (재정의 + P0-slots), §61 (rare 제거), §60 (v35 완주 + SEAL), §59 (v35 rev.2)
- probe: `scripts/probe_slot_headroom.py` / `scripts/summarize_slot_headroom.py`
- 반증된 구 proposal: region chunk-attention 제안 (폐기·삭제 2026-08-08, git 기록 보존; §62-1)
- 현재 아키텍처 전체: [`current_architecture.md`](current_architecture.md)
