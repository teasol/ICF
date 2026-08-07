# Archived sections — v33 Phase 0 (arm B/C) saga

Archived from `docs/current_status.md` on **2026-08-08** (handoff update §60).
These sections (§41–§48, 2026-08-05/06) are fully resolved: v33 Phase 0 arm B and
arm C were both **rejected** (§48), the NCCL P2P hang fix lives on in
`scripts/launch_interactive_training.sh` + agent_handoff §43 note, and the
PathoBench 5-fold / sample-context results are superseded by the official 50-fold
protocol (§52/§53). v30 baseline remains committed. Their current_status stubs
keep the section numbers so cross-references still resolve.

---

## 41. 2026-08-05 — v33 Phase 0 구현: arm B(C) 데이터 컨트롤 + 학습 런치

**상태**: v33 proposal §9에 따라 **arm B(v30 + six-task + B2)와 arm C(v30 +
legacy + B2b)** 구현·런칭 완료. 새 architecture(residual consensus)는 Phase 1
probe 통과 전까지 구현하지 않는다. ICI 잠금 유지.

- **B2b (per-bag cardinality) 구현**:
  - `SyntheticManifoldGenerator(per_bag_cardinality=True)` — 에피소드 내에서 각
    bag이 `n_b ~ LogUniform[1,1024]`을 독립 추첨. dense generation은
    `max(n_b)`에서 수행 후 per-bag subset으로 subsample(ragged list 반환).
  - dense cell은 bag 내 교환 가능하므로 subsample은 i.i.d. `n_i` 추첨과
    분포상 동일. sparse task(arm D 예비)는 각 양성 bag의 유지 subset 내에서
    m개의 shifted cell을 latent 공간에 marking해 subsample 후에도 보존.
  - oracle abundance/population features는 subsample 후 per-bag 재계산.
  - `SyntheticEpisode.x` 타입을 `Tensor | list[Tensor]`로 확장.
  - collator: ragged 배치 >1 거부(에피소드 간 stack 불가), eval collator는
    `len(x)`로 bag 수 처리. `training_step`에 ragged(Sequence) 단일 에피소드
    분기 추가. `episode_batch_size=1` 필수.
- **config**: `configs/train_v33_phase0_armB.yaml`(six-task
  `[0.32,0.24,0.04,0.04,0.16,0.20]`), `configs/train_v33_phase0_armC.yaml`
  (`per_bag_cardinality: true`, batch=1, shape_group_size=1).
- **테스트**: `tests/test_b2b.py` 신규 10개 추가. 전체 기본 suite
  **29 tests / 185.785s 전부 통과**.
- **GPU 스모크**: arm B/C 각 1 epoch + 16 train batch 정상(exit=0).
- **학습 런치** (detached, GPU 0):
  - arm B: `logs/20260805_214745/v33_phase0_armB.out`, ckpt
    `checkpoints/20260805_214745/v33_phase0_armB/` (50 ep, batch 8, ~2.9min/ep)
  - arm C: `logs/20260805_214751/v33_phase0_armC.out`, ckpt
    `checkpoints/20260805_214751/v33_phase0_armC/` (50 ep, batch 1, ~24min/ep)
- **arm C 비대칭 주의**: batch 1이라 epoch당 optimizer step이 v30 대비 8x.
  회귀 gate에 대해 보수적(더 많이 학습해도 회귀 시 = 강한 부정 신호).
- **운영 결정 — 최초 arm C 중단 후 step-matched로 재설계** (사용자 확정):
  - arm C는 `4096 steps/epoch × 50 = 204,800` optimizer step으로, arm B의
    `512 × 50 = 25,600`보다 8배 많아 예상 약 20시간이 걸렸다. 이는 의도한 공정한
    데이터 컨트롤이 아니므로 launcher PID `4183921`과 torchrun PID `4183927`에
    `SIGTERM`을 보내 중단했다. 생성된 checkpoint는 없다.
  - arm B는 원래 v30 architecture를 유지하고 `precision: bf16-mixed`를 명시했다.
    `logs/20260805_220642/v33_phase0_armB_bf16.out`, launcher PID `37183`; 기존
    `checkpoints/20260805_214745/v33_phase0_armB/last.ckpt`에서 복원해 진행 중이다.
  - arm C는 architecture/ragged B2b를 바꾸지 않고 train episode만 512/epoch로 조정했다.
    따라서 `512 episodes / batch 1 = 512 updates/epoch`로 arm B의
    `4096 / batch 8 = 512 updates/epoch`와 정확히 같다. config는
    `configs/train_v33_phase0_armC.yaml`이다.
  - precision 정렬 재시작 (2026-08-05): 기존 arm C가 FP32(`32-true`)인데 반해
    v30 baseline과 arm B가 `bf16-mixed`여서 요인 분리 계약에 어긋났다. FP32로 2
    epoch 진행한 `220843` run은 중단하고, `precision: bf16-mixed`로 정렬해
    **`logs/20260805_221615/v33_phase0_armC_bf16.out`**(launcher PID `61322`,
    ckpt `checkpoints/20260805_221615/v33_phase0_armC_bf16/`)에서 재시작했다.
    ragged B2b는 스모크에서 이미 bf16-mixed로 검증했다. (commit `4a39ab9`)
- **8× 에피소드 수 비대칭 (해석 주의)**: step은 arm B와 일치(25,600)하지만
  **총 에피소드 수는 arm C 25,600 vs v30/arm B 204,800으로 8× 차이**가 남는다.
  v30 val_ce 곡선을 보면 ep 25~49에서도 `0.4533→0.4442`로 완만하게 개선 중(best가
  마지막 epoch 48)이라 v30 자체도 에피소드 수가 아직 경계(binding)였고, arm C는
  v30의 약 epoch 6~7 수준(val_ce ~0.457)에 머물 것으로 추정된다. 따라서 arm C 회귀
  gate가 B2b 효과가 아니라 **과소학습 편향으로 오염될 수 있다**. 판정 시 arm C의
  val_ce가 epoch 50에도 내려가는 중이면(수렴 전) **top-up(추가 학습)으로 수렴점까지
  이어간 뒤 비교**한다. 엄밀한 대안은 에피소드-매치(4096/epoch, ~20h)지만 비용이 크다.
- **Phase 0 gate**: sparse task AUROC ≥ 0.75 (arm B), legacy overall 회귀
  ≤ 0.01 (paired CI가 0 포함), B2b가 full-vs-subsample margin drift ≥20% 감소
  (probe로 측정).
- **바로 다음**: Phase 0 결과 선택 → frozen-v30 multi-resolution probe
  (Phase 1) → paired AUROC `+0.01` 통과 시에만 v33 residual 구현.

## 42. 2026-08-06 — v33 Phase 0 arm B/C 학습 완료 + gate 평가

**상태**: arm B, arm C 모두 50 epochs 학습 완료. 1,000-episode 합성 평가로 Phase 0
gate 판정을 내렸다. **arm B는 sparse task gate 미달, arm C는 legacy overall 회귀
gate 미달** → v33 Phase 0의 두 주 효과(B: six-task+sparse, C: legacy+B2b) 모두
gate 통과 실패.

- **학습 완료**:
  - arm B(v30+six-task+B2): `checkpoints/20260805_220642/v33_phase0_armB_bf16/`,
    best `epoch=044 (val_ce_loss 0.4290)`. 50 ep / 204,800 ep / 25,600 steps.
  - arm C(v30+legacy+B2b): `checkpoints/20260805_221615/v33_phase0_armC_bf16/`,
    best `epoch=049 (val_ce_loss 0.5351)`. 50 ep / 25,600 ep / 25,600 steps.
    **best가 마지막 epoch** → 수렴 전 경계 상태(§41 과소학습 예측과 일치).
- **평가** (1,000 ep, seed 42, 고정 val 스트림 = v30 legacy B2 분포):
  - arm B six-task: overall AUROC **0.8461 [0.834, 0.857]**, log loss 0.4748.
    per-task: composition 0.8745 / state 0.8214 / covariance 0.7414 /
    interaction 0.8465 / combined 0.9515 / **any_positive_sparse 0.6747**.
  - arm B legacy: **0.8500 [0.839, 0.861]** vs v30 committed 0.8512 [0.840, 0.862]
    → 회귀 **-0.0012**, paired `P(arm B beats v30)=0.04`(에피소드·태스크 매칭 확인).
  - arm C legacy: **0.8139 [0.802, 0.825]** vs v30 committed 0.8512 [0.840, 0.862]
    → 회귀 **+0.0373**, paired `P(arm C beats v30)=0.00` (5000 bootstrap) — CI가
    겹치지 않고 v30이 사실상 100% 우세. 회귀는 통계적으로 확정적.
- **Gate 판정**:

  | Gate | arm B | arm C |
  |---|---|---|
  | sparse task AUROC ≥ 0.75 | ❌ **0.6747** | — |
  | legacy overall 회귀 ≤ 0.01 | ✅ -0.0012 | ❌ **+0.0373** |

- **해석**:
  - arm B: `any_positive_sparse` 태스크가 유용성 기준(0.75)에 미달. six-task 믹스의
    sparse 추가는 Phase 0에서 기각. legacy 성능은 v30 대비 소폭 열세(-0.0012)로 안전.
  - arm C: B2b 학습이 v30 legacy B2 val 스트림에서 0.037 회귀 → gate 실패.
    §41에서 예고한 과소학습 편향(8× 적은 에피소드, best=epoch 49, val_ce 0.5362→0.5351
    완만 하락)과 일치. protocol대로 **top-up(추가 학습)으로 수렴점까지 이어간 뒤
    재판정**이 필요할 수 있다.
  - B2b full-vs-subsample margin drift probe: 아직 미실시
    (`probe_v32_headroom.py`에는 drift 측정이 없음).
- **예측 파일**: `predictions/synthetic_v33_phase0_armB_6task_1000ep.pt`,
  `predictions/synthetic_v33_phase0_armB_legacy_1000ep.pt`,
  `predictions/synthetic_v33_phase0_armC_legacy_1000ep.pt`.
- **바로 다음**: ① arm C top-up 여부(사용자 결정), ② (선택) top-up 후 arm C 재평가,
  ③ Phase 0 결과 선택 → frozen-v30 multi-resolution probe(Phase 1).

## 43. 2026-08-06 — arm C top-up: 8×A6000 DDP 전환 + NCCL P2P hang 수정 + B200 vs A6000 속도 기록

**상태**: 사용자 결정으로 arm C top-up을 **8×RTX A6000 (48 GiB) DDP**로 재개했다.
에피소드-매치(`episodes_per_epoch: 4096`, v30과 동일 총 에피소드 예산)로 전환해
§41/§42의 과소학습 편향을 제거한다. 진행 중: 2026-08-06 12:54 시작, epoch 49 ckpt
에서 resume, 현재 ~epoch 53 (총 목표 150).

- **새 config**: `configs/train_v33_phase0_armC_ddp8.yaml` — medium 체인을 상속하지
  않는 자체 포함형. B2b ragged(`per_bag_cardinality: true`, `episode_batch_size: 1`),
  `episodes_per_epoch: 4096`, `trainer: devices 8 / ddp_find_unused_parameters_false /
  bf16-mixed / max_epochs 150`. resume: `archive/v33_phase0_armC_bf16/last.ckpt`.
- **NCCL P2P hang (gnode5) — 원인 진단 및 수정**:
  - 증상: 8-GPU torchrun이 `All distributed processes registered` 직후 영원히 hang.
    GPU 8장 모두 100% util인데 메모리 ~450 MiB 고정, rank CPU가 1코어씩 회전, 로그·
    metrics 무진행.
  - 진단: `scripts/archive/probes_smoke/nccl_probe.py`(신규, broadcast/all_reduce/대형 broadcast 최소 재현)로
    **NCCL comm init은 8 rank 모두 정상 완료**됨을 확인. 그러나 **첫 컬렉티브
    (`dist.barrier()`)에서 hang** → 통신 그룹 생성이 아니라 **전송(transport) 문제**.
    `NCCL_DEBUG=INFO`에서 `Channel ... via P2P/CUMEM` 채널 사용 확인.
  - 검증: `NCCL_P2P_DISABLE=1`만 8 rank 프로브가 통과. `NCCL_CUMEM_DISABLE=1`/
    `NCCL_P2P_LEVEL=SYS`는 여전히 hang → 이 머신의 NCCL P2P/CUMEM 전송이 불안정.
  - 수정: `scripts/launch_interactive_training.sh`에 `NCCL_P2P_DISABLE=1` 기본 적용 +
    detached 워커에 env 전달 추가. 단일 노드 8×A6000(NVLink 없음)이므로 SHM 전송으로
    동작.
- **B200 1장 vs A6000 8장 속도 비교** (동일 v30 arch · `episode_batch_size=1` ·
  bf16-mixed · bag 60–100, `n_b ~ LogUniform[1,1024]`, 전형 9k–15k 셀 / worst 102k):

  | 항목 | B200 1장 (기준, step-matched 512 ep/epoch) | A6000 8장 (현재, 4096 ep/epoch) |
  |---|---|---|
  | step당 시간 | **0.36 s/step** | **~0.66 s/step** (rank당) |
  | it/s | 2.5–2.8 it/s | **~1.5 it/s** (rank당) |
  | 에피소드 처리량 | 2.8 ep/s | **~1.5 ep/s** (rank당) |
  | epoch 시간 | ≈ 3:05 / epoch (512 steps) | **≈ 5:38 / epoch** (512 steps/rank) |
  | 50 epochs 총량 | ≈ 2.6 h (25,600 ep) | **≈ 4.7 h** (4096 ep/epoch 기준) |
  | 노드 총 처리량 | 2.8 ep/s | **≈ 12 ep/s** (8×1.5, ~4.3×) |

  - **해석**: A6000 1장 기준으로는 B200 대비 ~1.8× 느리다(step당 0.36→0.66 s). 원인은
    (1) A6000(48 GiB) vs B200(180 GiB) 연산·메모리 대역폭 차이, (2) NCCL P2P 비활성화로
    인한 all-reduce 오버헤드 증가, (3) GPU util이 23–100%로 불균일해 DDP 동기화 대기.
    그러나 **8장 병렬로 노드 총 처리량은 ~4.3×** (2.8→12 ep/s). 에피소드-매치(204,800 ep)
    예산 기준으로는 B200 2.6 h 대비 A6000 8장 약 4.7 h (P2P 비활성화 비용 포함).
- **검증 완료 (epoch 50–53)**: resume 성공(`Restored all states`), 첫 step VRAM
  peak 0.92 GiB(1.8%), epoch당 VRAM 3.8–7.8 GiB(A6000 48 GiB의 ~8–16%), 체크포인트
  `epoch=NNN` 자동 저장 확인. val_ce: 0.5381(50) → 0.5375(51) → 0.5377(52) → 0.5370(53).
- **바로 다음**: ① arm C top-up 완주(150 epoch) 후 §42 재평가(legacy overall 회귀 gate),
  ② 통과 시 frozen-v30 multi-resolution probe(Phase 1).

---

## 44. 2026-08-06 — 패딩 배칭 (B2b `episode_batch_size>1`) + 병목 프로파일

**상태**: arm C top-up의 핵심 병목(rank당 batch=1, VRAM ~5%, GPU util 불균일)을
프로파일로 확정하고 **ragged B2b 에피소드의 패딩 배칭**을 구현·검증했다.
commit `568c5f8`.

- **프로파일 (B200, arm C ddp8 config)**: step의 ~92%가 모델 forward+backward
  (~177 ms / ~190 ms). 원인은 집계기의 **bag별 Python 루프**(소형 커널 다수)이며,
  arm B(dense, batch 8)가 8 에피소드를 ragged 1개와 비슷한 벽시계로 처리하는 것으로
  확인 — 벡터화 dense 경로는 배칭이 핵심.
- **구현**: collator가 ragged 배치를 `(x, y, cell_mask, bag_mask)`로 패딩.
  집계기 전역에 cell mask + per-bag valid count 관통 (슬롯 할당은 softmax 후 0처리로
  all-`-inf` NaN 회피, tail/rare는 per-bag count + keep 마스킹, covariance 정규화,
  context-pool/앵커/CLS 폴링). `forward_episode_batch`는 집계기는 완전 배칭, bag-
  토큰 수준 meta-classifier만 에피소드별 루프(저비용). query rare-evidence에도
  per-query cell mask.
- **잠재 train/eval 불일치 수정**: `_forward_dense`의 `tail_fractions`가 softmax
  가중합을 쓰던 것을 list 경로(평가/Musk 기준)와 동일한 **산술 평균**으로 정렬.
  dense/패딩/평가가 이제 일치.
- **B200 실측 (동일 v30 arch, ddp8 데이터)**:

  | episode_batch_size | 처리량 | peak VRAM | 비고 |
  |---|---|---|---|
  | 1 (기존) | 5.8 ep/s | 1.1 GiB | rank당 1 에피소드 |
  | 2 | **~16 ep/s** | ~16 GiB | **A6000 48 GiB 안전** |
  | 4 | ~12 ep/s | ~32 GiB | A6000 경계 |
  | 8 | ~14 ep/s | ~68 GiB | A6000 OOM, B200 전용 |

  실 wall-clock은 배치로 Lightning 오버헤드가 분산되어 추가 이득.
- **config**: `configs/train_v33_phase0_armC_ddp8_batch2.yaml`
  (`episode_batch_size: 2`, episode-match 4096 ep/epoch → 256 steps/epoch).
- **검증**: `tests/test_ragged_batching.py` 3개 (패딩 collator + 패딩 배치 == 개별
  list 경로 logits, 1e-4). `test_b2b.py`의 "ragged 거부" 계약을 "패딩" 계약으로 갱신.
  **전체 38 tests 통과 (~256s)**.
- **바로 다음**: ① (선택) A6000 top-up을 batch2 config로 재런칭/적용 — 기존 batch=1
  런과의 비교 판단은 사용자 결정, ② §42 재평가, ③ Phase 1 probe.

## 45. 2026-08-06 — arm C top-up 중간 Musk zero-shot: 대형 bag(n>34) 개선 + 소형 trade-off

**상태**: arm C top-up 진행 중(epoch ~84/150)에 **중간 checkpoint(epoch 64,
`best_epoch64_valce0.5287.ckpt`)의 Musk zero-shot을 측정**해 v30 확정 baseline과
비교했다. 목적은 top-up 완주 전에 B2b 추가 학습이 Musk 방향을 어떻게 움직이는지
조기 신호를 잡는 것. **대형 bag(n>34)이 0.698→0.825로 크게 개선**됐고, 대신
**소형(n≤4)이 0.792→0.700으로 희생**됐다.

### 실행 환경 (gnode4, 8×A5000 — gnode5와 파일서버 공유)

- 이 세션은 NHN(B200)/gnode5(A6000)가 아닌 **gnode4**에서 진행. arm C top-up 자체는
  gnode5에서 돌며 NFS로 체크포인트/metrics가 gnode4에 실시간 동기된다.
- **v30 checkpoint는 워크스페이스 `checkpoints/`에 없고 `/home/kimds/archive/`에 있다**
  (`/data-hdd`는 백업 서버). 워크스페이스 root도 `/NHNHOME/BASE/kimds/ICF`가 아닌
  `/home/kimds/ICF` — 다중 위치 동기화 환경이라 경로 확인 필요.
- Musk 데이터: `/home/kimds/BagPFN/Data/Musk/musk.pkl` (NHN 경로 아님).
- 사용된 checkpoint:
  - v30 baseline: `archive/v30_cardinality_poolz_l2/epoch=048-val_ce_loss=0.4442.ckpt`
  - v33 arm C: `archive/v33_phase0_armC_ddp8_topup_20260806/best_epoch64_valce0.5287.ckpt`
- config: v30 `train_v30_cardinality_poolz_l2.yaml`, v33 `train_v33_phase0_armC_ddp8.yaml`
  (arm C도 `bag_representation: poolz_l2` — v30 arch 그대로. "legacy"는 데이터 믹스
  의미일 뿐 표현 아님). preprocess는 기본 `bag_view` (v30 S2 측정과 동일).

### 결과 (102 bags leave-one-out, seed 42)

| 지표 | v30 baseline | v33 arm C (ep64) |
|---|---|---|
| **AUROC [95% CI]** | **0.8539 [0.774, 0.925]** | **0.8799 [0.810, 0.946]** |
| Accuracy | 0.794 | 0.814 |
| Balanced acc | 0.785 (sens 0.744 / spec 0.825) | 0.796 (sens 0.718 / spec 0.873) |
| Log loss | 0.476 | 0.441 |
| corr(prob, log n) | +0.057 | −0.146 |

v30 baseline은 문서값과 **정확히 재현**(0.8539) — 체크포인트/파이프라인 무결성 확인.

### 밴드별 AUROC (stratified, 같은 102 bag)

| 밴드 | v30 baseline | v33 arm C | Δ |
|---|---|---|---|
| ALL | 0.854 | 0.880 | **+0.026** |
| **n≤4** | 0.792 | 0.700 | **−0.092** |
| 5..10 | 0.833 | 0.925 | +0.092 |
| 11..34 | 0.964 | 0.970 | +0.006 |
| **n>34 (대형)** | **0.698** | **0.825** | **+0.127** |

### paired bootstrap (4,000 resample)

| stratum | bags | v30 | v33 | Δ | 95% CI | P(v33>v30) |
|---|---|---|---|---|---|---|
| ALL | 102 | 0.854 | 0.880 | +0.026 | [−0.021, +0.078] | 0.858 |
| n≤4 | 29 | 0.792 | 0.700 | −0.092 | [−0.258, +0.033] | 0.084 |
| n>4 | 73 | 0.864 | 0.913 | +0.049 | [−0.014, +0.118] | 0.932 |

### 판독 (사용자 관점 포함)

- **전체 +0.026은 통계적으로 무의미**(CI 0 포함, P=0.858). 그러나 **구간 구조는 명확**:
  v33이 **소형(n≤4)을 팔아 중·대형 전 구간(5..10, n>34)을 샀다**.
- 원래 v30의 고질적 약점이던 **n>34가 0.698→0.825 (+0.127)** — 이것이 이 신호의 핵심.
  5..10도 +0.092. 대형·중형 양쪽에서 개선.
- **사용자 판단**: "소형 bag 희생할 만하다"는 쪽으로 기우는 중. (v30이 B2로 처음 고쳤던
  n≤4를 되돌리는 trade-off이므로, gate 관점에서 주의해서 볼 필요는 있음.)
- **한계**: ① **중간 checkpoint(epoch 64/150)** 기준 — 완주 후 재확인 필요. ② n>34는
  bag 수가 적어(약 13개) CI가 넓음(v33 0.825 [0.60, 0.98]). ③ synthetic legacy 회귀
  gate(§42, ≤0.01)는 별개로 아직 미검증 — top-up 완주 후 평가해야 함.

### 예측 산출물

- `predictions/musk_v30_baseline_best.pt`
- `predictions/musk_v33_armC_current_best.pt`

### 바로 다음

1. arm C top-up 완주(150 epoch) 후 §42 재평가(legacy 회귀 gate) + **Musk 재확인**(§45 신호가
   완주 후에도 유지되는지).
2. (논의) §45의 "대형 bag 개선 / 소형 희생"이 실질 개선이라면 Phase 0 결과 선택 기준 재검토.
3. frozen-v30 multi-resolution probe(Phase 1)는 Phase 0 결과 확정 후에만.

## 46. 2026-08-06 — PathoBench zero-shot 평가: per-task PCA 전처리 + 결과

**상태**: 실행 중인 arm C checkpoint(`v33_phase0_armC_ddp8_batch2` epoch 88)의
실세계 전체슬라이드 MIL(PathoBench) zero-shot 평가를 완료했다. 평가를 위해 **task별
8:2 분할 + train-only PCA(1536→512) 전처리 캐시** 파이프라인을 구축했다
(`scripts/prepare_pathobench.py`).

- **전처리 프로토콜 (사용자 확정, 2026-08-06 갱신)**: 각 task CSV의 train/test 분할을
  그대로 사용해 8:2로 나눈 뒤, **train 분할의 모든 타일에 PCA(1536→512)를 fit**하고
  train/test 모두 변환해 `data/pathobench/{task}_train.pt` / `{task}_test.pt`로 저장.
  - **타일 서브샘플링 없음** (기존 1024장/10만 샘플 제한 폐기). PCA는 두 패스
    (mean → centered covariance, float64 청크 누적)로 **전체 train 타일을 정확히** 사용.
  - **추론도 전체 타일 사용** (컨텍스트·query 모두 서브샘플 없음, `--max-tiles`/
    `--target-context-cells`/`--max-queries` 제거).
  - **bootstrap CI 폐기** — task가 많으므로 단일 테스트 결과만 출력 (CI 없음).
  - 평가는 `--data-dir` 기본 `data/pathobench`에서 캐시 우선 로드, 미존재 시 h5+PCA
    fallback. 캐시 형식: `{"slide_id": list, "bag": list[Tensor[n,512]], "label": list[int]}`.
  - **slide_id 문자열 캐스팅 추가** (BC_Therapy/CPTAC-CCRCC는 숫자 id라 pandas가
    int64로 읽어 h5 인덱스와 불일치 → 전부 누락 버그 수정).
  - **이진 task만 대상**: multi-class(BRACS 등) 제외. HerROI(`herroi_response`)는
    `features/HER2_tumor_ROIs_v3`가 빈 디렉토리라 피처 부재로 제외. **총 17개 이진 task**.
- **모델**: `epoch=088-val_ce_loss=0.5282.ckpt` (arch v24 내부, v30 `poolz_l2` +
  B2 log-uniform cardinality, 2026-08-06 16:03 저장, 아직 학습 진행 중).
- **결과 (zero-shot, sample-context 6/class, 전체 타일, seed 42, 단일 테스트)**:

  | task | test n | AUROC | Acc | BAcc |
  |---|---|---|---|---|
  | bc_therapy_er | 33 | 0.517 | 0.606 | 0.520 |
  | bc_therapy_grade | 33 | 0.538 | 0.545 | 0.531 |
  | bc_therapy_her2 | 33 | 0.542 | 0.455 | 0.510 |
  | cptac_brca_pik3ca | 21 | 0.582 | 0.333 | 0.429 |
  | cptac_brca_tp53 | 22 | 0.420 | 0.318 | 0.357 |
  | cptac_ccrcc_er | 33 | 0.517 | 0.606 | 0.520 |
  | cptac_ccrcc_grade | 33 | 0.538 | 0.545 | 0.531 |
  | cptac_ccrcc_her2 | 33 | 0.542 | 0.455 | 0.510 |
  | cptac_lscc_arid1a | 67 | 0.631 | 0.388 | 0.469 |
  | cptac_lscc_histologic | 57 | 0.597 | 0.596 | 0.610 |
  | cptac_lscc_keap1 | 51 | 0.590 | 0.510 | 0.484 |
  | cptac_luad_egfr | 59 | 0.637 | 0.458 | 0.526 |
  | cptac_luad_kras | 62 | 0.655 | 0.548 | 0.601 |
  | cptac_luad_stk11 | 67 | 0.682 | 0.522 | 0.621 |
  | cptac_luad_tp53 | 59 | 0.612 | 0.610 | 0.612 |
  | cptac_pda_smad4 | 55 | 0.309 | 0.509 | 0.438 |
  | ucla_lung_progression_regression | 22 | 0.598 | 0.682 | 0.645 |

- **해석**: sample-context(6 slide/class, 전체 타일)는 대부분 0.5~0.68의 랜덤~약상승
  수준. LUAD 계열(egfr/kras/stk11/tp53 0.61~0.68)과 LSCC(arid1a 0.631)가 상대적으로
  양호, PDA smad4는 랜덤 이하(0.309). **BC_Therapy와 CPTAC-CCRCC는 동일 슬라이드·동일
  라벨의 중복 데이터**(AUROC/Acc/logloss 완전 동일)로 확인. ⚠️ **§51(2026-08-07) 정정:
  이는 벤치마크 속성이 아니라 로컬 데이터 오류** — 로컬 `cptac_ccrcc_{er,grade,her2,residual}.csv`가
  `bc_therapy`의 바이트 단위 복사본이고, 공식 CPTAC-CCRCC 코호트(`C3L/C3N`)는 전혀 미포함이다.
  이전의 all-context(전체 train 슬라이드)가 sample보다 강했던 점(0.70~0.73)을 고려해, 전체 타일 기준
  all-context 재평가는 후속으로 가능.
- **파일**: `scripts/prepare_pathobench.py`, `scripts/test_pathobench.py`(갱신),
  `data/pathobench/{task}_{train,test}.pt` (17 task × 2), `predictions/pathobench_{task}_..._e88_full.pt`.
- **재실행**: `python scripts/test_pathobench.py --checkpoint <ckpt> --csv
  /NHNHOME/BASE/kimds/Data/PathoBench/csv/<task>.csv`. 전처리는
  `python scripts/prepare_pathobench.py --csv ...` 1회.

### all-context (전체 타일) — 5개 task 확장 (2026-08-06)

**상태**: sample-context가 대부분 랜덤~약상승이어서, 강세(LUAD)·약세(BRCA, PDA)·원래
벤치마크를 고르게 대표하는 **5개 task**를 `--context-mode all`(모든 train 슬라이드,
전체 타일)로 재평가.

- **OOM 수정**: 전체 타일 all-context를 패딩 dense 경로(`forward_episode_batch`)로
  돌리면 `[bags, max_cells, slots, dim]` 차이 텐서가 최대 bag 크기에 맞춰 폭발(69GB)
  → **ragged per-episode 경로**(`model.forward(x_list, y, mask_index)`, bag별 개별
  처리)로 전환. 메모리는 bag당으로 안전, 결과는 패딩 경로와 동일(1e-4 검증됨).
- **결과 (전체 타일, epoch 88, 단일 테스트)**:

  | task | test n | sample AUROC | **all AUROC** | Acc | BAcc |
  |---|---|---|---|---|---|
  | cptac_brca_tp53 | 22 | 0.420 | **0.696** | 0.545 | 0.616 |
  | cptac_luad_tp53 | 59 | 0.612 | **0.625** | 0.576 | 0.557 |
  | cptac_luad_stk11 | 67 | 0.682 | **0.786** | 0.776 | 0.785 |
  | cptac_lscc_arid1a | 67 | 0.631 | **0.748** | 0.821 | 0.648 |
  | cptac_pda_smad4 | 55 | 0.309 | **0.679** | 0.746 | 0.624 |

- **해석**: all-context가 전 task에서 sample 대비 개선. 특히 랜덤 이하였던
  **PDA smad4(0.309→0.679)**, **BRCA(0.420→0.696)**가 큰 반전. LUAD stk11 0.786,
  LSCC arid1a 0.748로 강세. LUAD tp53만 소폭(0.625). **컨텍스트 규모(전체 train
  슬라이드, 전체 타일)가 성능의 핵심 요인**임을 다시 확인.
- **파일**: `predictions/pathobench_{task}_armC_batch2_e88_allctx_full.pt` 5개,
  로그 `predictions/allctx_full_5.log`. BRACS coarse 데이터(캐시 2개 + 예측 2개)는
  multi-class 제외에 따라 삭제 (원본 `features/BRACS/`는 보존).

## 47. 2026-08-06 — 새 기준 checkpoint(e125) 재평가 + 타일 수 제한 실험

**상태**: **앞으로 모든 PathoBench 실험은 all-context 기준**(sample-context 폐기).
150 epoch 런의 best인 **`epoch=125-val_ce_loss=0.5142.ckpt`를 새 기준 checkpoint로
채택**하고, (1) e125로 5개 pathology all-context 재평가(val_ce 개선이 실제 test로
이어지는지), (2) bag별 타일 수 제한 실험을 실행 중.

- **스크립트 변경** (`scripts/test_pathobench.py`):
  - `--context-mode` 기본값 `all`로 변경 (sample은 deprecated, `--context-per-class`는
    sample 전용으로 유지).
  - **`--max-tiles`** 추가: bag(컨텍스트·query 모두)별 타일 상한. 지정 시 각 bag을
    trial별 랜덤 서브샘플.
  - **`--trials`** 추가: seed base + trial로 독립 추론 반복, trial별 지표 + 집계
    (mean/min/max) 출력. `evaluate_trial()`로 루프 추출.
  - 평가는 ragged per-episode 경로(`model.forward(x_list, y, mask_index)`) — bag별
    처리라 전체 타일 all-context도 메모리 안전 (§46).
- **Task 1 (진행 중)**: e125로 5개 task(`cptac_luad_tp53`, `cptac_luad_stk11`,
  `cptac_lscc_arid1a`, `cptac_brca_tp53`, `cptac_pda_smad4`) all-context 무제한.
  e88(val_ce 0.5282)과 e125(val_ce 0.5142) 비교 — val_ce 0.014 개선이 test AUROC로
  전이되는지 확인.
- **Task 2 (진행 중)**: bag별 타일 제한 `{1000, 2000, 5000}` × **5 trial**(랜덤
  서브샘플, trial별 seed) vs 무제한(1 trial). 인스턴스(타일) 수가 성능에 영향을 주는지
  확인.
- **실행**: `predictions/pathobench_{task}_armC_batch2_e125_allctx_full.pt` (무제한),
  `predictions/pathobench_{task}_armC_batch2_e125_mt{1000,2000,5000}.pt` (제한),
  로그 `predictions/pathobench_e125_allctx_tilesweep.log`.

### Task 1 결과 — e125(0.5142) vs e88(0.5282), all-context 무제한

val_ce 0.5282→0.5142 개선이 실제 test로 전이되는지 확인. test AUROC:

| task | e88 | e125 | Δ |
|---|---|---|---|
| cptac_brca_tp53 | 0.696 | **0.714** | +0.018 |
| cptac_luad_tp53 | 0.625 | **0.637** | +0.012 |
| cptac_luad_stk11 | 0.786 | **0.795** | +0.009 |
| cptac_lscc_arid1a | 0.748 | 0.738 | −0.010 |
| cptac_pda_smad4 | 0.679 | **0.710** | +0.031 |

**판정**: 5 task 중 4개 개선(평균 +0.012), 1개 소폭 하락. **val_ce 개선이 대체로 test로
전이됨.** e125를 향후 기준 checkpoint로 확정.

### Task 2 결과 — bag별 타일 수 제한 스윕 (5-trial mean vs 무제한 1-trial)

| task | 무제한(1 trial) | 1000 | 2000 | 5000 |
|---|---|---|---|---|
| cptac_luad_tp53 | 0.637 | **0.722** | **0.724** | **0.743** |
| cptac_luad_stk11 | 0.795 | **0.842** | **0.840** | **0.846** |
| cptac_lscc_arid1a | 0.738 | 0.694 | 0.670 | 0.696 |
| cptac_brca_tp53 | 0.714 | 0.652 | 0.655 | 0.671 |
| cptac_pda_smad4 | 0.710 | 0.592 | 0.616 | 0.703 |

(trial별 분포: 제한 케이스는 5 trial AUROC min/max, 로그 참조)

**해석**:
- **인스턴스(타일) 수는 성능에 뚜렷한 영향을 주며, 방향은 task 의존적.**
- **LUAD 계열은 타일 제한이 오히려 개선** (tp53 0.637→0.72~0.74, stk11 0.795→0.84).
  대형 bag(최대 ~3.5만 타일)이 노이즈/혼란을 유발하는 듯 — 대표 서브샘플이 더 강건.
- **BRCA/LSCC/PDA는 무제한이 우세** (제한 시 −0.04~−0.12), PDA는 5000에서 무제한과
  비슷(0.703). BRCA는 test 22장으로 trial 간 분산이 큼.
- **한계**: 무제한은 1 trial(결정적, 전체 타일) vs 제한은 5-trial mean(랜덤 서브샘플)
  이라 잡음 수준이 다름. 제한 케이스는 trial mean이 무제한 단일값과 비슷하거나 위면
  서브샘플이 무해~유익, 아래면 무해하지 않음. 전반적으로 **bag 크기 정규화의 효과가
  task별로 갈림** — 후속으로 LUAD 대형 bag 분석(어느 bag이 문제인지) 권장.

---

## 48. 2026-08-06 — arm C top-up 완주(150ep, best e125) + v33 Phase 0 평가 확정 + PathoBench v30 비교

**상태**: arm C top-up이 **150 epoch까지 완주**했다(8×A6000 DDP, 에피소드-매치
4096/epoch, batch2). val_ce 기준 best인 `epoch=125-val_ce_loss=0.5142.ckpt`
(`checkpoints/20260806_145050/v33_phase0_armC_ddp8_batch2/`)를 채택해 §42 legacy
회귀 gate 재평가 + §45 Musk 재확인 + PathoBench(v30 vs v33) 비교를 수행했다.
**결론: legacy 회귀 gate 여전히 미달(+0.041) — 과소학습 편향 가설 기각, B2b 데이터
자체가 회귀 원인. Musk n>34 개선은 유지되지만 PathoBench에서는 v30이 우위.**

- **완주 정보**: 150 epochs, best epoch 125(val_ce 0.5142). §43의 중간(~88/150)
  기준을 갱신. 이번 평가는 완주 checkpoint 기준 최종 판정이다.

### 1. 합성 legacy 회귀 gate 재평가 (e125, 1,000 ep, seed 42, v30 legacy B2 val 스트림)

- 실행: `evaluate_synthetic.py --checkpoint e125 --config
  configs/train_v30_cardinality_poolz_l2.yaml --val-episodes 1000
  --output predictions/synthetic_v33_phase0_armC_topup_e125_legacy_1000ep.pt`
- **결과**: overall AUROC **0.8100 [0.798, 0.822]**, log loss 0.5218.
  per-task: composition 0.8611 / state 0.7526 / covariance 0.6347 / interaction
  0.7547 / combined 0.9411.
- **v30 committed 대비**: 0.8100 vs 0.8512 [0.840, 0.862] → **회귀 +0.0412**.
  CI가 완전히 분리(0.798~0.822 vs 0.840~0.862)되어 통계적으로 확정. gate(회귀 ≤ 0.01)
  **실패**.
- **핵심 판독 — 과소학습 편향 가설 기각**: 50ep arm C(0.8139) → 완주 e125(0.8100)로
  사실상 변동 없음. val_ce는 0.5351→0.5142로 크게 개선됐는데 legacy AUROC는 회복되지
  않았다. 에피소드-매치(4096/epoch, 150ep)로도 회귀가 사라지지 않으므로 **B2b(per-bag
  cardinality) 데이터 자체가 v30 legacy B2 val 분포에서 성능 저하를 일으킨다.**

### 2. Musk 재확인 (e125, 102 bags leave-one-out, seed 42)

- 실행: `test_musk.py --data .../Musk/musk.pkl --checkpoint e125 --config
  configs/train_v33_phase0_armC_ddp8.yaml --output predictions/musk_v33_armC_e125.pt`
- **Overall**: AUROC **0.8616 [0.779, 0.932]** (v30 0.8539 → +0.008), Acc 0.696,
  BAcc 0.730 (sens 0.872 / spec 0.587), log loss 0.531.

| 밴드 | v30 baseline | arm C e64 (§45) | arm C e125 (완주) |
|---|---|---|---|
| ALL | 0.854 | 0.880 | **0.862** |
| n≤4 | 0.800 | 0.700 | **0.725** |
| 5..10 | 0.833 | 0.925 | **0.958** |
| 11..34 | 0.958 | 0.970 | **0.939** |
| n>34 | **0.698** | 0.825 | **0.849** |
| corr(prob, log n) | +0.059 | −0.146 | **−0.176** |

- **paired bootstrap (v30 vs e125, 4000 resample)**: ALL +0.008 [−0.062, +0.077]
  P=0.593(무의미), n≤4 −0.075 [−0.242, +0.058] P=0.144, n>4 +0.038 [−0.044,
  +0.123] P=0.812.
- **판독**: §45 신호(n>34 0.698→0.849)가 완주 checkpoint에서도 유지·개선. 5..10
  0.958. 소형 trade-off(n≤4 0.800→0.725)도 유지. overall +0.008은 무의미.

### 3. PathoBench v30 vs v33(e125) — all-context, 전체 타일, 1 trial

v30 baseline(`checkpoints/20260804_132334/v30_cardinality_poolz_l2/epoch=048-
val_ce_loss=0.4442.ckpt`)으로 §47과 동일 프로토콜(all-context, 무제한, 1 trial,
seed 42) 5-task 평가. 예측 파일 `predictions/pathobench_{task}_v30_allctx_full.pt`.

| task | v30 baseline | e125 (arm C) | Δ (v30−e125) |
|---|---|---|---|
| cptac_luad_tp53 | **0.7431** | 0.6366 | **+0.107** |
| cptac_luad_stk11 | **0.9154** | 0.7949 | **+0.121** |
| cptac_lscc_arid1a | 0.6214 | **0.7381** | −0.117 |
| cptac_brca_tp53 | **0.7857** | 0.7143 | **+0.071** |
| cptac_pda_smad4 | **0.7246** | 0.7101 | +0.015 |
| 평균 | **0.758** | 0.719 | **+0.039** |

**판독**:
- **v30이 5개 중 4개 task 우위, 평균 +0.039.** 특히 LUAD 계열(stk11 +0.121,
  tp53 +0.107)에서 크게 우세.
- 유일한 e125 승리는 **lscc_arid1a (+0.117)** — 최대 bag(4.4만 타일) task로, B2b의
  대형 bag 강점(§45 Musk n>34)과 같은 방향.
- 1 trial이라 task별 수치에 노이즈 있음(§47 타일 스윕의 5-trial 안정성 관찰 참조).
- 합성 legacy 회귀(+0.041)와 같은 방향 — **arm C(B2b)가 전통 분포·PathoBench 레짐
  모두에서 v30보다 약함.**

### 4. 종합 판정

- **Phase 0 두 주 효과 모두 gate 미달 확정**: arm B sparse 0.6747 (<0.75), arm C
  legacy 회귀 +0.0412 (>0.01). 과소학습 편향 가설 기각.
- arm C의 **Musk n>34 개선(0.698→0.849)** 과 **PathoBench lscc 개선**은 실측·재현되는
  실질 신호이나(대형 bag 강점), 전반적으로 v30 대비 열위 → **v30 baseline 유지,
  arm C(v33 Phase 0) 미채택**.
- **다음**: ① Phase 0 결과 선택(사용자) — n>34/대형 bag 개선을 실질 이득으로 볼지,
  소형 희생 trade-off를 감수할지. ② frozen-v30 multi-resolution probe(Phase 1,
  paired AUROC +0.01 headroom) — 아키텍처 가설은 여전히 미검증(§39). ③ ICI 잠금 유지.
- 예측 파일: `predictions/synthetic_v33_phase0_armC_topup_e125_legacy_1000ep.pt`,
  `predictions/musk_v33_armC_e125.pt`, `predictions/pathobench_{task}_v30_allctx_full.pt` (5개).
- 참고: `test_pathobench.py`에 `--context-max-tiles`(context만 절단, query 무제한) 옵션
  추가 — 컨텍스트 크기 격리 실험용(이번엔 미사용).
