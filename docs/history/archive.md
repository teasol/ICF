# Archived sections from current_status.md

## 2026-08-05 — v32/v32b DR-CCER proposal 종료

- 원안과 개선안은 각각 [`architecture_v32_dr_ccer_proposal.md`](architecture_v32_dr_ccer_proposal.md),
  [`architecture_v32b_dr_ccer_proposal.md`](architecture_v32b_dr_ccer_proposal.md)로 이관했다.
- Stage-0 P0–P3와 Stage-A 10-epoch 결과가 모두 음성(P1 standalone `0.51055`, P2
  `-0.00034`, P3 `+0.00000`, expert CE `0.6931`)이어서 CCER 계열을 폐기했다.
- v30 baseline은 유지하며 후속 활성안은
  [`../architecture_v33_multiresolution_bag_proposal.md`](../architecture_v33_multiresolution_bag_proposal.md)다.

This is the running archive for fully-resolved / superseded sections that were
moved out of `docs/current_status.md` to keep the living doc compact. Each
section keeps its original heading so cross-references still resolve.

- 2026-08-02: archived §4 (v22 retrieval removal decision), §9 (2026-07-31 session), §10 (2026-08-01 session) — all superseded by later sections (§3 v24 decision, §11 v25 retirement).

---

## 30. 2026-08-04 — v31 CCTS (Cardinality-Calibrated Tail Scan) 아키텍처 구현, Unit Test 통과 및 훈련 구동

**상태**: 코드 구현, `searchsorted` 메모리 최적화, Unit Test 통과 및 백그라운드 훈련 시작 ([v31_absolute_topk_tail_proposal.md](v31_absolute_topk_tail_proposal.md))

### 1. CCTS 구현 및 OOM 최적화 (`src/models/baseline.py`)
* **Context-Null Calibration**: Support Context 인스턴스들과의 경험적 Null 분포를 `torch.searchsorted` 기반 $O(M \log M)$ 알고리즘으로 텐서 메모리 조폭 할당(945GiB OOM)을 원천 차단하고 0-메모리 고속 연산 구현.
* **Expected-False-Positive Tail Scan**: $\lambda \in \{0.25, 1.0, 4.0\}$ 가짜 양성 예산 기반의 Soft Gate 및 5차원 신뢰도 메타데이터 연동.
* **Backward Compatibility**: `ccts_lambdas=()` 기본값 적용으로 기존 v30/v24 체크포인트 및 테스트와의 호환성 100% 보존.

### 2. 생성기 및 Config 연동 (`src/datasets/synthetic_data.py`, `configs/train_v31_ccts.yaml`)
* `SyntheticManifoldGenerator`에서 6개 과제 확률 정규화 및 `any_positive_sparse` 과제 호환 수술 완료.
* Config: `configs/train_v31_ccts.yaml` 작성 완료.

### 3. 검증 및 훈련 완료
* **Unit Test**: `tests/test_ccts.py` 작성 및 통과 (`Ran 1 test in 5.024s, OK`).
* **훈련 완료**: `v31_ccts` 50 Epoch 완료 (`val_ce_loss` 최저치 `0.4404` 달성).

---

## 31. 2026-08-05 — v31 CCTS 50 Epoch 완주, Musk Zero-shot 평가 및 대형 Bag 정체 정밀 분석

**상태**: 평가 완료 및 정밀 메커니즘 분석 완료 (`predictions/musk_v31_ccts_ep50best.pt`)

### 1. Musk Zero-shot 평가 수치 (`epoch=050-val_ce_loss=0.4404.ckpt`)
* **Overall AUROC**: **`0.8376`** (95% CI: `[0.756, 0.908]`)
* **`n <= 4` (소형 Bag)**: **`0.8333`** (v30의 0.8000 경신, 소형 구간 최고치)
* **`5..10` (중소형 Bag)**: **`0.8667`** (v30의 0.8250 경신, 중소형 구간 최고치)
* **`11..34` (중대형 Bag)**: **`0.9273`**
* **`n > 34` (대형 Bag)**: **`0.6032`** (Energy-Scaling 정석 보정 시 **`0.6111`**, Tiling 가중치 눈속임 시 `0.6984`)

### 2. 비판적 정밀 진단 (Forensic Diagnosis)
* **소형 Bag 성공 원인**: $n \le 10$ 소형 Bag에서는 $n \cdot \hat{p}_i \le \lambda$ 조건이 정확히 1개 세포만을 추출하여 무희석 Top-1 핀포인트 추출기로 동작 (`0.8333` / `0.8667`).
* **대형 Bag 실패 원인**: $n = 500 - 1000$ 대형 Bag에서는 추출 세포 수 $k$가 $n$에 비례하여 10 - 15개로 증가함으로써 활성 세포 1개 + 배경 세포 14개가 평균되어 신호 희석 재발.
* **차원 미스매치**: 166차원 원본 디스크립터를 Zero-padding함에 따라 512개 앵커 가중치 중 346개(67.5%)가 0과 곱해져 휴면(Dormant) 상태로 남아 스코어 스케일 왜곡 발생.

### 3. 정석적 해법
* 세포 개수 $n$과 무관하게 절대 1, 4, 8, 16개 세포만 단독 추출하는 **Absolute Top-K Tail (`absolute_tail_ks: [1, 4, 8, 16]`)** 및 166차원 특성을 512차원으로 직교 매핑하는 **Learned Read-Bridge ($W_{\text{bridge}} \in \mathbb{R}^{512 \times 166}$)** 도입 확정.

> **2026-08-05 재분류**: 위 결론은 CCTS 구현 결함 확인 전의 진단이다. Absolute Top-K와 Read-Bridge는 확정 해법이 아니라 독립 ablation 후보로 되돌린다.

---

## 32. 2026-08-05 — v31 CCER-Lite 구현 및 1차 학습 시작

**상태**: 구현·targeted test 완료, seed 42 학습 진행 중. v30 baseline 변경 없음.

### 구현

- `StructuredPopulationMetaClassifier`에 class-conditioned evidence router 추가.
- temperature `[0.25, 1.0, 4.0]`의 cardinality-normalized LogMeanExp route.
- support-class separation 기반 shared router, class-centering, explicit null gate.
- CCTS/Absolute Top-K는 신규 config에서 비활성화하고 기존 v30 rare branch를 control로 유지.
- Config: `configs/train_v31_ccer_lite.yaml`.

### 검증

- `tests/test_ccer.py` 3항목 및 기존 `tests/test_ccts.py` 통과: 총 4 tests.
- `PoolStandardizedBagRepresentationTest` 11 tests 통과.
- CCER router, support-router, null-threshold, residual 파라미터 gradient 확인.
- dense/list logit 동치 및 uniform instance duplication invariance 확인.
- merged config 모델 생성 성공: 9,453,478 trainable parameters.

### 실행

- Run time: `20260805_015000`
- Config: `configs/train_v31_ccer_lite.yaml`
- Seed / epochs: `42` / `50`
- Torchrun PID: `3210190`; worker PID: `3210265`
- Log: `logs/20260805_015000/v31_ccer_lite.out`
- Checkpoints: `checkpoints/20260805_015000/v31_ccer_lite/`
- CUDA sanity validation 통과 후 epoch 0 진행 확인.

이 run은 단일-seed 방향성 확인용이다. 승격은 synthetic/Musk 평가 및 후속 seed 반복 전까지 금지한다.

---

## 4. v22 결정: retrieval 완전 제거 (2026-07-29)

### 제거 근거 (3대 가설 검증 결과)

1. **retrieval은 ICI에서 이득이 없음**: retrieval을 끈 Phase 6c가 켠 Phase 6b와 AUROC 동일(0.5454 vs 0.5481)한데 Log Loss(0.7921 vs 0.8672)와 Accuracy(0.6092 vs 0.5747)는 오히려 **더 좋음**. ICI는 fold당 context가 ~69명뿐이라 24명 선별은 가용 labeled context의 65%를 버리는 것.
2. **구현이 문서화된 설계와 달랐음**: 설계는 "query 1명당 24명 맞춤 선별"이었으나, 실제로는 두 구현 모두 query 전체에 **공용 context 1세트**를 적용 (외부 collator는 첫 query만, 모델 내부는 query 평균 사용). 공용 context는 각 query의 개별 top-24와 평균 61.1%만 겹쳤고 반응자(y=1)에서 ~50%로 최악.
3. **Phase 5 사전학습은 Phase 1의 1/4만 학습됨**: `episode_batch_size` 8→32로 올리면서 `max_epochs: 20`을 그대로 둬 optimizer step이 10,240 → 2,560으로 감소. 게다가 validation이 retrieval 없이 수행되어 `epoch=014` 체크포인트가 학습 모드와 다른 기준으로 선택됨.

무엇보다 **위 차이들이 통계적으로 구분 불가능**하다는 점이 결정적이었습니다. 검출력 없는 지표 위에 복잡한 계층을 유지할 이유가 없다고 판단해 제거했습니다.

### 제거 범위

| 대상 | 조치 |
|---|---|
| `BaseModel.extract_bag_features` / `retrieve_context_indices` / `_retrieve_context_indices_impl` / `retrieve_context_indices_per_query` | 삭제 (303줄) |
| `BaseModel.forward(retrieval_k=...)` 파라미터 | 삭제 |
| `RetrievalEvaluationEpisodeCollator` / `RetrievalSyntheticTrainingEpisodeCollator` / `SignalAwarePretrainEpisodeCollator` | 삭제 |
| `ModelInterface`의 `retrieval_k` 배선 4곳 + `_build_model` pop-list 항목 | 삭제 |
| `scripts/test.py --retrieval-k` | 삭제 |
| retrieval 계열 config 12개, launch 스크립트 3개, VRAM 벤치 2개 | `configs/archive/v21_retrieval/`, `scripts/archive/v21_retrieval/`로 이관 |
| `tests/test_feature_retrieval.py`, `tests/test_large_context_pretrain.py` | 삭제 → `tests/test_batched_episode_forward.py`로 대체 |
| `architecture_version` | 21 → **22** |

**유지된 것**: v21 aggregator/meta-classifier 전부, 4대 수학 기술, batched multi-episode forward, 세션 중 고친 DataLoader CUDA/pin_memory 수정.

**복구 지점**: retrieval 최종 상태는 git tag **`v21-retrieval-final`** 로 보존되어 있습니다 (`git show v21-retrieval-final`).

### v20 롤백이 불가능했던 이유

사용자 요청은 "v20으로 롤백"이었으나 조사 결과:
* **v20은 코드로 존재하지 않습니다.** 이 브랜치 히스토리는 v18 → v19 → **v21**이며 (`ecf6199`가 19에서 21로 직접 점프), `main`은 아직 v18입니다. `configs/archive/v20/*.yaml`는 v19 코드 위에서 돌던 **설정 파일 시리즈**일 뿐입니다.
* **v21 ≈ v19 + retrieval + 사소한 2줄**. `ecf6199`가 baseline.py에서 제거한 실질 코드는 4줄뿐이고, "v21 4대 개혁"으로 문서화됐던 기술들은 이미 v19에 있었습니다.
* 따라서 retrieval 제거는 버전 롤백이 아니라 **덧붙은 계층의 절제**로 처리하는 것이 맞았고, 사용자 승인 하에 v22 신규 버전으로 진행했습니다.

---

## 9. 2026-07-31 세션 핸드오프 — v23/v24 bag collapse family

### 이번 세션에서 확정/진행한 것

- v23-A0 (exact mean) 50-epoch **완료**: best epoch 43 `val_ce_loss 0.5912154`
  (v22 0.5946 대비 -0.0034). §3 참고.
- v24-A0 (learned projection, slot 1) 50-epoch **완료**: best epoch 45
  `val_ce_loss 0.5976237`. slot 1개 정보 손실로 훈련 val에서 v22/v23보다 높음. §3 참고.
- v24-B0 (per-token bottleneck projection, slot 12 유지) 50-epoch **완료**:
  best epoch 46 `val_ce_loss 0.5923204`. v24-A0 대비 -0.0053 개선, v22 대비 -0.0023 개선. §3 참고.
- v24-B1 (residual mean + bottleneck projection, slot 12 유지) 50-epoch **완료**:
  best epoch 41 **`val_ce_loss 0.5903045` (전체 Bag-Collapse 모델 중 최저 기록 달성)**. §3 참고.
- 구현: v24 learned projection (`26b2b27`), v24-B0 병목 (`b2fb9d0`), v24-B1 residual 병목 (`4f984ca`).
  architecture_version 23/24 분리, 모든 단위 테스트 통과.

### 다음 Action (이 세션 당시 계획, 아래 §10에서 폐기됨)

1. ~~v23-A0 (epoch 43), v24-A0 (epoch 45), v24-B0 (epoch 46), v24-B1 (epoch 41) 4개 후보 체크포인트를
   동일 pool-400, 1,000 episodes, context `40/80/160/300`에서 평가.~~
2. ~~`scripts/compare_predictions.py`로 v22(`predictions/v22_medium_baseline_pool400_curve/`)와
   episode-cluster paired delta + CI 계산.~~
3. ~~판정: overall `+0.03` 또는 target task `+0.05`가 없으면 해당 후보 폐기.~~
4. 합성 Medium+Hard 후보가 확정되기 전까지 ICI는 실행하지 않습니다. **(이 항목만 유지됨 — ICI는 계속 잠금)**

---

## 10. 2026-08-01 세션 핸드오프 — v24 확정, 평가 계획 폐기

### 이번 세션에서 확정/진행한 것

- **사용자 결정**: §9의 4종 paired 비교 계획을 실행하지 않고 폐기. "단순히 label로 bag을 나누는 구조(class-memory 압축)가 별로"라는 문제 제기에서 시작해, v24-B1(residual + bottleneck bag projection)을 train `val_ce_loss` 순위만으로 최종 v24 아키텍처로 확정.
- v22(구 기준선), v23-A0, v24-A0, v24-B0 폐기. 상세는 §3 "최종 결정 (2026-08-01)".
- `configs/train_v23_medium_bag_mean.yaml`, `train_v24_medium_bag_proj.yaml`, `train_v24_medium_bag_proj_bottleneck.yaml`을 `configs/archive/v23_v24_candidates/`로 이관. `train_v24_medium_bag_proj_residual.yaml`이 v24 production entry point.
- `docs/architecture_v23_candidates.md`에 결정 기록을 남기고 `docs/history/`로 이관 (T5-A/B/C는 필요 시 재검토할 미실행 계획으로 보존).
- Git: 이 문서 갱신 커밋은 `d668d33`. `main`/`v24` 브랜치를 이 커밋(즉 `codex/v23-bag-mean` 최종 커밋)으로 fast-forward.

### 남겨진 것 (폐기되지 않음)

- T4 Medium→Hard bridge attribution (§6) — 이번 결정과 무관하게 계속 열려 있는 질문.
- v24용 ICI config 부재 — ICI를 돌리려면 `train_v22_ici_finetune.yaml`/`train_v22_ici_scratch.yaml`에 상응하는 v24 버전이 필요.
- v22 top-level config 6개는 ICI 파이프라인이 아직 참조하므로 archive로 옮기지 않음 — v24 ICI config가 만들어지면 재검토.

### 다음 Action

1. (선택) v24용 ICI finetune/scratch config를 만들고 ICI 실행 여부를 사용자에게 다시 확인.
2. (선택) T4 Medium→Hard attribution을 v24 위에서 이어갈지 결정.
3. 새 구조 변경이 필요해지면 이번처럼 train CE만으로 확정하지 말고, 최소한 §7 평가 프로토콜의 1,000-episode paired 비교를 다시 켤지 사용자와 사전에 정할 것.

---

## 12. 2026-08-02 세션 — 프로젝트 폴더 정리 (checkpoint/log/prediction purge)

사용자 지시로 폐기된 구버전 산출물을 정리했습니다. **코드/설정/문서는 불변**, 활성 run은
그대로 유지됐습니다.

### 삭제 내역 (모두 gitignore 대상 또는 git rm)
- **`checkpoints/` 53GB → 3.3GB**: v19/v20/v21/v22-era run 디렉터리 128개(20260722~20260729) +
  루트 느슨한 `.ckpt` 11개 + 폐기 확정 v23-A0/v24-A0/v24-B0 checkpoint
  (`20260731_v23_bag_mean_50e_resume`, `20260731_155635`, `20260731_182755`, `20260731_201252`).
  폐기된 candidate들의 수치는 `docs/history/v23_v24_bag_collapse_candidates.md`에 그대로 보존.
- **`logs/` 819MB → 532MB**: 20260722~20260729 dated 로그 + v19/v20-era named 로그
  (`tiranos/`, `pipeline_*`, `v19_covariance_candidates_*`, `2026072*_v19_*`) 삭제.
- **`predictions/`**: v19/v20/v21 ICI 예측 15개 삭제 (`ici_*.pt`, `ici_predictions_*.pt`,
  `v21_*_5fold.pt` 등). v22+ 합성 예측 및 curve는 유지.
- **v18**: `experiments/v18_learnability_c4_d_d0_d4.yaml`, `results/v18/` 삭제 (git rm).

### 유지 (활성/참조)
- checkpoints: `20260729_160643`(v22 baseline 참고), `20260731_035538`(v22 Hard),
  `20260731_220100`(v24 확정), `20260731_context300_ft`/`20260731_medium_context300_*`(T4),
  `20260801_020144`(v25), `20260801_075144`(v24-easy), `20260801_235601`(v25-easy, 진행 중).
- `data/`(14GB ICI 실데이터) 불변.

> [!NOTE]
> 향후 세션에서 이전에 참조되던 폐기 run checkpoint/log 경로를 찾을 경우, 해당 파일은
> 위 정리로 삭제됐고 수치만 문서에 남아 있습니다.

### Git 상태
- `git rm` 반영: `experiments/v18_*`, `results/v18/*`. 그 외 삭제는 전부 gitignore 대상이라
  git에 영향 없음. §11 "남겨진 것" item 1의 easy config 커밋은 이미 `b6aacf7`에서 완료됨
  (문서 스테일 — 여기서 해소).

---

## 13. 2026-08-02 세션 — 문서 및 config 정리 (docs/config/scripts purge)

사용자 지시로 v18/v19/v20-era 스테일 파일과 config를 정리했습니다. **코드(`src/`)와
활성 config·스크립트·living docs는 불변**, README/핸드오프 문서는 현재 상태로 갱신.

### 삭제 (git rm, 전부 구식/미참조)
- **루트 스테일**: `MODEL_ARCHITECTURE_KO.md`(v18 문서), `main.sh`/`main_medium.sh`/
  `main_minimum.sh`/`main_slurm.sh`/`test.sh`/`test_slurm.sh`(archive된 config·v16 checkpoint 참조).
- **구버전 스크립트** (`scripts/` 17개): `run_learnability_ladder.sh`,
  `run_sequential_pipeline.sh`, `run_v19_covariance_candidates.sh`, `sweep_csp_residual.py`,
  `benchmark_scalability.py`, `check_population_oracle.py`, `check_training_budget.py`,
  미참조 진단 `diagnose_{covariance_relations,covariance_subspace,local_geometry,
  anchor_candidates,context_size,tail_covariance,v19_branches,bag_label_selection,
  covariance_utilisation,oracle_covariance_upper_bound}.py`.
- **learnability-era 모듈 config** (`configs/` 14개): `callbacks/learnability.yaml`,
  `data/learnability_{a,b,manifold}.yaml`, `data/{synthetic,minimum}.yaml`,
  `model/covariance32.yaml`, `optimizer/adamw_learnability_5e4.yaml`,
  `scheduler/learnability.yaml`, `trainer/learnability_{a,b,d20}.yaml`,
  `trainer/{csp_short8,minimum_ddp8}.yaml`.

> [!WARNING]
> **이 삭제가 2026-08-02 이후 세션에서 `tests/test_learnability_ladder.py`의
> `test_d_stages_differ_only_in_selected_nuisance`를 깨뜨렸습니다** —
> `configs/trainer/learnability_d20.yaml` 삭제가 원인. §16(같은 날 이어진 세션)에서
> 전체 unittest 실행 중 발견, 이번 정리와 무관한 별도 작업이라 조치하지 않고 기록만
> 남겼음. 고치려면 이 config를 복원하거나 테스트를 갱신할 것.

### 유지
- `scripts/`: `train.py`, `evaluate_synthetic.py`, `compare_predictions.py`,
  `evaluate_protocol.py`, `power_analysis.py`, `launch_interactive_training.sh`,
  `launch_ici_protocol.sh`, living docs 참조 진단 4종(`diagnose_{cell_selection,
  state_upper_bound,selection_gain_curve,oracle_slot_alignment}.py`), smoke 유틸.
- `configs/` 최상위 10개 전부 (v24/v25/easy 활성 + v22 ICI/T4용) — 변경 없음.
- `configs/` 모듈 서브폴더의 generic 조각(default/medium/logger/adamw 등) — 유지.

### 문서 갱신
- `README.md` 전면 갱신: v18 설명·`main*.sh` 사용법 제거 → v24 아키텍처 + `scripts/
  launch_interactive_training.sh` 표준 런처 + 문서 맵으로 교체.
- `docs/agent_handoff.md` §7-1: config 최상위 유지 목록을 현재 상태로 갱신 (v23/v24-A0/B0
  config는 `configs/archive/v23_v24_candidates/` 이관 반영).
- `docs/current_status.md` §12는 이전 checkpoint/log purge, §13은 본 문서/config purge 기록.

### Git 상태
- 커밋 예정. 삭제 스크립트의 history 문서 내 참조(`learnability_ladder.md`,
  `v20_scalability_plan.md`)는 아카이브 문서라 그대로 둠.

---

## 14. 2026-08-02 세션 — src/scripts/tests 점검 및 v21 아카이브 정리

사용자 지시로 `src`/`scripts`/`tests`를 점검했습니다.

### `src/` — 변경 없음 (전부 사용 중)
동적 import 경로(`dataset_src`/`optimizer_src`/`scheduler_src` via
`src/modules/{data,model}_interface.py`의 `import_module`)로 8개 모듈 전부 런타임
참조를 확인 — 삭제 대상 없음.

### `scripts/` — v21 폐기 코드 제거 + 1건 복원
- **삭제**: `scripts/archive/v21_retrieval/` (`benchmark_vram.py`, `run_vram_quick.py`,
  `launch_phase{4,6,6b,6c}_5fold.sh`) — v21 완전 폐기. `scripts/archive/` 디렉터리 자체가
  비워져 제거됨. living docs 참조 없음 (history 문서만 참조).
- **⚠️ 복원**: `scripts/diagnose_context_size.py` — §13 정리에서 삭제했으나
  `test_context_size_diagnostic.py`가 import → `a5dfcf8^`에서 복원. 교훈: 스크립트 삭제 전 tests/
  의 `from scripts.* import` 의존성을 먼저 확인할 것.
- **유지 (사용자 결정)**: smoke 3종(`ddp_smoke.py` 8-GPU, `gpu_train_smoke.py` FP16,
  `medium_bf16_smoke.py` bf16) — 구식 정밀도/스케일이지만 보존.
- 참조 무결성 검증 완료: 남은 `src`/`scripts`/`tests`의 `from scripts.* import`가
  전부 실존 파일을 가리킴.

### `tests/` — 변경 없음 (10개 전부 유효)
- `test_learnability_ladder.py`는 실제로 fixed-episode-bank(`fixed_episode_count`) 기능
  테스트 (파일명 오해 소지, 기능 유효).
- 전체 unittest 실행은 이번엔 생략 (사용자 결정). 다음 코드 변경 시
  `timeout 1500s ... -m unittest discover -s tests -p "test_*.py"` 필수.


---

# Archived 2026-08-04 — IA-MIL 폐기 & 문서 정리 (해결·폐기된 세션/실험 섹션)

---

## 11. 2026-08-02 세션 핸드오프 — v25 Medium 평가 완료(사실상 동률), Easy tier 실험 진행 중

### v25(T5-A) Medium 1,000-episode pool-400 평가 — v24-B1과 사실상 동률, 승격 기준 미달

§3 "v25 (T5-A) 진행 중"에서 이어감. scratch 50-epoch 학습 완료: best `val_ce_loss 0.5939` @ epoch 24
(체크포인트 `checkpoints/20260801_020144/v25_medium_typed_bag/epoch=024-val_ce_loss=0.5939.ckpt`).
이번 세션에서 v24-B1과 함께 처음으로 1,000-episode pool-400 context curve를 평가했습니다
(v24-B1도 이전에는 이 평가를 받은 적이 없었음 — §3 "최종 결정"에서 스킵됐던 부분).

| context | v24-B1 AUROC [CI] | v25 AUROC [CI] |
|---:|---|---|
| 40  | 0.6774 [0.666, 0.688] | 0.6792 [0.668, 0.690] |
| 80  | 0.7223 [0.711, 0.732] | 0.7217 [0.710, 0.732] |
| 160 | 0.7688 [0.758, 0.778] | 0.7685 [0.759, 0.778] |
| 300 | 0.8036 [0.794, 0.812] | 0.8017 [0.792, 0.811] |

예측 파일: `predictions/v24_medium_bag_proj_residual_pool400_curve/`, `predictions/v25_medium_typed_bag_pool400_curve/`.
차이가 전부 CI 폭(±0.005~0.011)보다 훨씬 작아 **사실상 동률**입니다 — §6/§7 판정 기준(overall
+0.03 또는 target task +0.05)에 한참 못 미칩니다. 즉 **label 기반 class-memory를 우회하는
bag-preserving 분기를 추가해도 Medium에서는 아무 이득이 없었습니다.**

`scripts/compare_predictions.py`로 정식 paired win-rate 계산을 시도했으나 **주의**:
1. 첫 시도에서 두 예측 파일이 동일 파일명(`context_40.pt` 등)이라 스크립트 내부 dict가
   파일명(stem)을 key로 써서 하나가 덮어써지는 버그를 만났습니다 — 반드시 서로 다른 이름으로
   복사한 뒤 비교할 것 (`/tmp/.../scratchpad/compare_tmp/{v24b1,v25}_context_{size}.pt`로 해결).
2. 이 스크립트의 cluster bootstrap(5000 샘플, 순수 Python 루프)이 **매우 느립니다** —
   context 40 하나에 CPU 700%대로 수십 분이 걸렸습니다. 세션 재시작으로 한 번 유실되어
   `nohup ... & disown`으로 완전히 분리해 재실행했습니다 (PID `4153394`/`4153395`).
   **이 정식 paired win-rate 결과는 아직 안 나왔을 수 있습니다** — 다음 세션은
   `/tmp/claude-3564/-NHNHOME-kimds/f036ed83-abdf-4ebb-a5e9-d834e04e4f52/scratchpad/v25_vs_v24b1_compare_fixed2.log`를
   먼저 확인할 것. 위 표의 개별 AUROC+CI만으로도 결론(동률)은 이미 명확합니다.

### 🧪 Easy tier 실험 — "0.59가 이 아키텍처 계열의 물리적 한계 아니냐"는 질문 검증 중

사용자 관찰: v22~v25(mean/projection/bottleneck/residual/typed-bag) 전부 val CE
**0.5903~0.5976** 범위(폭 0.0073)에 몰려 있음. 과거 Tier 2 진단(§3 T2-1/T2-2: state AUROC가
관측 가능한 최선의 probe와 이미 동률, oracle은 0.88~0.90)과 맞물려 "bag summary 압축 방식이
아니라 이 아키텍처 계열 전체의 한계"라는 가설이 유력함. 이를 검증하기 위해 Hard tier가
건드린 축을 정확히 반대 방향으로 돌린 **Easy tier**를 만들어 v24/v25 두 아키텍처만 재학습·비교합니다
(6개 전부가 아니라 사용자 지시로 v24/v25만).

**Easy tier 정의** (`configs/train_v24_easy.yaml` 헤더 주석에 Medium/Hard/Easy 전체 표 있음):

| 축 | Medium | Hard | Easy |
|---|---|---|---|
| class_separation | [0.5, 1.4] | [0.2, 0.8] | **[1.2, 2.2]** |
| donor_shift_scale | 0.35 | 0.70 | **0.15** |
| donor_component_shift_scale | 0.12 | 0.25 | **0.06** |
| observation_noise | 0.01 | 0.05 | **0.002** |
| rare_response_probability | 0.15 | 0.25 | **0.05** |
| rare_response_fraction | [0.02,0.08] | [0.005,0.03] | **[0.05,0.15]** |
| response_mixture_effect_scale | 1.40 | (동일) | **1.80** |
| response_state_effect_scale | [0.45,1.00] | (동일) | **[0.90,1.60]** |
| response_covariance_effect_scale | [0.30,0.80] | (동일) | **[0.70,1.30]** |

`num_bags`/`num_cells`/`latent_dim`은 안 건드림 — 분리 가능성/노이즈 축만 바꿔 "신호가
많아지면 아키텍처가 갈리는가"만 순수하게 봅니다. Config: `configs/train_v24_easy.yaml`
(base: v24-B1 confirmed config), `configs/train_v25_easy.yaml` (base: v25 config). **아직 git에
커밋 안 됨(untracked)** — 다음 커밋에 포함할 것.

**진행 상황**:
- ✅ v24-easy: scratch 50-epoch 완료. Best `val_ce_loss` **0.3499** @ epoch 49
  (Medium 0.59 대비 압도적으로 낮음 — Easy tier가 실제로 훨씬 쉬움을 확인).
  Run `20260801_075144`, checkpoint `checkpoints/20260801_075144/v24_easy/epoch=049-val_ce_loss=0.3499.ckpt`,
  로그 `logs/20260801_075144/v24_easy.out`.
- ✅ **v25-easy: scratch 50-epoch 완료 (2026-08-02 03:49)**. Best `val_ce_loss`
  **0.3473** @ epoch 45 (v24-easy 0.3499 대비 $-0.0026$ 근소 우위).
  Run `20260801_235601`, checkpoint `checkpoints/20260801_235601/v25_easy/epoch=045-val_ce_loss=0.3473.ckpt`
  (top-3: e45 0.3473 / e47 0.3495 / e49 0.3494), 로그 `logs/20260801_235601/v25_easy.out`.
  > [!NOTE]
  > **속도 저하 10x 경고는 해소됨 (transient)**: 초기 epoch만 0.45~0.5 it/s로 느렸고
  > (epoch 0~1 기준), 후반 epoch는 4.3 it/s까지 회복. 총 50 epoch을 **약 3시간 50분**에
  > 완료 (16시간 예상보다 훨씬 빠름). 원인은 데이터 생성 warm-up/캐싱으로 추정되며
  > 지속적인 병목은 아님.

- ✅ **v24-easy vs v25-easy 1,000-episode 평가 완료 (2026-08-02 11:40)**:

  | 모델 | AUROC [CI] | Log loss |
  |---|---|---|
  | v24-easy | **0.9073** [0.901, 0.913] | 0.3828 |
  | v25-easy | **0.9106** [0.904, 0.917] | 0.3761 |

  - overall delta **+0.0033** — 승격 기준(+0.03)에 한참 미달. CI도 겹침.
  - Per-task AUROC (모두 v25 근소 우위이나 threshold +0.05 미달):
    composition 0.9439→0.9456 / state 0.8938→0.9010 / covariance 0.8186→0.8251 /
    interaction 0.8914→0.8954 / combined 0.9532→0.9546.
  - 예측: `predictions/v24_easy_1000ep.pt`, `predictions/v25_easy_1000ep.pt`.
    로그: `logs/20260802_easy_eval/easy_1000ep.out`.

- � **Easy tier 정식 paired 비교는 취소 (사용자 결정, 2026-08-02)**: marginal 평가
  (위 표)만으로 v24-easy ≈ v25-easy (delta +0.0033, 승격 기준 미달)가 확인됐고,
  v25는 어차피 폐기 확정이므로 paired win-rate 추가 계산 없이 종료.
  **결론: v24 확정 유지 (v25 폐기).**

### 🏁 v25(T5-A) 판정 근거 종합 (2026-08-02)

**Medium (1,000-episode pool-400, paired 5000 bootstrap — 이전 세션에서 완료된 로그를 기록)**:

| context | v24-B1 AUROC | v25 AUROC | P(v24-B1 beats v25) |
|---|---:|---:|---:|
| 40 | 0.6774 | 0.6792 | **0.04** (v25 우세) |
| 80 | 0.7223 | 0.7217 | 0.77 (구분 불가) |
| 160 | 0.7688 | 0.7685 | 0.65 (구분 불가) |
| 300 | 0.8036 | 0.8017 | **1.00** (v24-B1 압도) |

→ "사실상 동률"은 marginal CI 기준이며, paired bootstrap으로 보면 **맥락 의존적 trade-off**:
v25(typed bag-preserving)는 작은 context(40)에서 유의하게 우세, 큰 context(300)에서
v24-B1이 압도. 80/160은 구분 불가. 승격 기준(+0.03/+0.05)은 전 구간 미달.
(paired 로그: `/tmp/claude-3564/.../v25_vs_v24b1_compare_fixed2.log` — scratchpad,
repo 밖이므로 위 표로 기록 유지.)

**Easy**: v24-easy ≈ v25-easy (+0.0033, 승격 기준 미달).

**결론 (판정 기준 §11 item 4 적용)**: Easy tier에서도 두 아키텍처가 갈리지 않으므로
**"이 아키텍처 계열 전체의 한계" 가설이 강화됨.** v25(T5-A, typed bag-preserving)는
Medium/Easy 양쪽에서 승격 기준 미달 → **v25 최종 폐기 확정 (2026-08-02), v24 유지**
(Easy paired 추가 비교는 marginal delta +0.0033로 이미 승격 기준 미달임이 확인되어
사용자 결정으로 취소). 다음 방향: T5-B/T5-C 또는 완전히 다른 접근으로 이동 검토.
단, v25가 작은 context(40)에서 유의하게 우세했던 점은 ICI(~69 fold context)와 관련해
향후 새 설계 시 참고 가치 있음.

### 남겨진 것 / 다음 Action

1. ~~`configs/train_v24_easy.yaml`/`train_v25_easy.yaml` 커밋~~ → **완료** (`b6aacf7`에 이미 포함).
2. ~~v25 vs v24-B1 정식 paired win-rate 로그 확인, 문서에 최종 수치 기록~~ → **완료**
   (위 "판정 근거 종합" 표에 기록: v25 @40 우세, @300 열세, 80/160 구분 불가).
3. ~~v25-easy 학습 완료 대기 → v24-easy와 1,000-episode 평가로 비교~~ → **완료**
   (v24-easy 0.9073 vs v25-easy 0.9106, delta +0.0033).
4. Easy tier에서도 갈리지 않으므로 **"아키텍처 계열 전체의 한계" 가설 강화 → v25(T5-A)
   최종 폐기 확정 (2026-08-02), v24 유지**. Easy paired 추가 비교는 **사용자 결정으로
   취소** (marginal delta +0.0033로 이미 승격 기준 미달, v25는 폐기 확정).
   → 다음 방향: T5-B/T5-C 또는 완전히 다른 접근으로 이동 검토 (사용자 판단 필요).
5. ICI는 계속 잠금 (변경 없음).

---

---

## 15. 2026-08-02 세션 마무리 — 정리 3단계 + v25 폐기 확정 + 브랜치 정리

**세션 전체 요약**: 이번 세션은 (1) 폴더 정리, (2) 문서/config 정리, (3) src/scripts/tests
점검, (4) v25-easy 학습 완료 + Easy tier 평가, (5) v25(T5-A) 폐기 확정 + 브랜치 정리를
수행했다. 세부 기록은 §12~§14(정리), §11(평가·판정), §3(실험 현황) 참고.

### 확정/완료된 것
- **v25(T5-A) 폐기 확정**: Medium paired(맥락 의존 trade-off — v25 @ctx40 우세, v24-B1
  @ctx300 압도, 80/160 구분 불가) + Easy(v24-easy 0.9073 vs v25-easy 0.9106, Δ+0.0033)
  모두 승격 기준(+0.03/+0.05) 미달 → "아키텍처 계열 전체 한계" 가설 강화.
- **브랜치 정리**: `main` = v24 확정 SSOT로 fast-forward. v25 작업은 태그
  **`v25-typed-bag-final`**로 보존 후 로컬·원격 브랜치 `codex/v25-typed-bag` 삭제.
  v25 config는 `configs/archive/v25_typed_bag/`로 이관 (코드는 `typed_bag_*` gated로 잔존).
- **Easy paired 추가 비교는 취소** (사용자 결정): marginal Δ+0.0033로 이미 판정 가능.
- **정리 3단계**: checkpoints 53GB→3.3GB, logs 819MB→529MB, v19~v21 산출물·v18/learnability
  config·구식 스크립트 삭제, README/핸드오프 문서 현재화.
- **테스트 의존성 1건 복구**: `scripts/diagnose_context_size.py` (§13에서 삭제했으나
  `test_context_size_diagnostic.py`가 import → `a5dfcf8^`에서 복원).

### 오픈 문제 / 블로커
- **T4 Medium→Hard 성능 붕괴 attribution** (§6) — 여전히 열려 있음. v24 위에서 이어갈지
  결정 필요.
- **ICI 잠금 유지** — v24용 ICI config 부재, v24 확정이 §7 프로토콜 통과가 아님.
- **아키텍처 계열 한계 가설** — v22~v25 전부 val CE 0.59 근처/증분 미달. 다음 구조 변경은
  train CE 단독이 아니라 §7 프로토콜(1,000-episode paired)로 검증 (사용자 사전 합의 필요).

### 다음 단계 (사용자 판단 필요)
1. T5-B/T5-C 또는 완전히 다른 접근 설계 — 단, v25의 작은 context(40) 우세는 ICI(~69 fold)
   관점에서 새 설계에 참고 가치.
2. T4 attribution 재개 여부 (context-size curve는 §6에 이미 완료, raw-cell vs 40-token
   정보량 audit가 다음).
3. v24 ICI config 작성 후 ICI 실행 여부 재확인.

---

---

## 16. 2026-08-02 세션 (이어짐) — v26/v27/v29 설계안 검토, 학습 없는 게이트 3종,
## CLS-token pooling(v26) 구현·학습 시작, 제안서 archive

### 배경

§15 마무리 이후 같은 세션에서, 외부(다른 agent/사용자)가 작성한 신규 아키텍처
제안서 3건이 `docs/`에 추가됐습니다:
- `architecture_v26_proposal.md` — EC-MoE (Episode-Conditional MoE), 작성: DeepSeek V4 Pro
- `architecture_v27_proposal.md` — AC-ICAR (16-token + Riemannian branch), 작성: Antigravity AI
- `architecture_v29_proposal.md` — SP-SAT (40-token 보존 + slot-parallel self-attention),
  작성: Antigravity AI (사용자 아이디어 기반)

세 문서를 코드베이스 실측 근거와 대조해 비판적으로 분석했고(`docs/architecture_v28_proposal.md`
작성, 제 이름으로), 그 분석에서 나온 핵심 발견과 사전등록 게이트를 실제로 실행했습니다.

### `docs/architecture_v28_proposal.md` 전체 내용 이관 (archive 전 보존)

`architecture_v28_proposal.md`를 `docs/history/`로 archive하면서(docs 최상위에는
5개 living 문서만 두는 것이 원칙이라 — `agent_handoff.md` §6), 그 안의 실측 데이터와
근거를 여기로 전부 옮겨 적습니다. 원문 archive 위치는 §16 끝부분 참고.

**증거 카탈로그 (F1~F11, 전부 이 문서 다른 절에서 실측된 값)**:

| # | 사실 | 근거 |
|---|---|---|
| F1 | v22(40 token) `0.5946` vs v24(1 token) `0.5903` — 압축이 더 좋았다 | §3 |
| F2 | v25(bag-preserving, +4.4M params) Medium 동률, Easy Δ+0.0033 | §11 |
| F3 | v22~v25 val CE 전부 `0.5903~0.5976` (폭 0.0073) | §11 |
| F4 | state: 모델 `0.6217` = 모델 입력 토큰 probe `0.6210` | §3 T2-2 |
| F5 | state observable raw mean `0.5478`; oracle mask `0.8819~0.9013` | §3 T2-2 |
| F6 | effect scale 통일 시 covariance(0.6594)≈composition(0.6488), state만 최하위 | §3 T3-1 |
| F7 | context 40→300에서 v24 AUROC `0.6774→0.8036` (+0.126) | §11 |
| F8 | 세포 선택 점수 4종 전부 AUROC ~0.50; slot capture 0.155, fragmentation 0.963 | §3 T1-A/T1-B |
| F9 | 이득 곡선 선형: purity 0.11→1.00에서 covariance 0.517→0.888 | §3 T1-C 1 |
| F10 | bag 라벨 기반 세포 선택 purity 0.128 (무작위 0.110) → Tier 1 종료 | §3 T1-C 2 |
| F11 | v25가 작은 context(40)에서 유의하게 우세, 큰 context(300)에서 열세 | §11 |

**핵심 발견 (학습 없는 1,000-episode 감사, `diagnose_state_upper_bound.py`와 동일
스트림·분할 — T2-2의 `0.6217`/`0.9013`과 직접 비교 가능)**:

학습 파라미터 0개의 closed-form ridge가 v24의 slot별 충분통계(logit π, mean, log var)를
그대로 받으면 overall AUROC **0.700**이 나와, 학습된 9.45M 파라미터 v24(**0.708**)와
task별 ±0.02 이내로 사실상 동률입니다.

| variant | 차원 | ALL [95% CI] | composition | state | covariance | interaction | combined |
|---|---:|---|---:|---:|---:|---:|---:|
| bag_global (K=1, 분할 없음) | 32 | 0.6299 [0.619,0.640] | 0.7005 | 0.5538 | 0.5438 | 0.5715 | 0.7341 |
| v24 분할, π만 (≈현 ridge 입력) | 12 | 0.6240 [0.613,0.635] | 0.6611 | 0.5636 | 0.5410 | 0.5878 | 0.7245 |
| v24 분할, π+log σ² | 396 | 0.6569 [0.645,0.667] | 0.7315 | 0.5762 | 0.5593 | 0.6076 | 0.7656 |
| **v24 분할, π+μ** | 396 | **0.7001 [0.687,0.711]** | 0.7901 | 0.6186 | 0.5885 | 0.6441 | 0.8060 |
| v24 분할, π+μ+log σ² | 780 | 0.6947 [0.682,0.706] | 0.7837 | 0.6155 | 0.5842 | 0.6411 | 0.8016 |
| 동, 방향만 (반경 폐기) | 780 | 0.6972 [0.685,0.708] | 0.7864 | 0.6188 | 0.5854 | 0.6425 | 0.8037 |
| PCA k-means K=12 | 780 | 0.6908 [0.678,0.702] | 0.7730 | 0.5903 | 0.5886 | 0.6419 | 0.8061 |
| PCA k-means K=48 | 3120 | 0.6768 [0.664,0.689] | 0.7522 | 0.5804 | 0.5682 | 0.6227 | 0.8063 |
| **oracle 2-slot (responsive/배경)** | 130 | **0.9346 [0.929,0.940]** | 0.9374 | 0.9698 | 0.7585 | 0.9678 | 0.9790 |
| *참조: 학습된 v24/v22 모델* | 9.45M | *0.7078* | *0.7729* | *0.6215* | *0.6216* | *0.6628* | *0.8201* |

부수 결과: **π만(12차원)으로도 0.624** — 즉 v24의 `_abundance_ridge_logits`는 이미
상한의 대부분을 담고 있음. **반경 채널은 무관**(방향만 0.6972 vs 전체 0.6947, CI 겹침).
**K를 늘리면 나빠짐**(K=48이 K=12보다 −0.014, 분할 세분화는 답이 아님). **covariance만
모델(0.622)이 probe(0.589)보다 나음** → v24 전용 covariance branch는 실제로 일하고
있으므로 어떤 재설계에서도 유지해야 함.

**분할 품질 (1,000 episodes)**: v24(hard argmax, K=12) purity 0.2260/capture 0.1502,
PCA-32 k-means K=12 purity 0.2374, K=48 purity 0.3477/capture 0.0503. v24 soft
assignment의 정규화 엔트로피는 0.5404 — 할당이 뭉개진 게 아니라 **뭉치는 축이
responsive component가 아님**(capture 0.150 ≈ base rate 0.154).

**T1-C 2 재검정 (component 단위 다변량 규칙, 400 episodes) — 분할 발견 경로 세
번째로 폐쇄**: K=12/24/48에서 purity(선택) 0.19~0.23, base rate 대비 거의 무작위.
**oracle이 최적 slot을 골라줘도 purity 상한이 K=48에서 0.346** — 실패 원인은 고르는
규칙이 아니라 이 분할 계열 자체가 responsive component를 담아내지 못하는 것. 예외는
combined task(purity 0.431 @ K=48, base 0.183 — 세 채널이 동시에 걸려 신호가 가장
강할 때만 선택이 작동).

**결론**: v22~v25의 val CE 0.0073 정체는 아키텍처 탐색 실패가 아니라 "비지도
population slot으로 요약한 bag"이라는 특징 집합 자체의 정보 상한(≈0.70)이며, 그
상한 아래에서 토큰 구성·융합·routing을 바꾸는 제안(v23~v25 전부, 그리고 v26/v27/v29)은
구조적으로 ±0.02 안에서 움직입니다. 올바른 분할이 있으면 0.93까지 가능하지만
(oracle 2-slot), 그 분할을 관측값·bag 라벨로 찾는 경로는 세 번 독립적으로 닫혔습니다
(세포 선택 T1-A/T1-C 2, 분할 품질 §4.4, component 재검정 T1-C 2 재검정).

**v26(EC-MoE)/v27(AC-ICAR) 비판 요지** (전문은 `docs/history/`의 각 archived 문서 및
그 상단 배너 참고):
- 둘 다 동기 수치가 Hard tier 값을 Medium 논증에 섞어 씀 (실측 Medium: state 0.6215,
  covariance 0.6216 — Hard 값 0.52/0.55가 아님).
- v24 융합은 `global(계수 1.0, 고정) + sigmoid-gated residual 3개`인데, 둘 다 이를
  softmax simplex(`Σg=1`)로 바꾸자고 제안 — simplex 제약이 지배 항 global의 계수를
  1.0→0.2~0.5로 축소시켜 logit 크기가 붕괴하는 설계 버그.
- v26의 Top-2 sparse routing은 이 구조에서 연산 절감이 없음(어차피 4 branch 전부
  계산) — 불연속 gradient만 추가.
- v26의 load balancing(`1/K` 목표)은 token 단위 routing용 공식을 episode 단위
  routing(`episode_batch_size=8`)에 그대로 씀 — `f_k` 추정 불가.
- v24에서 slot-level population routing은 이미 **no-op**임(`project_structured_tokens`
  후 bag당 토큰이 1개라 softmax가 항상 1.0) — 그런데도 v22와 성능이 같았음(F1). 이
  경로 위에 더 정교한 routing을 얹는 v26/v27 전제를 약화시키는 강한 증거.
- v27의 16-token은 v22 40-token의 부분집합에 가까움(slot spread 12채널 소실).
  eigenvector 기반 covariance basis token은 부호/회전이 임의라 잘 정의된 함수가 아님.
- v27의 Riemannian branch는 실측 결과 실행 불가: 512×512 batched `eigh`가 step당
  1.99s로 v24(0.135~0.273s) 대비 7~15배, 인접 고유값 최소 간격이 `2.97e-07`이라
  backward의 `1/(λi−λj)` 항이 `bf16-mixed`에서 불안정(이 저장소가 bf16-mixed를
  강제하는 정확히 그 이유와 충돌). shrinkage는 condition number만 고치고 eigen-gap은
  못 고침(uniform shift는 간격 보존).
- v27은 T2-2를 오독 — raw mean probe(0.5478)와 oracle(0.8819)의 격차를 "요약
  손실"로 돌렸지만, 실제로는 모델 입력 토큰 probe(0.6210)가 이미 raw mean보다
  훨씬 높고 모델(0.6217)과 동률 — 격차의 원인은 요약 방식이 아니라
  `responsive_instance_mask`(정답 세포 선택) 접근 여부.

**남은 두 경로 (§5, E7/A4로 각각 사전검정 완료 — 결과는 바로 아래 표)**:
- **경로 A (저위험)**: context/label 효율. 후보 우선순위 4→1→2→3: (4) bag을 반으로
  쪼개 ridge 유효 n을 2배로(재학습 불필요, A4로 검정 완료) → (1) block별 shrinkage
  학습(현재 λ는 전역 스칼라 1개, context 작을 때 자동으로 강한 정규화) → (2) context
  크기를 명시 조건화 → (3) 작은 context(40~80)에 집중한 학습 분포.
- **경로 B (고위험·고수익)**: `responsive_instance_mask`(세포별)로 **분할 모듈만**
  보조 손실 학습 `L = L_CE + 0.10·L_rank + λ1·CE(할당, mask) + λ2·‖π̂-π_oracle‖²`.
  분할을 미분 가능하게(현재 `_context_anchors`는 고정 random 버퍼 + argmax, gradient
  안 흐름 — episode 공유 prototype + bag별 mixing weight, soft E-M으로 교체 필요).
  추론 시 오라클 미사용, meta-training에서만 사용. 위험: 합성 특화 과적합(ICI 전이
  실패), §4.5 결과상 관측값 신호가 거의 없어 학습 자체가 안 될 수 있음(→ E7로 사전검정).
  구현 시 `slot_std`가 `[bags,cells,K,512]`를 물리 생성하는 부분(`baseline.py:903`)을
  `E_k[x²]-(E_k[x])²` 항등식으로 바꿔야 K 확장이 무료가 됨.

**근거 코드 위치**: 분할 미학습(`baseline.py:501-503,737-745,747-817`) · slot별
통계 이미 계산(`:898-915`) · identity-aligned ridge가 metadata 2채널만 받음
(`:915,2774-2830,3345-3348`) · slot 경로가 방향만 받음(`:547-564`) · v24 slot
routing no-op(`:2570-2580,2614-2620`) · additive residual 융합(`:2766-2773`) ·
routing entropy/balance loss 존재하나 weight 0(`model_interface.py:530-550`) ·
K 확장 병목(`baseline.py:903`) · oracle 특징=(fraction,mean,variance)
(`synthetic_data.py:550-563`) · oracle abundance plumbing 존재, 최적화 미참여
(`:1152-1157`, `model_interface.py:646`) · covariance는 64-d random projection
sketch(`baseline.py:512-518,566-600`) · 반응 효과는 component 1개에만
(`synthetic_data.py:458-511`) · 관측 manifold가 episode마다 새 random MLP
(`:754-767`).

### 사전등록 게이트 3종 실행 결과 (§6.1, 전부 1,000/1,000/400 episode 실측)

| 게이트 | 무엇을 검증 | 결과 |
|---|---|---|
| **E2** | v26/v27의 gating 전제: 학습된 v24 고정, episode마다 fusion 가중치를 query 라벨에 대해 직접 최적화한 oracle 상한 | **FAIL** — 최적 λ에서 delta 정확히 `0.0000`. 모든 task도 ±0.0002 이내. **v26 폐기 확정, v27의 routing 부분 폐기 확정** |
| **E7** | v29가 재사용하는 population-slot 분할의 지도학습 상한 (세포 라벨로 held-out Fisher 판별) | **INCONCLUSIVE** — overall purity `0.3351` (게이트: ≥0.50 진행/<0.30 폐기, 그 사이). covariance만 단독으로 게이트 아래(`0.2726`) |
| **A4** | context bag을 반으로 쪼개 ridge 유효 표본만 늘리는 재학습 없는 조작 | **약함** — ctx40 `+0.0035`(P=0.950)로 방향은 맞으나 ctx160에서 소멸(`-0.0001`). 3시간 투자(A1)를 정당화하기엔 근거 부족 |

세 게이트 모두 "확실한 승격"을 주지 못했고, 이는 §4.3의 0.70 상한 가설과 일관됩니다.

### v26/v27/v29 최종 판정 — 전부 미구현 폐기, `docs/history/`로 이관

- `docs/history/architecture_v26_proposal_ec_moe_rejected.md` — E2로 폐기
- `docs/history/architecture_v27_proposal_ac_icar_rejected.md` — v22<v24 선례 + E2 +
  Riemannian branch 실측 비용(§3.4: 512×512 batched `eigh` step당 1.99s, v24 대비
  7~15배, 인접 고유값 간격 `2.97e-07`로 `bf16-mixed` backward 불안정 위험)로 폐기
- `docs/history/architecture_v29_proposal_sp_sat_rejected.md` — v22<v24 선례 + E7(분할
  상한 불확실)로 폐기
- `docs/architecture_v28_proposal.md`(제 분석/실측 문서)도 `docs/history/`로 이관했습니다
  (2026-08-02, 사용자 지시 — "docs 최상위에는 5개 living 문서만" 원칙 유지). 원문의
  실측 데이터·근거·code reference는 전부 위 절로 옮겨 적어서 유실 없음.

### 대안: 사용자 제안 CLS-token pooling → `architecture_version=26`으로 실제 구현

v29(SP-SAT)의 문제의식(40토큰이 이미 압축돼 정보가 없다)에는 동의하되, 그 해법(40토큰
위에 self-attention을 더 크게)은 "이미 있는 정보를 재조합할 뿐 새 정보를 만들지 못한다"는
논리로 반대했습니다. 대신 사용자가 제안한 대안 — **raw cell 전체(N개)를 학습된 CLS
cross-attention으로 직접 요약해 41번째 토큰으로 추가** — 는 기존 population-slot 분할에
전혀 의존하지 않는 별도 정보 경로라서 §4.3/E7의 상한 논리에 안 걸립니다. 이걸 구현했습니다.

- **설계**: `ClassTokenPooling`(`src/models/baseline.py`) — 학습되는 CLS 쿼리가 bag의
  모든 raw cell을 key/value로 cross-attention (self-attention 아님 — cell 수 최대
  1,500 기준 O(N²) 대신 O(N)으로, v27이 겪은 종류의 비용 폭증을 피함). 기존 40개
  구조화 토큰은 완전히 그대로 두고 41번째로 concat. `cls_token_pooling: true`로만
  켜짐(기본 off, 꺼져 있으면 v24와 100% 동일).
- **버그 발견 및 수정**: unit test(단일 에피소드 경로)는 다 통과했지만, 실제 학습에
  쓰이는 **4D 배치 경로**(`forward_batched`, `_class_memories_batched`,
  `_population_memory_logits_batched`)에 "40토큰 조립" 로직이 **별도로 3중 인라인
  복제**되어 있어 cls_token을 빠뜨리고 있었습니다. 공용 헬퍼
  `_all_structured_tokens_batched`로 통합해 수정 (실제 벤치마크 스크립트로 실행해보고서야
  발견 — unit test만으로는 안 잡히는 유형의 버그였음).
- **검증**: 신규 unit test 9개 추가, 전체 unittest **152/153 통과**(나머지 1개는
  `configs/trainer/learnability_d20.yaml` 삭제로 인한 기존 결함, `a5dfcf8`에서 이미
  삭제됨 — 이번 변경과 무관, 조치하지 않음). 실제 v24 config 기준 벤치마크: **step time
  +6.9%, 파라미터 9.45M→11.62M(+23%), VRAM 증가 거의 없음**. 1-epoch 스모크 트레이닝
  (실제 Lightning 루프) 크래시/NaN 없음 확인 후 본 학습 착수.
- **Config**: `configs/train_v26_medium_cls_token_pool.yaml` (base: v24 확정 config).
- **학습 실행 중**: Run `20260802_225848`, scratch Medium 50 epoch (v22~v25와 동일 방식,
  warm-start 아님). PID `375725`(launcher).
  - 학습 로그: `logs/20260802_225848/v26_medium_cls_token_pool.out`
  - Launcher 로그: `logs/20260802_225848/v26_medium_cls_token_pool_launcher.out`
  - 체크포인트: `checkpoints/20260802_225848/v26_medium_cls_token_pool/`
  - Sanity check 통과, ~4 it/s로 정상 진행 확인 (2026-08-02 22:59 KST). 512 steps/epoch ×
    50 epoch 기준 완료까지 약 1.8~2시간 예상.
  - **진행 업데이트 (23:24 KST)**: epoch 11/50 진행 중, PID `375803`(torchrun worker)
    생존 확인. Checkpoint 저장 정상: `epoch=007-val_ce_loss=0.5954.ckpt`,
    `epoch=008-val_ce_loss=0.5947.ckpt`, `epoch=010-val_ce_loss=0.5955.ckpt`.
    Best `val_ce_loss 0.5947`@epoch8 — v24 확정 기록(`0.5903`@epoch41)에 근접하지만
    아직 미달이며, 학습 초반(epoch 8/50)이라 **이 시점 비교는 아무 의미가 없습니다**.
    반드시 50-epoch 완주 후 최종 best checkpoint로 §6/§7 프로토콜 평가할 것.

> [!IMPORTANT]
> **v24는 여전히 확정 baseline입니다.** 이 학습은 아직 평가 전이므로, 완료 후 §6/§7
> 프로토콜(1,000-episode paired, overall +0.03 또는 target task +0.05)로 v24-B1과
> 비교하기 전까지 v26을 확정/승격하지 않습니다.

### 다음 Action
1. 학습 완료 대기 (`logs/20260802_225848/v26_medium_cls_token_pool.out` 확인).
2. best checkpoint로 1,000-episode pool-400 context curve 평가 → v24-B1과 paired 비교.
3. 승격 기준(+0.03/+0.05) 통과 여부로 v26 확정/폐기 판정.
4. scratchpad의 검증된 진단 스크립트(`probe_oracle_gating.py`,
   `probe_component_selection_bound.py`, `probe_split_context.py`)는 아직 `scripts/`로
   이관·커밋되지 않음 — 필요 시 처리.
5. E2b(fusion scale 상한 해제 FT)는 E2 결과상 생략 권고, 사용자 판단 필요.

---

## 17. 2026-08-03 세션 — v26 학습 완료 + CLS attention 진단 프로브 (24-CLS 제안 사전검정)

**배경**: v26(CLS-token pooling) scratch 학습 완료 — best `val_ce_loss 0.5908`@epoch 49
(checkpoint `epoch=049-val_ce_loss=0.5908.ckpt`, `last.ckpt` 동일 값). v24 확정 `0.5903`과
사실상 동률(Δ+0.0005 열세). 사용자가 "CLS 토큰 24개 + 토큰 간 self-attention"으로 확장하는
방안을 제안. §16의 승격 기준상 1,000-episode 평가가 원칙이지만, 사용자는 로스 기준으로
"0.1 이상 개선은 나올 수 없다"며 평가 skip을 결정하고 **재학습 없는 사전검정 프로브만** 요청.

**프로브 (access-capacity 구분)**: `scripts/diagnose_cls_attention.py` — 학습된 v26
`last.ckpt` 사용, val stream(seed 50042)에서 1,000 episodes, `_normalize_bags`+`_bag_view`로
`_forward_dense`가 `ClassTokenPooling`에 주는 입력을 그대로 재현 후 cross-attention
`need_weights=True`로 per-head 가중치를 뽑아 cell-level로 `responsive_instance_mask`와 대조.
**소스 코드 수정 없음.** 실행: `--episodes 1000`, 산출물
`logs/20260802_225848/v26_cls_attention_probe.csv`.

**결과 (cell level, 0.5 = 무작위; T1-A 세포선택은 ~0.50)**:

| 지표 | 값 |
|---|---|
| 전체 AUROC (head 평균) | **0.5027** |
| per-head AUROC | h0 0.4982 / h1 0.5017 / h2 0.5037 / h3 0.5006 (best 0.5037) |
| per-task AUROC | composition 0.5042, state 0.4974, covariance 0.5044, interaction 0.4980, combined 0.5041 |
| attention mass share on responsive | 0.1534 (base rate 0.1529) → **lift 1.003×** |
| per-bag AUROC 평균 (양 클래스 bag만) | 0.5014 |

**판정 — access-limited (b), 24-CLS+self-attention 전제 반증**: 학습된 CLS cross-attention이
반응세포를 전혀 선택하지 못함. attention이 사실상 **균등(uniform, lift 1.003×)** → CLS 토큰은
전역 평균과 다름없는 readout으로 붕괴 → v26 ≈ v24 동률의 직접적 원인. "readout 용량 부족(1개 →
24개)"이 아니라 "**관측 manifold에서 반응세포 신원에 접근 불가**"가 병목임을 확정. 이는
T1-A(세포선택 ~0.50)·T1-C(purity 0.128~0.23)·split-quality capture 0.150≈base rate와 **4번째
독립 확인**이며, §16의 0.70 정보 상한(비지도 slot 요약 기준)은 readout 용량이 아니라 세포 신원
접근 문제라는 결론을 강화. oracle 2-slot 0.93의 격차는 "어느 세포가 반응인지 아는 것"에만 달려
있고, 그 지식을 관측값·학습 readout으로 얻는 경로는 이제 네 번 모두 닫힘.

**권고**: 24 CLS 토큰 + self-attention **미추진** (2h scratch 학습 비용 대비 기대 이득 없음 —
학습된 attention이 균등이므로 24개를 늘려도 균등 readout 24개일 뿐). 남은 방향은 §16의
**경로 A(저위험: context/label 효율)** — (4) bag 분할 ridge 유효 n 2배(A4 약효) → (1) block별
shrinkage 학습 → (2) context 크기 명시 조건화 → (3) 작은 context(40~80) 학습 분포 — 또는
ICI 관점에서 v24/v26용 config 작성 여부 재확인.

**산출물/명령**:
- 스크립트: `scripts/diagnose_cls_attention.py`
- 결과: `logs/20260802_225848/v26_cls_attention_probe.csv`
- 재실행: `python scripts/diagnose_cls_attention.py --config configs/train_v26_medium_cls_token_pool.yaml --checkpoint checkpoints/20260802_225848/v26_medium_cls_token_pool/last.ckpt --episodes 1000`

**다음 Action (사용자 판단 필요)**: ① v26 1,000-episode 평가 skip 유지 여부(결론에 영향 없음 —
동률 확정이면 v26 폐기 경로), ② 경로 A 재개 여부, ③ v24/v26 ICI config 작성 여부.

---

## 18. 2026-08-03 — E7 재검정: 지도 component-selection 상한 재확인 (Path B 관문)

**배경**: §16의 E7 게이트가 INCONCLUSIVE(ALL purity 0.3351, 게이트 0.30~0.50 사이)로 남아
Path B(미분 가능 분할 + oracle mask 보조 손실, ~2h 학습) 착수 여부 판정이 막혀 있음. 원래
scratchpad 스크립트(`probe_component_selection_bound.py`)가 커밋되지 않아 저장소에 없어,
문서화된 방법론(`docs/history/architecture_v28_analysis_ceiling_and_gates.md` §6.1)대로
`scripts/diagnose_component_selection_bound.py`를 새로 구현·재검정. **재학습 없음, 모델 로딩 없음**
(순수 세포 기하).

**방법**: v24 분할 관점에서 각 bag의 세포 단위 **held-out Fisher 판별**(bag 정확히 반분 fit/held,
bag당 4회 독립 분할 평균, d=(mean_pos−mean_neg)/var) → held-out AUROC + **purity@k**(모델이 실제
쓰는 선택 비율 1/5/10/15/20%), 1,000 episodes, bag-level bootstrap 95% CI. 실행:
`python scripts/diagnose_component_selection_bound.py --config configs/train_v24_medium_bag_proj_residual.yaml --episodes 1000`

**결과 — 기존 E7 재현 확인 + 전 구간 스윕**:

| task | bags | base | heldAUROC | p@1% | p@5% | p@10% | p@15% | p@20% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 77,558 | 0.157 | 0.7167 | 0.476 | 0.395 | **0.351** | 0.322 | 0.301 |
| composition | 15,261 | 0.208 | 0.6855 | 0.458 | 0.403 | 0.371 | 0.350 | 0.335 |
| state | 14,204 | 0.120 | 0.7450 | **0.514** | 0.410 | 0.351 | 0.313 | 0.285 |
| covariance | 16,482 | 0.115 | 0.6980 | 0.421 | 0.332 | **0.286** | 0.258 | 0.238 |
| interaction | 15,812 | 0.139 | 0.7331 | **0.504** | 0.410 | 0.357 | 0.324 | 0.298 |
| combined | 15,799 | 0.204 | 0.7246 | 0.485 | 0.425 | 0.391 | 0.368 | 0.351 |

기존 E7 대비: ALL heldAUROC 0.7167 vs 0.7207, ALL purity@10% 0.3507 vs 0.3351, covariance purity
0.286 vs 0.2726 → **재현 성공** (원래 E7은 약 10% 선택 기준이었던 것으로 보임).

**판정 — 게이트 문자상 여전히 INCONCLUSIVE(0.30~0.50), 그러나 증거 무게는 Path B 기각 쪽**:
- **가장 유리한 1% 선택에서도 ALL purity 0.476 < 0.50** (모델이 실제 쓰는 5~15% 구간은 0.32~0.40).
- Path B가 가장 직접 노리는 **covariance는 전 구간에서 폐기 문턱 아래(0.286@10%)** — F6/covariance는
  cell-selection이 아니라 dispersion 신호라는 기존 결론과 정합.
- enrichment는 2~3x에 그침. 세포 라벨을 직접 주는 부정행위 상한에서조차 이 수준.
- 유일한 밝은 구간은 **state/interaction @1%(0.514/0.504)** 뿐인데, 이는 비현실적으로 공격적인 선택
  비율 + 세포 라벨 부정행위 상한이고, bag 라벨만 받는 실제 조건(T1-C 2: purity 0.128)에서는 도달 불가.
- T1-C 1 이득 곡선(순도 0.40 → covariance +0.107, 그러나 covariance는 에피소드의 5% → 전체 ~+0.005)과
  합치면, **Path B의 기대 전체 이득은 작고 ICI 과적합 리스크를 정당화하지 못함**.

**결론**: Path B를 **full-path 투자로 추진하지 않는다**는 방향 강화. 남는 협소한 선택지는 §6.1의 원래
권고대로 "composition/combined 위주로만 범위를 좁힌 실험"(purity 0.36~0.39로 상대적 최고)뿐이며,
이마저도 사용자 판단 필요. 나머지 실질 후보는 §17과 동일: 경로 A(저위험) 또는 ICI 관점.

**산출물/명령**:
- 스크립트: `scripts/diagnose_component_selection_bound.py`
- 결과: `logs/e7_retest_20260803.csv`
- 재실행: 위 명령 (`--episodes` 조정 가능)

---

## 19. 2026-08-03 — 정규화 천장 프로브: 고정 정규화가 천장을 제한하는가 (사용자 가설 검증)

**배경**: 사용자 가설 — "제거할 패턴(기증자/배경 구조)은 cross-feature이고, 현재 per-feature centering +
per-cell L2 정규화(`_bag_view`)가 그걸 잘못 제거하거나 정보를 잃는다. 지울 패턴을 배우면 천장이
올라갈 수 있다." 이를 재학습 없이 검증.

**방법**: `scripts/diagnose_normalization_ceiling.py` — 모델 없이, 각 에피소드 세포를 5가지 정규화로
변환 후 per-bag 충분통계(mean/mean_var/cov-sketch)로 **닫힌 해 ridge**(λ=1.0, 0 학습 파라미터) 천장
측정. 1,000 episodes, seed 50042. `whiten_ctx` = **context 세포로 whitening 변환을 학습해 적용**
("지울 패턴을 배운다"의 저비용 구현, query 누출 없음). 참조: F-시리즈 v24 slot ridge 0.700, bag_global 0.630.

**결과 (ALL AUROC, mean+var)**: **centered 0.6846 [0.675,0.696]** > current 0.6363 [0.626,0.648] >
whiten_ctx 0.5972 > raw 0.5748 > zscore 0.5054. Per-task(mean_var)에서 centered가 전 task 최고
(composition 0.760/state 0.621/covariance 0.599/interaction 0.639/combined 0.783); covariance에서
centered 0.599 vs whiten_ctx 0.528. 산출물 `logs/normalization_ceiling_20260803.csv`.

**판정 — 사용자 가설은 "부분 확인 + 제안 방향은 실측 기각"**:
1. **부분 확인 (정량화됨)**: `current`의 per-cell L2 정규화가 편차 크기(magnitude)를 지워 충분통계
   수준에서 천장을 **−0.048**(0.685→0.636) 떨어뜨린다. "정규화 토큰이 정보를 잃는다"는 직관은
   맞고, 그 비용이 이제 측정됨.
2. **"지울 패턴을 배운다"(whitening)는 더 나쁨**: whiten_ctx(0.597) < current(0.636) < centered(0.685).
   이유는 명확 — whitening이 제거하는 cross-feature 공분산 구조가 **반응 신호 자체**(특히 covariance
   task 0.528로 붕괴)이기 때문. "지울 패턴"과 "반응인 패턴"이 겹친다. **이 방향은 실측으로 닫힘.**
3. **흥미로운 부수 발견**: `centered` mean+var(0.685) ≈ v24 slot ridge 천장(0.70). 즉 **0.70 천장의
   실체는 "centered bag의 평균+분산"이지 12-slot 구조가 아님**. 그리고 이 수치(0.685)는 v24 모델
   (0.708)보다 **낮음** — 모델은 per-cell L2에도 불구하고 global_summary(spread)·covariance sketch
   경로로 magnitude를 이미 공급받아 그 이상을 뽑는다.
4. **결론**: 정규화는 0.1 로스 레버가 아님. 최선 정규화(centered)도 기존 0.70 천장을 넘지 못하고,
   모델은 이미 그 위(0.708). 유일하게 실행 가능한 nugget = per-cell L2 제거(centered 유지) 학습
   실험이나, 모델이 다른 경로로 magnitude를 받으므로 기대 이득은 작고 불확실(사용자 판단).

**산출물/명령**:
- 스크립트: `scripts/diagnose_normalization_ceiling.py`
- 결과: `logs/normalization_ceiling_20260803.csv`
- 재실행: `python scripts/diagnose_normalization_ceiling.py --episodes 1000 --bootstrap 300`
  (bootstrap은 그룹 Python 루프라 2000이면 ~7분, 300 권장 — §19 참고)

---

## 20. 2026-08-03 — v24 no-L2 ablation: per-cell L2 정규화 제거 학습 (진행 중)

**배경**: §19 정규화 천장 프로브가 per-cell L2가 충분통계 천장을 −0.048(0.685→0.636)
떨어뜨린다고 측정. 이를 모델 수준에서 검증하는 ablation 학습 착수 (사용자 지시 "L2 제거 진행").

**변경**: `src/models/baseline.py` — `_bag_view`에 `bag_centered_l2_normalize` 플래그 추가
(기본 `true` = 기존 동작 불변). `false`면 `classification_instances = centered_delta`
(편차 크기 보존, L2 미적용). `StructuredEpisodePopulationAggregator`·`BaseModel`에 파라미터
전달. 하위 slot 배정(내부 `F.normalize`)·token encoder(LayerNorm)가 재정규화하므로 안전.
새 unit test `test_bag_centered_l2_normalize_flag_controls_magnitude` 추가. arch version은
v24 유지 (ablation이므로).

**config**: `configs/train_v24_medium_no_l2.yaml` (base: v24 확정, `bag_centered_l2_normalize: false`,
experiment `v24_medium_no_l2`, 50 epoch).

**검증**: ① config build 확인 (flag 반영, arch 24, per-cell norm 0.73~5.5로 다양), ②
`tests.test_base_model` 88개 통과 (신규 테스트 포함), ③ 1-epoch smoke (실제 Lightning 루프,
limit 64/16) — NaN/크래시 없음, train_loss 0.715 / val_loss 0.692. ④ **전체 unittest 수트 완료:
154개 중 153개 통과, 1개 에러** — 에러는 `tests/test_learnability_ladder.py`가 archive된
v18/v19 config들의 `trainer: learnability_d20` 참조에서 발생(해당 config는 `a5dfcf8` 정리에서
삭제된 **기존 결함**, §16에 "이번 변경과 무관, 조치하지 않음"으로 이미 기록됨). 본 변경은
통과 테스트 +1만 추가, 회귀 없음.
> **[2026-08-04 해소]** 이 상시 실패는 삭제된 `configs/trainer/{learnability_d20,csp_short8}.yaml`을
> git(`a5dfcf8^`)에서 원본 복원하여 해결됐습니다. 현재 **unittest 154/154 통과**이며, `base_config`를
> 가진 config 65개 전부 로드됩니다. 이후 문서에 나오는 "153/154 통과 + 기존 결함 1건" 표기는
> 2026-08-04 이전 시점의 기록입니다. 상세: [`../current_status.md`](../current_status.md) §26. (스모크 중 CUDACachingAllocator OOM 경고 1회 — 배치 크기
그대로라 일시적·무해, 학습 지속 확인.)

**학습 실행 중**: Run `v24_medium_no_l2`, scratch Medium 50 epoch (v24와 동일 방식).
- PID `615588` (launcher). 시작 2026-08-03 02:11 KST.
- 로그: `logs/20260803_021140/v24_medium_no_l2.out`
- Launcher 로그: `logs/20260803_021140/v24_medium_no_l2_launcher.out`
- 체크포인트: `checkpoints/20260803_021140/v24_medium_no_l2/`
- 초기 확인: epoch 0, ~4.25 it/s 정상 (v24/v26과 동일). 완료까지 ~1.8~2시간 예상.

**다음 Action (학습 완료 후)**: best checkpoint로 v24 확정(`0.5903`)과 비교 —
1) val_ce_loss가 0.5903보다 낮은지, 2) 승격 시 1,000-episode paired 평가(사용자 판단,
§16 기준 +0.03/+0.05). §19의 기대는 "작고 불확실" — 모델이 global_summary/covariance로
magnitude를 이미 받으므로, val_ce 개선이 없으면 이 방향도 종료.

**최종 결과 (2026-08-03, 학습 완료)**: 50-epoch 완주, 프로세스 종료, GPU 해제. **best
`val_ce_loss 0.5925`@epoch 32** (top-k: 0.5925/0.5926/0.5930, 이후 개선 없음) —
**v24 확정(`0.5903`)보다 +0.0022 나쁨.**
> [!IMPORTANT]
> **음성 결과 — 이 방향 종료.** §19의 예측대로, per-cell L2가 충분통계 수준의 ridge 천장에선
> −0.048이었지만 전체 모델은 global_summary(spread)·covariance 경로로 이미 magnitude를
> 공급받아, L2 제거가 개선을 주지 못했다. val_ce가 v24보다 나쁘므로 1,000-episode 평가는
> 불필요(승격 후보 아님). `bag_centered_l2_normalize` 플래그는 기본 `true`(기존 동작)라 v24에
> 영향 없음 — 코드는 남기되 비활성.
> 재확인: 1-epoch 스모크 이후 전체 unittest 154/153 통과 + 기존 learnability_d20 결함 1건(§20).
> 체크포인트: `checkpoints/20260803_021140/v24_medium_no_l2/epoch=032-val_ce_loss=0.5925.ckpt`

---

## 21. 2026-08-03 — Zero-shot Musk (Musk2) MIL 벤치마크 테스트

**배경**: 사용자 요청 — `/NHNHOME/kimds/Data/Musk/musk.pkl`로 **Musk MIL 벤치마크** 테스트 파일 작성.
Musk2: 102개 분자(bag), 각 bag은 컨포머 인스턴스(166 화학 descriptor, 1~1044개), bag 라벨은
"아무 컨포머라도 musk면 양성". 전형적 multiple-instance 문제로 BagPFN(bag 단위 in-context
meta-classifier)의 자연스러운 테스트 대상.

**방법**: `scripts/test_musk.py` — **leave-one-out 에피소드 스윕**: 각 bag을 query로, 나머지 101개를
labeled context로 모델 forward(`model.model(x, y, mask_index)`, 가변 길이 bag 리스트 지원). **166→512
zero-padding**(모델 input_dim=512; OOD 브리지로 명시, 화학 descriptor에 대한 학습 시맨틱 없음 —
분포 이동 baseline). v24 확정 checkpoint(`epoch=041, val_ce 0.5903`) 사용, fp32 추론. 예측:
`predictions/musk_v24_zero_shot.pt`.

**결과 (zero-shot, v24 확정 체크포인트, 102 bag)**:

| 지표 | 값 |
|---|---|
| AUROC | **0.7766** [0.667, 0.878] |
| Accuracy | 0.7157 |
| Balanced accuracy | 0.6477 (sens 0.359 / spec 0.937) |
| Log loss | 0.5833 |
| predicted positive | 18/102 |

**해석**: 합성 학습 모델이 OOD(166차원 chemical descriptor zero-pad)임에도 **zero-shot AUROC 0.78**로
무작위(0.5)를 크게 상회 — **in-context 메타러닝이 classic MIL 벤치마크로 전이됨**을 보여주는 유망 신호.
단 보수적으로 행동(sens 0.359 / spec 0.937, 양성 18/102만 예측)하고, n=102 소규모로 CI가 넓음
([0.667, 0.878]). 이는 탐색적 결과이므로, 정식 벤치마크 비교를 위해서는 ① 166→512 학습 projection,
② MIL baseline(mi-SVM 등)과의 대조, ③ 적절한 CV 프로토콜이 필요.

**산출물/명령**:
- 스크립트: `scripts/test_musk.py`
- 예측: `predictions/musk_v24_zero_shot.pt`
- 재실행: `python scripts/test_musk.py --data /NHNHOME/kimds/Data/Musk/musk.pkl --config configs/train_v24_medium_bag_proj_residual.yaml --checkpoint checkpoints/20260731_220100/v24_medium_bag_proj_residual/epoch=041-val_ce_loss=0.5903.ckpt`

**다음 Action (사용자 판단)**: ① Musk 정식 검증(projection/MIL baseline) 진행 여부, ② no-L2 학습
완료 후 v24와 비교(§20 계속), ③ 경로 A/ICI 관점.

---

## 22. 2026-08-03 — 전략 전환: 생성기 개선 (Musk-like easy 데이터) — 가설 판정 완료

**배경/판정**: 사용자 전략 전환 — "현재 합성 생성기가 너무 lossy해서 세포 신원이 관측 불가능하고
(oracle 0.93, per-cell 선형 0.70), 그래서 모델이 데이터 정보 한계(~0.70)에 갇혀 있다. 모델 문제가
아니라 **데이터 생성 문제**다. 목표를 **적당한 난이도에서 bag을 완벽히 구분**(= Musk 수준)으로 잡고
그에 맞는 데이터를 생성하자." 이 전환은 §19/§20의 음성 결과(정규화·선택 변경 모두 0.70 천장)와
일치하며, E7-NL 대신 **생성기 강화**로 "데이터가 분리 가능하면 모델이 완벽히 분류하는가"를 직접 검증.

**변경**: `configs/train_v24_musklike_easy.yaml` (base: v24 확정, arch는 v24 유지) — 생성기 응답을
지배적으로 강화: `rare_response_probability 0`(전 응답이 명확한 분율), `response_mixture_effect_scale
2.5`, `response_state_effect_scale [1.5,3.0]`, `response_covariance_effect_scale [1.0,2.0]`,
`response_score_scale 1.5`, `response_score_min_margin 0.15`, `class_separation [1.0,2.0]`,
`observation_noise 0.005`.

**분리 가능성 검증 (재학습 없음, `diagnose_normalization_ceiling.py` 500 eps)**:
| 정규화 | 원래 Medium ridge | Musk-like Easy ridge |
|---|---:|---:|
| centered mean+var | 0.685 | **0.927** [0.918,0.936] |
| current(=모델 입력) mean+var | 0.636 | **0.894** [0.884,0.903] |

per-task(`current` mean_var): composition 0.958 / state 0.883 / covariance 0.751 / interaction 0.886 /
combined 0.959 → **Musk 수준(~0.9+) 분리 가능성 달성**. 데이터가 bag 레벨에서 명확히 분리됨.

**스모크 (1-epoch, 실 Lightning 루프)**: NaN/크래시 없음 — **val_loss 0.485 vs 원래 Medium 0.692**
(epoch 1부터 큰 차이 — 데이터가 분리 가능하면 모델 손실이 크게 낮아지는 초기 신호). (CUDACachingAllocator
OOM 경고 1회는 이전과 동일한 일시적 현상, 회복됨.)

**학습 완료 (2026-08-03 06:00 KST, 04:28 시작)**: Run `v24_musklike_easy`, scratch 50 epoch 정상 완주
(`Trainer.fit stopped: max_epochs=50 reached`, launcher "completed successfully"). 프로세스 종료, GPU 해제.
- 로그: `logs/20260803_042852/v24_musklike_easy.out`, 체크포인트: `checkpoints/20260803_042852/v24_musklike_easy/`
- 이상 없음: NaN/크래시 없음, 일시적 CUDACachingAllocator OOM 경고 2회만(회복됨, 기존과 동일).

**최종 결과 (2026-08-03, 1,000-episode 정식 평가)**: best `val_ce_loss 0.2552`@epoch 40 —
**v24 Medium(`0.5903`)의 절반 이하**. 예측: `predictions/synthetic_v24_musklike_easy_1000ep.pt`.

| 지표 | Musk-like easy (v24 arch) | v24 Medium (참고) |
|---|---:|---:|
| Best val_ce_loss | **0.2552** @ epoch 40 | 0.5903 @ epoch 41 |
| 1,000-ep AUROC | **0.9510** [0.946, 0.956] | ~0.70 천장 (v22~v26 전부) |
| Log loss | 0.2836 | — |
| task: composition / state / covariance / interaction / combined | 0.978 / 0.952 / 0.865 / 0.942 / 0.985 | — |

> [!IMPORTANT]
> **가설 확정: "0.70 한계 = 데이터 lossiness(생성기 문제)"**. 모델이 분리 가능한 데이터에서
> 근완벽 분류(AUROC 0.951) — 심지어 모델 입력 ridge 천장(current mean_var 0.894, centered
> 0.927)보다 높아, 모델이 선형 bag 통계 이상의 신호(covariance 등)도 활용함을 시사. 반대로
> Medium 데이터의 0.70 천장은 아키텍처 문제가 아니라 **생성기가 세포 신원을 잃게 만드는
> lossiness** 때문이라는 §19/§20 음성 결과와도 일치. v24는 여전히 확정 baseline(이 실험은
> 아키텍처가 아니라 데이터 가설 검증용).

**실제 Musk (Musk2) zero-shot 테스트 (2026-08-03, 후속)**: 위 best 체크포인트(`epoch=040`)로
`scripts/test_musk.py` 실행 — 102 bag leave-one-out. 예측: `predictions/musk_v24_musklike_easy.pt`.

| 지표 | musk-like easy (신규) | v24 Medium (§21 재사용) |
|---|---:|---:|
| AUROC | **0.8030** [0.705, 0.889] | 0.7766 [0.667, 0.878] |
| Balanced acc | **0.7747** (sens 0.692 / spec 0.857) | 0.6477 (sens 0.359 / spec 0.937) |
| Log loss | **0.5439** | 0.5833 |
| predicted positive | 36/102 | 18/102 |

**해석**: AUROC +0.026이지만 n=102 소규모로 CI 겹침 — paired bootstrap 승률 0.24로 **통계적으로
구분 불가**(사실상 동률). 단 운영점이 크게 개선됨: sensitivity 0.36→0.69, balanced acc 0.65→0.77,
log loss 0.58→0.54 — **분리 가능한 데이터로 학습한 모델이 실제 Musk에서도 덜 보수적이고 더 잘
보정됨**(양성 예측 18→36). §22 "데이터가 분리 가능하면 모델이 신호를 잘 쓴다"는 방향이 실제
데이터 전이에서도 방향성 있게 지지되는 탐색적 결과.

**실제 Musk — 입력 표현 병목 진단 (2026-08-03, 0.95 목표 대비)**: 사용자 질문 "padding이 평균
계산에 들어가는가?"에 대한 코드 검증 + 천장 프로브.

- **답: padding은 평균 계산에 안 들어감.** `_bag_view`의 `bag.mean(dim=-2)`는 feature별 cell 평균이라
  zero-padding 열(항상 0)은 평균 0 → 실제 166개 열 평균에 무영향. L2 norm(0²=0)도 동일.
  따라서 "centering 후 padding" = "padding 후 centering" 수학적으로 동일(모델이 어차피 재-centering).
- **진짜 병목: `_bag_view`가 bag 평균을 폐기** (summary = `global_spread`, bag_mean 미사용;
  `projection_residual_mean`도 raw 평균이 아니라 구조화 토큰 평균). Musk에서 bag 평균(분자별
  descriptor 프로필)이 최강 신호인데 제거됨.
- **천장 프로브 (LOO ridge AUROC)**: raw mean+var **0.829** / centering만 **0.567** (↓0.26) /
  L2만 0.742 / center+L2(=모델 입력) **0.554** / 인스턴스 약지도 → bag max·mean·softmax 풀링 **~0.90**.
  → "centering이 신호를 죽인다"가 확정, padding이 아님.
- **zero-shot 수정 (학습 파라미터 없음, config 오버라이드)**: `test_musk.py --preprocess raw`
  (`bag_centered_representation=False`, `global_summary=raw_mean`, `use_raw_mean_branch=True`).
  실측: bag_view **0.8030** → raw **0.8217** [0.733, 0.904], log loss 0.544→**0.511** (centered_no_l2는
  0.7912). **bag 평균 보존이 실제 모델에서도 방향성 개선** — 사용자 직관 검증됨.
- 커밋 `a9bb7b8` (test_musk.py `--preprocess` 추가).

**다음 Action (0.95 목표)**: ① (완료) `--preprocess raw`로 bag 평균 보존 → 0.822, ② 다음 지렛대:
MIL max/softmax 풀링(천장 ~0.90, 인스턴스 스코어링 + 풀링), ③ 비선형 인스턴스 인코더/Musk LOO
fine-tuning으로 0.83~0.90 천장 돌파, ④ 5-seed 앙상블로 n=102 분산 감소. (Musk2 문헌 accuracy
~0.85~0.90 — AUROC 0.95는 상위권 목표, 단계적 접근 필요.)

**다음 Action**: ① 세포 신원 활용 실험(메커니즘 A/B — 선택 학습)을 이 분리 가능한 데이터 위에서
재검증, ② 생성기 코드 수준 개선(크기 채널 보존 등)으로 Medium에서도 분리 가능하게 만들기,
③ 필요 시 v24_Musk-easy를 추후 아키텍처 비교용 상한 데이터로 활용, ④ (사용자 판단) Musk 정식
검증(166→512 학습 projection, MIL baseline 대조) 진행 여부.

---

## 23. 2026-08-03 — Musk 0.95 로드맵: raw bag-stat token (mean/skew/kurt) 학습 중

**배경**: Musk 0.95 목표의 두 번째 지렛대. 첫 지렛대(`--preprocess raw`, bag 평균 보존)로
0.803→0.822 확인했지만 가중치가 학습 때 mean을 못 봤던 한계. 사용자 제안: centered+L2 입력은
유지하고, **raw에서만 얻는 고차 통계량을 별도 token으로 추가**해 학습에서부터 mean/shape 신호를
쓰게 한다.

**중복 분석**: per-feature **분산은 `global_spread` summary(= std)와 중복** → 제외. **왜도(3차)/
첨도(4차)는 새 신호** (모델은 1/2차 적률 + tail/rare 순서통계량만 보유) → 추가. 왜도/첨도는
표준화 적률 비율이라 **scale-free → raw 전달**(정규화 불필요), mean은 **L2 정규화**(scale 의존이라
합성↔Musk 일관성).

**구현**: `include_bag_mean_token`(bool) → **`raw_stat_tokens: Sequence[str]`** (mean|variance|
skewness|kurtosis)로 일반화. `_raw_stat_tokens()`가 centered 변환 전 raw cell에서 per-feature
통계 계산. 각 통계 = 512-d token 1개, `structured_tokens_per_bag` = 40 + len. batched/list 양쪽
경로, `_validate_representation`, `forward_episode_batch` 모두 연결. 커밋 `7830b11`. 검증: forward
smoke finite(tokens 43), 적률 값 정확(exp 왜도~1.8/첨도~7.7, 가우시안 ~0/~3), unittest 153/154
통과(기존 learnability_d20 결함 1건 무관).

**config**:
- `train_v24_musklike_easy_rawstats.yaml` — `raw_stat_tokens: [mean, skewness, kurtosis]` (tokens 43)
- `train_v24_musklike_easy_mean_token.yaml` — `[mean]` 전용 (tokens 41)
- (기각된 raw 직접 전달 `rawmean` config는 삭제)

**학습 완료 (2026-08-03 12:38 KST, 11:06 시작)**: Run `v24_musklike_easy_rawstats`, scratch 50 epoch
정상 완주, launcher "completed successfully". best `val_ce_loss 0.2468`@epoch 48 — centered 모델
(`0.2552`)보다 소폭 개선. 로그 `logs/20260803_110607/`, 체크포인트
`checkpoints/20260803_110607/v24_musklike_easy_rawstats/epoch=048-val_ce_loss=0.2468.ckpt`.
> **버그 수정**: 가변 길이 bag(list 경로, 실제 Musk)에서 `torch.stack(raw_bags)` 실패 → per-bag
> 통계 후 stack으로 수정, 커밋 `2008278`. unittest 153/154 통과.

**최종 결과 (2026-08-03, 1,000-ep 합성 + Musk)**: 예측 `predictions/synthetic_v24_musklike_easy_rawstats_1000ep.pt`,
`predictions/musk_v24_musklike_easy_rawstats.pt`.

| 모델 | 합성 val AUROC | 실제 Musk AUROC |
|---|---:|---:|
| centered (musklike_easy) | 0.9510 [0.946,0.956] | 0.8030 [0.705,0.889] / 0.8217 (raw) |
| **rawstats (mean+skew+kurt)** | **0.9522** [0.948,0.957] | **0.7835** [0.681,0.876] |

> [!IMPORTANT]
> **음성 결과 — raw bag-stat token 지렛대 종료.** 합성은 동률(0.9522 vs 0.9510, log loss 0.278 vs
> 0.284로 소폭 개선)이지만, **실제 Musk는 0.7835로 centered(0.803/0.822)보다 낮음.** 해석: 모델이
> stat token을 합성(단위 cell) 분포에 맞춰 학습해서, Musk descriptor 통계(다른 scale/분포)에
> 전이되지 않음. mean token(L2 정규화)은 분포가 일치하지만 학습된 의미(합성 donor centroid)가
> Musk에 유용하지 않게 작동. → **학습에서 raw 통계를 추가하는 방향은 Musk 0.95에 도움이 안 됨.**
> Musk 기준은 centered musklike-easy(0.803, `--preprocess raw`면 0.822)로 유지.

**다음 Action**: ① (종료) raw-stat token 지렛대 — 음성, ② (진행) **Phase 1 IA-MIL** — 얕은 풀링 대신
**비선형·작업적응적 인스턴스 어텐션**으로 모델이 "어떤 세포/컨포머가 라벨을 결정하는가"를 학습
(§24), ③ Musk 166→512 읽기 브리지(Phase 2) 후 실제 Musk 검증.

---

## 24. 2026-08-03 — Phase 1 IA-MIL (Instance-Attention MIL) — 판정: 음성

**배경**: 사용자가 얕은 MIL 풀링 대신 "조금 더 복잡하고(비선형) 확실한" 방법 요구 — "지금 musk도
못하는데 ICI를 어떻게 하겠어". 진단: 기존 per-instance 경로가 **얕고 선형**(LayerNorm + Linear 1개 +
cosine + 고정 top-k, 보조 채널). 모델이 "어떤 인스턴스가 결정적인가"를 배우는 메커니즘 부재 =
Musk(any-positive)·ICI(희귀 반응 세포 아형) 공통 병목.

**구현** (`use_instance_attention_mil`, 기본 OFF — 완전 호환, 커밋 `e0620ac`):
- `mil_instance_encoder`: LayerNorm + 2층 MLP → 비선형 인스턴스 임베딩
- `mil_relevance_mlp`: MLP(h_i, 클래스 메모리 맥락) → **작업적응적** 관련도 (cosine 아님)
- 어텐션 soft-pool(Σa_ic·h_i) + **max-attention 인스턴스**(any-positive 편향) → `mil_score_head`
- **잔차 채널**로 추가(covariance/typed_bag와 동일 패턴, fusion_scorer 불변)
- batched(4D 학습) + list(가변 길이, Musk/ICI) 양쪽 경로
- 검증: batched/list/variable-length forward+backward finite, MIL 그라디언트 정상, 기본 config
  불변, unittest 153/154(기존 learnability_d20 결함 1건 무관)

**학습 3종 (병렬 2회 OOM → 순차 큐 버그 수정)**:
| Run | 데이터 | config | 상태 |
|---|---|---|---|
| `v24_musklike_easy_mil` (주) | musklike_easy + IA-MIL | `train_v24_musklike_easy_mil.yaml` | ✅ **50 epoch 완료** (best val_ce **0.2462**@39) |
| `v24_musklike_easy_rare_baseline` (판별) | rare-response(5~15% cell 반응) + no-MIL | `train_v24_musklike_easy_rare_baseline.yaml` | ✅ **50 epoch 완료** (best val_ce 0.2639@48) |
| `v24_musklike_easy_rare_mil` (판별) | rare-response + IA-MIL | `train_v24_musklike_easy_rare_mil.yaml` | ✅ **50 epoch 완료** (21:25 단독 재실행, best val_ce 0.2616@49) |

- **OOM 실패 ① (17:32)**: 3개 병렬 실행 → mil(91GB) + rare(77GB) 겹침 → rare 2종 exit 1.
- **OOM 실패 ② (19:10)**: 순차 큐의 `wait_gpu_free`가 **레이스** — `launch_interactive_training.sh`가
  detached worker를 백그라운드로 띄우고 즉시 반환 → 실행 직후 pgrep이 train.py를 아직 못 잡고 "GPU free"
  오판 → rare_baseline과 rare_mil을 동시에 실행 → rare_mil 2차 OOM (107GB + 53GB → 178GB 초과).
- **큐 버그 수정 (2026-08-03, 커밋 아래)**: `scripts/queue_phase1_rare.sh`에 `wait_launched_training_done`
  추가 — 각 run을 실행한 뒤 **train.py가 실제로 뜰 때까지 폴링(grace 180s) 후 그 학습이 끝날 때까지 블록**.
  기능 검증: mock 학습(2s 스폰 + 6s 실행)에 대해 8s 블록 확인, timeout 경로도 정상. 부가로 run 선택 인자
  지원: `./queue_phase1_rare.sh <run...>` 로 원하는 run만 순차 실행.
- **순차 큐 재실행 (21:25)**: 큐 버그 수정 후 `./queue_phase1_rare.sh v24_musklike_easy_rare_mil`로
  rare_mil 단독 재실행 → 23:00 **50 epoch 정상 완료** (수정된 큐가 스폰 대기→완료 블록 수행 확인).

**평가 결과 (2026-08-03, 1,000-episode, best ckpt)**:
| Run | 합성 AUROC [95% CI] | Log loss | Musk zero-shot AUROC |
|---|---|---|---|
| `mil` (주, musklike_easy) | **0.9520** [0.947, 0.957] | 0.2782 | **0.5545** [0.440, 0.672] ⚠️ |
| `rare_baseline` (판별) | **0.9492** [0.945, 0.954] | 0.2891 | — |
| `rare_mil` (판별) | **0.9224** [0.917, 0.928] | 0.3549 | — |

- 예측 파일: `predictions/synthetic_{run}_1000ep.pt`, `predictions/musk_v24_musklike_easy_mil.pt`

**판정 — Phase 1 IA-MIL 음성**:
| 성공 기준 | 결과 | 판정 |
|---|---|---|
| ① 주: 합성 ≥0.94 무회귀 | 0.9520 (§22 easy 0.9510 / rawstats 0.9522와 동률) | ✅ |
| ② 판별: IA-MIL > baseline 유의 | baseline 0.9492 vs mil 0.9224, paired P=1.00 | ❌ **IA-MIL 유의 열위** |
| ③ (예비) Musk ~0.80 유지 | 0.5545 (vs §22 easy 0.8030) | ❌ **큰 회귀** |

> [!IMPORTANT]
> **`use_instance_attention_mil` 실효성 없음 → 기본 OFF 유지, v24 확정 baseline 유지.**
> 합성 musklike_easy에선 무회귀(0.9520)지만, cell-identity 판별(rare)에서 유의 열위(P=1.00)이고
> 실 Musk zero-shot에서 0.8030→0.5545로 급락. 학습 손실(best 0.2462)과 달리 평가·전이에서 IA-MIL
> 잔차 채널이 해가 됨. Phase 2(166→512 읽기 브리지)는 **IA-MIL 채널 없이**만 고려(사용자 재확인).

**다음 Action (제안)**: ① IA-MIL 채널 진단(과적합/스케일) 또는 폐기 결정, ② Musk 0.95 로드맵 복귀
(지렛대 1 centered 표현 유지 — 현재 최고 0.822), ③ Phase 2 읽기 브리지는 IA-MIL 제외로 사용자 확인 후 진행.

**판별 실험 논리**: musklike_easy는 전 cell 반응(separable, 0.951 포화)이라 IA-MIL 이득이 안 보임.
rare-response(일부 cell만 반응)에서 "어느 cell이 반응하는가"가 중요 → IA-MIL이 baseline을 이기면
**cell-identity 메커니즘 검증** (ICI 희귀 반응 세포와 동형). → 결과적으로 IA-MIL이 유의 열위(P=1.00).

---


> **[아카이브 2026-08-04]** `current_status.md`에서 이관. 이 절의 T3-3 / 후보 동결 / T4 context-size curve는 모두 완료됐고, Action Plan 자체는 §21~§26(Musk 실데이터 전이)으로 대체됐습니다. 다만 **T4 Medium→Hard attribution의 잔여 항목**(raw-cell 대 40-token audit, token budget sweep, training scaling)은 수행되지 않은 **미결 질문**으로 남아 있습니다 — `current_status.md` §6 스텁 참고.

## 6. 다음 작업 세션 Action Plan — 구조적 변경 및 실험 목록

> [!IMPORTANT]
> 모든 판단은 **합성 val**에서 하고 ICI는 손대지 않습니다 (§5). 후보마다
> `scripts/evaluate_synthetic.py` → `scripts/compare_predictions.py`로 기준선
> (`predictions/synthetic_v22_baseline_1000ep.pt`, AUROC 0.7078 [0.696, 0.719])과
> paired cluster bootstrap 비교할 것.

### ✅ T3-3 v22 Hard 기준선 완료 (2026-07-31)

- Run: `v22_hard_baseline`, 50 epoch / 25,600 optimizer steps, 정상 종료
- Best checkpoint: `checkpoints/20260731_035538/v22_hard_baseline/epoch=044-val_ce_loss=0.6839.ckpt`
- Best `val_ce_loss`: **0.6839** (epoch 44), v21 Phase 2 참고값 0.6845와 사실상 동일
- 1,000-episode 평가: **AUROC 0.5483 [0.538, 0.558]**, Log Loss 0.6920, 15,373 query
- 예측: `predictions/synthetic_v22_hard_baseline_1000ep.pt`
- 로그: `logs/20260731_035538/v22_hard_baseline.out`

| task | AUROC | episodes |
|---|---:|---:|
| combined | 0.6095 | 213 |
| composition | 0.5614 | 204 |
| interaction | 0.5298 | 200 |
| state | 0.5167 | 177 |
| covariance | 0.5103 | 206 |

> [!IMPORTANT]
> Hard regime에서는 전체도 0.55에 불과하고 state/covariance는 거의 무작위입니다. Medium에서 종료한 Tier 1/2 방향을 다시 열 관측 근거가 생긴 것은 아닙니다. v21의 `val_ce_loss`를 재현했으므로 v22 구현 이상보다는 문제 난이도 자체의 한계로 읽는 것이 타당합니다.

### ✅ 최종 후보 동결 여부 결정 — **완료 (2026-08-01)**

사전에 계획한 Tier 1~3 합성 작업이 모두 끝났고, 이어서 진행한 v23/v24 bag-collapse family도 4종 모두 학습을 완주했습니다. 원래 계획은 이 4종을 1,000-episode paired 비교로 검증한 뒤 동결하는 것이었으나, **사용자가 이 비교를 건너뛰고 v24-B1(residual + bottleneck)을 직접 v24로 확정했습니다** (§3 "최종 결정" 참고). v22 Medium baseline은 폐기되었고 최종 후보는 v24입니다. **ICI는 여전히 잠금 상태** — v24용 ICI config가 없고, 이번 결정이 §7 평가 프로토콜을 통과한 것이 아니므로 ICI 해제는 별도로 다시 확인이 필요합니다.

### ➡ 다음 우선순위: T4 Medium→Hard 성능 붕괴 attribution

Hard는 Medium과 비교해 9개 축이 동시에 바뀝니다: class separation `0.5~1.4→0.2~0.8`, rare fraction `2~8%→0.5~3%`, rare probability `0.15→0.25`, donor shift `0.35→0.70`, component shift `0.12→0.25`, noise `0.01→0.05`, latent dim `32→64`, cells `500~1000→500~1500`, 그리고 batch/accumulation·covariance rank가 달라집니다. 현재 결과만으로는 어느 변화가 AUROC `0.7078→0.5483` 붕괴를 만들었는지 알 수 없습니다.

**세 가지 표본·압축 병목 가설 (2026-07-31)**

1. **Bag 내부 압축 병목 — 가능성 높음.** “40 instances”가 아니라 1 global + 12 slots×(center/spread/rare) + 3 tails의 **40개 512-d token**이며, 별도 slot metadata/covariance 경로도 있으므로 단순 1000→40 비율만으로 과압축이라 단정할 수는 없습니다. 하지만 Hard 반응세포는 0.5~3%뿐이고, 기존 sparse/slot 선택이 무작위·fragmented였으므로 label-relevant 소집단이 요약 과정에서 사라질 가능성은 높습니다. 판별은 같은 context에서 raw-cell observable distribution descriptor와 40-token descriptor를 직접 비교하고, 이후 12/24/48 slot·tail budget scaling으로 합니다.
2. **Episode 내부 context bag 부족 — 가능성 높음.** 50~100 bags에서 최대 20%를 query로 빼면 context는 대략 40~80 bags, 클래스당 약 20~40개입니다. 매 episode마다 manifold와 response direction이 새로 뽑히므로 이 수십 개 label로 512-d 이상의 episode-specific 관계를 다시 추정해야 합니다. 또한 context token은 class당 8 memory token으로 다시 압축되지만 global/ridge 분기는 원 bag 통계를 직접 쓰므로 두 경로를 분리해 봐야 합니다. 판별은 같은 query를 고정한 paired context-size curve(10/20/40/80/160)로 합니다.
3. **전체 training episode/step 부족 — 가능성 중간 이하, 아직 배제 불가.** Training은 seed 고정 dataset 재사용이 아니라 온라인 non-repeating stream입니다. Medium은 약 81,920 episodes/10,240 optimizer steps, Hard는 약 204,800 episodes/25,600 steps로 episode 절대량은 이미 큽니다. Hard CE도 epoch 30의 0.6845에서 epoch 44의 0.6839로만 개선되어 거의 plateau입니다. 따라서 우선순위는 낮지만, checkpoint별 1,000-episode curve와 1×/2×/4× update scaling으로 최종 확인합니다. Episode 수와 optimizer update 수는 별도로 통제합니다.

**판별 순서**: context-size curve(가장 저렴) → raw-cell 대 40-token information audit → token budget sweep → 마지막으로 training scaling. Context를 늘려도 평평하면 bag 표현 병목, raw-cell descriptor만 높으면 압축 병목, 둘 다 충분한데 checkpoint 성능만 낮으면 meta-training/optimizer 병목으로 판정합니다.

**Context-size curve 완료 기록 (2026-07-31 10:21~10:42 KST)**
- 고정: Hard best checkpoint, 동일 episode, 클래스별 query 10개(총 20), nested balanced context
- Context: 총 10/20/40/80/160 bags. 10~80은 학습 범위, 160은 의도적 OOD 상한
- 완료: 1,000 episode 중 991 사용, 9 skip; context 크기당 query 19,820개

| 총 context bags | AUROC (episode-cluster 95% CI) | Log loss |
|---:|---:|---:|
| 10 | 0.5084 [0.500, 0.516] | 0.7170 |
| 20 | 0.5193 [0.511, 0.528] | 0.7047 |
| 40 | 0.5312 [0.523, 0.539] | 0.6967 |
| 80 | 0.5505 [0.542, 0.559] | 0.6900 |
| 160 | 0.5737 [0.565, 0.583] | 0.6835 |

- **판정**: context 10→80에서 `+0.0421`, 80→160에서 추가 `+0.0232`, 전체 10→160에서 `+0.0653`으로 단조 증가한다. 따라서 episode 내부 labeled bag 부족은 실제 주요 병목이다.
- 200회 paired episode bootstrap 교차검증에서 `P(40 > 80)=0.00`, `P(80 > 160)=0.00`이었다. 즉 두 핵심 증가는 동일 episode/query 기준으로도 방향이 일관됐다.
- 80의 0.5505가 기존 Hard 기준선 0.5483과 정합하므로 통상 Hard 평가의 context 규모를 재현한다.
- 160은 학습 분포 밖 상한인데도 개선이 계속되지만 AUROC는 0.5737에 그친다. 따라서 context 부족만으로 Hard 붕괴 전체를 설명할 수 없고, bag 내부 정보 손실/압축 병목을 다음으로 분리한다.
- 실행 PID(종료): `2499444`; 로그: `logs/20260731_context_curve/context_curve.out`
- Summary: `logs/v22_hard_context_curve_1000ep.csv`; predictions: `predictions/v22_hard_context_curve/context_{10,20,40,80,160}.pt`
- Smoke: 2 episodes 전체 경로 성공; unit tests 3개 통과 (balanced/disjoint/nested 계약); 전체 unittest 114개 통과 (675.411초)

**Large-context 학습 용량 점검 (B200 183,359 MiB)**
- 현재 Hard 학습: `episode_batch_size=4`, `accumulate_grad_batches=2`, GPU 1장, BF16 mixed. 따라서 forward/backward당 4 episodes, optimizer update당 effective 8 episodes.
- Episode는 50~100 bags, training query는 5~12 bags이므로 실제 학습 context 범위는 38~95 bags.
- 현재 v22 6.57M 모델, batch 4, 1,500 cells/bag의 보수적 FP32 forward/backward 벤치:

| 총 bags/episode | Peak allocated VRAM | Step time |
|---:|---:|---:|
| 100 | 47,043.9 MiB | 0.135 s |
| 172 | 80,849.2 MiB | 0.213 s |
| 220 | 103,384.9 MiB | 0.273 s |

- 벤치는 bags 절반을 query로 둬 실제 training query 5~12보다 meta 경로가 더 무거운 보수적 조건이다. 따라서 `context 160 + query 최대 12 = 총 172 bags`는 현재 batch 4를 유지해도 충분한 여유가 있다.
- 총 220 bags도 단일 model step은 통과했으므로 context 약 200까지는 유력하지만, CUDA online generation/prefetch를 포함한 end-to-end smoke 전에는 정식 상한으로 확정하지 않는다.
- 실제 ICI context는 fold당 약 69명이므로 large-context-only 학습은 사용하지 않습니다. 작은 context를 포함한 mixed 학습을 하고 표준 40/80 평가 성능이 개선될 때만 학습 효과로 인정합니다.

**Batch 2 / context 300 추가 점검**

| Episode batch | 총 bags/episode | Peak allocated VRAM | Step time |
|---:|---:|---:|---:|
| 2 | 312 | 73,338.0 MiB | 0.196 s |
| 2 | 360 | 84,605.2 MiB | 0.223 s |

- `context 300 + query 최대 12 = 총 312 bags`는 B200에서 큰 메모리 여유로 통과했다. 이 벤치 역시 FP32이며 bags 절반을 query로 둔 보수적 model-step 조건이다.
- `episode_batch_size=2`로 내리면 `accumulate_grad_batches=4`로 올려 기존 effective batch 8과 epoch당 optimizer step 수를 유지한다. accumulation 2를 그대로 두면 effective batch가 4로 바뀌고 optimizer step이 2배가 되어 context 효과와 최적화 효과가 섞인다.
- 실제 ICI context가 약 69이므로 context 300 고정 학습은 피한다. 기존 38~95 구간을 충분히 포함하는 mixed/bucket sampling으로 최대 300까지 노출하고, 40/80 성능 개선을 1차 성공 기준으로 둔다.

**Mixed-context sampler 구현 (2026-07-31)**
- Config: `configs/train_v22_hard_context300.yaml`
- 각 training shape group에서 중심을 `[40, 80, 120, 160, 180, 240, 300]` 중 균등 선택하고 정수 jitter `[-5, +5]`를 적용한다.
- Dataset은 선택된 context에 query 5~12개를 더해 총 bag 수를 생성한다. `ModelInterface`는 실제 query 수를 필터링하여 query 제거 후 남는 context가 반드시 선택 가능한 중심의 `±5` 안에 들도록 보장한다.
- 최대 조합은 context 305 + query 12 = 총 317 bags이며, 이 범위를 train dataset에만 적용한다. Validation/test의 기존 50~100 bags 분포는 유지한다.
- Batch 2, accumulation 4로 effective 8 episodes/update와 epoch당 optimizer step 수를 기존 Hard 기준선과 동일하게 유지한다.
- 테스트: 관련 dataset/model 27개 통과, batched forward 포함 31개 통과(169.244초).
- 최초 training smoke에서 whole-batch CUDA prefetch가 다음 대형 episode 생성을 현재 forward와 겹치며 순간 allocator OOM 경고를 냈다. Prefetch를 끈 뒤에도 batch 내부 두 CUDA stream의 episode 생성이 겹칠 때 같은 경고가 재현됐다. 두 run은 중단했다.
- 최종 안전 설정은 `cuda_prefetch: false`, `parallel_cuda_generation: false`다. 두 episode tensor는 순차 생성한 뒤 stack하여 batch 2 forward/backward를 수행하므로 학습 통계와 effective batch는 바뀌지 않는다.
- Prefetch-off 최대 경계 smoke: `(batch=2, bags=317, cells=1500, dim=512)`, query 12, 실제 context 305. Online generation부터 forward/backward까지 peak 74,546.6 MiB로 통과했다.
- 외부 W&B 인증에 의존하지 않도록 파생 config는 local CSV logger를 사용한다.
- Fine-tuning: Hard best epoch 44 checkpoint에서 epoch 45~49를 이어 학습 완료.
- 로그: `logs/20260731_context300_ft/v22_hard_context300_ft_serial.out`
- Checkpoints: `checkpoints/20260731_context300_ft/v22_hard_context300_ft_serial/`

| Fine-tuning epoch | val CE | val AUROC (104 episodes, 참고용) |
|---:|---:|---:|
| 45 | 0.6867 | 0.5542 |
| 46 | 0.6889 | 0.5588 |
| **47** | **0.6853** | 0.5623 |
| 48 | 0.6864 | 0.5586 |
| 49 | 0.6890 | 0.5626 |

- 기존 Hard best CE `0.6839`를 넘지 못했으므로 CE 기준 개선 신호는 없다. 다만 104-episode AUROC는 불확실하므로 epoch 47 checkpoint를 동일 1,000-episode context curve로 최종 비교한다.
- Fine-tuned checkpoint 평가 완료: context 40/80/160/300, fixed query, pool 400 bags, 1,000 episodes.

| Context | AUROC (episode-cluster 95% CI) | Log loss |
|---:|---:|---:|
| 40 | 0.5331 [0.525, 0.542] | 0.6992 |
| 80 | 0.5553 [0.546, 0.565] | 0.6901 |
| 160 | 0.5842 [0.575, 0.593] | 0.6798 |
| 300 | 0.5977 [0.588, 0.607] | 0.6762 |

- 로그: `logs/20260731_context300_eval/context_curve.out`
- 산출물: `logs/v22_hard_context300_ft_curve_1000ep.csv`, `predictions/v22_hard_context300_ft_curve/context_{40,80,160,300}.pt`
- 동일 pool 400·동일 episode의 원본 Hard checkpoint 대조 평가 완료:

| Context | 원본 AUROC | Fine-tuned AUROC | Δ |
|---:|---:|---:|---:|
| 40 | 0.5284 | 0.5331 | +0.0047 |
| 80 | 0.5483 | 0.5553 | +0.0070 |
| 160 | 0.5734 | 0.5842 | +0.0108 |
| 300 | 0.5839 | 0.5977 | +0.0138 |

- Fine-tuning 이득은 context가 클수록 증가하지만 사전 구조 후보 기준 overall `+0.03`에는 전 구간 미달합니다. 실제 ICI 범위 40/80의 이득도 `+0.005~0.007`로 작다.
- 대조 로그: `logs/20260731_context300_baseline_eval/context_curve.out`; summary: `logs/v22_hard_baseline_pool400_curve_1000ep.csv`

**Medium context-to-oracle 실험**
- 공식 checkpoint: `checkpoints/20260729_160643/v22_medium_fixed/epoch=013-val_ce_loss=0.5946.ckpt`; 기존 AUROC `0.7078 [0.696, 0.719]`.
- Config: `configs/train_v22_medium_context300.yaml`; Hard와 같은 context 중심 `[40,80,120,160,180,240,300]±5`, batch 2/accumulation 4, 순차 CUDA generation.
- Medium 원본 checkpoint pool-400 context curve 완료:

| Context | AUROC (episode-cluster 95% CI) | Log loss |
|---:|---:|---:|
| 40 | 0.6757 [0.665, 0.686] | 0.6456 |
| 80 | 0.7204 [0.709, 0.730] | 0.6100 |
| 160 | 0.7654 [0.755, 0.775] | 0.5817 |
| 300 | 0.7989 [0.790, 0.807] | 0.5639 |

- 재학습 전 모델도 context 40→300에서 `+0.1232` 상승해 context 활용 능력이 분명하다. 다만 300에서도 overall 0.80이며, state task의 비현실적 oracle-mask 0.9013과는 직접 같은 지표가 아니므로 혼동하지 않습니다.
- Baseline 평가 로그: `logs/20260731_medium_context_baseline_eval/context_curve.out`; summary: `logs/v22_medium_baseline_pool400_curve_1000ep.csv`.
- Mixed-context fine-tuning 완료:

| Epoch | val CE | val AUROC (104 episodes, 참고용) |
|---:|---:|---:|
| **14** | **0.5953** | 0.7340 |
| 15 | 0.5970 | 0.7371 |
| 16 | 0.5961 | 0.7387 |
| 17 | 0.5955 | 0.7378 |
| 18 | 0.5978 | 0.7344 |
| 19 | 0.5953 | 0.7340 |

- Best epoch 14도 원본 best CE `0.5946`을 넘지 못했다. CE 기준 개선 신호는 없다.
- Fine-tuning 로그: `logs/20260731_medium_context300_ft/train.out`; checkpoints: `checkpoints/20260731_medium_context300_ft/v22_medium_context300_ft/`.
- 사용자 요청에 따라 20 epochs에서 결론 내리지 않고 `max_epochs=100`으로 확장했다. Epoch 19 `last.ckpt`의 model/optimizer/scheduler/global-step을 모두 복원해 epoch 20부터 이어간다.
- 중간 best epoch 14의 context curve(PID `2542931`)와 대기 중이던 state upper-bound(PID `2543210`)는 최종 100-epoch 후보로 대체되므로 중단했다.
- 100-epoch run PID: `2544289`; 로그: `logs/20260731_medium_context300_100e/train.out`.
- Checkpoints: `checkpoints/20260731_medium_context300_100e/v22_medium_context300_100e/`.
- 현재 속도 약 3.4분/epoch 기준 epoch 99까지 예상 약 4.5시간. 완료 후 best CE checkpoint를 동일 pool-400 curve 및 state observable/oracle 진단으로 평가한다.
- 진행 확인: epoch 31의 42%, 경과 40분 43초, GPU 약 102.7 GiB, OOM/Traceback 없음. Epoch 20~30 val CE는 `0.5955~0.5999`; mixed-context 전체 best는 여전히 epoch 14의 `0.5953`이며 공식 원본 best `0.5946`을 넘지 못했다.
- 사용자 판단에 따라 epoch 31 validation 완료 후 run을 중단했다. Epoch 31 CE `0.5947`, AUROC `0.7351`로 공식 CE `0.5946`과 사실상 동률이다. Train total/CE의 epoch 20~30 평균도 하락하지 않아 추가 epoch보다 architecture 개선으로 전환한다.
- 상세 architecture 근거와 v23 후보: [`architecture_v23_candidates.md`](architecture_v23_candidates.md).
- **다음 우선순위 T5-A**: 기존 40-token aggregator는 고정하고 token type/slot/tail identity를 부여한 뒤, bag 내부에서 structured embedding을 만들고 각 labelled bag을 fixed 8-token class memory 없이 direct ridge/cross-attention에 전달합니다. 이 실험으로 episode-level context 압축과 bag-level 정보 손실을 먼저 분리합니다.
- State oracle-mask `0.9013`/latent `1.0`은 정답 반응세포 또는 latent score를 사용하므로 달성 목표가 아니라 비현실적 참고 상한입니다. 실제 성공 기준은 관측 입력만으로 context 40/80 및 overall AUROC가 개선되는지다.

**T4-0. 재학습 없는 접근성 감사 [먼저]**
- Hard best checkpoint로 state `model_input`/`observable`/`oracle` 상한 재측정
  - 완료: 현재 모델 0.5175 [0.496, 0.538], model-input 최대 0.5336 [0.510, 0.556](+0.016, 기준 +0.05 미달), oracle-mask 0.6953
  - 산출물: `logs/v22_hard_state_upper_bound_1000ep.csv`; 로그 `logs/20260731_hard_accessibility/state_upper_bound.out`
- Hard config에서 covariance all-cell/oracle-mask 상한 재측정
  - 완료: all-cell 최대 0.5044 [0.483, 0.525], 현재 모델 0.5103과 동률; oracle-mask covariance 최대 0.6074 [0.564, 0.652]
  - 산출물: `logs/v22_hard_covariance_upper_bound_1000ep.csv`; 로그 `logs/20260731_hard_accessibility/covariance_upper_bound.out`
- effect scale 0.4/0.7 matched 평가로 Hard task 상대 약점이 생성기 scale 아티팩트인지 확인
  - 완료: scale 0.4 overall 0.5086 [0.500, 0.518], scale 0.7 overall 0.5282 [0.519, 0.538]; 모든 단일 task 0.498~0.531, combined만 scale 0.7에서 0.5645
  - 산출물: `predictions/synthetic_v22_hard_scale{0.4,0.7}_1000ep.pt`; 로그 `logs/20260731_hard_accessibility/matched_effect.out`

**T4-1. 누적 bridge ablation [T4-0 후]**
1. **Signal scarcity**: class separation + rare fraction/probability
2. **Nuisance**: donor shift + component shift + observation noise
3. **Geometry/scale**: latent dim + cell/bag 범위
4. **Optimization/model setting**: batch/accumulation + covariance rank

Medium에서 시작해 위 그룹을 하나씩 Hard 값으로 바꾸고, 동일 optimizer-step 예산과 1,000-episode paired evaluation을 사용합니다. 처음 성능이 급락하는 그룹 안에서만 one-factor ablation을 합니다. 모든 조합을 탐색하지 않습니다.

**판정 기준**
- overall AUROC 차이 **+0.03 이상** 또는 target task **+0.05 이상**
- episode cluster CI + paired bootstrap 필수
- 관측 가능한 헤드룸이 없는 상태에서 새 architecture head 추가 금지
- ICI는 Medium+Hard 후보 확정 전까지 실행 금지

### ~~Tier 2 — state 분기~~ — 🛑 **종료 (2026-07-31)**

상세 결과는 §3의 T2-1/T2-2 표를 참고합니다. `model_input`, `observable`, `oracle_mask`, `oracle_latent`를 처음부터 분리했고, 현재 모델이 관측 가능 probe와 동률이라는 사전 종료 조건이 충족됐습니다. 따라서 T2-3 이득 곡선과 state 아키텍처 변경은 진행하지 않습니다.

### Tier 3 — 방법론 (성능이 아니라 판단 신뢰도)

**T3-1. ✅ 완료 (2026-07-30) — 진짜 약점은 `state`.**
effect scale을 통일해 비교하니 covariance는 composition과 동률이고 **state가 전 구간 최하위**. 기본 config의 covariance 최하위는 생성기 아티팩트였습니다. 상세는 §3.
재현: `python scripts/evaluate_synthetic.py --checkpoint <best>.ckpt --config configs/train_v22_medium.yaml --val-episodes 400 --effect-scale 0.7`

<details><summary>원래 T3-1 계획 (완료)</summary>
현재 task별 AUROC는 생성기 effect scale(composition 1.40 / state 0.72 / covariance 0.55)에 오염되어 **아키텍처의 상대적 강약을 직접 비교할 수 없습니다.** 모든 task의 effect scale을 동일하게 맞춘 진단용 데이터 config를 만들면 "어느 메커니즘에 실제로 약한가"를 처음으로 공정하게 볼 수 있습니다. 학습된 모델을 그 데이터로 평가만 하면 되므로 재학습 불필요.
</details>

**T3-2. ✅ 완료 (2026-07-30) — 권고: val episode 1,000개.**
CI 폭 104→0.074 / 400→0.035 / 1,000→0.021. task별은 1,000에서 0.045. **task별 +0.05 미만을 노리는 실험은 1,000개로도 부족하므로 2,000개 이상 필요**합니다. 상세는 §3. 이 과정에서 공식 기준선도 1,000 episode 기준으로 갱신했습니다(0.7078).

<details><summary>원래 T3-2 계획 (완료)</summary>
현재 104 episodes → CI 폭 0.060. Tier 1/2에서 기대하는 개선폭이 그보다 작다면 검출이 안 됩니다. `val_dataset_kwargs.episodes_per_epoch`를 늘려 CI를 좁히세요 (episode 수가 실질 표본 크기, §5).
</details>

**T3-3. ✅ 완료 (2026-07-31) — v22 Hard 기준선.**
Best `val_ce_loss 0.6839`; 1,000-episode AUROC `0.5483 [0.538, 0.558]`. state 0.5167 / covariance 0.5103으로 거의 무작위이며, v21 Phase 2 CE 0.6845를 재현했습니다. 상세는 위 T3-3 완료 표 참고.

### ~~Tier 1 — 반응세포 식별 (Sparse Evidence)~~ — 🛑 **종료 (2026-07-29)**

**결론: 진행하지 않습니다.** 사전 판정 기준에 따라 T1-C 2단계에서 종료했습니다 (bag 라벨 purity 0.128 ≤ 0.15). 전체 근거 사슬은 §3의 T1-0 → T1-A → T1-B → T1-C 참고.

| 단계 | 질문 | 결과 |
|---|---|---|
| T1-0 | covariance 상한이 실재하는가 | 오라클 세포 0.893 / **실제 관측 가능 0.570** — 모델 0.612는 이미 그 위 |
| T1-A | 세포 선택이 반응세포를 찾는가 | 기하 3종 + 학습 1종 **전부 AUROC ~0.50** |
| T1-B | 슬롯 단위로는 되는가 | fragmentation entropy 0.963 — 12슬롯에 흩어짐. 단 세포 라벨 held-out은 0.697 |
| T1-C 1 | 부분 개선도 값이 있는가 | **있음** — 곡선 선형, 순도 0.40에서 분기 +0.107 |
| T1-C 2 | bag 라벨만으로 되는가 | **안 됨** — purity 0.128 (무작위 0.110) → **종료** |

<details><summary>종료된 Tier 1 상세 계획</summary>


T1-0/T1-1 진단으로 원래 가설이 무너지고 병목이 바뀌었습니다 (§3 참고). 요약:
- 모델 0.6122는 **진짜 관측 가능한 상한 0.5704를 이미 초과**합니다.
- 0.8931은 **반응세포 오라클 마스크**를 받은 경우이며 모델은 그 마스크를 못 받습니다.
- 반응세포는 전체의 **11.7%**뿐. **그 세포들을 찾아내는 것이 유일한 실질 레버**입니다.

**T1-A. ✅ 완료 (2026-07-29) — 세포 선택이 무작위와 구분되지 않음.**
기하학적 기준 3종(`outlier_distance` 0.5091 / `studentized` 0.5098 / `novelty` 0.4984)과 **학습된 기준**(`class_memory` 0.4971) 전부 AUROC ~0.5. precision = base rate, recall = 유지 비율. 4개 task 모두 동일. 상세는 §3.
**결론: k값 튜닝은 무의미합니다.** 랭킹에 신호가 없습니다.
재현: `python scripts/diagnose_cell_selection.py --config configs/train_v22_medium.yaml --val-episodes 400 --checkpoint <best>.ckpt`

**T1-B. ✅ 완료 (2026-07-29) — 슬롯 정렬 실패, 그러나 특징에는 신호 있음.**
슬롯 capture 0.155(무작위 0.083)·fragmentation entropy 0.963 → 반응 component는 12개 슬롯에 흩어짐. 반면 held-out LDA는 0.6969 → **특징에는 신호가 있고 메커니즘이 없는 것**. 상세는 §3.
재현: `python scripts/diagnose_oracle_slot_alignment.py --config configs/train_v22_medium.yaml` / `python scripts/diagnose_cell_selection.py --probe --checkpoint <best>.ckpt`

<details><summary>원래 T1-B 계획 (완료)</summary>
T1-A가 실패 원인을 짚어줬습니다: 반응세포는 `effect_mask = (component_index == effect_component_index)`, 즉 **latent mixture component 하나**이고, covariance task에서는 **위치가 아니라 분산이 바뀝니다.** 중심에서 먼 세포를 찾는 현재 기준으로는 구조적으로 못 찾습니다.
→ 개별 세포가 아니라 **component 단위**로 접근해야 합니다. aggregator는 이미 12개 slot에 세포를 배정하므로:
1. **반응 component가 특정 슬롯과 정렬되는지 측정** — 기존 도구 `scripts/diagnose_oracle_slot_alignment.py`가 이 용도입니다.
2. 정렬된다면 → 슬롯별 공분산(`slot_covariance_sketch`, 이미 계산 중)으로 0.89에 얼마나 근접하는지 측정.
3. 정렬되지 않는다면 → 슬롯 배정 자체(anchor 선정, `assignment_temperature`)가 문제.
</details>

**T1-C 1단계. ✅ 완료 (2026-07-29) — 이득 곡선 선형, Tier 1 유지.**
순도 0.11→1.00에서 covariance AUROC 0.517→0.888로 **선형 증가, 문턱 없음**. 현실적(held-out LDA, 순도 0.40) 기대치는 분기 **0.557 → 0.664 (+0.107)**. 상세는 §3.
재현: `python scripts/diagnose_selection_gain_curve.py --config configs/train_v22_medium.yaml --val-episodes 400`

**T1-C 2단계. [다음] bag 라벨만으로 판별 방향을 학습할 수 있는지 검증.**
남은 핵심 미지수는 하나입니다: **세포 라벨 없이, bag 라벨(R/NR)만으로 순도 0.4 수준의 선택을 학습할 수 있는가?** 이것이 되면 +0.107이 현실이고, 안 되면 Tier 1은 여기서 끝입니다.
- 값싼 선행 검증: episode 여러 개에 걸쳐 **bag 라벨로 학습한 선형 판별 방향**(R bag 세포 vs NR bag 세포)이 반응세포를 얼마나 골라내는지 측정. 세포 라벨을 안 쓰므로 **실제 학습 가능한 상한**이 됩니다. `diagnose_cell_selection.py`에 score를 하나 추가하는 수준의 작업입니다.
- 이 값이 순도 0.3~0.4를 내면 → 구조 변경(attention 기반 세포 가중) 착수 근거 확보.
- 0.15 이하면 → **Tier 1 종료**, Tier 2/3으로 이동.

> [!WARNING]
> **T3-2(val episode 증량)가 Tier 1 검증의 선행 조건입니다.** covariance는 전체의 20%라 성공해도 전체 AUROC로는 +0.014 수준이고 현재 CI 폭(0.060)에 묻힙니다. §3의 검출 가능성 경고 참고.

</details>

### 하지 말 것

- **ICI 실행 금지** — 합성에서 후보가 확정되기 전까지. §5 참고, 지금 돌리면 테스트 세트를 조기 소진합니다.
- **오라클 기반 상한을 목표치로 삼기 금지** — descriptor가 `responsive_instance_mask`나 latent 파라미터를 쓰는지 항상 먼저 확인하세요. 모델이 못 받는 정보로 만든 상한은 목표가 아닙니다 (§3 정정 사례).
- **effect scale 정규화 없이 task별 AUROC로 우열 판단 금지** (§3 T3-1). 기본 config의 task별 표는 생성기 난이도에 오염되어 있습니다.
- **상한 측정 없이 아키텍처부터 뜯기 금지** — Tier 1 전체가 학습 한 번 없이 진단만으로 닫혔습니다. 관측 가능한 상한 → 모델 위치 → 이득 곡선 순서를 지키세요.
- **104 episode 결과로 판정 금지** — `--val-episodes 1000` 사용 (§3 T3-2).

---

---

## 28. 2026-08-04 — v30 S1 실행 중: B1 `poolz` / `poolz_l2` 학습 + 자동 평가 큐

**상태**: **백그라운드 실행 중** (구현·테스트 완료, 커밋 `b3808cb`). 아키텍처 플래그는 기본 OFF이므로
**v24 확정 baseline은 그대로**이며, 승격 판정은 아래 게이트 통과 후 별도로 받습니다.

**구현** (`bag_representation`: `"legacy"`(기본) | `"poolz"` | `"poolz_l2"`):
per-bag centering을 **context-pool 대각 표준화**로 대체합니다 — `z = (x − μ_ctx)/σ_ctx`,
`poolz_l2`는 여기에 per-cell L2. `classification_instances`만 바뀌고 `global_summary`(per-bag
centered spread)와 `centered_delta`는 그대로라 **공분산/spread 분기는 v24와 동일**합니다.
pool 통계는 **context bag의 모든 세포**에서 cell-count 가중(실 bag 크기 1~1044 → per-bag 평균의
평균은 오가중), query 누출 없음. 주입 지점 3곳: `forward_episode_batch`(에피소드별),
`aggregator.forward`(`context_mask` 검증을 `_bag_view` 앞으로 이동), `BaseModel.forward`
(`is_context`를 중복 `_bag_view` 앞으로 이동 — 안 하면 rare-evidence/MIL 분기가 slot 분기와 다르게
정규화된 세포를 받음).

**검증**: unittest **164/164 통과**(136.6s, 신규 10건 — 기본값 legacy 유지, 잘못된 플래그 조합 거부,
pool 통계 없으면 raise, poolz는 magnitude 보존/poolz_l2는 단위 노름, pool 통계가 query bag 무시,
batched==list pool 통계 및 forward 일치(≤1e-4), forward/backward 유한, **n=1 bag이 legacy에서는
0벡터지만 poolz에서는 아님**). 1-epoch 실 스모크(poolz) `val_loss 0.436`(v24 musklike-easy 스모크
0.485), NaN·크래시 없음.
또한 `_context_pool_stats`를 `unbiased=False`로 batched twin과 정확히 일치시켰습니다 — Bessel 보정을
쓰면 학습(batched)과 실데이터 추론(list)이 **~0.25% 조용히 어긋납니다.**

**실행 중인 큐** (`scripts/queue_v30_poolz.sh`, 단일 B200이라 순차):
train 50 epoch → 1,000-ep 합성 eval → Musk zero-shot eval → cardinality 층화 리포트, 2개 run 연속.

| 항목 | 값 |
|---|---|
| Run 1 | `v30_musklike_easy_poolz` — **진행 중** (10:02:34 시작) |
| Run 1 PID / 로그 | `2174854` / `logs/20260804_100234/v30_musklike_easy_poolz.out` |
| Run 1 체크포인트 | `checkpoints/20260804_100234/v30_musklike_easy_poolz/` |
| Run 2 | `v30_musklike_easy_poolz_l2` — Run 1 종료 후 자동 시작 |
| 오케스트레이터 PID / 로그 | `2183035` / `logs/queue_v30_poolz.log` |
| 진행 상황 | epoch 1에서 `val_loss 0.409`, GPU 114GB/77%, ~105s/epoch → 50 epoch ≈ 90분/run |
| 예측 산출물 | `predictions/synthetic_v30_musklike_easy_poolz{,_l2}_1000ep.pt`, `predictions/musk_v30_musklike_easy_poolz{,_l2}.pt` |

**사전 등록 게이트** (config 주석에도 명시):
- `poolz`: 합성 musklike-easy 1,000-ep AUROC **≥0.94**(현 0.9510 무회귀) **AND** Musk 전체 **≥0.89**
- `poolz_l2`: 합성 **≥0.94** **AND** Musk 전체 **≥0.92**
- **층화 보고 필수**(n≤4 / 5–10 / 11–34 / n>34). 현 최고는 Musk 0.822이고 n≤4가 0.475이므로,
  소형 구간 개선이 없으면 전체 수치만으로는 무엇이 개선됐는지 판정 불가.
- 근거 천장(§26): `poolz` Musk 0.874 / n≤4 0.900, `poolz_l2` 0.912 / 0.967. 모델은 자기 입력의
  선형 천장을 +0.055~0.072 초과해 왔으므로 실현 기대치는 각각 **≈0.93 / ≈0.97**입니다.

**큐 스크립트 버그 수정 (실행 중 발견)**: 아카이브에서 계승한 `pgrep -f "scripts/train.py"`는
**그 문자열을 명령줄에 포함한 모든 프로세스**(예: 큐 상태를 확인하는 `pgrep -af scripts/train.py`
쉘)를 매칭해, 큐가 학습이 살아있다고 오판하고 `wait_gpu_free`에서 영구 대기했습니다.
`/proc/<pid>/comm`으로 **실제 python/torchrun 프로세스만** 필터링하도록 수정했습니다. 또한 launcher가
학습을 detach하므로 오케스트레이터를 재시작해도 중복 실행되지 않게 `ICF_ATTACH_FIRST=1`
(진행 중인 첫 run을 입양해 완료 대기 후 평가) 모드를 추가했습니다.

**다음 Action**: 큐 완료 후 ① 게이트 판정(합성 무회귀 + Musk 전체·층화), ② 통과 시 v30 승격 여부와
`poolz` vs `poolz_l2` 선택을 사용자 확인, ③ 미달 시 S2(B2 cardinality-faithful 샘플링)로 진행 —
B2는 아키텍처 무변경이며 소형 bag을 학습 분포에 넣어 B1의 천장을 실제로 실현시키는 단계입니다.
진행 확인: `timeout 5s tail -5 logs/queue_v30_poolz.log`.

### §28 결과 (1) — `poolz`: **게이트 미달, 음성** (2026-08-04 11:39)

50 epoch 정상 완주(`max_epochs=50 reached`, launcher "completed successfully"), best
`val_ce_loss` **0.2374** @ epoch 43 — musklike-easy 계열 **최저**(v24 0.2552, rawstats 0.2468,
IA-MIL 0.2462). **그런데 판정 지표는 통과하지 못했습니다.**

| 지표 | `poolz` | v24 musklike-easy (기준) | 게이트 | 판정 |
|---|---|---|---|---|
| 합성 1,000-ep AUROC | **0.9506** [0.946,0.955] | 0.9510 [0.946,0.956] | ≥0.94 | ✅ 무회귀(동률) |
| 합성 log loss | 0.2837 | 0.2836 | — | 동률 |
| **Musk zero-shot AUROC** | **0.7757** [0.671,0.866] | **0.8030** / 0.8217(raw) | **≥0.89** | ❌ **미달, 기준선보다 낮음** |
| Musk log loss | 0.6941 | 0.5439 | — | ❌ 크게 악화 |

층화(`--report stratified`): n≤4 **0.442**(v24 0.475, **개선 없음**) / 5–10 0.833(0.825) /
11–34 0.927(0.988) / n>34 **0.540**(0.667). **pearson(prob, log n) = +0.327** — §24 IA-MIL의
크기 편향 지문(+0.327)과 동일 수치, baseline은 +0.012.

> [!IMPORTANT]
> **판정: `poolz` 음성.** 예측(선형 천장 0.874 + 모델의 관측 초과폭 +0.06 → ≈0.93)이 실현치
> 0.776으로 **−0.15 어긋났습니다.** §26/§3.1에서 "초과폭이 새 표현에서도 유지된다는 것은 **가정**"
> 이라고 명시했던 그 가정이 **깨졌습니다.**
>
> **측정된 실패 메커니즘** — `‖분류 뷰의 bag 평균‖` vs bag 크기 n (Musk 102 bag, 모델 불필요):
>
> | 뷰 | corr(log‖·‖, log n) | log-log 기울기 | n≤4 중앙값 | n>34 중앙값 | 배율 |
> |---|---:|---:|---:|---:|---:|
> | legacy (center+L2) | **+0.528** | +2.67 | **0.000** | 0.028 | — |
> | `poolz` | **−0.535** | −0.153 | 11.43 | 6.66 | 1.7× |
> | `poolz_l2` | **−0.611** | −0.141 | 0.974 | 0.507 | 1.9× |
>
> legacy는 소형 bag의 **1차 적률을 0으로 없애고**(n=1이면 정확히 0, n=2면 대척쌍이라 합이 0),
> `poolz`는 그것을 **보존하지만 크기 n과 결합**시킵니다(표본 평균의 수축; feature 상관이 높아
> iid의 −0.5보다 완만한 −0.15). 즉 `poolz`는 "소형 bag에 신호가 없음"을 "신호는 있으나 **bag
> 크기와 교란됨**"으로 바꿨을 뿐입니다. 모델은 **에피소드 내 모든 bag이 동일 크기이고 n이 항상
> 500~1000인** 분포에서만 학습했으므로 이 교란을 분리하는 법을 배운 적이 없고, 그 결과가
> prob-vs-log n **+0.327**과 소형 구간 무개선입니다.
>
> **선형 ridge 천장이 예측력을 갖지 못한 이유도 이제 분명합니다**: ridge는 feature별로 재스케일할
> 수 있고 **n에 대해 평균하지 않습니다.** 반면 모델의 slot 통계는 bag의 n개 세포를 **평균**하므로
> 1/√n 계열 수축이 그대로 descriptor 스케일에 들어옵니다. 천장 프로브는 이 경로를 모사하지 않습니다.

**계획 수정 (자기 정정)**: §3.0에서 "`poolz`의 합성 대형 비용이 −0.019뿐이므로 **B2는 선행조건이
아니라 증폭 요인**"이라며 순서를 B1→B2로 바꿨는데, **그 철회가 틀렸습니다.** 원래 순서
**B2 → B1이 옳습니다** — 단 이유는 처음 생각한 "합성 천장 손실" 때문이 아니라, **magnitude를
보존하는 어떤 표현이든 bag 크기와 교란되며, 모델이 크기 불변성을 배우려면 학습 분포에 크기 변동이
있어야 한다**는 것입니다. B2(bag별 log-uniform[1,1024] 샘플링)가 **B1의 전제조건**입니다.

**`poolz_l2`(진행 중)에 대한 기대치 하향**: 위 표에서 `poolz_l2`도 corr **−0.611**로 같은 크기
교란을 보입니다(절대 크기만 1/12로 작음). per-cell L2로 magnitude를 유계화해도 **상대적 크기
의존성은 남습니다.** 따라서 통과를 예상하지 않습니다 — 다만 이미 학습 중이므로
"uncentering 자체"와 "magnitude 스케일"을 분리하는 값싼 데이터포인트로 완주시킵니다.

### §28 결과 (2) — `poolz_l2`: **게이트 미달. S1(B1 단독) 전체 음성 판정** (2026-08-04 13:21)

50 epoch 정상 완주, best `val_ce_loss` **0.2414** @ **epoch 49(마지막)** — epoch 43은 0.2433이었으므로
**아직 개선 중이었습니다(미수렴)**. 50 epoch가 이 변형에는 부족할 수 있다는 단서로 기재합니다.

**S1 최종 요약** (모든 수치 1,000-ep 합성 / 102-bag Musk LOO):

| run | 합성 AUROC | 합성 LL | **Musk 전체** | **n≤4** | 5–10 | 11–34 | **n>34** | corr(prob, log n) | 게이트 |
|---|---|---|---|---|---|---|---|---|---|
| v24 musklike-easy (기준) | 0.9510 | 0.2836 | **0.803** | 0.475 | 0.825 | 0.988 | 0.667 | +0.012 | — |
| `poolz` | 0.9506 | 0.2837 | 0.776 | 0.442 | 0.833 | 0.927 | 0.540 | **+0.327** | ❌ (≥0.89) |
| `poolz_l2` | **0.9520** | **0.2803** | 0.762 | **0.517** | 0.767 | 0.976 | **0.460** | +0.100 | ❌ (≥0.92) |

> [!IMPORTANT]
> **판정: B1(uncentering) 단독은 음성.** 두 변형 모두 합성 무회귀는 통과했으나(오히려 `poolz_l2`가
> 최고 0.9520) **Musk에서 기준선 0.803보다 낮습니다**(0.776 / 0.762). 승격하지 않으며
> `bag_representation` 기본값은 `legacy`로 유지합니다. **v24 확정 baseline 변경 없음.**
>
> **다만 실패의 모양이 진단을 지지합니다** — uncentering은 **예측한 곳에서 정확히 이겼고 다른 곳에서
> 잃었습니다**: `poolz_l2`의 n≤4는 **0.517로 6개 체크포인트 중 최고**(v24 0.475, 기준선 유일 초과)이고,
> magnitude를 유계화한 덕에 크기 편향도 +0.327 → **+0.100**으로 줄었습니다(제 예측 방향 일치).
> 그런데 **n>34에서 0.667 → 0.460으로 붕괴**해 총합이 음수가 됐습니다.
>
> **해석**: 모델은 **에피소드당 cell 수가 하나로 고정되고 항상 500~1000인** 분포에서만 학습했으므로
> **크기 불변성을 배운 적이 없고, 소형·대형 두 레지므를 동시에 만족시킬 수 없습니다.** 표현을 소형
> 쪽으로 옮기면 대형 쪽을 잃습니다. 이것이 §28 결과(1)에서 재확정한 **B2 선행 필요성**의 두 번째,
> 그리고 더 직접적인 증거입니다.
>
> **또 하나의 교훈 — 선형 ridge 천장은 이 모델의 예측 지표가 아닙니다.** `poolz_l2`는 합성 천장이
> −0.099로 가장 나빴는데 실현 합성 AUROC는 **최고**(0.9520)였고, Musk 천장은 0.912로 가장 좋았는데
> 실현 Musk는 **최악**(0.762)이었습니다. **부호가 양쪽 다 반대입니다.** 향후 천장 프로브는 방향
> 탐색용으로만 쓰고 **기대치 산출 근거로 쓰지 않습니다**(§26의 "+0.06 초과폭" 추정도 폐기).

**B2 실행 가능성 확인 (모델 불필요)**: 소형 bag 가드는 `pooled context candidates < num_slots(12)`
이며 **per-bag 제약이 아닙니다**(`baseline.py:302`, `:939`). 따라서 bag 수가 충분하면 n=1도 통과합니다:

| bags/episode | n=1 | n=2 | n=3 | n=4 | n=8 |
|---|---|---|---|---|---|
| 6 | ❌ ValueError | ❌ | ✅ | ✅ | ✅ |
| 20 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 60 | ✅ | ✅ | ✅ | ✅ | ✅ |

`num_bags: [60, 100]`이므로 **`num_cells: [1, 1024]` 전 구간 사용 가능**합니다. (Musk가 지금도 n=1 bag을
처리하는 이유도 동일 — context가 101 bag·~6,500 세포.)

**B2 구현**: `num_cells_log_uniform`(기본 OFF, 기존 경로 무변경). uniform은 소형 bag을 사실상 못 보므로
log-uniform이 필수입니다:

| 분포 | median | frac(n≤12) | frac(n≤34) |
|---|---|---|---|
| uniform [1,1024] | 510 | 0.01 | 0.03 |
| **log-uniform [1,1024]** | **34** | **0.36** | 0.50 |
| Musk 실제 | 12 | 0.50 | 0.75 |

(log-uniform이 실제 Musk보다 상단이 두꺼운 잔여 불일치는 그대로 기재합니다.)
**남은 구조적 격차**: 4D batched 경로와 `shape_group_size`가 에피소드 내 동일 shape을 요구하므로
B2는 **에피소드 간** 크기 변동만 줍니다 — Musk의 **에피소드 내** 혼합(1~1044 동시)은 아닙니다.

**다음 (S2, 큐 예정)**: ① `v30_cardinality_poolz_l2`(B2+B1) ② `v30_cardinality_legacy`(B2 단독 대조 —
단 legacy는 n=1을 0벡터로 만들므로 최소 에피소드가 퇴화 뷰로 학습됨, 그 비대칭이 비교의 요점).
게이트: 합성 ≥0.94 **AND** Musk ≥0.85 **AND** n≤4 ≥0.60 **AND** n>34가 0.60 미만으로 떨어지지 않을 것
— S1이 실패한 방식(한 구간을 다른 구간과 교환)을 전체 수치만으로는 구분할 수 없으므로 층화 필수.

### §28 (3) — S2 실행 시작 (2026-08-04 13:23)

| 항목 | 값 |
|---|---|
| Run 1 | `v30_cardinality_poolz_l2` (B2+B1) — **진행 중** (13:23:34 시작, PID `2617890`) |
| Run 1 로그 / 체크포인트 | `logs/20260804_132334/` / `checkpoints/20260804_132334/v30_cardinality_poolz_l2/` |
| Run 2 | `v30_cardinality_legacy` (B2 단독 대조) — Run 1 종료 후 자동 |
| 큐 로그 | `logs/queue_v30_s2.log` |
| 스모크 | 1-epoch 실 루프 정상(`val_loss 0.577`), 가변 shape 크래시 없음 |

진행 확인: `timeout 5s tail -5 logs/queue_v30_s2.log`

### §28 결과 (4) — **S2 `v30_cardinality_poolz_l2`(B2+B1): 게이트 통과, 첫 양성 결과** (2026-08-04 14:37)

50 epoch 정상 완주. best `val_ce_loss` 0.4442 @ 48 — **S1 수치(0.2374/0.2414)와 비교 불가**입니다
(데이터 분포가 log-uniform으로 바뀌어 bag당 세포가 훨씬 적으므로 CE가 당연히 높습니다).

| 지표 | B2+`poolz_l2` | v24 기준선 | S1 `poolz_l2` | 게이트 | 판정 |
|---|---|---|---|---|---|
| **Musk 전체** | **0.8539** [0.774,0.925] | 0.8030 | 0.762 | ≥0.85 | ✅ |
| **Musk n≤4** | **0.800** [0.52,0.98] | 0.475 | 0.517 | ≥0.60 | ✅ |
| Musk 5–10 | 0.833 | 0.825 | 0.767 | — | ✅ |
| Musk 11–34 | 0.958 | 0.988 | 0.976 | — | ~ |
| **Musk n>34** | **0.698** | 0.667 | 0.460 | ≥0.60 | ✅ |
| corr(prob, log n) | **+0.059** | +0.012 | +0.100 | — | ✅ 편향 거의 없음 |
| Musk log loss | **0.4746** | 0.5439 | 0.6001 | — | ✅ 보정 개선 |
| Balanced acc | **0.7845** | 0.7747 | 0.7173 | — | ✅ |
| 합성(log-uniform 분포) | 0.8512 [0.840,0.862] | — | — | — | ⚠ 아래 주석 |

**paired bootstrap** (`--report paired`, 4,000 resample; 두 모델이 **같은 102 bag**을 채점하므로
독립 CI 비교보다 검정력이 높음 — n=102에서 독립 CI는 크게 겹칩니다):

| stratum | bags | v24 | B2+`poolz_l2` | Δ | 95% CI of Δ | P(cand>base) |
|---|---|---|---|---|---|---|
| ALL | 102 | 0.803 | 0.854 | **+0.051** | [−0.002, +0.111] | **0.971** |
| **n≤4** | 29 | 0.475 | **0.800** | **+0.325** | **[+0.092, +0.567]** | **0.997** |
| n>4 | 73 | 0.863 | 0.863 | **+0.001** | [−0.054, +0.060] | 0.504 |

> [!IMPORTANT]
> **진단이 확증됐습니다.** 개선이 **예측한 지점에 정확히 국소화**되어 있습니다 —
> **소형 bag Δ+0.325, CI가 0을 제외**(P=0.997), 그리고 **대형 bag은 Δ+0.001로 완전 무해**(P=0.504).
> 전체 Δ+0.051의 CI가 0을 살짝 포함하는 것은 **102개 중 28개(n≤4)만 고장나 있었기 때문**이며,
> 부분집합만 고쳤으니 전체 지표에서 희석되는 것이 정상입니다.
>
> 인과 사슬이 전부 맞았습니다: **cardinality가 병목** → **B1 단독은 구간 교환**(S1, 소형 +0.04 /
> 대형 −0.21) → **B1+B2는 소형을 고치면서 대형을 잃지 않음**(+0.325 / +0.001).
> 현 최고 기록도 경신했습니다: 0.822(`--preprocess raw`) → **0.854**.

**⚠ 남은 게이트 항목 — 합성 무회귀는 아직 미검증**: 위 합성 0.8512는 **새 log-uniform 분포**에서
측정된 값이라 원래 기준(0.9510, n=500~1000)과 **비교할 수 없습니다**. `evaluate_synthetic.py`는 넘겨받은
config로 val 데이터셋을 만들기 때문입니다. 이를 위해 **모델은 고정하고 데이터만 원래대로 돌리는**
eval 전용 config를 준비했습니다: `configs/eval_v30_cardinality_{poolz_l2,legacy}_on_largebags.yaml`.
Run 2 종료 후 실행합니다.
> **주의(문서화됨)**: `bag_representation`은 **파라미터를 추가하지 않으므로**, poolz_l2 체크포인트를
> legacy config로 로드해도 **에러 없이 조용히 잘못된 뷰로 평가**됩니다. 위 eval config가 표현을
> 명시적으로 고정하는 이유입니다. (큐 자체는 항상 학습 config를 넘기므로 영향 없음.)

**진행 중**: Run 2 `v30_cardinality_legacy`(B2 단독 대조, 14:36:49 시작) — B2 단독으로 충분한지,
아니면 B1이 필요한지 귀속시킵니다.

### §28 결과 (5) — **S2 최종 판정: 게이트 전 항목 통과. B1+B2는 상호 필수** (2026-08-04 14:45)

**Run 2 `v30_cardinality_legacy`(B2 단독 대조)는 학습 자체가 불가능했습니다** — epoch 0, optimizer
step 490에서 프로젝트 자체 가드(`src/modules/model_interface.py:76`)가 발동:

```
RuntimeError: Non-finite gradients at epoch=0, optimizer step=490:
['model.aggregator.slot_residual_logit', 'model.aggregator.center_slot_encoder.0.weight', ...]
```

원인은 유닛테스트로 고정해 둔 그 사실입니다 — **legacy(center+L2)는 n=1 bag을 정확히 0벡터로,
n=2를 대척쌍으로 만듭니다.** log-uniform 샘플링에서 그런 에피소드가 다수 발생하므로 all-zero 뷰가
slot encoder 경로로 흘러 NaN 그라디언트가 됩니다. 즉 **B2 단독은 "성능이 낮다"가 아니라
v24 표현으로는 원리적으로 학습 불가**입니다.

> [!IMPORTANT]
> **B1과 B2는 상호 필수(mutually required)입니다.**
> B2는 학습 가능하려면 uncentered 뷰가 필요하고(위 NaN), B1은 구간 교환을 피하려면 B2가 필요합니다
> (S1: 소형 +0.04 / 대형 −0.21). **실패한 대조군이 귀속을 완성했습니다** — 이 조합의 이득은 어느 한
> 쪽으로 환원되지 않습니다.

**교차 분포 평가 (모델 고정, 데이터만 교체 — 마지막 게이트 항목)**:

| 체크포인트 | 평가 데이터 | AUROC | log loss | 비교 |
|---|---|---|---|---|
| **B2+`poolz_l2`** | **원래 대형 bag** (n=500~1000) | **0.9483** [0.944,0.953] | 0.2987 | v24 **0.9510** [0.946,0.956] → **Δ−0.003, 무회귀** |
| B2+`poolz_l2` | 신규 log-uniform | 0.8512 [0.840,0.862] | 0.4658 | ↓ |
| v24 musklike-easy | 신규 log-uniform | 0.8328 [0.820,0.845] | 0.4977 | → B2 학습이 **+0.018** 우위 |

**S2 Run 1 사전 등록 게이트 — 전 항목 통과**:

| # | 게이트 | 결과 | 판정 |
|---|---|---|---|
| 1 | 합성 무회귀 ≥0.94 (원래 분포) | **0.9483** (v24 0.9510, Δ−0.003) | ✅ |
| 2 | Musk 전체 ≥0.85 | **0.8539** [0.774,0.925] | ✅ |
| 3 | Musk n≤4 ≥0.60 | **0.800** (v24 0.475) | ✅ |
| 4 | Musk n>34 ≥0.60 유지 | **0.698** (v24 0.667) | ✅ |
| 5 | paired: 대형 구간 무해 | Δ**+0.001**, P=0.504 | ✅ |
| 6 | paired: 소형 구간 유의 개선 | Δ**+0.325**, CI [+0.092,+0.567] **0 제외** | ✅ |

> **판정: S2(B2 + B1 `poolz_l2`)는 사전 등록 게이트를 전부 통과했습니다.** Musk 최고 기록
> 0.822 → **0.854** 경신. **v30 승격 후보이며, 승격 여부는 v26/v27/v29와 동일한 규율에 따라
> 사용자 확인이 필요합니다** (`bag_representation` 기본값은 현재도 `legacy`, v24 baseline 무변경).

**다음 목표가 바뀌었습니다 — 이제 가장 약한 구간은 대형 bag입니다**:

| 구간 | v24 | **B2+B1** | 상태 |
|---|---|---|---|
| n≤4 (29) | 0.475 | **0.800** | 해결됨 (원래 병목) |
| 5–10 (22) | 0.825 | 0.833 | 양호 |
| 11–34 (25) | 0.988 | 0.958 | 양호 |
| **n>34 (26)** | 0.667 | **0.698** | ⚠ **최약 구간** |

0.95 목표까지 남은 거리는 **+0.096**이고 그 대부분이 **n>34 구간**에 있습니다. 원안의 다음 단계를
그대로 따르기보다 이 구간을 직접 겨냥하는 것이 타당합니다. 후보:
1. **B2b — 에피소드 내 cardinality 혼합**: 현재 B2는 4D batched 경로와 `shape_group_size` 제약 때문에
   **에피소드 간** 변동만 줍니다. Musk는 한 에피소드에 1~1044이 **동시에** 존재하므로, list 경로 또는
   크기 버킷팅으로 에피소드 내 혼합을 구현하면 대형·소형 동시 처리를 직접 학습시킬 수 있습니다.
2. **대형 bag 특이 진단**: n>34에서 무엇이 실패하는지 미규명입니다(§26 §6.3의 미해결 항목과 동일).
   top-1% tail이 n과 함께 커지며 신호가 희석되는지 먼저 측정할 것.
3. B3(2차 통계 shrinkage) / B4(생성기 any-positive)는 그 다음.

---
