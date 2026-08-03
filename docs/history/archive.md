# Archived sections from current_status.md

This is the running archive for fully-resolved / superseded sections that were
moved out of `docs/current_status.md` to keep the living doc compact. Each
section keeps its original heading so cross-references still resolve.

- 2026-08-02: archived §4 (v22 retrieval removal decision), §9 (2026-07-31 session), §10 (2026-08-01 session) — all superseded by later sections (§3 v24 decision, §11 v25 retirement).

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

