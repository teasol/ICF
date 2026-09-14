# RU-93 — ICMIL 공개 체크포인트 재현 (Gate 1 판정용 실측)

- 목적: ICLR 2027 투고(D-12) Gate 1 — ICMIL(arXiv 2606.06458) 공개 체크포인트가 논문 수치를 실측 재현하는지 확인.
- 실행 위치: `/home/kimds/ICF/scratch/icmil/ICMIL` (원본 `/tmp/ICMIL_probe`를 복사, ICF `.venv`/소스는 건드리지 않음).
- 격리 환경: `uv venv --python 3.12 .venv` + `uv pip install -e .` (torch 2.14.0+cu13.0, numpy 2.5.3, scikit-learn 1.6.1, h5py 3.16.0, schedulefree 1.4.1, tabpfn 2.2.1). ICF 저장소 커밋 없음.
- 실행 환경: nexgem 로그인 노드에는 GPU가 없어 Slurm `sbatch --gres=gpu:1`로 gnode1(RTX A5000, 24564 MiB)에 제출. 제출 전 `nvidia-smi`로 GPU0 유휴(0% util, 1 MiB 사용) 확인 후 잡이 그 GPU를 단독 점유. Job 138294, `COMPLETED`, Elapsed 00:01:02.

## 1. 12개 과제 AUROC 실측 (ICMIL 3-seed 앙상블 평균, mean ± SEM across seeds)

```
python -m icmil.reproduce --baselines none --icmil-seeds all --tasks all --output results_icmil_only
```

| 과제 | AUROC | ±SEM(seed간) |
|---|---:|---:|
| tcga_luad_vs_lusc | 87.92% | 0.11 |
| rsna_ich_resnet50_draws_100bags | 74.98% | 0.07 |
| mnist_xai_smil_100bags | 74.36% | 0.30 |
| mnist_xai_pos_neg_100bags | 86.94% | 0.21 |
| mnist_xai_adjacent_pairs_100bags | 77.62% | 0.49 |
| uci_musk1 | 93.34% | 0.58 |
| uci_musk2 | 90.63% | 1.75 |
| uci_letters | 95.73% | 0.71 |
| uci_hepmass | 84.53% | 0.71 |
| andrews_fox | 59.70% | 0.69 |
| andrews_tiger | 88.85% | 0.59 |
| andrews_elephant | 94.08% | 0.16 |
| **평균(12과제 단순평균)** | **84.06%** | — |

- **논문 대비**: 논문 보고 12-과제 평균 84.17 vs 실측 84.06, **Δ = -0.11 pp**. 실측이 재현 오차 범위 내에서 논문 수치와 매우 근접함(과제별 seed간 SEM이 0.07\~1.75pp 수준이므로 전체 Δ는 그 안에 든다).
- **과제별 논문 개별 수치 대조**: 저장소(`README.md`, `THIRD_PARTY.md`, docstring, 코드 주석) 어디에도 과제별 세부 AUROC 표는 들어있지 않음. `icmil/reproduce.py`가 생성하는 표만 있고, 논문 Table 자체는 동봉돼 있지 않아 **과제 단위 대조는 불가**(평균값만 논문 텍스트로 제공됨). 이 12개 과제·task 이름 나열(`TASKS` 리스트, `icmil/reproduce.py:41-54`)은 논문 Table 열 순서와 일치한다고 코드 주석에 명시돼 있어 열 대응 자체는 신뢰 가능.
- run_meta: `torch=2.14.0(cu13.0)`, `gpu=NVIDIA RTX A5000`, `allow_tf32={matmul: false, cudnn: true}`, 데이터 파일 sha256 기록됨 (`results_icmil_only/run_meta.json`).

## 2. `icmil/baselines/` 비교군 목록 (구현 확인, 실행은 미실행 — 아래 "막힌 것" 참조)

| 이름 | 파일 | 요지 (한 줄) |
|---|---|---|
| `mean_logreg` | `tabpfn_baselines.py::MeanLogRegBaseline` | 각 bag의 instance를 평균 풀링(mean-pool)한 뒤 `LogisticRegressionCV`(불가하면 plain `LogisticRegression`)로 분류. 학습이나 그래디언트 없는 가장 단순한 baseline. |
| `svm_summ` | `tabpfn_baselines.py::SVMSummBaseline` | bag을 (sum, mean, median, min, max, stdev) 6종 요약통계로 집계한 뒤 `SVC`(RBF, Platt-scaling)로 분류. `C`는 stratified K-fold `GridSearchCV`로 선택. |
| `abmil` | `abmil_baseline.py::ABMILBaseline` | Ilse et al. 2018 gated-attention MIL. patch MLP → gated attention pooling → linear head. `(lr, wd)` 그리드를 stratified val split + early stopping으로 매 split마다 학습. |
| `acmil` | `acmil_baseline.py::ACMILBaseline` | Zhang et al. 2023 ACMIL-GA. `n_token`개 병렬 gated-attention 브랜치 + stochastic top-k masking + composite loss(bag_ce+branch_ce+diversity)로 소수 instance 과의존을 억제. |
| `dsmil` | `dsmil_baseline.py::DSMILBaseline` | Li et al. 2021 DSMIL. instance-stream(선형분류+max pool)과 bag-stream(critical instance 기준 attention)의 로짓을 평균. |
| `tabpfn_concat` | `tabpfn_baselines.py::TabPFNConcatBaseline` | Bag을 원래 instance 순서로 flatten해 `max_tabpfn_features`까지 자르고, 고정 TabPFN v2 backbone에 단일 view로 통과. |
| `tabpfn_subsample` | `tabpfn_baselines.py::TabPFNSubsampleBaseline` | `n_views`회 각기 다른 instance 부분집합(`n_keep`)을 flatten해 TabPFN에 통과시키고 로짓을 평균(aggregation="mean_logits"). |
| `cluster_tabpfn` | `tabpfn_baselines.py::ClusterTabPFNBaseline` | instance를 KMeans(`n_clusters=5`)로 군집화한 요약을 TabPFN에 투입. |

- **`mean_logreg`의 정확한 정의**(docstring, `tabpfn_baselines.py:15-17`): "Mean-pools instances per bag and fits a scikit-learn `LogisticRegressionCV` (or plain `LogisticRegression` when stratified CV is infeasible)." 즉 각 bag의 instance 벡터를 feature 축 평균으로 접어(bag_size 차원 제거) 단일 벡터로 만들고, 그 위에 로지스틱 회귀만 적합하는 non-neural, non-attention 베이스라인. 과제 서두에서 언급한 "MeanLogReg 82.37"의 정체가 이것.
- 8종 baseline은 모두 코드 구조·docstring으로 확인했을 뿐 **실제 실행(학습)은 하지 않았음** — 아래 "막힌 것" 참조.

## 3. 비용 실측

| 항목 | 값 |
|---|---|
| (a) 체크포인트 파라미터 수 | `total_params = 8,257,552` (8.26M), 3개 seed 체크포인트 모두 동일 arch(`in_features=25, embedding_size=256, mlp_hidden_size=1054, num_attention_heads=4, num_column_row_iterations=6, feature_group_size=1, bag_chunk_size=32`). 파일 크기 32MB/seed(fp32 다중 optimizer state 포함 여부는 미확인 — `state_dict` 키는 `arch, epoch, model_state_dict`뿐이라 optimizer state는 없음; 8.26M×4B ≈ 33MB로 파일 크기와 부합). |
| (b) 과제 1건당 추론 wall-clock | `run_meta.json`의 `seconds_per_cell`(3-seed 합산, 50/50/5/5개 split 기준): tcga 3.61s, rsna 8.77s, mnist_xai 3종 각 ~3.5s, musk1 0.46s, musk2 7.49s, uci_letters 0.33s, uci_hepmass 0.33s, andrews 3종 각 ~0.60s. 가장 무거운 과제(rsna, musk2)도 10초 미만. |
| (c) 전체 재현 wall-clock | Slurm `sacct` 기준 job 138294 **Elapsed = 00:01:02**(62초, GPU 준비·파라미터 카운트·12과제×3seed 전수 실행·bag_size 스케일링 포함 전체). |

## 4. 컨텍스트 수용 한계 (bag_size / n_bags) — 코드 확인 + 실측

### 코드 확인 (`icmil/model.py`, `icmil/models/architecture.py`)
- **instance 축(bag 안의 N)에는 하드코딩된 상한이 없음.** `InstanceAggregationBlock.forward`(`architecture.py:39-65`)에서 cross-attention의 `kv`가 `(B*cnb*G, N, E)`로 reshape되며, `N`(instance 개수)에는 어떤 assert/cap도 걸려 있지 않다 — 순수히 메모리·시간 비용으로만 스케일.
- 유일한 명시적 상한은 `_SDPA_MAX_BATCH = 65535`(`architecture.py:23`)로, 이는 SDPA가 받는 **leading batch dim**(`B*chunk*G`, bag 수 축)에 대한 CUDA grid-dim-x 제한이며 `chunk = min(bag_chunk_size, max(1, 65535 // (B*G)))`로 자동 청킹되어 회피된다. instance 수(N)에는 적용되지 않는다.
- **학습 시 최대 bag size는 코드상 커리큘럼에만 등장**: `icmil/datagen/config.py:157` `CurriculumStage(until_batch=40000, min_bag_size=6, max_bag_size=20, ...)` — 두 번째(마지막) 커리큘럼 단계에서 `max_bag_size=20`으로 고정. 과제 서두 메모의 "논문은 20"과 일치. 즉 **훈련 분포의 최댓값이 20**일 뿐, 아키텍처 자체는 그 값을 강제하지 않는다(in-context 학습 모델이라 forward pass에 bag_size 관련 파라미터 shape 의존성이 없음 — `bag_latent_init`은 feature-group 축 `G`에만 의존, instance 축과 무관).
- 정리: **에러가 나는 하드 제약은 없고**, 학습 분포를 벗어난 bag size(20 초과)를 넣으면 "동작은 하되 학습 분포 밖 분포외 추론(out-of-distribution)"이 되어 정확도 보증이 없어지는 구조.

### 실측 (TCGA 과제, seed `c5trd795`, 실제 bag_size=10을 타일링+노이즈로 확장 — 정확도 주장 아님, 동작 여부만 확인)

```
python bag_scale_test.py   # results_icmil_only/bag_scale_results.json
```

| bag_size | 동작 여부 | wall-clock(1회 forward) | peak CUDA mem |
|---:|---|---:|---:|
| 10(실제) | OK | 0.199s | 118.6 MiB |
| 100 | OK | 0.052s | 696.3 MiB |
| 500 | OK | 0.230s | 3282.4 MiB |
| 1000 | OK | 0.455s | 6516.3 MiB |

- 세 크기(100/500/1000) 모두 **에러 없이 동작**. 메모리는 bag_size에 대해 대략 선형(1000에서 약 6.5GB, A5000 24564 MiB의 약 26.5%). wall-clock은 bag_size=10→100 구간에서 오히려 줄었는데(0.199s→0.052s) 이는 CUDA 커널 warm-up/캐시 효과로 추정되며(첫 호출이 10, cold start), 100 이후로는 예상대로 단조 증가.
- 이번 실측 범위(≤1000)에서는 **붕괴 지점을 관측하지 못함** — 예산(GPU 1장, 2시간) 안에서 사양대로 100/500/1000까지만 확인했고, OOM 등 실제 한계점 탐색은 지시 범위 밖.

## 검증 요약 (실행 로그 근거)
- `sbatch run_all.sbatch` → job 138294, `sacct` 상태 `COMPLETED`, `ExitCode 0:0`.
- stdout(`logs/repro_138294.out`)에 12개 과제 AUROC 출력 확인, `results_icmil_only/{benchmark_table.md,results.json,run_meta.json}` 저장 확인.
- `bag_scale_test.py` 실행 로그에서 4개 bag_size 모두 `OK` 확인, `results_icmil_only/bag_scale_results.json` 저장 확인.
- stderr(`logs/repro_138294.err`) 비어 있음(에러 없음).

## 사양과 어긋난 것
없음. 12개 과제 전수, 비용 3종, bag_size 확장(100/500/1000) 모두 사양대로 실측.

## 막힌 것 (Main 판단 필요)
1. **8종 baseline 실행 미실행.** 무엇: `--baselines all` 실행은 하지 않음(목록·구현 요지만 코드로 확인). 왜: 사양 §2가 "무엇이 구현돼 있는지 목록과 요지"만 요구했고, ABMIL/ACMIL/DSMIL은 과제·seed마다 `(lr, wd[, dropout])` 그리드 서치 + 최대 200 epoch 학습이 들어가 12과제×3seed×수십 조합이면 예산(2시간) 안에서 완주를 장담하기 어려움. 어느 부분을 바꾸면 뚫리는가: (a) 사양에 "베이스라인 AUROC도 실측하라"가 추가되면 별도 작업으로 `python -m icmil.reproduce --baselines all --icmil-seeds none`을 몇 개 과제만 골라(`--tasks`) 우선 실행해 예산 내 비용을 먼저 재고, 전체 실행 여부를 판단하는 방식을 제안.
2. **과제별 논문 개별 AUROC 대조 불가.** 무엇: 저장소 안에 과제별 논문 수치 표가 없음(README/THIRD_PARTY.md/docstring 어디에도 부재). 왜: 이 저장소는 "표 재현 코드"만 배포하고 논문 원문 수치는 담지 않음. 어느 부분을 바꾸면 뚫리는가: arXiv 2606.06458 PDF의 Table을 직접 대조 자료로 받으면 과제 단위 Δ까지 계산 가능(현재는 평균 84.17만 사양에 주어짐).
3. **bag_size 붕괴 지점(OOM 등) 미관측.** 무엇: 1000까지는 문제없이 동작해 사양이 요구한 상한(100/500/1000) 안에서는 실패 지점이 나타나지 않음. 왜: 사양이 정확히 그 세 값만 요구했고, A5000 24GB 기준 선형 외삽하면 6516MiB/1000 ≈ 6.5MiB/instance이므로 이론상 약 bag_size≈3600 부근에서 24GB에 근접(다른 프로세스와 공유 시 더 낮음) — 이는 **미측정 추정치**이며 실측이 아님. 어느 부분을 바꾸면 뚫리는가: "실제 붕괴점까지 실측"이 사양에 추가되면 bag_size를 2000/4000 등으로 늘려 추가 GPU 시간을 배정하면 확인 가능.

관련 파일:
- `/home/kimds/ICF/scratch/icmil/ICMIL/results_icmil_only/{benchmark_table.md,results.json,run_meta.json,bag_scale_results.json}`
- `/home/kimds/ICF/scratch/icmil/ICMIL/logs/repro_138294.{out,err}`
- `/home/kimds/ICF/scratch/icmil/ICMIL/{param_count.py,bag_scale_test.py,run_all.sbatch}`

[작성자: Lime / Coding Agent / claude-sonnet-5 (effort: medium) · 2026-09-14 14:20 KST]
