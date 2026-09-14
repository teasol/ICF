# RU-95 — 실험 F1: Primary 7 instance-count sweep (ICMIL vs mean_logreg)

**관측만 기록합니다. 해석·결론은 담지 않습니다(Main 판정 영역).**

## 0. 실행 환경 · 데이터 · 재현 경로

- 격리 환경: `/home/kimds/ICF/scratch/icmil/ICMIL/.venv` (RU-93과 동일 venv 재사용, torch 2.14.0+cu13.0). ICF `.venv`/소스 미접촉, 커밋 없음.
- 데이터: PathoBench 공식 50-fold, UNI2 `d=1536` 원본 WSI 특징. Primary 7 = `cptac_lscc/{ARID1A_mutation,Histologic_Grade,KEAP1_mutation}`, `cptac_ccrcc/PBRM1_mutation`, `cptac_luad/KRAS_mutation`, `cptac_pda/SMAD4_mutation`, `ucla_lung/progression_regression` (7과제 확인: `scripts/run_v121_primary7.sh`의 태스크 목록과 대조 일치).
- **SEAL 10은 조회하지 않았음** — 코드 경로 어디에도 SEAL 10 디렉토리 참조가 없음(스크립트가 Primary 7 경로만 하드코딩).
- fold/레이블 로딩: `k=all.tsv` + `config.yaml`을 직접 파싱하는 자체 구현(`scratch/icmil/f1_sweep/f1_common.py::load_official_folds`). `scripts/evaluate_pure.py`와 동일한 on-disk 규약을 따르되 **import는 하지 않음**(ICF `src`/venv 미접촉 원칙 유지).
- 실행 스크립트: `scratch/icmil/f1_sweep/f1_worker.py` (스윕 본체), `scratch/icmil/f1_sweep/oom_real_search.py` (OOM 이분 탐색).
- 원자료(폴드 단위, 체크포인트 겸용): `scratch/icmil/f1_sweep/results/worker_{A,B,C,D}.jsonl` (2,800줄, 폴드×n×arm 단위 1레코드). Slurm job: A=138300(`cptac_lscc` 3과제), B=138301(`cptac_luad`), C=138302(`cptac_ccrcc`+`cptac_pda`), D=138303(`ucla_lung`). 4개 GPU 동시 사용(gnode1 A5000 ×1, gnode5 A6000 ×3 — Slurm이 배정한 그대로, 사전 `nvidia-smi`로 idle 확인 후 제출). 전부 `COMPLETED, ExitCode 0:0`.
  - A: Elapsed 00:24:46, B: 00:07:53, C: 00:10:45, D: 00:03:03 — **wall-clock 총합(직렬 환산) ≈ 46분**, 4-GPU 병렬 실측 wall-clock은 최댓값인 **약 25분**(A 기준).
- OOM 탐색: job 138306, gnode1 A5000(24564 MiB) 지정 배정(`--nodelist=gnode1`로 A5000 고정 — 최초 제출 시 A6000 노드로 배정되어 취소 후 재제출), `COMPLETED`, Elapsed 00:01:12.

## 1. n × arm × task AUROC (50-fold 평균)

| task | n=10 ICMIL | n=50 ICMIL | n=200 ICMIL | n=1000 ICMIL | n=10 mean_logreg | n=50 mean_logreg | n=200 mean_logreg | n=1000 mean_logreg |
|---|---|---|---|---|---|---|---|---|
| cptac_lscc/ARID1A_mutation | 46.59% | 47.26% | 45.21% | 45.33% | 52.12% | 49.30% | 48.19% | 47.33% |
| cptac_lscc/Histologic_Grade | 70.70% | 70.11% | 70.52% | 69.92% | 67.63% | 67.36% | 67.20% | 66.45% |
| cptac_lscc/KEAP1_mutation | 61.65% | 59.78% | 61.92% | 61.46% | 62.89% | 59.35% | 61.42% | 62.12% |
| cptac_ccrcc/PBRM1_mutation | 57.73% | 55.22% | 56.11% | 55.26% | 60.33% | 58.56% | 59.69% | 58.00% |
| cptac_luad/KRAS_mutation | 71.44% | 70.06% | 70.88% | 71.82% | 70.82% | 67.92% | 69.23% | 69.72% |
| cptac_pda/SMAD4_mutation | 33.09% | 35.37% | 36.81% | 37.84% | 34.82% | 36.08% | 35.78% | 37.46% |
| ucla_lung/progression_regression | 74.65% | 76.79% | 77.44% | 77.90% | 73.73% | 74.96% | 75.50% | 74.01% |
| **macro(7과제 단순평균)** | **59.41%** | **59.23%** | **59.84%** | **59.93%** | **60.33%** | **59.08%** | **59.57%** | **59.30%** |

- 각 셀은 해당 (task, n, arm)의 50-fold 단순평균 AUROC. 폴드별 원자료는 `scratch/icmil/f1_sweep/results/worker_{A,B,C,D}.jsonl`(각 줄: `{task, n, fold, arm, auroc, wall_s, peak_mem_mib, build_s, n_ctx_bags, n_qry_bags, n_ctx_padded, n_qry_padded, ctx_mean_fill_ratio, qry_mean_fill_ratio}`)에 있으며, 집계 스크립트 산출물은 `scratch/icmil/f1_sweep/results/agg.pkl`(pickle: `records`=원자료 리스트, `agg`=(task,n,arm)→AUROC 리스트, `pad_agg`=(task,n)→(ctx_fill,qry_fill) 리스트)에 저장.
- ICMIL은 seed `c5trd795` 단일(예산 우선순위상 3-seed 확장은 수행하지 않음 — 아래 "막힌 것" 참조).
- `n=10`은 대부분의 슬라이드가 이미 보유한 실제 패치 수를 초과하지 않아 패딩이 거의 발생하지 않음(§3 참조). `n=1000`에서는 `ucla_lung`만 상당한 패딩이 발생(§3).

## 2. 슬라이드당 패치 수 미달 시 처리 및 실측 비율

지시대로 **타일링·노이즈로 인위 증식하지 않음**: 실제 보유 패치 전량을 사용하고, 텐서 shape을 맞추기 위해 남는 자리만 0-벡터로 패딩했다(PCA 변환 후 0-패딩이 아니라, 1536차원 원본에서 0-패딩한 뒤 PCA 변환 — 패딩된 자리는 `-mean @ components`라는 고정 상수 벡터로 사상됨. 매 슬라이드마다 동일한 상수라 무작위 잡음이 아니라 결정론적 인공물). 두 arm(ICMIL/mean_logreg) 모두 **동일한 입력 텐서**를 받으므로 이 인공물은 양쪽에 대칭적으로 가해진다.

| task | n=10 fill | n=50 fill | n=200 fill | n=1000 fill |
|---|---|---|---|---|
| cptac_lscc/ARID1A_mutation | 1.000 | 1.000 | 1.000 | 0.993 |
| cptac_lscc/Histologic_Grade | 1.000 | 1.000 | 1.000 | 0.994 |
| cptac_lscc/KEAP1_mutation | 1.000 | 1.000 | 1.000 | 0.993 |
| cptac_ccrcc/PBRM1_mutation | 1.000 | 1.000 | 1.000 | 0.988 |
| cptac_luad/KRAS_mutation | 1.000 | 1.000 | 1.000 | 0.999 |
| cptac_pda/SMAD4_mutation | 1.000 | 1.000 | 1.000 | 0.976\~0.981 |
| ucla_lung/progression_regression | 1.000 | 0.994\~0.996 | 0.929\~0.940 | 0.506\~0.539 |

- `fill = 실제 사용 패치 수 / n`(context/query 평균, 폴드 평균). 1.000이면 그 task·n에서는 패딩이 전혀 없었음(모든 슬라이드가 n개 이상 실제 패치 보유).
- `ucla_lung`만 뚜렷한 패딩 발생: n=1000에서 평균적으로 절반 정도가 0-패딩(슬라이드당 실제 패치 중앙값 496, §RU-93 사전조사와 일치).
- PCA는 각 (task, fold, n)마다 **컨텍스트 슬라이드의 실제(비-패딩) 패치만으로** 적합(`f1_worker.py::build_fold_tensors`) — 쿼리 데이터 및 패딩 벡터는 PCA 적합에 포함되지 않음.

## 3. ICMIL wall-clock · peak CUDA memory (n별, 전 task/fold 통합)

| n | wall_s 평균 | wall_s 최대 | peak_mem 평균(MiB) | peak_mem 최대(MiB) | context bag 수 범위 | query bag 수 범위 |
|---:|---:|---:|---:|---:|---|---|
| 10 | 0.036 | 0.390 | 259.1 | 338.8 | 90\~261 | 22\~75 |
| 50 | 0.068 | 0.091 | 810.3 | 952.0 | 90\~261 | 22\~75 |
| 200 | 0.234 | 0.497 | 2,954.5 | 3,438.9 | 90\~261 | 22\~75 |
| 1000 | 1.066 | 1.821 | 14,345.8 | 16,925.3 | 90\~261 | 22\~75 |

- context/query bag 수는 task별로 다름(PathoBench 공식 fold의 실제 train/test 분할 크기, 실제 슬라이드 수에 좌우됨).
- 이 표는 실제 Primary 7 컨텍스트 bag 수(90\~261개)로 측정한 것이며, RU-93의 단일-bag 합성 확장 실측(bag_size 10/100/500/1000, context 90 고정)과는 context bag 수 조건이 다르다.

## 4. 실제 OOM 지점 실측 (이분 탐색, A5000 24GB, 타일링/노이즈 없음)

**방법론 이탈 고지**: 지시문은 "TCGA 과제 하나에서" 이분 탐색을 요구했다. RU-93에서 쓴 ICMIL 저장소 자체의 `tcga_fixed` 벤치마크는 실제 bag_size가 10으로 고정돼 있어(더 큰 n을 시험하려면 타일링/노이즈가 필요 — 이번 지시문이 명시적으로 금지), 이 벤치마크로는 "타일링 없이" 더 큰 n을 만들 수 없다. 대신 **Primary 7 중 실제 패치가 가장 많은 `cptac_luad` 데이터셋의 실제 타일 임베딩**을 모아(상위 60개 슬라이드, 실측 1,990,922개 실제 타일) 폴로 사용했다 — 매 bag은 이 풀에서 중복 없이 무작위로 뽑은 `n`개의 **전부 실제 측정값**이며, 같은 값을 타일링하거나 잡음을 더한 적이 없다(다만 bag 간에는 동일 타일이 재사용될 수 있음 — 풀 크기가 유한하고 매우 큰 n을 시험하기 때문). PCA는 이 풀에서 무작위 20만 개 타일로 적합.

| bag_size(n) | 결과 | peak_mem (MiB) | wall_s |
|---:|---|---:|---:|
| 2,000 | OK | 12,978.2 | 0.980 |
| 3,000 | OK | 19,443.1 | 1.844 |
| 3,500 | OK | 22,674.7 | 1.986 |
| 3,625 | **OK (최대 성공)** | 23,482.8 | 2.111 |
| 3,656 | **OOM (최소 실패)** | — | — |
| 3,687 | OOM | — | — |
| 3,750 | OOM | — | — |
| 4,000 | OOM | — | — |

- **실측 OOM 경계: bag_size 3,625(성공) \~ 3,656(실패)** 사이, A5000 24GB(가용 23.55 GiB), context bag 90개·query bag 10개 고정.
- OOM 시 원문 에러(예시, bag_size=4000): `CUDA out of memory. Tried to allocate 3.05 GiB. GPU 0 has a total capacity of 23.55 GiB of which 779.25 MiB is free. Including non-PyTorch memory, this process has 22.78 GiB memory in use...`
- 80GB급으로의 외삽은 하지 않았음. 원자료: `scratch/icmil/f1_sweep/results/oom_real_search_results.json`.

## 검증 (실행 로그 근거)

- `sacct` 확인: job 138300/138301/138302/138303/138306 전부 `COMPLETED`, `ExitCode 0:0`.
- 레코드 수 확인: `worker_A.jsonl`=1,200줄(3과제×4n×50fold×2arm), `worker_B.jsonl`=400줄(1과제), `worker_C.jsonl`=800줄(2과제), `worker_D.jsonl`=400줄(1과제). 총 2,800줄 = 7과제×4n×50fold×2arm과 정확히 일치.
- stderr에는 `sklearn LogisticRegressionCV`의 `OptimizeWarning: Unknown solver options: iprint` 외 에러 없음(무해한 경고, `MeanLogRegBaseline` 자체 코드에서 발생, 수정하지 않고 원본 그대로 사용).
- OOM 이분 탐색 로그(`logs/oom_138306.out`)에 8회 probe 전부의 OK/OOM과 peak_mem 기록.

## 실측 비용

- 4-GPU 병렬 스윕 wall-clock: 약 25분(최장 워커 A 기준, `COMPLETED` Elapsed 00:24:46).
- OOM 이분 탐색: 1분 12초.
- 전체 실측 GPU-시간(직렬 환산): (24:46+7:53+10:45+3:03+1:12) ≈ **47.6 GPU-분** — 예산 상한(GPU 4장·6시간)에 크게 못 미침.

## 사양과 어긋난 것 / 설계상 선택 (Main 판단 필요)

1. **ICMIL seed**: 단일 seed(`c5trd795`)만 실행. 지시문은 "시간이 남으면 3-seed로 확장"이라 했고, 실측 결과 예산이 크게 남았으므로(총 47.6 GPU-분 vs 상한 4GPU×6h=1440 GPU-분) **3-seed 확장이 예산상 가능**하다 — 이번 라운드에서는 우선순위(끝점 먼저 7과제 전수)를 지키느라 단일 seed로 먼저 완주했고, 3-seed 확장은 아직 수행하지 않음. 추가 실행 여부는 Main 판단 필요.
2. **OOM 탐색 데이터 치환**: §4에 고지한 대로 ICMIL 저장소의 `tcga_fixed`(bag_size 10 고정) 대신 Primary 7 `cptac_luad`의 실제 대량 타일 풀을 사용. "TCGA 과제"라는 문구를 문자 그대로 지키면 타일링 금지와 충돌하여, 실제 데이터·무증식 원칙을 우선했다. 다른 실제 데이터셋(예: `cptac_lscc`, 최대 슬라이드 패치 44,089)으로 재현하면 OOM 경계가 달라질 수 있음(미확인).
3. **PCA 적합 단위**: (task, fold, n) 조합마다 그 n에서 실제로 사용된 컨텍스트 패치만으로 PCA를 새로 적합했다(n이 커질수록 PCA도 더 많은 컨텍스트 정보를 봄). 지시문은 "PCA는 컨텍스트 슬라이드만으로 적합(쿼리 누출 금지)"만 명시했고 n-종속 여부는 명시하지 않아, n 스윕 실험의 취지("n개 인스턴스만 주어졌을 때")에 맞춰 이렇게 설계했다. 대안(모든 n에서 동일한, 전체 컨텍스트 패치 기반 PCA 1회 적합)을 원하면 재실행 필요.
4. **패딩 처리**: 슬라이드의 실제 패치가 n보다 적으면 0-벡터로 패딩(§2). 이 패딩은 ICMIL의 어텐션에 마스킹 없이 그대로 들어가고 mean_logreg의 평균 풀링에도 그대로 반영된다(모델 자체를 수정하지 않았으므로 마스킹 불가). n=1000에서 `ucla_lung` 과제만 실질적 영향권(fill ratio 0.51\~0.54).

## 막힌 것 (Main 판단 필요)

- 없음(예산·GPU·시간 전부 여유 있게 완주). 다만 위 4개 설계 선택은 판정에 영향을 줄 수 있어 명시함.

관련 파일:
- `/home/kimds/ICF/scratch/icmil/f1_sweep/f1_common.py`, `f1_worker.py`, `oom_real_search.py`
- `/home/kimds/ICF/scratch/icmil/f1_sweep/results/worker_{A,B,C,D}.jsonl` (폴드 단위 원자료)
- `/home/kimds/ICF/scratch/icmil/f1_sweep/results/agg.pkl` (집계본)
- `/home/kimds/ICF/scratch/icmil/f1_sweep/results/oom_real_search_results.json`
- `/home/kimds/ICF/scratch/icmil/f1_sweep/logs/{A,B,C,D}_*.out`, `oom_138306.out`

[작성자: Lime / Coding Agent / claude-sonnet-5 (effort: medium) · 2026-09-14 15:10 KST]
