# Agent Handoff Reference (Stable Architecture & Contracts)

This document serves as the permanent, stable architectural reference for the In-Context Foundation (ICF) project under the **Universal Handoff Protocol**.
For live in-flight tasks, current milestones, and immediate execution commands, refer to [`docs/current_status.md`](current_status.md).

---

## 1. Project Overview

The **In-Context Foundation (ICF)** project develops in-context classification models for pathology whole-slide images (WSI) using pre-extracted multiple-instance learning (MIL) patch embeddings (e.g. UNI2, 1536D).

### Active Baseline: v120 (0-Parameter Deterministic 6-Branch Trimmed Mean Voting)
The active baseline is **v120**, a completely training-free (0 learned parameters), fully deterministic classifier ($\text{seed std} = 0.00000$):
- **Core Principle**: Given labeled Context slides and unlabeled Query slides, a within-slide PCA basis ($K=256$) is constructed using only Context slides. Six complementary statistical branches extract slide representations, compute independent decision margins, convert margins to probabilities via sigmoid, and aggregate them using **Trimmed Mean Voting** (discarding the highest and lowest probability per slide, averaging the remaining 4).
- **Benchmark Performance (6-branch 계보 기록값 — 현행 비교 기준 아님)**:
  - **Primary 7-Task Benchmark**: Macro Fold-mean AUROC **`0.6265`** (v119: 0.6247, v118: 0.6205).
  - **SEAL 10-Task Hold-out**: Macro Fold-mean AUROC **`0.6972`** (independent validation).
  - **All 17-Task Overall Macro**: **`0.6681`**.

### 현행 비교 기준: v121 5-Branch (CT 제외)
CT 브랜치는 k-means 병목으로 350-fold 소요를 45분 → 12분으로 늘리는 반면 기여가 확인되지 않아, §214-V부터 **모든 승격 비교는 CT를 제외한 5-branch(`CV, BM, BD, QA, DS`)에서 수행**한다.
- **공식 기준선**: v121 5-branch Trimmed Mean, Primary 7 Macro **`0.6171`** (`v121_baseline`, 50-fold, 재현 검증 완료).
- **Oracle 상한 (과제별 최상 단독 브랜치 선택 시)**: **`0.6565`** — 현 기준선 대비 **+3.94%p**. 이 격차가 현재 프로젝트의 실제 미회수 성능이다.
- 위 두 수치는 §214-V에서 `predictions/*_v121_baseline_official50_bf16.pt`로부터 독립 재계산하여 확인했다.

---

## 2. Architecture & Pipeline Contracts

### 2.1. Directory Structure & Key Roles
- `src/models/training_free.py`: Core implementation of the 6-branch pipeline, feature extraction, Dual Ridge solvers, and voting aggregations.
- `src/models/ct/`: Cell-Type abundance dictionary construction and soft-token assignment (k-means++ / Lloyd).
- `src/datasets/`: Dataset loaders and interfaces (`base_data.py`, `synthetic_data.py`).
- `scripts/lib/arms.sh`: Single source of truth for branch configurations and parameter injection for historical and active arms (`icf_arm_v120`).
- `scripts/eval_v120.sh`: Primary evaluation entrypoint for the v120 baseline.
- `scripts/run_v120_seal_multi_gpu.sh`: Multi-GPU evaluation runner across SEAL 10 hold-out tasks.
- `scripts/node_env.sh`: Centralized environment resolver; discovers Python interpreter (`.venv`), GPU counts, and paths.
- `scripts/run_tests.sh`: Fast regression test suite runner with thread capping and PYTHONPATH bootstrapping.
- `configs/`: Structured into `configs/baseline/` (v120, v119, v118), `configs/experiments/` (research templates), and `configs/archive/` (historical lineage and legacy groups).
- `docs/`: Living documentation (`agent_handoff.md`, `current_status.md`, `current_architecture.md`) and historical archives (`docs/history/`).



> ⚠️ **[STRICT RULE] 모델 구조, 브랜치 로직, 수식, 하이퍼파라미터를 분석하거나 수정할 때는 반드시 [`docs/current_architecture.md`](current_architecture.md)를 먼저 정독하고 동기화할 것.**

### 2.2. Six Branches

> ⚠️ **[§215 실측 정정] 이 브랜치들은 상보적(complementary)이지 않다.** 5-branch 앙상블의 유효 랭크는 **2.26 / 5 (명목의 45%)** 이며, `BM`·`QA`·`DS`는 모두 동일한 top-32 PCA 부분공간의 1차 모멘트로서 상호 상관 r = 0.57~0.93 (셋의 유효 랭크 **1.29 / 3**) — 사실상 하나의 신호다. 나머지와 직교하는 브랜치는 **`BD` 뿐**(|r| = 0.01~0.16)이다.
> 그 결과 Trimmed Mean은 독립 신호를 나르는 `CV`(슬라이드의 83.6%)와 `BD`(62.1%)를 우선적으로 절사하고, 중복 블록의 중앙값에 수렴한다. 신규 브랜치를 추가하거나 집계 방식을 바꾸기 전에 반드시 `branch_diagnostics.py --redundancy`로 기존 브랜치와의 상관을 먼저 측정할 것.
1. **CV (Cross-Covariance)**: Captures 2nd-order feature correlations via the upper-triangular off-diagonal elements of projected slide covariance ($32,640\text{D}$) $\to$ Class-balanced Dual Ridge ($\lambda=1.0$).
2. **CT (Cell-Type Abundance)**: Samples $1/8$ slide tokens, projects to 32D PCA, clusters into 256 tokens via seeded k-means++, computes soft abundance $\to$ Dual Ridge ($\lambda=1.0$).
3. **BM (Bag-Mean)**: Projects 1st-moment slide mean ($\bar{x}_i$) to the top 32 PCA dimensions $\to$ Dual Ridge ($\lambda=1.0$).
4. **BD (Bag-Dispersion)**: Normalizes eigenvalues of projected slide covariance to compute spectral entropy, extracting bounded ordered-typicality evidence ($\kappa=1.0$).
5. **QA (Quantiles & Extremum Evidence)**: Extracts non-Gaussian 4-quantiles $[0.05, 0.10, 0.90, 0.95]$ across top 32 PCA dimensions ($128\text{D}$) $\to$ Dual Ridge ($\lambda=1.0$).
6. **DS (Denoised Salience Bag-Mean)**: Reweights slide mean by token class-salience to suppress uninformative stroma tokens ($32\text{D}$) $\to$ Dual Ridge ($\lambda=1.0$).

---


## 3. Core Invariants & Constraints

1. **Zero Data Leakage (No-Leakage Contract)**:
   - Query slides must **never** participate in within-slide PCA basis construction, token dictionary generation, or normalization statistics.
2. **Exact Label Antisymmetry**:
   - Swapping labels ($y \to 1-y$) must produce exact sign reversal in branch margins and complement probabilities ($P(1-y) = 1 - P(y)$). Checked by regression suite.
3. **Deterministic Evaluation Protocol**:
   - $t$-statistics, $p$-values, and confidence intervals are strictly prohibited for model promotion decisions ($\text{seed std} = 0.00000$).
   - **비교 기준 (Comparison Basis, §214-V 이후)**: 모든 승격 비교는 **CT 제외 5-branch 구성**(`CV, BM, BD, QA, DS` / `ICF_FIXED_HEAD_CT_WEIGHT=0.0`)에서 수행한다.
     - 공식 기준선: **v121 5-branch Trimmed Mean, Primary 7 Macro `0.6171`** (`v121_baseline` 태그, 50-fold).
     - v120 6-branch 수치(`0.6265`)와 직접 비교 금지 — 브랜치 집합이 다르므로 비교 불가능하다.
   - 승격은 **Primary 7 sign agreement $\ge 5/7$**를 요구한다. Macro AUROC 상승만으로는 승격 근거가 되지 않는다.
   - **분해능 하한 (Resolution Floor)**: Primary 7 macro 변화량이 약 $\pm 1.0\%p$ 미만인 후보는 지금까지 예외 없이 sign agreement 3~4/7에 머물렀다 (§214-V에서 집계 방식 8종 및 브랜치 부분집합 31종 전수 확인). **1%p 미만의 macro 상승은 이 벤치마크로 판별 불가능한 잡음으로 간주하며, 승리로 기록하지 않는다.**
   - **SEAL 10-task hold-out은 사용자 결정에 따라 현재 유보(DEFERRED) 상태다 (§214-V).** hold-out 없이 내린 판정은 `current_status.md`와 `archive.md` 양쪽에 반드시 `hold-out 미검증`으로 명시한다.
4. **Closed Axes (Do NOT Retry)**:
   - Synthetic distribution alignment (§129), DD branch revival (§147, §195), CT cell count scaling (§159), Non-linear KRR (§199), LR direct patch likelihood (§200), Fisher subspace (§202), **In-Episode Context LOO 가중치 (§212, $\rho = -0.27$)**, **집계 함수 변형 탐색 (§214-V — 집계 8종·부분집합 31종 전수에서 5/7 도달 0건; 분해능 하한 아래)**.
5. **보고 무결성 계약 (Reporting Integrity Contract)** — §214에서 실제 위반이 확인되어 §214-V에서 신설:
   - **회귀 전량 명시 의무**: 벤치마크 결과 보고 시 **성능이 하락한 과제를 생략할 수 없다.** 상승 과제만 나열한 요약은 금지한다. §214는 `adaptive_trimmed`의 상승 3개만 적고 하락 3개(ARID1A −2.00%p, PBRM1 −2.22%p, Grade −0.41%p)를 누락했다.
   - **Sign agreement 병기 의무**: macro AUROC를 제시할 때는 반드시 `n/7` sign agreement를 같은 줄에 병기한다.
   - **측정 방식 정확 기술**: 저장된 브랜치 마진의 **오프라인 재집계**를 "전수 실측"으로 표기하지 않는다. 신규 파이프라인 실행(`logs/`·`predictions/`에 신규 산출물 발생)과 재집계를 용어로 구분한다.
   - **비교 모집단 한정**: "1위", "최고", "최고치 경신" 등의 표현에는 비교 대상 집합을 괄호로 한정한다 (예: "비교한 집계 방식 8종 중 1위").
   - **승리 수식어 금지**: "폭등", "초대형 성과", "사상 최고", 이모지 강조 등을 사용하지 않는다. 수치와 부호로만 기술한다.
   - **미검증 항목 표기 의무**: 수행하지 않은 검증(SEAL hold-out 등)은 침묵하지 않고 `미검증`으로 명시한다.
   - **자기 검증 우선**: 결과를 문서화하기 전에, 저장된 예측 파일로부터 독립적으로 재계산하여 수치를 대조한다.

---

## 4. Environment Prerequisites

- **Host Hardware**: Linux node (e.g. `gnode3`, 5x NVIDIA RTX A5000 24GB).
- **Python Environment**: Managed strictly with `uv` in `ICF/.venv` (Python 3.12.11).
  - Activate/source: Handled automatically by `source scripts/node_env.sh`.
  - Install/rebuild: `uv venv --python 3.12 .venv && uv pip install -r requirements.txt`
- **Core Dependencies**: PyTorch 2.14.0+cu130, Lightning 2.6.5, NumPy 2.4.4, SciPy 1.18.1, Pandas 3.0.3, Torcheval 0.0.7.
- **Threading Cap**: CPU BLAS operations must cap threads (`OMP_NUM_THREADS=8`) to prevent severe OpenMP oversubscription on many-core nodes.

---

## 5. Standard Verification Commands

```bash
# 1. Environment & Python check
source scripts/node_env.sh && echo "$PYTHON / NGPU=$NGPU"

# 2. Fast Regression Suite (16 modules / 119 tests, ~20s)
bash scripts/run_tests.sh

# 3. Single Test Module
bash scripts/run_tests.sh test_bd_branch.py

# 4. Primary 7-Task Benchmark (v121 CT-excluded 5-branch = 공식 비교 기준)
bash scripts/eval_v121.sh <gpu_id> <tag>

# 5. 브랜치 진단 / 집계·부분집합 절제 (저장 마진 오프라인 재집계, GPU 불필요)
$PYTHON scripts/analysis/branch_diagnostics.py --tag v121_baseline

# 6. SEAL 10-Task Multi-GPU Hold-out Evaluation (현재 유보 상태)
bash scripts/run_v120_seal_multi_gpu.sh <tag>
```

_by Antigravity on gnode3 at 2026-09-03 10:40:00_
