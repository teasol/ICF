# Agent Handoff Reference (Stable Architecture & Contracts)

This document serves as the permanent, stable architectural reference for the In-Context Foundation (ICF) project under the **Universal Handoff Protocol**.
For live in-flight tasks, current milestones, and immediate execution commands, refer to [`docs/current_status.md`](current_status.md).

---

## 1. Project Overview

The **In-Context Foundation (ICF)** project develops in-context classification models for pathology whole-slide images (WSI) using pre-extracted multiple-instance learning (MIL) patch embeddings (e.g. UNI2, 1536D).

### Active Baseline: v120 (0-Parameter Deterministic 6-Branch Trimmed Mean Voting)
The active baseline is **v120**, a completely training-free (0 learned parameters), fully deterministic classifier ($\text{seed std} = 0.00000$):
- **Core Principle**: Given labeled Context slides and unlabeled Query slides, a within-slide PCA basis ($K=256$) is constructed using only Context slides. Six complementary statistical branches extract slide representations, compute independent decision margins, convert margins to probabilities via sigmoid, and aggregate them using **Trimmed Mean Voting** (discarding the highest and lowest probability per slide, averaging the remaining 4).
- **Benchmark Performance**:
  - **Primary 7-Task Benchmark**: Macro Fold-mean AUROC **`0.6265`** (v119: 0.6247, v118: 0.6205).
  - **SEAL 10-Task Hold-out**: Macro Fold-mean AUROC **`0.6972`** (independent validation).
  - **All 17-Task Overall Macro**: **`0.6681`** (highest across all lineages).

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
- `configs/`: Active root contains only `train_v98_p1_reverse_1536_1gpu.yaml` as the checkpoint shell; historical configs reside in `configs/archive/`.
- `docs/`: Living documentation (`agent_handoff.md`, `current_status.md`, `current_architecture.md`, `current_experiments.md`) and historical archives (`docs/history/`).

### 2.2. Six Complementary Branches
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
   - Promotions are evaluated on **Primary 7-task sign agreement ($\ge 5/7$)** with **SEAL 10-task hold-out consistency**, subject to final user judgment.
4. **Closed Axes (Do NOT Retry)**:
   - Synthetic distribution alignment (§129), DD branch revival (§147, §195), CT cell count scaling (§159), Non-linear KRR (§199), LR direct patch likelihood (§200), Fisher subspace (§202).

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

# 4. Primary 7-Task Benchmark (v120 active baseline)
bash scripts/eval_v120.sh <gpu_id> <tag>

# 5. SEAL 10-Task Multi-GPU Hold-out Evaluation
bash scripts/run_v120_seal_multi_gpu.sh <tag>
```

_by Antigravity on gnode3 at 2026-09-03 10:40:00_
