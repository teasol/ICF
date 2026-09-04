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
- 위 수치는 §214-V에서 `predictions/*_v121_baseline_official50_bf16.pt`로부터 독립 재계산하여 확인했다.

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
> 그 결과 Trimmed Mean은 독립 신호를 나르는 `CV`(슬라이드의 83.6%)와 `BD`(62.1%)를 우선적으로 절사하고, 중복 블록의 중앙값에 수렴한다.
>
> **[§217 브랜치 분류 체계]** 전 브랜치는 두 부류뿐이다.
> - **위치(Location) 계열**: `CV`, `BM`, `QA`, `DS` — 상호 상관 0.25~0.93. `CV`는 2차 통계임에도 이 계열에 묶인다(0.44~0.69). §217에서 시험한 `RM`(꼬리 부분공간 bag-mean)도 여기 합류해 기각되었다.
> - **형상(Shape) 계열**: `BD`, 그리고 §218에서 채택된 `SH`(차원별 왜도·첨도, 슬라이드 자체 평균·표준편차로 표준화 → 위치·척도 불변). `SH`의 최대 상관은 0.418이며 7개 과제 전부에서 단독 AUROC > 0.5다. `BS`(로그 총분산)는 직교성은 충족했으나 정보량 게이트에서 기각되었다.
>
> ⚠️ **[§218 정정] 직교성은 필요조건이지 충분조건이 아니다.** 랭크 효율을 45% → 50%로 올려도 Primary 7 macro는 개선되지 않았고 **Oracle 상한은 `0.6565`에서 전혀 움직이지 않았다.** §215·§216이 Oracle 격차를 랭크 결핍에 귀속시킨 것은 과잉 귀속이다. 랭크 효율은 성능의 **예측자가 아니라 필터**로만 쓴다.
>
> **[신규 브랜치 채택 규칙 — 필수]**
> **게이트 ① 직교성 (라벨 무관)** — 성능을 재기 **전에** `branch_screen.py`로 기존 브랜치와의 상관을 측정하고, **max |r| > 0.6이면 성능과 무관하게 기각**한다 (`SCREEN_ONLY=1`로 앙상블에 넣지 않은 채 마진만 기록해 측정할 것). 랭크 **효율**(eff.rank / 브랜치 수)이 기존 구성보다 낮아져도 기각한다.
> **게이트 ② 정보량 (§218 신설 → §220에서 개정)** — 게이트 ①을 통과한 후보는 아래 **둘 중 하나**를 만족해야 한다. 라벨을 쓰지만 **후보당 사전 선언된 통계**이므로, 앙상블 구성을 탐색해 최댓값을 고르는 §214식 절차와 구분된다.
> - **②a 범용 경로**: 단독 AUROC가 Primary 7 **전 과제**에서 0.5 초과.
> - **②b 과제 특화 경로 (§220 신설 → §224 개정)**: 최소 1개 과제에서 **50 fold 중 40개 이상이 단독 AUROC 0.5 초과**(이항검정 $p < 0.01$)일 것. 직교성은 게이트 ①이 이미 담당한다.
>   - **§224 개정 사유**: 원안의 "기존 최상 단독 브랜치를 능가" 조건은 브랜치 간 최댓값 비교(이른바 Oracle)였다. 이 비교는 **평가에 쓰는 데이터로 브랜치를 고르므로 승자의 저주로 상향 편향**되며(split-half 측정 결과 편향 +0.0063, 브랜치 성능이 근접한 ARID1A에서는 +0.0144), 애초에 상한도 아니다 — 앙상블이 최상 단독 브랜치를 넘는 과제가 실재한다(Prog: trimmed 0.7892 > 최상 0.7779). **판정 기준에서 브랜치 간 최댓값 비교를 배제한다.**
>   - 개정 후 기준은 **다른 브랜치를 전혀 참조하지 않으므로** 선택 편향이 없다. 기존 4개 판정을 그대로 재현함을 확인했다: SHJ 48/50 채택, SH 44/50 채택, **BS 31/50 기각**($p = 0.059$), RM은 게이트 ① 탈락(|r| = 0.690).
>
> ②b의 세 조건은 함께 요구된다. 선두 과제에서의 fold 수준 재현성과 직교성을 함께 요구하는 이유는, "한 과제에서 우연히 높은 값"이 특화로 위장하는 것을 막기 위해서다.
>
> **[개정 이력 — 반드시 유지]** ②b는 **§219 결과를 본 뒤인 2026-09-04에 사용자 지시로 신설**되었다. 결과를 본 뒤 기준을 완화하는 것은 §214의 실패 양식이므로, 개정의 정당화는 §219 결과와 **독립적인 논거**에 둔다: **앙상블의 목적은 과제별 강점의 결합이므로, 전 과제 유효성을 요구하는 것은 앙상블 구성원이 아니라 단독 모델에 대한 요건이다.** 기존 브랜치들 자체가 이 기준을 만족하지 못한다 — `CV`는 ARID1A `0.4308`, `BD`는 3개 과제, `QA`는 ARID1A `0.4692`로 우연 이하이며, 그럼에도 정식 브랜치다. **즉 ②a는 신규 후보에게만 기존 브랜치보다 엄격한 기준을 적용하는 이중잣대였다.** 이 논거는 §219 이전에도 성립했다.
> ②b로 통과한 후보는 archive에 `과제 특화 채택`으로 명시하고, 승격(≥5/7)은 별개 요건으로 남는다.
> 3. **평균의 새로운 사영·재가중은 시도하지 않는다.** BM(사영), QA(분위수), DS(살리언스 재가중), RM(꼬리 사영)이 모두 한 계열로 수렴했다 — 위치 축은 포화 상태다. 유효 랭크를 올릴 수 있는 방향은 **형상 계열의 확장**(슬라이드 내 이질성, 모드 수, 토큰 분포 기하 등 위치와 무관한 통계)뿐이다.
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
   - **평가 프로토콜 확정 (사용자 결정, §216):**
     1. **Fold는 PathoBench가 제공하는 공식 분할 그대로 고정한다.** fold 수 증대나 재분할로 분해능을 높이는 것은 **금지**한다.
     2. **SEAL 10-task는 최종 평가 전용 hold-out이다.** 모델 선택·하이퍼파라미터 결정에 **어떤 형태로도 사용 금지**. 현재 유보(DEFERRED) 상태이며, hold-out 없이 내린 판정은 `current_status.md`와 `archive.md` 양쪽에 `hold-out 미검증`으로 명시한다.
     3. **모델 선택은 Primary 7만으로 수행한다.**
   - **위 3항의 필연적 귀결 (§216)**: 분해능 하한은 완화 불가능한 **영구 제약**이다. Primary 7이 1%p 부근을 분해하지 못하는데 데이터를 늘릴 수도, hold-out을 볼 수도 없으므로, **후보 선택의 근거를 성능이 아닌 라벨 무관 구조 기준(유효 랭크, 브랜치 상관 등)에 두고 Primary 7은 사후 확인용으로만 쓴다.** 성능 순위로 후보를 고르는 절차는 §214에서 실패가 입증되었으므로 반복하지 않는다.
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
   - **Oracle 금지 (§224 신설)**: **브랜치 간 최댓값("과제별 최상 단독 브랜치")을 판정 기준·상한·미회수 성능으로 사용하지 않는다.** ① 평가 데이터로 선택하므로 상향 편향되고(split-half 편향 +0.0063), ② 상한이 아니다(앙상블이 이를 넘는 과제가 존재). 참고 수치로 표시할 수는 있으나 **어떤 채택·승격·연구 방향 판정의 근거로도 쓰지 않는다.**
   - **통제 비교 의무 (§223 신설)**: 어떤 기법의 **효과**를 주장하려면 **동일 실행·동일 basis·동일 fold에서 그 기법만 토글한 비교**여야 한다. 서로 다른 실행의 표를 나란히 놓고 차이를 효과로 해석하지 않는다. §213은 subsample DS를 v120 6-branch 앙상블과 비교해 "+7.65%p 폭등"으로 기록했으나 실제 효과는 +0.43%p였다(§223). 통제 비교는 `ICF_DS_SUBSAMPLE_COMPARE=1`처럼 두 arm을 한 실행에서 산출해 구현한다.

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

# 6. GPU 실행 대기 (에이전트 폴링 금지 -- 아래 참조)
bash scripts/run_and_wait.sh <runner.sh> <tag> "<분석 명령>"

# 7. SEAL 10-Task Multi-GPU Hold-out Evaluation (현재 유보 상태)
bash scripts/run_v120_seal_multi_gpu.sh <tag>
```

> ⚠️ **[에이전트 작업 규칙] GPU 실행을 폴링으로 기다리지 말 것.**
> 진행 상황을 반복 확인하면 확인 1회당 도구 호출 1회와 토큰이 소모된다 (§218에서 5시간 사용량 +7%p의 대부분이 여기서 발생).
> 대신 `scripts/run_and_wait.sh`를 **`run_in_background`로 단 한 번** 실행한다. 이 스크립트가 완주까지 블로킹한 뒤 완료 요약(과제별 fold-mean AUROC, 오염 검사, 분석 결과)을 한 번에 출력하므로, 완료 알림 자체에 답이 담긴다.
> 중간 확인이 꼭 필요하면 최소 10분 간격으로 제한한다.

```bash
# (참고) 위 6번의 실제 사용 예
bash scripts/run_and_wait.sh scripts/run_v121_shape_screen.sh v121_sh_v2 \
  "PYTHONPATH=$PWD $PYTHON scripts/analysis/branch_screen.py --tag v121_sh_v2 --candidate m_sh"
```

_by Antigravity on gnode3 at 2026-09-03 10:40:00_
