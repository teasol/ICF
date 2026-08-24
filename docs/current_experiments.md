# Current Experiments & Active Queue (2026-08-23)

**Last updated**: `2026-08-23 09:30:00`

> [!IMPORTANT]
> **활성 baseline = v120** (`docs/current_status.md` §198, §203). 학습 파라미터 0, 완전 결정론적(Deterministic).
> **아키텍처**: 6-Branch (CV + CT + BM + BD + QA + DS, $w_{DD}=0.0$) + Trimmed Mean Voting
> **실측치**: Primary 7 **`0.6265`**, SEAL 10 **`0.6972`**, Total 17 **`0.6681`** (역대 전 계보 통합 최고치)
> ⚠️ **판정 프로토콜**: **Primary Benchmark (7 tasks)** 가 주 판정 기준이며, **Hold-out Validation (SEAL 10 tasks)** 는 독립 교차 검증용이다.
> 실행: `bash scripts/eval_v120.sh <gpu> <tag> [tasks...]` (기본: Primary 7 tasks)
> SEAL 10-Task 실행: `bash scripts/run_v120_seal_multi_gpu.sh <tag>`

---

## 1. 최근 종료된 실험 요약

| 실험 | 가설 및 설정 | 결과 | 판정 |
| :--- | :--- | :--- | :--- |
| **§198 v120 공식 승격** | 6-Branch (CV + CT + BM + BD + QA + DS) + Trimmed Mean | Primary 7 **`0.6265`**, SEAL 10 **`0.6972`**, All 17 **`0.6681`** | **v120 Baseline 확립** 🏆 |
| **§199 Non-Linear KRR 전면 도입** | RBF / Cosine 커널 기반 비선형 Kernel Ridge Regression | Primary 7 **0.6125** (-0.0140, Few-shot 과적합으로 SMAD4 0.4290 붕괴) | **기각 (Linear 유지)** |
| **§200 LR (Direct Likelihood Ratio)** | Context 패치 메모리 뱅크 직접 우도비 + Top-K MIL Extreme Pooling | 단독 **0.5874**, 7-Branch **0.6195** (원천 패치 간 염색 Confounder 간섭) | **기각** |
| **§201 절사 집계 방식 스윕** | Drop Min Only, Drop 2 Furthest vs Trimmed Mean 17개 전수 비교 | Drop Min(0.6247, 대칭성 파괴), Drop 2 Furthest(0.6576, 공분산 신호 소거) | **Trimmed Mean 최적성 재확인** |
| **§202 In-Context Fisher Subspace** | Supervised Contrastive Basis: $\mathbf{w}_{\text{Fisher}} = \Sigma_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)$ | Primary 7 **0.5709** (-0.0556 폭락, $N_{ctx}=40 \ll D=1536$ 차원의 저주) | **기각 (Within-PCA 유지)** |
| **§203 DE & SW 신규 브랜치 개발** | In-Subspace Dual Extreme (DE) 및 Sliced Wasserstein (SW) 개발 | DE 단독 **0.5954**, SW 단독 **0.5976** (KRAS 단독 1위 **`0.7374`**) | **독립 챔피언 확인** |
| **§204 단독 브랜치 전수 실측 & CT 비교** | 8개 단독 브랜치 50-Fold 전수 실측 및 CT 단독 vs v120 앙상블 비교 | CT 단독 Primary 7 **`0.6147`** (단독 1위), SEAL 10 **`0.7197`**, Musk **`~0.90`** | **CT 올라운더 & v120 사각지대 방어 확증** |

---

## 2. 활성 연구 큐 (Active Research Queue)

### [Exp 1] CT (Cell Tokenizer) 단독 브랜치 고도화 연구
- **가설**: K256 계층적 트리 토크나이저의 다중 스케일화, `cattopk` (Mean + Top-K) 풀링, 토큰별 국소 분산 모멘트 결합을 통해 `CT` 단독 성능을 극대화하여 `SMAD4` 사각지대 해소 및 단일 모델 완성도 제고.

---

## 3. 실험 프로토콜 및 산출물 규칙

1. **평가 스크립트 실행 전**:
   - 반드시 `. scripts/node_env.sh` 소싱 확인.
   - 환경변수 및 오버라이드 플래그 명시.
2. **결과 보고**:
   - `docs/current_status.md` §0-2 양식 준수 (가설 설명 $\to$ Primary 7-task 및 Hold-out 10-task 전체별 $\Delta$ 표 $\to$ Macro $\Delta$ 및 부호 일치 수).
   - 신규/수정 단락 작성 직후 스탬프 작성:
     `_by <LLM Name> on <server name> at <YYYY-MM-DD HH:MM:SS>_`

_by Antigravity on gnode3 at 2026-08-24 18:16:00_



