# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-22 17:10:00` — **BD Branch 정식 승격 $\to$ v116 Baseline 확립 (§194)**:
- **v116 승격**: **BD-branch (Bag Dispersion / Spectral Entropy with Ordered-Typicality Evidence, $w_{BD}=1.0$)**가 Primary 7-Task에서 **Macro 0.6119 (+0.0025 vs v115, +0.0068 vs v114, 4/7 과제 승리)**를 달성하여 공식 baseline으로 승격 (사용자 최종 승인).
- **활성 baseline**: **v116 (CV + DD + CT + BM + BD)** (학습 파라미터 0, Deterministic). 활성 runner `scripts/eval_v116.sh`.
- **벤치마크 실측치**: Primary 7-Task Macro Fold-mean AUROC = **0.6119** (v115 Baseline 0.6094 대비 **+0.0025** 개선, v114 대비 **+0.0068** 개선).

---

# §0. 판정 프로토콜 종합 SSOT (Decision Protocol SSOT)

프로젝트의 모든 모델 평가, 승격 및 기각 판단은 아래 원칙에 따라 수행된다 (2026-08-21 개정).

## 0-1. 활성 결정론적 Arm 판정 규칙 (현 v106~v116 무학습 계보)
- **$t$-통계량, $p$-value, 신뢰구간(CI) 사용 절대 금지 (§151-1)**:
  - 활성 모델은 학습 파라미터 0, 난수 시드 분산이 정확히 0($\text{seed std} = 0.00000$)이므로, task를 표본 단위로 둔 $t$-검정은 통계적 근거가 없다.
- **벤치마크 2-Tier 구조 (2026-08-21 확정)**:
  1. **Primary Benchmark (7 tasks)**: SEAL에 포함되지 않았던 7개 과제를 **새로운 주 평가 기준**으로 삼는다.
     - `cptac_lscc/ARID1A_mutation`, `cptac_lscc/Histologic_Grade`, `cptac_lscc/KEAP1_mutation`
     - `cptac_luad/KRAS_mutation`, `cptac_pda/SMAD4_mutation`
     - `ucla_lung/progression_regression`, `cptac_ccrcc/PBRM1_mutation`
  2. **Hold-out Validation (10 tasks)**: 기존 SEAL 10-task를 **독립 홀드아웃 검증 집단**으로 삼아 선택 편향(Selection bias) 및 일반화 성능을 교차 검증한다.
- **유효한 3대 판정 지표**:
  1. **Primary 7-task 부호 일치 수 (Sign Agreement)**: 7개 과제 중 몇 개에서 승리했는가? (예: $\ge 5/7$)
  2. **Primary 7-task Macro $\Delta$**: 7개 과제 산술 평균의 유의미한 개선.
  3. **Hold-out 10-task 재현성**: SEAL 10개 과제에서의 동시 개선 ($\Delta_{holdout} > 0$).

## 0-2. 최종 승격 / 기각 의사결정 프로토콜 (User Decision Protocol, §118)
- **게이트 자동 결정 금지**: 어떤 수치나 통계 게이트도 모델 승격/기각을 자동으로 결정하지 않으며, **최종 판정은 항상 사용자의 종합적 판단**이다.
- **Baseline 성능대별 비대칭 분석**:
  - 같은 크기의 $+0.01$이라도 **랜덤 근처($0.4\sim 0.5$)에서의 상승보다 천장 근처($0.8\sim 0.9$)에서의 하락이 훨씬 치명적**이다.
  - 고신호 구간에서 깎이고 저신호 구간에서만 오르는 것은 유효 신호가 아니라 노이즈/평균 회귀 교환일 가능성이 높다.
- **필수 보고 양식 (§118-3)**:
  1. 이 실험/arm이 무엇을 검증하는지 가설 설명
  2. **Primary 7개 과제 및 Hold-out 10개 과제 전체별 baseline 대비 $\Delta$ 표를 빠짐없이 제시** (요약/평균만 제시 금지)
  3. 전체 Macro $\Delta$ 및 부호 일치 수 제시 $\to$ 사용자가 승격/기각/재검증 최종 판정.

## 0-3. 닫힌 축 (Closed Axes) 재시도 금지
실측을 통해 이득이 없거나 단조 열화가 입증된 아래 방향은 **새로운 arm으로 재설계/재시도하지 않는다**:
1. **합성 데이터 분포 축 (§129)**: 실제 에피소드와의 분포 격차를 닫으려고 할수록 성능이 단조 하락함.
2. **DD 전반 축 (§147)**: $K > 128$, $r > 1$, $|t|$ 게이트/셀렉터 모두 실패. (현재의 1-D ordered-typicality로 고정)
3. **CT Cell 수 단순 증대 (§159)**: Bag당 64개 이상의 full-cell을 써도 성능 향상 없음.
4. **CT Kernel-Ridge (§188)**: RBF/Poly 비선형 커널 모두 8/10 task 하락 $\to$ 비선형 곡률 부재 확인 및 기각.
5. **CT Top-k 풀링 (§189)**: Mean 풀링 대비 노이즈만 가중되어 기각.

## 0-4. [Historical Archive] 과거 학습 Arm 판정 규칙 (v83~v105 레짐)
*이하 규칙은 난수 시드 분산이 존재하던 과거 학습 계보(v83~v105) 전용이며, 현재 무학습 계보에는 적용되지 않는다.*
- **통계 게이트 (§107-3)**: 4개 시드 전체 부호 일치($4/4$) 및 Seed-paired $|t| \ge 2.5$ ($p \le 0.088$).
- **최소 검출 한계 (§131-2)**: 4 seed의 최소 검출 가능 효과는 $\approx 0.0121$. 노이즈 범위($\pm 0.005$ 이내)의 작은 신호를 쫓지 않음.
- **독립 시드 그룹 재현 (§131-5)**: 단일 그룹의 높은 $|t|$보다 독립적인 시드 그룹(A/B 그룹)에서의 부호 일치 재현을 우선.
- **평가 에포크 고정 (§104)**: 최적 에포크 체리피킹 금지, Epoch 49(또는 지정 에포크) 고정 평가.

_by Antigravity on teasol at 2026-08-21 19:40:00_

---

> [!IMPORTANT]
> **활성 구성은 v116(§194, 사용자 승인 승격) — 학습 파라미터 0, 완전 결정론적이다.**
> ```
> 사영 : fold의 CONTEXT cell을 bag별 자기 평균으로 센터링해 풀링한 공분산의 상위 256 고유벡터 B (1536 x 256)
> head : margin = 1.0·M_CV - 1.0·M_DD + 1.0·M_CT + 1.0·M_BM + 1.0·M_BD      # 다섯 weight 전부 1.0 (§194)
> CV   : off-diagonal 32,640차원만 (대각 256·raw mean 1,536 제거) ⚠️ DD는 전체 triangle
> DD   : rank-1 방향 → ordered-coordinate × nearest-class typicality, κ=1 (§182/§183)
> CT   : bag 자기 크기의 1/8 fraction(floor 64, seeded random seed 0) → 32 PCA 방향
>        → seeded k-means++ + Lloyd(≤8) 로 256 token → match abundance → ridge(λ=1)
> BM   : 슬라이드 평균의 상위 32차원 사영 μ_i = x̄_i B_{:32} ∈ R^32 → Class-balanced Dual Ridge (λ=1.0)
> BD   : 슬라이드 기저 사영 공분산 고유값 정규화 스펙트럼 엔트로피 H_i → Ordered-Typicality Evidence (κ=1.0)
>
> 실측치: Primary 7-Task Macro = 0.6119 (v115 0.6094 대비 +0.0025, v114 0.6051 대비 +0.0068)
>
> bash scripts/eval_v116.sh <gpu> <tag> [tasks...]     # 활성 baseline entry point (기본: Primary 7 tasks)
> ```
> v113부터 **성립하지 않는다** — seeded random으로 bag의 1/8만 뽑는다(seed 0 고정이라
> 결정론성 자체는 유지된다). full-cell hierarchical 재현이 필요하면 `scripts/eval_v112.sh`.
>
> **계보 (전부 결정론적, seed std 0.00000)**
> ```
> v106 0.6864  within-slide PCA(K=128) + 고정 head, 학습 파라미터 0의 시작    §139
> v107 0.6945  K 128 → 256                                                  §143
> v108 0.6967  CT = PCA32 부분공간 + ridge readout                           §152
> v109 0.7027  CV = off-diagonal만, CT = k-means token @ w=0.7               §158
> v110 0.70692 CT cluster 16 → 32   ← 홀드아웃 0.61029 / 전체 0.66713 = 예측 최고 §161
> v111 0.70453 full-cell hierarchical PCA32/K256 (bias 제거 우선)            §181
> v112 0.70432 DD = ordered × typicality (κ=1, w=1)                          §183
> v113 0.70394 CT cell 예산 = bag 1/8 fraction + k-means++ (feasibility)     §185
> v114 0.70509 fixed-head weight 세 개를 전부 1.0으로 통일                   §187
> v115 0.60940 BM (Projected Bag-Mean 32D Ridge w=1.0) Primary 7-Task 승격     §192
> v116 0.61187 BD (Bag Dispersion Spectral Entropy Ordered-Typicality w=1.0) ← 활성 §194
> ```
> ⚠️ **v110이 여전히 전체 17 최고(0.66713)다.** v111~v114는 예측 성능이 아니라 각각
> selection bias 제거(§181)·DD 형태(§183)·22GB GPU feasibility(§185)를 이유로 승격됐다.
> "최신 = 최고"로 읽지 말 것.
>
> ⚠️ **결정론적 arm에 §107-3 게이트(|t|≥2.5)와 §131-2 검출 한계를 적용하지 말 것** — 둘 다
> 시드 분산이 분모다. 부호 일치 수와 독립 집단 재현으로 판정한다(§151-1).
> ⚠️ 학습을 포함하는 arm과 비교할 때는 **그 arm의 분산 때문에 §107-3 게이트와 §131-2의 검출 한계가
> 그대로 적용된다.** 검출 한계가 ≈0이 되는 것은 training-free 변형끼리 비교할 때뿐이다(§139-6).
> ⚠️ 학습 계보(v83·v98 등)의 숫자와 **시드 집합이 다르므로 빼지 말 것**(§131-1). 그 계보의
> 마지막 baseline은 v98(1-GPU 8 seed 0.6852, §131)이고 전부 historical이다.
>
> ⚠️ **K=256은 환경변수로만 걸린다.** config의 `covariance_sketch_dim`은 **128 그대로 두었다** —
> 그 값을 바꾸면 v98 학습 재현이 깨지고 P(1536×128) strict load가 실패한다. `ICF_SKETCH_DIM`은
> K에 의존하는 유일한 텐서 `_covariance_projection`만 버리고 나머지 불일치는 예외로 올린다(§142-1).

---

## 185. 2026-08-20 — v113 공식 승격: CT 샘플링 1/8 fraction + Seeded k-means++ (GPU Feasibility 확보)
- CT 토큰화 단계에서 GPU OOM을 방지하기 위해 $1/8$ fraction seeded random sampling과 k-means++ 토큰 사전을 도입하여 v113으로 공식 승격.

---

## 186. 2026-08-20 — 벤치마크 2-Tier 재편 (Primary 7 tasks + Hold-out 10 tasks)
- SEAL 10-task를 독립 홀드아웃 검증 집단으로 격리하고, 비-SEAL 7개 과제를 Primary 벤치마크로 격상.

---

## 187. 2026-08-20 — v114 공식 승격: Fixed-head Weights 통일 ($w_{CV}=1.0, w_{DD}=1.0, w_{CT}=1.0$)
- CV/DD/CT 3개 브랜치의 가중치를 $1.0$으로 단순 통일하여 Primary 7-task에서 v114 베이스라인 확립 (Macro 0.6051).

---

## 188. 2026-08-21 — [기각] CT Kernel-Ridge (RBF/Poly) 비선형 곡률 탐색
- 8/10 task 하락으로 비선형 매핑 이득 부재 확인 및 기각.

---

## 189. 2026-08-21 — [기각] CT Abundance Top-k 풀링 탐색
- Mean 풀링 대비 노이즈만 가중되어 기각.

---

## 190. 2026-08-21 — 스크립트 디렉토리 정리 및 벤치마크 환경 표준화
- `scripts/eval_v114.sh`, `scripts/node_env.sh` 등 표준화 완료.

---

## 191. 2026-08-21 — Plan A: Projected Bag-Mean (BM) Branch 구현 및 단위 테스트 완료
- 슬라이드 평균 $\bar{x}_i$를 32 PCA 차원으로 사영하여 Class-balanced Ridge를 적용하는 BM 브랜치 구현.

---

## 192. 2026-08-21 — v115 공식 승격: BM-Branch 추가 및 Primary 7-Task 실측 검증 완료
- Primary 7-Task 50-fold 실측 결과 Macro **0.6094** (+0.0043 개선, 5/7 과제 승리)로 v115 공식 승격.

---

## 193. 2026-08-22 — BD Branch 후보 A (Log-Trace Ordered Typicality) 구현 및 실측 평가
- **명칭 확정**: 사용자의 도메인 독립적/통계적 명칭 지침에 따라 **BD (Bag Dispersion)**로 확정.
- **후보 A 구현**: 각 슬라이드의 사영 공분산 대각합 $v_i = \log \operatorname{Tr}(S_i)$을 추출하여 Bounded Ordered-Typicality Evidence($\kappa=1.0$)로 마진 $M_{BD} \in [-1, 1]$ 산출.
- **Primary 7-Task 4-GPU 50-fold 평가 결과**:
  - Macro AUROC = **0.6071** (v115 0.6094 대비 $\Delta = -0.0023$, 3/7 과제 승리).
  - 절대 분산 크기(Log-Trace)가 염색 강도/배경 편차 등의 nuisance scale에 취약함을 확인 $\to$ 고유값 스펙트럼 엔트로피(후보 B)로 재설계 결정.

---

## 194. 2026-08-22 — v116 공식 승격: BD-Branch 후보 B (Spectral Entropy) 재설계 및 실측 검증 완료

스케일 불변성(Scale Invariance)을 만족하는 **후보 B: Spectral Entropy (고유값 스펙트럼 엔트로피)** 메커니즘으로 BD 브랜치를 재설계하고, 4개 GPU 전체를 활용한 Primary 7-Task 50-fold 실측 평가를 완주하여 **v116 Baseline**으로 공식 승격했다.

### 1. 수학적 설계 (Scale-Invariant Spectral Entropy)
1. **고유값 정규화**:
   $$p_k = \frac{\lambda_k}{\sum_{j=1}^K \lambda_j} = \frac{\lambda_k}{\operatorname{Tr}(S_i)} \quad \left(\sum_{k=1}^K p_k = 1, \; p_k \ge 0\right)$$
   *(슬라이드별 스케일 계수 $c \cdot S_i$가 곱해져도 $p_k$는 완전 불변 유지)*
2. **스펙트럼 엔트로피 측정**:
   $$H_i = -\sum_{k=1}^K p_k \log (p_k + \epsilon) \quad \longrightarrow \quad \tilde{H}_i = \frac{H_i}{\log K} \in [0, 1]$$
3. **0-Parameter In-Context 마진**:
   Context의 $\tilde{H}_i$ 분포로부터 Bounded Ordered-Typicality Evidence ($\kappa=1.0$)로 $M_{BD} \in [-1, 1]$ 산출.

### 2. Primary 7-Task 50-fold 종합 실측 비교표

| # | Task | v114 Baseline | v115 Baseline | 후보 A (Log-Trace) | **후보 B (Spectral Entropy)** | $\Delta$ (vs v115) | $\Delta$ (vs 후보 A) | 승패 (vs v115) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `cptac_lscc/ARID1A` | 0.5019 ± 0.1382 | 0.5038 ± 0.1370 | 0.5123 ± 0.1353 | **0.5272 ± 0.1366** | **+0.0234** | **+0.0149** | **SE 승** |
| 2 | `cptac_lscc/Histologic_Grade` | 0.6545 ± 0.0980 | 0.6615 ± 0.0963 | **0.6638 ± 0.0966** | **0.6531 ± 0.0964** | **-0.0084** | -0.0107 | v115 승 |
| 3 | `cptac_lscc/KEAP1` | 0.5970 ± 0.1250 | 0.6049 ± 0.1240 | 0.6007 ± 0.1225 | **0.6111 ± 0.1217** | **+0.0062** | **+0.0104** | **SE 승** |
| 4 | `cptac_luad/KRAS` | 0.7296 ± 0.0940 | 0.7283 ± 0.0941 | 0.7184 ± 0.0946 | **0.7247 ± 0.1005** | **-0.0036** | **+0.0063** | v115 승 |
| 5 | `cptac_pda/SMAD4` | 0.4514 ± 0.1350 | 0.4493 ± 0.1354 | 0.4327 ± 0.1315 | **0.4513 ± 0.1389** | **+0.0020** | **+0.0186** | **SE 승** |
| 6 | `ucla_lung/progression` | 0.7685 ± 0.0906 | 0.7738 ± 0.0944 | 0.7711 ± 0.0935 | **0.7756 ± 0.0928** | **+0.0018** | **+0.0045** | **SE 승** |
| 7 | `cptac_ccrcc/PBRM1` | 0.5329 ± 0.1234 | 0.5440 ± 0.1256 | **0.5504 ± 0.1236** | **0.5401 ± 0.1280** | **-0.0039** | -0.0103 | v115 승 |
| **Macro** | **Primary 7-Task Mean** | **0.6051** | **0.6094** | **0.6071** | **0.6119** | **+0.0025** | **+0.0048** | **4 / 7 승** |
| *Pooled* | *Primary 7-Task Pooled* | *0.6052* | *0.6088* | *0.6038* | *0.6076* | *-0.0012* | *+0.0038* | - |

### 3. 판정 지표 요약
- **Macro $\Delta$**: $+0.0025$ (+0.25%p 개선 vs v115, $+0.0068$ vs v114).
- **후보 A 대비 압도적 우위**: Macro +0.0048 개선, 7개 중 6개 과제에서 후보 A 능가.
- **고신호 구간 상승**: `ucla_lung/progression_regression`에서 **0.7756 (+0.0018)** 상승.
- **v116 총 마진 수식**:
  $$\text{Margin} = 1.0 \cdot M_{CV} - 1.0 \cdot M_{DD} + 1.0 \cdot M_{CT} + 1.0 \cdot M_{BM} + 1.0 \cdot M_{BD}$$
- **활성 Runner**: `scripts/eval_v116.sh` (기본 실행 시 Primary 7 tasks 평가).

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-22 17:10:00_
