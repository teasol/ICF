# Current Architecture Specification (v114 + BM Branch)

**Last updated**: `2026-08-21`

---

## 1. 아키텍처 개요 및 설계 철학

ICF(In-Context Foundation) 모델의 활성 베이스라인(v114)은 **학습 파라미터가 0개(0-parameter)인 완전 결정론적(Deterministic, seed std = 0.00000) 인컨텍스트 분류기**다.

- **원리**: 슬라이드(Bag) 단위의 다중 인스턴스(MIL) 병리 이미지 임베딩($X_i \in \mathbb{R}^{N_i \times 1536}$)에서, 레이블이 제공된 Context 슬라이드들만으로 Within-slide PCA 기저를 구축하고 4개 상보적 통계 브랜치(CV, DD, CT, BM)를 추출하여 닫힌 형태(Closed-form)의 Class-balanced Ridge 회귀로 마진을 산출한다.
- **불변식**: Query 슬라이드는 기저 생성, 토큰 군집화, 통계 표준화에 일절 참여하지 않으며(No-Leakage), 라벨 반전($y \to 1-y$) 시 마진 부호가 정확히 반전된다(Label Antisymmetry).

```
[Context Bags (Labeled) + Query Bags (Unlabeled)]
                      │
                      ▼
   Within-Slide PCA (Context-Only Centered, K=256)
                      │
     ┌────────────────┼────────────────┬────────────────┐
     │                │                │                │
     ▼                ▼                ▼                ▼
[CV Branch]      [DD Branch]      [CT Branch]      [BM Branch]
2차 공분산       1차원 분산       32D PCA 의사세포형  1차 모멘트 평균
비대각 32,640D   Typicality       256 토큰 조성비    32D 기저 사영
     │                │                │                │
     ▼                ▼                ▼                ▼
Dual Ridge (λ=1) Bounded Margin   Dual Ridge (λ=1) Dual Ridge (λ=1)
   (M_CV)           (M_DD)           (M_CT)           (M_BM)
     │                │                │                │
     └────────────────┼────────────────┴────────────────┘
                      ▼
  Total Margin = 1.0·M_CV - 1.0·M_DD + 1.0·M_CT + w_bm·M_BM
                      │
                      ▼
          Logits = (-M/2, +M/2)  -->  P(y=1) = sigmoid(M)
```

---

## 2. 4대 브랜치 상세 작동 원리 및 수식

### 2.1. 기저 구축 (Within-Slide PCA Basis)
- **입력**: Context 슬라이드 집합 $\{X_i\}_{i=1}^{n_{ctx}}$, 각 $X_i \in \mathbb{R}^{N_i \times 1536}$.
- **슬라이드 내 중심화(Within-slide Centering)**:
  $$C_{within} = \frac{1}{\sum N_i} \sum_{i=1}^{n_{ctx}} \sum_{j=1}^{N_i} (x_{ij} - \bar{x}_i)(x_{ij} - \bar{x}_i)^T$$
  *(슬라이드 간 배경/염색 편차인 between-slide nuisance를 소거)*
- **기저 $B$**: $C_{within}$의 상위 $K=256$개 고유벡터 $B \in \mathbb{R}^{1536 \times 256}$.

---

### 2.2. CV (Cross-Covariance / 2차 상관관계 브랜치)
- **설계 목적**: 슬라이드 내 세포 특징 간의 2차 상관 구조 포착.
- **디스크립터**: 각 슬라이드의 기저 사영 공분산 $S_i = B^T C_i B \in \mathbb{R}^{256 \times 256}$의 **상삼각 비대각(Off-diagonal) 요소 32,640차원**:
  $$\text{desc}_{CV}(i) = \text{triu\_offdiag}(S_i) \in \mathbb{R}^{32,640}$$
  *(1,536차원 Raw mean과 대각 256차원은 노이즈 제거를 위해 제외)*
- **분류기**: Context 디스크립터에 대한 Context-only 표준화 및 Class-balanced Dual Ridge ($\lambda=1.0$):
  $$M_{CV} = \text{logit}_1 - \text{logit}_0$$

---

### 2.3. DD (Directional Dispersion / 1-D 분산 Typicality 브랜치)
- **설계 목적**: 두 클래스 간 공분산 차이가 가장 극명한 단일 분산 축 상의 유계 증거 추출.
- **방향 벡터 $u$**: 축소 추정된 공분산 차이 행렬($\bar{S}_1 - \bar{S}_0$)의 최대 고유값에 해당하는 1차원 벡터 $u \in \mathbb{R}^{256}$.
- **투영 및 표준화**: $q_i = \log(u^T S_i u + \epsilon) \to$ Context 기준 표준화 $q_i$.
- **수식 (Bounded Ordered-Typicality Evidence, $\kappa=1$)**:
  $$m = \frac{p_0 + p_1}{2}, \quad h = \frac{|p_1 - p_0|}{2}, \quad s = \operatorname{sign}(p_1 - p_0)$$
  $$\sigma_{pool} = \sqrt{\frac{\sigma_0^2 + \sigma_1^2}{2}}, \quad h_{eff} = \max(h, \kappa \sigma_{pool})$$
  $$a(q) = \operatorname{clip}\left(s \frac{q - m}{h_{eff}}, -1, 1\right), \quad o(q) = \exp\left[-\frac{1}{2} \min_c \frac{(q - p_c)^2}{\sigma_c^2 + \epsilon}\right]$$
  $$M_{DD}(q) = a(q) \cdot o(q) \in [-1, 1]$$

---

### 2.4. CT (Cell-Type Abundance / 의사 세포형 조성비 브랜치)
- **설계 목적**: 슬라이드 내 주요 세포 아형(Subpopulations)의 상대적 빈도/조성비 비교.
- **작동 과정**:
  1. **샘플링**: 슬라이드 크기의 $1/8$ fraction (floor 64, seeded random seed 0).
  2. **저차원 사영**: $B$의 상위 **32 PCA 차원**으로 사영 후 Context 표준화.
  3. **토큰 사전 생성**: Context 세포들 위에서 **Seeded k-means++ (Lloyd $\le 8$, tol 1e-4)로 256개 토큰** 생성.
  4. **소프트 할당**: $a_{ik} = \frac{1}{N_i} \sum_j \operatorname{softmax}_k (-\|z_{ij} - t_k\|^2 / \tau)$ ($\tau=0.5$).
  5. **분류기**: 256차원 abundance 벡터에 대해 Class-balanced Dual Ridge ($\lambda=1.0$) 적용 $\to M_{CT} = \text{logit}_1 - \text{logit}_0$.

---

### 2.5. BM (Projected Bag-Mean / 1차 모멘트 브랜치 - Plan A)
- **설계 목적**: 2차 공분산/어번던스가 놓치는 슬라이드 전반의 세포 평균(1차 모멘트) 발현 신호 포착.
- **작동 과정**:
  1. 슬라이드 평균 $\bar{x}_i = \frac{1}{N_i} \sum_j x_{ij} \in \mathbb{R}^{1536}$ 계산.
  2. 기저 사영: $\mu_i = \bar{x}_i B_{:32} \in \mathbb{R}^{32}$ (노이즈 억제를 위해 상위 32차원만 사용).
  3. Context-only 표준화 및 Class-balanced Dual Ridge ($\lambda=1.0$) $\to M_{BM} = \text{logit}_1 - \text{logit}_0$.
- **가중치**: 기본값 $w_{BM} = 0.0$ (v114 100% 호환). 활성화 시 $w_{BM} > 0$.

---

## 3. Head 마진 결합

최종 마진 $M$은 4개 브랜치의 가중합으로 계산된다:
$$M = w_{CV} M_{CV} - w_{DD} M_{DD} + w_{CT} M_{CT} + w_{BM} M_{BM}$$

- **v114 기본 가중치**: $w_{CV} = 1.0, \quad w_{DD} = 1.0, \quad w_{CT} = 1.0, \quad w_{BM} = 0.0$
- **출력 로짓 및 확률**:
  $$\text{logits} = \left(-\frac{M}{2}, +\frac{M}{2}\right), \quad P(y=1) = \sigma(M)$$

---

## 4. 코드베이스 모듈 구조 및 개발자 가이드

### 4.1. 계층별 패키지 구성
```
ICF/
├── src/
│   ├── models/
│   │   ├── base.py                # InContextClassifierProtocol & BaseInContextClassifier
│   │   ├── registry.py            # @register_model 데코레이터 및 build_model 팩토리
│   │   ├── training_free.py       # v114 활성 baseline (TrainingFreeClassifier)
│   │   ├── ct/                    # CT Readout 서브패키지 (config, tokenizers, abundance, readout)
│   │   ├── ct_readout.py          # src.models.ct Re-export Facade (100% 하위 호환)
│   │   └── dd_adaptive_rank.py    # DD ordered-typicality margin
│   ├── datasets/
│   │   ├── synthetic/             # 합성 데이터 서브패키지 (types, generator, dataset)
│   │   └── synthetic_data.py      # src.datasets.synthetic Facade
│   └── modules/                   # Lightning 학습 인터페이스 (losses, diagnostics, guards)
├── scripts/
│   ├── node_env.sh                # 실행 노드별 Conda / Python 경로 자동 탐색 SSOT
│   ├── eval_v114.sh               # v114 활성 baseline 평가 스크립트
│   ├── eval_seal_tasks.sh         # SEAL 10 tasks 평가 러너
│   └── diagnostics/               # diagnose_*.py (14개 분석/진단 스크립트)
└── tests/                         # 핵심 회귀 테스트 스위트 (92개 핵심 계약 검증, ~13s)
```

### 4.2. 실행 가이드
```bash
# 1. 환경 로드
. scripts/node_env.sh

# 2. 회귀 테스트 실행
$PYTHON -m unittest discover -s tests -p "test_*.py"

# 3. v114 Baseline 평가
bash scripts/eval_v114.sh <gpu_id> <tag> [tasks...]
```

---

## 5. 닫힌 축 (Closed Axes — 재시도 금지)

| 축 | 기각 근거 |
| :--- | :--- |
| **합성 데이터 분포 축** | 실제 에피소드와의 분포 격차를 닫으려고 할수록 성능이 단조 하락함 (`docs/current_status.md` §129). |
| **DD 전반 축** | $K > 128, r > 1, |t|$ 게이트/셀렉터 모두 실패. 현 1-D typicality로 고정 (§147). |
| **CT Cell 수 단순 증대** | Bag당 64개 이상의 full-cell을 써도 성능 향상 없음 (§159). |
| **CT Kernel-Ridge** | RBF/Poly 비선형 커널 모두 8/10 task 하락 $\to$ 비선형 곡률 부재 확인 (§188). |
| **CT Top-k 풀링** | Mean 풀링 대비 노이즈만 가중되어 기각 (§189). |

---

## 6. 열려 있는 연구 방향

1. **[Plan A] BM Branch 가중치 및 사영 차원 스윕**: 1차 모멘트 결합 효과 검증 (`current_experiments.md` 참조).
2. **[Plan B] TH (Tumor Heterogeneity) 다양성 스칼라**: 슬라이드 내 세포 다양성/엔트로피 1D 증거 주입.
3. **v114의 홀드아웃 7 및 전체 17 실측**: SEAL 10 외 독립 집단에서의 재현성 확보.

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-21 15:16:00_
