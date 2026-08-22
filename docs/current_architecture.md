# Current Architecture Specification (v116 Active Baseline)

**Last updated**: `2026-08-22 17:10:00`

---

## 1. 아키텍처 개요 및 설계 철학

ICF(In-Context Foundation) 모델의 활성 베이스라인(v116)은 **학습 파라미터가 0개(0-parameter)인 완전 결정론적(Deterministic) 인컨텍스트 분류기**다.

- **원리**: 슬라이드(Bag) 단위의 다중 인스턴스(MIL) 병리 이미지 임베딩($X_i \in \mathbb{R}^{N_i \times 1536}$)에서, 레이블이 제공된 Context 슬라이드들만으로 Within-slide PCA 기저를 구축하고 5개 상보적 통계 브랜치(CV, DD, CT, BM, BD)를 추출하여 닫힌 형태(Closed-form)의 Class-balanced Ridge 회귀 및 Ordered-Typicality Evidence로 마진을 산출한다.
- **불변식**: Query 슬라이드는 기저 생성, 토큰 군집화, 통계 표준화에 일절 참여하지 않으며(No-Leakage), 라벨 반전($y \to 1-y$) 시 마진 부호가 정확히 반전된다(Label Antisymmetry).

```
[Context Bags (Labeled) + Query Bags (Unlabeled)]
                      │
                      ▼
   Within-Slide PCA (Context-Only Centered, K=256)
                      │
     ┌────────────────┼────────────────┬────────────────┬────────────────┐
     │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼
[CV Branch]      [DD Branch]      [CT Branch]      [BM Branch]      [BD Branch]
2차 공분산       1차원 분산       32D PCA 의사세포형 1차 모멘트       256D 스펙트럼
비대각 32,640D   Typicality       256 토큰 조성비    상위 32D 사영    엔트로피
     │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼
Dual Ridge (λ=1) Bounded Margin   Dual Ridge (λ=1) Dual Ridge (λ=1) Bounded Margin
   (M_CV)           (M_DD)           (M_CT)           (M_BM)           (M_BD)
     │                │                │                │                │
     └────────────────┼────────────────┴────────────────┴────────────────┘
                      ▼
  Total Margin = 1.0·M_CV - 1.0·M_DD + 1.0·M_CT + 1.0·M_BM + 1.0·M_BD
                      │
                      ▼
          Logits = (-M/2, +M/2)  -->  P(y=1) = sigmoid(M)
```

---

## 2. 5대 브랜치 상세 작동 원리 및 수식

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

### 2.5. BM (Bag-Mean / 1차 모멘트 저차원 사영 브랜치)
- **설계 목적**: 2차 공분산(CV/DD) 및 패치 군집(CT)이 담지 못하는 슬라이드 전체 1차 모멘트(세포 평균 $\bar{x}_i$)의 기준점(baseline shift) 정보 포착.
- **작동 과정**:
  1. **저차원 사영**: 각 슬라이드의 세포 평균 $\bar{x}_i \in \mathbb{R}^{1536}$을 Within-slide PCA 기저 $B$의 상위 **32차원**으로 사영:
     $$\mu_i = \bar{x}_i B_{:32} \in \mathbb{R}^{32}$$
  2. **분류기**: Context $\mu_i$에 대해 Class-balanced Dual Ridge ($\lambda=1.0$) 적용:
     $$M_{BM} = \text{logit}_1 - \text{logit}_0$$

---

### 2.6. BD (Bag-Dispersion / 고유값 스펙트럼 엔트로피 브랜치, v116 신규)
- **설계 목적**: 슬라이드 내 세포 임베딩의 전방위적 다형성 및 이질성(Pleomorphism / Spectral Diversity)을 측정하여 절대 분산 크기(스케일 편차)에 불변인 1차원 유계 증거 추출.
- **작동 과정**:
  1. **고유값 정규화 (Scale Invariance)**: 각 슬라이드의 기저 사영 공분산 $S_i = B^T C_i B \in \mathbb{R}^{256 \times 256}$의 고유값 $\{\lambda_1, \dots, \lambda_K\}$ 정규화:
     $$p_k = \frac{\lambda_k}{\sum_{j=1}^K \lambda_j} = \frac{\lambda_k}{\operatorname{Tr}(S_i)} \quad \left(\sum_{k=1}^K p_k = 1, \; p_k \ge 0\right)$$
  2. **스펙트럼 엔트로피 측정**:
     $$H_i = -\sum_{k=1}^K p_k \log (p_k + \epsilon) \quad \longrightarrow \quad \tilde{H}_i = \frac{H_i}{\log K} \in [0, 1]$$
  3. **분류기**: Context $\tilde{H}_i$로부터 Bounded Ordered-Typicality Evidence ($\kappa=1.0$) 적용:
     $$M_{BD} = a(\tilde{H}) \cdot o(\tilde{H}) \in [-1, 1]$$

---

## 3. Head 마진 결합

최종 마진 $M$은 5개 브랜치의 가중합으로 계산된다:
$$M = 1.0 \cdot M_{CV} - 1.0 \cdot M_{DD} + 1.0 \cdot M_{CT} + 1.0 \cdot M_{BM} + 1.0 \cdot M_{BD}$$

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
│   │   ├── training_free.py       # v116 활성 baseline (TrainingFreeClassifier: CV+DD+CT+BM+BD)
│   │   ├── stream_eval.py         # 고속 스트리밍 평가 및 통계 캐싱
│   │   ├── ct/                    # CT Readout 서브패키지 (config, tokenizers, abundance, readout)
│   │   ├── ct_readout.py          # src.models.ct Re-export Facade (100% 하위 호환)
│   │   └── dd_adaptive_rank.py    # DD & BD ordered-typicality margin
│   ├── datasets/
│   │   ├── synthetic/             # 합성 데이터 서브패키지 (types, generator, dataset)
│   │   └── synthetic_data.py      # src.datasets.synthetic Facade
│   └── modules/                   # Lightning 학습 인터페이스 (losses, diagnostics, guards)
├── scripts/
│   ├── node_env.sh                # 실행 노드별 Conda / Python 경로 자동 탐색 SSOT
│   ├── eval_v116.sh               # v116 활성 baseline 평가 스크립트 (기본: Primary 7 tasks)
│   ├── eval_v115.sh               # v115 baseline 평가 스크립트
│   ├── eval_seal_tasks.sh         # 태스크별 평가 러너
│   └── diagnostics/               # diagnose_*.py (14개 분석/진단 스크립트)
└── tests/                         # 핵심 회귀 테스트 스위트 (102개 핵심 계약 검증, ~20s)
    ├── test_bd_branch.py          # BD 브랜치 8대 불변식 계약 검증
    ├── test_bm_branch.py          # BM 브랜치 6대 불변식 계약 검증
    └── ...
```

### 4.2. 실행 가이드
```bash
# 1. 환경 로드
. scripts/node_env.sh

# 2. 회귀 테스트 실행
$PYTHON -m unittest discover -s tests -p "test_*.py"

# 3. v116 Baseline Primary 7-Task 평가 (기본)
bash scripts/eval_v116.sh <gpu_id> <tag>

# 4. v116 Hold-out 10-Task (SEAL) 검증
bash scripts/eval_v116.sh <gpu_id> <tag> \
  bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status \
  cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation \
  cptac_luad/EGFR_mutation cptac_luad/STK11_mutation cptac_luad/TP53_mutation \
  cptac_ccrcc/BAP1_mutation cptac_ccrcc/VHL_mutation
```

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-22 17:10:00_
