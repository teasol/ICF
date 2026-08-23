# Current Architecture Specification (v120 Active Baseline)

**Last updated**: `2026-08-23 09:30:00`

---

## 1. 아키텍처 개요 및 설계 철학

ICF(In-Context Foundation) 모델의 활성 베이스라인(v120)은 **학습 파라미터가 0개(0-parameter)인 완전 결정론적(Deterministic) 6-Branch Trimmed Mean 인컨텍스트 분류기**다.

- **원리**: 슬라이드(Bag) 단위의 다중 인스턴스(MIL) 병리 이미지 임베딩($X_i \in \mathbb{R}^{N_i \times 1536}$)에서, 레이블이 제공된 Context 슬라이드들만으로 Within-slide PCA 기저를 구축하고 6개 고성능 상보적 통계 브랜치(**CV, CT, BM, BD, QA, DS**)로부터 각각 독립적인 마진을 산출한 뒤, **Trimmed Mean Voting (최고/최저 1개씩 절사 후 중앙 4개 확률 평균)**으로 최종 예측 확률을 결합한다.
- **불변식**: Query 슬라이드는 기저 생성, 토큰 군집화, 통계 표준화에 일절 참여하지 않으며(No-Leakage), 라벨 반전($y \to 1-y$) 시 각 브랜치 마진과 앙상블 확률이 정확히 반전된다(Label Antisymmetry).

```
[Context Bags (Labeled) + Query Bags (Unlabeled)]
                      │
                      ▼
   Within-Slide PCA (Context-Only Centered, K=256)
                      │
     ┌────────────────┼────────────────┬────────────────┬────────────────┬────────────────┐
     │                │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼                ▼
[CV Branch]      [CT Branch]      [BM Branch]      [BD Branch]      [QA Branch]      [DS Branch]
2차 공분산       32D 의사세포형    1차 모멘트       256D 스펙트럼    128D 4분위수     32D Salience
비대각 32,640D   256 토큰 조성비   상위 32D 사영    엔트로피         분포 형태        Denoised Mean
     │                │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼                ▼
Dual Ridge (λ=1) Dual Ridge (λ=1) Dual Ridge (λ=1) Bounded Margin   Dual Ridge (λ=1) Dual Ridge (λ=1)
   (M_CV)           (M_CT)           (M_BM)           (M_BD)           (M_QA)           (M_DS)
     │                │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼                ▼
   p_CV             p_CT             p_BM             p_BD             p_QA             p_DS
(sigmoid)        (sigmoid)        (sigmoid)        (sigmoid)        (sigmoid)        (sigmoid)
     │                │                │                │                │                │
     └────────────────┼────────────────┼────────────────┴────────────────┴────────────────┘
                                       ▼
                     Sort: p_(1) <= p_(2) <= ... <= p_(6)
                     Trimmed Mean: P(y=1) = (p_(2) + p_(3) + p_(4) + p_(5)) / 4
                                       │
                                       ▼
                     Margin = logit(P) = log(P / (1 - P))  -->  AUROC Ranking
```

---

## 2. 6대 브랜치 상세 작동 원리 및 수식

### 2.1. 기저 구축 (Within-Slide PCA Basis)
- **입력**: Context 슬라이드 집합 $\{X_i\}_{i=1}^{n_{ctx}}$, 각 $X_i \in \mathbb{R}^{N_i \times 1536}$.
- **슬라이드 내 중심화(Within-slide Centering)**:
  $$C_{within} = \frac{1}{\sum N_i} \sum_{i=1}^{n_{ctx}} \sum_{j=1}^{N_i} (x_{ij} - \bar{x}_i)(x_{ij} - \bar{x}_i)^T$$
  *(슬라이드 간 배경/염색 편차인 between-slide nuisance를 소거)*
- **기저 $B$**: $C_{within}$의 상위 $K=256$개 고유벡터 $B \in \mathbb{R}^{1536 \times 256}$.

### 2.2. CV (Cross-Covariance / 2차 상관관계 브랜치)
- **설계 목적**: 슬라이드 내 세포 특징 간의 2차 상관 구조 포착.
- **디스크립터**: 각 슬라이드의 기저 사영 공분산 $S_i = B^T C_i B \in \mathbb{R}^{256 \times 256}$의 **상삼각 비대각(Off-diagonal) 요소 32,640차원**:
  $$\text{desc}_{CV}(i) = \text{triu\_offdiag}(S_i) \in \mathbb{R}^{32,640}$$
- **분류기**: Context 디스크립터에 대한 Class-balanced Dual Ridge ($\lambda=1.0$):
  $$M_{CV} = \text{logit}_1 - \text{logit}_0$$

### 2.3. CT (Cell-Type Abundance / 의사 세포형 조성비 브랜치)
- **설계 목적**: 슬라이드 내 주요 세포 아형(Subpopulations)의 상대적 빈도/조성비 비교.
- **작동 과정**:
  1. **샘플링**: 슬라이드 크기의 $1/8$ fraction (floor 64, seeded random seed 0).
  2. **저차원 사영**: $B$의 상위 **32 PCA 차원**으로 사영 후 Context 표준화.
  3. **토큰 사전 생성**: Context 세포들 위에서 **Seeded k-means++ (Lloyd $\le 8$, tol 1e-4)로 256개 토큰** 생성.
  4. **소프트 할당**: $a_{ik} = \frac{1}{N_i} \sum_j \operatorname{softmax}_k (-\|z_{ij} - t_k\|^2 / \tau)$ ($\tau=0.5$).
  5. **분류기**: 256차원 abundance 벡터에 대해 Class-balanced Dual Ridge ($\lambda=1.0$) 적용 $\to M_{CT} = \text{logit}_1 - \text{logit}_0$.

### 2.4. BM (Bag-Mean / 1차 모멘트 저차원 사영 브랜치)
- **설계 목적**: 2차 공분산(CV) 및 패치 군집(CT)이 담지 못하는 슬라이드 전체 1차 모멘트(세포 평균 $\bar{x}_i$)의 기준점(baseline shift) 정보 포착.
- **작동 과정**:
  1. **저차원 사영**: 각 슬라이드의 세포 평균 $\bar{x}_i \in \mathbb{R}^{1536}$을 Within-slide PCA 기저 $B$의 상위 **32차원**으로 사영:
     $$\mu_i = \bar{x}_i B_{:32} \in \mathbb{R}^{32}$$
  2. **분류기**: Context $\mu_i$에 대해 Class-balanced Dual Ridge ($\lambda=1.0$) 적용:
     $$M_{BM} = \text{logit}_1 - \text{logit}_0$$

### 2.5. BD (Bag-Dispersion / 고유값 스펙트럼 엔트로피 브랜치)
- **설계 목적**: 슬라이드 내 세포 임베딩의 전방위적 다형성 및 이질성(Pleomorphism / Spectral Diversity)을 측정하여 절대 분산 크기(스케일 편차)에 불변인 1차원 유계 증거 추출.
- **작동 과정**:
  1. **고유값 정규화 (Scale Invariance)**: 각 슬라이드의 기저 사영 공분산 $S_i = B^T C_i B \in \mathbb{R}^{256 \times 256}$의 고유값 $\{\lambda_1, \dots, \lambda_K\}$ 정규화:
     $$p_k = \frac{\lambda_k}{\sum_{j=1}^K \lambda_j} = \frac{\lambda_k}{\operatorname{Tr}(S_i)} \quad \left(\sum_{k=1}^K p_k = 1, \; p_k \ge 0\right)$$
  2. **스펙트럼 엔트로피 측정**:
     $$H_i = -\sum_{k=1}^K p_k \log (p_k + \epsilon) \quad \longrightarrow \quad \tilde{H}_i = \frac{H_i}{\log K} \in [0, 1]$$
  3. **분류기**: Context $\tilde{H}_i$로부터 Bounded Ordered-Typicality Evidence ($\kappa=1.0$) 적용:
     $$M_{BD} = a(\tilde{H}) \cdot o(\tilde{H}) \in [-1, 1]$$

### 2.6. QA (Quantile & Extremum Evidence / 비가우시안 4분위수 브랜치)
- **설계 목적**: 평균(1차 모멘트)과 분산(2차 모멘트)으로 포착할 수 없는 고차 비대칭성 및 극단 세포(Outlier Cells) 형태 추출.
- **디스크립터**: 상위 32 PCA 차원 각각에 대해 $[Q_{0.05}, Q_{0.10}, Q_{0.90}, Q_{0.95}]$ 분위수를 추출하여 결합한 128차원 벡터:
  $$\mathbf{q}_i = [Q_{0.05}(Z_i), Q_{0.10}(Z_i), Q_{0.90}(Z_i), Q_{0.95}(Z_i)] \in \mathbb{R}^{128}$$
- **분류기**: Class-balanced Dual Ridge ($\lambda=1.0$) 적용 $\to M_{QA} = \text{logit}_1 - \text{logit}_0$.

### 2.7. DS (Denoised Salience Bag-Mean / 정상 기질 희석 방지 브랜치)
- **설계 목적**: 90% 이상의 정상 기질(Stroma) 세포로 인해 희석되는 평균 벡터에서, 클래스 공통 토큰(Common Subpopulations)의 가중치를 낮추고 특이 토큰에 높은 가중치를 부여한 Salience-weighted Mean 산출.
- **수식**:
  $$s_k = \frac{|\bar{a}_{1,k} - \bar{a}_{0,k}|}{\bar{a}_{1,k} + \bar{a}_{0,k} + \epsilon}, \quad w_{i,j} \propto \sum_{k=1}^K a_{ij,k} \cdot s_k^\gamma$$
  $$\mathbf{z}_i^{DS} = \sum_{j=1}^{N_i} w_{i,j} \mathbf{z}_{i,j} \in \mathbb{R}^{32}$$
- **분류기**: Class-balanced Dual Ridge ($\lambda=1.0$) 적용 $\to M_{DS} = \text{logit}_1 - \text{logit}_0$.

---

## 3. Head 마진 결합: Trimmed Mean Voting (v120)

v120 베이스라인은 6개 활성 브랜치의 독립 확률을 정렬 후 상/하단 극단치를 1개씩 절사하고 중앙 4개를 평균하는 **Trimmed Mean Voting** 방식을 사용한다:

$$p_{(1)} \le p_{(2)} \le p_{(3)} \le p_{(4)} \le p_{(5)} \le p_{(6)}, \quad \text{where } p_b = \sigma(M_b)$$
$$P(y=1) = \frac{1}{4} \sum_{k=2}^5 p_{(k)}$$

- **유효 마진 및 로짓 반환**:
  $$M_{eff} = \operatorname{logit}(P(y=1)) = \log \frac{P(y=1)}{1 - P(y=1)}$$
  $$\text{logits} = \left(-\frac{M_{eff}}{2}, +\frac{M_{eff}}{2}\right)$$
- **장점**: 특정 단일 브랜치에서 발생하는 파멸적 오작동(False Positive/Negative)을 100% 차단하며, 동시에 독립성이 높은 다종 브랜치의 순수 신호를 손실 없이 융합함.

---

## 4. 코드베이스 모듈 구조 및 개발자 가이드

### 4.1. 계층별 패키지 구성
```
ICF/
├── src/
│   ├── models/
│   │   ├── base.py                # InContextClassifierProtocol & BaseInContextClassifier
│   │   ├── registry.py            # @register_model 데코레이터 및 build_model 팩토리
│   │   ├── training_free.py       # v120 활성 baseline (Trimmed Mean: CV+CT+BM+BD+QA+DS)
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
│   ├── eval_v118.sh               # v118 활성 baseline 평가 스크립트 (Soft Voting, Primary 7 tasks)
│   ├── eval_v117.sh               # v117 baseline 평가 스크립트 (No-DD Linear, Primary 7 tasks)
│   ├── eval_v116.sh               # v116 baseline 평가 스크립트 (5-Branch Linear)
│   ├── analyze_voting.py          # 저장된 5-Branch Logit 오프라인 앙상블 분석기
│   ├── eval_seal_tasks.sh         # 태스크별 평가 러너
│   └── diagnostics/               # diagnose_*.py (14개 분석/진단 스크립트)
└── tests/                         # 핵심 회귀 테스트 스위트 (107개 핵심 계약 검증)
    ├── test_soft_voting.py        # v118 Soft Voting 앙상블 불변식 및 계약 검증
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

# 3. v118 Baseline Primary 7-Task 평가 (기본)
bash scripts/eval_v118.sh <gpu_id> <tag>

# 4. v118 Hold-out 10-Task (SEAL) 검증
bash scripts/eval_v118.sh <gpu_id> <tag> \
  bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status \
  cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation \
  cptac_luad/EGFR_mutation cptac_luad/STK11_mutation cptac_luad/TP53_mutation \
  cptac_ccrcc/BAP1_mutation cptac_ccrcc/VHL_mutation
```

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-22 18:15:00_
