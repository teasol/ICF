# Current Architecture Specification

**Last updated**: `2026-09-09`

> **정본 분리.** 이 문서는 **브랜치 정의와 수식**의 정본이다. 성능 수치·비교 기준·승격 기준은
> 여기서 선언하지 않는다 — [`PROJECT.md`](PROJECT.md)를 본다.
>
> ⚠️ **아래 §1~§3은 6-branch 구성(CV·CT·BM·BD·QA·DS)을 기술한다.**
> **공식 비교 기준은 CT를 제외한 5-branch**이며([`PROJECT.md` §3.2](PROJECT.md)), CT의 수식은
> 계보 참조용으로 남겨 둔다. 채택됐으나 공식 구성에 들어가지 않은 형상 브랜치는 §2.8에 있다.

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

### 2.8. 형상(Shape) 계열 — 채택됐으나 공식 구성 밖

§217의 브랜치 분류 체계는 전 브랜치를 두 부류로 나눈다. **위치(Location) 계열**
(`CV`·`BM`·`QA`·`DS`, 상호 상관 0.25~0.93)은 표현이 *어디에 놓이는가*를 읽고,
**형상(Shape) 계열**은 분포의 *퍼짐과 이질성*을 읽는다. 형상 확장은 열린 연구 경로 중 하나다.
기존 위치 계열의 상관 관측을 다른 경로의 불가능으로 확대하지 않으며, 평균 차분 특징 FC도
비저촉으로 열려 있다 ([`closed_axes.md` `CA-09`](closed_axes.md), `D-022`).

현 형상 계열은 **`BD`·`SH`·`SJ` 3개**다 (`SJ`는 구 `SHJ`, `D-038`에서 2글자 규격으로 개정).
`BD`는 §2.5의 정식 브랜치이고, 나머지 둘은 게이트를 통과해 **채택(adopted)** 됐으나 공식 구성
승격은 별개 요건으로 남아 있다 ([`PROJECT.md` §5](PROJECT.md), [결정 이력](history/archive.md) `D-009`, `D-038`).

#### SJ (구 SHJ) — 백색화 반경 분포의 결합 형상

각 슬라이드를 **자기 자신의 평균과 공분산으로 백색화**한 뒤 토큰 구름의 반경 분포에서 8개
형상 기술자를 뽑는다. 구성상 위치·척도 불변이므로 평균을 다시 진술할 수 없다.

$$\text{proj} = X_i B_{:,:d},\quad C = \text{cov}(\text{proj}),\quad
W = (\text{proj} - \bar{\cdot}) V \Lambda^{-1/2},\quad r = \lVert W \rVert_2$$

$r$의 정렬된 분위수로부터 **왜도 · 초과첨도 · Bowley 왜도 · Moors 꼬리무게 ·
$q_{10}/q_{50}$ · $q_{90}/q_{50}$ · $q_{99}/q_{50}$ · $\text{IQR}/q_{50}$** 8차원을 만들고
클래스 균형 kernel ridge(선형)로 마진을 얻는다.

- **구현**: `src/models/branches/sj.py` (§222 정식 통합 후 `D-038`에서 명칭 통일, `shj.py`는 하위 호환 re-export 유지).
  기본 가중치 `weight_sj = 0.0` (`weight_shj` alias 지원) — 채택 상태이나 활성 앙상블에는 들어가지 않는다.
- ⚠️ **fp32 강제 필수.** 평가 파이프라인이 bf16 autocast 안에서 돌면 투영이 bf16(상대오차
  ~1e-3)으로 계산되고, 백색화의 `eigvals.clamp_min(1e-8).rsqrt()`가 이를 **약 100배 증폭**한다.
  `sj_slide_features()` 내부에서 `torch.autocast(enabled=False)`로 float32를 강제한다.

#### SH — 차원별 모멘트 형상 *(미통합)*

차원별 왜도·첨도를 슬라이드 자체 평균·표준편차로 표준화해 위치·척도 불변을 만든다
(§218에서 채택, max |r| = 0.418).

> ⚠️ **기술 부채: `SH`는 `src/models/`에 통합되어 있지 않다.** 구현은
> `scripts/test_pathobench.py` 안에만 있고 `ICF_SHAPE_SCREEN_ONLY`가 기본값 `1`이라 앙상블
> 경로에 들어가지 않는다. `SJ`만 §222에서 `src/models/branches/`로 이관됐다.
> 따라서 `branch_screen.py --adopted m_sh,m_sj`는 **SH 마진이 산출된 태그에서만** 완전한
> 심사를 수행하며, 없으면 경고를 출력한다 ([결정 이력](history/archive.md) `D-013`, `D-038`).

---

## 3. Head 마진 결합: Trimmed Mean Voting

v120 베이스라인은 6개 활성 브랜치의 독립 확률을 정렬 후 상/하단 극단치를 1개씩 절사하고 중앙 4개를 평균하는 **Trimmed Mean Voting** 방식을 사용한다:

$$p_{(1)} \le p_{(2)} \le p_{(3)} \le p_{(4)} \le p_{(5)} \le p_{(6)}, \quad \text{where } p_b = \sigma(M_b)$$
$$P(y=1) = \frac{1}{4} \sum_{k=2}^5 p_{(k)}$$

- **유효 마진 및 로짓 반환**:
  $$M_{eff} = \operatorname{logit}(P(y=1)) = \log \frac{P(y=1)}{1 - P(y=1)}$$
  $$\text{logits} = \left(-\frac{M_{eff}}{2}, +\frac{M_{eff}}{2}\right)$$
- **장점**: 특정 단일 브랜치에서 발생하는 파멸적 오작동(False Positive/Negative)을 100% 차단하며, 동시에 독립성이 높은 다종 브랜치의 순수 신호를 손실 없이 융합함.

---

## 4. 코드베이스 배치

브랜치 구현은 `src/models/branches/` 아래 한 브랜치당 한 파일이다.

```
src/models/
├── training_free.py          # 활성 파이프라인 — 기저 구축, 브랜치 호출, 집계 분기
├── config.py                 # 브랜치 가중치·차원·λ 기본값 (weight_shj 기본 0.0)
├── registry.py               # @register_model 데코레이터 및 build_model 팩토리
├── stream_eval.py            # 고속 스트리밍 평가 및 통계 캐싱
├── common/solvers.py         # Dual Ridge / kernel ridge 해법
├── branches/
│   ├── cv.py  bm.py  bd.py  qa.py  ds.py      # 공식 5-branch
│   ├── ct.py                                   # 계보 — 공식 비교 기준에서 제외
│   ├── sj.py                                   # 채택된 형상 브랜치 (§2.8, shj.py는 re-export)
│   ├── dd.py                                   # DD 는 CA-02 로 닫힘; BD 마진 제공
│   └── experimental/  de.py  lr.py  sw.py      # 기각·미판정 후보
├── ct/                       # CT 사전 구축 및 soft-token 할당
└── dd_adaptive_rank.py       # BD ordered-typicality 마진
```

- 디렉토리 전반의 역할표는 [`agent_handoff.md` §2.1](agent_handoff.md)에 있다.
- **실행 명령은 [`agent_handoff.md` §6](agent_handoff.md)이 정본이다.** 이 문서에 명령을
  중복해 적지 않는다 (과거 판이 존재하지 않는 `eval_v118.sh`·`eval_v117.sh`·`eval_v116.sh`를
  안내하고 있었다).
- 회귀 스위트는 브랜치별 불변식 계약을 검사한다 — `tests/test_bd_branch.py`,
  `test_bm_branch.py`, `test_qa_branch.py`, `test_sj_branch.py`(및 `test_shj_branch.py`), `test_soft_voting.py`,
  `test_core_contracts.py` 등.

_by Claude Opus 5 on nexgem-s1 at 2026-09-06_
