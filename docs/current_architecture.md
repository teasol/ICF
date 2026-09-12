# Current Architecture Specification

**Last updated**: `2026-09-09`

> **정본 분리.** 이 문서는 **브랜치 정의와 수식**의 정본이다. 성능 수치·비교 기준·승격 기준은
> 여기서 선언하지 않는다 — [`PROJECT.md`](PROJECT.md)를 본다.
>
> ⚠️ **아래 §1~§3은 6-branch 구성(CV·CT·BM·BD·QA·DS)을 기술한다.**
> **공식 비교 기준은 CT를 제외하고 형상 계열을 포함한 7-branch**
> `CV, BM, BD, QA, DS, SH, SJ`이며([`PROJECT.md` §3.2](PROJECT.md)), CT의 수식은
> 계보 참조용으로 남겨 둔다. `SH`·`SJ`의 승격 경위는 `D-042`이고 정의는 §2.8에 있다.

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

### 2.8. 형상(Shape) 계열 — `SH`·`SJ`는 공식 구성에 편입됨 (`D-042`)

§217의 브랜치 분류 체계는 전 브랜치를 두 부류로 나눈다. **위치(Location) 계열**
(`CV`·`BM`·`QA`·`DS`, 상호 상관 0.25~0.93)은 표현이 *어디에 놓이는가*를 읽고,
**형상(Shape) 계열**은 분포의 *퍼짐과 이질성*을 읽는다. 형상 확장은 열린 연구 경로 중 하나다.
기존 위치 계열의 상관 관측을 다른 경로의 불가능으로 확대하지 않으며, 평균 차분 특징 FC도
비저촉으로 열려 있다 ([`closed_axes.md` `CA-09`](closed_axes.md), `D-022`).

현 형상 계열은 **`BD`·`SH`·`SJ` 3개**이며 **셋 다 공식 구성의 정식 브랜치**다
(`SJ`는 구 `SHJ`, `D-038`에서 2글자 규격으로 개정).
`BD`는 §2.5부터 정식이었고, `SH`·`SJ`는 게이트 ①·②를 통과해 채택된 뒤
**`D-042`에서 승격**되어 공식 7-branch에 편입됐다 — 근거는 5-branch 대비 대응
`Δ_macro = +0.56%p`(기준 `+0.3%p`, `D-041`)와 랭크 효율 45.2% → 50.3% 상승이다
([`PROJECT.md` §3.2·§5](PROJECT.md), [결정 이력](history/archive.md) `D-009`, `D-038`, `D-041`, `D-042`).
**단, 세 과제(`Histologic_Grade`·`progression_regression`·`PBRM1`)는 형상 계열 투입으로
일관되게 악화된다. 기전 미확인이며 `hold-out 미검증`이다.**

#### SJ (구 SHJ) — 백색화 반경 분포의 결합 형상

각 슬라이드를 **자기 자신의 평균과 공분산으로 백색화**한 뒤 토큰 구름의 반경 분포에서 8개
형상 기술자를 뽑는다. 구성상 위치·척도 불변이므로 평균을 다시 진술할 수 없다.

$$\text{proj} = X_i B_{:,:d},\quad C = \text{cov}(\text{proj}),\quad
W = (\text{proj} - \bar{\cdot}) V \Lambda^{-1/2},\quad r = \lVert W \rVert_2$$

$r$의 정렬된 분위수로부터 **왜도 · 초과첨도 · Bowley 왜도 · Moors 꼬리무게 ·
$q_{10}/q_{50}$ · $q_{90}/q_{50}$ · $q_{99}/q_{50}$ · $\text{IQR}/q_{50}$** 8차원을 만들고
클래스 균형 kernel ridge(선형)로 마진을 얻는다.

- **구현**: `src/models/branches/sj.py` (§222 정식 통합 후 `D-038`에서 명칭 통일, `shj.py`는 하위 호환 re-export 유지).
  `weight_sj`의 코드 기본값은 `0.0`이며 (`weight_shj` alias 지원), **공식 구성에서는 활성화한다**
  (`D-042`). 기본값을 0으로 두는 것은 과거 설정의 재현성을 지키기 위한 것이지 미승격을 뜻하지 않는다.
- ⚠️ **fp32 강제 필수.** 평가 파이프라인이 bf16 autocast 안에서 돌면 투영이 bf16(상대오차
  ~1e-3)으로 계산되고, 백색화의 `eigvals.clamp_min(1e-8).rsqrt()`가 이를 **약 100배 증폭**한다.
  `sj_slide_features()` 내부에서 `torch.autocast(enabled=False)`로 float32를 강제한다.

#### SH — 차원별 모멘트 형상

차원별 왜도·첨도를 슬라이드 자체 평균·표준편차로 표준화해 위치·척도 불변을 만든다
(§218에서 채택, max |r| = 0.418). 토큰을 넓은 기저(`sh_wide`, 기본 256)에 한 번 투영한 뒤
차원별 모멘트를 읽고 각 모멘트의 앞 `sh_dim`(기본 32)개만 남긴다 — 나머지 스크린 전용
변형(`shs`/`shk`/`sh2`/`shr`/`shr2`)은 §219에서 실질 기각 상태로 스크립트에만 남는다.

- **구현**: `src/models/branches/sh.py` (스크리닝 폐쇄 함수에서 이관, `D-039` 정식 통합).
  `weight_sh`의 코드 기본값은 `0.0`이며, **공식 구성에서는 활성화한다** (`D-042`).
  기본값을 0으로 두는 것은 과거 설정의 재현성을 지키기 위한 것이지 미승격을 뜻하지 않는다.
- ⚠️ **fp32 강제 필수.** SJ와 동일 계약이다. bf16 autocast가 표준화의 분모(sd)를 흔들면
  4제곱 모멘트가 오차를 증폭하므로 `sh_slide_features()` 내부에서
  `torch.autocast(enabled=False)`로 float32를 강제한다.
- `SH`·`SJ`가 승격되어 `branch_screen.py`의 기준 집합(`BRANCHES`)에 편입됐으므로
  "채택됐으나 미승격" 목록(`--adopted`)은 **비어 있다** (`D-042`). 심사는 여전히 **`m_sh`·`m_sj`
  마진이 산출된 태그에서만** 완전하며, 통합 이전 태그처럼 기록에 없으면 경고를 출력한다
  ([결정 이력](history/archive.md) `D-013`, `D-038`, `D-042`).

#### BS — 로그 총분산 (게이트 ② 기각, 구현만 보존)

투영 토큰 구름의 **로그 총분산** $\log\big(\mathrm{tr}(B^\top S B)/n\big)$ 한 개 값이다.
`BD`의 엔트로피 경로가 $p = \text{eig}/\sum\text{eig}$로 정규화하며 버리는 축을 그대로 읽으므로
순수 척도(scale) 통계이며, 평행이동에 완전 불변이고 등방 스케일 $s$에 대해 정확히 $\log(s^2)$만큼
이동한다.

- **지위**: 게이트 ①은 통과했으나(max |r| = 0.262) **게이트 ②에서 기각**됐다 — 단독 AUROC가
  `0.4201 ~ 0.5478`로 Primary 7 중 4과제에서 우연 이하이고, 최선 과제도 50 fold 중 31개로
  `p = 0.059`다(컷 40/50, `p < 0.01`). §218이 "직교하나 무정보(orthogonal but uninformative)"라는
  범주를 만든 사례이며, 정보량 게이트 신설의 계기다 ([결정·이력](history/archive.md) §218).
- **구현**: `src/models/branches/bs.py`, 기본 가중치 `weight_bs = 0.0`. **기각 후보이므로 어떤
  구성에서도 승격 심사에 상정하지 않는다.** 구현을 보존하는 이유는 스크리닝 이력의 재현성과,
  게이트 ② 통계량을 단측에서 양측(`|AUROC − 0.5|`)으로 바꿀지에 대한 미결 판정
  ([`research_directions.md`](research_directions.md) 큐 `B5`)에 BS가 직접 걸리는 후보이기
  때문이다. 그 판정 전에는 BS를 되살리지 않는다.
- BS는 척도 통계라 백색화·표준화를 쓰지 않으므로 SH·SJ와 달리 **fp32 강제 계약이 없다**.
  단일 토큰 슬라이드(N = 1)에서도 NaN을 내지 않는다.

---

## 3. Head 마진 결합: Trimmed Mean Voting

v120 베이스라인은 6개 활성 브랜치의 독립 확률을 정렬 후 상/하단 극단치를 1개씩 절사하고 중앙 4개를 평균하는 **Trimmed Mean Voting** 방식을 사용한다:

$$p_{(1)} \le p_{(2)} \le p_{(3)} \le p_{(4)} \le p_{(5)} \le p_{(6)}, \quad \text{where } p_b = \sigma(M_b)$$
$$P(y=1) = \frac{1}{4} \sum_{k=2}^5 p_{(k)}$$

- **유효 마진 및 로짓 반환**:
  $$M_{eff} = \operatorname{logit}(P(y=1)) = \log \frac{P(y=1)}{1 - P(y=1)}$$
  $$\text{logits} = \left(-\frac{M_{eff}}{2}, +\frac{M_{eff}}{2}\right)$$
- **장점**: 특정 단일 브랜치에서 발생하는 파멸적 오작동(False Positive/Negative)을 100% 차단하며, 동시에 독립성이 높은 다종 브랜치의 순수 신호를 손실 없이 융합함.

### 3.1. 어떤 브랜치가 실제로 pool에 들어가는가 (`D-040`, 2026-09-12 단일화)

집계에 참여하는 브랜치는 **단일 정본**인 `src/models/aggregations/voting.py`의 `_fixed_branch_pairs`·`_shape_branch_pairs`에서 결정된다.
브랜치 순서는 `cv, dd, ct, bm, bd, qa, ds, lr, de, sw, sj, sh, bs`이며, 참여 조건은 `weight != 0 이고 마진이 존재함`이다.
(과거 `test_pathobench.py`에 이원화되어 존재하던 평가 분기는 RFC 2026-09-12 / RU-91 마이그레이션에서 완전히 제거되어 `voting.py` 단일 출처로 일원화되었다.)

**`ICF_SHAPE_SCREEN_ONLY`가 형상 계열(BS·SH·SJ)의 참여를 지배한다.**

| 값 | 동작 |
|:---|:---|
| `1` (기본) | 마진을 계산·저장하되 **어떤 집계 pool에도 넣지 않는다**. 스크리닝용이며, 이 상태의 실행 결과는 형상 브랜치가 없던 때와 비트 단위로 같다. |
| `0` | 다른 브랜치와 동일하게 5개 집계 전부(및 `context_loo`)에 참여한다. |

> ⚠️ **`D-040` 이전에는 이 계약이 성립하지 않았다.** 다섯 집계 분기가 형상 계열을 열거하지
> 않아 BS·SH·SJ는 `ICF_SHAPE_SCREEN_ONLY=0`으로도 최종 확률에 도달하지 못했다. 세 마진은
> `logits`에만 가산됐고, `logits`는 모든 브랜치 가중치가 0일 때만 도달하는 폴백에서만
> 읽힌다. 형상 계열이 실제로 융합된 수치는 전부 **저장 마진의 오프라인 재집계**로 산출된
> 것이며(RU-89·RU-90), live 경로로 재현 가능해진 것은 `D-040` 이후다.

---

## 4. 코드베이스 배치

브랜치 구현은 `src/models/branches/` 아래 한 브랜치당 한 파일이다.

```
src/models/
├── training_free.py          # 활성 파이프라인 — 기저 구축, 브랜치 호출, 집계 분기
├── config.py                 # 브랜치 가중치·차원·λ 기본값 (형상 계열 weight_sj·weight_sh 기본 0.0)
├── registry.py               # @register_model 데코레이터 및 build_model 팩토리
├── stream_eval.py            # 고속 스트리밍 평가 및 통계 캐싱
├── common/solvers.py         # Dual Ridge / kernel ridge 해법
├── branches/
│   ├── cv.py  bm.py  bd.py  qa.py  ds.py      # 공식 7-branch 중 5개
│   ├── ct.py                                   # 계보 — 공식 비교 기준에서 제외
│   ├── sj.py                                   # 공식 형상 브랜치 (§2.8, D-042 승격, shj.py는 re-export)
│   ├── sh.py                                   # 공식 형상 브랜치 (§2.8, D-042 승격, D-039에서 이관)
│   ├── bs.py                                   # 게이트 ② 기각 — 스크리닝 이력 보존 (§2.8, D-040)
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
  `test_bm_branch.py`, `test_qa_branch.py`, `test_sj_branch.py`(및 `test_shj_branch.py`),
  `test_sh_branch.py`, `test_bs_branch.py`, `test_shape_branch_invariance.py`,
  `test_soft_voting.py`, `test_core_contracts.py` 등.

### 4.1. 집계 로직의 단일 정본화 및 순수 러너 단일화 (2026-09-12 완료)

과거 집계 수식이 두 곳에 따로 구현돼 있던 기술부채는 2026-09-12 교살자 패턴 마이그레이션(RFC 2026-09-12, RU-91)을 통해 근본적으로 해소되었다:
1. **단일 정본 러너**: `scripts/evaluate_pure.py`가 전체 평가를 담당하며, `TrainingFreeClassifier`와 `src/models/aggregations/voting.py`만을 직접 사용한다.
2. **레거시 11,800라인 일괄 삭제**:
   - `src/datasets/` (2,004 lines)
   - `src/modules/` (1,529 lines)
   - `src/models/baseline.py` (2,317 lines)
   - `src/models/set_transformer_ridge.py` (2,474 lines)
   - `scripts/test_pathobench.py` (3,123 lines)
   - `src/utils/utils.py` 내 레거시 Trainer 인터페이스 (~350 lines)
3. **수치 패리티 확증**: Primary 7 태스크 50-fold 전체(17,723 슬라이드)에서 순수 러너와 레거시 오라클 간 Macro AUROC `0.6226` vs `0.6226` ($\Delta = -0.0000$, $\text{mean}|\Delta p| \sim 10^{-5}$)로 수치적 항등을 완벽히 입증(RU-91-B).
4. **골든 참조 보존**: 기존 레거시 `ru90_shape_triple` 예측값(`predictions/pathobench_*_ru90_shape_triple_official50_bf16.pt`)은 역사적 영구 참조 파일로 보존된다.

[작성자: Platform Agent / Gemini 3.8 Flash (effort: high) · 2026-09-12 17:55 KST]

