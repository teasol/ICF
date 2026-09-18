# 문헌 브리프 — pfn_uncertainty (로컬 모델 요약)

```
url: https://arxiv.org/abs/2505.11325
fetched_from: https://arxiv.org/html/2505.11325
fetched_at: 2026-09-18 13:01:06 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8003` · 2026-09-18 13:08 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- PFN은 표형 데이터 예측에서 small-to-moderate 데이터 크기에서 tuning 없이 state-of-the-art 성능을 달성하는 foundation model이다. (출처: Abstract, Section 1)
- PFN은 Bayesian idea에서 출발하지만 predictive mean, quantile 같은 예측 요약량에 대한 uncertainty quantification을 제공하지 않는다. (출처: Abstract, Section 1, Section 2.1)
- 저자들은 martingale posteriors 기반의 principled, efficient, tuning-free sampling procedure를 제안하고 수렴성을 증명한다. (출처: Abstract, Section 1)
- 제안된 AMP는 learned predictive distribution에 접근할 수 있는 PFN에 적용 가능하며, 실험에서는 TabPFN과 TabICL을 사용한다. (출처: Figure 1 caption, Section 4)
- PFN의 posterior predictive distribution은 aleatoric uncertainty와 epistemic uncertainty를 섞어 구분할 수 없다. (출처: Section 2.1)
- 최신 transformer 기반 PFN은 martingale property에서 벗어나며, PFN을 반복 호출하는 기존 MP 방식은 계산 비용이 크다. (출처: Section 3)
- AMP는 PFN의 PPD를 시작점으로만 쓰고, martingale property를 만족하는 nonparametric forward sampling으로 posterior를 구성한다. (출처: Section 3)
- posterior 차이에서 초기 PPD initialization이 지배적 요인이다. (출처: Corollary 3.1)
- 시뮬레이션과 실데이터에서 AMP는 대략적으로 올바른 coverage를 제공하며, bootstrap은 대부분 undercoverage를 보이고 더 느리다. (출처: Section 4.1, Section 4.3)
- AMP는 tuning-free이고 몇 초 안에 well-calibrated credible intervals를 만든다고 주장한다. (출처: Section 1, Section 5)

## 2. 방법의 핵심
- 구조: PFN이 초기 predictive distribution \(P_0(y \mid x, \mathcal{D}_n)\)를 제공한다. AMP는 \(B\)개의 독립 chain에서 \(P_0\)로 시작해 \(N\)번 forward sampling을 한다. 각 step에서 \(y_{n+k} \sim P_{k-1}\)를 샘플링하고, Gaussian copula update로 \(P_k\)를 만든다. 최종 empirical CDF \(\widehat{P}_N\)에서 conditional mean 또는 quantile 같은 \(\theta(x)\)를 계산하고, \(B\)개 결과의 empirical distribution을 martingale posterior estimate로 쓴다. (출처: Algorithm 1, Section 3.1, Section 3.2, Section 3.6)
- update 식: \(P_k(y) = (1-\alpha_{n+k-1})P_{k-1}(y) + \alpha_{n+k-1}H_\rho(P_{k-1}(y), P_{k-1}(y_{n+k}))\), 여기서 \(H_\rho(u,v)=\Phi((\Phi^{-1}(u)-\rho\Phi^{-1}(v))/\sqrt{1-\rho^2})\)이다. (출처: Section 3.2, Equation 2)
- 학습 방식: PFN은 synthetic datasets로 pre-training된 후 weights가 고정된다. AMP 자체는 PFN 파라미터를 업데이트하지 않고, PFN output을 informed prior처럼 사용한다. (출처: Section 2.1, Section 3)
- 추론 방식: 먼저 PFN을 한 번 forward pass해 PPD를 얻고, 이후 AMP update는 training set size \(n\)에 독립적이며 \(O(BN)\) 비용이다. outer loop \(B\)는 병렬/벡터화 가능하고 inner loop \(N\)은 순차적이다. (출처: Section 3.6)
- 재현에 필요한 본문 설정: 기본 실험은 \(N=50\), \(B=50\)이다. \(\rho=0.99\)를 사용한다. learning rate는 \(\alpha_i=\min\{1, C_{n,N}^{-1}2^\beta(i+1)^{-\beta}\}\), \(\beta=1/2+2/(1.1d+4)\)이다. \(C_{n,N}\approx[1-(1+N/n)^{1-2\beta}]^{1/2}\)이다. 실험 환경은 Python 3.12.7, GeForce RTX 4070 Ti, tabpfn==2.0.5(n_estimators=8), tabicl==2.0.3(n_estimators=2)이다. (출처: Section 3.2, Section 3.4, Section 3.6, Section 4, Appendix B.4)

## 3. 평가 설정
- 시뮬레이션 데이터셋: Bayesian additive model DGP. B-spline basis는 degree 3, 20 basis functions, \(\theta_j \sim \mathcal{N}(0,2I)\), \(n \in \{50,100,200,400,800\}\), \(d \in \{1,10,20\}\), \(J \in \{\lceil d/2\rceil, d\}\), \(Y \mid x,\theta \sim \mathcal{N}(\sum_j f_j(x_j),1)\). 각 \((n,d,J)\)당 20개 dataset, 100개 random test points를 평가한다. (출처: Section 4.1)
- 실데이터셋: UCI repository의 airfoil(n=1503, d=5), boston(n=252, d=15), concrete(n=1030, d=8), diabetes(n=442, d=10), energy(n=768, d=8), fish(n=908, d=6), forest_fire(n=516, d=13), real(n=413, d=7). (출처: Appendix B.2)
- 실데이터 평가 방식: 20-80 train-test split을 10회 random split으로 평균하고 \(\pm 2se\)를 보고한다. target coverage는 90%이다. true conditional quantile이 없으므로 test data만으로 학습/평가한 별도 PFN을 oracle surrogate로 사용해 predicted quantile이 credible interval 안에 들어가는지 확인한다. (출처: Section 4.3, Table 1 caption, Table 2 caption)
- 비교 기준선: 시뮬레이션에서는 true DGP를 prior로 쓰는 analytic credible sets를 Optimal이라 부른다. 실데이터에서는 bootstrap baseline을 비교하며, bootstrap은 training data를 with replacement로 재샘플링하고 TabPFN으로 quantile을 예측한 뒤 \(B=50\) replications의 empirical confidence interval을 만든다. (출처: Section 4.1, Section 4.3)
- ABMIL: 본문에서 확인 못함. ABMIL은 기준선으로 등장하지 않는다.
- 지표: coverage, interval width, runtime time. 시뮬레이션에서는 coverage와 CI width를 보고하며, 실데이터 Table 1/2에서는 coverage, width, time을 보고한다. (출처: Figure 2, Table 1, Table 2)

## 4. 정량 결과
- 기본 AMP 설정: \(N=50\), \(B=50\). (출처: Section 4)
- Gaussian copula bandwidth: \(\rho=0.99\). (출처: Section 3.2)
- 유한 \(N\) 보정 예시: \(n=200\), \(N=1000\), \(d=10\)이면 \(C_{n,N}\approx0.62\)이고, credible intervals가 거의 40% too small해질 수 있다. (출처: Section 3.6)
- 시뮬레이션: target coverage는 90%이며, 일부 설정에서 \(n=50\)일 때 최저 coverage는 약 80% 수준이다. (출처: Section 4.1, Figure 2)
- forward samples ablation: coverage는 blow-up factor 덕분에 약 \(N=20\)에서 안정화되는 것으로 보고된다. (출처: Section 4.2, Figure 3)
- Table 1, 90%-quantile regression, real 데이터셋 열: 첫 TabPFN 라벨 블록은 coverage \(0.88 \pm 0.03\), width \(2.15 \pm 0.35\), time \(5.1 \pm 0.1\) s이다. AMP TabICL 블록은 coverage \(0.87 \pm 0.03\), width \(2.43 \pm 0.29\), time \(5.0 \pm 0.0\) s이다. 세 번째 TabPFN 라벨 블록은 coverage \(0.55 \pm 0.04\), width \(0.52 \pm 0.05\), time \(10.3 \pm 1.1\) s이다. Bootstrap TabICL 블록은 coverage \(0.70 \pm 0.03\), width \(0.59 \pm 0.06\), time \(13.5 \pm 0.2\) s이다. (출처: Table 1)
- Table 2, median regression, real 데이터셋 열: 첫 TabPFN 라벨 블록은 coverage \(0.98 \pm 0.01\), width \(3.08 \pm 0.28\), time \(5.3 \pm 0.3\) s이다. AMP TabICL 블록은 coverage \(0.96 \pm 0.02\), width \(3.08 \pm 0.28\), time \(5.2 \pm 0.4\) s이다. 세 번째 TabPFN 라벨 블록은 coverage \(0.66 \pm 0.02\), width \(0.99 \pm 0.13\), time \(15.9 \pm 4.9\) s이다. Bootstrap TabICL 블록은 coverage \(0.75 \pm 0.04\), width \(1.09 \pm 0.12\), time \(33.2 \pm 13.2\) s이다. (출처: Table 2)

## 5. 우리 프로젝트와의 관련 (가설)
- 옮겨올 수 있는 것, 가설: fold context에서 closed-form ridge로 푸는 in-context 분류기가 label 또는 score에 대한 predictive distribution를 제공할 수 있다면, AMP처럼 초기 PPD를 받아 nonparametric resampling으로 epistemic uncertainty를 추정하는 후처리 레이어를 시도할 수 있다. 이는 stochastic fitting으로 인한 적합 분산이 승격 기준보다 얼마나 큰지를 interval 형태로 진단하는 데 도움이 될 수 있다.
- 옮겨올 수 있는 것, 가설: AMP의 learning rate schedule과 finite-\(N\) blow-up factor \(C_{n,N}^{-1}\)는 유한 샘플에서 posterior spread가 과소평가되는 문제를 교정하는 아이디어로, 우리 stochastic fitting의 분산이 승격 기준의 최대 2.8배로 커지는 문제를 이해하는 참고 틀이 될 수 있다.
- 옮겨올 수 있는 것, 가설: Corollary 3.1의 “초기 PPD가 posterior 차이를 지배한다”는 주장은, 우리 in-context 분류기의 확률 보정이 불완전하면 AMP 계열 UQ 레이어도 신뢰하기 어렵다는 가설을 제공한다.
- 옮길 수 없는 것, 이유: 이 논문은 iid tabular prediction과 pointwise query를 전제로 하며, WSI의 공간 의존성, patch-level MIL aggregation, image-level decision, ABMIL 비교를 다루지 않는다. 따라서 ABMIL 기준선을 넘는 성능 개선으로 직접 연결할 근거는 본문에 없다.
- 옮길 수 없는 것, 이유: branch 7개의 effective rank 3.52/7 같은 중복성 문제는 rank, covariance, branch redundancy를 다루지 않는 이 논문에서는 직접 해결 대상으로 제시되지 않는다.

## 6. 이 논문이 답하지 않는 것
- 병리 WSI 벤치마크나 MIL 설정에서 AMP가 ABMIL보다 좋은지: 본문에서 확인 못함.
- ABMIL을 baseline으로 비교했는지: 본문에서 확인 못함.
- branch 7개의 effective rank 3.52/7 중복을 줄이는 방법: 본문에서 확인 못함.
- stochastic fitting의 fit variance가 승격 기준의 최대 2.8배로 커지는 문제를 직접 줄이는 방법: 본문에서 확인 못함.
- deterministic closed-form ridge output에서 AMP가 요구하는 predictive distribution을 어떻게 생성할지: 본문에서 확인 못함.
- 여러 fold, branch, query point에 대한 joint posterior 또는 uniform credible set: 논문은 pointwise 한계를 명시하고, joint posterior는 future work로만 언급한다. (출처: Section 5, Limitations)
- non-tabular, dependent, hierarchical data로의 확장: 논문은 iid tabular prediction에 초점을 두고, 확장에는 추가 assumption과 analysis이 필요하다고 명시한다. (출처: Section 5, Limitations)
