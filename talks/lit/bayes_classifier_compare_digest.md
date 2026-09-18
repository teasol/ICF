# 문헌 브리프 — bayes_classifier_compare (로컬 모델 요약)

```
url: https://arxiv.org/abs/1609.08905
fetched_from: https://arxiv.org/html/1609.08905
fetched_at: 2026-09-18 13:01:10 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8001` · 2026-09-18 13:10 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 두 분류기의 교차검증 결과를 여러 데이터셋에서 동시에 분석하는 베이지안 계층 모델을 제안한다 (출처: Abstract, 1).
- 모델은 두 분류기의 정확도 차이가 실질적으로 동등하거나 유의하게 다른지 posterior probability를 반환한다 (출처: Abstract, 3.1).
- 기존 NHST는 관찰된 결과 하에서 한 분류기가 더 정확한지 확률을 계산하지 못하고, 실질적 유의성과 통계적 유의성을 구분하지 못하며, 귀무가설을 지지하는 증거를 제공하지 못한다 (출처: 1).
- 제안 모델은 데이터셋 간 실제 평균 차이 δ_i의 분포와 각 데이터셋 내 교차검증 결과의 분산 및 상관을 함께 모델링한다 (출처: 1, 3).
- 실질적 동등 영역(rope) 기본값은 정확도 차이가 (-0.01, 0.01)인 경우이다 (출처: 1, 3.1, 3.2).
- 계층 검정은 signed-rank보다 보수적이어서 귀무가설을 덜 기각하고, Type I 오류는 적지만 power는 낮을 수 있다 (출처: 1, 4.3, 4.4, 5).
- shrinkage 추정자는 상관 있는 교차검증 상황에서도 MLE보다 MSE가 낮다고 이론적으로 증명하고 실험으로 확인한다 (출처: 1, 3.3, 4.1, 4.7).
- 95% 임계값을 넘지 않아도 posterior odds로 의미 있는 결론을 도출할 수 있다 (출처: 4.6, 5).
- δ_i의 상위 분포로 Student-t를 쓰면 Gaussian보다 유연하고 outlier에 robust하며, 실험에서 더 좋은 fit을 제공한다고 보고한다 (출처: 3, 4.8, 5).
- Stan 구현으로 50개 데이터셋의 10회 10-fold CV 결과, 즉 5000 관측에 대한 추론이 표준 노트북에서 약 3분 소요된다 (출처: 3.4).

## 2. 방법의 핵심
- 분석 대상은 두 분류기의 paired k-fold CV 정확도 차이이며, m회 반복 시 n = mk 관측을 가진다 (출처: 2).
- 분류기 학습 자체는 이 논문의 대상이 아니고, 두 분류기의 CV 결과를 입력으로 받아 베이지안 계층 모델을 Stan으로 피팅한다 (출처: 3, 4).
- 계층 구조는 δ_i ~ Student-t(δ_0, σ_0, ν), σ_i ~ Uniform(0, σbar), x_i ~ MVN(1δ_i, Σ_i)이다 (출처: 3, Eq. 2-4).
- 교차검증 관측은 동일한 평균 δ_i, 동일한 표준편차 σ_i, 동일한 상관 ρ를 가진다고 가정하며, ρ = 1/k 근사는 Nadeau & Bengio(2003)에서 빌려온다 (출처: 2, 3).
- 선험은 δ_0 ~ Uniform(-1,1), σ_0 ~ Uniform(0, 1000 s_xbar), ν ~ Gamma(α, β), α ~ Uniform(0.5, 5), β ~ Uniform(0.05, 0.15), σbar = 1000 sbar이다 (출처: 3, Table 1).
- δ_0의 Uniform(-1,1) 선험은 정확도, AUC, precision, recall처럼 ±1로 제한된 지표에 적합하다고 서술한다 (출처: 3).
- ν prior에 민감할 수 있어 α와 β를 추가 확률변수로 두는 hierarchical prior를 사용한다 (출처: 3, 4.8).
- 추론은 Stan posterior에서 μ_0, σ_0, ν를 샘플링하고, 다음 데이터셋의 δ_next ~ t(δ_0, σ_0, ν)에서 left/rope/right 확률을 적분한다 (출처: 3.2).
- rope 반경 r = 0.01을 사용하고, N_s = 4000 샘플에서 left/rope/right 중 최대 확률 카운트를 계산한다 (출처: 3.2).
- 결정 규칙은 P(rope) > 1 - α이면 실질 동등, P(left) > 1 - α 또는 P(right) > 1 - α이면 유의 차이로 선언하는 것이다 (출처: 3.2).
- posterior odds 해석 기준은 1-3 weak, 3-20 positive, >20 strong이다 (출처: Table 3).
- shrinkage 추정자는 δhat_i = w xbar_i + (1 - w) (1/q)Σxbar_i이며, w = σhat_0^2 / (σhat_0^2 + σ_n^2)이다 (출처: 3.3, Eq. 8).
- 재현 관련 구체 설정은 실험에서 10회 10-fold CV, paired folds, Stan 코드 공개, likelihood를 quadratic matrix form으로 계산해 naive 대비 약 10배 speedup, 50개 데이터셋 5000 관측 시 약 3분 소요이다 (출처: 3.4, 4).
- σ_i와 σ_0 prior 상한을 1000배에서 100배로 바꿔도 posterior가 동일하다고 보고한다 (출처: 4.8.1).

## 3. 평가 설정
- 데이터셋: 알려진 정확도 차이 δ_i를 가진 synthetic naive Bayes two-feature 분류기 (출처: Appendix 6.1).
- 데이터셋: Friedman F#1-F#3 기반 54개 설정 (출처: 4.7, Table 4).
- F#1은 10개 uniform [0,1] feature 중 5개 사용, y = 10sin(πx1x2) + 20(x3 - 0.5)^2 + 10x4 + 5x5 + ε1, ε1 ~ N(0,1), y를 median으로 2분류화하며 median은 10,000 샘플로 추정 (출처: 4.7).
- F#2/F#3는 4개 feature ranges 0 ≤ x1 ≤ 100, 40π ≤ x2 ≤ 560π, 0 ≤ x3 ≤ 1, 1 ≤ x4 ≤ 11을 사용하며, F#2 y = (x1^2 + (x2x3 - (1/x2x4))^2)^0.5 + ε2, F#3 y = arctan((x2x3 - (1/x2x4))/x1) + ε3, ε2 ~ N(0, σ_ε2^2), ε3 ~ N(0, σ_ε3^2), original σ_ε2 = 125, σ_ε3 = 0.1, median으로 2분류화 (출처: 4.7).
- Friedman 설정은 함수당 18개씩 총 54개이며, F#1 σ_ε {0.5,1,2}, F#2 {62.5,125,250}, F#3 {0.05,0.1,0.2}, n {30,100,1000}, random features {0,20} (출처: Table 4).
- Friedman 실험의 분류기 쌍은 lda와 cart이며, caret R 패키지, hyper-parameter tuning 없음 (출처: 4.7).
- Friedman ground truth δ_i 측정은 500회 반복, 회당 large test set 5000 instances를 사용해 cart-lda 정확도 차이를 평균한다 (출처: 4.7).
- Friedman 평가는 200회 반복, 함수당 18개 설정 중 12개씩 총 36개 설정을 무작위 선택, 각 설정에서 10회 10-fold CV paired folds를 수행 (출처: 4.7).
- 데이터셋: WEKA 데이터셋 54개 (출처: 4.8).
- WEKA 실험의 분류기는 nbc, aode, hnb, j48, j48gr이며, 각 분류기·데이터셋마다 10회 10-fold CV, WEKA software 사용 (출처: 4.8).
- 비교 기준선: signed-rank test와 제안 계층 모델; 계층 모델 내부 비교로는 Gamma(2,0.1) prior variant와 hierarchical prior variant (출처: 4.2-4.9, Table 5, Table 6).
- ABMIL: 본문에서 확인 못함.
- 지표: left/rope/right posterior probability, posterior odds, power, Type I error, equivalence recognition, δ_i 추정 MSE, signed-rank p-value (출처: 3.2, 4.2-4.9, Tables 2-6).

## 4. 정량 결과
- Table 2 (Sec. 4.1): q = 5, 10, 50, 500 experiments, δ_i bimodal mixture μ1 = 0.005, μ2 = 0.02, σ = 0.001, π1 = π2 = 0.5, 10회 10-fold CV. MSE는 q = 5 MLE 0.00036 / Shrinkage 0.00017, q = 10 MLE 0.00036 / Shrinkage 0.00014, q = 50 MLE 0.00036 / Shrinkage 0.00012. 본문은 MSE_SHR이 일반적으로 MLE보다 50% 낮고, MSE_SHR은 q 증가 시 감소하지만 MLE는 q에 따라 변하지 않는다고 서술 (출처: Table 2, Sec. 4.1).
- Fig. 2 (Sec. 4.2): 동등 분류기 시뮬레이션, Cauchy scale factor rope length의 1/6, 80% δ_i가 rope 내, q = {10,20,30,40,50}, 500 experiments. q = 50일 때 평균 p(rope) > 90%, p(rope) > 95% 비율 약 0.7. signed-rank α = 0.05 기각 약 5%. 계층 모델은 p(left) > 95% 또는 p(right) > 95%를 추정하지 않아 Type I 오류 없음 (출처: Fig. 2, Sec. 4.2).
- Fig. 3/4 (Sec. 4.3): 실질 동등 시뮬레이션, δ_0 = 0.005, q = {10,20,30,40,50}, 500 experiments. q = 50에서 signed-rank 기각 약 25%, 계층 모델의 95% 동등 판정 약 40%. 계층 모델은 p(left/right) > 95%를 추정하지 않음 (출처: Fig. 3, Fig. 4, Sec. 4.3).
- Fig. 5 (Sec. 4.4): 실질 차이 시뮬레이션, δ_0 = {0.015, 0.02, 0.025, 0.03}, σ_0 = 0.01, q = 50, 500 experiments. signed-rank가 더 powerful하고, δ_0 > 0.02에서 두 검정 power 유사. 정확한 power 수치는 본문에서 확인 못함 (출처: Fig. 5, Sec. 4.4).
- Sec. 4.7: Friedman 실험에서 power는 signed-rank 28%, 계층 모델 27.5%. posterior odds에서 11%는 o(right,rope)와 o(right,left) 모두 >20으로 strong evidence, 33%는 모두 >3으로 positive evidence, 2%는 rope에 대한 잘못된 positive evidence. ground truth는 δ_i의 65%가 rope 오른쪽, 평균 δ_0 = 0.02 (출처: Sec. 4.7).
- Fig. 6 (Sec. 4.7): MSE 비교에서 average reduction of about 60%라고 기재되나, 원문 문장은 “MSE_MLE is much lower than MSE_Shr”로 되어 있어 어떤 MSE가 낮아졌는지 본문 문장만으로 불명확함 (출처: Fig. 6, Sec. 4.7).
- Table 5 (Sec. 4.8): posterior probabilities, left/rope/right. nbc-hnb: hierarchical 1.00/0.00/0.00, Gamma(2,0.1) 1.00/0.00/0.00. nbc-j48: 0.80/0.02/0.18 vs 0.80/0.01/0.20. nbc-j48gr: 0.84/0.02/0.14 vs 0.84/0.01/0.15. hnb-j48: 0.03/0.10/0.87 vs 0.03/0.02/0.95. hnb-j48gr: 0.03/0.07/0.90 vs 0.03/0.02/0.95. j48-j48gr: 0.00/1.00/0.00 vs 0.00/1.00/0.00 (출처: Table 5, Sec. 4.8).
- Table 6 (Sec. 4.9): signed-rank p-values. nbc-hnb 0.00, nbc-j48 0.46, nbc-j48gr 0.39, hnb-j48 0.07, hnb-j48gr 0.08, j48-j48gr 0.00. 계층 모델 posterior는 Table 5의 hierarchical 열과 동일 (출처: Table 6, Sec. 4.9).
- j48-j48gr 비교에서 rope를 (-0.005, 0.005)로 줄여도 posterior probabilities가 불변이라고 보고한다 (출처: 4.9).

## 5. 우리 프로젝트와의 관련 (가설)
- 옮겨올 수 있는 것 (가설): 우리 in-context 분류기와 ABMIL의 성능 차이를 여러 fold/데이터셋/벤치마크에서 비교할 때, rope 기반 실질 동등 판정을 쓰면 아주 작은 차이를 “실질 동등”으로 처리할 수 있어 과대 해석을 줄일 수 있다. 단, 우리 지표가 정확도 또는 ±1로 제한된 차이인지, rope 폭을 어떻게 정할지는 확인 필요 (출처: 3.1, 3.2).
- 옮겨올 수 있는 것 (가설): fold-level 또는 dataset-level 성능 차이가 상관 있는 CV 관측이라면, 계층 shrinkage가 per-fold/per-dataset 성능 추정 MSE를 줄여 승격 기준의 변동성을 낮출 수 있다. 단, 이는 분류기 성능 추정 안정화이지 모델 구조 개선이 아니다 (출처: 3.3, Table 2).
- 옮겨올 수 있는 것 (가설): posterior odds를 쓰면 95% 임계값을 넘지 않아도 후보 방법 간 상대적 증거를 순위화할 수 있다. 단, 본문의 odds는 두 분류기의 left/right/rope 비교용이다 (출처: 4.6, Table 3).
- 옮겨올 수 있는 것 (가설): Stan 구현의 계산 비용, 즉 50개 데이터셋·5000 관측 시 약 3분이라는 보고가 우리 fold 규모에서 허용 가능할 수 있다. 단, WSI 규모와 브랜치 수에 대한 비용은 본문에 없다 (출처: 3.4).
- 옮길 수 없는 것: 본 논문은 ABMIL, MIL, 병리 WSI, 브랜치 유효 랭크 중복, 확률적 적합의 적합 분산 문제를 다루지 않는다. 따라서 우리 프로젝트의 브랜치 중복이나 확률적 적합으로 인한 적합 분산 증가를 직접 해결하는 방법은 본문에서 확인 못함.
- 옮길 수 없는 것: 본 방법은 두 분류기 비교이며, 여러 브랜치를 동시에 선택하거나 closed-form ridge in-context classifier 구조를 제공하지 않는다 (출처: 1, 3).

## 6. 이 논문이 답하지 않는 것
- 우리 in-context 분류기가 ABMIL을 실제로 넘어서는지: 본문에서 확인 못함.
- 브랜치 유효 랭크 중복을 줄이는 방법: 본문에서 확인 못함.
- 확률적 적합으로 인한 적합 분산 증가를 승격 기준에 맞게 완화하는 방법: 본문에서 확인 못함.
- 병리 WSI 벤치마크, MIL, ABMIL에 대한 데이터셋·기준선·수치: 본문에서 확인 못함.
- 계층 shrinkage를 closed-form ridge in-context classifier의 fold context에 적용할 때의 구체적 설계: 본문에서 확인 못함.
- 여러 브랜치를 동시에 비교하거나 model selection하는 절차: 본문에서 확인 못함.
- 우리 지표에 맞는 rope 폭, prior, Student-t 가정의 적절성: 본문에서 확인 못함.
