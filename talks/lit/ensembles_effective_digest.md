# 문헌 브리프 — ensembles_effective (로컬 모델 요약)

```
url: https://arxiv.org/abs/2305.12313
fetched_from: https://arxiv.org/html/2305.12313
fetched_at: 2026-09-18 10:50:10 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8001` · 2026-09-18 12:48 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 앙상블의 이점은 보편적이지 않으며, 분류 작업에서 앙상블이 언제 유의한 성능 향상을 주는지를 이론과 실험으로 분석한다. (출처: Abstract, Section 1)
- ensemble improvement rate(EIR)를 정의하고, competence 조건 하에서 majority vote classifier의 평균 테스트 에러 상·하한을 개선한다. (출처: Abstract, Section 3)
- competent ensemble은 성능을 악화시키지 않으며, EIR ≥ 0이다. (출처: Section 3.1, Theorem 1)
- EIR은 disagreement-error ratio(DER)의 선형 함수로 상·하한을 가지며, DER가 평균 에러율보다 상대적으로 크면 앙상블 개선이 크다. (출처: Section 3.2, Theorem 2)
- DER < 1이면 앙상블 개선은 크지 않고, DER > 1이면 앙상블 개선이 보장되는 영역이다. (출처: Section 3.2)
- 실험에서 competence 조건이 다양한 앙상블에서 널리 성립하고, EIR과 DER는 선형 관계를 보인다. (출처: Section 4, Figure 1, Figure 2)
- interpolating ensemble은 non-interpolating ensemble보다 EIR과 DER가 낮으며, 특히 DER < 1 영역에서 앙상블 이점이 작다. (출처: Section 5, Figure 5)
- tree-based ensemble, 특히 random forest는 interpolation threshold 이후에도 DER/EIR이 유지되는 예외적 앙상블이다. (출처: Section 5, Figure 4)
- Bayesian ensemble은 DER > 1과 높은 EIR을 보이며, fine-tuned BERT ensemble은 DER < 1과 낮은 EIR을 보인다. (출처: Section 5, Figure 6, Figure 7)
- DER는 interpolation threshold 근처에서 phase-transition 같은 급격한 변화를 보일 수 있다. (출처: Section 5, Figure 5)

## 2. 방법의 핵심
- 분석 대상은 classifier 분포 ρ에서 정의되는 majority vote classifier h_MV이며, K-class classification 설정을 사용한다. (출처: Section 2.1, Definition 1)
- ρ는 가중치 있는 유한 classifier 집합, stochastic optimization의 parameter 분포, Bayesian posterior 등으로 해석할 수 있다. (출처: Section 2.1)
- 핵심 지표는 EIR = (평균 단일 분류기 에러 − majority vote 에러) / 평균 단일 분류기 에러이고, DER = 기대 disagreement rate / 평균 에러율이다. (출처: Section 3, Definition 3, Definition 4)
- competence 조건은 majority vote가 병리적으로 나빠지는 경우를 제거하는 mild condition이며, 이 조건에서 EIR ≥ 0이 성립한다. (출처: Section 3.1, Assumption 1, Theorem 1)
- K-class competent ensemble에 대해 DER ≥ EIR ≥ 2(K−1)/K · DER − (3K−4)/K가 성립한다. (출처: Section 3.2, Theorem 2)
- competence 검증은 hold-out/test 데이터에서 각 샘플에 대해 N개 classifier의 오류 비율 W_hat을 계산하고, 경험 CDF로 Assumption 1의 두 확률을 비교하는 방식으로 수행된다. (출처: Section 4.2)
- 추론 방식은 입력 x에 대해 ρ에서 뽑은 classifier들이 각 클래스를 예측할 확률의 최대값을 예측하는 majority vote이다. (출처: Section 2.1, Definition 1)
- deep ensemble은 독립 초기화로 학습된 neural network ensemble을 사용하고, Bayesian ensemble은 Hamiltonian Monte Carlo로 얻은 posterior 샘플을 사용한다. (출처: Table 1, Section 5, Appendix B.1)
- random feature bagging은 무작위 ReLU 특징 행렬 U를 만들고, 그 특징으로 multi-class logistic regression을 scikit-learn으로 적합하며, training data를 bootstrap sampling한다. (출처: Appendix B.1)
- random forest는 scikit-learn decision trees를 사용하고, max number of leaf nodes를 변화시켜 model complexity를 조절한다. (출처: Appendix B.1)
- ResNet18/CIFAR-10 deep ensemble은 SGD momentum 0.9, weight decay 5e-4, learning rate 0.1, 100 epochs로 학습하고, batch size와 width를 변화시키며, LR decay variant는 75 epochs 후 learning rate를 0.01로 낮춘다. (출처: Appendix B.1)
- BERT ensemble은 25개 pre-trained BERT 모델을 GLUE classification tasks에 fine-tune한다. (출처: Section 5, Appendix B.1)
- bagging 계열의 training error는 in-bag training error로 정의된다. (출처: Section 5)

## 3. 평가 설정
- 데이터셋: MNIST (5K subset), CIFAR-10, IMDB, QSAR, Thyroid, GLUE (7 tasks). (출처: Table 1)
- 클래스 수: MNIST 10, CIFAR-10 10, IMDB 2, QSAR 2, Thyroid 2, GLUE 2-3. (출처: Table 1)
- 앙상블/베이스 분류기: ResNet20-Swish Bayesian Ens. M=100, ResNet18 Deep Ens. M=5, CNN-LSTM Bayesian Ens. M=100, BERT fine-tune Deep Ens. M=25, Random Features Bagging M=30, Decision Trees Random Forests M=100. (출처: Table 1)
- 추가 OOD 평가 데이터셋: CIFAR-10.1, CIFAR-10-C. (출처: Appendix B.1)
- 지표: EIR, DER, average error, majority vote error, competence 확률, Pearson correlation. (출처: Section 3, Section 4, Figure 2)
- 비교 기준선: 단일 분류기 평균 에러와 majority vote ensemble 에러 비교; 이론적 비교로는 C-bound, 이전 bound. (출처: Section 2.2, Appendix A.3)
- ABMIL 기준선은 본문에서 확인 못함. (출처: 전체 본문)

## 4. 정량 결과
- Table 1: MNIST (5K subset) C=10, CIFAR-10 C=10, IMDB C=2, QSAR C=2, Thyroid C=2, GLUE C=2-3; ensemble sizes M=100, 5, 100, 25, 30, 100. (출처: Table 1)
- Figure 2: 8개 실험 설정 중 6개에서 EIR-DER Pearson R ≥ 0.96; Thyroid는 R ≈ 0.8. (출처: Figure 2, Section 4.3)
- Section 4.3: binary QSAR/Thyroid best fit은 EIR ≈ DER − 1; 10-class MNIST/CIFAR-10 best fit은 EIR ≈ 1.8 DER − 2.6. (출처: Section 4.3)
- Section 5: BERT GLUE test set size는 250 (RTE)에서 40,000 (QQP) 범위이며, BERT 모델 수는 25개. (출처: Section 5)
- Appendix B.1: ResNet18/CIFAR-10은 100 epochs, momentum 0.9, weight decay 5e-4, learning rate 0.1, LR decay variant는 75 epochs 후 0.01, ensemble size 5. (출처: Appendix B.1)
- Appendix B.1: random feature competence plot은 500 random features, M=100 classifiers; random forest는 20 trees. (출처: Appendix B.1)
- Appendix B.1: MNIST는 5000 train examples, class당 500; QSAR는 7.2k train, 1.8k test, 1024 features; Thyroid는 2.5k train, 633 test, 21 features. (출처: Appendix B.1)
- Theorem 2: K-class bound의 계수는 2(K−1)/K와 (3K−4)/K이며, non-trivial lower bound threshold는 DER ≥ (3K−4)/(2K−2)이다. (출처: Section 3.2, Theorem 2)
- Appendix A.3: K=2에서 본 논문 bound는 2E[L(h)] − E[D(h,h′)]이며, 이전 binary bound 4E[L(h)] − 2E[D(h,h′)]보다 factor 2 개선된다. (출처: Appendix A.3)

## 5. 우리 프로젝트와의 관련 (가설)
- 가설: 우리 fold context의 여러 브랜치를 ρ에서 뽑은 classifier로 본다면, 브랜치 간 중복이 크다는 것은 DER가 낮을 가능성을 시사하고, DER < 1이면 majority vote/평균화 이점이 작을 수 있다. (출처: Section 3.2, Section 5)
- 가설: closed-form ridge in-context classifier가 training error를 0에 가깝게 만들면 interpolating regime에 가까워져 EIR이 낮아질 수 있다. (출처: Section 5, Figure 5)
- 가설: probabilistic fitting이 브랜치 다양성을 높여 disagreement를 증가시키더라도, 평균 에러율에 비해 상대적으로 작으면 DER < 1로 남아 앙상블 이점이 제한적일 수 있다. (출처: Section 3.2)
- 가설: competence 조건을 hold-out fold에서 점검하면, 브랜치 앙상블이 평균보다 나빠지는 병리적 경우를 걸러낼 수 있다. (출처: Section 3.1, Section 4.2)
- 가설적으로 옮길 수 있는 것: EIR/DER 계산, competence 검증, interpolating threshold 진단, DER > 1 여부로 앙상블 도입 가치 판단. (출처: Section 3, Section 4, Section 5)
- 옮기기 어려운 것: 이 논문은 WSI, MIL, 학습 기반 MIL 기준선, effective rank, ridge in-context, probabilistic fitting variance, promotion criterion을 다루지 않아 직접 이식은 어렵다. (출처: 전체 본문)

## 6. 이 논문이 답하지 않는 것
- WSI 벤치마크에서 이 논문의 앙상블 지표가 학습 기반 MIL 기준선을 넘는지에 대한 답은 본문에서 확인 못함.
- 브랜치 중복/유효 랭크를 줄이는 구체적 방법은 본문에서 확인 못함.
- probabilistic fitting의 적합 분산과 promotion criterion tradeoff는 본문에서 확인 못함.
- closed-form ridge in-context classifier를 majority vote/weighted vote로 결합하는 방식은 본문에서 확인 못함.
- fold context/cross-validation 기반 평가 프로토콜은 본문에서 확인 못함.
- effective rank와 DER/EIR의 관계는 본문에서 확인 못함.
