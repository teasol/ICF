# 문헌 브리프 — pfn_foundations (로컬 모델 요약)

```
url: https://arxiv.org/abs/2305.11097
fetched_from: https://arxiv.org/html/2305.11097
fetched_at: 2026-09-18 10:32:00 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8003` · 2026-09-18 10:35 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- PFN은 관찰된 학습셋에 네트워크를 학습시키는 대신, 다양한 태스크에서 시뮬레이션된 작은 학습셋으로 고정 모델을 오프라인 사전학습한 뒤, 새로운 학습셋을 in-context로 받아 클래스 확률을 추정한다. (출처: Abstract, 1.1절)
- PFN은 사전학습된 고정 네트워크가 추론 단계에서 파라미터 업데이트 없이 단일 forward pass로 예측한다. (출처: 1.1절, 2.3절 Remark 2.4)
- TabPFN는 n≈1000까지의 시뮬레이션 데이터셋으로 사전학습되었고, 추론 시 1000개보다 큰 데이터셋을 주면 정확도가 계속 개선된다고 보고된다. (출처: 1.2절, 4절)
- PFN의 행동은 베이지안 PPD 근사보다, 사전 조정되었지만 학습되지 않은 예측자(frequentist untrained predictor)로 해석할 수 있다. (출처: Abstract, 5.1절)
- 고정 네트워크의 분산은 개별 학습 샘플에 대한 민감도가 사라지면 사라진다. (출처: Abstract, 5.3절 Theorem 5.2/Lemma 5.3)
- 편향이 사라지려면 예측자가 테스트 feature 주변으로 충분히 localized 되어야 한다. (출처: Abstract, 5.4절 Theorem 5.4)
- 현재 transformer PFN은 diminishing sensitivity로 vanishing variance를 보장하지만, Theorem 5.4 의미의 localization을 보장하지 않는다. (출처: Abstract, 6.3절)
- TabPFN의 편향은 post-hoc localization으로 더 개선될 수 있다. (출처: 1.3절, 6.4절, 6.5절)
- 사전학습 시 sample-size prior Π_N은 네트워크의 기대 복잡도에 대한 regularizer로 이해할 수 있다. (출처: 1.3절, 4절)
- 1-layer transformer PFN의 variance는 파라미터 θ와 무관하게 n^{-1/2}로 사라진다고 주장한다. (출처: 6.3절 Theorem 6.2)

## 2. 방법의 핵심
- 구조: PFN은 PPD π(y|x,D_n)을 근사하는 모델 q_θ이며, Hollmann et al.의 TabPFN는 transformer를 q_θ로 사용해 고정 네트워크가 임의 길이의 학습셋과 테스트 feature를 받아 클래스 확률을 출력한다. (출처: 2.2절, 2.3절 Remark 2.4)
- 학습 방식: prior Π와 sample-size prior Π_N에서 시뮬레이션 데이터셋을 생성하고, expected log-likelihood를 Monte-Carlo로 최대화해 θ를 사전학습한다; TabPFN는 Π_N을 {1,...,1023} 균등분포로 사용했다. (출처: 2.3절, 4절)
- 추론 방식: 학습된 θ를 고정하고, 새로운 학습셋 D_n과 테스트 feature x를 넣어 단일 forward pass로 q_θ(y|x,D_n)을 계산한다. (출처: 1.1절, 2.3절)
- 재현 관련 구체 설정: 6.5절 수치 검증은 Hollmann et al. TabPFN pip version 0.1.8을 사용하고, p0(1|X)=1/2+sin(1^T X)/2, Y∈{0,1}, X~N(0,I_5)에서 500개 데이터셋을 시뮬레이션하며, 테스트 샘플 100개로 average squared bias/variance를 계산한다. (출처: 6.5절)
- localized 변형: 테스트 feature x의 k_n nearest neighbors만 남긴 축소 학습셋 D_n(x)로 예측하며, 6.5절에서는 k_n=min{500, ceil(n^{4/(d+4)})}를 사용했다. (출처: 6.4절, 6.5절)

## 3. 평가 설정
- 데이터셋: 본문에 구체적인 실데이터셋 이름은 확인 못함. 1.2절에서는 'moderately large tabular data sets'와 'several benchmarks'만 언급하고, 6.5절의 평가는 p0(1|X)=1/2+sin(1^T X)/2, Y∈{0,1}, X~N(0,I_5)의 시뮬레이션 데이터 500개와 테스트 샘플 100개이다. (출처: 1.2절, 6.5절)
- 비교 기준선: 본문에서 명시된 외부 기준선 이름은 확인 못함. Figure 1에서는 pretrained TabPFN와 localized TabPFN 변형을 비교한다. (출처: 6.5절, Figure 1)
- ABMIL: 본문에서 확인 못함.
- 지표: average squared bias와 average squared variance이며, Figure 1에서 n에 따른 변화를 보고한다. (출처: 6.5절, Figure 1)

## 4. 정량 결과
- 표 번호가 있는 정량 결과: 본문에서 확인 못함.
- Figure 1 관련 본문 수치(표가 아님): 500개 시뮬레이션 데이터셋, 100개 테스트 샘플, d=5, n≈1000, k_n=min{500, ceil(n^{4/(d+4)})}. (출처: 6.5절, Figure 1)
- Figure 1에서 variance는 1/n 비율로 감소하고, bias는 n≈1000까지 감소한 뒤 사라지지 않는다고 본문이 서술한다. (출처: 6.5절, Figure 1)
- localized TabPFN은 n=1000을 넘어 bias가 계속 감소하지만, variance가 약간 커진다고 본문이 서술한다. (출처: 6.5절, Figure 1)
- TabPFN 사전학습 sample-size prior는 {1,...,1023} 균등분포이다. (출처: 4절)

## 5. 우리 프로젝트와의 관련 (가설)
- 옮겨올 수 있는 것 (가설): PFN의 고정 in-context 예측자 관점은 우리 closed-form ridge fold context 분류기를 bias/variance 분해로 진단하는 틀로 사용할 수 있을 수 있다. (출처: 5.1절)
- 옮겨올 수 있는 것 (가설): 브랜치 중복 문제는 paper의 multi-head attention이 여러 aspect/submodel을 선택·평균하는 메커니즘과 유사하게, 브랜치를 submodel ensemble로 보고 가중 평균하면 중복의 해가 될 수 있을 수 있다. 단, paper의 BIC ensemble은 classification tree에서 제시된 것이라 우리 ridge로 직접 이관은 확인 못함. (출처: 6.2절, 6.3절)
- 옮겨올 수 있는 것 (가설): 확률적 적합의 적합 분산 문제는, paper의 symmetrization과 diminishing sensitivity가 variance를 줄이는 구조적 조건을 제시하므로, patch/브랜치 출력에 permutation symmetrization 또는 개별 patch 민감도를 낮추는 평균화를 시도하면 분산이 줄 수 있을 수 있다. 단, paper의 bound는 고정 transformer/일정 조건 하의 결과라 우리 ridge에 자동 적용된다고 단정할 수 없다. (출처: 5.2절, 5.3절, Theorem 6.2)
- 옮겨올 수 있는 것 (가설): WSI에서 query patch 주변 spatial neighborhood만 context로 제한하는 post-hoc localization은 bias를 줄일 수 있을 수 있다. 단, paper는 localization이 sample size 감소로 variance를 약간 키울 수 있다고 했으므로, 우리 fold context에서는 편향 감소와 분산 증가의 trade-off가 생길 수 있다. (출처: 6.4절, 6.5절)
- 옮길 수 없는 것/이유: PFN은 prior Π와 sample-size prior Π_N에서 시뮬레이션 태스크를 만들어 오프라인 사전학습해야 하는데, 본문에는 우리 병리 태스크 prior나 학습 기반 MIL 기준선 대비 성능이 없다. (출처: 2.3절, 4절, 6.5절)
- 옮길 수 없는 것/이유: transformer PFN은 sample 수에 대해 quadratic scaling과 고정 feature size 제한이 있고, 본문은 WSI 규모의 patch 수나 MIL 구조를 다루지 않는다. (출처: 4절, 7절)
- 옮길 수 없는 것/이유: paper의 정량 검증은 5차원 tabular 시뮬레이션과 TabPFN bias/variance에 한정되어, 우리 브랜치 중복이나 확률적 적합 분산 문제를 직접 해결하는 수치를 제공하지 않는다. (출처: 6.5절)

## 6. 이 논문이 답하지 않는 것
- WSI 벤치마크에서 PFN 또는 localized PFN이 학습 기반 MIL 기준선을 넘을 수 있는지 본문에서 확인 못함.
- 병리 WSI patch에 맞는 prior Π, sample-size prior Π_N, feature dimension, patch 수 규모 설정은 본문에서 확인 못함.
- 우리 closed-form ridge 브랜치의 유효 랭크 중복을 줄이는 구체적 방법이나 수치 결과는 본문에서 확인 못함.
- 확률적 적합의 적합 분산이 승격 기준보다 커지는 문제를 PFN 메커니즘으로 줄일 수 있는지 직접 검증은 본문에서 확인 못함.
- 추론 시 사전학습 크기보다 훨씬 큰 WSI context에서 bias/variance가 어떻게 변하는지 본문에서 확인 못함.
- 학습 기반 MIL 기준선, MIL, pathology 데이터셋, accuracy/AUC 등 실제 분류 성능 지표 결과는 본문에서 확인 못함.
