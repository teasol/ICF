# 문헌 브리프 — pfn_imbalance (로컬 모델 요약)

```
url: https://arxiv.org/abs/2605.21742
fetched_from: https://arxiv.org/html/2605.21742
fetched_at: 2026-09-18 10:32:00 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8003` · 2026-09-18 10:35 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- PFN은 표분류에서 우수하지만 클래스 불균형에서 희귀 클래스 성능이 떨어진다 (출처: Abstract, Introduction).
- PFN의 in-context learning 특성 때문에 loss-based 전략은 불가능하고, 다른 기법은 검증되지 않았다 (출처: Abstract, Introduction).
- 저자들은 thresholding, downsampling, oversampling, synthetic upsampling을 PFN 분류에 적용했다 (출처: Abstract, Introduction, Section II-E, Section II-F).
- thresholding은 PFN의 calibration 특성 때문에 매우 잘 작동한다 (출처: Abstract, Section III-A, Section III-C).
- downsampling은 성능이 비슷하고 추론 계산 비용을 줄인다 (출처: Abstract, Section III-D, Section III-E).
- TabPFN은 balanced context에서는 잘 calibration되고, imbalanced context에서는 majority class bias를 보인다 (출처: Section III-A, Figure 2).
- binary에서 최대 balanced accuracy는 대략 τ=π1에서 나타난다 (출처: Section II-F, Section III-C, Figure 4).
- downsampling은 majority sample이 minority보다 많거나 같을 때 balanced accuracy가 대략 일정하고, majority가 증가하면 WCA가 감소한다 (출처: Section III-D, Figure 5).
- oversampling과 TabPFGen synthetic upsampling은 base model보다 나쁘다 (출처: Section III-E).
- 단순한 thresholding/downsampling이 구현이 쉽고 PFN 불균형 문제를 완화할 수 있다 (출처: Conclusion).

## 2. 방법의 핵심
- PFN은 synthetic data prior, 예로 structural causal models, 로 사전학습되어 context D와 query x에 대해 posterior predictive distribution P(y|x,D)를 추정한다 (출처: Introduction, Section II-A).
- 학습은 cross-entropy로 context label을 예측하는 in-context learning 형태이며, 새 태스크에서 가중치를 업데이트하지 않는다 (출처: Section II-A, Section II-B).
- 추론은 labeled context와 query를 transformer attention으로 처리해 soft score fθ(x,D)를 만들고, hard decision을 내린다 (출처: Section II-B, Section II-D, Section II-F).
- thresholding은 soft score에 임계값 τ를 적용해 1[fθ(x,D)>τ]로 결정하며, binary에서 τ=π1로 설정할 수 있다 (출처: Section II-F, Section III-C).
- downsampling은 context에서 majority 샘플을 제거해 π0=π1로 만든다 (출처: Section II-E).
- oversampling은 minority 샘플을 반복 포함해 π0=π1로 만든다 (출처: Section II-E).
- synthetic upsampling은 context에서 minority artificial samples를 생성해 class-balanced context를 만든다 (출처: Section II-E).
- 실험에서는 TabPFN-2.5를 사용하고, Table II 결과는 context size N∈{100,500,1000}과 imbalance π1∈{0.05,0.1,0.2,0.3}에 대한 평균이다 (출처: Section III, Table II).
- TabPFN query computation은 context size에 대해 quadratic하게 증가한다고 인용한다 (출처: Section III-D).

## 3. 평가 설정
- 데이터셋: OpenML-CC18에서 binary classification 11개를 선택했다 (출처: Section III, Table I).
- Table I의 11개 데이터셋: kr-vs-kp (N=3,196, π1=0.478), spambase (N=4,601, π1=0.394), electricity (N=45,312, π1=0.424), jm1 (N=10,885, π1=0.194), adult (N=48,842, π1=0.239), Bioresponse (N=3,751, π1=0.458), phoneme (N=5,404, π1=0.293), nomao (N=34,465, π1=0.286), PhishingWebsites (N=11,055, π1=0.443), bank-marketing (N=45,211, π1=0.117), numerai28.6 (N=96,320, π1=0.495) (출처: Table I).
- 선택 조건: binary classification, test size 500 examples per class, train size 500 minority examples and 950 majority examples로 기재되어 있다 (출처: Section III).
- 모델: TabPFN-2.5 (출처: Section III).
- 비교 기준선: None, Thresholding, OS, TabPFGen, DS (출처: Table II, Section III-E).
- ABMIL은 본문에서 확인 못함.
- 지표: per-class accuracy, balanced accuracy, worst-class accuracy(WCA), hold-out test set; ROC와 calibration도 분석한다 (출처: Section II-G, Section III-A, Section III-B).
- Table II는 N∈{100,500,1000}과 π1∈{0.05,0.1,0.2,0.3}에 대해 평균한다 (출처: Table II).

## 4. 정량 결과
- Table II의 dataset별 balanced/WCA 평균은 다음과 같다 (출처: Table II).

| Dataset | None Bal/WCA | Thrsh. Bal/WCA | OS Bal/WCA | TabPFGen Bal/WCA | DS Bal/WCA |
|---|---:|---:|---:|---:|---:|
| kr-vs-kp | 0.931 / 0.868 | 0.956 / 0.938 | 0.783 / 0.569 | 0.912 / 0.831 | 0.928 / 0.902 |
| spambase | 0.866 / 0.753 | 0.920 / 0.901 | 0.700 / 0.408 | 0.789 / 0.593 | 0.903 / 0.876 |
| electricity | 0.645 / 0.317 | 0.757 / 0.678 | 0.574 / 0.168 | 0.638 / 0.305 | 0.734 / 0.677 |
| jm1 | 0.534 / 0.096 | 0.637 / 0.537 | 0.529 / 0.102 | 0.539 / 0.113 | 0.632 / 0.560 |
| adult | 0.665 / 0.372 | 0.795 / 0.752 | 0.578 / 0.179 | 0.664 / 0.371 | 0.783 / 0.726 |
| Bioresponse | 0.588 / 0.212 | 0.704 / 0.646 | 0.539 / 0.094 | 0.585 / 0.204 | 0.683 / 0.643 |
| phoneme | 0.666 / 0.379 | 0.811 / 0.770 | 0.602 / 0.230 | 0.629 / 0.292 | 0.791 / 0.741 |
| nomao | 0.867 / 0.761 | 0.915 / 0.897 | 0.715 / 0.438 | 0.821 / 0.660 | 0.905 / 0.884 |
| PhishingWebsites | 0.882 / 0.779 | 0.921 / 0.893 | 0.758 / 0.525 | 0.872 / 0.757 | 0.913 / 0.889 |
| bank-marketing | 0.664 / 0.373 | 0.808 / 0.774 | 0.555 / 0.126 | 0.652 / 0.346 | 0.789 / 0.753 |
| numerai28.6 | 0.500 / 0.001 | 0.504 / 0.235 | 0.500 / 0.007 | 0.500 / 0.002 | 0.503 / 0.452 |
| Average | 0.710 / 0.446 | 0.793 / 0.729 | 0.621 / 0.259 | 0.691 / 0.407 | 0.779 / 0.737 |

- Figure 3에서 kr-vs-kp, N=25, π1=0.1, τ=0.5일 때 minority misdetection rate는 MD≈0.8이다 (출처: Figure 3).
- Figure 4에서 kr-vs-kp, N=500, π1∈{0.05,0.1,0.2,0.5}에서 최대 balanced accuracy가 τ=π1을 따른다 (출처: Figure 4).
- Figure 5에서 kr-vs-kp, Nmin=50으로 downsampling crossover를 보여준다 (출처: Figure 5).
- Figure 6에서 thresholding과 downsampling은 minority accuracy를 크게 올리고 majority accuracy를 약간 낮추며, downsampling은 minority 증가가 더 크지만 majority 감소도 더 크다 (출처: Figure 6).

## 5. 우리 프로젝트와의 관련 (가설)
- 가설: 우리 fold context in-context 분류기의 soft score가 이 논문의 TabPFN처럼 prior-driven bias만 보인다면, minority prior에 맞춘 thresholding이 rare-class 성능이나 WCA를 개선할 수 있다. 단, WSI/MIL에서는 calibration이 다를 수 있다.
- 가설: majority context를 downsample하면 context 중복과 추론 비용을 줄일 수 있고, 브랜치 간 유효 랭크 중복이 majority context의 반복에서 온다면 랭크 중복 완화와 연결될 수 있다. 단, 이 논문은 effective rank를 측정하지 않았다.
- 가설: 확률적 적합 대신 결정론적 thresholding/downsampling을 쓰면 적합 분산 증가를 피할 수 있을 수 있다. 단, 이 논문은 fitting variance를 측정하지 않았다.
- 가설: WCA, calibration curve, ROC diagnostics를 우리 평가에 추가하면 희귀 클래스 성능 저하를 더 잘 진단할 수 있다.
- 옮기기 어려운 점: PFN의 synthetic prior, TabPFN-2.5, tabular feature, binary classification 설정은 WSI patch-level MIL과 직접 동일하지 않다.
- 옮기기 어려운 점: 이 논문은 patch aggregation, slide-level label, spatial context, WSI benchmark를 다루지 않는다.
- 옮기기 어려운 점: multi-class 또는 hierarchical label 문제에서는 binary τ=π1 규칙이 직접 적용되지 않을 수 있다.

## 6. 이 논문이 답하지 않는 것
- 병리 WSI benchmark에서 thresholding/downsampling이 학습 기반 MIL 기준선을 넘게 하는지: 본문에서 확인 못함.
- ABMIL과 직접 비교: 본문에서 확인 못함.
- 브랜치 간 유효 랭크 중복을 줄이는지: 본문에서 확인 못함.
- 확률적 적합의 분산을 줄이는지: 본문에서 확인 못함.
- multi-class WSI 분류에 어떻게 확장되는지: 본문에서 확인 못함.
- patch-level MIL aggregation, slide-level label, spatial context에 효과가 있는지: 본문에서 확인 못함.
- closed-form ridge in-context 분류기에서 같은 calibration/thresholding 성립이 있는지: 본문에서 확인 못함.
- context downsampling이 effective rank나 브랜치 중복에 미치는 정량 효과: 본문에서 확인 못함.
