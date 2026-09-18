# 문헌 브리프 — variance_benchmarks (로컬 모델 요약)

```
url: https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf
fetched_from: https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf (PDF)
fetched_at: 2026-09-18 10:50:10 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8001` · 2026-09-18 10:54 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 벤치마크 성능은 데이터 샘플링, 증강, 파라미터 초기화, 하이퍼파라미터 선택 등 여러 변동 원인의 실현값이며, 알고리즘 비교는 이 분산을 고려해야 한다 (출처: Abstract, Section 2).
- 데이터 샘플링, 파라미터 초기화, 하이퍼파라미터 선택은 벤치마크 결과에 뚜렷한 영향을 준다 (출처: Abstract, Section 2.2).
- 불완전 추정자에 더 많은 변동 원인을 무작위화하면 이상적 추정자에 더 가까워질 수 있고, 계산 비용은 51× 감소할 수 있다 (출처: Abstract, Section 3.3).
- 평균 차이 비교보다 `P(A > B)` 기반 비교가 거짓 탐지와 누락 탐지를 더 균형 있게 통제한다 (출처: Section 4.1, Section 4.2, Figure 6).
- 가능한 한 많은 변동 원인을 무작위화하고, 고정 테스트셋보다 여러 무작위 분할을 사용하며, 결론에 분산을 반영할 것을 권고한다 (출처: Section 5).

## 2. 방법의 핵심
- 구조: 전체 벤치마크 과정을 확률 모델로 표현한다. 데이터셋 `S`, train/validation/test 분할, 하이퍼파라미터 최적화 `HOpt`, 학습 절차 `Opt`, 그리고 무작위 변동 원인 `ξ_H`, `ξ_O`를 명시하고, 성능을 기대 경험 리스크의 실현값으로 다룬다 (출처: Section 2.1, Equation 1–5).
- 학습 방식: 단일 모델을 제안하지 않고, 사례 연구의 학습 파이프라인을 대상으로 한다. 사례 연구로는 CIFAR10-VGG11, PascalVOC-FCN/ResNet18, GLUE SST-2/RTE-BERT, MHC peptide binding-shallow MLP를 사용한다 (출처: Section 2.2, Appendix D).
- 하이퍼파라미터 최적화 방식: random search, noisy grid search, Bayesian optimization을 비교한다. grid search의 임의적 범위 선택을 변동 원인으로 보기 위해 noisy grid search를 정의한다 (출처: Section 2.2, Appendix E).
- 추론/평가 방식: out-of-bootstrap으로 train/test를 생성하고, 성능을 측정한다. 이상 추정자는 각 분할마다 모든 `ξ_O`와 `ξ_H`를 무작위화하고 `HOpt`를 다시 실행한다. 편향 추정자는 `HOpt`를 1회만 실행하고 그 결과를 재사용하면서 `ξ_O`의 일부만 무작위화한다 (출처: Section 3.2, Algorithm 1, Algorithm 2, Appendix B).
- 결론 방식: 두 알고리즘 A, B의 paired empirical risks로 `P(A > B)`를 계산하고, percentile bootstrap confidence interval을 만든다. `H0: P(A > B) = 0.5`, `H1: P(A > B) = γ`로 설정하고, `CI_min > 0.5`와 `CI_max > γ`를 만족할 때 의미 있는 개선으로 판단한다 (출처: Section 4.1, Appendix C).
- 재현에 필요한 구체적 설정: 각 학습 절차 변동 원인은 seed 200회 무작위화, numerical noise는 고정 seed 200회 측정, HPO는 20회 독립 실행하고 최대 200 trials, 추정자 비교는 `k = 1...100`, 편향 추정자는 20회 반복, 시뮬레이션은 `k = 50`, 제안된 임계값 `γ = 0.75`, 평균 비교 임계값 `δ = 1.9952σ`, `γ = 0.75`와 `β = 0.05`일 때 최소 sample size 29 trainings (출처: Section 2.2, Section 3.3, Section 4.2, Appendix C, Figure C.1).
- 사례 연구 구체 설정: CIFAR10-VGG11은 learning rate default 0.03, weight decay 0.002, momentum 0.9, lr schedule γ 0.97, batch size 128 (출처: Table 2). BERT SST-2/RTE는 learning rate 2e-5, weight decay 0.0, weights init std 0.2, dropout 0.1, batch size 32, 3 epochs fine-tuning (출처: Table 3, Appendix D.2, Appendix D.3). PascalVOC는 learning rate 0.002, momentum 0.9, weight decay 1e-6, batch size 16 (출처: Table 5). MLP-MHC는 hidden layer size space `lin(20, 400)`, L2 space `log(0, 1)`, default hidden layer size 150, L2 0.001 (출처: Table 6, Table 7).

## 3. 평가 설정
- 데이터셋: CIFAR10, PascalVOC, GLUE SST-2, GLUE RTE, MHC class I peptide binding prediction (출처: Section 2.2, Appendix D).
- 데이터셋 규모: CIFAR10은 60,000개 32x32 이미지, 원래 분할은 50,000 train / 10,000 test (출처: Appendix D.1). SST-2는 약 68k entries (출처: Appendix D.2). RTE는 약 2.5k entries (출처: Appendix D.3). PascalVOC는 2,913 images, 원래 분할은 2,184 train / 729 validation (출처: Appendix D.4).
- 비교 대상/기준선: 추정자 비교로는 `IdealEst(k)`, `FixHOptEst(k, Init)`, `FixHOptEst(k, Data)`, `FixHOptEst(k, All)`를 사용한다. 결론 기준 비교로는 single point comparison, average comparison, t-test, probability of improvement를 사용한다 (출처: Section 3.3, Section 4.2, Figure 6).
- MHC task 비교 모델: NetMHCpan4, MHCflurry, MLP-MHC를 비교한다 (출처: Table 8, Table 9).
- 지표: CIFAR10, SST-2, RTE는 classification accuracy, PascalVOC는 mIoU, MHC task는 AUC와 PCC를 사용한다 (출처: Figure 5, Figure H.4, Appendix D.4, Table 8).
- HPO 목적 함수: CIFAR10/SST-2/RTE는 validation error-rate 계열, PascalVOC는 mean Jaccard Distance, 즉 mIoU의 complement를 최소화하는 방향으로 사용된다 (출처: Section 2.1, Appendix D.4, Figure F.2).
- ABMIL: 본문에서 확인 못함.

## 4. 정량 결과
- 전체 사례 연구 계산 환경은 약 8 GPU years (출처: Section 2.2).
- 각 학습 절차 변동 원인은 seed 200회 무작위화되었고, numerical noise는 고정 seed 200회로 측정되었다 (출처: Section 2.2).
- HPO는 20회 독립 실행되었고, 각 실행의 예산은 최대 200 trials였다 (출처: Section 2.2).
- 모델 초기화 분산은 bootstrap 분산의 50% 미만인 경우가 일반적이라고 보고된다 (출처: Section 2.2, Figure 1).
- 분류 사례의 test set size는 Glue-RTE `n' = 277`, Glue-SST2 `n' = 872`, CIFAR10 `n' = 10000` (출처: Figure 2).
- `IdealEst(k=100)`는 1,070 hours, `FixedHOptEst(k=100)`는 21 hours, 총 연구 비용은 6.4 GPU years (출처: Section 3.3).
- `FixHOptEst(k, Init)`은 Glue-RTE에서 `IdealEst(k=2)` 수준까지 수렴하고, `FixHOptEst(k, Data)`는 `IdealEst(k=2)`~`IdealEst(k=10)` 수준, `FixHOptEst(k, All)`는 `IdealEst(k=2)`~`IdealEst(k=100)` 수준까지 개선된다고 보고된다 (출처: Section 3.3, Figure 5).
- 시뮬레이션은 `k = 50` data splits를 사용하고, 평균 비교 임계값은 `δ = 1.9952σ`, probability of improvement 임계값은 `γ = 0.75` (출처: Section 4.2).
- 단일 점 비교는 false positives ≈ 10%, false negatives ≈ 75%를 보인다 (출처: Section 4.2, Figure 6).
- 평균 비교는 false positives < 5%, false negatives ≈ 90%를 보인다 (출처: Section 4.2, Figure 6).
- `P(A > B)` 기반 비교는 false positives ≈ 5%, false negatives ≈ 30%를 보인다 (출처: Section 4.2, Figure 6).
- `γ = 0.75`, `β = 0.05`일 때 최소 sample size는 29 trainings이다. `P(A > B) < 0.6`을 신뢰적으로 탐지하는 것은 비현실적이며, 0.55 미만에서는 700 trainings 초과가 필요하다고 보고된다 (출처: Appendix C, Figure C.1).
- CIFAR10-VGG11 hyperparameters: learning rate default 0.03, space `log(0.001, 0.3)`; weight decay default 0.002, space `log(1e-6, 1e-2)`; momentum default 0.9, space `lin(0.5, 0.99)`; lr schedule γ default 0.97, space `lin(0.96, 0.999)`; batch size 128 (출처: Table 2).
- BERT SST-2/RTE hyperparameters: learning rate default 2e-5, space `log(1e-5, 1e-4)`; weight decay default 0.0, space `log(1e-4, 2e-3)`; weights init std default 0.2, space `log(0.01, 0.5)`; dropout 0.1; batch size 32 (출처: Table 3).
- PascalVOC hyperparameters: learning rate default 0.002, space `log(1e-5, 1e-2)`; momentum default 0.9, space `lin(0.50, 0.99)`; weight decay default 1e-6, space `log(1e-8, 1e-1)`; batch size 16 (출처: Table 5).
- MLP-MHC hyperparameters: hidden layer size space `lin(20, 400)`, L2-weight decay space `log(0, 1)`, default hidden layer size 150, default L2 0.001 (출처: Table 6, Table 7).
- MHC 성능: NetMHCpan4 HPV AUC 0.53, PCC 0.39; MHCflurry HPV AUC 0.58, PCC 0.41; MLP-MHC HPV AUC 0.63, PCC 0.31 (출처: Table 8).
- MHC cross-validation split 성능: NetMHCpan4 AUC 0.854, PCC 0.620; MHCflurry AUC 0.964*, PCC 0.671*; MLP-MHC AUC 0.861, PCC 0.660 (출처: Table 8).
- HPO 최적화 곡선에서 standard deviation은 대부분 50 iterations 이전에 안정화된다고 보고된다 (출처: Figure F.2).

## 5. 우리 프로젝트와의 관련 (가설)
- 옮겨올 수 있는 것, 가설:
  - fold context 기반 in-context 분류기와 학습 기반 MIL 기준선을 비교할 때, 단일 fold 또는 단일 run 평균 대신 paired multiple folds에서 `P(A > B)`와 confidence interval을 계산하면 승격 판단의 변동 통제를 강화할 수 있을 가설이 있다. 근거는 이 논문이 pairing과 `P(A > B)`가 오류율을 개선한다고 주장하기 때문이며, 우리 프로젝트에 그대로 성립한다고 단정하지 않는다 (출처: Section 4, Appendix C).
  - branch stochastic fitting, fold sampling, data order, augmentation, HOpt seed 등을 무작위화하면 run 간 상관관계를 낮추어 추정 분산을 줄이는 방향일 수 있다. 이 논문은 더 많은 변동 원인을 무작위화할수록 편향 추정자의 MSE가 개선된다고 보고하지만, 우리 fold context/브랜치 적합 구조에서도 동일할지는 확인되지 않았다 (출처: Section 3.3, Figure H.5).
  - 고정 fold/test 대신 out-of-bootstrap 또는 여러 random splits를 사용하면 작은 개선 탐지력을 높이는 데 도움이 될 수 있다. 이는 WSI의 slide/patch 구조가 i.i.d. 가정을 만족하지 않을 수 있으므로 이치적으로만 적용 가능한 가설이다 (출처: Section 3.1, Section 5, Appendix B, Section 2.1).
  - `γ = 0.75`, `β = 0.05`에서 최소 29 trainings라는 수치는 우리 프로젝트의 fold 수 또는 run 수 설계에 참고점으로 사용할 수 있을 가설이다. 다만 이 수치는 본문의 5개 사례 연구에서 유도된 것이므로 WSI 벤치마크에 그대로 이식된다고 단정하지 않는다 (출처: Appendix C, Figure C.1).
- 옮길 수 없는 것, 가설/이유:
  - 이 논문은 pathology WSI, MIL, patch spatial correlation, slide-level grouping을 다루지 않으므로, WSI 특유의 의존 구조에 대한 분산 모델은 직접 옮기기 어렵다 (출처: 본문에서 확인 못함).
  - 브랜치 유효 랭크 중복, branch selection, closed-form ridge fitting variance와 같은 우리 프로젝트의 구체적 문제는 본문에서 다루지 않는다 (출처: 본문에서 확인 못함).
  - 본문의 계산 비용 수치, 예를 들어 1,070 hours, 21 hours, 6.4 GPU years는 CIFAR10, PascalVOC, GLUE, MHC 사례 연구 기반이므로 WSI 학습 비용에 직접 적용할 수 있을지는 알 수 없다 (출처: Section 3.3).
  - 본문의 i.i.d. 데이터 가정과 classification/regression 지표 중심 분석은 WSI의 grouped, spatially correlated, possibly non-normal 지표와 다를 수 있다 (출처: Section 2.1, Section 6).

## 6. 이 논문이 답하지 않는 것
- pathology WSI 벤치마크에서 학습 기반 MIL 기준선, 특히 ABMIL을 넘는지에 대한 성능 수치: 본문에서 확인 못함.
- ABMIL 또는 WSI MIL 파이프라인의 변동 원인 분해: 본문에서 확인 못함.
- 우리 프로젝트의 브랜치 유효 랭크 중복이 벤치마크 분산에 미치는 영향: 본문에서 확인 못함.
- 확률적 ridge/in-context fitting으로 인한 적합 분산이 승격 기준에 미치는 영향: 본문에서 확인 못함.
- WSI slide/patch 구조에 맞는 non-i.i.d. split, paired test, sample size 설계: 본문에서 확인 못함.
- closed-form ridge in-context classifier가 학습 기반 MIL 기준선보다 안정적으로 좋은지: 본문에서 확인 못함.
- WSI 벤치마크에 필요한 최소 fold 수, run 수, HPO 예산: 본문에서 확인 못함.
