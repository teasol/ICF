# 문헌 브리프 — icmil (로컬 모델 요약)

```
url: https://arxiv.org/abs/2606.06458
fetched_from: https://arxiv.org/html/2606.06458
fetched_at: 2026-09-18 10:28:41 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8001` · 2026-09-18 10:34 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 기존 MIL 알고리즘은 low-label regime에서 취약하며, 유연한 모델은 overfit하고 경직된 모델은 task에 적응하지 못한다고 주장한다. (출처: Abstract, §1)
- PFN 패러다임을 bag-structured data에 적용하면, synthetic data pretraining만으로 소수의 labeled bags에서 새 MIL task를 풀 수 있다고 주장한다. (출처: Abstract, §1, §2.2)
- 추론 시 gradient update나 task-specific training 없이 single forward pass로 bag label을 분류한다고 주장한다. (출처: Abstract, §1, §4.4)
- Perceiver-style architecture가 bag-structured in-context learning의 scalability, task-dependent compression, permutation invariance 문제를 해결한다고 주장한다. (출처: §1, §3.1, Fig. 2)
- factorized prior와 joint prior는 서로 다른 inductive bias를 가지며, mixture가 per-task 강점을 상속한다고 주장한다. (출처: §1, §3.2, §4.2, §4.3)
- mixture prior로 학습한 ICMIL이 twelve MIL benchmarks에서 low-label regime의 평균 AUROC와 rank를 개선하고, task-specific training이 필요한 supervised baselines를 능가한다고 주장한다. (출처: Abstract, §1, §4.4, Table 1)
- ICMIL은 labeled context와 일관된 가설에 대한 Bayesian model averaging을 근사하여, scarce labels에서 단일 aggregator fitting의 variance를 줄인다고 주장한다. (출처: §1, Fig. 1)
- scaling, 즉 학습 시간과 용량 증가가 일부 벤치마크에서 추가 성능을 낸다고 주장한다. (출처: §4.4, Fig. 6)
- PCA 25차원 평가, finetuning, high-dimensional foundation-model embeddings는 한계 또는 미래 작업이라고 주장한다. (출처: §5)

## 2. 방법의 핵심
- 구조: 입력 instance features를 feature group size `s`로 묶어 `G = ceil(F/s)`개의 feature-group embedding을 만들고, 각 bag에 learnable bag token `Q_b ∈ R^{G×E}`를 둔다. bag label도 embedding하며, test bag은 training label embedding의 mean을 받는다. (출처: §3.1)
- 구조: `T`회 반복하여, 먼저 bag token이 같은 bag의 instance token에 cross-attend하는 instance aggregation을 수행하고, 이후 label embedding을 추가한 뒤 모든 bag token과 label token 사이에서 column-row self-attention을 수행한다. test bag은 training context bag에만 attend하고 다른 test bag에는 attend하지 않는다. (출처: §3.1, Fig. 2)
- 구조: `T`회 반복 후 query bag의 label token을 decoder로 class distribution으로 변환한다. (출처: §3.1)
- 학습 방식: synthetic MIL dataset prior `p(D)`에서 context bags와 query bag을 주어 query bag label의 expected negative log-likelihood를 최소화하는 PFN 방식으로 학습한다. (출처: §3, Eq. 1)
- 학습 방식: synthetic prior는 factorized prior와 joint prior로 나뉜다. factorized prior는 instance-level transform `f`, aggregation `φ`, bag transform `g`로 `y = g(φ({f(x_i)}))`를 생성하며, `φ`는 discrete histogram/presence 또는 continuous mean/ABMIL pooling, `g`는 MLP/tree/lookup을 사용한다. joint prior는 flattened bag `[x_1;...;x_I]`에 대한 단일 함수로 bag label을 생성한다. (출처: §3.2, Fig. 3)
- 학습 방식: 최종 mixture prior는 Joint 0.70, Factorized(cont, MLP) 0.15, Factorized(disc, lookup) 0.15를 사용한다. (출처: §4.3)
- 재현 설정: reduced setup은 `T=6`, attention heads 4, embedding dimension `E=128`, MLP hidden size 512, feature group size 1, 25 features, 20,000 steps, batch size 128, dataset당 bag count uniform `[70,125]`, schedule-free AdamW, learning rate `1e-3`, no warmup, single NVIDIA A100, 약 2.56M synthetic datasets를 사용한다. (출처: §4.1, Appendix B)
- 재현 설정: scaled ICMIL은 40,000 steps, `E=256`, MLP hidden size 1054, learning rate `5e-4`, 2,500 warmup steps, 약 5.12M synthetic datasets를 사용하며, iteration 수와 attention heads는 reduced setup과 동일하다. (출처: §4.4, Appendix B)
- 재현 설정: curriculum은 0–500 steps에서 bag size `[2,8]` 및 최대 6 instance classes, 500–7,500 steps에서 `[4,15]` 및 최대 12 instance classes, 7,500–20,000 steps에서 `[6,20]` 및 최대 20 instance classes를 사용한다. (출처: Appendix B)
- 추론 방식: 실제 MIL dataset을 context로 넣고 test bags를 single forward pass로 분류하며, gradient update, hyperparameter tuning, task-specific training을 사용하지 않는다. (출처: Abstract, §1, §4.4)

## 3. 평가 설정
- 데이터셋: twelve MIL benchmarks를 사용한다. witness label rule 그룹은 SMIL, Musk1, Musk2, Letters, HEPMASS, RSNA-ICH이고, interaction 그룹은 Elephant, Fox, Tiger, TCGA, Adjacent Pairs, Pos/Neg이다. (출처: §4.1)
- 데이터셋: 큰 벤치마크는 약 100 bags로 subsample하며, feature dimension이 25를 초과하면 PCA로 25차원으로 줄인다. (출처: §4.1)
- 데이터셋: Appendix E에 따르면 Musk1은 92 bags, Musk2는 102 bags, Elephant/Fox/Tiger는 각 200 bags, SMIL/Pos/Neg/Adjacent Pairs/RSNA-ICH/TCGA는 100 bags로 subsample된다. (출처: Appendix E)
- 데이터셋: TCGA는 TCGA whole-slide embeddings, 즉 UNI2 patch features 1,536차원을 PCA 25차원으로 줄인 LUAD vs. LUSC binary task이다. (출처: Appendix E)
- 비교 기준선: MeanLogReg, SVM-Summ, ABMIL, DSMIL, ACMIL, TabPFN-Concat, TabPFN-Subsample, TabPFN-Cluster를 비교한다. ABMIL이 기준선에 포함된다. (출처: §4.1)
- ABMIL 기준선 설정: embedding dimension 500, attention dimension 128, learning rate `{1e-2, 5e-3, 1e-3, 5e-4, 1e-4}`, weight decay `{0, 1e-4, 5e-4}`, single stratified 10% held-out validation split, 최대 200 epochs full-batch Adam, early stopping patience 20, lowest validation cross-entropy 설정을 사용한다. (출처: §4.1)
- 지표: per-benchmark AUROC, average AUROC, average rank를 사용한다. (출처: Table 1)
- 지표: Fig. 1에서는 MNIST-Pos/Neg에서 20회 independently resampled training sets에 대한 AUROC 분포를 boxplot으로 보고한다. (출처: Fig. 1)
- 지표: Fig. 5에서는 Pos/Neg benchmark의 total wall-clock time을 보고한다. (출처: Fig. 5)

## 4. 정량 결과
- Table 1: ICMIL scaled setup의 average AUROC는 84.17, average rank는 4.67이다. (출처: Table 1)
- Table 1: baseline average AUROC/rank는 MeanLogReg 82.37/7.46, SVM-Summ 79.38/7.08, ABMIL 79.97/9.17, DSMIL 80.33/8.58, ACMIL 83.81/5.88, TabPFN-Concat 74.16/11.50, TabPFN-Subsample 74.82/10.67, TabPFN-Cluster 81.21/6.67이다. (출처: Table 1)
- Table 1: ICMIL per-benchmark AUROC는 Musk1 93.3±0.6, Musk2 90.8±1.1, HEPMASS 84.5±0.7, Letters 95.7±0.7, SMIL 74.4±0.3, Adj. Pairs 77.6±0.5, Pos/Neg 87.0±0.2, RSNA ICH 75.0±0.1, Elephant 94.2±0.1, Fox 60.7±0.8, Tiger 88.9±0.6, TCGA 87.9±0.1이다. (출처: Table 1)
- Table 1: ABMIL per-benchmark AUROC는 Musk1 85.7±3.5, Musk2 75.4±2.8, HEPMASS 75.7±1.7, Letters 97.6±0.8, SMIL 85.4±0.5, Adj. Pairs 65.5±1.1, Pos/Neg 74.6±1.6, RSNA ICH 71.2±0.3, Elephant 94.1±0.2, Fox 59.8±1.6, Tiger 83.9±1.5, TCGA 90.7±1.6이다. (출처: Table 1)
- Table 1: prior ablation average AUROC/rank는 Joint 82.56/5.79, Fact.(cont, MLP) 82.76/6.50, Fact.(cont, tree) 81.55/8.71, Fact.(disc, lookup) 80.98/9.17, Fact.(disc, MLP) 75.12/13.04, Fact.(disc, tree) 51.27/15.92, Mixed 83.38/5.21이다. (출처: Table 1)
- Fig. 4: prior별 mean rank는 Joint 1.92, Fact.(cont, MLP) 2.00, Fact.(cont, tree) 3.04, Fact.(disc, lookup) 3.12, Fact.(disc, MLP) 4.92, Fact.(disc, tree) 6.00이다. (출처: Fig. 4)
- §4.4: ICMIL은 가장 강한 supervised baseline인 ACMIL보다 average AUROC에서 0.36 높다. (출처: §4.4)
- §4.4: scaling으로 Mixed 대비 Fox +6.8, Musk2 +3.2, Letters +1.3, Musk1 +0.8이 개선되었고, HEPMASS는 2.7-point regression이 관찰되었다. (출처: §4.4, Fig. 6)
- §4.4: baseline 우위 예시로 ACMIL이 SMIL에서 +13.3, TabPFN-Cluster가 Letters에서 +3.5 높다. (출처: §4.4)
- §4.2: Factorized(disc, lookup) prior는 Joint prior 대비 Letters, SMIL, HEPMASS에서 최대 4 percentage points까지 개선된다. (출처: §4.2)
- Table 2: architectural ablation mean AUROC는 T=1 65.5, T=2 79.0, T=4 83.3, T=6 83.3이다. (출처: Table 2)
- Table 2: inter-bag attention disabled는 45.8, enabled는 83.3이다. (출처: Table 2)
- Table 2: embedding dimension E=48 83.1, E=64 82.4, E=96 83.3, E=128 83.3이다. (출처: Table 2)
- Table 3: prior mixture weights별 mean AUROC는 0.60/0.20/0.20 83.5, 0.70/0.15/0.15 83.3, 0.80/0.10/0.10 82.0이다. (출처: Table 3)
- Table 4: synthetic two-witness task에서 TabPFN-Cluster AUROC는 54.0, ICMIL AUROC는 88.0이다. (출처: Table 4)
- Fig. 1: MNIST-Pos/Neg 20회 resampling AUROC median/variance의 정확한 수치: 본문에서 확인 못함. (출처: Fig. 1)
- Fig. 5: Pos/Neg wall-clock time의 정확한 수치: 본문에서 확인 못함. (출처: Fig. 5)
- Fig. 6: Fox/Musk2 learning curve의 정확한 수치: 본문에서 확인 못함. (출처: Fig. 6)

## 5. 우리 프로젝트와의 관련 (가설)
- 가설: ICMIL의 fold context를 query bag label prediction에 쓰는 in-context formulation은 우리의 closed-form ridge in-context 분류기와 같은 문제 구조이므로, 병리 WSI low-label fold에서도 learned in-context model이 ABMIL 대비 variance를 줄일 수 있을 가능성이 있다. (출처: Abstract, §1, Table 1)
- 가설: ICMIL은 bag labels을 inter-bag attention에 넣어 instance compression을 task-aware하게 하므로, 우리의 ridge가 fold context labels을 feature compression에 직접 활용하지 못한다면 label-aware context mixing이 branch 중복을 줄이는 방향일 수 있다. 단, ICMIL이 branch rank 중복을 직접 해결한다는 보고는 본문에서 확인 못함. (출처: §3.1, Fig. 2)
- 가설: ICMIL은 joint/factorized prior mixture로 robust generalist를 만들므로, 병리 WSI의 spatial correlation, witness 구조, interaction 구조에 맞는 synthetic prior를 섞으면 ABMIL 기준선을 넘는 데 도움이 될 수 있다. 단, 논문은 pathology-specific spatial prior를 future work로만 언급한다. (출처: §3.2, §4.3, §6)
- 가설: ICMIL이 PFN을 통해 Bayesian model averaging을 근사하고 resampling variance를 줄인다고 주장하므로, 확률적 ridge의 적합 분산 문제가 amortized in-context prediction으로 완화될 수 있다. 단, ICMIL의 variance 결과는 MNIST-Pos/Neg와 twelve MIL benchmarks에서 보고되었고, probabilistic ridge나 branch rank와 직접 비교하지 않았다. (출처: §1, Fig. 1)
- 가설: ICMIL의 TCGA LUAD-LUSC 결과는 WSI patch embedding을 사용하는 병리 task에서 in-context MIL이 작동할 수 있음을 시사하지만, 우리의 특정 병리 WSI 벤치마크와 fold protocol에서는 전이가 보장되지 않는다. (출처: Appendix E, Table 1)
- 옮기기 어려운 점: ICMIL은 learned parameters와 대규모 synthetic pretraining을 요구하므로, 학습 파라미터 0개 또는 closed-form ridge를 유지하는 접근으로는 직접 재현이 어렵다. (출처: §3.1, Appendix B)
- 옮기기 어려운 점: ICMIL은 PCA 25차원으로 평가하고, high-dimensional foundation-model embeddings는 future work로 명시하므로, 우리의 고차원 patch embedding 환경에서는 성능 전이가 불확실하다. (출처: §4.1, §5, Appendix E)
- 옮기기 어려운 점: ICMIL은 7개 branch의 유효 랭크 3.52/7 같은 rank redundancy 문제를 다루지 않으므로, branch 중복 감소에 대한 직접적 방법은 본문에서 확인 못함. (출처: 본문 전체)
- 옮기기 어려운 점: ICMIL은 확률적 ridge 적합 분산이 승격 기준 대비 최대 2.8배인 문제를 재현하거나 비교하지 않으므로, 해당 분산 감소에 대한 직접 증거는 본문에서 확인 못함. (출처: 본문 전체)

## 6. 이 논문이 답하지 않는 것
- 우리의 특정 병리 WSI 벤치마크에서 ICMIL이 ABMIL을 넘는지에 대한 직접 결과: 본문에서 확인 못함. TCGA LUAD-LUSC와 RSNA-ICH는 포함되지만, 우리의 벤치마크와 fold protocol은 본문에서 확인 못함. (출처: §4.1, Appendix E)
- closed-form ridge in-context 분류기와 ICMIL의 head-to-head 비교: 본문에서 확인 못함.
- 7개 branch의 유효 랭크 3.52/7 중복을 줄이는 방법: 본문에서 확인 못함.
- 확률적 ridge 적합 분산이 승격 기준 대비 최대 2.8배인 문제를 ICMIL이 어떻게 줄이는지: 본문에서 확인 못함.
- PCA 25차원 없이 고차원 foundation-model embeddings, 예로 UNI2 1,536차원 또는 ResNet-50 2,048차원을 그대로 사용하는 성능: 본문에서 확인 못함. (출처: §4.1, §5, Appendix E)
- pathology spatial prior의 구체적 구현과 성능: 본문에서 확인 못함. future work로만 언급된다. (출처: §6)
- WSI-scale patch 수, slide 수, 런타임, 메모리 사용량: 본문에서 확인 못함. 복잡도 논의만 있고 WSI-scale 실행 수치는 본문에서 확인 못함. (출처: §3.1)
- calibration, uncertainty, branch rank metric, effective rank metric: 본문에서 확인 못함.
- real pathology corpora에서 finetuning한 성능: 본문에서 확인 못함. future work로만 언급된다. (출처: §5, §6)
