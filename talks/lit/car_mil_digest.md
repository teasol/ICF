# 문헌 브리프 — car_mil (로컬 모델 요약)

```
url: https://arxiv.org/abs/2609.08419
fetched_from: https://arxiv.org/html/2609.08419
fetched_at: 2026-09-18 12:45:05 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8000` · 2026-09-18 12:50 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- MIL attention은 instance 중요도를 충실히 반영하지 못하고 spuriously correlated region에 집중할 수 있으며, training instability로 attention collapse가 발생할 수 있다 (출처: Sec. 1).
- 기존 MIL 방법은 예측을 지지하는 evidence만 강조하고, 예측을 반박하는 evidence에 대해서는 거의 정보를 제공하지 않는다고 주장한다 (출처: Sec. 1).
- attention은 최적화의 부산물이 아니라, supporting/contradicting evidence가 instance에 어떻게 배분되는지를 반영하도록 training 중 guided 되어야 한다고 주장한다 (출처: Sec. 1).
- CAR-MIL은 counterfactual explanation에서 영감을 받은 counterfactual attention regularization을 MIL training에 도입하는 프레임워크라고 주장한다 (출처: Abstract, Sec. 1, Sec. 3.2).
- 기존 counterfactual explanation은 post-hoc 해석에 쓰이지만, CAR-MIL은 training time에 attention learning을 guide하도록 적응시켰다고 주장한다 (출처: Sec. 1, Sec. 2).
- CAR-MIL은 input space가 아니라 attention space에서 perturbation을 수행한다고 주장한다 (출처: Sec. 1, Sec. 3.2).
- lightweight counterfactual attention branch는 factual attention distribution에 가깝게 유지되면서 다른 예측을 생성하도록 학습된다고 주장한다 (출처: Abstract, Sec. 1, Sec. 3.2).
- factual attention은 예측을 지지하는 region을, counterfactual attention은 예측을 challenge하는 region을 드러내는 complementary evidence map이라고 주장한다 (출처: Abstract, Sec. 1, Fig. 1, Fig. 5).
- counterfactual branch는 encoder와 classifier를 공유하므로 additional supervision 없이 minimal computational overhead만 추가된다고 주장한다 (출처: Sec. 1, Sec. 3.2, Appendix A.7).
- MIL에 counterfactual attention regularization을 처음 도입했다고 주장한다 (출처: Sec. 1).
- 기존 counterfactual attention learning 방식은 random attention perturbations에 의존하지만, CAR-MIL은 prediction에 영향을 주는 perturbations을 학습한다고 주장한다 (출처: Sec. 1, Sec. 2).
- synthetic MIL benchmarks와 five digital pathology datasets across four tasks에서 평가했다고 주장한다 (출처: Abstract, Sec. 4).
- CAR-MIL은 competitive classification performance를 유지하면서, 더 challenging한 task에서 가장 큰 성능 향상이 관찰된다고 주장한다 (출처: Abstract, Sec. 4.2).
- CAR-MIL은 attention reliability를 정량 지표와 qualitative analysis 모두에서 개선한다고 주장한다 (출처: Abstract, Sec. 6).
- CAR-MIL은 DSMIL, TransMIL 등 다른 attention-based MIL architecture에도 통합 가능하다고 주장한다 (출처: Sec. 5, Table 3).
- CAR-MIL은 pathology-specific UNI features뿐 아니라 standard convolutional ResNet50 features에서도 ABMIL 대비 성능 향상을 보이며, out-of-domain features에서 효과가 더 뚜렷하다고 주장한다 (출처: Sec. 5, Table 4).
- paired statistical t-tests 결과, 어떤 dataset에서도 attention-based MIL method가 CAR-MIL보다 유의하게 우수하지 않다고 주장한다 (출처: Sec. 4.2).
- MORF perturbation test에서 CAR-MIL은 smooth and monotonic confidence drop을 보이고, ABMIL은 non-monotonic behavior을 보인다고 주장한다 (출처: Sec. 6, Fig. 4).
- factual attention은 ABMIL localization을 단순히 재현하지 않으며, factual/counterfactual branch는 서로 다른 region에 attention을 둔다고 주장한다 (출처: Sec. 6).
- multi-class setting에서 counterfactual logit shift는 dataset evidence structure에 따라 달라진다고 주장한다 (출처: Sec. 5).
- L1 proximity constraint 하에서 optimal counterfactual perturbation은 margin sensitivity가 가장 큰 instance들에 집중된다고 이론적으로 주장한다 (출처: Appendix A.4).

## 2. 방법의 핵심
- 구조: standard ABMIL의 encoder, attention module, bag classifier를 기본으로 하며, factual branch와 counterfactual branch를 가진 two-branch attention MIL이다 (출처: Sec. 3.1, Sec. 3.2, Fig. 2).
- factual branch: instance feature \(z_j\)에서 attention logit \(u_j=\psi(z_j)\), softmax attention \(a\), bag representation \(\hat Z=\sum_j a_j z_j\), class logits \(F(u)=\phi(\hat Z)\), bag prediction \(\hat y(u)=\text{softmax}(F(u))\)를 계산한다 (출처: Sec. 3.1, Eq. 1-2).
- counterfactual branch: 동일한 instance embeddings과 shared classifier를 사용하지만, 별도 attention module \(\psi_{cf}\)로 \(u^{cf}_j=\psi_{cf}(z_j)\), \(a^{cf}=\text{softmax}(u^{cf})\), \(\hat Z^{cf}=\sum_j a^{cf}_j z_j\), \(F(u^{cf})=\phi(\hat Z^{cf})\)를 계산한다 (출처: Sec. 3.2, Eq. 3).
- encoder \(\mathcal E\)와 downstream classifier \(\phi\)는 branch가 아니라 shared component이며, prediction difference는 attention weighting difference에서만 발생하도록 설계된다 (출처: Sec. 3.2).
- evidence differential은 logit space에서 \(\Delta F(u,u^{cf})=F(u)-F(u^{cf})\)로 정의된다 (출처: Sec. 3.2, Eq. 4).
- 학습 방식: total loss는 \(\mathcal L=\mathcal L_{cls}+\alpha \mathcal L_{diff}+\lambda \mathcal L_{div}\)이다 (출처: Sec. 3.3, Eq. 11).
- \(\mathcal L_{cls}\)는 factual branch prediction에 대한 standard cross-entropy loss이다 (출처: Sec. 3.3, Eq. 5).
- \(\mathcal L_{diff}\)는 \(\text{CE}(\text{softmax}(\Delta F),y)\)로, ground-truth class에서 factual/counterfactual prediction difference가 label-consistent하도록 만든다 (출처: Sec. 3.3, Eq. 6).
- \(\mathcal L_{div}\)은 factual/counterfactual attention logits의 proximity regularization이며, normalized L1 distance 또는 cosine-based dissimilarity를 사용할 수 있다 (출처: Sec. 3.3, Eq. 8-10).
- L1 variant는 \(\frac{1}{N}\sum_j |u_j-u^{cf}_j|\)로, 소수의 influential instance attention만 수정하도록 장려한다고 기술된다 (출처: Sec. 3.3, Eq. 9, Appendix A.4).
- cosine variant는 attention pattern의 directional disagreement를 scale-invariant하게 측정한다 (출처: Sec. 3.3, Eq. 10).
- 추론 방식: 본문에서 별도 inference procedure를 명시적으로 기술하지는 않았으며, bag-level prediction은 main/factual branch prediction으로 기술된다 (출처: Sec. 3.3, Fig. 2).
- 재현 설정, synthetic: MNIST images에 ImageNet pre-trained ResNet18 feature extractor를 사용하고, 10 repeats, learning rate 0.0002, SGD optimizer, L1 distance를 사용했다 (출처: Sec. 4.1).
- 재현 설정, pathology: patches는 256x256, 20X magnification으로 추출했으며, pre-extracted UNI-V1 foundation model features를 사용했다 (출처: Sec. 4.2).
- 재현 설정, pathology: TCGA-NSCLC, TCGA-BRCA, TCGA-LUAD는 5-fold cross-validation, learning rate 0.0002, Adam optimizer; BRACS는 original train-val-test split, 3 runs, learning rate 0.0001; CAMELYON16은 train-test split, 3 runs, learning rate 0.0002 (출처: Sec. 4.2).
- hyperparameter: \(\alpha\)는 evidence differential loss, \(\lambda\)는 attention-logit proximity loss를 제어하며, BRCA/LUAD single run에서 \(\{0.2,0.8,1.0\}\)과 ABMIL baseline \(\alpha=\lambda=0\)을 탐색했다 (출처: Sec. 5, Fig. 3).
- 다른 MIL architecture 통합: DSMIL은 query projection만 복제하여 counterfactual scores를 만들고, TransMIL은 Nyström attention 때문에 final transformer block을 parallel로 추가하여 class-token attention을 surrogate attention signal로 사용한다 (출처: Appendix A.7).
- computational overhead: ABMIL 대비 CAR-ABMIL의 추가 비용은 매우 작다고 보고되며, dummy input shape은 \(1\times10000\times1024\)로 측정했다 (출처: Appendix A.7, Table A.3).

## 3. 평가 설정
- synthetic dataset: MNIST-bags synthetic MIL datasets를 사용했으며, 두 variant는 4-bags/Four Bags와 Adjacent Pairs이다 (출처: Sec. 4.1, Appendix A.5.2).
- synthetic dataset 특징: instance-level ground truth evidence labels이 있어 attention behavior를 controlled analysis할 수 있다 (출처: Sec. 4.1).
- pathology dataset: TCGA-NSCLC와 TCGA-BRCA는 binary cancer subtyping, TCGA-LUAD는 TP53 mutation prediction, CAMELYON16은 binary metastasis detection, BRACS는 multiclass tissue classification task에 사용된다 (출처: Sec. 4.2).
- BRACS labeling: Sec. 4.2에서는 normal, benign, malignant categories로 기술되고, Table A.2에서는 7 diagnostic classes로 기술된다 (출처: Sec. 4.2, Table A.2).
- dataset 규모: TCGA-BRCA 977 slides, TCGA-NSCLC 956 slides, TCGA-LUAD TP53 427 slides, BRACS 546 slides, CAMELYON16 train 270 slides, CAMELYON16 test 129 slides (출처: Table A.2).
- instance-level annotation: CAMELYON16과 BRACS는 instance-level annotations을 포함하며, BRACS는 sparse labels이다 (출처: Sec. 4.2).
- synthetic baselines: ABMIL, AddMIL, TransMIL을 비교했으며, ABMIL은 기준선으로 포함된다 (출처: Sec. 4.1, Table 1).
- pathology baselines: Mean-Max MIL, ABMIL, CLAM, ACMIL, DSMIL, TransMIL, AddMIL, CIA-MIL을 비교했으며, Table 2에는 RRT-MIL도 포함된다 (출처: Sec. 4.2, Table 2).
- ABMIL 기준선 존재 여부: ABMIL은 synthetic와 pathology 평가 모두에서 기준선으로 존재한다 (출처: Sec. 4.1, Table 1, Sec. 4.2, Table 2).
- synthetic metrics: bag-level AUC, instance-level positive evidence AUPRC+, positive/negative evidence averaged AUPRC± (출처: Sec. 4.1, Table 1).
- pathology metrics: AUC, F1 score, balanced accuracy를 보고하며, CAMELYON16에서는 AUPRC+를 추가로 보고한다 (출처: Sec. 4.2, Table 2).
- evaluation protocol: synthetic는 10 repeats; TCGA-NSCLC/BRCA/LUAD는 5-fold cross-validation; BRACS와 CAMELYON16은 3 runs (출처: Sec. 4.1, Sec. 4.2).
- ablation/analysis: CAR-MIL을 DSMIL/TransMIL에 통합한 Table 3, ResNet50 features를 사용한 Table 4, attention interpretability proxy를 평가한 Table 5, WENO 비교 Table A.1, model complexity Table A.3을 포함한다 (출처: Table 3, Table 4, Table 5, Table A.1, Table A.3).

## 4. 정량 결과
- Table 1, Adjacent Pairs: ABMIL AUC 85.7±5.9, AUPRC+ 76.9±6.1, AUPRC± 61.5±0.9; TransMIL AUC 94.6±4.4, AUPRC+ 81.5±6.9, AUPRC± 61.7±2.1; CAR-MIL AUC 94.1±5.9, AUPRC+ 85.7±6.5, AUPRC± 84.6±5.2 (출처: Table 1).
- Table 1, Four Bags: ABMIL AUC 99.1±0.1, AUPRC+ 86.8±0.2, AUPRC± 53.1±0.1; TransMIL AUC 99.5±0.1, AUPRC+ 81.6±6.3, AUPRC± 51.9±0.8; CAR-MIL AUC 99.6±0.1, AUPRC+ 86.8±0.6, AUPRC± 89.7±0.9 (출처: Table 1).
- Table 2, BRCA: ABMIL AUC 95.3±1.7, F1 88.9±1.3; CAR-MIL L1 AUC 95.5±2.2, F1 87.9±1.5; CAR-MIL COS AUC 95.0±1.7, F1 89.6±1.4 (출처: Table 2).
- Table 2, NSCLC: ABMIL AUC 97.6±1.0, F1 94.2±1.2; CAR-MIL L1 AUC 98.0±0.5, F1 94.4±1.7; CAR-MIL COS AUC 97.5±0.9, F1 95.0±1.8 (출처: Table 2).
- Table 2, LUAD: ABMIL AUC 74.0±4.3, F1 72.1±5.4; CAR-MIL L1 AUC 76.4±4.9, F1 72.4±3.8; CAR-MIL COS AUC 78.1±4.1, F1 72.6±4.5 (출처: Table 2).
- Table 2, BRACS: ABMIL BACC 38.2±4.1, F1 36.0±4.5; CAR-MIL L1 BACC 40.8±2.5, F1 38.1±2.5; CAR-MIL COS BACC 43.1±2.0, F1 40.4±2.4 (출처: Table 2).
- Table 2, CAMELYON16: ABMIL AUC 98.7±0.3, AUPRC+ 93.2±1.2; CAR-MIL L1 AUC 99.9±0.1, AUPRC+ 94.8±1.0; CAR-MIL COS AUC 99.3±0.7, AUPRC+ 91.7±0.6 (출처: Table 2).
- Table 3, ABMIL vs CAR-MIL: BRCA 95.3±1.7/88.9±1.3 vs 95.0±1.7/89.6±1.4; NSCLC 97.6±1.0/94.2±1.2 vs 98.0±0.5/94.4±1.7; LUAD 74.0±4.3/72.1±5.4 vs 78.1±4.1/72.6±4.5; BRACS 38.2±4.1/36.0±4.5 vs 43.1±2.0/40.4±2.4; CAMELYON16 98.7±0.3/93.2±1.2 vs 99.9±0.1/94.8±1.0 (출처: Table 3).
- Table 3, DSMIL vs CAR-DSMIL: LUAD AUC 66.9±4.7 vs 71.2±4.1, F1 64.8±5.1 vs 67.8±3.4; BRACS BACC 42.3±2.0 vs 42.5±5.1, F1 40.0±2.0 vs 39.8±2.0; CAMELYON16 AUPRC+ 80.6±11.2 vs 87.8±2.9 (출처: Table 3).
- Table 3, TransMIL vs CAR-TransMIL: LUAD AUC 71.7±5.4 vs 73.6±4.4, F1 68.8±4.4 vs 69.8±4.0; BRACS BACC 40.0±3.6 vs 42.6±6.1, F1 37.6±3.8 vs 39.4±6.2; CAMELYON16 AUPRC+ 28.2±3.6 vs 32.6±0.3 (출처: Table 3).
- Table 4, ResNet50 features: BRCA ABMIL AUC 89.2±2.6, F1 79.7±3.9 vs CAR-MIL AUC 90.3±2.1, F1 81.9±5.0; NSCLC ABMIL AUC 93.4±1.7, F1 88.2±1.6 vs CAR-MIL AUC 94.3±1.2, F1 88.9±1.3; LUAD ABMIL AUC 68.2±5.6, F1 66.3±4.8 vs CAR-MIL AUC 70.4±6.1, F1 68.4±5.1 (출처: Table 4).
- Fig. 3, hyperparameter sensitivity: BRCA baseline AUC 94.5가 configuration에 따라 95.3~96.0 범위로 보고되며, LUAD baseline AUC 81.3가 최대 84.2~84.4까지 보고된다 (출처: Sec. 5, Fig. 3).
- Table 5, Adjacent Pairs attention proxy: Random AUPRC+ 54.2±0.9, AUPRC± 54.2±0.9; ABMIL attention AUPRC+ 76.9±6.1, AUPRC± 61.5±0.9; Perturbation AUPRC+ 78.5±1.0, AUPRC± 78.5±1.0; CAR-MIL attention AUPRC+ 85.7±6.5, AUPRC± 84.6±5.2 (출처: Table 5).
- Table 5, Four Bags attention proxy: Random AUPRC+ 30.6±0.2, AUPRC± 31.2±0.2; ABMIL attention AUPRC+ 86.8±0.2, AUPRC± 53.1±0.1; Perturbation AUPRC+ 87.9±1.6, AUPRC± 87.7±2.1; CAR-MIL attention AUPRC+ 86.8±0.6, AUPRC± 89.7±0.9 (출처: Table 5).
- Table A.1, WENO comparison: LUAD ABMIL AUC 74.0±4.3, F1 72.1±5.4; WENO AUC 55.5±10.7, F1 58.6±8.6; CAR-MIL L1 AUC 76.4±4.9, F1 72.4±3.8 (출처: Table A.1).
- Table A.1, WENO comparison: CAMELYON16 ABMIL AUC 98.7±0.3, AUPRC+ 93.2±1.2; WENO AUC 99.3±0.5, AUPRC+ 90.6±2.9; CAR-MIL L1 AUC 99.9±0.1, AUPRC+ 94.8±1.0 (출처: Table A.1).
- Table A.3, model complexity: ABMIL MACs 5.899G, Params 0.657M; CAR-ABMIL MACs 5.901G, Params 0.657M; DSMIL MACs 5.318G, Params 0.594M; CAR-DSMIL MACs 5.908G, Params 0.659M; TransMIL MACs 24.796G, Params 2.672M; CAR-TransMIL MACs 34.653G, Params 3.722M (출처: Table A.3).
- Sec. 6, attention correlation: representative BRACS WSI에서 factual attention과 ABMIL attention의 Spearman r은 0.67, factual/counterfactual attention의 r은 0.29 (출처: Sec. 6).
- Sec. 6, BRACS slides 전체 correlation: factual attention과 ABMIL attention은 L1에서 r=0.69±0.05, cosine에서 r=0.69±0.03; factual/counterfactual attention은 L1에서 r=-0.67±0.02, cosine에서 r=0.20±0.63 (출처: Sec. 6).

## 5. 우리 프로젝트와의 관련 (가설)
- 가설: 우리 목표가 pathology WSI 벤치마크에서 ABMIL 기준선을 넘는 것이므로, 이 논문의 ABMIL 대비 CAR-MIL 비교는 외부 기준선 설계 또는 attention regularization 후보로 참고할 수 있을 수 있다 (출처: Sec. 4.2, Table 2).
- 가설: 현재 학습 파라미터 허용 전제라면, CAR-MIL처럼 shared encoder/classifier 위에 lightweight counterfactual attention branch를 추가하는 방식이 구조적으로 고려 대상이 될 수 있다 (출처: Sec. 3.2, Table A.3).
- 가설: factual/counterfactual attention이 서로 다른 evidence region에 집중된다는 보고는, 우리 브랜치 간 중복을 줄이는 방향의 auxiliary contrast signal로 영감을 줄 수 있을 수 있다 (출처: Sec. 6, Fig. 5).
- 가설: evidence differential loss와 attention-logit proximity loss의 조합은, 여러 branch가 너무 유사해지지 않도록 branch behavior를 구조화하는 regularization 아이디어로 확장 가능할 수 있다 (출처: Sec. 3.3).
- 가설: AUPRC+, AUPRC±, MORF perturbation curve 같은 attention reliability 평가 지표는 slide-level AUC만 보는 평가에 보완 지표로 도입할 수 있을 수 있다 (출처: Sec. 4.1, Sec. 6, Table 5).
- 가설: CAR-MIL은 ABMIL, DSMIL, TransMIL 등 attention logits을 노출하는 MIL architecture에 통합되므로, 우리 파이프라인에 attention-based MIL branch가 있거나 추가된다면 이 논문의 regularization을 이식해 볼 수 있을 수 있다 (출처: Sec. 5, Table 3, Appendix A.7).
- 옮기기 어려운 점, 가설: 우리 현재 접근이 fold context에서 closed-form ridge로 푸는 in-context classifier라면, CAR-MIL의 핵심인 learned attention logits, softmax attention, gradient-based CE/proximity loss를 직접 사용할 수 없을 수 있다 (출처: Sec. 3.1, Sec. 3.3).
- 옮기기 어려운 점, 가설: CAR-MIL은 embedding-level attention MIL을 전제로 하므로, attention module이 없는 ridge classifier에는 counterfactual attention branch를 그대로 붙일 수 없을 수 있다 (출처: Sec. 3.1, Sec. 3.2).
- 옮기기 어려운 점, 가설: 이 논문은 effective rank, branch duplication, stochastic fitting variance, promotion criterion 같은 문제를 직접 다루지 않으므로, 우리 브랜치 중복과 적합 분산 문제를 해결하는 직접적 방법은 아닐 수 있다 (출처: 본문에서 확인 못함).
- 옮기기 어려운 점, 가설: 이 논문의 성능은 TCGA, CAMELYON16, BRACS, synthetic MNIST-bags에서 보고되며, 우리 WSI 벤치마크와 fold-context 설정에서 동일하게 전이된다는 보장은 본문에서 확인 못함 (출처: Sec. 4, 본문에서 확인 못함).

## 6. 이 논문이 답하지 않는 것
- 우리 프로젝트의 브랜치 유효 랭크 또는 branch duplication을 직접 측정하거나 줄이는 방법은 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- 확률적 적합의 적합 분산이 승격 기준과 어떻게 비교되는지, 또는 CAR-MIL이 그 분산을 줄이는지는 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- closed-form ridge in-context classifier에 CAR-MIL regularization을 적용할 수 있는지는 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- fold context를 활용한 in-context learning 설정에서 CAR-MIL이 유효한지는 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- 추론 시 counterfactual branch를 사용하는지, factual branch만 사용하는지 명시적 inference protocol은 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- 모든 dataset에서 최적 \(\alpha\), \(\lambda\) 값은 본문에서 확인 못함; Fig. 3은 BRCA/LUAD single run에 대한 sensitivity만 제공한다 (출처: Sec. 5, Fig. 3).
- ABMIL을 반드시 넘어서야 한다는 단일 목표를 두고 평가한 것은 아니며, ABMIL을 포함해 여러 MIL baseline과 비교한다 (출처: Sec. 4.2, Table 2).
- 우리 WSI 벤치마크에서의 성능, 우리 fold 설정에서의 성능, 우리 승격 기준 충족 여부는 본문에서 확인 못함 (출처: 본문에서 확인 못함).
