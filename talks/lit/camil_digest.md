# 문헌 브리프 — camil (로컬 모델 요약)

```
url: https://arxiv.org/abs/2305.05314
fetched_from: https://arxiv.org/html/2305.05314
fetched_at: 2026-09-18 13:01:06 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8003` · 2026-09-18 13:08 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 기존 attention-based MIL은 tumor와 neighboring tiles의 context를 무시해 오분류가 생길 수 있다고 문제 제기한다 (출처: Abstract, 1 Introduction).
- CAMIL은 neighbor-constrained attention으로 WSI 내 tile 간 의존성을 고려하고, contextual constraints를 prior knowledge로 MIL 모델에 통합한다 (출처: Abstract, 1 Introduction, 3 Material and methods).
- CAMIL은 TCGA-NSCLC subtyping과 CAMELYON16/CAMELYON17 lymph node metastasis detection에서 test AUC 97.5%, 95.9%, 88.1%를 달성해 다른 state-of-the-art methods보다 우수하다고 주장한다 (출처: Abstract).
- CAMIL은 attention weights로 high diagnostic value regions을 식별해 model interpretability를 향상시킨다고 주장한다 (출처: Abstract, 1 Introduction).
- cancer histopathology에서는 tile의 spatial arrangement이 중요하며, cellular landscape context가 미세한 이상 평가에 도움이 된다고 주장한다 (출처: 1 Introduction).
- CAMIL은 localized tumor의 detection/classification을 개선하고, isolated 또는 noisy instances의 오분류를 줄이려는 목적을 가진다 (출처: 1 Introduction).
- CAMIL은 각 tile의 attention score를 주변 tile들의 attention score를 집계해 recalibrate하며, 고점 tile이 고점 neighborhood에 있으면 중요하고 low-scoring neighborhood의 고점 tile은 noise일 수 있다고 설명한다 (출처: 3 Material and methods).
- Nystromformer는 global context를, neighbor-constrained attention은 local context를 담당하며, 둘의 결합이 성능에 필수적이라고 주장한다 (출처: 3 Material and methods, 5 Ablation studies, 6 Conclusion).
- SimCLR 기반 feature extractor는 neighbor-constrained attention mask의 similarity scores에 중요하며, ImageNet weights만 사용하면 성능이 낮아진다고 주장한다 (출처: A.8 SimCLR ablation study).
- Nystromformer 도입은 slide-level accuracy 향상과 localization 성능 감소 사이의 trade-off를 만들 수 있다고 설명한다 (출처: 4.2 Localization).

## 2. 방법의 핵심
- 구조: WSI preprocessing → tile feature extraction → Nystromformer global context module → neighbor-constrained attention local context module → feature aggregation/slide-level prediction으로 구성된다 (출처: Figure 1, 3 Material and methods).
- WSI preprocessing: tissue region을 자동 segment하고, 256 × 256 non-overlapping tiles로 분할한다; x20 magnification과 Otsu thresholding을 사용했다고 한다 (출처: 3.1 WSI preprocessing, A.3 WSI pre-processing).
- feature extractor: SimCLR 방식으로 ResNet-18을 학습한다; ResNet-18은 ImageNet pretrained이고, projection head는 two hidden layer MLP이다 (출처: 3.2 Feature extractor).
- SimCLR 학습: 같은 tile에 color distortion, zoom, rotation, reflection 중 두 가지 augmentation을 적용하고, NT-Xent loss로 final convolutional block과 projection head를 fine-tune한다 (출처: 3.2 Feature extractor).
- CAMIL 학습 시 feature extractor는 frozen으로 사용되며, tile feature dimension은 1024로 제시된다 (출처: 3.2 Feature extractor, Figure 1).
- neighbor distance: neighboring patches 간 feature distance는 squared difference 기반으로 계산된다고 한다 (출처: 3.2 Feature extractor).
- global context module: Nystromformer를 사용해 self-attention을 landmark 기반으로 approximate하고, 시간 복잡도를 O(n)으로 줄인다고 설명한다 (출처: 3.3 Transformer module to capture global contexts, A.6 Transformer module to capture global contexts).
- local context module: 각 tile은 최대 8개 adjacent patches와 연결된 undirected graph로 모델링된다 (출처: 3.4 Neighbor-constrained attention module to capture local contexts).
- neighbor similarity mask: 인접 tile 간 similarity는 `s_ij = exp(-sqrt((h_i - h_j)^2))`이고, 비인접은 0이다 (출처: 3.4 Neighbor-constrained attention module to capture local contexts).
- neighbor-constrained attention: transformed tile representations에서 Q/K/V를 만들고, query-key dot product에 similarity mask를 element-wise 곱한 뒤 neighbor coefficients를 sum하고 softmax로 attention weight를 만든다 (출처: 3.4 Neighbor-constrained attention module to capture local contexts).
- local feature: 각 tile의 neighbor-constrained feature vector는 `l_i = w_i V(t_i)`로 계산된다 (출처: 3.4 Neighbor-constrained attention module to capture local contexts).
- local-global fusion: `m = σ(l) ⊙ l + (1 - σ(l)) ⊙ t`로 local value vector와 global transformed features를 adaptive blending한다 (출처: 3.5 Feature aggregation and slide-level prediction).
- slide-level prediction: fused vector를 attention pooling로 aggregate한 뒤 sum pooling과 classification layer `W_c`로 slide prediction을 만들고, cross-entropy loss를 사용한다 (출처: 3.5 Feature aggregation and slide-level prediction).
- 학습 방식: slide-level weak supervision으로 학습하며, baselines와 동일한 contrastive learning 기반 feature extractor를 사용했다고 한다 (출처: 4.1 Classification performance, A.3 WSI pre-processing).
- 재현 관련 설정: GTP는 원 논문 설정을 따랐지만 CAMELYON16에서는 memory limitations으로 batch size를 2로 설정했다 (출처: 4.1 Classification performance).
- 하드웨어: 각 실험마다 NVIDIA Tesla P100 GPU 1개를 사용했다고 한다 (출처: A.1 Reproducibility statement).
- 추론 방식: tile features를 Nystromformer와 neighbor-constrained attention으로 변환·가중한 뒤 slide-level score를 산출하고, attention scores를 localization map으로 사용한다 (출처: 3.5 Feature aggregation and slide-level prediction, 4.2 Localization).
- localization mask 생성: model output probabilities에 threshold 0.5를 적용해 tile-level predicted masks를 만든다 (출처: 4.2 Localization).
- optimizer, learning rate, epoch 수, Nystromformer landmark 수 `m`, attention dimension `d_q/d_k/d_v`의 구체적 수치, 전체 hyperparameter 값은 본문에서 확인 못함 (출처: 본문에서 확인 못함).

## 3. 평가 설정
- 데이터셋: CAMELYON16, CAMELYON17, TCGA-NSCLC를 사용한다 (출처: 4 Experiments and Results, A.4 Datasets).
- CAMELYON16: 270 training slides와 129 test slides를 포함하며, pathologist annotation이 있고 일부 slide는 partial annotation이라고 한다 (출처: A.4 Datasets).
- CAMELYON17: 1000 WSIs 중 500 WSI가 labeled/public이고, pN0를 normal, pN0가 아닌 모든 class를 cancerous로 통합해 binary classification으로 만든다 (출처: A.4 Datasets).
- TCGA-NSCLC: LUAD와 LUSC 두 subtype으로 구성된 541 slides이며, CAMELYON16과 달리 annotations가 없다고 한다 (출처: A.4 Datasets).
- data split: CAMELYON16은 270 training WSIs를 5-fold cross-validation으로 80% train/20% validation으로 나누고, official test set 129 WSIs로 평가한다 (출처: A.5 Data splits).
- data split: CAMELYON17은 4-fold validation으로 65% train, 15% validation, 25% test를 사용한다 (출처: A.5 Data splits).
- data split: TCGA-NSCLC는 available images 전체에 5-fold cross-validation을 하고, 각 fold에서 training set을 80% train/20% validation으로 나눈다 (출처: A.5 Data splits).
- 비교 기준선: CLAM-SB, CLAM-MB, TransMIL, DTFD-MIL, DSMIL, GTP를 비교한다 (출처: 4.1 Classification performance).
- ABMIL: ABMIL 이름의 기준선은 본문에서 확인 못함; 본문에는 AB-MIL framework가 언급되고, CLAM-SB와 CLAM-MB가 AB-MIL framework 내 attention-based pooling operator를 사용한다고 설명된다 (출처: 4.1 Classification performance).
- classification 지표: ACC와 AUC를 사용하며, ACC는 threshold 0.5로 결정된다 (출처: 4.1 Classification performance).
- localization 지표: cancerous slides에서는 Dice score, normal slides에서는 tile-level specificity를 사용한다 (출처: 4.2 Localization).
- appendix 지표: ACC, F1, AUC를 제공한다고 하며, F1은 main text에서 space 절약으로 생략되었다 (출처: A.2 Performance metrics).
- localization ground truth: 5th magnification level의 expert tumor delineations를 reference mask로 사용한다 (출처: 4.2 Localization).
- localization tile 기준: tile이 annotated tumor의 20% 이상을 포함하면 cancerous tile로 간주한다 (출처: 4.2 Localization).
- localization prediction source: CAMIL, CLAM-SB, CLAM-MB, TransMIL, DSMIL은 scaled attention scores, DTFD-MIL은 tile-level logits, GTP는 GraphCAM을 사용한다 (출처: 4.2 Localization).

## 4. 정량 결과
- Abstract 수치: TCGA-NSCLC, CAMELYON16, CAMELYON17의 test AUC는 각각 97.5%, 95.9%, 88.1%라고 주장한다 (출처: Abstract).
- Table 1 classification 결과, CAMIL: CAMELYON16 ACC 0.917 ± 0.006, AUC 0.959 ± 0.001; TCGA-NSCLC ACC 0.916 ± 0.007, AUC 0.975 ± 0.003; CAMELYON17 ACC 0.843 ± 0.024, AUC 0.881 ± 0.039 (출처: Table 1).
- Table 1 ablation 결과, CAMIL-L: CAMELYON16 ACC 0.910 ± 0.010, AUC 0.953 ± 0.002; TCGA-NSCLC ACC 0.914 ± 0.011, AUC 0.975 ± 0.004; CAMELYON17 ACC 0.828 ± 0.027, AUC 0.881 ± 0.031 (출처: Table 1).
- Table 1 ablation 결과, CAMIL-G: CAMELYON16 ACC 0.891 ± 0.001, AUC 0.950 ± 0.009; TCGA-NSCLC ACC 0.907 ± 0.012, AUC 0.973 ± 0.004; CAMELYON17 ACC 0.818 ± 0.039, AUC 0.875 ± 0.036 (출처: Table 1).
- Table 1 기준선 중 DTFD-MIL: CAMELYON16 ACC 0.889 ± 0.007, AUC 0.941 ± 0.005; TCGA-NSCLC ACC 0.899 ± 0.010, AUC 0.964 ± 0.003; CAMELYON17 ACC 0.797 ± 0.029, AUC 0.884 ± 0.032 (출처: Table 1).
- Table 1 기준선 중 TransMIL: CAMELYON16 ACC 0.905 ± 0.005, AUC 0.950 ± 0.005; TCGA-NSCLC ACC 0.905 ± 0.011, AUC 0.974 ± 0.003; CAMELYON17 ACC 0.804 ± 0.021, AUC 0.873 ± 0.031 (출처: Table 1).
- Table 1 기준선 중 CLAM-SB: CAMELYON16 ACC 0.877 ± 0.029, AUC 0.933 ± 0.002; TCGA-NSCLC ACC 0.903 ± 0.011, AUC 0.972 ± 0.004; CAMELYON17 ACC 0.802 ± 0.039, AUC 0.849 ± 0.041 (출처: Table 1).
- Table 1 기준선 중 CLAM-MB: CAMELYON16 ACC 0.894 ± 0.010, AUC 0.938 ± 0.005; TCGA-NSCLC ACC 0.904 ± 0.010, AUC 0.973 ± 0.004; CAMELYON17 ACC 0.803 ± 0.036, AUC 0.858 ± 0.046 (출처: Table 1).
- Table 1 기준선 중 GTP: CAMELYON16 ACC 0.883 ± 0.026, AUC 0.921 ± 0.026; TCGA-NSCLC ACC 0.916 ± 0.015, AUC 0.973 ± 0.006; CAMELYON17 ACC 0.800 ± 0.037, AUC 0.762 ± 0.108 (출처: Table 1).
- Table 1 기준선 중 DSMIL: CAMELYON16 ACC 0.874 ± 0.066, AUC 0.949 ± 0.006; TCGA-NSCLC ACC 0.853 ± 0.031, AUC 0.954 ± 0.015; CAMELYON17 ACC 0.815 ± 0.031, AUC 0.863 ± 0.043 (출처: Table 1).
- 본문 설명: CAMIL은 CAMELYON16에서 기존 모델보다 AUC 최소 0.9%, ACC 1.2% 이상, CAMELYON17에서 ACC 3.8% 이상 우수하다고 설명한다 (출처: 4.1 Classification performance).
- 본문 설명: CAMIL은 CAMELYON17 AUC에서 DTFD-MIL에 0.003 뒤처진다고 설명한다 (출처: 4.1 Classification performance).
- Table 2 localization 결과, CAMIL: Dice 0.515 ± 0.058, Specificity 0.980 ± 0.040 (출처: Table 2).
- Table 2 localization 결과, DTFD-MIL: Dice 0.525 ± 0.033, Specificity 0.999 ± 0.001 (출처: Table 2).
- Table 2 localization 결과, TransMIL: Dice 0.103 ± 0.004, Specificity 0.999 ± 0.001 (출처: Table 2).
- Table 2 localization 결과, CLAM-SB: Dice 0.459 ± 0.037, Specificity 0.987 ± 0.008 (출처: Table 2).
- Table 2 localization 결과, CLAM-MB: Dice 0.406 ± 0.007, Specificity 0.573 ± 0.045 (출처: Table 2).
- Table 2 localization 결과, GTP: Dice 0.418 ± 0.068, Specificity 0.851 ± 0.116 (출처: Table 2).
- Table 2 localization 결과, DSMIL: Dice 0.259 ± 0.083, Specificity 0.863 ± 0.043 (출처: Table 2).
- Table 3 F1 결과, CAMIL: CAMELYON16 F1 0.881 ± 0.010 (출처: Table 3).
- Table 3 F1 결과, CAMIL-L: CAMELYON16 F1 0.872 ± 0.016 (출처: Table 3).
- Table 3 F1 결과, CAMIL-G: CAMELYON16 F1 0.866 ± 0.012 (출처: Table 3).
- Table 3 F1 결과, TransMIL: CAMELYON16 F1 0.888 ± 0.018 (출처: Table 3).
- Table 4 F1 결과, CAMIL: TCGA-NSCLC F1 0.918 ± 0.005 (출처: Table 4).
- Table 4 F1 결과, CAMIL-L: TCGA-NSCLC F1 0.921 ± 0.009 (출처: Table 4).
- Table 4 F1 결과, CAMIL-G: TCGA-NSCLC F1 0.918 ± 0.007 (출처: Table 4).
- Table 4 F1 결과, GTP: TCGA-NSCLC F1 0.917 ± 0.016 (출처: Table 4).
- Table 5 SimCLR ablation: ImageNet pretrained ResNet-18만 사용하고 SimCLR fine-tuning을 하지 않은 CAMIL은 CAMELYON16 ACC 0.723 ± 0.095, AUC 0.743 ± 0.077; TCGA-NSCLC ACC 0.692 ± 0.082, AUC 0.798 ± 0.059 (출처: Table 5).
- CAMELYON17 F1 수치는 본문에서 확인 못함 (출처: 본문에서 확인 못함).

## 5. 우리 프로젝트와의 관련 (가설)
- neighbor-constrained attention의 8-neighbor similarity mask는 우리 브랜치 중복을 줄이는 local regularizer로 작용할 수 있다는 가설이 있다; 이유는 고점 tile이 low-scoring neighborhood에 있으면 noise로 억제하는 구조이기 때문이지만, 이 논문은 effective rank를 측정하지 않아 rank 감소 효과는 확인 불가하다 (출처: 3.4 Neighbor-constrained attention module to capture local contexts, 4.1 Classification performance).
- Nystromformer 기반 global branch와 neighbor-constrained attention 기반 local branch의 역할 분리는 우리 브랜치 설계에 참고가 될 수 있다는 가설이 있다; 이유는 CAMIL-G와 CAMIL-L ablation에서 global/local 구성요소가 각각 다른 성능 패턴을 보이고 결합 모델이 최적이었기 때문이지만, 브랜치 간 중복도와 직접 연결은 본문에 없다 (출처: 5 Ablation studies, Table 1).
- frozen feature extractor로 만든 neighbor distance는 학습 파라미터가 적은 in-context 분류기에 저비용 context weight로 재사용될 수 있다는 가설이 있다; 이유는 feature extractor가 CAMIL 학습에서 frozen이고 neighbor similarity가 feature distance 기반으로 계산되기 때문이지만, closed-form ridge in-context classifier와의 호환성은 본문에서 확인 못함 (출처: 3.2 Feature extractor, 3.4 Neighbor-constrained attention module to capture local contexts).
- feature extractor 품질이 neighbor mask 성능에 영향을 줄 수 있다는 가설이 있다; 이유는 SimCLR fine-tuning 없이 ImageNet weights만 쓰면 CAMELYON16과 TCGA-NSCLC에서 성능이 크게 낮아졌기 때문이지만, 우리 feature extractor와 동일하지 않아 직접 전이는 어렵다 (출처: A.8 SimCLR ablation study, Table 5).
- CAMIL 전체 구조를 우리 closed-form ridge in-context classifier에 그대로 옮기기 어렵다는 가설이 있다; 이유는 learned Q/K/V, Nystromformer, slide-level cross-entropy training을 포함하는 학습 기반 MIL 구조이기 때문 (출처: 3.3 Transformer module to capture global contexts, 3.4 Neighbor-constrained attention module to capture local contexts, 3.5 Feature aggregation and slide-level prediction).
- 확률적 적합의 적합 분산 문제를 직접 해결하는 기법으로 옮기기 어렵다는 가설이 있다; 이유는 본문에 probabilistic fitting, fit variance, promotion criterion 관련 분석이 없기 때문 (출처: 본문에서 확인 못함).
- ABMIL 기준선을 넘는 근거로 바로 쓰기 어렵다는 가설이 있다; 이유는 본문에 ABMIL 이름의 기준선이 없고, AB-MIL framework 내 CLAM-SB/MB 등 다른 기준선과만 비교했기 때문 (출처: 4.1 Classification performance).

## 6. 이 논문이 답하지 않는 것
- ABMIL 기준선과의 직접 비교: ABMIL 이름의 기준선 성능은 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- effective rank 또는 브랜치 중복도: CAMIL이 tile/branch representation의 유효 랭크를 어떻게 변화시키는지 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- 확률적 적합의 fit variance: probabilistic fitting, 적합 분산, promotion criterion 관련 결과는 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- 우리 벤치마크 성능: 이 논문은 CAMELYON16, CAMELYON17, TCGA-NSCLC만 평가하며, 우리 프로젝트의 특정 벤치마크 성능은 본문에서 확인 못함 (출처: 4 Experiments and Results, A.4 Datasets).
- closed-form ridge in-context classifier와의 비교: in-context learning 또는 closed-form ridge classifier 관련 실험은 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- 계산 비용 상세: runtime, memory usage, tile 수별 비용, inference latency는 본문에서 확인 못함; Nystromformer의 O(n) 복잡도와 P100 1개 사용 사실만 확인됨 (출처: A.6 Transformer module to capture global contexts, A.1 Reproducibility statement).
- 핵심 hyperparameters: optimizer, learning rate, epoch 수, Nystromformer landmark 수, attention dimension 수치, regularization 설정은 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- CAMELYON17 F1: CAMELYON17 F1 수치는 본문에서 확인 못함 (출처: 본문에서 확인 못함).
- localization trade-off의 일반성: Nystromformer가 slide-level accuracy와 localization 사이의 trade-off를 만든다는 설명은 CAMELYON16 localization 중심으로 제시되며, 다른 데이터셋으로의 일반성은 본문에서 확인 못함 (출처: 4.2 Localization).
