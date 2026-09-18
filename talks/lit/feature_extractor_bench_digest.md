# 문헌 브리프 — feature_extractor_bench (로컬 모델 요약)

```
url: https://arxiv.org/abs/2311.11772
fetched_from: https://arxiv.org/html/2311.11772
fetched_at: 2026-09-18 10:50:32 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8003` · 2026-09-18 10:53 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 저자들은 stain normalisation과 image augmentation을 생략해도 성능 저하가 없으며, 메모리와 계산 부담이 줄어든다고 주장한다. (출처: Abstract; Section IV-C, IV-D, V)
- 저자들은 downstream slide-level 성능에서 가장 중요한 요인은 feature extractor 선택이라고 주장한다. (출처: Abstract; Section IV-B, V)
- 저자들은 top feature extractor의 경우 저배율, 약 9× / 1.14 MPP 슬라이드로 slide-level 분류에 충분하다고 주장한다. (출처: Abstract; Section IV-F, V)
- 저자들은 Lunit-DINO, UNI, CTransPath가 task-averaged downstream 성능에서 가장 좋은 feature extractor라고 주장한다. (출처: Section IV-B, V; Table I; Fig. 6)
- 저자들은 slidewise 또는 patchwise Macenko stain normalisation, rotation/flip, 전체 augmentation 모두 일관된 AUROC 이득을 보이지 않는다고 주장한다. (출처: Section IV-C, IV-C1, IV-D; Fig. 7; Appendix C)
- 저자들은 downstream aggregation model 중 AttMIL이 가장 좋지만, transformer와 mean pooling과의 차이는 작고 분산이 크다고 주장한다. (출처: Section IV-E; Fig. 8)
- 저자들은 pathology-specific SSL feature extractor가 ImageNet baseline보다 일반적으로 더 좋은 downstream 성능을 보인다고 주장한다. (출처: Section IV-B, V)
- 저자들은 stain normalisation을 계속 사용할 경우 계산 비용 때문에 patchwise 방식을 권장한다. (출처: Section IV-C1, V)
- 저자들은 augmentation을 사용하면 feature extraction과 training overhead가 크게 증가하며, 특히 all augmentations은 training 시간을 30배, rotation만은 5배 늘린다고 보고한다. (출처: Section F-B2)
- 저자들은 14개 feature extractor, 9개 task, 5개 dataset, 3개 downstream architecture, 2개 magnification에 걸쳐 10,000회 이상 training runs을 수행했다고 주장한다. (출처: Abstract)

## 2. 방법의 핵심
- 구조: WSI를 겹치지 않는 patch로 나누고, 각 patch에 augmentation을 선택적으로 적용한 뒤 frozen feature extractor로 patch feature를 추출한다. 이후 aggregation function이 patch features를 slide-level feature로 모으고, linear classifier가 slide label을 예측한다. (출처: Section I-A; Fig. 1)
- patch 설정: patch size는 P=224이며, 10× magnification에서 slide당 patch 수는 보통 1,000에서 10,000 사이로 기술된다. (출처: Section I-A)
- feature extractor: 14개 공개 pathology feature extractor를 frozen으로 사용한다. CTransPath, Phikon-S, Phikon-T, UNI, Lunit-DINO, RetCCL, Lunit-BT, Lunit-SwAV, Lunit-MoCo와 ImageNet baseline인 Swin, ViT-S, ViT-B, ViT-L, ResNet-50이 포함된다. (출처: Section IV-A1; Table III)
- aggregation model 1, AttMIL: patch embedding을 먼저 512-dim linear layer와 ReLU로 변환한 뒤, patch별 256-dim tanh hidden layer 두 개로 attention score를 계산하고 softmax로 정규화한 가중 평균을 slide embedding으로 사용한다. (출처: Section IV-A1; Section F-A2)
- aggregation model 2, two-layer transformer: decoder-only 2-layer transformer를 사용하며, hidden units 512, attention heads 8, dropout 0.1, GELU feedforward, attention 전 layer normalisation, masking 없음, 출력 token 평균으로 slide embedding을 만든다. (출처: Section IV-A1; Section F-A3)
- aggregation model 3, mean pooling: patch embedding을 512-dim linear layer와 ReLU로 변환한 뒤 단순 평균으로 slide embedding을 만든다. (출처: Section IV-A1; Section F-A1)
- classifier: slide embedding에 dropout 0.5를 적용한 뒤, 클래스 수에 맞는 linear layer와 softmax를 사용하고 cross-entropy loss를 사용한다. (출처: Section F-A)
- 학습 방식: AdamW optimizer, initial learning rate 1e-3, weight decay 1e-2, batch size 1, gradient accumulation 4, cosine annealing schedule, 최대 30 epochs, validation loss가 10 epochs 동안 개선되지 않으면 early stopping을 사용한다. (출처: Section IV-A3; Section F)
- 데이터 분할: training set의 80%를 training, 20%를 validation에 사용하며, 5개 random seeds로 실험한다. CAMELYON17은 leave-one-hospital-out cross-validation을 사용한다. (출처: Section IV-A3; Appendix A-B)
- preprocessing/augmentation: none, Macenko slidewise, Macenko patchwise, rotation/flip, all augmentations을 비교한다. augmentation은 training images에만 적용하며, stain normalisation 실험에서는 test set에도 같은 normalisation을 적용한다. (출처: Section IV-A3; Appendix D-A)
- 추론 방식: training 전에 patch features를 미리 추출하고 disk에 cache한다. augmentation 실험에서는 각 patch의 27개 augmented version features도 미리 추출해 둔다. training과 inference에서는 image를 다시 feature extraction하지 않고 cached features를 로드해 aggregation과 classification만 수행한다. (출처: Section IV-A3; Section F-B1)
- bag 처리: patient를 bag, patch를 instance로 보고, patient당 최대 8,192개 patch를 sampling한다. (출처: Section F)

## 3. 평가 설정
- 데이터셋: TCGA-BRCA, CPTAC-BRCA, CAMELYON17, TCGA-CRC, CPTAC-COAD를 사용한다. (출처: Section IV-A2; Table II; Appendix A-B)
- breast task: TCGA-BRCA를 train/val, CPTAC-BRCA를 test로 사용해 BRCA-subtype, BRCA-CDH1, BRCA-TP53, BRCA-PIK3CA를 예측한다. BRCA-subtype은 5-way classification이고, 나머지는 binary task로 처리된다. (출처: Section IV-A2; Appendix A-A)
- lymph node task: CAMELYON17에서 centre-wise cross-validation으로 BRCA-LN status를 예측한다. 원본 CAMELYON17의 virtual patient label 대신 slide-level label을 사용한다. (출처: Section IV-A2; Appendix A-B)
- colorectal task: TCGA-CRC를 train, CPTAC-COAD를 test로 사용해 CRC-MSI, CRC-KRAS, CRC-BRAF, CRC-SMAD4를 예측한다. (출처: Section IV-A2; Appendix A-A)
- dataset size: Table II에서 BRCA는 833 train, 208 val, 120 test; CAMELYON17은 320 train, 80 val, 100 test; CRC는 558 train, 110 test로 제시된다. (출처: Table II)
- feature extractor 비교 대상: CTransPath, Phikon-S, Phikon-T, UNI, Lunit-DINO, RetCCL, Lunit-BT, Lunit-SwAV, Lunit-MoCo, Swin, ViT-S, ViT-B, ViT-L, ResNet-50 총 14개이다. (출처: Table III)
- downstream architecture 비교 대상: AttMIL, two-layer transformer, mean pooling 총 3개이다. (출처: Section IV-A1)
- preprocessing 비교 대상: none, Macenko slidewise, Macenko patchwise, rotation/flip, all augmentations이다. (출처: Section IV-A3; Appendix D-A)
- magnification: 1.14 MPP, 약 9×와 0.5 MPP, 20×를 비교한다. (출처: Section IV-F)
- 지표: test AUROC를 기본 지표로 사용하고, feature extractor 상대 성능 비교를 위해 normalised differential AUROC score를 사용한다. 이 score는 낮을수록 좋다. (출처: Section IV-B; Eq. 3)
- stain/augmentation 효과 측정: stain normalisation 또는 augmentation 유무에 따른 test AUROC 차이를 bootstrap으로 추정한다. (출처: Section IV-C, IV-D)
- ABMIL: 본문에서 “ABMIL”이라는 이름은 확인 못함. 대신 “AttMIL”이 downstream aggregation model로 포함된다. (출처: Section IV-A1; Section F-A2)

## 4. 정량 결과
- 총 training model 수: low magnification 9,450개, high magnification 1,260개, 합계 10,710개이다. (출처: Section F-B3)
- GPU 시간: 총 8,140 GPU hours, 단일 GPU 기준 339.2 days; low magnification 5,990 GPU hours / 249.6 days, high magnification 2,150 GPU hours / 89.6 days이다. (출처: Section F-B3)
- augmentation overhead: all augmentations은 no augmentation 대비 training 시간이 30배, rotation augmentation만은 5배로 보고된다. (출처: Section F-B2)
- normalised differential AUROC 계산 규모: 5 seeds와 14 feature extractors 기준 5^14 ≈ 6.1×10^9 iterations per task; 5 seeds와 18 feature extractor-magnification 조합 기준 5^18 ≈ 3.8×10^12 iterations per task이다. (출처: Section F-C)
- Table I, AttMIL, no augmentation, low magnification task-averaged normalised differential AUROC: Lunit-DINO 0.047±0.030, UNI 0.048±0.034, CTransPath 0.053±0.034, RetCCL 0.074±0.035, Phikon-T 0.084±0.043, Lunit-MoCo 0.099±0.040, Phikon-S 0.100±0.049, Lunit-SwAV 0.102±0.052, ViT-B 0.117±0.042, ViT-S 0.138±0.052, ViT-L 0.143±0.062, ResNet-50 0.156±0.054, Swin 0.163±0.047, Lunit-BT 0.187±0.085이다. (출처: Table I)
- Table IX, AttMIL, no augmentation, low magnification seed-averaged test AUROC 예시: UNI는 BRCA-subtype 0.85±0.02, BRCA-TP53 0.85±0.02, BRCA-LN 0.93±0.04, CRC-MSI 0.87±0.04이다. (출처: Table IX)
- Table IX, AttMIL, no augmentation, low magnification seed-averaged test AUROC 예시: Lunit-DINO는 CRC-MSI 0.90±0.02, CRC-BRAF 0.76±0.04이다. (출처: Table IX)
- Table IX, AttMIL, no augmentation, low magnification seed-averaged test AUROC 예시: CTransPath는 BRCA-CDH1 0.81±0.02, CRC-MSI 0.82±0.03이다. (출처: Table IX)
- stain normalisation bootstrap 설정: feature extractor와 downstream model 고정 시 9 task × 5 seeds = 45 task-seed pair, 각 pair당 25회 resampling, 총 1,125 bootstraps를 사용한다. (출처: Section IV-C)
- augmentation 수: patchwise stain normalisation을 포함해 27개 augmentation을 연구한다. (출처: Appendix D)
- feature embedding size: CTransPath 768, Phikon-S 768, Phikon-T 768, UNI 1024, Lunit-DINO 384, RetCCL 2048, Lunit-BT 2048, Lunit-SwAV 2048, Lunit-MoCo 2048이다. (출처: Table III)
- high magnification 상대 순위: high magnification에서 top 3 feature extractor는 UNI, Phikon-T, Lunit-DINO 순으로 기술된다. 수치 표는 본문에서 확인 못함. (출처: Section IV-F1; Fig. 9)
- low/high magnification 비교: UNI와 Lunit-DINO는 두 magnification에서 상대 성능이 매우 유사하며, top 2 간 차이는 variance보다 작다고 기술된다. (출처: Section IV-F2; Fig. 10)

## 5. 우리 프로젝트와의 관련 (가설)
- 가설: pathology-specific SSL feature extractor, 특히 Lunit-DINO, UNI, CTransPath를 frozen feature로 사용하면, 우리 in-context 분류기의 입력 표현 품질이 개선되어 branch 간 중복이나 적합 불안정성이 완화될 수 있다. 근거는 이 논문에서 feature extractor 선택이 downstream 성능의 가장 큰 요인이라고 한 점이다. (출처: Section IV-B, V)
- 가설: strong frozen feature 위에서 mean pooling과 AttMIL의 차이가 작으므로, closed-form ridge 또는 단순 slide-level summary 기반 in-context 분류기가 학습된 MIL aggregator와 경쟁할 가능성이 있다. 다만 이 논문은 closed-form ridge나 in-context 학습을 평가하지 않았다. (출처: Section IV-E; Fig. 8; Table IV)
- 가설: stain normalisation과 augmentation을 제거하면 preprocessing 파이프라인의 변동 요인과 계산 비용이 줄어들 수 있다. 그러나 이 논문이 말하는 성능 무영향은 downstream trained model 기준이며, 우리 확률적 적합 분산 문제까지 직접 보장하지 않는다. (출처: Section IV-C, IV-D, F-B2)
- 가설: normalised differential AUROC score처럼 seed permutation을 고려한 상대 성능 지표는, seed나 fold에 따른 변동이 큰 우리 승격 기준 평가에 참고할 수 있다. (출처: Section IV-B; Eq. 3; Section F-C)
- 가설: external validation cohort와 SSL pretraining data와 겹치지 않는 test set 설계는, 우리 벤치마크에서 feature extractor 또는 in-context 분류기의 과적합 가능성을 점검하는 데 참고할 수 있다. (출처: Section IV-A2)
- 옮기기 어려운 점: 이 논문은 frozen feature + trained downstream aggregator 설정을 전제로 하며, fold context에서 동작하는 in-context classifier, closed-form ridge, branch rank 제어, effective rank 축소, 확률적 적합 분산 제어, 또는 ABMIL 대비 in-context 분류기 비교를 다루지 않는다. (출처: Section I-A, IV-A1, IV-B, V)
- 옮기기 어려운 점: 이 논문의 aggregation model은 모두 학습 가능한 downstream network 또는 learned projection을 포함한다. 따라서 “학습 파라미터 0개” 또는 매우 제한된 학습 파라미터를 가진 in-context classifier와 직접 동일하게 해석할 수 없다. (출처: Section IV-A1; Section F-A)

## 6. 이 논문이 답하지 않는 것
- fold context에서 closed-form ridge in-context classifier가 AttMIL 또는 다른 학습 기반 MIL보다 좋은지 여부는 본문에서 확인 못함.
- 다중 branch 구조에서 effective rank가 낮거나 branch 간 중복이 큰 문제를 줄이는 방법은 본문에서 확인 못함.
- 확률적 적합으로 인한 적합 분산을 승격 기준보다 낮게 제어하는 방법은 본문에서 확인 못함.
- in-context learning, prompt-based classification, zero-shot 또는 few-shot slide-level classification 설정은 본문에서 확인 못함.
- 우리 프로젝트의 특정 WSI benchmark, task, fold 분할, 승격 기준에 대해 어떤 feature extractor가 최적인지는 본문에서 확인 못함.
- branch 수, branch rank, low-rank regularization, branch fusion, 또는 rank-aware aggregation은 본문에서 확인 못함.
- calibration, uncertainty quantification, 또는 seed/fold 간 신뢰도 평가는 본문에서 확인 못함.
- ABMIL이라는 이름의 baseline과 직접 비교한 결과는 본문에서 확인 못함.
