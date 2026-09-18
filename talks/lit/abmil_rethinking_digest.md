# 문헌 브리프 — abmil_rethinking (로컬 모델 요약)

```
url: https://arxiv.org/abs/2404.00351
fetched_from: https://arxiv.org/html/2404.00351
fetched_at: 2026-09-18 12:45:03 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8000` · 2026-09-18 12:49 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- ABMIL의 attention mechanism은 instance를 구별하는 데 한계가 있고, 조직을 잘못 분류해 MIL 성능을 저하시킬 수 있다고 주장한다 (출처: Abstract).
- ABMIL에서 attention level만으로는 instance attribute를 신뢰할 수 없으며, positive patch와 negative patch 모두 bag prediction에 기여할 수 있다고 주장한다 (출처: Introduction).
- ABMIL의 최종 예측은 attention pooling과 bag classification head의 결합으로 결정되므로, attention만으로 instance contribution을 측정하는 것은 불완전하고 오도할 수 있다고 주장한다 (출처: Introduction, III-B).
- AttriMIL은 attention pooling과 bag classification head를 통합한 attribute scoring mechanism으로 각 instance의 bag prediction 기여도를 정량화한다고 주장한다 (출처: Abstract, III-B, Fig. 1, Fig. 2).
- attribute score의 부호는 instance attribute 방향을, 절대값은 네트워크가 해당 instance에 부여한 강조 수준을 나타낸다고 주장한다 (출처: III-B).
- spatial attribute constraint는 slide 내부 instance 간 공간적 상관관계를 모델링해 tumor localization과 bag classification을 개선한다고 주장한다 (출처: Abstract, III-C, IV-C1, Fig. 5).
- attribute ranking constraint는 slide 간 instance 상관관계를 모델링하고 positive/negative instance의 attribute score 차이를 강조해 hard instance를 구분한다고 주장한다 (출처: Abstract, III-D, IV-C2, Fig. 5).
- histopathology adaptive backbone은 사전학습 모델의 여러 단계에 adapter를 배치해 병리 특징 추출 능력을 높인다고 주장한다 (출처: Abstract, III-E, Fig. 3).
- 세 공개 벤치마크에서 AttriMIL이 기존 state-of-the-art framework보다 여러 평가 지표에서 우수하다고 주장한다 (출처: Abstract, IV-D, Table III).
- attention과 attribute score의 분포 위치 관계는 U-shape이며, attention은 뚜렷한 positive/negative 영역에 집중하는 경향이 있다고 보고한다 (출처: V-A, Fig. 7).
- AttriMIL은 모든 subtype branch의 bag score가 0보다 작으면 OOD bag으로 판단할 수 있는 가능성을 제시한다 (출처: V-B, Fig. 8).

## 2. 방법의 핵심
- 구조: WSI를 patch로 crop하고 histopathology adaptive backbone으로 instance feature를 추출한 뒤, 각 subtype branch에서 multi-class attribute scoring mechanism으로 instance attribute score를 생성한다 (출처: Fig. 2, IV-A1, IV-B).
- subtype branch: tumor detection task에서는 tumor/normal branch를 사용하고, 각 branch에서 해당 subtype WSI를 positive, 다른 subtype WSI를 negative로 취급한다 (출처: Fig. 2).
- attribute scoring: ABMIL의 unnormalized attention score \(u_i\)와 bag classification head weight \(c\)를 이용해 instance attribute score를 \(s_i = u_i h_i c\)로 정의한다 (출처: III-B, Eq. 7).
- bag prediction: bag prediction은 normalized instance attribute score의 합과 learnable bias로 표현되며, ABMIL의 순차적 attention 계산과 bag prediction을 병렬적 attribute scoring으로 변환한다고 설명한다 (출처: III-B, Eq. 6).
- spatial attribute constraint: WSI 내 인접 instance의 attribute score 차이를 평균하는 loss를 사용하고, 좌표와 이웃은 전처리에서 기록하며, 가장자리 결측 이웃은 해당 instance 자신으로 처리한다 (출처: III-C, Eq. 8).
- attribute ranking constraint: positive bag의 최대 attribute score가 negative bag의 최대 attribute score보다 크도록 hinge loss를 구성하고, multi-class task에서는 각 subtype branch에 positive bank와 negative bank를 두고 capacity \(K=4\)로 top-K instance를 사용해 branch별 ranking loss를 평균한다 (출처: III-D, Eq. 10, Eq. 11).
- histopathology adaptive backbone: ImageNet pre-trained ResNet-50의 첫 3개 블록을 feature extractor로 사용하고, batch normalization을 group normalization으로 교체한다 (출처: IV-B).
- adapter 구조: adapter는 bottleneck 구조로 두 개의 fully connected layer와 중간 activation layer로 구성되며, pre-trained network의 서로 다른 stage 뒤에 배치한다 (출처: III-E, Fig. 3).
- progressive training: adapter를 네트워크 깊이 순서대로 단계별로 학습하고, 현재 adapter 학습 시 global pooling을 사용한다 (출처: III-E, Fig. 3).
- 학습 loss: 전체 loss는 \( \mathcal{L} = \mathcal{L}_{ce} + \alpha \mathcal{L}_{spatial} + \beta \mathcal{L}_{rank} \)이며, 기본값은 \(\alpha=0.1\), \(\beta=0.001\)이다 (출처: IV-B, Eq. 12).
- 최적화 설정: Adam optimizer, constant learning rate 2e-4, mini-batch size 1 bag, PyTorch 1.10.0, Nvidia GeForce RTX 3090 GPU를 사용한다 (출처: IV-B).
- patch 전처리: Camelyon16은 20x에서 256×256 non-overlapping patches로 crop하고, TCGA-NSCLC는 foreground를 256×256 non-overlapping patches로 crop해 WSI당 약 15,000 patches를 얻으며, UniToPatho는 224×224 non-overlapping patches로 crop한다 (출처: IV-A1).
- 추론: score aggregation으로 C개의 bag score를 얻고 bag prediction probabilities를 생성하며, bag prediction에 해당하는 instance attribute scores를 tumor localization에 매핑한다 (출처: Fig. 2).
- OOD 판단: 각 branch의 aggregated bag score가 모두 0보다 작으면 해당 WSI를 OOD bag으로 판단하는 방식을 제시한다 (출처: V-B).

## 3. 평가 설정
- 데이터셋: Camelyon16, TCGA-NSCLC, UniToPatho의 세 공개 데이터셋을 사용한다 (출처: IV-A1).
- Camelyon16: breast cancer detection dataset으로 399 slides, 2 classes이며, official split은 training/testing 270/129 slides이다 (출처: IV-A1).
- TCGA-NSCLC: non-small cell lung cancer dataset으로 444 patients에서 507 LUDA slides, 452 patients에서 486 LUSC slides를 포함하며, 4-fold cross-validation을 사용하고 patient-level split으로 training:validation:testing = 60:15:25를 적용한다 (출처: IV-A1).
- UniToPatho: colon cancer dataset으로 292 WSIs에서 추출한 9,536 H&E patches, 6 classes이며, official split은 training/testing 204/88 slides이다 (출처: IV-A1).
- 평가 지표: class-wised average accuracy(ACC), F1-Score, AUC를 사용하고, AUC를 주요 지표로 Delong test를 수행한다 (출처: IV-A2).
- 반복 실행: Camelyon16과 UniToPatho에서는 4회 실행 후 평균 metrics를 보고하고, TCGA-NSCLC에서는 4-fold cross-validation 평균 metrics를 보고한다 (출처: IV-A2).
- 비교 기준선: Mean-Pooling, Max-Pooling, MIL-RNN, DSMIL, CLAM-SB, CLAM-MB, DGMIL, TransMIL, DTFD-MIL, PMIL, ABMIL을 비교한다 (출처: Table III).
- ABMIL 포함 여부: ABMIL이 비교 기준선에 포함된다 (출처: Table III).
- 추가 비교: histopathology adaptive backbone의 adapter 구성과 SimCLR, MoCov2를 Table II에서 비교한다 (출처: Table II).
- 비교 방식: 공정한 비교를 위해 기존 방법의 released code를 published instance features와 함께 재실행해 결과를 얻었다고 한다 (출처: IV-D).
- UniToPatho 비교 제외: DGMIL과 PMIL은 multi-subtype classification task를 고려하지 않아 Table III의 UniToPatho 결과에 포함되지 않는다 (출처: IV-D1).

## 4. 정량 결과
- Table I에서 Camelyon16 baseline, 즉 \(\alpha=0\), \(\beta=0\)의 AUC는 86.17%이고, best AUC는 \(\alpha=0.100\), \(\beta=0.001\)에서 91.31%이다 (출처: Table I).
- Table I에서 TCGA-NSCLC baseline, 즉 \(\alpha=0\), \(\beta=0\)의 AUC는 94.36%이고, best AUC는 \(\alpha=0.010\), \(\beta=1.000\)에서 95.49%이다 (출처: Table I).
- Table I caption은 baseline vs. best의 p-values가 모두 0.05보다 작다고 보고한다 (출처: Table I).
- 본문은 \(\alpha=0\), \(\beta=0.001\) 설정에서 baseline 대비 Camelyon16 AUC가 2.35%, TCGA-NSCLC AUC가 0.67% 개선된다고 보고한다 (출처: IV-C2).
- 본문은 spatial attribute constraint와 attribute ranking constraint의 조합이 baseline AUC를 Camelyon16에서 5.14%, TCGA-NSCLC에서 1.10% 높였다고 보고한다 (출처: IV-C2, Table I).
- Table II에서 adapter1 + adapter2 + adapter3 구성은 Camelyon16에서 ACC 86.63%, AUC 91.12%, TCGA-NSCLC에서 ACC 89.50%, AUC 95.11%를 기록한다 (출처: Table II).
- Table II에서 SimCLR은 Camelyon16에서 ACC 86.87%, AUC 89.95%, TCGA-NSCLC에서 ACC 89.41%, AUC 95.03%를 기록한다 (출처: Table II).
- Table II에서 MoCov2는 Camelyon16에서 ACC 86.21%, AUC 88.67%, TCGA-NSCLC에서 ACC 89.05%, AUC 94.88%를 기록한다 (출처: Table II).
- Table II에서 adapter3 only는 Camelyon16에서 ACC 84.49%, AUC 86.17%, TCGA-NSCLC에서 ACC 88.45%, AUC 94.36%를 기록한다 (출처: Table II).
- Table II caption은 adapter3 vs. adapter1 + adapter2 + adapter3의 p-values가 모두 0.05보다 작다고 보고한다 (출처: Table II).
- Table III에서 Camelyon16 ABMIL은 ACC 84.49±1.65%, F1-Score 76.74±3.06%, AUC 88.75±3.15%이고, AttriMIL은 ACC 90.69±1.02%, F1-Score 87.23±1.78%, AUC 93.90±1.23%이다 (출처: Table III).
- Table III에서 TCGA-NSCLC ABMIL은 ACC 88.90±0.51%, F1-Score 88.59±0.79%, AUC 94.95±0.30%이고, AttriMIL은 ACC 90.38±2.32%, F1-Score 90.24±2.21%, AUC 96.13±1.20%이다 (출처: Table III).
- Table III에서 UniToPatho ABMIL은 ACC 57.38±1.71%, F1-Score 44.27±4.45%, AUC 85.37±0.37%이고, AttriMIL은 ACC 66.92±3.41%, F1-Score 55.97±4.71%, AUC 88.99±1.31%이다 (출처: Table III).
- Table III caption은 ABMIL vs. AttriMIL의 p-values가 모두 0.05보다 작다고 보고한다 (출처: Table III).
- 본문은 Camelyon16 positive WSI에서 tumor area의 평균 비율이 10% 미만이라고 보고한다 (출처: IV-D1).
- 본문은 TCGA-NSCLC에서 AttriMIL이 다른 방법 대비 ACC 0.88%, AUC 0.61% 증가했다고 보고한다 (출처: IV-D1).
- Fig. 4와 본문은 vanilla training 과정에서 spatial attribute loss가 350,000을 초과할 수 있다고 보고한다 (출처: IV-C1, Fig. 4).
- Fig. 4와 본문은 attribute ranking loss가 training 말미에 약 2,000 수준으로 접근한다고 보고한다 (출처: IV-C2, Fig. 4).
- 본문은 OOD detection 실험에서 threshold를 2로 설정하면 더 많은 OOD samples을 식별할 수 있다고 보고한다 (출처: V-B, Fig. 8).

## 5. 우리 프로젝트와의 관련 (가설)
- 옮겨올 수 있는 것, 가설 1: attribute scoring의 아이디어를 fold context ridge의 bag/instance contribution 진단 지표로 사용할 수 있을지 확인해 볼 가치가 있다. 가설적으로 ridge coefficient가 bag head weight \(c\)에 대응한다면, attention-like weight와 feature-projection score의 곱이 branch 간 중복이나 hard instance를 드러내는 신호가 될 수 있다. 단, 이 논문은 학습된 ABMIL head를 전제로 하므로 closed-form ridge에 직접 적용됨을 보장하지는 않는다.
- 옮겨올 수 있는 것, 가설 2: spatial attribute constraint를 patch 좌표가 보존된 WSI fold에 적용해 인접 patch score를 매끄럽게 regularize하는 방식이 branch specialization을 도울 수 있을지 확인해 볼 가치가 있다. 가설적으로 공간적 smoothness가 중복된 branch의 score 패턴을 분리하거나 hard negative를 줄이는 데 기여할 수 있다. 단, fold context에 공간 좌표와 인접 관계가 없으면 직접 이식하기 어렵다.
- 옮겨올 수 있는 것, 가설 3: attribute ranking constraint를 positive/negative bag 또는 fold 간 hard instance 구분용 auxiliary loss로 사용할 수 있을지 확인해 볼 가치가 있다. 가설적으로 max-score hinge와 bank 기반 ranking이 승격 후보 branch의 positive/negative separation을 강화할 수 있다. 단, 이 논문은 slide-level bag label과 learned MIL을 전제로 하며, 확률적 적합의 분산이나 closed-form ridge의 승격 기준을 직접 다루지는 않는다.
- 옮겨올 수 있는 것, 가설 4: histopathology adaptive backbone의 progressive adapter가 multi-level feature로 branch 간 중복을 줄이는 데 도움이 될 수 있을지 확인해 볼 가치가 있다. 가설적으로 stage별 adapter가 서로 다른 수준에서 특징을 분리해 branch effective rank를 높이는 데 기여할 수 있다. 단, 이 논문은 feature extractor 개선이지 branch rank 문제를 직접 해결하지는 않는다.
- 옮겨올 수 있는 것, 가설 5: OOD 판단 방식, 즉 모든 branch score가 0보다 작으면 거부하는 방식을 불확실 fold 또는 branch 승격 보류 신호로 사용할 수 있을지 확인해 볼 가치가 있다. 가설적으로 branch score의 부호와 크기가 uncertain sample을 식별하는 데 유용할 수 있다. 단, 이 논문의 OOD는 slide-level all-branch negative 기준이며, 확률적 적합 분산 기준과는 다르다.
- 옮기기 어려운 것, 가설 1: 이 논문의 핵심은 학습된 attention, bag head, end-to-end loss를 전제로 하므로, 현재 closed-form ridge in-context classifier에 그대로 적용하기는 어려울 수 있다. 학습 파라미터 제약이 폐기됐다고 해도, 이 논문의 loss가 ridge의 closed-form 해를 유지하는지 본문에서 확인 못함.
- 옮기기 어려운 것, 가설 2: spatial attribute constraint는 WSI grid 좌표와 인접 patch를 필요로 하므로, fold context가 공간 좌표와 인접 관계를 보존하지 않으면 직접 이식하기 어렵다.
- 옮기기 어려운 것, 가설 3: attribute ranking constraint는 positive/negative bag label과 bank capacity \(K=4\) 등 slide-level weak supervision을 전제로 하므로, fold-level 승격 기준이나 probabilistic fitting variance를 직접 규제하지는 않을 수 있다.
- 옮기기 어려운 것, 가설 4: 이 논문의 성능 향상은 Camelyon16, TCGA-NSCLC, UniToPatho에서 ABMIL 대비 학습된 AttriMIL 결과이지, 우리 fold context ridge 기준선 대비 이득을 보장하지는 않는다.

## 6. 이 논문이 답하지 않는 것
- 이 논문은 branch 유효 랭크, branch 간 중복, effective rank 저하 문제를 다루지 않는다.
- 이 논문은 closed-form ridge, in-context classifier, fold context, probabilistic fitting variance, promotion criterion을 비교하거나 분석하지 않는다.
- attribute scoring, spatial attribute constraint, attribute ranking constraint가 branch effective rank나 적합 분산에 어떤 영향을 주는지 본문에서 확인 못함.
- 학습 파라미터 0개 또는 최소 파라미터 in-context 분류기와의 비교가 본문에서 확인 못함.
- ABMIL 기준선 대비 향상은 보고하지만, 우리 프로젝트의 fold context ridge 기준선 대비 향상은 본문에서 확인 못함.
- OOD detection은 slide-level all-branch score <0 기준이지, 확률적 적합의 승격 분산 기준과 동일한지 본문에서 확인 못함.
- branch 수 증가 시 계산 비용, runtime, memory, 분산 영향은 본문에서 확인 못함.
