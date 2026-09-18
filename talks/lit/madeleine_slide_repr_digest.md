# 문헌 브리프 — madeleine_slide_repr (로컬 모델 요약)

```
url: https://arxiv.org/abs/2411.13623
fetched_from: https://arxiv.org/html/2411.13623
fetched_at: 2026-09-18 10:50:35 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8003` · 2026-09-18 12:50 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 병리 WSI 표현 학습은 주로 MIL 기반 약감독에 의존해 특정 임상 과제에 맞춘 슬라이드 표현을 만든다고 주장한다 (출처: Abstract).
- 패치 단위 SSL foundation model은 성공했지만, 환자/슬라이드 레벨 임베딩 생성은 여전히 어렵다고 주장한다 (출처: Abstract).
- Cobra는 여러 foundation model의 tile embeddings을 통합하는 단일 모달리티 feature-space SSL 슬라이드 인코더를 제안한다 (출처: Abstract, 1 Introduction).
- Cobra는 Mamba-2와 multi-head gated attention, contrastive loss를 사용해 slide-level embeddings을 만든다고 주장한다 (출처: Abstract, 1 Introduction).
- Cobra는 TCGA 3048 WSIs로만 사전학습하고, CPTAC 4개 코호트에서 state-of-the-art slide encoders보다 평균 최소 +4.4% AUC를 낸다고 주장한다 (출처: Abstract).
- Cobra는 추론 시训练中 unseen feature extractors와 호환된다고 주장한다 (출처: Abstract).
- Cobra는 stochastic image augmentations 없이 frozen patch embeddings으로 학습/배포된다고 주장한다 (출처: 1 Introduction).
- Cobra는 3048 WSIs라는 소규모 사전학습 데이터로 SOTA unsupervised slide representations를 만든다고 주장한다 (출처: 1 Introduction, 표1).
- Cobra는 patch-level foundation models, 포함训练中 unseen models을 추가 fine-tuning 없이 더 나은 slide-level feature extractors로 만들 수 있다고 주장한다 (출처: 1 Introduction).
- Cobra는 다양한 WSI magnifications에서 배포 가능하고, 낮은 magnification이 계산 효율을 크게 높이며 downstream 성능 손실은 작다고 주장한다 (출처: 1 Introduction).
- Cobra는 FM-agnostic이고 task-agnostic인 slide representation learning 방법이라고 주장한다 (출처: 5 Conclusion).

## 2. 방법의 핵심
- 입력 전처리: WSI를 224×224 px 패치로 tessellate하고, Canny background detection으로 배경 tile을 제거한 뒤 pretrained foundation models로 patch embeddings을 추출한다. 사용 FM은 CTransPath, UNI, Virchow2, H-Optimus-0이며, embedding 차원은 768, 1024, 1280, 1536이다. magnification은 0.5, 1.14, 2 microns per pixel을 사용한다 (출처: 3.1 Preprocessing).
- 구조: FM별 embedding MLP가 LN → Lin → SiLU → Lin 형태로 다른 FM embedding 차원을 shared embedding space로 투영한다. 이후 두 개의 Mamba-2 state-space dual layer를 통과하고, multi-head gated attention으로 slide embedding을 만든다 (출처: 3.2 Architecture).
- 공식 구조: slide encoder는 z = f_A(f_S(f_E(H_fe)))로 정의된다. f_E는 embedding module, f_S는 state-space dual module, f_A는 aggregation module이다 (출처: 3.2 Architecture, Eq. 1–5).
- 학습 방식: MoCo-v3 스타일 contrastive self-supervised learning을 사용한다. InfoNCE loss, cosine similarity, temperature 0.2, momentum 0.99를 사용하며, 동일 환자의 다른 slide, patch, foundation model, magnification에서 생성된 embeddings을 정렬한다 (출처: 3.3 Inference modes, 3.4 Contrastive loss function).
- 재현 관련 사전학습 설정: batch size 1024, 2000 epochs, 4개 NVIDIA A100 GPU에서 약 40시간, 총 36576개 extracted feature embeddings을 사용했다. 3048 WSIs × 4개 FM × 3개 magnification에 해당한다 (출처: 4.2 Pretraining setup).
- 재현 관련 하이퍼파라미터: heads 8, Mamba-2 layers 2, embedding dimension 768, input dimensions 768/1024/1280/1536, dropout 0.25, attention hidden dimension 96, teacher momentum 0.99, contrastive loss temperature 0.2, optimizer AdamW, learning rate 5e-4, warmup epochs 50, weight decay 0.1, epochs 2000, batch size 1024, tile embeddings per patient 768 (출처: 부록 A 표5).
- 추론 방식: single-FM inference mode에서는 encoded embeddings로 attention weights를 계산하지만, 최종 slide embedding은 original patch embeddings의 가중 평균으로 만든다. 기본 Cobra는 Virchow2 patch embeddings을 입력으로 쓰는 single-FM mode를 가리킨다 (출처: 3.3 Inference modes, Eq. 6–7).
- multi-FM inference mode: 모든 training FM의 projected embeddings를 평균한 뒤 encode하고, attention weights를 계산한 뒤 선택된 primary FM의 original embeddings에 적용한다 (출처: 3.3 Inference modes, Eq. 8, 부록 B Inference modes).
- unseen FM inference mode: Cobra 사전학습 FM embeddings로 shared space에서 attention weights를 계산한 뒤,训练中 unseen FM의 original patch embeddings에 그 weights를 적용한다. 재학습 없이 unseen FM을 slide-level aggregator로 사용할 수 있다고 설명한다 (출처: 부록 B Inference modes, 4.5 Inference ablations).
- downstream 분류기: slide embedding에 2-layer MLP를 붙여 5-fold cross-validation으로 학습한다. MLP는 input 768, hidden 256, SiLU, dropout 50%, class-weighted cross-entropy, AdamW learning rate 0.0001, weight decay 0.01, one-cycle policy 32 epochs를 사용한다 (출처: 부록 A.1.1 MLP downstream classification).
- linear probing: sklearn logistic regression, L2 regularization 1.0, lbfgs solver, max iterations 10000, balanced class weights, stratified sampling, 10 random runs, class당 5/10/25 samples을 사용한다 (출처: 부록 A.1.2 Linear probing).

## 3. 평가 설정
- 사전학습/학습 데이터: TCGA 3048 WSIs, 2848 patients. 세부 코호트는 TCGA-BRCA 1112, TCGA-CRC 566, TCGA-LUAD 524, TCGA-LUSC 496, TCGA-STAD 350 (출처: 4.1 Dataset).
- 외부 검증 데이터: CPTAC 1604 WSIs, 444 patients. 세부 코호트는 CPTAC-BRCA 395, CPTAC-COAD 233, CPTAC-LUAD 498, CPTAC-LUSC 478 (출처: 4.1 Dataset).
- 전체 데이터 규모: TCGA+CPTAC 합계 4652 WSIs, 3292 patients. TCGA 3048 WSIs는 Cobra 사전학습과 classifier 학습에, CPTAC 1604 WSIs는 외부 검증에만 사용된다 (출처: 부록 C Data).
- TCGA-STAD는 Cobra 사전학습에만 사용되고 downstream task에는 사용되지 않는다 (출처: 부록 C Data).
- 평가 과제: 15개 classification tasks. NSCLC Subtyping; LUAD STK11, EGFR, TP53, KRAS; BRCA ESR1, PGR, ERBB2, PIK3CA; COAD MSI, BRAF, LN, KRAS, Side, PIK3CA (출처: 4.3 Tasks, 표2).
- 지표: 본문에서는 AUC를 주로 보고하고, 부록에서는 AUPRC, F1 score, balanced accuracy를 추가로 보고한다 (출처: 4.3 Tasks, 부록 D Results).
- 평가 프로토콜: TCGA training cohort에서 5-fold cross-validation으로 MLP classifier를 학습하고, 5개 classifier를 CPTAC 전체 external validation set에 배포한다 (출처: 4.4 Evaluation of patient embeddings).
- few-shot 설정: high-performance tasks에서 class당 5/10/25 samples, 10 runs로 linear probing을 수행한다. 대상 과제는 NSCLC Subtyping, LUAD STK11/EGFR/TP53, BRCA ESR1/PGR/ERBB2, COAD BRAF/Sidedness/MSI (출처: 4.4 Evaluation of patient embeddings, 부록 A.1.2 Linear probing).
- magnification: 0.5 microns per pixel (20×), 1.14 microns per pixel (9×), 2 microns per pixel (5×)을 평가한다 (출처: 4.3 Tasks).
- 비교 기준선: mean patch embeddings, Ensemble Prediction, Concatenated mean embeddings, slide encoders GigaPath-SE, MADELEINE, CHIEF, PRISM을 비교한다 (출처: 표2, 4.4 Evaluation of patient embeddings).
- ABMIL 기준선: ABMIL은 관련 연구에서 WSI 분류의 state-of-the-art로 언급되지만, 본문의 평가 비교 기준선 목록과 표에는 ABMIL이 없다 (출처: 2 Related work, 4.4 Evaluation of patient embeddings, 표2).
- data leakage 통제: Cobra와 사용된 FM들이 downstream 평가 데이터셋에 포함되지 않은 데이터로 사전학습되었다고 명시한다 (출처: 4.1 Dataset).

## 4. 정량 결과
- 표1: Cobra는 파라미터 15M, 사전학습 WSI 3K, patch FM은 CTransPath, UNI, Virchow2, H-Optimus-0을 사용한다 (출처: 표1).
- 표2: 평균 AUC는 Cobra 75.3, PRISM 70.9, Virchow2 mean 73.8, Concatenated 74.1, Ensemble Prediction 73.6, GigaPath-SE 62.8, MADELEINE 66.1, CHIEF 66.2이다 (출처: 표2).
- 표2: Cobra의 과제별 AUC 예시는 NSCLC ST 98.1, LUAD EGFR 80.0, BRCA ESR1 89.6, COAD MSI 94.1, COAD BRAF 87.8이다 (출처: 표2).
- Cobra는 PRISM보다 평균 AUC +4.4, Virchow2 mean patch embeddings보다 +1.5 높다 (출처: 4.4 Evaluation of patient embeddings).
- COAD MSI와 BRAF에서 Cobra는 다른 slide encoders보다 최소 +15%와 +24.2% average AUC를 낸다고 보고한다 (출처: 4.4 Evaluation of patient embeddings).
- BRCA에서 Cobra는 MADELEINE 대비 ESR1 +9.5, PGR +5.5, ERBB2 +4.9, PIK3CA -1.3 AUC이다 (출처: 4.4 Evaluation of patient embeddings).
- 표3: inference/magnification별 평균 AUC는 Cobra†-V2-20× 75.4, Cobra†-V2-5× 74.7, Cobra†-V2-9× 74.6, Cobra†-GP 74.0, GigaPath mean 71.5, PRISM 70.9이다 (출처: 표3).
- Cobra†-V2-5×와 Cobra†-V2-9×는 PRISM 대비 각각 +3.8%와 +3.7% average AUC 개선이라고 보고한다 (출처: 4.5 Inference ablations).
- combined inference mode는 2 microns per pixel에서 single-FM mode 대비 평균 +0.53% AUC 개선이라고 보고한다 (출처: 4.5 Inference ablations).
-训练中 unseen GigaPath embeddings에 Cobra를 적용하면 GigaPath mean baseline 대비 +2.5% average AUC, PRISM 대비 +3.1% average AUC 개선이라고 보고한다 (출처: 4.5 Inference ablations).
- Cobra-CTP는 CTransPath 기반이며, CTransPath는 약 30M 파라미터, Virchow2는 600M 이상, H-optimus-0는 1B 이상 파라미터라고 설명한다 (출처: 4.5 Inference ablations).
- 표4/4.6: multi-magnification pretraining은 single 0.5 MPP pretraining 대비 multi-FM inference mode에서 평균 +1.73% AUC 개선이다 (출처: 4.6 Pretraining ablation, 표4).
- 5× magnification NSCLC subtyping에서 multi-magnification pretraining 개선은 UNI +6.8, CTransPath +7.9, H-optimus-0 +8.1, Virchow2 +1.7 AUC이다 (출처: 4.6 Pretraining ablation, 표4).
- 표12: few-shot k=5 average AUC는 Cobra†-V2 67.3, PRISM 67.0, Virchow2 mean 62.4이다 (출처: 표12).
- 표13: few-shot k=10 average AUC는 Cobra†-V2 72.1, PRISM 70.3, Virchow2 mean 67.0이다 (출처: 표13).
- 표14: few-shot k=25 average AUC는 Cobra†-V2 75.7, PRISM 72.8, Virchow2 mean 71.9이다 (출처: 표14).
- 표6: encoded embeddings를 최종 가중 평균에 사용하는 Cobra-ENC는 평균 AUC 61.8이고, original embeddings를 사용하는 Cobra-V2는 75.3, Cobra†-V2는 75.4이다 (출처: 표6, 부록 D.1 Full Classification).
- ABMIL 대비 Cobra 또는 Cobra embedding 기반 classifier의 수치: 본문에서 확인 못함 (출처: 본문 전체).
- effective rank, fit variance, promotion criterion 관련 수치: 본문에서 확인 못함 (출처: 본문 전체).

## 5. 우리 프로젝트와의 관련 (가설)
- Cobra가 768-dim slide embedding을 생성하므로, 이 embedding을 fold/slide 단위 고정 feature로 사용해 closed-form ridge 또는 in-context classifier 입력으로 쓰는 실험을 설계할 수 있을 수 있다. 단, Cobra 논문 자체는 ridge/in-context classifier를 사용하지 않는다 (출처: 부록 A 표5, 4.4 Evaluation of patient embeddings).
- Cobra는 4개 FM과 3개 magnification을 feature-space augmentation으로 사용한다. 우리 쪽 branch 중복/유효 랭크 저하 문제와 연결하면, 서로 다른 FM 또는 magnification에서 나온 frozen embeddings를 branch pool로 쓰는 것이 유효 랭크를 높이는 가설을 테스트할 수 있을 수 있다 (출처: 3.1 Preprocessing, 4.6 Pretraining ablation).
- 표2에서 Concatenated mean embeddings가 Cobra와 근접한 평균 AUC 74.1을 보여, 여러 FM embeddings를 단순 병렬/concatenate하는 것만으로도 branch 다양성 효과가 있을 수 있다는 가설을 검토할 수 있다. 단, 이는 우리 벤치마크에서 재확인해야 한다 (출처: 표2).
- Cobra의 multi-head gated attention weights는 slide embedding 생성에 직접 사용되는 tile weights이므로, lightweight tile aggregator 또는 interpretability weight로 차용할 수 있을 수 있다. 단, Cobra weights는 unsupervised pretraining으로 학습된 것이므로 우리 fold context와 직접 호환된다는 보장은 없다 (출처: 3.2 Architecture, 4.7 Interpretability).
- Cobra는 낮은 magnification에서 계산 효율을 크게 높인다고 보고하므로, 우리 벤치마크에서 tile 수/추론 비용 절감 가설에 참고할 수 있다. 단, 성능 손실 크기는 우리 데이터셋에서 재확인해야 한다 (출처: 4.5 Inference ablations, 표3).
- Cobra의 few-shot linear probing 결과는 low-label 또는 in-context setting에서 slide embedding이 유용할 수 있다는 가설에 참고할 수 있다. 단, Cobra의 few-shot은 TCGA→CPTAC external validation 설정이지, 우리 fold context in-context 프로토콜은 아니다 (출처: 4.4 Evaluation of patient embeddings, 표12–14).
- Cobra는 15M 파라미터와 contrastive pretraining을 필요로 하므로, 학습 파라미터를 쓰지 않던 제약이나 closed-form ridge만 쓰는 설정과 직접 호환되지는 않을 수 있다. 다만 학습 파라미터 제약이 폐기되었다면 Cobra-style learned slide encoder를 비교 대상으로 삼을 수 있을 수 있다 (출처: 표1, 3.4 Contrastive loss function).
- Cobra는 ABMIL을 평가 기준선으로 삼지 않았다. 따라서 Cobra 또는 Cobra embedding이 ABMIL을 넘는지에 대한 직접 증거는 이 논문에서 얻을 수 없다 (출처: 2 Related work, 4.4 Evaluation of patient embeddings, 표2).
- Cobra는 patient/slide-level embedding을 생성하지만, fold context, fold-level in-context classification, closed-form ridge, effective rank, fit variance, promotion criterion을 다루지 않는다. 따라서 우리 문제의 핵심 지표에 대한 직접 이전 가능성은 가설 수준이다 (출처: 본문 전체).
- Cobra의 contrastive pretraining은 stochastic training을 포함하지만, downstream fit variance를 줄이는지에 대한 수치는 보고되지 않는다. 따라서 확률적 적합의 fit variance 문제를 해결할 수 있을지는 알 수 없다 (출처: 3.4 Contrastive loss function, 4.4 Evaluation of patient embeddings).

## 6. 이 논문이 답하지 않는 것
- ABMIL 대비 Cobra 또는 Cobra embedding 기반 classifier의 성능: 본문에서 확인 못함 (출처: 본문 전체).
- fold context 또는 fold-level in-context classification 프로토콜: 본문에서 확인 못함 (출처: 본문 전체).
- closed-form ridge classifier를 사용한 결과: 본문에서 확인 못함 (출처: 본문 전체).
- branch 유효 랭크, effective rank, branch 중복도 측정: 본문에서 확인 못함 (출처: 본문 전체).
- 확률적 적합의 fit variance 또는 승격 기준 대비 variance: 본문에서 확인 못함 (출처: 본문 전체).
- promotion criterion 또는 그와 유사한 모델 선택 기준: 본문에서 확인 못함 (출처: 본문 전체).
- 학습 파라미터를 쓰지 않는 zero-parameter classifier 결과: 본문에서 확인 못함 (출처: 본문 전체).
- 우리 프로젝트의 벤치마크 데이터셋, 과제, 코호트, fold 분할: 본문에서 확인 못함 (출처: 본문 전체).
- Cobra embedding을 ridge/in-context classifier에 넣었을 때의 calibration, uncertainty, per-fold variance: 본문에서 확인 못함 (출처: 본문 전체).
- Cobra 추론 시 exact FLOPs, memory, latency 수치: 본문에서 확인 못함 (출처: 본문 전체).
