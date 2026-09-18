# 문헌 브리프 — titan_slide_fm (로컬 모델 요약)

```
url: https://arxiv.org/abs/2411.19666
fetched_from: https://arxiv.org/html/2411.19666
fetched_at: 2026-09-18 10:50:32 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8001` · 2026-09-18 10:55 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- TITAN은 335,645 WSI 시각 SSL과 423,122 합성 ROI 캡션, 182,862 slide-report vision-language alignment로 사전학습되어, fine-tuning이나 clinical labels 없이 slide representation과 보고서 생성을 일반화한다고 주장한다 (출처: Abstract).
- TITAN은 ROI foundation model과 slide foundation model보다 linear probing, few-shot/zero-shot classification, rare cancer retrieval, cross-modal retrieval, pathology report generation에서 우수하다고 주장한다 (출처: Abstract).
- patch-level SSL recipe를 WSI slide-level로 확장하면 general-purpose slide representations를 얻을 수 있다고 주장한다 (출처: Introduction, Discussion).
- 8,192×8,192 ROI crop에서 학습하고 ALiBi로 전체 WSI로 extrapolate하는 train-short/test-long 전략이 slide-level 성능에 중요하다고 주장한다 (출처: Results, Online Methods).
- Stage 1 vision-only pretraining은 vision-language alignment만 하는 TITAN_L보다 좋은 초기화와 downstream 성능을 제공한다고 주장한다 (출처: Algorithmic design considerations, Figure 2D).
- fine-grained ROI captions과 coarse slide reports를 모두 aligning하는 것이 멀티스케일 slide 정보 처리에 필요하다고 주장한다 (출처: Discussion, Figure 3C).
- TITAN/TITAN_V slide embeddings은 low-data regime과 rare cancer retrieval에서 강하다고 주장한다 (출처: Results, Discussion).
- TITAN/TITAN_V에서 linear probe가 task-specific ABMIL보다 우수하며, 이는 대규모 멀티모달 pretraining이 task-specific supervised slide embedding보다 downstream에 유리하다고 주장한다 (출처: Comparison with different learning paradigms, Figure 2E).
- 같은 pretraining recipe에서 ViT+ALiBi가 ABMIL/MH-ABMIL architecture보다 우수하다고 주장한다 (출처: Algorithmic design considerations, Figure 3C).

## 2. 방법의 핵심
- 구조: TITAN slide encoder는 ViT 6 layers, 12 attention heads, head dimension 64, embedding dimension 768, hidden dimension 3,072이며, patch embedding layer 대신 MLP를 사용하고 2D feature grid에 2D ALiBi를 적용한다 (출처: Online Methods, Network architecture and training details; Results).
- 입력: WSI를 tissue segmentation/tiling 후 512×512 pixel patches at 20×로 추출하고 CONCHv1.5로 768-dimensional patch features를 추출한다. CONCHv1.5는 1.26 million image-caption pairs로 CoCa objective 20 epochs 학습된 patch encoder이다 (출처: Online Methods, Preprocessing).
- 학습 Stage 1: Mass-340K 335,645 WSIs에서 iBOT을 사용한다. 16×16 feature grid, 즉 8,192×8,192 pixel region crop에서 two global 14×14 crops와 ten local 6×6 crops를 생성하고, vertical/horizontal flipping과 posterize feature augmentation를 적용한다. 270 epochs, 91,260 iterations, 4 NVIDIA A100 80GB, local batch size 256 per GPU로 학습한다 (출처: Online Methods, Unimodal visual pretraining).
- 학습 Stage 2: 423,122 pairs of 8,192×8,192 ROIs와 PathChat synthetic captions으로 CoCa vision-language alignment를 수행한다. 8 NVIDIA A100 80GB, local batch size 196 per GPU, gradient accumulation 2, effective batch size 3,136을 사용한다 (출처: Online Methods, Vision-language continual pretraining).
- 학습 Stage 3: 182,862 pairs of WSIs와 pathology reports로 CoCa vision-language alignment를 수행한다. WSI를 64×64 feature grids, 즉 32,768×32,768 pixels로 random crop하고, local batch size 16 per GPU, gradient accumulation 2, effective batch size 256을 사용한다. vision backbone에는 smaller learning rate, smaller weight decay, slow warm-up을 적용한다 (출처: Online Methods, Vision-language continual pretraining).
- CoCa 구조: image encoder 위에 contrastive query 1개와 reconstruction queries 128개를 가진 attentional pooler를 추가한다. text encoder와 multimodal decoder는 CONCHv1.5 pretrained weights를 사용하고, 각각 12 Transformer layers, embedding dimension 768, hidden dimension 3,072이다 (출처: Online Methods, Network architecture and training details).
- 추론: 8,192×8,192 ROI crop에서 pretraining한 뒤 ALiBi로 full WSI inference를 수행한다. slide embedding은 linear probe, kNN/prototype, zero-shot classification, slide retrieval, cross-modal retrieval, report generation에 사용된다 (출처: Results, Online Methods).
- linear probe: frozen slide embeddings에 logistic regression을 적합한다. scikit-learn L-BFGS solver를 사용하고, validation set이 있으면 l2 regularization을 10^-6~10^5 사이 45 logarithmically spaced values에서 tuning하며 max iterations 500을 사용한다. validation set이 없으면 default l2=1, max iterations 1,000을 사용한다 (출처: Online Methods, Linear and K-nearest neighbors probe evaluation).
- kNN/prototype: SimpleShot은 class별 slide embedding average prototype을 사용하고, 20-nearest neighbors는 k=20을 사용한다. embeddings은 centered and normalized되고, Euclidean distance를 사용한다 (출처: Online Methods, Linear and K-nearest neighbors probe evaluation).
- zero-shot: CLIP-style 방식으로 class text prompt embeddings와 slide embedding의 cosine similarity를 계산하고, prompt ensemble을 사용한다 (출처: Online Methods, Zero-shot slide classification).
- report generation: CoCa text decoder를 사용하고, beam search decoding strategy with 5 beams and 1 beam group을 사용한다 (출처: Online Methods, Report generation).

## 3. 평가 설정
- pretraining 데이터셋: Mass-340K, 335,645 WSIs, 20 organs, H&E 90.9%, IHC 9.1%, neoplastic 70.0%, non-neoplastic 30.0% (출처: Results, Figure 1A, Online Methods).
- morphology/subtyping 데이터셋: TCGA-UT-8K 25,495 8,192×8,192 ROIs, 32 classes; TCGA-OT 11,186 WSIs, 46 OncoTree classes; OT108 5,564 WSIs, 108 classes; EBRAINS 2,319 WSIs, 30 classes; BRACS, PANDA, IMP-CRC, CRANE, renal allograft rejection 등 (출처: Results, Online Methods, Extended Data Tables 1-2, 9-10).
- molecular/survival 데이터셋: BCNB, MUT-HET, TCGA/CPTAC/EBRAINS molecular tasks, MGB internal molecular tasks, six TCGA site-preserving survival cohorts (출처: Results, Online Methods, Extended Data Tables 11-12).
- report/retrieval 데이터셋: TCGA-Slide-Reports 10,108 slide-report pairs; Rare-Cancer 186 cancer types, 19,626 WSIs, 43 rare cancer types 3,039 WSIs, 143 common cancer types 16,587 WSIs; Rare-Cancer-Public 127 cancer types, 14,062 WSIs, 29 rare cancer types 1,982 WSIs, 98 common cancer types 12,080 WSIs (출처: Results, Online Methods, Extended Data Tables 5-8).
- 비교 기준선: PRISM, GigaPath, CHIEF, mean pooling, ABMIL, finetuning, TITAN_L, ABMIL-L, MH-ABMIL. ABMIL이 기준선에 있다 (출처: Results, Online Methods, Baselines).
- ABMIL 설정: batch size 1, AdamW optimizer, weight decay 10^-5, cosine annealing learning rate scheduler, peak learning rate 10^-4, 20 epochs. patch encoder는 각 slide encoder framework에 맞게 선택된다 (출처: Online Methods, Supervised baselines).
- 지표: balanced accuracy, weighted F1-score, AUROC, quadratic-weighted Cohen’s κ, concordance index, Acc@K with K={1,3,5}, MVAcc@5, Recall@K with K={1,3,5,10}, mean recall, METEOR, ROUGE-1/ROGUE, BLEU-1 (출처: Online Methods, Evaluation metrics).

## 4. 정량 결과
- pretraining data scale: full Mass-340K TITAN_V가 12.5%, 25%, 50% Mass-340K 대비 4 subtyping tasks average +3.65%, +3.21%, +1.21% (출처: Results, Figure 2A).
- parameter efficiency: TITAN/TITAN_V는 48.5M/42.1M parameters이고, PRISM/GigaPath는 99.0M/86.3M parameters이다. 본문은 TITAN/TITAN_V가 더 무거운 PRISM/GigaPath보다 우수하다고 보고한다 (출처: Results, Figure 2B).
- morphology: TITAN/TITAN_V가 PRISM 대비 multi-class/binary subtyping tasks average +8.4%/+6.7%이다. TCGA-UT-8K +6%/+7.5%, TCGA-OT +7%/+9.5%, OT108 +10%/+16%, EBRAINS +9%/+9.1% (출처: Results, Figure 2C).
- prototype/kNN: TITAN/TITAN_V가 PRISM 대비 SimpleShot +14%/+9.5%, 20-nearest-neighbor +15%/+9.2% (출처: Results, Figure 2C).
- grading: TITAN이 CHIEF 대비 +3.2%, PRISM 대비 +4% quadratic-weighted Cohen’s κ average (출처: Results, Figure 2C).
- molecular: TITAN이 PRISM 대비 BCNB/MUT-HET +0.9%, TCGA +1.7%, internal BRCA/LUAD +3.7% averaged AUROC이다. IHC quantification에서 TITAN_V가 CHIEF 대비 +12%, TITAN이 CHIEF 대비 +7.3% quadratic-weighted Cohen’s κ이다. external evaluation에서 TITAN_V가 PRISM 대비 +0.9%, TITAN은 -0.3%이다 (출처: Results, Extended Data Figure 2).
- survival: TITAN/TITAN_V가 CHIEF 대비 disease-specific survival c-index +3.62%/+2.90% (출처: Results, Extended Data Table 55).
- ablation: ALiBi가 absolute positional encoding 대비 +2.01%이다. 6 layers, 43M parameters가 4 layers, 29M parameters 대비 +1.72%, 12 layers, 86M parameters 대비 +1.31%이다. TITAN이 TITAN_L 대비 +2.35%, ABMIL-L 대비 +3.62%이다 (출처: Results, Figure 2D, Extended Data Table 70).
- learning paradigms: ABMIL이 mean pooling보다 우수하고, linear probe가 ABMIL보다 우수하다고 보고한다. TITAN finetuning은 linear probe보다 대부분 우수하며, random initialization finetuning은 pretrained weights finetuning 대비 average -3.63%이다. GigaPath finetuning은 linear probe 대비 +14.9% 개선되지만 TITAN/TITAN_V linear probe 대비 -6.25%/-4.17%이다. CHIEF finetuning은 linear probe 대비 average -8.38%이다 (출처: Results, Figure 2E, Extended Data Figure 4, Extended Data Tables 56-59).
- few-shot: TITAN/TITAN_V가 same patch encoder ABMIL보다 all settings에서 우수하다. 1-shot OT108에서 TITAN이 ABMIL 대비 +56.7%이다. TITAN/TITAN_V가 CHIEF 16-shot 대비 TCGA-UT-8K +22.4%/+13.5%, TCGA-OT +18.7%/+6.8%이다 (출처: Results, Figure 2F, Extended Data Tables 72-75).
- zero-shot: TITAN이 PRISM 대비 multi-class balanced accuracy +56.52%, binary AUROC +13.8%이다. 30-class EBRAINS subtyping에서 TITAN balanced accuracy가 PRISM 대비 +121.9%이다 (출처: Results, Figure 3B, Extended Data Tables 80-92).
- zero-shot ablation: Stage 1 vision-pretraining removal -0.4%, Stage 2 ROI-caption alignment removal -3.6%, Stage 3 slide-report alignment removal -7.3%이다. MH-ABMIL variant는 TITAN with/without vision-pretraining 대비 -1.94%/-1.54%이다 (출처: Results, Figure 3C, Extended Data Tables 93-96).
- report generation: TITAN이 PRISM 대비 METEOR/ROGUE/BLEU average +161% (출처: Results, Figure 3D).
- rare cancer retrieval: TITAN/TITAN_V가 PRISM 대비 Accuracy@K +14.8%/+12.3%, MVAcc@K +18.1%/+13.4% (출처: Results, Figure 4A, Extended Data Table 129).
- cross-modal retrieval: TITAN이 PRISM 대비 report-to-slide +10.5%, slide-to-report +20.5%이다. slide-to-report retrieval에서 K=1일 때 +36.4%이고, single report retrieval 성능은 0.75로 보고된다 (출처: Results, Figure 4D, Extended Data Tables 138-139).
- Extended Data Table 14, TCGA-UT-8K linear probe balanced accuracy: TITAN 0.832±0.0056, TITAN_V 0.820±0.0055, mean pool CONCH 0.779±0.0064, PRISM 0.774±0.0062, GigaPath 0.700±0.0069, CHIEF 0.625±0.0079.
- Extended Data Table 15, TCGA-OT linear probe balanced accuracy: TITAN 0.587±0.0103, TITAN_V 0.558±0.0104, PRISM 0.508±0.0110, mean pool CONCH 0.476±0.0112, GigaPath 0.543±0.0178, CHIEF 0.528±0.0205.
- Extended Data Table 16, OT108 linear probe balanced accuracy: TITAN 0.735±0.0204, TITAN_V 0.732±0.0208, PRISM 0.674±0.0200, mean pool CONCH 0.644±0.0215, GigaPath 0.680±0.0217, CHIEF 0.598±0.0237.
- Extended Data Table 17, EBRAINS linear probe: TITAN AUROC 0.945±0.0174, balanced accuracy 0.848±0.0293; TITAN_V AUROC 0.948±0.0146, balanced accuracy 0.848±0.0376; PRISM AUROC 0.944±0.0241, balanced accuracy 0.850±0.0398; mean pool CONCH AUROC 0.927±0.0229, balanced accuracy 0.776±0.0626.
- Extended Data Table 18, TCGA-BRCA linear probe AUROC: TITAN 0.982±0.0058, TITAN_V 0.979±0.0059, PRISM 0.977±0.0058, mean pool CONCH 0.965±0.0147. CPTAC external: TITAN_V 0.986±0.0008, TITAN 0.985±0.0010, PRISM 0.973±0.0036.
- Extended Data Table 19, TCGA-NSCLC linear probe AUROC: TITAN 0.942±0.0222, TITAN_V 0.920±0.0331, CHIEF 0.913±0.0268, PRISM 0.866±0.0360. CPTAC-DHMC external: TITAN 0.918±0.0073, TITAN_V 0.910±0.0169, CHIEF 0.822±0.0143, PRISM 0.753±0.0508.
- Extended Data Table 21, BRACS fine subtype linear probe balanced accuracy: TITAN 0.702±0.0395, TITAN_V 0.679±0.0712, PRISM 0.658±0.0393, CHIEF 0.638±0.0547.
- Extended Data Table 23, CRANE cellular rejection linear probe AUROC: TITAN 0.915±0.0238, TITAN_V 0.880±0.0293, PRISM 0.820±0.0350, CHIEF 0.813±0.0336.
- Extended Data Table 28, PANDA Gleason grading quadratic-weighted Cohen’s κ: TITAN 0.854±0.0108, TITAN_V 0.856±0.0194, PRISM 0.860±0.0204, CHIEF 0.842±0.0302.
- Extended Data Table 31, SETD2 mutation linear probe AUROC: TITAN 0.920±0.0191, TITAN_V 0.917±0.0248, PRISM 0.901±0.0365, CHIEF 0.880±0.0200.
- Extended Data Table 34, HER2 linear probe AUROC: TCGA TITAN 0.972±0.0237, TITAN_V 0.966±0.0192, PRISM 0.936±0.0253, CHIEF 0.932±0.0245. EBRAINS TITAN 0.960±0.0026, TITAN_V 0.951±0.0044, PRISM 0.934±0.0085, CHIEF 0.923±0.0068.
- Extended Data Table 35, IDH mutation linear probe AUROC: TCGA TITAN 0.878±0.0675, TITAN_V 0.882±0.0678, PRISM 0.889±0.0425, CHIEF 0.882±0.0403. EBRAINS TITAN 0.960±0.0026, TITAN_V 0.951±0.0044, PRISM 0.934±0.0085, CHIEF 0.923±0.0068.

## 5. 우리 프로젝트와의 관련 (가설)
- ABMIL 기준선: 가설: 이 논문의 ABMIL 비교 프로토콜, 즉 CLAM scaffold 기반 weakly supervised MIL, batch size 1, 20 epochs, AdamW weight decay 10^-5, cosine annealing peak LR 10^-4는 우리 ABMIL 기준선 재현 또는 비교 시 참고할 수 있다. 단, TITAN의 ABMIL 우위 결과는 TITAN slide embeddings에 의존하므로 우리 fold context ridge에 직접 일반화되지 않을 수 있다 (출처: Online Methods, Supervised baselines; Results, Figure 2E).
- linear/ridge: 가설: TITAN linear probe는 l2-regularized logistic regression으로, 우리 closed-form ridge와 같은 l2 regularization family에 속할 수 있다. ridge strength validation tuning, frozen embedding 위 linear head, few-shot default l2=1 설정을 참고할 수 있다. 단, TITAN linear probe는 task-specific weights를 학습하므로 학습 파라미터 0개 접근과 동일하지 않다 (출처: Online Methods, Linear and K-nearest neighbors probe evaluation).
- in-context/few-shot: 가설: SimpleShot, 20-NN, few-shot linear probe 평가는 fold context in-context 분류기의 low-data 평가 프로토콜로 참고할 수 있다. 본문은 low-shot에서 TITAN/TITAN_V가 ABMIL보다 강하다고 주장하므로, in-context classifier가 ABMIL을 넘으려면 embedding/context representation 품질이 중요할 수 있다 (출처: Results, Figure 2F, Online Methods, Few-shot slide classification).
- spatial context/ALiBi: 가설: patch bag을 permutation-invariant로 처리하지 않고 2D grid와 ALiBi로 spatial context를 넣는 설계는 우리 WSI fold context에서 patch 중복 또는 브랜치 중복을 줄이는 방향과 관련될 수 있다. 단, TITAN은 self-attention ViT pretraining 모델이고, closed-form ridge/in-context head에 ALiBi를 그대로 옮길 수 있는지 본문에서 확인 못함 (출처: Results, Online Methods, Positional encoding).
- train short/test long: 가설: 8,192×8,192 ROI crop에서 학습하고 full WSI로 extrapolate하는 전략은 우리 fold context가 제한된 경우 context length extrapolation 설계에 참고할 수 있다. 단, TITAN의 extrapolation은 대규모 pretraining과 ALiBi에 기반하며, 우리 fold-level ridge의 승격 또는 분산 문제와 직접 연결되지 않는다 (출처: Results, Online Methods).
- 옮길 수 없는 것: TITAN 전체 pretraining, 즉 335,645 WSIs, iBOT/CoCa, A100 규모, CONCHv1.5 patch encoder는 우리 프로젝트의 fold context closed-form ridge/in-context classifier와 스케일과 목표가 다르다. 본문은 TITAN의 ABMIL 우위를 대규모 pretraining 효과로 설명하지만, 우리 브랜치 7개 유효 랭크 3.52/7 또는 확률적 적합 분산 2.8배 문제를 해결하는 직접적 방법은 본문에서 확인 못함 (출처: Online Methods, Results).

## 6. 이 논문이 답하지 않는 것
- 우리 브랜치 7개의 유효 랭크 3.52/7 중복을 측정하거나 줄이는 방법: 본문에서 확인 못함.
- 확률적 적합을 사용할 때 적합 분산이 승격 기준의 최대 2.8배로 커지는 원인과 완화 방법: 본문에서 확인 못함.
- closed-form ridge in-context classifier가 ABMIL을 넘는 조건, fold context 설계, branch/low-rank 구조, probabilistic fitting variance 분석: 본문에서 확인 못함.
- 학습 파라미터 0개 또는 최소 파라미터 in-context 분류기 설계: 본문에서 확인 못함. TITAN linear probe는 학습 파라미터가 있고, kNN/prototype만 parameter-free 평가이다.
- TITAN slide embedding의 effective rank, covariance, regularization path, Bayesian/probabilistic linear model, ridge strength와 분산 관계: 본문에서 확인 못함.
- 우리 benchmark의 fold context, 승격 기준, probabilistic fitting, branch redundancy에 대한 정량 비교: 본문에서 확인 못함.
