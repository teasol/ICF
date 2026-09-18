# 문헌 브리프 — multimodal_pfn (로컬 모델 요약)

```
url: https://arxiv.org/abs/2602.20223
fetched_from: https://arxiv.org/html/2602.20223
fetched_at: 2026-09-18 10:32:01 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8003` · 2026-09-18 10:37 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- MMPFN은 TabPFN을 tabular, image, text를 통합 처리하도록 확장하는 프레임워크라고 주장한다. (출처: Abstract, 1 Introduction)
- modality projector는 non-tabular 임베딩을 tabular-compatible token으로 변환하는 핵심 다리라고 주장한다. (출처: Abstract, 3.2)
- MGM과 CAP은 overcompressed non-tabular embeddings와 token-count-induced attention imbalance를 완화한다고 주장한다. (출처: Abstract, 1 Introduction, 3.2, 3.4)
- medical과 general-purpose multimodal dataset에서 MMPFN은 경쟁 SOTA보다 일관되게 우수하다고 주장한다. (출처: Abstract, 1 Introduction, 4.2)
- modalities가 추가될수록 MMPFN 성능이 양의 방향으로 스케일링된다고 주장한다. (출처: 1 Introduction, 4.3, Figure 5)
- low-data regime에서 MMPFN은 robust하다고 주장한다. (출처: 1 Introduction, 4.3, Table 7)
- TabPFN backbone은 projected non-tabular features만으로도 native tabular inputs에 국한되지 않는다고 주장한다. (출처: 4.3, Figure 2(a))
- CAP는 FiLM보다 token count를 제어하여 tabular-non-tabular fusion에서 우월하다고 주장한다. (출처: 4.3, Table 6)
- MGM은 single-head/multi-head projector 변형 중 최고 정확도를 낸다고 주장한다. (출처: 4.3, Table 5)
- MMD/JMMD distribution alignment은 추가 이득 없이 MGM baseline보다 낮다고 주장한다. (출처: Appendix S1, Figure S3)

## 2. 방법의 핵심
- 구조: per-modality encoders, modality projector, TabPFN backbone, lightweight decoder head로 구성된다. (출처: 3.2)
- tabular encoder는 TabPFN v2 encoder와 동일하고 fine-tuning 중 frozen이다. (출처: 3.2)
- image encoder는 DINOv2 ViT-B/14를 사용하고, 입력 이미지의 높이/폭을 14로 나누어 떨어지도록 resize한 뒤 최종 [CLS] token을 global representation으로 사용한다. (출처: 3.2)
- text encoder는 ELECTRA-based encoder를 사용하고, 최대 512 token으로 truncate한 뒤 [CLS] embedding을 사용한다. (출처: 3.2)
- text preprocessing은 TTT pipeline을 따르고, 여러 text attribute는 각각 embedding을 추출하며, ELECTRA가 Chinese를 지원하지 않아 Chinese 문자는 empty string으로 대체한다. (출처: Appendix S1)
- MGM은 [CLS] embedding을 N개의 MLP head로 d-dim token N개로 확장하고, GLU가 각 head 기여를 조절한다. (출처: 3.2)
- CAP는 N개 MGM token을 key/value로 사용하고, K개 learnable query vector가 cross-attention하여 K개 d-dim token을 만든 뒤 MLP로 refine한다. (출처: 3.2)
- CAP 출력 token은 tabular token과 feature dimension 방향으로 concatenate되어 TabPFN 입력 테이블을 만든다. (출처: 3.2)
- MGM head 수는 8~128 범위에서 실험되고, patch-token 비교에서는 128 heads가 사용된다. (출처: Appendix S1, Figure S1, Appendix S1)
- 구체적 token 설정: Figure 2(b)의 MGM+CAP는 24개 CAP token을 사용하고, Table 4에서는 PU20의 K=24, Cloth의 K=4를 사용한다. (출처: Figure 2(b), Table 4)
- 학습: modality encoders는 freeze하고, modality projector, TabPFN backbone, decoder를 학습한다. (출처: 3.3)
- 학습은 TabPFN in-context protocol을 따라 train/test embeddings를 하나의 테이블로 concatenate하고, test sample 예측으로 supervisory signal을 얻는다. (출처: 3.3)
- 재현 설정: cross-entropy loss, 100 iterations/steps, AdamW learning rate 1e-5, batch size 1, random seeds {0,1,2,3,4}, average accuracy 보고. (출처: 4.1, Appendix S1)
- 재현 설정: official TabPFN repository를 기반으로 MMPFNClassifier/MMPFNRegressor를 확장하고, train/validation split 후 validation loss로 업데이트하며, ScheduleFree를 사용한다. (출처: Appendix S1)
- 추론: TabPFN은 concatenated train/test rows를 mask로 처리하여 test rows가 training rows에만 cross-attend하게 하고, MLP head가 test embeddings를 예측으로 변환한다. (출처: 3.1)
- 추론: MMPFN은 TabPFN backbone이 multimodal embeddings를 jointly 처리하고 lightweight decoder head가 test sample 예측을 생성한다. (출처: 3.2)
- 구체적 분석 설정: cosine similarity 분석에서는 tabular group size를 1로 설정하여 feature-wise embedding을 비교한다. (출처: Appendix S1)

## 3. 평가 설정
- 데이터셋: PAD-UFES-20(PU20)은 2,298 samples, six skin lesion types, clinical image, up to 26 metadata features. (출처: 4.1, Table 1)
- 데이터셋: CBIS-DDSM(Mass/Calc)은 curated DDSM subset, digitized mammograms, calcification/mass ROI, biopsy-verified labels, lesion-level metadata; predefined train-test split. (출처: 4.1, Table 1)
- 데이터셋: Airbnb은 Melbourne homestay activity Dec 2018 snapshot이고, target을 10개 quantile-based groups로 discretize한다. (출처: 4.1, Table 1)
- 데이터셋: Salary은 India job postings 15,841 train/3,961 test이고, salary range 예측. (출처: 4.1, Table 1)
- 데이터셋: Cloth은 23,486 customer reviews, text fields, tabular features. (출처: 4.1, Table 1)
- 데이터셋: PetFinder은 14,000개 이상 pet profiles, image/text/structured attributes, five-class adoption-speed outcome. (출처: 4.1, Table 1)
- Table 1 stats: PU20 train 1838/test 460/features 21/numeric 3/categorical 18/images 1/classes 6; CBIS-DDSM(Mass) train 1318/test 378/features 8/numeric 3/categorical 5/images 3/classes 2; CBIS-DDSM(Calc) train 1545/test 326/features 8/numeric 3/categorical 5/images 3/classes 2; Airbnb train 18316/test 4579/features 50/numeric 27/categorical 23/text 1/classes 10; Salary train 15841/test 3961/features 4/numeric 1/categorical 3/text 3/classes 6; Cloth train 18788/test 4698/features 5/numeric 2/categorical 3/text 3/classes 5; PetFinder-I/T/A train 11721/test 2931/features 19/numeric 5/categorical 14/classes 5. (출처: Table 1)
- 비교 기준선(tabular-image): TabPFN, CatBoost, AutoGluon, MMCL, TIP, HEALNet, TIME, MMPFN. (출처: Table 2)
- 비교 기준선(tabular-text): TabPFN, CatBoost, AutoGluon, AllTextBERT, TFN, MulT, TTT, TabSTAR, MMPFN. (출처: Table 3)
- 비교 기준선(ablation): Linear, MLP, multi-head MLP, MoE, MGM; FiLM, CAP. (출처: Table 5, Table 6)
- 비교 기준선(low-data): TIP, MMPFN. (출처: Table 7)
- 비교 기준선(scaling): AutoGluon, MMPFN. (출처: Figure 5)
- 지표: accuracy(rank), 5 random seeds average, lower rank better. (출처: Table 2, Table 3)
- 지표: low-data에서는 accuracy와 full-data 대비 percentage change. (출처: Table 7)
- ABMIL: 본문에서 확인 못함 (Table 2와 Table 3의 기준선 목록에 없음). (출처: Table 2, Table 3)

## 4. 정량 결과
- Table 2 (tabular-image, accuracy(rank), 5 seeds): TabPFN PU20 82.17(2), Mass 71.27(5), Calc 73.31(2), Petfinder 36.33(8), Avg. 4.25; CatBoost 80.43(4), 78.31(1), 72.09(4), 38.69(4), Avg. 3.25; AutoGluon 81.09(3), 76.28(2), 71.04(6), 38.81(3), Avg. 3.50; MMCL 76.61(7), 57.62(7), 60.12(8), 36.61(7), Avg. 7.25; TIP 78.75(6), 73.12(4), 67.96(7), 37.28(5), Avg. 5.50; HEALNet 74.65(8), 68.10(6), 71.83(5), 37.03(6), Avg. 6.25; TIME 80.35(5), -, 72.70(3), 39.25(2), Avg. 3.33; MMPFN 85.22(1), 74.53(3), 75.40(1), 40.74(1), Avg. 1.50. (출처: Table 2)
- Table 3 (tabular-text, accuracy(rank), 5 seeds): TabPFN Airbnb 46.96(2), Salary 44.96(6), Cloth 55.07(9), Petfinder 36.33(7), Avg. 6.00; CatBoost 43.56(4), 40.36(9), 59.24(8), 35.47(8), Avg. 7.25; AutoGluon 44.60(3), 45.24(5), 72.07(1), 37.96(4), Avg. 3.25; AllTextBERT 30.9(9), 44.0(7), 68.0(3), 34.6(9), Avg. 7.00; TFN 35.7(8), 45.8(3), 60.1(7), 36.8(6), Avg. 6.00; MulT 36.3(7), 45.4(4), 63.6(6), 37.6(5), Avg. 5.50; TTT 38.3(6), 47.2(1), 65.5(5), 38.9(3), Avg. 3.75; TabSTAR 40.06(5), 43.75(8), 71.75(2), 41.53(1), Avg. 4.00; MMPFN 47.78(1), 46.17(2), 66.26(4), 39.04(2), Avg. 2.25. (출처: Table 3)
- Figure 2(a): image-only MMPFN with MGM은 DINOv2+MLP baseline 대비 69.30% vs 69.89%로 약 1% 이내이고, image token 수가 증가하면 accuracy가 증가하며, CAP는 modest gain을 제공한다. (출처: Figure 2(a), 4.3)
- Figure 2(b): multimodal 설정에서 두 modality의 token count가 유사할 때 성능이 가장 좋고, MGM+CAP는 24개 CAP token을 사용하며, MGM head 수가 증가하면 MGM+CAP가 꾸준히 개선되고 MGM alone보다 우수하다. (출처: Figure 2(b), 4.3)
- Figure 3: non-tabular token count가 증가하면 attention mass가 non-tabular modality로 monotonically 이동하고, tabular token attention mass는 감소한다; 12개 self-attention layer 평균이며, PU20과 Salary는 각각 11과 4개 tabular tokens를 사용한다. (출처: Figure 3)
- Appendix S1 Figure S2: Airbnb, Mass, Calc의 tabular token 수는 각각 25, 4, 4이다. (출처: Appendix S1, Figure S2)
- Table 4 (PU20): MGM 32=84.17, 4×32=82.43, 128=83.91; MGM+CAP K=24는 84.10, 83.57, 84.59. (출처: Table 4)
- Table 4 (Cloth): MGM 32=63.12, 4×32=61.76, 128=61.86; MGM+CAP K=4는 64.20, 64.22, 65.03. (출처: Table 4)
- Table 5 (MGM ablation, accuracy): Linear Avg. 53.86; MLP Avg. 54.14; multi-head MLP Avg. 55.81; MoE Avg. 53.23; MGM Avg. 57.37. (출처: Table 5)
- Table 6 (CAP ablation, accuracy): FiLM Avg. 55.48; CAP Avg. 57.37. (출처: Table 6)
- Table 7 (10% low-data): TIP PU20 70.44 (-10.6), Mass 68.31 (-6.58), Calc 62.27 (-8.37), PetFinder 34.86 (-6.49), Avg. 58.97 (-8.00); MMPFN PU20 72.87 (-14.14), Mass 76.13 (+0.75), Calc 72.09 (-5.23), PetFinder 35.73 (-12.30), Avg. 64.21 (-10.27). (출처: Table 7)
- Figure 5: PetFinder에서 tabular → tabular+text → tabular+image → tabular+image+text로 accuracy가 39% → 40% → 41%로 증가하고, MMPFN은 모든 modality combination에서 AutoGluon보다 우수하다. (출처: Figure 5)
- Appendix S1 Table S1: CBIS-DDSM(Mass)에서 GELU accuracy 72.09, orthogonality 0.0565; GLU accuracy 75.10, orthogonality 0.0913. Salary에서 GELU accuracy 45.04, orthogonality 0.04876; GLU accuracy 45.87, orthogonality 0.05831. (출처: Appendix S1, Table S1)
- Appendix S1 Table S2: ResNet50 PU20 83.26, Calc 73.94; DINOv2 PU20 85.22, Mass 74.53, Calc 75.40, Petfinder 40.74; DINOv3 PU20 85.61, Mass 75.48, Calc 76.75, Petfinder 40.57. (출처: Appendix S1, Table S2)
- Appendix S1 Table S3: Electra Airbnb 47.78, Salary 46.17; DeBERTa Airbnb 47.82, Salary 45.69. (출처: Appendix S1, Table S3)
- Appendix S1 patch-token 비교: PAD-UFES-20 이미지를 336=14×24로 resize하면 DINOv2 ViT-B/14가 이미지당 576 token embeddings를 생성한다; [CLS] embedding storage는 7.1 MB이고, all patch-token outputs storage는 4.1 GB이다. (출처: Appendix S1)
- Appendix S1 patch-token 성능: PU20에서 [CLS]-based MGM heads를 patch-token features로 대체하면 accuracy가 85.22%에서 84.02%로 0.85 percentage points 감소한다. (출처: Appendix S1)

## 5. 우리 프로젝트와의 관련 (가설)
- 가설: CAP의 learnable query pooling은 우리 fold context에서 branch output token/feature 수를 compact하게 조절하는 장치로 옮길 수 있을 수 있다. 논문은 token count 불균형이 attention mass를 왜곡한다고 보이기 때문. (출처: 3.2, 3.4, Figure 3)
- 가설: MGM의 multi-head + GLU는 단일 [CLS] embedding의 overcompression을 완화하고 출력 orthogonality를 높였으므로, 우리 브랜치 간 중복을 줄이는 후보 기법이 될 수 있다. effective rank 개선 효과는 본문에서 확인 못함. (출처: 3.2, Appendix S1, Table S1)
- 가설: TabPFN in-context table + light fine-tuning(100 steps, lr 1e-5)은 학습 파라미터를 쓰지 않던 제약 폐기 후, fold context 분류기에 작은 학습 파라미터를 추가하는 방식의 참고가 될 수 있다. closed-form ridge나 확률적 적합 분산은 본문에서 확인 못함. (출처: 3.1, 3.3, 4.1)
- 가설: low-data robustness 결과는 작은 fold context에서 pretrained prior가 도움이 될 수 있다는 가설을 제공한다. (출처: 4.3, Table 7)
- 옮기기 어려운 것: 논문은 tabular row-level multimodal classification이고, WSI patch/bag MIL 구조, 학습 기반 MIL 기준선, effective rank, stochastic fit variance, promotion criterion은 본문에서 확인 못함. (출처: Table 2, Table 3, Appendix S1)

## 6. 이 논문이 답하지 않는 것
- ABMIL 또는 학습 기반 MIL 기준선과의 비교: 본문에서 확인 못함. (출처: Table 2, Table 3)
- WSI patch/bag MIL 설정: 본문에서 확인 못함.
- 브랜치 유효 랭크 또는 중복도 측정: 본문에서 확인 못함.
- 확률적 적합의 분산 또는 promotion criterion: 본문에서 확인 못함.
- closed-form ridge in-context classifier: 본문에서 확인 못함.
- MMPFN의 token count balancing이 effective rank를 높이는지: 본문에서 확인 못함.
- WSI patch token 수에 따른 K/N 선택: 본문에서 확인 못함. (출처: 3.4, Figure 2, Appendix S1, Figure S1)
- calibration/uncertainty: 본문에서 확인 못함.
- full computational cost/latency numbers: 본문에서 확인 못함. (출처: Appendix S1)
