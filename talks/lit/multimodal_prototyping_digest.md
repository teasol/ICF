# 문헌 브리프 — multimodal_prototyping (로컬 모델 요약)

```
url: https://arxiv.org/abs/2407.00224
fetched_from: https://arxiv.org/html/2407.00224
fetched_at: 2026-09-18 12:45:09 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8000` · 2026-09-18 12:49 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- WSI는 patch token이 10^4개 이상이라 Transformer 융합 시 메모리 부담과 post-hoc 해석성 분석이 어렵다고 주장 (출처: Abstract, 1절).
- 형태 프로토타입으로 WSI를 300배 이상 압축하고, pathway 프로토타입으로 transcriptomics를 무감독으로 요약할 수 있다고 가정 (출처: Abstract, 1절).
- MMP는 6개 TCGA cancer cohort에서 거의 모든 uni- 및 multimodal baseline보다 성능을 내면서 연산량이 적다고 주장 (출처: Abstract, 1절, 5.1절, 5.5절).
- MMP는 small fixed token 수로 Transformer 또는 OT cross-alignment를 approximation 없이 사용할 수 있다고 주장 (출처: Abstract, 1절, 3.2절).
- GMM 기반 histology aggregation이 OT와 hard clustering보다 평균 C-Index에서 좋다고 보고 (출처: 5.3절, 표 3).
- Transformer cross-attention과 OT cross-alignment가 특정 조건에서 연결된다고 주장 (출처: 3.2.3절, Lemma 3.1).
- morphological prototype 수는 16까지 증가하면 성능이 좋아지고 이후 정체되어 C_h=16을 사용했다고 보고 (출처: 5.3절, 표 3, 부록 G).
- UNI patch encoder가 CTransPath와 ResNet50보다 평균 C-Index를 크게 높인다고 보고 (출처: 5.3절, 표 3).
- Cox loss가 NLL loss보다 평균 C-Index가 높다고 보고 (출처: 5.4절, 표 9).
- MMP는 histology-to-pathway와 pathway-to-histology 양방향 cross-attention 해석이 가능하다고 주장 (출처: 1절, 6절, 그림 2).
- C_h=16에서도 여러 프로토타입이 유사한 IDC 형태를 설명하는 prototype redundancy가 관찰된다고 보고 (출처: 부록 I, 부록 J).
- CRC에서는 unimodal histology baseline이 multimodal baseline보다 좋은 경우도 관찰된다고 보고 (출처: 5.1절).

## 2. 방법의 핵심
- 구조: WSI를 20x, 0.5 μm/pixel, 256×256 non-overlapping patches로 나누고 pretrained patch encoder UNI로 patch embedding을 생성한다 (출처: 3.1.1절, 4.2절, 부록 D).
- 구조: training patch embeddings 전체에 K-means를 적용해 C_h개 morphological prototypes를 만들고, GMM EM으로 slide summary를 만든다 (출처: 3.1.1절, 부록 A.3).
- 구조: GMM slide summary는 각 prototype마다 [π̂_c, μ̂_c, Σ̂_c]를 concat한 d_h=2D+1 차원 벡터이며, Σ는 diagonal covariance로 가정한다 (출처: 3.1.1절, 부록 A.3).
- 구조: transcriptomics는 50개 Hallmark pathway binary mask로 pathway summary를 만들고, pathway별 길이는 N_c,g<200으로 줄인다 (출처: 3.1.2절, 4.1절).
- 구조: histology summary는 linear projection으로, pathway summary는 pathway별 MLP 또는 SNN으로 공통 차원 d로 맞춘다 (출처: 3.2.1절).
- 구조: fusion은 C_g+C_h token에 대한 Transformer joint attention 또는 entropic-regularized OT cross-alignment + intra-modal Transformer self-attention으로 수행한다 (출처: 3.2.2절).
- 구조: post-attention embedding에는 prototype-specific FFN, layer normalization, modality 내 평균, concat, linear predictor를 붙여 patient-level risk를 만든다 (출처: 3.3절, 3.4절).
- 학습 방식: AdamW, learning rate 1×10^-4, cosine decay, weight decay 1×10^-5, 20 epochs로 학습한다 (출처: 4.3절).
- 학습 방식: MMP는 Cox loss, batch size 64를 사용하며, non-prototype baselines는 NLL survival loss, batch size 1을 사용한다 (출처: 4.3절).
- 학습 방식: MCAT, SurvPath, MOTCat, CMTA는 학습 시 WSI당 4,096 patches를 random sampling하고, 추론 시 whole WSI를 사용한다 (출처: 4.3절).
- 학습 방식: main MMP variant는 GMM histology aggregation, C_h=16, learnable random prototype encoding, prototype-specific FFN을 사용한다 (출처: 4.2절, 4.3절).
- 추론 방식: GMM EM은 feedforward network operation으로 수행할 수 있으며, 초기값은 π_c^(0)=1/C_h, μ_c^(0)=K-means centroid, Σ_c^(0)=I이다 (출처: 3.1.1절, 부록 A.3).
- 추론 방식: 본문은 EM iteration에서 보통 1 round가 충분했다고 보고한다 (출처: 부록 A.3).
- 재현 설정: patch encoder는 UNI, DINOv2-based ViT-Large, Mass General Brigham의 1×10^8 patches / 1×10^5 WSIs에서 pretrained된 모델을 사용한다 (출처: 4.2절).
- 재현 설정: ablation encoder로는 TCGA 3.2×10^4 WSIs에서 pretrained된 CTransPath와 ImageNet pretrained ResNet50을 사용한다 (출처: 4.2절).
- 재현 설정: pathway prototypes는 MSigDB Hallmark 50 gene sets이며, 4,241 unique genes, pathway size min 31 / max 199이다 (출처: 4.1절, 부록 D).
- 재현 설정: prototype encoding은 fixed one-hot d_e=C_g+C_h 또는 random-initialized learnable embedding d_e=32를 사용한다 (출처: 3.4절).

## 3. 평가 설정
- 데이터셋: TCGA 6 cancer types, BLCA n=359, BRCA n=868, LUAD n=412, STAD n=318, CRC n=296, KIRC n=340 (출처: 4.1절, 표 5).
- 데이터셋: disease-specific survival(DSS) 예측, 5-fold site-stratified cross-validation (출처: 4.1절).
- 데이터셋: bulk RNA-seq은 UCSC Xena의 log2-transformed TPM이며, 50 Hallmark pathways를 사용 (출처: 4.1절).
- 데이터셋: WSI 평균 patch 수는 BLCA 16,312, BRCA 11,565, LUAD 4,714, STAD 10,955, CRC 9,127, KIRC 12,802 (출처: 표 5).
- 비교 기준선: histology baseline으로 ABMIL, ABMIL-IB/IB-MIL, TransMIL, ILRA, AttnMISL, unimodal MMP를 사용 (출처: 4.2절, 표 1).
- 비교 기준선: transcriptomics baseline으로 2-layer MLP gene expression과 pathway-specific SNNs를 사용 (출처: 4.2절, 표 1).
- 비교 기준선: multimodal baseline으로 MCAT, SurvPath, MOTCat, CMTA를 사용 (출처: 4.2절, 표 1).
- ABMIL은 기준선으로 포함됨 (출처: 4.2절, 표 1).
- 지표: C-Index, log-rank p-value, GFLOPs (출처: 4.1절, 5.2절, 5.5절).

## 4. 정량 결과
- 표 1(5.1절) 평균 C-Index, 5-run std: MMP_Trans 0.665, MMP_OT 0.665, MOTCat 0.631, SurvPath 0.629, CMTA 0.623, MCAT 0.610, unimodal MMP 0.611, AttnMISL 0.605, ABMIL 0.599, ILRA 0.589, TransMIL 0.575, IB-MIL 0.569, gene exp 0.612, pathways 0.614, clinical 0.585.
- 표 1(5.1절) ABMIL per-dataset C-Index: BRCA 0.570±0.086, BLCA 0.550±0.039, LUAD 0.571±0.036, STAD 0.559±0.059, CRC 0.660±0.096, KIRC 0.684±0.115.
- 표 1(5.1절) MMP_Trans per-dataset C-Index: BRCA 0.738±0.069, BLCA 0.635±0.051, LUAD 0.642±0.037, STAD 0.598±0.051, CRC 0.630±0.125, KIRC 0.747±0.106.
- 표 1(5.1절) MMP_OT per-dataset C-Index: BRCA 0.753±0.069, BLCA 0.628±0.064, LUAD 0.643±0.013, STAD 0.580±0.071, CRC 0.636±0.120, KIRC 0.748±0.099.
- 5.1절: 저자는 MMP가 next-best multimodal과 unimodal 대비 각각 +5.4%, +7.8% avg라고 서술하고, 6개 disease 중 5개에서 top-2라고 서술한다.
- 표 2(5.2절) log-rank p-value: MMP_Trans는 BRCA 3.08×10^-5, BLCA 4.50×10^-2, LUAD 8.37×10^-5, STAD 1.40×10^-2, CRC 3.60×10^-2, KIRC 2.59×10^-8.
- 표 2(5.2절) log-rank p-value: MOTCat은 BRCA 6.16×10^-5, BLCA 8.60×10^-5, LUAD 9.65×10^-1, STAD 5.70×10^-2, CRC 7.04×10^-1, KIRC 2.40×10^-4.
- 5.2절: 0.05 기준에서 MMP는 6개 cancer type 모두 significant, MOTCat은 3개 cancer type significant라고 보고.
- 표 3(5.3절) ablation 평균 C-Index: full MMP 0.665; C_h=8 0.655(-1.5%), C_h=32 0.662(-0.5%); ResNet50 0.620(-6.8%), CTransPath 0.643(-3.3%); GMM→OT 0.658(-1.1%), GMM→HC 0.629(-5.4%); prototype embedding none 0.652(-2.0%), one-hot 0.660(-0.8%); prototype-specific FFN→shared 0.658(-1.1%); Transformer fusion→OT 0.665(-0.0%); early fusion→late fusion 0.646(-2.9%).
- 표 4(5.5절) average tokens per WSI: LUAD 4,714, KIRC 12,802.
- 표 4(5.5절) average GFLOPs: MCAT LUAD 2.10 / KIRC 5.49; SurvPath 2.00 / 5.41; CMTA 17.2 / 40.1; MMP total 0.334 / 0.864; MMP aggregation 0.309 / 0.839; MMP fusion 0.025 / 0.025.
- 5.5절: MMP는 cross-attention baselines보다 최소 5배 적은 giga-FLOPs를 달성한다고 보고.
- 표 9(5.4절, 부록 H) loss/batch ablation 평균 C-Index: Cox B=64 0.665, Cox B=32 0.660, Cox B=16 0.654, Cox B=128 0.655; NLL B=16 0.644, NLL B=1 0.621, NLL B=32 0.617, NLL B=64 0.608, NLL B=128 0.584.
- 표 7(부록 G) unimodal MMP C_h ablation 평균 C-Index: C_h=8 0.639, C_h=16 0.627, C_h=32 0.619.
- 표 8(부록 G) unimodal MMP encoder ablation 평균 C-Index: ResNet50 0.555, CTransPath 0.593, UNI 0.627.

## 5. 우리 프로젝트와의 관련 (가설)
- 가설: GMM/prototype summarization을 fold-context ridge classifier 앞에 두면 patch token을 고정된 C_h개 summary로 줄여, branch 출력이 공유하는 저차원 basis를 강제할 수 있을지 모른다. 근거는 MMP가 K-means prototypes와 GMM 파라미터로 slide를 고정 길이 summary로 변환한다는 점이지만, 이 논문은 effective rank나 fit variance를 측정하지 않아 가설만 가능하다 (출처: 3.1.1절, 부록 A.3).
- 가설: prototype identity encoding과 prototype-specific FFN을 우리 branch identity에 대응시켜 branch 중복을 줄이는 장치로 사용할 수 있을지 모른다. 근거는 prototype encoding/FFN이 MMP 성능에 기여했다는 ablation이지만, branch rank 문제와 직접 연결은 본문에 없다 (출처: 3.4절, 표 3).
- 가설: OT cross-alignment를 branch 간 정렬 또는 attention 대체로 시도할 수 있을지 모른다. 근거는 OT와 Transformer cross-attention의 연결이지만, rank/fit variance 효과는 본문에서 확인 못함 (출처: 3.2.2절, 3.2.3절).
- 가설: token 압축으로 batch-based training이 가능해지는 점은, 만약 survival loss를 쓴다면 참고할 수 있을지 모른다. 근거는 MMP가 Cox loss batch 64를 사용할 수 있었다는 점이지만, 우리 분류 벤치마크와 동일하지 않을 수 있다 (출처: 4.3절, 5.4절).
- 옮기기 어렵다: MMP의 ABMIL 우월 수치를 그대로 우리 ABMIL 기준선 초과 근거로 쓰지 말아야 한다. 이유는 TCGA survival, UNI encoder, transcriptomics fusion, Cox loss 설정이 우리 in-context classification fold context와 다르기 때문이다 (출처: 4.1절, 4.2절, 4.3절).
- 옮기기 어렵다: GMM/prototype가 branch 유효 랭크 중복을 줄인다고 단정할 수 없다. 이유는 본문에 effective rank, branch duplication, fit variance, promotion criterion 관련 측정이나 수치가 없기 때문이다.
- 옮기기 어렵다: zero learned parameter 정체성과 직접 비교하기 어렵다. 이유는 MMP가 learned projection, attention/OT, FFN, predictor를 학습하기 때문이다 (출처: 3.2절, 3.3절, 4.3절).
- 옮기기 어렵다: transcriptomics pathway prototype은 histology-only 문제에서 그대로 옮기기 어려울 수 있다. 이유는 pathway summary가 bulk RNA-seq과 Hallmark gene sets에 의존하기 때문이다 (출처: 3.1.2절, 4.1절).

## 6. 이 논문이 답하지 않는 것
- 이 논문은 fold-context in-context ridge classifier와 MMP/prototype를 결합하는 프로토콜을 제시하지 않는다.
- 브랜치 유효 랭크 중복을 측정하거나 줄이는 방법을 답하지 않는다. 관련 수치: 본문에서 확인 못함.
- 확률적 적합 시 fit variance가 승격 기준 대비 커지는 문제를 다루지 않는다. 관련 수치: 본문에서 확인 못함.
- ABMIL을 넘어서는 분류 벤치마크 성능을 답하지 않는다. 본문은 survival C-Index 중심이다 (출처: 4.1절, 표 1).
- learned parameters 허용 후 zero learned parameter 대비 이점을 답하지 않는다.
- C_h 선택이 우리 branch 수나 rank 구조에 어떻게 대응하는지 답하지 않는다.
- GMM EM 1 round가 우리 fold context에서 충분한지 답하지 않는다.
- prototype redundancy를 인정하지만, 이를 줄이는 구체적 해결책은 future work로만 제시한다 (출처: 부록 J).
- K-means/GMM에서 모든 patch가 단일 prototype에 할당될 수 있는 실패 가능성을 경고하지만, 우리 문제에서의 대응 방법은 제시하지 않는다 (출처: 부록 J).
