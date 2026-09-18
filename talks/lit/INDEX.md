# 문헌 색인 (자동 생성)

- 갱신: 2026-09-18 12:24 KST · 브리프 9건 · 요약 대기 2건
- 생성: `scripts/ops/lit_index.py` (코드 추출만, 모델이 쓰지 않음)
- **모든 브리프는 외부 주장이며 우리의 관측이 아니다**(`D-047`). 게이트·승격의 근거가
  될 수 없고, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.
- 원문은 `talks/lit/raw/`에 두고 git에 넣지 않는다. 기록된 URL로 다시 받을 수 있다.

| 브리프 | 제목 | 원문 | 수집 | 분량 | 이 프로젝트와의 관련 (브리프 §5 첫 줄) |
|---|---|---|---|---:|---|
| [`benchmark_lottery`](benchmark_lottery_digest.md) | The Benchmark Lottery | [원문](https://arxiv.org/abs/2107.07002) | 2026-09-18 | 20,343단어 | 옮겨올 수 있는 것: fold/과제/브랜치별 점수를 단일 aggregate로만 보고 승격하지 말고, fold subset별 Top-k 불일치와 Kendall rank correlation을 함께 보고, 특정 fold context에서만 우위가 나타나는지 점검할 수 있다. 이는 이 논문에 |
| [`cellpath_bench`](cellpath_bench_digest.md) | CellPath-Bench: A Multidimensional Benchmark for Whole-Slide Cellular Representations in Pathology Foundation Models | [원문](https://arxiv.org/abs/2608.21060) | 2026-09-18 | 144,922단어 | 우리가 옮겨올 수 있는 것(가설): registered nuclear coordinates에서 bilinear sampling하는 Nuc readout을 우리 WSI in-context 분류기에 적용하면, patch mean pooling보다 cell-level signal을 보존하여  |
| [`feature_extractor_bench`](feature_extractor_bench_digest.md) | Benchmarking Pathology Feature Extractors for Whole Slide Image Classification | [원문](https://arxiv.org/abs/2311.11772) | 2026-09-18 | 47,478단어 | 가설: pathology-specific SSL feature extractor, 특히 Lunit-DINO, UNI, CTransPath를 frozen feature로 사용하면, 우리 in-context 분류기의 입력 표현 품질이 개선되어 branch 간 중복이나 적합 불안정성이 완화될 |
| [`icmil`](icmil_digest.md) | In-Context Multiple Instance Learning | [원문](https://arxiv.org/abs/2606.06458) | 2026-09-18 | 12,514단어 | 가설: ICMIL의 fold context를 query bag label prediction에 쓰는 in-context formulation은 우리의 closed-form ridge in-context 분류기와 같은 문제 구조이므로, 병리 WSI low-label fold에서도 lear |
| [`multimodal_pfn`](multimodal_pfn_digest.md) | MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning | [원문](https://arxiv.org/abs/2602.20223) | 2026-09-18 | 10,130단어 | 가설: CAP의 learnable query pooling은 우리 fold context에서 branch output token/feature 수를 compact하게 조절하는 장치로 옮길 수 있을 수 있다. 논문은 token count 불균형이 attention mass를 왜곡한다고 보 |
| [`pfn_foundations`](pfn_foundations_digest.md) | Statistical Foundations of Prior-Data Fitted Networks | [원문](https://arxiv.org/abs/2305.11097) | 2026-09-18 | 14,606단어 | 옮겨올 수 있는 것 (가설): PFN의 고정 in-context 예측자 관점은 우리 closed-form ridge fold context 분류기를 bias/variance 분해로 진단하는 틀로 사용할 수 있을 수 있다. (출처: 5.1절) |
| [`pfn_imbalance`](pfn_imbalance_digest.md) | Correcting Class Imbalance in Prior-Data Fitted Networks for Tabular Classification | [원문](https://arxiv.org/abs/2605.21742) | 2026-09-18 | 4,364단어 | 가설: 우리 fold context in-context 분류기의 soft score가 이 논문의 TabPFN처럼 prior-driven bias만 보인다면, minority prior에 맞춘 thresholding이 rare-class 성능이나 WCA를 개선할 수 있다. 단, WSI/M |
| [`titan_slide_fm`](titan_slide_fm_digest.md) | Multimodal Whole Slide Foundation Model for Pathology | [원문](https://arxiv.org/abs/2411.19666) | 2026-09-18 | 50,308단어 | ABMIL 기준선: 가설: 이 논문의 ABMIL 비교 프로토콜, 즉 CLAM scaffold 기반 weakly supervised MIL, batch size 1, 20 epochs, AdamW weight decay 10^-5, cosine annealing peak LR 10^-4는 |
| [`variance_benchmarks`](variance_benchmarks_digest.md) | ACCOUNTING FOR VARIANCE IN M ACHINE L EARNING B ENCHMARKS | [원문](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf) | 2026-09-18 | 16,835단어 | 옮겨올 수 있는 것, 가설: |

## 요약 대기

- `ensembles_effective`
- `madeleine_slide_repr`
