# 문헌 색인 (자동 생성)

- 갱신: 2026-09-18 10:52 KST · 브리프 4건 · 요약 대기 7건
- 생성: `scripts/ops/lit_index.py` (코드 추출만, 모델이 쓰지 않음)
- **모든 브리프는 외부 주장이며 우리의 관측이 아니다**(`D-047`). 게이트·승격의 근거가
  될 수 없고, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.
- 원문은 `talks/lit/raw/`에 두고 git에 넣지 않는다. 기록된 URL로 다시 받을 수 있다.

| 브리프 | 제목 | 원문 | 수집 | 분량 | 이 프로젝트와의 관련 (브리프 §5 첫 줄) |
|---|---|---|---|---:|---|
| [`icmil`](icmil_digest.md) | In-Context Multiple Instance Learning | [원문](https://arxiv.org/abs/2606.06458) | 2026-09-18 | 12,514단어 | 가설: ICMIL의 fold context를 query bag label prediction에 쓰는 in-context formulation은 우리의 closed-form ridge in-context 분류기와 같은 문제 구조이므로, 병리 WSI low-label fold에서도 lear |
| [`multimodal_pfn`](multimodal_pfn_digest.md) | MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning | [원문](https://arxiv.org/abs/2602.20223) | 2026-09-18 | 10,130단어 | 가설: CAP의 learnable query pooling은 우리 fold context에서 branch output token/feature 수를 compact하게 조절하는 장치로 옮길 수 있을 수 있다. 논문은 token count 불균형이 attention mass를 왜곡한다고 보 |
| [`pfn_foundations`](pfn_foundations_digest.md) | Statistical Foundations of Prior-Data Fitted Networks | [원문](https://arxiv.org/abs/2305.11097) | 2026-09-18 | 14,606단어 | 옮겨올 수 있는 것 (가설): PFN의 고정 in-context 예측자 관점은 우리 closed-form ridge fold context 분류기를 bias/variance 분해로 진단하는 틀로 사용할 수 있을 수 있다. (출처: 5.1절) |
| [`pfn_imbalance`](pfn_imbalance_digest.md) | Correcting Class Imbalance in Prior-Data Fitted Networks for Tabular Classification | [원문](https://arxiv.org/abs/2605.21742) | 2026-09-18 | 4,364단어 | 가설: 우리 fold context in-context 분류기의 soft score가 이 논문의 TabPFN처럼 prior-driven bias만 보인다면, minority prior에 맞춘 thresholding이 rare-class 성능이나 WCA를 개선할 수 있다. 단, WSI/M |

## 요약 대기

- `benchmark_lottery`
- `cellpath_bench`
- `ensembles_effective`
- `feature_extractor_bench`
- `madeleine_slide_repr`
- `titan_slide_fm`
- `variance_benchmarks`
