# 문헌 브리프 — benchmark_lottery (로컬 모델 요약)

```
url: https://arxiv.org/abs/2107.07002
fetched_from: https://arxiv.org/html/2107.07002
fetched_at: 2026-09-18 10:50:10 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8001` · 2026-09-18 10:53 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- 벤치마크는 알고리즘의 근본적 우위보다 과제 선택, 커뮤니티 관행, 벤치마크 상태 등 다양한 요인에 의해 우열이 지각될 수 있다 (출처: Abstract, Section 1).
- 벤치마크 과제/데이터셋 선택만으로도 알고리즘 간 상대 성능이 크게 달라질 수 있다 (출처: Abstract, Section 3).
- 과제 선택 편향(task selection bias)은 정립된 벤치마크의 과제 선택이 알고리즘 비교에 편향을 만든다 (출처: Section 3).
- 커뮤니티 편향은 공동체가 어떤 벤치마크를 평가 기준으로 수용하는지 결정하고, 이를 통해 벤치마크 로또를 강화한다 (출처: Section 4).
- 벤치마크는 stateful이며, 후속 연구가 이전 트릭·코드·초기화·테스트셋 접근에 의존하면 비교의 통계적 유효성이 훼손될 수 있다 (출처: Section 5).
- 표준 벤치마크가 없는 영역에서는 연구자가 벤치마크를 모델에 맞게 조정하는 rigging the lottery가 발생할 수 있다 (출처: Section 6).
- 단일 평균 점수만 보고 모델을 선택하는 것은 부적합할 수 있고, 집계/순위 집계·통계적 유의성 검정이 필요하다 (출처: Section 3.2, Section 7.2).
- 벤치마크 제작·사용·리뷰 가이드라인, 통계적 유의성 검정, living benchmark가 완화책으로 제안된다 (출처: Section 7).

## 2. 방법의 핵심
- 구조: ML 벤치마크 과정의 메타 분석으로, 벤치마크 라이프사이클, task selection bias, community bias, statefulness, rigging the lottery, 완화 제안을 다룬다 (출처: Section 1, Section 2, Section 3, Section 4, Section 5, Section 6, Section 7).
- 학습 방식: 본문에서 확인 못함. 이 논문은 새로운 모델 학습 방법을 제안하지 않는다.
- 추론 방식: 본문에서 확인 못함. 이 논문은 새로운 모델 추론 방법을 제안하지 않는다.
- 재현에 필요한 구체적 설정: 기존 리더보드/보조 데이터에서 점수를 재집계한다. SuperGLUE에서는 55개 모델, 8개 과제, Top-k k∈{1,3,5,10}, 과제 조합별 평균 집계와 순위 불일치를 분석한다 (출처: Section 3.1.1). VTAB에서는 32개 모델, 19개 과제, 3개 카테고리 조합별 Kendall rank correlation을 분석한다 (출처: Section 3.1.2). LRA에서는 11개 efficient transformer 모델, 6개 과제, 과제 조합별 Top-3를 분석한다 (출처: Section 3.1.3, Table 1). RL Unplugged에서는 7개 baseline, Atari 2600 46 games, DM Control 9 tasks, 개별 과제와 집계 점수 간 rank correlation을 분석한다 (출처: Section 3.1.4, Figure 3).

## 3. 평가 설정
- 데이터셋/벤치마크: SuperGLUE 8개 과제 (출처: Section 3.1.1), VTAB 19개 과제와 Natural/Specialized/Structured 3개 카테고리 (출처: Section 3.1.2), LRA 6개 과제 (출처: Section 3.1.3), RL Unplugged의 Atari 2600 46 games와 DM Control 9 tasks (출처: Section 3.1.4), GLUE 8개 datasets (출처: Section 4.1), ALE/Atari 2600 (출처: Section 6.2), recommender systems 공개 데이터셋 목록 (출처: Appendix F, Table 4).
- 비교 기준선: SuperGLUE top performing models 55개 (출처: Section 3.1.1), VTAB representation learning models 32개 (출처: Section 3.1.2), LRA efficient transformer models 11개 (출처: Section 3.1.3), RL Unplugged baselines 7개 (출처: Section 3.1.4). ABMIL 기준선: 본문에 없음.
- 지표: aggregated score, Top-k ranking inconsistency, Kendall rank correlation, mean/median human normalized performance, rank aggregation 방법, 통계적 유의성 검정 (출처: Section 3.1.1, Section 3.1.2, Section 3.1.4, Section 3.2, Section 7.2).

## 4. 정량 결과
- SuperGLUE: 8개 과제 중 4개를 고르는 70개 조합에서 Top-1은 6개 서로 다른 모델로 나뉘고, Top-3 또는 Top-5는 거의 60/70 조합에서 서로 일치하지 않는다 (출처: Section 3.1.1, Figure 1).
- SuperGLUE: SuperGLUE mean score와 개별 과제 점수 간 평균 Kendall rank correlation은 0.648이다 (출처: Appendix B, Figure 5).
- VTAB: structured subcategory는 full VTAB score와 약 0.7의 rank correlation을 보이고, 개별 과제와 full score 간 평균 Kendall correlation은 약 0.60이다 (출처: Section 3.1.2, Figure 2).
- VTAB: 19개 개별 과제에서는 12개 서로 다른 top-1 모델이 나타나고, size-2 과제 subset에서는 20개 서로 다른 winner가 나타난다 (20/171) (출처: Appendix D, Figure 6).
- LRA: 6개 과제와 11개 모델에서 과제 subset을 바꾸면 Top-3 모델 구성이 자주 바뀐다 (출처: Section 3.1.3, Table 1; Appendix E, Table 3).
- RL Unplugged: Atari 2600에서 개별 과제와 집계 점수 간 평균 rank correlation은 약 0.49이고, DM Control에서는 약 0.54이다 (출처: Section 3.1.4, Figure 3).
- 커뮤니티 편향 사례: 가장 많이 쓰이는 10개 데이터셋 중 label error가 식별되었고, ImageNet validation set에는 6%의 label error가 있다고 인용된다 (출처: Section 4).
- GLUE: GLUE는 8개 datasets로 구성되며, 8개 중 7개는 matching task이다 (출처: Section 4.1).

## 5. 우리 프로젝트와의 관련 (가설)
- 옮겨올 수 있는 것: fold/과제/브랜치별 점수를 단일 aggregate로만 보고 승격하지 말고, fold subset별 Top-k 불일치와 Kendall rank correlation을 함께 보고, 특정 fold context에서만 우위가 나타나는지 점검할 수 있다. 이는 이 논문에서 과제 subset 선택이 모델 순위를 바꾼다는 관찰과 연결되는 가설이다 (출처: Section 3, Section 3.1.1, Section 3.1.2).
- 옮겨올 수 있는 것: 확률적 적합으로 인한 분산 문제를 다루기 위해, multiple fixed splits, multiple random seeds, variance source 분석, statistical significance testing, rank aggregation을 평가 프로토콜에 도입할 수 있다. 이는 이 논문의 통계적 유의성 검정과 variance accounting 권고와 연결되는 가설이다 (출처: Section 3.2, Section 7.2).
- 옮겨올 수 있는 것: test fold를 주기적으로 교체하거나, fold 접근 횟수를 제한하는 living-benchmark식 관리가 adaptive overfitting을 줄이는 데 도움이 될 수 있다. 이는 이 논문의 stateful benchmark와 living benchmark 논의와 연결되는 가설이다 (출처: Section 5, Section 7.3).
- 옮길 수 없는 것: 이 논문은 병리 WSI, MIL, ABMIL, closed-form ridge in-context classifier, 브랜치 유효 랭크, 확률적 적합 분산에 대한 구체적 방법이나 수치를 제공하지 않는다. 따라서 브랜치 중복 감소나 적합 분산 감소 알고리즘을 직접 가져오기 어렵다 (본문에서 확인 못함).
- 옮길 수 없는 것: ABMIL 대비 우월성을 판단하는 데는 이 논문만으로는 부족하다. ABMIL은 본문에 등장하지 않는다 (출처: 본문 전체, ABMIL 없음).

## 6. 이 논문이 답하지 않는 것
- ABMIL을 넘는 구체적 방법 또는 성능 수치: 본문에서 확인 못함.
- 병리 WSI 벤치마크에서 fold 선택이 모델 순위를 바꾸는지: 본문에서 확인 못함.
- closed-form ridge in-context classifier의 적합 분산을 줄이는 방법: 본문에서 확인 못함.
- 브랜치 7개 중 유효 랭크 3.52/7 같은 브랜치 중복을 줄이는 방법: 본문에서 확인 못함.
- 확률적 적합 분산이 승격 기준의 최대 2.8배로 커지는 문제를 직접 해결하는 방법: 본문에서 확인 못함.
- 학습 파라미터 0개 제약 폐기 후 어떤 학습 방식이 적합한지: 본문에서 확인 못함.
- WSI-specific 데이터셋, 지표, fold 설계, MIL aggregation 방식: 본문에서 확인 못함.
