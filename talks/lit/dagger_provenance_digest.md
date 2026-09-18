# 문헌 브리프 — dagger_provenance (로컬 모델 요약)

```
url: https://arxiv.org/abs/2006.07484
fetched_from: https://arxiv.org/html/2006.07484
fetched_at: 2026-09-18 13:01:07 KST
```

- 요약: Qwen3.8-27B @ `127.0.0.1:8001` · 2026-09-18 13:07 KST
- **오케스트레이터는 원문을 읽지 않았다.** 이 브리프는 로컬 모델의 요약이며 검증되지 않았다.
- **외부 주장이지 우리의 관측이 아니다.** 어떤 수치도 우리 게이트·승격의 근거가 될 수 없다
  (`D-047`). 여기서 나오는 것은 가설뿐이며, 우리 결론이 되려면 `Primary 7`에서 직접 측정한다.

---

## 1. 저자들이 주장하는 것
- dagger는 다단계 ML 실험에서 모델 상태와 결과의 provenance를 추적하는 재현 가능·재사용 가능 실험 오케스트레이션 프레임워크 (출처: Abstract)
- dagger는 경량·모듈러·모델 중심 워크플로우 생성 솔루션으로, 과학적 기여와 실험 추적 보일러플레이트를 분리 (출처: Sec. 1)
- dagger는 상태 S와 상태 변경 recipe R를 트리로 표현하고, root·무순환·연결 조건으로 각 상태의 unique path/provenance를 보장 (출처: Sec. 3.1)
- dagger는 Dask 기반 lazy evaluation으로 single-threaded(기본), multi-threaded, multi-process, distributed 실행을 지원하며 DL 라이브러리 무관·크로스 플랫폼·하드웨어 독립 (출처: Sec. 3.2)
- dagger는 상태 해시 h_j=H(S_j,h_i)로 계산 그래프를 캐싱하여 중복 계산 방지 (출처: Sec. 3.2)
- dagger는 MIT License로 GitHub에서 제공되고 CircleCI로 Linux/MacOS CI 테스트 (출처: Sec. 1)

## 2. 방법의 핵심
- 구조: Experiment 객체가 실험 그래프를 lazy 정의하고 파일시스템 디렉토리에 state를 직렬화/캐싱; ExperimentState는 노드, Recipe는 상태 변경 액션, Function은 상태 변경 없는 액션 (출처: Sec. 3.2, Listing 1)
- 학습 방식: dagger 자체 학습 알고리즘은 본문에서 확인 못함; 사용자가 Recipe.run에서 train_model 등을 호출하도록 정의 (출처: Sec. 3.2, Listing 1)
- 추론/평가 방식: Function 클래스로 non-state-mutating evaluation을 수행; 예제 eval_fn은 eval_model(state.model, state.eval_data) 후 Accuracy 출력 (출처: Sec. 3.2, Listing 1)
- 재현 설정: State subclass에서 PROPERTIES, NONHASHED_ATTRIBUTES, initialize_state를 정의; Recipe subclass에서 PROPERTIES, run을 정의; Experiment.spawn_new_tree, tag, run으로 실험 트리 생성/실행; analysis API로 restore(slim=True), graph.draw(), filter, restore (출처: Sec. 3.3, Listing 1-3)
- 해시/캐싱: h_j=H(S_j,h_i) (출처: Sec. 3.2)
- 실행 환경: Dask 기반 lazy evaluation; default single-threaded, multi-threaded/multi-process/distributed 가능 (출처: Sec. 3.2)

## 3. 평가 설정
- 데이터셋: cifar-10 (출처: Listing 2); 공식 벤치마크 데이터셋은 본문에서 확인 못함
- 비교 기준선: ABMIL은 본문에서 확인 못함(없음); 다른 비교 기준선은 본문에서 확인 못함
- 지표: Accuracy (출처: Listing 1); 공식 평가 지표/벤치마크는 본문에서 확인 못함
- 예제 실험 설정: vgg-11, lr 0.01/0.1, nb_epochs 100, pruning_technique lowest_magnitude, pruning_fraction 0.2 (출처: Listing 2)

## 4. 정량 결과
- 성능 정량 결과는 본문에서 확인 못함 (표 없음)
- 예제 설정 수치만 있음: lr 0.01, 0.1; nb_epochs 100; pruning_fraction 0.2 (출처: Listing 2; 표 없음)

## 5. 우리 프로젝트와의 관련 (가설)
- dagger의 state tree/provenance와 recipe caching을 현재 프로젝트의 fold context in-context 분류 실험에 적용하면, 각 fold/branch 상태의 계보와 설정을 재현 가능하게 추적할 수 있을 수 있다는 가설 (출처: Sec. 3.1, Sec. 3.2)
- 해시 기반 캐싱 h_j=H(S_j,h_i)을 사용하면 동일 fold context/설정/seed에서 반복 실험의 중복 계산을 줄이고, 확률적 적합의 적합 분산 원인(설정, seed, 상태 계보)을 비교 분석하는 데 도움이 될 수 있다는 가설 (출처: Sec. 3.2)
- dagger는 학습 파라미터 유무와 무관하게 Recipe로 train/fit/eval 액션을 정의하므로, 학습 파라미터 제약 폐기 후에도 closed-form ridge fit을 Recipe로 모델링할 수 있을 수 있다는 가설 (출처: Sec. 3.2, Listing 1)
- 옮길 수 없는 것: dagger는 WSI/MIL/ABMIL/ridge/rank/variance에 대한 알고리즘, 벤치마크, 성능 수치를 제공하지 않으므로, 브랜치 간 중복 감소나 확률적 적합 분산 감소, 학습 기반 기준선 초과를 직접 보장하지는 않을 것으로 보임 (출처: Abstract, Sec. 3; 관련 용어 본문에서 확인 못함)

## 6. 이 논문이 답하지 않는 것
- 병리 WSI 벤치마크에서 학습 기반 MIL 기준선(ABMIL)을 넘는 방법/수치는 본문에서 확인 못함
- fold context에서 closed-form ridge로 푸는 in-context 분류기의 성능/분산 개선 효과는 본문에서 확인 못함
- 브랜치 간 유효 랭크 중복을 줄이는 방법/수치는 본문에서 확인 못함
- 확률적 적합의 적합 분산 문제를 해결하는 방법/수치는 본문에서 확인 못함
- dagger를 적용했을 때 ABMIL 대비 성능 차이는 본문에서 확인 못함
