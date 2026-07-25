# BagPFN architecture v18

이 문서는 현재 구현인 `src/models/baseline.py`의 `BaseModel`을 기준으로 한다. README와 일부 테스트에 남아 있는 v16 표기는 문서 불일치이며, 실제 checkpoint 호환성 버전은 **18**이다.

## 목적과 입출력

BagPFN은 하나의 episode 안에서 여러 개의 bag을 보고, 일부 bag의 label을 context로 사용해 가려진 query bag의 이진 label을 예측하는 범용 multiple-instance episodic classifier다. 각 bag은 순서가 없는 instance 집합이며 특정 응용 도메인을 가정하지 않는다.

- 단일 episode 입력 `x`: `[bags, instances, 512]`
- label `y`: `[bags]`
- query index: `[queries]`
- batched 입력: `[episodes, bags, instances, 512]`
- 출력 logits: 단일 episode `[queries, 2]`, batched `[episodes, queries, 2]`

모델은 bag 순서와 instance 순서에 의존하지 않도록 set/subgroup 연산으로 구성된다. 코드의 `population`은 특정 응용 도메인의 실체를 가리키지 않고, episode 안에서 발견한 잠재 instance subgroup을 뜻하는 구현 용어다. query label은 forward에 사용되지 않으며, context label만 class memory를 만드는 데 사용한다.

## 전체 흐름

```text
instance embeddings
    │
    ├─ context instances로 episode 공통 subgroup anchors 생성
    │
    ├─ 각 bag을 mean / structured slots / novelty tails로 요약
    │
    ├─ labelled context token을 class별 memory로 압축
    │
    ├─ mean branch: ridge + bounded attention residual
    ├─ subgroup branch (`population`): query slots ↔ class memory
    ├─ tail branch: query instances의 rare evidence ↔ class memory
    │
    └─ 세 branch와 상호작용 항을 합쳐 binary logits 출력
```

## 1. Structured subgroup aggregator

`StructuredEpisodePopulationAggregator`가 모든 bag을 같은 episode 좌표계로 정렬한다. 클래스명의 `Population`은 잠재 subgroup을 뜻한다. anchor는 query를 제외한 context instance만으로 만든다.

현재 공통 설정:

| 항목 | 값 |
|---|---:|
| 입력 차원 | 512 |
| 전체 slot | 12 |
| density slot | 8 |
| rare slot | 4 |
| bag당 anchor 후보 instance | 32 |
| assignment temperature | 0.1 |
| density refinement | soft k-means 4회, temperature 0.15 |

Density anchor는 context 후보의 중심성 quantile로 초기화한 뒤 deterministic soft k-means로 갱신한다. 나머지 rare anchor는 density anchor에서 멀고 이미 선택된 rare anchor와도 다른 instance를 순차적으로 선택한다.

각 instance는 cosine similarity의 softmax로 12개 slot에 할당된다. 각 bag/slot에서 다음을 계산한다.

- abundance: soft assignment mass의 비율
- center: weighted instance mean
- spread: weighted per-feature standard deviation
- dispersion: anchor와 instance 간 cosine distance 평균
- within-slot rare state: 해당 slot에 속하면서 center에서 먼 상위 5% instance의 가중 평균

각 slot은 center, spread, rare의 세 토큰을 유지한다. 따라서 bag 하나는 `12 × 3 = 36`개의 structured token을 갖는다. 학습 가능한 encoder는 원래 통계량에 sigmoid-gated residual로 더해지며, 마지막 projection은 0으로 초기화된다.

별도로 다음 표현도 만든다.

- exact mean token 1개
- episode anchor에서 가장 먼 instance의 상위 1%, 5%, 15%를 요약한 novelty-tail token 3개

결과적으로 bag 하나의 class-memory 입력은 mean 1개 + slot 36개 + tail 3개, 총 **40개 token**이다.

## 2. Class memory

`StructuredPopulationMetaClassifier`는 labelled context bag의 40개 token을 class별로 모은다. 각 class마다 학습 가능한 memory seed가 전체 class token에 cross-attention하고, Transformer encoder가 memory token 사이의 관계를 갱신한다.

| 설정 | baseline | large ablation |
|---|---:|---:|
| hidden dimension | 256 | 512 |
| attention heads | 8 | 8 |
| memory Transformer layers | 1 | 3 |
| relation hidden dimension | 256 | 512 |
| ridge dimension | 64 | 128 |
| class memory tokens | 8 | 16 |
| 전체 parameter | 약 6.57M | 약 28.75M |

Large 설정은 aggregator, loss, 데이터 규칙은 유지하고 meta-classifier capacity만 확대한 ablation이다.

## 3. 세 evidence branch

### Mean branch

Bag의 exact mean을 사용한다. class-balanced ridge regression이 episode별 기본 decision rule을 만들고, set/cross-attention 결과는 학습 가능한 bounded residual로 더해진다. Ridge는 class label permutation에 대해 equivariant한 안정적인 기본 경로다.

### Subgroup branch (`population`)

Query의 36개 structured slot token 각각이 해당 class memory에 cross-attention한다. slot별 relation score를 만든 뒤 query-specific importance softmax로 가중합한다. routing temperature는 0.5다.

Subgroup residual scale은 학습 가능하지만 최소 0.10의 floor를 가지며 0으로 완전히 꺼질 수 없다. 초기값은 0.25다.

### Tail branch

요약 token이 아니라 query의 원 instance를 class memory와 직접 비교한다. 정규화된 similarity에서 class별 evidence를 만들고, instance evidence의 상위 1%, 5%, 10%, 20% 평균을 작은 head에 입력한다.

Tail residual scale 역시 최소 0.05의 floor를 가지며 초기값은 0.10이다. 민감한 tail encoder와 similarity 계산 일부는 FP32로 수행한다.

## 4. Evidence fusion

최종 logit은 다음 형태다.

```text
final = mean
      + population_scale × population
      + tail_scale × tail
      + fusion_scale × interaction
```

Interaction head는 세 branch logit 자체, 세 쌍의 곱, 세 쌍의 절댓값 차이 등 9개 feature를 입력받는다. fusion scale 초기값은 0.10이며 sigmoid로 제한된다.

이 구조에는 branch logit 자체의 절대 크기를 직접 정규화하는 제약은 없다. 따라서 residual scale이 bounded여도 branch 내부 logit norm이 커질 수 있다. 실제 large 실험에서 subgroup branch의 `population_logit_std`가 급증하면서 CE가 개선되지 않은 현상은 이 자유도와 관련된 진단 대상이다.

## 5. Outer episode batch

단일 B200에서는 한 optimizer step에 shape가 같은 8개 episode를 `forward_episode_batch`로 함께 계산한다. 이는 기존 8-GPU DDP에서 rank별 episode gradient 8개를 평균하던 의미를 단일 장치에서 재현한다.

- episode마다 anchor와 context/query split은 독립적이다.
- dense aggregation은 episode와 bag 축을 평탄화해 함께 계산한다.
- episode별 loss를 계산한 뒤 8개 평균으로 한 번 backward/optimizer step을 수행한다.
- 이 축은 한 episode 안의 bag 축과 구분된다.

## 6. 학습 objective와 안전장치

현재 medium objective는 다음과 같다.

```text
main_loss = cross_entropy + 0.10 × pairwise_ranking_loss
total_loss = main_loss + 0.01 × routing_balance_loss
```

- pairwise ranking: 한 episode의 positive query score가 negative보다 크도록 `softplus` margin을 적용한다.
- routing balance: episode 평균 subgroup-slot 사용량과 uniform distribution 사이의 KL이다.
- routing entropy는 기록하지만 가중치가 0이므로 loss에 더하지 않는다.
- mean/subgroup/tail logit std, residual scales, routing/rare entropy를 모니터링한다.
- optimizer step 전 gradient, step 후 parameter의 NaN/Inf를 검사한다.
- checkpoint의 `_architecture_version`이 18이 아니면 load를 거부한다.

## 관련 파일

- `src/models/baseline.py`: aggregator, meta-classifier, BaseModel
- `src/modules/model_interface.py`: episodic masking, loss, logging, finite 검사
- `configs/model/default.yaml`: baseline capacity
- `configs/model/large.yaml`: large-capacity ablation
- `configs/train_medium.yaml`, `configs/train_medium_large.yaml`: 조합 설정

