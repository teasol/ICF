# Medium synthetic multiple-instance data and training tasks

이 문서는 `src/datasets/synthetic_data.py`, `src/modules/data_interface.py`, `src/modules/model_interface.py`와 `configs/data/medium.yaml`의 현재 규칙을 설명한다. 생성기는 특정 응용 분야에 한정되지 않고, 순서 없는 instance 집합으로 이루어진 bag에서 분포의 구성·위치·분산·조건부 상호작용 신호를 찾는 범용 multiple-instance 문제를 생성한다. 데이터는 CUDA에서 online 생성되며 외부 데이터가 필요 없다.

## Episode 단위

하나의 episode는 새로운 bag 집합과 새로운 관측 manifold를 정의한다.

| 항목 | medium 설정 |
|---|---:|
| bags per episode | 60–100 |
| instances per bag | 500–1000 |
| latent dimension | 32 |
| observed embedding dimension | 512 |
| manifold MLP | hidden 96, 3 layers |
| training episodes/epoch | 4096 |
| validation/test episodes | 각각 104 |
| outer episode batch | 8 |

한 episode 안에서는 모든 bag이 동일한 instance 수를 사용한다. `shape_group_size=8`이므로 한 optimizer step에 묶이는 8개 training episode도 같은 shape를 가져 dense batch 연산이 가능하다.

## Label과 effect score

Medium은 `balanced: false`이므로 bag label을 독립적인 Bernoulli 이진 표본으로 생성한다. 이는 context의 class별 개수로 query label을 추측하는 class-count completion shortcut을 제거한다.

각 bag에는 label 부호와 연결된 연속 effect score가 있다. 코드에서는 하위 호환성을 위해 `response_score`라는 이름을 사용한다.

```text
magnitude = 0.08 + 0.80 × |Normal(0, 1)|
effect_score (`response_score`) = (2 × label - 1) × magnitude
```

따라서 label은 effect score의 부호이고, 효과 세기는 bag마다 연속적으로 달라진다. 최종 bag 순서는 무작위 permutation한다.

## Latent mixture 생성

Medium episode는 항상 shared-component 및 continuous-effect 경로를 사용한다.

- shared components: episode당 4–10개
- 일반적인 shared fraction: 0.82–0.96
- 나머지는 label-associated component
- component mean separation: 0.5–1.4
- per-dimension latent scale: 0.6–1.3

Episode-level component prototype 외에 bag별 nuisance를 추가한다.

- 전체 bag shift scale: 0.35
- bag×component shift scale: 0.12
- label-associated/shared fraction logit noise: 0.65
- episode 공통 shared mixture logit variation: 0.70
- bag별 shared mixture logit variation: 0.70

이 nuisance들은 label과 직접 연결되지 않으며, 모델이 단순 bag identity나 고정 component 위치에 과적합하지 않게 한다.

## 다섯 synthetic distribution task

Episode마다 다음 중 하나를 선택한다.

| task | 확률 | label 신호 |
|---|---:|---|
| composition | 0.40 | label-associated component 비율 변화 |
| state | 0.30 | label-associated component latent mean 이동 |
| covariance | 0.05 | 한 latent 방향의 분산 변화 |
| interaction | 0.05 | shared subgroup 하나에서 state+covariance 변화 |
| combined | 0.20 | composition+state+covariance 동시 변화 |

### Composition

Shared fraction의 logit에서 `1.40 × effect_multiplier × effect_score`를 뺀다. 따라서 positive score는 label-associated component의 비율을 연속적으로 높인다.

### State

Label-associated component에 속한 instance에 episode별 무작위 단위 방향의 shift를 더한다. shift 크기는 effect score, `response_state_effect_scale` 0.45–1.00, effect multiplier의 곱이다.

### Covariance

Label-associated component에서 episode별 무작위 latent 방향 하나를 정하고, 그 방향의 centered projection을 확대/축소한다. scale 범위는 0.30–0.80이며 effect score에 따라 log-scale이 바뀐다. log-scale은 수치 안정성을 위해 [-1.5, 1.5]로 clamp한다.

### Interaction

별도 label-associated component 대신 4–10개 shared component 중 하나를 effect-bearing subgroup으로 지정한다. 그 subgroup에 state와 covariance 효과를 함께 적용한다. 즉 label 신호는 “특정 잠재 subgroup 안의 위치/분산 변화”로 나타난다.

### Combined

Label-associated component의 비율, latent 위치, covariance를 동시에 바꾼다.

## Rare-effect episode

Interaction을 제외한 episode의 15%는 rare-effect로 만든다. 이때 label-associated component 비율은 2–8%이고 shared fraction은 92–98%다. 일반 episode보다 신호가 소수 instance에만 존재하므로 tail/rare evidence 경로를 훈련한다.

## Effect strength와 curriculum

각 episode의 effect multiplier는 0.9–2.1에서 균등 표본한다. `difficulty_curriculum_episodes=0`이고 start/end 범위가 같으므로 첫 epoch부터 끝까지 stationary distribution이며 curriculum 변화는 없다.

## 512차원 관측 공간

Latent instance를 episode마다 새로 표본한 random MLP로 512차원에 매핑한다.

```text
32 → 96 → 96 → 512
```

중간 layer에는 GELU를 사용한다. 이후 표준편차 0.01의 observation noise를 더하고 각 instance embedding을 L2-normalize한다. Episode마다 MLP가 달라지므로 모델은 고정 좌표나 특정 feature index가 아니라 episode 내부의 상대적 분포 관계를 학습해야 한다.

Generator는 effect-bearing subgroup의 실제 membership으로 oracle proportion/mean/variance feature도 계산할 수 있지만, 이는 진단 전용이며 training dataset은 `(x, y)`만 반환한다.

## Training task 구성

Training episode마다 5–12개 query bag을 무작위로 선택한다. 단, 각 class에서 최소 한 bag은 반드시 labelled context로 보호한다. Outer batch의 첫 episode에서 query 수를 뽑고 나머지 7개 episode도 같은 query 수를 사용한다.

모델이 받는 정보:

- 모든 bag의 instance embeddings
- context bag의 labels
- query bag index

모델이 받지 않는 정보:

- query labels
- effect score/task
- true component membership
- oracle subgroup features (코드 식별자: `oracle_population_features`)

한 optimizer step은 8개 episode loss의 평균이다. 각 episode의 query 수가 같으며, 최종 CE와 보조 loss는 query 수를 고려해 집계한다.

## Validation과 test task

Validation/test는 각각 고정 seed 50042/60042를 사용하므로 index별 episode가 재현된다. 각 episode에서 전체 bag의 약 20%, 최대 20개를 deterministic query로 선택한다. 관측된 각 class의 첫 bag은 context로 보호하며, 여러 query의 label이 서로 context로 노출되지 않도록 한 번에 함께 mask한다.

## 학습 loss

현재 medium 설정의 total loss는 다음과 같다.

```text
CE
+ 0.10 × pairwise ranking loss
+ 0.01 × subgroup routing balance loss
+ 0.00 × routing entropy
```

- CE: query의 binary classification cross-entropy
- pairwise ranking: positive query의 class-1 logit margin이 negative보다 크도록 학습
- routing balance: 한 episode에서 subgroup slot 평균 사용량이 한 slot에 독점되지 않도록 uniform usage와의 KL을 최소화
- routing entropy: 관찰용 metric이며 현재 objective에는 포함되지 않음

Rare-fraction entropy, branch별 logit std, subgroup/tail/fusion residual scale도 진단 metric으로 기록하지만 별도의 loss 항은 아니다.

## Reproducibility와 CUDA 생성

- Training은 고정된 index dataset이 아니라 접근할 때마다 새 episode를 생성하는 stream이다.
- rank가 여러 개면 seed에 `rank × 1,000,003`을 더해 rank별 내용을 분리한다.
- epoch 시작 시 sample stream 위치를 복원해 resume 시 epoch-level 연속성을 유지한다.
- validation/test는 `seed + index`로 고정된다.
- CUDA generation과 background prefetch를 사용해 다음 outer batch 생성을 현재 optimizer step과 겹친다.

## 관련 파일

- `src/datasets/synthetic_data.py`: episode와 manifold 생성 규칙
- `src/modules/data_interface.py`: outer batching, prefetch, train/validation collate
- `src/modules/model_interface.py`: query sampling과 objective
- `configs/data/medium.yaml`: 현재 medium 분포
- `configs/train_medium.yaml`, `configs/train_medium_large.yaml`: 현재 학습 과제 가중치

