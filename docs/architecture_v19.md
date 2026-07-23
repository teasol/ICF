# Architecture v19

Architecture v19는 architecture v18에서 확인된 global bag-shift 민감도를
제거한다. Bag마다 모든 instance에 공통으로 더해지는 translation은 분류에
사용하지 않고, bag 내부의 상대적 구조와 translation-invariant scale만 사용한다.

## Bag-centered input view

각 bag에서 raw mean을 뺀 centered delta를 만든다. Structured population과
query tail 경로에는 centered delta를 instance별 L2 정규화한 표현만 전달한다.
Raw input과 raw bag mean은 classification branch, class memory 또는 final
logit에 사용하지 않는다.

Global summary는 feature별 centered RMS spread다. 이 값은 기존 exact-mean
token과 exact-mean ridge branch를 대체한다.

## Structured representation

- global spread token 1개
- 12개 slot의 center, spread, within-slot rare token 36개
- novelty-tail token 3개
- bag당 총 40개 token

Context anchor 후보, density refinement, rare anchor, cosine assignment, slot
state, dispersion, within-slot rare state와 novelty tail은 모두 centered view에서
계산한다. Soft assignment mass에 기반한 abundance는 유지한다.

Bag마다 local population을 별도로 soft-clustering하고, context에서 만든
episode reference slot에 entropic Sinkhorn transport로 정렬한다. 정렬된 abundance와
dispersion만 아래 ridge 경로에 전달하며 population 순위 정렬은 사용하지 않는다.
따라서 bag-local shift 적응과 episode 내 population identity를 함께 유지한다.

Slot identity를 유지한 `slot_metadata`의 `(log abundance, dispersion)`은
flatten한 뒤 learned coordinate projection 없이 episode별 class-balanced ridge에
직접 입력한다. 따라서 동일한 slot permutation에는 정확히 equivariant하며, 기존
population memory attention은 bounded residual로만 더한다.

## Classification branches

Episode별 class-balanced ridge와 bounded attention residual은 global spread를
입력으로 사용하는 global-shape branch가 담당한다. Structured slot과 centered
query instance는 각각 population과 tail branch에 사용한다.

최종 logit은 다음 구조를 유지한다.

```text
global_shape
+ population_scale * population
+ tail_scale * tail
+ fusion_scale * interaction
```

## Compatibility

Architecture version은 19다. Version 18 또는 version metadata가 없는
checkpoint는 load하지 않는다. Raw-mean 동작은 새 checkpoint로 실행하는 명시적
diagnostic mode에서만 선택할 수 있으며 production 기본값은 다음과 같다.

```yaml
bag_centered_representation: true
global_summary: centered_spread
use_raw_mean_branch: false
```

성능과 수치 합격 기준은 `docs/v19_acceptance_protocol.md`에 고정한다.
