# Current architecture

Last updated: 2026-07-26  
Code baseline: architecture v19, commit `e2181c0`

이 문서는 현재 production config가 실제로 사용하는 모델 구조를 설명한다. 개발 상태와 다음 작업은 [`current_status.md`](current_status.md), 실험 protocol은 [`current_experiments.md`](current_experiments.md)를 참고한다.

## 1. 문제 정의와 입출력

한 episode는 labelled context bags와 query bags로 구성된다. Bag은 variable-length instance set이며 각 instance의 입력 차원은 512다. 모델은 context label만 이용해 episode-local class representation을 만들고 query bag의 binary class logits를 출력한다.

```text
input
  outer episodes E
  context instances [E, context bags, instances, 512]
  context labels    [E, context bags]
  query instances   [E, query bags, instances, 512]

output
  query logits      [E, query bags, 2]
```

Query label은 forward, normalization, class memory, ridge 및 covariance subspace fitting에 사용하지 않는다.

## 2. Bag-centered representation

각 bag에 공통으로 더해지는 translation을 제거한다.

```python
bag_mean = x.mean(dim=-2, keepdim=True)
centered_delta = x - bag_mean
centered_x = F.normalize(centered_delta, dim=-1, eps=1e-6)
global_spread = torch.sqrt(centered_delta.square().mean(dim=-2) + 1e-6)
```

Production 설정:

```yaml
bag_centered_representation: true
global_summary: centered_spread
use_raw_mean_branch: false
```

- `centered_x`: anchor, slot assignment, structured token과 tail evidence에 사용
- `centered_delta`: scale/covariance 구조 계산에 사용
- `global_spread`: raw mean을 대체하는 translation-invariant global summary
- raw bag mean과 raw coordinate: diagnostic 이외의 classification 경로에서 사용하지 않음

## 3. Structured population aggregator

현재 주요 설정:

```text
input dimension: 512
population slots: 12
initial density slots: 8
context samples per bag: 32
assignment temperature: 0.1
density refinement: 4 steps, temperature 0.15
within-slot rare fraction: 0.05
tail fractions: 0.01, 0.05, 0.15
```

Aggregator는 context에서 episode anchor를 만들고 각 bag의 local population을 soft assignment한다. Bag-local slots는 context reference slots에 entropic Sinkhorn transport로 정렬된다. 이 과정은 population identity를 episode 안에서 유지하면서 bag permutation과 instance permutation에 불변이어야 한다.

Bag당 token 구조:

```text
global spread token                         1
12 slots × (center, spread, rare state)    36
novelty-tail tokens                         3
                                           --
total                                      40
```

Slot metadata에는 identity-aligned log abundance와 dispersion이 들어간다. Abundance는 soft assignment mass에서 계산한다.

## 4. Covariance representations

현재 두 covariance 경로가 함께 존재한다.

### 4.1 Covariance sketch ridge

- centered representation을 64차원 covariance sketch로 요약
- production mode: correlation
- shrinkage: 0.1
- context-only class-balanced ridge로 query covariance logit 계산
- learned bounded residual scale로 final logit에 추가

### 4.2 Whitened CSP rank-1 relation

CSP 경로를 위해 `centered_delta`를 고정 projection으로 최대 32차원에 투영하고 bag별 covariance matrix를 만든다.

```text
projected instances: [instances, 32]
covariance matrix:   [32, 32]
```

각 outer episode에서 labelled context만 사용한다.

1. Class별 context covariance mean `C0`, `C1` 계산
2. `delta = C1 - C0`
3. Context pooled covariance에 shrinkage 0.1 적용
4. FP32 eigendecomposition으로 pooled covariance whitening
5. Whitened `delta`에서 절대 eigenvalue가 가장 큰 rank-1 direction 선택
6. 같은 filter로 context/query covariance를 scalar feature로 투영
7. Context 통계만 사용해 feature 정규화
8. Class prototype까지의 squared distance를 class dispersion으로 표준화
9. `score(class 1) - score(class 0)`을 relation logit으로 사용

CSP와 eigendecomposition은 BF16 autocast 밖의 FP32에서 수행한다. 이 경로는 episode-local non-parametric evidence이며 trainable parameter를 추가하지 않는다.

Production config:

```yaml
covariance_relation:
  enabled: true
  mode: standardized_distance
  granularity: subspace
  subspace_rank: 1
  subspace_whiten: true
  subspace_shrinkage: 0.1
  diagnostic_only: false
  residual_scale: 0.50
  eps: 1.0e-6
```

## 5. Meta-classification branches

### Global-shape branch

`global_spread`를 입력으로 사용하는 episode별 class-balanced ridge와 attention residual이다. Raw mean은 사용하지 않는다.

### Population branch

Identity-aligned slot abundance/dispersion을 flatten해 class-balanced ridge에 넣는다. Learned population class-memory attention은 bounded residual로 결합한다.

### Tail branch

Context structured tokens에서 class당 8개의 class-memory token을 만든다. Centered query instance와 class memory similarity를 계산하고 top fractions `0.01, 0.05, 0.10, 0.20`의 evidence를 pooling한다.

### Interaction branch

Global-shape, population, tail evidence의 원값, pairwise product와 absolute difference를 작은 learned scorer에 입력한다. Sigmoid-bounded fusion scale로 더한다.

### Covariance branches

기존 covariance-sketch ridge와 CSP relation은 위 네 경로 뒤에 별도 residual로 추가된다.

## 6. Final logit

현재 production forward의 개념적 계산:

```text
base = global_shape
     + population_scale * population
     + tail_scale * tail
     + fusion_scale * interaction

final = base
      + covariance_residual_scale * covariance_ridge
      + 0.50 * covariance_relation
```

Population, tail, fusion, covariance ridge scale은 learned/bounded parameter다. `0.50` CSP scale은 config의 고정 hyperparameter다. `diagnostic_only=true`이면 CSP metric은 계산하지만 final logit에는 추가하지 않는다.

## 7. Model capacity와 주요 차원

```text
trainable parameters: 약 6.6M
token dimension: 512
meta hidden dimension: 256
attention heads: 8
set layers: 1
ridge projection dimension: 64
class memory tokens: 8 per class
classes: 2
outer episode batch: production에서 8
```

여러 episode는 모델 내부 outer-batch 축에서 함께 계산한다. 각 episode의 context-only ridge/CSP fitting은 서로 독립적이다.

## 8. Training objective

현재 production 학습은 query CE를 사용한다.

```yaml
ranking_loss_weight: 0.0
routing_sparsity_weight: 0.0
routing_balance_weight: 0.0
```

Diagnostic metric과 oracle metadata는 모델 입력 또는 loss가 아니다.

## 9. Invariance 및 compatibility contract

반드시 유지해야 하는 조건:

- bag별 common translation에 대한 centered representation/final-logit 불변성
- instance permutation과 bag permutation 불변성
- context label permutation에 대한 class-logit equivariance
- query label leakage 없음
- 40-token shape와 outer-batch shape 유지
- dense와 variable-length bag 경로 일치
- logits, loss, gradient와 parameter finite
- architecture version은 19
- v18 또는 version metadata 없는 checkpoint load 거부

Relation config를 전달하지 않은 모델 생성자의 기본 relation은 비활성이다. 이는 기존 v19 state-dict compatibility를 불필요하게 깨지 않기 위한 것이며 production config는 relation을 명시적으로 활성화한다.

## 10. Source of truth

- Aggregator와 meta-classifier: `src/models/baseline.py`
- Config wiring/version: `BaseModel` in `src/models/baseline.py`
- Loss와 metric: `src/modules/model_interface.py`
- Production resolved entry: `configs/train_v19_medium.yaml`
- Architecture tests: `tests/test_base_model.py`

이 문서와 코드가 다르면 resolved config와 실제 코드를 우선하고 문서를 즉시 갱신한다.
