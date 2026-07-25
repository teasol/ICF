# Current experiments

Last updated: 2026-07-26

이 문서는 현재 architecture를 어떤 synthetic problem과 protocol로 평가하는지 설명한다. 모델 계산 구조는 [`current_architecture.md`](current_architecture.md), 최신 결과와 다음 결정은 [`current_status.md`](current_status.md)를 참고한다.

## 1. 현재 실험 목표

외부 ICI validation보다 먼저 medium synthetic data에서 다섯 task의 학습 가능성을 확보한다. 현재 질문은 전체 CE만 낮추는 것이 아니라, context–query 관계를 이용해 composition, state, covariance, interaction과 combined signal을 각각 구분할 수 있는지다.

현재 실행 중인 job은 없다. 기준 run은 100 epochs를 완료했으며 다음 구조 후보를 설계하는 단계다.

## 2. Production synthetic episode

Resolved config: `configs/train_v19_medium.yaml`

```text
online training episodes per epoch: 4,096
fixed validation episodes: 104, seed 50042
fixed test episodes: 104, seed 60042
outer episode batch: 8
bags per episode: 60–100
instances per bag: 500–1,000
latent dimension: 32
observed dimension: 512
random manifold: 3-layer MLP, hidden dimension 96
output normalization: enabled
class labels: not explicitly balanced
```

각 episode는 새로운 latent component와 nonlinear manifold를 갖는다. Training episode는 replay하지 않는 online stream이고 validation/test bank는 비교 가능성을 위해 고정한다.

## 3. Nuisance 구성

현재 medium production은 nuisance를 포함한다.

```text
global bag shift scale: 0.35
bag × component shift scale: 0.12
response/shared fraction logit noise: 0.65
episode-common shared mixture variation: 0.70
bag-specific shared mixture variation: 0.70
observation noise: 0.01
rare response probability: 0.15
rare response fraction: 0.02–0.08
```

이 nuisance는 architecture 변경 비교 중 임의로 낮추지 않는다. Bag-centered v19의 핵심 목적 중 하나는 global bag shift에 대한 불변성이다.

## 4. 다섯 training task

모든 task는 probability 0.20으로 균등 sampling된다. Task metadata를 반환해 validation metric을 따로 집계한다.

### Composition

Response score가 responsive component의 abundance를 변화시킨다. Positive response는 response-related component fraction을 증가시킨다. State/covariance effect는 task signal로 사용하지 않는다.

### State

Responsive component instance의 latent mean/state를 episode별 direction으로 이동시킨다. Abundance 변화는 label signal이 아니며 nuisance mixture variation과 구분해야 한다.

### Covariance

Responsive component의 episode별 latent direction을 따라 dispersion을 response score에 따라 확대/축소한다. Mean shift나 abundance가 아니라 within-population geometry가 signal이다.

### Interaction

Shared component 중 하나를 responsive population으로 선택하고 state와 covariance effect를 함께 적용한다. 어떤 shared population에서 변화가 일어났는지와 변화 형태를 결합해 해석해야 한다. 이 task에서는 rare response를 사용하지 않는다.

### Combined

Response-specific component에 composition, state와 covariance effect를 함께 적용한다. 여러 branch evidence를 결합해야 한다.

## 5. Label과 evaluation baseline

Medium generator는 bag label을 명시적으로 balanced하게 만들지 않는다. 따라서 accuracy만으로 학습을 판정하지 않는다.

항상 함께 확인할 항목:

- query positive fraction
- majority-class accuracy
- empirical-prior constant CE
- balanced accuracy
- AUROC
- class별 recall
- overall 및 task별 CE/AUROC

Checkpoint 선택은 사전에 고정한 `val_ce_loss`를 사용한다. 특정 task AUROC가 더 높다는 이유로 checkpoint 선택 규칙을 사후 변경하지 않는다.

## 6. Production training protocol

```text
optimizer: AdamW
learning rate: 5e-4
betas: 0.9, 0.999
weight decay: 0.01
precision: BF16 mixed
clip: global gradient norm 1.0
warm-up: 5 epochs, linear from 0.1× LR
scheduler: ReduceLROnPlateau after warm-up
monitor: val_ce_loss
patience: 10
factor: 0.5
cooldown: 5
minimum LR: 1e-6
maximum epochs: 100
hardware: B200 × 1
strategy: single-process, devices=1
loss: CE-only auxiliary weighting
```

기존 FP16 LR `1e-3` run은 GradScaler가 붕괴하고 train loss가 NaN이 되었으므로 resume 또는 기준 checkpoint로 사용하지 않는다.

## 7. 현재 기준 실험

### CSP short selection

- W&B: <https://wandb.ai/teasol/ICF/runs/tgm217sk>
- config: `configs/train_covariance_csp_scale05_short8.yaml`
- 8 epochs
- residual scale 0.5 선택
- best validation CE 0.5996
- overall AUROC 0.7283
- covariance AUROC 0.6218

### Production 100 epochs

- W&B: <https://wandb.ai/teasol/ICF/runs/in6rifr2>
- config: `configs/train_v19_medium.yaml`
- log: `logs/20260725_v19_medium_csp_rank1_100e/v19_medium_csp_rank1.out`
- best checkpoint: `epoch=052-val_ce_loss=0.5966.ckpt`
- final status: 100/100 epochs, finite, no OOM

핵심 best metric:

```text
validation CE: 0.596554
overall AUROC: 0.729964
composition AUROC: 0.801038
state AUROC: 0.629925
covariance AUROC: 0.608480
interaction AUROC: 0.758050
combined AUROC: 0.787634
covariance relation AUROC: 0.705748
```

해석과 다음 결정은 `current_status.md`를 기준으로 한다.

## 8. Architecture 후보 평가 구조

새 후보는 긴 job부터 실행하지 않는다.

### Phase 1 — Diagnostic-only

- 기존 best checkpoint 또는 동일 fixed validation bank 사용
- 새 relation/gate 자체의 AUROC, CE, logit std, class separation 측정
- query label은 metric 계산에만 사용
- base final logit을 변경하지 않는 diagnostic mode 우선
- 동일 episode bank로 후보를 직접 비교

### Phase 2 — Full-size BF16 smoke

최소 outer batch 8의 full-size episode에서 확인한다.

- loss finite
- 모든 사용 gradient finite
- global norm clipping 적용
- optimizer step과 parameter update 수행
- B200 memory 안정성
- 단일 episode와 outer-batch 결과 일치

### Phase 3 — Short training

- 동일 optimizer/precision/data semantics 유지
- warm-up 이후 target LR 구간까지 관찰
- overall뿐 아니라 다섯 task별 CE/AUROC 비교
- 기존 강한 task의 회귀 여부 확인

### Phase 4 — Production 100 epochs

Diagnostic과 short run을 통과한 하나의 후보만 100 epochs로 승격한다. 현재 관찰상 여러 seed 반복보다 동일 fixed validation bank에서 여러 구조 후보를 비교하는 것이 우선이다.

## 9. 현재 다음 후보의 평가 질문

다음 변경은 state/covariance 병목을 목표로 한다.

1. Context-only class separation으로 covariance relation residual을 gate하면 calibration이 개선되는가?
2. CSP-projected scalar/low-rank feature를 learned relation head에 넣으면 고정 AUROC 0.7057을 trainable evidence로 활용할 수 있는가?
3. 기존 covariance ridge와 CSP relation이 중복 또는 반대 방향으로 작동하는 episode가 있는가?
4. State와 covariance 양쪽에 공유 가능한 normalized context–query kernel feature가 있는가?
5. Composition/interaction/combined 성능과 bag-shift invariance를 보존하는가?

최초 diagnostic은 epoch 52 best checkpoint를 기준으로 한다. 같은 production run을 resume해 구조를 바꾸지 않는다.

## 10. 공통 metric

최소 기록 항목:

```text
train_loss, train_ce_loss, val_ce_loss
val_accuracy, val_balanced_accuracy, val_auroc
val/{task}/ce_loss
val/{task}/accuracy
val/{task}/balanced_accuracy
val/{task}/auroc
branch별 logit std
covariance relation AUROC/CE/logit std/class separation
learning rate
global gradient norm
```

Oracle diagnostic이 활성화된 ablation에서는 model AUROC와 oracle AUROC를 같은 aggregation 단위에서 비교하지만 oracle은 입력과 loss에 사용하지 않는다.

## 11. 실험 artifact 규칙

- 모든 새 run은 고유 `ICF_RUN_TIME`을 사용한다.
- 새 architecture 실험은 `ckpt_path=null`로 시작한다.
- Diagnostic checkpoint loading과 production resume를 구분한다.
- logs/checkpoints는 persistent launcher 경로를 사용한다.
- W&B credential이 있으면 online, 없으면 offline으로 실행한다.
- secret을 코드나 로그에 출력하지 않는다.
- 기존 artifact를 삭제하거나 덮어쓰지 않는다.

## 12. History와의 관계

과거 learnability ladder와 nuisance ablation은 [`history/`](history/)에 보존한다. 현재 experiment naming과 실행 기준은 이 문서가 우선이다. 과거 alias나 3-seed protocol을 새 실험에 자동 적용하지 않는다.
