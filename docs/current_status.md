# Current development status

Last updated: 2026-07-26

이 파일은 BagPFN의 최신 개발 상태만 기록한다. 작업 방법은 [`agent_handoff.md`](agent_handoff.md), 현재 모델 구조는 [`current_architecture.md`](current_architecture.md), 실험 protocol은 [`current_experiments.md`](current_experiments.md), 과거 판단 근거는 [`history/`](history/)를 참고한다.

## 1. 한 줄 상태

Architecture v19의 bag-centered representation과 whitened CSP rank-1 covariance relation을 확정했고, B200 1장에서 100-epoch medium synthetic pretraining을 정상 완료했다. 현재 실행 중인 학습은 없으며 다음 구조 변경을 결정하는 단계다.

## 2. 코드 상태

- branch: `architecture/v19-shift-invariance`
- latest architecture commit: `e2181c0` (`Add covariance subspace relation to v19`)
- architecture version: 19
- production config: `configs/train_v19_medium.yaml`
- tests at architecture commit: 91 passed
- documentation reorganization: 완료 (docs/ 하위 최신 문서 및 history/ 분리 정리 완료)

확정 covariance relation:

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

모델 config가 전달되지 않을 때 relation 기본값은 checkpoint/state-dict 호환성을 위해 비활성 상태다. Production config에서 명시적으로 활성화한다.

## 3. 완료된 production 학습

- W&B: <https://wandb.ai/teasol/ICF/runs/in6rifr2>
- run: `v19_medium_csp_rank1_20260725_v19_medium_csp_rank1_100e`
- config: `configs/train_v19_medium.yaml`
- log: `logs/20260725_v19_medium_csp_rank1_100e/v19_medium_csp_rank1.out`
- checkpoint directory: `checkpoints/20260725_v19_medium_csp_rank1_100e/v19_medium_csp_rank1/`
- selected checkpoint: `epoch=052-val_ce_loss=0.5966.ckpt`
- completion: 100/100 epochs, `max_epochs=100` 정상 종료
- runtime failures: OOM, traceback, NaN/Inf 없음

학습 조건:

```text
B200 × 1
outer episode batch = 8
AdamW, LR 5e-4
BF16 mixed precision
global gradient clipping 1.0
5-epoch linear warm-up
val_ce_loss plateau scheduler
five medium tasks sampled uniformly
CE-only auxiliary weighting
```

## 4. 결과

| Metric | Best epoch 52 | Final epoch 99 |
|---|---:|---:|
| validation CE | **0.596554** | 0.597604 |
| accuracy | 0.67079 | 0.67079 |
| balanced accuracy | **0.66829** | 0.66798 |
| overall AUROC | **0.72996** | 0.72865 |
| composition AUROC | **0.80104** | 0.79785 |
| state AUROC | 0.62992 | **0.63261** |
| covariance AUROC | 0.60848 | **0.61663** |
| interaction AUROC | **0.75805** | 0.75186 |
| combined AUROC | **0.78763** | 0.78226 |
| covariance relation AUROC | 0.70575 | 0.70575 |
| covariance relation CE | 0.64028 | 0.64028 |
| covariance relation logit std | 0.28522 | 0.28522 |

LR은 warm-up 후 `5e-4`에 도달했고 plateau scheduler로 약 epoch 20, 41, 64, 80, 96에 감소해 최종 `1.5625e-5`가 됐다.

## 5. 현재 판정

- 100-epoch 수치 안정성과 실행 재현성은 통과했다.
- Epoch 0 대비 best validation CE 개선은 약 0.00456, overall AUROC 상승은 약 0.006으로 작다.
- CSP residual이 제공한 short-run 개선은 유지됐지만 장기 학습의 추가 수렴은 제한적이다.
- Covariance relation AUROC가 epoch 전체에서 동일한 것은 labelled context로 계산하는 비모수 고정 evidence이기 때문이다. 오류나 gradient failure를 의미하지 않는다.
- Composition, interaction, combined는 상대적으로 강하다.
- State와 covariance가 현재 주 병목이다.
- Covariance AUROC는 epoch 99가 더 높지만 checkpoint 기준은 사전에 정한 `val_ce_loss`이므로 epoch 52를 선택한다.
- 같은 config의 추가 epoch, resume 또는 단순 seed 반복은 현재 우선순위가 아니다.

## 6. 다음 작업 계획

### 우선 목표

State와 covariance가 context–query 관계에서 학습 가능한 representation으로 전달되도록 개선하되, 이미 통과한 composition/interaction/combined와 bag-shift invariance를 보존한다.

### 다음 구조 후보를 정할 때의 원칙

1. 고정 CSP relation을 제거하지 않고 기준 evidence로 유지한다.
2. 전체 모델 capacity를 무작정 확대하지 않는다.
3. Context-labelled subspace와 learned instance/slot representation의 결합을 우선 검토한다.
4. Query label을 사용하지 않는 episode-local adaptation이어야 한다.
5. 후보는 짧은 동일 validation bank diagnostic으로 선별한 뒤 100-epoch run은 한 후보에만 사용한다.
6. 비교 기준은 현재 best checkpoint와 동일한 overall/task별 metric이다.

### 다음 agent가 먼저 할 구체적 작업

1. Epoch 52 checkpoint에서 state/covariance task의 branch별 logit correlation과 calibration을 진단한다.
2. Covariance relation이 final logit에 고정 scale로 더해질 때 base covariance ridge와 중복 또는 충돌하는지 측정한다.
3. 다음 세 범주의 후보 중 가장 작은 변경부터 설계한다.
   - trainable gate conditioned on context-only class separation
   - CSP-projected feature를 learned relation head에 입력
   - state/covariance task에 공유 가능한 normalized context–query kernel feature
4. 기존 branch 제거 또는 architecture version 증가 전 diagnostic-only로 신호를 확인한다.
5. 후보 선택 후 short run과 full-size BF16 smoke를 수행한다.

## 7. 보존 artifact

- Production best: `checkpoints/20260725_v19_medium_csp_rank1_100e/v19_medium_csp_rank1/epoch=052-val_ce_loss=0.5966.ckpt`
- Production last: `checkpoints/20260725_v19_medium_csp_rank1_100e/v19_medium_csp_rank1/last.ckpt`
- Short CSP selection run: <https://wandb.ai/teasol/ICF/runs/tgm217sk>
- Production run: <https://wandb.ai/teasol/ICF/runs/in6rifr2>

기존 artifact를 삭제하거나 덮어쓰지 않는다.

## 8. History 연결

- CSP 이전 v19 구조: [`history/architecture_v19.md`](history/architecture_v19.md)
- 초기 acceptance protocol: [`history/v19_acceptance_protocol.md`](history/v19_acceptance_protocol.md)
- learnability ladder: [`history/learnability_ladder.md`](history/learnability_ladder.md)
- nuisance 결과: [`history/nuisance_ablation_c4_d_d0_d4.md`](history/nuisance_ablation_c4_d_d0_d4.md)
- v18 및 B200 baseline: [`history/architecture_v18.md`](history/architecture_v18.md), [`history/medium_b200_baseline.md`](history/medium_b200_baseline.md)
- generator/task 정의: [`history/synthetic_data_and_tasks.md`](history/synthetic_data_and_tasks.md)
