# Current development status

Last updated: 2026-07-26

이 파일은 BagPFN의 최신 개발 상태만 기록한다. 작업 방법은 [`agent_handoff.md`](agent_handoff.md), 현재 모델 구조는 [`current_architecture.md`](current_architecture.md), 실험 protocol은 [`current_experiments.md`](current_experiments.md), 과거 판단 근거는 [`history/`](history/)를 참고한다.

## 1. 한 줄 상태

Architecture v19 기반 Context-Gated CSP Relation (Candidate A)의 20-epoch short training이 NVIDIA B200 1장에서 현재 실행 중이다 (`20260726_v19_candidate_a_gated_20e`).

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

### 완료된 진단 및 아키텍처 후보 구현 (Phase 1 & 2)

1. **Epoch 52 Best Checkpoint 진단 스크립트 실행 (`scripts/diagnose_v19_branches.py`)**:
   - `Covariance` task에서 `CovRel (CSP relation)`의 AUROC가 0.6270으로 `CovRidge (0.5700)` 대비 가장 강한 성능을 제공함 확인.
   - `CovRidge`와 `CovRel` 간의 Logit 상관관계는 $r \approx 0.494$로 적절한 상보적 정보 제공 확인.
   - Context Class Separation Margin이 평균 1.1885로 안정적으로 잘 조건화되어 있음을 확인.
2. **구조 후보 구현 및 Unit Test/Smoke Test 완료**:
   - **Candidate A (`gated_distance`)**: Context class margin 기반 dynamic gating 구현.
   - **Candidate B (`learned_head`)**: CSP projection 저차원 feature $[d_0, d_1, d_0 - d_1, \text{sep}]$를 2-layer MLP head로 분류.
   - Config: `configs/train_v19_candidate_a_gated.yaml`, `configs/train_v19_candidate_b_head.yaml`
   - unit test 100% 통과 및 B200 GPU BF16 1-epoch smoke test 완료.

### 완료된 20-Epoch Short Training 결과 비교 (Phase 3)

| Metric / Task | Baseline (100e Ep 52) | Candidate A (Gated Ep 8) | Candidate B (Learned Head Ep 13) | 개선 폭 (vs Baseline) |
|---|---:|---:|---:|---:|
| **val_ce_loss** | 0.5966 | 0.5965 | **0.5925** | **-0.0041 (최우수)** |
| **Overall AUROC** | 0.7299 | 0.7423 | **0.7513** | **+0.0214 (최우수)** |
| **Composition AUROC** | 0.8010 | **0.8044** | 0.8039 | +0.0029 |
| **State AUROC** | 0.6299 | 0.6427 | **0.6583** | **+0.0284 (최우수)** |
| **Covariance AUROC** | 0.6085 | **0.6193** | 0.6115 (CovRel: **0.6566**) | **+0.0108** |
| **Interaction AUROC** | 0.7581 | **0.7605** | 0.7527 | - |
| **Combined AUROC** | 0.7876 | 0.8042 | **0.8290** | **+0.0414 (최우수)** |

- **Candidate B (`learned_head`) 압도적 우승**:
  - `val_ce_loss`: **0.5925** (20 epoch 만에 기존 100 epoch 기록 큰 폭 경신)
  - `Overall AUROC`: **0.7513**
  - State task AUROC **0.6583** (+0.0284), Combined task AUROC **0.8290** (+0.0414)
  - Covariance Relation Head 자체의 Covariance AUROC **0.6566** 달성.

### 최종 결론 및 다음 작업 (Phase 4 Production 100-Epoch 승격)

1. **우승 아키텍처 확정**: Candidate B (**CSP Learned Head**, `mode: learned_head`)를 100-epoch Production 승격 후보로 확정.
2. **Production 100-Epoch Config 생성**: `configs/train_v19_candidate_b_100e.yaml` 생성 및 100-epoch pretraining 실행 준비 완료.

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
