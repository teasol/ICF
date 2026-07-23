# Learnability ladder

이 ladder는 구현 오류, finite-bank memorization 실패, episode 간 generalization 실패를 단계적으로 분리한다. 각 단계를 통과하기 전에는 다음 단계의 online 성능을 architecture 품질의 근거로 해석하지 않는다.

모든 bag은 순서 없는 instance 집합이다. 코드에 남은 `population`은 특정 도메인이 아니라 잠재 instance subgroup을 뜻한다.

## 공통 원칙

- architecture v18과 checkpoint 호환성 검사는 그대로 유지한다.
- A/B/C 모두 CE-only로 맞춰 objective 차이를 제거한다.
- `train_accuracy`, `val_accuracy`, CE, branch별 logit std와 residual scale을 기록한다.
- 각 실행은 checkpoint resume가 아닌 새 run이어야 한다.
- A와 B는 deterministic query/context split을 사용한다.
- 각 class의 첫 bag을 context로 보호하므로 class memory에는 항상 두 class가 존재한다.
- query positive fraction, majority-class accuracy, empirical-prior constant CE, balanced accuracy, AUROC와 class별 recall을 함께 기록한다.
- 한 class가 없는 query subset의 class별 metric은 `*_valid` 로그를 함께 확인한다.

## Test A — 한 episode 완전 과적합

설정: `configs/train_learnability_a.yaml`

- 하나의 고정 episode만 사용한다.
- 한 epoch에 같은 episode를 128번 반복한다.
- bag 64개, bag당 instance 256개다.
- composition task만 사용한다.
- episode, manifold, bag permutation과 labels는 seed 71001로 고정된다.
- bag shift, bag×component shift, mixture nuisance, observation noise를 끈다.
- rare-effect를 끈다.
- query 16개를 고정한다.
- CE만 최적화한다.

실행:

```bash
scripts/run_learnability_ladder.sh a
```

통과 기준은 train accuracy가 사실상 100%에 도달하고 train CE가 0에 가까워지는 것이다. 권장 판정은 `train_accuracy >= 0.99`다. 같은 episode를 쓰는 validation도 함께 보되, training과 validation의 query subset이 완전히 같지는 않으므로 구현 sanity의 일차 판정은 train accuracy다.

A가 실패하면 데이터 난이도나 meta-generalization을 논하기 전에 다음을 조사한다.

1. query/context masking과 query label 누출·오정렬
2. bag permutation 후 label 순서
3. class memory와 class label 연결
4. episode/outer-batch loss aggregation
5. ridge score 또는 최종 logit 방향
6. branch별 gradient 연결과 optimizer update

### 최초 실행 결과 (2026-07-22)

- run: `learnability_a_20260722_161231`
- W&B: https://wandb.ai/teasol/ICF/runs/qw5gd7p4
- 640 optimizer steps 후 `train_accuracy=1.0`, `val_accuracy=1.0`
- `train_ce_loss=1.63e-7`, `val_ce_loss=4.75e-6`
- Test A는 통과했으며 진단 목적 달성 후 수동 종료했다.

이 결과로 query/context 연결, label permutation, class-memory label 연결, loss 방향과 gradient 흐름이 최소한 단일 고정 episode에서는 정상임을 확인했다.

## Test B — 고정 64-episode bank 과적합

설정: `configs/train_learnability_b.yaml`

- seed 72001부터 생성한 서로 다른 64개 episode를 고정한다.
- 한 epoch에 bank를 네 번 순회하여 256 optimizer step을 수행한다.
- 각 episode는 서로 다른 고정 random manifold를 가진다.
- composition-only, nuisance off, rare-effect off, CE-only 조건은 A와 같다.
- episode별 query/context split도 고정한다.
- validation은 동일한 64개 bank를 사용한다.

실행:

```bash
scripts/run_learnability_ladder.sh b
```

판정:

- A와 B 모두 성공: architecture와 optimization이 적어도 finite episode family를 표현할 수 있다.
- A 성공, B 실패: capacity, class-memory 압축, branch fusion 또는 여러 manifold를 동시에 표현하는 구조가 병목일 가능성이 크다.
- B의 train만 성공하고 같은 bank validation이 실패: masking 차이 또는 query subset 의존성을 조사한다.

Bank 크기는 `fixed_episode_count`를 32–128 범위에서 바꿔 capacity curve를 측정할 수 있다. `episodes_per_epoch`는 bank 크기의 정수배로 유지한다.

### 최초 실행 결과 (2026-07-22)

- run: `learnability_b_20260722_161604`
- W&B: https://wandb.ai/teasol/ICF/runs/phws3m7n
- epoch 3에서 이미 `train_accuracy=1.0`, `val_accuracy=1.0`이었다.
- 종료 시점(epoch 4, 1,280 optimizer steps)은 `train_ce_loss=2.52e-5`, `val_ce_loss=2.08e-5`였다.
- best checkpoint: `checkpoints/20260722_161604/learnability_b/epoch=001-val_accuracy=1.0000.ckpt`
- Test B는 통과했으며 진단 목적 달성 후 종료했다.

따라서 architecture v18과 현재 optimizer는 서로 다른 고정 manifold를 갖는 64개 episode bank를 동시에 표현하고 암기할 capacity가 있다. A와 B의 결과만 보면 단일 episode 경로의 구현 오류나 단순한 finite-bank capacity 부족이 주된 병목일 가능성은 낮다.

## Test B2 — 고정 bank, random query

설정: `configs/train_learnability_b2.yaml`

B와 같은 고정 64-episode bank를 사용하되 `fixed_training_queries: false`로 바꿔 매 optimizer step마다 context/query subset을 다시 뽑는다.

### 최초 실행 결과 (2026-07-22)

- run: `learnability_b2_20260722_164318`
- W&B: https://wandb.ai/teasol/ICF/runs/3sjiprf1
- epoch 3: `train_accuracy=0.9990`, `val_accuracy=1.0`, `train_ce_loss=0.00407`, `val_ce_loss=6.10e-5`
- validation positive fraction `0.4663`, majority accuracy `0.6250`, empirical-prior CE `0.6432`
- validation balanced accuracy, AUROC, positive recall, negative recall은 모두 `1.0`

Test B2는 통과했다. random query masking과 context 구성 변화는 병목이 아니므로 C와의 차이는 unseen episode 또는 manifold 변화 쪽으로 좁혀진다.

## Test C — 새로운 episode generalization

설정: `configs/train_learnability_c.yaml`

- 현재 medium online episode stream을 사용한다.
- episode마다 새로운 random manifold를 생성한다.
- composition, state, covariance, interaction, combined task를 모두 복원한다.
- bag nuisance와 rare-effect를 복원한다.
- query 수와 query/context split을 다시 무작위화한다.
- A/B와 비교 가능하도록 CE-only는 유지한다.
- validation은 고정된 unseen episode set이다.

실행:

```bash
scripts/run_learnability_ladder.sh c
```

판정:

- A/B 성공, C가 random 수준: 구현과 finite-bank capacity는 동작하지만 episode 간 invariance를 학습하지 못한다.
- C train만 개선되고 validation이 정체: online episode family의 invariant rule 대신 generator-specific shortcut 또는 optimization 편향을 학습할 가능성이 있다.
- 쉬운 task부터 추가하는 세부 ladder가 필요하면 composition-only online → nuisance on → rare-effect on → 다섯 task 순으로 한 요소씩 복원한다.

### 예비 실행 결과 (2026-07-22)

- run: `learnability_c_20260722_161749`
- W&B: https://wandb.ai/teasol/ICF/runs/ol4unr4a
- resume 없이 새로운 run으로 시작해 5-epoch linear warm-up 전체를 검사했다.
- epoch 0: `train_accuracy=0.5511`, `val_accuracy=0.5477`, `train_ce_loss=0.68266`, `val_ce_loss=0.68279`
- epoch 1의 best validation: `val_accuracy=0.5548`, `val_ce_loss=0.68221`
- epoch 4: `train_accuracy=0.5488`, `val_accuracy=0.5518`, `train_ce_loss=0.68446`, `val_ce_loss=0.68347`
- best checkpoint: `checkpoints/20260722_161749/learnability_c/epoch=001-val_ce_loss=0.6822.ckpt`
- NaN, Inf, OOM 또는 실행 예외는 없었다. warm-up 종료 후 진단 run을 종료했다.

이 실행은 warm-up 종료 직후 멈췄으므로 최종 실패로 판정하지 않는다. target learning rate에서 최소 2–3 epoch를 추가 관찰하고 prevalence baseline과 discrimination metric을 함께 비교해야 plateau를 확정할 수 있다.

### Full-LR 재실행 결과 (2026-07-22)

- run: `learnability_c_full_lr_20260722_164520`
- W&B: https://wandb.ai/teasol/ICF/runs/shji88x1
- resume 없이 처음부터 다시 실행했고 epoch 5–7을 target LR `1.0e-3`에서 관찰했다.
- epoch 7: train accuracy `0.5558`, validation accuracy `0.5530`, train CE `0.68237`, validation CE `0.68202`
- validation positive fraction `0.4906`, episode-macro majority accuracy `0.6078`, empirical-prior CE `0.6583`
- validation balanced accuracy `0.5431`, AUROC `0.5670`, positive recall `0.5322`, negative recall `0.5540`

모델 accuracy는 majority baseline보다 낮고 CE도 empirical-prior CE보다 높다. full-LR 세 epoch에서도 반전이 없었으므로 원래 medium 조건의 Test C는 실패로 확정한다. AUROC가 0.5보다 조금 높아 약한 ranking signal은 있지만 유용한 decision rule로 연결되지 않는다.

## Manifold ladder

공통 controlled-online 설정은 `configs/data/learnability_manifold.yaml`이다. composition-only, nuisance off, rare-effect off, random query, CE-only를 유지하고 manifold만 바꾼다.

| Stage | Config | Manifold | 분리하는 질문 |
|---|---|---|---|
| C0 | `train_learnability_c0.yaml` | global fixed nonlinear MLP | 고정 좌표계에서 online composition rule을 학습하는가 |
| C1 | `train_learnability_c1.yaml` | episode별 orthogonal isometry | 거리와 각도를 보존한 rotation 변화에 일반화하는가 |
| C2 | `train_learnability_c2.yaml` | episode별 bounded linear, condition number ≤ 3 | 제한된 선형 왜곡에 일반화하는가 |
| C3 | `train_learnability_c3.yaml` | episode별 nonlinear MLP | 현재 nonlinear folding에 일반화하는가 |

C0 실패는 manifold 이전에 online composition rule이 병목임을 뜻한다. C0 성공/C1 실패는 rotation invariance 부족, C1 성공/C2 실패는 제한된 metric distortion, C2 성공/C3 실패는 nonlinear folding이 병목임을 뜻한다. C4–C6는 C0–C3에서 실제 통과한 manifold를 고정한 뒤 nuisance, rare-effect, 전체 task family 순서로 복원한다.

### 최초 실행 결과 (2026-07-22)

| Stage | Validation accuracy | Majority | Validation CE | Prior CE | Balanced accuracy | AUROC | 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| C0 | 0.9689 | 0.6220 | 0.1302 | 0.6462 | 0.9697 | 0.9965 | 통과 |
| C1 | 0.9756 | 0.6198 | 0.1371 | 0.6464 | 0.9765 | 1.0000 | 통과 |
| C2 | 0.9697 | 0.6117 | 0.1461 | 0.6526 | 0.9654 | 0.9973 | 통과 |
| C3 | 0.9408 | 0.6191 | 0.2056 | 0.6472 | 0.9381 | 0.9882 | 통과 |

C0–C3는 모두 첫 epoch에 통과했다. 따라서 controlled composition task에서는 shared, orthogonal, bounded-linear, random nonlinear manifold 모두 unseen episode에 일반화한다. 원래 C의 실패 원인을 manifold 변화 하나로 돌릴 수 없다.

후속 설정은 `train_learnability_c4.yaml`(composition+nuisance), `train_learnability_c5.yaml`(C4+rare-effect), `train_learnability_c6.yaml`(전체 task)이다. C4 이후에도 C와 동일하게 warm-up 이후 full-LR 3 epoch를 판정 구간으로 사용한다.

### C4 실행 결과와 교정 (2026-07-22)

- run: `learnability_c4_20260722_170744`
- W&B: https://wandb.ai/teasol/ICF/runs/bnqj657o
- epoch 9: train accuracy `0.5586`, validation accuracy `0.5707`, train CE `0.68129`, validation CE `0.67735`
- validation majority accuracy `0.5948`, empirical-prior CE `0.66142`, balanced accuracy `0.5711`, AUROC `0.5956`
- full-LR 구간에서도 baseline을 이기지 못해 실패로 판정하고 종료했다.

현재 C4 config는 C3에서 nuisance만 추가한 순수 ablation이 아니다. medium의 약한 effect range, component 수와 비율, bag/instance 수 범위도 동시에 복원한다. 따라서 C4 실패만으로 nuisance를 단독 원인으로 확정하지 않는다.

다음 최소 실험은 두 개다.

1. `C4-N`: C3의 강한 composition signal과 고정 크기는 유지하고 medium의 shift·mixture nuisance 다섯 항목만 켠다.
2. `C4-D`: nuisance는 끈 채 medium의 effect strength, component heterogeneity와 크기 범위만 복원한다.

`C4-N`만 실패하면 nuisance group을 global shift, component shift, mixture-logit variability 순으로 분해한다. `C4-D`만 실패하면 response margin/effect scale을 먼저 sweep하고 component heterogeneity와 크기 범위를 나중에 분리한다. 둘 다 통과할 때만 결합 interaction을 검사한다. C5 rare-effect와 C6 전체 task는 이 경계를 통과한 뒤 실행한다.

### C4-N / C4-D 분리 결과 (2026-07-22)

- C4-N, LR `5e-4`: epoch 65에서 validation accuracy `0.6546`, majority `0.6102`, validation CE `0.6251`, prior CE `0.6551`, balanced accuracy `0.6538`, AUROC `0.7024`. 수치 오류 없이 이전 LR `1e-3` 실패 지점을 넘겨 통과로 판정했다.
- C4-D, LR `5e-4`: epoch 77에서 validation accuracy `0.8887`, majority `0.5936`, validation CE `0.2698`, prior CE `0.6641`, balanced accuracy `0.8885`, AUROC `0.9466`. NaN, Inf, OOM 없이 통과로 판정했다.

두 축은 각각 학습 가능하다. 다음 단계는 medium difficulty와 nuisance를 함께 결합하되 composition-only와 rare off를 유지하여 interaction을 검사하는 것이다. 결합을 통과한 뒤 C5 rare-effect, 마지막으로 C6 전체 task family를 복원한다.

## D0–D5 nuisance ablation

C4-D를 D0 기준점으로 사용한다. 모든 stage는 online episode, composition-only, medium difficulty, episode별 nonlinear manifold, random query, rare-effect off, CE-only, AdamW LR `5e-4`를 공유한다. `return_oracle_diagnostics: true`이며 oracle scalar는 모델 입력이나 loss에 사용하지 않는다.

| Stage | 단독 활성 nuisance | Scale |
|---|---|---:|
| D0 | 없음 | 0 |
| D1 | global bag shift | 0.35 |
| D2 | bag×component shift | 0.12 |
| D3 | response/shared fraction logit noise | 0.65 |
| D4 | episode-common shared mixture variation | 0.70 |
| D5 | bag-specific shared mixture variation | 0.70 |

`configs/train_learnability_d0.yaml`이 공통 base이며 D1–D5는 `base_config`로 이를 상속하고 nuisance key 하나만 override한다. 기존 `configs/train_learnability_c4.yaml`은 다섯 nuisance를 모두 켠 D-all 대조군이다.

Oracle abundance는 generator가 아는 responsive component membership의 bag별 fraction이다. 전체 membership은 Dataset 밖으로 반환하지 않는다. Validation에서는 labelled context의 abundance와 label로만 1차원 ridge classifier를 fit하고 query abundance를 예측한다. Query label은 metric 계산에만 사용한다.

기록 metric:

- `val/oracle_abundance_accuracy`
- `val/oracle_abundance_balanced_accuracy`
- `val/oracle_abundance_auroc`
- `val/oracle_abundance_ce`
- `val/oracle_abundance_snr`
- `val/oracle_model_auroc_gap`

model과 oracle AUROC가 같이 내려가면 generator separability 감소로 해석한다. oracle은 유지되고 model만 내려가면 architecture nuisance 처리 문제다. D3 하락은 abundance signal과 fraction noise의 충돌, D1/D2 gap 증가는 embedding shift invariance, D4/D5 gap 증가는 mixture 변화에 따른 slot alignment 문제를 우선 조사한다.

실행:

```bash
scripts/run_learnability_ladder.sh d0
scripts/run_learnability_ladder.sh d1
scripts/run_learnability_ladder.sh d2
scripts/run_learnability_ladder.sh d3
scripts/run_learnability_ladder.sh d4
scripts/run_learnability_ladder.sh d5
```

실행 결과와 비교 표는 [D0–D5 nuisance ablation 결과](nuisance_ablation_d0_d5.md)에 정리했다.

## 구현된 진단 옵션

### `fixed_episode_count`

`SyntheticEpisodeDataset`에서 fixed seed와 함께 사용한다. Dataset index를 bank 크기로 modulo하여 epoch마다 동일한 episode bank를 반복한다. 기본값은 `null`이므로 기존 online medium stream에는 영향이 없다.

### `fixed_training_queries`

각 class의 첫 bag을 context로 보호하고 나머지 index 중 앞에서부터 고정 query를 고른다. 기본값은 `false`이며 production training의 무작위 masking은 유지된다.

## 결과 기록표

| Stage | Bank | Task family | Nuisance | Query | 목표 |
|---|---:|---|---|---|---|
| A | 1 | composition | off | fixed | train accuracy ≥ 0.99 |
| B | 64 | composition | off | fixed | 통과: train/val accuracy 1.0 |
| B2 | 64 | composition | off | random | 통과: train/val accuracy 약 1.0 |
| C | online | all five | on | random | 예비 결과: full-LR 확인 필요 |

각 run에서 최소한 다음을 함께 기록한다.

- best/final train CE와 accuracy
- best/final validation CE와 accuracy
- mean/subgroup/tail branch logit std
- subgroup/tail/fusion residual scale
- gradient finite 여부와 global norm clipping
- best checkpoint와 W&B URL
