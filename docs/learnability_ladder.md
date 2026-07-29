# Architecture v18 learnability ladder

이 문서는 architecture v18을 처음부터 재현하기 위한 현재 실험 정의만 기록한다.
과거 실행 결과, checkpoint, W&B run과 이전 stage 이름은 재현 기준으로 사용하지 않는다.
Machine-readable 평가 계약은 `experiments/v18_learnability_protocol.yaml`이 원본이다.

## 실행 환경

- Conda: `/home/kimds/miniconda3/envs/BagPFN`
- GPU: NVIDIA RTX A6000 4장
- launcher: external `torchrun` DDP 4 rank
- rank별 episode batch: 2
- effective episode batch: 8
- precision: BF16 mixed
- global gradient clipping: 1.0
- training seed: 42, 43, 44
- seed 실패 시 architecture나 학습 설정을 바꾸지 않고 45, 46, ... 순서로 아직
  사용하지 않은 seed를 대체 투입한다.
- 실패 run은 삭제하거나 성공 run으로 덮어쓰지 않는다. 최종 표에 실패 seed, 실패
  epoch/원인과 대체 seed의 대응 관계를 함께 기록한다.
- NCCL P2P/NVLink: disabled (`NCCL_P2P_DISABLE=1`) for this A6000 host
- smoke test는 seed 42만 허용하며 최종 판정은 세 seed를 사용한다.

Launcher는 BagPFN 환경의 `bin/torchrun`을 직접 사용한다. 환경 위치가 다르면
`BAGPFN_CONDA_ENV`를 지정한다.

## Stage 이름

Stage 이름은 다음 목록만 사용하며 alias는 두지 않는다.

```text
A, B, C
C0, C1, C2, C3, C4, C5
C4-D, C4-N
D0, D1, D2, D3, D4
```

### Sanity gate

| Stage | 목적 | 조건 |
|---|---|---|
| A | 단일 episode 완전 과적합 | 고정 episode 1개, composition-only, nuisance/rare off, fixed query |
| B | finite-bank 암기 | 고정 episode 64개, composition-only, nuisance/rare off, fixed query |

A/B는 memorization 실험이므로 독립 evaluation bank를 적용하지 않는다. Effective
batch 8에서도 과거와 같은 optimizer step 수를 갖도록 epoch당 각각 1,024/2,048개
episode를 반복한다.

### Online generalization ladder

| Stage | 의미 |
|---|---|
| C | medium 전체 task family와 전체 nuisance |
| C0 | shared nonlinear manifold, composition-only, nuisance/rare off |
| C1 | episode별 orthogonal manifold |
| C2 | episode별 bounded-linear manifold, condition number ≤ 3 |
| C3 | episode별 nonlinear MLP manifold |
| C4 | medium difficulty와 전체 nuisance, composition-only, rare off |
| C5 | C4 + rare effect |
| C4-N | C3의 강한 signal/고정 크기 + 전체 nuisance |
| C4-D | medium difficulty, nuisance/rare off, composition-only |

C는 전체 task 조건의 고유 이름이다. C6라는 이름은 사용하지 않는다. C4-D와 C4-N도
독립된 고유 stage이며 다른 stage 이름으로 부르지 않는다.

### Single-nuisance ladder

D0-D4는 C4-D와 같은 medium composition-only base에서 nuisance 하나만 활성화한다.

| Stage | 단독 nuisance | scale |
|---|---|---:|
| D0 | global bag shift | 0.35 |
| D1 | bag×component shift | 0.12 |
| D2 | response/shared fraction logit noise | 0.65 |
| D3 | episode-common shared mixture variation | 0.70 |
| D4 | bag-specific shared mixture variation | 0.70 |

`configs/train_learnability_d_base.yaml`은 config 상속용 구현 파일일 뿐 실험 stage가
아니다.

## 학습과 checkpoint 선택

- 모든 stage: 최대 20 epoch
- 5-epoch linear warm-up을 유지한다.
- epoch 5 이전 checkpoint는 선택 대상에서 제외한다.
- epoch 5-19에서 checkpoint를 선택한다.
- A/B는 `val_accuracy` 최대, 나머지는 `val_ce_loss` 최소 checkpoint 하나를 저장한다.
- AUROC를 이용해 checkpoint를 선택하지 않는다.
- 저장 위치는 `checkpoints/learnability_ladder/<STAGE>/seed_<SEED>/`다.
- 모든 실행은 resume 없이 새 run으로 시작한다.

## Diagnostic validation

Learnability ladder의 목적은 독립 test 성능 추정이 아니라 학습 가능/불가능 경계를
찾는 것이다. 따라서 별도의 final evaluation bank는 사용하지 않으며, 고정 validation
bank에서 checkpoint를 선택하고 같은 validation 결과로 stage를 판정한다. AUROC로
checkpoint를 선택하지 않기 때문에 stage 판정 metric을 직접 최적화하는
metric cherry-picking은 방지한다.

| family | validation seed |
|---|---:|
| C full task | 50042 |
| C0-C3, C4-N manifold | 73042 |
| C4, C5, C4-D, D0-D4 | 50042 |

각 training seed에서 validation bank의 전체 query prediction을 합친 뒤 다음 metric을
계산한다.

- model AUROC
- oracle AUROC와 oracle-model AUROC gap
- balanced accuracy와 accuracy
- majority accuracy
- cross entropy와 empirical-prior cross entropy
- positive/negative recall
- mean/population/tail branch logit standard deviation

최종 표에는 seed별 값, 3-seed 평균과 최소 seed 성능을 기록한다. Config, architecture
version, validation bank seed, finite 검사 또는 oracle 격리가 계약과 다르면 결과를
`INVALID`로 처리한다.

## 실행

```bash
scripts/run_learnability_ladder.sh A
scripts/run_learnability_ladder.sh C0
scripts/run_learnability_ladder.sh C4-D
scripts/run_learnability_ladder.sh D0
```

인자는 대소문자를 허용하고 C4-D/C4-N의 하이픈 또는 underscore 표기를 허용한다.
