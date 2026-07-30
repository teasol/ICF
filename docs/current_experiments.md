# Current experiments

**Last updated**: `2026-07-30 23:40:00 KST`
**Architecture Version**: `22` (`architecture_version = 22`)

이 문서는 v22 기준 실험 프로토콜과 실행 명령어를 설명합니다. v21 retrieval 시대의 실험 기록은 [`history/v21_retrieval_experiments.md`](history/v21_retrieval_experiments.md)로 이관되었습니다.

> [!CAUTION]
> **v22는 기존 체크포인트를 전부 무효화합니다.** retrieval 계층 제거로 `architecture_version`이 22로 올라갔고, `ModelInterface.on_load_checkpoint`가 v21 이하 체크포인트를 거부합니다. 아래 모든 실험은 **처음부터 다시** 돌려야 합니다.

---

## 0. 실험 전략: 합성 데이터로 결정하고, ICI는 최종 테스트에만 (2026-07-29 확정)

> [!IMPORTANT]
> **아키텍처 선택·하이퍼파라미터 튜닝·모든 반복 실험은 합성 데이터에서만 수행합니다.**
> **ICI 실데이터는 최종 확인 1회에만 사용합니다.**

### 왜 이렇게 나누는가

**실측 비교** (둘 다 95% bootstrap CI, 합성은 episode cluster 방식으로 보수적으로 계산):

| | 표본 | AUROC | 95% CI | **CI 폭** |
|---|---|---:|---|---:|
| ICI 5-fold (v21 Phase 6c) | 87명 | 0.5454 | [0.422, 0.664] | **0.242** |
| 합성 val (v22 기준선, 1,000 eps) | 1,000 episodes / 16,330 query | 0.7078 | [0.696, 0.719] | **0.021** |

합성 쪽 구간이 **약 12배 좁습니다** (0.021 vs 0.242). 게다가 합성은 `episodes_per_epoch`를 늘려 더 좁힐 수 있지만, ICI는 87명이 상한이라 아무리 seed를 늘려도 좁아지지 않습니다.

신호 자체도 합성 쪽이 훨씬 강합니다 (AUROC 0.71 vs ICI의 0.55). 즉 **아키텍처 변경이 실제로 효과가 있다면 합성에서 먼저, 더 뚜렷하게 보입니다.**

v21의 실패는 **검출력이 없는 지표 위에서 아키텍처를 반복 비교한 것**이었습니다. 또한 ICI를 반복해서 들여다보면 그 87명에 과적합됩니다 (선택 편향). ICI를 최종 테스트로 남겨두어야 그 숫자가 의미를 갖습니다.

### 규칙

| 단계 | 데이터 | 지표 | 반복 |
|---|---|---|---|
| 아키텍처 탐색·튜닝 | 합성 val | `scripts/evaluate_synthetic.py`의 AUROC + episode cluster CI | 자유롭게 |
| 최종 확인 | ICI 5 seed × 5 fold + 외부 코호트 | `scripts/evaluate_protocol.py` | **후보 확정 후 1회** |

- 합성 val에서 이긴 후보만 ICI로 넘깁니다. ICI 결과를 보고 다시 아키텍처를 고치기 시작하면 이 프로토콜은 무효가 됩니다.
- ICI 결과가 기대에 못 미쳐도 **그것 자체로는 아키텍처를 되돌릴 근거가 되지 않습니다** — §1에 따라 ±0.13 이내 변동은 이 코호트가 구분할 수 없는 범위입니다.
- 합성 데이터에서 검출력이 부족하면 `episodes_per_epoch`를 늘리면 됩니다. 이것이 ICI 대비 합성 데이터의 결정적 장점입니다.

---

## 1. 평가 프로토콜 (실험 시작 전 반드시 읽을 것)

### ① 이 코호트가 검출할 수 있는 효과 크기

ICI 코호트는 n=87 (positive 37 / negative 50)입니다. `scripts/power_analysis.py`로 측정한 검정력 (baseline AUROC 0.55, 모델 간 상관 ρ=0.7 — 실제 Phase 6b vs 6c의 Pearson ρ=0.737에서 추정):

| 실제 AUROC 향상 | 검출 확률(power) |
|---:|---:|
| +0.02 | 15% |
| +0.05 | 26% |
| +0.10 | 66% |
| **+0.15** | **92%** |
| +0.20 | 99% |

> [!IMPORTANT]
> **+0.13~0.15 AUROC 이상을 기대할 수 없는 실험은 이 코호트에서 돌릴 가치가 없습니다.** 그보다 작은 효과는 있어도 못 찾습니다(power < 80%). v21 시대에 쫓던 0.004~0.04 차이는 검출 확률이 15~26%로, 사실상 동전 던지기였습니다.
>
> 이 표가 뜻하는 것: 아키텍처를 조금씩 바꿔가며 ICI AUROC로 우열을 가리는 방식 자체가 **이 코호트에서는 작동하지 않습니다.** 큰 효과를 노리거나, 코호트를 키우거나, 합성 데이터처럼 n을 늘릴 수 있는 곳에서 판단해야 합니다.

재현: `python scripts/power_analysis.py`

### ② 세 가지 수치를 구분해서 볼 것

`scripts/evaluate_protocol.py`는 서로 다른 질문에 답하는 세 수치를 보고합니다:

| 수치 | 무엇을 재는가 | seed를 늘리면? |
|---|---|---|
| per-seed AUROC | 한 partition에서의 5-fold CV 결과 (87명 전원) | — |
| across-seed mean ± SD | partition/학습 재현성. SD가 크면 단일 seed 숫자는 못 믿음 | SD는 그대로, 평균의 표준오차는 감소 |
| pooled bootstrap CI | **코호트 자체의 표본 오차** | **줄어들지 않음** — 같은 87명을 재사용하므로 |

**핵심**: seed를 5개로 늘려도 CI는 좁아지지 않습니다. 87명이라는 한계는 사람을 더 모아야만 풀립니다. seed 확장은 "이 숫자가 partition 운에 좌우되는가"를 답할 뿐입니다.

### ③ 사용 가능한 자원 (v21 시대에는 1/5만 사용했음)

- **seed partition 5개**: `SEED42`, `SEED1234`, `SEED2026`, `SEED271828`, `SEED314159`. 각각 87명 전원을 5-fold로 정확히 한 번씩 덮는 독립 분할입니다 (CV0 기준 seed 간 val donor 겹침 1~5/18). **v21 실험은 전부 SEED42 하나만 썼습니다.**
- **외부 코호트**: `data/ICI_GSE285888_scConcept_512.pt` (26명, R 15 / NR 11). `ICIDataset(state='external')`로 로드되며 `scripts/test.py`가 `--validation-only` 없이 실행될 때 평가됩니다. **v21 실험은 이것도 쓰지 않았습니다.** 유일하게 진짜 독립적인 읽기이므로 CV 결과와 절대 합치지 말 것.

### ④ 보고 형식

`scripts/test.py`는 이제 모든 AUROC에 bootstrap 95% CI를 **자동으로** 붙여 출력합니다 (점추정치만 보고하는 것이 구조적으로 어렵게 만듦):

```
Fold 0 validation: accuracy=0.6092, AUROC=0.5454 [0.422, 0.664], log_loss=0.7921, ...
```

두 run을 비교할 때는 paired bootstrap 승률을 함께 봅니다 (0.5 = 구분 불가):

```bash
python scripts/compare_predictions.py predictions/<run_a>.pt predictions/<run_b>.pt
```

---

## 2. Context 구성 프로토콜 (v22)

**retrieval 없음.** 에피소드의 모든 context bag이 그대로 aggregator에 들어갑니다.

- 합성 데이터: `collate_synthetic_training_episode` / `collate_synthetic_evaluation_episode` — bag 전체 유지, 평가 시 클래스당 최소 1개를 context로 보호하고 나머지 20%(최대 20개)를 query로 사용.
- ICI 실데이터: `EvaluationEpisodeCollator` — fold의 train cohort 전체(~69명)를 context로 붙이고 held-out(17~18명)을 query로 마스킹.

v21의 K=24 retrieval을 제거한 근거는 [`current_status.md`](current_status.md) §4 참고.

---

## 3. 실험 파이프라인

### Stage 1: Medium 합성 사전학습
- **Config**: `configs/train_v22_medium.yaml`
- **실행**:
  ```bash
  CUDA_DEVICES=0 NPROC_PER_NODE=1 \
  TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun \
  NETRC=/NHNHOME/kimds/.netrc \
  scripts/launch_interactive_training.sh v22_medium configs/train_v22_medium.yaml
  ```
- **✅ v22 기준선 (2026-07-29 확정, 버그 수정 반영본)**: `val_ce_loss: 0.5946` @ epoch 13 / 20 epoch / 10,240 steps.
  체크포인트 `checkpoints/20260729_160643/v22_medium_fixed/epoch=013-val_ce_loss=0.5946.ckpt`, 로그 `logs/20260729_160643/v22_medium_fixed.out`.
- **참고 (v21 Phase 1)**: `val_ce_loss: 0.5921` @ 20 epoch / 10,240 steps — v22가 0.0025 이내로 재현했습니다. Phase 1도 full-context였으므로 **retrieval 제거가 사전학습 성능을 훼손하지 않았다는 확인**입니다.

> [!WARNING]
> **`episode_batch_size`를 바꾸면 `max_epochs`도 함께 조정할 것.** v21 Phase 5가 batch를 8→32로 올리면서 epoch 수를 그대로 둬 optimizer step이 10,240 → 2,560으로 줄었고, 그 결과 val_loss가 수렴하지 못한 채(0.606→0.608 평평한 노이즈) 학습이 끝났습니다. `steps = episodes_per_epoch / episode_batch_size * max_epochs`를 항상 확인하세요.

### Stage 2: Hard 합성 사전학습
- **Config**: `configs/train_v22_hard_realworld.yaml`
- **참고 기준선 (v21 Phase 2)**: `val_ce_loss: 0.6845` @ 50 epoch.

### Stage 2.5: 합성 검증 — **여기서 아키텍처를 결정합니다**

§0 전략에 따라 모든 아키텍처 비교는 이 단계에서 끝냅니다.

```bash
python scripts/evaluate_synthetic.py \
  --checkpoint checkpoints/<ts>/v22_medium/last.ckpt \
  --config configs/train_v22_medium.yaml \
  --output predictions/synthetic_<이름>.pt

# 두 후보 비교 (동일 config·seed로 평가해야 episode 구성이 일치)
python scripts/compare_predictions.py \
  predictions/synthetic_<A>.pt predictions/synthetic_<B>.pt
```

출력에는 전체 AUROC + **episode cluster bootstrap CI**, Log Loss, 그리고 5개 response task별 분해가 포함됩니다.

**✅ v22 공식 기준선 (2026-07-30, 1,000 episode)** — 새 후보는 이것과 비교합니다:

```text
AUROC     0.7078  95% CI [0.696, 0.719]   (1,000 episodes / 16,330 queries)
Log loss  0.6209

task             AUROC  CI폭  episodes
combined        0.8201  .039      213
composition     0.7729  .039      204
interaction     0.6628  .049      200
covariance      0.6216  .045      206   <- state와 동률로 최난이도
state           0.6215  .045      177   <-
```

예측 파일: `predictions/synthetic_v22_baseline_1000ep.pt`

> [!IMPORTANT]
> **반드시 `--val-episodes 1000`으로 평가하십시오.** 기본값 104개는 전체 CI 폭 0.074, task당 ~20 episode로 판정에 부적합합니다. 104개로 재던 이전 기준선(0.7466)은 폐기했고, task별 순위도 그때와 달라졌습니다(covariance가 state보다 나쁜 것이 아니라 동률). 상세: [`current_status.md`](current_status.md) §3.

> [!TIP]
> **개선 여지는 `covariance`(0.6216)와 `state`(0.6215)에 몰려 있으며 둘은 동률입니다.** composition/combined는 이미 0.80~0.82라 올릴 여지가 적습니다. 아키텍처 변경은 이 두 축을 겨냥하는 편이 효율적입니다. 단 task별 CI 폭이 0.045 수준이므로, task별로 **+0.05 이상** 차이만 신뢰하세요. 그보다 작은 개선을 노리면 val episode를 2,000개 이상으로 올려야 합니다.

> [!IMPORTANT]
> **CI는 query가 아니라 episode 단위로 계산됩니다.** 한 episode 안의 query들은 같은 context set과 같은 생성 파라미터를 공유하므로 독립이 아닙니다. query 단위로 재표집하면 상관된 예측을 독립으로 취급해 구간이 실제보다 훨씬 좁게 나옵니다(실측 1.5배 과소). 검출력이 부족하면 `val_dataset_kwargs.episodes_per_epoch`를 늘리세요 — **episode 수가 실질적인 표본 크기입니다.**

### Stage 3: ICI 실데이터 — 최종 테스트 (후보 확정 후 1회)

- **Config**: `configs/train_v22_ici_finetune.yaml` (fine-tune), `configs/train_v22_ici_scratch.yaml` (scratch 대조군).
  fold와 seed는 config에 박아두지 않고 **`--cv` / `--seed`로 주입**합니다 (per-fold config 5개를 두던 방식은 폐기). config의 `seed: 42` / `cv: 0`은 아무것도 지정하지 않았을 때의 기본값일 뿐입니다.

- **실행 (전체 sweep: 5 seed × 5 fold = 25 run)**:
  ```bash
  # scratch
  scripts/launch_ici_protocol.sh

  # v22 사전학습 체크포인트에서 미세조정
  PRETRAINED_CKPT=/abs/path/to/v22_pretrain.ckpt scripts/launch_ici_protocol.sh

  # 일부만
  SEEDS="42 1234" scripts/launch_ici_protocol.sh
  ```
  seed 내부의 5 fold는 동시 실행하고, seed끼리는 순차 실행합니다 (GPU 25중 점유 방지). run 목록은 `logs/v22_ici_sweep_manifest.tsv`에 기록됩니다.

- **seed별 평가** (manifest의 체크포인트 경로 사용):
  ```bash
  python scripts/test.py \
    --checkpoints <해당 seed의 fold0..4 last.ckpt 5개> \
    --config configs/train_v22_ici_finetune.yaml \
    --precision bf16-mixed --validation-only \
    --output predictions/v22_ici_seed<SEED>.pt
  ```
  `--validation-only`를 빼면 외부 코호트(GSE285888) 추론까지 수행하고 fold 앙상블 결과를 함께 저장합니다.

- **집계**:
  ```bash
  python scripts/evaluate_protocol.py \
    --predictions predictions/v22_ici_seed*.pt \
    --external predictions/v22_ici_external.pt
  ```

- **참고 기준선 (v21 Phase 6c, 동일한 no-retrieval 구성, SEED42 단일)**: AUROC 0.5454 [0.422, 0.664] / Log Loss 0.7921 / Accuracy 0.6092.
  §0-①에 따라 v22가 이보다 **+0.13 이상** 좋지 않으면 이 코호트에서는 "개선됐다"고 말할 수 없습니다.

---

## 4. 합성 데이터 파라미터

`configs/data/medium.yaml` 기준 (ICI 유사 에피소드 생성):

```text
episodes_per_epoch : 4096
num_bags           : [60, 100]
num_cells          : [500, 1000]
latent_dim         : 32      output_dim : 512
class_separation   : [0.5, 1.4]
donor_shift_scale  : 0.35
rare_response_probability : 0.15    rare_response_fraction : [0.02, 0.08]
observation_noise  : 0.01
response_task_probabilities : composition / state / covariance / interaction / combined
```

ICI 실데이터: 87명 환자, donor당 1,000 세포 샘플링, 512-dim scConcept 임베딩, 5-fold CV (`data/ICI_CVOnly_scConcept_512`).

---

## 5. 검증 스위트

```bash
timeout 1500s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
```

- `tests/test_base_model.py` — aggregator/meta-classifier 계약, `architecture_version == 22`
- `tests/test_model_interface.py` — 손실 항, 체크포인트 버전 게이트 (v21 거부 / v22 수용)
- `tests/test_batched_episode_forward.py` — 4D batched forward가 에피소드별 독립 forward와 일치하는지 (v21 retrieval 스위트 대체)
- `tests/test_evaluation_protocol.py` — AUROC/Log Loss/bootstrap CI 정확성, **"n이 작으면 CI가 넓어진다"는 프로토콜의 핵심 전제** 검증
- `tests/test_ici_dataset.py`, `tests/test_synthetic_variable_cells.py`, `tests/test_scheduler.py`, `tests/test_checkpoint_callback.py`, `tests/test_learnability_ladder.py`

---

## 6. 평가 도구 요약

| 스크립트 | 용도 |
|---|---|
| `scripts/power_analysis.py` | 실험 전: 이 코호트가 검출 가능한 효과 크기 확인 |
| `scripts/launch_ici_protocol.sh` | 5 seed × 5 fold sweep 실행 |
| `scripts/test.py` | 체크포인트 → 예측 파일 (AUROC에 CI 자동 부착) |
| `scripts/evaluate_protocol.py` | 다중 seed 집계 + 외부 코호트 보고 |
| `scripts/evaluate_synthetic.py` | 합성 val 평가 (episode cluster CI + task별 분해) |
| `scripts/compare_predictions.py` | 두 run 비교 (CI + paired bootstrap 승률, episode 있으면 cluster) |
| `src/utils/metrics.py` | 공용 지표 구현 (rank 기반 AUROC, cluster bootstrap) |
