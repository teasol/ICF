# Current experiments

**Last updated**: `2026-07-29 08:30:00 KST`
**Architecture Version**: `22` (`architecture_version = 22`)

이 문서는 v22 기준 실험 프로토콜과 실행 명령어를 설명합니다. v21 retrieval 시대의 실험 기록은 [`history/v21_retrieval_experiments.md`](history/v21_retrieval_experiments.md)로 이관되었습니다.

> [!CAUTION]
> **v22는 기존 체크포인트를 전부 무효화합니다.** retrieval 계층 제거로 `architecture_version`이 22로 올라갔고, `ModelInterface.on_load_checkpoint`가 v21 이하 체크포인트를 거부합니다. 아래 모든 실험은 **처음부터 다시** 돌려야 합니다.

---

## 0. 평가 프로토콜 (실험 시작 전 반드시 읽을 것)

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

## 1. Context 구성 프로토콜 (v22)

**retrieval 없음.** 에피소드의 모든 context bag이 그대로 aggregator에 들어갑니다.

- 합성 데이터: `collate_synthetic_training_episode` / `collate_synthetic_evaluation_episode` — bag 전체 유지, 평가 시 클래스당 최소 1개를 context로 보호하고 나머지 20%(최대 20개)를 query로 사용.
- ICI 실데이터: `EvaluationEpisodeCollator` — fold의 train cohort 전체(~69명)를 context로 붙이고 held-out(17~18명)을 query로 마스킹.

v21의 K=24 retrieval을 제거한 근거는 [`current_status.md`](current_status.md) §4 참고.

---

## 2. 실험 파이프라인

### Stage 1: Medium 합성 사전학습
- **Config**: `configs/train_v22_medium.yaml`
- **실행**:
  ```bash
  CUDA_DEVICES=0 NPROC_PER_NODE=1 \
  TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun \
  NETRC=/NHNHOME/kimds/.netrc \
  scripts/launch_interactive_training.sh v22_medium configs/train_v22_medium.yaml
  ```
- **참고 기준선 (v21 Phase 1)**: `val_ce_loss: 0.5921` @ 20 epoch / 10,240 optimizer steps.

> [!WARNING]
> **`episode_batch_size`를 바꾸면 `max_epochs`도 함께 조정할 것.** v21 Phase 5가 batch를 8→32로 올리면서 epoch 수를 그대로 둬 optimizer step이 10,240 → 2,560으로 줄었고, 그 결과 val_loss가 수렴하지 못한 채(0.606→0.608 평평한 노이즈) 학습이 끝났습니다. `steps = episodes_per_epoch / episode_batch_size * max_epochs`를 항상 확인하세요.

### Stage 2: Hard 합성 사전학습
- **Config**: `configs/train_v22_hard_realworld.yaml`
- **참고 기준선 (v21 Phase 2)**: `val_ce_loss: 0.6845` @ 50 epoch.

### Stage 3: ICI 실데이터 — 다중 seed 5-Fold CV + 외부 코호트

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

## 3. 합성 데이터 파라미터

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

## 4. 검증 스위트

```bash
timeout 1500s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
```

- `tests/test_base_model.py` — aggregator/meta-classifier 계약, `architecture_version == 22`
- `tests/test_model_interface.py` — 손실 항, 체크포인트 버전 게이트 (v21 거부 / v22 수용)
- `tests/test_batched_episode_forward.py` — 4D batched forward가 에피소드별 독립 forward와 일치하는지 (v21 retrieval 스위트 대체)
- `tests/test_evaluation_protocol.py` — AUROC/Log Loss/bootstrap CI 정확성, **"n이 작으면 CI가 넓어진다"는 프로토콜의 핵심 전제** 검증
- `tests/test_ici_dataset.py`, `tests/test_synthetic_variable_cells.py`, `tests/test_scheduler.py`, `tests/test_checkpoint_callback.py`, `tests/test_learnability_ladder.py`

---

## 5. 평가 도구 요약

| 스크립트 | 용도 |
|---|---|
| `scripts/power_analysis.py` | 실험 전: 이 코호트가 검출 가능한 효과 크기 확인 |
| `scripts/launch_ici_protocol.sh` | 5 seed × 5 fold sweep 실행 |
| `scripts/test.py` | 체크포인트 → 예측 파일 (AUROC에 CI 자동 부착) |
| `scripts/evaluate_protocol.py` | 다중 seed 집계 + 외부 코호트 보고 |
| `scripts/compare_predictions.py` | 두 run 비교 (CI + paired bootstrap 승률) |
