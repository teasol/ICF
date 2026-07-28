# Current experiments

**Last updated**: `2026-07-29 08:30:00 KST`
**Architecture Version**: `22` (`architecture_version = 22`)

이 문서는 v22 기준 실험 프로토콜과 실행 명령어를 설명합니다. v21 retrieval 시대의 실험 기록은 [`history/v21_retrieval_experiments.md`](history/v21_retrieval_experiments.md)로 이관되었습니다.

> [!CAUTION]
> **v22는 기존 체크포인트를 전부 무효화합니다.** retrieval 계층 제거로 `architecture_version`이 22로 올라갔고, `ModelInterface.on_load_checkpoint`가 v21 이하 체크포인트를 거부합니다. 아래 모든 실험은 **처음부터 다시** 돌려야 합니다.

---

## 0. 결과를 보고하기 전에 반드시 지킬 것

ICI 코호트는 n=87 (positive 37)입니다. 이 규모에서 AUROC의 95% 신뢰구간 폭은 **약 ±0.13**이며, v21 시대에 비교하던 0.004~0.04 수준의 차이는 **전부 노이즈였습니다** (paired bootstrap 승률 0.52~0.55). 상세: [`history/v21_retrieval_investigation.md`](history/v21_retrieval_investigation.md) §4-⑧.

따라서 **점추정치만 보고하지 말고 반드시 CI를 함께 보고**합니다:

```bash
python scripts/compare_predictions.py \
  predictions/<run_a>.pt \
  predictions/<run_b>.pt
```

출력: 각 run의 AUROC + bootstrap 95% CI, 그리고 pair별 승률(0.5 = 구분 불가).

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

### Stage 3: ICI 실데이터 5-Fold CV
- **Config**: `configs/train_v22_ici_finetune_fold{0..4}.yaml` (fine-tune), `configs/train_v22_ici_scratch_fold0.yaml` (scratch 대조군)
- **실행**:
  ```bash
  # 사전학습 체크포인트에서 미세조정
  PRETRAINED_CKPT=/abs/path/to/v22_pretrain.ckpt scripts/launch_ici_5fold.sh

  # 또는 scratch부터
  scripts/launch_ici_5fold.sh
  ```
- **평가**:
  ```bash
  python scripts/test.py \
    --checkpoints \
      checkpoints/<ts0>/v22_ici_finetune_f0/last.ckpt \
      checkpoints/<ts1>/v22_ici_finetune_f1/last.ckpt \
      checkpoints/<ts2>/v22_ici_finetune_f2/last.ckpt \
      checkpoints/<ts3>/v22_ici_finetune_f3/last.ckpt \
      checkpoints/<ts4>/v22_ici_finetune_f4/last.ckpt \
    --config configs/train_v22_ici_finetune_fold0.yaml \
    --precision bf16-mixed --validation-only \
    --output predictions/ici_predictions_v22_5fold.pt
  ```
- **참고 기준선 (v21 Phase 6c, 동일한 no-retrieval 구성)**: AUROC 0.5454 / Log Loss 0.7921 / Accuracy 0.6092.

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
- `tests/test_ici_dataset.py`, `tests/test_synthetic_variable_cells.py`, `tests/test_scheduler.py`, `tests/test_checkpoint_callback.py`, `tests/test_learnability_ladder.py`
