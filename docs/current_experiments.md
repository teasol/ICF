# Current experiments

**Last updated**: `2026-07-29 07:15:00 KST`  
**Architecture Version**: `21` (`architecture_version = 21`)

이 문서는 Architecture v21 평가에 사용되는 합성 데이터 파라미터, 실세계 ICI 데이터셋 24-Donor Retrieval 프로토콜, 그리고 5단계 검증 로드맵의 실행 명령어와 실증 수치를 시분초 단위로 명시적으로 설명합니다.

---

## 1. Class-Balanced 24-Donor Retrieval Protocol (ICI Real Data)

실제 면역관문억제제(ICI) 87명 환자 단일세포 데이터셋 평가 시 69명 다수 Context의 배치 효과(donor_shift_scale: 0.70) 오염을 막기 위해 **설계상으로는 Query 1명당 24명 맞춤형 선별 알고리즘**을 수행하기로 했음:

> [!WARNING]
> **실제 구현은 이 "Query 1명당 맞춤형" 설계를 지키지 않음** (2026-07-29 확인). 아래 ①②는 모두 **한 번의 호출에 들어온 17~18명 query 전원에게 동일한 공용 context 24명**을 적용함 — ①은 첫 번째 query만 기준으로, ②는 전체 query를 평균내어 선별. 측정 결과 이 공용 context는 각 query가 개별 선별했을 top-24와 평균 61.1%만 겹치며, 특히 반응자(y=1)에서 ~50%로 더 어긋남. 상세 및 대안 구현(`retrieve_context_indices_per_query`)은 [`current_status.md`](current_status.md) §4-⑧ 가설2 참고.

### ① Naive Retrieval Protocol (`src/modules/data_interface.py`)
- 1,000개 세포의 단순 평균(Mean) 및 표준편차(Spread) 기반 1024차원 Cosine Similarity 사용.
- $Y=0$ (NR) 상위 12명 + $Y=1$ (R) 상위 12명 선별 ($K=24$).
- ⚠ 기준 query: `evaluation_x[0]` — **첫 번째 query 1명**만 사용 ([`data_interface.py:85`](../src/modules/data_interface.py#L85)).

### ② Model-Level 40-token Signal-Aware Retrieval Protocol (`src/models/baseline.py`)
- 모델 내부 Aggregator가 분류기 입력용으로 이미 계산해 두는 40-token 구조화 요약(`_all_structured_tokens`: 1 global + 12 slots×3종 + 3 tail, 각 512-dim)을 재사용, flatten 후 Cosine Similarity 계산.
- 배경 노이즈가 아닌 세포 반응 신호(density slot, top-1%/5%/15% rare evidence, covariance sketch 정보가 이미 인코딩된 표현) 기준 Class-Balanced Top-24 Context 선별.
- ⚠ 기준 query: `bag_features[query_index].mean(dim=0)` — **전체 query의 평균** ([`baseline.py:3440`](../src/models/baseline.py#L3440)).

---

## 2. 5단계 파이프라인 실증 성과 및 실행 프로토콜

### Phase 1: v21 Medium Synthetic Problem (20 Epochs)
- **Config**: [`configs/train_v21_medium.yaml`](file:///NHNHOME/kimds/ICF/configs/train_v21_medium.yaml)
- **실행 명령어**:
  ```bash
  CUDA_DEVICES=0 NPROC_PER_NODE=1 TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun NETRC=/NHNHOME/kimds/.netrc scripts/launch_interactive_training.sh v21_medium configs/train_v21_medium.yaml
  ```
- **완료 성과**: **`val_ce_loss: 0.5921`** (Ep 18/20 수렴)
- **체크포인트**: `checkpoints/20260727_141002/v21_medium/epoch=018-val_ce_loss=0.5921.ckpt`
- **로그 파일**: `logs/20260727_141002/v21_medium.out`

---

### Phase 1-R: v21 Medium Pretrain (Naive Retrieval K=24)
- **Config**: [`configs/train_v21_medium_retrieved.yaml`](file:///NHNHOME/kimds/ICF/configs/train_v21_medium_retrieved.yaml)
- **실행 결과**: `val_ce_loss: 0.6839` (Ep 9/20 수렴 정체 - Naive Retrieval의 95% 배경 노이즈 주입이 원인으로 규명됨)
- **체크포인트**: `checkpoints/20260727_234145/v21_medium_retrieved/epoch=009-val_ce_loss=0.6839.ckpt`
- **로그 파일**: `logs/20260727_234145/v21_medium_retrieved.out`

---

### Phase 2: Stage 2 Hard Real-World Synthetic Problem (50 Epochs)
- **Config**: [`configs/train_v21_hard.yaml`](file:///NHNHOME/kimds/ICF/configs/train_v21_hard.yaml)
- **주요 파라미터**: $D=512$, Observation Noise `0.05`, Donor Shift `0.70`, Sub-1% Rare Cell (`0.005`~`0.03`)
- **실행 명령어**:
  ```bash
  CUDA_DEVICES=0 NPROC_PER_NODE=1 TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun NETRC=/NHNHOME/kimds/.netrc scripts/launch_interactive_training.sh v21_hard configs/train_v21_hard.yaml
  ```
- **완료 성과**: **`val_ce_loss: 0.6845`** (Ep 44/50 수렴)
- **체크포인트**: `checkpoints/20260727_150034/v21_hard/epoch=044-val_ce_loss=0.6845.ckpt`
- **로그 파일**: `logs/20260727_150034/v21_hard.out`

---

### Phase 2-R: v21 Hard Pretrain (Naive Retrieval K=24)
- **Config**: [`configs/train_v21_hard_retrieved.yaml`](file:///NHNHOME/kimds/ICF/configs/train_v21_hard_retrieved.yaml)
- **실행 결과**: `val_ce_loss: 0.6803` (Ep 12/50)
- **체크포인트**: `checkpoints/20260728_003034/v21_hard_retrieved/epoch=012-val_ce_loss=0.6803.ckpt`
- **로그 파일**: `logs/20260728_003034/v21_hard_retrieved.out`

---

### Phase 3-A & 3-B: ICI Fold 0 Scratch vs Fine-Tuning
- **Phase 3-A (Scratch)**: `configs/train_v21_ici_scratch_fold0.yaml` $\to$ AUROC: **0.5665**, Log Loss: **0.8236** (`checkpoints/20260727_201907/v21_ici_scratch_f0/last.ckpt`)
- **Phase 3-B (Fine-Tune)**: `configs/train_v21_ici_finetune_fold0.yaml` $\to$ AUROC: **0.5654**, Log Loss: **0.8232** (`checkpoints/20260727_201910/v21_ici_finetune_f0/last.ckpt`)

---

### Phase 4: Full ICI 5-Fold Cross-Validation (K=24 Naive Retrieval)
- **실행 성과**: AUROC: **0.5524**, **Log Loss: 0.7288 (0.0944 대폭 하강 달성)**
- **체크포인트 경로**: `checkpoints/20260728_013253/`
- **로그 파일 경로**: `logs/20260728_013253~/`

---

### Phase 5: Signal-Aware 40-token Feature Retrieval Pretraining (20 Epochs, 완료)
- **목표**: Naive Retrieval의 배경 노이즈 편향을 극복하기 위해 모델 내부 40-token Signal-Aware Retrieval 및 대형 candidate pool ($N=60 \sim 100$) 적용 사전학습.
- **Config**: [`configs/train_v21_large_context_pretrain.yaml`](file:///NHNHOME/kimds/ICF/configs/train_v21_large_context_pretrain.yaml) (`base_config: train_v21_medium.yaml`, `episode_batch_size: 32`, `retrieval_k: 24`, `dataset_kwargs.num_bags: [60, 100]`, `num_cells: [500, 1000]`)
- **실행 명령어**:
  ```bash
  CUDA_DEVICES=0 NPROC_PER_NODE=1 TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun NETRC=/NHNHOME/kimds/.netrc scripts/launch_interactive_training.sh v21_large_context_pretrain configs/train_v21_large_context_pretrain.yaml
  ```
- **완료 성과**: **Best `val_ce_loss: 0.5940`** (epoch 14) — Phase 1 Full-Context(`0.5921`) 대비 0.0019 이내로 근접, Phase 1-R Naive Retrieval(`0.6839`) 대비 0.09 이상 개선. 20 epoch 전체 완주.
- **체크포인트**: `checkpoints/20260728_144957/v21_large_context_pretrain/epoch=014-val_ce_loss=0.5940.ckpt`
- **로그 파일**: `logs/20260728_144957/v21_large_context_pretrain.out`
- **W&B**: https://wandb.ai/teasol/ICF/runs/9ldg44nr
- **검증 단원**: `tests/test_feature_retrieval.py`, `tests/test_large_context_pretrain.py`
- **주요 이슈 진단 및 조치**: 6번째 재시도 만에 성공 (config 오류, DataLoader CUDA fork 충돌, pin_memory 충돌, retrieval gradient 누수, `training_step`이 retrieval을 호출하지 않던 근본 문제 등 5건). 상세 경위는 [`current_status.md`](current_status.md) §4-④~⑤ 참고.

---

### Phase 6: ICI 5-Fold CV Fine-Tuning from Phase 5 Signal-Aware Pretrain (완료)
- **목표**: Phase 4(Naive Retrieval 사전학습 기반, Log Loss `0.7288`)와 동일한 ICI 5-Fold 미세조정 프로토콜을 Phase 5 체크포인트에 적용하여, Signal-Aware Retrieval 사전학습이 실데이터 성능을 추가로 개선하는지 확인.
- **Pretrained Checkpoint**: `checkpoints/20260728_144957/v21_large_context_pretrain/epoch=014-val_ce_loss=0.5940.ckpt`
- **실행 스크립트**: `scripts/launch_phase6_5fold.sh` (`launch_phase4_5fold.sh`와 동일 패턴, `PRETRAINED_CKPT`만 Phase 5 체크포인트로 교체)
- **실행 명령어**:
  ```bash
  scripts/launch_phase6_5fold.sh
  ```
- **주의**: `--ckpt-path`는 Lightning `trainer.fit(ckpt_path=...)` 풀 리쥼(전체 옵티마이저/스케줄러/`current_epoch` 상태 복원)이므로, 각 fold는 Phase 5 체크포인트의 epoch(14)부터 이어서 fold config의 `max_epochs: 50`까지(즉 36 epoch만) 학습함 — Phase 4도 동일한 방식(Phase 2-R epoch 12부터 이어서 38 epoch)으로 수행되었으므로 비교 가능.
- **Run Kind**: `v21_ici_finetune_phase6_f{0..4}`
- **시작**: 2026-07-28 17:57:10 KST, 5-fold 모두 정상 기동 및 50 epoch 완주 (에러 없음)
- **로그 파일**: `logs/20260728_1757{10,12,14,16,18}/v21_ici_finetune_phase6_f{0..4}.out`
- **평가 명령어** (Phase 4와 동일 프로토콜, `scripts/test.py`):
  ```bash
  python scripts/test.py \
    --checkpoints \
      checkpoints/20260728_175710/v21_ici_finetune_phase6_f0/last.ckpt \
      checkpoints/20260728_175712/v21_ici_finetune_phase6_f1/last.ckpt \
      checkpoints/20260728_175714/v21_ici_finetune_phase6_f2/last.ckpt \
      checkpoints/20260728_175716/v21_ici_finetune_phase6_f3/last.ckpt \
      checkpoints/20260728_175718/v21_ici_finetune_phase6_f4/last.ckpt \
    --config configs/train_v21_ici_finetune_fold0.yaml \
    --precision bf16-mixed --retrieval-k 24 --validation-only \
    --output predictions/ici_predictions_v21_phase6_5fold.pt
  ```
- **결과 (가설과 반대)**: **AUROC: 0.5081** (Phase 4 `0.5524` 대비 악화), **Log Loss: 0.9596** (Phase 4 `0.7288` 대비 대폭 악화), `p1_std: 0.2874` (Phase 4 `0.1664` 대비 과신 심화).
- **원인 가설**: 사전학습(모델 내부 Signal-Aware retrieval)과 미세조정(fine-tune config가 물려받은 외부 Naive Retrieval collator, `data.retrieval_k: 24`)의 context 선별 분포 불일치. 상세는 [`current_status.md`](current_status.md) §4-⑥ 참고.
- **예측 결과 파일**: `predictions/ici_predictions_v21_phase6_5fold.pt`

---

### Phase 6b: ICI 5-Fold CV Fine-Tune, 모델 내부 Signal-Aware Retrieval로 통일
- **목표**: Phase 6의 원인 가설(사전학습-미세조정 context 선별 방식 불일치) 검증. 미세조정 시에도 사전학습과 동일하게 모델 내부 `retrieve_context_indices`를 쓰도록 배선.
- **코드 변경**: `src/modules/model_interface.py`의 `_episode_losses`/`_evaluation_step`/`predict_step` 세 곳에 `retrieval_k=self.hparams.get("retrieval_k", 0)` 전달 추가 (기존에는 4D 배치 경로에만 있고 3D/non-episode 경로엔 없었음).
- **Config**: `configs/train_v21_ici_finetune_signalaware_fold{0..4}.yaml` (신규, 기존 finetune fold config를 보존하기 위해 복사 후 `data.retrieval_k` 삭제 + `model_kwargs.retrieval_k: 24` 추가)
- **실행 스크립트**: `scripts/launch_phase6b_5fold.sh`
- **평가 명령어**: `scripts/test.py`에 `--retrieval-k` 플래그를 주지 않음 (외부 Naive Retrieval을 다시 켜지 않기 위함 — 모델 내부 retrieval이 `model_kwargs.retrieval_k`를 통해 자동 적용됨)
  ```bash
  python scripts/test.py \
    --checkpoints \
      checkpoints/20260728_205237/v21_ici_finetune_phase6b_f0/last.ckpt \
      checkpoints/20260728_205239/v21_ici_finetune_phase6b_f1/last.ckpt \
      checkpoints/20260728_205241/v21_ici_finetune_phase6b_f2/last.ckpt \
      checkpoints/20260728_205243/v21_ici_finetune_phase6b_f3/last.ckpt \
      checkpoints/20260728_205245/v21_ici_finetune_phase6b_f4/last.ckpt \
    --config configs/train_v21_ici_finetune_signalaware_fold0.yaml \
    --precision bf16-mixed --validation-only \
    --output predictions/ici_predictions_v21_phase6b_5fold.pt
  ```
- **결과 (Phase 6 대비 개선, Phase 4에는 여전히 미달)**: **AUROC: 0.5481** (Phase 6 `0.5081`→`0.5481`), **Log Loss: 0.8672** (Phase 6 `0.9596`→`0.8672`), Accuracy: 0.5747 (Phase 4 `0.5287`보다도 높음), `p1_std: 0.2545`.
- **결론 (⚠ Phase 6c/통계 검정에서 뒤집힘)**: 당시엔 "Phase 4가 최선"으로 기록했으나, 아래 Phase 6c 및 [`current_status.md`](current_status.md) §4-⑧의 bootstrap 검정 결과 Phase 4/6b/6c 차이는 통계적으로 구분 불가능함.
- **예측 결과 파일**: `predictions/ici_predictions_v21_phase6b_5fold.pt`

---

### Phase 6c: ICI 5-Fold CV Fine-Tune, Retrieval 완전 비활성 (Full Context 대조군)
- **목표**: "retrieval 자체가 ICI에서 이득이 있는가"를 직접 검증. Phase 5 체크포인트를 retrieval 없이(전체 ~69명 context 그대로) 미세조정.
- **Config**: `configs/train_v21_ici_finetune_fullcontext_fold{0..4}.yaml` (신규, `data.retrieval_k` 제거만 하여 `EvaluationEpisodeCollator`로 폴백)
- **실행 스크립트**: `scripts/launch_phase6c_5fold.sh`
- **평가 명령어**:
  ```bash
  python scripts/test.py \
    --checkpoints \
      checkpoints/20260729_062833/v21_ici_finetune_phase6c_f0/last.ckpt \
      checkpoints/20260729_062835/v21_ici_finetune_phase6c_f1/last.ckpt \
      checkpoints/20260729_062837/v21_ici_finetune_phase6c_f2/last.ckpt \
      checkpoints/20260729_062839/v21_ici_finetune_phase6c_f3/last.ckpt \
      checkpoints/20260729_062841/v21_ici_finetune_phase6c_f4/last.ckpt \
    --config configs/train_v21_ici_finetune_fullcontext_fold0.yaml \
    --precision bf16-mixed --validation-only \
    --output predictions/ici_predictions_v21_phase6c_5fold.pt
  ```
- **결과**: **AUROC: 0.5454**, **Log Loss: 0.7921**, **Accuracy: 0.6092**, `p1_std: 0.2337`.
- **핵심 결론**: retrieval을 켠 Phase 6b(AUROC 0.5481 / LL 0.8672 / Acc 0.5747)와 비교하면 **AUROC는 사실상 동일한데 Log Loss와 Accuracy는 retrieval을 껐을 때가 더 좋음**. ICI 규모(fold당 context ~69명)에서 24명 retrieval은 가용 labeled context의 65%를 버리면서 얻는 것이 없음.
- **예측 결과 파일**: `predictions/ici_predictions_v21_phase6c_5fold.pt`

---

### 통계적 유의성 검정 (2026-07-29): 위 비교들이 검출력이 있는가?

n=87 (positive 37) 단일 코호트에서 5,000회 bootstrap:

| 실험 | AUROC | 95% CI |
|---|---:|---|
| Phase 4 (Naive 사전학습 + Naive retrieval) | 0.5524 | [0.424, 0.677] |
| Phase 6 (SA 사전학습 + Naive retrieval, 불일치) | 0.5081 | [0.383, 0.635] |
| Phase 6b (SA 사전학습 + SA retrieval) | 0.5481 | [0.421, 0.674] |
| Phase 6c (SA 사전학습 + retrieval 없음) | 0.5454 | [0.419, 0.674] |

- **모든 CI가 0.5(무작위)를 포함**하고 서로 거의 완전히 겹침.
- Paired bootstrap 승률: Phase 4 vs 6b = 0.53, Phase 4 vs 6c = 0.55 (동전 던지기). 유일하게 일관된 차이는 Phase 6(불일치)이 나쁘다는 것(0.78~0.81).
- **함의**: 현재 평가 세팅으로는 아키텍처 변경의 효과를 검증할 수 없음. 반복 seed·외부 코호트·CI 리포팅 도입이 선행되어야 함. 상세는 [`current_status.md`](current_status.md) §4-⑧ 참고.
