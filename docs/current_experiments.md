# Current experiments

**Last updated**: `2026-07-28 16:05:00 KST`  
**Architecture Version**: `21` (`architecture_version = 21`)

이 문서는 Architecture v21 평가에 사용되는 합성 데이터 파라미터, 실세계 ICI 데이터셋 24-Donor Retrieval 프로토콜, 그리고 5단계 검증 로드맵의 실행 명령어와 실증 수치를 시분초 단위로 명시적으로 설명합니다.

---

## 1. Class-Balanced 24-Donor Retrieval Protocol (ICI Real Data)

실제 면역관문억제제(ICI) 87명 환자 단일세포 데이터셋 평가 시 69명 다수 Context의 배치 효과(donor_shift_scale: 0.70) 오염을 막기 위해 **Query 1명당 24명 맞춤형 선별 알고리즘**을 수행함:

### ① Naive Retrieval Protocol (`src/modules/data_interface.py`)
- 1,000개 세포의 단순 평균(Mean) 및 표준편차(Spread) 기반 1024차원 Cosine Similarity 사용.
- $Y=0$ (NR) 상위 12명 + $Y=1$ (R) 상위 12명 선별 ($K=24$).

### ② Model-Level 40-token Signal-Aware Retrieval Protocol (`src/models/baseline.py`)
- 모델 내부 Aggregator가 분류기 입력용으로 이미 계산해 두는 40-token 구조화 요약(`_all_structured_tokens`: 1 global + 12 slots×3종 + 3 tail, 각 512-dim)을 재사용, flatten 후 Cosine Similarity 계산.
- 배경 노이즈가 아닌 세포 반응 신호(density slot, top-1%/5%/15% rare evidence, covariance sketch 정보가 이미 인코딩된 표현) 기준 Class-Balanced Top-24 Context 선별.

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
- **다음 단계**: 이 체크포인트로 ICI 5-Fold 미세조정(Phase 3/4 방식) 수행하여 실데이터 성능 개선 여부 확인 예정.
