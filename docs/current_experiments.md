# Current experiments

**Last updated**: `2026-07-28 10:30:00 KST`  
**Architecture Version**: `21` (`architecture_version = 21`)

이 문서는 Architecture v21 평가에 사용되는 합성 데이터 파라미터, 실세계 ICI 데이터셋 24-Donor Retrieval 프로토콜, 그리고 5단계 검증 로드맵의 실행 명령어와 실증 수치를 시분초 단위로 명시적으로 설명합니다.

---

## 1. Class-Balanced 24-Donor Retrieval Protocol (ICI Real Data)

실제 면역관문억제제(ICI) 87명 환자 단일세포 데이터셋 평가 시 69명 다수 Context의 배치 효과(donor_shift_scale: 0.70) 오염을 막기 위해 **Query 1명당 24명 맞춤형 선별 알고리즘**을 수행함:

### ① Naive Retrieval Protocol (`src/modules/data_interface.py`)
- 1,000개 세포의 단순 평균(Mean) 및 표준편차(Spread) 기반 1024차원 Cosine Similarity 사용.
- $Y=0$ (NR) 상위 12명 + $Y=1$ (R) 상위 12명 선별 ($K=24$).

### ② Model-Level 40-dim Signal-Aware Retrieval Protocol (`src/models/baseline.py`)
- 모델 내부 Aggregator의 40차원 특징 표현 ($Z \in \mathbb{R}^{40}$) 기반 Cosine Similarity 계산.
- 배경 노이즈가 아닌 세포 반응 신호(Top-1% rare evidence + density slots + covariance sketch) 기준 Class-Balanced Top-24 Context 선별.

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

### Phase 5: Signal-Aware 40-dim Feature Retrieval Pretraining & Fine-tuning
- **목표**: Naive Retrieval의 배경 노이즈 편향을 극복하기 위해 모델 내부 40-dim Feature Signal-Aware Retrieval 및 2-Pass Streaming ($N=60 \sim 100+$) 적용 사전학습 및 5-Fold 미세조정 구동.
- **검증 단원**: `tests/test_feature_retrieval.py`
