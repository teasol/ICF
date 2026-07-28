# Current development status & multi-location sync SSOT

**Last updated**: `2026-07-28 11:20:00 KST`  
**Latest Commit**: `67b75d8` (`feat(config): scale episode_batch_size to E=32 based on GPU throughput benchmark`)  
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)  
**Architecture Version**: `21` (`architecture_version = 21`)  
**Purpose**: 연구실 / 집 / 노트북 3개 작업 환경 간 대화 기록 비동기화 문제를 완벽 해결하기 위한 Single Source of Truth (SSOT) living document.

---

## 1. 멀티 작업공간 (연구실/집/노트북) 바톤 터치 지침

본 문서는 대화 세션이 분리된 환경(연구실 Desktop, 집 PC, 개인 노트북)에서 새로 접속한 AI Coding Agent가 이전 세션의 실험 수치, 핵심 논의, 실행 경로, 미결 과제를 100% 동일한 맥락으로 이어받을 수 있도록 작성되었습니다.

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 원칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 새로 접속한 Agent는 **`docs/` 최상위 루트의 Living md 파일 5개(`agent_handoff.md`, `current_status.md`, `current_architecture.md`, `current_experiments.md`, `README.md`)만 최우선으로 정독**하여 전체 개발 맥락과 프로젝트 규칙을 파악합니다.
> 3. 터미널 조회가 필요한 명령어는 NVML/쉘 hang 방지를 위해 **반드시 `timeout 3s ps aux | grep python`과 같이 타임아웃**을 적용합니다.
> 4. 코드 변경 시 unittest 통과 필수:
>    `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"`

---

## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture v21)

* **Python Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python`
* **Torchrun Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun`
* **Target Hardware**: NVIDIA B200 GPU 1장 (`CUDA_VISIBLE_DEVICES=0`, 180GB VRAM)
* **Precision Policy**: `bf16-mixed` (FP16 공분산 역행렬 연산 시 Exponent Overflow/NaN 문제 100% 완전 해결)
* **Architecture v21 4대 수학적 핵심 기술**:
  1. **Z-Score Bag Studentization**: Donor Centroid ($\mu_i$) & Standard Deviation ($S_i$) 기반 정규화로 세포 표현 스케일 정규화.
  2. **Top-1% Sparse Evidence Module**: 97%+ 배경세포에 의해 희석되는 Sub-1% (0.5%~3%) 희귀 세포 반응 신호 핀포인트 추출.
  3. **Covariance Subspace Shrinkage**: Shrinkage parameter `0.25`로 노이즈 축 whitening 방어 및 NaN 예방.
  4. **Auxiliary Pairwise Ranking Loss (`weight: 0.10`)**: Cross-Entropy 0.685 부근 Gradient 소멸 및 Local Minima 탈출.
* **Model-Level Signal-Aware 40-dim Feature Retrieval**:
  - `extract_bag_features(...)` 및 `retrieve_context_indices(...)`로 모델 내부 Aggregator 40차원 특징 표현 기반 Class-Balanced Top-24 ($K=24$) Context 동적 선별 구현 완료.

---

## 3. 실험 파이프라인 진행 현황 및 상세 실증 수치

| Phase | 실험 명칭 | Config 경로 | Epoch | 상태 / 수치 성과 | 최적 체크포인트 경로 | Log 파일 경로 |
|---|---|---|---:|---|---|---|
| **Phase 1** | v21 Medium Pretrain (Full Context) | `configs/train_v21_medium.yaml` | 20e | **수렴 완료**<br/>`val_ce_loss: 0.5921` | `checkpoints/20260727_141002/v21_medium/epoch=018-val_ce_loss=0.5921.ckpt` | `logs/20260727_141002/v21_medium.out` |
| **Phase 1-R**| v21 Medium Pretrain (Naive Retrieval K=24) | `configs/train_v21_medium_retrieved.yaml` | 20e | **수렴 정체**<br/>`val_ce_loss: 0.6839` | `checkpoints/20260727_234145/v21_medium_retrieved/epoch=009-val_ce_loss=0.6839.ckpt` | `logs/20260727_234145/v21_medium_retrieved.out` |
| **Phase 2** | v21 Hard Pretrain (Full Context) | `configs/train_v21_hard_realworld.yaml` | 50e | **수렴 완료**<br/>`val_ce_loss: 0.6845` | `checkpoints/20260727_150034/v21_hard/epoch=044-val_ce_loss=0.6845.ckpt` | `logs/20260727_150034/v21_hard.out` |
| **Phase 2-R**| v21 Hard Pretrain (Naive Retrieval K=24) | `configs/train_v21_hard_retrieved.yaml` | 50e | **수렴 정체**<br/>`val_ce_loss: 0.6803` | `checkpoints/20260728_003034/v21_hard_retrieved/epoch=012-val_ce_loss=0.6803.ckpt` | `logs/20260728_003034/v21_hard_retrieved.out` |
| **Phase 3-A**| ICI Fold 0 Scratch | `configs/train_v21_ici_scratch_fold0.yaml` | 50e | AUROC: 0.5665<br/>Log Loss: 0.8236 | `checkpoints/20260727_201907/v21_ici_scratch_f0/last.ckpt` | `logs/20260727_201907/v21_ici_scratch_f0.out` |
| **Phase 3-B**| ICI Fold 0 Fine-Tune | `configs/train_v21_ici_finetune_fold0.yaml` | 50e | AUROC: 0.5654<br/>Log Loss: 0.8232 | `checkpoints/20260727_201910/v21_ici_finetune_f0/last.ckpt` | `logs/20260727_201910/v21_ici_finetune_f0.out` |
| **Phase 4** | ICI 5-Fold CV (Retrieval K=24) | Fold 0~4 CV | 50e | AUROC: 0.5524<br/>**Log Loss: 0.7288 (0.0944 대폭 하강)** | `checkpoints/20260728_013253/` | `logs/20260728_013253~/` |
| **Phase 5** | Signal-Aware 40-dim Retrieval Pretraining | Model Retrieval | 20e/50e | **검증 완료** (`tests/test_feature_retrieval.py` 통과) | 구동 예정 | 로그 준비 중 |

---

## 4. 핵심 집중 이슈: Pretraining Stage Context Expansion & Signal-Aware Retrieval

### ① 문제 진단 (Why Naive Retrieval Failed in Pretraining)
* Pretraining 단계에서 Context set을 효율적으로 구성하기 위해 주입했던 기존 Naive Retrieval(`RetrievalEvaluationEpisodeCollator`)은 세포 1,000개의 단순 평균 및 표준편차 Cosine Similarity를 사용함.
* Single-cell 면역 데이터는 95%+가 공통 배경 세포이고 반응 신호는 <1%~5% 희귀 세포에 쏠려 있어, Naive Retrieval이 반응 유사 donor가 아닌 **95% 배경 노이즈가 유사한 donor**를 추출함.
* 이로 인해 사전학습 시 모델에 노이즈 donor가 주어지면서 `val_ce_loss`가 `0.5921`에서 `0.6839`로 상향 튐.

### ② 해결 설계: 40-dim Bag Feature Signal-Aware Retrieval
* Aggregator 내부에서 추출하는 **40차원 특징 표현 ($Z \in \mathbb{R}^{40}$)**:
  1. 12-dim Density Slots (세포 밀도 및 sub-population 분포)
  2. Top-1% / 5% / 15% Rare Evidence Tail Features
  3. Subspace Covariance Sketch Features
  4. Centered-Spread Scale Features
* 모델 내부 Aggregator가 추출한 40차원 특징 $Z$의 Cosine Similarity 기반으로 Class-Balanced Top-24 Context Donor를 동적 추출하도록 모델 레이어 통합 설계 (`extract_bag_features`, `retrieve_context_indices`).

---

## 5. 다음 작업 세션 Action Plan (Next Steps for Any Agent Location)

연구실, 집, 또는 노트북 어디서 접속하더라도 다음 순서로 작업을 수행하면 됩니다:

1. **단위 테스트 실행 및 검증 완료 확인**:
   - `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"` (All tests PASS, `test_feature_retrieval.py` 포함)
2. **Phase 5 Large Context + 40-dim Signal-Aware Retrieval Pretraining 구동**:
   - `scripts/launch_interactive_training.sh` 사용 구동 및 `logs/` 점검 후 본 문서(`current_status.md`) 업데이트.
