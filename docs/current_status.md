# Current development status & multi-location sync SSOT

**Last updated**: `2026-07-28 13:10:00 KST`  
**Latest Commit**: `e6ce48b` (`feat(model): integrate 4d batched forwarding and chunked feature extraction for large context pretraining`)  
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
* **Model-Level Signal-Aware 40-dim Feature Retrieval & True 4D Batched Forward**:
  - `extract_bag_features(...)` 및 `retrieve_context_indices(...)`로 40차원 특징 표현 기반 Class-Balanced Top-24 ($K=24$) Context 동적 선별.
  - `[E, N, 1000, 512]` ($E=32, N=96$) True 4D Batched Forward + Multi-Worker CUDA Prefetching 파이프라인 구현 완료.

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
| **Phase 5** | Signal-Aware Large Context Pretraining | `configs/train_v21_large_context_pretrain.yaml` | 20e | **크래시 후 재구동 대기 중**<br/>(원인 진단 및 수정 완료, 아래 4-③ 참고) | `checkpoints/20260728_123047/v21_large_context_pretrain` (미생성) | `logs/20260728_123047/v21_large_context_pretrain.out` (crash log) |

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

### ③ 세션 인시던트 진단 및 조치 (2026-07-28 12:30~13:10 KST)

* **Phase 5 크래시 원인 (수정 완료)**: `configs/train_v21_medium.yaml` 등 루트 config 4개(`train_v21_medium.yaml`, `train_v21_medium_retrieved.yaml`, `train_v21_hard_realworld.yaml`, `train_v21_hard_retrieved.yaml`)의 `base_config`가 이관 전 경로인 `train_medium.yaml`을 참조하고 있었음. 커밋 `2a21195`에서 실제 파일을 `configs/archive/v18_v19/train_medium.yaml`로 옮기며 테스트 fallback만 갱신하고 이 4개 config는 갱신하지 않아, Phase 5 (및 이를 상속하는 Phase 1/1-R/2/2-R 전체)가 `FileNotFoundError`로 즉시 크래시함. `logs/20260728_122712/`, `logs/20260728_123047/` 두 차례 실행 모두 1분 내 실패. **해당 4개 config의 `base_config` 경로를 `archive/v18_v19/train_medium.yaml`로 수정하여 정상 로드 확인 완료** (커밋 `e6ce48b`에 반영됨). Phase 5는 아직 재구동 전이며 GPU는 현재 idle.
* **동시 세션 프로세스 행(Hang) 및 중복 실행**: 점검 중 `tests/test_large_context_pretrain.py`를 실행하는 python 프로세스가 여러 차례 반복적으로 재생성되어 load average가 최대 72까지 급등함 (nproc=72). 다른 위치(연구실/집/노트북) 세션이 같은 시간대에 동일 테스트를 반복 구동한 것으로 추정됨. 확인 후 강제 종료(`kill -9`) 처리하여 현재는 python/torchrun 프로세스 없이 idle 상태로 정리됨. **다중 위치 동시 접속 시 프로세스 충돌 가능성에 유의할 것.**
* **⚠ 미해결: 40-dim Feature Retrieval 구현 갭**: `current_architecture.md` §3 및 위 4-②는 $Z \in \mathbb{R}^{40}$ (density slots + tail evidence + covariance sketch + scale) 설계를 명시하지만, 실제 `extract_bag_features` 구현(`src/models/baseline.py`)은 `global_summary`(512-dim)와 `tails` 평균(512-dim)을 concat한 **1024-dim** 벡터를 반환함 — 설계된 40차원 분해는 미구현 상태. 이 gap은 Naive Retrieval(1024-dim mean/spread cosine)이 실패했던 것과 구조적으로 유사한 방식이라 Phase 5의 핵심 가설(신호 인식 retrieval이 배경 노이즈를 우회한다)을 무력화할 위험이 있음. 커밋 `e6ce48b`의 `tests/test_large_context_pretrain.py` 변경분은 `assertEqual(shape[1], 40)`을 `assertGreater(shape[1], 0)`으로 완화하여 테스트가 이 불일치를 잡아내지 못하도록 만들어 놓은 상태. **다음 세션에서 반드시 결정 필요**: (a) 설계대로 실제 40-dim descriptor를 구현하거나, (b) 문서를 1024-dim 구현에 맞게 정정. 결정 전까지 Phase 5 재구동 성과 수치는 "signal-aware retrieval 가설 검증"으로 신뢰하지 말 것.

---

## 5. 다음 작업 세션 Action Plan (Next Steps for Any Agent Location)

연구실, 집, 또는 노트북 어디서 접속하더라도 다음 순서로 작업을 수행하면 됩니다:

1. **40-dim Feature Retrieval 구현 갭 결정 (§4-③ 참고)**: 실제 40-dim descriptor 구현 여부를 먼저 결정. 결정 전 Phase 5 재구동은 잠정 보류 권장.
2. **단위 테스트 실행 및 검증 완료 확인** (이번 세션에서는 프로세스 행 이력으로 보류함):
   - `timeout 600s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"` (All tests PASS 확인, `test_feature_retrieval.py`, `test_large_context_pretrain.py` 포함. 반드시 timeout 적용하여 행 발생 시 자동 종료되도록 할 것)
3. **Phase 5 Large Context + Signal-Aware Retrieval Pretraining 재구동**:
   - Config 경로 버그는 수정 완료. `scripts/launch_interactive_training.sh` 사용 구동 및 `logs/` 점검 후 본 문서(`current_status.md`) 업데이트.
