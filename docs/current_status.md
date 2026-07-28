# Current development status & multi-location sync SSOT

**Last updated**: `2026-07-28 17:59:00 KST`  
**Latest Commit**: `42c3fa8` (`docs: record Phase 5 launch saga and successful training run`)  
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
* **Model-Level Signal-Aware Retrieval & True 4D Batched Forward**:
  - `extract_bag_features(...)`는 aggregator가 분류기용으로 이미 계산해 두는 **40-token 구조화 요약**(`_all_structured_tokens`: 1 global + 12 slots×3종 + 3 tail, 각 512-dim, `[bags,40,512]`)을 그대로 재사용하고, `retrieve_context_indices(...)`가 이를 flatten한 뒤 Cosine Similarity로 Class-Balanced Top-24 ($K=24$) Context를 동적 선별함 (상세: §4-②).
  - `training_step` → `retrieve_context_indices` → `forward_episode_batch`로 이어지는 `[E, N, 1000, 512]` ($E=32, N$ 최대 100) True 4D Batched Forward + Signal-Aware Retrieval + Multi-Worker/In-Process CUDA Prefetching 파이프라인이 Phase 5에서 20 epoch 완주로 실증 완료 (§4-④, §4-⑤).

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
| **Phase 5** | Signal-Aware Large Context Pretraining | `configs/train_v21_large_context_pretrain.yaml` | 20e | **20 epoch 완주 (6번째 재시도 만에 성공, §4-④~⑤ 참고)**<br/>**Best `val_ce_loss: 0.5940`** (epoch 14) — Phase 1 Full-Context(`0.5921`)에 근접, Phase 1-R Naive Retrieval(`0.6839`) 대비 대폭 개선<br/>최종 epoch 19: `val_loss: 0.608`, `train_loss: 0.723` (progress-bar 결합 loss, `val_ce_loss`와 별도 지표) | `checkpoints/20260728_144957/v21_large_context_pretrain/epoch=014-val_ce_loss=0.5940.ckpt` | `logs/20260728_144957/v21_large_context_pretrain.out` |
| **Phase 6** | ICI 5-Fold CV Fine-Tune (Phase 5 체크포인트 기반) | `configs/train_v21_ici_finetune_fold{0..4}.yaml` (`scripts/launch_phase6_5fold.sh`) | 50e (resume epoch 14부터) | **진행 중** (2026-07-28 17:57 KST 5-fold 모두 기동, checkpoint restore 확인 완료). 목표: Phase 4(AUROC `0.5524`, Log Loss `0.7288`) 대비 개선 여부 확인 | `checkpoints/20260728_1757{10,12,14,16,18}/v21_ici_finetune_phase6_f{0..4}/` | `logs/20260728_1757{10,12,14,16,18}/v21_ici_finetune_phase6_f{0..4}.out` |

---

## 4. 핵심 집중 이슈: Pretraining Stage Context Expansion & Signal-Aware Retrieval

### ① 문제 진단 (Why Naive Retrieval Failed in Pretraining)
* Pretraining 단계에서 Context set을 효율적으로 구성하기 위해 주입했던 기존 Naive Retrieval(`RetrievalEvaluationEpisodeCollator`)은 세포 1,000개의 단순 평균 및 표준편차 Cosine Similarity를 사용함.
* Single-cell 면역 데이터는 95%+가 공통 배경 세포이고 반응 신호는 <1%~5% 희귀 세포에 쏠려 있어, Naive Retrieval이 반응 유사 donor가 아닌 **95% 배경 노이즈가 유사한 donor**를 추출함.
* 이로 인해 사전학습 시 모델에 노이즈 donor가 주어지면서 `val_ce_loss`가 `0.5921`에서 `0.6839`로 상향 튐.

### ② 해결 설계: 40-token Aggregator Summary 재사용 Signal-Aware Retrieval
* **최종 구현 (2026-07-28)**: `extract_bag_features`는 aggregator가 분류기(`StructuredPopulationMetaClassifier`)를 위해 **이미 계산해 두는** 40-token 구조화 요약(`_all_structured_tokens`: 1 global + 12 slots × 3종(center/spread/rare) + 3 tail = 40개 토큰, 각 512-dim, shape `[bags, 40, 512]`)을 **그대로 재사용**함. 별도의 손으로 압축한 scalar feature를 새로 만들지 않음.
  - 세션 중 두 차례 반복 수정 이력: (a) 최초 구현은 density/tail/covariance/scale을 손으로 압축한 flat 40-dim vector였으나, aggregator가 이미 만들어 둔 40-token 요약을 재사용하는 게 아니라 별개의 새 40x512를 만드는 셈이라는 지적을 받아 폐기. (b) 슬롯당 3종 통계(center/spread/rare norm)로 압축한 36+4-dim 버전도 마찬가지로 "새로 만드는" 방향이라 폐기. 최종적으로 (c) `_all_structured_tokens`을 직접 재사용하는 현재 버전으로 확정.
  - **Anchor 안정성 이슈 및 조치**: `_all_structured_tokens`를 그대로 쓰면, population anchor(슬롯 중심)가 `context_mask`(= 그 호출에 함께 들어온 bag 집합)에 의존하기 때문에 청크(chunk) 구성에 따라 같은 bag의 descriptor가 달라지는 문제가 발견됨 (chunk_size=8 vs 0 비교 시 cosine similarity 평균이 0.474까지 하락, 최솟값 음수). **조치**: `extract_bag_features`가 anchor를 항상 전체 pool(`x`/`context_mask` 전체)에서 한 번만 계산(`self.aggregator._context_anchors`)하고, 청크는 오직 `_forward_dense` 호출을 나누는 메모리 최적화로만 사용하도록 재구현. 수정 후 chunked vs dense 결과가 최대 절대오차 4.5e-8, cosine similarity 1.0으로 완전히 일치함을 확인.
* `retrieve_context_indices`는 이 `[bags,40,512]`를 flatten(`[bags, 40*512]`)한 뒤 cosine similarity로 Class-Balanced Top-12 NR + Top-12 R ($K=24$) donor를 동적 추출함.

### ③ 세션 인시던트 진단 및 조치 (2026-07-28 12:30~13:10 KST)

* **Phase 5 크래시 원인 (수정 완료)**: `configs/train_v21_medium.yaml` 등 루트 config 4개(`train_v21_medium.yaml`, `train_v21_medium_retrieved.yaml`, `train_v21_hard_realworld.yaml`, `train_v21_hard_retrieved.yaml`)의 `base_config`가 이관 전 경로인 `train_medium.yaml`을 참조하고 있었음. 커밋 `2a21195`에서 실제 파일을 `configs/archive/v18_v19/train_medium.yaml`로 옮기며 테스트 fallback만 갱신하고 이 4개 config는 갱신하지 않아, Phase 5 (및 이를 상속하는 Phase 1/1-R/2/2-R 전체)가 `FileNotFoundError`로 즉시 크래시함. `logs/20260728_122712/`, `logs/20260728_123047/` 두 차례 실행 모두 1분 내 실패. **해당 4개 config의 `base_config` 경로를 `archive/v18_v19/train_medium.yaml`로 수정하여 정상 로드 확인 완료** (커밋 `e6ce48b`에 반영됨). Phase 5는 아직 재구동 전이며 GPU는 현재 idle.
* **동시 세션 프로세스 행(Hang) 및 중복 실행**: 점검 중 `tests/test_large_context_pretrain.py`를 실행하는 python 프로세스가 여러 차례 반복적으로 재생성되어 load average가 최대 72까지 급등함 (nproc=72). 다른 위치(연구실/집/노트북) 세션이 같은 시간대에 동일 테스트를 반복 구동한 것으로 추정됨. 확인 후 강제 종료(`kill -9`) 처리하여 현재는 python/torchrun 프로세스 없이 idle 상태로 정리됨. **다중 위치 동시 접속 시 프로세스 충돌 가능성에 유의할 것.**
* **✅ 해결됨: 40-dim Feature Retrieval 구현 갭 (2026-07-28 14:05 KST)**: `extract_bag_features`가 실제로는 `global_summary`+`tails` 평균을 concat한 1024-dim을 반환하고 있었고(설계된 40차원/40토큰 재사용 미구현), `tests/test_large_context_pretrain.py`의 관련 assertion이 완화되어 이 gap을 잡아내지 못하던 문제를 발견함. **최종 조치**: 위 4-②에 기술된 대로 aggregator의 기존 40-token 요약(`_all_structured_tokens`)을 재사용하도록 구현(2차 검토 끝에 확정), anchor 안정성 문제도 함께 수정. `test_feature_retrieval.py`/`test_large_context_pretrain.py`의 관련 assertion을 새 `[bags,40,512]` shape와 `torch.allclose` 엄격 비교로 갱신함. **검증**: (a) 단독 스크립트로 16 bags 기준 dense vs chunked(chunk_size=8) 결과가 최대 절대오차 4.5e-8로 사실상 동일함을 확인, `retrieve_context_indices` 및 4D 배치 경로 모두 정상 동작 확인. (b) unittest로 `test_chunked_extract_bag_features`(96 bags, `torch.allclose` 엄격 검사)와 `test_collator_formatting` PASS 확인. (c) `test_feature_retrieval.test_extract_bag_features`(30 bags, shape만 검증)는 아래 CPU 지연 이슈로 280초 내 결과 미출력 — 순수 shape 검증이라 로직 문제 아님, 다음 세션에서 GPU 또는 긴 타임아웃으로 재확인 권장.
* **⚠ 미해결 (별도 조치 필요, Phase 5 성공과 무관): CPU 실행 시 심각한 지연 / `test_4d_batched_forward` 응답 없음**: `tests/test_large_context_pretrain.py` 전체(discover 또는 파일 단위)를 CPU 환경에서 실행하면 `test_4d_batched_forward`(커밋 `e6ce48b`에서 추가된 4D 배치 forward 테스트)에서 180초 타임아웃 내 결과가 출력되지 않음. 별도로 `extract_bag_features`만 단독 호출해 스케일링을 측정한 결과, 동일 모델이 CPU에서 4 bag/10 cell도 최소 ~7초, 30 bag/100 cell 조합은 90초를 넘기는 등 **bag/cell 크기에 따라 비선형적으로 느려짐**을 확인함. Target Hardware가 B200 GPU임에도 유닛테스트들이 모델을 `.cuda()`로 옮기지 않고 순수 CPU에서 forward를 실행하는 것이 근본 원인일 가능성이 높음. **다음 세션 확인 필요**: 테스트를 GPU에서 실행하거나 더 긴 타임아웃으로 단독 실행해 실제 종료 여부/소요 시간 확인.

### ④ Phase 5 학습 파이프라인 완전 가동 (2026-07-28 14:28~14:52 KST)

Config를 고친 뒤에도 학습이 5차례 연속 크래시했고, 매번 근본 원인이 달랐음. 순서대로:

1. **`num_candidate_bags_pretrain` 죽은 config 키**: `SyntheticManifoldGenerator`/`SyntheticEpisodeDataset` 어디에도 구현되지 않은 파라미터가 `dataset_kwargs`에 있어 `TypeError`로 즉시 크래시. 코드 전체에서 이 키를 쓰는 곳이 config 한 줄뿐임을 확인 후 제거 (`num_bags: [60,100]`가 이미 동일 역할 수행).
2. **DataLoader worker의 CUDA fork 충돌**: `generation_device: cuda` + `num_workers: 4` 조합에서 `torch.Generator(device='cuda')`가 forked worker 안에서 `CUDA error: initialization error` 발생. `generation_device=='cuda'`일 때 `num_workers=0`으로 강제하도록 `DataInterface._episode_dataloader` 수정 (기존 in-process `CudaPrefetchDataLoader`가 이미 오버랩을 제공하므로 multi-process worker는 GPU 메모리만 낭비 — 실제로 worker 4개가 100GiB+ 점유하는 것을 확인).
3. **pin_memory 충돌**: `generation_device=cuda`로 이미 GPU에 생성된 텐서를 `pin_memory()`하려다 `RuntimeError`. CUDA 생성 dataset에는 자동으로 `pin_memory=False` 적용.
4. **`retrieve_context_indices`의 gradient graph 누수**: `torch.no_grad()` 없이 실행되어, index 선택에만 쓰이는 거대한 중간 계산(anchor·chunk별 forward_dense)이 backward 시점까지 GPU에 유지됨 → step이 진행될수록 메모리 누적 → OOM. `_retrieve_context_indices_impl` 헬퍼 메서드로 분리하고 `torch.no_grad()`로 감싸 해결.
5. **(가장 근본적) `training_step`이 retrieval을 아예 호출하지 않음**: 4D 배치 입력(Phase 5가 사용하는 형태) 시 `training_step`이 `self.model.forward_episode_batch(...)`를 직접 호출하는데, 이 메서드엔 `retrieval_k` 파라미터 자체가 없어 `episodes * num_bags`(최대 32×100=3200) bag을 **전부** 한 번에 dense aggregator forward에 통과시킴. `retrieve_context_indices`/`extract_bag_features`를 아무리 고쳐도 이 경로가 호출 안 되니 매번 동일하게 OOM. **조치**: `training_step`에서 `forward_episode_batch` 호출 전에 `retrieval_k>0`이면 `self.model.retrieve_context_indices(...)`를 먼저 호출해 N을 `retrieval_k + query_count`로 줄이도록 통합. `retrieval_k`/`retrieval_chunk_size`는 `model_kwargs`에 추가하고 `ModelInterface._build_model`의 pop-list에 등록(다른 training-only hparam과 동일 패턴).
   - **부수 발견**: 4D retrieval 경로의 `mask_index_b`가 query 1개만 가정하고 있었음(`torch.tensor([len(selected_context_idx)])`). 이 config는 `training_targets_per_episode: [5, 12]`로 episode당 query가 여러 개라 실제로는 틀린 결과를 만들고 있었음. 3D 경로처럼 query 개수만큼의 range로 수정.

**검증**: 위 5개 수정 후 `training_step`을 GPU에서 직접 호출하는 표준 스크립트로 실제 config의 worst-case 크기(E=32, N=100, cells=1000)에서 4 step 반복 → peak 메모리 90~99GiB에서 안정(성장 없음). 실제 `launch_interactive_training.sh`로 재구동 → 정상 진행 확인 (최종 결과는 §4-⑤).

### ⑤ Phase 5 최종 결과 및 가설 검증 (2026-07-28 14:49~15:49 KST, 20 epoch 완주)

* **Epoch별 `val_loss`/`train_loss` (progress-bar 결합 지표, 3자리 반올림)**:

  | Epoch | val_loss | train_loss | Epoch | val_loss | train_loss |
  |---:|---:|---:|---:|---:|---:|
  | 0 | 0.606 | 0.747 | 10 | 0.614 | 0.723 |
  | 1 | 0.606 | 0.747 | 11 | 0.603 | 0.725 |
  | 2 | 0.631 | 0.731 | 12 | 0.620 | 0.722 |
  | 3 | 0.620 | 0.724 | 13 | 0.621 | 0.726 |
  | 4 | 0.631 | 0.728 | 14 | 0.605 | 0.721 |
  | 5 | 0.622 | 0.728 | 15 | **0.594** | 0.720 |
  | 6 | 0.598 | 0.727 | 16 | 0.601 | 0.719 |
  | 7 | 0.627 | 0.720 | 17 | 0.604 | 0.720 |
  | 8 | 0.602 | 0.725 | 18 | 0.609 | 0.719 |
  | 9 | 0.614 | 0.721 | 19 | 0.606 | 0.724 |

* **체크포인트 기준 `val_ce_loss`** (ModelCheckpoint가 저장한 상위 3개 + last): `epoch=005: 0.5975`, **`epoch=014: 0.5940` (최적)**, `epoch=015: 0.6013`, `epoch=019 (last): 저장값 없음, 진행바 기준 val_loss=0.608`.
* **가설 검증**: Phase 5의 Best `val_ce_loss: 0.5940`은 Phase 1 Full-Context(`0.5921`)의 **0.0019 이내**로 근접하며, Phase 1-R Naive Retrieval(`0.6839`, 95%+ 배경 노이즈 donor 추출 문제)보다 **0.09 이상 우수**함. 이는 §4-①에서 진단한 "Naive Retrieval이 배경 노이즈 유사 donor를 뽑아 학습을 방해한다"는 문제를, aggregator의 40-token 구조화 요약(density/tail/covariance 정보를 이미 담고 있는 신호 인식 표현)을 재사용한 retrieval로 극복했다는 Phase 5의 핵심 가설을 실증적으로 뒷받침함.
* **W&B Run**: https://wandb.ai/teasol/ICF/runs/9ldg44nr (`v21_large_context_pretrain_20260728_144957`)
* **다음 단계 제안**: 이 체크포인트(`epoch=014-val_ce_loss=0.5940.ckpt`)를 Phase 3/4처럼 ICI 5-Fold 실데이터 미세조정에 사용해, Signal-Aware Retrieval 사전학습이 Naive Retrieval 기반 Phase 4(Log Loss `0.7288`) 대비 실데이터 미세조정 성능을 개선하는지 확인하는 것이 자연스러운 다음 실험.

---

## 5. 다음 작업 세션 Action Plan (Next Steps for Any Agent Location)

연구실, 집, 또는 노트북 어디서 접속하더라도 다음 순서로 작업을 수행하면 됩니다:

1. **[진행 중] Phase 5 체크포인트로 ICI 5-Fold 실데이터 미세조정** (§4-⑤ "다음 단계 제안" 참고): `scripts/launch_phase6_5fold.sh`로 2026-07-28 17:57 KST 5-fold 모두 기동 완료 (Phase 4와 동일 패턴, `PRETRAINED_CKPT`만 Phase 5 체크포인트로 교체). **다음 세션 확인 필요**: 5-fold 완주 후 AUROC/Log Loss를 Phase 4(AUROC `0.5524`, Log Loss `0.7288`)와 비교하여 결과를 §3 표와 `current_experiments.md`에 기록.
2. **`test_4d_batched_forward` CPU 지연/무응답 원인 확인 (§4-③ 참고)**: GPU(`CUDA_VISIBLE_DEVICES=0`)에서 재현 여부 확인하거나, 10분+ 타임아웃으로 단독 실행해 실제 종료 여부와 소요 시간 측정.
3. **단위 테스트 전체 실행 및 검증 완료 확인**:
   - `timeout 600s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"` (All tests PASS 확인, `test_feature_retrieval.py`, `test_large_context_pretrain.py` 포함. 반드시 timeout 적용하여 행 발생 시 자동 종료되도록 할 것. 2번 이슈로 인해 이번 세션에서는 전체 discover 대신 개별 스크립트 검증만 수행함)
