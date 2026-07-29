# Current development status & multi-location sync SSOT

**Last updated**: `2026-07-29 08:30:00 KST`
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)
**Architecture Version**: `22` (`architecture_version = 22`)
**Purpose**: 연구실 / 집 / 노트북 3개 작업 환경 간 대화 기록 비동기화 문제를 해결하기 위한 Single Source of Truth (SSOT) living document.

---

## 1. 멀티 작업공간 (연구실/집/노트북) 바톤 터치 지침

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 원칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 새로 접속한 Agent는 **`docs/` 최상위 루트의 Living md 파일 5개(`agent_handoff.md`, `current_status.md`, `current_architecture.md`, `current_experiments.md`, `README.md`)만 최우선으로 정독**하여 전체 개발 맥락과 프로젝트 규칙을 파악합니다.
> 3. 터미널 조회가 필요한 명령어는 NVML/쉘 hang 방지를 위해 **반드시 `timeout 3s ps aux | grep python`과 같이 타임아웃**을 적용합니다.
> 4. 코드 변경 시 unittest 통과 필수:
>    `timeout 1500s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"`

---

## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture v22)

* **Python Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python`
* **Torchrun Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun`
* **Target Hardware**: NVIDIA B200 GPU 1장 (`CUDA_VISIBLE_DEVICES=0`, 183GB VRAM)
* **Precision Policy**: `bf16-mixed`
* **핵심 수학 기술 4종** (v19부터 이어져 v22에서도 그대로 유지):
  1. **Z-Score Bag Studentization**: Donor Centroid/Std 기반 세포 표현 스케일 정규화.
  2. **Top-1% Sparse Evidence Module**: 배경세포에 희석되는 희귀 반응 신호 핀포인트 추출.
  3. **Covariance Subspace Shrinkage** (`subspace_shrinkage: 0.25`): 노이즈 축 whitening 방어 및 NaN 예방.
  4. **Auxiliary Pairwise Ranking Loss** (`weight: 0.10`): CE 0.685 부근 gradient 소멸 탈출.
* **Batched Multi-Episode Forward**: `forward_episode_batch` 및 `BaseModel.forward`의 4D 분기가 `[episodes, bags, cells, dim]`을 한 optimizer step에 처리 (v22에서도 유지). 검증: `tests/test_batched_episode_forward.py`.
* **Retrieval 없음**: v22는 context 축소(retrieval) 계층이 **없습니다**. 에피소드의 전체 context bag이 그대로 aggregator에 들어갑니다. 제거 근거는 §4 참고.

---

## 3. 실험 현황

> [!CAUTION]
> **아래 수치들은 모두 v21 이하에서 측정된 것이며, v22 코드로 재현되지 않습니다.**
> v22는 retrieval을 제거하고 `architecture_version`을 22로 올렸기 때문에 **기존 체크포인트는 전부 로드 불가**입니다 (`ModelInterface.on_load_checkpoint` 버전 게이트가 거부). 재사용하려면 v22로 재학습이 필요합니다.

| Phase | 설명 | 지표 | 비고 |
|---|---|---|---|
| Phase 1 (v21) | Medium 합성 사전학습, full context | `val_ce_loss: 0.5921` | 20 epoch, 10,240 steps |
| Phase 2 (v21) | Hard 합성 사전학습, full context | `val_ce_loss: 0.6845` | 50 epoch |
| Phase 4 (v21) | ICI 5-fold, Naive retrieval | AUROC 0.5524 / LL 0.7288 | 95% CI [0.424, 0.677] |
| Phase 6b (v21) | ICI 5-fold, Signal-Aware retrieval | AUROC 0.5481 / LL 0.8672 | 95% CI [0.421, 0.674] |
| **Phase 6c (v21)** | **ICI 5-fold, retrieval 없음** | **AUROC 0.5454 / LL 0.7921 / Acc 0.6092** | 95% CI [0.419, 0.674] — v22 기본 구성에 해당 |

**핵심**: 위 세 ICI 구성의 **95% 신뢰구간이 전부 0.5(무작위)를 포함하고 서로 거의 완전히 겹칩니다.** paired bootstrap 승률도 0.52~0.55로 동전 던지기 수준입니다. 즉 n=87 코호트에서 이 차이들은 **검출 불가능한 노이즈**입니다. 상세 근거: [`history/v21_retrieval_investigation.md`](history/v21_retrieval_investigation.md) §4-⑧.

---

## 4. v22 결정: retrieval 완전 제거 (2026-07-29)

### 제거 근거 (3대 가설 검증 결과)

1. **retrieval은 ICI에서 이득이 없음**: retrieval을 끈 Phase 6c가 켠 Phase 6b와 AUROC 동일(0.5454 vs 0.5481)한데 Log Loss(0.7921 vs 0.8672)와 Accuracy(0.6092 vs 0.5747)는 오히려 **더 좋음**. ICI는 fold당 context가 ~69명뿐이라 24명 선별은 가용 labeled context의 65%를 버리는 것.
2. **구현이 문서화된 설계와 달랐음**: 설계는 "query 1명당 24명 맞춤 선별"이었으나, 실제로는 두 구현 모두 query 전체에 **공용 context 1세트**를 적용 (외부 collator는 첫 query만, 모델 내부는 query 평균 사용). 공용 context는 각 query의 개별 top-24와 평균 61.1%만 겹쳤고 반응자(y=1)에서 ~50%로 최악.
3. **Phase 5 사전학습은 Phase 1의 1/4만 학습됨**: `episode_batch_size` 8→32로 올리면서 `max_epochs: 20`을 그대로 둬 optimizer step이 10,240 → 2,560으로 감소. 게다가 validation이 retrieval 없이 수행되어 `epoch=014` 체크포인트가 학습 모드와 다른 기준으로 선택됨.

무엇보다 **위 차이들이 통계적으로 구분 불가능**하다는 점이 결정적이었습니다. 검출력 없는 지표 위에 복잡한 계층을 유지할 이유가 없다고 판단해 제거했습니다.

### 제거 범위

| 대상 | 조치 |
|---|---|
| `BaseModel.extract_bag_features` / `retrieve_context_indices` / `_retrieve_context_indices_impl` / `retrieve_context_indices_per_query` | 삭제 (303줄) |
| `BaseModel.forward(retrieval_k=...)` 파라미터 | 삭제 |
| `RetrievalEvaluationEpisodeCollator` / `RetrievalSyntheticTrainingEpisodeCollator` / `SignalAwarePretrainEpisodeCollator` | 삭제 |
| `ModelInterface`의 `retrieval_k` 배선 4곳 + `_build_model` pop-list 항목 | 삭제 |
| `scripts/test.py --retrieval-k` | 삭제 |
| retrieval 계열 config 12개, launch 스크립트 3개, VRAM 벤치 2개 | `configs/archive/v21_retrieval/`, `scripts/archive/v21_retrieval/`로 이관 |
| `tests/test_feature_retrieval.py`, `tests/test_large_context_pretrain.py` | 삭제 → `tests/test_batched_episode_forward.py`로 대체 |
| `architecture_version` | 21 → **22** |

**유지된 것**: v21 aggregator/meta-classifier 전부, 4대 수학 기술, batched multi-episode forward, 세션 중 고친 DataLoader CUDA/pin_memory 수정.

**복구 지점**: retrieval 최종 상태는 git tag **`v21-retrieval-final`** 로 보존되어 있습니다 (`git show v21-retrieval-final`).

### v20 롤백이 불가능했던 이유

사용자 요청은 "v20으로 롤백"이었으나 조사 결과:
* **v20은 코드로 존재하지 않습니다.** 이 브랜치 히스토리는 v18 → v19 → **v21**이며 (`ecf6199`가 19에서 21로 직접 점프), `main`은 아직 v18입니다. `configs/archive/v20/*.yaml`는 v19 코드 위에서 돌던 **설정 파일 시리즈**일 뿐입니다.
* **v21 ≈ v19 + retrieval + 사소한 2줄**. `ecf6199`가 baseline.py에서 제거한 실질 코드는 4줄뿐이고, "v21 4대 개혁"으로 문서화됐던 기술들은 이미 v19에 있었습니다.
* 따라서 retrieval 제거는 버전 롤백이 아니라 **덧붙은 계층의 절제**로 처리하는 것이 맞았고, 사용자 승인 하에 v22 신규 버전으로 진행했습니다.

---

## 5. 실험 전략 (2026-07-29 확정)

> [!IMPORTANT]
> **합성 데이터로 모든 결정을 내리고, ICI 실데이터는 최종 테스트에만 씁니다.**

| 단계 | 데이터 | 도구 | 반복 |
|---|---|---|---|
| 아키텍처 탐색·튜닝 | 합성 val | `scripts/evaluate_synthetic.py` (AUROC + episode cluster CI) | 자유롭게 |
| 최종 확인 | ICI 5 seed × 5 fold + 외부 코호트 26명 | `scripts/launch_ici_protocol.sh` → `scripts/evaluate_protocol.py` | **후보 확정 후 1회** |

**근거 (실측)**:

| | 표본 | AUROC | 95% CI | CI 폭 |
|---|---|---:|---|---:|
| ICI 5-fold (v21 Phase 6c) | 87명 | 0.5454 | [0.422, 0.664] | **0.242** |
| 합성 val (v22 1-epoch) | 104 episodes / 1,698 query | 0.7272 | [0.697, 0.759] | **0.062** |

합성 구간이 약 **4배 좁고**(그것도 episode cluster bootstrap이라는 보수적 계산으로), 신호도 훨씬 강합니다(1 epoch만에 0.73 vs 0.55). 합성은 `episodes_per_epoch`로 더 좁힐 수 있지만 ICI는 87명이 상한입니다. v21의 실패는 **검출력 없는 지표 위에서 아키텍처를 반복 비교한 것**이었고, ICI를 반복해서 보면 그 87명에 과적합됩니다.

**지켜야 할 선**: ICI 결과를 보고 아키텍처를 다시 고치기 시작하면 ICI는 더 이상 테스트 세트가 아닙니다. 또한 ICI에서 ±0.13 이내 변동은 그 자체로 아무 근거가 되지 않습니다.

**합성 평가의 CI는 query가 아니라 episode 단위(cluster bootstrap)로 계산합니다.** 한 episode의 query들은 context set과 생성 파라미터를 공유해 독립이 아니며, query 단위 재표집은 구간을 실제보다 좁게 만듭니다. 실질 표본 크기는 **episode 수**입니다.

---

## 6. 다음 작업 세션 Action Plan

1. **[완료] 평가 프로토콜 보강** (2026-07-29). §7 참고. 요약: 검정력 분석 결과 **이 코호트는 +0.13 AUROC 미만의 효과를 검출할 수 없습니다.** 5개 seed partition과 외부 코호트(26명)가 이미 디스크에 있었으나 v21 실험은 전부 SEED42 하나만 썼습니다. 이제 `scripts/launch_ici_protocol.sh`로 5 seed × 5 fold를 돌리고 `scripts/evaluate_protocol.py`로 집계하며, `scripts/test.py`는 모든 AUROC에 CI를 자동 부착합니다.
2. **v22 사전학습 재실행.** 모든 기존 체크포인트가 무효화되었으므로 v22 기준선을 새로 만들어야 합니다. `configs/train_v22_medium.yaml`로 시작하되, Phase 5의 실패를 반복하지 않도록 **`episode_batch_size`를 바꾸면 `max_epochs`도 함께 조정**해 optimizer step 수를 맞출 것 (Phase 1 기준 10,240 steps). 스모크 테스트에서 512 steps/epoch·epoch0 `val_loss=0.667`로 Phase 1과 일치함을 확인했습니다.
3. **v22 ICI 기준선 측정.** `scripts/launch_ici_protocol.sh` (2번의 v22 체크포인트를 `PRETRAINED_CKPT`로 전달, 미지정 시 scratch). 25개 run이므로 GPU 점유 시간을 고려해 계획할 것.
4. **[전략] ICI AUROC로 아키텍처 우열을 가리는 방식 재검토.** §7의 검정력 표가 보여주듯 이 코호트에서 소폭 개선을 가려내는 것은 불가능합니다. 합성 데이터(n을 원하는 만큼 늘릴 수 있음)에서 판단하고 ICI는 최종 확인용으로만 쓰거나, 코호트를 확대하는 방향을 고려해야 합니다.

---

## 7. 평가 프로토콜 보강 (2026-07-29)

v21 조사의 결론("모든 비교가 노이즈였다")에 대응해 평가 체계를 다시 만들었습니다.

### ① 검정력 분석 — 가장 중요한 결과

`scripts/power_analysis.py` (baseline AUROC 0.55, 모델 간 상관 ρ=0.7 — 실측 Phase 6b vs 6c Pearson ρ=0.737 기반):

| 실제 AUROC 향상 | 검출 확률 |
|---:|---:|
| +0.02 | 15% |
| +0.05 | 26% |
| +0.10 | 66% |
| **+0.15** | **92%** |

> **n=87에서는 +0.13~0.15 미만의 개선을 검출할 수 없습니다.** v21이 쫓던 0.004~0.04 차이는 검출 확률 15~26%였습니다. 이것이 그 실험들이 결론에 도달하지 못한 근본 이유입니다.

### ② 발견: 자원의 4/5를 안 쓰고 있었음

- **seed partition 5개**(`SEED42/1234/2026/271828/314159`)가 디스크에 있었고 각각 87명을 5-fold로 덮는 **독립 분할**입니다 (CV0 기준 seed 간 val donor 겹침 1~5/18). **v21 실험은 전부 SEED42 하나만 사용.**
- **외부 코호트** `data/ICI_GSE285888_scConcept_512.pt` (26명, R 15 / NR 11)도 이미 존재하고 `ICIDataset(state='external')`로 로드 가능하나 **한 번도 평가하지 않았음.**

### ③ 구축한 것

| 스크립트 | 역할 |
|---|---|
| `scripts/power_analysis.py` | 실험 전 검출 가능 효과 크기 확인 |
| `scripts/launch_ici_protocol.sh` | 5 seed × 5 fold sweep (seed 내 fold 병렬, seed 간 순차), manifest 기록 |
| `scripts/evaluate_protocol.py` | per-seed / across-seed SD / pooled bootstrap CI / 외부 코호트를 구분해 보고 |
| `scripts/test.py` | 모든 AUROC에 bootstrap 95% CI **자동 부착** |
| `scripts/compare_predictions.py` | 두 run의 CI + paired bootstrap 승률 |

부수 정리: `--cv`를 `launch_interactive_training.sh`에 추가해 fold를 주입식으로 바꿨고, per-fold config 5개를 `train_v22_ici_finetune.yaml` 하나로 통합했습니다.

### ④ 반드시 구분할 것: seed를 늘려도 CI는 안 줄어듦

- **across-seed SD**: partition/학습 재현성. seed를 늘리면 평균의 표준오차가 줄어듦.
- **pooled bootstrap CI**: 코호트 표본 오차. **seed를 아무리 늘려도 줄어들지 않음** — 같은 87명을 재사용하기 때문. 사람을 더 모아야만 좁아집니다.

이 전제는 `tests/test_evaluation_protocol.py::test_smaller_cohort_gives_a_wider_interval`로 테스트에 고정해 두었습니다.

---

## 8. Source of Truth 파일

- Backbone & Version: `src/models/baseline.py` (`architecture_version = 22`)
- Data Interface & Collators: `src/modules/data_interface.py`
- Loss & Metrics: `src/modules/model_interface.py`
- 통계 비교 도구: `scripts/compare_predictions.py` (§6 참고)
- 검증 스위트: `tests/test_base_model.py`, `tests/test_model_interface.py`, `tests/test_batched_episode_forward.py`, `tests/test_evaluation_protocol.py`
- 평가 프로토콜: `scripts/power_analysis.py`, `scripts/launch_ici_protocol.sh`, `scripts/evaluate_protocol.py`, `scripts/evaluate_synthetic.py`
- **공용 평가 지표 구현**: `src/utils/metrics.py` (rank 기반 AUROC, cluster bootstrap) — 모든 평가 스크립트가 이 하나를 사용하므로 지표가 스크립트마다 어긋날 수 없음
