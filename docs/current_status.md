# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-05` (**§36 CCER-v2 Epoch 18 평가 완료 — v30 Baseline 유지**)
**Status**: **v30은 확정 baseline 유지**. CCER-v2 20-epoch 학습 완료 후 epoch 18 best로 합성 1,000-episode (AUROC `0.8514`) 및 Musk zero-shot (AUROC `0.8470`) 평가를 완료했다. 수치상 v30 baseline (Musk `0.854`) 대비 승격 기준을 충족하지 못하므로 **v30 확정 baseline을 계속 유지**한다.
* **CCER-v2 실측 결과**: 합성 AUROC `0.8514`, Musk AUROC `0.8470` (n<=4: 0.792, 5..10: 0.842, 11..34: 0.933, n>34: 0.698).
* **다음 Action**: v30 medium 참조 재학습 및 대형 bag (n>34) 구간 진단 진행. ICI 잠금은 유지한다.

> **사용자 결정 (2026-08-05, 확정)**:
> 1. **v30 S2가 정식 확정 baseline 유지.** v31 CCTS/CCER-v2는 정식 baseline으로 승격/채택하지 않음 (실험 후보 기록만 남김).
> 2. **ICI는 손대지 않습니다.** (잠금 유지)
> 3. **Musk 목표는 0.95 유지.**

**Read first if you are picking this up**: **§36 (CCER-v2 평가 결과 및 v30 유지)**,
**§35 (CCER-v2 학습)**, **§33 (CCER-v2 계약)**, **§29 (v30 확정 baseline)**.

**열린 과제**: ① v30 medium 참조 재학습, ② n>34 최약 구간의 B2b/대형 bag 진단, ③ ICI 잠금 유지. 해결·폐기 기록은 [`history/archive.md`](history/archive.md).

**Branches**: `main` = v30 확정 baseline + v31 후보 구현. 참고용 branch/tag 구조는
[`history/branch_structure.md`](history/branch_structure.md).
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)
**Architecture Versions**: `30` 확정 baseline, `31` CCER-v2 평가 후보. v31 config는
`configs/train_v31_ccer_v2.yaml`; 코드 기본 `bag_representation`은 `legacy` 유지.
**Purpose**: 연구실 / 집 / 노트북 간 상태를 동기화하는 SSOT living document.

---

## 0. 30초 요약 — 새 세션은 여기부터

> **2026-08-04 갱신**: **v30 확정 baseline (B1 `poolz_l2` + B2 cardinality-faithful)** — Musk zero-shot
> **0.854** 경신, 게이트 6항목 통과 (§28·§29). v24는 이전 확정 baseline으로 보존. 현 최고: **v30
> musklike-easy Musk zero-shot AUROC 0.854** (이전: 지렛대 1 centered 표현 0.822, §22). 열린 과제:
> ① v30 medium 재학습(참조 수치), ② n>34 최약 구간(B2b/대형 진단), ③ ICI 잠금, ④ Musk 0.95.

**지금 어디까지 왔나**
- v22 = v21에서 **retrieval 계층을 완전히 제거**한 버전. 아키텍처 본체는 v21과 동일 (§4). **2026-08-01부로 v24에 자리를 내주고 폐기.**
- **v24 확정 (2026-08-01)**: residual + bottleneck bag projection (구 v24-B1). best `val_ce_loss 0.5903045` @ epoch 41. 근거와 범위는 §3 "최종 결정" 참고.
- v22 공식 기준선(폐기 전 참고값): 합성 val AUROC **0.7078 [0.696, 0.719]** (1,000 episodes), `val_ce_loss 0.5946`. 예측 파일 `predictions/synthetic_v22_baseline_1000ep.pt` (§3).
- **평가 프로토콜 재구축 완료**: 모든 비교에 bootstrap CI 필수, 합성은 episode cluster CI, ICI는 최종 테스트 1회만 (§5, §7). **단, v24 확정은 이 프로토콜에 따른 paired 비교 없이 사용자 결정으로 진행됨** — §3 참고.

**이번 라운드에 확정된 것 (전부 학습 없이 진단으로)**
| 결론 | 근거 |
|---|---|
| 🛑 **Tier 1 (covariance/세포선택) 종료** | 세포 선택 점수 4종 전부 무작위 수준. bag 라벨로는 purity 0.128 (사전 판정 기준 ≤0.15) |
| 🎯 **진짜 약점은 `state`** | effect scale 통일 시 covariance는 composition과 동률, state만 전 구간 최하위 |
| 📏 **val episode 1,000개 필요** | 104개는 CI 폭 0.074 + task당 ~20 episode로 판정 불가 |
| 🛑 **Tier 2 (state) 종료** | 현재 모델 0.6217과 model-input probe 0.6196~0.6210이 동률. raw mean 관측 통계는 0.5273~0.5578 |
| ✅ **T3-3 Hard 기준선 완료** | best val CE 0.6839; AUROC 0.5483 [0.538, 0.558], 1,000 episodes |
| ✅ **T4 context 병목 확인** | context 10→20→40→80→160에서 AUROC 0.5084→0.5193→0.5312→0.5505→0.5737로 단조 증가 |

**다음 할 일 — v24 확정 이후**
1. ~~v23-A0/v24-A0/v24-B0/v24-B1 4종 1,000 pool-400 paired 비교~~ → **사용자 결정으로 폐기 (2026-08-01)**. v24-B1을 그대로 v24로 확정.
2. ICI를 v24로 돌리려면 `train_v22_ici_finetune.yaml`/`train_v22_ici_scratch.yaml`에 상응하는 v24 config가 아직 없음 — 필요 시 신규 작성.
3. **ICI는 여전히 잠금.** 이번 결정은 "합성 비교 통과"가 아니라 사용자의 직접 판단이므로, ICI 실행 여부는 별도로 다시 확인받을 것.

구현 검증: v24-B1 도입 시점 신규 테스트(`test_bottleneck_projection_with_residual_mean`)를 포함한 전체 unittest
**123개 통과** (`696.503s`, v23-A0 도입 커밋 기준 — v24-B1 이후 재실행 기록 없음, 코드 변경 없었으므로 유효).

**작업 규칙 4가지**
- 평가는 `--val-episodes 1000`, 비교는 `scripts/compare_predictions.py` (paired cluster bootstrap).
- **오라클을 쓰는 상한을 목표로 삼지 말 것** — descriptor가 `responsive_instance_mask`나 latent를 쓰는지 먼저 확인 (§3에서 실제로 틀린 사례 있음).
- 기본 task별 AUROC 표로 아키텍처 강약을 판단하지 말 것 — 생성기 effect scale에 오염됨 (§3 T3-1).
- **다음 Action과 판정 기준이 명확하면 재확인 없이 실행**. 각 논리 단위마다 결과·명령·로그/산출물·판단·다음 Action을 이 문서에 갱신하고 Git 커밋하여, 다른 작업공간이 문서와 `git log`만으로 이어받을 수 있게 할 것.

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

## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture **v24 확정**)

* **Python Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python`
* **Torchrun Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun`
* **Target Hardware**: NVIDIA B200 GPU 1장 (`CUDA_VISIBLE_DEVICES=0`, 183GB VRAM)
* **Precision Policy**: `bf16-mixed`
* **핵심 수학 기술 4종** (v19부터 이어져 v24에서도 그대로 유지):
  1. **Bag Centering + Per-Cell L2 Projection**: per-bag centroid $\mu_i$를 빼고 **per-cell L2**로
     사영합니다 — $\tilde{x}_{i,j} = \delta_{i,j}/\|\delta_{i,j}\|_2$, $\delta_{i,j}=x_{i,j}-\mu_i$.
     > **2026-08-04 정정**: 이 항목은 오래도록 "Z-Score Bag Studentization: Donor Centroid/**Std**
     > 기반 스케일 정규화"라고 적혀 있었으나 **코드는 per-feature std $S_i$로 나누지 않습니다**
     > (`_bag_view`, `baseline.py:618-644`). $S_i$는 `global_summary` 토큰으로 **출력**되고,
     > centering 전 편차 $\delta$는 공분산 스케치 경로의 입력으로 쓰입니다. 문서가 기술했던 그
     > 공식(`zscore` 변형)은 실측 최하위였습니다(합성 Medium 0.5054, §19). 상세: §26 및
     > [`current_architecture.md`](current_architecture.md) ①.
     > **알려진 한계**: per-bag centering은 rank $(N_i-1)$ 사영이라 $N_i$가 작으면 파괴적입니다
     > ($N_i=1$이면 0벡터). Musk(median 12 instances) 병목의 실체 — §26.
  2. **Top-1% Sparse Evidence Module**: 배경세포에 희석되는 희귀 반응 신호 핀포인트 추출.
     선별은 bag 중심 거리가 아니라 **context anchor novelty**로 정렬합니다(①의 L2 때문에
     $\|\tilde{x}\|_2$가 항등적 1.0이므로 — §26).
  3. **Covariance Subspace Shrinkage** (`subspace_shrinkage: 0.25`): 노이즈 축 whitening 방어 및 NaN 예방.
  4. **Auxiliary Pairwise Ranking Loss** (`weight: 0.10`): CE 0.685 부근 gradient 소멸 탈출.
* **Batched Multi-Episode Forward**: `forward_episode_batch` 및 `BaseModel.forward`의 4D 분기가 `[episodes, bags, cells, dim]`을 한 optimizer step에 처리 (v24에서도 유지). 검증: `tests/test_batched_episode_forward.py`.
* **Retrieval 없음**: v22부터 context 축소(retrieval) 계층이 **없습니다**. 에피소드의 전체 context bag이 그대로 aggregator에 들어갑니다. 제거 근거는 §4 참고.
* **Bag Projection (v24 확정 요소)**: residual + bottleneck bag projection
  (`project_structured_tokens: true`, `projection_bottleneck_dim: 64`, `projection_residual_mean: true`).

---

## 3. 실험 현황

### ✅ v23-A0 / v24-A0 / v24-B0: 50-epoch 완료, 모두 폐기 (2026-07-31 완료 → 2026-08-01 폐기)

세 candidate 모두 Medium에서 scratch 50-epoch 완료했으나 (best `val_ce_loss`:
v23-A0 exact mean `0.5912` / v24-A0 learned projection(slot 1) `0.5976` /
v24-B0 per-token bottleneck(slot 12) `0.5923`) v24-B1이 최종 확정되며 함께
폐기됐습니다. 전체 config/checkpoint/param-count 기록은
[`history/v23_v24_bag_collapse_candidates.md`](history/v23_v24_bag_collapse_candidates.md)로 이관.

### ✅ v24-B1 residual bottleneck projection: 50-epoch 완료 (2026-07-31)

v24-B0 병목 구조(40×64=2560d)에 v23-A0에서 효과적이었던 **Exact Arithmetic Mean Token(512d)**을 Concat(3072d)하여 `Linear(3072→512)`로 압축하는 Residual Shortcut 적용 variant.

| 항목 | 값 |
|---|---|
| Branch / implementation | `codex/v23-bag-mean`, commit `4f984ca` |
| Config | `configs/train_v24_medium_bag_proj_residual.yaml` (`project_structured_tokens: true`, `projection_bottleneck_dim: 64`, `projection_residual_mean: true`, `max_epochs=50`) |
| Run | `20260731_220100`, scratch Medium 50 epochs |
| **완료 상태** | **50-epoch 완주, 정상 수렴 완료** (PID `3332090` 종료) |
| **Best `val_ce_loss`** | **`0.5903045`** @ epoch 41 (**전체 Bag-Collapse 모델 중 최저 기록**, v22 대비 $-0.0043$, v23-A0 대비 $-0.0009$ 추가 개선) |
| Model size | 9.45M trainable params (v22 6.57M + residual 병목 projection ≈2.88M) |
| Best checkpoint | `checkpoints/20260731_220100/v24_medium_bag_proj_residual/epoch=041-val_ce_loss=0.5903.ckpt` |
| Training log | `logs/20260731_220100/v24_medium_bag_proj_residual.out` |
| Checkpoints | `checkpoints/20260731_220100/v24_medium_bag_proj_residual/` |
| Verification | 신규 테스트 `test_bottleneck_projection_with_residual_mean` 통과 |

판정: 50-epoch 학습 완료. v23-A0 / v24-A0 / v24-B0 / v24-B1 4종을 동일 기준 (1,000 pool-400, context 40/80/160/300, v22 paired) 평가 예정이었으나, **아래 "최종 결정"에 따라 이 평가는 실행하지 않고 폐기됨.**

### 🏁 최종 결정 (2026-08-01): v24-B1(residual + bottleneck)을 v24로 확정

> [!IMPORTANT]
> **사용자가 §6/§7에 정의된 1,000-episode paired 비교 프로토콜을 거치지 않고 직접 확정했습니다.** train loss 순위(0.5903 < 0.5912 < 0.5923 < 0.5946 < 0.5976)만으로 결정한 것이며, 합성 val AUROC paired delta나 CI 검증은 수행되지 않았습니다. 이는 §6/§7에 명시된 "overall +0.03 또는 target task +0.05 없으면 폐기" 판정 기준을 적용하지 않은 것입니다. 이후 세션은 이 결정을 재검증 대상이 아니라 확정 사실로 취급하되, 이 문서에 그 근거(train CE 순위만)를 남겨 향후 필요 시 사용자가 재논의할 수 있게 합니다.

- **확정 아키텍처**: v24-B1 구조 = v24 (동일 `architecture_version=24`). `project_structured_tokens: true`, `projection_bottleneck_dim: 64`, `projection_residual_mean: true`.
- **확정 config**: `configs/train_v24_medium_bag_proj_residual.yaml` (top-level 유지).
- **확정 checkpoint**: `checkpoints/20260731_220100/v24_medium_bag_proj_residual/epoch=041-val_ce_loss=0.5903.ckpt`.
- **폐기**: v22(구 기준선), v23-A0(exact mean), v24-A0(learned projection slot 1), v24-B0(bottleneck without residual). config는 `configs/archive/v23_v24_candidates/`로 이관 (v22 config는 ICI 파이프라인이 아직 참조하므로 유지, 상세는 §6).
- **폐기된 계획**: 4종 1,000-episode pool-400 paired 비교 평가, T5-A(typed bag-preserving branch)/T5-B/T5-C — 필요 시 향후 다시 꺼낼 수 있도록 [`history/architecture_v23_candidates.md`](history/architecture_v23_candidates.md)에 보존.
- **미완 항목**: v24용 ICI config 없음. Medium→Hard bridge attribution(T4, §6)은 이번 결정과 별개로 계속 열려 있는 질문임 — 폐기되지 않음.

> [!WARNING]
> **v24-B1은 "label로 bag을 나누는" class-memory 압축을 고치지 않았습니다.** 이 결정의 출발점이었던 문제 제기(context bag을 label별로 묶어 8개 memory token으로 압축하는 구조가 별로라는 지적, `architecture_v23_candidates.md` bottleneck #1)는 `_class_memories`가 여전히 `context_labels == class_index`로 bag을 나눠 pooling하는 구조 그대로입니다 (`src/models/baseline.py:2446-2454`). v23/v24 계열이 바꾼 것은 **bag 내부(40 structured token → 1 token) 압축**뿐입니다. label 기반 분할 자체를 없애는 것은 폐기된 T5-A(typed bag-preserving branch)가 다루던 문제이므로, 그 concern이 여전히 유효하다면 T5-A를 별도로 다시 논의해야 합니다.

### 🗑️ v25 (T5-A) — **폐기 확정** (2026-08-02)

> [!IMPORTANT]
> **v25(T5-A, typed bag-preserving branch)는 2026-08-02 폐기 확정됨.**
> - Medium 1,000-episode pool-400 paired 비교: v25 @context40 우세, @context300 열세,
>   승격 기준(+0.03/+0.05) 전 구간 미달 (§11 "판정 근거 종합").
> - Easy tier: v24-easy 0.9073 vs v25-easy 0.9106, delta +0.0033 — 갈리지 않음 → "아키텍처
>   계열 전체 한계" 가설 강화.
> - config는 `configs/archive/v25_typed_bag/`로 이관, 브랜치 `codex/v25-typed-bag`는
>   **태그 `v25-typed-bag-final`로 보존 후 로컬·원격 모두 삭제 (2026-08-02)**, main을
>   현재 SSOT로 fast-forward. 코드(`baseline.py`의 `typed_bag_*` 분기)는
>   gated 상태로 main에 잔존 (config만 비활성).
> - 잔여 검토 가치: v25의 작은 context(40) 우세는 ICI(~69 fold context)와 관련해
>   향후 T5-B/T5-C 설계 시 참고.

원래 기록 (2026-08-01): label 기반 class-memory 문제("label로 bag을 나누는" class-memory 압축)를 실제로 건드리기 위해, 폐기했던 T5-A(typed, bag-preserving structured context branch)를 v25로 되살렸습니다. v24-B1 확정 이후 새 브랜치 `codex/v25-typed-bag`에서 작업 중이며, `main`/`v24`는 손대지 않았습니다.

- **구조**: 기존 40개 structured token에 학습된 token-type(global/center/spread/rare/tail) + tail-fraction identity embedding을 더하고, v24-B1과 동일한 구조(전용 bottleneck `Linear(512→64)` × 40 + exact mean residual → `Linear(3072→512)`, 단 완전히 별도 가중치)로 bag당 1개 embedding으로 압축합니다. 이 embedding을 **두 번째** `RidgeResidualMetaClassifier`(`typed_bag_classifier`)에 label과 1:1로 정렬된 채로 그대로 통과시켜, `_class_memories`처럼 label별로 뭉쳐서 압축하지 않고 context bag 정체성을 끝까지 보존합니다. 결과 logit은 기존 최종 fusion 뒤에 `covariance_residual_scale`과 동일한 sigmoid-gated, 작게 초기화된 residual(`typed_bag_residual_scale`, 초기값 0.02)로 더합니다. **기존 class-memory 경로(`_class_memories`/`_population_tokens`)는 완전히 그대로 유지** — 두 경로를 나란히 두고 비교하기 위함입니다.
- **원안(T5-A 문서)과의 의도적 차이**: slot-index embedding은 추가하지 않았습니다. slot은 episode마다 새로 도는 spherical k-means cluster id(`_context_spherical_kmeans_anchors`)라서 bag/episode 간 안정적인 의미가 없고, token-type/tail-fraction처럼 "모든 bag이 공유하는 고정된 역할/기준값"이 아니기 때문입니다. 상세 근거는 `src/models/baseline.py`의 `StructuredPopulationMetaClassifier.__init__` 주석 참고.
- **Config**: `configs/archive/v25_typed_bag/train_v25_medium_typed_bag.yaml` (폐기 후 이관.
  원래 `base_config: train_v24_medium_bag_proj_residual.yaml` + `typed_bag_preserving_branch: true`,
  `typed_bag_bottleneck_dim: 64`).
- **architecture_version**: `25` (`typed_bag_preserving_branch=true`일 때 24/23/22보다 우선). v24 checkpoint는 거부됩니다 (신규 가중치 없음).
- **구현 검증**: 전체 unittest **141개 통과** (`786.814s`). 신규 테스트: token-identity layout, typed pooling shape/값(bottleneck 유무), class-memory 경로 불변 확인, v25 architecture_version + checkpoint gating(v25가 v24 거부), BaseModel end-to-end forward+backward finite gradient.
- **학습 실행 중**: Run `20260801_020144`, scratch Medium 50 epoch (v22~v24-B1과 동일 방식 — 사용자 결정으로 warm-start 아님). PID `3612802`(bash 래퍼) / torchrun worker `3612807`. 모델 13.9M trainable params (v22 6.57M + v24 residual bottleneck ~2.88M + 신규 typed-bag 분기 ~4.4M).
  - 학습 로그: `logs/20260801_020144/v25_medium_typed_bag.out`
  - Launcher 로그: `logs/20260801_020144/v25_medium_typed_bag_launcher.out`
  - 체크포인트: `checkpoints/20260801_020144/v25_medium_typed_bag/`
  - Sanity check 통과, epoch 0 정상 진행 확인 (2026-08-01 02:02 KST). 완료까지 약 2~2.5시간 예상 (512 steps/epoch × 50 epoch, ~3 it/s).
- **판정 계획**: 사용자 결정으로 **이번엔 §6/§7 원래 프로토콜대로 1,000-episode paired 합성 평가까지 수행**합니다 (v24 확정 때와 달리 train CE만으로 판단하지 않음). 완료 후 `scripts/evaluate_synthetic.py` → `scripts/compare_predictions.py`로 v24-B1과 paired 비교, overall `+0.03` 또는 target task `+0.05` 기준 적용.
- **다음 Action**: 학습 완료 대기 → best checkpoint로 1,000 pool-400 episode, context 40/80/160/300 평가 → v24-B1과 paired 비교 → 승격/폐기 판정.
- **결과 (2026-08-02)**: 위 프로토콜대로 Medium/Easy 평가 완료, 승격 기준 미달 → **폐기 확정** (§11 "판정 근거 종합", §3 v25 배너).

### ✅ v22 공식 기준선 (2026-07-30 갱신 — **1,000 episode 기준**)

| 항목 | 값 |
|---|---|
| Config / 코드 | `configs/train_v22_medium.yaml`, 커밋 `be36c59` (Cholesky + rank-local 수정 반영) |
| 학습 | 20 epoch, 512 steps/epoch = **10,240 steps** |
| Best `val_ce_loss` | **0.5946** (epoch 13) |
| **합성 val AUROC** | **0.7078**, 95% CI **[0.696, 0.719]** (1,000 episodes / 16,330 query, episode cluster bootstrap) |
| 합성 val Log Loss | 0.6209 |
| 체크포인트 | `checkpoints/20260729_160643/v22_medium_fixed/epoch=013-val_ce_loss=0.5946.ckpt` |
| 예측 파일 | **`predictions/synthetic_v22_baseline_1000ep.pt`** ← 앞으로 비교 대상 |

**task별 AUROC (1,000 episodes, task당 177~213 episodes, CI 폭 0.039~0.049)**:

| task | AUROC | CI 폭 | episodes |
|---|---:|---:|---:|
| combined | 0.8201 | 0.039 | 213 |
| composition | 0.7729 | 0.039 | 204 |
| interaction | 0.6628 | 0.049 | 200 |
| **covariance** | **0.6216** | 0.045 | 206 |
| **state** | **0.6215** | 0.045 | 177 |

> [!IMPORTANT]
> **104 episode로 잰 이전 기준선(0.7466)은 폐기합니다.** 1,000 episode 기준 0.7078이 훨씬 신뢰할 수 있는 추정값입니다.
> **task별 순위도 바뀌었습니다.** 104 episode에서는 covariance(0.6122)가 state(0.6595)보다 명확히 나빠 보였지만, 1,000 episode에서는 **covariance 0.6216 / state 0.6215로 사실상 동률**입니다. interaction도 0.7453 → 0.6628로 크게 내려왔습니다.
> task당 episode가 15~29개였던 것이 원인입니다 — **T3-2를 하기 전 per-task 판단은 신뢰할 수 없었습니다.**

<details><summary>폐기된 104-episode 기준선 (참고)</summary>

AUROC 0.7466 [0.716, 0.776] / Log Loss 0.5943 / `predictions/synthetic_v22_baseline_fixed.pt`
task별: composition 0.8022 / combined 0.8170 / interaction 0.7453 / state 0.6595 / covariance 0.6122
(버그 수정 전 100611 run: 0.7463)
</details>

### 📏 T3-2 결과 (2026-07-30): 필요한 val episode 수

동일 예측을 episode 단위로 서브샘플링해 측정한 CI 폭:

| val episodes | 전체 AUROC CI 폭 |
|---:|---:|
| 104 (기존 기본값) | **0.074** |
| 200 | 0.049 |
| 400 | 0.035 |
| 600 | 0.030 |
| **1,000** | **0.021** |

`1/√n`에 맞게 줄어듭니다. task별로는 1,000 episode에서 task당 ~200개, CI 폭 0.039~0.049입니다.

> [!IMPORTANT]
> **권고: 아키텍처 판정에는 val episode 1,000개를 쓰십시오** (`--val-episodes 1000`).
> - 전체 CI 폭 0.021 → 0.03 이상 차이는 검출 가능
> - **task별 CI 폭 0.045** → task별로 +0.05 이상 차이만 신뢰 가능. **task별로 +0.05 미만을 노리는 실험은 1,000개로도 부족하므로 2,000개 이상 필요**합니다.
> - 기본값 104개는 전체 CI 폭 0.074, task당 ~20 episode로 **어떤 판정에도 부적합**합니다.
>
> 참고로 무작위 104개 서브샘플의 CI는 [0.647, 0.721]로 1,000-episode 추정치(0.7078)를 포함하지만, **기본 val split의 첫 104개는 0.7466**이 나왔습니다. n=104에서는 점추정치가 모집단에서 0.04쯤 벗어나는 일이 예사롭다는 뜻이며(CI 폭 0.074의 절반 수준), 편향이라 단정할 근거는 아니지만 **바로 그 부정확성이 문제**입니다.

### 🎯 T3-1 결과 (2026-07-30): **진짜 약점은 covariance가 아니라 `state`**

세 response effect scale을 **하나의 값으로 통일**해(`--effect-scale`) task를 동일 조건에서 비교했습니다 (400 episodes/run, task당 73~90 episodes):

| effect scale | combined | interaction | covariance | composition | **state** |
|---:|---:|---:|---:|---:|---:|
| 0.4 | 0.6228 | 0.6146 | 0.5714 | 0.5713 | **0.5419** |
| 0.7 | 0.7484 | 0.7266 | 0.6594 | 0.6488 | **0.6115** |
| 1.0 | 0.8226 | 0.7914 | 0.7249 | 0.7200 | **0.6793** |
| 1.4 | 0.8763 | 0.8429 | 0.7786 | 0.7970 | **0.7492** |

> [!IMPORTANT]
> **동일 effect scale에서 covariance는 약점이 아닙니다.** 모든 지점에서 composition과 사실상 동률입니다 (0.4에서 0.5714 vs 0.5713, 1.0에서 0.7249 vs 0.7200).
> **`state`가 네 scale 전부에서 일관되게 최하위**입니다.
>
> 기본 config에서 covariance가 최하위로 보였던 것은 **순전히 생성기 effect scale 차이 때문**이었습니다 (composition 1.40 / state 0.45~1.00 / covariance 0.30~0.80). §5 전략대로 "합성에서 판단"하더라도 **task별 비교는 scale을 통제하지 않으면 무의미**하다는 것이 실증되었습니다.

**민감도(기울기)는 task별로 비슷합니다** (0.4→1.4 구간에서 +0.207~+0.254). 즉 `state`는 신호 크기에 둔감한 것이 아니라 **모든 구간에서 일정하게 뒤처지는 구조적 handicap**을 가집니다.

> [!NOTE]
> **범위 밖 외삽 주의.** 학습 시 범위는 composition 1.40 고정 / state 0.45~1.00 / covariance 0.30~0.80입니다. 따라서 scale 1.4는 state·covariance에게 외삽이고, composition에게만 학습 조건입니다. **state와 covariance가 모두 학습 범위 안인 scale 0.4·0.7이 가장 공정한 비교 지점**이며, 그 두 지점에서도 결론(state 최하위)은 동일합니다.

**→ Tier 2(state)가 이제 명확한 최우선 대상입니다.** 폐기된 Tier 1(covariance)은 애초에 이 아티팩트를 쫓고 있었습니다.

### 🛑 T2-1/T2-2 결과 (2026-07-31): state도 관측 descriptor 헤드룸 없음 → Tier 2 종료

`scripts/diagnose_state_upper_bound.py`, 1,000 validation episodes 중 state 177 episodes / 2,910 query, episode cluster bootstrap:

| access | descriptor | AUROC | 95% CI |
|---|---|---:|---|
| **현재 모델** | end-to-end v22 | **0.6217** | **[0.597, 0.644]** |
| model input | slot center tokens | 0.6210 | [0.598, 0.643] |
| model input | global summary + slot centers | 0.6209 | [0.598, 0.643] |
| model input | global summary | 0.6196 | [0.597, 0.641] |
| observable | raw mean + spread | 0.5578 | [0.535, 0.580] |
| observable | raw mean | 0.5478 | [0.526, 0.570] |
| observable | centered-direction mean | 0.5273 | [0.504, 0.550] |
| oracle mask | responsive population features | 0.9013 | [0.889, 0.913] |
| oracle mask | responsive-cell mean | 0.8819 | [0.868, 0.895] |
| oracle latent | response score | 1.0000 | [1.000, 1.000] |

> [!IMPORTANT]
> **사전 T2-2 판정 기준에 따라 Tier 2를 종료합니다.** 모델이 실제로 받는 global/slot-center 토큰에 context-label ridge probe를 붙여도 현재 모델과 완전히 동률이고, 세포를 특정하지 않는 raw mean 계열은 오히려 모델보다 낮습니다. 기대했던 “state 위치 이동이 bag 평균에 선형 누적되어 큰 관측 헤드룸을 만든다”는 가설은 기각되었습니다.
>
> 0.88~0.90은 `responsive_instance_mask`로 정답 세포를 고른 오라클입니다. **모델이 접근할 수 없는 격차를 구조 변경 목표로 삼지 않습니다.** 이 진단은 절대적인 모든 함수의 수학적 상한을 증명하지는 않지만, 사전에 지정한 관측 descriptor 어디에도 +0.05 이상의 실행 가능한 헤드룸이 없음을 보여줍니다.

재현:
```bash
/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python scripts/diagnose_state_upper_bound.py \
  --val-episodes 1000 --bootstrap 2000 \
  --output logs/v22_state_upper_bound_1000ep.csv
```

산출물: `logs/v22_state_upper_bound_1000ep.csv`; 실행 로그/PID 기록: `logs/20260731_state_upper_bound/`.

검증: 100-episode smoke test 성공, 정식 1,000-episode 실행 성공, 전체 unittest **111개 통과** (670.595초).

### 🔬 covariance 진단 전체 기록 (2026-07-29) — 🛑 **종료된 라인**

> 🗄️ 본문 전체는 [`docs/history/covariance_tier1_diagnosis.md`](history/covariance_tier1_diagnosis.md)로 이동했습니다 (2026-07-31).

### v21 이하 과거 수치 (참고용)

> [!CAUTION]
> **아래 수치들은 v21 이하에서 측정된 것이며 v22 코드로 재현되지 않습니다.**
> v22는 `architecture_version`이 22이므로 **기존 체크포인트는 전부 로드 불가**입니다 (`ModelInterface.on_load_checkpoint` 버전 게이트가 거부).

| Phase | 설명 | 지표 | 비고 |
|---|---|---|---|
| Phase 1 (v21) | Medium 합성 사전학습, full context | `val_ce_loss: 0.5921` | 20 epoch, 10,240 steps — v22 기준선이 재현함 |
| Phase 2 (v21) | Hard 합성 사전학습, full context | `val_ce_loss: 0.6845` | 50 epoch |
| Phase 4 (v21) | ICI 5-fold, Naive retrieval | AUROC 0.5524 / LL 0.7288 | 95% CI [0.424, 0.677] |
| Phase 6b (v21) | ICI 5-fold, Signal-Aware retrieval | AUROC 0.5481 / LL 0.8672 | 95% CI [0.421, 0.674] |
| **Phase 6c (v21)** | **ICI 5-fold, retrieval 없음** | **AUROC 0.5454 / LL 0.7921 / Acc 0.6092** | 95% CI [0.419, 0.674] — v22 기본 구성에 해당 |

**핵심**: 위 세 ICI 구성의 **95% 신뢰구간이 전부 0.5(무작위)를 포함하고 서로 거의 완전히 겹칩니다.** paired bootstrap 승률도 0.52~0.55로 동전 던지기 수준입니다. 즉 n=87 코호트에서 이 차이들은 **검출 불가능한 노이즈**입니다. 상세 근거: [`history/v21_retrieval_investigation.md`](history/v21_retrieval_investigation.md) §4-⑧.

---

## 4. v22 결정: retrieval 완전 제거 (2026-07-29)

> 아카이브됨 (2026-08-02): v22는 폐기된 구버전이라 이 결정 기록은 역사적.
> 제거 근거(3대 가설)·제거 범위·v20 롤백 불가 사유 원문 전체:
> [`docs/history/archive.md`](history/archive.md) §4.

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
| 합성 val (v22 기준선, 1,000 eps) | 1,000 episodes / 16,330 query | 0.7078 | [0.696, 0.719] | **0.021** |

합성 구간이 약 **12배 좁고**(그것도 episode cluster bootstrap이라는 보수적 계산으로), 신호도 훨씬 강합니다(0.71 vs 0.55). 합성은 `episodes_per_epoch`로 더 좁힐 수 있지만 ICI는 87명이 상한입니다. v21의 실패는 **검출력 없는 지표 위에서 아키텍처를 반복 비교한 것**이었고, ICI를 반복해서 보면 그 87명에 과적합됩니다.

**지켜야 할 선**: ICI 결과를 보고 아키텍처를 다시 고치기 시작하면 ICI는 더 이상 테스트 세트가 아닙니다. 또한 ICI에서 ±0.13 이내 변동은 그 자체로 아무 근거가 되지 않습니다.

**합성 평가의 CI는 query가 아니라 episode 단위(cluster bootstrap)로 계산합니다.** 한 episode의 query들은 context set과 생성 파라미터를 공유해 독립이 아니며, query 단위 재표집은 구간을 실제보다 좁게 만듭니다. 실질 표본 크기는 **episode 수**입니다.

---

## 6. 다음 작업 세션 Action Plan — 구조적 변경 및 실험 목록 (아카이브됨)

전문은 [`history/archive.md`](history/archive.md)로 이관했습니다 (T3-3 Hard 기준선, 최종 후보 동결
결정, T4 context-size curve — 모두 완료). Action Plan 자체는 **§21~§26(Musk 실데이터 전이)으로
대체**됐습니다. 아래 3가지만 여전히 유효하므로 여기 남깁니다:

1. **판정 기준(유효)**: 후보 인정은 1,000-episode paired cluster bootstrap에서 overall **+0.03**
   또는 target task **+0.05** 이상일 때만. (`current_experiments.md`도 이 기준을 참조합니다.)
2. **미결 질문(유효)**: **T4 Medium→Hard attribution**은 context-size curve만 완료됐고
   (10→160에서 AUROC 0.5084→0.5737 단조 증가), 잔여 항목(raw-cell 대 40-token information audit,
   token budget sweep, training update scaling)은 **미수행**입니다. v24 동결 및 Musk 전환으로
   현재는 휴면 상태이지만 폐기된 질문은 아닙니다.
3. **하지 말 것(유효)**: ICI 실행 금지(§5) / 오라클 기반 상한을 목표치로 삼기 금지 / effect scale
   정규화 없이 task별 AUROC로 우열 판단 금지. 요약은 §0 "작업 규칙"에도 있습니다.

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
- 브랜치/버전 정책: [`history/branch_structure.md`](history/branch_structure.md)

### 진단 도구 (Tier 1/2 조사에서 구축)

| 스크립트 | 무엇을 답하는가 |
|---|---|
| `diagnose_oracle_covariance_upper_bound.py` | descriptor × relation별 상한. `--val-episodes`로 규모 조절, episode cluster CI 포함. **오라클/관측가능 descriptor 구분 사례 포함** |
| `diagnose_state_upper_bound.py` | state descriptor를 `model_input`/`observable`/`oracle_mask`/`oracle_latent`로 분리하고 현재 모델과 동일 episode cluster CI로 비교 |
| `diagnose_cell_selection.py` | 세포 랭킹 점수가 반응세포를 맞히는가 (AUROC + precision/recall@k, `--probe`로 지도학습 상한) |
| `diagnose_bag_label_selection.py` | **bag 라벨만으로** 반응세포를 찾을 수 있는가 (Tier 1 종료 근거) |
| `diagnose_selection_gain_curve.py` | 선택 품질 → task AUROC 이득 곡선. **구조 변경 전 기대 수익 견적용** |
| `diagnose_covariance_utilisation.py` | 학습된 융합 게이트 값 + relation mode별 분기 성능 |
| `diagnose_oracle_slot_alignment.py` | 반응 component가 슬롯과 정렬되는가 (purity/capture/fragmentation) |

---

## 9. 2026-07-31 세션 핸드오프 — v23/v24 bag collapse family

> 아카이브됨 (2026-08-02): v23-A0/v24-A0/v24-B0/v24-B1 학습 완료 기록 — v24-B1이
> 이후 v24로 확정되어 §3 "최종 결정"에 흡수됨. 원문 전체:
> [`docs/history/archive.md`](history/archive.md) §9.

---

## 10. 2026-08-01 세션 핸드오프 — v24 확정, 평가 계획 폐기

> 아카이브됨 (2026-08-02): v24 확정(사용자 결정, train CE 순위) + 4종 paired 비교
> 폐기 기록 — v24 확정 내용은 §3 "최종 결정"에 보존. 원문 전체:
> [`docs/history/archive.md`](history/archive.md) §10.

---

## 11. 2026-08-02 세션 핸드오프 — v25 Medium 평가 완료(사실상 동률), Easy tier 실험 진행 중

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 12-14. 2026-08-02 세션 — 폴더/문서·config/src·scripts·tests 정리 3단계

> 아카이브됨 (2026-08-02, 핸드오프 정리): checkpoint/log/prediction purge(53GB→3.3GB),
> 구버전 문서·config·스크립트 삭제, src/scripts/tests 참조 무결성 점검 기록. 전문:
> [`history/archive.md`](history/archive.md) §12-14.
>
> **하나만 아직 열려 있음**: §13의 config 삭제가 `test_d_stages_differ_only_in_selected_nuisance`를
> 깨뜨림 (`configs/trainer/learnability_d20.yaml` 삭제, §16에서 발견·미조치) — 상세는
> archive.md §13 경고 참고.

---

## 15. 2026-08-02 세션 마무리 — 정리 3단계 + v25 폐기 확정 + 브랜치 정리

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 16. 2026-08-02 세션 (이어짐) — v26/v27/v29 설계안 검토, 학습 없는 게이트 3종,
## CLS-token pooling(v26) 구현·학습 시작, 제안서 archive

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 17. 2026-08-03 세션 — v26 학습 완료 + CLS attention 진단 프로브 (24-CLS 제안 사전검정)

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 18. 2026-08-03 — E7 재검정: 지도 component-selection 상한 재확인 (Path B 관문)

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 19. 2026-08-03 — 정규화 천장 프로브: 고정 정규화가 천장을 제한하는가 (사용자 가설 검증)

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 20. 2026-08-03 — v24 no-L2 ablation: per-cell L2 정규화 제거 학습 (진행 중)

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 21. 2026-08-03 — Zero-shot Musk (Musk2) MIL 벤치마크 테스트

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 22. 2026-08-03 — 전략 전환: 생성기 개선 (Musk-like easy 데이터) — 가설 판정 완료

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 23. 2026-08-03 — Musk 0.95 로드맵: raw bag-stat token (mean/skew/kurt) 학습 중

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

## 24. 2026-08-03 — Phase 1 IA-MIL (Instance-Attention MIL) — 판정: 음성

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history/archive.md`](history/archive.md) 해당 섹션 참고.

---

## 25. 2026-08-04 — IA-MIL 폐기 + 문서/파일 정리 + 핸드오프

**IA-MIL 폐기 확정**: Phase 1 IA-MIL(`use_instance_attention_mil`, 커밋 `e0620ac`)을 폐기.
1,000-episode 평가 + paired bootstrap + Musk zero-shot 종합:

| 항목 | 결과 | 판정 |
|---|---|---|
| 합성 musklike-easy (`mil`) | AUROC **0.9520** [0.947, 0.957] | 무회귀 ✅ |
| rare 판별 (baseline vs mil) | 0.9492 vs **0.9224**, paired P=1.00 | IA-MIL **유의 열위** ❌ |
| Musk zero-shot (`mil`) | **0.5545** [0.440, 0.672] (vs §22 easy 0.8030) | **큰 회귀** ❌ |

→ 학습 손실(best val_ce 0.2462)은 좋았으나 평가·전이에서 IA-MIL 잔차 채널이 해가 됨.
**`use_instance_attention_mil` 기본 OFF 유지, v24 확정 baseline 유지.**

**폐기·정리 조치 (2026-08-04)**:
- configs → `configs/archive/ia_mil/`: `train_v24_musklike_easy_{mil,rare_baseline,rare_mil}.yaml`;
  → `configs/archive/musklike_easy_levers/`: `train_v24_musklike_easy_{rawstats,mean_token}.yaml`
  (닫힌 지렛대). 루트에는 활성 `train_v24_musklike_easy.yaml`만 유지.
- scripts → `scripts/archive/ia_mil/`: `queue_phase1_rare.sh` (wait_gpu_free 레이스 버그 수정본 포함 —
  스폰 대기 + 완료 블록 + run 선택 인자).
- docs: 해결·폐기된 §11~§24 섹션을 `docs/history/archive.md`로 이관, 스텁+링크만 남김.
  Living 루트는 §0~§10 + §25 (핵심 5문서 유지).
- 체크포인트: IA-MIL 3종(`20260803_173202`, `20260803_190843`, `20260803_212552`)은 물리 보존,
  위 평가 수치로 폐기 표기. 예측: `predictions/synthetic_*_1000ep.pt`, `predictions/musk_v24_musklike_easy_mil.pt`.

**현재 최고 (Musk 0.95 로드맵)**: **지렛대 1(centered 입력 표현, musklike-easy 학습) Musk zero-shot
AUROC 0.822** — `checkpoints/20260803_042852/v24_musklike_easy/` (§22, 아카이브). 지렛대 2(raw
bag-stat)와 지렛대 3(IA-MIL) 모두 음성으로 종료.

**열린 과제 / 다음 Action (사용자 판단)**:
1. **Phase 2**: 166→512 읽기 브리지 (IA-MIL 제외) 진행 여부 → **§26에서 보류 권고** (근거 부재).
2. **ICI**: 여전히 잠금 — v24/v26용 ICI config 작성 + 실행은 사용자 재확인 필요.
   **추가로 §26에서 ICI 블로킹 버그 발견** (cell-축 zero-padding 미마스킹).
3. **Musk 0.95**: 아키텍처 개선 proposal 작성 완료 — [`history/musk095_architecture_proposal.md`](history/musk095_architecture_proposal.md):
   P0(5-seed 앙상블, 즉시) → P2(bag-mean 보존 채널, 합성 선검증) → P1(166→512 읽기 브리지) → P3(단순
   인스턴스 풀링, 사전 게이트). 핵심 근거: `_bag_view` center+L2가 Musk 최고 신호(bag-mean, ridge 0.829)
   를 삭제(0.554) + zero-padding OOD 브리지.
   → **2026-08-04 §26에서 P1/P2 기각, P3 연기. 아래 §26 참조.**

---

## 26. 2026-08-04 — Musk 전이 재진단: P1/P2 기각 + v30(CFMT) 제안

**상태**: 진단 완료(학습 없음, 전부 재현 가능) / 제안 미구현 — 사용자 판단 대기
**문서**: [`history/musk_transfer_diagnosis_v30_proposal.md`](history/musk_transfer_diagnosis_v30_proposal.md)
**신규 스크립트**: `scripts/diagnose_musk_cardinality.py` (체크포인트·학습 불필요)

**핵심 결론**: §25의 proposal은 "입력 표현이 병목"이라는 **방향은 맞지만 메커니즘·수정 방향·검증
경로가 모두 틀렸습니다.** 진짜 병목은 **per-bag centering이 작은 bag을 rank 결핍으로 소멸시키는 것**
이며, 수정은 "raw bag 평균 채널 추가"(P2)가 아니라 **"centering 제거 + per-cell L2 유지"** 입니다.

**측정 1 — Musk bag은 작다**: 102 bag, instances/bag **median 12** (min 1, p25 4, max 1044).
**n≤4가 29개(28%), n≤2가 12개, n=1이 2개.** 반면 학습 분포는 `num_cells: [500,1000]`이고 **에피소드
내 모든 bag이 동일 크기**. bag 크기 단독 AUROC 0.549 [0.436,0.665] → 라벨 누출 아님.

**측정 2 — 학습된 4개 체크포인트 전부가 소형 bag에서 무작위 이하**:

| 체크포인트 | ALL | **n≤4** | 5–10 | 11–34 | n>34 | pearson(prob, log n) |
|---|---|---|---|---|---|---|
| v24 Medium | 0.777 | **0.383** | 0.858 | 0.879 | 0.690 | +0.078 |
| musklike-easy (현 최고) | 0.803 | **0.475** | 0.825 | 0.988 | 0.667 | +0.012 |
| rawstats (§23 음성) | 0.783 | **0.500** | 0.817 | 0.958 | 0.556 | +0.032 |
| IA-MIL (§24 음성) | 0.555 | **0.325** | 0.633 | 0.573 | 0.357 | **+0.327** |

**측정 3 — Musk LOO ridge 천장 (λ 쓸기 + CI): 원 proposal의 순서가 역전**

| 정규화 | 전체 | **n≤4** | n>4 |
|---|---|---|---|
| **center+L2 (현 모델 입력)** | 0.746 | **0.625** [0.21,0.96] | 0.765 |
| center only | 0.763 | **0.475** [0.09,0.85] | 0.769 |
| raw | 0.880 | **0.975** [0.88,1.00] | 0.881 |
| **L2 only (centering 제거)** | **0.911** | 0.892 [0.73,1.00] | 0.911 |
| **pool-z + L2** | **0.921** | **0.967** [0.87,1.00] | 0.913 |

> [!IMPORTANT]
> 소형 bag에서 **정보는 존재하지만(uncentered 0.967~0.975) centering이 파괴합니다(0.475~0.625).**
> 모델 실측(0.475)은 그 구간 표현 천장(0.625)보다도 낮고, **표현을 uncentered로 바꾸면 천장이
> 0.625 → 0.967~0.975로 +0.35 오릅니다.** (소형 구간 CI는 매우 넓음 — 양성 bag 5개.)
> 또한 λ 4개 **전부**에서 **L2-only(0.911) > raw(0.880)** — 원 proposal의 "raw 0.829 >
> L2 0.742"와 **반대**이며, 그 수치는 커밋된 스크립트가 없어 재현 불가였습니다.
> ⇒ **per-cell L2는 해롭지 않고 이롭습니다. 범인은 centering 단독.** P2의 전제가 무너집니다.

**측정 4 — P2는 합성에서 검증 자체가 불가능**: `configs/data/medium.yaml:61 normalize_output: true` →
`synthetic_data.py:543 F.normalize(x, dim=-1)`로 **모든 합성 세포가 단위 노름**(musklike-easy도 상속).
스케일/bag-평균 신호가 **구조적으로 부재** → 게이트 "합성 무회귀 ≥0.94"는 채널이 무시당해도 통과.
§23 raw-stat token의 실패 메커니즘과 동일.

**측정 5 (B0, 신규 실행) — 합성 소형 bag 천장** (`num_cells: [2,16]`, 300 ep):
centered 0.7013 / current 0.6470 / **raw 0.6755** (§19 대형 bag: centered 0.6846 / current 0.6363 /
raw 0.5748). → cardinality가 합성-Musk 선호 충돌의 주원인임은 확인되나 완전 역전은 아님.
**⇒ `l2_only` 단독 적용은 현재 대형 bag 분포에서 합성 천장을 −0.062 낮춰 기각될 것. 반드시
cardinality 샘플링을 먼저 넣어야 함(순서 B2 → B1).**

**측정 6 — §24 IA-MIL 기각은 무효한 게이트**: 생성기는 라벨을 `torch.randint`로 **데이터 이전에**
뽑으므로 **any-positive 과제가 아예 없음**. "rare 5~15%" 실측은 composition/combined(40% 에피소드)
양성 **~75~91% 반응(541~591 cells)**, state/cov(60%)는 **두 클래스 반응 세포 수가 동일**(라벨은
shift 부호). + IA-MIL은 **크기 편향 도입**(+0.327). → 인스턴스 수준 기여가 무용하다는 증거가 아님.

**신규 발견 결함 3건**:
1. **ICI 블로킹 버그**: `src/datasets/base_data.py`가 cell 축을 `target_cells`(1000)까지 **마스크 없이
   zero-padding** → 0 행이 `_bag_view`의 `bag.mean(dim=-2)`·`global_spread`를 오염. **ICI 잠금 해제 전
   필수 수정.**
2. **문서-코드 모순**: `current_architecture.md` 기술 ①은 `(x−μ)/(S+1e-5)`라 명시하나 코드는
   **per-cell L2**로 나눔. 기술 ②의 `d=‖x̃‖₂`는 코드에서 **항등적 1.0**. (문서의 그 공식 `zscore`는
   §19에서 최하위 0.5054로 측정된 변형.)
3. **아카이브 config 로드 불가**: `base_config`가 자기 디렉터리 기준으로 해석되어 `FileNotFoundError`
   — 아카이빙 커밋들의 회귀. **총 12건 확인 → 이번 세션에서 전부 수정·로드 검증 완료** (아래).

**측정 7 (B0b, 신규) — `pool-z`가 양쪽 데이터셋에서 최선**: `diagnose_normalization_ceiling.py`에
`poolz`/`poolz_l2` 변형을 추가하고(기존에 pool 통계를 쓰는 변형은 전체 whitening뿐이었음), 합성·Musk를
**동일 ridge 규약**(`--design-norm scalar`)으로 비교:

| 표현 | 합성 대형(500–1000) | 합성 소형(2–16) | Musk 전체 | Musk n≤4 |
|---|---:|---:|---:|---:|
| **current = `F.normalize(bag−μ_bag)`** | **0.8959** | 0.6470 | 0.759 | **0.500** |
| centered (L2 없음) | **0.9296** | 0.7013 | 0.771 | 0.442 |
| **`poolz` = (bag−μ_ctx)/σ_ctx** | 0.8766 | **0.7363** | **0.874** | **0.900** |
| **`poolz_l2`** | 0.7966 | 0.6754 | **0.912** | **0.967** |
| raw | 0.7984 | 0.6755 | 0.775 | 0.450 |

> [!IMPORTANT]
> **`poolz`(context-pool 대각 표준화)는 현 표현 대비 합성 대형 −0.019 / 합성 소형 +0.089 /
> Musk +0.115** — 거의 무료로 Musk를 크게 얻습니다. `poolz_l2`는 Musk가 더 높지만(0.912, n≤4 0.967)
> 합성 대형에서 −0.099이고, **그 손실은 생성기 인공물**입니다(합성 세포가 `normalize_output: true`로
> 이미 단위 노름 → pool-z가 복원한 스케일을 뒤이은 L2가 다시 버림; 그래서 `poolz_l2`≈`raw`).
> ⇒ **B4(`normalize_output: false`)가 `poolz_l2`의 해금 조건이며 0.95 달성 경로입니다.**
> 또한 매칭 프로토콜에서 `raw`의 소형 값은 0.450으로 나빠 **"raw가 신호를 보존한다"는 §22 서술은
> 프로토콜 의존적**이고, 두 프로토콜에서 일관되게 좋은 것은 pool-z 계열뿐입니다.

**목표: Musk 0.95 유지 (사용자 결정 2026-08-04).** 초판의 "0.90 하향" 권고는 **철회**합니다 —
선형 ridge 천장은 모델의 상한이 아닙니다. 모델은 자기 입력 표현의 선형 천장을 **일관되게 초과**합니다:
합성 Medium 0.6363→**0.708**(+0.072), 합성 musklike-easy 0.8959→**0.9510**(+0.055),
**Musk(zero-shot OOD) 0.746→0.8030(+0.057)**. 초과폭 +0.06을 적용하면 **0.95에 필요한 선형 기반은
≈0.89**이고 `poolz_l2`(0.912)는 이미 그 위입니다. 0.95는 공격적이지만 정합적입니다.

**제안 (v30 CFMT) 실행 순서**: **S1 B1 `poolz` 표현**(`bag_representation` 플래그, 기본 OFF;
Musk 기대 ≈0.93) → **S2 B2 cardinality-faithful 샘플링**(bag별 log-uniform[1,1024], 아키텍처 무변경) →
S3 B3 2차 통계 shrinkage(**1차 적률에는 금지** — 평균 shrinkage는 소형 bag에서 0.967→0.833으로 해로움) →
**S4 B4 생성기 확장**(`normalize_output: false` + any-positive 과제 → `poolz_l2` 해금, 기대 ≈0.97) →
S5 인스턴스 수준 채널(§22의 독립 신호원, 0.95 마진 확보). **층화 보고(n≤4/5–10/11–34/n>34) 항상 필수.**

**ICI: 사용자 지시로 손대지 않음 (2026-08-04).** 위 패딩 결함은 **기록만 하고 수정하지 않으며**,
향후 ICI 잠금 해제 시 반드시 먼저 처리할 항목으로 남깁니다. S6(ICI)는 이 제안 범위에서 제외.

**이번 세션 완료 조치**:
- `scripts/diagnose_musk_cardinality.py` 신규 (cardinality/stratified/decompose/ceiling 4종 리포트,
  `--design-norm {feature,scalar}`) — 위 수치 전부 재현. src/ 변경 없음.
- `scripts/diagnose_normalization_ceiling.py`에 `poolz`/`poolz_l2` 변형 추가.
- **config 재현성 전면 복구 — `base_config`를 가진 65개 config 전부 로드 성공(실패 0)**:
  1) 상대경로 13건 수정: v20(2), v21_retrieval(3), v23_v24_candidates(3), v25_typed_bag(1),
     musklike_easy_levers(2), ia_mil(2).
  2) 커밋 `a5dfcf8`(v18/v19/v20 purge)이 삭제한 `configs/trainer/{learnability_d20,csp_short8}.yaml`
     (각 10줄)을 git에서 **원본 그대로 복원** — 아카이브 v18_v19 17건이 이 두 파일을 참조합니다.
     그 purge 커밋은 "활성 config 10개가 모두 resolve됨"만 확인하고 **아카이브는 확인하지 않았음**.
  3) 커밋 `fbc3ba1`이 삭제한 `train_v21_medium.yaml`을 `configs/archive/v21_retrieval/`로 복원.
  → **부수 효과: `learnability_d20` 결함으로 오래 실패해 온 테스트가 해소되어 unittest 154/154 통과**
     (기존 문서 표기 "153/154 통과, 기존 결함 1건"은 이제 무효 — §23/§24 주석 참고).
  → **교훈(규칙 보강 필요)**: `docs/`·`configs/` 아카이빙 시 **아카이브 대상의 `base_config`/
     `resolve_config_group` 참조도 함께 검증**해야 합니다. 지금까지 모든 아카이빙 커밋이 이를
     누락했습니다. 검증 1줄: `base_config`를 가진 전 config에 `merge_train_config()`를 돌려보기.
- `docs/current_architecture.md` 기술 ①② 를 코드와 일치하도록 정정.

**다음 Action (사용자 판단)**: ① B1을 v30으로 승격해 living 문서 갱신할지, ② B1 형태를 `poolz`
하나로 갈지 `poolz_l2`도 함께 학습할지, ③ P1 완전 보류 여부. 상세: 제안 문서 §5.


---

## 27. 2026-08-04 — 세션 마무리: 사용자 결정 반영, 문서 압축, config 재현성 복구

**상태**: 완료 (코드 미변경 / 학습 없음). 커밋 `1b9ee22`, `45a6466`.

**사용자 결정 2건 반영** (헤더 블록에 확정 기재):
1. **ICI 미개입** — §26의 cell-축 zero-padding 결함은 기록만 유지, 수정 안 함. 제안의 S6(ICI) 범위 제외.
2. **Musk 목표 0.95 유지** — §26 초판의 "0.90 하향" 권고 철회. 근거: 모델은 자기 입력 표현의 **선형
   ridge 천장을 일관되게 초과**한다 — 합성 Medium 0.6363→0.708(+0.072), 합성 musklike-easy
   0.8959→0.9510(+0.055), **Musk zero-shot(OOD) 0.746→0.8030(+0.057)**. 초과폭 +0.06이면 0.95에
   필요한 선형 기반은 ≈0.89이고 `poolz_l2`(0.912)가 이미 그 위이므로 0.95는 정합적.

**§26 초판의 자기 정정 2건** (측정으로 발견):
- B0 표의 대형 열에 §19의 **Medium** 수치를 넣고 소형 열의 **musklike-easy**와 비교했음(서로 다른
  데이터셋). 양쪽을 musklike-easy로 재측정 → 대형 열이 §22 공개값(centered 0.927 / current 0.894)을
  재현하여 프로토콜 건전성 확인.
- 그에 따라 **"B1은 B2 없이 배포 불가"라는 결론 철회**: 올바른 후보 `poolz`의 합성 대형 비용은
  −0.019뿐이므로 B2는 선행조건이 아니라 **증폭 요인**. 실행 순서를 **B1(`poolz`) → B2**로 확정.

**문서 압축 (이번 세션 신규)**:
- **§6(다음 작업 세션 Action Plan, 284줄)을 [`history/archive.md`](history/archive.md)로 이관**하고
  스텁 남김. T3-3 Hard 기준선 / 최종 후보 동결 / T4 context-size curve는 모두 완료됐고 Action Plan
  자체는 §21~§26으로 대체됐습니다. 스텁에는 여전히 유효한 3가지를 명시 보존: ① 판정 기준(overall
  +0.03 / target task +0.05), ② **미결 질문 T4 잔여 항목**(raw-cell 대 40-token audit, token budget
  sweep, training scaling — 휴면이나 폐기 아님), ③ "하지 말 것" 3개.
- `current_status.md` **937 → 약 700줄**.
- **§2 정정**: 제목 "Architecture v22" → **v24 확정**, 핵심기술 ①을 코드와 일치시킴(per-feature std로
  나누지 않고 **per-cell L2**로 사영; $S_i$는 `global_summary` 토큰으로 출력), ②의 선별 기준을
  anchor novelty로 정정, v24 bag projection 항목 추가. → living 문서 5개 상호 일관성 회복.

**config 재현성 전면 복구 — `base_config`를 가진 65개 전부 로드 성공(실패 0)**:
- 상대경로 13건 수정(`base_config`는 config 자기 디렉터리 기준 해석 → 아카이빙 시 조용히 깨짐):
  v20(2) / v21_retrieval(3) / v23_v24_candidates(3) / v25_typed_bag(1) / musklike_easy_levers(2) / ia_mil(2).
- 삭제 파일 3개를 git에서 **원본 복원**: `configs/trainer/{learnability_d20,csp_short8}.yaml`
  (커밋 `a5dfcf8`가 purge, 아카이브 v18_v19 17건이 참조) + `train_v21_medium.yaml`(커밋 `fbc3ba1`,
  `configs/archive/v21_retrieval/`로 복원).
- **부수 효과: `learnability_d20` 결함으로 상시 실패해 온 테스트 해소 → unittest 154/154 통과**
  (65.0s). 이전 문서의 "153/154 통과 + 기존 결함 1건" 표기는 2026-08-04 이전 기록입니다.
- **규칙 신설**: [`agent_handoff.md`](agent_handoff.md) §7-2에 아카이빙/삭제 커밋 전 **전체** config
  로드 검증 절차 추가 (지금까지 모든 아카이빙 커밋이 활성 config만 확인하고 아카이브를 누락했음).

**신규/변경 스크립트** (src/ 모델 코드 변경 없음 → 모델 거동 불변):
- `scripts/diagnose_musk_cardinality.py` (신규): cardinality / stratified / decompose / ceiling 4종
  리포트, `--design-norm {feature,scalar}`. §26의 모든 Musk 수치를 재현합니다.
- `scripts/diagnose_normalization_ceiling.py`: `poolz` / `poolz_l2` 변형 추가.

**다음 Action**: 헤더 "열린 과제" 3건 — ① v30 승격 여부, ② `poolz` 단독 vs `poolz_l2` 병행 학습,
③ P1 보류 여부. B1 구현 시 착수점은 `_bag_view`(`baseline.py:618`)에 `bag_representation` 플래그
(기본 OFF)를 추가하고 pool 통계를 3개 호출 지점에서 주입하는 것 — 상세는
[`history/musk_transfer_diagnosis_v30_proposal.md`](history/musk_transfer_diagnosis_v30_proposal.md) §3.1.

---

## 28. 2026-08-04 — v30 S1/S2 판정 과정 (B1 `poolz_l2`·B2 cardinality) — **아카이브됨, §29로 승격 완료**

v30 B1/B2 실험·판정 전체 기록(S1 `poolz`/`poolz_l2` 음성, S2 B2+B1 양성, paired bootstrap,
B1·B2 상호 필수 근거, 교차 분포 합성 무회귀)은 [`history/archive.md`](history/archive.md) §28로
이관되었습니다. 최종 결론 요약은 헤더 Status와 §29 참고.


---

## 29. 2026-08-04 — **v30 확정 baseline: B1 `poolz_l2` + B2 cardinality-faithful (사용자 승격 결정)**

> **사용자 승격 확정 (2026-08-04)**: §28 결과 (4)/(5)의 승격 요청에 대한 최종 확답.
> **v30 = v24(residual+bottleneck bag proj) + B1(`poolz_l2` 표현) + B2(cardinality-faithful 샘플링)를
> 확정 baseline으로 승격.** v24는 이전 확정 baseline으로 보존합니다.

### 승격 내용

| 항목 | 값 |
|---|---|
| 아키텍처 | v24 그대로 + **`bag_representation: poolz_l2`** (context-pool 대각 표준화 `z = normalize((x−μ_ctx)/σ_ctx)`) |
| 데이터 분포 | **B2**: `num_cells: [1,1024]` + `num_cells_log_uniform: true` (log-uniform — uniform은 n≤12에 1%뿐) |
| 확정 config (medium 기준) | **`configs/train_v30_medium_bag_proj_residual.yaml`** (신규, 미학습) |
| Musk 경로 (S2 실측 승리) | `configs/train_v30_cardinality_poolz_l2.yaml` (musklike-easy + B2 + `poolz_l2`) |
| B1·B2 상호 필수 | B1 단독(S1) 음성(구간 교환), B2 단독(legacy) 학습 불가(NaN) — §28 결과 (5) |

### 승격 근거 요약 (전부 §28에 실측)

1. **합성 무회귀**: B2+`poolz_l2` 체크포인트를 원래 대형 bag 분포에서 평가 → **0.9483** (v24 0.9510, Δ−0.003).
2. **Musk zero-shot 0.8539** [0.774,0.925] — 종전 최고 0.822 경신.
3. **구간 균형**: n≤4 0.475→**0.800**, n>34 0.667→**0.698** (한 구간을 팔아 다른 구간을 사지 않음).
4. **paired bootstrap** (같은 102 bag): 소형 Δ+0.325 CI [+0.092,+0.567] **0 제외** (P=0.997),
   대형 Δ+0.001 (P=0.504) — 개선이 예측 지점에 정확히 국소화.
5. **크기 편향**: corr(prob, log n) +0.012 → +0.059 (거의 무편향), log loss 0.5439→0.4746.

### 승격 방식 — config + living 문서 수준 (코드 기본값은 의도적으로 유지)

`bag_representation` 코드 기본값은 **`legacy` 유지**, `architecture_version` 범프도 하지 않았습니다:

- **기본값을 `poolz_l2`로 플립하면** 기존 `_bag_view` 직접 호출 테스트(poolz 모드는 pool 통계 없이
  raise)와 `bag_representation`을 고정하지 않은 v24/ICI config가 **조용히 깨집니다.**
- **`architecture_version`을 30으로 올리면** `model_interface.py:38` 가드가 이미 학습된 v30
  체크포인트(arch 24로 저장됨) 로드를 **거부**해 재평가가 불가능해집니다.
- v24 확정(v24-B1)도 코드 기본값을 바꾸지 않고 config+문서로 확정한 것과 동일한 관례입니다.

따라서 새 작업은 **확정 config를 base로 상속**하거나 `bag_representation: poolz_l2` + B2를 명시해야
합니다. v24 재현은 `configs/train_v24_medium_bag_proj_residual.yaml`로 그대로 가능합니다.

### 다음 Action

1. **v30 medium 기준 재학습 진행 중 (PID 2899744)** — `train_v30_medium_bag_proj_residual.yaml`로 50 epoch 진행 중, 참조 수치 확보 예정.
2. **n>34 대형 bag 희석 결함 규명 및 v31 아키텍처 수술 완료**:
   * `diagnose_tail_dilution.py`로 n > 34 희석 메커니즘 정량 입증.
   * `baseline.py`에 Absolute Top-K Tail Token (`absolute_tail_ks`) 추가 및 `synthetic_data.py`에 `any_positive_sparse` 과제 추가 완료.
   * 신규 config: `configs/train_v31_absolute_topk_tail.yaml` 작성 완료.
3. ICI 잠금 유지.

---

## 30. 2026-08-04 — v31 CCTS 구현·학습 (아카이브됨)

CCTS 구현과 50-epoch 학습 기록은 후속 CCER-v2로 완전히 대체되어
[`history/archive.md`](history/archive.md#30-2026-08-04--v31-ccts-cardinality-calibrated-tail-scan-아키텍처-구현-unit-test-통과-및-훈련-구동)로 이동했다.

---

## 31. 2026-08-05 — v31 CCTS Musk 평가·진단 (아카이브됨)

CCTS Musk `0.8376`, 대형 bag `0.6032` 결과와 구현 결함 재분류 기록은
[`history/archive.md`](history/archive.md#31-2026-08-05--v31-ccts-50-epoch-완주-musk-zero-shot-평가-및-대형-bag-정체-정밀-분석)로 이동했다.

---

## 32. 2026-08-05 — v31 CCER-Lite 구현·학습 (아카이브됨)

CCER-Lite 구현과 학습 기록은 contribution이 `~1.4e-4`로 사실상 비활성임을 확인한 뒤
[`history/archive.md`](history/archive.md#32-2026-08-05--v31-ccer-lite-구현-및-1차-학습-시작)로 이동했다.

---

## 33. 2026-08-05 — v31 CCER-v2 아키텍처 구현 완료 (학습 미시작)

**상태**: CCER-Lite 실패 분석을 반영한 독립 아키텍처 구현과 회귀 검증 완료. 기존
CCER-Lite와 해당 체크포인트는 재현성을 위해 그대로 보존했다.

### 변경점

- class prototype은 bag projection 이후 memory가 아니라 **projection 전 aligned
  slot-center**에서 직접 만든다.
- query cell encoder와 support prototype encoder를 기존 rare branch와 완전히 분리했다.
- evidence route는 cardinality와 무관한 `Top-1`, 미세 population용 `Top-4`, dense
  shift용 `mean` 세 경로다.
- router는 어떤 route도 제거할 수 없도록 총 `0.30`의 route floor를 갖는다.
- 별도 null gate를 제거하고 class-centered logits만 사용한다.
- 새 output head를 0으로 초기화해 v30 checkpoint 주입 직후 전체 logits를 정확히
  보존한다. 첫 update에는 output head가 열리고 이후 encoder/router로 gradient가
  전달되는 staged adaptation이다.
- optimizer는 CCER-v2에 base LR, v30 backbone에 `0.05x` LR을 사용한다.
- Lightning resume와 분리된 `--init-checkpoint` weight-only 초기화를 추가했다.

### 설정과 검증

- Config: `configs/train_v31_ccer_v2.yaml` (v30 data/task distribution 그대로 상속,
  20 epochs, v30 best warm-start).
- 신규/기존 CCER targeted tests: **6 tests, OK**.
- 실제 v30 best checkpoint load: 신규 tensor 21개와 architecture marker만 missing,
  unexpected key 0개.
- 동일 synthetic episode에서 warm-start v30과 v31-v2의 초기 logit 최대 차이:
  **`0.0`**.
- route floor, Top-1 background invariance, dense/single-episode path 동치, zero-init
  gradient staging을 검증했다.

학습은 아직 시작하지 않았다. architecture correctness가 확보된 이 상태를 고정한 뒤
단일 20-epoch 방향성 run을 실행하면 된다.

---

## 34. 2026-08-05 — v31 CCER-v2 20-epoch 학습 시작

**상태**: 완료. 최종 수치와 artifact 검증은 §35로 통합했다.

- Run time: `20260805_123630`
- Config: `configs/train_v31_ccer_v2.yaml`
- Log: `logs/20260805_123630/v31_ccer_v2.out`
- Checkpoints: `checkpoints/20260805_123630/v31_ccer_v2/`
- Initialization: v30 best
  `epoch=048-val_ce_loss=0.4442.ckpt` weight-only load, 신규 tensor 21개 유지.
- GPU: B200 device 0, single process.
- 시작 검증: CUDA 연결, sanity validation 2/2 통과, epoch 0 진입 확인.

초기 detached launcher가 worker 생성 전에 종료되어 checkpoint/log를 만들지 않았고, 같은
run 경로를 foreground persistent session으로 재시작했다. 중복 학습 없이 20 epochs를
정상 완주했다.

---

## 35. 2026-08-05 — CCER-v2 구현·검증·20 epoch 학습 완료

**상태**: architecture-first 변경과 seed 42 방향성 학습 완료. 합성/Musk 평가는 대기 중이며
v30은 계속 확정 baseline이다.

### 구현과 안전 조건

- `src/models/baseline.py`: projection 전 class-slot prototype, 독립 support/query encoder,
  class-centered `Top-1/Top-4/mean` evidence router, 총 route floor `0.30`, zero-init output head.
- `scripts/train.py` / `src/utils/utils.py`: Lightning resume와 분리된 `--init-checkpoint`
  weight-only warm-start.
- `src/modules/model_interface.py`: CCER-v2 base LR + v30 backbone `0.05x` param groups와
  contribution/route diagnostics logging.
- 실제 v30 best load에서 신규 CCER-v2 tensor 21개만 missing, unexpected 0개였고 동일
  episode 초기 logit 최대 차이는 정확히 `0.0`이었다.
- targeted CCER tests 6개와 ModelInterface tests 14개 통과. `git diff --check`와 변경 Python
  파일 compile도 통과했다.

### 학습 결과와 artifact 생존 확인

- Run / seed / epochs: `20260805_123630` / `42` / `20`.
- Log: `logs/20260805_123630/v31_ccer_v2.out`.
- Checkpoints: `checkpoints/20260805_123630/v31_ccer_v2/`.
- `last.ckpt`와 epoch 18/19 checkpoint가 `2026-08-05 13:04~13:06`, 각
  `118,523,863` bytes로 생성됐고 log는 `max_epochs=20 reached`를 기록했다.
- Best: epoch 18, `val_ce_loss=0.443786`, `val_loss=0.477892`, `val_auroc=0.825494`.
- Best validation CCER-v2 diagnostics: logit std `0.051835`, residual scale `0.141363`,
  route entropy `0.651253`. CCER-Lite의 `~1.4e-4` contribution과 달리 새 branch가 실제로
  활성화됐지만 성능 승격 여부는 아직 판단할 수 없다.

### 다음 Action (완료)

1. epoch 18 best checkpoint로 기존 고정 protocol의 synthetic 평가를 실행한다. (완료: AUROC 0.8514)
2. 같은 checkpoint로 Musk overall 및 4개 cardinality band를 평가한다. (완료: AUROC 0.8470)
3. 평가 수치 기반 승격 여부 판단: v30 baseline 유지 결정.

---

## 36. 2026-08-05 — v31 CCER-v2 Epoch 18 합성/Musk 평가 완료 (v30 Baseline 유지)

**상태**: Epoch 18 best checkpoint (`epoch=018-val_ce_loss=0.4438.ckpt`)에 대한 합성 1,000 episode 및 Musk zero-shot 평가를 완료했다. 수치상 v30 baseline 대비 승격 기준을 충족하지 못하므로 **v30 확정 Baseline을 지속 유지**한다.

### 1. 실측 수치 요약

- **Synthetic Validation (1,000 episodes, 16,330 queries)**:
  - Overall AUROC: `0.8514` [95% CI 0.840, 0.862] (Log loss: `0.4650`)
  - Per-task AUROC: `combined` 0.9514, `composition` 0.8824, `state` 0.8194, `interaction` 0.8125, `covariance` 0.7164
  - Saved predictions: `predictions/synthetic_v31_ccer_v2.pt`
- **Musk Zero-Shot Meta-Test (102 bags)**:
  - Overall AUROC: `0.8470` [0.765, 0.919] (Accuracy: 0.7941, Balanced acc: 0.7894, Log loss: 0.4818)
  - Saved predictions: `predictions/musk_v31_ccer_v2.pt`
  - Cardinality Stratification:
    - `ALL`: `0.847` [0.76, 0.92] (v30 baseline: `0.854`)
    - `n <= 4`: `0.792` [0.53, 0.98] (v30 baseline: `0.800`)
    - `5 .. 10`: `0.842` [0.64, 0.99] (v30 baseline: `0.833`)
    - `11 .. 34`: `0.933` [0.81, 1.00] (v30 baseline: `0.958`)
    - `n > 34`: `0.698` [0.37, 0.98] (v30 baseline: `0.698`)

### 2. 판정 및 결론

- CCER-v2는 residual scale `0.141` 및 독립 support/query encoder 구조를 통해 CCER-Lite의 브랜치 묻힘 현상을 완벽히 해결했으나, 최종 Musk AUROC `0.8470` 및 합성 AUROC `0.8514`로 v30 baseline (Musk `0.854`)을 능가하지 못함.
- 수칙과 사전 승격 기준에 따라 **v30 S2가 확정 Baseline을 지속 유지**하며, CCER-v2는 실험 후보 기록으로 보존한다.
- ICI 잠금은 유지한다.
