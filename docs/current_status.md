# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-01 13:20:00 KST`
**Status**: **v24 확정 (사용자 결정, 2026-08-01)**. 4종 candidate 중 v24-B1(residual + bottleneck bag projection)을 최종 아키텍처로 확정하고 나머지(v22 기준선, v23-A0, v24-A0, v24-B0)는 폐기. 원래 계획했던 1,000-episode 4종 paired 합성 비교 평가는 **실행하지 않고 폐기**. `main`/`v24` 브랜치를 `codex/v23-bag-mean` 최종 커밋으로 fast-forward. ICI 잠금은 유지 — v24용 ICI config가 아직 없음.
**Read first if you are picking this up**: §3 "최종 결정 (2026-08-01)", §3의 v24-B1 완료 기록, §6 Action Plan(갱신), §9 세션 핸드오프.
**Branches**: `main` = `v24` = `codex/v23-bag-mean` 최종 커밋 (fast-forward 완료) / `v22`(구 기준선, 참조용 보존) / `v19` / `v18`(다른 서버) — 구조: [`history/branch_structure.md`](history/branch_structure.md)
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)
**Architecture Version**: **`24` 확정** (`project_structured_tokens: true, projection_bottleneck_dim: 64, projection_residual_mean: true`). `22`/`23`은 폐기된 구버전.
**Purpose**: 연구실 / 집 / 노트북 3개 작업 환경 간 대화 기록 비동기화 문제를 해결하기 위한 Single Source of Truth (SSOT) living document.

---

## 0. 30초 요약 — 새 세션은 여기부터

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

### ✅ v23-A0 exact bag-mean: 50-epoch 완료 (2026-07-31)

각 bag의 `1 global + 36 slot-statistic + 3 tail = 40` structured token을
exact arithmetic mean 1개로 압축했습니다. Context와 query 모두 같은
mean을 사용하고, class-memory에는 `bags × 40`이 아니라 bag당 1개가
입력됩니다. 기존 v22 동작은 기본값 `false`로 보존되며 활성화한
checkpoint는 `architecture_version=23`입니다.

| 항목 | 값 |
|---|---|
| Branch / implementation | `codex/v23-bag-mean`, commit `8edd7c1` |
| Config | `configs/train_v23_medium_bag_mean.yaml` |
| Run | `20260731_155635`, scratch Medium 20 epochs / 10,240 steps |
| Completion | launcher가 `training completed successfully` 기록; PID 종료 |
| Best checkpoint | `checkpoints/20260731_155635/v23_medium_bag_mean/epoch=019-val_ce_loss=0.5934.ckpt` |
| Last checkpoint | `checkpoints/20260731_155635/v23_medium_bag_mean/last.ckpt` |
| Best `val_ce_loss` | **0.5933738** @ epoch 19 |
| Epoch 19 val AUROC | 0.7322822 (104-episode training val; 최종 판정용 아님) |
| Highest run val AUROC | 0.7379618 @ epoch 8 (동일한 작은 104-episode val) |
| Logs / metrics | `logs/20260731_155635/v23_medium_bag_mean.out`; `logs/v23_medium_bag_mean/version_0/metrics.csv` |
| Verification | 전체 unittest **123개 통과** (`696.503s`) |

20-epoch 경계에서 validation CE와 total loss가 모두 run best였고,
epoch 16→19 CE가 `0.59550→0.59452→0.59369→0.59337`로 연속 개선되어
추가 수렴 여지를 확인합니다. `last.ckpt`의 model/optimizer/scheduler/global
step을 복원하여 총 50 epochs까지 연장했습니다.

| 50-epoch 재개 항목 | 값 |
|---|---|
| Config commit | `4d784dc` (`episode_batch_size=8`, `shape_group_size=8`, `max_epochs=50`) |
| Resume source | `checkpoints/20260731_155635/v23_medium_bag_mean/last.ckpt` |
| Active epoch range | epoch 20~49 (추가 30 epochs) |
| PID | `2671747` (종료) |
| **완료 상태** | **`max_epochs=50` 도달, 정상 종료** |
| **50-epoch best** | **epoch 43 `val_ce_loss=0.5912154`, `val_auroc=0.7383`** |
| Best checkpoint | `checkpoints/20260731_v23_bag_mean_50e_resume/v23_medium_bag_mean_50e/epoch=043-val_ce_loss=0.5912.ckpt` |
| Training log | `logs/20260731_v23_bag_mean_50e_resume/v23_medium_bag_mean_50e.out` |
| Metrics | `logs/v23_medium_bag_mean/version_1/metrics.csv` |

50-epoch 연장으로 훈련 val CE가 `0.59337 → 0.59122`로 개선됐고
v22 공식 best `0.5946`보다 `0.0034` 낮습니다. 104-episode val AUROC는
분산이 커서 승패 판정에 쓰지 않습니다. **v23 50-epoch 평가(1,000 pool-400,
context 40/80/160/300, v22 paired)는 아직 실행 전입니다** — 사용자 지시
대기 중. ICI는 계속 잠급니다.

### ✅ v24-A0 learned bag projection: 50-epoch 완료 (2026-07-31)

exact mean(v23) 대신 **learned linear projection**으로 bag을 1토큰으로
압축합니다. Slot을 12→1로 줄여 bag당 `1 global + 3 slot-statistic + 3 tail
= 7` 토큰을 만들고, concat(`7×512=3584`) → `Linear(3584, 512)` → bag당
512-d 1토큰을 생성합니다. 활성화 checkpoint는 `architecture_version=24`.

| 항목 | 값 |
|---|---|
| Branch / implementation | `codex/v23-bag-mean`, commit `26b2b27` |
| Config | `configs/train_v24_medium_bag_proj.yaml` (`project_structured_tokens: true`, slot 1 / density slot 1, `max_epochs=50`) |
| Run | `20260731_182755`, scratch Medium 50 epochs |
| **완료 상태** | **`max_epochs=50` 도달, 정상 종료** |
| **Best `val_ce_loss`** | **0.5976237** @ epoch 45 (마지막 epoch 49: 0.59819) |
| Best val AUROC | 0.7339473 (104-episode training val; 최종 판정용 아님) |
| Best checkpoint | `checkpoints/20260731_182755/v24_medium_bag_proj/epoch=045-val_ce_loss=0.5976.ckpt` |
| Model size | 8.4M trainable params (v22 6.57M + bag_token_projection ≈1.8M) |
| Training log | `logs/20260731_182755/v24_medium_bag_proj.out` |
| Metrics | `logs/v24_medium_bag_proj/version_0/metrics.csv` |
| Verification | 신규 테스트 5개 포함 `test_base_model` + `test_model_interface` **76개 통과** (`553.453s`) |

> [!NOTE]
> 훈련 val CE `0.5976`은 v22(0.5946)와 v23-A0(0.5912)보다 높습니다. slot을
> 1개로 줄인 정보 손실 영향으로 보이며, 최종 판정은 1,000 pool-400 episodes
> context `40/80/160/300` paired 비교로만 합니다.

판정 계획: v23/v24 둘 다 1,000 pool-400 episodes, context `40/80/160/300`에서
v22(`predictions/v22_medium_baseline_pool400_curve/`)와 paired episode-cluster
bootstrap 비교. Overall `+0.03` 또는 target task `+0.05`가 없으면 폐기.
ICI는 잠금 유지.

### ✅ v24-B0 per-token bottleneck projection: 50-epoch 완료 (2026-07-31)

v24-A0가 slot 1개로 정보를 잃는 문제를 해결하기 위한 variant. **12 slot 유지**
(40 tokens) + 토큰별 전용 `Linear(512→64)` 40개 적용 → concat(40×64=2560) →
`Linear(2560→512)` → bag당 512-d 1토큰. 직결 40×512→512(~10.5M) 대신
병목으로 파라미터를 ~2.62M로 제한.

| 항목 | 값 |
|---|---|
| Branch / implementation | `codex/v23-bag-mean`, commit `b2fb9d0` |
| Config | `configs/train_v24_medium_bag_proj_bottleneck.yaml` (`project_structured_tokens: true`, `projection_bottleneck_dim: 64`, 12 slot, `max_epochs=50`) |
| Run | `20260731_201252`, scratch Medium 50 epochs |
| **완료 상태** | **`max_epochs=50` 도달, 정상 종료** (PID `3033073` 종료) |
| **Best `val_ce_loss`** | **0.5923204** @ epoch 46 (v22 baseline 0.5946 대비 -0.0023, v24-A0 0.5976 대비 -0.0053 개선) |
| Model size | 9.2M trainable params (v22 6.57M + 병목 projection ≈2.62M) |
| Best checkpoint | `checkpoints/20260731_201252/v24_medium_bag_proj_bottleneck/epoch=046-val_ce_loss=0.5923.ckpt` |
| Training log | `logs/20260731_201252/v24_medium_bag_proj_bottleneck.out` |
| Checkpoints | `checkpoints/20260731_201252/v24_medium_bag_proj_bottleneck/` |
| Verification | 신규 테스트 4개 포함 `test_base_model` + `test_model_interface` **80개 통과** (`578.291s`) |

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
| 합성 val (v22 기준선, 1,000 eps) | 1,000 episodes / 16,330 query | 0.7078 | [0.696, 0.719] | **0.021** |

합성 구간이 약 **12배 좁고**(그것도 episode cluster bootstrap이라는 보수적 계산으로), 신호도 훨씬 강합니다(0.71 vs 0.55). 합성은 `episodes_per_epoch`로 더 좁힐 수 있지만 ICI는 87명이 상한입니다. v21의 실패는 **검출력 없는 지표 위에서 아키텍처를 반복 비교한 것**이었고, ICI를 반복해서 보면 그 87명에 과적합됩니다.

**지켜야 할 선**: ICI 결과를 보고 아키텍처를 다시 고치기 시작하면 ICI는 더 이상 테스트 세트가 아닙니다. 또한 ICI에서 ±0.13 이내 변동은 그 자체로 아무 근거가 되지 않습니다.

**합성 평가의 CI는 query가 아니라 episode 단위(cluster bootstrap)로 계산합니다.** 한 episode의 query들은 context set과 생성 파라미터를 공유해 독립이 아니며, query 단위 재표집은 구간을 실제보다 좁게 만듭니다. 실질 표본 크기는 **episode 수**입니다.

---

## 6. 다음 작업 세션 Action Plan — 구조적 변경 및 실험 목록

> [!IMPORTANT]
> 모든 판단은 **합성 val**에서 하고 ICI는 손대지 않습니다 (§5). 후보마다
> `scripts/evaluate_synthetic.py` → `scripts/compare_predictions.py`로 기준선
> (`predictions/synthetic_v22_baseline_1000ep.pt`, AUROC 0.7078 [0.696, 0.719])과
> paired cluster bootstrap 비교할 것.

### ✅ T3-3 v22 Hard 기준선 완료 (2026-07-31)

- Run: `v22_hard_baseline`, 50 epoch / 25,600 optimizer steps, 정상 종료
- Best checkpoint: `checkpoints/20260731_035538/v22_hard_baseline/epoch=044-val_ce_loss=0.6839.ckpt`
- Best `val_ce_loss`: **0.6839** (epoch 44), v21 Phase 2 참고값 0.6845와 사실상 동일
- 1,000-episode 평가: **AUROC 0.5483 [0.538, 0.558]**, Log Loss 0.6920, 15,373 query
- 예측: `predictions/synthetic_v22_hard_baseline_1000ep.pt`
- 로그: `logs/20260731_035538/v22_hard_baseline.out`

| task | AUROC | episodes |
|---|---:|---:|
| combined | 0.6095 | 213 |
| composition | 0.5614 | 204 |
| interaction | 0.5298 | 200 |
| state | 0.5167 | 177 |
| covariance | 0.5103 | 206 |

> [!IMPORTANT]
> Hard regime에서는 전체도 0.55에 불과하고 state/covariance는 거의 무작위입니다. Medium에서 종료한 Tier 1/2 방향을 다시 열 관측 근거가 생긴 것은 아닙니다. v21의 `val_ce_loss`를 재현했으므로 v22 구현 이상보다는 문제 난이도 자체의 한계로 읽는 것이 타당합니다.

### ✅ 최종 후보 동결 여부 결정 — **완료 (2026-08-01)**

사전에 계획한 Tier 1~3 합성 작업이 모두 끝났고, 이어서 진행한 v23/v24 bag-collapse family도 4종 모두 학습을 완주했습니다. 원래 계획은 이 4종을 1,000-episode paired 비교로 검증한 뒤 동결하는 것이었으나, **사용자가 이 비교를 건너뛰고 v24-B1(residual + bottleneck)을 직접 v24로 확정했습니다** (§3 "최종 결정" 참고). v22 Medium baseline은 폐기되었고 최종 후보는 v24입니다. **ICI는 여전히 잠금 상태** — v24용 ICI config가 없고, 이번 결정이 §7 평가 프로토콜을 통과한 것이 아니므로 ICI 해제는 별도 확인이 필요합니다.

### ➡ 다음 우선순위: T4 Medium→Hard 성능 붕괴 attribution

Hard는 Medium과 비교해 9개 축이 동시에 바뀝니다: class separation `0.5~1.4→0.2~0.8`, rare fraction `2~8%→0.5~3%`, rare probability `0.15→0.25`, donor shift `0.35→0.70`, component shift `0.12→0.25`, noise `0.01→0.05`, latent dim `32→64`, cells `500~1000→500~1500`, 그리고 batch/accumulation·covariance rank가 달라집니다. 현재 결과만으로는 어느 변화가 AUROC `0.7078→0.5483` 붕괴를 만들었는지 알 수 없습니다.

**세 가지 표본·압축 병목 가설 (2026-07-31)**

1. **Bag 내부 압축 병목 — 가능성 높음.** “40 instances”가 아니라 1 global + 12 slots×(center/spread/rare) + 3 tails의 **40개 512-d token**이며, 별도 slot metadata/covariance 경로도 있으므로 단순 1000→40 비율만으로 과압축이라 단정할 수는 없습니다. 하지만 Hard 반응세포는 0.5~3%뿐이고, 기존 sparse/slot 선택이 무작위·fragmented였으므로 label-relevant 소집단이 요약 과정에서 사라질 가능성은 높습니다. 판별은 같은 context에서 raw-cell observable distribution descriptor와 40-token descriptor를 직접 비교하고, 이후 12/24/48 slot·tail budget scaling으로 합니다.
2. **Episode 내부 context bag 부족 — 가능성 높음.** 50~100 bags에서 최대 20%를 query로 빼면 context는 대략 40~80 bags, 클래스당 약 20~40개입니다. 매 episode마다 manifold와 response direction이 새로 뽑히므로 이 수십 개 label로 512-d 이상의 episode-specific 관계를 다시 추정해야 합니다. 또한 context token은 class당 8 memory token으로 다시 압축되지만 global/ridge 분기는 원 bag 통계를 직접 쓰므로 두 경로를 분리해 봐야 합니다. 판별은 같은 query를 고정한 paired context-size curve(10/20/40/80/160)로 합니다.
3. **전체 training episode/step 부족 — 가능성 중간 이하, 아직 배제 불가.** Training은 seed 고정 dataset 재사용이 아니라 온라인 non-repeating stream입니다. Medium은 약 81,920 episodes/10,240 optimizer steps, Hard는 약 204,800 episodes/25,600 steps로 episode 절대량은 이미 큽니다. Hard CE도 epoch 30의 0.6845에서 epoch 44의 0.6839로만 개선되어 거의 plateau입니다. 따라서 우선순위는 낮지만, checkpoint별 1,000-episode curve와 1×/2×/4× update scaling으로 최종 확인합니다. Episode 수와 optimizer update 수는 별도로 통제합니다.

**판별 순서**: context-size curve(가장 저렴) → raw-cell 대 40-token information audit → token budget sweep → 마지막으로 training scaling. Context를 늘려도 평평하면 bag 표현 병목, raw-cell descriptor만 높으면 압축 병목, 둘 다 충분한데 checkpoint 성능만 낮으면 meta-training/optimizer 병목으로 판정합니다.

**Context-size curve 완료 기록 (2026-07-31 10:21~10:42 KST)**
- 고정: Hard best checkpoint, 동일 episode, 클래스별 query 10개(총 20), nested balanced context
- Context: 총 10/20/40/80/160 bags. 10~80은 학습 범위, 160은 의도적 OOD 상한
- 완료: 1,000 episode 중 991 사용, 9 skip; context 크기당 query 19,820개

| 총 context bags | AUROC (episode-cluster 95% CI) | Log loss |
|---:|---:|---:|
| 10 | 0.5084 [0.500, 0.516] | 0.7170 |
| 20 | 0.5193 [0.511, 0.528] | 0.7047 |
| 40 | 0.5312 [0.523, 0.539] | 0.6967 |
| 80 | 0.5505 [0.542, 0.559] | 0.6900 |
| 160 | 0.5737 [0.565, 0.583] | 0.6835 |

- **판정**: context 10→80에서 `+0.0421`, 80→160에서 추가 `+0.0232`, 전체 10→160에서 `+0.0653`으로 단조 증가한다. 따라서 episode 내부 labeled bag 부족은 실제 주요 병목이다.
- 200회 paired episode bootstrap 교차검증에서 `P(40 > 80)=0.00`, `P(80 > 160)=0.00`이었다. 즉 두 핵심 증가는 동일 episode/query 기준으로도 방향이 일관됐다.
- 80의 0.5505가 기존 Hard 기준선 0.5483과 정합하므로 통상 Hard 평가의 context 규모를 재현한다.
- 160은 학습 분포 밖 상한인데도 개선이 계속되지만 AUROC는 0.5737에 그친다. 따라서 context 부족만으로 Hard 붕괴 전체를 설명할 수 없고, bag 내부 정보 손실/압축 병목을 다음으로 분리한다.
- 실행 PID(종료): `2499444`; 로그: `logs/20260731_context_curve/context_curve.out`
- Summary: `logs/v22_hard_context_curve_1000ep.csv`; predictions: `predictions/v22_hard_context_curve/context_{10,20,40,80,160}.pt`
- Smoke: 2 episodes 전체 경로 성공; unit tests 3개 통과 (balanced/disjoint/nested 계약); 전체 unittest 114개 통과 (675.411초)

**Large-context 학습 용량 점검 (B200 183,359 MiB)**
- 현재 Hard 학습: `episode_batch_size=4`, `accumulate_grad_batches=2`, GPU 1장, BF16 mixed. 따라서 forward/backward당 4 episodes, optimizer update당 effective 8 episodes.
- Episode는 50~100 bags, training query는 5~12 bags이므로 실제 학습 context 범위는 38~95 bags.
- 현재 v22 6.57M 모델, batch 4, 1,500 cells/bag의 보수적 FP32 forward/backward 벤치:

| 총 bags/episode | Peak allocated VRAM | Step time |
|---:|---:|---:|
| 100 | 47,043.9 MiB | 0.135 s |
| 172 | 80,849.2 MiB | 0.213 s |
| 220 | 103,384.9 MiB | 0.273 s |

- 벤치는 bags 절반을 query로 둬 실제 training query 5~12보다 meta 경로가 더 무거운 보수적 조건이다. 따라서 `context 160 + query 최대 12 = 총 172 bags`는 현재 batch 4를 유지해도 충분한 여유가 있다.
- 총 220 bags도 단일 model step은 통과했으므로 context 약 200까지는 유력하지만, CUDA online generation/prefetch를 포함한 end-to-end smoke 전에는 정식 상한으로 확정하지 않는다.
- 실제 ICI context는 fold당 약 69명이므로 large-context-only 학습은 사용하지 않는다. 작은 context를 포함한 mixed 학습을 하고 표준 40/80 평가 성능이 개선될 때만 학습 효과로 인정한다.

**Batch 2 / context 300 추가 점검**

| Episode batch | 총 bags/episode | Peak allocated VRAM | Step time |
|---:|---:|---:|---:|
| 2 | 312 | 73,338.0 MiB | 0.196 s |
| 2 | 360 | 84,605.2 MiB | 0.223 s |

- `context 300 + query 최대 12 = 총 312 bags`는 B200에서 큰 메모리 여유로 통과했다. 이 벤치 역시 FP32이며 bags 절반을 query로 둔 보수적 model-step 조건이다.
- `episode_batch_size=2`로 내리면 `accumulate_grad_batches=4`로 올려 기존 effective batch 8과 epoch당 optimizer step 수를 유지한다. accumulation 2를 그대로 두면 effective batch가 4로 바뀌고 optimizer step이 2배가 되어 context 효과와 최적화 효과가 섞인다.
- 실제 ICI context가 약 69이므로 context 300 고정 학습은 피한다. 기존 38~95 구간을 충분히 포함하는 mixed/bucket sampling으로 최대 300까지 노출하고, 40/80 성능 개선을 1차 성공 기준으로 둔다.

**Mixed-context sampler 구현 (2026-07-31)**
- Config: `configs/train_v22_hard_context300.yaml`
- 각 training shape group에서 중심을 `[40, 80, 120, 160, 180, 240, 300]` 중 균등 선택하고 정수 jitter `[-5, +5]`를 적용한다.
- Dataset은 선택된 context에 query 5~12개를 더해 총 bag 수를 생성한다. `ModelInterface`는 실제 query 수를 필터링하여 query 제거 후 남는 context가 반드시 선택 가능한 중심의 `±5` 안에 들도록 보장한다.
- 최대 조합은 context 305 + query 12 = 총 317 bags이며, 이 범위를 train dataset에만 적용한다. Validation/test의 기존 50~100 bags 분포는 유지한다.
- Batch 2, accumulation 4로 effective 8 episodes/update와 epoch당 optimizer step 수를 기존 Hard 기준선과 동일하게 유지한다.
- 테스트: 관련 dataset/model 27개 통과, batched forward 포함 31개 통과(169.244초).
- 최초 training smoke에서 whole-batch CUDA prefetch가 다음 대형 episode 생성을 현재 forward와 겹치며 순간 allocator OOM 경고를 냈다. Prefetch를 끈 뒤에도 batch 내부 두 CUDA stream의 episode 생성이 겹칠 때 같은 경고가 재현됐다. 두 run은 중단했다.
- 최종 안전 설정은 `cuda_prefetch: false`, `parallel_cuda_generation: false`다. 두 episode tensor는 순차 생성한 뒤 stack하여 batch 2 forward/backward를 수행하므로 학습 통계와 effective batch는 바뀌지 않는다.
- Prefetch-off 최대 경계 smoke: `(batch=2, bags=317, cells=1500, dim=512)`, query 12, 실제 context 305. Online generation부터 forward/backward까지 peak 74,546.6 MiB로 통과했다.
- 외부 W&B 인증에 의존하지 않도록 파생 config는 local CSV logger를 사용한다.
- Fine-tuning: Hard best epoch 44 checkpoint에서 epoch 45~49를 이어 학습 완료.
- 로그: `logs/20260731_context300_ft/v22_hard_context300_ft_serial.out`
- Checkpoints: `checkpoints/20260731_context300_ft/v22_hard_context300_ft_serial/`

| Fine-tuning epoch | val CE | val AUROC (104 episodes, 참고용) |
|---:|---:|---:|
| 45 | 0.6867 | 0.5542 |
| 46 | 0.6889 | 0.5588 |
| **47** | **0.6853** | 0.5623 |
| 48 | 0.6864 | 0.5586 |
| 49 | 0.6890 | 0.5626 |

- 기존 Hard best CE `0.6839`를 넘지 못했으므로 CE 기준 개선 신호는 없다. 다만 104-episode AUROC는 불확실하므로 epoch 47 checkpoint를 동일 1,000-episode context curve로 최종 비교한다.
- Fine-tuned checkpoint 평가 완료: context 40/80/160/300, fixed query, pool 400 bags, 1,000 episodes.

| Context | AUROC (episode-cluster 95% CI) | Log loss |
|---:|---:|---:|
| 40 | 0.5331 [0.525, 0.542] | 0.6992 |
| 80 | 0.5553 [0.546, 0.565] | 0.6901 |
| 160 | 0.5842 [0.575, 0.593] | 0.6798 |
| 300 | 0.5977 [0.588, 0.607] | 0.6762 |

- 로그: `logs/20260731_context300_eval/context_curve.out`
- 산출물: `logs/v22_hard_context300_ft_curve_1000ep.csv`, `predictions/v22_hard_context300_ft_curve/context_{40,80,160,300}.pt`
- 동일 pool 400·동일 episode의 원본 Hard checkpoint 대조 평가 완료:

| Context | 원본 AUROC | Fine-tuned AUROC | Δ |
|---:|---:|---:|---:|
| 40 | 0.5284 | 0.5331 | +0.0047 |
| 80 | 0.5483 | 0.5553 | +0.0070 |
| 160 | 0.5734 | 0.5842 | +0.0108 |
| 300 | 0.5839 | 0.5977 | +0.0138 |

- Fine-tuning 이득은 context가 클수록 증가하지만 사전 구조 후보 기준 overall `+0.03`에는 전 구간 미달한다. 실제 ICI 범위 40/80의 이득도 `+0.005~0.007`로 작다.
- 대조 로그: `logs/20260731_context300_baseline_eval/context_curve.out`; summary: `logs/v22_hard_baseline_pool400_curve_1000ep.csv`

**Medium context-to-oracle 실험**
- 공식 checkpoint: `checkpoints/20260729_160643/v22_medium_fixed/epoch=013-val_ce_loss=0.5946.ckpt`; 기존 AUROC `0.7078 [0.696, 0.719]`.
- Config: `configs/train_v22_medium_context300.yaml`; Hard와 같은 context 중심 `[40,80,120,160,180,240,300]±5`, batch 2/accumulation 4, 순차 CUDA generation.
- Medium 원본 checkpoint pool-400 context curve 완료:

| Context | AUROC (episode-cluster 95% CI) | Log loss |
|---:|---:|---:|
| 40 | 0.6757 [0.665, 0.686] | 0.6456 |
| 80 | 0.7204 [0.709, 0.730] | 0.6100 |
| 160 | 0.7654 [0.755, 0.775] | 0.5817 |
| 300 | 0.7989 [0.790, 0.807] | 0.5639 |

- 재학습 전 모델도 context 40→300에서 `+0.1232` 상승해 context 활용 능력이 분명하다. 다만 300에서도 overall 0.80이며, state task의 비현실적 oracle-mask 0.9013과는 직접 같은 지표가 아니므로 혼동하지 않는다.
- Baseline 평가 로그: `logs/20260731_medium_context_baseline_eval/context_curve.out`; summary: `logs/v22_medium_baseline_pool400_curve_1000ep.csv`.
- Mixed-context fine-tuning 완료:

| Epoch | val CE | val AUROC (104 episodes, 참고용) |
|---:|---:|---:|
| **14** | **0.5953** | 0.7340 |
| 15 | 0.5970 | 0.7371 |
| 16 | 0.5961 | 0.7387 |
| 17 | 0.5955 | 0.7378 |
| 18 | 0.5978 | 0.7344 |
| 19 | 0.5953 | 0.7340 |

- Best epoch 14도 원본 best CE `0.5946`을 넘지 못했다. CE 기준 개선 신호는 없다.
- Fine-tuning 로그: `logs/20260731_medium_context300_ft/train.out`; checkpoints: `checkpoints/20260731_medium_context300_ft/v22_medium_context300_ft/`.
- 사용자 요청에 따라 20 epochs에서 결론 내리지 않고 `max_epochs=100`으로 확장했다. Epoch 19 `last.ckpt`의 model/optimizer/scheduler/global-step을 모두 복원해 epoch 20부터 이어간다.
- 중간 best epoch 14의 context curve(PID `2542931`)와 대기 중이던 state upper-bound(PID `2543210`)는 최종 100-epoch 후보로 대체되므로 중단했다.
- 100-epoch run PID: `2544289`; 로그: `logs/20260731_medium_context300_100e/train.out`.
- Checkpoints: `checkpoints/20260731_medium_context300_100e/v22_medium_context300_100e/`.
- 현재 속도 약 3.4분/epoch 기준 epoch 99까지 예상 약 4.5시간. 완료 후 best CE checkpoint를 동일 pool-400 curve 및 state observable/oracle 진단으로 평가한다.
- 진행 확인: epoch 31의 42%, 경과 40분 43초, GPU 약 102.7 GiB, OOM/Traceback 없음. Epoch 20~30 val CE는 `0.5955~0.5999`; mixed-context 전체 best는 여전히 epoch 14의 `0.5953`이며 공식 원본 best `0.5946`을 넘지 못했다.
- 사용자 판단에 따라 epoch 31 validation 완료 후 run을 중단했다. Epoch 31 CE `0.5947`, AUROC `0.7351`로 공식 CE `0.5946`과 사실상 동률이다. Train total/CE의 epoch 20~30 평균도 하락하지 않아 추가 epoch보다 architecture 개선으로 전환한다.
- 상세 architecture 근거와 v23 후보: [`architecture_v23_candidates.md`](architecture_v23_candidates.md).
- **다음 우선순위 T5-A**: 기존 40-token aggregator는 고정하고 token type/slot/tail identity를 부여한 뒤, bag 내부에서 structured embedding을 만들고 각 labelled bag을 fixed 8-token class memory 없이 direct ridge/cross-attention에 전달한다. 이 실험으로 episode-level context 압축과 bag-level 정보 손실을 먼저 분리한다.
- State oracle-mask `0.9013`/latent `1.0`은 정답 반응세포 또는 latent score를 사용하므로 달성 목표가 아니라 비현실적 참고 상한이다. 실제 성공 기준은 관측 입력만으로 context 40/80 및 overall AUROC가 개선되는지다.

**T4-0. 재학습 없는 접근성 감사 [먼저]**
- Hard best checkpoint로 state `model_input`/`observable`/`oracle` 상한 재측정
  - 완료: 현재 모델 0.5175 [0.496, 0.538], model-input 최대 0.5336 [0.510, 0.556](+0.016, 기준 +0.05 미달), oracle-mask 0.6953
  - 산출물: `logs/v22_hard_state_upper_bound_1000ep.csv`; 로그 `logs/20260731_hard_accessibility/state_upper_bound.out`
- Hard config에서 covariance all-cell/oracle-mask 상한 재측정
  - 완료: all-cell 최대 0.5044 [0.483, 0.525], 현재 모델 0.5103과 동률; oracle-mask covariance 최대 0.6074 [0.564, 0.652]
  - 산출물: `logs/v22_hard_covariance_upper_bound_1000ep.csv`; 로그 `logs/20260731_hard_accessibility/covariance_upper_bound.out`
- effect scale 0.4/0.7 matched 평가로 Hard task 상대 약점이 생성기 scale 아티팩트인지 확인
  - 완료: scale 0.4 overall 0.5086 [0.500, 0.518], scale 0.7 overall 0.5282 [0.519, 0.538]; 모든 단일 task 0.498~0.531, combined만 scale 0.7에서 0.5645
  - 산출물: `predictions/synthetic_v22_hard_scale{0.4,0.7}_1000ep.pt`; 로그 `logs/20260731_hard_accessibility/matched_effect.out`

**T4-1. 누적 bridge ablation [T4-0 후]**
1. **Signal scarcity**: class separation + rare fraction/probability
2. **Nuisance**: donor shift + component shift + observation noise
3. **Geometry/scale**: latent dim + cell/bag 범위
4. **Optimization/model setting**: batch/accumulation + covariance rank

Medium에서 시작해 위 그룹을 하나씩 Hard 값으로 바꾸고, 동일 optimizer-step 예산과 1,000-episode paired evaluation을 사용합니다. 처음 성능이 급락하는 그룹 안에서만 one-factor ablation을 합니다. 모든 조합을 탐색하지 않습니다.

**판정 기준**
- overall AUROC 차이 **+0.03 이상** 또는 target task **+0.05 이상**
- episode cluster CI + paired bootstrap 필수
- 관측 가능한 헤드룸이 없는 상태에서 새 architecture head 추가 금지
- ICI는 Medium+Hard 후보 확정 전까지 실행 금지

### ~~Tier 2 — state 분기~~ — 🛑 **종료 (2026-07-31)**

상세 결과는 §3의 T2-1/T2-2 표를 참고합니다. `model_input`, `observable`, `oracle_mask`, `oracle_latent`를 처음부터 분리했고, 현재 모델이 관측 가능 probe와 동률이라는 사전 종료 조건이 충족됐습니다. 따라서 T2-3 이득 곡선과 state 아키텍처 변경은 진행하지 않습니다.

### Tier 3 — 방법론 (성능이 아니라 판단 신뢰도)

**T3-1. ✅ 완료 (2026-07-30) — 진짜 약점은 `state`.**
effect scale을 통일해 비교하니 covariance는 composition과 동률이고 **state가 전 구간 최하위**. 기본 config의 covariance 최하위는 생성기 아티팩트였습니다. 상세는 §3.
재현: `python scripts/evaluate_synthetic.py --checkpoint <best>.ckpt --config configs/train_v22_medium.yaml --val-episodes 400 --effect-scale 0.7`

<details><summary>원래 T3-1 계획 (완료)</summary>
현재 task별 AUROC는 생성기 effect scale(composition 1.40 / state 0.72 / covariance 0.55)에 오염되어 **아키텍처의 상대적 강약을 직접 비교할 수 없습니다.** 모든 task의 effect scale을 동일하게 맞춘 진단용 데이터 config를 만들면 "어느 메커니즘에 실제로 약한가"를 처음으로 공정하게 볼 수 있습니다. 학습된 모델을 그 데이터로 평가만 하면 되므로 재학습 불필요.
</details>

**T3-2. ✅ 완료 (2026-07-30) — 권고: val episode 1,000개.**
CI 폭 104→0.074 / 400→0.035 / 1,000→0.021. task별은 1,000에서 0.045. **task별 +0.05 미만을 노리면 2,000개 이상 필요.** 상세는 §3. 이 과정에서 공식 기준선도 1,000 episode 기준으로 갱신했습니다(0.7078).

<details><summary>원래 T3-2 계획 (완료)</summary>
현재 104 episodes → CI 폭 0.060. Tier 1/2에서 기대하는 개선폭이 그보다 작다면 검출이 안 됩니다. `val_dataset_kwargs.episodes_per_epoch`를 늘려 CI를 좁히세요 (episode 수가 실질 표본 크기, §5).
</details>

**T3-3. ✅ 완료 (2026-07-31) — v22 Hard 기준선.**
Best `val_ce_loss 0.6839`; 1,000-episode AUROC `0.5483 [0.538, 0.558]`. state 0.5167 / covariance 0.5103으로 거의 무작위이며, v21 Phase 2 CE 0.6845를 재현했습니다. 상세는 위 T3-3 완료 표 참고.

### ~~Tier 1 — 반응세포 식별 (Sparse Evidence)~~ — 🛑 **종료 (2026-07-29)**

**결론: 진행하지 않습니다.** 사전 판정 기준에 따라 T1-C 2단계에서 종료했습니다 (bag 라벨 purity 0.128 ≤ 0.15). 전체 근거 사슬은 §3의 T1-0 → T1-A → T1-B → T1-C 참고.

| 단계 | 질문 | 결과 |
|---|---|---|
| T1-0 | covariance 상한이 실재하는가 | 오라클 세포 0.893 / **실제 관측 가능 0.570** — 모델 0.612는 이미 그 위 |
| T1-A | 세포 선택이 반응세포를 찾는가 | 기하 3종 + 학습 1종 **전부 AUROC ~0.50** |
| T1-B | 슬롯 단위로는 되는가 | fragmentation entropy 0.963 — 12슬롯에 흩어짐. 단 세포 라벨 held-out은 0.697 |
| T1-C 1 | 부분 개선도 값이 있는가 | **있음** — 곡선 선형, 순도 0.40에서 분기 +0.107 |
| T1-C 2 | bag 라벨만으로 되는가 | **안 됨** — purity 0.128 (무작위 0.110) → **종료** |

<details><summary>종료된 Tier 1 상세 계획</summary>


T1-0/T1-1 진단으로 원래 가설이 무너지고 병목이 바뀌었습니다 (§3 참고). 요약:
- 모델 0.6122는 **진짜 관측 가능한 상한 0.5704를 이미 초과**합니다.
- 0.8931은 **반응세포 오라클 마스크**를 받은 경우이며 모델은 그 마스크를 못 받습니다.
- 반응세포는 전체의 **11.7%**뿐. **그 세포들을 찾아내는 것이 유일한 실질 레버**입니다.

**T1-A. ✅ 완료 (2026-07-29) — 세포 선택이 무작위와 구분되지 않음.**
기하학적 기준 3종(`outlier_distance` 0.5091 / `studentized` 0.5098 / `novelty` 0.4984)과 **학습된 기준**(`class_memory` 0.4971) 전부 AUROC ~0.5. precision = base rate, recall = 유지 비율. 4개 task 모두 동일. 상세는 §3.
**결론: k값 튜닝은 무의미합니다.** 랭킹에 신호가 없습니다.
재현: `python scripts/diagnose_cell_selection.py --config configs/train_v22_medium.yaml --val-episodes 400 --checkpoint <best>.ckpt`

**T1-B. ✅ 완료 (2026-07-29) — 슬롯 정렬 실패, 그러나 특징에는 신호 있음.**
슬롯 capture 0.155(무작위 0.083)·fragmentation entropy 0.963 → 반응 component는 12개 슬롯에 흩어짐. 반면 held-out LDA는 0.6969 → **특징에는 신호가 있고 메커니즘이 없는 것**. 상세는 §3.
재현: `python scripts/diagnose_oracle_slot_alignment.py --config configs/train_v22_medium.yaml` / `python scripts/diagnose_cell_selection.py --probe --checkpoint <best>.ckpt`

<details><summary>원래 T1-B 계획 (완료)</summary>
T1-A가 실패 원인을 짚어줬습니다: 반응세포는 `effect_mask = (component_index == effect_component_index)`, 즉 **latent mixture component 하나**이고, covariance task에서는 **위치가 아니라 분산이 바뀝니다.** 중심에서 먼 세포를 찾는 현재 기준으로는 구조적으로 못 찾습니다.
→ 개별 세포가 아니라 **component 단위**로 접근해야 합니다. aggregator는 이미 12개 slot에 세포를 배정하므로:
1. **반응 component가 특정 슬롯과 정렬되는지 측정** — 기존 도구 `scripts/diagnose_oracle_slot_alignment.py`가 이 용도입니다.
2. 정렬된다면 → 슬롯별 공분산(`slot_covariance_sketch`, 이미 계산 중)으로 0.89에 얼마나 근접하는지 측정.
3. 정렬되지 않는다면 → 슬롯 배정 자체(anchor 선정, `assignment_temperature`)가 문제.
</details>

**T1-C 1단계. ✅ 완료 (2026-07-29) — 이득 곡선 선형, Tier 1 유지.**
순도 0.11→1.00에서 covariance AUROC 0.517→0.888로 **선형 증가, 문턱 없음**. 현실적(held-out LDA, 순도 0.40) 기대치는 분기 **0.557 → 0.664 (+0.107)**. 상세는 §3.
재현: `python scripts/diagnose_selection_gain_curve.py --config configs/train_v22_medium.yaml --val-episodes 400`

**T1-C 2단계. [다음] bag 라벨만으로 판별 방향을 학습할 수 있는지 검증.**
남은 핵심 미지수는 하나입니다: **세포 라벨 없이, bag 라벨(R/NR)만으로 순도 0.4 수준의 선택을 학습할 수 있는가?** 이것이 되면 +0.107이 현실이고, 안 되면 Tier 1은 여기서 끝입니다.
- 값싼 선행 검증: episode 여러 개에 걸쳐 **bag 라벨로 학습한 선형 판별 방향**(R bag 세포 vs NR bag 세포)이 반응세포를 얼마나 골라내는지 측정. 세포 라벨을 안 쓰므로 **실제 학습 가능한 상한**이 됩니다. `diagnose_cell_selection.py`에 score를 하나 추가하는 수준의 작업입니다.
- 이 값이 순도 0.3~0.4를 내면 → 구조 변경(attention 기반 세포 가중) 착수 근거 확보.
- 0.15 이하면 → **Tier 1 종료**, Tier 2/3으로 이동.

> [!WARNING]
> **T3-2(val episode 증량)가 Tier 1 검증의 선행 조건입니다.** covariance는 전체의 20%라 성공해도 전체 AUROC로는 +0.014 수준이고 현재 CI 폭(0.060)에 묻힙니다. §3의 검출 가능성 경고 참고.

</details>

### 하지 말 것

- **ICI 실행 금지** — 합성에서 후보가 확정되기 전까지. §5 참고, 지금 돌리면 테스트 세트를 조기 소진합니다.
- **오라클 기반 상한을 목표치로 삼기 금지** — descriptor가 `responsive_instance_mask`나 latent 파라미터를 쓰는지 항상 먼저 확인하세요. 모델이 못 받는 정보로 만든 상한은 목표가 아닙니다 (§3 정정 사례).
- **effect scale 정규화 없이 task별 AUROC로 우열 판단 금지** (§3 T3-1). 기본 config의 task별 표는 생성기 난이도에 오염되어 있습니다.
- **상한 측정 없이 아키텍처부터 뜯기 금지** — Tier 1 전체가 학습 한 번 없이 진단만으로 닫혔습니다. 관측 가능한 상한 → 모델 위치 → 이득 곡선 순서를 지키세요.
- **104 episode 결과로 판정 금지** — `--val-episodes 1000` 사용 (§3 T3-2).

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

### 이번 세션에서 확정/진행한 것

- v23-A0 (exact mean) 50-epoch **완료**: best epoch 43 `val_ce_loss 0.5912154`
  (v22 0.5946 대비 -0.0034). §3 참고.
- v24-A0 (learned projection, slot 1) 50-epoch **완료**: best epoch 45
  `val_ce_loss 0.5976237`. slot 1개 정보 손실로 훈련 val에서 v22/v23보다 높음. §3 참고.
- v24-B0 (per-token bottleneck projection, slot 12 유지) 50-epoch **완료**:
  best epoch 46 `val_ce_loss 0.5923204`. v24-A0 대비 -0.0053 개선, v22 대비 -0.0023 개선. §3 참고.
- v24-B1 (residual mean + bottleneck projection, slot 12 유지) 50-epoch **완료**:
  best epoch 41 **`val_ce_loss 0.5903045` (전체 Bag-Collapse 모델 중 최저 기록 달성)**. §3 참고.
- 구현: v24 learned projection (`26b2b27`), v24-B0 병목 (`b2fb9d0`), v24-B1 residual 병목 (`4f984ca`).
  architecture_version 23/24 분리, 모든 단위 테스트 통과.

### 다음 Action (이 세션 당시 계획, 아래 §10에서 폐기됨)

1. ~~v23-A0 (epoch 43), v24-A0 (epoch 45), v24-B0 (epoch 46), v24-B1 (epoch 41) 4개 후보 체크포인트를
   동일 pool-400, 1,000 episodes, context `40/80/160/300`에서 평가.~~
2. ~~`scripts/compare_predictions.py`로 v22(`predictions/v22_medium_baseline_pool400_curve/`)와
   episode-cluster paired delta + CI 계산.~~
3. ~~판정: overall `+0.03` 또는 target task `+0.05`가 없으면 해당 후보 폐기.~~
4. 합성 Medium+Hard 후보가 확정되기 전까지 ICI는 실행하지 않습니다. **(이 항목만 유지됨 — ICI는 계속 잠금)**

---

## 10. 2026-08-01 세션 핸드오프 — v24 확정, 평가 계획 폐기

### 이번 세션에서 확정/진행한 것

- **사용자 결정**: §9의 4종 paired 비교 계획을 실행하지 않고 폐기. "단순히 label로 bag을 나누는 구조(class-memory 압축)가 별로"라는 문제 제기에서 시작해, v24-B1(residual + bottleneck bag projection)을 train `val_ce_loss` 순위만으로 최종 v24 아키텍처로 확정.
- v22(구 기준선), v23-A0, v24-A0, v24-B0 폐기. 상세는 §3 "최종 결정 (2026-08-01)".
- `configs/train_v23_medium_bag_mean.yaml`, `train_v24_medium_bag_proj.yaml`, `train_v24_medium_bag_proj_bottleneck.yaml`을 `configs/archive/v23_v24_candidates/`로 이관. `train_v24_medium_bag_proj_residual.yaml`이 v24 production entry point.
- `docs/architecture_v23_candidates.md`에 결정 기록을 남기고 `docs/history/`로 이관 (T5-A/B/C는 필요 시 재검토할 미실행 계획으로 보존).
- Git: `main`/`v24` 브랜치를 `codex/v23-bag-mean` 최종 커밋으로 fast-forward. (커밋 해시는 이 문서를 갱신하는 커밋에서 확정 — 커밋 후 아래에 기록.)

### 남겨진 것 (폐기되지 않음)

- T4 Medium→Hard bridge attribution (§6) — 이번 결정과 무관하게 계속 열려 있는 질문.
- v24용 ICI config 부재 — ICI를 돌리려면 `train_v22_ici_finetune.yaml`/`train_v22_ici_scratch.yaml`에 상응하는 v24 버전이 필요.
- v22 top-level config 6개는 ICI 파이프라인이 아직 참조하므로 archive로 옮기지 않음 — v24 ICI config가 만들어지면 재검토.

### 다음 Action

1. (선택) v24용 ICI finetune/scratch config를 만들고 ICI 실행 여부를 사용자에게 다시 확인.
2. (선택) T4 Medium→Hard attribution을 v24 위에서 이어갈지 결정.
3. 새 구조 변경이 필요해지면 이번처럼 train CE만으로 확정하지 말고, 최소한 §7 평가 프로토콜의 1,000-episode paired 비교를 다시 켤지 사용자와 사전에 정할 것.
