# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-02 12:00:00 KST`
**Status**: v24 확정 유지. **v25(T5-A) Medium paired 평가 — 맥락 의존적 trade-off** (v25 우세 @context40, v24-B1 압도 @context300, 승격 기준 미달). **Easy tier 완료 — v24-easy(0.9073) ≈ v25-easy(0.9106), delta +0.0033 승격 기준 미달** → "아키텍처 계열 전체 한계" 가설 강화, **v25 최종 폐기 권고 (사용자 판단 대기)**. Easy 정식 paired 비교 실행 중(PID 277412). 상세는 §11.
**Read first if you are picking this up**: §11 (신규, 가장 중요), §3 "🧪 v25 (T5-A) 진행 중", §3 "최종 결정 (2026-08-01)" (v24 확정 배경), §6 Action Plan.
**Branches**: `codex/v25-typed-bag` = v25 작업 중 (base: `v24`) / `main` = `v24` = `codex/v23-bag-mean` 최종 커밋 / `v22`(구 기준선, 참조용 보존) / `v19` / `v18`(다른 서버) — 구조: [`history/branch_structure.md`](history/branch_structure.md)
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)
**Architecture Version**: `24` 확정 유지. **`25`는 아직 미확정 — v25 학습/평가 완료 후 판정.** (`project_structured_tokens: true, projection_bottleneck_dim: 64, projection_residual_mean: true` + `typed_bag_preserving_branch: true, typed_bag_bottleneck_dim: 64`). `22`/`23`은 폐기된 구버전.
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

### 🧪 v25 (T5-A) 진행 중 (2026-08-01) — label 기반 class-memory 문제를 실제로 다루는 시도

위 경고에서 지적한 문제("label로 bag을 나누는" class-memory 압축)를 실제로 건드리기 위해, 폐기했던 T5-A(typed, bag-preserving structured context branch)를 v25로 되살렸습니다. v24-B1 확정 이후 새 브랜치 `codex/v25-typed-bag`에서 작업 중이며, `main`/`v24`는 손대지 않았습니다.

- **구조**: 기존 40개 structured token에 학습된 token-type(global/center/spread/rare/tail) + tail-fraction identity embedding을 더하고, v24-B1과 동일한 구조(전용 bottleneck `Linear(512→64)` × 40 + exact mean residual → `Linear(3072→512)`, 단 완전히 별도 가중치)로 bag당 1개 embedding으로 압축합니다. 이 embedding을 **두 번째** `RidgeResidualMetaClassifier`(`typed_bag_classifier`)에 label과 1:1로 정렬된 채로 그대로 통과시켜, `_class_memories`처럼 label별로 뭉쳐서 압축하지 않고 context bag 정체성을 끝까지 보존합니다. 결과 logit은 기존 최종 fusion 뒤에 `covariance_residual_scale`과 동일한 sigmoid-gated, 작게 초기화된 residual(`typed_bag_residual_scale`, 초기값 0.02)로 더합니다. **기존 class-memory 경로(`_class_memories`/`_population_tokens`)는 완전히 그대로 유지** — 두 경로를 나란히 두고 비교하기 위함입니다.
- **원안(T5-A 문서)과의 의도적 차이**: slot-index embedding은 추가하지 않았습니다. slot은 episode마다 새로 도는 spherical k-means cluster id(`_context_spherical_kmeans_anchors`)라서 bag/episode 간 안정적인 의미가 없고, token-type/tail-fraction처럼 "모든 bag이 공유하는 고정된 역할/기준값"이 아니기 때문입니다. 상세 근거는 `src/models/baseline.py`의 `StructuredPopulationMetaClassifier.__init__` 주석 참고.
- **Config**: `configs/train_v25_medium_typed_bag.yaml` (`base_config: train_v24_medium_bag_proj_residual.yaml` + `typed_bag_preserving_branch: true`, `typed_bag_bottleneck_dim: 64`).
- **architecture_version**: `25` (`typed_bag_preserving_branch=true`일 때 24/23/22보다 우선). v24 checkpoint는 거부됩니다 (신규 가중치 없음).
- **구현 검증**: 전체 unittest **141개 통과** (`786.814s`). 신규 테스트: token-identity layout, typed pooling shape/값(bottleneck 유무), class-memory 경로 불변 확인, v25 architecture_version + checkpoint gating(v25가 v24 거부), BaseModel end-to-end forward+backward finite gradient.
- **학습 실행 중**: Run `20260801_020144`, scratch Medium 50 epoch (v22~v24-B1과 동일 방식 — 사용자 결정으로 warm-start 아님). PID `3612802`(bash 래퍼) / torchrun worker `3612807`. 모델 13.9M trainable params (v22 6.57M + v24 residual bottleneck ~2.88M + 신규 typed-bag 분기 ~4.4M).
  - 학습 로그: `logs/20260801_020144/v25_medium_typed_bag.out`
  - Launcher 로그: `logs/20260801_020144/v25_medium_typed_bag_launcher.out`
  - 체크포인트: `checkpoints/20260801_020144/v25_medium_typed_bag/`
  - Sanity check 통과, epoch 0 정상 진행 확인 (2026-08-01 02:02 KST). 완료까지 약 2~2.5시간 예상 (512 steps/epoch × 50 epoch, ~3 it/s).
- **판정 계획**: 사용자 결정으로 **이번엔 §6/§7 원래 프로토콜대로 1,000-episode paired 합성 평가까지 수행**합니다 (v24 확정 때와 달리 train CE만으로 판단하지 않음). 완료 후 `scripts/evaluate_synthetic.py` → `scripts/compare_predictions.py`로 v24-B1과 paired 비교, overall `+0.03` 또는 target task `+0.05` 기준 적용.
- **다음 Action**: 학습 완료 대기 → best checkpoint로 1,000 pool-400 episode, context 40/80/160/300 평가 → v24-B1과 paired 비교 → 승격/폐기 판정.

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

사전에 계획한 Tier 1~3 합성 작업이 모두 끝났고, 이어서 진행한 v23/v24 bag-collapse family도 4종 모두 학습을 완주했습니다. 원래 계획은 이 4종을 1,000-episode paired 비교로 검증한 뒤 동결하는 것이었으나, **사용자가 이 비교를 건너뛰고 v24-B1(residual + bottleneck)을 직접 v24로 확정했습니다** (§3 "최종 결정" 참고). v22 Medium baseline은 폐기되었고 최종 후보는 v24입니다. **ICI는 여전히 잠금 상태** — v24용 ICI config가 없고, 이번 결정이 §7 평가 프로토콜을 통과한 것이 아니므로 ICI 해제는 별도로 다시 확인이 필요합니다.

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
- 실제 ICI context는 fold당 약 69명이므로 large-context-only 학습은 사용하지 않습니다. 작은 context를 포함한 mixed 학습을 하고 표준 40/80 평가 성능이 개선될 때만 학습 효과로 인정합니다.

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

- Fine-tuning 이득은 context가 클수록 증가하지만 사전 구조 후보 기준 overall `+0.03`에는 전 구간 미달합니다. 실제 ICI 범위 40/80의 이득도 `+0.005~0.007`로 작다.
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

- 재학습 전 모델도 context 40→300에서 `+0.1232` 상승해 context 활용 능력이 분명하다. 다만 300에서도 overall 0.80이며, state task의 비현실적 oracle-mask 0.9013과는 직접 같은 지표가 아니므로 혼동하지 않습니다.
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
- **다음 우선순위 T5-A**: 기존 40-token aggregator는 고정하고 token type/slot/tail identity를 부여한 뒤, bag 내부에서 structured embedding을 만들고 각 labelled bag을 fixed 8-token class memory 없이 direct ridge/cross-attention에 전달합니다. 이 실험으로 episode-level context 압축과 bag-level 정보 손실을 먼저 분리합니다.
- State oracle-mask `0.9013`/latent `1.0`은 정답 반응세포 또는 latent score를 사용하므로 달성 목표가 아니라 비현실적 참고 상한입니다. 실제 성공 기준은 관측 입력만으로 context 40/80 및 overall AUROC가 개선되는지다.

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
CI 폭 104→0.074 / 400→0.035 / 1,000→0.021. task별은 1,000에서 0.045. **task별 +0.05 미만을 노리는 실험은 1,000개로도 부족하므로 2,000개 이상 필요**합니다. 상세는 §3. 이 과정에서 공식 기준선도 1,000 episode 기준으로 갱신했습니다(0.7078).

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
- Git: 이 문서 갱신 커밋은 `d668d33`. `main`/`v24` 브랜치를 이 커밋(즉 `codex/v23-bag-mean` 최종 커밋)으로 fast-forward.

### 남겨진 것 (폐기되지 않음)

- T4 Medium→Hard bridge attribution (§6) — 이번 결정과 무관하게 계속 열려 있는 질문.
- v24용 ICI config 부재 — ICI를 돌리려면 `train_v22_ici_finetune.yaml`/`train_v22_ici_scratch.yaml`에 상응하는 v24 버전이 필요.
- v22 top-level config 6개는 ICI 파이프라인이 아직 참조하므로 archive로 옮기지 않음 — v24 ICI config가 만들어지면 재검토.

### 다음 Action

1. (선택) v24용 ICI finetune/scratch config를 만들고 ICI 실행 여부를 사용자에게 다시 확인.
2. (선택) T4 Medium→Hard attribution을 v24 위에서 이어갈지 결정.
3. 새 구조 변경이 필요해지면 이번처럼 train CE만으로 확정하지 말고, 최소한 §7 평가 프로토콜의 1,000-episode paired 비교를 다시 켤지 사용자와 사전에 정할 것.

---

## 11. 2026-08-02 세션 핸드오프 — v25 Medium 평가 완료(사실상 동률), Easy tier 실험 진행 중

### v25(T5-A) Medium 1,000-episode pool-400 평가 — v24-B1과 사실상 동률, 승격 기준 미달

§3 "v25 (T5-A) 진행 중"에서 이어감. scratch 50-epoch 학습 완료: best `val_ce_loss 0.5939` @ epoch 24
(체크포인트 `checkpoints/20260801_020144/v25_medium_typed_bag/epoch=024-val_ce_loss=0.5939.ckpt`).
이번 세션에서 v24-B1과 함께 처음으로 1,000-episode pool-400 context curve를 평가했습니다
(v24-B1도 이전에는 이 평가를 받은 적이 없었음 — §3 "최종 결정"에서 스킵됐던 부분).

| context | v24-B1 AUROC [CI] | v25 AUROC [CI] |
|---:|---|---|
| 40  | 0.6774 [0.666, 0.688] | 0.6792 [0.668, 0.690] |
| 80  | 0.7223 [0.711, 0.732] | 0.7217 [0.710, 0.732] |
| 160 | 0.7688 [0.758, 0.778] | 0.7685 [0.759, 0.778] |
| 300 | 0.8036 [0.794, 0.812] | 0.8017 [0.792, 0.811] |

예측 파일: `predictions/v24_medium_bag_proj_residual_pool400_curve/`, `predictions/v25_medium_typed_bag_pool400_curve/`.
차이가 전부 CI 폭(±0.005~0.011)보다 훨씬 작아 **사실상 동률**입니다 — §6/§7 판정 기준(overall
+0.03 또는 target task +0.05)에 한참 못 미칩니다. 즉 **label 기반 class-memory를 우회하는
bag-preserving 분기를 추가해도 Medium에서는 아무 이득이 없었습니다.**

`scripts/compare_predictions.py`로 정식 paired win-rate 계산을 시도했으나 **주의**:
1. 첫 시도에서 두 예측 파일이 동일 파일명(`context_40.pt` 등)이라 스크립트 내부 dict가
   파일명(stem)을 key로 써서 하나가 덮어써지는 버그를 만났습니다 — 반드시 서로 다른 이름으로
   복사한 뒤 비교할 것 (`/tmp/.../scratchpad/compare_tmp/{v24b1,v25}_context_{size}.pt`로 해결).
2. 이 스크립트의 cluster bootstrap(5000 샘플, 순수 Python 루프)이 **매우 느립니다** —
   context 40 하나에 CPU 700%대로 수십 분이 걸렸습니다. 세션 재시작으로 한 번 유실되어
   `nohup ... & disown`으로 완전히 분리해 재실행했습니다 (PID `4153394`/`4153395`).
   **이 정식 paired win-rate 결과는 아직 안 나왔을 수 있습니다** — 다음 세션은
   `/tmp/claude-3564/-NHNHOME-kimds/f036ed83-abdf-4ebb-a5e9-d834e04e4f52/scratchpad/v25_vs_v24b1_compare_fixed2.log`를
   먼저 확인할 것. 위 표의 개별 AUROC+CI만으로도 결론(동률)은 이미 명확합니다.

### 🧪 Easy tier 실험 — "0.59가 이 아키텍처 계열의 물리적 한계 아니냐"는 질문 검증 중

사용자 관찰: v22~v25(mean/projection/bottleneck/residual/typed-bag) 전부 val CE
**0.5903~0.5976** 범위(폭 0.0073)에 몰려 있음. 과거 Tier 2 진단(§3 T2-1/T2-2: state AUROC가
관측 가능한 최선의 probe와 이미 동률, oracle은 0.88~0.90)과 맞물려 "bag summary 압축 방식이
아니라 이 아키텍처 계열 전체의 한계"라는 가설이 유력함. 이를 검증하기 위해 Hard tier가
건드린 축을 정확히 반대 방향으로 돌린 **Easy tier**를 만들어 v24/v25 두 아키텍처만 재학습·비교합니다
(6개 전부가 아니라 사용자 지시로 v24/v25만).

**Easy tier 정의** (`configs/train_v24_easy.yaml` 헤더 주석에 Medium/Hard/Easy 전체 표 있음):

| 축 | Medium | Hard | Easy |
|---|---|---|---|
| class_separation | [0.5, 1.4] | [0.2, 0.8] | **[1.2, 2.2]** |
| donor_shift_scale | 0.35 | 0.70 | **0.15** |
| donor_component_shift_scale | 0.12 | 0.25 | **0.06** |
| observation_noise | 0.01 | 0.05 | **0.002** |
| rare_response_probability | 0.15 | 0.25 | **0.05** |
| rare_response_fraction | [0.02,0.08] | [0.005,0.03] | **[0.05,0.15]** |
| response_mixture_effect_scale | 1.40 | (동일) | **1.80** |
| response_state_effect_scale | [0.45,1.00] | (동일) | **[0.90,1.60]** |
| response_covariance_effect_scale | [0.30,0.80] | (동일) | **[0.70,1.30]** |

`num_bags`/`num_cells`/`latent_dim`은 안 건드림 — 분리 가능성/노이즈 축만 바꿔 "신호가
많아지면 아키텍처가 갈리는가"만 순수하게 봅니다. Config: `configs/train_v24_easy.yaml`
(base: v24-B1 confirmed config), `configs/train_v25_easy.yaml` (base: v25 config). **아직 git에
커밋 안 됨(untracked)** — 다음 커밋에 포함할 것.

**진행 상황**:
- ✅ v24-easy: scratch 50-epoch 완료. Best `val_ce_loss` **0.3499** @ epoch 49
  (Medium 0.59 대비 압도적으로 낮음 — Easy tier가 실제로 훨씬 쉬움을 확인).
  Run `20260801_075144`, checkpoint `checkpoints/20260801_075144/v24_easy/epoch=049-val_ce_loss=0.3499.ckpt`,
  로그 `logs/20260801_075144/v24_easy.out`.
- ✅ **v25-easy: scratch 50-epoch 완료 (2026-08-02 03:49)**. Best `val_ce_loss`
  **0.3473** @ epoch 45 (v24-easy 0.3499 대비 $-0.0026$ 근소 우위).
  Run `20260801_235601`, checkpoint `checkpoints/20260801_235601/v25_easy/epoch=045-val_ce_loss=0.3473.ckpt`
  (top-3: e45 0.3473 / e47 0.3495 / e49 0.3494), 로그 `logs/20260801_235601/v25_easy.out`.
  > [!NOTE]
  > **속도 저하 10x 경고는 해소됨 (transient)**: 초기 epoch만 0.45~0.5 it/s로 느렸고
  > (epoch 0~1 기준), 후반 epoch는 4.3 it/s까지 회복. 총 50 epoch을 **약 3시간 50분**에
  > 완료 (16시간 예상보다 훨씬 빠름). 원인은 데이터 생성 warm-up/캐싱으로 추정되며
  > 지속적인 병목은 아님.

- ✅ **v24-easy vs v25-easy 1,000-episode 평가 완료 (2026-08-02 11:40)**:

  | 모델 | AUROC [CI] | Log loss |
  |---|---|---|
  | v24-easy | **0.9073** [0.901, 0.913] | 0.3828 |
  | v25-easy | **0.9106** [0.904, 0.917] | 0.3761 |

  - overall delta **+0.0033** — 승격 기준(+0.03)에 한참 미달. CI도 겹침.
  - Per-task AUROC (모두 v25 근소 우위이나 threshold +0.05 미달):
    composition 0.9439→0.9456 / state 0.8938→0.9010 / covariance 0.8186→0.8251 /
    interaction 0.8914→0.8954 / combined 0.9532→0.9546.
  - 예측: `predictions/v24_easy_1000ep.pt`, `predictions/v25_easy_1000ep.pt`.
    로그: `logs/20260802_easy_eval/easy_1000ep.out`.

- 🔄 **Easy tier 정식 paired 비교 실행 중** (2026-08-02, PID `277412`):
  `compare_predictions.py predictions/v24_easy_1000ep.pt predictions/v25_easy_1000ep.pt`,
  로그 `logs/20260802_easy_eval/easy_paired_compare.out`.

### 🏁 v25(T5-A) 판정 근거 종합 (2026-08-02)

**Medium (1,000-episode pool-400, paired 5000 bootstrap — 이전 세션에서 완료된 로그를 기록)**:

| context | v24-B1 AUROC | v25 AUROC | P(v24-B1 beats v25) |
|---|---:|---:|---:|
| 40 | 0.6774 | 0.6792 | **0.04** (v25 우세) |
| 80 | 0.7223 | 0.7217 | 0.77 (구분 불가) |
| 160 | 0.7688 | 0.7685 | 0.65 (구분 불가) |
| 300 | 0.8036 | 0.8017 | **1.00** (v24-B1 압도) |

→ "사실상 동률"은 marginal CI 기준이며, paired bootstrap으로 보면 **맥락 의존적 trade-off**:
v25(typed bag-preserving)는 작은 context(40)에서 유의하게 우세, 큰 context(300)에서
v24-B1이 압도. 80/160은 구분 불가. 승격 기준(+0.03/+0.05)은 전 구간 미달.
(paired 로그: `/tmp/claude-3564/.../v25_vs_v24b1_compare_fixed2.log` — scratchpad,
repo 밖이므로 위 표로 기록 유지.)

**Easy**: v24-easy ≈ v25-easy (+0.0033, 승격 기준 미달).

**결론 (판정 기준 §11 item 4 적용)**: Easy tier에서도 두 아키텍처가 갈리지 않으므로
**"이 아키텍처 계열 전체의 한계" 가설이 강화됨.** v25(T5-A, typed bag-preserving)는
Medium/Easy 양쪽에서 승격 기준 미달 → **v25 최종 폐기 권고**, T5-B/T5-C 또는 완전히 다른
접근으로 이동 검토. 단, v25가 작은 context(40)에서 유의하게 우세했던 점은
ICI(~69 fold context)와 관련해 추가 검토 가치가 있음 — 사용자 판단 필요.

### 남겨진 것 / 다음 Action

1. ~~`configs/train_v24_easy.yaml`/`train_v25_easy.yaml` 커밋~~ → **완료** (`b6aacf7`에 이미 포함).
2. ~~v25 vs v24-B1 정식 paired win-rate 로그 확인, 문서에 최종 수치 기록~~ → **완료**
   (위 "판정 근거 종합" 표에 기록: v25 @40 우세, @300 열세, 80/160 구분 불가).
3. ~~v25-easy 학습 완료 대기 → v24-easy와 1,000-episode 평가로 비교~~ → **완료**
   (v24-easy 0.9073 vs v25-easy 0.9106, delta +0.0033).
4. Easy tier에서도 갈리지 않으므로 **"아키텍처 계열 전체의 한계" 가설 강화 → v25(T5-A)
   최종 폐기 권고, T5-B/T5-C 또는 완전히 다른 접근으로 이동 검토** — **사용자 판단 필요**.
   진행 중: Easy 정식 paired 비교(PID 277412) 완료 후 최종 수치 기록.
   Easy tier에서 유의미하게 갈리면 Medium이 ceiling/floor effect로 아키텍처 차이를
   가려온 것 → v25를 Medium에서 더 오래/다르게 학습해볼 근거가 생김.
5. ICI는 계속 잠금 (변경 없음).

---

## 12. 2026-08-02 세션 — 프로젝트 폴더 정리 (checkpoint/log/prediction purge)

사용자 지시로 폐기된 구버전 산출물을 정리했습니다. **코드/설정/문서는 불변**, 활성 run은
그대로 유지됐습니다.

### 삭제 내역 (모두 gitignore 대상 또는 git rm)
- **`checkpoints/` 53GB → 3.3GB**: v19/v20/v21/v22-era run 디렉터리 128개(20260722~20260729) +
  루트 느슨한 `.ckpt` 11개 + 폐기 확정 v23-A0/v24-A0/v24-B0 checkpoint
  (`20260731_v23_bag_mean_50e_resume`, `20260731_155635`, `20260731_182755`, `20260731_201252`).
  폐기된 candidate들의 수치는 `docs/history/v23_v24_bag_collapse_candidates.md`에 그대로 보존.
- **`logs/` 819MB → 532MB**: 20260722~20260729 dated 로그 + v19/v20-era named 로그
  (`tiranos/`, `pipeline_*`, `v19_covariance_candidates_*`, `2026072*_v19_*`) 삭제.
- **`predictions/`**: v19/v20/v21 ICI 예측 15개 삭제 (`ici_*.pt`, `ici_predictions_*.pt`,
  `v21_*_5fold.pt` 등). v22+ 합성 예측 및 curve는 유지.
- **v18**: `experiments/v18_learnability_c4_d_d0_d4.yaml`, `results/v18/` 삭제 (git rm).

### 유지 (활성/참조)
- checkpoints: `20260729_160643`(v22 baseline 참고), `20260731_035538`(v22 Hard),
  `20260731_220100`(v24 확정), `20260731_context300_ft`/`20260731_medium_context300_*`(T4),
  `20260801_020144`(v25), `20260801_075144`(v24-easy), `20260801_235601`(v25-easy, 진행 중).
- `data/`(14GB ICI 실데이터) 불변.

> [!NOTE]
> 향후 세션에서 이전에 참조되던 폐기 run checkpoint/log 경로를 찾을 경우, 해당 파일은
> 위 정리로 삭제됐고 수치만 문서에 남아 있습니다.

### Git 상태
- `git rm` 반영: `experiments/v18_*`, `results/v18/*`. 그 외 삭제는 전부 gitignore 대상이라
  git에 영향 없음. §11 "남겨진 것" item 1의 easy config 커밋은 이미 `b6aacf7`에서 완료됨
  (문서 스테일 — 여기서 해소).

---

## 13. 2026-08-02 세션 — 문서 및 config 정리 (docs/config/scripts purge)

사용자 지시로 v18/v19/v20-era 스테일 파일과 config를 정리했습니다. **코드(`src/`)와
활성 config·스크립트·living docs는 불변**, README/핸드오프 문서는 현재 상태로 갱신.

### 삭제 (git rm, 전부 구식/미참조)
- **루트 스테일**: `MODEL_ARCHITECTURE_KO.md`(v18 문서), `main.sh`/`main_medium.sh`/
  `main_minimum.sh`/`main_slurm.sh`/`test.sh`/`test_slurm.sh`(archive된 config·v16 checkpoint 참조).
- **구버전 스크립트** (`scripts/` 17개): `run_learnability_ladder.sh`,
  `run_sequential_pipeline.sh`, `run_v19_covariance_candidates.sh`, `sweep_csp_residual.py`,
  `benchmark_scalability.py`, `check_population_oracle.py`, `check_training_budget.py`,
  미참조 진단 `diagnose_{covariance_relations,covariance_subspace,local_geometry,
  anchor_candidates,context_size,tail_covariance,v19_branches,bag_label_selection,
  covariance_utilisation,oracle_covariance_upper_bound}.py`.
- **learnability-era 모듈 config** (`configs/` 14개): `callbacks/learnability.yaml`,
  `data/learnability_{a,b,manifold}.yaml`, `data/{synthetic,minimum}.yaml`,
  `model/covariance32.yaml`, `optimizer/adamw_learnability_5e4.yaml`,
  `scheduler/learnability.yaml`, `trainer/learnability_{a,b,d20}.yaml`,
  `trainer/{csp_short8,minimum_ddp8}.yaml`.

### 유지
- `scripts/`: `train.py`, `evaluate_synthetic.py`, `compare_predictions.py`,
  `evaluate_protocol.py`, `power_analysis.py`, `launch_interactive_training.sh`,
  `launch_ici_protocol.sh`, living docs 참조 진단 4종(`diagnose_{cell_selection,
  state_upper_bound,selection_gain_curve,oracle_slot_alignment}.py`), smoke 유틸.
- `configs/` 최상위 10개 전부 (v24/v25/easy 활성 + v22 ICI/T4용) — 변경 없음.
- `configs/` 모듈 서브폴더의 generic 조각(default/medium/logger/adamw 등) — 유지.

### 문서 갱신
- `README.md` 전면 갱신: v18 설명·`main*.sh` 사용법 제거 → v24 아키텍처 + `scripts/
  launch_interactive_training.sh` 표준 런처 + 문서 맵으로 교체.
- `docs/agent_handoff.md` §7-1: config 최상위 유지 목록을 현재 상태로 갱신 (v23/v24-A0/B0
  config는 `configs/archive/v23_v24_candidates/` 이관 반영).
- `docs/current_status.md` §12는 이전 checkpoint/log purge, §13은 본 문서/config purge 기록.

### Git 상태
- 커밋 예정. 삭제 스크립트의 history 문서 내 참조(`learnability_ladder.md`,
  `v20_scalability_plan.md`)는 아카이브 문서라 그대로 둠.

---

## 14. 2026-08-02 세션 — src/scripts/tests 점검 및 v21 아카이브 정리

사용자 지시로 `src`/`scripts`/`tests`를 점검했습니다.

### `src/` — 변경 없음 (전부 사용 중)
동적 import 경로(`dataset_src`/`optimizer_src`/`scheduler_src` via
`src/modules/{data,model}_interface.py`의 `import_module`)로 8개 모듈 전부 런타임
참조를 확인 — 삭제 대상 없음.

### `scripts/` — v21 폐기 코드 제거 + 1건 복원
- **삭제**: `scripts/archive/v21_retrieval/` (`benchmark_vram.py`, `run_vram_quick.py`,
  `launch_phase{4,6,6b,6c}_5fold.sh`) — v21 완전 폐기. `scripts/archive/` 디렉터리 자체가
  비워져 제거됨. living docs 참조 없음 (history 문서만 참조).
- **⚠️ 복원**: `scripts/diagnose_context_size.py` — §13 정리에서 삭제했으나
  `tests/test_context_size_diagnostic.py`가 `parse_sizes`/`split_indices`를 import하므로
  **테스트 스위트가 깨질 수 있었음**. `a5dfcf8^`에서 복원. 교훈: 스크립트 삭제 전 tests/
  의 `from scripts.* import` 의존성을 먼저 확인할 것.
- **유지 (사용자 결정)**: smoke 3종(`ddp_smoke.py` 8-GPU, `gpu_train_smoke.py` FP16,
  `medium_bf16_smoke.py` bf16) — 구식 정밀도/스케일이지만 보존.
- 참조 무결성 검증 완료: 남은 `src`/`scripts`/`tests`의 `from scripts.* import`가
  전부 실존 파일을 가리킴.

### `tests/` — 변경 없음 (10개 전부 유효)
- `test_learnability_ladder.py`는 실제로 fixed-episode-bank(`fixed_episode_count`) 기능
  테스트 (파일명 오해 소지, 기능 유효).
- 전체 unittest 실행은 이번엔 생략 (사용자 결정). 다음 코드 변경 시
  `timeout 1500s ... -m unittest discover -s tests -p "test_*.py"` 필수.
