# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-04` (**§28 S2(B2+B1) 게이트 전 항목 통과 — Musk 0.854, v30 승격 후보 — 최신**)
**Status**: **v24가 확정 baseline (변경 없음)** — v30 B1 플래그는 구현됐으나 **기본 OFF**이며 승격 전입니다. **§28 S2 = B2(cardinality-faithful log-uniform 샘플링) + B1(`poolz_l2`)이 사전 등록 게이트 6항목을 전부 통과**: Musk **0.8539**(종전 최고 0.822 경신), n≤4 **0.475→0.800**, n>34 0.667→0.698, 합성 무회귀 0.9483(v24 0.9510). paired bootstrap에서 **소형 구간 Δ+0.325 CI가 0을 제외**(P=0.997), **대형 구간 Δ+0.001 무해**(P=0.504). **B1과 B2는 상호 필수** — B2 단독(legacy)은 n=1 bag이 0벡터가 되어 NaN 그라디언트로 **학습 자체가 불가**, B1 단독(S1)은 구간 교환으로 음성이었다. **v30 승격 후보 — 승격은 사용자 확인 필요**(`bag_representation` 기본 `legacy`, v24 무변경). 부산물: **선형 ridge 천장은 예측 지표로 폐기**(부호가 양쪽 반대였음). **다음 최약 구간은 n>34(0.698)** — 0.95까지 남은 +0.096의 대부분. **§26에서 Musk 로드맵을 재설정**했습니다 —
`musk095_architecture_proposal.md`의 **P1/P2는 기각, P3는 연기**. 진짜 병목은 "입력이 bag 평균을
버린다"가 아니라 **per-bag centering이 작은 bag(Musk median 12, n≤4가 28%)을 rank 결핍으로
소멸시키는 것**입니다 — 소형 bag에서 현 표현 **0.500** vs pool-z **0.900~0.967**, 그리고 학습된 4개
체크포인트 전부 소형 bag에서 **무작위 이하**(0.325~0.500).
**해법 후보 확정: `poolz`(context-pool 대각 표준화)** — 현 표현 대비 합성 대형 −0.019 / 합성 소형
+0.089 / **Musk +0.115**. `poolz_l2`는 Musk 최대(0.912)지만 합성 대형 −0.099이며 그 손실은 생성기
인공물(`normalize_output: true`)이라 **B4가 해금 조건**입니다.
현재 최고는 여전히 **Musk zero-shot AUROC 0.822** (musklike-easy + `--preprocess raw`).
지렛대 2(§23)·3(§24 IA-MIL)은 음성이나 **§24 게이트는 무효**로 판정(생성기에 any-positive 과제 부재
+ IA-MIL 크기 편향 +0.327).

> **사용자 결정 (2026-08-04, 확정 — 재논의 불필요)**:
> 1. **ICI는 손대지 않습니다.** §26의 cell-축 zero-padding 미마스킹 결함은 **기록만 유지**하고
>    수정하지 않으며, 향후 ICI 잠금 해제 시 **반드시 먼저 처리할 항목**으로 남깁니다.
> 2. **Musk 목표는 0.95 유지.** §26 초판의 "0.90 하향" 권고는 **철회**됐습니다 — 선형 ridge 천장은
>    모델의 상한이 아니며(모델이 자기 입력의 선형 천장을 +0.055~0.072 일관 초과), 0.95에 필요한
>    선형 기반 ≈0.89는 `poolz_l2`(0.912) 아래입니다.

**열린 과제**: ① B1을 v30으로 승격해 living 문서 갱신할지, ② B1 형태를 `poolz` 하나로 갈지
`poolz_l2`도 함께 학습할지, ③ P1(읽기 브리지) 완전 보류 여부. 이전 결과: v26 0.5908(동률, 미채택),
no-L2 음성(0.5925), v25 폐기.
**Read first if you are picking this up**: **§26 (Musk 재진단·v30 제안 — 최신)** 및
[`history/musk_transfer_diagnosis_v30_proposal.md`](history/musk_transfer_diagnosis_v30_proposal.md),
그다음 **§28 (실행 중인 v30 S1 — PID·로그·게이트)**, §27 (문서 압축·재현성 복구 기록), §25 (IA-MIL 폐기·정리), §3 (실험 현황·최종 결정),
§5 (실험 전략), §7 (평가 프로토콜).
해결·폐기된 세션/실험 기록(**§6**, §11~§24, §22 musk-like easy 가설 / §23 raw-stat 음성 /
§24 IA-MIL 음성 포함)은 [`history/archive.md`](history/archive.md) 참고.
**Branches**: `main` = `v24` 확정 (현재 SSOT) / 참고용 `v22`·`v24`·`v19`·`codex/v23-bag-mean` / v25는 태그 **`v25-typed-bag-final`**로 보존 (브랜치 삭제) — 구조: [`history/branch_structure.md`](history/branch_structure.md)
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)
**Architecture Version**: `24` 확정 baseline (변경 없음). `26`(CLS-token pooling)은 2026-08-02 구현,
2026-08-03 학습 완료(val_ce 0.5908, v24와 동률 → **미채택**). `25`(T5-A) 폐기. `use_instance_attention_mil`
(IA-MIL)은 **2026-08-04 폐기** (§25). `22`/`23` 폐기된 구버전.
**Purpose**: 연구실 / 집 / 노트북 3개 작업 환경 간 대화 기록 비동기화 문제를 해결하기 위한 Single Source of Truth (SSOT) living document.

---

## 0. 30초 요약 — 새 세션은 여기부터

> **2026-08-04 갱신**: Phase 1 IA-MIL(`use_instance_attention_mil`) **폐기** — 합성 무회귀(0.9520)지만
> rare 판별 유의 열위(baseline 0.9492 vs mil 0.9224, paired P=1.00), Musk zero-shot 큰 회귀
> (0.8030→0.5545). configs/scripts 아카이빙, 해결된 §11~§24 섹션은 `docs/history/archive.md`로 정리.
> **v24가 확정 baseline 유지.** 현 최고: **지렛대 1(centered 입력 표현) Musk zero-shot AUROC 0.822**
> (musklike-easy 학습). 열린 과제: ① Phase 2 읽기 브리지(IA-MIL 제외) 여부, ② ICI 잠금, ③ Musk 0.95
> 달성 전략. (이전: v25 폐기, v26 미채택, no-L2 음성)

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

## 28. 2026-08-04 — v30 S1 실행 중: B1 `poolz` / `poolz_l2` 학습 + 자동 평가 큐

**상태**: **백그라운드 실행 중** (구현·테스트 완료, 커밋 `b3808cb`). 아키텍처 플래그는 기본 OFF이므로
**v24 확정 baseline은 그대로**이며, 승격 판정은 아래 게이트 통과 후 별도로 받습니다.

**구현** (`bag_representation`: `"legacy"`(기본) | `"poolz"` | `"poolz_l2"`):
per-bag centering을 **context-pool 대각 표준화**로 대체합니다 — `z = (x − μ_ctx)/σ_ctx`,
`poolz_l2`는 여기에 per-cell L2. `classification_instances`만 바뀌고 `global_summary`(per-bag
centered spread)와 `centered_delta`는 그대로라 **공분산/spread 분기는 v24와 동일**합니다.
pool 통계는 **context bag의 모든 세포**에서 cell-count 가중(실 bag 크기 1~1044 → per-bag 평균의
평균은 오가중), query 누출 없음. 주입 지점 3곳: `forward_episode_batch`(에피소드별),
`aggregator.forward`(`context_mask` 검증을 `_bag_view` 앞으로 이동), `BaseModel.forward`
(`is_context`를 중복 `_bag_view` 앞으로 이동 — 안 하면 rare-evidence/MIL 분기가 slot 분기와 다르게
정규화된 세포를 받음).

**검증**: unittest **164/164 통과**(136.6s, 신규 10건 — 기본값 legacy 유지, 잘못된 플래그 조합 거부,
pool 통계 없으면 raise, poolz는 magnitude 보존/poolz_l2는 단위 노름, pool 통계가 query bag 무시,
batched==list pool 통계 및 forward 일치(≤1e-4), forward/backward 유한, **n=1 bag이 legacy에서는
0벡터지만 poolz에서는 아님**). 1-epoch 실 스모크(poolz) `val_loss 0.436`(v24 musklike-easy 스모크
0.485), NaN·크래시 없음.
또한 `_context_pool_stats`를 `unbiased=False`로 batched twin과 정확히 일치시켰습니다 — Bessel 보정을
쓰면 학습(batched)과 실데이터 추론(list)이 **~0.25% 조용히 어긋납니다.**

**실행 중인 큐** (`scripts/queue_v30_poolz.sh`, 단일 B200이라 순차):
train 50 epoch → 1,000-ep 합성 eval → Musk zero-shot eval → cardinality 층화 리포트, 2개 run 연속.

| 항목 | 값 |
|---|---|
| Run 1 | `v30_musklike_easy_poolz` — **진행 중** (10:02:34 시작) |
| Run 1 PID / 로그 | `2174854` / `logs/20260804_100234/v30_musklike_easy_poolz.out` |
| Run 1 체크포인트 | `checkpoints/20260804_100234/v30_musklike_easy_poolz/` |
| Run 2 | `v30_musklike_easy_poolz_l2` — Run 1 종료 후 자동 시작 |
| 오케스트레이터 PID / 로그 | `2183035` / `logs/queue_v30_poolz.log` |
| 진행 상황 | epoch 1에서 `val_loss 0.409`, GPU 114GB/77%, ~105s/epoch → 50 epoch ≈ 90분/run |
| 예측 산출물 | `predictions/synthetic_v30_musklike_easy_poolz{,_l2}_1000ep.pt`, `predictions/musk_v30_musklike_easy_poolz{,_l2}.pt` |

**사전 등록 게이트** (config 주석에도 명시):
- `poolz`: 합성 musklike-easy 1,000-ep AUROC **≥0.94**(현 0.9510 무회귀) **AND** Musk 전체 **≥0.89**
- `poolz_l2`: 합성 **≥0.94** **AND** Musk 전체 **≥0.92**
- **층화 보고 필수**(n≤4 / 5–10 / 11–34 / n>34). 현 최고는 Musk 0.822이고 n≤4가 0.475이므로,
  소형 구간 개선이 없으면 전체 수치만으로는 무엇이 개선됐는지 판정 불가.
- 근거 천장(§26): `poolz` Musk 0.874 / n≤4 0.900, `poolz_l2` 0.912 / 0.967. 모델은 자기 입력의
  선형 천장을 +0.055~0.072 초과해 왔으므로 실현 기대치는 각각 **≈0.93 / ≈0.97**입니다.

**큐 스크립트 버그 수정 (실행 중 발견)**: 아카이브에서 계승한 `pgrep -f "scripts/train.py"`는
**그 문자열을 명령줄에 포함한 모든 프로세스**(예: 큐 상태를 확인하는 `pgrep -af scripts/train.py`
쉘)를 매칭해, 큐가 학습이 살아있다고 오판하고 `wait_gpu_free`에서 영구 대기했습니다.
`/proc/<pid>/comm`으로 **실제 python/torchrun 프로세스만** 필터링하도록 수정했습니다. 또한 launcher가
학습을 detach하므로 오케스트레이터를 재시작해도 중복 실행되지 않게 `ICF_ATTACH_FIRST=1`
(진행 중인 첫 run을 입양해 완료 대기 후 평가) 모드를 추가했습니다.

**다음 Action**: 큐 완료 후 ① 게이트 판정(합성 무회귀 + Musk 전체·층화), ② 통과 시 v30 승격 여부와
`poolz` vs `poolz_l2` 선택을 사용자 확인, ③ 미달 시 S2(B2 cardinality-faithful 샘플링)로 진행 —
B2는 아키텍처 무변경이며 소형 bag을 학습 분포에 넣어 B1의 천장을 실제로 실현시키는 단계입니다.
진행 확인: `timeout 5s tail -5 logs/queue_v30_poolz.log`.

### §28 결과 (1) — `poolz`: **게이트 미달, 음성** (2026-08-04 11:39)

50 epoch 정상 완주(`max_epochs=50 reached`, launcher "completed successfully"), best
`val_ce_loss` **0.2374** @ epoch 43 — musklike-easy 계열 **최저**(v24 0.2552, rawstats 0.2468,
IA-MIL 0.2462). **그런데 판정 지표는 통과하지 못했습니다.**

| 지표 | `poolz` | v24 musklike-easy (기준) | 게이트 | 판정 |
|---|---|---|---|---|
| 합성 1,000-ep AUROC | **0.9506** [0.946,0.955] | 0.9510 [0.946,0.956] | ≥0.94 | ✅ 무회귀(동률) |
| 합성 log loss | 0.2837 | 0.2836 | — | 동률 |
| **Musk zero-shot AUROC** | **0.7757** [0.671,0.866] | **0.8030** / 0.8217(raw) | **≥0.89** | ❌ **미달, 기준선보다 낮음** |
| Musk log loss | 0.6941 | 0.5439 | — | ❌ 크게 악화 |

층화(`--report stratified`): n≤4 **0.442**(v24 0.475, **개선 없음**) / 5–10 0.833(0.825) /
11–34 0.927(0.988) / n>34 **0.540**(0.667). **pearson(prob, log n) = +0.327** — §24 IA-MIL의
크기 편향 지문(+0.327)과 동일 수치, baseline은 +0.012.

> [!IMPORTANT]
> **판정: `poolz` 음성.** 예측(선형 천장 0.874 + 모델의 관측 초과폭 +0.06 → ≈0.93)이 실현치
> 0.776으로 **−0.15 어긋났습니다.** §26/§3.1에서 "초과폭이 새 표현에서도 유지된다는 것은 **가정**"
> 이라고 명시했던 그 가정이 **깨졌습니다.**
>
> **측정된 실패 메커니즘** — `‖분류 뷰의 bag 평균‖` vs bag 크기 n (Musk 102 bag, 모델 불필요):
>
> | 뷰 | corr(log‖·‖, log n) | log-log 기울기 | n≤4 중앙값 | n>34 중앙값 | 배율 |
> |---|---:|---:|---:|---:|---:|
> | legacy (center+L2) | **+0.528** | +2.67 | **0.000** | 0.028 | — |
> | `poolz` | **−0.535** | −0.153 | 11.43 | 6.66 | 1.7× |
> | `poolz_l2` | **−0.611** | −0.141 | 0.974 | 0.507 | 1.9× |
>
> legacy는 소형 bag의 **1차 적률을 0으로 없애고**(n=1이면 정확히 0, n=2면 대척쌍이라 합이 0),
> `poolz`는 그것을 **보존하지만 크기 n과 결합**시킵니다(표본 평균의 수축; feature 상관이 높아
> iid의 −0.5보다 완만한 −0.15). 즉 `poolz`는 "소형 bag에 신호가 없음"을 "신호는 있으나 **bag
> 크기와 교란됨**"으로 바꿨을 뿐입니다. 모델은 **에피소드 내 모든 bag이 동일 크기이고 n이 항상
> 500~1000인** 분포에서만 학습했으므로 이 교란을 분리하는 법을 배운 적이 없고, 그 결과가
> prob-vs-log n **+0.327**과 소형 구간 무개선입니다.
>
> **선형 ridge 천장이 예측력을 갖지 못한 이유도 이제 분명합니다**: ridge는 feature별로 재스케일할
> 수 있고 **n에 대해 평균하지 않습니다.** 반면 모델의 slot 통계는 bag의 n개 세포를 **평균**하므로
> 1/√n 계열 수축이 그대로 descriptor 스케일에 들어옵니다. 천장 프로브는 이 경로를 모사하지 않습니다.

**계획 수정 (자기 정정)**: §3.0에서 "`poolz`의 합성 대형 비용이 −0.019뿐이므로 **B2는 선행조건이
아니라 증폭 요인**"이라며 순서를 B1→B2로 바꿨는데, **그 철회가 틀렸습니다.** 원래 순서
**B2 → B1이 옳습니다** — 단 이유는 처음 생각한 "합성 천장 손실" 때문이 아니라, **magnitude를
보존하는 어떤 표현이든 bag 크기와 교란되며, 모델이 크기 불변성을 배우려면 학습 분포에 크기 변동이
있어야 한다**는 것입니다. B2(bag별 log-uniform[1,1024] 샘플링)가 **B1의 전제조건**입니다.

**`poolz_l2`(진행 중)에 대한 기대치 하향**: 위 표에서 `poolz_l2`도 corr **−0.611**로 같은 크기
교란을 보입니다(절대 크기만 1/12로 작음). per-cell L2로 magnitude를 유계화해도 **상대적 크기
의존성은 남습니다.** 따라서 통과를 예상하지 않습니다 — 다만 이미 학습 중이므로
"uncentering 자체"와 "magnitude 스케일"을 분리하는 값싼 데이터포인트로 완주시킵니다.

### §28 결과 (2) — `poolz_l2`: **게이트 미달. S1(B1 단독) 전체 음성 판정** (2026-08-04 13:21)

50 epoch 정상 완주, best `val_ce_loss` **0.2414** @ **epoch 49(마지막)** — epoch 43은 0.2433이었으므로
**아직 개선 중이었습니다(미수렴)**. 50 epoch가 이 변형에는 부족할 수 있다는 단서로 기재합니다.

**S1 최종 요약** (모든 수치 1,000-ep 합성 / 102-bag Musk LOO):

| run | 합성 AUROC | 합성 LL | **Musk 전체** | **n≤4** | 5–10 | 11–34 | **n>34** | corr(prob, log n) | 게이트 |
|---|---|---|---|---|---|---|---|---|---|
| v24 musklike-easy (기준) | 0.9510 | 0.2836 | **0.803** | 0.475 | 0.825 | 0.988 | 0.667 | +0.012 | — |
| `poolz` | 0.9506 | 0.2837 | 0.776 | 0.442 | 0.833 | 0.927 | 0.540 | **+0.327** | ❌ (≥0.89) |
| `poolz_l2` | **0.9520** | **0.2803** | 0.762 | **0.517** | 0.767 | 0.976 | **0.460** | +0.100 | ❌ (≥0.92) |

> [!IMPORTANT]
> **판정: B1(uncentering) 단독은 음성.** 두 변형 모두 합성 무회귀는 통과했으나(오히려 `poolz_l2`가
> 최고 0.9520) **Musk에서 기준선 0.803보다 낮습니다**(0.776 / 0.762). 승격하지 않으며
> `bag_representation` 기본값은 `legacy`로 유지합니다. **v24 확정 baseline 변경 없음.**
>
> **다만 실패의 모양이 진단을 지지합니다** — uncentering은 **예측한 곳에서 정확히 이겼고 다른 곳에서
> 잃었습니다**: `poolz_l2`의 n≤4는 **0.517로 6개 체크포인트 중 최고**(v24 0.475, 기준선 유일 초과)이고,
> magnitude를 유계화한 덕에 크기 편향도 +0.327 → **+0.100**으로 줄었습니다(제 예측 방향 일치).
> 그런데 **n>34에서 0.667 → 0.460으로 붕괴**해 총합이 음수가 됐습니다.
>
> **해석**: 모델은 **에피소드당 cell 수가 하나로 고정되고 항상 500~1000인** 분포에서만 학습했으므로
> **크기 불변성을 배운 적이 없고, 소형·대형 두 레지므를 동시에 만족시킬 수 없습니다.** 표현을 소형
> 쪽으로 옮기면 대형 쪽을 잃습니다. 이것이 §28 결과(1)에서 재확정한 **B2 선행 필요성**의 두 번째,
> 그리고 더 직접적인 증거입니다.
>
> **또 하나의 교훈 — 선형 ridge 천장은 이 모델의 예측 지표가 아닙니다.** `poolz_l2`는 합성 천장이
> −0.099로 가장 나빴는데 실현 합성 AUROC는 **최고**(0.9520)였고, Musk 천장은 0.912로 가장 좋았는데
> 실현 Musk는 **최악**(0.762)이었습니다. **부호가 양쪽 다 반대입니다.** 향후 천장 프로브는 방향
> 탐색용으로만 쓰고 **기대치 산출 근거로 쓰지 않습니다**(§26의 "+0.06 초과폭" 추정도 폐기).

**B2 실행 가능성 확인 (모델 불필요)**: 소형 bag 가드는 `pooled context candidates < num_slots(12)`
이며 **per-bag 제약이 아닙니다**(`baseline.py:302`, `:939`). 따라서 bag 수가 충분하면 n=1도 통과합니다:

| bags/episode | n=1 | n=2 | n=3 | n=4 | n=8 |
|---|---|---|---|---|---|
| 6 | ❌ ValueError | ❌ | ✅ | ✅ | ✅ |
| 20 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 60 | ✅ | ✅ | ✅ | ✅ | ✅ |

`num_bags: [60, 100]`이므로 **`num_cells: [1, 1024]` 전 구간 사용 가능**합니다. (Musk가 지금도 n=1 bag을
처리하는 이유도 동일 — context가 101 bag·~6,500 세포.)

**B2 구현**: `num_cells_log_uniform`(기본 OFF, 기존 경로 무변경). uniform은 소형 bag을 사실상 못 보므로
log-uniform이 필수입니다:

| 분포 | median | frac(n≤12) | frac(n≤34) |
|---|---|---|---|
| uniform [1,1024] | 510 | 0.01 | 0.03 |
| **log-uniform [1,1024]** | **34** | **0.36** | 0.50 |
| Musk 실제 | 12 | 0.50 | 0.75 |

(log-uniform이 실제 Musk보다 상단이 두꺼운 잔여 불일치는 그대로 기재합니다.)
**남은 구조적 격차**: 4D batched 경로와 `shape_group_size`가 에피소드 내 동일 shape을 요구하므로
B2는 **에피소드 간** 크기 변동만 줍니다 — Musk의 **에피소드 내** 혼합(1~1044 동시)은 아닙니다.

**다음 (S2, 큐 예정)**: ① `v30_cardinality_poolz_l2`(B2+B1) ② `v30_cardinality_legacy`(B2 단독 대조 —
단 legacy는 n=1을 0벡터로 만들므로 최소 에피소드가 퇴화 뷰로 학습됨, 그 비대칭이 비교의 요점).
게이트: 합성 ≥0.94 **AND** Musk ≥0.85 **AND** n≤4 ≥0.60 **AND** n>34가 0.60 미만으로 떨어지지 않을 것
— S1이 실패한 방식(한 구간을 다른 구간과 교환)을 전체 수치만으로는 구분할 수 없으므로 층화 필수.

### §28 (3) — S2 실행 시작 (2026-08-04 13:23)

| 항목 | 값 |
|---|---|
| Run 1 | `v30_cardinality_poolz_l2` (B2+B1) — **진행 중** (13:23:34 시작, PID `2617890`) |
| Run 1 로그 / 체크포인트 | `logs/20260804_132334/` / `checkpoints/20260804_132334/v30_cardinality_poolz_l2/` |
| Run 2 | `v30_cardinality_legacy` (B2 단독 대조) — Run 1 종료 후 자동 |
| 큐 로그 | `logs/queue_v30_s2.log` |
| 스모크 | 1-epoch 실 루프 정상(`val_loss 0.577`), 가변 shape 크래시 없음 |

진행 확인: `timeout 5s tail -5 logs/queue_v30_s2.log`

### §28 결과 (4) — **S2 `v30_cardinality_poolz_l2`(B2+B1): 게이트 통과, 첫 양성 결과** (2026-08-04 14:37)

50 epoch 정상 완주. best `val_ce_loss` 0.4442 @ 48 — **S1 수치(0.2374/0.2414)와 비교 불가**입니다
(데이터 분포가 log-uniform으로 바뀌어 bag당 세포가 훨씬 적으므로 CE가 당연히 높습니다).

| 지표 | B2+`poolz_l2` | v24 기준선 | S1 `poolz_l2` | 게이트 | 판정 |
|---|---|---|---|---|---|
| **Musk 전체** | **0.8539** [0.774,0.925] | 0.8030 | 0.762 | ≥0.85 | ✅ |
| **Musk n≤4** | **0.800** [0.52,0.98] | 0.475 | 0.517 | ≥0.60 | ✅ |
| Musk 5–10 | 0.833 | 0.825 | 0.767 | — | ✅ |
| Musk 11–34 | 0.958 | 0.988 | 0.976 | — | ~ |
| **Musk n>34** | **0.698** | 0.667 | 0.460 | ≥0.60 | ✅ |
| corr(prob, log n) | **+0.059** | +0.012 | +0.100 | — | ✅ 편향 거의 없음 |
| Musk log loss | **0.4746** | 0.5439 | 0.6001 | — | ✅ 보정 개선 |
| Balanced acc | **0.7845** | 0.7747 | 0.7173 | — | ✅ |
| 합성(log-uniform 분포) | 0.8512 [0.840,0.862] | — | — | — | ⚠ 아래 주석 |

**paired bootstrap** (`--report paired`, 4,000 resample; 두 모델이 **같은 102 bag**을 채점하므로
독립 CI 비교보다 검정력이 높음 — n=102에서 독립 CI는 크게 겹칩니다):

| stratum | bags | v24 | B2+`poolz_l2` | Δ | 95% CI of Δ | P(cand>base) |
|---|---|---|---|---|---|---|
| ALL | 102 | 0.803 | 0.854 | **+0.051** | [−0.002, +0.111] | **0.971** |
| **n≤4** | 29 | 0.475 | **0.800** | **+0.325** | **[+0.092, +0.567]** | **0.997** |
| n>4 | 73 | 0.863 | 0.863 | **+0.001** | [−0.054, +0.060] | 0.504 |

> [!IMPORTANT]
> **진단이 확증됐습니다.** 개선이 **예측한 지점에 정확히 국소화**되어 있습니다 —
> **소형 bag Δ+0.325, CI가 0을 제외**(P=0.997), 그리고 **대형 bag은 Δ+0.001로 완전 무해**(P=0.504).
> 전체 Δ+0.051의 CI가 0을 살짝 포함하는 것은 **102개 중 28개(n≤4)만 고장나 있었기 때문**이며,
> 부분집합만 고쳤으니 전체 지표에서 희석되는 것이 정상입니다.
>
> 인과 사슬이 전부 맞았습니다: **cardinality가 병목** → **B1 단독은 구간 교환**(S1, 소형 +0.04 /
> 대형 −0.21) → **B1+B2는 소형을 고치면서 대형을 잃지 않음**(+0.325 / +0.001).
> 현 최고 기록도 경신했습니다: 0.822(`--preprocess raw`) → **0.854**.

**⚠ 남은 게이트 항목 — 합성 무회귀는 아직 미검증**: 위 합성 0.8512는 **새 log-uniform 분포**에서
측정된 값이라 원래 기준(0.9510, n=500~1000)과 **비교할 수 없습니다**. `evaluate_synthetic.py`는 넘겨받은
config로 val 데이터셋을 만들기 때문입니다. 이를 위해 **모델은 고정하고 데이터만 원래대로 돌리는**
eval 전용 config를 준비했습니다: `configs/eval_v30_cardinality_{poolz_l2,legacy}_on_largebags.yaml`.
Run 2 종료 후 실행합니다.
> **주의(문서화됨)**: `bag_representation`은 **파라미터를 추가하지 않으므로**, poolz_l2 체크포인트를
> legacy config로 로드해도 **에러 없이 조용히 잘못된 뷰로 평가**됩니다. 위 eval config가 표현을
> 명시적으로 고정하는 이유입니다. (큐 자체는 항상 학습 config를 넘기므로 영향 없음.)

**진행 중**: Run 2 `v30_cardinality_legacy`(B2 단독 대조, 14:36:49 시작) — B2 단독으로 충분한지,
아니면 B1이 필요한지 귀속시킵니다.

### §28 결과 (5) — **S2 최종 판정: 게이트 전 항목 통과. B1+B2는 상호 필수** (2026-08-04 14:45)

**Run 2 `v30_cardinality_legacy`(B2 단독 대조)는 학습 자체가 불가능했습니다** — epoch 0, optimizer
step 490에서 프로젝트 자체 가드(`src/modules/model_interface.py:76`)가 발동:

```
RuntimeError: Non-finite gradients at epoch=0, optimizer step=490:
['model.aggregator.slot_residual_logit', 'model.aggregator.center_slot_encoder.0.weight', ...]
```

원인은 유닛테스트로 고정해 둔 그 사실입니다 — **legacy(center+L2)는 n=1 bag을 정확히 0벡터로,
n=2를 대척쌍으로 만듭니다.** log-uniform 샘플링에서 그런 에피소드가 다수 발생하므로 all-zero 뷰가
slot encoder 경로로 흘러 NaN 그라디언트가 됩니다. 즉 **B2 단독은 "성능이 낮다"가 아니라
v24 표현으로는 원리적으로 학습 불가**입니다.

> [!IMPORTANT]
> **B1과 B2는 상호 필수(mutually required)입니다.**
> B2는 학습 가능하려면 uncentered 뷰가 필요하고(위 NaN), B1은 구간 교환을 피하려면 B2가 필요합니다
> (S1: 소형 +0.04 / 대형 −0.21). **실패한 대조군이 귀속을 완성했습니다** — 이 조합의 이득은 어느 한
> 쪽으로 환원되지 않습니다.

**교차 분포 평가 (모델 고정, 데이터만 교체 — 마지막 게이트 항목)**:

| 체크포인트 | 평가 데이터 | AUROC | log loss | 비교 |
|---|---|---|---|---|
| **B2+`poolz_l2`** | **원래 대형 bag** (n=500~1000) | **0.9483** [0.944,0.953] | 0.2987 | v24 **0.9510** [0.946,0.956] → **Δ−0.003, 무회귀** |
| B2+`poolz_l2` | 신규 log-uniform | 0.8512 [0.840,0.862] | 0.4658 | ↓ |
| v24 musklike-easy | 신규 log-uniform | 0.8328 [0.820,0.845] | 0.4977 | → B2 학습이 **+0.018** 우위 |

**S2 Run 1 사전 등록 게이트 — 전 항목 통과**:

| # | 게이트 | 결과 | 판정 |
|---|---|---|---|
| 1 | 합성 무회귀 ≥0.94 (원래 분포) | **0.9483** (v24 0.9510, Δ−0.003) | ✅ |
| 2 | Musk 전체 ≥0.85 | **0.8539** [0.774,0.925] | ✅ |
| 3 | Musk n≤4 ≥0.60 | **0.800** (v24 0.475) | ✅ |
| 4 | Musk n>34 ≥0.60 유지 | **0.698** (v24 0.667) | ✅ |
| 5 | paired: 대형 구간 무해 | Δ**+0.001**, P=0.504 | ✅ |
| 6 | paired: 소형 구간 유의 개선 | Δ**+0.325**, CI [+0.092,+0.567] **0 제외** | ✅ |

> **판정: S2(B2 + B1 `poolz_l2`)는 사전 등록 게이트를 전부 통과했습니다.** Musk 최고 기록
> 0.822 → **0.854** 경신. **v30 승격 후보이며, 승격 여부는 v26/v27/v29와 동일한 규율에 따라
> 사용자 확인이 필요합니다** (`bag_representation` 기본값은 현재도 `legacy`, v24 baseline 무변경).

**다음 목표가 바뀌었습니다 — 이제 가장 약한 구간은 대형 bag입니다**:

| 구간 | v24 | **B2+B1** | 상태 |
|---|---|---|---|
| n≤4 (29) | 0.475 | **0.800** | 해결됨 (원래 병목) |
| 5–10 (22) | 0.825 | 0.833 | 양호 |
| 11–34 (25) | 0.988 | 0.958 | 양호 |
| **n>34 (26)** | 0.667 | **0.698** | ⚠ **최약 구간** |

0.95 목표까지 남은 거리는 **+0.096**이고 그 대부분이 **n>34 구간**에 있습니다. 원안의 다음 단계를
그대로 따르기보다 이 구간을 직접 겨냥하는 것이 타당합니다. 후보:
1. **B2b — 에피소드 내 cardinality 혼합**: 현재 B2는 4D batched 경로와 `shape_group_size` 제약 때문에
   **에피소드 간** 변동만 줍니다. Musk는 한 에피소드에 1~1044이 **동시에** 존재하므로, list 경로 또는
   크기 버킷팅으로 에피소드 내 혼합을 구현하면 대형·소형 동시 처리를 직접 학습시킬 수 있습니다.
2. **대형 bag 특이 진단**: n>34에서 무엇이 실패하는지 미규명입니다(§26 §6.3의 미해결 항목과 동일).
   top-1% tail이 n과 함께 커지며 신호가 희석되는지 먼저 측정할 것.
3. B3(2차 통계 shrinkage) / B4(생성기 any-positive)는 그 다음.
