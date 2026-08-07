# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-07` (**§59 v35 제안서 rev.2 전면 개정(§58 결정 3건 중 2건 폐기 권고 — anchor 오염·집계 수학 오류·Musk 소형 bag 파괴·구현 불가) + 정확 스트리밍 구현(peak VRAM 2.16× 절감, 수치 동일 검증) + VRAM 가드 버그 수정 + v35 학습 시작(2×B200, 진행 중) + ⚠️ 완료된 50-fold 9개 수치가 `5869535` 이후 stale** + §58 v35 설계(rev.1) + **§57 case leakage 진단** + §56 config 리팩터링)  
**Status**: **v30 확정 baseline 유지, CCER 계열 폐기**. arm C top-up **150 epoch 완주**(8×A6000 DDP, best `epoch=125-val_ce_loss=0.5142.ckpt`). 완주 후 §42 재평가: legacy overall **0.8100 [0.798, 0.822]** vs v30 committed 0.8512 → **회귀 +0.0412로 gate 미달** — val_ce는 0.5351→0.5142로 개선됐지만 legacy AUROC는 50ep(0.8139)와 동일 → **과소학습 편향 가설 기각, B2b 데이터 자체가 회귀 원인**. Musk는 n>34 0.698→0.849(개선 유지)·5..10 0.833→0.958, n≤4 0.800→0.725(trade-off), overall +0.008(무의미). PathoBench all-context 5-task는 **v30이 4/5 우위(평균 +0.039)**, 유일한 e125 승리 lscc_arid1a(+0.117). **Phase 0 두 주 효과 모두 gate 미달 확정 → v30 baseline 유지, arm C 미채택.**
* **v32b 결론**: donor-resolved evidence도 v30에 보완 정보를 추가하지 못했다. Stage B 이후는 실행하지 않는다.
* **v34 확정 (§52·§53·§56)**: **v34-1536을 PathoBench 보고용 모델로 확정**(사용자 결정). 평가는 **공식 Patho-Bench 프로토콜**(공식 k=all.tsv fold·코호트·라벨) 기준 **50-fold**(SEAL macro-AUC와 동일 구조) — **5/17 완료**(bc_therapy er 0.672 / grade 0.713 / her2 0.670, cptac_brca_PIK3CA 0.569, brca_TP53), **12개는 config 수정으로 재시작**(§56, 백그라운드). v30은 합성/Musk baseline 유지. 이전 5-fold와 수치 ±0.04 이내 동일(평가 견고성). config 시스템을 v34 base + group default 참조형으로 리팩터링(§56). 자세한 진행 §53·§56.
* **v35 (§58 rev.1 설계 → §59 rev.2 개정 + 학습 시작)**: rev.1의 3개 결정 중 **①rare 제거·③context/query 대형화 분리는 폐기 권고**(§59.1: anchor 오염, 집계 수학 오류, Musk 소형 bag 파괴 = 확정 목표 위반, query 위치를 dataset이 알 수 없어 구현 불가, 그리고 동기 자체에 직접 반증 — context 2k cap Δpooled **−0.0019**). ②chunk는 **근사 평균이 아닌 정확 충분통계 축약**으로 재설계. 구현·검증 완료분: **bag 단위 정확 스트리밍**(peak VRAM 40,990 → 18,930 MiB, AUROC 동일), `num_cells_log_uniform_power`, VRAM 가드 `episode_batch_size` 누락 버그 수정, **41 tests**. **학습 진행 중**: 데이터 단독 arm(`num_cells [1,32768]` power 1.5, rare branch 유지), `logs/20260807_203606/`, 2×B200 GPU 0·1, 51,200 episodes(v34와 에피소드 매칭).
* **다음 Action**: ① **v35 학습 완주 → 공식 50-fold 평가**(단 §59.5: **v34도 현재 코드로 재실행**해야 공정 비교), ② **P0 게이트(무료, 학습 0)** — query 크기 스윕 + `rare_logits=0` ablation; 전자가 +0.005 미달이면 대형화 노선 폐기(rev.2 §4), ③ 공식 50-fold **잔여 8개**(스트리밍으로 workers 2 → 8+ 가능), ④ v30 vs v34 공정 비교용 **PCA-per-fold CV**(미지원), ⑤ **v34-512 학습**, ⑥ rev.2 §3의 **chunk 단위**(bag 내부) 스트리밍 — 현재는 bag 단위까지만, ⑦ rev.2 §8 zero-init chunk-attention(ABMIL 격차 대응).

> **사용자 결정 (2026-08-05, 확정)**:
> 1. **v30 S2가 정식 확정 baseline 유지.** v31 CCTS/CCER-v2는 정식 baseline으로 승격/채택하지 않음 (실험 후보 기록만 남김).
> 2. **ICI는 손대지 않습니다.** (잠금 유지)
> 3. **Musk 목표는 0.95 유지.**

**Read first if you are picking this up**: **§59 (v35 rev.2 개정 + 스트리밍 구현 + 학습 시작 + 50-fold stale 경고)**, **§58 (v35 rev.1 설계 — §59가 상당 부분 정정하므로 §59와 함께 읽을 것)**, **§57 (50-fold case leakage 진단)**, **§53 (v34 확정 + 공식 50-fold 표 — §59.5에 따라 재실행 필요)**,

**열린 과제**: ① **v35 학습 완주 + 평가**(§59.6-7), ② **P0 게이트**(query 크기 스윕 / rare ablation, 무료), ③ **§53 표 9개 재실행**(`5869535` 이후 stale, §59.5) + 공식 50-fold 잔여 8개 → 17개 최종 표 + SEAL 재비교, ④ v30 vs v34 CV 공정 비교(PCA-per-fold), ⑤ v34-512 학습, ⑥ **chunk 단위(bag 내부) 스트리밍** 미구현(rev.2 §3), ⑦ v30 six-task / B2b cardinality 효과 분리, ⑧ frozen-v30 multi-resolution headroom, ⑨ v30 medium 참조 재학습. 해결·폐기 기록은 [`history/archive.md`](history/archive.md).

**Branches**: `main` = v30 확정 baseline + 미채택 v31 CCER-v2 재현 코드. 참고용 branch/tag 구조는
[`history/branch_structure.md`](history/branch_structure.md).
**Project**: ICF (BagPFN Single-Cell In-Context Meta-Classifier)
**Architecture Versions**: `30` 확정 baseline; `31` CCER-v2와 `32` DR-CCER 미채택(재현용 보존);
`33` MR-BagPFN은 proposal-only. 코드 기본 `bag_representation`은 `legacy` 유지.
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
> 2. 새로 접속한 Agent는 **`docs/` 최상위 루트의 Living md 파일 5개(`agent_handoff.md`, `current_status.md`, `current_architecture.md`, `current_experiments.md`, `README.md`)와 현행 `architecture_*_proposal.md` 1개를 최우선으로 정독**하여 전체 개발 맥락과 프로젝트 규칙을 파악합니다.
> 3. 터미널 조회가 필요한 명령어는 NVML/쉘 hang 방지를 위해 **반드시 `timeout 3s ps aux | grep python`과 같이 타임아웃**을 적용합니다.
> 4. 코드 변경 시 unittest 통과 필수:
>    `timeout 300s /home/aibio_3/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"`

---

## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture **v24 확정**)

* **Python Binary**: `/home/aibio_3/miniconda3/envs/BagPFN/bin/python`
* **Torchrun Binary**: `/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun`
* **Target Hardware**: **8× NVIDIA B200 노드, 사용 GPU 0·1 (2장)** (`CUDA_VISIBLE_DEVICES=0,1`, 183GB VRAM/장) —
  2026-08-07 8-GPU 컨테이너 전환, 워크스페이스 `/NHNHOME/BASE/kimds/ICF`, conda
  `/home/aibio_3/miniconda3/envs/BagPFN` (이전 `/NHNHOME/kimds` 경로 폐기)
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

`scripts/archive/diagnostics/diagnose_state_upper_bound.py`, 1,000 validation episodes 중 state 177 episodes / 2,910 query, episode cluster bootstrap:

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
/home/aibio_3/miniconda3/envs/BagPFN/bin/python scripts/archive/diagnostics/diagnose_state_upper_bound.py \
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

`scripts/archive/probes_smoke/power_analysis.py` (baseline AUROC 0.55, 모델 간 상관 ρ=0.7 — 실측 Phase 6b vs 6c Pearson ρ=0.737 기반):

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
| `scripts/archive/probes_smoke/power_analysis.py` | 실험 전 검출 가능 효과 크기 확인 |
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
- 평가 프로토콜: `scripts/archive/probes_smoke/power_analysis.py`, `scripts/launch_ici_protocol.sh`, `scripts/evaluate_protocol.py`, `scripts/evaluate_synthetic.py`
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
**신규 스크립트**: `scripts/archive/diagnostics/diagnose_musk_cardinality.py` (체크포인트·학습 불필요)

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
- `scripts/archive/diagnostics/diagnose_musk_cardinality.py` 신규 (cardinality/stratified/decompose/ceiling 4종 리포트,
  `--design-norm {feature,scalar}`) — 위 수치 전부 재현. src/ 변경 없음.
- `scripts/archive/diagnostics/diagnose_normalization_ceiling.py`에 `poolz`/`poolz_l2` 변형 추가.
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
- `scripts/archive/diagnostics/diagnose_musk_cardinality.py` (신규): cardinality / stratified / decompose / ceiling 4종
  리포트, `--design-norm {feature,scalar}`. §26의 모든 Musk 수치를 재현합니다.
- `scripts/archive/diagnostics/diagnose_normalization_ceiling.py`: `poolz` / `poolz_l2` 변형 추가.

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

## 33. 2026-08-05 — v31 CCER-v2 아키텍처 구현 완료 (학습 미시작) (아카이브됨)

CCER-v2 아키텍처 구현·검증 기록. §38에서 CCER 계열 폐기 판정으로 대체. 본문은 [`history/archive.md`](history/archive.md#2026-08-05-v31-ccer-v2-아키텍처-구현-완료-학습-미시작)로 이동했다.

---

## 34. 2026-08-05 — v31 CCER-v2 20-epoch 학습 시작 (아카이브됨)

CCER-v2 20-epoch 학습 시작 기록. §38에서 폐기 판정. 본문은 [`history/archive.md`](history/archive.md#2026-08-05-v31-ccer-v2-20-epoch-학습-시작)로 이동했다.

---

## 35. 2026-08-05 — CCER-v2 구현·검증·20 epoch 학습 완료 (아카이브됨)

CCER-v2 구현·20 epoch 학습 완료 기록. §38에서 폐기 판정. 본문은 [`history/archive.md`](history/archive.md#2026-08-05-ccer-v2-구현검증20-epoch-학습-완료)로 이동했다.

---

## 36. 2026-08-05 — v31 CCER-v2 Epoch 18 합성/Musk 평가 완료 (v30 Baseline 유지) (아카이브됨)

CCER-v2 epoch 18 합성/Musk 평가(v30 미달) 기록. §38에서 폐기 판정. 본문은 [`history/archive.md`](history/archive.md#2026-08-05-v31-ccer-v2-epoch-18-합성musk-평가-완료-v30-baseline-유지)로 이동했다.

---

## 37. 2026-08-05 — CCER-v2 결과 기반 v32 DR-CCER proposal 작성 (아카이브됨)

v32 DR-CCER proposal 작성 기록. §38에서 폐기 판정. 본문은 [`history/archive.md`](history/archive.md#2026-08-05-ccer-v2-결과-기반-v32-dr-ccer-proposal-작성)로 이동했다.

---

## 38. 2026-08-05 — v32b DR-CCER: 비판적 검토 반영 개선안 + 구현 + Stage A 학습 시작

**상태**: v30 baseline 유지. v32 원안을 비판적으로 재검토한 **v32b 개선안**
([`history/architecture_v32b_dr_ccer_proposal.md`](history/architecture_v32b_dr_ccer_proposal.md))을 작성하고,
이를 바탕으로 **P0–P3 probe 스크립트 + DR-CCER 아키텍처 + Stage A 학습**을 구현·실행 중.

### 1. 비판적 검토 요약 (v32 원안 → v32b)

| v32 원안 문제 | v32b 수정 |
|---|---|
| probe가 donor-resolved 전제(§4.1)를 검증하지 못함 | **P3(donor-agreement headroom) 추가** — 기존 checkpoint의 per-donor evidence로 zero-training 검증 |
| n>34 메커니즘 미설명 | P0/P1에 n>34·11–34 band 분해 포함 |
| generator+아키텍처 번들링 (요인 비분리) | B2b를 Stage A에서 분리(Phase 1b로 지연), "v30 on new task mix" baseline을 별도 Phase 1로 |
| 게이트 통계 무효(legacy ±0.01, n>34 단일 수치) | legacy ±0.03, n>34 CI 기반, Stage A에 sparse 전용 + v30 잔차 비상관 조건 |
| mixture logit scale 비대칭 | expert logit을 v30 margin 척도로 표준화(공역혼합) + gate 초기 g≈0.018로 v30 보존 |
| 0.95 목표와 소형 bag 정렬 부재 | 모든 stage에 4-band stratified 보고 의무화 |

### 2. 구현 (Stage-0 probe + DR-CCER 아키텍처)

- **probe**: `scripts/archive/probes_smoke/probe_v32_headroom.py` — P0(분기/백본 분해) + P1(standalone evidence,
  route별 AUROC, v30 오류 조건부 AUROC) + P2(episode-grouped CV logistic fusion headroom) +
  P3(donor-agreement headroom) + n>34/11–34 band 분해. 6-task 혼합(any_positive_sparse 포함)
  1,000-episode 고정 스트림에서 CCER-v2 epoch-18/v30 체크포인트를 paired 평가.
- **모델** (`src/models/baseline.py`, `architecture_version=32`):
  - donor-resolved support bank: per-donor class prototype(전 사영 aligned slot-center),
    top-k donor mean(반복 증거), donor agreement, dispersion, max(upper envelope)의 per-class
    대비 마진.
  - null-contrasted multi-scale scan: absolute(top-1/4/16) + fractional(1%/5%) + dense + 
    bottom-tail + agreement 라우트, donor dispersion으로 표준화, 중복 k 마스킹(floor는 unmasked에만).
  - standalone evidence expert: 2-class logit, zero-init head, CE + 0.1·ranking + 0.05·donor-
    consistency 손실.
  - reliability-gated convex mixture: `final = (1−g)·v30 + g·expert` (expert margin을 v30
    척도로 표준화), gate 초기 g≈0.018 → v30 예측 보존.
- **model_interface** (`src/modules/model_interface.py`): dr_ccer expert 손실/진단,
  `dr_ccer_stage`(A/B) freeze → optimizer는 trainable param만, backbone 0.05x와 독립.
- **config**: `configs/train_v32_dr_ccer.yaml` — v30 best weight-only warm-start,
  6-task mix(`any_positive_sparse` 0.20, legacy 재정규화), `dr_ccer_stage: A`, 10 epochs.
- **test**: `tests/test_dr_ccer.py` 6개 — v30 ranking 보존, dense/list 동치, donor 순열 불변,
  라벨 동변성(마진 부호 반전), 중복 라우트 마스킹, expert 학습·gate 유계.

### 3. Stage A 학습 (완료 2026-08-05, 결과: expert standalone 미학습 → 게이트 실패)

- **Run**: `20260805_182126`, PID `4053492`, config `configs/train_v32_dr_ccer.yaml`
  (v30 best에서 weight-only init, dr_ccer expert만 167K trainable / v30 9.5M frozen).
- **Log**: `logs/20260805_182126/v32_dr_ccer.out` / checkpoint `checkpoints/20260805_182126/v32_dr_ccer/`
  (best `epoch=008-val_ce_loss=0.4307.ckpt`).
- **초기 검증**: v30 warm-start 성공, sanity check 통과, 10 epochs 무결점 완주(오류/NaN 없음).
- **결과 (에폭별 val 메트릭)**:

| epoch | val_ce_loss | val_expert_ce | val_expert_logit_std | val_gate | val_routed_std |
|---|---:|---:|---:|---:|---:|
| 0 | 0.4310 | 0.69315 | 9.43* | 0.01799 | 0.0087 |
| 4 | 0.4316 | 0.69330 | 0.00284 | 0.01799 | 0.0177 |
| 8 | 0.4307 | 0.69311 | 0.00360 | 0.01799 | 0.0084 |
| 9 | 0.4311 | 0.69323 | 0.00606 | 0.01799 | 0.0085 |

  *epoch 0의 logit_std 9.43은 초기 1회 spike(수치 검증), 이후 정상.
- **핵심 해석**: expert의 **standalone CE가 전 에폭 0.693(2-class 무작위)에 정체** → donor-resolved
  expert가 v30 고정 조건에서 **standalone 판별을 학습하지 못함** (expert logit std 0.006 수준).
  gate는 Stage A에서 의도대로 초기값(0.018)에 고정. 따라서 **v32b Stage A 게이트(standalone expert
  AUROC ≥ 0.70, sparse +0.03) 실패** — donor-resolved evidence 경로는 이 분포에서 보완 판별 신호를
  만들지 못함(CCER-v2의 0.999 상관 결과와 정합). Stage B(router) 진행 근거 없음.

### 4. Stage-0 probe (P0–P3) — **모든 게이트 실패: CCER 계열 실증적 폐기**

- `scripts/archive/probes_smoke/probe_v32_headroom.py`, 100-episode 6-task 혼합, CCER-v2 epoch-18 vs v30 paired
  (공유 서버 고부하로 1,000→500→100 ep로 축소; delta 부호/0은 ep 수에 무관하게 결정적).
  결과: `logs/probe_v32_headroom_20260805.csv` + `/tmp/probe_v32_100.log`.

| Probe | 결과 | 게이트(≥ +0.005) |
|---|---|---|
| P0 분기/백본 분해 | CCER-v2 full 0.87679 ≈ branch-zeroed 0.87681 ≈ v30 0.87654 (branch 기여 ~0, n>34 0.9277 동일) | 분기 무기여 |
| P1 standalone | **branch standalone AUROC 0.5106 (무작위)**, corr(v30)=0.0096(무상관), effective contribution SD 0.021, route별 0.51 전부 | branch = 잡음 |
| P2 fusion headroom | combiner delta **-0.00034** | **FAIL** |
| P3 donor-agreement headroom | donor 피처 combiner delta **+0.00000** (개별 8피처 전부 +0.00000) | **FAIL** |

- **해석**: CCER-v2의 standalone branch는 무작위(0.51)에 v30과 무상관(corr 0.0096)이며, full 예측이 v30과
  0.999 상관이던 것은 branch가 v30에 **소량 잡음 섭동(0.021 SD)**을 더했기 때문이다(§37의 "작은 보정"
  해석보다 더 강한 반증). donor-resolved pooling(donor agreement/상위사분위/중앙값)도 v30 margin에
  **0.00000** 추가 — v32 §4.1 전제가 0회 재학습 probe로 직접 반증됨.
- **결론**: v32b §2 Stage-0 게이트 FAIL → **CCER 계열(현 CCER-v2 표현 + donor-resolved 변형) 폐기**,
  Stage B(router)/C(seed)/D(Musk) 진행 근거 없음. 데이터 측 경로(task mix / 소형 bag 노출 / Phase 1
  v30-on-6-task-mix)로 전환.

### 5. 종합 판정 (2026-08-05)

- **v30 baseline 유지** (변경 없음). CCER-v2·DR-CCER 모두 미채택, 재현 코드만 보존.
- **전체 unittest 178개 통과** (1534.9s) — DR-CCER 6개 포함 회귀 없음.
- **Stage A 학습**(`20260805_182126`) 완주: donor-resolved expert standalone CE 0.693(무작위) 정체
  → Stage A 게이트 실패. **Stage-0 probe(P0–P3)**도 전 게이트 실패(P2 -0.00034, P3 +0.00000).
- **다음 Action**: ① CCER 계열 폐기 기록(`history/archive.md`), ② Phase 1 "v30 on 6-task mix"
  재학습으로 데이터 효과 측정(any_positive_sparse가 v30에 무엇을 더하는지), ③ 소형 bag(n≤4, 0.80)과
  n>34(0.70)가 0.95 목표의 실질 병목 — 데이터/분포 쪽 레버 우선, ④ ICI 잠금 유지.

---

## 39. 2026-08-05 — v32b 완료 결과 평가 + v33 MR-BagPFN proposal

**상태**: 결과 평가와 proposal 작성만 완료. v33 구현·학습은 시작하지 않았고 v30 baseline은 유지한다.

- 현행 proposal: [`history/architecture_v33_multiresolution_bag_proposal.md`](history/architecture_v33_multiresolution_bag_proposal.md)
- v32/v32b proposal은 구현·평가 종료에 따라 `history/`로 이관했다.
- **평가**: P1 standalone `0.51055`, P2 fusion `-0.00034`, P3 donor fusion `+0.00000`,
  Stage-A expert CE `0.6931` 정체가 같은 결론을 지지한다. CCER/DR-CCER 표현에는 v30의 오류를
  교정할 ranking information이 없다. 100-episode probe의 검정력 한계는 존재하지만, 독립 Stage-A
  학습까지 무작위로 붕괴했으므로 추가 CCER 학습의 근거가 되지 않는다.
- **폐기 범위**: class-conditioned cell↔support similarity, donor 통계, slot/Top-K/residual 확대.
  기존 v30 rare/population branch, six-task 학습, B2b, multi-resolution v30 view는 아직 반증되지 않았다.
- **v33 방향**: (A/B/C/D) v30 task-mix×B2/B2b 요인 분리 → frozen-v30 multi-resolution probe
  → paired AUROC `+0.01` headroom 확인 시에만 zero-init consensus residual 구현.
- **바로 다음 단계**: Phase 0 arm B(v30 + six-task + B2)와 arm C(v30 + legacy + B2b).
  새 architecture부터 구현하지 않는다. ICI 잠금 유지.

---

## 40. 2026-08-05 — 기본 unittest suite compact화

**상태**: 완료. 기본 discovery는 19개 핵심 계약만 실행하고, 나머지는 history에서 보존한다.

- 이전 기본 suite: unittest 178개 + discovery에서 누락된 pytest-style ICI 1개, 최근 실측
  `1534.9s`.
- 새 기본 suite: `tests/test_core_contracts.py`, `test_scheduler.py`,
  `test_checkpoint_callback.py`의 **19개**, CPU 실측 **`142.820s`**, 전부 통과.
- 유지 계약: v30 forward/backward, query-label isolation, label equivariance,
  cell/context permutation invariance, ragged bags, poolz query isolation, n=1, batched/list,
  ranking/checkpoint marker, synthetic split/cardinality/sparse task, AUROC/bootstrap/log loss,
  ICI all-cell mean, scheduler, last-checkpoint 저장.
- 폐기 architecture와 연구용 상세 회귀 **175개 method / 12개 파일**은
  `tests/history/legacy_*.py`로 이동했고 파일명 패턴상 기본 `test_*.py`에서 실행되지 않는다.
- archive 정책: [`../tests/history/README.md`](../tests/history/README.md). 보존 코드 경로를
  직접 수정할 때만 해당 legacy module을 명시 실행한다.
- 표준 명령: `timeout 300s /home/aibio_3/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"`.


---

## 41. 2026-08-05 — v33 Phase 0 구현: arm B(C) 데이터 컨트롤 + 학습 런치

**상태**: v33 proposal §9에 따라 **arm B(v30 + six-task + B2)와 arm C(v30 +
legacy + B2b)** 구현·런칭 완료. 새 architecture(residual consensus)는 Phase 1
probe 통과 전까지 구현하지 않는다. ICI 잠금 유지.

- **B2b (per-bag cardinality) 구현**:
  - `SyntheticManifoldGenerator(per_bag_cardinality=True)` — 에피소드 내에서 각
    bag이 `n_b ~ LogUniform[1,1024]`을 독립 추첨. dense generation은
    `max(n_b)`에서 수행 후 per-bag subset으로 subsample(ragged list 반환).
  - dense cell은 bag 내 교환 가능하므로 subsample은 i.i.d. `n_i` 추첨과
    분포상 동일. sparse task(arm D 예비)는 각 양성 bag의 유지 subset 내에서
    m개의 shifted cell을 latent 공간에 marking해 subsample 후에도 보존.
  - oracle abundance/population features는 subsample 후 per-bag 재계산.
  - `SyntheticEpisode.x` 타입을 `Tensor | list[Tensor]`로 확장.
  - collator: ragged 배치 >1 거부(에피소드 간 stack 불가), eval collator는
    `len(x)`로 bag 수 처리. `training_step`에 ragged(Sequence) 단일 에피소드
    분기 추가. `episode_batch_size=1` 필수.
- **config**: `configs/train_v33_phase0_armB.yaml`(six-task
  `[0.32,0.24,0.04,0.04,0.16,0.20]`), `configs/train_v33_phase0_armC.yaml`
  (`per_bag_cardinality: true`, batch=1, shape_group_size=1).
- **테스트**: `tests/test_b2b.py` 신규 10개 추가. 전체 기본 suite
  **29 tests / 185.785s 전부 통과**.
- **GPU 스모크**: arm B/C 각 1 epoch + 16 train batch 정상(exit=0).
- **학습 런치** (detached, GPU 0):
  - arm B: `logs/20260805_214745/v33_phase0_armB.out`, ckpt
    `checkpoints/20260805_214745/v33_phase0_armB/` (50 ep, batch 8, ~2.9min/ep)
  - arm C: `logs/20260805_214751/v33_phase0_armC.out`, ckpt
    `checkpoints/20260805_214751/v33_phase0_armC/` (50 ep, batch 1, ~24min/ep)
- **arm C 비대칭 주의**: batch 1이라 epoch당 optimizer step이 v30 대비 8x.
  회귀 gate에 대해 보수적(더 많이 학습해도 회귀 시 = 강한 부정 신호).
- **운영 결정 — 최초 arm C 중단 후 step-matched로 재설계** (사용자 확정):
  - arm C는 `4096 steps/epoch × 50 = 204,800` optimizer step으로, arm B의
    `512 × 50 = 25,600`보다 8배 많아 예상 약 20시간이 걸렸다. 이는 의도한 공정한
    데이터 컨트롤이 아니므로 launcher PID `4183921`과 torchrun PID `4183927`에
    `SIGTERM`을 보내 중단했다. 생성된 checkpoint는 없다.
  - arm B는 원래 v30 architecture를 유지하고 `precision: bf16-mixed`를 명시했다.
    `logs/20260805_220642/v33_phase0_armB_bf16.out`, launcher PID `37183`; 기존
    `checkpoints/20260805_214745/v33_phase0_armB/last.ckpt`에서 복원해 진행 중이다.
  - arm C는 architecture/ragged B2b를 바꾸지 않고 train episode만 512/epoch로 조정했다.
    따라서 `512 episodes / batch 1 = 512 updates/epoch`로 arm B의
    `4096 / batch 8 = 512 updates/epoch`와 정확히 같다. config는
    `configs/train_v33_phase0_armC.yaml`이다.
  - precision 정렬 재시작 (2026-08-05): 기존 arm C가 FP32(`32-true`)인데 반해
    v30 baseline과 arm B가 `bf16-mixed`여서 요인 분리 계약에 어긋났다. FP32로 2
    epoch 진행한 `220843` run은 중단하고, `precision: bf16-mixed`로 정렬해
    **`logs/20260805_221615/v33_phase0_armC_bf16.out`**(launcher PID `61322`,
    ckpt `checkpoints/20260805_221615/v33_phase0_armC_bf16/`)에서 재시작했다.
    ragged B2b는 스모크에서 이미 bf16-mixed로 검증했다. (commit `4a39ab9`)
- **8× 에피소드 수 비대칭 (해석 주의)**: step은 arm B와 일치(25,600)하지만
  **총 에피소드 수는 arm C 25,600 vs v30/arm B 204,800으로 8× 차이**가 남는다.
  v30 val_ce 곡선을 보면 ep 25~49에서도 `0.4533→0.4442`로 완만하게 개선 중(best가
  마지막 epoch 48)이라 v30 자체도 에피소드 수가 아직 경계(binding)였고, arm C는
  v30의 약 epoch 6~7 수준(val_ce ~0.457)에 머물 것으로 추정된다. 따라서 arm C 회귀
  gate가 B2b 효과가 아니라 **과소학습 편향으로 오염될 수 있다**. 판정 시 arm C의
  val_ce가 epoch 50에도 내려가는 중이면(수렴 전) **top-up(추가 학습)으로 수렴점까지
  이어간 뒤 비교**한다. 엄밀한 대안은 에피소드-매치(4096/epoch, ~20h)지만 비용이 크다.
- **Phase 0 gate**: sparse task AUROC ≥ 0.75 (arm B), legacy overall 회귀
  ≤ 0.01 (paired CI가 0 포함), B2b가 full-vs-subsample margin drift ≥20% 감소
  (probe로 측정).
- **바로 다음**: Phase 0 결과 선택 → frozen-v30 multi-resolution probe
  (Phase 1) → paired AUROC `+0.01` 통과 시에만 v33 residual 구현.

## 42. 2026-08-06 — v33 Phase 0 arm B/C 학습 완료 + gate 평가

**상태**: arm B, arm C 모두 50 epochs 학습 완료. 1,000-episode 합성 평가로 Phase 0
gate 판정을 내렸다. **arm B는 sparse task gate 미달, arm C는 legacy overall 회귀
gate 미달** → v33 Phase 0의 두 주 효과(B: six-task+sparse, C: legacy+B2b) 모두
gate 통과 실패.

- **학습 완료**:
  - arm B(v30+six-task+B2): `checkpoints/20260805_220642/v33_phase0_armB_bf16/`,
    best `epoch=044 (val_ce_loss 0.4290)`. 50 ep / 204,800 ep / 25,600 steps.
  - arm C(v30+legacy+B2b): `checkpoints/20260805_221615/v33_phase0_armC_bf16/`,
    best `epoch=049 (val_ce_loss 0.5351)`. 50 ep / 25,600 ep / 25,600 steps.
    **best가 마지막 epoch** → 수렴 전 경계 상태(§41 과소학습 예측과 일치).
- **평가** (1,000 ep, seed 42, 고정 val 스트림 = v30 legacy B2 분포):
  - arm B six-task: overall AUROC **0.8461 [0.834, 0.857]**, log loss 0.4748.
    per-task: composition 0.8745 / state 0.8214 / covariance 0.7414 /
    interaction 0.8465 / combined 0.9515 / **any_positive_sparse 0.6747**.
  - arm B legacy: **0.8500 [0.839, 0.861]** vs v30 committed 0.8512 [0.840, 0.862]
    → 회귀 **-0.0012**, paired `P(arm B beats v30)=0.04`(에피소드·태스크 매칭 확인).
  - arm C legacy: **0.8139 [0.802, 0.825]** vs v30 committed 0.8512 [0.840, 0.862]
    → 회귀 **+0.0373**, paired `P(arm C beats v30)=0.00` (5000 bootstrap) — CI가
    겹치지 않고 v30이 사실상 100% 우세. 회귀는 통계적으로 확정적.
- **Gate 판정**:

  | Gate | arm B | arm C |
  |---|---|---|
  | sparse task AUROC ≥ 0.75 | ❌ **0.6747** | — |
  | legacy overall 회귀 ≤ 0.01 | ✅ -0.0012 | ❌ **+0.0373** |

- **해석**:
  - arm B: `any_positive_sparse` 태스크가 유용성 기준(0.75)에 미달. six-task 믹스의
    sparse 추가는 Phase 0에서 기각. legacy 성능은 v30 대비 소폭 열세(-0.0012)로 안전.
  - arm C: B2b 학습이 v30 legacy B2 val 스트림에서 0.037 회귀 → gate 실패.
    §41에서 예고한 과소학습 편향(8× 적은 에피소드, best=epoch 49, val_ce 0.5362→0.5351
    완만 하락)과 일치. protocol대로 **top-up(추가 학습)으로 수렴점까지 이어간 뒤
    재판정**이 필요할 수 있다.
  - B2b full-vs-subsample margin drift probe: 아직 미실시
    (`probe_v32_headroom.py`에는 drift 측정이 없음).
- **예측 파일**: `predictions/synthetic_v33_phase0_armB_6task_1000ep.pt`,
  `predictions/synthetic_v33_phase0_armB_legacy_1000ep.pt`,
  `predictions/synthetic_v33_phase0_armC_legacy_1000ep.pt`.
- **바로 다음**: ① arm C top-up 여부(사용자 결정), ② (선택) top-up 후 arm C 재평가,
  ③ Phase 0 결과 선택 → frozen-v30 multi-resolution probe(Phase 1).

## 43. 2026-08-06 — arm C top-up: 8×A6000 DDP 전환 + NCCL P2P hang 수정 + B200 vs A6000 속도 기록

**상태**: 사용자 결정으로 arm C top-up을 **8×RTX A6000 (48 GiB) DDP**로 재개했다.
에피소드-매치(`episodes_per_epoch: 4096`, v30과 동일 총 에피소드 예산)로 전환해
§41/§42의 과소학습 편향을 제거한다. 진행 중: 2026-08-06 12:54 시작, epoch 49 ckpt
에서 resume, 현재 ~epoch 53 (총 목표 150).

- **새 config**: `configs/train_v33_phase0_armC_ddp8.yaml` — medium 체인을 상속하지
  않는 자체 포함형. B2b ragged(`per_bag_cardinality: true`, `episode_batch_size: 1`),
  `episodes_per_epoch: 4096`, `trainer: devices 8 / ddp_find_unused_parameters_false /
  bf16-mixed / max_epochs 150`. resume: `archive/v33_phase0_armC_bf16/last.ckpt`.
- **NCCL P2P hang (gnode5) — 원인 진단 및 수정**:
  - 증상: 8-GPU torchrun이 `All distributed processes registered` 직후 영원히 hang.
    GPU 8장 모두 100% util인데 메모리 ~450 MiB 고정, rank CPU가 1코어씩 회전, 로그·
    metrics 무진행.
  - 진단: `scripts/archive/probes_smoke/nccl_probe.py`(신규, broadcast/all_reduce/대형 broadcast 최소 재현)로
    **NCCL comm init은 8 rank 모두 정상 완료**됨을 확인. 그러나 **첫 컬렉티브
    (`dist.barrier()`)에서 hang** → 통신 그룹 생성이 아니라 **전송(transport) 문제**.
    `NCCL_DEBUG=INFO`에서 `Channel ... via P2P/CUMEM` 채널 사용 확인.
  - 검증: `NCCL_P2P_DISABLE=1`만 8 rank 프로브가 통과. `NCCL_CUMEM_DISABLE=1`/
    `NCCL_P2P_LEVEL=SYS`는 여전히 hang → 이 머신의 NCCL P2P/CUMEM 전송이 불안정.
  - 수정: `scripts/launch_interactive_training.sh`에 `NCCL_P2P_DISABLE=1` 기본 적용 +
    detached 워커에 env 전달 추가. 단일 노드 8×A6000(NVLink 없음)이므로 SHM 전송으로
    동작.
- **B200 1장 vs A6000 8장 속도 비교** (동일 v30 arch · `episode_batch_size=1` ·
  bf16-mixed · bag 60–100, `n_b ~ LogUniform[1,1024]`, 전형 9k–15k 셀 / worst 102k):

  | 항목 | B200 1장 (기준, step-matched 512 ep/epoch) | A6000 8장 (현재, 4096 ep/epoch) |
  |---|---|---|
  | step당 시간 | **0.36 s/step** | **~0.66 s/step** (rank당) |
  | it/s | 2.5–2.8 it/s | **~1.5 it/s** (rank당) |
  | 에피소드 처리량 | 2.8 ep/s | **~1.5 ep/s** (rank당) |
  | epoch 시간 | ≈ 3:05 / epoch (512 steps) | **≈ 5:38 / epoch** (512 steps/rank) |
  | 50 epochs 총량 | ≈ 2.6 h (25,600 ep) | **≈ 4.7 h** (4096 ep/epoch 기준) |
  | 노드 총 처리량 | 2.8 ep/s | **≈ 12 ep/s** (8×1.5, ~4.3×) |

  - **해석**: A6000 1장 기준으로는 B200 대비 ~1.8× 느리다(step당 0.36→0.66 s). 원인은
    (1) A6000(48 GiB) vs B200(180 GiB) 연산·메모리 대역폭 차이, (2) NCCL P2P 비활성화로
    인한 all-reduce 오버헤드 증가, (3) GPU util이 23–100%로 불균일해 DDP 동기화 대기.
    그러나 **8장 병렬로 노드 총 처리량은 ~4.3×** (2.8→12 ep/s). 에피소드-매치(204,800 ep)
    예산 기준으로는 B200 2.6 h 대비 A6000 8장 약 4.7 h (P2P 비활성화 비용 포함).
- **검증 완료 (epoch 50–53)**: resume 성공(`Restored all states`), 첫 step VRAM
  peak 0.92 GiB(1.8%), epoch당 VRAM 3.8–7.8 GiB(A6000 48 GiB의 ~8–16%), 체크포인트
  `epoch=NNN` 자동 저장 확인. val_ce: 0.5381(50) → 0.5375(51) → 0.5377(52) → 0.5370(53).
- **바로 다음**: ① arm C top-up 완주(150 epoch) 후 §42 재평가(legacy overall 회귀 gate),
  ② 통과 시 frozen-v30 multi-resolution probe(Phase 1).

---

## 44. 2026-08-06 — 패딩 배칭 (B2b `episode_batch_size>1`) + 병목 프로파일

**상태**: arm C top-up의 핵심 병목(rank당 batch=1, VRAM ~5%, GPU util 불균일)을
프로파일로 확정하고 **ragged B2b 에피소드의 패딩 배칭**을 구현·검증했다.
commit `568c5f8`.

- **프로파일 (B200, arm C ddp8 config)**: step의 ~92%가 모델 forward+backward
  (~177 ms / ~190 ms). 원인은 집계기의 **bag별 Python 루프**(소형 커널 다수)이며,
  arm B(dense, batch 8)가 8 에피소드를 ragged 1개와 비슷한 벽시계로 처리하는 것으로
  확인 — 벡터화 dense 경로는 배칭이 핵심.
- **구현**: collator가 ragged 배치를 `(x, y, cell_mask, bag_mask)`로 패딩.
  집계기 전역에 cell mask + per-bag valid count 관통 (슬롯 할당은 softmax 후 0처리로
  all-`-inf` NaN 회피, tail/rare는 per-bag count + keep 마스킹, covariance 정규화,
  context-pool/앵커/CLS 폴링). `forward_episode_batch`는 집계기는 완전 배칭, bag-
  토큰 수준 meta-classifier만 에피소드별 루프(저비용). query rare-evidence에도
  per-query cell mask.
- **잠재 train/eval 불일치 수정**: `_forward_dense`의 `tail_fractions`가 softmax
  가중합을 쓰던 것을 list 경로(평가/Musk 기준)와 동일한 **산술 평균**으로 정렬.
  dense/패딩/평가가 이제 일치.
- **B200 실측 (동일 v30 arch, ddp8 데이터)**:

  | episode_batch_size | 처리량 | peak VRAM | 비고 |
  |---|---|---|---|
  | 1 (기존) | 5.8 ep/s | 1.1 GiB | rank당 1 에피소드 |
  | 2 | **~16 ep/s** | ~16 GiB | **A6000 48 GiB 안전** |
  | 4 | ~12 ep/s | ~32 GiB | A6000 경계 |
  | 8 | ~14 ep/s | ~68 GiB | A6000 OOM, B200 전용 |

  실 wall-clock은 배치로 Lightning 오버헤드가 분산되어 추가 이득.
- **config**: `configs/train_v33_phase0_armC_ddp8_batch2.yaml`
  (`episode_batch_size: 2`, episode-match 4096 ep/epoch → 256 steps/epoch).
- **검증**: `tests/test_ragged_batching.py` 3개 (패딩 collator + 패딩 배치 == 개별
  list 경로 logits, 1e-4). `test_b2b.py`의 "ragged 거부" 계약을 "패딩" 계약으로 갱신.
  **전체 38 tests 통과 (~256s)**.
- **바로 다음**: ① (선택) A6000 top-up을 batch2 config로 재런칭/적용 — 기존 batch=1
  런과의 비교 판단은 사용자 결정, ② §42 재평가, ③ Phase 1 probe.

## 45. 2026-08-06 — arm C top-up 중간 Musk zero-shot: 대형 bag(n>34) 개선 + 소형 trade-off

**상태**: arm C top-up 진행 중(epoch ~84/150)에 **중간 checkpoint(epoch 64,
`best_epoch64_valce0.5287.ckpt`)의 Musk zero-shot을 측정**해 v30 확정 baseline과
비교했다. 목적은 top-up 완주 전에 B2b 추가 학습이 Musk 방향을 어떻게 움직이는지
조기 신호를 잡는 것. **대형 bag(n>34)이 0.698→0.825로 크게 개선**됐고, 대신
**소형(n≤4)이 0.792→0.700으로 희생**됐다.

### 실행 환경 (gnode4, 8×A5000 — gnode5와 파일서버 공유)

- 이 세션은 NHN(B200)/gnode5(A6000)가 아닌 **gnode4**에서 진행. arm C top-up 자체는
  gnode5에서 돌며 NFS로 체크포인트/metrics가 gnode4에 실시간 동기된다.
- **v30 checkpoint는 워크스페이스 `checkpoints/`에 없고 `/home/kimds/archive/`에 있다**
  (`/data-hdd`는 백업 서버). 워크스페이스 root도 `/NHNHOME/BASE/kimds/ICF`가 아닌
  `/home/kimds/ICF` — 다중 위치 동기화 환경이라 경로 확인 필요.
- Musk 데이터: `/home/kimds/BagPFN/Data/Musk/musk.pkl` (NHN 경로 아님).
- 사용된 checkpoint:
  - v30 baseline: `archive/v30_cardinality_poolz_l2/epoch=048-val_ce_loss=0.4442.ckpt`
  - v33 arm C: `archive/v33_phase0_armC_ddp8_topup_20260806/best_epoch64_valce0.5287.ckpt`
- config: v30 `train_v30_cardinality_poolz_l2.yaml`, v33 `train_v33_phase0_armC_ddp8.yaml`
  (arm C도 `bag_representation: poolz_l2` — v30 arch 그대로. "legacy"는 데이터 믹스
  의미일 뿐 표현 아님). preprocess는 기본 `bag_view` (v30 S2 측정과 동일).

### 결과 (102 bags leave-one-out, seed 42)

| 지표 | v30 baseline | v33 arm C (ep64) |
|---|---|---|
| **AUROC [95% CI]** | **0.8539 [0.774, 0.925]** | **0.8799 [0.810, 0.946]** |
| Accuracy | 0.794 | 0.814 |
| Balanced acc | 0.785 (sens 0.744 / spec 0.825) | 0.796 (sens 0.718 / spec 0.873) |
| Log loss | 0.476 | 0.441 |
| corr(prob, log n) | +0.057 | −0.146 |

v30 baseline은 문서값과 **정확히 재현**(0.8539) — 체크포인트/파이프라인 무결성 확인.

### 밴드별 AUROC (stratified, 같은 102 bag)

| 밴드 | v30 baseline | v33 arm C | Δ |
|---|---|---|---|
| ALL | 0.854 | 0.880 | **+0.026** |
| **n≤4** | 0.792 | 0.700 | **−0.092** |
| 5..10 | 0.833 | 0.925 | +0.092 |
| 11..34 | 0.964 | 0.970 | +0.006 |
| **n>34 (대형)** | **0.698** | **0.825** | **+0.127** |

### paired bootstrap (4,000 resample)

| stratum | bags | v30 | v33 | Δ | 95% CI | P(v33>v30) |
|---|---|---|---|---|---|---|
| ALL | 102 | 0.854 | 0.880 | +0.026 | [−0.021, +0.078] | 0.858 |
| n≤4 | 29 | 0.792 | 0.700 | −0.092 | [−0.258, +0.033] | 0.084 |
| n>4 | 73 | 0.864 | 0.913 | +0.049 | [−0.014, +0.118] | 0.932 |

### 판독 (사용자 관점 포함)

- **전체 +0.026은 통계적으로 무의미**(CI 0 포함, P=0.858). 그러나 **구간 구조는 명확**:
  v33이 **소형(n≤4)을 팔아 중·대형 전 구간(5..10, n>34)을 샀다**.
- 원래 v30의 고질적 약점이던 **n>34가 0.698→0.825 (+0.127)** — 이것이 이 신호의 핵심.
  5..10도 +0.092. 대형·중형 양쪽에서 개선.
- **사용자 판단**: "소형 bag 희생할 만하다"는 쪽으로 기우는 중. (v30이 B2로 처음 고쳤던
  n≤4를 되돌리는 trade-off이므로, gate 관점에서 주의해서 볼 필요는 있음.)
- **한계**: ① **중간 checkpoint(epoch 64/150)** 기준 — 완주 후 재확인 필요. ② n>34는
  bag 수가 적어(약 13개) CI가 넓음(v33 0.825 [0.60, 0.98]). ③ synthetic legacy 회귀
  gate(§42, ≤0.01)는 별개로 아직 미검증 — top-up 완주 후 평가해야 함.

### 예측 산출물

- `predictions/musk_v30_baseline_best.pt`
- `predictions/musk_v33_armC_current_best.pt`

### 바로 다음

1. arm C top-up 완주(150 epoch) 후 §42 재평가(legacy 회귀 gate) + **Musk 재확인**(§45 신호가
   완주 후에도 유지되는지).
2. (논의) §45의 "대형 bag 개선 / 소형 희생"이 실질 개선이라면 Phase 0 결과 선택 기준 재검토.
3. frozen-v30 multi-resolution probe(Phase 1)는 Phase 0 결과 확정 후에만.

## 46. 2026-08-06 — PathoBench zero-shot 평가: per-task PCA 전처리 + 결과

**상태**: 실행 중인 arm C checkpoint(`v33_phase0_armC_ddp8_batch2` epoch 88)의
실세계 전체슬라이드 MIL(PathoBench) zero-shot 평가를 완료했다. 평가를 위해 **task별
8:2 분할 + train-only PCA(1536→512) 전처리 캐시** 파이프라인을 구축했다
(`scripts/prepare_pathobench.py`).

- **전처리 프로토콜 (사용자 확정, 2026-08-06 갱신)**: 각 task CSV의 train/test 분할을
  그대로 사용해 8:2로 나눈 뒤, **train 분할의 모든 타일에 PCA(1536→512)를 fit**하고
  train/test 모두 변환해 `data/pathobench/{task}_train.pt` / `{task}_test.pt`로 저장.
  - **타일 서브샘플링 없음** (기존 1024장/10만 샘플 제한 폐기). PCA는 두 패스
    (mean → centered covariance, float64 청크 누적)로 **전체 train 타일을 정확히** 사용.
  - **추론도 전체 타일 사용** (컨텍스트·query 모두 서브샘플 없음, `--max-tiles`/
    `--target-context-cells`/`--max-queries` 제거).
  - **bootstrap CI 폐기** — task가 많으므로 단일 테스트 결과만 출력 (CI 없음).
  - 평가는 `--data-dir` 기본 `data/pathobench`에서 캐시 우선 로드, 미존재 시 h5+PCA
    fallback. 캐시 형식: `{"slide_id": list, "bag": list[Tensor[n,512]], "label": list[int]}`.
  - **slide_id 문자열 캐스팅 추가** (BC_Therapy/CPTAC-CCRCC는 숫자 id라 pandas가
    int64로 읽어 h5 인덱스와 불일치 → 전부 누락 버그 수정).
  - **이진 task만 대상**: multi-class(BRACS 등) 제외. HerROI(`herroi_response`)는
    `features/HER2_tumor_ROIs_v3`가 빈 디렉토리라 피처 부재로 제외. **총 17개 이진 task**.
- **모델**: `epoch=088-val_ce_loss=0.5282.ckpt` (arch v24 내부, v30 `poolz_l2` +
  B2 log-uniform cardinality, 2026-08-06 16:03 저장, 아직 학습 진행 중).
- **결과 (zero-shot, sample-context 6/class, 전체 타일, seed 42, 단일 테스트)**:

  | task | test n | AUROC | Acc | BAcc |
  |---|---|---|---|---|
  | bc_therapy_er | 33 | 0.517 | 0.606 | 0.520 |
  | bc_therapy_grade | 33 | 0.538 | 0.545 | 0.531 |
  | bc_therapy_her2 | 33 | 0.542 | 0.455 | 0.510 |
  | cptac_brca_pik3ca | 21 | 0.582 | 0.333 | 0.429 |
  | cptac_brca_tp53 | 22 | 0.420 | 0.318 | 0.357 |
  | cptac_ccrcc_er | 33 | 0.517 | 0.606 | 0.520 |
  | cptac_ccrcc_grade | 33 | 0.538 | 0.545 | 0.531 |
  | cptac_ccrcc_her2 | 33 | 0.542 | 0.455 | 0.510 |
  | cptac_lscc_arid1a | 67 | 0.631 | 0.388 | 0.469 |
  | cptac_lscc_histologic | 57 | 0.597 | 0.596 | 0.610 |
  | cptac_lscc_keap1 | 51 | 0.590 | 0.510 | 0.484 |
  | cptac_luad_egfr | 59 | 0.637 | 0.458 | 0.526 |
  | cptac_luad_kras | 62 | 0.655 | 0.548 | 0.601 |
  | cptac_luad_stk11 | 67 | 0.682 | 0.522 | 0.621 |
  | cptac_luad_tp53 | 59 | 0.612 | 0.610 | 0.612 |
  | cptac_pda_smad4 | 55 | 0.309 | 0.509 | 0.438 |
  | ucla_lung_progression_regression | 22 | 0.598 | 0.682 | 0.645 |

- **해석**: sample-context(6 slide/class, 전체 타일)는 대부분 0.5~0.68의 랜덤~약상승
  수준. LUAD 계열(egfr/kras/stk11/tp53 0.61~0.68)과 LSCC(arid1a 0.631)가 상대적으로
  양호, PDA smad4는 랜덤 이하(0.309). **BC_Therapy와 CPTAC-CCRCC는 동일 슬라이드·동일
  라벨의 중복 데이터**(AUROC/Acc/logloss 완전 동일)로 확인. ⚠️ **§51(2026-08-07) 정정:
  이는 벤치마크 속성이 아니라 로컬 데이터 오류** — 로컬 `cptac_ccrcc_{er,grade,her2,residual}.csv`가
  `bc_therapy`의 바이트 단위 복사본이고, 공식 CPTAC-CCRCC 코호트(`C3L/C3N`)는 전혀 미포함이다.
  이전의 all-context(전체 train 슬라이드)가 sample보다 강했던 점(0.70~0.73)을 고려해, 전체 타일 기준
  all-context 재평가는 후속으로 가능.
- **파일**: `scripts/prepare_pathobench.py`, `scripts/test_pathobench.py`(갱신),
  `data/pathobench/{task}_{train,test}.pt` (17 task × 2), `predictions/pathobench_{task}_..._e88_full.pt`.
- **재실행**: `python scripts/test_pathobench.py --checkpoint <ckpt> --csv
  /NHNHOME/BASE/kimds/Data/PathoBench/csv/<task>.csv`. 전처리는
  `python scripts/prepare_pathobench.py --csv ...` 1회.

### all-context (전체 타일) — 5개 task 확장 (2026-08-06)

**상태**: sample-context가 대부분 랜덤~약상승이어서, 강세(LUAD)·약세(BRCA, PDA)·원래
벤치마크를 고르게 대표하는 **5개 task**를 `--context-mode all`(모든 train 슬라이드,
전체 타일)로 재평가.

- **OOM 수정**: 전체 타일 all-context를 패딩 dense 경로(`forward_episode_batch`)로
  돌리면 `[bags, max_cells, slots, dim]` 차이 텐서가 최대 bag 크기에 맞춰 폭발(69GB)
  → **ragged per-episode 경로**(`model.forward(x_list, y, mask_index)`, bag별 개별
  처리)로 전환. 메모리는 bag당으로 안전, 결과는 패딩 경로와 동일(1e-4 검증됨).
- **결과 (전체 타일, epoch 88, 단일 테스트)**:

  | task | test n | sample AUROC | **all AUROC** | Acc | BAcc |
  |---|---|---|---|---|---|
  | cptac_brca_tp53 | 22 | 0.420 | **0.696** | 0.545 | 0.616 |
  | cptac_luad_tp53 | 59 | 0.612 | **0.625** | 0.576 | 0.557 |
  | cptac_luad_stk11 | 67 | 0.682 | **0.786** | 0.776 | 0.785 |
  | cptac_lscc_arid1a | 67 | 0.631 | **0.748** | 0.821 | 0.648 |
  | cptac_pda_smad4 | 55 | 0.309 | **0.679** | 0.746 | 0.624 |

- **해석**: all-context가 전 task에서 sample 대비 개선. 특히 랜덤 이하였던
  **PDA smad4(0.309→0.679)**, **BRCA(0.420→0.696)**가 큰 반전. LUAD stk11 0.786,
  LSCC arid1a 0.748로 강세. LUAD tp53만 소폭(0.625). **컨텍스트 규모(전체 train
  슬라이드, 전체 타일)가 성능의 핵심 요인**임을 다시 확인.
- **파일**: `predictions/pathobench_{task}_armC_batch2_e88_allctx_full.pt` 5개,
  로그 `predictions/allctx_full_5.log`. BRACS coarse 데이터(캐시 2개 + 예측 2개)는
  multi-class 제외에 따라 삭제 (원본 `features/BRACS/`는 보존).

## 47. 2026-08-06 — 새 기준 checkpoint(e125) 재평가 + 타일 수 제한 실험

**상태**: **앞으로 모든 PathoBench 실험은 all-context 기준**(sample-context 폐기).
150 epoch 런의 best인 **`epoch=125-val_ce_loss=0.5142.ckpt`를 새 기준 checkpoint로
채택**하고, (1) e125로 5개 pathology all-context 재평가(val_ce 개선이 실제 test로
이어지는지), (2) bag별 타일 수 제한 실험을 실행 중.

- **스크립트 변경** (`scripts/test_pathobench.py`):
  - `--context-mode` 기본값 `all`로 변경 (sample은 deprecated, `--context-per-class`는
    sample 전용으로 유지).
  - **`--max-tiles`** 추가: bag(컨텍스트·query 모두)별 타일 상한. 지정 시 각 bag을
    trial별 랜덤 서브샘플.
  - **`--trials`** 추가: seed base + trial로 독립 추론 반복, trial별 지표 + 집계
    (mean/min/max) 출력. `evaluate_trial()`로 루프 추출.
  - 평가는 ragged per-episode 경로(`model.forward(x_list, y, mask_index)`) — bag별
    처리라 전체 타일 all-context도 메모리 안전 (§46).
- **Task 1 (진행 중)**: e125로 5개 task(`cptac_luad_tp53`, `cptac_luad_stk11`,
  `cptac_lscc_arid1a`, `cptac_brca_tp53`, `cptac_pda_smad4`) all-context 무제한.
  e88(val_ce 0.5282)과 e125(val_ce 0.5142) 비교 — val_ce 0.014 개선이 test AUROC로
  전이되는지 확인.
- **Task 2 (진행 중)**: bag별 타일 제한 `{1000, 2000, 5000}` × **5 trial**(랜덤
  서브샘플, trial별 seed) vs 무제한(1 trial). 인스턴스(타일) 수가 성능에 영향을 주는지
  확인.
- **실행**: `predictions/pathobench_{task}_armC_batch2_e125_allctx_full.pt` (무제한),
  `predictions/pathobench_{task}_armC_batch2_e125_mt{1000,2000,5000}.pt` (제한),
  로그 `predictions/pathobench_e125_allctx_tilesweep.log`.

### Task 1 결과 — e125(0.5142) vs e88(0.5282), all-context 무제한

val_ce 0.5282→0.5142 개선이 실제 test로 전이되는지 확인. test AUROC:

| task | e88 | e125 | Δ |
|---|---|---|---|
| cptac_brca_tp53 | 0.696 | **0.714** | +0.018 |
| cptac_luad_tp53 | 0.625 | **0.637** | +0.012 |
| cptac_luad_stk11 | 0.786 | **0.795** | +0.009 |
| cptac_lscc_arid1a | 0.748 | 0.738 | −0.010 |
| cptac_pda_smad4 | 0.679 | **0.710** | +0.031 |

**판정**: 5 task 중 4개 개선(평균 +0.012), 1개 소폭 하락. **val_ce 개선이 대체로 test로
전이됨.** e125를 향후 기준 checkpoint로 확정.

### Task 2 결과 — bag별 타일 수 제한 스윕 (5-trial mean vs 무제한 1-trial)

| task | 무제한(1 trial) | 1000 | 2000 | 5000 |
|---|---|---|---|---|
| cptac_luad_tp53 | 0.637 | **0.722** | **0.724** | **0.743** |
| cptac_luad_stk11 | 0.795 | **0.842** | **0.840** | **0.846** |
| cptac_lscc_arid1a | 0.738 | 0.694 | 0.670 | 0.696 |
| cptac_brca_tp53 | 0.714 | 0.652 | 0.655 | 0.671 |
| cptac_pda_smad4 | 0.710 | 0.592 | 0.616 | 0.703 |

(trial별 분포: 제한 케이스는 5 trial AUROC min/max, 로그 참조)

**해석**:
- **인스턴스(타일) 수는 성능에 뚜렷한 영향을 주며, 방향은 task 의존적.**
- **LUAD 계열은 타일 제한이 오히려 개선** (tp53 0.637→0.72~0.74, stk11 0.795→0.84).
  대형 bag(최대 ~3.5만 타일)이 노이즈/혼란을 유발하는 듯 — 대표 서브샘플이 더 강건.
- **BRCA/LSCC/PDA는 무제한이 우세** (제한 시 −0.04~−0.12), PDA는 5000에서 무제한과
  비슷(0.703). BRCA는 test 22장으로 trial 간 분산이 큼.
- **한계**: 무제한은 1 trial(결정적, 전체 타일) vs 제한은 5-trial mean(랜덤 서브샘플)
  이라 잡음 수준이 다름. 제한 케이스는 trial mean이 무제한 단일값과 비슷하거나 위면
  서브샘플이 무해~유익, 아래면 무해하지 않음. 전반적으로 **bag 크기 정규화의 효과가
  task별로 갈림** — 후속으로 LUAD 대형 bag 분석(어느 bag이 문제인지) 권장.

---

## 48. 2026-08-06 — arm C top-up 완주(150ep, best e125) + v33 Phase 0 평가 확정 + PathoBench v30 비교

**상태**: arm C top-up이 **150 epoch까지 완주**했다(8×A6000 DDP, 에피소드-매치
4096/epoch, batch2). val_ce 기준 best인 `epoch=125-val_ce_loss=0.5142.ckpt`
(`checkpoints/20260806_145050/v33_phase0_armC_ddp8_batch2/`)를 채택해 §42 legacy
회귀 gate 재평가 + §45 Musk 재확인 + PathoBench(v30 vs v33) 비교를 수행했다.
**결론: legacy 회귀 gate 여전히 미달(+0.041) — 과소학습 편향 가설 기각, B2b 데이터
자체가 회귀 원인. Musk n>34 개선은 유지되지만 PathoBench에서는 v30이 우위.**

- **완주 정보**: 150 epochs, best epoch 125(val_ce 0.5142). §43의 중간(~88/150)
  기준을 갱신. 이번 평가는 완주 checkpoint 기준 최종 판정이다.

### 1. 합성 legacy 회귀 gate 재평가 (e125, 1,000 ep, seed 42, v30 legacy B2 val 스트림)

- 실행: `evaluate_synthetic.py --checkpoint e125 --config
  configs/train_v30_cardinality_poolz_l2.yaml --val-episodes 1000
  --output predictions/synthetic_v33_phase0_armC_topup_e125_legacy_1000ep.pt`
- **결과**: overall AUROC **0.8100 [0.798, 0.822]**, log loss 0.5218.
  per-task: composition 0.8611 / state 0.7526 / covariance 0.6347 / interaction
  0.7547 / combined 0.9411.
- **v30 committed 대비**: 0.8100 vs 0.8512 [0.840, 0.862] → **회귀 +0.0412**.
  CI가 완전히 분리(0.798~0.822 vs 0.840~0.862)되어 통계적으로 확정. gate(회귀 ≤ 0.01)
  **실패**.
- **핵심 판독 — 과소학습 편향 가설 기각**: 50ep arm C(0.8139) → 완주 e125(0.8100)로
  사실상 변동 없음. val_ce는 0.5351→0.5142로 크게 개선됐는데 legacy AUROC는 회복되지
  않았다. 에피소드-매치(4096/epoch, 150ep)로도 회귀가 사라지지 않으므로 **B2b(per-bag
  cardinality) 데이터 자체가 v30 legacy B2 val 분포에서 성능 저하를 일으킨다.**

### 2. Musk 재확인 (e125, 102 bags leave-one-out, seed 42)

- 실행: `test_musk.py --data .../Musk/musk.pkl --checkpoint e125 --config
  configs/train_v33_phase0_armC_ddp8.yaml --output predictions/musk_v33_armC_e125.pt`
- **Overall**: AUROC **0.8616 [0.779, 0.932]** (v30 0.8539 → +0.008), Acc 0.696,
  BAcc 0.730 (sens 0.872 / spec 0.587), log loss 0.531.

| 밴드 | v30 baseline | arm C e64 (§45) | arm C e125 (완주) |
|---|---|---|---|
| ALL | 0.854 | 0.880 | **0.862** |
| n≤4 | 0.800 | 0.700 | **0.725** |
| 5..10 | 0.833 | 0.925 | **0.958** |
| 11..34 | 0.958 | 0.970 | **0.939** |
| n>34 | **0.698** | 0.825 | **0.849** |
| corr(prob, log n) | +0.059 | −0.146 | **−0.176** |

- **paired bootstrap (v30 vs e125, 4000 resample)**: ALL +0.008 [−0.062, +0.077]
  P=0.593(무의미), n≤4 −0.075 [−0.242, +0.058] P=0.144, n>4 +0.038 [−0.044,
  +0.123] P=0.812.
- **판독**: §45 신호(n>34 0.698→0.849)가 완주 checkpoint에서도 유지·개선. 5..10
  0.958. 소형 trade-off(n≤4 0.800→0.725)도 유지. overall +0.008은 무의미.

### 3. PathoBench v30 vs v33(e125) — all-context, 전체 타일, 1 trial

v30 baseline(`checkpoints/20260804_132334/v30_cardinality_poolz_l2/epoch=048-
val_ce_loss=0.4442.ckpt`)으로 §47과 동일 프로토콜(all-context, 무제한, 1 trial,
seed 42) 5-task 평가. 예측 파일 `predictions/pathobench_{task}_v30_allctx_full.pt`.

| task | v30 baseline | e125 (arm C) | Δ (v30−e125) |
|---|---|---|---|
| cptac_luad_tp53 | **0.7431** | 0.6366 | **+0.107** |
| cptac_luad_stk11 | **0.9154** | 0.7949 | **+0.121** |
| cptac_lscc_arid1a | 0.6214 | **0.7381** | −0.117 |
| cptac_brca_tp53 | **0.7857** | 0.7143 | **+0.071** |
| cptac_pda_smad4 | **0.7246** | 0.7101 | +0.015 |
| 평균 | **0.758** | 0.719 | **+0.039** |

**판독**:
- **v30이 5개 중 4개 task 우위, 평균 +0.039.** 특히 LUAD 계열(stk11 +0.121,
  tp53 +0.107)에서 크게 우세.
- 유일한 e125 승리는 **lscc_arid1a (+0.117)** — 최대 bag(4.4만 타일) task로, B2b의
  대형 bag 강점(§45 Musk n>34)과 같은 방향.
- 1 trial이라 task별 수치에 노이즈 있음(§47 타일 스윕의 5-trial 안정성 관찰 참조).
- 합성 legacy 회귀(+0.041)와 같은 방향 — **arm C(B2b)가 전통 분포·PathoBench 레짐
  모두에서 v30보다 약함.**

### 4. 종합 판정

- **Phase 0 두 주 효과 모두 gate 미달 확정**: arm B sparse 0.6747 (<0.75), arm C
  legacy 회귀 +0.0412 (>0.01). 과소학습 편향 가설 기각.
- arm C의 **Musk n>34 개선(0.698→0.849)** 과 **PathoBench lscc 개선**은 실측·재현되는
  실질 신호이나(대형 bag 강점), 전반적으로 v30 대비 열위 → **v30 baseline 유지,
  arm C(v33 Phase 0) 미채택**.
- **다음**: ① Phase 0 결과 선택(사용자) — n>34/대형 bag 개선을 실질 이득으로 볼지,
  소형 희생 trade-off를 감수할지. ② frozen-v30 multi-resolution probe(Phase 1,
  paired AUROC +0.01 headroom) — 아키텍처 가설은 여전히 미검증(§39). ③ ICI 잠금 유지.
- 예측 파일: `predictions/synthetic_v33_phase0_armC_topup_e125_legacy_1000ep.pt`,
  `predictions/musk_v33_armC_e125.pt`, `predictions/pathobench_{task}_v30_allctx_full.pt` (5개).
- 참고: `test_pathobench.py`에 `--context-max-tiles`(context만 절단, query 무제한) 옵션
  추가 — 컨텍스트 크기 격리 실험용(이번엔 미사용).

---

## 49. 2026-08-07 — 아키텍처 효율화(MLA-slot) + v34-1536 대규모 컨텍스트 학습 완주 + PathoBench 5-fold CV

**상태**: v30 baseline 유지. 대규모 컨텍스트 학습을 가능하게 한 아키텍처 효율화 작업(MLA 계열)을
커밋·정리하고, **v34-1536(1024ep×50, batch=4) 학습을 완주**했다. PathoBench zero-shot 평가를
**5-fold CV**(전체 슬라이드, raw 1536-d)로 확장해 v34-1536을 평가했다.

### 1. 아키텍처 효율화 스택 (전부 커밋, 직전 세션 미문서화분 정리)

- `bfaee6a` **MLA 레이어** (`src/models/mla.py`): standalone MultiheadLatentAttention —
  W_DQ/UQ/DKV/UK/UV/O. 훈련(확장) vs 추론(KV-cache + matrix absorption) 동치 검증(5e-15).
  d_c=512 → KV cache **64× 압축**.
- `e98b3e2` **slot MLA 저랭크 affinity**: `slot_latent_dim`(d_c=64)/`slot_query_latent_dim`(128)/
  `slot_affinity_dim`(512) + `slot_w_dq/dkv/uq/uk`. None이면 full-dim dot과 **byte-identical**, 파라미터 0.
- `17a1c36` **slot_std 분산 트릭**: Var=E[X²]−E[X]², slot_distance=E_d[x²]−2(x·m)/dim+E_d[m²]
  → `[cells,slots,dim]` 텐서 제거 (default 경로 byte-identical).
- `7700e85` **배치 population candidates**: `_population_candidates_batched` — [C,max_cells,k]
  단일 masked softmax (per-bag 대비 ~2×, 285→147ms @1.93M-cell). 수치 동일 (전 bag ≥32 cells일 때).
- `778b40b` **정규화 통합**: `_instances_are_unit` — poolz_l2/centered+l2는 `_bag_view`에서 이미 unit
  → `_forward_dense`/`_population_candidates_batched`의 중복 F.normalize 생략 (mean 157→152ms, peak 47.5→43.8GB).
- `8571798` **smoke `--profiler`**: `scripts/smoke_train_budget.py` op-level 병목 테이블.
- 병목 진화: topk/sort → _population_candidates → div/mean/AdamW (~152ms/step @[1,32768]).

### 2. v34 config + 배치=4 판정

- `40950de`/`1f2a23e`: `train_v34_phase0_largectx_512.yaml` ([1,32768], slot MLA) /
  `..._1536.yaml` (1536-d, [1,8192]) — scratch.
- 배치=4 스모크(1536): fp32 peak **85.5GB**(B200 178GB 중 48%). bf16 autocast는 **더 나쁨**
  (peak 110.2GB, mean 669ms vs fp32 412ms — aggregator의 `.float()`/fallthrough가 autocast 이득 상쇄)
  → **fp32 유지** (v30 baseline과도 일치). 스모크에 `--bf16` A/B 플래그 추가.

### 3. v34-1536 학습 (1024×50, batch=4) — 완주

- Run `20260806_215800`, `train_v34_phase0_largectx_1536.yaml` (1024 ep/epoch × 50 = 51,200 ep,
  batch 4, fp32, slot MLA). 완주 ~50min (~5.5 it/s).
- **best epoch=048: val_ce 0.4419 / val_auroc 0.8353** (v30 best 0.4442보다 소폭 하회 — 더 어려운
  대규모 컨텍스트 분포에서도 수렴). 마지막까지 발산 없음.
- 체크포인트: `checkpoints/20260806_215800/v34_phase0_largectx_1536/epoch=048-val_ce_loss=0.4419.ckpt`.

### 4. PathoBench 5-fold CV (신규 도구) + 결과

- `test_pathobench.py --cv-folds N`: **train+test 통합 전체 슬라이드에 stratified K-fold**. fold k는
  나머지 N−1 fold가 all-context. per-fold + pooled AUROC 보고.
- **영구 fold .pt 파일** (`data/pathobench/{task}_cvfold{i}.pt`, raw 1536-d): 1회 h5에서 생성,
  이후 실행은 **h5/cache 완전 스킵** (동일 fold 분할, 체크포인트 간 공정 비교).
- cache dim 버그 수정: 512-d 캐시는 input_dim==512일 때만 사용 (1536-d 모델의 캐시 오사용 방지).
- eval OOM 수정 (`000aead`): `_context_anchors`의 배치 후보 경로를 **훈련 전용**으로 제한, eval은
  **per-bag 루프** (수치 동일). 배치 경로가 [C,max_cells,1536] 패딩으로 luad_tp53/stk11/lscc에서
  48-61GB OOM을 일으켰던 문제 해결.

**결과** (raw 1536-d, all-context, 전체 타일):

| task | slides | fold-mean | pooled |
|---|---:|---:|---:|
| brca_tp53 | 112 | 0.8600 | 0.8484 |
| pda_smad4 | 242 | 0.8297 | 0.8314 |
| luad_stk11 | 324 | **0.9745** | **0.9774** |
| luad_tp53 | 324 | 0.9456 | 0.9442 |
| lscc_arid1a | 304 | 0.9157 | 0.9083 |
| **평균** | | **0.905** | **0.902** |

- LUAD 계열 강세(stk11 0.977, tp53 0.944), lscc 0.908, brca 0.848, pda 0.831.
  pda_smad4는 §46에서 랜덤 이하(0.309)였던 task가 5-fold에서 0.83까지 상승.
- 예측: `predictions/pathobench_{task}_v34_1536_cv5.pt` (5개).
- §48의 v30(train/test split, PCA 512-d) 평균 0.758과는 프로토콜이 달라 **직접 비교 불가**.

### 5. 열린 과제

- **v34-512 미학습** (512-d config만 준비). 학습 시 같은 fold 파일로 평가 가능.
- **v30 vs v34 CV 비교 불가 (현재)**: CV는 raw 1536-d 전용 (PCA-per-fold 미지원).
  공정 비교하려면 fold별 context-only PCA 로직이 필요.
- **v34-1536 Musk zero-shot 평가 미실행** (musklike-easy).
- v34-1536 합성 val_auroc ~0.835는 v30 synthetic(~0.95)보다 낮지만 분포가 달라 직접 비교 불가
  — PathoBench가 실질 판정 기준.

---

## 50. 2026-08-07 — v34-1536 추가 평가: Musk 패딩 브리지(타일), ICI(랜덤), PathoBench 17-task 전체 CV

**상태**: v30 baseline 유지. v34-1536의 나머지 평가(Musk 브리지 실험, ICI 최종 테스트, PathoBench 전체 17개 binary task)를 완료했다.

### 1. Musk — 패딩 브리지 실험

- `test_musk.py`가 config input_dim으로 동적 패딩(`4aca7f1`), `--pad-mode` 추가(`6d4c5bc`).
- **zero-pad**(166→1536, 90% 0): AUROC **0.8217** [0.731, 0.905].
- **tile**(166×9 + 42 zero): AUROC **0.8575** [0.772, 0.928] (+0.036) — 입력 대부분이
  실제 신호를 담아 개선, v30(0.854)과 동등~소폭 상회.
- tile stratified vs v30: n≤4 0.667(0.800), 5..10 0.917(+0.084), 11..34 0.988(+0.030),
  n>34 0.683(0.698) — 중간 밴드 크게 개선, 소형 bag은 여전히 trade-off.

### 2. ICI — 실세계 최종 테스트 (명시적 잠금 해제, 헤더 사용자 결정 2항 예외)

- `ICIDataset`에 input_dim/pad_mode 추가(`f8181be`): ICI 512-d scConcept를
  512×3=1536으로 타일. `target_cells=-1` = **전체 cell**(1.2k~6.3k/donor, 1000 제한 없음).
- config `test_v34_phase0_largectx_1536_ici.yaml`.
- 5-seed 프로토콜: **across-seed AUROC 0.5117 ± 0.0268** (범위 0.4859..0.5449),
  seed 평균 per-donor **0.5070 [0.381, 0.629]** — **실질 랜덤**. 기존 ICI 결과
  (0.5454~0.5665)와 일관. 실세계 ICI는 여전히 미통과.

### 3. PathoBench — 전체 17개 binary task 5-fold CV

- 신규 12개: `bc_therapy_{er,grade,her2}`, `brca_pik3ca`, `ccrcc_{er,grade,her2}`,
  `lscc_{histologic,keap1}`, `luad_{egfr,kras}`, `ucla_lung_progression_regression`
  (§49의 5개와 합쳐 17개).
- **14개 유효 고유 task 평균 pooled 0.843** (fold-mean 0.846) ⚠️ §51 정정: 로컬 `cptac_ccrcc_*`는
  `bc_therapy`의 잘못된 복사본이라 실측 데이터셋은 **6개(BC_Therapy/BRCA/LSCC/LUAD/PDA/UCLA)**다.
  - 강함(0.91~0.99): lscc_keap1 0.985, luad_stk11 0.977, luad_kras 0.958, lscc_histologic 0.948,
    luad_egfr 0.945, luad_tp53 0.944, lscc_arid1a 0.908
  - 중간(0.78~0.86): brca_tp53 0.848, pda_smad4 0.831, ucla_lung 0.784
  - 약함(0.63~0.70): bc/ccrcc_er 0.704, bc/ccrcc_grade 0.674, bc/ccrcc_her2 0.673, brca_pik3ca 0.627
  - ⚠️ bc/ccrcc_er·grade·her2는 모두 **bc_therapy 수치의 재표기**(cv5 예측 바이트 동일)이며,
    **실제 CPTAC-CCRCC 코호트(BAP1/PBRM1/VHL/Immune/OS)는 미평가** (§51).
- **유전체 alteration task(KEAP1/STK11/KRAS 등)에서 매우 강함**, 호르몬/등급 표현형은 상대적 약세.

### 4. 열린 과제

- **PCA-per-fold CV**(v30 vs v34 공정 비교) — 여전히 미지원.
- **v34-512 학습 미실행** (학습 시 같은 fold 파일로 재평가 가능).
- v34-1536 종합: PathoBench·Musk는 실질 신호, ICI는 랜덤 — 채택/판정은 사용자.

---

## 51. 2026-08-07 — PathoBench 원본 검증: 로컬 cptac_ccrcc CSV는 bc_therapy의 잘못된 복사본

**상태**: 사용자 요청으로 **공식 Patho-Bench(HF `MahmoodLab/Patho-Bench`, arXiv:2502.06750) 원본
스플릿을 확보**해 로컬 데이터와 대조했다. **결론: 로컬 `cptac_ccrcc_{er,grade,her2,residual}.csv`는
`bc_therapy`의 바이트 단위 복사본이며, 공식 CPTAC-CCRCC 코호트를 전혀 담지 않은 로컬 데이터 오류다.**
이전 §46·§50의 “BC_Therapy==CPTAC-CCRCC 동일 슬라이드” 표현은 벤치마크 속성이 아니라
**로컬 취득 오류**로 정정한다. 예측 수치(0.843)는 그대로지만 커버리지 해석은 바뀐다.

### 1. 공식 원본 (HuggingFace `MahmoodLab/Patho-Bench`, 2026-08-07 확보, `/tmp/pb_official/`)

- 공식 task 이름 (`available_splits.yaml`):
  - `bc_therapy`: **er_status / grade / her2_status / residual_cancer_burden** (166 슬라이드, 숫자 ID)
  - `cptac_ccrcc`: **BAP1_mutation / Immune_class / OS / PBRM1_mutation / VHL_mutation** (245 슬라이드, `C3L-*`/`C3N-*`)
- `bc_therapy`·`cptac_ccrcc`는 **서로 다른 소스 데이터셋** (Zenodo 6337925 vs TCIA CPTAC-CCRCC).

### 2. 로컬 vs 공식 대조 결과

| 대조 | 결과 |
|---|---|
| 로컬 `bc_therapy_er.csv` vs 공식 `bc_therapy/er_status/k=all.tsv` | **166/166 슬라이드 일치, 라벨 불일치 0** → 로컬 bc_therapy는 정상(이름만 축약) |
| 로컬 `cptac_ccrcc_er.csv` vs 공식 `cptac_ccrcc/BAP1_mutation/k=all.tsv` | **0/166 슬라이드 일치** |
| 공식 ccrcc 슬라이드 vs 로컬 ccrcc | 겹침 0 (로컬은 `634925` 등 bc_therapy 숫자 ID) |
| 로컬 `cptac_ccrcc_er.csv` vs `bc_therapy_er.csv` | **바이트 단위 동일** (diff IDENTICAL, er/grade/her2/residual 전부) |
| cv5 예측 `cptac_ccrcc_{er,grade,her2}` vs `bc_therapy_*` | **fold AUROC·pooled·확률 전부 바이트 동일** |

### 3. §50 “17개 task CV”에 대한 영향

- 17개 항목 중 **3개(`cptac_ccrcc_{er,grade,her2}`)는 bc_therapy의 잘못된 중복** → 유효 고유 task
  **14개** (실측 **6개 데이터셋**: BC_Therapy/BRCA/LSCC/LUAD/PDA/UCLA).
- **“14개 유효 task 평균 pooled 0.843”** 수치는 14개 task 단순평균으로 **수치 자체는 유효**
  (bc/ccrcc 3개를 1회로 중복 제거한 계산과 일치).
- 단, **실제 CPTAC-CCRCC 코호트(BAP1/PBRM1/VHL/Immune/OS)는 한 번도 평가되지 않음** —
  §49·§50의 “전체 17개 binary task”는 사실상 ccrcc 코호트가 빠진 6개 데이터셋 평가였다.
- `bc/ccrcc_er 0.704 / grade 0.674 / her2 0.673`은 bc_therapy 수치의 재표기.

### 4. 열린 과제 (수정 방향)

- (권장) 로컬 `cptac_ccrcc_*` CSV를 공식 TSV로 교체/삭제하고, **실제 cptac_ccrcc
  task(BAP1/PBRM1/VHL/Immune/OS 등) 5-fold CV 재실행**해 보고 평균을 재계산.
- 공식 스플릿은 `k=all.tsv`(case_id/slide_id/label + fold_0..49) — 기존 `slide_id,label,split`
  CSV 포맷으로 변환 필요.
- `data/pathobench/` 캐시(fold .pt)도 해당 task 재생성 필요.

### 5. 전체 30개 CSV 전수 감사 (2026-08-07, `/NHNHOME/BASE/kimds/Data` 검증)

사용자 요청으로 `features/*.h5`(원시 피처)와 **전체 30개 로컬 CSV**를 공식 스플릿과
전수 대조했다. **원시 피처(h5)는 정상**: 모든 데이터셋 keys=[barcodes,coords,features],
[n,1536] float32, NaN 0건 — 손상 없음.

| 상태 | CSV |
|---|---|
| ✅ 슬라이드·라벨 정상 | `bc_therapy_{er,grade,her2,residual}`(166/166·159/159), `cptac_brca_{immune,pik3ca,tp53}`(112), `cptac_lscc_{arid1a,histologic,immune,keap1}`(304/292), `cptac_luad_{egfr,immune,kras,stk11,tp53}`(324/312), `cptac_luad_os`(313)·`cptac_pda_os`(227)·`mbc_os`(96), `cptac_pda_{immune,smad4}`(242), `mbc_recist`(97), `ucla_lung_progression_regression`(112), `bracs_{coarse,fine}`(547), `herroi_response`(85) — 공식 `task_col`과 슬라이드·라벨 100% 일치 |
| ❌ **데이터 소스 오류** | `cptac_ccrcc_{er,grade,her2,residual}` — `bc_therapy`의 바이트 동일 복사본 (§51 상세) |
| ⚠️ 피처 부재(기지) | `herroi_response`: `HER2_tumor_ROIs_v3` 빈 폴더(85장 전부) → 평가 제외. `bracs`: 7장 피처 없음 |

- **OS task는 survival task다**: 공식 `task_type: survival`, `task_col: OS`(0~7 복합 라벨 =
  OS_days 사분위 + OS_event), `extra_cols: [OS_event, OS_days]`, metric `cindex`.
  ⚠️ 초기 감사에서 이진 `OS_event`와 잘못 비교해 OS 3종을 "라벨 오류"로 오탐했으나,
  공식 `OS` 컬럼과 **100% 일치**로 정정 — **로컬 OS CSV는 정상**.
- **§49·§50의 17-task CV 수치에는 영향 없음**: 평가된 17개 중 OS task는 없고, ccrcc 3개는
  중복 제거됨 → 14개 유효 task 수치(0.843)는 유효.
- **결론**: "데이터 자체"는 정상(피처 무손상, **26/30 CSV 정상**). 잘못된 것은 **ccrcc 4건
  (소스 오류)뿐**이다.

### 6. 공식 fold별 split 확보 (2026-08-07, `scripts/fetch_pathobench_official.py`)

사용자 요청으로 **공식 fold별 split 전체를 다운로드**했다:
`/NHNHOME/BASE/kimds/Data/PathoBench/official/{source}/{task}/{k=all.tsv, config.yaml}` (10개 소스,
**31개 task**). `k=all.tsv`는 `case_id, slide_id, <task_col>, fold_0..fold_49`(값 train/val/test),
`config.yaml`이 `task_col`/`label_dict`/`task_type`을 정의한다. `SplitFactory.from_local` 호환 레이아웃.

- 재다운로드: `python scripts/fetch_pathobench_official.py [--source ...]`.
- 참고: OS·bracs task는 `nfold=5`, 그 외는 50 — 공식 설정이 task별로 다르다.

### 7. 공식 라벨 CSV 생성 (2026-08-07, `scripts/build_pathobench_official_csvs.py`)

공식 `k=all.tsv`(+`config.yaml`의 `task_col`)에서 **슬라이드 단위 공식 라벨 CSV**를 생성했다:
`/NHNHOME/BASE/kimds/Data/PathoBench/csv_official/{source}_{task}.csv` (`slide_id,label,split`,
split은 지정 fold(기본 fold_0)의 train/val/test) — **31개 task 전부**.

- **교차 검증 (legacy CSV 존재 26개)**: 전부 **라벨 100% 일치** (bc_therapy/brca/lscc/luad/pda/
  mbc/ucla/bracs/herroi + OS 3종). → 기존 로컬 CSV는 ccrcc 4건 외엔 모두 정상임을 공식 라벨로 재확인.
- **실제 cptac_ccrcc 5개 task 라벨 신규 확보** (기존에 없었음): `BAP1_mutation`(245장, 이진),
  `PBRM1_mutation`(245), `VHL_mutation`(245), `Immune_class`(245, 3클래스), `OS`(218, 7클래스) —
  전부 `C3L/C3N` 슬라이드.
- 재생성: `python scripts/build_pathobench_official_csvs.py [--fold 0]`.
- 다음: 실제 ccrcc task(BAP1/PBRM1/VHL 등) 5-fold CV 평가 + 보고 평균 재계산.

---

## 52. 2026-08-07 — 실제 ccrcc 평가 완료 + SEAL baseline 비교 + 공식 50-fold 평가 계획

### 1. 실제 cptac_ccrcc 5-fold CV (v34-1536, all-context, 전체 타일) — 완료

| task | 슬라이드 | per-fold AUROC | fold-mean | pooled |
|---|---|---|---|---|
| `BAP1_mutation` | 245 | 0.894 / 0.978 / 0.908 / 0.906 / 0.922 | **0.9218** | **0.9223** |
| `PBRM1_mutation` | 245 | 0.752 / 0.806 / 0.820 / 0.648 / 0.896 | **0.7844** | **0.7782** |
| `VHL_mutation` | 245 | 0.877 / 0.772 / 0.900 / 0.645 / 0.840 | **0.8064** | **0.8082** |

- 예측: `predictions/pathobench_cptac_ccrcc_{BAP1,PBRM1,VHL}_mutation_v34_1536_cv5.pt`,
  fold 파일 `data/pathobench/cptac_ccrcc_*_cvfold{0..4}.pt`.
- 기존 §50의 "ccrcc_er 0.704"는 로컬 오류(BC Therapy 복사본)였고, **진짜 CCRCC 코호트는
  유전체 변이 3종 모두 실질 신호** (BAP1 0.922 특히 강함).

### 2. SEAL baseline(지도 ABMIL/MeanMIL, 50-fold macro-AUC) vs 우리 기록

파일 `docs/seal_univ2_baseline_17tasks.csv` (SEAL 논문 표에서 정리). **10개 task가 SEAL에 존재**.
우리 기록 = v34-1536 **zero-shot in-context**(학습 없음) 5-fold CV **pooled AUROC**.

| task | SEAL ABMIL | SEAL MeanMIL | 우리 | 비교 |
|---|---|---|---|---|
| bc_therapy er_status | 0.717±.086 | 0.712±.091 | 0.704 | ≈ |
| bc_therapy grade | 0.770±.066 | 0.751±.058 | 0.674 | ▼ |
| bc_therapy her2_status | 0.663±.092 | 0.684±.073 | 0.673 | ≈ |
| cptac_brca PIK3CA_mutation | 0.595±.103 | 0.544±.120 | 0.627 | ▲ |
| cptac_brca TP53_mutation | 0.801±.093 | 0.787±.088 | 0.848 | ▲ |
| cptac_luad EGFR_mutation | 0.830±.089 | 0.777±.099 | 0.945 | ▲▲ |
| cptac_luad STK11_mutation | 0.908±.052 | 0.873±.072 | 0.977 | ▲ |
| cptac_luad TP53_mutation | 0.751±.102 | 0.735±.102 | 0.944 | ▲▲ |
| cptac_ccrcc BAP1_mutation | 0.693±.150 | 0.720±.145 | 0.922 | ▲▲ |
| cptac_ccrcc VHL_mutation | 0.538±.128 | 0.542±.133 | 0.808 | ▲▲ |

- 요약: **10개 중 우리 상회 8, 유사 1(her2), 하회 1(grade)**. 유전체 변이 task에서 특히 우위.
- ⚠️ **프로토콜 차이 (논문 작성 시 명시 필수)**: SEAL은 지도 MIL 학습(ABMIL/MeanMIL,
  50-fold, macro-AUC), 우리는 **zero-shot in-context(학습 없음)** all-context 5-fold pooled
  AUROC. fold 수(50 vs 5)·컨텍스트 구성이 다르므로 "동일 프로토콜 직접 비교"는 아님.
- ⚠️ **n 차이(ccrcc)**: SEAL ccrcc BAP1·VHL n=218 vs 우리 245 슬라이드 → §52.4에서 규명:
  SEAL은 임상(OS) 데이터 보유 94 case/218장만 사용, 공식 Patho-Bench는 103 case/245장 전체.
  **우리는 공식 프로토콜(245장)을 따른다.**
- SEAL에 없는 7개 (우리만 평가): lscc_arid1a 0.908, lscc_histologic 0.948, lscc_keap1 0.985,
  luad_kras 0.958, pda_smad4 0.831, ucla_lung 0.784, ccrcc_pbrm1 0.778.

### 3. 공식 50-fold 평가 모드 구현 + 계획

- **`test_pathobench.py --official-folds <task_dir>` 신규 구현**: 공식 `k=all.tsv`(fold_0..) +
  `config.yaml`(task_col)을 읽어, **각 공식 fold k**에서 `fold_k=='test'` 슬라이드를 쿼리하고
  나머지(train+val)를 all-context로 사용. per-fold AUROC + **fold-mean±std + pooled** 보고
  (SEAL의 50-fold macro-AUC 프로토콜과 동일 구조). `--official-nfolds N`으로 부분 검증 가능.
  raw 1536-d 필요(입력 1536). `--csv` 없이 실행 가능.
- **검증 완료**: bc_therapy/er_status 2-fold smoke (166 슬라이드, 50 공식 fold 인식).
- **다음 할일**: 17개 task(7개 데이터셋)에 대해 공식 fold 따라 **50-fold 평가** 실행 →
  SEAL과 동일 프로토콜의 수치로 재비교. 예상: task당 ~50분, 17개 전체는 수 시간(GPU 백그라운드).

### 4. ccrcc 코호트 차이 규명 + 우리 프로토콜 명시

- **공식 Patho-Bench ccrcc = 245장 / 103 case** (BAP1/PBRM1/VHL, 50 folds) — HF 커밋 이력
  검증(2025-02 추가 이후 0장 증감, 버전 무관).
- **SEAL n=218 = 임상(OS) 데이터 보유 subset(94 case)** — OS task 슬라이드 수(218)와 정확히
  일치. 차이 27장 = OS 데이터 없는 9개 case(`C3L-00812/13/14`, `C3N-00148/49/50/54`,
  `C3N-00573`, `C3N-00646`). 이 9개 case는 타일 수 정상(중앙값 8,935 vs 4,379)이나
  **BAP1 양성률 2배(33% vs 16%)** — 추출 실패가 아닌 임상 데이터 유무 차이.
- **우리 프로토콜 명시**: 모든 PathoBench 평가는 **공식 Patho-Bench 프로토콜을 따른다** —
  ① 공식 `k=all.tsv`의 **공식 fold**(50-fold 등), ② **공식 코호트**(ccrcc 변이 = 103 case/245장,
  SEAL의 218장 subset 미채택), ③ **공식 라벨**(`config.yaml`의 `task_col`). SEAL과 비교 시에는
  코호트 차이(245 vs 218)를 명시하고, 필요하면 218장 subset 병행 보고로 투명화한다.

---

## 53. 2026-08-07 — v34 최종 확정 + 공식 50-fold 평가(SEAL 동일 프로토콜) 진행

**상태**: 사용자 결정으로 **v34-1536을 PathoBench 보고용 모델로 확정**. 평가는 **공식
Patho-Bench 프로토콜**(공식 k=all.tsv fold · 공식 코호트 · 공식 라벨 task_col)로 **50-fold**를
돌리고 있다(SEAL의 macro-AUC 프로토콜과 동일 구조). 17개 task 중 일부 완료, 배치 진행 중
(`logs/official50/batch.log`).

### 1. v34 확정 사항

- **모델**: v34-1536 (slot MLA 저랭크 affinity + slot_std 분산 트릭 + 배치 population
  candidates + 정규화 통합, scratch). best `val_ce 0.4419`
  (`checkpoints/20260806_215800/v34_phase0_largectx_1536/epoch=048-val_ce_loss=0.4419.ckpt`).
- **평가 프로토콜(확정)**: 공식 Patho-Bench — 공식 `k=all.tsv`의 공식 fold(50-fold), 공식
  코호트(ccrcc 변이 245장/103 case, SEAL의 218장 subset 미채택), 공식 라벨(`config.yaml` task_col).
  all-context, 전체 타일, raw 1536-d, **zero-shot in-context(학습 없음)**.
- **도구**: `test_pathobench.py --official-folds <task_dir>` + 병렬 러너
  `scripts/run_official_folds_parallel.py` (워커 분할 10→6→4→2 자동 축소 + **per-fold
  체크포인트** → 중단 후 리쥼, fold는 정적·결정적이라 재계산 불필요). h5는 워커별 직접 로드.

### 2. 공식 50-fold 결과 (완료분, fold-mean±std / pooled)

| task | 50-fold mean±std | pooled |
|---|---|---|
| bc_therapy/er_status | 0.6741 ± 0.101 | 0.6721 |
| bc_therapy/grade | 0.7148 ± 0.072 | 0.7126 |
| bc_therapy/her2_status | 0.6715 ± 0.076 | 0.6696 |
| cptac_brca/PIK3CA_mutation | 0.5748 ± 0.106 | 0.5690 |

- 나머지 13개 task 배치 실행 중 (`logs/official50/batch.log`) — 완료 후 이 표 갱신.
- bc_therapy 3개: 5-fold pooled(0.704/0.674/0.673) vs 50-fold pooled(0.672/0.713/0.670)는
  **±0.03~0.04 이내로 동일**(fold std 안) → 평가 견고성 확인.
- workers=10 병렬로 소형 task ~11분/개; 큰 task는 OOM 시 자동 축소(10→6→4→2), 결과 동일.

### 3. SEAL baseline과 50-fold 비교 (완료분)

| task | SEAL ABMIL | SEAL MeanMIL | 우리 50-fold(pooled) |
|---|---|---|---|
| bc_therapy/er_status | 0.717 | 0.712 | 0.672 |
| bc_therapy/grade | 0.770 | 0.751 | 0.713 |
| bc_therapy/her2_status | 0.663 | 0.684 | 0.670 |
| cptac_brca/PIK3CA_mutation | 0.595 | 0.544 | 0.569 |

- bc_therapy 계열은 SEAL과 비슷~소폭 하회, PIK3CA는 SEAL 사이. 나머지 task는 배치 완료 후 갱신.

### 4. 열린 과제

- 공식 50-fold 17개 전체 완료 → 최종 표 + SEAL 재비교 (§53 갱신).
- **v34-512 학습** (미실행). **PCA-per-fold CV** (v30 vs v34 공정 비교, 미지원).
- v30 medium 참조 재학습, frozen-v30 multi-resolution probe(§39) 등은 기존 열린 과제 유지.

---

## 54-55. 2026-08-07 — 아카이빙 정리 + 리팩터링 1단계 (완료, 아카이브됨)

두 절 모두 종료된 정리 작업이라 전문을 [`history/archive.md`](history/archive.md)로 이관했다.
요약: §54 = 구버전 문서/config/스크립트 아카이빙 + v34 태그, §55 = AST 정적 분석으로 미사용
함수 제거. 열린 과제 없음.

---

## 56. 2026-08-07 — config 시스템 리팩터링(v34 base·default 참조·재아카이빙) + 공식 50-fold 재시작

**상태**: **v34-1536 = PathoBench 보고용 확정 유지**. config 시스템을 **v34 base + group default
참조형**으로 재구성하고, v30/v24/v22 체인을 **자체 포함형 아카이빙**으로 정리했다. 공식 50-fold
배치는 아카이빙 회귀로 전부 실패했던 것을 config 수정으로 해결하고 **5/17 완료 → 12개 재시작**
(백그라운드).

### 1. v34 config = default 참조형 (단일 진실 공급원)

- `configs/train_v34_phase0_largectx_1536.yaml`·`_512.yaml`을 `data/model/optimizer/scheduler/
  trainer/logger/callbacks: default` 참조로 단순화. **group default를 v34-1536 해석값으로 설정**
  (`configs/{data,model,optimizer,scheduler,trainer,callbacks,logger}/default.yaml` — optimizer/
  scheduler/logger는 신규).
- `src/utils/utils.py merge_train_config`에 **`logger_overrides`·`trainer_overrides` 지원 추가**
  (experiment_name/max_epochs 등 run별·arm별 override용).
- v34-512는 dimension(512) + arm-D 레시피(batch 1, num_cells [1,32768], episodes 256, epochs 25)만
  override로 유지 (사용자 결정).
- 검증: 두 v34 config 해석 결과가 이전(자체 포함형/원본)과 **딥 이퀄**. 전체 config 141개 해석 성공.

### 2. 재아카이빙 + 아카이빙 정책

- root = **v34 3종만** (`train_v34_1536`/`_512`/`test_v34_1536_ici`). v30 5종+eval_v30 2종 →
  `archive/v30/`, v24 2종 → `archive/v24/`, v22_medium → `archive/v22/`. 이동 시 base_config
  상대경로를 `../v22/`·`../v24/`·`../v30/`·`../v18_v19/`로 보정 — **아카이브 전체 자기완결**.
- 기존 아카이브의 숨은 깨짐(ia_mil·musklike_easy_levers·v23_v24_candidates·v25·v26·v31·v32·v33,
  19개)도 모두 수정. v18_v19의 learnability 10개는 커밋 a5dfcf8에서 의도적으로 purge된 data 모듈을
  참조하는 **기존 결함**(역사 보존용, 활성/체인과 무관).
- **아카이빙 정책 신설(handoff §7 규칙 3)**: 아카이빙 config는 `base_config` 없이 **전부 인라인
  (자체 포함형)**으로 보관 → 상대경로 깨짐 원천 차단.

### 3. 공식 50-fold 재시작 (config 회귀 해결)

- 원인: 이전 배치(12:54~12:59)가 아카이빙된 `configs/train_v24_musklike_easy.yaml`을 참조해 17개
  전부 rc=1 실패 → v34 config 자체 포함/default 참조화로 해결 (smoke에서 config 해석 통과 확인).
- 배치 스크립트 신규: `scripts/run_official50_batch.sh` (17개 task, 완료분 스킵, workers
  10→6→4→2 자동 축소, per-fold 체크포인트 리쥼). 로그 `logs/official50/batch_resume.log`.
- **완료 5개(pooled)**: bc_therapy er 0.672 / grade 0.713 / her2 0.670, cptac_brca_PIK3CA 0.569,
  cptac_brca_TP53.
- **재시작(14:17 KST, 12개 백그라운드)**: lscc(3)·luad(4)·pda(1)·ucla_lung(1)·ccrcc(3).
- ⚠️ **14:17 1차 재시작은 ARID1A에서 OOM 연쇄로 중단**: `run_official_folds_parallel.py`가 worker
  실패 시 형제 worker를 종료하지 않아(고아 3개가 GPU ~166GB 점유) workers 10→6→4→2 재시도가 전부
  즉시 OOM. **러너 수정**(worker 실패 시 전체 worker kill → GPU 해제) 후 **14:26 재실행**(nohup,
  PID 723428) — ARID1A(304 슬라이드, worker당 ~50GB)는 깨끗한 GPU에서 workers=2로 수용. 완료 후
  §53 표 갱신 + SEAL 재비교.
- ARID1A 2-fold smoke는 10분 timeout으로 종료(대형 task 1-fold 평가가 10분 초과 — config 문제 아님).

### 4. 공식 50-fold 진행 (6/17 완료) + 리팩터링 최신화

- **ARID1A 완료 (6/17, 15:37)**: 50-fold mean **0.4693 ± 0.1093**, pooled **0.4616**
  (`predictions/pathobench_cptac_lscc_ARID1A_mutation_v34_1536_official50.pt`).
  이전 5-fold(§50 lscc_arid1a 0.908)와 큰 차이 — **공식 fold/코호트 프로토콜 차이**로 기록.
- **배치 일시정지 (사용자 요청)**: ARID1A 완료 직후 감시 스크립트(`/tmp/pause_after_arid1a.sh`)
  가 배치 스크립트+워커 종료. **잔여 11개**: lscc(2)·luad(4)·pda(1)·ucla_lung(1)·ccrcc(3).
  재개: `nohup bash scripts/run_official50_batch.sh` (완료분 스킵).
- **리팩터링 (폐기 분기 최신화, §56.8-9)**: 백업(태그 `repro-pre-deprecated-cleanup-20260807` +
  `src/repro_backup_20260807/`) 후 ① 죽은 메서드 3개(§56.8), ② **CCER(v31) ~570줄**, ③
  **DR-CCER(v32) ~800줄** 제거 — 각각 파라미터 시그니처 동일(220그룹/41.67M)·forward 동치
  (dense/ragged diff 0)·checkpoint strict 로드(0/0) 검증, **전체 테스트 32개 통과(148.5s)**.
  남은 폐기 분기: typed_bag(v25)·cls_token(v26)·IA-MIL·CCTS/absolute_tail·mean_pool(v23).

### 5. 다음

- 50-fold 잔여 11개 재개 → §53 표 **17개 전체 갱신** + SEAL 재비교.
- 폐기 분기 최신화 계속(typed_bag→cls_token→MIL→CCTS→mean_pool) 또는 여기서 종료.
- v34-512 학습 + 동일 평가(열린 과제 ③), v30 vs v34 PCA-per-fold 공정 비교.

---

## 57. 2026-08-07 — 50-fold 재개 전 진단: 5-fold CV의 case leakage로 lscc_arid1a 0.908이 부풀려짐

**상태**: 공식 50-fold 재개 전에 "이상한 AUROC"(lscc_arid1a 50-fold pooled **0.462** vs
기존 5-fold **0.908**)를 진단했다. 결론: **50-fold 계산은 정상**이며, **기존 5-fold CV가
case leakage로 multi-slide task의 AUROC를 부풀렸다.** 공식 50-fold 0.4616이 ARID1A의
정직한 성능(실질 랜덤)이다.

### 1. 검증된 사실 (계산 버그 아님)

- 완료된 50-fold 6개 전부 **pooled 재계산 = 저장값 일치** (bc_therapy 0.6721/0.7126/0.6696,
  PIK3CA 0.5690, TP53 0.8084, ARID1A 0.4616).
- 50-fold와 5-fold는 **같은 체크포인트**(`epoch=048-val_ce_loss=0.4419.ckpt`)·같은 raw 1536-d·
  all-context·전체 타일. 라벨도 공식/repro/legacy **전부 일치** (304장, diff 0).
- 차이는 오직 **폴드 구성**:
  - **cv5(§50)**: slide-level 층화 5-fold — **108 case 중 82개 case가 여러 fold에 분산**
    (총 406개 cross-fold case slot). → query slide의 **같은 case 슬라이드가 context에 존재**.
  - **공식 50-fold**(k=all.tsv, `sample_col: case_id`): fold 내 case 분할 **0건** (case-disjoint).

### 2. leakage 메커니즘 직접 확인 (cv5 예측 파일 재분석)

| query 그룹 | n | AUROC | pos prob mean | neg prob mean |
|---|---:|---:|---:|---:|
| case-mate가 context에 있음 | 268/304 (88%) | **0.9258** | 0.561 | 0.139 |
| case-mate 없음 | 36 | 0.6857 (양성 1장뿐, 사실상 랜덤) | 0.322 | 0.283 |

→ 모델이 형태학적으로 동일한 **case-mate 슬라이드를 context에서 "인식"해 라벨을 예측**.
leakage가 없으면 ARID1A는 zero-shot in-context로 실질 랜덤 (공식 50-fold fold-mean
0.4693 ± 0.1093, pooled 0.4616).

### 3. 영향 (중요)

- **공식 50-fold 프로토콜이 정직한 기준** — case-disjoint이므로 재개·보고해도 안전 (잔여 11개).
- **§50/§52의 5-fold 결과는 multi-slide task에 대해 case leakage로 부풀려짐**:
  - bc_therapy (166 case = 166 slide, 1:1): 안정 (er 0.704→0.672, grade 0.674→0.713, her2 0.673→0.670).
  - brca (112 slide/103 case): 하락 (PIK3CA 0.627→0.569, TP53 0.848→0.808).
  - **ARID1A (304 slide/108 case): 붕괴 (0.908→0.462)**.
- §52 SEAL 재비교는 17개 50-fold 완료 후 **공식 50-fold 수치로 전면 갱신**해야 한다.
  lscc_arid1a 0.908 (leaked) → 0.462 (honest)는 논문에 반드시 명시.

### 4. 다음

- 공식 50-fold **잔여 11개 재개** (`nohup bash scripts/run_official50_batch.sh`) → §53 표 17개 전체 갱신.
- (선택) multi-slide task에 대한 **case-disjoint 5-fold 재평가**로 §50 표 교체 (보고 시).
- 폐기 분기 최신화(§56.5)는 우선순위 낮음.
- 활성 스크립트 참조 정리 완료: `queue_v30_poolz.sh`·`evaluate_synthetic.py`·`test_musk.py` →
  archive 경로.

---

## 58. 2026-08-07 — v35 설계 확정: rare-instance 제거 + context/query 공통 chunk + 대형화

**상태**: PathoBench 보고 모델 v34-1536의 재학습(v35) 설계를 확정하고 제안서를 문서화했다. **코드 변경
없음, 학습·평가 미시작.** 이 세션 산출물: ① 모델 구조 정독 기반 아키텍처 분석, ② v35 설계 결정 3건, ③ 제안서
작성 (`docs/history/architecture_v35_tokenonly_chunked_query_proposal.md`).

### 1. 아키텍처 분석 (코드로 확인)

- **분류 헤드는 가변 길이 query sequence를 보지 않는다.** 모든 bag(context/query)은 aggregator
  (`StructuredEpisodePopulationAggregator`)에서 고정 token set(`global_summary` 1 + `slots` 12 +
  `tails` + covariance sketch)으로 압축되고, meta-classifier(`StructuredPopulationMetaClassifier`)는
  그 token을 class memory(8/class)와만 비교한다.
- **raw cell을 소비하는 유일한 branch는 `_rare_instance_logits`(~3438, batched ~3792)다.**
  v34는 `use_instance_attention_mil` off → query cell을 쓰는 곳은 이 branch뿐.
- query token 수가 실제로 중요한 곳은 ① per-cell compute/memory, ② mean/cov/slot 추정 품질(통계적),
  ③ rare/tail `topk(fraction)` 극단값(구조적) — ③이 rare branch 제거로 사라짐.
- **context `num_cells [1,8192]` 상한은 VRAM 가드일 뿐**(config 주석 확인) 구조적 제약이 아니다.
  실제 PathoBench slide는 context·query 모두 15k~30k → **train/eval 불일치는 query뿐 아니라
  context에도 동일하게 존재**했다.

### 2. v35 설계 결정 (3건)

1. **Rare-instance branch 제거**: `query_instances`/`query_cell_mask` 인자, `instance_input_norm`/
   `instance_input_projection`/`rare_evidence_head`/`rare_similarity_log_scale`/`tail_residual_logit`/
   `minimum_tail_residual_scale`/`rare_evidence_fractions` 제거, `_fuse_evidence`(~3527) 3→2증거 단순화.
   **aggregator의 `tails` token과 `aggregator_slot_rare_fraction`은 유지** (고정 per-bag 요약).
   ablation knob(`meta_enable_rare_evidence`, default off) 검토 중 — 회귀 시 원복용.
2. **Chunk-as-pseudo-bag (context/query 공통)**: 데이터 경계에서 ≤2048 결정적 분할 → pseudo-bag dense
   배치(**padding ≤2048로 VRAM 상한 제어**) → 원본 bag 단위 token 집계(count 가중 평균;
   `covariance_matrix`는 between-chunk 보정 `Σn_c(Σ_c+δ_cδ_cᵀ)/N`으로 **원본 공분산과 수학적으로 동일**) →
   meta-classifier 1회. 별도 query 전용 코드 불필요 → 오히려 설계 단순화.
3. **데이터 대형화**: `num_cells_log_uniform_power` 1.5~2.0, context `[1,30000]`, query `[3000,50000]`
   (**query ≥ context 유지 근거**: context는 40~80 bag 평균으로 per-bag 오차 상쇄, query는 단일 bag이
   결정적 증거), `num_bags [40,80]`, `episode_batch_size 2`.

### 3. 제안서 내용

- **구현 6단계**: rare 제거 → dataset(log-power + role별 범위 + query 위치 정렬) → chunk-as-pseudo-bag →
  model token 집계 → eval chunk 일관성 → `train_v35_*.yaml`(100ep, 2 GPU DDP).
- **검증**: ① 1-chunk direct == 집계 결과(불변성), ② 결정성, ③ rare 제거(aux 키 부재), ④ 50k query
  smoke(peak VRAM/step), ⑤ 공식 50-fold v34(fixed) vs v35.
- **오픈 문제**: ablation knob 여부, slot/tail 집계(평균 vs attention), LR 스케줄러(patience 10→5·
  cooldown 5→3 vs cosine), dataset query 위치와 `model_interface._sample_training_queries` mask 정렬.

### 4. 세션 진행 상황 (현재)

- model fix `5869535`(tanh margin, query-count-invariant) 커밋됨 — v35가 전제하는 chunk/multi-query
  동일성의 기반.
- bc_therapy/er_status 재실행 **bit-identical** (fold-mean 0.6741±0.1006, pooled 0.6721) — 재현성 확인.
- STK11 partial 25/50 fold (pooled 0.8300) — `predictions/pathobench_cptac_luad_STK11_mutation_v34_1536_official50_PARTIAL.pt`.
- GPU 0·1 여유, ICF 실행 프로세스 없음. **공식 50-fold 잔여 11개 미재개**.

### 5. 다음

- ① v35 오픈 문제 확정: ablation knob, slot/tail 집계, LR 스케줄러.
- ② 결정 1(rare 제거) 구현 + 32 tests 갱신.
- ③ v35 재학습 config 작성(100ep, 2 GPU 0·1 DDP) + 학습.
- ④ (병행 가능) 공식 50-fold 잔여 11개 재개 → 17개 최종 표 + SEAL 재비교.

---

## 59. 2026-08-07 — v35 제안서 비판적 재검토(rev.2) + 정확 스트리밍 구현 + v35 학습 시작

**상태**: §58 v35 설계를 **코드 대조로 비판적 재검토**해 제안서를 **rev.2로 전면 개정**했고
(결정 3건 중 2건 폐기 권고), 개정안의 1단계(정확 스트리밍 축약)와 5단계(데이터)를 구현·검증하고
**2-GPU 학습을 시작했다**. 테스트 32 → **41개 통과**.

### 1. §58 설계의 치명적 결함 (전부 코드로 확인)

§58.1의 구조 분석(고정 token, raw cell 소비자는 `_rare_instance_logits`뿐)은 **정확**했다. 문제는
거기서 도출한 결정들이다.

| §58의 주장 | 코드 확인 결과 |
|---|---|
| §58.2-2 "`_context_anchors`는 chunk에 안전" | **반대.** `_population_candidates`(977)는 bag 크기와 무관하게 **bag당 정확히 32개**(`context_samples_per_bag`) 후보를 낸다. 50k bag을 25 chunk로 쪼개면 그 bag 후보가 **800개(25×)**가 되고, anchor는 이 풀 위의 k-means 유사 refinement(1106-1116)+argmax farthest-point(1128-1136)라 **대형 bag이 anchor를 지배**한다. anchor는 에피소드 전역 → 모든 bag의 모든 token이 바뀐다 |
| §58.2-2 "`covariance_matrix`는 보정식으로 수학적으로 동일" | **구현 불가.** 보정식이 요구하는 chunk 평균 `μ_c`를 `_bag_view`가 `poolz_l2` 경로에서 **반환하지 않고 버린다**(826-830). representation dict에도 없다 |
| §58.2-2 "count 가중 평균" 일반 | `global_summary`는 1차 모멘트가 **아니라** bag 평균에 대한 **표준편차**(`global_summary: centered_spread`, `_bag_view` 789 `sqrt(mean(cd²)+1e-6)`). `covariance_sketch`는 `correlation` 모드라 rsqrt 정규화가 들어가고, `slot_covariance`는 `diagonal.log()`, `slot_metadata`는 `log(proportion)` — **전부 비선형**이라 chunk 평균은 편향. `slot_metadata`는 §58 집계표에 아예 누락 |
| §58.2-3 "query `[3000,50000]`" | **확정 목표 위반.** `sample_num_cells` docstring이 log-uniform 이유를 명시("Musk2 spans 1..1044 with a median of 12"). 하한 3000은 소형 bag 학습을 제거 → **Musk 0.95(2026-08-05 사용자 확정)** 와 충돌 |
| §58.2-3 context/query 분리 | **폐기된 B2b 재도입.** bag별 다른 cell 수 = per-bag cardinality = v33 arm C, 이미 **회귀 0.0412로 폐기**(§4·§42, "B2b 데이터 자체가 회귀 원인") |
| §58.3 "query 위치 정렬"은 검증 항목 | **설계 모순.** query 위치는 dataset이 정하지 않는다 — `_sample_training_queries`(`src/modules/model_interface.py` 440)가 **훈련 스텝에서 무작위**로 뽑고(535) 개수도 매번 다르며 클래스당 1개는 보호(541-556). dataset은 알 수 없다 → §58의 query≥context 비대칭 전체가 성립 불가 |
| 동기: train/eval 크기 불일치 | **직접 반증.** 본 세션 실측(EGFR 공식 50-fold, 동일 ckpt): context 2,000 tile cap **pooled 0.7695** vs full **0.7714** → **Δ −0.0019**. token은 전부 표본통계(일치추정량)라 bag이 커지면 기대값이 아니라 **분산만** 준다 |
| 방향성 | chunk token 평균 = region-level **mean pooling**. 우리 위치는 이미 MeanMIL 수준·ABMIL 미달(EGFR 0.771 vs 0.777/0.830, STK11 0.828 vs 0.873/0.908)인데, §58은 평균 쪽으로 가면서 **유일한 선택적 기제(rare branch)를 삭제**한다 |

### 2. rev.2 재설계 (제안서 전면 개정)

- **핵심 전환**: 완성 token을 평균하지 말고 **충분통계를 누적**한다. `assignment = softmax(·, dim=-1)`은
  **slot 축** softmax라 cell별 독립(1753) → slot 통계는 순수 합. `_population_candidates`는 cell축
  softmax 가중평균 → **online softmax로 정확** 축약. `topk`는 분산 top-k merge로 정확.
  **⇒ 근사할 필요가 없다.** 그러면 ① v34 ckpt가 그대로 유효, ② **rare branch를 지울 이유가 없어짐**,
  ③ 저장소 관행(MLA "byte-identical", §56 "forward 동치 diff 0")과 일치.
- **P0 게이트 신설(무료, 학습 0)**: ⓐ query 크기 스윕(2k→full 이득 **+0.005 미만이면 대형화 프로그램 폐기**),
  ⓑ `rare_logits=0` ablation으로 rare branch 기여도 측정 → 결정 1의 근거를 숫자로 종결.
- 데이터는 **하한 1 유지** + `power` tilt로 대형 노출만 추가(§5).

### 3. 구현 (전부 수치 중립 검증)

| 변경 | 내용 |
|---|---|
| `_context_pool_stats`(713) | 전체 context cell `torch.cat` 제거 → **bag별 float64 2-pass 스트리밍**. 기존 cat은 full-tile 에피소드에서 ~12 GB |
| `BaseModel.forward`(~4890) | `_bag_view`를 **query bag에만** 적용(기존: 전체 bag → 또 ~12 GB를 만들어 1장으로 인덱싱) |
| aggregator `forward` | **2-pass bag 스트리밍**(`stream_eval_bags`, 기본 on, list/eval 경로 한정). pass 1 = bag별 후보 32개 → anchors, pass 2 = 토큰 루프에서 bag별 view 재계산. `BAGPFN_DISABLE_BAG_STREAMING=1`로 A/B |
| `_select_anchors` 신설 | `_context_anchors`에서 anchor 선택부 분리 → 스트리밍이 **동일 후보 풀**을 넘겨 bit-identical |
| `num_cells_log_uniform_power` | `sample_num_cells`에 `fraction = U**(1/power)` 추가. `power=1.0`은 v34 draw와 동일 |
| **VRAM 가드 버그 수정** | `estimate_training_vram_bytes`가 **`episode_batch_size`를 무시**하고 있었다(4× batch가 공짜로 보임). batch 반영 + 배수를 실측 기반 재교정(21× → 7×; 실측 v34 6.0×·v35 6.5×) |
| 신규 config | `configs/train_v35_phase0_largebag_1536.yaml` |
| 신규 테스트 | `tests/test_stream_eval_bags.py`(7개) + vram 2개 → **41 tests, 48s** |

### 4. 검증 (실측)

- **스트리밍 == eager, 실데이터**: cptac_lscc/KEAP1 fold 1 (304 슬라이드, 전체 타일) AUROC **0.7804 동일**,
  peak VRAM **40,990 → 18,930 MiB (2.16× 절감)**. workers>2가 OOM이던 원인이 이것(2×40 GB=80 GB, 4×40=164 GB).
- **9개 representation key 전부** + 최종 logit이 `‖Δ‖∞ < 1e-4`, anchors는 **bit-identical**(테스트로 고정).
- **worst-case 학습 smoke**: 100 bags × 32768 cells = 3.28M cells → peak **122.4 GiB (69%)**, 1.9 s/step.
- 전체 config 81개(base_config 보유) 해석 성공.

### 5. ⚠️ 중요: 완료된 공식 50-fold 9개 수치가 **stale**

`5869535`(tanh margin) 이후 **1-query eval 결과가 바뀐다**(커밋 메시지대로 기존 margin_rms는 1-query에서
tanh(±1) sign-only로 붕괴했음). bc_therapy/er_status fold 1-3:

| | fold 1 | fold 2 | fold 3 |
|---|---:|---:|---:|
| 저장된 예측(§53 표의 근거) | 0.43913 | 0.78696 | 0.69130 |
| 현재 코드 | 0.4348 | 0.7565 | 0.7217 |

**내 변경이 원인이 아님을 확인**: HEAD(`0487b6d`, 내 변경 없음)를 별도 worktree에서 실행해 현재 코드와
**완전히 동일**(0.4348/0.7565/0.7217)했다. 따라서 §58.4의 "bc_therapy 재실행 bit-identical(pooled 0.6721)"은
**fix 적용 전 검증**이다. → **§53의 9개 task 표는 `5869535` 이후 코드로 재실행해야 한다** (|Δ| 최대 0.030).

### 6. v35 학습 시작 (진행 중)

- **단일 인자 arm**: 데이터만 변경, 아키텍처는 v34-1536 그대로(**rare branch 유지**, context/query 분리 없음,
  cardinality는 에피소드 단위 1회 = B2b 아님).
- 데이터: `num_cells [1, 32768]`, `num_cells_log_uniform_power 1.5`. 닫힌 형식: `P(n≤34)=19.8%`(Musk 밴드 보존),
  `P(n≥8192)=19.3%`, median 700, **E[n]=4487 vs v34 909 = 4.94×**.
- VRAM: peak는 **스텝당 총 cell 수**에 비례 → `episode_batch_size 1 × 100 bags × 32768 = 3.28M cells`로
  v34(`4 × 100 × 8192`)와 **동일 envelope**. 실측 peak **119.95 GiB (62.6%)**.
- 예산: **에피소드 매칭**(§42 교훈) — 1024 ep/epoch × 50 epoch = **51,200 episodes**, v34와 동일.
- 실행: `logs/20260807_203606/v35_largebag.out`, ckpt `checkpoints/20260807_203606/v35_largebag/`,
  PID 369656, 2×B200(GPU 0·1) DDP, 512 steps/epoch, **~92 s/epoch**(50 epoch 약 1.5시간).
- **초기 val_ce 추이**: ep0 `0.4547` → ep1 `0.4289` → ep2 `0.4143` → **ep3 `0.4096`**.
  **v34의 best(`0.4419` @ ep48)를 epoch 3에서 이미 하회**했다. 단 ⓐ val 분포도 같이 커졌으므로
  (val_dataset_kwargs가 같은 generator 설정을 상속) **v34와 val_ce를 직접 비교하는 것은 부당**하고,
  ⓑ 판정은 §59.7-1대로 **공식 50-fold AUROC**로 해야 한다. 유망한 초기 신호로만 기록한다.
- **주의**: `episode_batch_size`가 4→1이라 DDP 2랭크로도 유효 batch가 2(v34는 4)다. 강제된 2차 변경이므로
  결과 해석 시 명시할 것.

### 7. 다음

1. **v35 학습 완주** → 공식 50-fold 평가. 단 §5 때문에 **v34도 현재 코드로 재실행**해야 공정 비교가 된다.
2. **P0 게이트 실행**(무료): ⓐ query 크기 스윕, ⓑ `rare_logits=0` ablation. rev.2 §4의 판정 기준대로,
   ⓐ가 미달이면 대형화 노선 자체를 접는다.
3. 공식 50-fold **잔여 8개**(KEAP1·KRAS·TP53·BAP1·PBRM1·VHL·SMAD4·ucla_lung) — 스트리밍 덕분에
   workers를 2 → 8 이상으로 올릴 수 있다.
4. rev.2 §3의 **chunk 단위**(bag 내부) 스트리밍은 미구현 — 현재는 **bag 단위**까지만. 단일 bag이
   메모리를 넘길 때 필요하며 설계는 rev.2 §3.2-3.3에 확정되어 있다.
5. rev.2 §8: chunk token 위 **zero-init residual attention**(region-level ABMIL 유사물)이 ABMIL 격차에
   대한 올바른 베팅 — 단독 arm + 동일 게이트로만.
