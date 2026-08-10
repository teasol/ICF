# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-10` (§72~§81 — 세션 요약은 **§72**, 최신 변경은 **§81**)

**한 줄**: CV-2 손잡이는 **소진**됐고(§75·§76·§77), 소스에서 죽은 분기를 **실제로 삭제**했으며
(§73, −11,285줄), 학습을 **2.4배** 빠르게 했다(§74). 새 계보 B는 두 판본 모두 **기각**됐다 —
재설계가 합성 지표를 0.784→0.849로 올리고도 SEAL은 0.6619→0.6526으로 내렸다(§79-6).
**여전히 v41_K128(0.6940)이 최고다.**

**Status**: **두 계보 병행.**

* **계보 A = CV-only** (`src/models/baseline.py`, 학습 파라미터 **229개**).
  현행 최고 **v41_K128 = SEAL 10개 0.6940** (ABMIL 0.727에 −0.033).
  §73에서 죽은 5개 분기를 소스에서 삭제해 `baseline.py`가 5,685 → **2,224줄**이 됐다.
  ⚠️ **prune 이전 ckpt는 현재 트리로 로드 불가** — `8caa96c` 고정 worktree
  (`/NHNHOME/BASE/kimds/ICF_pre_prune`)를 쓸 것.
* **계보 B = Encoder+Ridge** (`src/models/set_transformer_ridge.py`, **5.01M개**).
  §69가 확인한 "label-free 사영은 전부 0.68 천장"을 우회하는 유일한 축 —
  **라벨을 보는(학습되는) 사영**. 첫 판본(v50~v52)은 내 설계 오류로 기술자가 256차원에
  묶였다(0.6047/0.6619). 재설계(v53/v54)로 세포 간 attention과 16,384차원 기술자를 얻어
  합성 val_auroc가 0.784 → **0.849**로 올랐으나 **SEAL은 0.6619 → 0.6526으로 내려갔다**
  (§79-6). **현재 형태로는 기각.** 문제는 용량이 아니라 일반화다.
* **CV-2는 더 파지 말 것** — margin activation(−0.017), subspace_rank(±0.001),
  head 구조(−0.0003) 셋 다 10개 평균을 못 움직였다. 병목이 아니다.
* **판정은 SEAL 10개 macro 평균만** (§71-4). 합성 val_ce·val_AUROC는 신뢰하지 않는다.
* **GPU 정책**: 0·1을 우선 사용, 다른 GPU는 사용자 허락 후.

현행 아키텍처 명세는 [`current_architecture.md`](current_architecture.md),
실험 절차·결과표·금지사항은 [`current_experiments.md`](current_experiments.md).

**지금 돌아가는 것 (2026-08-10)**

**돌고 있는 작업 없음.** v53/v54 학습·채점 모두 완료(§79-6).

결과 재확인:
```bash
for tag in v53_enc v54_enc; do
  printf "%-10s " $tag
  grep -hoP 'fold-mean AUROC: \K[0-9.]+' logs/official50/*_${tag}.log \
    | awk '{s+=$1;k++} END{printf "%.4f (%d개)\n", s/k, k}'
done
```

---

> **사용자 결정 (2026-08-05, 확정)**:
> 1. **v30 S2가 정식 확정 baseline 유지.** v31 CCTS/CCER-v2는 정식 baseline으로 승격/채택하지 않음 (실험 후보 기록만 남김).
> 2. **ICI는 손대지 않습니다.** (잠금 유지)
> 3. **Musk 목표는 0.95 유지.**

**Read first if you are picking this up**: **§72 (이번 세션 요약 — 어디로 갈지 §80)**, **§79 (계보 B 재설계 — 첫 판본이 내 설계 오류였다)**, **§73 (소스 prune — prune 이전 ckpt는 고정 worktree로만 채점 가능)**, **§74 (학습이 평가 경로를 타고 있었다 + 범인이 아니었던 것들)**, **§76·§77 (CV-2 손잡이 소진 — 더 파지 말 것)**, **§71 (판정 기준 = SEAL 10개 macro 평균)**, **§69 (label-free 사영 8종 전부 천장 / 합성 지표 불신)**, **§68 (CV-only가 baseline이 된 근거)**, **§67 (clipping 켜지 말 것)**, **§66 (CV-1 제거 불가)**.

> [!IMPORTANT]
> **방법론 경고 3건 — 다음 arm 설계 전에 읽을 것**:
> 0. **합성 val 지표(val_ce·val_AUROC)로 arm을 고르지 말 것 (§69-6).** CV-only의 합성 val AUROC는
>    ep0 0.8885 = ep49 0.8882로 평평한데 er_status는 +0.037 오른다. **판정은 er_status 50-fold로만**
>    (캐싱으로 45초). 단일 측정의 요동이 ±0.05이므로 seed 반복도 필수다.
> 1. **val_ce로 arm을 고르지 말 것.** v37 쌍은 val_ce가 확실히 좋았으나(0.3354 vs 0.3402) 50-fold는
>    **−0.0068로 나빴다**(CI가 0 제외). 200 epoch은 합성 생성기에 과적합한다.
> 2. **학습 길이가 다른 arm 간 비교는 그 자체로 교란이다** (§42-43 arm C 교훈의 재확인).
>    control은 항상 같은 epoch 수로 새로 학습할 것.

**열린 과제 (CV-only 노선, 우선순위 순)**:
① **`subspace_rank` 2·4 판정** — 진행 중, SEAL 10개 채점 자동 대기.
② **learnable 사영** — label-free 축 8개가 전부 0.68±0.03 천장이므로 **라벨이 남은 유일한
   정보원**이다. P는 1536×K(98K~197K)로 이 모델에서 가장 큰 잠재 파라미터인데 완전히 고정돼
   있다. ⚠️ CV-1이 closed-form이라 gradient가 ridge solve를 통과해야 하므로 **CV-2 쪽부터**
   붙이는 것이 안전하다(§66 ridge 제거 시 gradient 발산 전력).
③ **v40_cv_only / v38_control의 SEAL 10개 채점** — §70의 "대역폭+CV-2 = +0.0271"이 er_status
   기준이라 10개 기준의 실제 크기를 모른다. 각 20분.
④ **K=256** — 차원 유효가 §71-5로 확인됐으므로 재검토 가치(VRAM 22%로 여유). ridge-only
   진단상 K128→256은 +0.003이라 기대는 낮다.
⑤ **seed 반복** — 지금까지 arm당 1 seed. 요동이 ±0.02~0.05다.
⑥ **task별 편차 원인 규명** — 같은 TP53이 brca +0.018 / luad −0.066. ccrcc VHL은 0.4503으로
   랜덤 이하. 코호트 크기(112 vs 324)나 조직 특성으로 추정되나 미규명.
⑦ CV-2의 거리 평균 연산(`.square().mean(dim=-1)`) — rank를 올려도 MLP 입력이 스칼라 4개로
   고정되는 병목. ①이 무변화로 나오면 여기가 다음 손잡이다.

**해결·폐기**: 6-분기 아키텍처 전체(§68), v36 Q1·v37(§65), ridge ablation 계열(§66·§67),
G-2 제거 확정(§68에서 분기 통째 제거로 해소), E>1 노선(§68-5), label-free 사영 축 8개(§69).
상세 기록은 [`history.md`](history.md).

**Branches**: `main` = v30 확정 baseline + 미채택 v31 CCER-v2 재현 코드. 참고용 branch/tag 구조는
[`history.md`](history.md).
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
[`history.md`](history.md)로 이관.

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
- **폐기된 계획**: 4종 1,000-episode pool-400 paired 비교 평가, T5-A(typed bag-preserving branch)/T5-B/T5-C — 필요 시 향후 다시 꺼낼 수 있도록 [`history.md`](history.md)에 보존.
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

> 🗄️ 본문 전체는 [`docs/history.md`](history.md)로 이동했습니다 (2026-07-31).

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

**핵심**: 위 세 ICI 구성의 **95% 신뢰구간이 전부 0.5(무작위)를 포함하고 서로 거의 완전히 겹칩니다.** paired bootstrap 승률도 0.52~0.55로 동전 던지기 수준입니다. 즉 n=87 코호트에서 이 차이들은 **검출 불가능한 노이즈**입니다. 상세 근거: [`history.md`](history.md) §4-⑧.

---

## 4. v22 결정: retrieval 완전 제거 (2026-07-29)

> 아카이브됨 (2026-08-02): v22는 폐기된 구버전이라 이 결정 기록은 역사적.
> 제거 근거(3대 가설)·제거 범위·v20 롤백 불가 사유 원문 전체:
> [`docs/history.md`](history.md) §4.

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

전문은 [`history.md`](history.md)로 이관했습니다 (T3-3 Hard 기준선, 최종 후보 동결
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
- 브랜치/버전 정책: [`history.md`](history.md)

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
> [`docs/history.md`](history.md) §9.

---

## 10. 2026-08-01 세션 핸드오프 — v24 확정, 평가 계획 폐기

> 아카이브됨 (2026-08-02): v24 확정(사용자 결정, train CE 순위) + 4종 paired 비교
> 폐기 기록 — v24 확정 내용은 §3 "최종 결정"에 보존. 원문 전체:
> [`docs/history.md`](history.md) §10.

---

## 11. 2026-08-02 세션 핸드오프 — v25 Medium 평가 완료(사실상 동률), Easy tier 실험 진행 중

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 12-14. 2026-08-02 세션 — 폴더/문서·config/src·scripts·tests 정리 3단계

> 아카이브됨 (2026-08-02, 핸드오프 정리): checkpoint/log/prediction purge(53GB→3.3GB),
> 구버전 문서·config·스크립트 삭제, src/scripts/tests 참조 무결성 점검 기록. 전문:
> [`history.md`](history.md) §12-14.
>
> **하나만 아직 열려 있음**: §13의 config 삭제가 `test_d_stages_differ_only_in_selected_nuisance`를
> 깨뜨림 (`configs/trainer/learnability_d20.yaml` 삭제, §16에서 발견·미조치) — 상세는
> archive.md §13 경고 참고.

---

## 15. 2026-08-02 세션 마무리 — 정리 3단계 + v25 폐기 확정 + 브랜치 정리

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 16. 2026-08-02 세션 (이어짐) — v26/v27/v29 설계안 검토, 학습 없는 게이트 3종,
## CLS-token pooling(v26) 구현·학습 시작, 제안서 archive

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 17. 2026-08-03 세션 — v26 학습 완료 + CLS attention 진단 프로브 (24-CLS 제안 사전검정)

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 18. 2026-08-03 — E7 재검정: 지도 component-selection 상한 재확인 (Path B 관문)

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 19. 2026-08-03 — 정규화 천장 프로브: 고정 정규화가 천장을 제한하는가 (사용자 가설 검증)

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 20. 2026-08-03 — v24 no-L2 ablation: per-cell L2 정규화 제거 학습 (진행 중)

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 21. 2026-08-03 — Zero-shot Musk (Musk2) MIL 벤치마크 테스트

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 22. 2026-08-03 — 전략 전환: 생성기 개선 (Musk-like easy 데이터) — 가설 판정 완료

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 23. 2026-08-03 — Musk 0.95 로드맵: raw bag-stat token (mean/skew/kurt) 학습 중

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

## 24. 2026-08-03 — Phase 1 IA-MIL (Instance-Attention MIL) — 판정: 음성

> 아카이브됨 (2026-08-04, IA-MIL 폐기·문서 정리): 전체 원문은 [`docs/history.md`](history.md) 해당 섹션 참고.

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
- docs: 해결·폐기된 §11~§24 섹션을 `docs/history.md`로 이관, 스텁+링크만 남김.
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
3. **Musk 0.95**: 아키텍처 개선 proposal 작성 완료 — [`history.md`](history.md):
   P0(5-seed 앙상블, 즉시) → P2(bag-mean 보존 채널, 합성 선검증) → P1(166→512 읽기 브리지) → P3(단순
   인스턴스 풀링, 사전 게이트). 핵심 근거: `_bag_view` center+L2가 Musk 최고 신호(bag-mean, ridge 0.829)
   를 삭제(0.554) + zero-padding OOD 브리지.
   → **2026-08-04 §26에서 P1/P2 기각, P3 연기. 아래 §26 참조.**

---

## 26. 2026-08-04 — Musk 전이 재진단: P1/P2 기각 + v30(CFMT) 제안

**상태**: 진단 완료(학습 없음, 전부 재현 가능) / 제안 미구현 — 사용자 판단 대기
**문서**: [`history.md`](history.md)
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
- **§6(다음 작업 세션 Action Plan, 284줄)을 [`history.md`](history.md)로 이관**하고
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
[`history.md`](history.md) §3.1.

---

## 28. 2026-08-04 — v30 S1/S2 판정 과정 (B1 `poolz_l2`·B2 cardinality) — **아카이브됨, §29로 승격 완료**

v30 B1/B2 실험·판정 전체 기록(S1 `poolz`/`poolz_l2` 음성, S2 B2+B1 양성, paired bootstrap,
B1·B2 상호 필수 근거, 교차 분포 합성 무회귀)은 [`history.md`](history.md) §28로
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
[`history.md`](history.md)로 이동했다.

---

## 31. 2026-08-05 — v31 CCTS Musk 평가·진단 (아카이브됨)

CCTS Musk `0.8376`, 대형 bag `0.6032` 결과와 구현 결함 재분류 기록은
[`history.md`](history.md)로 이동했다.

---

## 32. 2026-08-05 — v31 CCER-Lite 구현·학습 (아카이브됨)

CCER-Lite 구현과 학습 기록은 contribution이 `~1.4e-4`로 사실상 비활성임을 확인한 뒤
[`history.md`](history.md)로 이동했다.

---

## 33. 2026-08-05 — v31 CCER-v2 아키텍처 구현 완료 (학습 미시작) (아카이브됨)

CCER-v2 아키텍처 구현·검증 기록. §38에서 CCER 계열 폐기 판정으로 대체. 본문은 [`history.md`](history.md)로 이동했다.

---

## 34. 2026-08-05 — v31 CCER-v2 20-epoch 학습 시작 (아카이브됨)

CCER-v2 20-epoch 학습 시작 기록. §38에서 폐기 판정. 본문은 [`history.md`](history.md)로 이동했다.

---

## 35. 2026-08-05 — CCER-v2 구현·검증·20 epoch 학습 완료 (아카이브됨)

CCER-v2 구현·20 epoch 학습 완료 기록. §38에서 폐기 판정. 본문은 [`history.md`](history.md)로 이동했다.

---

## 36. 2026-08-05 — v31 CCER-v2 Epoch 18 합성/Musk 평가 완료 (v30 Baseline 유지) (아카이브됨)

CCER-v2 epoch 18 합성/Musk 평가(v30 미달) 기록. §38에서 폐기 판정. 본문은 [`history.md`](history.md)로 이동했다.

---

## 37. 2026-08-05 — CCER-v2 결과 기반 v32 DR-CCER proposal 작성 (아카이브됨)

v32 DR-CCER proposal 작성 기록. §38에서 폐기 판정. 본문은 [`history.md`](history.md)로 이동했다.

---

## 38. 2026-08-05 — v32b DR-CCER: 비판적 검토 반영 개선안 + 구현 + Stage A 학습 시작

**상태**: v30 baseline 유지. v32 원안을 비판적으로 재검토한 **v32b 개선안**
([`history.md`](history.md))을 작성하고,
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
- **다음 Action**: ① CCER 계열 폐기 기록(`history.md`), ② Phase 1 "v30 on 6-task mix"
  재학습으로 데이터 효과 측정(any_positive_sparse가 v30에 무엇을 더하는지), ③ 소형 bag(n≤4, 0.80)과
  n>34(0.70)가 0.95 목표의 실질 병목 — 데이터/분포 쪽 레버 우선, ④ ICI 잠금 유지.

---

## 39. 2026-08-05 — v32b 완료 결과 평가 + v33 MR-BagPFN proposal

**상태**: 결과 평가와 proposal 작성만 완료. v33 구현·학습은 시작하지 않았고 v30 baseline은 유지한다.

- 현행 proposal: [`history.md`](history.md)
- v32/v32b proposal은 구현·평가 종료에 따라 `history.md` §10에 요약·보존했다.
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

## 41. 2026-08-05 — v33 Phase 0 구현: arm B(C) 데이터 컨트롤 + 학습 런치 — **아카이브됨**

B2b(per-bag cardinality) 구현·arm B/C config 런칭, 8× 에피소드 비대칭 주의 기록. 전문은
[`history.md`](history.md) §41.

## 42. 2026-08-06 — v33 Phase 0 arm B/C 학습 완료 + gate 평가 — **아카이브됨**

arm B(스파스 gate 미달 0.6747)·arm C(legacy 회귀 +0.0373) 50ep 완료 — **Phase 0 두
gate 모두 미달**. 전문은
[`history.md`](history.md) §42.

## 43. 2026-08-06 — arm C top-up: 8×A6000 DDP 전환 + NCCL P2P hang 수정 + 속도 기록 — **아카이브됨**

arm C top-up을 8×A6000 DDP로 재개. **NCCL P2P hang 진단/수정(`NCCL_P2P_DISABLE=1`,
런처 기본 적용)** + B200 vs A6000 8장 속도 비교(~4.3× 노드 총 처리량). 전문은
[`history.md`](history.md) §43.

---

## 44. 2026-08-06 — 패딩 배칭 (B2b `episode_batch_size>1`) + 병목 프로파일 — **아카이브됨**

ragged B2b 에피소드의 패딩 배칭 구현·검증(commit `568c5f8`, batch2에서 ~16 ep/s).
전문은
[`history.md`](history.md) §44.

## 45. 2026-08-06 — arm C top-up 중간 Musk zero-shot: 대형 bag 개선 + 소형 trade-off — **아카이브됨**

arm C 중간(ep64) Musk: **n>34 0.698→0.825 개선**, 소형(n≤4) 0.792→0.700 희생.
완주 checkpoint 재확인은 §48. 전문은
[`history.md`](history.md) §45.

## 46. 2026-08-06 — PathoBench zero-shot 평가: per-task PCA 전처리 + 결과 — **아카이브됨**

per-task PCA(1536→512) 캐시 파이프라인 구축 + sample-context 17개 task(대부분
0.5~0.68) + all-context 5개 task(개선). **§51 정정: 로컬 ccrcc CSV는 bc_therapy
복사본 오류**. 프로토콜은 이후 공식 50-fold(§52/§53)로 대체. 전문은
[`history.md`](history.md) §46.

## 47. 2026-08-06 — 새 기준 checkpoint(e125) 재평가 + 타일 수 제한 실험 — **아카이브됨**

`--context-mode all` 기본화 + `--max-tiles`/`--trials` 추가. e125(0.5142)를 새 기준으로
채택(val_ce 개선이 test로 대체로 전이), 타일 제한 스윕은 **task 의존**(LUAD는 제한이
개선). 전문은
[`history.md`](history.md) §47.

---

## 48. 2026-08-06 — arm C top-up 완주(150ep, best e125) + v33 Phase 0 평가 확정 + PathoBench v30 비교 — **아카이브됨**

arm C top-up 150ep 완주(e125). **legacy 회귀 gate 여전히 미달(+0.0412) — 과소학습
편향 가설 기각, B2b 데이터 자체가 회귀 원인. Musk n>34 개선(0.698→0.849) 유지,
PathoBench는 v30 우위(5-task 평균 +0.039). → v30 baseline 유지, arm C 미채택.**
전문은
[`history.md`](history.md) §48.

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

두 절 모두 종료된 정리 작업이라 전문을 [`history.md`](history.md)로 이관했다.
요약: §54 = 구버전 문서/config/스크립트 아카이빙 + v34 태그, §55 = AST 정적 분석으로 미사용
함수 제거. 열린 과제 없음.

---

## 56. 2026-08-07 — config 시스템 리팩터링(v34 base·default 참조·재아카이빙) + 공식 50-fold 재시작 — **아카이브됨**

> 아카이브됨 (2026-08-08, §64 정리): config 리팩터링은 완료됐고 지속되는 규칙은
> [`agent_handoff.md`](agent_handoff.md) §7(config 관리·자체 포함형 아카이빙·참조 검증)에 있다.
> 여기서 재시작한 공식 50-fold는 §57(case leakage)에 이어 §64(fp32 수치는 참고용)로 대체됐다.
> 전문: [`history.md`](history.md)

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
작성 (`docs/history.md`).

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

### 6. v35 학습 — 1차 [1,32768] OOM 크래시 → 2차 [1,16384] 재개 (진행 중)

- **단일 인자 arm**: 데이터만 변경, 아키텍처는 v34-1536 그대로(**rare branch 유지**, context/query 분리 없음,
  cardinality는 에피소드 단위 1회 = B2b 아님).
- **1차 런 `[1, 32768]` (20260807_203606) = CUDA OOM 크래시**: epoch 0부터 `expandable_segments: memory
  mapping failed with OOM` 324회(free ~10 MB), 21:24:08 `rank1 SIGABRT(exit -6)` → launcher exit 1.
  런치 시 VRAM 가드가 **162.72 GiB (85% of B200) -- caution**으로 경고했는데도 진행했고, 실제로 GPU를 거의
  가득 채워 죽었다(§59.6 당시 "실측 peak 119.95 GiB" 기록은 이 크래시로 반증됨). best val_ce
  **0.3574 @ ep6**(ep0 0.4547 → ep4 0.3883 → ep5 0.3643 → ep6 0.3574) — v34 best(0.4419)는 이미 하회.
- **2차 런 `[1, 16384]` (20260807_224559, 사용자 결정)**: 상한을 절반으로 축소. 닫힌 형식
  `P(n≤34)=21.9%`(Musk 밴드 보존), `P(n≥8192)=10.5%`, median 452, **E[n]=2394 vs v34 909 = 2.63×**.
  VRAM 가드 **82.19 GiB (42.9%) -- OK**, 실측 peak **54.72 GiB (28.6%)**. `last.ckpt`(ep6)에서 resume,
  **Epoch 7부터 연속**(8.6 it/s, 512 steps/epoch).
- 예산: **에피소드 매칭**(§42 교훈) — 1024 ep/epoch × 50 epoch = **51,200 episodes**, v34와 동일.
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

---

## 60. 2026-08-08 — v35-16384 50ep 완주 + 메모리/val plateau 진단 + v35 공식 50-fold 평가(EGFR·PIK3CA 완료) + SEAL 비교

**상태**: §59의 v35 데이터 단독 arm이 **50 epochs 정상 완주**(OOM 크래시 없이).
완주 전 메모리 급증("누수?")과 val loss plateau 의심을 진단해 **둘 다 정상/설명 가능**으로
결론냈다. 완주 후 판정 게이트인 **공식 50-fold 평가를 진행 — LUAD EGFR·BRCA PIK3CA 2개 완료**,
SEAL(지도 ABMIL/MeanMIL)과 비교했다. 이번 세션에서 §41–§48(v33 arm C saga)을
`history.md`로 아카이브했다.

### 1. v35 2차 런 `[1,16384]` 완주 (commit `51b5093` 이후)

- 1차 `[1,32768]`이 CUDA OOM 크래시(epoch 0부터 324회 경고, 21:24 rank1 SIGABRT) → 사용자 결정으로
  `num_cells` 상한 32768 → **16384** 축소, `last.ckpt`(ep6)에서 재개(`logs/20260807_224559/`).
- **완주**: 50 epochs, `Trainer.fit stopped: max_epochs=50`. best val_ce **0.3469 @ ep48**
  (ep27 0.3545 → ep47 0.3485 → ep48 0.3469 → ep49 0.3477). v34-1536 best(0.4419)보다 낮지만
  val 분포가 다르므로 직접 비교 부당(§59.6 ⓐ).
- VRAM 가드: 82.19 GiB (42.9%) -- OK. 실측 학습 peak 54.72 GiB. smoke 실측 69.7 GiB(1.01M cells worst).

### 2. 메모리 진단 — 단조 누수가 아니라 `expandable_segments` + 극단 ragged shape

- 증상: 시작 시 54.72 GiB → epoch 9·23에서 **~178 GiB 스파이크**(OOM 경고, free ~6MB) → 스텝 사이 92 GiB로 하락.
- 판정: **Python/텐서 참조 누수 아님**. 코드에 스텝 간 누적 패턴 없음(register_buffer 전부 persistent=False).
  메모리가 에피소드 크기에 따라 출렁이고, 한 번 커진 세그먼트를 driver로 반환하지 않는
  **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`(런처 기본) 동작**이 원인 후보.
  - 근거: 같은 config를 기본 allocator로 돌린 smoke는 peak 54–70 GiB에 그침(확장 안 됨).
    실제 학습(expandable_segments ON)은 ~178 GB까지 확장 후 보유.
  - 1차 런(32768)도 같은 패턴으로 결국 SIGABRT. 2차(16384)는 worst-case 1.64M cells가 ~64 GiB
    실할당에 그쳐 크래시 없이 완주.
- **교훈/미해결**: `expandable_segments`가 이 workload(에피소드당 1~16k cells의 극단 변동)에서
  fragmentation을 악화시키는지, 아니면 단순 peak 초과인지 확정 검증은 못함(런 종료 전 검증 불가).
  추후 large-bag 훈련에서 peak를 다시 볼 때 `expandable_segments:False` A/B 또는 상한 추가 축소를 검토.

### 3. val loss plateau 진단 — 정상 수렴(v34와 동일), val 셋 노이즈 큼

- RUN2 val_ce: ep7 0.3665 → ep12 0.3574 → ep23 0.3560 → ep27 0.3545 → ep48 **0.3469**.
  plateau처럼 보이지만 **끝까지 천천히 개선**(~0.0005/epoch, ±0.01 요동).
- **v34-1536과 동일한 모양**: v34도 ep10 이후 30+ 에포크 평탄(ep10 0.4601 → ep48 0.4419, 개선 미미).
  → 이 아키텍처의 정상 수렴 특성.
- **과적합 아님**(train_loss도 0.427→0.403으로 평탄), **LR 부족 아님**(여전히 0.0005, 스케줄러 미감소).
- **val 셋 노이즈 큼**: `val_dataset_kwargs.episodes_per_epoch: 104` (§5: 104개는 CI 폭 0.074로 판정 불가).
  ep별 ±0.005~0.01 요동이 바로 이 노이즈.
- **판정 메트릭은 val_ce가 아님**: §59.7-1대로 **공식 50-fold AUROC**가 v35 arm의 유일한 판정 수단.

### 4. LUAD EGFR v35 공식 50-fold 평가 (완료)

- 실행: `scripts/run_official_folds_parallel.py` — ckpt `checkpoints/20260807_224559/v35_largebag/
  epoch=048-val_ce_loss=0.3469.ckpt`, config `configs/train_v34_phase0_largectx_1536.yaml`(동일 아키텍처),
  task `cptac_luad/EGFR_mutation`(k=all.tsv 50-fold), 4 workers / GPU 0·1(워커당 ~25 GiB, GPU당 2), 2305s.
- **결과**: pooled **0.7819** / macro(fold-mean) **0.7889 ± 0.0919**. 로그
  `logs/official50/luad_egfr_v35_official50.log`, 출력
  `predictions/pathobench_cptac_luad_EGFR_mutation_v35_official50.pt`.
- **v34 비교**: `pathobench_cptac_luad_EGFR_mutation_v34_1536_official50.pt`(pooled 0.7714,
  §59.1 "full" 실측과 일치) → v35 **+0.0105 pooled**. 단 §59.5로 v34 재실행이 엄밀.

### 5. BRCA PIK3CA v35 공식 50-fold 평가 (완료)

- 실행: 동일 ckpt/config, task `cptac_brca/PIK3CA_mutation`, **8 workers / GPU 0·1**(소형 코호트라 워커 증설),
  287s. 로그 `logs/official50/brca_pik3ca_v35_official50.log`, 출력
  `predictions/pathobench_cptac_brca_PIK3CA_mutation_v35_official50.pt`.
- **결과**: pooled **0.5668** / macro **0.5743 ± 0.1086**. v34 비교
  `pathobench_cptac_brca_PIK3CA_mutation_v34_1536_official50.pt`(pooled 0.5690) → **v35 ≈ v34**
  (Δ −0.002, 둘 다 랜덤 수준). SEAL에서도 ABMIL 0.595로 거의 랜덤인 어려운 task.

### 6. SEAL baseline 비교 (지도 ABMIL/MeanMIL, 공식 50-fold macro-AUC — §53 프로토콜·코호트 유의)

| task | v35 pooled | v35 macro | v34 pooled | SEAL ABMIL | SEAL MeanMIL | v35 vs ABMIL | v35 vs MeanMIL |
|---|---|---|---|---|---|---|---|
| LUAD EGFR | 0.7819 | 0.7889±0.092 | 0.7714 | 0.830±0.089 | 0.777±0.099 | **−0.041** | **+0.012** |
| BRCA PIK3CA | 0.5668 | 0.5743±0.109 | 0.5690 | 0.595±0.103 | 0.544±0.120 | **−0.021** | **+0.030** |

- **판독**: zero-shot(v35)이 두 task 모두 **지도 MeanMIL과 동급~우위**, **지도 ABMIL에는 근접**(−0.02~−0.04, std 겹침). §59.7-5의 ABMIL 격차가 그대로 확인됨(rev.2 §8 zero-init chunk-attention이 이 격차를 노림).
  EGFR은 v35가 v34보다 +0.01 우위(large-bag 노출이 이 task에선 소폭 긍정), PIK3CA는 v35 ≈ v34(둘 다 랜덤).
- **SEAL 최약 task**: CCRCC VHL(ABMIL 0.538) / BRCA PIK3CA(0.595) — 지도 모델도 못 푸는 task. v35로 VHL 등 약한 task를 추가 돌리는 것이 강건성 테스트가 될 수 있음(제안).

### 7. 다음

1. v35 arm 판정은 §59.7-1대로 50-fold AUROC — **잔여 15개 task** + v34 재실행(공정 비교, §59.5).
2. (제안) CCRCC VHL 등 SEAL 최약 task 추가 평가(강건성 테스트).
3. **P0 게이트**(§59.7-2, 무료): query 크기 스윕 + `rare_logits=0` ablation — +0.005 미달 시 대형화 노선 폐기.
4. 대형 bag 훈련의 `expandable_segments` A/B 검증은 다음 large-bag 런에서 (옵션).
5. val 신호 정확화가 필요하면 `val episodes 104 → 1000`(§5 권장).

---

## 61. 2026-08-08 — P0-b 게이트 통과 + rare branch 제거 (rev.2 step 5) — **아카이브됨**

`rare_logits=0` ablation이 |Δpooled| **0.0009 < 0.003**으로 게이트를 통과해 rare 분기를 제거했다
(`meta_enable_rare_evidence: false`, 코드 삭제가 아니라 강제 0 — ckpt 호환·가역). 이후 모든 arm이
rare-free이므로 **평가는 반드시 그 arm의 훈련 config로** 해야 한다. 전문: [`history.md`](history.md).

## 62. 2026-08-08 — v36 제안서 비판적 재검토 + P0-slots 무료 probe (Q1 확정 / Q2 보류)

**상태**: `docs/architecture_v36_region_chunk_attention_proposal.md`(zero-init region **chunk**
attention)를 코드로 검증하며 비판적으로 재검토한 결과 **핵심 전제 3건이 반증**됐고, 대안으로
**좌표 없는 slot 기반**으로 문제를 재정의했다. 이어서 **학습 0의 P0-slots probe**를 구현·실행해
3개 task × 4 config × 공식 50-fold를 측정했다. 결론: **Q1(40→1 압축 해제)은 +0.16으로 확정 진행,
Q2(num_slots 증설)는 task별 부호 불일치로 보류.**

### 1. v36 제안서(chunk-attention)의 반증된 전제

1. **합성 훈련 분포에 region 구조가 없다.** `synthetic_data.py:322` docstring이 명시한다 —
   "Dense cells are exchangeable within a bag". sequential chunk 15개는 같은 분포의 iid 표본이라
   region 간 차이가 전부 샘플링 노이즈다. 훈련 분포에서 region attention의 최적해는 **정확히 균등
   평균**(= zero-init 상태)이고, 실제 WSI에서는 분포 외 사용이 된다. §31 측정 4(합성에 신호가
   구조적으로 부재해 채널이 무시당해도 게이트 통과) · §23 raw-stat 음성과 동일한 실패 모드.
2. **"선택 기제가 없다"는 사실이 아니다.** `_instance_attention_mil_logits`(baseline.py:3980)가
   존재한다 — nonlinear task-adaptive relevance MLP + attention pooling + max-instance. §24에서
   기각됐으나 **§31 측정 6이 그 게이트를 무효로 선언**했다. 제안서는 이를 언급하지 않는다.
3. **region 수 실측이 제안서의 1/4이다.** feature h5 60장 표본: tiles median **6,942**
   (mean 10,206 / p75 15,312 / max 38,305) → chunk 2048에서 region **median 3.4개**,
   슬라이드의 **57%가 ≤4개**, **18%가 1개(헤드가 항등)**. 제안서의 "N/2048 ≈ 15"는 N≈30k 가정.
   또 h5 파일 순서는 lexicographic 정렬(60/60)이라 chunk는 "인접 region"이 아니라 수직 슬랩이다
   (첫 chunk의 bbox가 슬라이드 bbox의 median 14%).

기타: §6-2의 chunk 경계 불변성 테스트는 §3.1의 region 의미와 **모순**(작동하는 v36을 버그로 판정),
§5-1ⓑ region 셔플 ablation은 permutation-invariant 헤드에 대해 **공허**, §7의 "stage 0 선행이 기준선
오염을 막는다"는 논거는 stage 0이 수치 무변화라 **성립하지 않음**. 동기인 ABMIL 격차도
EGFR −0.041(t −2.27) / PIK3CA −0.021(t −0.98)로, fold가 같은 코호트의 반복 랜덤 분할이라
`sd/√50`은 낙관적 — **게이트("격차 절반 축소")가 격차 추정치의 SE(0.018)보다 작은 변화를 요구**한다.

**사용자 결정**: 좌표(coords)는 쓰지 않는다 — 위치를 모른다고 가정하고 접근한다.

### 2. 재정의 — 좌표 없는 "region"은 slot, 그리고 선택 기제는 **죽어 있다**

- slot anchor는 에피소드 단위로 뽑히고 **에피소드 내 모든 bag이 공유**하며, assignment가 slot축
  softmax라 **cell 순서에 불변**. 좌표·chunk 경계 하이퍼파라미터가 불필요하다.
- 진짜 병목: **bag 내부 구조 40 token**(global_summary 1 + slot 12×3 + tail 3)을 **라벨 정보가
  들어오기 전에** 고정·라벨 무관 선형사상 `_projected_bag_tokens`로 **1개 token으로 압축**한다.
  구조는 위치별 `Linear(1536→64)` 40개 + concat(2560) + exact mean residual(1536) →
  `Linear(4096→1536)`. mean pooling이 아니라 **위치별 병목 + concat**이다.
- 그 결과 `_population_memory_logits`(baseline.py:3509)의 routing softmax가 **길이 1 축에 걸린다**.
  v35 config로 실측: `population_slot_weights` shape **(2,1), 값 전부 1.0** →
  **selection 기제가 구현돼 있으나 무력화**. v35 config의 `routing_sparsity_weight: 0.0` /
  `routing_balance_weight: 0.0`(둘 다 off)도 같은 정황.
- 즉 MeanMIL↔ABMIL 차이를 좌표 없이 정확히 기술하면: **어떤 region 정의를 쓰든 원리적으로
  task 적응적 within-bag 선택이 불가능**하다.

### 3. P0-slots probe 구현 (학습 0)

- 신규 `scripts/probe_slot_headroom.py` / `scripts/summarize_slot_headroom.py`.
- **aggregator에는 `num_slots`에 의존하는 파라미터가 하나도 없다** (실측: 12 vs 24에서 29개 텐서
  shape 완전 동일). 전체 모델에서 shape 불일치는 **`meta_classifier.bag_token_projection.weight`
  단 1개**. → frozen ckpt로 임의의 slot 수에서 구조 token을 뽑을 수 있다.
- 방법: 폴드마다 aggregator **1회** 실행(context_mask로 context 지정) → bag별 구조 token →
  context bag으로 ridge 적합(λ는 context 내부 inner-CV, 표준화도 context 통계만) → query AUROC.
- **정확성 검증(통과)**: pool 통계·anchor가 context 전용이라 합동 패스와 쿼리별 패스가 같아야
  하는데, 실측 **bit-identical (‖Δ‖∞ = 0.000e+00, 3개 쿼리)**.
- **비용**: 폴드당 aggregator 1회 = 배포 eval(쿼리당 1회)의 **약 1/50**. EGFR config당 573s 단일 GPU
  vs 전체 모델 공식 50-fold 2,305s×4 worker(≈2.6 GPU-시간).
- ckpt `checkpoints/20260807_224559/v35_largebag/epoch=048-val_ce_loss=0.3469.ckpt`,
  config `configs/train_v34_phase0_largectx_1536.yaml`, GPU 0·1.
- 산출물(gitignore): `predictions/probe_slots_{luad_egfr,luad_stk11,brca_pik3ca}.pt`,
  로그 `logs/probe_slots/*.log`.
- ⚠️ `num_slots ≠ 12`에서는 `bag_token_projection`이 랜덤 초기화이므로 그 변형은
  **`projected_random`**(랜덤 사영 대조군)으로 따로 라벨링했다 — 배포 bag token이 아니다.
- ⚠️ slot encoder는 12 slot으로 학습된 가중치라 24/48 실행은 그 인코더에게 **분포 외**다.
  이 sweep은 재학습 모델 헤드룸의 **방향성 있는 하한**이지 증명이 아니다.

### 4. 결과 — Q1: 40→1 압축이 버리는 정보 (fold-paired, 공식 50-fold)

| task | `all@12` (40 token) | `projected@12` (배포 bag token) | paired Δ | 95% CI |
|---|---:|---:|---:|---|
| LUAD EGFR | **0.7486** ± 0.097 | 0.5889 ± 0.097 | **+0.1597** | [+0.1357, +0.1840] ✅ |
| LUAD STK11 | **0.8379** ± 0.073 | 0.6802 ± 0.103 | **+0.1577** | [+0.1313, +0.1841] ✅ |
| BRCA PIK3CA | 0.5039 ± 0.110 | 0.4978 ± 0.129 | +0.0061 | [−0.0249, +0.0357] |

- 신호가 있는 두 task에서 **+0.16**, 리포 게이트(+0.005~+0.01)를 자릿수 단위로 상회.
  PIK3CA는 전 구간이 랜덤(0.50)이라 버릴 정보가 없어 판정 불가 — 예상된 결과.
- 압축기의 **품질 문제가 아니다**: STK11에서 랜덤 초기화 압축(`projected_random@8`) 0.6496 vs
  학습된 압축 0.6802 — 둘 다 token 집합보다 0.15~0.19 낮다. 원인은 **40→1 압축 자체**.
- 참고: STK11의 ridge probe 0.8379는 v34 전체 모델의 공식 50-fold 0.828(rev.2 §2.10)보다 높다
  (ckpt가 v35라 엄밀한 대응은 아니나, token 집합에 신호가 이미 있다는 뜻).
- 해석 주의: `all`은 61,440차원, `projected`는 1,536차원 ridge라 용량 차이가 섞인다. projection이
  선형이라 40-token ridge가 projected-token ridge를 포함하므로, 정확한 진술은 **"고정 선형 압축이
  선형 판독기 기준 정보 보존적이지 않고, 그 손실이 0.15~0.19"**다.

### 5. 결과 — Q2: num_slots 증설은 **task별 부호가 갈린다**

| task | all@8 − all@12 | all@24 − all@12 | all@48 − all@12 |
|---|---|---|---|
| LUAD EGFR | −0.0035 [−0.0129, +0.0064] | −0.0081 [−0.0212, +0.0045] | **−0.0105** [−0.0231, +0.0015] |
| LUAD STK11 | +0.0034 [−0.0059, +0.0132] | **+0.0163** [+0.0067, +0.0261] ✅ | **+0.0204** [+0.0120, +0.0288] ✅ |
| BRCA PIK3CA | +0.0123 [−0.0067, +0.0321] | +0.0181 [−0.0024, +0.0392] | **+0.0370** [+0.0128, +0.0614] ✅ |

- STK11·PIK3CA는 오르고 **EGFR은 내려간다** → **게이트 미달, 보류**. PIK3CA만 봤다면 잘못된
  결론으로 갔을 것이므로 3 task 전체 실행이 옳았다.
- 차원 교란은 상당 부분 해소된다 — 차원 증가만으로 ridge가 유리하다면 EGFR도 올라야 하는데
  내려간다. slot 해상도 효과는 실재하되 **task 의존적**.
- 미해결: EGFR의 음수가 "해상도가 해롭다"인지 "12-slot 인코더에 분포 외라서 깨진다"인지
  이 실험은 **구분하지 못한다**. 재학습으로만 답이 나오므로 더더욱 Q1이 먼저다.

### 6. 결정 사항 (사용자)

1. **좌표 미사용** — 위치를 모른다고 가정. chunk-region 노선 폐기.
2. **num_slots 증설을 먼저 보자** → P0-slots probe로 검증한 결과 **부호 불일치로 보류**.
   (재학습 비용도 확인: ckpt 비호환 1개 텐서 + aggregator O(N×num_slots) + VRAM 재추정 필요.
   단 shape 불일치가 1개뿐이라 scratch가 아닌 **weight-only warm start**가 가능하다 —
   aggregator 29개 텐서를 승계하고 `bag_token_projection` 스택만 재초기화. 단 위치별 bottleneck은
   토큰 배치가 밀려 1:1 승계는 안 된다(global index 0만 동일).)
3. **zero-init gate는 쓰지 않는다** — 아예 변경한다. 근거:
   ⓐ population 분기의 모든 파라미터가 token 개수가 아니라 `token_dim`/`hidden_dim`으로만 크기가
   정해져 **이 변경은 shape 보존 = ckpt strict 로드 + 신규 파라미터 0개**인데, 게이트가 그 성질을
   깨고 유일한 신규 파라미터가 된다. ⓑ 이 리포의 zero-init 게이트 전력이 나쁘다 — v31은 v30과
   예측 상관 0.99928(사실상 미기여), rare는 floor 0.05를 강제하고도 §61에서 |Δ| 0.0009.
   Δ≈0이 나오면 "가설이 틀림"과 "게이트가 안 열림"을 **구분할 수 없다** → 게이트가 가설과 교란.
   ⓒ 어차피 재학습해 끝점을 비교하므로 초기 동일성은 보고하지 않는 값.
   → 대신 **학습되는 게이트가 아니라 config 플래그**로 가역성만 확보한다.
4. **공식 50-fold 재채점은 공짜가 아니다**(EGFR ≈2.6 GPU-시간) → 학습 전 진단은
   **1~2 fold routing entropy 확인**으로 축소한다.

### 7. 다음

1. **Q1 단독 arm 구현**: config 플래그 `meta_population_token_mode: projected | structured`
   (기본 `projected` = 현행). **반드시 두 경로** — `_population_memory_logits`(eval/ragged)와
   `_population_memory_logits_batched`(**훈련/dense**, 로직이 인라인 복제됨; 파일 주석이
   "drifting one copy ... is exactly how the cls token was first missed here"로 경고) + 두 경로
   동치 테스트.
2. **1~2 fold routing entropy·선택성 진단**: softmax가 비퇴화가 되는지, 40 token에 어떻게
   분포하는지. ⚠️ `routing_temperature: 0.5` + sparsity/balance 항이 둘 다 0이라 **한 token으로
   붕괴하면 40→1 병목을 다른 경로로 재현**하는 셈 — 붕괴 조짐이면 temperature/balance를 후속
   단독 arm으로.
3. 통과 시 **scratch 학습**(warm start 아님 — T=1에서 학습된 분기를 T=40 입력에 넣는 변화이고
   `_fuse_evidence` 스케일까지 흔들리며, 이제 초기 동일성 보호가 없다) + **episode-matched** 비교.
4. (별건, 판단 필요) **폴드 단위 representation 캐싱 eval**: pool 통계·anchor가 context 전용이고
   §62-3에서 bit-identical이 실측됐으므로, 캐시한 표현을 쿼리별로 잘라 meta-classifier를 쿼리당
   1회 호출하면 **bit-identical하게 ~50× 가속**된다(EGFR 폴드당 16,306 → 324 bag-인코딩).
   §53 stale 9개 + 잔여 8개 = 20~40 GPU-시간이 걸린 상황이라 상시 이득이 크다.
5. 후속 단독 arm 후보(한 번에 하나): ⓐ `routing_sparsity_weight`/`balance_weight` 복원,
   ⓑ `slot_importance`를 class_memory 조건부로(현재 가중치는 task 무관, relation만 task 의존 —
   ABMIL과의 진짜 차이가 여기 남아 있다), ⓒ num_slots(§62-5 재검토), ⓓ IA-MIL(§31 측정 6).

---

## 63. 2026-08-08 — current_architecture v34 개편 검토 + bf16-mixed 계약 실제 강제 — **아카이브됨**

`configs/trainer/default.yaml`이 precision을 설정하지 않아 v34/v35가 fp32로 조용히 학습됐던 것을
확인하고 bf16-mixed를 예외 없이 강제했다(`tests/test_precision_contract.py`). 계약 본문은
[`agent_handoff.md`](agent_handoff.md) §3.4에 있다. 전문: [`history.md`](history.md).

## 64. 2026-08-08 — 평가도 bf16-mixed 강제 + 폴드 단위 context 캐싱(bit-identical, 7.1×) + bc_therapy/er_status 기본 평가 확정

**상태**: 사용자 결정 2건을 반영했다 — ① **평가 경로도 bf16-mixed 강제**(과거 50-fold 수치는
전부 **참고용**으로 격하), ② **bc_therapy/er_status를 기본 평가 task로 확정**. 그 위에서
멀티프로세싱 한계를 실측해 **워커 증설이 무의미함**을 확인했고, §62-3에서 예고한
**폴드 단위 context 캐싱**을 구현해 bit-identical하게 **356s → 50s**를 달성했다.

### 1. 평가 bf16-mixed 강제 (사용자 결정)

- **발견된 3중 불일치**: 같은 ckpt가 스크립트마다 다른 정밀도로 채점되고 있었다 —
  `evaluate_synthetic.py`는 **bf16 autocast**, `test_pathobench.py`/`test_musk.py`는 **fp32**,
  `test.py`는 기본값 **`16-mixed`(fp16!)** — 마지막은 §3.4가 금지하는 바로 그 오버플로 경로다.
- **조치**: `src/utils/utils.py`에 단일 정의 `eval_autocast(device, precision)` +
  `add_eval_precision_argument(parser)` 추가. fp16은 **ValueError로 거부**하고 `32-true`만
  탈출구로 남긴다. 배선: `test_pathobench.py`(forward 2곳 + `evaluate_trial` 3개 호출부),
  `test_musk.py`, `probe_slot_headroom.py`, `run_official_folds_parallel.py`(워커 전달),
  `test.py`(기본값 `16-mixed` → `bf16-mixed`, choices 제한). CPU는 autocast를 건너뛴다
  (CPU bf16 matmul은 에뮬레이션이라 테스트만 느려지고 손실이 생김).
- ⚠️ **과거 수치 전부 참고용**: 2026-08-08 이전 공식 50-fold AUROC는 전부 fp32 산출물이다.
- **정밀도 효과 실측** (§59.5가 기록한 바로 그 er_status fold와 직접 대조):

  | fold | fp32 (§59.5, `5869535` 이후) | bf16 (신규) | Δ |
  |---|---|---|---|
  | 1 | 0.4348 | 0.5130 | **+0.078** |
  | 2 | 0.7565 | 0.7652 | +0.009 |
  | 3 | 0.7217 | 0.7609 | +0.039 |

  fold 단위로 최대 +0.08까지 이동한다 — "참고용 격하" 결정을 수치가 뒷받침한다.

### 2. bc_therapy/er_status = 기본 평가 task (사용자 결정)

- 코호트: **166 slides**(라벨 51/115), 50 folds, fold당 test 33 / context 133.
- 타일: 총 453,211, **median 2,672** / mean 2,730 / p90 4,165 / max 6,487.
  전체 feature가 fp32로 **2.6 GiB**뿐이라 LUAD(324 slides × ~7k tiles) 대비 훨씬 가볍다.

### 3. 멀티프로세싱 한계 — **워커를 늘려도 안 빨라진다** (실측)

| 실행 | 워커 | GPU | folds | 시간 | **GPU당 fold 처리율** | peak VRAM |
|---|---|---|---|---|---|---|
| 실현성 | 13 | 1 | 13 | 184s | **14.2 s/fold** | 58,775 MiB (32%) |
| 본 실행 | 25 | 2 | 50 | 356s | **14.2 s/fold** | 59,451 / 54,459 MiB |

- GPU당 처리율이 **소수점까지 동일** → 메모리는 32%만 쓰지만 **연산은 이미 포화**. 워커는
  time-slice할 뿐이다. **워커 증설은 무효**(이전 세션의 "50 워커도 가능" 조언은 철회).
- 두 점 분해(`S+1F=184`, `S+2F=356`): **fold당 172s**, 기동·로드 12s → 시간의 97%가 fold 자체.
- ⚠️ **러너는 26을 요청해도 25 워커**를 띄운다: `chunk = ceil(50/26) = 2` → 25 청크.
- ⚠️ **재개 캐시 위험**: 모든 워커가 `{tmp_dir}/{task}_official_folds.ckpt` **하나를 공유**하고
  이미 있는 fold를 건너뛴다. fp32 시절 캐시가 남아 있으면 **정밀도가 조용히 섞인다.**
  `/tmp/pathobench_official_workers/`에 EGFR(08-08 06:00)·Histologic_Grade·KEAP1(08-07) 잔존.
  **재실행 시 새 `--tmp-dir`을 쓰거나 캐시를 지울 것.** (근본 해결은 캐시 키에 precision 포함.)

### 4. 폴드 단위 context 캐싱 구현 (§62-3 예고분)

- **원리**: all-context fold에서 모든 쿼리가 같은 context를 보고, aggregator의 에피소드 상태
  (`_context_pool_stats`·`_context_anchors`)는 **context bag만으로** 결정된다. 따라서 폴드당
  aggregator를 **1회**만 돌리고 쿼리별로 슬라이스한 뒤 meta-classifier를 **쿼리당 1회** 호출한다
  (→ `--batch-queries`가 깨뜨린 `_covariance_relation_scores` 단일 쿼리 거동 유지).
- **정확성 가드 2개** (둘 다 만족해야 캐싱 사용, 아니면 기존 경로로 폴백):
  ⓐ `context_mode == "all"`, ⓑ **`context_limit is None`** — `--max-tiles`/`--context-max-tiles`가
  설정되면 공유 `generator`가 쿼리마다 전진해 **context 부표본이 매번 달라지므로** 캐싱이
  한 번 뽑은 draw를 고정해버린다. 공식 50-fold는 full-tile이라 해당 없음.
- CLI: 기본 **ON**, `--no-cache-context`로 A/B.
- **검증**: ⓐ 2-fold A/B 66 쿼리 `max |Δp| = 0.000e+00`, ⓑ 50-fold 1,650 쿼리를 25-worker 실행과
  대조해 `max |Δp| = 0.000e+00`, pooled `0.692497` 완전 일치. **bit-identical.**

| | 워커 | GPU | 시간 | GPU-초 |
|---|---|---|---|---|
| 기존 (멀티프로세싱) | 25 | 2 | 356s | 712 |
| **캐싱 (단일)** | **1** | **1** | **50s** | **50** |

  벽시계 **7.1×**, GPU 시간 **14.2×** 절감. 이론치 26.6×에 못 미치는 이유는 캐싱이 없애는 것이
  context 재인코딩뿐이고 **meta-classifier는 여전히 쿼리마다** 돌기 때문이다(이 평가는 v34 config라
  rare 분기가 켜져 있어 쿼리 raw cell도 소비).

### 5. 신규 기준선 — bc_therapy/er_status (v35 ckpt, bf16, 캐싱)

```
fold-mean(macro) AUROC: 0.6975 ± 0.0895    pooled AUROC: 0.6925    (50 folds, 1,650 queries)
```

| | macro | pooled |
|---|---|---|
| **v35 (bf16, 신규 기준선)** | **0.6975 ± 0.090** | **0.6925** |
| SEAL ABMIL (지도) | 0.717 ± 0.086 | — |
| SEAL MeanMIL (지도) | 0.712 ± 0.091 | — |
| §53 기록 (v34, fp32) | — | 0.672 **(참고용)** |

- ABMIL 대비 **−0.020**, MeanMIL 대비 **−0.015**. EGFR/PIK3CA에서 보였던 "MeanMIL 동급~우위"와
  달리 이 task는 **MeanMIL에도 소폭 미달**이다.
- 산출물: `predictions/pathobench_bc_therapy_er_status_v35_official50_bf16_cached.pt`
  (25-worker 판본 `..._bf16.pt`도 보존, 수치 동일). 로그 `logs/official50/er_status_*.log`.

### 6. 테스트 (45 → **51 tests**, 약 65초)

- `tests/test_precision_contract.py` +2: 평가 헬퍼 기본값이 bf16이고 fp16을 거부하는지,
  추론 스크립트들이 **공용 헬퍼를 쓰는지**(자체 precision 하드코딩 금지).
- **신규** `tests/test_context_cache_equivalence.py` (4): 합성 에피소드에서 ⓐ 합동 패스 == 쿼리별
  패스, ⓑ context 표현의 쿼리 무관성, ⓒ pool 통계가 query cell을 무시, ⓓ anchor가 query cell을
  무시. 캐싱을 무효화하는 변경(쿼리 누출)이 50-fold 실행이 아니라 **여기서** 깨지도록 고정.

### 7. 다음

1. **Q1 단독 arm** (§62-7): `meta_population_token_mode: projected | structured`를
   `_population_memory_logits`와 `_population_memory_logits_batched` **두 경로**에 구현 +
   동치 테스트 → 1~2 fold routing entropy 진단 → scratch·episode-matched 학습.
   진단·평가는 이제 er_status 50-fold가 **50초**면 되므로 반복 비용이 사라졌다.
2. 정밀도 교란(§63-7)은 그대로 유효 — 새 학습은 bf16, 비교 대상 v35 ckpt는 fp32 학습본이다.
3. (선택) 공식 50-fold 재실행이 필요한 나머지 task들: §53 표 9개 + 잔여 8개. 캐싱으로 task당
   비용이 ~1/14로 줄었으니 전체 재산출이 현실적이다. **실행 전 `/tmp/pathobench_official_workers/`
   fp32 캐시 정리 필수**(§64-3).

---

## 65. 2026-08-09 — v36 Q1 / v37 두 arm 평가 완료: **둘 다 게이트 미달**, 40→1 압축은 원인이 아니었다

**상태**: §62-4의 P0-slots probe(구조 token이 압축 token보다 EGFR **+0.1597** / STK11 **+0.1577**
더 많은 라벨 정보를 담음)를 근거로 만든 두 아키텍처 arm이 **모두 실패**했다. 학습은 2026-08-08에
끝났으나 평가가 기록되지 않은 채 세션이 끊겨 있었다(§64 이후 커밋 `5241cc2`·`ce54f07`이
current_status에 미반영). 이 절이 그 공백을 메운다.

### 1. 평가 프로토콜

bc_therapy/er_status 공식 50-fold, **bf16-mixed + 폴드 단위 context 캐싱**(§64), arm당 약 45초.
**각 arm을 자기 훈련 config로 채점**했다 — 네 arm 모두 rare-free라 `train_v34_phase0_largectx_1536.yaml`
로 채점하면 미학습 rare 분기가 주입된다. 러너는 `scripts/eval_v36_v37_arms.sh`.

### 2. 결과

| arm | epochs | best val_ce | macro AUROC | pooled |
|---|---|---|---|---|
| v36 q1_baseline (projected, control) | 50 | 0.3402 | **0.7007 ± 0.087** | 0.6953 |
| v36 q1_structured (Q1) | 50 | 0.3405 | 0.6983 ± 0.087 | 0.6937 |
| v37 baseline (projected, control) | **171/200** | **0.3354** | 0.6939 ± 0.086 | 0.6921 |
| v37 context_adaptive | 200 | 0.3372 | 0.6938 ± 0.084 | 0.6911 |
| (참고) v35, §64 기준선 | 50 | 0.3469 | 0.6975 ± 0.089 | 0.6925 |

fold-paired Δ (20k bootstrap, 50 folds):

| 비교 | Δmacro | 95% CI | 이긴 fold |
|---|---|---|---|
| **Q1 structured − projected control** | **−0.0024** | [−0.0058, +0.0006] | 18/50 |
| **context_adaptive − v37 control** | **−0.0001** | [−0.0040, +0.0039] | 24/50 |
| v37 control − v36 control | −0.0068 | [−0.0124, −0.0011] | 16/50 |

### 3. 판정

- **v36 Q1 기각.** population 분기에 40 token을 전부 통과시켜도 **−0.0024**로, +0.005 게이트에
  미달일 뿐 아니라 부호가 음수 쪽이다. §62-4 probe가 측정한 **+0.16은 "token 안에 정보가 있다"**는
  뜻이었지 **"학습된 모델이 그 정보로 라우팅할 수 있다"**는 뜻이 아니었다. §62-7의 routing 진단이
  이미 예고했다 — entropy가 uniform의 **99.0%**였고 `slot_importance`는 softmax가 길이 1 축에
  걸린 탓에 선택성 gradient를 받은 적이 없었다. 50 epoch으로는 그 선택성이 생기지 않았다.
- **v37 기각.** **−0.0001**, 24/50 — 완벽한 null이다. 압축 가중치를 에피소드 의존으로 만든 것이
  측정 가능한 변화를 낳지 못했다. ⚠️ 단, 이 arm은 **사용자 결정으로 label-free**라 §62-2 진단의
  **절반만** 답한다. **라벨 조건화는 미검정 레버로 남는다.**
- ⚠️ **v37 control은 171/200 epoch에서 크래시**했다 — `PermissionError: [Errno 13]`로
  `logs/v37_baseline/version_0/metrics.csv` 기록 실패(CSVLogger, 2026-08-09 00:46). 파일은
  `kimds:kimds` 644로 지금은 쓰기 가능하므로 일시적 NFS 문제로 보인다. best는 ep132이라
  ckpt 자체는 유효하지만 **쌍의 학습 길이가 171 vs 200으로 어긋나 있다.**

### 4. 부산물 — **val_ce와 50-fold AUROC가 어긋난다**

v37 쌍은 4× 긴 학습으로 val_ce를 확실히 개선했으나(0.3354 vs v36의 0.3402) 50-fold는 **오히려
나쁘다**(−0.0068, CI가 0 제외). **200 epoch은 합성 생성기에 과적합**한다. 결론 2개:
① **val_ce로 arm을 고르지 말 것**, ② 학습 길이가 다른 arm 간 비교는 그 자체로 교란이다.

네 arm 전부 **0.694–0.701의 0.007 밴드** 안에 있다. 지도학습 SEAL은 ABMIL 0.717 / MeanMIL 0.712.

**산출물**: `predictions/pathobench_bc_therapy_er_status_{v36,v37}_*_official50_bf16.pt`,
로그 `logs/official50/er_status_v3{6,7}_*.log`.

---

## 66. 2026-08-09 — ridge ablation (v38): **G-2 global ridge는 무기여 / P-2·CV-1은 제거 시 학습 붕괴**

**상태**: 사용자 가설 — "**closed-form ridge가 라벨 정보를 너무 많이 가져가 학습 분기가 gradient를
받지 못한다**". §65의 두 실패가 모두 "학습된 선택 기제를 살리려는" 시도였다는 공통점에서 나왔다.
ridge를 하나씩 제거하는 ablation으로 검정했다. **결과는 가설과 반대 방향이다.**

### 1. 구현 (`73cd3dd`)

세 closed-form ridge solve를 **독립적으로** 제거하는 플래그(기본 전부 true = 현행 동작):

| flag | site | 제거해도 남는 것 |
|---|---|---|
| `meta_enable_global_ridge` | G-2 global_shape | set/cross-attention residual (G-3) |
| `meta_enable_abundance_ridge` | P-2 population | `population_attention` (Q-5) |
| `meta_enable_covariance_ridge` | CV-1 covariance | CV-2 relation 분기는 무관 |

각 플래그는 **자기 ridge 항만** 0으로 만들고 그 분기의 학습 residual은 남긴다 — 분기 전체가 아니라
**ridge 하나를 격리**한다. dense/ragged **두 경로 전부**에 배선했고, global은 solve 자체를 건너뛴다.
신규 파라미터 0개, shape 보존 → ckpt strict 로드 양방향.

> [!WARNING]
> **ablation된 ridge 파라미터는 gradient를 받지 않아 init 상태로 남는다.** 그 ckpt는 **반드시 같은
> 플래그로 평가**해야 한다 — ridge를 다시 켜면 미학습 분기가 logits에 주입된다. rare-free와 정확히
> 같은 함정(§61)이며 `tests/test_ridge_ablation.py::test_ablated_ridge_parameters_get_no_gradient`가 고정한다.

테스트 **66 → 74** (217s). `tests/test_ridge_ablation.py` 8개 — 기본값 퇴화 고정, 플래그 명시 True가
no-op, 각 플래그가 logits를 실제로 이동, 해당 항이 정확히 0, global 제거 시 attention residual 생존,
**dense/ragged 동치(‖Δ‖∞<1e-4)**, ckpt strict 로드, ablation된 `ridge_projection`의 grad=None.

### 2. arm 설계 (`75f3f00`)

**backbone은 v37 context_adaptive로 고정**(사용자 결정 2026-08-09: "v37 위에서만"). 최초에는 v36
계보(projected) 위에 올렸다가 재구성했다. 4 arm 전부 50 epoch·rare-free·bf16·seed 42·devices 1이며
**ridge 플래그 하나만 다르다**. **50 epoch control을 새로 학습**했다 — 기존 v37 ckpt는 200 epoch이라
그대로 쓰면 ridge 플래그와 학습 길이가 교란되고(§42-43 arm C 교훈), v36 q1_baseline은 projected
backbone이라 대체 불가다. VRAM이 arm당 ~98 GB(183 GB)라 **2 wave × 2 arm**으로 돌렸다.

### 3. 결과 — er_status 50-fold (bf16 + 캐싱, arm당 ~45초)

| arm | 제거 | val_ce | macro AUROC | pooled | 학습 상태 |
|---|---|---|---|---|---|
| control | 없음 | 0.3411 | **0.6994 ± 0.087** | 0.6946 | 50ep 정상 |
| global | G-2 | 0.3417 | 0.6990 ± 0.087 | 0.6942 | 50ep 정상 |
| abundance | P-2 | 0.3593 | 0.6670 ± 0.097 | 0.6637 | **ep13 크래시** |
| covariance | CV-1 | 0.4765 | 0.6049 ± 0.103 | 0.6000 | **발산, best=ep0** |

fold-paired Δ vs control (20k bootstrap, 50 folds):

| arm | Δmacro | 95% CI | 이긴 fold |
|---|---|---|---|
| **global** | **−0.0004** | [−0.0043, +0.0034] | 22/50 |
| abundance | −0.0323 | [−0.0443, −0.0207] | 12/50 ** |
| covariance | −0.0945 | [−0.1177, −0.0716] | 6/50 ** |

### 4. 판정

- ✅ **G-2 global ridge는 무기여 — 이번 실험의 유일한 깨끗한 결론이다.** 통째로 삭제해도
  Δ **−0.0004**, CI가 0을 정확히 감싸고 fold 승패 22/50. **control과 arm 둘 다 50 epoch 정상
  완주라 교란이 없다.** 파라미터를 죽여도 성능이 그대로다.
- ⚠️ **P-2 / CV-1 수치는 공정 비교가 아니다 — 참고용이다.** abundance는 **13 epoch**, covariance는
  사실상 **0 epoch** 모델이다. AUROC 하락이 "ridge 없이는 성능이 안 나온다"인지 "학습이 안 끝났다"인지
  **이 숫자만으로 분리되지 않는다.**
- **하락의 원인은 정보량이 아니라 수치 안정성 쪽을 가리킨다.** abundance는
  `RuntimeError: Non-finite gradients at epoch=13, optimizer step=13806`으로 죽었고, 터진 파라미터가
  aggregator 전반이다(`slot_w_dq/dkv/uq/uk`, `center/spread_slot_encoder`, `slot_residual_logit`).
  covariance는 50 epoch을 다 돌았으나 top-3 ckpt가 전부 **epoch 0·1·5**이고 val_loss가 0.545에서
  평평하다(control 0.366). → closed-form ridge는 라벨 신호를 **선점**하는 게 아니라 학습 초기에
  **안정적 gradient를 공급하는 앵커**로 동작하고 있었고, 사라지자 attention 분기가 혼자 떠맡으며
  발산했다는 그림에 가깝다. **사용자 가설은 기각 방향이다.**
- **§65와 달리 val_ce와 AUROC의 순위가 일치했다.**

### 5. 이번 세션에서 드러난 운영 함정 2건

1. **launcher wrapper가 torchrun child보다 먼저 종료한다.** wrapper PID만 kill하면 **GPU가 계속
   잡혀 있다**(실측 153 GB 잔존). 프로세스 그룹(`kill -TERM -$pgid`)으로 죽여야 한다.
2. **`while pgrep -f "scripts/train.py"` 대기 루프는 자기 자신에 매칭된다** — 그 bash 프로세스의
   커맨드라인에 패턴 문자열이 들어 있어 **영원히 끝나지 않는다.** wave 2는 끝났는데 후속 eval이
   실행되지 않은 원인이 이것이다. `scripts/queue_v38_wave2.sh`처럼 **launcher 로그 + 프로세스 부재를
   함께** 확인하거나, 패턴을 자기 자신과 겹치지 않게 쓸 것.

### 6. 다음

1. **G-2 제거 확정 전 task 1~2개 추가 확인** — er_status 단일 task·단일 seed다. 확인되면 코드에서
   실제 삭제 가능(파라미터 감소).
2. **P-2 / CV-1 재판정**: gradient clipping 등으로 안정화 후 50 epoch 완주. ⚠️ 안정화 조치가
   control과의 **두 번째 차이**가 되므로 **control도 같은 조치로 재학습**해야 공정하다 (2 wave, ~3시간).
3. **v37 label 조건화** — §65-3이 남긴 미검정 레버.

---

## 67. 2026-08-09 — v39 수치 안정화: **역효과**(clipping이 −0.0317) + LR 가설 반증

**상태**: §66에서 P-2/CV-1 제거 arm이 붕괴한 것을 "수치 불안정"으로 보고 안정화 2종을 넣었는데,
**안정화가 멀쩡하던 baseline을 망가뜨렸다.**

### 1. 구현 (`7079680`)

`nonfinite_gradient_policy: raise`(기본, 현행) | `zero`. clipping만으로는 안 되는 이유:
가드가 `on_before_optimizer_step`에 있고 Lightning은 이 훅을 **clipping보다 먼저** 부르므로
raise 정책에서는 clipping이 실행될 기회조차 없다. 게다가 `clip_grad_norm_`은 NaN을 고치지
못한다 — non-finite 항 하나가 total norm을 오염시키면 clip 계수를 통해 **모든** gradient가
망가진다. 두 레버는 대체재가 아니라 보완재다. 79 tests.

### 2. 결과 — er_status 50-fold

| arm | 안정화 | val_ce | macro AUROC |
|---|---|---|---|
| v38_global (G-2 제거) | ✗ | 0.3417 | **0.6990** |
| v39_baseline (G-2 제거) | ✓ | 0.3560 (ep13) | 0.6673 |
| v39_no_abundance (G-2+P-2) | ✓ | 0.3538 | 0.6902 |
| v39_no_covariance (G-2+CV-1) | ✓ | 0.4888 (**ep2**) | 0.5817 |

**v39_baseline vs v38_global(플래그 동일, 안정화만 상이): fold-paired −0.0317
[−0.0450, −0.0183], 13/50.** 안정화가 필요 없던 arm에 넣어 −0.032를 잃었다.

### 3. LR 가설 반증 (사용자 제기 → 실측)

`lr-AdamW`와 `nonfinite_gradient_steps`를 epoch별로 대조:

| epoch | v39_baseline LR | 누적 non-finite | v38_global LR |
|---|---|---|---|
| 0–20 | 5.0e-4 (최대) | **0** | 5.0e-4 |
| 25 | 2.5e-4 (감소) | 406 | 5.0e-4 |
| 44 | 1.25e-4 | 1,992 (+695/epoch) | 5.0e-4 |

**LR이 가장 높은 구간에서 non-finite가 0이고, LR을 낮출수록 폭증한다** — 방향이 정반대다.
결정적 대조: v38_global은 **완전히 같은 5e-4를 50 epoch 내내 유지**하고도 non-finite 0건이다.
LR 감소는 원인이 아니라 **증상**이다(plateau 스케줄러가 정체를 보고 깎았고, 정체의 원인이
그 불안정이었다). 두 run의 유일한 차이는 clipping이다.

### 4. 판정

- **CV-1 제거는 안정화 유무와 무관하게 붕괴**한다(0.6049 → 0.5817, best가 각각 ep0/ep2).
  수치 불안정이 아니라 **학습 자체가 성립하지 않는다.** CV-1 제거 불가 확정.
- P-2는 두 번 다 손상된 조건에서만 측정돼 **여전히 미결**이었으나, §68에서 무의미해졌다
  (CV-only가 P 분기 전체를 제거하고도 동률이므로).
- ⚠️ **clipping을 기본으로 켜지 말 것.** 이 모델에서는 순손해다.

---

## 68. 2026-08-09 — 분기 기여도 진단 → **CV-only 성공**: 6개 분기 중 2개만 남겨도 동률

**상태**: 사용자 질문("왜 covariance ridge는 제거 불가인가, 왜 내 아키텍처만으로는 학습이
안 되는가")에서 출발해 분기별 기여도를 측정했고, 그 결과가 **모델의 5/6을 삭제할 수 있다**는
결론으로 이어졌다. 이번 세션 최대 성과다.

### 1. 분기 기여도 진단 (신규 `scripts/diagnose_branch_contributions.py`)

v38_control ckpt, 합성 200 에피소드, 각 분기의 logit 단독 AUROC:

| 분기 | 종류 | AUROC | 기여 std |
|---|---|---|---|
| **FINAL (모델 출력)** | — | **0.9199** | 2.398 |
| **CV-1 covariance ridge** | closed-form | **0.9052** | 0.586 |
| CV-2 covariance relation | learned | 0.8867 | 0.804 |
| P-2 abundance ridge | closed-form | 0.6254 | 0.356 |
| G global_shape | mixed | 0.5949 | 0.288 |
| **Q-5 population attention** | learned | **0.5000** | **0.0000** |
| R rare/tail | learned | 0.5196 | 0.0008 |

**Q-5는 상수를 뱉는다** — AUROC 정확히 0.5000, std 0.0000. §62-2는 "routing softmax가 길이 1
축에 걸려 무력"이라 진단했지만 실제로는 **분기 출력 자체가 상수**였다.
→ **v36 Q1(−0.0024)과 v37(−0.0001)이 왜 실패했는지 설명된다: 상수를 뱉는 모듈에 더 좋은
입력을 넣은 것이라 달라질 게 없었다.** 두 아키텍처 실험 모두 죽은 모듈을 고치고 있었다.

### 2. 융합이 오히려 해가 되는 경우

| arm | FINAL | 최고 단일 분기 | 융합 효과 |
|---|---|---|---|
| v38_control | 0.9199 | CV-1 0.9052 | +0.015 |
| v39_no_abundance | 0.9192 | CV-1 0.8960 | +0.023 |
| **v39_no_covariance** | **0.7929** | **CV-2 0.8706** | **−0.078** |

CV-1을 뺀 모델은 **자기 최고 분기보다 0.078 나쁘다.** AUROC 0.70짜리 population이 기여
std 1.25로 CV-2(0.945)보다 크게 실려 **약한 분기가 좋은 분기를 끌어내린다.**
(부수 확인: CV-2는 CV-1 없이도 0.8706까지 학습된다 — "CV-1이 CV-2의 발판" 가설은 반증.)

### 3. CV-only arm — **전 분기 모델과 동률**

`meta_covariance_only: true`로 `final = cov_res·CV-1 + cov_rel_res·CV-2`만 남긴다.
global_shape(게이트 없는 베이스 항), population, rare, fusion interaction을 전부 제거.

| arm | 남긴 분기 | val_ce | macro AUROC | pooled |
|---|---|---|---|---|
| v38_control | 전부 | 0.3411 | 0.6994 ± 0.087 | 0.6946 |
| **v40_cv_only** | **CV-1 + CV-2 뿐** | **0.3401** | **0.6989 ± 0.087** | **0.6956** |

**fold-paired −0.0005 [−0.0037, +0.0024], 26/50 — 완전한 무차이.** val_ce와 pooled는 오히려
CV-only가 미세하게 낫다.

### 4. 죽은 분기 연산 skip (`fb926f8`)

출력만 0으로 만드는 게 아니라 **계산 자체를 건너뛴다**:

| 경로 | 전 분기 | CV-only skip | |
|---|---|---|---|
| 훈련 `forward_episode_batch` | 16.91 ms | **2.85 ms** | 5.9× |
| 평가 ragged forward | 19.20 ms | 9.39 ms | 2.0× |
| peak VRAM (60bags×16384) | 50,527 MiB | **14,720 MiB** | 3.4× |
| epoch 시간 | 98s | **60s** | 1.65× |

건너뛰는 것: context pool 통계, 전 셀 poolz_l2 표준화, per-episode anchors(top-k),
slot assignment/MLA affinity/encoder, tails, metadata, class memories, population attention, rare.
**핵심 근거: `centered_delta`(`_bag_view`의 세 번째 반환값)는 pool 통계에 의존하지 않는다.**

> [!IMPORTANT]
> **죽은 key는 zeros가 아니라 부재다** (사용자 결정). `_validate_representation`이 CV-only
> 전용의 더 작은 계약을 강제하고(빠지거나 남으면 ValueError), 실수로 읽으면 KeyError로 즉시
> 터진다. zeros였다면 shape 검증이 조용히 통과하고 0이 살아있는 분기로 흘러들었을 것이다.
> **실제로 이 설계가 소비처 3곳을 잡았다**: `BaseModel.forward`의 auxiliary 조립부,
> `_losses_from_output`의 routing entropy(`population_slot_weights`), 그리고 E>1 dense 경로의
> 에피소드 축 슬라이스. routing 진단은 0을 넣지 않고 건너뛴다 — "분기가 안 돌았다"와
> "돌았는데 0"의 구분은 Q-5가 죽은 걸 발견한 바로 그 신호라 잃으면 안 된다.

⚠️ **첫 skip 구현은 틀렸고 테스트가 잡았다**: `_forward_dense`에 넘어오는 `instances`는 이미
pool 표준화된 값이고 `centered_delta`는 별도 인자로 전달되는데, `instances`에서 재계산해
dense/ragged가 **2.4e-2** 어긋났다. 그대로 갔으면 다른 모델을 조용히 학습했을 것이다.
`test_skip_matches_a_full_branch_model_with_the_same_weights`가 방지선이다.
**등가성 end-to-end 확인**: 같은 config·seed로 old/skip 두 구현을 나란히 학습해 val_ce가
epoch 0–4에서 소수 4자리까지 완전 일치(0.4924/0.4639/0.4348/0.4017/0.3771).

### 5. E=4는 접었다

VRAM 실측으로 CV-only는 E=4까지 가능하지만(~97 GiB), **skip끼리 비교하면 E=1 60s vs E=4 86s로
43% 느리다.** `parallel_cuda_generation`이 켜져 생성 병목이 풀릴 것이라는 가설은 반증됐다.
(⚠️ 중간에 E=4(skip)를 E=1(**old**)과 비교해 "12% 빠르다"고 잘못 보고했다가 사용자 지적으로
정정. 구현 차이를 E 차이로 읽은 오류.)
부수 수정: VRAM 가드가 CV-only를 몰라 E=4를 169%로 오판·차단 → `activation_layers=1`로 보정
(실측비 0.291 vs (1+1)/(1+6)=0.286). `tests/test_vram_guard.py` +2.

---

## 69. 2026-08-09 — covariance sketch 기저 진단: **label-free 축 8개 전부 무효**, 차원만 유효

**상태**: CV-only에서 성능을 만드는 것이 covariance sketch 하나이므로 그 구성을 정밀 진단했다.
사용자 아이디어 3건(다중 주파수 / 차원 ablation / learnable 사영)에서 출발.

### 1. sketch 구성 (baseline.py:687-705)

```
centered_delta (N×1536) --P--> (N×64) --> 64×64 공분산 --> 상관행렬 --> shrinkage 0.1
                                                              --> 상삼각 2080 --> CV-1 ridge
P[d,k] = QR( sin(a·d·k) + cos(b·(d+1)·k) ),  a=0.019, b=0.011 하드코딩, persistent=False
```
`d`=임베딩 채널(1..1536), `k`=사영 방향 번호 **이자 주파수 배수**(1..64), `a`,`b`=주파수 사다리 간격.
`a=b`면 삼각합성으로 단일 주파수로 퇴화한다(실측 최하 0.6180).
**`persistent=False`** = ckpt에 저장되지 않음 → 결정적 공식을 쓰는 **공학적** 이유(랜덤이면
98K float를 매 ckpt에 저장하거나 DDP 랭크 간 seed를 맞춰야 한다). 통계적 이유는 문서에 없다
(v19 커밋 `03e7923`에 주석 없이 도입).

### 2. 진단 방법

CV-1과 **동일한 class-balanced dual ridge**로 sketch만 바꿔 er_status 공식 50-fold 채점
(`scripts/diagnose_covariance_sketch.py` 외 2종). 학습 없음, 설정당 수 분.

### 3. label-free 축은 전부 무효 — 8개 축, 전부 0.68 ± 0.03

| 축 | 결과 |
|---|---|
| 랜덤 vs 결정적 사인 | 차이 없음 (보존율 0.0412 vs 0.0419, 둘 다 64/1536=0.0417) |
| **PCA (분산 15배 보존: 63% vs 4%)** | **차이 없음** (0.6806 vs 0.6801) |
| Sobol QMC (10 seed) | **−0.016** (0.6631 vs 가우시안 0.6795) |
| 앨리어싱 여부 | 차이 없음 (0.6864 vs 0.6848) |
| 사다리 간격 `a` (0.019~2.5) | 요동뿐 |
| 대역폭 균일 사용 (0.85π) | 요동뿐 (0.6666) |
| **위상만 변경(주파수 설계 고정)** | **0.6500~0.7535, 폭 0.10** ← 요동의 크기 |

**위상 seed가 결정적이다.** `a=0.0385` 하나로 고정하고 열별 위상만 바꿔도 0.10이 흔들린다 —
`a`를 두 자릿수 범위로 바꿨을 때보다 크다. 위상만 밀어도 부분공간 겹침이 0.385로 떨어지므로
(무관한 랜덤끼리는 0.175) 사실상 다른 실현이다. 한때 최고로 보였던 `(1.5,1.1)` 0.7632는
재현되지 않았고(0.6882), 이 요동 분포의 위쪽 꼬리였다.

⚠️ 진행 중 잘못된 중간 결론 3건을 실측으로 폐기: "앨리어싱이라 의사난수화"(나이퀴스트 오독,
`a·k=1.216 < π`라 앨리어싱 아님), "앨리어싱될수록 좋다", "QMC 등분포라 랜덤보다 낫다".

### 4. 차원은 유효하다 — **단, 대역폭을 고정해야 보인다** (사용자 제안)

기존 스윕은 `a` 고정이라 K를 키우면 대역폭(`a·K`)이 0.304→4.86 rad로 함께 변해 **차원 효과와
대역폭 효과가 교란**됐다(그래서 "64에서 포화"로 잘못 보였다). `a = 0.85π/K`로 대역폭을 고정하면:

| K | feats | 사다리 (6 seed) | 가우시안 (6 seed) |
|---|---|---|---|
| 16 | 136 | 0.6203 ± 0.042 | 0.6455 ± 0.075 |
| 32 | 528 | 0.6715 ± 0.026 | 0.6191 ± 0.030 |
| **64 (현행)** | 2,080 | 0.6824 ± 0.023 | 0.6760 ± 0.021 |
| 128 | 8,256 | **0.6979 ± 0.032** | 0.6829 ± 0.014 |
| 256 | 32,896 | **0.7009 ± 0.011** | 0.6951 ± 0.018 |

**두 가족 모두 단조 증가**(가족 무관 → 차원 자체의 효과), **std도 단조 감소**(0.042→0.011).
K=128→256은 feature 4배에 +0.003으로 수익 체감. **현행 K=64는 최적이 아니다.**

### 5. 비용은 병목이 아니다 (전제 정정)

full 1536² 공분산 실측: 60 bags×16384 cells에서 **2.81 ms / 0.53 GiB**. 연산이 아니라 **통계**가
제약이다 — 상삼각 118만 feature를 셀 2,672개로 추정해 슬라이드 133개로 ridge를 맞춘다.
사영은 압축이 아니라 **정칙화**다. split-half 재현성이 K=16 0.983 → 256 0.935로 하락하는 것이
성분당 표본잡음(`1/√N_cells`)의 증거이나, §4에서 보듯 그 잡음보다 신호 증가가 빠르다.

### 6. epoch 사다리 — **50 epoch은 필요하다** (합성 지표가 거짓말)

`configs/callbacks/save_all.yaml`(save_top_k: -1) 신설, LR 스케줄을 건드리지 않고 max_epochs만
12로 줄여 epoch 0–11이 50-epoch run과 **동일 LR 궤적**이 되게 했다.

| epoch | 0 | 3 | 7 | 11 | **49** |
|---|---|---|---|---|---|
| er_status macro | 0.6617 | 0.6557 | 0.6620 | 0.6702 | **0.6989** |

합성 val AUROC는 ep0 0.8885 = ep49 0.8882로 **완전히 평평한데** er_status는 **+0.037 오른다**.
⚠️ **합성 val_ce·val_AUROC로 arm을 고르지 말 것** — §65의 val_ce 불일치에 이은 세 번째 사례이고,
이번엔 합성 AUROC조차 못 믿는다는 뜻이다. **판정은 er_status 50-fold로만**(캐싱으로 45초).

### 7. 다음

1. **K=128 (a=0.85π/K) CV-only 학습 arm** — ridge만으로 +0.016~0.019, 학습을 얹으면 더 갈 여지.
   ⚠️ ckpt 비호환(P shape·triangle 변경), 비용 K².
   **CV-2 경로 확인 완료 (2026-08-09)**: `_projected_covariance_matrix(centered_delta, dimension=32)`의
   호출부 4곳 모두 `dimension`을 넘기지 않아 기본값 32가 쓰이고, `min(32, K)`이므로 CV-2는
   **P의 앞 32열만** 본다(문서의 "covariance_matrix는 64×64"는 **오류**, 실제 32×32).
   → **K를 128/256으로 키워도 CV-2는 32에 고정**되어 이득이 CV-1에만 간다. 반대로 K<32면
   CV-2도 함께 줄어든다. K 증설 arm에서는 `dimension`을 K에 연동할지가 별도 결정 사항이다.
2. **learnable 사영** — label-free 축 8개가 전부 같은 천장이므로 천장을 뚫는 유일한 정보원은 라벨.
3. seed 반복 필수: 단일 측정의 요동이 ±0.05다.

---

## 70. 2026-08-09 — v41: er_status 0.7303 (**+0.031**) — 이득의 정체는 차원이 아니라 대역폭·CV-2  ⚠️ **§71이 정정: 10개 task로 넓히면 SEAL 상회 주장은 성립하지 않는다**

**상태**: §69의 sketch 기하 손잡이 2개를 실제 학습 arm에 적용했다. **er_status 50-fold에서
0.6989 → 0.7303**으로 이 task에서는 SEAL ABMIL(0.717)을 앞섰다.
> [!WARNING]
> **§71이 이 절의 주장을 정정한다.** 아래 수치는 전부 **er_status 단일 task**다. SEAL 대상
> 10개 task로 넓히면 평균 0.6940 vs ABMIL 0.727로 **−0.033 밀리고 상회는 3/10뿐**이다.
> "지도학습 SEAL을 넘었다"는 **er_status에만 해당**하며 일반화되지 않는다.

### 1. 결과

| arm | 구성 | val_ce | macro AUROC | pooled |
|---|---|---|---|---|
| v38_control | 전 분기 | 0.3411 | 0.6994 ± 0.087 | 0.6946 |
| v40_cv_only | CV-only, K=64, `a=0.019` 고정, CV-2=32 | 0.3401 | 0.6989 ± 0.087 | 0.6956 |
| **v41_K64** | + 대역폭 정규화 + CV-2 연동 | 0.3455 | **0.7260 ± 0.088** | 0.7234 |
| **v41_K128** | + 차원 증설 | **0.3333** | **0.7303 ± 0.096** | **0.7289** |
| *(참고) SEAL ABMIL* | *지도학습* | — | *0.717 ± 0.086* | — |
| *(참고) SEAL MeanMIL* | *지도학습* | — | *0.712 ± 0.091* | — |

fold-paired (20k bootstrap, 50 folds):

| 비교 | Δmacro | 95% CI | 이긴 fold |
|---|---|---|---|
| **v41_K64 − v40_cv_only** | **+0.0271** | [+0.0108, +0.0437] | 31/50 ** |
| v41_K128 − v41_K64 | +0.0043 | [−0.0073, +0.0160] | 27/50 |
| **v41_K128 − v38_control** | **+0.0310** | [+0.0149, +0.0464] | 37/50 ** |

### 2. 이득은 차원이 아니라 대역폭·CV-2에서 나왔다

**K를 64로 그대로 두고** 두 손잡이만 바꾼 arm이 **+0.0271**(CI가 0에서 멀리)이고,
K를 배로 키운 추가 이득은 **+0.0043**으로 CI가 0을 포함한다 — §69에서 예고한 대로
위상 요동(±0.02~0.03) 규모에 묻힌다.

**K=64 arm을 함께 돌린 것이 결정적이었다.** 그게 없었으면 +0.031 전체를 "차원 효과"로
잘못 귀속했을 것이다. (ridge-only 진단이 K 64→128에 +0.016을 예측했으나 학습 arm에서는
+0.004에 그쳤다 — ridge-only 진단이 학습 arm의 이득을 과대평가한다는 첫 사례.)

### 3. ⚠️ 아직 분리되지 않은 것

v41_K64가 **두 가지를 동시에** 바꿨다:
- `a = 0.85π/K` 대역폭 정규화 (0.019 → 0.041724, `a·K` 1.216 → 2.6704 rad)
- CV-2의 `covariance_matrix_dim` 32 → 64 (K 연동)

ridge-only 진단에서는 대역폭 정규화만으로 K=64가 0.6824(현행 사다리 0.6614 대비 +0.021)
였으므로 **대역폭 쪽이 주된 기여로 추정**되나, CV-2가 두 배 넓어진 효과도 실재할 수 있다.
**손잡이를 하나씩 끄는 arm 2개로 분리해야 한다.**

### 4. 합성 지표 불신, 네 번째 사례

v41_K64는 val_ce가 **0.3455**로 v40_cv_only(0.3401)보다 **나쁜데** er_status는 **+0.027 좋다.**
(K128은 val_ce·AUROC가 같은 방향이라 일관되지 않다.) §69-6의 경고가 다시 확인됐다 —
**판정은 er_status 50-fold로만.**

### 5. 비용

| arm | epoch 시간 | peak VRAM | feats |
|---|---|---|---|
| K=64 | ~72s (14.29 it/s) | 41.7 GiB (22%) | 2,080 |
| K=128 | ~95s (10.79 it/s) | 41.8 GiB (22%) | 8,256 |

VRAM이 22%뿐이라 K=256도 여유가 크다(§69-4 ridge-only에서 0.7009로 최고였으나
K128→256 이득이 +0.003이라 기대는 낮다).

### 6. 다음

1. **원인 분리** — 대역폭 정규화만 / CV-2 연동만, arm 2개(각 ~1시간). 어느 손잡이가 진짜인지
   알아야 다음 설계가 선다.
2. **seed 반복** — K64/K128 각 2~3 seed. +0.027은 요동보다 크지만 여유가 크지 않다.
3. **learnable 사영**(§69-7) — label-free 축이 전부 천장이었으므로 남은 유일한 정보원은 라벨.
   이제 P가 config로 교체 가능해졌으니 착수 비용이 낮다.
4. **다른 task로 일반화 확인** — 지금까지 전부 er_status 단일 task다. SEAL 우위 주장을 하려면
   최소 2~3개 task가 필요하다.

---

## 71. 2026-08-09 — SEAL 10개 task 전면 평가: **일반화 실패**, er_status는 가장 유리한 task였다

**상태**: §70이 er_status 단일 task로 "SEAL 상회"를 주장했으므로, 사용자 지시로 **SEAL 대상
10개 task 전부**를 v41_K128로 채점했다. **주장은 성립하지 않는다.**

### 1. 대상 선정

`docs/seal_univ2_baseline_17tasks.csv`의 **`in_seal=yes` 10개**만이 SEAL과 **같은 코호트·같은
50-fold**로 직접 비교 가능하다. 나머지 7개는 SEAL에 대응 수치가 없다(CPTAC-LSCC 미평가,
PDA는 회귀 전용, ucla_lung 미사용, ccrcc PBRM1은 다른 코호트).
러너: `scripts/eval_seal_tasks.sh`. 2 GPU 분할, 9개에 약 20분.

### 2. 결과 — v41_K128

| task | v41_K128 | ABMIL | MeanMIL | Δ ABMIL | Δ Mean |
|---|---|---|---|---|---|
| bc_therapy er_status | 0.7303 | 0.717 | 0.712 | **+0.013** | +0.018 |
| bc_therapy grade | 0.7451 | 0.770 | 0.751 | −0.025 | −0.006 |
| bc_therapy her2 | 0.6792 | 0.663 | 0.684 | **+0.016** | −0.005 |
| cptac_brca PIK3CA | 0.5476 | 0.595 | 0.544 | −0.047 | +0.004 |
| cptac_brca TP53 | 0.8188 | 0.801 | 0.787 | **+0.018** | +0.032 |
| cptac_luad EGFR | 0.7642 | 0.830 | 0.777 | −0.066 | −0.013 |
| cptac_luad STK11 | 0.8891 | 0.908 | 0.873 | −0.019 | +0.016 |
| cptac_luad TP53 | 0.6846 | 0.751 | 0.735 | −0.066 | −0.050 |
| cptac_ccrcc BAP1 | 0.6312 | 0.693 | 0.720 | −0.062 | −0.089 |
| cptac_ccrcc VHL | 0.4503 | 0.538 | 0.542 | −0.088 | −0.092 |
| **평균 (n=10)** | **0.6940** | **0.727** | **0.713** | **−0.033** | **−0.018** |

**ABMIL 상회 3/10, MeanMIL 상회 4/10.**

### 3. 판정

- ⚠️ **"지도학습 SEAL을 넘었다"는 er_status 단일 task 현상이다.** 10개 평균으로는 ABMIL에
  −0.033, MeanMIL에 −0.018 밀린다. §70의 헤더·커밋 메시지 표현은 과했다.
- ⚠️ **er_status가 10개 중 가장 유리한 task였을 가능성이 높다.** 지금까지의 모든 arm 선택
  (K, 대역폭 정규화, CV-2 연동, CV-only 자체, ridge ablation, v36/v37 기각)이 **전부 er_status
  단일 기준**이었다. **er_status에 과적합된 설계일 위험이 실재한다.**
- **ccrcc VHL 0.4503은 랜덤 이하**다(SEAL도 0.538로 낮은 task이나 우리는 −0.088).
- **같은 유전자도 코호트에 따라 정반대**다: TP53이 brca +0.018 / luad −0.066.
  코호트 특성(슬라이드 112 vs 324)이 작용하는 것으로 보인다.
- **MeanMIL 대비(−0.018)가 ABMIL 대비(−0.033)보다 낫다.** ABMIL은 attention 선택을 하는데
  우리는 §68에서 Q-5(선택 기제)를 삭제했고 covariance는 전 세포 통계다. 그 차이로 보인다.

### 4. 운영 원칙 변경 (필수)

**판정 기준을 er_status 단일에서 SEAL 10개 macro 평균으로 바꾼다.** 지금까지의 결론이 전부
단일 task에서만 검증됐으므로, 10개 기준으로는 다르게 나올 수 있다. 비용은 2 GPU로 약 20분이라
감당 가능하다.

### 5. 다음

1. **v41_K64 10개 평가**(진행 중) — §70의 "K=128 우위(+0.0043)"와 "대역폭·CV-2 이득(+0.0271)"이
   10개 기준으로도 유지되는지. **이게 §70의 결론이 er_status 특수 현상인지 가리는 첫 시험이다.**
2. v40_cv_only(이전 baseline) 10개 평가 — 이득의 절대 크기 확인.
3. 이후 모든 arm은 10개 기준으로 판정.

---

## 72. 2026-08-09/10 — 세션 요약: CV-2 손잡이 소진, 소스 prune, 학습 2.4배 가속, 계보 B 재설계

이 세션은 크게 다섯 갈래다. 각 갈래를 §73~§80에 나눠 적는다.

| § | 내용 | 결론 |
|---|---|---|
| §73 | 소스 prune (−11,285줄) | 죽은 분기를 실제로 삭제, golden fixture로 검증 |
| §74 | 학습 속도 (74.2 → 31.3 ms/step) | 학습이 **평가용 ragged 경로**를 타고 있었다 |
| §75 | v42 rank 2/4 | 무효 (±0.001) |
| §76 | v43/v44 identity margin | **기각** (−0.017), tanh 유지 |
| §77 | v45 paired_head | 동률(−0.0003), 그러나 라벨 대칭성은 얻음 |
| §78 | ridge dual 검증 | dual이 옳다 (30배) |
| §79 | 계보 B 재설계 (v50~v54) | 첫 판본은 **내 설계 오류**, 재설계 후 궤적 반전 |

---

## 73. 2026-08-09 — config로 끄기만 했던 5개 분기를 소스에서 삭제 (−11,285줄)

### 73-1. 근거

v41_K128 ckpt에서 파라미터별 gradient를 직접 측정:

```
전체 43,198,660 / gradient를 받는 것 229 (0.0005%)
나머지 43,198,431 = CV-only가 호출조차 하지 않는 코드
가장 큰 셋(center/spread/rare_slot_encoder)만 21.3M
```

### 73-2. 삭제 목록

| 대상 | 규모 |
|---|---|
| `src/repro_backup_20260807/` (아무도 import 안 함) | 6,826줄 |
| `MeanAggregator`/`MeanResidual`/`ClassTokenPooling`/`SetCrossAttention`/`RidgeResidual` | 714줄 |
| 메타분류기 죽은 메서드 (global/population/rare/fusion/memory/slot-token) | 868줄 |
| 메타분류기 `__init__` | 631 → 156줄, 인자 45 → 6 |
| `forward`/`forward_batched` 전 분기 본체 | 435줄 |
| aggregator slot 파이프라인 | 776줄 |
| `BaseModel` 인자 27개 + `meta_covariance_only` 플래그 자체 | |

`baseline.py` **5,685 → 2,224줄**. CV-1이 쓰는 `_solve_ridge_system`은 모듈 함수
`solve_ridge_system`으로 살렸다.

### 73-3. 안전장치와 그 성과

`TestCovarianceOnly`가 등가성을 재던 상대(전 분기 모델)가 사라지므로, 삭제 **전에**
출력을 녹화해 fixture로 고정했다(`tests/fixtures/cvonly_golden.pt`, 59KB).

⚠️ 첫 판본은 전체 state_dict를 담아 **691MB**였다. CV-only가 도달 가능한 가중치
(이름에 `covariance` 포함)만 남겨 59KB로 줄였고, 부수적으로 **나머지를 복원하지 않아도
출력이 일치**한다는 사실 자체가 "CV-only는 그 43.2M을 읽지 않는다"의 직접 증거가 됐다.

**fixture가 실제로 일했다.** v44 한 에피소드가 어긋나 끝까지 추적한 결과:

```
covariance_sketch                    비트 단위 동일
covariance_matrix                    최대 1.2e-10 (absmax 8.5e-4의 ~1 ulp)
_bag_view/_covariance_sketch/
  _projected_covariance_matrix       두 트리에서 md5 일치 (바이트 동일)
같은 트리 3회 반복                    12자리까지 동일
```

수식은 그대로고, 파라미터 43M이 사라지며 텐서 주소·정렬이 바뀌어 oneDNN/cuBLAS가 다른
벡터화 커널을 고른 것이다. 그 1 ulp가 CV-2의 고유분해(거의 축퇴)를 지나며 증폭된다.

### 73-4. 부수 발견 — VRAM 가드 회귀

가드가 `activation_layers`를 `meta_covariance_only` **config 키**로 정했는데 그 키를
지웠으므로, 이후 모든 모델이 조용히 6층으로 추정됐다(CV-only를 3.4배 과대평가).
아키텍처에 대한 사실을 config 키에 둔 것이 잘못이었다 → 이제 **모델이
`vram_activation_layers`로 직접 선언**한다.

### 73-5. 진행 중 실험 보호

prune 이전 ckpt는 현재 트리로 strict 로드가 깨진다. `8caa96c`에 고정한 worktree를
`/NHNHOME/BASE/kimds/ICF_pre_prune`에 두고 v43/v44 채점에 썼다. **이 worktree는 유지할 것.**

---

## 74. 2026-08-10 — 학습이 평가용 ragged 경로를 타고 있었다 (74.2 → 31.3 ms/step)

### 74-1. 원인

`_episode_losses`가 `self.model(x, y, ...)` — **ragged forward**를 부른다. bag 길이가
제각각인 평가를 위해 bag마다 Python 루프를 도는 구현이다. `episode_batch_size: 1`이면
`x.ndim == 3`이라 `training_step`의 배치 분기(`ndim == 4`)가 발동하지 않아, 크기가 같은
학습 에피소드가 조용히 그 경로를 탔다.

동일 Lightning 루프, 128 step A/B:

```
ragged (이전)   9.5 s = 74.2 ms/step
dense  (변경 후) 4.0 s = 31.3 ms/step     2.4배
```

ragged의 74.2 ms가 실제 학습 73~79 s/epoch(1024 step)과 일치해 측정이 검증된다.
두 경로 출력 차이는 3.3e-07.

### 74-2. 범인이 **아니었던** 것들 (측정으로 배제)

| 가설 | 실측 |
|---|---|
| step 단위 로깅 | 이미 epoch 단위였다(`on_step=False`). 지표 계산 비용 2.36 ms(3%), 로거를 통째로 제거해도 76.99 ms로 동일 |
| CPU 비동기 생성 | 생성은 에피소드당 **35 GFLOP 연산**. GPU 3.2 ms vs CPU 2,579 ms(**805배**). 옮기면 매 step H2D(pinned 5.1 ms, `pin_memory` 자체 37 ms)가 새로 붙고, 최대 5.9 GB 꼬리 에피소드에 워커가 1분 넘게 묶인다 |
| 프리페치 깊이 | GPU 배경 스트림 프리페치는 **이미 있었다**. depth 1 = 3.9 s vs depth 3 = 4.8 s. dense 전환 후 생성(16.9 ms)이 모델(8.5 ms)보다 길어 생산자가 포화 |

`cuda_prefetch_depth`는 손잡이로 남겼으나 **기본값 1을 유지**한다.

### 74-3. step 구성 (최종)

에피soде 생성(GPU) 22% / 모델 28% / Lightning 오버헤드 50%.
에피소드 크기: 중앙값 193,040셀 × 1536 = **1.2 GB**, 평균 321,500, 최대 952,864.

---

## 75. 2026-08-10 — v42 subspace_rank 2/4: 무효

| arm | SEAL 10개 | Δ vs v41 |
|---|---|---|
| v41_K128 (rank 1) | 0.6940 | — |
| v42_rank2 | 0.6944 | +0.0004 |
| v42_rank4 | 0.6932 | −0.0008 |

`.square().mean(dim=-1)`이 rank 축을 평균내 없애 MLP 입력이 항상 스칼라 4개다.

⚠️ **정정**: "rank가 출력에 도달 못 한다"는 서술은 **틀렸다**. rank를 바꾸면 고르는
고유벡터가 달라져 출력은 변한다(rank 1 vs 3에서 0.178). 정확한 결함은 head가 **어느
차원이 신호를 갖고 있었는지 알 수 없다**는 것이다.

---

## 76. 2026-08-10 — v43/v44 identity margin: **기각**, tanh 유지

| arm | 온도 | SEAL 10개 |
|---|---|---|
| v41_K128 (tanh) | — | **0.6940** |
| v43_notanh | 150 → 34.0 | 0.6770 (−0.0170) |
| v44_lowT | 4 → 2.84 | 0.6763 (−0.0177) |

### 76-1. 최초 진단이 틀렸다 (철회)

"CV-only에서 gradient를 받는 파라미터가 3개뿐이고 tanh가 head를 죽였다"고 적었으나
**틀렸다**. tanh 판본도 init에서나 학습된 ep49 ckpt에서나 grad≠0 파라미터가 **229개**다.

실제 현상은 **에피소드 단위 포화**다(v41_K128 ep49, 60 에피소드):

```
|tanh(margin)| > 0.99    58.3%   국소 기울기 2e-2
|tanh(margin)| > 0.999   48.3%                2e-3
|tanh(margin)| > 0.9999  40.0%                2e-4
```

### 76-2. 온도 초기값은 성능과 무관했다

v43(T→34.0)과 v44(T→2.84)가 **10개 task 전부 셋째 자리까지 일치**한다(최대 차이 0.0040).
온도 초기값을 37배, 최종값을 12배 바꿔도 결과가 같다 → **CV-2의 출력 스케일은 성능에
영향이 없다**. §68의 "CV-1이 지배한다"와 일치한다.

그리고 tanh의 포화는 손해가 아니라 **극단값 억제**로 작동하고 있었다 — identity에서는
작고 잡음 큰 bag에서 logit이 **±713**까지 간다(같은 트리에서 CPU/GPU가 1.23 차이).

---

## 77. 2026-08-10 — v45 paired_head: 동률, 그러나 라벨 대칭성을 얻었다

| arm | SEAL 10개 |
|---|---|
| v41_K128 (learned_head, rank 1) | 0.6940 |
| v45_paired (paired_head, rank 4) | 0.6937 (−0.0003) |

3승 7패에 부호가 뒤섞인 잡음이다. v43/v44의 10/10 하락과는 질적으로 다르다.

### 77-1. 고친 결함 — CV-2가 라벨 대칭성을 깨고 있었다

```
CV-1 ridge          0.0e+00  (정확히 등변)
CV-2 learned_head   4.4e-02  (깨짐)
```

즉 **클래스 이름만 바꿔도 답이 달라졌다.** `test_label_swap_is_equivariant`가 못 잡은
이유는 그 테스트가 `covariance_relation`을 끈 모델을 만들기 때문이다.

`paired_head`는 `margin = mean_r [h(e0,e1,s) − h(e1,e0,s)]`로 **구성상** 반대칭이다.

### 77-2. 결론

구조적 결함 둘(차원별 정보 손실, 라벨 비대칭)을 고쳤는데 성능이 안 움직였다 → **그
결함들이 병목이 아니었다.** **CV-2 쪽 손잡이는 소진됐다** — margin activation, rank,
head 구조 셋 다 10개 평균을 못 움직였다.

---

## 78. 2026-08-10 — CV-1의 dual(kernel) ridge는 옳다

"instance N > 차원 d니 kernel이 불필요하지 않나"라는 질문에 대한 검증.

**전제가 어긋난다. ridge는 instance를 보지 않는다.** 설계행렬의 행은 **bag**이고 열은
sketch 특징이다. instance와 임베딩 차원은 이미 공분산으로 요약돼 사라진 뒤다.

| | 값 |
|---|---|
| bag 수 (학습) | 60~95 |
| bag 수 (er_status context) | ~133 |
| sketch 특징 F = K(K+1)/2 (K=128) | **8,256** |

실측: dual(gram 133×133) **0.378 ms** vs primal(gram 8,256×8,256) **9.977 ms** — **26배**.
bag 수 ≪ 특징 수이므로 dual이 유리한 방향이고, 질문의 논리를 그대로 적용해도 dual이 답이다.

---

## 79. 2026-08-10 — 계보 B (Encoder+Ridge): 첫 판본은 설계 오류, 재설계 후 궤적 반전

### 79-1. 왜 만들었나

§69가 label-free 사영 8종을 전부 시험해 0.68 천장을 확인했다. **라벨을 보는 사영**이
유일한 미시험 축이다. `src/models/set_transformer_ridge.py` — baseline과 공유하는 것은
ridge 솔버 하나뿐이다.

### 79-2. ⚠️ 첫 판본(v50~v52)의 설계 오류

세포끼리 attend하지 않는 **inducing-point** 인코더를 썼다. 근거는 "셀-셀 attention은
에피소드당 2.7e10 쌍이라 불가"였는데, **쌍의 개수를 실행 불가로 번역한 것이 오류**다.
flash 계열 커널은 attention 행렬을 만들지 않아 메모리가 O(N)이다:

```
bag  85 x  2,836셀 (중앙값)    1.7 ms    0.46 GiB
bag 100 x 16,384셀 (최악)     55.0 ms    3.13 GiB
```

그 오판이 bag 기술자를 **256개 숫자**로 좁혔다(sketch는 8,256).

| arm | LR | SEAL 10개 |
|---|---|---|
| v50 | 5e-4 | 발산 (train_ce까지 상승) |
| v51 | 1e-4 | 0.6047 |
| v52 | 2e-5 | 0.6619 |

LR이 낮을수록 나았고, 나는 이것을 "학습 신호가 표현을 개선하지 못한다"로 읽었다.
**그 해석도 틀렸다** — 표현할 수 있는 것 자체가 좁았다.

### 79-3. 재설계 (v53/v54)

```
세포 (N × 1536) → 상한 8,192 추출 → 사영 (N × 512)
  → summary token 32개를 CLS처럼 앞에 붙임
  → Encoder 2층: 세포끼리 self-attention + summary token도 서로 attend
  → summary 32개 flatten = 32 × 512 = 16,384차원 → dual ridge
```

**attention 백엔드는 cuDNN.** B200(sm_100)에서 flash보다 2.7배 빠르다(6.5 vs 17.7 ms).
FA3/FA4는 쓸 수 없다 — PyPI `flash-attn`은 2.8.3(FA2 계열)이 최신, FA3는 Hopper 타깃.
모델이 실제로 cuDNN을 타는지 확인함(기본 42.9 ms == CUDNN 강제, FLASH 강제 63.9 ms).

### 79-4. 궤적이 뒤집혔다

합성 val_auroc:

| arm | ep0 | ep6 | ep12 | ep25 | ep49 |
|---|---|---|---|---|---|
| v52 (구판, 2e-5) | 0.7815 | — | — | 0.7841(정점) | — |
| **v53 (신판, 2e-5)** | 0.7650 | 0.7885 | 0.8173 | 0.8459 | **0.8486** |
| **v54 (신판, 1e-4)** | 0.7665 | 0.8316 | 0.8500 | 0.8623 | **0.8496** |

구판은 ep0가 최고였고 이후 악화됐다. 신판은 **단조 개선**하고, ep49에 0.8486/0.8496으로
CV-only의 0.8885에 근접한다(구판은 0.78에서 멈춤).

**LR 단조성이 뒤집혔다**: 구판은 낮을수록 좋았는데(2e-5 > 1e-4), 신판은 **1e-4가 더 빠르게
오른다**. 학습이 되는 구조가 되자 LR 상한이 달라졌다.

### 79-6. ⚠️ SEAL 채점 결과: 재설계는 **개선이 아니었다**

| arm | 합성 val_auroc (ep49) | **SEAL 10개** |
|---|---|---|
| v41_K128 (계보 A) | 0.8885 | **0.6940** |
| v52 (B 구판, 2e-5) | 0.7841 | **0.6619** |
| v53 (B 신판, 2e-5) | 0.8486 | **0.6526** |
| v54 (B 신판, 1e-4) | 0.8496 (ep25 0.8623) | **0.6219** |

합성 지표는 0.784 → 0.849로 크게 올랐는데 **SEAL은 오히려 내려갔다**(0.6619 → 0.6526).
그리고 합성에서 가장 좋았던 v54가 **SEAL에서는 가장 나쁘다**(0.6219).

**이것이 §69-6의 가장 강한 사례다.** 이전까지는 "합성 지표가 평평한데 실제는 오른다"였는데,
여기서는 **방향이 반대**다. 구조를 키우고 LR을 올려 얻은 합성 성능은 **합성 생성기에 대한
과적합**이었다.

⚠️ 채점 도중 9개 시점의 부분 평균으로 "재설계는 개선"이라고 말했으나, 마지막 task(VHL,
v53 0.4708 vs v52 0.5104)가 들어오며 뒤집혔다. **부분 집계로 판단하지 말 것.**

### 79-7. 그래서 계보 B는 현재 형태로 기각

세포 간 attention도, 16,384차원 기술자도, 학습되는 사영도 SEAL을 올리지 못했다.
문제는 **용량이나 구조가 아니라 일반화**다. 더 키우기 전에 "왜 합성만 좋아지는가"를
답해야 한다.

다만 **ccrcc VHL**은 여전히 예외다 — CV-only 0.4503(랜덤 이하)인데 계보 B는
0.4708~0.5104다. 10개 중 유일하게 계보 B가 이기는 task다.

### 79-5. 부수 수정

- **평가기가 BaseModel 내부에 의존**했다(`aggregator`, `meta_classifier`). 모델 내부를
  모르는 일반 경로를 추가하고, `ICF_FORCE_GENERIC_EVAL=1`로 검증했다:
  v45 er_status 표준 0.7281 == generic 0.7281.
- **YAML 함정**: `lr: 2e-05`는 **문자열**로 파싱된다(YAML 1.1은 소수점과 부호 있는 지수
  요구). AdamW 내부까지 흘러가 `TypeError`로 터진다. 값을 출력해 확인하면 못 잡는다 —
  `tests/test_config_numeric_types.py`가 타입으로 검사한다.
- `optimizer_overrides`는 **지원되지 않는다**(data/model/trainer/logger만). LR 변형은
  `configs/optimizer/*.yaml`을 만들어 연결한다.

---

## 80. 다음 세션이 할 일

1. **계보 B를 더 키우지 말 것.** 구조 확장(세포 간 attention, 16,384차원)이 합성을
   0.784→0.849로 올리고도 SEAL은 0.6619→0.6526으로 내렸다(§79-6). 용량 문제가 아니다.
2. **"왜 합성만 좋아지는가"가 다음 질문이다.** 합성 생성기가 실제 병리 데이터와 다른
   무엇을 보상하는지 규명하지 못하면, 어떤 구조를 시험해도 같은 함정에 빠진다.
   합성 생성기 자체를 점검하거나, 판정을 아예 SEAL만으로 좁히는 선택지가 있다.
3. **ccrcc VHL 단서** — CV-only가 랜덤 이하(0.4503)인 유일한 task에서 계보 B는
   0.4708~0.5104다. 두 계보가 상보적일 가능성. 앙상블/잔차가 여기서 의미 있을 수 있다.
4. **증류 제안은 `current_experiments.md` 5-3 참조.** ⚠️ 순수 증류로는 teacher를 넘을 수
   없다(CV-1은 결정적 특징 맵이라 완벽 모방 = 0.6940).
5. **seed 반복** — 지금까지 arm당 1 seed다. 계보 A의 동률 arm들(v41/v42/v45)이 정말
   동률인지는 seed 반복 없이는 말할 수 없다.

**환경 수칙**: GPU 0·1을 우선 사용하고 다른 GPU는 사용자 허락을 받을 것.

---

## 81. 2026-08-10 — episode 내부 bag별 cardinality + zero-padding/mask

**변경**: 합성 `default` 데이터의 `per_bag_cardinality`를 켰다. 이제 `[1,16384]`,
log-uniform power 1.5에서 **bag마다 독립적으로** cell 수를 뽑는다. ragged list는
`collate_synthetic_training_episode`가 batch 크기 1에서도
`[episode, bag, max_cells, dim]`으로 zero-padding하고 `cell_mask`/`bag_mask`를 함께 반환한다.
CV-only와 Encoder+Ridge의 `forward_episode_batch`는 mask된 cell을 통계·attention에서 제외한다.

**검증**: 신규 `tests/test_per_bag_cardinality_padding.py`와 golden/mask/dense-path를 포함한
compact suite — **109 tests 통과**.

**비용 경고(1,000 episode cardinality Monte Carlo)**: bag 60~100개, 상한 16,384에서는
episode max 중앙값 **15,568**, zero-padding 유효률 평균 **15.85%**로 약 **6.3배 padding
overhead**다. 정확성 변경은 완료했지만 다음 학습 전에 length-bucket encoder를 구현·측정하거나,
이 비용을 받아들이고 GPU 0·1 smoke benchmark로 step time/VRAM을 확인해야 한다.
