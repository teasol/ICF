# Current development status & multi-location sync SSOT

**Last updated**: `2026-07-29 20:15:00 KST`
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

### ✅ v22 기준선 (2026-07-29 확정) — **버그 수정 반영본이 공식 기준선**

| 항목 | 값 |
|---|---|
| Config | `configs/train_v22_medium.yaml` (20 epoch, 512 steps/epoch = **10,240 steps**, Phase 1과 동일) |
| 코드 상태 | **Cholesky backward + rank-local 수정 반영 후** (커밋 `be36c59`) |
| Best `val_ce_loss` | **0.5946** (epoch 13) |
| 합성 val AUROC | **0.7466**, 95% CI [0.716, 0.776] (episode cluster bootstrap, 104 episodes / 1,698 query) |
| 합성 val Log Loss | 0.5943 |
| Task별 AUROC | composition 0.8022 / combined 0.8170 / interaction 0.7453 / state 0.6595 / **covariance 0.6122 (최난이도)** |
| 체크포인트 | `checkpoints/20260729_160643/v22_medium_fixed/epoch=013-val_ce_loss=0.5946.ckpt` |
| 예측 파일 | `predictions/synthetic_v22_baseline_fixed.pt` |
| 로그 | `logs/20260729_160643/v22_medium_fixed.out` |

**버그 수정 전후 비교** (`scripts/compare_predictions.py`, paired cluster bootstrap):

| | 수정 전 (`100611`) | **수정 후 (`160643`, 공식)** |
|---|---:|---:|
| AUROC | 0.7463 [0.715, 0.775] | **0.7466 [0.716, 0.775]** |
| `val_ce_loss` | 0.5930 | 0.5946 |
| paired 승률 | \- | **0.42 (구분 불가)** |

> [!NOTE]
> **두 수치는 통계적으로 구분되지 않습니다** (승률 0.42, CI 거의 동일). Cholesky 수정은 성능을 올리는 변경이 아니라 **near-singular 상황에서 backward가 non-finite로 터지는 것을 막는 보험**입니다. 평균적인 run에서는 차이가 안 나는 것이 정상이며, 그래도 반영한 이유는 언제 터질지 모르는 잠재 위험을 남겨둘 이유가 없기 때문입니다.
> 이전 수정 전 기준선(`0.7463` / `checkpoints/20260729_100611/`)은 참고용으로만 남기고, **앞으로의 비교는 수정 후 기준선을 씁니다.**

**v21 Phase 1과 사실상 동일**: `val_ce_loss` 0.5946 vs 0.5921 (차이 0.0025), val_loss 궤적도 0.667 → 0.650 vs 0.667 → 0.644로 겹칩니다. Phase 1도 full-context였으므로 **retrieval 제거가 합성 사전학습 성능을 전혀 훼손하지 않았다는 확인**입니다.

**앞으로 아키텍처 변경은 이 수치와 비교합니다** (§5 전략에 따라 ICI가 아니라 여기서 판단). 비교 시 `scripts/compare_predictions.py`로 위 예측 파일과 paired cluster bootstrap을 돌릴 것.

### 🔬 covariance 진단 (2026-07-29) — **병목은 공분산 활용이 아니라 반응세포 식별**

`scripts/diagnose_oracle_covariance_upper_bound.py`, covariance task 206 episodes, episode cluster bootstrap:

| descriptor | relation | AUROC | 95% CI | 모델이 도달 가능? |
|---|---|---:|---|---|
| `latent_dispersion` | `standardized_distance` | 1.0000 | [1.000, 1.000] | ❌ latent 파라미터 오라클 |
| `observed_covariance` | `prototype_cosine` | 0.8931 | [0.876, 0.910] | ❌ **반응세포 마스크 오라클** |
| `observed_covariance` | `multiscale_rbf` | 0.8903 | [0.873, 0.908] | ❌ 동일 |
| **`allcell_covariance`** | `prototype_cosine` | **0.5704** | **[0.550, 0.592]** | ✅ **유일하게 진짜 관측 가능** |
| **학습된 v22 모델** | — | **0.6122** | — | — |
| `observed_spectral` / `local_*` / `oracle_population` | (전부) | 0.52~0.58 | — | 혼재, 전부 무신호 수준 |

> [!IMPORTANT]
> **모델(0.6122)은 진짜 관측 가능한 상한(0.5704)을 이미 넘어섰습니다.**
> 공분산 정보를 "못 쓰고 있는" 것이 아니라, 통짜 bag 공분산에서 뽑을 수 있는 것보다 **이미 더 뽑고 있습니다.**
>
> **0.89는 달성 가능한 목표가 아닙니다.** 그 descriptor는 `episode.responsive_instance_mask`로 세포를 골라 공분산을 계산하는데, 이 마스크는 코드 주석에 명시된 대로 *"Diagnostic-only instance membership. SyntheticEpisodeDataset never exposes this field in training or evaluation batches"* — **모델이 절대 받지 못하는 정답**입니다.
>
> **`observed_` 접두사가 오해를 유발합니다.** 여기서 "observed"는 "latent 파라미터가 아니라 관측된 세포 특징으로 계산"이라는 뜻일 뿐, **어떤 세포를 쓸지는 오라클이 정해줍니다.**

**따라서 0.61 → 0.89 격차의 정체는 공분산 활용이 아니라 반응세포 식별입니다.** covariance 에피소드에서 반응세포는 전체의 **평균 11.7%** (중앙값 11.5%, 범위 2.4~21.4%)에 불과합니다. 그 12%를 알면 0.89, 모르고 전부 쓰면 0.57입니다. 즉 **어느 세포가 반응세포인지 찾아내는 것이 전부**이고, 이는 정확히 **Top-1% Sparse Evidence 모듈이 담당해야 할 일**입니다.

#### T1-1 결과 (2026-07-29): 관계식·게이트 모두 병목이 아님 — 직접 측정으로 확인

`scripts/diagnose_covariance_utilisation.py` (학습된 baseline 체크포인트, covariance episodes):

**(a) 융합 게이트는 열려 있습니다** — 꺼져 있지 않습니다.

| gate | 값 |
|---|---:|
| `covariance_residual_scale` (sigmoid) | 0.2951 |
| `covariance_ridge_scale` (exp, clamp) | 2.3897 |
| **ridge 항 실효 배율** | **0.7052** |
| `covariance_relation_residual_scale` (config 고정) | 0.5000 |
| (참고) population / tail / fusion | 0.2445 / 0.1041 / 0.1137 |

**(b) 관계식 4종이 전부 동률 — 전부 무작위 수준** (모델이 실제로 보는 all-cell 스케치 기준):

| relation mode | AUROC | 95% CI |
|---|---:|---|
| `learned_head` (현재) | 0.5133 | [0.442, 0.589] |
| `prototype_cosine` | 0.5074 | [0.448, 0.567] |
| `multiscale_rbf` | 0.5084 | [0.451, 0.562] |
| `standardized_distance` | 0.5146 | [0.453, 0.571] |
| **전체 모델 (end-to-end)** | **0.6202** | [0.541, 0.692] |

> [!IMPORTANT]
> **가설 (a)·(b) 모두 기각.** 게이트는 실효 0.705로 열려 있고, 관계식은 무엇을 써도 0.51 근처로 동일합니다. `learned_head`를 `prototype_cosine`으로 바꿔봐야 얻을 것이 없습니다.
> 그리고 **전체 모델(0.620)이 어떤 단일 관계식보다도 높습니다** — covariance 분기가 정보를 흘리는 것이 아니라, all-cell 공분산에 애초에 신호가 별로 없는 것입니다.
> **남는 결론은 하나: 병목은 반응세포 식별입니다.**

> [!WARNING]
> **이전 판단 정정.** 세션 중 한때 "0.89가 관측만으로 달성 가능하므로 covariance 분기가 정보를 못 쓰고 있다(헤드룸 +0.28)"고 기록했으나, **틀렸습니다.** `observed_covariance`가 오라클 마스크를 쓴다는 점을 확인하지 못한 결과입니다. `allcell_covariance`를 추가 측정해 바로잡았습니다. 이에 따라 Tier 1의 원래 가설 (a) fusion 희석 / (b) learned_head 열위는 **근거를 잃었습니다** — 관계식(relation)이 병목이라는 증거가 없기 때문입니다.

#### 🔴 T1-A 결과 (2026-07-29): 세포 선택이 **무작위와 구분되지 않음**

`scripts/diagnose_cell_selection.py` — 모델의 세포 랭킹 점수가 `responsive_instance_mask`를 맞히는지 측정 (covariance 80 episodes, 반응세포 11.0%):

| score | 정체 | AUROC | 95% CI |
|---|---|---:|---|
| `studentized` | z-scoring 후 centroid 거리 | 0.5098 | [0.500, 0.520] |
| `outlier_distance` | 문서에 적힌 Top-1% 기준 | 0.5091 | [0.499, 0.519] |
| `novelty` | **aggregator tail이 실제 쓰는 점수** | 0.4984 | [0.489, 0.508] |
| `class_memory` | **meta-classifier rare-evidence의 학습된 점수** | 0.4971 | [0.486, 0.508] |

**precision은 전부 base rate(0.110)와 같고, recall은 유지 비율과 정확히 일치**합니다 (1%→0.010, 5%→0.050, 10%→0.10, 20%→0.20). 이는 **무작위 추출의 정의 그대로**입니다.

**4개 task 전부 동일** (composition 0.499~0.501 / state 0.517~0.520 / combined 0.501~0.503). covariance만의 문제가 아닙니다.

> [!IMPORTANT]
> **v21 "4대 수학 기술" 중 하나인 Top-1% Sparse Evidence 모듈이 반응세포를 전혀 찾지 못합니다.** 기하학적 기준 3종과 **학습된 기준 1종 모두** 무작위 수준입니다. 학습된 `class_memory`가 오히려 가장 낮습니다(0.4971).
>
> **따라서 `rare_evidence_fractions`나 `tail_fractions`의 k값을 조정하는 것은 무의미합니다.** 랭킹 자체에 신호가 없으므로 어디서 자르든 결과는 같습니다.

**왜 실패하는지 — 생성기를 보면 명확합니다.** `effect_mask = (component_index == effect_component_index)` — 반응세포는 **latent mixture component 하나**입니다. 그리고 covariance task에서 그 component에 가해지는 효과는 **위치 이동이 아니라 분산(dispersion) 변화**입니다(`z = z + effect_mask * response_shift`는 composition/state 계열, covariance는 공분산 방향 스케일링).

즉 **현재 선택 기준은 "중심에서 멀리 떨어진 세포"를 찾는데, 반응세포는 멀리 있지 않습니다.** 정상적인 mixture component이고, 단지 퍼진 모양이 다를 뿐입니다. 구조적으로 못 찾는 것이 당연합니다.

**→ T1-B 방향이 명확해졌습니다: 개별 세포의 이상치 정도가 아니라 mixture component(=슬롯) 단위로 판별해야 합니다.** aggregator는 이미 12개 population slot에 세포를 배정하고 있으므로, **반응 component가 특정 슬롯과 정렬되는지** 먼저 확인하세요. 기존 도구 `scripts/diagnose_oracle_slot_alignment.py`가 바로 이 용도입니다.

#### T1-B 결과 (2026-07-29): 슬롯도 못 잡지만, 신호는 특징 안에 있음

**① 반응 component는 슬롯과 정렬되지 않습니다** (`scripts/diagnose_oracle_slot_alignment.py`, v22):

| task | best slot purity | best slot capture | fragmentation entropy |
|---|---:|---:|---:|
| covariance | 0.173 | 0.155 | **0.963** |
| composition | 0.253 | 0.175 | 0.920 |
| state | 0.209 | 0.160 | 0.961 |
| combined | 0.256 | 0.176 | 0.916 |
| (무작위 기준) | ~base rate 0.110 | **~1/12 = 0.083** | 1.000 |

capture 0.155는 무작위 0.083보다 약 1.9배 낫지만, **fragmentation entropy 0.963은 반응세포가 12개 슬롯에 거의 균등하게 흩어져 있다는 뜻**입니다. "반응 component = 특정 슬롯" 구도는 성립하지 않습니다.

**② 그러나 세포 특징에는 신호가 있습니다** (`diagnose_cell_selection.py --probe`, covariance 80 eps):

| score | AUROC | 95% CI | 성격 |
|---|---:|---|---|
| `lda_probe` | 0.8090 | [0.802, 0.816] | ⚠ 같은 세포로 적합·평가 → **512차원 과적합으로 과대평가** |
| **`lda_heldout`** | **0.6969** | **[0.687, 0.707]** | ✅ **정직한 지도학습 상한** (절반으로 적합, 나머지로 평가) |
| `studentized` / `outlier_distance` | 0.510 / 0.509 | — | 모델 기하 기준 |
| `novelty` / `class_memory` | 0.498 / 0.497 | — | 모델 실사용 기준 |

> [!IMPORTANT]
> **반응세포는 세포 특징만으로 식별 가능합니다 — 단 세포 단위 라벨이 있을 때 AUROC 0.70 수준.** 즉 "정보가 없어서 못 찾는" 것이 아니고, **찾을 메커니즘이 없는** 것입니다. 현재 모든 기준이 0.50인데 정직한 상한은 0.70이므로 **0.50 → 0.70 구간이 실제 개선 여지**입니다.
>
> **단, 기대치를 낮게 잡으십시오.** ⑴ 0.70은 **세포 단위 정답 라벨**을 쓴 수치이고 모델에는 bag 단위 라벨(R/NR)만 있습니다. bag 라벨만으로 이 방향을 학습해야 하므로 0.70은 도달하기 어려운 상한입니다. ⑵ covariance AUROC 0.89는 **완벽한** 세포 선택을 가정한 값입니다. 부분적 선택은 부분적 이득만 줍니다. **+0.28을 기대하면 안 됩니다.**

**T1-C 방향**: 세포 선택은 (a) 이상치 거리도 (b) 슬롯 배정도 아닌, **bag 라벨로부터 학습되는 판별 방향(discriminative direction)** 이어야 합니다. `lda_heldout`이 찾은 방향은 episode 내 생성 과정이 일관되므로 원리적으로 학습 가능하나, 이는 노브 조정이 아니라 **새 메커니즘 추가**입니다. 비용/이득을 §5 전략(합성에서 판단)에 따라 먼저 견적하고 결정하세요.

#### T1-C 1단계 결과 (2026-07-29): 이득 곡선은 선형 — 부분 개선도 값이 있음

`scripts/diagnose_selection_gain_curve.py` — 반응세포 마스크에 노이즈를 섞어 선택 순도(purity)별 covariance AUROC 측정 (80 episodes, relation은 `prototype_cosine` 고정, 스케치·관계식 모두 모델 것 그대로. **세포 선택만** 변화):

| 선택 순도 | covariance AUROC | 95% CI |
|---:|---:|---|
| 0.11 (=무작위) | 0.5169 | [0.481, 0.555] |
| 0.20 | 0.5767 | [0.539, 0.615] |
| 0.30 | 0.6252 | [0.588, 0.662] |
| **0.40** | **0.6834** | [0.640, 0.725] |
| 0.50 | 0.7317 | [0.695, 0.768] |
| 0.60 | 0.7780 | [0.743, 0.813] |
| 0.70 | 0.8174 | [0.785, 0.851] |
| 0.80 | 0.8334 | [0.799, 0.868] |
| 0.90 | 0.8642 | [0.832, 0.895] |
| 1.00 (오라클) | 0.8881 | [0.859, 0.915] |
| — 전 세포 사용 (현재 방식) | 0.5571 | [0.525, 0.591] |
| **— held-out LDA 선택 (실측 순도 0.40)** | **0.6638** | **[0.621, 0.708]** |

> [!IMPORTANT]
> **곡선이 처음부터 선형입니다** (0.11→0.40 구간에서 순도 +0.29당 AUROC +0.166, 약 0.57/순도). **문턱 효과가 없어 부분적 개선도 즉시 값을 냅니다.** → **Tier 1은 접을 이유가 없습니다.**
>
> **현실적 기대치**: held-out LDA 품질(순도 0.40)의 선택 메커니즘이면 covariance 분기가 **0.5571 → 0.6638 (+0.107)**. 혼합 곡선의 같은 순도 지점(0.6834)과 근접해 "순도"가 타당한 요약 변수임도 확인됩니다.
>
> 단 이마저 낙관적입니다: held-out LDA는 **세포 단위 라벨**로 적합했고, 실제 메커니즘은 **bag 라벨(R/NR)만** 가지고 이 방향을 학습해야 합니다.

> [!WARNING]
> **⚠ 검출 가능성 문제 — Tier 1을 진행한다면 T3-2가 선행 조건입니다.**
> covariance는 전체 episode의 20%입니다. 따라서 covariance-task AUROC가 +0.07 개선돼도 **전체 합성 AUROC로는 약 +0.014**에 불과한데, 현재 합성 val CI 폭은 **0.060**입니다. **성공해도 전체 지표에서는 보이지 않습니다.**
> → ⑴ 판정은 **task별 covariance AUROC로** 해야 하고, ⑵ 기본 val split의 covariance episode는 18개뿐이므로 **`episodes_per_epoch`를 늘려(T3-2) per-task CI를 좁히는 것이 Tier 1 검증의 전제**입니다.

#### 🛑 T1-C 2단계 결과 (2026-07-29): **bag 라벨만으로는 못 찾음 → Tier 1 종료**

`scripts/diagnose_bag_label_selection.py` — context bag에 **bag 라벨(R/NR)만으로** 판별 규칙을 적합하고 query bag에서 평가 (세포 라벨은 채점에만 사용, covariance 1,321 query bags):

| rule | 무엇을 노리는가 | AUROC | 95% CI | purity@k |
|---|---|---:|---|---:|
| `bag_label_lda` | R/NR 세포의 **평균 차이** | 0.5020 | [0.498, 0.506] | 0.111 |
| `bag_label_csp` | R/NR 세포의 **분산 비** (dispersion) | 0.5136 | [0.509, 0.518] | 0.128 |
| (무작위) | — | 0.5000 | — | 0.110 |

> [!CAUTION]
> **사전에 정한 판정 기준(purity ≤ 0.15 → Tier 1 종료)에 따라 Tier 1을 종료합니다.** 최고 purity 0.128은 무작위 0.110과 거의 같습니다.
>
> 판정 기준은 **결과를 보기 전 커밋 `f5ddbf5`에 미리 기록**했습니다. 사후에 문턱을 옮기지 않습니다.

**왜 실패하는가**: R bag 안에서도 반응세포는 11%뿐이고 나머지 89%는 NR bag 세포와 구별되지 않습니다. 여기에 covariance effect scale이 0.30~0.80이라, bag 단위로 집계한 평균 차이나 분산 비에서는 신호가 배경에 묻힙니다. 예상대로 **분산 기반(`csp`)이 평균 기반(`lda`)보다 낫긴 했지만**(0.514 vs 0.502) 그 차이도 무의미한 수준입니다.

**정리하면**: 세포 특징에는 정보가 있습니다(세포 라벨 사용 시 held-out 0.697). 하지만 **모델이 실제로 가진 감독 신호(bag 라벨)로는 도달할 수 없습니다.**

> [!NOTE]
> **이 결론의 한계**: 검증한 것은 **episode별 closed-form 선형 규칙 2종**입니다. 수천 episode에 걸쳐 end-to-end 학습되는 **비선형 attention 메커니즘**은 episode 하나에서 못 뽑는 통계적 힘을 누적할 수 있어 원리적으로 배제되지는 않습니다.
> 다만 이제 **입증 책임이 넘어갔습니다** — 값싼 검증들이 전부 무신호를 보였으므로, 비용을 들일 **긍정적 근거가 없습니다.** 나중에 이 방향을 다시 열려면 "bag 라벨로 학습 가능하다"는 증거를 먼저 확보하고 시작하세요.

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
| 합성 val (v22 기준선) | 104 episodes / 1,698 query | 0.7466 | [0.716, 0.776] | **0.060** |

합성 구간이 약 **4배 좁고**(그것도 episode cluster bootstrap이라는 보수적 계산으로), 신호도 훨씬 강합니다(0.75 vs 0.55). 합성은 `episodes_per_epoch`로 더 좁힐 수 있지만 ICI는 87명이 상한입니다. v21의 실패는 **검출력 없는 지표 위에서 아키텍처를 반복 비교한 것**이었고, ICI를 반복해서 보면 그 87명에 과적합됩니다.

**지켜야 할 선**: ICI 결과를 보고 아키텍처를 다시 고치기 시작하면 ICI는 더 이상 테스트 세트가 아닙니다. 또한 ICI에서 ±0.13 이내 변동은 그 자체로 아무 근거가 되지 않습니다.

**합성 평가의 CI는 query가 아니라 episode 단위(cluster bootstrap)로 계산합니다.** 한 episode의 query들은 context set과 생성 파라미터를 공유해 독립이 아니며, query 단위 재표집은 구간을 실제보다 좁게 만듭니다. 실질 표본 크기는 **episode 수**입니다.

---

## 6. 다음 작업 세션 Action Plan — 구조적 변경 및 실험 목록

> [!IMPORTANT]
> 모든 판단은 **합성 val**에서 하고 ICI는 손대지 않습니다 (§5). 후보마다
> `scripts/evaluate_synthetic.py` → `scripts/compare_predictions.py`로 기준선
> (`predictions/synthetic_v22_baseline_fixed.pt`, AUROC 0.7466 [0.716, 0.776])과
> paired cluster bootstrap 비교할 것.

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

### ➡ 다음 우선순위: Tier 3 → Tier 2

Tier 1이 닫혔으므로 **Tier 3(방법론)을 먼저** 하는 것이 합리적입니다. Tier 1에서 배운 교훈이 그대로 적용됩니다:
- **T3-1 (effect scale 정규화)**: task별 AUROC가 난이도에 오염되어 있어 아키텍처 강약을 비교할 수 없습니다. Tier 2에 들어가기 전 필수.
- **T3-2 (val episode 증량)**: per-task CI가 넓어 어떤 개선도 검증이 안 됩니다. covariance episode 18개는 너무 적습니다.
- **T3-3 (v22 hard 기준선)**: 대조군 부재.
그 다음 **Tier 2(state)** — 단, T2-1의 경고대로 오라클/관측가능 descriptor를 처음부터 분리해 측정하세요.

> [!NOTE]
> **폐기된 가설**: (a) fusion 희석, (b) `learned_head` < `prototype_cosine`, (c) sketch 압축 손실.
> 관계식이나 스케치가 병목이라는 증거가 없습니다 — 모델은 이미 통짜 공분산으로 가능한 것 이상을 하고 있습니다. `learned_head` A/B는 하고 싶다면 값싸게 해볼 수는 있으나 **기대 효과는 낮습니다.**

### Tier 2 — state 분기 (0.6595, 근거 중간)

**T2-1. state용 상한 진단 도구를 만들어 먼저 측정.** covariance에는 `diagnose_oracle_covariance_upper_bound.py`가 있지만 state용은 없습니다.

> [!CAUTION]
> **Tier 1에서 저지른 실수를 반복하지 마세요.** 도구를 만들 때 **오라클을 쓰는 descriptor와 진짜 관측 가능한 descriptor를 반드시 분리해 라벨링**하세요. covariance 진단의 `observed_covariance`는 이름과 달리 `responsive_instance_mask`(정답)로 세포를 골라 0.89를 냈고, 이 때문에 한동안 잘못된 결론을 기록했습니다. 진짜 관측 가능한 값은 0.57이었습니다. state 도구에서도 **모델이 실제로 받는 입력만으로 계산한 기준선을 반드시 포함**시키세요.

### Tier 3 — 방법론 (성능이 아니라 판단 신뢰도)

**T3-1. task별 effect scale 정규화 실험.**
현재 task별 AUROC는 생성기 effect scale(composition 1.40 / state 0.72 / covariance 0.55)에 오염되어 **아키텍처의 상대적 강약을 직접 비교할 수 없습니다.** 모든 task의 effect scale을 동일하게 맞춘 진단용 데이터 config를 만들면 "어느 메커니즘에 실제로 약한가"를 처음으로 공정하게 볼 수 있습니다. 학습된 모델을 그 데이터로 평가만 하면 되므로 재학습 불필요.

**T3-2. 합성 val 검정력 확보.**
현재 104 episodes → CI 폭 0.060. Tier 1/2에서 기대하는 개선폭이 그보다 작다면 검출이 안 됩니다. `val_dataset_kwargs.episodes_per_epoch`를 늘려 CI를 좁히세요 (episode 수가 실질 표본 크기, §5).

**T3-3. Stage 2 (Hard) 기준선.**
`configs/train_v22_hard_realworld.yaml` 아직 v22로 미실행 (v21 Phase 2 참고값 `val_ce_loss 0.6845`). Tier 1 변경이 medium에서만 좋고 hard에서 무너지지 않는지 확인할 대조군이 필요합니다.

### 하지 말 것

- **ICI 실행 금지** — 후보 확정 전까지. §5 참고, 지금 돌리면 테스트 세트 조기 소진.
- **오라클 기반 상한을 목표치로 삼기 금지** — descriptor가 `responsive_instance_mask`나 latent 파라미터를 쓰는지 항상 먼저 확인하세요. 모델이 못 받는 정보로 만든 상한은 목표가 아닙니다 (§3 정정 사례).
- **T1-A/B 없이 covariance 아키텍처 뜯기 금지** — 세포 선택 규칙이 상한을 얼마나 회수하는지 모른 채 관계식을 바꾸면 헛수고입니다.
- **effect scale 정규화 없이 task별 AUROC로 우열 판단 금지** (T3-1).

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
