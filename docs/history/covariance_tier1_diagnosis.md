# Covariance Tier 1 diagnosis (archived 2026-07-31)

Moved verbatim from `docs/current_status.md` §3 on 2026-07-31. Closed line: Tier 1 (cell selection / covariance) investigation — cell selection is not learnable from bag labels (purity 0.128) and covariance was not the weakness (T3-1: ties composition at matched effect scale). Preserved for its reproducible diagnostic tools and oracle-trap examples.

---

### 🔬 covariance 진단 전체 기록 (2026-07-29) — 🛑 **종료된 라인**

> [!NOTE]
> **이 절은 Tier 1(covariance) 조사의 전체 기록이며, 결론은 "종료"입니다.** 두 가지 이유로 닫혔습니다:
> ⑴ 세포 선택을 bag 라벨로 학습할 수 없음(T1-C 2단계, purity 0.128), ⑵ **애초에 covariance는 약점이 아니었음**(T3-1 — effect scale을 통일하면 composition과 동률).
> 아래 T1-0~T1-C는 그 과정에서 얻은 사실들로, **재현 가능한 진단 도구와 오라클 함정 사례**로서 가치가 있어 보존합니다. 지금 할 일을 찾는다면 §0 또는 §6을 보세요.

#### (당시 진단) 병목은 공분산 활용이 아니라 반응세포 식별

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

