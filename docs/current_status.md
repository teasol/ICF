# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-13` (§113 — **v87 rare 재검증도 null**, VHL/BAP1 타겟 가설도 반증 —
데이터 생성기 축(noise·rare) 소진; §112 — **v86 noise 재검증 null**; §111 — **§107-6(fixed P ×
Medium, v85) 취소**, 사용자 결정; §110 — **v84 deep-head 기각**으로 relation head 깊이 축 마감;
§109 — **baseline을 v83 linear head로 승격**, §107-3 판정 게이트 미달 상태에서의 사용자 결정)

> [!IMPORTANT]
> **활성 baseline은 v83 linear head(§109)이고 공식 SEAL 10-task macro는 1-GPU 4 seed 평균 0.6880이다.**
> ```
> config: configs/train_v83_linear_head_1536_1gpu.yaml   (self-contained)
> ckpts:  checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/  (epoch 49)
> tags:   v83_linear_head_seed4{2..5}_ep49
> macro:  0.6905 / 0.6896 / 0.6774 / 0.6944  →  mean 0.6880, seed std 0.0074
> ```
> ⚠️ **이 승격은 §107-3 판정 게이트(4/4 시드 부호 일치 + `|t| ≥ 2.5`)를 충족하지 못한 상태에서
> 내려진 사용자 결정이다(§109).** v82 baseline(0.6846/0.6870/0.6821/0.6802) 대비 seed-paired Δ는
> +0.0059 / +0.0026 / **−0.0047** / +0.0142, 평균 **+0.0045**, **t ≈ 1.15** — seed 44가 부호 반전이고
> (3/4 양수), `|t|`도 게이트에 크게 못 미친다. "뚜렷하진 않아도 올랐다고 보는 게 맞다"는 사용자
> 판단으로 승격했다(2026-08-13) — **통계적 미판정 상태의 승격**임을 인용할 때마다 밝힐 것
> (§104-5·§105-4가 경고한 "게이트 미달 승격"과 같은 범주). 직전 baseline **v82 Medium**
> (1-GPU 4 seed 0.6835)은 historical로 남는다. 자세한 계산은 §108, 결정 배경은 §109.
>
> ⚠️ **0.6880이라는 값은 옛 v77 DDP4 baseline(§104)의 0.6880과 숫자만 같다** — 레짐이 전혀
> 다른(1-GPU 4 seed vs DDP4 1 seed) 별개의 수치다. 혼동하지 말 것(§107-1·§107-2).
>
> 채점은 계속 **epoch 49 고정**이다(§104-2) — val_ce가 평평한 arm에서 validation-best가
> 과소학습 지점을 고른다(v80 seed 43이 epoch 16에 걸려 −0.0089).
>
> ⚠️ **relation head 깊이 축은 §110에서 마감됐다.** §108(GELU 제거, `Linear(12,1)`)이 미판정이었던
> 반대쪽 — `12→32→32→1`로 심화한 v84 — 은 **명확히 기각**됐다(v82 기준 Δ−0.0057 t=−3.63, v83 기준
> Δ−0.0102 t=−3.61, 둘 다 4/4 시드 부호 일치). 이 축에서 새 arm을 더 설계하지 말 것.

**한 줄**: 활성 baseline은 v83 linear head **1-GPU 4 seed 평균 = SEAL macro 0.6880**(§109, 사용자 결정 — §107-3 게이트 미달인 채로 승격), 새 arm 판정은 여전히 **같은 레짐·4 seed의 seed-paired Δ + t**(§107-3)로 하며 시드별 fold-paired CI(§99)는 보조 근거다.

**Status**: **활성 baseline v83 linear head(relation head를 32-hidden GELU에서 bare `Linear(12,1)`로 축소, trainable 197,057 → 196,621), 1-GPU 4 seed 평균 epoch 49 = 0.6880 (seed std 0.0074). §109에서 사용자 결정으로 승격했으나 §107-3 판정 게이트(4/4 시드 부호 일치 + |t|≥2.5)는 충족하지 못한다 — v82 대비 seed-paired Δ +0.0045, t≈1.15, seed 44만 부호 반전(3/4 양수)(§108). 직전 baseline v82 Medium ClassSep `[0.5,1.4]` 1-GPU 4 seed 0.6835는 historical. ⚠️ v83의 1-GPU 4 seed macro 0.6880은 옛 v77 DDP4 baseline(§104)의 0.6880과 숫자만 같은 별개 수치다 — 혼동 주의. §107에서 판정 레짐이 DDP4 1 seed → 1-GPU 4 seed로 전환됐고 이전 DDP4 숫자(v77 0.6880 tag `v77_hard_ep49`, Medium 0.6881, v41_K128 0.6940)와는 직접 비교 불가다. 실행 중인 학습·평가 없음. §106에서 v77/v80/v81/v82를 각각 4 seed로 1-GPU 동일 레짐 비교 — Medium 0.6835 > Hard 0.6781 > fixed-P 0.6734 > shallow-MLP 0.6722. 난이도는 Medium이 이기고(+0.0053, t=3.0) learnable P는 여전히 미판정(+0.0048, t=1.5, seed 44 부호 반전)이다. §105에서 과거 판정 36건을 감사하고 27개 arm을 epoch 49로 재채점했다 — 계보의 두 승격(v74→v76 +0.0004, Hard vs Medium +0.0001)이 모두 "판정 불가"로 내려갔고 ridge calibration 기각은 철회됐다. ClassSep sweep의 Medium 값은 §91의 오기였다(0.6823 → 0.6881). v78(−0.0004/−0.0047)·v79(−0.0105)·v80 shallow MLP(−0.0158, 4 seed) 모두 기각. fixed P × Medium 4 seed(§107-6)는 **취소됐다**(§111, 사용자 결정 — 새 실험 재기획 중). §110에서 head를 `12→32→32→1`로 심화한 v84도 4 seed로 평가했다 — v82 기준 Δ−0.0057(t=−3.63)·v83 기준 Δ−0.0102(t=−3.61), 둘 다 4/4 시드 부호 일치로 **기각**되어 relation head 깊이 축은 얕음(미판정)·기본·깊음(기각) 세 지점이 다 나와 소진으로 본다. macro seed std 0.0051 실측 → 단일 시드 판정 게이트 ≈ 0.010(2σ)이고 task별 CI는 판정 근거로 쓰지 않는다(§104). 역사적 전체 최고는 v41_K128 0.6940(레짐 상이, 직접 비교 불가).**

> [!IMPORTANT]
> **읽는 순서 (2026-08-13)**: **§109를 먼저 읽을 것** — baseline이 **v83 linear head**로 바뀐 절이고,
> 이 승격은 **§107-3 판정 게이트를 충족하지 못한 상태에서의 사용자 결정**이다. 공식 숫자는
> **0.6880**(1-GPU 4 seed)이며, 이전 v77 DDP4의 0.6880과 **값은 같지만 레짐이 다른 별개의
> 숫자다** — 혼동하지 말 것. 그다음 **§108** — v83 실험 자체(relation head의 GELU를 없애고
> bare `Linear(12,1)`로 줄인 ablation)와 v82 대비 seed-paired Δ(+0.0045, t≈1.15, 미판정)를
> 담은 절이다. 그다음 **§110** — §108의 반대 방향(head를 `12→32→32→1`로 심화한 v84)이며
> 이건 **양쪽 baseline(v82·v83) 모두 기준으로 기각**됐다(4/4 시드, |t|>3.6) — head 깊이 축은
> 이걸로 소진으로 본다. 그다음 **§107** — 판정 레짐이 DDP4 1 seed → 1-GPU 4 seed로 전환된 절이며, v82가
> 그 레짐에서 처음 승격된 baseline이었다(지금은 historical). 새 arm 판정 절차는 §107-3,
> 무효화되는 비교 대상은 §107-2다. 그다음 **§106** — 네 arm을 각각 4 seed로 같은 레짐에서
> 비교한 결과이자 §107 결정의 근거다. **Hard는 최적이 아니고(Medium +0.0053, 4/4 seed), learnable P는 같은 난이도에서
> t=1.5로 여전히 미판정이며, 1-GPU는 DDP4보다 −0.0098이다.** v80의 −0.0158은 그 레이아웃
> confound가 섞인 값이고 정당한 control 대비로는 −0.0059다. 그다음 **§105 → §104**. §105는 과거 판정 36건 감사와
> 27개 arm의 epoch 49 재채점 결과다 — **§91의 ClassSep Medium 값이 오기(0.6823 → 0.6881)이고,
> 계보의 두 승격(v74→v76, Hard 선택)이 판정 불가로 내려갔다.** 그다음 §104 —
> 채점 규칙(epoch 고정)과 **task별 CI 사용 금지**가 여기서 정해졌다.
> ⚠️ §104가 정한 **baseline 숫자(0.6880, DDP4)와 단일 시드 게이트(0.010)는 §107이 대체**했고,
> §107의 baseline(v82, 0.6835)은 **§109가 대체**했다 — 지금 활성 baseline은 v83 0.6880(1-GPU 4 seed)이다.
> 그다음 §99(fold-paired Δ + CI, `scripts/compare_arms_paired.py`) — 시드별 보조 근거로 쓴다.
> §98 판정표 4건은 §99-1에서 fold-paired CI로 재검증되어 전부 유지됐으나, §104-4가 그중
> 일부(ridge calibration −0.0033, v78 무가중 −0.0047)를 **seed 노이즈와 구분 불가**로 되돌렸다.
> **v78·v79·v80 모두 기각**이고 CV/DD 배선 축은 소진으로 본다(§103-5).
> **§103-6의 seed 반복 선행 조건은 §104-3에서 해소됐다.**
> §2~§97 본문은 [`history.md`](history.md) §20–§23으로 아카이빙됐다(§101).

* **계보 A = CV-only** (`src/models/baseline.py`, 학습 파라미터 **229개**).
  현행 최고 **v41_K128 = SEAL 10개 0.6940** (ABMIL 0.727에 −0.033).
  ⚠️ **이 값은 현행 판정 레짐(1-GPU 4 seed) 밖의 단일 시드 기록**이라 v83 0.6880와 직접 뺄 수
  없다(§107-2). 1-GPU 페널티 0.0098을 감안하면 근접하지만 **"따라잡았다"고 쓰지 말 것.**
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
* **GPU 정책**: ICF는 GPU 0–3만 사용한다. GPU 4–7은 사용하지 않는다.

현행 아키텍처 명세는 [`current_architecture.md`](current_architecture.md),
실험 절차·결과표·금지사항은 [`current_experiments.md`](current_experiments.md).

**지금 돌아가는 것 (2026-08-13)**: 없음, 어느 노드에서도. §108의 v83 4 seed 평가와 §110의 v84
4 seed 평가가 모두 끝났다 — §109에서 baseline을 v83 linear head(1-GPU 4 seed 0.6880)로
승격했고(사용자 결정, §107-3 게이트 미달), §110에서 v84(deep head)는 양쪽 baseline 기준 모두
기각했다. **fixed P × Medium(§107-6, v85)은 취소됐다**(§111, 2026-08-13 사용자 결정) — 진행할
필요가 없다고 판단해 실험 계획을 접었고 config(한 번도 실행되지 않음)도 삭제했다. 새 실험
방향은 재기획 중이다. §109 승격의 통계적 근거를 보강하려면 `scripts/compare_arms_paired.py`로
fold-paired CI를 확인하는 것도 다음 후보다.

결과 재확인:
```bash
for tag in v53_enc v54_enc; do
  printf "%-10s " $tag
  grep -hoP 'fold-mean AUROC: \K[0-9.]+' logs/official50/*_${tag}.log \
    | awk '{s+=$1;k++} END{printf "%.4f (%d개)\n", s/k, k}'
done
```

---

> **사용자 결정 (2026-08-12, 최신)**:
> 1. **v77 baseline은 `epoch 49` checkpoint이고 SEAL macro는 `0.6880`이다.** 앞으로 채점은
>    epoch 49 고정으로 하고 validation-best 선택을 판정에 쓰지 않는다(§104).
> 2. **Hard v76을 canonical v77 baseline으로 승격.** v30 S2 결정은 역사적 기록이다.
> 3. **ICI는 기본 잠금 유지.** §50과 §86은 사용자 명시 해제에 따른 예외 평가.
> 4. **Musk 목표는 0.95 유지.**

**Read first if you are picking this up**: **§104 (baseline epoch 49 = 0.6880, 판정 게이트, v80 기각)**, **§98 (v77 명명·baseline 승격)**, §97 (large-ragged), §96 (아키텍처 SSOT), §91 (Hard 선택), §89 (v76 구조/학습 경계),
§88 (v74 baseline/CT/v71–v74 판정),
§87 (DD/v70/synthetic 일반화),
§86 (canonical CV+mean 계약), §85 (v62–v66), §71 (SEAL 판정), §73–§74
(호환성/학습 경로), §79 (generic 평가/YAML).

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

**활성 baseline: v83 linear head, 공식 SEAL 10-task macro 0.6880**
= **1-GPU 4 seed 평균**(§109, **사용자 결정 — §107-3 판정 게이트 미달**). config
`configs/train_v83_linear_head_1536_1gpu.yaml`,
ckpts `checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/`, tags `v83_linear_head_seed4{2..5}_ep49`,
시드별 0.6905/0.6896/0.6774/0.6944 (std 0.0074).
`CovarianceMeanLearnablePDDCTMLPModel`, 학습 파라미터 196,621개(P 196,608 + head 13) —
**모델은 v82/v77과 거의 동일하고 relation head만 `12→32→1`(GELU)에서 bare `Linear(12,1)`로
바뀌었다.**

⚠️ **이 승격은 §107-3 게이트(4/4 시드 부호 일치 + |t|≥2.5)를 충족하지 못했다.** v82(0.6835) 대비
seed-paired Δ +0.0045, t≈1.15, seed 44만 부호 반전(3/4 양수) — "뚜렷하진 않아도 올랐다고 보는 게
맞다"는 사용자 판단으로 승격했다(§108→§109). 인용할 때 이 점을 함께 밝힐 것.

⚠️ **0.6880이 옛 v77 DDP4 baseline(§104)의 0.6880과 숫자가 같은 건 우연이다.** 완전히 다른 레짐
(1-GPU 4 seed vs DDP4 1 seed)의 별개 값이며 "제자리로 돌아왔다"로 읽으면 안 된다. 직전 baseline
v82 Medium은 같은 1-GPU 4 seed 레짐에서 0.6835, v77 Hard는 0.6781이다(§107-1).
역사적 전체 최고 **v41_K128 CV-only 0.6940**(229 파라미터)도 DDP4 1 seed라 직접 비교 대상이
아니다. 지도학습 ABMIL 0.7266과의 격차는 여전히 크다.

**지금 돌아가는 것**: 없음. §108(v83)·§110(v84) 4 seed 평가가 모두 끝났고 §109에서
baseline을 v83으로 승격했다(사용자 결정). v84(head 심화)는 §110에서 기각 — head 깊이 축은
소진. **fixed P × Medium(§107-6, v85)은 취소됐다**(§111, 사용자 결정) — 새 실험 방향은 재기획
중이다. §109 승격의 통계적 근거 보강도 후보다. CV/DD 배선 축은 소진으로 본다(§103-5).

**판정 방법은 §107에서 정해진 그대로다 — baseline만 §109에서 v83으로 바뀌었다.**
arm과 baseline(v83)을 각각 **1-GPU 4 seed(42–45)**로 돌려 **같은 시드끼리 뺀** 평균 Δ와 t로
판정한다(**4/4 부호 일치 + |t| ≥ 2.5**, 갈리면 미판정). 시드별 fold-paired CI는 보조 근거다.
GPU 불필요:

```bash
python scripts/compare_arms_paired.py --baseline v83_linear_head_seed42_ep49 --arm <TAG>_seed42_ep49
```

**세 줄 아키텍처**: bag의 cell을 learnable P(1536×128)에 사영해 covariance를 만들고, CV(ridge)·
DD(dispersion)·CT(abundance) 세 branch가 에피소드마다 **closed-form으로** 12개 relation feature를
만들어 (활성 baseline v83에서는) bare `Linear(12,1)`이 읽는다(v82/v77은 `12→32→1` GELU MLP였다).
분류기 weight를 저장하지 않고 ridge를 매 에피소드 다시 푼다.
현행 스펙은 [`current_architecture.md`](current_architecture.md).

**열린 과제**
1. ~~**학습 seed 노이즈가 미측정**~~ **해소 (§104-3·§106-1·§107)**: seed std가 실측됐고
   (arm별 0.0018~0.0053) 판정 단위가 4 seed가 됐다. 남은 부채는 **과거 arm 전부가 DDP4 1 seed
   기록**이라는 것 — 새 arm과 비교하려면 그 arm을 1-GPU 4 seed로 다시 돌려야 한다.
   latent sweep의 비단조성(L16 딥)도 그 부채에 포함된다.
2. **cptac_ccrcc VHL 0.4385 — 랜덤 이하**. large-ragged가 +0.0090(CI 0 제외)로 올려도 여전히
   0.45 미만. 노이즈가 아니라 체계적 부호 문제로 의심된다.
3. **cptac_ccrcc BAP1이 large-bag에서만 −0.0179로 무너진다** (§99-2).
4. ICI는 사용자 지시로 잠금.

**작업 규칙 4가지**
- 판정은 **SEAL 10개 macro**, 그것도 **fold-paired Δ + CI**로 (§99). 합성 val 지표는 checkpoint
  선택에만 쓴다 — 이 리포에서 합성이 좋아지고 SEAL이 내려간 사례가 반복됐다(v54가 최악의 예).
- **clipping 금지**, **bf16-mixed 필수**(학습·평가 양쪽), GPU는 **0–3만** 사용.
- 장시간 작업은 완전 이탈형 백그라운드로 띄우고 PID/PGID·로그 경로를 즉시 기록한다.
  프로세스는 **프로세스 그룹**으로 죽인다(wrapper PID만 kill하면 GPU가 안 풀린다).
- 다음 Action과 판정 기준이 명확하면 재확인 없이 실행하고, 각 논리 단위마다 결과·명령·산출물
  경로·판단·다음 Action을 이 문서에 갱신한 뒤 커밋한다.

⚠️ **다시 열지 말 것**: [`history.md`](history.md) §19가 닫힌 결론 10건을 모아둔다
(retrieval, 세포 선택, CCER 계열, Q-5 상수 분기, CV-2 손잡이, label-free 사영 등).

## 1. 멀티 작업공간 (연구실/집/노트북) 바톤 터치 지침

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 원칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 새로 접속한 Agent는 **`docs/` 최상위 루트의 Living md 파일 5개(`agent_handoff.md`, `current_status.md`, `current_architecture.md`, `current_experiments.md`, `README.md`)와 현행 `architecture_*_proposal.md` 1개를 최우선으로 정독**하여 전체 개발 맥락과 프로젝트 규칙을 파악합니다.
> 3. 터미널 조회가 필요한 명령어는 NVML/쉘 hang 방지를 위해 **반드시 `timeout 3s ps aux | grep python`과 같이 타임아웃**을 적용합니다.
> 4. 코드 변경 시 unittest 통과 필수:
>    `timeout 300s /home/aibio_3/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"`

---

## 2. 프로젝트 핵심 아키텍처 및 환경 명세 (Architecture **v24 확정**) — **아카이브됨**

v24 확정 시점의 아키텍처·환경 명세. 현행 스펙은 [`current_architecture.md`](current_architecture.md). 전문은 [`history.md`](history.md) §6.

## 3. 실험 현황 — **아카이브됨**

v22~v24 시대 실험 현황표(212줄). 결론은 bag-collapse family 진단으로 수렴. 전문은 [`history.md`](history.md) §6.

## 4. v22 결정: retrieval 완전 제거 (2026-07-29) — **아카이브됨**

retrieval 완전 제거 결정. ICI 규모에서 이득 없음 — 다시 열지 말 것. 전문은 [`history.md`](history.md) §5.

## 5. 실험 전략 (2026-07-29 확정) — **아카이브됨**

v22 시점 실험 전략. 전문은 [`history.md`](history.md) §1.

## 6. 다음 작업 세션 Action Plan — 구조적 변경 및 실험 목록 (아카이브됨)

구조 변경·실험 목록(당시 Action Plan). 이미 소진. 전문은 [`history.md`](history.md) §6.

## 7. 평가 프로토콜 보강 (2026-07-29) — **아카이브됨**

평가 프로토콜 보강. 검증된 불변식은 history §2에 통합. 전문은 [`history.md`](history.md) §2.

## 8. Source of Truth 파일 — **아카이브됨**

Source of Truth 파일 목록. 현행은 이 문서 상단과 [`README.md`](README.md). 전문은 [`history.md`](history.md) §0.

## 9. 2026-07-31 세션 핸드오프 — v23/v24 bag collapse family — **아카이브됨**

v23/v24 bag collapse family 세션 핸드오프. 전문은 [`history.md`](history.md) §6.

## 10. 2026-08-01 세션 핸드오프 — v24 확정, 평가 계획 폐기 — **아카이브됨**

v24 확정, 당시 평가 계획 폐기. 전문은 [`history.md`](history.md) §6.

## 11. 2026-08-02 세션 핸드오프 — v25 Medium 평가 완료(사실상 동률), Easy tier 실험 진행 중 — **아카이브됨**

v25 Medium 사실상 동률, Easy tier 진행. 전문은 [`history.md`](history.md) §6.

## 12-14. 2026-08-02 세션 — 폴더/문서·config/src·scripts·tests 정리 3단계

> 아카이브됨 (2026-08-02, 핸드오프 정리): checkpoint/log/prediction purge(53GB→3.3GB),
> 구버전 문서·config·스크립트 삭제, src/scripts/tests 참조 무결성 점검 기록. 전문:
> [`history.md`](history.md) §12-14.
>
> **하나만 아직 열려 있음**: §13의 config 삭제가 `test_d_stages_differ_only_in_selected_nuisance`를
> 깨뜨림 (`configs/trainer/learnability_d20.yaml` 삭제, §16에서 발견·미조치) — 상세는
> archive.md §13 경고 참고.

---

## 15. 2026-08-02 세션 마무리 — 정리 3단계 + v25 폐기 확정 + 브랜치 정리 — **아카이브됨**

정리 3단계 + v25 폐기 확정 + 브랜치 정리. 전문은 [`history.md`](history.md) §18.

## 16. 2026-08-02 세션 (이어짐) — v26/v27/v29 설계안 검토, 학습 없는 게이트 3종, CLS-token pooling(v26) 구현·학습 시작, 제안서 archive
## 17. 2026-08-03 세션 — v26 학습 완료 + CLS attention 진단 프로브 (24-CLS 제안 사전검정) — **아카이브됨**

v26 학습 완료 + CLS attention 진단 프로브. 전문은 [`history.md`](history.md) §7.

## 18. 2026-08-03 — E7 재검정: 지도 component-selection 상한 재확인 (Path B 관문) — **아카이브됨**

E7 재검정 — 지도 component-selection 상한 재확인(Path B 관문). 전문은 [`history.md`](history.md) §7.

## 19. 2026-08-03 — 정규화 천장 프로브: 고정 정규화가 천장을 제한하는가 (사용자 가설 검증) — **아카이브됨**

고정 정규화가 천장을 제한하는가 — 정규화 천장 프로브. 전문은 [`history.md`](history.md) §2.

## 20. 2026-08-03 — v24 no-L2 ablation: per-cell L2 정규화 제거 학습 (진행 중) — **아카이브됨**

v24 no-L2 ablation(per-cell L2 제거). 전문은 [`history.md`](history.md) §6.

## 21. 2026-08-03 — Zero-shot Musk (Musk2) MIL 벤치마크 테스트 — **아카이브됨**

Zero-shot Musk(Musk2) MIL 벤치마크. 전문은 [`history.md`](history.md) §8.

## 22. 2026-08-03 — 전략 전환: 생성기 개선 (Musk-like easy 데이터) — 가설 판정 완료 — **아카이브됨**

전략 전환 — Musk-like easy 생성기 개선, 가설 판정 완료. 전문은 [`history.md`](history.md) §8.

## 23. 2026-08-03 — Musk 0.95 로드맵: raw bag-stat token (mean/skew/kurt) 학습 중 — **아카이브됨**

Musk 0.95 로드맵 — raw bag-stat token(mean/skew/kurt). 전문은 [`history.md`](history.md) §8.

## 24. 2026-08-03 — Phase 1 IA-MIL (Instance-Attention MIL) — 판정: 음성 — **아카이브됨**

Phase 1 IA-MIL 판정 **음성**. 세포 선택은 bag 라벨로 학습 불가(네 번 닫힌 경로). 전문은 [`history.md`](history.md) §19.

## 25. 2026-08-04 — IA-MIL 폐기 + 문서/파일 정리 + 핸드오프 — **아카이브됨**

IA-MIL 폐기 + 문서/파일 정리. 전문은 [`history.md`](history.md) §19.

## 26. 2026-08-04 — Musk 전이 재진단: P1/P2 기각 + v30(CFMT) 제안 — **아카이브됨**

Musk 전이 재진단 — P1/P2 기각 + v30(CFMT) 제안. 전문은 [`history.md`](history.md) §8.

## 27. 2026-08-04 — 세션 마무리: 사용자 결정 반영, 문서 압축, config 재현성 복구 — **아카이브됨**

세션 마무리 — 문서 압축, config 재현성 복구. 전문은 [`history.md`](history.md) §9.

## 28. 2026-08-04 — v30 S1/S2 판정 과정 (B1 `poolz_l2`·B2 cardinality) — **아카이브됨, §29로 승격 완료**

v30 B1/B2 실험·판정 전체 기록(S1 `poolz`/`poolz_l2` 음성, S2 B2+B1 양성, paired bootstrap,
B1·B2 상호 필수 근거, 교차 분포 합성 무회귀)은 [`history.md`](history.md) §28로
이관되었습니다. 최종 결론 요약은 헤더 Status와 §29 참고.


---

## 29. 2026-08-04 — **v30 확정 baseline: B1 `poolz_l2` + B2 cardinality-faithful (사용자 승격 결정)** — **아카이브됨**

**v30 확정 baseline** = B1 `poolz_l2` + B2 cardinality-faithful. 두 손잡이는 상호 필수. 전문은 [`history.md`](history.md) §9.

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

## 38. 2026-08-05 — v32b DR-CCER: 비판적 검토 반영 개선안 + 구현 + Stage A 학습 시작 — **아카이브됨**

v32b DR-CCER — **CCER 계열 실증적 폐기**. branch 활성 ≠ 상보 정보. 전문은 [`history.md`](history.md) §10.

## 39. 2026-08-05 — v32b 완료 결과 평가 + v33 MR-BagPFN proposal — **아카이브됨**

v32b 결과 평가 + v33 MR-BagPFN proposal. 전문은 [`history.md`](history.md) §11.

## 40. 2026-08-05 — 기본 unittest suite compact화 — **아카이브됨**

기본 unittest suite compact화. 전문은 [`history.md`](history.md) §18.

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

## 49. 2026-08-07 — 아키텍처 효율화(MLA-slot) + v34-1536 대규모 컨텍스트 학습 완주 + PathoBench 5-fold CV — **아카이브됨**

MLA-slot 효율화 + v34-1536 대규모 컨텍스트 학습 완주 + PathoBench 5-fold CV. 전문은 [`history.md`](history.md) §12.

## 50. 2026-08-07 — v34-1536 추가 평가: Musk 패딩 브리지(타일), ICI(랜덤), PathoBench 17-task 전체 CV — **아카이브됨**

v34-1536 추가 평가 — Musk 타일 패딩 브리지, ICI(랜덤), 17-task CV. 전문은 [`history.md`](history.md) §12.

## 51. 2026-08-07 — PathoBench 원본 검증: 로컬 cptac_ccrcc CSV는 bc_therapy의 잘못된 복사본 — **아카이브됨**

⚠️ 로컬 `cptac_ccrcc_*` CSV가 `bc_therapy` 복사본으로 확정 — 데이터 출처 검증 교훈. 전문은 [`history.md`](history.md) §2.

## 52. 2026-08-07 — 실제 ccrcc 평가 완료 + SEAL baseline 비교 + 공식 50-fold 평가 계획 — **아카이브됨**

실제 ccrcc 평가 + SEAL baseline 비교 + 공식 50-fold 계획. 전문은 [`history.md`](history.md) §12.

## 53. 2026-08-07 — v34 최종 확정 + 공식 50-fold 평가(SEAL 동일 프로토콜) 진행 — **아카이브됨**

v34 최종 확정 + 공식 50-fold(SEAL 동일 프로토콜) 진행. 전문은 [`history.md`](history.md) §12.

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

## 57. 2026-08-07 — 50-fold 재개 전 진단: 5-fold CV의 case leakage로 lscc_arid1a 0.908이 부풀려짐 — **아카이브됨**

5-fold CV의 **case leakage**로 lscc_arid1a 0.908이 부풀려짐 → 공식 50-fold 0.462가 정직한 값. 전문은 [`history.md`](history.md) §2.

## 58. 2026-08-07 — v35 설계 확정: rare-instance 제거 + context/query 공통 chunk + 대형화 — **아카이브됨**

v35 설계 확정 — rare 제거 + 공통 chunk + 대형화. 전문은 [`history.md`](history.md) §13.

## 59. 2026-08-07 — v35 제안서 비판적 재검토(rev.2) + 정확 스트리밍 구현 + v35 학습 시작 — **아카이브됨**

v35 제안서 rev.2 재검토 + 정확 스트리밍 구현. 전제 3건이 반증됐다. 전문은 [`history.md`](history.md) §13.

## 60. 2026-08-08 — v35-16384 50ep 완주 + 메모리/val plateau 진단 + v35 공식 50-fold 평가(EGFR·PIK3CA 완료) + SEAL 비교 — **아카이브됨**

v35-16384 완주 + 메모리/val plateau 진단 + 공식 50-fold 평가. 전문은 [`history.md`](history.md) §13.

## 61. 2026-08-08 — P0-b 게이트 통과 + rare branch 제거 (rev.2 step 5) — **아카이브됨**

`rare_logits=0` ablation이 |Δpooled| **0.0009 < 0.003**으로 게이트를 통과해 rare 분기를 제거했다
(`meta_enable_rare_evidence: false`, 코드 삭제가 아니라 강제 0 — ckpt 호환·가역). 이후 모든 arm이
rare-free이므로 **평가는 반드시 그 arm의 훈련 config로** 해야 한다. 전문: [`history.md`](history.md).

## 62. 2026-08-08 — v36 제안서 비판적 재검토 + P0-slots 무료 probe (Q1 확정 / Q2 보류) — **아카이브됨**

v36 제안서 재검토 + P0-slots probe. **진단은 유효하나 처방은 §65가 반증** — routing softmax가 길이 1 축에 걸려 무력(계약은 [`agent_handoff.md`](agent_handoff.md)). 전문은 [`history.md`](history.md) §14.

## 63. 2026-08-08 — current_architecture v34 개편 검토 + bf16-mixed 계약 실제 강제 — **아카이브됨**

`configs/trainer/default.yaml`이 precision을 설정하지 않아 v34/v35가 fp32로 조용히 학습됐던 것을
확인하고 bf16-mixed를 예외 없이 강제했다(`tests/test_precision_contract.py`). 계약 본문은
[`agent_handoff.md`](agent_handoff.md) §3.4에 있다. 전문: [`history.md`](history.md).

## 64. 2026-08-08 — 평가도 bf16-mixed 강제 + 폴드 단위 context 캐싱(bit-identical, 7.1×) + bc_therapy/er_status 기본 평가 확정 — **아카이브됨**

평가도 bf16-mixed 강제 + 폴드 단위 context 캐싱(bit-identical, 7.1×). 계약은 [`agent_handoff.md`](agent_handoff.md). 전문은 [`history.md`](history.md) §3.

## 65. 2026-08-09 — v36 Q1 / v37 두 arm 평가 완료: **둘 다 게이트 미달**, 40→1 압축은 원인이 아니었다 — **아카이브됨**

v36 Q1 / v37 **둘 다 게이트 미달** — 40→1 압축은 원인이 아니었다. 전문은 [`history.md`](history.md) §14.

## 66. 2026-08-09 — ridge ablation (v38): **G-2 global ridge는 무기여 / P-2·CV-1은 제거 시 학습 붕괴** — **아카이브됨**

ridge ablation(v38) — G-2 무기여 / P-2·CV-1은 제거 시 학습 붕괴. 전문은 [`history.md`](history.md) §15.

## 67. 2026-08-09 — v39 수치 안정화: **역효과**(clipping이 −0.0317) + LR 가설 반증 — **아카이브됨**

v39 수치 안정화 **역효과** — clipping이 −0.0317. clipping 금지. 전문은 [`history.md`](history.md) §15.

## 68. 2026-08-09 — 분기 기여도 진단 → **CV-only 성공**: 6개 분기 중 2개만 남겨도 동률 — **아카이브됨**

분기 기여도 진단 → **CV-only 성공**. 6개 분기 중 2개만 남겨도 동률. 전문은 [`history.md`](history.md) §16.

## 69. 2026-08-09 — covariance sketch 기저 진단: **label-free 축 8개 전부 무효**, 차원만 유효 — **아카이브됨**

covariance sketch 기저 진단 — **label-free 축 8개 전부 무효**, 차원만 유효(대역폭 고정 시). 전문은 [`history.md`](history.md) §17.

## 70. 2026-08-09 — v41: er_status 0.7303 (**+0.031**) — 이득의 정체는 차원이 아니라 대역폭·CV-2  ⚠️ **§71이 정정: 10개 task로 넓히면 SEAL 상회 주장은 성립하지 않는다** — **아카이브됨**

v41 er_status 0.7303 — 이득은 차원이 아니라 대역폭·CV-2. ⚠️ §71이 정정. 전문은 [`history.md`](history.md) §17.

## 71. 2026-08-09 — SEAL 10개 task 전면 평가: **일반화 실패**, er_status는 가장 유리한 task였다 — **아카이브됨**

SEAL 10개 전면 평가 — **일반화 실패**, er_status가 가장 유리한 task였다. 판정 기준 변경. 전문은 [`history.md`](history.md) §17.

## 72. 2026-08-09/10 — 세션 요약: CV-2 손잡이 소진, 소스 prune, 학습 2.4배 가속, 계보 B 재설계 — **아카이브됨**

세션 요약 — CV-2 손잡이 소진, 소스 prune, 학습 2.4배 가속, 계보 B 재설계. 전문은 [`history.md`](history.md) §20.

## 73. 2026-08-09 — config로 끄기만 했던 5개 분기를 소스에서 삭제 (−11,285줄) — **아카이브됨**

config로 끄던 5개 분기를 소스에서 삭제(−11,285줄). ⚠️ prune 이전 ckpt는 `ICF_pre_prune` worktree 필요. 전문은 [`history.md`](history.md) §20.

## 74. 2026-08-10 — 학습이 평가용 ragged 경로를 타고 있었다 (74.2 → 31.3 ms/step) — **아카이브됨**

학습이 평가용 ragged 경로를 타고 있었다(74.2 → 31.3 ms/step). 전문은 [`history.md`](history.md) §20.

## 75. 2026-08-10 — v42 subspace_rank 2/4: 무효 — **아카이브됨**

v42 subspace_rank 2/4 — 무효. 전문은 [`history.md`](history.md) §20.

## 76. 2026-08-10 — v43/v44 identity margin: **기각**, tanh 유지 — **아카이브됨**

v43/v44 identity margin **기각**, tanh 유지. 전문은 [`history.md`](history.md) §20.

## 77. 2026-08-10 — v45 paired_head: 동률, 그러나 라벨 대칭성을 얻었다 — **아카이브됨**

v45 paired_head — 동률이나 라벨 대칭성을 구성으로 획득. 전문은 [`history.md`](history.md) §20.

## 78. 2026-08-10 — CV-1의 dual(kernel) ridge는 옳다 — **아카이브됨**

CV-1의 dual(kernel) ridge는 옳다 — 근거는 bag 수 ≪ 특징 수, 실측 30배. 전문은 [`history.md`](history.md) §20.

## 79. 2026-08-10 — 계보 B (Encoder+Ridge): 첫 판본은 설계 오류, 재설계 후 궤적 반전 — **아카이브됨**

계보 B(Encoder+Ridge) — 첫 판본은 설계 오류, 재설계도 SEAL 하락. **문제는 일반화**. 전문은 [`history.md`](history.md) §20.

## 80. 다음 세션이 할 일

당시(2026-08-10) 기준 Action Plan으로 소진됐다. 현행 다음 작업은 **§99-5**와 **§100-5**.

## 81. 2026-08-10 — episode 내부 bag별 cardinality + zero-padding/mask — **아카이브됨**

episode 내부 bag별 cardinality + zero-padding/mask 계약. 전문은 [`history.md`](history.md) §21.

## 82. 2026-08-10 — factorized response/XOR 데이터 arm — **아카이브됨**

factorized response/XOR 데이터 arm — v57~v61 전부 v41 미달. 전문은 [`history.md`](history.md) §21.

## 83. 2026-08-10 — v61: random MLP를 orthogonal linear projection으로 교체 — **아카이브됨**

v61 — random MLP를 orthogonal linear projection으로 교체. 전문은 [`history.md`](history.md) §21.

## 84. 2026-08-10 — v62 Linear-16 + CV-1 K128 hybrid — **아카이브됨**

v62 Linear-16 + CV-1 K128 hybrid. 4,096 cap을 생성 전에 적용(OOM 교훈). 전문은 [`history.md`](history.md) §21.

## 85. 2026-08-11 — v62–v66 hybrid 결과, branch 명칭 확정, 4-pop DDP8 완료 — **아카이브됨**

v62–v66 hybrid 결과, branch 명칭 확정, 4-pop DDP8 완료. 전문은 [`history.md`](history.md) §21.

## 86. 2026-08-11 — v66 기각 + CV의 raw bag-mean 승격 — **아카이브됨**

v66 기각 + **CV의 raw bag-mean 승격**(canonical CV). 현행 스펙은 [`current_architecture.md`](current_architecture.md) F. 전문은 [`history.md`](history.md) §22.

## 87. 2026-08-11 — Dispersion Distance와 v70 relation MLP: synthetic 일반화 신호 확인 — **아카이브됨**

Dispersion Distance와 v70 relation MLP. 현행 스펙은 [`current_architecture.md`](current_architecture.md) G. 전문은 [`history.md`](history.md) §22.

## 88. 2026-08-11 — v71–v74 ablation 완료, v74 CV+DD+CT를 활성 baseline으로 확정 — **아카이브됨**

v71–v74 ablation 완료, v74 CV+DD+CT 확정. 현행 스펙은 [`current_architecture.md`](current_architecture.md) H. 전문은 [`history.md`](history.md) §22.

## 89. 2026-08-11 — v76 learnable P를 활성 baseline으로 승격 — **아카이브됨**

v76 learnable P를 활성 baseline으로 승격(fixed-P 0.6731 → 0.6748). 전문은 [`history.md`](history.md) §22.

## 90. 2026-08-12 — provisional v77-pop-residual 기각, synthetic 난이도 축 분해 — **아카이브됨**

provisional v77-pop-residual 기각(0.6750), synthetic 난이도 축 분해. 전문은 [`history.md`](history.md) §23.

## 91. 2026-08-12 — ClassSep sweep 완료 — **아카이브됨** ([history.md §24](history.md))

⚠️ 이 절의 Medium 행(0.6823)은 오기였고, **§105-3**이 0.6881로 정정했다. 결론도 §106-2가 갱신했다 — **Medium이 Hard보다 +0.0053 낫다**.

---

## 92. 2026-08-12 — Active: Hard latent dimension 2/4/8/16 ablation, 8×GPU — **아카이브됨**

Hard latent dimension ablation — L2/L4/L8/L16 = 0.6775/0.6781/0.6771/0.6662, L32(=v77) 0.6873. 전문은 [`history.md`](history.md) §23.

## 93. 2026-08-12 — Active: Hard fixed 3-layer MLP-bank sweep — **아카이브됨**

Hard fixed 3-layer MLP-bank sweep — M=128~4096 최고 0.6779(M=1024), v77 미달. 전문은 [`history.md`](history.md) §23.

## 94. 2026-08-12 — Active: Hard 50:50 infinite-linear + MLP-1024 — **아카이브됨**

Hard 50:50 infinite-linear + MLP-1024 — 0.6755, 기각. 전문은 [`history.md`](history.md) §23.

## 95. 2026-08-12 — Active: Hard orthogonal + learned ridge calibration — **아카이브됨**

Hard orthogonal + learned ridge calibration — 0.6840, 기각. 전문은 [`history.md`](history.md) §23.

## 96. 2026-08-12 — architecture/handoff SSOT 정리 — **아카이브됨**

architecture/handoff SSOT 정리. 전문은 [`history.md`](history.md) §23.

## 97. 2026-08-12 — Active: Hard v76 warm-start, 2k–16k ragged training — **아카이브됨**

Hard v77 warm-start 2k–16k ragged — epoch 34 best 0.6885(+0.0012). ⚠️ 첫 launch는 CUDA prefetch 중첩으로 OOM. 전문은 [`history.md`](history.md) §23.

## 98. 2026-08-12 — Hard v76을 canonical v77 baseline으로 승격 — **아카이브됨** ([history.md §24](history.md))

여기서 정한 0.6873은 `epoch=048` validation-best다. 활성 baseline 숫자는 **§104**가 `epoch 49 = 0.6880`으로 대체했다.

---

## 99. 2026-08-12 — 판정 프로토콜: fold-paired Δ + bootstrap CI (사용자 지시)

지금까지 arm 판정은 task별 `fold-mean AUROC` 점추정 10개를 평균한 macro끼리 빼서 했다. CI도
pairing도 bootstrap도 없었다. §65 시절 er_status 단일 task에서는 fold-paired 20k bootstrap을
썼으나, §71에서 판정 기준이 SEAL 10-task macro로 넓어질 때 **pairing과 CI가 함께 넘어오지
않았다**. 문제는 크기다 — er_status가 `fold-mean 0.7023 ± 0.0903`인데 §98 판정표의 Δ는
0.0012~0.0118로 fold 산포가 판정 대상 효과의 8~75배다.

사용자 지시로 앞으로 모든 arm 비교는 **fold별 차이를 통계 단위로** 삼는다.

- 도구: `scripts/compare_arms_paired.py` (신규). GPU 불필요, 재평가 불필요 —
  `test_pathobench.py`가 이미 저장한 `predictions/pathobench_{task}_{tag}_official50_bf16.pt`를
  읽는다.
- 방법: `d_f = auroc_arm(f) − auroc_base(f)`. `d_f`가 이미 차분이므로 fold resample은 구성상
  paired다. fold 20,000회 resample → percentile CI. macro는 task별로 독립 resample한 뒤
  10개 평균을 replicate로 삼는다.
- pairing은 가정하지 않고 **검증**한다: fold 수, `fold_indices`, fold별 `slide_id` 순서, label이
  모두 일치해야 하며 어긋나면 unpaired 폴백이 아니라 `PairingError`다. AUROC는 양쪽을 같은
  코드(`auroc_rows`)로 재계산하고 저장값과 교차검증한다.
- 사용법: `python scripts/compare_arms_paired.py --baseline <TAG> --arm <TAG> [--arm <TAG> ...]`

### 1. §98 판정표 재검증 — 4건 모두 유지

macro 점추정은 §98과 정확히 재현됐다(0.6873 / 0.6885 / 0.6840 / 0.6779 / 0.6755).

| arm | Δmacro | 95% CI | 상승 task | 재판정 |
|---|---:|---|---:|---|
| large-ragged 2k–16k warm-start | +0.0012 | [−0.0008, +0.0032] | 5/10 | CI가 0 포함 — 동률 확증 |
| learned ridge λ + logit scale | −0.0033 | [−0.0058, −0.0010] | 5/10 | CI가 0 제외 — 기각 확증 |
| MLP bank M=1024 | −0.0094 | [−0.0135, −0.0053] | 3/10 | CI가 0 제외 — 기각 확증 |
| 50:50 fresh-linear + MLP-1024 | −0.0118 | [−0.0159, −0.0075] | 2/10 | CI가 0 제외 — 기각 확증 |

CI 폭이 0.004~0.008로, fold 산포 ±0.09 대비 한 자릿수 배 이상 좁다. 점추정 판정이 결과적으로
전부 옳았지만, 그건 사후에 확인된 것이고 그 판정 시점에는 근거가 없었다.

### 2. large-ragged는 "동률"이 아니라 "재분배"다

macro Δ는 0이지만 개별 task 6개의 CI가 0을 제외한다.

| task | Δ | 95% CI | 이긴 fold |
|---|---:|---|---:|
| bc_therapy grade | +0.0111 | [+0.0068, +0.0157] | 37/50 |
| cptac_ccrcc VHL | +0.0090 | [+0.0038, +0.0139] | 34/50 |
| cptac_luad TP53 | +0.0074 | [+0.0023, +0.0123] | 34/50 |
| cptac_brca TP53 | +0.0068 | [+0.0027, +0.0109] | 31/50 |
| cptac_luad EGFR | +0.0053 | [+0.0020, +0.0086] | 31/50 |
| **cptac_ccrcc BAP1** | **−0.0179** | [−0.0282, −0.0070] | 15/50 |

5개 task를 실제로 올리고 BAP1 하나가 그것을 상쇄한다. "+0.0012라 파생 실험 유지"보다 정보량이
크다. **대형 bag에서 BAP1만 무너지는 이유**가 별도 조사 대상이다.

ridge calibration 기각의 주동인은 PIK3CA −0.0272 [−0.0407, −0.0160] (8/50)이며 STK11 −0.0093,
EGFR −0.0059가 뒤따른다. er_status는 반대로 +0.0081이었다.

### 3. latent sweep 비단조성은 fold 노이즈가 아니다

| 비교 | Δmacro | 95% CI | 상승 task |
|---|---:|---|---:|
| L16 − L8 | −0.0108 | [−0.0154, −0.0063] | 4/10 |
| L32 − L8 | +0.0103 | [+0.0054, +0.0151] | 9/10 |

두 CI 모두 0을 제외하므로 L16의 딥은 **주어진 checkpoint 기준으로는 견고**하다. fold 노이즈가
배제되었으니 남은 설명은 ⓐ latent_dim 효과가 실제로 들쭉날쭉하다, ⓑ realization(학습 seed)
노이즈 둘뿐이다. ⓑ는 pairing으로 줄일 수 없는 축이다(두 학습 run에 공유 난수가 없어 상쇄할
공통항이 없다). **L8/L16/L32 각 2 seed 추가**가 이 둘을 가른다.

### 4. 한계 — CI를 하한으로 읽을 것

- 50 fold가 166장 슬라이드를 겹쳐 쓰므로 fold를 독립 표본으로 보는 bootstrap은 분산을
  **과소추정**할 가능성이 크다.
- fold 노이즈만 다룬다. **학습 seed 노이즈**(arm당 checkpoint 1개)와 **task 선택 노이즈**(고정
  10개)는 포함하지 않는다. 스크립트가 출력 말미에 이 한계를 명시한다.

### 5. 다음 Action

1. seed 반복 — v77 + L8/L16/L32. arm당 학습 약 15분 + 평가 1–2분. §3의 ⓐ/ⓑ를 가르고, macro
   seed std를 얻어 앞으로의 +0.005 게이트에 분모를 준다.
2. BAP1이 large-bag에서만 무너지는 원인 진단(§2).

## 100. 2026-08-12 — v78 DD quadratic-form gradient path (구현) — **아카이브됨** ([history.md §24](history.md))

실행 결과와 기각 판정은 §102-5·§103-1, epoch 49 재채점은 §105-4에 있다. DD 미분 금지 계약은 `agent_handoff.md` 상단이 SSOT다.

---

## 101. 2026-08-12 — 문서 압축: §2~§97 본문을 history.md로 아카이빙 — **아카이브됨** ([history.md §24](history.md))

---

## 102. 2026-08-12 — configs 정리: 루트 67개 → 2개 — **아카이브됨** ([history.md §24](history.md))

config 관리 규칙은 `agent_handoff.md` §7이 SSOT다. 이후 v80~v82 arm이 추가돼 루트 config는 다시 늘었다(§106-6).

---

## 103. 2026-08-12 — v78 무가중 기각(단조 악화 확정), v79 dual projection 진행 중

### 1. v78 weight 스윕 완결 — DD gradient의 방향이 해롭다

사용자 지시로 `dd_projection_gradient_weight`를 제한하지 않은 arm을 돌렸다. 결과가 balanced와
합쳐져 **단조 용량-반응**이 됐다.

| weight | SEAL macro | fold-paired Δ vs v77 | 95% CI | 상승 task | 판정 |
|---:|---:|---:|---|---:|---|
| 0 (v77) | 0.6873 | — | — | — | baseline |
| 0.02 (≈1/52) | 0.6869 | −0.0004 | [−0.0021, +0.0013] | 5/10 | 구별 불가 |
| **1.0 (무가중)** | **0.6826** | **−0.0047** | **[−0.0082, −0.0013]** | 3/10 | **CI 0 제외 — 유의하게 나쁨** |

두 arm 직접 비교도 **−0.0043 [−0.0075, −0.0013]**로 CI가 0을 제외한다. 세 점이 단조 감소하고
그 감소가 통계적으로 실재한다.

**해석이 확정됐다.** balanced가 null일 때 남아 있던 두 해석(ⓐ DD가 보탤 게 없다 / ⓑ 0.02가 너무
조였다) 대신 **제3의 답**이다 — **DD는 P를 실제로 움직이며, 그 방향이 전체 readout에 해롭다.**
`cos(grad_CV, grad_CV+DD) = −0.068`(거의 직교, 살짝 음수)과 부합한다. 기제가 P에 도달함을
테스트로 단정하고 크기도 맞춰둔 상태였으므로 §66 함정에 걸리지 않는다 — **가설 기각**이다.

⚠️ **task별로 보면 §71 패턴이 재현된다**: 무가중은 er_status만 **+0.0277**(44/50)로 크게 올리고
PIK3CA −0.0349, grade −0.0093, STK11 −0.0079를 떨어뜨린다. er_status 단독으로 보면 "개선"으로
오판할 arm이었다.

**`train_dd_projection`은 되살리지 말 것.** 계약과 근거는 `current_architecture.md` **G-5**.

### 2. DD 명세를 재정리했다 (current_architecture G절)

DD 관련 서술이 여러 절에 흩어져 있고 "어느 P를 읽는지"가 명시되지 않아 arm 간 차이를 읽을 수
없었다. G절을 DD 전용 명세로 다시 썼다.

- **G-0 (신설)**: DD는 자기 사영을 갖지 않고 **CV가 만든 covariance를 재사용**한다. 따라서 "DD의
  subspace"는 arm마다 다르다 — v74 fixed / **v77 CV가 학습한 P** / v78 같음+gradient 개방 /
  **v79 fixed(분리)**. 표로 고정했다.
- **G-4 (신설)**: rank-1 방향을 **어느 arm에서도 미분하지 않는 이유**. eigh backward의
  `1/(λ_i−λ_j)`와 hard argmax 불연속, 그리고 ⚠️ **shrinkage `+0.25τI`가 고윳값을 균일하게 밀어
  간격을 바꾸지 않으므로 backward를 전혀 보호하지 못한다**는 점. `nonfinite_gradient_policy: zero`
  때문에 이 실패가 **조용하다**는 경고와, 미분 가능 우회(Newton–Schulz + `A²` power iteration,
  미구현)도 적었다.
- **G-5 (신설)**: 위 1절의 스윕 결과와 "되살리지 말 것".
- **G-6 (신설)**: **미측정 항목 2건** — ⓐ **학습된 P에서의 DD-only 성능**(G-2의 0.5862는 fixed P
  수치다). DD는 학습 파라미터가 없으므로 v77의 학습된 P로 DD-only를 채점하면 "DD가 CV의 학습된
  subspace에서 손해를 보는가"를 **학습 없이** 확인할 수 있다 — v79의 전제를 값싸게 검증하는
  진단이며 미실행이다. ⓑ DD 전용 learnable 사영(v79는 fixed로 되돌리기만 했다).

### 3. v79 dual projection — 진행 중

사용자 지시 설계: **learnable CV + fixed CV + fixed DD + CT → 16 feature MLP**. v78처럼 중재하지
않고 **공유를 끊는다**.

- 클래스 `DualProjectionCVDDCTMLPModel`, **`architecture_version = 56`** (v77 ckpt와 strict-load
  **비호환**). 상세 명세는 `current_architecture.md` **Active-6**.
- descriptor를 `[cov_learnable 8,256, mean 1,536, cov_fixed 8,256]` = **18,048**로 확장하고 세
  block을 각각 독립 context-only center/scalar-RMS로 정규화한다. raw bag mean은 사영과 무관하므로
  두 CV branch가 공유한다.
- **fixed-P CV를 독립 evidence block으로 남긴 것**이 두 번째 요점이다. fixed P는 옛 기본값이
  아니라 **v41_K128이 0.6940을 낸 기저**이고 그것이 여전히 역사적 전체 최고다. head가 학습된
  subspace와 고정 subspace를 저울질하게 한다.
- `train_dd_projection`은 **ValueError로 거부**한다 — DD가 buffer를 읽으니 조용한 no-op이 될 뿐이다.
- trainable **197,185** (P 196,608 + head 577).
- config `configs/train_v79_dual_projection_1536.yaml`, runner
  `scripts/run_v79_dual_projection.py`, tag `v79_dual_projection_best`.
- artifacts `checkpoints/20260812_v79_dual_projection/`, `logs/20260812_v79_dual_projection/`.
- **runner PID/PGID `2329816`** (PPID 1, 완전 이탈), GPU 0–3, DDP rank 4개. first-step peak
  allocated 10.69 GiB (v77/v78과 동일). 문서 갱신 시점 epoch 14, val_ce 0.2017.
  ⚠️ 새 16-input head가 랜덤 초기화라 초반 val_ce가 v77보다 높다 — 판정 근거가 아니다.

**검증**: 신규 테스트 4개 중 둘이 설계의 구조적 주장을 고정한다 — ⓐ P를 흔들어도 `cov_fixed`와
`mean` block이 **정확히 불변**(DD가 CV의 subspace를 타지 않음), ⓑ v79의 fixed block이 독립
`CovarianceMeanDDCTMLPModel`의 CV와 **수치적으로 동일**(재파라미터화가 아니라 진짜 v41-스타일 CV).
전체 suite **153 tests**, 실패는 기존 1건(`test_mlp_manifold_bank.py`의 `pytest` import).
CUDA bf16 smoke(60 bags × 4,096 cells) loss 0.5056, P grad norm 1.78e-01 finite, peak 2.38 GiB.
numeric-type·precision 계약 7개 통과.

### 4. v79 결과 — 기각. 세 arm 중 가장 나쁘다

| arm | macro | Δ vs v77 | 95% CI | 상승 task |
|---|---:|---:|---|---:|
| v77 (baseline) | 0.6873 | — | — | — |
| v78 balanced (0.02) | 0.6869 | −0.0004 | [−0.0021, +0.0013] | 5/10 |
| v78 무가중 (1.0) | 0.6826 | −0.0047 | [−0.0082, −0.0013] | 3/10 |
| **v79 dual projection** | **0.6768** | **−0.0105** | **[−0.0137, −0.0074]** | 3/10 |

PIK3CA −0.0518은 이번 세션 단일 task 최대 하락이고 VHL은 0.4166으로 랜덤에서 더 멀어졌다.
er_status만 또 +0.0168로 오른다(§71 패턴 세 번째).

**진단 ⓐ — 과소학습이 아니다. 반대다.** v79 best val_ce **0.1687**(epoch 48)로 v77의 0.1697보다
**좋다**. 즉 합성 val이 개선되는 동안 SEAL이 −0.0105 떨어졌다. 이 리포의 대표 병리가 세 번째로
재현된 것이다 — v54(§79-6, 합성 최고 = SEAL 최저), mixed manifold(§94, val CE 0.2218 개선 /
SEAL 0.6755 하락), 그리고 v79. **50 epoch이 부족한 게 아니라 합성에 더 잘 맞춘 것이 손해였다.**

**진단 ⓑ — head가 선택하지 않고 분산시켰다.** 학습된 head 1층의 block별 column norm share는
CV(learnable) 31.4% / CV(fixed) 26.7% / DD(fixed) 25.5% / CT 16.5%로 거의 균등하다. "fixed CV가
learnable CV를 밀어낸다"가 아니라 **16개 상관된 입력에 weight가 퍼졌다** — 두 CV branch는 mean
block을 공유하고 covariance 정보도 중복된다. ⚠️ column norm은 **거친 대리 지표**다(feature마다
스케일이 달라 곧 기여도가 아니다). 방향성 증거로만 읽을 것.

### 5. 이 축은 소진된 것으로 본다

v78 balanced → 무가중 → v79가 **−0.0004 → −0.0047 → −0.0105**로 단조 악화한다. gradient 개방 /
무가중 / 완전 분리 세 방식으로 CV·DD·사영 배선을 건드렸고 셋 다 졌으며 **건드린 정도가 클수록 더
졌다**. headroom이 이 배선에 있지 않다는 결론이 세 방향에서 수렴한다. 새 arm을 이 축에서 더
설계하지 말 것.

### 6. 다음 Action

1. **seed 반복 — 이제 선행 조건이다.** §99-3·§102-6·§103에서 세 번 미뤘다. 판정 대상 Δ가
   −0.0004~−0.0105인데 realization 노이즈 크기를 모른다. v77 3 seed(약 51분)로 macro seed std를
   먼저 확보하고, 겸해서 L8/L16/L32로 §99-3의 ⓐ/ⓑ를 가른다. **이것 없이는 다음 arm의 판정도
   공허하다.**
2. **G-6ⓐ 진단** — 학습된 P에서의 DD-only(학습 불필요). v79가 졌으므로 "DD가 CV의 학습된
   subspace에서 손해를 보는가" 자체는 아직 답이 없다.
3. **task-side 진단** — VHL 랜덤 이하(§0 열린 과제 2), BAP1 large-bag 붕괴(§99-2).
   배선 축이 막혔으므로 남은 레버는 여기와 §99-2/§0의 reliability feature 쪽이다.
   ⚠️ **§104-5가 이 항목의 근거를 약화시켰다** — BAP1의 large-bag 붕괴 −0.0179는 시드만 바꿔도
   나오는 크기(−0.0402)보다 작다.

---

## 104. 2026-08-12 — baseline을 epoch 49 = 0.6880으로 확정, v80 shallow MLP 기각, seed std 실측

### 0. 이 절이 정한 것 (요약)

| 항목 | 결정 |
|---|---|
| **활성 baseline checkpoint** ⚠️ §107이 대체 | `checkpoints/20260812_v76_classsep_sweep/hard/periodic-epoch=049-val_ce_loss=0.1717.ckpt` |
| **활성 baseline 수치** ⚠️ §107→§109가 대체 | **SEAL 10-task macro `0.6880`** (tag `v77_hard_ep49`, DDP4 1 seed) — §107은 v82 Medium 4 seed 0.6835로, §109는 v83 linear head 4 seed **0.6880**(숫자만 같은 별개 레짐)으로 교체했다 |
| 이전 표기 0.6873의 정체 | 같은 run의 `epoch=048` **validation-best**. Δ +0.0007 [+0.0000, +0.0014] |
| **채점 규칙** | **epoch 49 고정.** validation-best 선택은 판정에 쓰지 않는다 |
| **macro seed std** | **0.0051** (epoch 고정, n=4) / 0.0023 (val-best 선택 시) |
| **판정 게이트** | macro Δ **≈0.010(2σ)** 이상일 때만 단정. 그 미만은 "판정 불가" |
| **task별 CI** | **판정 근거로 쓰지 않는다** (§104-5) |
| v80 shallow MLP | **기각.** 4 seed 평균 0.6722, Δ −0.0158 |

### 1. v80 arm — Hard에서 한 번도 안 돌린 칸이었다

`manifold_mode`에서 이 리포의 "infinite"는 **bank size 무한 = episode마다 새로 뽑음**을 뜻한다.
Hard `[0.2,0.8]`에서 돌린 manifold arm은 **유한 bank뿐**이었다(`mlp_bank` M=128~4096, `mixed` 50:50).
`nonlinear`(fresh MLP per episode, M=∞)는 ClassSep baseline `[1.0,2.0]` 시절 v72(0.6709)가 마지막이고
**Hard에서는 미측정**이었다. ClassSep이 유일하게 먹힌 레버였으므로 난이도별 manifold 순위는
확립된 것이 아니었다.

깊이는 사용자 지시로 **가장 얕은 진짜 MLP**를 썼다. `mlp_num_layers`는 weight 행렬 개수이고
GELU는 마지막 층에 붙지 않는다(`_map_to_manifold`):

| `mlp_num_layers` | 차원 | GELU | 판정 |
|---|---|---:|---|
| 1 | `[32→1536]` | **0** | MLP가 아니다. orthogonal의 비-isometric 열등판 — 쓰지 말 것 |
| **2** | `[32→96→1536]` | **1** | **가장 얕은 진짜 MLP. v80이 이것** |
| 3 | `[32→96→96→1536]` | 2 | bank sweep이 쓴 깊이 |

config `configs/train_v80_hard_shallow_mlp_1536.yaml`(DDP4 정의) +
`..._1gpu.yaml`(4-seed 배치용). v77 대비 바뀐 resolved 키는 **정확히 두 개**
(`manifold_mode: orthogonal→nonlinear`, `mlp_num_layers: 3→2`)이고 타입도 int로 확인했다(§79 함정).
사전 검증: weight shape `[(96,32),(1536,96)]`, superposition gap 0.3965(선형이면 0),
episode마다 다른 맵. 셀 특징은 `synthetic_data.py:792`에서 L2 정규화되므로 nonlinear의 출력
스케일 차이는 모델 입력에서 사라진다 — 스케일 confound 없음.

### 2. 채점을 epoch 49로 고정한 이유 — validation-best가 과소학습 지점을 골랐다

v80 4 seed의 validation-best epoch가 **44 / 16 / 11 / 49**로 흩어졌다. val_ce 스프레드가
0.2276~0.2312(0.0036)뿐이어서 곡선이 거의 평평하고, 어느 epoch이 "best"로 뽑히는지가 사실상
무작위였다. 사용자 지시로 전부 `periodic-epoch=049`로 다시 채점했다.

| seed | ep49 macro | val-best macro | 차이 | val-best epoch |
|---|---:|---:|---:|---:|
| 42 | 0.6688 | 0.6667 | +0.0021 | 44 |
| 43 | **0.6795** | 0.6706 | **+0.0089** | 16 |
| 44 | 0.6686 | 0.6690 | −0.0004 | 11 |
| 45 | 0.6720 | 0.6720 | ±0.0000 | 49 (동일 ckpt) |
| 평균 | **0.6722** | 0.6696 | +0.0026 | |

**seed 43이 결정적이다.** val_ce는 epoch 16(0.2276)이 epoch 49(0.2290)보다 0.0014 좋아 보였지만
SEAL은 **−0.0089 손해**였다. §69-6("합성 지표는 평평한데 실제 task는 계속 오른다")이 이 arm에서
재현됐고, val-best 선택이 seed 43을 과소학습 지점에서 잘라냈다.

**baseline 쪽은 둔감했다**: v77 ep49 0.6880 vs val-best(epoch 48) 0.6873, Δ **+0.0007
[+0.0000, +0.0014]**. 즉 baseline 숫자는 규칙을 바꿔도 흔들리지 않는다. v77의 val_ce는
0.1697 부근에서 안정적이고 v80은 평평했다는 차이다.

⚠️ **주의**: epoch 고정은 분산을 줄이지 않았다. 오히려 **늘렸다**(§104-3). validation-best 선택은
각 시드가 자기 궤적의 최고점을 고르므로 시드 간 차이를 부분적으로 상쇄한다 — 대신 seed 43 같은
편향을 만든다. 편향을 없애고 분산을 드러내는 쪽을 택한 것이다.

### 3. macro seed std 실측 — §103-6의 선행 조건 해소

동일 config·동일 50 epoch, `SEED`만 42/43/44/45로 바꾼 4 run(1-GPU, GPU 0–3 병렬):

| 채점 규칙 | mean | **seed std** | range |
|---|---:|---:|---:|
| epoch 49 고정 | 0.6722 | **0.0051** | 0.0109 |
| validation-best | 0.6696 | 0.0023 | 0.0053 |

⚠️ **n=4라 std 추정 자체가 매우 불확실하다**(자유도 3이면 std의 95% 구간이 대략 0.6~2.9배).
"epoch 고정이 std를 2.2배 늘렸다"는 방향성으로만 읽을 것.

`SEED`는 `train.py`가 `seed_everything`으로 라우팅해 **모델 초기화와 training episode 스트림**을
움직인다. training dataset은 `seed: null` 유지(`episode_dataset: true`라 CLI seed로 덮이지 않는다),
val/test는 50042/60042 고정 — 네 시드가 **동일한 val/test episode**로 채점됐다.

### 4. 판정 게이트가 엄격해졌다 — 기존 판정 3건이 "판정 불가"로 내려간다

epoch-고정 seed std **0.0051**을 분모로 과거 Δ를 다시 읽으면:

| arm | Δmacro | σ 배수 | 재해석 |
|---|---:|---:|---|
| v78 balanced (0.02) | −0.0004 | 0.1σ | 노이즈 (기존 "동률"과 일치) |
| large-ragged warm-start | +0.0012 | 0.2σ | 노이즈 (미승격이 옳았다) |
| ridge calibration | −0.0033 | 0.6σ | **판정 불가** — 기존 "CI 0 제외 → 기각"이 성립하지 않는다 |
| v78 무가중 (1.0) | −0.0047 | 0.9σ | **판정 불가** — 같은 문제 |
| MLP bank M=1024 | −0.0094 | 1.8σ | 경계 |
| v79 dual projection | −0.0105 | 2.1σ | 겨우 유지 |
| **v80 shallow MLP** | **−0.0158** | **3.1σ** | **기각 (4 seed)** |

**§103-5의 논거가 좁아진다.** "v78 balanced → 무가중 → v79가 −0.0004 → −0.0047 → −0.0105로
**단조 악화**하므로 축이 소진됐다"에서 중간 단계(−0.0047)가 seed 노이즈와 구분되지 않는다.
소진 결론 자체는 v79의 −0.0105(2.1σ)가 버티므로 방향은 유지하지만, **"단조성"을 증거로 인용하지 말 것.**

### 5. task별 fold-paired CI는 판정 근거로 쓸 수 없다 — 실측

**시드만 다른 두 run**(v80 seed42 vs seed45, val-best 채점)을 fold-paired로 비교했더니
**CI가 0을 제외하는 task가 6개** 나왔다. 처치는 없었다.

| task | Δ (시드 차이뿐) | 95% CI |
|---|---:|---|
| **cptac_ccrcc BAP1** | **−0.0402** | [−0.0666, −0.0156] |
| cptac_ccrcc VHL | +0.0324 | [+0.0089, +0.0570] |
| cptac_brca PIK3CA | +0.0259 | [+0.0019, +0.0500] |
| bc_therapy er_status | +0.0199 | [+0.0068, +0.0339] |
| cptac_luad STK11 | +0.0119 | [+0.0014, +0.0225] |
| bc_therapy grade | −0.0120 | [−0.0208, −0.0030] |
| MACRO | +0.0053 | [−0.0001, +0.0107] (0 포함 — macro는 정직했다) |

task별 seed std(val-best 채점, n=4)는 BAP1 0.0299 / PIK3CA 0.0237 / luad TP53 0.0225 /
VHL 0.0214 / grade 0.0191이고 평균 **0.0161** — macro(0.0023)의 7배다. **평균이 노이즈를 지운다.**
epoch-고정에서도 BAP1은 0.5461~0.6464(range 0.100)로 흔들린다.

⚠️ **따라서 §99-2의 "large-ragged는 동률이 아니라 재분배다"는 성립하지 않는다.** 그 판정의
근거는 task 6개의 CI 0 제외와 **BAP1 −0.0179**였는데, 시드만 바꿔도 같은 BAP1이 **−0.0402**
(2.2배)로 움직인다. pairing은 fold 노이즈만 잡고 realization 노이즈는 그대로 남긴다 —
§99-4가 한계로 적어둔 것이 실측된 것이다. **arm이 다른 두 run 사이의 task별 CI는 해석하지 말 것.**

### 6. v80 최종 판정 — 기각

epoch 49 규칙을 baseline에도 적용한 비교:

| arm | macro | Δ vs v77 ep49 | 95% CI | 상승 task |
|---|---:|---:|---|---:|
| **v77 Hard orthogonal ep49** | **0.6880** | — | — | — |
| v80 seed 42 | 0.6688 | −0.0192 | [−0.0240, −0.0142] | 3/10 |
| v80 seed 43 | 0.6795 | −0.0085 | [−0.0134, −0.0036] | 5/10 |
| v80 seed 44 | 0.6686 | −0.0194 | [−0.0241, −0.0149] | 2/10 |
| v80 seed 45 | 0.6720 | −0.0160 | [−0.0210, −0.0111] | 3/10 |
| **v80 평균** | **0.6722** | **−0.0158** | | |

네 시드 모두 CI가 0을 제외한다. v80 seed SE 0.0025에 v77의 seed 노이즈를 v80과 같다고 가정해
합성하면 SE≈0.0057, **t≈2.8**. ⚠️ **v77은 시드 1개뿐이므로 이 마진은 그 가정에 의존한다.**

사전 증거와 방향이 일치한다: bank sweep M=1024→4096 하강(0.6779→0.6649)의 M→∞ 외삽, 그리고
v72의 0.6709. **얕은 infinite MLP도 fresh orthogonal linear를 넘지 못한다 — Hard 난이도에서도
manifold 순위는 뒤집히지 않았다.**

⚠️ **남은 confound (판정을 뒤집을 정도는 아니나 명시해야 한다)**: v80은 **1-GPU**
(1024 step/epoch, effective batch 1), v77 baseline은 **DDP4**(rank당 256 step/epoch,
gradient 4개 평균)다. Lightning이 loader를 DistributedSampler로 감싸기 때문이다. 같은 LR에서
optimizer step이 4배, gradient 노이즈가 4배다. **−0.0158 전부를 manifold 효과로 귀속할 수 없다.**
순수 manifold 효과를 원하면 v77을 `..._1gpu` 레이아웃으로 4 seed 돌려야 한다(약 55분, 미실행).

### 7. 산출물

- 학습: `logs/20260812_v80_shallow_mlp_seeds/v80_shallow_mlp_seed4{2,3,4,5}.out`,
  ckpt `checkpoints/20260812_v80_shallow_mlp_seeds/v80_shallow_mlp_seed4{2,3,4,5}/`
  (4개 전부 `training completed successfully`). 1-GPU 4병렬로 학습 약 50분.
- 평가 태그: `v80_shallow_mlp_seed4{2,3,4,5}_best`(val-best), `..._ep49`(epoch 고정),
  `v77_hard_ep49`(baseline). 예측 90개 = 40+40+10, 로그 `logs/official50/*_<tag>.log`.
- 재확인:
  ```bash
  for tag in v77_hard_ep49 v80_shallow_mlp_seed42_ep49 v80_shallow_mlp_seed43_ep49 \
             v80_shallow_mlp_seed44_ep49 v80_shallow_mlp_seed45_ep49; do
    printf "%-32s " $tag
    grep -hoP 'fold-mean AUROC: \K[0-9.]+' logs/official50/*_${tag}.log \
      | awk '{s+=$1;k++} END{printf "%.4f (%d)\n", s/k, k}'
  done
  ```
- ⚠️ v80 config 2개는 기각된 arm이므로 다음 정리에서 v78 두 개와 함께
  `configs/archive/`로 이관할 것(§7 규칙).

### 8. 다음 Action

1. **v77 1-GPU 4 seed control** — v80의 batch-regime confound를 없애고, 동시에 baseline
   자신의 seed std를 얻는다(현재 v77은 시드 1개다). 약 55분, GPU 0–3.
2. **§99-2·§103-6의 task-side 항목 재검토** — BAP1 large-bag 붕괴와 VHL 랜덤 이하는 §104-5에
   비추면 근거가 약하다. 다시 세우려면 **arm마다 최소 3 seed**가 필요하다.
3. **판정 게이트 0.010을 넘길 만한 축을 찾을 것.** 지금까지의 arm은 대부분 이 게이트 아래에서
   싸웠다 — 미세 배선이 아니라 큰 레버(ClassSep이 그랬던 것처럼)를 찾아야 한다.

---

## 105. 2026-08-12 — 과거 판정 전수 감사 + 27개 arm epoch 49 재채점

### 0. 이 절이 뒤집은 것 (요약)

| 항목 | 이전 기록 | 재채점 후 (epoch 49) |
|---|---|---|
| **ClassSep Medium** | 0.6823 (§91) | **0.6881** — §91 표가 Very-hard 값을 잘못 복사했다 |
| **ClassSep 선택** | Hard가 최고, 단봉 | **Medium ≡ Hard** (Δ +0.0001 [−0.0022,+0.0024]). Hard 선택은 근거 없음 |
| **v74 → v76 승격** | +0.0017 | **+0.0004 [−0.0030,+0.0038] → 판정 불가.** learnable P 도입에 SEAL 근거가 없다 |
| ridge calibration | −0.0033, 기각 | **−0.0010 [−0.0021,+0.0002] → 기각 철회, 판정 불가** |
| v78 무가중 | −0.0047, 기각 | −0.0039 [−0.0075,−0.0004] → CI는 0을 제외하나 **게이트(0.010) 미만** |
| v79 dual projection | −0.0105, 기각 | **−0.0112 [−0.0144,−0.0080] → 기각 유지** (게이트 초과) |

### 1. 감사 방법 — 로그가 checkpoint를 기록해 두었다

`logs/official50/*.log`의 `Model: arch vNN, checkpoint <파일명>` 줄로 **10-task 완주 tag 80개
전부**가 어느 checkpoint로 채점됐는지 복원했다. 문서가 실제 판정에 쓴 36건을 추리면:

| 분류 | 건수 |
|---|---:|
| Δ < 0.005 (게이트 0.010의 절반에도 미달) | **17건** |
| **최종 epoch을 쓰지 않음**(중간 validation-best) | **27건** |
| 둘 다 | 11건 |
| 온전한 판정 | 4건 |

최악은 ridge calibration **epoch 27**, mlpbank2048/4096 **epoch 28/27**, latent4/8 **epoch 27/37**,
axis noise/rare **epoch 33/32**다. `scripts/rescore_final_epoch.sh`로 27개 arm을 최종 epoch
(200-epoch combo는 epoch 199)에서 재채점했다 — 270회 평가, 4-GPU 병렬 약 30분, 전부 10/10 성공.

### 2. 재채점 결과 — 차이는 작았다 (부호가 뒤집힌 arm은 없다)

| arm | 기존 ep | 기존 | **ep49** | Δ |
|---|---:|---:|---:|---:|
| mlpbank128 | 40 | 0.6697 | **0.6734** | +0.0037 |
| v73_magnitude | 46 | 0.6473 | **0.6509** | +0.0035 |
| ridge_calibration | 27 | 0.6840 | **0.6870** | +0.0031 |
| mlpbank4096 | 27 | 0.6649 | **0.6678** | +0.0030 |
| mlpbank2048 | 28 | 0.6751 | **0.6776** | +0.0025 |
| classsep_veryhard | 40 | 0.6823 | **0.6843** | +0.0020 |
| v78_unweighted | 42 | 0.6826 | **0.6841** | +0.0015 |
| v78_balanced | 42 | 0.6870 | **0.6879** | +0.0009 |
| combo_200ep | 95 | 0.6751 | **0.6761** (ep199) | +0.0009 |
| axis_noise | 33 | 0.6856 | **0.6839** | −0.0017 |
| **v76_learnable_p** | 46 | 0.6748 | **0.6735** | −0.0014 |
| latent8 | 37 | 0.6771 | **0.6759** | −0.0011 |
| axis_rare | 32 | 0.6847 | **0.6838** | −0.0008 |
| latent4 | 27 | 0.6781 | **0.6776** | −0.0005 |
| v69/v71/v72/v75/latent2/latent16/mlpbank512/mlpbank1024/mixed50/axis_classsep/axis_response/classsep_mild/v79 | — | — | 0.6712/0.6668/0.6706/0.6724/0.6775/0.6665/0.6730/0.6780/0.6755/0.6881/0.6777/0.6854/0.6768 | \|Δ\|≤0.0004 |

**최대 변화가 +0.0037이다.** 미완주 채점은 실재한 문제였지만 **효과 크기는 seed std(0.0051)보다
작다** — 즉 과거 판정이 통째로 틀린 것은 아니었다. 다만 판정 대상 Δ 자체가 0.001~0.005였으므로
**게이트 대비 위치는 여러 arm에서 바뀐다**(§105-4).

### 3. §91 ClassSep sweep 표에 오기가 있었다 — 결론이 바뀐다

§91은 Medium `[0.5,1.4]`를 **0.6823**으로 적었는데, 그 값은 같은 표의 Very-hard 값과 동일하고
어느 tag에도 대응하지 않는다. 실제 Medium arm은 `v76_axis_classsep_medium`이고 —
resolved config를 `classsep_mild`와 diff하면 **`class_separation`만 다르다**(확인함) —
**0.6882(ep48) / 0.6881(ep49)**다.

epoch 49로 통일한 ClassSep sweep:

| ClassSep | 범위 | ep49 macro | Δ vs baseline | Δ vs Hard | CI (vs Hard) |
|---|---|---:|---:|---:|---|
| baseline | `[1.0,2.0]` | 0.6735 | — | −0.0145 | — |
| **Medium** | `[0.5,1.4]` | **0.6881** | **+0.0146** | **+0.0001** | [−0.0022, +0.0024] 동률 |
| Mild | `[0.8,1.7]` | 0.6854 | +0.0119 | −0.0026 | [−0.0055, +0.0003] 동률 |
| **Hard** | `[0.2,0.8]` | **0.6880** | **+0.0145** | — | — |
| Very-hard | `[0.1,0.5]` | 0.6843 | +0.0108 | −0.0037 | [−0.0053, −0.0022] |

**⚠️ §91의 "단봉·매끄러워 실효과로 읽힌다"는 성립하지 않는다.** 실제 모양은
0.6735 → **0.6881** → 0.6854 → **0.6880** → 0.6843로, Medium과 Hard가 동률이고 그 사이 Mild가
살짝 내려가는 **평탄한 고원**이다. 네 난이도 arm의 폭은 0.0038로 **게이트(0.010)와 seed
std(0.0051) 모두 아래**다.

**살아남는 결론**: ClassSep을 `[1.0,2.0]`에서 조이는 것 자체는 **+0.011~+0.015로 유효하다**(게이트
초과). **사라지는 결론**: 그 안에서 `[0.2,0.8]`이 최적이라는 것 — Hard 선택은 노이즈 안의 임의
선택이었다. baseline을 옮길 이유는 없지만(Medium과 동률이므로), **"Hard가 최적"이라고 쓰지 말 것.**

참고: 2026-08-11의 `v76_medium`(0.6723)은 같은 `[0.5,1.4]`지만 **다른 arm**이다 — resolved diff에서
`observation_noise` 0.01, `rare_response_probability` 0.15, response scale 3종이 함께 달랐다.
난이도 축을 한꺼번에 5개 움직인 arm이므로 ClassSep sweep과 같은 표에 놓지 말 것.

### 4. 판정이 바뀌는 arm

v77 ep49 = 0.6880 기준, fold-paired:

| arm | ep49 Δ | 95% CI | 이전 판정 | **새 판정** |
|---|---:|---|---|---|
| axis_classsep (=Medium) | +0.0001 | [−0.0022, +0.0024] | (미승격) | 동률 — Hard와 구분 불가 |
| v78_balanced | −0.0001 | [−0.0006, +0.0003] | 동률 | 동률 (유지) |
| ridge_calibration | **−0.0010** | [−0.0021, +0.0002] | **기각** | **기각 철회 → 판정 불가** |
| classsep_mild | −0.0026 | [−0.0055, +0.0003] | 미승격 | 판정 불가 |
| classsep_veryhard | −0.0037 | [−0.0053, −0.0022] | 미승격 | 게이트 미만 |
| v78_unweighted | −0.0039 | [−0.0075, −0.0004] | 기각 | 게이트 미만 — 단정 불가 |
| **v79_dual** | **−0.0112** | [−0.0144, −0.0080] | 기각 | **기각 유지** |

그리고 v74(0.6731) 기준:

| arm | ep49 Δ | 95% CI | **새 판정** |
|---|---:|---|---|
| **v76_learnable_p** | **+0.0004** | [−0.0030, +0.0038] | **판정 불가 — 승격 근거 없음** |
| v75_cv2 | −0.0007 | [−0.0020, +0.0007] | 동률 (유지) |
| v72_mlp1 (nonlinear manifold) | −0.0025 | [−0.0045, −0.0005] | 게이트 미만 |

**⚠️ 계보의 두 승격 모두 근거를 잃었다.** v74 → v76(learnable P)은 +0.0004,
v76 → v77(Hard 선택)은 Medium과 +0.0001이다. 현행 baseline이 **틀렸다는 뜻은 아니다** — 지는
증거도 없다. 다만 **"learnable P가 fixed P보다 낫다"와 "Hard가 최적 난이도다"는 둘 다
미측정 상태**이고, 확정하려면 arm당 3 seed가 필요하다.

### 5. §103-5와 §104-6의 논거 재확인 — 둘 다 살아남았다

**단조 악화(§103-5)**: epoch 49에서도 v78 balanced −0.0001 → 무가중 −0.0039 → v79 −0.0112로
**순서가 유지된다.** 다만 세 점 중 둘이 게이트 미만이므로 "단조성"은 방향 증거로만 쓸 것.

**mlpbank M→∞ 외삽(§104-6)**: 재채점 전에는 하강 구간이 epoch 28/27에서 채점돼 학습 부족
artifact 의심이 있었다. epoch 49로 통일한 결과는 **0.6734 / 0.6730 / 0.6780 / 0.6776 / 0.6678**
(M=128/512/1024/2048/4096) — 1024와 2048이 고원을 이루고 **4096에서 −0.0102 하강한다.**
큰 M에서의 하강은 학습 부족이 아니라 실재하며, v80(M=∞) 기각의 보강 근거는 **유효하다.**

**latent sweep 비단조성(§99-3)**: epoch 49에서 L2 0.6775 / L4 0.6776 / L8 0.6759 / **L16 0.6665** /
L32 0.6880 — L16 딥이 그대로 남았다. 학습량 차이 가설은 배제됐다.

**axis sweep** (baseline 0.6735 기준, ep49): classsep +0.0146 / noise +0.0104 / rare +0.0103 /
response +0.0042. 앞의 셋은 게이트 초과다.

### 6. 재채점 불가 3종

1. **large-ragged warm-start** — epoch 34에서 조기 종료돼 epoch 49가 없다. +0.0012는 구조적으로
   epoch 통일이 불가능하고, 어차피 게이트 미만이다.
2. **CV-only 계보(v41_K128 0.6940, v41_K64, v42, v45)** — prune 이전 ckpt라 현재 트리로 로드
   불가(§73). `8caa96c` worktree 필요. 다행히 **원래 epoch 49/last로 채점돼 이미 최종 epoch**이다.
   v41_K128의 0.6940은 9-task 로그 + er_status 별도 채점으로 정확히 복원됐다.
3. **v43_notanh(0.6770) / v44_lowT(0.6763)** — `logs/official50`·`predictions/`에 **산출물이 전혀
   없다.** 결과표 수치를 artifact로 되짚을 수 없다 → `current_experiments.md`에 "artifact 없음"으로
   표기했다.

### 7. 산출물

- 러너: `scripts/rescore_final_epoch.sh` (경로를 사전 해석해 `manifest.txt`에 기록 — 누락은 조용히
  빠지지 않고 `SKIP` 줄로 남는다. 이번 실행은 skip 0건).
- 로그 `logs/20260812_rescore_ep49/{runner.out,manifest.txt,<tag>.out}`,
  평가 로그 `logs/official50/*_<tag>_ep49.log`, 예측 `predictions/*_ep49_official50_bf16.pt` 270개.
- 태그 규칙: 기존 `_best` 태그는 남기고 `_ep49`(combo는 `_ep199`)를 새로 만들었다 — 두 채점을
  나란히 비교할 수 있다.

### 8. 다음 Action

1. **seed 반복이 여전히 병목이고, 대상이 늘었다.** 이제 판정 불가로 내려간 것이 4건
   (v76 learnable P, Hard vs Medium, ridge calibration, v78 무가중)이다. 우선순위는
   **v74 vs v76 (learnable P)** — 계보의 뿌리이고, 여기가 무효면 v77 전체가 fixed-P로 되돌아간다.
2. **ClassSep을 "조이면 좋다"까지만 쓰고 특정 범위를 최적이라 쓰지 말 것.** Medium과 Hard의
   3 seed 비교가 필요하다.
3. §91의 Medium 행은 오기다 — 아카이빙할 때 이 절을 함께 참조하도록 표시했다.

---

## 106. 2026-08-13 — 4 arm × 4 seed 정면 비교: Medium이 Hard를 이기고, learnable P는 여전히 미판정

### 0. 이 절이 정한 것

| 질문 | 답 | 근거 |
|---|---|---|
| **Hard가 최적 난이도인가** | **아니다. Medium이 +0.0053 낫다** | 4/4 seed 양수, seed-paired t=3.0 |
| **learnable P가 fixed P보다 나은가** | **여전히 판정 불가 (+0.0048, t=1.5)** | seed 44가 부호 반전(−0.0043) |
| **1-GPU와 DDP4는 얼마나 다른가** | **DDP4가 +0.0098 높다** | v77 1-GPU 4 seed 0.6782 vs DDP4 0.6880 |
| v80 shallow MLP의 진짜 크기 | **−0.0059** (기존 −0.0158은 부풀려진 값) | 같은 레이아웃 control과 비교 |
| v77 seed std | **0.0053** | §104-3이 가정했던 0.0051과 일치 |

### 1. 배치 구성 — 처음으로 모든 arm이 같은 레짐이다

`SEED` 42/43/44/45, 1-GPU 레이아웃(`*_1gpu.yaml`), 50 epoch, **epoch 49 채점**으로 네 arm을
동일 조건에서 돌렸다. 각 arm은 v77에서 **한 축만** 바꾼 것이다.

| arm | v77에서 바뀐 것 | mean | seed std | range |
|---|---|---:|---:|---:|
| **v82** | ClassSep `[0.2,0.8]` → **`[0.5,1.4]`** | **0.6835** | 0.0030 | 0.0068 |
| **v77** | (기준) | 0.6782 | **0.0053** | 0.0120 |
| **v81** | learnable P → **fixed P** (197,057 → 449) | 0.6734 | 0.0018 | 0.0041 |
| **v80** | manifold orthogonal → **shallow MLP** | 0.6722 | 0.0051 | 0.0109 |

**seed std가 arm마다 3배 차이 난다** — fixed P 0.0018 < Medium 0.0030 < Hard 0.0053 ≈ shallow MLP
0.0051. 학습 파라미터가 449개뿐인 v81이 가장 안정적이고, **학습되는 P를 가진 arm일수록 realization
노이즈가 크다.** §104-5에서 "task별 CI가 seed 노이즈에 압도된다"고 한 것도 learnable-P arm의
성질이었다(v81은 task별 산포도 작다: brca TP53 폭 0.0008, EGFR 0.0034, 단 BAP1만 0.0256).

### 2. 난이도 — Hard가 최적이 아니다

seed-paired (같은 시드끼리 뺀 뒤 평균), fold-paired CI는 시드별:

| seed | v77 Hard | v82 Medium | Δ | 95% CI |
|---|---:|---:|---:|---|
| 42 | 0.6811 | 0.6846 | +0.0035 | [+0.0007, +0.0063] |
| 43 | 0.6834 | 0.6870 | +0.0036 | [+0.0006, +0.0067] |
| 44 | 0.6714 | 0.6821 | **+0.0107** | [+0.0075, +0.0140] |
| 45 | 0.6767 | 0.6802 | +0.0036 | [+0.0006, +0.0066] |
| **평균** | 0.6782 | **0.6835** | **+0.0053** | seed-paired SE 0.0018, **t=3.0** |

**4/4 시드에서 양수이고 시드별 CI도 모두 0을 제외한다.** §105-3은 단일 시드로 "Medium ≡ Hard
(+0.0001)"라 했는데, 4 seed로 보면 **Medium이 유의하게 낫다.** §91의 "Hard가 최고"는 오기(§105-3)
때문이었고, 정정 후에도 남아 있던 "동률" 판정마저 뒤집힌 셈이다.

⚠️ 다만 +0.0053은 §104-4의 게이트 0.010(단일 시드 2σ 기준) 미만이다. **4 seed paired 설계에서는
SE가 0.0018로 줄어 이 크기를 판정할 수 있다** — 게이트는 시드 1개짜리 비교에 대한 규칙이지
seed-paired 설계에 그대로 적용하는 값이 아니다.

### 3. learnable P — 같은 난이도에서 비교하면 여전히 미판정

| seed | v81 fixed P | v77 learnable P | Δ | 95% CI |
|---|---:|---:|---:|---|
| 42 | 0.6716 | 0.6811 | +0.0095 | [+0.0049, +0.0141] |
| 43 | 0.6739 | 0.6834 | +0.0095 | [+0.0049, +0.0140] |
| 44 | 0.6757 | 0.6714 | **−0.0044** | [−0.0086, −0.0001] |
| 45 | 0.6724 | 0.6767 | +0.0042 | [−0.0006, +0.0090] |
| **평균** | 0.6734 | 0.6782 | **+0.0048** | seed-paired SE 0.0033, **t=1.5** |

**시드 44에서 부호가 뒤집히고 그 CI도 0을 제외한다.** 즉 fold-paired CI만 보면 "seed 42·43은
learnable 승, seed 44는 fixed 승"으로 서로 모순되는 결론이 나온다 — §104-5가 경고한 그대로,
**realization 노이즈가 이 효과 크기를 삼킨다.** §105-4의 "v74→v76 승격은 근거 없음"이 유지된다.
197,057개 파라미터가 449개 대비 얻는 것은 **아직 증명되지 않았다.**

⚠️ 앞서 v82(Medium, learnable) − v81(Hard, fixed) = **+0.0101 (t=5.8)**로 크게 이긴 것은
**두 축이 겹친 결과**다: 난이도 +0.0053과 P +0.0048의 합이며, 개별로는 후자가 유의하지 않다.
**이 +0.0101을 "learnable P의 이득"으로 인용하지 말 것.**

### 4. 레이아웃 효과를 처음 실측했다 — DDP4가 +0.0098 높다

v77 1-GPU 4 seed **0.6782** vs 같은 config의 DDP4 단일 run **0.6880** → **−0.0098**.
1-GPU는 1024 step/epoch·effective batch 1, DDP4는 rank당 256 step·gradient 4개 평균이다.
합성 val_ce는 반대 방향이었다(1-GPU 0.1650~0.1692 < DDP4 0.1717) — **더 많이 학습해 합성 loss는
낮아졌지만 SEAL은 −0.0098 떨어졌다.** 합성 지표와 실데이터의 괴리가 레이아웃 축에서도 재현된다.

⚠️ DDP4 쪽은 시드 1개이므로 −0.0098에는 DDP4의 realization 노이즈(≈0.005)가 섞여 있다. 크기는
seed std의 약 2배이므로 방향은 신뢰하되 정확한 값으로 인용하지 말 것.

**이 confound가 §104-6의 v80 판정을 부풀렸다.** 같은 레이아웃 control과 비교하면:

| 비교 | Δ | 판정 |
|---|---:|---|
| v80 vs **DDP4** v77 0.6880 (§104-6에 기록된 값) | −0.0158 | 부풀려짐 |
| **v80 vs 1-GPU v77 4 seed (정당한 control)** | **−0.0059** (t=−2.7, 4/4 시드 음수) | **기각 유지** |

v80 기각 자체는 유지되지만 **크기는 2.7배 과장돼 있었다.** §104-6은 이 confound를 명시했고
"matched control 미실행"이라 적어 두었으므로 기록은 정직했으나, 이제 실측값으로 대체한다.

### 5. 승격 판단은 사용자 몫으로 남긴다

Medium이 Hard보다 나은 것은 4 seed로 확인됐지만, 이는 **1-GPU 레짐에서의 결과**다. 활성
baseline은 DDP4 v77 ep49 0.6880이고, Medium의 DDP4 수치는 단일 시드 0.6881뿐이다.
**baseline을 Medium으로 옮기려면 DDP4 Medium을 3~4 seed로 확인하는 것이 순서다**(arm당 약 28분).
지금 문서의 baseline 정의는 바꾸지 않았다.

### 6. 산출물

| arm | config | checkpoints | tag |
|---|---|---|---|
| v77 4 seed | `train_v77_hard_orthogonal_1536_1gpu.yaml` | `20260813_v77_baseline_seeds/` | `v77_baseline_seed4{2..5}_ep49` |
| v82 Medium | `train_v82_medium_classsep_1536{,_1gpu}.yaml` | `20260813_v82_medium_seeds/` | `v82_medium_seed4{2..5}_ep49` |
| v81 fixed P | `train_v81_hard_fixed_p_1536{,_1gpu}.yaml` | `20260812_v81_fixed_p_seeds/` | `v81_fixed_p_seed4{2..5}_ep49` |
| v80 shallow MLP | `train_v80_hard_shallow_mlp_1536{,_1gpu}.yaml` | `20260812_v80_shallow_mlp_seeds/` | `v80_shallow_mlp_seed4{2..5}_ep49` |

v81 config 검증: 초기 시점에 v77과 **수치적으로 동일**함을 확인했다 — P 차이 3.6e-07, 8,256차원
covariance descriptor 차이 1.0e-06. v77의 클래스는 부모의 `_covariance_projection` buffer를
`nn.Parameter`로 재등록하는 것이 전부이고 그 buffer는 이미 `QR(sin/cos).Q`로 orthonormal이라,
`model_src`를 부모로 되돌리면 **P가 v77의 초기값에서 얼어붙는다.**
v82 config 검증: resolved config가 기존 단일시드 Medium arm(`v76_axis_classsep_medium`)과
**완전히 동일**하고 v77과는 `class_separation`만 다르다.

### 7. 다음 Action

1. **DDP4 Medium 3~4 seed** — baseline 승격 여부를 가르는 유일한 남은 측정이다.
2. **learnable P를 살리려면 다른 증거가 필요하다.** 같은 난이도에서 t=1.5이고 시드 하나가 반대
   방향이다. Medium 난이도에서 fixed P를 돌려(4 seed) 상호작용을 보는 것이 다음 후보다 —
   "난이도를 조이는 이득이 learnable P에서만 나온다"는 가설은 v81(Hard, fixed) 0.6734 ≈
   v74(easy, fixed) 0.6731이라는 관측과 부합하지만, 아직 fixed×Medium 칸이 비어 있다.
3. **레이아웃은 DDP4로 통일할 것.** 1-GPU는 −0.0098이고 합성 val_ce는 오히려 좋아지므로
   착시를 만든다. 앞으로 arm 비교는 DDP4로 돌리거나, 1-GPU를 쓰면 control도 1-GPU로 맞춘다.

---

## 107. 2026-08-13 — baseline을 v82 Medium으로 승격, 판정 레짐을 1-GPU 4 seed로 전환 (사용자 결정)

### 0. 이 절이 정한 것

| 항목 | 이전 | **확정** |
|---|---|---|
| **활성 baseline** ⚠️ §109가 대체 | v77 Hard ClassSep `[0.2,0.8]` | **v82 Medium ClassSep `[0.5,1.4]`** (§109 이후는 v83 linear head) |
| **공식 baseline 숫자** ⚠️ §109가 대체 | 0.6880 (DDP4, 시드 1개) | **0.6835 = 1-GPU 4 seed 평균** (§109 이후는 v83 4 seed **0.6880**, 숫자만 같은 별개 레짐) |
| **판정 레짐** | DDP4, arm당 시드 1개 | **1-GPU, arm당 4 seed (42/43/44/45)** |
| **판정 통계** | fold-paired Δ + CI (단일 시드) | **seed-paired Δ + t** (시드별 fold-paired CI는 보조) |
| **채점 epoch** | epoch 49 고정 | 변경 없음 — **epoch 49 고정** |

**근거는 §106**이다. 네 arm을 처음으로 같은 레짐·4 seed로 돌린 결과 Medium이 Hard를 **+0.0053
(4/4 시드 양수, seed-paired t=3.0)**으로 이겼고, 그 전까지 "Hard가 최적"이라는 판정은 §91의 오기와
단일 시드 노이즈 위에 서 있었다(§105-3). 사용자 결정으로 **DDP4 Medium 4 seed를 기다리지 않고**
이미 4 seed가 존재하는 1-GPU 레짐을 공식 판정 레짐으로 채택한다.

### 1. ⚠️ 0.6835 < 0.6880은 성능 하락이 아니다 — 가장 큰 오독 위험

**숫자가 내려간 것은 arm이 나빠져서가 아니라 레짐이 바뀌었기 때문이다.** 1-GPU는 같은 config에서
DDP4보다 SEAL이 **−0.0098** 낮다(§106-4). 같은 레짐 안에서 비교하면 순위는 이렇다.

| 레짐 | v77 Hard | v82 Medium | Δ |
|---|---:|---:|---:|
| **1-GPU 4 seed (공식)** | 0.6781 | **0.6835** | **+0.0053** (t=3.0) |
| DDP4 1 seed (구 공식) | 0.6880 | 0.6881 | +0.0001 (판정 불가) |

**두 레짐의 숫자를 빼지 말 것.** v82 1-GPU 0.6835를 v77 DDP4 0.6880과 비교해 "Medium이 진다"고
읽는 것이 정확히 §106-4가 경고한 confound이며, 그 오독이 §104-6에서 v80의 기각 크기를 2.7배
부풀린 바로 그 실수다.

### 2. 무엇이 비교 대상에서 빠지는가

**이 문서·`current_experiments.md`의 기존 결과표는 전부 DDP4 단일 시드다.** 그 값들은 **자기들끼리는
여전히 유효**하지만 **새 1-GPU 4-seed arm과 직접 뺄 수 없다**. 구체적으로:

- **§105의 27개 arm epoch-49 재채점표** — 전부 DDP4 1 seed. 역사적 기록으로 유지한다.
- **v41_K128 0.6940** (역사적 전체 최고) — 레짐 미상. **여전히 미달 상태로 본다** — 1-GPU 페널티
  0.0098을 감안해도 v82 0.6835 + 0.0098 ≈ 0.693으로 근접할 뿐이고, 이 보정 자체가 시드 1개짜리
  추정(§106-4 경고)이라 **"따라잡았다"고 쓰지 말 것.**
- **ClassSep sweep 표(§105-3)** — Medium 0.6881, Hard 0.6880 등. DDP4 1 seed 기록으로 남긴다.

새 arm은 **`train_v82_medium_classsep_1536_1gpu.yaml` 4 seed와 직접 비교**한다. 다른 레짐의 숫자를
control로 쓰지 않는다.

### 3. 새 판정 절차

1. arm을 **1-GPU, SEED 42/43/44/45, 50 epoch**로 돌린다 (arm당 약 28분 × 4 = 약 2시간, GPU 0–3에
   4개를 동시에 올리면 약 28분).
2. **epoch 49 checkpoint**를 `_ep49` 태그로 채점한다.
3. baseline 4 seed와 **같은 시드끼리 빼서**(seed-paired) 평균 Δ와 SE를 낸다. 시드별
   fold-paired CI(`scripts/compare_arms_paired.py`)는 **보조 근거**로만 읽는다.
4. 판정: **4/4 시드 부호 일치 + |t| ≥ 2.5**면 판정 가능. 부호가 갈리면 **미판정**이다 —
   learnable P(+0.0048, t=1.5, seed 44 반전)가 그 예다.

⚠️ **단일 시드 게이트 0.010(2σ)은 이제 1 seed 비교에만 적용한다.** 4 seed paired 설계에서는
SE가 0.002 안팎이라 +0.005도 판정된다. 다만 **seed std가 arm마다 3배 차이 난다**(fixed P 0.0018 <
Medium 0.0029 < Hard 0.0053 ≈ shallow MLP 0.0051) — 학습되는 P를 가진 arm일수록 노이즈가 크므로
SE는 arm마다 실측할 것.

### 4. 승격되지 않은 것

- **learnable P는 여전히 미판정이다** (+0.0048, t=1.5, seed 44 −0.0044). v82가 learnable P를
  쓰는 것은 v77 계보를 이어받은 결과이지 learnable P가 증명돼서가 아니다. §105-4의 "v74→v76 승격은
  근거 없음"은 **유지**된다.
- **fixed P × Medium 칸은 여전히 비어 있다.** v82가 baseline이 된 지금 이 칸은 "baseline에서
  파라미터 197,057개를 449개로 줄여도 되는가"라는 질문이 되어 **우선순위가 올라간다.**
- **DDP4 레짐 자체를 폐기한 것은 아니다.** 최종 보고 숫자를 DDP4로 낼 필요가 생기면 baseline과
  arm을 **함께** DDP4 4 seed로 재측정한다.

### 5. 산출물

| 항목 | 값 |
|---|---|
| canonical config | `configs/train_v82_medium_classsep_1536_1gpu.yaml` (**self-contained로 전환**) |
| DDP4 판본 | `configs/train_v82_medium_classsep_1536.yaml` (base 체인 유지, 참고용) |
| checkpoints | `checkpoints/20260813_v82_medium_seeds/seed4{2..5}/` |
| tags | `v82_medium_seed4{2..5}_ep49` |
| 시드별 macro | 0.6846 / 0.6870 / 0.6821 / 0.6802 (mean **0.6835**, std 0.0029) |
| 이전 baseline | v77 Hard — 1-GPU 4 seed 0.6781, DDP4 1 seed 0.6880 (historical) |

canonical config를 self-contained로 인라인할 때 **resolved 결과가 chain 판본과 byte-identical**임을
확인했다 (`merge_train_config` 출력 sha256 `1b9ed13a…35cc5` 일치) — 이미 돌린 4 seed의 재현성이
깨지지 않는다.

### 6. 다음 Action

1. **fixed P × Medium 4 seed** — 비어 있는 칸이자 baseline의 파라미터 수를 197,057 → 449로 줄일 수
   있는지 가르는 측정이다. config는 v81을 Medium ClassSep으로 복제하면 된다.
2. **v82 Medium 위에서 큰 레버 탐색.** ClassSep을 조인 것은 +0.011~+0.015로 유효했고(§105-3),
   CV/DD·사영 배선 축은 소진이다(§103-5). 남은 축은 데이터 생성기 쪽 — noise(+0.0104)·rare(+0.0103)가
   ClassSep과 같은 크기의 레버였다(§105-6 axis sweep).
3. **레짐 전환에 따른 문서 정합성**은 이 커밋에서 처리했다 — `agent_handoff.md`,
   `current_experiments.md`, `current_architecture.md`의 baseline 계약을 모두 v82/1-GPU 4 seed로
   갱신했다.

## 108. 2026-08-13 — v83 linear-head ablation: GELU 제거해도 baseline과 통계적으로 구분되지 않음 (미판정)

_Recorded by: nhn-SMC-claude — 2026-08-13 17:30_

**질문**: relation head는 v70부터 12개 closed-form feature
(`[CV0,CV1,CV1-CV0,SEP_CV, D0,D1,D1-D0,SEP_DD, q0,q1,q0-q1,SEP_CT]`)를 `12→32→1`에 GELU 하나를
거쳐 결합해왔다 — 이 비선형성이 실제로 뭔가 기여하는지 한 번도 검증된 적이 없었다. `v83`은
`ct_head_hidden_dims: []`로 hidden layer와 GELU를 없애 head를 bare `Linear(12,1)`로 줄인다.
P(196,608)와 그 위 전부는 v82와 동일 — trainable **197,057 → 196,621**. v82 체크포인트와는
head shape가 달라 strict-load 불가, 처음부터 학습(`architecture_version=54`는 그대로 — 모델
클래스는 같고 head hidden dims만 다른 config 값이다).

**산출물**:
```
config: configs/train_v83_linear_head_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/  (epoch 49)
tags:   v83_linear_head_seed4{2..5}_ep49
macro:  0.6905 / 0.6896 / 0.6774 / 0.6944  →  mean 0.6880, seed std 0.0074
```

**baseline(`v82_medium_seed4{2..5}_ep49`, 0.6846/0.6870/0.6821/0.6802, mean 0.6835) 대비
seed-paired Δ**:

| seed | v82 | v83 | Δ(v83−v82) |
|---|---:|---:|---:|
| 42 | 0.6846 | 0.6905 | +0.0059 |
| 43 | 0.6870 | 0.6896 | +0.0026 |
| 44 | 0.6821 | 0.6774 | **−0.0047** |
| 45 | 0.6802 | 0.6944 | +0.0142 |
| 평균 | 0.6835 | 0.6880 | **+0.0045** |

평균 Δ = +0.0045, SD(Δ) ≈ 0.0078, SE ≈ 0.0039, **t ≈ 1.15**.

**판정 (§107-3 기준)**: **미판정**. 4/4 시드 부호 일치 조건을 충족하지 못하고(seed 44만 음수,
3/4 양수) `|t| ≈ 1.15`로 게이트 `|t| ≥ 2.5`에도 크게 못 미친다. GELU를 없애도 뚜렷한 손해는
없지만 뚜렷한 이득도 아니다 — 서두의 두 가설(GELU가 장식이다 / head가 실제로 비선형 결합을
한다) 중 어느 쪽도 이 4 seed로는 확정할 수 없다.

⚠️ **계산 방법 주의**: 위 macro는 각 task의 `fold-mean AUROC`(50-fold 평균) 10개를 평균한 값이며
`scripts/test_pathobench.py` 로그에서 직접 집계했다. **fold-paired CI(`scripts/compare_arms_paired.py`)는
아직 돌리지 않았다** — task별 세부 비교나 더 정밀한 유의성 검정이 필요하면 그 스크립트로 재확인할 것.

**per-task 값(seed42, 참고)**: bc_therapy/er_status 0.7219, grade 0.7444, her2_status 0.6405,
cptac_brca/PIK3CA 0.5592, TP53(brca) 0.8321, cptac_luad/EGFR 0.7753, STK11 0.8645, TP53(luad) 0.6725,
cptac_ccrcc/BAP1 0.6246, VHL 0.4699. (seed43/44/45도 동일 10-task 순서로 로그에 남아 있다,
`logs/official50/*_v83_linear_head_seed{42,43,44,45}_ep49.log`.)

→ 승격 여부에 대한 사용자 결정은 **§109**.

## 109. 2026-08-13 — baseline을 v83 linear head로 승격 (사용자 결정, §107-3 게이트 미달)

_Recorded by: nhn-SMC-claude — 2026-08-13 17:30_

§108의 결과(Δ +0.0045, t≈1.15, 3/4 시드 양수)는 §107-3이 정한 판정 게이트(4/4 시드 부호 일치 +
`|t| ≥ 2.5`)를 충족하지 못한다. 사용자가 이 수치를 검토한 뒤 "뚜렷하진 않아도 올랐다고 보는 게
맞다"고 판단해 **v83을 baseline으로 승격하기로 결정**했다(2026-08-13).

**⚠️ 이것은 통계적 판정이 아니라 사용자의 연구적 판단(override)이다.** §104-5·§105-4가 경고한
"게이트 미달 상태에서의 승격"과 같은 범주이며, 과거에 이런 승격 중 일부(§91 Hard 최적 판정,
v74→v76 승격)가 나중에 근거 없음으로 되돌려진 전례가 있다. 이 승격도 같은 리스크를 안고 있다는
것을 이어받는 모든 arm 비교에서 유의할 것.

**새 활성 baseline**:
```
config: configs/train_v83_linear_head_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/  (epoch 49)
tags:   v83_linear_head_seed4{2..5}_ep49
SEAL 10-task macro: 0.6905 / 0.6896 / 0.6774 / 0.6944 → mean 0.6880 (seed std 0.0074)
trainable parameters: 196,621 (P 196,608 + head Linear(12,1) = 13)
```

⚠️ **0.6880이라는 값이 옛 v77 DDP4 baseline(§104)의 0.6880과 우연히 숫자가 같다.** 완전히 다른
레짐(1-GPU 4 seed vs DDP4 1 seed)의 별개 수치이니 혼동하지 말 것 — §107-1·§107-2가 경고한
"레짐이 다른 숫자를 빼지 말 것"이 여기서도 그대로 적용된다.

**직전 baseline** v82 Medium ClassSep `[0.5,1.4]`(1-GPU 4 seed 0.6835)은 historical로 남는다.
v82와 v83의 모델 클래스·`architecture_version=54`는 동일하고 `class_separation`도 동일
(`[0.5,1.4]`, Medium) — 유일한 차이는 relation head 구조(32-hidden GELU → bare `Linear(12,1)`)뿐이다.

**바뀌지 않는 것**:
- §107의 판정 레짐(1-GPU · SEED 42/43/44/45 · epoch 49)과 §107-3 판정 절차는 그대로 유지된다 —
  새 arm은 **v83 4 seed와 seed-paired**로 비교한다.
- learnable P 미판정(§107-8), CV/DD·사영 배선 축 소진(§103-5) 등 §107이 정한 나머지 결론은
  전부 유지된다. v83 promotion은 head 구조에 대한 것이고 이 결론들과 무관하다.
- ⚠️ §107-6(fixed P × Medium 4 seed 빈칸)은 §111에서 **취소**됐다 — 더 이상 열린 과제가 아니다.

**다음 Action**:
1. 이 승격의 통계적 근거를 보강하려면 `scripts/compare_arms_paired.py`로 fold-paired CI를 뽑아
   task별 그림을 확인할 것 (§108은 macro만 계산했다).
2. seed를 4개 더 늘려(46–49) 8 seed paired 비교를 하면 t가 안정화될 수 있다 — 지금 t≈1.15는
   n=4에서 흔들림이 크다(§107-1 참고, n=4 std 추정 구간이 0.6~2.9배).

## 110. 2026-08-13 — v84 deep-head ablation: 더 깊은 head는 명확히 손해 (기각)

_Recorded by: nhn-NEXGEM-claude — 2026-08-13 18:15_

**질문(§108의 반대 방향)**: §108은 relation head의 GELU를 없앴을 때(`Linear(12,1)`)를 봤다 —
미판정이었다. 나머지 절반은 head에 오히려 용량을 더 주면 어떻게 되는가다. `v84`는
`ct_head_hidden_dims: [32, 32]`로 hidden layer를 2단으로 늘려 `12→32→32→1`(GELU 2개)로 만든다.
P(196,608)와 그 위는 v82/v83과 동일 — trainable **197,057 → 198,113** (head 1,505개). v82/v83
체크포인트와는 head shape가 달라 strict-load 불가, 처음부터 학습(`architecture_version=54` 동일).

**산출물**:
```
config: configs/train_v84_deep_head_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_163412/v84_deep_head_seed4{2..5}/  (epoch 49)
tags:   v84_deep_head_seed4{2..5}_ep49
macro:  0.6786 / 0.6783 / 0.6752 / 0.6789  →  mean 0.6777, seed std 0.0018
```

**seed-paired Δ, 두 baseline 모두에 대해**:

| seed | v82(구 baseline) | v83(현 baseline) | v84 | Δ(v84−v82) | Δ(v84−v83) |
|---|---:|---:|---:|---:|---:|
| 42 | 0.6846 | 0.6905 | 0.6786 | −0.0060 | −0.0119 |
| 43 | 0.6870 | 0.6896 | 0.6783 | −0.0087 | −0.0113 |
| 44 | 0.6821 | 0.6774 | 0.6752 | −0.0069 | −0.0022 |
| 45 | 0.6802 | 0.6944 | 0.6789 | −0.0013 | −0.0155 |
| 평균 | 0.6835 | 0.6880 | 0.6777 | **−0.0057** | **−0.0102** |

v82 기준: SD(Δ) ≈ 0.0032, SE ≈ 0.0016, **t ≈ −3.63**. v83 기준: SD(Δ) ≈ 0.0057, SE ≈ 0.0028,
**t ≈ −3.61**. 둘 다 4/4 시드 부호 일치(전부 음수).

**판정 (§107-3 기준)**: **기각, 양쪽 baseline 모두 기준으로**. 4/4 시드 부호 일치 +
`|t| ≥ 2.5`를 (구 baseline v82, 현 baseline v83) 둘 다에 대해 충족한다. head를 `12→32→32→1`로
심화하는 것은 뚜렷한 손해다.

**§108과 종합**: relation head는 **얕게 만들면 미판정(±0 근처)**, **깊게 만들면 명확히
손해**다 — 비대칭적이다. 12개 closed-form feature를 결합하는 데 GELU 하나(`12→32→1`)가 이미
충분하거나 과분하고, 그 이상의 용량은 information bottleneck이 아니라 순수 노이즈로 작용하는
것으로 보인다. 이 결과는 v83 promotion 결정(§109)을 뒤집지 않는다 — 오히려 "지금 head가 이미
적정 크기"라는 그림과 일치한다.

**바뀌지 않는 것**: 판정 레짐·baseline(v83, §109)은 그대로다. head 구조 축은 이제 얕음(미판정)·
기본(baseline)·깊음(기각) 세 지점이 다 나와서 **소진으로 본다** — 이 축에서 새 arm을 더 설계하지
말 것. (§107-6 fixed P × Medium은 이후 §111에서 취소됐다.)

**평가 방법**: `scripts/eval_seal_tasks.sh`로 GPU 0-3에 seed당 1개씩 올려 10-task 전부를 한
GPU에서 순차 실행(각 checkpoint 독립 실행이라 2-GPU 분할 없이도 4 seed가 병렬로 끝난다).
macro는 `logs/official50/*_v84_deep_head_seed{42,43,44,45}_ep49.log`의
`fold-mean AUROC`를 집계한 값이다. fold-paired CI는 아직 안 돌렸다 — 필요하면
`scripts/compare_arms_paired.py`로 추가 확인할 것.

## 111. 2026-08-13 — §107-6(fixed P × Medium, v85) 취소 (사용자 결정)

_Recorded by: nhn-NEXGEM-claude — 2026-08-13 19:05_

§107-6은 v81(fixed P, Hard 난이도)을 현재 baseline 난이도(Medium)에서 다시 측정해 "baseline
파라미터를 196,621 → 449로 줄여도 되는가"를 가르려는 계획이었다. config
`train_v85_medium_fixed_p_1536_1gpu.yaml`을 만들고(§107-6, 449 trainable parameters 확인)
다른 노드에 실행을 맡겼으나, 실제로는 어느 노드에서도 학습이 시작되지 않았다("진행 중"이라는
이전 기록은 착오 — 정정됨).

**사용자가 이 실험 자체가 필요 없다고 판단해 취소했다** — 굳이 지금 할 필요가 없다는 판단.
config는 한 번도 실행되지 않은 채 리포에서 삭제했다(재현 증거로 보존할 학습 결과가 없으므로
`configs/archive/`로 옮기지 않고 그냥 지웠다 — v84처럼 실행 후 기각된 arm과는 다르다).

**바뀌는 것**: §107-6은 더 이상 열린 과제가 아니다. §107이 남긴 다른 열린 항목(learnable P
미판정 §107-8, CV/DD·사영 축 소진 §103-5)은 이 취소와 무관하게 그대로 유지된다.

**다음**: 데이터 생성기 축(§105-6이 남겨둔 noise/rare 레버)을 재기획한다 — §112.

## 112. 2026-08-13 — v86 noise 재검증: 옛 효과가 재현되지 않는다 (null)

_Recorded by: nhn-NEXGEM-claude — 2026-08-13 20:23_

**질문**: §105-6 axis sweep(2026-08-12, DDP4 1seed, 옛 v74/v76-era baseline 0.6735)에서
`observation_noise: 0.005 → 0.01`(관측 노이즈 2배)이 **+0.0104**로 ClassSep 조이기(+0.0146)와
비슷한 크기의 레버였다. 그 이후 재확인도, 현재 baseline·레짐으로의 이관도 없이 방치돼 있었다 —
이 arm이 그 gap을 메운다.

**메커니즘** (`src/datasets/synthetic_data.py`): manifold로 만든 cell embedding에
`x = x + observation_noise * randn_like(x)`로 순수 가우시안 노이즈를 더한다(정규화 직전).
`class_separation`(class-conditional 평균 간 거리)과는 독립적인 축이다.

**변경**: v83 기준 `observation_noise` 0.005 → 0.01만 바꿈(옛 스윕과 동일한 스텝). 모델·head·P는
v83과 byte-identical(architecture_version=54, trainable 196,621) — 데이터만 다르다.

**산출물**:
```
config: configs/train_v86_noise_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_191148/v86_noise_seed4{2..5}/  (epoch 49)
tags:   v86_noise_seed4{2..5}_ep49
macro:  0.6896 / 0.6902 / 0.6775 / 0.6962  →  mean 0.6884, seed std 0.0072
```

**baseline(v83, 0.6905/0.6896/0.6774/0.6944, mean 0.6880) 대비 seed-paired Δ**:

| seed | v83 | v86 | Δ(v86−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6896 | −0.0009 |
| 43 | 0.6896 | 0.6902 | +0.0006 |
| 44 | 0.6774 | 0.6775 | +0.0001 |
| 45 | 0.6944 | 0.6962 | +0.0018 |
| 평균 | 0.6880 | 0.6884 | **+0.0004** |

SD(Δ) ≈ 0.0011, SE ≈ 0.0006, **t ≈ 0.71**. 3/4 시드만 양수.

**판정 (§107-3 기준)**: **완전 무효과(null)** — 미판정보다 더 명확하다. 4/4 부호 일치도 못
채우고 `|t|`가 게이트(2.5)의 3분의 1에도 못 미친다. §105-6의 +0.0104는 **지금 baseline·레짐에서
재현되지 않는다.**

**해석**: 두 가능성이 있다 — ① 그 옛 결과 자체가 DDP4 단일 시드의 요행이었을 가능성(realization
노이즈 ≈0.005 규모를 감안하면 +0.0104는 이례적으로 크긴 하지만 n=1이라 배제 못 함), ② 그때
baseline(ClassSep `[1.0,2.0]`, GELU head, 다른 response 파라미터 조합)과 지금 baseline(v83,
ClassSep `[0.5,1.4]`, linear head)이 달라 레버 자체가 baseline-dependent일 가능성. 어느 쪽이든
**observation_noise 0.005→0.01 스텝은 현재 baseline에서 쓸 수 있는 레버가 아니다.**

**바뀌지 않는 것**: v83 baseline·판정 레짐은 그대로다. §105-6의 나머지 축(rare, response,
classsep)에 대한 재검증 여부는 각각 별도로 판단할 것 — 이 null이 그것들의 재현성까지 부정하지
않는다.

**다음**: rare 축(§105-6에서 +0.0103, noise와 비슷한 크기) 재검증을 계획 중이다.

## 113. 2026-08-13 — v87 rare 재검증: 역시 재현 안 됨, 타겟 task(VHL/BAP1)도 무효 (null)

_Recorded by: nhn-NEXGEM-claude — 2026-08-13 21:37_

**질문**: §112(v86, noise)와 짝을 이루는 arm. §105-6에서 `rare_response_probability: 0.0 → 0.15`가
+0.0103으로 noise와 비슷한 크기였다. noise와 달리 rare는 실제로 지금 깨져 있는 두 task —
**cptac_ccrcc VHL(랜덤 이하)**과 **BAP1(large-bag에서만 붕괴)** — 와 메커니즘이 맞아떨어질
가능성이 있었다: 둘 다 "신호를 가진 세포가 소수"인 상황에서 못 찾는 패턴이고,
`rare_response_probability`는 정확히 그 상황(반응 세포 비율 1~8%)을 훈련에 주입한다.

**메커니즘** (`src/datasets/synthetic_data.py`): 에피소드별로 `rare_response_probability`
확률로 반응 세포 비율을 `shared_component_fraction`(기본, ~4~18%) 대신
`rare_response_fraction`(0.01~0.08, 1~8%)에서 뽑는다. 0.15면 학습 에피소드 약 7개 중 1개가
이 극소수-신호 모드로 생성된다.

**변경**: v83 기준 `rare_response_probability` 0.0 → 0.15만 바꿈(옛 스윕과 동일 스텝). 모델·head·P는
v83과 byte-identical(architecture_version=54, trainable 196,621) — 데이터만 다르다.

**산출물**:
```
config: configs/train_v87_rare_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_203144/v87_rare_seed4{2..5}/  (epoch 49)
tags:   v87_rare_seed4{2..5}_ep49
macro:  0.6872 / 0.6904 / 0.6802 / 0.6887  →  mean 0.6866, seed std 0.0043
```

**baseline(v83, 0.6905/0.6896/0.6774/0.6944, mean 0.6880) 대비 seed-paired Δ (macro)**:

| seed | v83 | v87 | Δ(v87−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6872 | −0.0033 |
| 43 | 0.6896 | 0.6904 | +0.0008 |
| 44 | 0.6774 | 0.6802 | +0.0028 |
| 45 | 0.6944 | 0.6887 | −0.0057 |
| 평균 | 0.6880 | 0.6866 | **−0.0013** |

SD(Δ) ≈ 0.0037, SE ≈ 0.0018, **t ≈ −0.70**. 2/4만 부호 일치.

**타겟 task 개별 확인 (VHL, BAP1)** — macro가 안 움직여도 이 둘은 따로 좋아질 수 있다는 가설을
직접 검정:

| seed | VHL v83 | VHL v87 | Δ | BAP1 v83 | BAP1 v87 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.4699 | 0.4589 | −0.0110 | 0.6246 | 0.6021 | −0.0225 |
| 43 | 0.4142 | 0.4222 | +0.0080 | 0.6619 | 0.6678 | +0.0059 |
| 44 | 0.4374 | 0.4301 | −0.0073 | 0.6082 | 0.6396 | +0.0314 |
| 45 | 0.4224 | 0.4090 | −0.0134 | 0.6795 | 0.6727 | −0.0068 |
| 평균 | 0.4360 | 0.4300 | **−0.0059** (t≈−1.23, 1/4) | 0.6436 | 0.6455 | **+0.0020** (t≈0.18, 2/4) |

**판정 (§107-3 기준)**: **완전 무효과(null)**, macro도 타겟 task도. macro는 4/4는커녕 2/4
부호 일치에 `|t|`가 게이트의 4분의 1 수준이다. **VHL은 오히려 약하게 반대 방향**(1/4만 양수) —
rare-episode 학습이 VHL을 돕는다는 가설과 맞지 않는다. BAP1은 seed마다 부호·크기가 요동해
방향성 자체가 없다(−0.0225~+0.0314).

**해석**: §112(noise)와 같은 결론 — §105-6 축 스윕의 효과들이 지금 baseline(v83)·레짐에서
재현되지 않는다. rare의 경우 추가로, **VHL/BAP1의 실패가 "훈련 데이터에 rare-abundance 신호가
부족해서"라는 가설도 반증됐다** — 이 메커니즘으로는 두 task를 못 고친다. 두 task의 실패는
데이터 생성기 축이 아니라 다른 원인(레이블 자체, 코호트 특성, 구조적 문제)일 가능성이 높다.

**바뀌지 않는 것**: v83 baseline·판정 레짐은 그대로다. §105-6의 남은 축(response, classsep)은
이 두 null과 별개로 각자 판단할 것 — 다만 classsep은 이미 baseline에 반영돼 있어 재검증 대상이
아니다(§107).

**다음**: 데이터 생성기 축(noise·rare)은 여기서 소진으로 본다. VHL/BAP1 문제는 별도 조사가
필요하다 — 다음 실험 방향은 재기획 중.

## 114. 2026-08-14 — v88 PA(label-conditioned population branch): 명확히 기각, VHL/BAP1도 개선 없음

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 02:11_

**질문**: 지금까지의 relation source는 전부 **per-cell 레이블을 fit에 넣지 않는다** — CV는 bag
descriptor 위의 ridge, DD는 label-free 분산 방향, CT는 label-free farthest-point 토큰을 뽑아
**그 다음에** 레이블로 점수만 매긴다. §65가 "미검정 레버"로 남겨둔 마지막 항목이 그것,
즉 **support 레이블을 보고 "이 population이 0/1을 가르는 데 중요한가"를 직접 학습**하는 분기다.
v88(PA, population attention)은 그 분기를 처음으로 구현한 arm이다.

**메커니즘** (`CovarianceMeanLearnablePDDCTPAMLPModel`, architecture_version=57):
1. context bag마다 `pa_cells_per_bag: 64` 세포를 샘플하고, **각 세포에 자기 bag의 레이블을
   상속**시킨다 — 노이즈 레이블이다(대부분의 세포는 공유 배경이고 소수만 판별적이다).
2. 그 세포 전체에 대해 **한 번의 ridge 회귀**를 풀어 방향 `w`와 절편을 얻는다
   (`_pa_cell_direction`, `solve_ridge_system`). "세포 하나를 이 축에 투영하면 label-1스러움
   점수가 나온다"는 축이다.
3. bag마다 **양쪽 방향의 soft abundance를 독립적으로** 잰다 —
   `abundance1 = sigmoid((z−τ)/T).mean()`, `abundance0 = sigmoid((−z−τ)/T).mean()`.
   ⚠️ 초기 설계는 같은 signed 축의 top-k-mean/bottom-k-mean이었는데, 그 둘은 거의 서로의
   거울상이어서 심어놓은 신호를 **우연 수준(8/16)으로만** 잡아냈다. 독립 abundance로 바꾼 뒤
   90~100%가 됐다(`tests/test_population_attention.py::PopulationAttentionAliveTest`).
   §62-2가 역사적으로 죽였던 population attention의 실패 모드와 **정확히 같은 종류**다.
4. 기존 12개 feature에 (PA0, PA1, PA1−PA0, SEP_PA) 4개가 붙어 head가 16-wide로 재구성된다.
   PA 계산 자체는 CT/DD와 마찬가지로 training-free(`no_grad`)고, 학습되는 건 P(196,608)와
   head(577)뿐 — **trainable 197,185**.

**GPU를 쓰기 전에 구현 버그 3개를 자체 테스트로 잡았다** (이게 이 절의 부수적 성과다):
- P의 gradient가 `None`이었다 — 내 freeze 루프가 부모가 learnable로 만든 `_covariance_projection`을
  다시 얼렸다. 해제 후 정상(§100 계약).
- 위의 top/bottom-k-mean 설계 결함(우연 수준).
- `train_dd_projection=True` 경로가 PA의 무조건 `no_grad` 안에 갇혀 조용히 죽었다 — scoping
  수정 + regression 테스트 추가.

**산출물**:
```
model:  src/models/set_transformer_ridge.py  CovarianceMeanLearnablePDDCTPAMLPModel (arch=57)
tests:  tests/test_population_attention.py   (9 tests, 전부 통과)
config: configs/train_v88_population_attention_1536_1gpu.yaml   (self-contained)
ckpts:  checkpoints/20260813_224003/v88_population_attention_seed4{2..5}/  (epoch 49)
tags:   v88_population_attention_seed4{2..5}_ep49
macro:  0.6803 / 0.6782 / 0.6700 / 0.6790  →  mean 0.6769, seed std 0.0047
```
⚠️ v83 checkpoint와 **strict-load 불가**(head 12 vs 16 in_features, arch 54 vs 57).

**baseline(v83, mean 0.6880) 대비 seed-paired Δ (macro)**:

| seed | v83 | v88 | Δ(v88−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6803 | −0.0102 |
| 43 | 0.6896 | 0.6782 | −0.0114 |
| 44 | 0.6774 | 0.6700 | −0.0074 |
| 45 | 0.6944 | 0.6790 | −0.0154 |
| 평균 | 0.6880 | 0.6769 | **−0.0111** |

SD(Δ) ≈ 0.0033, SE ≈ 0.0017, **t ≈ −6.69**, **4/4 시드 부호 일치**.

**타겟 task 개별 확인 (VHL, BAP1)** — PA를 만든 동기가 "소수 population을 못 찾아서 깨지는
두 task"였으므로 §113과 같은 방식으로 직접 검정:

| seed | VHL v83 | VHL v88 | Δ | BAP1 v83 | BAP1 v88 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 0.4699 | 0.4267 | −0.0432 | 0.6246 | 0.6308 | +0.0062 |
| 43 | 0.4142 | 0.4408 | +0.0266 | 0.6619 | 0.6368 | −0.0251 |
| 44 | 0.4374 | 0.3657 | −0.0717 | 0.6082 | 0.6051 | −0.0031 |
| 45 | 0.4224 | 0.3946 | −0.0278 | 0.6795 | 0.6209 | −0.0586 |
| 평균 | 0.4360 | 0.4070 | **−0.0290** (t≈−1.41, 1/4) | 0.6436 | 0.6234 | **−0.0202** (t≈−1.40, 1/4) |

**판정 (§107-3 기준)**: **기각**. macro가 4/4 부호 일치 + |t|=6.69로 게이트를 기각 방향으로
확실히 넘는다 — 이 세션에서 관측된 가장 강한 기각이다(v84의 |t|=3.61보다 크다). 타겟 task
쪽은 게이트에 미달해 확정 판정은 아니지만 **둘 다 1/4만 양수로 방향이 가설과 반대**다.

**해석**: PA는 죽지 않았다 — 합성 planted-signal 테스트에서 90~100%로 신호를 잡는다. 그런데
실제 SEAL에서는 head에 **노이즈만 추가한** 셈이 됐다. 두 가지로 읽을 수 있다: (a) 12개
CV/DD/CT feature가 이미 이 정보를 담고 있어 4개가 중복·잡음으로만 작용, (b) bag 레이블을
세포에 상속시키는 근사가 실제 데이터에서는 신호 대비 노이즈가 너무 크다(합성 데이터는
판별 세포 비율이 깨끗하게 심어져 있지만 실제 코호트는 그렇지 않다). 어느 쪽이든 **"레이블을
fit에 직접 넣는" 축은 이 형태로는 안 된다**.

**추가로 확정된 것**: VHL/BAP1의 실패는 §113(rare 데이터 주입)으로도, §114(레이블 조건
population 분기)로도 고쳐지지 않는다. "소수 판별 population을 못 찾아서"라는 **메커니즘 가설이
두 방향에서 반증**됐다 — 원인은 표현/구조가 아니라 레이블·코호트 쪽일 가능성이 더 커졌다.

**바뀌지 않는 것**: baseline은 v83(0.6880), 판정 레짐도 §107-3 그대로다. v88 코드는 기각됐지만
**삭제하지 않고 남긴다** — 테스트가 §62-2 실패 모드에 대한 살아있는 probe 역할을 하고,
"레이블 조건 분기를 이미 해봤다"는 음성 결과의 근거이기 때문이다.

**다음**: §65의 미검정 레버가 이걸로 소진됐다. 남은 방향은 재기획이 필요하다.

## 115. 2026-08-14 — 진단: 학습 에피소드가 실제 평가 에피소드의 **모양**을 한 번도 모사한 적이 없다

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 10:18_

**이 절이 정한 것**: §110·§112·§113·§114가 연속으로 실패한 뒤, arm을 더 던지는 대신 **평가 지표를
task 단위로 분해**했다. 그 결과 (a) v83의 task별 성적을 공식 지도학습 baseline과 처음으로 정면
대조했고, (b) **학습 분포와 평가 분포가 두 축에서 어긋나 있으며 그 어긋남이 한 번도 모델링된 적이
없다**는 것을 실측했다. v89는 그중 한 축을 움직이는 arm이고 현재 학습 중이다.

### 1. v83 task별 성적 vs 공식 지도학습 baseline

출처는 `docs/seal_univ2_baseline_17tasks.csv`(SEAL 논문의 50-fold ABMIL/MeanMIL, task별 mean±std).
v83 숫자는 4 seed(42–45) `logs/official50/*_v83_linear_head_seed4{2..5}_ep49.log`의 fold-mean AUROC
평균이다.

| task | v83 | ABMIL | Δ | MeanMIL | Δ |
|---|---:|---:|---:|---:|---:|
| bc_therapy er_status | 0.7276 | 0.717 | **+0.0106** | 0.712 | **+0.0156** |
| bc_therapy grade | 0.7259 | 0.770 | −0.0441 | 0.751 | −0.0251 |
| bc_therapy her2 | 0.6417 | 0.663 | −0.0213 | 0.684 | −0.0423 |
| cptac_brca PIK3CA | 0.5588 | 0.595 | −0.0362 | 0.544 | **+0.0148** |
| cptac_brca TP53 | 0.8270 | 0.801 | **+0.0260** | 0.787 | **+0.0400** |
| cptac_luad EGFR | 0.7761 | 0.830 | −0.0539 | 0.777 | −0.0009 |
| cptac_luad STK11 | 0.8754 | 0.908 | −0.0326 | 0.873 | **+0.0024** |
| cptac_luad TP53 | 0.6678 | 0.751 | −0.0832 | 0.735 | −0.0672 |
| cptac_ccrcc BAP1 | 0.6436 | 0.693 | −0.0494 | 0.720 | −0.0764 |
| cptac_ccrcc VHL | 0.4360 | 0.538 | −0.1020 | 0.542 | −0.1060 |
| **macro** | **0.6880** | **0.7266** | **−0.0386** | **0.7125** | **−0.0245** |

**읽는 법 세 가지**:
1. **모델이 전반적으로 약한 게 아니다.** ABMIL을 2/10, MeanMIL을 **5/10**에서 이긴다. 결손은
   특정 지점에 몰려 있다 — VHL(0.102) + luad TP53(0.083) + BAP1(0.049) = 0.234로 **총 결손
   0.386의 61%**가 3개 task에서 나온다.
2. ⚠️ **VHL은 지도학습으로도 거의 랜덤이다 — ABMIL 0.538 ± 0.128 (n=218, 50 fold).** std가
   0.128이라 0.5와 구분되지 않는다. **"VHL을 정상 task 수준으로 고친다"는 목표는 존재하지 않는다.**
   §113·§114가 VHL을 겨냥해 실패한 것은 상한이 0.538인 task를 겨눈 탓이 크다.
3. 그럼에도 **v83의 0.4360은 여전히 이상하다.** 신호 없는 모델은 0.5로 간다. 4/4 시드가 전부
   0.5 아래(0.4699/0.4142/0.4374/0.4224)인 것은 역상관 신호를 잡고 있다는 뜻이다. 그리고 이건
   실익이 있다 — **랜덤(0.5)까지만 돌려놔도 macro +0.0064, ABMIL 수준이면 +0.0102**로 §107-3
   게이트급이다.

### 2. 두 축의 train/eval mismatch — 실측

**클래스 비율.** `src/datasets/synthetic_data.py::_sample_labels`에 **클래스 사전확률 knob이 아예
없다**. `balanced: true`는 정확히 50/50이고, v83~v89가 쓰는 `balanced: false`는 불균형이 아니라
**Bernoulli(0.5)**다. 코드 주석이 의도를 명시한다 — *"Independent labels remove the episode-level
class-count variable: context label counts contain no information about a masked target."* 즉
context의 레이블 구성비가 정보를 갖지 **않도록 일부러 제거한** 설계다. 실제 task는 정반대다.

**context 크기.** 평가 경로 `scripts/test_pathobench.py`의 `--context-mode` 기본값은 `all`이고
`scripts/eval_seal_tasks.sh`는 이 인자를 넘기지 않는다. 따라서 `sample_context_ids()`가
`return list(train_ids)` — **train fold 전체를 자연 비율 그대로** 쓴다. 균형을 맞추는 `sample`
모드는 존재하지만 deprecated이고 사용되지 않는다.

| | 클래스 비율 | context 크기 |
|---|---|---|
| **학습** (v83 config) | 0.500 ± 0.055 (num_bags 60–100) | **60–100 bags** |
| **평가** (50 fold 실측) | **0.178 ~ 0.780** | **90 ~ 261 slides** |

| task | ctx | 양성비 | 학습분포 기준 σ | Δ ABMIL |
|---|---:|---:|---:|---:|
| er_status | 133 | 0.692 | +3.5σ | +0.0106 |
| grade | 133 | 0.617 | +2.1σ | −0.0441 |
| her2 | 133 | 0.391 | −2.0σ | −0.0213 |
| PIK3CA | 90 | 0.358 | −2.6σ | −0.0362 |
| brca TP53 | 90 | 0.404 | −1.7σ | +0.0260 |
| EGFR | 261 | 0.342 | −2.9σ | −0.0539 |
| STK11 | 261 | 0.178 | **−5.9σ** | −0.0326 |
| luad TP53 | 261 | 0.592 | +1.7σ | −0.0832 |
| BAP1 | 197 | 0.181 | **−5.8σ** | −0.0494 |
| VHL | 197 | 0.780 | **+5.1σ** | −0.1020 |

**in-distribution인 task가 하나도 없다.** 가장 가까운 것도 1.7σ 밖이고, context 크기는 6/10이
학습 상한(100)의 2~2.6배다. brca 두 개만 크기 범위 안에 있다.

⚠️ **상관은 약하다, 과대해석 금지**: corr(context 크기, Δ) = **−0.607**, corr(|비율 편향|, Δ) =
**−0.27**. n=10에 코호트가 4개뿐이라 코호트 정체성과 완전히 교란돼 있어 **어느 쪽도 결정적이지
않다**. 또 AUROC는 클래스 사전확률에 불변이므로 불균형이 지표를 기계적으로 깎지는 않는다 —
영향은 ridge가 치우친 design에서 방향을 추정하게 되는 간접 경로로만 온다. **결론으로 주장할 수
있는 것은 "이 두 축이 모사된 적이 없다"는 사실뿐이고, "그래서 성능이 낮다"는 아직 가설이다.**
§112·§113이 옛 스윕 수치를 재현하려다 실패한 것과 달리 이건 실측된 mismatch라는 점이 다르다.

### 3. 셀 축은 이미 반대로 어긋나 있다 (v89 해석에 필수)

실제 slide의 tile 수(`Data/PathoBench/features/*.h5`, 923개 실측):

| 코호트 | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| ccrcc | 358 | 2,638 | 4,988 | 9,635 | 28,831 | 6,947 |
| luad | 364 | 2,950 | 5,215 | 10,137 | 35,107 | 7,203 |
| brca | 1,282 | 4,059 | 7,736 | 13,149 | 33,297 | 8,975 |
| bc_therapy | 392 | 2,011 | 2,674 | 3,246 | 6,487 | 2,730 |

학습은 bag당 평균 **2,724** cell(cap 4,096)이다. 즉 셀 축은 **이미 실제보다 짧다**.

### 4. v89 — bag 축 arm (완료, 결과는 §7)

```
config: configs/train_v89_episode_shape_1536_1gpu.yaml   (self-contained)
run:    checkpoints/20260814_094411/v89_episode_shape_seed4{2..5}/
logs:   logs/20260814_094411/v89_episode_shape_seed4{2..5}.out
```

v83 대비 데이터 knob **4개만** 변경(모델·head·P는 byte-identical — arch 54, trainable 196,621로
실측 확인):

| knob | v83 | v89 |
|---|---|---|
| `num_bags` | [60, 100] | **[180, 300]** |
| `num_cells` | [256, 8192] | **[85, 2731]** |
| `per_bag_max_cells` | 4096 | **1365** |

**예산 중립이 설계 의도**다. log-uniform 추출을 시뮬레이션하면 E[cells/bag] 2,724 → 909(0.334배)로
E[total cells/episode] 217,936 → 218,167(**+0.1%**)이다. `scripts/smoke_train_budget.py --bf16`
실측(24 step, GPU 0):

| | v83 | v89 |
|---|---:|---:|
| mean cells/step | 297,643 | 298,480 (**+0.3%**) |
| mean step | 78 ms | 102 ms (+31%) |
| peak VRAM | 11.9 GiB | 13.3 GiB |
| episode shape | `(1, 79, 4096, 1536)` | `(1, 230, 1365, 1536)` |

셀 예산은 중립인데 step 시간이 31% 는다 — **bag당 연산(covariance sketch, DD `eigh`, CT 토큰)이
cell이 아니라 bag 수에 비례**하기 때문이다. epoch ~2분, 50 epoch ~100분/seed.

⚠️ **판정 시 반드시 지킬 것 — 두 축이 같이 움직인다.** 예산 고정은 곧 "셀 축이 bag 축의 비용을
지불한다"는 뜻이고, §3에서 보듯 셀 축은 이미 짧았다(평균 2,724 → 909, cap 4,096 → 1,365; 실제
중앙값 대비 약 2배 아래에서 **3~8배 아래**로). 따라서:
- **양성이면 명확하다** — 셀 축을 악화시키고도 이겼다는 뜻이니 bag 축 효과가 그만큼 크다.
- **음성/null이면 분리되지 않는다** — bag 축이 무효인지 두 효과가 상쇄된 건지 구분할 수 없다.
  그때는 **예산 3배(셀 유지, bag만 3배)** 변형으로 갈라야 하고 seed당 ~3.3시간이 든다.
  이 경우를 "bag 축 소진"으로 기록하지 말 것.

### 5. v90 — 클래스 비율 축 (준비 완료, 미실행)

§2의 나머지 한 축. `_sample_labels`에 knob이 없으므로 생성기 변경이 필요하다 — 상세는 §116.

### 6. 다음 Action

1. **v89 판정** — epoch 49, §107-3(1-GPU 4 seed, seed-paired Δ+t, 게이트 4/4 + |t|≥2.5),
   v83(0.6880) 대비. macro와 **10개 task 전부** per-seed로 뽑아 §1 표를 갱신할 것.
2. **v90 실행** — nhn-SMC에 준비 완료 상태로 인계됨(§116).
3. ~~**VHL을 겨냥한 arm 설계**~~ **하지 말 것** — §1-2에 따라 상한이 0.538이다. VHL은 "고칠
   task"가 아니라 "0.436 → 0.5의 역상관을 없앨 대상"이고, 그마저 별도 진단(fold별 부호 분포)
   사안이지 학습 arm 사안이 아니다.

### 7. v89 결과 — 미판정 (2026-08-14 11:36)

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 11:36_

```
ckpts: checkpoints/20260814_094411/v89_episode_shape_seed4{2..5}/  (epoch 49)
tags:  v89_episode_shape_seed4{2..5}_ep49
macro: 0.6835 / 0.6782 / 0.6822 / 0.6889  →  mean 0.6832, seed std 0.0044
```

| seed | v83 | v89 | Δ(v89−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6835 | −0.0070 |
| 43 | 0.6896 | 0.6782 | −0.0114 |
| 44 | 0.6774 | 0.6822 | **+0.0048** |
| 45 | 0.6944 | 0.6889 | −0.0055 |
| 평균 | 0.6880 | 0.6832 | **−0.0048** |

SD(Δ) ≈ 0.0069, SE ≈ 0.0035, **t ≈ −1.39**, **1/4 부호 일치**.

**판정 (§107-3)**: **미판정.** 게이트(4/4 + |t|≥2.5) 미달이다. 방향은 음수지만 기각도 아니다.

**⚠️ task별 수치는 판정에 쓰지 않는다 — 이 arm이 그 함정을 실제로 보여준다.** 10개를 모두
뽑으면 게이트를 넘는 task가 **2개** 나온다:

| task | v83 | v89 | Δ | t | 부호 |
|---|---:|---:|---:|---:|---|
| bc_therapy her2 | 0.6417 | 0.6737 | **+0.0320** | +2.59 | 4/4 |
| cptac_luad TP53 | 0.6678 | 0.6361 | **−0.0317** | −3.04 | 0/4 |

**둘 다 우연 범위다.** df=3에서 `P(|t| ≥ 2.5) ≈ 0.088`이므로 task 10개면 귀무가설 아래에서도
**기대 0.9개**가 게이트를 넘는다. 2개 관측은 놀랄 일이 아니다. §1 금지 사항 표의 *"task별 CI로
판정하지 않는다"*(§104-5, task별 seed std가 macro의 7배)가 그대로 적용된다. **판정은 macro로만
하고 task별은 방향 기록으로만 남긴다.**

방향 참고(판정 아님): VHL 0.4360 → 0.4538 (+0.0179, 3/4, 여전히 랜덤 이하) / BAP1 0.6436 →
0.6342 (−0.0094, 2/4, 방향성 없음) / STK11 0.8754 → 0.8760 (±0).
⚠️ §1-1이 "진짜 헤드룸"으로 지목한 **luad TP53이 오히려 내려갔다**. 게이트 논리로는 우연
범위지만 v90·v91에서 이 task의 거동은 계속 볼 것.

**⚠️ 이 결과를 "bag 축 소진"으로 기록하지 말 것 — §4의 사전 경고가 그대로 발동했다.** 예산
고정 때문에 bag 3배(효과 미상)와 cell 1/3(§3에 따르면 악화 방향)이 **동시에** 움직였고,
결과가 null/음성이므로 **"bag 축이 무효"와 "두 효과가 상쇄"를 구분할 수 없다.** 이건 사후
변명이 아니라 arm을 만들 때 config 헤더와 §4에 미리 적어둔 조건이다.

**분리 방법 — 처음 제안한 것보다 싸다.** 원래는 "예산 3배(cell 유지, bag만 3배), ~3.3시간/seed"를
생각했으나, **반대 방향이 훨씬 효율적이다**:

> **v91 = v83 + cell만 1/3 (bag은 60–100 그대로)** — cell 축 단독 효과를 직접 잰다.
> 예산이 **1/3**이라 **~22분/seed**로 지금까지 중 가장 싼 arm이다.

- v91 Δ ≈ −0.005 → cell 축이 전부 설명, **bag 축 ≈ 0**
- v91 Δ ≈ 0 → **bag 축이 진짜 −0.005**

⚠️ 가법성을 완전히 가정하는 건 아니다(교호작용이 있을 수 있다). 그래도 지금의 완전한 모호함을
가장 싼 비용으로 크게 줄인다.

## 116. 2026-08-14 — v90 클래스 비율 arm: 생성기에 `class_prior` 추가 (준비 완료, 미실행)

_Recorded by: nhn-NEXGEM-claude — 2026-08-14 11:05_

**질문**: §115-2의 나머지 한 축. 실제 task는 전부 불균형인데(양성비율 0.178~0.780) 학습은
Bernoulli(0.5)만 봤다. **불균형 자체를 학습시키면 달라지는가?**

**왜 knob을 새로 만들어야 했나**: `_sample_labels`에 클래스 사전확률 개념이 없었다. `balanced:
true`는 정확히 50/50, `balanced: false`는 Bernoulli(0.5)로 **둘 다 균형**이다. 후자의 주석이
설계 의도를 밝힌다 — *"Independent labels remove the episode-level class-count variable"*.
지름길을 제거한 합리적 선택이었지만, 그 대가로 **모델은 context에서 유병률을 읽는 법을 배우지
못한다**. 실제 에피소드는 그 정보를 항상 갖고 있다.

**구현** (`SyntheticManifoldGenerator`):
```yaml
class_prior: [0.15, 0.85]     # None이면 기존 동작 그대로
```
- 에피소드마다 `p ~ U(0.15, 0.85)`를 **한 번** 뽑고 `labels ~ Bernoulli(p)`. bag마다 뽑으면
  중심극한정리로 매 에피소드가 0.5 근처에 몰려 knob이 무의미해진다 — 테스트가 이걸 고정한다.
- `balanced: true`와 동시 지정은 **에러**다(균형 경로가 prior를 조용히 버리므로).
- `_repair_missing_classes`: 클래스가 하나라도 비면 무작위 bag 하나를 그 클래스로 뒤집는다.
  `ModelInterface._sample_query_index`가 단일 클래스 에피소드에서 **예외를 던지기** 때문에
  필수다. 지금 설정에서는 ~1e-13 사건이지만, epoch당 1024 에피소드 × 50 epoch에서 "드묾"은
  안전을 보장하지 않는다.

**테스트** (`tests/test_class_prior.py`, 12개 전부 통과):
- ⚠️ **가장 중요한 것 — `test_default_is_bit_identical_to_the_old_stream`**: knob 미지정 시
  레이블 스트림이 v83과 **bit-identical**이어야 한다. v90은 v83과 seed-paired로 비교되므로,
  파라미터 추가만으로 기본 경로가 흔들렸다면 비교가 prior가 아니라 리팩터를 재는 게 된다.
- 에피소드 단위 추출 검증(bag 단위였다면 sd≈0.03, 에피소드 단위면 U(0.15,0.85)의 sd=0.202)
- 실측 실제 범위(0.178~0.780) 포함 여부
- 극단 prior(0.01/0.99) × 400회에서 단일 클래스 에피소드 0건
- config `dataset_kwargs` → `**generator_kwargs` 배선

**산출물**:
```
generator: src/datasets/synthetic_data.py  class_prior + _repair_missing_classes
tests:     tests/test_class_prior.py       (12 tests)
config:    configs/train_v90_class_prior_1536_1gpu.yaml   (self-contained)
```
v83 대비 config diff는 `class_prior` 3줄 + `experiment_name`뿐이다. 모델·head·P는 byte-identical
(arch 54, trainable 196,621). `smoke_train_budget.py --bf16` 실측: peak VRAM 12.0 GiB, episode
shape `(1, 72, 4096, 1536)` — **v83과 같은 모양**이므로 비용도 v83의 것(~65분/seed)이지 v89의
것이 아니다. 실측 양성비율 분포(60 에피소드): min 0.125 / med 0.450 / max 0.887.

**알려진 부작용 (버그 아님)**: `_pairwise_ranking_loss`는 뽑힌 query가 전부 한 클래스면 0을
반환한다. prior가 치우치면 이 일이 잦아진다(p=0.15, query 12개면 약 14%). 즉
`ranking_loss_weight`가 실질적으로 더 적은 에피소드만 본다. **이것도 이 arm이 재는 대상의
일부다** — 평가 시 불균형이 문제라면 그 대가를 치르고도 이득이 나야 한다.

**v89와의 관계**: 두 arm은 §115-2의 서로 다른 축이고 **둘 다 v83 기준으로 독립 판정**된다.
순서 무관, 병렬 가능. ⚠️ 둘 다 이기더라도 **효과가 더해진다고 가정하지 말 것** — 결합 arm은
별도 검정이다.

**판정**: §107-3 (1-GPU, SEED 42/43/44/45, epoch 49, v83 0.6880 대비 seed-paired Δ+t, 게이트
4/4 + |t|≥2.5). ⚠️ **macro와 10개 task 전부를 per-seed로 보고할 것** — 이 arm의 요점은 task별
이고, macro만으로는 불균형이 심한 task(STK11 0.178 / BAP1 0.181 / VHL 0.780)가 움직였는지
알 수 없다.

**상태**: **완료 — 미판정 (4/4 방향 일치, 게이트 미달). 상세는 §117.**
```
run:  checkpoints/20260814_103137/v90_class_prior_seed4{2..5}/
logs: logs/20260814_103137/v90_class_prior_seed4{2..5}.out
gpu:  0, 1, 3, 5 (2·4번은 이 노드의 다른 사용자 작업 중이라 제외)
```

_Recorded by: nhn-SMC-claude — 2026-08-14 10:32_

## 117. 2026-08-14 — v90 class_prior 결과: 4/4 방향 일치하지만 게이트 미달 (미판정)

_Recorded by: nhn-SMC-claude — 2026-08-14 11:48_

**산출물**:
```
config: configs/train_v90_class_prior_1536_1gpu.yaml
ckpts:  checkpoints/20260814_103137/v90_class_prior_seed4{2..5}/  (epoch 49)
tags:   v90_class_prior_seed4{2..5}_ep49
macro:  0.6904 / 0.6851 / 0.6725 / 0.6829  →  mean 0.6827 (seed std 0.0074)
```

**baseline(v83, 0.6905/0.6896/0.6774/0.6944, mean 0.6880) 대비 seed-paired Δ**:

| seed | v83 | v90 | Δ(v90−v83) |
|---|---:|---:|---:|
| 42 | 0.6905 | 0.6904 | −0.0001 |
| 43 | 0.6896 | 0.6851 | −0.0045 |
| 44 | 0.6774 | 0.6725 | −0.0049 |
| 45 | 0.6944 | 0.6829 | **−0.0115** |
| 평균 | 0.6880 | 0.6827 | **−0.0053** |

평균 Δ = −0.0053, SD(Δ) ≈ 0.0047, SE ≈ 0.0024, **t ≈ −2.23**.

**판정 (§107-3 기준)**: **미판정**. 4/4 시드가 전부 음수라 **v89(1/4)보다 방향 일관성은 뚜렷하지만**,
`|t| ≈ 2.23`으로 게이트 `|t| ≥ 2.5`에 근소하게 못 미친다(4/4 부호 일치는 충족). "게이트 바로
아래"라는 점에서 완전한 null(v86·v87처럼 |t|<1)과는 성격이 다르다 — 방향은 일관되게 음수이되
확정하기엔 표본이 부족하다.

**per-task 값(참고 기록용 — 판정 근거 아님)**: NEXGEM이 v89에서 지적한 다중비교 문제(df=3에서
`P(|t|≥2.5)≈0.088`, task 10개면 귀무가설 하에서도 기대 0.9개가 게이트를 넘음)가 여기도 그대로
적용된다. 아래는 방향 참고용으로만 기록한다.

| task | v83(4-seed 평균, §115-1) | v90(4-seed 평균) | Δ |
|---|---:|---:|---:|
| er_status | 0.7276 | 0.7083 | −0.0193 |
| grade | 0.7259 | 0.7167 | −0.0092 |
| her2_status | 0.6417 | 0.6566 | +0.0149 |
| PIK3CA | 0.5588 | 0.5389 | −0.0199 |
| brca TP53 | 0.8270 | 0.8179 | −0.0091 |
| EGFR | 0.7761 | 0.7649 | −0.0112 |
| STK11 | 0.8754 | 0.8612 | −0.0142 |
| luad TP53 | 0.6678 | 0.6754 | +0.0076 |
| BAP1 | 0.6436 | 0.6610 | +0.0174 |
| VHL | 0.4360 | 0.4265 | −0.0095 |

⚠️ **class_prior가 "겨냥한" task가 예상대로 움직이지 않았다**: 가장 불균형한 STK11(0.178)과
VHL(0.780)은 오히려 소폭 하락했고, BAP1(0.181)만 상승했다. luad TP53(§115에서 "진짜 헤드룸"으로
지목된 task)은 소폭 상승(+0.0076)했지만 v89에서는 하락했었다 — 두 arm이 같은 task에 반대 방향
신호를 준 셈이라 여기서 결론을 내리지 않는다. 이 표는 전부 **다중비교 미보정, CI 없음** — 방향
참고 그 이상으로 쓰지 말 것.

**v89와의 관계**: 같은 §115-2 mismatch의 다른 축이다. v89(bag/cell, 미판정 Δ−0.0048, t=−1.39,
1/4)보다 v90(class_prior, 미판정 Δ−0.0053, **t=−2.23, 4/4**)이 방향 일관성은 더 강하지만 둘 다
게이트를 넘지 못했다. **두 축 다 "무효"라고 결론 내리기엔 이르다** — v90의 t가 게이트에 근접한
점을 고려하면 seed를 늘려 재검증할 가치가 있다.

**다음 Action**:
1. ~~**v90 seed 추가 재검증 후보**~~ **불필요해짐 (§118)** — 사용자가 종합적 판단으로 이미
   기각을 확정했다.
2. **v91(cell 축 단독, NEXGEM 인계)과 함께 놓고 판단** — v89·v90·v91 세 결과를 모아야 bag/cell/
   class_prior 세 축 중 무엇이 진짜 신호인지 정리된다.
3. ~~승격·폐기 판단은 사용자 몫이며, 이 결과만으로는 아직 어느 쪽도 아니다.~~ **§118에서 사용자가
   기각으로 확정했다.**

## 118. 2026-08-14 — 판정 프로토콜 변경: 통계 게이트 → 사용자의 종합적 판단 (사용자 결정), v90 기각 확정

_Recorded by: nhn-SMC-claude — 2026-08-14 11:58_

**계기**: §117의 v90 결과가 통계 게이트(§107-3, 4/4 부호 일치 + `|t|≥2.5`)에 근소 미달(t≈−2.23)
했다. 사용자가 macro 숫자만으로 판단하지 말고 **task 10개 전부를 baseline 성능대와 함께** 보라고
지적했다 — "0.4→0.5는 의미가 작지만 0.9→0.95는 의미가 크다"는 논리, 즉 **같은 크기의 Δ라도
baseline이 어디 있는지(랜덤 근처 vs 천장 근처)에 따라 실질적 의미가 다르다.**

**v90 재분석 (§117 데이터, baseline 성능대별 재배열)**:

| task | v83(baseline) | v90 | Δ | 구간 |
|---|---:|---:|---:|---|
| STK11 | 0.8754 | 0.8612 | −0.0142 | 고성능(천장 근접) — 하락 |
| brca TP53 | 0.8270 | 0.8179 | −0.0091 | 고성능 — 하락 |
| er_status | 0.7276 | 0.7083 | −0.0193 | 중상위 — 하락(최대 낙폭) |
| EGFR | 0.7761 | 0.7649 | −0.0112 | 중상위 — 하락 |
| grade | 0.7259 | 0.7167 | −0.0092 | 중상위 — 하락 |
| luad TP53 | 0.6678 | 0.6754 | +0.0076 | 중간 — 상승 |
| her2_status | 0.6417 | 0.6566 | +0.0149 | 중간 — 상승 |
| BAP1 | 0.6436 | 0.6610 | +0.0174 | 저신호(랜덤 근접) — 상승 |
| PIK3CA | 0.5588 | 0.5389 | −0.0199 | 저신호(랜덤 근접) — 하락 |
| VHL | 0.4360 | 0.4265 | −0.0095 | 랜덤 이하 — 하락 |

**패턴**: 하락 7개 중 5개(STK11·brcaTP53·er_status·EGFR·grade)가 **0.72~0.88의 고성능·고신호
구간**에 몰려 있고, 상승 3개(luadTP53·her2·BAP1)는 전부 **0.64~0.68의 저신호 구간**이다. 유일한
예외적 큰 낙폭(PIK3CA −0.0199)과 VHL(−0.0095)은 원래 거의 랜덤인 자리라 의미가 작다. **손해는
신호가 확실한 자리에 몰리고 이득은 애매한 자리에 몰린 비대칭** — 단순 macro Δ나 t보다 이 비대칭이
더 걱정스러운 신호라는 것이 사용자 판단이다.

**결정 (사용자, 2026-08-14)**: **v90 기각.** 통계 게이트(§107-3)는 근소 미달(미판정)이었지만,
task별 비대칭 패턴을 근거로 **사용자가 종합적 판단으로 기각을 확정**했다. §117의 "seed 추가
재검증" 액션은 이걸로 불필요해졌다.

**프로토콜 변경 (이 리포 전체, 모든 노드 적용)**:
1. **§107-3의 통계 게이트는 폐지되지 않는다** — seed-paired macro Δ, t, 4/4 부호 일치 여부는
   계속 계산하고 보고한다. 이건 여전히 1차 근거다.
2. **다만 게이트 통과/미달이 승격/기각을 자동으로 결정하지 않는다.** 최종 판정은 **사용자의
   종합적 판단**이다 — 판단 재료는 (a) macro seed-paired Δ+t, (b) **task 10개 전부**를 baseline
   값과 나란히(방향 + baseline이 랜덤 근처인지 천장 근처인지) 놓고 본 패턴, (c) 다른 arm·다른
   축과의 일관성.
3. **보고 형식이 바뀐다** — 평가가 끝나면 항상 (i) 이 arm이 무엇을 테스트하는지 먼저 설명하고,
   (ii) task 10개 전부를 baseline과 함께 표로 보여주고(요약만 하지 않는다), (iii) macro
   seed-paired Δ+t를 마지막에 보고한다. 최종 승격·기각·재검증 여부는 사용자가 정한다.
4. task별 다중비교 문제(§104-5, NEXGEM이 v89에서 지적)는 여전히 유효하다 — per-task 숫자를
   **판정의 통계적 근거**로 오독하지 말라는 경고는 그대로 유지된다. 다만 사용자가 여러 task에
   걸친 **패턴**(방향 일관성, baseline 성능대별 비대칭)을 종합적으로 읽는 것은 개별 task를
   개별 가설 검정으로 오용하는 것과 다르다 — 이번 결정도 개별 task 하나의 CI가 아니라 10개
   전체의 구조적 패턴에 근거했다.
5. 이 프로토콜은 이 리포를 공유하는 모든 노드(NEXGEM 포함)의 arm 판정에 적용된다.
