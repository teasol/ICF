# Agent handoff guide

> [!NOTE]
> **문서 기록 규칙 — 모든 노드 공통**
>
> 이 리포(`~/ICF`)는 여러 노드(NEXGEM, SMC, EWHA 등)가 같은 공유 스토리지를 마운트해서
> 동시에 세션을 돌린다. `docs/` 아래 문서(`agent_handoff.md`, `current_status.md`,
> `current_experiments.md`, `current_architecture.md`, `history.md`, `README.md`)를
> 수정할 때는 **어느 노드의 세션이 기록했는지 명시**할 것 — 형식은 `nhn-<NODE>-claude`
> (예: `nhn-EWHA-claude`, `nhn-NEXGEM-claude`, `nhn-SMC-claude`) + 날짜 및 시각(타임스탬프,
> `YYYY-MM-DD HH:MM`).
>
> 예: `_Recorded by: nhn-EWHA-claude — 2026-08-13 17:20_`
>
> 여러 노드가 같은 공유 문서에 동시에 쓸 수 있어서, 서명이 없으면 어느 항목을 어느 노드가
> 기록했는지 구분할 수 없다.

> [!IMPORTANT]
> **§115 진단 (2026-08-14) — 지금 작업의 최전선. 새 arm을 설계하기 전에 이걸 먼저 읽을 것**
>
> §110·§112·§113·§114가 연속 실패한 뒤 평가 지표를 task 단위로 분해했고, 두 가지가 나왔다.
>
> **① 학습 에피소드가 실제 평가 에피소드의 모양을 한 번도 모사한 적이 없다.**
> ```
> 축            학습 (v83)              실제 평가 (50-fold 실측)
> 클래스 비율   0.500 ± 0.055           0.178 ~ 0.780
> context bags  60 - 100                90 / 133 / 197 / 261
> ```
> **10개 task 중 in-distribution인 것이 0개다**(최소 1.7σ, VHL +5.1σ, STK11 −5.9σ, BAP1 −5.8σ).
> 평가는 `--context-mode all`이 기본이라 train fold를 자연 비율 그대로 쓴다(균형 맞추는 `sample`
> 모드는 deprecated). ⚠️ **§112·§113의 데이터 축 null과 혼동 금지** — 그건 옛 스윕 수치 재현
> 시도였고 이건 실측된 mismatch다. **v89**(bag 축)와 **v90**(비율 축)이 각각을 검정한다.
> ⚠️ 다만 corr(ctx, Δ)=−0.607 / corr(|편향|, Δ)=−0.27로 **상관은 약하다**(n=10, 코호트 4개로
> 교란). 주장할 수 있는 건 "모사된 적 없다"는 사실이고 "그래서 낮다"는 아직 가설이다.
>
> **② VHL은 고칠 수 있는 task가 아니다 — 지도학습 상한이 ABMIL 0.538 ± 0.128로 사실상 랜덤이다.**
> §113·§114가 VHL을 겨냥해 실패한 큰 이유가 이것이다. **VHL을 목표로 한 arm을 더 만들지 말 것.**
> (다만 v83의 0.4360이 4/4 시드 모두 0.5 아래인 건 여전히 이상하고 — 랜덤까지만 돌려놔도
> macro +0.0064다 — 그건 학습 arm이 아니라 fold별 부호 분포 진단 사안이다.)
> ABMIL 대비 총 결손 0.386의 **61%가 VHL+luadTP53+BAP1 3개**에 몰려 있고, MeanMIL 기준으로는
> **5/10 task에서 이긴다.** 진짜 헤드룸은 **luad TP53**(상한 0.751, 현재 −0.083)이다.

**Last updated**: `2026-08-14` (nhn-NEXGEM-claude 11:05) — 활성 baseline은 **v83 linear head**(relation head를 32-hidden GELU에서 bare `Linear(12,1)`로 축소)이고 공식 SEAL macro는 **1-GPU 4 seed 평균 0.6880**다(§109, **사용자 결정 — §107-3 판정 게이트 미달**). v82 대비 seed-paired Δ는 +0.0045, t≈1.15로 **4/4 시드 부호 일치도 |t|≥2.5도 충족하지 못한 채 승격**됐다(§108). §108의 반대 방향인 **v84(head를 `12→32→32→1`로 심화)는 §110에서 양쪽 baseline(v82·v83) 기준 모두 명확히 기각**됐다(4/4 시드, |t|>3.6) — relation head 깊이 축은 이걸로 소진이다. §107-6(fixed P × Medium)은 **취소**됐다(§111, 사용자
결정 — 한 번도 실행되지 않은 채 config 삭제). 데이터 생성기 축 재검증으로 넘어가 **v86(observation_noise
0.005→0.01)과 v87(rare_response_probability 0.0→0.15)을 v83 기준으로 재측정한 결과 둘 다 완전
무효과**였다(v86: +0.0004 t=0.71 3/4 — §112; v87: −0.0013 t=−0.70 2/4 — §113). §105-6의 옛
+0.0104/+0.0103은 지금 baseline/레짐에서 재현되지 않는다. v87은 추가로 VHL/BAP1 개별 확인까지
했으나 "rare 학습이 두 task를 돕는다"는 가설도 반증됐다(VHL은 오히려 약한 반대 방향). **noise·rare
축은 여기서 소진**이다. 그 다음으로 §65가 남겨둔 마지막 미검정 레버 — **support 레이블을 fit에
직접 넣는 population 분기(v88 PA, arch 57)** — 를 구현해 검정했고 **명확히 기각**됐다(§114):
v83 기준 Δ **−0.0111**, **t=−6.69**, **4/4 시드** — 이 레짐에서 관측된 가장 강한 기각이다.
분기 자체는 죽지 않았고(합성 planted-signal 90~100%) 실제 데이터에서 노이즈로만 작용했다.
VHL/BAP1도 각각 −0.0290/−0.0202로 동기가 반증돼, §113과 합쳐 **"VHL/BAP1은 소수 population
탐지 실패 때문"이라는 가설이 두 방향에서 반증**됐다. 새 실험 방향은 재기획 중이다.
직전 baseline **v82 Medium**(1-GPU 4 seed 0.6835)은 historical. ⚠️ v83의
0.6880은 옛 v77 DDP4 baseline(§104)의 0.6880과 **숫자만 같은 별개 레짐의 값**이다 — 혼동하지 말 것.
**판정 레짐 자체(1-GPU 4 seed, seed-paired Δ+t, §107-3)는 그대로 유지**되고, 이전 DDP4 숫자(v77
0.6880, v41_K128 0.6940, §105 재채점표 27개)와는 여전히 **직접 비교 불가**다(§107-2). 진행 상태는
`current_status.md` §108–§116. **v89**(bag 축)는 `checkpoints/20260814_094411/`에서 **학습 중**이고,
**v90**(클래스 비율 축, 생성기에 `class_prior` knob 신설)은 **준비 완료·미실행**으로 nhn-SMC에
인계됐다 — 둘 다 §115의 mismatch를 겨냥하며 각각 독립으로 v83과 비교된다.

> [!IMPORTANT]
> **baseline 숫자 계약 (§109, 2026-08-13) — 이것부터 읽을 것**
>
> ```
> config: configs/train_v83_linear_head_1536_1gpu.yaml   (self-contained, canonical)
> ckpts:  checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/  (epoch 49)
> tags:   v83_linear_head_seed4{2..5}_ep49
> SEAL 10-task macro: 0.6905 / 0.6896 / 0.6774 / 0.6944 → mean 0.6880 (seed std 0.0074)
> ```
> **활성 baseline은 v83 linear head이고 공식 값은 4 seed 평균 `0.6880`다.**
>
> **⚠️ 이 승격은 §107-3 판정 게이트를 충족하지 못한 상태에서의 사용자 결정이다.** v82 baseline
> (0.6846/0.6870/0.6821/0.6802) 대비 seed-paired Δ는 +0.0059/+0.0026/**−0.0047**/+0.0142,
> 평균 **+0.0045**, **t ≈ 1.15** — seed 44가 부호 반전이고(3/4 양수), 게이트(4/4 부호 일치 +
> `|t|≥2.5`)에 못 미친다. "뚜렷하진 않아도 올랐다고 보는 게 맞다"는 사용자 판단으로 승격했다
> (2026-08-13, §109). **인용할 때 이 승격이 통계적 미판정 상태였다는 것을 함께 밝힐 것.**
>
> **⚠️ 0.6880 < 이전 0.6880(v77 DDP4)은 숫자가 같아 보이지만 다른 것이다.** 완전히 다른 레짐
> (1-GPU 4 seed vs DDP4 1 seed)의 별개 수치다 — 착시로 "제자리로 돌아왔다"고 읽지 말 것.
>
> **새 arm은 1-GPU·SEED 42/43/44/45·epoch 49로 돌려 이제는 v83 4 seed와 seed-paired로 비교한다.**
> 판정 조건은 **4/4 시드 부호 일치 + |t| ≥ 2.5**이고, 부호가 갈리면 **미판정**이다(§107-3, 그대로 유지).
>
> **왜 epoch 고정인가**: val_ce 곡선이 평평한 arm에서 validation-best 선택이 **과소학습 지점을
> 고르는 것을 실측**했다 — v80 seed 43은 val-best가 epoch 16을 골라 epoch 49 대비 **−0.0089**를
> 잃었고, val_ce로는 epoch 16이 0.0014 더 좋아 보였다(§104-2). v77 자신은 이 선택에 둔감했다.
>
> **직전 baseline**: v82 Medium(1-GPU 4 seed 0.6835, tags `v82_medium_seed4{2..5}_ep49`, §107).
> **이전 baseline (historical, DDP4 1 seed)**: `v77_hard_ep49` = 0.6880,
> `checkpoints/20260812_v76_classsep_sweep/hard/periodic-epoch=049-val_ce_loss=0.1717.ckpt`.
> 같은 run의 epoch 48 validation-best는 0.6873이다(Δ +0.0007 [+0.0000, +0.0014]).

> [!IMPORTANT]
> **활성 baseline: v83 linear head (§109, 사용자 결정 — §107-3 게이트 미달)**
>
> ⚠️ **v82와의 유일한 차이는 relation head 구조다.** `ct_head_hidden_dims: []`로 hidden layer와
> GELU를 없애 head가 `12→32→1`(GELU 포함)에서 bare **`Linear(12,1)`**로 줄었다 — trainable
> **197,057 → 196,621**. 그 외 모델 클래스·텐서 구조·`architecture_version=54`·`class_separation
> [0.5,1.4]`(Medium)는 v82와 완전히 같다. **head shape가 달라 v82 checkpoint는 v83 arm으로
> strict-load되지 않는다** (`tests/test_relation_head_depth.py`가 이 실패를 pin한다) — 처음부터
> 학습했다. canonical config는 `train_v83_linear_head_1536_1gpu.yaml`로 바뀐다.
>
> DD는 support label로 generalized covariance direction을 만들고 standardized dispersion
> distances `D0,D1`을 계산한다. CT는 support cells에서 label-free farthest-point 후보
> token 16개를 만든 뒤, bag-level abundance의 표준화 class 차이로 label-0/label-1
> discriminative token을 대칭 선택한다. query label은 사용하지 않는다.
>
> head 입력은 v76부터 동일한 12개 feature다:
> `[CV0,CV1,CV1-CV0,SEP_CV,D0,D1,D1-D0,SEP_DD,q0,q1,q0-q1,SEP_CT]`
> — 다만 v83은 이를 `32→1` GELU 은닉층 없이 **바로 `Linear(12,1)`**로 결합한다(§108의 질문:
> 이 GELU 비선형성이 실제로 기여하는지 검증). P(1536×128)와 head만 학습되어
> **196,621 trainable parameters**다 (P 196,608 + head 13).
> P는 CV ridge gradient로 학습되고 DD는 현재 P의 covariance를 읽되 DD→P gradient는 없다.
> DD/CT 자체는 training-free다.
> synthetic는 **Medium ClassSep `[0.5,1.4]`**, fresh orthogonal manifold, 50 epochs다.
> canonical config는 `configs/train_v83_linear_head_1536_1gpu.yaml`(self-contained)이고
> **판정용 checkpoint는 4 seed × epoch 49**다:
> `checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/`
> → 공식 SEAL macro **0.6880** (4 seed 평균, tags `v83_linear_head_seed4{2..5}_ep49`, §109).
> **§108의 seed-paired 결과(GELU 있는 head 대비)**: 미판정(Δ+0.0045, t≈1.15, seed 44 반전) —
> 이 승격은 그 미판정 상태에서 내려진 사용자 결정이다.
>
> **직전 baseline** v82 Medium은 1-GPU 4 seed 0.6835(tags `v82_medium_seed4{2..5}_ep49`, §107).
> **historical (DDP4 1 seed, 직접 비교 불가)**: v77 Hard `v77_hard_ep49` 0.6880
> (val-best epoch 48은 0.6873, Δ +0.0007 [+0.0000, +0.0014]);
> 같은 1-GPU 4 seed 레짐에서 v77은 **0.6781**이다.
> ⚠️ v83의 1-GPU 4 seed 0.6880과 v77의 DDP4 1 seed 0.6880은 **숫자만 같은 별개 수치**다.
> v76 easy predecessor는 0.6748, v74는 fixed-P control 0.6731,
> v70은 재현 control, v71(CV+MLP) 0.6667,
> v72(nonlinear manifold) 0.6709, v73(+Magnitude) 0.6473으로 승격하지 않는다(전부 DDP4 1 seed).
> **v80 shallow infinite MLP manifold도 1-GPU 4 seed 평균 0.6722로 기각**이다 — 같은 레짐
> control 대비 **−0.0059**(t=−2.7)이며, §104-6의 −0.0158은 레이아웃 confound로 부풀려진
> 값이다(§106-4).
>
> 이 승격은 head 구조 승격이며 텐서 구조는 v76과 같아 내부 `architecture_version=54`를
> 유지한다. 예전에 v77이라 부른 `PopulationTokenResidualModel`(0.6750)은 **retired provisional
> v77-pop-residual**이며 내부 version 55는 replay용으로 보존한다.

> [!IMPORTANT]
> **v84 deep-head 기각 (§110, 2026-08-13) — relation head 깊이 축 소진**
>
> §108이 GELU를 없앤 방향(미판정)을 봤다면, **v84는 반대 방향** — `ct_head_hidden_dims: [32, 32]`로
> hidden layer를 2단으로 늘려 head를 `12→32→32→1`(GELU 2개)로 심화한다. P(196,608)와 그 위는
> v82/v83과 동일, trainable **197,057 → 198,113**(head 1,505개). v82/v83 checkpoint와는 head shape가
> 달라 strict-load 불가.
>
> **판정용 checkpoint는 4 seed × epoch 49**: `checkpoints/20260813_163412/v84_deep_head_seed4{2..5}/`
> → SEAL macro 0.6786/0.6783/0.6752/0.6789, mean **0.6777**(seed std 0.0018).
>
> **양쪽 baseline 모두 기준으로 기각**: v82(구 baseline, 0.6835) 대비 Δ**−0.0057**(t≈**−3.63**),
> v83(현 baseline, 0.6880) 대비 Δ**−0.0102**(t≈**−3.61**) — 둘 다 **4/4 시드 부호 일치 +
> |t|≥2.5**로 §107-3 게이트를 충족한다(기각 방향으로 판정 확정, §108의 미판정과 다름).
>
> **종합**: relation head는 얕게 만들면 미판정(§108)이고 깊게 만들면 명확히 손해(§110)다 — 지금
> `12→32→1`(GELU 하나)이 이미 적정 크기라는 그림과 일치한다. **이 축에서 새 arm을 더 설계하지
> 말 것.** v83 promotion(§109) 결정 자체는 바뀌지 않는다.

> [!IMPORTANT]
> **판정 계약 (§107, 2026-08-13, 게이트는 그대로 유지 — 단 §109가 baseline 자체를 미판정 상태로 승격시켰다는 점은 예외로 기록됨) — 판정 단위가 4 seed다**
>
> **⓪ 모든 arm은 1-GPU · SEED 42/43/44/45 · 50 epoch · epoch 49 채점으로 돌린다.**
> baseline 4 seed(현재 `v83_linear_head_seed4{2..5}_ep49`, §109)와 **같은 시드끼리 빼서**(seed-paired)
> 평균 Δ와 SE를 낸다. **판정 조건은 4/4 시드 부호 일치 + |t| ≥ 2.5**이고 부호가 갈리면 **미판정**이다.
> 시드별 fold-paired CI(`scripts/compare_arms_paired.py`)는 **보조 근거**다.
> arm당 약 28분이고 GPU 0–3에 4 seed를 동시에 올리면 한 배치가 약 28분에 끝난다.
>
> **① macro seed std는 arm마다 다르다** (epoch 49 고정, 동일 config·`SEED`만 42~45, n=4):
> fixed P 0.0018 < Medium 0.0029 < Hard 0.0053 ≈ shallow MLP 0.0051. **학습되는 P를 가진 arm일수록
> realization 노이즈가 크다** — SE는 arm마다 실측할 것. ⚠️ n=4라 std 추정의 95% 구간이 대략
> 0.6~2.9배다.
>
> **② 단일 시드 게이트 ≈ macro Δ 0.010(2σ)은 이제 1 seed 비교에만 적용한다.** 4 seed paired
> 설계에서는 SE가 0.002 안팎이라 **+0.005도 판정된다**(Medium−Hard가 그 예다). 단일 시드 기준에서
> 과거 판정 2건이 내려갔다 — ridge calibration(−0.0033, 0.6σ)과 **v78 무가중(−0.0047, 0.9σ)**.
> §103-5의 "단조 악화" 논거는 중간 단계를 잃었으므로 **인용하지 말 것**(축 소진 결론 자체는
> v79 −0.0105가 지탱한다).
>
> **③ task별 fold-paired CI는 판정 근거가 아니다.** **시드만 다른 두 run**에서 task 6개의 CI가
> 0을 제외했고 BAP1은 **−0.0402**였다(처치 없음). task별 seed std는 평균 0.0161로 macro의 7배다.
> pairing은 fold 노이즈만 잡고 realization 노이즈는 남긴다. **§99-2의 "large-ragged는 재분배다"와
> BAP1 large-bag 붕괴 조사 항목은 근거를 잃었다** — 다시 세우려면 arm마다 최소 3 seed다.
>
> **④ 채점은 epoch 49 고정.** validation-best는 val_ce가 평평한 arm에서 과소학습 지점을 고른다.
> 과거 arm 27개는 `scripts/rescore_final_epoch.sh`로 재채점을 마쳤다(`_ep49` 태그, §105).
> 재채점의 최대 변화는 +0.0037로 seed std보다 작았다 — 미완주 채점은 실재했지만 크기는 작다.
>
> **⑤ 계보의 두 승격은 미측정이다 (§105-4).** `v74 → v76`(learnable P) **+0.0004
> [−0.0030,+0.0038]**, `v76 → v77`(Hard 대 Medium) **+0.0001 [−0.0022,+0.0024]**. 현행 baseline이
> 틀렸다는 뜻은 아니지만 **"learnable P가 fixed P보다 낫다"·"Hard가 최적 난이도다"를 확정된
> 사실로 쓰지 말 것.** 확정에는 arm당 3 seed가 필요하고, 우선순위는 v74 vs v76이다 —
> 여기가 무효면 계보 전체가 fixed-P로 되돌아간다.
>
> **⑥ ClassSep 서술 규칙**: **Medium `[0.5,1.4]`가 Hard `[0.2,0.8]`보다 +0.0053 낫고**
> (4 seed, 4/4 양수, seed-paired t=3.0) **§107에서 Medium이 baseline이 됐다.** "Hard가 최적"은
> 성립하지 않는다. ClassSep을 `[1.0,2.0]`에서 조이는 것 자체는 +0.011~+0.015로 유효하다.
> §91 표의 Medium 0.6823은 **오기**다(Very-hard 값 중복, 실제 0.6881).
>
> **⑦ 레이아웃을 섞지 말 것 (§106-4)**: **1-GPU는 DDP4보다 SEAL이 −0.0098 낮다**
> (v77 1-GPU 4 seed 0.6781 vs DDP4 0.6880). 그런데 합성 val_ce는 **반대로** 1-GPU가 더 좋다
> (0.1650~0.1692 < 0.1717) — 착시를 만든다. **현행 판정 레짐은 1-GPU다**(§107) — 모든 arm과
> control을 1-GPU로 맞추고, **DDP4 시절 숫자(0.6880·0.6940·§105 재채점표 27개)와는 빼지 않는다.**
> 이 confound가 §104-6의 v80 −0.0158을 부풀렸다(정당한 control 대비 **−0.0059**).
> 최종 보고를 DDP4로 내야 하면 baseline과 arm을 **함께** DDP4 4 seed로 재측정한다.
>
> **⑧ learnable P는 아직 증명되지 않았다 (§106-3)**: 같은 Hard 난이도·같은 레짐·4 seed에서
> v77(197,057 파라미터) − v81(449 파라미터) = **+0.0048, t=1.5**이고 **seed 44는 −0.0044로 부호가
> 뒤집힌다.** v82−v81의 +0.0101을 "learnable P의 이득"으로 인용하지 말 것 — 난이도(+0.0053)와
> P(+0.0048)가 겹친 값이다.
> ⚠️ **v82/v83가 baseline이 됐다고 learnable P가 승격된 것이 아니다** — 둘 다 v77 계보를 이어받아
> learnable P를 쓸 뿐이고, §105-4의 "v74→v76 승격은 근거 없음"은 유지된다.
> ⚠️ **§107-6(fixed P × Medium, v85)은 취소됐다 (2026-08-13, 사용자 결정)** — 진행할 필요가
> 없다고 판단해 그 실험 계획 자체를 접었다. 새 실험 방향은 재기획 중이다.
>
> **⑨ baseline이 미판정 상태에서 승격된 사례가 생겼다 (§109, 2026-08-13)**: v83(linear head, GELU
> 제거)이 v82 대비 Δ+0.0045, t≈1.15, seed 44 부호 반전으로 **④·⓪의 판정 조건(4/4 부호 일치 +
> |t|≥2.5)을 충족하지 못했는데도** 사용자 결정으로 baseline이 됐다(§108→§109). **판정 게이트
> 자체는 바뀌지 않았다** — 새 arm 비교는 여전히 4/4 + |t|≥2.5를 요구한다. 다만 baseline 자체가
> 이 기준 미달로 승격된 전례가 생겼으니, v83을 인용할 때는 "확정된 승격"이 아니라 "사용자가 미판정
> 상태에서 승격을 결정했다"고 쓸 것 — ⑤가 경고한 "확정된 사실로 쓰지 말 것"과 같은 주의가 적용된다.
>
> **⑩ relation head 깊이 축은 소진이다 (§110, 2026-08-13)**: v84(`12→32→32→1`)는 v82·v83 양쪽
> 기준 모두 4/4 시드 부호 일치 + |t|>3.6으로 **기각**됐다. §108(얕게, 미판정)과 종합하면 head
> 구조 축은 얕음·기본·깊음 세 지점이 다 나왔다 — **더 파지 말 것.**
>
> **⑪ "레이블을 fit에 직접 넣는" 축도 소진이다 (§114, 2026-08-14)**: v88 PA(context 세포에 bag
> 레이블을 상속시켜 세포 단위 ridge를 풀고, bag별 양방향 soft abundance를 feature로 추가 —
> 12→16, `CovarianceMeanLearnablePDDCTPAMLPModel`, arch 57)는 v83 기준 Δ−0.0111, **t=−6.69,
> 4/4**로 **이 레짐에서 가장 강하게 기각**됐다. 코드·config·테스트는 **삭제하지 않고 남겨둔다**
> — `tests/test_population_attention.py`가 §62-2 dead-branch 실패 모드에 대한 살아있는 probe이고,
> 음성 결과의 근거이기 때문이다. **이 형태로 다시 시도하지 말 것.** 또한 §113과 합쳐 **VHL/BAP1의
> 실패를 "소수 population 탐지 실패"로 설명하는 가설은 반증**됐다 — 데이터 주입(§113)도, 전용
> 분기(§114)도 못 고쳤다.
>
> **⑫ 새 relation 분기를 만들 때는 GPU 전에 planted-signal probe를 먼저 쓸 것 (§114)**: v88
> 개발 중 실제 버그 3개가 이 방식으로 잡혔다 — ① 내 freeze 루프가 부모가 learnable로 만든
> `_covariance_projection`을 다시 얼려 P gradient가 `None`이 됐다, ② 분기 설계가 우연 수준
> (같은 signed 축의 top-k/bottom-k-mean이 서로 거울상)이었다, ③ `train_dd_projection=True`
> 경로가 무조건 `no_grad` 안에 갇혀 조용히 죽었다. 셋 다 `nonfinite_gradient_policy: zero`
> 아래에서는 **학습이 정상처럼 보인다** — 테스트 없이는 GPU 시간만 태운다.

> [!IMPORTANT]
> **DD 계약 — 미분 금지 + gradient 개방 금지 (§100·§103, 2026-08-12)**
>
> **① rank-1 방향은 어느 arm에서도 미분하지 않는다.** eigh backward가 `1/(λ_i−λ_j)`를 갖고,
> `+shrinkage·trace·I`는 고윳값을 **균일하게 밀어 간격을 바꾸지 않으며**(forward `rsqrt`만 보호),
> 방향 선택은 hard argmax라 불연속이다. `nonfinite_gradient_policy: zero` 때문에 이 실패는
> **조용하다** — DD 경로를 손대면 **P의 gradient가 finite·nonzero이고 control과 다른지를 테스트로
> 단정할 것.** 미분 가능 우회(Newton–Schulz + `A²` power iteration)는 미구현이다.
>
> **② `train_dd_projection`은 기각됐다 — 되살리지 말 것.** weight 0/0.02/1.0에서 SEAL macro가
> 0.6873/0.6869/**0.6826**으로 **단조 악화**하고 무가중은 fold-paired Δ −0.0047
> [−0.0082, −0.0013]로 CI가 0을 제외한다. DD는 P를 실제로 움직이지만 **그 방향이 해롭다.**
> 무가중은 er_status만 +0.0277로 올려 단독으로 보면 오판하게 만든다(§71 패턴).
>
> **③ DD가 어느 P를 읽는지는 arm마다 다르다** — v74 fixed / v77 CV가 학습한 P / v79 fixed(분리).
> DD는 자기 사영을 갖지 않고 CV의 covariance를 재사용한다. 상세 표는
> `current_architecture.md` **G-0**, 명세 전체는 **G절**.

> [!IMPORTANT]
> **v79 dual projection 계약 (§103) — 기각됨 (Δ −0.0105 [−0.0137,−0.0074])**
>
> `DualProjectionCVDDCTMLPModel`, **`architecture_version = 56`** — v77 ckpt와 strict-load
> **비호환**이다. CV는 learnable P를, **fixed-P CV와 DD는 고정 sin/cos 기저**를 쓰고 CT까지
> 4 branch × 4 feature = **16 → 32 → 1**이다. descriptor는
> `[cov_learnable 8,256, mean 1,536, cov_fixed 8,256]` = **18,048**이며 세 block을 각각 독립
> context-only center/scalar-RMS로 정규화한다(raw bag mean은 두 CV branch가 공유).
> trainable **197,185** (P 196,608 + head 577). fixed 사영은 `_fixed_covariance_projection`
> buffer(`persistent=False`)로 `super().__init__` 직후 snapshot한다.
> `train_dd_projection`은 이 클래스에서 **ValueError로 거부**된다(조용한 no-op 방지).
> fixed-P CV를 남기는 이유: fixed P는 **v41_K128이 0.6940을 낸 기저**이고 그것이 여전히 역사적
> 전체 최고다 — 학습된 것이 고정된 것을 대체하는 게 아니라 head가 둘을 저울질하게 한다.
>
> ⚠️ **결과: 세 arm 중 가장 나빴다.** 과소학습이 아니다 — best val_ce 0.1687로 v77의 0.1697보다
> **좋은데** SEAL이 떨어졌다(합성 개선 ↔ 실데이터 악화의 세 번째 사례). head는 네 block에 weight를
> 거의 균등 분산시켰다(31/27/26/17%). **v78 balanced → 무가중 → v79가 −0.0004 → −0.0047 →
> −0.0105로 단조 악화하므로 CV/DD·사영 배선 축은 소진으로 본다** — 이 축에서 새 arm을 설계하지 말 것.

> [!IMPORTANT]
> **v77 ridge calibration 계약 (2026-08-12)**
>
> 기본 v76은 `ridge_log_lambda=log(1)`, `ridge_log_scale=log(2)`를 동결해 P와 relation head만
> 학습한다. `train_ridge_calibration: true`인 전용 arm에서만 두 scalar를 동결 해제하며
> trainable parameter는 197,057→197,059가 된다. 기존 config/checkpoint 의미는 유지된다.
> Hard orthogonal arm 결과는 0.6840으로 v77 baseline보다 낮아 기각했다.

> **Large ragged opt-in**: `data.ragged_training: true`는 batch 1 전용이며 training collator가
> list-of-bags를 보존한다. 기본 dense/padded 경로는 바뀌지 않는다. 2k–16k arm은 v77
> best를 weight-only warm-start해 0.6885를 얻었으나 +0.0012라 파생 실험으로 유지한다(§97–§98).

> [!IMPORTANT]
> **Canonical CV branch 계약 (§86, 2026-08-11)**
>
> 앞으로 CV는 covariance 단독이 아니라 **fixed-projection centered covariance upper
> triangle + 중심화 전 raw bag mean**이다. K128/1536-d에서는 8,256+1,536=9,792차원,
> ICI 512-d에서는 8,256+512=8,768차원이다. 두 block은 context-only로 각각 독립
> center/scalar-RMS 정규화하고 padding을 제외한다.
>
> canonical class는 CovarianceSetTransformerRidgeModel v46 / STCVLPRidgeModel v47.
> v62–v66 replay는 LegacyCovarianceSetTransformerRidgeModel v42 /
> LegacySTCVLPRidgeModel v43을 쓴다. CovarianceOnlyRidgeModel은 historical control이다.

> [!IMPORTANT]
> **모델이 둘이다 (2026-08-10)**
>
> | | A. CV-only | B. Encoder+Ridge |
> |---|---|---|
> | 파일 | `src/models/baseline.py` | `src/models/set_transformer_ridge.py` |
> | 학습 파라미터 | 229개 | 5,010,946개 |
> | SEAL 10개 | **0.6940** (v41_K128) | 0.6526 (기각, §79-6) |
>
> 공유하는 코드는 ridge 솔버(`solve_ridge_system`) 하나뿐이다. 새 모델을 붙일 때
> `ModelInterface`가 요구하는 것은 `forward` / `forward_episode_batch` /
> `_architecture_version` 셋뿐이고 auxiliary는 전부 `.get()` 가드다.

> [!IMPORTANT]
> **§73 prune — 이전 ckpt 호환성 (필독)**
> config 플래그로 끄기만 하던 5개 분기를 **소스에서 삭제**했다. 근거는 v41_K128 ckpt의
> 파라미터별 gradient 실측: 43,198,660개 중 gradient를 받는 것이 **229개**뿐이었다.
> `meta_covariance_only` 플래그 자체도 없앴다 — 선택할 대안이 없으므로.
>
> ⚠️ **prune 이전 체크포인트는 현재 트리로 strict 로드가 깨진다.** 채점하려면
> `8caa96c`에 고정한 worktree `/NHNHOME/BASE/kimds/ICF_pre_prune`를 쓸 것. **유지 필수.**
>
> ⚠️ 대규모 삭제 전에는 출력을 fixture로 녹화할 것. `tests/fixtures/cvonly_golden.pt`가
> 그 예다. 실제로 1 ulp 차이를 잡아냈고, 추적 끝에 "수식은 동일, 텐서 정렬이 달라져
> 커널 선택이 바뀜"으로 규명됐다. fixture는 **도달 가능한 가중치만** 담을 것 —
> 전체를 담으면 691MB가 git에 들어간다.
>
> ⚠️ **아키텍처에 대한 사실을 config 키에 두지 말 것.** VRAM 가드가
> `meta_covariance_only`를 읽고 있었는데 그 키를 지우자 조용히 6층 추정으로 돌아가
> 가드가 무력화됐다. 이제 모델이 `vram_activation_layers`로 직접 선언한다.

> [!IMPORTANT]
> **§74 학습 경로 — dense를 쓸 것**
> `_episode_losses`는 단일 에피소드도 `forward_episode_batch`로 보낸다. 이전에는
> `self.model(...)`(ragged, bag마다 Python 루프)을 불러 **2.4배 느렸다**.
> `tests/test_training_uses_dense_path.py`가 이 경로를 고정한다.
>
> **속도 개선 시 범인이 아닌 것들** (전부 측정으로 배제):
> 로깅(이미 epoch 단위, 지표 계산 3%) / CPU 비동기 생성(GPU 3.2 ms vs CPU 2,579 ms,
> **805배**) / 프리페치 깊이(depth 1 > depth 3).

> [!IMPORTANT]
> **CV-only 계약 (§68)**: `final = cov_res·CV-1 + cov_rel_res·CV-2`.
> ⚠️ **죽은 key는 zeros가 아니라 부재다.** `_validate_representation`이
> `{covariance_sketch, covariance_matrix}`만 허용한다(빠지거나 남으면 ValueError).
> 새 소비처에서 KeyError가 나면 **그게 정상 동작**이다 — 0으로 채우지 말고 분기를 가드할 것.

> [!IMPORTANT]
> **CV-2는 더 파지 말 것 (§75·§76·§77)**
> margin activation(identity −0.017), subspace_rank(±0.001), head 구조(paired_head −0.0003)
> 셋 다 SEAL 10개 평균을 못 움직였다. v43(T→34.0)과 v44(T→2.84)는 **10개 task 전부 셋째
> 자리까지 일치** — CV-2의 출력 스케일은 성능과 무관하다.
> 다만 `paired_head`는 **라벨 대칭성**을 정확히 만든다(`learned_head`는 4.4e-2로 깨져 있다).

> **clipping 금지 (§67 실측)**: `gradient_clip_val: 1.0`은 er_status를 **−0.0317** 떨어뜨린다.
> `nonfinite_gradient_policy: zero`는 non-finite가 없으면 no-op이라 안전하다.

> **sketch 기하 계약 (§69·§70)**: **`a = 0.85π/K`로 대역폭을 고정해야 K 스윕이 공정하다.**
> ⚠️ ridge-only 진단은 학습 arm의 이득을 과대평가한다(예측 +0.016 vs 실측 +0.004).

> **평가 기준 (§71, 필수)**: 판정은 **SEAL 10개 task macro 평균**
> (`seal_univ2_baseline_17tasks.csv`의 `in_seal=yes`). **er_status 단일로 판정하지 말 것.**
>
> ⚠️ **점추정 macro끼리 빼서 판정하지 말 것 (§99, 2026-08-12 사용자 지시)**. task 내 fold 산포는
> ±0.09인데 판정 대상 Δ는 0.001~0.012다. 모든 arm이 같은 공식 fold를 쓰므로 **fold별로 먼저 뺀
> 뒤 평균**해 CI를 낸다 — `scripts/compare_arms_paired.py`. GPU 불필요, 저장된 예측만 읽는다.
> realization(학습 seed) 노이즈는 pairing으로 줄일 수 없어 별도 seed 반복이 필요하다.
> 평가기에는 모델 내부를 모르는 **generic 경로**가 있다(aggregator 없는 모델용).
> `ICF_FORCE_GENERIC_EVAL=1`로 알려진 모델을 그 경로에 태워 검증할 수 있다.

> **YAML 함정 (§79)**: `lr: 2e-05`는 **문자열**로 파싱된다(소수점과 부호 있는 지수 필요).
> 값을 출력하면 숫자처럼 보이므로 **타입을 볼 것**. `tests/test_config_numeric_types.py`가
> 모든 `train_*.yaml`을 검사한다. `optimizer_overrides`는 지원되지 않는다 —
> LR 변형은 `configs/optimizer/*.yaml`을 만들어 연결한다.

> **GPU 정책 (2026-08-12, 사용자 지시)**: 앞으로 ICF 학습·평가는 **GPU 0–3만 사용**한다.
> GPU 4–7은 사용하지 않는다. 4-GPU arm은 0–3에서 하나씩 순차 실행한다.

> [!IMPORTANT]
> **합성 response/cardinality 계약 (§81·§82, 2026-08-10)**
>
> - episode 안에서도 bag마다 `[1,16384]` cell 수를 독립 draw한다. training collate는 최대 **4096**까지 zero-pad하며 초과 bag은 매번 `randperm` subsample한다. mask 밖 padding은 모델 통계/attention에서 제외한다.
> - 생성기는 `response_dim`, `responsive_population_count`, `label_rule(single|xor)`, `random_causal_factors`, `separate_nuisance_rng`를 지원한다. 기본값은 scalar + 1 population으로 기존 경로를 exact 보존한다.
> - XOR label은 두 causal factor bit가 다를 때 1이다. 각 단일 causal factor는 label과 marginally independent하다. 8-factor/4-pop은 population당 factor 2개로 모든 factor가 실제 state/covariance 효과에 도달한다. sparse task는 현재 arm에서 제외한다.
> - `separate_nuisance_rng`는 donor mixture/shift, component shift, observation noise의 RNG를 label/factor draw와 분리한다. composition은 호환성을 위해 causal response의 scalar aggregate를 쓴다.
> - Arm 1은 **기존 v54 결과**다. Arm 2–5는 v54 계보(`set_transformer_ridge` + AdamW 1e-4)다. v41 CV-only로 잘못 시작한 `logs/20260810_143000/`은 폐기다.
> - 현재 사용자 GPU 허가는 **0–3**까지다. 실행 매핑과 상태는 `current_status.md` §82.
> - 완료 결과: v57 0.6127 / v58 0.5530 / v59 0.5616 / v60 0.6090 / v61 orthogonal 0.6157. 모두 v41 0.6940 미달이며 승격하지 않는다(§82·§83).
> - v62부터 per-bag raw cardinality는 `[256,8192]`, log-uniform power 2.0이며, 4096 cap을 dense episode 생성 전에 적용한다. 이전에는 raw Nmax로 전 bag을 먼저 생성해 OOM이 났다(§84).

> **죽은 분기 (§68-1 실측)**: Q-5 population attention은 **상수를 뱉는다**(AUROC 0.5000,
> std 0.0000). v36 Q1과 v37이 겨냥한 것이 바로 이 모듈이라 둘 다 Δ≈0으로 끝났다.
> 새 아키텍처 arm을 설계하기 전에 **그 분기가 살아 있는지부터**
> `scripts/diagnose_branch_contributions.py`로 확인할 것.
>
> **clipping 금지 (§67 실측)**: `gradient_clip_val: 1.0`을 넣으면 er_status 50-fold가
> **−0.0317** [−0.0450,−0.0183] 떨어진다(플래그 동일, clipping만 상이). non-finite가 없던 arm에
> 없던 불안정을 만든다. `nonfinite_gradient_policy: zero`는 non-finite가 없으면 완전한 no-op이라
> 안전하지만, clipping은 기본으로 켜지 말 것.
>
> **sketch 기하 계약 (§69·§70)**: `aggregator_covariance_slopes`(기본 null = 하드코딩
> (0.019, 0.011) 재현)와 `aggregator_covariance_matrix_dim`(기본 32 = 현행, null = K 연동)이
> config 손잡이다. **`a = 0.85π/K`로 대역폭을 고정해야 K 스윕이 공정하다** — `a` 고정 시
> 대역폭(`a·K`)이 함께 변해 차원 효과가 가려진다. 0.85는 가드밴드(`a·K = π`면 sin 항 소멸).
> **실측 이득은 차원이 아니라 이 두 손잡이에서 나온다**(K 고정 +0.0271, 차원 증설 +0.0043).
> ⚠️ 두 손잡이가 아직 분리되지 않았다(§70-3).
> ⚠️ ridge-only 진단은 학습 arm의 이득을 과대평가한다(K 64→128 예측 +0.016 vs 실제 +0.004).
>
> **평가 기준 (§71, 필수)**: 판정은 **SEAL 대상 10개 task macro 평균**으로 한다
> (`seal_univ2_baseline_17tasks.csv`의 `in_seal=yes`). **er_status 단일로 판정하지 말 것** —
> §70까지의 모든 arm 선택(CV-only, ridge ablation, v36/v37 기각, K·대역폭·CV-2)이 er_status
> 단일 기준이었고, 10개로 넓히니 SEAL 상회 주장이 무너졌다(3/10). **er_status가 10개 중 가장
> 유리한 task였을 가능성이 높아 과적합 위험이 실재한다.**
>
> **아키텍처 판단의 전제 3건 (§65·§69 실측, 다음 arm 설계 전 필독)**:
> 0. **합성 val 지표(val_ce·val_AUROC)로 arm을 고르지 말 것.** CV-only의 합성 val AUROC는
>    ep0 0.8885 = ep49 0.8882로 평평한데 er_status는 +0.037 오른다. **판정은 er_status
>    50-fold로만**(45초). 단일 측정 요동이 ±0.05라 **seed 반복 필수**.
> 1. **val_ce로 arm을 고르지 말 것** — v37 쌍은 val_ce가 더 좋았으나(0.3354 vs 0.3402) 50-fold는
>    **−0.0068**로 나빴다(CI가 0 제외). 200 epoch은 합성 생성기에 과적합한다.
> 2. **학습 길이가 다른 arm 간 비교는 그 자체로 교란** — control은 항상 같은 epoch 수로 새로 학습한다.
>
> **probe 해석 주의**: §62-4의 P0-slots probe(+0.16)는 **"token에 정보가 존재한다"**는 측정이지
> **"학습된 모델이 그 정보로 라우팅할 수 있다"**가 아니다. §65가 실증적으로 분리했다.
> [!WARNING]
> **완료된 공식 50-fold 9개 수치는 `5869535`(tanh margin fix) 이후 stale하다** (§59.5). 같은 ckpt·같은 폴드로 bc_therapy/er_status fold 1-3이 `0.43913/0.78696/0.69130` → `0.4348/0.7565/0.7217`로 바뀐다. HEAD를 별도 worktree에서 돌려 **내 변경이 원인이 아님**을 확인했다. §53 표는 재실행 후 갱신할 것.
>
> **평가 경로 계약 (§59.3)**: aggregator의 ragged/eval 경로는 기본적으로 **bag 단위 정확 스트리밍**(`stream_eval_bags`, 기본 on)을 쓴다. `_bag_view`/`_covariance_sketch`를 나중에 계산하고 보관하지 않을 뿐이라 **수치가 동일**하며(9개 representation key `‖Δ‖∞<1e-4`, anchors bit-identical, 실데이터 AUROC 동일), peak VRAM은 **40,990 → 18,930 MiB**로 준다. A/B는 `BAGPFN_DISABLE_BAG_STREAMING=1`. 훈련(dense) 경로는 손대지 않았다.
>
> **VRAM 가드 계약 (§59.3)**: `estimate_training_vram_bytes`는 이제 **`episode_batch_size`를 반영**한다(이전엔 무시 → 4× batch가 공짜로 보였다). 배수는 실측 기반 재교정(21× → 7×; 실측 v34 6.0×·v35 6.5×). peak는 **스텝당 총 cell 수**(batch × bags × cells)에 비례한다는 것이 핵심 불변식이다.

**§57 진단**: 50-fold 재개 전 확인 결과 **계산은 정상**, 기존 5-fold CV는 **case leakage**(cv5 slide-level split, 108 case 중 82개가 fold 간 분산)로 lscc_arid1a 0.908이 부풀려짐 → 공식 50-fold **0.462가 정직한 값(실질 랜덤)**. 공식 50-fold는 case-disjoint라 **재개 안전**. config 시스템 v34 base + group default 참조형 + 재아카이빙(§56). **폐기 분기 최신화**: CCER(v31)·DR-CCER(v32) 제거(검증 완료, 32 tests 통과), 백업 태그 `repro-pre-deprecated-cleanup-20260807` + `src/repro_backup_20260807/`. 로컬 ccrcc CSV 오류 정정(§51), SEAL baseline 비교(§52).

**Confirmed baseline**: v30 = v24 residual+bottleneck bag projection + B1
`bag_representation: poolz_l2` + B2 log-uniform cardinality `[1,1024]`. Musk zero-shot
`0.8539`, 기존 대형 합성 분포 `0.9483`; 상세는 [`current_status.md`](current_status.md)
§29·§28이다. 코드 기본 `bag_representation`은 checkpoint/config의 조용한 의미 변경을 막기
위해 계속 `legacy`다.

**확정 — v34 large-context + 아키텍처 효율화 (2026-08-07)**: PathoBench 규모(3k~30k+ 타일)
컨텍스트 학습을 위한 MLA 계열 효율화를 커밋·적용했다. ① `src/models/mla.py` standalone
MLA(`bfaee6a`), ② aggregator **slot MLA 저랭크 affinity** (`aggregator_slot_latent_dim`/
`slot_query_latent_dim`/`slot_affinity_dim` + `slot_w_dq/dkv/uq/uk`, `e98b3e2` — None이면
full-dim dot과 byte-identical, 파라미터 0), ③ **slot_std 분산 트릭**(`17a1c36`, [cells,slots,dim]
텐서 제거, default 경로 byte-identical), ④ **배치 population candidates**
(`_population_candidates_batched`, `7700e85` — 수치 동일, **훈련 전용**), ⑤ **정규화 통합**
(`_instances_are_unit`, `778b40b` — 수치 동일). eval은 항상 per-bag 루프(배치 경로의
[C,max_cells,1536] 패딩 OOM 방지, `000aead`). config: `train_v34_phase0_largectx_512.yaml`
([1,32768]) / `..._1536.yaml`(1536-d, [1,8192]), 둘 다 scratch + slot MLA. **v34-1536
(1024ep×50, batch=4, fp32) 완주** — best val_ce 0.4419
(`checkpoints/20260806_215800/v34_phase0_largectx_1536/epoch=048-...`).

평가(§50): ① PathoBench **17개 binary task 5-fold CV 평균 pooled 0.843** (LUAD/LSCC 유전체
task 0.91~0.99 강세). ⚠️ **§51 정정: 로컬 `cptac_ccrcc_{er,grade,her2}` CSV가 `bc_therapy`
복사본으로 확정 — 실제 CPTAC-CCRCC 코호트는 미평가, 실측 6개 데이터셋·14개 유효 task.** ② Musk — `test_musk.py`가 config
input_dim 동적 패딩 + `--pad-mode`(zero/tile, `4aca7f1`/`6d4c5bc`): **tile(166×9+42) 0.858**
vs zero-pad 0.822 (v30 0.854와 동등). ③ **ICI 실세계 5-seed 0.512±0.027 = 랜덤** (명시적
잠금 해제, `f8181be`: `ICIDataset` input_dim/pad_mode 타일 + `test_v34_phase0_largectx_1536_ici.yaml`).
v30과 CV 직접 비교는 **PCA-per-fold 미지원으로 보류**. 상세 §49·§50.

**v34 확정 (§53·§56, 2026-08-07)**: 사용자 결정으로 **v34-1536을 PathoBench 보고용 모델로 확정**.
평가는 **공식 Patho-Bench 프로토콜**(공식 k=all.tsv의 50-fold, 공식 코호트 245장, 공식 라벨
`config.yaml` task_col)로 진행 — **6/17 완료(pooled)**: bc_therapy er 0.672/grade 0.713/her2 0.670,
cptac_brca_PIK3CA 0.569, brca_TP53, **cptac_lscc_ARID1A 0.462**. **잔여 11개는 배치 일시정지
상태** (사용자 요청, `scripts/run_official50_batch.sh`로 재개). 이전 배치가 아카이빙된
`train_v24_musklike_easy.yaml`을 참조해 전부 실패했던 회귀를 v34 config 자체 포함/default
참조화로 해결. **폐기 분기 최신화**: CCER(v31)+DR-CCER(v32) 제거, 검증(파라미터·forward·
checkpoint·32 tests) 완료. v30은 합성/Musk baseline
유지. SEAL(지도 ABMIL/MeanMIL)과는
프로토콜(지도 vs zero-shot in-context)·코호트(ccrcc 218 vs 245) 차이 명시. 상세
§52·§53·§56.

**Active — v35 데이터 단독 arm (2026-08-07, §59)**: v35 제안서는 **rev.2로 개정**됐다
([`history.md`](history.md)).
rev.1의 결정 3건 중 **①rare branch 제거와 ③context/query 분리 대형화는 폐기 권고**다: anchor 후보가
bag당 32개 고정이라 chunk 분할 시 대형 bag이 anchor를 지배하고(§59.1), 집계표의 `global_summary`는
1차 모멘트가 아니라 표준편차이며 `covariance_matrix` 보정식은 `_bag_view`가 버리는 chunk 평균을
요구하고, `query_num_cells [3000,50000]`은 Musk 소형 bag 학습을 없애 **확정 목표(Musk 0.95)와 충돌**하며,
query 위치는 `_sample_training_queries`가 훈련 스텝에서 뽑으므로 dataset이 알 수 없다. 동기 자체도
반증됐다 — context 2,000 tile cap의 pooled AUROC 차이는 **−0.0019**뿐이다. ②chunk는 **근사 평균이
아닌 정확 충분통계 축약**으로 재설계했고(`assignment`가 slot축 softmax라 cell별 독립 → slot 통계는
순수 합, 후보는 online softmax, tail/rare는 분산 top-k merge로 전부 정확), 이 경우 **rare branch를
지울 이유가 없다**. 구현된 것은 **bag 단위** 스트리밍까지이며 **chunk 단위(bag 내부)는 미구현**이다.
학습 arm은 **데이터만 바꾼 단독 arm**: `num_cells [1,32768]` + `num_cells_log_uniform_power 1.5`
(`P(n≤34)=19.8%`로 Musk 밴드 보존, `E[n]=4487` = v34의 4.94×), `episode_batch_size 1`로
v34와 **동일 cell envelope**(3.28M cells/step), 51,200 episodes로 **에피소드 매칭**.
`logs/20260807_203606/`, 2×B200(GPU 0·1). **P0 게이트(무료)** 미실행: query 크기 스윕이 +0.005
미달이면 대형화 노선을 접는다(rev.2 §4).

**Active — v36 재정의: 40→1 압축 해제 (2026-08-08, §62)**: 원안
region **chunk** attention 제안서(폐기·삭제 2026-08-08, git 기록 보존)는 **핵심 전제 3건이 코드·실측으로 반증**됐다 —
① 합성 bag의 cell은 exchangeable(`synthetic_data.py:322`)이라 sequential chunk에 **학습 신호가
구조적으로 없다**, ② "선택 기제 부재"는 사실이 아님(`_instance_attention_mil_logits`가 이미
존재, §24 기각은 §31 측정 6이 무효 선언), ③ region 수는 15가 아니라 **median 3.4개**
(슬라이드 57%가 ≤4, 18%가 1개). **사용자 결정: 좌표(coords) 미사용** → chunk-region 노선 폐기.
재정의된 문제는 **좌표 없는 slot 기반**이다. 상세 §62. → [새 Q1 proposal](architecture_v36_q1_structured_population_proposal.md).

> [!IMPORTANT]
> **아키텍처 계약 — routing softmax 무력화 (§62-2, 실측)**: `project_structured_tokens: true`
> (v34/v35 기본)에서 `_projected_bag_tokens`가 bag의 **구조 token 40개**(global_summary 1 +
> slot 12×3 + tail 3)를 **라벨 정보가 들어오기 전에** 고정·라벨 무관 선형사상으로 **1개로 압축**한다
> (위치별 `Linear(1536→64)` 40개 + concat 2560 + exact mean residual 1536 → `Linear(4096→1536)`;
> mean pooling이 아니다). 그 결과 `_population_memory_logits`(baseline.py:3509)의 routing
> softmax가 **길이 1 축**에 걸려 `population_slot_weights`가 **shape (Q,1), 값 전부 1.0**이 된다 —
> ABMIL형 선택 기제가 구현돼 있으나 **무력**하다. `routing_sparsity_weight`/`routing_balance_weight`가
> 둘 다 `0.0`인 것도 같은 정황. **P0-slots probe 실측: 이 압축이 버리는 정보는 EGFR +0.1597 /
> STK11 +0.1577 (fold-paired, 95% CI가 0에서 멀리 떨어짐)**.
>
> **slot 수 계약 (§62-3, 실측)**: **aggregator에는 `num_slots`에 의존하는 파라미터가 없다**
> (12 vs 24에서 29개 텐서 shape 완전 동일; anchor는 데이터 유래, slot encoder는 공유).
> 전체 모델에서 shape 불일치는 **`meta_classifier.bag_token_projection.weight` 단 1개**
> (+ `bag_token_bottlenecks` 개수). 따라서 ⓐ frozen ckpt로 임의 slot 수의 구조 token을 뽑을 수 있고,
> ⓑ num_slots 변경은 ckpt 비호환이지만 **weight-only warm start**가 가능하다.
>
> **eval 캐싱 계약 (§62-3 발견 → §64 구현, bit-identical)**: pool 통계(`_context_pool_stats`)와
> anchor(`_context_anchors`)는 **context bag 전용**이고 `_bag_view`/slot 통계는 per-bag이므로,
> 한 폴드의 표현을 **1회 패스로 전부 계산**해도 쿼리별 패스와 **‖Δ‖∞ = 0.000e+00**이다.
> `evaluate_trial`이 **기본으로 캐싱**하며(`--no-cache-context`로 A/B) meta-classifier는 여전히
> **쿼리당 1회** 호출해 `--batch-queries`가 깨뜨렸던 `_covariance_relation_scores` 단일 쿼리
> 거동을 유지한다. er_status 50-fold 실측 **356s(25 worker/2 GPU) → 50s(1 worker/1 GPU)**,
> 1,650 쿼리 `max|Δp| = 0.000e+00`. `tests/test_context_cache_equivalence.py`가 고정.
>
> ⚠️ **캐싱 가드 2개** (둘 다 만족해야 사용, 아니면 자동 폴백): ⓐ `context_mode == "all"`,
> ⓑ **`context_limit is None`** — `--max-tiles`/`--context-max-tiles`를 주면 공유 `generator`가
> 쿼리마다 전진해 **context 부표본이 쿼리마다 다르므로** 캐싱이 한 draw를 고정해버린다.
>
> **병렬 워커는 무효 (§64-3 실측)**: GPU당 fold 처리율이 13 worker/1 GPU와 25 worker/2 GPU에서
> **14.2 s/fold로 동일**하다(메모리는 32%만 사용) — 이미 연산 포화이므로 워커 증설은 소용없다.
> 러너는 `--workers 26`을 줘도 `chunk=ceil(50/26)=2` 때문에 **25 워커**를 띄운다.
> ⚠️ 워커들이 `{tmp_dir}/{task}_official_folds.ckpt` **하나를 공유**하고 완료 fold를 건너뛰므로,
> fp32 시절 캐시가 남아 있으면 **정밀도가 조용히 섞인다** — 재실행 시 새 `--tmp-dir`을 쓸 것.

**결과 — v36 Q1 / v37 모두 기각 (§65, 2026-08-09)**: Q1(40→1 압축 해제)은 er_status 50-fold
fold-paired **−0.0024** [−0.0058, +0.0006], v37(context-adaptive 압축)은 **−0.0001**
[−0.0040, +0.0039]로 둘 다 +0.005 게이트 미달이며 부호도 음수 쪽이다. 아래 §62 관련 서술은
**진단으로서는 유효하나 처방으로서는 반증**됐다. v37은 **label-free**라 §62-2 진단의 절반만
답했고, **라벨 조건화는 미검정 레버**로 남는다.

> [!IMPORTANT]
> **ridge ablation 계약 (§66, 2026-08-09)**: 세 closed-form ridge solve를 독립 제거하는 config
> 플래그가 있다 — `meta_enable_global_ridge`(G-2) / `meta_enable_abundance_ridge`(P-2) /
> `meta_enable_covariance_ridge`(CV-1), **기본 전부 `true` = 현행 동작**. 각 플래그는 자기 ridge
> 항만 0으로 만들고 그 분기의 학습 residual은 남긴다(분기 전체가 아니라 **ridge 하나를 격리**).
> dense/ragged **두 경로 전부**에 배선, 신규 파라미터 0개, shape 보존 → ckpt strict 로드 양방향.
> ⚠️ **ablation된 ridge 파라미터는 gradient를 받지 않아 init 상태로 남는다** — 그 ckpt는 **반드시
> 같은 플래그로 평가**할 것(rare-free와 같은 함정). `tests/test_ridge_ablation.py` 8개가 고정한다.
>
> **실측**: **G-2는 무기여**(Δ −0.0004, CI가 0 포함, 22/50 — control·arm 둘 다 50ep 정상 완주).
> **P-2·CV-1은 제거 시 학습 붕괴**(P-2 ep13 non-finite gradient 크래시, CV-1 발산·best=ep0) —
> 다만 학습 길이가 달라 **그 AUROC 수치는 공정 비교가 아니라 참고용**이다.

**운영 함정 2건 (§66-5, 이번 세션 실측)**:
1. **launcher wrapper가 torchrun child보다 먼저 종료한다** — wrapper PID만 kill하면 GPU가 계속
   잡혀 있다(실측 153 GB 잔존). **프로세스 그룹**(`kill -TERM -$pgid`)으로 죽일 것.
2. **`while pgrep -f "scripts/train.py"` 대기 루프는 자기 자신에 매칭돼 영원히 끝나지 않는다** —
   그 bash 프로세스의 커맨드라인에 패턴이 들어 있다. launcher 로그 + 프로세스 부재를 **함께**
   확인하거나(`scripts/queue_v38_wave2.sh`) 패턴이 자기 자신과 겹치지 않게 쓸 것.

**진행 방침 (사용자 결정, §62-6)**: **zero-init gate를 쓰지 않고 아예 변경**한다 — population 분기의
모든 파라미터가 token 개수가 아니라 `token_dim`/`hidden_dim`으로만 크기가 정해져 이 변경은
**shape 보존(ckpt strict 로드, 신규 파라미터 0개)**인데 게이트가 그 성질을 깨고, 이 리포의 zero-init
게이트는 열리지 않은 전력이 있어(v31 예측 상관 0.99928, rare는 floor 강제에도 |Δ| 0.0009)
Δ≈0이 나오면 가설 기각과 게이트 미개방을 **구분할 수 없다**. 대신 config 플래그
`meta_population_token_mode: projected | structured`(기본 `projected` = 현행)로 가역성만 확보하고,
**`_population_memory_logits`(eval/ragged)와 `_population_memory_logits_batched`(훈련/dense) 두 경로를
모두** 바꾼 뒤 동치 테스트를 붙인다. **num_slots 증설은 §62-5 부호 불일치로 보류**.

**Rejected candidate — architecture v31 CCER-v2**: projection 전 aligned slot-center로
support class prototype을 만들고, 기존 rare branch와 독립인 support/query encoder에서
class-centered cell evidence를 계산한다. `Top-1`, `Top-4`, `mean` route는 총 `0.30`의
floor를 가지며 별도 null gate는 없다. 최종 output head는 zero-init이므로 v30 weight-only
초기화 직후 logits가 정확히 동일하다. 신규 module은 base LR, 공통 v30 backbone은 `0.05x`
LR을 사용한다. Config는 `configs/archive/v31/train_v31_ccer_v2.yaml`, architecture marker는 `31`이다.
Seed 42 20-epoch 학습 best는 epoch 18 `val_ce_loss=0.443786`이었으나 synthetic AUROC
`0.8514`, Musk `0.8470`으로 v30 Musk `0.8539`를 넘지 못했고 대형 bag은 `0.698`로
동일했다. 따라서 미채택이며 재현용 코드만 보존한다. 상세는
[`current_status.md`](current_status.md) §35·§36이다. 현재 활성 v31 학습은 없다.

**Proposed next investigation — v32 DR-CCER**: CCER-v2 예측은 v30과 synthetic 상관
`0.99928`, Musk 상관 `0.99311`이고 Musk `n>34`가 `0.69841`로 완전히 동일했다. 따라서
단순 slot/Top-K 확대 대신 donor-resolved support evidence와 independently supervised expert,
reliability-gated mixture를 제안한다. 구현 전 P0–P2 checkpoint 진단이 필수다. 상세는
[`history.md`](history.md)와
[`current_status.md`](current_status.md) §37이다. 아직 구현·학습 승인 또는 활성 run은 없다.

**Active — v32b DR-CCER (2026-08-05)**: v32 원안의 비판적 재검토 개선안
([`history.md`](history.md))을 작성하고,
P0–P3 probe(`scripts/archive/probes_smoke/probe_v32_headroom.py`) + DR-CCER 아키텍처(`architecture_version=32`,
donor-resolved expert + reliability-gated convex mixture)를 구현했다. **결과: CCER 계열 실증적
폐기** — ① Stage A 학습(`20260805_182126`, 10 epochs)에서 donor-resolved expert standalone CE가
0.693(무작위) 정체, ② Stage-0 probe에서 P2 fusion headroom **-0.00034**, P3 donor-agreement
headroom **+0.00000** (둘 다 게이트 +0.005 미달), CCER-v2 standalone branch AUROC 0.51(무작위,
v30과 corr 0.0096). 따라서 v32 미채택, 재현 코드만 보존, v30 baseline 유지. 상세는
[`current_status.md`](current_status.md) §38이다. **다음 방향**: 데이터 측 — Phase 1 "v30 on
6-task mix"(any_positive_sparse 포함) 재학습, 소형 bag(n≤4)·n>34 분포 레버. 새 세션은 §38부터 읽을 것.

**Active — v33 Phase 0 (2026-08-05)**: v33 MR-BagPFN proposal의 §9 지침대로 **arm B(v30 +
six-task + B2)와 arm C(v30 + legacy + B2b) 데이터 컨트롤을 먼저 구현·런칭**했다. B2b는
`SyntheticManifoldGenerator(per_bag_cardinality=True)`로 에피소드 내 per-bag
`n_b ~ LogUniform[1,1024]`을 추첨해 ragged list-of-bags를 반환하는 새 데이터 경로다
(collator/training_step ragged 분기, `episode_batch_size=1` 필요). config:
`configs/archive/v33/train_v33_phase0_armB.yaml`·`armC.yaml`. 신규 테스트 `tests/test_b2b.py` 10개 포함
기본 suite는 현재 **41 tests / 약 48초**다 (§59 기준). 학습: arm B는
`logs/20260805_220642/`에서 BF16으로 기존 checkpoint를 복원해 계속 진행한다. 최초 arm C
`logs/20260805_214751/`는 batch 1에서 4096 updates/epoch가 되어 중단했다. 해결 run은
`logs/20260805_220843/`: train episode를 512/epoch로 줄여 arm B와 동일한
**512 optimizer updates/epoch**를 사용하며, ragged B2b와 v30 architecture는 유지한다
(실측 약 3분/epoch, 전체 약 2.5–3시간). **Phase 1 frozen-v30 multi-resolution probe는
Phase 0 결과 확인 후에만 구현한다.** 상세는
[`current_status.md`](current_status.md) §41·§39다.

**Active — arm C top-up, 8×A6000 DDP (2026-08-06)**: §42에서 arm C가 과소학습 편향
(에피소드 8× 부족)으로 gate 미달이었으므로, 사용자 결정으로 **8×RTX A6000 DDP +
에피소드-매치**로 재개했다. 새 config는 `configs/archive/v33/train_v33_phase0_armC_ddp8.yaml`
(자체 포함형, medium 체인 미상속, `episodes_per_epoch: 4096`, devices 8 /
`ddp_find_unused_parameters_false` / bf16-mixed / max_epochs 150)이고 `archive/v33_phase0_armC_bf16/last.ckpt`에서
resume한다. **gnode5 필수**: 이 머신의 NCCL P2P/CUMEM 전송이 hang을 일으켜
`scripts/launch_interactive_training.sh`에 `NCCL_P2P_DISABLE=1`을 기본 적용했다
(진단용 `scripts/archive/probes_smoke/nccl_probe.py` 신규). B200 1장 대비 A6000 1장은 ~1.8× 느리지만
8장 병렬로 노드 총 처리량은 ~4.3× (상세 표는 [`current_status.md`](current_status.md) §43).
**다음**: top-up 완주(150 epoch) 후 §42 재평가.

**Proposed next investigation — v33 MR-BagPFN (아키텍처)**: CCER와 다른 새 cell evidence를
만들지 않고, 검증된 v30 bag representation을 동일 bag의 full/partition/subsample view에서
공유해 sampling resolution 정보를 보존한다. 단, Phase 0(arm B/C) 결과와 frozen-v30
multi-resolution combiner의 paired AUROC `+0.01` headroom 확인 후에만 구현한다. 상세는
[`history.md`](history.md)와
[`current_status.md`](current_status.md) §39다.

**Persistent invariants**: ICI는 사용자 지시로 잠금 상태다. 잠금 해제 시
`src/datasets/base_data.py`의 cell-axis zero-padding이 bag mean/global spread를 오염하는 문제를
먼저 처리한다. 이전 v24/v25/v26/IA-MIL 결정과 config 복구 기록은 §25~§29 및
[`history.md`](history.md)에 보존한다.

이 문서는 BagPFN 저장소를 처음 맡은 coding agent가 안전하게 작업을 시작하기 위한 운영 및 핸드오프 지침입니다. 최신 개발 및 실험 진행 상황은 [`current_status.md`](current_status.md), 현재 모델 명세는 [`current_architecture.md`](current_architecture.md), 현재 실험 프로토콜은 [`current_experiments.md`](current_experiments.md)를 참고합니다.

---

## 1. 새 세션 접속 Agent의 최우선 정독 및 Git 파악 원칙 (New Session Protocol)

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 행동 수칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 접속한 AI Coding Agent는 세션 간 맥락 단절을 방지하기 위해 **`docs/` 최상위 루트의 Living `.md` 파일 5개와 현행 `architecture_*_proposal.md` 1개를 최우선으로 즉시 정독**합니다.
> 3. Living 문서 정독 직후, **반드시 Git 상태 및 최신 커밋 내역/Diff를 확인**하여 이전 세션의 정밀 코드 변경점과 작업 히스토리를 파악합니다:
>    ```bash
>    GIT_OPTIONAL_LOCKS=0 timeout 3s git status -uno
>    GIT_OPTIONAL_LOCKS=0 timeout 3s git --no-pager log -n 5 --stat
>    GIT_OPTIONAL_LOCKS=0 timeout 3s git --no-pager diff HEAD~1 HEAD
>    ```
> 4. Living 문서 5개, 현행 proposal, Git commit log/diff를 종합하여 확정 baseline, 코드 수정 내역, 완료된 실험 수치, 미결 과제 및 다음 Action Plan을 이어받아야 합니다.

---

## 2. Git 중심 개발 및 세션 핸드오프 수칙 (Git-Centric Workflow)

0. **명확한 다음 단계는 자율적으로 연속 실행**:
   - [`current_status.md`](current_status.md)의 Action Plan과 판정 기준이 명확하면 사용자에게 “진행할까요?”라고 다시 묻지 않고 실행합니다.
   - 진단 결과가 사전 판정 기준을 만족해 다음 단계가 하나로 정해지는 경우, 구현·검증·후속 진단까지 같은 범위에서 계속 진행합니다.
   - 단, 모순되는 선택지, 파괴적 변경, 외부 공개/비용, 누락된 필수 입력처럼 새로운 사용자 판단이나 권한이 필요한 경우에는 중단하고 확인합니다.
1. **잦은 커밋 (Frequent Commits)**:
   - 논리 단위 작업(기능 추가, 버그 수정, 문서 개정, config 정돈, 단위 테스트 작성 등)이 완료될 때마다 즉시 커밋을 수행하여 작업 이력을 세분화합니다.
2. **상세한 커밋 메시지 작성 (Detailed Commit Messages)**:
   - 커밋 메시지는 제목(Subject)과 상세 본문(Body)을 명확히 구분하여 작성합니다:
     - `feat`: 신규 모델 아키텍처, 텐서 연산, 평가 프로토콜 기능 구현
     - `docs`: Living 문서 개정, 아키텍처 스펙 문서화, 작업 수칙 업데이트
     - `chore`: 디렉터리 아카이빙, config 정돈, 환경 파일 설정
     - `test`: 단위 테스트 수트 작성 및 검증
   - 본문(Body)에는 **변경 동기(Why)**, **구현 세부사항(What)**, **검증 결과(Verification)**를 정밀하게 명시합니다.
3. **세션 종료 및 핸드오프 시 커밋 필수**:
   - 턴이나 대화 세션을 마무리하기 전 Working Tree의 모든 변경 사항을 남김없이 커밋하고, 생성된 Commit Hash와 핵심 요약을 [`current_status.md`](current_status.md)에 갱신하여 바톤 터치합니다.
4. **진행상황 follow-up 가능성 보장**:
   - 각 논리 단위가 끝날 때 [`current_status.md`](current_status.md)에 상태, 핵심 수치, 실행 명령, 로그/PID/체크포인트/예측 파일 경로, 성공·중단 판단 근거, 바로 다음 Action을 기록합니다.
   - 실행법이나 모델 계약이 바뀌면 `current_experiments.md` 또는 `current_architecture.md`도 같은 논리 단위에서 함께 갱신합니다.
   - 장시간 작업은 완전 이탈형 백그라운드로 실행하고, 시작 직후 PID와 로그 경로를 기록하며, 완료 후 최종 결과와 산출물 경로를 추가합니다.
   - 다른 작업공간의 Agent가 대화 기록 없이 Living 문서와 `git log`만으로 작업을 이어갈 수 없는 상태는 완료된 핸드오프로 보지 않습니다.

---

## 3. 필수 작업 지침 & Multi-Location 동기화 규칙

1. **연구실 / 집 / 노트북 3원화 대화 동기화 완벽 대응**:
   - 세 장소 간 대화 히스토리가 비동기적이므로, 작업 진행 상황 및 수치/경로는 **반드시 [`current_status.md`](current_status.md)에 상세히 기록**하고 읽는다.
2. **명령어 Hang 타임아웃 필수 적용**:
   - NVML/드라이버/쉘 블로킹으로 인한 대화창 멈춤(Hang)을 방지하기 위해 터미널 조회가 필요한 모든 명령어에는 **`timeout 3s ps aux | grep python`** 또는 `timeout 3s tail -n 20 <LOG>`와 같이 반드시 타임아웃을 강제 적용한다.
3. **완전 이탈형 백그라운드 구동**:
   - 장시간 실행되는 훈련/평가 명령어는 **반드시 `scripts/launch_interactive_training.sh` 독립 백그라운드 스크립트**나 short `WaitMsBeforeAsync` 태스크로 띄운다.
4. **수치 안전성 계약 (2026-08-08 실제 강제)**:
   - 공분산 스케치 역행렬 연산 시 FP16 계수 오버플로우 및 NaN 발생 방지를 위해 **`bf16-mixed` 정밀도를 필수 적용**한다.
   - ⚠️ 이 계약은 §56(v34 group default 신설) 이후 **선언만 되어 있고 강제되지 않았다** — `configs/trainer/default.yaml`이 precision을 아예 설정하지 않아 v34/v35 entry point가 Lightning 기본값 **32-true(fp32)**로 조용히 해석됐다. **확정 v34-1536 ckpt와 v35-16384 ckpt는 fp32로 학습된 것**이며, 지금 재실행하면 bf16-mixed로 돌아가 그 ckpt를 재현하지 않는다(정확한 역사적 재현이 필요하면 `trainer_overrides.precision: 32-true`).
   - **2026-08-08부터 예외 없이 강제**한다 (사용자 결정: "앞으로 항상 bf16-mixed"). `tests/test_precision_contract.py`가 ⓐ 활성 entry point `configs/train_*.yaml` 전부와 ⓑ **선택 가능한 `configs/trainer/*.yaml` group 전부**를 검사하므로, 다른 group을 골라 계약을 우회할 수 없다. 새 학습 config·새 trainer group은 이 테스트를 통과해야 한다.
   - `configs/trainer/ddp5.yaml`·`ddp8.yaml`의 `16-mixed`(fp16) 위반은 **해소 완료** (bf16-mixed로 교체). ddp8 주석의 "RTX A5000은 FP16 경로" 근거는 Ampere가 bf16을 지원하므로 무효다. 이를 참조하던 아카이브 config는 `configs/archive/v18_v19/train_synthetic.yaml` 1개뿐이며 폐기된 v18/v19 아키텍처다(원본 값은 git 이력에 보존).
   - **평가에도 강제한다 (2026-08-08 사용자 결정, §64)**. 이전에는 같은 ckpt가 스크립트마다 다르게 채점됐다 — `evaluate_synthetic.py` bf16 / `test_pathobench.py`·`test_musk.py` fp32 / `test.py` 기본값 **`16-mixed`(fp16!)**. 이제 `src/utils/utils.py`의 **`eval_autocast(device, precision)`가 단일 정의**이고 `add_eval_precision_argument(parser)`로 모든 추론 스크립트가 `--precision`(기본 `bf16-mixed`)을 갖는다. **fp16은 ValueError로 거부**되고 `32-true`만 탈출구다.
   - ⚠️ **2026-08-08 이전에 산출된 공식 50-fold AUROC는 전부 fp32 산출물이며 참고용이다** (사용자 결정). 실측 이동폭: er_status fold 1이 `0.4348 → 0.5130`(**+0.078**), fold 3이 `0.7217 → 0.7609`(§64-1).
   - `configs/archive/`는 폐기 아키텍처의 재현 기록이므로 검사에서 제외한다(근거는 `tests/test_precision_contract.py` docstring).
5. **테스트 검증 필수**:
   - 코드를 변경한 뒤에는 아래 unittest 수트를 통과해야 완결로 인정한다:
     ```bash
     timeout 300s /home/aibio_3/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
     ```
   - 기본 스위트는 현재 **153 tests, 약 42초**다 (2026-08-13 실측 — 이전 판의 "86 tests, 약 225초"는
     오래된 값이다).
     ⚠️ **알려진 기준선 실패 1건**: `tests/test_mlp_manifold_bank.py`가 `import pytest`를 하는데
     `BagPFN` env에 pytest가 없어 **ImportError로 항상 실패**한다(`7de8b70`부터). 즉 정상 상태는
     `Ran 153 tests ... FAILED (errors=1)`이며, **이 1건이 늘어나지 않는지만 확인**하면 된다.
     (§67 nonfinite policy 5개, §68 ridge ablation+CV-only 15개, VRAM 가드 2개 포함) (§59: streaming 7개 + vram 2개, §63: precision 계약 4개, §64: precision-eval 2개 + context 캐싱 등가성 4개, §65: v36 population token mode 6개 + v37 context-adaptive 9개, §66: ridge ablation 8개). 폐기 architecture/연구 진단 175개는
     `tests/history/legacy_*.py`로 이관되어 기본 discovery에서 실행되지 않는다. archive suite는
     수정 대상이 해당 보존 경로일 때만 개별 실행한다.

---

## 4. 작업 위치 및 바이너리 경로 명세

- **Workspace Root**: `/NHNHOME/BASE/kimds/ICF`
- **Python Binary**: `/home/aibio_3/miniconda3/envs/BagPFN/bin/python`
- **Torchrun Binary**: `/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun`
- **Netrc File**: `/NHNHOME/BASE/kimds/.netrc`
- **Target Hardware**: **8× NVIDIA B200 노드, 사용 GPU 0·1 (2장)** (`CUDA_VISIBLE_DEVICES=0,1`, 180GB VRAM/장)

> **2026-08-07 환경 전환 (8-GPU 컨테이너)**: 이전 `/NHNHOME/kimds` 경로는 폐기하고
> 워크스페이스는 `/NHNHOME/BASE/kimds/ICF`로, conda env는 `/home/aibio_3/miniconda3/envs/BagPFN`으로
> 변경됐다. 저장소 내 스크립트·문서·설정의 `/NHNHOME/kimds` 참조는 전부 이 경로로
> 마이그레이션 완료. `logs/`·`predictions/`의 과거 실행 로그(gitignore)는 역사적 기록으로 유지.
> **GPU 배정 (사용자 지정)**: 노드에 8×B200이 보이지만 사용하는 GPU는 **0·1 (2장)**만.

---

## 5. 독립 실행 스크립트 표준 명령 구문

SSH 연결이나 VS Code 터미널이 종료되어도 백그라운드에서 지속해서 안정 구동되는 표준 실행 명령:

```bash
cd /NHNHOME/BASE/kimds/ICF

CUDA_DEVICES=0,1 \
NPROC_PER_NODE=2 \
TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun \
NETRC=/NHNHOME/BASE/kimds/.netrc \
scripts/launch_interactive_training.sh \
  <RUN_NAME> \
  <CONFIG_PATH>
```

훈련 시작 후 반드시 생성된 `logs/{RUN_TIME}/` 경로의 `.out` 로그 tail을 확인하여 정상 작동 여부를 정량적으로 검증하고 [`current_status.md`](current_status.md)를 즉시 갱신합니다.

---

## 6. Documentation 관리 및 아카이빙 규칙 (Docs Organization Rules)

1. **`docs/` 최상위 루트 규칙 (Active Living Docs + Current Proposal)**:
   - `docs/` 최상위 루트에는 새 Agent가 즉시 정독해야 하는 **핵심 Living 문서 5개와 현행 proposal 1개만 존재**해야 합니다:
     - [`agent_handoff.md`](agent_handoff.md): 운영 규칙, 바이너리 경로, Git 수칙, Docs/Config 관리 지침
     - [`current_status.md`](current_status.md): 개발 현황, 최신 수치, Git 커밋 이력, 이슈 진단 및 Action Plan (SSOT)
     - [`current_architecture.md`](current_architecture.md): **CV-only(v40~) 명세** — 공분산 sketch + closed-form ridge(CV-1) + subspace/prototype(CV-2) 2개 분기만. 이전 6-분기 판은 `history.md`
     - [`current_experiments.md`](current_experiments.md): **CV-only 이후 실험 절차** — 판정은 SEAL 10개 macro 평균, 금지된 판정 방식 목록, 표준 실행 명령. 이전 판은 `history.md`
     - [`README.md`](README.md): 전체 문서 맵 및 갱신 규칙
     - `architecture_*_proposal.md`: 현재 활성 개선안 1개. 완료·폐기 시 핵심 결론을 `history.md`에 기록하고 원문은 git 이력에 보존
     (**2026-08-09 현재 활성 proposal 없음** — v36 Q1·v37 모두 기각되어 history.md에 요약)
   - 최상위 Living 문서와 현행 proposal은 항상 서로 일관된 맥락을 유지합니다.

2. **`docs/history.md` 아카이빙 규칙 (Historical & Deep-Dive Docs)**:
   - 과거 딥다이브 분석서·옛 아키텍처 설계안·폐기 proposal·과거 세션 아카이브의 **지속 관리 가치가
     있는 결론(ADR·설계 이유·트레이드오프·레슨런)은 `docs/history.md`의 해당 시기 절에 요약해
     추가**합니다. 개별 파일을 `docs/history/` 폴더에 두지 않습니다(2026-08-09 통합, 원문은 git 이력 보존).
   - 아카이빙 시 기준: ① 주요 결정/설계 이유/트레이드오프 ② 현재 문서 관점에서 향후 참조가 필요한
     맥락 ③ 중요 레슨런. 단순 변경 이력이나 현재 스펙과 중복되는 작업 정보는 제외합니다.
   - 출처(원본 파일명·작성 시점)를 함께 기재해 git 이력에서 원문을 되짚을 수 있게 합니다.

---

## 7. Config 관리 및 아카이빙 규칙 (Config Organization Rules)

1. **`configs/` 최상위 루트 유지 조건**:
   - 현재 활성 파이프라인에서 직접 사용하는 entry point config만 `configs/` 최상위에 유지합니다.
   - **현재 `configs/` 최상위 유지 대상 (§109, 2026-08-13)**:
     `train_v83_linear_head_1536_1gpu.yaml` (**canonical baseline, 자체 포함형** — §109 사용자
     결정, §107-3 게이트 미달인 채로 승격됨을 인용 시 함께 밝힐 것),
     `train_v82_medium_classsep_1536_1gpu.yaml` (직전 baseline, 참고용),
     `train_v82_medium_classsep_1536.yaml` (v82의 DDP4 판본, 참고용),
     `train_v77_hard_orthogonal_1536{,_1gpu}.yaml` (historical control).
     v83 canonical은 v82와 마찬가지로 이전 base 체인을 인라인한 자체 포함형이다 — v34·v77·v82가
     받았던 처리와 같다.
     ⚠️ 기각된 arm의 config(v78 2개·v79·v80 2개·v81 2개)는 `configs/archive/` 아래
     `v78_dd_gradient/`·`v79_dual_projection/`·`v80_v82_seed_batch/`로 이관했다(§107).
   - 종결된 arm 64개는 시대별로 이관됐다: `configs/archive/` 아래 `v34_largectx/`,
     `v40_v45_cvonly/`, `v50_v54_encoder/`, `v57_v61_data_arms/`, `v62_v68_hybrid/`,
     `v69_v76_relation/`, `v77_pop_residual/`. 전부 `base_config` 없는 자체 포함형이다.
   - v30/v24/v22/eval_v30 체인은 `configs/archive/v30/`·`archive/v24/`·`archive/v22/`로 이관
     (2026-08-07 §56 재아카이빙).
   - 폐기 확정 config 이관: v23-A0/v24-A0/v24-B0 → `configs/archive/v23_v24_candidates/`;
     v25 → `configs/archive/v25_typed_bag/`.
   - ICI의 fold/seed는 config에 박지 않고 `--cv` / `--seed`로 주입합니다 (`scripts/launch_ici_protocol.sh`).
2. **구버전 Config 아카이빙 조건**:
   - 구버전 아키텍처의 config는 `configs/archive/` 하위로 즉시 이관합니다: `archive/v18_v19/`, `archive/v20/`, `archive/v21_retrieval/`.
   - 폐기된 기능의 실행 스크립트도 같은 규칙으로 `scripts/archive/`(예: `scripts/archive/v21_retrieval/`)로 옮깁니다.
3. **아카이빙 config는 자체 포함형(인라인)으로 보관 (2026-08-07 신설)**:
   - 아카이빙하는 config는 `base_config`를 남기지 않고 **전부 인라인**으로 변환해 보관합니다 (v34가
     인라인한 방식처럼 `data`/`model`/`optimizer`/`scheduler`/`trainer`/`logger`/`callbacks` 전체 값을
     직접 기술). 이렇게 하면 아카이빙 후에도 자기 디렉터리 기준 상대경로가 깨질 일이 없고, 참조 검증
     없이 항상 재현 가능합니다.
   - 2026-08-07 §56 적용: v34 자체 포함형 전환 + v30/v24/v22 체인 재아카이빙(141개 config 전부 해석
     성공). 이후 아카이빙은 base 체인 config를 root에서 인라인 후 보관.

   > [!IMPORTANT]
   > **아카이빙·삭제 시 참조 검증 필수 (2026-08-04 신설).** `base_config`는
   > `utils.py`의 `_load_train_config`가 **config 자기 디렉터리 기준**으로 해석하므로, config를 하위
   > 폴더로 옮기면 상대경로가 조용히 깨집니다. 또 `resolve_config_group`은 모듈 조각을
   > `configs/<group>/`에서만 찾으므로, 모듈 config를 삭제하면 이를 참조하는 **아카이브** config가
   > 깨집니다. 실제로 2026-08-04까지 **모든** 아카이빙 커밋이 이 검증을 누락해 config 18개가
   > 로드 불가 상태였고 unittest 1건이 상시 실패했습니다 (복구 기록: [`current_status.md`](current_status.md) §26).
   > **아카이빙/삭제 커밋 전 반드시 아래를 통과시킬 것** (활성 config만이 아니라 **전체**).
   > ⚠️ **이전 판의 이 명령은 `if 'base_config' not in p.read_text(): continue`로 걸러서 삭제된
   > module fragment 때문에 깨진 config를 못 잡았다** — 실제로 `configs/archive/v18_v19/` 10개가
   > 그 상태로 "failing: 0"을 통과하고 있었다(§102). 그 줄을 없애고 module 조각 디렉터리만 제외한다:
   > ```bash
   > timeout 300s /home/aibio_3/miniconda3/envs/BagPFN/bin/python -c "
   > import sys; sys.path.insert(0,'.')
   > from pathlib import Path
   > from src.utils.utils import merge_train_config
   > MODULES={'callbacks','data','logger','model','optimizer','scheduler','trainer'}
   > bad=[]
   > for p in sorted(Path('configs').rglob('*.yaml')):
   >     if p.parts[1] in MODULES: continue   # 조각은 entry point가 아니다
   >     try: merge_train_config(p)
   >     except Exception as e: bad.append((p, e))
   > print('failing:', len(bad)); [print(' ', p, e) for p, e in bad]"
   > ```
   > ⚠️ **문자열 검색만으로 참조를 찾지 말 것 (§102 실측).** 여러 테스트가
   > `REPO_ROOT / "configs" / "<name>.yaml"`처럼 경로를 **조립**하므로 `configs/<name>` 문자열이
   > 파일에 아예 없다. `configs/` prefix로만 grep하면 테스트 5개·35 errors를 놓친다 —
   > **basename으로도 grep**할 것. `tests/fixtures/cvonly_golden.pt`처럼 **fixture 내부에 config
   > 경로가 저장**된 경우도 있다(그 fixture는 pre-prune 기록이라 재생성하면 의미가 사라지므로,
   > 테스트가 basename으로 폴백해 해석하도록 고쳤다).
   > ⚠️ **알려진 예외 — `failing: 10`이 정상 기준선이다 (§102).**
   > `configs/archive/v18_v19/` 10개는 `a5dfcf8`에서 삭제된 module fragment를 참조해 그 이후로
   > 로드 불가다(폐기된 v18/v19 아키텍처의 재현 기록). **인라인으로 고치려 시도했다가 되돌렸다** —
   > 이 파일들은 v19~v33 config 50개의 base이고, 인라인하면 group 참조가 해석된 dict로 바뀌어
   > **자식의 group override 병합 의미가 달라져 그 50개의 merged 결과가 전부 바뀐다**(스냅샷 대조로
   > 검출). 원문이 필요하면 `git show a5dfcf8^:configs/data/<name>.yaml`로 fragment를 복구할 것.
   > 새 작업이 이 10개를 늘리지 않는지만 확인하면 된다.
4. **모듈형 Component 설정 분리**:
   - `callbacks/`, `data/`, `logger/`, `model/`, `optimizer/`, `scheduler/`, `trainer/` 등 모듈 조각은 해당 서브폴더에 구성합니다.
