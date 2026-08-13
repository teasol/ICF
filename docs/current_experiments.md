# Current experiments (2026-08-13)

CV-only 이전(v22~v39, 합성 중심 판정)은 [`history.md`](history.md).
**그 문서의 판정 절차는 폐기됐다** — 합성 지표로 arm을 고르는 방식이 반복적으로 실패했다.

---

## 1. 판정 기준 (§71-4 확정)

> [!IMPORTANT]
> **판정은 SEAL 대상 10개 task의 macro-AUROC 평균으로 한다.**
> ```bash
> bash scripts/eval_seal_tasks.sh <gpu> <ckpt> <config> <tag> <task>...
> ```
> 대상은 `docs/seal_univ2_baseline_17tasks.csv`의 **`in_seal=yes` 10개**뿐이다 — SEAL과
> **같은 코호트·같은 공식 50-fold**로 비교 가능한 행. 나머지 7개는 대응 수치가 없다.

> [!IMPORTANT]
> **baseline·채점 규칙·게이트 (§109, 2026-08-13) — 판정 단위가 4 seed다**
>
> - **baseline = v83 linear head, 1-GPU 4 seed 평균 `0.6880`**
>   (config `configs/train_v83_linear_head_1536_1gpu.yaml`,
>   ckpts `checkpoints/20260813_153750/v83_linear_head_seed4{2..5}/`,
>   tags `v83_linear_head_seed4{2..5}_ep49`, 시드별 0.6905/0.6896/0.6774/0.6944, std 0.0074).
> - **⚠️ 이 승격은 §107-3 게이트를 충족하지 못한 상태에서의 사용자 결정이다(§108→§109).** v82
>   대비 seed-paired Δ +0.0045, t≈1.15, seed 44만 부호 반전(3/4 양수) — 4/4 부호 일치도
>   `|t|≥2.5`도 아니다. 인용할 때 "미판정 상태의 승격"임을 밝힐 것.
> - **⚠️ 0.6880은 옛 v77 DDP4 baseline(§104)의 0.6880과 숫자만 같은 별개 레짐의 값이다** —
>   착시로 "제자리로 돌아왔다"고 읽지 말 것.
> - **직전 baseline**: v82 Medium ClassSep `[0.5,1.4]`, 1-GPU 4 seed 0.6835(historical). 그 이전
>   DDP4 baseline은 0.6880(`v77_hard_ep49`). **두 레짐의 숫자를 빼지 말 것** — 아래 §3-2 결과표는
>   대부분 DDP4 1 seed 기록이다.
> - **모든 arm은 1-GPU · SEED 42/43/44/45 · 50 epoch · epoch 49 채점**으로 돌리고 이제는 **v83**
>   4 seed와 **seed-paired**로 비교한다. **판정 조건: 4/4 시드 부호 일치 + |t| ≥ 2.5**, 부호가
>   갈리면 미판정 — 이 게이트 자체는 §109로 바뀌지 않았다.
>   시드별 fold-paired CI(`scripts/compare_arms_paired.py`)는 보조 근거다.
> - **채점은 epoch 49 고정.** validation-best 선택은 val_ce가 평평한 arm에서 과소학습 지점을
>   고른다 — v80 seed 43이 epoch 16에 걸려 −0.0089를 잃었다.
> - **macro seed std**는 arm마다 다르다(§106-1): fixed P 0.0018 / Medium 0.0029 / **Hard 0.0053** /
>   shallow MLP 0.0051 / **linear head(v83) 0.0074** — 학습되는 P를 가진 arm일수록 노이즈가
>   크므로 SE는 arm마다 실측한다. 단일 시드 게이트 macro Δ ≈ 0.010(2σ)은 **이제 1 seed 비교에만**
>   적용한다.
> - **task별 CI로 판정하지 않는다** — 시드만 달라도 task 6개의 CI가 0을 제외한다(BAP1 −0.0402).
> - **4 arm × 4 seed 정면 비교 (1-GPU, epoch 49, §106)**: Medium 0.6835 > Hard 0.6781 > fixed-P
>   0.6734 > shallow-MLP 0.6722. **Medium−Hard +0.0053 (t=3.0, 4/4 seed)** → §107에서 baseline
>   승격, learnable−fixed P +0.0048 (t=1.5, seed 44 부호 반전 → **미판정**), shallow MLP −0.0059
>   (t=−2.7, 기각). **v83(linear head) vs Medium(v82)**: +0.0045 (t≈1.15, 3/4 seed → **미판정**,
>   §108) → §109에서 사용자 결정으로 baseline 승격.

**금지 사항 (전부 실측으로 무너진 판정 방식)**

| 하지 말 것 | 근거 |
|---|---|
| **합성 val_ce로 arm 고르기** | v37은 val_ce가 더 좋았으나 50-fold는 −0.0068 (§65) |
| **합성 val_AUROC로 arm 고르기** | CV-only는 ep0 0.8885 = ep49 0.8882로 평평한데 er_status는 +0.037 (§69-6) |
| **er_status 단일로 판정** | §70이 K 증설을 "무의미"로 오판, 10개로는 +0.0127에 9/10 (§71). v45도 er_status만 보면 −0.002지만 10개로는 동률 |
| **단일 측정으로 단정** | 기저 요동 ±0.05, seed 반복 필수 (§69-3) |
| **점추정 macro끼리 빼서 판정** | task 내 fold 산포가 ±0.09인데 판정 대상 Δ는 0.001~0.012다. 모든 arm이 같은 공식 fold를 쓰므로 **fold별로 먼저 뺀 뒤 평균**해 CI를 낼 것 — `scripts/compare_arms_paired.py` (§99) |
| **ridge-only 진단치를 기대값으로** | K 64→128 예측 +0.016 vs 실측 +0.004 (§70-2) |
| **task별 CI로 판정하기** | 시드만 다른 두 run에서 task 6개 CI가 0을 제외, BAP1 −0.0402. task별 seed std 평균 0.0161 = macro의 7배 (§104-5) |
| **macro Δ 0.010 미만을 단정하기** | macro seed std 0.0051. ridge calibration(−0.0033)·v78 무가중(−0.0047)은 "판정 불가"로 내려갔다 (§104-4) |
| **validation-best로 채점하기** | val_ce가 평평하면 과소학습 지점을 고른다. v80 seed 43이 epoch 16에 걸려 −0.0089 (§104-2). **epoch 49 고정으로 채점** |
| **학습 레이아웃이 다른 arm 비교** | 1-GPU는 1024 step/epoch·batch 1, DDP4는 rank당 256 step·gradient 4개 평균. v80의 −0.0158에 이 confound가 남아 있다 (§104-6) |
| **학습 길이가 다른 arm 비교** | control은 항상 같은 epoch 수로 새로 학습 (§42-43, §65) |
| **값만 보고 config 검증** | `lr: 2e-05`는 YAML이 **문자열**로 읽는다. 출력하면 숫자처럼 보인다 — 타입을 볼 것 (§79) |

---

## 2. 표준 실행

**학습 — 4 seed 배치가 판정 단위다 (§107).** GPU 0–3에 시드 하나씩 올려 한 배치 약 28분.
```bash
for i in 0 1 2 3; do
  SEED=$((42+i)) CUDA_DEVICES=$i NPROC_PER_NODE=1 \
  TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun \
  NETRC=/NHNHOME/BASE/kimds/.netrc \
  bash scripts/launch_interactive_training.sh <RUN_NAME>_seed$((42+i)) <CONFIG_1gpu> &
done; wait
```
⚠️ **1-GPU config를 쓸 것** — 판정 레짐이 1-GPU다(§107). DDP4 config로 돌린 숫자는 baseline
4 seed와 비교할 수 없다(레이아웃 차이 −0.0098, §106-4).
⚠️ **50 epoch 유지** — 합성 지표는 평평해도 실제 task는 계속 오른다(§69-6).
⚠️ **`gradient_clip_val` 켜지 말 것** — er_status −0.0317 (§67).

**baseline 4 seed** (비교 대상, §109):
`configs/train_v83_linear_head_1536_1gpu.yaml`,
tags `v83_linear_head_seed4{2..5}_ep49`, macro 0.6905/0.6896/0.6774/0.6944 (mean **0.6880**).
직전 baseline은 `configs/train_v82_medium_classsep_1536_1gpu.yaml`,
tags `v82_medium_seed4{2..5}_ep49`, mean **0.6835**(historical, §107).

**평가** (SEAL 10개, 2 GPU 분할 약 5분)
```bash
CK=<ckpt>; CFG=<config>; TAG=<tag>
bash scripts/eval_seal_tasks.sh 0 "$CK" "$CFG" "$TAG" \
  bc_therapy/{er_status,grade,her2_status} cptac_brca/{PIK3CA,TP53}_mutation &
bash scripts/eval_seal_tasks.sh 1 "$CK" "$CFG" "$TAG" \
  cptac_luad/{EGFR,STK11,TP53}_mutation cptac_ccrcc/{BAP1,VHL}_mutation &
wait
```
⚠️ **각 arm은 자기 훈련 config로 채점**한다.
⚠️ **prune(§73) 이전 체크포인트는 현재 트리로 로드 불가**다. 채점하려면
`8caa96c`에 고정한 worktree(`/NHNHOME/BASE/kimds/ICF_pre_prune`)를 쓴다.

**결과 집계**
```bash
grep -hoP 'fold-mean AUROC: \K[0-9.]+' logs/official50/*_<TAG>.log \
  | awk '{s+=$1;k++} END{printf "%.4f (%d)\n", s/k, k}'
```

---

## 3. 결과표

### 3-1. 현행 판정 레짐 — 1-GPU × 4 seed (§106·§107·§109)

**새 arm은 이 표하고만 비교한다 — 현재 기준 arm은 v83이다.**

| arm | baseline에서 바뀐 것 | 4 seed 평균 | seed std | 판정 |
|---|---|---:|---:|---|
| **v83 linear head** | (**활성 baseline**, §109) | **0.6880** | 0.0074 | head `12→32→1`(GELU) → `Linear(12,1)`, tags `v83_linear_head_seed4{2..5}_ep49` |
| v82 Medium | head `Linear(12,1)` → `12→32→1`(GELU) | 0.6835 | 0.0029 | v83 기준 −0.0045 (t≈−1.15, 3/4) **미판정** — 직전 baseline(§107) |
| v84 deep head | head `12→32→1`(GELU) → `12→32→32→1`(GELU×2) | 0.6777 | 0.0018 | v82 기준 −0.0057 (t=−3.63, 4/4) / v83 기준 −0.0102 (t=−3.61, 4/4) **기각, 양쪽** (§110) |
| v77 Hard | ClassSep `[0.5,1.4]` → `[0.2,0.8]` (v82 기준) | 0.6781 | 0.0053 | v82 기준 −0.0053 (t=−3.0, 4/4) |
| v81 fixed P | learnable P → fixed P (197,057 → 449, v77 기준) | 0.6734 | 0.0018 | Hard 기준 −0.0048 (t=−1.5) **미판정** |
| v80 shallow MLP | orthogonal manifold → shallow MLP (v77 기준) | 0.6722 | 0.0051 | Hard 기준 −0.0059 (t=−2.7) **기각** |

⚠️ **v83 승격은 §107-3 게이트(4/4 부호 일치 + |t|≥2.5) 미달 상태의 사용자 결정이다(§108·§109)** —
"확정된 승격"으로 인용하지 말 것.
⚠️ **relation head 깊이 축은 §110으로 소진됐다** — v84(심화)는 양쪽 baseline 기준 모두 4/4 +
|t|>3.6으로 기각. §108(얕게, 미판정)과 종합하면 이 축에서 새 arm을 더 설계하지 말 것.
⚠️ **fixed P × Medium 칸은 여전히 비어 있다** — v81은 Hard에서만 돌았다(§107-6). v83이 baseline이
된 지금은 "fixed P × Medium × linear-head" 조합도 미측정으로 남는다.
⚠️ **정정 (2026-08-13, nhn-SMC-claude)**: "v85로 nhn-SMC에서 측정 진행 중"이라는 이전 기록은
**착오였다** — nhn-SMC 노드에서 실제로 GPU/프로세스를 확인한 결과 v85 학습이 돌고 있지 않다.
`train_v85_medium_fixed_p_1536_1gpu.yaml` config는 존재하지만 아직 아무 노드에서도 실행되지
않았다. §107-6은 여전히 미착수 상태로 남는다.

### 3-2. 역사적 표 — DDP4 1 seed (SEAL 10개 macro 평균)

> [!WARNING]
> **아래는 전부 DDP4 단일 시드 기록이라 §3-1과 직접 뺄 수 없다** (레이아웃 차이 −0.0098,
> §106-4). 자기들끼리는 여전히 유효하다.

| arm | 계보 | 평균 | 비고 |
|---|---|---|---|
| SEAL ABMIL (지도학습) | — | **0.727** | 비교 상대 |
| SEAL MeanMIL (지도학습) | — | 0.713 | |
| **v41_K128** | A | **0.6940** | **역사적 전체 최고**. K=128, CV-2=128, `a=0.85π/K`. ⚠️ 레짐이 달라 v83 0.6880와 직접 비교 불가 (§107-2) |
| v77 Hard orthogonal ep49 | relation | 0.6880 | **historical** (`v77_hard_ep49`, §104). §107에서 v82로, §109에서 v83으로 교체됨. ⚠️ v83의 1-GPU 4 seed 0.6880과 숫자만 같은 별개 레짐의 값 |
| v77 Hard orthogonal val-best | relation | 0.6873 | 같은 run의 `epoch=048`. 역사적 표기 |
| ClassSep Medium ep49 | relation | 0.6881 | `[0.5,1.4]`. DDP4 1 seed로는 Hard와 동률(+0.0001)이었으나 **4 seed로는 +0.0053** (§106-2). ⚠️ §91의 0.6823은 오기 (§105-3) |
| ClassSep Mild ep49 | relation | 0.6854 | `[0.8,1.7]`. Hard 대비 −0.0026, 판정 불가 |
| ClassSep Very-hard ep49 | relation | 0.6843 | `[0.1,0.5]`. Hard 대비 −0.0037, 게이트 미만 |
| v78 balanced ep49 | relation | 0.6879 | −0.0001. 동률 |
| v78 무가중 ep49 | relation | 0.6841 | −0.0039. 게이트 미만 — 단정 불가 |
| ridge calibration ep49 | relation | 0.6870 | −0.0010 [−0.0021,+0.0002]. **기각 철회 → 판정 불가** |
| v79 dual projection ep49 | relation | 0.6768 | −0.0112. **기각 유지** |
| v76 learnable P ep49 | relation | 0.6735 | v74(0.6731) 대비 **+0.0004 [−0.0030,+0.0038] → 승격 근거 없음** (§105-4) |
| v80 shallow MLP | relation | (§3-1 참조) | ⚠️ 4 seed는 1-GPU 레짐이라 이 표가 아니다. §104의 Δ −0.0158은 레이아웃 confound로 부풀려진 값이고 정당한 control 대비 −0.0059다 (§106-4) |
| mlpbank ep49 (M=128~4096) | relation | 0.6734/0.6730/0.6780/0.6776/0.6678 | 1024·2048 고원 + 4096 −0.0102. M→∞ 하강은 실재 (§105-5) |
| latent ep49 (2/4/8/16/32) | relation | 0.6775/0.6776/0.6759/0.6665/0.6880 | L16 딥은 학습량 artifact가 아니다 (§105-5) |
| ⚠️ v43_notanh / v44_lowT | A | 0.6770 / 0.6763 | **artifact 없음** — `logs/official50`·`predictions/`에 산출물이 없어 재확인 불가 (§105-6) |
| v77 large-ragged warm-start | relation | 0.6885 | 파생 실험; +0.0012라 미승격 |
| v77 ridge calibration | relation | 0.6840 | λ/logit scale 학습, 기각 |
| v76 learnable P (easy predecessor) | relation | 0.6748 | 이전 baseline |
| v74 fixed P + DD + CT | relation | 0.6731 | v76 control |
| v45_paired | A | 0.6937 | paired_head + rank 4. 동률(−0.0003) |
| v42_rank2 | A | 0.6944 | 동률 |
| v42_rank4 | A | 0.6932 | 동률 |
| v41_K64 | A | 0.6814 | K=64 |
| v43_notanh | A | 0.6770 | identity margin, T=150→34.0 |
| v44_lowT | A | 0.6763 | identity margin, T=4→2.84 |
| v52_lr2e5 | B(구판) | 0.6619 | inducing-point, 기술자 256차원 |
| v51_lr1e4 | B(구판) | 0.6047 | 〃 |
| v53_enc | B(신판) | 0.6526 | 세포 간 attention + 16,384차원, lr 2e-5 |
| v54_enc | B(신판) | 0.6219 | 〃 lr 1e-4 |
| v57_resp2pop | B(데이터) | 0.6127 | scalar + 2 populations; v60과 동률 |
| v58_xor2 | B(데이터) | 0.5530 | 2-factor XOR, 기각 |
| v59_xor8pop4 | B(데이터) | 0.5616 | 8-factor/4-pop XOR, 기각 |
| v60_scalar1ctrl | B(데이터) | 0.6090 | per-bag cardinality/padding control |
| v61_linear | B(데이터) | 0.6157 | orthogonal manifold; v60 +0.0067, v41 −0.0783, 승격 기각 |

**task별** (주요 arm)

| task | v41 (A) | v45 (A) | v52 (B구판) | v53 (B신판) | v54 (B신판) |
|---|---|---|---|---|---|
| er_status | 0.7303 | 0.7281 | 0.6487 | 0.6515 | 0.6377 |
| grade | 0.7451 | 0.7389 | 0.7398 | 0.7208 | 0.6579 |
| her2_status | 0.6792 | 0.6652 | 0.6190 | 0.6366 | 0.5586 |
| brca PIK3CA | 0.5476 | 0.5603 | 0.5657 | 0.5067 | 0.4845 |
| brca TP53 | 0.8188 | 0.8104 | 0.7656 | 0.7723 | 0.6374 |
| ccrcc BAP1 | 0.6312 | 0.6589 | 0.6317 | 0.6280 | 0.6146 |
| **ccrcc VHL** | 0.4503 | 0.4362 | **0.5104** | 0.4708 | **0.5006** |
| luad EGFR | 0.7642 | 0.7603 | 0.7408 | 0.7072 | 0.6975 |
| luad STK11 | 0.8891 | 0.8843 | 0.8039 | 0.8242 | 0.7942 |
| luad TP53 | 0.6846 | 0.6948 | 0.5939 | 0.6081 | 0.6362 |
| **평균** | **0.6940** | 0.6937 | 0.6619 | 0.6526 | 0.6219 |

⚠️ **ccrcc VHL 단서**: CV-only는 0.44~0.45로 **랜덤 이하**인데 계보 B만 0.5104다. 10개 중
유일하게 계보 B가 이긴 task이고, 하필 CV-only가 가장 크게 실패하는 곳이다. 두 계보가 다른
것을 보고 있다는 유일한 증거. 표본 1개라 단정 불가.

---

## 4. 확정된 것 (재실험 불필요)

| 결론 | 근거 |
|---|---|
| **6개 분기 중 CV-1·CV-2만 남겨도 동률** | fold-paired −0.0005 (§68) → §73에서 실제 삭제 |
| **Q-5 population attention은 상수를 뱉는다** | AUROC 0.5000, std 0.0000 (§68-1) |
| **CV-1 제거 불가** | 안정화 무관하게 학습 붕괴, 2회 재현 (§66·§67) |
| **G-2 global ridge 무기여** | Δ −0.0004, CI가 0 포함 (§66) |
| **label-free 사영은 전부 천장** | 8개 축 모두 0.68±0.03 (§69-3) |
| **차원(K)은 유효, 단 대역폭 고정 시** | 9/10 task, +0.0127 (§71-5) |
| **CV-2 손잡이는 소진됐다** | margin activation(−0.017), rank(±0.001), head 구조(−0.0003) 셋 다 10개 평균을 못 움직임 (§76·§77) |
| **CV-2 출력 스케일은 성능과 무관** | v43(T→34.0)과 v44(T→2.84)가 10개 task 전부 셋째 자리까지 일치 (§76) |
| **학습은 dense 경로로** | ragged 74.2 ms vs dense 31.3 ms (§74) |
| **CV-1의 dual은 옳다** | dual 0.33 ms vs primal 9.8 ms (§78) |
| **셀-셀 attention은 감당된다** | 최악 구성 3.13 GiB (§79) |
| **계보 B의 구조 확장은 SEAL을 못 올린다** | 합성 0.78→0.849인데 SEAL 0.6619→0.6526 (§79-6) |
| **합성 val_AUROC는 SEAL과 역상관할 수 있다** | v54가 합성 최고(0.8623)인데 SEAL 최저(0.6219) (§79-6) |

---

## 5. 미해결 / 다음

1. **계보 B는 현재 형태로 기각** — 재설계(세포 간 attention + 16,384차원)가 합성 지표를
   0.78 → 0.849로 크게 끌어올렸는데 **SEAL은 오히려 내려갔다**(0.6619 → 0.6526).
   구조 확장이 아니라 **일반화가 문제**다. 더 키우기 전에 왜 합성만 좋아지는지부터.
2. **ccrcc VHL 단서** — 계보 B만 랜덤을 넘는다. 두 계보가 상보적인지 확인 가치.
3. **증류/잔차** — CV-1을 teacher로 계보 B를 학습. ⚠️ **순수 증류로는 teacher를 넘을 수
   없다**(CV-1은 결정적 특징 맵이라 완벽 모방 = 0.6940). 출발점 이동이나 잔차 학습으로만
   의미가 있다.
4. ~~**seed 반복** — 지금까지 arm당 1 seed다.~~ **해소됨 (§107)**: 판정 단위가 1-GPU × 4 seed로
   바뀌었다. 남은 부채는 **과거 arm 전부가 DDP4 1 seed 기록**이라는 것 — 새 arm과 비교하려면
   그 arm을 1-GPU 4 seed로 다시 돌려야 한다.
4a. **v83 linear-head 승격의 통계 근거 보강 (§109)** — Δ+0.0045, t≈1.15는 §107-3 게이트 미달인
   채로 baseline이 됐다. `scripts/compare_arms_paired.py`로 fold-paired CI를 확인하거나 seed를
   추가(46–49)해 재검증할 것.
4b. **fixed P × Medium 4 seed** — 비어 있는 칸이자 baseline 파라미터를 196,621 → 449로 줄일 수
   있는지 가르는 측정이다(§107-6). learnable P는 같은 난이도 4 seed에서도 t=1.5로 미판정이다.
   ⚠️ config(`train_v85_medium_fixed_p_1536_1gpu.yaml`)는 있지만 **아직 어느 노드에서도 실행되지
   않았다** — "다른 노드에서 진행 중"은 착오였다(2026-08-13 정정, nhn-SMC-claude). 미착수.
4c. ~~**relation head 깊이 축**~~ **해소됨 (§110)**: v84(심화)가 양쪽 baseline 기준 모두 4/4 +
   |t|>3.6으로 기각됐다. §108(얕게, 미판정)과 종합해 이 축은 소진 — 더 파지 말 것.
5. **병목이 표현이 아닐 가능성** — CV-1 단독 0.9052 vs 전체 0.9199. 그 위에 얹은 모든
   시도(v36·v37·v42·v43·v44·v45)가 Δ≈0이었다. task 자체의 정보 한계일 수 있다.
6. **새 합성 cardinality arm** — 기본 데이터가 bag별 독립 cell 수 + zero-padding/mask로
   바뀌었다(§81). 이전 arm과 데이터 분포가 다르므로 새 학습은 반드시 새 arm으로 취급한다.
   4,096 random cap 적용 후 padding 유효률은 34.10%, 원 cell 유지율은 58.65%다. 긴 bag은
   step마다 다른 부분집합을 보므로 cardinality 교정과 cell-level augmentation을 함께 얻는다.

---

## 6. 성능 (2026-08-10 기준)

| | CV-only | Encoder+Ridge |
|---|---|---|
| step | 31.3 ms | 45.7 ms (중앙값) |
| epoch | 33 s | 56~69 s |
| 50 epoch | ~28분 | ~55분 |

step 구성(측정): 에피소드 생성(GPU) 22% / 모델 28% / Lightning 오버헤드 50%.
⚠️ **CPU 비동기 생성은 역효과** — 생성은 에피소드당 35 GFLOP 연산이라 GPU 3.2 ms vs
CPU 2,579 ms(805배)이고, 옮기면 매 step H2D 전송이 새로 붙는다(§74).
⚠️ **프리페치 깊이를 올려도 소용없다** — depth 1이 3.9 s, depth 3이 4.8 s. 생성이 모델보다
길어 생산자가 포화 상태다(§74).

## 7. v67 무학습 CV mean ablation (2026-08-11)

동일한 fixed P(K128), ridge lambda=1, logit scale=2에서 covariance-only와
covariance+raw-bag-mean을 비교했다. 학습 없이 fold마다 closed-form ridge만 풀었다.

| 평가 | covariance-only | canonical CV(+mean) | delta |
|---|---:|---:|---:|
| PathoBench SEAL 10-task macro | 0.6630 | **0.6667** | **+0.0037** |
| ICI 5-seed mean | 0.5381±0.0177 | **0.5449±0.0180** | **+0.0068** |
| ICI seed-averaged donor | 0.5357 | **0.5476** | **+0.0119** |
| ICI mean log loss | 0.8998 | **0.8897** | −0.0101 |

결정: 앞으로 CV branch는 covariance upper triangle과 중심화 전 raw bag mean을 항상 concat한다.
다만 ICI의 95% CI는 각각 [0.414,0.657], [0.427,0.669]로 모두 랜덤을 포함한다.
CovarianceOnlyRidgeModel은 이 결정을 재검증할 historical control로만 남긴다.

## 8. DD와 v70 learned relation head (2026-08-11)

### 8-1. training-free DD ablation

canonical CV와 같은 fixed P(K128) covariance에서 rank-1 DD를 만들었다. 학습 없이
support label로 generalized covariance direction을 구하고 query마다 standardized distances
`D0,D1`을 계산했다.

| arm | 결합 | SEAL 10-task macro | vs CV |
|---|---|---:|---:|
| canonical CV | `softmax(CV logits)` | 0.6667 | — |
| DD-only | opposite-distance probability | 0.5862 | −0.0805 |
| CV+DD equal | `0.5 CV + 0.5 DD` | 0.6441 | −0.0226 |
| CV+0.1DD | `(CV + 0.1 DD)/1.1` | 0.6688 | +0.0021 |

DD는 주 classifier로 약하지만 작은 residual evidence로는 7/10 task를 올렸다. BAP1에서
고정 결합이 크게 실패해 task-independent weight의 한계가 드러났다.

### 8-2. v70 CV+DD+MLP

1-population, scalar response, single label rule, episode-wise orthogonal linear manifold 합성
task로 frozen CV/DD 출력 위 8→32→1 MLP(321 trainable parameters)만 50 epoch 학습했다.
GPU 0–3 DDP4, bf16, best epoch 48 `val_ce_loss=0.120259`, 오류/non-finite 없음.

| task | canonical CV | CV+0.1DD | **v70 MLP** | v70−CV |
|---|---:|---:|---:|---:|
| er_status | 0.6821 | 0.6902 | **0.7002** | +0.0181 |
| grade | 0.6717 | 0.6851 | **0.7081** | +0.0364 |
| her2_status | 0.6388 | 0.6502 | **0.6657** | +0.0269 |
| brca PIK3CA | 0.5124 | 0.5103 | **0.5131** | +0.0007 |
| brca TP53 | 0.7957 | 0.8022 | **0.8146** | +0.0189 |
| luad EGFR | 0.7298 | 0.7387 | **0.7502** | +0.0204 |
| luad STK11 | 0.8405 | 0.8496 | **0.8692** | +0.0287 |
| luad TP53 | **0.6891** | 0.6823 | 0.6659 | −0.0232 |
| ccrcc BAP1 | **0.6978** | 0.6614 | 0.6054 | −0.0924 |
| ccrcc VHL | 0.4087 | 0.4181 | **0.4226** | +0.0139 |
| **macro** | 0.6667 | 0.6688 | **0.6715** | **+0.0048** |

v70은 CV 대비 8/10 task를 올렸고 fixed 0.1 결합보다 +0.0027 높았다. 전체 프로젝트 최고
v41 0.6940에는 아직 −0.0225이며, BAP1 실패가 가장 큰 제한이다.

### 8-3. synthetic task 일반화에 대한 갱신된 해석

같은 1-pop linear synthetic task를 Set Transformer의 고차원 representation 학습에 썼을 때는
합성 validation은 좋아져도 SEAL 개선이 작거나 없었다(v61/v62 계보). 따라서 이전에는
synthetic task 자체가 실데이터로 일반화되지 않는다고 해석하기 쉬웠다.

그러나 v70에서는 representation을 새로 만들지 않고 이미 강한 canonical CV와 DD 통계를
고정한 채, **끝단의 저차원 관계 함수만** 같은 synthetic task로 학습했고 SEAL 8/10 task,
macro +0.0048을 얻었다. 이는 synthetic task에 실데이터로 이전되는 일반화 신호가 없었던 것이
아니라, 그 신호가 고용량 Set Transformer 표현 학습보다 **구조화된 통계 사이의 decision rule /
calibration 학습에 더 잘 전달된다**는 증거다.

다만 개선폭이 작고 BAP1/LUAD TP53가 하락했으므로 “합성 task가 일반적으로 해결됐다”는
주장은 하지 않는다. 현 단계 결론은 **일반화 효과는 존재하며, 효과의 크기는 학습을 삽입하는
위치와 inductive bias에 강하게 의존한다**이다. v71 CV+MLP가 DD가 실제로 추가 일반화 신호를
제공했는지 분리한다.

## 9. v71–v74 relation-head ablation과 활성 baseline

| arm | relation feature | synthetic manifold | SEAL macro | 판정 |
|---|---|---|---:|---|
| v70 | CV 4 + DD 4 | orthogonal linear | 0.6715 | control |
| v71 | CV 4 | orthogonal linear | 0.6667 | DD 제거로 −0.0048 |
| v72 | CV 4 + DD 4 | 1-hidden-layer MLP | 0.6709 | v70과 동률 |
| v73 | CV 4 + DD 4 + Magnitude 4 | orthogonal linear | 0.6473 | 기각 |
| **v74** | **CV 4 + DD 4 + CT 4** | **orthogonal linear** | **0.6731** | **활성 baseline** |

v74 task별 fold-mean AUROC는 ER 0.7045, grade 0.7070, HER2 0.6675, BRCA
PIK3CA 0.5142, BRCA TP53 0.8155, LUAD EGFR 0.7517, LUAD STK11 0.8663,
LUAD TP53 0.6632, CCRCC BAP1 0.6201, CCRCC VHL 0.4210이다. v70보다 6/10 상승,
macro +0.0016이며 BAP1 +0.0147이 가장 크다.

앞으로 relation-head arm의 control은 v74와 같은 scalar/1-pop/single-label/orthogonal-linear
50-epoch 조건을 사용한다. best checkpoint는
`checkpoints/20260811_172825/v74_cv_dd_ct_mlp_1pop_linear/epoch=049-val_ce_loss=0.1197.ckpt`,
config는 `configs/archive/v69_v76_relation/train_v74_cv_dd_ct_mlp_1pop_linear_1536.yaml`, 평가 tag는
`v74_ct_e49`다. arm 판정은 공식 10-task macro와 task별 regression을 함께 본다.

## 10. Hard fixed nonlinear manifold-bank protocol (2026-08-12)

`mlp_bank`는 latent 32를 `32→128→128→1536`의 3-layer MLP(GELU hidden activation)로
mapping한다. bank 크기 M만 바꾸며 bank member는 `manifold_seed`와 member ID로 고정된다.
episode마다 ID를 uniform sampling하고 episode 내부 모든 bag/population/class에 같은 map을 쓴다.
M=128/512/1024/2048/4096을 각각 v76, Hard ClassSep, DDP4, bf16, 50 epochs로 학습하며,
validation-best checkpoint 하나를 SEAL 10-task로 평가한다. MLP weight를 bank 전체로 저장하지
않고 선택된 member만 deterministic regeneration하여 bank 크기에 따른 host/GPU memory 차이를
제거한다.

후속 50:50 혼합 arm은 각 episode에서 fresh orthogonal linear 또는 MLP-1024 member 하나를
동일 확률로 선택한다. 두 mapping을 한 episode/cell feature 안에서 합산하지 않으므로 branch별
distribution과 label semantics가 섞이지 않는다. 나머지 Hard/v76 학습·평가 조건은 동일하다.

Hard ridge-calibration arm은 mixed manifold를 사용하지 않는다. 기존 Hard fresh orthogonal
control과 완전히 같은 데이터에서 `ridge_log_lambda`와 `ridge_log_scale`만 동결 해제한다.
따라서 결과 차이는 projection/head/data 변화가 아니라 두 scalar meta-parameter의 효과다.
