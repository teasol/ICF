# Current experiments (2026-08-11)

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

**금지 사항 (전부 실측으로 무너진 판정 방식)**

| 하지 말 것 | 근거 |
|---|---|
| **합성 val_ce로 arm 고르기** | v37은 val_ce가 더 좋았으나 50-fold는 −0.0068 (§65) |
| **합성 val_AUROC로 arm 고르기** | CV-only는 ep0 0.8885 = ep49 0.8882로 평평한데 er_status는 +0.037 (§69-6) |
| **er_status 단일로 판정** | §70이 K 증설을 "무의미"로 오판, 10개로는 +0.0127에 9/10 (§71). v45도 er_status만 보면 −0.002지만 10개로는 동률 |
| **단일 측정으로 단정** | 기저 요동 ±0.05, seed 반복 필수 (§69-3) |
| **ridge-only 진단치를 기대값으로** | K 64→128 예측 +0.016 vs 실측 +0.004 (§70-2) |
| **학습 길이가 다른 arm 비교** | control은 항상 같은 epoch 수로 새로 학습 (§42-43, §65) |
| **값만 보고 config 검증** | `lr: 2e-05`는 YAML이 **문자열**로 읽는다. 출력하면 숫자처럼 보인다 — 타입을 볼 것 (§79) |

---

## 2. 표준 실행

**학습** (50 epoch. CV-only 약 28분, Encoder+Ridge 약 55분)
```bash
CUDA_DEVICES=<gpu> NPROC_PER_NODE=1 \
TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun \
NETRC=/NHNHOME/BASE/kimds/.netrc \
bash scripts/launch_interactive_training.sh <RUN_NAME> <CONFIG>
```
⚠️ **50 epoch 유지** — 합성 지표는 평평해도 실제 task는 계속 오른다(§69-6).
⚠️ **`gradient_clip_val` 켜지 말 것** — er_status −0.0317 (§67).

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

## 3. 결과표 (SEAL 10개 macro 평균)

| arm | 계보 | 평균 | 비고 |
|---|---|---|---|
| SEAL ABMIL (지도학습) | — | **0.727** | 비교 상대 |
| SEAL MeanMIL (지도학습) | — | 0.713 | |
| **v41_K128** | A | **0.6940** | **현행 최고**. K=128, CV-2=128, `a=0.85π/K` |
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
4. **seed 반복** — 지금까지 arm당 1 seed다.
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
config는 `configs/train_v74_cv_dd_ct_mlp_1pop_linear_1536.yaml`, 평가 tag는
`v74_ct_e49`다. arm 판정은 공식 10-task macro와 task별 regression을 함께 본다.
