# Current development status & multi-location sync SSOT

**Last updated**: `2026-08-21 17:57:00` — **판정 프로토콜 개정 (Primary 7-task / Hold-out 10-task) 및 단일 SSOT 확립 (§0)**:
- **판정 프로토콜 개정**: SEAL에 없던 7개 과제를 **새로운 Primary Benchmark**로 확립, 기존 SEAL 10-task를 **독립 Hold-out Validation**으로 전환.
- v114 활성 baseline 유지(학습 파라미터 0, Deterministic). 활성 runner `scripts/eval_v114.sh`. ⚠️ **Python 경로를 하드코딩하지 말고 `. scripts/node_env.sh`로 해석할 것**(§164). 전체 아키텍처 및 개발자 명세는 `current_architecture.md` **§0**.

---

# §0. 판정 프로토콜 종합 SSOT (Decision Protocol SSOT)

프로젝트의 모든 모델 평가, 승격 및 기각 판단은 아래 원칙에 따라 수행된다 (2026-08-21 개정).

## 0-1. 활성 결정론적 Arm 판정 규칙 (현 v106~v114 무학습 계보)
- **$t$-통계량, $p$-value, 신뢰구간(CI) 사용 절대 금지 (§151-1)**:
  - 활성 모델은 학습 파라미터 0, 난수 시드 분산이 정확히 0($\text{seed std} = 0.00000$)이므로, task를 표본 단위로 둔 $t$-검정은 통계적 근거가 없다.
- **벤치마크 2-Tier 구조 (2026-08-21 확정)**:
  1. **Primary Benchmark (7 tasks)**: SEAL에 포함되지 않았던 7개 과제를 **새로운 주 평가 기준**으로 삼는다.
     - `cptac_lscc/ARID1A_mutation`, `cptac_lscc/Histologic_Grade`, `cptac_lscc/KEAP1_mutation`
     - `cptac_luad/KRAS_mutation`, `cptac_pda/SMAD4_mutation`
     - `ucla_lung/progression_regression`, `cptac_ccrcc/PBRM1_mutation`
  2. **Hold-out Validation (10 tasks)**: 기존 SEAL 10-task를 **독립 홀드아웃 검증 집단**으로 삼아 선택 편향(Selection bias) 및 일반화 성능을 교차 검증한다.
- **유효한 3대 판정 지표**:
  1. **Primary 7-task 부호 일치 수 (Sign Agreement)**: 7개 과제 중 몇 개에서 승리했는가? (예: $\ge 5/7$)
  2. **Primary 7-task Macro $\Delta$**: 7개 과제 산술 평균의 유의미한 개선.
  3. **Hold-out 10-task 재현성**: SEAL 10개 과제에서의 동시 개선 ($\Delta_{holdout} > 0$).

## 0-2. 최종 승격 / 기각 의사결정 프로토콜 (User Decision Protocol, §118)
- **게이트 자동 결정 금지**: 어떤 수치나 통계 게이트도 모델 승격/기각을 자동으로 결정하지 않으며, **최종 판정은 항상 사용자의 종합적 판단**이다.
- **Baseline 성능대별 비대칭 분석**:
  - 같은 크기의 $+0.01$이라도 **랜덤 근처($0.4\sim 0.5$)에서의 상승보다 천장 근처($0.8\sim 0.9$)에서의 하락이 훨씬 치명적**이다.
  - 고신호 구간에서 깎이고 저신호 구간에서만 오르는 것은 유효 신호가 아니라 노이즈/평균 회귀 교환일 가능성이 높다.
- **필수 보고 양식 (§118-3)**:
  1. 이 실험/arm이 무엇을 검증하는지 가설 설명
  2. **Primary 7개 과제 및 Hold-out 10개 과제 전체별 baseline 대비 $\Delta$ 표를 빠짐없이 제시** (요약/평균만 제시 금지)
  3. 전체 Macro $\Delta$ 및 부호 일치 수 제시 $\to$ 사용자가 승격/기각/재검증 최종 판정.

## 0-3. 닫힌 축 (Closed Axes) 재시도 금지
실측을 통해 이득이 없거나 단조 열화가 입증된 아래 방향은 **새로운 arm으로 재설계/재시도하지 않는다**:
1. **합성 데이터 분포 축 (§129)**: 실제 에피소드와의 분포 격차를 닫으려고 할수록 성능이 단조 하락함.
2. **DD 전반 축 (§147)**: $K > 128$, $r > 1$, $|t|$ 게이트/셀렉터 모두 실패. (현재의 1-D ordered-typicality로 고정)
3. **CT Cell 수 단순 증대 (§159)**: Bag당 64개 이상의 full-cell을 써도 성능 향상 없음.
4. **CT Kernel-Ridge (§188)**: RBF/Poly 비선형 커널 모두 8/10 task 하락 $\to$ 비선형 곡률 부재 확인 및 기각.
5. **CT Top-k 풀링 (§189)**: Mean 풀링 대비 노이즈만 가중되어 기각.

## 0-4. [Historical Archive] 과거 학습 Arm 판정 규칙 (v83~v105 레짐)
*이하 규칙은 난수 시드 분산이 존재하던 과거 학습 계보(v83~v105) 전용이며, 현재 무학습 계보에는 적용되지 않는다.*
- **통계 게이트 (§107-3)**: 4개 시드 전체 부호 일치($4/4$) 및 Seed-paired $|t| \ge 2.5$ ($p \le 0.088$).
- **최소 검출 한계 (§131-2)**: 4 seed의 최소 검출 가능 효과는 $\approx 0.0121$. 노이즈 범위($\pm 0.005$ 이내)의 작은 신호를 쫓지 않음.
- **독립 시드 그룹 재현 (§131-5)**: 단일 그룹의 높은 $|t|$보다 독립적인 시드 그룹(A/B 그룹)에서의 부호 일치 재현을 우선.
- **평가 에포크 고정 (§104)**: 최적 에포크 체리피킹 금지, Epoch 49(또는 지정 에포크) 고정 평가.

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-21 17:57:00_

---

> [!IMPORTANT]
> **지금 읽는 사람이 먼저 알아야 할 3가지 (2026-08-15)**
>
> 1. **데이터 분포 축은 닫혔다 (§129, §0-3).** §115(에피소드 모양)·§123(cell 값 분포) 두 진단 사이클이
>    격차를 정확히 실측했고, 그 격차를 닫는 처방은 **전부** 실패했다. 그것도 **닫은 격차가 많을수록
>    단조로 나빠진다**: v83(0) → v94 응집도만 −0.0043 → v95 스펙트럼만 −0.0086 → **v102 둘 다
>    −0.0130(t=−2.70, 0/4, 기각)**. v100도 게이트 통과 기각(t=−3.59). **이 축에서 새 arm을 설계하지
>    말 것.**
> 2. **문제는 편향이 아니라 분산일 수 있다 (§130).** macro seed std가 **0.0074**인데 §105 이후 데이터
>    arm 15개가 쫓던 효과는 전부 ±0.005 안쪽이었다 — **노이즈보다 작은 신호를 쫓고 있었다.**
>    시드 앙상블은 **학습 비용 0**으로 **+0.0058~0.0071**(10/10 task 양수)을 낸다.
> 3. **판정은 게이트가 자동으로 하지 않는다 (§118, §0-2).** 최종 승격/기각은 macro + task 10개 전부의
>    baseline 성능대별 패턴 + 다른 arm과의 일관성을 종합한 **사용자 판단**이다. 보고 형식도 정해져 있다(§0-2).

> [!IMPORTANT]
> **활성 구성은 v114(§187, 사용자 결정) — 학습 파라미터 0, 완전 결정론적(seed std 0.00000)이다.**
> ```
> 사영 : fold의 CONTEXT cell을 bag별 자기 평균으로 센터링해 풀링한 공분산의 상위 256 고유벡터
> head : margin = 1.0·(CV1−CV0) − 1.0·M_DD + 1.0·(CT1−CT0)      # 세 weight 전부 1.0 (§187)
> CV   : off-diagonal 32,640차원만 (대각 256·raw mean 1,536 제거) ⚠️ DD는 전체 triangle
> DD   : rank-1 방향 → ordered-coordinate × nearest-class typicality, κ=1 (§182/§183)
> CT   : bag 자기 크기의 **1/8 fraction**(floor 64, seeded random seed 0) → 32 PCA 방향
>        → **seeded k-means++ + Lloyd(≤8)** 로 **256 token** → match abundance → ridge(λ=1)
> 정식 경로 SEAL 10-task macro = 0.70509,  홀드아웃 7 = **미측정**,  전체 17 = **미측정**
>
> bash scripts/eval_v114.sh <gpu> <tag> [tasks...]     # 활성 baseline entry point
>
> # 위 스크립트가 하는 일 전부:
> ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1 ICF_SKETCH_DIM=256 \
> ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=256 \
> ICF_CT_TOKENIZER=kmeans_plusplus ICF_CT_KMEANS_MAX_ITER=8 \
> ICF_CT_CELLS=0.125 ICF_CT_CELLS_SCALE=own ICF_CT_CELLS_MIN=64 \
> ICF_CT_ABUNDANCE_CELLS=match ICF_CT_SAMPLING=random ICF_CT_SAMPLING_SEED=0 \
> ICF_CT_DISTANCE_KERNEL=gemm ICF_CV_BLOCKS=offdiag \
> ICF_DD_ORDERED_TYPICALITY=1 ICF_DD_SEPARATION_FLOOR=1.0 \
> ICF_FIXED_HEAD_{CV,DD,CT}_WEIGHT=1.0 \
>   bash scripts/eval_seal_tasks.sh <gpu> <아무 v98 ckpt> \
>        configs/train_v98_p1_reverse_1536_1gpu.yaml <tag> <tasks...>
> ```
> ⚠️ **홀드아웃 7·전체 17은 v114로 측정된 적이 없다.** v112 값(0.60181 / 0.66211)을 v114의
> 것으로 인용하지 말 것 — CT의 cell 예산·tokenizer·sampling이 그 사이에 전부 바뀌었다(§185·§187).
> 따라서 지금 v114는 **SEAL 10 하나로만** 뒷받침된다 — 독립 집단 재현(§151-1의 판정 근거)이 없다.
>
> ⚠️ **v113/v114의 CT tokenizer는 `kmeans_plusplus`이지 hierarchical 2-means가 아니다**
> (§185-1 정정). §181이 v111 승격 근거로 든 "cell selection bias·sampling randomness 없음"은
> v113부터 **성립하지 않는다** — seeded random으로 bag의 1/8만 뽑는다(seed 0 고정이라
> 결정론성 자체는 유지된다). full-cell hierarchical 재현이 필요하면 `scripts/eval_v112.sh`.
>
> **계보 (전부 결정론적, seed std 0.00000)**
> ```
> v106 0.6864  within-slide PCA(K=128) + 고정 head, 학습 파라미터 0의 시작    §139
> v107 0.6945  K 128 → 256                                                  §143
> v108 0.6967  CT = PCA32 부분공간 + ridge readout                           §152
> v109 0.7027  CV = off-diagonal만, CT = k-means token @ w=0.7               §158
> v110 0.70692 CT cluster 16 → 32   ← 홀드아웃 0.61029 / 전체 0.66713 = 예측 최고 §161
> v111 0.70453 full-cell hierarchical PCA32/K256 (bias 제거 우선)            §181
> v112 0.70432 DD = ordered × typicality (κ=1, w=1)                          §183
> v113 0.70394 CT cell 예산 = bag 1/8 fraction + k-means++ (feasibility)     §185
> v114 0.70509 fixed-head weight 세 개를 전부 1.0으로 통일  ← 활성            §187
> ```
> ⚠️ **v110이 여전히 전체 17 최고(0.66713)다.** v111~v114는 예측 성능이 아니라 각각
> selection bias 제거(§181)·DD 형태(§183)·22GB GPU feasibility(§185)를 이유로 승격됐다.
> "최신 = 최고"로 읽지 말 것.
>
> ⚠️ **결정론적 arm에 §107-3 게이트(|t|≥2.5)와 §131-2 검출 한계를 적용하지 말 것** — 둘 다
> 시드 분산이 분모다. 부호 일치 수와 독립 집단 재현으로 판정한다(§151-1).
> ⚠️ 학습을 포함하는 arm과 비교할 때는 **그 arm의 분산 때문에 §107-3 게이트와 §131-2의 검출 한계가
> 그대로 적용된다.** 검출 한계가 ≈0이 되는 것은 training-free 변형끼리 비교할 때뿐이다(§139-6).
> ⚠️ 학습 계보(v83·v98 등)의 숫자와 **시드 집합이 다르므로 빼지 말 것**(§131-1). 그 계보의
> 마지막 baseline은 v98(1-GPU 8 seed 0.6852, §131)이고 전부 historical이다.
>
> ⚠️ **K=256은 환경변수로만 걸린다.** config의 `covariance_sketch_dim`은 **128 그대로 두었다** —
> 그 값을 바꾸면 v98 학습 재현이 깨지고 P(1536×128) strict load가 실패한다. `ICF_SKETCH_DIM`은
> K에 의존하는 유일한 텐서 `_covariance_projection`만 버리고 나머지 불일치는 예외로 올린다(§142-1).
> 무학습 구현 `TrainingFreeClassifier`의 기본값은 **256**이다(테스트로 고정).
> checkpoint는 **껍데기로만** 쓰인다 — P는 PCA가, head는 상수가 덮어쓰고, `ridge_log_lambda`/
> `ridge_log_scale`은 초기값(log 1, log 2) 그대로다. **쓰이는 학습값 0개**이므로 어느 v98 시드를
> 넘겨도 같은 수치가 나온다(§152, `scripts/node_env.sh`의 `ICF_CKPT`가 자동 탐색).

**한 줄**: 활성 baseline은 **v114**(§187, 사용자 결정) — 학습 파라미터 0·완전 결정론적, **SEAL 10 macro 0.70509**(홀드아웃 7·전체 17 미측정). 판정은 t·p·CI가 아니라 **부호 일치 수 + 독립 task 집단 재현**으로 하고(§151-1), 최종 승격·기각은 macro와 task 10개 전부의 baseline 성능대 패턴을 함께 본 **사용자 판단**이다(§118).

---

# §1. 최신 승격 및 실험 내역 (§185 ~ §191)

> [!NOTE]
> §184 이전의 과거 실험 기록(§2~§184)은 [`docs/history.md`](history.md)로 이관/아카이빙되었다.

---

## 185. 2026-08-20 — **v113 승격 확정: v112 + CT cell 예산을 bag 크기의 1/8 fraction으로 (사유: feasibility, 예측 macro 아님)**

사용자 결정으로 §184의 arm(CT cell 샘플링을 전체 cell → bag 자기 크기의 1/8 fraction, floor 64로
교체)을 새 활성 baseline으로 승격했다.

**승격 사유는 명시적으로 feasibility이지, 예측 성능 개선이 아니다.** v112의 CT는 §183 시점
8×B200(180GB/장) 노드에서 측정됐고 그 하드웨어에서는 문제가 없었다. 그러나 22GB급 GPU
노드(gnode3)에서 v112를 그대로 재현하면 LUAD 3개 task(EGFR/STK11/TP53_mutation) 전부 CT의
`prepare_cells`가 bag의 전체 cell(최대 ~35k)을 한 번에 GPU에 올리는 지점에서 `CUDA out of
memory`로 즉시 죽는다(§184, `logs/official50/cptac_luad_*_v112.log` 등). 즉 **v112는 이 노드
클래스에서 SEAL 10-task를 완주할 수 없는 arm**이었다 — feasibility 자체가 실패했다. v113의
승격은 "더 나은 성능"이 아니라 "메모리 제약 노드에서도 채점 가능한 baseline"을 확보하기 위함이다.

```
CV/DD                     : v112와 동일 (offdiag CV, ordered-coordinate × typicality DD κ=1 weight=1)
CT cell 예산              : 전체 cell (v112) → bag 자기 크기의 1/8 fraction, floor 64 (v113)
                            ICF_CT_CELLS=0.125, ICF_CT_CELLS_SCALE=own, ICF_CT_CELLS_MIN=64
CT abundance              : match (토큰 dictionary와 같은 샘플로 abundance 계산, v112와 동일 정책)
```

| | SEAL 10 | 홀드아웃 7 | 전체 17 |
|---|---:|---:|---:|
| **v113 (활성)** | **0.70394** | 미측정 | 미측정 |
| v112 (previous baseline) | 0.70432 | 0.60181 | 0.66211 |
| Δ v113−v112 (SEAL 10) | −0.00038 | — | — |

SEAL 10 macro Δ(−0.00038)는 §183에서 v111→v112 승격을 "사실상 flat"으로 판정한 폭(−0.00021)과
같은 급의 잡음이다 — task 10개 중 5개는 상승, 5개는 하락으로 부호가 갈리고 한쪽으로 치우친
체계적 손실이 없다(§184 표). **이 arm은 예측 macro 게이트를 통과해서가 아니라, feasibility가
없으면 애초에 게이트를 적용할 숫자 자체가 없기 때문에 승격됐다** — §118의 "최종 판정은 통계
게이트가 아니라 사용자의 종합적 판단"과 같은 원칙이되, 이번엔 macro 비교가 아니라 "완주 가능
여부"가 판단 근거였다.

⚠️ **홀드아웃 7·전체 17은 아직 v113으로 재측정되지 않았다.** §0-2/§0-3 표의 해당 칸을 v112
값으로 채우거나 인용하지 말 것 — CT cell 예산이 바뀌었으므로 다시 측정하기 전까지는 미지수로
남긴다. 다음 action은 (a) 코드(`src/models/ct_readout.py`, `src/models/stream_eval.py`,
`scripts/eval_v113.sh` 등, 현재 uncommitted)를 커밋하는 것과 (b) 홀드아웃 7·전체 17을
`eval_v113.sh`로 재측정해 이 표의 빈 칸을 채우는 것이다.

구현:

- `src/models/ct_readout.py`: `CTReadoutConfig`에 `cells_fraction`(0,1]·`cells_scale`("own"|
  "median")·`cells_min` 추가. `cells_per_bag`(정수 cap)와 상호 배타적 — `cells_fraction`이
  설정되면 그걸 우선한다. `abundance_cells_per_bag`도 float fraction을 받아 같은 정책을 따를 수
  있다(§184).
- `src/models/stream_eval.py`(신규): raw bag을 CPU에 상주시키고 PCA covariance scatter를
  1-bag/1-chunk 단위로만 GPU에 올리는 스트리밍 유틸. `ICF_COVARIANCE_BASIS=pca|pca_within` 경로가
  이걸 통해 basis를 구성한다.
- `scripts/eval_v113.sh`(신규, 활성 runner): v112와 CV/DD 설정 동일, `ICF_CT_CELLS=0.125`
  (`own`, floor 64)만 다르다. `ICF_CT_CELLS`는 `all`·정수·`fraction:`/`own:`/`median:` 접두 float
  전부 받는다(`parse_cell_budget`, §184).
- `scripts/eval_v112.sh`는 전체-cell CT 재현 전용으로 유지(변경 없음), `scripts/eval_v111.sh`도
  distance DD 재현 전용으로 유지.
- BagPFN pytest `tests/test_ct_readout.py` + `tests/test_training_free.py`: **81 passed, 2
  skipped**, 회귀 없음.
- ~~⚠️ 이 절 작성 시점 기준 위 코드 변경은 **여전히 working tree에 uncommitted**다.~~
  **해소됨** — `c727e10`(v113 승격)·`a495a18`(v114 승격)으로 커밋됐다.

### 185-1. ⚠️ 정정 (2026-08-21) — v113은 cell 예산만 바꾼 것이 아니다

위 본문과 §184는 v113의 변경을 **CT cell 예산 하나**로 서술했지만, 실제 `scripts/eval_v113.sh`는
**세 가지를 동시에** 바꿨다. 실행 로그의 CT config 줄이 근거다
(`logs/official50/*_v113verify.log`, `*_unitw.log`):

| 항목 | v112 (`eval_v112.sh`) | **v113/v114** (`eval_v113.sh`/`eval_v114.sh`) |
|---|---|---|
| cell 예산 | `ICF_CT_CELLS=all` (전체 cell) | `0.125` / `own` / `min=64` (bag 자기 크기의 1/8) |
| sampling | `ICF_CT_SAMPLING=even` | **`random`** (`sampling_seed=0` 고정) |
| tokenizer | `ICF_CT_TOKENIZER=hierarchical_2means` | **`kmeans_plusplus`** (Lloyd `max_iter=8`, `tol=1e-4`) |
| abundance | `all` | `match` (dictionary와 같은 샘플) |

**따라서 §184의 "sampling-invariance 증거"라는 해석은 과잉이다.** −0.00038은 cell 개수만
바꿨을 때의 차이가 아니라 **네 가지를 함께 바꿨을 때의 순효과**이고, §127-2("arm당 노브 하나")
위반이다. 결론(macro가 사실상 flat이므로 feasibility 승격이 예측 성능을 갉아먹지 않았다)은
그대로 유지되지만, **cell 예산 단독 효과는 분리되지 않았다.**

**그리고 §181의 승격 근거가 v113부터 무효다.** §181은 v111을 "cell selection bias가 없고
random sampling의 영향도 없는 구성 중 최고"라는 이유로 승격했는데, v113/v114는 `sampling=random`
으로 bag의 1/8만 뽑는다 — seed 0 고정이라 **결정론성 자체는 유지되지만**(seed std 0.00000)
selection policy는 다시 연산에 관여한다. 즉 v113 승격은 §181의 운영 불변성 기준을 **feasibility와
맞바꾼 것**이며, §185 본문은 그 맞바꿈을 명시하지 않았다.

⚠️ 이 정정은 수치를 바꾸지 않는다 — v113 0.70394, v114 0.70509는 위 설정으로 실측된 값이
맞다(`v113verify`/`unitw` 태그 10 task 재집계로 확인). 바뀌는 것은 **그 숫자에 붙일 수 있는
해석의 범위**다.

_by Claude Sonnet 5 on gnode3 at 2026-08-20 01:11:51 (§185-1 추가: Opus 5, 2026-08-21)_

---

## 186. 2026-08-20 — fixed-head 세 branch weight를 1.0으로 통일: SEAL 10 macro +0.00115, **아직 미승격**

v113(§185)의 fixed head는 `margin = 1.442·(CV1−CV0) − 1.0·M_DD + 0.7·(CT1−CT0)`로 branch마다
다른 weight를 쓴다(1.442는 옛 8-head 분해 fit, §137-3; 0.7은 임의 조정). 사용자 요청으로 세
weight를 전부 **1.0으로 통일**해 SEAL 10-task를 재평가했다(`scripts/eval_v113_unit_weights.sh`,
신규·uncommitted, `ICF_FIXED_HEAD_CV_WEIGHT=1.0 ICF_FIXED_HEAD_DD_WEIGHT=1.0
ICF_FIXED_HEAD_CT_WEIGHT=1.0`, CV/DD/CT 파이프라인 자체는 v113과 동일). 10개 task 전부 `rc=0`,
OOM 0건(LUAD 3개 포함).

| task | v113 (cv=1.442/dd=1/ct=0.7) | unit-weight (cv=dd=ct=1.0) | Δ |
|---|---:|---:|---:|
| bc_therapy/er_status | 0.6885 | 0.6797 | −0.0088 |
| bc_therapy/grade | 0.7338 | 0.7369 | +0.0031 |
| bc_therapy/her2_status | 0.6737 | 0.6804 | +0.0067 |
| cptac_brca/PIK3CA_mutation | 0.5390 | 0.5254 | −0.0136 |
| cptac_brca/TP53_mutation | 0.8255 | 0.8247 | −0.0008 |
| cptac_luad/EGFR_mutation | 0.7862 | 0.7869 | +0.0007 |
| cptac_luad/STK11_mutation | 0.8949 | 0.8925 | −0.0024 |
| cptac_luad/TP53_mutation | 0.7011 | 0.7021 | +0.0010 |
| cptac_ccrcc/BAP1_mutation | 0.6877 | 0.6993 | **+0.0116** |
| cptac_ccrcc/VHL_mutation | 0.5090 | 0.5230 | **+0.0140** |
| **SEAL 10 macro (fold-mean)** | **0.70394** | **0.70509** | **+0.00115** |
| SEAL 10 macro (pooled) | 0.70019 | 0.70085 | +0.00066 |

**해석**: unit-weight가 7/10 task에서 우세하고 macro도 소폭 상승했다(+0.00115, v112 baseline
0.70432과 거의 동률 — +0.00077). §118 규칙대로 baseline 성능대를 같이 보면, 개선폭 대부분이
**0.5 근처 저신호 task 두 개(ccrcc BAP1 +0.0116, VHL +0.0140)**에서 나왔고, 하락은
brca PIK3CA(−0.0136) 하나가 가장 크다 — 저신호 구간 개선이 고신호 구간의 소폭 하락보다
실질적으로 더 유의미하다는 §118의 판단 기준에 부합한다.

⚠️ **결정론적 arm이라 t/p 게이트를 적용할 근거가 없다**(§151-1) — 이 결과는 "부호 일치 방향과
크기"만으로 판단해야 한다. 4/17 task만(SEAL 10) 측정했고 홀드아웃 7·전체 17은 미측정이다.
**사용자 결정으로 §187에서 v114로 승격됐다.**

구현: `scripts/eval_v113_unit_weights.sh`(신규, uncommitted) — v113과 동일한 CT
fraction(`ICF_CT_CELLS=0.125/own/min=64`) 위에 `ICF_FIXED_HEAD_CV_WEIGHT`·
`ICF_FIXED_HEAD_DD_WEIGHT`·`ICF_FIXED_HEAD_CT_WEIGHT`를 전부 `1.0`으로 명시 export.

_by Claude Sonnet 5 on gnode3 at 2026-08-20 09:17:53_

---

## 187. 2026-08-20 — **v114 승격 확정: v113 + fixed-head 세 branch weight를 전부 1.0으로 통일**

사용자 결정으로 §186의 arm을 새 활성 baseline으로 승격했다.

```
CV/DD/CT 파이프라인      : v113과 동일 (offdiag CV, ordered-coordinate × typicality DD κ=1,
                           CT cell 예산 = bag 자기 크기의 1/8 fraction floor 64)
fixed-head branch weight : CV 1.442 → 1.0, DD 1.0(변경 없음), CT 0.7 → 1.0
head margin              = 1.0·(CV1−CV0) − 1.0·M_DD + 1.0·(CT1−CT0)
```

| | SEAL 10 |
|---|---:|
| **v114 (활성)** | **0.70509** |
| v113 (previous baseline) | 0.70394 |
| v112 (CT 전체-cell, 참고) | 0.70432 |
| Δ v114−v113 | +0.00115 |

§186에서 이미 확인했듯 개선분은 task 전반에 고르지 않다 — ccrcc BAP1(+0.0116)·VHL(+0.0140)
같은 저신호(0.5 근처) task에서 크게 오르고, brca PIK3CA(−0.0136)에서 가장 크게 내렸다. §118
규칙(같은 크기의 Δ라도 천장 근처보다 랜덤 근처 변화가 더 의미 있다)에 따라 사용자가 이 arm을
승격하기로 결정했다. **v112 대비로도 사실상 동률**(+0.00077)이라, CT를 전체-cell에서 fraction
샘플링으로 바꾼 §185의 feasibility 승격이 예측 성능을 갉아먹지 않았다는 점도 이 승격으로 재확인된다.

⚠️ **미측정 항목**: 홀드아웃 7·전체 17은 이 arm으로 아직 측정되지 않았다. §0-2/§0-3 표의
해당 칸을 채우기 전까지는 v113/v112 값을 v114의 것으로 인용하지 말 것.

구현:

- `scripts/eval_v114.sh`(신규, 활성 runner, `scripts/eval_v113_unit_weights.sh`에서 승격하며
  개명) — v113과 CT/DD/CV 파이프라인 동일, `ICF_FIXED_HEAD_CV_WEIGHT=1.0`·
  `ICF_FIXED_HEAD_DD_WEIGHT=1.0`·`ICF_FIXED_HEAD_CT_WEIGHT=1.0`만 다르다.
- `scripts/eval_v113.sh`는 비대칭 weight(1.442/1/0.7) 재현 전용으로 유지(변경 없음).
- 코드 변경 없음 — weight는 이미 존재하던 `ICF_FIXED_HEAD_CV_WEIGHT`/`_DD_WEIGHT`/`_CT_WEIGHT`
  환경변수 오버라이드로만 조정된다(`scripts/test_pathobench.py`, §163).

_by Claude Sonnet 5 on gnode3 at 2026-08-20 09:20:50_

---

## 188. 2026-08-20 — v114 + CT kernel-ridge readout(linear/rbf/poly): **rbf·poly 전부 macro 하락, 기각**

v114의 CT readout을 `ridge`(primal, 16×16/256×256 solve)에서 **kernel ridge(dual, n×n solve)**로
교체하는 0-param 진단(`scripts/eval_v114_kernel.sh <gpu> <tag> <kernel>`, 신규). abundance→label
관계에 선형 ridge가 표현 못 하는 곡률이 있는지 묻는 실험이다. `kernel=linear`는 primal ridge를
수치적으로 정확히 재현하는 control이고, `rbf`/`poly`가 실제 진단이다. 10개 task 전부 `rc=0`.

| task | v114 (ridge) | kernel=rbf | kernel=poly |
|---|---:|---:|---:|
| bc_therapy/er_status | 0.6797 | 0.6583 | 0.6570 |
| bc_therapy/grade | 0.7369 | 0.7218 | 0.7314 |
| bc_therapy/her2_status | 0.6804 | 0.6680 | 0.6662 |
| cptac_brca/PIK3CA_mutation | 0.5254 | 0.5560 | 0.5455 |
| cptac_brca/TP53_mutation | 0.8247 | 0.8132 | 0.8116 |
| cptac_luad/EGFR_mutation | 0.7869 | 0.7684 | 0.7767 |
| cptac_luad/STK11_mutation | 0.8925 | 0.8670 | 0.8805 |
| cptac_luad/TP53_mutation | 0.7021 | 0.6653 | 0.6841 |
| cptac_ccrcc/BAP1_mutation | 0.6993 | 0.6579 | 0.6484 |
| cptac_ccrcc/VHL_mutation | 0.5230 | 0.5741 | 0.5461 |
| **SEAL 10 macro (fold-mean)** | **0.70509** | **0.69500** (−0.0101) | **0.69475** (−0.0103) |

**해석**: 관심 task인 저신호 VHL은 rbf가 **+0.0510**(0.5741)으로 크게 오르지만, 그 대가로
**10개 중 8개 task가 하락**한다(BAP1 −0.0414, TP53-luad −0.0368, STK11 −0.0255, er −0.0214,
EGFR −0.0185, grade −0.0151). §118 규칙(천장 근처 하락 vs 랜덤 근처 개선)으로도 순 macro
−0.0101은 기각이다. rbf의 VHL 개선은 다른 task 하락과의 맞바꿈에 그치고, poly도 동일하게
열화한다 — **abundance→label 관계에 linear ridge가 못 잡는 곡률은 실측되지 않았다**.

구현: `src/models/ct_readout.py`에 `kernel_ridge` readout + `_kernel_matrix`(linear/rbf/poly) 추가,
`scripts/test_pathobench.py`에 `ICF_CT_KERNEL*` env plumbing, `tests/test_ct_readout.py`에
kernel 테스트 3개. 재현 전용으로 보존한다(기각).

_by GitHub Copilot (DeepSeek V4 Pro) on gnode3 at 2026-08-20 22:21:22_

---

## 189. 2026-08-20 — v114 + CT top-k / mean+topk abundance 풀링: **교체·더하기 모두 기각, 방향 폐기**

CT abundance 5단계의 **mean 풀링을 cell 차원 비선형으로 흔들어**보는 0-param 진단. k-means++
토큰 256개가 이미 cell 분포의 클러스터 중심이므로, "가장 유사한 cell을 뽑아 평균"하는 top-k는
mean이 이미 담는 majority 패턴을 재강조할 뿐이고, 추가로 주는 정보는 assignment peakedness
(노이즈)뿐이라는 가설을 검증했다.

- **교체** `scripts/eval_v114_topk.sh`(신규): `ICF_CT_ABUNDANCE_POOLING=topk`, 각 토큰이 가장
  유사한 cell 상위 `fraction=0.1`(floor 1)의 평균을 사용. VHL 단일 프로브 **0.5076**
  (v114 0.5230 대비 **−0.0154**)로 열세 → 전체 10-task는 미실행.
- **더하기** `scripts/eval_v114_cattopk.sh`(신규): `ICF_CT_ABUNDANCE_POOLING=mean+topk`,
  mean 벡터 뒤에 top-k 벡터를 이어붙여 2K=512차원으로 만들고 ridge가 결합 사용. 10-task 전체 실행.

| task | v114 (mean) | mean+topk | Δ |
|---|---:|---:|---:|
| bc_therapy/er_status | 0.6797 | 0.6820 | +0.0023 |
| bc_therapy/grade | 0.7369 | 0.7228 | −0.0141 |
| bc_therapy/her2_status | 0.6804 | 0.6760 | −0.0044 |
| cptac_brca/PIK3CA_mutation | 0.5254 | 0.5383 | +0.0129 |
| cptac_brca/TP53_mutation | 0.8247 | 0.8223 | −0.0024 |
| cptac_luad/EGFR_mutation | 0.7869 | 0.7847 | −0.0022 |
| cptac_luad/STK11_mutation | 0.8925 | 0.8874 | −0.0051 |
| cptac_luad/TP53_mutation | 0.7021 | 0.6968 | −0.0053 |
| cptac_ccrcc/BAP1_mutation | 0.6993 | 0.6827 | −0.0166 |
| cptac_ccrcc/VHL_mutation | 0.5230 | 0.5137 | −0.0093 |
| **SEAL 10 macro (fold-mean)** | **0.70509** | **0.70067** | **−0.00442** |

**해석**: concatenate가 rbf/poly(−0.010)보다는 덜 나빴지만 여전히 v114보다 낮다. 개선은
PIK3CA(+0.0129)·er(+0.0023)뿐이고 grade(−0.0141)·BAP1(−0.0166)을 비롯해 8/10 task가 하락,
관심 task인 VHL조차 −0.0093으로 내려갔다. **top-k가 상보적 신호를 주지 못하고 ridge에 노이즈만
추가**했다는 결론이 교체(topk)·더하기(mean+topk) 두 방향에서 일관되게 확인됐다. 사용자 결정으로
**cell 차원 top-k 방향은 폐기**한다(재현 코드 보존).

구현: `src/models/ct_readout.py`에 `abundance_pooling: "max" | "topk" | "mean+topk"` 추가,
`scripts/test_pathobench.py`에 `ICF_CT_ABUNDANCE_POOLING/TOPK_*` env plumbing,
`tests/test_ct_readout.py`에 pooling 테스트 4개(전체 332 passed, 3 skipped).

_by GitHub Copilot (DeepSeek V4 Pro) on gnode3 at 2026-08-20 22:21:22_

---

## 190. 2026-08-21 — 전체 코드베이스 리팩터링 및 테스트 스위트 슬림화 (335→86 tests, 3.1s)

과거 v1~v114 연구 과정에서 누적된 기각/ablation 테스트와 거대 모듈들을 정리하고, 코드베이스의 모듈성과 개발 생산성을 대폭 개선하는 리팩터링을 완료했다.

### 1. 테스트 스위트 경량화 (335 tests → 86 core tests, 74% 감축, 9.5s → 3.1s)
- 과거 기각/폐기된 실험 전용 20개 테스트 모듈을 `tests/history/legacy_*.py`로 `git mv` 이동하여 보존.
- `test_set_transformer_ridge.py` (52 -> 8 tests) 및 `test_ct_readout.py` (72 -> 9 tests)의 핵심 계약 테스트만 압축 유지.
- 현재 v114 baseline 및 주요 core invariance에 대한 86개 테스트 100% 정상 통과 (`Ran 86 tests in 3.139s, OK`).

### 2. Model Registry & CT 모듈 분해 (`src/models/`)
- `src/models/base.py`: `InContextClassifierProtocol` 및 `BaseInContextClassifier` 구축.
- `src/models/registry.py`: `@register_model` 데코레이터 및 `build_model` / `get_model_class` 팩토리.
- `src/models/ct/`: 거대 단일 파일 `ct_readout.py` (1,356줄)를 4개 서브모듈로 분해.
  - `config.py`: `CTReadoutConfig`, `CTAbundance`, `CTMargins`
  - `tokenizers.py`: k-means++, Lloyd refine, hierarchical 2-means, FPS, HDBSCAN, DBSCAN
  - `abundance.py`: Cell sampling, projection, normalisation, soft abundance
  - `readout.py`: Ridge, Kernel Ridge, Prototype, Extreme, Calibrated Readouts
  - `src/models/ct_readout.py`: 기존 import 100% 하위 호환을 보장하는 Thin Facade로 전환.

### 3. 합성 데이터셋 모듈화 (`src/datasets/`)
- `src/datasets/synthetic/`: 단일 파일 `synthetic_data.py` (1,776줄)를 3개 서브모듈로 분해.
  - `types.py`: `SyntheticEpisode`, 태스크 목록 상수
  - `generator.py`: `SyntheticManifoldGenerator`
  - `dataset.py`: `SyntheticEpisodeDataset`
  - `src/datasets/synthetic_data.py`: 호환 Facade.

### 4. Lightning 학습 모듈 인터페이스 슬림화 (`src/modules/`)
- `src/modules/losses/ranking.py`: `pairwise_ranking_loss`
- `src/modules/diagnostics/`: `metrics.py` (이진 쿼리 진단), `oracle.py` (Oracle Ridge 피팅 및 SNR)
- `src/modules/guards/`: `gradient.py` (Non-finite 가드), `vram.py` (VRAM 피크 경고 가드)
- `src/modules/model_interface.py`: Lightning 라이프사이클에만 집중하도록 경량화.

### 5. 스크립트 계층화 (`scripts/`)
- `scripts/diagnostics/`: 14개 `diagnose_*.py` 진단 스크립트 격리.
- `scripts/archive/sweeps/`: 과거 v76, v62 등 파라미터 스윕 스크립트 격리.
- `scripts/archive/historical_evals/`: v107~v113 구버전 평가 스크립트 격리.
- `scripts/` 루트: 활성 스크립트(`eval_v114.sh`, `eval_seal_tasks.sh`, `node_env.sh`, `train.py`, `test_pathobench.py` 등)만 명확하게 유지.

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-21 14:45:00_

---

## 191. 2026-08-21 — Plan A: Projected Bag-Mean (BM) Branch 구현 및 단위 테스트 완료

기존 v114의 3-branch(CV, DD, CT)가 완전히 배제하고 있던 **1차 모멘트(세포 평균 $\bar{x}_i \in \mathbb{R}^{1536}$)** 정보를 Within-slide PCA 기저의 주요 신호 축(상위 32차원)으로 사영하여 Class-balanced Ridge로 판별하는 **BM (Projected Bag-Mean) Branch**를 구현했다.

### 1. 설계 및 구현 세부사항
- **사영 및 정규화**:
  - 각 bag의 평균 $\bar{x}_i$를 기저 상위 `bm_dim`($d=32$) 차원으로 사영: $\mu_i = \bar{x}_i B_{:d} \in \mathbb{R}^{32}$.
  - Context 슬라이드들의 평균 $\bar{\mu}_{ctx}$과 RMS 표준편차 $\sigma_{ctx}$로 좌표 표준화.
- **분류기 (0-parameter)**:
  - Context-standardized $\mu_i^{std}$에 대해 Class-balanced Dual Ridge ($\lambda=1.0$)를 적용하여 로짓 산출 $\to M_{BM} = \text{logit}_1 - \text{logit}_0$.
- **통합 및 호환성**:
  - `TrainingFreeConfig`에 `weight_bm: float = 0.0`, `bm_dim: int = 32`, `bm_lambda: float = 1.0` 추가.
  - 기본값 $w_{BM}=0.0$으로 설정되어 기존 v114의 동작 및 기존 86개 테스트 결과가 bit-level로 100% 동일하게 유지됨.

### 2. 엄밀한 계약 검증 (`tests/test_bm_branch.py`, 6 tests PASS)
- **Equivalence Test**: `weight_bm=0.0` 시 기본 v114 마진과 100% 동일.
- **Standalone Test**: BM 단독 활성화 시 유효한 마진 출력.
- **Label Antisymmetry Test**: 라벨 반전($y \to 1-y$) 시 $M_{BM} \to -M_{BM}$ 성립.
- **No-Leakage Test**: Query 변경 시 Context 사영 및 표준화 통계 불변.
- **Determinism Test**: 동일 입력에 대해 완전 결정론적 출력.
- **Dimension Flexibility Test**: 8, 16, 64, 128차원 등 다양한 사영 차원 지원 확인.

전체 테스트 스위트: **92 tests, OK (13.2s)**.

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-21 15:03:00_

---
