# `scripts/analysis/` 중복 감사 — 같은 이름이 다른 뜻을 갖는 곳

**범위**: `scripts/analysis/*.py` 50개 파일. 필요할 때 `src/`와 `scripts/`의 대응 구현을
참조했으나, 수정·이동·삭제는 하지 않았다.

**방법**: 50개 파일 전체의 `def`/`class` 목록을 뽑아 이름별로 묶고, 묶인 함수는 본문을
전부 읽어 **직접 대조**했다. 이름이 같다는 이유로 같다고 가정하지 않았다. 함수 본문을
공백·주석을 제거해 해시로 비교해 1차 후보를 추린 뒤, 실제로 다른 것은 줄 단위로 차이를
확인했다.

**요약**: 이름이 같은데 구현이 다른 사고성 중복은 **함수 6건, 클래스 1건, 상수 3계열**이다.
반면, 배경에 적힌 `effective_rank` 사고는 현재 코드에서는 **같은 이름 충돌로 남아 있지
않다**(§A-6). 사고의 두 정의는 `effective_rank`와 `entropy_rank`로 이름이 분리되어 있다.

---

## A. 이름이 같은데 구현이 다른 것 (최우선)

### A-1. `eval_method` — 완전히 다른 함수 두 개

| 파일:줄 | 시그니처 | 하는 일 |
|---|---|---|
| `eval_gated_voting.py:35` | `eval_method(agg_fn)` | Primary 7 전체를 돌며 `agg_fn`(집계 함수)의 task별 평균 AUROC 계산 |
| `furthest_trimming.py:55` | `eval_method(pts, method)` | 9개 문자열 method 분기로 trimmed/drop-k/median 집계를 계산 |

**구현이 다른가**: 전혀 다르다. 공통점은 이름뿐이다.

- `eval_gated_voting.eval_method`는 집계 함수 객체를 받아 `probs` 스택에 적용한다.
- `furthest_trimming.eval_method`는 `method` 문자열로 분기하며, **6-branch를 전제로
  나눗셈 상수 `4.0`/`5.0`을 하드코딩**한다(`furthest_trimming.py:73,80,87,94,101,109,117`).
  branch 수가 바뀌면 조용히 틀린다.

**통합 위치**: 성격이 다르므로 통합 대상이 아니다. 이름만 분리해야 한다
(예: `furthest_trimming.eval_aggregation_method`, `eval_gated_voting.eval_task_auroc`).
`furthest_trimming`의 나눗셈 상수는 `len(probs)` 기반으로 바꿔야 한다.

### A-2. `trimmed_mean` — 같은 이름, 입력 규약이 다름 (사고 위험 최상)

5개 정의 + `src/models/aggregations/voting.py:244` 1개.

| 파일:줄 | 입력 | 내부 sigmoid | min/max 제거 | B<3 | clamp | dtype |
|---|---|---|---|---|---|---|
| `branch_diagnostics.py:49` | `probs:[B,N]` | 아니오 (호출자가 함) | `sort[1:-1]` | mean으로 대체 | 없음 | 입력 dtype |
| `decision_precision.py:58` | `probs:[B,N]` | 아니오 | `sort[1:-1]` | mean으로 대체 | 없음 | 입력 dtype |
| `eval_gated_voting.py:52` | `probs:[B,N]` | 아니오 | `sort[1:-1]` | **가드 없음** | 없음 | 입력 dtype |
| `ru82_lambda_ensemble.py:28` | `list[Tensor]` | 아니오 | `stack().sort[1:-1]` | 가드 없음 | 없음 | 입력 dtype |
| `ru87_precision.py:95` | `list[마진]`(로짓) | **예, 내부에서** | sum−min−max | **예외 발생** | `[1e-7, 1-1e-7]` | **float64** |
| `src/.../voting.py:244` | 생산 경로 | — | — | — | — | — |

**구현이 실제로 다른가**: **다르다.** 셋 중 최소 두 가지가 의미를 바꾼다.

1. **입력이 확률인가 마진(로짓)인가.** `ru87_precision.trimmed_mean`은 마진 목록을 받아
   내부에서 `sigmoid`를 적용한다(`ru87_precision.py:101`). 나머지는 이미 확률이라고
   가정한다. 같은 이름으로 마진을 넘기면 `branch_diagnostics` 쪽은 시그모이드를 빠뜨린
   채 평균 내고, 반대로 확률을 `ru87_precision`에 넘기면 이중 시그모이드가 된다.
   **같은 행렬에서 다른 숫자가 나오는 정확한 구조다.**
2. **float64/clamp/가드.** `ru87_precision`만 float64·clamp·`<3` 예외를 쓴다. 나머지는
   bf16/fp32 그대로이며, `eval_gated_voting`/`ru82`는 branch 수가 2 이하일 때
   `sort[1:-1]`이 빈 텐서가 되어 조용히 NaN을 낳는다.
3. **결과 방향.** `branch_diagnostics`의 `sorted[1:-1].mean()`과
   `ru87_precision`의 `(sum-min-max)/(n-2)`는 수학적으로 같지만, 전자는 정렬,
   후자는 산술이라 tie/부동소수점에서 최하위 비트가 다를 수 있다.

**인라인 복제**: 같은 "min 하나·max 하나 제거" 로직이 함수로 추출되지 않고 7곳에 다시
쓰였다. `eval_combinations.py:124-129`, `search_optimal_combinations.py:79-85`,
`furthest_trimming.py:69-73`, `drop2_furthest_detail.py:62-66`,
`compare_all_voting.py:28`, `ru81_probe.py:41-43`, `ru81_audit.py:22-26`.
특히 `ru81_audit.combine`은 6-branch를 하드코딩해 `ordered[1]+ordered[2]+ordered[3])/3`로
계산하며, float32 반올림 순서를 보존하려는 의도가 주석에 있다
(`ru81_audit.py:24-26`). 일반화된 `trimmed_mean`과 **일부러 다른 구현**이므로 통합 시
이 의도를 보존해야 한다.

**통합 위치**: `src/utils/metrics.py`에 `trimmed_mean_prob(probs)`(확률 입력)와
`trimmed_mean_margin(margins)`(마진 입력, sigmoid·clamp 포함)를 각각 정의하고,
`src/models/aggregations/voting.py`의 생산 구현과 결과가 일치하는지 골든 테스트를 둔다.
`ru81_audit.combine`은 감사 독립성을 위해 의도적으로 남긴다.

### A-3. `load` — 같은 이름, 반환 계약이 세 가지

| 파일:줄 | 시그니처 | 반환 | 실패 시 |
|---|---|---|---|
| `branch_diagnostics.py:57` | `load(tag)` | `{task: per_fold}` (PRIMARY7 고정) | 예외 전파 |
| `decision_precision.py:65` | `load(tag)` | `{task: per_fold}` (PRIMARY7) | 예외 전파 |
| `ru89_shape_joint.py:38` | `load(tag)` | `{task: per_fold}` + `m_sj←m_shj` 폴백 | 예외 전파 |
| `ru90_wiring_equivalence.py:78` | `load_folds(tag, tasks)` | `{task: per_fold}` + 폴백, task 지정 | 예외 전파 |
| `ru86_gate2.py:64` | `load(stem, tag)` | **원시 blob 전체** | `SystemExit` |
| `ru87_precision.py:88` | `load(task, tag)` | **원시 blob 전체** | `PairingError` |
| `compare_arms_paired.py:62` | `load_arm(root, task, tag)` | **원시 blob 전체** | `PairingError` |
| `compare_predictions.py:30` | `load(path)` | `(prob, target, episode)` 튜플 | 예외 전파 |

**구현이 실제로 다른가**: 다르다. 최소 세 계약이 `load` 한 이름에 겹쳐 있다.
`branch_diagnostics.load`는 task를 **인자로 받지 않고 PRIMARY7로 고정**하므로,
task가 다르면 조용히 빈/다른 데이터가 된다. `ru86`/`ru87`/`compare_arms_paired`는 blob을
그대로 돌려준다. `ru89`만 `m_sj←m_shj` 폴백을 넣었고, 같은 폴백이
`branch_screen.py:48-50`에 **또 인라인**되어 있다.

게다가 `ru87_precision.PairingError`(`ru87_precision.py:80`)와
`compare_arms_paired.PairingError`(`compare_arms_paired.py:54`)는 **서로 다른 클래스**다.
둘 다 `RuntimeError`를 상속하지만 서로 import하지 않으므로, 한쪽에서
`except PairingError`로 다른 쪽 예외를 잡지 못한다.

**통합 위치**: `scripts/analysis/_io.py`(또는 기존 공용 모듈)에
`load_per_fold(tag, tasks) -> dict`, `load_blob(task, tag) -> dict`,
`with_sj_fallback(folds)`를 두고 `PairingError`는 하나만 정의해 공유한다.

### A-4. `fold_aurocs` — 같은 이름, 키·반환형·입력이 다름

- `compare_arms_paired.py:69` `fold_aurocs(record) -> torch.Tensor`
  — `record["per_fold"]`의 `probability`에 `src.utils.metrics.auroc_rows` 적용, fold 순서 텐서.
- `parity_paired_delta.py:32` `fold_aurocs(path) -> dict[int, float]`
  — **파일 경로를 직접 받아 로드**하고 `rec["fold"]`를 키로 하는 dict, `auroc` 적용.

**구현이 실제로 다른가**: 다르다. 입력이 레코드 vs 경로, 반환형이 텐서 vs dict,
키가 암묵적 순서 vs `"fold"` 명시값, AUROC 구현이 `auroc_rows` vs `auroc`이다.
`parity_paired_delta`는 이어서 `set(fa) & set(fb)`로 fold 키가 다르면
**경고만 하고 제외**한다(`parity_paired_delta.py:54-56`). `compare_arms_paired`는
같은 불일치를 `PairingError`로 **하드 실패**시킨다(`compare_arms_paired.py:88-92`).
같은 이름의 함수가 같은 문제를 정반대로 처리한다.

**통합 위치**: `fold_aurocs(record) -> dict[int, float]` 하나로 통일하고, 파싱된 레코드만
받는다. 경고-제외 vs 예외 정책은 호출부 파라미터로 드러낸다.

### A-5. `prediction_path` — 같은 패턴, 시그니처만 다름

- `compare_arms_paired.py:58` `prediction_path(root, task, tag) -> Path`
- `ru87_precision.py:84` `prediction_path(task, tag) -> Path`
- `ru90_wiring_equivalence.py:66` `_path(task, tag) -> str` (상대경로 문자열)

**구현이 실제로 다른가**: 본문 패턴은 같고 root를 인자로 받는지만 다르다.
그러나 같은 문자열 `predictions/pathobench_{task}_{tag}_official50_bf16.pt`가
**51개 파일에 35번 인라인**되어 있어, 이 함수를 쓰지 않는 곳이 대부분이다.

**통합 위치**: `prediction_path(root, task, tag)` 하나로 모으고, `root`는 공용 상수에서
가져온다.

### A-6. `effective_rank` — **현재는 안전** (배경 사고의 재확인)

- `branch_diagnostics.py:86` — `(sum L)^2 / sum L^2`, numpy, `clip(0)`.
- `branch_redundancy.py:42` — `(sum L)^2 / sum L^2`, torch, `clamp(min=0)`.

**구현이 실제로 같은가**: **실질적으로 같다.** 입력만 numpy vs torch이고 수식이 동일하다.
배경에 적힌 사고의 다른 정의(엔트로피 `exp(-sum p log p)`)는 같은 이름이 아니라
**`entropy_rank`라는 별도 이름**으로 존재한다(`branch_redundancy.py:56`). 그리고
`branch_redundancy.py:42-60`의 docstring이 그 사고(3.52 vs 4.61)를 명시적으로 기록한다.

즉 이 사고는 **이미 이름 분리로 교정되어 있다.** 다만 세 번째 구현이 다른 이름으로
숨어 있다: `ru88_fingerprint.f3_spectrum`(`ru88_fingerprint.py:226-234`)가
`pr = sum^2 / sum^2`로 참여비를 계산하면서 `F3b_participation_ratio`라는 이름을 쓴다.
같은 값이 `effective_rank`와 `F3b_participation_ratio` 두 이름으로 보고된다.

**통합 위치**: `src/utils/metrics.py`의 `participation_ratio(eigenvalues)` 하나로 모으고,
`entropy_rank`는 이름과 반환 키에 "not comparable to participation ratio"를 유지한다.
`branch_redundancy.py`의 경고 docstring은 그대로 보존한다(현재 유일한 사고 방지 장치).

---

## B. 이름은 다르지만 하는 일이 같은 것

### B-1. AUROC — 여러 이름, 4가지 구현, 2가지 단일클래스 계약

| 이름 | 위치 | 구현 |
|---|---|---|
| `compute_auroc` | `context_weighting.py:19`, `eval_combinations.py:26`, `eval_pure_in_episode_weighting.py:26`, `parse_de_sw_results.py:5`, `search_optimal_combinations.py:18`, `test_saved_logits_weighting.py:27` | Mann-Whitney rank + average ties. **6개 본문이 (docstring 제외) 사실상 동일** |
| `compute_auroc` (별칭) | `eval_gated_voting.py:13`, `analyze_voting.py:16` | `src.utils.metrics.auroc`를 import해 이름만 `compute_auroc`로 바꿈 |
| `auroc` | `src/utils/metrics.py:60` / `auroc_rows:18` | vectorized rank, float64 |
| `auc` | `ru81_audit.py:15` | pairwise `O(n_pos*n_neg)`, tie=0.5. 감사 독립성 목적 |
| `auroc_pairwise` | `ru82_lambda_ensemble.py:55` | pairwise tie=0.5, `auroc()` 교차검증용 |
| `per_fold_mdx_auroc` | `ru86_gate2.py:71` | `auroc` 래퍼 |
| `per_fold_branch_auroc` | `decision_precision.py:75` | `auroc` 래퍼 |
| `per_fold_auroc` | `ru89_shape_joint.py:50`, `ru90_wiring_equivalence.py:90` | 트림 평균 래퍼. **두 본문이 완전히 동일** (해시 일치) |

**실제로 같은가**: 수학적 결과는 모두 Mann-Whitney U로 같다. 그러나 **단일클래스
처리 계약이 갈린다.**

- 6개 로컬 복제는 `positives == 0 or negatives == 0`이면 **`None`을 반환**한다
  (`parse_de_sw_results.py:9-10`). 호출부는 `if fa is not None`으로 거른다.
- `src.utils.metrics.auroc`는 **NaN을 반환**한다(`metrics.py:52-56`). 별칭을 쓴
  `eval_gated_voting.py:45`는 NaN을 그대로 리스트에 넣고 `np.mean`을 취해
  **macro 전체가 NaN으로 오염**될 수 있다.

이것이 배경 사고와 같은 구조다: 이름은 `compute_auroc`으로 같은데 단일클래스 계약이
달라, 같은 데이터에서 한쪽은 값을 버리고 한쪽은 NaN을 낸다.

**통합 위치**: `src/utils/metrics.py`의 `auroc` 하나로 통일하되, 단일클래스 처리를
명시적 옵션(`on_single_class="nan"|"none"|"raise"`)으로 드러낸다. `ru81_audit.auc`와
`ru82.auroc_pairwise`는 독립 교차검증 목적이므로 남기되 docstring에 이유를 명시한다.

### B-2. fold-paired / task-cluster 신뢰구간 — 여러 이름

같은 "task별 평균을 군집으로 보고 t(df=n−1) 구간" 로직이 반복된다.

- `ru82_lambda_ensemble.py:66` `cluster_interval(task_means)`
- `ru83_rank_size.py:77` `cluster(task_means)` — **`cluster_interval`과 수식이 동일**,
  부호 일치 수를 덧붙임
- `ru89_shape_joint.py:57` `task_cluster_ci(task_means)` — t 임계값이 scipy 실패 시
  `2.446949` 하드코딩(`ru89_shape_joint.py:31-35`)
- `ru90_wiring_equivalence.py:51` — `ru89`의 `task_cluster_ci`를 import해 재사용(유일한 재사용)
- `decision_precision.py:94` `cluster_se(values, task_ix)` — CI 없이 SE만
- `gf_paired_delta.py:39` `paired_ci(a, b)` — t=2.0096 하드코딩
- `parity_paired_delta.py:75-82` — t 테이블 `{6:2.447,5:2.571,4:2.776}` 하드코딩
- `ru81_report.py:48-49` — `t.ppf(.975,6)` 인라인
- `ru81_audit.py:96` — `t.ppf(.975,6)` 인라인
- `compare_arms_paired.py:105` `bootstrap_means` + `:114` `percentile_interval`
- `summarize_slot_headroom.py:21` `paired_bootstrap`

**실제로 같은가**: `cluster_interval`/`cluster`/`task_cluster_ci`는 수식이 같고 반환형만
다르다(SE 포함 여부, 부호 카운트). t 임계값이 scipy 실패 시 `2.446949` vs `2.447` vs
`2.0096`(fold df=49)로 **하드코딩 출처가 제각각**이다. `bootstrap_means`와
`paired_bootstrap`은 둘 다 fold 재표집이지만 난수 생성 방식(`torch.randint` vs 동일)과
CI 산출(분위수 vs t)이 다르다.

**통합 위치**: `src/utils/metrics.py`에 `task_cluster_ci(task_means, level=0.95)`와
`paired_fold_interval(deltas, method="bootstrap"|"t")`로 모은다.

### B-3. 참여비(participation ratio) — B-1·A-6 참조

`effective_rank`(2곳) + `f3_spectrum`의 `pr`(`ru88_fingerprint.py:233`)이 같은 수식이다.

### B-4. rank / zscore 정규화 — 4가지 이름, 2쌍 동일

- `analyze_voting.py:139` `rank_score` == `compare_all_voting.py:13` `rank_norm`
  (본문 동일: `argsort` → `arange` → `/ (n-1)`).
- `analyze_voting.py:153` `zscore` == `compare_all_voting.py:19` `zscore_norm`
  (본문 동일: `std(unbiased=False).clamp_min(1e-6)`).
- `ru88_fingerprint.py:269` `rank` — numpy, **tie 평균** 포함. 위 둘은 tie 처리가 없다.
  같은 이름의 "rank"가 tie에서 다른 값을 낸다.
- AUROC 내부 ranking과도 tie 처리가 다르다.

추가로 `compare_all_voting.py:2`는 `from scripts.analyze_voting import ...`를 쓰는데,
저장소에 `scripts/analyze_voting.py`는 없고 실제 모듈은
`scripts/analysis/analyze_voting.py`다. **import가 깨져 있다**(실행으로 확인하지는
않았고 `ls`로 경로 부재만 확인 — 코드 수정 금지 지시 준수).

**통합 위치**: tie 처리를 명시한 `rankdata`(tie 평균) 하나로 통일하고, tie 없음이
보장될 때만 빠른 경로를 쓴다. import 경로는 `scripts.analysis.*`로 바로잡아야 한다.

### B-5. fold 로딩 / task 로딩 — A-3 참조

`load` 계열 + `per_fold_macro`(`gf_paired_delta.py:30`) + `load_curves`
(`gf_variance_decomp.py:36`) + `load_raw`(`ru90:70`) + `read_task`
(`ru88_fingerprint.py:146`) + `index_h5`(`ru88:136`)가 각자 파일 로딩을 중복 구현한다.

### B-6. 상관 계산 — 5가지 이름

- `branch_diagnostics.py:76` `corr_matrix(folds, subset)` — fold별 Pearson 평균
- `ru85_gate1.py:34` `correlate(a, b)` — `np.corrcoef`, 표준편차 0 가드
- `ru88_fingerprint.py:282` `cluster_correlation` — Pearson + 군집 SE
- `ru88_fingerprint.py:299` `spearman_cluster` — rank 후 위 함수
- `ru83_rank_size.py:13,53,67`, `ru81_report.py:42`, `loo_capacity.py:21` — `scipy.spearmanr` 직접 호출

**실제로 같은가**: 계산하는 통계량이 Pearson vs Spearman vs (Pearson+군집 CI)로 섞여
있다. 이름만 "correlation"으로 뭉뚱그리면 A-6형 사고가 재발한다.

### B-7. ICC — `icc` vs `draw_icc`

- `subsample_sweep.py:29` `icc(draws)`
- `subsample_selector.py:37` `draw_icc(draws)`

**실제로 같은가**: 수식은 같다(`between / (between + within)`). **차이는 분모가 0일 때**다.
`icc`는 `1.0`, `draw_icc`는 `0.0`을 반환한다(`subsample_sweep.py:33` 대
`subsample_selector.py:43`). 상수 draw(분산 0)에서 두 함수가 정반대 값을 낸다.

**통합 위치**: `src/utils/metrics.py`의 `intraclass_correlation(draws)` 하나로 모으고
0분산 정책을 인자로 고정한다.

---

## C. 같은 상수가 여러 파일에 하드코딩된 곳

### C-1. 과제 목록 — **두 가지 표기법** (사고 위험)

같은 과제가 두 표기로 흩어져 있다.

- 슬래시형 `cptac_lscc/ARID1A_mutation`: `branch_redundancy.py:36-39`,
  `context_weighting.py:7-15`, `eval_combinations.py:5-13`,
  `eval_in_episode_loo.py:21-29`, `eval_pure_in_episode_weighting.py:5-13`,
  `parse_de_sw_results.py:24-32`, `profile_loo_fold.py`, `ru81_launch.py:18-21`,
  `ru88_fingerprint.py:105-115`, `search_optimal_combinations.py:6-14`,
  `test_saved_logits_weighting.py:6-14`
- 밑줄형 `cptac_lscc_ARID1A_mutation`: `analyze_voting.py:19-27`,
  `branch_diagnostics.py:34-42`, `decision_precision.py:45-53`,
  `drop2_furthest_detail.py:15-23`, `eval_gated_voting.py:15-23`,
  `furthest_trimming.py:15-23`, `parity_paired_delta.py:26-29`,
  `parse_ds_aug_results.py`, `parse_v121_results.py`, `ru86_gate2.py:48-56`,
  `ru87_precision.py:62-70`, `ru88_fingerprint.py:107-114`

`ru88_fingerprint.py`는 **두 표기를 동시에** 담고 변환한다(`:107-114`).
이름을 key로 쓰는 코드에서 두 표기가 섞이면 "문서 수치 재현 실패"로 오진된다.

추가로 `SEAL_10_TASKS`가 `analyze_voting.py:29-40`과
`compare_arms_paired.py:40-51`에 **순서까지 다른 채** 두 번 나열된다
(`analyze_voting`은 `cptac_*` 먼저, `compare_arms_paired`는 `bc_therapy_*` 먼저).

**통합 위치**: 하나의 `scripts/analysis/tasks.py`에 정본 목록을 두고,
`canonical(task)` / `display(task)` 변환을 제공한다.

### C-2. `V120_BASELINE` — 7개 파일에 값 중복

- 스칼라 형태 `0.6265`: `context_weighting.py:17`, `search_optimal_combinations.py:16`
- dict 형태(7과제 + Macro): `eval_combinations.py:15-24`,
  `eval_in_episode_loo.py:31-40`, `eval_pure_in_episode_weighting.py:15-24`,
  `parse_de_sw_results.py:44-53`, `test_saved_logits_weighting.py:16-25`

dict 5개의 값이 모두 동일하다. Macro `0.6265`는 스칼라 2곳과 dict 5곳, 총 7곳에 있다.
기준선이 갱신되면 7곳을 동시에 고쳐야 하고, 하나라도 놓치면 파일별로 다른 Δ가 보고된다.

**통합 위치**: `scripts/analysis/tasks.py`에 `V120_BASELINE` dict 하나.

### C-3. branch 목록 — 8가지 정의

| 위치:줄 | 내용 |
|---|---|
| `branch_diagnostics.py:43` | `["m_cv","m_bm","m_bd","m_qa","m_ds","m_sh","m_sj"]` |
| `branch_diagnostics.py:45` | `BRANCHES_V121_5 = [...5개]` |
| `decision_precision.py:54` | `[...5개]` (동일 but 별도 정의) |
| `drop2_furthest_detail.py:38` | `[...m_ct 포함 6개]` |
| `furthest_trimming.py:38` | `[...m_ct 포함 6개]` |
| `eval_combinations.py:49` | `[...m_de,m_sw 포함 8개]` |
| `ru87_precision.py:72` | `ARM_A_BRANCHES` 5개 + `ARM_B_BRANCHES` |
| `compare_all_voting.py:36-38` | config별 4·5개 |
| `ru82_lambda_ensemble.py:22-23`, `ru83_rank_size.py:22-23` | `('cv','bm','qa','ds')`(접두사 없음) |

`m_` 접두사 유무, CT 포함 여부, SH/SJ 포함 여부가 파일마다 다르다. 같은 이름
`BRANCHES`가 5개(`decision_precision`)와 7개(`branch_diagnostics`)를 가리킨다.

**통합 위치**: `BRANCHES_V121_5`, `BRANCHES_V121_7`, `BRANCHES_V120_6`,
`BRANCHES_V121_8`을 한 모듈에 상수로 두고, 각 스크립트는 이름으로 선택한다.

### C-4. 예측 파일 경로 패턴

`predictions/pathobench_{task}_{tag}_official50_bf16.pt`가 **35곳에 인라인**.
`predictions/` 디렉터리 상수도 곳곳에 직접 쓰인다. A-5의 `prediction_path`로 모아야 한다.

### C-5. 저장소 루트 계산 — 3가지 관례

`Path(__file__).resolve().parents[2]`가 26곳에 인라인. 그런데
`eval_in_episode_loo.py:11`은 `parents[1]`(=`scripts/`)을 `PROJECT_ROOT`로 쓰며,
`compare_predictions.py:23`, `eval_combinations.py`, `furthest_trimming.py:7`,
`drop2_furthest_detail.py:7` 등은 `parents[2]`를 쓴다. **동시에 두 관례가 존재**한다.
`PROJECT_ROOT`, `ROOT`, `PRED_ROOT`, `PROJECT_ROOT` 이름도 제각각이다.

### C-6. 임계값·상수

- `0.5`(chance floor): `V120_BASELINE`, `ICC_THRESHOLD`(`subsample_selector.py:34`),
  `FOLD_BAR`(`branch_significance.py:26`), `REJECT_ABOVE`(`branch_screen.py:20`),
  `FLOOR`(`subsample_sweep.py:26`) — **서로 다른 뜻의 0.5가 같은 파일군에 병존**.
- t 임계값: `2.446949`(`ru89_shape_joint.py:34`), `2.447`(`parity_paired_delta.py:78`),
  `2.0096`(`gf_paired_delta.py:47`) — 출처가 하드코딩으로 흩어짐.
- `Z_ALPHA, Z_POWER = 1.959964, 0.841621`(`decision_precision.py:55`)이 다른 파일의
  `t.ppf`/`1.96`과 공존.
- 차원 상수 `SKETCH_DIM=256`, `EMBED_DIM=32`, `FEATURE_DIM=1536`
  (`ru88_fingerprint.py:123-130`, `probe_slot_headroom.py:68`,
  `patch_likelihood.py:11,12`, `fisher_basis_probe.py:18,19`, `loo_capacity.py:26`).
  `FEATURE_DIM`은 `scripts/evaluate_pure`에서 import하는 곳과 직접 쓰는 곳이 섞인다.

### C-7. 기본 태그

`"v121_baseline"`이 `branch_diagnostics.py:112`, `decision_precision.py:106`,
`ru86_gate2.py:59`, `ru87_precision.py:77` 등에 기본값으로 하드코딩.
`"ru85_stageB"`, `"ensemble_8branch_primary7"`, `"v121_sh_variants"`,
`"qa_w1_primary7"` 등 tag도 여러 파일에 중복 문자열로 등장한다.

---

## D. 통합 우선순위 (권고)

코드 수정은 하지 않았다. 순서만 제안한다.

1. **A-2 `trimmed_mean`** — 입력이 확률인지 마진인지 갈리는 것이 가장 위험. 여기서
   A-6형 오진이 재발할 수 있다.
2. **A-1 `eval_method`** — 이름만 같고 완전히 다른 함수. `furthest_trimming`의 상수
   나눗셈은 branch 수 변경 시 조용히 틀린다.
3. **B-1 AUROC 단일클래스 계약** — `None` vs `NaN`이 macro 오염으로 이어질 수 있다.
4. **C-1 과제 두 표기 / C-2 `V120_BASELINE`** — 상수 드리프트의 진원지.
5. **A-3 `load` / A-4 `fold_aurocs` / A-5 `prediction_path`** — 계약 통일.
6. **A-6·B-3 참여비** — 이미 이름 분리됨. `participation_ratio` 하나로 모으고
   `entropy_rank`의 경고 docstring을 보존.
7. B-2·B-4·B-5·B-6·B-7·C-3\~C-7 — 공용 모듈로 점진 이동.

**하지 않은 것**: 코드 수정, 파일 삭제, 통합 실행. 본 보고서의 모든 "구현이 다른가"는
파일을 직접 읽어 확인한 내용이며, 실행 검증(예: `compare_all_voting.py`의 import 오류)은
지시에 따라 수행하지 않고 **추정**으로 표시했다.

---

[작성자: opencode / 소집자(사용자 지시에 의한 단일 세션 감사) / deepseek-v4.1-flash (effort: 미확인) · 2026-09-18 KST]
