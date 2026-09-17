# ICF Council — C-20260918-9

- **연구 질문**: 계측 결함이 확인됐다. 이 회차는 그 파급 범위를 정한다.

확인된 사실(첨부 보고서 §3~§6): (1) ICF_SHAPE_SCREEN_ONLY 환경변수는 비테스트 코드 어디에서도 읽히지 않는다. (2) ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT를 1.0에서 0.0으로 바꿔도 SMAD4 0.4661, PBRM1 0.5408로 소수 4자리까지 출력이 동일하다. 두 실행은 결정론적으로 같으므로 잡음이 아니다. (3) 문서에 고정된 5-branch 오염 검사 앵커 SMAD4 0.4421, PBRM1 0.5553은 현재 코드로 재현되지 않는다. (4) 재생성 macro 0.6226은 5-branch 0.6171이 아니라 7-branch 0.6227에 맞는다.

질문: (1) 이 결함이 과거 결과 중 무엇을 무효화하는가? PROJECT.md가 기록한 승격 판정(D-042, 7-branch 채택)과 게이트 1 스크리닝 결과 중 어디까지 영향을 받는가? (2) 무효화되지 않고 살아남는 것은 무엇인가? 무엇을 근거로 살아남는다고 말하는가? (3) 이 결함을 확정하려면 코드에서 무엇을 확인해야 하는가 - 확인 항목을 구체적으로 나열하라. (4) 오염 검사를 다시 세운다면 무엇을 검사 대상으로 삼아야 하는가. 값 앵커(AUROC 4자리 일치)가 애초에 적절한 검사였는가? (5) 앵커가 재현되지 않는 원인이 'knob이 무효'가 아니라 다른 것일 가능성은 무엇인가.

주의: 보고서 §6의 '러너가 항상 7-branch를 산출한다'는 추론이지 확인된 사실이 아니다. 확인된 것과 추론을 섞지 말라. 문서에 없는 수치를 지어내지 말라.
- **RU 유형**: 진단
- **최대 권한 등급**: T1
- **소집자**: Claude Code (claude-opus-5)
- **좌석**: A1(A, temp 0.2, http://127.0.0.1:8000/v1/chat/completions), R1(R, temp 0.5, http://127.0.0.1:8001/v1/chat/completions), T1(T, temp 0.4, http://127.0.0.1:8003/v1/chat/completions), P1(P, temp 0.7, http://127.0.0.1:8000/v1/chat/completions), S1(S, temp 0.2, http://127.0.0.1:8001/v1/chat/completions)
- **호출**: 9/14 · **소요**: 709.4초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 기권한 좌석

- [Phase 1] `R1` (반증): 본문이 비어 있다 (finish_reason=stop); 추론 토큰이 예산을 소진했을 가능성이 높다

## 편성 근거 (결과 전 기록)

계측 결함의 파급 범위 판정이므로 감사(A)를 첫 좌석에 두고 온도를 최저로 했다. 이 회차의 핵심 위험은 결함을 과대 해석해 과거 결과를 전부 무효라고 선언하거나, 반대로 축소해 승격 판정을 그대로 두는 것이다. 반증(R)이 양쪽을 다 공격하게 한다. 이론(T)은 '값 앵커가 애초에 적절한 검사였는가'라는 설계 질문을 맡는다. 제안(P)은 대체 검사 설계를 내되 온도를 0.7로 낮췄다 - 창의성보다 실행 가능성이 중요하다. 일반화(G)와 실측(M)은 제외: 질문이 내부 계측 정합성이고 GPU는 유휴지만 확인 항목이 정해지기 전에는 실행할 것이 없다.

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: 브랜치 집합 통제를 명시적 branch list 계약으로 교체하고 실행 전 코드 감사를 강제한다
- **바꾸는 것**: runner, `branch_screen.py`, aggregator에서 브랜치 포함 집합을 `ICF_SHAPE_SCREEN_ONLY`나 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`로 결정하지 말고, `BRANCHES_V121_5` 또는 공식 7-branch 집합과 같은 명시적 상수/CLI 인자를 단일 결정점으로 만든다. 가중치는 branch list가 확정된 이후에만 적용하고, 가중치 0이면 해당 branch score가 ensemble input에 존재하지 않음을 코드 단언과 테스트로 고정한다. 실행 전 감사 항목은 다음을 포함해야 한다: run script → aggregator의 branch list 호출 경로, `ICF_FIXED_HEAD_*_WEIGHT` reader와 ensemble 반영 여부, `ICF_SHAPE_SCREEN_ONLY` reader, `BRANCHES_V121_5` 사용 부위, score 파일/manifest의 branch list·코드 해시·fold id, 그리고 5-branch와 7-branch를 명시할 때 ensemble input이 실제로 달라지는 단위 테스트.
- **바꾸는 전제**: 문서와 스크립트 머리말이 env knob으로 브랜치 집합을 통제한다고 놓는 전제를 뒤집는다. 확인된 사실은 `ICF_SHAPE_SCREEN_ONLY`가 비테스트 코드에서 읽히지 않고, `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`를 1.0에서 0.0으로 바꿔도 SMAD4 0.4661, PBRM1 0.5408이 소수 4자리까지 바뀌지 않았다는 것이다.
- **기대하는 차이**: branch set 오류가 실행 결과의 값 불일치 뒤에 드러나지 않고, 실행 전 또는 단위 테스트 단계에서 차단된다. 명시적 5-branch 실행의 SMAD4/PBRM1 최종 값은 정량 예측 불가이며, 다만 현재 7-branch 재생성과 동일한 0.4661/0.5408을 다시 산출하면 branch list 미반영, score 재사용, 또는 aggregator 결함이다. 5-branch 앵커 0.4421/0.5553의 재현 여부는 실행 전 예측할 수 없다.
- **기전**: ensemble 출력은 포함된 branch score와 가중치에 의해 결정되므로, branch list가 집계 경로에 도달하지 않으면 knob 변경과 무관하게 고정 집합 출력이 유지된다. 다만 “러너가 항상 7-branch를 산출한다”는 것은 보고서의 추론이지 확인된 사실이 아니므로, 이 제안은 그 추론을 확인하는 것이 아니라 branch list가 단일 결정점이 되도록 만들어 기전을 코드 경로로 고정한다.
- **실패 조건**: 결함이 env knob이 아니라 score 생성 단계에서 이미 7-branch score가 저장/재사용되거나, aggregator가 branch list를 무시하고 존재하는 score 파일을 합산하는 경우; 과거 실행은 현재와 다른 코드 버전에서 knob이 정상 동작했을 수 있으므로 현재 grep/실행만으로 과거 무효를 확정할 수 없는 경우; 앵커 재현 실패의 원인이 다른 코드, 다른 데이터, 다른 fold, 반올림, 또는 `BRANCHES_V121_5` 상수의 의미 변경이라면 branch list 계약만으로는 해결되지 않는다.

## 제안 2: 오염 검사를 값 앵커 일치에서 입력·경로 불변량 검사 + 재생성 지문으로 재설계한다
- **바꾸는 것**: `PROJECT.md` §3.4와 오염 검사 스크립트를 “SMAD4 0.4421, PBRM1 0.5553 소수 4자리 일치”에서 “입력·경로 불변량 검사”로 바꾼다. 검사 대상은 다음이다: 의도된 base branch list와 실제 ensemble input branch list의 일치, screen 대상 branch score의 ensemble input 부재, weight vector와 branch list의 정합, score 파일·fold·코드 해시 manifest 일치, 집계 설정의 결정성. 기존 4자리 값 앵커는 검증된 명시적 5-branch 재생성 이후에만 보조 regression fingerprint로 남기고, 현재 재현 불가능한 0.4421/0.5553을 pass/fail 기준으로 쓰지 않는다.
- **바꾸는 전제**: “4자리 AUROC 앵커가 일치하면 스크리닝이 앙상블을 건드리지 않은 것”이라는 전제를 뒤집는다. AUROC는 branch set, 데이터, fold, 코드, 수치 정밀도의 합성 신호이며, knob이 무효이면 7-branch 산출물이 5-branch 실행처럼 기록될 수 있다. 따라서 값 앵커 단독은 오염 검사가 아니라 값 회귀 테스트에 불과하다.
- **기대하는 차이**: 오염 검사가 값이 우연히 일치해도 branch set/input이 다르면 실패하도록 바뀐다. 정량 예측 불가. 현재 관측 0.4661/0.5408은 기존 값 앵커와 불일치하지만, 새 검사에서는 단순 “legacy anchor mismatch”가 아니라 “branch set/input provenance 검증 필요”로 보고되고, manifest가 7-branch를 입증하면 5-branch BASE arm으로 간주하지 않는다.
- **기전**: 오염의 정의는 스크리닝 대상 branch가 base ensemble에 영향을 주는 것이므로, ensemble input과 aggregation path의 불변량을 직접 검사하면 값과 무관하게 오염을 판별할 수 있다. 값 앵커는 그 downstream 결과일 뿐, 오염의 necessary/sufficient 조건이 아니다.
- **실패 조건**: screen branch가 ensemble input이 아니라 전처리, 정규화, cache, weight adaptation, fold 매핑 등 지문 대상에 포함되지 않는 경로로 영향을 줄 수 있다. 과거 실행에 manifest/해시가 없어 소급 적용이 불가능하면 새 검사는 미래만 보호하고 과거 판정 유효성을 결정하지 못한다. 또한 branch set이 같아도 수치/모델 버전 차이가 AUROC를 바꿀 수 있어 값 fingerprint를 너무 엄격히 쓰면 오판이 생긴다.

## 제안 3: 과거 판정의 유효성을 산출물 provenance 기준으로 재분류하고, D-042와 게이트 1은 조건부로 보류한다
- **바꾸는 것**: 과거 판정의 유효성을 “문서 기록”이 아니라 “산출물 provenance” 기준으로 재분류한다. 상태는 `verified` / `execution-invalid` / `unverified-hold`로 둔다. D-042에 대해, 7-branch 공식 기준선 0.6227과 재생성 0.6226은 per-fold 정밀 대조와 branch manifest 확인 전까지 `verified`가 아니라 `current-reference-hold`로 두고, 확인되면 현재 공식 기준으로 유지한다. D-042의 5-branch 0.6171 대비 +0.56%p 승격 근거는 5-branch baseline이 현재 코드로 재현되지 않으므로, 명시적 5-branch 재실행 또는 과거 산출물 proof 없이는 `unverified-hold` 또는 `execution-invalid`로 하향한다. 게이트 1 스크리닝에 대해, `SCREEN_ONLY=1`과 오염 검사를 전제로 한 screening 결과는 `branch_screen.py`가 ensemble input에 screen branch를 넣지 않았음을 코드와 산출물로 증명하기 전까지 `unverified-hold`로 둔다. 문서에 기록된 SJ 48/50, SH 44/50, BS 31/50, RM 기각 실적도 해당 screening 실행의 provenance가 검증되지 않으면 같은 방식으로 보류한다.
- **바꾸는 전제**: “`PROJECT.md`에 기록된 승격/게이트 판정은 knob이 정상 작동했을 것이므로 자동 유효하다”는 전제를 뒤집는다. 확인된 현재 knob 무효만으로 과거 실행 전체를 무효라고 단정할 근거는 부족하므로, 산출물이 증명하는 것만 `verified`로 남기고 나머지는 hold/invalid로 분리한다.
- **기대하는 차이**: 무효화 범위가 전체 과거 결과가 아니라 항목별로 좁혀진다. 정량 예측 불가. 지표는 D-042 승격 근거 상태, 게이트 1 screening 상태, 공식 7-branch baseline 상태가 된다. 살아남을 수 있는 것은 다음이다: (1) 확인된 계측 결함 사실, (2) 재생성된 7-branch 산출물과 0.6226/0.6227 일치(정밀 대조 전), (3) 대응 per-fold Δ 설계와 hold-out 금지 규칙, (4) 7-branch 운영 구성 자체는 사용자 결정 D-042로 유지 가능하되 개선 증거는 별도. 근거는 각각 보고서 §6, 보고서 §2, `PROJECT.md` §4와 §3.1이다.
- **기전**: 승격 estimand은 같은 fold에서 후보−기준선 Δ의 평균이므로, 기준선의 branch set이 문서와 다르면 추정 대상이 바뀐다. 게이트 1은 screen-only 조건에서 라벨 무관 상관을 재는 것이므로, screen-only 경로가 의도대로 ensemble exclusion을 보장하지 못하면 입력 자체가 의도한 게이트가 아니다.
- **실패 조건**: 과거 실행의 per-fold score, branch list manifest, 코드 버전이 보존되어 있고 5-branch가 실제로 생성됐음을 증명하면 D-042 승격 근거와 게이트 1 screening은 살아남아야 하며, 이 제안은 과도한 hold가 된다. 반대로 7-branch 재생성의 0.6226이 per-fold 대조에서 0.6227과 불일치하면 현재 공식 기준선 자체가 재설정 대상이다. 또한 결함이 신규 regression이 아니라 앵커 문서의 오타나 다른 fold 집계였다면 과거 판정보다 상수·문서가 수정 대상이다.

### A1 (감사)

## 점검 1: D-042 승격 판정의 무효화 범위
- **확인하려는 것**: 현재 확인된 사실만으로 D-042의 7-branch 채택 수치 자체가 무효화되는지, 아니면 5-branch baseline arm의 provenance에 따라 판별 불가/실행 무효로 갈리는지.
- **문서에서 확인되는가**: 문서에 없음 — PROJECT.md §3.2는 D-042 승격 근거로 5-branch `0.6171` 대비 대응 per-fold `Δ_macro +0.56%p`, CI, sign agreement만 기록한다. 그 paired Δ를 만든 5-branch/7-branch per-fold 예측 파일, branch list, weight, env, git commit, fold hash, 매니페스트는 문서에 없다.
- **샌다면 나타날 증상**: 현재 확인된 사실만으로 무효화되는 것은 D-042의 `+0.56%p` 자체가 아니라, 현재 knob 경로로 5-branch 기저를 생성·검증했다는 절차적 주장이다. 5-branch baseline arm이 실제로 5-branch가 아니거나, 후보 7-branch와 baseline의 fold 대응이 깨지거나, baseline이 현재 무효 knob 경로로 생성됐다면 `+0.56%p`는 §4의 estimand가 아니다. 원시 paired Δ가 없으면 D-042는 “판별 불가”이고, baseline arm이 무효 경로로 확인되면 “실행 무효”다.
- **확인 방법**: (1) D-042/RU-90 시점 git commit에서 `scripts/run_ru90_shape_triple.sh`, `scripts/analysis/branch_screen.py`, aggregation 모듈의 branch set/weight/env 처리를 읽는다. (2) `predictions/`의 5-branch와 7-branch per-fold `.pt` metadata에 branch list, weights, env, commit, fold hash가 있는지 확인한다. (3) 저장된 per-fold AUROC로 §4 공식 `Δ_macro`를 재계산한다. (4) 현재 코드로 `BRANCHES_V121_5`를 직접 지정해 5-branch baseline을 재생성하고 SMAD4·PBRM1·macro를 4자리 이상 대조한다.

## 점검 2: 게이트 ① 스크리닝 결과의 무효화 범위
- **확인하려는 것**: 게이트 ①의 `SCREEN_ONLY=1` 측정과 오염 검사 통과 전제가 현재 결함으로 무효화되는 범위.
- **문서에서 확인되는가**: 문서에 없음 — PROJECT.md §5는 게이트 ①이 `SCREEN_ONLY=1`, 오염 검사 통과, `max |r|`, `eff.rank`를 요구한다고 기록한다. 과거 실적 중 명시된 게이트 ① 탈락은 `RM max |r| = 0.690`뿐이고, `SJ 48/50`, `SH 44/50`, `BS 31/50`은 게이트 ② 실적으로 기록된다. 각 실행의 screen-only 실제 branch set, margin 산출물, 오염 검사 통과 기록, 코드 버전은 문서에 없다.
- **샌다면 나타날 증상**: `SCREEN_ONLY`가 후보/형상 브랜치를 집계 풀에서 실제로 빼지 못했다면, `max |r|`·`eff.rank`는 라벨 무관 직교성이 아니라 앙상블에 섞인 마진의 상관이다. 통과/기각 모두 게이트 ① estimand에 대한 유효 판정이 아니다. `RM` 기각도 “게이트 ① 기각”이라기보다 “screen-only 전제 미확정 상태의 기각”으로 내려앉고, `SH`/`SJ` 채택→D-042 승격 연결도 “판별 불가”가 된다.
- **확인 방법**: `scripts/analysis/branch_screen.py`와 호출 스크립트에서 `ICF_SHAPE_SCREEN_ONLY`, `ICF_FIXED_HEAD_*_WEIGHT`, branch list, margin 생성/집계 mask를 읽는다. 과거 게이트 ① margin 산출물 metadata를 확인한다. 동일 fold에서 base ensemble score가 screen 실행 전후로 변하지 않는지 paired score 비교를 수행한다.

## 점검 3: 무효화되지 않고 살아남는 항목
- **확인하려는 것**: 결함에도 살아남는 관찰·설계·경계와, 현재 무효로 남는 것.
- **문서에서 확인되는가**: 확인됨 — 보고서 §1·§6의 현재 실행 결과(7-branch tag macro `0.6226`, SMAD4 `0.4661`, PBRM1 `0.5408`, weight `1.0`/`0.0` 결정론적 동일은 SMAD4·PBRM1 fold-mean AUROC에 한정, 앵커 미재현), PROJECT.md §3.1의 fold 고정·SEAL 10 유보, §4의 paired Δ 설계는 문서에 확인된다. D-042/RU-97의 구체적 생존 여부는 문서에 없다.
- **샌다면 나타날 증상**: 생존 범위를 넘겨 “공식 5-branch baseline 무결” 또는 “D-042 유효”로 쓰면, 무효 knob 경로로 생성된 arm을 정답 기준으로 다시 사용하게 된다. 7-branch macro `0.6226`을 4자리 정밀 대조 없이 공식 baseline 재확인으로 쓰면 rounding 오차를 기전 입증으로 바꾸게 된다. 현재 상태의 후보 성능 판단은 보고서 §4의 분류에 따라 실행 무효로 두어야 한다.
- **확인 방법**: 보고서의 “확인된 것”과 “추론(미검증)”을 분리한다. PROJECT.md §3.2의 `0.6227`과 보고서 `0.6226`은 per-task 4자리 평균 오차로만 일치한다고 기록한다. 저장된 per-fold 예측으로 4자리 이상 재대조 전까지 “정밀 재현 확인됨”이라 쓰지 않는다. RU-97이 broken path를 사용했는지 `current_status.md`와 RU-97 보고서 원시 산출물에서 확인한다.

## 점검 4: 결함 확정을 위한 코드·산출물 확인 항목
- **확인하려는 것**: 결함 확정을 위한 코드·산출물 확인 항목.
- **문서에서 확인되는가**: 문서에 없음 — grep 결과와 weight 실행 결과는 보고서에 확인되지만, 코드 내부 경로 전체, 매니페스트, fold hash, 전처리·정규화·fit 범위, 과거 코드 버전은 문서에 없다.
- **샌다면 나타날 증상**: knob만 의심하면 branch set 하드코딩, aggregation mask, data/fold drift, anchor transcription error를 놓친다. 매니페스트가 없으면 현재 실행과 과거 앵커를 같은 실행으로 전제하게 된다.
- **확인 방법**: (1) `grep -RIn "ICF_SHAPE_SCREEN_ONLY" --include="*.py" --include="*.sh" .`로 비테스트 코드 읽기를 재확인하고, repo 외부 의존성/생성 코드까지 포함한다. (2) `grep -RIn "ICF_FIXED_HEAD_BS_WEIGHT\|ICF_FIXED_HEAD_SH_WEIGHT\|ICF_FIXED_HEAD_SJ_WEIGHT\|ICF_FIXED_HEAD_CT_WEIGHT" --include="*.py" --include="*.sh" .`로 weight 파싱·적용 위치를 확인한다. (3) `scripts/run_ru90_shape_triple.sh`가 호출하는 runner/aggregation 모듈에서 branch list 상수(`BRANCHES_V121_5`, 공식 7-branch), CLI/config/env 우선순위, CT 제외 메커니즘을 확인한다. (4) weight `0.0`이 branch 제외인지 head scaling인지, `0.0`을 falsy로 처리해 default로 되돌리는 bug가 있는지 확인한다. (5) ensemble probability, margin, AUROC, trimmed mean, `eff.rank` 집계에서 어떤 branch mask가 쓰이는지 확인한다. (6) `scripts/analysis/branch_screen.py`가 어떤 margin을 입력으로 받는지, `SCREEN_ONLY`가 그 margin 생성에 영향을 주는지 확인한다. (7) `git log`/`git blame`으로 D-042/RU-90 시점과 현재 코드 차이를 확인한다. (8) `predictions/pathobench_*_ru90_shape_triple_official50_bf16.pt`와 과거 5-branch 산출물의 metadata(branch list, weights, env, commit, fold hash, data version, library version)를 확인한다. (9) fold 파일 중복·누락·train/test overlap, PathoBench 공식 fold hash를 확인한다. (10) 전처리·정규화·GMM/fit 범위(`max_iter`, sample count, seed)가 공식 baseline과 RU-97에서 동일한지 확인한다. (11) tests가 env/weight 동작을 실제 runner로 검증하는지, mock만인지 확인한다.

## 점검 5: 오염 검사 재설계의 검사 대상
- **확인하려는 것**: 오염 검사를 다시 세운다면 무엇을 검사 대상으로 삼아야 하는지.
- **문서에서 확인되는가**: 문서에 없음 — PROJECT.md §3.4는 현재 5-branch 값 앵커만 고정하고, 재설계된 검사 대상은 없다.
- **샌다면 나타날 증상**: 값만 검사하면 branch set이 바뀌어도 AUROC가 4자리에서 같으면 통과하고, rounding/numerical/fit variance 차이로 실패한다. “오염 없음”을 성능 관측으로 입증하게 된다.
- **확인 방법**: 검사 대상은 (a) base ensemble의 branch set identity, (b) weight vector, (c) aggregation mask, (d) margin/score 생성 경로, (e) fold mapping, (f) config/manifest hash다. 실행 전후 base run과 screen run의 per-fold score를 paired로 비교한다. 결정론적 구성은 bit-level 또는 `max|diff| = 0`을, 학습 파라미터가 있는 구성은 적합 분산과 함께 허용 오차를 사전 고정한다. 현재 공식 구성이 7-branch라면 현재 후보의 base는 7-branch base를, 과거 5-branch 대조만 5-branch base를 사용한다.

## 점검 6: AUROC 4자리 값 앵커의 적절성
- **확인하려는 것**: 값 앵커(AUROC 4자리 일치)가 애초에 적절한 검사였는지.
- **문서에서 확인되는가**: 확인됨 — PROJECT.md §3.4가 “소수 4자리까지 일치”와 SMAD4 `0.4421`, PBRM1 `0.5553`을 고정한다. 적절성 판단 자체는 문서에 없다.
- **샌다면 나타날 증상**: 값 앵커를 단독 기준으로 쓰면, (1) 오염이 있어도 AUROC가 4자리에서 변하지 않으면 통과하고, (2) branch set은 같아도 rounding/numerical/fit variance로 실패하며, (3) 5-branch 앵커가 현재 코드와 다른 집계·데이터 버전에서 나왔는지 판별 불가하다. 값 앵커는 과거 실행의 출력을 정답으로 삼는 회귀 테스트이지, 오염 기전의 입증이 아니다.
- **확인 방법**: 값 앵커는 기전 검사가 아니라 회귀 센티널로만 사용한다. 통과 조건을 branch set/aggregation invariant + paired score invariance + 값 회귀로 다층화한다. D-048 이후 학습 파라미터가 허용되므로, 4자리 exact match를 학습 구성의 오염 판정 기준으로 쓰지 않는다.

## 점검 7: 앵커 미재현의 대체 원인
- **확인하려는 것**: 앵커가 재현되지 않는 원인이 “knob 무효”가 아니라 다른 것일 가능성.
- **문서에서 확인되는가**: 문서에 없음 — 보고서 §3은 stale header와 env 미제거 두 가지를 가능 원인으로만 두고 기전 미확인이라 기록한다. 다른 원인은 문서에 없다.
- **샌다면 나타날 증상**: “knob 무효”로 확정하면 code drift, data/fold drift, aggregation change, anchor transcription error를 놓쳐 잘못된 코드를 고치거나 잘못된 baseline을 재생성한다.
- **확인 방법**: (a) `git diff`로 anchor 생성 시점과 현재 코드의 branch definition, preprocessing, normalization, AUROC computation, trimmed mean, fold mapping 차이를 확인한다. (b) fold 파일 hash와 slide feature/data version을 확인한다. (c) background GMM/fit 범위(`max_iter`, sample count, seed)와 library version을 확인한다. (d) PROJECT.md §3.4 상수가 다른 run/task/fold/branch set에서 잘못 복사된지 anchor 원출을 확인한다. (e) weight/env 파싱 bug(`0.0` falsy, name mismatch, config override)를 확인한다. (f) 현재 runner가 branch set을 하드코딩하는지, env/weight와 무관하게 7-branch를 산출하는지 aggregation 코드에서 확인한다. (g) bf16/GPU/PyTorch/sklearn 버전 차이로 4자리가 이동하는지 동일 환경 재실행으로 확인한다.

## 점검 8: 확인된 사실과 추론의 분리
- **확인하려는 것**: 보고서 §6의 “러너가 항상 7-branch를 산출한다”가 확인된 사실이 아닌지.
- **문서에서 확인되는가**: 확인됨 — 보고서 §6이 “러너가 항상 7-branch 구성을 산출하고 있을 가능성”을 “추론(미검증)”으로 표시하고, 집계 코드 미읽음이라 기록한다.
- **샌다면 나타날 증상**: 추론을 사실로 쓰면, 과거 `SCREEN_ONLY` 실행이 모두 screen-only가 아니었다고 과대 일반화하거나, D-042/게이트 ①을 자동 무효로 선언하게 된다.
- **확인 방법**: aggregation 코드에서 실제 branch set을 출력/assert하는 경로를 확인한다. 동일 입력에서 branch list, weight, mask를 dump하는 디버깅 로그 또는 테스트를 추가한다. 5-branch를 직접 지정해 재생성했을 때 앵커가 재현되는지 확인한다. 재현되면 knob/branch 지정 경로 문제로, 재현되지 않으면 code/data/aggregation drift로 확장한다.

### T1 (이론)

**이론적 병목 N**
- **가정**: 현재 접근(문서 knob으로 5-branch 기저를 재현하고 §3.4 값 앵커로 오염을 검사하며, 그 검사를 통과한 실행을 승격·게이트 판단에 쓰는 것)은 다음을 암묵적으로 참으로 둔다. (1) `ICF_SHAPE_SCREEN_ONLY`와 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`가 비테스트 코드에서 실제로 읽혀 브랜치 집합·가중치를 결정한다. (2) 따라서 `SCREEN_ONLY` 또는 해당 가중치 0 실행은 `BRANCHES_V121_5` 5-branch 기저 arm을 산출하며, 그 출력을 §3.4의 5-branch 앵커와 비교할 수 있다. (3) §3.4의 SMAD4 `0.4421`, PBRM1 `0.5553`은 현재 코드·데이터·fold·정규화·반올림 경로에서 재현 가능한 동일 estimand의 값이다. (4) 소수 4자리 fold-mean AUROC 일치/불일치는 스크리닝이 앙상블에 영향을 주지 않았는지를 판별하는 데 충분한 관측이며, 검사 대상은 실제 경로/브랜치 집합이 아니라 값이다. (5) D-042 승격과 게이트 1 스크리닝에 쓰인 값들도 같은 knob/브랜치 라벨이 실제 실행 경로와 일치했다는 전제 위에 있다.
- **이 가정이 깨지는 조건**: 비테스트 코드에서 `ICF_SHAPE_SCREEN_ONLY`를 읽지 않거나, `ICF_FIXED_HEAD_*_WEIGHT`가 실제 브랜치 가중치·집계 풀에 연결되지 않거나, 러너/집계 코드가 문서 라벨과 다른 브랜치 상수를 고정 사용하거나, 과거 앵커가 현재 코드 버전·데이터·fold·정규화·반올림 경로와 다른 estimand에서 생성된 경우다. 또한 4자리 fold-mean AUROC가 브랜치 집합·경로 차이를 판별할 수 없는 해상도라면 값 앵커는 경로 무오염의 증거가 아니다. 과거 D-042·게이트 1의 무효화는 해당 실행의 실제 브랜치 집합·저장 예측·knob 경로가 검증되기 전에는 판별 불가다.
- **깨질 때 나타나는 관측 가능한 증상**: knob 변경에도 출력이 결정론적으로 동일해지고, `SCREEN_ONLY`가 비테스트 코드에서 미읽힘으로 확인되며, 문서 5-branch 앵커가 현재 실행과 4자리 불일치하고, 재생성 macro가 5-branch `0.6171`과 불일치해 7-branch `0.6227`에 가까워진다. 이 상태에서는 오염 검사 통과/실패를 쓸 수 없고, 불일치의 원인이 knob 무효·코드 경로 오류·앵커 estimand 불일치 중 어느 것인지 판별 불가하다. 단, “러너가 항상 7-branch를 산출한다”는 이 증상들로부터의 추론이지 확인된 사실이 아니다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: (1) `ICF_SHAPE_SCREEN_ONLY`는 비테스트 코드 어디에서도 읽히지 않는다(grep 확인). (2) `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`를 1.0에서 0.0으로 바꿔도 SMAD4 `0.4661`, PBRM1 `0.5408`로 소수 4자리까지 동일하고 두 실행은 결정론적으로 같다. (3) §3.4의 5-branch 앵커 SMAD4 `0.4421`, PBRM1 `0.5553`은 현재 코드로 재현되지 않는다. (4) 재생성 macro `0.6226`은 5-branch `0.6171`보다 7-branch `0.6227`에 가깝다. 단, `0.6226`이 7-branch 산출물이라는 것은 수치 일치에 기반한 추론이지 코드 경로 확인은 아니다. (5) 스크립트 머리말은 `ICF_SHAPE_SCREEN_ONLY=1`이 BS/SH/SJ를 모든 집계 풀에서 제외해 공식 5-branch BASE arm을 만든다고 적지만, 관측은 5-branch 앵커와 불일치한다. (6) 과거 D-042·게이트 1 실행이 같은 knob 결함을 가졌다는 직접 충돌 관측은 문서에 없음.

---

## Phase 2 — 교차

### A1 (감사)

## 점검 1: 제안 1 — branch list 단일 결정점이 score 생성·재사용·캐시를 통제하는가
- **확인하려는 것**: 명시적 branch list가 aggregator 입력뿐 아니라 score 생성, score 파일 로딩, 캐시 키, 파일명 매핑까지 통제하는지. 5-branch를 명시해도 이미 저장된 7-branch score를 재사용하면 ensemble input만 바뀌지 실제 산출물은 바뀌지 않는다.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: branch list를 `BRANCHES_V121_5`로 명시해도 `predictions/pathobench_*_ru90_shape_triple_official50_bf16.pt` 같은 기존 score 파일을 그대로 로드하거나, 캐시가 branch list를 무시해 SMAD4 `0.4661`, PBRM1 `0.5408`이 고정된다. score 파일 해시·branch list·fold id가 서로 불일치한다.
- **확인 방법**: `scripts/run_ru90_shape_triple.sh`, aggregator, score 로더를 읽고 `grep -RIn "BRANCHES_V121_5\|branch_list\|branches\|load\|cache" --include="*.py" .`로 호출 경로를 추적한다. `sha256sum predictions/pathobench_*_ru90_shape_triple_official50_bf16.pt`로 파일 지문을 확인하고, 5-branch/7-branch에 서로 다른 dummy score를 주입해 ensemble output이 실제로 달라지는지 단위 테스트로 확인한다.

## 점검 2: 제안 1 — weight 0이 ensemble input 부재를 보장하는가
- **확인하려는 것**: `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0`이 해당 branch score를 ensemble input에서 제거하는지, 아니면 input에는 존재하지만 기여만 0으로 처리되는지, 아니면 weight reader 자체가 없는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: weight 0 branch score가 input tensor에 남아 normalization, rank, cache, 로그에 영향을 줄 수 있다. 또는 weight가 읽히지 않아 1.0/0.0 실행이 소수 4자리까지 동일해진다. “가중치 0이면 부재”라는 단언이 코드 경로로 입증되지 않는다.
- **확인 방법**: `grep -RIn "ICF_FIXED_HEAD" --include="*.py" --include="*.sh" .`로 reader를 찾고, aggregation 함수에서 weight가 score 합산 전에 input을 필터링하는지 확인한다. nonzero dummy score를 두고 weight 0일 때 ensemble input에 해당 branch가 없는지, 또는 기여가 정확히 0인지 `pytest -q`로 검증한다.

## 점검 3: 제안 1 — 현재 코드 감사만으로 과거 실행 무효를 확정할 수 없는가
- **확인하려는 것**: 현재 코드에서 knob이 무효라는 사실이 과거 D-042, gate 1, gate 2 실행도 동일하게 무효였음을 증명하는지. 과거 실행은 다른 코드 버전, 다른 branch list, 다른 score 파일에서 수행됐을 수 있다.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 현재 grep 결과만으로 과거 전체를 `execution-invalid`로 분류하거나, 반대로 과거 기록만 믿어 `verified`로 남긴다. 코드 해시, run metadata, score provenance가 없으면 무효화 범위가 과대 또는 과소 결정된다.
- **확인 방법**: 과거 실행의 코드 버전(`git rev-parse HEAD`, `git log -1`, run log 내 commit hash), score 파일 생성 시각, branch list manifest, fold id, 데이터 버전을 찾는다. 해당 산출물이 현재 코드와 동일한지 대조하고, 없으면 `unverified-hold` 또는 `판별 불가`로만 둔다.

## 점검 4: 제안 2 — input·경로 불변량 검사가 전처리·정규화·fit 범위 누출을 잡는가
- **확인하려는 것**: screen branch가 ensemble input에 없더라도 전처리, 정규화, 통계량 fit, 라벨 접근, fold mask를 통해 base 산출물에 영향을 줄 수 있는지를 검사하는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: branch list와 ensemble input은 동일하지만, margin/score가 train fold가 아닌 전체 fold 또는 테스트 라벨을 보고 fit되어 gate ①의 라벨 무관성이 깨진다. `LSAK` 사양이 문서에 없어 fit 범위, 차원, bandwidth, 정규화, margin 환원이 검증되지 않는다.
- **확인 방법**: `scripts/analysis/branch_screen.py`와 전처리 모듈에서 `grep -RIn "fit\|transform\|normalize\|whiten\|label\|target\|fold" --include="*.py" .`로 fit 범위와 라벨 접근을 확인한다. fold별 train/test mask가 score 생성 전에 고정되는지, label permutation 실행에서 screen margin이 변하지 않는지 확인한다.

## 점검 5: 제안 2 — 값 앵커를 보조 fingerprint로 남기는 기준이 독립 검증에 기반하는가
- **확인하려는 것**: 기존 `0.4421`/`0.5553` 또는 새 fingerprint를 사용할 때, 현재 결함 코드 산출물을 기준으로 삼지 않고 독립적으로 검증된 5-branch 재생성에만 의존하는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 현재 코드가 재현하지 못하는 앵커를 그대로 pass/fail 기준으로 쓰거나, 반대로 현재 결함 코드의 `0.4661`/`0.5408`을 새 fingerprint로 고정해 결함을 정상값으로 만든다. 값 fingerprint가 오염 검사가 아니라 회귀 테스트로 전락한다.
- **확인 방법**: 과거 5-branch per-fold score, 코드 해시, 데이터 버전, fold id를 찾는다. 독립 구현 또는 보존된 산출물로 SMAD4/PBRM1 fold-mean AUROC를 소수 4자리 재계산한다. 해당 증거가 없으면 값 앵커를 사용하지 않고 `판별 불가`로 분류한다.

## 점검 6: 제안 2 — 오염 정의가 shared state·cache·seed·GPU 부수 효과를 포함하는가
- **확인하려는 것**: screen-only 실행이 ensemble input을 건드리지 않더라도 cache, random state, 모델 가중치, 파일 부수 효과, GPU 상태, fold 순서를 통해 base 출력을 바꿀 수 있는지를 검사하는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: branch list와 ensemble input은 동일하지만, screen 실행 전후 base per-fold 예측이 달라진다. 또는 결정론적 출력 때문에 오염이 숨겨진다. “ensemble input 불변”만으로 오염 없음으로 결론짓는다.
- **확인 방법**: 격리된 임시 디렉터리에서 base 실행, screen 실행, base 재실행을 같은 seed·fold·데이터로 수행하고 per-fold 예측 해시를 비교한다. cache 디렉터리, `torch.load`, `load_state_dict`, 파일 쓰기, seed 초기화 코드를 점검한다.

## 점검 7: 제안 3 — D-042 승격 근거의 baseline provenance가 확인되는가
- **확인하려는 것**: D-042의 대응 per-fold Δ_macro `+0.56%p`가 실제로 5-branch baseline과 7-branch 후보를 같은 fold에서 짝지어 계산한 것인지. baseline이 5-branch가 아니었다면 estimand 자체가 바뀐다.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 5-branch baseline이 실제로는 7-branch score, 다른 fold, 다른 코드 버전, 다른 집계 설정이었다면 `+0.56%p`는 공식 승격 기준 `+0.3%p`와 비교할 수 없는 값이 된다. D-042 승격 근거는 `execution-invalid` 또는 `판별 불가`다.
- **확인 방법**: `v121_baseline` 또는 5-branch baseline의 per-fold AUROC/score, branch list, fold id, 코드 해시, 집계 로그를 찾는다. 7-branch 후보와 같은 fold에서 Δ를 재계산해 기록된 `+0.56%p`와 대조한다. 산출물이 없으면 승격 근거를 `verified`로 남기지 않는다.

## 점검 8: 제안 3 — gate ①/② screening provenance가 확인되는가
- **확인하려는 것**: `branch_screen.py`가 screen branch를 ensemble에서 실제로 제외했는지, gate ① margin이 라벨 무관하게 생성됐는지, gate ② 단독 AUROC score도 동일한 결함 경로에서 나오지 않았는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: `SCREEN_ONLY=1`이 읽히지 않으면서도 screen branch가 ensemble에 들어가거나, 라벨로 학습한 margin이 gate ①에 들어가면 `SJ 48/50`, `SH 44/50`, `BS 31/50`, `RM` max |r| `0.690` 같은 실적이 의도한 게이트가 아니다. gate ② 결과도 score 생성 경로 결함에 의해 오염될 수 있다.
- **확인 방법**: `scripts/analysis/branch_screen.py`, margin 산출 파일, run log, 라벨 접근 코드, gate ② score 파일을 점검한다. `grep -RIn "SCREEN_ONLY\|screen\|adopted\|label\|auroc" --include="*.py" .`로 경로를 확인하고, 명시적 exclusion 재실행 결과가 과거 margin과 일치하는지 비교한다.

## 점검 9: 제안 3 — 7-branch macro 일치만으로는 생존 근거가 아닌가
- **확인하려는 것**: 재생성 macro `0.6226`과 공식 기준선 `0.6227`의 일치, 특히 차이 `0.0001`이 per-fold 정확성을 증명하는지. 보고서 §2는 per-fold 정밀 대조를 미수행이라고 명시한다.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: macro는 일치하지만 per-fold AUROC, fold 순서, task 매핑, 집계 방식이 다르면 공식 7-branch baseline `0.6227`이 검증된 것이 아니다. macro 일치만 “생존” 근거로 쓰면 증거 범위를 넘는다.
- **확인 방법**: `predictions/pathobench_*_ru90_shape_triple_official50_bf16.pt`를 로드해 과제·fold별 AUROC를 재계산한다. 공식 7-branch per-fold 값이 보존돼 있으면 fold id별로 대조하고, 없으면 macro 일치만 `current-reference-hold` 또는 `판별 불가`로 둔다.

## 점검 10: 제안 3 — provenance 상태 분류의 증거 기준이 명시적인가
- **확인하려는 것**: `verified` / `execution-invalid` / `unverified-hold` 분류가 macro, 로그 문장, 문서 기록이 아니라 branch list, 코드 해시, 데이터·fold 해시, per-fold score, 집계 설정, run 순서 같은 증거에 기반하는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 문서에 기록된 수치만 보고 `verified`로 남기거나, manifest가 없다는 사실만으로 `execution-invalid`로 확정한다. 반대로 산출물이 존재해도 branch list와 코드 버전이 불일치하면 `verified`가 될 수 없다.
- **확인 방법**: D-042, gate ①/②, baseline 재생성 각각에 대해 branch list, 코드 해시, fold id, 데이터 버전, score 파일 해시, 집계 설정, run log를 체크리스트로 대조한다. 항목별 증거가 부족하면 `unverified-hold` 또는 `판별 불가`로만 분류한다.

## 점검 11: 제안 1/2 — 스크립트 머리말과 관측의 모순을 실행 무효로 분류하는가
- **확인하려는 것**: `scripts/run_ru90_shape_triple.sh` 머리말이 `ICF_SHAPE_SCREEN_ONLY=1` 실행을 공식 5-branch BASE arm이라고 주장하지만, 보고서 §3은 macro `0.6226`과 앵커 불일치를 관측했다. 이 모순이 단순 문서 노후가 아니라 5-branch BASE arm 실행 무효로 분류되는지.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: 실행이 5-branch BASE arm으로 기록되고 후보 성능 판단에 사용되지만, 실제 산출물은 7-branch에 가깝다. 오염 검사 대상이 의도와 달라져 후보 실행이 `execution-invalid`가 될 수 있다.
- **확인 방법**: `scripts/run_ru90_shape_triple.sh` 머리말, 보고서 §3 표, `ICF_SHAPE_SCREEN_ONLY` grep 결과, run metadata를 대조한다. 5-branch BASE arm 주장과 관측 macro `0.6226`, SMAD4 `0.4661`, PBRM1 `0.5408`이 충돌하면 해당 실행의 5-branch BASE arm 효력을 `execution-invalid` 또는 `판별 불가`로 분류한다.

## 점검 12: 제안 3 — 살아남는 것의 범위가 설계 규칙과 수치 판정으로 분리되는가
- **확인하려는 것**: 대응 per-fold Δ 설계, Primary 7 선택 집합, SEAL 10 hold-out 금지, `+0.003` 승격 기준 같은 설계 규칙은 knob 결함과 독립적으로 살아남을 수 있으나, 결함 경로에 의존한 수치 판정까지 함께 살아남는 것은 아닌가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 설계 규칙이 유효하다는 이유로 D-042의 `+0.56%p` 수치, gate ①/② 수치, 5-branch 앵커 불일치 상태의 후보 판단까지 자동으로 유효하다고 결론짓는다. 또는 수치 무효를 피하려 임계값 `+0.003`을 사후 변경한다.
- **확인 방법**: `PROJECT.md` §3.1, §4, §5의 규칙과 D-042·gate 실적의 산출물 provenance를 분리 대조한다. 규칙 자체의 고정 시점과 결과 조회 시점을 확인하고, 수치 판정은 개별적으로 `verified` / `execution-invalid` / `unverified-hold` / `판별 불가`로 분류한다.

## 점검 13: 제안 3 — 선택 이력과 hold-out 경계가 provenance에 포함되는가
- **확인하려는 것**: D-042, gate ①/②, baseline 재생성에서 Primary 7만 선택에 사용되고 SEAL 10 또는 테스트 라벨이 선택에 쓰이지 않았는지. gate ①이 성능 조회 전에 수행됐는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: SEAL 10 결과, 테스트 라벨, 후보 성능 조회가 branch 선택이나 승격 판단에 앞서 사용됐다면 Primary 7 내부 선택 편향과 hold-out 경계 위반이 동시에 생긴다. baseline `0.6227`이나 후보 구성이 결과를 보고 조정된 것처럼 기록될 수 있다.
- **확인 방법**: 결정 이력, run 순서, `branch_screen.py` 로그, 후보 등록 시점, 성능 조회 시점, SEAL 10 접근 로그, 라벨 사용 코드를 대조한다. 선택에 사용된 데이터 집합과 평가 집합이 분리되어 있는지, gate ①이 성능 조회 이전에 완료됐는지 확인한다.

### R1 (반증)

## 반론 1
- **공격 대상**: 제안 1의 "브랜치 집합 통제를 명시적 branch list 계약으로 교체하고 실행 전 코드 감사를 강제한다"는 처방, 특히 knob 무효를 '브랜치 집합 통제 결함'으로 일반화하는 전제.
- **더 단순한 설명**: `ICF_SHAPE_SCREEN_ONLY`와 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`는 현재 코드에 연결되지 않은 죽은 knob일 뿐이고, 실제 브랜치 집합은 이미 스크립트/런너의 하드코딩된 상수나 기본값 하나로 결정될 수 있다. 이 설명만으로도 env var 미읽힘, 가중치 0.0 변경에도 SMAD4 0.4661·PBRM1 0.5408이 소수 4자리까지 동일, macro 0.6226이 7-branch 0.6227에 가까운 관측을 낳는다. 앵커 0.4421/0.5553 불일치는 5-branch 재현 실패가 아니라 과거 앵커 산출 코드, 데이터, fold 집계, 반올림이 현재와 달라진 stale constant일 수 있다. 따라서 명시적 branch list 계약은 미래 실행을 정리할 수 있지만, 현재 관측의 가장 단순한 설명은 "문서화된 knob이 죽었고 실제 결정점은 따로 있다"이지, 새로운 계약 결함 자체가 아니다.
- **무엇이 참이면 이 주장이 무너지는가**: 코드 감사에서 branch list가 이미 단일 결정점이고, 과거 실행의 per-fold score·branch list·코드 버전이 보존되어 의도한 5-branch/7-branch가 실제로 생성됐음을 증명할 수 있다면 제안 1의 새 계약은 과거 유효성 회복에 불필요하다. 0.4421/0.5553이 현재 코드가 아니라 과거 다른 집계/데이터/반올림에서 나온 값으로 확인되면, 이 값을 재현하도록 강제하는 것은 오히려 잘못된 기준을 고착한다. 또한 결함이 score 생성/재사용 단계에 있으면 aggregator의 branch list 계약만으로는 관측을 설명하지 못한다.
- **이 반론의 한계**: env var가 동적 문자열로 읽히거나 여러 경로에서 branch set을 부분 결정하는 코드가 실제로 존재하고, 과거 산출물에 manifest가 없어 현재 코드만으로 과거 실행을 증명할 수 없다면, 제안 1의 명시적 단일 결정점과 실행 전 감사는 여전히 필요하다.

## 반론 2
- **공격 대상**: 제안 2의 "값 앵커 4자리 일치가 애초에 적절한 오염 검사가 아니라 입력·경로 불변량 + 재생성 지문으로 재설계해야 한다"는 일반화.
- **더 단순한 설명**: 현재 문제는 값 앵커의 개념적 부적절함이 아니라, `SCREEN_ONLY` 경로 자체가 비테스트 코드에서 읽히지 않으므로 '스크리닝이 앙상블에 포함되지 않는 실행'이 존재하지 않을 수 있다는 점이다. 더 단순한 조치는 screen-only라는 이름을 버리고, screening 계산이 ensemble 코드와 공유하지 않는 별도 경로인지 코드 단언으로 확인하는 것이다. 공유 경로가 없으면 오염 검사 대상이 없다. 값 앵커는 pipeline identity가 고정된 뒤의 회귀 테스트로 충분할 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 오염이 일어날 수 있는 경로가 ensemble input 하나뿐이고, "screen branch score가 ensemble input에 없다"는 단언 하나와 결정적 재현 값 1~2개로 과거/현재 실행을 판별할 수 있다면, manifest·코드 해시·fold id·weight vector 등 광범위한 지문 체계는 과잉이다. 값 불일치의 원인이 데이터 버전, fold 매핑, AUROC 계산, 반올림 등 pipeline drift라면, 입력·경로 불변량은 통과하면서 값만 실패할 수 있어 제안 2는 '오염'이 아닌 '회귀'를 오염으로 오분류할 수 있다.
- **이 반론의 한계**: screen branch가 ensemble input이 아니라 cache, 전처리, 정규화, fold 매핑, weight adaptation 등 지문에 포함되지 않는 경로로 영향을 줄 수 있고, 과거 실행에 manifest/해시가 없어 소급 검증이 불가능하다면, 제안 2의 불변량·지문은 미래 실행을 보호하는 최소 장치일 수 있다. 값 앵커 단독은 knob 무효 상태에서 branch set을 판별하지 못하므로 완전히 폐기할 수는 없다.

## 반론 3
- **공격 대상**: 제안 3의 "D-042 승격 근거와 게이트 1 스크리닝을 provenance 검증 전까지 unverified-hold/execution-invalid로 보류한다"는 재분류.
- **더 단순한 설명**: 현재 확인된 사실은 '현재 코드에서 문서화된 knob이 무효이고 현재 실행이 5-branch 앵커를 재현하지 못한다'는 것이지, 과거 D-042나 게이트 1 실행이 의도한 구성으로 실행되지 않았다는 증거는 아니다. 더 단순한 분류는 "과거 기록은 기록대로 유지하되, 현재 결함으로 인해 새 실행의 5-branch BASE arm과 오염 검사만 중단한다"는 것이다. 앵커 불일치가 상수 오타/과거 집계 차이로 확인되면, 무효화 대상은 §3.4 상수이지 D-042 전체가 아니다.
- **무엇이 참이면 이 주장이 무너지는가**: 과거 실행의 per-fold 예측, branch list manifest, 코드 버전, 또는 screening margin 산출 경로가 보존되어 있고, 5-branch baseline과 screen-only exclusion이 실제로 수행됐음을 증명하면 D-042의 +0.56%p 근거와 SJ 48/50·SH 44/50·BS 31/50·RM 기각 실적은 hold에서 풀려야 한다. 특히 게이트 1/2가 ensemble output이 아니라 standalone margin/AUROC에서 계산됐고, screen branch가 그 입력에 영향을 주지 않는 경로라면, 오염 검사 실패만으로 게이트 실적을 무효화하는 것은 과도하다. 0.6226과 0.6227의 차이가 per-task 4자리 평균 반올림으로 설명되고 per-fold raw 대조에서도 공식 7-branch와 일치하면, 7-branch 기준선을 current-reference-hold로 묶는 것도 불필요하다.
- **이 반론의 한계**: 과거 실행의 provenance가 전혀 없고, 같은 죽은 knob이 과거 코드에도 존재했거나 5-branch baseline을 현재/과거 산출물로 재증명할 수 없다면, D-042의 승격 근거와 screen-only 전제 게이트 실적은 실제로 unverified-hold 또는 execution-invalid로 내려야 한다. 7-branch 재생성이 per-fold 대조에서 공식 값과 불일치하면 현재 기준선 자체도 재설정 대상이다.

### T1 (이론)

## 이론적 병목 N
- **가정**: P1·P2·P3은 공통으로 "확인된 knob 무효와 앵커 미재현을 branch set/provenance 통제 결함의 증상으로 해석하고, 명시적 branch list·input manifest·provenance 상태가 estimand를 회복할 수 있다"고 둔다. P1은 knob 무효를 'branch list가 집계 경로에 도달하지 못함'으로, ensemble input이 달라지면 출력이 의도대로 바뀐다고 전제한다. P2는 오염을 'screen branch score가 ensemble input에 존재/부재'로 환원하고, manifest·해시가 신뢰할 수 있는 provenance 증거라고 전제한다. P3은 과거 유효성을 '산출물 provenance가 증명하는 범위'로 환원하고, 0.6226/0.6227의 macro 일치만으로도 current-reference-hold가 가능하다고 전제한다. 이 환원은 확인된 사실(비테스트 코드에서 SCREEN_ONLY 미읽힘, weight 0.0 전환에도 0.4661/0.5408 불변, 0.4421/0.5553 미재현, 0.6226이 0.6227에 맞는 보고서 관측)을 넘어, '현재 산출물이 7-branch 산출물이고, 5-branch 앵커는 같은 코드·데이터·fold에서 재현 가능해야 하며, branch set이 출력을 결정하는 유일한 통제 변수'라는 미확인 기전을 전제한다.
- **이 가정이 깨지는 조건**: (1) 0.4661/0.5408/0.6226이 7-branch 산출물이라는 것이 아니라 저장 score 재사용, cache, 다른 코드 버전, 다른 fold/데이터, 반올림/집계 방식, BRANCHES_V121_5 의미 변경, weight가 다른 경로에 적용되는 등으로 설명될 때; (2) 명시적 branch list를 주어도 ensemble input이 실제로 달라지지 않거나, input이 달라져도 모델 상태·score 파일·cache·집계 코드가 출력을 고정할 때; (3) screen branch가 ensemble input이 아니라 전처리·정규화·cache·weight adaptation·fold 매핑·랜덤 상태·모델 체크포인트 경로로 영향을 줄 때; (4) 과거 실행에 manifest·코드 해시·per-fold score·branch list가 없거나, 과거 코드에서는 knob이 동작했을 때; (5) 0.4421/0.5553이 현재 코드에서 재현되지 않는 원인이 branch set이 아니라 문서 상수 오류, fold 집계 오류, 과거 코드/데이터, 또는 5-branch 정의 변경일 때; (6) gate 1의 유효성이 실제로 screen-only exclusion에 의존하지 않고 branch_screen.py의 마진 기록 경로 자체로 성립할 때.
- **깨질 때 나타나는 관측 가능한 증상**: 명시적 5-branch 지시로 실행해도 SMAD4 0.4661·PBRM1 0.5408·macro 0.6226이 그대로면 branch list 미반영, score 재사용, 또는 집계 고정; 명시적 5-branch가 다른 값을 내지만 0.4421/0.5553이 아니면 앵커의 코드/데이터/fold/반올림 불일치; 코드 감사에서 branch list가 aggregator에 도달하지만 출력 불변이면 cache/모델/집계 경로가 knob을 우회; manifest가 없거나 manifest와 산출물이 불일치하면 provenance 분류는 판별 불가; 과거 산출물에서 5-branch score가 실제로 존재하면 P3의 hold가 과도하고, 존재하지 않으면 과거 screening/promotion 근거는 execution-invalid 또는 unverified-hold로 갈리지만 현재 문서만으로는 판별 불가; 0.6226과 0.6227의 per-fold 정밀 대조가 불일치하면 current-reference-hold 자체가 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: 보고서 §6은 "러너가 항상 7-branch를 산출한다"를 확인된 사실이 아니라 추론으로 명시하고, knob 무효와 앵커 미재현만 확인했다. 제안들이 branch set/provenance 결함을 기전처럼 다루는 것은 이 미확인과 충돌한다. PROJECT.md §3.4는 5-branch 앵커를 공식 7-branch 전환 후에도 그대로 유지한다고 하지만, 현재 코드로 0.4421/0.5553이 재현되지 않는다는 관측은 "값 앵커가 5-branch 경로의 안정 불변량"이라는 가정과 충돌한다. P3의 0.6226/0.6227 일치로 current-reference-hold를 두는 것은 보고서 §2의 "정밀 대조는 저장된 per-fold 예측으로 다시 해야 한다(미수행)"와 충돌한다.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

## A1 점검 1
- (1) 수용하고 수정: A1은 올바르다. 문서에는 `BRANCHES_V121_5`나 명시적 branch list가 score 생성, score 파일 로드, 캐시 키, 파일명 매핑까지 통제하는지 나와 있지 않다. 따라서 제안 1은 aggregator 입력만 통제하는 것으로 불충분하다. 수정: branch list 단일 결정점의 범위를 runner → score 생성 → score 저장/로드 → 캐시 키 → manifest → aggregator 전체로 확장한다. 5-branch를 명시해도 기존 7-branch score 파일을 재사용하면 출력이 고정될 수 있으므로, score provenance와 cache key에 branch list가 포함되는지 실행 전 감사와 단위 테스트로 확인해야 한다.

## A1 점검 2
- (1) 수용하고 수정: A1은 올바르다. 문서에는 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0`이 해당 branch score를 ensemble input에서 제거하는지, input에는 두고 기여만 0으로 하는지, 아니면 reader 자체가 없는지 나와 있지 않다. 확인된 사실은 1.0에서 0.0으로 바꿔도 SMAD4 0.4661, PBRM1 0.5408이 소수 4자리까지 동일하다는 것이지, weight 0의 의미까지 증명하는 것이 아니다. 수정: “가중치 0이면 부재”라는 단언을 코드 경로로 입증하기 전에는 쓰지 않는다. reader 존재, score 합산 전 필터링 여부, nonzero dummy score에서 weight 0일 때 input 부재 또는 기여 0을 단위 테스트로 확인한다.

## A1 점검 3
- (1) 수용하고 수정: A1은 올바르다. 현재 코드에서 knob이 무효라는 사실만으로 과거 D-042, 게이트 1, 게이트 2 실행까지 동일하게 무효였다고 확정할 근거가 문서에 없다. 과거 실행은 다른 코드 버전, 다른 branch list, 다른 score 파일, 다른 데이터/fold에서 수행됐을 수 있다. 수정: 현재 grep/실행 결과는 현재 코드의 문서화 knob 무효와 현재 5-branch BASE arm 미확립만 말한다. 과거 실행은 코드 해시, run log, branch list manifest, per-fold score, fold id, 데이터 버전이 없으면 `unverified-hold` 또는 `판별 불가`로만 둔다. `execution-invalid`는 과거 산출물이 의도된 구성이 아님을 증명할 때만 쓴다.

## A1 점검 4
- (1) 수용하고 수정: A1은 올바르다. 제안 2의 input·경로 불변량은 ensemble input에만 집중하면 전처리, 정규화, 통계량 fit, 라벨 접근, fold mask 누출을 놓칠 수 있다. 문서에는 LSAK의 fit 범위, 차원, bandwidth, 정규화, margin 환원이 없어 라벨 무관성을 검증할 근거가 없다. 수정: 오염 검사 범위에 screen branch의 ensemble input 부재뿐 아니라 전처리/정규화 fit 범위, train/test fold mask 고정, label/target 접근 부재, margin 기록 경로의 독립성을 포함한다. 가능하면 label permutation 실행에서 screen margin이 변하지 않는지 확인한다.

## A1 점검 5
- (1) 수용하고 수정: A1은 올바르다. 현재 결함 경로에서 나온 값을 새 fingerprint로 고정하면 결함을 정상값으로 만들 수 있고, 현재 재현되지 않는 0.4421/0.5553을 그대로 pass/fail 기준으로 쓰면 미검증 상수를 강제한다. 문서에는 0.4421/0.5553을 독립적으로 재검증한 산출물이 없다. 수정: 값 앵커는 독립적으로 검증된 명시적 5-branch 재생성 또는 보존된 과거 per-fold score가 있을 때만 보조 regression fingerprint로 쓴다. 그런 증거가 없으면 값 앵커는 오염 검사 기준이 아니라 `판별 불가` 또는 문서 정정 대상이다.

## A1 점검 6
- (1) 수용하고 수정: A1은 올바르다. “ensemble input에 screen branch가 없다”는 사실만으로 오염이 없다고 결론짓기 어렵다. cache, random state, 모델 가중치, 파일 부수 효과, GPU 상태, fold 순서 등이 base 출력을 바꿀 수 있다. 문서에는 이러한 shared state 경로가 통제되는지 나와 있지 않다. 수정: 오염 검사는 격리된 임시 디렉터리에서 base 실행, screen 실행, base 재실행을 같은 seed·fold·데이터로 수행하고 per-fold 예측 해시를 비교하는 것을 포함한다. ensemble input 불변은 필요 조건이지 충분 조건이 아니다.

## A1 점검 7
- (1) 수용하고 수정: A1은 올바르다. D-042의 승격 근거 `+0.56%p`가 유효하려면 7-branch 후보와 5-branch baseline이 같은 fold에서 짝지어졌고, baseline이 실제로 5-branch 구성이었음이 증명되어야 한다. 문서에는 `v121_baseline`의 per-fold score, branch list, 코드 해시, fold id, 집계 로그가 없다. 수정: D-042 승격 근거는 현재 `verified`가 아니다. 과거 산출물이 없으면 `unverified-hold` 또는 `판별 불가`로 둔다. baseline이 5-branch가 아님을 산출물로 증명해야만 `execution-invalid`로 분류한다.

## A1 점검 8
- (1) 수용하고 수정: A1은 올바르다. 게이트 ①/② 실적은 `branch_screen.py`가 screen branch를 ensemble에서 실제로 제외했고, margin/AUROC 산출 경로가 의도한 standalone 경로였음을 전제로 한다. 현재 코드에서 `SCREEN_ONLY`가 읽히지 않는다는 사실은 확인됐지만, 과거 실행의 screen 경로가 동일하게 무효였는지는 문서에 없다. 수정: SJ 48/50, SH 44/50, BS 31/50, RM 기각 실적은 screening margin 산출 경로, 라벨 접근, ensemble 입력 제외, run log, score 파일 provenance가 검증되기 전까지 `unverified-hold` 또는 `판별 불가`로 둔다. standalone 경로가 증명되고 라벨 누출이 없으면 생존할 수 있다.

## A1 점검 9
- (1) 수용하고 수정: A1은 올바르다. 보고서 §2는 재생성 macro 0.6226과 공식 7-branch 0.6227의 차이가 per-task 4자리 반올림 평균에서 나오는 오차 범위 안이라고 했지만, per-fold 정밀 대조는 미수행이다. 따라서 macro 일치만으로는 공식 7-branch baseline이 검증됐다고 말할 수 없다. 수정: 0.6226/0.6227 일치는 `verified` 근거가 아니라 `current-reference-hold` 또는 `판별 불가` 수준이다. per-fold raw AUROC, fold 순서, task 매핑, 집계 방식 대조가 완료되기 전에는 공식 기준선 생존 근거로 쓰지 않는다.

## A1 점검 10
- (1) 수용하고 수정: A1은 올바르다. `verified` / `execution-invalid` / `unverified-hold` 분류가 문서 기록이나 macro 일치만으로 이루어지면 과대·과소 분류가 생긴다. 수정: 상태별 증거 기준을 명시한다. `verified`는 branch list, 코드 해시, 데이터/fold 해시, per-fold score, 집계 설정, run log가 일치할 때만 쓴다. `execution-invalid`는 의도된 branch set, 입력, 절차가 실제로 사용되지 않았음을 산출물로 증명할 때만 쓴다. 증거가 부족하면 `unverified-hold` 또는 `판별 불가`로 둔다.

## A1 점검 11
- (1) 수용하고 수정: A1은 올바르다. 스크립트 머리말은 `ICF_SHAPE_SCREEN_ONLY=1` 실행을 공식 5-branch BASE arm이라고 주장하지만, 관측은 macro 0.6226, SMAD4 0.4661, PBRM1 0.5408로 문서의 5-branch 앵커와 충돌한다. 수정: 이 실행은 “5-branch BASE arm” 또는 “오염 검사 통과 실행”으로는 `실행 무효`로 분류한다. 다만 “러너가 항상 7-branch를 산출한다”는 추론이지 확인된 사실이 아니므로, 이 실행이 실제로 7-branch 산출물인지는 per-fold 대조와 branch manifest 확인 전까지 `판별 불가`로 둔다.

## A1 점검 12
- (1) 수용하고 수정: A1은 올바르다. 설계 규칙과 수치 판정은 분리해야 한다. Primary 7 선택 집합, SEAL 10 hold-out 금지, fold 고정, 대응 per-fold Δ estimand, `+0.003` 승격 기준은 knob 결함과 독립적으로 살아남을 수 있다. 그러나 D-042의 `+0.56%p`, 게이트 ①/② 수치, 5-branch 앵커, 현재 재생성 0.6226은 결함 경로에 의존한 수치 판정이므로 자동으로 유효하지 않다. 수정: 규칙의 생존과 수치 판정의 생존을 별도 항목으로 분류한다.

## A1 점검 13
- (1) 수용하고 수정: A1은 올바르다. provenance에는 산출물 정체성뿐 아니라 선택 이력과 hold-out 경계가 포함되어야 한다. D-042, 게이트 ①/②, baseline 재생성에서 Primary 7만 선택에 사용되고 SEAL 10 또는 테스트 라벨이 선택에 쓰이지 않았는지, 게이트 ①이 성능 조회 전에 수행됐는지는 문서에 없다. 수정: provenance 체크리스트에 데이터 집합 접근 순서, 라벨 사용 코드, 후보 등록 시점, 성능 조회 시점, SEAL 10 접근 여부, run 순서를 포함한다. 증거가 없으면 선택 편향 또는 hold-out 경계 위반을 배제할 수 없어 `판별 불가`로 둔다.

## R1 반론 1
- (1) 수용하고 수정: R1은 올바르다. 확인된 사실은 문서화된 env knob이 비테스트 코드에서 읽히지 않고, weight 1.0→0.0 전환에 SMAD4 0.4661, PBRM1 0.5408이 소수 4자리까지 변하지 않는다는 것이다. 이는 “branch list가 집계 경로에 도달하지 않는다”는 기전을 직접 증명하지 않는다. 실제 결정점은 하드코딩 상수, score 재사용, cache, stale anchor일 수 있다. 수정: 제안 1은 현재 관측의 기전 설명으로 쓰지 않는다. 먼저 코드 감사로 실제 branch set 결정점을 찾아내고, 그 결정점이 단일화되지 않았다면 명시적 branch list 계약으로 교체한다. 과거 산출물이 의도된 구성을 증명하면 제안 1은 과거 무효화 장치가 아니라 미래 통제 강화로 한정된다.

## R1 반론 2
- (1) 수용하고 수정: R1은 올바르다. 현재 가장 직접적인 문제는 `SCREEN_ONLY` 경로가 비테스트 코드에서 읽히지 않으므로 “screen-only 실행” 자체가 의도대로 존재하지 않을 수 있다는 점이다. 값 앵커 단독은 knob 무효 상태에서 branch set을 판별하지 못하지만, 오염이 ensemble input 하나뿐인 경로로만 가능하다는 증거도 문서에 없다. 수정: 제안 2를 “오염 = 값 불일치”로 쓰지 않는다. 1차 오염 검사는 screen branch가 ensemble input과 shared state에 영향을 주지 않는 코드 경로 증명과 격리 재현이다. 값 앵커는 pipeline identity가 검증된 뒤의 regression fingerprint로만 남기고, input·경로 불변량이 통과해도 값만 실패하면 `regression` 또는 `판별 불가`로 분류한다.

## R1 반론 3
- (1) 수용하고 수정: R1은 올바르다. 현재 확인된 사실은 현재 코드의 knob 무효와 현재 5-branch 앵커 미재현이지, 과거 D-042나 게이트 1 실행이 의도한 구성으로 실행되지 않았다는 증거가 아니다. 수정: 제안 3은 현재 결함만으로 과거 판정을 일괄 보류하지 않는다. 항목별 의존성에 따라 분류한다. D-042는 5-branch baseline provenance에 의존한다. 게이트 ①은 문서상 `SCREEN_ONLY=1`과 오염 검사 절차를 요건으로 하므로 그 절차가 실제로 충족됐는지 검증해야 한다. 게이트 ②는 standalone AUROC 산출 경로에 의존한다. 과거 산출물이 의도된 실행을 증명하면 hold를 해제한다. 증거가 없으면 `unverified-hold` 또는 `판별 불가`로만 둔다.

## T1 이론적 병목 N
- (1) 수용하고 수정: T1은 올바르다. 제안 1~3은 공통으로 “현재 산출물이 7-branch 산출물이다”, “branch set이 출력을 결정하는 유일한 통제 변수다”, “0.4421/0.5553은 같은 코드·데이터·fold에서 재현 가능해야 한다”는 미확인 기전을 전제로 할 위험이 있다. 보고서 §6은 “러너가 항상 7-branch를 산출한다”를 확인된 사실이 아니라 추론으로 명시한다. 수정: 제안 1~3 모두에서 확인된 사실과 미확인 기전을 분리한다. 확인된 사실은 비테스트 코드에서 `ICF_SHAPE_SCREEN_ONLY` 미읽힘, `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT` 1.0→0.0 전환에도 SMAD4 0.4661·PBRM1 0.5408 소수 4자리 불변, 0.4421/0.5553 현재 코드 미재현, macro 0.6226과 0.6227의 보고된 일치, per-fold 정밀 대조 미수행으로 한정한다. 제안은 기전 입증이 아니라 판별 실험, 통제 교체, provenance 재분류로만 서술한다. 0.6226/0.6227 일치는 `current-reference-hold` 또는 `판별 불가`를 넘지 못한다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)
- **현재 계측 경로에 결함이 있다.** 보고서 §6에서 확인된 사실은 (1) `ICF_SHAPE_SCREEN_ONLY`가 비테스트 코드 어디에서도 읽히지 않는다는 grep 결과, (2) `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`를 1.0에서 0.0으로 바꿔도 SMAD4 `0.4661`, PBRM1 `0.5408`이 소수 4자리까지 결정론적으로 동일하다는 실행 결과, (3) 문서의 5-branch 앵커 SMAD4 `0.4421`, PBRM1 `0.5553`이 현재 코드로 재현되지 않는다는 사실이다. 이 세 관측은 문서가 제공한 knob 경로가 현재 코드에서 의도대로 동작하지 않음을 뒷받침한다.
- **현재 실행은 문서가 주장한 5-branch BASE arm으로 사용할 수 없다.** `scripts/run_ru90_shape_triple.sh` 머리말은 `ICF_SHAPE_SCREEN_ONLY=1`이 BS/SH/SJ를 집계 풀에서 제외해 공식 5-branch BASE arm을 만든다고 적지만, 보고서 §3의 관측은 macro `0.6226`, SMAD4 `0.4661`, PBRM1 `0.5408`으로 5-branch 앵커와 불일치한다. 또한 weight 1.0→0.0 전환에도 출력이 동일하므로, 이 실행이 5-branch 기저를 만들었다고 보는 주장은 현재 증거와 충돌한다.
- **`PROJECT.md` §3.4의 오염 검사 절차는 현재 수행 불가하다.** §3.4는 SMAD4 `0.4421`, PBRM1 `0.5553`의 소수 4자리 일치를 오염 검사 기준으로 고정한다. 그러나 현재 코드로 그 값이 재현되지 않고, 그 값을 생성할 knob 경로도 무효로 확인됐다. 따라서 현재 상태에서는 “오염 검사 통과”도 “오염 검사 실패”도 쓸 수 없다.
- **D-042 승격 근거의 5-branch baseline arm은 현재 검증되지 않는다.** `PROJECT.md` §3.2는 D-042 승격 근거로 5-branch `0.6171` 대비 대응 per-fold `Δ_macro +0.56%p`를 기록한다. 그러나 그 5-branch baseline의 per-fold score, branch list, 코드 해시, fold id, 데이터 버전, 집계 로그는 문서에 없고, 현재 코드는 5-branch 앵커를 재현하지 못한다. 따라서 D-042의 `+0.56%p`가 §4의 estimand에 부합하는지 현재 증거만으로는 `verified`라고 말할 수 없다.
- **게이트 ① 스크리닝 결과는 현재 검증되지 않는다.** `PROJECT.md` §5는 게이트 ①이 `SCREEN_ONLY=1` 측정과 §3.4 오염 검사 통과를 요건으로 한다. 현재 코드에서 `ICF_SHAPE_SCREEN_ONLY`가 읽히지 않고, 오염 검사 경로도 수행 불가하므로, 게이트 ①에 의존한 결과(예: `RM` max |r| `0.690` 기각)는 현재 증거만으로 `verified`라고 말할 수 없다.
- **설계 규칙 자체는 계측 결함과 독립적으로 살아남을 수 있다.** Primary 7만 선택에 사용하고 SEAL 10을 hold-out으로 유보하는 규칙, fold 고정, 대응 per-fold Δ estimand, `+0.003` 승격 기준, `hold-out 미검증` 명시 의무는 `PROJECT.md` §3.1·§4·§5에 선언된 규칙이다. 이 규칙들의 텍스트가 현재 knob 결함 때문에 자동 무효화된다는 증거는 없다. 다만 과거 실행이 실제로 그 규칙을 지켰는지는 별개의 provenance 문제다.
- **현재 7-branch 재생성은 `current-reference-hold` 수준으로만 살아남는다.** 보고서 §2는 재생성 macro `0.6226`이 공식 7-branch `0.6227`과 `0.0001` 차이로, per-task 4자리 반올림 평균 오차 범위 안에 있다고 기록한다. 그러나 per-fold 정밀 대조는 미수행이다. 따라서 이 재생성은 공식 7-branch baseline이 검증됐다는 증거가 아니라, 현재 기준선과 정밀 대조가 필요한 현재 산출물이다.
- **값 앵커 단독은 오염 검사의 충분 조건이 아니다.** AUROC fold-mean은 branch set, 데이터, fold, 코드, 수치 정밀도, 집계 방식의 합성 신호다. knob이 무효이면 branch set이 의도와 다르면 AUROC가 4자리에서 우연히 같을 수 있고, branch set이 같아도 코드/데이터/반올림 차이로 다를 수 있다. 현재 5-branch 앵커가 재현되지 않는다는 사실은 값 앵커가 경로 무오염을 직접 증명하지 못함을 보여준다.

## 반박 (증거가 반박함)
- **“현재 실행이 5-branch BASE arm이다”는 주장은 반박된다.** 보고서 §3은 macro `0.6226`, SMAD4 `0.4661`, PBRM1 `0.5408`이 5-branch 앵커 `0.6171`, `0.4421`, `0.5553`과 불일치한다고 기록한다. 보고서 §6은 weight 1.0→0.0에도 SMAD4·PBRM1이 소수 4자리까지 동일하다고 확인한다. 이 관측들은 스크립트 머리말의 5-branch BASE arm 주장과 충돌한다.
- **“문서화된 knob으로 현재 코드에서 브랜치 집합을 바꿀 수 있다”는 주장은 반박된다.** `ICF_SHAPE_SCREEN_ONLY`는 비테스트 코드에서 읽히지 않고, `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT` 1.0→0.0 전환에도 SMAD4·PBRM1 출력이 소수 4자리까지 동일하다. 따라서 이 knob들이 현재 실행의 branch set을 통제한다는 주장은 현재 증거에 반한다.
- **“`PROJECT.md` 기록만으로 D-042 승격 근거가 verified다”는 주장은 반박된다.** D-042 승격 근거는 5-branch baseline과 7-branch 후보의 대응 per-fold Δ에 의존한다. 그러나 5-branch baseline의 per-fold score, branch list, 코드 버전, fold id, 데이터 버전, 집계 로그는 문서에 없고, 현재 코드는 5-branch 앵커를 재현하지 못한다. 기록된 수치만으로는 estimand의 무결성을 증명하지 못한다.
- **“게이트 ① 실적이 기록만으로 verified다”는 주장은 반박된다.** 게이트 ①은 `SCREEN_ONLY=1`과 오염 검사 통과를 요건으로 한다. 현재 코드에서 `SCREEN_ONLY`가 읽히지 않고, 오염 검사 경로도 수행 불가하다. 따라서 게이트 ① 결과의 유효성은 기록된 수치만으로 확립되지 않는다.
- **“macro `0.6226`과 `0.6227`의 일치로 공식 7-branch baseline이 verified다”는 주장은 반박된다.** 보고서 §2는 차이 `0.0001`이 per-task 4자리 반올림 평균에서 나올 수 있다고 하고, per-fold 정밀 대조는 미수행이라고 명시한다. macro 일치만으로는 fold 순서, task 매핑, per-fold raw AUROC, 집계 방식의 동일성을 증명하지 못한다.
- **“러너가 항상 7-branch를 산출한다”는 주장은 현재 확인된 사실이 아니다.** 보고서 §6은 이 문장을 “추론(미검증)”으로 표시하고, 집계 코드를 읽어 확인하지 않았다고 명시한다. macro `0.6226`이 7-branch `0.6227`에 맞는 것은 수치 일치 관측이지, branch set 결정 경로의 확인이 아니다.
- **“현재 코드 결함만으로 과거 D-042·게이트 ① 실행 전체가 무효다”는 주장은 반박된다.** 현재 grep과 weight 실행은 현재 코드의 knob 무효와 현재 5-branch 앵커 미재현을 보여준다. 그러나 과거 실행의 코드 버전, branch list, score 파일, fold id, 데이터 버전, run log는 문서에 없다. 현재 결함만으로 과거 실행이 동일하게 결함을 가졌다고 확정할 근거는 없다.
- **“소수 4자리 AUROC 앵커 일치가 오염 검사의 적절하고 충분한 기준이다”는 주장은 반박된다.** §3.4의 값 앵커는 5-branch 기저의 downstream 값이다. knob이 무효이고 branch set 통제가 확인되지 않은 상태에서 값 일치는 오염 부재를 증명하지 못한다. 또한 현재 앵커가 재현되지 않으므로, 이 값을 그대로 pass/fail 기준으로 쓰는 것은 미검증 상수를 강제한다.

## 판별 불가 (증거가 가르지 못함)
- **D-042 승격 근거 `+0.56%p`가 `execution-invalid`인지 `unverified-hold`인지 판별 불가하다.** 판별에는 5-branch baseline의 per-fold score, 7-branch 후보의 per-fold score, 같은 fold 짝짓기, branch list, weight, 코드 해시, fold id, 데이터 버전, 집계 로그가 필요하다. 현재 문서에는 이 provenance가 없다.
- **게이트 ① 결과의 유효성 범위가 어디까지인지 판별 불가하다.** `RM` max |r| `0.690` 기각, 그리고 SH/SJ/BS가 게이트 ①을 통과했는지 여부는 `branch_screen.py`의 margin 생성 경로, `SCREEN_ONLY`의 과거 동작, ensemble exclusion, 라벨 접근, run log, 과거 코드 버전이 확인되어야 한다. 현재 증거는 현재 코드에서 `SCREEN_ONLY`가 읽히지 않는다는 사실만 제공한다.
- **게이트 ② standalone AUROC 실적(SJ 48/50, SH 44/50, BS 31/50)이 이 결함에 의해 영향을 받는지는 판별 불가하다.** 게이트 ②는 standalone AUROC에 의존한다. 그러나 그 standalone score가 같은 broken branch/weight/score 생성 경로를 썼는지, label leakage가 있었는지, fold mask가 의도대로 고정됐는지는 문서에 없다.
- **현재 재생성 `0.6226`이 공식 7-branch `0.6227`과 per-fold 수준에서 동일한지 판별 불가하다.** 보고서 §2는 per-fold 정밀 대조를 미수행이라고 한다. 따라서 현재 산출물이 공식 7-branch baseline의 재현인지, 아니면 macro만 우연히 가까운 다른 산출물인지는 저장된 per-fold 예측 대조 전까지 알 수 없다.
- **5-branch 앵커 `0.4421`/`0.5553`이 재현되지 않는 원인이 knob 무효인지, stale anchor인지, 코드/데이터/fold drift인지 판별 불가하다.** 가능한 원인에는 과거 코드 버전, 다른 fold 집계, 다른 데이터 버전, 반올림/집계 방식 변화, `BRANCHES_V121_5` 의미 변경, score 재사용, cache, 하드코딩된 branch set, weight 파싱 bug, 전처리/정규화/GMM/fit 범위 변화, bf16/라이브러리 버전 차이가 포함된다. 현재 증거는 이들을 분리하지 못한다.
- **과거 `SCREEN_ONLY` 실행이 실제로 screen-only였는지 판별 불가하다.** 현재 코드에서 `SCREEN_ONLY`가 읽히지 않는다는 사실은 과거 코드에서도 읽히지 않았음을 증명하지 못한다. 과거 run의 코드 해시, env, branch list, margin 산출물, ensemble input dump가 필요하다.
- **명시적 branch list 계약이 score 생성·재사용·캐시·aggregation 전체를 통제하는지 판별 불가하다.** 문서에는 `BRANCHES_V121_5`나 branch list가 score 파일 로드, cache key, 파일명 매핑, aggregator 입력까지 통제하는지 나와 있지 않다. 따라서 branch list를 명시해도 기존 7-branch score가 재사용되거나 cache가 우회될 수 있는지 현재 증거만으로는 알 수 없다.
- **RU-97 등 다른 과거 결과가 같은 broken path를 사용했는지 판별 불가하다.** `current_status.md`와 RU-97 보고서는 GF 탐색 결과를 기록하지만, 그 실행이 `ICF_SHAPE_SCREEN_ONLY`나 `ICF_FIXED_HEAD_*_WEIGHT` 경로에 의존했는지, branch list가 무엇이었는지, score provenance가 무엇인지는 문서에 없다.

## 실행 무효 (실행·계측 결함으로 판단 불가)
- **현재 `run_ru90_shape_triple.sh` 실행의 “5-branch BASE arm” 효력은 실행 무효다.** 스크립트 머리말의 5-branch BASE arm 주장과 관측 macro `0.6226`, SMAD4 `0.4661`, PBRM1 `0.5408`이 충돌하고, knob 경로가 무효로 확인됐다. 따라서 이 실행을 5-branch 기저 arm으로 사용하는 판단은 계측 결함으로 무효다.
- **현재 실행의 “오염 검사 통과” 효력은 실행 무효다.** §3.4 오염 검사는 5-branch 앵커 4자리 일치를 요구한다. 현재 코드는 그 앵커를 재현하지 못하고, 앵커를 생성할 knob 경로도 무효다. 따라서 이 실행이 오염 검사를 통과했다고 판단할 수 없다.
- **현재 knob을 사용해 branch set을 통제하거나 weight 0으로 branch를 제외했다고 주장하는 판단은 실행 무효다.** `ICF_SHAPE_SCREEN_ONLY`는 비테스트 코드에서 읽히지 않고, `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT` 1.0→0.0 전환에도 SMAD4·PBRM1 출력이 소수 4자리까지 동일하다. 따라서 이 knob을 근거로 한 branch set 판단은 계측 결함으로 무효다.
- **현재 오염 검사 통과를 전제로 한 후보 성능 판단·승격 판단은 실행 무효다.** 보고서 §4는 오염 검사 상수 불일치 상태에서 후보 성능을 판단하는 실행을 무효로 분류한다고 기록한다. 현재 상태에서는 오염 검사를 수행할 경로가 없으므로, 그 통과를 전제로 한 후보 판단은 수행할 수 없다.
- **현재 관측값 `0.4661`/`0.5408`을 5-branch 앵커 대용으로 고정하는 판단은 실행 무효다.** 이 값들은 현재 결함 경로에서 나온 관측값이지, 검증된 5-branch 기저 값이 아니다. per-fold 대조와 branch provenance 확인 전에는 새 앵커로 사용할 수 없다.

## 좌석 공백 (기권·실패로 빠진 관점)
- **Phase 1 R1(반증) 좌석이 기권했다.** 제공된 Phase 1 R1 본문은 비어 있다. 따라서 Phase 1에서는 반증 좌석의 독립적 반박·반증 산출물이 없다. 이 공백은 어떤 주장이 옳거나 틀렸다는 증거가 아니라, 반증 관점의 Phase 1 출력이 부재하다는 운영 공백이다.

## 다음 행동 (권한 등급과 함께)
- **T0 — 즉시 동결**: 현재 `ICF_SHAPE_SCREEN_ONLY` 또는 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`를 근거로 한 5-branch BASE arm 생성, 오염 검사 통과 주장, 후보 성능 판단, 승격 판단을 중단한다. 근거: 보고서 §6의 knob 무효 확인과 §3.4 앵커 미재현.
- **T0 — 상태 재분류**: D-042 승격 근거, 게이트 ① 결과, 게이트 ② 실적, 현재 7-branch 재생성을 `verified`로 남기지 않고, provenance 확인 전까지 `unverified-hold` 또는 `판별 불가`로 표시한다. 근거: 과거 per-fold score, branch list, 코드 해시, fold id, 데이터 버전, run log가 문서에 없음.
- **T0 — 코드 감사**: 다음을 확인한다.  
  1. `ICF_SHAPE_SCREEN_ONLY`의 비테스트 코드 reader 재확인. 동적 문자열, config, shell export, 생성 코드 포함.  
  2. `ICF_FIXED_HEAD_{BS,SH,SJ,CT}_WEIGHT` reader와 파싱 위치. `0.0`이 falsy로 처리되어 default로 되돌리는 bug, 이름 불일치, config override 확인.  
  3. `scripts/run_ru90_shape_triple.sh` → runner → branch score 생성 → score 저장/로드 → cache key → aggregator → AUROC/trimmed mean 호출 경로.  
  4. branch list 결정점: `BRANCHES_V121_5`, 공식 7-branch 상수, CLI, config, env, 하드코딩 중 어디가 실제 결정하는지.  
  5. `BRANCHES_V121_5`의 정의와 사용 부위. score 생성과 aggregation 양쪽에 도달하는지.  
  6. ensemble input 구성: branch score 포함 여부, weight vector, mask, normalization, rank, trimmed mean. weight 0이 input 부재인지 기여 0인지.  
  7. `scripts/analysis/branch_screen.py`의 margin 생성 경로. `SCREEN_ONLY`가 margin 생성에 영향을 주는지, ensemble input에 screen branch가 들어가는지, label/target 접근이 있는지, fit 범위가 train fold만인지.  
  8. `predictions/pathobench_*_ru90_shape_triple_official50_bf16.pt`의 metadata: branch list, weights, env, 코드 해시, fold id, 데이터 버전, 라이브러리 버전, 파일 해시.  
  9. 저장된 per-fold 예측으로 공식 7-branch `0.6227`과 재생성 `0.6226`의 per-fold raw AUROC, fold 순서, task 매핑, 집계 방식 대조.  
  10. `git log`/`git blame`으로 D-042/RU-90 시점 코드와 현재 코드 차이. branch definition, preprocessing, normalization, AUROC computation, fold mapping, trimmed mean 변화 확인.  
  11. fold 파일 무결성: PathoBench 공식 fold hash, 중복·누락·train/test overlap.  
  12. 데이터/feature 버전, background GMM/fit 범위(`max_iter`, sample count, seed), bf16/GPU/PyTorch/sklearn 버전.  
  13. tests가 env/weight/branch list 동작을 실제 runner로 검증하는지, mock만인지.  
  14. cache/shared state: 임시 디렉터리, `torch.load`, `load_state_dict`, 파일 쓰기, seed 초기화, GPU 상태, fold 순서.
- **T0 — provenance 인벤토리**: D-042, 게이트 ①, 게이트 ②, 5-branch baseline, 7-branch baseline 각각에 대해 다음 증거를 찾는다. per-fold score, branch list, weight vector, 코드 해시, run log, fold id, 데이터 버전, score 파일 해시, 집계 설정, 후보 등록 시점, 성능 조회 시점, SEAL 10 접근 여부, 라벨 사용 코드. 증거가 없으면 `unverified-hold` 또는 `판별 불가`로 둔다.
- **T1 — 계측 경로 복구**: branch set을 env knob이 아니라 명시적 branch list 단일 결정점으로 만든다. runner → score 생성 → score 저장/로드 → cache key → manifest → aggregator 전체에 branch list가 포함되도록 한다. weight는 branch list 확정 이후에만 적용하고, weight 0의 의미를 코드 단언과 단위 테스트로 고정한다. 근거: 현재 knob 경로가 무효로 확인됐고, score 재사용/cache 우회 가능성을 배제하려면 provenance가 필요하다.
- **T1 — 오염 검사 재설계**: 1차 검사는 값 앵커가 아니라 입력·경로 불변량으로 한다. 검사 대상은 (a) 의도된 base branch list와 실제 ensemble input branch list 일치, (b) screen branch의 ensemble input 부재, (c) weight vector와 branch list 정합, (d) aggregation mask, (e) fold mapping, (f) config/manifest hash, (g) label-free margin 경로, (h) train/test fold mask 고정, (i) shared state/cache/seed/GPU 부수 효과 격리다. 결정론적 구성은 per-fold 예측 해시 또는 `max|diff| = 0`을, 학습 파라미터가 있는 구성은 사전 고정 허용 오차와 적합 분산을 함께 쓴다.
- **T1 — 값 앵커의 위치 재조정**: SMAD4 `0.4421`, PBRM1 `0.5553`은 독립적으로 검증된 5-branch 재생성 또는 보존된 과거 per-fold score가 있을 때만 보조 regression fingerprint로 쓴다. 현재 재현되지 않는 이 값을 pass/fail 기준으로 쓰지 않고, 현재 `0.4661`/`0.5408`을 새 앵커로 고정하지도 않는다.
- **T1 — 명시적 5-branch 재생성**: `BRANCHES_V121_5` 또는 동등한 명시적 5-branch 집합을 직접 지정해 SMAD4·PBRM1·macro를 재생성한다. 결과가 `0.4421`/`0.5553`과 일치하면 knob/branch 지정 경로 문제로 좁혀지고, 일치하지 않으면 앵커의 코드/데이터/fold/반올림/집계 provenance 문제로 확장한다.
- **T2 — 판정 재분류와 사용자 결정**: provenance 감사 결과에 따라 D-042 승격 근거와 게이트 ①/② 실적을 `verified`, `execution-invalid`, 또는 `unverified-hold`로 재분류한다. 공식 7-branch baseline 유지·변경, D-042 채택 상태 유지·취소, 게이트 실적 복원·무효화는 사용자 결정이 필요하다. 근거: 승격·채택은 운영 결정이며, 현재 증거는 과거 실행의 무결성을 확정하지 못한다.

## 하지 말 것 (이유와 함께)
- **“러너가 항상 7-branch를 산출한다”를 확인된 사실로 쓰지 말 것.** 보고서 §6은 이를 추론(미검증)으로 명시한다. macro `0.6226`과 `0.6227`의 수치 일치는 branch set 결정 경로의 확인이 아니다.
- **현재 `0.4661`/`0.5408`을 5-branch 앵커 대용으로 고정하지 말 것.** 이 값들은 현재 결함 경로에서 나온 관측값이다. 검증된 5-branch 기저 값이 아니므로 새 기준값으로 쓰는 것은 결함을 정상값으로 만들 위험이 있다.
- **macro `0.6226`과 `0.6227`의 일치만으로 공식 7-branch baseline을 verified라고 쓰지 말 것.** 보고서 §2는 per-fold 정밀 대조를 미수행이라고 한다. macro 일치는 fold 순서, task 매핑, per-fold raw AUROC, 집계 방식의 동일성을 증명하지 못한다.
- **현재 코드 결함만으로 과거 D-042·게이트 ①·게이트 ② 전체를 무효라고 확정하지 말 것.** 과거 실행의 코드 버전, branch list, score 파일, fold id, 데이터 버전, run log가 문서에 없다. 현재 grep/실행 결과는 현재 코드의 knob 무효와 현재 5-branch 앵커 미재현만 말한다.
- **좌석 몇 개가 동의했는지를 근거로 쓰지 말 것.** 표결은 증거가 아니다. 각 주장은 보고서·PROJECT.md의 구체적 관측, 코드 경로, provenance 증거로만 판단한다.
- **오염 검사를 값 앵커 4자리 일치만으로 수행하지 말 것.** 값 앵커는 branch set, 데이터, fold, 코드, 수치 정밀도의 합성 downstream 신호다. knob 무효 상태에서는 값 일치가 오염 부재를 증명하지 못하고, 값 불일치가 반드시 오염임을 증명하지도 못한다.
- **weight 0을 branch score의 ensemble input 부재로 단정하지 말 것.** 확인된 사실은 weight 1.0→0.0 전환에 SMAD4·PBRM1 출력이 소수 4자리까지 동일하다는 것이지, weight 0이 input에서 branch를 제거하는지, 기여만 0으로 하는지, reader가 없는지는 코드 경로 확인 전까지 알 수 없다.
- **오염 검사 통과를 전제로 후보 성능 판단·승격 판단을 진행하지 말 것.** 보고서 §4는 오염 검사 상수 불일치 상태에서 후보 성능을 판단하는 실행을 무효로 분류한다. 현재 오염 검사 경로가 수행 불가하므로, 그 통과를 전제로 한 판단은 실행 무효다.
- **`+0.003` 승격 기준, CI 보고, sign agreement 보조 지표, hold-out 금지 규칙을 사후 변경하지 말 것.** 계측 결함은 기준 변경의 근거가 아니다. 기준 변경은 사용자 결정과 사전 설계가 필요하다.
- **SEAL 10을 선택·튜닝·개선 방향 결정에 사용하지 말 것.** `PROJECT.md` §3.1과 `D-047`은 SEAL 10을 최종 hold-out으로 유보하고, 그 결과로 모델·후보·구조를 고르는 것을 금지한다.
- **문서에 없는 수치, 과거 run metadata, per-fold score, 코드 해시를 지어내지 말 것.** 없는 provenance는 “문서에 없음” 또는 “판별 불가”로만 기록한다.
