# ICF Council — C-20260918-11

- **연구 질문**: 직전 회차(C-20260918-10)가 provenance v1 스키마를 냈고, 핵심은 requested와 effective를 분리해 기록하고 불일치 플래그를 두는 것이다. 이 회차는 그 스키마의 가장 약한 고리를 공격한다.

문제: effective_branch_list를 기록하는 주체는 결국 같은 실행 코드다. 그런데 이번에 드러난 결함은 바로 그 코드가 요청을 무시했다는 것이다(src/가 환경변수를 하나도 읽지 않아 ICF_* export가 전부 무효였고, 가중치를 0으로 바꿔도 출력이 소수 4자리까지 동일했다). 버그의 당사자가 자기 상태를 보고하게 하는 구조다.

질문: (1) effective 기록을 신뢰할 수 있는 조건은 무엇인가? 자기 보고가 아니라 외부에서 검증 가능한 신호는 무엇인가. (2) '가중치 0이 branch 제외와 동등한가'를 기록이 아니라 실행으로 확인하는 방법은? 서로 다른 구성이 서로 다른 출력을 내는지 검사하는 최소 실험을 설계하라. (3) 지금 같은 결함 - 설정이 무시되는데 출력은 정상으로 보이는 - 을 잡아낼 수 있는 불변식은 무엇인가. sidecar 존재만으로는 왜 부족한가. (4) 이 검사를 회귀 스위트에 넣는다면 어떤 형태여야 하는가? GPU 없이 도는가? (5) 과거 산출물에는 requested도 effective도 없다. 소급 귀속이 가능한 최대 범위는 어디까지인가.

문서에 없는 수치를 지어내지 말라. 첨부된 10회차 보고서의 스키마 제안을 근거로 삼되 그것을 방어하지 말라.
- **RU 유형**: 진단
- **최대 권한 등급**: T1
- **소집자**: Claude Code (claude-opus-5)
- **좌석**: A1(A, temp 0.2, http://127.0.0.1:8000/v1/chat/completions), T1(T, temp 0.4, http://127.0.0.1:8001/v1/chat/completions), R1(R, temp 0.5, http://127.0.0.1:8003/v1/chat/completions), P1(P, temp 0.7, http://127.0.0.1:8000/v1/chat/completions), S1(S, temp 0.2, http://127.0.0.1:8001/v1/chat/completions)
- **호출**: 9/14 · **소요**: 627.7초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 편성 근거 (결과 전 기록)

10회차 산출물의 약한 고리를 공격하는 회차다. 감사(A)를 첫 좌석에 두고 온도를 최저로 했다 - '자기 보고를 신뢰할 조건'이 정확히 감사 좌석의 질문이다. 이론(T)은 '설정이 무시되는데 출력은 정상인 상태'를 잡는 불변식 설계를 맡는다. 반증(R)은 스키마가 형식만 늘리고 검증력이 없는 경우를 공격한다. 제안(P)은 최소 실험을 설계하되 온도를 0.7로 낮췄다 - 창의성보다 실행 가능성이 중요하다. 일반화(G)와 실측(M)은 제외: 질문이 내부 계측 신뢰성이고, GPU는 별도 잡무가 쓰고 있다. 10회차 보고서를 좌석 입력에 넣었으므로 자기 방어 편향이 있을 수 있고, 그래서 프롬프트에 '그것을 방어하지 말라'를 명시했다. 한계로 결론에 적는다.

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: requested/effective 두 증언과 canary 도달 불변식
- **바꾸는 것**: provenance v1의 `effective_branch_list` 기록을 단일 sidecar 자기 보고에서 분리한다. (a) wrapper는 실행 전에 `requested manifest`를 쓴다: `execution_id`, `requested_branch_list`, `requested_weights`, `requested_mode`, `requested_canary_token`, `script_path`, `git_commit`(있으면), `requested_manifest_hash`. (b) `src/`의 ensemble builder는 branch selection과 score 산출 직전에 `effective manifest`를 쓴다: `execution_id`, `effective_branch_list`, `effective_weights`, `effective_mode`, `effective_canary_token`, `resolved_config_hash`, `code_hash`, `environment_hash`, `data_version`, `fold_manifest_hash`, `prediction_file_hash`, `effective_manifest_hash`, `record_writer`, `cache_or_source_flag`. (c) orchestrator는 `execution_id`로 requested, effective, prediction file을 결합하고 `requested_effective_mismatch`, `canary_mismatch`, `hash_mismatch`를 계산한다. sidecar 존재는 `verified` 조건에서 제외한다. `effective` 기록의 신뢰 조건은 다음이다: effective 기록 주체가 실제 branch selection/score 산출 상태를 관찰하는지, requested 기록과 분리된 기록인지, canary가 requested에서 effective로 도달했는지, prediction file과 hash가 바인딩됐는지, 통제 변수 변경이 paired 실행에서 artifact를 바꾸는지, unrecorded switch가 없는지 코드 감사로 확인되는지. 외부 검증 신호는 `requested_canary_token`의 effective 도달, paired 실행의 `prediction_file_hash` 차이, hash의 독립 재계산이다. 불변식은: `requested_canary_token == effective_canary_token` AND `requested_effective_mismatch == false` AND `prediction_file_hash == sidecar.prediction_file_hash` AND `sidecar_hash == recomputed_hash`; 출력에 영향을 주도록 선언된 통제 변수를 변경하면 effective state 또는 `prediction_file_hash`가 변해야 한다. sidecar 존재만으로는 부족하다: sidecar를 만든 코드와 산출물을 만든 코드가 동일하고, requested를 effective로 복사하거나 config loader 값을 기록할 수 있으며, 사후 추가·수정이 가능하기 때문이다.
- **바꾸는 전제**: 지금 참이라고 두고 있는 것은 “`effective_branch_list`는 같은 실행 코드가 기록하면 신뢰할 수 있고, sidecar가 있으면 기록이 실행을 반영한다”는 전제다. 이를 뒤집는다: 자기 보고는 requested와 calculation-time effective가 분리 기록되고 canary로 연결되기 전까지 `unverified`이며, sidecar 존재는 신뢰 신호가 아니다.
- **기대하는 차이**: `src/`가 env를 읽지 않는 현재 결함에서 requested canary가 effective에 없거나 `mismatch=true`가 되어 실행 무효로 잡힌다. `ru90_shape_triple`처럼 태그와 실제 구성이 어긋난 경우를 사후 대조에서 막는다. 정량 예측 불가.
- **기전**: wrapper가 요청을 독립 기록하고, calculation이 실제 branch selection 상태에 canary를 기록하면, env/설정 경로가 끊어져도 requested와 effective가 불일치한다. 그러나 canary가 실제 계산 상태가 아니라 metadata 복사로 채워지거나 effective writer가 config loader라면 기전 미확인이며, 이 제안은 코드 수정과 감사 없이는 성립하지 않는다.
- **실패 조건**: effective manifest가 requested 값을 복사하거나 config loader에서 기록하면 canary 통과해도 실제 계산과 무관하다. requested manifest가 wrapper와 calculation 사이에서 수정·누락될 수 있고, canary token이 출력·계산에 영향 없는 경로만 지나면 설정 도달을 검증하지 못한다. sidecar가 prediction file과 hash 바인딩 없이 사후 작성되면 존재만으로는 무의미하다.

## 제안 2: 가중치 0·제외·비0의 CPU negative control
- **바꾸는 것**: 회귀 스위트에 SEAL 10 미사용, GPU 미사용, 소형 synthetic task/fold manifest로 실제 evaluation code path를 통과하는 paired 실험을 넣는다. target branch B에 대해 3 arm을 돌린다: (1) B를 branch list에서 제거, (2) B를 포함하되 weight `0.0`, (3) B를 포함하되 nonzero weight. seed 고정, deterministic precision, cache off, fold manifest 고정. `per_fold`, `fold_aurocs`, `prediction_file_hash`를 비교한다. 판정: (3)이 (1)/(2)와 같으면 설정 도달 실패; (1)과 (2)가 같으면 그 경로에서 weight 0은 제외와 동등; 다르면 weight 0을 제외로 해석할 수 없고 `effective_branch_list`/`effective_weights` 기록에서 불일치로 표시한다. 결정론을 보장하지 못하면 exact equality를 쓰지 않고 해당 검사는 판별 불가/실행 무효로 둔다. GPU 없이 CPU로 돌린다; production 경로가 CPU에서 실행되지 않으면 회귀는 통과가 아니라 blocked다. 이 검정은 GPU 수치 경로 전체의 등가 입증이 아니라 설정 도달과 기록 불변식의 smoke다.
- **바꾸는 전제**: 가중치 0은 branch 제외와 동등하다고 문서/스키마가 두고 있고, 출력이 정상이면 설정이 반영된 것으로 둔다. 이를 뒤집는다: 동등성과 도달성은 실행으로 확인되기 전까지 무효이며, 출력 정상은 설정 반영 증거가 아니다.
- **기대하는 차이**: env/weight가 무시되는 결함에서 (3)과 (1)/(2)가 동일해 회귀가 실패한다. weight 0이 normalization/aggregation에서 제외와 다르면 (1)/(2) 불일치로 기록 해석이 수정된다. 정량 예측 불가.
- **기전**: branch score가 ensemble score에 가중치로 기여하면 nonzero weight는 score를 바꾸어야 하고, weight 0은 기여를 없애야 한다. 그러나 현재 aggregation이 선형인지, weight 0 branch가 denominator/side effect에 남는지 확인되지 않았으므로 기전 미확인이다. 실험은 관측으로 판별할 뿐 기전을 입증하지 않는다.
- **실패 조건**: synthetic test가 production과 다른 builder를 mock하면 false pass/fail. target branch contribution이 0이거나 synthetic data가 branch를 구분 못 하면 (3)이 변하지 않아 설정 무응답과 구분 불가. CPU/GPU/precision 차이가 exact equality를 깨면 오거부. 학습 nondeterminism이 있으면 exact equality가 성립하지 않아 판별 불가. weight 0을 의도적으로 제외와 다르게 설계했는데 equality를 강제하면 false failure.

## 제안 3: 소급 귀속의 최대 범위를 증거 클로저 후보 집합으로 제한
- **바꾸는 것**: 과거 산출물에 requested/effective를 사후 생성하지 않는다. `provenance_registry`에 `artifact_hash`, `evidence_list`(`script_path`, `git_commit`, `code_closure_hash`, `run_log`, config files, timestamps, values), `candidate_configurations`, `attribution_status`, `max_attribution_scope`, `assumptions`, `contradictions`를 기록한다. status는: `verified`는 미래 sidecar + canary + hash binding 통과에만. `inferred`는 독립 증거가 execution 시점 code closure를 특정하고 그 closure가 effective 구성을 유일하게 결정할 때만. 그 외에는 candidate set. `unattributed`는 독립 증거가 없거나 모순. 태그, macro 근접, 문서 기대는 단독 근거가 아니라 consistency check만. requested는 script/log/env export가 보존돼야 추론 가능하고, effective는 code closure가 결정해야 추론 가능. 최대 범위는: execution 시점 code closure가 특정된 effective 구성까지, 그리고 script/log closure가 특정된 requested 구성까지다. 그 이상은 불가. 현재 `ru90_shape_triple`은 run-time commit과 script가 보고 §8의 5-branch 경로와 동일함이 확인되면 inferred 5-branch candidate로만, commit/코드 시점이 확인되지 않으면 unattributed 또는 candidate set. 공식 7-branch baseline `0.6227`은 verified로 승격되지 않는다.
- **바꾸는 전제**: 과거 산출물은 태그·값·일시로 특정 구성에 귀속할 수 있고, requested/effective가 없어도 문서 기대가 구성을 특정한다고 둔다. 이를 뒤집는다: 소급 귀속은 독립 증거가 허용하는 최대 후보 집합을 넘지 못하고, requested와 effective를 분리한다.
- **기대하는 차이**: `ru90_shape_triple`을 공식 7-branch score로 오귀속하는 실행을 막고, baseline `unverified-hold`를 유지한다. 과거 산출물의 사용 가능 범위가 명시된다. 정량 예측 불가.
- **기전**: 증거 클로저가 가능한 구성 집합을 축소한다. closure가 유일하면 inferred, 다수이면 candidate set, 없으면 unattributed. 그러나 증거가 실제 실행을 완전히 특정한다는 입증이 아니라, 현재 증거 하에서 허용되는 범위를 표시하는 절차다. 코드 이력/로그가 불완전하면 기전 미확인.
- **실패 조건**: git commit, untracked changes, run log, config closure가 불완전하면 inferred가 overclaim한다. value coincidence가 후보를 좁히는 것처럼 쓰이면 오귀속. candidate set이 지나치게 넓으면 모든 과거가 unattributed가 되어 실용성이 떨어진다. confidence를 주관 등급으로 두면 status가 불안정해진다.

### A1 (감사)

## 점검 1: `effective_branch_list`가 실제 앙상블 계산에 반영되는지 — 자기 보고의 신뢰 조건
- **확인하려는 것**: sidecar의 `effective_branch_list`·`effective_branch_weights`가 config 객체의 선언값인지, 아니면 앙상블 계산에 실제로 사용된 값인지. `src/`가 env를 읽지 않는 현재 상태에서 config 객체가 단일 진실원천인지. 외부에서 검증 가능한 신호가 무엇인지.
- **문서에서 확인되는가**: 모순됨. 보고 §8에서 `src/`는 `os.environ`/`getenv`을 읽지 않는 파일이 0개이고, `src/models/config.py` 기본값은 `weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`이며, 사용된 YAML(`train_v98_p1_reverse_1536_1gpu.yaml`)은 브랜치 가중치를 설정하지 않는다. 보고 §6에서 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0`을 export해도 SMAD4 `0.4661`, PBRM1 `0.5408`으로 4자리까지 동일하다. 그러나 config 객체의 가중치 0이 앙상블 계산에서 branch를 실제로 제외하는지, 아니면 가중치 0으로 기여하지만 계산에 포함되는지는 문서에 코드 대조 결과가 없다.
- **샌다면 나타날 증상**: sidecar가 `effective_branch_list=["CV","BM","BD","QA","DS"]`를 기록하지만, 실제 앙상블 계산은 하드코딩된 다른 branch 집합을 사용하거나, 가중치 0인 branch가 계산에 포함된다. `requested_branch_list`가 7-branch, `effective_branch_list`가 5-branch로 불일치하고, 어느 쪽이 "실제 실행"인지 불분명하다. sidecar가 config 선언값만 기록하면, `src/`가 env를 읽지 않는 한 requested와 effective를 구분하지 못한다.
- **확인 방법**: `src/models/`의 앙상블 계산 코드를 읽어 config 객체의 `branch_weights`가 실제로 어떤 branch를 선택하는지, 가중치 0인 branch가 제외되는지 확인. `grep -rn "weight_sh\|weight_sj\|weight_bs" src/`로 가중치 필드가 앙상블 계산에서 사용되는 위치를 확인. `scripts/run_ru90_shape_triple.sh`가 어떤 eval 스크립트를 호출하는지 확인. `icf_arm_v121()`이 `icf_arm_v120` 호출 뒤 `ICF_FIXED_HEAD_CT_WEIGHT=0.0`을 export할 뿐이라는 보고 §8의 확인과 대조.

## 점검 2: 가중치 0이 branch 제외와 동등한지 — 실행으로 확인하는 최소 실험
- **확인하려는 것**: 가중치 0을 부여한 branch가 앙상블 계산에서 실제로 제외되는지, 아니면 가중치 0으로 기여하지만 계산에 포함되는지. 서로 다른 구성이 서로 다른 출력을 내는지.
- **문서에서 확인되는가**: 문서에 없음. 보고 §6에서 env 가중치 0이 출력에 영향을 주지 않았지만, 이는 `src/`가 env를 읽지 않기 때문이며, `config.py` 기본값 `weight_sh=0.0` 등이 앙상블 계산에서 branch를 실제로 제외하는지는 문서에 코드 대조 결과가 없다. 보고 §8은 "5-branch 드리프트의 원인(코드 변경 / 매니페스트 / 전처리 / 수치 정밀도). 미확인"이라고 명시한다.
- **샌다면 나타날 증상**: 가중치 0인 branch가 계산에 포함되면, branch 제외와 가중치 0이 동등하지 않아 구성 차이를 관측된 수치 차이로 오인한다. 또는 가중치 0이 제외를 의미하면, config 기본값 5-branch가 실제로 5-branch 산출물을 만드는지 확인 불가하다. `eval_v121.sh`가 정의상 5-branch arm이지만, `ICF_FIXED_HEAD_CT_WEIGHT=0.0` export가 `src/`에 도달하지 않으므로, CT가 실제로 제외되는지 확인 불가.
- **확인 방법**: (a) `src/models/config.py`에서 가중치 0인 branch가 앙상블 계산에서 제외되는지 코드 대조. (b) 최소 실험: 동일 fold에서 (i) SH 가중치 1.0, (ii) SH 가중치 0.0으로 앙상블 계산만 수행하고 per-fold 예측을 비교. 출력이 다르면 가중치 0이 제외와 동등. 동일하면 가중치 0이 제외와 동등하지 않거나, 가중치가 계산에 반영되지 않음. (c) `ICF_FIXED_HEAD_CT_WEIGHT=0.0`을 export한 뒤 config 객체의 `weight_ct` 값이 0.0인지, 그리고 그 0.0이 CT branch를 앙상블에서 제외하는지 코드 대조.

## 점검 3: 설정 무시 + 출력 정상 불변식 — sidecar 존재만으로는 부족한 이유
- **확인하려는 것**: 설정이 무시되는데 출력은 정상으로 보이는 결함을 잡아낼 불변식. sidecar 존재만으로는 왜 부족한지.
- **문서에서 확인되는가**: 확인됨. 보고 §6에서 가중치를 0으로 두어도 소수 4자리까지 값이 바뀌지 않고, 두 실행 사이에 결정론적으로 동일하다. 보고 §8에서 `src/`는 환경변수를 하나도 읽지 않아 실행 스크립트의 모든 `ICF_*` export가 무효다. `run_ru90_shape_triple.sh` 산출물은 공식 7-branch가 아니라 5-branch 산출물이며, 공식 태그로 잘못 이름 붙었다. macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이다.
- **샌다면 나타날 증상**: sidecar가 존재하지만 내용이 무효. `requested_branch_list`가 7-branch, `effective_branch_list`가 5-branch로 불일치하고, sidecar가 config 선언값만 기록하면 이 불일치를 감지하지 못한다. sidecar 존재만으로 `verified`를 주장하면, 설정 표면이 무효인 상태에서 구성 귀속이 보장되지 않는다. 값 앵커(AUROC 4자리 일치)가 폐지된 뒤, provenance 대조만으로 구성 동일성을 판정하면, 설정 무시 결함이 감지되지 않는다.
- **확인 방법**: sidecar의 `requested_branch_list`와 `effective_branch_list`를 비교하고, 불일치 시 `mismatch=true` 플래그를 기록. sidecar의 `effective_branch_list`가 실제 앙상블 계산에 반영되는지 코드 대조. 불변식: `requested_branch_list != effective_branch_list`이면 해당 산출물은 `verified`가 아니다. sidecar 생성 시점에 `effective_branch_list`가 앙상블 계산에 반영되는지 확인하는 검증을 수행하고, 확인되지 않으면 `effective_branch_list`를 `미확인`으로 표시.

## 점검 4: 회귀 스위트 형태 — GPU 없이 도는지
- **확인하려는 것**: 이 검사를 회귀 스위트에 넣는다면 어떤 형태여야 하는지. GPU 없이 도는지.
- **문서에서 확인되는가**: 문서에 없음. 회귀 스위트에 대한 문서가 없다. `tests/test_docs_consistency.py`가 존재한다는 언급만 `PROJECT.md` 머리말에 있다.
- **샌다면 나타날 증상**: 회귀 스위트에 설정 무시 결함이 잡히지 않아, 코드 변경 후에도 설정 표면이 무효인 상태로 유지된다. `src/`가 env를 읽지 않는 결함이 재발해도 감지되지 않는다. 가중치 0이 branch 제외와 동등하지 않은 결함이 재발해도 감지되지 않는다.
- **확인 방법**: 회귀 스위트에 다음을 추가: (a) env 변수가 `src/`에 도달하는지 확인 — `ICF_FIXED_HEAD_SH_WEIGHT=1.0`을 export한 뒤 config 객체의 `weight_sh` 값이 1.0인지 확인. (b) 가중치 0이 branch 제외와 동등한지 확인 — SH 가중치 0.0인 config로 앙상블 계산 시 SH가 제외되는지 확인. (c) `requested_branch_list`와 `effective_branch_list` 불일치 시 실패. GPU 없이: config 객체 로드와 앙상블 계산의 branch 선택 로직만 확인하면 GPU 불필요. 실제 AUROC 계산은 GPU 필요하지만, branch 선택 로직과 가중치 반영 로직은 CPU에서 확인 가능.

## 점검 5: 과거 산출물 소급 귀속의 최대 범위
- **확인하려는 것**: 과거 산출물에는 requested도 effective도 없다. 소급 귀속이 가능한 최대 범위는 어디까지인지.
- **문서에서 확인되는가**: 확인됨. 보고 §8에서 저장된 예측 파일에도 branch list·가중치·코드 해시가 기록돼 있지 않아 과거 산출물을 특정 구성에 귀속시킬 수 없다. `run_ru90_shape_triple.sh`로 생성된 7건은 공식 7-branch가 아니라 5-branch 산출물이며, 공식 태그로 잘못 이름 붙었다. macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이다. 기록된 5-branch 앵커 `0.4421`/`0.5553`은 현재 코드로 재현되지 않는다.
- **샌다면 나타날 증상**: 과거 산출물을 현재 구성으로 오귀속하여 관측된 차이가 구성 차이에서 새어 나간다. 태그 또는 값 단독으로 구성을 특정한다. `ru90_shape_triple`처럼 태그와 실제 구성이 어긋난 사례가 반복된다. `PROJECT.md` §3.3의 "직접 비교 금지" 위반.
- **확인 방법**: 과거 산출물의 파일명 태그, 생성 시점, git 이력, 실행 로그를 대조. 소급 귀속 가능/불가능 표시. `ru90_shape_triple`은 태그와 실제 구성이 어긋난 사례이므로, 태그 단독으로 소급 귀속하지 않는다. macro 근접성(`0.6226` vs `0.6227`)은 우연으로 기록됐으므로, 값 단독으로 소급 귀속하지 않는다. 소급 귀속이 가능한 최대 범위: git 이력과 실행 로그가 존재하고, 그 시점의 코드에서 `src/`가 env를 읽었는지, config 기본값이 현재와 동일한지 확인 가능한 경우. 이 조합이 없으면 `unattributed`.

### T1 (이론)

## 이론적 병목 1
- **가정**: effective 기록을 신뢰할 수 있는 조건은, 기록이 실제 앙상블 계산이 구성을 소비하는 지점에서 생성되고, 예측 파일과 hash·execution_id로 바인딩되며, 같은 코드의 자기 보고가 아니라 외부 신호(독립 재계산, 구성 민감성 실험, 코드 경로 감사)로 교차 검증된다는 것이다. 현재 접근이 암묵적으로 두는 것은 requested/effective 분리 자체만으로 effective가 참이라고 주장할 수 있다는 점이다.
- **이 가정이 깨지는 조건**: effective 필드가 config 로드값·스크립트 export·기본값·문서 기대를 복사할 때; branch 포함·가중치·모드가 config 밖의 상수·import side effect·캐시·전처리·정규화·global state·hardcoded path로 결정될 때; 가중치 0이 branch 제외와 동등하지 않을 때; 기록이 예측 파일과 다른 프로세스·시점·파일에서 생성될 때; 예측이 캐시·이전 실행·수동 복사일 때; sidecar가 사후 수정 가능할 때; requested가 src에 전달되지 않아 effective recorder가 requested를 알 수 없을 때.
- **깨질 때 나타나는 관측 가능한 증상**: sidecar가 requested 7-branch, effective 5-branch를 쓰지만 어느 쪽이 실제 계산인지 외부 신호 없이 판별 불가; effective 기록이 5-branch인데 산출물 태그·문서 기대는 7-branch; 같은 effective 기록으로 재실행 시 prediction_file_hash·fold_aurocs가 다름; 같은 예측 파일에 서로 다른 effective 기록; requested/effective mismatch 플래그가 true인데 출력 민감성·재계산이 없어 구성 귀속 실패 또는 판별 불가; sidecar 존재만으로 verified 주장 시 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: src/가 os.environ·getenv을 읽지 않아 ICF_* export가 무효; src/models/config.py 기본값이 weight_sh=0.0·weight_sj=0.0·weight_bs=0.0이고 사용된 YAML이 브랜치 가중치를 설정하지 않음; ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0 export에도 SMAD4 0.4661·PBRM1 0.5408로 소수 4자리 동일; run_ru90_shape_triple 산출물이 공식 태그를 쓰지만 5-branch로 확인; 기록된 5-branch 앵커 0.4421·0.5553·0.6171이 현재 코드로 재현되지 않음.

## 이론적 병목 2
- **가정**: 가중치 0이 branch 제외와 동등한지는 기록이 아니라 실행으로 확인해야 하며, 확인을 위해선 동일 데이터·코드·시드·fold에서 branch absent, weight 0, nonzero weight 구성이 서로 다른 출력을 낼 수 있어야 하고, 그 차이를 사전 등록된 판정 기준으로 측정할 수 있다고 둔다.
- **이 가정이 깨지는 조건**: absent 구성을 코드에서 표현할 수 없을 때; nonzero weight branch의 기여가 0이거나 출력에 민감하지 않을 때; 코드·설정 경로가 두 구성 모두를 무시할 때; 학습·seed·수치 병렬로 출력이 변동하고 허용 기준이 없을 때; positive control이 없어 출력 변화 자체가 불가능한 실행을 구분하지 못할 때; raw per-branch inputs가 없어 재계산이 불가능할 때.
- **깨질 때 나타나는 관측 가능한 증상**: weight 0과 absent가 사전 기준 내에서 동일하면 해당 출력에 대해 operationally equivalent; 다르면 동등하지 않음. nonzero weight와 weight 0/absent가 동일하면 weight 경로가 무효이거나 branch 기여가 0/민감도 부족; positive control도 동일하면 실행 무효. stochastic이고 허용 기준 없으면 판별 불가. 기록만으로는 기전 미확인.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: SH/SJ/BS 가중치를 명시적으로 0.0으로 두고 SMAD4·PBRM1만 재실행해도 0.4661·0.5408로 소수 4자리 동일; ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT가 출력에 아무 영향이 없다는 코드 확인; 가중치 0이 branch 제외와 동등한지 코드 대조 결과가 문서에 없음; D-048 이후 seed std 0.00000이 더는 성립하지 않고 RU-97 §8에서 동일 구성에서 seed만 바꿔 macro 폭 0.0200을 실측.

## 이론적 병목 3
- **가정**: 설정 무시 결함을 잡는 불변식은, 문서화된 조작이 honored되면 출력이 달라야 하는 구성 쌍에서 출력이 실제로 달라야 한다는 구성 민감성 불변식과, 저장된 예측이 기록된 raw branch inputs·effective weights·aggregation code로 독립 재계산과 일치해야 한다는 재계산 불변식으로 설정할 수 있다고 둔다. sidecar 존재 또는 requested/effective mismatch 플래그만으로는 이 결함을 잡지 못한다.
- **이 가정이 깨지는 조건**: 출력에 민감한 구성 쌍이 없거나 branch 기여가 0일 때; raw branch inputs가 저장되지 않아 재계산 불가할 때; sidecar가 같은 코드에 의해 자기 보고될 때; side effect가 branch list·weights에 기록되지 않을 때; stochastic 변동에 대한 사전 기준이 없을 때; prediction file hash가 exact equality만 요구해 seed·numerical 변동과 구성 결함을 구분하지 못할 때.
- **깨질 때 나타나는 관측 가능한 증상**: sidecar가 존재하고 requested/effective mismatch가 기록돼도, 민감 구성 쌍에서 출력이 동일하면 설정 무시 또는 branch 무관; 재계산이 저장 예측과 다르면 effective 기록이 계산에 반영되지 않거나 inputs 불완전, 원인 확인 전 기전 미확인; provenance 일치인데 base arm 값이 다르면 오염·드리프트 판별 불가; exact hash 불일치가 seed·numerical인지 구성 결함인지 구분 못 하면 판별 불가; sidecar 존재만으로 verified 주장 시 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: 가중치 0 export에도 소수 4자리 출력 동일; src/가 env를 읽지 않아 export 무효; run_ru90 공식 태그 산출물이 5-branch; 값 앵커 4자리 일치로는 구성 변화와 드리프트를 구분할 수 없다는 D-050 설계 한계.

## 이론적 병목 4
- **가정**: 회귀 스위트는 sidecar 필드 존재가 아니라, requested→effective plumbing, effective→output sensitivity, raw inputs→prediction recomputation, prediction/sidecar binding을 실제 실행 경로에서 검증해야 한다. GPU 없이 돌 수 있는지는 실제 모델 추론을 합성 branch score·ensemble aggregation으로 대체할 수 있는지에 달려 있다.
- **이 가정이 깨지는 조건**: 테스트가 sidecar 존재·필드 타입만 검사할 때; requested/effective mismatch 플래그만 검사하고 출력 민감성을 검사하지 않을 때; mock이 production code path를 우회할 때; 민감한 synthetic branch inputs가 없을 때; 허용 기준 없이 exact equality만 사용할 때; 실제 평가 경로가 GPU를 요구하는데 CPU 대체 경로가 없을 때; 테스트가 default 구성만 통과시킬 때.
- **깨질 때 나타나는 관측 가능한 증상**: 스위트가 green인데 production은 env를 무시하고 5-branch만 산출; weight 0 vs nonzero가 출력을 바꾸는지 assert가 없어 결함 통과; CPU 테스트는 aggregation만 증명하고 실제 branch margin 생성 경로 미검증; GPU 경로가 회귀에서 빠지면 결함 재발; exact hash 테스트가 seed·numerical 변동으로 flaky하거나, 소수 4자리 드리프트를 놓침.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: 값 앵커는 소수 4자리 불일치를 보였지만 구성 변화와 드리프트를 구분하지 못해 폐지·재설계 대상; 가중치 0 export에도 4자리 출력 동일; 공식 태그 산출물이 5-branch였음에도 macro 근접성만으로는 기전 증명이 되지 않음.

## 이론적 병목 5
- **가정**: 과거 산출물에 requested/effective 기록이 없어도, 소급 귀속은 증거가 허용하는 구성 집합으로만 가능하다. 최대 범위는 복원된 코드·스크립트·데이터·fold·seed와 저장된 예측이 허용하는 후보 구성 집합이며, 그 집합이 1개일 때만 특정 구성으로 귀속할 수 있다. 태그·파일명·macro 근접성·현재 코드 감사만으로 특정 구성을 주장하려면, 그 증거가 다른 구성을 배제해야 한다.
- **이 가정이 깨지는 조건**: 태그 의미가 시간에 따라 바뀌거나 같은 태그가 다른 구성에 사용될 때; macro 근접성이 우연일 때; 현재 코드 감사가 과거 코드 상태를 특정하지 못할 때; code hash·run log·data version·fold manifest·seed가 복원되지 못할 때; 여러 구성이 저장된 fold_aurocs·per_fold와 동일하게 재계산될 때; raw branch inputs가 없어 후보 구성을 재계산으로 판별하지 못할 때.
- **깨질 때 나타나는 관측 가능한 증상**: ru90_shape_triple·official50 태그가 7-branch로 읽히지만 실제 5-branch; macro 0.6226이 0.6227에 가까워도 우연으로 기록; 현재 코드가 5-branch 경로여도 과거 실행이 shape margin을 포함했는지 미확인; 태그+값 단독으로 high confidence 부여 시 오귀속; 후보 구성 집합이 1개 미만이면 unattributed, 2개 이상이면 판별 불가, 재계산이 아무 후보와도 맞지 않으면 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: 저장 예측 파일 키에 branch list·가중치·코드 해시·데이터 버전 없음; run_ru90 산출물이 공식 태그를 쓰지만 5-branch; macro 근접은 우연; 기록된 5-branch 앵커 재현 불가; 과거 ru90 실행이 당시 shape margin을 포함했는지 미확인.

### R1 (반증)

## 반론 1
- **공격 대상**: "provenance v1이 requested와 effective를 분리하고 불일치 플래그를 두면 effective_branch_list를 신뢰할 수 있다"는 주장.
- **더 단순한 설명**: effective_branch_list는 결국 같은 실행 코드가 자기 상태를 보고하는 것이다. 보고 §8에서 `src/`는 환경변수를 하나도 읽지 않고, 보고 §6에서 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0`을 export해도 SMAD4 `0.4661`, PBRM1 `0.5408`로 소수 4자리까지 동일했다. 따라서 더 단순한 설명은 "branch 선택/가중치가 실행 경로에 연결되어 있지 않아 어떤 요청이 와도 같은 기본 경로가 돌고, effective 필드는 그 기본 경로의 상수/설정값을 복사할 뿐"이다. 이 설명은 requested/effective 불일치 플래그가 없거나 `false`여도 같은 관측을 낳는다. 외부에서 검증 가능한 신호는 (a) 문서화된 knob을 바꾸면 resolved effective state가 바뀌는지, (b) 그 effective state가 실제 앙상블 산출물을 바꾸는지, (c) 그 변화가 예측 파일/per_fold/fold_aurocs에서 독립적으로 관찰되는지다. 이 세 가지가 없으면 effective 기록은 자기 보고일 뿐이다.
- **무엇이 참이면 이 주장이 무너지는가**: effective 필드가 config 선언값이나 스크립트 export가 아니라 실제 앙상블 builder가 계산에 사용한 branch/weight/mode에서 생성되고, 그 필드를 바꾸면 같은 fold에서 per_fold 또는 fold_aurocs가 바뀌는 실험이 확인되면 신뢰 가능해진다. 반대로 effective 필드가 config 로드 값만 기록하고, 가중치 0이 branch 제외인지 미확인이라면 "sidecar가 있으니 effective를 신뢰한다"는 주장은 반박된다. 현재 문서에는 config 값이 앙상블 계산에 반영되는지 코드 대조가 없으므로 판별 불가다.
- **이 반론의 한계**: `src/`가 env를 읽지 않더라도 CLI/arg/YAML/하드코딩 상수 등 다른 경로로 effective가 계산에 반영되고, 그 경로가 코드 감사와 mutation test로 확인되면 반론은 약해진다. 다만 문서에는 그런 경로가 확인되지 않았다.

## 반론 2
- **공격 대상**: "가중치 0을 branch 제외와 동일하게 취급하거나, 가중치 0 export 실험으로 branch 제외 여부를 판단할 수 있다"는 주장.
- **더 단순한 설명**: 가중치 0이 출력을 바꾸지 않는 관측은 "가중치 0이 branch 제외와 동등하다"에서 나올 수도 있지만, 더 단순한 설명은 "가중치 필드가 앙상블 계산에 연결되어 있지 않다" 또는 "branch 포함/제외는 weight가 아니라 별도 list/상수로 결정된다"이다. 보고 §6의 가중치 `1.0`/`0.0` 두 실행이 소수 4자리까지 동일한 것은 env 경로가 `src/`에 도달하지 않기 때문으로 확인됐고, `config.py` 기본값 `weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`이 실제 계산에서 branch를 제외하는지는 문서에 코드 대조가 없다. 따라서 weight 0 = exclusion은 기전 미확인이다.
- **무엇이 참이면 이 주장이 무너지는가**: 최소 실험은 같은 task, 같은 fold, 같은 data/code/environment에서 한 branch의 제어 변수만 바꾸는 것이다. 제어 변수가 weight라면 `0.0`과 `1.0`을, 제어 변수가 list라면 branch 포함/제외를 비교한다. per_fold 예측과 fold_aurocs를 저장된 정밀도로 대조한다. 출력이 다르면 그 제어 변수가 실행에 영향을 준다. 출력이 소수 4자리까지 같으면, 제어 변수가 무효인지, branch 기여가 0인지, 집계 경로가 동일인지 구분할 수 없어 판별 불가다. 문서화된 knob이 `src/`에 도달하지 않고 대체 경로도 없다면 이 실험은 구성 확인이 아니라 실행 무효다.
- **이 반론의 한계**: branch score가 실제로 0이거나, branch가 이미 list에서 제외되어 있어 weight만 바뀌어도 출력이 같을 수 있다. 이 경우 실험은 weight 0과 branch 제외의 동등성을 판별하지 못한다. 독립적으로 nonzero 기여를 가진 canary branch나 코드 감사 없이는 한계가 남는다.

## 반론 3
- **공격 대상**: "sidecar에 requested/effective와 mismatch flag를 두면 설정이 무시되고 출력이 정상으로 보이는 결함을 잡는다"는 주장.
- **더 단순한 설명**: sidecar는 예측 파일 옆에 쓰이는 기록 파일일 뿐, 계산에 참여하지 않는다. 같은 결함 코드가 requested=7-branch, effective=5-branch, mismatch=true를 쓸 수도 있고, requested=7-branch, effective=7-branch, mismatch=false를 쓸 수도 있다. 보고 §8에서 `run_ru90_shape_triple` 산출물은 공식 태그를 쓰지만 5-branch 산출물이었고, macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이었다. 태그/문서 기대와 실제 산출이 분리될 수 있음을 보여준다. sidecar 존재만으로는 "기록이 계산에 사용됐다"를 보장하지 못한다.
- **무엇이 참이면 이 주장이 무너지는가**: 불변식은 다음을 포함해야 한다. (1) 문서화된 knob을 바꾸면 resolved effective state가 바뀌어야 한다. (2) resolved effective state가 바뀌면 canary 출력 또는 계산 해시가 바뀌어야 한다. (3) requested/effective mismatch가 있으면 해당 산출물은 승격/오염 검사에 사용할 수 없어야 한다. (4) sidecar는 prediction_file_hash와 execution_id로 묶이고, 예측을 계산하는 같은 경로에서 생성되어야 한다. sidecar가 예측 파일과 무관하게 쓰이거나, 상수/설정값만 복사하면 이 불변식은 성립하지 않는다.
- **이 반론의 한계**: sidecar가 실제 branch contribution tensor나 계산 그래프의 해시를 포함하고, 예측 파일과 원자적으로 묶이며, 사후 수정 시 해시 불일치가 잡힌다면 sidecar 존재는 무결성 증거가 된다. 그러나 그것이 semantic correctness, 즉 기록된 effective가 실제로 계산에 쓰였다는 것을 증명하려면 mutation test가 여전히 필요하다.

## 반론 4
- **공격 대상**: "회귀 스위트를 full 50-fold/GPU pipeline으로만 두면 이 결함을 잡는다" 또는 "GPU 없이는 검증할 수 없다"는 주장.
- **더 단순한 설명**: 이번 결함은 환경변수/설정값이 계산 경로에 도달하지 않는 배선 결함이다. full pipeline을 돌리기 전에, (a) knob이 config/resolved state에 반영되는지, (b) resolved state가 앙상블 builder 출력을 바꾸는지, (c) sidecar mismatch가 생성되는지를 작은 결정론 입력으로 테스트할 수 있다. full 50-fold는 느리고, fold/과제 효과가 배선 결함을 가릴 수 있다. GPU-free 여부는 문서에 없으므로 "문서에 없음"이다. 다만 회귀 스위트는 Primary 7 50-fold나 SEAL 10을 사용하지 않고, 단일 task/단일 fold 또는 더 작은 고정 입력으로, seed/적합 변동이 없거나 seed를 고정해야 한다. D-048 이후 `seed std 0.00000`이 더는 성립하지 않으므로, 학습/seed가 있는 경로에서는 1회 대조만으로 구성 효과를 확정할 수 없다.
- **무엇이 참이면 이 주장이 무너지는가**: 예측 생성의 유일한 경로가 GPU full pipeline이고, CPU/stub 경로가 없다면 GPU-free 회귀는 불가능하다. 그러나 그것은 문서상 공백이지, full GPU 실행만이 유일한 방패라는 근거가 아니다. CPU mini-test가 통과해도 GPU/CUDA/bf16/BLAS 경로가 다르면 GPU-specific drift를 잡지 못할 수 있다.
- **이 반론의 한계**: 앙상블 builder의 행위가 GPU kernel, 수치 정밀도, data order에 의존하면 CPU 회귀는 배선만 증명하고 전체 수치 재현성은 증명하지 못한다. 이 경우 GPU integration test가 별도로 필요하다.

## 반론 5
- **공격 대상**: "과거 산출물에 requested/effective가 없어도 태그, macro 근접, 생성 일시, git 이력으로 소급 귀속을 충분히 할 수 있다"는 주장.
- **더 단순한 설명**: 과거 산출물의 branch 구성은 태그/값이 아니라 실행 시점의 코드·설정·데이터·fold가 결정한다. 현재 문서에서 `ru90_shape_triple` 태그는 공식 7-branch를 연상시키지만 실제 5-branch 산출물이었고, macro `0.6226`과 `0.6227`의 근접은 우연으로 기록됐다. 기록된 5-branch 앵커 `0.4421`/`0.5553`/`0.6171`은 현재 코드로 재현되지 않고, 현재 5-branch 실측은 `0.4661`/`0.5408`/`0.6226`이다. 따라서 태그/값 단독은 구성을 특정하지 못한다. 더 단순한 설명은 과거 산출물은 branch 구성에 대해 unattributed일 뿐이라는 것이다.
- **무엇이 참이면 이 주장이 무너지는가**: 저장된 예측 파일에 `task_dir`, `fold_indices`, `fold_aurocs`, `fold_auroc_mean`, `per_fold`만 있으므로, 직접 특정 가능한 것은 task와 fold 및 저장된 수치다. branch list/weight/effective를 소급 귀속하려면 생성 시점의 코드 상태, 실행 스크립트/로그, 데이터/manifest, 그리고 그 코드가 단일 결정 경로를 가졌다는 확인이 필요하다. run log가 env export만 기록하면 requested만 보여 effective가 아니다. git commit만 있어도 그 commit에서 `src/`가 env를 읽지 않고 config 기본값만 사용했는지, config가 계산에 반영되는지 확인해야 한다. 현재 문서에는 과거 산출물의 code hash, data version, fold manifest hash, run log가 없으므로 "문서에 없음"이다.
- **이 반론의 한계**: 과거 산출물에 immutable sidecar나 effective state를 기록한 로그가 있고, 그 code hash가 단일 결정 경로에 대응하며, output hash가 일치한다면 제한된 inferred 귀속은 가능하다. 그러나 그것은 verified가 아니라 inference이며, 태그/값 단독으로 upgraded될 수 없다.

---

## Phase 2 — 교차

### A1 (감사)

## 점검 1: 제안 1 — canary token의 src/ 전달 경로 부재
- **확인하려는 것**: `requested_canary_token`이 wrapper에서 `src/`의 ensemble builder로 전달되는 경로가 존재하는지. `src/`가 env를 읽지 않는 현재 상태에서 canary token을 src/에 전달하는 메커니즘(CLI argument, config 필드, 파일)이 무엇인지.
- **문서에서 확인되는가**: 문서에 없음. `src/`가 `os.environ`/`getenv`을 읽지 않는 것은 확인됨(§8)이지만, canary token을 전달하는 대체 경로는 문서에 없다. 제안은 "calculation이 실제 branch selection 상태에 canary를 기록"한다고 하지만, canary token이 calculation에 도달하는 경로가 정의되지 않았다.
- **샌다면 나타날 증상**: canary token이 src/에 도달하지 못해 `effective_canary_token`이 null이거나 wrapper가 effective manifest에 값을 직접 복사. `requested_canary_token == effective_canary_token`이 성립하지만, 이는 canary가 계산에 도달한 것이 아니라 기록 주체가 값을 복제한 것. canary 불변식이 false pass. §8의 "src/가 env를 읽지 않아 ICF_* export가 전부 무효"와 동일한 결함이 canary 경로에도 재현.
- **확인 방법**: `src/`에서 canary token을 읽는 코드 경로 grep. CLI argument 파서, config 로더, 파일 리더 중 canary token을 처리하는 코드가 있는지 확인. 없으면 canary는 wrapper 자기 보고에 불과.

## 점검 2: 제안 1 — 불변식에 "통제 변수 변경 시 출력 변화" 조건 누락
- **확인하려는 것**: 불변식 수식(`requested_canary_token == effective_canary_token` AND `requested_effective_mismatch == false` AND `prediction_file_hash == sidecar.prediction_file_hash` AND `sidecar_hash == recomputed_hash`)에 "출력에 영향을 주도록 선언된 통제 변수를 변경하면 effective state 또는 `prediction_file_hash`가 변해야 한다"는 조건이 명시적으로 포함되는지.
- **문서에서 확인되는가**: 모순됨. 제안 본문은 이 조건을 문장으로 언급하지만, 불변식 수식에는 4개 조건뿐이다. "통제 변수 변경 시 출력 변화"는 불변식의 항이 아니다.
- **샌다면 나타날 증상**: canary가 일치하고 mismatch가 false인데도, weight를 0에서 nonzero로 바꿔도 `prediction_file_hash`가 변하지 않는 경우를 불변식이 통과. 이는 §6의 "가중치를 0으로 두어도 소수 4자리까지 값이 바뀌지 않는다"와 동일한 결함인데, 불변식이 잡지 못함. canary는 "기록 경로"를 검증할 뿐 "계산 경로"를 검증하지 못한다.
- **확인 방법**: 불변식 수식에서 paired 실험(제안 2)의 결과가 항으로 포함되는지 확인. 포함되지 않으면 canary 불변식과 paired 실험은 별개 검사이며, canary 통과만으로 설정 도달을 주장할 수 있는지 확인.

## 점검 3: 제안 1 — `sidecar_hash == recomputed_hash`의 자기 참조
- **확인하려는 것**: `recomputed_hash`가 sidecar 파일의 byte sequence를 다시 해시하는 것이라면, sidecar가 사후 수정되면 `recomputed_hash`도 같이 변해 불변식이 항상 성립하는지.
- **문서에서 확인되는가**: 문서에 없음. `recomputed_hash`의 계산 주체(orchestrator? 외부 감사?), 계산 대상(sidecar byte sequence? 파싱 후 재직렬화?), sidecar의 무결성 보장(불변 저장소, append-only)이 정의되지 않았다.
- **샌다면 나타날 증상**: sidecar가 사후에 수정되면(예: `effective_branch_list`를 5-branch에서 7-branch로 변경), `recomputed_hash`도 같이 변해 `sidecar_hash == recomputed_hash`가 성립. 불변식이 sidecar 무결성을 검증하지 못하고, §8의 "저장된 예측 파일에도 branch list·가중치·코드 해시가 기록돼 있지 않아 과거 산출물을 특정 구성에 귀속시킬 수 없다"의 사후 수정 경로를 막지 못함.
- **확인 방법**: `recomputed_hash`가 sidecar 파일 byte를 해시하는지, 내용을 파싱해서 재계산하는지 확인. sidecar가 예측 파일과 원자적으로 묶여 불변 저장소에 쓰이는지, 사후 수정이 가능한 파일시스템에 쓰이는지 확인.

## 점검 4: 제안 2 — CPU negative control이 production 경로(GPU bf16)의 설정 도달을 검증하지 못할 수 있음
- **확인하려는 것**: CPU fp32에서 실행한 3 arm의 설정 도달 행동이 GPU bf16 production 경로와 동일한지.
- **문서에서 확인되는가**: 문서에 없음. production 산출물 파일명이 `bf16`인 것은 확인됨(§1: `predictions/pathobench_*_ru90_shape_triple_official50_bf16.pt`)이지만, CPU fp32와 GPU bf16의 수치 경로가 weight 0 처리에서 동일한지 문서에 없다.
- **샌다면 나타날 증상**: CPU fp32에서는 weight 0 branch가 aggregation에서 완전히 제외되지만, GPU bf16에서는 normalization denominator에 0이 남아 수치 정밀도 차이로 결과가 달라지는 경우. CPU negative control이 통과해도 production 경로에서 설정 도달 실패. 또는 그 반대: CPU에서는 weight 0이 제외와 동등하지 않지만 GPU에서는 동등.
- **확인 방법**: CPU와 GPU에서 동일한 3 arm을 실행하고 `prediction_file_hash` 대조. 또는 production 경로(GPU bf16)에서 3 arm을 실행하는 회귀를 추가. CPU-only 회귀가 production 경로의 등가 입증이 아님을 명시.

## 점검 5: 제안 2 — "seed 고정"이 D-048 이후 결정론을 보장하지 않음
- **확인하려는 것**: "seed 고정, deterministic precision, cache off, fold manifest 고정"이 D-048 이후 학습 nondeterminism을 배제하는지.
- **문서에서 확인되는가**: 모순됨. D-048은 "결정론(`seed std 0.00000`)이 더는 성립하지 않는다"고 명시. RU-97 §8은 "동일 구성에서 seed만 바꿔 macro 폭 `0.0200`"을 실측. seed를 고정해도 GPU/BLAS/thread에 따라 결과가 달라질 수 있다.
- **샌다면 나타날 증상**: seed를 고정해도 `fold_aurocs`가 실행마다 달라져 exact equality가 성립하지 않음. 3 arm 비교가 판별 불가. 제안은 "결정론을 보장하지 못하면 exact equality를 쓰지 않고 해당 검사는 판별 불가/실행 무효로 둔다"고 하지만, 이 조건이 항상 발동하면 검사가 무의미.
- **확인 방법**: 동일한 seed·동일 구성으로 2회 실행하고 `fold_aurocs` 대조. 결정론이 성립하지 않으면 exact equality를 쓰지 않는 조건이 항상 발동하는지 확인.

## 점검 6: 제안 2 — 3 arm 실행을 위한 weight 변경 경로 부재
- **확인하려는 것**: `src/`가 env를 읽지 않는 현재 상태에서, 3 arm(B 제거, B weight 0.0, B weight nonzero)을 실행하는 경로가 존재하는지.
- **문서에서 확인되는가**: 문서에 없음. `src/`가 env를 읽지 않는 것은 확인됨(§8)이지만, weight를 변경하는 경로(config 파일 수정, CLI argument, 코드 상수 수정)는 문서에 없다. `src/models/config.py` 기본값이 `weight_sh=0.0`이고 YAML이 가중치를 설정하지 않는 것은 확인됨(§8)이지만, nonzero weight를 설정하는 경로가 있는지 확인되지 않았다.
- **샌다면 나타날 증상**: 3 arm을 실행하려면 `src/` 코드나 config 파일을 arm마다 수정해야 하는데, 이 경우 `code_hash`가 arm마다 달라져 "동일 code_hash에서 3 arm 비교"가 불가능. 또는 `src/`를 수정하지 않으면 3 arm이 모두 동일한 구성(5-branch, weight 0)으로 실행되어 설정 도달 실패를 검증하지 못함. 검사가 실행 무효.
- **확인 방법**: `src/`에서 weight를 읽는 코드 경로 확인. config 파일에서 weight를 설정하는 경로 확인. CLI argument에서 weight를 받는 경로 확인. 경로가 없으면 3 arm 실행이 불가능.

## 점검 7: 제안 3 — "code closure가 effective 구성을 유일하게 결정" 조건의 사후 검증 불가
- **확인하려는 것**: execution 시점 code closure가 effective 구성을 유일하게 결정하는지를 사후에 검증할 수 있는지.
- **문서에서 확인되는가**: 문서에 없음. code closure의 범위(`src/`, `scripts/`, config, data, environment, untracked file, generated code, package version)가 완전히 특정되지 않았다. §8은 "5-branch 드리프트의 원인(코드 변경 / 매니페스트 / 전처리 / 수치 정밀도). 미확인"이라고 명시.
- **샌다면 나타날 증상**: code closure가 `src/` + config만 포함하면, untracked file이나 package version 차이로 effective 구성이 달라지는 경우를 포착하지 못함. inferred 5-branch로 귀속했지만, 실제로는 untracked file이 SH/SJ를 켜고 있었던 경우. 또는 code closure가 완전하지만, 그 closure가 effective 구성을 "유일하게" 결정하는지(예: config 기본값과 YAML이 충돌하면 어느 쪽이 effective인지) 사후에 판별 불가.
- **확인 방법**: code closure의 파일 목록을 사후에 재구성하는 방법 확인. git status, untracked file, package version이 code closure에 포함되는지 확인. closure가 effective 구성을 유일하게 결정하는지(모순 없는 단일 값) 확인.

## 점검 8: 제안 3 — consistency check 기준 부재
- **확인하려는 것**: "tag, macro 근접, 문서 기대는 단독 근거가 아니라 consistency check만"이라고 하지만, consistency check의 기준(어떤 값이 일관된 것인지, 어떤 값이 모순인지, 근접성의 허용 범위)이 정의되어 있는지.
- **문서에서 확인되는가**: 문서에 없음. consistency check의 기준이 정의되지 않았다. §8은 macro `0.6226`이 `0.6227`에 가까운 것을 "우연"이라고 하지만, 이 판단의 기준(허용 오차, 근접성 정의)이 문서에 없다.
- **샌다면 나타날 증상**: macro `0.6226`이 7-branch `0.6227`에 "근접"하는 것이 consistency check에서 "일관된" 것인지 "모순"인지 판단 불가. `ru90_shape_triple` 태그가 7-branch를 연상시키는 것이 "일관된" 것인지 "모순"인지 판단 불가. consistency check가 주관적 판단으로 전락.
- **확인 방법**: consistency check의 기준(허용 오차, 근접성 정의, 모순 판정 규칙)이 문서에 명시되어 있는지 확인. 없으면 consistency check는 검증력이 없는 형식.

## 점검 9: 제안 3 — baseline unattributed 시 승격 비교(§4) 불가
- **확인하려는 것**: 공식 7-branch baseline `0.6227`이 unattributed가 되면, §4의 대응 per-fold Δ 계산이 가능한지.
- **문서에서 확인되는가**: 확인됨. PROJECT.md §3.2는 `0.6227`을 `unverified-hold`로 명시. 제안 3은 "공식 7-branch baseline `0.6227`은 verified로 승격되지 않는다"고 하지만, unattributed가 되면 승격 비교 자체가 불가능하다는 문제를 인식하지 못함.
- **샌다면 나타날 증상**: baseline이 unattributed이면, baseline의 `fold_aurocs`가 특정 구성에 귀속되지 않아 후보의 `fold_aurocs`와 짝짓는 것이 의미 없음. §4의 `Δ_i = (후보의 fold-i AUROC) − (기준선의 fold-i AUROC)` 계산이 구성 불일치에서 새어 나옴. 모든 승격 비교가 실행 무효. 제안 3은 baseline의 unattributed 상태를 인정하지만, 이 상태에서의 승격 비교가 불가능하다는 결론을 도출하지 못함.
- **확인 방법**: baseline이 unattributed인 상태에서 §4의 Δ 계산이 가능한지 확인. baseline의 `fold_aurocs`가 특정 구성에 귀속되지 않으면, 후보와 baseline의 fold 짝짓기가 유효한지 확인.

### T1 (이론)

## 이론적 병목 N1
- **가정**: 제안 1은 `requested` 표면이 `requested_branch_list`·`requested_weights`·`requested_mode`로 정규화되고, `effective` 상태가 `effective_branch_list`·`effective_weights`·`effective_mode`로 충분히 표현되며, `requested_canary_token == effective_canary_token`, `requested_effective_mismatch == false`, `prediction_file_hash` 바인딩, `sidecar_hash` 재계산이 “설정 도달”과 “산출물-기록 결합”을 외부에서 검증한다고 둔다. 즉, canary와 hash가 내부 metadata·파일 무결성 수준에서 일치하면 `effective` 기록이 신뢰 가능하다고 둔다.
- **이 가정이 깨지는 조건**: canary token이 requested manifest에서 effective manifest로 metadata 복사될 수 있을 때; effective writer가 config loader·설정 로더 상태만 관찰하고 실제 branch selection/score 산출 상태를 관찰하지 않을 때; requested·effective writer가 같은 프로세스·파일시스템·`execution_id`에서 사후 수정·재기록 가능할 때; `prediction_file_hash`가 파일 무결성만 검증하고 그 파일이 기록된 계산에서 나왔는지 검증하지 못할 때; branch list+weights+mode가 aggregation의 denominator, cache, feature selection, side effect, numerical path를 특정하지 못할 때; paired control variable change가 branch contribution 0, cancellation, numerical tolerance, uncontrolled variable 때문에 hash를 바꾸지 못할 때; code audit가 dynamic import·generated code·package·runtime state를 배제하지 못할 때; requested surface에 branch list/weights/mode로 정규화되지 않는 env·script·YAML 경로가 남을 때.
- **깨질 때 나타나는 관측 가능한 증상**: requested 7-branch인데 effective 5-branch이고 canary equality가 true; `requested_effective_mismatch`가 false인데 `ICF_*` export가 무효; `prediction_file_hash` 재계산이 일치하는데 산출물 태그·문서 기대·실제 구성이 불일치; 통제 변수를 변경해도 hash가 불변; weight 0 branch가 list에 남아 denominator·side effect로 출력을 바꾸지만 `effective_branch_list`만으로는 미구분; sidecar와 prediction file이 함께 사후 수정돼도 hash가 일치; canary가 metadata 전달 경로만 지나 effective manifest에 기록. 이 경우 제안 1의 `verified` 판정은 실행 무효 또는 판별 불가다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `src/`가 `os.environ`/`getenv`을 읽지 않아 실행 스크립트의 `ICF_*` export가 무효라는 관측; `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0` export에도 SMAD4 `0.4661`, PBRM1 `0.5408`으로 소수 4자리까지 동일하다는 관측; `run_ru90_shape_triple` 산출물이 공식 태그를 쓰지만 5-branch 산출물이라는 관측; macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이라는 관측. 동적 import·생성 코드·사후 수정 경로 자체에 대해서는 해당 관측 없음.

## 이론적 병목 N2
- **가정**: 제안 2는 CPU synthetic paired 3-arm에서 seed 고정·deterministic precision·cache off·fold manifest 고정이면 `per_fold`·`fold_aurocs`·`prediction_file_hash`의 exact equality가 설정 도달과 weight 0/branch 제외 동등성을 판정하는 불변식이라고 둔다. 또한 한 target branch B의 결과가 전체 branch 설정 도달로 일반화되고, CPU 경로가 production 경로와 동일하게 동작한다고 둔다.
- **이 가정이 깨지는 조건**: synthetic task/fold manifest가 production evaluation code path와 다르면; target branch B가 synthetic data에서 contribution 0·constant·다른 branch와 identical이면 nonzero weight가 출력을 바꾸지 못해 설정 무응답과 구분 불가하면; CPU와 production GPU/bf16의 수치·코드 분기·library·thread·reduction order가 다르면; exact equality를 보장하는 결정론이 없거나 사전 등록 허용 기준이 없으면; (1)과 (2)의 차이가 weight 0의 의미적 차이 아니라 정렬·rounding·정밀도에서 오면; (3)의 차이가 통제 변수가 아니라 uncontrolled variable에서 오면; 한 branch만 테스트해 branch-specific path를 놓치면; synthetic test가 production builder를 mock하면.
- **깨질 때 나타나는 관측 가능한 증상**: (3)이 (1)/(2)와 같는데 설정은 도달했고 branch contribution이 0; (1)과 (2)가 다르면 weight 0이 제외와 다름이 아니라 수치·정렬 차이; CPU 회귀가 production에서 반복되지 않음; exact equality가 실행마다 달라 판별 불가; B는 통과하지만 SH/SJ/BS는 무시되는 branch-specific 결함을 놓침; 검사가 false pass/fail을 낳거나, 결정론을 보장하지 못해 실행 무효가 됨. weight 0과 branch 제외의 동등성은 이 실험만으로 기전 미확인 상태로 남을 수 있다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `D-048`으로 `seed std 0.00000`이 더는 성립하지 않고, `RU-97` §8에서 동일 구성에서 seed만 바꿔 macro 폭 `0.0200`이 실측됐다는 관측; 보고 실행은 GPU 4 단독·bf16 산출물이라는 관측; §6은 가중치 0 export에도 소수 4자리 동일과 결정론적 동일성을 보고하지만, `per_fold`·`fold_aurocs`·`prediction_file_hash`의 exact equality 기준을 확인하지 않는다. CPU 경로가 production을 대표하는지에 대해서는 해당 관측 없음.

## 이론적 병목 N3
- **가정**: 제안 3은 과거 산출물에 대해 `evidence_list`와 execution-time code closure가 특정되면 `candidate_configurations`를 유한하게 제한하고, code closure가 effective 구성을 유일하게 결정하면 `inferred`로 승격할 수 있다고 둔다. requested는 script/log/env export closure로, effective는 code closure로 분리 추론할 수 있고, 태그·값·문서 기대는 consistency check로만 쓰이면 소급 귀속의 최대 범위가 정의된다고 둔다.
- **이 가정이 깨지는 조건**: code closure가 untracked files, generated code, package versions, data/manifest, environment, cache, runtime state를 포함하지 못하면; script/log/env export가 같은 broken wrapper에서 생성돼 독립 증거가 아니면; `candidate_configurations`를 열거하거나 상한을 둘 수 없으면; value proximity가 우연히 후보를 좁히는 것처럼 쓰이면; requested와 effective를 같은 closure로 묶으면; official baseline의 원래 code/data closure가 복원 불가능한데도 문서 기대가 verified처럼 작용하면; evidence_list가 requested만 특정하고 effective를 특정하지 못하는데 status가 `inferred`로 표시되면.
- **깨질 때 나타나는 관측 가능한 증상**: `inferred`가 부여되지만 untracked/generated code/package/data 버전 차이로 effective 구성이 다름; `ru90_shape_triple`이 5-branch candidate로 좁혀지지만 requested 7-branch와 effective 5-branch를 구분하지 못함; official 7-branch baseline `0.6227`이 `unverified-hold`인데도 태그·값·문서 기대가 verified처럼 인용됨; candidate set이 5/7/unknown으로 넓어 사용 불가; evidence_list가 script_path/run_log만 있어 requested만 추론되고 effective는 unattributed인데 `inferred`로 표시. 이 경우 소급 귀속은 판별 불가 또는 실행 무효다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: 저장된 예측 파일의 키에 branch list·가중치·코드 해시·데이터 버전이 없다는 관측; `run_ru90_shape_triple` 산출물이 공식 태그를 쓰지만 5-branch 산출물이라는 관측; macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이라는 관측; 5-branch 앵커 `0.4421`/`0.5553`이 현재 코드로 재현되지 않고 현재 5-branch 실측은 `0.4661`/`0.5408`이라는 관측; 공식 7-branch 재현 경로가 현재 없다는 관측.

### R1 (반증)

## 반론 1
- **공격 대상**: 제안 1의 `requested_canary_token == effective_canary_token`, `requested_effective_mismatch == false`, `prediction_file_hash == sidecar.prediction_file_hash`, `sidecar_hash == recomputed_hash` 불변식과 “canary 도달·paired hash 차이·독립 재계산이 외부 검증 신호”라는 주장.
- **더 단순한 설명**: canary token이 execution_id 같은 일반 metadata로만 흐르면, effective writer가 requested manifest나 config loader 값을 복사하는 것만으로 canary equality가 성립한다. 이 경우 “canary 일치, mismatch=false, hash 일치”라는 같은 관측은 설정 도달이 아니라 “기록 경로가 requested를 effective로 복사했다”는 더 단순한 설명으로도 낳는다. `prediction_file_hash == sidecar.prediction_file_hash`는 sidecar가 예측 파일 해시를 쓴 이상 자기 확인에 가깝고, `sidecar_hash == recomputed_hash`는 sidecar 파일 무결성만 확인한다. 사후에 sidecar를 작성하고 해시를 재계산해도 같은 관측을 낳는다. paired 실행에서 `prediction_file_hash`가 변하지 않는 관측은 설정 무시뿐 아니라 target branch 기여가 0, weight 0이 제외와 동등, 캐시/정밀도/라운드오프 차이로도 설명된다. 문서에는 가중치 0이 branch 제외와 동등한지와 aggregation 기전이 확인되지 않았다.
- **무엇이 참이면 이 주장이 무너지는가**: canary token이 branch selection/score 산출에 결합된 통제 변수로 주입되고, 그 변화가 `prediction_file_hash`를 바꾸는 경로가 코드 감사로 확인되며, effective writer가 config loader가 아니라 계산 시점 branch selection 상태를 관찰하는 것이 확인되면, canary 도달과 paired hash 변화는 외부 검증 신호가 된다.
- **이 반론의 한계**: 제안 1은 “통제 변수 변경이 paired 실행에서 artifact를 바꾸는지”와 “unrecorded switch가 없는지 코드 감사”를 신뢰 조건으로 두고 있다. 이 조건들이 실제로 충족되면, canary가 metadata 복사라는 더 단순한 설명은 배제된다.

## 반론 2
- **공격 대상**: 제안 1의 requested/effective 분리만으로 `effective_branch_list` 기록의 신뢰 조건이 충족된다는 주장.
- **더 단순한 설명**: effective manifest가 ensemble builder가 아니라 config loader/설정 로더에서 쓰이면, 기록은 로드된 config이지 계산에 쓰인 state가 아니다. 보고 §8은 `src/`가 env를 읽지 않음을 확인하지만, config 기본값/YAML 값이 branch selection에 실제로 반영되는지는 문서에 코드 대조가 없다. 따라서 requested와 effective가 일치하는 sidecar가 존재해도, 실제 계산은 하드코딩 상수, 캐시, import side effect, 전처리/집계 경로에서 다를 수 있다. 같은 “verified처럼 보이는 sidecar” 관측은 “기록 주체가 계산 상태를 관찰했다”보다 “기록 주체가 설정 상태를 관찰했다”는 더 단순한 설명으로도 성립한다.
- **무엇이 참이면 이 주장이 무너지는가**: ensemble builder가 `effective_branch_list`/`effective_weights`를 계산에 직접 사용하고, 가중치 0이 제외와 동등한지, unrecorded switch가 없는지 코드 감사로 확인되면, effective manifest는 계산 상태를 반영한다고 볼 수 있다.
- **이 반론의 한계**: 제안 1은 effective 기록 주체가 실제 branch selection/score 산출 상태를 관찰하는지와 unrecorded switch 감사를 신뢰 조건으로 명시한다. 이 감사가 완료되면 이 반론은 적용되지 않는다.

## 반론 3
- **공격 대상**: 제안 2의 CPU paired negative control과 exact equality 판정, 특히 “(3)이 (1)/(2)와 같으면 설정 도달 실패”와 “(1)과 (2)가 같으면 weight 0은 제외와 동등”이라는 규칙.
- **더 단순한 설명**: nonzero weight arm이 제거 arm과 weight 0 arm과 같은 `prediction_file_hash`를 낸 관측은 설정 무시보다 synthetic target branch가 계산에 기여하지 않거나, CPU 경로가 production builder/metric과 다름으로 설명된다. 반대로 (1)과 (2)가 다른 관측은 weight 0이 제외와 다름보다, branch 포함 여부에 따라 계산 순서, 정규화, 라운드오프 경로가 달라 bit-exact가 깨진 것일 수 있다. 문서에는 aggregation 공식이 없어 기전 미확인이다. D-048 이후 seed·적합 변동이 허용되고 RU-97 §8에서 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`이 실측됐으므로, exact equality는 실패(설정 무시)와 판별 불가/실행 무효(정밀도·결정론 문제)를 구분하지 못할 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 코드 감사로 aggregation이 nonzero weight에 반응하고, synthetic branch의 기여가 0이 아님을 확인하며, CPU 경로가 production과 동일한 builder/metric을 사용하고, deterministic bit-exact가 보장됨을 확인하면, exact equality 판정은 유효해진다.
- **이 반론의 한계**: 제안 2는 production 경로가 CPU에서 실행되지 않으면 blocked라고 한정하고, GPU 수치 경로 전체의 등가 입증이 아니라 설정 도달 smoke라고 명시한다. 결정론을 보장하지 못하면 exact equality를 쓰지 않고 판별 불가/실행 무효로 둔다고 했으므로, 이 반론은 오거부 가능성을 지적할 뿐이다.

## 반론 4
- **공격 대상**: 제안 3의 `inferred` 조건 “execution 시점 code closure가 특정되고 그 closure가 effective 구성을 유일하게 결정할 때만”과 소급 귀속 최대 범위 주장.
- **더 단순한 설명**: 같은 code closure라도 실행 시점 YAML, env, data/fold manifest, cache, package/environment, CLI arg가 다르면 effective 구성이 달라질 수 있다. 따라서 “code closure가 effective 구성을 유일하게 결정한다”는 관측은 실제로 모든 runtime input이 고정되어 있다는 더 강한 전제를 숨긴 것일 수 있다. 과거 `ru90_shape_triple`의 경우, 보고 §8의 5-branch 경로 확인은 현재 코드에 대한 것이지, 과거 execution 시점 code closure가 동일하다는 증거가 아니다. 같은 태그/값 관측은 과거 코드가 env를 읽었을 가능성, 다른 config/데이터/캐시였을 가능성으로도 설명된다.
- **무엇이 참이면 이 주장이 무너지는가**: `evidence_list`가 code closure뿐 아니라 실행 시점 config, env, data/fold manifest, package/environment, cache state, run log를 포함하고, 코드 경로에 unrecorded switch가 없음을 확인하면, closure가 effective 구성을 유일하게 결정할 수 있다.
- **이 반론의 한계**: 제안 3은 `evidence_list`에 config files, run log, timestamps, values를 두고 `candidate_configurations`와 `contradictions`를 기록한다. 과거 실행의 run log/git commit/config closure가 실제로 보존되어 있고 모순이 없으면 inferred candidate는 성립할 수 있다. 보존 여부는 문서에 없다.

## 반론 5
- **공격 대상**: 제안 3의 “requested는 script/log/env export가 보존돼야 추론 가능” 주장.
- **더 단순한 설명**: `src/`가 env를 읽지 않는 현재 코드에서는 script의 `ICF_*` export가 `src/`에 도달하지 않으므로, run log의 env export는 “src가 요청받은 구성”이 아니라 “스크립트가 의도한 구성”이다. 같은 run log 관측—`ICF_FIXED_HEAD_SH_WEIGHT=1.0` export—은 effective 7-branch가 아니라 requested 7-branch/effective 5-branch라는 더 단순한 설명으로도 성립한다. 과거 코드가 현재와 다르면 더 그렇다.
- **무엇이 참이면 이 주장이 무너지는가**: 과거 execution 시점 `src/`가 env를 읽었고, 그 env가 branch selection에 반영됨이 코드 closure/run log로 확인되면, script/log env export는 requested 추론에 유효하다.
- **이 반론의 한계**: 제안 3은 requested와 effective를 분리하고, effective는 code closure가 결정해야 추론 가능하다고 한정한다. 따라서 script export만으로 effective를 주장하지 않는다면, 이 반론은 오귀속 가능성을 줄일 뿐이다.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

## A1 점검 1
- **판정**: 수용·수정
- **근거**: 보고 §8에서 `src/`는 `os.environ`/`getenv`을 읽지 않는 파일이 0개다. canary를 env로 전달하면 `src/`에 도달하지 못한다. canary 전달 경로(CLI argument, config 필드, 파일)는 문서에 없다.
- **수정**: canary token을 `src/`가 실제로 읽는 경로로 전달하도록 설계한다. `src/`가 canary를 읽는 코드 경로가 확인되기 전까지 `requested_canary_token == effective_canary_token`은 신뢰 신호가 아니다. canary 전달 경로 부재는 실행 무효 조건이다.

## A1 점검 2
- **판정**: 수용·수정
- **근거**: 제안 1의 불변식 수식은 4개 항(`canary equality`, `mismatch=false`, `prediction_file_hash` 바인딩, `sidecar_hash` 재계산)이다. "통제 변수 변경 시 출력 변화"는 문장만 있고 항이 아니다. 보고 §6에서 가중치 0으로 두어도 소수 4자리까지 값이 바뀌지 않았다.
- **수정**: 불변식 수식에 "출력에 영향을 주도록 선언된 통제 변수를 변경하면 `effective_branch_list`·`effective_weights`·`effective_mode` 중 하나 또는 `prediction_file_hash`가 변해야 한다"를 항으로 포함한다. 이 항이 없으면 canary는 기록 경로만 검증하고 계산 경로를 검증하지 못한다.

## A1 점검 3
- **판정**: 수용·수정
- **근거**: `recomputed_hash`의 계산 주체, 계산 대상, sidecar 무결성 보장은 문서에 없다. sidecar가 사후 수정되면 `recomputed_hash`도 같이 변해 불변식이 항상 성립할 수 있다.
- **수정**: `recomputed_hash`는 sidecar 파일 byte sequence가 아니라, sidecar 내용을 파싱 후 canonical serialization을 재계산한 값으로 정의한다. 계산 주체는 sidecar를 작성한 프로세스가 아닌 외부 검증 스크립트다. sidecar와 prediction file은 원자적으로 묶여 불변 저장소(append-only 또는 WORM)에 쓰여야 하며, 사후 수정이 가능한 파일시스템에 쓰이면 `sidecar_hash == recomputed_hash`는 무결성을 검증하지 못한다.

## A1 점검 4
- **판정**: 판별 불가
- **근거**: production 산출물 파일명이 `bf16`인 것은 확인됐다(보고 §1). CPU fp32와 GPU bf16의 수치 경로가 weight 0 처리에서 동일한지는 문서에 없다.
- **수정**: CPU negative control은 production 경로(GPU bf16)의 등가 입증이 아니라 설정 도달 smoke로 한정한다. production 경로 검증은 GPU bf16에서 동일한 3 arm을 실행하는 별도 회귀가 필요하다. CPU-only 회귀가 production을 대표하지 못함을 명시한다.

## A1 점검 5
- **판정**: 수용·수정
- **근거**: `PROJECT.md` §1에서 `D-048`은 결정론(`seed std 0.00000`)이 더는 성립하지 않는다고 명시한다. `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. seed 고정만으로 GPU/BLAS/thread 변동이 배제되지 않는다.
- **수정**: exact equality를 쓰기 전, 동일 seed·동일 구성·동일 경로로 2회 실행하고 `fold_aurocs`·`prediction_file_hash` 대조로 결정론을 확인한다. 결정론이 성립하지 않으면 exact equality를 쓰지 않고 해당 검사는 판별 불가/실행 무효로 둔다. 결정론 확인 자체를 회귀 스위트의 선행 조건으로 포함한다.

## A1 점검 6
- **판정**: 판별 불가
- **근거**: `src/`가 env를 읽지 않는 것은 확인됐다(보고 §8). weight를 변경하는 경로(config 파일 수정, CLI argument, 코드 상수 수정)는 문서에 없다. nonzero weight를 설정하는 경로가 존재하는지 확인되지 않았다.
- **수정**: 3 arm 비교를 "동일 code_hash에서 수행"하는 전제를 버린다. arm별 code_hash를 기록하고 비교한다. 또는 `src/`가 weight를 읽는 경로(config 파일, CLI argument)가 확인된 뒤 3 arm을 실행한다. weight 변경 경로가 확인되기 전까지 3 arm 실행은 실행 무효다.

## A1 점검 7
- **판정**: 수용·수정
- **근거**: code closure의 범위(`src/`, `scripts/`, config, data, environment, untracked file, generated code, package version)는 문서에 완전 특정되지 않았다. 보고 §8은 5-branch 드리프트 원인을 코드 변경/매니페스트/전처리/수치 정밀도로 열어두고 "미확인"이라고 명시한다.
- **수정**: `inferred` 조건을 "execution 시점 code closure가 특정되고, 그 closure가 unrecorded switch 없이 effective 구성을 유일하게 결정함을 코드 감사로 확인"으로 강화한다. 코드 감사로 확인되지 않으면 `inferred`가 아니라 `candidate_configurations` 또는 `unattributed`로 둔다.

## A1 점검 8
- **판정**: 판별 불가
- **근거**: consistency check의 기준(허용 오차, 근접성 정의, 모순 판정 규칙)은 문서에 없다. 보고 §8에서 macro `0.6226`이 `0.6227`에 가까운 것을 "우연"이라고 했지만, 이 판단의 기준은 문서에 없다.
- **수정**: consistency check는 단독 근거가 아니라, 독립 증거와 모순되는지 확인하는 데만 사용한다. 기준이 없으면 consistency check는 검증력이 없는 형식이다. 허용 오차, 근접성 정의, 모순 판정 규칙은 사전 등록해야 하며, 문서에 없는 수치를 지어 쓰지 않는다.

## A1 점검 9
- **판정**: 수용·수정
- **근거**: `PROJECT.md` §3.2는 `0.6227`을 `unverified-hold`로 명시한다. baseline의 `fold_aurocs`가 특정 구성에 귀속되지 않으면, 후보의 `fold_aurocs`와 짝짓는 §4의 `Δ_i` 계산은 구성 불일치에서 새어 나온다.
- **수정**: baseline이 `unattributed`이면 §4의 승격 비교를 보류한다. 새 스키마로 baseline을 재실행하거나 소급 검증을 통과한 뒤 승격 비교를 수행한다. baseline `unverified-hold` 상태에서는 `verified` 기준선 없이 승격 비교를 수행하지 않는다.

## T1 N1
- **판정**: 수용·수정
- **근거**: canary token이 metadata 복사될 수 있고, effective writer가 config loader일 수 있으며, `src/`가 env를 읽지 않는 것은 확인됐다(보고 §8). `prediction_file_hash`는 파일 무결성만 검증하고 그 파일이 기록된 계산에서 나왔는지 검증하지 못할 수 있다. branch list+weights+mode가 aggregation의 denominator, cache, feature selection, side effect, numerical path를 특정하지 못할 수 있다.
- **수정**: canary token이 branch selection/score 산출에 결합된 통제 변수로 주입되고, 그 변화가 `prediction_file_hash`를 바꾸는 경로가 코드 감사로 확인되어야 한다. effective writer가 config loader가 아니라 계산 시점 branch selection 상태를 관찰하는 것이 확인되어야 한다. requested/effective writer가 분리되고, 사후 수정이 불가능한 불변 저장소에 쓰여야 한다. `prediction_file_hash`는 파일 무결성뿐 아니라 그 파일이 기록된 계산에서 나왔는지 검증한다. branch list+weights+mode가 aggregation의 모든 구성 요소를 특정하지 못하면, 해당 차원은 `unattributed`로 둔다. paired control variable change가 hash를 바꾸지 못하면, 설정 무시, branch contribution 0, weight 0 동등, 캐시/정밀도 차이 중 무엇인지 구분 불가다. code audit가 dynamic import·generated code·package·runtime state를 배제하지 못하면 `code_hash` 완전성은 `미확인`으로 표시한다.

## T1 N2
- **판정**: 수용·수정
- **근거**: synthetic task/fold manifest가 production evaluation code path와 다를 수 있다. target branch B가 synthetic data에서 contribution 0이면 nonzero weight가 출력을 바꾸지 못해 설정 무응답과 구분 불가하다. CPU와 production GPU/bf16의 수치·코드 분기·library·thread·reduction order가 다를 수 있다. `D-048` 이후 seed·적합 변동이 허용되고 `RU-97` §8에서 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`이 실측됐다.
- **수정**: synthetic paired 3-arm은 production evaluation code path를 통과해야 한다. target branch B가 synthetic data에서 contribution 0이 아님을 확인한다. CPU와 production 경로가 동일하지 않으면 CPU 회귀는 smoke로 한정한다. exact equality를 쓰기 전 결정론을 확인한다. (1)과 (2)의 차이가 weight 0의 의미적 차이 아니라 정렬·rounding·정밀도에서 올 수 있으므로, bit-exact가 아니라 사전 등록 허용 기준이 필요하다. (3)의 차이가 통제 변수가 아니라 uncontrolled variable에서 올 수 있으므로, 통제 변수 외 변수를 고정한다. 한 branch만 테스트해 branch-specific path를 놓칠 수 있으므로, 모든 branch에 대해 테스트한다. synthetic test가 production builder를 mock하면 false pass/fail이므로 mock을 금지한다.

## T1 N3
- **판정**: 수용·수정
- **근거**: code closure가 untracked files, generated code, package versions, data/manifest, environment, cache, runtime state를 포함하지 못할 수 있다. script/log/env export가 같은 broken wrapper에서 생성돼 독립 증거가 아닐 수 있다. `candidate_configurations`를 유한하게 열거하거나 상한을 둘 수 없을 수 있다. value proximity가 우연히 후보를 좁히는 것처럼 쓰일 수 있다. requested와 effective를 같은 closure로 묶을 수 있다. official baseline의 원래 code/data closure가 복원 불가능한데도 문서 기대가 verified처럼 작용할 수 있다. evidence_list가 requested만 특정하고 effective를 특정하지 못하는데 status가 `inferred`로 표시될 수 있다.
- **수정**: code closure는 untracked files, generated code, package versions, data/manifest, environment, cache, runtime state를 포함해야 한다. script/log/env export가 독립 증거인지 확인한다. `candidate_configurations`를 유한하게 열거한다. value proximity는 consistency check로만 사용한다. requested와 effective를 분리한다. official baseline은 `unverified-hold`로 유지한다. evidence_list가 requested만 특정하면 effective는 `unattributed`로 둔다.

## R1 반론 1
- **판정**: 수용·수정
- **근거**: canary token이 execution_id 같은 일반 metadata로만 흐르면, effective writer가 requested manifest나 config loader 값을 복사하는 것만으로 canary equality가 성립한다. `prediction_file_hash == sidecar.prediction_file_hash`는 sidecar가 예측 파일 해시를 쓴 이상 자기 확인에 가깝다. `sidecar_hash == recomputed_hash`는 sidecar 파일 무결성만 확인한다. paired 실행에서 `prediction_file_hash`가 변하지 않는 관측은 설정 무시뿐 아니라 target branch 기여가 0, weight 0이 제외와 동등, 캐시/정밀도/라운드오프 차이로도 설명된다.
- **수정**: canary token이 branch selection/score 산출에 결합된 통제 변수로 주입되고, 그 변화가 `prediction_file_hash`를 바꾸는 경로가 코드 감사로 확인되어야 한다. effective writer가 config loader가 아니라 계산 시점 branch selection 상태를 관찰하는 것이 확인되어야 한다. `prediction_file_hash`는 파일 무결성뿐 아니라 그 파일이 기록된 계산에서 나왔는지 검증한다. paired 실행에서 hash가 변하지 않으면, 설정 무시, branch contribution 0, weight 0 동등, 캐시/정밀도 차이 중 무엇인지 구분 불가다.

## R1 반론 2
- **판정**: 수용·수정
- **근거**: effective manifest가 ensemble builder가 아니라 config loader/설정 로더에서 쓰이면, 기록은 로드된 config이지 계산에 쓰인 state가 아니다. 보고 §8은 `src/`가 env를 읽지 않음을 확인했지만, config 기본값/YAML 값이 branch selection에 실제로 반영되는지는 문서에 코드 대조가 없다.
- **수정**: effective manifest는 ensemble builder가 계산 시점 branch selection/score 산출 상태를 관찰하여 작성해야 한다. config loader가 작성하면 설정 상태만 기록한다. ensemble builder가 `effective_branch_list`/`effective_weights`를 계산에 직접 사용하는지, 가중치 0이 제외와 동등한지, unrecorded switch가 없는지 코드 감사로 확인되어야 한다. 확인 전까지 effective manifest의 신뢰도는 `미확인`이다.

## R1 반론 3
- **판정**: 수용·수정
- **근거**: nonzero weight arm이 제거 arm과 weight 0 arm과 같은 `prediction_file_hash`를 낸 관측은 설정 무시보다 synthetic target branch가 계산에 기여하지 않거나, CPU 경로가 production builder/metric과 다름으로 설명된다. (1)과 (2)가 다른 관측은 weight 0이 제외와 다름보다, branch 포함 여부에 따라 계산 순서, 정규화, 라운드오프 경로가 달라 bit-exact가 깨진 것일 수 있다. `D-048` 이후 seed·적합 변동이 허용되고 `RU-97` §8에서 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`이 실측됐다.
- **수정**: nonzero weight arm이 hash를 바꾸지 않으면, 설정 무시, branch contribution 0, CPU/production 경로 차이, weight 0 동등 중 무엇인지 구분 불가다. (1)과 (2)가 다르면, weight 0 의미적 차이, 계산 순서/정규화/라운드오프 차이 중 무엇인지 구분 불가다. exact equality를 쓰기 전 결정론을 확인한다. 결정론이 성립하지 않으면 판별 불가/실행 무효로 둔다.

## R1 반론 4
- **판정**: 수용·수정
- **근거**: 같은 code closure라도 실행 시점 YAML, env, data/fold manifest, cache, package/environment, CLI arg가 다르면 effective 구성이 달라질 수 있다. 보고 §8의 5-branch 경로 확인은 현재 코드에 대한 것이지, 과거 execution 시점 code closure가 동일하다는 증거가 아니다.
- **수정**: code closure가 effective 구성을 유일하게 결정하려면, 실행 시점 YAML, env, data/fold manifest, cache, package/environment, CLI arg가 모두 고정되어야 한다. 과거 execution 시점 code closure가 현재 코드와 동일하다는 증거가 없으면, 과거 산출물은 `unattributed` 또는 `candidate_configurations`로 둔다.

## R1 반론 5
- **판정**: 수용·수정
- **근거**: `src/`가 env를 읽지 않는 현재 코드에서는 script의 `ICF_*` export가 `src/`에 도달하지 않으므로, run log의 env export는 "src가 요청받은 구성"이 아니라 "스크립트가 의도한 구성"이다. 과거 코드가 현재와 다르면 더 그렇다.
- **수정**: run log의 env export는 requested만 추론 가능하고, effective 추론은 code closure가 결정해야 한다. 과거 execution 시점 `src/`가 env를 읽었고, 그 env가 branch selection에 반영됨이 코드 closure/run log로 확인되면, script/log env export는 requested 추론에 유효하다. 확인 불가하면 requested도 `unattributed`로 둔다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)

- **현재 값 앵커 기반 오염 검사는 유효한 통과/실패 판정 도구가 아니다.**  
  근거: `PROJECT.md` §3.4의 `D-050`은 값 앵커 오염 검사를 폐지하고 provenance 기록 대조로 재설계하도록 정한다. 보고 §6은 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0` export에도 SMAD4 `0.4661`, PBRM1 `0.5408`으로 소수 4자리까지 동일했고, 기록된 5-branch 앵커 `0.4421`/`0.5553`과 다르다고 확인한다. 보고 §8은 `src/`가 `os.environ`/`getenv`을 읽지 않는 파일이 0개라고 확인한다.

- **실행 스크립트의 `ICF_*` export는 `src/` 평가 경로에 도달하지 않는다.**  
  근거: 보고 §8의 코드 확인에서 `src/`는 환경변수를 읽지 않는 파일이 0개이고, `scripts/eval_seal_tasks.sh`에 `ICF_` 참조가 0건이며, `scripts/evaluate_pure.py`의 `ICF_SHAPE_SCREEN_ONLY` 언급은 독스트링뿐이다. 보고 §6은 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`가 출력에 아무 영향이 없다고 확인한다.

- **`run_ru90_shape_triple` 산출물은 공식 7-branch score로 사용할 수 없다.**  
  근거: 보고 §8은 `eval_v121.sh`가 정의상 5-branch arm이고, `src/models/config.py` 기본값이 `weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`이며, 사용된 YAML은 브랜치 가중치를 설정하지 않는다고 확인한다. `src/`가 env를 읽지 않으므로 스크립트 export는 계산에 도달하지 않는다. 따라서 `ru90_shape_triple_official50_bf16` 산출물은 공식 7-branch SH/SJ 활성화 경로로 만들었다고 볼 수 없다.

- **공식 7-branch 기준선 `0.6227`은 `verified` 기준선이 아니다.**  
  근거: `PROJECT.md` §3.2는 `D-050` 이후 이 값을 `unverified-hold`로 명시한다. 보고 §8은 `SH`/`SJ`를 켜는 문서화된 수단이 동작하지 않아 공식 7-branch 기준선을 재현할 경로가 현재 없다고 확인한다.

- **저장된 예측 파일은 구성 재구성에 불충분하다.**  
  근거: `PROJECT.md` 배경과 `C-20260918-10` 보고서에서 저장된 예측 파일의 키는 `task_dir`, `fold_indices`, `fold_aurocs`, `fold_auroc_mean`, `per_fold`뿐이며, branch list·가중치·코드 해시·데이터 버전이 없다고 한다. 따라서 과거 산출물을 특정 구성에 귀속시키는 식별자가 현재 파일에 없다.

- **`effective` 기록은 requested/env/config 선언값이 아니라 계산 시점 상태를 기록해야 한다.**  
  근거: `ICF_*` export가 `src/`에 도달하지 않고, 스크립트 머리말·태그·env 요청과 실제 산출 구성이 분리될 수 있다는 관측이 있다. `run_ru90_shape_triple`은 공식 7-branch를 연상시키는 태그를 쓰지만 공식 7-branch 활성화 경로가 아님을 보여준다.

- **sidecar 존재만으로는 `verified`가 아니다.**  
  근거: sidecar를 작성하는 코드와 산출물을 작성하는 코드가 동일할 수 있고, requested 값을 effective로 복사하거나 config loader 값을 기록할 수 있다. 보고 §8은 태그·문서 기대·env export와 실제 산출 경로가 분리될 수 있음을 확인한다.

- **provenance에는 코드·데이터·fold manifest·예측 파일 식별자가 필요하다.**  
  근거: 현재 예측 파일에는 branch list·가중치·코드 해시·데이터 버전이 없다. `PROJECT.md` §4.1은 대응 per-fold Δ의 정밀도가 fold 짝짓기에 의존한다고 명시한다. `D-048` 이후 seed·적합 변동이 허용되고 `RU-97` §8에서 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`이 실측됐으므로, 산출물 자체를 특정할 식별자도 필요하다.

- **값 앵커가 제공하던 수치 드리프트/결정론 기능은 provenance 대조만으로 대체되지 않는다.**  
  근거: provenance 대조는 기록된 구성이 일치하는지 확인하지만, 동일 구성에서 값이 실제로 동일한지는 확인하지 않는다. `D-048`으로 `seed std 0.00000`이 더는 성립하지 않고, `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다.

- **과거 산출물은 태그·값 단독으로 구성을 특정할 수 없다.**  
  근거: `ru90_shape_triple`은 공식 7-branch를 연상시키는 태그를 쓰지만 보고 §8에서 공식 7-branch 활성화 경로가 아님으로 확인됐다. macro `0.6226`이 `0.6227`에 가까운 것은 우연으로 기록됐다. 기록된 5-branch 앵커도 현재 코드로 재현되지 않는다.

- **회귀는 설정 도달과 출력 민감성을 검사해야 한다.**  
  근거: 보고 §6은 가중치를 0으로 두어도 소수 4자리까지 값이 바뀌지 않았다. 보고 §8은 `src/`가 env를 읽지 않아 설정 표면 전체가 무효라고 확인한다. 따라서 sidecar 필드 존재만 검사하면 이 결함을 잡지 못한다.

- **소급 귀속은 증거 클로저가 허용하는 후보 집합으로 제한된다.**  
  근거: 과거 산출물에는 requested/effective 기록이 없고, 저장 예측 파일에도 branch list·가중치·코드 해시·데이터 버전이 없다. 보고 §8은 과거 `ru90_shape_triple` 실행이 당시 shape margin을 포함했는지를 아직 확인하지 않은 것으로 남긴다.

## 반박 (증거가 반박함)

- **“실행 스크립트의 `ICF_*` export가 branch list·가중치·모드를 통제한다”는 주장은 반박된다.**  
  근거: 보고 §8에서 `src/`는 환경변수를 읽지 않는 파일이 0개라고 확인됐다. 보고 §6에서 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0`를 export해도 SMAD4·PBRM1 값이 소수 4자리까지 바뀌지 않았다.

- **`run_ru90_shape_triple`이 공식 7-branch score를 만들었다는 주장은 반박된다.**  
  근거: 보고 §8에서 해당 경로는 `eval_v121.sh`의 5-branch arm을 거치며, `config.py` 기본값과 YAML이 `SH`/`SJ`/`BS` 가중치를 켜지 않는다. `src/`가 env를 읽지 않으므로 스크립트 export는 계산에 도달하지 않는다.

- **기록된 5-branch 앵커가 현재 코드로 재현된다는 주장은 반박된다.**  
  근거: 보고 §6·§8에서 현재 5-branch 경로 실측은 SMAD4 `0.4661`, PBRM1 `0.5408`, macro `0.6226`이고, 기록된 앵커는 `0.4421`, `0.5553`, `0.6171`이다. 두 실행이 결정론적으로 동일했으므로 잡음이 아니다.

- **공식 7-branch 기준선 `0.6227`이 `verified`라는 주장은 반박된다.**  
  근거: `PROJECT.md` §3.2는 `D-050` 이후 이 값을 `unverified-hold`로 명시한다. 보고 §8은 공식 7-branch를 재현할 문서화 경로가 현재 없다고 확인한다.

- **스크립트 머리말·태그·env export만으로 산출물 구성을 특정할 수 있다는 주장은 반박된다.**  
  근거: `run_ru90_shape_triple.sh` 머리말은 `SCREEN_ONLY`가 5-branch BASE arm을 만든다고 적지만 관측은 7-branch macro에 가까웠다. 코드 확인 결과 해당 산출물은 공식 7-branch 활성화 경로가 아니었다. 태그 `official50`도 실제 구성과 어긋났다.

- **macro `0.6226`과 `0.6227`의 근접성이 7-branch 귀속을 지지한다는 주장은 반박된다.**  
  근거: 보고 §8은 macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이라고 확인한다. 코드 확인상 해당 산출물은 공식 7-branch 활성화 경로가 아니다.

- **provenance 대조만으로 값 앵커의 모든 기능을 대체한다는 주장은 반박된다.**  
  근거: provenance 대조는 구성 일치 확인에 해당하지만, 동일 구성에서 seed·적합·수치 정밀도·환경 차이로 값이 달라지는 드리프트/결정론 문제를 자동으로 확인하지 않는다. `D-048` 이후 seed·적합 변동이 허용되고 `RU-97` §8에서 동일 구성 seed 변경으로 macro 폭 `0.0200`이 실측됐다.

- **`requested`/`effective` 분리만으로도 `effective_branch_list`를 신뢰할 수 있다는 강한 주장은 반박된다.**  
  근거: `effective` 기록 주체가 결국 같은 실행 코드일 수 있고, requested 값을 effective로 복사하거나 config loader 값을 기록할 수 있다. 보고 §8은 env 요청과 실제 계산 경로가 분리될 수 있음을 확인한다.

- **가중치 0 export 실험이 “가중치 0 = branch 제외”를 확인했다는 주장은 반박된다.**  
  근거: 보고 §8은 `src/`가 env를 읽지 않으므로 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT` export가 계산에 도달하지 않는다고 확인한다. 따라서 그 실험은 env 경로 무효를 보여줄 뿐, config 내부 가중치 0의 계산 의미를 확인한 것이 아니다.

## 판별 불가 (증거가 가르지 못함)

- **현재 5-branch 실측값이 기록된 5-branch 앵커와 다른 원인은 판별 불가하다.**  
  보고 §8은 원인을 코드 변경 / 매니페스트 / 전처리 / 수치 정밀도로 열어두고 “미확인”이라고 명시한다. 현재 증거는 값 불일치만 확인하고, 어떤 요인이 원인이었는지는 가르지 못한다.

- **`config.py`의 가중치 0이 실제로 branch 계산 제외와 동등한지는 판별 불가하다.**  
  보고 §6에서 env 가중치 0이 출력에 영향을 주지 않았지만, 이는 `src/`가 env를 읽지 않기 때문일 수 있다. `config.py` 기본값 `weight_sh=0.0` 등이 앙상블 계산에서 branch를 실제로 제외하는지, 아니면 가중치 0으로 계산에 포함되는지는 문서에 코드 대조 결과가 없다. 기전 미확인.

- **`run_ru90_shape_triple` 산출물의 정확한 effective branch list는 판별 불가하다.**  
  공식 7-branch가 아님은 뒷받침되지만, zero-weight `SH`/`SJ`/`BS`가 계산에서 완전히 제외되는지, 아니면 가중치 0으로 포함되는지는 코드 감사 전 판별 불가하다.

- **과거 `ru90_shape_triple` 실행이 당시에는 shape margin을 포함했는지는 판별 불가하다.**  
  보고 §8은 “과거 `ru90_shape_triple` 실행이 당시에는 shape margin을 포함했는지”를 아직 확인하지 않은 것으로 남긴다. 현재 코드 확인은 과거 실행 시점의 구성을 특정하지 못한다.

- **공식 7-branch 기준선 `0.6227`의 원래 산출물이 어떤 구성·코드·데이터에서 나왔는지는 판별 불가하다.**  
  과거 산출물에 sidecar가 없고, 저장 예측 파일에도 branch list·가중치·코드 해시·데이터 버전이 없다. `PROJECT.md`는 이 기준선을 `unverified-hold`로 둔다.

- **canary token이 `src/` 계산 경로로 전달될 수 있는지는 판별 불가하다.**  
  `src/`가 env를 읽지 않는 것은 확인됐지만, canary token을 전달하는 CLI argument, config 필드, 파일 등 대체 경로는 문서에 없다. canary 전달 경로가 확인되기 전까지 canary equality는 신뢰 신호가 아니다.

- **CPU negative control이 production GPU bf16 경로의 설정 도달을 검증하는지는 판별 불가하다.**  
  production 산출물 파일명이 `bf16`인 것은 확인됐지만, CPU fp32와 GPU bf16의 수치 경로가 weight 0 처리에서 동일한지는 문서에 없다.

- **D-048 이후 full pipeline에서 exact equality 또는 exact hash가 유효한지는 판별 불가하다.**  
  `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. seed 고정만으로 GPU/BLAS/thread 변동이 배제되는지는 문서에 없다. 허용 기준이 없으면 exact hash 불일치를 구성 실패, seed 변동, 수치 정밀도 차이 중 무엇으로 구분할 수 없다.

- **과거 산출물의 code closure가 effective 구성을 유일하게 결정하는지는 판별 불가하다.**  
  untracked files, generated code, package versions, data/manifest, environment, cache, runtime state가 실행에 영향을 줄 수 있으나, 문서에는 과거 execution 시점 closure가 복원됐다는 확인이 없다.

- **consistency check의 기준은 문서에 없다.**  
  태그·macro 근접·문서 기대를 consistency check로만 쓰려면 허용 오차, 근접성 정의, 모순 판정 규칙이 필요하지만 문서에 없다. 보고 §8에서 macro `0.6226`이 `0.6227`에 가까운 것을 “우연”이라고 했지만, 그 판단의 기준은 문서에 없다.

- **sidecar hash 재계산이 사후 수정을 막는지는 판별 불가하다.**  
  `recomputed_hash`의 계산 주체, 계산 대상, sidecar 무결성 보장(append-only/WORM 등)이 문서에 없다. sidecar와 prediction file이 사후에 함께 수정되면 hash 불일치가 잡히지 않을 수 있다.

- **3-arm 실험을 위한 weight 변경 경로가 존재하는지는 판별 불가하다.**  
  `src/`가 env를 읽지 않는 것은 확인됐지만, weight를 변경하는 config 파일, CLI argument, 코드 상수 경로가 문서에 없다. nonzero weight를 설정하는 경로가 확인되지 않았다.

- **raw branch inputs가 저장되어 독립 재계산이 가능한지는 문서에 없다.**  
  저장 예측 파일의 키에는 `per_fold`가 있지만, raw branch inputs가 저장되어 있는지, effective weights와 aggregation code로 독립 재계산이 가능한지는 문서에 명시되지 않았다.

- **회귀 스위트를 GPU 없이 돌 수 있는지는 문서에 없다.**  
  production 평가 경로가 CPU에서 동일하게 실행되는지, CPU 경로가 production builder/metric과 동일한지는 문서에 없다.

## 실행 무효 (실행·계측 결함으로 판단 불가)

- **현재 값 앵커 기반 오염 검사 실행은 무효다.**  
  이유: `src/`가 env를 읽지 않아 intended base/screen arm을 통제할 경로가 없고, 기록된 5-branch 앵커가 현재 코드로 재현되지 않는다. 따라서 4자리 AUROC 일치로 오염 검사 통과를 주장할 수 없다.

- **재설계 완료 전 오염 검사 통과를 주장하는 모든 실행은 무효다.**  
  이유: `PROJECT.md` §3.4 `D-050`이 명시한다.

- **공식 7-branch 기준선 `0.6227`을 `verified` 기준선으로 사용하는 실행은 무효다.**  
  이유: `PROJECT.md` §3.2는 `unverified-hold`로 명시하고, 보고 §8은 공식 7-branch 재현 경로가 현재 없다고 확인한다.

- **`ru90_shape_triple_official50_bf16` 산출물을 공식 7-branch score로 사용하는 실행은 무효다.**  
  이유: 보고 §8에서 해당 산출물은 공식 7-branch 활성화 경로가 아님으로 확인됐다.

- **공식 7-branch score를 입력으로 요구하는 후보 실행은 현재 무효다.**  
  이유: 보고 §8은 공식 7-branch score를 만들 문서화 경로가 현재 없다고 확인한다.

- **기록된 5-branch 앵커와 현재 5-branch 실측값의 4자리 대조로 구성 동일성을 판정하는 실행은 무효다.**  
  이유: 앵커가 재현되지 않고, 값 불일치의 원인이 미확인이다. 값 불일치가 구성 차이인지 드리프트인지 구분할 수 없다.

- **`D-042` 승격 근거 `+0.56%p`, 게이트 ① 결과, 게이트 ② 실적을 `verified`로 인용하는 실행은 무효다.**  
  이유: `PROJECT.md` §3.2 `D-050`은 계측 결함으로 이 기준선을 재현할 경로가 현재 없으므로, 해당 승격 근거·게이트 결과를 `verified`로 인용하지 않는다고 명시한다.

- **D-048 이후 seed·적합 변동이 있는 경로에서 exact hash만으로 드리프트/오염을 판정하는 실행은 무효다.**  
  이유: `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. 사전 등록된 허용 범위 또는 불확실성 기준이 없으면 exact hash 불일치를 판정할 수 없다.

- **sidecar 존재만으로 `verified`를 주장하는 실행은 무효다.**  
  이유: sidecar가 config 선언값만 기록하거나, effective 상태가 실제 계산에 반영되는지 확인되지 않으면 기록과 실행이 어긋날 수 있다.

- **canary equality만으로 설정 도달을 주장하는 실행은 무효다.**  
  이유: canary token이 `src/`가 실제로 읽는 경로로 전달됨이 확인되기 전까지, canary는 metadata 복사일 수 있다.

- **weight 변경 경로와 결정론 확인 없이 수행된 3-arm 비교는 실행 무효다.**  
  이유: `src/`가 weight를 읽는 경로가 확인되지 않았고, D-048 이후 exact equality의 유효성이 확인되지 않았다.

- **CPU-only 회귀가 production GPU bf16 경로의 등가를 입증한다고 주장하는 실행은 무효다.**  
  이유: CPU 경로와 production 경로가 동일한지 문서에 확인이 없다.

- **과거 산출물에 태그 또는 AUROC 근접성만으로 구성을 귀속시키는 실행은 무효다.**  
  이유: `ru90_shape_triple`은 공식 7-branch를 연상시키는 태그를 쓰지만 공식 7-branch 활성화 경로가 아니었고, macro 근접은 우연으로 기록됐다.

## 좌석 공백 (기권·실패로 빠진 관점)

- 없음. 입력에서 기권한 좌석은 없다.

## 다음 행동 (권한 등급과 함께)

- **[사용자 결정]** 공식 7-branch 기준선 지위를 확정하라.  
  (a) branch·config 통제 경로를 복구한 뒤 provenance sidecar와 함께 공식 7-branch를 재실행해 `verified` 기준선을 만들거나,  
  (b) `0.6227`을 `unverified-hold`로 유지하고 `verified` 기준선이 없을 때까지 승격 비교를 보류한다.  
  이유: 현재 공식 7-branch 재현 경로가 없고, `verified` 기준선 없이 승격 비교를 수행하면 구성 귀속이 불확실한 비교가 된다.

- **[T1 좌석 권고]** `verified` 조건을 코드 감사로 제한하라.  
  ensemble builder가 `effective_branch_list`·`effective_branch_weights`를 실제로 계산에 반영하는지, 가중치 0이 branch 제외와 동등한지, config가 단일 진실원천인지, unrecorded switch가 없는지, 라벨 사용 경로와 통계량 fit 범위가 기록되는지 확인하기 전에는 sidecar 존재만으로 `verified`를 주장하지 않는다.  
  이유: `src/`가 env를 읽지 않고, config 값이 계산에 반영되는지가 문서에서 확인되지 않았다.

- **[T1 좌석 권고]** `effective` 기록의 신뢰 조건을 고정하라.  
  최소 조건:  
  1. 기록 주체가 ensemble builder의 계산 시점 branch selection/score 산출 상태를 관찰한다.  
  2. `requested` 기록과 `effective` 기록이 분리된다.  
  3. canary 또는 통제 변수가 `src/`가 실제로 읽는 경로로 전달된다.  
  4. paired 실행에서 통제 변수 변경이 `effective` 상태 또는 `prediction_file_hash`를 바꾼다.  
  5. `prediction_file_hash`와 sidecar hash가 외부 검증 스크립트로 재계산되고 바인딩된다.  
  6. sidecar와 prediction file이 원자적으로 묶여 사후 수정이 어려운 저장소에 쓰인다.  
  7. unrecorded switch가 없는지 코드 감사로 확인된다.  
  외부 검증 신호: paired control variable change에 의한 산출물 변화, 독립 재계산 일치, 외부 hash 재계산, 코드 경로 감사. raw branch inputs가 저장되지 않으면 독립 재계산은 현재 문서 기준으로 불가능하다.

- **[T1 좌석 권고]** 기록 생성을 두 단계로 분리하고 `execution_id`로 결합하라.  
  실행 스크립트는 `requested_branch_list`·`requested_execution_mode`·`execution_id`를 기록하고, `src/`는 실제 resolved effective 상태와 `prediction_file_hash`를 기록하며, orchestrator가 두 기록과 예측 파일을 `execution_id`로 바인딩한다. `requested_effective_mismatch`가 true이면 해당 산출물은 `verified`가 아니다.  
  이유: `src/`는 requested env를 알 수 없고, 사후 기록은 실제 실행과 어긋날 수 있다.

- **[T1 좌석 권고]** 가중치 0·branch 제외·nonzero weight를 확인하는 최소 paired 실험을 설계·실행하라.  
  전제:  
  - SEAL 10을 사용하지 않는다.  
  - production evaluation code path를 통과해야 하며 mock을 금지한다.  
  - 고정 data/fold/seed/precision/cache off를 사용한다.  
  - target branch B가 nonzero contribution을 가진지 positive control로 확인한다.  
  - arm별 `code_hash`를 기록한다.  
  - 동일 arm 2회 실행으로 결정론을 확인한다. 결정론이 성립하지 않고 사전 등록 허용 기준이 없으면 판별 불가/실행 무효로 둔다.  
  arm:  
  1. B를 branch list에서 제거.  
  2. B를 포함하되 weight `0.0`.  
  3. B를 포함하되 nonzero weight.  
  비교: `per_fold`, `fold_aurocs`, `prediction_file_hash`.  
  해석:  
  - arm 3이 arm 1/2와 다르면, 그 경로에서 통제 변수가 계산에 도달했음을 보여준다.  
  - arm 3이 arm 1/2와 같으면, 설정 무시, branch contribution 0, weight 0 동등, 수치/경로 차이 중 무엇인지 구분 불가하다. positive control도 동일하면 실행 무효다.  
  - arm 1과 arm 2가 다르면, 그 경로에서 weight 0은 branch 제외와 동등하지 않다.  
  - arm 1과 arm 2가 같으면, 그 경로에서 operationally equivalent일 뿐 기전 입증이 아니다.  
  GPU: CPU에서 production builder/metric이 동일하게 실행되면 CPU로 설정 도달 smoke를 돌릴 수 있다. 그러나 CPU 결과가 production GPU bf16 경로의 등가를 입증하지는 못한다. production 경로가 GPU만 지원하면 GPU 회귀가 필요하고, 없으면 blocked다.

- **[T1 좌석 권고]** 회귀 스위트를 다음 형태로 넣는다.  
  1. knob이 resolved effective state에 반영되는지 검사.  
  2. resolved effective state가 앙상블 출력을 바꾸는지 paired sensitivity 검사.  
  3. `requested_branch_list`와 `effective_branch_list` 불일치 시 실패.  
  4. sidecar와 prediction file의 `execution_id`·hash 바인딩 검사.  
  5. 동일 arm 2회 실행의 결정론 선행 검사.  
  6. 3-arm paired negative control.  
  7. mock 금지, production builder/metric 사용.  
  8. SEAL 10 미사용.  
  GPU 여부: production 경로가 CPU에서 동일하게 실행되는지 확인 전에는 “문서에 없음”으로 둔다. CPU-only 회귀는 production 수치 경로 전체의 등가 입증이 아니라 설정 도달 smoke로 한정한다.

- **[T1 좌석 권고]** 과거 산출물 소급 귀속 레지스터를 만들되 최대 범위를 제한한다.  
  status:  
  - `verified`: 미래 실행에서 sidecar + canary/통제 변수 도달 + hash binding + 코드 감사 통과.  
  - `inferred`: execution 시점 code closure와 runtime inputs(config, env, data/fold manifest, seed, package/environment, cache)이 복원되고, 코드 감사로 unrecorded switch 없이 effective 구성이 유일하게 결정될 때만.  
  - `candidate_configurations`: 복원된 증거가 허용하는 복수 구성 집합.  
  - `unattributed`: 독립 증거가 없거나 모순.  
  requested는 script/log/env export가 보존되고, 그 시점에 `src/`가 해당 경로를 읽었음이 확인되어야 추론 가능하다. 태그·macro 근접·문서 기대는 단독 근거가 아니라 consistency check만이다.  
  현재 `ru90_shape_triple`: execution 시점 closure가 현재 5-branch 경로와 동일하고 unrecorded switch가 없음을 확인하면 effective 5-branch candidate로만 둘 수 있다. 확인되지 않으면 `unattributed` 또는 candidate set이다. 공식 7-branch baseline `0.6227`은 `verified`로 승격되지 않는다.

- **[사용자 결정]** 오염 검사 대체 규칙을 사전 등록하라.  
  provenance 대조만으로 통과를 인정할지, base arm 예측 산출물의 수치 비교를 함께 요구할지, 수치 비교의 판정 기준을 무엇으로 할지를 결정해야 한다.  
  이유: provenance 대조는 구성 일치만 확인하고, 수치 드리프트·side effect·결정론 문제는 별도 기준이 필요하다.

## 하지 말 것 (이유와 함께)

- **`0.6227`을 `verified` 공식 7-branch 기준선으로 인용하지 말 것.**  
  이유: `PROJECT.md` §3.2는 `D-050` 이후 이 값을 `unverified-hold`로 명시하고, 보고 §8은 재현 경로가 현재 없다고 확인한다.

- **`ru90_shape_triple_official50_bf16` 산출물을 공식 7-branch score로 사용하지 말 것.**  
  이유: 보고 §8에서 해당 산출물은 공식 7-branch 활성화 경로가 아님으로 확인됐다.

- **`ICF_*` export, 스크립트 머리말, 파일 태그만으로 산출물 구성을 특정하지 말 것.**  
  이유: `src/`가 env를 읽지 않고, `run_ru90_shape_triple` 태그와 실제 구성이 어긋난 사례가 확인됐다.

- **현재 값 앵커 4자리 일치로 오염 검사 통과를 주장하지 말 것.**  
  이유: 기록된 앵커가 현재 코드로 재현되지 않고, intended base/screen arm을 통제할 경로가 무효다.

- **sidecar가 존재한다는 사실만으로 `verified`를 주장하지 말 것.**  
  이유: sidecar가 config 선언값만 기록하거나, effective 상태가 실제 계산에 반영되는지 확인되지 않으면 기록과 실행이 어긋날 수 있다.

- **`requested_branch_list`와 `effective_branch_list`를 동일 필드로 병합하지 말 것.**  
  이유: env 요청이 `src/`에 도달하지 않아 요청 구성과 실제 구성이 분리될 수 있다.

- **D-048 이후 seed·적합 변동이 있는 경로에서 exact hash만으로 드리프트/오염을 판정하지 말 것.**  
  이유: `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. 사전 등록된 허용 범위 또는 불확실성 기준이 없으면 판별 불가하다.

- **과거 산출물에 태그 또는 AUROC 근접성만으로 `high` confidence를 부여하지 말 것.**  
  이유: `ru90_shape_triple`은 공식 7-branch를 연상시키는 태그를 쓰지만 공식 7-branch 활성화 경로가 아니었고, macro 근접은 우연으로 기록됐다.

- **provenance 대조만으로 값 앵커의 모든 기능을 대체한다고 주장하지 말 것.**  
  이유: provenance 대조는 구성 일치 확인이지, 동일 구성에서의 수치 드리프트·결정론·side effect 감지를 자동으로 대체하지 않는다.

- **가중치 0을 branch 제외와 동일하다고 기정사실화하지 말 것.**  
  이유: `config.py` 가중치 0이 앙상블 계산에서 branch를 실제로 제외하는지는 코드 대조 결과가 없다. 기전 미확인.

- **CPU-only 회귀 결과를 production GPU bf16 경로 전체의 등가로 확장하지 말 것.**  
  이유: CPU 경로와 production 경로가 동일한지 문서에 확인이 없다.

- **문서에 없는 허용 오차, 임계값, confidence 수치를 새로 지어 쓰지 말 것.**  
  이유: 현재 문서에는 D-048 이후 드리프트 판정에 쓸 사전 등록 허용 기준이 없다.
