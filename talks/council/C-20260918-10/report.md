# ICF Council — C-20260918-10

- **연구 질문**: D-050으로 값 앵커 오염 검사(AUROC 4자리 일치)를 폐지하고 provenance 기록 대조로 재설계하기로 했다. 이 회차는 그 기록 스키마를 설계한다.

배경: 저장된 예측 파일의 키는 task_dir, fold_indices, fold_aurocs, fold_auroc_mean, per_fold 뿐이다. branch list, 가중치, 코드 해시, 데이터 버전이 하나도 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다. 또 src/가 환경변수를 하나도 읽지 않아 실행 스크립트의 ICF_* export가 전부 무효이며, 기록된 5-branch 앵커도 재현되지 않는다.

설계할 것: (1) 산출물에 무엇을 기록해야 구성을 재구성할 수 있는가 - 필드를 구체적으로 나열하라. (2) 무엇을 해시할 것인가. 코드 해시는 무엇의 해시인가(파일/디렉토리/git). (3) 기록이 실제 실행과 어긋날 수 있는 경로는 무엇이고 무엇으로 막는가. 기록은 실행과 같은 지점에서 생성돼야 하는가? (4) 값 앵커가 제공하던 기능 중 provenance 대조로 대체되지 않는 것이 있는가. 있다면 무엇으로 메우는가. (5) 과거 산출물에는 이 기록이 없다. 소급 귀속이 가능한 경우와 불가능한 경우를 어떻게 구분해 표시하는가.

문서에 없는 수치를 지어내지 말라.
- **RU 유형**: 탐색
- **최대 권한 등급**: T1
- **소집자**: Claude Code (claude-opus-5)
- **좌석**: P1(P, temp 0.8, http://127.0.0.1:8000/v1/chat/completions), A1(A, temp 0.2, http://127.0.0.1:8001/v1/chat/completions), R1(R, temp 0.5, http://127.0.0.1:8003/v1/chat/completions), T1(T, temp 0.4, http://127.0.0.1:8000/v1/chat/completions), S1(S, temp 0.2, http://127.0.0.1:8001/v1/chat/completions)
- **호출**: 9/14 · **소요**: 721.4초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 편성 근거 (결과 전 기록)

계측 복구의 첫 단계 설계다. 감사(A)를 핵심으로 두었다 - 기록과 실행이 어긋나는 경로를 찾는 것이 이 스키마의 존재 이유다. 반증(R)은 '기록만 늘리면 해결된다'는 결론과 스키마가 실제로는 검증력이 없는 형식주의가 되는 경로를 공격한다. 제안(P)이 스키마를 낸다. 이론(T)은 '값 앵커가 제공하던 기능 중 대체되지 않는 것'이라는 질문을 맡는다. 일반화(G)와 실측(M)은 제외: 질문이 내부 계측 설계이고 실행 대상이 아직 없다.

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: 산출물 provenance sidecar 스키마 (필드 + hash + 값 앵커 대체)
- **바꾸는 것**: 예측 파일 저장 시, 동일 디렉토리에 `<prediction_file>.provenance.json` sidecar를 생성한다. sidecar는 구성 재구성에 필요한 필드만 담고, 예측 값(`per_fold`, `fold_aurocs`)은 예측 파일에 유지한다. sidecar 필드는 다음과 같다.
  - `schema_version`: sidecar 스키마 버전 (예: `"1.0"`)
  - `created_at`: ISO 8601 timestamp
  - `task_dir`: (예측 파일과 동일)
  - `fold_indices`: (예측 파일과 동일)
  - `branch_list_effective`: `src/`가 실제로 앙상블에 포함한 branch 이름 목록 (예: `["CV","BM","BD","QA","DS"]`)
  - `branch_weights`: branch별 가중치 (예: `{"CV":1.0, "BM":1.0, ...}`)
  - `requested_branch_list`: 실행 스크립트가 env/arg로 요청한 branch (src가 읽지 못해도 기록)
  - `execution_mode`: `"FULL"` 또는 `"SCREEN_ONLY"` 등, `src/`가 실제로 적용한 모드
  - `config_hash`: 실행 시 로드된 config 객체의 canonical serialization SHA-256
  - `code_hash`: 실행에 영향을 주는 코드 파일들의 content hash (제안 2 참조)
  - `data_version`: 데이터셋 버전/해시
  - `fold_manifest_hash`: fold 분할 manifest의 SHA-256
  - `script`: 실행된 스크립트 경로 (예: `scripts/run_ru90_shape_triple.sh`)
  - `git_commit`: git commit hash (있다면, 없으면 `null`)
  - `seed`: random seed
  - `environment`: `{python_version, package_versions, gpu_model?}`
  - `prediction_file_hash`: 예측 파일 자체의 SHA-256
- **바꾸는 전제**: "구성은 실행 스크립트의 env/문서에서 추론할 수 있다"는 전제를 뒤집는다. `src/`가 env를 하나도 읽지 않으므로, 실행 시 실제 로드된 구성만 산출물에 기록해야 한다.
- **기대하는 차이**: 과거 산출물을 특정 구성에 귀속시킬 수 없었던 문제가 해결된다. 정량 예측 불가.
- **기전**: 실행 시 `src/`가 로드한 config 객체를 직렬화해 sidecar에 쓰고, 이후 산출물 대조 시 sidecar를 기준으로 한다.
- **실패 조건**: sidecar 생성이 실패하거나, config 객체가 실제 실행과 다르면 (예: `src/`가 env를 읽지 않아 YAML 기본값만 기록) 여전히 5/7-branch 구분 불가. 이 경우 sidecar는 존재하지만 내용이 무효이며, `execution_mode`과 `branch_list_effective`가 실제 실행을 반영하지 못한다.

## 제안 2: 코드 해시를 content hash로 재정의
- **바꾸는 것**: 코드 해시를 git commit hash(또는 git repo directory)로만 쓰는 대신, 실행이 import하는 모든 Python 파일 + config YAML + 실행 스크립트의 byte sequence를 canonical order로 묶어 SHA-256으로 계산한다. 구체적으로:
  - `src/` 전체 Python 파일
  - 실행 스크립트 (`scripts/*.sh`, `scripts/*.py`)
  - 사용된 config YAML (`train_v98_p1_reverse_1536_1gpu.yaml` 등)
  - 각 파일의 content hash를 정렬된 `(file_path, sha256)` 쌍으로 재해시
- **바꾸는 전제**: "코드 버전은 git commit 하나면 특정된다"는 전제를 뒤집는다. 로컬 수정, untracked 파일, vendor된 라이브러리, 실행 스크립트, config YAML이 commit과 달라질 수 있다.
- **기대하는 차이**: 코드 해시만으로 산출물의 코드 상태를 재구성할 수 있다. 정량 예측 불가.
- **기전**: 실행 시작 시점에 import graph를 따라 파일 목록을 수집하고, 각 파일의 content hash를 정렬된 키-값 쌍으로 재해시한다.
- **실패 조건**: import graph를 완전히 수집 못 하거나, 동적 import/생성 코드가 있으면 해시가 실제 실행 코드와 어긋난다. 이 경우 code_hash는 무효이며, 산출물을 특정 코드 상태에 귀속시킬 수 없다.

## 제안 3: 기록 생성 지점 고정과 소급 귀속 표시
- **바꾸는 것**:
  1. 기록 생성 지점을 실행 스크립트가 env를 export하는 지점(현재 무효)에서, `src/`가 config를 로드하고 예측을 산출한 직후로 고정한다. 예측 파일 쓰기와 sidecar 쓰기를 동일 transaction으로 묶어, 예측 파일만 생성되고 sidecar가 누락되는 상태를 막는다.
  2. 과거 산출물에는 sidecar가 없으므로, `provenance_registry.json`에서 file path를 키로 해서 `provenance_status`를 기록한다.
     - `provenance_status`:
       - `"verified"`: sidecar 존재, 실행 시 기록됨.
       - `"inferred"`: sidecar 없음, 파일명/태그/일시/코드 해시로 추론.
       - `"unattributed"`: 추론 불가, 구성 특정 불가.
     - `inference_basis`: 어떤 근거로 추론했는지 (예: 파일명 태그, 생성 일시, git commit 범위)
     - `confidence`: `"high"`, `"medium"`, `"low"`
  3. 값 앵커(AUROC 4자리 일치)가 제공하던 기능 중 provenance 대조로 대체되지 않는 것은 **수치 드리프트 감지**다. Provenance가 동일해도, 동일 산출물이 재현되는지 확인은 값/hash가 필요하다. 메우는 방법: `prediction_file_hash`를 sidecar에 기록하고, 재현 실행에서 hash 대조한다. 또는 `fold_aurocs` 전체의 canonical serialization hash를 대조한다. AUROC 4자리 일치 대신, fold_aurocs 전체의 exact equality 또는 hash를 쓴다.
- **바꾸는 전제**: "기록은 실행 전/외부에서 생성 가능하고, 과거 산출물은 태그/일시만으로 구성이 특정된다"는 전제를 뒤집는다. `src/`가 env를 읽지 않으므로, 기록은 실행과 같은 지점에서 생성되어야 한다. 과거 산출물은 태그 모호성으로 구성 특정 불가.
- **기대하는 차이**: 기록-실행 불일치 경로가 제거되고, 과거 산출물의 귀속 상태가 명시된다. 정량 예측 불가.
- **기전**: `src/`가 config를 로드한 객체를 sidecar에 쓰고, 예측 파일 쓰기와 sidecar 쓰기를 transaction으로 묶는다. 과거 산출물은 태그/일시/코드 해시로 추론하고, `provenance_registry.json`에 기록한다.
- **실패 조건**: `src/`가 env를 읽지 않으면 sidecar에 기록되는 config는 YAML 기본값(5-branch)뿐이고, 실행 스크립트가 의도한 7-branch는 기록되지 않는다. 이 경우 sidecar는 존재하지만 `branch_list_effective`가 실제 실행을 반영하지 못하며, `verified` 상태의 신뢰도가 무너진다. 과거 산출물 추론은 태그 모호성(예: `ru90_shape_triple`이 공식 7-branch를 의미하지만 실제로는 5-branch일 수 있음)으로 불확실하며, `inferred`의 `confidence`가 `"low"`로 고정된다.

### A1 (감사)

## 점검 1: 산출물 키에 구성 식별자(branch list, 가중치) 누락
- **확인하려는 것**: 저장된 예측 파일이 특정 branch 구성(branch list, 가중치)에 귀속되는지. 구성이 다르면 파일이 구분되는지. 관측된 macro 차이가 실제 구성 차이인지, 파일 오귀속인지 판별 가능한가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 5-branch 산출물이 7-branch 태그로 이름 붙여져, 관측된 macro `0.6226`이 7-branch `0.6227`에 "일치"하는 것으로 오독됨. 실제는 5-branch 산출물(§8). 구성 차이를 관측된 수치 차이로 오인하거나, 그 반대로 오인.
- **확인 방법**: `predictions/pathobench_*_ru90_shape_triple_official50_bf16.pt` 파일의 키 구조 확인. `scripts/run_ru90_shape_triple.sh`와 `src/models/config.py` 기본값(`weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`) 대조.

## 점검 2: 환경변수 경로 무효로 설정 표면 전체 무효
- **확인하려는 것**: 실행 스크립트의 `ICF_*` export가 평가 코드로 전달되어 구성을 변경하는지. 스크립트가 선언한 구성과 실제 산출된 구성이 일치하는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 스크립트가 `ICF_FIXED_HEAD_SH_WEIGHT=1.0`을 export해도 `config.py` 기본값 `weight_sh=0.0` 사용. 5-branch가 항상 산출. 스크립트 머리말이 "7-branch"라고 적어도 실제는 5-branch. 관측된 차이가 설정 표면 무효에서 새어 나옴.
- **확인 방법**: `grep -rln "os.environ\|getenv" --include="*.py" src/` (결과 0개 파일). `src/models/config.py` 기본값 확인. `scripts/eval_v121.sh`의 `icf_arm_v121()` 확인.

## 점검 3: fold manifest 해시 누락
- **확인하려는 것**: fold 분할이 PathoBench 공식 분할 그대로 고정되고, 매니페스트 해시가 산출물에 기록되는지. 대응 per-fold Δ의 fold 짝짓기가 동일한 fold 집합 위에서 수행되는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: fold 재분할/변경으로 관측된 Δ가 fold 구성 차이에서 새어 나옴. 대응 per-fold Δ가 무의미(§4.1: 짝짓기가 정밀도를 결정, fold는 과제 내에서 독립 아님, 설계 효과 5.12배).
- **확인 방법**: fold manifest 파일(과제별 fold 슬라이드 목록) 해시 계산. `task_dir`/`fold_indices`와 PathoBench 공식 분할 대조.

## 점검 4: 전처리·정규화·통계량 fit 범위 기록 누락
- **확인하려는 것**: 전처리(whitening, within-slide PCA 등)와 통계량이 train fold에만 fit되고 test fold에 새어 나가지 않는지. fit 범위가 산출물에 기록되어 fold 간 독립성이 검증 가능한지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test fold 정보가 전처리/정규화에 새어 관측된 AUROC 과대평가. fold 간 독립성 위반. §4.1의 대응 Δ 정밀도 전제(과제 내 fold 독립) 붕괴. 관측된 Δ가 전처리 누출에서 새어 나옴.
- **확인 방법**: 전처리 코드에서 fit/transform 분리 확인. fold 경계에서 fit 범위 확인. 산출물에 fit 범위(train fold ID 목록) 기록 여부 확인.

## 점검 5: 라벨 사용 경로 기록 누락
- **확인하려는 것**: 라벨(context/query)이 어떤 경로로 평가에 사용되는지. 선택 신호로 쓰이지 않는지. 라벨 사용 경로가 산출물에 기록되어 §224 Oracle 금지를 사후 검증할 수 있는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: query 라벨이 전처리/선택에 새어 oracle 편향. §224 Oracle 금지 위반. 관측된 Δ가 라벨 누출에서 새어 나옴. CA-06(라벨 사용 판별 기저) 경계 저촉.
- **확인 방법**: 라벨 접근 경로 코드 감사. 선택 이력 기록. 산출물에 라벨 사용 경로(context only / query used) 기록 여부 확인.

## 점검 6: 코드 해시 대상 미정의
- **확인하려는 것**: 코드 해시가 무엇의 해시인지(파일/디렉토리/git commit). 산출물에 기록되어 코드 변경으로 인한 관측된 차이를 판별할 수 있는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 코드 변경으로 관측된 차이가 코드 버전 차이에서 새어 나옴. 재현 불가. §8의 "5-branch 드리프트 원인(코드 변경/매니페스트/전처리/수치 정밀도) 미확인" 상태 지속. 기전 미확인.
- **확인 방법**: git commit hash 기록. `src/` 디렉토리 전체 해시(파일 목록+내용). 산출물에 코드 해시 필드 추가 여부 확인.

## 점검 7: 기록 생성 시점(실행과 같은 지점) 미정의
- **확인하려는 것**: provenance 기록이 실행과 같은 지점에서(실행 중) 생성되는지. 후속 수동 기록으로 실제 실행 구성과 어긋날 수 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 기록이 실행 후 수동으로 작성되어 실제 실행 구성과 다름. §8의 "스크립트 머리말이 낡았다"와 같은 문서-코드 불일치가 기록에도 발생. provenance 대조 자체가 무의미.
- **확인 방법**: 기록 생성 코드 경로 확인(실행 중 자동 생성 vs 후속 수동). 실행 로그와 기록 대조.

## 점검 8: 값 앵커의 수치 드리프트 감지 기능 대체
- **확인하려는 것**: 값 앵커(AUROC 4자리 일치)가 제공하던 "수치 드리프트 감지" 기능이 provenance 대조로 대체되는지. 구성이 동일해도 수치 드리프트(전처리/수치 정밀도/하드웨어/seed)를 감지할 수 있는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 구성이 동일(provenance 일치)해도 수치 드리프트가 감지되지 않음. D-048로 적합 분산 보고 필수이나, provenance 대조만으로는 seed/적합 분산 감지 불가. RU-97 §8의 동일 구성 seed만 바꿔 macro 폭 `0.0200`을 provenance 대조로 감지 불가.
- **확인 방법**: 동일 구성(provenance 일치) 재실행 시 AUROC 비교. 수치 정밀도(bf16/fp32) 기록. seed 기록. 적합 분산 보고.

## 점검 9: 과거 산출물 소급 귀속 구분
- **확인하려는 것**: 과거 산출물에 provenance 기록이 없을 때 소급 귀속이 가능한 경우와 불가능한 경우를 어떻게 구분해 표시하는지. 과거 산출물을 현재 구성으로 오귀속하여 관측된 차이가 구성 차이에서 새어 나가지 않는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 과거 산출물을 현재 구성으로 오귀속하여 관측된 차이가 구성 차이에서 새어 나옴. §3.3의 "직접 비교 금지" 위반. §8의 "과거 `ru90_shape_triple` 실행이 당시에는 shape margin을 포함했는지 미확인" 상태 지속.
- **확인 방법**: 과거 산출물 메타데이터(파일 생성 시점, git 이력, 실행 로그) 대조. 소급 귀속 가능/불가능 표시 스키마 정의.

### R1 (반증)

## 반론 1
- **공격 대상**: "값 앵커 오염 검사를 폐지하고 provenance 기록 대조로 재설계한다"는 D-050 결정의 전제, 즉 "provenance 기록을 추가하면 구성을 재구성할 수 있다"는 주장
- **더 단순한 설명**: `src/`가 환경변수를 하나도 읽지 않는다는 코드 확인 결과(§8)에 따르면, "설정"이라는 개념 자체가 `src/` 실행 경로에 존재하지 않는다. 실행 스크립트가 `ICF_*`를 export하든 마든 `src/`가 사용하는 값은 항상 `src/models/config.py` 기본값(`weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`)이다. 이 상태에서 provenance 기록 스키마에 "branch list", "가중치" 필드를 설계하면, 기록할 대상이 "실행 스크립트가 export한 값"인지 "src/가 실제로 사용한 값"인지 모호해진다. 더 단순한 설명: 설정 표면이 무효인 한, provenance 기록은 "의도된 설정"과 "실제 적용된 설정"을 구분할 수 없고, 이 구분 없이는 기록 대조가 구성 재구성을 보장하지 않는다.
- **무엇이 참이면 이 주장이 무너지는가**: `src/`가 환경변수를 읽도록 수정되고, 그 수정이 "설정 표면 복구"로 인정되며, provenance 기록이 `src/` 내부에서 "실제 적용된 값"을 기록하도록 설계되면, 구성 재구성 주장은 유효해진다.
- **이 반론의 한계**: 제안은 (3)에서 "기록이 실제 실행과 어긋날 수 있는 경로는 무엇이고 무엇으로 막는가. 기록은 실행과 같은 지점에서 생성돼야 하는가?"를 명시적으로 묻고 있다. 즉, 제안은 "기록과 실행의 어긋남"을 인식하고 있고, "설정 표면 복구"를 전제하지 않고도 "실제 적용된 값"을 기록하는 메커니즘을 설계할 수 있다. 이 반론은 제안이 (3)의 질문에 답하지 못한다고 가정하지만, 실제 설계에서 `src/` 내부에서 config 값을 직접 읽어 기록하는 방식으로 답할 수 있다.

## 반론 2
- **공격 대상**: "값 앵커(AUROC 4자리 일치)로는 구성이 바뀌었는지와 수치가 드리프트했는지를 구분할 수 없다는 설계 한계"라는 D-050의 원인 규명
- **더 단순한 설명**: 값 앵커 재현 실패(SMAD4 `0.4421`→`0.4661`, PBRM1 `0.5553`→`0.5408`, macro `0.6171`→`0.6226`)는 "값 앵커 설계의 한계"가 아니라 "현재 코드가 앵커를 만든 원래 구성을 재현하지 못하는 코드 결함"일 수 있다. §8은 "5-branch 드리프트의 원인(코드 변경 / 매니페스트 / 전처리 / 수치 정밀도). 미확인"이라고 명시한다. 즉, 구성이 바뀌지 않았고(항상 5-branch), 단지 수치만 드리프트한 것일 수 있다. 이 경우 값 앵커를 폐지할 것이 아니라, 드리프트 원인을 찾아 원래 구성을 재현할 수 있도록 복구한 뒤 값 앵커를 다시 사용해야 한다. 값 앵커 폐지는 "설계 한계"가 아니라 "미확인 코드 결함"에 대한 과도한 대응일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 드리프트 원인이 "코드 변경"으로 확인되고, 그 변경이 앵커 생성 시점 이후에 발생했으며, 원래 코드를 복구해도 앵커가 재현되지 않으면(예: 데이터 버전 변경), 값 앵커 폐지는 유효해진다.
- **이 반론의 한계**: §8은 드리프트 원인을 "미확인"으로 남겨두고 있다. "코드 결함"일 수도 있고 "매니페스트 변경"일 수도 있고 "전처리 변경"일 수도 있고 "수치 정밀도 변경"일 수도 있다. 이 반론은 "코드 결함"이라는 하나의 가능성을 더 단순한 설명으로 제시하지만, 다른 가능성(데이터 버전 변경 등)도 같은 관측을 낳을 수 있다. 또한, D-050은 "설계 한계"를 폐지 사유로 제시하지만, 실제로는 "미확인 결함"에 대한 임시 조치일 수 있다. 이 반론은 "폐지" 자체를 공격하지만, "재설계" 방향(provenance 기록)은 드리프트 원인과 무관하게 유효할 수 있다.

## 반론 3
- **공격 대상**: "provenance 기록 대조로 오염 검사를 재설계한다"는 제안의 유효 범위, 특히 과거 산출물에 대한 적용
- **더 단순한 설명**: provenance 기록은 "실행 시점"에 생성되는 것이다. 과거 산출물에는 이 기록이 없으므로, provenance 대조는 "향후 실행"에만 적용된다. 과거 산출물의 구성을 구별하려면, provenance 기록이 아닌 다른 증거(AUROC 일치, 파일명 태그, git commit hash 등)를 사용해야 한다. 하지만 §8은 "macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이다"라고 명시하고, "저장된 예측 파일에도 branch list·가중치·코드 해시가 기록돼 있지 않아 과거 산출물을 특정 구성에 귀속시킬 수 없다"고 한다. 즉, AUROC 일치만으로 구성을 귀속할 수 없고, provenance 기록도 없으므로, 과거 산출물은 "소급 귀속 불가"일 수밖에 없다. 더 단순한 설명: "소급 귀속이 가능한 경우"라는 범주 자체가 존재하지 않을 수 있다. 모든 과거 산출물은 "소급 귀속 불가"로 표시하면 된다.
- **무엇이 참이면 이 주장이 무너지는가**: 과거 산출물의 파일명 태그(예: `ru90_shape_triple`), 생성 시점, git commit hash, 또는 다른 메타데이터를 이용해 구성을 구별할 수 있는 경로가 확인되면, "소급 귀속 가능한 경우"가 존재한다.
- **이 반론의 한계**: 제안은 (5)에서 "소급 귀속이 가능한 경우와 불가능한 경우를 어떻게 구분해 표시하는가"를 묻고 있다. 즉, 제안은 "가능한 경우"와 "불가능한 경우"가 공존할 수 있다는 것을 전제하고 있다. 이 반론은 "가능한 경우"가 존재하지 않는다고 주장하지만, 제안은 "구분해 표시하는 방법"을 묻고 있을 뿐, "가능한 경우"의 존재를 확정하지는 않는다. 또한, §8은 "과거 `ru90_shape_triple` 실행이 당시에는 shape margin을 포함했는지"를 "아직 확인하지 않은 것"으로 남겨두고 있다. 즉, 과거 산출물 중 일부는 "소급 귀속 가능"일 수 있고, 이 반론은 그 가능성을 배제하지 못한다.

## 반론 4
- **공격 대상**: "값 앵커 오염 검사를 폐지하고 provenance 기록 대조로 재설계한다"는 D-050 결정이 "값 앵커가 제공하던 모든 기능을 provenance 대조로 대체한다"는 암묵적 전제
- **더 단순한 설명**: 값 앵커는 두 가지 기능을 제공했다: (a) "스크리닝이 기저 앙상블을 건드리지 않았는지" 확인, (b) "같은 구성에서 같은 값이 나오는가"(결정론 검증) 확인. provenance 대조는 (a)를 대체할 수 있다(스크리닝 실행과 기저 실행의 branch list·가중치가 동일한지 확인). 하지만 (b)는 대체되지 않는다. provenance 대조는 "기록된 구성이 일치하는지"를 확인하지만, "같은 구성에서 같은 값이 나오는가"는 확인하지 않는다. 즉, 결정론 검증은 provenance 대조와 별개다. 더 단순한 설명: 값 앵커 폐지로 "결정론 검증" 기능이 소실되고, provenance 대조는 이 기능을 대체하지 않는다.
- **무엇이 참이면 이 주장이 무너지는가**: provenance 대조 스키마에 "같은 구성에서 같은 값이 나오는가"를 확인하는 메커니즘(예: 동일 provenance를 가진 두 실행의 AUROC 대조)이 포함되면, 결정론 검증 기능은 대체된다.
- **이 반론의 한계**: 제안은 (4)에서 "값 앵커가 제공하던 기능 중 provenance 대조로 대체되지 않는 것이 있는가. 있다면 무엇으로 메우는가"를 명시적으로 묻고 있다. 즉, 제안은 "대체되지 않는 기능이 있을 수 있다"는 것을 인식하고 있고, "무엇으로 메우는가"를 묻고 있다. 이 반론은 "대체되지 않는다"는 것을 지적하지만, 제안은 이미 이 문제를 인식하고 있고, "메우는 방법"을 설계할 예정이다. 또한, §3.2는 "결정성: 태그 간 per-fold AUROC `max|diff| = 0.000e+00`"을 이미 보고하고 있으므로, 결정론 검증은 값 앵커에 의존하지 않고도 수행될 수 있다.

## 반론 5
- **공격 대상**: "값 앵커 오염 검사를 폐지한다"는 D-050 결정의 필요성, 즉 "오염 검사가 더 이상 유효하지 않다"는 주장
- **더 단순한 설명**: 오염 검사의 원래 목적은 "SCREEN_ONLY 실행이 앙상블에 영향을 주지 않았는지 확인"이다. §6은 "ICF_SHAPE_SCREEN_ONLY는 비테스트 코드 어디에서도 읽히지 않는다"고 확인했다. 즉, SCREEN_ONLY=1도 SCREEN_ONLY=0도 `src/`에 도달하지 않으므로, 스크리닝 실행과 기저 실행은 동일한 결과를 낸다. 이 경우 "오염"은 발생할 수 없다. 더 단순한 설명: 오염 검사가 폐지되는 이유는 "값 앵커가 재현되지 않는다"가 아니라, "오염이 발생할 수 있는 경로가 현재 코드에 존재하지 않는다"이다. 즉, 오염 검사는 "값 앵커 결함"이 아니라 "오염 경로 부재"로 인해 불필요하다.
- **무엇이 참이면 이 주장이 무너지는가**: `src/`가 환경변수를 읽도록 수정되고, SCREEN_ONLY=1이 실제로 앙상블 구성을 바꿀 수 있으면, 오염은 발생할 수 있고, 오염 검사는 필요하다. 이 경우 "값 앵커 결함"이 아니라 "오염 경로 존재"가 오염 검사의 필요성을 결정한다.
- **이 반론의 한계**: §6은 "ICF_SHAPE_SCREEN_ONLY는 비테스트 코드 어디에서도 읽히지 않는다"고 확인했지만, "비테스트 코드"의 범위가 `src/` 전체인지, `scripts/`를 포함하는지, `tests/`를 제외하는지는 명시되지 않는다. 또한, §8은 "src/는 환경변수를 하나도 읽지 않는다"고 확인했지만, "테스트 코드"나 "분석 스크립트"가 환경변수를 읽을 가능성은 배제하지 않는다. 이 반론은 "오염 경로 부재"를 주장하지만, "오염 경로"의 정의(어떤 코드 경로가 오염을 일으킬 수 있는가)가 모호하다.

### T1 (이론)

## 이론적 병목 1
- **가정**: provenance 기록은 실제 실행이 사용한 branch list·weights·code·data·fold manifest를 실행 경로 내부에서 생성하고, 산출물과 함께 불변으로 묶인다. 기록은 스크립트 머리말, 환경변수 export, 파일명, 태그, 사후 메타데이터가 아니다.
- **이 가정이 깨지는 조건**: 기록이 실행 래퍼/스크립트에서 사후에 쓰이거나, `src/`가 읽지 않는 `ICF_*` 값을 복사하거나, 기록 파일이 예측 파일과 분리되어 수정 가능하거나, 코드 경로가 기록 필드를 채우지 않고 기본값/외부 값을 쓰거나, 기록과 산출물 간 해시/시퀀스 바인딩이 없는 경우.
- **깨질 때 나타나는 관측 가능한 증상**: 기록상 구성 A인데 예측 값/행동이 B; 기록만 바뀌고 예측 파일은 동일; 같은 예측 파일에 서로 다른 기록; 스크립트 export와 `src/` 실제 설정 불일치; 태그와 실제 branch list 불일치. 이 불일치가 확인되면 해당 산출물의 구성 귀속은 실행 무효 또는 판별 불가다.
- **문서의 어떤 관측이 이 가정과 충돌하는가** (없으면 '해당 관측 없음'): `src/`가 환경변수를 읽지 않아 `ICF_*` export가 무효라는 관측; `run_ru90_shape_triple.sh` 산출물이 `ru90_shape_triple_official50` 태그로 저장됐으나 코드 확인상 5-branch 산출물이라는 관측; 스크립트 머리말의 SCREEN_ONLY 기대와 관측 macro `0.6226`의 불일치.

## 이론적 병목 2
- **가정**: branch list, weights, code hash, data version, fold manifest hash의 기록만으로 실제 실행 구성을 재구성할 수 있다. code hash는 실행에 영향을 주는 모든 코드(`src/`, `scripts/`, config, generated code, dependencies)를 포괄하고, data/fold hash는 raw data만 아니라 전처리·slide/context manifest·라벨·fold 구성을 포괄한다.
- **이 가정이 깨지는 조건**: 기록되지 않은 요인이 AUROC에 영향: YAML/기본값, Python/library/GPU/CUDA, seed, bf16/정밀도, thread, 전처리 캐시, normalization, checkpoint, slide/context list, 운영체제. code hash가 `src/`만, git commit만, 파일 일부만 해시하거나; data version이 dataset ID만; fold manifest가 `fold_indices`만 해시하는 경우.
- **깨질 때 나타나는 관측 가능한 증상**: provenance 필드 일치인데 per-fold AUROC가 다름; provenance 불일치인데 AUROC가 같음; 같은 code hash에서 다른 `scripts/`/YAML로 다른 branch list; data version 동일인데 전처리 캐시 변경으로 값 변화. 원인이 확인되지 않으면 기전 미확인으로 둔다.
- **문서의 어떤 관측이 이 가정과 충돌하는가** (없으면 '해당 관측 없음'): 저장 예측 파일의 키는 `task_dir`, `fold_indices`, `fold_aurocs`, `fold_auroc_mean`, `per_fold`뿐이며 branch list·weights·code hash·data version이 없다는 관측. 코드 확인상 5-branch로 산출된 실행의 SMAD4 `0.4661`, PBRM1 `0.5408`, macro `0.6226`이 기록된 5-branch 앵커 `0.4421`, `0.5553`, `0.6171`과 다르다는 관측(원인 미확인). 실제 설정은 `src/models/config.py` 기본값과 YAML에서 오고, 스크립트 export는 무효라는 관측.

## 이론적 병목 3
- **가정**: provenance 대조만으로 값 앵커의 오염 검사 기능을 대체한다. 즉, 기대 구성과 provenance가 같으면 SCREEN_ONLY가 base 앙상블에 영향을 주지 않았거나, 구성 변경이 없으므로 오염 검사 통과로 간주할 수 있다. 수치 불변식(AUROC 4자리 일치)은 불필요하다.
- **이 가정이 깨지는 조건**: 같은 provenance에서 side effect, nondeterminism, unrecorded factor로 base 예측이 바뀌거나; screen-only 경로가 branch list/weights를 바꾸지 않고도 집계 입력/캐시/모델 상태에 영향을 주거나; 코드/데이터 버전이 같아도 수치 정밀도/환경/seed로 값이 달라지는 경우.
- **깨질 때 나타나는 관측 가능한 증상**: provenance 대조 통과인데 base arm per-fold AUROC가 기대/기준값과 다름; provenance 불일치인데 base arm 값이 같음; 동일 provenance 2회 실행에서 값이 다름; 2회 실행 결정론적 동일하지만 기록 앵커와 다름. provenance 통과만으로는 오염 여부를 판별 불가하며, 수치 대조/재실행이 없으면 오염 검사 통과 주장은 실행 무효다.
- **문서의 어떤 관측이 이 가정과 충돌하는가** (없으면 '해당 관측 없음'): 가중치를 `0.0`으로 두어도 SMAD4 `0.4661`, PBRM1 `0.5408`로 4자리 동일하고 기록 앵커와 다르다는 관측. `D-048`으로 seed std `0.00000`이 더는 성립하지 않으며, `RU-97` §8은 동일 구성에서 seed만 바꿔 macro 폭 `0.0200`을 실측했다는 관측.

## 이론적 병목 4
- **가정**: 과거 산출물은 provenance 기록이 없어도 파일명·태그·macro 근접성·코드 이력·로그 등으로 특정 구성에 소급 귀속하거나, 귀속 가능/불가능을 명확히 구분할 수 있다.
- **이 가정이 깨지는 조건**: 여러 구성이 같은 태그/값 범위에 들어오거나, 코드 이력이 불완전하거나, 값 근접성이 우연이거나, 기록이 없어 실제 구성과 의도 구성을 분리할 수 없거나, 과거 실행의 환경/데이터 버전이 복원 불가능한 경우.
- **깨질 때 나타나는 관측 가능한 증상**: `official50`/`ru90_shape_triple` 태그가 실제 branch list와 다름; macro `0.6226`이 7-branch `0.6227`에 근접하지만 실제 5-branch; 같은 값이 여러 구성에서 나올 수 있어 판별 불가; 소급 귀속이 가능하다고 주장할 증거가 없음.
- **문서의 어떤 관측이 이 가정과 충돌하는가** (없으면 '해당 관측 없음'): `run_ru90_shape_triple.sh`가 생성한 파일은 공식 7-branch가 아니라 5-branch 산출물이며 공식 태그로 잘못 이름 붙었다는 관측; macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이라는 관측; 5-branch 앵커가 현재 코드로 재현되지 않는다는 관측.

## 이론적 병목 5
- **가정**: code hash는 파일/디렉토리/git 중 하나로 정의해도 실행 코드의 상태를 유일하게 식별한다. 특히 `src/` 디렉토리 해시 또는 git commit hash가 `scripts/`, config, dependency, working tree, generated code를 포괄한다.
- **이 가정이 깨지는 조건**: 실행이 uncommitted changes, untracked files, submodules, generated code, installed packages, `scripts/`, YAML, environment, bytecode에 의존하거나; git commit hash가 working tree를 반영하지 못하거나; dir hash가 `scripts/`/config/dependencies를 제외하거나; file hash가 단일 파일만 해시하는 경우.
- **깨질 때 나타나는 관측 가능한 증상**: 같은 git commit에서 untracked file/env 차이로 다른 branch list/값; 같은 `src/` hash에서 `scripts/`/YAML 차이로 다른 실행; code hash가 형식 변경으로 바뀌지만 행동은 동일; code hash가 동일하지만 `scripts/`가 달라 실제 구성 다름.
- **문서의 어떤 관측이 이 가정과 충돌하는가** (없으면 '해당 관측 없음'): `src/`가 환경변수를 읽지 않지만, `scripts/eval_v121.sh`는 5-branch arm을 정의하고 `run_ru90_shape_triple.sh`는 실행 태그/스크립트 경로를 결정한다는 관측. 실제 설정은 `src/models/config.py` 기본값과 YAML에서 오므로, `src/`만 해시하면 `scripts/`/config를 놓친다.

## 이론적 병목 6
- **가정**: 공식 구성(7-branch 또는 5-branch base)은 실제로 실행 가능한 경로가 있고, 그 구성의 provenance를 기대값으로 생성·비교할 수 있다.
- **이 가정이 깨지는 조건**: `SH`/`SJ`를 켜는 경로가 무효이거나, 공식 7-branch를 산출하는 코드가 없거나, base arm을 산출하는 경로가 문서와 다르면, 기대 provenance를 만들 수 없다.
- **깨질 때 나타나는 관측 가능한 증상**: 기대 7-branch provenance를 가진 산출물이 하나도 없음; 모든 실제 기록이 5-branch; 공식 태그 산출물이 5-branch; 비교 대상이 없어 오염/승격 비교가 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가** (없으면 '해당 관측 없음'): `src/`가 환경변수를 읽지 않아 `ICF_*` export가 무효라는 관측; `SH`/`SJ`를 켜는 문서화된 수단이 동작하지 않는다는 관측; `run_ru90_shape_triple` 산출물은 5-branch이며 공식 태그로 잘못 이름 붙었다는 관측; 공식 7-branch 기준선 재현 경로가 현재 없다는 관측.

---

## Phase 2 — 교차

### A1 (감사)

## 점검 1: `branch_list_effective`·`branch_weights`가 실제 앙상블 계산에 사용된 값을 반영하는가
- **확인하려는 것**: sidecar의 `branch_list_effective`와 `branch_weights`가 예측 산출에 실제로 사용된 branch 집합과 가중치인지, config 객체의 값인지.
- **문서에서 확인되는가**: 모순됨. 보고 §8에서 `src/`가 환경변수를 읽지 않고, `config.py` 기본값이 `weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`이며, YAML이 브랜치 가중치를 설정하지 않음. 제안은 `branch_list_effective`를 "src가 실제로 앙상블에 포함한 branch 이름 목록"으로 정의하지만, config 객체가 앙상블 계산에 사용되는지 확인되지 않았다. 보고 §6에서 가중치를 0으로 두어도 값이 4자리까지 바뀌지 않았으므로, 가중치 0이 branch 제외를 의미하지 않을 수 있다.
- **샌다면 나타날 증상**: sidecar가 `branch_list_effective=["CV","BM","BD","QA","DS"]`를 기록하지만, 실제 앙상블 계산은 하드코딩된 다른 branch 집합을 사용. 또는 config 객체의 가중치 0이 branch 제외를 의미하지 않아, 가중치 0인 branch가 계산에 포함. `requested_branch_list`가 7-branch, `branch_list_effective`가 5-branch로 불일치하고, 어느 쪽이 "실제 실행"인지 불분명.
- **확인 방법**: `src/models/`의 앙상블 계산 코드를 읽어, config 객체의 `branch_weights`가 실제로 어떤 branch를 선택하는지, 가중치 0인 branch가 제외되는지 확인. `run_ru90_shape_triple.sh`가 어떤 eval 스크립트를 호출하는지 확인.

## 점검 2: `config_hash`의 canonical serialization과 범위
- **확인하려는 것**: `config_hash`가 "실행 시 로드된 config 객체의 canonical serialization SHA-256"이라고 하지만, canonical serialization의 정의가 있는지, config 객체가 실제 실행에 영향을 주는 모든 파라미터(branch list, weights, fold 분할, 전처리, 정규화, 통계량 fit 범위, 라벨 사용, evaluation 독립성)를 포함하는지.
- **문서에서 확인되는가**: 문서에 없음. config 객체의 필드 목록, canonical serialization 정의, config 객체가 앙상블 계산·fold 분할·전처리·정규화·통계량 fit·라벨 사용·evaluation 독립성에 사용되는지 문서에 없음.
- **샌다면 나타날 증상**: 같은 구성이라도 config 객체의 dict 순서, float 표현, None/0, default 값 포함 여부에 따라 hash가 달라져 대조 불가. 또는 config 객체에 fold 분할, 전처리 파라미터가 없어, `fold_manifest_hash`와 `data_version`만으로는 평가 설계를 특정하지 못함.
- **확인 방법**: `src/models/config.py`의 config 객체 필드 목록을 확인. config 객체가 앙상블 계산, fold 분할, 전처리, 정규화, 통계량 fit, 라벨 사용, evaluation 독립성에 사용되는지 코드 대조.

## 점검 3: `code_hash`의 범위와 environment 분리
- **확인하려는 것**: `code_hash`가 "src/ 전체 Python 파일" + scripts + config YAML이라고 하지만, 실행에 영향을 주는 코드의 범위가 명확한지, environment(package_versions, gpu_model)가 code_hash와 분리되어 있어, 같은 code_hash라도 package 버전이 다르면 결과가 달라질 수 있는지.
- **문서에서 확인되는가**: 문서에 없음. code_hash의 파일 범위, import graph 수집 방법, environment hash 정의 문서에 없음.
- **샌다면 나타날 증상**: src 전체를 해시하면 관련 없는 파일 변경으로 hash가 바뀌어 대조 무의미. 또는 import graph 수집 실패로 동적 import, C extension, numpy/scikit-learn/torch 버전이 code_hash에 반영되지 않아, 같은 code_hash라도 다른 package 버전에서 결과가 다름.
- **확인 방법**: `src/`의 import graph를 확인. 동적 import, 생성 코드, C extension이 있는지 확인. environment 필드가 code_hash와 함께 대조되는지 확인.

## 점검 4: `data_version`·`fold_manifest_hash`의 범위
- **확인하려는 것**: `data_version`이 원본 데이터셋만 가리키는지, 전처리 파생 데이터(임베딩, 특징)의 버전이 포함되는지. `fold_manifest_hash`가 fold 분할 manifest의 SHA-256이라고 하지만, manifest에 slide ID, task, fold index, train/test split, context/query 슬라이드 목록, 라벨이 포함되는지.
- **문서에서 확인되는가**: 문서에 없음. data_version의 정의, fold manifest의 내용 문서에 없음.
- **샌다면 나타날 증상**: data_version이 원본 WSI만 가리키면, 임베딩/특징 전처리 변경이 반영되지 않아, 같은 data_version이라도 다른 특징에서 결과가 다름. fold_manifest_hash가 slide-level 50-fold CV의 train-test 분리만 포함하면, context/query 슬라이드 목록, 라벨 변경이 반영되지 않아, 평가 독립성(라벨 사용, hold-out)을 특정하지 못함.
- **확인 방법**: data_version이 어떤 파일을 가리키는지 확인. fold manifest 파일의 내용을 확인. context/query 슬라이드 목록, 라벨이 manifest에 포함되는지 확인.

## 점검 5: `prediction_file_hash`·`fold_aurocs` hash의 exact equality와 드리프트 감지
- **확인하려는 것**: 제안 3은 "값 앵커가 제공하던 기능 중 provenance 대조로 대체되지 않는 것은 수치 드리프트 감지"라고 인정하고, "prediction_file_hash 또는 fold_aurocs hash"로 메운다고 한다. 그러나 hash는 exact equality만 가능하고, 허용 오차(4자리)를 제공할 수 없다. D-050은 "값 앵커(AUROC 4자리 일치)를 폐지"라고 한다.
- **문서에서 확인되는가**: 모순됨. D-050은 값 앵커 폐지. 제안 3은 hash로 드리프트 감지를 메운다고 하지만, hash는 exact equality. 드리프트 감지를 위해 허용 오차가 필요하면, 4자리 일치를 재도입해야 하는데, 이는 폐지.
- **샌다면 나타날 증상**: 같은 구성이라도 수치 정밀도, bf16, GPU, 라이브러리 버전에 따라 fold_aurocs가 0.0001 차이로 달라져 hash 대조 실패. 또는 0.0001 차이가 드리프트인지 재현성 실패인지 구분 불가.
- **확인 방법**: fold_aurocs의 canonical serialization 정의 확인. 허용 오차 임계값이 있는지 확인. D-050의 폐지 방향과 충돌하는지 확인.

## 점검 6: 기록 생성 지점과 sidecar 무효 경로
- **확인하려는 것**: 제안 3은 "기록 생성 지점을 src가 config를 로드하고 예측을 산출한 직후로 고정"한다고 한다. 그러나 src가 env를 읽지 않는 한, config 로드 시점에 이미 실행 스크립트의 env는 반영되지 않았다. sidecar가 기록하는 config는 실제 실행에 사용된 config일 수 있지만, 실행 스크립트가 의도한 config와 다를 수 있다.
- **문서에서 확인되는가**: 확인됨. 보고 §8에서 src가 env를 읽지 않음.
- **샌다면 나타날 증상**: sidecar가 config 객체(YAML 기본값, 5-branch)를 기록하지만, 실행 스크립트가 의도한 7-branch는 기록되지 않음. `requested_branch_list`가 7-branch, `branch_list_effective`가 5-branch로 불일치. 이 경우 sidecar는 존재하지만, "verified" 상태의 신뢰도가 무너짐.
- **확인 방법**: sidecar 생성 지점이 config 로드 직후인지, 예측 산출 함수가 실제로 받은 인자/상태를 기록하는지 확인. `requested_branch_list`와 `branch_list_effective`가 불일치할 때 어떤 것이 "실제 실행"인지 정의 확인.

## 점검 7: 과거 산출물 소급 귀속의 confidence 기준
- **확인하려는 것**: 제안 3은 `provenance_registry.json`에서 verified/inferred/unattributed, inference_basis, confidence high/medium/low를 기록한다고 한다. 그러나 confidence 기준이 없다.
- **문서에서 확인되는가**: 문서에 없음. confidence high/medium/low의 기준 문서에 없음.
- **샌다면 나타날 증상**: confidence가 주관적으로 정해져, inferred와 unattributed의 구별이 어려움. `ru90_shape_triple`은 공식 7-branch 태그로 잘못 이름 붙여진 5-branch 산출물일 수 있으므로, inferred confidence가 low여도 5-branch인지 7-branch인지 판별 불가.
- **확인 방법**: confidence 기준을 확인. `ru90_shape_triple` 산출물의 소급 귀속이 가능한지 확인.

## 점검 8: 기준선 unattributed 문제
- **확인하려는 것**: 공식 7-branch 기준선(0.6227)은 unverified-hold 상태. 과거 기준선 산출물에는 sidecar가 없으므로, 기준선도 unattributed. provenance 대조로 기준선을 검증할 수 없다.
- **문서에서 확인되는가**: 확인됨. PROJECT.md §3.2에서 공식 기준선 0.6227 unverified-hold. 보고 §8에서 공식 7-branch 기준선을 재현할 경로가 현재 없음.
- **샌다면 나타날 증상**: 기준선이 unattributed이면, 후보와 기준선의 비교 자체가 무효. provenance 대조로 기준선의 구성을 특정하지 못하므로, 승격 비교를 할 수 없음.
- **확인 방법**: 공식 7-branch 기준선 산출물의 provenance_status 확인. unattributed이면 승격 비교가 가능한지 확인.

### R1 (반증)

## 반론 1
- **공격 대상**: 제안 1의 `branch_list_effective`("src/가 실제로 앙상블에 포함한 branch 이름 목록")와 제안 3의 "기록 생성 지점을 src가 config를 로드하고 예측을 산출한 직후로 고정"이라는 주장.
- **더 단순한 설명**: sidecar가 기록하는 것은 config 객체다. 문서 §8에 따르면 `src/`는 환경변수를 하나도 읽지 않으므로, config 객체는 YAML 기본값(`weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`)만 담는다. 따라서 sidecar는 7-branch 실행을 5-branch로 기록한다. 이 더 단순한 설명은 제안 1 실패 조건이 스스로 인정한 "sidecar는 존재하지만 `branch_list_effective`가 실제 실행을 반영하지 못한다"는 관측을 그대로 낳는다. sidecar 스키마는 config 객체와 실제 앙상블 계산 경로를 연결하는 검증을 포함하지 않으므로, config 객체가 계산에 반영되었는지 확인하지 못한다.
- **무엇이 참이면 이 주장이 무너지는가**: `src/`가 환경변수를 읽도록 수정되고, config 객체가 실제 앙상블 계산에 반영됨이 코드 대조로 확인되면, sidecar의 `branch_list_effective`는 유효해진다.
- **이 반론의 한계**: `src/`가 환경변수를 읽지 않더라도, config 객체가 별도 경로(예: 하드코딩된 상수, CLI argument)로 실제 계산에 반영되는 것이 확인되면, sidecar는 유효할 수 있다. 다만 문서에는 그런 경로가 언급되지 않는다.

## 반론 2
- **공격 대상**: 제안 3의 "값 앵커가 제공하던 기능 중 provenance 대조로 대체되지 않는 것은 수치 드리프트 감지다. 메우는 방법: `prediction_file_hash`를 sidecar에 기록하고, 재현 실행에서 hash 대조한다. 또는 `fold_aurocs` 전체의 canonical serialization hash를 대조한다."
- **더 단순한 설명**: 값 앵커가 제공하던 기능은 "스크리닝이 기저 앙상블을 건드리지 않았는가"를 확인하는 실행 간 구성 불변성 확인이다. `prediction_file_hash`는 같은 실행의 재현성만 확인하고, 다른 실행(예: SCREEN_ONLY 실행)이 기저 앙상블과 동일한지 확인하려면 두 실행의 `fold_aurocs`를 비교해야 한다. `fold_aurocs`는 float이므로 exact equality 또는 hash는 float round-off에 취약하다. 문서 §3.4가 "소수 4자리까지 일치"를 요구한 것은 float round-off를 고려한 것이다. 더 단순한 설명: `fold_aurocs`의 `max|diff|`를 기록하고, 사전 정의된 임계값(예: `1e-6`)과 대조한다. 이 방법은 float round-off를 허용하면서도 실행 간 구성 불변성을 확인한다.
- **무엇이 참이면 이 주장이 무너지는가**: `fold_aurocs`가 결정론적으로 동일함이 확인되고(문서 §6에 따르면 2회 실행 결정론적으로 동일), exact equality가 float round-off를 허용하는 방식으로 정의되면, `prediction_file_hash` 또는 `fold_aurocs` hash는 유효할 수 있다.
- **이 반론의 한계**: `fold_aurocs`가 항상 결정론적으로 동일하면 exact equality는 유효하다. 다만 문서 §8에 "5-branch 드리프트의 원인(코드 변경 / 매니페스트 / 전처리 / 수치 정밀도). 미확인"이므로, 수치 정밀도 드리프트가 존재할 수 있으며, 이 경우 exact equality는 오거부를 낳을 수 있다.

## 반론 3
- **공격 대상**: 제안 2의 "import graph를 따라 파일 목록을 수집하고, 각 파일의 content hash를 정렬된 키-값 쌍으로 재해시한다."
- **더 단순한 설명**: `src/` 전체 Python 파일 + `scripts/` 전체 + 사용된 config YAML의 content hash를 계산하면 import graph 추적이 불필요하다. import graph 수집은 동적 import, C extension, JIT, runtime code generation, environment-dependent import를 포착하지 못할 수 있다. 제안 2 실패 조건이 스스로 인정한 "동적 import/생성 코드가 있으면 해시가 실제 실행 코드와 어긋난다"는 관측을 낳는다. 더 단순한 설명: 실행 시작 시점에 `src/` 전체 + `scripts/` 전체 + config YAML의 content hash를 계산한다. 이 방법은 import graph 추적을 피하고, 실패 모드를 줄인다.
- **무엇이 참이면 이 주장이 무너지는가**: import graph 수집이 동적 import 등을 포착할 수 있는 방식으로 구현되고, `src/` 전체 hash가 관련 없는 파일(예: vendor된 라이브러리, 테스트 파일)을 포함하지 않으면, import graph 수집은 유효할 수 있다.
- **이 반론의 한계**: `src/` 전체 hash는 관련 없는 파일 변경에도 해시가 변하므로, 변경 감지 정밀도가 낮아진다. import graph 수집은 관련 파일만 hash하므로 정밀도가 높다. 다만 문서에는 vendor된 라이브러리나 동적 import가 언급되지 않으므로, 이 한계의 실재성은 미확인이다.

## 반론 4
- **공격 대상**: 제안 1의 `requested_branch_list`("실행 스크립트가 env/arg로 요청한 branch (src가 읽지 못해도 기록)")와 `execution_mode`("src/가 실제로 적용한 모드")를 sidecar에 기록한다는 주장.
- **더 단순한 설명**: `src/`가 환경변수를 읽지 않으므로, `src/`는 `requested_branch_list`와 `execution_mode`를 알 수 없다. 따라서 이 필드는 실행 스크립트가 기록해야 한다. 제안은 sidecar를 `src/`가 생성한다고 했는데, `src/`가 `requested_branch_list`를 기록할 수 없다는 점에서 모순이다. 이 더 단순한 설명은 "sidecar에 `requested_branch_list`가 누락되거나, `src/`가 생성하지 못하는 필드가 sidecar에 포함된다"는 관측을 낳는다. 더 단순한 설명: sidecar를 두 단계로 생성한다. 실행 스크립트가 `requested_branch_list`와 `execution_mode`를 기록하고, `src/`가 config 객체와 `code_hash`를 추가한다.
- **무엇이 참이면 이 주장이 무너지는가**: `src/`가 환경변수를 읽도록 수정되면, `src/`는 `requested_branch_list`와 `execution_mode`를 기록할 수 있다.
- **이 반론의 한계**: `src/`가 환경변수를 읽지 않더라도, 실행 스크립트가 sidecar를 생성하는 방식으로 수정되면, 이 모순은 해결된다. 다만 제안은 sidecar 생성 지점을 `src/`로 고정하고 있으므로, 이 수정은 제안의 전제를 바꾼다.

### T1 (이론)

**이론적 병목 N1**
- **가정**: 제안 1은 `branch_list_effective`, `branch_weights`, `execution_mode`, `config_hash`가 실제 앙상블 계산에 반영된 구성을 무결하게 나타낸다고 둔다. `requested_branch_list`를 기록하려면 기록 생성 코드가 요청 구성을 관찰할 수 있다고도 둔다. 즉, config 객체가 branch 포함·가중치·모드의 단일 진실원천이고, config 밖의 코드 상수·import side effect·캐시·전처리·global state가 계산 구성을 바꾸지 않는다고 둔다.
- **이 가정이 깨지는 조건**: branch 포함/가중치가 config 필드가 아니라 하드코딩 상수, import side effect, 캐시, 전처리, 데이터 의존 경로, config 로드 후 mutation으로 결정될 때. `requested_branch_list`가 `src/`에 전달되지 않거나 `src/`가 env를 읽지 않아 내부에서 관찰 불가능할 때. `execution_mode`가 실제 계산 경로에 적용되지 않을 때. canonical serialization이 일부 필드를 누락할 때. 가중치 0이 branch 계산 제외와 동등하지 않을 때.
- **깨질 때 나타나는 관측 가능한 증상**: sidecar가 `verified`인데 `requested_branch_list`와 `branch_list_effective`가 다름. 같은 `branch_list_effective`/`config_hash`로 재실행하면 `fold_aurocs`/`prediction_file_hash`가 다름. sidecar effective가 5-branch인데 산출물 태그/문서 기대는 7-branch. `requested_branch_list`가 null이거나 추측값. `config_hash`가 동일인데 실행 경로에 따라 `branch_list_effective`가 다름.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `src/`는 환경변수를 하나도 읽지 않아 실행 스크립트의 `ICF_*` export가 무효이고, `config.py` 기본값은 `weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`이며 사용된 YAML은 가중치를 설정하지 않는다. `run_ru90_shape_triple.sh`로 생성된 `ru90_shape_triple` 산출물은 공식 태그를 쓰지만 5-branch 산출물이며, 기록된 5-branch 앵커는 현재 코드로 재현되지 않는다. 이 관측들은 요청/문서 기대와 실제 계산이 분리될 수 있고, config가 단일 진실원천이라는 가정이 현재 계측 결함과 충돌함을 보여준다.

**이론적 병목 N2**
- **가정**: 제안 3은 기록 생성 지점을 `src/` config 로드/예측 산출 직후로 고정하고 예측 파일과 sidecar를 동일 transaction으로 쓰면 기록-실행 불일치 경로가 제거된다고 둔다. 즉, 기록이 프로세스의 자기 보고만으로 충분하고, 예측 파일이 그 프로세스의 직접 산출물이며, 캐시·복사·포크·외부 수정이 없다고 둔다.
- **이 가정이 깨지는 조건**: sidecar가 설정 로더의 상태만 기록하고 실제 계산 그래프를 기록하지 않을 때. 예측 파일이 캐시, 이전 실행, 외부 스크립트, 수동 복사로부터 생성될 때. sidecar 작성 프로세스와 실제 계산 프로세스가 다를 때. `requested` 값을 `effective`로 잘못 채울 때. 파일시스템 쓰기가 원자적이지 않거나 사후 수정이 가능할 때.
- **깨질 때 나타나는 관측 가능한 증상**: sidecar와 `prediction_file_hash`가 함께 존재하지만, 같은 sidecar 필드로 재현 실행 시 다른 예측이 생성됨. sidecar가 requested 7-branch를 effective처럼 기록함. 예측 파일만 수정되어 sidecar hash 불일치가 발생함. `verified` 상태인데 산출물 태그와 실제 구성이 불일치함.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `run_ru90_shape_triple.sh` 산출물은 공식 태그를 쓰지만 5-branch 산출물이며, `src/`가 env를 읽지 않아 스크립트 export가 무효다. 이는 스크립트/태그의 의도와 실제 계산이 분리될 수 있음을 보여준다. 동일 transaction은 기록과 파일의 동시 생성만 보장할 뿐, 기록 내용이 실제 계산과 동일한 구성을 나타낸다는 것을 보장하지 않는다.

**이론적 병목 N3**
- **가정**: 제안 2는 import graph의 Python 파일 + 실행 스크립트 + config YAML의 content hash가 실행에 영향을 주는 코드 상태를 완전히 특정한다고 둔다. 즉, 동적 코드, compiled extension, package version, runtime environment, data manifest, metric code, GPU/BLAS/thread behavior가 hash에 포함되지 않아도 동일 code_hash는 동일 행동을 의미한다고 둔다.
- **이 가정이 깨지는 조건**: dynamic import, generated code, untracked imported file, monkey-patching, package version 변화, CUDA/cuDNN/BLAS/GPU/thread 차이, prediction 생성 import graph에 없는 전처리/평가/AUROC 계산 코드가 실행에 영향을 줄 때. import graph 수집이 불완전할 때.
- **깨질 때 나타나는 관측 가능한 증상**: code_hash가 동일인데 package version/GPU/thread/BLAS 차이로 `fold_aurocs`/`prediction_file_hash`가 다름. import graph 수집 실패로 일부 파일이 hash에서 누락됨. prediction file은 동일인데 downstream AUROC 계산 코드 변경으로 `fold_aurocs`가 다름.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: 해당 관측 없음

**이론적 병목 N4**
- **가정**: 제안 3은 `prediction_file_hash` 또는 `fold_aurocs` canonical hash 대조가 값 앵커의 수치 드리프트 감지와 오염 검사 대체 기능을 한다고 둔다. 즉, 기록된 구성 변수가 드리프트/오염의 전 인과 변수이고, 동일 구성 재현은 bit-exact 또는 exact equality로 판정 가능하다고 둔다.
- **이 가정이 깨지는 조건**: 학습 파라미터 허용으로 seed/적합 변동이 존재할 때. GPU/parallel/numerical nondeterminism, library version, rounding, data order, unmodeled side effect, metric code 변경이 예측이나 AUROC를 바꿀 때. exact hash가 허용 변동보다 엄격할 때.
- **깨질 때 나타나는 관측 가능한 증상**: 동일 provenance 필드인데 `prediction_file_hash`/`fold_aurocs` hash가 실행마다 다름. hash가 다름에도 4자리 AUROC는 같음. hash가 같음에도 downstream AUROC 계산이 다름. hash 불일치가 구성 실패가 아니라 seed/numerical 변동으로 인한 판별 불가. provenance는 일치하는데 4자리 값이 달라 unmodeled 오염/드리프트가 감지되지 않음.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `D-048`은 `seed std 0.00000`이 더는 성립하지 않는다고 하고, `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. 또한 `ICF_SHAPE_SCREEN_ONLY`는 비테스트 코드에서 읽히지 않고, `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT`도 출력에 영향이 없으며, 기록된 5-branch 앵커는 재현되지 않는다. 이는 exact hash 재현 가정과 provenance-only 오염 검사 가정에 충돌한다.

**이론적 병목 N5**
- **가정**: 제안 3은 과거 산출물을 file path, tag, timestamp, code hash로 `verified`/`inferred`/`unattributed`로 구분하고 confidence를 부여할 수 있다고 둔다. 즉, 태그/일시/값이 구성에 대한 신뢰할 수 있는 대리변수라고 둔다.
- **이 가정이 깨지는 조건**: 태그 의미가 시간에 따라 바뀌거나, 같은 태그가 서로 다른 구성에 사용되거나, 값이 우연히 겹치거나, 코드 히스토리가 없거나, 앵커가 재현되지 않거나, file path가 이동/중복될 때.
- **깨질 때 나타나는 관측 가능한 증상**: `ru90_shape_triple` 태그가 공식 7-branch로 읽히지만 실제 5-branch. macro `0.6226`이 7-branch `0.6227`에 가까워 7-branch로 오추론 가능. 5-branch 앵커 불일치로 값 기반 추론이 신뢰할 수 없음. 과거 산출물이 대량으로 `unattributed` 또는 `inferred low`가 되고, high/medium confidence가 근거 없이 부여됨.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `run_ru90_shape_triple.sh`로 생성된 7건은 공식 태그를 쓰지만 5-branch 산출물이며, macro `0.6226`이 7-branch `0.6227`에 가까운 것은 우연이다. 기록된 5-branch 앵커 `0.4421`/`0.5553`은 현재 코드로 재현되지 않고, 현재 5-branch 실측은 `0.4661`/`0.5408`이다. 이는 태그/값 기반 소급 추론이 구성을 특정하지 못할 수 있음을 보여준다.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

- **A1-1**: (1) 수용·수정 — `branch_list_effective`·`branch_weights`는 config 객체의 선언값이 아니라, 예측을 실제로 산출한 ensemble builder의 resolved 상태(포함 여부, 기여 가중치, code path id)를 기록하도록 바꾼다. `requested_branch_list`와 `branch_list_effective`가 다르면 `mismatch=true`로 표시한다. `src/`가 env를 읽지 않는 현재 상태에서 config 값이 계산에 반영되는지 코드 대조 전까지는 `verified`를 주장하지 않는다. 기전 미확인.

- **A1-2**: (1) 수용·수정 — `config_hash`는 canonical serialization의 정의 없이 쓰면 대조 가치가 없다. key 정렬, `null` 명시, float 표현 방식, default 포함 여부를 고정하고, config가 실제 실행에 영향을 주는 branch·mode·fold·전처리·정규화·통계량 fit·라벨 사용·평가 코드를 모두 포함해야 한다. config에 없는 필드는 `pipeline_manifest_hash`로 따로 기록한다. 누락된 구성 차원이 있으면 해당 차원은 `unattributed`로 둔다.

- **A1-3**: (1) 수용·수정 — `code_hash`는 import graph 추적만으로 정의하지 않는다. v1에서는 실행에 필요한 고정 파일 closure(`src/`, 실행 스크립트, config YAML, 데이터/평가 manifest, metric 코드)의 content hash를 쓰고, package·GPU·CUDA·BLAS·thread는 `environment_hash`로 분리한다. 동적 import·생성 코드·C extension이 존재하면 runtime import manifest를 추가로 기록하고, 없으면 `code_hash`의 완전성은 `미확인`으로 표시한다.

- **A1-4**: (1) 수용·수정 — `data_version`은 원본 WSI만 가리키면 부족하다. 원본 슬라이드, 임베딩/특징 파생물, 전처리 산출물, 라벨 파일, 배제 목록까지 버전 또는 hash로 특정해야 한다. `fold_manifest_hash`는 task, fold index, slide ID, train/test/context/query 분리, 라벨 버전, fold 고정 분할의 내용을 포함해야 한다. 이 정보가 없으면 평가 설계와 hold-out 독립성은 `unattributed`로 둔다.

- **A1-5**: (1) 수용·수정 — `prediction_file_hash` 또는 `fold_aurocs` hash의 exact equality는 bit-exact 재현성 검증이지, D-048 이후 seed·적합 변동이 허용되는 상태의 드리프트 감지가 아니다. 결정론 경로에서만 exact 대조를 쓰고, 학습·seed가 있는 경로에서는 seed와 함께 사전 등록된 허용 범위 또는 불확실성 기준으로 비교한다. 허용 범위 수치는 문서에 없으므로 새로 지어 쓰지 않으며, 4자리 AUROC 앵커를 재도입하지 않는다. 허용 기준이 없으면 해당 비교는 판별 불가로 둔다.

- **A1-6**: (1) 수용·수정 — 기록 생성 지점은 src 한 곳으로 고정하지 않는다. 실행 스크립트는 `requested_branch_list`·`execution_mode`·`execution_id`를, src는 실제 resolved effective 상태와 `prediction_file_hash`를 기록하도록 한다. 두 기록은 `execution_id`로 결합한다. src가 requested를 알 수 없으므로 requested를 src sidecar에 넣지 않는다. effective 상태를 실제로 관찰할 수 없으면 `verified`가 아니다.

- **A1-7**: (1) 수용·수정 — confidence는 주관 등급으로 두지 않고 근거 조합으로 정의한다. `verified`: sidecar 존재, validator 통과, `prediction_file_hash` 일치, `execution_id` 연결. `inferred`: 코드/스크립트 경로, run log, 파일 태그, 값 일관성 중 독립 근거가 2개 이상이고 모순이 없어야 하며, code hash 또는 run log가 있으면 high, 태그+run log만 있으면 medium, 태그 또는 값 단독이면 low로 둔다. `unattributed`: 독립 근거가 없거나 모순. `ru90_shape_triple`은 태그와 macro 근접만으로 high가 될 수 없다.

- **A1-8**: (1) 수용·수정 — 공식 7-branch 기준선 `0.6227`은 현재 `unverified-hold`이고, 과거 산출물에 sidecar가 없으므로 baseline의 구성 귀속이 `verified`로 확인되기 전에는 승격 비교의 전제 충족을 주장하지 않는다. baseline이 `unattributed`이면 해당 비교는 `verified` 판정 대상이 아니며, 새 스키마로 재실행하거나 불변 산출물에서 소급 검증을 통과해야 한다.

- **R1-1**: (1) 수용·수정 — sidecar가 config 객체만 기록하면 src가 env를 읽지 않는 현재에서는 요청 구성과 실제 구성을 구분하지 못한다. 수정안은 config가 아니라 실제 계산에 도달한 resolved state를 기록하고, requested/effective 불일치를 명시하는 것이다. src가 env를 읽도록 수정되지 않거나 config가 계산에 반영됨이 코드 대조로 확인되기 전까지는 sidecar의 `verified` 신뢰도는 성립하지 않는다.

- **R1-2**: (1) 수용·수정 — 값 앵커가 제공하던 기능 중 하나는 실행 간 구성 불변성 확인이다. `prediction_file_hash`는 동일 산출물의 bit-exact 재현성만 확인하고, SCREEN_ONLY 경로가 base 앙상블을 건드렸는지는 base 예측 산출물과 provenance를 비교해야 한다. 결정론 경로에서는 exact/hash 대조가 가능하지만, 수치 정밀도·seed·적합 변동이 있으면 exact 대조는 오거부를 낳을 수 있다. 허용 기준은 사전 등록하고, 4자리 앵커를 재도입하지 않는다.

- **R1-3**: (1) 수용·수정 — import graph 수집은 v1의 필수 요건으로 두지 않는다. 고정 파일 closure의 content hash를 우선 쓰고, 동적 import·생성 코드·외부 모듈이 실행에 영향을 줄 가능성이 있으면 runtime import manifest를 추가한다. 관련 없는 파일 변경으로 hash가 자주 달라지는 문제는 closure 범위를 최소화하고, 필요하면 repo tree hash를 보조 필드로 둔다. closure가 불완전하면 `code_hash`는 완전 코드 상태 특정이라 주장하지 않는다.

- **R1-4**: (1) 수용·수정 — src가 env를 읽지 않으므로 src가 `requested_branch_list`·`execution_mode`를 기록하는 것은 성립하지 않는다. sidecar를 두 단계로 만든다. 실행 스크립트는 requested manifest를, src는 effective manifest와 예측 파일 hash를, orchestrator는 `execution_id`로 결합한다. src가 requested를 알 수 없는 필드를 src sidecar에 넣는 초안은 폐기한다.

- **T1-N1**: (1) 수용·수정 — config가 branch 포함·가중치·모드의 단일 진실원천이라는 가정은 현재 문서의 관측과 충돌할 수 있다. `src/`가 env를 읽지 않고, config 기본값과 YAML이 가중치를 설정하지 않으며, 가중치 0이 branch 제외와 동등한지 미확인이다. 따라서 sidecar는 config 선언값이 아니라 계산 경로에서 관찰된 effective state를 기록해야 하고, config와 effective가 다르면 mismatch를 표시한다. 현재 코드가 단일 진실원천인지 문서만으로는 판별 불가이므로, 확인 전 `verified`를 금지한다.

- **T1-N2**: (1) 수용·수정 — 예측 파일과 sidecar를 같은 transaction으로 쓰는 것만으로는 기록-실행 불일치를 막지 못한다. `execution_id`, 예측 파일의 출처(cache hit/miss, 이전 실행, 수동 복사), `prediction_file_hash`, sidecar hash를 연결하고, 가능하면 불변 저장소에 쓴다. sidecar가 설정 로더 상태만 기록하고 실제 계산 상태를 기록하지 않으면 `verified`가 아니다. 동일 transaction은 동시 생성만 보장할 뿐, 내용 적정성을 보장하지 않는다.

- **T1-N3**: (1) 수용·수정 — `code_hash`만으로는 동일 행동을 특정하지 못한다. package 버전, GPU/CUDA/BLAS/thread, metric 코드, 전처리/평가 코드는 `environment_hash`·`data_version`·`metric_code_hash`로 분리해 함께 대조한다. `code_hash`가 동일해도 environment가 다르면 provenance 불일치로 둔다. 동적 코드·외부 모듈이 closure에 없으면 `code_hash` 완전성은 `미확인`으로 표시한다.

- **T1-N4**: (1) 수용·수정 — exact hash 재현은 학습 파라미터 허용 이후 무조건 성립하지 않는다. D-048으로 `seed std 0.00000`이 더는 성립하지 않고, RU-97 §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. 따라서 bit-exact 재현성과 오염/드리프트 감지를 분리한다. 오염 검사는 provenance 불변성과 base 예측 산출물의 변경 여부를 확인하고, seed 변동이 있는 비교는 사전 등록된 불확실성 기준으로만 한다. 허용 기준이 없으면 판별 불가로 둔다.

- **T1-N5**: (1) 수용·수정 — 태그·일시·값은 구성의 대리변수로 불충분하다. 특히 `ru90_shape_triple`은 공식 태그를 쓰지만 보고 §8에서 5-branch 산출물로 확인되었고, macro `0.6226`이 `0.6227`에 가까운 것은 우연으로 기록됐다. 따라서 소급 귀속은 태그 단독으로 하지 않고, code hash, run log, script path, sidecar, 값 일관성, 모순 부재를 조합해 판정한다. 이 조합이 없으면 `unattributed`, 단독 태그 또는 값만 있으면 `inferred low`로 둔다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)

- **현재 값 앵커 오염 검사는 유효한 통과/실패 판정 도구가 아니다.**  
  근거: 보고 §6에서 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0` export에도 SMAD4 `0.4661`, PBRM1 `0.5408`로 4자리까지 동일했고, 기록된 5-branch 앵커 `0.4421`/`0.5553`과 다르다. 보고 §8에서 `src/`는 `os.environ`/`getenv`을 읽지 않아 실행 스크립트의 `ICF_*` export가 무효라고 확인됐다. `PROJECT.md` §3.4의 `D-050`도 재설계 완료 전 오염 검사 통과 주장을 무효로 둔다.

- **저장된 예측 파일은 구성 재구성에 불충분하다.**  
  근거: `PROJECT.md` 배경과 §3.2 `D-050`에서 저장 예측 파일의 키는 `task_dir`, `fold_indices`, `fold_aurocs`, `fold_auroc_mean`, `per_fold`뿐이며, branch list·가중치·코드 해시·데이터 버전이 없다. 따라서 과거 산출물을 특정 구성에 귀속시키는 데 필요한 식별자가 현재 파일에 없다.

- **공식 7-branch 기준선 `0.6227`은 `verified`로 사용할 수 없다.**  
  근거: `PROJECT.md` §3.2는 `D-050` 이후 이 값을 `unverified-hold`로 명시한다. 보고 §8은 `SH`/`SJ`를 켜는 문서화된 수단이 동작하지 않아 공식 7-branch 기준선을 재현할 경로가 현재 없다고 확인한다.

- **`run_ru90_shape_triple` 산출물은 공식 7-branch가 아니다.**  
  근거: 보고 §8에서 `eval_v121.sh`는 정의상 5-branch arm이고, `src/models/config.py` 기본값은 `weight_sh=0.0`, `weight_sj=0.0`, `weight_bs=0.0`이며, 사용된 YAML은 브랜치 가중치를 설정하지 않는다. `src/`가 env를 읽지 않으므로 스크립트 export는 무효다. 따라서 `ru90_shape_triple_official50_bf16` 산출물은 공식 7-branch가 아니라 5-branch 산출물로 잘못 이름 붙였다고 확인됐다. macro `0.6226`이 `0.6227`에 가까운 것은 우연이다.

- **provenance 기록은 requested/env 값이 아니라 실제 계산에 도달한 effective 상태를 기록해야 한다.**  
  근거: `ICF_*` export가 `src/`에 도달하지 않고, 스크립트 머리말·태그·env 요청과 실제 산출 구성이 분리될 수 있다는 관측이 있다. 따라서 “스크립트가 요청한 구성”과 “실제로 계산에 쓰인 구성”을 구분하지 않으면 provenance 대조가 구성 재구성을 보장하지 못한다.

- **provenance에 코드·데이터·fold manifest·예측 파일 식별자가 필요하다.**  
  근거: 현재 예측 파일에는 branch list·가중치·코드 해시·데이터 버전이 없어 과거 산출물 귀속이 불가능하다. `PROJECT.md` §4.1은 대응 per-fold Δ의 정밀도가 fold 짝짓기에 의존한다고 명시하므로, fold 분할의 동일성을 특정할 manifest 해시가 필요하다. `D-048` 이후 seed·적합 변동이 허용되고 `RU-97` §8에서 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`이 실측됐으므로, 산출물 자체를 특정할 예측 파일 해시 또는 fold AUROC 식별자도 필요하다.

- **값 앵커가 제공하던 수치 드리프트/결정론 기능은 provenance 대조만으로 대체되지 않는다.**  
  근거: provenance 대조는 기록된 구성이 일치하는지 확인하지만, 동일 구성에서 값이 실제로 동일한지는 확인하지 않는다. `D-048`으로 `seed std 0.00000`이 더는 성립하지 않고, `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. 따라서 구성 일치와 수치 불변은 별개 검증이다.

- **과거 산출물은 태그·값 단독으로 구성을 특정할 수 없다.**  
  근거: `ru90_shape_triple`은 공식 7-branch를 연상시키는 태그를 쓰지만 보고 §8에서 5-branch 산출물로 확인됐다. macro `0.6226`이 `0.6227`에 가까운 것은 우연으로 기록됐다. 기록된 5-branch 앵커도 현재 코드로 재현되지 않는다. 따라서 태그 또는 AUROC 근접성 단독은 구성 귀속 근거가 되지 못한다.

## 반박 (증거가 반박함)

- **“실행 스크립트의 `ICF_*` export가 branch list·가중치·모드를 통제한다”는 주장은 반박된다.**  
  근거: 보고 §8에서 `src/`는 환경변수를 읽지 않는 파일이 0개라고 확인됐다. 보고 §6에서 `ICF_FIXED_HEAD_{BS,SH,SJ}_WEIGHT=0.0`를 export해도 SMAD4·PBRM1 값이 4자리까지 바뀌지 않았다.

- **`run_ru90_shape_triple`이 공식 7-branch score를 만들었다는 주장은 반박된다.**  
  근거: 보고 §8에서 해당 경로는 `eval_v121.sh`의 5-branch arm을 거치며, `config.py` 기본값과 YAML이 `SH`/`SJ`/`BS` 가중치를 켜지 않는다. `src/`가 env를 읽지 않으므로 스크립트 export는 계산에 도달하지 않는다.

- **기록된 5-branch 앵커가 현재 코드로 재현된다는 주장은 반박된다.**  
  근거: 보고 §6·§8에서 현재 5-branch 경로 실측은 SMAD4 `0.4661`, PBRM1 `0.5408`, macro `0.6226`이고, 기록된 앵커는 `0.4421`, `0.5553`, `0.6171`이다. 두 실행이 결정론적으로 동일했으므로 잡음이 아니다.

- **공식 7-branch 기준선 `0.6227`이 `verified`라는 주장은 반박된다.**  
  근거: `PROJECT.md` §3.2는 `D-050` 이후 이 값을 `unverified-hold`로 명시한다. 보고 §8은 공식 7-branch를 재현할 문서화 경로가 현재 없다고 확인한다.

- **스크립트 머리말·태그·env export만으로 산출물 구성을 특정할 수 있다는 주장은 반박된다.**  
  근거: `run_ru90_shape_triple.sh` 머리말은 `SCREEN_ONLY`가 5-branch BASE arm을 만든다고 적지만 관측은 7-branch macro에 가까웠다. 코드 확인 결과 해당 산출물은 5-branch였다. 태그 `official50`도 실제 구성과 어긋났다.

- **provenance 대조만으로 값 앵커의 모든 기능을 대체한다는 주장은 반박된다.**  
  근거: provenance 대조는 구성 일치 확인에 해당하지만, 동일 구성에서 seed·적합·수치 정밀도·환경 차이로 값이 달라지는 드리프트/결정론 문제를 자동으로 확인하지 않는다. `D-048` 이후 seed·적합 변동이 허용되고 `RU-97` §8에서 동일 구성 seed 변경으로 macro 폭 `0.0200`이 실측됐다.

## 판별 불가 (증거가 가르지 못함)

- **현재 5-branch 실측값이 기록된 5-branch 앵커와 다른 원인은 판별 불가하다.**  
  보고 §8은 원인을 코드 변경 / 매니페스트 / 전처리 / 수치 정밀도로 열어두고 “미확인”이라고 명시한다. 현재 증거는 값 불일치만 확인하고, 어떤 요인이 원인이었는지는 가르지 못한다.

- **`config.py`의 가중치 0이 실제로 branch 계산 제외와 동등한지는 판별 불가하다.**  
  보고 §6에서 env 가중치 0이 출력에 영향을 주지 않았지만, 이는 `src/`가 env를 읽지 않기 때문일 수 있다. `config.py` 기본값 `weight_sh=0.0` 등이 앙상블 계산에서 branch를 실제로 제외하는지는 문서에 코드 대조 결과가 없다.

- **과거 `ru90_shape_triple` 실행이 당시에는 shape margin을 포함했는지는 판별 불가하다.**  
  보고 §8은 “과거 `ru90_shape_triple` 실행이 당시에는 shape margin을 포함했는지”를 아직 확인하지 않은 것으로 남긴다. 현재 코드 확인은 과거 실행 시점의 구성을 특정하지 못한다.

- **공식 7-branch 기준선 `0.6227`의 원래 산출물이 어떤 구성·코드·데이터에서 나왔는지는 판별 불가하다.**  
  과거 산출물에 sidecar가 없고, 저장 예측 파일에도 branch list·가중치·코드 해시·데이터 버전이 없다. `PROJECT.md`는 이 기준선을 `unverified-hold`로 둔다.

- **제안된 `code_hash`의 범위가 실행 코드를 완전히 특정하는지는 판별 불가하다.**  
  `src/`, 실행 스크립트, config YAML, manifest, metric 코드, package 버전, GPU/CUDA/BLAS/thread, 동적 import, 생성 코드, C extension 등이 모두 실행에 영향을 줄 수 있으나, 문서에는 이 범위를 완전히 특정하는 코드 감사 결과가 없다.

- **`data_version`과 `fold_manifest_hash`가 평가 설계를 완전히 특정하는지는 판별 불가하다.**  
  원본 WSI, 임베딩/특징 파생물, 전처리 산출물, 라벨, 배제 목록, context/query 슬라이드 목록, train/test 분리, 통계량 fit 범위 등이 모두 AUROC에 영향을 줄 수 있으나, 문서에는 현재 fold manifest의 실제 내용과 data version의 범위가 없다.

- **D-048 이후 exact hash 또는 exact equality가 드리프트 감지에 적합한지는 판별 불가하다.**  
  `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. 그러나 어떤 허용 범위 또는 불확실성 기준으로 드리프트를 판정할지는 문서에 없다. 허용 기준이 없으면 exact hash 불일치를 구성 실패, seed 변동, 수치 정밀도 차이 중 무엇으로 구분할 수 없다.

- **provenance 대조만으로 SCREEN_ONLY 같은 side effect를 완전히 감지할 수 있는지는 판별 불가하다.**  
  branch list·가중치가 동일해도 집계 입력, 캐시, 모델 상태, 전처리 상태, 수치 경로가 바뀌면 base 예측이 바뀔 수 있다. 현재 증거는 env 경로 무효만 확인하고, side effect 전체를 배제하지 못한다.

- **과거 산출물의 `inferred` confidence를 high/medium/low로 구분할 기준은 아직 확립되지 않았다.**  
  태그·값 단독은 신뢰할 수 없음을 확인했지만, code hash, run log, script path, 값 일관성, 모순 부재 등을 어떻게 조합해 confidence를 부여할지는 문서에 확정 기준이 없다.

## 실행 무효 (실행·계측 결함으로 판단 불가)

- **현재 값 앵커 기반 오염 검사 실행은 무효다.**  
  이유: `src/`가 env를 읽지 않아 intended base/screen arm을 통제할 경로가 없고, 기록된 5-branch 앵커가 현재 코드로 재현되지 않는다. 따라서 4자리 AUROC 일치로 오염 검사 통과를 주장할 수 없다.

- **재설계 완료 전 오염 검사 통과를 주장하는 모든 실행은 무효다.**  
  이유: `PROJECT.md` §3.4 `D-050`이 명시한다.

- **공식 7-branch 기준선 `0.6227`을 `verified` 기준선으로 사용하는 실행은 무효다.**  
  이유: `PROJECT.md` §3.2는 `unverified-hold`로 명시하고, 보고 §8은 공식 7-branch 재현 경로가 현재 없다고 확인한다.

- **`ru90_shape_triple_official50_bf16` 산출물을 공식 7-branch score로 사용하는 실행은 무효다.**  
  이유: 보고 §8에서 해당 산출물은 5-branch 산출물이며 공식 태그로 잘못 이름 붙였다고 확인됐다.

- **공식 7-branch score를 입력으로 요구하는 후보 실행은 현재 무효다.**  
  이유: 보고 §8은 공식 7-branch score를 만들 문서화 경로가 현재 없다고 확인한다.

- **기록된 5-branch 앵커와 현재 5-branch 실측값의 4자리 대조로 구성 동일성을 판정하는 실행은 무효다.**  
  이유: 앵커가 재현되지 않고, 값 불일치의 원인이 미확인이다. 값 불일치가 구성 차이인지 드리프트인지 구분할 수 없다.

- **`D-042` 승격 근거 `+0.56%p`, 게이트 ① 결과, 게이트 ② 실적을 `verified`로 인용하는 실행은 무효다.**  
  이유: `PROJECT.md` §3.2 `D-050`은 계측 결함으로 이 기준선을 재현할 경로가 현재 없으므로, 해당 승격 근거·게이트 결과를 `verified`로 인용하지 않는다고 명시한다.

- **D-048 이후 seed·적합 변동이 있는 경로에서 exact hash만으로 드리프트/오염을 판정하는 실행은 무효다.**  
  이유: `RU-97` §8은 동일 구성에서 seed만 바꿔도 macro 폭 `0.0200`을 실측했다. 사전 등록된 허용 범위 또는 불확실성 기준이 없으면 exact hash 불일치를 판정할 수 없다.

## 좌석 공백 (기권·실패로 빠진 관점)

- 없음. 입력에서 기권한 좌석은 없다.

## 다음 행동 (권한 등급과 함께)

- **[사용자 결정] 공식 7-branch 기준선 지위를 확정하라.**  
  (a) branch·config 통제 경로를 복구한 뒤 provenance sidecar와 함께 공식 7-branch를 재실행해 `verified` 기준선을 만들거나, (b) `0.6227`을 `unverified-hold`로 유지하고 `verified` 기준선이 없을 때까지 승격 비교를 보류한다.  
  이유: 현재 공식 7-branch 재현 경로가 없고, `verified` 기준선 없이 승격 비교를 수행하면 구성 귀속이 불확실한 비교가 된다.

- **[좌석 권고] provenance v1 필드를 확정하라.**  
  최소한 다음을 기록해야 한다: `execution_id`, `created_at`, `task_dir`, `fold_indices`, `requested_branch_list`, `requested_execution_mode`, `effective_branch_list`, `effective_branch_weights`, `effective_execution_mode`, `requested_effective_mismatch`, `config_hash`, `code_hash`, `environment_hash`, `data_version`, `fold_manifest_hash`, `script_path`, `git_commit`(있으면), `seed`, `prediction_file_hash`, 기록 생성 주체, 캐시/출처 플래그, 불변 저장 여부.  
  이유: 현재 예측 파일에는 구성 식별자가 없고, requested와 effective가 분리될 수 있다.

- **[좌석 권고] `verified` 조건을 코드 감사로 제한하라.**  
  ensemble builder가 `effective_branch_list`·`effective_branch_weights`를 실제로 계산에 반영하는지, 가중치 0이 branch 제외와 동등한지, config가 단일 진실원천인지, 라벨 사용 경로와 통계량 fit 범위가 기록되는지 확인하기 전에는 sidecar 존재만으로 `verified`를 주장하지 않는다.  
  이유: `src/`가 env를 읽지 않고, config 값이 계산에 반영되는지가 문서에서 확인되지 않았다.

- **[좌석 권고] 기록 생성을 두 단계로 분리하고 `execution_id`로 결합하라.**  
  실행 스크립트는 `requested_branch_list`·`requested_execution_mode`·`execution_id`를 기록하고, `src/`는 실제 resolved effective 상태와 `prediction_file_hash`를 기록하며, orchestrator가 두 기록과 예측 파일을 `execution_id`로 바인딩한다. 예측 파일과 sidecar는 원자적·불변적으로 묶어야 한다.  
  이유: `src/`는 requested env를 알 수 없고, 사후 기록은 실제 실행과 어긋날 수 있다.

- **[좌석 권고] `code_hash`를 고정 파일 closure의 content hash로 정의하고 environment를 분리하라.**  
  v1에서는 `src/`, 실행 스크립트, config YAML, 데이터/평가 manifest, metric 코드 등 실행에 필요한 고정 파일 closure의 content hash를 쓰고, package 버전·GPU·CUDA·BLAS·thread는 `environment_hash`로 분리한다. 동적 import·생성 코드·C extension이 존재하면 runtime import manifest를 추가하고, 확인되지 않으면 `code_hash` 완전성을 `미확인`으로 표시한다.  
  이유: 스크립트·config·manifest가 실행 구성에 관여하며, package/환경 차이도 수치에 영향을 줄 수 있다.

- **[좌석 권고] `data_version`과 `fold_manifest_hash`의 범위를 평가 설계 전체로 확장하라.**  
  `data_version`은 원본 슬라이드, 임베딩/특징 파생물, 전처리 산출물, 라벨 파일, 배제 목록까지 특정해야 한다. `fold_manifest_hash`는 task, fold index, slide ID, train/test/context/query 분리, 라벨 버전, fold 고정 분할 내용을 포함해야 한다. 이 정보가 없으면 해당 차원은 `unattributed`로 둔다.  
  이유: fold 짝짓기와 hold-out 독립성, 전처리/라벨 경로가 AUROC에 영향을 줄 수 있다.

- **[좌석 권고] 과거 산출물 소급 귀속 레지스터를 만들되 태그·값 단독을 금지하라.**  
  `verified` / `inferred` / `unattributed`를 구분한다. `inferred`는 code hash, run log, script path, 값 일관성, 모순 부재 등 독립 근거의 조합이 있어야 한다. 태그 또는 값 단독은 `low`로만 둔다. `ru90_shape_triple`처럼 태그와 실제 구성이 어긋난 사례가 있으므로, 태그+macro 근접만으로 `high`를 부여하지 않는다.  
  이유: 과거 산출물에는 sidecar가 없고, 태그/값은 이미 오귀속 사례가 확인됐다.

- **[좌석 권고] bit-exact 재현성과 드리프트/오염 검사를 분리하라.**  
  결정론 경로에서만 exact hash 또는 exact equality를 쓰고, 학습·seed·적합 변동이 있는 경로에서는 seed와 함께 사전 등록된 허용 범위 또는 불확실성 기준으로 비교한다. 허용 기준이 없으면 해당 비교는 `판별 불가`로 둔다. 4자리 AUROC 앵커를 재도입하지 않는다.  
  이유: `D-048` 이후 seed·적합 변동이 허용되고, `RU-97` §8에서 동일 구성 seed 변경으로 macro 폭 `0.0200`이 실측됐다.

- **[사용자 결정] 오염 검사 대체 규칙을 사전 등록하라.**  
  provenance 대조만으로 통과를 인정할지, base arm 예측 산출물의 수치 비교를 함께 요구할지, 수치 비교의 판정 기준을 무엇으로 할지를 결정해야 한다.  
  이유: provenance 대조는 구성 일치만 확인하고, 수치 드리프트·side effect·결정론 문제는 별도 기준이 필요하다.

## 하지 말 것 (이유와 함께)

- **`0.6227`을 `verified` 공식 7-branch 기준선으로 인용하지 말 것.**  
  이유: `PROJECT.md` §3.2는 `D-050` 이후 이 값을 `unverified-hold`로 명시하고, 보고 §8은 재현 경로가 현재 없다고 확인한다.

- **`ru90_shape_triple_official50_bf16` 산출물을 공식 7-branch score로 사용하지 말 것.**  
  이유: 보고 §8에서 해당 산출물은 5-branch 산출물이며 공식 태그로 잘못 이름 붙였다고 확인됐다.

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
  이유: `ru90_shape_triple`은 공식 7-branch를 연상시키는 태그를 쓰지만 5-branch 산출물이었고, macro 근접은 우연으로 기록됐다.

- **provenance 대조만으로 값 앵커의 모든 기능을 대체한다고 주장하지 말 것.**  
  이유: provenance 대조는 구성 일치 확인이지, 동일 구성에서의 수치 드리프트·결정론·side effect 감지를 자동으로 대체하지 않는다.

- **문서에 없는 허용 오차, 임계값, confidence 수치를 새로 지어 쓰지 말 것.**  
  이유: 현재 문서에는 D-048 이후 드리프트 판정에 쓸 사전 등록 허용 기준이 없다.
