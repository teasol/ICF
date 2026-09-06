# Agent Handoff Reference — 작업 규범 · 불변식 · 실행 환경

이 문서는 In-Context Foundation (ICF) 프로젝트에서 **에이전트가 어떻게 일하는가**를 정한다.

> **정본 분리 원칙.** 이 문서는 수치를 선언하지 않는다.
> 기준선·판정 기준은 [`PROJECT.md`](PROJECT.md), 닫힌 축은 [`closed_axes.md`](closed_axes.md),
> 결정 이력은 [`decisions.md`](decisions.md), 현재 상태는 [`current_status.md`](current_status.md),
> 브랜치 수식은 [`current_architecture.md`](current_architecture.md)가 정본이다.

---

## 0. 목표 중심 연구 원칙

모든 분석·구현·실험은 **명시된 현재 목표를 달성하거나, 그 목표 달성을 가로막는 불확실성을 해소하기 위해** 수행합니다. 탐색은 허용하되, 결과가 어떤 의사결정을 바꿀 수 있는지 설명할 수 있어야 합니다. 흥미롭다는 이유만으로 연구 범위를 확장하지 않습니다. 계획은 목표 달성을 위한 수단이며, 새로운 근거에 따라 수정할 수 있습니다.

1. **현재 목표를 하나로 명시합니다.** 최종 목표 아래에 지금 달성해야 할 목표와 완료 조건을 두고 `docs/current_status.md`에 기록합니다. 여러 작업을 병렬로 수행하더라도 같은 현재 목표에 기여해야 합니다. 목표가 불명확하면 먼저 기존 기록에서 확인하고, 확인되지 않은 부분은 명시하여 목표 설정부터 진행합니다.
2. **착수 전에 세 가지를 답합니다.** “현재 목표의 어떤 장애물을 해결하는가?”, “무엇을 확인할 것인가?”, “결과에 따라 어떤 결정이 달라지는가?” 답할 수 없는 작업은 보류 목록에 둡니다. 기록의 분량은 작업 규모에 맞추며, 작은 확인 작업은 짧은 목적 설명으로 충분합니다.
3. **다음 결정을 내리는 데 필요한 만큼 탐색합니다.** 탐색 전에 시간·계산 비용의 한도와 종료 조건을 정합니다. 실험은 가설, 비교 조건, 판정 기준, 결과별 후속 행동도 정한 뒤 실행합니다. 다음 행동을 선택할 근거가 확보되면 탐색을 끝내고, 한도 내 결론이 나지 않으면 추가 탐색의 가치와 비용을 다시 판단합니다.
4. **새로운 근거 없이 같은 고민을 반복하지 않습니다.** 결정의 근거와 재검토 조건을 함께 기록합니다. 새로운 증거, 전제의 변화, 기존 실험의 결함이 확인됐을 때 재검토합니다. 후보를 기각하거나 불확실성을 해소한 실험도 유효한 진척으로 인정합니다.
5. **방향 변경을 명시적인 결정으로 남깁니다.** 현재 접근이 목표를 달성하기 어렵다는 근거가 나오면 변경 이유, 새 접근이 해결할 문제, 중단할 기존 작업을 기록합니다. 새 아이디어가 나올 때마다 현재 목표를 조용히 바꾸지 않습니다.
6. **부수 문제로 작업 범위를 확장하지 않습니다.** 작업 중 발견한 부수 문제는 현재 목표의 달성이나 결과의 신뢰성을 막는 경우에만 즉시 다룹니다. 그 외에는 기록하고 현재 작업을 계속합니다.

중단할 고민의 기준은 실패 여부나 성과의 크기가 아니라, **결과가 나와도 다음 결정에 영향을 주지 않는가**입니다. 현재 목표·진행 상태·후속 행동은 `docs/current_status.md`에, 완료된 실험과 결정의 근거는 기존 연구 기록 체계에 남겨 다음 세션에서 같은 고민을 반복하지 않도록 합니다.

---

## 1. 진행 프로세스 — 연구 단위(RU) 카드

> **왜 도입했는가.** 450커밋 총람이 측정한 바, Era 6(RU-41~78)은 RU당 커밋이 **2.6개**로
> 떨어졌고(Era 3은 14.1개) 78개 중 17개가 단일 커밋이다. "하나의 연구 질문"이 아니라
> "한 번의 시도"가 단위가 되면서, 2026-08-23 이후 **19 RU / 46커밋 / 13일간 공식 승격 0건**,
> 그리고 총람이 기록한 정정·철회 6건이 전부 이 구간에 몰렸다.
> 진행 단위를 커밋이 아니라 **연구 단위(RU)** 로 되돌린다. ([`decisions.md`](decisions.md) `D-015`)

### 1.1 착수 전 — RU 카드를 먼저 등록한다

GPU를 쓰든 안 쓰든, 하나의 연구 질문에 답하려는 작업은 시작 전에 카드를 만든다.

```bash
$PYTHON scripts/docs/ru.py open --title "<한 줄 제목>"
```

카드의 필수 필드는 **여섯 개**이고, 하나라도 비어 있으면 착수하지 않는다.

| 필드 | 내용 | 비어 있으면 |
|:---|:---|:---|
| `question` | 답하려는 연구 질문 하나 | 무엇을 하는지 모르는 것이다 |
| `hypothesis` | 사전에 선언한 예상 | 사후에 결과를 해석하게 된다 |
| `criteria` | **판정 기준을 결과를 보기 전에 고정** | §214·§220의 실패 양식이다 |
| `budget` | GPU-h 또는 소요 상한 | 탐색이 끝나지 않는다 |
| `kill` | **중단 조건** — 어떤 관측이 나오면 즉시 종료하는가 | 미세 진동이 시작된다 |
| `next_actions` | **결과별 후속 행동** (통과 시 / 실패 시 각각) | 결과가 다음 결정을 바꾸지 못한다 |

> `next_actions`가 통과·실패 양쪽에서 같다면 그 작업은 하지 않는다.
> §0-3의 *"결과가 나와도 다음 결정에 영향을 주지 않는가"* 가 중단 기준이다.

### 1.2 종료 시 — 이력에 append하고 결정을 남긴다

```bash
$PYTHON scripts/docs/ru.py close --id RU-79     # 관측·결정을 채운 뒤 실행
```

- 관측과 결정은 [`history/research_units_all.md`](history/research_units_all.md)와 `.json`에
  **append**된다. 총람은 이제 사후에 재구성하는 것이 아니라 실시간으로 쌓인다.
- **규범·기준·구성이 바뀌었다면** [`decisions.md`](decisions.md)에 결정 레코드를 추가한다.
- **축을 닫거나 열었다면** [`closed_axes.md`](closed_axes.md)에 반영한다. 축을 닫을 때
  `기각 기전`·`경계 안`·`경계 밖`·`재개 조건` 네 필드가 모두 필요하다.
- 커밋 메시지 본문에 RU ID를 적는다 (`feat(sh): ... (RU-79)`).

### 1.3 정정은 본문을 덮어쓴다

낡은 서술을 주석으로 덧붙이지 않는다. **본문은 항상 현행 사실만 담고**, 시점 한정 사실과
번복 이력은 [`decisions.md`](decisions.md)에만 남긴다.
§226에서 낡은 사실 2건이 브리핑 팩을 통해 7개 에이전트 전부에 전파된 사고가 이 규칙의 근거다
(`D-014`).

### 1.4 문서 정합성은 테스트가 지킨다

```bash
$PYTHON -m unittest tests.test_docs_consistency -v
```

기준선 수치가 [`PROJECT.md`](PROJECT.md) 밖에서 다시 선언되거나, 존재하지 않는 축 ID·결정
ID가 인용되면 실패한다. `scripts/run_tests.sh`에 포함되어 있다.

---

## 2. 프로젝트 개요

**ICF**는 미리 추출된 MIL 패치 임베딩(UNI2, 1536D) 위에서 병리 WSI를 분류하는 **in-context
분류기**다. 활성 구성은 **학습 파라미터 0개의 완전 결정론적 통계 앙상블**이다 —
Context 슬라이드만으로 within-slide PCA 기저(K=256)를 만들고, 상보적 통계 브랜치들이 각각
독립 마진을 산출한 뒤 Trimmed Mean으로 결합한다.

- 목표·기준선·승격 기준: [`PROJECT.md`](PROJECT.md)
- 브랜치 정의와 수식: [`current_architecture.md`](current_architecture.md)

### 2.1 디렉토리와 역할

| 경로 | 역할 |
|:---|:---|
| `src/models/training_free.py` | 활성 파이프라인 핵심 — 브랜치 특징 추출, Dual Ridge, 집계 |
| `src/models/branches/` | 개별 브랜치 구현 (`shj.py` 등) |
| `src/models/ct/` | CT 사전 구축 및 soft-token 할당 (현재 비교 기준에서 제외) |
| `src/models/dd_adaptive_rank.py` | BD ordered-typicality 마진 (DD는 `CA-02`로 닫힘) |
| `src/datasets/` | 데이터 로더 (`base_data.py`, `synthetic/`) |
| `scripts/node_env.sh` | **노드 종속 설정의 단일 출처** — 인터프리터·GPU 수·경로 탐색 |
| `scripts/lib/arms.sh` | arm 구성과 환경변수 주입의 단일 출처 (`icf_arm_v1xx`) |
| `scripts/eval_v121.sh` | 공식 비교 기준(5-branch) 평가 진입점 |
| `scripts/analysis/branch_screen.py` | 게이트 ① 직교성 스크리닝 |
| `scripts/analysis/branch_diagnostics.py` | 저장 마진 오프라인 재집계 (GPU 불필요) |
| `scripts/docs/ru.py` | RU 카드 생성·종료 |
| `configs/` | `baseline/` · `experiments/` · `archive/`. **루트에는 활성 진입점만 둔다** |

> ⚠️ 모델 구조·브랜치 로직·수식·하이퍼파라미터를 다루기 전에
> [`current_architecture.md`](current_architecture.md)를 정독하고 동기화한다.

---

## 3. 핵심 불변식 (Core Invariants)

회귀 스위트가 검사한다. 위반하는 변경은 병합하지 않는다.

1. **무유출 계약 (Zero Data Leakage)**
   Query 슬라이드는 within-slide PCA 기저 구축, 토큰 사전 생성, 정규화 통계에 **일절
   참여하지 않는다.**
2. **정확한 라벨 반대칭 (Exact Label Antisymmetry)**
   라벨 반전($y \to 1-y$) 시 브랜치 마진의 부호가 정확히 반전되고 확률이 여집합이 된다
   ($P(1-y) = 1 - P(y)$).
3. **결정론적 평가 (Deterministic Evaluation)**
   seed std는 `0.00000`이어야 한다. 따라서 **t-통계량·p-value·신뢰구간은 승격 판정에 사용
   금지**다 (`decisions.md` `D-003`).
   비교 기준·승격 기준·분해능 하한의 **수치는 [`PROJECT.md` §3~§4](PROJECT.md)가 정본**이다.
4. **닫힌 축 준수**
   [`closed_axes.md`](closed_axes.md)의 `경계 안`에 해당하는 제안은 성능·점수와 무관하게
   기각한다. 경계 판단이 갈리면 **임의로 한쪽을 채택하지 않고** 같은 문서 §3의 미확정
   목록으로 올린다.

---

## 4. 보고 무결성 계약 (Reporting Integrity Contract)

§214에서 실제 위반이 확인되어 신설됐다 (`decisions.md` `D-006`·`D-010`·`D-011`).

- **회귀 전량 명시 의무**: 성능이 하락한 과제를 생략할 수 없다. 상승 과제만 나열한 요약은 금지한다.
- **Sign agreement 병기 의무**: macro AUROC를 제시할 때 `n/7`을 같은 줄에 병기한다.
- **측정 방식 정확 기술**: 저장된 마진의 **오프라인 재집계**를 "전수 실측"으로 표기하지 않는다.
  신규 파이프라인 실행(`logs/`·`predictions/`에 신규 산출물 발생)과 재집계를 용어로 구분한다.
- **비교 모집단 한정**: "1위", "최고", "경신"에는 비교 대상 집합을 괄호로 한정한다.
- **승리 수식어 금지**: "폭등", "사상 최고", 이모지 강조 등을 쓰지 않는다. 수치와 부호로만 기술한다.
- **미검증 항목 표기 의무**: 수행하지 않은 검증(SEAL hold-out 등)은 침묵하지 않고 `미검증`으로 명시한다.
- **자기 검증 우선**: 문서화 전에 저장된 예측 파일로부터 독립 재계산해 대조한다.
- **Oracle 금지**: 브랜치 간 최댓값("과제별 최상 단독 브랜치")을 판정 기준·상한·미회수 성능으로
  쓰지 않는다. 참고 수치로 표시할 수는 있으나 **어떤 채택·승격·연구 방향 판정의 근거로도 쓰지 않는다.**
- **통제 비교 의무**: 어떤 기법의 **효과**를 주장하려면 **동일 실행·동일 basis·동일 fold에서
  그 기법만 토글한** 비교여야 한다. 서로 다른 실행의 표를 나란히 놓고 차이를 효과로 해석하지 않는다.

---

## 5. 실행 환경

- **`nexgem`은 로그인 노드다.** GPU가 없고 `nvidia-smi`도 설치되어 있지 않다. 학습·추론·스윕·
  대규모 전처리는 Slurm `batch` 파티션에 `sbatch`로 제출한다. 로그인 노드에서는 편집·`git`·
  문서 작업과 가벼운 문법 검사만 한다.
- **노드 스펙·선택 우선순위·로그 규약의 정본은 `/home/kimds/slurm_rules.md`다.** 요지: GPU가
  필요 없으면 `node1`~`node5`로 보내고, 필요하면 A5000(`gnode1-4`) → A6000(`gnode5`) →
  H100(`gnode6`) 순으로 **충분한 최저 등급**을 쓴다. 같은 등급에서는 한산한 노드를 고른다.
  Slurm에 GPU 타입이 등록돼 있지 않아 `--gres=gpu:<type>:N`은 동작하지 않는다 — `--nodelist`로 지정한다.
- **job 로그**는 `slurm_outputs/YYYY-MM-DD/HHMM/%x-%j.{out,err}`에 남긴다. 경로는 절대경로로 적고
  제출 전에 `mkdir -p`한다 (Slurm은 디렉토리를 만들지 않는다). **제출 전 사용자 승인을 받는다.**
- **노드 종속 설정은 `scripts/node_env.sh` 하나로 모았다.** 인터프리터·경로는 이 스크립트가
  보고하는 값을 쓰고, 문서에 적힌 과거 값을 신뢰하지 않는다. `NGPU`는 Slurm 할당
  (`SLURM_GPUS_ON_NODE` → `CUDA_VISIBLE_DEVICES` → `nvidia-smi -L` 순)에서 유도된다.
- **Python**: `uv`로 관리하는 `ICF/.venv` (Python 3.12.11).
  재구축은 `uv venv --python 3.12 .venv && uv pip install -r requirements.txt`.
- **스레드 상한**: 다중 코어 노드에서 OpenMP 과다구독을 막기 위해 `OMP_NUM_THREADS=8`로 제한한다.
- **드라이버 호환성**: `.venv`의 torch 빌드가 요구하는 CUDA 런타임과 노드 드라이버가 맞아야 한다.
  cu130 빌드는 드라이버 **≥ 580**을 요구한다. 노드별 확인 현황은
  [`current_status.md`](current_status.md)에 있다. 새 노드를 처음 쓸 때는 점검 job으로
  `torch.cuda.is_available()`를 실측한다 — `device_count()`는 드라이버가 맞지 않아도 값을 반환한다.

---

## 6. 표준 검증 명령

> 4·6·7번은 GPU를 쓴다. **로그인 노드에서 직접 실행하지 않고** `sbatch` 스크립트 안에서
> 호출한다. 노드 선택과 로그 경로는 §5를 따른다.

```bash
# 1. 환경 로드
source scripts/node_env.sh && echo "$PYTHON / NGPU=$NGPU"

# 2. 회귀 스위트 (147 tests, ~22s, CPU) — sbatch로 node1~5에 제출한다
bash scripts/run_tests.sh

# 3. 단일 모듈
bash scripts/run_tests.sh test_bd_branch.py

# 4. 공식 비교 기준 평가 (5-branch)
bash scripts/eval_v121.sh <gpu_id> <tag>

# 5. 저장 마진 오프라인 재집계 (GPU 불필요)
$PYTHON scripts/analysis/branch_diagnostics.py --tag v121_baseline

# 6. GPU 실행 대기 (폴링 금지 — 아래 참조)
bash scripts/run_and_wait.sh <runner.sh> <tag> "<분석 명령>"

# 7. SEAL 10-task hold-out (현재 유보 — PROJECT.md §3.1)
bash scripts/run_v120_seal_multi_gpu.sh <tag>
```

> ⚠️ **GPU 실행을 폴링으로 기다리지 말 것.**
> 진행 상황을 반복 확인하면 확인 1회당 도구 호출과 토큰이 소모된다(§218에서 5시간 사용량
> +7%p의 대부분이 여기서 발생했다). 대신 `scripts/run_and_wait.sh`를 **`run_in_background`로
> 단 한 번** 실행한다. 이 스크립트가 완주까지 블로킹한 뒤 완료 요약(과제별 fold-mean AUROC,
> 오염 검사, 분석 결과)을 한 번에 출력하므로 완료 알림 자체에 답이 담긴다.
> 중간 확인이 꼭 필요하면 최소 10분 간격으로 제한한다.

```bash
# (참고) 6번의 실제 사용 예
bash scripts/run_and_wait.sh scripts/run_v121_shape_screen.sh v121_sh_v2 \
  "PYTHONPATH=$PWD $PYTHON scripts/analysis/branch_screen.py --tag v121_sh_v2 --candidate m_sh"
```

_by Claude Opus 5 on nexgem at 2026-09-06_
