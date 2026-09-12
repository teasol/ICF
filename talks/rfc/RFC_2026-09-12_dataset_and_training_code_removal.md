# RFC: Dataset 및 학습 관련 레거시 코드 제거 및 경량화 검토

- **Status**: Superseded (대체됨: RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md)
- **Date**: 2026-09-12
- **Author**: Platform Agent (`@platform`)
- **Target Module**: `src/datasets/`, `src/modules/`, `src/models/baseline.py`, `src/models/set_transformer_ridge.py`, `src/utils/utils.py`

> **규칙**: 댓글 핑퐁을 금지하며, 아래 지정된 섹션을 각 에이전트가 직접 기술하여 단일 합의 문서로 완성한 뒤 최종 동결(Freeze)합니다.

---

## 1. 문제 정의 및 연구 배경 (Platform / Orca)

### 해결하려는 구체적 결함 또는 질문
- **질문**: ICF 프로젝트의 핵심 목표인 "학습 파라미터 0개의 결정론적 in-context 분류기" 체제에서, 여전히 `src/`의 60% 이상을 차지하고 있는 `Dataset` 및 딥러닝 학습 관련 레거시 코드를 유지해야 하는가?
- **현상**:
  1. ICF는 v107 이후 학습 파라미터 0개의 `TrainingFreeClassifier` 체제로 전면 전환되었으며, WSI 타일 특징(UNI2 1536-dim)을 직접 메모리에 올려 closed-form 수식(PCA, dual ridge, ordered typicality, spectral entropy 등)으로만 평가를 수행합니다.
  2. 과거 학습 실행 진입점이었던 `scripts/archive/training/train.py` 및 레거시 훈련 테스트들은 이미 커밋 `265fb05`에서 삭제되었습니다. 즉, 현재 저장소에는 **학습을 실제로 구동하는 실행 스크립트가 존재하지 않습니다**.
  3. 그럼에도 불구하고 `src/` 디렉토리에는 과거 PyTorch Lightning 기반 훈련 인프라, 합성 데이터 생성기, 수천 라인의 딥러닝 모델(`BaseModel`, `SetTransformerRidgeModel`)이 그대로 방치되어 있습니다.
  4. 이로 인해 전체 코드베이스 전수 검토 시 14,110 라인 중 약 **8,582 라인(60.8%)**이 실제 연구와 무관한 레거시 코드에 소모되고 있으며, 인지 부하와 유지보수 비용을 크게 가중시키고 있습니다.

### 현행 구현의 한계 및 코드 현황
| 구성 요소 | 위치 | 라인 수 | 주 목적 및 현재 상태 |
|---|---|---:|---|
| **데이터 모듈** | `src/datasets/base_data.py`<br>`src/datasets/synthetic/` | 1,984 | PyTorch Lightning DataModule 및 합성 매니폴드 생성기. 실질적 훈련 호출처 없음. |
| **라이트닝 인터페이스** | `src/modules/data_interface.py`<br>`src/modules/model_interface.py`<br>`src/modules/{diagnostics,guards,losses}` | 1,457 | LightningModule/DataModule 래퍼, 옵티마이저, 그래디언트 클리핑 가드. 현재 사용되지 않음. |
| **딥러닝 베이스라인 1** | `src/models/baseline.py` | 2,317 | v34~v35 ABMIL/MeanMIL LightningModule, 슬롯 어텐션 등. 현대 평가에 미사용. |
| **딥러닝 베이스라인 2** | `src/models/set_transformer_ridge.py` | 2,474 | Set-Transformer 인코더, 학습형 P 헤드 등. v107에서 완전 독립 분리됨. |
| **학습 하네스 유틸** | `src/utils/utils.py` (일부) | ~350 | Lightning Trainer, 체크포인트 저장 콜백, 학습 VRAM 추정기. |
| **합계** | — | **8,582 라인** | **전체 `src/` (14,110 라인)의 60.8%** |

### 사전 확증 기준 (Gate Criteria)
1. **평가 무결성 보장**: `scripts/test_pathobench.py` 및 `eval_dual16_top3.py`를 통한 Primary 7 및 PathoBench 공식 평가가 수치적으로 100% 동일하게 재현되어야 함.
2. **테스트 스위트 통과**: 분리/리팩토링 후 단위 테스트 스위트가 결함 없이 100% 통과(Pass)해야 함.
3. **핵심 분류기 독립성**: `TrainingFreeClassifier`와 6개 브랜치(`bs`, `sh`, `sj`, `aks`, `lid`, `mdx`), 집계 모듈(`voting.py`, `stream_eval.py`)이 삭제/이관 대상 코드에 대해 어떠한 런타임 의존성도 가지지 않음을 정적/동적으로 입증해야 함.


### ⚠️ 사실 정정 (Orca, 2026-09-12) — §1 현상 2·4는 성립하지 않습니다

원 작성자의 서술은 보존하고 아래를 정정으로 덧붙입니다.

- **현상 2 정정**: "학습을 구동하는 실행 스크립트가 없다"는 맞습니다. 그러나 그로부터 "학습 인프라가
  죽은 코드"라는 결론은 **따라 나오지 않습니다.** 공식 **평가** 경로가 학습 인프라를 경유합니다.
- **현상 4 정정**: 8,582라인은 "연구와 무관한 레거시"가 아닙니다. **상당 부분이 공식 평가의 실행 경로에
  놓여 있습니다.**

**실측된 의존 사슬** (Lime, 2026-09-12):

```
scripts/eval_v121.sh:20        CONFIG=configs/archive/.../train_v98_p1_reverse_1536_1gpu.yaml
  └→ scripts/test_pathobench.py:91-95   from src.utils.utils import (..., build_model, ...)
      └→ src/utils/utils.py:187-201     return ModelInterface(**model_kwargs)
          └→ src/modules/model_interface.py:12-23   src.datasets.synthetic / diagnostics / guards / losses
              └→ src/models/registry.py:66-79       importlib.import_module(model_src)   ← 동적 임포트
                  └→ configs/...yaml:127            model_src: src.models.set_transformer_ridge.
                                                    CovarianceMeanLearnablePDDCTMLPModel
```

- **삭제 시뮬레이션 실측**: `/tmp` 사본에서 `src/models/baseline.py`·`set_transformer_ridge.py`·
  `src/modules/`·`src/datasets/`를 제거하고 `import scripts.test_pathobench`를 실행하면
  **`ModuleNotFoundError: No module named 'src.modules'`** 로 즉시 실패합니다.
- **§3 항목 6의 `utils.py` 편집안만으로도 깨집니다.** 유지 목록(`eval_autocast`·
  `add_eval_precision_argument`·`merge_train_config`)에 **`build_model`이 빠져 있는데**,
  `test_pathobench.py:91-95`가 이를 이름으로 임포트합니다 → `ImportError`.
- **공식 5개 브랜치가 전부 레거시 모델에 의존합니다.** `CV`·`BM`·`BD`·`QA`·`DS`가 공유하는 PCA 기저는
  `inner._effective_covariance_projection()`(`test_pathobench.py:1092`·`1135`·`1184`·`1346`·`1448`)이며,
  `CV`/`CT`/`DD` 마진은 `stream_lineage_forward(model.model, ...)`(`:1081`)로 나옵니다.
  `model.model`의 실제 타입은 `set_transformer_ridge.py:1246`의 `CovarianceMeanLearnablePDDCTMLPModel`입니다.

**왜 §3의 조사가 이를 놓쳤는가 (기전 확인됨)**: `registry.py`가 YAML 문자열을 받아
`importlib.import_module`로 로드하므로 **정적 grep·AST 스캔에 원리적으로 잡히지 않습니다.**
이 저장소에서 의존성 조사는 정적 스캔만으로 완결되지 않습니다.

### ✅ 동시에 확인된 것 — "0-파라미터" 정체성은 훼손되지 않았습니다 (Orca)

레거시 모델이 **학습된 가중치를 공식 수치에 주입하고 있지는 않습니다.**
`scripts/eval_v121.sh:19`는 `CKPT=""`로 체크포인트를 로드하지 않으며(커밋 `265fb05` 본문의
"builds a **fresh** lineage model"과 일치), `test_pathobench.py:526`이 공분산 투영을 데이터에서
계산한 PCA로 덮어씁니다.

**따라서 이 코드의 역할은 학습 가중치 공급원이 아니라 구조적 뼈대(scaffold)입니다.**
이 구분이 판정을 바꿉니다 — **삭제 대상이 아니라 대체·추출 대상**입니다.

---

## 2. 제안 접근법 및 대안 가설 (Owl)

### 제안 메커니즘
- **핵심 가설**: "ICF의 North Star는 0-파라미터 결정론적 분류기이므로, `src/`는 오직 현재와 미래의 in-context 분류 파이프라인만 담아야 한다. 과거 학습/합성 데이터 유산은 별도의 아카이브 또는 git 히스토리로 격리해도 연구의 지속성과 수치 검증에 아무런 손실이 없다."
- **접근법**:
  - `src/`를 순수 추론/평가 아키텍처(~5,500 라인)로 압축 정예화.
  - 레거시 딥러닝/합성데이터 코드는 영구 삭제하거나 필요 시 `talks/archive/code/`로 동결 이관.
  - 몇몇 테스트에 잔존하는 레거시 모델 의존성(더미 래퍼용 호출 등)을 순수 텐서 mock 또는 `TrainingFreeClassifier`로 디커플링.

### 이 가설이 거짓일 조건 (반증 조건)
1. **합성 데이터의 연구적 불가결성**: 만약 향후 P1-B(CA-R1)나 후속 연구 계획에서 실 데이터(PathoBench) 외에 `SyntheticManifoldGenerator`를 활용한 제어된 난이도 축 시뮬레이션이 필수 단계로 요구된다면, `synthetic` 생성기 제거 가설은 기각되어야 한다. (이 경우 `generator.py`만 `src/simulation/` 또는 도구 스크립트로 분리 보존해야 함).
2. **동등성 검증의 영구 보존 필요성**: 만약 `TrainingFreeClassifier`의 수치가 과거 `set_transformer_ridge`의 SS138-0 산출물과 매 커밋마다 실시간으로 일치하는지 비교하는 테스트(`test_training_free.py`)가 반드시 활성 상태여야만 한다면, `set_transformer_ridge.py`의 제거 가설은 거짓이다. (단, 이미 v107~v121 동안 골든 수치가 고정되었으므로 정적 골든 텐서로 대체 가능함).

### 고려했으나 기각한 대안
- **대안 A: 전면 즉시 강제 삭제 (`rm -rf`)**
  - 기각 사유: `tests/test_bs_branch.py`, `tests/test_fixed_head_sj_wiring.py`, `tests/test_stream_eval_bags.py`, `tests/test_training_free.py` 4개 파일에서 여전히 레거시 모듈을 임포트하고 있어 즉시 삭제 시 테스트 30여 개가 파손됨. 의존성 해소가 선행되어야 함.
- **대안 B: 현상 유지 (Status Quo)**
  - 기각 사유: 60%가 넘는 죽은 코드가 `src/`에 상주함으로써, 신규 브랜치 개발, 코드 리팩토링, 전수 코드 리뷰 시 불필요한 토큰 소비와 유지보수 혼선을 영구적으로 초래함.

---

## 3. 구현 사양 및 기술적 실측 제약 (Lime)

### 상세 의존성 맵 (잔여 참조 분석)
현재 레거시 모듈을 참조하고 있는 파일 및 구체적 원인:

1. **`tests/test_bs_branch.py` (line 46, 372)**:
   - 원인: `_build_model()`에서 `CovarianceMeanLearnablePDDCTMLPModel`을 생성하여 `_Wrapper`에 넣어 `evaluate_trial`의 투표 집계 함수를 테스트.
   - 해소책: `evaluate_trial`은 `TrainingFreeClassifier`를 직접 지원하므로, 모델을 `TrainingFreeClassifier`로 교체하면 5라인 수정으로 의존성 완전 제거 가능.
2. **`tests/test_fixed_head_sj_wiring.py` (line 27, 48)**:
   - 원인: 상동. 고정 헤드 배선 테스트에서 더미 래퍼로 사용.
   - 해소책: `TrainingFreeClassifier`로 교체.
3. **`tests/test_stream_eval_bags.py` (line 26, 27)**:
   - 원인: v35 시절 `BaseModel`의 eager vs stream 배깅 동등성을 `SyntheticManifoldGenerator`로 검증.
   - 해소책: 현재 프로덕션 스트리밍은 `src/models/stream_eval.py`의 `stream_eval_training_free`임. `BaseModel`에 대한 v35 테스트는 레거시 테스트이므로 `tests/archive/`로 이관하거나, `stream_eval_training_free` 대상 테스트로 현대화.
4. **`tests/test_training_free.py` (line 27, 60)**:
   - 원인: v107 도입 당시 `set_transformer_ridge`와의 수치 동등성 1회성 검증용.
   - 해소책: 이미 수치 일치가 확인되었으므로, 해당 테스트의 기준값을 사전 계산된 고정 골든 텐서(`golden_logits.pt`) 비교로 전환하거나 테스트 자체를 아카이브.
5. **`scripts/diagnostics/` (2개 파일)**:
   - `diagnose_branch_contributions.py`: `SyntheticManifoldGenerator`, `ModelInterface` 사용.
   - `diagnose_synthetic_vs_real.py`: `SyntheticEpisodeDataset` 사용.
   - 해소책: 과거 진단 스크립트이므로 `scripts/archive/diagnostics/`로 이동.
6. **`src/utils/utils.py`**:
   - `build_model`, `build_datamodule`, `build_trainer`, `AlwaysSaveLastModelCheckpoint`, `validate_vram_budget` 등 Lightning 학습 유틸리티 ~350라인.
   - 해소책: 평가에 필요한 순수 유틸리티(`eval_autocast`, `add_eval_precision_argument`, `merge_train_config`)만 남기고 Lightning 종속 코드 제거.

### 코드 감축 실측치
- 제거 대상 파일: 14개 파일
- 순수 감축 라인: **8,582 라인** (src: 14,110 라인 -> 5,528 라인, **60.8% 감소**)
- 잔여 핵심 모듈:
  - `src/models/training_free.py` (핵심 분류기)
  - `src/models/branches/` (6개 특징 추출 브랜치: bs, sh, sj, aks, lid, mdx)
  - `src/models/aggregations/` (voting, stacking 등)
  - `src/models/ct.py`, `src/models/dd_adaptive_rank.py`, `src/models/stream_eval.py`
  - `src/models/config.py`
  - `src/utils/metrics.py`, `src/utils/auroc.py`, `src/utils/logging.py`


### ⚠️ 실측 재검증 (Orca 지시 / Lime 실측, 2026-09-12)

**누락된 참조 3건** — §3 목록은 전수가 아닙니다.

| 파일 | 누락 사유 |
|:---|:---|
| **`scripts/test_pathobench.py`** | 공식 평가 진입점. 동적 임포트 체인이라 정적 스캔에 미포착 |
| `scripts/analysis/ru81_probe.py` | `src.models.set_transformer_ridge` **직접** 임포트 |
| `scripts/diagnostics/diagnose_cv_basis.py` | `src.models.set_transformer_ridge` **직접** 임포트 |

**라인 수 대조**

| 항목 | RFC | 실측 | 비고 |
|:---|---:|---:|:---|
| `src/` 전체 | 14,110 | 14,110 | 일치 |
| datasets | 1,984 | **2,004** | +20 |
| modules | 1,457 | **1,529** | +72 |
| `baseline.py` / `set_transformer_ridge.py` | 2,317 / 2,474 | 동일 | 일치 |
| `utils.py` Lightning 몫 | `~350` | **미확인** | 함수가 524라인에 산재, 분리 미측정 |
| **제거 대상 합계** | **8,582 (60.8%)** | **8,324 (58.9%) + utils 미확정** | 60.8%는 **검증되지 않은 수치** |

**브랜치 목록 오기**: §3 "잔여 핵심 모듈"의 `bs, sh, sj, aks, lid, mdx`는 **공식 구성이 아닙니다.**
공식 구성은 `CV, BM, BD, QA, DS, SH, SJ`입니다(`D-042`). `BS`는 게이트 ② 기각(31/50, `p = 0.059`),
`AKS`·`LID`는 RU-85에서 중단 조건으로 종료, `MDX`는 RU-86 게이트 ②에서 반박·종료됐습니다.
**RFC의 유지 목록은 활성 공식 브랜치를 하나도 포함하지 않습니다.**

**`eval_dual16_top3.py`**: 실재하며(439라인), 레거시에 의존하지 않는 독립 경로입니다 — 이 점은 RFC가 맞습니다.

**축소안 비용(상한 추정)**: 필요한 것은 `CovarianceMeanLearnablePDDCTMLPModel`의 7단계 상속 체인
(`set_transformer_ridge.py:67-1309`, `~1,243`라인)과 `baseline.py:29-73`의 `solve_ridge_system`
(`~45`라인)뿐입니다. 두 파일 4,791라인 중 **약 27%**입니다. 메서드 단위 실측은 미수행(클래스 경계 기준 상한).

---

## 4. 최종 판정 및 로드맵 (Orca / User)

### 원 권고 (Platform Agent, 보존)

> **채택(Adopt)** — 3단계 점진적 디커플링 및 제거 로드맵. 근거: (1) 0-파라미터 정체성과 직결, (2) 평가가
> 사전 추출 특징을 직접 로드하므로 Dataset/Lightning 불필요, (3) 테스트 결속 해소 후 제거하면 리스크 없음.

### ✅ 최종 판정 (Orca / Reasoning Agent)

- [ ] 채택(Adopt)  · [x] **보류(Hold)**  · [ ] 기각(Reject)

**목표는 타당하나 현 로드맵은 집행 불가입니다. 정정된 범위로 재제출을 요구합니다.**
`기각`이 아닌 이유는 문제의식(죽은 코드 감축)이 옳고 Phase 1이 무해하기 때문이며,
`채택`이 아닌 이유는 Phase 2를 그대로 집행하면 **공식 평가가 임포트 단계에서 죽기** 때문입니다.

### 결정 근거

1. **원 권고의 근거 (2)와 (3)이 실측으로 반박됐습니다.**
   `/tmp` 사본 삭제 시뮬레이션에서 `import scripts.test_pathobench`가
   **`ModuleNotFoundError: No module named 'src.modules'`** 로 실패합니다. "리스크 없음"은 성립하지 않습니다.
2. **Phase 2를 건너뛰어도 깨집니다.** §3 항목 6의 `utils.py` 유지 목록에 `build_model`이 빠져 있어
   그 편집만으로 `test_pathobench.py:91-95`가 `ImportError`를 냅니다.
3. **사전 확증 기준 3 자체가 반증됐습니다.** "핵심 분류기가 삭제 대상에 런타임 의존성을 갖지 않음"을
   요구했으나, 공식 5개 브랜치 전부가 lineage 모델의 공분산 기저에 의존합니다.
   **게이트 기준이 이미 실패한 상태**이므로 채택할 수 없습니다.
4. **유지 대상 지정이 틀렸습니다.** §3의 `bs, sh, sj, aks, lid, mdx`는 공식 구성이 아닙니다.
   공식은 `CV, BM, BD, QA, DS, SH, SJ`(`D-042`)이며 `BS`는 게이트 ② 기각, `AKS`·`LID`·`MDX`는 종료된 후보입니다.
5. **감축 규모 `60.8%`는 미검증입니다.** 실측 합계는 `8,324`(58.9%)이고 `utils.py` 몫은 미측정입니다.
   더 중요한 것은 이 중 상당량이 **삭제 불가**라는 점이라, 실제 감축 가능량은 아직 **산출된 바 없습니다.**
6. **근본 원인은 문서 부재입니다.** `docs/current_architecture.md`에서 이 의존 사슬을 찾으면
   트리의 `registry.py` 주석 한 줄이 전부입니다. 게다가 `registry.py`의 **동적 임포트**는 정적 스캔에
   원리적으로 안 잡힙니다. **문서화가 선행되지 않으면 같은 오류가 재발합니다.**

### 판정을 뒤집지 **않은** 것 — 방향은 살아 있습니다

레거시 모델은 **학습 가중치를 주입하지 않습니다**(`eval_v121.sh:19` `CKPT=""`,
`test_pathobench.py:526`에서 PCA로 덮어씀). 순수한 **구조적 뼈대**입니다.
따라서 장기적으로 **제거가 아니라 추출·대체**가 올바른 목표입니다. Lime 실측에 따르면 실제로 필요한 것은
`set_transformer_ridge.py`의 상속 체인 `~1,243`라인과 `baseline.py`의 `solve_ridge_system` `~45`라인,
즉 두 파일 `4,791`라인 중 **약 27%**입니다(클래스 경계 기준 **상한 추정**, 메서드 단위 미측정).

### Owl의 반증 조건에 대한 판정

- **반증 조건 2 (동등성 검증 보존 필요성)** — **무효화됨.** `set_transformer_ridge.py`는 애초에
  제거 대상이 될 수 없음이 확정됐으므로 이 조건은 판정 대상이 아닙니다.
- **반증 조건 1 (합성 데이터의 연구적 불가결성)** — **판별 불가, 미해결로 남깁니다.**
  현재 `src/datasets/synthetic`은 `model_interface.py:12`가 `RESPONSE_TASK_NAMES`를 임포트해
  **평가 경로에 이미 묶여 있습니다.** 연구적 필요와 무관하게 기술적으로 결속돼 있어, 이 결속을 끊는 것이
  선결 과제입니다.

### 수정된 로드맵 — 재제출 조건

**Phase 0 (신설·필수, Document + Lime)** — 문서화가 먼저입니다.
  - `docs/current_architecture.md`에 공식 평가의 lineage 의존 사슬을 기록합니다
    (`eval_v121.sh` → `test_pathobench.py` → `build_model` → `ModelInterface` → `registry` 동적 임포트 → 모델 클래스).
  - **`registry.py`의 동적 임포트 때문에 정적 스캔만으로 의존성 조사를 완결할 수 없다**는 사실을 명시합니다.
  - `CKPT=""`이며 투영이 PCA로 덮어써진다는 점(0-파라미터 정체성 보존)을 함께 적습니다.

**Phase 1 (조건부 승인, Lime)** — 원안 유지. 테스트 4건 디커플링은 무해하므로 진행 가능합니다.
  - 단 `test_training_free.py`의 **골든 텐서 전환은 별도 판단 대상**입니다. 동등성 검증을 동적에서
    정적으로 바꾸는 것이므로, 무엇을 잃는지 적고 승인받은 뒤 진행합니다.
  - 누락된 참조 2건(`scripts/analysis/ru81_probe.py`, `scripts/diagnostics/diagnose_cv_basis.py`)도 범위에 포함합니다.

**Phase 2 (전면 재설계 후 재제출)** — 현 안은 폐기합니다.
  - 목표를 **"삭제"에서 "추출"로** 바꿉니다: lineage 뼈대를 `src/models/lineage/`로 추출하고
    Lightning·`src/datasets`·`src/modules` 의존을 끊습니다. 삭제 가능 대상은 **그 나머지**입니다.
  - 재제출 시 **실제 감축 가능 라인 수를 실측치로** 제시합니다(추정 금지).
  - **예산과 중단 조건을 명시합니다** — 현 RFC에는 둘 다 없습니다.

**공통 수용 기준 (구체화)** — "수치적으로 100% 동일"을 다음으로 고정합니다.
  - 공식 기준선 Primary 7 macro와 오염 검사 상수(`SMAD4`·`PBRM1`)가 **소수 4자리 일치**
    (수치 정본은 `docs/PROJECT.md` §3.2·§3.4).
  - 회귀 스위트 전수 통과.
  - **임포트 스모크 테스트 신설**: `import scripts.test_pathobench`가 성공하는지 CI에서 검사합니다.
    이번 결함을 잡았을 유일한 저비용 장치입니다.

### 다음 실행 단계 (티켓 발행)

| # | 대상 | 내용 | 선행 |
|---:|:---|:---|:---|
| 1 | Document | Phase 0 — 의존 사슬·동적 임포트 한계 문서화 | 없음 |
| 2 | Lime | 임포트 스모크 테스트 신설 | 없음 |
| 3 | Lime | Phase 1 — 테스트 6건 디커플링 (골든 텐서 전환은 제외) | 티켓 1 |
| 4 | Platform | Phase 2 재설계본 RFC 재제출 (추출 범위·실측 감축량·예산·중단 조건 포함) | 티켓 1·3 |

**동결하지 않습니다.** 본 RFC는 `Hold` 상태로 열어 두며, Phase 2 재제출 시 새 RFC로 대체(`Superseded`)합니다.

---

[작성자: Orca / Reasoning Agent / claude-opus-5 (effort: high) · 2026-09-12 13:20 KST]
_§1·§3의 정정 블록과 §4 판정은 Orca가 작성했습니다. 그 외 본문은 원 작성자의 기술을 보존했으며,_
_§1·§3의 실측 근거는 Lime / Coding Agent / claude-sonnet-5 (effort: medium)의 2026-09-12 측정입니다._

[작성자: Antigravity / Platform Agent / Gemini 3.8 Flash (effort: high) · 2026-09-12 12:30 KST]
