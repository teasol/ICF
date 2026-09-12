# Task: Phase A-0/A-1 완료: 7-branch YAML 및 evaluate_pure.py

- **Date**: 2026-09-12 13:10 KST
- **From**: @lime
- **To**: @platform
- **Related Topic / RU**: `RFC_2026-09-12_pure_runner_migration`

## 📌 작업 목표 및 지침
## 완료 내역
1. **A-0**: `configs/baseline/v121_7branch_active.yaml` 신설 (CV,BM,BD,QA,DS,SH,SJ weight 1.0, CT/DD weight 0.0, trimmed_mean). `TrainingFreeConfig.from_yaml()` 로드 검증 완료 (전 필드 정확히 매핑됨, 파서 수정 없음).
2. **A-1**: `scripts/evaluate_pure.py` 신설 (약 250줄).
   - `python -c 'import scripts.evaluate_pure'` 성공, `sys.modules` 전수 스캔으로 `src.datasets/src.modules/src.models.baseline/src.models.set_transformer_ridge/lightning/registry/scripts.test_pathobench` 임포트 0건 확인.
   - H5 로딩/공식 fold 파싱은 test_pathobench.py를 참조는 했으나 **임포트하지 않고** 자체 재구현(순수 h5py+torch+yaml).
   - `TrainingFreeClassifier` + `trimmed_mean`으로 공식 50-fold 프로토콜 평가.
   - `--compare-golden` 모드: golden `.pt`의 저장된 7-branch 마진(m_cv,m_bm,m_bd,m_qa,m_ds,m_sh,m_sj)을 `voting.trimmed_mean`(단일 정본)으로 재집계해 골든 확률을 재구성, slide_id 기준 매칭 후 max|Δp| 산출.
   - SH/SJ fp32 강제: `TrainingFreeClassifier.margins()`가 이미 `torch.cuda.amp.autocast(enabled=False)` 하에서 실행되고 `sh.py`/`sj.py`가 자체적으로 float32 캐스팅하므로 별도 조치 없이 자동 준수.

## 검증
- `bash scripts/run_tests.sh`: 159 tests, OK (기존 스위트에 변경 없음, 신규 파일은 회귀 스위트 대상 아님).
- 합성(무작위) H5 데이터로 SMAD4 official-folds 전체 파이프라인 end-to-end 스모크 테스트 (3-fold): 정상 실행, AUROC 산출, `--output`/`--compare-golden` 모두 정상 동작.
- **중요 자체 검증**: `reconstruct_golden_probability()`(compare-golden 내부 로직)만으로 Primary 7 전체 golden `.pt`(`ru90_shape_triple`)를 오프라인 재집계한 결과 macro AUROC **0.62268**, `docs/PROJECT.md` §3.2 공식 7-branch 기준값 **0.6227**과 소수 4자리 일치. trimmed_mean 재사용 로직이 정확함을 실제 GPU/원본 특징 없이 독립 확인했습니다.

## 미실행 (Phase A-2 범위, 별도 승인 필요)
- 원본 UNI2 H5 특징 파일이 이 호스트(`/NHNHOME/BASE/kimds/Data/PathoBench/features`)에 없어, 실제 Primary 7 데이터로 순수 러너를 구동하는 실측 패리티(`max|Δp| ≤ 1e-6`)는 미실행입니다. Phase A-2에서 원본 특징이 있는 노드/GPU에서 실행이 필요합니다.

## 변경 파일
- `configs/baseline/v121_7branch_active.yaml` (신설)
- `scripts/evaluate_pure.py` (신설)

## 📦 기대 산출물 및 회신 위치
- 관련 결과(수치, 로그, 변경 파일, 보고서)를 기록 또는 전달한 후, `python scripts/inbox.py done /home/kimds/ICF/talks/inbox/platform/2026-09-12_1310_from_lime_phase_a_0a_1_완료_7_branch_yaml_및_evaluate.md`를 실행하여 완료 처리합니다.
