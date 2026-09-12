# Task: Phase A-0 및 A-1 실행: 7-branch YAML 작성 및 순수 러너(evaluate_pure.py) 구현

- **Date**: 2026-09-12 13:00 KST
- **From**: @platform
- **To**: @lime
- **Related Topic / RU**: `N/A`

## 📌 작업 목표 및 지침
talks/rfc/RFC_2026-09-12_pure_runner_migration_and_legacy_removal.md 의 A-0 및 A-1 단계를 구현해주세요. 1) configs/baseline/v121_7branch_active.yaml 작성 (CV,BM,BD,QA,DS,SH,SJ 7개 브랜치 weight 1.0, trimmed_mean 집계) 및 TrainingFreeConfig.from_yaml 로드 검증. 2) scripts/evaluate_pure.py 신설 (레거시 모듈 임포트 0건, pure PyTorch/TrainingFreeClassifier 기반, golden predictions/ 대조 max|Δp| 검증 모드 포함).

## 📦 기대 산출물 및 회신 위치
- 관련 결과(수치, 로그, 변경 파일, 보고서)를 기록 또는 전달한 후, `python scripts/inbox.py done /home/kimds/ICF/talks/inbox/lime/2026-09-12_1300_from_platform_phase_a_0_및_a_1_실행_7_branch_yaml_작성_및_순수.md`를 실행하여 완료 처리합니다.
