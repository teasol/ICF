# Agent Handoff Guide & Quick Resume

**Last updated**: `2026-08-22 18:18:00`

---

## 1. 60초 빠른 재개 절차 (Quick Resume)

1. **환경 로드 및 파이썬 확인**:
   ```bash
   . scripts/node_env.sh
   echo "$PYTHON_BIN / NGPU=$NGPU"
   ```
   *(하드코딩된 `python3`나 conda 경로 대신 반드시 `$PYTHON`을 사용)*

2. **회귀 테스트 검증 (107 core tests)**:
   ```bash
   $PYTHON -m unittest discover -s tests -p "test_*.py"
   # 결과: Ran 107 tests, OK
   ```

3. **활성 Baseline**:
   - **v118 (0-parameter, Deterministic, 4-Branch Soft Voting)**
   - Within-slide PCA (K=256) + CV(off-diagonal 32,640D, w=1.0) + CT(1/8 fraction, seeded k-means++ 256 token, w=1.0) + BM(Bag-mean leading 32D PCA ridge, w=1.0) + BD(Bag-Dispersion Spectral Entropy ordered-typicality, K=256, w=1.0) + **DD 제거 ($w_{DD}=0.0$)** + **Soft Voting (Sigmoid 확률 평균)**
   - Primary 7-task macro: **0.6205** (v116 0.6119 대비 **+0.0086**, 6/7 과제 승리, v114 0.6051 대비 **+0.0154**)
   - v117 (No-DD Linear Sum): Macro **0.6191** (+0.0072 vs v116, 5/7 과제 승리)
   - 활성 러너: `bash scripts/eval_v118.sh <gpu> <tag> [tasks...]` (기본: Primary 7 tasks)
   - v117 러너: `bash scripts/eval_v117.sh <gpu> <tag> [tasks...]`

---

## 2. 프로젝트 단일 문서 맵 (SSOT Map)

모든 정보는 아래 4개 핵심 문서에 단일 출처(SSOT)로 유지된다:

| 문서 | 역할 및 포함 내용 |
| :--- | :--- |
| [`docs/current_architecture.md`](current_architecture.md) | **현행 아키텍처 완전 명세**: 활성 4개 브랜치(CV, CT, BM, BD) 및 Soft Voting의 수식, 작동 원리, 모듈 패키지 구조 |
| [`docs/current_status.md`](current_status.md) | **현재 개발 상태 & 판정 SSOT**: §0 판정 종합 규칙(Primary 7-task / Hold-out 10-task, t 금지, 부호 일치 수, 사용자 판단), 최신 승격 및 실험 내역(§185~§196: v117 No-DD, v118 Soft Voting 승격) |
| [`docs/current_experiments.md`](current_experiments.md) | **실험 큐**: [완료] 5-Branch 로짓 저장 및 보팅 전수 분석, v117/v118 승격, [Exp 1] v118 Hold-out 10-task 재현성 검증, [Plan C] QA 분위수 브랜치 |
| [`docs/history.md`](history.md) | **과거 기록 아카이브**: 과거 학습 계보(v83~v98), 이전 세션 실험(§2~§184), 설계 결정 이력 |

---

## 3. 핵심 개발 및 판정 원칙

1. **결정론적 Arm 판정 원칙 ([`docs/current_status.md` §0-1](current_status.md))**:
   - $t$-통계량, $p$-value, CI 사용 절대 금지.
   - **Primary 7-task 부호 일치 수 ($\ge 5/7$)** 및 **Hold-out 10-task(SEAL 10) 동시 재현**으로 판정.
2. **사용자 종합 판단 원칙 ([`docs/current_status.md` §0-2](current_status.md))**:
   - 게이트 자동 결정 금지. 고신호 구간 하락 패턴 분석 및 7개/10개 전체 task 표 보고 후 사용자가 최종 결정.
3. **닫힌 축 재시도 금지 ([`docs/current_status.md` §0-3](current_status.md))**:
   - 합성 데이터 분포, DD 전반(v117에서 공식 제거 확인), CT cell 수, CT Kernel-Ridge, CT Top-k 풀링 등 기각된 축 탐색 차단.
4. **문서 작성 스탬프 규칙**:
   - 새 단락 및 긴 내용 작성 직후 반드시 스탬프 첨부:
     `_by <LLM Name> on <server name> at <YYYY-MM-DD HH:MM:SS>_`

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-22 18:18:00_
