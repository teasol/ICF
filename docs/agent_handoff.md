# Agent Handoff Guide & Quick Resume

**Last updated**: `2026-08-21 15:45:00`

---

## 1. 60초 빠른 재개 절차 (Quick Resume)

1. **환경 로드 및 파이썬 확인**:
   ```bash
   . scripts/node_env.sh
   echo "$PYTHON_BIN / NGPU=$NGPU"
   ```
   *(하드코딩된 `python3`나 conda 경로 대신 반드시 `$PYTHON`을 사용)*

2. **회귀 테스트 검증 (92 core tests)**:
   ```bash
   $PYTHON -m unittest discover -s tests -p "test_*.py"
   # 결과: Ran 92 tests in ~13s, OK
   ```

3. **활성 Baseline**:
   - **v114 (0-parameter, Deterministic)**
   - Within-slide PCA (K=256) + CV(off-diagonal 32,640D) + DD(ordered-typicality, κ=1) + CT(1/8 fraction, seeded k-means++ 256 token)
   - SEAL 10-task macro: **0.70509** (활성 러너: `bash scripts/eval_v114.sh <gpu> <tag> [tasks...]`)

---

## 2. 프로젝트 단일 문서 맵 (SSOT Map)

모든 정보는 아래 4개 핵심 문서에 단일 출처(SSOT)로 유지된다:

| 문서 | 역할 및 포함 내용 |
| :--- | :--- |
| [`docs/current_architecture.md`](current_architecture.md) | **현행 아키텍처 완전 명세**: 활성 3개 브랜치(CV, DD, CT) 및 실험 브랜치(BM)의 수식, 작동 원리, 모듈 패키지 구조 |
| [`docs/current_status.md`](current_status.md) | **현재 개발 상태 & 판정 SSOT**: §0 판정 종합 규칙(t 금지, 부호 일치 수, 사용자 판단), 최신 승격 및 실험 내역(§185~§191) |
| [`docs/current_experiments.md`](current_experiments.md) | **실험 큐**: [Plan A] BM 평가 계획, [Plan B] TH 이질성 브랜치, [Plan C] QA 분위수 브랜치, 홀드아웃 7 실측 |
| [`docs/history.md`](history.md) | **과거 기록 아카이브**: 과거 학습 계보(v83~v98), 이전 세션 실험(§2~§184), 설계 결정 이력 |

---

## 3. 핵심 개발 및 판정 원칙

1. **결정론적 Arm 판정 원칙 ([`docs/current_status.md` §0-1](current_status.md))**:
   - $t$-통계량, $p$-value, CI 사용 절대 금지.
   - **부호 일치 수 (Sign Agreement)** 및 **독립 집단(SEAL 10 / 홀드아웃 7) 동시 재현**으로 판정.
2. **사용자 종합 판단 원칙 ([`docs/current_status.md` §0-2](current_status.md))**:
   - 게이트 자동 결정 금지. 고신호 구간 하락 패턴 분석 및 10/17개 전체 task 표 보고 후 사용자가 최종 결정.
3. **닫힌 축 재시도 금지 ([`docs/current_status.md` §0-3](current_status.md))**:
   - 합성 데이터 분포, DD 전반, CT cell 수, CT Kernel-Ridge, CT Top-k 풀링 등 기각된 축 탐색 차단.
4. **문서 작성 스탬프 규칙**:
   - 새 단락 및 긴 내용 작성 직후 반드시 스탬프 첨부:
     `_by <LLM Name> on <server name> at <YYYY-MM-DD HH:MM:SS>_`

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-21 15:45:00_
