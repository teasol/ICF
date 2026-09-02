# Agent Handoff Guide & Quick Resume

**Last updated**: `2026-08-23 09:30:00`

---

## 1. 60초 빠른 재개 절차 (Quick Resume)

1. **환경 로드 및 파이썬 확인**:
   ```bash
   . scripts/node_env.sh
   echo "$PYTHON / NGPU=$NGPU"
   ```
   *(하드코딩된 `python3`나 conda 경로 대신 반드시 `$PYTHON`을 사용)*

   프로젝트 환경은 **uv venv (`ICF/.venv`, Python 3.12)** 다 — conda `BagPFN` 환경은
   2026-09-02에 소실됐고 되살리지 않는다(§206). `node_env.sh`가 `.venv`를 자동으로
   찾으므로 activate 단계는 없다. venv가 없으면:
   ```bash
   uv venv --python 3.12 .venv && uv pip install -r requirements.txt
   ```

2. **회귀 테스트 검증**:
   ```bash
   $PYTHON -m unittest discover -s tests -p "test_*.py"   # 16 모듈 / 119 테스트
   ```
   ⚠️ **CPU 전용이고 매우 느리다**: 테스트는 CPU 텐서로 에피소드를 만들므로 6-브랜치
   파이프라인이 전부 CPU BLAS로 돈다. 계측치(§206) — **`predict_proba` 1회 = 97.3초**,
   `test_soft_voting`(5개) = 4분 이상. **스위트 전체는 수십 분**이고, 노드가 다른
   사용자로 포화(load 160/72코어)면 더 늘어난다. 시작 전에 `uptime`으로 부하를
   확인하고, 급하면 모듈 단위로 돌린다:
   ```bash
   $PYTHON -m unittest discover -s tests -p "test_soft_voting.py"
   ```

3. **활성 Baseline**:
   - **v120 (0-parameter, Deterministic, 6-Branch Trimmed Mean Voting)**
   - Within-slide PCA (K=256) + CV(off-diagonal 32,640D, w=1.0) + CT(1/8 fraction, seeded k-means++ 256 token, w=1.0) + BM(Bag-mean leading 32D PCA ridge, w=1.0) + BD(Bag-Dispersion Spectral Entropy ordered-typicality, K=256, w=1.0) + QA(Quantiles & Extremum Evidence Ridge, 128D, w=1.0) + DS(Salience Denoised Bag-Mean Ridge, 32D, w=1.0) + **Trimmed Mean (최저 1개, 최고 1개 절사 후 중앙 4개 평균)**
   - **Primary 7-task macro**: **`0.6265`** (v119 0.6247 대비 +0.0018, v118 0.6205 대비 +0.0060)
   - **SEAL 10-task macro**: **`0.6972`** (Hold-out 검증)
   - **All 17-task total macro**: **`0.6681`** (역대 전 계보 통합 최고치)
   - 활성 러너: `bash scripts/eval_v120.sh <gpu> <tag> [tasks...]` (기본: Primary 7 tasks)
   - SEAL 10-Task 멀티 GPU 러너: `bash scripts/run_v120_seal_multi_gpu.sh <tag>`

---

## 2. 프로젝트 단일 문서 맵 (SSOT Map)

모든 정보는 아래 4개 핵심 문서에 단일 출처(SSOT)로 유지된다:

| 문서 | 역할 및 포함 내용 |
| :--- | :--- |
| [`docs/current_architecture.md`](current_architecture.md) | **현행 아키텍처 완전 명세**: 활성 6개 브랜치(CV, CT, BM, BD, QA, DS) 및 Trimmed Mean Voting의 수식, 작동 원리, 모듈 패키지 구조 |
| [`docs/current_status.md`](current_status.md) | **현재 개발 상태 & 판정 SSOT**: §0 판정 종합 규칙, 최신 승격 및 실험 내역(§198: v120 승격, §199~§203: 후속 실험 사후분석, §204: 단독 브랜치 전수 실측 및 CT 단독 심층 비교) |
| [`docs/current_experiments.md`](current_experiments.md) | **실험 큐**: [완료] v120 승격, [완료] DE/SW 개발, [완료] 단독 브랜치 실측 및 CT 비교 분석, [Next] CT 단독 고도화 연구 |
| [`docs/history.md`](history.md) | **과거 기록 아카이브**: 과거 학습 계보(v83~v98), 이전 세션 실험(§2~§184), 설계 결정 이력 |

---

## 3. 핵심 개발 및 판정 원칙

1. **결정론적 Arm 판정 원칙 ([`docs/current_status.md` §0-1](current_status.md))**:
   - $t$-통계량, $p$-value, CI 사용 절대 금지.
   - **Primary 7-task 부호 일치 수 ($\ge 5/7$)** 및 **Hold-out 10-task(SEAL 10) 동시 재현**으로 판정.
2. **사용자 종합 판단 원칙 ([`docs/current_status.md` §0-2](current_status.md))**:
   - 게이트 자동 결정 금지. 고신호 구간 하락 패턴 분석 및 7개/10개 전체 task 표 보고 후 사용자가 최종 결정.
3. **닫힌 축 재시도 금지 ([`docs/current_status.md` §0-3](current_status.md))**:
   - DD 전반, CT cell 수 단순 증대, Non-Linear KRR Readout(§199), Direct Likelihood Matching(§200), 비대칭/편차 절사 집계(§201), In-Context 1536D Fisher 감독형 부분공간(§202) 등 기각된 축 탐색 차단.
4. **문서 작성 스탬프 규칙**:
   - 새 단락 및 긴 내용 작성 직후 반드시 스탬프 첨부:
     `_by <LLM Name> on <server name> at <YYYY-MM-DD HH:MM:SS>_`

_by Antigravity on gnode3 at 2026-08-24 18:16:00_


