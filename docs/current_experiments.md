# Current Experiments & Active Queue (2026-08-22)

**Last updated**: `2026-08-22 21:30:00`

> [!IMPORTANT]
> **활성 baseline = v119** (`docs/current_status.md` §200, 사용자 결정). 학습 파라미터 0, 완전 결정론적(Deterministic).
> **아키텍처**: 4-Branch (CT + BM + BD + QA, $w_{CV}=0.0, w_{DD}=0.0$) + Trimmed Mean Voting
> ⚠️ **판정 프로토콜**: **Primary Benchmark (7 tasks)** 가 주 판정 기준이며, **Hold-out Validation (SEAL 10 tasks)** 는 독립 교차 검증용이다.
> 실행: `bash scripts/eval_v119.sh <gpu> <tag> [tasks...]` (기본: Primary 7 tasks)
> 
> ⚠️ **판정 규칙 종합 SSOT**: [`docs/current_status.md` §0](current_status.md)을 준수할 것. 결정론적 arm에는 t·p·CI를 일절 쓰지 않으며, **Primary 7-task 부호 일치 수 ($\ge 5/7$)** 및 **Hold-out 10-task 동시 재현**으로 판정하고 최종 승격/기각은 **사용자 판단**이다.

---

## 1. 최근 종료된 실험 요약

| 실험 | 가설 및 설정 | 결과 | 판정 |
| :--- | :--- | :--- | :--- |
| **§196 4-Branch Soft Voting (v118)** | 4-Branch Soft Voting ($P = \frac{1}{4}\sum \sigma(M_b)$, $w_{DD}=0.0$) | Macro **0.6205** (+0.0086 vs v116, 6/7 과제 승리) | **v118 공식 승격** 🚀 |
| **§197 v118 SEAL 10 홀드아웃 검증** | 4-GPU로 SEAL 10개 과제(50-fold) 독립 교차 검증 | SEAL 10 Macro **0.7011** (프로젝트 사상 최초 0.70 돌파) | **검증 완료** 🏆 |
| **§198 QA (Quantile Evidence) 실측** | 32 PCA 차원별 4대 분위수 $[Q_{0.05}, Q_{0.10}, Q_{0.90}, Q_{0.95}]$ Ridge | 단독 Progression **0.8068**, KRAS **0.7420**, Grade **0.6930** | **채택 완료** |
| **§199 9대 Voting 전수 비교** | 5개 브랜치 기반 Soft/Linear/Median/Trimmed Mean/Rank/Z-score 비교 | Trimmed Mean 적용 시 Primary 7 Macro **`0.6275`** 달성 | **Trimmed Mean 채택** |
| **§200 v119 공식 승격** | 4-Branch (CT + BM + BD + QA) + Trimmed Mean Voting | Primary 7 **`0.6275`**, SEAL 10 **`0.7026`**, All 17 **`0.6716`** | **v119 공식 승격** 🚀 |

---

## 2. 활성 실험 큐 (Active Experiment Queue)

### [Exp 1] SA (In-Context Salience / Denoising Attention) 브랜치 연구
- **가설**: Context 지원 벡터와의 유사도에 따라 패치 가중치를 부여하여 정상 조직 패치 노이즈를 억제.

---

## 3. 실험 프로토콜 및 산출물 규칙

1. **평가 스크립트 실행 전**:
   - 반드시 `. scripts/node_env.sh` 소싱 확인.
   - 환경변수 및 오버라이드 플래그 명시.
2. **결과 보고**:
   - `docs/current_status.md` §0-2 양식 준수 (가설 설명 $\to$ Primary 7-task 및 Hold-out 10-task 전체별 $\Delta$ 표 $\to$ Macro $\Delta$ 및 부호 일치 수).
   - 신규/수정 단락 작성 직후 스탬프 작성:
     `_by <LLM Name> on <server name> at <YYYY-MM-DD HH:MM:SS>_`

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-22 21:30:00_

