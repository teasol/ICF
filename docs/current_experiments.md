# Current Experiments & Active Queue (2026-08-21)

**Last updated**: `2026-08-21 19:40:00`

> [!IMPORTANT]
> **활성 baseline = v115** (`docs/current_status.md` §190, 사용자 결정). 학습 파라미터 0, 완전 결정론적(Deterministic).
> ⚠️ **판정 프로토콜**: **Primary Benchmark (7 tasks)** 가 주 판정 기준이며, **Hold-out Validation (SEAL 10 tasks)** 는 독립 교차 검증용이다.
> 실행: `bash scripts/eval_v115.sh <gpu> <tag> [tasks...]` (기본: Primary 7 tasks)
> 
> ⚠️ **판정 규칙 종합 SSOT**: [`docs/current_status.md` §0](current_status.md)을 준수할 것. 결정론적 arm에는 t·p·CI를 일절 쓰지 않으며, **Primary 7-task 부호 일치 수 ($\ge 5/7$)** 및 **Hold-out 10-task 동시 재현**으로 판정하고 최종 승격/기각은 **사용자 판단**이다.

---

## 1. 최근 종료된 실험 요약

| 실험 | 가설 및 설정 | 결과 | 판정 |
| :--- | :--- | :--- | :--- |
| **§188 CT Kernel-Ridge** | Abundance $\to$ label 관계의 비선형 곡률 포착 (RBF, Poly 커널) | RBF 0.69500(−0.0101), Poly 0.69475(−0.0103), 둘 다 8/10 task 하락 | **기각 (곡률 부재)** |
| **§189 CT Top-k 풀링** | Mean abundance 대신/추가로 Top-k 점수 풀링 | Top-k 단독 VHL −0.0154, Mean+Topk 0.70067(−0.0044, 8/10 하락) | **기각 (노이즈 가중)** |
| **§190 전체 리팩터링** | 모듈 책임 분리, 레지스트리 구축, 86 core tests (3.1s) 경량화 | 86 tests 100% PASS, 하위 호환 Facade 완벽 보존 | **승격 및 반영** |
| **§191 Plan A BM 구현** | 1차 모멘트(Bag Mean) 32D 사영 Ridge 브랜치 추가 | 6개 전용 단위 테스트 통과, 92 tests 100% PASS | **구현 완료** |
| **§192 BM 7-Task 실측** | BM Branch ($w_{BM}=1.0$) Primary 7-Task 50-fold 평가 | Macro 0.6094 (+0.0043), 7개 중 5개 과제 승리 ($5/7$) | **v115 공식 승격** |

---

## 2. 활성 실험 큐 (Active Experiment Queue)

### [Exp 1] v115 Hold-out 10-Task (SEAL 10) 독립 교차 검증 (준비 완료)
- **목적**: Primary 7-Task에서 승격된 **v115 (CV + DD + CT + BM)** 아키텍처의 홀드아웃 10개 과제(SEAL 10) 재현성 및 일반화 성능 검증.
- **실행**:
  ```bash
  . scripts/node_env.sh
  bash scripts/eval_v115.sh 0 v115_holdout10 \
    bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status \
    cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation \
    cptac_luad/EGFR_mutation cptac_luad/STK11_mutation cptac_luad/TP53_mutation \
    cptac_ccrcc/BAP1_mutation cptac_ccrcc/VHL_mutation
  ```

---

### [Exp 2] [Plan B] TH (Tumor Heterogeneity / Dispersion Entropy) 브랜치 연구
- **가설**: 암종 및 분자 아형에 따라 종양 세포의 **이질성(Heterogeneity / Dispersion)** 정도가 유의미하게 다르다. 슬라이드 내 세포 임베딩의 공분산 스펙트럼 엔트로피 $H_i$ 또는 유클리드 분산(Trace)을 1차원 스칼라로 측정하여 Class-balanced Ordered Evidence로 주입한다.
- **수식**:
  $$v_i = \log \operatorname{Tr}(S_i) \quad \text{또는} \quad H_i = -\sum_{k=1}^K \tilde{\lambda}_k \log \tilde{\lambda}_k$$
  $$M_{TH} = a(v) \cdot o(v) \in [-1, 1] \quad (\text{Bounded Typicality})$$
- **기대 효과**: DD(단일 분산 방향)가 놓치는 전방위적 세포 다형성(Pleomorphism) 신호 포착.

---

### [Exp 3] [Plan C] QA (Quantile / Tail Abundance) 브랜치 연구
- **가설**: 종양 특이적 신호는 평균 조성비(CT)뿐 아니라, 특정 기준 방향으로 가장 멀리 떨어진 **극단값 세포(Tail 5% or 95th percentile)**에 집중되어 있을 수 있다.
- **기대 효과**: 희귀한 고등급 종양 세포 클론의 존재 유무를 1차/2차 통계와 독립적으로 탐지.

---

## 3. 실험 프로토콜 및 산출물 규칙

1. **평가 스크립트 실행 전**:
   - 반드시 `. scripts/node_env.sh` 소싱 확인.
   - 환경변수 및 오버라이드 플래그 명시.
2. **결과 보고**:
   - `docs/current_status.md` §0-2 양식 준수 (가설 설명 $\to$ Primary 7-task 및 Hold-out 10-task 전체별 $\Delta$ 표 $\to$ Macro $\Delta$ 및 부호 일치 수).
   - 신규/수정 단락 작성 직후 스탬프 작성:
     `_by <LLM Name> on <server name> at <YYYY-MM-DD HH:MM:SS>_`

_by Antigravity on teasol at 2026-08-21 19:40:00_
