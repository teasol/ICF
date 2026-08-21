# Current Experiments & Active Queue (2026-08-21)

**Last updated**: `2026-08-21`

> [!IMPORTANT]
> **활성 baseline = v114** (`docs/current_status.md` §187, 사용자 결정). 학습 파라미터 0, 완전 결정론적(seed std 0.00000).
> **SEAL 10 macro 0.70509**, ⚠️ **홀드아웃 7·전체 17은 미측정**.
> 실행: `bash scripts/eval_v114.sh <gpu> <tag> [tasks...]`
> 
> ⚠️ **판정 규칙 종합 SSOT**: [`docs/current_status.md` §0](current_status.md)을 준수할 것. 결정론적 arm에는 t·p·CI를 일절 쓰지 않으며, **부호 일치 수(Sign Agreement)** 와 **독립 집단(SEAL 10 / 홀드아웃 7) 동시 재현**으로 판정하고 최종 승격/기각은 **사용자 판단**이다.

---

## 1. 최근 종료된 실험 요약

| 실험 | 가설 및 설정 | 결과 | 판정 |
| :--- | :--- | :--- | :--- |
| **§188 CT Kernel-Ridge** | Abundance $\to$ label 관계의 비선형 곡률 포착 (RBF, Poly 커널) | RBF 0.69500(−0.0101), Poly 0.69475(−0.0103), 둘 다 8/10 task 하락 | **기각 (곡률 부재)** |
| **§189 CT Top-k 풀링** | Mean abundance 대신/추가로 Top-k 점수 풀링 | Top-k 단독 VHL −0.0154, Mean+Topk 0.70067(−0.0044, 8/10 하락) | **기각 (노이즈 가중)** |
| **§190 전체 리팩터링** | 모듈 책임 분리, 레지스트리 구축, 86 core tests (3.1s) 경량화 | 86 tests 100% PASS, 하위 호환 Facade 완벽 보존 | **승격 및 반영** |
| **§191 Plan A BM 구현** | 1차 모멘트(Bag Mean) 32D 사영 Ridge 브랜치 추가 | 6개 전용 단위 테스트 통과, 92 tests 100% PASS | **구현 완료 (평가 대기)** |

---

## 2. 활성 실험 큐 (Active Experiment Queue)

### [Plan A] BM (Projected Bag-Mean) Branch 벤치마크 평가 (즉시 실행 가능)
- **가설**: CV(2차 공분산), DD(1D 분산), CT(의사세포형 조성비)가 모두 배제하고 있던 슬라이드 내 **1차 모멘트(세포 평균 $\bar{x}_i \in \mathbb{R}^{1536}$)**를 Within-slide PCA 기저 상위 32차원으로 사영하여 Class-balanced Ridge로 결합하면, 세포 집단 중심 이동(Global shift) 신호가 보강된다.
- **수식**:
  $$\mu_i = \bar{x}_i B_{:32} \in \mathbb{R}^{32}, \quad M_{BM} = \text{logit}_1 - \text{logit}_0 \quad (\lambda=1.0)$$
  $$\text{Total Margin} = M_{CV} - M_{DD} + M_{CT} + w_{BM} M_{BM}$$
- **실험 계획**:
  1. **가중치 스윕 ($w_{BM} \in \{0.1, 0.2, 0.5, 0.7, 1.0\}$)** on SEAL 10 tasks
  2. **독립 집단 재현**: 홀드아웃 7 tasks 동시 평가
  3. **사영 차원 감도 분석 ($d \in \{16, 32, 64, 128\}$)**

---

### [Plan B] TH (Tumor Heterogeneity / Dispersion Entropy) 브랜치 (다음 제안)
- **가설**: 암종 및 분자 아형에 따라 종양 세포의 **이질성(Heterogeneity / Dispersion)** 정도가 유의미하게 다르다. 슬라이드 내 세포 임베딩의 공분산 스펙트럼 엔트로피 $H_i$ 또는 유클리드 분산(Trace)을 1차원 스칼라로 측정하여 Class-balanced Ordered Evidence로 주입한다.
- **수식**:
  $$v_i = \log \operatorname{Tr}(S_i) \quad \text{또는} \quad H_i = -\sum_{k=1}^K \tilde{\lambda}_k \log \tilde{\lambda}_k$$
  $$M_{TH} = a(v) \cdot o(v) \in [-1, 1] \quad (\text{Bounded Typicality})$$
- **기대 효과**: DD(단일 분산 방향)가 놓치는 전방위적 세포 다형성(Pleomorphism) 신호 포착.

---

### [Plan C] QA (Quantile / Tail Abundance) 브랜치 (제안)
- **가설**: 종양 특이적 신호는 평균 조성비(CT)뿐 아니라, 특정 기준 방향으로 가장 멀리 떨어진 **극단값 세포(Tail 5% or 95th percentile)**에 집중되어 있을 수 있다.
- **수식**:
  $$\tau_{ik} = \operatorname{Quantile}_{95\%} (\{z_{ij} \cdot t_k\}_{j=1}^{N_i})$$
  $$M_{QA} = \text{Ridge}(\tau^{std}, y)$$
- **기대 효과**: 희귀한 고등급 종양 세포 클론의 존재 유무를 1차/2차 통계와 독립적으로 탐지.

---

### [Plan D] v114 홀드아웃 7 및 전체 17 공식 실측
- **목적**: 현재 v114의 공식 수치가 SEAL 10(0.70509)만 측정되어 있으므로, 홀드아웃 7개 과제를 측정하여 17개 전체 벤치마크 점수를 확정.
- **실행**:
  ```bash
  . scripts/node_env.sh
  bash scripts/eval_v114.sh 0 heldout_v114 cptac_lscc/ARID1A_mutation cptac_lscc/KDM6A_mutation cptac_lscc/NF1_mutation cptac_luad/KRAS_mutation cptac_pda/KRAS_mutation cptac_pda/SMAD4_mutation cptac_ucec/TP53_mutation
  ```

---

## 3. 실험 프로토콜 및 산출물 규칙

1. **평가 스크립트 실행 전**:
   - 반드시 `. scripts/node_env.sh` 소싱 확인.
   - 환경변수 및 오버라이드 플래그 명시.
2. **결과 보고**:
   - `docs/current_status.md` §0-2 양식 준수 (가설 설명 $\to$ 10/17개 전체 task별 $\Delta$ 표 $\to$ Macro $\Delta$ 및 부호 일치 수).
   - 신규/수정 단락 작성 직후 스탬프 작성:
     `_by <LLM Name> on <server name> at <YYYY-MM-DD HH:MM:SS>_`

_by Gemini 3.7 Flash (High) on gnode3 at 2026-08-21 15:17:00_
