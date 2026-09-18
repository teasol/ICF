# RU-92 — SEAL 10개 공식 과제 순수 러너(v121 7-Branch) 전수 평가 및 비교 분석

- 유형: `holdout_measurement` · 근거: 연구 책임자(kimds) 지시 (2026-09-14)
- 실행 환경: `nexgem-s1` GPU 8장 (`cuda:0` ~ `cuda:7`) 병렬 워커 풀
- 실행 스크립트: [`scripts/run_seal10_evaluation.py`](../../scripts/run_seal10_evaluation.py) (내부 러너: [`scripts/evaluate_pure.py`](../../scripts/evaluate_pure.py))
- 대상 구성: [`configs/baseline/v121_7branch_active.yaml`](../../configs/baseline/v121_7branch_active.yaml) (7-Branch: CV + BM + BD + QA + DS + SH + SJ, Trimmed Mean)
- 산출물:
  - 예측 파일: `predictions/seal10_pure/pure_v121_7branch_{task}.pt` (10개 과제, 500 folds 전수)
  - 실행 로그: `logs/seal10_v121_7branch/*.log`
  - 수치 요약: [`docs/history/ru92_seal10_v121_7branch_results.json`](../history/ru92_seal10_v121_7branch_results.json)

---

## 1. 평가 개요 및 요약

연구 책임자의 지시에 따라, 현행 공식 7-Branch 베이스라인(`v121_7branch_active`)을 SEAL 10개 공식 과제(50 folds, 총 500회 평가) 전수에 대해 순수 러너(`evaluate_pure.py`)로 실행 완료하였습니다.

- **v121 7-Branch Macro AUROC**: **`0.6869`**
- **v120 6-Branch Macro AUROC**: **`0.6972`**
- **$\Delta(\text{v121} - \text{v120})$**: **`-0.0102` (-1.02%p 하락, 10개 중 7개 과제 악화)**
- **SEAL 논문 공개 지도학습 벤치마크 대비**:
  - **SEAL 지도학습 ABMIL (`0.7266`) 대비**: $\Delta = -0.0397$ (-3.97%p 차이)
  - **SEAL 지도학습 MeanMIL (`0.7125`) 대비**: $\Delta = -0.0256$ (-2.56%p 차이)

---

## 2. 과제별 실측 상세 비교표

| 과제명 (`Task`) | v121 7-Branch (실측) | v120 6-Branch (기존) | $\Delta(\text{v121}-\text{v120})$ | SEAL ABMIL (지도학습) | $\Delta(\text{v121}-\text{ABMIL})$ | SEAL MeanMIL (지도학습) | $\Delta(\text{v121}-\text{MeanMIL})$ | v120 대비 우세 fold |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `bc_therapy/er_status` | 0.6691 ± 0.099 | 0.6825 ± 0.096 | **-0.0134** | 0.7170 | -0.0479 | 0.7120 | -0.0429 | 16/50 |
| `bc_therapy/grade` | 0.7272 ± 0.066 | 0.7414 ± 0.063 | **-0.0142** | 0.7700 | -0.0428 | 0.7510 | -0.0238 | 15/50 |
| `bc_therapy/her2_status` | 0.6581 ± 0.082 | 0.6700 ± 0.075 | **-0.0119** | 0.6630 | -0.0049 | 0.6840 | -0.0259 | 14/50 |
| `cptac_brca/PIK3CA_mutation` | 0.5547 ± 0.140 | 0.5466 ± 0.141 | **+0.0081** | 0.5950 | -0.0403 | 0.5440 | **+0.0107** | 26/50 |
| `cptac_brca/TP53_mutation` | 0.7952 ± 0.081 | 0.7928 ± 0.087 | **+0.0024** | 0.8010 | -0.0058 | 0.7870 | **+0.0082** | 23/50 |
| `cptac_luad/EGFR_mutation` | 0.7387 ± 0.100 | 0.7604 ± 0.092 | **-0.0217** | 0.8300 | -0.0913 | 0.7770 | -0.0383 | 11/50 |
| `cptac_luad/STK11_mutation` | 0.8687 ± 0.092 | 0.8801 ± 0.086 | **-0.0114** | 0.9080 | -0.0393 | 0.8730 | -0.0043 | 15/50 |
| `cptac_luad/TP53_mutation` | 0.6798 ± 0.100 | 0.6911 ± 0.098 | **-0.0113** | 0.7510 | -0.0712 | 0.7350 | -0.0552 | 14/50 |
| `cptac_ccrcc/BAP1_mutation` | 0.6452 ± 0.131 | 0.6871 ± 0.121 | **-0.0418** | 0.6930 | -0.0478 | 0.7200 | -0.0748 | 12/50 |
| `cptac_ccrcc/VHL_mutation` | 0.5325 ± 0.161 | 0.5195 ± 0.156 | **+0.0129** | 0.5380 | -0.0055 | 0.5420 | -0.0095 | 29/50 |
| **Macro Average** | **0.6869** | **0.6972** | **-0.0102** | **0.7266** | **-0.0397** | **0.7125** | **-0.0256** | — |

---

## 3. 핵심 발견 및 시사점

1. **형상 브랜치(SH, SJ)의 SEAL 과제 음성 전이(Negative Transfer)**:
   - 개발 과제(Primary 7)에서는 형상 브랜치 추가 시 Macro AUROC가 `0.6170`에서 `0.6226`으로 `+0.56%p` 유의미하게 개선되어 D-042 공식 승격이 이루어졌음.
   - 그러나 SEAL 10개 독립 과제에서는 형상 브랜치 결합 시 **10개 중 7개 과제에서 성능이 하락**하였으며, 전체 Macro AUROC가 `0.6972`에서 `0.6869`로 **`-1.02%p` 하락**함.
   - 특히 `cptac_ccrcc/BAP1_mutation`(-4.18%p) 및 `cptac_luad/EGFR_mutation`(-2.17%p)에서 두드러진 성능 침하가 관측됨.
2. **6-Branch(v120)의 뛰어난 일반화 복원력 재확인**:
   - 기존 6-Branch(`CV + CT + BM + BD + QA + DS`) 구성은 SEAL 10개 과제에서 **`0.6972`**를 기록하여, 지도학습 MeanMIL(`0.7125`)과 불과 1.5%p, ABMIL(`0.7266`)과 약 2.9%p 차이밖에 나지 않는 강력한 성능을 입증함.
3. **학습 없는(0-parameter) 인컨텍스트 추론의 실증적 경쟁력**:
   - 경사하강법 기반 파인튜닝/학습 없이, 순수 컨텍스트 통계와 닫힌 형태 Ridge 분류기만으로도 지도학습 MIL 성능의 **96%~98%** 수준을 안정적으로 달성함.
   - `cptac_brca/PIK3CA` 및 `cptac_brca/TP53`에서는 지도학습 MeanMIL을 능가함.

---
[작성자: Penguin / Platform Agent / Gemini 3.8 Flash (effort: high) · 2026-09-14 10:45 KST]

---

## 정정 이력

- **2026-09-14 12:15 KST — v120 브랜치 수 표기 정정.** 본 보고서 초판이 `0.6972`를 산출한 구성을
  `v120 5-Branch (CV + BM + BD + QA + DS)`로 적었으나, 이는 오기다. 근거:
  `configs/baseline/v120_active.yaml:2` (`"6-Branch Trimmed Mean Voting Baseline (CV + CT + BM + BD + QA + DS)"`)
  및 [`PROJECT.md` §3.3](../PROJECT.md) (`v120 6-branch (CT 포함)`). **`0.6972`는 CT를 포함한 6-branch 구성의 값**이다.
  측정값 `0.6972`·`0.6869` 및 과제별 수치는 변경되지 않았고, 구성 표기만 정정했다.
  이 정정은 v121 7-branch가 CT를 제외한 구성이라는 점을 바꾸지 않으며, 따라서 §3-1의
  음성 전이 관측(형상 브랜치 추가 시 SEAL 하락)은 **CT 제외·형상 추가라는 두 변경이 겹친 비교**임을
  함께 명시한다 — 하락 원인을 형상 브랜치 단독으로 귀속할 수 없다. **기전 미확인.**

[정정 작성자: Claude Code / Platform Agent / claude-opus-5 (effort: high) · 2026-09-14 12:15 KST]
