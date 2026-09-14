# RU-96 · Token-MIL Primary 7 실측 및 기각 (2026-09-14)

> 상태: **기각(사전 등록 기준 미달)** · 소속 스프린트는 `D-044`로 중단 판정
> 관련: [RU-93](RU-93_icmil_reproduction.md) · [RU-95](RU-95_bagsize_sweep_primary7.md) · `D-044`

## 1. 배경

2026-09-14 ICLR 2027 스프린트(계획 문서 개정 9, R33~R35 — 산출물은 D-044 발동으로 삭제됨)에서
사용자 확정을 거쳐 구현한 새 모델. S1/S2/S3 후보가 전량 폐기된 뒤, CT 브랜치의 결정론적 token
방식 요약을 확장하는 방향으로 승인되어 구현됐다.

**사양 (고정 규칙, 하이퍼 서치 없음)**:
- 컨텍스트 전용 PCA-32 표준화 → k-means++(seed 0) + Lloyd(8 iter) 토큰 코드북 **T=32**
- bag 요약 φ = **[token abundance (32), token-conditional content (32×8)] = 288차원 고정**
  - abundance: 토큰별 배정 질량 (softmax(−dist/0.5))
  - content: 토큰별 배정 instance의 상위 8 PCA 성분 질량 가중 평균 (빈 토큰은 0)
- head: 컨텍스트 표준화 + class-balanced ridge λ=1, margin = logit1−logit0
- 학습 파라미터 0. fold당 약 25\~40초 (RTX A5000 1장)

## 2. Primary 7 실측 (50-fold, Δ는 v121 7-branch 동일 fold paired)

| 과제 | v121 7-branch | Token-MIL | Δ |
|:---|---:|---:|---:|
| `cptac_lscc/ARID1A_mutation` | 0.5795 | 0.4177 | −0.1618 |
| `cptac_lscc/Histologic_Grade` | 0.6665 | 0.6372 | −0.0293 |
| `cptac_lscc/KEAP1_mutation` | 0.6248 | 0.5935 | −0.0313 |
| `cptac_luad/KRAS_mutation` | 0.7031 | 0.7364 | **+0.0333** |
| `cptac_pda/SMAD4_mutation` | 0.4661 | 0.3874 | −0.0787 |
| `ucla_lung/progression_regression` | 0.7772 | 0.7431 | −0.0341 |
| `cptac_ccrcc/PBRM1_mutation` | 0.5408 | 0.5125 | −0.0283 |
| **MACRO** | **0.6226** | **0.5754** | **−0.0472 (sign 1/7)** |

## 3. 판정

사전 등록 채택 기준(macro Δ ≥ −0.5pp)에서 크게 미달 → **기각**.
- 사전 등록 대안(T=16, φ 144차원)의 발동 조건은 "과소결정 병리(반예측 등)"였으나,
  0.5 미만은 `ARID1A` 한 과제뿐이고 거기서는 ICMIL(0.4659)·ICF 계열(S1 0.4073)도 같이 무너진다 —
  모델 고유 병리가 아니라 과제 특성(기전 미확인)으로 보이므로 **T=16 재측정은 실행하지 않았다.**
- 유일한 승리 과제 `KRAS`(+3.33%p)를 제외한 6과제에서 소폭\~중폭 열세.
- token-conditional content 요약이 abundance 대비 정보를 추가했는지에 대한 판별은 이 실측만으로
  불가 (abundance 단독 arm을 함께 재지 않았다). `기전 미확인`.

## 4. 산출물 처리 (D-044)

- 구현 코드(`src/models/token_mil.py`, `scripts/eval_token_mil.py`)와 드라이버·로그는
  스프린트 산출물 정리에 따라 **삭제**했다. 본 보고서 §1 사양 기술이 재구현에 충분한 수준으로
  남는다. 원시 per-fold 데이터는 `docs/history/ru96/*.jsonl`에 보존.
- 재사용된 기반은 프로젝트 live 코드로 남는다: `src/models/branches/ct.py`(셀프컨테인 통합본),
  `scripts/evaluate_pure.py::fit_pca`.

[작성자: GitHub Copilot / GLM-5.3-Flash on nexgem-s1 at 2026-09-14 17:05:00]
