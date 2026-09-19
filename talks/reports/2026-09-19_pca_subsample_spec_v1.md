# PCA · subsample 사양 v1 — 회차 16 기록 공백 해소 (2026-09-19)

- **원문**: [`C-20260918-16/report.md`](../council/C-20260918-16/report.md) Phase 1 P1(제안 1\~3), Phase 3 종합.
- **상태**: **미실행 · 미채택.** 회차 16은 이 v1을 제안했으나 정본에 옮겨지지 않았다. 이 문서는 그
  기록 공백을 해소하고, 이후 결정으로 **해소된 단서**와 **남은 막힘**을 구분한다.
- 원문의 판정: 지지(PCA train-context-only·subsample off·사양 고정 절차), 반박(PCA 차원 공식·bit-identity만으로 라벨 판정·single-seed 승격), 판별 불가(다수), 실행 무효(계측 복구 전 verified 비교 등).

## 1. 사양 v1 (원문 그대로)

| 항목 | v1 고정값 |
|:---|:---|
| PCA 적합 범위 | fold의 **train context slide descriptor만**. test split의 context/query descriptor는 fit에 참여하지 않는다 |
| 차원 | `n_components = min(n_train_context − 1, feature_dim)` — rank-preserving, K축 축소 안 함 |
| 화이트닝 | 없음 |
| 정규화 | PCA 단계에서 feature scaling 없음. train-context centering만 |
| 라벨 사용 | PCA 단계에서 context·query 라벨 **금지** |
| 파이프라인 지점 | slide descriptor 생성 직후, branch margin/readout 직전. context·query를 같은 fit으로 변환 |
| subsample | **발동 안 함(임계 무한대)**. slide 수 정의 = fold train context slide 수. query/inference subsample 금지 |
| combined 순서(향후 발동 시) | slide 수 계산 → context subsample → PCA fit(subsample된 train context) → train/test 변환 |
| 사양 변경 통제 | 첫 성능 조회 전 hash 고정(config·code·data/fold manifest·seed list·PCA/subsample params·eval/aggregation code). 성능 조회 후 변경은 v2만, v1 결과는 v2 선택·승격에 사용 금지 |

## 2. 실행 무효 조건 (원문)

- PCA(또는 upstream) fit에 test context/query 포함, PCA 경로에 라벨 사용, provenance 필드 누락,
  일부 fold·과제 점수화, 적합 분산 미포함 single-seed 결과의 승격 사용, preflight metric의 선택 사용,
  과거 산출물 baseline 재사용, OOM/NaN/4 GPU-h 초과.

## 3. 이후 결정으로 해소된 단서

원문은 작성 시점(2026-09-18)의 미해결을 전제했다. 이후 결정이 일부를 닫았다.

| 원문 단서 | 이후 상태 |
|:---|:---|
| "적합 분산 미포함 single-seed 결과는 승격 근거가 아니다" | **해소.** `D-054`(2026-09-18): 현행 branch 구성·집계는 closed-form 결정론이라 `R=1`이 근사가 아니라 정확. 확률적 후보만 seed 사전 고정. **현행 후보에 한해 single-seed 승격 가능** |
| "계측 복구 전 verified 비교는 실행 무효" | **해소.** `RU-100`이 공식 러너에 provenance를 넣고, `D-042` 승격 비교를 verified로 올렸다. `check_artifacts.py`(L1 귀속 검사)도 신설 |
| "fold 적합 시간 미실측" | **해소.** `foldfit_cost`(5-branch 20.0초/fold) → `RU-99` 디바이스 상주화 → `RU-100` 7-branch 2.0초/fold(H100). 700 fold 2-arm이 4 GPU-h 안에 들어온다 |
| "bit-identity만으로 라벨 판정은 반박" | **부분 해소.** 현행 closed-form은 `D-054`상 결정론이라 bit-identity 검증이 성립. 확률적 후보에는 적용하지 않는다 |

## 4. 남은 막힘

- **`feature_dim` 미확정.** 차원 공식 `min(n_train_context−1, feature_dim)`을 실행하려면 `feature_dim`이
  단일 값이어야 하나 문서에 32D·256·1536이 공존하고 downstream 차원 계약이 없다(원문 반박 545).
- **upstream 전처리 fit 범위 미상.** standardization·stain normalization 등 PCA 이전 단계가
  train-context-only인지 문서화된 pipeline inventory가 없다.
- **라벨 무사용 확인 절차 미구현.** static call-graph 검사 + label permutation deterministic control의
  구현 경로가 코드에 없다(오염 검사 L1은 **귀속**만 검증하며 라벨 무사용은 검증하지 않는다).
- **PCA/subsample 계열 seed 분산 미측정**(확률적 변형 시에만 필요).

## 5. 다음에 이 사양을 실행하려면

1. §4의 `feature_dim`·upstream inventory를 먼저 확정한다(코드 열람, GPU 0).
2. v1 hash를 첫 metric 생성 전에 고정한다. 기존 `predictions/`는 git-ignore이므로 manifest로 영속화한다(`RU-100` 전례).
3. `D-047`·`D-057`에 따라 `SEAL 10`은 선택에 쓰지 않는다.

[작성자: opencode / Platform Agent / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 20:10 KST]
