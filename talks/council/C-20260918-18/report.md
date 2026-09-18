# ICF Council — C-20260918-18

- **연구 질문**: 최종 목표는 Primary 7에서 얻은 근거만으로 모델을 개선해 SEAL 10에서 ABMIL을 이기는 것이다. 이 회차는 아직 시도되지 않은 축을 찾는다.

현재 구조(확인된 사실): slide의 patch feature에서 7개의 branch가 각각 closed-form ridge로 증거를 만들고, trimmed mean voting으로 집계한다. branch는 cv(공분산 비대각), bm(PCA-32 bag mean), bd(스펙트럼 엔트로피 ordered typicality), qa(분위수/극값), ds(salience denoised bag mean), sh, sj다. 학습되는 파라미터가 없다.

실측된 제약:
- branch 간 중복이 크다. 유효 랭크가 7 중 3.52다.
- 같은 구성을 seed만 바꿔 재적합하면 표준편차 0.0021~0.0083이 나온다. 승격 임계는 +0.0030이다. 즉 단일 적합의 차이는 대부분 뽑기다.
- 여섯 개의 실행 러너가 전부 같은 하나의 설정으로 수렴해 있었다. 따라서 과거의 branch 조합 비교 결과는 구성 차이가 없는 실행들의 비교였고, 근거로 쓸 수 없다.
- 예산은 GPU 1장, 합계 4 GPU-hour. fold 적합 1회의 소요 시간은 측정된 적이 없다.

규칙: SEAL 10 결과를 보고 모델을 고르는 것은 금지다. Primary 7 결과만 개선에 쓸 수 있다.

질문:
(1) 유효 랭크 3.52는 branch를 더 넣는 방향이 막혀 있다는 뜻인가, 아니면 서로 다른 것을 재는 branch를 아직 못 만들었다는 뜻인가? 두 해석을 가르는 관측은 무엇인가.
(2) 집계(trimmed mean voting)를 바꾸는 축은 아직 거의 탐색되지 않았다. 학습 파라미터를 도입하지 않으면서 집계를 바꾸는 방법에는 무엇이 있는가?
(3) 적합 분산이 승격 임계보다 큰 상황에서, 분산 자체를 줄이는 것이 성능을 올리는 것보다 나은 투자인가? 분산을 줄이는 수단을 제시하고, 그것이 성능 개선과 어떻게 다른지 밝혀라.
(4) 지금까지 아무도 건드리지 않은 축이 있다면 무엇인가. 건드리지 않은 이유가 타당한 것인지, 아니면 그냥 생각이 안 난 것인지 구분하라.
(5) 위 후보들 중 4 GPU-hour 안에서 판별 가능한 것은 무엇인가.

문서에 없는 수치를 지어내지 마라. 특히 fold 적합 시간과 branch별 기여도는 측정된 적이 없다.
- **RU 유형**: 탐색
- **최대 권한 등급**: T1
- **소집자**: Claude Code (claude-opus-5)
- **좌석**: P1(P, temp 0.7, http://127.0.0.1:8000/v1/chat/completions), P2(P, temp 0.8, http://127.0.0.1:8001/v1/chat/completions), T1(T, temp 0.4, http://127.0.0.1:8003/v1/chat/completions), M1(M, temp 0.3, http://127.0.0.1:8000/v1/chat/completions), R1(R, temp 0.5, http://127.0.0.1:8003/v1/chat/completions), A1(A, temp 0.2, http://127.0.0.1:8001/v1/chat/completions), S1(S, temp 0.2, http://127.0.0.1:8001/v1/chat/completions)
- **호출**: 13/18 · **소요**: 857.9초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 편성 근거 (결과 전 기록)

탐색 질문이므로 제안(P) 둘을 온도 0.7·0.8로 두어 서로 다른 방향을 뽑는다. 이론(T)은 질문 1의 유효 랭크 해석을, 실측(M)은 질문 3·5의 분산과 비용을 맡는다. 반증(R)은 '새 축이 있다'는 낙관과 '다 막혔다'는 비관을 둘 다 공격한다. 감사(A)는 제안이 과거의 무효한 비교 결과에 기대고 있지 않은지 본다. 앞선 회차 보고서는 입력에 넣지 않았다(회차 간 앵커링 금지).

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: Provenance-gated orthogonality probe
- **바꾸는 것**: 성능 비교를 먼저 하지 않고, (1) 실행 산출물에 branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed·데이터 버전을 기록하는 provenance writer를 복구하고, (2) 러너가 폴백으로 같은 7-branch로 수렴하면 실행을 `실행 무효`로 표기하는 통제를 넣고, (3) 공식 7-branch 대비 후보 branch 1개(예: `CA-09` 경계 밖 `FC` — 여러 초점 강도의 slide mean 차분)에 대해 `branch_screen.py`의 라벨 무관 게이트 ①(max |r|, eff.rank)만 실행한다. 후보 정의와 강도 집합은 Primary 7 성능 조회 전에 RU 카드에 등록한다.
- **바꾸는 전제**: "eff.rank 3.52/7은 branch 추가 방향이 이미 막혔다는 증거다"와 "과거 branch 조합 비교 결과가 서로 다른 구성의 비교였다"는 전제를 뒤집는다.
- **기대하는 차이**: AUROC 개선의 정량 예측은 불가. 판별 관측은 8-branch에서 max |r| ≤ 0.6이고 eff.rank ≥ 4.02이면 "서로 다른 것을 재는 branch를 아직 못 만들었다" 쪽에 가중되고, max |r| > 0.6 또는 eff.rank가 하락하면 "재는 축이 중복" 쪽에 가중된다. 성능을 조회하지 않으므로 결과는 `지지 / 반박 / 판별 불가` 중 하나로만 기록하며, 모든 Primary 7 판정은 hold-out 미검증으로 둔다. fold 적합 시간은 문서에 없으므로 4 GPU-hour 내 판별 가능 여부는 1회 fit 시간 측정 후에만 결정된다.
- **기전**: 유효 랭크는 branch margin 공분산의 실측 랭크이므로, 기존 7개와 상관이 낮은 후보가 추가되면 랭크가 오를 수 있다. provenance가 없으면 후보와 기준이 같은 7-branch로 수렴해 관측 자체가 무효가 된다. 후보가 실제로 다른 물리량을 재는지의 기전은 후보 정의에 달려 있고, 현재 문서에는 `FC`의 성능·상관 실측이 없으므로 기전 미확인.
- **실패 조건**: (a) 러너 폴백 또는 provenance 누락으로 arm을 구성에 귀속시킬 수 없으면 `실행 무효`. (b) 후보를 Primary 7 성능을 보고 고르면 §214식 선택 편향. (c) 후보가 `CA-09`·`CA-10`·`CA-04` 경계 안에 들어가면 성능과 무관하게 기각. (d) 게이트 ①에 필요한 350-fold margin 기록이 4 GPU-hour를 초과하면 이 제안은 예산 내 판별 불가.

## 제안 2: PSW predictive-variance trimmed mean
- **바꾸는 것**: 기존 집계(trimmed mean of branch margins)를, 각 branch의 closed-form ridge 예측분산으로 표준화한 margin의 trimmed mean으로 변경한다. 학습 파라미터를 도입하지 않는다. fold·과제별로 branch b의 margin `m_b`와 fold context에서 계산되는 예측분산 `v_b`를 구하고, `z_b = m_b / sqrt(v_b)`로 변환한 뒤 기존 최저·최고 1개 절사 평균을 적용한다. `v_b`가 비유한 또는 0인 처리 규칙, 변환 식, seed 정책, 후보 등록 시점은 Primary 7 성능 확인 전에 고정하고, SEAL 10으로 선택하지 않는다.
- **바꾸는 전제**: 7개 branch margin이 같은 scale과 같은 신뢰도로 해석될 수 있고, trimmed mean은 branch별 불확실성을 무시해도 된다는 전제를 뒤집는다.
- **기대하는 차이**: 정량 예측 불가. Primary 7 대응 paired Δ_macro, 과제 군집 SE, 적합 분산, sign agreement, 악화 과제를 보고한다. `v_b`가 실제 fold 오차를 예측한다면 저분산 branch가 더 큰 영향을 주어 macro가 오를 수 있으나, 문서에 그 근거가 없다. 모든 Primary 7 판정은 hold-out 미검증으로 둔다.
- **기전**: closed-form ridge 해의 조건부 불확실성이 context 규모·패치 분포에 따라 달라지고, margin 크기와 오차가 비례하지 않으면 variance 표준화가 branch 간 commensurability를 개선할 수 있다. 이는 `CA-08` 경계 밖의 새 정보(예측분산, PSW)에 해당한다. 다만 `v_b`가 오차를 예측한다는 실측이 없으므로 기전 미확인.
- **실패 조건**: (a) `v_b`가 context 크기, 조건부 특이성, 구현 노이즈 등 실제 오차와 무관한 요인을 반영하면 가중이 노이즈가 된다. (b) 고분산 branch가 옳은 극값/이질성 정보를 담고 있으면 표준화가 신호를 죽인다. (c) Primary 7에서 `v_b` 변환을 보고 나서 변형을 고르면 `CA-08`의 선택 편향으로 `실행 무효` 또는 `반박`. (d) 현재 코드에 `v_b` 산출 경로가 없으면 구현 전까지 성능 비교 불가.

## 제안 3: Seed-averaged paired Δ estimand
- **바꾸는 것**: 승격 비교의 추정 대상을 단일 적합의 macro에서, 동일 구성을 사전 등록된 여러 seed/refit으로 재적합한 뒤 fold별로 짝지은 Δ의 평균으로 바꾼다. 기준과 후보에 같은 seed 집합을 적용하고, seed·fold별 AUROC를 기록한 뒤 fold별 seed 평균 AUROC를 만들어 350-fold 대응 Δ를 계산한다. best-seed 선택을 금지하고, seed 평균 Δ, fold SE, 적합 분산, seed별 분포, 악화 과제를 함께 보고한다. seed 수와 seed 집합은 성능 확인 전에 고정한다. fold 적합 시간은 문서에 없으므로, 4 GPU-hour 예산을 초과하면 seed 평균이 아니라 단일 적합 분산 보고로만 `판별 불가` 또는 `실행 무효`를 기록한다.
- **바꾸는 전제**: 단일 적합의 차이를 모델 차이로 해석하고, 적합 분산은 무시하거나 후보 기각 사유로만 쓰는 전제를 뒤집는다. 단일 적합은 같은 분포의 한 draw다.
- **기대하는 차이**: 성능 자체를 올리는 제안은 아니다. 기대하는 차이는 판정 가능 상태: seed std 0.0021~0.0083로, 상한이 승격 임계 +0.0030을 초과하므로 단일 적합으로는 `판별 불가`가 잦을 수 있다. seed 효과가 독립이라면 평균의 SE가 감소하지만 감소 폭은 문서에 없고, `RU-97`에서 fold×seed 잔차가 seed 주효과보다 컸으므로 현재 7-branch에서 평균화가 실제로 줄어드는지 기전 미확인. 모든 Primary 7 판정은 hold-out 미검증으로 둔다.
- **기전**: 재적합의 stochastic path를 평균하면 seed 주효과 성분이 축소되고, 후보와 기준의 대응 Δ 평균이 더 안정적으로 추정될 수 있다. 학습 파라미터 없이 closed-form ridge를 여러 번 적합하는 방식이다.
- **실패 조건**: (a) fold 적합 시간이 4 GPU-hour를 초과하면 seed 평균 실행 불가. (b) 적합 분산의 대부분이 fold×seed 잔차 또는 fold 데이터 민감성이라면 seed 평균으로 줄지 않는다. (c) seed 수나 seed 집합을 Primary 7 결과를 보고 정하면 선택 편향. (d) best seed를 승격 근거로 쓰면 `실행 무효`.

### P2 (제안)

## 제안 1: 유효 랭크 3.52를 family span 포화 진단으로 재해석
- **바꾸는 것**: `eff.rank=3.52/7`을 “브랜치 추가 방향이 닫혔다”는 결론의 근거로 쓰지 않고, Primary 7 slide-level branch margin을 provenance와 함께 저장한 뒤 family 단위로 covariance spectrum·max|R²|를 재는 진단으로 바꾼다. 구체적으로 7 전체, mean/projection 계열(`BM`, `QA`, `DS`), covariance 계열(`CV`), shape 계열(`BD`, `SH`, `SJ`), 그리고 leave-one-out 구성의 effective rank와 계열 간 상관을 계산한다. probe margin이 필요하다면 CA-10 경계 밖으로 명시된 각도·모드·국소 내재차원 계열 중 사용자 판정(G/H)을 통과한 것만 `SCREEN_ONLY`로 넣고, 기존 7 margin span에 대한 R²와 eff.rank 증분을 측정한다.
- **바꾸는 전제**: “현재 7개 branch의 유효 랭크가 낮으므로 서로 다른 branch를 추가하는 방향 자체가 막혔다”는 전제를 뒤집는다. 유효 랭크는 현재 margin이 span하는 선형 공간의 중복을 재는 값이지, 다른 정보원을 아직 만들지 못했다는 증거가 아니다. 특히 `BM·QA·DS`의 유효 랭크 `1.29/3`처럼 family 내부 중복이 전체 `3.52/7`을 지배할 수 있다.
- **기대하는 차이**: 성능 개선이 아니라 두 해석을 가르는 관측을 만든다. (a) family 내부 effective rank, (b) 계열 간 max|r|, (c) 기존 span에서 벗어난 probe의 R²가 판별 자료다. family 내부 중복만 크고 계열 간 직교성이 유지되면 “서로 다른 것을 재는 branch를 아직 못 만들었다”는 해석이 강해진다. probe가 기존 span에서 낮은 R²를 보이고 게이트 ①을 통과하면 branch 추가가 아직 열려 있다는 근거가 된다. 반대로 모든 family와 경계 밖 probe가 공통 span에 포화되면 “branch 추가 방향이 막혔다”는 해석이 강해진다. `Δ_macro`에 대한 정량 예측 불가. 이 진단 자체는 hold-out 미검증이다.
- **기전**: branch margin을 slide 단위 벡터로 보면 covariance 고유값 분포는 몇 개 공통 방향으로 정보가 몰렸는지를 나타내므로, family별 spectrum으로 중복의 위치를 분리할 수 있다. 이 선형 span 포화가 실제 AUROC 정보량 부족과 일치하는지는 기전 미확인.
- **실패 조건**: (1) provenance가 없어 margin을 특정 7-branch 구성에 귀속시키지 못하면 실행 무효. (2) fold 적합 1회 시간이 문서에 없으므로 재적합이 필요하면 4 GPU-hour 내 완료를 보장할 수 없다. (3) probe를 넣지 못하면 “못 만들었다”와 “막혔다”를 완전히 가를 수 없어 판별 불가. (4) margin 간 선형 중복이 낮아도 비선형 정보 중복이 있으면 이 진단이 branch 추가 가능을 과대 해석할 수 있다.

## 제안 2: trimmed mean을 rank-median으로 바꾸는 단일 집계 후보
- **바꾸는 것**: Primary 7 slide-level score를 branch raw margin의 trimmed mean에서, task·fold 내부 rank의 median으로 바꾼다. 각 task·fold에서 각 branch margin을 slide 순위(또는 empirical CDF; ties는 평균 순위)로 변환하고, 같은 slide의 7개 rank의 median을 slide score로 쓴 뒤 AUROC를 계산한다. 학습 파라미터를 도입하지 않는다. 여러 집계 함수를 Primary 7에서 비교한 뒤 좋은 것을 고르지 않고, rank-median 하나만 사전 등록해 baseline trimmed mean과 대응 비교한다. 이미 Primary 7 margin을 보고 고른 것이라면 선택 이력을 기록하고 승격 확증으로 쓰지 않는다.
- **바꾸는 전제**: “7개 branch margin은 서로 다른 scale·분산을 가진 채 additive scale에서 직접 평균해도 된다”는 전제를 뒤집는다. trimmed mean도 raw margin의 scale과 tail에 따라 branch별 실질 가중이 달라질 수 있다. rank 변환은 monotone transformation으로 slide 순서만 보존하고 branch별 scale·극단값의 영향을 제거한다.
- **기대하는 차이**: `Δ_macro`의 방향·크기에 대한 정량 예측 불가. paired per-fold Δ와 fit variance를 함께 보고, `+0.0030` 초과인지, 0 포함으로 판별 불가인지, 0 미만으로 반박인지 구분한다. rank score가 [0,1]로 제한되므로 seed 재적합에서 극단 raw margin이 최종 score에 주는 변동이 줄어 fit variance가 작아질 수 있으나, 문서에 근거가 없어 확정 예측 불가. 모든 판정은 Primary 7에 근거하고 hold-out 미검증이다.
- **기전**: raw margin이 branch별 scale·분산에 따라 다른 가중을 받으면 trimmed mean도 고분산 branch의 outlier에 민감하다. rank-median은 branch 기여를 순서 정보로 제한해 scale 편향과 outlier 영향을 줄인다. 이것이 Primary 7 AUROC를 실제로 높이는지는 기전 미확인.
- **실패 조건**: (1) margin의 절대 크기(증거 강도)가 순서보다 정보라면 rank 변환이 정보를 버려 Δ가 0 이하가 된다. (2) branch별 정보량이 다르면 rank가 모든 branch를 동등하게 취급해 성능이 떨어진다. (3) rank-median 외에 다른 집계 함수를 실행해 Primary 7에서 좋은 값을 선택하면 CA-08의 선택 편향을 반복해 실행 무효. (4) margin 재적합이 필요하면 fold 적합 1회 시간이 문서에 없어 4 GPU-hour 내 완료를 보장할 수 없다. (5) baseline과 candidate가 같은 fold·seed·provenance로 짝지어지지 않거나, 과거 여섯 러너처럼 폴백으로 같은 설정에 수렴하면 비교가 실행 무효.

## 제안 3: 승격 estimand를 seed 평균 paired Δ로 변경
- **바꾸는 것**: 승격 비교의 estimand를 “단일 적합의 fold paired Δ”에서 “K개 독립 적합의 seed 평균 fold paired Δ”로 바꾼다. baseline과 candidate를 같은 task·fold, 가능하면 같은 seed 인덱스로 짝짓고, 각 fold AUROC를 seed 평균한 뒤 Δ를 계산한다. 단일 seed의 최댓값을 선택하지 않는다. fold 적합 1회 시간이 문서에 없으므로 착수 전 1 fold 1 seed 시간을 실측하고, 4 GPU-hour에서 가능한 K를 사전 등록한다.
- **바꾸는 전제**: “단일 적합의 점추정이 후보의 성능을 충분히 대표한다”는 전제를 뒤집는다. seed 재적합 표준편차 `0.0021~0.0083`은 승격 임계 `+0.0030`과 같은 크기이므로, 단일 적합 비교는 적합 변동에 의한 뽑기일 수 있다.
- **기대하는 차이**: 성능의 기대값을 올리는 것이 아니라 Δ 추정의 불확실성을 줄이는 것이다. seed가 독립 적합 노이즈라면 seed 평균의 분산은 `1/K`로 줄고, `+0.0030` 근처의 진짜 Δ를 판별할 가능성이 높아진다. 그러나 seed 간 독립성, fold×seed 잔차, paired Δ 구조는 문서에 없으므로 정량 예측 불가. 진짜 Δ가 0이면 이 변화는 성능 개선이 아니라 null을 더 분명하게 반박하거나 판별 불가로 남게 한다. 모든 판정은 Primary 7에 근거하고 hold-out 미검증이다.
- **기전**: 독립 적합의 평균은 sampling noise를 줄인다. baseline과 candidate에 같은 seed/fold를 공통으로 쓰면 seed 주효과가 Δ에서 상쇄된다. RU-97의 `m32_raw` fold×seed 잔차 `23.9%`가 크면 seed 평균만으로 분산이 충분히 줄지 않을 수 있어 실제 감소 기전은 미확인.
- **실패 조건**: (1) fold 적합 시간이 4 GPU-hour를 초과해 K=1만 가능하면 기존 단일 적합 판정과 달라지지 않아 실패. (2) seed 변동이 독립 noise가 아니라 fold 순서·데이터 구조에 따른 불안정성이라면 seed 평균으로 줄지 않아 실패. (3) baseline/candidate의 seed pairing이 없으면 variance reduction 효과가 없어 실행 무효. (4) K를 늘린 뒤 Primary 7에서 좋은 seed·fold만 선택하면 선택 편향으로 실행 무효.

### T1 (이론)

## 이론적 병목 1
- **가정**: 유효 랭크 `3.52 / 7`은 “branch를 더 넣는 방향이 막혔다”는 사실로 읽힌다. 즉, 현재 7개 margin이 slide에서 측정 가능한 독립 증거 축을 대부분 포괄하고, 새 branch는 기존 축의 중복만 만든다고 암묵적으로 둔다.
- **이 가정이 깨지는 조건**: 성능 조회 이전의 게이트 ①에서 max |r| ≤ 0.6이고 8-branch 기준 eff.rank ≥ 4.02를 유지하며, 게이트 ②a 또는 ②b를 통과하는 branch가 존재한다. 또는 CA-10 경계 밖처럼 기존 branch가 재지 않는 slide 속성(각도·모드·국소 내재차원 등)을 라벨 무관하게 재는 branch가 만들어진다.
- **깨질 때 나타나는 관측 가능한 증상**: SCREEN_ONLY 마진에서 기존 7 branch와 낮은 상관, eff.rank 상승, 그리고 같은 fold paired Δ에서 +0.0030 초과 개선이 적합 분산 포함 시에도 0을 배제한다. 이 증상은 채택/승격의 필요 조건이지, 성능 관측만으로 기전을 입증하는 것은 아니다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: branch별 기여도는 문서에 없다. CA-09는 위치 계열 `BM·QA·DS`의 유효 랭크 `1.29 / 3`과 `RM`의 max |r| = 0.690으로 중복을 지지하지만, 동시에 형상 계열(`BD·SH·SJ`)을 “유효 랭크를 올린 경로”로 기록한다. CA-10은 형상 병목을 절사가 아니라 정보량으로 둔다. 따라서 “branch 추가 전체가 막혔다”는 가정과 충돌한다.

## 이론적 병목 2
- **가정**: `trimmed mean voting`은 branch margin을 같은 척도의 증거로 보고 최상·최하 1개 절사로 충분하다고 둔다. 즉, 집계 함수는 branch 구성 뒤에 놓인 고정 절차일 뿐, 성능의 1차 탐색 축이 아니라고 암묵적으로 둔다.
- **이 가정이 깨지는 조건**: 학습 파라미터를 도입하지 않고, 사전에 선언된 집계 규칙이 기존 trimmed mean과 다른 paired Δ를 낸다. 허용 가능한 방향은 CA-08 경계 밖인 새 정보(예측분산·Context 신호 기하)를 쓰는 `PSW`·`TGW` 계열이며, 같은 자료에서 여러 결합을 탐색한 뒤 최댓값을 확증으로 쓰는 것은 저촉이다.
- **깨질 때 나타나는 관측 가능한 증상**: 동일 branch set·동일 fold·동일 seed 조건에서 집계만 바꿨을 때 per-task Δ와 macro Δ가 변하고, 그 변화가 fold/seed 불확실성 범위보다 크다. 또는 절사된 branch가 특정 과제에서 신호였음이 paired Δ로 보인다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: CA-08은 §214의 선택 편향으로 기존 탐색을 무효화했지만, “결합 함수 일반의 무효 기전은 미확인”이고 `PSW`·`TGW`를 비저촉으로 둔다. 따라서 “집계 축은 이미 닫혔다”는 암묵적 가정과 충돌한다.

## 이론적 병목 3
- **가정**: 적합 분산(`seed std 0.0021~0.0083`)은 제거 가능한 잡음이고, 분산을 줄이면 승격 임계 `+0.0030`을 넘는 성능 개선이 자동으로 따라온다고 둔다. 또는 단일 적합의 점추정으로 후보를 고르는 것이 허용된다고 둔다.
- **이 가정이 깨지는 조건**: 분산만 줄이고 평균 Δ가 변하지 않는 구성이 나온다. 분산 감소 수단은 seed 평균, fold paired Δ, 사전 고정 seed set, closed-form readout 유지, GMM 설정 고정 등이며, 이들은 추정 불확실성을 줄이지 성능의 기대값을 올리는 것은 아니다. `D-048` 이후 best-seed 선택은 무효다.
- **깨질 때 나타나는 관측 가능한 증상**: seed sd가 줄지만 적합 평균 macro 또는 paired Δ의 점추정이 그대로다. CI가 좁아지지만 0을 포함하거나 `+0.0030`에 도달하지 못한다. RU-97처럼 단일 적합 `+0.0099`가 seed 42의 1.5 sd 뽑기로 재해석된다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: RU-97에서 seed std `0.0021~0.0083`이 승격 임계 `+0.0030`의 0.7~2.8배이고, `M32−M8`은 `+0.0003 [-0.0067, +0.0073]`로 0을 포함한다. 화이트닝은 macro를 올리지 않고 적합 분산을 4배 키운다. `D-048`은 적합 분산 보고를 필수로 한다. 기전은 미확인이다.

## 이론적 병목 4
- **가정**: 모델 개선의 축은 branch·집계·readout 구조에만 있고, 실행·provenance·평가 설계는 중립적 인프라라고 둔다. 즉, “모델을 바꿨다”는 주장은 실제 실행된 branch set·가중치·코드 해시가 기록되고 통제된다고 전제한다.
- **이 가정이 깨지는 조건**: 러너가 설정을 넘기지 못해 다른 arm이 같은 구성으로 수렴하거나, 산출물에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시가 없어 과거 결과를 특정 구성에 귀속시킬 수 없다. 그러면 비교는 실행 무효다.
- **깨질 때 나타나는 관측 가능한 증상**: 서로 다른 설정을 의도한 실행이 저장된 margin/AUROC에서 동일하거나, provenance 필드가 비어 있다. D-050처럼 여섯 러너가 전부 같은 7-branch 설정으로 수렴해 과거 branch 조합 비교가 구성 차이가 없는 실행 비교가 된다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: D-050은 `eval_v121.sh` 등 여섯 러너가 폴백에 걸려 arm이 다른 실행들이 전부 같은 7-branch 설정으로 수렴했고, 저장 예측 파일에 provenance가 없어 과거 산출물을 구성에 귀속시킬 수 없었다고 기록한다. 이 축이 미건드려진 이유는 생각 부족이 아니라 계측 복구 전 verified 비교가 불가능했기 때문이었다.

## 이론적 병목 5
- **가정**: 4 GPU-hour 예산 안에서 Primary 7 50-fold 후보 비교가 가능하다고 둔다. 즉, fold 적합 1회 비용이 충분히 작거나 이미 알려져 있어, 후보 수와 seed 수를 예산에 맞출 수 있다고 전제한다.
- **이 가정이 깨지는 조건**: fold 적합 1회 소요 시간이 측정되지 않았거나, 단일 M 2시간 또는 합계 4 GPU-hour 초과가 예상된다. 그러면 50-fold paired Δ를 산출할 수 없어 판별 불가/실행 무효다. 4h 내 판별 가능하려면 fold 1회 시간과 seed 수를 사전에 고정하고, 예산 초과 전에 중단해야 한다.
- **깨질 때 나타나는 관측 가능한 증상**: 예산 초과 중단, fold 일부만 완료, 또는 seed 수를 1로 줄여 적합 분산을 보고할 수 없는 상태. fold 적합 시간이 문서에 없으므로, 4h 내 판별 가능하다는 주장은 현재 근거가 없다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `PROJECT.md` §2와 `current_status.md`는 GPU 1장 합계 최대 4 GPU-hour, 단일 M 2시간 초과를 중단 조건으로 두고, fold 적합 1회 소요 시간은 측정된 적이 없다. 따라서 “4h 안에 50-fold 비교가 가능하다”는 가정과 충돌한다.

## 이론적 병목 6
- **가정**: Primary 7의 paired Δ가 SEAL 10 hold-out 개선의 유효한 대리 지표라고 둔다. 즉, Primary 7에서 `+0.0030`을 넘으면 최종 목표(ABMIL `0.7266`, MeanMIL `0.7125`)에 도달할 확률이 충분히 크다고 암묵적으로 둔다.
- **이 가정이 깨지는 조건**: Primary 7에서 승격 기준을 넘지만 SEAL 10 개봉에서 ABMIL/MeanMIL을 넘지 못한다. 단, 그 결과를 보고 모델을 고르면 `D-047`의 선택 금지 위반이다.
- **깨질 때 나타나는 관측 가능한 증상**: Primary 7 승격과 SEAL 10 macro-AUC 목표 도달이 분리된다. ICF v120 6-branch의 SEAL 10 `0.6972`는 ABMIL 대비 `−0.0294`, MeanMIL 대비 `−0.0153` 거리로 남아 있다. 모든 후보 판정은 hold-out 미검증이다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `PROJECT.md` §1은 “Primary 7의 개선이 hold-out 개선을 보장하지 않는다”고 명시한다. 따라서 Primary 7 개선만으로 SEAL 10 이긴다는 주장은 증거 범위를 넘는다.

### M1 (실측)

## 실측 1: 유효 랭크 3.52 해석 판별
- **건드려야 하는 코드·설정**: `scripts/analysis/branch_screen.py`(`SCREEN_ONLY=1`), 공식 7-branch 설정 `configs/baseline/v121_active.yaml`, 후보 branch margin 생성 코드(문서에 없음), provenance writer·오염 검사 재설계(구체 경로 문서에 없음). 후보 margin을 Primary 7 50-fold에서 생성해 기존 7-branch와 max |r|, 8-branch eff.rank를 측정한다.
- **비용**: 후보 margin 생성과 branch_screen 비용은 미실측. fold 적합이 필요하면 fold 적합 1회 시간 미실측, Primary 7 350-fold 총 GPU-hour 미실측, 메모리·연산 미실측. 저장 patch feature에서 계산 가능해도 GPU/CPU 소요는 미실측.
- **막는 것**: 유효 랭크 3.52 자체는 “branch 추가 불가”를 증명하지 못한다. D-050으로 gate ① 실행 경로가 현재 무효이고, 저장 예측에 provenance가 없어 margin을 구성에 귀속시킬 수 없다. 후보 branch가 아직 없으면 “못 만들었다”와 “막혔다”를 구분할 관측이 없다. 후보 1개라도 max |r| ≤ 0.6이고 8-branch eff.rank ≥ 4.02이면 “아직 못 만들었다” 쪽 관측이 된다.
- **중단 기준**: 후보 1개라도 max |r| > 0.6 또는 8-branch eff.rank < 4.02이면 성능 조회 없이 게이트 ① 실패로 종료. provenance·오염 검사를 복구하지 못하면 실행 무효. 단일 M 2시간 또는 합계 4 GPU-hour 초과 시 중단.

## 실측 2: 학습 파라미터 없는 집계 변경
- **건드려야 하는 코드·설정**: 집계 함수 경로(문서에 없음). 고정 집계(mean, median, min, max, rank average, 다른 trim)와 새 정보 가중(PSW/TGW)을 사전등록으로 구현. per-branch margin은 `configs/baseline/v121_active.yaml`을 `evaluate_pure.py`에 직접 넘겨 provenance와 함께 생성한다.
- **비용**: attributable per-branch margin이 이미 있으면 재집계 비용은 미실측(오프라인, GPU 필요 여부 미실측). 없으면 7-branch margin 재생성에 fold 적합 1회 시간 미실측, 350-fold 총 GPU-hour 미실측. 변형 수는 재집계 비용만 추가하고 fold 적합 비용은 공유 가능.
- **막는 것**: CA-08은 기존 branch margin/probability 위의 결합 탐색과 같은 Primary 7에서 최댓값을 확증으로 쓰는 절차를 닫았다. 따라서 ad hoc 탐색은 저촉. 사전등록·선택 이력·대응 Δ가 없으면 실행 무효. provenance 부재로 과거 margin을 구성에 귀속시킬 수 없다.
- **중단 기준**: per-branch margin을 provenance와 함께 만들 수 없으면 중단. 결과를 보고 변형을 고르면 중단. 부분집합을 성능으로 골라야 하면 중단. 같은 fold 짝짓기 Δ를 계산할 수 없으면 중단. 4 GPU-hour 초과 시 중단. Δ가 불확실성 안에 있으면 판별 불가로 기록.

## 실측 3: 적합 분산 감소
- **건드려야 하는 코드·설정**: seed/무작위 소스 제어와 seed averaging(구체 설정 경로 문서에 없음), `scripts/eval_v121.sh`/`eval_seal_tasks.sh`/`evaluate_pure.py`의 폴백 경로 수정, 적합 분산을 포함한 승격 추정 코드(미승인·구체 경로 문서에 없음).
- **비용**: seed R회 반복은 fold 적합 비용을 R배로 한다. fold 적합 1회 시간 미실측이므로 총 GPU-hour 미실측. R=8 잠정 보정도 단일 적합 비용 미실측이면 총 비용 미실측. 메모리·연산 미실측.
- **막는 것**: 분산 감소는 기대 AUROC를 올리는 성능 개선이 아니라 Δ의 SE를 줄이는 정밀도 개선이다. 보고된 seed sd 0.0021~0.0083 중 0.0050·0.0083은 +0.0030보다 크므로, 분산 감소 없이는 단일 적합 Δ를 판정 근거로 쓰기 어렵다. 그러나 적합 분산 포함 승격 절차가 미승인이고, seed 변동의 원인이 문서에 없어 기전 미확인이다.
- **중단 기준**: R회 반복이 4 GPU-hour를 넘으면 중단. 분산 감소 후 점추정 Δ가 +0.0030 미만이면 승격 불가이며, SE에 따라 판별 불가 또는 반박으로 기록. seed averaging이 새 무작위 소스나 provenance 미기록 구성을 만들면 실행 무효.

## 실측 4: 건드리지 않은 축
- **건드려야 하는 코드·설정**: (a) 집계: 위 실측 2. (b) LSAK/QJF: 사양(L/S 원출, 차원, bandwidth, 정규화, margin 스칼라, max |r|·eff.rank 단위)이 문서에 없어 구현 경로 없음. (c) CA-R1: salience anchor subsampling 코드(문서에 경로 없음)와 라벨 무관 선택 신호 부재. (d) AKS/LID: `sj.py` 주변 형상 하위축, 사용자 판정 대기.
- **비용**: 집계는 margin 재생성 비용 미실측. LSAK Gram 계산 비용은 저장 feature에서 가능해도 미실측, fold 적합 필요하면 미실측. CA-R1은 강도 선택 신호가 없어 과제별 강도를 정할 수 없고, 실행 비용 미실측. AKS/LID 추가 측정 비용 미실측.
- **막는 것**: 집계가 건드리지 않은 이유는 CA-08의 선택 편향 기각이 ad hoc 탐색에 타당하지만, 사전등록 제한 연구는 경계 밖이다. LSAK가 건드리지 않은 이유는 사양 부재로 타당 운영 막힘. CA-R1은 이득이 과제별이고 선택 신호가 없어 타당 과학적 막힘. fold 적합 시간 미측정은 연구 축이 아니라 운영 계측 누락이다.
- **중단 기준**: 사전등록 없이 집계 변형을 늘리면 중단. LSAK 사양을 확정하지 않고 Gram을 돌리면 실행 무효. CA-R1에서 결과 AUROC를 보고 강도를 고르면 Oracle 금지 위반으로 중단. 4 GPU-hour 초과 시 중단.

## 실측 5: 4 GPU-hour 내 판별 가능
- **건드려야 하는 코드·설정**: 먼저 provenance writer smoke test와 단일 fold 타이밍(`evaluate_pure.py` + `configs/baseline/v121_active.yaml`, 단일 fold manifest 경로 문서에 없음). 이후 후보별 Primary 7 50-fold 대응 비교. SEAL 10은 열지 않는다.
- **비용**: fold 적합 1회 시간 미실측, Primary 7 350-fold 총 GPU-hour 미실측. 따라서 4 GPU-hour 내 50-fold 성능 비교 가능 여부는 현재 판별 불가. 오프라인 재집계나 gate ① margin 계산도 비용 미실측.
- **막는 것**: D-050으로 gate·승격 비교를 verified로 수행할 수 없다. 적합 분산 포함 승격 절차가 미승인이다. branch별 기여도가 미실측이라 기여도 기준으로 우선순위를 정할 수 없다. fold 적합 시간이 알려지지 않아 예산 내 실행 수를 계산할 수 없다.
- **중단 기준**: 단일 fold 타이밍 또는 smoke test가 2시간을 넘으면 중단. 50-fold 1회 비용이 4 GPU-hour를 넘으면 승격 비교 중단. 부분 fold만으로는 승격 판정 불가이며 탐색 기록으로만 처리. provenance 없이 산출된 성능 값은 실행 무효로 처리. 예산 내 판별 가능한 것은 (1) 계측 복구/타이밍, (2) provenance margin 생성이 완료되면 gate ① 직교성 판정, (3) 사전등록된 재집계만 가능하면 집계 후보의 대응 Δ 탐색 기록. 성능 승격 판정은 현재 판별 불가.

### R1 (반증)

## 반론 1
- **공격 대상**: "유효 랭크 3.52/7은 branch 추가 방향이 막혔는지, 아니면 서로 다른 branch를 아직 못 만들었는지 중 하나를 판정한다"는 주장.
- **더 단순한 설명**: 3.52는 현재 7개 branch의 중복 정도를 재는 값이지, 탐색 공간 전체의 상한이 아니다. 같은 patch feature와 closed-form ridge에서 나온 branch, 특히 위치 계열 `BM`·`QA`·`DS`가 이미 유효 랭크 1.29/3으로 겹친다는 기존 관측이 3.52를 더 단순하게 설명한다. 따라서 "막혔다" 또는 "못 만들었다"를 3.52만으로 선택하는 것은 증거 범위를 넘는다.
- **무엇이 참이면 이 주장이 무너지는가**: 사전에 고정한 신규 branch 후보가 게이트 ①에서 `max |r| ≤ 0.6`이고, 8-branch 기준 `eff.rank ≥ 4.02`를 만족하면 "3.52 때문에 막혔다"는 주장이 무너진다. 반대로 여러 독립 계열 후보가 게이트 ①에서 탈락하면 "현재 후보 공간에서 막혔다"는 쪽이 지지된다.
- **이 반론의 한계**: 게이트 ① 통과가 성능 개선, 승격, 또는 최종 hold-out 이득을 보장하지 않는다. `max |r|`와 `eff.rank`는 라벨 무관 중복 지표일 뿐이며, 게이트 ② 정보량과 §4 대응 Δ는 별개다. 또한 3.52는 공식 7-branch의 50-fold 측정값이지 다른 feature·readout·context 구성의 보편 상한이 아니다.

## 반론 2
- **공격 대상**: "집계(trimmed mean voting)를 바꾸는 축은 아직 거의 탐색되지 않았고, 학습 파라미터를 도입하지 않으면서 집계를 바꾸는 것이 유효한 새 축이다"는 주장.
- **더 단순한 설명**: "거의 탐색되지 않았다"는 관측은 여섯 러너가 같은 7-branch 설정으로 수렴해 과거 arm 비교가 무효였다는 사실로 설명된다. 즉 축이 열려서가 아니라, 유효한 비교 경로가 없었기 때문에 탐색 기록이 없는 것일 수 있다. 게다가 `CA-08`은 기존 branch margin 위의 결합 함수·부분집합 탐색을 선택 편향으로 닫았다. 같은 저랭크 margin을 다른 함수로 섞는 것은 3.52차원 중복 정보를 재가중하는 데 그칠 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 탐색에 쓴 평가자료의 최댓값을 확증으로 쓰지 않고, 사전에 고정한 집계 규칙(또는 라벨 무관 기준)을 적용해 §4 대응 per-fold Δ와 적합 분산을 함께 보고, `+0.0030`을 넘으면 이 반론은 무너진다. `CA-08` 경계 밖인 새 정보(예측분산·Context 신호 기하 등)를 쓰는 경우에도, 선택 이력을 기록하고 성능 확증을 분리하면 저촉이 아니다.
- **이 반론의 한계**: `CA-08`이 닫은 것은 "기존 탐색 절차의 선택 편향"이지 결합 함수 일반의 무효 기전이 아니다. 문서도 "결합 함수 일반의 무효 기전은 미확인"이라 적는다. 따라서 새 정보나 사전 규칙을 쓰면 열릴 수 있고, 이 반론은 모든 집계 변경을 막는 것이 아니다.

## 반론 3
- **공격 대상**: "적합 분산 0.0021~0.0083이 승격 임계 `+0.0030`보다 크므로, 성능을 올리는 것보다 분산을 줄이는 것이 더 나은 투자다"는 주장.
- **더 단순한 설명**: 0.0021~0.0083은 문서에서 RU-97 GF 구성의 적합 sd 범위이고, 0.0021은 임계보다 작다. 승격 추정 대상은 절대 macro가 아니라 같은 fold에서 짝지은 Δ다. 따라서 seed macro sd가 임계보다 크다는 사실만으로는 "단일 적합 차이가 대부분 뽑기" 또는 "분산 감소가 우선"을 지지하지 않는다. 더 직접적인 병목은 `D-050`의 provenance 무효다.
- **무엇이 참이면 이 주장이 무너지는가**: 계측이 복구된 상태에서, 후보-기준 대응 Δ의 적합 분산이 `+0.0030`보다 크고, seed에 따라 지지/반박/판별 불가 판정이 바뀌는 것이 실측되면, 분산 감소가 성능 개선보다 우선할 수 있다. 그러나 §4는 95% CI 하한이 `+0.3%p`를 넘을 것을 승격 필수 조건으로 요구하지 않는다.
- **이 반론의 한계**: 분산 감소가 승격의 필수 조건은 아니지만, `D-048` 이후 적합 분산 보고는 필수이고, 높은 분산은 과제별 악화나 부호 불확실성을 숨길 수 있다. 또한 0.0021~0.0083은 RU-97 GF 구성의 적합 sd로 특정되므로, 공식 7-branch 전체의 적합 분산으로 자동 확장하면 안 된다.

## 반론 4
- **공격 대상**: "아직 아무도 건드리지 않은 모델 축이 남아 있고, 그것이 이번 4 GPU-hour의 대상이 될 수 있다"는 주장.
- **더 단순한 설명**: 현재 상태 문서의 현재 목표는 "계측 복구"다. provenance writer, `SH`/`SJ` control, 오염 검사 재설계, 단위 테스트가 복구되기 전에는 게이트·승격 비교를 verified로 수행할 수 없다. 따라서 가장 명확하게 건드리지 않은(또는 복구되지 않은) 축은 새 모델 축이 아니라 실행·기록 경로다. 모델 축들은 이미 closed_axes에 닫힘/재개/경계 미확정으로 포진해 있고, 재개 축도 선택 신호·사양·재현 조건이 미해소다.
- **무엇이 참이면 이 주장이 무너지는가**: closed_axes에 저촉되지 않는 새 축이, (1) 사전 등록 설계, (2) 라벨 무관 게이트 ① 비용, (3) fold 적합 시간 실측, (4) provenance 기반 오염 검사를 통과할 수 있는 실행 경로를 갖추면 "모델 축이 남아 있다"는 주장이 성립할 수 있다.
- **이 반론의 한계**: closed_axes의 `CLOSED`는 영구 불가능이 아니라 평가된 주장의 반박 또는 자원 배분상 보류다. `D-048`로 `CA-14`가 폐기되고 학습 파라미터 후보가 열렸으며, `CA-04`·`CA-02`·`CA-R1`도 재개됐다. 따라서 "아무도 건드리지 않은 축이 없다"고 단정하면 닫힌 축의 경계를 넘어서는 과잉 일반화가 된다.

## 반론 5
- **공격 대상**: "위 후보들 중 4 GPU-hour 안에서 판별 가능한 후보가 있다"는 주장.
- **더 단순한 설명**: fold 적합 1회 소요 시간은 측정된 적이 없다. Primary 7 공식 비교는 7 과제 × 50 fold = 350 fold fit을 필요로 한다. 4 GPU-hour 예산에서 1회 전체 실행만 가능하려면 fold당 GPU 시간이 4/350시간 이하여야 하고, 후보·기준·적합 분산 재적합까지 포함하면 더 짧아야 한다. 이 실측이 없으면 "판별 가능"은 예산 전제가 없는 추측이다.
- **무엇이 참이면 이 주장이 무너지는가**: fold 적합 GPU 시간을 실측해, 기준선 재현·후보 1개·필요한 seed 재적합이 합계 4 GPU-hour 안에 들어가고, provenance가 산출물에 기록되어 독립 재계산이 가능하면 판별 가능하다고 말할 수 있다.
- **이 반론의 한계**: fold fit이 GPU를 거의 쓰지 않거나 feature 캐시가 있으면 4 GPU-hour가 충분할 수 있다. 그러나 문서에 fold 적합 시간이 없으므로, 충분함을 주장하려면 실측이 선행된다. 또한 4 GPU-hour 안에 "성능 확인"이 아니라 게이트 ① 수준의 저비용 중복 검사만 가능할 수 있고, 그것은 승격 판정이 아니다.

## 반론 6
- **공격 대상**: "Primary 7에서 근거를 얻어 모델을 개선하면 SEAL 10에서 ABMIL을 이기는 경로가 열린다"는 전제.
- **더 단순한 설명**: Primary 7에서 여러 후보를 탐색하고 그 결과로 고르면, §214와 같은 선택 편향이 생긴다. 7개 과제는 작고, fold는 과제 내 독립이 아니므로, 탐색 횟수가 커질수록 Primary 7 점수는 실제 일반화 성능보다 선택 노이즈를 더 반영할 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 후보 수·탐색 이력·적응을 사전에 등록하고, 승격은 §4의 대응 Δ와 적합 분산으로만 판정하며, SEAL 10은 최종 개봉 전에 어떤 후보 선택에도 쓰지 않으면, Primary 7 근거의 사용은 허용된다. 그러나 이 조건이 만족돼도 SEAL 10 이득은 보장되지 않는다.
- **이 반론의 한계**: `D-047`은 SEAL 10 결과로 고르는 것을 금지할 뿐, Primary 7 기반 개선 자체를 금지하지 않는다. 따라서 이 반론은 "Primary 7 개선 = SEAL 10 승리"라는 인과적·보장적 주장에만 해당한다.

### A1 (감사)

## 점검 1: Primary 7/SEAL 10 선택 경계
- **확인하려는 것**: 모델·집계·branch 개선 후보가 SEAL 10 결과로 선택되지 않고 Primary 7만으로 결정되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: SEAL 10에서 좋은 후보가 Primary 7 근거 없이 승격되고, 최종 hold-out 개봉이 선택 편향으로 실행 무효.
- **확인 방법**: `docs/PROJECT.md` §3.1·D-047, `docs/current_status.md` 현재 목표, `docs/history/archive.md`와 `talks/council/`에서 SEAL 10 수치 인용 시점과 결정 시점 대조.

## 점검 2: arm이 다른 실행의 구성 수렴
- **확인하려는 것**: 과거 branch 조합 비교가 실제로 서로 다른 branch 구성으로 실행됐는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: arm이 다른 실행이 같은 7-branch 설정으로 수렴해 구성 차이가 없는 Δ가 branch 조합 효과로 해석되고, 과거 비교가 실행 무효.
- **확인 방법**: `scripts/eval_v121.sh`, `scripts/eval_seal_tasks.sh`, `configs/`, 실행 로그의 branch list·가중치·코드 해시, `docs/PROJECT.md` D-050 정정 대조.

## 점검 3: 공식 기준선 0.6227의 provenance 부재
- **확인하려는 것**: 0.6227을 특정 7-branch 구성·fold manifest·데이터 버전·코드 해시로 귀속시켜 대응 Δ의 기준선으로 쓸 수 있는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 기준선 구성이 불명확하면 후보 Δ에 다른 구성의 차이가 섞여 승격 판정이 실행 무효.
- **확인 방법**: `docs/PROJECT.md` §3.2·D-050, 저장 예측 파일의 provenance 필드, `scripts/eval_v121.sh` 재현 로그, macro 0.6226/0.6227 대조.

## 점검 4: fold manifest 고정
- **확인하려는 것**: 후보와 기준선이 같은 PathoBench 50-fold를 같은 순서·데이터 버전으로 사용했는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: fold 재분할·누락·순서 차이가 Δ에 섞여 과제 난이도 편향으로 오기입되고, 대응 비교의 정밀도가 무효.
- **확인 방법**: 산출물에 fold manifest hash 기록 여부, `scripts/` fold 로딩 코드, 데이터 버전, 후보/기준선 fold ID 교차 대조.

## 점검 5: 전처리·PCA·GMM fit 범위
- **확인하려는 것**: PCA-32, whitening, background GMM, 정규화 통계가 fold context만 fit되고 query/test·SEAL 10을 포함하지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test/query fold의 분산·평균·GMM 성분이 fit에 유입돼 AUROC가 인플레이트되고, Primary 7 개선이 hold-out으로 전이되지 않음.
- **확인 방법**: `src/models/`에서 fit 호출 위치, config의 fit scope, fold별 fit 로그, 합성 데이터로 test fold 포함 시 margin 변화 단위 테스트.

## 점검 6: 라벨 사용 범위
- **확인하려는 것**: 7개 branch margin 생성과 closed-form ridge readout에 query/test 라벨이 쓰이지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: query 라벨이 margin/readout에 유입되면 Primary 7과 SEAL 10 모두 인플레이트되고, hold-out 판정이 실행 무효.
- **확인 방법**: 코드에서 label tensor 사용 지점, `configs/` label flag, 라벨 셔플 시 margin 변화 테스트, context/query 라벨 분리 단위 테스트.

## 점검 7: 유효 랭크 3.52의 계산 단위
- **확인하려는 것**: 3.52/7이 branch margin의 실제 중복을 재는가, 아니면 스케일·과제·fold 구조에서 나온 산술인가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: branch가 실제로 직교해도 스케일 차이로 랭크가 낮게, 또는 공통 전처리 때문에 높게 나와 "branch 추가 불가"로 오해.
- **확인 방법**: `scripts/analysis/branch_screen.py`의 effective rank 계산 코드, margin 표준화 단위(fold/task/slide), 350-fold 짝짓기, 표준화 전후 재계산.

## 점검 8: branch별 기여도 측정 부재
- **확인하려는 것**: 7개 branch 중 어떤 것이 중복·기여하는지 측정된 근거가 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 유효 랭크 3.52만 보고 branch 추가/제거를 결정하면, 실제 정보량 병목과 스케일 문제를 구분 못 해 판별 불가.
- **확인 방법**: SCREEN_ONLY margin으로 branch 간 상관, ablation paired Δ, contribution decomposition을 provenance와 함께 실행.

## 점검 9: seed 분산과 승격 Δ의 estimand 불일치
- **확인하려는 것**: seed std 0.0021~0.0083이 승격 기준의 대응 per-fold Δ_macro와 같은 추정 대상에서 측정됐는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 절대 macro의 seed 변동으로 대응 Δ의 불확실성을 판단해, +0.0030 임계와 비교 불가하거나 단일 적합 차이를 뽑기로 오해.
- **확인 방법**: 동일 seed/fold에서 후보-기준선 paired Δ 재계산, seed별 Δ의 SE·CI, 절대 macro sd와 Δ sd 분리 보고.

## 점검 10: 적합 분산의 원인 미확인
- **확인하려는 것**: seed만 바꿔도 달라지는 원인이 GMM·subsample·PCA·ridge·데이터 순서 중 무엇인가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: seed 평균으로 분산을 줄여도 fold×seed 잔차 때문에 줄지 않거나, 성능 개선이 아니라 측정 노이즈만 줄이는 것을 성능으로 오기입.
- **확인 방법**: stochastic component별 고정 실험(GMM 고정, subsample 고정, PCA 고정), RNG 로그, fold×seed 분산분해 재계산.

## 점검 11: 분산 감소와 성능 개선의 구분
- **확인하려는 것**: seed averaging 등 분산 감소가 Primary 7 대응 Δ의 기댓값을 올리는지, 아니면 단일 적합의 불확실성만 줄이는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 분산 감소 arm이 Δ CI를 좁히기만 하면 성능 개선으로 승격돼 hold-out에서 이득 없음.
- **확인 방법**: 사전 등록한 분산 감소 arm의 paired Δ 점추정과 CI, seed variance 보고, 성능 개선과 측정 정밀도 개선 분리.

## 점검 12: 기존 집계 탐색의 선택 편향
- **확인하려는 것**: 기존 집계 탐색에서 Primary 7에서 고른 구성을 같은 Primary 7에서 성공으로 판정한 선택 편향이 있었는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 여러 집계 중 Primary 7 최댓값을 고르면 CA-08의 선택 편향이 반복되고 승격 근거가 실행 무효.
- **확인 방법**: `docs/closed_axes.md` CA-08, RU-90/§214 기록, 신규 집계 후보의 사전 등록·선택 이력·확인 분리 설계 확인.

## 점검 13: 신규 집계 후보의 확인 분리 설계
- **확인하려는 것**: 학습 파라미터 없는 집계 변경을 탐색할 때 선택에 쓴 Primary 7을 확증으로 재사용하지 않는 유효한 설계가 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 탐색과 확증이 같은 fold·같은 자료에서 수행돼 최댓값 편향이 승격 기준에 섞임.
- **확인 방법**: RU 카드에 탐색/확증 자료 접근, 후보 수, 선택 규칙, paired Δ 계산, SE 병기 여부 확인.

## 점검 14: 기준선 자체의 선택 편향
- **확인하려는 것**: 공식 7-branch 기준선이 Primary 7에서 선택된 결과로, 후보 비교의 기준선이 낙관 편향될 수 있는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 후보가 이미 Primary 7에서 선택된 기준선을 이겨야 하므로, 기준선 선택에 쓴 탐색 횟수를 반영하지 않으면 Δ 과소/과대 평가.
- **확인 방법**: D-042, RU-90, 5-branch 0.6171 vs 7-branch 0.6227 paired Δ, 선택 이력과 승격 기준 적용 기록.

## 점검 15: 건드리지 않은 축의 근거
- **확인하려는 것**: 건드리지 않은 축이 closed_axes의 경계·재개 조건·사용자 판정 대기 등 문서화된 근거에 의해 닫혀 있는가, 아니면 기록이 없는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 근거 없이 미탐색 축이 "생각이 안 난" 축으로 남거나, 닫힌 축을 경계 없이 재시도해 실행 무효.
- **확인 방법**: `docs/closed_axes.md` CA-08·CA-09·CA-10·CA-R1·§3-G/H/I/J, D-048·D-049 대조.

## 점검 16: 4 GPU-hour 판별 가능성
- **확인하려는 것**: fold 적합 1회 시간이 측정되어 4 GPU-hour 안에 후보·seed·fold 수를 계산할 수 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: fold 적합 시간 미측정 상태에서 "4 GPU-hour 안에 판별 가능"을 주장하면 예산 초과 또는 판별 불가.
- **확인 방법**: GPU 4에서 단일 fold 적합 wall time 측정, seed 수·후보 수·fold 수 예산 계산, OOM/중단 조건 기록.

## 점검 17: 오염 검사 재설계 전 검증
- **확인하려는 것**: SCREEN_ONLY가 앙상블에 영향을 주지 않았는지 provenance로 검증할 수 있는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 값 앵커 4자리 정밀도를 유지 못 해 구성 변화와 수치 드리프트를 구분 못 하고, 게이트 ① 통과 주장이 실행 무효.
- **확인 방법**: `docs/PROJECT.md` §3.4, anchor 0.4421/0.5553 vs direct 0.4426/0.5546, 7-branch 0.4661/0.5408, provenance 필드 대조.

## 점검 18: 계측 복구 전 verified 판정 금지
- **확인하려는 것**: provenance writer, SH/SJ control, 오염 검사 재설계, 단위 테스트 복구 전 게이트·승격 비교를 verified로 인용하지 않는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 복구 전 산출된 Δ가 verified로 인용돼 실행 무효.
- **확인 방법**: `docs/current_status.md` 현재 목표, `D-050`, `tests/` 통과 여부, provenance 필드 존재.

---

## Phase 2 — 교차

### T1 (이론)

## 이론적 병목 1
- **가정**: P1-1은 (i) provenance writer·폴백 통제가 복구되어 arm을 구성에 귀속시킬 수 있고, (ii) `branch_screen.py`의 max |r|·eff.rank가 branch margin의 선형 중복만으로 “branch 추가 방향이 막혔다/아직 못 만들었다”를 판별할 수 있고, (iii) FC 1개 probe가 그 판별에 충분하며, (iv) 350-fold margin 기록이 4 GPU-hour 안에 가능하다고 두고 있다.
- **이 가정이 깨지는 조건**: provenance 필드가 없거나 러너가 7-branch로 수렴하면 실행 무효. FC가 CA-09·CA-10·CA-04 경계에 들어가면 기각. fold 적합/마진 기록 시간이 4 GPU-hour를 초과하면 예산 내 판별 불가. FC 1개만 실패해도 전체 branch 추가 방향을 막혔다고 결론짓지 못하며, gate ① 통과만으로는 gate ② 정보량과 승격을 보장하지 못한다.
- **깨질 때 나타나는 관측 가능한 증상**: 저장 산출물에 branch list·가중치·코드 해시·fold manifest 해시가 누락되거나, arm이 같은 7-branch로 수렴하면 실행 무효. max |r| > 0.6 또는 8-branch eff.rank < 4.02이면 중복 쪽으로 반박. max |r| ≤ 0.6이고 eff.rank ≥ 4.02여도 gate ② 미충족이면 지지/반박이 아니라 판별 불가. 성능 조회 전에 후보가 등록되지 않으면 선택 편향으로 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: D-050에서 여섯 러너가 같은 7-branch로 수렴했고, 저장 예측 파일에 provenance가 없어 과거 산출물을 구성에 귀속시킬 수 없다. current_status의 현재 목표가 계측 복구이며, closed_axes §3-I는 게이트 ① 실행 경로(SCREEN_ONLY)가 현재 무효라고 기록한다. 공식 7-branch 값은 unverified-hold다.

## 이론적 병목 2
- **가정**: P1-2는 closed-form ridge의 예측분산 `v_b`가 각 branch margin의 실제 fold 오차 또는 신뢰도를 반영하고, `m_b / sqrt(v_b)` 표준화가 branch 간 scale을 교정하며, `v_b` 산출 경로가 이미 존재하거나 성능 조회 전에 고정 가능하다고 두고 있다.
- **이 가정이 깨지는 조건**: `v_b`가 context 크기, 조건부 특이성, 구현 노이즈, fold 난이도 등 margin 오차와 무관한 요인을 반영하면 가중이 노이즈가 된다. `v_b`가 0 또는 무한에 가까워 처리 규칙이 없으면 score가 미정의다. 고분산 branch가 극값·이질성 신호를 담고 있으면 표준화가 신호를 죽인다. `v_b` 공식·seed·fold·코드 해시를 기록하지 않으면 비교가 실행 무효다. Primary 7 결과를 보고 `v_b` 변형을 고르면 CA-08 선택 편향으로 실행 무효 또는 반박.
- **깨질 때 나타나는 관측 가능한 증상**: `v_b`가 context 슬라이드 수나 fold 난이도와 강하게 상관하면 기전 미확인이고 성능 개선 근거가 아니다. z 변환 후 seed 재적합에서 score 변동이 줄지 않으면 적합 분산 감소 실패. paired Δ가 0을 포함하면 판별 불가, 음이면 반박. `v_b` 산출 경로가 코드에 없으면 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `v_b`가 실제 fold 오차를 예측한다는 실측은 문서에 없다(해당 관측 없음). 다만 CA-08은 PSW를 경계 밖으로 두면서도 “예측분산의 유효성”을 남은 검증으로 남겨 두고 있고, D-050의 provenance 결함은 baseline/candidate의 대응 비교 전제와 충돌한다.

## 이론적 병목 3
- **가정**: P1-3과 P2-3은 seed 재적합 변동이 독립 적합 노이즈이고, baseline/candidate를 같은 seed·fold로 짝지으면 seed 주효과가 Δ에서 상쇄되며, seed 효과가 baseline/candidate 간 공통 additive이고 seed×candidate 상호작용이 작으며, seed 평균으로 Δ의 불확실성이 충분히 줄어 +0.0030 근처를 판별 가능하다고 두고 있다.
- **이 가정이 깨지는 조건**: fold×seed 잔차, fold 순서, 데이터 민감성, stochastic preprocessing가 주원인이라면 seed 평균으로 줄지 않는다. seed×candidate 상호작용이 크면 같은 seed pairing으로도 상쇄되지 않는다. fold 적합 1회 시간이 문서에 없어 4 GPU-hour 안에서 K seed를 확보하지 못하면 실행 무효 또는 판별 불가. seed pairing이 없으면 variance reduction 효과가 없다. seed 수·seed 집합을 Primary 7 결과 보고 정하면 선택 편향이고, best-seed를 쓰면 실행 무효다.
- **깨질 때 나타나는 관측 가능한 증상**: seed 평균 Δ의 구간이 여전히 0을 포함하면 판별 불가. seed별 Δ 부호가 갈리면 seed 주효과 상쇄 실패. fold×seed 잔차가 seed 주효과보다 크면 seed 평균의 유효성 반박. K를 늘려도 SE가 줄지 않으면 기전 미확인. 예산 초과 또는 seed별 fold AUROC 기록 누락이면 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: RU-97에서 `m32_raw`의 fold×seed 잔차가 23.9%로 seed 주효과보다 크다고 실측되어, seed 평균만으로 분산이 줄지 않을 수 있다. current_status는 적합 분산을 포함한 승격 판정 절차가 미승인이라고 기록한다. fold 적합 1회 시간은 문서에 없다. 또한 공식 7-branch는 태그 간 per-fold AUROC max|diff| = 0.000e+00으로 결정성이 기록된 층이 있고, RU-97 GF에서는 seed std 0.0021~0.0083이 실측된 층이 있어, seed 평균 estimand가 적용되는 구성이 불명확하다.

## 이론적 병목 4
- **가정**: P2-1은 branch margin의 선형 covariance spectrum·effective rank·family R²가 AUROC 정보 중복을 대표하고, family 내부 중복과 계열 간 직교성으로 “branch 추가 방향이 막혔다/아직 못 만들었다”를 가를 수 있으며, CA-10 경계 밖 probe가 이미 사용 가능하다고 두고 있다.
- **이 가정이 깨지는 조건**: margin 간 비선형 정보 중복이 있으면 선형 eff.rank로 branch 추가 가능을 과대 또는 과소 해석한다. CA-10 관련 G/H가 사용자 판정 대기 상태면 probe를 넣을 수 없어 두 해석을 완전히 가를 수 없어 판별 불가. provenance가 없으면 margin을 공식 7-branch에 귀속시킬 수 없어 실행 무효. fold margin 재적합이 필요하고 시간이 4 GPU-hour를 초과하면 예산 내 판별 불가. family grouping이 margin 생성 기전과 일치하지 않으면 family span 해석 자체가 무효다.
- **깨질 때 나타나는 관측 가능한 증상**: family eff.rank가 낮고 계열 간 max|r|도 높으면 중복 쪽으로 반박. probe가 기존 span에서 낮은 R²를 보이지만 gate ②를 통과하지 못하면 “직교”와 “정보량”을 구분 못 해 판별 불가. probe가 G/H를 통과하지 못해 부재하면 실행 무효 또는 판별 불가. provenance 불일치 또는 예산 초과이면 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: closed_axes §3-G·H는 AKS/LID 하위축이 아직 사용자 판정 대기이므로 “probe를 통과한 것만 쓴다”는 전제와 충돌한다. CA-10은 형상 계열 병목이 절사가 아니라 정보량이라고 기록해, 선형 span 포화만으로 branch 추가 가능을 판별한다는 전제와 충돌한다. D-050의 provenance 결함도 margin 귀속 전제와 충돌한다.

## 이론적 병목 5
- **가정**: P2-2는 branch margin의 절대 크기보다 slide 순위가 증거이며, 7개 branch margin은 같은 방향(높은 margin이 같은 클래스)으로 정렬되어 있고, task·fold 내 rank 변환 후 median이 branch 간 scale·tail 문제를 해결하며, 이 단일 사전 등록 변형은 CA-08의 선택 편향을 피한다고 두고 있다.
- **이 가정이 깨지는 조건**: margin의 cardinality가 증거 강도·신뢰도를 담고 있으면 rank 변환이 정보를 버려 Δ가 0 이하가 된다. 어떤 branch margin이 반대 방향이거나 scale이 순서와 무관한 신호를 담으면 rank-median이 잘못된 결합을 만든다. rank-median 외에 다른 집계 함수를 Primary 7에서 실행해 좋은 값을 고르면 CA-08 선택 편향으로 실행 무효. baseline/candidate가 같은 fold·seed·provenance로 짝지어지지 않으면 실행 무효. margin 재적합이 필요하고 fold 적합 시간이 4 GPU-hour를 초과하면 예산 내 판별 불가.
- **깨질 때 나타나는 관측 가능한 증상**: paired Δ가 0을 포함하면 판별 불가, 음이면 반박. rank-median score가 seed 재적합에서 극단 raw margin보다 변동이 줄지 않으면 적합 분산 감소 실패. branch별 rank 분포가 fold·seed마다 불안정하면 rank 변환의 안정성 반박. 선택 이력·rank 변환식·seed·fold 기록이 누락되면 실행 무효.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: margin 절대 크기가 정보인지에 대해서는 해당 관측 없음. 다만 CA-08은 기존 마진 위의 결합 함수·부분집합 탐색 반복을 경계 안으로 두고 선택 편향을 기각 기전으로 삼고 있으며, D-050의 provenance 결함은 대응 비교 전제와 충돌한다.

## 이론적 병목 6
- **가정**: P1-2와 P2-2는 7개 branch margin이 이미 충분한 정보를 담고 있고, trimmed mean의 scale·tail 처리가 성능을 제한한다고 두고 있다.
- **이 가정이 깨지는 조건**: branch margin의 선형 또는 비선형 span이 AUROC에 쓸 수 있는 정보의 상한이면, 집계 변경은 기존 정보의 재가중일 뿐이다. branch 간 중복이 크고 family 내부 유효 랭크가 낮으면 집계 변경으로 새로운 정보가 생기지 않는다.
- **깨질 때 나타나는 관측 가능한 증상**: 집계 변경 후 paired Δ가 0을 포함하면 판별 불가, 음이면 반박. eff.rank가 변하지 않고 branch margin 분포만 scale이 바뀌면 재가중만. branch별 기여도 실측이 없어 어떤 branch가 scale/tail로 손해를 보는지 특정되지 못하면 기전 미확인.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: eff.rank 3.52/7, BM·QA·DS의 유효 랭크 1.29/3, 위치 계열 상호 상관 0.25~0.93, CA-10의 “형상 계열 병목이 절사가 아니라 정보량” 기록은 집계만 바꾼다고 branch 정보량이 늘지 않는다는 관측과 일치하며, “집계가 병목” 전제와 충돌한다.

## 이론적 병목 7
- **가정**: 모든 제안은 현재 계측 상태에서 Primary 7 대응 비교의 결과를 verified 승격 근거로 사용할 수 있고, Primary 7 paired Δ 개선이 SEAL 10 ABMIL 목표와 같은 방향이라고 두고 있다.
- **이 가정이 깨지는 조건**: provenance writer 복구, 러너 폴백 통제, 오염 검사 재설계가 완료되지 않으면 산출물을 구성에 귀속시킬 수 없어 실행 무효. Primary 7 paired Δ가 SEAL 10 hold-out 개선으로 전이된다는 근거가 없으면 최종 목표에 대한 결론은 hold-out 미검증으로만 남는다. SEAL 10 결과를 보고 방향을 고르면 D-047 위반.
- **깨질 때 나타나는 관측 가능한 증상**: arm이 같은 설정으로 수렴, 저장 예측에 branch list·가중치·코드 해시·fold manifest 해시 누락, 오염 검사 앵커 불일치, unverified-hold 표기가 유지되면 실행 무효 또는 verified 인용 불가. Primary 7 Δ가 +0.0030을 넘더라도 SEAL 10 개봉 전에는 최종 목표 달성을 지지한다고 쓸 수 없다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: D-050은 공식 7-branch 값을 unverified-hold로 두고, 여섯 러너가 같은 7-branch로 수렴했고, provenance가 없어 과거 산출물을 구성에 귀속시킬 수 없다고 기록한다. current_status의 현재 목표가 계측 복구이며, PROJECT.md §1은 Primary 7 개선이 hold-out 개선을 보장하지 않는다고 명시한다.

### M1 (실측)

## 실측 1: P1-1 Provenance-gated orthogonality probe
- **건드려야 하는 코드·설정**: `scripts/analysis/branch_screen.py`의 SCREEN_ONLY margin 기록과 max |r|·eff.rank 계산; `scripts/eval_v121.sh` 등 여섯 러너가 `eval_seal_tasks.sh` 폴백으로 같은 7-branch로 수렴하는 경로 제어; `evaluate_pure.py`와 `configs/baseline/v121_active.yaml`의 branch list·가중치·집계 함수; 산출물에 branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed·데이터 버전을 쓰는 provenance writer(구체적 함수·파일 경로 문서에 없음); 후보 `FC` branch 구현과 강도 집합 설정(경로·정의 문서에 없음). `src/`의 `ICF_*` 환경변수 경로는 무효이므로 YAML 경로만 사용해야 한다.
- **비용**: fold 적합 1회 시간 미실측; baseline 7-branch와 후보 1개에 대한 Primary 7 350-fold margin 재기록이 필요하면 두 arm의 fold-fit이 필요하고, seed 재적합을 추가하면 seed 수만큼 배증되며, 총 GPU-hour·메모리·연산은 미실측이다. 저장된 예측 파일은 provenance가 없어 재사용할 수 없다. 4 GPU-hour 내 완료 여부 미실측.
- **막는 것**: D-050 계측 복구(provenance writer, 러너 폴백 통제, 오염 검사 재설계, 단위 테스트)가 완료되지 않아 SCREEN_ONLY 게이트 ①을 verified로 수행할 수 없다. 러너가 폴백으로 수렴하면 후보 arm을 공식 7-branch와 구별할 수 없어 실행 무효다. 후보 `FC`의 정의·강도 집합·코드 경로가 사전 등록되지 않으면 구현할 대상이 없다. 오염 검사 재설계 전 통과를 주장하는 실행은 무효다.
- **중단 기준**: 산출물에 provenance 필드가 없거나 러너 폴백으로 arm이 구별되지 않으면 즉시 실행 무효로 중단한다. 1 fold 1 seed 시간 실측 후, 이 제안에 필요한 fold-fit 총량이 4 GPU-hour 또는 단일 M 2시간을 초과하면 중단한다. NaN/Inf, OOM, 입력 누락, responsibility 합 오차 1e-4 초과, sklearn 대비 component 불일치 시 중단한다. Primary 7 성능을 보고 후보를 고르면 선택 편향으로 실행 무효다.

## 실측 2: P1-2 PSW predictive-variance trimmed mean
- **건드려야 하는 코드·설정**: closed-form ridge의 예측분산 `v_b` 산출 경로(문서에 없음); branch margin과 `v_b`를 fold·과제별로 저장하는 provenance writer(구체적 경로 문서에 없음); `evaluate_pure.py` 또는 집계 모듈(구체적 경로 문서에 없음)에서 `z_b = m_b / sqrt(v_b)` 변환과 기존 trimmed mean 대체; `v_b`가 0·비유한일 처리 규칙, seed 정책, 후보 등록 시점을 고정하는 설정; baseline과 후보를 같은 fold·seed·provenance로 짝짓는 평가 경로.
- **비용**: `v_b`가 저장된 ridge 해에서 오프라인으로 계산 가능하면 fold-fit 없이 가능하지만, 저장 ridge 해의 존재·형식·재사용 가능 여부는 미실측이다. 재적합이 필요하면 Primary 7 350-fold에 baseline·후보 두 arm, seed 재적합 시 seed 수만큼 배증되며, 총 GPU-hour·메모리·연산은 미실측이다. 4 GPU-hour 내 완료 여부 미실측.
- **막는 것**: 현재 문서에 `v_b` 산출 경로가 없어 구현 전 비용과 동작을 확정할 수 없다. ridge 해의 정확한 형태(λ, 정규화, context/query 분리, dual ridge 여부)를 확인하지 않으면 `v_b`가 모델의 실제 예측분산인지 미확인이다. provenance 부재로 baseline·후보 arm을 구성에 귀속시킬 수 없다. `v_b`가 실제 fold 오차를 예측한다는 실측이 없어 기전 미확인이다. Primary 7에서 `v_b` 변환을 보고 변형을 고르면 CA-08 선택 편향으로 실행 무효가 된다.
- **중단 기준**: `v_b`를 현재 closed-form ridge에서 모델 의미를 바꾸지 않고 계산할 수 없으면 중단한다. `v_b`의 0·비유한 처리가 사전 등록되지 않으면 중단한다. 1 fold 1 seed 시간 실측 후, 필요한 fold-fit 총량이 4 GPU-hour 또는 단일 M 2시간을 초과하면 중단한다. NaN/Inf, OOM, 입력 누락, responsibility 합 오차 1e-4 초과, sklearn 대비 component 불일치 시 중단한다.

## 실측 3: P1-3 Seed-averaged paired Δ estimand
- **건드려야 하는 코드·설정**: seed를 받는 러너 경로(`scripts/eval_v121.sh` 등 여섯 러너와 `eval_seal_tasks.sh` 폴백 제어 포함); seed·fold별 AUROC를 저장하는 provenance writer(구체적 경로 문서에 없음); baseline과 후보를 같은 task·fold·seed 인덱스로 짝짓는 평가 스크립트(문서에 없음); seed 평균 fold AUROC에서 350-fold 대응 Δ를 계산하는 집계 스크립트(문서에 없음); K와 seed 집합을 성능 확인 전에 고정하는 사전 등록 파일.
- **비용**: K seed × baseline/candidate 두 arm × Primary 7 350-fold fold-fit이 필요하며, 1 fold 1 seed 시간 미실측이다. 총 GPU-hour·메모리·연산은 미실측이다. K=1이면 기존 단일 적합과 동일하므로 판정 정밀도 개선이 없다. 잠정 R=8을 쓰면 seed 수만큼 fold-fit이 증가하지만 실제 시간 미실측이다.
- **막는 것**: 적합 분산을 포함한 승격 estimand 변경은 현재 사용자 승인 전이다. fold 적합 1회 시간이 미실측이라 4 GPU-hour 내 가능한 K를 사전 등록할 수 없다. provenance 부재로 seed·fold별 AUROC를 특정 구성에 귀속시킬 수 없다. seed 간 독립성과 fold×seed 잔차 구조가 미확인이라 seed 평균이 실제로 분산을 줄이는지 기전 미확인이다. best-seed 선택을 금지하는 통제가 없으면 실행 무효다.
- **중단 기준**: 1 fold 1 seed 시간 실측 후, 사전 등록한 K에 필요한 fold-fit 총량이 4 GPU-hour 또는 단일 M 2시간을 초과하면 중단한다. K=1만 가능하면 estimand 변경 효과가 없어 중단한다. baseline과 후보의 seed pairing을 만들 수 없으면 중단한다. K·seed 집합을 Primary 7 결과를 보고 바꾸면 선택 편향으로 실행 무효다. NaN/Inf, OOM, 입력 누락, responsibility 합 오차 1e-4 초과, sklearn 대비 component 불일치 시 중단한다.

## 실측 4: P2-1 유효 랭크 3.52를 family span 포화 진단으로 재해석
- **건드려야 하는 코드·설정**: provenance와 함께 Primary 7 slide-level branch margin을 저장하는 writer(구체적 경로 문서에 없음); family별 covariance spectrum·max R²·leave-one-out effective rank를 계산하는 분석 스크립트(문서에 없음; `scripts/analysis/branch_screen.py`는 max |r|·eff.rank만 문서화); family grouping(`CV`, `BM·QA·DS`, `BD·SH·SJ` 등)을 사전 등록하는 설정; probe branch는 G/H 사용자 판정을 통과한 것만 포함하되 현재 G·H는 모두 판정 대기이므로 포함 대상 없음.
- **비용**: provenance가 있는 margin이 이미 있으면 오프라인 CPU 분석이지만 소요 시간 미실측이다. 현재 저장 예측은 provenance가 없어 재사용 불가하므로 350-fold margin 재적합이 필요할 수 있고, 이 경우 fold 적합 1회 시간 미실측, 총 GPU-hour·메모리·연산 미실측이다. probe를 넣으면 추가 fold-fit이 필요하지만 probe가 현재 허용되지 않아 비용 미실측이다.
- **막는 것**: provenance 부재로 margin을 공식 7-branch 구성에 귀속시킬 수 없다. G/H 하위축이 사용자 판정 대기이므로 AKS·MDX·LID 계열 probe를 넣을 수 없다. probe 없이 “branch 추가 방향이 막혔다”와 “서로 다른 것을 재는 branch를 아직 못 만들었다”를 완전히 가를 수 없다. margin 간 선형 중복만 재면 비선형 정보 중복은 미확인이다. family grouping과 R² 기준을 사전 등록하지 않으면 탐색이 선택 편향으로 바뀔 수 있다.
- **중단 기준**: margin을 특정 7-branch 구성에 귀속시킬 수 없으면 실행 무효로 중단한다. 1 fold 1 seed 시간 실측 후, 재적합 총량이 4 GPU-hour 또는 단일 M 2시간을 초과하면 중단한다. probe를 넣을 수 없으면서 probe가 필요한 판별이라면 판별 불가로 중단한다. 비선형 중복이 미확인인 상태에서 “branch 추가가 열려 있다”고 결론짓지 않는다. NaN/Inf, OOM, 입력 누락, responsibility 합 오차 1e-4 초과, sklearn 대비 component 불일치 시 중단한다.

## 실측 5: P2-2 trimmed mean을 rank-median으로 바꾸는 단일 집계 후보
- **건드려야 하는 코드·설정**: 집계 모듈(구체적 경로 문서에 없음)에서 task·fold 내부 branch margin rank 변환, ties 평균 순위, 7개 rank의 median slide score 계산; baseline trimmed mean과 candidate rank-median을 같은 fold·seed·provenance로 짝짓는 평가 경로; `evaluate_pure.py` 진입점과 `configs/baseline/v121_active.yaml`의 집계 함수 설정; rank-median 하나만 사전 등록하는 후보 등록 파일; provenance writer.
- **비용**: provenance가 있는 margin을 재사용하면 오프라인 재집계지만 소요 시간 미실측이다. 현재 저장 예측은 provenance가 없어 재사용 불가하므로 350-fold margin 재적합이 필요할 수 있고, fold 적합 1회 시간 미실측, 총 GPU-hour·메모리·연산 미실측이다. D-048 이후 적합 분산 보고가 필요하면 seed 재적합이 추가되어 seed 수만큼 배증된다.
- **막는 것**: provenance 부재로 baseline과 candidate를 같은 구성·fold·seed로 짝지을 수 없다. 여섯 러너의 폴백 수렴 문제가 해결되지 않으면 candidate가 실제로 rank-median인지 확인 불가다. rank-median 하나만 사전 등록하지 않고 여러 집계 함수를 돌린 뒤 Primary 7에서 좋은 값을 고르면 CA-08 선택 편향으로 실행 무효다. branch별 기여도가 미측정이므로 rank 변환이 증거 강도 정보를 보존하는지 기전 미확인이다. 적합 분산 보고가 필요하면 비용이 증가한다.
- **중단 기준**: baseline과 candidate가 같은 fold·seed·provenance로 짝지어지지 않으면 실행 무효로 중단한다. ties 처리와 rank 변환 규칙이 사전 등록되지 않으면 중단한다. 1 fold 1 seed 시간 실측 후, 재적합 총량이 4 GPU-hour 또는 단일 M 2시간을 초과하면 중단한다. rank-median 외 집계 함수를 실행해 Primary 7에서 선택하면 실행 무효다. NaN/Inf, OOM, 입력 누락, responsibility 합 오차 1e-4 초과, sklearn 대비 component 불일치 시 중단한다.

## 실측 6: P2-3 승격 estimand를 seed 평균 paired Δ로 변경
- **건드려야 하는 코드·설정**: P1-3과 동일한 seed·fold pairing 러너, seed·fold별 AUROC 저장, seed 평균 paired Δ 집계 스크립트, provenance writer, K·seed 집합 사전 등록 파일. P2-3은 P1-3과 별도로 추가 구현 이점이 문서에 없다.
- **비용**: K seed × baseline/candidate 두 arm × Primary 7 350-fold fold-fit이 필요하며, 1 fold 1 seed 시간 미실측이다. 총 GPU-hour·메모리·연산은 미실측이다. K=1이면 기존 단일 적합과 동일하다. seed 재적합을 위한 GPU 사용량도 미실측이다.
- **막는 것**: estimand 변경은 사용자 승인 전이다. fold 적합 1회 시간이 미실측이라 4 GPU-hour 내 가능한 K를 사전 등록할 수 없다. provenance 부재로 seed·fold별 결과를 구성에 귀속시킬 수 없다. seed 변동이 독립 noise인지, fold 순서·데이터 구조에 따른 불안정성인지 미확인이다. RU-97에서 fold×seed 잔차가 seed 주효과보다 컸다는 관측은 seed 평균이 실제로 분산을 줄인다는 기전의 입증이 아니다.
- **중단 기준**: 1 fold 1 seed 시간 실측 후, 사전 등록한 K에 필요한 fold-fit 총량이 4 GPU-hour 또는 단일 M 2시간을 초과하면 중단한다. K=1만 가능하면 중단한다. baseline과 후보의 seed pairing이 없으면 중단한다. K·seed 집합을 Primary 7 결과를 보고 바꾸면 실행 무효다. NaN/Inf, OOM, 입력 누락, responsibility 합 오차 1e-4 초과, sklearn 대비 component 불일치 시 중단한다.

## 실측 7: 4 GPU-hour 예산 내 판별 가능 후보
- **건드려야 하는 코드·설정**: 먼저 1 fold 1 seed fold-fit 시간을 provenance와 함께 실측하는 최소 실행; 각 제안에 필요한 fold-fit 수를 계산하는 예산 표; 모델 선택이나 성능 조회를 하지 않는 측정 스크립트.
- **비용**: 1 fold 1 seed 시간 미실측이므로, P1-1·P1-2·P1-3·P2-1·P2-2·P2-3 모두 4 GPU-hour 내 판별 가능 여부는 미실측이다. 1 fold 측정 자체의 GPU-hour·메모리도 미실측이다.
- **막는 것**: fold 적합 1회 시간이 측정되지 않아 어떤 제안도 예산 내 완료를 보장할 수 없다. provenance 부재로 과거 margin·예측을 재사용할 수 없어 대부분 재적합이 전제된다. 현재 목표가 계측 복구이므로, 복구 전에는 게이트·승격 비교를 verified로 수행할 수 없다. GPU 1장 합계 4 GPU-hour 예산은 단일 실험에 집중해야 하므로 여러 제안을 동시에 판별할 수 없다.
- **중단 기준**: 1 fold 1 seed 시간 실측 후, 각 제안의 필요 fold-fit 총량이 4 GPU-hour 또는 단일 M 2시간을 초과하면 해당 제안을 판별 불가 또는 실행 무효로 표기하고 중단한다. 어떤 제안도 예산에 들어가지 않으면 측정 결과만 기록하고 중단한다. 1 fold 시간 미실측 상태에서는 “4 GPU-hour 내 판별 가능”이라고 주장하지 않는다.

### R1 (반증)

## 반론 1
- **공격 대상**: P1-1과 P2-1의 “게이트 ①(max |r|, eff.rank) 또는 family span 진단이 ‘branch 추가가 막혔다’와 ‘서로 다른 것을 아직 못 만들었다’를 가른다”는 주장.
- **더 단순한 설명**: eff.rank 3.52/7은 7개 margin이 같은 patch feature와 closed-form ridge에서 나왔기 때문에 선형 span이 낮다는 것일 뿐, label-predictive 정보의 상한이 낮다는 증거는 아니다. 더 단순한 설명은 “새 branch가 기존 7개와 선형 상관이 낮아도 AUROC에 쓸모없는 방향일 수 있다”이다. 게이트 ①은 라벨 무관 직교성만 재므로, 통과가 정보량 부족을 해소했음을 보이지 않는다.
- **무엇이 참이면 이 주장이 무너지는가**: (a) 기존 7개 margin이 이미 이 feature에서 label을 예측하는 유효 방향을 대부분 span하고, 새 branch는 선형 독립이더라도 label과 무관한 방향만 추가한다면 게이트 ① 통과는 branch 추가가 열렸다는 증거가 아니다. (b) 반대로 새 branch가 기존 span과 선형 중복이지만 비선형적으로 label 정보를 가진다면 게이트 ① 기각은 branch 추가가 막혔다는 증거가 아니다. (c) family spectrum이 family 내부 중복만 보여, family 밖 정보 부재와 family 내 중복을 구분하지 못하면 P2-1의 판별 관측이 성립하지 않는다.
- **이 반론의 한계**: 게이트 ①이 성능을 보장하지는 못하지만, 선형 중복을 제거하는 필요 조건으로 기능한다면, 게이트 ① 통과 후보는 성능 조회 전에 살릴 가치가 있다. 또한 새 branch가 실제로 기존 margin과 비선형 독립이고 label 정보량이 있다면 이 반론은 지나치게 보수적이다.

## 반론 2
- **공격 대상**: P1-2 PSW의 “closed-form ridge 예측분산으로 margin을 표준화하면 branch 간 commensurability가 개선되고 AUROC가 오를 수 있다”는 기전.
- **더 단순한 설명**: 관측된 branch 중복과 seed 변동이 branch margin의 scale 차이에서 온 것이 아니라, 같은 patch feature에서 나온 margin들이 공통 정보를 공유하는 것에서 온 것이라면, scale 표준화는 새 정보를 만들지 못한다. `v_b`는 ridge 해의 조건부 분산이지 fold AUROC 오차의 캘리브레이션된 추정치가 아니므로, `m_b / sqrt(v_b)`는 단순한 재가중일 수 있다. 문서에는 `v_b`가 실제 fold 오차를 예측한다는 실측이 없다.
- **무엇이 참이면 이 주장이 무너지는가**: (a) `v_b`가 실제 fold 오차와 무관하거나, context 크기·ridge 조건부 특이성·구현 노이즈 같은 margin과 상관된 요인을 반영하면 표준화가 신호를 왜곡한다. (b) 고분산 branch가 극값·이질성 같은 진짜 신호를 담고 있으면, 분산으로 나누는 것이 신호를 죽인다. (c) `v_b` 산출 경로가 없거나 fold별 `v_b`가 seed마다 달라 paired 비교가 동일 fold·seed·provenance로 짝지어지지 않으면 실행 무효.
- **이 반론의 한계**: `v_b`가 fold 오차를 실제로 예측하고, branch별 오차의 이질성이 AUROC를 제한하는 주된 원인이라면 PSW는 유효한 집계 축이다. 이 경우 “기전 미확인”이라는 반론은 후보를 기각하는 근거가 아니라 사전 검증 조건일 뿐이다.

## 반론 3
- **공격 대상**: P2-2 rank-median의 “rank 변환이 branch별 scale·극단값 영향을 제거해 trimmed mean보다 낫거나 fit variance를 줄인다”는 주장.
- **더 단순한 설명**: AUROC는 slide score의 단조 변환에 불변이므로, rank-median이 최종 slide 순서를 trimmed mean과 크게 바꾸지 못하면 Δ는 0이다. trimmed mean은 이미 최저·최고 1개를 절사하므로, 7개 branch에서 극단 1개 수준의 outlier 영향은 상당 부분 제거되어 있다. rank 변환이 추가하는 이득은 branch margin의 절대 강도 정보가 순서 정보보다 중요하지 않을 때만 생긴다.
- **무엇이 참이면 이 주장이 무너지는가**: (a) branch margin의 크기(증거 강도)가 slide 순서보다 정보라면 rank 변환은 정보를 버린다. (b) 7개 branch의 slide 순위가 fold·seed 간에 안정적이라면 rank-median과 trimmed mean의 slide score 순위가 거의 같아 Δ가 승격 임계 +0.0030보다 작아 판별 불가가 된다. (c) ties 처리가 slide score를 지배하면 rank-median의 변동은 branch 정보가 아니라 tie 규칙의 산물이 된다.
- **이 반론의 한계**: branch margin의 scale이 task·fold마다 달라 raw trimmed mean이 일부 branch에 편향된 가중을 주고, 그 편향이 AUROC를 제한한다면 rank-median은 유효한 단일 집계 후보가 된다. 이 경우 “AUROC 불변”은 최종 score의 불변이지 branch 결합 단계의 불변이 아니므로 반론이 약해진다.

## 반론 4
- **공격 대상**: P1-3/P2-3 seed-averaged paired Δ estimand의 “seed 평균이 적합 분산을 줄여 +0.0030 근처의 Δ를 판별 가능하게 한다”는 주장.
- **더 단순한 설명**: seed std 0.0021~0.0083은 동일 구성의 절대 macro 변동이다. 승격 estimand는 paired per-fold Δ이며, baseline과 candidate를 같은 fold·seed에서 짝지으면 공통 fold 난이도와 seed 주효과가 Δ에서 상당 부분 상쇄될 수 있다. 따라서 same-config seed std가 크다고 해서 paired Δ의 seed 변동이 반드시 크다는 결론은 아니다.
- **무엇이 참이면 이 주장이 무너지는가**: (a) seed 변동이 후보와 기준에 공통이 아니라 모델별·구현별로 다르게 붙는다면 seed 평균이 paired Δ의 불확실성을 줄일 수 있다. (b) fold×seed 잔차가 seed 주효과보다 크면 seed 평균으로 줄지 않는다. RU-97의 GF 구성에서는 fold×seed 잔차가 23.9%로 seed 주효과보다 크다는 실측이 있으나, 이는 공식 7-branch 전체의 분산 구조를 보장하지 않는다. (c) fold 적합 1회 시간이 문서에 없으므로 4 GPU-hour 안에서 K>1을 사전 등록할 수 없고, K=1이면 기존 단일 적합 estimand와 동일하다. (d) 적합 분산을 포함한 승격 판정 절차가 미승인 상태라면 seed 평균 estimand는 운영 규칙 변경이지 분석 장치에 그친다.
- **이 반론의 한계**: seed 효과가 독립적이고 fold×seed 잔차가 작으며, 4 GPU-hour 안에 K≥2 이상을 실행할 수 있다면 seed 평균은 paired Δ의 SE를 줄이는 유효한 방법이다. 또한 D-048 이후 적합 분산 보고가 의무이므로, seed 평균은 승격 판정의 필수 구성요소가 될 수 있다.

## 반론 5
- **공격 대상**: P2-1의 “family span 포화 진단이 branch 추가 방향의 병목을 family 내부 중복과 family 밖 정보 부재로 분리한다”는 주장.
- **더 단순한 설명**: eff.rank는 margin 공분산의 선형 중복을 재지, branch별 AUROC 기여도를 재지 않는다. 3.52/7은 한두 family의 강한 방향이 covariance를 지배하는 것만으로 나올 수 있다. branch별 기여도는 문서에 없으므로, family spectrum만으로는 “어떤 family가 성능을 제한하는가”를 구분하지 못한다.
- **무엇이 참이면 이 주장이 무너지는가**: (a) family 내부 eff.rank가 낮아도 그 family가 label 정보를 대부분 담고 있고, family 밖 probe는 label 무관한 방향만 추가한다면 family span 진단은 branch 추가 가능을 과대 해석한다. (b) CA-10 경계 밖 probe(AKS·MDX·LID 등)가 RU-85의 등방성(`H_q > 0.99`)이나 LID 추정기 불안정(`|r| = 0.314`) 때문에 정보량이 없다면, probe 실패는 “branch 추가가 막혔다”가 아니라 “그 probe가 무용하다”는 증거다. (c) 선형 span이 포화되어도 비선형 결합에서 정보가 남을 수 있으므로, 선형 spectrum만으로 branch 추가 축을 닫거나 열 수 없다.
- **이 반론의 한계**: family별 covariance spectrum이 실제로 중복의 위치를 정확히 보여주고, probe가 기존 span에서 낮은 R²를 보이며 게이트 ①을 통과한다면, 이 진단은 “아직 못 만들었다” 쪽에 가중하는 유효한 관측을 만든다. 또한 branch별 기여도를 재지 않더라도, 선형 중복 제거는 새 branch 채택의 필요 조건으로 쓸 수 있다.

## 반론 6
- **공격 대상**: P1-1의 “provenance writer 복구 + 폴백 통제를 먼저 하면 4 GPU-hour 안에 후보 branch의 게이트 ① 판별이 가능하다”는 암묵적 예산 전제.
- **더 단순한 설명**: 과거 여섯 러너가 같은 7-branch로 수렴한 것은 학습 설정이 폴백에 걸린 실행 경로 문제다. 따라서 가장 단순한 설명은 “YAML 경로를 직접 사용하고 산출물에 branch list·가중치·코드 해시를 기록하는 최소 통제”로 충분하다는 것이다. 전체 provenance writer와 fold manifest 해시, 데이터 버전 기록까지 복원해야 후보 판별이 가능하다는 주장은 과잉일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: (a) fold 적합 1회 시간이 문서에 없으므로, baseline 7-branch와 후보 1개의 350-fold margin을 새로 만들어야 한다면 4 GPU-hour 초과가 가능하고, 이 제안은 예산 내 판별 불가다. (b) 저장된 margin을 재사용하려면 provenance가 필요하지만, 과거 산출물은 구성 귀속이 불가하므로 재사용이 무효다. (c) 최소 통제만으로는 향후 실행 무효를 막을 수 없다면, 게이트 ① 결과가 지지/반박으로 기록될 수 없다.
- **이 반론의 한계**: fold 적합 시간이 짧아 4 GPU-hour 안에 baseline+후보 margin을 만들고, 최소 provenance만으로도 arm을 구성에 귀속시킬 수 있다면, 이 제안은 예산 내 판별 가능한 사전 게이트가 된다. 또한 D-050 이후 provenance 기반 오염 검사 재설계가 필수이므로, writer 복구는 연구 축이 아니라 유효한 실행의 전제 조건이다.

## 반론 7
- **공격 대상**: P1-2/P2-2의 “집계(trimmed mean)를 바꾸는 축은 아직 거의 탐색되지 않았다”는 전제.
- **더 단순한 설명**: CA-08이 “기존 브랜치 확률·마진 위의 결합 함수·부분집합 탐색”을 선택 편향으로 닫았다. trimmed mean도 이미 최저·최고 1개 절사라는 robust aggregation이다. PSW/rank-median은 새 정보나 제한된 단일 후보가 아니라, 기존 margin의 변환을 탐색하는 것일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: CA-08 경계 밖이 “새 정보(예측분산·Context 신호 기하)를 사용하는 PSW·TGW”와 “탐색임을 명시한 제한된 결합 연구”를 허용하므로, 사전 등록 단일 후보는 저촉이 아니다. 따라서 “아직 거의 탐색되지 않았다”는 운영상 맞을 수 있다. rank-median이 Primary 7 결과를 보고 고르지 않고 단일 후보로 사전 등록된다면, CA-08의 선택 편향 실패 양식과 구분될 수 있다.
- **이 반론의 한계**: CA-08의 폐쇄는 결합 함수 일반의 무효가 아니라 선택 편향에 근거한다. 따라서 사전 등록 단일 집계 후보는 여전히 유효한 탐색 대상이 될 수 있다. 이 반론이 유효한 경우는 PSW의 `v_b`가 새 정보가 아니라 기존 margin의 재가중일 때, 또는 rank-median이 여러 집계 함수 탐색으로 확장될 때다.

### A1 (감사)

## 점검 1: provenance writer와 러너 폴백 통제
- **확인하려는 것**: P1-1·P2-1·P2-2의 기준/후보 margin이 branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed·데이터 버전으로 귀속되고, 러너 폴백으로 7-branch로 수렴한 실행이 `실행 무효`로 잡히는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: `D-050`처럼 arm이 다른 실행이 같은 7-branch로 수렴하면 후보/기준 비교가 `실행 무효`가 된다. 저장 margin을 특정 구성에 귀속 못 해 gate ①·승격 Δ가 무효다.
- **확인 방법**: `scripts/eval_v121.sh`, `eval_seal_tasks.sh`, `src/` config 파싱, provenance writer, 저장 예측 메타데이터, `tests/test_docs_consistency.py`, `talks/reports/2026-09-18_ru90_baseline_regen.md`를 대조한다. 합성 설정에서 branch weight를 바꿔 출력 메타데이터와 margin이 달라지는지 확인한다.

## 점검 2: baseline의 verified 상태
- **확인하려는 것**: P1-2·P2-2·P1-3·P2-3의 paired Δ가 verified 7-branch baseline과 같은 fold manifest에서 짝지어지는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: `0.6227`은 `unverified-hold`다. baseline provenance가 없으면 `+0.0030` 판정의 기준이 불명확해 승격 비교가 `실행 무효`가 된다.
- **확인 방법**: 공식 7-branch YAML을 사용해 baseline을 재실행하고, branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed·데이터 버전을 기록한다. 기존 `0.6227`을 verified 기준으로 인용하지 않는지 확인한다.

## 점검 3: SCREEN_ONLY 오염 검사의 provenance 기반 검증
- **확인하려는 것**: P1-1의 `SCREEN_ONLY=1` 후보 margin이 앙상블에 영향을 주지 않는지 provenance 기반으로 검증할 수 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 값 앵커 오염 검사는 폐지·재설계 중이다. 재설계 전 오염 검사 통과를 주장하면 gate ① 결과가 무효다. SCREEN_ONLY가 baseline을 바꾸면 후보 margin이 오염된다.
- **확인 방법**: `scripts/analysis/branch_screen.py`, `SCREEN_ONLY=1` 경로, baseline 출력 provenance, 오염 검사 재설계 상태를 확인한다. 소수 4자리 AUROC 일치 대신 branch list·가중치·코드 해시·fold manifest 해시를 대조한다.

## 점검 4: FC 후보의 closed-axis 경계
- **확인하려는 것**: P1-1의 예시 후보 `FC`가 `CA-09` 경계 밖인가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: `CA-09` 경계 안이면 성능과 무관하게 기각해야 한다. 경계 안 후보를 gate ①·성능 조회에 쓰면 `실행 무효`다.
- **확인 방법**: `docs/closed_axes.md` `CA-09`의 경계 밖 문구와 후보 정의·강도 집합을 대조한다. `FC`가 “여러 초점 강도의 slide mean 차분”으로 고정되었는지 RU 카드에서 확인한다.

## 점검 5: G/H probe의 사용자 판정 전제
- **확인하려는 것**: P2-1의 probe가 `CA-10` 경계 밖 각도·모드·국소 내재차원 계열 중 사용자 판정 `G`/`H`를 통과한 것만 `SCREEN_ONLY`로 들어가는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: `G`/`H`는 사용자 판정 대기다. 통과 확인 없이 probe를 넣으면 경계 미확정 축을 임의로 착수하는 것이 되어 gate ① 결과가 `실행 무효`가 될 수 있다.
- **확인 방법**: `docs/closed_axes.md` §3 `G`·`H`, 사용자 결정 기록, RU 카드의 probe 포함 조건을 대조한다.

## 점검 6: 후보·변형·K의 사전 등록과 선택 이력
- **확인하려는 것**: P1-1 후보/강도, P1-2 `v_b` 처리 규칙, P2-2 rank-median, P1-3/P2-3 seed 수·seed 집합이 Primary 7 성능 조회 전에 고정되고 선택 이력이 기록되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: Primary 7을 보고 후보·변형·K를 고르면 §214식 선택 편향이다. 성능 확증이 `실행 무효` 또는 `반박`으로 전락한다.
- **확인 방법**: RU 카드, git commit/config hash, 선택 로그, Primary 7 결과 접근 로그를 확인한다. 제안서만으로는 사전 등록으로 인정할 수 없다.

## 점검 7: gate ① 지표로 두 해석을 가르는 유효성
- **확인하려는 것**: P1-1·P2-1이 `max |r| ≤ 0.6`, `eff.rank ≥ 4.02`, family spectrum으로 “서로 다른 것을 재는 branch를 못 만들었다”와 “branch 추가 방향이 막혔다”를 실제로 가르는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: `max |r|`과 `eff.rank`는 선형 중복 지표다. 통과를 AUROC 정보량 확보나 비선형 중복 부재로 해석하면 증거 범위를 넘는다. gate 통과를 “branch 추가 가능”으로 확정하면 `판별 불가`를 `지지`로 바꾸는 오류다.
- **확인 방법**: `branch_screen.py` 출력, covariance spectrum, multivariate `R²`, non-linear dependence 진단을 확인한다. 문서에는 임계값만 있고, 선형 rank가 정보량을 입증한다는 근거는 없다.

## 점검 8: PSW `v_b`의 fit 범위와 label leakage
- **확인하려는 것**: P1-2의 `v_b`가 fold context와 candidate margin에서만 계산되고 query/test label, SEAL 10, cross-fold Primary 7 aggregate를 쓰지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: `v_b`에 query label이나 test fold 정보가 들어가면 AUROC leakage다. fold별 `v_b`가 fold difficulty에 fit되면 paired Δ가 편향된다.
- **확인 방법**: ridge predictive variance 구현, data loader split, config, unit test를 확인한다. query label을 바꿔도 `v_b`와 slide score가 불변인지 확인한다.

## 점검 9: PSW의 `CA-08` 경계
- **확인하려는 것**: P1-2가 `CA-08` 경계 밖의 `PSW`에 해당하는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: `CA-08` 경계 안이면 기존 margin 결합 탐색 반복으로 `실행 무효`다.
- **확인 방법**: `docs/closed_axes.md` `CA-08`의 경계 밖 문구에서 `PSW`가 명시적으로 비저촉으로 되어 있는지 확인한다.

## 점검 10: PSW 처리 규칙의 사전 고정
- **확인하려는 것**: P1-2의 `v_b` zero/infinite 처리, seed 정책, 후보 등록 시점이 Primary 7 성능 확인 전에 고정되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: `v_b`가 비유한 fold를 결과 보고 후 제외하거나 임계값을 바꾸면 선택 편향이다. 성능 확증이 `실행 무효`가 된다.
- **확인 방법**: RU 카드, config hash, 코드 경로, Primary 7 접근 로그를 확인한다.

## 점검 11: rank-median의 `CA-08` 경계 판정
- **확인하려는 것**: P2-2 rank-median이 `CA-08` 경계 밖의 제한된 결합 연구인가, 아니면 기존 margin 결합 함수 탐색 반복인가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: rank-median 외에 다른 집계 함수를 돌려 Primary 7에서 좋은 값을 고르면 `CA-08`의 선택 편향을 반복한다. 단일 후보라도 선택 이력이 없으면 성능 확증으로 쓸 수 없다.
- **확인 방법**: `docs/closed_axes.md` `CA-08`, RU 카드, config, 선택 로그를 확인한다. rank-median 하나만 사전 등록했는지, 다른 집계 함수를 실행하지 않았는지 확인한다.

## 점검 12: rank-median의 rank 계산 범위
- **확인하려는 것**: P2-2의 rank/empirical CDF가 task·fold 내부 margin만으로 계산되고 label을 쓰지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: rank 계산에 label, cross-fold 정보, SEAL 10 정보가 들어가면 leakage다. tie 처리가 seed마다 달라지면 paired Δ가 `실행 무효`가 된다.
- **확인 방법**: rank 변환 코드, unit test, synthetic margin으로 label 불변성을 확인한다. 같은 task·fold 내 slide margin만 입력으로 쓰는지 확인한다.

## 점검 13: seed average estimand의 승격 사용 승인
- **확인하려는 것**: P1-3/P2-3가 seed average paired Δ를 승격 판정에 바로 쓰려는 것이 현행 승격 절차와 모순되지 않는가.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 미승인이라고 한다. 미승인 estimand로 승격을 주장하면 `실행 무효`다. `D-048`의 적합 분산 보고 의무도 충족되지 않는다.
- **확인 방법**: `docs/current_status.md` Open Issues, `docs/PROJECT.md` §4, 사용자 승인 기록을 확인한다.

## 점검 14: seed 재적합의 fold manifest 고정과 독립성
- **확인하려는 것**: P1-3/P2-3의 seed 재적합이 fold manifest를 고정하고 seed만 바꾸며, baseline/candidate가 같은 seed·fold pairing을 갖는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: seed가 fold 순서·context 구성을 바꾸면 seed 평균이 독립 draw 평균이 아니다. `RU-97`의 fold×seed 잔차 `23.9%`처럼 seed 평균으로 분산이 줄지 않아 `판별 불가`가 남는다.
- **확인 방법**: seed 정책, fold manifest hash, job script, variance decomposition을 확인한다. baseline/candidate의 seed pairing이 같은지 확인한다.

## 점검 15: 4 GPU-hour 내 판별 가능성
- **확인하려는 것**: P1-1·P2-1·P1-2·P2-2·P1-3·P2-3가 4 GPU-hour 안에서 완료되어 `지지 / 반박 / 판별 불가`를 낼 수 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: fold 적합 1회 시간이 측정되지 않았다. 예산 초과 시 실행이 중단되고 `실행 무효` 또는 `판별 불가`만 남는다. “판별 가능”을 주장하면 근거가 없다.
- **확인 방법**: baseline/candidate 1 fold 1 seed 실행 시간을 실측하고, 필요한 K·fold 수·GPU-hour를 사전 계산한다.

## 점검 16: family span diagnostic의 사양 고정
- **확인하려는 것**: P2-1의 family grouping, leave-one-out 구성, `R²` 대상, `eff.rank` 집계 단위가 사전 고정되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: family를 결과 보고 후 나누면 선택 편향이다. `R²` 대상 span이 불명확하면 “기존 span에서 벗어났다”는 판정이 재현 불가능하다.
- **확인 방법**: RU 카드, 분석 코드, config, margin 저장 형식을 확인한다. family 정의와 `R²` 계산 대상이 Primary 7 성능 조회 전에 고정되었는지 확인한다.

## 점검 17: SEAL 10 선택 금지의 실행 검증
- **확인하려는 것**: P1·P2 실행에서 SEAL 10 결과가 후보·집계 함수·K·seed 선택에 사용되지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: SEAL 10을 보고 후보를 고르면 `D-047` 위반이다. 모든 Primary 7 기반 판정이 `실행 무효`가 된다.
- **확인 방법**: SEAL 10 결과 파일 접근 로그, job script, council card, RU 카드, 선택 로그를 확인한다.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

- **T1-1**: (1) 수용하고 수정 — P1-1은 provenance writer·러너 폴백 통제가 복구되고 1-fold 1-seed 시간이 실측된 후에만 유효하다. `branch_screen.py`의 max |r|·eff.rank는 라벨 무관 선형 중복만 재므로, FC 1개 probe가 통과해도 branch 추가 방향 전체가 열렸거나 gate ②·승격이 보장된 것이 아니다. D-050의 여섯 러너 폴백 수렴, 저장 산출물 provenance 부재, `closed_axes.md` §3-I의 SCREEN_ONLY 무효 기록이 이 수정을 강제한다.

- **T1-2**: (1) 수용하고 수정 — P1-2는 `v_b`가 fold 오차를 실제로 예측한다는 실측이 문서에 없으므로 성능 개선 제안이 아니라, `v_b` 산출 경로·유한성 처리·leakage 부재 확인이 선행되는 구현 검증안으로 낮춘다. `CA-08`도 예측분산의 유효성을 남은 검증으로 두고 있다. `v_b` 경로가 없거나 query/test label을 쓰면 `실행 무효`다.

- **T1-3**: (1) 수용하고 수정 — P1-3/P2-3은 seed 재적합 변동이 독립 적합 노이즈이고 baseline·candidate 간 seed 효과가 공통 additive라는 전제를 입증하지 못했다. 동일 구성의 seed std `0.0021~0.0083`은 절대 macro 변동이지 paired per-fold Δ의 seed 변동이 아니다. RU-97의 `m32_raw` fold×seed 잔차 23.9%는 seed 평균이 실제로 Δ 불확실성을 줄일 수 없다는 근거도, 줄인다는 근거도 아니다. current_status의 “적합 분산을 포함한 승격 판정 절차 미승인” 때문에 승격 estimand 변경이 아니라 미승인 분석안으로만 둔다.

- **T1-4**: (1) 수용하고 수정 — P2-1은 margin covariance의 선형 span만 재므로, family 내부 중복과 family 밖 정보 부재를 가를 수 있을 뿐 AUROC 정보량 상한을 판정하지 못한다. G/H probe는 사용자 판정 대기이므로 포함할 수 없고, probe 없이 “branch 추가 방향이 막혔다”와 “서로 다른 것을 아직 못 만들었다”를 완전히 가르는 것은 아니다. provenance 부재 시 margin을 공식 7-branch에 귀속시켜 실행 무효가 된다.

- **T1-5**: (1) 수용하고 수정 — P2-2는 rank-median 하나만 사전 등록하고, 다른 집계 함수를 돌려 Primary 7에서 좋은 값을 고르지 않는다. margin의 절대 크기가 증거 강도를 담는지 문서에 실측이 없으므로, rank 변환이 정보를 버리는 실패 조건은 유지한다. baseline·candidate가 같은 fold·seed·provenance로 짝지어지지 않으면 `실행 무효`다.

- **T1-6**: (1) 수용하고 수정 — P1-2·P2-2가 “branch 정보량을 늘린다”고 주장하는 문구를 제거한다. 기존 7개 margin이 고정된 상태에서는 집계 변경은 새 정보 생성이 아니라 기존 margin의 재가중·재추출일 뿐이다. eff.rank `3.52/7`, `BM·QA·DS` 유효 랭크 `1.29/3`, `CA-10`의 “형상 계열 병목은 절사가 아니라 정보량” 기록은 이 수정을 지지한다. 성공은 paired Δ로만 판정한다.

- **T1-7**: (1) 수용하고 수정 — 모든 제안의 산출물은 provenance 복구 전에는 verified 승격 근거로 인용할 수 없다. `0.6227`은 `unverified-hold`이고, Primary 7 paired Δ가 SEAL 10 개선으로 전이된다는 근거도 문서에 없다. 따라서 모든 Primary 7 판정은 `hold-out 미검증`으로만 기록하고, SEAL 10 결과를 후보·집계·seed 선택에 쓰지 않는다.

- **M1-1**: (1) 수용하고 수정 — P1-1의 실행 순서를 (i) YAML 경로를 사용해 branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed를 기록하는 최소 provenance, (ii) 러너 폴백으로 7-branch로 수렴하면 `실행 무효` 처리, (iii) 1-fold 1-seed 시간 실측, (iv) 350-fold baseline+후보 margin 기록으로 고정한다. `ICF_*` 환경변수 경로는 무효이므로 YAML 경로만 쓴다. 4 GPU-hour 초과 시 예산 내 판별 불가로 종료한다.

- **M1-2**: (1) 수용하고 수정 — P1-2는 `v_b` 산출 경로가 문서에 없으므로, closed-form ridge 해에서 모델 의미를 바꾸지 않고 fold context만으로 `v_b`를 계산할 수 있는지 코드 검사가 선행된다. 저장 ridge 해의 존재·형식·재사용 가능 여부가 확인되지 않으면 재적합 비용은 미실측이다. `v_b`가 0·비유한 fold를 결과 보고 후 제외하면 선택 편향으로 `실행 무효`다.

- **M1-3**: (1) 수용하고 수정 — P1-3은 seed·fold별 AUROC 저장, baseline·candidate의 같은 seed·fold pairing, K·seed 집합 사전 등록이 모두 필요하고, fold 적합 1회 시간이 문서에 없으므로 4 GPU-hour 내 가능한 K를 사전 등록할 수 없다. 적합 분산 포함 estimand는 사용자 승인 전이므로 승격 판정에 쓰지 못하고, K=1이면 기존 단일 적합과 동일해 중단한다.

- **M1-4**: (1) 수용하고 수정 — P2-1은 provenance가 있는 Primary 7 slide-level margin이 없으면 재적합이 필요하고, 그 비용은 미실측이다. G/H probe는 사용자 판정 대기이므로 넣을 수 없고, probe 없이 family span 진단만으로는 두 해석을 완전히 가를 수 없다. family grouping·R² 대상·leave-one-out 구성이 사전 등록되지 않으면 탐색이 선택 편향으로 바뀔 수 있다.

- **M1-5**: (1) 수용하고 수정 — P2-2는 집계 모듈에 task·fold 내부 rank 변환, ties 평균 순위, 7개 rank median을 넣어야 하고, baseline trimmed mean과 같은 fold·seed·provenance로 짝지어야 한다. rank-median 하나만 사전 등록하지 않고 여러 집계 함수를 실행해 Primary 7에서 좋은 값을 고르면 `CA-08` 선택 편향으로 `실행 무효`다. 재적합이 필요하면 fold 적합 시간 미실측 때문에 4 GPU-hour 완료를 보장할 수 없다.

- **M1-6**: (1) 수용하고 수정 — P2-3은 P1-3과 별도 구현 이점이 문서에 없으므로 P1-3으로 병합한다. seed·fold pairing 러너, seed·fold별 AUROC 저장, seed 평균 paired Δ 집계, provenance writer, K·seed 집합 사전 등록이 모두 필요하며, estimand 변경은 사용자 승인 전이다. K=1이면 중단한다.

- **M1-7**: (1) 수용하고 수정 — fold 적합 1회 시간이 문서에 없으므로, 어떤 제안도 4 GPU-hour 내 판별 가능하다고 주장할 수 없다. 먼저 1-fold 1-seed 실행 시간을 provenance와 함께 실측하고, 각 제안의 필요 fold-fit 총량을 계산한 후에만 예산 내 가능 여부를 판단한다. 이 측정 전까지 “4 GPU-hour 내 판별 가능”이라는 표현은 사용하지 않는다.

- **R1-1**: (1) 수용하고 수정 — gate ①과 family span 진단은 선형 중복만 재므로, “branch 추가 방향이 막혔다”와 “서로 다른 것을 아직 못 만들었다”를 라벨 예측 정보량 기준으로 가를 수 없다. PROJECT §5의 gate ①은 max |r|·eff.rank로 라벨 무관 직교성만 검사한다. 따라서 P1-1·P2-1의 결론 범위를 “선형 span 포화 여부”로만 제한하고, 정보량·AUROC 개선으로 확장하지 않는다.

- **R1-2**: (1) 수용하고 수정 — P1-2의 PSW는 `v_b`가 실제 fold 오차를 예측한다는 실측이 없으므로, closed-form ridge 조건부 분산 기반 재가중으로만 해석한다. `v_b`가 context 크기·조건부 특이성·구현 노이즈를 반영하면 신호 왜곡이 될 수 있다. 성능 주장은 paired Δ와 적합 분산 실측 후에만 가능하고, 그 전까지는 기전 미확인이다.

- **R1-3**: (2) 근거 반박 — AUROC의 단조 변환 불변은 최종 ensemble score에 대한 성질이지, branch margin의 rank 변환 후 median이 trimmed mean score와 항상 단조 관계라는 뜻이 아니다. 7개 branch margin을 slide rank로 변환한 뒤 median을 취하면 raw margin의 scale·tail에 의존하지 않는 새로운 slide score가 생길 수 있고, 그 score가 trimmed mean score와 다른 slide 순서를 만들면 AUROC가 달라질 수 있다. 다만 margin cardinality가 증거 강도인지, rank가 fold·seed 간에 안정적지는 문서에 실측이 없어 실패 조건으로 유지한다.

- **R1-4**: (1) 수용하고 수정 — 동일 구성의 seed std `0.0021~0.0083`은 절대 macro의 적합 변동이지, paired per-fold Δ의 seed 변동을 직접 측정하는 값이 아니다. PROJECT §4의 estimand는 같은 fold에서 짝지은 Δ이고, baseline·candidate를 같은 seed·fold로 짝지어야 seed 주효과 상쇄가 가능하다. fold×seed 잔차 구조가 미확인이고, estimand 변경은 사용자 승인 전이므로 seed 평균은 승격 판정이 아니라 분석안으로만 둔다.

- **R1-5**: (1) 수용하고 수정 — P2-1의 family spectrum은 margin 간 선형 중복의 위치를 보일 수 있지만, branch별 AUROC 기여도는 문서에 없다. 따라서 “어떤 family가 성능을 제한하는가”를 구분하지 못한다. G/H probe가 RU-85의 등방성·추정기 불안정으로 무용하다면, probe 실패는 branch 추가 방향 전체가 막혔다는 증거가 아니라 그 probe가 무용하다는 증거다. 선형 span 포화만으로 비선형 정보 중복 부재를 주장하지 않는다.

- **R1-6**: (1) 수용하고 수정 — P1-1의 provenance 요구를 전체 계측 복원보다 최소 필드로 줄인다. arm을 구성에 귀속시키는 데 필요한 최소 필드는 branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed이며, 데이터 버전은 이후 오염 검사 재설계에서 강화할 수 있다. D-050 정정에 따르면 YAML 경로는 정상이고 무효였던 것은 러너 폴백 경로이므로, 최소 통제만으로 폴백 수렴을 잡을 수 있다. 다만 1-fold 시간 미실측은 여전히 예산 게이트다.

- **R1-7**: (2) 근거 반박 — `CA-08`이 닫은 것은 “기존 브랜치 확률·마진 위의 결합 함수·부분집합 탐색을 반복하고, 선택에 쓴 평가자료의 최댓값을 그대로 확증 성과로 사용하는 절차”이지, 결합 함수 변형 일반이 아니다. `closed_axes.md` CA-08의 경계 밖 문구는 “탐색임을 명시한 제한된 결합 연구”와 “새 정보를 사용하는 PSW·TGW”를 허용한다. 따라서 rank-median이나 PSW를 단일 후보로 사전 등록하고 최댓값 선택을 하지 않는다면 CA-08 경계 안으로 단정할 근거가 없다. 다만 여러 후보를 돌린 뒤 Primary 7에서 좋은 값을 고르면 선택 편향으로 `실행 무효`다.

- **A1-1**: (1) 수용하고 수정 — provenance writer와 러너 폴백 통제는 P1-1·P2-1·P2-2의 전제 조건이다. 산출물에 branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed가 없거나, 러너가 폴백으로 같은 7-branch로 수렴하면 arm을 구성에 귀속시켜 실행 무효가 된다. D-050이 이 결함을 실측으로 확인했다.

- **A1-2**: (1) 수용하고 수정 — `0.6227`은 `unverified-hold`이므로 verified baseline으로 인용하지 않는다. paired Δ는 공식 7-branch YAML을 사용해 재실행한 baseline과 같은 fold manifest에서 짝지어야 한다. D-050 정정에서 공식 7-branch 재현이 macro `0.6226`으로 성공했다고 해도, provenance 부재 때문에 verified 기준은 아니다.

- **A1-3**: (1) 수용하고 수정 — `SCREEN_ONLY=1` 후보 margin이 앙상블에 영향을 주지 않는지는 값 앵커가 아니라 provenance 기반으로 검증해야 한다. 오염 검사 재설계가 완료되기 전에는 오염 검사 통과를 주장할 수 없고, SCREEN_ONLY가 baseline을 바꾸면 gate ① 결과가 무효다. D-050이 값 앵커 오염 검사의 정밀도 한계와 재설계 필요를 기록한다.

- **A1-4**: (2) 근거 반박 — `FC`는 `CA-09`의 경계 밖으로 명시되어 있다. `closed_axes.md` CA-09는 “여러 초점 강도의 평균 차분을 특징으로 만드는 FC도 비저촉”이라고 기록한다. 따라서 FC 예시 자체가 closed-axis 기각 사유가 되지는 않는다. 다만 강도 집합·정의를 Primary 7 성능 조회 전에 RU 카드에 등록해야 한다.

- **A1-5**: (1) 수용하고 수정 — P2-1의 probe는 `CA-10` 경계 밖 각도·모드·국소 내재차원 계열 중 사용자 판정 G/H를 통과한 것만 넣을 수 있다. G/H는 현재 사용자 판정 대기이므로, 통과 확인 없이 probe를 넣으면 경계 미확정 축을 임의로 착수하는 것이 된다. probe를 넣지 못하면 P2-1은 판별 불가로만 종료된다.

- **A1-6**: (1) 수용하고 수정 — 후보 branch, 강도 집합, `v_b` 처리 규칙, rank-median 변환, seed 수·seed 집합, family grouping, R² 대상은 모두 Primary 7 성능 조회 전에 RU 카드에 고정되어야 한다. 제안서 본문만으로는 사전 등록으로 인정할 수 없다. 성능을 보고 후보·변형·K를 고르면 §214식 선택 편향으로 `실행 무효` 또는 `반박`이다.

- **A1-7**: (1) 수용하고 수정 — max |r|·eff.rank·family spectrum은 선형 중복 지표이지 AUROC 정보량 지표가 아니다. gate ① 통과를 “branch 추가 가능”으로 확정하면 선형 직교성과 정보량을 혼동하는 증거 범위 초과다. P1-1·P2-1의 판정 범위는 선형 span 포화 여부로만 제한한다.

- **A1-8**: (1) 수용하고 수정 — P1-2의 `v_b`는 fold context와 candidate margin에서만 계산되어야 하고, query/test label, SEAL 10, cross-fold Primary 7 aggregate를不得使用하다. query label을 바꿔도 `v_b`와 slide score가 불변인지 단위 테스트로 확인해야 한다. leakage가 확인되면 `실행 무효`다.

- **A1-9**: (2) 근거 반박 — PSW는 `CA-08`의 경계 밖에 명시적으로 있다. `closed_axes.md` CA-08은 “새 정보(예측분산·Context 신호 기하)를 사용하는 PSW·TGW”를 경계 밖으로 둔다. 다만 예측분산의 유효성은 남은 검증이고, `v_b`가 fold 오차를 예측한다는 실측은 아직 문서에 없다.

- **A1-10**: (1) 수용하고 수정 — P1-2의 `v_b` zero/infinite 처리, seed 정책, 후보 등록 시점은 Primary 7 성능 확인 전에 고정되어야 한다. 결과 보고 후 임계값을 바꾸거나 비유한 fold를 제외하면 선택 편향으로 `실행 무효`다.

- **A1-11**: (2) 근거 반박 — rank-median이 CA-08 경계 안으로 단정될 근거는 없다. CA-08의 경계 밖 문구는 “탐색임을 명시한 제한된 결합 연구”를 허용하고, rank-median 하나만 사전 등록하고 최댓값 선택을 하지 않는다면 그 범위에 들어간다. 다만 rank-median이 승격 확증으로 쓰이려면 선택 이력과 paired Δ 설계를 별도로 충족해야 한다.

- **A1-12**: (1) 수용하고 수정 — P2-2의 rank/empirical CDF는 같은 task·fold 내부 slide margin만으로 계산되어야 하고 label, cross-fold 정보, SEAL 10 정보를不得使用하다. tie 처리가 seed마다 달라지면 paired Δ가 무효가 된다.

- **A1-13**: (1) 수용하고 수정 — seed 평균 paired Δ를 승격 판정에 바로 쓰는 것은 current_status의 “적합 분산을 포함한 승격 판정 절차 미승인”과 모순된다. 사용자 승인 전에는 seed 평균을 승격 estimand로 사용하지 못하고, 적합 분산 보고와 분석 기록으로만 쓴다.

- **A1-14**: (1) 수용하고 수정 — seed 재적합은 fold manifest를 고정하고 seed만 바꿔야 하며, baseline·candidate가 같은 seed·fold pairing을 가져야 한다. seed가 fold 순서·context 구성을 바꾸면 seed 평균은 독립 draw 평균이 아니다. RU-97의 fold×seed 잔차 23.9%는 이 독립성 전제가 자동 성립하지 않음을 보여준다.

- **A1-15**: (1) 수용하고 수정 — fold 적합 1회 시간이 문서에 없으므로, P1·P2 전체가 4 GPU-hour 안에서 완료되어 `지지 / 반박 / 판별 불가`를 낼 수 있는지 현재로선 판단할 수 없다. 먼저 1-fold 1-seed 시간을 실측하고, 필요 fold-fit 총량이 4 GPU-hour 또는 단일 M 2시간을 초과하면 해당 제안을 `판별 불가` 또는 `실행 무효`로 종료한다.

- **A1-16**: (1) 수용하고 수정 — P2-1의 family grouping, leave-one-out 구성, R² 대상 span, eff.rank 집계 단위는 Primary 7 성능 조회 전에 고정되어야 한다. family를 결과 보고 후 나누면 선택 편향이고, R² 대상이 불명확하면 “기존 span에서 벗어났다”는 판정이 재현 불가능하다.

- **A1-17**: (1) 수용하고 수정 — P1·P2 실행에서 SEAL 10 결과는 후보 branch, 집계 함수, K, seed, 강도 선택에 어떤 형태로도不得使用하다. D-047은 SEAL 10의 경계를 열람 금지가 아니라 선택 금지로 고정한다. SEAL 10 접근 로그·job script·RU 카드·선택 로그로 선택 금지를 검증해야 한다.

### P2 (제안)

- **T1-1** — (1) 수용·수정: `D-050`은 여섯 러너가 같은 7-branch로 수렴했고, 저장 예측에 provenance가 없어 구성 귀속이 불가능하다고 기록한다. `closed_axes.md` §3-I도 게이트 ① 실행 경로(`SCREEN_ONLY`)가 현재 무효라고 기록한다. 따라서 제안 1은 provenance writer·폴백 통제 복구와 1-fold fit 시간 실측 전에는 `실행 무효`다. 수정: FC 1개 실패로 “branch 추가 방향 전체가 막혔다”고 결론짓지 않고, 해당 후보의 선형 중복 판정만 기록한다.

- **T1-2** — (1) 수용·수정: `v_b`가 실제 fold 오차를 예측한다는 실측은 문서에 없다. `v_b` 산출 경로도 문서에 없다. 수정: 제안 2는 `v_b`가 closed-form ridge 해에서 query/test label leakage 없이 계산 가능함을 확인하고, zero/infinite 처리·seed·fold 정책을 성능 조회 전에 고정해야 한다. `v_b`와 fold 오차의 관계는 기전 미확인으로 둔다.

- **T1-3** — (1) 수용·수정: `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 미승인이라고 기록한다. `RU-97`에서 `m32_raw`의 fold×seed 잔차 23.9%는 seed 평균만으로 분산이 줄지 않을 가능성을 보여준다. fold 적합 1회 시간은 문서에 없다. 수정: 제안 3은 seed 평균 paired Δ를 승격 estimand로 바로 쓰지 않고, 적합 분산 보고를 위한 후보 설계로만 둔다. 사용자 승인 전 승격 근거로 사용하지 않는다.

- **T1-4** — (1) 수용·수정: `CA-10`은 형상 계열 병목이 절사가 아니라 정보량이라고 기록한다. §3-G·H는 AKS/LID 하위축이 사용자 판정 대기다. 수정: probe 없이 family span 진단만으로는 “branch 추가 방향이 막혔다”와 “서로 다른 것을 아직 못 만들었다”를 완전히 가를 수 없다. 결과는 선형 span 포화/미포화만 기록하고, probe 부재 시 `판별 불가`로 둔다.

- **T1-5** — (1) 수용·수정: rank-median은 CA-08에서 명시적으로 허용된 PSW·TGW가 아니다. 기존 margin 위의 결합 함수 변형으로 볼 여지가 있다. margin 절대값이 증거 강도인지도 문서에 없다. 수정: rank-median은 단일 사전 등록 후보로만 허용하고, 다른 집계 함수를 Primary 7에서 선택하지 않는다. CA-08 경계 판정 전에는 성능 확증으로 사용하지 않는다.

- **T1-6** — (1) 수용·수정: eff.rank 3.52/7, `BM·QA·DS` 유효 랭크 1.29/3, 위치 계열 상호 상관 0.25~0.93, `CA-10`의 정보량 병목 기록은 집계 변경이 새 정보를 만들지 못할 수 있음을 지지한다. 수정: 제안 2·rank-median은 “정보량 증가”가 아니라 기존 margin의 결합 효율 개선으로 한정한다. 기대 차이는 Δ_macro 0 포함 또는 `판별 불가`가 될 수 있다.

- **T1-7** — (1) 수용·수정: 공식 7-branch `0.6227`은 `unverified-hold`다. `PROJECT.md` §1은 Primary 7 개선이 hold-out 개선을 보장하지 않는다고 명시한다. 수정: 모든 제안은 provenance 복구와 verified 기준선 재구성 전에는 `실행 무효`다. 모든 Primary 7 판정은 `hold-out 미검증`으로 둔다.

- **M1-1** — (1) 수용·수정: fold 적합 1회 시간, provenance writer 함수·파일 경로, FC 구현 경로·강도 집합은 문서에 없다. 수정: 제안 1은 1-fold 1-seed 시간 실측, FC 정의·강도 집합 사전 등록, D-050 복구 완료 전에는 수행하지 않는다. 350-fold margin 재기록 비용은 실측 후 계산한다.

- **M1-2** — (1) 수용·수정: `v_b` 산출 경로, 저장 ridge 해의 형태·재사용 가능 여부, zero/infinite 처리는 문서에 없다. 수정: 제안 2는 `v_b`를 현재 closed-form ridge에서 모델 의미를 바꾸지 않고 계산할 수 없으면 중단한다. 계산 가능 확인과 처리 규칙 사전 등록이 선행 조건이다.

- **M1-3** — (1) 수용·수정: 적합 분산 포함 승격 estimand는 사용자 승인 전이다. fold 적합 1회 시간은 문서에 없다. 수정: 제안 3은 K·seed 집합을 성능 확인 전에 고정하고, 1-fold 시간 실측 후 4 GPU-hour 내 가능한 K를 사전 등록한다. K=1만 가능하면 estimand 변경 효과가 없어 중단한다.

- **M1-4** — (1) 수용·수정: provenance가 없는 margin은 공식 7-branch로 귀속시킬 수 없다. G/H probe는 사용자 판정 대기다. 수정: 제안 1은 margin을 특정 7-branch 구성에 귀속시킬 수 없으면 `실행 무효`로 중단한다. probe 없이 두 해석을 완전히 가를 수 없으므로 `판별 불가`로 둔다.

- **M1-5** — (1) 수용·수정: baseline과 candidate가 같은 fold·seed·provenance로 짝지어지지 않으면 대응 비교가 성립하지 않는다. rank-median 외 집계 함수 선택은 CA-08 선택 편향 위험이다. 수정: 제안 2는 ties 처리·rank 변환 규칙을 사전 등록하고, baseline/candidate pairing을 provenance로 검증해야 한다.

- **M1-6** — (1) 수용·수정: P2-3은 P1-3과 동일한 seed 평균 paired Δ estimand를 다룬다. 별도 구현 이점은 문서에 없다. 수정: P2-3을 별도 제안으로 유지하지 않고 P1-3의 estimand 변형으로 통합한다.

- **M1-7** — (1) 수용·수정: fold 적합 1회 시간은 문서에 없다. 따라서 어떤 제안도 4 GPU-hour 내 완료를 보장할 수 없다. 수정: 첫 실행은 모델 선택이나 성능 조회를 하지 않는 1-fold 1-seed 시간 실측으로 한정한다. 이후 각 제안의 필요 fold-fit 총량을 계산한다.

- **R1-1** — (1) 수용·수정: 게이트 ①의 max |r|·eff.rank는 branch margin의 선형 중복만 재고, label-predictive 정보량을 재지 않는다. 수정: 게이트 ① 통과는 “branch 추가 방향이 열렸다”는 증거가 아니라 선형 중복에 대한 필요 조건 통과일 뿐이다. 정보량 확인은 게이트 ② 또는 별도 성능 비교로 분리한다.

- **R1-2** — (1) 수용·수정: `v_b`는 ridge 해의 조건부 분산이지 fold AUROC 오차의 캘리브레이션된 추정치는 아니다. 문서에 `v_b`와 fold 오차 관계의 실측이 없다. 수정: `v_b`가 context 크기·조건부 특이성·구현 노이즈를 반영하는지 사전 진단하고, 기전은 미확인으로 둔다.

- **R1-3** — (2) 반박: AUROC는 최종 slide score의 단조 변환에 불변이지, branch 결합 전 raw margin의 rank 변환은 최종 slide score를 변경할 수 있다. trimmed mean은 7개 중 최저·최고 1개만 절사하므로 branch별 scale·tail 영향을 제거한다고 주장할 문서 근거는 없다. 다만 rank-median의 실제 Δ와 fit variance 효과는 정량 예측 불가다.

- **R1-4** — (1) 수용·수정: seed std 0.0021~0.0083은 동일 구성의 절대 macro 변동이다. paired per-fold Δ의 seed 변동은 같은 fold·seed pairing에서 실측해야 한다. 수정: seed 평균의 이득은 paired Δ의 seed 표준편차와 fold×seed 잔차 확인 전까지 `판별 불가`로 둔다.

- **R1-5** — (1) 수용·수정: eff.rank는 branch별 AUROC 기여도를 재지 않는다. branch별 기여도는 문서에 없다. 수정: family span 진단은 중복의 위치만 보여주고, 어떤 family가 성능을 제한하는지 구분하지 못한다. probe 실패는 “branch 추가가 막혔다”가 아니라 해당 probe의 무용일 수 있다.

- **R1-6** — (1) 수용·수정: D-050은 provenance 기반 재설계로 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 지목한다. 따라서 “전체 writer”라는 표현은 과잉이다. 수정: 제안 1의 provenance는 D-050이 지정한 최소 필드 집합으로 한정한다. fold 적합 1회 시간이 문서에 없으므로 예산 내 완료를 보장하지 않는다.

- **R1-7** — (1) 수용·수정: CA-08의 폐쇄는 선택 편향에 근거하며, 접근 일반의 영구 금지가 아니다. PSW·TGW는 명시적으로 비저촉이다. rank-median은 CA-08에서 명시되지 않았다. 수정: “집계 축이 아직 거의 탐색되지 않았다”는 운영상 유지하되, rank-median은 CA-08 경계 미확정으로 두고 단일 사전 등록 제한된 결합 연구로만 허용한다.

- **A1-1** — (1) 수용·수정: provenance writer와 러너 폴백 통제의 현재 동작은 문서에 없다. D-050은 arm이 다른 실행이 같은 7-branch로 수렴했다고 기록한다. 수정: 산출물에 branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed·데이터 버전이 기록되지 않으면 `실행 무효`로 중단한다.

- **A1-2** — (1) 수용·수정: 공식 7-branch `0.6227`은 `unverified-hold`다. verified baseline 재구성 전 paired Δ의 기준선이 불확실하다. 수정: 모든 승격 비교는 provenance가 있는 baseline 재실행 결과와 같은 fold manifest에서 짝지을 수 있을 때만 `verified`로 수행한다.

- **A1-3** — (1) 수용·수정: 값 앵커 오염 검사는 폐지·재설계 중이다. 재설계 전 `SCREEN_ONLY` 통과를 주장하는 실행은 무효다. 수정: 제안 1의 게이트 ①은 provenance 기반 오염 검사 재설계가 완료된 후에만 수행한다.

- **A1-4** — (2) 반박: `closed_axes.md` CA-09는 “여러 초점 강도의 평균 차분을 특징으로 만드는 `FC`도 비저촉이다”라고 명시한다. 따라서 FC는 CA-09 경계 밖이다. 다만 FC의 강도 집합과 정의는 RU 카드에 사전 등록되어야 한다.

- **A1-5** — (1) 수용·수정: §3-G·H는 사용자 판정 대기다. 통과 확인 없이 AKS·MDX·LID 계열 probe를 넣으면 경계 미확정 축을 임의로 착수하는 것이다. 수정: P2-1은 G/H 판정을 통과한 probe만 포함하고, 통과 전에는 probe 없이 `판별 불가`로 둔다.

- **A1-6** — (1) 수용·수정: 후보·변형·K의 사전 등록과 선택 이력 기록은 문서에 없다. 수정: RU 카드, config hash, 선택 로그, Primary 7 결과 접근 로그가 없으면 성능 확증은 `실행 무효`다.

- **A1-7** — (1) 수용·수정: max |r|·eff.rank가 AUROC 정보량이나 비선형 중복 부재를 입증한다는 근거는 문서에 없다. 수정: 게이트 ① 결과는 선형 중복의 필요 조건 판정만 기록한다. “branch 추가 가능” 확정은 `판별 불가`로 둔다.

- **A1-8** — (1) 수용·수정: PSW의 `v_b`가 fold context에서만 계산되는지, query/test label을 쓰지 않는지는 문서에 없다. 수정: query label을 바꿔도 `v_b`와 slide score가 불변인지 확인하기 전에는 `v_b`를 사용하지 않는다.

- **A1-9** — (2) 반박: `closed_axes.md` CA-08은 “새 정보(예측분산·Context 신호 기하)를 사용하는 `PSW`·`TGW`”를 경계 밖으로 명시한다. 따라서 PSW는 CA-08 경계 밖이다.

- **A1-10** — (1) 수용·수정: PSW의 zero/infinite 처리, seed 정책, 후보 등록 시점은 문서에 없다. 수정: Primary 7 성능 확인 전에 고정되지 않으면 `v_b` 변환을 성능 확증에 사용하지 않는다.

- **A1-11** — (3) 판별 불가로 남김: rank-median은 CA-08에서 명시적으로 허용되지 않았다. “제한된 결합 연구”로 해석할 여지가 있으나, 기존 margin 결합 함수 탐색 반복으로 볼 여지도 있다. 성능 확인 전 CA-08 경계 판정이 필요하다.

- **A1-12** — (1) 수용·수정: rank-median의 rank 계산 범위는 문서에 없다. 수정: task·fold 내부 margin만 사용하고 label·cross-fold·SEAL 10 정보를 쓰지 않는 것을 확인하기 전에는 `실행 무효`로 둔다.

- **A1-13** — (1) 수용·수정: `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 미승인이라고 기록한다. seed 평균 paired Δ를 승격 판정에 바로 쓰는 것은 현행 승격 절차와 모순된다. 수정: seed 평균 paired Δ는 사용자 승인 전 승격 근거로 사용하지 않는다.

- **A1-14** — (1) 수용·수정: seed 재적합이 fold manifest를 고정하고 seed만 바꾸는지, baseline/candidate가 같은 seed·fold pairing을 갖는지는 문서에 없다. 수정: pairing과 fold manifest hash를 확인하기 전 seed 평균은 독립 draw 평균으로 인정하지 않는다.

- **A1-15** — (1) 수용·수정: fold 적합 1회 시간은 문서에 없다. 수정: 1-fold 1-seed 시간 실측 전에는 어떤 제안도 4 GPU-hour 내 `지지 / 반박 / 판별 불가`를 낸다고 주장하지 않는다.

- **A1-16** — (1) 수용·수정: family grouping, leave-one-out 구성, R² 대상, eff.rank 집계 단위는 문서에 없다. 수정: Primary 7 성능 조회 전에 사전 등록하지 않으면 family span 진단은 선택 편향 위험으로 `실행 무효`로 둔다.

- **A1-17** — (1) 수용·수정: SEAL 10 결과가 후보·집계 함수·K·seed 선택에 사용되지 않는 실행 검증은 문서에 없다. 수정: SEAL 10 결과 파일 접근 로그와 선택 로그가 확인되기 전에는 D-047 위반 가능성을 배제할 수 없다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)
- **과거 branch 조합 비교는 근거로 쓸 수 없다**: `D-050`은 여섯 러너가 학습 설정을 넘기지 못해 `eval_seal_tasks.sh` 폴백에 걸리고, arm이 다른 실행들이 전부 같은 7-branch 설정으로 수렴했다고 기록한다. 저장 예측 파일에는 branch list·가중치·코드 해시 등 provenance가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다.
- **현재 7-branch의 선형 중복이 크다**: 공식 7-branch의 유효 랭크 `3.52 / 7`, `BM·QA·DS`의 유효 랭크 `1.29 / 3`, 위치 계열 상호 상관 `0.25~0.93`이 문서에 기록되어 있다.
- **모든 비교에 적합 분산을 함께 다뤄야 한다**: `D-048`은 결정론이 더는 성립하지 않으므로 fold 분산과 적합 분산을 함께 다루도록 한다. `RU-97`은 동일 구성 재적합의 seed std `0.0021~0.0083`과, `M32−M8`이 적합 분산 포함 시 `+0.0003 [-0.0067, +0.0073]`으로 0을 포함함을 실측했다.
- **승격 추정 대상은 절대 macro가 아니라 같은 fold에서 짝지은 Δ다**: `PROJECT.md` §4는 `Δ_macro`를 7 과제 × 50 fold의 대응 per-fold Δ 평균으로 고정한다.
- **SEAL 10은 모델·후보·구조 선택에 사용할 수 없다**: `D-047`은 SEAL 10의 경계를 열람 금지가 아니라 선택 금지로 고정한다. 개선 근거는 Primary 7뿐이다.
- **CA-08은 집계 함수 일반을 닫은 것이 아니다**: `closed_axes.md` CA-08의 기각 기전은 선택 편향이며, 경계 밖에는 “탐색임을 명시한 제한된 결합 연구”와 “새 정보(예측분산·Context 신호 기하)를 사용하는 `PSW`·`TGW`”가 있다.
- **`FC`는 CA-09 경계 밖이다**: `closed_axes.md` CA-09는 “여러 초점 강도의 평균 차분을 특징으로 만드는 `FC`도 비저촉”이라고 명시한다.
- **`PSW`는 CA-08 경계 밖이다**: `closed_axes.md` CA-08은 예측분산을 새 정보로 든 `PSW`를 경계 밖으로 둔다.

## 반박 (증거가 반박함)
- **“유효 랭크 3.52만으로 branch 추가 방향이 막혔다”는 일반화**: `CA-09`는 `FC`를 경계 밖으로 두고, `CA-10`은 형상 계열 병목을 절사가 아니라 정보량으로 둔다. 또한 형상 계열(`BD·SH·SJ`)이 유효 랭크를 올린 경로로 기록되어 있다. 따라서 3.52는 현재 7개 margin의 중복을 재는 값이지, 탐색 공간 전체의 상한을 증명하지 않는다.
- **`0.6227`을 verified baseline으로 인용한다**: `D-050`은 공식 7-branch 값을 `unverified-hold`로 둔다. provenance가 없어 특정 구성·fold manifest·코드 해시로 귀속시키지 못한다.
- **seed std `0.0021~0.0083`만으로 단일 적합 Δ가 뽑기다**: 승격 estimand는 같은 fold에서 짝지은 Δ다. 기록된 seed std는 동일 구성의 절대 macro 변동이며, paired Δ의 seed 변동은 아니다. 또한 `0.0021`은 승격 임계 `+0.0030`보다 작다.
- **분산 감소가 성능 개선이다**: `RU-97`에서 화이트닝은 macro를 올리지 않고 적합 분산을 4배 키웠다. seed 평균·fold pairing 등은 추정 불확실성을 줄이는 정밀도 개선이지, 기대 AUROC를 올리는 성능 개선으로 입증되지 않았다.
- **Primary 7 개선이 SEAL 10에서 ABMIL 승리를 보장한다**: `PROJECT.md` §1은 “Primary 7의 개선이 hold-out 개선을 보장하지 않는다”고 명시한다.
- **게이트 ① 통과가 branch 추가 가능 또는 정보량을 보장한다**: `PROJECT.md` §5의 게이트 ①은 `max |r|`과 `eff.rank`로 라벨 무관 직교성만 검사한다. 게이트 ② 정보량과 §4 승격은 별개다.
- **fold 적합 시간 미측정 상태에서 “4 GPU-hour 내 판별 가능”을 확정한다**: fold 적합 1회 소요 시간은 문서에 없다. Primary 7 350-fold 비교, 후보·기준·seed 재적합의 총 비용을 계산할 수 없다.
- **rank-median이 CA-08에서 명시적으로 허용된다**: CA-08 경계 밖 문구에는 `PSW`·`TGW`와 제한된 결합 연구가 있으나, rank-median 자체는 명시적으로 열거되지 않았다.

## 판별 불가 (증거가 가르지 못함)
- **유효 랭크 3.52가 “branch 추가 방향이 막혔다”인지 “서로 다른 것을 재는 branch를 아직 못 만들었다”인지**: 현재 증거는 선형 중복만 보여 준다. 두 해석을 가르려면 provenance가 있는 margin, 사전 등록된 후보, 게이트 ①(`max |r|`, `eff.rank`), 가능하면 family span 또는 probe R²가 필요하다. 게이트 ① 통과는 정보량·성능을 보장하지 않는다.
- **학습 파라미터 없는 집계 변경이 Primary 7 Δ를 올리는지**: `PSW`, rank-median, `TGW` 등 후보의 paired Δ, 적합 분산, 과제별 악화, sign agreement 실측이 없다. `v_b`가 실제 fold 오차를 예측하는지, rank 변환이 증거 강도를 보존하는지도 문서에 없다.
- **분산 감소가 성능 개선보다 나은 투자인지**: 공식 7-branch의 paired Δ 적합 분산, fold×seed 구조, seed 평균의 실제 감소 폭, 4 GPU-hour 내 가능한 seed 수가 문서에 없다. `RU-97`의 fold×seed 잔차 `23.9%`는 특정 GF 구성의 실측이지 공식 7-branch 전체의 분산 구조를 확정하지 못한다.
- **아직 건드리지 않은 축 중 어떤 것이 우선인지**: 집계, `LSAK`, `CA-R1`, AKS/LID, 학습 파라미터 후보 모두 문서에 등장하지만, 각각 사양 부재, 선택 신호 부재, 사용자 경계 판정 대기, 비용 미측정, 승격 절차 미승인으로 막혀 있다. 어느 축이 4 GPU-hour 안에 판별 가능한지는 fold 적합 시간 실측 전까지 알 수 없다.
- **4 GPU-hour 안에서 어떤 후보를 판별할 수 있는지**: fold 적합 1회 시간이 문서에 없으므로, 350-fold 기준선 재현, 후보 margin 생성, seed 재적합이 예산에 들어가는지 현재로선 판단할 수 없다.
- **Primary 7 개선이 SEAL 10 hold-out으로 전이되는지**: 모든 후보 판정은 `hold-out 미검증`이다. SEAL 10 결과를 보고 전이를 판단하는 것은 금지된다.
- **branch별 기여도**: 7개 branch 중 어떤 것이 중복·기여하는지에 대한 측정값은 문서에 없다.
- **rank-median의 CA-08 경계**: “제한된 결합 연구”로 볼 여지와 기존 margin 결합 탐색 반복으로 볼 여지가 모두 남아 있다. 사용자 경계 판정 또는 사전 등록 설계가 필요하다.

## 실행 무효 (실행·계측 결함으로 판단 불가)
- **계측 복구 전의 verified 게이트·승격 비교**: provenance writer, 러너 폴백 통제, 오염 검사 재설계, 단위 테스트가 복구되기 전에는 게이트·승격 비교를 `verified`로 수행할 수 없다.
- **provenance 없는 저장 예측 사용**: branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed가 없으면 산출물을 특정 구성에 귀속시킬 수 없다.
- **`0.6227`을 verified baseline으로 사용**: `unverified-hold` 상태이므로 승격 비교의 verified 기준선으로 인용할 수 없다.
- **과거 여섯 러너의 branch 조합 비교 사용**: arm이 다른 실행이 같은 7-branch로 수렴했으므로 구성 차이가 없는 실행 비교다.
- **`ICF_*` 환경변수 경로 사용**: `src/`가 환경변수를 읽지 않아 무효다. YAML 경로만 정상으로 확인됐다.
- **값 앵커 오염 검사 사용**: 소수 4자리 정밀도를 시스템이 유지하지 못해 구성 변화와 수치 드리프트를 구분할 수 없다. provenance 기반 재설계가 선행된다.
- **Primary 7 또는 SEAL 10 결과를 보고 후보·집계 함수·K·seed·강도를 선택**: §214식 선택 편향이며, `D-047` 위반 가능성이 있다.
- **seed 평균 paired Δ를 사용자 승인 전 승격 estimand로 사용**: 적합 분산을 포함한 승격 판정 절차가 미승인이다.
- **G/H 사용자 판정 대기 probe 임의 실행**: AKS·LID 등 경계 미확정 하위축은 사용자 판정 전 착수할 수 없다.
- **여러 집계 함수를 돌려 Primary 7에서 최댓값을 선택**: CA-08의 선택 편향 실패 양식을 반복한다.
- **라벨로 학습한 margin을 라벨 무관 게이트 ①에 투입**: 현재 절차 구멍으로, 게이트 ①의 라벨 무관성을 깨뜨린다.
- **1-fold 타이밍과 provenance 없이 350-fold 후보 비교를 verified로 수행**: 예산 초과와 구성 귀속 불가가 동시에 생긴다.

## 좌석 공백 (기권·실패로 빠진 관점)
- 없음.

## 다음 행동 (권한 등급과 함께)
- **[T0 · 즉시]** 계측 복구를 완료한다. 최소 provenance 필드로 branch list·가중치·집계 함수·코드 해시·fold manifest 해시·seed를 기록하고, 러너가 폴백으로 같은 7-branch로 수렴하면 `실행 무효`로 표기하는 통제를 넣는다. 오염 검사는 값 앵커가 아니라 provenance 대조로 재설계한다. 복구 전에는 verified 게이트·승격 비교를 하지 않는다.
- **[T0 · 즉시]** 모델 선택이나 성능 조회를 하지 않고 1-fold 1-seed fold-fit 시간을 실측한다. 이 실측 전에는 “4 GPU-hour 내 판별 가능”이라고 주장하지 않는다. 실측 후 350-fold 기준선 재현, 후보 margin 생성, 필요한 seed 재적합이 4 GPU-hour와 단일 M 2시간 안에 들어가는지 계산한다.
- **[T1 · 사전 등록 후]** 질문 (1)을 판별하는 단일 라벨 무관 probe로 `FC`를 사전 등록한다. `FC`의 정의·강도 집합·계산 범위는 Primary 7 성능 조회 전에 RU 카드에 고정한다. 실행은 게이트 ①(`max |r|`, 8-branch `eff.rank ≥ 4.02`)만 수행하고 성능을 조회하지 않는다. 게이트 ① 통과를 “branch 추가 가능”이나 “정보량 확보”로 확장하지 않는다. 예산에 들어가지 않으면 실행하지 않고 `판별 불가`로 기록한다.
- **[T1 · 사전 등록 후]** 집계 축을 실행하려면 rank-median을 CA-08 경계 밖 “제한된 결합 연구”로 인정하는 사용자 경계 판정을 먼저 받는다. 판정 전에는 rank-median을 실행하지 않는다. `PSW`를 선택하려면 `v_b` 산출 경로, query/test label leakage 부재, zero/infinite 처리, seed·fold 정책을 Primary 7 성능 확인 전에 고정해야 한다.
- **[T2 · 사용자 결정]** seed 평균 paired Δ를 승격 estimand로 바꾸는 것은 사용자 승인 전까지 하지 않는다. 승인 전에는 적합 분산 보고와 분석 기록으로만 사용한다.
- **[T2 · 사용자 결정]** SEAL 10은 최종 hold-out 개봉 전까지 후보·집계·seed·강도 선택에 어떤 형태로도 사용하지 않는다.

## 하지 말 것 (이유와 함께)
- **provenance 복구 전 과거 예측을 재사용하지 않는다**: 구성 귀속이 불가능해 비교가 `실행 무효`가 된다.
- **`0.6227`을 verified 기준으로 인용하지 않는다**: `unverified-hold`이며, provenance가 없다.
- **여러 후보를 돌려 Primary 7에서 좋은 값을 고르지 않는다**: §214식 선택 편향이며 CA-08의 실패 양식이다.
- **SEAL 10 결과를 보고 방향을 고르지 않는다**: `D-047`의 선택 금지 위반이다.
- **gate ① 통과를 성능 개선으로 해석하지 않는다**: 라벨 무관 선형 중복 검사일 뿐, 게이트 ②와 승격은 별개다.
- **seed 평균을 승격 근거로 쓰지 않는다**: 적합 분산 포함 승격 절차가 미승인이다.
- **fold 적합 시간 미측정 상태에서 4 GPU-hour 완료를 주장하지 않는다**: 예산 계산의 입력값이 없다.
- **분산 감소만 보고 성능 개선으로 기록하지 않는다**: 추정 불확실성 감소와 기대 성능 개선은 다른 estimand다.
- **G/H probe를 사용자 판정 전에 넣지 않는다**: 경계 미확정 축을 임의로 착수하는 것이 된다.
- **branch별 기여도나 기전을 문서에 없는 수치로 채우지 않는다**: branch별 기여도는 측정되지 않았고, 여러 기전은 `미확인`이다.
