# ICF Council — C-20260918-19

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
- **호출**: 20/26 · **소요**: 1399.3초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 편성 근거 (결과 전 기록)

탐색 질문이므로 제안(P) 둘을 온도 0.7·0.8로 두어 서로 다른 방향을 뽑는다. 이론(T)은 질문 1의 유효 랭크 해석을, 실측(M)은 질문 3·5의 분산과 비용을 맡는다. 반증(R)은 '새 축이 있다'는 낙관과 '다 막혔다'는 비관을 둘 다 공격한다. 감사(A)는 제안이 과거의 무효한 비교 결과에 기대고 있지 않은지 본다. 앞선 회차 보고서는 입력에 넣지 않았다(회차 간 앵커링 금지).

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: Provenance-gated 1-fold timing probe
- **바꾸는 것**: 모델 구조가 아니라 비교를 가능하게 하는 실행 계측을 바꾼다. (1) 모든 실행 산출물에 branch list, 가중치, 코드 해시, 데이터 버전, fold manifest 해시, seed, run ID를 기록한다. (2) 러너가 YAML 설정의 branch 구성을 명시적으로 전달하도록 고정하고, 폴백으로 다른 구성으로 수렴하는 경로를 제거한다. (3) 350-fold 전체 재적합 전에 Primary 7에서 최소 `7 과제 × 1 fold` timing/provenance smoke를 돌려, 4 GPU-hour 안에 전체 paired comparison이 가능한지 먼저 판별한다. SEAL 10은 열지 않는다.
- **바꾸는 전제**: 저장된 예측 파일과 과거 러너 결과가 의도한 구성에 귀속 가능하다는 전제를 뒤집는다. `D-050`에서 여섯 러너가 같은 7-branch 설정으로 수렴했고, 저장 산출물에 provenance가 없어 과거 비교를 구성 차이에 귀속시키지 못한다. 따라서 저장 산출물 기반 비교는 `unverified-hold`이며, 새 paired Δ는 provenance가 붙은 재생성 결과로만 만든다.
- **기대하는 차이**: AUROC의 정량 예측은 불가하다. 바뀌는 지표는 (a) provenance 필드 완전성, (b) 1-fold 소요 시간으로 4 GPU-hour 안에 7 과제 × 50 fold paired comparison을 수행할 수 있는지의 판별, (c) 후보 비교의 상태가 `실행 무효` 또는 `판별 불가`에서 `verified paired Δ` 가능 상태로 이동하는 것이다. 이 제안 자체는 성능을 올리는 것이 아니라, 다른 제안을 판정 가능하게 하는 전제를 복구한다.
- **기전**: provenance 기록이 구성 귀속과 오염 검사를 값 앵커가 아닌 기록 대조로 가능하게 하고, timing이 예산을 실제 비교 가능성으로 변환한다. 성능 개선 기전은 아니다.
- **실패 조건**: `7 과제 × 1 fold` probe가 4 GPU-hour를 초과하면 350-fold paired comparison은 `실행 무효`다. provenance 필드가 누락되거나 러너가 여전히 폴백 구성으로 수렴하면 산출물은 `실행 무효`다. 1-fold 시간이 과제·fold별로 대표성이 없으면 예산 판단은 `판별 불가`다. 이 복구 전에 제안 2·3을 `verified` 승격 근거로 쓰면 `실행 무효`다.

## 제안 2: Seed-averaged closed-form margin
- **바꾸는 것**: fold·task당 단일 closed-form fit margin을 쓰는 것을, R개 seed fit margin을 평균한 seed-averaged margin으로 바꾼다. 기준선과 후보는 같은 R-seed 프로토콜로 재생성하고, paired Δ는 seed-averaged margin에서 계산한다. R은 문서에 잠정 `R=8`이 있으나 승인 전이므로 4 GPU-hour timing에 따라 축소한다. seed sd와 가능하면 fold×seed 잔차를 함께 보고한다. 학습 파라미터를 도입하지 않는다.
- **바꾸는 전제**: 단일 적합의 Δ가 estimand를 충분히 대표한다는 전제를 뒤집는다. 동일 구성에서 seed만 바꿔 재적합하면 표준편차 `0.0021~0.0083`이 나오고, 승격 임계는 `+0.0030`이다. 따라서 단일 적합 차이는 대부분 뽑기일 수 있고, `D-048` 이후 적합 분산은 비교에 필수다.
- **기대하는 차이**: AUROC mean이 얼마나 오를지에 대한 정량 예측은 불가하다. 기대하는 지표 변화는 seed-averaged Δ의 seed 주효과 변동 감소, 그리고 `+0.0030` 근처 후보의 판정 상태가 `판별 불가`에서 `지지` 또는 `반박`으로 이동하는 것이다. `RU-97`의 GF 공간에서는 seed 주효과가 전체의 `0~2.9%`로 작았고, `m32_raw`에서는 fold×seed 잔차가 `23.9%`로 seed 주효과보다 컸다. 따라서 seed 평균이 총 적합 분산을 충분히 줄일지는 현행 7-branch에서 미확인이며, 문서에 현행 7-branch 분산분해는 없다.
- **기전**: 독립적 stochastic fit의 평균이 기대 margin의 추정 분산을 줄이는 통계적 경로다. 그러나 seed 변동의 원천이 무작위 표본, PCA, bag/draw, 수치적 불안정 중 무엇인지 문서에 없으므로, seed averaging이 현재 파이프라인의 어떤 불확실성을 줄이는지는 기전 미확인이다.
- **실패 조건**: fold×seed 잔차가 지배하면 R을 늘려도 승격 판정의 noise floor가 충분히 줄지 않아 `판별 불가`가 남는다. R×350-fold fit이 4 GPU-hour를 초과하면 `실행 무효`다. seed 평균이 seed 의존적 systematic bias를 숨기면 `지지`/`반박` 판단이 틀릴 수 있다. 적합 분산을 포함한 승격 판정 절차가 아직 미승인 상태이므로, 이 결과를 승인 전 승격 판정으로 쓰면 `실행 무효`다.

## 제안 3: Pre-declared fold-wise rank aggregation probe
- **바꾸는 것**: trimmed mean of raw branch margins 대신, fold 내 7개 branch margin을 rank로 변환한 뒤 동일한 trim/mean을 적용하는 단일 대안을 Primary 7 paired Δ로 비교한다. subset search, per-task selection, max picking은 금지한다. 성능 조회 전에 raw margin과 rank margin의 `max |r|`, effective rank, seed sd를 기록한다. `CA-08` 경계 밖으로 두려면 후보는 1개만, 사전 등록, 선택 이력 기록, 성능 조회 후 변형 금지를 함께 지킨다.
- **바꾸는 전제**: raw margin의 scale이 branch 간에 비교 가능하고, trimmed mean이 그 scale을 적절히 반영한다는 전제를 뒤집는다. 현재 effective rank가 `3.52 / 7`인 중복이 branch가 같은 정보를 재측정해서 생긴 것인지, 아니면 scale/variance 차이로 일부 branch가 집계를 지배해서 생긴 것인지 구분하지 않았다. rank 변환은 scale을 제거하고 ordering만으로 집계하는 단일 변형이다.
- **기대하는 차이**: AUROC의 정량 예측은 불가하다. 기대하는 지표는 paired Δ_macro, seed sd, 그리고 raw margin 대비 rank margin의 effective rank 변화다. rank margin의 effective rank가 raw margin보다 오르면 중복이 scale-driven일 가능성이 높아지고, 그대로면 branch가 실제로 같은 정보를 재측정하는 information duplication일 가능성이 높아진다. 이 관측은 effective rank `3.52`의 두 해석을 가르는 label-free 진단이 된다.
- **기전**: fold 내 monotonic rank 변환은 branch margin의 marginal scale을 제거하고, aggregation이 magnitude가 아니라 ordering에 기반하도록 만든다. 그러나 ordering이 magnitude보다 판별에 유용하다는 것은 아직 입증되지 않았으므로 기전 미확인이다.
- **실패 조건**: branch margin의 신호가 magnitude에 있으면 rank 변환은 정보 손실이 되어 Δ가 악화되고, 이 제안의 전제는 `반박`된다. `CA-08`를 단일 사전 등록 제한 연구가 아니라 aggregation variant 일반까지 포함하는 경계로 해석하면 `실행 무효`다. 350-fold paired regeneration이 4 GPU-hour를 초과하면 `실행 무효`다. rank ties나 branch 누락으로 집계가 불안정하면 `판별 불가`다. 제안 1의 provenance/timing gate를 통과하기 전에 수행하면 `실행 무효`다.

질문: 현행 7-branch에서 seed 주효과와 fold×seed 잔차의 분산분해, 그리고 seed 재적합 변동의 원천(무작위 표본, PCA 초기화, bag/draw, 수치적 불안정 등)이 문서나 코드에 기록되어 있는가?

### P2 (제안)

## 제안 1: 1-fold timing·provenance 실행 게이트
- **바꾸는 것**: Primary 7 후보 비교를 350-fold 전체 실행으로 바로 돌리는 관행을, 1-fold timing과 provenance smoke가 먼저 통과한 뒤에 전체 실험을 허용하는 관행으로 변경한다. 7과제 각각 fold 1회(가능하면 과제당 1 fold)를 full pipeline으로 실행하고, wall time, branch list, 가중치, 코드 해시, 데이터 버전, fold manifest 해시, slide-level branch margin을 기록한다. 이 결과로 350-fold 또는 350×K seed 실행이 4 GPU-hour 안에 가능한지를 사전 판정한다. SEAL 10은 열지 않는다.
- **바꾸는 전제**: fold 적합 시간은 미측정이어도 4h 안에 350-fold가 가능하다고 두고, 저장된 산출물이 특정 구성에 귀속된다고 두고 있는 전제를 뒤집는다. `D-050` 이후 provenance가 없으면 `verified` 비교가 불가하며, fold time이 없으면 4h 초과 중단을 사전에 막을 수 없다.
- **기대하는 차이**: `Δ_macro`를 생산하지 않는다. 산출물은 (a) 350-fold 또는 350×K seed 실행 가능/불가 판정, (b) provenance verified 여부다. 이 판정이 없으면 다른 제안의 Primary 7 결과는 `실행 무효` 또는 `hold-out 미검증` 탐색 기록으로만 남는다. 성능 정량 예측 불가.
- **기전**: 성능 기전이 아니라 계측 복구와 예산 회계다. provenance가 없으면 산출물을 특정 branch/weight/code에 귀속시킬 수 없어 비교 자체가 무효가 되고, fold time이 없으면 4 GPU-hour 예산 내 판별 가능성을 결정할 수 없다.
- **실패 조건**: provenance writer가 산출물에 필수 필드를 쓰지 못하면 `실행 무효`다. 1-fold timing 자체가 4h를 초과하면 `실행 무효`다. timing이 350-fold 또는 350×K seed를 초과하면 전체 승격 비교는 `실행 무효`이며, 일부 fold 결과는 승격 근거가 되지 못한다. timing이 가능해도 이 제안 자체는 성능 개선이 아니므로 `Δ < +0.0030`을 실패로 치지 않는다.

## 제안 2: Fit-variance weighted seed ensemble aggregation
- **바꾸는 것**: 단일 fold fit의 7-branch margin을 균등 trimmed mean으로 집계하는 것을, 같은 fold를 K seed로 재적합한 뒤 branch별 seed-average margin과 seed variance를 계산하고, trimmed mean의 절사 후 branch에 inverse-variance 가중을 적용하는 것으로 변경한다. K는 성능 조회 전, 1-fold timing 결과 4 GPU-hour 안에 350×K fold fits를 수행할 수 있는 K 중 `K ≥ 3`으로 고정한다. `K = 1` 또는 `K = 2`이면 이 제안은 `실행 무효`다. best seed 선택을 금지한다. fold `f`, branch `b`의 variance `σ_{b,f}^2`는 slide `i`에 대한 K seed margin의 variance를 slide 평균으로 계산하고, slide `i`에서 기존처럼 7개 margin 중 최저·최고 1개씩 raw margin 기준으로 절사한 뒤 남은 5개에 `w_{b,f} ∝ 1/σ_{b,f}^2` 가중 평균을 적용한다. zero-variance guard는 성능 조회 전에 고정된 수치 규칙으로만 허용한다.
- **바꾸는 전제**: 단일 fit이 fold의 대표값이고 7 branch가 동등하게 신뢰할 수 있다는 전제를 뒤집는다. seed variance는 label-free 정보이며, 변동이 큰 branch의 기여를 낮추는 것이 균등 절사보다 fit variance에 더 직접적이다.
- **기대하는 차이**: Primary 7 paired `Δ_macro`는 0 근처일 수 있고 정량 예측 불가다. 보고할 지표는 (i) final macro seed sd, (ii) branch-wise seed sd, (iii) paired `Δ`와 과제 군집 95% CI, (iv) 악화 과제 전량이다. 기대는 `+0.0030` 성능 증가가 아니라, 임계 근처 후보의 판정을 `판별 불가`에서 `지지`/`반박`으로 옮기는 것이다. 모든 결과는 `hold-out 미검증`이다.
- **기전**: 독립 seed error를 평균하면 seed 주효과 분산이 감소하고, inverse-variance 가중은 독립·무편향 추정치에서 최종 분산을 줄이는 표준 경로다. 그러나 현행 branch margin은 상관, 즉 유효 랭크 `3.52/7` 수준으로 중복이 크고 fold×seed 잔차가 있을 수 있어 AUROC 개선으로 이어지는 기전은 미확인이다.
- **실패 조건**: 350×K fit이 4h 초과면 `실행 무효`다. provenance 없이 seed margin을 특정 fit에 귀속시키지 못하면 `실행 무효`다. variance 계산에 AUROC/라벨을 쓰거나 결과 보고 후 가중·K를 고르면 `CA-08`/Oracle 위반으로 `실행 무효`다. `Δ < +0.0030`이면 승격 실패다. seed sd가 감소하지 않고 paired CI가 0을 포함하면 `판별 불가`다. 가중이 branch 중복을 키우고 다수 과제에서 악화를 만들면서 seed sd도 줄지 않으면 “fit-variance 가중이 trimmed mean보다 낫다”는 가설은 `반박`이다.

## 제안 3: Robust LID branch probe — effective-rank discriminator
- **바꾸는 것**: 공식 7-branch에 8번째 branch로 LID(local intrinsic dimension) margin을 임시 추가하고, 성능 조회 전에 gate ①만 수행한다. LID는 슬라이드 내 patch/token 구름의 k-NN 거리비 계열 local dimension 추정값이며, RU-85 H에서 확인된 부분추출 불안정(`|r|(m=4096, m=1024) = 0.314`)을 피하기 위해 결정론적 full-token 또는 사전 고정 deterministic subsampling과 sample-size normalization을 사용한다. margin은 기존 closed-form ridge 또는 스칼라 margin으로 환원한다. gate ①에서 `max |r| ≤ 0.6`과 8-branch `eff.rank ≥ 4.02`를 확인한다. gate ① 실패 시 AUROC를 조회하지 않고 종료한다. 이 실험은 제안 1의 provenance·timing 게이트 통과 후에만 `verified`로 수행한다.
- **바꾸는 전제**: 유효 랭크 `3.52/7`은 branch 추가 방향이 구조적으로 막혀 있다는 해석을, 아직 서로 다른 기하를 재는 branch를 만들지 못했다는 해석과 대조한다. LID가 gate ①을 통과하면 후자가 가능함을 보여준다.
- **기대하는 차이**: 1차 산출물은 성능 `Δ`가 아니라 gate ① 지표다. `max |r| ≤ 0.6`이고 `eff.rank ≥ 4.02`이면 `3.52`가 구조 상한이 아니라는 관측을 얻는다. 통과 후 gate ②(전과제 단독 AUROC 0.5 초과, 또는 1과제 40/50 fold 초과 `p < 0.01`)와 §4 paired `Δ ≥ +0.0030`은 별개다. gate ① 실패 시 성능 미조회이므로 “새 branch가 성능을 올린다”는 주장은 없다.
- **기전**: LID는 mean·covariance off-diagonal·spectral entropy·quantile·salience mean·`SH`/`SJ`가 요약하지 않는 local geometry를 읽을 후보다. 그러나 token 구름에 disease-relevant local dimension이 있는지, 그리고 추정기가 fold×seed에서 안정한지는 미확인이다. RU-85 H는 국소 차원 정보 자체의 부재가 아니라 추정기의 표본 민감성 문제를 실측했다.
- **실패 조건**: LID margin이 seed/subsample에서 불안정하면 이 추정기 변형은 `반박` 또는 `실행 무효`다. `max |r| > 0.6` 또는 `eff.rank < 4.02`면 gate ① 실패이고, 성능은 조회하지 않는다. 350-fold LID margin 계산이 4h 초과면 `실행 무효`다. `CA-10`의 모멘트 변형으로 재해석되면 `실행 무효`다. 본 제안은 `CA-10` 경계 밖인 local intrinsic dimension probe이며, 위치 계열 평균 재가중이 아니다.

질문: 현재 공식 7-branch에서 fold 1회 margin 산출에 걸리는 GPU wall time, seed 1회 재적합의 추가 시간, 그리고 slide-level branch margin을 fold/seed/provenance와 함께 저장할 수 있는지의 실측 상태는 무엇인가?

### T1 (이론)

## 이론적 병목 1
- **가정**: 유효 랭크 `3.52/7`은 “현재 7개 margin이 이미 읽을 수 있는 label-free 정보를 대부분 포함한다”는 뜻으로, branch 추가 방향이 구조적으로 막혔다고 읽힌다.
- **이 가정이 깨지는 조건**: margin의 스케일·seed·fold 변동이 rank를 지배하거나, 기존 7 margin에 직교하는 label-free margin이 존재한다. 구체적으로 성능 조회 전에 후보 margin의 기존 7 margin에 대한 회귀 잔차 유효 랭크가 높고, 게이트 ①의 `max |r| ≤ 0.6` 및 8-branch `eff.rank ≥ 4.02`를 만족하면 깨진다.
- **깨질 때 나타나는 관측 가능한 증상**: seed 평균화·fold-wise rank 변환·스케일 제거 후 기존 7의 유효 랭크가 크게 변하거나, 새 margin이 게이트 ①을 통과한다. 통과하지 못하면 “아직 못 만들었다”도 “막혔다”도 확정되지 않고 판별 불가다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `CA-09`는 형상 계열(`BD`·`SH`·`SJ`)이 유효 랭크를 올린 경로로 기록되고 “다른 경로의 불가능을 뜻하지 않는다”고 한다. `CA-10`과 `RU-85`는 일부 형상 하위축(`AKS` 등방성, `LID` 추정기 민감성)이 정보량 또는 추정 안정성 병목임을 보여, `3.52`를 전체 상한으로 읽는 것을 지지하지 않는다.

## 이론적 병목 2
- **가정**: 학습 파라미터가 없으면 집계(`trimmed mean voting`) 변경은 `CA-08`에 저촉되지 않고, `trimmed mean`은 branch 중복과 fit variance를 독립적으로 처리하는 고정 계측이다.
- **이 가정이 깨지는 조건**: `CA-08` 경계는 학습 여부보다 선택 절차에 있다. Primary 7 성능을 보고 결합 함수·부분집합·절사 폭·가중치를 고르면 경계 안이다. 반면 사전 등록한 label-free 규칙(예: seed ensemble 평균, fold-wise rank 변환, 고정 절사 폭, 예측분산·Context 신호 기하를 쓰는 `PSW`·`TGW` 계열)이고 성능 확증을 같은 자료의 최댓값으로 하지 않으면 경계 밖일 수 있다.
- **깨질 때 나타나는 관측 가능한 증상**: 같은 fold·seed margin에서 사전 등록 집계만 바꾸면 대응 Δ와 SE가 달라진다. 만약 최선 집계가 Primary 7 조회 후에 선택됐다면 결과는 실행 무효다. 사전 등록 규칙이 mean을 거의 바꾸지 않고 SE만 낮추면 계측 개선이지 성능 개선이 아니다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `CA-08`에서 확인된 기각 기전은 선택 편향이고, 결합 함수 일반의 무효는 미확인이다. `D-022`는 `PSW`·`TGW`를 비저촉으로 판정했고, `CA-08` 재개 조건은 질문·예산·선택 이력 명시와 유효한 평가 설계를 요구한다. 또한 여섯 러너가 같은 7-branch 설정으로 수렴해 과거 branch 조합 비교가 실행 무효였으므로, 과거 집계 비교는 근거가 아니다.

## 이론적 병목 3
- **가정**: fit variance 감소는 성능 mean을 올리는 것과 같은 estimand의 개선이고, 승격 임계 `+0.0030`은 단일 적합의 점추정으로 평가해도 된다.
- **이 가정이 깨지는 조건**: `D-048` 이후 적합 분산을 포함한 비교가 필수이고, seed ensemble 등으로 estimand를 ensemble mean으로 바꾸면 단일 적합 승격은 더는 유효하지 않다. 분산 감소 수단은 seed ensemble averaging, stochastic component 제거, Procrustes alignment(기전 미확인), closed-form downweighting, 고정 robust aggregation이다. 이 수단은 `E[Δ]`를 올리는 것이 아니라 `Var(Δ)`/SE를 줄이는 것이므로, `+0.0030` 근처 후보를 판별 불가에서 지지/반박으로 옮길 수 있지만 성능 개선이 아니다. 적합 분산 포함 승격 절차가 미승인인 상태에서는 variance reduction을 승격 근거로 쓰는 것도 실행 무효다.
- **깨질 때 나타나는 관측 가능한 증상**: 단일 적합 Δ가 `+0.0030`을 넘어도 seed ensemble 평균 Δ의 구간이 0을 포함하면 단일 적합은 뽑기다. seed 주효과와 fold×seed 잔차를 분리하지 못하면 seed 평균화가 fold×seed 잔차까지 줄인다고 주장할 수 없다. `RU-97`에서 fold×seed 잔차가 `23.9%`로 seed 주효과보다 크므로, seed 평균만으로 승격 판정 noise floor가 충분히 내려간다는 가정은 기전 미확인이다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `RU-97`에서 `M32−M8`의 단일 적합 `+0.0099`는 seed 42가 1.5 sd 높았던 뽑기였고, 적합 4회 평균 Δ는 `+0.0003 [-0.0067, +0.0073]`로 0을 포함한다. seed sd `0.0021~0.0083`은 승격 임계 `+0.0030`의 최대 2.8배다. 공식 7-branch 승격 근거의 과제 군집 95% 구간도 0을 포함한다.

## 이론적 병목 4
- **가정**: “아무도 건드리지 않은 축”은 실행 가능한 미탐색 축이며, 시도되지 않은 이유는 단순 미고려다.
- **이 가정이 깨지는 조건**: 축이 `closed_axes`의 재개 조건을 충족하지 못하거나, 계측·사양·선택 신호·사용자 승인 같은 전제가 없으면 시도되지 않은 것은 미고려가 아니라 유효하게 막혀 있다. 반대로 `D-048`으로 열린 학습 파라미터 축, fold time, provenance, seed ensemble, 사전 등록 집계, `LSAK`는 전제만 없으면 아직 판별 불가일 뿐이다.
- **깨질 때 나타나는 관측 가능한 증상**: 전제 실측이 나오면 축이 판별 가능/실행 무효로 갈린다. 예: fold time이 350-fold 재적합을 허용하면 seed ensemble이 판별 가능해지고, 허용하지 않으면 smoke만 남는다. provenance가 복구되면 과거 산출물 귀속과 게이트 ①이 verified로 가능해진다. 선택 신호가 없으면 `CA-R1`은 여전히 활용 불가다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: `closed_axes`는 `CA-08`(선택 편향), `CA-R1`(선택 신호 부재), `CA-10`(정보량 병목), `CA-06`(라벨 기저 붕괴), `CA-07`(분포 이동), `CA-11`(구조적 무변별), `CA-13`(이득 없음), `CA-05`(염색 정규화 부재), `CA-03`(CT 기여 미확인)을 유효한 사유로 닫거나 보류해 있다. `PROJECT.md` §2는 계측 복구 전 게이트·승격 비교를 verified로 수행할 수 없다고 하고, `current_status`는 `LSAK` 사양 부재와 적합 분산 승격 절차 미승인을 기록한다. `PROJECT.md` §1은 `D-048`으로 `CA-14`를 폐기했고, `current_status`는 첫 학습 파라미터 후보의 사전 등록을 다음 명령으로 둔다.

## 이론적 병목 5
- **가정**: GPU 1장 4-hour 예산 안에서 Primary 7 350-fold 재적합 기반 후보 판별(seed ensemble, 집계 비교, branch 추가)이 가능하다.
- **이 가정이 깨지는 조건**: fold 적합 1회 시간이 문서에 없고, 350회 재적합이 4 GPU-hour를 초과하면 깨진다. 이 경우 seed ensemble이나 전량 Primary 7 집계 비교는 실행 무효/판별 불가다.
- **깨질 때 나타나는 관측 가능한 증상**: 1-fold timing 실측이 350회 ≤ 4h를 주면 seed ensemble inference나 사전 등록 집계 비교가 판별 가능해진다. 350회 > 4h를 주면 1-fold timing, provenance 복구, margin 재계산 smoke, provenance가 복구된 margin의 오프라인 재집계만 남는다. timing이 없으면 어떤 전량 fold 재적합 후보도 4h 내 판별 가능하다고 확정할 수 없다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: fold 적합 1회 소요 시간은 문서에 없음. 예산은 GPU 1장 합계 4 GPU-hour이고, Primary 7은 7 과제 × 50 fold = 350 fold다. 따라서 4h 내 350-fold 재적합 가능 여부는 현재 판별 불가다. 또한 모든 Primary 7 판정은 hold-out 미검증이고, `PROJECT.md` §1은 Primary 7의 개선이 hold-out 개선을 보장하지 않는다고 한다.

질문: fold 적합 1회 소요 시간(또는 350-fold 재적합 총 GPU-hour)의 실측값이 있는가?

### M1 (실측)

## 실측 1: fold 적합 1회 시간 측정 — 350-fold 재적합·seed ensemble 예산 판별
- **건드려야 하는 코드·설정**: `scripts/eval_v121.sh` 또는 `eval_seal_tasks.sh` 폴백을 거치지 않고 `evaluate_pure.py`에 공식 7-branch YAML을 직접 넘기는 경로. 공식 7-branch YAML 경로는 문서에 없음. 5-branch 앵커로만 `configs/baseline/v121_active.yaml`이 확인된다. Primary 7 전 과제 1 fold씩, 1 seed로 wall time·GPU memory·branch list·코드 해시를 기록해야 한다.
- **비용**: fold 적합 1회 시간은 미실측. 과제별 fold 적합 시간을 `t_task`로 두면, 7 과제 1-fold 타이밍 측정 비용은 `Σ t_task`이고, 350-fold 단일 재적합 비용은 `50 × Σ t_task`, seed R개 ensemble은 `50R × Σ t_task`다. 4 GPU-hour 안에 들어가려면 각각 `Σ t_task < 4h`, `50 × Σ t_task < 4h`, `50R × Σ t_task < 4h`가 필요하다. 실제 `t_task`는 미실측이다.
- **막는 것**: fold 적합 시간이 미실측이라 350-fold 재적합, seed ensemble, 신규 branch 50-fold 비교가 4h 안에 가능한지 판별 불가다. 여섯 러너가 학습 설정을 넘겨 폴백으로 같은 7-branch 설정에 수렴한 전력이 있어, 타이밍 실행도 실제로 공식 7-branch가 실행되는지 provenance로 확인해야 한다.
- **중단 기준**: NaN/Inf, responsibility 합 오차 `1e-4` 초과, 입력 누락, OOM, 단일 M 2시간 초과, 합계 4 GPU-hour 초과에서 중단. 7 과제 1-fold 타이밍의 `Σ t_task`가 4h를 초과하면 타이밍 측정 자체가 이 예산에서 중단. `50 × Σ t_task > 4h`이면 단일 seed 350-fold 재적합은 실행 무효로 중단. `50R × Σ t_task > 4h`이면 해당 R의 seed ensemble은 실행 무효로 중단.

## 실측 2: provenance·러너 설정 경로 복구 — 비교 실행 무효 제거
- **건드려야 하는 코드·설정**: `scripts/eval_v121.sh`, `eval_seal_tasks.sh`, `evaluate_pure.py`, 산출물 writer, `src/`의 `ICF_*` 환경변수 경로. `ICF_*` 환경변수 경로는 무효로 확인됐고, YAML 경로의 branch 가중치 파싱은 정상으로 확인됐다. 산출물에 branch list, 가중치, 코드 해시, 데이터 버전, fold manifest hash를 기록해야 한다. 회귀는 `scripts/run_tests.sh`와 `tests/test_docs_consistency.py`로 확인한다.
- **비용**: 개발·검증 시간은 미실측. 알려진 회귀 스위트는 189 tests, 1 failed, 109초지만, provenance writer 추가 후 소요는 미실측이다. 합성 데이터에서 `sh`/`sj`를 `1.0`→`0.0`으로 바꾸면 margin `max|diff|=2.525e-01`로 달라지는 것은 확인됐으나, Primary 7 1-fold smoke 소요는 미실측이다.
- **막는 것**: 저장된 예측 파일에 branch list·가중치·코드 해시가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다. 여섯 실행 러너가 전부 같은 하나의 설정으로 수렴한 전력이 있어, 복구 전 branch 조합 비교는 실행 무효다. `D-050`으로 게이트·승격 비교를 `verified`로 수행할 수 없다.
- **중단 기준**: branch list, 가중치, 코드 해시, fold manifest hash 중 하나라도 산출물에 기록되지 않으면 중단. YAML에서 `SH`/`SJ`를 `1.0`→`0.0`으로 바꿨을 때 margin 차이가 provenance와 함께 재현되지 않으면 중단. 과거 오염 검사의 값 앵커 4자리 일치는 폐지·재설계 대상이므로, 복구 중단 기준으로 쓰지 않는다.

## 실측 3: 저장 margin 재집계로 집계 축 판별 — CA-08 경계 내 사전 설계
- **건드려야 하는 코드·설정**: 저장된 per-branch margin/prediction 파일과 fold manifest. 파일 경로와 포맷은 문서에 없음. 재집계 스크립트에서 현재 trimmed mean 외에 median, mean, fold-wise branch rank average, inverse fold-variance weight, trimmed rank, 절사 폭 사전 고정 변형 등을 계산한다. 후보는 성능 조회 전에 고정하고 선택 이력을 기록해야 한다.
- **비용**: raw margin이 fold별로 저장되어 있고 provenance가 검증되면 GPU 없이 CPU 재집계로 가능하나 소요 시간은 미실측이다. margin이 없거나 fold 대조가 불가능하면 350-fold 재적합이 필요하고, 비용은 실측 1의 `50 × Σ t_task`에 의존한다.
- **막는 것**: provenance 부재로 저장 margin을 공식 7-branch 특정 구성에 귀속시킬 수 없다. `CA-08` 경계에서 기존 margin 위의 결합 함수·부분집합 탐색을 같은 자료에서 최댓값으로 확증하는 것은 저촉이다. fold-wise rank가 branch 중복을 줄인다는 주장은 문서에 근거가 없으며, 상관·eff.rank 실측 전까지 기전 미확인이다.
- **중단 기준**: fold 대조, branch list, 가중치, 코드 해시 중 하나라도 검증 불가하면 실행 무효로 중단. Primary 7에서 후보 집계를 사후 최댓값으로 고르면 중단. SEAL 10을 조회해 집계 방식을 고르면 실행 무효다.

## 실측 4: seed ensemble·closed-form variance downweighting — 적합 분산 감소
- **건드려야 하는 코드·설정**: fold fit 루프에 seed 반복, per-seed margin 저장, seed 주효과와 fold×seed 잔차 분산분해, closed-form 하향 가중, fold-specific PCA 기저의 Procrustes 정렬. 구체적 파일 경로는 문서에 없음. fit/eval 경로(`evaluate_pure.py`, 러너)와 margin 저장 writer를 수정해야 한다.
- **비용**: seed R개는 `50R × Σ t_task` fold fit이 필요하다. `t_task`는 미실측이다. R=2는 `50 × 2 × Σ t_task < 4h`, R=4는 `50 × 4 × Σ t_task < 4h` 조건이 필요하다. Procrustes나 가중 계산 자체의 CPU·메모리 비용은 미실측이다.
- **막는 것**: 동일 구성 seed 재적합의 표준편차 `0.0021~0.0083`은 승격 임계 `+0.0030`과 같은 크기대다. RU-97에서 seed 주효과는 전체의 `0~2.9%`이고, `m32_raw`의 fold×seed 잔차는 `23.9%`로, seed 평균이 fold×seed 잔차를 줄인다는 근거는 없다. seed 변동의 원천도 미확인이다. 적합 분산을 포함한 승격 판정 절차가 사용자 미승인이라, 분산 감소 결과를 승격 근거로 쓰는 것은 현재 보류다. closed-form downweighting이 trimmed mean 잡음을 줄이고 평균 성능은 거의 안 바뀐다는 주장은 문서에 근거가 없다.
- **중단 기준**: `50R × Σ t_task > 4h`이면 해당 R의 seed ensemble은 실행 무효로 중단. 재적합 후 분산분해에서 fold×seed 잔차가 줄지 않으면 seed ensemble은 판정 분해능 개선 수단으로 반박돼 중단. Procrustes를 쓰려면 fold-specific PCA 기저의 seed 의존성이 먼저 확인돼야 하며, 미확인 상태에서는 기전 미확인으로만 기록한다. 적합 분산 승격 절차가 승인되지 않으면, 분산 감소 결과는 기록만 하고 승격 판단으로 연결하지 않는다.

## 실측 5: effective rank 3.52 해석 판별 — label-free 신규 margin 게이트 ①
- **건드려야 하는 코드·설정**: `scripts/analysis/branch_screen.py`, `SCREEN_ONLY=1`, 신규 label-free margin 생성 코드. 관측은 “공식 7-branch margin과 max |r| ≤ 0.6이고, 편입 시 eff.rank가 4.02 이상인 margin이 존재하는가”다. 후보는 위치/평균 계열이 아닌 슬라이드 내 이질성·2차 기하 등이어야 하며, `CA-10` 경계 밖으로 언급된 각도·모드 수·국소 내재차원 등은 G·H 사용자 판정 대기 상태다.
- **비용**: 신규 margin 생성과 스크리닝 시간은 미실측. GPU 필요 여부는 후보 구현에 따라 미실측이다. 성능 조회 전 종료 가능해야 한다.
- **막는 것**: `D-050`으로 게이트 ① 실행 경로가 현재 무효/`verified` 불가로 기재돼 있다. provenance 복구 전에는 max |r|·eff.rank 결과를 verified로 쓰지 못한다. 한 후보만으로는 “구조 상한”과 “아직 못 만든 중복”을 확정할 수 없다. 신규 margin이 통과하면 3.52가 구조 상한이 아니라는 증거가 되지만, 승격은 아니다. 여러 후보가 게이트 ①에서 탈락하면 위치/평균 계열 포화 가능성을 높이지만, 후보 공간 소진 없이는 반박이 아니다.
- **중단 기준**: max |r| > 0.6이면 성능 조회 없이 기각. eff.rank가 4.02 미만이면 기각. 통과해도 이 실측만으로 승격하지 않고, §4 대응 비교로만 판단한다. 모든 판정은 SEAL 10 hold-out 미검증 상태로 둔다.

## 실측 6: 4 GPU-hour 내 판별 가능 후보 우선순위
- **건드려야 하는 코드·설정**: 순서는 실측 1(7 과제 1-fold timing), 실측 2(provenance smoke), 실측 3(offline re-aggregation)이다. 전량 350-fold 재적합, seed ensemble, 신규 branch 50-fold 비교는 실측 1의 `t_task` 조건에 부차적으로만 넣는다.
- **비용**: 7 과제 1-fold timing은 `Σ t_task`가 4h 미만이면 완료 가능하고, 미실측이다. provenance smoke는 개발 시간과 1-fold smoke 비용이 미실측이다. offline re-aggregation은 margin 존재 시 CPU, 미실측이다. 350-fold 단일 재적합은 `50 × Σ t_task < 4h`, seed R은 `50R × Σ t_task < 4h` 조건이다.
- **막는 것**: 가장 큰 막힘은 fold 적합 시간 미실측이다. 다음 막힘은 provenance 부재로 과거 산출물 비교가 실행 무효가 되는 점이다. 적합 분산 승격 절차 미승인은 variance reduction의 승격 활용을 막는다. 집계 축은 `CA-08`에서 선택 편향으로 타당하게 제한됐으나, 경계 밖 사전 설계 결합 연구는 열려 있다. 적합 분산 축은 `D-048` 이후 의무화됐지만 절차 미승인으로 미탐색이다. fold fit time 미측정은 닫힌 축이 아니라 계측 공백이다.
- **중단 기준**: `Σ t_task > 4h`이면 타이밍 측정조차 중단. `50 × Σ t_task > 4h`이면 350-fold 재적합 기반 후보는 이 예산에서 중단. provenance smoke가 통과하지 못하면 저장 산출물 재집계와 과거 비교는 실행 무효로 중단. offline re-aggregation에서 fold 대조가 불가능하면 중단. 모든 판정은 `hold-out 미검증`으로 기록한다.

질문: 공식 7-branch 실행에 실제로 사용되는 YAML 설정 파일 경로, 저장된 per-branch margin/prediction 파일의 저장 경로와 포맷, 그리고 `SCREEN_ONLY=1` 게이트 ①을 현재 무효로 만든 구체적 결함 파일/경로는 무엇인가?

### R1 (반증)

## 반론 1
- **공격 대상**: “유효 랭크 3.52/7은 branch 추가 방향이 막혔다는 뜻이다” 또는 그 반대 단정.
- **더 단순한 설명**: 3.52는 현재 7개 margin이 같은 patch feature·bag mean·PCA/whitening 계열에서 나왔기 때문에 낮은 것일 수 있다. 즉 병목이 “branch 수”가 아니라 “공용 재료”일 수 있다. 이 설명은 새 branch가 같은 재료를 쓰면 상관 높고, 다른 재료를 쓰면 다를 수 있다는 관측을 낳는다.
- **무엇이 참이면 이 주장이 무너지는가**: 라벨 무관 후보 margin이 기존 7 margin과 max |r| ≤ 0.6이고 8-branch eff.rank ≥ 4.02를 만족하면 “막혀 있다”는 단정은 틀린다. 반대로, 서로 다른 재료(예: `sj.py`가 버리는 성분, 국소 내재차원, 살리언스 기하)에서 만든 후보가 모두 게이트 ①에서 탈락하면 “아직 못 만들었다”도 현재 증거로 지지되지 않는다.
- **이 반론의 한계**: 현재 effective rank는 7 margin에만 대한 측정이다. 아직 만들지 않은 branch의 상한을 결정하지 못한다. CA-10 outside subaxes(G/H)는 미확정이고, AKS는 1과제에서 `H_q > 0.99`·std 0.0017로 등방 관측, LID는 부분추출 민감성 `|r| = 0.314`로 추정기 문제 가능성이 있어 “불가능”으로 단정할 수 없다.

## 반론 2
- **공격 대상**: “집계(trimmed mean voting)를 바꾸는 축은 아직 거의 탐색되지 않았다”는 주장.
- **더 단순한 설명**: CA-08이 “집계 함수 변형 탐색”을 선택 편향으로 CLOSED로 기록했으므로, 축 전체가 미탐색이 아니라 기존 탐색이 종료된 상태다. 더 단순한 설명은, branch margin이 이미 중복(위치/평균 계열 `BM`·`QA`·`DS` eff.rank 1.29/3, 상호 상관 0.25~0.93)이므로 집계만 바꾸어도 새로운 정보가 생기지 않고 같은 잡음을 다른 형태로 재배열할 수 있다는 것이다.
- **무엇이 참이면 이 주장이 무너지는가**: CA-08의 CLOSED 기록과 경계 밖 예외가 실제 적용된다고 인정하면 “아직 거의 탐색되지 않았다”는 주장은 무너진다. 반대로, 사전 등록한 라벨 무관 또는 승인된 readout 기반 집계 규칙이 Primary 7 대응 Δ에서 +0.0030을 넘고, 선택에 쓴 자료의 최댓값을 확증으로 쓰지 않는 설계를 만족하면 “탐색 가치가 없다”는 공격이 무너진다.
- **이 반론의 한계**: CA-08 경계 밖에는 PSW·TGW처럼 예측분산·Context 신호 기하 등 새 정보를 쓰는 결합 연구가 허용된다. 따라서 “집계 변경은 전부 닫혔다”는 공격은 과다다. 허용 여부는 새 정보 사용과 사전 설계에 달려 있다.

## 반론 3
- **공격 대상**: “seed ensemble averaging은 성능 mean을 바꾸지 않더라도 승격 판정의 noise floor를 낮추는 데 먼저 쓸 가치가 있다”는 주장.
- **더 단순한 설명**: seed 평균은 seed 주효과만 줄이고 fold×seed 잔차까지 줄인다는 보장이 없다. RU-97에서 `m32_raw`는 fold×seed 잔차가 23.9%로 seed 주효과보다 크고, seed 주효과는 전체의 0~2.9%에 불과했다. 따라서 seed 평균으로 Δ의 SE가 충분히 줄지 않아 +0.0030 근처 후보가 여전히 판별 불가일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 후보의 분산분해에서 fold×seed 잔차가 seed 주효과보다 지배적이고, seed 평균 후 대응 Δ의 SE가 +0.0030 판정에 실질적으로 줄지 않으면 “seed ensemble을 먼저 쓴다”는 주장은 무너진다. 반대로, seed 주효과가 지배적이고 seed 평균 후 SE가 충분히 줄면 분산 감소는 성능 개선과 별개로 투자 가치가 있다.
- **이 반론의 한계**: RU-97의 분산분해는 GF 단독 탐색 구성의 실측이지, 현재 7-branch 공식 구성의 분산분해가 아니다. fold 적합 시간도 문서에 없으므로 4 GPU-hour 내 seed ensemble이 가능한지는 판별 불가다.

## 반론 4
- **공격 대상**: “fold-wise rank 변환으로 branch 중복의 영향을 줄이고 CA-08 경계 밖으로 갈 수 있다”는 주장.
- **더 단순한 설명**: rank 변환은 margin의 단조 변환일 뿐, margin이 같은 slide-level 신호를 공유하면 rank margin도 같은 순위를 공유한다. 즉 중복의 원인이 스케일이 아니라 공통 정보원이라면 rank 변환은 effective rank를 크게 올리지 못할 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: raw margin과 rank margin의 상관·eff.rank를 성능 조회 전에 측정했을 때 rank margin이 max |r| ≤ 0.6과 eff.rank 비하락을 만족하지 못하면, “rank 변환이 중복을 줄인다”는 주장은 무너진다. 이 경우 rank 변환은 단순한 집계 변형에 불과하고 CA-08 경계 안으로 볼 가능성이 높다.
- **이 반론의 한계**: 현재 문서에는 branch별 raw margin 상관계수나 rank margin 상관계수가 없다. CA-09의 위치 계열 상관 0.25~0.93과 전체 eff.rank 3.52/7만으로는 rank 변환의 효과를 판정할 수 없다.

## 반론 5
- **공격 대상**: “기존 7 margin으로 회귀한 잔차의 유효 랭크가 올라가는 label-free margin이 있으면, 3.52는 구조 상한이 아니라 아직 못 만든 중복이다”는 주장.
- **더 단순한 설명**: 7 margin에 대한 회귀 잔차는 새 정보일 수도 있지만, fold·seed·과제 특이 노이즈일 수도 있다. 특히 seed 재적합 변동(0.0021~0.0083)이 승격 임계 +0.0030과 같은 크기이므로, 잔차 margin이 seed 간에 불안정하면 effective rank 상승은 정보 증가가 아니라 노이즈 축 추가일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 잔차 margin의 seed std가 승격 임계보다 크거나 fold 간 부호가 흔들리면 “잔차 margin이 새 정보다”는 주장은 무너진다. 반대로, 잔차 margin이 라벨 무관이고, seed·fold 간에 안정하며, 게이트 ①(max |r|, eff.rank)과 게이트 ②(단독 AUROC 기준)를 성능 조회 전에 통과하면 “아직 못 만든 중복” 해석이 가능하다.
- **이 반론의 한계**: branch별 기여도는 문서에 없으므로, 어느 margin이 잔차에 가장 큰 영향을 주는지 말하지 못한다. 잔차의 기전도 미확인이다.

## 반론 6
- **공격 대상**: “closed-form에서 seed 분산이 존재한다는 것은 파이프라인에 불필요한 확률 요소가 있다는 뜻이고, 식별·제거하면 seed 평균보다 깨끗한 분산 감소가 된다”는 주장.
- **더 단순한 설명**: seed 분산의 원인이 “불필요한 확률”이 아니라 추정 자체의 무작위 하위샘플링·PCA 기저 비정규성·ridge 조건화일 수 있다. 이 경우 확률을 제거하면 편향이 바뀌거나 margin 분포가 달라져 성능 mean이 변할 수 있다. 즉 분산 감소와 성능 보존은 동시에 성립하지 않을 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 확률 요소를 제거한 뒤 margin mean이 바뀌거나 특정 fold/과제에서 악화가 나타나면 “불필요한 확률 제거는 단순 분산 감소다”라는 주장은 무너진다. 반대로, seed 변동의 원인이 코드 경로에서 식별되고, 해당 확률을 결정론적 규칙으로 바꿨을 때 margin mean은 유지되고 fold×seed 잔차까지 줄면 주장은 성립한다.
- **이 반론의 한계**: 현재 문서에는 seed 변동의 코드 경로별 원인이 없다. 어떤 branch에서 seed 변동이 발생하는지는 실측이 필요하다.

## 반론 7
- **공격 대상**: “fold×seed 상호작용이 fold-specific PCA 기저의 seed 의존성에서 온다면, 기저 정렬(Procrustes)이 seed 평균보다 직접적으로 잔차를 줄인다”는 주장.
- **더 단순한 설명**: fold×seed 잔차의 원인이 PCA 기저 회전/부호일 수도 있지만, 무작위 bag sampling, salience denoising, ridge closed-form의 조건부 민감성, 또는 fold 내 slide 구성 차이일 수도 있다. 더 단순한 설명은, 기저 정렬 전에 먼저 seed 변동이 어느 branch·어느 단계에서 발생하는지를 재야 한다는 것이다.
- **무엇이 참이면 이 주장이 무너지는가**: Procrustes 정렬 후 fold×seed 잔차가 거의 줄지 않으면 “기저 정렬이 seed 평균보다 직접적이다”는 주장은 무너진다. 반대로, 정렬 후 fold×seed 잔차가 유의하게 줄고 margin mean은 유지되면 기저 의존성 설명이 지지된다.
- **이 반론의 한계**: 기전은 현재 미확인이다. 문서에 fold-specific PCA 기저의 seed 의존성을 실측한 수치가 없으므로, 이 반론은 “판별 불가”를 지적하는 것이지 기저 정렬의 불가능을 입증하는 것이 아니다.

## 반론 8
- **공격 대상**: “절사 폭을 구조적 근거로 사전 고정하고 비교하는 것은 CA-08 경계 밖이며, Δ의 SE를 낮추는 계측 개선이지 모델 변경이 아니다”는 주장.
- **더 단순한 설명**: 절사 폭은 trimmed mean의 하이퍼파라미터이고, 최종 margin에 영향을 준다. 따라서 “계측 개선”이 아니라 모델/앙상블 출력 변경에 가깝다. CA-08은 기존 margin 위의 결합 함수·부분집합 탐색을 경계 안에 두고, 선택·적응을 반영한 유효한 설계가 없으면 종료된 탐색의 반복으로 볼 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 절사 폭 후보를 Primary 7 성능으로 골라 최댓값을 승격 근거로 쓰면 “CA-08 경계 밖”이라는 주장은 무너진다. 반대로, 절사 폭 후보가 성능 조회 전에 구조적 근거로 고정되고, Primary 7에서 후보별 대응 Δ와 SE를 모두 보고, 그 중 최댓값을 승격 근거로 쓰지 않는다면 CA-08 경계 밖의 제한된 결합 연구로 볼 여지가 있다.
- **이 반론의 한계**: CA-08의 경계는 “전면 금지”가 아니라 선택 편향과 확인 절차를 문제 삼는다. 따라서 사전 등록과 새 정보 사용이 있으면 저촉하지 않을 수 있다.

## 반론 9
- **공격 대상**: 과거 branch 조합 비교 결과를 근거로 쓰는 주장.
- **더 단순한 설명**: 여섯 개의 실행 러너가 전부 같은 하나의 7-branch 설정으로 수렴해 있었다면, 과거 조합 비교의 차이는 구성 차이가 아니라 실행·seed·fold 변동일 수 있다. 이 경우 “branch A가 B보다 좋다”는 관측은 구성 효과가 아니라 실행 노이즈를 재측정한 것일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: provenance 복구 후 arm별 branch list·가중치·코드 해시가 실제로 동일했음이 확인되면, 과거 branch 조합 비교를 근거로 쓰는 주장은 실행 무효로 무너진다. 반대로, arm별 설정이 서로 다름이 확인되면 과거 비교가 완전히 무효였다는 공격은 틀린다.
- **이 반론의 한계**: 저장된 예측 파일에 provenance가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다는 점은 문서에 있다. 그러나 어떤 러너가 어떤 fold·seed·branch 설정으로 실행됐는지에 대한 전량 실측은 문서에 없다.

## 반론 10
- **공격 대상**: “지금까지 아무도 건드리지 않은 축이 있다”는 일반적 주장, 특히 “fold fit time 미측정 때문에 4h 내 판별 가능한 실험이 제한된다”는 운영 판단.
- **더 단순한 설명**: “건드리지 않은 축”이 아니라 “닫힌 축·미확정 경계·계측 복구 전제”가 쌓여 있어 착수하지 못한 것일 수 있다. 현재 목표는 계측 복구(provenance writer, `SH`/`SJ` control, 오염 검사 재설계, 단위 테스트)이고, 복구 전 게이트·승격 비교를 `verified`로 수행할 수 없다. 따라서 새 모델 축을 4h 안에 검증하는 것보다 복구와 timing이 먼저일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: provenance 복구 완료, 1-fold timing 실측, 그리고 350-fold 재적합이 4 GPU-hour 안에 가능한 증거가 있으면 “판별 불가”가 아니라 후보별 대응 비교를 설계할 수 있다. 반대로, 1-fold timing조차 4h를 초과하거나 저장 margin의 재계산이 불가능하면 새 축 탐색은 실행 무효에 가깝다.
- **이 반론의 한계**: fold 적합 1회 시간은 문서에 없으므로, 4h 내 350-fold 가능 여부를 판정할 수 없다. 이 반론은 예산 부족을 입증하는 것이 아니라 판별 불가 상태를 지적한다.

## 반론 11
- **공격 대상**: “Primary 7에서 +0.0030을 넘으면 SEAL 10에서 ABMIL을 이긴다”는 암묵적 전제.
- **더 단순한 설명**: Primary 7 개선은 SEAL 10 개선을 보장하지 않는다. PROJECT.md는 “Primary 7의 개선이 hold-out 개선을 보장하지 않는다”고 명시한다. 더 단순한 설명은, Primary 7의 7과제에서 대응 Δ +0.0030은 해당 fold 분포의 추정값일 뿐, SEAL 10의 과제 분포·난이도·분산은 다를 수 있다는 것이다.
- **무엇이 참이면 이 주장이 무너지는가**: Primary 7 승격 후보가 SEAL 10 개봉에서 ABMIL 0.7266을 넘으면 최종 목표는 달성된다. 그러나 그 전에 Primary 7에서 +0.0030만 보고 “이긴다”고 단정하면 증거 범위를 넘는다. SEAL 10은 열람/선택 금지이므로, 개선 방향을 SEAL 10 결과로 고를 수 없다.
- **이 반론의 한계**: SEAL 10 수치는 기준선만 인용할 수 있고, 후보의 SEAL 10 성능은 미검증이다. 이 반론은 “Primary 7 승격이 무의미하다”는 것이 아니라 “승격과 최종 목표 달성을 동일시하면 안 된다”는 것이다.

## 반론 12
- **공격 대상**: “현재 7-branch 중복은 위치/평균 계열에 갇혀 있어서, 슬라이드 내 이질성의 2차 기하를 label-free margin으로 만들면 eff.rank가 오를 것이다”는 주장.
- **더 단순한 설명**: 2차 기하도 같은 whitened 32D 구름에서 나오면 기존 `BD`/`SH`/`SJ`와 상관할 수 있다. CA-10은 형상 계열의 병목이 절사가 아니라 정보량이라고 기록했고, G/H에서는 AKS 등방 관측(`H_q > 0.99`, std 0.0017)과 LID 추정기 불안정(`|r| = 0.314`)이 이미 보인다. 따라서 “2차 기하를 만들면 eff.rank가 오른다”는 주장은 아직 관측이 아니라 기대다.
- **무엇이 참이면 이 주장이 무너지는가**: 2차 기하 label-free margin이 게이트 ①(max |r|, eff.rank)과 게이트 ②를 성능 조회 전에 통과하지 못하면, “eff.rank가 오를 것이다”는 주장은 반박된다. 반대로, seed·fold 간 안정성과 게이트 통과가 확인되면 “아직 못 만든 독립 축” 해석이 가능해진다.
- **이 반론의 한계**: 해당 2차 기하 margin의 실제 상관계수·eff.rank·seed std는 문서에 없다. G/H는 하위축 미확정 상태이므로, 형상 축 전체를 닫았다고 단정할 수 없다.

## 반론 13
- **공격 대상**: “CA-R1이 reopened이므로 salience anchor subsampling 강도를 선택해 성능을 올릴 수 있다”는 주장.
- **더 단순한 설명**: CA-R1은 열렸지만 활용 불가다. 이득이 전역이 아니라 과제별이고(ARID1A +2.97%p vs PBRM1 −6.64%p), 라벨 없이 과제를 판별하는 선택 신호가 없다. 따라서 강도를 고르는 순간 §224 Oracle/selection bias 문제가 된다.
- **무엇이 참이면 이 주장이 무너지는가**: 라벨 없이 과제를 판별하는 선택 신호가 확립되고, 강도 선택이 사전 등록되어 Primary 7 성능 조회 뒤에 고르는 것이 아니면 “선택 신호가 없어 활용 불가”라는 주장은 무너진다. 반대로, 강도를 과제별 성능으로 고르면 Oracle 금지와 충돌한다.
- **이 반론의 한계**: D-049는 context 라벨만 쓰는 스칼라 분리도 통계를 비저촉으로 확정했지만, 게이트 ① 실행 경로가 현재 무효다. CA-R1 다중 강도 동시 투입은 §3-J에서 사용자 판정 대기다.

질문: fold 적합 1회의 실측 시간, 4 GPU-hour 내 350 fold 재적합 가능 여부, 그리고 현재 7-branch 공식 구성의 seed 변동 원인(무작위 하위샘플링·PCA 기저·ridge)과 fold×seed 분산분해 실측 기록이 있는가?

### A1 (감사)

## 점검 1: 공식 기준선 `0.6227`의 구성 귀속
- **확인하려는 것**: `0.6227`가 7-branch(`CV,BM,BD,QA,DS,SH,SJ`, `ICF_FIXED_HEAD_CT_WEIGHT=0.0`), trimmed mean, Primary 7 7×50 fold, PathoBench fold, 동일 전처리·코드에서 나온 값인지; 후보와 기준선이 같은 fold manifest와 raw margin에서 짝지어지는지. `0.6226` 재현값과 공식값의 차이는 재현 오차인지 구성·계측 차이인지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 기준선이 다른 arm·구성·폴드·전처리에서 나와 Δ가 구성 효과가 아니라 실행 편차로 보임; `+0.56%p` 승격 근거를 `verified`로 인용할 수 없음; 오염 검사·구성 대조가 실패하거나 통과를 주장할 수 없음.
- **확인 방법**: 저장된 예측·마진 파일의 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 대조; `configs/baseline/v121_active.yaml`을 직접 넘긴 5-branch/7-branch 실행으로 per-fold AUROC를 재계산; `scripts/eval_v121.sh`, `eval_seal_tasks.sh`의 폴백 경로를 점검; `tests/test_docs_consistency.py`와 provenance writer 테스트를 실행.

## 점검 2: 러너 폴백으로 인한 arm 동일 수렴
- **확인하려는 것**: 과거 branch 조합 비교가 실제로 서로 다른 구성이었는지, 아니면 `eval_v121.sh` 등 여섯 러너가 학습 설정을 넘겨 `eval_seal_tasks.sh` 폴백에 걸려 전부 같은 7-branch로 수렴했는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: branch 조합 비교가 구성 차이가 없는 실행들의 비교가 됨; arm 간 Δ가 코드·폴드·seed 편차로만 설명됨; `ICF_*` 환경변수 경로가 무효로 남아 과거 gate·promotion 근거가 실행 무효가 됨.
- **확인 방법**: runner 스크립트·로그·YAML·저장 예측 파일을 점검; 명시적 YAML 실행으로 branch list를 검증; 합성 데이터에서 `sh`/`sj` 가중치 `1.0→0.0` 변경 시 margin `max|diff|=2.525e-01` 제어 실험을 재확인; 산출물에 provenance를 기록한 뒤 arm별 구성을 대조.

## 점검 3: fold manifest 고정과 후보·기준선 짝짓기
- **확인하려는 것**: Primary 7 fold가 PathoBench 공식 분할 그대로 고정됐는지, 후보와 기준선이 같은 fold `i`에서 짝지어졌는지, fold 수 증대·재분할이 없었는지, Primary 7과 SEAL 10 fold·과제가 섞이지 않았는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: fold 재분할·과제 난이도 편차가 Δ에 섞임; 절대 macro 차이로 판정하게 됨; paired Δ의 정밀도가 무효화되고 설계 효과가 무시됨.
- **확인 방법**: baseline·candidate fold ID·manifest 해시를 대조; PathoBench split 로더를 점검; 7 과제 × 50 fold = 350 fold pair 수를 검증; Primary 7과 SEAL 10 간 fold·과제 중복 여부를 확인; fold manifest 해시를 provenance에 기록.

## 점검 4: 적합 분산과 `+0.0030` 승격 임계의 비교 단위
- **확인하려는 것**: `+0.0030` 승격 기준이 단일 적합 Δ에 그대로 적용되는지, 아니면 `D-048` 이후 적합 분산을 포함한 estimand로 판정하는지; seed std `0.0021~0.0083`가 임계보다 큰 구성에서 단일 적합 비교가 유효한지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 단일 적합 Δ가 임계 초과해도 seed·fit 분산 때문에 판별 불가가 됨; best-seed 선택으로 편향이 생김; RU-97의 `+0.0099`처럼 seed 42의 뽑기가 효과로 오인됨.
- **확인 방법**: promotion 결정 기록, RU-97 보고서, 협의체 조건부 설계안을 점검; seed·fold별 paired Δ를 재계산; 총 SE와 seed std를 함께 보고; best-seed 선택이 코드·기록에 없는지 확인.

## 점검 5: 유효 랭크 `3.52/7`의 계산 범위와 라벨 무관성
- **확인하려는 것**: 유효 랭크 `3.52/7`이 어떤 데이터·fold·margin 스케일·라벨 사용·seed/적합에서 계산됐는지; branch 추가가 막혔는지, 아니면 서로 다른 것을 재는 label-free branch를 아직 못 만들었는지를 가르는 관측이 게이트 ①의 `max |r|`과 `eff.rank`인지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 라벨 의존 margin이나 선택된 fold에서 계산돼 게이트 ①이 무효가 됨; margin 스케일·정규화에 따라 `eff.rank`가 변동함; 단일 적합의 seed 변동이 직교성 판단에 섞임.
- **확인 방법**: `scripts/analysis/branch_screen.py`를 `SCREEN_ONLY=1`로 실행; covariance·rank 계산 코드, 정규화, 라벨 사용 여부를 점검; 입력 manifest·branch list·seed를 기록; `3.52/7`과 재계산값을 대조; 신규 label-free margin의 `max |r|`과 8-branch 기준 `eff.rank ≥ 4.02` 충족 여부를 성능 조회 전에 확인.

## 점검 6: branch margin의 전처리·정규화·fit 범위
- **확인하려는 것**: 각 branch의 PCA-32, whitening, closed-form ridge, GMM 등 통계량이 fold context/train 범위에서만 fit되는지; query/test fold, SEAL 10, 또는 전체 17 과제가 fit에 포함되지 않는지; background GMM이 기존 3M 표본·CPU thread 8·`max_iter=100` 범위만인지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test fold feature가 PCA·whitening·ridge·GMM fit에 들어가 leakage가 됨; GMM 미수렴·responsibility 합 오차 `1e-4` 초과·sklearn 불일치가 발생함; stain normalization 부재로 branch margin 비교가 무효가 됨.
- **확인 방법**: `src/models` branch 코드, config, 로그를 점검; fit 호출의 입력 배열이 query label·test fold를 포함하지 않는지 확인; 단위 테스트를 실행; GMM 수렴·responsibility sum·sklearn 대조를 확인; fold별 fit 범위를 provenance에 기록.

## 점검 7: 라벨 사용 경계 — query label, context label, oracle
- **확인하려는 것**: branch margin과 집계에서 query label이 사용되지 않는지; context label 사용이 승인된 closed-form readout 범위인지; 라벨로 학습한 margin이 라벨 무관 게이트 ①에 들어가지 않는지; AUROC를 보고 강도·변형을 고르는 oracle이 없는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: query label leakage가 발생함; 게이트 ①의 label-free 요건이 위반됨; `CA-06` 또는 §224 oracle 금지를 우회함; 성능 조회 후 선택이 발생함.
- **확인 방법**: branch·집계 함수의 label 인자를 코드 감사; `SCREEN_ONLY` 경로가 label을 쓰지 않는지 확인; config와 실행 로그를 점검; AUROC가 강도·변형·집계 선택에 사용되지 않았는지 선택 이력으로 확인.

## 점검 8: 집계 변경의 사전 고정과 `CA-08` 경계
- **확인하려는 것**: trimmed mean 대체 후보(예: fold-wise branch rank, median, pre-fixed trim, seed ensemble average, variance-stabilizing transform)가 Primary 7 성능 조회 전에 사전 고정되는지; `CA-08` 경계 밖의 제한된 결합 연구인지; 선택에 쓴 Primary 7 최댓값을 확증으로 쓰지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 같은 Primary 7에서 집계 후보를 고르고 다시 성공으로 판정하는 선택 편향이 발생함; `CA-08` 경계에 저촉됨; variant max가 과적합됨.
- **확인 방법**: RU 카드·사전 등록·선택 로그를 점검; 집계 변형을 사전 규칙대로 고정 fold에서 실행; 모든 variant 결과를 보고; SEAL 10을 선택에 사용하지 않는지 확인; 성능 확증을 탐색과 분리한 설계를 확인.

## 점검 9: seed ensemble·분산 감소와 성능 개선의 구분
- **확인하려는 것**: seed ensemble·평균화, rank 변환, closed-form 하향 가중 등 분산 감소 수단이 seed 주효과를 줄이는지 fold×seed 잔차까지 줄이는지; 성능 mean(paired Δ)과 noise floor(SE·seed std)을 구분하는지; best-seed 선택이 없는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: seed 평균이 fold×seed 잔차를 줄이지 못해 승격 판정 noise floor가 유지됨; best-seed 선택으로 편향이 생김; 분산 감소가 성능 개선으로 오기됨.
- **확인 방법**: fold별 R seed 재적합을 실행; seed 주효과와 fold×seed 잔차를 분산분석으로 분리; seed 선택 코드가 없는지 확인; paired Δ mean과 총 SE를 함께 보고; `R=8` 잠정 보정 등 승인 절차를 확인.

## 점검 10: fold 적합 시간과 4 GPU-hour 판별 가능성
- **확인하려는 것**: fold 적합 1회 시간이 4 GPU-hour 안에 7 과제 × 50 fold × seed/변형 수를 허용하는지; 1-fold timing 실측이 있는지; 단일 M 2시간 또는 합계 4 GPU-hour 초과 중단 조건과 충돌하지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 4 GPU-hour 안에 판별 가능 후보를 결정할 수 없음; 예산 초과로 실행 무효가 됨; 단일 M 2시간 초과로 중단됨.
- **확인 방법**: GPU 4에서 baseline·candidate 1-fold timing을 실행; wall time·GPU time을 기록; seed·변형 수로 외삽하고 불확실성을 보고; SEAL 10을 사용하지 않음.

## 점검 11: provenance writer와 오염 검사 재설계
- **확인하려는 것**: provenance writer와 오염 검사 재설계가 실제로 산출물에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 기록하는지; `SCREEN_ONLY`가 앙상블을 건드리지 않는지; 값 앵커 대신 provenance 대조가 가능한지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: screening이 앙상블에 영향을 줌; 구성 변화와 수치 드리프트를 구분할 수 없음; 오염 검사 통과·실패 모두 무의미해짐; 과거 산출물을 특정 구성에 귀속시킬 수 없음.
- **확인 방법**: provenance writer 코드·테스트를 점검; 산출물 metadata에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시가 있는지 확인; `SCREEN_ONLY=1` 실행 후 앙상블 출력이 변하지 않는지 대조; 단위 테스트를 실행.

## 점검 12: SEAL 10 선택 금지 준수
- **확인하려는 것**: 모델 개선 근거가 Primary 7뿐인지; SEAL 10 결과가 후보·구조·집계·강도 선택에 사용되지 않았는지; 현재 목표에서 SEAL 10을 열지 않는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: holdout peeking이 발생함; 선택 편향이 생김; 최종 hold-out 판정이 무효가 됨.
- **확인 방법**: 접근 로그, RU 카드, 결정 기록을 점검; 후보 선택에 SEAL 10 수치가 사용되지 않았는지 확인; 실행 스크립트가 SEAL 10을 선택 경로에 연결하지 않는지 확인.

## 점검 13: baseline·임계값·선택 이력의 사전 고정
- **확인하려는 것**: baseline `0.6227`, threshold `+0.0030`, `D-042` 승격, RU-80 MDE 등 선택·기준 설정 이력이 성능 확증 전에 고정됐는지; 탐색·적응이 반영된 유효한 설계인지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 적응적 선택이 Δ를 과대평가함; baseline이 후보 보고 후 고정됨; 승격 판정이 실행 무효가 됨.
- **확인 방법**: `history/archive.md`, `D-041`·`D-042`, RU-80, 사전 등록 카드를 점검; baseline이 후보 확증 전에 고정됐는지 확인; 탐색한 구성과 선택 이력을 기록했는지 확인.

## 점검 14: branch별 기여도와 과제별 악화 관측
- **확인하려는 것**: SH/SJ 단독·7-branch에서 악화 과제가 같은 방향이라는 관측이 branch별 기여도 측정에서 나온 것인지, 아니면 아직 branch별 기여도가 측정되지 않았는지; 관측된 악화를 기전으로 해석하지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 관측된 과제별 악화를 특정 branch 기전으로 해석함; branch 기여도 미측정 상태에서 승격·기각이 이루어짐; 기전 미확인을 기전 입증으로 바꾸어 씀.
- **확인 방법**: 사전 지정된 leave-one-branch paired Δ를 실행; 전 과제 결과를 보고; margin 파일에서 branch별 기여도를 독립 재계산; 기전은 `기전 미확인`으로 명시.

질문: 유효 랭크 `3.52/7`을 계산한 데이터 집합(Primary 7/SEAL 10/전체 17), fold 수, branch margin의 스케일·정규화, 라벨 사용 여부, seed/적합 횟수, `branch_screen.py`의 rank 정의는 무엇인가?
질문: fold 적합 1회의 GPU 시간·벽 시간은 Primary 7 7 과제 × 50 fold × seed/변형 수에서 4 GPU-hour 안에 가능한가? 1-fold timing 실측값이 있는가?
질문: 현재 저장된 예측·마진 파일에 branch list, 가중치, 코드 해시, 데이터 버전, fold manifest 해시가 기록되어 있는가? provenance writer가 구현·유효한가?
질문: 각 branch의 PCA/whitening/ridge/GMM fit이 fold context/train 범위에서만 수행되는지, query/test fold 또는 SEAL 10에 fit되지 않는지 코드·로그로 확인된 기록이 있는가?
질문: 집계 변경 후보(trimmed mean, rank, seed ensemble, variance rule)가 Primary 7 성능 조회 전에 사전 등록·고정된 RU 카드가 있는가?
질문: seed ensemble/평균화에서 best-seed 선택이 금지되고, seed 주효과와 fold×seed 잔차를 분리해 보고하는 절차가 승인·구현되어 있는가?

---

## Phase 2 — 교차

### T1 (이론)

- **가정**: 제안 1~3은 공통으로 “seed 재적합·1-fold timing·rank 변환이 Primary 7 고정 fold에서 후보-기준선 paired Δ의 estimand를 바꾸지 않는 중립 계측이다”라고 두고 있다. P2는 seed fit이 같은 fold·task·branch margin에 대한 독립적 무작위 적합 오차이고, seed 주효과와 fold×seed 잔차로 분해 가능하며, seed 평균 또는 분산만 쓰는 inverse-variance 가중이 paired Δ의 불확실성만 줄인다고 둔다. P1은 1-fold timing/provenance가 350-fold 또는 350×K paired comparison의 예산과 구성 귀속을 대표한다고 둔다. P3는 fold-wise rank 변환이 branch margin의 scale nuisance만 제거하고, rank 변환 전후 effective rank 변화가 scale-driven 중복과 정보 중복을 판별한다고 둔다.

- **이 가정이 깨지는 조건**: seed 변동이 무작위 오차가 아니라 fold-specific PCA 기저, bag/draw, 표본화, 초기화, 수치 경로가 margin의 의미 자체를 바꾸는 systematic variation일 때; fold×seed 상호작용이 seed 주효과보다 커서 R 증가로 제거되지 않을 때; seed 평균이 서로 다른 estimand를 섞어 paired Δ의 부호를 R에 의존하게 만들 때; 1-fold timing이 과제·fold·seed·데이터 로딩·메모리·폴백 경로에 비선형일 때; rank 변환이 effective rank를 기계적으로 바꿔 중복 원인의 판별 지표가 되지 못할 때; branch margin의 상관이 분산만 쓰는 가중을 무효화할 때; provenance 필드가 같아도 실행 경로(폴백, 캐시, 환경)가 margin을 바꿀 때.

- **깨질 때 나타나는 관측 가능한 증상**: seed 평균 후 paired Δ가 R 변화에 따라 부호를 바꾸거나 +0.0030 근처에서 계속 0을 포함해 `판별 불가`가 남는 것; branch별 seed sd가 fold·task마다 체계적으로 달라 inverse-variance 가중이 특정 branch를 구조적으로 제거하거나 과제를 악화시키는 것; rank 변환 후 effective rank가 올라가도 paired Δ가 악화되면 P3의 scale-removal 전제는 `반박`되고, effective rank만 바뀌고 Δ가 0을 포함하면 `판별 불가`다; 1-fold timing은 4 GPU-hour 안에 가능하지만 350-fold 또는 350×K가 초과해 `실행 무효`가 되는 것; provenance 필드가 기록돼도 margin이 seed/코드/데이터 버전 외 경로에서 달라져 `실행 무효`가 되는 것; fold×seed 잔차가 seed 주효과보다 커서 R 증가가 noise floor를 거의 못 줄이는 것.

- **문서의 어떤 관측이 이 가정과 충돌하는가**: 현행 7-branch의 seed 주효과/fold×seed 잔차 분산분해와 seed 재적합 변동 원천에 대한 직접 충돌 관측은 해당 관측 없음 — 주어진 문서에는 기록이 없고, 원천은 기전 미확인이다. fold 적합 1회 소요 시간도 문서에 측정치가 없어 P1의 1-fold timing이 350-fold 예산을 대표한다는 가정은 아직 판별 불가하다. 다만 P2가 암묵적으로 두는 “분산만 알면 inverse-variance 가중이 충분하다”는 부분 가정과 충돌하는 관측은 유효 랭크 `3.52/7`이다. RU-97의 `m32_raw` fold×seed 잔차 `23.9%`와 seed 주효과 `0~2.9%`는 seed 평균이 fold×seed 잔차를 제거한다는 전제와 충돌할 수 있으나, 이는 GF 탐색 공간의 관측이지 현행 7-branch가 아니다. D-050의 여섯 러너가 같은 7-branch 설정으로 수렴하고 provenance 부재로 과거 비교를 구성에 귀속할 수 없다는 관측은 제안 1의 필요성을 지지할 뿐, 제안 2·3의 계측 중립성과 직접 충돌하지 않는다.

### M1 (실측)

## 실측 1: P1-1 Provenance-gated 1-fold timing probe
- **건드려야 하는 코드·설정**: `scripts/eval_v121.sh` 등 여섯 러너의 YAML 설정 전달 경로, `scripts/eval_seal_tasks.sh`의 폴백 분기, provenance writer(구체적 파일 경로 미확인 — `D-050`에서 "provenance writer"로만 언급, `current_status.md` 현재 목표에 "provenance writer, SH/SJ branch control 경로, 오염 검사 재설계, 단위 테스트"로 등재), 산출물 파일에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시·seed·run ID 필드 추가. `configs/baseline/v121_active.yaml`의 branch 가중치 파싱은 `D-050` 정정으로 정상 확인됨.
- **비용**: 7 과제 × 1 fold = 7 fold 실행. 1-fold wall time **미실측**. 7 fold 총 시간 **미실측**. 메모리·연산 **미실측**. `current_status.md` 회귀 테스트 109초(189 tests, 1 failed)는 테스트 시간이지 fold 적합 시간이 아니다.
- **막는 것**: (1) fold 적합 1회 시간이 미실측이므로 7 fold probe가 4 GPU-hour(14400초) 안에 끝나는지 사전 판정 불가. 1-fold가 200초면 7 fold = 1400초(0.39h)로 가능, 1-fold가 2000초면 14000초(3.9h)로 간신히 가능, 1-fold가 2000초 초과면 4h 초과. (2) provenance writer가 현재 결함 상태 — `current_status.md` "1 failed (16 skipped)" 중 1건이 기존 결함, `D-050`에서 "승격 근거를 verified로 인용하지 않는다"고 명시. writer가 필드를 쓰지 못하면 산출물은 `실행 무효`. (3) 1 fold만 재면 과제·fold별 시간 편차가 미실측으로 남아 350-fold 전체 예산 판단은 `판별 불가`.
- **중단 기준**: 7 fold probe 합계 4 GPU-hour 초과, provenance 필드 누락, NaN/Inf, OOM, 단일 M 2시간 초과, GMM 미수렴. 1-fold 시간이 과제 간 편차(최대/최소 비)를 보여주면 350-fold 외삽은 `판별 불가`로 종료.

## 실측 2: P1-2 Seed-averaged closed-form margin
- **건드려야 하는 코드·설정**: fold·task당 단일 fit을 R개 seed fit으로 교체하는 실행 루프, R개 margin의 평균화 로직, seed sd와 fold×seed 잔차 보고. 구체적 파일 경로 미확인. `D-048` 이후 적합 분산 보고가 필수.
- **비용**: 350 × R fold fits. R=8(잠정, 승인 전)이면 2800 fits. 1-fold wall time **미실측**. 4 GPU-hour(14400초) 안에 2800 fits을 수행하려면 1-fold ≤ 5.14초. 1-fold가 5초면 2800×5=14000초(3.89h), 1-fold가 10초면 28000초(7.78h)로 초과. R=4이면 1400 fits, 1-fold ≤ 10.3초. R=2이면 700 fits, 1-fold ≤ 20.6초. 전처리(patch feature 추출, GMM, PCA) 포함인지 **미실측**. closed-form ridge 자체의 연산 시간은 **미실측**.
- **막는 것**: (1) 1-fold 시간 미실측 — R=8이 4h 안에 들어가는지 판정 불가. (2) `RU-97` §8에서 `m32_raw`의 fold×seed 잔차가 23.9%로 seed 주효과보다 큼 — seed 평균이 총 적합 분산을 충분히 줄일지 **미확인**. 이 수치는 GF 공간(`M ∈ {8,16,32}` × 화이트닝 × 표본)의 결과이지 현행 7-branch의 분산분해가 아니다. 현행 7-branch의 seed 주효과·fold×seed 잔차 분산분해는 문서에 **없음**. (3) 적합 분산을 포함한 승격 판정 절차가 `current_status.md` 미해결로 "사용자 승인 전까지 학습 파라미터 후보의 승격 판정은 보류" — 이 제안은 학습 파라미터가 아니지만, 적합 분산 보고 절차 미승인 상태이므로 결과를 승격 근거로 쓰면 `실행 무효`.
- **중단 기준**: 350×R fit 합계 4 GPU-hour 초과, 1-fold가 5.14초(R=8) 또는 10.3초(R=4) 초과, provenance 필드 누락, seed sd가 감소하지 않고 paired CI가 0 포함이면 `판별 불가`.

## 실측 3: P1-3 Pre-declared fold-wise rank aggregation probe
- **건드려야 하는 코드·설정**: trimmed mean(raw margin)을 fold 내 7개 branch margin의 rank 변환 후 trim/mean으로 교체하는 집계 로직. 구체적 파일 경로 미확인. `CA-08` 경계: "기존 브랜치 확률·마진 위의 결합 함수·부분집합 탐색을 반복"이 경계 안, "탐색임을 명시한 제한된 결합 연구"가 경계 밖. 이 제안은 단일 사전 등록 변형이므로 경계 밖에 해당할 수 있으나, "결합 함수 변형"이라는 형식 자체가 `CA-08` 기각 기전(§214-V 선택 편향)과 겹치므로 경계 판정이 **미확정**.
- **비용**: 350-fold paired regeneration(기준선 + 후보). 1-fold wall time **미실측**. 4 GPU-hour 안에 350 fits을 수행하려면 1-fold ≤ 41.1초. rank 변환 자체는 집계 단계의 연산이므로 추가 fit이 불필요할 수 있으나, `D-050`으로 저장된 margin이 provenance 없이 특정 구성에 귀속되지 않으므로 재생성이 선행 필요. 재생성 비용 = 350-fold fit.
- **막는 것**: (1) 1-fold 시간 미실측. (2) `CA-08` 경계 미확정 — "결합 함수 변형"이 경계 안인지 경계 밖인지 `closed_axes.md` §3에 확정 판정 없음. A~F는 판정 완료, G·H·I·J는 미확정. rank aggregation은 G~J에 해당하지 않으므로 **경계 미확정**. (3) rank 변환이 effective rank `3.52`의 두 해석(구조 상한 vs scale-driven 중복)을 가르는지 **미확인** — rank margin의 effective rank가 raw margin과 다른지 실측 전에는 알 수 없음.
- **중단 기준**: 350-fold 재생성이 4 GPU-hour 초과, `CA-08` 경계 안으로 판정되면 `실행 무효`, rank ties로 집계 불안정, provenance 필드 누락.

## 실측 4: P2-1 1-fold timing·provenance 실행 게이트
- **건드려야 하는 코드·설정**: 실측 1과 동일. `scripts/eval_v121.sh`, `scripts/eval_seal_tasks.sh`, provenance writer(경로 미확인), `configs/baseline/v121_active.yaml`.
- **비용**: 실측 1과 동일. 7 fold, 1-fold **미실측**, 총 **미실측**.
- **막는 것**: 실측 1과 동일. 추가: P1-1과 P2-1은 동일 실험의 재진술. 두 좌석 모두 "7 과제 × 1 fold"를 제안하지만, 1 fold가 350-fold 전체를 대표하는지(과제별·fold별 시간 편차) **미실측**. 1 fold만 재면 350-fold 예산 판단은 `판별 불가`.
- **중단 기준**: 실측 1과 동일.

## 실측 5: P2-2 Fit-variance weighted seed ensemble aggregation
- **건드려야 하는 코드·설정**: K seed fit 루프, branch별 seed variance 계산(slide `i`에서 K seed margin의 variance를 slide 평균), trimmed mean 절사 후 inverse-variance 가중(`w_{b,f} ∝ 1/σ_{b,f}^2`), zero-variance guard(성능 조회 전 고정 수치 규칙). 구체적 파일 경로 미확인.
- **비용**: 350 × K fold fits. K≥3이면 최소 1050 fits. 1-fold wall time **미실측**. 4 GPU-hour 안에 1050 fits을 수행하려면 1-fold ≤ 13.7초. K=3, 1-fold 10초면 10500초(2.92h), 1-fold 15초면 15750초(4.38h)로 초과. K=5이면 1750 fits, 1-fold ≤ 8.2초. 전처리 포함인지 **미실측**.
- **막는 것**: (1) 1-fold 시간 미실측. (2) `CA-08` 경계: "새 정보(예측분산·Context 신호 기하)를 사용하는 PSW·TGW"가 경계 밖으로 명시됨. fit-variance는 "예측분산"에 해당할 수 있으나, `closed_axes.md` §3-A에서 "예측분산의 유효성"이 "남은 검증"으로 등재 — 즉 예측분산이 실제로 branch margin의 적합 분산을 대표하는지 **미확인**. (3) `RU-97` fold×seed 잔차 23.9% — seed variance가 fold×seed 잔차를 포착하는지 **미확인**. (4) inverse-variance 가중이 branch 중복(effective rank 3.52/7)을 키울 수 있음 — **미실측**.
- **중단 기준**: 350×K fit 합계 4 GPU-hour 초과, 1-fold가 13.7초(K=3) 초과, variance 계산에 AUROC/라벨 사용, 결과 보고 후 K·가중 변경(`CA-08`/Oracle 위반), seed sd 감소 없이 paired CI 0 포함이면 `판별 불가`.

## 실측 6: P2-3 Robust LID branch probe
- **건드려야 하는 코드·설정**: LID margin 계산(k-NN 거리비 계열 local dimension 추정), full-token 또는 deterministic subsampling, sample-size normalization, closed-form ridge 또는 스칼라 margin 환원, `scripts/analysis/branch_screen.py`로 gate ①(`max |r| ≤ 0.6`, 8-branch `eff.rank ≥ 4.02`) 실행. `CA-10` 경계 밖: "국소 내재차원"이 명시적으로 경계 밖으로 등재. `CA-10` 경계 안: "`SH`의 모멘트 정의를 변주하는 것(차수 추가, 로버스트 추정량 교체, 성분 분리)" — LID가 모멘트 변주로 재해석되면 `실행 무효`.
- **비용**: 350-fold LID margin 계산. 슬라이드당 token 수 **미실측**. k-NN 거리비 계산의 연산 복잡도 **미실측**. full-token 사용 시 슬라이드당 token 수 × k-NN 계산. 1-fold wall time **미실측**. 4 GPU-hour 안에 350-fold LID margin을 계산할 수 있는지 **판정 불가**. `RU-85` H에서 부분추출 크기 m=4096, m=1024 사용 — full-token은 m이 더 크므로 계산 비용이 더 클 수 있으나 **미실측**.
- **막는 것**: (1) `RU-85` H에서 부분추출 크기만 바꾼 두 마진의 상관 `|r|(m=4096, m=1024) = 0.314`로 강건성 임계 0.8을 크게 밑돎 — "개념이 아니라 추정기의 표본 민감성이 죽었다." full-token이 이 문제를 해결하는지 **미확인**. (2) 슬라이드당 token 수, k-NN 계산 비용 **미실측** — 4h 예산 내 350-fold가 가능한지 판정 불가. (3) `CA-10` 경계: LID가 "국소 내재차원"이면 경계 밖이지만, "모멘트 정의 변주"로 재해석되면 경계 안. 이 구분이 **미확정**. (4) gate ① 실행 경로(`SCREEN_ONLY`)가 `D-050`으로 현재 무효 — "게이트 ① 실행 경로(`SCREEN_ONLY`)가 현재 무효다(`D-050`). 계측이 복구된 뒤 라벨 무관·저비용 게이트 ①을 먼저 돌린다."
- **중단 기준**: gate ① 실패(`max |r| > 0.6` 또는 `eff.rank < 4.02`) 시 성능 미조회로 종료, 350-fold LID margin 계산이 4 GPU-hour 초과, LID margin이 seed/subsample에서 불안정, `CA-10` 경계 안으로 판정, provenance 필드 누락.

### R1 (반증)

## 반론 1
- **공격 대상**: P1 제안 1과 P2 제안 1의 "provenance-gated 1-fold timing probe가 4 GPU-hour 내 350-fold paired comparison 가능 여부를 판별하는 최소 게이트이며, 이 게이트 없이는 다른 제안이 verified가 될 수 없다"는 주장.
- **더 단순한 설명**: 과거 여섯 러너가 같은 7-branch로 수렴한 관측은 러너가 YAML 설정을 넘기지 않아 폴백에 걸린 단일 실행 경로 결함으로 설명된다. 저장 산출물의 귀속 불가는 provenance 부재로 설명되지만, 새 실행은 provenance 필드를 쓰는 즉시 귀속된다. D-050 정정은 YAML 경로가 정상이고 공식 7-branch 재현이 이미 성공했다고 실측했으나, unverified-hold 유지의 이유는 저장 산출물 provenance 부재와 러너 폴백이지, 새 실행의 구성 통제 불가능이 아니다. 1-fold timing은 이 관측을 설명하는 데 필요한 추가 변수가 아니다.
- **무엇이 참이면 이 주장이 무너지는가**: fold 적합 1회 wall time은 문서에 없으며, 7 과제 × 1 fold의 시간이 350-fold 총 시간을 대표한다는 근거도 문서에 없다. context 슬라이드 수는 90~261로 fold 간에 다르므로, 1-fold timing이 과제·fold 이질성을 대표하지 못하면 예산 판정은 판별 불가다. 또한 provenance writer가 실제로 산출물에 필드를 쓰기 전에는 1-fold 산출물 자체가 verified가 아니므로, 이 게이트는 자기 전제(provenance 복구)를 순환적으로 요구한다. 1-fold probe가 4h를 초과하지 않아도 350-fold가 4h를 초과할 수 있고, 그 반대도 가능하다.
- **이 반론의 한계**: fold별 계산 비용이 slide 수에 거의 선형이고, 7 과제 × 1 fold가 그 선형 계수를 안정적으로 추정하며, provenance writer가 이미 구현되어 1-fold 실행에 필드를 쓸 수 있다면, 이 반론은 틀린다.

## 반론 2
- **공격 대상**: P1 제안 2의 "seed-averaged closed-form margin이 seed 뽑기를 줄여 +0.0030 근처 후보를 판별 가능하게 한다"는 주장.
- **더 단순한 설명**: 동일 구성 seed 재적합에서 std 0.0021~0.0083이 나온 관측은 closed-form ridge의 적합 오차라기보다, bag/draw·PCA 초기화·subsample 등 upstream stochastic step이 margin에 전이된 결과로 설명된다. seed 평균은 fold의 단일 진실 margin을 더 잘 추정하는 것이 아니라, 여러 stochastic representation의 평균이라는 새 estimand를 만든다. seed averaging이 현재 파이프라인의 어떤 불확실성을 줄이는지는 기전 미확인이다.
- **무엇이 참이면 이 주장이 무너지는가**: paired Δ에서 baseline과 candidate가 같은 seed를 공유하면 seed 주효과는 부분적으로 상쇄될 수 있으므로, seed averaging의 실제 이득은 fold×seed 잔차에 달려 있다. 현행 7-branch의 seed 주효과와 fold×seed 잔차 분산분해는 문서에 없으며, seed 변동 원천도 문서에 없다. fold×seed 잔차가 지배하면 R을 늘려도 paired Δ 불확실성이 충분히 줄지 않아 판별 불가가 남는다. RU-97 GF 공간에서 m32_raw의 fold×seed 잔차가 seed 주효과보다 컸다는 실측은 이 위험의 사례지만, 현행 7-branch로 일반화할 근거는 없다. 적합 분산을 포함한 승격 판정 절차가 미승인 상태이므로, seed-averaged Δ를 승격 근거로 쓰면 실행 무효다.
- **이 반론의 한계**: seed 변동이 독립·무편향 적합 오차이고, R×350-fold가 4h 안에 가능하며, seed-averaged estimand와 적합 분산 보고가 승격 절차로 승인된다면, seed averaging은 noise floor를 줄일 수 있다.

## 반론 3
- **공격 대상**: P1 제안 3의 "fold-wise rank 변환이 raw margin의 scale 문제와 information duplication을 가르는 진단이며, CA-08 경계 밖의 단일 사전 등록 probe로 유효하다"는 주장.
- **더 단순한 설명**: effective rank 3.52/7은 branch margin의 scale 차이보다는, 7개 branch가 같은 patch feature와 유사한 closed-form readout에서 나온 margin이라 선형적으로 중복된다는 설명으로 더 단순하다. scale이 아니라 shared feature basis가 원인이라면 rank 변환은 중복을 줄이지 못한다.
- **무엇이 참이면 이 주장이 무너지는가**: rank 변환은 monotonic nonlinear transform이므로, scale 문제가 없어도 effective rank를 바꿀 수 있다. 따라서 rank margin의 effective rank 변화만으로 "scale-driven 중복"을 입증하지 못한다. trimmed mean은 이미 최저·최고 1개를 절사하므로 rank보다 약한 scale robustness를 가진다. branch별 기여도는 문서에 없으므로, 어떤 branch가 scale로 집계를 지배하는지도 알 수 없다. magnitude에 신호가 있으면 rank 변환은 정보 손실이 되어 Δ를 악화시키고, 이 제안의 전제는 반박된다. CA-08은 기존 margin 위의 결합 함수 반복을 경계 안으로 두므로, rank probe가 사전 등록·단일 후보·선택 이력 기록을 완벽히 지키지 않으면 실행 무효다.
- **이 반론의 한계**: branch margin의 판별 정보가 ordering에 있고, scale은 nuisance이며, rank 변환 전후의 correlation/eff.rank 변화가 scale normalization과 일치한다면, 이 반론은 약해진다.

## 반론 4
- **공격 대상**: P2 제안 2의 "branch별 seed variance에 inverse-variance 가중하면 trimmed mean보다 최종 분산을 줄이고 판정을 개선한다"는 주장.
- **더 단순한 설명**: seed variance는 적합 불안정성의 대수일 뿐, branch의 정확도나 정보량을 측정하지 않는다. low variance branch는 stochastic step에 덜 민감하거나 margin이 평탄해서일 수 있다. 따라서 가중은 "더 정확한 branch"가 아니라 "더 안정한 branch"를 높일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: inverse-variance 가중의 표준 전제는 추정치들이 독립적이고 무편향이라는 것이다. 그러나 공식 7-branch의 유효 랭크는 3.52/7로 중복이 크므로 독립성 전제는 약하다. 상관된 추정치에서 inverse variance만 쓰면 최적 가중이 될 수 없고, K가 작으면 variance 추정 자체가 불안정하다. per-fold variance로 가중하면 Primary 7 데이터에 적응한 가중이 되어 CA-08/Oracle 우려와 충돌할 수 있다. zero-variance guard는 성능 조회 전에 고정해도 임의적일 수 있다. final macro seed sd가 줄어도 paired Δ의 과제 군집 CI가 0을 포함하면 판별 불가다. AUROC 개선으로 이어지는 기전은 미확인이다.
- **이 반론의 한계**: seed variance가 독립 error variance의 신뢰 가능한 proxy이고, K가 충분히 크며, 가중 규칙이 사전 등록·cross-fit되어 평가 fold에 적응하지 않는다면, 분산 감소는 가능할 수 있다.

## 반론 5
- **공격 대상**: P2 제안 3의 "LID 8번째 branch의 gate ① 통과가 effective rank 3.52가 구조 상한이 아니라 아직 다른 기하를 재는 branch를 못 만들었다는 해석을 지지한다"는 주장.
- **더 단순한 설명**: gate ①은 라벨 무관 직교성만 재므로, LID가 기존 margin과 직교하더라도 과제 판별에 무관한 기하를 읽을 수 있다. effective rank가 낮은 관측은 patch feature 표현의 판별 자유도 자체가 제한되어 있다는 설명으로도 가능하다. LID는 그 제한을 넘지 못하고, 단지 다른 무관한 차원을 추가할 뿐이다.
- **무엇이 참이면 이 주장이 무너지는가**: RU-85 H에서 LID 추정기는 부분추출 크기에 민감했다(|r|(m=4096, m=1024)=0.314). full-token 또는 deterministic subsampling이 그 민감성을 해소한다는 증거는 문서에 없다. full-token LID의 350-fold 계산 비용도 fold time 미측정으로 4h 내 가능 여부를 알 수 없다. gate ①을 소수 fold에서 측정하면 correlation/eff.rank가 대표성을 잃는다. gate ① 통과가 보여줄 수 있는 것은 8-branch에서 eff.rank ≥ 4.02 가능이라는 제한적 사실뿐이며, "아직 못 만들었다"는 일반적 해석을 입증하지 못한다. 만약 LID 구현이 CA-10의 로버스트화·차원 확대 변형으로 환원되면 실행 무효다.
- **이 반론의 한계**: LID margin이 full-token에서 seed/subsample 안정적이고, 350-fold gate ①이 4h 안에 가능하며, gate ① 통과 후 gate ②/§4에서도 정보량이 확인된다면, 이 반론은 틀린다.

## 반론 6
- **공격 대상**: P1·P2 전체의 "provenance/timing, seed averaging, rank, LID 중 4 GPU-hour 안에 판별 가능한 후보가 있다"는 주장.
- **더 단순한 설명**: 현재 상태 문서는 fold 적합 시간이 측정되지 않았고, 계측 복구 전에는 게이트·승격 비교를 verified로 수행할 수 없다고 한다. 따라서 4h 안에 판별되는 것은 성능 개선이 아니라 계측 복구와 timing일 수 있다. 이는 최근 hold-out 측정 기준 ABMIL과의 거리(−0.0294)를 줄이지 않는다.
- **무엇이 참이면 이 주장이 무너지는가**: 1-fold timing이 총 350-fold 또는 350×K fit 시간을 신뢰할 수 있게 추정하지 못하면, seed ensemble/rank/LID의 4h 판별성은 없다. provenance writer 구현·테스트에 4h가 소모되면, 어떤 모델 후보도 Primary 7 paired Δ로 판정되지 못한다. fit variance 축은 +0.0030 근처 후보의 판정을 돕지만, ABMIL을 넘는 성능 mean 개선은 아니다.
- **이 반론의 한계**: 1-fold timing이 선형적으로 확장 가능하고, provenance writer가 이미 동작하며, 350-fold 또는 350×K fit이 4h 안에 실측된다면, 해당 후보는 판별 가능할 수 있다.

### A1 (감사)

## 점검 1: 1-fold timing probe의 fold 대표성
- **확인하려는 것**: 7 과제 × 1 fold의 wall time이 350-fold(또는 350×K seed)의 총 시간을 대표하는지. fold 1의 slide 수·patch 수가 50 fold의 중앙값·평균과 일치하는지.
- **문서에서 확인되는가**: 문서에 없음. fold별 slide 수·patch 수의 분포, fold 1의 계산 비용 대표성에 대한 실측이 없다. PROJECT.md §4.1에서 "fold는 과제 내에서 독립이 아니다"라고 했지만, fold별 계산 비용의 분포는 언급되지 않는다.
- **샌다면 나타날 증상**: fold 1이 anomalously fast(예: slide 수 적음)이면 350-fold가 4 GPU-hour 안에 가능하다고 판단한 뒤 중간에 중단. fold 1이 anomalously slow이면 실행 가능하다고 판단한 뒤 4 GPU-hour 초과. 1-fold timing이 350-fold의 총 시간을 대표하지 못하면 예산 판단이 `판별 불가`.
- **확인 방법**: fold manifest에서 7 과제 × 50 fold의 slide 수·patch 수 분포를 계산. fold 1이 중앙값·평균에서 벗어나는지 확인. fold manifest 파일 경로는 문서에 없음. 1-fold timing을 fold 1뿐 아니라 최소 3개 fold(예: fold 1, 25, 50)에서 측정하여 fold 간 변동성을 확인.

## 점검 2: Provenance 필드의 실제 실행 반영
- **확인하려는 것**: provenance writer가 기록한 branch list·가중치·코드 해시가 실제 실행에 반영된 값과 일치하는지. D-050에서 "src/가 환경변수를 하나도 읽지 않아 실행 스크립트의 모든 ICF_* export가 무효"였고, 정정에서 "YAML 경로는 정상"이라고 했으므로, provenance writer가 YAML에서 읽은 값을 기록하는지, 실행 스크립트가 export한 값을 기록하는지.
- **문서에서 확인되는가**: 모순됨. D-050 원문은 "재현할 경로가 현재 없다"고 했고, 정정은 "공식 7-branch 재현은 이미 성공했다(macro 0.6226 vs 공식 0.6227)"고 했다. provenance writer의 동작 방식(YAML 직접 읽기 vs 환경변수 읽기)은 문서에 명시되지 않는다.
- **샌다면 나타날 증상**: provenance에 5-branch로 기록되지만 실제는 7-branch로 실행됨(또는 그 반대). 코드 해시가 src/ 전체가 아니라 실행 스크립트만 포함하면, src/ 변경이 provenance에 반영되지 않아 과거 산출물을 특정 코드 버전으로 귀속시킬 수 없음.
- **확인 방법**: (1) `src/`의 config loading 경로 코드 확인 — YAML 파일을 직접 읽는지, 환경변수를 읽는지. (2) 합성 데이터에서 YAML의 branch 가중치를 1.0→0.0으로 바꾸고 margin 변화 확인(D-050 정정에서 이미 수행, max|diff|=2.525e-01). (3) provenance writer가 기록하는 코드 해시의 범위(src/ 전체, 실행 스크립트, config YAML)를 코드에서 확인. (4) `scripts/eval_v121.sh` 등 여섯 러너가 YAML 경로를 실제로 사용하는지 확인.

## 점검 3: Seed averaging의 estimand 변경
- **확인하려는 것**: seed-averaged margin의 paired Δ가 PROJECT.md §4.1의 estimand("Primary 7의 고정 fold 분포에서 후보와 기준선의 대응 차이 Δ의 기댓값", 표본 단위 fold, N=350)와 동일한지. seed averaging이 estimand를 "seed 평균 margin의 fold별 대응 차이"로 변경하는지.
- **문서에서 확인되는가**: 문서에 없음. seed averaging이 estimand를 변경하는지에 대한 논의가 없다. PROJECT.md §4.1은 단일 fit margin의 paired Δ를 estimand로 정의한다.
- **샌다면 나타날 증상**: seed-averaged Δ가 원래 estimand와 다르면, +0.0030 승격 기준이 seed-averaged estimand에 적용되는지 원래 estimand에 적용되는지 불분명. 승격 판정이 잘못된 estimand에 기반. 예: 원래 estimand에서 Δ=+0.002(승격 실패)인데 seed-averaged estimand에서 Δ=+0.004(승격 통과)이면, 어떤 estimand가 정당한지 판정 불가.
- **확인 방법**: PROJECT.md §4.1의 estimand 정의와 seed-averaged margin의 paired Δ 정의를 비교. seed averaging이 slide-level margin에서 이루어지는지, fold-level AUROC에서 이루어지는지 확인. slide-level seed 평균 후 fold-level AUROC를 계산하면 estimand가 "seed 평균 slide margin의 fold별 AUROC 대응 차이"가 되고, fold-level AUROC를 seed별로 계산 후 평균하면 "seed별 fold AUROC의 평균 대응 차이"가 된다. 둘 다 원래 estimand와 다를 수 있다.

## 점검 4: Fold×seed 잔차의 seed averaging 무력화
- **확인하려는 것**: seed averaging이 fold×seed 잔차를 줄이는지. RU-97에서 m32_raw의 fold×seed 잔차가 23.9%로 seed 주효과보다 컸고 "fold를 더 모아도 줄지 않는다"고 실측됐다. 현행 7-branch에서 fold×seed 잔차 비율이 측정된 적이 있는지.
- **문서에서 확인되는가**: 확인됨(RU-97 GF 공간에서 fold×seed 잔차 23.9% 실측). 현행 7-branch에서 fold×seed 잔차 비율은 문서에 없음.
- **샌다면 나타날 증상**: fold×seed 잔차가 지배하면 R을 늘려도 seed-averaged Δ의 변동이 충분히 줄지 않아 판별 불가가 남음. P1-2와 P2-2 모두 "fold×seed 잔차가 지배하면 R을 늘려도 noise floor가 충분히 줄지 않아 판별 불가가 남는다"고 실패 조건으로 인정하지만, fold×seed 잔차 비율을 실행 전에 측정할 방법이 1-fold timing probe에는 없다.
- **확인 방법**: RU-97 보고서 §8~§10의 분산분해 확인. 현행 7-branch에서 fold×seed 잔차 비율을 측정하려면 최소 2 seed × 350 fold fit이 필요하며, 이는 4 GPU-hour 예산에 포함해야 한다. 1-fold timing probe로는 fold×seed 잔차를 측정할 수 없다.

## 점검 5: P2-2 fit-variance 가중의 CA-08 경계
- **확인하려는 것**: fit-variance weighted aggregation이 CA-08 경계 밖("새 정보(예측분산·Context 신호 기하)를 사용하는 PSW·TGW")에 해당하는지, 경계 안("기존 브랜치 확률·마진 위의 결합 함수·부분집합 탐색")에 해당하는지.
- **문서에서 확인되는가**: 문서에 없음. fit-variance 가중이 PSW·TGW와 같은 범주인지 명시되지 않는다. D-022 경계 판정은 PSW·TGW를 비저촉으로 판정했지만, fit-variance 가중이 PSW·TGW와 동일한 "예측분산" 범주인지, 아니면 "기존 마진 위의 결합 함수 변형"인지 판정되지 않았다.
- **샌다면 나타날 증상**: CA-08 경계 안으로 해석되면 실행 무효. fit-variance 가중이 "기존 7 branch margin의 variance를 계산해서 가중을 정하는 것"이므로, "기존 브랜치 마진 위의 결합 함수 변형"에 해당할 수 있다. variance가 label-free이긴 하지만, CA-08의 경계 안은 "라벨 사용"을 요건으로 하지 않는다.
- **확인 방법**: closed_axes.md CA-08의 경계 안/경계 밖 정의와 D-022 경계 판정 확인. PSW·TGW의 정의(문서에 없음)와 fit-variance 가중을 비교. PSW·TGW가 "예측분산"을 사용하는지, fit-variance 가중이 "예측분산"에 해당하는지 판정. CA-08 경계 밖의 "새 정보"가 무엇인지 명시.

## 점검 6: P2-3 LID의 SCREEN_ONLY 경로 무효
- **확인하려는 것**: 게이트 ① 실행 경로(SCREEN_ONLY)가 현재 무효(D-050)인 상태에서 LID gate ①을 수행할 수 있는지. P2-3은 "제안 1의 provenance·timing 게이트 통과 후에만 verified로 수행"한다고 하지만, SCREEN_ONLY 경로 자체가 무효인지, provenance 복구만으로 복구되는지.
- **문서에서 확인되는가**: 확인됨. closed_axes.md §3-I에서 "게이트 ① 실행 경로(SCREEN_ONLY)가 현재 무효다(D-050). 계측이 복구된 뒤 라벨 무관·저비용 게이트 ①을 먼저 돌린다."라고 명시.
- **샌다면 나타날 증상**: provenance 복구만으로 SCREEN_ONLY 경로가 복구되지 않으면 LID gate ①을 수행할 수 없음. SCREEN_ONLY 경로가 무효인 원인이 provenance writer뿐인지, SCREEN_ONLY 자체의 로직 결함인지(예: margin 기록 경로, 앙상블 분리) 불분명.
- **확인 방법**: `scripts/analysis/branch_screen.py`의 SCREEN_ONLY 경로 코드 확인. D-050의 무효 원인이 provenance writer뿐인지, SCREEN_ONLY의 margin 기록·앙상블 분리 로직 결함인지 확인. `SCREEN_ONLY=1` 실행이 앙상블에 영향을 주지 않는지 확인(PROJECT.md §5 게이트 ①).

## 점검 7: Seed averaging의 기준선 재생성 의존성
- **확인하려는 것**: seed-averaged paired Δ를 계산하려면 기준선도 R-seed로 재생성해야 하는데, 현재 공식 기준선(0.6227)이 unverified-hold 상태이므로 재생성 가능한지. 기준선 재생성(350 × R fold fit)과 후보 재생성(350 × R fold fit)을 합산하여 4 GPU-hour 안에 가능한지.
- **문서에서 확인되는가**: 확인됨. PROJECT.md §3.2에서 "이 절의 값은 unverified-hold다"라고 했고, 정정에서 "공식 7-branch 재현은 이미 성공했다(macro 0.6226 vs 공식 0.6227)"라고 했다. 즉, 단일 fit 기준선 재생성은 가능하지만, R-seed 기준선 재생성은 4 GPU-hour 안에 가능한지 미확인.
- **샌다면 나타날 증상**: 기준선 재생성이 4 GPU-hour 예산에 포함되지 않으면, 후보 R-seed 실행만 수행되고 기준선은 단일 fit으로 남아 paired Δ가 불균형. 기준선 단일 fit의 seed 변동이 paired Δ에 편향을 넣을 수 있다.
- **확인 방법**: 1-fold timing probe에서 기준선 1-fold fit 시간과 후보 1-fold fit 시간을 각각 측정. 350 × R × 2(기준선+후보) fold fit이 4 GPU-hour 안에 가능한지 계산. R=8이면 350 × 8 × 2 = 5600 fold fit. 1-fold timing이 4 GPU-hour / 5600 = 2.57초/fold 미만이어야 한다. fold 적합 1회의 소요 시간은 문서에 없음.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

1. **T1**: 수용하고 수정. T1의 지적은 근거가 있다. fold 적합 1회 시간은 문서에 없고, 현행 7-branch의 seed 주효과·fold×seed 잔차 분산분해도 문서에 없으며, 유효 랭크 `3.52/7`은 branch 간 의존성이 크므로 variance만 쓰는 inverse-variance 가중의 독립성 전제가 약하다. RU-97의 fold×seed 잔차 `23.9%`는 GF 탐색 공간의 관측이지 현행 7-branch가 아니다. 따라서 P1의 1-fold timing은 350-fold 예산의 대표값으로 쓰지 않고, P2의 seed averaging은 estimand 사전 선언과 분산분해 측정 없이는 승격 근거로 쓰지 않으며, P1-3 rank probe는 effective rank 변화만으로 scale 중복과 정보 중복을 구분하지 못한다고 수정한다.

2. **M1-1**: 수용하고 수정. fold 적합 1회 시간 미실측, provenance writer 결함, 1-fold 대표성 미실측은 모두 문서와 일치한다. `D-050` 이후 저장 산출물의 구성 귀속이 불가하고, `current_status.md`는 계측 복구 전 verified 게이트·승격 비교가 불가하다고 한다. 따라서 7 과제 × 1 fold는 smoke test로만 쓰고, fold manifest의 slide·patch 분포와 최소 다중 fold timing이 없으면 350-fold 가능 여부는 `판별 불가`로 둔다. provenance writer가 필드를 실제로 쓰지 못하면 산출물은 `실행 무효`다.

3. **M1-2**: 수용하고 수정. 350×R fit 비용은 fold time 미실측으로 판정 불가이며, RU-97의 분산분해는 GF 공간이지 현행 7-branch가 아니다. 또한 `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 사용자 승인 전이라고 한다. 따라서 seed averaging은 R을 timing 후에 고정하고, 기준선과 후보 양쪽을 같은 R-seed 프로토콜로 재생성해야 하며, 승인 전에는 승격 근거로 쓰지 않는다. 예산이 R=1만 허용하면 paired seed-averaged 비교는 `실행 무효`다.

4. **M1-3**: 수용하고 수정. `CA-08`이 fold-wise rank aggregation을 명시적으로 경계 밖으로 판정하지는 않았다. `CA-08` 재개 조건은 질문·예산·선택 이력 명시와 유효한 평가 설계를 요구하므로, rank probe는 단일 후보·사전 등록·변형 금지·선택 이력 기록을 지킨 제한된 결합 연구로만 수행한다. 만약 경계 안으로 판정되면 `실행 무효`다. rank 변환은 effective rank `3.52`의 원인을 가르는 진단일 뿐, 성능 개선 기전이 입증된 것이 아니다.

5. **M1-4**: 수용하고 수정. P1-1과 P2-1은 모두 7 과제 × 1 fold timing·provenance smoke와 4 GPU-hour 예산 게이트를 요구하므로 동일 실험의 재진술이다. 두 항목을 하나의 provenance·timing gate로 통합하고, 중복 실행을 하지 않는다. 이 통합 게이트도 1-fold만으로 350-fold 또는 350×K seed 실행을 대표한다고 주장하지 않는다.

6. **M1-5**: 수용하고 수정. fit-variance 가중이 `CA-08`의 “예측분산”에 해당하는지 문서에 명시되지 않았고, 예측분산이 branch margin의 적합 분산을 대표하는지도 미확인이다. 또한 유효 랭크 `3.52/7`은 branch가 상관되어 있어 variance만 쓰는 가중이 최적이라는 전제를 약화시킨다. 따라서 P2-2는 성능 개선 제안이 아니라 fit-variance 보고 또는 사전 선언된 covariance-aware 규칙으로 낮추고, 라벨·AUROC를 variance 계산에 쓰거나 결과 보고 후 K·가중을 바꾸면 `실행 무효`로 둔다.

7. **M1-6**: 수용하고 수정. LID는 `CA-10`의 경계 밖으로 등재돼 있으나, `RU-85` H에서 부분추출 민감성 `|r|(m=4096, m=1024)=0.314`가 실측됐고, full-token 또는 deterministic subsampling이 이를 해소한다는 근거는 문서에 없다. 슬라이드당 token 수와 k-NN 계산 비용도 미실측이므로 4 GPU-hour 내 350-fold 가능성은 `판별 불가`다. 또한 `SCREEN_ONLY` 게이트 경로가 `D-050`으로 무효이므로, 계측 복구 전 LID gate ①은 verified가 될 수 없다. LID는 먼저 안정성·비용 smoke를 거쳐야 하며, gate ① 통과는 “eff.rank 3.52가 구조 상한이 아니다”를 입증하지 못하고 8-branch에서 `eff.rank ≥ 4.02` 가능이라는 제한적 사실만 보여준다.

8. **R1-1**: 반박. R1의 “과거 문제는 러너 폴백 하나로 설명되므로 1-fold timing이 불필요하다”는 주장은 근거가 부족하다. fold 적합 1회 시간은 문서에 없고, `current_status.md`는 계측 복구 전 verified 게이트·승격 비교가 불가하다고 하며, context 슬라이드 수는 90~261로 fold 간 계산 비용 이질성이 가능하다. 따라서 새 실행의 예산 판단과 구성 귀속을 위해 provenance·timing gate은 여전히 필요하다. 다만 1-fold가 350-fold 전체를 대표한다는 주장은 철회하고, fold manifest와 다중 fold timing으로 수정한다.

9. **R1-2**: 수용하고 수정. seed averaging이 같은 fold·task·branch margin의 독립적 무작위 오차를 평균한다는 전제는 문서에 없다. seed 변동의 원천이 무작위 표본, PCA, bag/draw, 수치 경로 중 무엇인지도 문서에 없으므로 기전 미확인이다. 또한 seed-averaged margin의 paired Δ는 `PROJECT.md` §4.1의 단일 fit margin estimand와 다를 수 있다. 따라서 seed averaging은 estimand을 사전 선언하고, fold×seed 잔차를 측정할 때까지 승격 근거로 쓰지 않으며, 적합 분산 포함 승격 절차가 승인되기 전에는 판정을 보류한다.

10. **R1-3**: 수용하고 수정. effective rank `3.52/7`이 scale 차이 때문인지 shared feature basis 때문인지 문서로 구분할 수 없다. rank 변환은 monotonic nonlinear transform이므로 scale 문제가 없어도 effective rank를 바꿀 수 있고, branch별 기여도는 문서에 없다. 따라서 rank probe는 raw margin, scale-normalized margin, rank margin의 label-free 상관·effective rank를 비교하는 진단으로만 쓰고, rank effective rank 상승만으로 scale-driven 중복을 입증한다고 주장하지 않는다. `CA-08` 경계가 명시적으로 결정되지 않으면 사전 등록 제한 연구로만 수행한다.

11. **R1-4**: 수용하고 수정. inverse-variance 가중의 표준 전제는 독립적이고 무편향인 추정치인데, 유효 랭크 `3.52/7`은 branch 간 의존성이 크다는 실측이다. seed variance가 branch의 정확도나 정보량이 아니라 안정성만을 반영할 수 있고, K가 작으면 variance 추정 자체가 불안정하다. 따라서 P2-2는 “trimmed mean보다 성능이 낫다”는 기전으로 쓰지 않고, fit-variance 보고 또는 사전 선언된 covariance-aware 규칙으로 제한한다. per-fold variance가 Primary 7에 적응해 `CA-08`/Oracle 우려와 충돌하면 `실행 무효`로 둔다.

12. **R1-5**: 수용하고 수정. gate ①은 라벨 무관 직교성만 재므로, LID가 기존 margin과 직교하더라도 과제 판별에 무관한 기하일 수 있다. RU-85 H의 부분추출 불안정, full-token 비용 미실측, `CA-10` 관련 사용자 판정 대기 모두 LID의 4h 내 verified 실행을 막는다. 따라서 LID probe는 gate ① 통과만으로 “아직 다른 기하를 재는 branch를 못 만들었다”는 일반 해석을 입증하지 못한다고 수정하고, 먼저 안정성·비용 smoke와 경계 결정을 전제로 한다.

13. **R1-6**: 수용하고 수정. 4 GPU-hour 안에 먼저 판별될 수 있는 것이 성능 개선이 아니라 계측 복구·timing일 수 있다는 지적은 유효하다. fold time 미실측과 `D-050`의 unverified-hold 상태 때문에, 이 회차에서 직접 ABMIL 거리 `−0.0294`를 줄인다고 주장할 수 없다. 그러나 provenance·timing 복구가 없으면 이후 모든 Primary 7 paired Δ는 `verified`가 될 수 없으므로, 이 항목은 성능 개선이 아니라 성능 판정을 가능하게 하는 전제 복구로 재정의한다.

14. **A1-1**: 수용하고 수정. fold 1의 slide 수·patch 수가 50 fold를 대표하는지는 문서에 없다. context 슬라이드 수는 90~261로 fold 간에 다르므로, 1-fold timing만으로 350-fold 또는 350×K seed 예산을 판단하면 `판별 불가`가 될 수 있다. 따라서 fold manifest에서 slide·patch 분포를 확인하고, 가능하면 fold 1·25·50 등 다중 fold timing을 측정해야 한다. 이 정보가 없으면 4h 예산 내 전체 paired comparison은 `판별 불가`로 둔다.

15. **A1-2**: 수용하고 수정. provenance writer가 YAML에서 읽은 값을 기록하는지, 실행 스크립트가 export한 값을 기록하는지, 코드 해시가 `src/` 전체를 포함하는지는 문서에 명시되지 않았다. `D-050` 정정은 YAML 경로가 정상이라고 확인했지만, provenance 필드가 실제 실행에 반영됐는지 보장하지는 않는다. 따라서 provenance writer는 runtime branch list·가중치·코드 해시 범위·config hash·fold manifest hash·seed·run ID를 기록하고, 합성 데이터에서 branch 가중치 변경 시 margin 변화와 provenance 일치를 검증해야 한다. 불일치가 있으면 산출물은 `실행 무효`다.

16. **A1-3**: 수용하고 수정. seed-averaged margin의 paired Δ가 `PROJECT.md` §4.1의 estimand과 동일한지는 문서에 없다. slide-level margin을 seed 평균한 뒤 fold AUROC를 계산하는 방식과, fold AUROC를 seed별로 계산한 뒤 평균하는 방식은 다른 estimand를 만들 수 있다. 따라서 seed averaging을 쓰려면 estimand을 사전 선언하고, 기존 단일 fit margin estimand과 비교해야 하며, estimand가 바뀌면 `+0.0030` 승격 기준 적용은 사용자 승인 전까지 보류한다.

17. **A1-4**: 수용하고 수정. 현행 7-branch의 seed 주효과와 fold×seed 잔차 분산분해는 문서에 없다. RU-97의 `m32_raw` fold×seed 잔차 `23.9%`는 GF 공간의 실측이지 현행 7-branch로 일반화할 수 없다. 따라서 seed averaging이 noise floor를 줄이는지 실행 전에 판단할 수 없다. 예산이 허용하면 2 seed 이상으로 분산분해 프로브를 먼저 수행하고, 허용하지 않으면 seed averaging의 효과는 `판별 불가`로 둔다.

18. **A1-5**: 판별 불가. `CA-08` 문서에는 “새 정보(예측분산·Context 신호 기하)를 사용하는 PSW·TGW”가 경계 밖으로 언급되지만, fit-variance weighted aggregation이 PSW·TGW와 동일한 “예측분산” 범주인지 명시되지 않았다. 따라서 이 경계는 문서만으로 확정할 수 없다. 운영적으로는 단일 사전 등록 제한 연구로만 수행하고, 경계 안으로 판정되면 `실행 무효`로 둔다. 사용자 또는 경계 판정이 없으면 `CA-08` 위반 여부를 `판별 불가`로 남긴다.

19. **A1-6**: 수용하고 수정. `closed_axes.md` §3-I은 `SCREEN_ONLY` 게이트 ① 실행 경로가 `D-050`으로 현재 무효라고 명시한다. 따라서 LID gate ①은 provenance·timing 복구 후에만 verified로 수행할 수 있다. 다만 provenance 복구만으로 `SCREEN_ONLY`가 복구되는지는 코드 확인이 필요하고, `SCREEN_ONLY`의 margin 기록·앙상블 분리 로직 결함도 함께 확인해야 한다. 복구 전 LID gate ① 결과는 `실행 무효` 또는 탐색 기록으로만 쓴다.

20. **A1-7**: 수용하고 수정. paired Δ는 기준선과 후보를 같은 프로토콜로 재생성해야 성립한다. 공식 기준선 `0.6227`은 `unverified-hold`이고, R-seed 기준선 재생성 비용은 350×R fit으로 후보와 합산하면 700×R fit이 된다. fold time 미실측이므로 이 비용이 4 GPU-hour 안에 들어가는지 문서로 판단할 수 없다. 따라서 seed-averaged paired Δ를 만들려면 기준선 R-seed 재생성을 예산에 포함해야 하며, 포함하지 않으면 paired Δ는 불균형 비교가 되어 `실행 무효`로 둔다.

### P2 (제안)

T1
- **결정: 수용하고 수정**  
  T1이 지적한 “현행 7-branch의 seed 주효과/fold×seed 잔차 분산분해가 문서에 없다”, “seed 재적합 변동의 원천이 문서에 없다”, “fold 적합 1회 시간이 문서에 없다”는 점에 동의한다. 따라서 제안 2의 seed averaging과 제안 3의 rank 변환은 성능 개선 기전이 아니라 **기전 미확인**의 탐지로만 남는다. 수정 내용: (1) seed averaging이 총 적합 분산을 줄인다고 주장하지 않고, seed sd와 가능하면 fold×seed 잔차를 함께 보고한다. (2) inverse-variance 가중은 branch 간 상관, 즉 유효 랭크 `3.52/7` 수준의 중복을 무시할 수 없으므로, 분산만 쓰는 가중은 독립성 전제가 약하다. (3) 1-fold timing은 350-fold 전체를 대표하는 증거가 아니라 예산 게이트일 뿐이며, 대표성을 검증할 수 없으면 350-fold 실행 가능 여부는 `판별 불가`로 둔다. (4) rank 변환 후 effective rank 변화만으로는 scale-driven 중복과 정보 중복을 구분하지 못하므로, 이 제안은 discriminator가 아니라 단일 사전 등록 집계 probe로 축소한다.

M1-실측 1: P1 제안 1
- **결정: 수용하고 수정**  
  fold 적합 1회 시간이 문서에 없으므로, 7 과제 × 1 fold probe가 4 GPU-hour 안에 끝나는지 사전에 확신할 수 없다. 또한 provenance writer가 실제로 산출물에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시·seed·run ID를 쓰는지 문서에 확인된 동작이 없다. 수정 내용: (1) provenance writer 복구가 probe의 선행 조건이다. writer가 필드를 쓰지 못하면 probe 결과도 `실행 무효`다. (2) 1-fold probe는 wall time뿐 아니라 fold별 slide 수·patch 수·context slide 수 등 계산 비용 대리변수도 함께 기록한다. (3) 1-fold만으로는 350-fold 예산을 확정할 수 없으므로, 1-fold 결과가 예산 한계에 가깝거나 fold 간 편차를 확인할 수 없으면 350-fold paired comparison 가능 여부는 `판별 불가`로 종료한다. (4) 가능하면 과제당 1 fold가 아니라 여러 fold를 측정하지만, 그 비용도 fold time 미실측 때문에 4 GPU-hour 게이트에 포함해야 한다.

M1-실측 2: P1 제안 2
- **결정: 수용하고 수정**  
  R=8은 `current_status.md`에 잠정적으로만 등장하며 승인 전이다. fold 적합 1회 시간이 문서에 없으므로, 350×R fit이 4 GPU-hour 안에 가능한지 알 수 없다. 또한 RU-97의 fold×seed 잔차 `23.9%`는 GF 탐색 공간의 관측이지 현행 7-branch의 분산분해가 아니다. 수정 내용: (1) R을 8로 고정하지 않고, 1-fold timing 결과와 기준선+후보 전체 fit 수에 따라 R을 축소한다. (2) R=2 미만이면 seed averaging 자체를 수행하지 않는다. R=2라도 fold×seed 잔차를 완전히 제거한다고 보지 않는다. (3) seed-averaged margin의 paired Δ는 기존 단일 적합 estimand와 동일하다고 주장하지 않는다. (4) 적합 분산을 포함한 승격 판정 절차가 사용자 승인 전이므로, seed-averaged Δ를 승격 근거로 쓰지 않는다. (5) seed sd가 감소하지 않거나 paired CI가 0을 포함하면 `판별 불가`로 둔다.

M1-실측 3: P1 제안 3
- **결정: 반박**  
  M1이 “CA-08 경계 미확정”이라 주장한 부분에 대해서는 반박한다. `closed_axes.md` CA-08의 경계 밖에는 “탐색임을 명시한 제한된 결합 연구”가 비저촉으로 명시되어 있고, 재개 조건도 “질문·예산·선택 이력을 명시하고, 성능 확증에는 선택·적응을 반영한 유효한 평가 설계를 갖춘다”는 것이다. P1 제안 3은 subset search, per-task selection, max picking을 금지하고 단일 rank 변환만 사전 등록하므로, CA-08 경계 안의 “반복” 또는 선택 편향 절차에 해당하지 않는다. 따라서 CA-08만으로 `실행 무효`라고 단정할 근거가 부족하다. 다만 M1의 나머지 지적은 유효하다. 350-fold 재생성 비용은 fold time 미실측 때문에 `판별 불가`이며, rank 변환 후 effective rank 변화만으로는 scale 문제와 정보 중복을 기계적으로 구분할 수 없다. 그래서 이 제안은 CA-08 통과를 전제로 하되, discriminative claim은 기전 미확인 상태로만 유지한다.

M1-실측 4: P2 제안 1
- **결정: 수용하고 수정**  
  P2 제안 1은 P1 제안 1과 사실상 같은 실행 게이트다. 따라서 별도 실험으로 분리하지 않고 P1 제안 1에 통합한다. 수정 내용: (1) 1-fold timing과 provenance smoke를 하나의 게이트로 수행한다. (2) 1-fold만으로는 350-fold 또는 350×K seed 실행의 예산을 확정할 수 없으므로, fold manifest 기반 비용 대리변수와 가능하면 추가 fold timing을 포함한다. (3) 대표성을 검증할 수 없으면 350-fold 또는 350×K 실행 가능 여부는 `판별 불가`로 둔다. (4) provenance 필드가 누락되면 모든 산출물은 `실행 무효`다.

M1-실측 5: P2 제안 2
- **결정: 수용하고 수정**  
  fit-variance 가중은 seed variance를 새 정보로 쓰되, branch 간 상관과 K가 작을 때의 variance 추정 불안정성을 무시하면 안 된다. 유효 랭크 `3.52/7`는 branch margin이 크게 중복됨을 보여, 독립 추정치에 대한 inverse-variance 가중의 표준 전제가 약하다. 수정 내용: (1) per-fold variance로 직접 가중하는 대신, 성능 조회 전에 고정된 pooled variance 또는 cross-fit된 variance 규칙을 쓴다. (2) K가 작으면 variance 자체가 불안정하므로, K가 사전 최소 기준을 만족하지 못하면 가중을 하지 않고 seed average만 보고한다. (3) 가중 규칙, zero-variance guard, K 선택은 모두 성능 조회 전에 고정한다. (4) variance 계산에 AUROC나 라벨을 쓰지 않는다. (5) final macro seed sd가 줄어도 paired Δ의 과제 군집 CI가 0을 포함하면 `판별 불가`다. (6) CA-08 경계는 M1-5와 A1-5에서 별도로 판별 불가로 두지만, 이 제안은 단일 사전 등록 제한 연구로만 수행한다.

M1-실측 6: P2 제안 3
- **결정: 수용하고 수정**  
  LID branch probe는 CA-10 경계 밖의 “국소 내재차원”으로 명시되어 있지만, RU-85 H에서 부분추출 크기에 따른 LID 마진 상관이 `|r|(m=4096, m=1024)=0.314`로 강건성 임계 0.8을 밑돌았으므로, full-token 또는 deterministic subsampling이 그 민감성을 해소한다는 증거는 문서에 없다. 또한 slide당 token 수, k-NN 계산 비용, 350-fold LID margin 계산 시간이 문서에 없어 4 GPU-hour 내 가능 여부를 알 수 없다. 수정 내용: (1) LID probe는 provenance·SCREEN_ONLY·timing 게이트 통과 후에만 `verified`로 수행한다. (2) 350-fold 전체 전에, 사전 등록된 두 가지 subsampling 크기에서 LID margin 상관을 재는 안정성 probe를 둔다. (3) 이 안정성 probe에서 상관이 여전히 낮으면 LID 추정기 변형은 `반박` 또는 `실행 무효`로 종료한다. (4) 350-fold LID margin 계산이 4 GPU-hour를 초과하면 `실행 무효`다. (5) gate ① 실패 시 AUROC를 조회하지 않는다.

R1-반론 1
- **결정: 반박**  
  R1의 “provenance만 있으면 1-fold timing이 추가 변수가 아니다”라는 주장은 근거가 부족하다. `current_status.md`와 연구 질문 모두 fold 적합 1회 시간이 측정되지 않았다고 하고, 예산은 4 GPU-hour로 고정되어 있다. 또한 `closed_axes.md` CA-04·CA-06에서 context slide 수가 90~261장으로 fold 간에 다르다고 실측되어 있으므로, 1-fold wall time이 350-fold 전체를 대표한다고 단정할 근거는 문서에 없다. provenance는 구성 귀속을 해결하지만, 예산 초과를 방지하지는 않는다. 따라서 1-fold timing은 350-fold 전체를 확증하는 것이 아니라, 4 GPU-hour 안에서 실행이 가능한지를 사전에 걸러내는 게이트로 필요하다. 다만 R1이 지적한 “1-fold가 대표하지 못할 수 있다”는 점은 유효하므로, 1-fold만으로는 350-fold 예산을 `verified`로 확정하지 않고, 대표성을 확인할 수 없으면 `판별 불가`로 둔다.

R1-반론 2
- **결정: 수용하고 수정**  
  seed 재적합 변동이 무작위 적합 오차가 아니라 bag/draw, PCA 초기화, 표본화, 수치 경로 등에서 오는 systematic variation일 수 있다는 지적은 유효하다. 현행 7-branch에서 seed 변동의 원천은 문서에 없다. 따라서 seed averaging이 현재 파이프라인의 어떤 불확실성을 줄이는지는 기전 미확인이다. 수정 내용: (1) seed-averaged margin은 “fold의 단일 진실 margin을 더 잘 추정한다”고 주장하지 않고, 여러 stochastic fit의 평균이라는 새 estimand로 취급한다. (2) 기준선과 후보는 같은 seed 프로토콜을 쓴다. (3) seed 주효과와 fold×seed 잔차를 분리할 수 있으면 보고한다. (4) fold×seed 잔차가 지배하면 R을 늘려도 `판별 불가`가 남는다고 명시한다. (5) 적합 분산을 포함한 승격 판정 절차가 승인되기 전에는 seed-averaged Δ를 승격 근거로 쓰지 않는다.

R1-반론 3
- **결정: 수용하고 수정**  
  rank 변환은 monotonic nonlinear transform이므로, scale 문제가 없어도 effective rank를 바꿀 수 있다. 따라서 rank margin의 effective rank 변화만으로 “scale-driven 중복”을 입증하지 못한다. 또한 branch별 기여도는 문서에 없어, 어떤 branch가 scale로 집계를 지배하는지 알 수 없다. 수정 내용: (1) P1 제안 3은 effective rank 3.52의 두 해석을 결정적으로 가르는 discriminator가 아니라, raw margin과 rank margin의 paired Δ·seed sd·상관 구조를 비교하는 단일 probe로 축소한다. (2) rank 변환 후 effective rank가 올라가도 paired Δ가 악화되면 magnitude에 신호가 있었다는 쪽으로 `반박` 증거를 둔다. (3) effective rank만 바뀌고 Δ가 0을 포함하면 `판별 불가`다. (4) 이 probe의 기전은 미확인이며, CA-08 통과를 전제로 한다.

R1-반론 4
- **결정: 수용하고 수정**  
  seed variance는 branch의 정확도나 정보량이 아니라 적합 불안정성의 대수일 수 있다. low variance branch가 더 정확한 것이 아니라 더 평탄하거나 stochastic step에 덜 민감할 수 있다. 또한 공식 7-branch의 유효 랭크 `3.52/7`는 추정치 간 독립성 전제가 약함을 보여, variance만 쓰는 inverse-variance 가중이 최적 가중이 되지 않을 수 있다. 수정 내용: (1) per-fold adaptive 가중을 피하고, 성능 조회 전에 고정된 pooled variance 또는 cross-fit variance 규칙을 쓴다. (2) K가 작으면 variance 추정이 불안정하므로, K가 사전 최소 기준을 만족하지 못하면 가중을 하지 않는다. (3) branch 간 상관을 함께 보고한다. (4) 가중이 특정 branch를 구조적으로 제거하거나 다수 과제를 악화시키면 “fit-variance 가중이 trimmed mean보다 낫다”는 가설은 `반박`된다. (5) final macro seed sd가 줄어도 paired Δ CI가 0을 포함하면 `판별 불가`다.

R1-반론 5
- **결정: 수용하고 수정**  
  gate ①은 라벨 무관 직교성만 재므로, LID가 기존 margin과 직교하더라도 과제 판별에 무관한 기하를 읽을 수 있다. 또한 RU-85 H에서 LID 추정기는 부분추출 크기에 민감했고, full-token이 그 민감성을 해소한다는 증거는 문서에 없다. 따라서 gate ① 통과만으로 “effective rank 3.52는 아직 못 만들었다는 해석”을 일반화할 수 없다. 수정 내용: (1) LID gate ① 통과의 의미는 8-branch에서 `eff.rank ≥ 4.02`가 가능했다는 제한적 사실로만 보고한다. (2) gate ① 통과 전에 subsampling 안정성 probe를 수행한다. (3) 350-fold LID margin 계산이 4 GPU-hour 안에 가능한지 fold time·token 수·k-NN 비용이 확인되기 전에는 `판별 불가`다. (4) LID가 CA-10의 모멘트 변형·로버스트화·차원 확대 변형으로 환원되면 `실행 무효`다.

R1-반론 6
- **결정: 수용하고 수정**  
  fold 적합 시간이 미실측이고 계측 복구 전에는 verified 비교가 불가능하므로, 4 GPU-hour 안에 판별되는 것이 성능 개선이 아니라 계측 복구와 timing일 수 있다. 수정 내용: (1) 실행 순서를 P1의 provenance·timing gate → 남은 예산에 따라 seed averaging 또는 rank probe 또는 LID probe 중 하나로 고정한다. (2) 1-fold timing 후 350-fold 또는 350×R fit이 불가능하면, 해당 후보의 Primary 7 paired Δ는 `실행 무효` 또는 `판별 불가`로 종료한다. (3) provenance writer 복구에 예산이 소모되면, 그 결과 자체를 “이번 4 GPU-hour에서는 성능 후보를 판정할 수 없었다”고 기록한다. (4) 이 경우에도 ABMIL과의 거리 `−0.0294`가 줄었다고 주장하지 않는다.

A1-점검 1
- **결정: 수용하고 수정**  
  fold 1의 wall time이 350-fold 전체를 대표하는지는 문서에 없다. fold별 slide 수·patch 수·context slide 수 분포도 문서에 없다. 수정 내용: (1) 1-fold timing과 함께 fold manifest에서 fold별 slide 수·patch 수·context slide 수 분포를 계산한다. (2) fold 1이 중앙값·평균에서 벗어나면 350-fold 예산 외삽은 `판별 불가`로 둔다. (3) 가능하면 과제당 fold 1뿐 아니라 추가 fold를 측정하지만, 추가 fold 비용도 4 GPU-hour 게이트에 포함한다. (4) 대표성을 확인할 수 없으면 1-fold timing은 예산 하한 또는 상한이 아니라 `판별 불가` 신호로만 쓴다.

A1-점검 2
- **결정: 수용하고 수정**  
  D-050 정정은 YAML 경로가 정상이고 공식 7-branch 재현이 성공했다고 확인했지만, provenance writer가 실제로 어떤 값을 기록하는지는 문서에 확정적으로 없다. `src/`가 환경변수를 읽지 않는다는 원문과 YAML 경로 정상이라는 정정은 서로 다른 실행 경로에 관한 것이므로, provenance writer가 YAML에서 읽은 값을 기록하는지, 실행 스크립트가 export한 값을 기록하는지는 별도 확인이 필요하다. 수정 내용: (1) config loading 코드와 provenance writer 코드를 검사한다. (2) provenance에 기록되는 코드 해시의 범위가 `src/` 전체, 실행 스크립트, config YAML 중 어디까지인지 확인한다. (3) 합성 데이터에서 branch 가중치를 바꾸고 margin과 provenance가 실제로 달라지는지 확인한다. (4) 이 확인이 완료되기 전에는 provenance attached 실행도 `verified`라고 부르지 않는다.

A1-점검 3
- **결정: 수용하고 수정**  
  `PROJECT.md` §4.1의 estimand는 Primary 7 고정 fold 분포에서 후보와 기준선의 대응 fold AUROC 차이의 기댓값이다. seed averaging을 slide-level margin에서 한 뒤 fold AUROC를 계산하면, estimand가 “seed 평균 slide margin에서 만든 fold AUROC의 대응 차이”가 된다. fold-level AUROC를 seed별로 계산한 뒤 평균하면 “seed별 fold AUROC의 평균 대응 차이”가 된다. 이 둘은 기존 단일 적합 estimand와 동일하다고 단정할 수 없다. 수정 내용: (1) seed averaging의 위치(slide margin 평균 vs fold AUROC 평균)를 사전 등록한다. (2) 어떤 estimand를 계산하는지 보고서에 명시한다. (3) 기존 단일 적합 paired Δ와 seed-averaged paired Δ를 혼동하지 않는다. (4) 적합 분산을 포함한 승격 판정 절차가 승인되기 전에는 seed-averaged estimand로 승격을 판단하지 않는다.

A1-점검 4
- **결정: 판별 불가**  
  현행 7-branch에서 seed 주효과와 fold×seed 잔차의 분산분해는 문서에 없다. RU-97의 `m32_raw` fold×seed 잔차 `23.9%`와 seed 주효과 `0~2.9%`는 GF 탐색 공간의 관측이지 현행 7-branch로 일반화할 근거가 없다. 1-fold timing probe만으로는 fold×seed 잔차를 측정할 수 없다. fold×seed 잔차를 측정하려면 최소 2 seed × 동일 fold 재적합이 필요하고, 그 비용도 fold time 미실측 때문에 4 GPU-hour 안에 가능한지 알 수 없다. 따라서 seed averaging이 현행 7-branch의 총 적합 분산을 충분히 줄이는지는 현재 `판별 불가`다.

A1-점검 5
- **결정: 판별 불가**  
  fit-variance weighted aggregation이 CA-08 경계 밖의 “새 정보(예측분산·Context 신호 기하)를 사용하는 PSW·TGW”에 해당하는지, 경계 안의 “기존 브랜치 마진 위의 결합 함수 변형”에 해당하는지는 문서에 확정적으로 명시되어 있지 않다. PSW·TGW의 정의를 문서에서 직접 확인하지도 못했다. 따라서 이 경계는 현재 `판별 불가`로 둔다. 경계가 비저촉으로 확정되기 전에는 P2 제안 2를 `verified` 승격 후보로 수행하지 않는다. 만약 비저촉으로 확정되더라도, 단일 사전 등록 제한 연구, 성능 조회 전 K·가중 고정, AUROC/라벨 사용 금지, 선택 이력 기록을 함께 지켜야 한다.

A1-점검 6
- **결정: 수용하고 수정**  
  `closed_axes.md` §3-I에서 “게이트 ① 실행 경로(`SCREEN_ONLY`)가 현재 무효다(`D-050`). 계측이 복구된 뒤 라벨 무관·저비용 게이트 ①을 먼저 돌린다”고 되어 있으므로, LID gate ①은 SCREEN_ONLY 경로 복구가 선행되어야 한다. provenance writer 복구만으로 SCREEN_ONLY가 복구된다고 단정할 근거는 문서에 없다. 수정 내용: (1) `scripts/analysis/branch_screen.py`의 SCREEN_ONLY 경로가 margin만 기록하고 앙상블에 영향을 주지 않는지 코드 수준에서 확인한다. (2) D-050의 무효 원인이 provenance writer뿐인지, margin 기록·앙상블 분리 로직 결함인지 구분한다. (3) SCREEN_ONLY 경로가 복구되지 않으면 LID gate ①은 `verified`로 수행할 수 없다.

A1-점검 7
- **결정: 수용하고 수정**  
  seed-averaged paired Δ는 후보만 R-seed로 만들고 기준선을 단일 fit으로 두면, 기준선의 seed 변동이 paired Δ에 편향을 넣을 수 있다. 공식 7-branch 기준선 `0.6227`은 `unverified-hold`이며, D-050 정정에서 단일 fit 공식 7-branch 재현은 성공했지만 R-seed 기준선 재생성 비용은 문서에 없다. 수정 내용: (1) seed-averaged paired Δ를 만들려면 기준선과 후보를 같은 R-seed 프로토콜로 재생성한다. (2) 1-fold timing probe에서는 기준선 1-fold fit 시간과 후보 1-fold fit 시간을 각각 측정한다. (3) 350×R×2 fit이 4 GPU-hour 안에 불가능하면, seed-averaged 승격 비교는 `실행 무효` 또는 `판별 불가`로 종료한다. (4) 후보만 R-seed로 수행하고 기준선을 단일 fit으로 두면, 그 결과는 승격 근거가 아니라 탐색 기록으로만 둔다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)

- **(1) 유효 랭크 `3.52/7`는 현재 7개 margin의 중복을 실측한 값이지, label-free margin 공간의 구조 상한을 입증하는 값이 아니다.**  
  근거: `PROJECT.md` §3.2에서 `3.52/7`은 현행 7-branch의 실측 유효 랭크로만 기록된다. `closed_axes.md` `CA-09`는 형상 계열(`BD`·`SH`·`SJ`)이 유효 랭크를 올린 경로로 기록되면서도 “다른 경로의 불가능을 뜻하지 않는다”고 명시한다. `CA-10`은 형상 계열의 병목이 절사가 아니라 정보량이라고 하고, `RU-85` G·H는 AKS·LID 하위축이 아직 사용자 판정 대기 또는 추정기 민감성 문제로 남아 있음을 보여준다. 따라서 “서로 다른 것을 재는 branch를 아직 못 만들었을 수 있다”는 해석이 열려 있다는 점은 증거가 뒷받침한다. 다만 이것이 이미 존재함을 입증하는 것은 아니다.

- **(1) 두 해석을 가르는 관측은 성능 조회 전 label-free 게이트 ①이다.**  
  근거: `PROJECT.md` §5는 신규 branch 채택 게이트 ①으로 `max |r| > 0.6`이면 성능과 무관하게 기각하고, 8-branch 기준 `eff.rank ≥ 4.02`를 요구한다. 즉 후보 margin이 기존 7 margin과 직교하고, 편입 시 유효 랭크 효율을 비하락시키는지, 그리고 seed·subsample에서 안정한지를 성능 조회 전에 재야 한다. 다만 한 후보의 게이트 ① 통과는 “8-branch에서 `eff.rank ≥ 4.02`가 가능했다”는 제한적 사실만 보여줄 뿐, `3.52`가 구조 상한이 아님을 일반화하는 증거는 아니다.

- **(2) 집계 축은 “모든 집계 변경이 닫혔다”는 뜻이 아니라, 선택 편향 절차가 닫혔다.**  
  근거: `closed_axes.md` `CA-08`의 기각 기전은 §214의 선택 편향이며, “결합 함수 일반의 무효 기전”은 미확인이다. `CA-08` 경계 밖에는 “탐색임을 명시한 제한된 결합 연구”와 “새 정보(예측분산·Context 신호 기하)를 사용하는 `PSW`·`TGW`”가 비저촉으로 언급된다. `D-022`도 `PSW`·`TGW`를 비저촉으로 판정했다. 따라서 사전 등록·단일 후보·선택 이력 기록·성능 조회 후 변형 금지를 지킨 제한된 집계 probe는 닫힌 축 전체의 반복으로 단정할 근거가 없다.

- **(3) 적합 분산 감소는 성능 mean 개선과 다른 estimand 문제다.**  
  근거: `D-048` 이후 적합 분산 보고가 필수화됐다. `PROJECT.md` §4.1의 estimand는 Primary 7 고정 fold 분포에서 후보와 기준선의 대응 fold AUROC 차이의 기댓값이다. `RU-97`은 동일 구성에서 seed만 바꿔 재적합 시 `M32−M8` 단일 적합 `+0.0099`가 seed 42의 뽑기였고, 적합 4회 평균은 `+0.0003 [-0.0067, +0.0073]`로 0을 포함함을 실측했다. 따라서 분산 감소 수단은 `E[Δ]`를 올리는 것이 아니라 Δ의 SE·seed sd·판정 상태를 바꾸는 계측·추정 문제다.

- **(4) “아무도 건드리지 않은 축”은 단순 미고려라기보다 유효한 전제 결여로 막혀 있다.**  
  근거: `current_status.md`의 현재 목표는 계측 복구(provenance writer, `SH`/`SJ` control, 오염 검사 재설계, 단위 테스트)다. `D-050`은 공식 기준선 `0.6227`과 과거 승격 근거를 `unverified-hold`로 두고, 저장 산출물에 provenance가 없어 구성 귀속이 불가하다고 한다. fold 적합 1회 시간은 문서에 없다. `LSAK` 사양은 문서에 없다. 적합 분산을 포함한 승격 판정 절차는 사용자 미승인이다. `CA-R1`은 reopened이면서도 라벨 없는 선택 신호 부재로 활용 불가다. 따라서 미탐색 축이 존재하되, 착수 전제가 유효하게 막혀 있다는 해석이 증거에 의해 뒷받침된다.

- **(5) 현재 문서에서 4 GPU-hour 안에 `verified` paired Δ로 진행할 수 있다고 확정할 성능 후보는 없다.**  
  근거: fold 적합 1회 시간이 미실측이고, 1-fold가 350-fold 또는 350×R seed 실행을 대표하는지도 미실측이다. `D-050` 이후 provenance가 없으면 산출물을 특정 구성에 귀속시킬 수 없다. 따라서 4h 안에서 판별 가능한 것은 성능 개선 자체가 아니라, provenance·timing smoke의 통과/실패와 350-fold 재적합 가능 여부에 대한 추가 판별이다.

## 반박 (증거가 반박함)

- **“`3.52/7`만으로 branch 추가 방향이 막혔다”는 단정은 반박된다.**  
  근거: `CA-09`는 형상 계열이 유효 랭크를 올린 경로로 기록되고 “다른 경로의 불가능을 뜻하지 않는다”고 한다. `CA-10`은 형상 하위축 전체를 닫은 것이 아니라 정보량 병목을 지적한다. `RU-85` G·H는 AKS·LID 하위축이 아직 미확정 또는 추정기 문제일 수 있음을 보여준다. 따라서 `3.52` 자체는 구조 상한 결론을 지지하지 못한다.

- **단일 적합의 `+0.0099`를 실제 개선으로 쓰는 것은 반박된다.**  
  근거: `RU-97`에서 `M32−M8`의 단일 적합 `+0.0099`는 seed 42가 해당 구성 평균보다 1.5 sd 높았던 뽑기였고, 적합 4회 평균은 `+0.0003 [-0.0067, +0.0073]`로 0을 포함한다. 동일 구성 seed 재적합의 표준편차 `0.0021~0.0083`은 승격 임계 `+0.0030`과 같은 크기대다.

- **과거 branch 조합 비교를 근거로 쓰는 것은 반박된다.**  
  근거: `D-050` 정정에 따르면 `scripts/eval_v121.sh` 등 여섯 러너가 학습 설정을 넘겨 `eval_seal_tasks.sh` 폴백에 걸려 arm이 다른 실행들이 전부 같은 7-branch 설정으로 수렴했다. 저장된 예측 파일에는 branch list·가중치·코드 해시가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다. 따라서 과거 조합 비교는 구성 차이가 없는 실행들의 비교일 수 있어 근거로 쓸 수 없다.

- **“seed averaging이 fold×seed 잔차까지 줄인다”는 무조건적 주장은 반박된다.**  
  근거: `RU-97`에서 `m32_raw`는 fold×seed 잔차가 `23.9%`로 seed 주효과보다 크고, “fold를 더 모아도 줄지 않는다”고 실측됐다. 다만 이는 GF 탐색 공간의 관측이지 현행 7-branch의 분산분해는 아니다. 따라서 seed averaging이 현행 7-branch의 총 적합 분산을 충분히 줄인다는 일반적 주장은 현재 증거로 성립하지 않는다.

- **“`CA-08`이 모든 집계 함수 변경을 닫았다”는 주장은 반박된다.**  
  근거: `CA-08`의 기각 기전은 선택 편향이고, 결합 함수 일반의 무효는 미확인이다. 경계 밖에는 제한된 결합 연구와 `PSW`·`TGW` 계열이 허용된다. 따라서 사전 등록·선택 이력·성능 조회 후 변형 금지를 지킨 단일 집계 probe는 `CA-08`만으로 일괄 무효라고 단정할 근거가 없다.

- **“Primary 7에서 `+0.0030`을 넘으면 SEAL 10에서 ABMIL을 이긴다”는 주장은 반박된다.**  
  근거: `PROJECT.md` §1은 Primary 7의 개선이 hold-out 개선을 보장하지 않는다고 명시한다. `D-047`은 SEAL 10 결과를 모델·후보·구조 선택에 쓰는 것을 금지한다. 따라서 Primary 7 승격과 최종 목표 달성을 동일시하면 증거 범위를 넘는다.

- **“적합 분산 감소는 성능 개선이다”는 동일시는 반박된다.**  
  근거: `PROJECT.md` §4.1의 estimand는 대응 Δ의 기댓값이다. seed averaging·variance downweighting·rank 변환 등은 Δ의 SE·seed sd·판정 상태를 바꿀 수 있으나, 그것이 `E[Δ]`를 올린다는 것은 별도 증거가 필요하다.

- **“건드리지 않은 축은 그냥 생각이 안 난 것”이라는 설명은 반박된다.**  
  근거: `closed_axes.md`는 `CA-08`, `CA-R1`, `CA-10`, `CA-06`, `CA-07`, `CA-11`, `CA-13`, `CA-05`, `CA-03` 등을 기각 기전·경계·재개 조건과 함께 닫거나 보류해 있다. `current_status.md`는 계측 복구, `LSAK` 사양 부재, 적합 분산 승격 절차 미승인을 기록한다. 따라서 미착수는 단순 미고려가 아니라 전제 결여와 유효한 경계에 의해 막혀 있는 경우가 많다.

## 판별 불가 (증거가 가르지 못함)

- **(1) `3.52/7`이 구조 상한인지, 아직 못 만든 중복인지는 현재 증거로 판별 불가다.**  
  근거: `CA-09`·`CA-10`·`RU-85`는 열려 있는 하위축을 보여주지만, 실제로 기존 7 margin과 직교하는 label-free margin이 존재함을 입증하지는 못한다. 한 후보의 게이트 ① 통과는 제한적 사실만 주고, 여러 후보의 게이트 ① 탈락도 후보 공간 소진 없이는 “막혀 있다”는 반박이 되지 않는다. branch별 기여도는 문서에 없다.

- **(1) fold-wise rank 변환이 `3.52`의 원인을 가르는지 판별 불가다.**  
  근거: rank 변환은 monotonic nonlinear transform이므로 scale 문제가 없어도 effective rank를 바꿀 수 있다. 현재 문서에는 branch별 raw margin 상관계수, rank margin 상관계수, branch별 기여도가 없다. 따라서 rank margin의 effective rank 변화만으로 scale-driven 중복과 정보 중복을 구분할 수 없다.

- **(2) 어떤 집계 변경이 trimmed mean보다 나은지 판별 불가다.**  
  근거: median, mean, rank average, inverse-variance weight, trim 폭 변형, seed ensemble average 등에 대한 Primary 7 paired Δ·seed sd·과제 군집 CI 실측이 문서에 없다. `CA-08`은 선택 편향 절차를 닫았지만, 사전 등록 제한 연구의 성능 결과는 아직 없다.

- **(2) fit-variance weighted aggregation이 `CA-08` 경계 밖인지 문서만으로 판별 불가다.**  
  근거: `CA-08` 경계 밖에는 “예측분산·Context 신호 기하를 사용하는 `PSW`·`TGW`”가 언급되지만, fit-variance 가중이 그 “예측분산” 범주에 해당하는지 명시되지 않았다. `D-022`는 `PSW`·`TGW`를 비저촉으로 판정했지만, fit-variance 가중까지 포함하는지는 확정되지 않았다.

- **(3) 적합 분산 감소가 성능 개선보다 나은 투자인지 판별 불가다.**  
  근거: 현행 7-branch의 seed 주효과와 fold×seed 잔차 분산분해는 문서에 없다. `RU-97`의 `23.9%` fold×seed 잔차와 `0~2.9%` seed 주효과는 GF 탐색 공간의 실측이지 현행 7-branch로 일반화할 근거가 없다. seed 변동의 원천(무작위 표본, PCA 초기화, bag/draw, 수치적 불안정 등)도 문서에 없다. 적합 분산을 포함한 승격 판정 절차도 사용자 미승인이다.

- **(3) seed averaging이 `+0.0030` 근처 후보를 판별 가능하게 하는지 판별 불가다.**  
  근거: 350×R fold fit이 4 GPU-hour 안에 가능한지 fold time 미실측으로 알 수 없다. 기준선과 후보를 같은 R-seed 프로토콜로 재생성해야 하는데, 그 비용도 미실측이다. seed-averaged margin의 paired Δ가 `PROJECT.md` §4.1의 단일 적합 estimand와 동일한지도 문서에 없다.

- **(4) LID branch가 `3.52` 해석을 가르는지 판별 불가다.**  
  근거: `RU-85` H에서 LID 추정기는 부분추출 크기에 민감했고, `|r|(m=4096, m=1024)=0.314`로 강건성 임계 0.8을 밑돌았다. full-token 또는 deterministic subsampling이 그 민감성을 해소한다는 증거는 문서에 없다. 슬라이드당 token 수, k-NN 계산 비용, 350-fold LID margin 계산 시간도 미실측이다. `SCREEN_ONLY` 게이트 ① 실행 경로는 `D-050`으로 현재 무효다.

- **(4) LSAK·학습 파라미터 후보가 4h 안에 판별 가능한지 판별 불가다.**  
  근거: `current_status.md`는 `LSAK` 사양이 문서에 없다고 한다. 학습 파라미터 후보는 `D-048`으로 열렸지만, 적합 분산을 포함한 승격 판정 절차가 사용자 미승인이다. fold 적합 시간도 미실측이다.

- **(5) 4 GPU-hour 안에 350-fold 또는 350×R seed 재적합이 가능한지 판별 불가다.**  
  근거: fold 적합 1회 소요 시간은 문서에 없다. 7 과제 × 1 fold timing이 350-fold 전체를 대표하는지도 fold별 slide 수·patch 수·context slide 수 분포가 미실측이라 알 수 없다. context 슬라이드 수는 90~261장으로 fold 간에 다르다.

- **provenance writer가 현재 실제로 유효한지 판별 불가다.**  
  근거: `D-050`은 `unverified-hold`를 유지하고, `current_status.md`는 현재 목표에 provenance writer 복구를 둔다. 그러나 writer가 산출물에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시·seed·run ID를 실제로 쓰는지 문서에 확인된 동작이 없다.

- **저장 margin의 오프라인 재집계가 `verified`로 가능한지 판별 불가하다.**  
  근거: 저장된 예측·마진 파일에 provenance가 없어 특정 구성에 귀속시킬 수 없다. 파일 경로·포맷·fold 대조 가능 여부도 문서에 없다. 따라서 오프라인 재집계는 provenance 복구와 fold manifest 대조가 선행되지 않으면 `verified` 판단이 불가하다.

## 실행 무효 (실행·계측 결함으로 판단 불가)

- **과거 저장 산출물 기반 branch 조합 비교는 실행 무효다.**  
  이유: `D-050`에 따라 여섯 러너가 같은 7-branch 설정으로 수렴했고, 저장 예측 파일에 branch list·가중치·코드 해시가 없어 구성 귀속이 불가하다.

- **공식 기준선 `0.6227`과 과거 `+0.56%p` 승격 근거를 `verified`로 인용하는 것은 실행 무효다.**  
  이유: `PROJECT.md` §3.2는 해당 값을 `unverified-hold`로 둔다. 계측 복구 전에는 게이트·승격 비교를 `verified`로 수행할 수 없다.

- **provenance writer와 `SCREEN_ONLY` 경로 복구 전의 게이트 ①·승격 비교는 실행 무효다.**  
  이유: `current_status.md`와 `closed_axes.md` §3-I은 계측 복구 전 라벨 무관 게이트 ①을 `verified`로 돌릴 수 없다고 한다.

- **적합 분산을 포함한 승격 판정 절차가 사용자 승인되기 전의 seed-averaged·fit-variance 승격 판단은 실행 무효다.**  
  이유: `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 미승인이라고 한다. 승인 전에는 분산 감소 결과를 승격 근거로 연결할 수 없다.

- **1-fold timing이 350-fold 또는 350×R seed fit을 4 GPU-hour 초과로 판정하면 해당 전량 비교는 실행 무효다.**  
  이유: 예산은 GPU 1장 합계 4 GPU-hour이고, 단일 M 2시간 또는 합계 4 GPU-hour 초과 시 중단 조건이다. fold time 미실측 상태에서 초과가 확인되면 해당 실험은 이 예산에서 판단 불가다.

- **SEAL 10 결과를 후보·집계·강도·구조 선택에 쓰는 것은 실행 무효다.**  
  이유: `D-047`은 SEAL 10 결과를 모델 개선 방향에 쓰는 것을 금지한다.

- **best-seed 선택, 성능 조회 후 trim 폭·K·가중·집계 방식 선택은 실행 무효다.**  
  이유: `CA-08`과 §224 Oracle 금지는 선택에 쓴 평가자료의 최댓값을 확증 성과로 쓰는 절차를 문제 삼는다.

- **`SCREEN_ONLY` 경로 복구 전의 LID gate ①은 실행 무효다.**  
  이유: `closed_axes.md` §3-I은 게이트 ① 실행 경로가 현재 무효라고 한다. 복구 전 결과는 `verified`로 쓸 수 없다.

- **provenance·fold manifest·branch list·가중치·코드 해시 대조 없이 저장 margin을 재집계해 승격 근거로 쓰는 것은 실행 무효다.**  
  이유: 구성 귀속이 불가능하면 재집계 값도 특정 후보·기준선에 귀속시킬 수 없다.

- **단일 적합 Δ가 `+0.0030` 근처일 때 적합 분산 보고 없이 승격 판단을 내리는 것은 실행 무효다.**  
  이유: `D-048` 이후 적합 분산 보고가 필수이고, `RU-97`은 단일 적합 차이가 seed 뽑기일 수 있음을 실측했다.

## 좌석 공백 (기권·실패로 빠진 관점)

- 기권·실패로 빠진 좌석 없음.

## 다음 행동 (권한 등급과 함께)

- **T0 (기록·판정)**  
  이번 종합의 결론을 기록한다: 현재 `verified` paired Δ로 승격할 후보는 없으며, 모든 Primary 7 판정은 `hold-out 미검증`이고, SEAL 10으로 모델·후보·집계·강도를 선택하지 않는다.

- **T1 (계측 복구·smoke 실행)**  
  현재 목표 범위에서 provenance writer와 러너 YAML 경로를 복구·검증한다. 산출물에 branch list, 가중치, 코드 해시, 데이터 버전, fold manifest 해시, seed, run ID를 기록하도록 하고, 합성 데이터에서 `sh`/`sj` 가중치 `1.0→0.0` 변경 시 margin 변화와 provenance 일치를 확인한다. 필드 누락, margin·provenance 불일치, 단위 테스트 실패 시 중단한다.

- **T1 (1-fold timing·provenance smoke)**  
  GPU 4에서 Primary 7 7 과제 × 1 fold를 full pipeline으로 실행해 wall time, GPU memory, slide 수, patch 수, context slide 수, branch list, 코드 해시, fold manifest 해시, seed, run ID를 기록한다. 이 smoke는 350-fold 또는 350×R seed 실행을 `verified`로 확정하는 것이 아니라, 4 GPU-hour 예산 게이트와 대표성 판별 신호로만 쓴다. fold manifest에서 fold별 slide·patch·context 분포를 확인하지 못하거나, 1-fold가 예산 한계에 가깝거나, fold 간 편차를 확인할 수 없으면 350-fold 가능 여부는 `판별 불가`로 둔다.

- **T1 (조건부)**  
  provenance·timing smoke가 통과하고 남은 예산이 있으면, 350-fold paired Δ를 승격 근거로 쓰지 않는 범위에서 사전 등록된 label-free 진단만 수행한다. 예: 재생성된 margin에서 raw margin과 rank margin의 상관·effective rank·seed sd를 성능 조회 전에 기록한다. 이 진단은 `CA-08` 경계 안으로 판정되면 즉시 중단하고, 성능 조회 후 변형·최댓값 선택을 하지 않는다.

- **T2 (보류·금지)**  
  350-fold paired Δ 승격, seed-averaged estimand 승격, fit-variance 가중 승격, LID 채택, LSAK 실험, 학습 파라미터 후보 승격, SEAL 10 개봉·선택은 다음 전제 충족 전 보류한다: (1) provenance writer와 `SCREEN_ONLY` 경로 복구, (2) 1-fold 또는 다중 fold timing으로 4 GPU-hour 내 실행 가능 확인, (3) 적합 분산 포함 승격 절차 사용자 승인, (4) 사전 등록과 선택 이력 기록.

## 하지 말 것 (이유와 함께)

- **SEAL 10 결과를 모델·후보·집계·강도 선택에 사용하지 않는다.**  
  이유: `D-047`은 SEAL 10 결과를 개선 방향에 쓰는 것을 금지한다.

- **`0.6227`과 과거 `+0.56%p`를 `verified` 승격 근거로 인용하지 않는다.**  
  이유: `D-050`은 해당 값을 `unverified-hold`로 둔다.

- **과거 branch 조합 비교를 근거로 사용하지 않는다.**  
  이유: 여섯 러너가 같은 7-branch 설정으로 수렴했고, 저장 산출물에 provenance가 없다.

- **단일 적합 Δ를 `+0.0030` 승격 판단에 단독으로 사용하지 않는다.**  
  이유: 동일 구성 seed 재적합의 표준편차 `0.0021~0.0083`은 승격 임계와 같은 크기대이고, `RU-97`은 단일 적합 차이가 seed 뽑기일 수 있음을 실측했다.

- **best-seed 선택을 하지 않는다.**  
  이유: 선택 편향과 Oracle 금지에 저촉될 수 있다.

- **성능 조회 후 trim 폭·K·가중·집계 방식을 바꾸지 않는다.**  
  이유: `CA-08`의 선택 편향 기전과 §224 Oracle 금지에 저촉된다.

- **seed averaging이 fold×seed 잔차까지 줄인다고 주장하지 않는다.**  
  이유: 현행 7-branch의 seed 주효과·fold×seed 잔차 분산분해가 문서에 없고, `RU-97`의 GF 공간 실측만으로는 일반화할 수 없다.

- **rank 변환이 branch 중복을 해결한다고 단정하지 않는다.**  
  이유: rank 변환은 monotonic nonlinear transform이므로 effective rank를 기계적으로 바꿀 수 있고, branch별 기여도와 raw/rank 상관이 미실측이다.

- **LID gate ① 통과만으로 “`3.52`는 구조 상한이 아니다”를 일반화하지 않는다.**  
  이유: 한 후보의 통과는 8-branch에서 `eff.rank ≥ 4.02` 가능이라는 제한적 사실만 준다. LID 추정기 안정성과 비용도 미확인이다.

- **1-fold timing만으로 350-fold 또는 350×R seed 실행을 `verified`로 확정하지 않는다.**  
  이유: fold별 slide·patch·context 분포와 1-fold 대표성이 미실측이다.

- **4 GPU-hour 초과가 예상되는 350-fold·seed ensemble·LID·rank aggregation을 강행하지 않는다.**  
  이유: 예산 초과 시 실행 무효이며, 중단 조건에 해당한다.

- **적합 분산 감소 결과를 승격 근거로 쓰지 않는다.**  
  이유: 적합 분산을 포함한 승격 판정 절차가 사용자 미승인이다.

- **fold 수를 늘리거나 fold를 재분할하지 않는다.**  
  이유: `PROJECT.md` §3.1은 PathoBench 공식 fold 분할을 고정하고 fold 수 증대·재분할을 금지한다.

- **성능 관측을 기전 입증으로 바꾸어 쓰지 않는다.**  
  이유: 연구 규범상 불명확한 기전은 `기전 미확인`으로만 기록한다.
