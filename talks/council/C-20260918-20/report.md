# ICF Council — C-20260918-20

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
- **호출**: 14/18 · **소요**: 1018.3초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 편성 근거 (결과 전 기록)

탐색 질문이므로 제안(P) 둘을 온도 0.7·0.8로 두어 서로 다른 방향을 뽑는다. 이론(T)은 질문 1의 유효 랭크 해석을, 실측(M)은 질문 3·5의 분산과 비용을 맡는다. 반증(R)은 '새 축이 있다'는 낙관과 '다 막혔다'는 비관을 둘 다 공격한다. 감사(A)는 제안이 과거의 무효한 비교 결과에 기대고 있지 않은지 본다. 앞선 회차 보고서는 입력에 넣지 않았다(회차 간 앵커링 금지).

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: fold-global rank-normalized trimmed mean — 단일 사전고정 집계 후보
- **바꾸는 것**: 공식 7-branch의 slide-level trimmed mean(최저·최고 1개 절사 후 나머지 평균)을, 각 branch margin을 해당 fold의 slide 전체에서 rank 또는 empirical CDF로 변환한 뒤 slide마다 최저·최고 1개만 절사하고 나머지 5개의 평균을 최종 score로 쓰는 단일 후보로 교체한다. rank 변환은 라벨 없이 fold 내 slide 순서만 사용한다. CA-08의 경계 밖인 “제한된 결합 연구”로 수행하며, 후보는 1개만 검증하고 Primary 7에서 여러 집계 함수를 돌려 최댓값을 고르지 않는다.
- **바꾸는 전제**: branch margin의 절대 스케일·분산이 서로 commensurable이고, 기존 trimmed mean의 절사 한 개가 branch outlier를 충분히 처리한다는 전제를 뒤집는다. AUROC는 최종 score의 순서만 사용하므로, branch별 절대 크기보다 fold-global rank가 더 안전한 공통 단위라는 전제로 바꾼다.
- **기대하는 차이**: Primary 7 paired Δ_macro에 대한 정량 예측은 불가. 보고할 판별 지표는 (a) baseline 대비 paired Δ_macro가 +0.0030 이상인지, (b) 7개 과제 전량의 부호, (c) 기존에 악화된 과제 전량의 부호 변화다. 판별 비용은 provenance-locked 7-branch margin 1회만 존재하면 재집계는 GPU 0이다. 그러나 7-branch fold 적합 시간은 문서에 없으므로 4 GPU-hour 내 7-branch margin 1회 생성이 보장되지 않는다. 5-branch paired 2 arm은 3.89 GPU-hour로 측정됐으나, 이는 공식 7-branch 승격 하위계는 아니다.
- **기전**: rank 변환이 branch 간 스케일·꼬리 차이를 제거해, 한 branch의 outlier margin이 최종 slide 순서를 왜곡하는 경로를 줄인다고 가정한다. rank 변환은 각 branch margin에 대한 monotone 변환이므로 branch 내부의 slide 순서 정보는 보존되지만, 7개 branch를 결합하는 단계의 scale mixing은 바뀐다. 다만 rank-normalized trimmed mean이 실제 AUROC를 개선하는지는 문서에 실측이 없으므로 기전 미확인이다.
- **실패 조건**: branch margin의 절대 크기가 ridge 해의 신뢰도·크기를 담고 있고 rank 변환이 그 정보를 버리면 Δ가 0 또는 음수일 수 있다. fold-global rank는 slide 수에 의존하는 이산/준연속 변환이므로, fold 내 slide 수가 부족하거나 ties가 많으면 score 분해능이 떨어져 판별 불가로 갈 수 있다. 기존 trimmed mean이 이미 최저·최고 1개를 절사해 추가 효과가 없으면 Δ≈0이다. 7개 branch margin의 방향이 모두 현재 trimmed mean과 동일하지 않고, 라벨 없이 방향 정합을 확정할 수 없으면 실행 무효다. Primary 7에서 이 후보가 실패한 뒤 다른 rank/집계 변형을 골라 최댓값을 확증으로 쓰면 CA-08 경계 안으로 회귀한다.

## 제안 2: deterministic branch construction — 적합 분산 원천 제거
- **바꾸는 것**: branch 생성 단계의 stochastic step, 즉 token subsampling, PCA/SVD initialization, random permutation, GPU reduction order 등을 deterministic canonical ordering으로 교체한다. full-token 계산이 불가능하면 고정 stratified subsampling manifest를 코드에 고정하고 provenance에 해시를 기록한다. seed 값만 바꾸는 재적합이 아니라, stochastic step 자체를 제거하거나 고정하는 변경이다.
- **바꾸는 전제**: seed std 0.0021~0.0083은 fold data에서 오는 불가피한 적합 변동이라는 전제를 뒤집는다. 이 변동의 일부는 임의 subsampling, PCA rotation/sign, reduction order에서 온다는 전제로 바꾼다.
- **기대하는 차이**: 성능 개선이 아니라 분산 저감이다. 정량 예측은 불가. 목표는 seed std를 +0.0030 승격 임계보다 아래로 줄이거나, 적어도 paired Δ의 fit-variance 구성요소를 제거하는 것이다. 성능 개선과의 차이는 명확하다: macro mean을 올리는 것이 아니라, 같은 구성의 재적합에서 score가 흔들리는 폭을 줄여 Δ 판정의 신뢰도를 높이는 투자다. mean이 그대로여도 +0.0030 미만 후보를 “판별 불가”에서 “지지/반박”으로 분리할 수 있게 하면 가치가 있다. 5-branch paired 2 arm은 3.89 GPU-hour로 측정됐고, 7-branch 비용은 문서에 없으므로 4 GPU-hour 내 7-branch paired 검증은 보장되지 않는다.
- **기전**: 임의 subsampling, PCA sign/rotation, GPU reduction이 branch margin에 추가 노이즈를 만들면, deterministic full-token 또는 고정 manifest가 그 노이즈를 제거한다. 그러나 seed std 0.0021~0.0083의 실제 원인은 문서에 없으므로 기전 미확인이다. 코드 감사에서 stochastic step이 없으면 이 제안은 실행 무효다.
- **실패 조건**: seed std가 fold 내 sample composition, ridge solver, 또는 subsampling의 regularizer 효과에서 오면 deterministic 교체로 줄지 않는다. full-token 또는 고정 manifest가 branch 분포를 바꿔 mean을 악화하면, 분산 저감과 성능 개선이 상충한다. 이 제안이 성공해도 “성능이 좋아졌다”는 증거가 아니라 “판정 정밀도가 좋아졌다”는 증거만 된다. 따라서 +0.0030 미만 후보를 살리는 수단이지, 성능 자체의 증명은 아니다.

## 제안 3: residualized mode-count branch — 유효 랭크 3.52의 두 해석을 가르는 probe
- **바꾸는 것**: 공식 7-branch에 후보 브랜치 1개를 추가한다. 후보는 `SJ`가 whitened 32D 구름을 1D radius로 붕괴시키며 버리는 성분에서 읽는 multimodality scalar, 즉 mode count다. 추정기는 성능 조회 전에 고정하고 bandwidth/subsample 탐색을 하지 않는다. 이 scalar를 fold별로 기존 7 branch margin에 closed-form ridge로 회귀시켜 잔차를 최종 margin으로 한다. 게이트 ①의 max |r| ≤ 0.6, 8-branch 기준 eff.rank ≥ 4.02, 그리고 게이트 ②a 또는 ②b를 모두 통과해야만 ensemble에 넣는다.
- **바꾸는 전제**: 유효 랭크 3.52/7은 “branch를 더 넣는 방향이 막혔다”는 해석을 뒤집고, “서로 다른 것을 재는 branch를 아직 만들지 못했다”는 해석을 시험한다. 기존 margin과 직교하게 만든 새 형상 통계가 게이트 ①을 통과하면, rank가 막힌 것이 아니라 재는 대상이 부족했다는 쪽에 관측 증거가 된다.
- **기대하는 차이**: 정량 성능 예측은 불가. 판별 지표는 게이트 ①의 max |r|, eff.rank/branch, 게이트 ②a/b, 그리고 통과 시에만 paired Δ_macro다. 해석은 다음과 같이 고정한다: 게이트 ① 통과 + 게이트 ② 실패면 “rank는 올라가지만 정보 없음”, 게이트 ① 통과 + 게이트 ② 통과면 “서로 다른 측정이 가능”, 게이트 ① 실패면 “직교화 실패 또는 기존 공간과 중복”이다. mode count 계산 비용은 문서에 없으므로 4 GPU-hour 내 screen/fit이 보장되지 않는다.
- **기전**: mode count는 slide 내 token cloud의 모드 수/이질성을 읽고, 현재 위치·형상 계열이 읽지 않는 기하를 담을 수 있다는 가정이다. residualization은 기존 7 branch와 공유되는 변동을 제거해 독립 증거만 남긴다. 그러나 mode count가 Primary 7 label과 관련이 있는지는 문서에 없으므로 기전 미확인이다.
- **실패 조건**: mode count estimator가 LID처럼 부분추출/시드에 민감하거나, whitened cloud가 AKS처럼 등방해 변별 자유도가 없으면 게이트 ②에서 죽는다. residualization이 noise만 남기면 eff.rank는 오를 수 있으나 정보량은 없다. CA-10 경계에서 이는 `SH` moment 변형이 아니라 `SJ`가 버린 성분의 모드 수를 읽는 비저촉 영역을 사용하는 것이며, 성능 조회 전에 게이트 ①을 먼저 실행해야 한다. 게이트 ①에서 탈락하면 성능을 조회하지 않고 종료한다.

질문: 7-branch Primary 7 50-fold 1회 실행의 fold 적합 1회 시간 또는 총 GPU-hour가 측정된 값이 있는가?
질문: seed std 0.0021~0.0083을 만드는 stochastic step이 코드 어디에서 발생하는지 기록된 진단이 있는가?
질문: 현재 Primary 7 저장 산출물에서 branch별 margin을 offline 재집계할 수 있는 provenance가 있는가?
질문: 7개 branch margin 모두 현재 trimmed mean에서 같은 방향, 즉 높은 margin이 positive 쪽으로 사용되는가?
질문: mode count 후보의 SCREEN_ONLY fold 비용이 측정된 적이 있는가?

### P2 (제안)

## 제안 1: 사전 선언된 prediction-variance weighted trimmed mean (PSW)
- **바꾸는 것**: 공식 7-branch의 fold별 branch margin을 동등한 trimmed mean으로 집계하는 규칙을, closed-form ridge fit에서 산출되는 query slide별 예측분산의 역수 가중 trimmed mean으로 바꾼다. 가중치 상수·절사 개수·사용 분산의 정의는 Primary 7 결과 조회 전에 한 번 고정하고, 여러 가중 변형을 탐색한 뒤 최댓값을 고르는 절차를 쓰지 않는다. 이는 `CA-08`의 기존 margin 결합 탐색이 아니라 `D-022`가 비저촉으로 둔 새 정보, 즉 예측분산을 쓰는 PSW 계열이다.
- **바꾸는 전제**: 7개 branch margin이 fold마다 동등한 신뢰도를 가진다는 전제를, ridge fit이 이미 산출하는 예측분산이 margin 신뢰도의 label-free 대리변수라는 전제로 뒤집는다.
- **기대하는 차이**: trimmed mean 대비 Primary 7 대응 Δ_macro가 양의 방향이거나, 적어도 fold/seed 수준 Δ 변동이 줄기를 기대한다. 그러나 문서에 PSW 효과 수치가 없어 정량 예측 불가. 한 arm에서 margin과 예측분산을 저장할 수 있으면 같은 fold에서 baseline과 후보를 오프라인 재집계해 paired Δ를 만들 수 있다.
- **기전**: 독립 오류 가정에서 역분산 가중은 aggregate 오류의 분산을 줄인다. 다만 branch margin 오류가 독립인지, ridge 예측분산이 실제 AUROC 오류를 반영하는지는 미확인이다. 따라서 기전 미확인.
- **실패 조건**: 예측분산이 task difficulty·context size·염색 난이도의 대리로만 작동해 노이즈가 아닌 신호를 절단하면 Δ가 0 또는 음수. 여러 가중 변형을 Primary 7에서 골라야 하면 `CA-08` 선택 편향으로 실행 무효. 한 arm 저장 재사용 경로가 없으면 2 arm이 필요하고, `current_status`의 5-branch fold 20.0s 기록 기준 3.89 GPU-hour로 4h에 여유가 거의 없다. 7-branch fold 시간은 문서에 없어 확정 불가.

## 제안 2: 공통 무작위화(CRN) paired 비교
- **바꾸는 것**: 후보와 baseline을 비교할 때 fold 짝짓기뿐 아니라 seed·subsample·context selection 등 무작위 요인을 동일하게 고정하고, 4h 안에서는 R>1 seed averaging을 하지 않는다. 대신 paired Δ와, 가능하면 동일 구성의 기존 seed std `0.0021~0.0083`을 함께 보고한다.
- **바꾸는 전제**: "seed std가 +0.0030 임계와 같은 크기이므로 단일 적합 비교는 대부분 뽑기"라는 전제를, "뽑기의 상당 부분이 baseline과 candidate에 공통인 무작위 요인이며, 같은 무작위 흐름을 쓰면 Δ에서 상쇄된다"는 전제로 뒤집는다.
- **기대하는 차이**: AUROC 기대값 자체는 바뀌지 않는다. paired Δ의 표준편차/분산이 줄어 +0.0030 판별력이 올라갈 수 있으나, seed 변동의 출처가 문서에 없어 정량 예측 불가. `current_status`의 5-branch fold 20.0s 기록 기준 2 arm Primary 7은 3.89 GPU-hour라 4h 안에 실행 가능하고, R=4 2×2는 31.1 GPU-hour라 4h에서는 실행 무효.
- **기전**: common random numbers는 baseline/candidate fold AUROC 사이에 공통 변동 상쇄를 만들어 차이의 분산을 줄인다. 그러나 seed가 어떤 무작위 연산에 영향을 주는지 문서에 없어 기전 미확인.
- **실패 조건**: seed 변동이 공통 노이즈가 아니라 후보 변경과 상호작용하는 무작위 경로 차이이면 CRN 효과가 0. seed가 AUROC 불변인 PCA 부호 같은 요인만 바꾸면 변동 감소 없음. 승격 판정에 독립 적합 분산이 필수로 요구되면 단일 CRN 비교만으로는 실행 무효. 7-branch fold 시간이 5-branch 20.0s보다 길면 2 arm도 4h를 초과해 판별 불가.

## 제안 3: robust LID branch probe (유효 랭크 해석 판별)
- **바꾸는 것**: 기존 7-branch에 patch cloud의 국소 내재차원(LID) branch를 `SCREEN_ONLY`로 추가하고, 성능 조회 전에 게이트 ①만 수행한다. `CA-10` 경계 밖인 discarded component reading에 해당한다. RU-85의 부분추출 민감성을 피하기 위해 사전 선언된 안정화 두 가지만 쓴다: 슬라이드별 동일 토큰 수 정규화, `m=1024`와 `m=4096` 두 크기에서 ID_TwoNN 마진을 계산한 뒤 두 마진의 `|r| ≥ 0.8`이면 평균으로 최종 margin, 아니면 성능 조회 없이 기각.
- **바꾸는 전제**: 유효 랭크 `3.52 / 7`을 "branch 추가 방향이 막혔다"고 읽는 전제를, "기존 7 branch가 같은 인자를 중복해서 읽고, BD/SH/SJ가 읽지 않는 분포 기하에 독립 정보가 있을 수 있다"는 전제로 뒤집는다.
- **기대하는 차이**: 8-branch에서 `max |r| ≤ 0.6`이고 eff.rank `≥ 4.02`면, 유효 랭크 3.52는 정보 상한이 아니라 기존 family 중복의 결과라는 해석을 지지한다. 반대로 상관 초과 또는 효율 하락이면 branch 추가 방향이 현재 family 안에서는 막혀 있다. 성능 Δ_macro는 정량 예측 불가.
- **기전**: LID는 slide 내 patch cloud의 국소 선형 차원을 읽어서 covariance 비대각, bag mean, spectral entropy, quantile, salience, SH/SJ radius와 다른 fold 차이를 만들 수 있다. 그러나 LID가 질병/형태 신호를 담는지, 동일 토큰 정규화·이중 크기 평균이 표본 민감성을 해소하는지는 미측정이다. 기전 미확인.
- **실패 조건**: LID가 토큰 수·염색·배경 노이즈에 지배되어 게이트 ②에서 전 과제 또는 40/50 fold 기준을 못 통과하면 반박. `m=1024/4096 |r| < 0.8`이면 사전 기각. k-NN 추가 비용이 fold 20.0s를 크게 넘으면 4h 안에 Primary 7 `SCREEN_ONLY` 완료가 불가해 판별 불가. 이 probe는 게이트 ① 해석만 판별하고, 통과해도 승격은 별도 후보 arm이 필요하므로 같은 4h 안에서 성능 개선 확인은 불가.

질문: 7-branch 공식 구성의 fold 적합 1회 소요 시간, 한 arm의 fold별 branch margin·ridge 예측분산·LID margin을 저장해 같은 fold에서 오프라인 재집계/재검증할 수 있는 저장 경로, 그리고 seed만 바꿔 재적합 시 무작위 요인(서브샘플링, PCA, context selection 등)이 무엇인지 알고 싶다.

### T1 (이론)

- **가정**: 현재 접근은 다음을 암묵적으로 참으로 둔다. (1) 7개 branch margin은 서로 다른 증거를 재는 채널이며, 유효 랭크 `3.52/7`은 “아직 다른 것을 재는 branch를 만들지 못했다”는 상태지, patch feature에서 더 읽을 정보가 없다는 증거로 쓰지 않는다. (2) 단일 적합의 Primary 7 대응 Δ는 구성의 고정 속성이고, seed 재적합 변동은 fold 짝짓기 오차보다 작거나 `+0.0030` 승격 임계보다 충분히 작다. (3) trimmed mean voting은 branch 중복을 무해하게 평균화하는 집계이지, 그 자체가 성능·판정 불확실성의 1차 변수가 아니다. 학습 파라미터 없는 집계 변경은 closed-form prediction variance 가중(PSW), Context 신호 기하 가중(TGW), 사전 선언된 rank/median/trim 규칙, 라벨 무관 게이트로 고정된 중복 제거 정도만 허용된다고 둔다. (4) 과거 arm 비교는 서로 다른 구성을 실제로 비교한 것이고, Primary 7 개선은 SEAL 10 개선의 충분한 프록시라고 둔다. (5) 적합 분산/seed 복제 설계와 PSW/TGW 같은 집계 축은 아직 1차 병목이 아니라 부수 항목으로 둔다.

- **이 가정이 깨지는 조건**: (1) 유효 랭크 `3.52`의 두 해석을 가르는 관측은 성능 조회 이전의 라벨 무관 게이트 ①이다. 서로 다른 수학적 계열의 후보 margin이 `max |r| ≤ 0.6`이고 8-branch `eff.rank ≥ 4.02`를 만족하면 “아직 못 만들었다”는 쪽이 지지된다. 반대로 여러 독립 계열 후보가 `max |r| > 0.6`이거나 `eff.rank`가 `3.52`에서 비하락이면 “branch 추가 방향이 막혔다”로 전환된다. (2) 동일 구성 seed 재적합 std가 `+0.0030`과 같은 크기면 단일 적합 Δ는 기호가 바뀔 수 있고, 승격 판단은 적합 분산이 포함된 설계 없이는 무효/판별 불가가 된다. (3) 집계 규칙을 바꾸지 않고도 seed만 바꿔 Δ가 임계 위아래로 흔들리면 집계는 무해한 평균이 아니라 판정을 좌우하는 변수다. (4) 러너가 다른 arm을 같은 설정으로 수렴시켰다면 arm 간 Δ는 구성 효과가 아니라 적합 변동이다. (5) Primary 7 Δ가 SEAL 10 Δ를 대표하지 않으면 최종 목표 프록시가 깨진다. (6) 적합 분산이 `+0.0030`과 같은 크기면 미건드리기 축이었던 복제 설계가 1차 병목이 된다. `D-048` 이전에는 `seed std 0.00000`이 성립해 건드리지 않은 이유가 타당했으나, `D-048` 이후에는 미승인 절차와 4h 예산 부족이 이유다. (7) PSW/TGW가 게이트 ①/승격에서 seed std보다 큰 Δ를 내지 못하면 집계 축도 정보량 병목에 막힌다.

- **깨질 때 나타나는 관측 가능한 증상**: 동일 구성 seed 실행이 `+0.0030`을 넘나듦; `RU-97 §8`의 macro 폭 `0.0200`과 같은 폭이 fold Δ에도 전이; 2-arm paired 비교에서 Δ가 seed 평균에 따라 부호/통과 여부 변동; branch 추가 시 `eff.rank`가 `3.52`에서 거의 안 오름; 게이트 ①에서 `max |r| > 0.6` 반복; branch 제거/집계 변경 효과가 seed std 범위 안에 들어감; provenance가 없어 산출물을 arm에 귀속 못 함. branch별 기여도는 문서에 없으므로 “branch 제거 효과”는 관측 가능한 증상의 후보이지 현재 측정값이 아니다. 분산 감소 수단의 증상은 성능 점수 상승이 아니라 seed 간 Δ의 폭과 부호 일치율 감소다. 성능 개선의 증상은 seed 평균 Δ가 `+0.0030`을 넘는 것이지, 단일 적합의 임계 초과가 아니다. 4h 예산에서는 5-branch fold 적합 `20.0`초 기준으로 2-arm paired single-fit만 계산 가능하고, 문서의 `R=4 2×2 31.1 GPU-hour`와 2-arm `3.89 GPU-hour`에 비추어 추가 seed 복제는 4h를 초과하므로 성능 개선과 분산 감소의 동시 판별은 불가능하다.

- **문서의 어떤 관측이 이 가정과 충돌하는가**: `PROJECT.md §4`의 승격 임계 `+0.0030`, `§3.2`의 유효 랭크 `3.52/7`, `§1 D-048`의 적합 분산 보고 필수, `CA-14` 폐기로 `seed std 0.00000` 성립 불가, `RU-97 §8`의 동일 구성 seed만 바꿔 macro 폭 `0.0200`, `§1`의 “Primary 7의 개선이 hold-out 개선을 보장하지 않는다”; `current_status.md D-053`의 여섯 러너가 전부 같은 7-branch 설정으로 수렴, 과거 arm 간 차이 보고는 전부 적합 변동; `current_status.md`의 fold 적합 1회 `20.0`초(과제 3개 평균, 5-branch)와 2-arm paired 700 fold `3.89 GPU-hour`, `R=4 2×2 31.1 GPU-hour`로 4h 내 적합 분산 확증 불가; `closed_axes.md CA-09`의 `BM·QA·DS` 유효 랭크 `1.29/3`, `CA-10`의 형상 계열 병목이 정보량, `CA-08`의 집계 탐색 선택 편향 폐기. 질문 본문은 fold 적합 시간이 측정되지 않았다고 하나 `current_status.md`에는 5-branch `20.0`초가 기록되어 있어 이 충돌도 확인이 필요하다. seed 변동의 원인은 문서에 없으므로 기전 미확인이다. 7-branch 공식 구성의 fold 적합 시간, `SCREEN_ONLY` 게이트 ① 1회 비용, branch별 기여도는 문서에 없다.

질문: fold 적합 `20.0`초가 5-branch 3과제 평균이라는 `current_status.md` 기록과 질문 본문 “측정된 적이 없다” 진술 중 정본이 무엇인지, 그리고 7-branch 공식 구성의 fold 적합 1회 시간과 `SCREEN_ONLY` 게이트 ① 1회 비용이 문서에 있는지.

### M1 (실측)

## 실측 1: 유효 랭크 3.52의 두 해석 판별
- **건드려야 하는 코드·설정**: `scripts/analysis/branch_screen.py`를 `SCREEN_ONLY=1`으로 실행; 공식 7-branch YAML 경로는 문서에 없음(5-branch 직접 실행용 `configs/baseline/v121_active.yaml`만 인용); 후보 branch margin 산출 코드 경로는 문서에 없음.
- **비용**: 후보 margin이 저장된 patch feature에서 CPU로만 산출되면 GPU 비용은 미실측(가정상 GPU 불필요), CPU 시간·메모리 미실측; fold 적합이 필요하면 5-branch 실측 20.0초/fold로 350-fold 1.94 GPU-hour, 7-branch fold 시간·메모리·연산은 미실측.
- **막는 것**: `D-050`으로 provenance 기록이 없어 margin을 특정 구성에 귀속시킬 수 없고 오염 검사 재설계 미완료라 verified 게이트는 막혀 있다. 후보가 기존 7 margin과 max |r| > 0.6이거나 eff.rank < 4.02이면 게이트 ①에서 막힌다.
- **중단 기준**: 후보가 max |r| ≤ 0.6이고 eff.rank ≥ 4.02이면 “branch 추가가 막힌 것이 아니라 서로 다른 것을 재는 branch를 아직 못 만들었다”로 판정; 여러 독립 후보가 모두 max |r| > 0.6 또는 eff.rank < 4.02이면 “추가 방향이 막혔다”로 판정. provenance 기록 불가, 또는 fold 적합이 4 GPU-hour 초과 시 중단.

## 실측 2: 학습 파라미터 없는 집계(trimmed mean voting) 변형
- **건드려야 하는 코드·설정**: `evaluate_pure.py`(인용)와 집계 함수 구현 경로(문서에 구체 경로 없음); 공식 7-branch YAML 경로 문서에 없음; 후보 집계: 고정 trim 개수, median, mean, rank/Borda, min/max, PSW/TGW(예측분산·Context 신호 기하).
- **비용**: per-branch per-fold margin이 provenance와 함께 저장되어 있으면 재집계의 GPU·CPU·메모리 비용은 미실측(저장 margin 재집계 가정 시 GPU 불필요); margin이 없으면 arm 1회 Primary 7에 5-branch 기준 350-fold 1.94 GPU-hour, 7-branch fold 시간·메모리·연산 미실측.
- **막는 것**: `CA-08`으로 기존 margin 위에서 Primary 7에서 최댓값을 골라 같은 자료로 확증하는 절차는 선택 편향으로 저촉; 저장 예측 파일에 provenance가 없어 과거 margin을 구성에 귀속시킬 수 없음; `D-050` 오염 검사 재설계 미완료.
- **중단 기준**: 집계 변형을 Primary 7 결과로 선별한 뒤 같은 자료에서 승격 판정하면 중단; provenance 없이 margin을 쓰면 중단; 재집계 외 fold 적합이 필요해 4 GPU-hour를 초과하면 중단.

## 실측 3: 적합 분산 감소가 성능 개선보다 나은 투자인지
- **건드려야 하는 코드·설정**: seed/적합 반복을 제어하는 실행 스크립트(`scripts/eval_v121.sh`, `eval_seal_tasks.sh`, `evaluate_pure.py` 인용; seed 인자 구체 경로 문서에 없음); PCA·subsample 결정론 사양은 문서에 없음.
- **비용**: 시드 고정만으로는 추가 GPU·CPU·메모리 비용 미실측(추가 실행 없음); 시드 평균은 R=1 paired 2-arm 3.89 GPU-hour, R=4 2×2 31.1 GPU-hour(문서 실측), R=2·메모리·연산 미실측.
- **막는 것**: seed std 0.0021~0.0083이 승격 임계 +0.0030보다 크므로 단일 적합 비교는 판별 불가; 4 GPU-hour로는 R=1 paired만 가능하고 R=4는 7.8배 초과; 적합 분산을 포함한 승격 절차가 미승인; 분산의 원인이 seed인지 subsampling/PCA인지 미확인. 분산 감소는 기대 Δ를 올리는 것이 아니라 Δ의 신뢰구간을 좁히는 것이므로, 기대 Δ가 0이면 성능 개선과 다름.
- **중단 기준**: 분산 감소가 R≥2 또는 4 GPU-hour 초과를 요구하면 중단; 시드 고정만으로 “분산 감소”를 주장하면 중단(단일 적합 선택); 기대 Δ가 0인데 신뢰구간만 좁아지는 결과를 승격 근거로 쓰면 중단.

## 실측 4: 건드리지 않은 축 — 적합 분산 감소(시드 고정/시드 평균)
- **건드려야 하는 코드·설정**: `scripts/eval_v121.sh`, `eval_seal_tasks.sh`, `evaluate_pure.py`(인용)의 seed/반복 제어; 공식 7-branch YAML 경로 문서에 없음; PCA·subsample 사양 문서화 경로 없음.
- **비용**: 시드 고정: 추가 GPU·CPU·메모리 비용 미실측(추가 실행 없음); 시드 평균: R=1 paired 2-arm 3.89 GPU-hour, R=4 2×2 31.1 GPU-hour; R=2·메모리·연산 미실측.
- **막는 것**: `D-048` 이후 적합 분산 보고는 필수지만 분산 감소 절차는 미승인; 4 GPU-hour 예산에서 R=1 paired만 가능; `PCA·subsample 사양이 문서에 없다`로 결정론적 재현의 대상이 불명확; 과거 여섯 러너가 같은 설정으로 수렴해 arm 차이가 없었으므로 적합 변동만 남음. 건드리지 않은 이유는 단순 망각이 아니라 예산·절차 미승인.
- **중단 기준**: 분산 감소가 승격 판정 절차 승인 없이 수행되면 중단; R=1 단일 시드 고정만으로는 적합 분산을 줄였다고 보고하면 중단; 4 GPU-hour 초과하는 시드 평균을 시작하면 중단.

## 실측 5: 4 GPU-hour 안에서 판별 가능한 후보
- **건드려야 하는 코드·설정**: paired 2-arm은 baseline과 candidate를 `evaluate_pure.py`와 각 YAML 설정으로 실행; 공식 7-branch YAML 경로와 candidate config 경로는 문서에 없음; 러너는 `scripts/eval_v121.sh`/`eval_seal_tasks.sh`(인용); provenance writer 복구 전에는 verified 판정 불가.
- **비용**: 5-branch fold 20.0초 실측 기준, arm 1회 Primary 7 350-fold 1.94 GPU-hour, paired 2-arm 700-fold 3.89 GPU-hour; 7-branch fold 시간은 미실측이므로 7-branch paired 2-arm이 4 GPU-hour 내인지 미실측. 메모리·연산 미실측.
- **막는 것**: 공식 기준 7-branch의 fold 적합 시간이 미실측; baseline이 `unverified-hold`이고 저장 예측에 provenance가 없어 과거 baseline을 paired 비교에 쓸 수 없음; 오염 검사 재설계 미완료; 3.89h는 예산의 97%로 오버헤드·재시도 여유 없음.
- **중단 기준**: 7-branch 700-fold가 4 GPU-hour를 초과하면 paired 확증을 중단; baseline arm을 새로 돌리지 않고 과거 예측을 쓰면 중단; provenance가 기록되지 않으면 중단; 게이트 ①에서 max |r| > 0.6 또는 eff.rank < 4.02이면 성능 조회 전에 중단.

질문: 공식 7-branch 구성의 fold 적합 1회 시간, 공식 7-branch YAML 경로, 집계 함수와 seed/적합 반복 인자의 구체적 코드 경로는 무엇인가?

### R1 (반증)

## 반론 1
- **공격 대상**: "이 회차에서 아직 시도되지 않은 축을 찾으면 Primary 7 근거로 모델 개선 후보를 검증할 수 있다"는 전제.
- **더 단순한 설명**: 현재 막히는 것은 새 축 부재가 아니라 계측·provenance 무효다. `D-050`은 공식 기준과 승격 근거를 `unverified-hold`로 두고, `D-053`은 여섯 러너가 같은 7-branch 설정으로 수렴해 과거 arm 차이를 구성 차이가 아닌 적합 변동으로 만든다. 따라서 "새 축이 없다"가 아니라 "유효한 paired 비교를 만들 수 있는 계측이 없다"는 설명이 같은 관측(과거 비교 근거 없음, gate·promotion을 `verified`로 못 함)을 더 간단히 낳는다.
- **무엇이 참이면 이 주장이 무너지는가**: provenance writer, branch list·가중치·코드 해시·fold manifest 기록, `SCREEN_ONLY` 게이트 ①, 원시 margin에서의 독립 paired Δ 재계산이 복구되면, 새 축을 찾지 않아도 기존 후보를 검증할 수 있다. 그러면 "축 부족이 병목"이라는 주장이 무너진다.
- **이 반론의 한계**: 저장된 원시 margin이 충분해 provenance 없이도 paired Δ를 재계산할 수 있다면, 계측 복구가 선행 조건이 아니라 선택 사항이 된다. 문서에는 저장 예측 파일에 provenance가 없다고 되어 있으나, 원시 margin의 가용성은 명시되지 않았다.

## 반론 2
- **공격 대상**: "유효 랭크 3.52/7은 branch 추가 방향이 막혔다는 뜻이다" 또는 그 반대 "서로 다른 것을 재는 branch를 아직 못 만들었다는 뜻이다"를 유효 랭크 하나로 확정하는 주장.
- **더 단순한 설명**: 7개 branch가 같은 patch feature에서 closed-form ridge margin을 만들고, 위치·분포·형상 통계가 같은 표현공정과 ridge readout을 공유하면, 관측된 유효 랭크 3.52는 "새로운 측정이 불가능하다"기보다 "현재 margin 공간의 중복"을 나타낼 수 있다. 즉 branch 개념의 부재가 아니라 현재 feature·readout 공유에 의한 저랭크라는 설명이 같은 수치를 낳는다.
- **무엇이 참이면 이 주장이 무너지는가**: label-free 게이트 ①에서 후보 branch가 max |r| ≤ 0.6이고, 8-branch 유효 랭크가 현행 50.3% 비하락선인 4.02 이상이면, branch 추가가 구조적으로 막혔다는 주장이 무너진다. 반대로 여러 서로 다른 통계 후보가 모두 게이트 ①에서 상관으로 탈락하면 "막혀 있다"는 쪽이 강화된다.
- **이 반론의 한계**: 유효 랭크는 margin 수준 지표다. 후보가 게이트 ①을 통과해도 paired Δ가 +0.003을 내지 못하면 성능 개선에는 쓸 수 없다. 또한 branch별 기여도는 문서에 없으므로, 어떤 branch가 중복인지 특정할 수 없다.

## 반론 3
- **공격 대상**: "집계(trimmed mean voting)를 바꾸는 축은 아직 거의 탐색되지 않았다"는 주장.
- **더 단순한 설명**: `closed_axes.md`의 `CA-08`이 "집계 함수 변형 탐색"을 §214-V 근거로 `CLOSED`로 기록한다. 따라서 축 자체가 미탐색이 아니라, 기존 margin 위의 결합·부분집합 탐색이 선택 편향으로 종료됐고, `D-053` 이후 일부 과거 실행의 provenance가 무효라는 설명이 더 간단하다. "거의 탐색되지 않았다"는 말은 "유효한 paired 비교가 없다"는 사실과 "접근이 닫혀 있다"는 사실을 섞는다.
- **무엇이 참이면 이 주장이 무너지는가**: §214-V 집계 비교의 원시 margin·설정·선택 이력이 남아 있고 재계산 가능하면, 축은 이미 탐색된 것이므로 "미탐색" 주장이 무너진다. 반대로 그 산출물도 러너 폴백으로 같은 구성에 수렴해 구성 귀속이 불가능하면, "유효한 탐색 없음"으로만 읽어야 한다.
- **이 반론의 한계**: `CA-08`의 경계 밖에는 PSW·TGW처럼 새 정보를 쓰는 결합 연구가 허용된다. 만약 제안이 기존 margin 재결합이 아니라 예측분산·context 기하 등 새 정보를 쓰는 사전 선언된 제한 탐색이면, "닫힌 축"이라는 반론은 적용되지 않는다.

## 반론 4
- **공격 대상**: "적합 분산이 승격 임계보다 큰 상황에서, 분산을 줄이는 것이 성능을 올리는 것보다 나은 투자다"는 주장.
- **더 단순한 설명**: seed std 0.0021~0.0083은 단일 적합의 Δ가 때로는 임계 +0.0030보다 작고 때로는 큰 불확실성을 가진다는 뜻이지, 분산 감소가 성능 개선보다 낫다는 뜻은 아니다. 더 간단한 설명은 "관측된 차이의 상당 부분이 뽑기일 수 있으므로 paired fold, seed 반복, provenance가 필요하다"는 것이다. 분산 감소는 추정 정밀도를 바꾸고, 성능 개선은 Δ 평균을 바꾼다.
- **무엇이 참이면 이 주장이 무너지는가**: 후보의 진짜 Δ가 0이거나 +0.003 미만이면, 아무리 분산을 줄여도 승격이 되지 않으므로 분산 감소가 더 나은 투자가 아니다. 반대로 여러 후보의 진짜 Δ가 +0.003 근처에 있고 현재 seed std 때문에 판별 불가라면, 분산 감소가 선행 투자일 수 있다. 문서에는 후보별 Δ 분포가 없으므로 후자를 입증할 근거가 없다.
- **이 반론의 한계**: 만약 주요 후보들의 paired Δ가 모두 +0.003 근처이고, 비교별 seed std가 +0.0030을 넘는다고 실측되며, 예산이 seed 평균을 허용하면, 분산 감소가 승격 판정을 가능하게 하는 선행 투자일 수 있다. 현재 문서에는 후보별 Δ와 비교별 seed std가 없어 이 조건을 확인하지 못했다.

## 반론 5
- **공격 대상**: "아직 아무도 건드리지 않은 축이 있고, 그 이유가 그냥 생각이 안 난 것이다"라는 주장.
- **더 단순한 설명**: 문서에 남은 축들은 대부분 닫힌 축, 재개 조건, 또는 계측·예산 제약으로 설명된다. `CA-08`(집계), `CA-09`(평균 재가중), `CA-10`(형상 변형), `CA-11`(ICC), `CA-13`(retrieval), `CA-05`(Top-K), `CA-06`(Fisher 기저) 등은 기각 기전과 경계가 기록되어 있다. `D-050`/`D-053`은 gate·promotion을 `verified`로 못 하게 하고, 4 GPU-hour는 반복 확증을 제한한다. 따라서 "생각이 안 난 것"보다 "닫혔거나 검증 경로가 막혀 있다"는 설명이 더 간단하다.
- **무엇이 참이면 이 주장이 무너지는가**: `closed_axes.md`에 없고, `D-050`/`D-053`의 계측 무효와 무관하며, 4 GPU-hour 안에 label-free 게이트 ①이나 paired Δ를 돌릴 수 있는 축이 실제로 존재하면, "건드리지 않은 이유가 타당하다"는 반론이 무너진다.
- **이 반론의 한계**: `closed_axes.md`는 기록된 축만 담는다. 레지스트리에 없는 새로운 축은 "아무도 생각하지 않았다"로 볼 수 있다. 다만 그것이 성능 개선에 쓸 수 있으려면 게이트 ①·②와 §4 paired Δ를 통과해야 하며, 현재 문서에는 그런 축의 후보별 비용·상관·Δ가 없다.

## 반론 6
- **공격 대상**: "4 GPU-hour 안에서 후보를 판별할 수 있다"는 주장.
- **더 단순한 설명**: 상태 문서의 5-branch 3과제 평균 fold 적합 20.0초를 그대로 쓰면, 한 arm Primary 7 350 fold는 1.94 GPU-hour, 2-arm paired 700 fold는 3.89 GPU-hour로 예산의 97%다. 즉 4시간은 "여러 후보 탐색"이 아니라 "단일 paired 비교 1회"에 가까우며, seed 반복이나 적합 분산 확증은 남지 않는다. 따라서 판별 가능해 보이는 것은 예산 산술이지, 통계적 확증 역량이 아니다.
- **무엇이 참이면 이 주장이 무너지는가**: 공식 7-branch 전체 7과제 50-fold의 fold 적합 1회 시간이 4h/700 이하이고, 한 후보의 paired Δ만 계산하면 된다면, 4h 안에 단일 paired Δ 산출은 가능하다. 그러나 `D-048` 이후 적합 분산을 함께 다루어야 하므로, seed 반복이 필요한 승격 판정으로는 여전히 부족하다.
- **이 반론의 한계**: 20.0초는 5-branch 3과제 평균이다. 7-branch 전체 과제가 더 느리면 2-arm paired도 4h를 초과할 수 있고, 반대로 더 빠르면 단일 비교가 여유로워진다. 문서에는 공식 7-branch 전체 fold 적합 시간이 없으므로, "불가능"을 확정하지는 못한다.

질문: 공식 7-branch Primary 7 50-fold의 fold 적합 1회 시간 또는 2-arm paired 700fold 총 소요가 측정되었는가? 상태 문서의 20.0초는 5-branch 3과제 평균인데 7-branch 전체에 적용 가능한가?
질문: seed std 0.0021~0.0083을 만든 무작위 단계가 무엇인가? PCA, subsampling, ridge, 학습 파라미터 중 어느 것이 seed에 의존하는가?
질문: §214-V 집계 함수 비교와 과거 arm 비교의 원시 margin·설정·선택 이력이 남아 있어 paired Δ를 독립 재계산할 수 있는가?

### A1 (감사)

## 점검 1: arm parity 실행의 구성 수렴
- **확인하려는 것**: `eval_v121.sh` 등 여섯 러너가 arm별 다른 branch 구성을 실제로 실행했는지, 아니면 `eval_seal_tasks.sh` 폴백으로 공식 7-branch로 수렴했는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: arm 이름만 다른 실행의 Δ가 구성 효과로 해석됨; 5-branch/7-branch 비교가 동일 7-branch의 적합 변동이 됨; 과거 arm 비교 근거 무효.
- **확인 방법**: `scripts/eval_v121.sh`, `scripts/eval_seal_tasks.sh`, 그리고 문서에 인용된 `eval_v120`·`eval_ct_alone`·`eval_ds_aug`·`run_v120_clean_loo_experiments`·`run_v121_salience_anchor` 러너의 `ICF_CONFIG`/`model:` 블록 추적; 폴백 실패 로그의 config hash; `evaluate_pure.py configs/baseline/v121_active.yaml`과 7-branch YAML 직접 실행 대조; `logs/official50/parity_*`의 provenance.

## 점검 2: 공식 기준선 0.6227의 provenance
- **확인하려는 것**: 공식 기준선 `0.6227`이 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시로 특정 구성에 귀속되는지, `unverified-hold` 상태인지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 기준선 구성 불명확하면 paired Δ의 기준 fold가 후보와 동일 구성/데이터인지 판별 불가; `0.6226` 재현과 `0.6227` 공식값의 차이를 코드/데이터 드리프트로 해석 불가; `+0.56%p` 승격 근거를 verified로 인용 불가.
- **확인 방법**: `PROJECT.md` §3.2/D-050, `talks/reports/2026-09-18_ru90_baseline_regen.md` §9~10, provenance writer 산출물, raw margin 독립 재계산, fold manifest hash, code hash.

## 점검 3: 오염 검사와 SCREEN_ONLY 독립성
- **확인하려는 것**: SCREEN_ONLY가 앙상블에 영향을 주지 않는지, 값 앵커 4자리가 구성 변화와 수치 드리프트를 구분할 수 있는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 5-branch 앵커 `0.4421`/`0.5553`이 직접 실행 `0.4426`/`0.5546`과 어긋나 오염 통과/실패가 아무 정보도 주지 못함; 7-branch 폴백 값 `0.4661`/`0.5408`을 5-branch로 오귀속 가능; gate1 SCREEN_ONLY 결과 무효.
- **확인 방법**: `PROJECT.md` §3.4, `evaluate_pure.py configs/baseline/v121_active.yaml`, 7-branch YAML, `SCREEN_ONLY=1` margin, provenance 기반 오염 검사, 코드/데이터 변경 이력.

## 점검 4: fold 분할 고정과 paired Δ
- **확인하려는 것**: Primary 7 50-fold가 PathoBench 공식 분할로 고정되고, 후보-기준선이 같은 fold에서 짝지어지는지, fold 재분할/증대가 없는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: fold 재분할로 MDE를 낮추거나, 후보/기준 fold가 달라 paired Δ가 과제 난이도·fold 효과에 오염됨; SEAL fold와 Primary fold 혼동.
- **확인 방법**: fold manifest 해시, `PROJECT.md` §3.1/§4, raw margin에서 `Δ_i = AUC_candidate_fold_i - AUC_baseline_fold_i` 재계산, `tests/test_query_independence.py`.

## 점검 5: query 라벨 누수
- **확인하려는 것**: query 라벨이 margin/readout/집계에 들어가지 않는지, query 교란이 margin을 바꾸지 않는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: query 라벨이 closed-form ridge나 trimmed mean voting에 들어가면 Primary 7 성능이 라벨 누수로 과대평가; query 50배 스케일/제거/순서변경 시 margin 변화.
- **확인 방법**: `tests/test_query_independence.py`, `margins()` 시그니처, readout 코드에서 query label 사용 grep, fold별 raw margin 재계산.

## 점검 6: context 라벨 사용과 oracle 선택
- **확인하려는 것**: context 라벨만 fold별 closed-form readout에 쓰이고, 성능 조회 후 강도/변형/집계를 고르는 oracle이 없는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: Primary 7 AUROC를 보고 강도·집계·branch를 선택한 뒤 같은 자료에서 승격 판정; CA-08 선택 편향; CA-06 라벨 기저 회전; CA-R1 과제별 강도 oracle.
- **확인 방법**: `closed_axes.md` CA-06/CA-08/CA-R1/I/J, 선택 이력, 사전 선언된 통계, gate1 성능 조회 이전, Dual Ridge 계수 산출 코드.

## 점검 7: 유효 랭크 3.52의 전처리·fit 범위
- **확인하려는 것**: 유효 랭크 `3.52/7`을 산출한 branch margin의 PCA·subsample·whitening·ridge fit 범위가 고정되어 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 유효 랭크가 전처리 사양에 따라 달라져 “branch를 더 넣는 방향이 막혔다”와 “서로 다른 것을 재는 branch를 못 만들었다”를 구분 불가; 8-branch 통과선 `eff.rank ≥ 4.02`가 같은 fit 범위에서 비교되지 않음.
- **확인 방법**: `scripts/analysis/branch_screen.py`의 covariance/whitening/PCA 코드, config YAML, raw margin, PCA/subsample 사양 문서, fold/과제 범위.

## 점검 8: 집계 변경 축의 선택 편향
- **확인하려는 것**: trimmed mean voting 대안 등 집계 변경 후보가 CA-08 경계 밖으로 사전 선언되고, 선택에 쓴 Primary 7 자료를 확증으로 재사용하지 않는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 기존 branch margin 위에서 집계 함수/부분집합을 탐색한 뒤 같은 Primary 7에서 최댓값을 승격 근거로 사용; 학습 파라미터 없는 집계 변경이라도 선택 편향 발생; fit variance 무시.
- **확인 방법**: `closed_axes.md` CA-08, `PROJECT.md` §4.1, 사전 선언된 집계 후보, paired Δ, 선택 이력.

## 점검 9: 적합 분산과 승격 절차
- **확인하려는 것**: seed refit 분산이 +0.003 승격 임계보다 큰 상황에서, 승격 판정에 적합 분산과 R(반복)을 포함하는 절차가 사전 고정/승인되어 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 단일 적합의 +0.003 초과를 개선으로 오인; 질문 본문에 제시된 seed std 범위 또는 문서의 RU-97 macro 폭 `0.0200`이 승격 판정에 반영되지 않음; paired Δ의 0 포함 구간을 무시.
- **확인 방법**: `PROJECT.md` §4/§4.1, `closed_axes.md` CA-14, `current_status.md` Open Issues, RU-97 §8, seed 반복 실험 설계, fold 분산과 적합 분산 분리.

## 점검 10: fold 적합 시간·예산·판별 가능 설계의 일치
- **확인하려는 것**: fold 적합 1회 시간 측정값, 2 arm paired `3.89h`, R=4 `31.1h`, “4 GPU-hour 내 확증 프로브 불가능” 결론, 질문 본문의 “측정된 적이 없다” 주장이 서로 일치하는지.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: `3.89h`가 4h의 97%로 산출됐으면서도 확증 프로브가 불가능하다고 결론; fold fit time 미측정이라 하면 산출 근거 없음; 4h 내 판별 가능 설계(단일 arm 350 fold, 2 arm 700 fold, R=4 5600 fold)를 문서만으로 확정 불가.
- **확인 방법**: `current_status.md` fold fit cost table, `talks/reports/2026-09-18_foldfit_cost.md`, 질문 본문 “측정된 적이 없다” 대조, 실제 GPU 4에서 5-branch fold fit 타이밍 재측정.

## 점검 11: SEAL 10 hold-out 사용 금지
- **확인하려는 것**: 모델 개선 방향이 Primary 7만으로 결정되고, SEAL 10 결과가 선택에 사용되지 않는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: SEAL 10 macro `0.6972`/ABMIL `0.7266`을 보고 branch/집계/강도를 고르면 holdout leak; Primary 7 개선이 holdout 개선을 보장하지 않음에도 SEAL로 방향 선택.
- **확인 방법**: `PROJECT.md` §1/§3.1, `current_status.md` “SEAL 10은 열지 않으며”, 결정 이력, 후보 선택 로그.

## 점검 12: branch별 기여도/점유율
- **확인하려는 것**: branch별 기여도/점유율이 raw margin에서 독립 재계산되어 현재 저장되어 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 유효 랭크 `3.52`와 branch 중복을 해석할 때 어떤 branch가 중복인지, SH/SJ 기여가 무엇인지 판별 불가; 승격/기각 근거가 aggregate만.
- **확인 방법**: raw margin, component occupancy output, provenance, `PROJECT.md` §2 완료 조건, `closed_axes.md` CA-10의 일부 점유율 인용 대조.

## 점검 13: trimmed mean voting 정의의 재현성
- **확인하려는 것**: trimmed mean voting의 정의가 코드와 산출물에 고정되어 재현되는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 문서/코드 집계 불일치, branch margin에서 최종 AUROC 재계산 불가, `0.6227` 재현 `0.6226` 차이 원인 불명.
- **확인 방법**: aggregation code, raw margins, config hash, `bash scripts/run_tests.sh`, `PROJECT.md` §3.2.

## 점검 14: 게이트/승격 임계값의 사전 고정
- **확인하려는 것**: 게이트/승격 임계값(max |r| 0.6, eff.rank ≥ 4.02, gate2 0.5/40/50 p<0.01, promotion +0.003)이 성능 조회 전에 고정되고 변경 이력이 있는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 결과 보고 후 임계값 완화/신설, ②b처럼 후속 기준 변경이 독립 논거 없이 반복, 후보 선별과 확증 혼동.
- **확인 방법**: `PROJECT.md` §4/§5/§5.1, 결정 이력 D-041/D-049, gate1 execution before performance.

## 점검 15: 건드리지 않은 축의 타당성
- **확인하려는 것**: provenance writer, PCA/subsample 사양, 적합 분산 승격 절차, 오염 검사 재설계가 건드리지 않은 축인지, 그 이유가 타당한지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 계측 복구 전 게이트/승격 비교가 verified로 수행 불가; PCA/subsample 사양 부재로 유효 랭크/branch 비교 재현 불가; 적합 분산 절차 미승인으로 단일 적합 차이 오인.
- **확인 방법**: `current_status.md` Open Issues, `PROJECT.md` §2, D-050/D-053, `closed_axes.md` CA-14.

질문: fold 적합 1회 20.0초 실측(`current_status.md`)과 질문 본문의 “측정된 적이 없다” 중 현재 정본은 무엇이며, 20.0초가 7-branch/Primary 7 전체 fold에 적용 가능한 측정 범위인가? 또한 유효 랭크 3.52와 branch별 기여도/점유율을 산출한 PCA·subsample·whitening 사양, raw margin 파일, seed refit std 범위의 출처와 fold/과제 범위는 어디에 있는가?

---

## Phase 2 — 교차

### T1 (이론)

- **가정**: P1·P2 전체가 Primary 7 paired Δ_macro `+0.0030` 개선이 SEAL 10 hold-out에서 ABMIL `0.7266` 초과로 전이되는 유효한 조작 변수라고 둔다.
- **이 가정이 깨지는 조건**: Primary 7의 과제 분포·난이도·fold 구성이 SEAL 10과 다르고, 개선이 Primary 7에 국한되거나, Primary 7 승격이 hold-out 개봉에서 목표 미달로 끝나면.
- **깨질 때 나타나는 관측 가능한 증상**: Primary 7에서 `+0.0030`을 충족해도 SEAL 10 macro-AUC가 `0.7266`에 도달하지 못함. 현재 SEAL 10이 유보되어 있으므로 이 회차에서는 판별 불가.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: PROJECT.md §1은 Primary 7의 개선이 hold-out 개선을 보장하지 않는다고, ABMIL `0.7266`, ICF v120 `0.6972`, ICF와의 거리 `−0.0294`라고 기록한다.

- **가정**: P1-1 fold-global rank-normalized trimmed mean은 (a) 7개 branch margin이 모두 같은 방향의 slide score, 즉 높은 margin이 positive evidence이고, (b) branch별 절대 스케일·분산은 AUROC 순서에 불필요하거나 해로우며 fold-global rank가 label-free 공통 단위이고, (c) 저장된 7-branch margin을 공식 7-branch 구성으로 provenance 없이 재집계해도 되고, (d) 단일 후보 사전고정으로 CA-08 선택 편향을 피한다고 둔다.
- **이 가정이 깨지는 조건**: branch 또는 과제에서 margin 방향이 반대이거나, rank 변환이 ridge 해의 크기·신뢰도 정보를 제거해 slide 순서를 악화시키거나, fold 내 slide 수·ties로 rank 분해능이 부족하거나, 저장 margin의 branch list·가중치·코드 해시가 특정 구성에 귀속되지 못하거나, rank 변환이 기존 margin 결합의 반복으로 CA-08 경계 안에 들어가거나, 실제로는 여러 집계 변형 중 하나를 Primary 7 진단 뒤에 고른 경우.
- **깨질 때 나타나는 관측 가능한 증상**: paired Δ_macro가 0 또는 음수, 기존 악화 과제(`Histologic_Grade`, `progression_regression`, `PBRM1`)의 부호가 개선되지 않거나 반대, branch 단독 AUROC 0.5 미만 과제가 더 악화, rank score ties 증가, 재집계 산출물이 provenance 불일치로 실행 무효, CA-08 선택 편향으로 실행 무효. Δ가 양이어도 rank 변환이 outlier 경로를 줄였다는 기전 입증이 아니다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: PROJECT.md §5.1은 CV ARID1A `0.4308`, QA ARID1A `0.4692`, BD 3개 과제가 우연 이하로 기록되어 같은 방향·유효 증거 가정과 충돌한다. D-050/D-053은 저장 예측에 provenance가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없고 여섯 러너가 같은 7-branch 설정으로 수렴했다고 기록한다. current_status는 7-branch fold 적합 시간이 아니라 5-branch `20.0`초와 2 arm `3.89` GPU-hour만 기록한다. CA-08은 기존 margin 결합 반복을 경계 안으로, 제한된 결합 연구만 조건부로 열어 두었다.

- **가정**: P1-2 deterministic branch construction은 seed std `0.0021~0.0083`의 상당 부분이 token subsampling·PCA/SVD initialization·random permutation·GPU reduction order 같은 stochastic step에서 오며, 이를 제거하거나 고정 manifest로 바꾸면 mean을 크게 바꾸지 않고 적합 분산만 줄여 `+0.0030` 판별력이 오른다고 둔다. 분산 저감이 성능 개선보다 나은 투자라고도 둔다.
- **이 가정이 깨지는 조건**: seed 변동이 fold data composition, context selection, ridge solver/regularization, 또는 후보 변경과 상호작용하는 경로에서 오거나, deterministic full-token/fixed manifest가 branch 분포와 mean을 바꾸거나, 코드에 해당 stochastic step이 없거나, 7-branch deterministic 실행 비용이 4 GPU-hour를 초과하거나, provenance가 없어 seed 실행을 구성에 귀속시킬 수 없거나, 분산이 줄어도 true Δ가 `+0.0030` 미만이면.
- **깨질 때 나타나는 관측 가능한 증상**: deterministic 재적합 후 seed std가 `+0.0030` 이하로 줄지 않거나, paired Δ의 부호·크기가 seed별로 달라지거나, mean이 악화되어 분산 저감과 성능 개선이 상충하거나, seed std는 줄어도 paired Δ가 `+0.0030` 미만으로 승격 불가, 코드 감사에서 stochastic step 부재로 실행 무효, 7-branch 비용 미측정으로 판별 불가.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: D-050/D-053은 저장 산출물에 provenance가 없어 과거 실행을 특정 구성에 귀속시킬 수 없다고 기록한다. current_status는 PCA·subsample 사양이 문서에 없고, 7-branch fold 적합 시간은 문서에 없으며 5-branch `20.0`초와 2 arm `3.89` GPU-hour만 기록한다. PROJECT.md §1은 Primary 7 개선이 hold-out 개선을 보장하지 않는다고 기록한다.

- **가정**: P1-3 residualized mode-count branch는 유효 랭크 `3.52/7`이 기존 branch의 중복 때문이며, SJ가 1D radius로 붕괴시키며 버린 성분의 mode count에 독립 정보가 있고, 기존 7 margin에 회귀한 잔차는 공유 변동만 제거해 독립 증거를 남긴다고 둔다. 게이트 ① 통과가 “서로 다른 것을 재는 branch를 만들 수 있다”는 증거가 된다고도 둔다.
- **이 가정이 깨지는 조건**: mode count가 whitened cloud 등방성·부분추출·염색·토큰수에 지배되어 변별 자유도가 없거나, residualization이 같은 fold margin을 사용해 직교성을 인공적으로 만들거나, 게이트 ① 통과 후 게이트 ②a/b를 못 통과하거나, ensemble가 직교 branch를 균등 기대보다 낮게 쓰거나, screen/fit 비용이 4 GPU-hour를 초과하는 경우.
- **깨질 때 나타나는 관측 가능한 증상**: `max |r| ≤ 0.6`과 eff.rank `≥ 4.02`를 만족해도 단독 AUROC 0.5 초과 전 과제 또는 40/50 fold를 못 넘기거나, paired Δ_macro 0/음수, branch 점유율이 균등 기대 `28.6%` 미만, seed·부분추출에 따라 margin 불안정, 비용 초과로 판별 불가.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: CA-10은 형상 계열 병목이 절사가 아니라 정보량이라고, §219는 신규 직교 브랜치가 우선 절사된다는 통념이 반증됐다고(7-branch에서 BS `22.3%` / SH `22.3%` < `28.6%`) 기록한다. RU-85 G는 whitened 32D 각도 산포가 `H_q > 0.99`, 슬라이드 간 std `0.0017`로 사실상 등방이라고 기록한다. CA-09는 RM이 `max |r| = 0.690`으로 탈락한 원인이 CV가 256차원 투영 공간 전체를 이미 읽는 중복이었다고 기록한다.

- **가정**: P2-1 PSW는 closed-form ridge가 산출하는 query slide별 예측분산이 branch margin 오류의 label-free 신뢰도 대리변수이며, 역분산 가중은 aggregate 오류를 줄이고, branch 오류는 독립이거나 최소한 예측분산이 과제 난이도·context 규모·염색 난이도와 혼동되지 않는다고 둔다.
- **이 가정이 깨지는 조건**: 예측분산이 task difficulty·context size·염색 난이도의 대리로만 작동하거나, branch margin 오류가 상관되어 있거나, 가중치가 hard slide/fold를 과소평가해 paired Δ의 estimand를 바꾸거나, 여러 가중 변형 중 Primary 7에서 최댓값을 고르거나, 예측분산이 저장되지 않거나 재현되지 않는 경우.
- **깨질 때 나타나는 관측 가능한 증상**: paired Δ_macro 0/음수, fold/seed Δ 변동 감소 없이 과제별 악화, 예측분산과 fold AUROC 또는 context 규모·과제 난이도의 높은 상관, 기존 악화 과제 부호 개선 없음, 선택 편향으로 실행 무효. Δ가 양이어도 역분산 가중이 신뢰도를 반영했다는 기전 입증이 아니다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: PROJECT.md §4.1은 fold별 AUROC 산포의 대부분이 과제 난이도 차이이고, 절대 macro SE의 설계 효과가 `5.12`배라 대응 비교만 정밀하다고 기록한다. CA-08은 PSW를 비저촉으로 두되 “예측분산의 유효성”을 남은 검증으로 남겨 두었다. current_status는 7-branch fold 적합 시간과 margin·예측분산 저장 경로가 문서에 없으며 5-branch `20.0`초, 2 arm `3.89` GPU-hour만 기록한다.

- **가정**: P2-2 CRN paired 비교는 seed std `0.0021~0.0083`의 상당 부분이 baseline과 candidate에 공통인 무작위 요인이며, 같은 seed·subsample·context selection을 쓰면 paired Δ에서 상쇄되어 `+0.0030` 판별력이 오르고, 4h 안의 단일 CRN 비교가 승격 판정에 충분한 증거가 된다고 둔다.
- **이 가정이 깨지는 조건**: seed 변동이 후보 변경과 상호작용하는 무작위 경로에서 오거나, 공통 random stream이 baseline/candidate의 stochastic path를 정렬하지 못하거나, 승격에 독립 적합 분산 또는 R>1이 요구되거나, 7-branch 2 arm 비용이 4 GPU-hour를 초과하는 경우.
- **깨질 때 나타나는 관측 가능한 증상**: CRN paired Δ의 분산이 독립 seed 비교보다 안 줄거나, CRN에서는 Δ가 `+0.0030`을 넘는데 독립 seed에서는 부호·크기가 달라지거나, 적합 분산 절차 미승인으로 실행 무효, 7-branch 비용 미측정으로 판별 불가.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: current_status는 적합 분산을 포함한 승격 판정 절차가 미승인이고 R도 미정이며, 4 GPU-hour로는 확증이 불가능하다고, 2 arm `3.89` GPU-hour, R=4 2×2 `31.1` GPU-hour라고 기록한다. D-048은 단일 적합 per-fold 구간이 실제 불확실성을 과소평가한다고 기록한다. seed 무작위 요인의 출처는 문서에 없다.

- **가정**: P2-3 robust LID branch probe는 LID가 patch cloud의 국소 내재차원을 읽고 기존 7 branch와 다른 fold 차이를 만들며, 슬라이드별 동일 토큰 수 정규화와 `m=1024`/`m=4096` 이중 계산 평균은 RU-85의 부분추출 민감성을 해소하고, 게이트 ① 통과/실패가 유효 랭크 `3.52`의 두 해석을 판별한다고 둔다.
- **이 가정이 깨지는 조건**: LID가 토큰 수·염색·배경 노이즈에 지배되거나, `m=1024`/`m=4096` 마진의 `|r| < 0.8`으로 사전 기각되거나, 게이트 ① 통과 후 게이트 ②a/b를 못 통과하거나, LID가 기존 family와 중복되거나, SCREEN_ONLY fold 비용이 4 GPU-hour를 초과하는 경우.
- **깨질 때 나타나는 관측 가능한 증상**: `|r| < 0.8`로 성능 조회 없이 기각, 게이트 ② 실패, paired Δ_macro 0/음수, LID margin과 log N 또는 염색 대리변수의 높은 상관, screen 비용 초과로 판별 불가.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: RU-85는 LID 부분추출 크기만 바꾼 두 마진의 `|r|(m=4096, m=1024)=0.314`로 강건성 임계 `0.8`을 밑돈다고 기록한다. CA-10은 형상 계열 병목이 정보량이라고, CA-09는 위치 계열 중복으로 유효 랭크가 낮았다고 기록한다. current_status는 7-branch fold 적합 시간과 SCREEN_ONLY 비용이 문서에 없으며 5-branch `20.0`초만 기록한다.

### M1 (실측)

## 실측 1: P1-1 fold-global rank-normalized trimmed mean
- **건드려야 하는 코드·설정**: 집계 함수(trimmed mean) 구현 파일. 문서에 구체적 경로 없음. `evaluate_pure.py`가 margin 산출·평가 진입점이지만 집계 함수가 그 안에 있는지 별도 모듈인지 미확인. `configs/baseline/v121_active.yaml`의 집계 관련 키도 미확인.
- **비용**: margin 1회만 있으면 재집계 GPU 0. 그러나 7-branch margin 1회 생성 비용은 **미실측**. 5-branch fold 20.0s·2 arm 3.89 GPU-hour는 5-branch 측정값이며 7-branch에 외삽 불가. 7-branch 2 arm paired가 4h 이내인지 **미실측**.
- **막는 것**: (1) 7-branch fold 시간 미실측 → 4h 내 margin 1회 생성 보장 불가. (2) `D-050` provenance 결함: 저장된 예측 파일에 branch list·가중치·코드 해시 없음 → "provenance-locked 7-branch margin 1회" 전제가 현재 충족 안 됨. (3) 7개 branch margin의 방향 정합(높은 margin = positive)이 **미확인**. rank 변환은 monotone이므로 branch 내부 순서는 보존하지만, 어떤 branch가 음의 방향이면 fold-global rank가 방향을 잃음. (4) `D-053` 이후 폴백이 큰 소리로 실패하도록 수정됐으나, arm parity 실험 결과 미읽힘 → 현재 7-branch 실행이 실제로 7-branch로 수렴하는지 확인 전.
- **중단 기준**: 7-branch margin 1회 생성이 4h를 초과하면 중단. provenance(branch list·코드 해시)가 산출물에 기록되지 않으면 실행 무효. 7개 branch 중 방향이 다른 것이 확인되면 rank 변환 설계 자체가 실행 무효.

## 실측 2: P1-2 deterministic branch construction
- **건드려야 하는 코드·설정**: token subsampling, PCA/SVD initialization, random permutation, GPU reduction order. 구체적 파일 경로 **문서에 없음**. `src/` 하위 어디에 stochastic step이 있는지 코드 감사 전 미확인.
- **비용**: 코드 감사 시간 **미실측**. deterministic full-token 계산의 fold 시간·메모리 **미실측**. 고정 stratified subsampling manifest 생성·검증 시간 **미실측**. B200 183,359 MiB에서 full-token이 OOM인지 **미실측**.
- **막는 것**: (1) seed std 0.0021~0.0083의 원인이 **문서에 없음**. stochastic step이 실제로 존재하는지, fold data composition에서 오는지 구분 불가. (2) full-token 계산이 가능한지(메모리·시간) 미확인. (3) deterministic 교체가 branch margin 분포를 바꿔 mean을 악화할 수 있음 — 분산 저감과 성능 개선이 상충할 수 있고, 그 크기는 **미실측**. (4) 이 제안이 성공해도 성능 개선 증거가 아니라 판정 정밀도 개선 증거만 됨. +0.0030 미만 후보를 "판별 불가"에서 분리하는 수단이지, ABMIL 초과에 직접 기여하는 경로는 **미실측**.
- **중단 기준**: 코드 감사에서 stochastic step이 없으면 실행 무효. full-token OOM이면 중단. deterministic fold 시간이 5-branch 20.0s를 크게 초과해 2 arm이 4h를 넘으면 중단.

## 실측 3: P1-3 residualized mode-count branch
- **건드려야 하는 코드·설정**: `sj.py`(구 `shj.py`)의 `whitened.norm(dim=1)` 이후 버리는 성분. mode count estimator(구체적 구현 없음). residualization: 기존 7 branch margin에 closed-form ridge 회귀. `scripts/analysis/branch_screen.py`로 게이트 ①.
- **비용**: mode count 계산 fold 시간 **미실측**. SCREEN_ONLY fold 비용 **미실측**. residualization ridge 해 비용 **미실측**. 4h 내 Primary 7 전량 screen+fit **미실측**.
- **막는 것**: (1) `G` 경계 실측: `cptac_luad/KRAS_mutation` 50 fold에서 whitened 32D 각도 산포 스펙트럼의 정규화 엔트로피 H_q가 슬라이드 100%에서 0.99 초과, 슬라이드 간 std 0.0017. mode count도 같은 whitened cloud에서 읽으면 **같은 등방 벽에 막힐 가능성**. 단, G 측정은 1과제·32D·이 백색화 구성이므로 Primary 7 전량·다른 차원에서는 **미실측**. (2) LID(RU-85)처럼 부분추출·시드 민감할 가능성 — mode count estimator의 재현성은 **미실측**. (3) residualization이 noise만 남기면 eff.rank는 오를 수 있으나 게이트 ② 정보량 **미실측**. (4) `D-050` 이후 SCREEN_ONLY 게이트 ① 실행 경로가 무효였으며, `D-053`에서 폴백 수정·arm parity 실험 진행 중이나 결과 미읽힘 → 게이트 ①을 실제로 돌릴 수 있는 상태인지 **미확인**.
- **중단 기준**: 게이트 ①에서 max |r| > 0.6이면 성능 조회 없이 종료. 8-branch eff.rank < 4.02이면 종료. mode count의 fold 간 재현성(시드·부분추출)이 확인되지 않으면 게이트 ②로 진행 불가.

## 실측 4: P2-1 PSW (prediction-variance weighted trimmed mean)
- **건드려야 하는 코드·설정**: closed-form ridge fit에서 query slide별 예측분산 산출. trimmed mean에 역분산 가중. ridge fit 구현 파일 **문서에 구체적 경로 없음**. 예측분산이 현재 코드에서 산출·저장되는지 **미확인**.
- **비용**: 한 arm에서 margin+예측분산 저장 후 오프라인 재집계 GPU 0. 그러나 7-branch 1 arm fold 시간 **미실측**. 2 arm paired 3.89 GPU-hour는 5-branch 기준. 7-branch 2 arm **미실측**.
- **막는 것**: (1) ridge fit이 예측분산을 실제로 산출하는지, 그 값이 margin 신뢰도의 대리변수인지 **미실측**. (2) branch margin 오류의 독립성 **미확인** — 독립 오류 가정에서 역분산 가중이 aggregate 오류 분산을 줄인다는 기전은 미확인. (3) 7-branch fold 시간 미실측 → 4h 내 2 arm 보장 불가. (4) `CA-08` 경계: PSW는 `D-022`로 비저촉이나, "여러 가중 변형을 탐색한 뒤 최댓값을 고르는 절차" 금지. 단일 사전고정 변형 1개만. 이 조건을 코드에서 강제하는 메커니즘 **미확인**.
- **중단 기준**: ridge fit에서 예측분산이 산출되지 않으면 실행 무효. 7-branch 2 arm이 4h 초과면 중단. 가중 변형을 2개 이상 Primary 7에서 비교하면 `CA-08` 선택 편향으로 실행 무효.

## 실측 5: P2-2 CRN (common random numbers) paired 비교
- **건드려야 하는 코드·설정**: seed·subsample·context selection을 baseline과 candidate 간 동일하게 고정하는 설계. 구체적 고정 대상 코드 경로 **문서에 없음**. seed가 어떤 무작위 연산에 영향을 주는지 **미확인**.
- **비용**: 2 arm Primary 7 = 3.89 GPU-hour (5-branch fold 20.0s 기준). 7-branch 2 arm **미실측**. R=4 2×2 = 31.1 GPU-hour (5-branch 기준) → 4h 초과, 실행 무효.
- **막는 것**: (1) seed std 0.0021~0.0083을 만드는 무작위 요인(subsampling, PCA, context selection 중 어디)이 **문서에 없음**. CRN이 상쇄할 공통 요인을 특정하지 못하면 설계 자체가 불확실. (2) seed 변동이 baseline과 candidate 간 공통이 아니라 후보 변경과 상호작용하는 무작위 경로 차이이면 CRN 효과 0 — **미실측**. (3) 7-branch fold 시간 미실측 → 2 arm이 4h를 초과할 수 있음. (4) 승격 판정에 독립 적합 분산이 필수로 요구되면 단일 CRN 비교만으로는 **실행 무효** — `current_status`에 "적합 분산을 포함한 승격 판정 절차가 미승인"으로 기록됨.
- **중단 기준**: seed 변동의 원인이 특정되지 않으면 CRN 설계의 유효성 **미확인**으로 중단. 7-branch 2 arm이 4h 초과면 중단. 독립 적합 분산 보고가 승격 필수로 승인되면 단일 CRN은 실행 무효.

## 실측 6: P2-3 robust LID branch probe
- **건드려야 하는 코드·설정**: LID branch(ID_TwoNN). m=1024·m=4096 두 크기. 슬라이드별 동일 토큰 수 정규화. `scripts/analysis/branch_screen.py`로 게이트 ①.
- **비용**: k-NN 추가 fold 시간 **미실측**. 5-branch fold 20.0s를 "크게 넘으면" 4h 불가 — "크게"의 임계 **미실측**. Primary 7 전량 SCREEN_ONLY 총 시간 **미실측**.
- **막는 것**: (1) `H` 경계 실측: RU-85에서 |r|(m=4096, m=1024) = **0.314**. 제안의 사전 기각 조건은 "|r| ≥ 0.8이면 평균, 아니면 성능 조회 없이 기각". 0.314 < 0.8이므로 **사전 기각 조건이 이미 충족됨**. 이 probe는 실행 전에 기각. (2) RU-85의 측정은 결정론적 스트라이드 부분추출 하의 것이며, 제안의 "슬라이드별 동일 토큰 수 정규화"가 이 0.314를 바꿀지는 **미실측**. (3) k-NN 비용 미실측 → 4h 내 Primary 7 전량 screen **미실측**. (4) 게이트 ① 실행 경로 상태: `D-050` 이후 무효, `D-053`에서 폴백 수정·arm parity 진행 중이나 결과 미읽힘 → 게이트 ① 실행 가능 여부 **미확인**.
- **중단 기준**: |r|(m=4096, m=1024) < 0.8이면 성능 조회 없이 기각. RU-85 실측 0.314로 이 조건이 이미 성립하므로, 동일 토큰 수 정규화가 이 값을 바꾼다는 **실측 증거 없이** 실행하면 안 됨. k-NN fold 시간이 5-branch 20.0s의 2배를 초과하면 4h 내 Primary 7 전량 screen 불가 → 중단.

### R1 (반증)

## 반론 1
- **공격 대상**: P1 제안 1의 “fold-global rank-normalized trimmed mean이 branch margin의 스케일·꼬리 차이를 제거해 AUROC를 개선할 수 있다”는 주장.
- **더 단순한 설명**: AUROC는 최종 slide score의 순서만 사용하므로, 7개 branch margin이 이미 같은 slide 순서 정보를 중복해서 담고 있다면(유효 랭크 3.52/7) rank 변환은 새 정보를 추가하지 않고 기존 3.52차원 증거의 재가중일 수 있다. trimmed mean이 이미 slide당 최저·최고 1개 branch를 절사하므로, branch outlier가 최종 순서에 미치는 영향이 제한적일 수도 있다.
- **무엇이 참이면 이 주장이 무너지는가**: (a) branch margin의 절대 크기가 ridge 해의 신뢰도·크기를 담고 rank 변환이 이를 버리면 Δ가 0 또는 음수일 수 있다. (b) 7개 branch margin의 방향 정합이 라벨 없이 보장되지 않으면 rank 변환 전 부호 정렬이 필요하고, 없으면 실행 무효다. (c) fold 내 slide 수/ties로 rank score 분해능이 떨어져 paired Δ를 +0.0030과 구분 못 하면 판별 불가다. (d) Primary 7에서 이 후보 실패 후 다른 rank/집계 변형을 골라 최댓값을 쓰면 CA-08 선택 편향으로 실행 무효다.
- **이 반론의 한계**: branch margin의 스케일·꼬리가 실제로 slide 순서를 왜곡하고, 단일 사전고정 rank-normalized trimmed mean이 paired Δ_macro +0.0030을 넘기면 이 반론은 틀린다. 단, 개선 기전은 미확인이며 Primary 7 paired Δ로만 판정한다.

## 반론 2
- **공격 대상**: P1 제안 2의 “seed std 0.0021~0.0083의 일부는 stochastic step에서 오므로 deterministic canonical ordering으로 교체하면 적합 분산이 줄고 +0.0030 판별력이 오른다”는 주장.
- **더 단순한 설명**: seed std는 stochastic estimator의 실제 적합 변동일 수 있다. deterministic fixed manifest는 seed만 바꿀 때의 흔들림을 0에 가깝게 만들 수 있지만, 그 fixed subsample의 대표성이나 fold 간 변동은 줄이지 못한다. 즉 “분산 저감”이 아니라 “무작위화를 숨기는 것”일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 코드 감사에서 stochastic step이 없으면 실행 무효다. seed std가 fold composition, ridge solver, regularizer 효과에서 오면 deterministic 교체로 줄지 않아 반박이다. full-token 또는 고정 manifest가 branch margin 분포를 바꿔 mean을 악화하면 분산 저감과 성능 개선이 상충한다. 승격 판정에 독립 적합 분산이 필수로 요구되면 단일 deterministic 적합만으로는 실행 무효다.
- **이 반론의 한계**: stochastic step이 seed std의 상당 부분을 만들고, deterministic full-token 또는 고정 manifest가 mean을 크게 악화하지 않으면서 seed std를 +0.0030 아래로 줄이면 이 반론은 틀린다. 단, 성능 개선 증거가 아니라 판정 정밀도 증거만 된다.

## 반론 3
- **공격 대상**: P1 제안 3의 “mode count를 기존 7 branch margin에 회귀시켜 residualize하면 게이트 ①으로 유효 랭크 3.52의 두 해석을 가를 수 있다”는 주장.
- **더 단순한 설명**: residualization은 후보 margin을 기존 7 branch margin의 span에서 뺀 것이므로, 잔차는 기존 margin과 직교하도록 만들어진다. 따라서 max |r| ≤ 0.6과 eff.rank 상승은 새 정보의 증거가 아니라 residualization의 수학적 결과일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: mode count에 정보가 없어도 noise만 남으면 게이트 ①은 통과하고 게이트 ②는 실패할 수 있다. 이 경우 “rank는 올라가지만 정보 없음”이지 “서로 다른 측정이 가능”을 지지하지 못한다. mode count estimator가 부분추출/시드에 민감하면 판별 불가다. 게이트 ① 실패는 후보 한 건의 직교화 실패일 뿐, 다른 family에서 새 branch가 불가능하다는 증거가 아니다. SCREEN_ONLY 비용이 4h를 넘으면 실행 무효/판별 불가다.
- **이 반론의 한계**: mode count가 안정적으로 기존 7 branch와 직교하고, 게이트 ②a/b를 통과하면 “재는 대상이 부족했다”는 해석에 관측 증거가 된다. 단, 성능 개선은 별도 paired Δ로만 판정한다.

## 반론 4
- **공격 대상**: P2 제안 1의 “closed-form ridge fit의 query slide별 예측분산이 margin 신뢰도의 label-free 대리변수이므로 역분산 가중 trimmed mean이 AUROC를 개선하거나 Δ 변동을 줄인다”는 주장.
- **더 단순한 설명**: 예측분산은 task difficulty, context size, 염색/배경 노이즈, ridge 조건화 같은 slide-level 난이도의 대리에 불과할 수 있다. 이 경우 역분산 가중은 노이즈가 아니라 신호를 절단하거나 과제별 난이도 편향을 강화한다. branch margin 오류가 서로 상관되어 있으면 독립 오류 가정의 역분산 가중은 aggregate 오류를 줄이지 못한다.
- **무엇이 참이면 이 주장이 무너지는가**: 예측분산이 실제 margin 오류와 관련이 없거나 역상관이면 Δ가 0 또는 음수다. 여러 가중 변형을 Primary 7에서 골라 최댓값을 쓰면 CA-08 선택 편향으로 실행 무효다. 한 arm에서 margin과 예측분산을 저장해 재집계할 경로가 없으면 2 arm이 필요하고, 7-branch fold 시간이 문서에 없어 4h 내 실행이 보장되지 않아 판별 불가다.
- **이 반론의 한계**: 예측분산이 branch별 실제 오류를 label-free로 반영하고, 단일 사전고정 가중이 paired Δ_macro +0.0030을 넘기면 이 반론은 틀린다. 단, 기전은 미확인이다.

## 반론 5
- **공격 대상**: P2 제안 2의 “seed·subsample·context selection 등 무작위 요인을 baseline/candidate에 동일하게 고정하면 paired Δ의 분산이 줄어 +0.0030 판별력이 오른다”는 주장.
- **더 단순한 설명**: seed std 0.0021~0.0083은 fold data의 finite-sample 변동이거나, 후보 변경과 상호작용하는 stochastic path 차이일 수 있다. 이 경우 공통 무작위화로 상쇄되지 않는다. 또는 seed가 AUROC에 불변인 PCA 부호/순서만 바꾸면 Δ 변동 감소가 거의 없다.
- **무엇이 참이면 이 주장이 무너지는가**: seed 변동이 후보 변경과 상호작용하면 CRN 효과가 0이다. 승격 판정에 독립 적합 분산이 필수로 요구되면 단일 CRN 비교만으로는 실행 무효다. 7-branch fold 시간이 5-branch 20.0s보다 길면 2 arm이 4h를 초과해 판별 불가다. R=4 2×2는 31.1 GPU-hour로 4h에서는 실행 무효다.
- **이 반론의 한계**: seed가 baseline/candidate에 공통으로 작용하는 subsampling/context selection 변동이고, CRN이 paired Δ의 표준편차를 줄이면 이 반론은 틀린다. 단, AUROC 기대값 개선이 아니라 판정 정밀도 개선일 뿐이다.

## 반론 6
- **공격 대상**: P2 제안 3의 “LID branch를 동일 토큰 수 정규화와 m=1024/4096 이중 크기 평균으로 안정화하면 유효 랭크 3.52의 두 해석을 가를 probe가 된다”는 주장.
- **더 단순한 설명**: RU-85에서 LID 추정기는 부분추출에 민감했고, m=4096/1024 마진 상관이 0.314로 0.8 임계를 밑돌았다. 두 크기 평균은 추정기 민감성을 없애기보다 두 민감한 추정기의 평균일 수 있다. LID가 토큰 수/염색/배경 노이즈에 지배되면 게이트 ②에서 죽을 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 사전 선언된 |r| < 0.8이면 성능 조회 없이 기각이다. 게이트 ① 통과해도 게이트 ② 실패면 정보 없음이다. k-NN 추가 비용이 fold 20.0s를 크게 넘으면 4h 안에 SCREEN_ONLY 완료가 불가해 실행 무효/판별 불가다. 게이트 ① 실패는 LID family가 막혔다는 증거일 뿐, 다른 분포 기하 branch가 불가능하다는 증거가 아니다.
- **이 반론의 한계**: 동일 토큰 수 정규화와 이중 크기 평균이 LID 마진을 안정화하고, max |r| ≤ 0.6, eff.rank ≥ 4.02, 게이트 ②를 통과하면 “기존 family 중복” 해석에 관측 증거가 된다. 단, 성능 개선은 별도다.

## 반론 7
- **공격 대상**: P1/P2 공통 전제인 “provenance-locked margin 1회만 있으면 재집계 GPU 0” 또는 “2 arm 3.89h”로 4h 내 판별 가능하다는 주장.
- **더 단순한 설명**: 3.89 GPU-hour는 5-branch fold 20.0s 기반 추정이고, 공식 승격 기준은 7-branch다. 7-branch fold 적합 시간은 문서에 없다. 저장 예측 파일에 provenance가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다. D-050은 계측 복구 전 게이트·승격 비교를 verified로 수행할 수 없다고 한다. 따라서 “4h 내 판별 가능”은 미측정 비용과 미복구 계측을 전제한 외삽일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 7-branch fold 시간이 측정되지 않았거나 5-branch보다 길면 2 arm이 4h를 초과해 실행 무효/판별 불가다. provenance-locked margin/예측분산 저장 경로가 없으면 오프라인 재집계는 실행 무효다. 5-branch paired 2 arm은 공식 7-branch 승격 기준의 하위계가 아니다. 게이트 ① 실행 경로가 verified로 복구되지 않으면 gate 기반 probe는 verified 판정이 아니다.
- **이 반론의 한계**: 7-branch fold 시간이 측정되어 2 arm이 4h 이내이고, provenance-locked margin/variance 저장 경로와 게이트 실행 경로가 실제로 복구되면 이 반론은 틀린다.

### A1 (감사)

## 점검 1: P1-1 rank 변환의 fit 범위 — query/test slide 포함 여부
- **확인하려는 것**: fold-global rank가 해당 fold의 query slide만, context slide 포함, 아니면 context+query 전체 slide에서 계산되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: query slide만 rank에 쓰면 test-time normalization이 되고, context+query 전체를 쓰면 fold별 slide 수·난이도·염색 분포가 score 스케일에 섞인다. paired Δ가 fold composition 차이로 오염될 수 있다.
- **확인 방법**: `src/`에서 rank 또는 empirical CDF 변환 코드와 fold manifest를 대조한다. `evaluate_pure.py` 또는 branch margin 생성 함수의 입력 인덱스가 query/context를 어떻게 분리하는지 확인한다. 고정 fold 1개에서 rank 입력 slide ID 목록을 dump하고 query ID와 비교한다.

## 점검 2: P1-1 branch margin 방향 정합
- **확인하려는 것**: 7개 branch margin 모두 높은 값이 positive class 쪽으로 정렬되어 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: rank 변환은 각 branch의 순서만 보존하므로, 방향이 반대인 branch가 섞이면 trimmed mean of ranks가 신호를 상쇄한다. 특정 과제에서 candidate Δ가 음수, sign agreement 하락, branch별 순서 상관이 낮아진다.
- **확인 방법**: branch margin 생성 코드에서 label 사용 부위와 sign convention을 확인한다. 합성 positive/negative slide로 margin 분포를 확인한다. 저장 margin과 query label의 fold별 상관을 구현 검증용으로 계산한다.

## 점검 3: P1-1 rank 변환이 baseline의 monotone transform인지
- **확인하려는 것**: rank-normalized trimmed mean이 baseline trimmed mean의 monotone transform인지, 아니면 aggregation 자체가 실제로 바뀌는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 최종 score가 baseline score의 monotone transform이면 AUROC는 불변이고 Δ=0이다. rank 변환이 branch별 outlier만 줄이는지, 아니면 branch 결합 규칙을 바꾼 것인지 구분되지 않는다.
- **확인 방법**: 같은 fold margin에서 baseline score와 candidate score를 계산한다. per-fold AUROC, score rank correlation, trimmed mean 절사 대상 branch의 변화를 비교한다.

## 점검 4: P1-1 오프라인 재집계 전제 — provenance
- **확인하려는 것**: “provenance-locked 7-branch margin 1회만 있으면 재집계 GPU 0”이 현재 저장 산출물로 성립하는지.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: 기존 margin 파일을 baseline/candidate 재집계에 쓰면 branch list·가중치·코드 해시·fold manifest가 없어 특정 구성에 귀속할 수 없다. D-050의 `unverified-hold` 상태가 유지되고, paired Δ가 실행 무효가 된다.
- **확인 방법**: `logs/`, `outputs/`의 margin/예측 파일에 branch list, weight, code hash, data version, fold manifest hash가 있는지 확인한다. `scripts/eval_v121.sh`, `eval_seal_tasks.sh`, `evaluate_pure.py`의 provenance writer 존재와 동작을 확인한다. `bash scripts/run_tests.sh`의 provenance 관련 테스트를 확인한다.

## 점검 5: P1-1 7-branch fold 비용
- **확인하려는 것**: 7-branch Primary 7 50-fold 1회 실행의 fold 적합 시간 또는 총 GPU-hour가 4 GPU-hour 안에 들어가는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 5-branch `20.0초`, 2 arm `3.89 GPU-hour` 수치를 7-branch에 외삽해 예산을 잘못 판단한다. 7-branch margin 1회 생성이 4h를 초과하면 후보 판별이 불가하다.
- **확인 방법**: GPU 4에서 공식 7-branch YAML로 fold 1개 또는 3개 과제를 타이밍 측정한다. `time` 또는 실행 로그에서 fold 적합 시간을 기록하고 350 fold로 환산해 4h와 비교한다.

## 점검 6: P1-1 단일 사전고정 후보 준수
- **확인하려는 것**: rank-normalized trimmed mean만 검증되고, 다른 집계 변형과 비교하거나 Primary 7에서 최댓값을 고르지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 여러 집계 함수를 Primary 7에서 돌려 최댓값을 고르면 `CA-08` 경계 안의 선택 편향으로 회귀한다. 단일 후보 주장이 무효가 된다.
- **확인 방법**: RU 카드, 사전 설계 문서, git commit, config 파일, 실행 로그에서 검증된 집계 후보가 하나인지 확인한다. 사용된 config hash가 하나인지 확인한다.

## 점검 7: P1-2 seed std 원천 — stochastic step 진단
- **확인하려는 것**: seed std `0.0021~0.0083`을 만드는 무작위 연산, 즉 subsampling, PCA/SVD initialization, random permutation, GPU reduction order 등이 코드 어디에서 발생하는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: deterministic 교체를 적용해도 seed std가 줄지 않거나, 줄어든 것이 PCA sign 불변이 아니라 subsampling regularizer 변화 때문일 수 있다. 분산 저감 주장을 기전으로 일반화할 수 없다.
- **확인 방법**: `src/`에서 `random`, `torch.rand`, `np.random`, `subsample`, `kmeans`, `svd`, `pca`, `permute`, `scatter`, `reduce` 호출부를 감사한다. seed 고정/변경 실행에서 random call 수와 margin 차이를 로그로 확인한다.

## 점검 8: P1-2 deterministic 교체의 mean shift
- **확인하려는 것**: full-token 또는 고정 manifest가 branch margin의 mean을 바꾸지 않고 분산만 줄이는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: deterministic 후보의 seed std는 낮아지지만 paired Δ_macro가 음수일 수 있다. 분산 저감을 성능 개선으로 오인하거나, +0.0030 미만 후보를 살리는 수단으로만 제한해야 함을 무시할 수 있다.
- **확인 방법**: 동일 fold에서 stochastic baseline과 deterministic candidate의 margin mean, fold AUROC, paired Δ, seed std를 함께 보고한다. full-token과 fixed manifest의 margin 분포 차이를 라벨 없이 확인한다.

## 점검 9: P1-3 residualization이 게이트 ①을 무력화하는가
- **확인하려는 것**: mode-count scalar를 기존 7 margin에 회귀시켜 잔차를 최종 margin으로 한 뒤 게이트 ① max |r|을 측정하는지, 아니면 raw scalar로 측정하는지.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: residual margin은 기존 margin과 상관이 0에 가까워 max |r| ≤ 0.6과 eff.rank ≥ 4.02를 기계적으로 통과할 수 있다. 게이트 ①이 “새 정보가 아니라 기존 margin에서 제거된 잔차”를 통과시켜, noise branch가 성능 조회로 넘어갈 수 있다.
- **확인 방법**: `scripts/analysis/branch_screen.py` 입력이 raw mode margin인지 residual margin인지 확인한다. raw와 residual 각각의 max |r|, eff.rank를 계산한다. 게이트 ① 통과 판정 로직이 residualization을 허용하는지 코드와 문서를 대조한다.

## 점검 10: P1-3 mode-count screen 비용
- **확인하려는 것**: mode-count 후보의 SCREEN_ONLY fold 비용이 4 GPU-hour 안에 Primary 7 전 과제를 끝낼 수 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: mode count estimator의 k-NN/모드 추정 비용이 fold `20.0초`를 크게 초과하면 screen만으로도 예산을 초과한다. 게이트 ① 결과를 얻지 못하고 판별 불가가 된다.
- **확인 방법**: mode-count 구현의 fold별 wall-clock을 1개 fold에서 측정한다. 350 fold로 환산하고 GPU/CPU 병목을 확인한다.

## 점검 11: P2-1 PSW 예측분산의 fit 범위와 라벨 사용
- **확인하려는 것**: query slide별 예측분산이 context fit에서만 산출되고 query label을 쓰지 않는지. fold 내 context/query 분할을 지키는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 예측분산이 query label, residual, 또는 test fold 전체 fit에서 나오면 가중치가 정답을 본다. paired Δ가 성능이 아니라 label leakage가 된다.
- **확인 방법**: ridge fit 코드에서 train/context 인덱스와 query 인덱스 분리를 확인한다. prediction variance 계산 함수의 입력이 query label을 받지 않는지 시그니처를 감사한다. label shuffle 테스트에서 variance가 변하면 실패로 판정한다.

## 점검 12: P2-1 PSW 한 arm 재사용 전제
- **확인하려는 것**: 한 arm에서 margin과 예측분산을 저장해 baseline/candidate를 같은 fold에서 오프라인 재집계할 수 있는 provenance가 있는가.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: 저장 파일에 branch list·가중치·코드 해시·fold manifest가 없어 margin/variance를 특정 구성에 귀속할 수 없다. 2 arm 재실행이 필요하고, 7-branch 비용 미측정으로 4h 초과가 가능하다.
- **확인 방법**: PSW 실행 산출물에 margin, variance, branch list, weight, code hash, fold manifest hash 저장 여부를 확인한다. `evaluate_pure.py`의 저장 경로와 provenance writer를 확인한다.

## 점검 13: P2-1 PSW 가중치 사전 고정
- **확인하려는 것**: 가중치 상수, 절사 개수, 분산 정의, clipping이 Primary 7 성능 조회 전에 고정되고 선택 이력이 기록되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 여러 PSW 변형을 Primary 7에서 돌려 최댓값을 고르면 `CA-08` 경계 안의 선택 편향이 된다. 단일 후보 주장이 무효가 된다.
- **확인 방법**: RU 카드, 사전 설계 문서, git commit, config 파일에서 PSW 파라미터가 성능 실행 전에 고정됐는지 확인한다. 실행 로그에 사용된 config hash가 하나인지 확인한다.

## 점검 14: P2-2 CRN seed 무작위 요인과 정렬
- **확인하려는 것**: seed가 실제로 어떤 무작위 연산에 영향을 주는지, candidate가 코드 경로·random call 수를 바꾸면 baseline과 random draw가 같은 순서로 정렬되는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: seed std가 fold data composition이나 ridge solver에서 오면 CRN으로 상쇄할 수 없다. candidate가 random call 수를 바꾸면 “공통 seed”만으로는 CRN이 성립하지 않고, paired Δ 분산 감소가 없다.
- **확인 방법**: seed 변경 실행에서 random call trace, subsample index, PCA rotation, context selection 로그를 baseline/candidate 각각 수집한다. random generator state를 fold 시작/끝에서 dump하고 baseline/candidate의 random call 수와 seed sequence를 비교한다.

## 점검 15: P2-2 CRN 단일 비교로 적합 분산 보고 가능 여부
- **확인하려는 것**: D-048이 요구하는 적합 분산 보고를 단일 CRN paired 비교로 충족할 수 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 단일 CRN은 Δ point estimate만 줄 수 있고, seed 분산 추정은 못 줄 수 있다. 적합 분산 미보고로 승격 비교가 불완전해진다. `R=4` 2×2는 `31.1 GPU-hour`로 4h에서는 실행 무효다.
- **확인 방법**: R>1 seed 실행 계획, 예산, 적합 분산 산출 코드를 확인한다. 단일 CRN 결과에 fold 분산과 적합 분산이 함께 보고되는지 확인한다.

## 점검 16: P2-3 LID 강건성 임계 0.8의 사전성
- **확인하려는 것**: m=1024/4096 |r| ≥ 0.8 임계가 Primary 7 성능 조회와 RU-85 결과 이전에 고정됐는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 임계가 RU-85의 `0.314`를 보고 정해졌다면 기각이 사전 기각이 아니라 사후 합리화가 된다. LID probe의 판별 가치가 하락한다.
- **확인 방법**: RU-85 카드, LID 제안서, git 이력에서 0.8 임계 선언 시점과 0.314 측정 시점을 비교한다. 성능 조회 이전인지 확인한다.

## 점검 17: P2-3 LID k-NN 비용
- **확인하려는 것**: LID SCREEN_ONLY fold 비용이 4 GPU-hour 안에 Primary 7 전 과제를 끝낼 수 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: k-NN/ID_TwoNN 계산이 fold `20.0초`를 초과하면 screen 예산을 초과한다. 게이트 ① 결과를 얻지 못하고 판별 불가가 된다.
- **확인 방법**: LID 구현의 fold별 wall-clock을 1개 fold에서 측정한다. 350 fold로 환산하고 GPU/CPU 병목을 확인한다.

## 점검 18: 공식 baseline verified 상태
- **확인하려는 것**: P1/P2의 paired Δ가 verified 공식 7-branch baseline과 비교되는지.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: D-050에서 공식 `0.6227`은 `unverified-hold`이고 verified로 인용 금지다. D-053에서 여섯 러너가 같은 설정으로 수렴해 과거 arm 비교가 무효다. 계측 복구 전 승격 비교는 실행 무효가 된다.
- **확인 방법**: `docs/PROJECT.md` §3.2 D-050, `docs/current_status.md`의 arm parity 미읽힘 상태를 확인한다. 새 baseline 실행의 provenance와 50-fold 전량 저장을 확인한다.

## 점검 19: fold 고정과 SEAL 접근
- **확인하려는 것**: 제안 실행이 fold 재분할, fold 수 증대, SEAL 10 조회를 하지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: fold 재분할은 §3.1 위반이고, SEAL 10 조회는 모델 선택 금지 위반이다. Primary 7만 써야 할 개선 근거가 hold-out으로 오염된다.
- **확인 방법**: fold manifest hash, config, 실행 로그에서 fold 수와 task ID를 확인한다. SEAL 10 task ID가 학습·선택 경로에 접근했는지 로그를 감사한다.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

- T1-1: (1) 수용하고 수정 — P1·P2 전체를 SEAL 10에서 ABMIL `0.7266` 초과로 직접 전이된다고 주장하지 않는다. `PROJECT.md` §1은 Primary 7 개선이 hold-out 개선을 보장하지 않으며, ABMIL `0.7266`, ICF v120 `0.6972`, ICF와의 거리 `−0.0294`를 기록한다. 따라서 모든 제안의 판정 범위는 Primary 7 paired Δ_macro `+0.0030`과 적합 분산 보고로 한정하고, SEAL 10은 `hold-out 미검증`으로 둔다.

- T1-2: (1) 수용하고 수정 — P1-1 fold-global rank-normalized trimmed mean은 7개 branch margin의 방향 정합, provenance, 단일 사전고정 후보를 전제한다. `D-050`/`D-053`은 저장 예측 파일에 branch list·가중치·코드 해시가 없고, 여섯 러너가 같은 7-branch 설정으로 수렴했다고 기록한다. `PROJECT.md` §5.1은 `CV` ARID1A `0.4308`, `QA` ARID1A `0.4692`, `BD` 3개 과제가 우연 이하라고 기록한다. 따라서 provenance-locked margin, 방향 확인, 단일 후보 조건 없이는 P1-1을 실행 무효로 둔다.

- T1-3: (1) 수용하고 수정 — P1-2 deterministic branch construction의 seed std `0.0021~0.0083` 원인과 7-branch 비용은 문서에 없다. `current_status.md`는 PCA·subsample 사양이 문서에 없고, 7-branch fold 적합 시간은 미측정이며, 5-branch `20.0초`와 2 arm `3.89 GPU-hour`만 기록한다. 따라서 P1-2를 성능 개선 제안이 아니라 코드 감사와 적합 분산 진단으로 낮추고, stochastic step 부재 또는 4 GPU-hour 초과면 중단한다.

- T1-4: (1) 수용하고 수정 — P1-3 residualized mode-count branch에서 residualization은 게이트 ①을 기계적으로 통과시킬 수 있다. `CA-10`은 형상 계열의 병목이 절사가 아니라 정보량이라고 기록하고, RU-85 G는 whitened 32D 각도 산포가 사실상 등방이라고 기록한다. 따라서 raw mode-count scalar로 게이트 ①·안정성·게이트 ②를 먼저 판단하고, residual margin은 raw 통과 후 별도 후보로만 평가한다.

- T1-5: (1) 수용하고 수정 — P2-1 PSW의 closed-form ridge 예측분산이 실제로 산출·저장되는지, branch margin 오류의 label-free 신뢰도인지, branch 오류가 독립인지 문서에 실측이 없다. `CA-08`은 PSW를 비저촉으로 두되 예측분산의 유효성을 남은 검증으로 남겨 두고, `current_status.md`는 7-branch fold 시간과 margin·예측분산 저장 경로가 미기록이라고 한다. 따라서 ridge가 label-free 예측분산을 산출·저장하고, 단일 사전고정 가중만 검증하는 조건으로 수정한다.

- T1-6: (1) 수용하고 수정 — P2-2 CRN paired 비교의 seed 무작위 요인과 승격 절차 요건은 미확인이다. `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 미승인이고, `R`도 미정이며, 2 arm `3.89 GPU-hour`, `R=4` 2×2 `31.1 GPU-hour`를 기록한다. 따라서 CRN을 승격 증거가 아니라 paired Δ 분산 진단으로만 쓰고, seed 경로 감사 없이는 기전 미확인으로 둔다.

- T1-7: (1) 수용하고 수정 — P2-3 robust LID branch probe에서 슬라이드별 동일 토큰 수 정규화와 `m=1024`/`m=4096` 이중 계산이 RU-85의 부분추출 민감성을 해소하는지는 미실측이다. RU-85는 `|r|(m=4096, m=1024)=0.314`를 기록하지만, 제안의 동일 토큰 수 정규화 조건과 동일하지 않다. 따라서 RU-85 값을 전수 기각으로 쓰지 않고, 안정화 후 `|r|`과 fold 비용을 먼저 측정해 `0.8` 미만이면 성능 조회 없이 기각한다.

- M1-1: (1) 수용하고 수정 — P1-1은 집계 함수 코드 경로, 7-branch fold 비용, provenance, branch margin 방향 정합이 선행되어야 한다. 문서에는 5-branch fold `20.0초`, 2 arm `3.89 GPU-hour`만 있고 7-branch 비용은 미실측이다. `D-050`은 provenance 결함을, `D-053`은 arm parity 실험 결과가 아직 읽히지 않았음을 기록한다. 따라서 7-branch fold 타이밍, provenance writer, 방향 확인, arm parity 확인이 선행 조건이다.

- M1-2: (1) 수용하고 수정 — P1-2의 stochastic step 위치, full-token 계산 OOM 여부, deterministic fold 시간은 미실측이다. `current_status.md`는 PCA·subsample 사양이 문서에 없다고 한다. 따라서 코드 감사에서 stochastic step이 없으면 실행 무효로, full-token OOM 또는 deterministic 2 arm이 4 GPU-hour를 초과하면 중단한다.

- M1-3: (1) 수용하고 수정 — P1-3의 mode-count fold 시간, SCREEN_ONLY 비용, residualization ridge 비용, 게이트 ① 실행 가능 여부는 미실측이다. `D-050` 이후 SCREEN_ONLY 게이트 ① 경로가 무효였으며, `D-053`에서 폴백 수정과 arm parity 실험이 진행 중이나 결과는 미읽힘이다. 따라서 게이트 ① verified 상태 확인, raw scalar 안정성, fold 비용 측정이 선행 조건이다.

- M1-4: (1) 수용하고 수정 — P2-1 PSW의 ridge 구현 경로, 예측분산 산출·저장 여부, branch 오류 독립성, 7-branch 2 arm 비용은 미실측이다. `CA-08`은 PSW를 비저촉으로 두되 여러 가중 변형을 탐색해 최댓값을 고르는 절차를 금지한다. 따라서 ridge가 예측분산을 실제로 산출하지 않으면 실행 무효, 7-branch 2 arm이 4 GPU-hour 초과면 중단, 가중 변형 비교는 실행 무효로 둔다.

- M1-5: (1) 수용하고 수정 — P2-2 CRN의 seed 무작위 요인, 7-branch 2 arm 비용, 독립 적합 분산 요건은 미확인이다. `current_status.md`는 5-branch 2 arm `3.89 GPU-hour`, `R=4` 2×2 `31.1 GPU-hour`, 적합 분산 승격 절차 미승인을 기록한다. 따라서 CRN은 단일 비교로 승격을 주장하지 않고, seed 경로 감사와 7-branch 비용 측정 전에는 판정 불가로 둔다.

- M1-6: (2) 반박 — M1의 “RU-85 실측 `0.314`로 사전 기각 조건이 이미 충족”이라는 결론은 근거가 부족하다. RU-85의 `0.314`는 결정론적 스트라이드 부분추출 조건에서 측정한 값이고, P2-3은 슬라이드별 동일 토큰 수 정규화를 추가한 다른 조건이다. 다만 k-NN fold 비용과 게이트 ① verified 상태가 미실측이므로, 안정화 후 `|r|`과 fold 비용을 먼저 측정하는 것으로 수정한다.

- R1-1: (3) 판별 불가 — rank 변환이 기존 유효 랭크 `3.52/7`의 재가중인지, branch outlier 경로를 줄이는지는 문서에 실측이 없다. 유효 랭크 `3.52/7`은 branch 간 중복을 나타내지만, rank-normalized trimmed mean이 최종 AUROC 순서에 무효과라는 증거는 아니다. 따라서 P1-1은 paired Δ_macro, score rank correlation, monotone transform 검증을 통해만 판정한다.

- R1-2: (1) 수용하고 수정 — deterministic replacement가 분산 저감인지 무작위화를 숨기는 것인지 구분하려면 mean shift와 fold 간 변동을 함께 봐야 한다. 따라서 P1-2는 seed std만 보고하지 않고 margin mean, fold AUROC, paired Δ, fold variance를 함께 보고해 분산 저감과 성능 변화를 구분한다.

- R1-3: (1) 수용하고 수정 — residual margin은 기존 7 branch margin의 span에서 제거된 값이므로 max `|r| ≤ 0.6`과 eff.rank 상승이 수학적 결과일 수 있다. 따라서 P1-3은 raw mode-count scalar로 게이트 ①을 판단하고, residual margin은 raw 통과 후 별도 후보로만 평가한다.

- R1-4: (3) 판별 불가 — closed-form ridge의 query slide별 예측분산이 실제 margin 오류의 신뢰도인지, task difficulty·context size·염색 난이도 대리는 문서에 실측이 없다. 따라서 P2-1은 paired Δ와 예측분산-난이도 대리변수 상관 진단을 함께 보고하고, 역분산 가중의 기전은 미확인으로 둔다.

- R1-5: (3) 판별 불가 — seed 변동이 baseline과 candidate에 공통인 무작위 요인인지, 후보 변경과 상호작용하는 stochastic path 차이인지 문서에 없다. 따라서 P2-2는 CRN paired Δ 분산 감소와 독립 seed 비교를 함께 측정해야 하며, 그 전에는 판별 불가로 둔다.

- R1-6: (3) 판별 불가 — LID의 동일 토큰 수 정규화와 `m=1024`/`m=4096` 이중 계산이 표본 민감성을 해소하는지, LID가 토큰 수·염색·배경 노이즈에 지배되는지는 미실측이다. 따라서 P2-3은 안정화 후 `|r|`, log N·염색 대리변수 상관, 게이트 ②를 측정해야 하고, 그 전에는 판별 불가로 둔다.

- R1-7: (1) 수용하고 수정 — `3.89 GPU-hour`는 5-branch fold `20.0초` 기반 추정이고, 공식 승격 비교 기준은 7-branch다. `current_status.md`는 7-branch fold 적합 시간이 문서에 없다고 하고, `D-050`은 저장 산출물에 provenance가 없어 구성 귀속이 불가능하다고 한다. 따라서 7-branch fold 타이밍과 provenance-locked margin 저장 없이는 4 GPU-hour 내 판별 가능하다고 주장하지 않는다.

- A1-1: (1) 수용하고 수정 — rank 변환의 fit 범위는 query slide만 사용할 것을 사전 고정한다. context+query 전체 slide를 쓰면 fold별 slide 수·난이도·염색 분포가 score 스케일에 섞일 수 있고, 문서에는 현재 rank 입력 범위가 없다.

- A1-2: (1) 수용하고 수정 — 7개 branch margin의 방향 정합은 문서에 없다. rank 변환은 각 branch의 순서만 보존하므로, 방향이 반대인 branch가 섞이면 trimmed mean of ranks가 신호를 상쇄할 수 있다. 따라서 코드 sign convention과 합성 positive/negative slide로 확인하고, 확인 전에는 P1-1을 실행하지 않는다.

- A1-3: (1) 수용하고 수정 — rank-normalized trimmed mean이 baseline trimmed mean의 monotone transform인지 같은 fold margin에서 score rank correlation과 per-fold AUROC로 확인한다. 최종 score가 baseline score의 monotone transform이면 AUROC는 불변이므로 Δ=0이고, 후보를 중단한다.

- A1-4: (1) 수용하고 수정 — `D-050`은 저장 예측 파일에 branch list·가중치·코드 해시가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다고 기록한다. 따라서 provenance-locked margin/예측분산이 없으면 오프라인 재집계로 paired Δ를 만들지 않고, 새 실행을 한다.

- A1-5: (1) 수용하고 수정 — 7-branch fold 적합 시간과 총 GPU-hour는 문서에 없다. 5-branch `20.0초`와 2 arm `3.89 GPU-hour`를 7-branch에 외삽하지 않고, GPU 4에서 7-branch fold 1개 또는 소수 과제를 타이밍 측정해 350 fold로 환산 후 4 GPU-hour와 비교한다.

- A1-6: (1) 수용하고 수정 — 단일 사전고정 후보 준수 여부는 RU 카드, config hash, 실행 로그로 확인한다. Primary 7에서 여러 집계 변형을 돌려 최댓값을 고르면 `CA-08` 경계 안의 선택 편향이므로 P1-1을 실행 무효로 둔다.

- A1-7: (1) 수용하고 수정 — seed std `0.0021~0.0083`을 만드는 stochastic step 원인은 문서에 없다. P1-2는 `src/`에서 random·subsample·PCA/SVD·permutation·reduction 호출부 감사와 seed 고정/변경 실행의 random call trace가 선행 조건이다.

- A1-8: (1) 수용하고 수정 — deterministic 교체의 mean shift는 문서에 없다. P1-2는 seed std 감소만 보고하지 않고 margin mean, fold AUROC, paired Δ, fold variance를 함께 보고해 분산 저감과 성능 변화를 구분한다.

- A1-9: (1) 수용하고 수정 — residual margin으로 게이트 ①을 측정하면 max `|r|`과 eff.rank가 기계적으로 좋아질 수 있다. 따라서 P1-3은 raw mode-count scalar로 게이트 ①을 판단하고, residual margin은 raw 통과 후 별도 후보로만 평가한다.

- A1-10: (1) 수용하고 수정 — mode-count SCREEN_ONLY fold 비용은 문서에 없다. 1개 fold wall-clock을 측정해 350 fold로 환산하고, 4 GPU-hour 초과면 게이트 ① 완료가 불가하므로 중단한다.

- A1-11: (1) 수용하고 수정 — PSW 예측분산의 fit 범위와 라벨 사용은 문서에 없다. context fit에서만 query slide별 예측분산을 산출하고 query label을 쓰지 않아야 하며, label shuffle 테스트에서 variance가 변하면 실행 무효로 둔다.

- A1-12: (1) 수용하고 수정 — 한 arm 재사용은 provenance-locked margin과 예측분산 저장이 전제다. `D-050`은 저장 산출물에 branch list·가중치·코드 해시가 없다고 기록한다. 따라서 해당 저장이 없으면 2 arm 재실행이 필요하고, 7-branch 비용 미측정으로 4 GPU-hour 초과가 가능하다.

- A1-13: (1) 수용하고 수정 — PSW 가중치 상수·절사 개수·분산 정의·clipping은 Primary 7 성능 조회 전에 고정해야 한다. RU 카드와 config hash에 기록하고, 여러 변형을 Primary 7에서 비교하면 `CA-08` 선택 편향으로 실행 무효로 둔다.

- A1-14: (1) 수용하고 수정 — CRN은 seed가 영향을 주는 무작위 연산과 random call 수를 baseline/candidate에서 동일하게 정렬해야 성립한다. 문서에는 seed 무작위 요인이 없으므로 random call trace와 generator state dump가 선행 조건이다.

- A1-15: (1) 수용하고 수정 — 단일 CRN은 paired Δ point estimate만 줄 수 있고 적합 분산을 추정하지 못한다. `D-048`은 적합 분산 보고를 필수로 하고, `current_status.md`는 적합 분산 승격 절차가 미승인이라고 한다. 따라서 CRN은 승격 증거가 아니라 분산 진단으로만 쓴다.

- A1-16: (1) 수용하고 수정 — LID `|r| ≥ 0.8` 임계의 사전성은 RU-85 카드, LID 제안서, git 이력으로 확인해야 한다. RU-85 이후에 정해졌다면 성능 조회 이전의 진단용 stop rule로 명시하고, 성능 선택으로 쓰지 않는다.

- A1-17: (1) 수용하고 수정 — LID k-NN fold 비용은 문서에 없다. 1개 fold wall-clock을 측정해 350 fold SCREEN_ONLY로 환산하고, 4 GPU-hour 초과면 게이트 ① 완료가 불가하므로 중단한다.

- A1-18: (1) 수용하고 수정 — 공식 baseline `0.6227`은 `D-050`으로 `unverified-hold`이고, `D-053`은 arm parity 실험 결과를 아직 읽지 않았다고 기록한다. 따라서 provenance-locked 7-branch baseline 재실행과 arm parity 확인 전에는 paired Δ를 verified 승격 근거로 쓰지 않는다.

- A1-19: (1) 수용하고 수정 — fold 재분할과 SEAL 10 접근은 금지다. fold manifest hash, config, 실행 로그로 fold 수·task ID·SEAL 접근 부재를 감사하고, 위반 시 실행 무효로 둔다.

### P2 (제안)

- **T1-0 (P1·P2 공통 전제: Primary 7 개선 → SEAL 10 ABMIL 초과 전이)**: **수용**.  
  근거: `PROJECT.md` §1은 “Primary 7의 개선이 hold-out 개선을 보장하지 않는다”, ABMIL `0.7266`, ICF v120 `0.6972`, `SEAL 10`은 `DEFERRED`라고 기록한다. 따라서 제안은 SEAL 10 전이를 입증하는 것이 아니라, **Primary 7 paired Δ_macro `+0.0030` 검증과 적합 분산 보고를 위한 후보 탐색**으로 한정한다. SEAL 10 결과로 방향을 고르지 않는다.

- **T1-1 (P1-1 rank-normalized trimmed mean)**: **수용**.  
  근거: `D-050`/`D-053`은 저장 예측 파일에 branch list·가중치·코드 해시가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다고 기록한다. `current_status`는 7-branch fold 적합 시간이 아니라 5-branch `20.0초`, 2 arm `3.89 GPU-hour`만 기록한다. `CA-08`은 기존 margin 결합 반복을 경계 안으로 둔다.  
  수정: rank 후보는 (a) 새 7-branch margin 실행에서 provenance 기록, (b) 7개 branch margin의 방향 정합 확인, (c) 단일 후보 사전고정, (d) CA-08 “제한된 결합 연구”로만 수행한다. rank 변환이 baseline score의 monotone transform인지 같은 fold margin에서 확인한다.

- **T1-2 (P1-2 deterministic branch construction)**: **수용**.  
  근거: `current_status`는 seed std `0.0021~0.0083`의 원인이 되는 stochastic step이 문서에 없고, PCA·subsample 사양도 문서에 없다고 기록한다. 7-branch fold 시간도 미측정이다.  
  수정: 코드 감사에서 subsampling, PCA/SVD init, permutation, reduction order 중 seed 민감 연산이 확인되기 전까지 실행하지 않는다. deterministic 교체 후 seed std, paired Δ_macro, branch margin mean을 함께 보고한다. 성공해도 “성능 개선”이 아니라 “판정 정밀도 개선”으로만 기록한다.

- **T1-3 (P1-3 residualized mode-count branch)**: **수용**.  
  근거: `CA-10`은 형상 계열 병목이 정보량이라고, `RU-85 G`는 whitened 32D 각도 산포가 사실상 등방이라고, `CA-09`는 기존 공간 중복으로 gate ① 탈락한 선례가 있다. residual margin은 기존 7 margin에서 회귀 제거하므로 max |r| ≤ 0.6을 기계적으로 만족할 수 있다.  
  수정: **raw mode margin**을 먼저 gate ①에 넣는다. raw margin이 max |r| ≤ 0.6, eff.rank ≥ 4.02를 통과해야만 “서로 다른 측정” 해석에 관측 증거가 된다. residual margin은 gate ① 통과 증거로 쓰지 않고, raw 통과 후에만 ensemble readout 후보로 검토한다.

- **T1-4 (P2-1 PSW)**: **수용**.  
  근거: `CA-08`/`D-022`는 PSW를 비저촉으로 두지만 “예측분산의 유효성”을 남은 검증으로 둔다. `D-050`은 margin·variance 저장 provenance가 없어 재집계를 특정 구성에 귀속시킬 수 없다고 기록한다. 7-branch fold 시간은 문서에 없다.  
  수정: (a) ridge fit이 query slide별 예측분산을 label-free로 산출하는지 label shuffle 검증, (b) margin+variance+provenance 저장, (c) 가중 상수·절사 개수·분산 정의·clipping을 Primary 7 조회 전에 고정, (d) 단일 변형만 검증한다.

- **T1-5 (P2-2 CRN paired 비교)**: **판별 불가**.  
  근거: `current_status`는 seed std의 무작위 요인이 문서에 없고, 적합 분산을 포함한 승격 판정 절차가 미승인이며, R=4 2×2는 `31.1 GPU-hour`라고 기록한다. CRN이 paired Δ 분산을 줄일지, seed 변동이 baseline/candidate 공통인지 문서에 근거가 없다.  
  결론: seed trace와 공통 무작위 요인 확인 전까지 CRN은 **판정 정밀도 개선 후보**로만 두고, 단독 승격 증거로 쓰지 않는다. 독립 적합 분산이 승격 필수로 승인되면 단일 CRN은 **실행 무효**다.

- **T1-6 (P2-3 robust LID branch probe)**: **수용**.  
  근거: `RU-85 H`는 `|r|(m=4096, m=1024)=0.314`로 강건성 임계 `0.8`을 밑돈다고 기록한다. 제안의 사전 기각 조건은 `|r| < 0.8`이면 성능 조회 없이 기각이다.  
  결론: 동일 토큰 수 정규화가 이 값을 `0.8` 이상으로 바꾼다는 실측 증거가 없으므로, LID probe는 현재 **실행 전 기각/판별 불가**다.

- **M1-1 (P1-1 코드·비용·막는 것)**: **수용**.  
  근거: 7-branch fold 시간 미실측, `D-050` provenance 결함, branch 방향 정합 미확인, arm parity 결과 미읽힘은 모두 문서에 있다.  
  수정: 7-branch margin 1회 생성 전 timing 측정, provenance writer 확인, branch direction 확인, arm parity 확인을 선행 조건으로 둔다.

- **M1-2 (P1-2 코드·비용·막는 것)**: **수용**.  
  근거: stochastic step 위치, full-token OOM, deterministic fold 시간, mean shift 모두 문서에 없다.  
  수정: 코드 감사와 fold timing/OOM 확인 전까지 실행하지 않는다.

- **M1-3 (P1-3 mode-count gate·비용)**: **수용**.  
  근거: `RU-85 G`의 등방성은 1과제·32D 측정이고 Primary 7 전량은 미실측이다. `D-050` 이후 gate ① 실행 경로가 verified로 복구됐는지 `current_status`의 arm parity 미읽힘으로 미확인이다.  
  수정: gate ① 실행 경로 복구, raw mode margin 재현성 확인, screen cost 측정 전까지 실행하지 않는다.

- **M1-4 (P2-1 PSW 코드·비용)**: **수용**.  
  근거: ridge fit이 예측분산을 산출·저장하는지, 7-branch 1 arm 비용, CA-08 단일 변형 강제 메커니즘은 문서에 없다.  
  수정: variance 산출/저장 확인, 7-branch timing, 단일 config hash 기록을 선행 조건으로 둔다.

- **M1-5 (P2-2 CRN 코드·비용)**: **판별 불가**.  
  근거: seed가 어떤 무작위 연산에 영향을 주는지, 승격에 독립 적합 분산이 필수인지, 7-branch 2 arm 비용이 4h 이내인지 문서에 근거가 없다.  
  결론: seed trace와 절차 승인 전까지 CRN은 **판별 불가**다.

- **M1-6 (P2-3 LID 사전 기각)**: **수용**.  
  근거: `RU-85` 실측 `0.314 < 0.8`이다. 동일 토큰 수 정규화가 이 값을 바꾼다는 실측 증거가 없다.  
  결론: LID probe는 **실행 전 기각** 또는 동일 토큰 정규화 효과에 대한 별도 실측 증거 확보 전까지 **판별 불가**다.

- **R1-1 (P1-1: rank 변환이 새 정보가 아닐 수 있다)**: **판별 불가**.  
  근거: 유효 랭크 `3.52/7`은 branch margin 간 중복을 보여 주지만, fold-global rank 변환이 baseline trimmed mean의 monotone transform인지, branch scale mixing을 실제로 바꾸는지 문서에 실측이 없다.  
  결론: 같은 fold margin에서 baseline score와 candidate score의 rank correlation, 절사 대상 branch, per-fold AUROC를 비교해야 한다.

- **R1-2 (P1-2: deterministic 교체가 무작위화 숨김일 수 있다)**: **판별 불가**.  
  근거: seed std `0.0021~0.0083`이 stochastic step에서 오는지, fold composition/ridge solver에서 오는지 문서에 근거가 없다.  
  결론: stochastic step이 seed std의 상당 부분을 만든다는 코드 감사 결과 전까지 “분산 저감”과 “무작위화 숨김”을 구분할 수 없다.

- **R1-3 (P1-3: residualization이 gate ①을 무력화한다)**: **수용**.  
  근거: residual margin은 기존 7 margin의 span에서 제거된 값이므로 max |r| ≤ 0.6을 수학적으로 만족할 수 있다.  
  수정: gate ①은 raw mode margin으로만 수행하고, residual margin의 gate ① 통과는 “새 정보” 증거로 인정하지 않는다.

- **R1-4 (P2-1: 예측분산이 난이도 대리에 불과할 수 있다)**: **판별 불가**.  
  근거: ridge 예측분산이 branch margin 오류를 label-free로 반영하는지, task difficulty·context size·염색 난이도와 혼동되는지 문서에 실측이 없다.  
  결론: 예측분산과 fold AUROC, context 규모, 염색 대리변수의 상관, paired Δ를 함께 보고해야 한다.

- **R1-5 (P2-2: CRN 효과가 0일 수 있다)**: **판별 불가**.  
  근거: seed 변동이 baseline/candidate 공통인지, 후보 변경과 상호작용하는 무작위 경로 차이인지 문서에 근거가 없다.  
  결론: random call trace와 seed별 paired Δ 변동 비교 전까지 CRN 효과는 **판별 불가**다.

- **R1-6 (P2-3: LID 0.314로 사전 기각)**: **수용**.  
  근거: `RU-85`는 `|r|=0.314`를 실측했다. 제안의 기각 임계는 `0.8`이다.  
  결론: LID probe는 현재 증거 범위에서 **실행 전 기각**이다.

- **R1-7 (P1/P2 공통: 4h 내 판별 가능 주장)**: **수용**.  
  근거: `current_status`의 `3.89 GPU-hour`는 5-branch fold `20.0초` 기준이고, 7-branch fold 시간은 문서에 없다. `D-050`은 provenance 결함으로 과거 산출물을 구성에 귀속시킬 수 없다고 한다.  
  수정: “4h 내 판별 가능”은 7-branch fold timing, provenance writer, baseline verified 복구 전까지 **실행 무효/판별 불가**로 둔다.

- **A1-1 (P1-1 rank fit 범위)**: **판별 불가**.  
  근거: fold-global rank가 query slide만, context 포함, context+query 전체에서 계산되는지 문서에 없다.  
  결론: rank 입력 slide ID 목록을 dump하고 query ID와 대조하기 전까지 판별 불가다.

- **A1-2 (P1-1 branch margin 방향 정합)**: **판별 불가**.  
  근거: 7개 branch margin 모두 높은 값이 positive 쪽인지 문서에 없다.  
  결론: branch margin 생성 코드와 fold별 margin-label 상관 확인 전까지 rank 변환 실행은 **실행 무효**다.

- **A1-3 (P1-1 baseline monotone transform 여부)**: **판별 불가**.  
  근거: rank-normalized trimmed mean이 baseline score의 monotone transform인지 문서에 실측이 없다.  
  결론: 같은 fold margin에서 baseline/candidate score를 비교해야 한다.

- **A1-4 (P1-1 오프라인 재집계 provenance)**: **수용**.  
  근거: `D-050`/`D-053`은 저장 예측 파일에 branch list·가중치·코드 해시·fold manifest가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다고 기록한다.  
  결론: 기존 margin 파일을 baseline/candidate 재집계에 쓰지 않는다. 새 실행에서 provenance 기록이 선행된다.

- **A1-5 (P1-1 7-branch fold 비용)**: **판별 불가**.  
  근거: 7-branch fold 적합 시간 또는 총 GPU-hour는 문서에 없다. 5-branch `20.0초`/`3.89 GPU-hour`를 7-branch에 외삽할 근거가 없다.  
  결론: 7-branch 1 fold 또는 3 과제 timing 전까지 4h 내 실행 가능 여부를 판정할 수 없다.

- **A1-6 (P1-1 단일 사전고정 후보 준수)**: **수용**.  
  근거: `CA-08`은 Primary 7에서 여러 집계 변형을 돌려 최댓값을 고르는 선택 편향을 경계 안으로 둔다.  
  수정: rank-normalized trimmed mean만 검증하고, RU 카드·config hash·선택 이력을 기록한다.

- **A1-7 (P1-2 seed std 원천)**: **판별 불가**.  
  근거: seed std `0.0021~0.0083`을 만드는 stochastic step이 코드 어디에서 발생하는지 문서에 없다.  
  결론: random call trace 전까지 deterministic 교체의 대상과 효과를 판정할 수 없다.

- **A1-8 (P1-2 deterministic mean shift)**: **판별 불가**.  
  근거: full-token 또는 fixed manifest가 branch margin mean을 바꾸지 않고 분산만 줄이는지 문서에 실측이 없다.  
  결론: deterministic 후보의 margin mean, fold AUROC, paired Δ, seed std를 함께 보고해야 한다.

- **A1-9 (P1-3 residualization이 gate ①을 무력화하는가)**: **수용**.  
  근거: residual margin은 기존 margin과 상관이 0에 가까워 gate ①을 기계적으로 통과할 수 있다.  
  수정: gate ①은 raw mode margin으로만 수행한다. residual margin의 gate ① 통과는 “직교성” 증거로 쓰지 않는다.

- **A1-10 (P1-3 mode-count screen 비용)**: **판별 불가**.  
  근거: mode-count 후보의 SCREEN_ONLY fold 비용은 문서에 없다.  
  결론: 1 fold wall-clock 측정 전까지 4h 내 Primary 7 screen 가능 여부를 판정할 수 없다.

- **A1-11 (P2-1 PSW 예측분산 fit 범위·라벨 사용)**: **판별 불가**.  
  근거: query slide별 예측분산이 context fit에서만 산출되고 query label을 쓰지 않는지 문서에 없다.  
  결론: label shuffle 테스트와 fit/query 인덱스 감사 전까지 PSW 실행은 **실행 무효**다.

- **A1-12 (P2-1 PSW 한 arm 재사용 전제)**: **수용**.  
  근거: `D-050`은 margin/variance를 특정 구성에 귀속시킬 provenance가 없다고 기록한다.  
  결론: 한 arm 재사용은 margin+variance+provenance 저장이 확인되기 전까지 인정하지 않는다.

- **A1-13 (P2-1 PSW 가중치 사전 고정)**: **수용**.  
  근거: `CA-08`은 Primary 7에서 여러 가중 변형을 골라 최댓값을 쓰는 절차를 금지한다.  
  수정: 가중 상수, 절사 개수, 분산 정의, clipping을 Primary 7 조회 전에 고정하고 config hash를 기록한다.

- **A1-14 (P2-2 CRN seed 무작위 요인과 정렬)**: **판별 불가**.  
  근거: seed가 어떤 무작위 연산에 영향을 주는지, candidate가 random call 수를 바꾸면 baseline과 draw가 정렬되는지 문서에 없다.  
  결론: random generator state dump와 call trace 비교 전까지 CRN 설계는 **판별 불가**다.

- **A1-15 (P2-2 CRN 단일 비교로 적합 분산 보고 가능 여부)**: **판별 불가**.  
  근거: 단일 CRN paired 비교로 D-048이 요구하는 적합 분산을 충족할 수 있는지 문서에 근거가 없다. R=4 2×2는 `31.1 GPU-hour`다.  
  결론: 적합 분산 산출 계획과 예산 승인 전까지 CRN은 **판별 불가**다.

- **A1-16 (P2-3 LID 0.8 임계의 사전성)**: **판별 불가**.  
  근거: `0.8` 임계가 RU-85의 `0.314` 측정 이전에 고정됐는지 문서에 확인 가능한 기록이 없다. A1이 “확인됨”이라 했지만, 그 확인을 뒷받침하는 문서 인용이 없다.  
  결론: RU-85 카드, LID 제안서, git 이력에서 선언 시점과 측정 시점을 대조하기 전까지 임계의 사전성은 **판별 불가**다.

- **A1-17 (P2-3 LID k-NN 비용)**: **판별 불가**.  
  근거: LID SCREEN_ONLY fold 비용은 문서에 없다.  
  결론: 1 fold wall-clock 측정 전까지 4h 내 Primary 7 screen 가능 여부를 판정할 수 없다.

- **A1-18 (공식 baseline verified 상태)**: **수용**.  
  근거: `PROJECT.md` §3.2는 공식 `0.6227`을 `unverified-hold`로, `D-050`은 verified로 인용 금지라고 기록한다. `D-053`은 여섯 러너가 같은 7-branch 설정으로 수렴했다고 한다.  
  결론: 새 paired Δ는 provenance-locked 7-branch baseline과 비교되어야 하며, 계측 복구 전 승격 비교는 **실행 무효**다.

- **A1-19 (fold 고정과 SEAL 접근 금지)**: **수용**.  
  근거: `PROJECT.md` §3.1은 fold 재분할·fold 수 증대 금지, SEAL 10 선택·튜닝 금지를 사용자 결정으로 고정한다.  
  수정: 모든 제안 실행은 고정 fold manifest hash, Primary 7 task ID만, SEAL 10 task ID 학습/선택 경로 접근 금지 로그를 포함해야 한다. 위반 시 실행 무효다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)
- 공식 7-branch 기준선 `0.6227`과 직전 승격 근거 `+0.56%p`는 현재 `unverified-hold`다. `PROJECT.md §3.2`의 `D-050`은 이 값을 `verified`로 인용하지 말라고 하고, `D-053`은 여섯 러너가 같은 7-branch 설정으로 수렴해 과거 arm 차이를 구성 효과가 아닌 적합 변동으로 만들었다고 기록한다.
- 과거 arm 간 비교를 “구성 차이”로 쓰는 것은 뒷받침되지 않는다. `current_status.md`의 `D-053`은 `eval_v121.sh` 등이 학습 설정을 넘겨 `eval_seal_tasks.sh` 폴백에 걸렸고, `ICF_*` 환경변수는 `src/`에서 읽히지 않아 arm 이름만 다른 실행이 같은 7-branch로 수렴했다고 기록한다.
- 4 GPU-hour 예산으로는 seed 반복 확증이 불가능하다. `current_status.md`는 5-branch fold 적합 `20.0초`(과제 3개 평균) 기준으로 한 arm Primary 7 `1.94 GPU-hour`, 2-arm paired `3.89 GPU-hour`, `R=4` 2×2 `31.1 GPU-hour`를 기록한다. `3.89h`는 예산의 `97%`이고, `R=4`는 4h를 크게 초과한다.
- 유효 랭크 `3.52/7`은 현재 7-branch margin 공간의 중복을 보여 준다. `PROJECT.md §3.2`는 `3.52/7`(50.3%)을 기록하고, `CA-09`는 `BM·QA·DS`의 유효 랭크가 `1.29/3`이라고 기록한다. 이는 “branch 추가가 구조적으로 막혔다”는 결론을 직접 주지는 않지만, 현재 margin이 중복된다는 관측을 뒷받침한다.
- 집계 축은 “아무 제한 없이 열려 있다”기보다 경계가 있다. `CA-08`은 기존 branch margin 위에서 탐색한 뒤 같은 Primary 7에서 최댓값을 확증으로 쓰는 선택 편향을 닫았고, `D-022`는 예측분산·Context 신호 기하 등 새 정보를 쓰는 `PSW`·`TGW`를 비저촉으로 둔다.
- `D-048` 이후 단일 적합만으로는 불확실성을 과소평가한다. `PROJECT.md §1`과 `CA-14`는 학습 파라미터 0개 제약을 폐기하되, 적합 분산 보고를 필수로 하고, `RU-97 §8`의 동일 구성 seed만 바꿔 macro 폭 `0.0200`을 인용한다.
- Primary 7 개선이 SEAL 10 개선을 보장하지 않는다. `PROJECT.md §1`은 ABMIL `0.7266`, ICF v120 `0.6972`, ICF와의 거리 `−0.0294`를 기록하고, SEAL 10은 `DEFERRED`라고 한다. 따라서 이 회차의 모든 판정은 `hold-out 미검증`으로 한정된다.
- query 라벨 누수에 대한 제한적 지지만 있다. `current_status.md`는 `tests/test_query_independence.py`에서 query 스케일·제거·순서 변경 시 margin 변화가 `1e-5` 미만이라고 기록한다. 이는 테스트된 교란에서 query 라벨이 margin 경로에 직접 들어가지 않음을 지지하지만, 모든 누수 경로를 배제하는 증거는 아니다.
- 값 기반 오염 검사는 구성 변화와 수치 드리프트를 구분하지 못한다. `PROJECT.md §3.4`는 5-branch 앵커 `0.4421`/`0.5553`이 직접 실행 `0.4426`/`0.5546`과 어긋나고, 소수 4자리 정밀도를 시스템이 유지하지 못한다고 기록한다. 따라서 provenance 기반 재설계가 선행되어야 한다.
- 승격 추정 대상은 절대 macro가 아니라 대응 Δ다. `PROJECT.md §4.1`은 fold 짝짓기에서 `s_d = 0.0033`, 독립 350 가정 SE `0.018%p`, 과제 군집 SE `0.022%p`, RU-80 MDE `0.072%p`를 기록하고, 절대 macro의 설계 효과 `5.12배`를 인용한다. 이는 대응 비교의 정밀성을 지지하지만, RU-80 특정 비교의 MDE를 모든 후보의 공통 한계로 쓰지는 못한다.

## 반박 (증거가 반박함)
- 유효 랭크 `3.52/7`만으로 “branch를 더 넣는 방향이 막혔다” 또는 “서로 다른 것을 재는 branch를 아직 못 만들었다”를 확정하는 것은 반박된다. branch별 기여도는 문서에 없고, raw 후보 margin의 게이트 ① 실측도 없으며, PCA·subsample·whitening 사양도 문서에 없다.
- 과거 arm 간 차이를 구성 효과로 인용하는 것은 반박된다. `D-053`은 arm이 다른 실행들이 전부 같은 7-branch 설정으로 수렴했다고 기록하므로, 과거 arm 비교는 적합 변동으로만 읽어야 한다.
- 공식 기준선 `0.6227`과 `+0.56%p`를 `verified`로 쓰는 것은 반박된다. `D-050`은 계측 결함으로 이 기준선을 재현할 verified 경로가 없다고 판정했고, `D-053`은 그 사유를 러너 폴백과 provenance 부재로 강화했다.
- “fold 적합 시간이 측정된 적이 없다”는 일반화는 반박된다. `current_status.md`에는 5-branch 3과제 평균 fold 적합 `20.0초`가 실측으로 존재한다. 다만 공식 7-branch 전체의 fold 적합 시간과 7-branch 2-arm 총 소요는 문서에 없다.
- residualized margin의 게이트 ① 통과는 “새로운 직교 정보” 증거로 쓰면 반박된다. 후보 margin을 기존 7-branch margin에 closed-form ridge로 회귀해 잔차를 만들면, 기존 margin과의 상관은 구조적으로 낮아질 수 있다. 따라서 게이트 ①의 직교성 판정은 raw candidate margin으로 수행되어야 한다.
- 적합 분산 저감을 성능 개선으로等同시키는 것은 반박된다. `PROJECT.md §4.1`의 추정 대상은 대응 Δ의 기댓값이다. seed std를 줄이는 것은 Δ의 불확실성을 줄이는 투자이지, Δ 평균을 올리는 증거가 아니다.
- LID 추정기가 RU-85의 결정론적 스트라이드 부분추출 조건에서 부분추출에 강건하다는 주장은 반박된다. `RU-85 H`는 `|r|(m=4096, m=1024)=0.314`로 강건성 임계 `0.8`을 밑돈다고 기록한다. 다만 이는 동일 토큰 수 정규화 조건까지 반박하지는 않는다.
- 4 GPU-hour 안에서 seed averaging이나 적합 분산 확증을 수행할 수 있다는 주장은 반박된다. `R=4` 2×2는 `31.1 GPU-hour`이고, 2-arm paired만도 5-branch 기준으로 `3.89 GPU-hour`다.
- “집계 축이 완전히 닫혀 있다”는 주장과 “집계 축이 아무 제한 없이 미탐색이다”는 주장 모두 반박된다. `CA-08`은 기존 margin 결합·부분집합 탐색의 선택 편향을 닫았고, `D-022`는 `PSW`·`TGW` 같은 새 정보 기반 제한 연구를 비저촉으로 남겼다.
- 절대 macro를 “과제 모집단의 기대 성능”으로 쓰는 것은 반박된다. `PROJECT.md §4.1`은 fold가 과제 내에서 독립이 아니라며, 절대 macro의 SE에서 설계 효과 `5.12배`를 실측했다.

## 판별 불가 (증거가 가르지 못함)
- 유효 랭크 `3.52/7`의 두 해석은 아직 가려지지 않았다. “branch 추가 방향이 막혔다”와 “서로 다른 것을 재는 branch를 못 만들었다”를 가리려면, raw candidate margin으로 게이트 ① `max |r| ≤ 0.6`과 8-branch `eff.rank ≥ 4.02`를 측정해야 한다. 현재 branch별 기여도, PCA·subsample 사양, raw 후보 margin 실측은 문서에 없다.
- fold-global rank-normalized trimmed mean이 AUROC를 개선하는지는 판별 불가하다. 7개 branch margin의 방향 정합, rank 입력 범위(query만인지 context 포함인지), baseline score의 monotone transform 여부, provenance-locked margin, paired Δ_macro 모두 문서에 실측이 없다.
- deterministic branch construction이 seed std `0.0021~0.0083`을 줄이는지는 판별 불가하다. seed std를 만드는 stochastic step이 subsampling, PCA/SVD, permutation, GPU reduction 중 어디에서 오는지 문서에 없다. full-token 또는 고정 manifest의 fold 시간, OOM 여부, margin mean shift도 미실측이다.
- PSW가 margin 신뢰도를 label-free로 반영하는지는 판별 불가하다. closed-form ridge가 query slide별 예측분산을 실제로 산출·저장하는지, 그 분산이 branch margin 오류와 관련 있는지, task difficulty·context size·염색 난이도의 대리에 불과한지 문서에 실측이 없다.
- CRN paired 비교가 paired Δ 분산을 줄이는지는 판별 불가하다. seed가 어떤 무작위 연산에 영향을 주는지, baseline과 candidate의 random call 수가 정렬되는지, seed 변동이 후보 변경과 상호작용하는지 문서에 근거가 없다.
- LID probe의 동일 토큰 수 정규화 버전이 강건한지는 판별 불가하다. `RU-85`의 `0.314`는 결정론적 스트라이드 부분추출 조건이고, 제안된 동일 토큰 수 정규화 조건에서의 `|r|`은 미실측이다. k-NN fold 비용도 문서에 없다.
- 공식 7-branch 2-arm paired가 4 GPU-hour 안에 들어가는지는 판별 불가하다. 5-branch `20.0초`와 `3.89 GPU-hour`는 7-branch에 외삽할 수 없다. 7-branch fold 적합 1회 시간과 700-fold 총 소요는 문서에 없다.
- 적합 분산 저감이 성능 개선보다 나은 투자인지는 판별 불가하다. 후보의 진짜 Δ가 `+0.0030` 근처인지, `+0.0030` 미만인지 문서에 없다. 진짜 Δ가 임계 미만이면 분산 저감만으로는 승격이 되지 않는다.
- “건드리지 않은 축”의 이유가 단순 망각인지 타당한 차단인지는 축별로 갈린다. 계측·provenance·적합 분산 절차는 `PROJECT.md §2`, `D-050`, `D-053`, `current_status.md`의 현재 목표와 미해결 문제로 타당하게 차단되어 있다. 반면 PSW·TGW, rank/median 등 사전고정 집계, deterministic 진단, raw shape branch probe는 완전히 닫힌 것이 아니라 게이트·비용·provenance 조건이 해소되지 않은 상태다.
- 5-branch 앵커 재현 실패의 원인은 판별 불가하다. `current_status.md`는 실행이 결정론적이며 부호가 반대인 차이가 앵커 기록 이후 실제 변경을 뜻한다고 하지만, 코드 변경인지 데이터 변경인지 원인은 미확인이다.
- 게이트 ① 실행 경로가 현재 verified로 복구되었는지는 판별 불가하다. `D-050` 이후 SCREEN_ONLY 경로가 무효였고, `D-053`에서 폴백이 큰 소리로 실패하도록 수정됐으나 arm parity 실험 결과는 아직 읽히지 않았다.
- mode-count 후보가 정보량을 가지는지는 판별 불가하다. `RU-85 G`는 1과제·32D whitened cloud에서 각도 산포가 사실상 등방(`H_q > 0.99`, 슬라이드 간 std `0.0017`)이라고 하지만, Primary 7 전량·다른 차원·mode-count estimator의 안정성과 게이트 ②는 미실측이다.

## 실행 무효 (실행·계측 결함으로 판단 불가)
- provenance writer 복구 전의 verified 게이트·승격 비교는 실행 무효다. `D-050`은 저장 예측 파일에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다고 한다.
- 저장된 과거 예측 파일을 offline 재집계해 paired Δ를 만드는 것은 실행 무효다. provenance가 없으므로 baseline과 candidate가 같은 구성·fold·데이터인지 확인할 수 없다.
- 과거 arm 간 비교를 승격·게이트 근거로 쓰는 것은 실행 무효다. `D-053`은 arm이 다른 실행들이 같은 7-branch로 수렴했다고 기록한다.
- 값 기반 오염 검사 통과를 주장하는 실행은 실행 무효다. `PROJECT.md §3.4`는 구성을 바꾸지 않아도 4자리 정밀도를 유지하지 못해 통과/실패가 오염에 대해 정보를 주지 못한다고 한다.
- Primary 7에서 여러 집계 변형을 돌려 최댓값을 고른 뒤 같은 자료에서 승격을 판정하는 것은 실행 무효다. `CA-08`은 이 절차를 선택 편향으로 닫았다.
- branch margin 방향 정합과 rank 입력 범위를 확인하기 전 rank-normalized trimmed mean을 실행하는 것은 실행 무효다. 방향이 반대인 branch가 섞이면 rank 평균이 신호를 상쇄할 수 있고, context 포함 rank는 fold composition이 score 스케일에 섞일 수 있다.
- ridge가 label-free 예측분산을 산출·저장하는지 확인하기 전 PSW를 실행하는 것은 실행 무효다. query label이나 test fold 전체 fit이 variance에 들어가면 성능이 아니라 누수가 된다.
- 적합 분산 승격 절차가 승인되기 전 단일 CRN 또는 단일 deterministic 비교를 승격 증거로 쓰는 것은 실행 무효다. `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 미승인이라고 기록한다.
- 4 GPU-hour 안에서 `R=4` 2×2 seed averaging을 실행하는 것은 실행 무효다. `current_status.md`는 `31.1 GPU-hour`로 4h의 7.8배라고 기록한다.
- 5-branch paired 2-arm을 공식 7-branch 승격 비교로 대체하는 것은 실행 무효다. `PROJECT.md §3.2`는 모든 승격 비교를 7-branch 구성에서 수행한다고 고정한다. 5-branch는 과거 재현·진단용 기준일 뿐이다.
- 5-branch fold `20.0초`와 `3.89 GPU-hour`를 7-branch 비용으로 승인하는 것은 실행 무효다. `current_status.md`는 이 값이 5-branch 3과제 평균이라고 한정하고, 7-branch fold 시간은 문서에 없다고 한다.
- k-NN 또는 mode-count fold 비용 미측정 상태에서 해당 SCREEN_ONLY를 4h 내 완료 예정으로 실행하는 것은 실행 무효다. 1-fold wall-clock과 350-fold 환산이 없으면 예산 내 완료를 승인할 수 없다.
- SEAL 10 결과를 보고 branch·집계·강도·후보를 고르는 모든 실행은 실행 무효다. `PROJECT.md §3.1`은 SEAL 10을 최종 hold-out으로 두고 선택·튜닝에 어떤 형태로도 사용 금지로 고정한다.

## 좌석 공백 (기권·실패로 빠진 관점)
- 없음. 제공된 논의에는 기권한 좌석이 없으며, 실패로 빠진 좌석도 명시되지 않았다.

## 다음 행동 (권한 등급과 함께)
- [실행 전 필수] 공식 7-branch fold 적합 1회 시간을 측정한다. GPU 4에서 7-branch YAML로 1-fold 또는 소수 fold를 타이밍하고, 350-fold 1-arm과 700-fold 2-arm을 4 GPU-hour와 비교한다. 700-fold 2-arm이 4h를 초과하면 paired 승격 프로브를 시작하지 않는다.
- [실행 전 필수] provenance writer를 복구하고 검증한다. 산출물에 branch list, 가중치, 코드 해시, 데이터 버전, fold manifest 해시가 기록되는지 확인하고, `bash scripts/run_tests.sh`의 관련 테스트를 통과시킨다. 이 전에는 verified 게이트·승격 비교를 수행하지 않는다.
- [실행 전 필수] arm parity 실험 결과를 읽고 config hash를 arm별로 대조한다. `logs/official50/parity_*`에서 5-branch와 7-branch가 실제로 다른 구성으로 실행됐는지 확인한다. 같은 구성으로 수렴했다면 과거 arm 비교를 무효로 표시한다.
- [연구자 자율] 코드 감사를 수행한다. seed 민감 stochastic step(subsampling, PCA/SVD, permutation, reduction order), branch margin 방향 정합, rank 입력 범위, PSW 예측분산의 label-free 산출 여부, random call trace를 확인한다. 이 단계에서는 성능을 조회하지 않는다.
- [사용자 승인] 적합 분산을 포함한 승격 판정 절차를 승인받는다. `R`, fold 분산과 적합 분산의 보고 형식, 단일 CRN 또는 deterministic 비교가 승격에 충분한지, 아니면 R>1이 필요한지 사전 고정해야 한다. 현재 `current_status.md`는 이 절차가 미승인이라고 기록한다.
- [연구자 자율] 집계 후보를 진행한다면 단일 후보만 사전 선언한다. rank-normalized trimmed mean 또는 PSW 중 하나를 고르고, 가중 상수·절사 개수·분산 정의·clipping·rank 입력 범위를 Primary 7 조회 전에 고정한다. config hash와 선택 이력을 기록한다.
- [연구자 자율] 신규 branch probe를 진행한다면 1-fold wall-clock을 먼저 측정한다. mode-count 또는 LID의 350-fold SCREEN_ONLY 환산이 4 GPU-hour를 초과하면 게이트 ① 완료를 중단한다.
- [사용자 승인/예산] 7-branch 2-arm이 4h를 초과하면 예산 증액 또는 설계 축소를 사용자 판단으로 넘긴다. 5-branch paired를 공식 7-branch 승격으로 대체하지 않는다.

## 하지 말 것 (이유와 함께)
- SEAL 10 결과를 모델 개선 방향에 사용하지 말 것. `PROJECT.md §3.1`은 SEAL 10을 최종 hold-out으로 두고 선택·튜닝에 어떤 형태로도 사용 금지로 고정한다.
- `0.6227`과 `+0.56%p`를 `verified`로 인용하지 말 것. `D-050`은 이 값을 `unverified-hold`로 두고 verified 인용을 금지한다.
- provenance가 없는 저장 예측 파일을 paired Δ 재집계에 사용하지 말 것. `D-050`/`D-053`은 산출물을 특정 구성에 귀속시킬 수 없다고 한다.
- 5-branch `20.0초`와 `3.89 GPU-hour`를 7-branch 실측 비용으로 쓰지 말 것. `current_status.md`는 5-branch 3과제 평균으로 한정한다.
- Primary 7에서 여러 집계 변형을 탐색한 뒤 최댓값을 승격 근거로 사용하지 말 것. `CA-08`은 이 절차를 선택 편향으로 닫았다.
- residualized margin의 게이트 ① 통고를 “새로운 직교 정보”로 사용하지 말 것. 기존 7 margin에 회귀한 잔차는 상관을 구조적으로 낮출 수 있다.
- seed std 감소만 보고 “성능 개선”이라고 쓰지 말 것. `PROJECT.md §4.1`의 추정 대상은 대응 Δ의 기댓값이지 적합 분산이 아니다.
- 4 GPU-hour 안에서 `R=4` 2×2 seed averaging을 시작하지 말 것. `current_status.md`는 `31.1 GPU-hour`로 예산 초과를 기록한다.
- 5-branch paired를 공식 7-branch 승격 비교로 사용하지 말 것. `PROJECT.md §3.2`는 승격 비교를 7-branch 구성에서 수행하도록 고정한다.
- AUROC 변화만으로 기전을 입증하지 말 것. rank 변환, PSW, deterministic construction, LID, mode-count 모두 현재 문서에 기전 실측이 없으므로 “기전 미확인”으로 둔다.
- `RU-85`의 LID `0.314`를 동일 토큰 수 정규화 조건에 그대로 적용해 실행 전 기각으로 확정하지 말 것. 측정 조건이 다르므로, 해당 조건에서 `|r|`을 재측정하기 전에는 판별 불가다.
- 값 기반 오염 검사 통과를 주장하지 말 것. `PROJECT.md §3.4`는 4자리 정밀도를 시스템이 유지하지 못해 통과/실패가 오염에 대해 정보를 주지 못한다고 한다.
- 절대 macro를 과제 모집단의 기대 성능으로 해석하지 말 것. `PROJECT.md §4.1`은 fold 간 비독립과 설계 효과 `5.12배`를 실측했다.
