## B1 (턴 1)
## 제안 1: seed-averaged paired Δ로 estimand 변경
- **바꾸는 것**: Primary 7 승격 비교에서 단일 적합의 per-fold AUROC를 후보 점수로 쓰는 것을 중단하고, 사전 선언된 K개 독립 seed/초기화 적합의 per-fold 평균 AUROC를 후보 점수로 쓴다. 기준선도 동일하게 seed 평균한다. best seed 선택을 금지하고, seed sd·fold×seed 잔차를 함께 보고한다. K는 fold 적합 시간 실측과 4 GPU-hour 예산 허용 범위 안에서 결정하되, 성능을 보고 K를 바꾸지 않는다.
- **바꾸는 전제**: “단일 적합이 모델의 성능을 측정한다”는 전제를 뒤집는다. 단일 적합은 적합 분포의 한 표본이고, 성능 판단은 적합 평균과 적합 분산을 함께 가진 estimand로 해야 한다는 것이다.
- **기대하는 차이**: Primary 7 macro AUROC의 절대 개선은 정량 예측 불가. 기대되는 차이는 판별력이다. seed std 0.0021~0.0083이 승격 임계 +0.0030과 같은 규모이므로, seed 평균이 seed contribution을 줄이면 임계 근처 후보의 거짓 통과/기각이 줄어든다. true Δ가 임계 미만이면 명확한 기각, true Δ가 임계 초과이면 명확한 승격이 가능해질 수 있다. 단, fold×seed 잔차가 23.9%로 실측된 만큼 seed 평균으로 실제로 줄어드는 폭은 정량 예측 불가이며, 모든 결과는 hold-out 미검증이다.
- **기전**: closed-form ridge 경로에도 seed에 따른 nuisance 변동(예: GMM/PCA/표본화 관련 변동)이 있으면, seed 평균은 독립성 정도만큼 seed 특이 변동을 상쇄한다. 성능 개선과 다른 점은 true Δ를 바꾸지 않고 Δ 추정량의 분산만 줄인다는 것이다. fold×seed 잔차가 구조적 fold 상호작용이라면 seed 평균으로 줄지 않으므로 기전 미확인이다.
- **실패 조건**: fold 적합 시간이 측정되지 않았으므로 K×Primary 7이 4 GPU-hour를 초과하면 실행 무효다. seed 평균이 fold×seed 상호작용을 줄이지 못해 Δ SE가 여전히 임계보다 크면 판별 불가다. 적합 분산 포함 승격 절차가 사용자 승인 전이면 verified 승격 근거로 사용할 수 없다. provenance가 없으면 어떤 fit이 어떤 arm인지 귀속할 수 없어 실행 무효다.

## 제안 2: prediction-variance weighted trimmed mean, 단일 적합 재집계
- **바꾸는 것**: trimmed mean voting의 절사 규칙은 유지하되, 절사 후 평균만 가중 평균으로 바꾼다. 가중치는 각 branch의 closed-form fit에서 나오는 context residual variance 또는 PSW/TGW 계열의 fit-derived prediction variance를 사전 선언된 규칙으로 역수화한 값으로 고정한다. 가중치는 Primary 7 성능을 보고 조정하지 않고, branch subset 탐색도 하지 않는다. 실행은 단일 Primary 7 적합으로 7-branch margin을 저장한 뒤, 같은 margin 위에서 baseline trimmed mean과 candidate weighted trimmed mean을 재계산해 paired fold Δ를 구한다.
- **바꾸는 전제**: “branch margin은 모두 같은 scale·신뢰도로 동등 평균해도 된다”는 전제를 뒤집는다. branch별 scale·noise가 다르고, 그 차이는 성능 조회 전에 fit-derived 분산/신호기하로 측정 가능할 수 있다는 것이다. CA-08이 닫은 것은 선택 편향 절차이지, 새 정보를 쓰는 결합 연구 일반은 아니다.
- **기대하는 차이**: 정량 예측 불가. branch scale 불일치가 aggregate noise의 주원인이라면 trimmed mean 대비 fold Δ의 변동이 줄거나, true Δ가 masked되어 있던 후보의 부호가 더 분명해질 수 있다. 4 GPU-hour에서는 seed ensemble보다 단일 적합 재집계 쪽이 판별 가능성이 높다. 단, 이 방식은 true 성능 자체를 올리는 메커니즘이 아니라 집계 단계의 scale 편향만 조정할 수 있다.
- **기전**: 고분산 branch의 기여를 낮춰 aggregate margin의 noise를 줄인다는 것이다. 그러나 CA-08이 이미 “결합 함수 일반의 무효 기전은 미확인”이라고 했으므로, prediction-variance 가중이 실제로 Primary 7 AUROC를 개선하는 기전은 미확인이다.
- **실패 조건**: 단일 Primary 7 전체 적합이 4 GPU-hour를 넘으면 재집계 실험 자체가 실행 무효다. prediction variance가 branch noise가 아니라 과제 난이도·context 크기를 반영하면 가중이 task-specific 편향을 만든다. branch margin scale이 이미 정규화되어 있어 가중 변화가 거의 없으면 Δ가 0으로 판별 불가다. D-050 계측 복구와 provenance/오염 검사 재설계 전에는 margin 저장 산출물을 verified로 쓸 수 없어 실행 무효다.

## 제안 3: gate-first FC 직교성 프로브로 유효 랭크 3.52 해석 분리
- **바꾸는 것**: “branch 추가 여부를 성능 탐색으로 판단”하는 것을 “성능 조회 전 SCREEN_ONLY/provenance 기반 gate ①으로 판단”하는 것으로 바꾼다. 첫 후보는 CA-09 경계 밖으로 명시된 FC, 즉 여러 초점 강도의 bag mean 차분을 쓰는 branch다. 초점 강도 수는 성능으로 고르지 않고 사전 고정한다. 후보를 8-branch로 놓고 max |r|과 eff.rank만 본다. max |r| > 0.6이면 성능을 조회하지 않고 종료한다. 8-branch 효율 비하락 기준 eff.rank ≥ 4.02를 통과하는 경우에만 gate ②와 성능 비교로 넘긴다.
- **바꾸는 전제**: “현 7-branch가 이미 서로 다른 것을 재는 한계”와 “아직 서로 다른 것을 재는 branch를 못 만든 것”을 동시에 주장하지 않는다. 둘은 라벨 무관 직교성 관측으로 분리할 수 있는 별개 가설이다.
- **기대하는 차이**: 성능 개선 예측 불가. 기대되는 것은 판별 관측이다. FC가 gate ①을 통과하면 유효 랭크 3.52는 “아직 만들지 못한 branch” 쪽에 가깝다. 구조적으로 다른 FC마저 max |r| > 0.6 또는 eff.rank 비하락이면, 적어도 FC 계열에서는 현재 descriptor family의 branch 추가가 중복 축에 가깝다. 단, 단일 후보 통과/실패로 전체 branch 공간의 상한을 증명하지는 못한다.
- **기전**: 기존 branch는 PCA-32/공분산/분위수/살리언스 등 slide-level 1차·스펙트럼 통계에 중복될 수 있다. FC는 초점 강도별 평균의 차분을 읽으므로, 단순 mean·quantile·salience가 읽지 않는 scale derivative를 읽을 수 있다. 그러나 이것이 slide-level task signal을 실제로 분리하는지는 기전 미확인이다.
- **실패 조건**: D-050 계측 복구 전에는 SCREEN_ONLY/provenance가 verified가 아니므로 gate 결과가 실행 무효다. FC margin 계산 비용이 fold 적합과 유사하고 4 GPU-hour 내 실행되지 못하면 판별 불가다. FC가 CA-09 경계에서 저촉으로 판정되면 제안 무효다. gate ①을 통과해도 gate ②나 성능 비교에서 이득이 없으면 “정보량 병목”으로 전환해야 하며, 이 경우 rank 해석만 판별할 뿐 성능 개선은 아니다.

가설: seed 평균이 실제로 Δ SE를 줄이려면 seed 주효과가 fold×seed 잔차보다 지배적이어야 하는데, RU-97의 fold×seed 23.9%는 seed 평균의 상한이 낮을 수 있음을 암시한다.
가설: trimmed mean의 병목이 branch 오류가 아니라 margin scale 불일치라면, prediction-variance 가중은 성능 개선이 아니라 aggregate noise의 scale 편향만 제거할 수 있다.
가설: 유효 랭크 3.52의 병목이 branch 생성이 아니라 UNI2 patch feature의 slide-level 압축 단계에 있다면, FC 같은 구조적으로 다른 후보도 gate ①에서 높은 상관을 보일 것이다.

## B2 (턴 1)
## 제안 1: Provenance-anchored paired re-fit probe
- **바꾸는 것**: 모델 구조가 아니라 비교와 실행 경로를 바꾼다. 여섯 실행 러너가 폴백으로 같은 7-branch 설정으로 수렴했던 문제를 없애기 위해, YAML 설정을 직접 읽고 branch list·가중치·코드 해시·seed·fold manifest·데이터 버전을 산출물에 기록하는 단일 러너를 만든다. 산출물은 macro가 아니라 fold별·branch별 margin/AUROC와 paired Δ를 저장한다. 4 GPU-hour의 일부로 fold 1회 적합 시간 실측과, baseline·후보 1개를 같은 fold에서 짝지은 재적합을 수행한다. 오염 검사는 소수 4자리 AUROC 앵커 일치가 아니라 provenance 대조로 바꾼다.
- **바꾸는 전제**: “저장된 과거 실행이 서로 다른 branch 구성을 반영한다”와 “절대 macro와 fold만 있으면 비교가 유효하다”는 전제를 뒤집는다. 실제로는 러너 폴백으로 arm이 다른 실행이 같은 7-branch로 수렴했고, 저장 예측 파일에 provenance가 없어 구성 귀속이 불가능하다.
- **기대하는 차이**: 성능 지표의 정량 예측 불가. 연구 유효성 지표만 바뀐다. fold 1회 적합 시간은 문서에 없으므로 실측 전 예산 판단이 불가능하다. 이 제안이 통과하면 fold별 paired Δ, seed/fold 분산 기저, 후보 간 SE 산출이 가능해지고, 기존 비교의 실행 무효 상태를 줄인다. fold 1회 시간이 350-fold 재적합에 비해 너무 길면 후보 판별은 4 GPU-hour 안에서 불가능해질 수 있다.
- **기전**: provenance가 구성 귀속을 가능하게 하고, 같은 fold에서 짝짓기가 과제 난이도 차이에 의한 fold 간 산포를 상쇄한다. 이는 비교의 유효성을 높이는 경로이지 AUROC를 직접 올리는 경로가 아니다.
- **실패 조건**: YAML 경로라도 숨은 폴백, 라이브러리 nondeterminism, fold manifest 불일치, seed 기록 누락이 남아 구성을 특정하지 못하면 실행 무효다. fold 1회 실측 결과 4 GPU-hour 안에 baseline과 후보를 350-fold로 재적합할 수 없으면 판별 불가다. SH/SJ 가중치·branch list를 기록해도 실제 margin이 그 설정을 반영하지 못하면 실패다.

## 제안 2: Fit-averaged paired Δ aggregation
- **바꾸는 것**: 단일 적합 margin을 branch의 증거로 쓰는 것을 바꾼다. 각 fold에서 baseline과 후보를 같은 seed 세트 R개로 짝지어 재적합하고, branch margin은 R개 적합의 median으로 만든 뒤 기존 trimmed mean으로 집계한다. 승격 비교는 단일 적합의 fold Δ가 아니라 R개 paired seed fit의 fold Δ를 평균한 fit-averaged paired Δ로 둔다. best-seed 선택은 금지한다. 학습 파라미터는 도입하지 않는다. R은 fold 1회 시간 실측 뒤 운영적으로 정하고, Primary 7 성능을 보고 R을 고르지 않는다.
- **바꾸는 전제**: “한 fold의 단일 적합 margin이 branch의 증거”라는 전제를 “branch margin은 적합 seed에 대한 분포이며 단일 적합은 그 분포의 한 뽑기”로 뒤집는다. “적합 분산은 부수적으로 보고할 것”이라는 전제도 “적합 분산은 paired Δ의 정밀도에 직접 반영되는 것”으로 바꾼다.
- **기대하는 차이**: seed std `0.0021~0.0083`은 승격 임계 `+0.0030`보다 작거나 같은 크기이므로, 단일 적합 비교에서는 임계 크기의 차이를 뽑기가 가릴 수 있다. 독립 적합 평균 가정이면 seed 성분은 `1/sqrt(R)` 정도로 줄지만, 문서 실측에서는 seed 주효과가 전체의 `0~2.9%`에 불과하고 fold×seed 잔차가 더 클 수 있다고 되어 있으므로 총분산 감소와 `Δ_macro` 변화는 정량 예측 불가다. 이 제안은 성능 평균 자체를 올린다고 보장하지 않는다. 효과가 `+0.0030` 근처에 있는 후보를 더 신뢰 있게 판정하게 하거나, 효과가 없는 후보를 더 확실하게 음성으로 확인하게 하는 것이다.
- **기전**: 독립적인 적합 노이즈를 평균하면 branch margin의 분산이 줄고, trimmed mean은 더 안정적인 branch 증거를 모은다. 이것이 AUROC를 올리는지는 기전 미확인이다.
- **실패 조건**: fold×seed 잔차나 fold 특이 노이즈가 지배하면 seed 평균으로 paired Δ의 fold-level 불확실성이 충분히 줄지 않아 판별 불가가 남는다. seed가 실제로 독립적이지 않거나, stochastic 원인이 subsampling·initialization·library nondeterminism 중 무엇인지 특정되지 않으면 감소 효과가 없다. fold 1회 시간이 길어 350-fold × R개를 4 GPU-hour 안에 돌리지 못하면 실행 무효다. margin의 median이 AUROC rank를 왜곡해 성능을 떨어뜨리면 실패다. 승격으로 쓰려면 적합 분산을 포함한 판정 절차의 사용자 승전이 선행되어야 한다.

## 제안 3: Non-redundant rank probe — robust full-token LID branch
- **바꾸는 것**: branch 수를 무한정 늘리는 것이 아니라, 현재 7-branch와 중복되지 않는지 먼저 판별하는 단일 후보 branch를 넣는다. 후보는 whitened patch-feature cloud의 local intrinsic dimension, 예컨대 ID_TwoNN 계열 k-NN 거리비 스칼라다. 계산은 full-token 또는 결정론적 full-token pass, 고정 k-NN 스케일, token 수 정규화, 결정론적 tie·stride 규칙을 쓰고, closed-form margin으로 만든다. 성능 조회 전에 게이트 ①을 돌린다. 공식 7-branch 대비 `max |r| > 0.6`이거나 eff.rank가 `4.02` 미만이면 종료한다.
- **바꾸는 전제**: “유효 랭크 `3.52/7`은 branch 추가 방향이 막혀 있다는 뜻”이라는 전제를 “현재 7개는 같은 위치·형상·1차 모멘트 계열을 중복해서 읽고, 서로 다른 기하 축을 아직 만들지 못했다는 뜻일 수 있다”로 뒤집는다. 또한 “LID가 부분추출에서 불안정했으므로 축 전체가 무효”라는 전제를 “불안정성은 추정기·정규화 문제일 수 있다”로 바꾼다. branch별 기여도는 문서에 없으므로 어떤 기존 branch를 제거해야 하는지는 이 제안에서 판단하지 않는다.
- **기대하는 차이**: 게이트 ① 통과 시 eff.rank가 현재 `3.52`에서 `4.02` 이상으로 오르고 `max |r| <= 0.6`이 된다. 이 관측은 유효 랭크가 측정 중복인지, 아니면 서로 다른 것을 재는 branch 부재인지 구분하는 직접적 신호다. 게이트 ① 실패 시 성능을 조회하지 않는다. AUROC 변화는 정량 예측 불가다. 단일 후보 실패만으로 “branch 추가가 막혀 있다”고 결론 지을 수 없고, 그 경우 판정은 후보 기각이지 축 전체 반박이 아니다.
- **기전**: LID는 슬라이드 내 토큰 구름의 국소 차원 기하를 읽으므로, 1차 모멘트 위치 계열이나 스펙트럼 엔트로피 스칼라와 상관만 낮다면 trimmed mean에 독립적 증거를 추가한다. 이것이 AUROC를 올리는지는 기전 미확인이다.
- **실패 조건**: LID가 token 수·크기 proxy이거나 기존 `BD`·`SH`·`SJ`와 상관만 높으면 게이트 ①에서 실패한다. full-token 계산 비용이 4 GPU-hour를 벗어나면 실행 무효다. gate ①을 통과해도 paired Δ가 `+0.0030`에 미달하면, 구간이 0을 포함하면 판별 불가이고, 사전 설계된 비교에서 명확한 음성이면 반박이다. token 수 정규화 후에도 fold 간 불안정성이 남으면 후보 실패다.

가설: 유효 랭크 3.52의 중복은 BM·QA·DS 위치 계열에 집중되어 있어, 후보 branch를 추가하기 전에 기존 7개 margin의 라벨 무관 고유벡터 분해만으로도 “측정 중복”인지 “정보 부재”인지 방향을 좁힐 수 있다.
가설: fit-averaged paired Δ에서 같은 seed 짝짓기로 seed main은 상쇄되지만 fold×seed 잔차는 남으므로, 4 GPU-hour의 첫 실측은 fold 1회 시간보다 fold×seed 잔차가 paired Δ 정밀도에 얼마나 기여하는지 재는 데 우선되어야 한다.
가설: LID가 token 수 proxy가 아니라면 log N 정규화 후 max|r|이 낮아질 것이므로, 성능 조회 전에 max|r|과 |r|(LID, log N)만으로 후보를 죽일 수 있다.

## B1 (턴 2)
## 제안 1: 동일 seed·동일 draw paired Δ와 1-fold time/variance probe
- **바꾸는 것**: 승격 비교의 기본 점수를 단일 적합의 fold AUROC에서, 같은 fold·같은 seed·같은 stochastic draw(동일 subsample index, GMM initialization, PCA sign/phase, library random state)를 공유한 baseline과 candidate의 paired fold Δ로 바꾼다. seed average K개를 우선하지 않는다. 4 GPU-hour에서는 먼저 사전 고정된 최소 fold 수로 fold 1회 적합 시간과 paired Δ의 변동 성격을 실측하고, 350-fold paired 재적합이 예산 안에 들어가는 경우에만 Primary 7 전체 비교로 확장한다. 최소 fold 실험은 판정이 아니라 예산·분산 진단이다.
- **바꾸는 전제**: “적합 분산은 seed를 여러 번 평균해서만 줄일 수 있다”와 “seed std가 fold Δ의 주된 노이즈”라는 전제를 뒤집는다. seed std의 상당 부분이 baseline과 candidate 공통의 nuisance라면, K-seed average보다 동일 draw pairing이 더 적은 fit 수로 seed main을 상쇄한다. B1의 K-seed average는 유효할 수 있으나, fold 적합 시간이 미측정인 상황에서는 fit 수 부담이 크므로 paired probe가 먼저여야 한다.
- **기대하는 차이**: Primary 7 macro AUROC의 절대 개선은 정량 예측 불가. 기대되는 차이는 paired Δ의 SE 감소다. RU-97 GF에서 seed 주효과가 전체의 0~2.9%로 작았고, m32_raw에서 fold×seed 잔차가 23.9%로 실측됐다. 이 값은 현재 7-branch로 직접 일반화할 수 없지만, seed average만으로 fold×seed 성분을 충분히 줄인다고 단정할 근거가 부족함을 보여준다. 따라서 paired Δ SE 감소 폭은 정량 예측 불가이며, +0.0030 임계 근처 후보의 거짓 통과/기각이 줄거나 음성이 더 분명해질 수 있다. 모든 결과는 hold-out 미검증이다.
- **기전**: baseline과 candidate가 동일한 stochastic realization을 공유하면, seed-specific 변동은 Δ에서 상쇄된다. 단, fold×seed 상호작용이 candidate-specific이면 남아 있다. 이 제안은 true Δ를 직접 올리는 메커니즘이 아니라 Δ 추정량의 불확실성을 줄이는 경로다. AUROC를 실제로 개선하는 기전은 미확인이다.
- **실패 조건**: fold 1회 적합 시간이 문서에 없으므로, 350-fold paired 재적합이 4 GPU-hour를 초과하면 실행 무효다. 두 arm의 stochastic draw를 실제로 동일하게 전달할 수 없으면 상쇄가 성립하지 않는다. fold×seed 잔차 또는 candidate-specific seed 민감도가 지배하면 SE가 여전히 +0.0030보다 커 판별 불가다. D-050 이전의 provenance/오염 검사 복구 전에는 verified 승격 근거로 쓸 수 없다. 적합 분산을 포함한 승격 절차가 사용자 미승인 상태이면 verified 승격은 불가하다. baseline 재적합까지 포함하면 2×350 fold fit이 필요할 수 있어, K-seed average보다 싼 것이 아니게 되면 제안이 무력화된다.

## 제안 2: context-only robust scale + closed-form branch-combination ridge
- **바꾸는 것**: 기존 trimmed mean voting을, fold별 closed-form branch-combination ridge로 바꾼다. 구체적으로 각 fold에서 7개 branch margin을 context-slide margin의 median/MAD로 robust하게 scale한 뒤, context 라벨만으로 ridge 계수를 닫힌 형태로 풀고 query slide margin에 적용한다. 경사하강 학습 파라미터는 없다. 정규화 강도는 사전 고정하고 Primary 7 성능을 보고 고르지 않는다. branch subset 탐색도 하지 않는다. trimmed mean은 기준선으로 유지하고, candidate는 이 단일 aggregation만 쓴다.
- **바꾸는 전제**: “7개 branch margin의 scale과 신뢰도가 이미 비교 가능하고, 등가 trimmed mean이 최적”이라는 전제를 뒤집는다. branch별 scale·noise·fold별 기여가 다를 수 있고, 그 차이는 context 라벨을 쓰는 closed-form readout으로 fold별로 조정할 수 있다는 것이다. B1의 prediction-variance weighted trimmed mean보다 강한 버전으로, 가중치를 fit-derived 분산에 고정하는 것이 아니라 context에서 closed-form ridge로 풀게 한다. CA-06의 경계 밖은 “기저는 무감독으로 두고 readout 단에서만 라벨을 쓰는 것”이므로, branch margin 위의 readout으로 해석된다.
- **기대하는 차이**: 정량 예측 불가. branch weight가 fold/context별로 달라야 한다면 Primary 7 macro AUROC가 오를 수 있다. branch가 이미 redundant하고 trimmed mean이 거의 최적이라면 Δ는 0에 가깝다. 이 제안은 variance reduction만이 아니라 slide score ordering 자체를 바꾸므로 expected AUROC와 fit variance 둘 다를 바꿀 수 있다. N_ctx는 문서 실측으로 90~261장이므로 7차원 branch margin에 대한 ridge는 자유도상 가능하지만, 과적합 여부는 미검증이다.
- **기전**: ridge는 branch margin의 선형 결합을 context 라벨로부터 닫힌 형태로 추정한다. robust scale은 high-scale branch가 정규화 페널티와 slide score를 지배하는 것을 낮춘다. 유효 랭크 3.52/7으로 branch 간 중복이 크다면 ridge가 중복을 완화할 수 있으나, 중복을 정보 부재로 바꾸지는 못한다. AUROC를 실제로 개선하는 기전은 미확인이다.
- **실패 조건**: context-only median/MAD가 0에 가깝거나 fold별로 불안정하면 scale 변환 자체가 노이즈가 된다. context와 query 분포 이대가 있으면 context에서 추정한 계수가 query에서 역효과를 낼 수 있다. branch margin이 저장되지 않거나 provenance가 없으면 재집계 자체가 실행 무효다. 단일 Primary 7 전 fold 적합이 4 GPU-hour를 넘으면 실행 무효다. 여러 ridge 강도·aggregation 후보를 돌리고 Primary 7 최댓값을 고르면 CA-08의 선택 편향에 걸리므로, 이 제안은 단일 사전 고정 규칙으로만 유효하다. closed-form 계수라도 fold별 변동이 크면 적합 분산이 커져 승격 판정을 더 어렵게 만들 수 있다.

## 제안 3: descriptor-level effective-rank audit + 단일 FC 직교성 게이트
- **바꾸는 것**: “margin 유효 랭크 3.52/7”만으로 branch 추가 여부를 판단하는 것을, (a) 기존 7개 branch margin의 rank, (b) branch가 읽는 raw slide-level descriptor block(cv non-diagonal, PCA-32 bag mean, spectrum, quantile, salience, shape/radius 계열)의 effective rank, (c) 사전 고정된 단일 후보 FC의 gate ① 관측으로 분리하는 것으로 바꾼다. FC는 CA-09 경계 밖으로 명시된 “여러 초점 강도의 bag mean 차분” 계열이며, 초점 강도 수는 성능을 보고 고르지 않는다. 성능 조회 전에 max |r|, 8-branch eff.rank, 필요시 token 수/크기 proxy 상관을만 본다.
- **바꾸는 전제**: “margin rank만으로도 branch 추가 가능 여부를 판단할 수 있다”는 전제를 뒤집는다. margin rank는 readout/branch 설계가 정보를 압축한 결과이지, raw descriptor에 unused axis가 있는지를 직접 말해주지 않는다. B2의 full-token LID probe보다 먼저, 이미 존재하는 descriptor의 rank를 재어 “새 branch를 만들 가치가 있는가”를 더 싸게 판별한다.
- **기대하는 차이**: AUROC 개선 예측 불가. 기대되는 것은 label-free 진단이다. raw descriptor block의 effective rank가 margin rank보다 유의하게 높고, FC가 max |r| ≤ 0.6 및 8-branch eff.rank ≥ 4.02를 통과하면 “서로 다른 것을 재는 branch를 아직 못 만들었다”는 해석에 무게가 실린다. 반대로 raw descriptor rank도 낮거나, 구조적으로 다른 FC마저 gate ①을 통과하지 못하면 “현재 patch feature/slide-level compression에서 branch 추가가 중복 축에 가깝다”는 해석에 무게가 실린다. 단, 단일 후보 통과/실패로 전체 branch 공간의 상한을 증명하지는 못한다.
- **기전**: margin은 descriptor의 closed-form readout이다. descriptor 공간에 아직 읽히지 않는 axis가 있으면, FC처럼 다른 수학적 대상(scale derivative)을 읽는 branch가 margin rank를 높일 수 있다. 반면 patch feature, PCA-32, slide-level aggregation 자체가 low-rank이면 어떤 branch readout도 rank를 크게 올리지 못한다. FC가 AUROC를 실제로 개선하는지는 기전 미확인이다.
- **실패 조건**: raw descriptor가 저장되어 있지 않거나 patch feature 재추출 시간이 문서에 없으므로, 4 GPU-hour 안에 재계산이 불가능하면 실행 무효다. D-050으로 provenance/오염 검사 재설계가 완료되기 전이면 gate ① 산출물을 verified 근거로 쓰지 못한다. FC 계산 비용이 fold 적합과 유사하거나 더 크면 예산 내 판별이 어렵다. FC가 CA-09 경계에서 다시 저촉으로 판정되면 제안 무효다. gate ①을 통과해도 paired Δ가 +0.0030에 미달하면, 이 제안이 보여주는 것은 rank 해석이지 성능 개선이 아니다. branch별 기여도는 문서에 없으므로 어떤 기존 branch를 제거해야 하는지 이 제안으로 판단하지 않는다.

가설: fold 1회 적합 시간이 4 GPU-hour의 첫 병목이면, margin 재집계나 descriptor audit보다 먼저 1-fold time probe와 context/query margin 저장 경로가 다음 판정의 실행 가능성을 결정한다.
가설: closed-form branch-combination ridge가 folded context에서 계수를 풀 때 계수 자체의 fold×seed 변동이 크면, AUROC bias는 줄고 fit variance만 커지는 결과가 나와 trimmed mean 대비 paired Δ SE를 오히려 악화시킬 수 있다.
가설: raw descriptor rank audit에서 margin rank보다 raw descriptor rank가 높지 않다면, 새 branch 추가보다 patch feature의 slide-level compression/whitening/PCA-32 축을 건드리는 쪽이 더 유효한 후보일 수 있다.

## B2 (턴 2)
## 제안 1: Provenance-anchored paired fold-time/variance probe
- **바꾸는 것**: 모델 구조를 먼저 바꾸지 않는다. 실행·비교 경로를 바꾼다. 여섯 러너가 폴백으로 같은 7-branch 설정으로 수렴했던 문제를 없애기 위해, YAML 설정을 직접 읽고 branch list·가중치·코드 해시·seed·fold manifest 해시·데이터 버전을 산출물에 기록하는 단일 러너를 만든다. 4 GPU-hour의 첫 job은 fold 1회 적합 시간과 provenance round-trip만 측정한다. 이후 예산이 허용하는 최소 범위로 baseline 7-branch와 한 후보를 같은 fold에서 짝지은 재적합을 수행한다. seed가 관측되면 R개 독립 seed를 같은 fold·seed로 짝지어 margin을 평균하고, best-seed 선택을 금지한다. full 350-fold가 4 GPU-hour 안에 불가능하면 승격 시도로 삼지 않고 `실행 무효`로 기록한다.
- **바꾸는 전제**: “저장된 과거 실행은 서로 다른 arm을 반영한다”와 “fold 1회 적합 시간이 예산에 맞는지 이미 알고 있다”는 전제를 뒤집는다. 실제로는 러너 폴백으로 arm이 다른 실행이 같은 구성으로 수렴했고, fold 1회 시간은 문서에 없다.
- **기대하는 차이**: AUROC의 정량 예측 불가. 기대되는 차이는 판정 유효성이다. fold 1회 시간, seed 주효과, fold×seed 잔차, paired Δ의 SE, provenance 기반 구성 귀속이 실측된다. seed std `0.0021~0.0083`이 승격 임계 `+0.0030`과 같은 규모이므로, seed 성분이 독립적이라면 R개 seed 평균은 Δ SE를 줄여 임계 근처 후보의 거짓 통과/기각을 줄일 수 있다. fold×seed 잔차가 지배하면 감소가 제한적이다. 모든 결과는 `hold-out 미검증`이다.
- **기전**: provenance가 어떤 fit이 어떤 arm인지 귀속 가능하게 하고, 같은 fold에서 짝짓기가 과제 난이도 차이에 의한 fold 간 산포를 상쇄한다. 독립 적합 노이즈의 평균은 seed 특이 변동을 줄인다. 그러나 fold×seed 잔차가 구조적 fold 상호작용이라면 seed 평균으로 줄지 않으므로, Δ SE 감소 폭은 `기전 미확인`이다.
- **실패 조건**: hidden fallback, 라이브러리 nondeterminism, fold manifest 불일치, seed·코드 해시 누락, provenance 대조 실패가 있으면 `실행 무효`다. fold 1회 시간 실측 결과 baseline+후보의 full 350-fold 재적합이 4 GPU-hour를 초과하면 `실행 무효`다. R개 seed 평균을 했어도 Δ 구간이 여전히 0을 포함하면 `판별 불가`다. seed가 실제로 독립적이지 않거나 stochastic 원인이 fold 순서·subsampling·library nondeterminism으로 특정되지 않으면 분산 감소 기전이 성립하지 않는다. `D-050` 계측 복구 전에는 산출물을 `verified`로 쓸 수 없어 승격 근거로 사용 불가다.

## 제안 2: Fit-derived uncertainty-weighted trimmed mean
- **바꾸는 것**: trimmed mean voting의 절사 규칙은 유지하되, 절사 후 central average만 equal-weight가 아니라 fit-derived inverse variance/scale weight로 바꾼다. 가중치는 각 branch의 closed-form fit에서 나온 context residual variance 또는 prediction variance를 사전 선언된 공식으로 역수화한 값, 또는 deterministic per-fold margin scale 정규화 값으로 고정한다. 경사하강 학습 파라미터를 넣지 않고, branch subset 탐색을 하지 않고, Primary 7 AUROC를 보고 가중치를 고르지 않는다. 한 full Primary 7 적합에서 7-branch margin을 저장한 뒤, baseline trimmed mean과 candidate weighted trimmed mean을 같은 fold에서 paired Δ로 재계산한다.
- **바꾸는 전제**: “7개 branch margin은 같은 scale·신뢰도로 동등 평균해도 된다”는 전제를 뒤집는다. branch별 scale·noise가 다르고, 그 차이는 성능 조회 전에 fit-derived 분산·신호기하로 측정 가능할 수 있다는 것이다. `CA-08`이 닫은 것은 선택 편향에 기반한 결합 탐색이지, prediction variance 같은 새 정보를 쓰는 사전 선언 결합 연구 일반은 아니다.
- **기대하는 차이**: 정량 예측 불가. branch scale·noise 불일치가 aggregate margin noise의 주원인이라면 trimmed mean 대비 fold Δ의 변동이 줄거나, true Δ가 mask되어 있던 후보의 부호가 더 분명해질 수 있다. 그러나 이는 true 성능 평균을 올리기보다 집계 단계의 scale·노이즈 편향을 조정하는 경로다. margin scale이 이미 충분히 정규화되어 있으면 paired Δ는 0 근처로 `판별 불가`가 된다. 모든 결과는 `hold-out 미검증`이다.
- **기전**: 고분산 또는 scale-distorted branch의 기여를 낮춰 aggregate margin의 noise를 줄인다는 것이다. 다만 `CA-08`이 “결합 함수 일반의 무효 기전은 미확인”이라고 했으므로, prediction-variance 가중이 Primary 7 AUROC를 실제로 개선하는 기전은 `기전 미확인`이다.
- **실패 조건**: one full Primary 7 적합이 4 GPU-hour를 초과하면 `실행 무효`다. `D-050` 계측 복구 전 margin 산출물을 `verified`로 쓸 수 없어 `실행 무효`다. 가중치가 branch noise가 아니라 과제 난이도·context 크기·염색 편차를 반영하면 task-specific bias를 만들어 후보를 왜곡한다. 한 후보만 사전 선언해도 결과 AUROC를 보고 가중식·절사 수·subset을 바꾸면 `CA-08` 선택 편향 경계에 저촉된다. paired Δ 구간이 0을 포함하면 `판별 불가`이며, 사전 설계된 paired 비교에서 명확한 음성이면 후보 기각이지 축 전체 반박은 아니다.

## 제안 3: Gate-first FC rank probe with eigen diagnostic
- **바꾸는 것**: “branch 추가 여부를 성능 탐색으로 판단”하는 것을 “성능 조회 전 라벨 무관 gate ①으로 판단”하는 것으로 바꾼다. 먼저 기존 7-branch margin의 standardize한 상관 행렬과 고유값 스펙트럼을 계산해 중복이 특정 family에 집중되어 있는지, 아니면 전체 descriptor family에 걸쳐 있는지 본다. 이후 `CA-09` 경계 밖으로 명시된 FC 후보, 즉 여러 초점 강도의 bag mean 차분을 쓰는 closed-form branch를 `SCREEN_ONLY`로 넣는다. 초점 강도 수는 성능으로 고르지 않고 사전 고정한다. 8-branch 기준 `max |r| <= 0.6`이고 eff.rank가 `4.02` 이상이면 gate ②와 paired 성능 비교로 넘긴다. 실패하면 성능을 조회하지 않고 후보 기각으로 종료한다. 기존 branch 제거는 제안하지 않는다. branch별 기여도는 문서에 없다.
- **바꾸는 전제**: “유효 랭크 `3.52/7`은 branch 추가 방향이 이미 막혀 있다는 뜻”이라는 전제를 “현재 7개는 같은 위치·형상·1차 모멘트 계열을 중복해서 읽고, 서로 다른 것을 재는 branch를 아직 만들지 못했을 수 있다”는 가설로 바꾼다. 둘은 라벨 무관 직교성 관측으로 분리할 수 있는 별개 가설이다.
- **기대하는 차이**: AUROC의 정량 예측 불가. 기대되는 것은 판별 관측이다. FC가 gate ①을 통과하면, 적어도 FC 계열에서는 “서로 다른 것을 재는 branch를 아직 못 만들었다” 쪽에 가깝다. FC가 `max |r| > 0.6` 또는 eff.rank `4.02` 미만이면, FC 계열에서는 현재 7-branch와 중복이 크다는 후보 기각을 얻지만, branch 공간 전체의 상한을 증명하지는 못한다. eigen diagnostic는 중복이 `BM·QA·DS` 같은 위치 계열에 집중되어 있는지, 아니면 전체 margin family에 걸쳐 있는지 구분하는 보조 신호다.
- **기전**: FC는 초점 강도별 bag mean의 차분을 읽으므로, 단순 mean·quantile·salience가 직접 읽지 않는 scale derivative를 읽을 수 있다. 이 차분이 기존 branch와 직교하면 trimmed mean에 독립적 증거를 추가할 수 있다. 그러나 이것이 slide-level task signal을 실제로 분리해서 AUROC를 올리는지는 `기전 미확인`이다.
- **실패 조건**: `D-050`으로 `SCREEN_ONLY`/provenance 경로가 복구되지 않으면 gate ① 결과가 `실행 무효`다. FC margin 계산 비용이 fold 적합과 유사하거나 더 커서 4 GPU-hour 안에 baseline+FC gate를 완료하지 못하면 `실행 무효`다. FC가 `CA-09` 경계에서 저촉으로 판정되면 제안 무효다. gate ①을 통과해도 paired Δ가 `+0.0030`에 미달하면, 구간이 0을 포함하면 `판별 불가`이고, 사전 설계된 비교에서 명확한 음성이면 후보 반박이다. gate ① 실패는 FC 후보 기각이지 “branch 추가 축 전체가 막혀 있다”는 반박이 아니다.

가설: 4 GPU-hour에서 verified 승격 판별이 가능하려면 fold 1회 시간 실측이 먼저이고, 그 시간이 350-fold baseline+후보 재적합을 허용하지 않으면 이 회차는 `실행 무효`를 기록하고 축을 바꾸어야 한다.  
가설: fit-derived variance 가중이 trimmed mean을 이기려면, branch margin의 fold 내 scale·노이즈 편차가 task signal보다 크다는 라벨 무관 진단이 먼저 나와야 한다.  
가설: FC가 gate ①을 통과해도 paired Δ가 `+0.0030`에 미달하면, 병목은 branch 수나 유효 랭크가 아니라 UNI2 patch feature의 slide-level 압축 정보량이다.

## B1 (턴 3)
## 제안 1: Provenance-common-draw 1-fold timing/variance probe
- **바꾸는 것**: 모델 구조 변경을 보류하고, 단일 러너가 YAML 설정·branch list·가중치·코드 해시·seed·fold manifest 해시·데이터 버전을 산출물에 기록하게 한다. 4 GPU-hour의 첫 작업은 fold 1회 적합 시간과 provenance round-trip, 그리고 최소 fold 수에서 baseline/candidate를 같은 fold·같은 seed·같은 stochastic draw로 짝지어 Δ의 변동 성분을 재는 probe다. 이 probe는 승격 판정이 아니라 예산·귀속·분산 진단이다.
- **바꾸는 전제**: “과거 저장 실행은 서로 다른 arm을 반영한다”와 “fold 1회 시간이 예산 판단에 이미 알려져 있다”는 전제를 뒤집는다. 여섯 러너가 폴백으로 같은 7-branch로 수렴했고 fold 1회 시간은 문서에 없으므로, 먼저 실행 유효성과 시간 문턱을 실측해야 한다.
- **기대하는 차이**: AUROC 정량 예측 불가. 기대되는 차이는 (a) full 350-fold single fit 또는 baseline+candidate 700-fit이 4 GPU-hour에 가능한지, (b) seed std `0.0021~0.0083` 중 common-draw pairing이 상쇄하는 공통 seed 성분의 규모, (c) fold×seed 잔차의 상대 규모를 아는 것이다. true Δ는 변하지 않고 paired Δ estimator의 불확실성만 줄이는 방향이다.
- **기전**: provenance는 arm 귀속을 가능하게 하고, 같은 draw pairing은 baseline/candidate 공통 stochastic nuisance를 Δ에서 상쇄한다. 단, candidate-specific seed 민감도나 fold×seed 상호작용이 남으면 감소 폭은 정량 예측 불가이며, AUROC 개선 기전은 미확인이다.
- **실패 조건**: hidden fallback, seed·code hash·fold manifest 누락, 라이브러리 nondeterminism으로 provenance 대조가 실패하면 실행 무효. fold 1회 시간 실측 후 full 350-fold 또는 700-fit이 4 GPU-hour를 넘으면 full 승격 비교는 실행 무효. `D-050` 계측 복구 전 산출물은 verified가 아니므로 승격 근거로 불가. common-draw pairing을 실제로 전달할 수 없으면 분산 감소 기전이 성립하지 않고, Δ 구간이 0을 포함하면 판별 불가.

## 제안 2: Context-only closed-form meta-ridge aggregation
- **바꾸는 것**: trimmed mean voting의 equal-weight central average를, fold별로 7개 branch margin을 median/MAD로 robust scale한 뒤 context 라벨만으로 closed-form ridge 계수를 구해 query margin에 적용하는 단일 aggregation으로 바꾼다. 경사하강 학습 파라미터는 없고, branch subset 탐색도 없고, Primary 7 AUROC를 보고 ridge 강도·절사 수·가중식을 바꾸지 않는다. 기존 trimmed mean은 기준선으로 유지한다.
- **바꾸는 전제**: “7개 branch margin은 같은 scale·신뢰도로 비교 가능하고, 등가 trimmed mean이 이미 적절하다”는 전제를 뒤집는다. branch별 fold/context별 상대 신뢰도가 달라야 한다면, 그 차이는 context 라벨을 readout 단에서만 쓰는 closed-form으로 조정할 수 있다는 것이다.
- **기대하는 차이**: 정량 예측 불가. branch weight가 context/task별로 달라야 한다면 Primary 7 macro AUROC가 오를 수 있고, branch가 이미 redundant하면 Δ는 0 근처로 판별 불가. 이 방식은 slide score ordering 자체를 바꾸므로 expected AUROC와 적합 분산 둘 다를 바꿀 수 있다. `N_ctx` 90~261이면 7차원 ridge는 자유도상 가능하지만 과적합은 미검증.
- **기전**: robust scale은 scale이 큰 branch가 slide score를 지배하는 것을 낮추고, context-only ridge는 context 슬라이드에서 branch margin의 선형 결합을 closed-form으로 추정한다. `CA-06`의 경계 밖인 readout-only label use에 해당한다고 보지만, context↔query 분포 이동이 있으면 계수가 query에서 역효과를 낼 수 있어 성능 개선 기전은 미확인.
- **실패 조건**: context margin의 median/MAD가 0 근처 또는 fold별 불안정하면 scale 변환이 노이즈. context와 query 분포 이동이 크면 계수 과적합으로 paired Δ가 음수. `D-050`/provenance 복구 전 margin을 verified로 쓸 수 없어 실행 무효. one full Primary 7 fit이 4 GPU-hour를 넘으면 실행 무효. ridge 강도·subset을 결과 보고 바꾸면 `CA-08` 선택 편향에 저촉. closed-form 계수라도 fold별 변동이 크면 적합 분산만 커져 승격 판정을 어렵게 만들 수 있다.

## 제안 3: Descriptor-rank audit + FC gate-first probe
- **바꾸는 것**: 유효 랭크 `3.52/7`만으로 branch 추가를 막았다고 결론짓는 것을, (a) 기존 7-branch margin의 standardize 상관/고유값, (b) branch가 읽는 raw slide-level descriptor block의 effective rank, (c) `CA-09` 경계 밖 FC 단일 후보의 gate ①으로 분리하는 절차로 바꾼다. FC는 초점 강도별 bag mean 차분을 읽는 closed-form branch이고, 강도 수는 성능으로 고르지 않는다. `max |r| > 0.6` 또는 8-branch eff.rank `< 4.02`면 성능을 조회하지 않고 후보 기각.
- **바꾸는 전제**: “margin 유효 랭크 `3.52/7`은 branch 추가 방향이 이미 막혔다는 직접 증거”라는 전제를 “margin rank는 readout/압축 결과일 뿐, raw descriptor에 아직 읽히지 않는 축이 있는지는 별개 관측”으로 바꾼다.
- **기대하는 차이**: AUROC 정량 예측 불가. raw descriptor rank가 margin rank보다 높고 FC가 `max |r| ≤ 0.6`, eff.rank `≥ 4.02`를 통과하면 “서로 다른 것을 재는 branch를 아직 못 만들었다”는 해석에 무게. raw descriptor rank도 낮거나 FC가 실패하면 “현재 patch feature/slide-level compression에서 branch 추가가 중복 축에 가깝다”는 해석에 무게. 단, 단일 후보 실패로 전체 branch 공간 상한을 증명하지 못한다.
- **기전**: margin은 descriptor의 closed-form readout이다. raw descriptor에 unused axis가 있으면 FC처럼 scale derivative를 읽는 branch가 rank를 높일 수 있고, patch feature/PCA-32/slide-level aggregation 자체가 low-rank이면 readout 변경만으로 rank가 크게 오르지 않는다. FC가 AUROC를 올리는지는 기전 미확인.
- **실패 조건**: `D-050`/provenance/`SCREEN_ONLY` 복구 전 gate 산출물은 verified가 아니므로 실행 무효. raw descriptor 재추출 또는 FC margin 계산이 fold 적합과 유사하거나 4 GPU-hour를 넘으면 실행 무효. FC가 `CA-09` 경계에서 저촉으로 판정되면 제안 무효. gate ①을 통과해도 paired Δ가 `+0.0030`에 미달하면, 구간이 0을 포함하면 판별 불가이고 사전 설계된 paired 비교에서 명확한 음성이면 후보 반박이지 축 전체 반박이 아니다.

가설: fold 1회 시간이 4 GPU-hour의 문턱을 정한다면, 1-fold timing probe가 통과하지 못하면 context meta-ridge와 FC gate 모두 full 350-fold 판별이 아니라 재집계/스크리닝만 남는다.  
가설: context-only meta-ridge의 이득이 branch margin의 fold별 상대 신뢰도에서 나오려면, context residual variance 또는 margin scale의 fold 분산이 task signal보다 먼저 라벨 무관으로 높아야 한다.  
가설: raw descriptor rank audit에서 raw rank가 margin rank보다 높지 않다면, 새 branch보다 slide-level compression(PCA-32/whitening/aggregation)을 바꾸는 쪽이 더 낮은 비용으로 effective rank를 올릴 수 있다.

## B2 (턴 3)
## 제안 1: Provenance-anchored common-draw paired Δ probe
- **바꾸는 것**: 승격 비교를 “각 arm의 독립적 단일 적합”에서 “같은 fold·같은 stochastic draw를 공유하는 paired 재적합”으로 바꾼다. 단일 러너가 YAML에서 branch list·가중치·코드 해시·seed·fold manifest 해시·데이터 버전을 읽고, 산출물에 per-fold margin/AUROC와 provenance를 기록한다. baseline과 후보는 가능한 한 같은 subsample index, GMM initialization, PCA sign/phase, library random state를 공유한다. 4 GPU-hour의 첫 job은 full 350-fold 승격이 아니라, fold 1회 적합 시간과 paired Δ의 변동 성격을 실측하는 사전 고정 진단으로 둔다. full 350-fold가 예산 안에 들어가는 경우에만 Primary 7 전체 paired 비교로 확장하고, 그 전에 후보를 성능으로 바꾸지 않는다.
- **바꾸는 전제**: “저장된 과거 실행이 서로 다른 arm을 반영한다”는 전제와 “fold 1회 적합 시간이 이미 예산과 양립한다는 전제”를 뒤집는다. 문서에 fold 적합 시간은 없으며, 여섯 러너가 폴백으로 같은 7-branch 설정으로 수렴한 적이 있으므로, 구성 귀속이 확인되지 않는 단일 적합 비교는 연구 근거가 약하다. 또한 “적합 분산은 seed를 여러 번 평균해야만 줄 수 있다”는 전제도 뒤집는다. baseline과 후보가 공유하는 nuisance가 있으면, K-seed average보다 common-draw pairing이 더 적은 fit 수로 seed 특이 변동을 상쇄할 수 있다.
- **기대하는 차이**: Primary 7 macro AUROC의 절대 개선은 정량 예측 불가. 기대되는 차이는 paired Δ의 SE와 실행 유효성이다. seed std `0.0021~0.0083`이 승격 임계 `+0.0030`과 같은 규모이므로, common-draw pairing이 공통 seed 변동을 상쇄하면 임계 근처 후보의 거짓 통과/기각이 줄 수 있다. 그러나 fold×seed 잔차나 candidate-specific seed 민감도가 지배하면 SE 감소가 제한되어 `판별 불가`가 남는다. 모든 결과는 `hold-out 미검증`이다.
- **기전**: provenance는 어떤 fit이 어떤 arm·설정·fold인지 귀속 가능하게 한다. 같은 fold에서 짝짓기는 과제 난이도 차이에 의한 fold 간 산포를 상쇄한다. common-draw pairing은 baseline과 후보가 공유하는 stochastic nuisance를 Δ에서 상쇄한다. 단, 라이브러리 nondeterminism, fold 순서, candidate-specific seed 민감도, context↔query 분포 이동이 남아 있으면 Δ SE 감소 폭은 `기전 미확인`이다.
- **실패 조건**: hidden fallback, 코드 해시·seed·fold manifest 누락, 라이브러리 nondeterminism, fold manifest 불일치 중 하나가 확인되면 `실행 무효`다. fold 1회 시간 실측 결과 baseline+후보의 full 350-fold paired 재적합이 4 GPU-hour를 초과하면 full 승격 비교는 `실행 무효`이고, 이번 회차는 예산·정밀도 진단으로만 종료된다. `D-050` 계측 복구와 provenance 기반 오염 검사 재설계 전에는 산출물을 `verified` 승격 근거로 쓸 수 없다. 공통 draw를 실제로 전달할 수 없으면 상쇄가 성립하지 않는다. paired Δ 구간이 0을 포함하면 `판별 불가`, 사전 고정 비교에서 명확한 음성이면 후보 기각이지 축 전체 반박은 아니다.

## 제안 2: Fold-local rank/quantile trimmed mean
- **바꾸는 것**: 7-branch margin을 그대로 쓰되, trimmed mean voting 전에 각 branch margin을 fold-local rank 또는 quantile로 결정론적으로 변환한다. 변환은 context slide margin 분포만 사용하고, query label은 쓰지 않는다. context slide의 margin empirical quantile을 만들고, query slide margin을 그 context 분포에서 차지하는 quantile 위치로 환산한다. 환산 뒤 기존처럼 최저·최고 1개 절사 후 중앙값 평균을 쓴다. 경사하강 학습 파라미터는 없으며, branch subset 탐색도 없고, Primary 7 AUROC를 보고 rank/quantile 방식을 바꾸지 않는다. trimmed mean은 기준선으로 유지하고, 이 rank/quantile trimmed mean만 단일 후보로 둔다.
- **바꾸는 전제**: “branch margin은 이미 같은 scale과 신뢰도를 가지므로 동등 평균해도 된다”는 전제를 뒤집는다. AUROC는 rank 기반이므로 branch별 monotone transform은 branch 단독 AUROC를 바꾸지 않으면서, branch 간 상대 scale·tail·noise가 ensemble ordering에 주는 영향을 바꿀 수 있다. `CA-08`이 닫은 것은 성능 조회 뒤 최댓값을 고르는 탐색 절차이지, 사전 선언된 단일 결합 규칙 자체는 아니다.
- **기대하는 차이**: Primary 7 macro AUROC 개선은 정량 예측 불가. 기대되는 차이는 branch scale 불일치로 인한 aggregate noise 감소 또는 ensemble ordering 개선이다. branch margin scale이 fold별로 다르고, 고scale branch가 trimmed mean을 지배하는 편향이 크면 paired Δ가 `+0.0030`에 가까워질 수 있다. 반대로 margin scale이 이미 충분히 비교 가능하면 paired Δ는 0 근처에서 `판별 불가`가 된다. 적합 분산도 함께 보고해야 하며, 모든 결과는 `hold-out 미검증`이다.
- **기전**: rank/quantile 환산은 각 branch margin을 공통 확률 스케일로 옮겨, branch별 scale·tail이 ensemble margin에 주는 비대칭 영향을 줄인다는 것이다. 이것이 AUROC를 실제로 개선하는지는 `기전 미확인`이다. context 분포가 query 분포를 잘 대표하면 query slide의 branch별 상대 위치가 더 안정적으로 반영될 수 있으나, context↔query 분포 이대가 크면 quantile 환산 자체가 노이즈가 된다.
- **실패 조건**: `CA-07` 또는 `CA-08` 경계에서 context 신호 사용·결합 변형이 저촉으로 판정되면 제안 무효다. context slide 수가 `90~261`으로 fold마다 다르므로, context quantile 추정 자체가 fold 간 불안정하면 fit variance가 커진다. context와 query margin 분포 이대가 크면 후보가 기준선보다 나빠질 수 있다. `D-050` 계측 복구 전 margin 산출물을 `verified`로 쓸 수 없어 `실행 무효`다. full Primary 7 적합이 4 GPU-hour를 초과하면 `실행 무효`다. paired Δ 구간이 0을 포함하면 `판별 불가`이며, 사전 고정 비교에서 명확한 음성이면 후보 기각이다.

## 제안 3: Label-free margin eigen audit + 단일 FC orthogonality gate
- **바꾸는 것**: “branch를 더 넣을지 말지”를 성능 탐색으로 판단하는 것을, 성능 조회 전 라벨 무관 진단으로 바꾼다. 먼저 이미 존재하는 7-branch margin의 standard화 상관 행렬과 고유값 스펙트럼을 계산해, 중복이 `BM·QA·DS` 같은 위치 계열에 집중되어 있는지, 아니면 전체 margin family에 걸쳐 있는지 본다. 이후 `CA-09` 경계 밖으로 명시된 FC 후보, 즉 사전 고정된 여러 초점 강도의 bag mean 차분을 쓰는 closed-form branch를 `SCREEN_ONLY`로 넣는다. 초점 강도 수는 Primary 7 성능으로 고르지 않는다. 8-branch 기준 `max |r| <= 0.6`이고 eff.rank가 `4.02` 이상이면 gate ②와 paired 성능 비교로 넘긴다. 실패하면 성능을 조회하지 않고 후보 기각으로 종료한다. 기존 branch 제거는 제안하지 않는다 — branch별 기여도는 문서에 없다.
- **바꾸는 전제**: “유효 랭크 `3.52/7`만으로 branch 추가 방향이 막혔는지 판단할 수 있다”는 전제를 뒤집는다. margin effective rank는 readout·branch 설계가 정보를 압축한 결과이지, raw descriptor에 아직 읽히지 않는 axis가 있는지를 직접 말해주지 않는다. 따라서 “branch 추가가 막혀 있다”와 “서로 다른 것을 재는 branch를 아직 못 만들었다”는 라벨 무관 직교성 관측으로 분리해야 하는 별개 가설이다.
- **기대하는 차이**: AUROC 개선은 정량 예측 불가. 기대되는 것은 판별 관측이다. FC가 gate ①을 통과하면, 적어도 FC 계열에서는 “서로 다른 것을 재는 branch를 아직 못 만들었다”는 해석에 무게가 실린다. FC가 `max |r| > 0.6` 또는 8-branch eff.rank `4.02` 미만이면, FC 계열에서는 후보 기각을 얻지만, branch 공간 전체의 상한을 증명하지는 못한다. margin eigen audit는 중복이 특정 family에 집중되어 있는지 구분하는 보조 신호다.
- **기전**: FC는 초점 강도별 bag mean의 차분을 읽으므로, 단순 mean·quantile·salience·spectrum이 직접 읽지 않는 scale derivative를 읽을 수 있다. 이 차분이 기존 7-branch margin과 직교하면 trimmed mean에 독립적 증거를 추가할 수 있다. 그러나 이것이 slide-level task signal을 실제로 분리해서 AUROC를 올리는지는 `기전 미확인`이다.
- **실패 조건**: `D-050`으로 `SCREEN_ONLY`/provenance 경로가 복구되지 않으면 gate ① 결과가 `실행 무효`다. margin이 저장되어 있지 않고 full Primary 7 적합이 4 GPU-hour를 초과하면 `실행 무효`다. FC margin 계산 비용이 fold 적합과 유사하거나 더 커서 gate를 완료하지 못하면 `실행 무효`다. FC가 `CA-09` 경계에서 다시 저촉으로 판정되면 제안 무효다. gate ①을 통과해도 paired Δ가 `+0.0030`에 미달하면, 구간이 0을 포함하면 `판별 불가`이고, 사전 고정 비교에서 명확한 음성이면 후보 반박이다. gate ① 실패는 FC 후보 기각이지 “branch 추가 축 전체가 막혀 있다”는 반박이 아니다.

가설: common-draw pairing이 paired Δ의 SE를 실제로 줄이려면, 같은 fold·같은 seed 재실행의 deterministic residual이 seed std보다 작아야 하므로, 4 GPU-hour의 첫 실측은 fold time보다 nondeterminism residual이 더 먼저 나와야 한다.  
가설: context-only rank/quantile mapping은 AUROC의 rank 불변성을 이용해 branch scale을 제거하지만, context와 query의 margin 분포 이대가 크면 query slide의 quantile 자체가 노이즈가 되어 ensemble ordering을 망가뜨릴 수 있다.  
가설: margin eigen audit에서 중복이 BM·QA·DS 위치 계열에 집중되어 있으면, FC보다 먼저 위치 계열 readout의 rank를 바꾸는 것이 더 싼 판별일 수 있으나, CA-09/CA-10 경계와 충돌할 수 있어 경계 판정이 선행된다.