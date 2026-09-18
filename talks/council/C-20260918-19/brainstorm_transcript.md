## B1 (턴 1)
3.52는 “branch 추가가 막혔다”기보다 지금 7개가 같은 축을 중복해서 재고 있다는 말이야. 가르는 관측은 gate①에서 label-free 후보 margin을 넣었을 때 `max|r|`과 `eff.rank` 변화지. `max|r|` 낮고 rank가 오르면 “서로 다른 것을 아직 못 만든 것”, `max|r|` 높으면 중복, 둘 다 낮아도 rank가 안 오르면 정보량 부족이야.

이견: 유효 랭크 3.52 하나로 “branch 추가 방향이 막혔다”고 읽으면 안 돼. 현재 구성의 중복만 보여.

집계 축은 CA-08 때문에 아무 함수나 Primary 7에서 골라 쓰면 선택 편향이니까, 비저촉은 예측분산·컨텍스트 기하 같은 새 정보로 가중하거나 fold-pair Δ 중심의 고정 규칙이야. 아무도 안 건드린 축은 fit-level 재적합 변동성 자체를 신뢰도 신호로 쓰는 거야. CA-11은 draw ICC라 다르고, 이건 seed 재적합 분산이 +0.0030 승격 임계와 같은 크기이므로 Δ의 SE를 줄이는 축이지 mean을 직접 올리는 축은 아니야. 4 GPU-hour 내 판별은 fold 적합 시간을 한 번도 안 잰 상태에서는 못 말하고, 먼저 provenance와 1-fold 타이밍이 필요해.

가설: seed 재적합에서 변동이 큰 branch/slide margin을 closed-form으로 하향 가중하면 trimmed mean의 잡음은 줄고 평균 성능은 거의 안 바뀔 것이다.
가설: 현재 7-branch의 중복은 위치/평균 계열에 갇혀 있어서, 슬라이드 내 이질성의 2차 기하를 label-free margin으로 만들면 `eff.rank`가 오를 것이다.
가설: 집계 함수를 성능 조회 전에 fold 간 변동성만 보는 규칙으로 고정하면 CA-08의 선택 편향을 피하면서 trimmed mean보다 작은 Δ를 더 깨끗하게 재는 것이 가능할 것이다.

## B2 (턴 1)
이견: 유효 랭크 3.52를 “branch 추가가 막혔다”로 바로 읽는 건 아직 이르다.
그 방향이라면 rank 3.52는 현재 7개 margin의 중복이지, 새 margin을 못 만든다는 증명은 아니야. 가르는 관측은 구조적으로 다른 margin 하나를 넣고 게이트 ①에서 max |r|이 0.6 이하이고, 8-branch 기준 eff.rank가 4.02 이상인지 보는 거야. 올라가면 “아직 못 만들었다” 쪽, 구조가 달라도 계속 0.6 초과면 표현공간/readout이 좁은 쪽.
집계는 rank 평균, branch별 fold 내 표준화 후 trimmed mean, seed 평균화, 사전 규칙으로 절사 폭을 바꾸는 식의 learning-free 방법이 있어. CA-08이 막은 것은 Primary 7에서 고르고 같은 자료로 확증한 선택 편향이지, 방법 일반이 아니야.
분산 감소는 mean을 안 올리고 +0.0030을 판별 가능하게 만드는 투자야. seed 평균화나 다중 random start의 closed-form 해 평균은 진짜 분산 감소고, seed 하나 고정은 불확실성을 숨기는 거고.
건드리지 않은 축은 fit variance를 estimand로 다루는 seed-ensemble inference야. 건드리지 않은 이유는 과거 zero-param 결정론 정체성이지, 방법이 불가능해서가 아니야.
4 GPU-hour 안 판별은 fold fit time이 미측정이므로, 먼저 1-fold timing probe와 provenance 복구 없이는 full 50-fold 비교를 약속할 수 없어. saved margin이 구성 귀속 안 되니까 재집계도 아직 근거로 못 써.
가설: seed ensemble averaging은 성능 mean을 바꾸지 않더라도 승격 판정의 noise floor를 낮추는 데 먼저 쓸 가치가 있다.
가설: trimmed mean을 raw margin이 아니라 fold 내 branch rank로 바꾸면 branch 중복의 영향을 줄이고 CA-08 경계 밖으로 갈 수 있다.

## B1 (턴 2)
그 방향이면 3.52를 가르는 건 max |r| 하나보다 기존 7 margin으로 회귀한 잔차의 유효 랭크야. 구조가 다른 margin이 max |r|은 낮고 잔차 랭크를 올리면 아직 못 만든 쪽이고, max |r|은 낮아도 잔차가 안 오르면 정보량·readout 쪽이야. 4 GPU-hour도 fold 1회 시간이 미측정이면 50-fold 비교가 아니라 1-fold timing과 provenance 계측만 해.

이견: branch별 fold 내 표준화나 rank 평균을 Primary 7에서 골라 쓰면 CA-08 선택 편향으로 빠질 수 있어.

가설: 기존 7 margin으로 회귀한 잔차의 유효 랭크가 올라가는 label-free margin이 있으면, 3.52는 구조 상한이 아니라 아직 못 만든 중복이다.
가설: seed 평균화는 seed 주효과는 줄여도 fold×seed 잔차까지는 못 줄이므로, 분산 감소가 승격 판정에 쓰이려면 먼저 이 둘을 나눠야 한다.

## B2 (턴 2)
rank 관측으로 가르는 건 동의. 다만 3.52는 readout나 patch feature 상한이 아니라 현재 7개 margin의 중복이지, 새 margin이 max|r| 낮아도 정보량이 없으면 rank가 안 올라. 그래서 가르는 관측은 max|r|, eff.rank, 그리고 label-free margin의 fold dispersion 셋이야. 집계도 CA-08이 막는 게 방법이 아니라 Primary 7에서 고르고 같은 자료로 확증하는 절차니까, seed 평균·fold-wise rank·절사 폭을 성능 조회 전에 고정해야 해. 건드리지 않은 축은 fit variance를 estimand로 다루는 seed-ensemble inference고, 이건 과거 zero-param 결정론 때문에 생각 안 난 쪽이야. 4h는 fold fit time 미측정이니까 50-fold 승격 비교를 약속할 수 없고, 먼저 1-fold timing과 provenance smoke가 선행돼.

이견: seed 재적합 분산으로 branch/slide를 하향 가중하면 분산 감소가 아니라 Primary 7에서 본 변동성에 적응한 모델 변경이 될 수 있어. 승격 후보로 쓰려면 사전 규칙·적응 반영이 없으면 안 되고, 분산 감소는 seed 평균화처럼 관찰된 성능을 보정값으로 쓰지 않는 절차로 분리해야 해.

가설: 7-branch의 중복이 margin 스케일에서 온 것이라면, fold-wise rank 변환보다 seed ensemble 평균을 먼저 하면 trimmed mean Δ의 부호 안정성이 더 먼저 올라갈 것이다.
가설: fit variance 축은 성능 mean을 안 올리더라도, +0.0030 승격 임계 근처의 후보를 '판별 불가'에서 '지지/반박'으로 옮기는 데 먼저 쓸 가치가 있다.
가설: fold fit time이 4h 안에 7 과제 × 50 fold를 못 채울 정도로 크다면, untouched axis 탐색보다 provenance 복구와 1-fold timing이 실질적으로 유일한 판별 가능한 실험이다.

## B1 (턴 3)
그럼 4h의 첫 산출물은 50-fold 승격 비교가 아니라 소수 fold timing과 provenance hash야. fold 시간 미측정이면 그 전에는 어떤 후보도 ‘판별 가능’으로 못 올리고, 350 fold 재적합도 약속할 수 없지.

이견: fold fit time이 크다는 가정만으로 untouched axis 탐색을 미루면, 시간 측정만으로는 margin 재계산·provenance 귀속이 실제로 가능한지 안 보이므로 timing과 smoke를 묶어야 해.

가설: 1-fold timing이 4h 안에 350 fold 재적합을 허용하면 seed ensemble inference가 가장 먼저 판별 가능하고, 허용하지 않으면 provenance 복구와 margin 재계산 smoke만 실험이다.
가설: 기존 7 margin의 중복이 스케일에 온 것이라면, fold-wise rank 변환은 seed 평균화보다 seed 재적합 분산을 더 빨리 줄일 수 있다.

## B2 (턴 3)
잔차 랭크로 가르는 게 max|r| 하나보다 직접적이야. fold dispersion을 셋째로 둔 것도 동의 — rank에 안 보이는 fold 간 흔들림이 있으니까.

이견: seed 평균화를 "진짜 분산 감소"로 부르기 전에 fold×seed 잔차 23.9%를 먼저 봐야 해. RU-97에서 이게 seed 주효과보다 컸어. seed 평균은 주효과만 줄이고 상호작용은 안 줄여. 근데 closed-form 모델에서 seed가 뭔가를 바꾸고 있다는 거 자체가 아직 안 알려진 게 있고, 그 원인을 찾는 게 seed 평균보다 앞선 진단이야.

집계는 CA-08이 "함수 변경"을 막는 게 아니라 "Primary 7에서 고르고 같은 자료로 확증"하는 절차를 막는 거야. 그래서 구조적 근거로 사전 고정하면 비저촉 — "절사 1은 임의니까 0·2를 비교"나 "branch scale이 다르니까 rank 변환"은 설계 선택이지 탐색이 아니야. 근데 셋 다 돌려보고 좋은 걸 고르면 여전히 편향이야.

4h는 fold 1회 미측정이면 뭐든 약속 불가. closed-form이면 fit은 선형 해고 느린 건 feature 추출일 거야. pre-extracted면 350-fold가 4h 안에 들어올 수 있고, GPU에서 뽑아야 하면 빠듯해. 1-fold timing이 이걸 가려.

가설: closed-form에서 seed 분산이 존재한다는 건 파이프라인에 불필요한 확률 요소가 있다는 뜻이고, 그것을 식별·제거하면 seed 평균보다 깨끗한 분산 감소가 된다.
가설: fold×seed 상호작용이 fold-specific PCA 기저의 seed 의존성에서 온다면, 기저 정렬(Procrustes)이 seed 평균보다 직접적으로 잔차를 줄인다.
가설: 절사 폭을 구조적 근거로 사전 고정하고 비교하는 것은 CA-08 경계 밖이며, 이 비교 자체가 Δ의 SE를 낮추는 계측 개선이지 모델 변경이 아니다.