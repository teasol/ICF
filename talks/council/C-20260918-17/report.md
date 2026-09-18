# ICF Council — C-20260918-17

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
- **호출**: 19/26 · **소요**: 1503.3초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 편성 근거 (결과 전 기록)

탐색 질문이므로 제안(P) 둘을 온도 0.7·0.8로 두어 서로 다른 방향을 뽑는다. 이론(T)은 질문 1의 유효 랭크 해석을, 실측(M)은 질문 3·5의 분산과 비용을 맡는다. 반증(R)은 '새 축이 있다'는 낙관과 '다 막혔다'는 비관을 둘 다 공격한다. 감사(A)는 제안이 과거의 무효한 비교 결과에 기대고 있지 않은지 본다. 앞선 회차 보고서는 입력에 넣지 않았다(회차 간 앵커링 금지).

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: 4 GPU-hour feasibility · effective-rank gate
- **바꾸는 것**: "모델 구조를 먼저 바꾼다"를 "먼저 1-fold fit time과 provenance(branch list·가중치·코드 해시·fold manifest·seed/draw)를 실측하고, label-free로 7개 branch margin과 raw slide descriptor의 effective rank·고유벡터 중복을 비교한 뒤, 그 결과로 다음 축을 사전에 정한다"로 바꾼다. raw rank가 margin rank보다 높으면 slide-level compression/whitening/readout 손실 후보를, raw rank가 margin rank 이하이면 branch 추가를 중단하고 aggregation·fit-variance 축으로 이동한다.
- **바꾸는 전제**: fold fit time이 미측정이더라도 4 GPU-hour 안에 baseline+후보 비교가 가능하고, effective rank `3.52/7`만으로 "branch 추가가 막혔다" 또는 "서로 다른 branch가 아직 없다"를 판단할 수 있다고 두고 있다.
- **기대하는 차이**: 직접 AUROC 변화 없음. (a) fold time·provenance gate는 full paired Δ가 가능하면 지지, 불가능하면 실행 무효를 구분한다. (b) rank audit는 effective rank `3.52/7`을 측정 중복/투영 손실과 정보 부재로 가른다. 정량 예측 불가. 모든 판정은 Primary 7 기반이고 hold-out 미검증이다.
- **기전**: 예산은 fold 1회 시간과 재적합 수에 비례하고, provenance가 없으면 여섯 러너가 같은 하나의 설정으로 수렴했던 실행 무효가 반복된다. margin rank는 raw descriptor를 closed-form ridge·집계로 투영한 결과이므로, raw rank와 margin rank의 상대 크기만으로는 중복/부재를 좁힐 수 있다. 그러나 raw rank가 높다고 compression 변경이 AUROC를 올리는 것은 기전 미확인이다.
- **실패 조건**: fold 1회 시간이 4 GPU-hour 안에 baseline+후보 재적합을 허용하지 않거나, raw descriptor 재계산·provenance 저장이 불가능하면 실행 무효다. rank가 scale·차원·과제 난이도·표본 수 차이로 왜곡되면 판별 불가다. raw rank > margin rank여도 compression/readout 변경이 paired Δ `+0.0030`을 내지 못하면 반박이다.

## 제안 2: fit-averaged paired Δ (same-seed pairing)
- **바꾸는 것**: 승격 비교의 추정 대상을 단일 적합의 Δ_macro에서 fold×seed 분포의 기대 Δ로 바꾼다. baseline과 후보를 같은 fold·같은 seed·같은 random draw(있는 경우)로 짝짓고, R개 seed의 fit-averaged fold별 AUROC 또는 margin으로 paired Δ를 계산한다. R은 제안 1의 1-fold timing probe와 4 GPU-hour 예산으로 사전 결정한다. best-seed 선택은 금지한다.
- **바꾸는 전제**: 같은 fold만 짝지으면 fit 변동이 충분히 작고, 단일 적합의 seed std `0.0021~0.0083`은 승격 임계 `+0.0030`보다 작거나 무시 가능하다고 두고 있다.
- **기대하는 차이**: 성능 자체를 올리는 것이 아니라 Δ의 SE를 줄인다. seed main effect가 상쇄되고, 독립 fit noise는 평균으로 감소한다. RU-97 `m32_raw`에서 seed 주효과가 `0~2.9%`, fold×seed 잔차가 `23.9%`였으므로 seed 평균만으로는 상한이 낮을 수 있다. 이는 RU-97 구성 공간의 실측이며 현재 7-branch의 공통 상한은 아니다. 정량 예측 불가.
- **기전**: same-seed pairing은 seed main을 차이에 상쇄하고, R 평균은 fit noise를 줄인다. fold×seed residual은 재적합이 fold 순서·random draw와 상호작용하는 성분이므로 같은 seed 짝짓기로 완전히 사라지지 않고 평균으로만 감소한다. 이 경위는 RU-97의 분산분해와 일치하지만, 현재 7-branch에서 동일하게 성립하는지는 미확인이다.
- **실패 조건**: fold 1회 시간이 4 GPU-hour 안에 baseline+후보 R-seed 350-fold를 허용하지 않으면 실행 무효다. fold×seed residual이 지배적이거나 R을 늘려도 Δ의 SE가 `+0.0030` 판별에 필요한 수준으로 안 떨어지면 판별 불가다. 이 제안은 기대 AUROC를 바꾸지 않으므로, true Δ가 임계보다 훨씬 크다면 불필요한 투자일 수 있다.

## 제안 3: per-fold rank aggregation
- **바꾸는 것**: trimmed mean voting의 입력을 raw branch margin에서, fold·task 내 query slide rank(또는 quantile)로 변환한 margin으로 바꾼다. branch 수는 7 그대로이고 학습 파라미터를 도입하지 않는다. CA-08 경계를 벗어나려면 단일 사전 등록 변형으로 한정하고, 부분집합 탐색이나 Primary 7에서 여러 aggregation 변형을 보고 최댓값을 확증으로 쓰는 절차를 쓰지 않는다.
- **바꾸는 전제**: branch margin의 raw scale이 서로 비교 가능하고, trimmed mean이 scale 불일치를 다루지 않아도 된다고 두고 있다.
- **기대하는 차이**: branch 간 scale 불일치가 aggregate noise의 원인이라면 Primary 7 paired Δ가 개선되거나, fold별 aggregate score의 변동이 줄어들 수 있다. 단독 branch AUROC는 rank 변환으로 바뀌지 않지만, 앙상블 점수의 ranking은 바뀐다. 정량 예측 불가. hold-out 미검증이다.
- **기전**: AUROC는 rank 불변이므로 branch별 monotone scale 변환은 단독 순위를 보존한다. 그러나 raw margin 평균에서는 scale이 큰 branch가 점수를 지배할 수 있고, rank 변환은 branch 간 가중을 균등화한다. scale 자체가 신호라면 rank 변환이 정보를 버린다. 이 인과는 기전 미확인이다.
- **실패 조건**: branch margin의 scale·분포가 이미 유사하거나, scale에 유용한 magnitude 정보가 있으면 이득이 없다. fold 내 query slide 수가 rank를 안정적으로 만들지 못하면 판별 불가다. 여러 aggregation 변형을 Primary 7에서 탐색한 뒤 최댓값을 승격 근거로 쓰면 CA-08 선택 편향으로 실행 무효다. rank 변환이 paired Δ `+0.0030`을 내지 못하면 반박이다.

### P2 (제안)

## 제안 1: `common-random-number paired-Δ probe`
- **바꾸는 것**: 승격 비교의 기본 실행 단위를 “단일 적합 1회의 Primary 7 Δ”에서 “baseline과 후보를 같은 fold·같은 seed/draw·같은 fold 처리 순서로 짝짓고, 4 GPU-hour에서 가능한 최대 R로 seed 평균 paired Δ를 산출하는 것”으로 바꾼다. 첫 실행은 1-fold timing probe이며, 그 probe가 허용하는 범위에서만 paired variance probe를 진행한다. 모든 실행은 branch list·가중치·코드 해시·fold manifest를 provenance로 저장해야 하며, SEAL 10은 열지 않는다.
- **바꾸는 전제**: 단일 적합의 Primary 7 차이를 신호로 볼 수 있다는 전제를 뒤집는다. 동일 구성에서 seed만 바꿔도 표준편차 `0.0021~0.0083`이 나오고 승격 임계는 `+0.0030`이므로, 단일 적합 Δ는 대부분 뽑기일 수 있다.
- **기대하는 차이**: Δ_macro의 기댓값 자체를 올리지는 않는다. 기대되는 차이는 paired Δ의 SE 감소와 `+0.0030` 판정의 신뢰도 증가다. fold 적합 시간, 가능한 seed 수 R, fold×seed 잔차가 문서에 없으므로 정량 예측 불가.
- **기전**: common random numbers는 fold 효과와 seed 주효과를 paired Δ에서 상쇄하고, seed 평균은 잔차 분산을 통계적으로 축소하는 경로다. RU-97의 `m32_raw` 분산분해에서 fold×seed 잔차가 `23.9%`로 seed 주효과보다 크다는 실측은 seed 평균의 상한이 낮을 수 있음을 시사하지만, 현재 7-branch에 그대로 전이되지는 않으므로 기전 미확인.
- **실패 조건**: 1-fold timing probe가 4 GPU-hour에서 R=1만 허용하면 적합 분산을 추정할 수 없어 판별 불가. R>1이어도 seed 평균 후 paired Δ의 잔차 불확실성이 `+0.0030`보다 크면 판별 불가. arm routing·provenance가 복구되지 않은 채 실행하면 실행 무효. 이 제안은 성능 개선이 아니라 측정 정밀도와 판정 유효성 개선이다.

## 제안 2: `context-quantile trimmed mean`
- **바꾸는 것**: 집계만 바꾼다. 각 branch의 query slide margin을, 같은 fold의 context slide margin 경험분포(라벨 미사용)로 quantile 매핑한 뒤, 기존 trimmed mean을 그대로 적용한다. 학습 파라미터는 0개다. 여러 집계 변형을 탐색해 최댓값을 고르지 않고, 이 단일 규칙만 결과 조회 전에 고정한다. 저장 margin의 provenance가 없으면 먼저 verified margin 추출을 해야 한다.
- **바꾸는 전제**: 7개 branch margin이 절대 scale에서 직접 비교되어도 된다는 전제를 뒤집는다. branch별 scale·분포 차이가 min/max 절사나 평균에서 slide 증거가 아니라 scale 편향을 제거하게 만들 수 있다.
- **기대하는 차이**: branch scale 불일치가 ensemble ordering을 깨고 있다면, quantile 매핑은 branch별 분포 차이를 제거하고 순서 정보만 유지하므로 aggregate noise를 줄여 paired Δ_macro가 `+0.0030`을 넘을 수 있다. 그러나 효과 크기는 문서에 근거가 없어 정량 예측 불가.
- **기전**: quantile mapping은 branch별 monotonic transform이므로 단독 branch AUROC는 rank 불변성 때문에 바뀌지 않지만, branch 간 상대 scale이 바뀌어 최종 ensemble ordering은 바뀐다. margin의 크기가 이미 잘 보정된 evidence라면 이 변환은 정보 손실이다. 실제 경로와 이득은 기전 미확인.
- **실패 조건**: context-query 분포 이대, context slide 수 `90~261`에서의 quantile 추정 노이즈, magnitude 정보 손실로 paired Δ가 `+0.0030`에 미달하거나 악화하면 반박. 여러 집계 변형을 Primary 7에서 비교한 뒤 최댓값을 확증으로 쓰면 CA-08 선택 편향에 해당하므로 실행 무효. provenance가 없으면 어떤 Δ도 구성에 귀속시킬 수 없어 실행 무효.

## 제안 3: `raw-descriptor rank audit + FC gate`
- **바꾸는 것**: branch 추가 판단 기준을 “branch margin 유효 랭크 `3.52/7`이 낮으므로 branch를 더 넣자”에서 “branch 생성 전 raw slide descriptor의 label-free effective rank와 7 branch margin effective rank의 gap을 먼저 재자”로 바꾼다. raw rank > margin rank이면 `FC`를 후보로 gate ①에 올리고, raw rank ≤ margin rank이면 branch 추가 방향을 중단한다. `FC`는 `CA-09`에서 비저촉으로 명시된 후보이며, gate ①에서 `max |r| > 0.6` 또는 eff.rank 효율 하락이면 성능 조회 없이 종료한다.
- **바꾸는 전제**: 유효 랭크 `3.52/7`은 이미 “branch를 더 넣으면 늘 수 있다”는 전제를 뒤집는다. 중복이 branch 생성 부족 때문인지, slide-level patch feature의 upstream 압축·공통 구조 때문인지 구분하지 않으면 branch 추가는 중복만 늘릴 수 있다.
- **기대하는 차이**: 직접 성능 예측은 하지 않는다. raw rank가 margin rank보다 높고 `FC`가 gate ①을 통과하면, 8-branch에서 기존 효율 `50.3%` 비하락을 위해 eff.rank `≥ 4.02`를 만족할 가능성이 열리며, 정보량 증가 경로가 열린다. raw rank가 margin rank를 넘지 못하면 branch 추가 방향은 막힌 것으로 판정된다. 성능 개선량은 정량 예측 불가.
- **기전**: branch margin들이 raw descriptor의 중복 사영에 머물면 margin effective rank가 낮아진다. 반면 raw descriptor에는 아직 읽히지 않는 방향이 있으면, 구조적으로 다른 branch가 직교 정보를 제공할 수 있다. 어느 쪽인지, 그리고 `FC`가 실제로 그 정보를 읽는지 기전 미확인.
- **실패 조건**: raw feature/margin 저장 경로가 없거나 4 GPU-hour에서 재계산·gate ①이 불가하면 실행 무효. effective rank 추정이 slide 수·정의 선택에 민감해 raw/margin gap을 판별하지 못하면 판별 불가. `FC`가 gate ①에서 `max |r| > 0.6`이나 eff.rank 효율 하락으로 기각되면 branch 추가 후보는 반박. raw rank가 낮으면 병목이 upstream에 있으나, 이 제안 안에서 4 GPU-hour 모델 개선으로 전환되지 못한다.

### T1 (이론)

## 이론적 병목 N1
- **가정**: 유효 랭크 3.52/7은 branch margin이 서로 다른 slide-level 정보를 충분히 만들지 못했다는 branch 생성 병목을 뜻하며, 같은 patch feature에서 직교 margin을 추가하면 독립 증거가 늘어난다고 두고 있다.
- **이 가정이 깨지는 조건**: margin rank를 제한하는 것이 branch 정의가 아니라 upstream slide-level representation(PCA-32, whitening, bag mean/salience aggregation, patch feature 압축)의 정보량일 때. raw descriptor 또는 slide-level compressed feature의 유효 자유도가 margin rank보다 낮거나, 구조적으로 다른 후보도 기존 margin과 높은 상관을 보이면 깨진다.
- **깨질 때 나타나는 관측 가능한 증상**: (a) raw descriptor rank audit에서 raw rank가 margin rank보다 높지 않음; (b) FC·LID·AKS·LSAK 등 구조가 다른 후보가 gate ①에서 max |r| > 0.6 또는 eff.rank ≥ 4.02 미달로 탈락; (c) 위치 계열(BM·QA·DS)을 바꾸지 않고 shape 계열만 추가해도 eff.rank가 오르지 않음; (d) upstream compression/whitening 변경이 branch 추가보다 margin 분산 구조를 더 크게 바꿈. (a)와 (b)가 함께 나타나면 upstream 정보량 병목으로, raw rank가 높고 새 branch가 직교하면 branch 생성 병목으로 판별한다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: CA-09는 BM·QA·DS 위치 계열의 유효 랭크 1.29/3과 상호 상관 0.25~0.93을 기록하고, RM은 CV가 이미 꼬리 부분공간을 읽고 있어 기각됐다. CA-10은 형상 변형 7종이 게이트 ②를 통과하지 못하고 병목이 절사가 아니라 정보량이라고 결론지었다. 그러나 raw descriptor rank와 branch별 기여도 실측은 문서에 없어 최종 판별은 불가하고, upstream 병목 기전은 미확인이다.

## 이론적 병목 N2
- **가정**: trimmed mean voting은 7개 branch margin이 같은 scale·같은 오류 구조를 갖고, 중앙값 평균이 branch 간 결합에 적절하다고 두고 있다.
- **이 가정이 깨지는 조건**: branch margin의 scale, fold 내 노이즈, task-specific ordering, context/query 분포 이대가 branch마다 다르고, AUROC의 rank 불변성 때문에 margin 절대값보다 순서가 더 중요할 때. trimmed mean이 scale 불일치나 상관 오류를 중화하지 못하면 깨진다.
- **깨질 때 나타나는 관측 가능한 증상**: 같은 fold margin에서 (a) branch별 rank/quantile 정규화, (b) median·mean·다른 trim, (c) prediction-variance 가중(PSW), (d) context signal geometry 가중(TGW)이 trimmed mean과 다른 paired Δ 또는 sign agreement를 내면 가정 위반이다. 차가 fit variance보다 작으면 판별 불가이고, SE만 줄면 variance reduction이며, expected Δ가 바뀌어야 성능 개선이다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: 해당 관측 없음.

## 이론적 병목 N3
- **가정**: 승격 판정은 Primary 7의 fold pairing과 7 과제 동등 가중 macro Δ만으로 +0.0030을 안정적으로 측정할 수 있고, 적합 분산과 과제 이질성은 부수적 nuisance라고 두고 있다.
- **이 가정이 깨지는 조건**: 동일 구성의 seed refit 분산이 승격 임계와 같은 크기이거나, fold×seed 잔차가 seed 평균으로 상쇄되지 않거나, 7 과제가 exchangeable하지 않아 macro Δ가 과제별 악화를 감출 때. D-048 이후 적합 분산 보고가 필수인데 승인된 절차가 없으면 깨진다.
- **깨질 때 나타나는 관측 가능한 증상**: (a) seed만 바꿨을 때 Δ의 부호가 바뀌거나 CI가 0을 포함; (b) best-seed 선택 시 +0.0030 초과가 반복적으로 사라짐; (c) fold×seed 잔차가 seed 주효과보다 커서 fold 수만 늘려도 SE가 줄지 않음; seed averaging은 seed 주효과가 fold×seed 잔차보다 지배적일 때만 유효하다. (d) common-seed pairing, seed averaging, seed 의존성을 줄이는 closed-form 안정화가 mean을 거의 바꾸지 않고 SE만 줄이면 variance reduction으로 판정한다. 현재 점추정이 0을 포함해 판별 불가라면 variance reduction은 mean을 추격하는 것보다 우선할 수 있으나, true mean이 임계 아래라면 실패를 더 분명하게 드러낼 뿐이다. (e) sign agreement가 낮고 특정 과제가 같은 방향으로 크게 악화되면 macro Δ는 과제 이질성을 감춘다. 성능 개선은 expected Δ를 바꾸는 것, variance reduction은 Δ_hat의 분산·CI 폭만 줄이는 것이다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: RU-97에서 동일 구성 seed refit의 sd 0.0021~0.0083이 승격 임계 +0.0030과 같은 크기이고, M=32−M=8은 +0.0003 [-0.0067, +0.0073]로 0을 포함한다. fold×seed 잔차가 23.9%로 seed 주효과보다 크고, 적합 분산을 포함한 승격 판정 절차는 미승인이다. D-050으로 unverified-hold인 공식 7-branch 승격 근거 +0.56%p의 과제 군집 95% 구간은 [-1.15%p, +2.28%p]이고 sign agreement는 4/7이며, Histologic_Grade -1.08%p, progression_regression -1.13%p, PBRM1 -1.44%p가 악화됐다. 이 관측들은 적합 분산과 과제 이질성이 임계와 같은 규모일 수 있음을 보여주지, 기전 입증은 아니다.

## 이론적 병목 N4
- **가정**: "아직 건드리지 않은 축"은 closed_axes에 없으므로 유효한 연구 대상이고, 건드리지 않은 이유는 주로 생각이 안 난 것이라고 두고 있다.
- **이 가정이 깨지는 조건**: 해당 축이 실제로는 경계 안(oracle, 선택 편향, hold-out 사용, 구조적 무의미, 비용 대비 무이득)에 걸리거나, 계측·provenance·게이트 ①이 무효라서 실행 자체가 무효일 때. 또는 4 GPU-hour에서 판별이 불가능할 때.
- **깨질 때 나타나는 관측 가능한 증상**: (a) 제안이 CA-08의 선택 편향, CA-09/CA-10의 평균·형상 변형, CA-11의 ICC 구조, CA-13의 retrieval 비용, CA-05의 염색 교란, CA-06의 라벨 기저 붕괴 중 하나와 동일 기전을 재현; (b) D-050으로 provenance·오염 검사·gate ①이 복구되기 전에는 실행이 verified로 불가능; (c) fold 1회 시간이 미측정이라 full 350-fold가 예산을 초과할 수 있어 실행 무효. 건드리지 않은 축 중 fit variance reduction/common-seed pairing, raw descriptor rank audit, PSW/TGW aggregation은 닫힌 축이 아니라 계측·예산·선택 이력 문제로 보류된 축이다. 학습 파라미터 readout은 D-048으로 CA-14가 폐기됐지만, CA-14의 과거 관측이 남아 있어 새 증거가 필요한 축이다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: closed_axes.md가 CA-08·CA-09·CA-10·CA-11·CA-13·CA-05·CA-06의 경계와 기각 기전을 명시하고, CA-14의 폐기와 과거 관측을 기록한다. PROJECT.md §3.4/§5와 D-050이 provenance·오염 검사·gate ①의 무효를 기록하며, current_status.md가 fold 적합 시간 미측정과 4 GPU-hour 예산, 적합 분산 승격 절차 미승인을 기록한다.

## 이론적 병목 N5
- **가정**: 4 GPU-hour 안에서 baseline과 후보의 350-fold paired Δ, 적합 분산, 그리고 필요 시 gate ①까지 판별 가능하다고 두고 있다.
- **이 가정이 깨지는 조건**: fold 1회 적합 시간이 미측정이고, 350-fold × (baseline+후보) × seed 수를 4 GPU-hour에 넣지 못할 때. 또는 provenance가 없어 실행을 구성에 귀속시킬 수 없고, 승격 절차가 미승인일 때.
- **깨질 때 나타나는 관측 가능한 증상**: (a) 1-fold timing probe가 예산을 초과하거나 fold×seed 재적합을 허용하지 못함; (b) 저장 예측을 baseline/후보 구성에 귀속시킬 수 없어 재집계만 가능; (c) full 350-fold paired Δ를 못 내고 subset·screening·re-aggregation로 끝나면 승격 판정은 실행 무효; (d) 4 GPU-hour 안에서 판별 가능할 수 있는 후보는 1-fold timing/provenance probe, 새로 생성한 소수 fold margin에 대한 label-free rank/correlation audit, 같은 margin에서의 집계 변형 재집계, provenance 복구 후 gate ① screening뿐이다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: fold 적합 1회 시간은 측정된 적이 없고, 예산은 GPU 1장 합계 4 GPU-hour이며, D-050은 저장 예측 파일에 branch list·가중치·코드 해시가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다고 기록한다. 적합 분산을 포함한 승격 판정 절차는 미승인이고, 모든 후보 판정은 hold-out 미검증이다.

## 이론적 병목 N6
- **가정**: Primary 7에서 +0.0030 paired Δ를 얻으면 SEAL 10에서 ABMIL을 넘을 가능성이 충분하다고 두고 있다.
- **이 가정이 깨지는 조건**: Primary 7의 task/fold 분포가 SEAL 10 hold-out의 분포와 다르고, Primary 7 개선이 hold-out의 과제 구성·난이도·데이터 특성에 전이되지 않을 때.
- **깨질 때 나타나는 관측 가능한 증상**: Primary 7에서 지지/승격이 나와도 hold-out 개봉에서 목표 미달. 그러나 SEAL 10은 선택에 사용할 수 없으므로 이 증상은 한 번의 최종 판정에서만 보이고, 그 결과를 개선 방향 선택에 쓰면 규칙 위반이다. 현재는 hold-out 미검증으로만 기록해야 한다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: PROJECT.md §1은 Primary 7의 개선이 hold-out 개선을 보장하지 않는다고 명시하고, D-047은 SEAL 10 결과를 개선 방향 선택에 쓰는 것을 금지한다. 모든 후보 판정은 hold-out 미검증이다.

### M1 (실측)

## 실측 1: 유효 랭크 3.52 해석 판별 — branch 추가 막힘 vs 서로 다른 branch 미생성
- **건드려야 하는 코드·설정**: `scripts/analysis/branch_screen.py`로 기존 7-branch margin의 `max |r|`·eff.rank를 라벨 무관으로 재계산. margin 생성은 `evaluate_pure.py` + 7-branch YAML 경로를 사용해야 하며, 7-branch YAML의 정확한 파일명은 문서에 없어 미확인. `ICF_*` 환경변수 경로는 `D-050`으로 무효이므로 YAML 경로만 사용. raw descriptor rank audit를 위해 branch readout 이전 slide descriptor를 저장하는 코드가 필요하나, 해당 경로는 문서에 없어 미확인. 선행으로 provenance writer(branch list·가중치·코드 해시·fold manifest 해시)가 필요.
- **비용**: 1-fold 파이프라인 검증의 시간·메모리는 미실측. 350-fold margin 재산출은 fold 적합 1회 시간이 미실측이므로 총 GPU-hour는 미실측. 저장된 margin을 재사용하면 CPU 재집계이지만, 현재 저장 예측 파일은 provenance가 없어 공식 7-branch에 귀속 불가. raw descriptor 저장 시 메모리는 미실측.
- **막는 것**: (1) provenance 부재로 저장 산출물을 특정 구성에 귀속시킬 수 없음. (2) fold 적합 시간 미실측으로 350-fold 재산출이 4 GPU-hour 안에 들어가는지 판별 불가. (3) raw descriptor audit 코드가 문서에 없어 구현 필요. (4) rank 비교 단위가 `branch_screen.py`의 게이트 ① 단기와 달라지면 판정 무효. (5) SEAL 10을 쓰면 선택 금지 위반.
- **중단 기준**: 1-fold probe에서 OOM·NaN·Inf·provenance 미기록이면 실행 무효로 종료. 350-fold 비용 역산이 4 GPU-hour 초과면 full audit 중단. raw descriptor eff.rank가 margin eff.rank 3.52보다 높지 않거나, 후보 branch가 `max |r| > 0.6` 또는 8-branch 기준 eff.rank `< 4.02`이면 branch 추가 방향은 이 probe에서 실패 또는 판별 불가로 기록하고 성능을 조회하지 않음. raw rank가 높고 후보가 직교하면 “측정 중복” 해석만 좁혀도 기전 입증은 아님. 결과가 혼합되면 판별 불가.

## 실측 2: 학습 파라미터 없는 집계 변경의 실측 가능성
- **건드려야 하는 코드·설정**: 집계 함수의 정확한 경로는 문서에 없어 미확인. `evaluate_pure.py`는 실행 진입점 후보지만 확정 아님. YAML로 집계 방식을 선택하는 플래그, per-branch margin 저장, provenance 기록을 추가해야 함. 후보는 사전 선언된 고정 규칙만 허용: rank/quantile normalization, closed-form ridge 기반 prediction-variance 가중(PSW), context 신호 기하 가중(TGW). 학습 파라미터는 도입하지 않음.
- **비용**: provenance-verified margin이 있으면 재집계 CPU 시간은 미실측. margin이 없으면 350-fold 재적합이 필요하고, fold 적합 시간은 미실측이므로 총 GPU-hour는 미실측. prediction-variance 계산 CPU 시간은 미실측. TGW의 구체적 사양은 문서에 없어 구현 비용 미실측.
- **막는 것**: (1) `CA-08`으로 기존 branch margin 위의 결합 함수 탐색은 닫혀 있음. Primary 7 결과를 보고 함수를 고르면 선택 편향으로 실행 무효. (2) rank/quantile 단순 교체도 `CA-08` 경계 판정이 필요. (3) provenance 부재로 per-branch margin을 구성에 귀속시킬 수 없음. (4) 적합 분산을 포함한 승격 판정 절차가 사용자 미승인. (5) PSW/TGW의 계산 단위·정규화·margin 환원이 문서에 없어 구현 전 설계 필요.
- **중단 기준**: per-branch margin과 provenance를 저장하지 못하면 중단. 후보 집계를 Primary 7 결과 조회 뒤에 선택하면 실행 무효. 경계 판정에서 `CA-08` 경계 안으로 판정되면 성능 조회 없이 종료. 350-fold 재적합 비용이 4 GPU-hour 초과로 역산되면 full 판별 중단.

## 실측 3: 적합 분산 축소의 실측 가능성과 성능 개선과의 구분
- **건드려야 하는 코드·설정**: `scripts/eval_v121.sh` 등 러너가 학습 설정을 넘기지 못하고 `eval_seal_tasks.sh` 폴백으로 같은 7-branch에 수렴하는 문제를 수정. 공통 랜덤 숫자 설계: 같은 fold, 같은 seed, 같은 patch/cell draw를 후보와 기준선에 공통 사용. seed·스레드·GMM 초기화 고정 설정은 구체적 플래그가 문서에 없어 미확인. 동일 설정 2회 실행으로 nondeterminism 잔차를 측정. 배경 GMM은 기존 3M 표본, CPU thread 8, `max_iter=100` 범위만 사용.
- **비용**: 1-fold duplicate run은 fold 적합 2회로, 시간은 미실측. R-seed 350-fold 재적합은 arm당 350R fold fit으로, 총 GPU-hour는 fold 적합 시간 미실측이므로 미실측. CPU GMM 시간은 GPU-hour 예산에 포함되지 않을 수 있으나 wall-time 병목은 미실측.
- **막는 것**: (1) fold 적합 시간 미실측으로 seed 평균 비용이 불가. (2) `RU-97`에서 seed 주효과는 전체의 0~2.9%이고 fold×seed 잔차가 23.9%로, seed 평균만으로는 fold×seed 잔차가 충분히 줄지 않을 수 있음. (3) 적합 분산을 포함한 승격 판정 절차가 미승인. (4) 러너 폴백으로 arm이 다른 실행이 같은 구성으로 수렴한 이력이 있어, 공통 짝짓기 전 실행은 구성 비교로 무효.
- **중단 기준**: 동일 설정 2회 실행의 AUROC `max|diff|`가 seed std 범위 0.0021~0.0083보다 크거나 승격 임계 +0.0030보다 크면, 공통 짝짓기·seed 평균으로 승격을 판별할 수 없어 판별 불가 또는 실행 무효로 중단. 추정 비용이 4 GPU-hour 초과면 중단. 분산 축소는 paired Δ의 SE를 줄이는 것이지 기대 Δ를 올리는 것이 아님. 점추정 Δ가 +0.0030 근처에서 불확실성에 막혀 있을 때만 성능 개선보다 나은 투자일 수 있고, 기대 Δ가 없으면 정확한 null만 남음.

## 실측 4: 건드리지 않은 축의 실측 우선순위
- **건드려야 하는 코드·설정**: (1) 1-fold timing probe: Primary 7 fold 1개를 `evaluate_pure.py` + 7-branch YAML로 실행해 wall time, GPU memory, provenance 기록. (2) provenance writer: 산출물에 branch list·가중치·코드 해시·fold manifest 해시 기록. (3) raw descriptor audit: branch readout 이전 descriptor 저장 경로 구현, 경로 미확인. (4) duplicate nondeterminism probe: 동일 설정 1-fold 2회 실행.
- **비용**: 1-fold probe의 소요 시간은 이번 probe의 산출이며 사전 미실측. GPU memory는 미실측. provenance writer 코드 변경 CPU 시간은 미실측. raw descriptor 저장 메모리·시간은 미실측.
- **막는 것**: 현재 목표가 계측 복구(provenance writer, `SH`/`SJ` control 경로, 오염 검사 재설계, 단위 테스트)이고, 복구 전 게이트·승격 비교를 `verified`로 수행할 수 없음. fold 적합 시간 미실측은 기본 실행 미지로, “아직 필요하지 않아서 안 잰 것”으로 타당하지 않음. aggregation 미탐색은 `CA-08` 선택 편향으로 타당하지만, PSW/TGW 같은 새 정보 경로는 아직 설계되지 않음. upstream slide-level compression/PCA-32/whitening 미탐색은 전체 branch baseline을 무효화하고 재적합 비용이 미실측이라 타당한 비용 차단. raw descriptor audit 미탐색은 provenance와 feature 저장 부재로 타당.
- **중단 기준**: 1-fold probe가 OOM·NaN·provenance 미기록이면 중단. provenance writer를 넣지 못하면 verified 비교 중단. raw feature를 예산 내 재산출하지 못하면 raw audit 중단. 1-fold time이 full 350-fold를 불가능하게 만들면 full 실험을 실행 무효로 기록하고 축을 바꾸어야 함.

## 실측 5: 4 GPU-hour 내 판별 가능 여부
- **건드려야 하는 코드·설정**: 먼저 1-fold timing/provenance probe. 이후 fold 적합 시간 `t_fold`를 실측해 full 350-fold two-arm 또는 candidate-only 실행을 결정. `evaluate_pure.py`로 margin 생성, `scripts/analysis/branch_screen.py`로 게이트 ①, provenance-verified margin이 있으면 재집계. SEAL 10은 열지 않음.
- **비용**: `t_fold`는 미실측. full 350-fold two-arm single-seed는 700 fold fit으로, 예산 역산 임계는 `t_fold <= 4/700 GPU-hour = 20.6 sec/fold`. 기준선 margin이 verified로 재사용 가능하면 candidate-only 350-fold는 `t_fold <= 4/350 GPU-hour = 41.1 sec/fold`. R-seed two-arm은 700R fold fit으로, R=2이면 `t_fold <= 10.3 sec/fold`. 1-task 50-fold two-arm은 100 fold fit으로 `t_fold <= 14.4 sec/fold`, candidate-only는 `t_fold <= 28.8 sec/fold`. 위 임계는 예산에서 역산한 값이지 실측이 아님. offline 재집계 CPU 시간은 미실측.
- **막는 것**: `t_fold` 미실측으로 full 승격 판별은 현재 판별 불가. 공식 기준선은 `unverified-hold`이고 provenance가 없어 baseline 재사용이 불가능할 가능성이 큼. 적합 분산 포함 승격 절차가 미승인. GPU memory OOM 가능성은 미실측.
- **중단 기준**: 1-fold `t_fold`가 41.1 sec/fold를 넘고 baseline 재사용이 불가능하면 350-fold fit 자체가 4 GPU-hour 초과로 full 실행 무효. `t_fold`가 20.6 sec/fold를 넘고 baseline 재적합이 필요하면 two-arm full 실행 무효. 1-fold `t_fold`가 임계 이하라도 duplicate nondeterminism 잔차가 +0.0030보다 크면 full 실행해도 판별 불가일 수 있어 중단 또는 재설계. 4 GPU-hour 안에서 1-fold 또는 1-task probe만 가능하면, full 승격 판별은 이 예산에서 불가능하다고 기록.

### R1 (반증)

## 반론 1
- **공격 대상**: "유효 랭크 3.52/7은 branch 추가 방향이 이미 막혔다" 또는 그 반대로 "아직 서로 다른 것을 재는 branch를 못 만들었을 뿐"을 이미 결론 지었다는 주장.
- **더 단순한 설명**: 현재 7개 margin이 같은 patch feature, slide-level aggregation, PCA-32/whitening, closed-form ridge 경로를 공유하므로 선형 중복이 클 수 있다는 설명이다. 이는 "추가 불가"가 아니라 "현재 생성된 margin들이 같은 low-rank 표현을 읽고 있다"는 더 좁은 관측이다.
- **무엇이 참이면 이 주장이 무너지는가**: 라벨 무관 게이트 ①에서 구조적으로 다른 후보 branch가 max |r| ≤ 0.6이고, 8-branch 기준 eff.rank ≥ 4.02를 유지하면 "branch 추가 방향이 막혀 있다"는 주장이 반박된다. 두 해석을 가르는 관측은 branch margin의 effective rank와 raw slide descriptor의 effective rank 비교다. raw descriptor rank가 margin rank보다 높으면 branch 생성 단계에서 차원이 붕괴했다는 단순 설명이, raw descriptor rank도 비슷하게 낮으면 upstream slide-level 표현이 이미 low-rank라는 단순 설명이 지지된다. 단, raw rank가 높다는 사실만으로는 task 정보가 있음을 증명하지 못한다.
- **이 반론의 한계**: effective rank는 선형 중복 지표일 뿐이고, nonlinear 정보량이나 AUROC를 보장하지 않는다. raw descriptor audit를 실행하려면 provenance와 재계산 경로가 복구되어야 하며, fold 적합 1회 시간은 문서에 없으므로 4 GPU-hour 내 실행 가능성은 아직 판별 불가다.

## 반론 2
- **공격 대상**: "집계(trimmed mean voting)를 바꾸는 축은 아직 거의 탐색되지 않았고, 학습 파라미터를 도입하지 않는 집계 변경은 유효한 새 축이다"라는 주장.
- **더 단순한 설명**: CA-08은 기존 branch margin 위의 결합 함수·부분집합 탐색을 선택 편향으로 이미 종료했다. 따라서 "아직 거의 탐색되지 않았다"보다 더 단순한 설명은 "과거 집계 비교는 여섯 러너가 같은 7-branch 설정으로 수렴해 구성 차이가 없었으므로 근거로 쓸 수 없다"는 것이다. 또한 BM·QA·DS 위치 계열의 유효 랭크가 1.29/3이고 상호 상관이 0.25~0.93으로 기록되어 있으므로, 집계 변경이 새 정보를 만드는 것이 아니라 중복된 margin의 가중만 바꿀 수 있다는 설명이 더 단순하다.
- **무엇이 참이면 이 주장이 무너지는가**: 사전 선언된, Primary 7에서 최댓값을 보고 고르지 않는 집계 변형이 대응 per-fold Δ ≥ +0.0030을 적합 분산과 함께 보여주면 집계 축은 유효하다. 학습 파라미터 없는 후보로는 per-fold rank/quantile normalization 후 trimmed mean, rank 기반 median/trimmed mean, CA-08 경계 밖으로 허용된 PSW·TGW 계열의 new-info 가중 등이 있다. 그러나 같은 Primary 7에서 집계 변형을 탐색하고 그 최댓값을 승격 근거로 쓰면 CA-08 경계에 저촉된다.
- **이 반론의 한계**: CA-08은 접근 일반을 영구 금지한 것이 아니라, 선택·적응을 반영한 유효한 평가 설계가 없으면 확증할 수 없다고 닫았다. 따라서 사전 설계와 선택 이력 기록이 있으면 집계 축은 다시 열 수 있다. margin scale 불일치가 실제 병목인지, 아니면 중복성 때문인지는 아직 기전 미확인이다.

## 반론 3
- **공격 대상**: "적합 분산이 0.0021~0.0083로 승격 임계 +0.0030보다 크므로, 분산을 줄이는 것이 성능을 올리는 것보다 나은 투자다"라는 주장.
- **더 단순한 설명**: 적합 분산은 모델의 기대 성능이 아니라 추정치의 불확실성이다. 더 단순한 설명은 "단일 적합의 차이는 뽑기일 수 있다"는 관측이 "분산 축소 자체가 성능 개선이다"를 증명하지 못한다는 것이다. 분산 축소는 Δ의 SE를 줄일 수 있지만, 후보의 Δ mean이 +0.0030에 도달하지 못하면 승격 기준을 넘지 못한다.
- **무엇이 참이면 이 주장이 무너지는가**: common fold·common seed pairing, seed averaging, deterministic closed-form 경로 등으로 분산을 줄인 뒤에도 고정 후보의 paired Δ mean이 +0.0030 미만이면, 분산 축소는 성능 개선이 아니라 측정 정밀도 개선에 그친다. 반대로 분산 축소 전에 +0.0030 미만이었으나 분산 축소 후 +0.0030 이상으로 보이는 경우, 그것은 모델 개선이 아니라 추정치가 임계 근처에서 흔들린 것이므로 승격 근거로 쓰기 전에 선택·적응 이력을 반영한 설계를 요구받는다. RU-97의 m32_raw에서는 fold×seed 잔차가 23.9%로 seed 주효과보다 크고 seed 주효과는 전체의 0~2.9%로 기록되어 있어, seed averaging의 상한이 낮을 수 있음을 시사한다. 그러나 이는 GF 단독 탐색의 관측이지 현재 7-branch 공식 구성의 분산분해는 아니다.
- **이 반론의 한계**: 만약 적합 분산의 원인이 mean bias를 동반하는 불안정한 random subsampling, GMM, 또는 closed-form 해의 fold별 변동이라면, 분산을 줄이는 개입이 mean 자체를 바꿀 수 있다. 현재 7-branch의 seed main, fold main, fold×seed 잔차 분해는 문서에 없으므로, "분산 축소만으로는 부족하다"는 일반화는 아직 판별 불가다.

## 반론 4
- **공격 대상**: "아직 아무도 건드리지 않은 모델 축이 있다"는 주장, 특히 집계·branch·readout 중 아직 untouched인 모델 축이 남아 있다는 전제.
- **더 단순한 설명**: 현재 상태 문서상 더 단순한 설명은 "건드리지 않은 것이 모델 축이 아니라 계측·provenance·fold timing"이라는 것이다. D-050은 여섯 실행 러너가 전부 같은 7-branch 설정으로 수렴했고, 저장된 예측 파일에 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없다고 기록한다. 공식 7-branch 기준도 unverified-hold다. 따라서 모델 축은 "생각이 안 난" 것이 아니라, 계측 복구 전에는 verified 비교를 할 수 없는 상태에 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 특정 모델 후보가 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 산출물에 기록하고, 오염 검사 재설계 기준을 통과하며, 4 GPU-hour 안에서 Primary 7 대응 비교를 완주할 수 있고, closed_axes.md의 경계 안에 들지 않는다면 untouched 모델 축이 존재한다고 볼 수 있다. 그렇지 않다면 다음 축은 모델 변경이 아니라 provenance writer, SH/SJ control 경로, 오염 검사, fold timing probe다.
- **이 반론의 한계**: CA-04의 LSAK·QJF, CA-02 DD, CA-R1, CA-09 경계 밖 FC, CA-08 경계 밖 PSW·TGW 등은 닫힌 것이 아니므로 "아무도 건드리지 않은" 모델 축이 아예 없다는 뜻은 아니다. 다만 그것들은 이미 경계 판정·제안 이력·미해결 조건이 있으므로 untouched라기보다 "열려 있으나 실행 전 조건이 있다"는 표현이 정확하다.

## 반론 5
- **공격 대상**: "4 GPU-hour 안에서 verified 승격 판별이 가능한 후보가 있다"는 주장.
- **더 단순한 설명**: fold 적합 1회 시간은 문서에 없으므로, 350-fold paired baseline+candidate 재적합이 4 GPU-hour에 들어가는지는 아직 알 수 없다. 더 단순한 설명은 "4 GPU-hour는 먼저 1-fold timing probe와 provenance smoke test에 쓰일 가능성이 크다"는 것이다. verified 승격 판별에는 같은 fold에서 baseline과 후보의 AUROC를 짝짓고, 적합 분산을 함께 보고해야 하므로, fold time이 측정되지 않으면 예산 적합성은 판별 불가다.
- **무엇이 참이면 이 주장이 무너지는가**: 1-fold fit time을 실측한 뒤, baseline 350-fold와 후보 350-fold, 재기동·기록·오염 검사 오버헤드를 포함해 4 GPU-hour 이하로 완주 가능하면 특정 후보의 verified 승격 판별은 가능하다. 반대로 이 합계가 4 GPU-hour를 초과하면, 해당 후보의 full 350-fold 승격 판별은 실행 무효가 아니라 아직 예산 초과로 판별 불가이며, 예산을 늘리거나 재집계·스크리닝으로 축을 바꿔야 한다.
- **이 반론의 한계**: 저장된 margin 또는 예측 파일을 provenance와 함께 재사용할 수 있다면 full 재적합 비용은 줄어들 수 있다. 그러나 현재 문서는 저장된 예측 파일에 provenance가 없고, 오염 검사 재설계가 필요하다고 기록하므로 재사용은 verified가 아니다. 따라서 "4 GPU-hour 안에 판별 가능"을 특정 후보에 대해 주장하려면 fold time과 provenance 재현 둘 다 먼저 실측되어야 한다.

## 반론 6
- **공격 대상**: "branch margin eigen audit만으로 '측정 중복'인지 '정보 부재'인지 방향을 좁힐 수 있다"는 주장.
- **더 단순한 설명**: eigen audit는 기존 7개 margin의 선형 중복을 재는 것이지, task 정보량을 재는 것이 아니다. 유효 랭크 3.52/7은 "측정 중복"과 "정보 부재"를 구분하지 못한다. 더 단순한 설명은 "현재 margin들이 선형적으로 중복된다"는 것뿐이며, 그것이 UNI2 patch feature의 slide-level 압축 때문인지, branch 생성자 때문인지, 아니면 task-relevant 신호 자체가 제한적이지는 eigen audit만으로는 알 수 없다.
- **무엇이 참이면 이 주장이 무너지는가**: eigen audit가 중복을 BM·QA·DS 위치 계열에 집중시켜 보여주고, 동시에 라벨 무관 게이트 ①을 통과하는 구조적으로 다른 branch가 존재하면, "정보 부재"라는 일반화는 무너진다. 반대로 raw descriptor rank가 margin rank보다 높지 않거나, 여러 구조적으로 다른 후보가 모두 max |r| > 0.6으로 게이트 ①에서 탈락하면, upstream 표현 또는 공유 slide-level 통계가 병목이라는 단순 설명이 강해진다. 그러나 이 경우에도 task 정보 부재라고 확정할 수는 없다.
- **이 반론의 한계**: eigen audit는 비용이 낮고 라벨 무관 게이트 ①과 연결될 수 있어 유용하다. 그러나 branch별 기여도는 문서에 없고, effective rank는 AUROC를 보장하지 않으므로, "방향 좁히기"는 가능하지만 "정보 부재 입증"은 아니다.

## 반론 7
- **공격 대상**: "seed 평균이 실제로 Δ SE를 줄이려면 seed 주효과가 fold×seed 잔차보다 지배적이어야 하며, RU-97의 fold×seed 23.9%는 seed 평균의 상한이 낮다"는 가설을 현재 7-branch에 바로 적용하는 주장.
- **더 단순한 설명**: RU-97의 분산분해는 GF 단독 탐색의 특정 구성에서 나온 관측이다. 더 단순한 설명은 "현재 7-branch 공식 구성의 seed std 0.0021~0.0083도 같은 분산 구조를 가질 수 있다"는 추측이 아니라, 현재 구성의 분산분해가 아직 실측되지 않았으므로 seed averaging의 효과를 전이할 수 없다는 것이다.
- **무엇이 참이면 이 주장이 무너지는가**: 현재 7-branch에서 common seed pairing 또는 seed averaging을 적용했을 때 paired Δ의 SE가 실질적으로 줄고, 그 감소가 fold×seed 잔차보다 seed 주효과에 의해 설명된다면 가설은 지지된다. 반대로 현재 구성에서도 fold×seed 잔차가 지배적이고 common seed pairing으로도 SE가 거의 줄지 않는다면, seed 평균은 승격 판정의 정밀도를 크게 높이지 못한다. 또한 common random numbers가 동일한 random draw를 공유할 수 있다면 fold×seed 잔차 중 일부는 제거될 수 있으므로, "fold×seed = 제거 불가"로 단정할 수 없다.
- **이 반론의 한계**: RU-97의 fold×seed 23.9%와 seed 주효과 0~2.9%는 실제 실측이므로 무시할 근거는 없다. 다만 그것이 현재 공식 7-branch의 적합 분산 구조를 대표한다는 증거는 문서에 없다. 따라서 이 가설은 현재 구성에서 분산분해와 common-seed paired Δ SE를 재보기 전까지 판별 불가다.

### A1 (감사)

## 점검 1: 유효 랭크 3.52의 산출 무결성
- **확인하려는 것**: `3.52 / 7`이 Primary 7 고정 50-fold의 라벨 무관 margin에서 후보·집계 선택 전에 계산됐는지, SEAL 10·성능 조회·fold 재분할이 섞이지 않았는지, branch list·가중치·코드 해시·fold manifest가 고정돼 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: rank가 후보 선별 후 또는 SEAL 열람 후 계산됐다면 “branch 추가가 막혔다”와 “서로 다른 것을 재는 branch를 아직 못 만들었다”를 가르는 판단이 데이터 설계에서 새어 나온 값이 된다. provenance가 없으면 3.52를 특정 7-branch 구성에 귀속시킬 수 없어 branch 추가·집계 변경 판단의 근거가 무효해진다.
- **확인 방법**: `docs/PROJECT.md` §3.2·§5, `docs/closed_axes.md` `CA-09`/`CA-10`, `scripts/analysis/branch_screen.py`, 저장 margin·fold manifest·코드 해시를 대조한다. `SCREEN_ONLY=1`으로 Primary 7 고정 fold에서만 재계산하고, `grep -R "SEAL" talks/reports talks/council`, `sha256sum <fold_manifest>`, `git rev-parse HEAD`로 선택 이전·고정 여부를 확인한다.

## 점검 2: 유효 랭크 해석을 가르는 라벨 무관 관측
- **확인하려는 것**: “측정 중복”과 “정보 부재”를 가르는 관측이 라벨·성능 조회 없이 가능할지. raw descriptor rank와 7-branch margin rank, BM·QA·DS 위치 계열의 상관 집중 여부, fold·seed·정규화 범위가 사전 고정됐는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: raw descriptor audit에 query 라벨, fold별 성능, SEAL, 후보별 최적 정규화를 쓰면 해석이 새어 나온다. margin rank만 보고 raw feature 정보 부재를 결론짓거나, 반대로 raw rank만 보고 margin 중복을 결론짓는 것도 설계 오류다. rank 관측은 기전 입증이 아니라 중복의 위치를 좁히는 진단에 그친다.
- **확인 방법**: Primary 7 고정 fold에서 label-free raw descriptor와 7-branch margin의 eigen/correlation을 사전 선언된 fold·seed·정규화 범위에서 계산한다. `branch_screen.py` 또는 별도 audit 스크립트로 `max|r|`, eff.rank, 계열별 분산을 기록하고, 성능 조회 전 게이트 ①처럼 종료한다. `docs/closed_axes.md` `CA-09`/`CA-10` 경계와 대조한다.

## 점검 3: 집계 변경의 선택 편향
- **확인하려는 것**: trimmed mean voting을 바꾸는 후보(중앙값, rank/quantile 정규화, 라벨 무관 variance-weighted, PSW·TGW 등)가 Primary 7에서 최댓값을 보고 고른 뒤 같은 자료에서 승격 판정하는 CA-08 양식인지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 여러 집계 함수를 돌리고 Primary 7 macro가 가장 높은 것을 “개선”으로 기록하면, fold 짝짓기·seed 변동과 무관하게 선택 편향이 +0.0030을 넘을 수 있다. SEAL 10으로 집계 함수를 고르면 D-047 위반이다.
- **확인 방법**: `docs/closed_axes.md` `CA-08`, `D-022`, `docs/PROJECT.md` §4.1을 대조한다. 후보 집계 규칙을 성능 조회 전에 사전 등록하고, 같은 350 fold에서 baseline과 paired Δ만 계산한다. `talks/council/` 카드, 실험 로그, 선택 이력, `bash scripts/run_tests.sh`, `python -m pytest tests/test_docs_consistency.py`로 확인한다.

## 점검 4: 적합 분산과 승격 임계의 estimand
- **확인하려는 것**: seed std 0.0021~0.0083과 +0.0030을 비교할 때 estimand가 단일 적합 Δ인지 적합 평균 Δ인지, fold·seed 짝짓기가 같은지, best-seed 선택이 없는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 단일 적합 Δ에 seed std를 단순 비교하거나, fit-averaged Δ를 승격 기준으로 쓰면서 §4의 단일 fold Δ 공식을 그대로 쓰면 불확실성 과소평가/과대평가가 생긴다. best seed 선택, seed별 Δ 최댓값, fold 재분할은 새어 나온 승격 근거가 된다.
- **확인 방법**: `docs/PROJECT.md` §4, `D-048`, `docs/current_status.md` 미해결, `talks/reports/2026-09-17_ru97_gf_primary7.md` §8~§10을 대조한다. 승격 estimand를 사전 승인받기 전까지 학습 파라미터 후보 승격은 보류한다. same fold·same seed paired Δ, seed list 고정, best-seed 금지 여부를 코드와 로그로 확인한다.

## 점검 5: 분산 감소와 성능 개선의 구분
- **확인하려는 것**: seed averaging, common random numbers, deterministic closed-form, fixed manifest, provenance 등 분산 감소가 모델의 기대 Δ를 바꾸는 것이 아니라 측정 SE만 줄이는 것인지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 좋은 seed를 골라 평균하거나, fold 중 쉬운 fold만 남기거나, 후보별 seed 수를 다르게 하면 분산 감소가 성능 개선으로 위장한다. variance-weighted aggregation이 branch scale 편향을 제거하는지 task signal을 고르는지 구분되지 않으면 새어 나온다. RU-97의 fold×seed 잔차(예: m32_raw 23.9%)는 seed 평균만으로 사라지지 않는 잔여 불확실성이다.
- **확인 방법**: 동일 구성에서 seed·fold를 고정하고 분산 감소 전/후 paired Δ와 SE를 비교한다. `talks/reports/2026-09-17_ru97_gf_primary7.md`의 seed 주효과·fold×seed 잔차 기록, `D-048` 적합 분산 보고 의무, 코드에서 seed 선택·fold 필터링·후보별 seed 수 불일치가 없는지 감사한다.

## 점검 6: 건드리지 않은 축의 경계
- **확인하려는 것**: “아무도 건드리지 않은 축”이 closed_axes의 경계 밖인지, open issue인지, 아니면 단순 미고안인지. 특히 provenance writer, 적합 분산 승격 절차, LSAK 사양, fold timing, raw descriptor rank audit, 집계 scale 진단이 해당되는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: closed_axes의 경계 안을 “새 축”으로 포장해 실행하면 과거 기각 기전을 우회한다. 반대로 open issue(provenance, fit-variance procedure, LSAK spec, CA-R1)를 무시하고 성능 탐색을 계속하면 평가 무결성이 먼저 무너진다.
- **확인 방법**: `docs/closed_axes.md` 전체, `docs/current_status.md` 미해결, `docs/PROJECT.md` §5를 대조한다. 후보 축을 CA-08/09/10/14, CA-R1, §3-G/H/I/J와 대조한다. 경계 미확정(G/H/I/J)은 사용자 판정 대기이므로 착수하지 않는지 확인한다.

## 점검 7: 4 GPU-hour 판별 가능성
- **확인하려는 것**: fold 1회 적합 시간 실측값이 존재하는지, 그리고 그 값으로 4 GPU-hour 안에 baseline+후보 350-fold paired 비교를 계획할 수 있는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: fold 시간을 추측해 full 350-fold를 계획하면 시간 초과, 일부 fold만 실행, baseline과 후보의 fold 수 불일치, 재사용 캐시가 다른 구성으로 수렴하는 실행 무효가 생긴다.
- **확인 방법**: `docs/current_status.md` 예산, `docs/PROJECT.md` §4를 확인한다. GPU 4에서 1-fold timing probe를 사전 등록하고 wall time·메모리·fold manifest 해시를 기록한다. probe가 4 GPU-hour 안에 baseline+후보 paired 재적합을 허용하지 않으면 이번 판정을 `실행 무효`로 기록하고 재집계/스크리닝으로 축을 바꾼다.

## 점검 8: baseline 0.6227의 verified 상태
- **확인하려는 것**: 공식 기준선 0.6227이 unverified-hold인데, 후보 Δ를 verified baseline으로 계산하지는 않는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: provenance가 없는 과거 예측을 baseline으로 쓰면 후보와 baseline이 다른 branch list·가중치·코드·fold manifest일 수 있다. +0.0030 Δ가 구성 차이·러너 폴백·seed 변동에서 새어 나올 수 있다.
- **확인 방법**: `docs/PROJECT.md` §3.2 D-050, `talks/reports/2026-09-18_ru90_baseline_regen.md`를 확인한다. baseline 재현 시 branch list·가중치·코드 해시·데이터 버전·fold manifest 해시를 기록하고, `scripts/eval_v121.sh`→`eval_seal_tasks.sh` 폴백 경로가 7-branch로 수렴하지 않는지 확인한다. `bash scripts/run_tests.sh`, `python -m pytest tests/test_docs_consistency.py`를 실행한다.

## 점검 9: SEAL 10 선택 금지
- **확인하려는 것**: 개선 방향·후보·집계·임계값 선택에 SEAL 10 결과가 어떤 형태로도 쓰이지 않았는지.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: SEAL 10 macro를 보고 후보를 고르면 Primary 7 개선이 hold-out 개선을 보장하지 않는다는 §1 경고를 무력화한다. SEAL 10을 탐색 기록으로만 쓰고도 후보 선별에 반영하면 D-047 위반이다.
- **확인 방법**: `docs/PROJECT.md` §1·§3.1, `D-047`, `docs/current_status.md`를 확인한다. 실험 로그·council 카드·보고서에서 SEAL 열람 시점과 후보 선택 시점을 대조한다. `grep -R "SEAL" talks/council talks/reports`, `git log -p -- src scripts configs`로 선택 전후 변경을 확인한다. SEAL 10 실행 결과가 모델 선택에 입력되지 않는지 데이터 흐름을 감사한다.

## 점검 10: 라벨·fit 범위·정규화 누수
- **확인하려는 것**: PCA-32, whitening, ridge, quantile, variance 추정 등 전처리·통계량이 fold context/train 범위에서만 fit되는지, query/test/SEAL/전체 Primary 7을 fit에 쓰지 않는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test/query/SEAL로 정규화하거나, 전체 Primary 7 fold를 합쳐 fit한 뒤 fold별 AUROC를 비교하면 fold 간 독립이 깨지고 Δ가 새어 나온다. context 라벨을 readout에 쓰는 것은 D-049 범위와 달리 query 라벨·AUROC 후 선택을 쓰면 oracle/CA-06 저촉이다.
- **확인 방법**: `src/`의 fit·normalize·PCA·ridge·quantile 호출부, `configs/baseline/v121_active.yaml`, `evaluate_pure.py`, `D-049`, `CA-06`을 감사한다. fold별 fit 범위를 코드 감사로 확인하고, 합성 데이터로 query/test를 넣었을 때 margin이 변하는지 단위 테스트한다. `bash scripts/run_tests.sh`를 실행한다.

## 점검 11: paired fold 평가 독립성
- **확인하려는 것**: candidate와 baseline이 같은 350 fold, 같은 fold manifest, 같은 seed, 같은 전처리에서 짝지어지는지.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: fold 재분할, fold 수 불일치, baseline과 후보의 seed/manifest 불일치, cached prediction 재사용으로 Δ가 과제 난이도·fold 효과에서 새어 나온다.
- **확인 방법**: `docs/PROJECT.md` §4.1을 기준으로 저장 예측 파일의 fold id·seed·manifest hash를 1:1 join한다. missing fold 0건, hash 일치, baseline과 후보의 전처리 버전 일치를 확인한다. `scripts/eval_v121.sh`, `evaluate_pure.py`, 저장 margin·prediction 파일의 provenance 필드를 대조한다.

---

## Phase 2 — 교차

### T1 (이론)

## 이론적 병목 1
- **가정**: P1·P2의 제안들은 공통으로, (a) raw slide descriptor의 label-free effective rank와 7-branch margin effective rank(3.52/7)의 gap이 “branch 추가가 막혔는가/아직 다른 branch를 못 만들었는가/upstream compression·readout 손실인가”를 인과적으로 구분하고, (b) 1-fold fit time probe와 same-seed/draw/fold 처리 순서 fit averaging이 4 GPU-hour 안에서 +0.0030 승격 판별을 가능하게 하며 fit variance를 충분히 줄이고, (c) fold/task rank 또는 context quantile 변환이 branch margin의 scale nuisance를 제거해 ensemble ordering을 개선하고, (d) branch list·가중치·코드 해시·fold manifest·seed/draw provenance만으로 산출물을 구성에 귀속시켜 verified 비교를 가능하게 하고, (e) Primary 7의 +0.0030 paired Δ 개선이 SEAL 10 ABMIL 초과로 전이될 수 있는 선택 신호라고 두고 있다. 즉, 성능 조회 이전의 label-free 진단량·비용 측정·집계 변환·provenance·Primary 7 paired Δ가 다음 연구 축을 선택하고 최종 목표에 도달하는 충분한 판별자라고 참으로 둔다.
- **이 가정이 깨지는 조건**: effective rank가 정규화·차원·fold/task 표본 수·공분산 추정기·ridge 정규화·노이즈에 민감해 raw/margin gap이 정보 손실이 아니라 측정 산출물일 때; raw rank가 높아도 과제 관련 방향이 없거나, margin rank가 낮아도 nonlinear/readout/ensemble 변화로 AUROC가 개선되거나, raw rank ≤ margin rank여도 기존 low-rank 정보의 다른 결정 규칙이 성능을 올릴 때; fold fit time이 task/fold/context slide 수(90~261)에 따라 비균일하거나 1-fold probe가 전체 비용을 대표하지 못하거나, GPU-hour 예산이 I/O·CPU·GMM·전처리 비용을 포함하지 못하거나, R=1만 가능하거나, fold×seed residual이 seed main보다 지배적이거나, baseline과 후보의 랜덤 요인이 same seed/draw/fold 순서로도 coupling되지 않을 때; fit-variance 포함 승격 절차가 사용자 미승인으로 남아 있을 때; branch margin magnitude가 증거 강도를 담고 있거나 context-query margin 분포 이대가 quantile mapping을 왜곡하거나 query slide 표본이 rank/quantile을 불안정하게 할 때; provenance가 데이터 버전·라이브러리 버전·GPU nondeterminism·RNG state·fold 처리 순서·전처리 중 하나를 놓쳐 구성 귀속을 보장하지 못할 때; Primary 7과 SEAL 10의 task·fold·slide 분포가 달라 Primary 7 paired Δ가 hold-out 성능을 대표하지 못할 때.
- **깨질 때 나타나는 관측 가능한 증상**: raw rank > margin rank여도 FC 또는 다른 후보가 gate ①에서 max |r| > 0.6 또는 eff.rank 효율 하락으로 기각되고 paired Δ가 +0.0030에 미달·악화하면 (a)의 인과 해석은 반박; raw rank ≤ margin rank여도 aggregation/fit-variance/readout 변경이 paired Δ를 개선하면 branch 추가 중단 규칙은 반박; effective rank 값이 스케일·차원·정의 변경에 따라 뒤집히면 판별 불가. 1-fold time이 4 GPU-hour를 초과하거나, GPU time은 예산 내인데 I/O·CPU·GMM·전처리 때문에 전체가 초과하면 실행 무효; R=1로 적합 분산 추정이 불가하면 판별 불가; same-seed paired Δ의 SE가 +0.0030보다 크거나 seed 평균을 늘려도 fold×seed residual이 줄지 않으면 판별 불가; baseline과 후보의 seed 민감도 차이로 paired Δ가 상쇄되지 않으면 (b)의 fit-averaged estimand는 반박. rank/context-quantile 변환 후 단독 branch AUROC는 변하지 않으나 ensemble paired Δ가 +0.0030에 미달·악화하면 (c)의 scale nuisance 가정은 반박; context-query 분포 이대가 큰 fold/task에서 ensemble ordering이 불안정하거나 magnitude 손실이 나타나면 quantile mapping은 반박; 여러 aggregation 변형을 Primary 7에서 비교한 뒤 최댓값을 승격 근거로 쓰면 CA-08 선택 편향으로 실행 무효. 같은 branch list·가중치·코드 해시·fold manifest·seed여도 fold AUROC가 달라지거나 과거 산출물을 구성에 귀속시킬 수 없으면 (d)의 provenance 가정은 반박되고 게이트·승격 비교는 실행 무효. Primary 7 paired Δ가 +0.0030을 넘어도 최종 hold-out 판정에서 목표를 넘지 못하면 (e)의 전이 가정은 반박되나, 이 결과는 선택에 쓰지 않고 최종 판정만 사용한다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: CA-10에서 형상통계 강화 7종이 게이트 ②를 통과하지 못해 형상 계열 병목이 절사가 아니라 정보량이라고 결론지은 관측은 (a)의 effective rank/투영 손실 gap만으로 “측정 중복 vs 정보 부재”를 구분한다는 일반화와 충돌한다. RU-97에서 동일 구성 seed std 0.0021~0.0083, 승격 임계 +0.0030, m32_raw fold×seed 잔차 23.9%로 seed 주효과보다 크다는 관측은 (b)의 same-seed fit averaging만으로도 fit variance를 충분히 줄인다는 가정과 충돌한다. current_status의 “적합 분산을 포함한 승격 판정 절차가 미승인”은 fit-averaged paired Δ를 승격 estimand로 직접 사용한다는 (b)와 충돌한다. CA-07에서 context↔query 분포 이동을 LOO 폐기 원인으로 확인하고 §215 SMAD4 역전을 미해결로 둔 관측은 (c)의 context 경험분포가 query margin을 안정적으로 대표한다는 가정과 충돌한다. D-050에서 여섯 러너가 폴백으로 같은 7-branch 설정으로 수렴했고, 저장 예측 파일에 provenance가 없어 과거 산출물을 구성에 귀속시킬 수 없으며, 5-branch 앵커 재현 실패 원인이 미확인인 관측은 (d)의 제안 provenance 필드로 구성 귀속이 보장된다는 가정과 충돌한다. PROJECT.md §1의 “Primary 7의 개선이 hold-out 개선을 보장하지 않는다”는 문장은 (e)의 Primary 7 paired Δ가 SEAL 10 초과로 전이된다는 가정과 충돌한다. fold 적합 1회 시간과 raw descriptor effective rank 측정값은 문서에 없어, 1-fold probe의 충분성과 raw/margin gap 판별은 아직 미확인이다.

### M1 (실측)

## 실측 1: P1 제안 1 — 4 GPU-hour feasibility · effective-rank gate
- **건드려야 하는 코드·설정**: `scripts/analysis/branch_screen.py`(게이트 ①), `scripts/eval_v121.sh`·`eval_seal_tasks.sh`·`evaluate_pure.py`(러너 폴백·평가 진입점), `configs/baseline/v121_active.yaml`(branch 가중치 확인 경로; 7-branch 설정 파일 경로 미확인), provenance writer(경로 미확인), raw slide descriptor/margin 저장·재계산 코드(경로 문서에 없음). fold timing probe를 위해 평가 루프에 timer와 provenance 기록을 추가해야 한다.
- **비용**: fold 적합 1회 시간 `t`는 미실측이고, `t`의 정의(데이터 로딩 포함/제외, GPU/CPU)도 미실측이다. 1-fold probe는 `1t`, baseline 350-fold는 `350t`, baseline+candidate R=1은 `700t`다. 4 GPU-hour=14400s에서 R=1이면 `t ≤ 20.6s`, R=2이면 `t ≤ 10.3s`, R=4이면 `t ≤ 5.1s`가 필요하며 이는 조건식이지 실측이 아니다. raw descriptor rank audit는 검증된 margin/raw가 이미 있으면 CPU 연산(미실측)이고, 없으면 350t 이상에 raw 추출·저장 비용(미실측)이 추가된다. 메모리·디스크는 미실측.
- **막는 것**: fold time 미실측으로 4h 내 full paired Δ 가능 여부를 판정할 수 없다. provenance(branch list·가중치·코드 해시·fold manifest·seed/draw)가 없어 저장 margin을 구성에 귀속할 수 없고, D-050으로 공식 기준은 unverified-hold다. 여섯 러너가 폴백으로 같은 7-branch로 수렴한 이력이 있어 arm 전달 경로 복구가 선행되어야 한다. 오염 검사 재설계 전 SCREEN_ONLY 통과를 주장하면 무효다. raw descriptor 저장 경로와 effective rank 정의(차원·스케일·정규화·slide 수)가 문서에 없어 raw rank와 margin rank 비교가 재현 불가하다. 3.52/7은 unverified-hold 계측이며, raw rank gap이 compression 손실과 정보 부재를 가르는 기전은 미확인이다.
- **중단 기준**: 1-fold probe에서 `700Rt > 14400s`이면 해당 R의 full 판별은 실행 무효로 종료한다. provenance writer 없이 margin을 verified로 쓰면 중단한다. raw rank가 랭크 정의·스케일·과제 난이도 변경에 따라 부호나 크기가 바뀌면 판별 불가로 종료한다. raw rank ≤ margin rank이면 FC 등 branch 추가 후보는 이 제안에서 중단한다.

## 실측 2: P1 제안 2 — fit-averaged paired Δ (same-seed pairing)
- **건드려야 하는 코드·설정**: seed·random draw·fold 처리 순서를 고정하는 코드(경로 문서에 없음), paired Δ 계산 코드(경로 문서에 없음), `evaluate_pure.py`/러너(`scripts/eval_v121.sh`·`eval_seal_tasks.sh`)의 arm 전달 경로, provenance writer(경로 미확인). `ICF_*` 환경변수 경로는 무효이므로 YAML 또는 코드 인자 경로로만 고정해야 한다.
- **비용**: baseline과 candidate를 같은 fold·seed·draw로 R회 재적합하면 fold fit 수는 `700R`이다. `t` 미실측이며, 4h에서 R=1은 `t ≤ 20.6s`, R=2는 `t ≤ 10.3s`, R=4는 `t ≤ 5.1s`가 필요하다. per-fold AUROC/margin 저장량은 `350R`개지만 파일 크기는 미실측. seed 평균·paired Δ 계산 CPU 비용은 미실측.
- **막는 것**: 적합 분산을 포함한 승격 판정 절차가 현재 미승인 상태다. RU-97의 seed sd `0.0021~0.0083`과 fold×seed 잔차 `23.9%`는 RU-97 GF 구성의 실측이지 현재 7-branch의 공통 상한이 아니다. same-seed pairing이 seed main을 상쇄하는 것은 수학적으로 옳지만, candidate가 RNG 소비 순서를 바꾸면 같은 seed가 같은 draw를 보장하지 않는다. draw·RNG 상태를 통제하는 경로가 문서에 없다. provenance가 없으면 어떤 fit도 구성에 귀속 불가하다.
- **중단 기준**: 4h에서 R=1만 가능하면 적합 분산을 추정할 수 없어 판별 불가로 종료한다. same fold·seed·draw를 코드 수준에서 보장하지 못하면 실행 무효로 종료한다. 측정된 paired Δ의 SE를 산출하지 못하면 판별 불가다. 승격 절차 미승인 상태에서 Δ를 승격 근거로 쓰면 중단하고, Primary 7 탐색 기록으로만 남긴다.

## 실측 3: P1 제안 3 — per-fold rank aggregation
- **건드려야 하는 코드·설정**: trimmed mean 집계 구현(경로 문서에 없음), fold·task 내 query slide rank/quantile 변환 코드(경로 문서에 없음), `evaluate_pure.py`/러너의 arm 전달, provenance writer(경로 미확인), 단일 변형을 결과 조회 전에 고정하는 사전 등록 기록(경로 문서에 없음).
- **비용**: 검증된 baseline/candidate margin이 이미 저장되어 있으면 재집계 CPU 비용만(미실측)이다. 그러나 현재 provenance가 없어 verified margin으로 인정할 수 없으므로, paired Δ를 verified로 만들려면 candidate 350t와 baseline 350t 재적합이 필요하고 R=1이면 `700t`다. 4h 조건은 `t ≤ 20.6s`(R=1). rank 변환·집계 CPU 비용은 미실측.
- **막는 것**: CA-08 경계에서 단일 사전 등록 변형만 허용되는지, 그리고 선택 이력을 반영한 평가 설계가 필요한지 아직 승인된 절차가 없다. branch margin의 scale 불일치가 aggregate noise의 원인이라는 진단이 실측되지 않았고, branch별 기여도도 미실측이다. query slide 수·fold별 slide 수 분포가 문서에 없어 rank 변환의 안정성을 판정할 수 없다. rank 변환이 단독 branch AUROC를 보존하는 것은 rank 불변성으로 옳지만, ensemble ordering 개선으로 이어지는지는 기전 미확인이다. 여러 aggregation 변형을 Primary 7에서 비교한 뒤 최댓값을 쓰면 CA-08 선택 편향으로 실행 무효다. provenance가 없으면 실행 무효다.
- **중단 기준**: 결과 조회 전에 단일 규칙과 선택 이력을 고정하지 못하면 실행 무효로 종료한다. verified margin을 만들 수 없으면 중단한다. rank 변환 후 단독 branch AUROC가 구현 오류로 바뀌면 실행 무효로 종료한다. fit 분산을 포함한 verified paired Δ가 `+0.0030`에 미달하면 반박으로 종료한다.

## 실측 4: P2 제안 1 — common-random-number paired-Δ probe
- **건드려야 하는 코드·설정**: 1-fold timing probe 코드(경로 문서에 없음), seed·draw·fold 순서 고정 코드(경로 문서에 없음), paired Δ 계산 코드(경로 문서에 없음), `scripts/eval_v121.sh`·`eval_seal_tasks.sh`·`evaluate_pure.py`의 arm 전달, provenance writer(경로 미확인).
- **비용**: 1-fold probe는 `1t`다. 이후 paired variance probe는 baseline+candidate `700R` fold fit이다. `t` 미실측이며 4h에서 R=1은 `t ≤ 20.6s`, R=2는 `t ≤ 10.3s`, R=4는 `t ≤ 5.1s`다. probe 산출물(시간·provenance) 저장량은 미실측.
- **막는 것**: 이 제안은 P1 제안 2와 동일한 비용 구조와 막는 것을 가진다. 추가 차별점은 1-fold timing probe를 먼저 둔다는 점뿐이며, probe가 통과해도 draw 통제·provenance·승격 절차 미승인 문제가 남는다. current_status의 현재 목표는 계측 복구(provenance writer, SH/SJ control, 오염 검사 재설계, 단위 테스트)이고, 복구 전 게이트·승격 비교를 verified로 수행할 수 없다. RU-97의 fold×seed 잔차 `23.9%`를 현재 7-branch에 전이하는 근거가 없다.
- **중단 기준**: 1-fold probe에서 `700Rt > 14400s`이면 해당 R의 probe 이후 full 판별은 실행 무효로 종료한다. R=1만 가능하면 적합 분산 추정 불가로 판별 불가다. same fold·seed·draw를 보장하는 코드 경로가 없으면 실행 무효다. provenance writer 복구 전 verified Δ를 주장하면 중단한다. SEAL 10을 열어 후보를 고르면 선택 금지 위반으로 실행 무효다.

## 실측 5: P2 제안 2 — context-quantile trimmed mean
- **건드려야 하는 코드·설정**: context slide margin 경험분포 계산 코드(경로 문서에 없음), query margin quantile 매핑 코드(경로 문서에 없음), trimmed mean 집계 구현(경로 문서에 없음), `evaluate_pure.py`/러너 arm 전달, provenance writer(경로 미확인). 라벨 미사용을 코드에서 보장하는 경로가 문서에 없다.
- **비용**: 검증된 margin이 있으면 quantile 매핑·재집계 CPU 비용만(미실측)이다. 현재 provenance가 없으므로 verified paired Δ를 만들려면 baseline+candidate 재적합 `700t`(R=1)가 필요하고, 4h 조건은 `t ≤ 20.6s`다. context margin 분포 계산은 fold당 context slide `90~261`장을 쓰지만 연산 비용은 미실측.
- **막는 것**: CA-07이 context↔query 분포 이동을 실측으로 기록하고 있으므로, context 경험분포로 query margin을 quantile 매핑하는 것이 scale 보정이 아니라 분포 이동 노이즈를 심을 수 있다. context slide 수는 `90~261`이지만 quantile 추정 노이즈와 context-query 분포 이대 크기는 미실측이다. branch margin의 magnitude가 신호인지 노이즈인지 실측되지 않았다. CA-08 경계에서 단일 사전 등록 변형과 선택 이력 기록이 필요하며, 승인된 절차가 없다. 여러 quantile/aggregation 변형을 비교한 뒤 최댓값을 선택하면 실행 무효다. provenance가 없으면 실행 무효다.
- **중단 기준**: context margin 분포를 라벨 없이 재현 가능하게 저장하지 못하면 실행 무효로 종료한다. quantile 매핑 후 단독 branch AUROC가 바뀌면 구현 오류로 실행 무효다. context-query 분포 이대 진단을 사전에 정의하지 못하면 판별 불가로 종료한다. fit 분산 포함 verified paired Δ가 `+0.0030`에 미달하면 반박으로 종료한다.

## 실측 6: P2 제안 3 — raw-descriptor rank audit + FC gate
- **건드려야 하는 코드·설정**: `scripts/analysis/branch_screen.py`(게이트 ①), FC branch 구현(경로 문서에 없음), raw slide descriptor 추출·저장 코드(경로 문서에 없음), `evaluate_pure.py`/러너 arm 전달, provenance writer(경로 미확인), 오염 검사 재설계 코드(경로 문서에 없음).
- **비용**: FC 구현·디버깅 시간은 미실측. SCREEN_ONLY margin 산출은 raw feature가 있으면 CPU(미실측), 없으면 350t 이상이다. raw descriptor rank audit는 저장 raw/margin이 있으면 CPU(미실측), 없으면 재계산 `350t` + raw 추출(미실측)이다. 8-branch 게이트 ①에 필요한 eff.rank 계산 CPU 비용은 미실측.
- **막는 것**: 게이트 ①은 max |r|과 eff.rank 효율을 요구하지만, 오염 검사 재설계 전 SCREEN_ONLY 통과를 주장하면 무효다. provenance가 없어 margin을 FC 또는 기존 7-branch 구성에 귀속할 수 없다. FC는 `CA-09`에서 비저촉으로 명시됐지만, raw rank gap이 FC가 읽을 수 있는 정보를 보장하지는 않는다. effective rank 정의가 스케일·차원·slide 수·정규화에 민감하면 raw/margin gap이 재현 불가하다. 3.52/7과 8-branch 통과선 eff.rank ≥ 4.02는 unverified-hold 계측에 의존한다.
- **중단 기준**: 게이트 ①에서 max |r| > 0.6이면 성능 조회 없이 기각으로 종료한다. 8-branch eff.rank < 4.02(효율 50.3% 하락)이면 기각으로 종료한다. 오염 검사를 통과할 provenance 기반 재설계가 없으면 실행 무효로 종료한다. raw rank ≤ margin rank이면 branch 추가 후보는 중단한다. FC 구현·margin 산출 비용이 4h를 초과하면 실행 무효로 종료한다.

### R1 (반증)

## 반론 1
- **공격 대상**: P1-1과 P2-3의 “raw slide descriptor의 label-free effective rank와 7 branch margin effective rank의 gap이, branch 추가 부족과 slide-level compression/whitening/readout 손실을 가른다”는 주장.
- **더 단순한 설명**: 같은 관측(raw rank가 높고 margin rank가 낮음)은 branch가 부족하다는 설명보다 더 단순한 요인으로도 생긴다. raw descriptor가 patch-level 고차원·노이즈 성분을 포함하는 반면, branch margin은 within-context PCA·covariance·closed-form ridge·bag mean 같은 슬라이드 수준 압축을 거친 결과일 수 있다. 문서상 context 슬라이드 수는 90~261장이지만 feature 차원은 1536으로 언급되므로, context covariance 기반 readout의 유효 자유도는 구조적으로 제한될 수 있다. 또한 effective rank는 차원, 스케일, 표본 수, 정규화, ridge 강도, rank estimator 정의에 민감하다. 따라서 raw rank > margin rank는 “아직 읽히지 않는 정보가 있다”는 것보다 “raw와 margin이 다른 차원·노이즈·정규화 조건에서 측정됐다”는 설명이 더 단순하다.
- **무엇이 참이면 이 주장이 무너지는가**: raw와 margin의 effective rank가 같은 fold·과제·슬라이드 집합, 같은 label-free rank estimator, 같은 스케일/정규화, 같은 노이즈/차원 대조 조건에서 측정되지 않으면 gap은 판별력이 없다. margin rank가 closed-form ridge의 축소 효과나 context covariance의 표본 크기 제한에서 나온 것이라면, raw rank가 높다는 사실만으로 compression/whitening/readout 변경이 paired Δ +0.0030을 낳는다고 추론할 수 없다. 반대로 raw rank ≤ margin rank가 estimator 차이에서 나온 것이라면, 그 근거로 branch 추가를 중단하는 규칙도 무너진다. 이 경우 판별은 판별 불가다.
- **이 반론의 한계**: raw descriptor와 margin을 동일 슬라이드·fold에서 동일 estimator로 측정하고, 차원/노이즈 대조(예: 동일 차원 재표본, 동일 스케일)를 한 뒤에도 raw rank 우위가 유지되면, 그 gap은 “투영 손실 후보”를 가리키는 진단으로 쓸 수 있다. 다만 그것이 AUROC 개선으로 전이되는지는 별도 paired Δ 검증이 필요하고, rank gap 자체는 성능 기전이 아니다.

## 반론 2
- **공격 대상**: P1-2와 P2-1의 “same fold·same seed·same draw pairing과 R-seed 평균이 seed main을 상쇄하고 fit noise를 줄여, +0.0030 승격 판정을 가능하게 한다”는 주장.
- **더 단순한 설명**: seed std 0.0021~0.0083이 독립적인 적합 노이즈가 아니라 random draw, fold 처리 순서, branch별 subsampling, RNG 스트림 소비 차이 같은 실행 경로 변동에서 나온 것일 수 있다. 같은 seed를 부여한다고 해서 baseline과 후보가 실제로 같은 random number를 같은 순서로 소비하는 것은 아니다. RU-97의 fold×seed 잔차 23.9%는 seed 평균의 상한이 낮을 수 있음을 시사하지만, 그것은 RU-97의 GF 구성 공간 실측이며 현재 7-branch에 그대로 전이된다는 증거는 문서에 없다. 따라서 “same-seed pairing이면 SE가 충분히 줄어든다”는 주장보다, “seed 변동이 common random number로 상쇄되지 않는 실행 경로 상호작용일 수 있다”는 설명이 더 단순하다.
- **무엇이 참이면 이 주장이 무너지는가**: baseline과 후보의 random draw·fold order·RNG consumption이 동일하지 않거나, fold×seed 상호작용이 seed main보다 크거나, 4 GPU-hour에서 R=1만 가능하면 fit-averaged paired Δ는 +0.0030을 판별하지 못해 판별 불가다. 또한 적합 분산을 포함한 승격 판정 절차가 현재 사용자 미승인 상태이므로, 이 estimand를 승격 근거로 바로 쓰는 것은 실행 무효 또는 미검증이다. baseline 예측이 provenance 없이 unverified-hold라면, 어떤 paired Δ도 verified 승격 근거가 될 수 없다.
- **이 반론의 한계**: baseline과 후보가 실제로 동일한 common random number를 소비하고, fold×seed 잔차가 seed main보다 작으며, R이 충분히 크고, 적합 분산 포함 승격 절차가 승인되면, same-seed fit-averaged paired Δ는 Δ의 SE를 줄이는 유효한 측정 설계가 될 수 있다.

## 반론 3
- **공격 대상**: P1-3의 “fold·task 내 query slide rank 변환”과 P2-2의 “context margin 경험분포로 query margin을 quantile 매핑한 뒤 trimmed mean”이 branch scale 불일치를 제거하고 paired Δ를 개선할 수 있다는 주장.
- **더 단순한 설명**: rank/quantile 변환은 scale bias 제거가 아니라 branch 재가중으로 설명될 수 있다. 각 branch margin의 절대 크기를 fold·task 내 순위나 context quantile로 바꾸면, scale이 큰 branch가 trimmed mean을 지배하던 효과가 균등화된다. 이는 aggregate noise 감소가 아니라 branch effective weight 변경으로 같은 관측을 낳을 수 있다. 만약 margin magnitude 자체가 증거 강도·신뢰도 신호라면, rank/quantile 변환은 정보를 버린다. P2-2의 context quantile은 context margin 분포가 query margin 분포를 대표한다는 전제를 쓰지만, context↔query 분포 이동은 이미 CA-07과 §215에서 문제화된 축이다. 따라서 “scale 불일치 제거”보다 “branch 재가중 또는 context-query 분포 이대로 인한 왜곡”이 더 단순한 설명이다.
- **무엇이 참이면 이 주장이 무너지는가**: branch margin scale이 이미 비교 가능하거나, magnitude에 유용한 신호가 있거나, context-query 분포 이대가 context quantile을 query rank로 왜곡하거나, fold·task 내 query slide 수가 rank를 안정적으로 만들지 못하면 이득이 없다. query slide 수는 문서에 없으므로 rank 안정성은 미확인이다. 여러 aggregation 변형을 Primary 7에서 돌려 최댓값을 승격 근거로 쓰면 CA-08의 선택 편향에 해당해 실행 무효다. 단일 변형이라도 paired Δ +0.0030에 미달하면 반박이다.
- **이 반론의 한계**: 라벨 무관 진단에서 branch scale 편차가 task signal보다 지배적이고, context-query 분포가 충분히 유사하며, 단일 사전 등록 변형만 검증한다면 rank/quantile 변환은 aggregate noise 감소가 아니라 재가중이라는 비판을 피할 수 있는 제한된 후보가 될 수 있다.

## 반론 4
- **공격 대상**: P2-3과 P1-1의 “raw rank > margin rank이면 FC를 gate ①에 올리고, raw rank ≤ margin rank이면 branch 추가를 중단한다”는 주장.
- **더 단순한 설명**: 유효 랭크 3.52/7은 branch 종류가 부족하다는 것보다, 7개 branch가 같은 slide-level descriptor 계열(PCA-32 bag mean, whitened 32D 구름, within-context covariance 등)에서 나온 중복 사영이라는 설명이 더 단순할 수 있다. FC도 같은 patch feature/slide descriptor에서 나오는 구조이므로, 기존 위치·형상 branch와 상관할 가능성이 있다. raw rank 우위는 raw descriptor의 차원·노이즈 효과일 수 있고, branch별 기여도는 문서에 없으므로 raw rank gap만으로 FC가 직교 정보를 읽는지 특정할 수 없다.
- **무엇이 참이면 이 주장이 무너지는가**: FC margin이 기존 7 branch와 max |r| > 0.6이거나, 8-branch에서 기존 효율 50.3% 비하락 조건인 eff.rank ≥ 4.02를 만족하지 못하면 branch 추가 방향은 gate ①에서 기각 또는 반박이다. raw rank gap이 차원·노이즈·정규화 차이로 설명되면, raw rank ≤ margin rank일 때 branch 추가를 중단하는 규칙은 근거가 없다. 이 경우 FC 후보는 성능 조회 없이 종료되거나, raw rank audit는 판별 불가 진단에 머문다.
- **이 반론의 한계**: FC가 CA-09에서 비저촉으로 명시된 구조적 변형이고, gate ①에서 max |r| ≤ 0.6이며 eff.rank 효율 비하락을 라벨 무관으로 통과하면, branch 추가 후보로 심사할 가치가 있다. 다만 gate ① 통과는 정보량·성능을 보장하지 않으므로 승격과는 별개다.

## 반론 5
- **공격 대상**: P1·P2 전체가 전제하는 “4 GPU-hour 안에서 fold time probe, provenance, rank audit, paired Δ 또는 aggregation 후보를 판별 가능하다”는 실행 가능성 주장.
- **더 단순한 설명**: 현재 병목이 새 모델 축이 아니라 계측 복구와 fold time 미측정일 수 있다. 상태 문서상 현재 목표는 계측 복구(provenance writer, SH/SJ control 경로, 오염 검사 재설계, 단위 테스트)이고, 복구 전에는 게이트·승격 비교를 verified로 수행할 수 없다. 또한 여섯 실행 러너가 전부 같은 하나의 7-branch 설정으로 수렴해 과거 branch 조합 비교가 구성 차이가 없는 실행 비교였으며, 저장 예측 파일에는 branch list·가중치·코드 해시·fold manifest 같은 provenance가 없다. 따라서 새 실행도 provenance를 저장하지 않으면 같은 실행 무효가 반복된다. fold fit 1회 시간은 문서에 없으므로, 4 GPU-hour 안에 350-fold baseline+후보 비교가 가능하다는 주장은 아직 근거가 없다.
- **무엇이 참이면 이 주장이 무너지는가**: 저장 예측을 verified baseline으로 쓸 수 없다면 baseline과 후보를 모두 재적합해야 한다. 4 GPU-hour는 240 GPU-min이고, fold-fit만 GPU-time이라면 baseline+후보 R=1은 350+350=700 fold-fit이 240 GPU-min 안에 들어가야 하므로 1-fold 평균 GPU-time이 240/700분 이하여야 한다. R>1이면 700R fold-fit이 필요해 더 짧아야 한다. fold-fit의 GPU-time 점유가 문서에 없거나, 1-fold timing probe가 이 조건을 통과하지 못하면 full paired Δ는 실행 무효다. provenance writer·branch control·오염 검사 재설계가 검증되지 않으면, 어떤 Δ도 unverified-hold 또는 실행 무효다.
- **이 반론의 한계**: 1-fold timing probe가 4 GPU-hour 안에서 baseline+후보 R≥1 재적합을 허용하고, provenance(branch list·가중치·코드 해시·fold manifest·seed/draw)가 검증되며, 오염 검사 재설계가 통과하면, 제한된 paired Δ 또는 rank audit는 실행 가능할 수 있다.

### A1 (감사)

## 점검 1: provenance·arm routing 복구 전 실행
- **확인하려는 것**: baseline·후보 margin/AUROC 산출물에 branch list, 가중치, 코드 해시, fold manifest 해시, seed/draw가 기록되고, aggregation·branch 설정이 러너 폴백으로 같은 7-branch로 수렴하지 않는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 서로 다른 arm이 같은 7-branch margin을 내고, 산출물을 특정 구성에 귀속시킬 수 없어 paired Δ·게이트·승격이 실행 무효가 된다.
- **확인 방법**: `docs/PROJECT.md` §3.2·§3.4의 `D-050`과 `docs/current_status.md`의 현재 목표(계측 복구)를 대조한다. `scripts/eval_v121.sh`, `scripts/eval_seal_tasks.sh`, `configs/baseline/v121_active.yaml`, `evaluate_pure.py`에서 설정 전달과 폴백 경로를 확인한다. 합성 데이터에서 aggregation 또는 branch weight를 바꿔 margin `max|diff|`가 0이 아닌지 확인하고, 산출물에 provenance 필드와 fold manifest 해시가 존재하는지 확인한다.

## 점검 2: baseline per-fold 검증
- **확인하려는 것**: paired Δ의 기준선 `0.6227`이 `unverified-hold`가 아니라, 같은 fold manifest·코드·구성에서 재계산된 per-fold baseline AUROC/margin으로 대체되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: macro `0.6227`만 있고 per-fold baseline이 없거나 provenance가 없으면, 후보−기준선 짝짓기가 불가능하거나 잘못된 baseline과 비교되어 승격 근거가 실행 무효가 된다.
- **확인 방법**: `docs/PROJECT.md` §3.2에서 `0.6227`이 `unverified-hold`임을 확인한다. 저장된 7-branch per-fold 예측/margin 파일의 코드 해시·branch list·fold manifest 해시를 확인한다. 원시 margin에서 macro를 독립 재계산해 `0.6227` 또는 재현값 `0.6226`과 대조한다. 재계산이 불가하면 baseline 재생성 전에 paired Δ를 수행하지 않는다.

## 점검 3: fold fit time·4 GPU-hour feasibility
- **확인하려는 것**: fold 1회 적합 시간이 실측되고, 그 시간으로 baseline+후보의 350-fold·R-seed 재적합이 4 GPU-hour 안에 가능한지 사전 판정되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: fold 시간이 4 GPU-hour를 초과하면 R=1 또는 일부 fold만 가능해 적합 분산 추정이 불가능하고, paired Δ probe가 실행 무효로 전환된다.
- **확인 방법**: `docs/current_status.md`의 예산 4 GPU-hour와 fold 적합 시간 미측정 사실을 확인한다. Primary 7 고정 fold 1개에서 baseline/candidate 1-fold timing probe를 GPU 4로 실행하고 wall time·GPU time·RAM·OOM을 기록한다. 기록된 시간으로 350×R 가능 여부를 계산하고, SEAL 10 경로가 호출되지 않는지 명령 로그를 확인한다. 단일 fold 시간이 350-fold 전체를 대표하지 못할 수 있으므로, 예산이 허용하면 과제별 1-fold 또는 소수 fold로 편차를 확인한다.

## 점검 4: fit-averaged paired Δ 승격 estimand 승인
- **확인하려는 것**: 단일 적합 Δ_macro가 아니라 fold×seed 기대 Δ를 승격 판정에 쓰는 것이 사용자 승인된 절차인가, `+0.0030` 임계가 그 estimand에 그대로 적용되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 미승인 estimand로 후보를 승격하거나, 단일 적합 기준과 fit-averaged 기준을 혼용하면 승격 판정이 실행 무효 또는 선택 편향이 된다.
- **확인 방법**: `docs/PROJECT.md` §4의 승격 공식, `D-048`의 적합 분산 보고 의무, `docs/current_status.md`의 “적합 분산을 포함한 승격 판정 절차가 미승인” 사실을 확인한다. 회차 카드에 estimand, R, seed pairing, best-seed 금지, 임계 적용 방식이 사전 등록되어 있는지 확인한다. 사용자 승인 기록이 없으면 fit-averaged paired Δ는 probe로만 한정한다.

## 점검 5: same-seed pairing의 실제 무작위 원천
- **확인하려는 것**: 현재 7-branch에서 seed/draw가 실제로 margin을 바꾸는 무작위 원천이고, baseline·후보가 같은 fold·seed·draw·fold 처리 순서로 짝지어지는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: seed가 closed-form ridge·PCA·subsample 중 어떤 단에만 작용하거나, 후보/기준의 draw가 다르면 seed main이 상쇄되지 않아 paired Δ의 SE 감소가 성립하지 않는다.
- **확인 방법**: `src/models/` branch 구현에서 random seed, subsampling, PCA/whitening, ridge solver의 무작위성을 확인한다. 같은 fold에서 seed만 바꿔 margin/AUROC `max|diff|`를 실측한다. seed/draw가 provenance에 기록되는지 확인한다. RU-97 GF의 seed std `0.0021~0.0083`을 현재 7-branch에 전이하지 않고, 현재 구성에서 재측정한다.

## 점검 6: RU-97 fold×seed 잔차 전이
- **확인하려는 것**: RU-97 `m32_raw`의 fold×seed 잔차 `23.9%`가 현재 7-branch paired Δ에도 동일하게 적용되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: seed 평균의 상한을 RU-97 구성 값으로 과대/과소 평가해 R을 잘못 정하고, `+0.0030` 판별이 판별 불가로 전환된다.
- **확인 방법**: `docs/current_status.md`에서 RU-97 분산분해가 GF 구성 공간의 실측임을 확인한다. 현재 7-branch에서 소수 fold×seed subset으로 variance decomposition을 수행하고, seed 주효과·fold×seed 잔차·fold 효과를 분리한다. 전체 350-fold 전에 이 subset 결과로 R과 예산을 재판정한다.

## 점검 7: rank/quantile 변환의 fit 범위
- **확인하려는 것**: P1의 per-fold query slide rank와 P2의 context-quantile 중 어떤 변환을 쓰는지, 그 변환이 query/test set에 fit되는가, 라벨을 쓰지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: query slide rank는 평가 집합의 구성·크기에 따라 branch 상대 가중이 바뀌어 fold 간 비교가 오염되고, context-quantile은 context에 query가 섞이면 train-test leakage가 된다.
- **확인 방법**: 제안서와 코드에서 rank/quantile의 fit set을 확인한다. query slide가 fit set에 포함되지 않는지, context slide가 같은 fold train만인지 fold manifest로 확인한다. 라벨 컬럼이 변환에 사용되지 않는지 코드 경로를 확인한다. query 순서만 바꿨을 때 context-quantile 결과가 불변인지, query-rank 결과는 query 집합에 의존하는지 확인한다.

## 점검 8: context-quantile의 context-query 분포 이대
- **확인하려는 것**: context slide margin 경험분포로 query margin을 quantile 매핑할 때, context slide 수 `90~261`에서 quantile 추정이 안정하고 context-query 분포 이대로 ordering을 깨지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: context 분포가 query와 이대이면 quantile이 노이즈가 되어 ensemble ordering이 불안정해지고, paired Δ가 `+0.0030`에 미달하거나 fold 간 변동만 커진다. 기전 미확인.
- **확인 방법**: `docs/closed_axes.md` CA-07의 context↔query 분포 이동 미해결 사실을 확인한다. Primary 7 일부 fold에서 context/query margin 분포의 overlap·quantile bootstrap 변동·fold 간 quantile mapping 안정성을 라벨 무관으로 측정한다. context slide 수 `90~261` 범위에서 최소 context 수 임계와 실패 조건을 사전 등록한다.

## 점검 9: raw descriptor effective rank와 margin effective rank의 비교 가능성
- **확인하려는 것**: raw slide descriptor의 effective rank와 7 branch margin effective rank `3.52/7`이 같은 estimator·차원 정규화·표본 수·라벨 무관 조건에서 계산되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: raw descriptor가 고차원·노이즈·과제 난이도로 effective rank가 높게 나와, branch 추가가 열렸다고 오판정하거나, margin rank만 낮다고 upstream 정보 부재를 오판정한다.
- **확인 방법**: `docs/PROJECT.md` §5의 effective rank 효율 `50.3%`와 8-branch 통과선 eff.rank `≥ 4.02`를 확인한다. `scripts/analysis/branch_screen.py`의 effective rank 정의·정규화·fold 단위를 확인한다. raw slide descriptor가 저장되어 있는지, 없으면 Primary 7 고정 fold에서 label-free로 재계산한다. raw/margin rank를 같은 slide·fold·estimator로 계산하고, 차원·표본 수 보정 또는 bootstrap 변동을 보고한다.

## 점검 10: FC gate 실행 경로와 선택 이력
- **확인하려는 것**: raw rank audit 결과로 FC를 선택한 뒤 gate ①을 수행할 때, SCREEN_ONLY·오염 검사·provenance가 복구된 상태이고, FC margin이 라벨 무관이며, 성능 조회 전에 종료되는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: `D-050` 이후 SCREEN_ONLY 경로가 무효인 상태에서 FC gate를 돌리거나, Primary 7에서 여러 후보를 보고 FC를 고른 뒤 성능을 조회하면 CA-08 선택 편향으로 실행 무효가 된다.
- **확인 방법**: `docs/closed_axes.md` CA-09의 FC 비저촉, §3-I의 SCREEN_ONLY 무효, `docs/PROJECT.md` §5의 gate ① 기준을 확인한다. 회차 카드에 FC 후보 선택 시점·raw rank audit 결과·gate ① 임계(`max |r| > 0.6`, eff.rank `≥ 4.02`)가 사전 등록되어 있는지 확인한다. `scripts/analysis/branch_screen.py`로 FC margin만 기록하고 AUROC를 조회하지 않는지 로그를 확인한다.

## 점검 11: Primary 7 내 다중 후보 탐색과 확인 분리
- **확인하려는 것**: rank audit, FC gate, aggregation 변형, fit-averaged probe를 같은 Primary 7에서 순차 수행할 때, 각 후보의 선택 이력이 기록되고, 같은 자료에서 고른 최댓값을 확증으로 쓰지 않는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: 여러 축을 Primary 7에서 시도한 뒤 가장 좋은 결과를 승격 근거로 쓰면 §214식 선택 편향이 재현되고, 모든 paired Δ가 실행 무효가 된다.
- **확인 방법**: `docs/PROJECT.md` §4.1의 탐색·확증 분리·선택 이력 기록, `docs/closed_axes.md` CA-08의 경계를 확인한다. 회차 카드에 각 후보의 사전 등록 순서, 데이터 접근 로그, 성능 조회 시점, 단일 후보 확정 규칙이 있는지 확인한다. 후보 변경이 Primary 7 성능 조회 이후에 발생했는지 결정 이력과 대조한다.

## 점검 12: SEAL 10 독립성
- **확인하려는 것**: 제안 실행의 모든 판정이 Primary 7과 라벨 무관 게이트에서만 나오는지, SEAL 10 결과·baseline CSV가 후보 선택·R 선택·aggregation 선택에 사용되지 않는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: SEAL 10 hold-out을 열람하거나 그 결과로 후보를 고르면 `D-047` 위반으로 최종 판정이 무효가 된다.
- **확인 방법**: `docs/PROJECT.md` §3.1과 `D-047`을 확인한다. 실행 명령·스크립트·로그에서 `seal_univ2_baseline_17tasks.csv`, SEAL 10 task fold, hold-out 평가 경로가 호출되지 않는지 확인한다. 산출물 경로에 Primary 7만 포함되는지, SEAL 10 결과가 기록되더라도 선택에 사용되지 않는지 선택 이력과 대조한다.

## 점검 13: 기존 전처리·ridge fit 범위
- **확인하려는 것**: PCA-32, whitening, closed-form ridge, branch margin 생성이 context/train fold에만 fit되고, query/test/SEAL 10을 fit set에 쓰지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: query/test patch feature로 PCA/whitening/ridge를 fit하면 train-test leakage가 발생하고, Primary 7 paired Δ가 과대평가되어 SEAL 10에서 재현되지 않는다.
- **확인 방법**: `src/models/` branch 구현에서 PCA/whitening/ridge fit 호출의 입력 슬라이드 집합을 확인한다. fold manifest와 context/query 분리를 확인한다. baseline/candidate가 같은 fit set을 쓰는지 확인한다. SEAL 10 슬라이드가 fit set에 포함되지 않는지 확인한다.

## 점검 14: baseline 선택 경로의 SEAL 10 독립성
- **확인하려는 것**: baseline 7-branch `0.6227`이 Primary 7 승격으로 고정된 것이지, SEAL 10 결과를 보고 고른 것이 아닌가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: baseline이 SEAL 10에 맞춰 선택됐다면, 후보와 baseline 비교 자체가 hold-out 오염으로 무효가 된다.
- **확인 방법**: `docs/PROJECT.md` §3.2에서 `D-042` 승격 근거가 Primary 7 대응 Δ `+0.56%p`임을 확인한다. `D-047` 이후 SEAL 10 선택 금지 사실을 확인한다. baseline 고정 시점의 결정 이력과 SEAL 10 열람 기록을 대조한다.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

- **T1-1 · 이론적 병목 1**: 수용(수정) — (a)~(d) 지적은 문서 근거가 있으므로 P 제안의 인과적·충분성 표현을 완화한다. raw/margin rank gap은 “측정 중복/정보 부재를 인과적으로 가른다”가 아니라, 동일 estimator·스케일·정규화·표본 조건에서의 라벨 무관 진단 신호로 한정한다. same-seed fit averaging은 `+0.0030` 판별을 보장하는 것이 아니라 SE를 줄일 수 있는 경로다. rank/quantile 변환은 “scale nuisance 제거”가 아니라 “branch 재가중/분포 변환 후보”로 명시한다. provenance는 구성 귀속의 필요 조건이지 충분 조건이 아니다. 다만 (e) “Primary 7 Δ가 SEAL 10 ABMIL 초과로 전이되는 선택 신호”는 P 제안이 둔 전제가 아니다. P1-1은 “모든 판정은 Primary 7 기반이고 hold-out 미검증”이라 쓰고, P2-1은 “SEAL 10은 열지 않는다”고 쓰며, `PROJECT.md §1`의 “Primary 7의 개선이 hold-out 개선을 보장하지 않는다”를 그대로 따른다.

- **M1-1 · P1 제안 1 — 4 GPU-hour feasibility · effective-rank gate**: 수용(수정) — fold fit time 미실측, provenance 없음, raw rank 정의 미확정, `3.52/7`이 `unverified-hold`라는 지적은 `PROJECT.md §3.2`·`D-050`·`current_status.md`와 일치한다. P1-1을 (1) 1-fold timing probe, (2) provenance/arm routing 복구 확인, (3) raw/margin rank audit의 3단계로 분리한다. raw rank gap은 동일 estimator·스케일·정규화·표본 조건이 확정되기 전에는 판별 불가로 둔다. `3.52/7`은 verified가 아닌 `unverified-hold` 계측으로 인용한다.

- **M1-2 · P1 제안 2 — fit-averaged paired Δ (same-seed pairing)**: 수용(수정) — 적합 분산 포함 승격 절차가 미승인이라는 지적은 `current_status.md`와 일치한다. P1-2의 fit-averaged paired Δ를 승격 estimand로 쓰지 않고, 사용자 승인 전에는 Primary 7 탐색·측정 probe로만 한정한다. same-seed pairing은 seed·draw·RNG state·fold 처리 순서가 코드 수준에서 동일함을 보장할 때만 유효하며, 그렇지 않으면 실행 무효다. RU-97의 `23.9%` fold×seed 잔차는 GF 구성 실측이므로 현재 7-branch에 전이하지 않고, 소수 fold×seed subset으로 재측정한다.

- **M1-3 · P1 제안 3 — per-fold rank aggregation**: 수용(수정) — CA-08 경계, scale 불일치 미측정, query slide 수 미확정, provenance 부재 지적은 타당하다. P1-3을 단일 사전 등록 변형으로 유지하되, 결과 조회 전에 (1) branch margin scale 편차의 라벨 무관 진단, (2) query slide 수·분포 확인, (3) verified margin 확보 또는 재적합 계획을 기록한다. rank 변환이 단독 branch AUROC를 보존하는 것은 rank 불변성으로 성립하지만, ensemble paired Δ 개선은 기전 미확인이므로 `+0.0030` 미달이면 반박으로 종료한다.

- **M1-4 · P2 제안 1 — common-random-number paired-Δ probe**: 수용(수정) — P2-1은 P1-2와 동일한 비용 구조·선행 조건을 가지며, 추가 가치는 1-fold timing probe를 먼저 둔다. `current_status.md`의 현재 목표가 계측 복구이므로, provenance writer·arm routing·오염 검사 재설계·단위 테스트가 검증되기 전에는 verified 게이트·승격 비교를 수행하지 않는다. 1-fold probe가 `700Rt > 4 GPU-hour`를 초과하면 해당 R의 full 판별은 실행 무효다. R=1만 가능하면 적합 분산 추정 불가로 판별 불가다.

- **M1-5 · P2 제안 2 — context-quantile trimmed mean**: 수용(수정) — CA-07에서 context↔query 분포 이동이 실측으로 기록됐으므로, P2-2의 context-quantile은 “scale 보정”이 아니라 분포 이동 노이즈를 도입할 수 있는 후보로 강등한다. context-query overlap·quantile bootstrap 변동·최소 context 수 임계를 라벨 무관으로 사전 정의하지 못하면 실행하지 않는다. P1-3의 fold·task 내 query rank와 P2-2의 context-quantile은 둘 다 쓰지 않고, 사전 등록된 단일 규칙 하나만 검증한다.

- **M1-6 · P2 제안 3 — raw-descriptor rank audit + FC gate**: 수용(수정) — FC gate는 SCREEN_ONLY·오염 검사·provenance 복구 전에는 실행할 수 없다. raw rank > margin rank는 FC가 직교 정보를 읽는 보장이 아니라, FC를 게이트 ①에 올리는 라벨 무관 신호일 뿐이다. `3.52/7`과 8-branch 통과선 `eff.rank ≥ 4.02`는 `unverified-hold` 계측에 의존하므로, verified margin·provenance가 복구된 뒤에만 게이트 ①을 수행한다. FC 구현·margin 산출 비용이 4 GPU-hour를 초과하면 실행 무효다.

- **R1-1 · raw/margin effective rank gap**: 수용(수정) — raw rank > margin rank는 “compression/whitening/readout 손실”을 인과적으로 구분하지 못한다. raw와 margin이 차원·스케일·노이즈·정규화·표본 수에서 다르면 gap은 측정 산출물일 수 있다. P 제안에서 “가른다”는 표현을 삭제하고, raw/margin rank gap은 동일 fold·과제·슬라이드 집합, 동일 label-free rank estimator, 동일 스케일/정규화, 동일 차원/노이즈 대조 조건에서만 “투영 손실 후보를 가리키는 진단”으로 쓴다. 그 진단이 paired Δ `+0.0030`을 낳는지는 별도 검증이 필요하고, rank gap 자체는 성능 기전이 아니다.

- **R1-2 · same-seed pairing과 fit averaging**: 수용(수정) — same-seed pairing은 baseline과 후보가 같은 random number를 같은 순서로 소비할 때만 seed main을 상쇄한다. draw·RNG state·fold order·subsample 순서가 다르면 상쇄되지 않는다. RU-97의 `23.9%` fold×seed 잔차는 GF 구성 실측이므로 현재 7-branch에 전이 근거가 없다. P 제안은 same-seed pairing을 “SE 감소 경로”로만 주장하고, `+0.0030` 판별을 보장하지 않는다. 적합 분산 포함 승격 절차가 미승인인 상태에서는 probe로만 한정한다.

- **R1-3 · rank/quantile 변환의 해석**: 수용(수정) — rank/quantile 변환은 “scale nuisance 제거”가 아니라 “branch effective weight 변경”으로 읽힐 수 있다. margin magnitude에 신호가 있으면 정보를 버린다. P 제안에서 rank/quantile을 “scale 불일치 제거”가 아니라 “branch 재가중/분포 변환 후보”로 재명명한다. 실행 전 라벨 무관 진단으로 (1) branch margin scale 편차가 task signal보다 지배적인지, (2) context-query 분포 이대가 quantile을 왜곡하지 않는지 확인한다. 둘 중 하나를 확인하지 못하면 판별 불가로 둔다.

- **R1-4 · `3.52/7`과 FC 후보**: 수용(수정) — `3.52/7`은 branch 부족이라는 결론을 보장하지 않으며, 같은 slide-level descriptor 계열의 중복 사영일 수 있다. FC도 같은 patch feature/slide descriptor에서 나오므로 기존 branch와 상관할 수 있다. P 제안에서 “raw rank ≤ margin rank이면 branch 추가를 중단한다”는 규칙을 삭제한다. raw rank audit는 branch 추가를 중단하는 규칙이 아니라, FC 등 구조적 후보의 우선순위를 낮추거나 높이는 라벨 무관 신호로만 쓴다. 최종 기각은 게이트 ①의 `max |r| > 0.6` 또는 `eff.rank` 효율 하락으로만 한다.

- **R1-5 · 4 GPU-hour 실행 가능성**: 수용(수정) — 현재 병목이 새 모델 축이 아니라 계측 복구·fold time 미측정이라는 지적은 `current_status.md`·`D-050`과 일치한다. P 제안의 모든 4 GPU-hour 실행은 1-fold timing probe와 provenance/arm routing 복구 확인을 선행으로 둔다. 저장 예측을 verified baseline으로 쓸 수 없으면 baseline과 후보를 모두 재적합해야 하며, 그 비용이 4 GPU-hour를 초과하면 실행 무효다. `700R` fold-fit이 4 GPU-hour 안에 들어가는지 fold time 실측으로만 판정하고, fold time 미측정 상태에서는 “가능하다”고 주장하지 않는다.

- **A1-1 · provenance·arm routing 복구 전 실행**: 수용(수정) — provenance·arm routing 복구 전 실행은 실행 무효다. `D-050`에서 여섯 러너가 폴백으로 같은 7-branch로 수렴했고, 저장 예측 파일에 provenance가 없어 과거 산출물을 구성에 귀속시킬 수 없다. P 제안의 모든 실행은 산출물에 branch list·가중치·코드 해시·fold manifest 해시·seed/draw를 기록하고, 서로 다른 arm이 같은 margin을 내지 않는지 확인한 뒤에만 paired Δ·게이트·승격을 수행한다.

- **A1-2 · baseline per-fold 검증**: 수용(수정) — baseline `0.6227`은 `unverified-hold`다. paired Δ의 기준선은 같은 fold manifest·코드·구성에서 재계산된 per-fold baseline AUROC/margin으로 대체해야 한다. 저장된 per-fold 예측/margin에 provenance가 없으면, baseline 재생성 또는 verified per-fold baseline 확보 전에는 후보−기준선 짝짓기를 수행하지 않는다.

- **A1-3 · fold fit time·4 GPU-hour feasibility**: 판별 불가(수정) — fold fit 1회 시간은 문서에 없다. 4 GPU-hour 안에 350-fold·R-seed 재적합이 가능하다는 주장은 근거가 없다. P 제안은 1-fold timing probe를 첫 실행으로 고정하고, wall time·GPU time·RAM·OOM을 기록한다. 단일 fold가 전체를 대표하지 못할 수 있으므로, 예산이 허용하면 과제별 1-fold 또는 소수 fold로 편차를 확인한다. probe가 4 GPU-hour 조건을 통과하지 못하면 full paired Δ는 실행 무효로 전환된다.

- **A1-4 · fit-averaged paired Δ 승격 estimand 승인**: 수용(수정) — 적합 분산 포함 승격 판정 절차가 미승인이다. P 제안의 fit-averaged paired Δ를 승격 estimand로 바로 쓰지 않는다. 사용자 승인 전에는 Primary 7 탐색·측정 probe로만 한정하고, 회차 카드에 estimand·R·seed pairing·best-seed 금지·임계 적용 방식을 사전 등록한다. 승인 없이는 `+0.0030` 승격 판정을 내리지 않는다.

- **A1-5 · same-seed pairing의 실제 무작위 원천**: 판별 불가(수정) — 현재 7-branch에서 seed/draw가 실제로 margin을 바꾸는 무작위 원천인지, baseline·후보가 같은 fold·seed·draw·fold 처리 순서로 짝지어지는지는 문서에 없다. P 제안은 같은 fold에서 seed만 바꿔 margin/AUROC `max|diff|`를 실측하고, seed/draw/RNG state를 provenance에 기록한다. 이 확인이 없으면 same-seed pairing의 SE 감소 주장은 판별 불가로 둔다.

- **A1-6 · RU-97 fold×seed 잔차 전이**: 판별 불가(수정) — RU-97 `m32_raw`의 fold×seed 잔차 `23.9%`가 현재 7-branch에도 적용되는지는 문서에 없다. P 제안은 RU-97 값을 현재 7-branch의 상한으로 전이하지 않는다. 전체 350-fold 전에 소수 fold×seed subset으로 seed 주효과·fold×seed 잔차·fold 효과를 분리하고, 그 결과로 R과 예산을 재판정한다. subset 결과가 없으면 R 선택은 판별 불가다.

- **A1-7 · rank/quantile 변환의 fit 범위**: 판별 불가(수정) — rank/quantile 변환의 fit set, query slide 포함 여부, 라벨 미사용 코드 경로는 문서에 없다. P 제안은 rank/quantile fit set을 context/train fold로 한정하고, query/test/SEAL 10을 fit set에 포함하지 않는다. query slide가 fit set에 포함되면 fold 간 비교가 오염되고, context에 query가 섞이면 train-test leakage가 된다. 이 확인이 없으면 rank/quantile 변환은 실행하지 않는다.

- **A1-8 · context-quantile의 context-query 분포 이대**: 판별 불가(수정) — context slide 수 `90~261`에서의 quantile 추정 안정성과 context-query 분포 이대 크기는 문서에 없다. CA-07에서 context↔query 분포 이동이 미해결로 기록됐으므로, context-quantile은 분포 이대 진단 전 실행하지 않는다. Primary 7 일부 fold에서 context/query margin 분포 overlap·quantile bootstrap 변동·fold 간 mapping 안정성을 라벨 무관으로 측정하고, 최소 context 수 임계를 사전 등록한다. 확인 전에는 P2-2는 보류다.

- **A1-9 · raw descriptor effective rank와 margin effective rank의 비교 가능성**: 판별 불가(수정) — raw slide descriptor와 7 branch margin의 effective rank가 같은 estimator·차원 정규화·표본 수·라벨 무관 조건에서 계산되는지는 문서에 없다. `3.52/7`은 margin effective rank의 `unverified-hold` 계측이다. P 제안은 raw/margin rank 비교를 동일 fold·과제·슬라이드 집합, 동일 estimator, 동일 스케일/정규화, 동일 표본 조건에서 수행하기 전에는 판별 불가로 둔다. raw descriptor가 저장되어 없으면 Primary 7 고정 fold에서 label-free로 재계산하고, 재계산 비용이 4 GPU-hour를 초과하면 실행 무효다.

- **A1-10 · FC gate 실행 경로와 선택 이력**: 수용(수정) — FC gate는 SCREEN_ONLY·오염 검사·provenance 복구 전에는 실행할 수 없다. `D-050` 이후 SCREEN_ONLY 경로가 무효인 상태에서 FC gate를 돌리면 실행 무효다. P 제안은 FC 후보 선택 시점·raw rank audit 결과·게이트 ① 임계(`max |r| > 0.6`, `eff.rank ≥ 4.02`)를 회차 카드에 사전 등록하고, `SCREEN_ONLY=1`로 FC margin만 기록한 뒤 AUROC를 조회하지 않는다.

- **A1-11 · Primary 7 내 다중 후보 탐색과 확인 분리**: 수용(수정) — Primary 7에서 rank audit·FC gate·aggregation 변형·fit-averaged probe를 순차 수행할 때, 같은 자료에서 고른 최댓값을 확증으로 쓰면 §214식 선택 편향이 재현된다. P 제안은 각 후보의 사전 등록 순서·데이터 접근 로그·성능 조회 시점·단일 후보 확정 규칙을 기록한다. 후보 변경이 Primary 7 성능 조회 이후에 발생하면 해당 후보의 Δ는 실행 무효로 처리한다.

- **A1-12 · SEAL 10 독립성**: 수용(수정) — SEAL 10 독립성은 `D-047`과 `PROJECT.md §3.1`에 따라 유지된다. P 제안의 모든 실행은 SEAL 10 task fold·baseline CSV·hold-out 평가 경로를 호출하지 않는다. SEAL 10 결과가 기록되더라도 후보 선택·R 선택·aggregation 선택에 사용하지 않는다. 실행 명령·스크립트·로그에서 SEAL 10 경로가 호출되지 않는지 확인하고, 선택 이력과 대조한다.

- **A1-13 · 기존 전처리·ridge fit 범위**: 판별 불가(수정) — PCA-32·whitening·closed-form ridge·branch margin 생성의 fit set이 context/train fold인지, query/test/SEAL 10을 포함하지 않는지는 문서에 없다. P 제안은 fold manifest와 context/query 분리를 확인하고, baseline/candidate가 같은 fit set을 쓰는지 확인한다. query/test/SEAL 10이 fit set에 포함되면 train-test leakage로 Primary 7 paired Δ가 과대평가될 수 있으며, 이 확인이 없으면 paired Δ는 실행 무효로 둔다.

- **A1-14 · baseline 선택 경로의 SEAL 10 독립성**: 반박 — A1-14는 baseline `0.6227`이 SEAL 10 결과를 보고 고르면 무효라고 점검하지만, `PROJECT.md §3.2`는 `D-042` 승격 근거가 Primary 7 대응 Δ `+0.56%p`임을 확인하고, `D-047` 이후 SEAL 10 선택 금지를 확인한다. 따라서 P 제안이 baseline 선택 경로를 SEAL 10에 의존한다는 반론은 문서 근거가 없다. P 제안은 baseline을 `unverified-hold` 상태로 재확인해야 한다는 A1-2와 연결하되, baseline 선택 자체가 SEAL 10 오염이라는 주장은 반박한다.

### P2 (제안)

## T1
- **판정**: 수용·수정
- **근거/수정**: T1이 지적한 fold 적합 시간 미실측, provenance 미복구, 적합 분산 승격 절차 미승인, RU-97 fold×seed 잔차의 전이 불가, Primary 7→SEAL 10 전이 미보장은 모두 문서에 근거가 있으므로 수용한다. 수정 내용: (a) raw/margin effective-rank gap은 “측정 중복 vs 정보 부재”를 인과적으로 구분하는 것이 아니라, 동일 estimator·차원·스케일·표본 조건에서 대조 가능한가 여부를 먼저 확인하는 **가설 생성 단계**로 낮춘다. (b) fit-averaged paired Δ는 승격 estimand가 아니라, 승격 절차 승인 전에는 **probe/탐색 기록**으로만 쓴다. (c) 1-fold timing probe는 4 GPU-hour 내 `700R` fold-fit 가능 여부를 산술적으로 판단하는 데만 쓰고, fold time 실측 전에는 full paired Δ를 실행하지 않는다. (d) T1의 “CA-10이 (a)와 충돌한다”는 주장은 근거가 부족하다: CA-10은 형상통계 강화 변형 7종이 게이트 ②를 통과하지 못했다는 관측이지, raw/margin rank gap 진단 자체가 실패했다는 관측이 아니므로, 이 충돌만은 내 제안을 반박하지 못한다.

## M1-1
- **판정**: 수용·수정
- **근거/수정**: fold 적합 1회 시간 `t`가 미실측이고, `t`의 정의(데이터 로딩 포함/제외, GPU/CPU, I/O 포함 여부)도 미실측이므로, `700Rt ≤ 14400s`는 실측이 아니라 산술 조건식이다. 수정 내용: (a) 첫 실행을 1-fold timing probe로 고정하고, wall time·GPU time·RAM·OOM·I/O를 provenance와 함께 기록한다. (b) `t` 실측 전에는 baseline+후보 full 350-fold 재적합을 시작하지 않는다. (c) raw descriptor rank audit는 raw descriptor 저장 경로와 effective rank 정의(차원, 스케일, 정규화, slide 수, estimator)가 문서화되지 않으면 실행하지 않는다. (d) 3.52/7은 `unverified-hold` 계측이므로, provenance 복구 전에는 verified 기준값이 아니라 과거 기록값으로만 인용한다.

## M1-2
- **판정**: 수용·수정
- **근거/수정**: same-seed pairing이 seed main을 상쇄하는 수학적 주장은 유지하되, 같은 seed가 baseline과 후보의 동일 random draw·RNG consumption·fold 처리 순서를 보장하지는 못한다는 지적은 수용한다. 수정 내용: (a) seed·draw·fold order·RNG stream state를 코드 수준에서 고정하는 경로가 확인되지 않으면 paired Δ는 실행 무효로 종료한다. (b) 적합 분산 포함 승격 절차가 사용자 미승인 상태이므로, fit-averaged paired Δ는 승격 근거가 아니라 **Primary 7 탐색 기록**으로만 남긴다. (c) RU-97의 seed std `0.0021~0.0083`과 fold×seed 잔차 `23.9%`는 GF 구성 공간 실측이므로, 현재 7-branch에 전이하지 않고 현재 구성에서 소수 fold×seed subset으로 재측정한다. (d) 4 GPU-hour에서 R=1만 가능하면 적합 분산 추정이 불가하므로 판별 불가로 종료한다.

## M1-3
- **판정**: 수용·수정
- **근거/수정**: per-fold rank aggregation은 branch 수 7개 유지·학습 파라미터 0개를 유지하되, CA-08 경계를 벗어나려면 단일 사전 등록 변형으로 한정해야 한다. 수정 내용: (a) 결과 조회 전에 rank 변환 규칙, fit set( context/train fold만), 라벨 미사용 보장, 선택 이력을 사전 등록한다. (b) branch margin의 scale 불일치가 aggregate noise 원인이라는 진단이 실측되지 않았으므로, 이 제안의 기대 차이는 “scale nuisance 제거”가 아니라 “branch effective weight 균등화”로 재정의한다. (c) verified margin이 없으면 baseline+후보 재적합이 선행 조건이고, `t` 미실측으로 재적합 예산을 통과하지 못하면 실행 무효다. (d) 여러 rank/aggregation 변형을 Primary 7에서 비교한 뒤 최댓값을 쓰면 CA-08 선택 편향으로 실행 무효다.

## M1-4
- **판정**: 수용·수정
- **근거/수정**: common-random-number paired-Δ probe는 P1-2와 동일한 비용 구조와 전제를 가지며, 1-fold timing probe를 먼저 둔다는 점만 추가된다. 수정 내용: (a) 1-fold probe가 `700Rt ≤ 14400s`를 통과하지 못하면 해당 R의 full 판별은 실행 무효로 종료한다. (b) probe가 통과해도 same fold·seed·draw·RNG consumption 보장이 없으면 실행 무효다. (c) current_status의 현재 목표가 계측 복구(provenance writer, SH/SJ control, 오염 검사 재설계, 단위 테스트)이므로, 복구 전에는 게이트·승격 비교를 verified로 수행하지 않는다. (d) RU-97 fold×seed 잔차 `23.9%`를 현재 7-branch에 전이하는 근거가 없으므로, R 결정 전 현재 구성 variance decomposition을 소수 fold subset으로 수행한다.

## M1-5
- **판정**: 수용·수정
- **근거/수정**: context-quantile trimmed mean은 CA-07이 기록한 context↔query 분포 이동 문제를 직접 건드리므로, context 경험분포가 query margin을 안정적으로 대표한다는 전제는 미확인이다. 수정 내용: (a) context-query margin 분포 overlap, quantile bootstrap 변동, fold 간 mapping 안정성을 라벨 무관으로 사전 측정한다. (b) context slide 수 `90~261`에서 quantile 추정 노이즈와 context-query 분포 이대 크기는 문서에 없으므로, 최소 context 수 임계와 실패 조건을 사전 등록한다. (c) branch margin magnitude가 신호인지 노이즈인지 실측되지 않았으므로, quantile mapping은 “scale 보정”이 아니라 “rank-only 재가중”으로 재정의한다. (d) provenance가 없으면 어떤 Δ도 구성에 귀속 불가하므로 실행 무효다.

## M1-6
- **판정**: 수용·수정
- **근거/수정**: raw-descriptor rank audit + FC gate는 FC를 `CA-09` 경계 밖 후보로 취급하지만, raw rank gap이 FC가 읽을 수 있는 직교 정보를 보장하지는 못한다. 수정 내용: (a) 게이트 ① 실행 전 오염 검사 재설계와 provenance writer 복구가 선행 조건이다. (b) FC margin 산출은 `SCREEN_ONLY=1`로 앙상블에 넣지 않고, AUROC를 조회하지 않는다. (c) raw rank ≤ margin rank이면 branch 추가 후보를 중단하지만, 이 판정은 effective rank 정의(스케일·차원·slide 수·estimator)가 재현 가능할 때만 유효하다. (d) FC 구현·margin 산출 비용이 4 GPU-hour를 초과하면 실행 무효로 종료한다. (e) gate ① 통과는 정보량·성능을 보장하지 않으므로 승격과는 별개다.

## R1-1
- **판정**: 수용·수정
- **근거/수정**: raw rank > margin rank는 “아직 읽히지 않는 정보가 있다”보다 “raw와 margin이 다른 차원·노이즈·정규화·표본 조건에서 측정됐다”는 설명이 더 단순할 수 있다. 수정 내용: (a) raw descriptor와 margin effective rank는 같은 fold·과제·slide 집합, 같은 label-free estimator, 같은 스케일/정규화, 같은 차원/노이즈 대조 조건에서 측정되지 않으면 gap은 판별력이 없다. (b) raw rank audit는 인과 구분이 아니라, “compression/readout 손실 후보”와 “정보 부재 후보”를 좁히는 진단으로 한정한다. (c) 동일 조건 대조 후에도 raw rank 우위가 유지되면 compression/readout 변경 후보를 열 수 있지만, 그것이 paired Δ `+0.0030`을 낳는다는 것은 별도 검증이 필요하고 rank gap 자체는 성능 기전이 아니다.

## R1-2
- **판정**: 수용·수정
- **근거/수정**: seed std `0.0021~0.0083`이 독립 적합 노이즈가 아니라 random draw, fold 처리 순서, branch별 subsampling, RNG 스트림 소비 차이에서 나온 것일 수 있다는 지적은 수용한다. 수정 내용: (a) same-seed pairing은 baseline과 후보가 실제로 동일한 common random number를 같은 순서로 소비할 때만 seed main을 상쇄한다. (b) fold×seed 잔차 `23.9%`는 RU-97 GF 구성 실측이므로 현재 7-branch에 전이하지 않는다. (c) 4 GPU-hour에서 R=1만 가능하면 fit-averaged paired Δ는 `+0.0030`을 판별하지 못해 판별 불가다. (d) 적합 분산 포함 승격 절차 미승인 상태에서는 이 estimand를 승격 근거로 쓰지 않는다.

## R1-3
- **판정**: 수용·수정
- **근거/수정**: rank/quantile 변환은 scale bias 제거가 아니라 branch 재가중으로 설명될 수 있다. 수정 내용: (a) P1-3의 per-fold query slide rank와 P2-2의 context-quantile은 “scale nuisance 제거”가 아니라 “branch effective weight 균등화”로 재정의한다. (b) margin magnitude가 증거 강도·신뢰도 신호라면 rank/quantile 변환은 정보를 버린다. (c) context-quantile은 context margin 분포가 query margin 분포를 대표한다는 전제를 쓰는데, CA-07과 §215에서 context↔query 분포 이동이 문제화된 축이므로 이 전제는 미확인이다. (d) query slide 수가 문서에 없으므로 rank 안정성은 판별 불가다. (e) 단일 변형이라도 paired Δ `+0.0030`에 미달하면 반박이다.

## R1-4
- **판정**: 수용·수정
- **근거/수정**: 유효 랭크 3.52/7은 branch 종류 부족보다 7개 branch가 같은 slide-level descriptor 계열에서 나온 중복 사영이라는 설명이 더 단순할 수 있다. 수정 내용: (a) FC도 같은 patch feature/slide descriptor에서 나오므로 기존 위치·형상 branch와 상관할 가능성이 있다. (b) raw rank gap만으로 FC가 직교 정보를 읽는지 특정할 수 없으므로, raw rank audit는 FC 후보를 선택하는 신호일 뿐, gate ①과 승격은 별개다. (c) FC margin이 기존 7 branch와 `max |r| > 0.6`이거나, 8-branch에서 `eff.rank ≥ 4.02`를 만족하지 못하면 branch 추가 방향은 gate ①에서 기각 또는 반박이다. (d) raw rank gap이 차원·노이즈·정규화 차이로 설명되면, raw rank ≤ margin rank일 때 branch 추가를 중단하는 규칙은 근거가 없어 판별 불가로 전환된다.

## R1-5
- **판정**: 수용·수정
- **근거/수정**: 현재 병목이 새 모델 축이 아니라 계측 복구와 fold time 미측정일 수 있다는 지적은 수용한다. 수정 내용: (a) current_status의 현재 목표는 provenance writer, SH/SJ control 경로, 오염 검사 재설계, 단위 테스트 복구이며, 복구 전에는 게이트·승격 비교를 verified로 수행할 수 없다. (b) fold fit 1회 시간이 문서에 없으므로, 4 GPU-hour 안에 350-fold baseline+후보 비교가 가능하다는 주장은 근거가 없다. (c) `700R` fold-fit이 14400s 안에 들어가야 하므로, 1-fold timing probe가 통과하지 못하면 full paired Δ는 실행 무효다. (d) provenance writer·branch control·오염 검사 재설계가 검증되지 않으면, 어떤 Δ도 unverified-hold 또는 실행 무효다.

## A1-1
- **판정**: 수용·수정
- **근거/수정**: D-050에서 여섯 러너가 폴백으로 같은 7-branch 설정으로 수렴했고, 저장 예측 파일에 provenance가 없어 과거 산출물을 구성에 귀속시킬 수 없다는 관측은 유효하다. 수정 내용: (a) 모든 실행 산출물에 branch list, 가중치, 코드 해시, fold manifest 해시, seed/draw를 기록한다. (b) `scripts/eval_v121.sh`, `scripts/eval_seal_tasks.sh`, `configs/baseline/v121_active.yaml`, `evaluate_pure.py`에서 설정 전달과 폴백 경로를 확인한다. (c) 합성 데이터에서 aggregation 또는 branch weight를 바꿔 margin `max|diff|`가 0이 아닌지 확인한다. (d) provenance 필드와 fold manifest 해시가 존재하지 않으면 paired Δ·게이트·승격은 실행 무효다.

## A1-2
- **판정**: 수용·수정
- **근거/수정**: paired Δ의 기준선 `0.6227`이 `unverified-hold`라는 지적은 유효하다. 수정 내용: (a) paired Δ는 `0.6227` 절대값이 아니라, 같은 fold manifest·코드·구성에서 재계산된 per-fold baseline AUROC/margin과 후보의 짝짓기로만 수행한다. (b) 저장된 7-branch per-fold 예측/margin 파일의 코드 해시·branch list·fold manifest 해시를 확인한다. (c) 원시 margin에서 macro를 독립 재계산해 `0.6227` 또는 재현값 `0.6226`과 대조한다. (d) 재계산이 불가하면 baseline 재생성 전에 paired Δ를 수행하지 않는다.

## A1-3
- **판정**: 수용·수정
- **근거/수정**: fold 1회 적합 시간이 실측되지 않았으므로, 4 GPU-hour feasibility를 사전 판정할 수 없다는 지적은 유효하다. 수정 내용: (a) Primary 7 고정 fold 1개에서 baseline/candidate 1-fold timing probe를 GPU 4로 실행하고 wall time·GPU time·RAM·OOM을 기록한다. (b) 기록된 시간으로 `700R` fold-fit 가능 여부를 산술적으로 계산한다. (c) 단일 fold 시간이 350-fold 전체를 대표하지 못할 수 있으므로, 예산이 허용하면 과제별 1-fold 또는 소수 fold로 편차를 확인한다. (d) fold time 미실측 상태에서는 full paired Δ를 시작하지 않는다.

## A1-4
- **판정**: 수용·수정
- **근거/수정**: 적합 분산 포함 승격 판정 절차가 사용자 미승인 상태라는 지적은 유효하다. 수정 내용: (a) fit-averaged paired Δ는 승격 estimand가 아니라, 사용자 승인 전에는 probe/탐색 기록으로만 쓴다. (b) 회차 카드에 estimand, R, seed pairing, best-seed 금지, 임계 적용 방식을 사전 등록한다. (c) `+0.0030` 임계가 fit-averaged estimand에 그대로 적용되는지는 사용자 승인 전까지 미확정이다. (d) 승인 전 fit-averaged paired Δ를 승격 근거로 쓰면 실행 무효다.

## A1-5
- **판정**: 판별 불가
- **근거/수정**: 현재 7-branch에서 seed/draw가 실제로 margin을 바꾸는 무작위 원천인지, 그리고 baseline·후보가 같은 fold·seed·draw·fold 처리 순서로 짝지어지는지는 문서에 실측이 없다. RU-97의 seed std `0.0021~0.0083`은 GF 구성 공간 실측이지 현재 7-branch 실측이 아니다. 따라서 same-seed pairing의 실제 효과는 현재 문서로 판별 불가다. 실행하려면 `src/models/` branch 구현에서 random seed, subsampling, PCA/whitening, ridge solver의 무작위성을 확인하고, 같은 fold에서 seed만 바꿔 margin/AUROC `max|diff|`를 실측해야 한다.

## A1-6
- **판정**: 판별 불가
- **근거/수정**: RU-97 `m32_raw`의 fold×seed 잔차 `23.9%`가 현재 7-branch paired Δ에도 동일하게 적용되는지는 문서에 실측이 없다. RU-97은 GF 구성 공간(`M ∈ {8,16,32}` × 화이트닝 × 표본)의 실측이며, 현재 7-branch의 공통 상한이 아니다. 따라서 seed 평균의 상한을 RU-97 값으로 전이하는 것은 근거가 부족하다. 현재 7-branch에서 소수 fold×seed subset으로 variance decomposition을 수행해야 판별 가능하다.

## A1-7
- **판정**: 수용·수정
- **근거/수정**: rank/quantile 변환의 fit 범위가 제안서에 명시되지 않았다는 지적은 유효하다. 수정 내용: (a) P1-3의 per-fold query slide rank는 fold·task 내 query slide 순서만 사용하지만, query/test set 구성·크기에 따라 branch 상대 가중이 바뀔 수 있으므로 fit set을 명시한다. (b) P2-2의 context-quantile은 context slide margin 경험분포만 fit set으로 쓰고, query/test slide는 transform에만 사용한다. (c) context slide가 같은 fold train만인지 fold manifest로 확인한다. (d) 라벨 컬럼이 변환에 사용되지 않는지 코드 경로를 확인한다. (e) query 순서만 바꿨을 때 context-quantile 결과가 불변인지, query-rank 결과는 query 집합에 의존하는지 확인한다.

## A1-8
- **판정**: 판별 불가
- **근거/수정**: context slide 수 `90~261`에서 context margin 경험분포로 query margin을 quantile 매핑할 때, quantile 추정이 안정하고 context-query 분포 이대로 ordering을 깨지 않는지는 문서에 실측이 없다. CA-07에서 context↔query 분포 이동이 문제화된 축으로 기록되어 있지만, quantile 추정 노이즈와 분포 이대 크기는 미실측이다. 따라서 context-quantile의 안정성은 현재 문서로 판별 불가다. 실행하려면 Primary 7 일부 fold에서 context/query margin 분포 overlap, quantile bootstrap 변동, fold 간 mapping 안정성을 라벨 무관으로 측정해야 한다.

## A1-9
- **판정**: 판별 불가
- **근거/수정**: raw slide descriptor의 effective rank와 7 branch margin effective rank `3.52/7`이 같은 estimator·차원 정규화·표본 수·라벨 무관 조건에서 계산되는지는 문서에 실측이 없다. raw descriptor 저장 경로, effective rank 정의, 차원·스케일·정규화 조건이 문서에 없으므로, raw/margin rank 비교 가능 여부는 현재 문서로 판별 불가다. 실행하려면 `scripts/analysis/branch_screen.py`의 effective rank 정의·정규화·fold 단위를 확인하고, raw slide descriptor가 저장되어 있는지, 없으면 Primary 7 고정 fold에서 label-free로 재계산해야 한다.

## A1-10
- **판정**: 수용·수정
- **근거/수정**: FC gate 실행 경로와 선택 이력 지적은 유효하다. 수정 내용: (a) `D-050` 이후 SCREEN_ONLY 경로가 무효인 상태에서 FC gate를 돌리지 않는다. (b) 회차 카드에 FC 후보 선택 시점, raw rank audit 결과, gate ① 임계(`max |r| > 0.6`, eff.rank `≥ 4.02`)를 사전 등록한다. (c) `scripts/analysis/branch_screen.py`로 FC margin만 기록하고 AUROC를 조회하지 않는지 로그를 확인한다. (d) Primary 7에서 여러 후보를 보고 FC를 고른 뒤 성능을 조회하면 CA-08 선택 편향으로 실행 무효다.

## A1-11
- **판정**: 수용·수정
- **근거/수정**: Primary 7 내 다중 후보 탐색과 확인 분리 지적은 유효하다. 수정 내용: (a) rank audit, FC gate, aggregation 변형, fit-averaged probe를 같은 Primary 7에서 순차 수행할 때, 각 후보의 선택 이력을 기록한다. (b) 회차 카드에 각 후보의 사전 등록 순서, 데이터 접근 로그, 성능 조회 시점, 단일 후보 확정 규칙을 명시한다. (c) 후보 변경이 Primary 7 성능 조회 이후에 발생했는지 결정 이력과 대조한다. (d) 같은 자료에서 고른 최댓값을 확증으로 쓰면 §214식 선택 편향이 재현되어 모든 paired Δ가 실행 무효다.

## A1-12
- **판정**: 수용·수정
- **근거/수정**: SEAL 10 독립성 지적은 유효하다. 수정 내용: (a) 제안 실행의 모든 판정은 Primary 7과 라벨 무관 게이트에서만 나온다. (b) SEAL 10 결과·baseline CSV가 후보 선택·R 선택·aggregation 선택에 사용되지 않는다. (c) 실행 명령·스크립트·로그에서 `seal_univ2_baseline_17tasks.csv`, SEAL 10 task fold, hold-out 평가 경로가 호출되지 않는지 확인한다. (d) 산출물 경로에 Primary 7만 포함되는지 확인한다. (e) SEAL 10 결과가 기록되더라도 선택에 사용되지 않는지 선택 이력과 대조한다.

## A1-13
- **판정**: 판별 불가
- **근거/수정**: PCA-32, whitening, closed-form ridge, branch margin 생성이 context/train fold에만 fit되고, query/test/SEAL 10을 fit set에 쓰지 않는지는 문서에 실측이 없다. 따라서 train-test leakage 여부는 현재 문서로 판별 불가다. 실행하려면 `src/models/` branch 구현에서 PCA/whitening/ridge fit 호출의 입력 슬라이드 집합을 확인하고, fold manifest와 context/query 분리를 확인해야 한다.

## A1-14
- **판정**: 수용·수정
- **근거/수정**: baseline 선택 경로의 SEAL 10 독립성 지적은 유효하다. 수정 내용: (a) `PROJECT.md` §3.2에서 `D-042` 승격 근거가 Primary 7 대응 Δ `+0.56%p`임을 확인한다. (b) `D-047` 이후 SEAL 10 선택 금지 사실을 확인한다. (c) baseline 고정 시점의 결정 이력과 SEAL 10 열람 기록을 대조한다. (d) baseline이 SEAL 10에 맞춰 선택됐다는 증거는 문서에 없으나, 확인 절차는 제안 실행의 선행 조건으로 유지한다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)

- **과거 branch 조합 비교는 근거로 쓸 수 없다.** `D-050`은 여섯 실행 러너가 학습 설정을 넘기지 못하고 `eval_seal_tasks.sh` 폴백에 걸려 **전부 같은 7-branch 설정으로 수렴**했다고 기록한다. 따라서 과거 branch 조합 비교는 구성 차이가 없는 실행들의 비교이며, 모델 개선 근거로 사용할 수 없다.

- **공식 `0.6227`과 유효 랭크 `3.52/7`은 현재 `verified`가 아니다.** `PROJECT.md §3.2`는 이 값들을 공식 기준으로 두지만, `D-050`은 이 절의 값을 `unverified-hold`로 명시한다. 저장 예측 파일에 branch list·가중치·코드 해시·fold manifest 해시가 없어 과거 산출물을 특정 구성에 귀속시킬 수 없으므로, 이 값들은 승격 비교의 verified 기준선으로 바로 쓸 수 없다.

- **적합 분산은 이제 비교의 필수 요소다.** `D-048`은 “학습 파라미터 0개” 제약을 폐기하고, 결정론이 더 이상 성립하지 않으므로 모든 비교에 fold 분산과 적합 분산을 함께 다루도록 한다. `RU-97`은 동일 구성에서 seed만 바꿔도 macro 표준편차 `0.0021~0.0083`이 나온다고 실측하며, 이는 승격 임계 `+0.0030`과 같은 규모다. 따라서 단일 적합의 Δ를 안정 신호로 취급하는 것은 지지되지 않는다. 다만 `RU-97`의 분산분해는 GF 구성 공간의 실측이지, 현재 7-branch 공식 구성의 공통 상한은 아니다.

- **유효 랭크 `3.52/7`만으로는 “branch 추가가 막혔다” 또는 “서로 다른 branch를 아직 못 만들었다”를 결정할 수 없다.** `CA-09`는 `BM·QA·DS` 위치 계열의 유효 랭크가 `1.29/3`이고 상호 상관이 `0.25~0.93`이라고 기록하며, `CA-10`은 형상통계 강화 7종이 게이트 ②를 통과하지 못해 형상 계열 병목이 절사가 아니라 정보량이라고 결론지었다. 그러나 raw slide descriptor의 effective rank, branch별 기여도, upstream compression 손실은 문서에 실측이 없다.

- **집계 축은 완전히 닫힌 것이 아니라, 선택 편향과 계측 복구 제약이 있는 축이다.** `CA-08`은 기존 branch margin 위의 결합 함수·부분집합 탐색을 선택 편향으로 종료했지만, “새 정보(예측분산·Context 신호 기하)를 사용하는 `PSW`·`TGW`”와 “탐색임을 명시한 제한된 결합 연구”는 경계 밖으로 둔다. 따라서 학습 파라미터를 도입하지 않는 집계 변경은 가능하지만, 단일 사전 등록·선택 이력 기록·provenance·성능 조회 전 고정 없이는 확증 근거가 될 수 없다.

- **Primary 7 개선이 SEAL 10 ABMIL 초과를 보장하지 않는다.** `PROJECT.md §1`은 Primary 7의 개선이 hold-out 개선을 보장하지 않는다고 명시하고, `D-047`은 SEAL 10 결과를 모델·후보·구조 선택에 쓰는 것을 금지한다. 따라서 모든 후보 판정은 `hold-out 미검증`으로 남는다.

- **현재의 우선 병목은 새 모델 축보다 계측 복구다.** `current_status.md`의 현재 목표는 provenance writer, `SH`/`SJ` branch control 경로, 오염 검사 재설계, 단위 테스트 복구이며, 복구 전에는 게이트·승격 비교를 `verified`로 수행할 수 없다고 기록한다. `D-050` 정정은 YAML 경로가 정상이고 공식 7-branch 재현이 `0.6226`으로 성공했다고 확인하지만, 저장 산출물의 provenance 부재 때문에 `unverified-hold` 판정은 유지된다.

- **분산 축소는 성능 개선과 개념적으로 다르다.** common fold·common seed·common draw 짝짓기, seed averaging, deterministic closed-form 경로, fixed manifest, provenance 기록은 Δ의 SE를 줄이는 경로다. 이들이 후보의 기대 Δ 자체를 바꾸지 않는 한, 성능 개선이 아니라 측정 정밀도 개선이다. 다만 random subsampling·GMM·closed-form 해의 fold별 변동이 mean bias를 동반한다면 분산 축소 개입이 mean을 바꿀 수 있으므로, 현재 7-branch에서는 mean과 SE를 별도로 실측해야 한다.

## 반박 (증거가 반박함)

- **“유효 랭크 `3.52/7`만으로 branch 추가 방향을 결정한다”는 주장은 반박된다.** `3.52/7`은 `unverified-hold` 계측이며, effective rank는 선형 중복 지표일 뿐 task 정보량이나 AUROC를 보장하지 않는다. raw descriptor rank, branch별 기여도, upstream compression 손실이 실측되지 않았으므로, 이 값만으로 “branch 추가 막힘” 또는 “서로 다른 branch 미생성”을 결론짓는 것은 증거 범위를 넘는다.

- **“과거 branch 조합 비교가 구성 차이를 반영한다”는 주장은 반박된다.** `D-050`은 여섯 러너가 같은 7-branch 설정으로 수렴했다고 기록한다. 따라서 과거 branch 조합 비교는 구성 비교가 아니라 동일 구성의 반복 실행 비교다.

- **“공식 `0.6227` 또는 `3.52/7`을 verified 기준으로 승격 근거에 쓴다”는 주장은 반박된다.** `D-050`은 이 값들을 `unverified-hold`로 명시하고, provenance가 없는 저장 예측을 특정 구성에 귀속시킬 수 없다고 기록한다.

- **“SEAL 10 결과를 보고 모델·후보·집계·임계값을 고를 수 있다”는 주장은 반박된다.** `D-047`과 `PROJECT.md §3.1`은 SEAL 10을 최종 hold-out으로 유지하고, 선택·튜닝에 어떤 형태로도 사용하는 것을 금지한다.

- **“학습 파라미터 0개 제약을 여전히 유지해야 한다”는 주장은 반박된다.** `D-048`은 이 제약을 폐기하고 `CA-14`를 `ABOLISHED`로 바꿨다. 다만 `CA-14`의 과거 관측, 즉 within-context PCA + closed-form ridge가 학습 계보보다 일반화·OOD에서 우월했다는 관측은 소급 취소되지 않으므로, 학습 파라미터를 되돌리는 구성은 새 증거가 필요하다.

- **“Primary 7에서 `+0.0030`을 얻으면 SEAL 10 ABMIL 초과가 보장된다”는 주장은 반박된다.** `PROJECT.md §1`은 Primary 7 개선이 hold-out 개선을 보장하지 않는다고 명시한다.

- **“`RU-97`의 fold×seed 잔차 `23.9%`를 현재 7-branch에 그대로 전이한다”는 주장은 반박된다.** `current_status.md`와 `RU-97` 기록은 이 분산분해가 GF 단독 탐색의 특정 구성 공간에서 나온 실측이라고 한정한다. 현재 7-branch 공식 구성의 seed main·fold×seed 잔차 분해는 문서에 없다.

- **“집계 축은 아무 제약 없이 탐색 가능한 untouched 축이다”는 주장은 반박된다.** `CA-08`은 기존 branch margin 위의 결합 함수·부분집합 탐색을 선택 편향으로 종료했다. 또한 `D-050`은 provenance·오염 검사·arm routing이 복구되기 전에는 verified 게이트·승격 비교를 수행할 수 없다고 기록한다.

- **“same-seed pairing만으로도 `+0.0030` 판별이 보장된다”는 주장은 반박된다.** same-seed pairing이 seed main을 상쇄하려면 baseline과 후보가 같은 random number를 같은 순서로 소비해야 한다. 현재 7-branch에서 seed·draw·RNG state·fold 처리 순서가 실제로 margin을 바꾸는 무작위 원천인지, 그리고 baseline·후보가 동일하게 짝지어지는지는 문서에 실측이 없다. `RU-97`의 fold×seed 잔차도 현재 구성에 전이할 수 없다.

- **“`D-050` 초기 판단처럼 재현 경로가 전혀 없다”는 주장은 부분적으로 반박된다.** `D-050` 정정은 YAML 설정의 branch 가중치가 정확히 파싱되고, 합성 데이터에서 `sh`/`sj`를 `1.0→0.0`으로 바꾸면 margin이 달라지며, 공식 7-branch 재현이 `0.6226`으로 성공했다고 확인한다. 그러나 `ICF_*` 환경변수 경로는 무효이고, 저장 예측 파일의 provenance 부재 때문에 `unverified-hold` 판정 자체는 유지된다.

## 판별 불가 (증거가 가르지 못함)

- **유효 랭크 `3.52/7`의 원인이 branch 생성 부족인지, upstream slide-level compression·whitening·PCA-32·bag mean·salience aggregation 손실인지, 아니면 task 정보 부재인지는 가르지 못한다.** 두 해석을 가르는 관측은 동일 fold·과제·slide 집합, 동일 label-free rank estimator, 동일 스케일·정규화·표본 조건에서 raw slide descriptor effective rank와 7-branch margin effective rank를 비교하고, 구조적으로 다른 후보가 게이트 ①에서 `max |r| ≤ 0.6`과 8-branch 기준 `eff.rank ≥ 4.02`를 만족하는지 확인하는 것이다. 그러나 raw descriptor rank, branch별 기여도, raw/margin rank gap은 문서에 없다.

- **trimmed mean voting의 병목이 branch 오류인지, margin scale 불일치인지, branch 간 중복인지 가르지 못한다.** branch margin의 scale 편차, fold 내 노이즈, magnitude가 신호인지 노이즈인지, branch별 기여도는 문서에 실측이 없다.

- **분산 축소가 성능 개선보다 나은 투자인지는 가르지 못한다.** 현재 7-branch의 true paired Δ, seed main, fold main, fold×seed 잔차, 적합 분산 구조는 문서에 없다. `RU-97`의 `0.0021~0.0083`과 `23.9%`는 GF 구성 공간 실측이지 현재 7-branch의 공통 상한이 아니다. 따라서 분산 축소가 `+0.0030` 판별을 가능하게 하는지, 아니면 정확한 null만 더 분명하게 드러내는지 현재 증거로는 판별 불가다.

- **same-seed pairing·seed averaging이 현재 7-branch에서 Δ의 SE를 충분히 줄이는지는 가르지 못한다.** 현재 7-branch에서 seed·draw·RNG state·fold 처리 순서가 실제로 margin을 바꾸는 무작위 원천인지, baseline과 후보가 common random number를 동일하게 소비하는지는 문서에 실측이 없다.

- **per-fold rank/quantile 변환, context-quantile trimmed mean, PSW·TGW 등 집계 변경이 paired Δ를 `+0.0030` 이상 개선하는지는 가르지 못한다.** query slide 수, fold·task 내 rank 안정성, context-query margin 분포 이대, quantile 추정 노이즈, margin magnitude의 신호 여부는 문서에 없다. `CA-07`은 context↔query 분포 이동을 이미 문제화된 축으로 기록하지만, 그 크기와 quantile 매핑에 대한 영향은 미실측이다.

- **FC가 게이트 ①을 통과하는지는 가르지 못한다.** `CA-09`는 FC를 비저촉으로 명시하지만, FC margin, FC와 기존 7-branch의 `max |r|`, 8-branch `eff.rank`, raw rank gap은 문서에 없다. 또한 `D-050` 이후 SCREEN_ONLY·오염 검사·provenance 경로가 복구되기 전에는 verified 게이트 ①을 수행할 수 없다.

- **4 GPU-hour 안에서 baseline+후보 350-fold paired Δ, 적합 분산, 게이트 ① 중 무엇을 완주할 수 있는지는 가르지 못한다.** fold 적합 1회 시간은 문서에 없다. 따라서 `700R` fold-fit이 4 GPU-hour 안에 들어가는지, R=1만 가능한지, R>1이 가능한지는 현재 문서만으로 판별 불가다.

- **PCA-32, whitening, closed-form ridge, branch margin 생성이 context/train fold에만 fit되고 query/test/SEAL 10을 fit set에 쓰지 않는지는 가르지 못한다.** fit 범위, fold manifest, context/query 분리, SEAL 10 배제 여부는 문서에 실측이 없다.

- **`SH`/`SJ` 편입 후 `Histologic_Grade`, `progression_regression`, `PBRM1`이 악화된 기전은 가르지 못한다.** `PROJECT.md §3.2`는 세 과제가 `SH` 단독·`SJ` 단독에서도 같은 방향으로 악화되지만, 기전은 `미확인`이라고 명시한다.

- **Primary 7 개선이 SEAL 10 hold-out으로 전이되는지는 가르지 못한다.** SEAL 10은 보류 hold-out이며, 선택에 사용할 수 없다. 따라서 현재 모든 후보는 `hold-out 미검증`이다.

## 실행 무효 (실행·계측 결함으로 판단 불가)

- **`D-050` 이전 여섯 러너의 branch 조합 비교를 근거로 쓰는 것은 실행 무효다.** arm이 다른 실행들이 전부 같은 7-branch 설정으로 수렴했으므로, 그 비교는 구성 차이를 측정하지 못했다.

- **provenance가 없는 저장 예측 파일을 baseline 또는 후보 margin으로 쓰는 것은 실행 무효다.** branch list·가중치·코드 해시·fold manifest 해시·seed/draw가 없으면 산출물을 특정 구성에 귀속시킬 수 없다.

- **`ICF_*` 환경변수 경로로 branch·aggregation을 통제하는 실행은 실행 무효다.** `D-050` 정정은 `ICF_*` 환경변수 경로가 무효이고, YAML 경로가 정상이라고 확인한다.

- **오염 검사 재설계 전 값 앵커 오염 검사 통과를 주장하는 것은 실행 무효다.** `D-050`은 기존 값 앵커가 재현되지 않고, 소수 4자리 정밀도를 시스템이 유지하지 못한다고 기록하며, provenance 기반 재설계가 완료될 때까지 오염 검사 통과를 주장하는 실행은 무효라고 명시한다.

- **`D-050` 이후 SCREEN_ONLY·게이트 ①을 provenance·오염 검사 복구 전에 verified로 수행하는 것은 실행 무효다.** `current_status.md`는 복구 전에는 게이트·승격 비교를 `verified`로 수행할 수 없다고 기록한다.

- **SEAL 10 결과를 후보·집계·R·임계값 선택에 쓰는 것은 실행 무효다.** `D-047`은 SEAL 10 결과를 모델 개선 방향에 쓰는 것을 금지한다.

- **Primary 7에서 여러 집계 변형을 돌려 최댓값을 승격 근거로 쓰는 것은 실행 무효다.** `CA-08`과 `PROJECT.md §4.1`은 선택·적응을 반영하지 않은 같은 자료 내 최댓값 확증을 선택 편향으로 금지한다.

- **적합 분산을 포함한 승격 판정 절차가 사용자 승인되기 전에 fit-averaged paired Δ를 승격 근거로 쓰는 것은 실행 무효다.** `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 미승인이라고 기록한다. 승인 전에는 fit-averaged paired Δ는 Primary 7 탐색·측정 probe로만 한정해야 한다.

- **provenance·arm routing 복구 전 full 350-fold paired Δ를 verified 승격 근거로 실행하는 것은 실행 무효다.** fold time 미측정 자체는 아직 실행 무효가 아니라 판별 불가지만, 계측 복구 없이 실행하면 구성 귀속이 불가능해 무효가 된다.

## 좌석 공백 (기권·실패로 빠진 관점)

- 없음. 기권한 좌석은 없다.

## 다음 행동 (권한 등급과 함께)

- **[T2] 적합 분산 포함 승격 절차에 사용자 승인을 받는다.** estimand를 단일 적합 Δ가 아니라 fold×seed 기대 Δ로 쓸지, R을 어떻게 정할지, same fold·same seed·same draw·same fold order를 어떻게 보장할지, best-seed 선택을 금지할지, `+0.0030` 임계를 어떤 estimand에 적용할지를 사전 등록한다. 승인 전에는 fit-averaged paired Δ를 승격 근거로 쓰지 않는다.

- **[T1] 계측 복구를 먼저 완료한다.** provenance writer(branch list·가중치·코드 해시·fold manifest 해시·seed/draw), arm routing 복구(`ICF_*` 환경변수 경로 폐기, YAML 경로 사용), 오염 검사 재설계, 단위 테스트를 완료한다. 완료 전에는 verified 게이트·승격 비교를 수행하지 않는다.

- **[T1] 1-fold timing probe를 실행한다.** Primary 7 고정 fold 1개를 GPU 4에서 실행하고 wall time, GPU time, RAM, OOM, I/O, provenance를 기록한다. fold 적합 1회 시간 실측 전에는 full 350-fold baseline+후보 재적합을 시작하지 않는다. probe 결과로 `700R` fold-fit이 4 GPU-hour 안에 가능한지 산술적으로만 판단하고, 그 산술은 실측이 아니다.

- **[T1] verified per-fold baseline을 재생성한다.** 공식 7-branch YAML 경로로 Primary 7 50-fold를 재적합하고, provenance를 기록한 뒤 원시 margin에서 macro를 독립 재계산한다. `0.6227` 또는 `0.6226`은 재현 대조값으로 쓰고, provenance가 없는 과거 예측을 baseline으로 쓰지 않는다.

- **[T0] 집계 후보를 단일 사전 등록으로 제한한다.** rank/quantile 변환, PSW·TGW 등 후보를 쓰려면 결과 조회 전에 변환 규칙, fit set(context/train fold만), 라벨 미사용 보장, 선택 이력, 단일 후보 확정 규칙을 기록한다. 여러 집계 변형을 Primary 7에서 비교한 뒤 최댓값을 승격 근거로 쓰지 않는다.

- **[T0] raw/margin rank audit를 사전 설계한다.** raw slide descriptor와 7-branch margin을 동일 fold·과제·slide 집합, 동일 label-free estimator, 동일 스케일·정규화·표본 조건에서 비교할 수 있도록 raw descriptor 저장 경로와 effective rank 정의를 문서화한다. 이 audit는 성능 기전이 아니라 “투영 손실 후보”와 “정보 부재 후보”를 좁히는 진단으로만 쓴다.

- **[T0] 현재 7-branch의 소수 fold×seed variance decomposition을 설계한다.** `RU-97`의 `23.9%`를 전이하지 않고, 현재 공식 7-branch에서 seed main, fold main, fold×seed 잔차를 소수 fold subset으로 재측정한다. 이 결과가 없으면 R 선택과 seed averaging의 효과는 판별 불가로 둔다.

- **[T0] 선택 이력을 기록한다.** rank audit, FC gate, aggregation 변형, fit-averaged probe를 같은 Primary 7에서 순차 수행할 때, 각 후보의 사전 등록 순서, 데이터 접근 로그, 성능 조회 시점, 후보 변경 시점을 기록한다. 후보 변경이 Primary 7 성능 조회 이후에 발생하면 해당 후보의 Δ는 실행 무효로 처리한다.

## 하지 말 것 (이유와 함께)

- **SEAL 10 결과를 후보·집계·R·임계값 선택에 사용하지 않는다.** `D-047`은 SEAL 10을 최종 hold-out으로 유지하고, 그 결과로 모델 개선 방향을 고르는 것을 금지한다.

- **provenance가 없는 저장 예측을 baseline 또는 후보로 사용하지 않는다.** `D-050`은 저장 예측 파일에 branch list·가중치·코드 해시·fold manifest 해시가 없어 구성 귀속이 불가능하다고 기록한다.

- **`ICF_*` 환경변수 경로로 실험을 수행하지 않는다.** `D-050` 정정은 이 경로가 무효이고, YAML 경로가 정상이라고 확인한다.

- **여러 집계 변형을 Primary 7에서 탐색한 뒤 최댓값을 승격 근거로 사용하지 않는다.** `CA-08`은 선택에 쓴 평가자료의 최댓값을 확증 성과로 쓰는 절차를 선택 편향으로 종료했다.

- **`RU-97`의 fold×seed 잔차 `23.9%`나 seed std `0.0021~0.0083`을 현재 7-branch에 그대로 전이하지 않는다.** 이 값들은 GF 구성 공간의 실측이며, 현재 7-branch 공식 구성의 분산분해는 문서에 없다.

- **유효 랭크 `3.52/7`을 verified 값이나 인과적 결론으로 사용하지 않는다.** 이 값은 `unverified-hold`이며, effective rank는 선형 중복 지표일 뿐 task 정보량이나 AUROC를 보장하지 않는다.

- **1-fold timing probe와 provenance 복구 전 full 350-fold verified 승격 실험을 시작하지 않는다.** fold 적합 1회 시간이 문서에 없으므로, 예산 적합성은 현재 판별 불가다. 계측 복구 없이 실행하면 구성 귀속이 불가능해 실행 무효가 된다.

- **적합 분산 포함 승격 절차 승인 전 fit-averaged paired Δ를 승격 근거로 사용하지 않는다.** `current_status.md`는 이 절차가 미승인이라고 기록한다.

- **성능 관측을 기전 입으로 바꾸지 않는다.** `SH`/`SJ` 편입 후 특정 과제 악화, branch 중복, 유효 랭크 저하는 관측이지, 기전이 확인된 것이 아니다. 기전이 확인되지 않으면 `기전 미확인`으로 둔다.

- **좌석 동의 수를 근거로 쓰지 않는다.** 표결은 증거가 아니며, 판단은 문서에 있는 실측·결정·경계와 그 적용 범위만으로 내려야 한다.
