# ICF Council — C-20260918-15

- **연구 질문**: 2×2 요인 프로브의 사전 등록(pre-registration)을 확정한다. 결과를 보기 전에 고정해야 하므로 이 회차가 마지막 기회다.

배경(전부 확인된 사실): 우리 목표는 Primary 7 과제에서만 근거를 얻어 모델을 개선하는 것이다. 후보 축은 두 가지다 — (a) context feature에 PCA를 적용하는 것, (b) context slide를 subsample 하는 것. 앞선 탐색에서 두 축을 한꺼번에 바꾼 실행이 있었고, 그래서 어느 축이 효과를 냈는지 가를 수 없었다. 또한 적합 분산이 크다는 것이 실측됐다: 같은 구성을 seed만 바꿔 다시 적합하면 GF 계열에서 표준편차 0.0021~0.0083이 나왔고, 이는 우리가 쓰던 승격 임계값 +0.0030보다 크다.

또 하나의 확인된 사실: 실행 스크립트가 넘겨받은 설정을 조용히 다른 설정으로 교체하는 폴백을 갖고 있었다. 그래서 과거 산출물은 어떤 구성으로 만들어졌는지 귀속할 수 없고 unverified-hold 상태다. 새 실행에는 requested/effective 설정 해시를 기록한다.

질문:
(1) 2×2 (baseline / PCA-only / subsample-only / combined)에서 무엇을 estimand로 선언해야 하는가? 특히 subsample은 context slide가 일정 수 이상인 fold에서만 발동하므로, 무조건부 50-fold Δ와 발동 fold 한정 Δ 중 무엇이 주 결과인가. 하나를 고르고 이유를 써라.
(2) 적합 분산 0.0021~0.0083이 실측된 상태에서, 성공 판정 임계값을 어떻게 정해야 하는가? 단일 seed 단일 draw로는 무엇을 말할 수 없는지 명시하고, 필요한 반복 횟수를 근거와 함께 제시하라. 반복 비용이 과도하면 '이 프로브로는 판별 불가'가 정당한 결론인지도 판단하라.
(3) 상호작용(PCA × subsample)을 주장하려면 무엇이 필요한가? 우리가 가진 fold 수로 상호작용이 검출 가능한지, 아니면 사전에 '상호작용은 판별 불가로 둔다'고 선언해야 하는지 답하라.
(4) 이 프로브가 실패했을 때 그것을 실패로 인정하게 만드는 사전 조건은 무엇인가? 사후에 '다른 각도로 보면 유망하다'로 빠져나갈 수 있는 구멍을 찾아 미리 막아라.
(5) 이 프로브에 자원을 쓰는 것이 옳은가? GPU 1장, 과제 7개, fold 50개 기준으로 대략의 비용을 추정하고, 같은 자원으로 할 수 있는 더 나은 대안이 있으면 제시하라.

문서에 없는 수치를 지어내지 마라. 위 분산 수치의 출처는 저장된 실행 로그다.
- **RU 유형**: 확증
- **최대 권한 등급**: T1
- **소집자**: Claude Code (claude-opus-5)
- **좌석**: R1(R, temp 0.5, http://127.0.0.1:8003/v1/chat/completions), M1(M, temp 0.3, http://127.0.0.1:8000/v1/chat/completions), T1(T, temp 0.4, http://127.0.0.1:8001/v1/chat/completions), P1(P, temp 0.7, http://127.0.0.1:8000/v1/chat/completions), A1(A, temp 0.2, http://127.0.0.1:8003/v1/chat/completions), S1(S, temp 0.2, http://127.0.0.1:8001/v1/chat/completions)
- **호출**: 11/16 · **소요**: 819.6초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 편성 근거 (결과 전 기록)

사전 등록이므로 반증(R)을 첫 좌석에 둔다 — 빠져나갈 구멍을 먼저 찾는 것이 이 회차의 핵심 산출이다. 실측(M)은 질문 2와 5, 즉 적합 분산 대비 필요한 반복 횟수와 자원 비용을 맡는다. 이론(T)은 estimand 선언과 상호작용 검출 가능성을 맡는다. 제안(P)은 온도 0.7로 설계안을 낸다. 감사(A)는 사전 등록이 사후 변경 가능한 형태로 쓰이지 않았는지 본다. 외부(E)·일반화(G)는 제외 — 우리 프로브의 설계 문제이고 외부 수치는 근거가 될 수 없다. 앞선 회차 보고서는 입력에 넣지 않았다(회차 간 앵커링 금지).

## 사전 고정 기준 (결과 전 기록)

이 회차의 산출은 사전 등록 문서다. 결과 수치가 아직 없으므로, 좌석은 '무엇이 나오면 성공/실패인가'만 쓴다. 어떤 좌석도 AUROC 값을 예측하거나 인용해서는 안 된다.

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: 주 estimand를 “무조건부 50-fold paired Δ”로 고정하고, 발동 fold Δ와 combined를 2차로 격하
- **바꾸는 것**: 2×2의 확인적 estimand를 baseline 대비 paired Δ의 고정 7 과제 × 50 fold 평균으로 선언한다. PCA-only와 subsample-only를 primary estimand로 두고, combined는 additivity/interaction 진단으로만 둔다. subsample은 “무조건부 50-fold Δ”를 주 결과로 선언한다. subsample이 발동하지 않는 fold는 제외하지 않고, 해당 fold에서 실제로 관측된 candidate−baseline 차를 estimand에 포함한다. 발동 조건과 발동 fold manifest는 결과 보기 전에 고정하고 산출물에 기록한다. 모든 cell은 과거 unverified 산출물을 재사용하지 않고 새 실행에서 requested/effective 설정 해시를 함께 남긴다.
- **바꾸는 전제**: (1) subsample은 발동 fold에서만 평가해야 한다는 전제, (2) combined 결과가 단일 축의 실패를 사후에 구제할 수 있다는 전제, (3) 과거 baseline 산출물을 재사용해도 된다는 전제를 뒤집는다.
- **기대하는 차이**: 정량 예측 불가. 그러나 판정 구조는 바뀐다. 발동 fold 한정 Δ가 크고 무조건부 Δ가 작으면, “발동 시에는 좋아 보이나 배포했을 때 Primary 7 평균 개선은 임계 미달”로 처리된다. 반대로 발동 fold가 많고 효과도 유지되면 무조건부 Δ가 승격 임계 +0.0030에 도달할 수 있다. 이는 §4의 승격 산출식이 사용하는 350-fold paired Δ와 같은 표본 범위를 유지한다.
- **기전**: 성능 개선 기전은 미확인. estimand 정합의 경로는 확인 가능하다. §3.1은 fold를 고정하고, §4는 Primary 7의 고정 50-fold에서 candidate와 baseline을 같은 fold에서 짝지은 Δ의 평균을 승격 근거로 쓴다. subsample의 조건부 발동은 배포 규칙이므로, 실제 운영 효과를 추정하려면 비발동 fold도 포함해야 한다. 발동 fold만 쓰면 분모가 바뀌고, 발동 여부라는 자료 구조에 의해 평가 집합이 선택되는 경로가 생긴다.
- **실패 조건**: 이 제안이 틀렸다면 가장 그럴듯한 이유는, subsample의 진짜 과학적 질문이 “발동했을 때 효과가 있는가”이고, 발동 fold가 드물어서 무조건부 Δ가 희석된 경우다. 그러면 이 estimand는 발동 시 명확한 개선에도 실패로 처리할 수 있다. 그러나 이 프로브의 목적이 Primary 7 전체 평균을 기준으로 한 모델 개선 근거 생산이라면, 이 희석은 버그가 아니라 운영 효과를 반영하는 보수성이다.

## 제안 2: 단일 seed 단일 draw를 확인적 근거로 쓰지 말고, matched-seed 반복과 SE 게이트를 판정에 연결
- **바꾸는 것**: 단일 seed 단일 draw의 점추정을 확인적 근거로 쓰지 않는다. estimand를 R개 matched seeds에서 baseline과 같은 seed, 같은 fold로 짝지은 paired Δ의 평균으로 바꾼다. baseline도 과거 unverified-hold 값이 아니라 같은 seed set으로 새 실행한다. best-seed 선택을 금지한다. 확인적 R로 R=16을 제안한다. R=16이 예산을 초과하면, R=8으로 확인적 프로브를 강행하기보다 R=4 2×2 variance/provenance diagnostic로 전환하고, 확인적 판별은 “판별 불가”로 종료하는 쪽이 정직하다. 판정 규칙은 다음과 같이 고정한다.  
  - 지지: 해당 축의 point Δ ≥ +0.0030이고, 총 SE < +0.0030.  
  - 판별 불가: 실행은 유효하지만 총 SE ≥ +0.0030.  
  - 실패/지지 불가: 실행은 유효하고 총 SE < +0.0030이지만 point Δ < +0.0030.  
  총 SE는 seed-averaged fold-level Δ에 대한 과제 군집 SE와, R개 seed-level Δ 평균의 SE를 모두 계산한 뒤 둘 중 더 큰 값을 사용한다.
- **바꾸는 전제**: (1) 단일 seed 점추정으로 +0.0030 여부를 말할 수 있다는 전제, (2) 적합 분산은 보고만 하면 되고 판정에서 무시해도 된다는 전제, (3) 여러 seed 중 좋은 seed를 선택해도 된다는 전제를 뒤집는다.
- **기대하는 차이**: 정량 예측 불가. 그러나 반복 횟수의 근거는 실측 적합 분산에서 나온다. 저장된 실행 로그에서 같은 구성의 seed std는 0.0021~0.0083이다. 이는 승격 임계 +0.0030과 같은 크기이거나 크다. 보수적으로 candidate와 baseline의 seed sd가 모두 0.0083이고 서로 독립이라고 계산하면, Δ의 seed sd는 약 0.0117이고, R=16의 SE는 약 0.0029, R=8의 SE는 약 0.0041이다. 이 계산은 공분산 미측정 상태의 상한 가정이지만, R=8이 worst case에서 임계보다 크고 R=16이 임계 아래로 내려갈 수 있음을 보여준다. 따라서 R=16은 “보장”이 아니라 “worst case에서 판별 가능할 최소 보수 계획값”이다.
- **기전**: 성능 개선 기전은 미확인. 분산 경로는 확인 가능하다. D-048 이후 결정론이 깨졌고, 동일 구성의 재적합에서 macro 변동이 실측됐다. 단일 seed 단일 draw는 fold 변동만 아니라 적합 변동의 한 번 뽑기이므로, +0.0030이 실제 효과인지 seed noise인지 구별할 수 없다. matched seeds와 seed 평균은 common random numbers에 의해 seed 주효과를 줄이고, seed-level SE로 적합 분산을 판정에 넣는다.
- **실패 조건**: 이 제안이 틀렸다면 가장 그럴듯한 이유는, candidate와 baseline의 seed 효과가 실제로 강하게 상관되어 R=16이 과도하다는 것이다. 또는 이 프로브의 exact pipeline은 GF 계열보다 적합 분산이 훨씬 작아서 R=16이 불필요할 수 있다. 반대로 R=16이 예산을 초과하면, R=8으로 “지지”를 얻는 것은 seed noise와 구별되지 않으므로 확인적 승격 근거로 정당하지 않다. 그 경우 R=8 + SE 게이트로 “판별 불가”를 선언하거나, R=4 diagnostic로 분산만 재는 것이 더 낫다.

## 제안 3: 상호작용은 사전에 “판별 불가”로 선언하고, combined를 구제 수단이 아니라 진단으로만 쓰며, 실패/실행 무효 조건과 자원 대안을 고정
- **바꾸는 것**: 상호작용 contrast, combined의 역할, 실패 정의, 자원 대안을 결과 보기 전에 고정한다.  
  - 상호작용 contrast는 combined − PCA-only − subsample-only + baseline로 계산하지만, 확인적 주장 대상이 아니라 진단으로만 둔다. 상호작용을 주장하려면 4 cell이 같은 fold·seed에서 측정되고, contrast의 SE가 사전 계산되며, main effect를 넘어서는 사전 지정된 효과가 필요하다. 현재 fold pair는 350개지만 독립 단위는 7 과제 군집에 가깝고, interaction contrast의 variance는 문서에 없다. 따라서 상호작용은 사전에 “판별 불가”로 선언한다.  
  - combined는 PCA-only 또는 subsample-only가 실패했을 때의 사후 구제 수단이 아니다. PCA-only와 subsample-only가 모두 지지 조건을 만족하지 못하면, combined point Δ가 +0.0030 이상이어도 이 프로브의 attribution 목표는 실패다. combined만 좋다면 새 조합 후보로 별도의 사전 등록이 필요하다.  
  - 실패/무효를 사전에 구분한다.  
    1. 실행 무효: requested/effective hash 불일치, 폴백 설정 탐지, 7×50×R 중 누락, provenance 기반 오염 검사 재설계 실패, 코드/설정 drift, unverified 과거 산출물 재사용.  
    2. 판별 불가: 실행은 유효하지만 총 SE ≥ +0.0030.  
    3. 프로브 실패: 실행은 유효하고 총 SE < +0.0030이지만, PCA-only와 subsample-only 둘 다 Δ ≥ +0.0030을 만족하지 못함.  
    4. 사후 재해석 금지: 발동 fold로 전환, best-seed 선택, 일부 과제만 선택, 임계값 변경, combined를 단일 축으로 재해석, SEAL 10 결과를 방향 선택에 사용.  
  - 자원 대안: GPU 1장, 과제 7, fold 50 기준으로 1 seed 4 cell은 1,400 fold-fits이다. R=16 2×2는 22,400 fold-fits이고, R=8 2×2는 11,200 fold-fits, 3-cell R=8은 8,400 fold-fits, R=4 2×2는 5,600 fold-fits이다. 1 fold-fit의 GPU 시간은 문서에 없으므로 GPU-hour는 정량 예측 불가하다. R=16이 예산을 초과하면, 2×2 확인적 프로브보다 (a) baseline/PCA-only/subsample-only만 R=8로 실행해 main effect만 확인하거나, (b) R=4 2×2로 provenance와 seed×fold 분산만 재고 확인적 판별은 “판별 불가”로 종료하는 것이 더 낫다.
- **바꾸는 전제**: (1) combined가 통과하면 그중 하나의 축이 유효하다는 전제, (2) 350 fold pair가 있으면 상호작용도 검출 가능하다는 전제, (3) 실패 후에 발동 fold, combined, 일부 과제로 재해석할 수 있다는 전제를 뒤집는다.
- **기대하는 차이**: 정량 예측 불가. 그러나 결과 해석의 경로가 하나로 고정된다. combined 점수가 높아도 단일 축의 확인적 증거가 없으면 attribution 성공이 되지 못하고, 발동 fold 점수가 높아도 무조건부 Δ가 임계에 못 미치면 승격 지지 근거가 되지 못한다. 이로 인해 “다른 각도로 보면 유망하다”는 사후 이탈이 구조적으로 막힌다.
- **기전**: 상호작용 기전은 미확인. 검정력 한계의 경로는 부분적으로 확인된다. §4.1은 fold 내 종속성과 과제 군집 효과를 이미 실측했고, 7 과제 군집은 df=6의 작은 독립 단위다. interaction contrast는 main effect보다 variance가 클 가능성이 높지만, 그 variance는 문서에 없다. 따라서 상호작용의 MDE는 문서에 없고, 현재 fold 수로 상호작용을 확인적으로 검출 가능하다고 말할 근거가 없다.
- **실패 조건**: 이 제안이 틀렸다면 가장 그럴듯한 이유는, combined-only 효과가 실제로 운영上有 가치 있고, main effect가 희석되어 실패로 처리하면 개선 기회를 놓치는 경우다. 또는 상호작용이 실제로 크는데 사전에 “판별 불가”로 선언해 정보 손실이 나는 경우다. 그러나 이 프로브가 확인적 attribution을 목표로 한다면, combined-only 신호는 새 후보의 탐색적 신호이지, PCA와 subsample 중 어느 축이 원인이라는 증거가 아니다. 그런 경우를 구제하려면 combined 구성 자체를 새로운 candidate로 놓고, 선택 이력과 분산을 반영한 새 사전 등록을 해야 한다.

### R1 (반증)

## 반론 1
- **공격 대상**: "2×2 프로브를 지금 확정하면 앞선 combined 실행의 효과를 PCA/subsample 축으로 귀속할 수 있다"는 주장.
- **더 단순한 설명**: 앞선 combined 실행의 차이는 축 효과가 아니라, 실행 스크립트의 폴백으로 requested/effective 설정이 달라진 것, seed 적합 변동, fold/과제 난이도 변동에서 왔다. 과거 산출물은 `unverified-hold`이므로 어떤 구성에서 나왔는지 귀속 자체가 안 된다.
- **무엇이 참이면 이 주장이 무너지는가**: 새 실행에서 requested/effective 해시가 불일치하거나, baseline이 §3.2/§4의 대응 비교로 검증되지 않으면 2×2 결과는 `실행 무효`다. 또한 동일 cell의 seed 간 변동이 cell 간 차이보다 크면, 축 귀속은 `판별 불가`다.
- **이 반론의 한계**: provenance writer·branch control·오염 검사 재설계가 완료되고, baseline과 후보가 같은 fold에서 짝지어 requested/effective 해시가 검증되면, 2×2는 귀속을 위한 유효한 설계가 될 수 있다.

## 반론 2
- **공격 대상**: "subsample의 주 결과를 발동 fold 한정 Δ로 선언하면, subsample이 실제로 작용한 fold에서만 평가하므로 더 타당하다"는 주장.
- **더 단순한 설명**: 발동 fold 한정 Δ의 차이는 subsample 효과가 아니라, 발동 fold가 baseline에서 원래 다른 fold들과 난이도·분산이 달라서 생긴 fold 선택 편차일 수 있다. slide 수 임계값으로 발동이 정해지면, 그 fold 집합은 baseline 성능과 상관 있을 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: §4의 승격 estimand는 Primary 7 전체 7×50 fold의 대응 Δ 평균이다. 발동 fold 한정 Δ는 다른 estimand이므로, 이를 주 결과로 "Primary 7 개선" 또는 승격을 말하면 기준과 불일치한다. 따라서 발동 fold Δ가 양수여도 Primary 7 전체 개선의 근거가 되지 못한다.
- **이 반론의 한계**: estimand를 "발동 fold에서의 조건부 효과"로만 한정하고 승격이 아니라 탐색/보조 결과로 쓰면, 그 범위 내에서는 유효하다. 발동 fold 비율이 높고 사전 고정돼 있으면 보조로 보고할 수 있다.

## 반론 3
- **공격 대상**: "무조건부 50-fold Δ를 주 estimand로 삼으면 subsample 효과를 평가한다"는 주장.
- **더 단순한 설명**: 무조건부 Δ의 0 또는 +0.0030 미만은 "subsample 효과가 없다"는 것이 아니라, 발동 fold가 적어 효과가 0으로 희석된 것일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 발동 fold 비율이 낮으면, 조건부 효과가 있어도 전체 Δ는 0에 수렴한다. 따라서 발동 fold 비율과 검정력을 사전 계산하지 않으면, 무조건부 Δ의 음성은 `반박`이 아니라 `판별 불가`다.
- **이 반론의 한계**: 발동 fold 비율이 충분히 높고, 무조건부 Δ의 MDE가 +0.0030 이하로 사전 확인되면, 무조건부 Δ는 Primary 7 전체 개선의 유효한 주 estimand다.

## 반론 4
- **공격 대상**: "GF 계열에서 실측된 적합 분산 0.0021~0.0083을 그대로 2×2 cell의 paired Δ 변동으로 쓰고, 단일 seed 또는 R=8로 판정할 수 있다"는 주장.
- **더 단순한 설명**: 0.0021~0.0083은 GF 계열 macro의 seed sd이지, PCA/subsample cell의 paired Δ seed sd가 아니다. 따라서 이 수치로 2×2의 판별력을 확정하는 것은 다른 구성의 분산을 이식하는 것이다. 단일 seed에서 +0.0030 초과가 보이면, 후보가 우월한 것이 아니라 그 seed가 평균보다 높은 draw였을 수 있다. RU-97에서 단일 적합의 +0.0099는 seed 42가 해당 구성 평균보다 1.5 sd 높았던 뽑기로 설명됐다.
- **무엇이 참이면 이 주장이 무너지는가**: PCA/subsample의 paired Δ seed sd가 0.0021~0.0083과 다르면, 이 수치로 한 R·MDE 계산은 무효다. 문서에는 paired Δ의 seed sd가 없다. 만약 0.0083을 paired Δ seed sd s의 상한으로 가정하고 seed 평균의 SE만 0.0030 이하로 만들려면 R≥(0.0083/0.0030)^2≈8이다. 95% CI 폭 또는 MDE≤0.0030을 요구하면 정규 근사로 R은 각각 ≈30, ≈60이다. 이 수치는 문서의 0.0083에서 유도한 계산이지, 2×2 cell의 실측이 아니다. R=8은 SE만 줄일 뿐, +0.0030 효과를 0과 구분하는 검정력을 확보한다고 말하기 어렵다.
- **이 반론의 한계**: paired Δ의 seed sd가 실제로 0.0021에 가깝거나 fold×seed 잔차가 seed 평균화에서 줄어든다면, 필요한 R은 줄어든다. 다만 그 sd는 문서에 없으므로 실측 전 확정할 수 없다.

## 반론 5
- **공격 대상**: "2×2로 PCA×subsample 상호작용을 주장할 수 있다"는 주장.
- **더 단순한 설명**: 상호작용 양수는 PCA/subsample의 main effect가 fold/seed 변동과 섞인 것일 수 있다. 상호작용은 cell 간 차이의 차이이므로, 단일 seed에서는 4개 cell의 적합 변동이 모두 잔차로 남는다.
- **무엇이 참이면 이 주장이 무너지는가**: 상호작용 contrast의 SE를 사전 계산하지 못했거나, 그 MDE가 +0.0030보다 크면, 상호작용은 `판별 불가`로 선언해야 한다. 50 fold×7 task의 fold 정밀도는 fold 효과를 줄이지만, seed 적합 변동은 fold를 늘려도 줄지 않는다(RU-97의 fold×seed 잔차 관측). 따라서 상호작용 검출은 seed 반복 없이는 성립하지 않는다.
- **이 반론의 한계**: seed 반복을 충분히 하고, 상호작용 contrast의 SE가 사전에 +0.0030 이하로 확인되면, 상호작용을 보조적으로 검정할 수 있다. 단, 승격 기준과 별도로 선언해야 한다.

## 반론 6
- **공격 대상**: "실패 후 '다른 각도로 보면 유망하다'는 사후 해석을 허용해도 된다"는 주장.
- **더 단순한 설명**: 2×2에서 한 cell만 양수하거나 일부 과제만 개선되면, 그것은 축 효과가 아니라 seed/fold noise, 또는 사후에 고른 하위집합의 최댓값일 수 있다. 같은 자료로 고르고 같은 자료로 성공을 판정한 §214식 선택 편향과 같은 구조다.
- **무엇이 참이면 이 주장이 무너지는가**: 사전에 ① 주 estimand(무조건부 50-fold paired Δ 또는 seed 평균 Δ), ② cell 비교 대상, ③ seed 평균 규칙과 best-seed 금지, ④ 발동 fold 재정의 금지, ⑤ 과제 하위집합·sign agreement 구제 금지, ⑥ SEAL 10으로 방향 선택 금지, ⑦ requested/effective 해시 불일치 시 `실행 무효`를 고정하지 않으면, 실패가 사후 유망으로 바뀐다.
- **이 반론의 한계**: 사전에 허용된 보조 분석(예: 발동 fold 조건부 estimand, 악화된 과제 전량 보고)은 그 범위 내에서 사후 해석이 아니다.

## 반론 7
- **공격 대상**: "GPU 1장·과제 7·fold 50으로 2×2 프로브를 돌리는 것이 같은 자원으로 최선이다"는 주장.
- **더 단순한 설명**: 같은 자원으로 먼저 baseline과 각 cell의 seed 반복을 돌려 paired Δ의 seed sd와 provenance를 확인하는 것이 더 단순하다. 적합 분산이 임계값보다 크다는 것이 이미 실측된 상태에서 전체 2×2를 돌리면, 결과가 `판별 불가`로 끝날 가능성이 크다.
- **무엇이 참이면 이 주장이 무너지는가**: 단일 run의 GPU-hour가 문서에 없고, 현재 상태 문서의 4 GPU-hour 예산이 이 프로브에 배정된 것인지도 명시되지 않았다. 따라서 4 cell × R repeat의 비용을 문서에서 산출할 수 없다. 만약 예산을 넘으면 `실행 무효`다. 더 나은 대안은 (1) provenance 검증과 baseline 재현, (2) 1~2 cell의 seed 반복으로 s_d 실측, (3) s_d에 따라 R을 결정하거나 `판별 불가` 선언, (4) 상호작용을 포기하고 main effect만 baseline 대비 비교로 한정하는 3-cell(baseline/PCA-only/subsample-only) 설계로 비용을 줄이는 것이다.
- **이 반론의 한계**: 단일 run 비용이 실제로 작아 4 cell × 필요한 R이 예산 안에 들어오고, provenance가 검증되면 2×2는 가능한 대안이다.

### M1 (실측)

## 실측 1: 2×2 estimand 고정과 측정 파이프라인
- **건드려야 하는 코드·설정**: 실행 스크립트의 조용한 폴백을 제거하거나 requested/effective 불일치 시 실패로 바꾸고, requested/effective 설정 해시(코드 해시, 데이터 버전, fold manifest, branch list, 가중치, PCA 파라미터, subsample 파라미터, seed)를 산출물에 기록한다. context feature 생성부에 PCA 옵션과 context slide subsample 옵션을 넣고, PCA 차원·subsample 크기·subsample 발동 임계값을 결과 조회 전에 고정한다. fold별 context slide 수, 발동 mask, per-fold/per-seed AUROC를 저장한다. primary estimand는 **무조건부 50-fold paired Δ**로 고정한다. 문서에 명시된 구체적 구현 경로가 없으므로 확인된 관련 경로만 적는다: `src/`(환경변수 미읽기·폴백), `scripts/run_tests.sh`(회귀), `scripts/analysis/branch_screen.py`(필요 시 라벨 무관 게이트). PCA/subsample의 실제 파일 경로는 문서에 없음.
- **비용**: 4 cell × R seed × 7 과제 × 50 fold. R=1이면 1,400 fold-fit, R=8이면 11,200 fold-fit. wall time·메모리·연산은 미실측. B200 183,359 MiB는 문서 실측이고 OOM 여부를 보장하지 않는다.
- **막는 것**: `D-050`으로 `src/`가 `ICF_*` export를 읽지 않아 requested 설정이 effective 설정으로 귀속되지 않고, 과거 산출물은 unverified-hold다. 폴백이 설정을 교체하면 cell 귀속이 불가능하다. subsample 미발동 fold에서 subsample on/off cell 간 Δ가 0이 아니면 무조건부 Δ가 오염된다. PCA/subsample의 고정 파라미터와 발동 임계값이 문서에 없어 현재로서는 실행 가능하지 않다. 실측 context slide 수는 90~261장이므로 임계값이 이 범위를 벗어나면 항상 발동 또는 항상 미발동이 된다.
- **중단 기준**: requested/effective 해시 불일치, 폴백 발동, fold·seed·과제 누락, NaN/Inf, OOM, subsample 미발동 fold에서 Δ가 0이 아니고 사전 허용 오차가 없으면 실행 무효. 발동 임계값·PCA 차원·subsample 크기가 사후 변경되면 실행 무효. `SEAL 10`을 선택에 사용하면 실행 무효.

## 실측 2: 적합 분산 보정 반복과 성공 임계값
- **건드려야 하는 코드·설정**: 4 cell 모두에 같은 seed 집합을 사용하고, cell별 R회 적합을 돌린다. best-seed 선택을 금지하고, seed별 macro·per-fold Δ·seed 평균 Δ·총 SE를 저장한다. 승격 임계값은 `+0.0030`을 유지하되, 점추정 `Δ_mean ≥ +0.0030`을 성공으로, 총 SE와 적합 분산을 병기한다. CI 하한 조건을 사용자 승인 없이 추가하지 않는다.
- **비용**: R=8이면 4×8×7×50=11,200 fold-fit, 즉 full Primary 7 50-fold run 32회. full run 시간 T가 미실측이므로 총 시간은 32T로만 표현된다. 현재 상태 문서의 4 GPU-hour 예산을 적용하면 T<0.125 GPU-hour여야 하며, 이 T는 미실측이다. R=1이면 1,400 fold-fit이지만 적합 분산 보정이 불가능하다.
- **막는 것**: 실측 적합 sd `0.0021~0.0083`의 최대값은 임계값 `+0.0030`보다 크다. 단일 seed 단일 draw로는 (i) `+0.0030` 초과가 적합 잡음인지 구분 불가, (ii) seed 평균과 CI 산출 불가, (iii) fold 분산과 적합 분산 분리 불가, (iv) 과거 combined 실행의 축 귀속 불가. R=8은 관찰된 최대 sd `0.0083`으로 seed 평균의 적합 SE를 `0.0083/sqrt(8)≈0.00294`로 임계값 이하로 만드는 최소값이지, 후보별 실제 paired Δ 분산이나 검정력을 보장하지 않는다. 후보별 분산은 미실측이며, `RU-97`의 fold×seed 잔차는 fold를 더 모아도 줄지 않는다고 실측됐다.
- **중단 기준**: R=8을 실행할 예산·시간이 없으면 "판별 불가"로 종료한다. R=8을 돌렸지만 총 SE가 임계값보다 크고 사용자 승인된 추가 반복이 없으면 성공·실패가 아니라 판별 불가로 기록한다. best-seed 사용, seed 집합 사후 변경, 임계값 사후 변경, 적합 분산 미보고는 실행 무효.

## 실측 3: 상호작용(PCA × subsample) 검출 가능 여부
- **건드려야 하는 코드·설정**: 4 cell을 같은 fold·seed로 짝지어 interaction contrast `Δ_combined − Δ_PCA − Δ_subsample + Δ_baseline`를 fold·seed 단위로 계산한다. baseline Δ는 0이어야 하므로 구현 검증에 쓴다. interaction 점추정·과제별 분포·총 SE를 보고, 사전에 interaction 판정 기준을 고정해야 한다.
- **비용**: 2×2 실행이 이미 예정되어 있으면 추가 실행 비용은 분석 수준이다. 2×2 R=8의 비용은 실측 2와 동일하다. interaction 검정에 필요한 최소 R은 문서에 interaction 분산 실측이 없어 확정할 수 없다.
- **막는 것**: 문서에는 interaction contrast의 표준편차·MDE·검정력이 없다. Primary 7은 과제 군집 7개(df=6)로 fold가 과제 내 독립이 아니고, 적합 분산은 임계값보다 크다. 따라서 현재 fold 수와 R=8로 interaction이 검출 가능하다고 말할 근거가 없다. interaction을 주장하려면 (i) 같은 fold·seed 짝짓기, (ii) R회 적합, (iii) interaction contrast의 실측 분산, (iv) 사전 고정된 최소 효과 기준이 필요하다.
- **중단 기준**: 사전 interaction 기준이 없으면 interaction은 "판별 불가/탐색적"으로만 기록한다. combined cell이 main effect보다 높다는 사후 관찰만으로는 상호작용을 주장하지 않는다. interaction 확인을 위해 fold 수를 늘리거나 과제를 사후 선택하면 실행 무효.

## 실측 4: 실패 인정과 사후 회피 구멍 차단
- **건드려야 하는 코드·설정**: 결과 조회 전에 실행 카드에 cell 목록, baseline fresh run, PCA/subsample 파라미터, 발동 임계값, seed 목록, R, primary estimand(무조건부 50-fold Δ), secondary estimand(발동 fold 한정 Δ), 성공/실패/판별 불가/실행 무효 기준, requested/effective 해시 스키마를 기록한다. raw margin과 per-fold/per-seed AUROC를 저장하고 독립 재계산 가능하게 한다.
- **비용**: 코드·테스트·기록 비용은 미실측. 실험 비용은 실측 1·2와 동일하다. 사후 구멍을 막는 추가 GPU 비용은 없다.
- **막는 것**: 사후 구멍은 (1) 무조건부 Δ가 미달이면 발동 fold 한정 Δ로 주 결과를 바꾸는 것, (2) best-seed 또는 좋은 seed만 인용, (3) 개선된 과제만 보고 악화 과제를 생략, (4) PCA/subsample 파라미터·임계값·fold mask를 사후 조정, (5) `SEAL 10`을 열람해 방향을 고르는 것, (6) 단일 seed 결과를 대표값으로 쓰는 것, (7) 성능 관측을 기전으로 전환하는 것, (8) unverified-hold 과거 산출물을 baseline으로 재사용하는 것이다. 이들은 `D-050`, `D-047`, `PROJECT.md` §4의 선택 편향·hold-out 금지·투명 보고 규범에 저촉된다.
- **중단 기준**: valid execution 조건(해시 일치, 폴백 없음, 전 fold·seed 저장, no-op 검증, R 충족)에서 primary Δ < `+0.0030`이면 실패로 인정한다. R 미충족이면 실패가 아니라 판별 불가다. provenance 누락·폴백·hold-out 사용·사후 재정의는 실행 무효다. "다른 각도로 보면 유망하다"는 사전에 secondary로 등록되지 않은 각도로는 결론을 바꿀 수 없다.

## 실측 5: 자원 배분 판단과 대안
- **건드려야 하는 코드·설정**: 먼저 `D-050` 계측 복구(provenance writer, `SH`/`SJ` branch control, 오염 검사 재설계, 단위 테스트)를 완료한다. 그 뒤 full 2×2 전에 wall time과 메모리를 측정하기 위해 성능 선택에 쓰지 않는 timed run을 허용한다. timed run에서 성능 지표를 계산하더라도 폐기하고, 후보 선택·R 조정·성공 판정에 사용하지 않는다.
- **비용**: full 2×2 R=8은 32 full Primary 7 runs(11,200 fold-fit). interaction을 사전에 판별 불가로 둔다면 combined cell을 빼고 3 cell R=8로 24 full Primary 7 runs(8,400 fold-fit)으로 줄일 수 있다. full run 시간 T 미실측. 4 GPU-hour 예산 기준, 2×2 R=8은 T<0.125h, 3 cell R=8은 T<0.1667h여야 한다. timed run 1 과제 × 50 fold × 2 seed × 2 cell(baseline/subsample)은 200 fold-fit으로, task-run 시간 T_task 미실측.
- **막는 것**: T가 미실측이라 "GPU 1장으로 얼마가 드는가"를 수치로 확정할 수 없다. `D-050` 복구 전에는 어떤 실행도 verified가 아니다. 적합 분산이 최대 `0.0083`이면 R=8 미만은 임계값 판별에 부적합하다. PCA 축의 단독 효과에 대한 실측 근거는 문서에 없고, subsample 축은 `CA-R1`이 reopened되어 있으나 선택 신호 부재로 활용 불가로 기록돼 있다.
- **중단 기준**: timed run에서 3 cell 또는 2×2 R=8이 4 GPU-hour를 초과하면, 추가 예산 승인 전에는 "판별 불가"로 종료한다. 계측 복구가 완료되지 않으면 실험을 시작하지 않는다. interaction을 주장하지 않기로 했으면 combined cell을 위해 예산을 쓰지 않는다. 같은 자원으로 더 나은 대안은 (a) 계측 복구, (b) 3 cell main-effect probe, (c) 필요 시 subsample 단일 축 우선이다. 2×2 전체는 interaction 기준이 사전에 없고 비용이 미실측이므로 현재로서는 정당화되지 않을 수 있다.

### T1 (이론)

- **가정**: 현재 2×2 프로브는 PCA와 subsample을 같은 Primary 7 50-fold에서 대응 비교할 수 있는 고정된 두 수준으로 취급한다. subsample이 context slide 수가 일정 수 이상인 fold에서만 발동해도, 무조건부 50-fold Δ와 발동 fold 한정 Δ 중 하나를 주 결과로 고르면 그것이 해당 축의 효과를 나타내는 단일 estimand라고 두고 있다. combined 셀은 PCA와 subsample의 적용 순서에 무관하게 잘 정의된 수준이며, fold 50개 × 과제 7개의 대응 Δ가 주효과와 상호작용을 분리하는 충분한 실험 단위라고 두고 있다. seed를 한 번만 고정하면 적합 불확실성이 무시할 수 있고, estimand의 평균 Δ가 승격 임계 +0.0030과 직접 비교 가능하다고 두고 있다.

- **이 가정이 깨지는 조건**: subsample 발동 fold가 개입 부여와 독립이 아니라 context slide 수, 과제/fold 난이도, baseline 성능, PCA 효과, 적합 seed 변동과 상관된다. 미발동 fold에서 subsample은 baseline과 동일하지만 “효과 0”이 아니라 “적용 불가”일 수 있다. combined 셀의 PCA·subsample 순서가 estimand를 바꾸면 2×2의 네 셀은 같은 인자 수준이 아니다. fold가 과제 내에서 독립이 아니라, 상호작용의 불확실성은 과제 군집 수준에서 커진다. baseline과 후보의 적합 seed 변동이 대응 Δ에서 상쇄되지 않으면 50-fold 짝짓기만으로는 적합 불확실성을 제거하지 못한다.

- **깨질 때 나타나는 관측 가능한 증상**: 무조건부 Δ와 발동 fold Δ의 부호·크기가 갈린다. 부호까지 갈리면 해당 estimand의 효과 주장은 반박되고, 부호는 같아도 크기가 estimand 선택에 따라 크게 달라지면 판별 불가로 남는다. combined가 PCA-only와 subsample-only의 합과 크게 어긋나지만 상호작용을 검출할 fold·seed 수가 부족해 판별 불가로 남는다. 과제·fold·seed subgroup에서 방향이 갈리고, seed를 바꾸면 Δ의 부호·크기가 바뀌어 단일 seed 50-fold Δ가 estimand의 점추정으로 해석할 수 없다. requested/effective 설정 해시 불일치가 확인되면 어떤 셀이 어떤 구성인지 귀속할 수 없어 실행 무효가 된다.

- **문서의 어떤 관측이 이 가정과 충돌하는가**: 문서와 주어진 배경에서 subsample은 context slide가 일정 수 이상인 fold에서만 발동한다. PROJECT.md §4.1은 fold가 과제 내에서 독립이 아니라고 실측했다(절대 macro SE 독립 0.85%p 대 과제 군집 4.34%p, 설계 효과 5.12배; Δ는 1.2배). 저장 실행 로그에서 같은 구성의 seed만 바꾼 GF 계열 적합 표준편차는 0.0021~0.0083으로 +0.0030보다 크다. RU-97에서 M=32−M=8은 적합 분산 포함 시 +0.0003 [−0.0067, +0.0073]로 0을 포함했고, 단일 적합의 +0.0099는 seed 42가 해당 구성 평균보다 1.5 sd 높았던 뽑기로 기록됐다. fold×seed 잔차가 23.9%로 seed 주효과보다 크다고 실측됐다. D-050 폴백으로 과거 산출물은 unverified-hold다.

### A1 (감사)

## 점검 1: estimand의 fold 집합 — subsample 발동 fold의 사전 고정
- **확인하려는 것**: "context slide가 일정 수 이상인 fold"의 임계값과 발동 fold 집합이 결과 보기 전에 고정되어 있는가. 그 fold 집합이 라벨·성능에 의존하지 않는가. 무조건부 50-fold Δ와 발동 fold 한정 Δ 중 어느 것이 estimand로 선언되는가.
- **문서에서 확인되는가**: 문서에 없음 (구체적 임계값, 발동 fold 수, estimand 선택 미기재)
- **샌다면 나타날 증상**: 발동 fold가 사후에 선택되면 estimand가 선택 편향으로 오염된다. 발동 fold 수와 범위가 고정되지 않으면 Δ의 해석이 불가능하다. 무조건부 50-fold Δ를 주 결과로 쓰면, 발동하지 않은 fold에서 baseline과 동일한 Δ=0이 희석되어 실제 효과가 과소추정된다. 반대로 발동 fold 한정 Δ를 쓰면 fold 수가 줄어 SE가 커지고, 그 fold 집합이 사전에 고정되지 않으면 "발동 fold만 보면 효과가 있다"는 사후 주장이 가능하다.
- **확인 방법**: 사전 등록 문서에 estimand 선언과 발동 fold 임계값·fold ID 목록이 기록되어 있는가. fold별 context slide 수 매니페스트(`docs/` 또는 `scripts/` 산출물)가 결과 조회 전에 생성되어 있는가. `scripts/`의 fold 분할 코드에서 임계값이 하드코딩되어 있는가.

## 점검 2: PCA fit 범위와 라벨 사용
- **확인하려는 것**: context feature PCA가 fold별 train만에서 fit되는가. 라벨을 사용하는가. fit 범위가 fold 간 정보 누출을 막는가. 주성분 수가 fold별로 고정되어 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 전체 50-fold에서 fit하면 fold 간 정보 누출로 Δ가 과대추정된다. 라벨을 쓰면 `CA-06`(라벨 사용 판별 기저) 경계 저촉 가능성이 있고, `closed_axes.md` §3-I의 "컨텍스트 라벨만 쓰는 판별 통계"와 충돌할 수 있다. 주성분 수가 fold별로 다르면 Δ가 PCA 효과가 아니라 주성분 수 차이에 기인한다.
- **확인 방법**: `src/`의 PCA 코드에서 fit이 fold별 train만인지 확인. requested/effective 설정 해시에 PCA fit 범위·주성분 수·라벨 사용 여부가 기록되어 있는가. `closed_axes.md` §3-I의 "금지(Oracle)" 조항과 대조.

## 점검 3: subsample 발동 fold의 baseline 대비 동일성
- **확인하려는 것**: subsample이 발동하지 않는 fold에서는 baseline과 완전히 동일한가. 발동 fold에서 subsample의 seed가 고정되어 있는가. combined 구성에서 PCA와 subsample의 적용 순서가 고정되어 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 발동하지 않는 fold에서 코드 경로·seed·전처리 순서의 미세한 차이가 있으면, Δ가 subsample 효과가 아니라 구현 차이로 새어 나온다. combined 구성에서 PCA→subsample과 subsample→PCA의 순서가 다르면, 상호작용 Δ가 순서 효과와 분리되지 않는다.
- **확인 방법**: baseline과 subsample-only 코드의 diff. 발동하지 않는 fold에서 requested/effective 해시가 baseline과 동일한가. combined 구성의 적용 순서가 사전 등록 문서에 고정되어 있는가.

## 점검 4: 적합 분산과 반복 횟수의 사전 고정
- **확인하려는 것**: +0.0030 임계값이 적합 분산 0.0021~0.0083(RU-97 실측)보다 작거나 같으므로, 단일 seed 단일 draw로는 무엇을 말할 수 없는가. 반복 횟수 R이 사전에 고정되어 있는가. R=8이 충분한 근거가 있는가.
- **문서에서 확인되는가**: 확인됨 (RU-97 §8~§10: 적합 4회, sd 0.0021/0.0050/0.0083, M=32−M=8 = +0.0003 [−0.0067, +0.0073], seed 42가 1.5 sd 높음, fold×seed 잔차 23.9%. `current_status.md` 미해결: R=8 잠정 보정, 사용자 승인 전 승격 판정 보류)
- **샌다면 나타날 증상**: 단일 seed로 +0.0030을 넘으면 성공으로 판정하면, seed 변동이 효과로 오인된다(RU-97에서 seed 42가 1.5 sd 높아 +0.0099를 냈으나 4회 평균에서는 +0.0003). R이 사전에 고정되지 않으면, 사후에 반복 횟수를 조절해 결과를 "통과" 또는 "실패"로 만들 수 있다. fold×seed 잔차 23.9%는 fold를 더 모아도 줄지 않으므로, fold 수 증대로 적합 분산을 줄일 수 없다.
- **확인 방법**: 사전 등록 문서에 R이 결과 보기 전에 고정되어 있는가. R=8의 근거(검출력 계산 또는 RU-97 실측 기반)가 문서에 있는가. `talks/reports/2026-09-17_ru97_gf_primary7.md` §8~§10의 분산분해 수치와 대조.

## 점검 5: 상호작용 검출력
- **확인하려는 것**: 2×2 설계에서 PCA×subsample 상호작용을 검출할 검출력이 있는가. fold 50 × 과제 7 = 350 fold, 적합 분산 0.0021~0.0083 포함 시.
- **문서에서 확인되는가**: 문서에 없음 (상호작용 검출력 계산 없음)
- **샌다면 나타날 증상**: 상호작용이 작으면 검출 불가인데, 사후에 "combined이 둘 중 어느 것보다 낫다"고 주장하면, 그 차이가 적합 분산 내인지 구분할 수 없다. RU-80의 대응 Δ 표준편차 s_d=0.0033, 군집 SE 0.022%p, MDE 0.072%p(α=0.05, power 0.80, df=6)는 단일 효과의 검출력이며, 상호작용은 4개 셀의 차이의 차이이므로 검출력이 더 낮다. 적합 분산이 추가되면 상호작용 MDE는 단일 효과 MDE보다 크다.
- **확인 방법**: 사전 등록 문서에 상호작용 MDE 계산이 있는가. 4개 셀(baseline/PCA-only/subsample-only/combined) 각각의 R회 적합 평균 Δ의 SE가 산출되어 있는가. 상호작용이 "판별 불가"로 사전 선언되어 있는가.

## 점검 6: 폴백과 설정 귀속
- **확인하려는 것**: requested/effective 설정 해시가 기록되어, 각 산출물이 어떤 구성으로 만들어졌는지 귀속 가능한가. 폴백이 작동하면 requested와 effective가 달라지는데, 이 차이가 감지되는가.
- **문서에서 확인되는가**: 확인됨 (`D-050`: `src/`가 환경변수를 읽지 않아 `ICF_*` export 무효, `SH`/`SJ` 켜는 수단 동작하지 않음, 저장 예측 파일에 branch list·가중치·코드 해시 미기록. 새 실행에 requested/effective 해시 기록)
- **샌다면 나타날 증상**: 폴백이 작동하면 PCA-only가 실제로 combined으로, 또는 baseline으로 실행되어 2×2 셀이 혼동된다. requested/effective 해시가 기록되지 않으면, 과거와 같은 unverified-hold 상태가 반복된다.
- **확인 방법**: 실행 로그에 requested/effective 해시가 모두 기록되어 있는가. requested ≠ effective인 경우 실행이 중단되는가. `src/`의 폴백 코드가 제거되었거나, 폴백 발생 시 에러를 내는가.

## 점검 7: 기준선의 독립성
- **확인하려는 것**: baseline이 후보의 설정(PCA 주성분 수, subsample 임계값, 적용 순서)을 알고 있는가. baseline이 후보의 fold별 발동 여부를 알고 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: baseline이 후보의 정보를 쓰면 Δ가 과소추정(후보가 baseline을 "맞추어" 실행) 또는 과대추정(baseline이 후보의 약점을 회피)된다. 예를 들어 baseline이 subsample 발동 fold에서 동일한 subsample을 적용하면, Δ가 0이 되어 효과가 사라진다.
- **확인 방법**: baseline과 후보의 코드·설정 diff. baseline의 requested/effective 해시에 후보 관련 파라미터가 없는가. fold별 실행 로그에서 baseline과 후보가 동일한 fold·seed·데이터에서 실행되었는가.

## 점검 8: 자원 비용과 대안
- **확인하려는 것**: GPU 1장, 과제 7개, fold 50개, 2×2 셀, R회 반복 기준 대략의 비용. `current_status.md` 예산(합계 최대 4 GPU-hour) 내에서 실행 가능한가.
- **문서에서 확인되는가**: 문서에 없음 (2×2 프로브의 구체적 GPU-hour 미기재. `current_status.md` 예산: NEXGEM B200 1장 합계 최대 4 GPU-hour, 단일 M 2시간 초과 시 중단)
- **샌다면 나타날 증상**: 2×2 셀 × R회 × 7 과제 × 50 fold의 총 실행 시간이 4 GPU-hour를 초과하면 프로브가 실행 무효가 된다. R=8이면 4 셀 × 8 = 32회 적합이 필요하고, 각 적합이 7 과제 × 50 fold를 처리하므로, 단일 M 2시간 제한에 걸릴 수 있다.
- **확인 방법**: `current_status.md`의 예산·중단 조건과 대조. 1회 적합(7 과제 × 50 fold)의 GPU-hour가 문서에 기록되어 있는가. R=8 × 4 셀 = 32회 적합의 총 시간이 4 GPU-hour 이내인가. 초과 시 "이 프로브로는 판별 불가"가 사전에 선언되어 있는가.

---

## Phase 2 — 교차

### R1 (반증)

## 반론 1
- **공격 대상**: P1의 “무조건부 50-fold paired Δ를 주 estimand로 고정하고, PCA-only와 subsample-only를 모두 primary로, combined는 진단으로만 둔다”는 설계.
- **더 단순한 설명**: 무조건부 Δ가 작거나 0에 가까운 관측은 “subsample 효과가 없다”는 것보다, subsample이 발동하지 않는 fold에서 candidate가 baseline과 동일하거나 거의 동일해져 효과가 희석된 것일 수 있다. 발동 fold 비율은 문서에 없으므로, 발동이 드문 경우 무조건부 estimand는 subsample 축에 대해 거의 민감도가 없다. 또한 “PCA-only 또는 subsample-only 중 하나가 +0.0030을 넘는다”는 관측은 진짜 축 효과보다, 두 축을 동시에 검정할 때 우연히 하나 이상이 임계값을 넘는 다중 비교, 또는 combined에 포함된 다른 축의 효과로 설명될 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 만약 P1의 판정 규칙이 PCA-only와 subsample-only를 독립적으로 확인적 근거로 쓰도록 허용하면서, 단일 primary contrast, 계층적 검정, 또는 다중 비교 통제를 고정하지 않으면, 이 제안은 “한 축이 임계 초과”라는 관측을 축 귀속으로 오인할 수 있다. 특히 무조건부 Δ가 실패했는데 발동 fold Δ가 큰 경우를 “2차 결과”로만 두면서도, P1 본문에 발동 fold Δ로 주 결과를 바꾸는 사후 이탈을 명시적으로 금지하는 조항이 없으면, estimand 선언이 불완전하다. P3가 사후 재해석 금지를 추가하지만, P1 자체의 estimand 선언만으로는 구멍이 남는다.
- **이 반론의 한계**: `PROJECT.md` §4의 승격 산출식이 원래 7 과제 × 50 fold 전체의 paired Δ를 쓰므로, 승격 결정의 estimand로 무조건부 Δ를 고르는 것은 정합적이다. 또한 발동 fold 비율이 실제로 높고, 이 프로브가 확인적 승격이 아니라 탐색적 귀속이라면 두 축을 동시에 primary로 두는 것이 허용될 수 있다.

## 반론 2
- **공격 대상**: P2의 “확인적 R=16”, “총 SE는 과제 군집 SE와 seed-level SE 중 더 큰 값”, “point Δ ≥ +0.0030이고 총 SE < +0.0030이면 지지” 판정 규칙.
- **더 단순한 설명**: 실측된 적합 분산 0.0021~0.0083은 GF 계열의 재적합에서 나온 macro 변동이지, PCA/subsample 프로브의 matched-seed paired Δ 분산이 아니다. 같은 “적합 분산이 크다”는 관측은 (1) GF 최적화 특유의 seed 민감도, (2) fold×seed residual, (3) macro 절대값 변동 중 어느 것에서든 나올 수 있다. 따라서 R=16은 이 프로브의 실제 Δ 분산을 근거로 한 필수 반복 횟수라기보다, 다른 계열의 worst-case 상한을 그대로 이식한 값일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 만약 PCA/subsample에서 baseline과 candidate를 같은 seed·같은 fold로 짝지었을 때 seed 주효과가 상당 부분 상쇄되고, 남은 변동이 fold×seed residual이라면, R=16으로 seed-level SE만 줄여도 총 SE가 +0.0030 아래로 내려가지 않을 수 있다. `current_status.md`의 “fold×seed 잔차가 23.9%로 seed 주효과보다 크다”는 관측은 이 가능성을 지지한다. 기전은 미확인이다. 또한 P2의 총 SE를 두 SE 중 더 큰 값으로 두는 것은 fold/과제 군집 변동과 seed 변동을 결합한 총 불확실성이 아니라, 두 추정량 중 하나를 선택하는 것에 가깝다. 독립 성분이 있다면 총 SE는 일반적으로 둘의 합성이어야 하므로, max 규칙은 실제 총 분산을 과소 또는 과대 추정할 수 있다. 결정 규칙 “point Δ ≥ +0.0030이고 총 SE < +0.0030”도 가설 검정의 제1종 오류를 통제하지 않는다. 진짜 효과가 0이어도 점추정이 임계 근처에서 흔들리면 “지지”로 분류될 수 있다.
- **이 반론의 한계**: 만약 PCA/subsample의 실제 seed 분산이 GF worst-case와 비슷하고, matched-seed Δ가 독립 macro 변동에 가깝다면 R=16은 보수적으로 합리적이다. 또한 `PROJECT.md` §4의 승격 기준은 CI 하한이 아니라 point Δ ≥ +0.0030이므로, P2의 SE 게이트는 프로젝트 규범보다 강한 추가 보수 장치일 뿐, §4 자체의 필수 조건은 아니다.

## 반론 3
- **공격 대상**: P3의 “상호작용은 사전에 판별 불가로 선언”, “combined만 좋으면 attribution 실패”, “combined point Δ ≥ +0.0030이어도 단일 축 실패라면 프로브 실패” 조건.
- **더 단순한 설명**: 과거 combined 실행의 효과는 상호작용이 아니라, 한 축의 main effect에 seed/fold 잡음이 겹친 것일 수 있다. 따라서 상호작용을 확인적으로 검정하지 않는 것은 더 단순한 설명에 부합한다. 그러나 P3의 “combined만 좋으면 attribution 실패”는 또 다른 더 단순한 설명을 놓칠 수 있다. 한 축은 실제로 효과가 있고, 다른 축은 음의 효과를 내어 단일 cell에서 희석 또는 마스크된 경우, combined만 임계 초과할 수 있다. 이 경우 combined 신호는 두 축 중 하나를 확인한 증거는 아니지만, 새로운 조합 후보로서의 탐색적 신호일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 만약 이 프로브의 목표가 “PCA와 subsample 중 어느 축이 원인인지를 확인하는 것”뿐이라면, combined-only 신호는 그 목표에 대한 확인적 증거가 아니므로 P3의 처리는 정합적이다. 그러나 목표가 “Primary 7에서 모델 개선 근거를 얻는 것”이기도 하면, combined가 +0.0030을 넘을 때를 단순히 “프로브 실패”로만 라벨링하는 것은 틀린다. 그것은 attribution 실패이지, 개선 신호의 부정이 아니다. 자원이 1회뿐이라면, combined 신호를 “별도 사전 등록 필요”로 넘기는 것은 실제로는 신호를 버리는 것과 같아질 수 있다. 또한 상호작용 variance가 문서에 없다는 이유로 무조건 “판별 불가”로 선언하면, 350 paired fold가 충분히 큰 interaction contrast를 잡을 수 있는 상황이 있어도 사전에 정보를 포기하게 된다.
- **이 반론의 한계**: 이 프로브의 유일한 확인적 목표가 두 축 중 하나의 귀속이라면, combined-only 성공은 그 목표에 대한 증거가 아니다. interaction contrast의 variance와 power가 문서에 없으므로, 사전에 “판별 불가”로 두는 것은 정직한 선택일 수 있다.

## 반론 4
- **공격 대상**: P2/P3의 자원 계획, 특히 R=16 2×2를 확인적 프로브로 확정하는 것.
- **더 단순한 설명**: 현재 귀속이 불가능한 더 단순한 원인은 과거 실행의 silent fallback과 unverified-hold 상태이지, 효과가 작아서 반복이 부족한 것이 아니다. 따라서 먼저 필요한 것은 R=16 확인적 실험이 아니라, requested/effective hash가 실제로 기록되는지, 4 cell의 effective setting이 intended와 일치하는지, 7×50 fold manifest가 누락 없는지를 확인하는 최소 provenance 실행일 수 있다.
- **무엇이 참이면 이 주장이 무너지는가**: 1 fold-fit의 GPU 시간은 문서에 없으므로, R=16 2×2의 22,400 fold-fits가 몇 GPU-hour인지 문서 근거로 추정할 수 없다. 만약 실제 per-fold-fit 시간이 예산을 초과하면, R=16은 실행 무효 또는 중단으로 갈 수 있고, “마지막 기회”가 아니라 예산 초과 설계가 된다. R=4 diagnostic만 가능하면 P2/P3의 규칙은 “판별 불가”로 수렴한다. 이 경우 2×2 전체보다, (a) provenance 검증 + R=1 point estimate, (b) 한 축만 baseline 대비 반복, (c) fold/seed variance를 재는 작은 고정 subset diagnostic가 더 단순하고 실행 가능한 대안이다.
- **이 반론의 한계**: 만약 1 fold-fit 시간이 충분히 짧아 22,400 fold-fits가 예산 내이고, attribution이 최우선이라면 2×2 R=16은 정당한 확인적 설계일 수 있다. 또한 P3는 baseline을 4 cell 중 하나로 이미 비용에 포함하고 있으므로, “baseline 재사용으로 비용을 줄인다”는 주장은 이 설계에서는 효과가 없을 수 있다.

### M1 (실측)

## 실측 1: 제안 P1의 2×2 프로브 실현 가능성
- **건드려야 하는 코드·설정**:
  - `src/` 전체: D-050에 따르면 환경변수를 하나도 읽지 않아 `ICF_*` export가 무효. PCA를 켜는 경로, subsample을 켜는 경로, seed를 전달하는 경로가 현재 존재하지 않음.
  - 실행 스크립트: 폴백이 requested 설정을 조용히 다른 설정으로 교체. requested/effective 해시 기록을 구현하려면 폴백 제거 또는 우회 코드가 필요. 현재 존재 여부 미확인.
  - provenance writer: 현재 목표에 포함되어 있음. requested/effective 해시, branch list, 가중치, 코드 해시, 데이터 버전, fold manifest 해시 기록에 의존. 미구현.
  - 오염 검사: §3.4 재설계 미완료. 재설계 완료 전까지 실행 무효.
  - PCA 적용 코드: 차원, 적용 시점, 정규화 방식 미확정. LSAK 사양 미기재와 동일한 결함.
  - subsample 적용 코드: 강도, 발동 조건("context slide가 일정 수 이상"의 "일정 수"), 발동 시 슬라이드 선택 방식 미확정. CA-R1에 "과제별 강도를 고를 선택 신호가 없다"고 명시.
  - matched-seed 구현: baseline과 candidate를 같은 seed, 같은 fold로 짝지우는 경로 미확인. D-050에 따르면 seed 전달 경로도 무효일 수 있음.
  - 총 SE 계산: seed-averaged fold-level Δ의 과제 군집 SE와 R개 seed-level Δ 평균의 SE를 계산하는 코드 미구현.
  - baseline 7-branch 재실행: D-050에 따르면 SH/SJ를 켜는 문서화된 수단이 동작하지 않음. baseline 재실행 불가.
  - 게이트 ①: PCA/subsample이 신규 브랜치에 해당하는지 미확인. 해당 시 SCREEN_ONLY 경로가 무효(D-050).

- **비용**:
  - 1 fold-fit GPU 시간: 미실측. 문서에 없음.
  - 1 seed 4 cell = 1,400 fold-fits: 미실측.
  - R=16 2×2 = 22,400 fold-fits: 미실측.
  - R=8 2×2 = 11,200 fold-fits: 미실측.
  - R=4 2×2 = 5,600 fold-fits: 미실측.
  - 예산: NEXGEM B200 1장 합계 최대 4 GPU-hour.
  - 메모리: 미실측.
  - 연산: 미실측.
  - R=16이 예산 내인지 판단 불가. 1 fold-fit 시간 실측이 선행 조건.
  - 제안의 "R=16의 SE 약 0.0029, R=8의 SE 약 0.0041"은 GF 계열 seed sd 0.0083을 PCA/subsample에 적용하고 candidate·baseline 독립을 가정한 추정치. PCA/subsample의 seed sd는 미실측.

- **막는 것**:
  1. `src/`가 환경변수를 읽지 않아 PCA/subsample을 켜는 경로가 무효 (D-050). 제안은 "새 실행에서 requested/effective 설정 해시를 함께 남긴다"고 하지만, requested 설정을 전달하는 경로 자체가 존재하지 않음.
  2. 실행 스크립트의 폴백이 requested 설정을 조용히 교체. 폴백 제거 또는 우회 코드가 현재 존재하는지 미확인.
  3. 1 fold-fit GPU 시간이 미실측. R=16 (22,400 fold-fits)이 4 GPU-hour 예산 내인지 판단 불가. 1 seed 4 cell (1,400 fold-fits)의 시간도 미실측.
  4. PCA 사양(차원, 적용 시점, 정규화) 미확정. LSAK 사양 미기재와 동일한 결함.
  5. subsample 사양(강도, 발동 조건, "일정 수") 미확정. CA-R1에 "과제별 강도를 고를 선택 신호가 없다"고 명시.
  6. matched-seed 구현 경로 미확인. D-050에 따르면 seed 전달 경로도 무효일 수 있음.
  7. 총 SE 계산 코드 미구현. 제안의 판정 규칙("총 SE < +0.0030")을 적용할 수 없음.
  8. 오염 검사 재설계 미완료. §3.4에 "재설계가 완료될 때까지 오염 검사 통과를 주장하는 실행은 무효"라고 명시.
  9. provenance writer 미구현. requested/effective 해시 기록에 의존.
  10. 적합 분산 0.0021~0.0083은 GF 계열 실측. PCA/subsample에 적용하는 것은 추정치. 미실측.
  11. SH/SJ를 켜는 문서화된 수단이 동작하지 않음 (D-050). baseline 7-branch 재실행 불가. paired Δ 계산 불가.
  12. 게이트 ① SCREEN_ONLY 경로 무효 (D-050). PCA/subsample이 게이트 ① 대상인지 미확인.

- **중단 기준**:
  1. 1 seed 4 cell (1,400 fold-fits) 실행 시간이 4 GPU-hour를 초과하면 R=16 불가. R=4 2×2 (5,600 fold-fits) variance/provenance diagnostic으로 전환.
  2. `src/` 환경변수 전달 경로 복구가 불가능하면 실행 무효.
  3. 폴백 제거/우회가 불가능하면 실행 무효.
  4. PCA/subsample 사양 확정이 불가능하면 실행 무효.
  5. 오염 검사 재설계 완료 전까지 실행 불가.
  6. provenance writer 구현 전까지 requested/effective 해시 기록 불가 → 실행 무효.
  7. matched-seed 구현이 불가능하면 단일 seed로 전환 → 판별 불가.
  8. 총 SE 계산이 불가능하면 판정 규칙 적용 불가 → 판별 불가.
  9. baseline 7-branch 재실행이 불가능하면 paired Δ 계산 불가 → 실행 무효.
  10. 게이트 ①이 적용되고 SCREEN_ONLY 경로가 복구 불가능하면 실행 무효.

### T1 (이론)

- **가정**: P1은 `seed-averaged fold-level Δ`의 과제 군집 SE와 `R`개 `seed-level Δ` 평균의 SE 중 더 큰 값이 총 SE를 유효하게 대표한다고 두고 있다. 즉 fold/task 이질성과 적합 seed 이질성, 그리고 재적합에 의한 seed×fold 변동이 동시에 크지 않거나, 그 변동이 두 SE 중 하나에 이미 흡수된다고 두고 있다.

- **이 가정이 깨지는 조건**: 두 불확실성 성분이 같은 크기대로 공존하거나, seed마다 fold별 Δ의 크기와 부호가 달라지는 seed×fold 상호작용이 존재하거나, GF 계열의 seed std `0.0021~0.0083`이 본 PCA/subsample 파이프라인의 적합 분산 구조를 대표하지 못하는 경우.

- **깨질 때 나타나는 관측 가능한 증상**: R=4 diagnostic에서 과제 군집 SE와 seed-level SE가 모두 `+0.0030` 근처로 나오고, mixed-model/bootstrap 총 SE가 max 값보다 크게 나오는 것; seed별 fold-level Δ의 부호나 과제별 순위가 seed마다 바뀌는 것; fold×seed residual이 fold 주효과와 seed 주효과보다 큰 것. 이 경우 `총 SE < +0.0030` 게이트가 정밀성을 과소평가하고, 점추정 `+0.0030` 초과를 지지로 잘못 분류하거나, 실제 판별 불가인 상태를 지지로 잘못 분류할 수 있다.

- **문서의 어떤 관측이 이 가정과 충돌하는가**: `current_status.md`의 RU-97 분산분해는 seed 주효과가 전체의 `0~2.9%`로 작으면서도 seed sd가 기준의 최대 `2.8배`이고, `m32_raw`의 fold×seed 잔차가 `23.9%`로 seed 주효과보다 크다고 실측한다. `PROJECT.md` §4.1은 대응 per-fold Δ의 표준편차 `0.0033`과 과제 군집 SE `0.022%p`를 기록해 fold/task 이질성도 별도 성분으로 존재함을 보여준다. 이 관측들은 seed×fold 변동의 기전을 입증하지는 않지만, “더 큰 SE 하나만 취하면 총 SE가 된다”는 가정에 대한 충돌 관측이다.

### A1 (감사)

## 점검 1: PCA fit 범위
- **확인하려는 것**: PCA가 train fold의 context feature에서만 `fit()`되고, test fold에는 `transform()`만 적용되는지.
- **문서에서 확인되는가**: 문서에 없음. 제안은 "context feature에 PCA를 적용하는 것"이라고만 하고, fit/transform의 데이터 범위에 대해 언급하지 않는다.
- **샌다면 나타날 증상**: PCA가 train+test 전체에서 fit되면, test fold의 context feature 분포가 train에 새어 나온다. 주성분이 test fold의 구조를 반영하여 candidate의 AUROC가 인위적으로 상승한다. paired Δ가 실제 효과보다 과대 추정되고, +0.0030 임계를 넘더라도 이는 fit 범위 누수의 산물일 수 있다.
- **확인 방법**: `src/`의 PCA 관련 코드에서 `fit()` 호출의 입력이 train fold의 context feature만인지 감사. 실행 스크립트에서 fold별 PCA fit/transform의 데이터 범위를 확인. `requested/effective` 설정 해시에 PCA fit 범위가 포함되는지 확인.

## 점검 2: subsample의 적용 대상 (train vs test)
- **확인하려는 것**: subsample이 train fold의 context slide에서만 수행되는지, test fold의 context slide에도 적용되는지.
- **문서에서 확인되는가**: 문서에 없음. 제안은 "context slide를 subsample 하는 것"이라고만 하고, 적용 대상이 train인지 test인지 명시하지 않는다.
- **샌다면 나타날 증상**: test fold의 context slide가 subsample되면, candidate의 test set이 baseline의 test set과 달라진다. §4의 paired Δ는 "같은 fold에서 짝지은" 차이를 전제하므로, test set이 다르면 paired Δ의 전제가 위반된다. candidate가 더 적은 context slide로 평가되어 AUROC가 하락하거나, test fold의 어려운 slide가 제거되어 AUROC가 상승할 수 있다. 어느 방향이든 Δ는 subsample의 효과와 test set 변경의 효과가 혼합된다.
- **확인 방법**: 실행 스크립트에서 subsample이 train fold의 context slide에서만 수행되는지 코드 감사. baseline과 candidate의 test fold context slide 수가 fold별로 동일한지 확인. `requested/effective` 설정 해시에 subsample 적용 대상이 포함되는지 확인.

## 점검 3: 발동 조건 임계값의 사전 고정
- **확인하려는 것**: "context slide가 일정 수 이상"의 "일정 수"가 결과 보기 전에 고정되었는지, 그리고 이 임계값이 fold의 어떤 속성(train context slide 수, test context slide 수)에 기반하는지.
- **문서에서 확인되는가**: 문서에 없음. 제안은 "발동 조건과 발동 fold manifest는 결과 보기 전에 고정"이라고 하지만, 임계값 자체와 발동 조건의 정의(어떤 fold의 어떤 속성)는 문서에 없다.
- **샌다면 나타날 증상**: 임계값이 사후에 조정되면, 발동 fold의 선택이 성능에 의존한다. 발동 fold Δ가 인위적으로 상승하고, 무조건부 Δ와 발동 fold Δ의 차이가 임계값 선택의 산물이 된다. "발동 fold 한정 Δ"를 2차 지표로 두더라도, 이 지표의 신뢰성은 임계값의 사전 고정에 의존한다.
- **확인 방법**: 사전 등록 문서에서 발동 임계값이 명시적으로 기록되었는지 확인. 발동 조건의 정의(어떤 fold의 어떤 속성)가 결과 보기 전에 고정되었는지 확인. 실행 로그에서 발동 fold의 context slide 수 분포와 임계값의 관계를 확인.

## 점검 4: baseline의 구성과 기준선
- **확인하려는 것**: 새 baseline이 §3.2의 공식 7-branch 구성(`CV,BM,BD,QA,DS,SH,SJ`, `ICF_FIXED_HEAD_CT_WEIGHT=0.0`)과 동일한지, 그리고 새 baseline의 Primary 7 macro가 0.6227과 일치하는지.
- **문서에서 확인되는가**: 모순됨. 제안은 "과거 unverified-hold 값이 아니라 같은 seed set으로 새 실행한다"고 하지만, 새 baseline의 구성이 §3.2의 공식 구성과 동일한지, 그리고 0.6227과 일치하는지 언급하지 않는다. §3.2의 0.6227은 unverified-hold 상태이므로, 새 baseline이 0.6227과 다를 수 있다.
- **샌다면 나타날 증상**: 새 baseline이 0.6227과 다르면, 승격 기준 +0.0030의 기준선이 모호해진다. candidate가 새 baseline 대비 +0.0030을 넘지만 0.6227 대비로는 미달일 수 있고, 그 반대도 가능하다. §4의 승격 산출식은 "후보−기준선 AUROC"를 쓰지만, "기준선"이 0.6227인지 새 baseline인지 명확하지 않으면 +0.0030의 의미가 달라진다.
- **확인 방법**: 새 baseline의 실행 구성이 §3.2의 7-branch 구성과 동일한지 provenance 기록으로 확인. 새 baseline의 Primary 7 macro가 0.6227과 얼마나 다른지 확인. 승격 기준 +0.0030의 기준선이 새 baseline인지 0.6227인지 사전 등록 문서에 명시되었는지 확인.

## 점검 5: SE 게이트의 통계적 적절성
- **확인하려는 것**: "point Δ ≥ +0.0030이고, 총 SE < +0.0030"이 95% CI가 0을 배제하는 것과 동등한지, 또는 최소한 0을 배제하는 조건을 만족하는지.
- **문서에서 확인되는가**: 모순됨. point Δ = 0.0030, SE = 0.0029이면, 95% CI 하한은 0.0030 − 1.96×0.0029 = −0.0027로 0을 포함한다. 제안의 "지지" 조건은 0을 배제하지 못하는 결과를 지지로 판정할 수 있다.
- **샌다면 나타날 증상**: 0을 배제하지 못하는 결과가 "지지"로 판정된다. 승격 기준이 통계적으로 지지되지 않는 구성이 통과한다. §4는 "95% 신뢰구간 하한에 +0.3%p를 요구하는 기준이 아니다"라고 하지만, SE 게이트가 최소한 0을 배제하는 조건을 만족하는지는 별개다.
- **확인 방법**: 제안의 SE 게이트를 95% CI 하한 > 0으로 재정의하는지 확인. 또는 "point Δ ≥ +0.0030"과 "총 SE < +0.0030"이 함께 만족될 때, 95% CI 하한이 0을 배제하는지 수치적으로 확인. §4의 승격 기준과 SE 게이트의 관계를 사전 등록 문서에 명시했는지 확인.

## 점검 6: interaction contrast의 MDE 사전 계산 누락
- **확인하려는 것**: 350 fold pair, 7 과제 군집에서 interaction contrast의 MDE를 사전에 계산할 수 있는지, 그리고 "판별 불가" 선언이 이 계산에 기반하는지.
- **문서에서 확인되는가**: 문서에 없음. 제안은 "interaction contrast의 variance는 문서에 없다"고 하고, 사전에 "판별 불가"로 선언한다. 하지만 §4.1의 MDE 계산 방법(α=0.05 양측, power 0.80, 군집 t, df=6)을 interaction contrast에 적용하면, 4 cell의 paired Δ의 variance를 사전에 추정하여 MDE를 계산할 수 있다.
- **샌다면 나타날 증상**: 상호작용이 실제로 검출 가능할 수 있는데, 사전에 "판별 불가"로 선언하여 정보 손실이 발생한다. 또는 상호작용이 검출 불가능한데, 사후에 "상호작용이 보인다"고 주장하는 구멍이 열린다. "판별 불가" 선언의 근거가 "문서에 없다"는 사실만으로는 부족하다.
- **확인 방법**: §4.1의 MDE 계산 방법을 interaction contrast에 적용. 4 cell의 paired Δ의 variance를 사전에 추정(과거 실행 로그의 fold-level Δ variance를 사용). 7 과제 군집의 df=6에서 interaction contrast의 MDE를 계산. 이 MDE가 +0.0030보다 큰지 작은지 확인.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

- **R1-1**: 수용. P1을 수정한다. 주 estimand는 baseline 대비 무조건부 7×50 paired Δ로 고정하고, 발동 fold Δ는 2차 진단으로만 사용하며 결과 확인 후 주 estimand로 교체하는 것을 명시적으로 금지한다. PCA-only와 subsample-only를 “최소 하나 성공”이라는 합성 확인적 endpoint로 쓰지 않고, 각 축별 candidate-specific estimand로만 둔다. 단일 축만 임계 초과하면 “해당 축 후보에 대한 확인적 지지, combined attribution 미확립”으로만 기록하고, combined가 임계 초과해도 단일 축이 미달이면 이 프로브의 확인적 귀속은 실패이며 combined는 새 탐색 후보로만 넘긴다. 근거: R1은 P1 본문에 사후 estimand 전환 금지와 다중 primary 통제 부재를 지적했고, `PROJECT.md` §4는 고정 350-fold paired Δ를 승격 계산에 쓰므로 무조건부 Δ와 정합하다.

- **R1-2**: 수용. P2를 수정한다. R=16은 GF 계열 seed sd `0.0083`을 worst-case로 이식한 계획값이지, PCA/subsample 프로브의 실측 근거가 아니다. 확인적 R=16 전에 R=4 2×2 diagnostic로 seed, fold, seed×fold 분산과 총 SE를 먼저 측정한다. 총 SE는 과제 군집 SE와 seed-level SE 중 max가 아니라 seed-averaged fold-level Δ에 task/seed clustering을 반영한 mixed model 또는 bootstrap으로 추정한다. 독립성 미확인 시 보수적 합성 또는 bootstrap을 쓰고, 계산 불가하면 판별 불가로 둔다. “지지”는 point Δ ≥ `+0.0030`과 95% CI 하한 > 0을 함께 요구한다. 근거: `current_status.md` RU-97은 fold×seed 잔차 `23.9%`가 seed 주효과보다 크다고 실측했고, `PROJECT.md` §4.1은 과제 군집 SE `0.022%p`를 기록해 seed 변동과 fold/task 변동이 별개 성분일 수 있음을 보여준다. seed×fold 기전은 미확인이다.

- **R1-3**: 수용. P3를 수정한다. combined point Δ ≥ `+0.0030`이고 단일 축 둘 다 미달이면, 라벨을 “attribution 실패 + combined 탐색 신호”로 고정한다. 이 프로브에서 combined를 승격 근거로 쓰지 않고, 새 후보로서의 탐색 신호만 기록하며 후속 사전 등록과 선택 이력 기록을 요구한다. 상호작용은 “variance 문서 없음”만으로 일괄 판별 불가가 아니라, 저장 fold-level Δ variance가 있으면 §4.1 방법으로 interaction contrast MDE를 사전 계산하고, 없으면 R=4 diagnostic에서 추정한다. MDE/SE를 계산할 수 없으면 상호작용만 판별 불가로 둔다. 근거: R1은 combined가 모델 개선 신호일 수 있음을 지적했고, P3의 “프로브 실패”는 attribution 실패와 개선 신호 부정을 혼동했다. combined 기전은 미확인이다.

- **R1-4**: 수용. P2/P3 자원 계획을 수정한다. 확인적 R=16 전에 provenance 최소 실행을 먼저 둔다. requested/effective 해시 기록, 폴백 탐지, 7×50 fold manifest 누락 검사, baseline 7-branch 재실행, 1 fold-fit 시간을 측정한다. 이 중 하나라도 실패하면 실행 무효. 4 GPU-hour 예산에서 R=16 2×2가 불가능하면 R=4 2×2 diagnostic로 전환하고 확인적 판별은 판별 불가로 종료한다. 근거: `current_status.md`는 예산을 4 GPU-hour로 고정했고, M1은 1 fold-fit GPU 시간이 문서에 없으므로 R=16 실행 가능성을 아직 확인할 수 없다.

- **M1**: 수용. 모든 P 제안을 “계측 복구 완료 전에는 확인적 결과로 무효”라는 전제 게이트로 수정한다. 착수 전 `src/` 환경변수 경로 복구, 폴백 제거 또는 우회, provenance writer, PCA 사양(fit 범위·차원·정규화), subsample 사양(발동 임계·선택 방식·적용 대상), matched-seed 경로, 총 SE 계산 코드, baseline 7-branch 재실행, 오염 검사 재설계 통과를 완료해야 한다. PCA/subsample가 신규 브랜치로 분류되면 게이트 ① `SCREEN_ONLY` 복구도 선행 조건이다. 이 중 하나라도 불가능하면 실행 무효 또는 판별 불가. 근거: M1은 `D-050`으로 현재 경로가 무효이고, `PROJECT.md` §3.4는 재설계 전 오염 검사 통과를 주장하는 실행을 무효로 규정했다.

- **T1**: 수용. P2의 총 SE=max 규칙을 폐기한다. seed×fold 상호작용이 있으면 max는 총 불확실성을 과소평가할 수 있다. 총 SE는 seed-averaged fold-level Δ에서 task cluster와 seed cluster를 함께 반영하는 mixed model 또는 bootstrap으로 추정한다. 성분이 독립이면 제곱합, 독립이 아니면 bootstrap을 쓰고, 계산 불가 시 확인적 지지 대신 판별 불가로 둔다. 근거: T1의 가설은 `current_status.md`의 fold×seed 잔차 `23.9%`와 `PROJECT.md` §4.1의 과제 군집 SE와 충돌한다. seed×fold 기전은 미확인이다.

- **A1-1**: 수용. P1에 PCA fit 범위를 고정한다. PCA는 train fold의 context feature에서만 fit하고, test fold에는 transform만 적용한다. fit/transform 범위를 requested/effective 해시에 기록한다. test fold에 fit이 포함되면 실행 무효. 근거: A1은 fit 범위 누수가 paired Δ를 과대 추정할 수 있음을 지적했고, 문서에는 PCA fit 범위가 없다.

- **A1-2**: 수용. P1에 subsample 적용 대상을 고정한다. subsample은 test fold에서 평가하는 query slide 집합을 바꾸면 안 된다. train/context 구성에만 적용하거나, 추론 시 context slide만 줄이는 inference-time 규칙으로 명시하되 query slide와 fold test set은 동일해야 한다. test query slide를 제거하거나 baseline과 candidate의 평가 대상 slide가 다르면 paired Δ 전제 위반으로 실행 무효. 적용 대상을 requested/effective 해시에 기록한다. 근거: A1은 paired Δ가 같은 fold의 대응 비교를 전제하므로 test set 변경이 Δ를 오염시킨다고 지적했고, 문서에는 적용 대상이 없다.

- **A1-3**: 수용. P1에 발동 조건을 고정한다. “context slide가 일정 수 이상”의 임계값, 그 slide 수의 정의(train context/test context/episode 전체), 발동 fold manifest를 결과 보기 전에 고정한다. 임계값이 사후 조정되면 발동 fold Δ는 신뢰할 수 없고, 무조건부 Δ와의 차이도 해석 불가이므로 실행 무효. 근거: A1은 임계값이 문서에 없고 사후 조정 시 선택 편향이 생길 수 있음을 지적했다.

- **A1-4**: 수용. P2/P3에 baseline 기준선을 고정한다. 새 baseline은 §3.2 공식 7-branch(`CV,BM,BD,QA,DS,SH,SJ`, `ICF_FIXED_HEAD_CT_WEIGHT=0.0`)를 같은 seed/fold로 재실행한 값이다. 승격/지지 판단의 `+0.0030`은 후보−새 baseline paired Δ로 계산한다. `0.6227`은 `unverified-hold`이므로 절대 기준선으로 쓰지 않고, 새 baseline macro와 `0.6227` 간 차이는 provenance 참고값으로만 기록한다. 근거: A1은 `0.6227`이 `unverified-hold`이고 새 baseline과 다를 수 있음을 지적했고, `D-050`은 이 값을 `verified`로 인용하지 않는다.

- **A1-5**: 수용. P2의 지지 규칙을 수정한다. “point Δ ≥ `+0.0030`이고 총 SE < `+0.0030`”만으로는 0을 배제하지 못한다. 확인적 지지는 point Δ ≥ `+0.0030`과 95% CI 하한 > 0(또는 사전 고정한 α=0.05 검정 통과)을 함께 요구한다. point만 충족하고 CI가 0을 포함하면 “승격 기준 점수 충족, 확증적 지지 불충분”으로 기록한다. 근거: A1의 수치 예(point `0.0030`, SE `0.0029` → 95% CI 하한 음수)는 SE 게이트가 제1종 오류를 통제하지 못함을 보여준다.

- **A1-6**: 수용. P3에 interaction MDE 사전 계산 단계를 추가한다. 저장된 fold-level paired Δ variance가 있으면 §4.1 방법(α=0.05 양측, power 0.80, 군집 t, df=6)으로 interaction contrast MDE를 계산한다. 없으면 R=4 diagnostic에서 4 cell의 paired Δ variance를 추정해 계산한다. MDE를 계산할 수 없으면 상호작용은 판별 불가로 선언하고, MDE를 계산했어도 MDE보다 큰 contrast와 CI 하한 > 0이 아니면 상호작용을 주장하지 않는다. 근거: A1은 “문서에 없다”만으로 판별 불가 선언이 부족하고, §4.1의 MDE 방법을 interaction contrast에 적용할 수 있음을 지적했다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)

- **주 estimand는 baseline 대비 무조건부 Primary 7 7×50 paired Δ로 고정해야 한다.**  
  근거: `PROJECT.md` §4는 승격 계산식을 “7 과제 × 고정 50 fold에서 후보−기준선을 같은 fold에서 짝지은 Δ의 평균”으로 정의한다. §4.1도 추정 대상을 절대 macro가 아니라 대응 Δ의 기댓값으로 둔다. subsample이 일부 fold에서만 발동하더라도, Primary 7 전체 평균 개선과 승격 기준에 정합하는 estimand는 무조건부 50-fold paired Δ다.

- **발동 fold 한정 Δ는 2차 진단으로만 두어야 한다.**  
  근거: 발동 fold만 취하면 평가 집합이 발동 조건이라는 자료 구조에 의해 선택된다. `PROJECT.md` §4의 승격 estimand와 분모가 달라진다. 따라서 발동 fold Δ는 “발동했을 때의 조건부 신호”로만 보고할 수 있고, Primary 7 승격 근거로 승격할 수 없다.

- **과거 combined 실행의 산출물은 PCA/subsample 축으로 귀속할 수 없다.**  
  근거: `D-050`은 실행 스크립트가 requested 설정을 조용히 다른 설정으로 교체하는 폴백을 가졌고, 저장 예측 파일에 branch list·가중치·코드 해시가 없어 과거 산출물을 특정 구성에 귀속할 수 없다고 확인했다. 공식 기준선 `0.6227`도 `unverified-hold`다. 따라서 새 실행에서 baseline과 후보를 모두 재실행하고 requested/effective 설정 해시를 기록해야 한다.

- **단일 seed 단일 draw는 확인적 근거로 사용할 수 없다.**  
  근거: 저장 실행 로그에서 동일 구성의 seed만 바꾼 GF 계열 적합 표준편차는 `0.0021~0.0083`으로, 승격 임계 `+0.0030`보다 크다. `RU-97`에서는 단일 적합에서 `+0.0099`가 보였으나, 적합 4회 평균에서는 `+0.0003 [-0.0067, +0.0073]`로 0을 포함했고, 단일 값은 seed 42가 해당 구성 평균보다 1.5 sd 높았던 뽑기로 기록됐다. `D-048` 이후 모든 비교에 적합 분산을 포함하는 것이 의무다.

- **matched-seed 반복과 총 불확실성 보고가 필요하다.**  
  근거: `PROJECT.md` §4.1은 fold가 과제 내에서 독립이 아니라고 실측했고, 대응 Δ의 과제 군집 SE와 fold 짝짓기의 정밀성을 구분한다. `RU-97`은 fold×seed 잔차가 seed 주효과보다 크다고 실측했다. 따라서 seed 주효과, fold/task 변동, seed×fold 잔차를 함께 다루지 않으면 총 불확실성을 과소평가할 수 있다.

- **상호작용(PCA × subsample)은 사전에 확인적 주장 대상에서 제외하거나, 측정 가능한 MDE가 확인될 때까지 판별 불가로 두어야 한다.**  
  근거: 문서에는 interaction contrast의 표준편차·MDE·검정력이 없다. `PROJECT.md` §4.1의 MDE는 특정 비교의 실측이며 모든 후보에 공통이 아니다. 7 과제 군집(df=6)과 적합 분산이 존재하는 상태에서, 350 fold pair만으로 상호작용을 확인적으로 검출할 수 있다고 말할 근거가 없다.

- **combined cell은 단일 축의 확인적 귀속 증거가 될 수 없다.**  
  근거: 2×2에서 combined만 양수라는 관측은 PCA-only 또는 subsample-only 중 하나가 원인이라는 증거가 아니다. combined는 additivity/interaction 진단 또는 새 탐색 후보 신호로만 사용할 수 있다.

- **사후 재해석 구멍은 결과 보기 전에 막아야 한다.**  
  근거: `PROJECT.md` §4는 악화된 과제 전량 보고와 판정/채택 분리를 요구한다. `D-047`은 SEAL 10으로 개선 방향을 고르는 것을 금지한다. `D-050`은 provenance가 없으면 verified 비교가 불가능하다고 확인했다. §214의 선택 편향 선례도 같은 자료로 고르고 같은 자료로 성공을 판정한 구조를 경고한다.

- **전체 확인적 2×2보다 먼저 provenance·비용·분산 진단을 두어야 한다.**  
  근거: 1 fold-fit의 GPU 시간은 문서에 없다. 현재 예산은 NEXGEM B200 1장 합계 최대 4 GPU-hour이고, 단일 M 2시간 초과 시 중단 조건이 있다. `D-050` 계측 복구 전에는 어떤 실행도 verified가 아니다. 따라서 R=16 2×2(22,400 fold-fits) 같은 큰 실행을 확정하기 전에 timed run과 R=4 2×2 diagnostic(5,600 fold-fits)가 선행되어야 한다.

## 반박 (증거가 반박함)

- **“과거 combined 실행의 차이를 PCA 또는 subsample 축 효과로 귀속할 수 있다”는 주장은 반박된다.**  
  근거: 폴백으로 requested/effective 설정이 교체될 수 있었고, 과거 산출물은 `unverified-hold`다. 어떤 구성으로 만들어졌는지 귀속할 수 없으므로 축 효과로 해석할 수 없다.

- **“subsample의 주 결과를 발동 fold 한정 Δ로 삼아 Primary 7 승격을 말할 수 있다”는 주장은 반박된다.**  
  근거: `PROJECT.md` §4의 승격 estimand는 Primary 7 전체 7×50 paired Δ다. 발동 fold 한정 Δ는 다른 estimand이므로, 이를 주 결과로 승격 근거로 쓰면 기준과 불일치한다.

- **“단일 seed 단일 draw로 +0.0030 초과 여부를 확인적으로 말할 수 있다”는 주장은 반박된다.**  
  근거: 실측 적합 분산 `0.0021~0.0083`은 임계 `+0.0030`보다 크다. `RU-97`은 단일 seed 값이 구성 평균보다 1.5 sd 높은 뽑기일 수 있음을 실측했다.

- **“GF 계열의 seed sd `0.0021~0.0083`을 PCA/subsample 프로브의 paired Δ seed sd로 확정할 수 있다”는 주장은 반박된다.**  
  근거: 이 수치의 출처는 GF 계열 실행 로그의 재적합 macro 변동이다. PCA/subsample cell의 matched-seed paired Δ 분산은 문서에 없다. 다른 계열의 분산을 그대로 이식해 R을 확정하는 것은 실측 근거가 아니다.

- **“350 fold pair만 있으면 상호작용을 확인적으로 검출할 수 있다”는 주장은 반박된다.**  
  근거: fold는 과제 내에서 독립이 아니고, 독립 단위는 7 과제 군집에 가깝다. `PROJECT.md` §4.1은 fold 내 종속성과 과제 군집 효과를 실측했다. `RU-97`은 fold×seed 잔차가 fold를 더 모아도 줄지 않는다고 실측했다. interaction contrast의 분산과 MDE가 없으므로, fold 수만으로 상호작용 검출을 주장할 수 없다.

- **“과제 군집 SE와 seed-level SE 중 더 큰 값만 취하면 총 SE가 된다”는 주장은 반박된다.**  
  근거: `RU-97`은 seed 주효과가 전체의 `0~2.9%`로 작으면서도 seed sd가 기준의 최대 2.8배이고, fold×seed 잔차가 `23.9%`로 seed 주효과보다 크다고 실측했다. `PROJECT.md` §4.1도 fold/task 이질성을 별도 성분으로 기록한다. max 규칙은 seed×fold 변동과 성분 간 결합을 대표하지 못할 수 있다.

- **“point Δ ≥ +0.0030이고 총 SE < +0.0030이면 확증적 지지”라는 규칙만으로는 0을 배제하지 못한다.**  
  근거: 점추정이 임계 근처이고 SE가 임계와 비슷하면 95% CI가 0을 포함할 수 있다. `PROJECT.md` §4는 CI 하한 조건을 사용자 승인 없이 승격 필수로 덧붙이지 말라고 하지만, “확증적 지지”를 선언하려면 0 배제 또는 사전 승인된 검정 규칙이 필요하다.

- **“combined만 +0.0030을 넘으면 PCA 또는 subsample 중 하나가 원인이다”는 주장은 반박된다.**  
  근거: combined는 두 축의 합, 상호작용, 구현 순서 효과, seed/fold 변동이 섞인 cell이다. 단일 축의 paired Δ가 임계에 못 미쳤다면 combined만 좋다는 사실은 attribution 증거가 아니다.

## 판별 불가 (증거가 가르지 못함)

- **PCA 축이 Primary 7을 실제로 개선하는지 여부.**  
  문서에는 PCA 적용의 fresh valid 결과가 없다. 과거 combined 실행은 `unverified-hold`다.

- **subsample 축이 Primary 7을 실제로 개선하는지 여부.**  
  문서에는 subsample 단독의 fresh valid 결과가 없다. `CA-R1`은 reopened이지만 과제별 이득과 선택 신호 부재로 활용 불가로 기록돼 있다.

- **무조건부 Δ가 0 또는 +0.0030 미만이라는 결과가 “subsample 효과가 없다”는 의미인지 여부.**  
  subsample 발동 fold 비율은 문서에 없다. 발동 fold가 드물면 조건부 효과가 있어도 무조건부 Δ는 희석될 수 있다. 따라서 발동 fold 비율과 사전 검정력이 없으면 무조건부 음성은 반박이 아니라 판별 불가일 수 있다.

- **PCA/subsample 프로브에 필요한 반복 횟수 R.**  
  PCA/subsample의 matched-seed paired Δ seed sd, fold×seed 잔차, 총 SE는 문서에 없다. GF 계열의 `0.0021~0.0083`은 이식 가능한 실측이 아니다. 따라서 R=8 또는 R=16을 실측 근거로 확정할 수 없다.

- **상호작용 contrast의 MDE와 검출 가능 여부.**  
  interaction contrast의 variance, SE, MDE는 문서에 없다. 350 fold pair와 7 과제 군집만으로 상호작용이 검출 가능하다고 말할 수 없다.

- **GPU-hour 비용.**  
  1 fold-fit의 GPU 시간은 문서에 없다. 1 seed 4 cell = 1,400 fold-fits, R=4 2×2 = 5,600 fold-fits, R=8 2×2 = 11,200 fold-fits, R=16 2×2 = 22,400 fold-fits는 fold-fit 수로만 표현할 수 있고, 4 GPU-hour 예산 내인지 문서 근거로 확정할 수 없다.

- **PCA 사양의 유효성.**  
  PCA의 fit 범위(train-only인지 전체 fold인지), 차원, 정규화, 라벨 사용 여부, 적용 시점은 문서에 없다. fit 범위가 train+test를 포함하면 paired Δ가 과대추정될 수 있다.

- **subsample 사양의 유효성.**  
  “context slide가 일정 수 이상”의 임계값, slide 수의 정의, 발동 fold manifest, slide 선택 방식, 적용 대상(train context인지 inference context인지), combined에서 PCA와 subsample의 적용 순서는 문서에 없다.

- **새 baseline의 Primary 7 macro가 `0.6227`과 일치하는지 여부.**  
  `0.6227`은 `unverified-hold`다. 새 baseline을 재실행하기 전에는 새 baseline macro와 `0.6227`의 차이를 알 수 없다.

- **2×2 전체가 3-cell main-effect probe 또는 diagnostic보다 나은지 여부.**  
  per-fold-fit 시간과 메모리가 미실측이므로, 같은 4 GPU-hour 안에서 어떤 설계가 실행 가능한지 문서 근거로 판단할 수 없다.

- **성능 관측의 기전.**  
  PCA 또는 subsample이 성능을 바꾼다고 해도 그 기전은 문서에 없다. 성능 관측을 기전 입증으로 전환할 수 없다.

## 실행 무효 (실행·계측 결함으로 판단 불가)

- **`D-050` 계측 복구 전의 실행.**  
  `src/`가 환경변수를 읽지 않아 `ICF_*` export가 무효이고, `SH`/`SJ`를 켜는 문서화된 수단이 동작하지 않으며, 저장 예측 파일에 branch list·가중치·코드 해시가 없다. provenance writer, 폴백 제거 또는 우회, 오염 검사 재설계, 단위 테스트가 완료되기 전에는 verified 비교를 할 수 없다.

- **requested/effective 설정 해시 불일치.**  
  폴백이 requested 설정을 조용히 다른 설정으로 교체하면, 어떤 cell이 어떤 구성인지 귀속할 수 없다. requested ≠ effective이면 실행 무효다.

- **과거 `unverified-hold` 산출물을 baseline 또는 후보로 재사용하는 것.**  
  `D-050`은 과거 산출물을 특정 구성에 귀속할 수 없다고 확인했다. 재사용하면 paired Δ의 기준선이 무효다.

- **7×50×R fold·seed·과제 결과 누락.**  
  어떤 fold, seed, 과제라도 누락되면 §4의 대응 Δ 평균을 독립 재계산할 수 없다.

- **subsample 미발동 fold에서 candidate와 baseline이 동일하지 않은 경우.**  
  subsample이 발동하지 않는 fold에서는 candidate가 baseline과 동일해야 한다. 미발동 fold에서 Δ가 0이 아니고 사전 허용 오차가 없으면 구현 차이 또는 오염이다.

- **test/query slide 집합이 baseline과 candidate에서 달라지는 경우.**  
  paired Δ는 같은 fold에서 대응 비교를 전제한다. candidate가 test query slide를 제거하거나 평가 대상 slide를 바꾸면 Δ는 subsample 효과가 아니라 평가 집합 변경 효과와 섞인다.

- **PCA가 train fold 외 범위에 fit되는 경우.**  
  PCA가 train+test 전체에서 fit되면 fold 간 정보 누수가 생기고 paired Δ가 과대추정될 수 있다. PCA는 train fold의 context feature에서만 fit하고 test fold에는 transform만 적용해야 한다.

- **PCA 또는 subsample이 라벨을 사용해 fold별 구성을 선택하는 경우.**  
  라벨로 기저를 회전·선택하거나 결과 AUROC를 보고 강도를 고르면 `CA-06`, §224 Oracle 금지, 선택 편향 규범과 충돌할 수 있다.

- **발동 임계값, PCA 차원, subsample 크기, 적용 순서, fold manifest를 사후 변경하는 경우.**  
  결과 확인 후 estimand나 구성을 바꾸면 §214식 선택 편향과 동일 구조다.

- **best-seed 선택 또는 좋은 seed만 인용하는 경우.**  
  적합 분산이 임계보다 큰 상태에서 seed를 사후 선택하면 seed noise를 효과로 오인한다.

- **SEAL 10 결과를 방향 선택에 사용하는 경우.**  
  `D-047`은 SEAL 10으로 모델·후보·구조를 고르는 것을 금지한다.

- **사전 고정 R을 채우지 못해 확인적 지지/실패를 선언하는 경우.**  
  R 미충족은 실패가 아니라 판별 불가다. 확인적 판단을 내리면 판단 무효다.

- **예산 초과.**  
  단일 M 2시간 또는 합계 4 GPU-hour를 초과하면 중단 조건에 해당한다. 추가 승인 없이 초과 실행을 verified로 사용할 수 없다.

- **NaN/Inf, OOM, responsibility 합 오차 초과, 입력 누락, 코드/설정 drift.**  
  `PROJECT.md` §2의 중단 조건과 계측 무결성 요구에 저촉된다.

## 좌석 공백 (기권·실패로 빠진 관점)

- 없음.

## 다음 행동 (권한 등급과 함께)

- **T0(사용자 승인): 적합 분산을 포함한 승격/판정 절차 승인.**  
  승인 대상:  
  1. 승격 점수 기준은 `+0.0030`을 유지하되, 단일 seed 단일 draw는 확인적 근거로 쓰지 않는다.  
  2. matched-seed 반복과 총 SE/CI 보고를 필수로 한다.  
  3. 총 SE 또는 CI가 0을 배제하지 못하면 “점수 충족, 확증적 지지 불충분” 또는 “판별 불가”로 기록한다.  
  4. `PROJECT.md` §4에 없는 95% CI 하한 > 0 같은 추가 조건을 쓰려면 사용자 승인이 필요하다.  
  5. R=8 잠정 보정 또는 diagnostic-first 계획을 승인한다.  
  6. combined-only 신호를 “attribution 실패 + 탐색 신호”로 처리한다.  
  7. 4 GPU-hour 예산 초과 시 추가 승인 전 중단한다.

- **T1(협의체 확정): 사전 등록 카드 고정.**  
  고정 항목:  
  1. 주 estimand: baseline 대비 무조건부 Primary 7 7×50 paired Δ.  
  2. 2차 estimand: 발동 fold 한정 Δ, 과제별 Δ, sign agreement, 악화된 과제 전량.  
  3. cell: baseline / PCA-only / subsample-only / combined.  
  4. baseline: §3.2 공식 7-branch(`CV,BM,BD,QA,DS,SH,SJ`, `ICF_FIXED_HEAD_CT_WEIGHT=0.0`)를 새 실행으로 재실행. `0.6227`은 `unverified-hold`이므로 절대 기준선으로 쓰지 않는다.  
  5. PCA: train fold context feature에서만 fit, test fold에는 transform만, 라벨 사용 금지, 차원과 정규화 사전 고정.  
  6. subsample: 발동 임계값, slide 수 정의, slide 선택 방식, 적용 대상, combined 적용 순서 사전 고정. test/query slide 집합은 baseline과 동일하게 유지.  
  7. seed: matched seed, seed 목록 사전 고정, best-seed 금지.  
  8. R: 확인적 R을 확정하기 전에 R=4 2×2 diagnostic를 먼저 둔다. diagnostic에서 총 SE와 분산 구조를 측정한다.  
  9. 상호작용: interaction contrast variance/MDE를 사전 계산할 수 없으면 “판별 불가”로 선언한다.  
  10. 판정:  
     - 실행 무효: provenance·폴백·누락·사후 변경·SEAL 10 선택 등 결함.  
     - 판별 불가: 실행은 유효하지만 총 불확실성으로 0 또는 임계를 구분할 수 없음.  
     - 실패: 실행 유효하고 판별 가능하지만 주 estimand가 임계 미달.  
     - 지지: 실행 유효하고 주 estimand가 임계 초과이며, 승인된 불확실성 조건을 만족.  
  11. 사후 금지: estimand 전환, 발동 fold로 주 결과 교체, best-seed, 과제 하위집합 구제, 임계값 변경, combined 재해석, SEAL 10 방향 선택.

- **T2(집행): `D-050` 계측 복구 완료.**  
  대상:  
  1. `src/` 환경변수 경로 복구.  
  2. 실행 스크립트 폴백 제거 또는 requested/effective 불일치 시 실패 처리.  
  3. provenance writer: requested/effective 설정 해시, 코드 해시, 데이터 버전, fold manifest 해시, branch list, 가중치, PCA/subsample 파라미터, seed 기록.  
  4. 오염 검사 재설계: 값 앵커가 아니라 provenance 대조.  
  5. baseline 7-branch 재실행 경로 복구.  
  6. PCA/subsample 적용 코드와 matched-seed 경로 구현.  
  7. 총 SE/CI 계산 코드: task cluster, seed cluster, seed×fold 잔차를 반영하는 mixed model 또는 bootstrap. 계산 불가하면 판별 불가로 둔다.  
  8. 회귀 테스트 통과.

- **T2(집행): 성능 선택에 쓰지 않는 timed run.**  
  목적: 1 fold-fit 또는 1 task-run의 wall time, 메모리, OOM 여부를 측정한다.  
  규칙: timed run에서 나온 성능 수치는 후보 선택·R 조정·성공 판정에 사용하지 않는다.  
  중단: 4 GPU-hour 또는 단일 M 2시간을 초과하면 추가 승인 전 중단.

- **T2(집행): R=4 2×2 diagnostic.**  
  목적: seed, fold, seed×fold 분산, 총 SE, provenance 무결성, subsample 발동 fold manifest, PCA/subsample 구현 동일성을 측정한다.  
  산출: 4 cell의 matched-seed paired Δ, fold-level Δ, seed-level Δ, task-level Δ, 발동 fold 비율, requested/effective 해시, 총 SE/CI.  
  이후:  
  - 총 SE가 임계보다 작고 예산이 허용하면 T0/T1로 확인적 R을 승인·확정한다.  
  - 총 SE가 임계보다 크거나 예산이 부족하면 “이 프로브로는 판별 불가”로 종료한다.  
  - provenance 결함이 있으면 실행 무효로 종료한다.

- **T1(협의체 확정): 2×2 전체보다 작은 대안 선택.**  
  interaction을 확인적으로 주장하지 않기로 했다면 combined cell을 위해 예산을 쓰지 않는 3-cell(baseline/PCA-only/subsample-only) main-effect probe를 고려한다.  
  단, 3-cell도 per-fold-fit 시간이 미실측이므로 timed run과 예산 확인 후에만 확정한다.

## 하지 말 것 (이유와 함께)

- **과거 `unverified-hold` 산출물을 baseline으로 재사용하지 마라.**  
  이유: `D-050`은 폴백과 provenance 부재로 과거 산출물을 특정 구성에 귀속할 수 없다고 확인했다.

- **발동 fold 한정 Δ를 주 estimand로 쓰거나 결과 보고 후 주 estimand로 바꾸지 마라.**  
  이유: `PROJECT.md` §4의 승격 estimand와 불일치하고, 발동 fold 선택은 선택 편향 구멍이 된다.

- **단일 seed 단일 draw로 확인적 지지/실패를 선언하지 마라.**  
  이유: 실측 적합 분산 `0.0021~0.0083`이 임계 `+0.0030`보다 크고, `RU-97`은 단일 seed 값이 높은 뽑기일 수 있음을 실측했다.

- **best-seed를 선택하지 마라.**  
  이유: 적합 분산이 임계보다 큰 상태에서 좋은 seed만 인용하면 seed noise를 효과로 오인한다.

- **GF 계열의 seed sd를 PCA/subsample의 paired Δ sd로 확정하지 마라.**  
  이유: 출처가 다른 계열의 실행 로그이고, PCA/subsample의 matched-seed paired Δ 분산은 문서에 없다.

- **350 fold pair만으로 상호작용을 확인적으로 주장하지 마라.**  
  이유: interaction contrast의 variance/MDE가 없고, fold는 과제 내에서 독립이 아니며, 적합 분산은 fold를 늘려도 줄지 않는다.

- **combined-only 성공을 단일 축의 attribution 성공으로 바꾸지 마라.**  
  이유: combined는 두 축의 합, 상호작용, 순서 효과, seed/fold 변동이 섞인 cell이다.

- **SEAL 10 결과를 방향 선택에 사용하지 마라.**  
  이유: `D-047`은 SEAL 10으로 모델·후보·구조를 고르는 것을 금지한다.

- **`PROJECT.md` §4에 없는 CI 하한 조건을 사용자 승인 없이 승격 필수로 추가하지 마라.**  
  이유: §4는 CI 하한 `> +0.3%p` 또는 `> 0`을 사용자 승인 없이 승격 필수 조건으로 덧붙이지 말라고 명시한다.

- **PCA/subsample 파라미터, 발동 임계값, fold manifest, 적용 순서를 사후 변경하지 마라.**  
  이유: 결과 확인 후 변경은 §214식 선택 편향과 동일 구조다.

- **test/query slide 집합을 candidate에서 바꾸지 마라.**  
  이유: paired Δ는 같은 fold에서 대응 비교를 전제한다. 평가 집합이 다르면 Δ가 subsample 효과가 아니다.

- **PCA를 train fold 외 범위에 fit하지 마라.**  
  이유: fold 간 정보 누수로 paired Δ가 과대추정될 수 있다.

- **fold 수를 늘리거나 재분할하지 마라.**  
  이유: `PROJECT.md` §3.1은 fold를 PathoBench 공식 분할 그대로 고정하고 fold 수 증대나 재분할을 금지한다.

- **성능 관측을 기전으로 바꾸어 쓰지 마라.**  
  이유: 성능 변화가 확인되더라도 기전은 문서에 없다. 불명확하면 “기전 미확인”으로 둔다.

- **좌석 동의 수를 근거로 쓰지 마라.**  
  이유: 표결은 증거가 아니다. 근거는 문서 실측, 사전 고정 규칙, 계측 무결성, 통계적 범위에서만 나온다.
