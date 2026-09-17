# ICF Council — C-20260918-7

- **연구 질문**: 직전 회차(C-20260918-6)가 확정한 사전 등록안을 공격한다. 그 설계는 공식 7-branch score에 더하는 73파라미터 residual head를 seed 1~8로 재학습하고, 적합 불확실성을 s_fit^2/8 항으로 다루며 총 SE를 sqrt(SE_taskcluster^2 + s_fit^2/8)로 둔다.

그런데 새 실측이 그 가정을 위협한다. RU-97 §11 분산분해에서 재적합의 영향은 seed 주효과가 아니라 fold×seed 잔차로 나타났다: m8_raw 13.2%, m16_raw 23.0%, m32_raw 23.9%, whiten_d256_m8 6.7%. seed 주효과는 0~2.9%에 불과했다. 즉 재적합은 모든 fold를 같이 밀어 올리는 것이 아니라 fold 순서를 뒤섞는다.

질문: (1) s_fit^2/8 형태의 항이 fold×seed 상호작용이 지배적인 상황에서 타당한가? 타당하지 않다면 총 SE를 어떻게 다시 써야 하는가? (2) seed 평균 Δ를 estimand로 쓰는 것은 이 상호작용 아래에서 무엇을 추정하는 것이 되는가? (3) 8회 재적합이 충분한지 판단할 근거가 있는가? (4) 이 분산 구조가 residual head(73파라미터, zero-init)에도 나타날 것이라고 가정할 근거가 있는가, 아니면 GMM 적합 고유의 성질인가? (5) 사전 등록안을 고쳐야 한다면 무엇을 어떻게 고치는가 - 결과를 보기 전에 고칠 수 있는 형태로 써라.

문서에 없는 수치를 지어내지 말라. RU-97의 분산분해는 GMM 적합에 대한 것이며 학습 head에 대한 실측은 아직 없다는 점을 혼동하지 말라.
- **RU 유형**: 진단
- **최대 권한 등급**: T1
- **소집자**: Claude Code (claude-opus-5)
- **좌석**: R1(R, temp 0.5, http://127.0.0.1:8000/v1/chat/completions), T1(T, temp 0.4, http://127.0.0.1:8001/v1/chat/completions), A1(A, temp 0.2, http://127.0.0.1:8003/v1/chat/completions), P1(P, temp 0.7, http://127.0.0.1:8000/v1/chat/completions), S1(S, temp 0.2, http://127.0.0.1:8001/v1/chat/completions)
- **호출**: 9/14 · **소요**: 984.5초

> 이 문서는 로컬 모델이 생성한 것이며 **검증되지 않았습니다.** 좌석이 인용한 수치는
> 정본 문서와 대조하기 전에는 결론에 넣지 마십시오. 좌석 수와 동의 수는 증거가 아닙니다.

## 기권한 좌석

- [Phase 2] `R1` (반증): 본문이 비어 있다 (finish_reason=length); 추론 토큰이 예산을 소진했을 가능성이 높다

## 편성 근거 (결과 전 기록)

직전 회차의 산출물을 공격하는 회차이므로 반증(R)을 첫 좌석으로 두고 제안(P)의 온도를 낮췄다(0.7) - 이번에 필요한 것은 새 아이디어가 아니라 기존 설계의 수리다. 이론(T)은 분산 성분 구조와 estimand 정의의 관계를 본다. 감사(A)는 RU-97의 GMM 분산분해를 학습 head에 그대로 전이하는 추론의 결함을 점검한다 - 이 회차에서 가장 새기 쉬운 지점이다. 일반화(G)와 실측(M)은 제외: 질문이 통계 절차의 내적 정합성이고, GPU 4는 공식 baseline 재생성에 쓰이고 있다. 같은 협의체가 만든 설계를 같은 협의체가 공격하므로 자기 방어 편향이 있을 수 있고, 그 한계를 결론에 적는다.

---

## Phase 1 — 독립

### P1 (제안)

## 제안 1: fold×seed 잔차를 SE에 포함하는 mixed-model SE로 교체
- **바꾸는 것**:  
  `C-20260918-6`의 총 SE 산출식을 `sqrt(SE_taskcluster^2 + s_fit^2/8)`에서 폐기하고, seed 1~8의 fold별 paired difference 전체를 쓰는 model-based SE로 바꾼다.  
  구체적으로 `t = 1..7`(Primary 7 과제), `f = 1..50`(fold), `s = 1..8`(seed)에 대해  
  `Δ_{t,f,s} = AUC_candidate(t, f, s) - AUC_baseline(t, f)`  
  를 계산한다. baseline은 공식 7-branch의 결정론적 per-fold AUROC다. primary 점추정은  
  `Δ_macro = (1/7)Σ_t (1/50)Σ_f (1/8)Σ_s Δ_{t,f,s}`  
  로 유지한다. 총 SE는 REML linear mixed model  
  `Δ_{t,f,s} = μ + τ_t + u_{t,f} + v_s + w_{t,f,s} + e_{t,f,s}`  
  에서 `μ`의 SE로 정의한다. 여기서 `τ_t`는 task 효과, `u_{t,f}`는 task 내 fold 효과, `v_s`는 seed 주효과, `w_{t,f,s}`는 fold×seed 잔차, `e_{t,f,s}`는 나머지 residual이다. df는 Satterthwaite 또는 Kenward-Roger로 보고한다. `s_fit^2/8` 형태의 별도 항은 넣지 않는다. 모델이 수렴하지 않거나 fold×seed 분산성분을 추정할 수 없으면, 해당 후보의 승격 판정을 `실행 무효`로 처리하고 기존 quadrature로 되돌리지 않는다. 또한 seed 주효과, fold×seed, residual의 분산성분과 상대 비중을 반드시 보고한다.
- **바꾸는 전제**:  
  현재 사전 등록안은 적합 불확실성이 fold와 무관한 공통 이동, 즉 seed 주효과로 요약되고, 그 seed 평균의 SE를 `s_fit/√8`으로 잡아 taskcluster SE와 독립적으로 quadrature할 수 있다고 둔다. RU-97 §11의 GMM 실측에서는 재적합 효과가 seed 주효과(`0~2.9%`)보다 fold×seed 잔차(`m8_raw 13.2%`, `m16_raw 23.0%`, `m32_raw 23.9%`, `whiten_d256_m8 6.7%`)로 나타났다. 이 수치는 GMM 적합에 대한 것이므로 73파라미터 residual head에 대한 증거는 아니다. 그러나 결과 전에는 head에서도 fold×seed가 나타날 가능성을 배제할 근거가 없으므로, SE는 seed 주효과만 전제하는 `s_fit^2/8`이 아니라 fold×seed를 명시적으로 포함하는 구조로 써야 한다.
- **기대하는 차이**:  
  `Δ_macro` 점추정 자체는 변하지 않는다. 변하는 것은 총 SE의 산출 근거다. head에서 fold×seed 잔차가 실제로 크면, seed 주효과만 반영한 `s_fit^2/8`은 fold별 재적합 변동을 덜 반영해 총 SE를 과소평가할 수 있다. 반대로 head 변동이 seed 주효과에 가까우면 mixed-model SE는 현재 방식과 비슷하거나 더 보수적일 수 있다. head fit variance 실측이 없으므로 SE 변화량, MDE 변화, 승격 판정 변화는 정량 예측 불가다.
- **기전**:  
  seed마다 residual head의 최적화 경로가 달라 fold별 candidate score가 서로 다른 방향으로 흔들리면, seed 평균 fold Δ에도 fold×seed 잔차가 남는다. mixed model은 이 잔차를 `w_{t,f,s}`로 분리해 `μ`의 SE에 반영한다. 다만 73파라미터 zero-init residual head에서 fold×seed가 실제로 지배적인지는 기전 미확인이며, RU-97 GMM 실측만으로는 head의 기전을 입증하지 못한다.
- **실패 조건**:  
  가장 그럴듯한 실패는 73파라미터 zero-init head의 seed 변동이 GMM처럼 fold×seed가 아니라 seed 주효과에 가깝거나 거의 결정적이라, mixed model이 불필요한 보수성을 만들거나 분산성분을 과적합하는 경우다. 7 task, 50 fold, 8 seed에서 fold×seed 분산성분이 추정 불가능하면 이 제안은 `실행 무효`로 전환해야 하며 `판별 불가`가 아니다. baseline 7-branch가 더 이상 seed 무관 결정론이 아니라 seed에 의존하는 구성으로 바뀌면 paired Δ 정의 자체가 무효다.

## 제안 2: seed 평균 Δ의 estimand을 ‘seed 기대 fold-averaged paired Δ’로 고정하고 fold-level 해석을 제한
- **바꾸는 것**:  
  사전 등록안의 estimand를 “seed 1~8 평균 Δ”라는 모호한 표현에서 명시적 수식으로 바꾼다.  
  `μ = (1/8)Σ_s (1/7)Σ_t (1/50)Σ_f Δ_{t,f,s}`  
  로 고정하고, 이것이 “training seed를 새로 뽑았을 때 fold-averaged paired Δ의 기대”의 8-seed 경험 근사임을 선언한다. 이는 “각 fold에서 일관된 개선”이 아니라 “seed 평균과 fold 평균을 모두 취한 overall mean”이다.  
  필수 보고 항목을 추가한다:  
  1. seed별 `Δ_macro` 8개,  
  2. task별 seed-averaged `Δ` 7개,  
  3. head-specific fold×seed 분산성분,  
  4. seed-averaged fold Δ가 음인 fold 수와 해당 과제 수,  
  5. “본 판정은 fold별·seed별 일관성을 주장하지 않는다”는 문구.  
  head-specific ANOVA 또는 mixed model에서 fold×seed 잔차가 seed 주효과보다 크다고 나오면, 보고서에 `mean estimand only; fold-level consistency 판별 불가`를 강제한다. 이 규칙은 결과를 본 뒤 estimand를 바꾸는 것이 아니라, 결과 전에 어떤 문구로 해석을 제한할지를 고정하는 것이다.
- **바꾸는 전제**:  
  현재 문구가 seed 평균을 마치 deterministic fold-level improvement에서 잡음만 제거한 값처럼 읽히게 둔다. fold×seed 상호작용이 있으면 seed 평균은 fold별 패턴을 상쇄한 mean일 뿐이다. estimand를 명시하지 않으면, positive `Δ_macro`를 fold 전체 개선 또는 seed 전체 개선으로 해석할 위험이 있다. RU-97 GMM에서 fold×seed 잔차가 seed 주효과보다 컸다는 점은 head에도 같은 해석 제한이 필요할 수 있음을 보여주는 동기일 뿐, head 실측은 없다.
- **기대하는 차이**:  
  성능 점수나 승격 점수 자체를 바꾸지 않는다. 같은 `Δ_macro`에서 “지지”가 fold-level 개선 주장으로 확대되는 것을 막고, 상호작용이 크면 fold-level claim을 `판별 불가`로 분리한다. 정량 예측 불가다.
- **기전**:  
  수학적으로 `Δ_{t,f,s}`에 fold×seed 성분이 있으면, seed 평균은 fold별 expected difference를 만들고, fold 평균은 overall mean을 만든다. 따라서 seed 평균 Δ는 fold별 일관된 효과가 아니라 seed 분포와 fold 평균에 대한 mean을 추정한다. residual head가 왜 fold×seed를 만드는지는 기전 미확인이다.
- **실패 조건**:  
  프로젝트의 decision target이 오직 overall mean뿐이고 fold-level 해석이 실제로 혼동되지 않는다면, 추가 보고는 과잉이다. 또는 fold×seed가 없고 seed main만 있으면, 현재 seed 평균 Δ가 이미 안정적 mean을 추정하므로 이 제안은 불필요한 복잡성이다. head 실측 전에는 어느 쪽인지 판별 불가다.

## 제안 3: 8회 재적합의 충분성과 GMM 전이를 결과 전 규칙으로 분리
- **바꾸는 것**:  
  사전 등록안에서 `seed 1~8`을 “적합 불확실성을 충분히 잡는 replicate”로 암시하는 표현을 제거한다. 대신 아래 규칙을 결과 전에 고정한다.  
  1. `seed 1~8`은 고정 실행 규모일 뿐, 충분성 증명이 아니다.  
  2. RU-97 §11의 GMM 분산분해 수치는 head SE 산출에 직접 쓰지 않는다. `m8_raw 13.2%`, `m16_raw 23.0%`, `m32_raw 23.9%`, `whiten_d256_m8 6.7%`는 GMM 적합 실측이며, 73파라미터 residual head의 기대 분산이 아니다.  
  3. 8회 완료 후 head-specific `Δ_{t,f,s}`에서 fold×seed 분산성분과 총 SE를 산출한다.  
  4. 분산성분과 총 SE가 유한하게 산출되면 §4의 점수 기준 `Δ_macro ≥ +0.003`으로 판정 진행한다.  
  5. 분산성분이나 총 SE를 산출할 수 없으면 해당 후보의 승격 판정을 `실행 무효`로 종료한다.  
  6. 추가 seed는 같은 후보의 Primary 7 결과를 본 뒤가 아니라, 새 사전 등록과 함께 결과 전에만 허용한다.  
  7. 문서에 8회 충분성을 뒷받침하는 power 분석, target SE, head fit variance 실측이 없으므로, “8회면 충분하다”는 결론을 내려서는 안 된다.
- **바꾸는 전제**:  
  현재 설계가 8회이면 적합 불확실성을 충분히 평균할 수 있고, RU-97 GMM에서 본 fold×seed 구조가 73파라미터 residual head에도 유사하게 나타날 것이라고 전제할 위험을 제거한다. RU-97의 분산분해는 GMM 적합에 대한 것이며, 학습 head에 대한 실측은 아직 없다. 따라서 GMM의 fold×seed 비중을 head의 SE 설계 근거로 쓰는 것은 결과 전에도, 결과 후에도 타당하지 않다. 8회 충분성도 문서에 근거가 없으므로 결과 전 판정 불가다.
- **기대하는 차이**:  
  GMM 수치를 head 설계 근거로 오용하는 것을 막고, 8회 실행을 fit uncertainty measurement로 기능하게 한다. head 실측 전에는 SE 크기나 충분성 달성 여부는 정량 예측 불가다. head seed 변동이 매우 작으면 8회로 충분하고, fold×seed가 크면 8회로도 SE가 커져 fold-level 또는 mean claim의 판별 가능성이 낮아질 수 있다.
- **기전**:  
  GMM EM/초기화와 73파라미터 zero-init residual head의 학습 변동은 stochastic source가 다를 수 있다. zero-init은 initialization variance를 줄이고 optimizer, 데이터 순서, regularization, batch 구성 등 다른 변동만 남길 수 있다. head에서 fold×seed가 나타나는지, 나타날 경우 GMM과 같은 비중인지 기전 미확인이다.
- **실패 조건**:  
  가장 그럴듯한 실패는 RU-97 GMM이 head와 동일한 pipeline, fold, score aggregation, stochastic fit 구조를 공유해 사실상 informative prior가 되므로, 이를 완전히 배제하는 것이 보수적 과잉이 되는 경우다. 또는 8회 이후 head-specific 분산성분이 추정 가능하고 SE가 승격 기준보다 훨씬 작아 충분성이 실제로 충족됐는데, “8회 증명이 아니다”라는 형식 때문에 추가 실험을 요구하게 된다면 제안은 비효율적이다. 추가 seed를 결과 전에만 허용하는 규칙이 예산을 늘릴 수 있다.

### R1 (반증)

## 반론 1
- **공격 대상**: (1) `총 SE = sqrt(SE_taskcluster^2 + s_fit^2/8)`가 fold×seed 상호작용이 지배적인 재적합 불확실성을 적절히 표현한다는 C-20260918-6 설계.
- **더 단순한 설명**: RU-97 §11의 관측(seed 주효과 0~2.9%, fold×seed 잔차 m8_raw 13.2%·m16_raw 23.0%·m32_raw 23.9%·whiten_d256_m8 6.7%)은 “seed가 모든 fold에 공통 offset을 준다”는 설명보다 “seed별 적합이 fold마다 다른 Δ를 만든다”는 설명으로 더 직접적으로 읽힌다. 이 경우 `s_fit^2/8`은 seed macro 평균의 변동만 줄 수 있고, fold-level Δ의 불확실성을 대표하지 않는다.
- **무엇이 참이면 이 주장이 무너지는가**: 주어진 문서에는 `s_fit`의 산출 정의가 문서에 없다. 만약 `s_fit`이 seed macro Δ의 sd라면, `s_fit^2/8`은 8개 seed macro 평균의 변동에 해당하므로 fold 변동의 일부가 이미 포함될 수 있고, 단일 적합 `SE_taskcluster`를 더하는 것은 중복 또는 누락이 통제되지 않은 사각형 합이다. 반대로 `s_fit`이 fold-level residual sd라면, `s_fit^2/8`은 macro mean의 variance가 아니라 fold-level 변동의 일부일 뿐이다. fold×seed 잔차가 무시할 수 없고 `s_fit`이 mixed model의 residual seed variance로 정의되지 않았다면, 총 SE는 `sqrt(SE_taskcluster^2 + s_fit^2/8)`이 아니라 전체 `7×50×8` 배열에서 estimand의 SE로 다시 써야 한다. 구체적으로 `Δ_{t,f,r}`에 대해 `Δ_{t,f,r} = μ + τ_t + u_{t,f} + a_r + (ar)_{t,f,r} + ε_{t,f,r}` 형태의 mixed model 또는 task/seed cluster bootstrap으로 `μ = (1/7)Σ_t (1/50)Σ_f (1/8)Σ_r Δ_{t,f,r}`의 SE를 추정해야 한다. `s_fit`을 fold/task 효과를 제거한 residual seed variance로 정의하고 독립성을 사전에 인정하는 경우에만 사각형 합을 근사로 유지할 수 있다.
- **이 반론의 한계**: head의 seed 효과가 공통 shift이고, `s_fit`이 seed-level estimand의 sd로 정확히 정의되며, 8 seed가 독립 replicate라면 `s_fit^2/8`은 seed 평균의 변동으로 타당하다. 또한 promotion statistic이 macro mean만 사용하고 fold×seed interaction의 macro 평균이 0이라면 fold-level 불안정성이 macro SE를 자동으로 무효화하지는 않는다. 이 반론은 공식의 반박이지 head 성능 실패의 증거는 아니다.

## 반론 2
- **공격 대상**: (2) seed 평균 Δ를 estimand로 쓰는 것.
- **더 단순한 설명**: fold×seed interaction이 있으면 seed 평균 Δ가 0에 가까워도 많은 fold에서 큰 양/음 Δ가 상쇄된 것일 수 있다. 즉 “효과가 없다”는 것보다 “fold-specific 효과가 상쇄됐다”가 더 단순한 설명이다.
- **무엇이 참이면 이 주장이 무너지는가**: seed 평균 Δ는 `E_s[E_{t,f}Δ]`, 또는 8개 고정 seed의 평균 Δ를 추정한다. fold별 공통 개선량, 특정 fold에서의 개선, task-level stable improvement를 추정하지 않는다. 8 seed가 seed 분포를 대표하지 못하면 estimand는 seed set에 의존한다. 따라서 “공식 7-branch score에 head를 더하면 fold 분포에서 개선된다”는 해석은 seed 평균 Δ만으로 지지되지 않는다. fold×seed interaction이 지배적일 때 seed 평균 Δ는 marginal mean이지 fold-common effect가 아니다.
- **이 반론의 한계**: PROJECT.md §4.1의 estimand가 Δ의 기댓값이고, stochastic head의 경우 random seed에 대한 기대를 명시한다면 seed 평균 Δ는 합법적인 marginal estimand다. 다만 fold/task heterogeneity를 함께 보고해야 한다.

## 반론 3
- **공격 대상**: (3) 8회 재적합이 충분하다는 암묵적 전제.
- **더 단순한 설명**: RU-97의 fold×seed 잔차 관측은 4 seed GMM 검사의 유한표본 변동이거나 소수 seed/fold outlier에 의한 것일 수 있다. 같은 관측을 “8 seed로 수렴한다”는 설명보다 “seed 수에 따른 안정성이 아직 측정되지 않았다”는 설명이 더 단순하다.
- **무엇이 참이면 이 주장이 무너지는가**: head-specific 충분성 기준이 문서에 없으면 8회 충분성은 근거 없다. 8회는 “최소 반복”으로만 쓰고, seed-level Δ_s, `s_fit`, `s_fit/√8`, leave-one-seed-out 변화, fold×seed variance component를 의무 보고해야 한다. 이 진단이 불안정하면 판정 상태를 판별 불가 또는 실행 무효로 처리하는 규칙을 결과 전에 고정해야 한다. 현재 문서에는 8회 충분성 근거가 문서에 없다.
- **이 반론의 한계**: head-specific pilot 또는 사전 power/stability 기준이 8회로 SE와 variance component가 안정됨을 보여준다면 충분성을 주장할 수 있다. 그러나 그 기준은 Primary 7 성능 결과를 선택에 쓰지 않는 범위에서만 허용된다.

## 반론 4
- **공격 대상**: (4) RU-97 GMM의 fold×seed 분산 구조가 73파라미터 zero-init residual head에도 나타난다고 가정하는 것.
- **더 단순한 설명**: RU-97의 잔차는 GMM 적합 고유의 초기화 민감성일 수 있다. RU-97에서는 seed에 따라 유효 성분이 흔들린다는 관측이 있다(whiten_d256_m8 seed별 2.83~5.74, 비화이트닝 구성은 1.0대 붕괴). head는 라벨로 학습하는 73파라미터 zero-init regression head일 수 있고, seed 효과가 작거나 공통 shift일 수 있다. 같은 GMM 관측이 head의 분산 구조를 예측하는 것은 아니다.
- **무엇이 참이면 이 주장이 무너지는가**: head의 seed 1~8 실측에서 fold×seed 잔차가 GMM과 유사하게 크다면 가정은 지지된다. 그렇지 않다면 GMM 구조를 이식하는 것은 근거 없다. 현재 residual head의 seed 분산 실측은 문서에 없다. 기전은 미확인이다.
- **이 반론의 한계**: head 학습도 non-convex하고 seed에 따라 다른 local optimum에 수렴하며, 그 차이가 fold별로 다른 AUROC 변화를 만든다면 유사한 fold×seed 구조가 나타날 수 있다. 그러나 그것은 추측이 아니라 head 실측으로 확인해야 한다.

## 반론 5
- **공격 대상**: C-20260918-6 사전 등록안을 현재 형태로 유지할 수 있다는 것.
- **더 단순한 설명**: 새 실측이 위협하는 것은 학습 head 자체의 실패가 아니라, estimand와 SE 공식이 fold×seed 구조를 반영하지 못하는 설계 불일치일 수 있다. 따라서 후보를 폐기할 필요 없이 estimand·SE·보고 항목을 고치는 것이 더 단순한 해결이다.
- **무엇이 참이면 이 주장이 무너지는가**: 결과를 보기 전에 다음을 고쳐야 한다. (a) estimand를 seed-averaged fold Δ의 marginal mean으로 명시. (b) 총 SE를 `sqrt(SE_taskcluster^2 + s_fit^2/8)`가 아니라 `7×50×8` 배열의 mixed model/bootstrap SE로 대체. (c) head-specific 분산분해(fold, seed, fold×seed)와 seed-level Δ_s, `s_fit`, `s_fit/√8`, leave-one-seed-out을 의무 보고. (d) 8회는 충분성이 아니라 최소 반복으로 취급하고, 불안정 시 판별 불가/실행 무효 규칙을 사전 고정. (e) best-seed 선택 금지 유지. (f) fold×seed 잔차가 지배적일 때 seed 평균 Δ를 fold-common improvement로 해석하지 않도록 보고 문구를 고정. 이 수정이 없으면 원 SE로 내린 승격 판정은 불확실성 측정이 estimand와 일치하지 않으므로 실행 무효다.
- **이 반론의 한계**: head 실측에서 seed 효과가 seed 주효과로만 나타나고 `s_fit`이 residual seed variance로 정확히 정의된다면, 원 공식은 근사적으로 유지될 수 있다. 이 반론은 후보 기각이 아니라 설계 유효성 요건이다.

### T1 (이론)

## 이론적 병목 1
- **가정**: `s_fit^2/8`과 `sqrt(SE_taskcluster^2 + s_fit^2/8)`은 재적합/재학습 변동이 fold·task와 무관한 공통 seed 주효과이고, seed 간 독립이며, task-cluster 불확실성과 독립적으로 가산되고, R=8 seed 평균으로 `1/R`만 나누면 실제 fit uncertainty가 된다고 두고 있다.
- **이 가정이 깨지는 조건**: fold×seed interaction이 0이 아니고, seed 효과가 fold별 paired Δ의 부호·크기를 바꾸면 `s_fit^2/8`은 seed 주효과만 다루고 interaction 잔차를 놓친다. 이 경우 총 SE는 per-fold·per-seed paired Δ `Δ_{t,f,s}`에서 mixed model/resampling으로 `Var(\barΔ)`를 직접 추정해야 한다. 분산성분으로 쓸 수 있는 형태는 `SE_total^2 = SE_taskcluster^2 + σ^2_seed/R + σ^2_{fold×seed}/(R·m_eff)`이며, `m_eff`는 fold의 과제 내 독립성·군집을 반영한 effective fold 수다. 가산이 성립하지 않으면 mixed model의 총 분산을 쓴다.
- **깨질 때 나타나는 관측 가능한 증상**: seed 주효과가 작고 fold×seed residual이 크다. 같은 fold의 Δ 부호가 seed마다 달라진다. seed 평균은 0 근처인데 per-fold/per-task 구간이 seed마다 반대 방향이다. 단일 seed에서 우위로 보인 차이가 seed 평균에서 사라진다. `s_fit^2/8` 기반 SE가 per-fold·per-seed 변동보다 작고, fold를 더 모아도 fit uncertainty가 감소하지 않는다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: RU-97 §11에서 `m8_raw` fold×seed 13.2%, `m16_raw` 23.0%, `m32_raw` 23.9%, `whiten_d256_m8` 6.7%이고 seed 주효과는 0~2.9%다. current_status도 재적합이 fold 순서를 뒤섞고 fold를 더 모아도 줄지 않으며 seed 평균만 보고해도 잡히지 않는다고 기록한다. RU-97 §8/§9/§10의 단일 적합 우위가 seed 인공물이었다는 기록도 충돌한다.

## 이론적 병목 2
- **가정**: `seed 평균 Δ`를 estimand로 쓰는 것은 fold-invariant한 방법 효과를 추정하고, 8회 재적합이면 fit noise가 충분히 평균된다고 두고 있다.
- **이 가정이 깨지는 조건**: fold×seed interaction이 존재하면 `seed 평균 Δ`는 fold-invariant main effect가 아니라 고정 fold 집합에서 seed 분포를 평균한 양, 즉 `(1/F)Σ_f E_s[Δ_{f,s}]`를 추정한다. interaction의 fold별 평균이 0이 아니거나 fold 난이도와 상관 있으면, 이는 intended method effect와 다른 혼합 추정량이 된다. 8회 충분성은 문서에 근거가 없다. RU-97은 4 seed 실측만 있고, R에 따른 interaction variance 수렴 곡선이 없다. 결과를 보기 전에 고칠 수 있는 형태: (i) estimand를 "seed 1~8의 paired fold-average Δ 평균"으로 명시하고 best-seed/단일 적합 금지, (ii) `s_fit^2/8`을 기본 SE로 쓰지 말고 per-fold·per-seed Δ로 fold/seed/fold×seed 분산성분을 추정, (iii) 8회를 "충분"이 아니라 "측정 횟수"로만 서술하고 interaction variance의 CI를 보고, (iv) interaction variance를 신뢰하게 추정하지 못하면 "판별 불가", (v) `+0.0030` 기준은 유지하되 총 SE/CI는 interaction을 포함.
- **깨질 때 나타나는 관측 가능한 증상**: seed 평균 Δ가 작아도 per-fold Δ의 seed 간 변동이 크다. sign agreement가 seed마다 흔들린다. 8 seed의 seed 평균 Δ CI가 넓고, interaction variance CI가 0을 포함한다. fold 수 증가나 seed 평균으로 fit uncertainty가 줄지 않는다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: RU-97 §11의 fold×seed 잔차와 "seed 평균만 보고해도 잡히지 않는다"는 관측이 충돌한다. 8회 충분성에 대한 문서 근거는 "문서에 없음"이다.

## 이론적 병목 3
- **가정**: RU-97 GMM 적합에서 나온 fold×seed 분산 구조가 73파라미터 zero-init residual head의 재학습에도 동일하게 나타난다고 두고 있다.
- **이 가정이 깨지는 조건**: head 학습은 GMM 적합과 다른 확률적 과정(초기화, optimizer, data order, batch, zero-init 이후의 gradient update)을 가지므로, GMM의 분산분해를 head에 이식할 근거가 없는 경우. 이 경우 head의 fit uncertainty는 GMM의 `s_fit`이나 fold×seed 비율로 추정할 수 없고, head의 seed 1~8 per-fold·per-seed Δ로 별도 분산분해를 수행해야 한다. GMM 고유 성질인지 head 공통 성질인지 기전 미확인이다.
- **깨질 때 나타나는 관측 가능한 증상**: head의 seed 주효과가 GMM과 달리 크거나, fold×seed residual이 GMM과 다른 비율로 나타난다. zero-init이 초기 seed 차이를 줄여도 training 중 seed 경로가 갈라진다. GMM에서 interaction이 작았던 구성이 head에서 interaction이 커진다.
- **문서의 어떤 관측이 이 가정과 충돌하는가**: RU-97의 분산분해는 GMM 적합에 대한 것이며, 73파라미터 residual head에 대한 실측은 문서에 없다. 따라서 전이 가정을 지지하거나 반박하는 head 관측은 "문서에 없음"이다.

### A1 (감사)

## 점검 1: `s_fit^2/8` 항이 fold×seed 상호작용을 다루는가
- **확인하려는 것**: `s_fit^2/8`이 유효하려면 (a) 적합 불확실성이 fold와 무관한 seed 주효과, (b) `s_fit`이 seed 평균 Δ의 표준편차 또는 mixed-model에서 estimand에 해당하는 분산 성분, (c) `SE_taskcluster^2`와 독립·중복 없음이 필요하다. RU-97 §11의 fold×seed 잔차(`m8_raw` 13.2%, `m16_raw` 23.0%, `m32_raw` 23.9%, `whiten_d256_m8` 6.7%)와 seed 주효과 0~2.9%가 이 가정을 지지하는지, 그리고 `s_fit`의 정의가 카드에 고정됐는지 확인한다.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: seed macro sd는 작지만 fold별 seed-averaged Δ의 변동은 큼; `SE_taskcluster^2 + s_fit^2/8`이 fold×seed 잔차를 놓치거나 single-fit SE와 이중 계산; seed마다 fold 순위가 뒤집히는데 총 SE는 좁음; `+0.0030` 경계에서 위양성; variance component 미보고; GMM 실측을 head에 무비판적으로 전이.
- **확인 방법**: `talks/council/C-20260918-6_card.json`에서 `s_fit` 정의·SE 공식 확인; `talks/reports/2026-09-17_ru97_gf_primary7.md` §11 대조; `*_per_fold.pt`로 `d_{t,f,s}` 재구성; `\bar d_{t,f}=(1/8)Σ_s d_{t,f,s}`에 task-cluster bootstrap; mixed model `d_{t,f,s} ~ task + fold(task) + seed + fold:seed`로 `σ²_seed`, `σ²_fold×seed` 추정; `s_fit^2/8`과 `\bar d_{t,f}`의 task-cluster SE 비교. 개정 방향: seed-averaged fold Δ의 task-cluster SE를 총 SE로 쓰거나, mixed-model 성분으로 `σ²_seed/(7·50·8)+σ²_fold×seed/(7·50·8)`를 fold main SE와 결합하되 중복 금지.

## 점검 2: seed 평균 Δ의 estimand
- **확인하려는 것**: seed 평균 Δ가 fixed-seed mean인지 random-seed expectation인지, fold×seed 하에서 무엇을 추정하는지. `docs/PROJECT.md` §4.1의 estimand는 Primary 7 고정 fold 분포에서 후보−기준선 대응 Δ의 기댓값이며 표본 단위는 fold다. seed 평균 Δ가 이 fold estimand를 확장하는지, 아니면 seed 집합의 평균 성능을 추정하는지 사전 문구가 있는지 확인한다.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: seed 평균을 seed-independent improvement로 해석; fold×seed cancellation으로 평균은 작고 fold별 부호는 반대; fixed seeds 1~8의 평균과 random-seed expectation 혼동; best-seed 선택; fold-specific 악화가 평균에 숨김; “모델이 개선됐다”는 주장이 estimand보다 넓어짐.
- **확인 방법**: `talks/council/C-20260918-6_card.json`의 estimand 문구 확인; `docs/PROJECT.md` §4.1 대조; per-seed fold Δ 프로파일 계산; `E_seed(E_fold Δ)`와 `E_fold(E_seed Δ)`를 별도로 보고; fixed seeds 1~8로 명시하거나 random-seed claim 시 seed 분포·분산·판정 규칙을 사전 고정; best-seed 선택 금지 확인.

## 점검 3: 8회 재적합 충분성
- **확인하려는 것**: 8회 재적합이 충분한지 판단할 근거가 있는지. fixed-seed estimand라면 8은 estimand의 정의이지 충분성 결론이 아니다. random-seed estimand라면 seed 1~8은 seed 모집단에서 뽑은 표본이므로, mean과 fit variance/fold×seed 구성을 추정하는 데 필요한 seed 수·정밀도 기준·중단 규칙이 사전에 있어야 한다.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 8회를 충분하다고 주장하지만 variance component CI 없음; 추가 seed 시 point estimate 또는 SE가 materially change; fold×seed residual 추정이 불안정; seed set에 따른 결정 변동; 결과를 본 뒤 seed 추가·삭제; `R=8`을 잠정 보정이 아니라 확정 근거로 사용.
- **확인 방법**: `talks/council/C-20260918-6_card.json`에서 충분성 기준 확인; `docs/current_status.md`의 “`R=8`은 잠정 보정” 대조; RU-97의 4 seed GMM 실측은 head에 적용 불가임을 확인; head fit variance 실측이 없는지 확인; random-seed claim이면 seed 수·target SE·variance precision 기준을 사전 고정; fixed-seed claim이면 8을 estimand 구성으로만 기록; leave-one-seed-out sensitivity를 사전 등록하되 결과 후 seed 집합 변경 금지.

## 점검 4: GMM fold×seed 구조를 residual head에 전이하는 근거
- **확인하려는 것**: RU-97 §11의 분산분해는 GMM 적합에 대한 것이다. 73파라미터 zero-init residual head에도 같은 fold×seed 지배 구조가 나타날 것이라고 가정할 근거가 있는지 확인한다. head의 stochastic source는 optimizer, minibatch order, initialization, dropout, nondeterministic ops 등 GMM의 responsibility collapse와 다를 수 있다.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: GMM의 fold×seed 잔차를 head의 실측인 것처럼 인용; head가 deterministic 또는 GMM과 동일하게 불안정하다고 단정; `s_fit`을 GMM seed sd로 대체; head-specific fit variance 미측정; 기전 미확인을 기전 확립으로 서술.
- **확인 방법**: `docs/` 전체에서 residual head fit variance 실측 검색; `talks/reports/2026-09-17_ru97_gf_primary7.md`가 GMM 전용임을 확인; head training code에서 seed가 제어하는 stochastic source와 제어하지 않는 nondeterministic op 확인; head-specific variance measurement protocol이 사전 등록됐는지 확인; RU-97 수치를 head 근거로 사용하지 않는지 인용 대조.

## 점검 5: 데이터·평가 절차 누수
- **확인하려는 것**: residual head 학습과 평가에서 split 단위, 중복, 전처리·정규화·통계량의 fit 범위, 라벨 사용, 평가 독립성, 매니페스트 고정, 임계값 선택, 기준선 오염이 없는지. `docs/PROJECT.md` §3.1의 fold 고정·SEAL 10 금지는 확인되나, head 전용 학습·평가 프로토콜은 문서에 없다.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test fold 라벨이 head 학습에 사용; normalization·whitening·scaler가 test fold 포함 fit; slide/patch level train-test 중복; early stopping이 Primary 7 test에 반응; hyperparameter가 Primary 7 결과를 보고 조정; baseline per-fold AUROC가 후보 실행 후 재계산; `+0.0030` 임계값이 결과에 맞게 변경; seed 선택; SEAL 10 열람; label-trained margin을 라벨 무관 게이트 ①에 투입.
- **확인 방법**: head training/evaluation script, fold manifest, baseline per-fold AUROC artifact, `*_per_fold.pt`, `scripts/analysis/branch_screen.py`, `tests/test_docs_consistency.py` 확인; seed 1~8 목록이 실행 전 고정됐는지; baseline tag와 fold ID가 후보와 동일 manifest인지; train/test slide/patch disjointness; normalization fit scope가 fold train only인지; test label 접근 로그; Primary 7 hyperparameter tuning 이력; SEAL 10 접근 금지 확인; ensemble 변경 시 §3.4 오염 검사 상수 영향 확인.

## 점검 6: 사전 등록안 개정
- **확인하려는 것**: 결과를 보기 전에 사전 등록안을 어떻게 고쳐야 하는지. 현재 요약된 설계는 `s_fit^2/8`과 seed 평균 Δ를 포함하지만, fold×seed 상호작용·head 전이·8 seed 충분성·평가 독립성이 미확정이다.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 무효 SE로 승격 판정; 결과 후 estimand·SE·seed 수·임계값 변경; best-seed 사용; missing fold/seed 무시; variance component 미보고; GMM 실측을 head 근거로 사용; label-trained head의 게이트 처리 누락; 실행 무효와 판별 불가 구분 없이 “성공/실패”만 보고.
- **확인 방법**: `talks/council/C-20260918-6_card.json`을 아래로 개정하고 `scripts/council.py check`, `bash scripts/run_tests.sh`, manifest diff로 검증. (1) estimand: seed 1~8 고정, fold-paired, task-equal macro Δ; best-seed 금지; random-seed claim은 별도 승인 또는 pilot 한정. (2) SE: `s_fit^2/8` 삭제 또는 재정의; `\bar d_{t,f}=(1/8)Σ_s d_{t,f,s}`의 task-cluster SE 또는 mixed-model `σ²_seed/(7·50·8)+σ²_fold×seed/(7·50·8)` 사용; fold·seed·fold×seed component 보고. (3) 판정: `+0.0030` 유지; 95% CI·sign agreement·악화 과제 전량 보고; CI 하한을 승격 필수 조건으로 만들지 않음; 실행 무효=missing data·leakage·manifest mismatch·seed 선택·SEAL 10 사용·임계값 변경, 판별 불가=분산 구성 추정 불가 또는 CI가 0 포함으로 부호 미확정, 반박=Δ_macro < +0.0030, 지지=Δ_macro ≥ +0.0030. (4) 데이터 무결성: 8×7×50 AUC 전량, baseline 고정 artifact, train/test disjoint, normalization train-only, test label 금지, Primary 7 후속 tuning 금지. (5) 게이트: label-trained head를 라벨 무관 게이트 ①에 넣지 않거나 사용자 면제·대안 절차를 사전 기록. (6) 충분성: fixed-seed이면 8은 정의, random-seed이면 seed 수·정밀도 기준 사전 고정, 결과 후 seed 추가 금지.

---

## Phase 2 — 교차

### T1 (이론)

이론적 병목 1 — 제안 1: mixed-model SE로 교체
- **가정**:  
  1. baseline `AUC_baseline(t,f)`가 seed와 무관한 고정 값이다.  
  2. 적합 불확실성이 seed 공통 이동(`v_s`)으로 요약되고, 그 이동이 fold/task cluster와 독립이라 `s_fit^2/8`을 quadrature할 수 있다고 둔다.  
  3. `s_fit`을 seed 주효과 분산의 추정치로 취급한다. 그러나 문서의 “적합 sd”는 seed macro의 경험 sd로 기록될 수 있고, 그 경우 fold×seed의 macro 평균 기여를 이미 일부 포함한다.  
  4. fold×seed가 지배적이라는 사실만으로 `s_fit^2/8`이 mean estimand의 SE를 과소평가한다고 둔다.  
  5. 제안된 LMM이 additive이고, task·fold·seed·fold×seed random effects가 서로 독립이며, 8개 seed가 exchangeable이고, fold도 exchangeable하다고 둔다.  
  6. `(t,f,s)`당 관측이 1개인데도 `w_{t,f,s}`와 `e_{t,f,s}`를 분리할 수 있다고 둔다.  
  7. 8개 seed로 seed 주효과와 fold×seed 분산성분을 안정적으로 추정할 수 있다고 둔다.

- **이 가정이 깨지는 조건**:  
  - baseline도 seed에 의존하면 paired Δ 정의 자체가 무효다. → **실행 무효**.  
  - `(t,f,s)`당 1관측이면 `w_{t,f,s}`와 `e_{t,f,s}`는 분리 불가하다. 제안된 모델은 그대로면 과파라미터화·미동정이다. → **실행 무효**.  
  - 재학습 단위가 fold×seed로 교차하지 않으면 fold×seed 잔차 구조가 성립하지 않는다. → **실행 무효**.  
  - seed가 독립·교환적이지 않으면 seed variance component 해석이 무의미하다. → **판별 불가**.  
  - fold×seed 분산이 task·fold별로 이질적이면 단일 residual variance 가정은 μ의 SE를 잘못 준다. → 모델 오규격.  
  - head가 사실상 결정적이거나 seed 주효과에 가까우면 mixed model은 불필요한 복잡성이다. → **실패/과잉**.  
  - LMM이 수렴하지 않거나 분산성분을 못 추정하면, 제안은 quadrature 복귀 없이 실행 무효로 만든다. fold×seed가 실제로 없어도 model failure만으로 무효가 될 수 있다. → **과잉 실행 무효**.  
  - head에서 fold×seed가 존재하는지는 아직 **판별 불가**다.

- **깨질 때 나타나는 관측 가능한 증상**:  
  - REML 수렴 실패, 음의 분산성분, 유한하지 않은 분산성분 추정.  
  - seed macro sd는 작으면서 fold-level Δ의 부호·순위가 seed마다 뒤집힘.  
  - 같은 fold의 candidate−baseline Δ가 seed마다 다른 방향으로 흔들림.  
  - task별 residual variance가 크게 달라짐.  
  - `s_fit^2/8` 기반 SE와 mixed-model SE가 head 실측에서 크게 달라짐.

- **문서의 어떤 관측이 이 가정과 충돌하는가**:  
  - RU-97 §11의 GMM 분산분해는 seed 주효과 `0~2.9%`에 비해 fold×seed 잔차가 `m8_raw 13.2%`, `m16_raw 23.0%`, `m32_raw 23.9%`, `whiten_d256_m8 6.7%`로 나타난다. 이는 GMM 적합 범위에서 “적합 변동이 seed 공통 이동이 아니라 fold 구조와 결합할 수 있다”는 반박 관측이다.  
  - 그러나 이 관측은 GMM 적합에 대한 것이지 73파라미터 residual head에 대한 실측이 아니다. head에 대한 충돌 관측은 **문서에 없음**.  
  - PROJECT §3.2의 공식 7-branch 결정성 기록은 현재 baseline이 seed 무관 고정일 가능성을 지지하지만, 후보 head가 추가된 뒤에도 baseline이 동일하게 고정된다는 것은 사전 등록안의 전제다.

이론적 병목 2 — 제안 2: seed 평균 Δ의 estimand 고정
- **가정**:  
  1. seed 평균 Δ가 “새 training seed를 뽑았을 때 fold-averaged paired Δ의 기대”를 추정한다고 둔다.  
  2. seed 1~8이 seed 모집단을 대표하는 표본이라고 둔다.  
  3. fold×seed 분산성분이 seed 주효과보다 크면 fold-level consistency를 “판별 불가”로 규정할 수 있다고 둔다.  
  4. seed-averaged fold Δ가 음인 fold 수와 과제 수만 보고하면 fold-level 해석 제한이 충분하다고 둔다.  
  5. seed 평균 Δ를 estimand로 고정이 사용자 승인 전이라도 설계 단계에서 확정 가능하다고 둔다.

- **이 가정이 깨지는 조건**:  
  - seed 1~8이 고정된 실행 라벨일 뿐 seed 모집단 표본이 아니면, μ는 population expectation이 아니라 8개 seed의 유한 평균이다. → **판별 불가/해석 제한**.  
  - 8개 seed가 대표적이지 않으면 “seed 기대”라는 표현은 근거 없다. → **실패**.  
  - fold×seed와 seed 주효과 분산성분 추정이 불안정하면, “fold×seed > seed main” 트리거는 결과에 따라 흔들린다. → **판별 불가**.  
  - 승격 결정이 overall mean만으로 내려지고 fold-level claim이 실제로 사용되지 않으면, fold-level disclaimer는 과잉이다. → **실패/과잉**.  
  - fold×seed가 없어도 fold별 이질성이 크면, 제안의 variance-ratio 규칙은 fold-level 문제를 놓친다. → **실패**.  
  - current_status는 적합 분산을 포함한 승격 판정 절차가 아직 미승인이라고 기록한다. estimand를 확정하는 것은 승인 전 설계 확정이다. → **실행 무효 위험**.  
  - head에서 fold×seed가 실제로 존재하는지는 **판별 불가**다.

- **깨질 때 나타나는 관측 가능한 증상**:  
  - `Δ_macro`는 양인데 seed-averaged fold Δ가 음인 fold가 많다.  
  - seed마다 fold별 Δ 순위가 달라진다.  
  - fold×seed와 seed 주효과 분산성분 비교가 작은 재분석에 따라 뒤집힌다.  
  - “mean estimand only” 문구가 실제 승격 결정에서는 사용되지 않는다.  
  - fold-level consistency를 주장하지 않음에도 fold-level 보고가 결정에 영향을 준다.

- **문서의 어떤 관측이 이 가정과 충돌하는가**:  
  - RU-97 §8은 동일 구성에서 seed 1이 원본 대비 `+0.0078 [+0.0043, +0.0113]`로 우위, seed 3이 `-0.0122 [-0.0178, -0.0066]`으로 열위라고 기록한다. 이는 단일 적합의 fold-level claim이 seed에 따라 뒤집힐 수 있다는 GMM 범위 관측이다.  
  - RU-97 §11은 fold×seed 잔차가 seed 주효과보다 큰 구성이 있음을 보여준다.  
  - 그러나 이 관측들은 GMM 적합에 대한 것이다. residual head의 seed 평균 Δ가 fold-level consistency를 추정하지 않는다는 점에 대한 head 실측은 **문서에 없음**.

이론적 병목 3 — 제안 3: 8회 재적합의 충분성과 GMM 전이 분리
- **가정**:  
  1. seed 1~8은 고정 실행 규모일 뿐 충분성 증명이 아니라고 둔다.  
  2. RU-97 §11의 GMM 분산분해 수치는 head SE 산출에 직접 쓰면 안 된다고 둔다.  
  3. 8회 이후 head-specific 분산성분을 유한하게 산출할 수 있다고 둔다.  
  4. 분산성분과 총 SE가 유한하면 점수 기준 `Δ_macro ≥ +0.003`으로 진행해도 된다고 둔다.  
  5. 추가 seed는 같은 후보 결과를 본 뒤가 아니라 새 사전 등록 전에만 허용한다고 둔다.  
  6. zero-init residual head의 seed 변동 구조가 GMM과 충분히 다르거나, 적어도 GMM 수치와 직접 같지 않다고 둔다.

- **이 가정이 깨지는 조건**:  
  - GMM과 head가 같은 fold·score aggregation·stochastic fitting 구조를 공유해 구조적 교훈만이라도 설계에 유용하면, GMM을 완전히 배제하는 것은 과잉 보수일 수 있다. → **실패/과잉**.  
  - 8회로 head-specific fold×seed 분산성분을 추정할 수 없으면, 제안은 승격 판정이 아니라 실행 무효로 종료된다. → **실행 무효**.  
  - 분산성분과 총 SE가 유한해도, SE가 `+0.003` 기준의 해석을 지지하지 못할 정도로 크면 점수 기준만으로 진행하는 것은误导적일 수 있다. → **판별 불가/실패**.  
  - 8회 결과를 보고 추가 seed가 필요함을 알게 되더라도 새 사전 등록 전에는 추가할 수 없으면, 적합 불확실성 측정이 완료되지 않은 채 종료된다. → **실행 무효**.  
  - zero-init이 seed 변동의 구조를 GMM과 다르게 만든다는 것은 **기전 미확인**이다.  
  - head의 fold×seed 구조가 GMM과 같은지 다른지는 **판별 불가**다.

- **깨질 때 나타나는 관측 가능한 증상**:  
  - 8개 seed macro가 거의 같으면서 fold-level Δ가 seed마다 크게 달라진다.  
  - REML 또는 ANOVA로 fold×seed 분산성분을 유한하게 추정하지 못한다.  
  - 총 SE가 유한하지만, `Δ_macro`의 `+0.003` 기준 해석을 지지하지 못할 정도로 크다.  
  - head에서 GMM과 유사한 fold×seed 양식이 나타나거나, 전혀 나타나지 않는다.  
  - 추가 seed가 필요해 보이지만 사전 등록 규칙 때문에 추가할 수 없다.

- **문서의 어떤 관측이 이 가정과 충돌하는가**:  
  - head fit variance 실측은 **문서에 없음**.  
  - 8회 재적합의 충분성을 뒷받침하는 power 분석, target SE, head-specific fit variance 실측은 **문서에 없음**.  
  - RU-97 §11의 GMM 분산분해는 head의 8회 충분성이나 GMM 전이를 직접 반박하지 않는다. head에 대한 충돌 관측은 **해당 관측 없음**.

### A1 (감사)

## 점검 1: 제안 1의 후보 학습 fold 단위 분리
- **확인하려는 것**: residual head seed 1~8 학습이 평가 fold의 라벨·슬라이드를 train/val에 포함하지 않는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: seed 평균 Δ가 `+0.003`을 넘고, fold별 Δ가 일관되게 양이며, mixed-model SE가 작게 나올 수 있다. 이는 성능 개선이 아니라 평가 무효(실행 무효)에 해당한다.
- **확인 방법**: 후보 학습 스크립트와 fold manifest에서 fold별 train/test ID, label 로딩, early stopping/val split을 확인한다. `grep -R "train" scripts/`, `grep -R "val" scripts/`, `grep -R "fold" scripts/`, per-fold raw margin 저장 여부를 확인한다.

## 점검 2: 제안 1의 baseline per-fold AUROC 고정
- **확인하려는 것**: `AUC_baseline(t,f)`가 후보 실행 전 고정 artifact로 저장되고, 후보·seed 결과를 보고 baseline을 재학습·재선택하지 않는 절차가 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: baseline이 후보에 맞춰 재선택되면 paired Δ가 후보 유리하게 이동하고, `Δ_macro`는 더 이상 고정 기준선 대비 차이가 아니다.
- **확인 방법**: baseline 태그 `ru90_shape_triple`, per-fold AUROC 파일 hash, 코드 버전, 후보 카드에서 baseline 재사용 규칙을 확인한다. `talks/council/C-20260918-6_card.json`, `PROJECT.md` §3.2, §4를 대조한다.

## 점검 3: 제안 1·2의 estimand/SE 공식이 SSOT와 일치하는가
- **확인하려는 것**: 제안 1의 mixed-model SE와 제안 2의 seed 평균 estimand가 `PROJECT.md` §4/§4.1의 fold 단위 `N=350` estimand와 seed 없는 `Δ_macro` 공식과 일치하거나, 승격 전에 SSOT로 반영되는가.
- **문서에서 확인되는가**: 모순됨
- **샌다면 나타날 증상**: SSOT와 다른 공식으로 승격 판정이 이루어지고, 재현·감사가 불가능하다. fold 단위 추정과 seed 평균 추정을 혼동해 fold-level claim으로 확대할 수 있다.
- **확인 방법**: `PROJECT.md` §4, §4.1과 `talks/council/C-20260918-6_card.json`의 estimand·SE 수식을 대조한다. `tests/test_docs_consistency.py`가 SSOT 불일치를 잡는지 확인한다.

## 점검 4: 제안 1의 seed 1~8 독립성과 중복
- **확인하려는 것**: seed 1~8이 초기화, data order, dropout, batch 구성 등 실제로 독립적인 무작위성을 바꾸는가. zero-init에서 seed가 어떤 변동을 만드는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: seed 간 per-fold Δ 상관계수가 1에 가까우면 fold×seed 분산이 0에 수렴하고, mixed-model SE가 과소평가된다. 반대로 seed가 data order만 바꾸고 fold train/test가 중복되면 평가 독립성이 깨진다.
- **확인 방법**: 학습 config에서 seed 사용 위치, 초기화, data shuffling, dropout, batch 구성을 확인한다. seed별 per-fold AUC 상관계수와 학습 로그를 비교한다.

## 점검 5: 제안 1의 seed 실패·제외 정책
- **확인하려는 것**: NaN, OOM, 수렴 실패 seed를 어떻게 처리하는지가 결과 전에 고정되어 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 실패 seed를 사후에 제외하면 유리한 seed만 평균에 남고, seed 평균 Δ와 SE가 편향된다. 이는 seed 선택 편향으로 평가 무효에 해당한다.
- **확인 방법**: 후보 카드와 실행 로그에서 seed 실패 처리 규칙을 확인한다. `grep -R "NaN" scripts/`, `grep -R "OOM" scripts/`, seed별 종료 상태를 확인한다.

## 점검 6: 제안 1의 mixed-model 식별성과 df
- **확인하려는 것**: 7 task × 50 fold × 8 seed로 `τ_t`, `u_{t,f}`, `v_s`, `w_{t,f,s}`, `e_{t,f,s}`를 분리하고, `μ`의 SE를 PROJECT §4의 7과제 등가 평균에 맞게 산출할 수 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: REML 수렴 실패, 음의 분산성분, df 불안정, SE 과소 또는 과대가 나타난다. 이 경우 판별 불가가 아니라 실행 무효로 처리해야 한다.
- **확인 방법**: mixed-model fit 로그, 분산성분 추정치, Satterthwaite/Kenward-Roger df, bootstrap 재추정, 모델 수렴 상태를 확인한다.

## 점검 7: 제안 2의 seed expectation 근거
- **확인하려는 것**: seed 1~8 평균을 “seed 기대”라고 부를 수 있는 seed 생성·표본화 과정이 사전에 정의되어 있는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 고정된 8개 seed의 평균을 모집단 seed 기대로 과장한다. positive `Δ_macro`를 fold 전체 개선 또는 seed 전체 개선으로 해석할 위험이 커진다.
- **확인 방법**: seed 목록이 결과 전에 고정되었는지, seed 생성 규칙이 있는지, seed sensitivity 분석이 사전에 정의되었는지 후보 카드와 결정 이력을 확인한다.

## 점검 8: 제안 3의 RU-97 GMM 수치 직접 사용 금지
- **확인하려는 것**: `m8_raw 13.2%`, `m16_raw 23.0%`, `m32_raw 23.9%`, `whiten_d256_m8 6.7%` 같은 RU-97 GMM 분산비율이 head SE 공식의 수치로 직접 사용되지 않는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: GMM 적합의 fold×seed 비중을 73파라미터 residual head에 전이하면, head 실측 없이 SE를 과대 또는 과소평가한다. 이는 기전 미확인을 기전으로 쓰는 오류다.
- **확인 방법**: SE 산출 코드와 후보 카드에서 RU-97 수치가 hardcode되지 않았는지 확인한다. `grep -R "13.2" scripts/`, `grep -R "23.9" scripts/`, `grep -R "RU-97" talks/council/`를 확인한다.

## 점검 9: RU-97 Primary 7 결과를 이용한 설계 변경 기록
- **확인하려는 것**: RU-97 Primary 7 fold×seed 결과를 보고 mixed-model SE로 바꾸는 것이 후보 결과 전 설계 적응으로 기록되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: 같은 Primary 7 자료로 설계 고치고, 같은 Primary 7 자료로 승격을 판정하면 선택 편향이 남는다. 데이터 접근과 선택 이력이 없으면 판별 불가다.
- **확인 방법**: `history/archive.md`, 후보 카드, 결정 로그에서 RU-97 결과 확인 시점과 설계 변경 시점을 대조한다.

## 점검 10: 전처리·정규화·score 스케일 fit 범위
- **확인하려는 것**: head 입력 feature, normalization, whitening, head output scale이 평가 fold를 포함해 fit되지 않고 고정되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: test fold 통계가 학습에 새면 fold 난이도가 상쇄되고, fold별 Δ가 인위적으로 작아진다. seed별 head 스케일 차이가 fold×seed 잔차로 위장되어 SE가 왜곡될 수 있다.
- **확인 방법**: feature pipeline, fit/transform 분리, score aggregation, head output calibration, LSAK 사양을 확인한다. `scripts/` 내 normalization·whitening·score 합산 코드를 검토한다.

## 점검 11: learned margin의 label-free gate ① 우회
- **확인하려는 것**: learned residual head margin을 라벨 무관 gate ① orthogonality에 넣지 않는 절차가 후보에 반영되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: label-dependent margin을 gate ①에 사용하면 §5의 라벨 무관 게이트가 우회되고, 탐색·선택 편향이 승격 경로로 새어 들어간다.
- **확인 방법**: `scripts/analysis/branch_screen.py` 사용 여부, `SCREEN_ONLY=1` 실행, §3.4 오염 검사, 후보 카드에서 gate ① 적용 범위를 확인한다.

## 점검 12: 원시 margin 저장과 독립 재계산
- **확인하려는 것**: `Δ_{t,f,s}`를 원시 margin에서 독립 재계산할 수 있는 per-seed per-fold raw score가 저장되는가.
- **문서에서 확인되는가**: 문서에 없음
- **샌다면 나타날 증상**: macro 또는 seed 평균만 저장되면 mixed-model, fold×seed 분산, 오염 검사, 독립 재계산이 불가능하다. 이는 재현 불가와 평가 무효에 해당한다.
- **확인 방법**: per-seed per-fold raw score artifact, `*_per_fold.pt`와 동등한 저장물, `scripts/run_tests.sh`, `tests/test_docs_consistency.py`를 확인한다.

## 점검 13: `+0.003` 임계값의 결과 전 고정
- **확인하려는 것**: mixed-model SE, 분산성분, seed 실패 처리를 보고 `+0.003` 임계값을 바꾸지 않는가.
- **문서에서 확인되는가**: 확인됨
- **샌다면 나타날 증상**: SE나 분산성분을 보고 임계값을 조정하면 사후 성공 컷오프가 되고, 승격 기준이 후보별로 달라진다.
- **확인 방법**: `PROJECT.md` §4, 후보 카드, 결정 로그에서 `+0.003`이 결과 전에 고정되었는지 확인한다.

---

## Phase 2b — 제안 좌석 응답

### P1 (제안)

- **T1-1**: 수용하고 수정 — T1의 식별성 지적은 옳다. `(t,f,s)`당 paired Δ가 1개이므로 `w_{t,f,s}`와 `e_{t,f,s}`를 분리할 수 없다. 제안 1의 모델을 `Δ_{t,f,s} = μ + α_t + u_{t,f} + v_s + r_{t,f,s}`로 바꾼다. `α_t`는 7개 과제의 고정 효과, `u_{t,f}`는 task 내 fold 효과, `v_s`는 seed 주효과, `r_{t,f,s}`는 fold×seed 상호작용과 측정 노이즈를 합친 잔차다. `s_fit^2/8` 항은 폐기한다. task random 효과는 7개 task만으로 불안정할 수 있으므로 1차 모델은 task fixed, 민감도 분석으로 task random을 둔다. 잔차 이질성이 의심되면 task별 잔차 분산 또는 sandwich robust SE를 사전에 두고, 수렴 실패·음/무한 분산성분은 `실행 무효`로 처리한다. `AUC_baseline(t,f)`가 seed 무관 고정 artifact가 아니면 paired Δ 정의 자체가 무효다.

- **T1-2**: 수용하고 수정 — seed 1~8이 seed 모집단의 표본임을 문서가 지지하지 않으므로, estimand을 “seed 기대”가 아니라 **고정 seed 1~8에 대한 fold-averaged paired Δ의 평균**으로 한정한다. `μ = (1/8)Σ_s (1/7)Σ_t (1/50)Σ_f Δ_{t,f,s}`. seed 생성·표본화 규칙이 정의되기 전에는 population expectation 해석을 쓰지 않는다. fold×seed/seed main ratio를 fold-level claim의 자동 트리거로 쓰지 않는다. 이 설계는 overall mean만 추정하고 fold-level 일관성은 추정하지 않는다. fold별 Δ, seed-averaged fold Δ가 음인 fold 수, task별 Δ, 분산성분 ratio는 보조 보고로만 둔다.

- **T1-3**: 수용하고 수정 — 8회는 충분성 증명이 아니라 고정 실행 규모다. RU-97 GMM의 fold×seed 비중은 head SE 산출에 수치로 직접 쓰지 않는다. 다만 GMM은 “재적합이 fold 순서를 뒤섞을 수 있다”는 질적 경고로만 사용한다. 8회 후 head-specific `Δ_{t,f,s}`에서 fold×seed 잔차와 총 SE를 산출한다. 총 SE가 유한하면 `+0.003` 점수 기준으로 진행하되, 총 SE가 관측 Δ보다 크면 부호를 확정하지 못해 `판별 불가`로 보고한다. 분산성분 비유한·수렴 실패는 `실행 무효`다. 추가 seed는 같은 후보 결과를 본 뒤가 아니라 새 사전 등록 전에만 허용한다. zero-init 73파라미터 head의 seed 변동 구조는 기전 미확인이다.

- **A1-1**: 수용하고 수정 — residual head seed 1~8 학습이 평가 fold의 라벨·슬라이드를 train/val에 포함하지 않는지를 결과 전 검증해야 한다. 문서에는 fold manifest와 학습 분리 절차가 없으므로, 위반이 확인되면 paired Δ가 더 이상 유효하지 않아 `실행 무효`로 처리한다.

- **A1-2**: 수용하고 수정 — 공식 7-branch baseline의 per-fold AUROC를 후보·seed 실행 전에 고정 artifact로 저장하고 코드 버전·hash를 고정해야 한다. 후보 결과를 보고 baseline을 재학습·재선택하면 paired Δ의 기준선이 바뀌므로 `실행 무효`다. PROJECT §3.2의 결정성 기록은 baseline이 seed 무관 고정일 가능성을 지지하지만, artifact 고정 절차 자체는 문서에 없으므로 추가한다.

- **A1-3**: 수용하고 수정 — 제안 1·2의 seed 포함 estimand와 mixed-model SE는 현행 PROJECT §4/§4.1의 seed 없는 `N=350` 공식과 다르다. 승격 전에 SSOT와 `tests/test_docs_consistency.py`에 반영해야 한다. `current_status.md`의 “적합 분산을 포함한 승격 판정 절차가 미승인”과 일치하므로, 사용자 승인 전에는 학습 파라미터 후보의 승격 판정을 보류한다.

- **A1-4**: 수용하고 수정 — seed 1~8이 초기화, data order, dropout, batch 구성 중 어떤 무작위성을 실제로 바꾸는지 결과 전에 고정해야 한다. zero-init에서 seed 변동이 거의 없거나 seed 간 per-fold Δ 상관계수가 1에 수렴하면 fold×seed 분산 추정 무효다. 이 경우 `실행 무효` 또는 재설계다. seed 변동 구조 자체는 현재 문서에 없으므로 결과 전 판별 불가이나, 검증 절차는 필수다.

- **A1-5**: 수용하고 수정 — NaN, OOM, 수렴 실패 seed의 처리를 결과 전에 고정해야 한다. 사후에 유리한 seed만 남기는 제외는 seed 선택 편향이다. 8/8 완료 또는 사전 등록된 동일 seed 재시도만 허용하고, 그래도 실패하면 estimand의 seed 집합이 설계와 달라져 `실행 무효`로 처리한다.

- **A1-6**: 수용하고 수정 — T1-1과 동일하게 식별 가능한 모델로 교체한다. `w_{t,f,s}`와 `e_{t,f,s}`를 분리하지 않고 `r_{t,f,s}`로 합친다. 1차 모델은 task fixed, fold(task) random, seed random, residual random. Satterthwaite 또는 Kenward-Roger df를 보고하고, 분산성분과 상대 비중을 필수 보고한다. REML 수렴 실패, 음/무한 분산성분, df 산출 실패는 `실행 무효`다.

- **A1-7**: 수용하고 수정 — “seed 기대”라는 표현은 seed 생성·표본화 과정이 사전 정의되지 않으면 근거가 없다. seed 1~8은 결과 전 고정된 유한 실행 세트일 뿐이다. 따라서 estimand는 고정 seed 평균으로 한정하고, seed 모집단 기대 해석은 seed 생성 규칙이 정의된 후에만 추가한다.

- **A1-8**: 반박(수정 불필요) — A1도 “확인됨”으로 표시했으므로, RU-97 GMM 수치가 head SE 공식에 직접 사용되지 않는다는 조건은 이미 제안 3에 반영되어 있다. 추가 수정은 없다. 다만 GMM 수치를 head SE 수치로 쓰지 않고 질적 경고로만 사용한다는 문구를 유지한다.

- **A1-9**: 수용하고 수정 — RU-97 Primary 7 GMM 결과를 보고 사전 등록안을 고치는 것은 후보 head 결과 이전의 설계 적응이다. 그래도 선택 편향 가능성을 남기지 않으려면 RU-97 접근 시점, 설계 변경 시점, 후보 head Primary 7 결과 열람 전임을 결정 로그에 기록해야 한다. 후보 head Primary 7 결과를 확인한 뒤 설계 기준을 바꾸면 `실행 무효`다.

- **A1-10**: 수용하고 수정 — head 입력 feature, normalization, whitening, score scale의 fit 범위를 결과 전에 고정해야 한다. 평가 fold의 통계가 fit에 포함되면 fold 난이도가 상쇄되고, seed별 스케일 차이가 fold×seed 잔차로 위장될 수 있다. 위반은 `실행 무효`다. LSAK 사양 미기재 open issue와 연결된다.

- **A1-11**: 수용하고 수정 — learned residual head margin을 라벨 무관 게이트 ①에 넣으면 `D-048` 이후의 절차 구멍을 우회한다. learned margin을 게이트 ①에 쓰려면 label-free margin 환원 절차를 먼저 정의하거나, 게이트 ① 적용을 보류해야 한다. 우회 사용은 선택 편향으로 `실행 무효`다.

- **A1-12**: 수용하고 수정 — per-seed per-fold raw margin 또는 raw score를 저장해야 한다. seed 평균·macro만 저장하면 mixed-model, fold×seed 분산, 오염 검사, 독립 재계산이 불가능하다. 이 경우 재현 불가와 평가 무효에 해당하므로 `실행 무효`다.

- **A1-13**: 반박(수정 불필요) — PROJECT §4에서 `+0.003`은 결과 전 고정 기준이고, A1도 확인됨으로 표시했다. SE나 분산성분을 보고 임계값을 바꾸면 사후 성공 컷오프가 된다. SE가 크면 `판별 불가`로 보고할 수 있지만, 임계값 자체는 변경하지 않는다.

---

## Phase 3 — 종합

### S1 (종합)

## 지지 (증거가 뒷받침함)

- **RU-97 §11의 GMM 분산분해는 “재적합 효과가 fold와 무관한 seed 공통 이동”이라는 가정을 지지하지 않는다.**  
  주어진 실측에서 seed 주효과는 `0~2.9%`인데, fold×seed 잔차는 `m8_raw 13.2%`, `m16_raw 23.0%`, `m32_raw 23.9%`, `whiten_d256_m8 6.7%`로 나타난다. `current_status.md`도 “재적합이 fold 순서를 뒤섞으며, fold를 더 모아도 줄지 않는다”고 기록한다. 따라서 fold×seed 구조가 존재하는 경우, seed 주효과만 전제하는 `s_fit^2/8`은 estimand의 불확실성으로 불충분할 수 있음을 지지한다.

- **총 SE는 estimand와 동일한 paired Δ 배열에서 산출되어야 한다는 원칙이 지지된다.**  
  `PROJECT.md` §4.1의 estimand는 Primary 7 고정 fold 분포에서 후보−기준선 대응 Δ의 기댓값이다. `D-048` 이후 적합 분산을 포함해야 하므로, seed 1~8을 쓰는 설계라면 총 SE는 seed 평균 Δ를 만드는 `Δ_{t,f,s}` 배열에서 fold·task·seed·fold×seed 구조를 반영하는 model-based 또는 resampling SE로 산출하는 것이 estimand와 일치한다.

- **seed 평균 Δ는 fold×seed 상호작용 하에서 fold-level 일관된 개선이 아니라 marginal mean으로 해석되어야 한다는 점이 지지된다.**  
  seed 1~8이 고정 실행 세트라면 seed 평균 Δ는 “고정 seed 1~8에 대한 fold-averaged paired Δ의 평균”이다. fold×seed 상호작용이 있으면 seed 평균은 fold별 양/음 효과가 상쇄된 overall mean일 수 있다. 따라서 “모든 fold에서 개선됐다”는 fold-level claim으로 확장하면 안 된다.

- **원안의 `s_fit^2/8`을 그대로 유지할 근거가 부족하다는 점이 지지된다.**  
  주어진 문서에는 `s_fit`의 산출 정의, fold×seed 잔차 계측, head-specific fit variance 실측, 8회 충분성 기준이 없다. RU-97의 분산분해는 GMM 적합에 대한 것이므로, 73파라미터 residual head의 SE 설계 근거로 직접 쓰일 수 없다. 따라서 원안의 quadrature SE는 결과 전 검증이 필요하다.

## 반박 (증거가 반박함)

- **GMM 범위에서 “재적합이 모든 fold를 같이 밀어 올리는 seed 주효과”라는 설명은 반박된다.**  
  RU-97 §11에서 seed 주효과는 작고 fold×seed 잔차가 더 큰 구성이 있다. 특히 `m16_raw`와 `m32_raw`에서 fold×seed 잔차가 `23.0%`, `23.9%`로 seed 주효과보다 크다. 이는 재적합이 fold별 Δ를 서로 다른 방향으로 흔들 수 있음을 보여준다. 다만 이 반박은 GMM 적합에 한정된다.

- **단일 적합의 fold-level 우위 주장은 GMM 범위에서 반박된다.**  
  RU-97 §8에서 동일 구성 `whiten_d256_m8`의 seed 1은 원본 대비 `+0.0078 [+0.0043, +0.0113]`로 우위, seed 3은 `-0.0122 [-0.0178, -0.0066]`으로 열위로 나타난다. RU-97 §10에서도 `m32_raw`의 단일 seed 우위가 seed 평균에서는 사라졌다. 따라서 단일 적합 또는 단일 seed의 fold-level claim은 seed 변동에 의해 뒤집힐 수 있다.

- **RU-97 GMM의 fold×seed 비중을 73파라미터 residual head의 수치적 근거로 전이하는 것은 반박된다.**  
  RU-97의 분산분해는 GMM 적합에 대한 실측이다. 73파라미터 zero-init residual head의 seed 분산 실측은 주어진 문서에 없다. GMM의 EM/초기화 민감성과 학습 head의 optimizer·data order·batch·zero-init 이후 gradient update는 다른 stochastic source일 수 있다. 따라서 GMM 수치를 head SE 공식의 수치로 쓰는 것은 근거가 없다.

- **`s_fit^2/8`이 fold×seed 지배 상황에서 총 SE의 유효한 보정이라는 주장은, `s_fit`이 estimand-compatible 분산 성분으로 정의되지 않는 한 반박된다.**  
  `s_fit`이 seed macro sd라면 fold×seed의 macro 평균 기여를 일부 포함할 수 있으나, `SE_taskcluster`와의 중복·독립 관계가 통제되지 않는다. `s_fit`이 seed 주효과만 반영한다면 fold×seed 잔차를 놓친다. 주어진 문서에는 이 구분이 없으므로, 원안 공식은 fold×seed 구조를 적절히 반영한다고 볼 수 없다.

## 판별 불가 (증거가 가르지 못함)

- **73파라미터 zero-init residual head에서도 fold×seed 잔차가 지배적인지는 판별 불가하다.**  
  RU-97의 fold×seed 관측은 GMM 적합에 대한 것이다. residual head의 seed 1~8 per-fold·per-seed Δ 실측은 문서에 없다. zero-init이 seed 변동을 줄일지, optimizer·data order·batch가 fold×seed를 만들지는 기전 미확인이다.

- **8회 재적합이 충분한지는 판별 불가하다.**  
  문서에는 8회 충분성을 뒷받침하는 power 분석, target SE, head fit variance 실측, seed 수에 따른 분산성분 수렴 곡선이 없다. fixed-seed estimand라면 8은 estimand의 정의일 뿐 충분성 증명이 아니다. random-seed estimand라면 seed 1~8이 seed 모집단을 대표한다는 근거도 없다.

- **원안의 `sqrt(SE_taskcluster^2 + s_fit^2/8)`이 head에서 총 SE를 과소평가하는지 과대평가하는지는 판별 불가하다.**  
  `s_fit`의 정의가 문서에 없고, head-specific fold×seed 분산성분도 없다. 따라서 원안 SE와 head 실측 기반 SE의 차이를 정량적으로 판단할 수 없다.

- **7 task × 50 fold × 8 seed 배열에서 fold×seed 분산성분을 안정적으로 추정할 수 있는지는 판별 불가하다.**  
  `(t,f,s)`당 paired Δ가 1개이면 fold×seed 상호작용과 측정 노이즈를 분리할 수 없다. fold×seed와 residual noise를 하나의 잔차 성분으로 합쳐야 식별 가능성이 있으나, 8 seed로 seed 주효과와 잔차 분산성분을 안정적으로 추정할 수 있는지는 head 실측 전에는 알 수 없다.

- **baseline 공식 7-branch가 후보 실행에서도 seed 무관 고정 artifact로 유지되는지는 절차적으로 판별 불가하다.**  
  `PROJECT.md` §3.2는 공식 7-branch의 결정성을 기록하지만, 후보 head 실행 전 baseline per-fold AUROC artifact를 hash·코드 버전으로 고정하는 절차는 주어진 문서에 없다.

- **seed 1~8이 실제로 독립적·교환적인 무작위 실행인지 판별 불가하다.**  
  seed가 초기화, data order, dropout, batch 구성 중 어떤 stochastic source를 바꾸는지 문서에 없다. zero-init에서 seed 간 차이가 거의 없거나, seed가 data order만 바꾸는 경우 fold×seed 분산 해석이 달라진다.

- **GMM fold×seed 잔차의 기전은 판별 불가하다.**  
  RU-97은 유효 성분 붕괴, 초기화 민감성, component alignment artifact 가능성을 언급하지만 기전을 확정하지 않는다. “GMM 고유의 성질”이라고 단정할 근거도, “head에도 동일하게 나타난다”고 단정할 근거도 없다.

## 실행 무효 (실행·계측 결함으로 판단 불가)

- **원안대로 `sqrt(SE_taskcluster^2 + s_fit^2/8)`만 쓰고 head-specific fold×seed 잔차를 계측하지 않으면, 승격 판단은 실행 무효다.**  
  fold×seed 잔차가 존재할 수 있는데도 seed 주효과만 전제하는 SE로 판단하면, 계측이 estimand와 일치하지 않는다. 특히 `s_fit` 정의가 없거나 fold×seed 잔차를 분리하지 않으면 총 SE는 유효하지 않다.

- **baseline per-fold AUROC가 후보·seed 실행 전에 고정 artifact로 저장되지 않으면 paired Δ는 실행 무효다.**  
  후보 결과를 보고 baseline을 재학습·재선택·재계산하면 기준선이 바뀌어 `Δ_{t,f,s}`가 더 이상 고정 기준선 대비 차이가 아니다.

- **per-seed per-fold raw margin 또는 raw score가 저장되지 않으면 mixed-model SE, fold×seed 분산, 독립 재계산이 불가능하므로 실행 무효다.**  
  seed 평균 macro만 저장하면 fold×seed 구조를 검증할 수 없고, 오염 검사·재현도 불가능하다.

- **실패 seed를 사후에 제외하거나 best seed를 선택하면 seed 선택 편향으로 실행 무효다.**  
  NaN·OOM·수렴 실패 seed의 처리는 결과 전에 고정되어야 한다. 8/8 완료 또는 사전 등록된 동일 seed 재시도만 허용하고, 그래도 실패하면 estimand의 seed 집합이 설계와 달라져 판단 불가가 아니라 실행 무효로 처리해야 한다.

- **평가 fold의 라벨·슬라이드가 head 학습·validation·normalization fit에 포함되면 실행 무효다.**  
  test fold 통계가 학습에 새면 fold 난이도가 상쇄되고, seed별 스케일 차이가 fold×seed 잔차로 위장될 수 있다.

- **`(t,f,s)`당 1관측인데 fold×seed 효과와 residual noise를 분리하는 mixed model을 그대로 실행하면 식별 불가로 실행 무효다.**  
  fold×seed와 측정 노이즈는 하나의 잔차 성분으로 합쳐야 한다. 분리된 두 성분을 추정하려 하면 모델이 미동정될 수 있다.

- **mixed model이 수렴하지 않거나 분산성분이 음·무한·비유한이면, 기존 `s_fit^2/8`으로 되돌리지 말고 실행 무효로 처리해야 한다.**  
  계측 결함을 더 단순한 공식으로 덮으면 불확실성 측정이 estimand와 일치하지 않는다.

- **learned residual head margin을 라벨 무관 게이트 ①에 넣으면 `D-048` 이후의 절차 구멍을 우회하므로 실행 무효다.**  
  label-dependent margin을 label-free orthogonality 게이트에 쓰려면 label-free margin 환원 절차를 먼저 정의해야 한다.

## 좌석 공백 (기권·실패로 빠진 관점)

- **Phase 2 R1(반증) 좌석이 본문 비어 기권/실패했다.**  
  따라서 반증 좌석의 Phase 2 최종 반론은 이 종합에 반영되지 않았다. 이는 관점 공백이지, 해당 주장이 증거로 부정됐다는 뜻이 아니다.

## 다음 행동 (권한 등급과 함께)

- **T1 — 사용자 승인 필요: `C-20260918-6` 사전 등록안을 실행 전에 개정한다.**  
  1. estimand를 “seed 1~8 고정 세트에 대한 fold-averaged paired Δ의 평균”으로 명시한다.  
     `μ = (1/8)Σ_s (1/7)Σ_t (1/50)Σ_f Δ_{t,f,s}`.  
  2. `s_fit^2/8` 항을 폐기하거나, `s_fit`을 estimand-compatible 분산 성분으로 재정의한다.  
  3. 총 SE는 `Δ_{t,f,s}` 배열에서 task·fold·seed·fold×seed 구조를 반영하는 mixed model 또는 task/seed cluster bootstrap으로 산출한다.  
  4. mixed model은 fold×seed와 residual noise를 분리하지 말고 하나의 잔차 성분으로 합친다.  
  5. 분산성분, seed별 Δ, task별 Δ, seed-averaged fold Δ가 음인 fold 수, leave-one-seed-out 민감도를 의무 보고한다.  
  6. `+0.003` 기준은 유지하되, 총 SE·CI·분산성분은 fold×seed를 포함해야 한다.

- **T1 — 사용자 승인 필요: 실행 무효 조건을 결과 전에 고정한다.**  
  baseline artifact hash·코드 버전 고정, per-seed per-fold raw margin 저장, train/test fold 분리, normalization·whitening·score scale의 fit 범위, seed 실패 처리, best-seed 금지, SEAL 10 선택 금지를 사전 등록에 넣는다. 위반이 확인되면 승격 판단을 `실행 무효`로 처리한다.

- **T0 — 기록: RU-97 GMM 수치를 head SE의 수치적 근거로 사용하지 않는다.**  
  `m8_raw 13.2%`, `m16_raw 23.0%`, `m32_raw 23.9%`, `whiten_d256_m8 6.7%`는 GMM 적합 실측이다. head SE 공식에 hardcode하지 않고, “재적합이 fold 순서를 뒤섞을 수 있다”는 질적 경고로만 사용한다. RU-97 접근 시점과 설계 변경 시점을 결정 로그에 기록한다.

- **T1 — 사용자 승인 필요: `PROJECT.md` §4/§4.1과 `tests/test_docs_consistency.py`를 seed-aware 승격 절차로 갱신한다.**  
  현재 SSOT는 seed 없는 `N=350` paired Δ estimand를 기준으로 한다. 학습 파라미터 후보의 seed 평균 estimand와 mixed-model SE를 승격 전에 SSOT로 반영해야 한다. 사용자 승인 전에는 학습 파라미터 후보의 승격 판정을 보류한다.

- **T2 — 사용자 최종 승인: 적합 분산 포함 승격 절차가 승인되기 전에는 승격 판단을 하지 않는다.**  
  `current_status.md`는 적합 분산을 포함한 승격 판정 절차가 미승인이라고 기록한다. 모든 후보 판정은 `SEAL 10` hold-out 미검증 상태를 유지하며, `SEAL 10` 결과로 모델·후보·구조를 고르는 것은 금지된다.

- **T1 — 사용자 승인 필요: 8회 재적합을 “충분”이 아니라 “고정 실행 규모”로만 기록한다.**  
  8회 후 head-specific 분산성분과 총 SE를 산출한다. 총 SE가 유한하면 `+0.003` 점수 기준으로 진행하되, 총 SE가 관측 Δ보다 크거나 CI가 0을 포함하면 `판별 불가`로 보고한다. 분산성분 비유한·수렴 실패는 `실행 무효`다. 추가 seed는 같은 후보 결과를 본 뒤가 아니라 새 사전 등록 전에만 허용한다.

## 하지 말 것 (이유와 함께)

- **RU-97 GMM의 fold×seed 비중을 residual head SE의 수치로 사용하지 말 것.**  
  RU-97은 GMM 적합에 대한 실측이고, 73파라미터 residual head의 seed 분산 실측은 문서에 없다. 전이는 기전 미확인이다.

- **8회 재적합을 충분하다고 주장하지 말 것.**  
  power 분석, target SE, head fit variance 실측, seed 수에 따른 안정성 곡선이 문서에 없다. 8회는 고정 실행 규모일 뿐이다.

- **seed 평균 Δ를 fold-level 일관된 개선으로 해석하지 말 것.**  
  fold×seed 상호작용이 있으면 seed 평균은 fold별 효과가 상쇄된 marginal mean일 수 있다. fold-level claim은 별도 증거가 필요하다.

- **best seed를 선택하거나 실패 seed를 사후에 제외하지 말 것.**  
  seed 선택 편향은 paired Δ와 SE를 왜곡하고, 승격 판단을 실행 무효로 만든다.

- **mixed model 실패 시 기존 `s_fit^2/8`으로 되돌리지 말 것.**  
  계측 결함을 더 단순한 공식으로 덮으면 fold×seed 잔차를 무시하게 된다. 실패는 `실행 무효`로 처리해야 한다.

- **`+0.003` 임계값을 SE·분산성분·seed 결과를 보고 바꾸지 말 것.**  
  사후 성공 컷오프가 되고, 후보별 승격 기준이 달라진다.

- **`SEAL 10`을 모델·후보·구조 선택에 사용하지 말 것.**  
  `D-047`에 따라 `SEAL 10`은 최종 hold-out이며, 그 결과로 개선 방향을 고르는 것은 부정행위다.

- **GMM 관측을 head 기전의 입증으로 쓰지 말 것.**  
  GMM의 fold×seed 잔차와 head의 학습 변동은 다른 stochastic source일 수 있다. 기전은 미확인이다.

- **learned margin을 라벨 무관 게이트 ①에 넣지 말 것.**  
  `D-048` 이후 label-trained margin을 label-free orthogonality 게이트에 넣는 절차 구멍이 아직 메워지지 않았다.
