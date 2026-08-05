# v31 Proposal: CCTS에서 Class-Conditional Evidence Router(CCER)로
## Generic novelty가 아니라 support class를 직접 지지하는 희소·밀집 증거 라우팅

**작성일**: 2026-08-04

**상태**: CCER-Lite 구현 및 seed 42 1차 학습 진행 중 — 미평가

**기준선**: v30 S2 (`poolz_l2` + bag cardinality log-uniform training)

**예상 수정 범위**: [`src/models/baseline.py`](../src/models/baseline.py), [`src/datasets/synthetic_data.py`](../src/datasets/synthetic_data.py), 신규 진단/평가 스크립트와 테스트

---

## 0. 제안 요약

v31의 핵심 질문은 “Top-K의 K를 1, 4, 8, 16 중 무엇으로 정할 것인가?”가 아니다. 더 근본적인 문제는 **bag이 커질수록 무관한 배경 인스턴스 중에서도 큰 novelty가 우연히 나타난다**는 것이다. 고정 Top-K는 선택 개수만 고정할 뿐 이 극값 편향을 보정하지 못한다.

따라서 v31은 기존 global/fractional tail에 absolute Top-K를 덧붙이는 대신 다음 세 요소를 하나의 검증 가능한 가설로 제안한다.

1. **Context-calibrated instance evidence**: 각 인스턴스의 novelty를 support/context의 경험적 null 분포에 대조해 tail probability로 변환한다.
2. **Expected-false-positive tail scan**: 고정 K 대신 `n * p_i`를 기준으로 여러 희소도 스케일을 스캔한다. 같은 토큰은 bag 크기가 달라도 null에서 비슷한 수준의 우연한 검출을 보도록 설계한다.
3. **Counterfactual sparse meta-training**: 양성 bag에만 outlier를 넣는 쉬운 생성 과제를 피하고, 두 클래스 모두 nuisance outlier를 가지되 일부만 label-causal하도록 만든다. 인스턴스 수 역시 bag별로 다르게 샘플링한다.

이 설계의 목표는 Musk 점수 하나를 설명하는 사후 서사가 아니라 다음 가설을 반증 가능하게 시험하는 것이다.

> 대형 bag 성능 저하의 일부가 uncalibrated extreme-value evidence에서 발생한다면, context-null로 보정된 tail scan은 background cardinality를 바꿔도 예측을 더 안정적으로 유지하면서 sparse-response 판별력을 높일 것이다.

현재 증거만으로 이 가설이 참이라고 단정하지 않는다.

---

## 1. 현재 관측으로 말할 수 있는 것과 없는 것

### 1.1 확인된 관측

- v30 S2의 Musk overall AUROC는 0.8539이고 `n > 34` band는 0.698이다.
- [`scripts/diagnose_tail_dilution.py`](../scripts/diagnose_tail_dilution.py)의 별도 ridge probe에서 `n > 34` AUROC는 `top_1=0.611`, `top_3=0.627`, `frac_0.15=0.690`이었다.
- Musk `n > 34` band는 25 bags뿐이며 class 구성은 positive 7 / negative 18이다. 이 band의 AUROC는 표본 변동성이 크다.

### 1.2 아직 확인되지 않은 것

- 위 probe는 실제 `shared_tail_encoder`, context anchors, meta-classifier를 재현하지 않으므로 tail branch의 인과적 결함을 입증하지 않는다.
- `frac_0.15`가 관측상 `top_1`과 `top_3`보다 좋으므로, 이 결과는 absolute Top-K의 우월성을 지지하지 않는다.
- Musk의 양성 신호가 실제로 1개 또는 0.1% 인스턴스에 존재한다는 instance-level annotation은 없다.
- 이미 Musk 결과를 반복 관찰해 설계를 선택했으므로 Musk는 더 이상 완전히 untouched한 confirmatory zero-shot test로 부를 수 없다. v31에서는 Musk를 transfer development benchmark로 명시하고, 최종 확인용 외부 MIL 데이터셋을 별도로 잠가야 한다.

---

## 2. 왜 고정 Absolute Top-K가 충분하지 않은가

### 2.1 고정 K는 cardinality-invariant하지 않다

`k=16`은 `n=16`에서는 bag의 100%, `n=100`에서는 16%, `n=1000`에서는 1.6%다. 선택 개수는 제한되지만 의미하는 prevalence가 계속 변한다. 또한 `k > n`이면 여러 스케일이 동일한 전체 bag으로 붕괴한다.

### 2.2 큰 bag에서는 null maximum도 커진다

배경 novelty score를 확률변수 `S`라 하고 threshold가 `t`일 때, 독립 근사 아래 순수 배경 bag에서도

\[
P(\max_i S_i > t \mid n)=1-F_0(t)^n
\]

이다. `n`이 커질수록 label과 무관하게 큰 score가 관찰될 가능성이 증가한다. Top-1이나 Top-16은 이 다중 비교 효과를 제거하지 않는다.

### 2.3 novelty와 label-causal evidence는 다르다

현재 tail 선택은 nearest anchor와의 cosine novelty로 결정된다. 희귀한 nuisance, 측정 오류, 또는 정상적인 작은 population도 높은 novelty를 가질 수 있다. “가장 이상한 인스턴스”가 “분류에 필요한 인스턴스”라는 보장은 없다.

### 2.4 고정 후보 `(1, 4, 8, 16)`에는 데이터 기반 근거가 없다

현재 probe는 `k=4,8,16`을 직접 비교하지 않았고 `(1,4,8,16)`은 일정 비율의 지수 수열도 아니다. 이를 아키텍처 상수로 확정하면 Musk에 대한 사후 튜닝과 작은 표본의 우연을 구조에 고정할 위험이 있다.

---

## 3. 제안 아키텍처: Context-Calibrated Tail Scan

### 3.1 인스턴스 표현과 scalar evidence

bag `b`의 인스턴스 `x_i`를 현재 방식대로 nearest context anchor `a_{j(i)}`에 정렬하고 deviation을 만든다.

\[
d_i=x_i-a_{j(i)}, \qquad h_i=E_\theta(d_i)
\]

별도의 작은 head가 scalar tail score를 출력한다.

\[
s_i=g_\phi(h_i,\;1-\cos(x_i,a_{j(i)}),\;m_{j(i)})
\]

여기서 `m_j`는 slot proportion/dispersion/reliability metadata다. `h_i`는 방향과 상태 정보를 보존하고, `s_i`는 어떤 인스턴스를 tail evidence로 볼지 결정한다.

### 3.2 context-null calibration

각 query instance의 score를 **query bag을 제외한 context instances**의 같은 slot score 분포와 비교한다. Context 표본이 작은 slot은 global context distribution으로 shrinkage한다.

\[
\hat p_i=
\frac{1+\sum_{r\in C_{j(i)}}\mathbf 1[s_r\ge s_i]}
{1+|C_{j(i)}|}
\]

학습 시에는 indicator를 temperature-controlled sigmoid로 바꾼 soft empirical CDF를 사용하고, 평가 시에는 hard rank 버전도 함께 보고 calibration 차이를 점검한다. 이는 엄밀한 독립 p-value를 보장하려는 장치가 아니라, **support episode 내부의 관측된 null tail scale로 score를 표준화하는 장치**다.

### 3.3 expected-false-positive scan tokens

각 인스턴스에 대해 multiplicity-corrected evidence를 정의한다.

\[
e_i=-\log(\max(\hat p_i,\epsilon))-\log n
\]

그리고 고정 K 대신 expected false-positive budget `lambda`에 따른 soft gate를 사용한다.

\[
w_i^{(\lambda)}=
\sigma\left(\frac{\log\lambda-\log(n\hat p_i)}{\tau}\right),
\qquad
t_\lambda=
\frac{\sum_i w_i^{(\lambda)}h_i}
{\sum_i w_i^{(\lambda)}+\epsilon}
\]

초기 후보는 `lambda in {0.25, 1, 4}`로 제한한다. 이는 “상위 몇 개를 무조건 선택”하는 스케일이 아니라, null에서 허용할 우연한 exceedance 수준을 뜻한다. 각 token에는 다음 reliability metadata를 projection해 더한다.

- `log1p(sum(w))`: 유효 tail mass
- `max(e_i)`: 가장 강한 보정 evidence
- `sum(w)/n`: 선택 비율
- `log(n)`: 남아 있는 cardinality dependence를 classifier가 명시적으로 조정할 정보
- context null sample count: calibration 신뢰도

### 3.4 sparse/dense 역할 분리

- 기존 global summary와 slot center/spread/rare tokens는 composition 및 dense response를 담당한다.
- CCTS tokens는 sparse, context-unexpected evidence를 담당한다.
- 기존 fractional global-tail tokens와 새 CCTS를 무조건 병렬로 누적하지 않는다. 먼저 `fractional-only`, `CCTS-only`, `hybrid`를 ablation하고 token 수 증가 자체의 효과를 통제한다.

토큰 수를 늘린 모델에는 동일 parameter/FLOP budget의 dummy-token 또는 wider-baseline control을 둔다. 개선이 단순 용량 증가 때문인지 분리하기 위해서다.

---

## 4. 메타 학습 재설계: Counterfactual Sparse Mixture Curriculum

기존 `any_positive_sparse`처럼 양성 bag에만 1–3개 shifted instance를 넣으면 모델이 “outlier 존재 여부”라는 쉬운 shortcut을 학습할 수 있다. 또한 `n=1..3`에서는 이 과제가 전혀 sparse하지 않다. v31 생성기는 다음 계약을 따라야 한다.

### 4.1 bag별 cardinality

- episode마다 하나의 `n`을 공유하지 않고 각 bag의 `n_b`를 독립적으로 log-uniform 샘플링한다.
- 한 episode 안에 small/medium/large bag이 함께 존재해야 한다.
- padded tensor + instance mask 또는 기존 ragged path 중 하나를 공식 학습 경로로 정하고 dense/ragged 동치 테스트를 둔다.

### 4.2 prevalence curriculum

반응 개수 `r_b`는 고정 1–3이 아니라 두 regime에서 샘플링한다.

- sparse: `r_b in {1,2,4}`에서 `r_b <= n_b` 적용
- proportional: `pi_b ~ LogUniform(0.005, 0.30)`, `r_b=ceil(pi_b n_b)`

`n_b < 16`에서는 sparse-task 비중을 낮추거나 별도 small-bag task로 표시해 “1개 = 100%”를 rare example로 세지 않는다.

### 4.3 hard-negative nuisance matching

- 두 클래스 모두 동일한 개수와 비슷한 크기의 nuisance outlier를 가진다.
- label-causal shift만 episode-specific response direction 및 population membership과 일치한다.
- random isolated outlier, batch-like global shift, 정상 rare component를 음성 hard negative로 포함한다.
- 반응 인스턴스는 무작위 위치뿐 아니라 coherent component 내부에서도 샘플링해 “독립된 극단점”과 “희귀 subpopulation”을 모두 다룬다.

### 4.4 counterfactual pairs와 보조 손실

같은 background bag으로 다음 쌍을 생성한다.

- response를 주입한 bag / 주입하지 않은 bag
- background instance만 추가·삭제한 두 bag
- 동일 causal instances를 유지하고 background만 복제한 두 bag

학습 손실은 다음과 같이 분리한다.

\[
L=L_{episode}
+\alpha L_{localize}
+\beta L_{cardinality-consistency}
+\gamma L_{null-calibration}
\]

- `L_episode`: 기존 episode classification loss
- `L_localize`: synthetic에서만 instance mask를 이용한 causal-evidence ranking loss
- `L_cardinality-consistency`: background 증감 전후 prediction/token consistency
- `L_null-calibration`: no-effect instance의 calibrated tail score가 특정 bag size에서 체계적으로 커지지 않도록 하는 loss

보조 loss는 실제 데이터 평가에는 label이나 instance mask를 요구하지 않는다.

---

## 5. 구현 순서와 안전 계약

### Phase 0 — 현재 회귀 복구

새 실험 전에 아래를 먼저 고친다.

1. Dense fractional-tail path가 token을 append하도록 복구한다.
2. Absolute-tail 실험 코드는 유지하려면 ragged path에도 동일 semantics로 구현한다.
3. `structured_tokens_per_bag`와 실제 token 수를 forward 초기에 명시적으로 검증한다.
4. `absolute_tail_ks=()`에서 v30 checkpoint/config forward가 그대로 동작함을 회귀 테스트한다.

이 단계는 CCTS의 성능 주장과 분리된 correctness prerequisite다.

### Phase 1 — 진단 가능한 CCTS 모듈

신규 모듈은 최소한 다음 auxiliary 값을 반환해야 한다.

- raw score `s_i`
- calibrated `p_i`
- scale별 effective tail mass
- injected causal instance recall@weight-mass (synthetic only)
- null bag에서 `max(e_i)`와 `log(n)`의 관계

### Phase 2 — 생성기와 ragged training

bag별 cardinality, counterfactual pairs, nuisance-matched sparse task를 추가한다. 기존 response tasks의 확률을 임의로 줄이지 않고 총 episode 수 또는 sampling schedule을 조정해 기존 task의 절대 학습량을 보존한다.

### Phase 3 — 모델/데이터 요인 분리 실험

| ID | Architecture | Sparse generator | Consistency/localization |
|---|---|---|---|
| A | corrected v30 | 기존 | 없음 |
| B | absolute Top-K control | 기존 | 없음 |
| C | CCTS | 기존 | 없음 |
| D | corrected v30 | 신규 | 없음 |
| E | CCTS | 신규 | 없음 |
| F | CCTS | 신규 | 있음 |

`A→C`는 architecture 효과, `A→D`는 data 효과, `C/D→E`는 interaction, `E→F`는 auxiliary objective 효과를 측정한다.

---

## 6. 사전등록 평가 계획

### 6.1 실행 규칙

- 최소 5개 training seed를 사용한다.
- 모든 후보는 동일 episode 수, optimizer budget, checkpoint-selection rule을 사용한다.
- Musk 결과를 보고 `lambda`, temperature, loss weight를 바꾸지 않는다. 이 값은 synthetic validation과 null-calibration suite에서만 선택한다.
- 평균만 보고하지 않고 seed별 결과와 paired bootstrap 95% CI를 함께 보고한다.

### 6.2 필수 지표

| 영역 | 지표 | 목적 |
|---|---|---|
| Correctness | dense/ragged token 및 logit 동치 | 두 실행 경로의 의미 일치 |
| Synthetic dense | AUROC | 기존 composition/dense 성능 보존 |
| Synthetic sparse | AUROC, AUPRC, causal instance recall | 희소 판별과 localization 확인 |
| Cardinality counterfactual | background 증감 전후 `abs(delta probability)` | 크기 불변성 직접 측정 |
| Null calibration | label별 `corr(max evidence, log n)` | 극값 크기 편향 확인 |
| Musk | overall 및 사전 정의 4개 band AUROC | transfer 성능 |
| Reliability | band 표본 수, class 수, bootstrap CI | 작은 band의 불확실성 공개 |

### 6.3 승격 조건

v31 승격은 단일 Musk 목표치가 아니라 아래를 모두 만족할 때만 허용한다.

1. corrected v30 대비 synthetic dense AUROC 차이의 95% CI 하한이 `-0.005` 이상.
2. synthetic sparse AUROC 또는 AUPRC가 seed median 기준 `+0.03` 이상 개선되고 causal-instance recall도 함께 개선.
3. cardinality counterfactual의 median `abs(delta probability)`가 v30 대비 30% 이상 감소.
4. Musk overall AUROC가 `-0.01`보다 크게 회귀하지 않음.
5. Musk `n > 34` AUROC가 median 기준 `+0.05` 이상 개선. 단, positive 7 / negative 18의 작은 표본이므로 CI가 0을 포함하면 “유망하지만 미확정”으로 판정.
6. 개선이 5개 seed 중 최소 4개에서 같은 방향.

Musk overall `>=0.90`이나 large-band `>=0.85`는 연구 목표로 기록할 수 있지만, 근거 없이 승격 gate로 확정하지 않는다.

---

## 7. 실패 판정과 후속 의사결정

| 관측 | 해석 | 결정 |
|---|---|---|
| Null calibration은 개선되나 sparse AUROC 불변 | 극값 편향은 줄였지만 score가 causal하지 않음 | evidence head/localization 재검토 |
| Synthetic sparse만 개선, Musk 불변 | 생성 과제와 Musk mechanism 불일치 | Musk를 rare-instance 문제로 단정하지 않음 |
| Large band 개선, small band 회귀 | calibration shrinkage 또는 small-context reliability 문제 | slot/global null fallback 조정 |
| 모든 모델이 generator 변경만으로 개선 | architecture보다 meta-distribution이 핵심 | 단순한 D안 우선 승격 |
| Absolute Top-K가 CCTS와 동률 또는 우세 | 복잡성 정당화 실패 | 더 단순한 Top-K control 선택 |
| Seed/CI 불안정 | 표본 또는 최적화 불확실성 | 확정 승격 금지, 외부 데이터 확인 |

---

## 8. 최종 제안

v31은 “Absolute Top-K + Any-Positive”로 확정하지 않는다. 정식 후보는 **Cardinality-Calibrated Tail Scan + nuisance-matched counterfactual sparse curriculum**으로 둔다.

이 방향의 장점은 특정 `k`가 Musk에 맞을 것이라는 추측에 의존하지 않고, 대형 bag에서 반드시 발생하는 multiple-comparison/extreme-value 문제를 직접 모델링한다는 점이다. 동시에 architecture, generator, auxiliary loss를 분리 평가하므로 실패하더라도 무엇이 틀렸는지 알 수 있다.

구현에 앞서 Phase 0 회귀 복구와 테스트 보강이 필수이며, 그 전까지 상태는 **proposal only**다.

---

## 9. 2026-08-05 Architecture Upgrade: CCER-Lite

CCTS 실험은 Musk overall 0.8376, `n > 34` 0.6032로 v30을 넘지 못했다. 그러나 구현 감사에서 `detach → sort → searchsorted`가 `ccts_score_head`의 gradient를 완전히 차단하고, dense 학습과 ragged 평가의 null calibration semantics도 다름이 확인되었다. 따라서 이 결과는 학습 가능한 CCTS 전체에 대한 반증이 아니라 현재 구현 후보의 실패로 해석한다.

후속 v31은 aggregator 내부의 label-free anomaly score를 더 복잡하게 만들지 않는다. 기존 class memory와 query instance 사이의 discriminative evidence를 episode-level classifier에서 직접 계산하는 **CCER-Lite**를 1차 후보로 둔다.

### 9.1 구현된 계약

- support label로 만들어진 class memory와 query instance의 class별 evidence를 사용한다.
- hard Top-K/searchsorted 없이 `temperature in {0.25, 1.0, 4.0}`의 cardinality-normalized LogMeanExp를 사용한다.
- 모든 class에 공통인 generic anomaly evidence를 class축 centering으로 제거한다.
- support class separation으로 temperature mixture를 조정하는 label-equivariant router를 사용한다.
- class-discriminative evidence가 약하면 residual을 0으로 보내는 explicit null gate를 사용한다.
- 기존 v30 rare branch는 control로 보존하고 CCER residual은 0.10에서 시작한다.
- CCTS와 Absolute Top-K는 CCER-Lite config에서 비활성화한다.

### 9.2 검증된 안전 조건

- CCER router/null/residual 파라미터에 finite gradient가 도달한다.
- dense episode path와 list/ragged path logits가 허용 오차 안에서 일치한다.
- 모든 instance를 동일 횟수 복제해도 LogMeanExp route score가 변하지 않는다.
- projection-enabled v30 구조에서 forward/backward가 정상 동작한다.

### 9.3 실행 및 다음 ablation

1차 run은 [`configs/train_v31_ccer_lite.yaml`](../configs/train_v31_ccer_lite.yaml), seed 42, 50 epochs다. 이 run은 구현 smoke와 방향성 확인용이며 단일 seed 승격 판정에 사용하지 않는다. 완료 후 corrected CCTS, CCER-Lite, effective-cardinality 보정, full support-conditioned router, 입력 read-bridge를 각각 분리해 비교한다. Read-bridge는 zero-padding이 실패 원인이라는 선결론 없이 독립 요인으로만 평가한다.

---

## 10. 2026-08-05 Architecture Upgrade: CCER-v2

CCER-Lite의 Musk 결과와 on/off 검사는 residual contribution이 약 `1.4e-4`에 불과해
사실상 v30 그대로였음을 보였다. 원인은 단순한 residual scale 부족이 아니라 (1) 기존
rare branch와 표현을 공유한 점, (2) projected class memory가 세포 수준 identity를 이미
압축한 점, (3) dense route 선호와 null gate가 희소 route를 동시에 약화한 점, (4) v30과
다른 task mixture로 backbone까지 재학습한 점의 결합이다. 따라서 scale tuning 실험을
반복하지 않고 경로 자체를 교체한다.

### 10.1 확정 아키텍처

1. support class prototype은 projection 전 aligned slot-center에서 계산한다.
2. support/query encoder는 기존 rare evidence encoder와 parameter를 공유하지 않는다.
3. query cell과 class-slot prototype의 cosine evidence를 만든 뒤 class축 centering한다.
4. `Top-1`, `Top-4`, `mean`을 병렬 route로 두고 learned router로 결합한다.
5. 각 route에는 최소 `0.30 / 3 = 0.10`의 weight를 보장한다.
6. 별도 null gate는 두지 않는다. 무정보 상황은 class-centered logit 자체가 0에
   가까워지는 구조로 처리한다.
7. 최종 1x1 output head는 정확히 0으로 초기화한다. 따라서 v30 warm-start 시 새 branch가
   기존 예측을 교란하지 않으며, optimizer가 데이터로부터 유효한 방향을 먼저 정한다.

### 10.2 학습 계약

- `configs/train_v31_ccer_v2.yaml`은 v30 config를 직접 상속하며 task probability를
  재정의하지 않는다.
- v30 best checkpoint는 resume가 아니라 weight-only initialization으로 읽는다.
- 새 CCER-v2 parameter는 base LR, 공통 v30 backbone은 `0.05x` LR로 학습한다.
- 1차 run은 20 epochs로 제한한다. 목적은 광범위한 tuning이 아니라 새 경로가 실제
  contribution과 대형-bag 개선을 만드는지 판정하는 것이다.

### 10.3 구현 완료 조건

- v30 checkpoint의 공통 key를 모두 읽고 신규 CCER-v2 key 외 missing이 없어야 한다.
- warm-start 직후 v30/v31 logits의 최대 차이는 0이어야 한다.
- Top-1 route는 더 낮은 background cell 추가에 불변이어야 한다.
- 모든 route weight는 설정된 floor 이상이어야 한다.
- batched와 single-episode forward가 일치해야 한다.
- zero-init 첫 backward에서 output head에 finite nonzero gradient가 도달하고, head가 열린
  뒤 독립 encoder에도 gradient가 도달해야 한다.

위 조건은 모두 구현 및 targeted test에서 충족됐다. CCER-Lite는 checkpoint 호환과 실패
재현을 위해 제거하지 않는다.
