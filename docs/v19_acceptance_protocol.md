# Architecture v19 acceptance protocol

이 문서는 architecture v19 구현 전에 고정한 합격 기준이다. 결과를 확인한
뒤 threshold나 checkpoint 선택 규칙을 변경하지 않는다.

## 평가 대상

전체 learnability ladder를 반복하지 않는다.

- C4-D: nuisance-off composition 기준 성능 보존
- D0: global bag shift 단독 조건의 직접 해결
- C4: 모든 nuisance가 결합된 조건의 개선

Oracle abundance는 validation diagnostic으로만 사용하며 모델 입력이나 loss에
사용하지 않는다.

## 공통 학습 및 평가 규칙

- 최대 20 epoch, 5-epoch linear warm-up
- AdamW, learning rate 5e-4, BF16 mixed precision
- global gradient norm clipping 1.0
- 최종 확인 seed 42, 43, 44
- checkpoint는 epoch 5–19 중 val_ce_loss가 가장 낮은 것을 선택
- validation은 checkpoint 선택에만 사용
- 최종 metric은 별도의 고정 evaluation bank에서 계산
- evaluation bank는 stage별 최소 4,096 episode

개발 중 seed 42가 명백히 실패하면 seed 43과 44는 실행하지 않는다.

## 필수 구조 및 수치 게이트

- centered_delta의 instance 축 평균이 수치 오차 범위에서 0
- bag마다 서로 다른 동일-instance shift를 더해도 centered_delta,
  centered_x, global_spread와 final logits가 동일
- FP32 final-logit max absolute difference <= 1e-5, relative difference
  <= 1e-4
- BF16 final-logit max absolute difference <= 5e-3
- dense와 variable-length bag 경로 모두 통과
- instance/context-bag permutation invariance와 label permutation
  equivariance 유지
- query label이 forward에 사용되지 않음
- token 수 1 + 36 + 3 = 40과 outer episode batch shape 유지
- raw mean과 raw query instance가 class memory, branch logit 또는 final logit에
  사용되지 않음
- v18 또는 version 없는 checkpoint는 거부하고 v19 checkpoint만 load
- full-size BF16 forward/backward에서 loss, branch logits, gradients와
  parameters가 finite
- global norm clipping 적용 및 optimizer parameter update 확인

## 성능 기준

| Stage | 필수 기준 |
|---|---|
| C4-D | 3-seed mean AUROC >= 0.92, 각 seed >= 0.88, matched v18 대비 하락 <= 0.03 |
| D0 | 3-seed mean AUROC >= 0.85, 각 seed >= 0.80, v18 대비 상승 >= 0.15 |
| D0 oracle | oracle AUROC >= 0.95, oracle-model gap <= 0.15 |
| C4 | 3-seed mean AUROC >= 0.75, 각 seed >= 0.70, v18 대비 상승 >= 0.10, oracle-model gap <= 0.20 |

권장 목표는 D0 AUROC >= 0.90, D0 oracle-model gap <= 0.10, C4
AUROC >= 0.80이다.

## 판정

- PASS: 모든 구조·수치 게이트, C4-D, D0, C4 기준 통과
- CONDITIONAL PASS: C4-D와 D0는 통과했지만 C4 기준 미달
- FAIL: D0 기준 미달 또는 C4-D 비열등성 실패
- INVALID: 불변성, version, finite, config 또는 평가 protocol 위반
