# Documentation map

문서는 역할에 따라 세 영역으로 나눈다.

## 처음 프로젝트를 맡은 agent

[`agent_handoff.md`](agent_handoff.md)를 먼저 읽는다. 저장소 진입 순서, 실행 환경, 변경 금지 조건, 핵심 코드와 검증 명령을 담는다.

## 현재 문서

- [`current_status.md`](current_status.md): 현재 상태, 최신 결과, 진행 중인 작업과 다음 결정
- [`current_architecture.md`](current_architecture.md): production 모델의 실제 representation, branch와 final logit
- [`current_experiments.md`](current_experiments.md): 현재 synthetic task, 학습·평가 protocol과 후보 승격 절차

## 과거 기록

[`history/`](history/)는 설계 및 실험 판단의 근거다. 현재 실행 지침이 아니며 과거 threshold, alias, config와 현재 상태가 다를 수 있다.

- [`history/architecture_v18.md`](history/architecture_v18.md): v18 구조
- [`history/architecture_v19.md`](history/architecture_v19.md): CSP 확정 전 v19 centered 구조
- [`history/v19_acceptance_protocol.md`](history/v19_acceptance_protocol.md): 초기 v19 acceptance 기준
- [`history/learnability_ladder.md`](history/learnability_ladder.md): ladder 설계와 단계 정의
- [`history/nuisance_ablation_c4_d_d0_d4.md`](history/nuisance_ablation_c4_d_d0_d4.md): nuisance ablation 결과
- [`history/medium_b200_baseline.md`](history/medium_b200_baseline.md): B200 medium baseline
- [`history/synthetic_data_and_tasks.md`](history/synthetic_data_and_tasks.md): synthetic generator와 task 정의

## 갱신 규칙

- 반복해서 따라야 하는 작업 규칙은 `agent_handoff.md`에 둔다.
- 지금 무엇을 하고 있는지와 최신 결과는 `current_status.md`에만 둔다.
- 현재 모델 계산 구조는 `current_architecture.md`, 실험 protocol은 `current_experiments.md`에 둔다.
- 완료되어 현재 판단 근거로만 남는 내용은 `history/`로 이동한다.
- 동일한 수치나 상태를 여러 최신 문서에 중복 기록하지 않는다.
