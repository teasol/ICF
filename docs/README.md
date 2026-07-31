# Documentation map

**Last updated**: `2026-08-01 13:20:00 KST`
**Architecture Version**: `24` (`architecture_version = 24`) — v24 확정 (residual + bottleneck bag projection)

문서는 **새 대화 세션으로 접속하는 Agent가 최우선으로 읽는 Living 문서 5개(`docs/` 루트)**와 **과거 기록/딥다이브 분석서(`docs/history/`)**로 이원화하여 관리합니다.

---

## 1. 새 세션 접속 Agent가 최우선으로 정독하는 Living 문서 (Docs Root)

사용자가 매번 새 채팅 세션으로 접속할 때, 새로 시작한 Agent는 아래 `docs/` 최상위 루트의 Living md 파일 5개를 우선 정독하고 **Git commit log/diff**를 조회하여 작업 맥락을 동기화합니다:

1. [`agent_handoff.md`](agent_handoff.md): 새 세션 Agent 초기화 수칙, Git 기반 워크플로우, 실행 환경, 타임아웃, 테스트 검증, Docs/Config 정리 규칙
2. [`current_status.md`](current_status.md): 개발 현황, 최신 실증 수치, v22 retrieval 제거 결정과 근거, 실험 전략(합성=결정 / ICI=최종 테스트), 평가 프로토콜, Next Action Plan (SSOT)
3. [`current_architecture.md`](current_architecture.md): Architecture v22 모델 구조 (4대 수학 기술, retrieval 없는 context 구성), Logit Fusion 수식
4. [`current_experiments.md`](current_experiments.md): 실험 전략과 검정력, 평가 프로토콜, Stage 1~3 실행 명령어 및 실증 수치
5. [`README.md`](README.md): 문서 맵 및 갱신/아카이빙 가이드라인

---

## 2. 과거 기록 및 딥다이브 분석서 (History & Deep-Dive Archives)

[`docs/history/`](history/)는 과거 설계, 특정 시점의 딥다이브 분석 및 실험 판단 근거 자료를 보관합니다. 현재 실행 지침이 아니며 `docs/` 최상위 루트를 깨끗하게 유지하기 위해 하위로 아카이빙되었습니다:

- [`history/v21_retrieval_investigation.md`](history/v21_retrieval_investigation.md): **v22에서 retrieval을 제거하기까지의 전체 조사 기록** — 3대 가설 검증, 구현 오류 규명, 그리고 n=87에서 모든 비교가 통계적으로 구분 불가능했다는 결론 (§4-⑧)
- [`history/v21_retrieval_experiments.md`](history/v21_retrieval_experiments.md): v21 retrieval 시대(Phase 1~6c)의 실험 프로토콜과 실측값
- [`history/retrieval_architecture_analysis.md`](history/retrieval_architecture_analysis.md): Naive Cosine Retrieval 실패 원인 상세 분석 및 40차원 Signal-Aware Retrieval 2-Pass Streaming 설계 (v21 당시 설계 문서)
- [`history/v20_scalability_plan.md`](history/v20_scalability_plan.md): v20/v21 아키텍처 Scalability 검증 및 Hard Real-World 문제 프로토콜
- [`history/architecture_v18.md`](history/architecture_v18.md): v18 구조
- [`history/architecture_v19.md`](history/architecture_v19.md): CSP 확정 전 v19 centered 구조
- [`history/v19_acceptance_protocol.md`](history/v19_acceptance_protocol.md): 초기 v19 acceptance 기준
- [`history/candidate_a_b_comparison.md`](history/candidate_a_b_comparison.md): Candidate A vs B 20-epoch short training 및 v20 선정 결과
- [`history/learnability_ladder.md`](history/learnability_ladder.md): ladder 설계와 단계 정의
- [`history/nuisance_ablation_c4_d_d0_d4.md`](history/nuisance_ablation_c4_d_d0_d4.md): nuisance ablation 결과
- [`history/architecture_v23_candidates.md`](history/architecture_v23_candidates.md): v23/v24 bag-collapse 후보 설계 및 T5-A/B/C 제안. **2026-08-01 v24 확정 결정과 T5-A 미해결 사유가 문서 상단에 기록됨**
- [`history/medium_b200_baseline.md`](history/medium_b200_baseline.md): B200 medium baseline
- [`history/synthetic_data_and_tasks.md`](history/synthetic_data_and_tasks.md): synthetic generator와 task 정의

---

## 3. 갱신 및 관리 수칙 (Maintenance Rules)

- **Living 문서 유지**: `docs/` 최상위 루트에는 오직 5개의 Living 문서만 유지합니다.
- **Git 커밋 동기화**: 세션 핸드오프 시 작업을 남김없이 커밋하고 커밋 내역/diff를 `agent_handoff.md` 및 `current_status.md`에 반영합니다.
- **아카이빙 규칙**: 특정 버전 딥다이브 보고서나 계획 문서는 완료 시 즉시 `docs/history/`로 이관하여 `docs/` 루트를 단순하고 가독성 높게 유지합니다.
- **Config 루트 관리**: `configs/` 루트에는 확정된 v24 entry point(`train_v24_medium_bag_proj_residual.yaml`)와, ICI 파이프라인이 아직 참조하는 v22 entry point(`train_v22_medium`, `train_v22_hard_realworld`, `train_v22_ici_finetune`, `train_v22_ici_scratch` 등)를 유지합니다. 폐기된 v23/v24 candidate(`train_v23_medium_bag_mean`, `train_v24_medium_bag_proj`, `train_v24_medium_bag_proj_bottleneck`)는 `configs/archive/v23_v24_candidates/`로, 구버전(v18~v21)은 `configs/archive/`로 관리합니다. ICI의 fold/seed는 config가 아니라 `--cv`/`--seed`로 주입하므로 fold별 config를 만들지 않습니다.
- **단일 출처화**: 동일한 수치나 진행 상태를 중복해 기록하지 않으며, 상태는 `current_status.md`에만 기록합니다.
- **자율 연속 실행과 추적성**: `current_status.md`에 다음 Action과 판정 기준이 명확하면 재확인 없이 실행합니다. 각 논리 단위의 결과·명령·로그/산출물 경로·판단·후속 Action을 SSOT에 기록하고 Git 커밋하여 다른 작업공간이 즉시 이어받을 수 있게 합니다.
