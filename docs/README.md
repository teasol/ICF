# Documentation map

**Last updated**: `2026-09-03`
⚠️ **서버를 옮겼다면 `agent_handoff.md`를 먼저 읽을 것** — 노드 종속 설정은 `scripts/node_env.sh` 하나로 모았다.
**Active 구성**: **v120 — 학습 파라미터 0 (6-Branch Trimmed Mean Voting)**. within-slide PCA(K=256) 사영 +
**CV(off-diag 32,640) + CT(k-means++ 256 token) + BM(PCA-32 bag-mean) + BD(spectral entropy) + QA(128D quantiles) + DS(32D salience denoised)** + Trimmed Mean (최저/최고 1개 절사 후 중앙 4개 평균, DD는 제거).
Primary 7 macro **0.6265**, SEAL 10 macro **0.6972**, 전체 17 macro **0.6681**, seed std 0.00000.
실행: `bash scripts/eval_v120.sh <gpu> <tag>`. 전체 명세는 `agent_handoff.md` 및 `current_architecture.md`.
⚠️ **결정론적이므로 t·p·CI 금지**(§151-1) — Primary 7 부호 일치와 독립 집단 재현으로 판정.
실행 중: 없음.


> **새 세션이 먼저 알아야 할 3가지 (2026-08-15, 상세는 `current_status.md` 최상단)**
> 1. **데이터 분포 축은 닫혔다 (§129)** — 합성 에피소드를 실제 UNI2 통계에 맞추는 것은 도움이 안 될
>    뿐 아니라 **닫은 격차가 많을수록 단조로 나빠진다**(v102 t=−2.70 기각, v100 t=−3.59 기각).
>    이 축에서 새 arm을 설계하지 말 것.
> 0. **학습 파라미터 0 구성이 확정됐다 (§139)** — 사영은 fold context의 within-slide PCA, head는
>    라벨 반대칭이 강제하는 상수 3개(CV:DD:CT = 1 : −0.238 : +0.199). 재현:
>    `ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1`.
> 2. **문제는 편향이 아니라 분산일 수 있다 (§130·§131)** — **4 seed의 최소 검출 효과가 0.0121**인데
>    15개 arm이 쫓던 효과는 전부 ±0.005 안쪽이었다. **"미판정"은 "효과 없음"이 아니라 "측정 불가"다.**
>    시드 앙상블은 학습 비용 0으로 +0.004~0.006을 주고 **독립 3회 재현**됐다.
> 3. **판정은 게이트가 자동으로 하지 않는다 (§118)** — 최종 승격/기각은 macro + task 10개 전부의
>    성능대별 패턴 + arm 간 일관성을 종합한 **사용자 판단**이고, 보고 형식이 정해져 있다(§118-3).

문서는 **새 대화 세션으로 접속하는 Agent가 최우선으로 읽는 Living 문서 5개와 현행 proposal 1개(`docs/` 루트)**, **과거 기록/딥다이브 분석서([`docs/history.md`](history.md))**로 이원화하여 관리합니다.

---

## 1. 새 세션 접속 Agent가 최우선으로 정독하는 Living 문서 (Docs Root)

사용자가 매번 새 채팅 세션으로 접속할 때, 새로 시작한 Agent는 아래 `docs/` 최상위 루트의 Living md 파일 5개와 현재 `architecture_*_proposal.md` 1개를 우선 정독하고 **Git commit log/diff**를 조회하여 작업 맥락을 동기화합니다:

1. [`agent_handoff.md`](agent_handoff.md): 새 세션 Agent 초기화 수칙, Git 기반 워크플로우, 실행 환경, 타임아웃, 테스트 검증, Docs/Config 정리 규칙
2. [`current_status.md`](current_status.md): 개발 현황, 최신 실증 수치, 판정 프로토콜(§107-3 게이트 + **§118 사용자 종합 판단**), 열린 과제, Next Action Plan (SSOT). §2~§97 본문은 `history.md` §20–§23으로 아카이빙되고 스텁+포인터만 남았다(§101)
3. [`current_architecture.md`](current_architecture.md): 활성 **v83 linear head** relation 구조와 역사적 CV-only/Encoder+Ridge 비교, 데이터·학습 계약
4. [`current_experiments.md`](current_experiments.md): 실험 전략과 검정력, 평가 프로토콜, Stage 1~3 실행 명령어 및 실증 수치
5. [`README.md`](README.md): 문서 맵 및 갱신/아카이빙 가이드라인
6. `architecture_*_proposal.md`: 현재 활성 개선 proposal. 완료·폐기 시 핵심 결론을 `history.md`에 기록하고 원문은 git 이력에 보존. (**2026-08-09 현재 활성 proposal 없음**)

---

## 2. 과거 기록 및 딥다이브 분석서 (History & Deep-Dive Archives)

[`docs/history.md`](history.md)는 과거 설계·딥다이브 분석·실험 판단 근거의 **통합·요약본**입니다. `docs/history/` 폴더의 개별 문서(딥다이브 분석서, 옛 아키텍처 설계안, 폐기된 proposal, 과거 세션 아카이브)를 2026-08-09에 한 파일로 통합했고, 원문은 git 이력에 보존됩니다. 현재 실행 지침이 아니며 `docs/` 루트를 깨끗하게 유지합니다.

주요 내용:

- **§1** 버전 관리 정책(정수 `architecture_version`, semver 폐기, v18→v22 미적용 수정)
- **§2** 평가 방법론 불변식(paired bootstrap CI, 합성 지표 불신, SEAL 10개 macro 평균 판정)
- **§3** bf16-mixed/수치 안전 계약
- **§4~§17** 아키텍처 진화 연대기 — v18 learnability ladder · v20 Candidate A/B · v21 retrieval 제거 · v22~v24 bag-collapse · 0.70 정보 상한 · v26/v27/v29 폐기 · v30(B1+B2) · Musk 0.95 로드맵 · v31~v33 CCER/DR-CCER/MR-BagPFN 폐기 · v34 large-context · v35 스트리밍 · v36/v37 기각 · CV-only 전환 · v41 sketch 기하
- **§18** 운영·인프라 교훈(NCCL P2P, launcher/큐 함정, purge 교훈)
- **§19** 다시 열면 안 되는 결론 / 재개 시 전제 (quick reference)
- **§20~§23** (2026-08-12 추가) CV-2 손잡이 소진과 계보 B 일반화 실패 · 합성 데이터 축(per-bag cardinality, XOR, manifold) · canonical CV/DD/CT와 v70~v77 계보 · v77 파생 arm 전수 기각과 판정 프로토콜 전환(eigen 미분 금지 포함)
- 각 항목에 원본 파일명·작성 시점을 출처로 병기

폐기 architecture/연구 진단 테스트의 archive 정책은 [`../tests/history/README.md`](../tests/history/README.md)를 참고합니다.

---

## 3. 갱신 및 관리 수칙 (Maintenance Rules)

- **Living 문서 유지**: `docs/` 최상위 루트에는 5개의 Living 문서와 현행 proposal 1개만 유지합니다.
- **Git 커밋 동기화**: 세션 핸드오프 시 작업을 남김없이 커밋하고 커밋 내역/diff를 `agent_handoff.md` 및 `current_status.md`에 반영합니다.
- **아카이빙 규칙**: 특정 버전 딥다이브 보고서나 계획 문서는 완료 시 **핵심 결론(ADR·트레이드오프·레슨)을 `docs/history.md`의 해당 시기 절에 추가**하고, 개별 원문 파일은 새로 만들지 않습니다(원문은 git 이력에 보존). `docs/` 루트를 단순하고 가독성 높게 유지합니다.
- **Config 루트 관리**: `configs/` 루트에는 **현재 활성 파이프라인의 entry point만** 둔다.
  ⚠️ 이 규칙의 상세는 원래 `agent_handoff.md` §7에 있었으나 그 문서가 3개 절로 압축될 때
  사라졌다 — **이 절이 현행 단일 출처**다(archive 파일 헤더 200여 개가 인용하는 "agent_handoff
  SS7.3"도 같은 옛 절을 가리키는 역사적 표기다). **2026-09-02 정리 기준 루트 구성**:

  | config | 역할 |
  |---|---|
  | `train_v98_p1_reverse_1536_1gpu.yaml` | **루트에 남은 유일한 config**. v120 활성 경로가 로드하는 **체크포인트 껍데기**([`scripts/node_env.sh`](../scripts/node_env.sh) `ICF_CONFIG` 기본값). v106+ 가 projection과 head를 덮어쓰므로 이 파일의 학습값은 마진에 닿지 않는다(§152) |

  ⚠️ **활성 baseline v120은 학습 파라미터가 0개**다. 따라서 루트에 학습 arm config를 새로 만들
  이유가 없고, arm 정의는 config가 아니라 [`scripts/lib/arms.sh`](../scripts/lib/arms.sh)의
  `icf_arm_v1xx` 함수 + 환경변수로 주입한다. ICI의 fold/seed도 config가 아니라 `--cv`/`--seed`로
  주입하므로 fold별 config를 만들지 않는다.

  **2026-09-02 정리 내역 (§205)**: 학습 파라미터 계보(v77~v105) 루트 config 26개를 시대별
  `configs/archive/` 폴더로 이관하고, 참조가 0인 config-group 파일 23개와 Research Harness
  전용 yaml 5개를 삭제했다(tracked config 277 → 249). 이관 폴더는
  `v77_hard_orthogonal/`, `v80_v82_seed_batch/`, `v83_linear_head/`,
  `v86_v93_episode_shape/`, `v94_v102_cell_value/`, `v103_v105_head_proj/`이며,
  기존 시대별 폴더(`v34_largectx/` … `v84_deep_head/`)와 함께 전부 `base_config` 없는
  **자체 포함형**으로 보관한다. 원문은 git 이력에 보존된다.
- **단일 출처화**: 동일한 수치나 진행 상태를 중복해 기록하지 않으며, 상태는 `current_status.md`에만 기록합니다.
- **자율 연속 실행과 추적성**: `current_status.md`에 다음 Action과 판정 기준이 명확하면 재확인 없이 실행합니다. 각 논리 단위의 결과·명령·로그/산출물 경로·판단·후속 Action을 SSOT에 기록하고 Git 커밋하여 다른 작업공간이 즉시 이어받을 수 있게 합니다.


---

## 2026-08-09 재편 (CV-only 전환)

- `current_architecture.md` / `current_experiments.md`를 **CV-only(v40~) 기준으로 전면
  재작성**했다. 이전 판(v34~v39 6-분기)의 핵심은 `history.md` §12~§16에 요약.
- v36 Q1 / v37 proposal은 **둘 다 기각**(§65). 현재 활성 proposal은 **없다** — 다음 노선(learnable 사영)이 정해지면 새로 작성한다.
- 판정 기준이 **SEAL 10개 task macro 평균**으로 바뀌었다(§71-4). `seal_univ2_baseline_17tasks.csv`의
  `in_seal=yes` 행이 대상이고, 러너는 `scripts/eval_seal_tasks.sh`다.
- `docs/history/` 폴더의 개별 문서를 **`docs/history.md` 단일 파일로 통합**하고 폴더를 제거했다(원문은 git 이력 보존).
