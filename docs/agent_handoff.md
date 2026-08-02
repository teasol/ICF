# Agent handoff guide

**Last updated**: `2026-08-02 23:00:00 KST`
**Architecture Version**: **v24가 여전히 확정 baseline** — residual + bottleneck bag projection (구 v24-B1): `project_structured_tokens: true`, `projection_bottleneck_dim: 64`, `projection_residual_mean: true`. v22(구 기준선)/v23-A0/v24-A0/v24-B0는 폐기. Config: `configs/train_v24_medium_bag_proj_residual.yaml`. 상세 결정 근거: [`current_status.md`](current_status.md) §3 최종 결정. **v25(T5-A typed bag-preserving branch)는 2026-08-02 폐기 확정** (Medium/Easy 평가 모두 승격 기준 미달) — [`current_status.md`](current_status.md) §3/§11 참고. v25 config는 `configs/archive/v25_typed_bag/`, 태그 `v25-typed-bag-final`로 보존. **`26`(CLS-token pooling, `cls_token_pooling: true`)은 2026-08-02 구현 완료, scratch 학습 실행 중 — 아직 평가 전, v24를 대체하지 않음.** 같은 날 제안된 v26(EC-MoE)/v27(AC-ICAR)/v29(SP-SAT) 설계안은 학습 없는 게이트(E2/E7/A4)로 검토 후 전부 미구현 폐기, `docs/history/`로 이관 — [`current_status.md`](current_status.md) §16 참고.

이 문서는 BagPFN 저장소를 처음 맡은 coding agent가 안전하게 작업을 시작하기 위한 운영 및 핸드오프 지침입니다. 최신 개발 및 실험 진행 상황은 [`current_status.md`](current_status.md), 현재 모델 명세는 [`current_architecture.md`](current_architecture.md), 현재 실험 프로토콜은 [`current_experiments.md`](current_experiments.md)를 참고합니다.

---

## 1. 새 세션 접속 Agent의 최우선 정독 및 Git 파악 원칙 (New Session Protocol)

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 행동 수칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 접속한 AI Coding Agent는 세션 간 맥락 단절을 방지하기 위해 **`docs/` 최상위 루트의 Living `.md` 파일 5개만 최우선으로 즉시 정독**합니다.
> 3. Living 문서 정독 직후, **반드시 Git 상태 및 최신 커밋 내역/Diff를 확인**하여 이전 세션의 정밀 코드 변경점과 작업 히스토리를 파악합니다:
>    ```bash
>    timeout 3s git status -uno
>    timeout 3s git log -n 5 --stat
>    timeout 3s git diff HEAD~1 HEAD
>    ```
> 4. Living 문서 5개와 Git commit log/diff를 종합하여 v22 baseline과 활성 v23-A0/v24-A0/v24-B0 실험, 코드 수정 내역, 완료된 실험 수치, 미결 과제 및 다음 Action Plan을 100% 동일한 맥락으로 완벽히 이어받아야 합니다.

---

## 2. Git 중심 개발 및 세션 핸드오프 수칙 (Git-Centric Workflow)

0. **명확한 다음 단계는 자율적으로 연속 실행**:
   - [`current_status.md`](current_status.md)의 Action Plan과 판정 기준이 명확하면 사용자에게 “진행할까요?”라고 다시 묻지 않고 실행합니다.
   - 진단 결과가 사전 판정 기준을 만족해 다음 단계가 하나로 정해지는 경우, 구현·검증·후속 진단까지 같은 범위에서 계속 진행합니다.
   - 단, 모순되는 선택지, 파괴적 변경, 외부 공개/비용, 누락된 필수 입력처럼 새로운 사용자 판단이나 권한이 필요한 경우에는 중단하고 확인합니다.
1. **잦은 커밋 (Frequent Commits)**:
   - 논리 단위 작업(기능 추가, 버그 수정, 문서 개정, config 정돈, 단위 테스트 작성 등)이 완료될 때마다 즉시 커밋을 수행하여 작업 이력을 세분화합니다.
2. **상세한 커밋 메시지 작성 (Detailed Commit Messages)**:
   - 커밋 메시지는 제목(Subject)과 상세 본문(Body)을 명확히 구분하여 작성합니다:
     - `feat`: 신규 모델 아키텍처, 텐서 연산, 평가 프로토콜 기능 구현
     - `docs`: Living 문서 개정, 아키텍처 스펙 문서화, 작업 수칙 업데이트
     - `chore`: 디렉터리 아카이빙, config 정돈, 환경 파일 설정
     - `test`: 단위 테스트 수트 작성 및 검증
   - 본문(Body)에는 **변경 동기(Why)**, **구현 세부사항(What)**, **검증 결과(Verification)**를 정밀하게 명시합니다.
3. **세션 종료 및 핸드오프 시 커밋 필수**:
   - 턴이나 대화 세션을 마무리하기 전 Working Tree의 모든 변경 사항을 남김없이 커밋하고, 생성된 Commit Hash와 핵심 요약을 [`current_status.md`](current_status.md)에 갱신하여 바톤 터치합니다.
4. **진행상황 follow-up 가능성 보장**:
   - 각 논리 단위가 끝날 때 [`current_status.md`](current_status.md)에 상태, 핵심 수치, 실행 명령, 로그/PID/체크포인트/예측 파일 경로, 성공·중단 판단 근거, 바로 다음 Action을 기록합니다.
   - 실행법이나 모델 계약이 바뀌면 `current_experiments.md` 또는 `current_architecture.md`도 같은 논리 단위에서 함께 갱신합니다.
   - 장시간 작업은 완전 이탈형 백그라운드로 실행하고, 시작 직후 PID와 로그 경로를 기록하며, 완료 후 최종 결과와 산출물 경로를 추가합니다.
   - 다른 작업공간의 Agent가 대화 기록 없이 Living 문서와 `git log`만으로 작업을 이어갈 수 없는 상태는 완료된 핸드오프로 보지 않습니다.

---

## 3. 필수 작업 지침 & Multi-Location 동기화 규칙

1. **연구실 / 집 / 노트북 3원화 대화 동기화 완벽 대응**:
   - 세 장소 간 대화 히스토리가 비동기적이므로, 작업 진행 상황 및 수치/경로는 **반드시 [`current_status.md`](current_status.md)에 상세히 기록**하고 읽는다.
2. **명령어 Hang 타임아웃 필수 적용**:
   - NVML/드라이버/쉘 블로킹으로 인한 대화창 멈춤(Hang)을 방지하기 위해 터미널 조회가 필요한 모든 명령어에는 **`timeout 3s ps aux | grep python`** 또는 `timeout 3s tail -n 20 <LOG>`와 같이 반드시 타임아웃을 강제 적용한다.
3. **완전 이탈형 백그라운드 구동**:
   - 장시간 실행되는 훈련/평가 명령어는 **반드시 `scripts/launch_interactive_training.sh` 독립 백그라운드 스크립트**나 short `WaitMsBeforeAsync` 태스크로 띄운다.
4. **수치 안전성 계약**:
   - 공분산 스케치 역행렬 연산 시 FP16 계수 오버플로우 및 NaN 발생 방지를 위해 **`bf16-mixed` 정밀도를 필수 적용**한다.
5. **테스트 검증 필수**:
   - 코드를 변경한 뒤에는 아래 unittest 수트를 통과해야 완결로 인정한다:
     ```bash
     timeout 1500s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
     ```
   - 전체 스위트는 약 12분(현재 123 tests) 걸립니다. 타임아웃 없이 돌리면 §3-2 원칙에 어긋나고 hang 시 세션이 멈춥니다.

---

## 4. 작업 위치 및 바이너리 경로 명세

- **Workspace Root**: `/NHNHOME/kimds/ICF`
- **Python Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python`
- **Torchrun Binary**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun`
- **Netrc File**: `/NHNHOME/kimds/.netrc`
- **Target Hardware**: NVIDIA B200 GPU 1장 (`CUDA_VISIBLE_DEVICES=0`, 180GB VRAM)

---

## 5. 독립 실행 스크립트 표준 명령 구문

SSH 연결이나 VS Code 터미널이 종료되어도 백그라운드에서 지속해서 안정 구동되는 표준 실행 명령:

```bash
cd /NHNHOME/kimds/ICF

CUDA_DEVICES=0 \
NPROC_PER_NODE=1 \
TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun \
NETRC=/NHNHOME/kimds/.netrc \
scripts/launch_interactive_training.sh \
  <RUN_NAME> \
  <CONFIG_PATH>
```

훈련 시작 후 반드시 생성된 `logs/{RUN_TIME}/` 경로의 `.out` 로그 tail을 확인하여 정상 작동 여부를 정량적으로 검증하고 [`current_status.md`](current_status.md)를 즉시 갱신합니다.

---

## 6. Documentation 관리 및 아카이빙 규칙 (Docs Organization Rules)

1. **`docs/` 최상위 루트 규칙 (Active Living Docs Only)**:
   - `docs/` 최상위 루트에는 새 Agent가 즉시 정독해야 하는 **핵심 Living 문서 5개만 존재**해야 합니다:
     - [`agent_handoff.md`](agent_handoff.md): 운영 규칙, 바이너리 경로, Git 수칙, Docs/Config 관리 지침
     - [`current_status.md`](current_status.md): 개발 현황, 최신 수치, Git 커밋 이력, 이슈 진단 및 Action Plan (SSOT)
     - [`current_architecture.md`](current_architecture.md): Architecture v22 수학적 기술 명세 (retrieval 없음)
     - [`current_experiments.md`](current_experiments.md): 실험 전략(합성=결정 / ICI=최종 테스트), 검정력, 평가 프로토콜, Stage 1~3 실행 명령어
     - [`README.md`](README.md): 전체 문서 맵 및 갱신 규칙
   - 최상위 Living 문서 5개는 항상 서로 100% 일관된 맥락을 유지합니다. 현재는 v22 baseline과 조건부 v23-A0/v24-A0/v24-B0 ablation을 명확히 구분합니다.

2. **`docs/history/` 하위 아카이빙 규칙 (Historical & Deep-Dive Docs)**:
   - 특정 시점의 딥다이브 분석서, 옛 버전 아키텍처 설계안, 과거 벤치마크 플랜(예: `v20_scalability_plan.md`, `retrieval_architecture_analysis.md`, `architecture_v18.md` 등)은 **모두 `docs/history/` 하위 폴더로 이동하여 보관**합니다.

---

## 7. Config 관리 및 아카이빙 규칙 (Config Organization Rules)

1. **`configs/` 최상위 루트 유지 조건**:
   - 현재 활성 파이프라인에서 직접 사용하는 entry point config만 `configs/` 최상위에 유지합니다.
   - 현재 `configs/` 최상위 유지 대상: v24 확정(`train_v24_medium_bag_proj_residual.yaml`),
     v26 평가 중(`train_v26_medium_cls_token_pool.yaml` — CLS-token pooling, scratch 학습
     실행/평가 중, 승격/폐기 판정 전까지 유지),
     v22 기준선·참조용(`train_v22_medium.yaml` — `evaluate_synthetic.py` 기본 config,
     `train_v22_medium_context300.yaml`/`train_v22_hard_context300.yaml`/`train_v22_hard_realworld.yaml` — T4,
     `train_v22_ici_finetune.yaml`/`train_v22_ici_scratch.yaml` — ICI).
   - 폐기 확정 config 이관: v23-A0/v24-A0/v24-B0(`train_v23_medium_bag_mean.yaml`,
     `train_v24_medium_bag_proj.yaml`, `train_v24_medium_bag_proj_bottleneck.yaml`) →
     `configs/archive/v23_v24_candidates/`; v25(`train_v25_medium_typed_bag.yaml`,
     `train_v25_easy.yaml`) → `configs/archive/v25_typed_bag/`.
   - ICI의 fold/seed는 config에 박지 않고 `--cv` / `--seed`로 주입합니다 (`scripts/launch_ici_protocol.sh`).
2. **구버전 Config 아카이빙 조건**:
   - 구버전 아키텍처의 config는 `configs/archive/` 하위로 즉시 이관합니다: `archive/v18_v19/`, `archive/v20/`, `archive/v21_retrieval/`.
   - 폐기된 기능의 실행 스크립트도 같은 규칙으로 `scripts/archive/`(예: `scripts/archive/v21_retrieval/`)로 옮깁니다.
3. **모듈형 Component 설정 분리**:
   - `callbacks/`, `data/`, `logger/`, `model/`, `optimizer/`, `scheduler/`, `trainer/` 등 모듈 조각은 해당 서브폴더에 구성합니다.
