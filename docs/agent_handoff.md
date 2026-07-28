# Agent handoff guide

**Last updated**: `2026-07-28 10:45:00 KST`  
**Architecture Version**: `22` (`architecture_version = 22`)

이 문서는 BagPFN 저장소를 처음 맡은 coding agent가 안전하게 작업을 시작하기 위한 운영 및 핸드오프 지침입니다. 최신 개발 및 실험 진행 상황은 [`current_status.md`](current_status.md), 현재 모델 명세는 [`current_architecture.md`](current_architecture.md), 현재 실험 프로토콜은 [`current_experiments.md`](current_experiments.md)를 참고합니다.

---

## 1. 새 세션 접속 Agent의 최우선 정독 및 Git 파악 원칙 (New Session Protocol)

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 행동 수칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 접속한 AI Coding Agent는 세션 간 맥락 단절을 방지하기 위해 **`docs/` 최상위 루트의 Living `.md` 파일 5개만 최우선으로 즉시 정독**합니다.
> 3. Living 문서 정독 직후, **반드시 Git 상태 및 최신 커밋 내역/Diff를 확인**하여 이전 세션의 정밀 코드 변경점과 작업 히스토리를 파악합니다:
>    ```bash
>    git status
>    git log -n 5 --stat
>    git diff HEAD~1 HEAD
>    ```
> 4. Living 문서 5개와 Git commit log/diff를 종합하여 현재 아키텍처 버전(v22), 코드 수정 내역, 완료된 실험 수치, 미결 과제 및 다음 Action Plan을 100% 동일한 맥락으로 완벽히 이어받아야 합니다.

---

## 2. Git 중심 개발 및 세션 핸드오프 수칙 (Git-Centric Workflow)

1. **잦은 커밋 (Frequent Commits)**:
   - 논리 단위 작업(기능 추가, 버그 수정, 문서 개정, config 정돈, 단위 테스트 작성 등)이 완료될 때마다 즉시 커밋을 수행하여 작업 이력을 세분화합니다.
2. **상세한 커밋 메시지 작성 (Detailed Commit Messages)**:
   - 커밋 메시지는 제목(Subject)과 상세 본문(Body)을 명확히 구분하여 작성합니다:
     - `feat`: 신규 모델 아키텍처, 텐서 연산, Feature Retrieval 기능 구현
     - `docs`: Living 문서 개정, 아키텍처 스펙 문서화, 작업 수칙 업데이트
     - `chore`: 디렉터리 아카이빙, config 정돈, 환경 파일 설정
     - `test`: 단위 테스트 수트 작성 및 검증
   - 본문(Body)에는 **변경 동기(Why)**, **구현 세부사항(What)**, **검증 결과(Verification)**를 정밀하게 명시합니다.
3. **세션 종료 및 핸드오프 시 커밋 필수**:
   - 턴이나 대화 세션을 마무리하기 전 Working Tree의 모든 변경 사항을 남김없이 커밋하고, 생성된 Commit Hash와 핵심 요약을 [`current_status.md`](current_status.md)에 갱신하여 바톤 터치합니다.

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
     /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
     ```

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
     - [`current_experiments.md`](current_experiments.md): Phase 1~5 실험 프로토콜 및 실증 성과 수치
     - [`README.md`](README.md): 전체 문서 맵 및 갱신 규칙
   - 최상위 Living 문서 5개는 항상 서로 100% 일관된 맥락과 동일한 아키텍처 버전(v22)을 유지합니다.

2. **`docs/history/` 하위 아카이빙 규칙 (Historical & Deep-Dive Docs)**:
   - 특정 시점의 딥다이브 분석서, 옛 버전 아키텍처 설계안, 과거 벤치마크 플랜(예: `v20_scalability_plan.md`, `retrieval_architecture_analysis.md`, `architecture_v18.md` 등)은 **모두 `docs/history/` 하위 폴더로 이동하여 보관**합니다.

---

## 7. Config 관리 및 아카이빙 규칙 (Config Organization Rules)

1. **`configs/` 최상위 루트 유지 조건**:
   - 현재 활성 파이프라인에서 직접 사용하는 **Architecture v22 entry point config만 `configs/` 최상위에 유지**합니다 (10개 내외).
   - 예: `train_v22_medium.yaml`, `train_v22_hard_realworld.yaml`, `train_v22_ici_finetune_fold0.yaml`~`fold4.yaml` 등.
2. **구버전 Config 아카이빙 조건**:
   - 구버전 아키텍처(v18, v19, v20 등)의 config는 `configs/archive/v18_v19/`, `configs/archive/v20/` 하위 폴더로 즉시 이관합니다.
3. **모듈형 Component 설정 분리**:
   - `callbacks/`, `data/`, `logger/`, `model/`, `optimizer/`, `scheduler/`, `trainer/` 등 모듈 조각은 해당 서브폴더에 구성합니다.
