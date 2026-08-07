# Agent handoff guide

**Last updated**: `2026-08-07` — v34-1536 확정(PathoBench 보고용). 공식 Patho-Bench 프로토콜(공식 fold 50-fold·코호트·라벨) 평가 재시작(§56, config 회귀 해결 — 5/17 완료 → 12개 백그라운드). config 시스템을 v34 base + group default 참조형으로 리팩터링 + v30/v24/v22 체인 자체 포함형 아카이빙(§56). 로컬 ccrcc CSV 오류 정정(§51), SEAL baseline 비교(§52).

**Confirmed baseline**: v30 = v24 residual+bottleneck bag projection + B1
`bag_representation: poolz_l2` + B2 log-uniform cardinality `[1,1024]`. Musk zero-shot
`0.8539`, 기존 대형 합성 분포 `0.9483`; 상세는 [`current_status.md`](current_status.md)
§29·§28이다. 코드 기본 `bag_representation`은 checkpoint/config의 조용한 의미 변경을 막기
위해 계속 `legacy`다.

**확정 — v34 large-context + 아키텍처 효율화 (2026-08-07)**: PathoBench 규모(3k~30k+ 타일)
컨텍스트 학습을 위한 MLA 계열 효율화를 커밋·적용했다. ① `src/models/mla.py` standalone
MLA(`bfaee6a`), ② aggregator **slot MLA 저랭크 affinity** (`aggregator_slot_latent_dim`/
`slot_query_latent_dim`/`slot_affinity_dim` + `slot_w_dq/dkv/uq/uk`, `e98b3e2` — None이면
full-dim dot과 byte-identical, 파라미터 0), ③ **slot_std 분산 트릭**(`17a1c36`, [cells,slots,dim]
텐서 제거, default 경로 byte-identical), ④ **배치 population candidates**
(`_population_candidates_batched`, `7700e85` — 수치 동일, **훈련 전용**), ⑤ **정규화 통합**
(`_instances_are_unit`, `778b40b` — 수치 동일). eval은 항상 per-bag 루프(배치 경로의
[C,max_cells,1536] 패딩 OOM 방지, `000aead`). config: `train_v34_phase0_largectx_512.yaml`
([1,32768]) / `..._1536.yaml`(1536-d, [1,8192]), 둘 다 scratch + slot MLA. **v34-1536
(1024ep×50, batch=4, fp32) 완주** — best val_ce 0.4419
(`checkpoints/20260806_215800/v34_phase0_largectx_1536/epoch=048-...`).

평가(§50): ① PathoBench **17개 binary task 5-fold CV 평균 pooled 0.843** (LUAD/LSCC 유전체
task 0.91~0.99 강세). ⚠️ **§51 정정: 로컬 `cptac_ccrcc_{er,grade,her2}` CSV가 `bc_therapy`
복사본으로 확정 — 실제 CPTAC-CCRCC 코호트는 미평가, 실측 6개 데이터셋·14개 유효 task.** ② Musk — `test_musk.py`가 config
input_dim 동적 패딩 + `--pad-mode`(zero/tile, `4aca7f1`/`6d4c5bc`): **tile(166×9+42) 0.858**
vs zero-pad 0.822 (v30 0.854와 동등). ③ **ICI 실세계 5-seed 0.512±0.027 = 랜덤** (명시적
잠금 해제, `f8181be`: `ICIDataset` input_dim/pad_mode 타일 + `test_v34_phase0_largectx_1536_ici.yaml`).
v30과 CV 직접 비교는 **PCA-per-fold 미지원으로 보류**. 상세 §49·§50.

**v34 확정 (§53·§56, 2026-08-07)**: 사용자 결정으로 **v34-1536을 PathoBench 보고용 모델로 확정**.
평가는 **공식 Patho-Bench 프로토콜**(공식 k=all.tsv의 50-fold, 공식 코호트 245장, 공식 라벨
`config.yaml` task_col)로 진행 — **5/17 완료(pooled)**: bc_therapy er 0.672/grade 0.713/her2 0.670,
cptac_brca_PIK3CA 0.569, brca_TP53. **12개는 §56 config 수정으로 재시작**(백그라운드,
`scripts/run_official50_batch.sh`). 이전 배치가 아카이빙된 `train_v24_musklike_easy.yaml`을
참조해 전부 실패했던 회귀를 v34 config 자체 포함/default 참조화로 해결. v30은 합성/Musk baseline
유지. SEAL(지도 ABMIL/MeanMIL)과는
프로토콜(지도 vs zero-shot in-context)·코호트(ccrcc 218 vs 245) 차이 명시. 상세 §52·§53·§56.

**Rejected candidate — architecture v31 CCER-v2**: projection 전 aligned slot-center로
support class prototype을 만들고, 기존 rare branch와 독립인 support/query encoder에서
class-centered cell evidence를 계산한다. `Top-1`, `Top-4`, `mean` route는 총 `0.30`의
floor를 가지며 별도 null gate는 없다. 최종 output head는 zero-init이므로 v30 weight-only
초기화 직후 logits가 정확히 동일하다. 신규 module은 base LR, 공통 v30 backbone은 `0.05x`
LR을 사용한다. Config는 `configs/train_v31_ccer_v2.yaml`, architecture marker는 `31`이다.
Seed 42 20-epoch 학습 best는 epoch 18 `val_ce_loss=0.443786`이었으나 synthetic AUROC
`0.8514`, Musk `0.8470`으로 v30 Musk `0.8539`를 넘지 못했고 대형 bag은 `0.698`로
동일했다. 따라서 미채택이며 재현용 코드만 보존한다. 상세는
[`current_status.md`](current_status.md) §35·§36이다. 현재 활성 v31 학습은 없다.

**Proposed next investigation — v32 DR-CCER**: CCER-v2 예측은 v30과 synthetic 상관
`0.99928`, Musk 상관 `0.99311`이고 Musk `n>34`가 `0.69841`로 완전히 동일했다. 따라서
단순 slot/Top-K 확대 대신 donor-resolved support evidence와 independently supervised expert,
reliability-gated mixture를 제안한다. 구현 전 P0–P2 checkpoint 진단이 필수다. 상세는
[`history/architecture_v32_dr_ccer_proposal.md`](history/architecture_v32_dr_ccer_proposal.md)와
[`current_status.md`](current_status.md) §37이다. 아직 구현·학습 승인 또는 활성 run은 없다.

**Active — v32b DR-CCER (2026-08-05)**: v32 원안의 비판적 재검토 개선안
([`history/architecture_v32b_dr_ccer_proposal.md`](history/architecture_v32b_dr_ccer_proposal.md))을 작성하고,
P0–P3 probe(`scripts/archive/probes_smoke/probe_v32_headroom.py`) + DR-CCER 아키텍처(`architecture_version=32`,
donor-resolved expert + reliability-gated convex mixture)를 구현했다. **결과: CCER 계열 실증적
폐기** — ① Stage A 학습(`20260805_182126`, 10 epochs)에서 donor-resolved expert standalone CE가
0.693(무작위) 정체, ② Stage-0 probe에서 P2 fusion headroom **-0.00034**, P3 donor-agreement
headroom **+0.00000** (둘 다 게이트 +0.005 미달), CCER-v2 standalone branch AUROC 0.51(무작위,
v30과 corr 0.0096). 따라서 v32 미채택, 재현 코드만 보존, v30 baseline 유지. 상세는
[`current_status.md`](current_status.md) §38이다. **다음 방향**: 데이터 측 — Phase 1 "v30 on
6-task mix"(any_positive_sparse 포함) 재학습, 소형 bag(n≤4)·n>34 분포 레버. 새 세션은 §38부터 읽을 것.

**Active — v33 Phase 0 (2026-08-05)**: v33 MR-BagPFN proposal의 §9 지침대로 **arm B(v30 +
six-task + B2)와 arm C(v30 + legacy + B2b) 데이터 컨트롤을 먼저 구현·런칭**했다. B2b는
`SyntheticManifoldGenerator(per_bag_cardinality=True)`로 에피소드 내 per-bag
`n_b ~ LogUniform[1,1024]`을 추첨해 ragged list-of-bags를 반환하는 새 데이터 경로다
(collator/training_step ragged 분기, `episode_batch_size=1` 필요). config:
`configs/train_v33_phase0_armB.yaml`·`armC.yaml`. 신규 테스트 `tests/test_b2b.py` 10개 포함
기본 suite는 compact화 후 **19 tests / 약 143초**다. 학습: arm B는
`logs/20260805_220642/`에서 BF16으로 기존 checkpoint를 복원해 계속 진행한다. 최초 arm C
`logs/20260805_214751/`는 batch 1에서 4096 updates/epoch가 되어 중단했다. 해결 run은
`logs/20260805_220843/`: train episode를 512/epoch로 줄여 arm B와 동일한
**512 optimizer updates/epoch**를 사용하며, ragged B2b와 v30 architecture는 유지한다
(실측 약 3분/epoch, 전체 약 2.5–3시간). **Phase 1 frozen-v30 multi-resolution probe는
Phase 0 결과 확인 후에만 구현한다.** 상세는
[`current_status.md`](current_status.md) §41·§39다.

**Active — arm C top-up, 8×A6000 DDP (2026-08-06)**: §42에서 arm C가 과소학습 편향
(에피소드 8× 부족)으로 gate 미달이었으므로, 사용자 결정으로 **8×RTX A6000 DDP +
에피소드-매치**로 재개했다. 새 config는 `configs/train_v33_phase0_armC_ddp8.yaml`
(자체 포함형, medium 체인 미상속, `episodes_per_epoch: 4096`, devices 8 /
`ddp_find_unused_parameters_false` / bf16-mixed / max_epochs 150)이고 `archive/v33_phase0_armC_bf16/last.ckpt`에서
resume한다. **gnode5 필수**: 이 머신의 NCCL P2P/CUMEM 전송이 hang을 일으켜
`scripts/launch_interactive_training.sh`에 `NCCL_P2P_DISABLE=1`을 기본 적용했다
(진단용 `scripts/archive/probes_smoke/nccl_probe.py` 신규). B200 1장 대비 A6000 1장은 ~1.8× 느리지만
8장 병렬로 노드 총 처리량은 ~4.3× (상세 표는 [`current_status.md`](current_status.md) §43).
**다음**: top-up 완주(150 epoch) 후 §42 재평가.

**Proposed next investigation — v33 MR-BagPFN (아키텍처)**: CCER와 다른 새 cell evidence를
만들지 않고, 검증된 v30 bag representation을 동일 bag의 full/partition/subsample view에서
공유해 sampling resolution 정보를 보존한다. 단, Phase 0(arm B/C) 결과와 frozen-v30
multi-resolution combiner의 paired AUROC `+0.01` headroom 확인 후에만 구현한다. 상세는
[`history/architecture_v33_multiresolution_bag_proposal.md`](history/architecture_v33_multiresolution_bag_proposal.md)와
[`current_status.md`](current_status.md) §39다.

**Persistent invariants**: ICI는 사용자 지시로 잠금 상태다. 잠금 해제 시
`src/datasets/base_data.py`의 cell-axis zero-padding이 bag mean/global spread를 오염하는 문제를
먼저 처리한다. 이전 v24/v25/v26/IA-MIL 결정과 config 복구 기록은 §25~§29 및
[`history/archive.md`](history/archive.md)에 보존한다.

이 문서는 BagPFN 저장소를 처음 맡은 coding agent가 안전하게 작업을 시작하기 위한 운영 및 핸드오프 지침입니다. 최신 개발 및 실험 진행 상황은 [`current_status.md`](current_status.md), 현재 모델 명세는 [`current_architecture.md`](current_architecture.md), 현재 실험 프로토콜은 [`current_experiments.md`](current_experiments.md)를 참고합니다.

---

## 1. 새 세션 접속 Agent의 최우선 정독 및 Git 파악 원칙 (New Session Protocol)

> [!IMPORTANT]
> **새 대화 세션 시작 시 Agent 초기화 행동 수칙**:
> 1. 사용자는 매번 **새 대화 세션(New Chat Session)**으로 접속합니다.
> 2. 접속한 AI Coding Agent는 세션 간 맥락 단절을 방지하기 위해 **`docs/` 최상위 루트의 Living `.md` 파일 5개와 현행 `architecture_*_proposal.md` 1개를 최우선으로 즉시 정독**합니다.
> 3. Living 문서 정독 직후, **반드시 Git 상태 및 최신 커밋 내역/Diff를 확인**하여 이전 세션의 정밀 코드 변경점과 작업 히스토리를 파악합니다:
>    ```bash
>    GIT_OPTIONAL_LOCKS=0 timeout 3s git status -uno
>    GIT_OPTIONAL_LOCKS=0 timeout 3s git --no-pager log -n 5 --stat
>    GIT_OPTIONAL_LOCKS=0 timeout 3s git --no-pager diff HEAD~1 HEAD
>    ```
> 4. Living 문서 5개, 현행 proposal, Git commit log/diff를 종합하여 확정 baseline, 코드 수정 내역, 완료된 실험 수치, 미결 과제 및 다음 Action Plan을 이어받아야 합니다.

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
     timeout 300s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m unittest discover -s tests -p "test_*.py"
     ```
   - 기본 스위트는 현재 **19 tests, 약 143초**다. 폐기 architecture/연구 진단 175개는
     `tests/history/legacy_*.py`로 이관되어 기본 discovery에서 실행되지 않는다. archive suite는
     수정 대상이 해당 보존 경로일 때만 개별 실행한다.

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

1. **`docs/` 최상위 루트 규칙 (Active Living Docs + Current Proposal)**:
   - `docs/` 최상위 루트에는 새 Agent가 즉시 정독해야 하는 **핵심 Living 문서 5개와 현행 proposal 1개만 존재**해야 합니다:
     - [`agent_handoff.md`](agent_handoff.md): 운영 규칙, 바이너리 경로, Git 수칙, Docs/Config 관리 지침
     - [`current_status.md`](current_status.md): 개발 현황, 최신 수치, Git 커밋 이력, 이슈 진단 및 Action Plan (SSOT)
     - [`current_architecture.md`](current_architecture.md): Architecture v22 수학적 기술 명세 (retrieval 없음)
     - [`current_experiments.md`](current_experiments.md): 실험 전략(합성=결정 / ICI=최종 테스트), 검정력, 평가 프로토콜, Stage 1~3 실행 명령어
     - [`README.md`](README.md): 전체 문서 맵 및 갱신 규칙
     - `architecture_*_proposal.md`: 현재 활성 개선안 1개. 완료·폐기 시 `history/`로 이동
   - 최상위 Living 문서와 현행 proposal은 항상 서로 일관된 맥락을 유지합니다.

2. **`docs/history/` 하위 아카이빙 규칙 (Historical & Deep-Dive Docs)**:
   - 특정 시점의 딥다이브 분석서, 옛 버전 아키텍처 설계안, 과거 벤치마크 플랜(예: `v20_scalability_plan.md`, `retrieval_architecture_analysis.md`, `architecture_v18.md` 등)은 **모두 `docs/history/` 하위 폴더로 이동하여 보관**합니다.

---

## 7. Config 관리 및 아카이빙 규칙 (Config Organization Rules)

1. **`configs/` 최상위 루트 유지 조건**:
   - 현재 활성 파이프라인에서 직접 사용하는 entry point config만 `configs/` 최상위에 유지합니다.
   - 현재 `configs/` 최상위 유지 대상: **v34 전용 — `train_v34_phase0_largectx_1536.yaml`
     (PathoBench 보고용, 자체 포함형)**, `train_v34_phase0_largectx_512.yaml` (arm D),
     `test_v34_phase0_largectx_1536_ici.yaml` (평가). **v34 config는 `base_config` 없이 전체
     base 체인(v30→v24→v22→v18_v19)과 named group을 인라인한 자체 포함형**이라 아카이브와
     무관하게 단독 실행됩니다 (2026-08-07 §56).
   - v30/v24/v22/eval_v30 체인은 `configs/archive/v30/`·`archive/v24/`·`archive/v22/`로 이관
     (2026-08-07 §56 재아카이빙, base_config 상대경로는 `../` 로 보정해 재현 가능).
   - 폐기 확정 config 이관: v23-A0/v24-A0/v24-B0(`train_v23_medium_bag_mean.yaml`,
     `train_v24_medium_bag_proj.yaml`, `train_v24_medium_bag_proj_bottleneck.yaml`) →
     `configs/archive/v23_v24_candidates/`; v25(`train_v25_medium_typed_bag.yaml`,
     `train_v25_easy.yaml`) → `configs/archive/v25_typed_bag/`.
   - ICI의 fold/seed는 config에 박지 않고 `--cv` / `--seed`로 주입합니다 (`scripts/launch_ici_protocol.sh`).
2. **구버전 Config 아카이빙 조건**:
   - 구버전 아키텍처의 config는 `configs/archive/` 하위로 즉시 이관합니다: `archive/v18_v19/`, `archive/v20/`, `archive/v21_retrieval/`.
   - 폐기된 기능의 실행 스크립트도 같은 규칙으로 `scripts/archive/`(예: `scripts/archive/v21_retrieval/`)로 옮깁니다.
3. **아카이빙 config는 자체 포함형(인라인)으로 보관 (2026-08-07 신설)**:
   - 아카이빙하는 config는 `base_config`를 남기지 않고 **전부 인라인**으로 변환해 보관합니다 (v34가
     인라인한 방식처럼 `data`/`model`/`optimizer`/`scheduler`/`trainer`/`logger`/`callbacks` 전체 값을
     직접 기술). 이렇게 하면 아카이빙 후에도 자기 디렉터리 기준 상대경로가 깨질 일이 없고, 참조 검증
     없이 항상 재현 가능합니다.
   - 2026-08-07 §56 적용: v34 자체 포함형 전환 + v30/v24/v22 체인 재아카이빙(141개 config 전부 해석
     성공). 이후 아카이빙은 base 체인 config를 root에서 인라인 후 보관.

   > [!IMPORTANT]
   > **아카이빙·삭제 시 참조 검증 필수 (2026-08-04 신설).** `base_config`는
   > `utils.py`의 `_load_train_config`가 **config 자기 디렉터리 기준**으로 해석하므로, config를 하위
   > 폴더로 옮기면 상대경로가 조용히 깨집니다. 또 `resolve_config_group`은 모듈 조각을
   > `configs/<group>/`에서만 찾으므로, 모듈 config를 삭제하면 이를 참조하는 **아카이브** config가
   > 깨집니다. 실제로 2026-08-04까지 **모든** 아카이빙 커밋이 이 검증을 누락해 config 18개가
   > 로드 불가 상태였고 unittest 1건이 상시 실패했습니다 (복구 기록: [`current_status.md`](current_status.md) §26).
   > **아카이빙/삭제 커밋 전 반드시 아래를 통과시킬 것** (활성 config만이 아니라 **전체**):
   > ```bash
   > timeout 300s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -c "
   > import sys; sys.path.insert(0,'.')
   > from pathlib import Path
   > from src.utils.utils import merge_train_config
   > bad=[]
   > for p in sorted(Path('configs').rglob('*.yaml')):
   >     if 'base_config' not in p.read_text(): continue
   >     try: merge_train_config(p)
   >     except Exception as e: bad.append((p, e))
   > print('failing:', len(bad)); [print(' ', p, e) for p, e in bad]"
   > ```
4. **모듈형 Component 설정 분리**:
   - `callbacks/`, `data/`, `logger/`, `model/`, `optimizer/`, `scheduler/`, `trainer/` 등 모듈 조각은 해당 서브폴더에 구성합니다.
