# BagPFN (ICF) — Single-Cell In-Context Meta-Classifier

세포 발현 표현(bag)들을 입력으로 받아 query bag의 이진 라벨(반응/비반응)을 예측하는
In-Context Meta-Learning 모델입니다. 합성 데이터로 meta-training 후 ICI(real-world)
데이터에 적용합니다.

## 현재 아키텍처 (v24)

**확정 (2026-08-01)**: residual + bottleneck bag projection. v22(retrieval 제거)에
bag 내부 40-token 구조화 요약을 1개 학습 projection token으로 압축하는 계층을 추가.
`architecture_version = 24`.

- Config: `configs/train_v24_medium_bag_proj_residual.yaml`
  (`project_structured_tokens: true`, `projection_bottleneck_dim: 64`, `projection_residual_mean: true`)
- 4대 수학 핵심 기술: ① Z-Score Bag Studentization ② Top-1% Sparse Evidence
  ③ Covariance Subspace Shrinkage(`subspace_shrinkage: 0.25`) ④ Auxiliary Pairwise Ranking Loss(`weight: 0.10`)
- Precision: **`bf16-mixed` 필수** (공분산 스케치 역행렬 NaN 방지)
- **진행 중**: v25(T5-A typed bag-preserving branch, 브랜치 `codex/v25-typed-bag`) +
  Easy tier 실험 — [`docs/current_status.md`](docs/current_status.md) §3/§11
- 상세 수학 명세: [`docs/current_architecture.md`](docs/current_architecture.md)

## 문서 맵 (Living Docs 5 + history)

| 문서 | 내용 |
|---|---|
| [`docs/agent_handoff.md`](docs/agent_handoff.md) | 운영 규칙, 바이너리 경로, Git/문서/config 수칙 |
| [`docs/current_status.md`](docs/current_status.md) | 개발 현황 **SSOT** — 최신 수치·커밋·Action Plan |
| [`docs/current_architecture.md`](docs/current_architecture.md) | Architecture v24 수학적 명세 |
| [`docs/current_experiments.md`](docs/current_experiments.md) | 실험 전략·검정력·평가 프로토콜 |
| [`docs/README.md`](docs/README.md) | 문서 갱신 규칙 |
| [`docs/history/`](docs/history/) | 아카이브된 구버전/딥다이브 문서 |

## 실행 환경

- **Workspace Root**: `/NHNHOME/kimds/ICF`
- **Python / Torchrun**: `/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/{python,torchrun}`
- **Hardware**: NVIDIA B200 GPU 1장 (`CUDA_VISIBLE_DEVICES=0`)
- ICI 데이터(`data/`)는 Git에 포함되지 않으며 저장소 루트에 위치:
  `ICI_CVOnly_scConcept_512/`, `ICI_GSE285888_scConcept_512.pt`, `ICI_GSE285888_scConcept_512_info.csv`

## 학습/평가 실행

표준 런처(`nohup + setsid` 완전 이탈형 백그라운드, SSH/터미널 종료와 무관하게 지속):

```bash
cd /NHNHOME/kimds/ICF

CUDA_DEVICES=0 \
NPROC_PER_NODE=1 \
TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun \
scripts/launch_interactive_training.sh \
  <RUN_NAME> <CONFIG_PATH>
```

- 학습 config: `configs/train_v24_medium_bag_proj_residual.yaml` (v24),
  `configs/train_v25_medium_typed_bag.yaml` (v25) — 학습 후 `logs/{RUN_TIME}/`의 `.out` 로그로 정량 검증
- 합성 평가: `scripts/evaluate_synthetic.py --config ... --val-episodes 1000`
- Paired 비교: `scripts/compare_predictions.py <pred_a.pt> <pred_b.pt>` (cluster bootstrap CI)
- ICI 평가: `scripts/launch_ici_protocol.sh`

## 테스트

코드 변경 후 반드시 전체 unittest 통과 (약 12분, 141 tests):

```bash
timeout 1500s /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python \
  -m unittest discover -s tests -p "test_*.py"
```

## Git 규칙

- 활성 브랜치: `codex/v25-typed-bag` (v25 작업 중, base: v24) / `main` = v24 확정 / `v22`·`v19` 참고용 보존
- 논리 단위마다 커밋 + 상세 메시지 (`feat`/`docs`/`chore`/`test`)
- 세션 종료 시 `current_status.md`에 결과·명령·경로 기록 후 커밋 (3원화 동기화 SSOT)
