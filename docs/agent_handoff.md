# Coding agent handoff

이 문서는 BagPFN 저장소를 처음 맡은 coding agent가 안전하게 작업을 시작하기 위한 운영 지침이다. 최신 상태는 [`current_status.md`](current_status.md), 현재 모델은 [`current_architecture.md`](current_architecture.md), 현재 실험 protocol은 [`current_experiments.md`](current_experiments.md)를 읽는다. 과거 설계 및 실험은 [`history/`](history/)에 있으며 현재 실행 기준으로 사용하지 않는다.

## 1. 처음 할 일

저장소 전체를 무작정 탐색하거나 긴 로그를 출력하지 않는다. 다음 순서로 필요한 범위만 확인한다.

1. `git status -sb`로 branch와 사용자 변경사항을 확인한다.
2. 이 문서와 `docs/current_status.md`, `docs/current_architecture.md`, `docs/current_experiments.md`를 읽는다.
3. 작업에 필요한 config를 `--print-config`로 resolve한다.
4. 관련 source와 test만 검색한다.
5. 실행 전 GPU 점유 상태와 기존 프로세스를 확인한다.
6. 변경 후 targeted test, 전체 test, full-size BF16 smoke를 위험도에 맞게 수행한다.

현재 기준 저장소 위치와 환경:

```text
repository: /NHNHOME/kimds/ICF
python: /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python
torchrun: /NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun
hardware: NVIDIA B200 1 GPU
```

## 2. 프로젝트 개요

BagPFN은 labelled context bag과 unlabelled query bag으로 구성된 episode를 처리하는 multiple-instance episodic meta-classifier다. 각 bag은 여러 instance의 집합이다. Query label은 loss와 metric 계산에만 사용하며 representation, normalization, class memory, ridge 또는 covariance subspace fitting에 사용하면 안 된다.

Medium synthetic pretraining은 CUDA에서 episode를 online 생성하므로 외부 ICI 데이터가 필요 없다. ICI 평가는 별도 `data/`가 필요한 후속 단계다.

용어 규칙: 모든 신규 코드 주석과 문서에서는 개별 관측 단위를 **instance**라고 부른다.

## 3. 현재 architecture 불변 조건

현재 모델은 architecture version 19다.

- bag마다 raw mean을 제거한 `centered_delta`와 normalized `centered_x`를 사용한다.
- `global_spread`가 translation-invariant global summary다.
- raw bag mean과 raw instance coordinate는 classification 우회 경로로 사용하지 않는다.
- bag당 token 수는 `1 global spread + 36 slot + 3 tail = 40`을 유지한다.
- outer episode batch shape를 보존한다.
- context에서만 class memory, ridge, normalization과 CSP 통계를 계산한다.
- v18 또는 version metadata가 없는 checkpoint는 v19에 load하지 않는다.
- architecture/checkpoint compatibility 검사를 제거하지 않는다.

현재 final logit의 개념적 구조:

```text
base = global_shape
     + population_scale * population
     + tail_scale * tail
     + fusion_scale * interaction

final = base
      + covariance_residual_scale * covariance_ridge
      + covariance_relation_residual_scale * covariance_relation
```

정확한 최신 파라미터와 성능 상태는 `current_status.md`를 기준으로 한다.

## 4. 핵심 코드 위치

- `src/models/baseline.py`
  - `StructuredEpisodePopulationAggregator`: centered representation, slot/tail token, covariance matrix
  - `StructuredPopulationMetaClassifier`: episode classifier branches, covariance relation, final fusion
  - `BaseModel`: config wiring과 architecture version
- `src/datasets/synthetic_data.py`: online synthetic episode와 task metadata
- `src/modules/data_interface.py`: dataset config 전달
- `src/modules/model_interface.py`: loss, overall/task별 validation metric, diagnostic aggregation
- `configs/train_v19_medium.yaml`: 현재 medium production 진입 config
- `scripts/train.py`: config resolve와 학습 entrypoint
- `scripts/launch_interactive_training.sh`: persistent log/checkpoint를 사용하는 managed foreground launcher
- `tests/test_base_model.py`: architecture, invariance, batch와 covariance relation test

## 5. 변경 금지 및 안전 규칙

- loss, architecture, synthetic difficulty를 환경 문제 해결을 이유로 임의 변경하지 않는다.
- 기존 checkpoint를 새 run에 자동 resume하지 않는다.
- 기존 사용자 파일과 checkpoint를 삭제하거나 덮어쓰지 않는다.
- FP16 LR `1e-3` 실패 run은 재사용하지 않는다. GradScaler 붕괴와 NaN 이력이 있다.
- secret을 config, 코드 또는 로그에 출력하지 않는다. W&B는 `/NHNHOME/kimds/.netrc`를 `NETRC`로 지정한다.
- `NCCL_P2P_DISABLE=1`은 실제 NCCL hang이 재현될 때만 사용한다.
- B200 1장에서는 `devices=1`, `NPROC_PER_NODE=1`, single-process strategy를 사용한다.
- managed cloud job은 detached/nohup 대신 `ICF_FOREGROUND=1`을 사용한다.
- 긴 로그 전체를 읽지 말고 `rg`, 제한된 `tail`, W&B history의 필요한 key만 사용한다.

## 6. 실행 전 검증

GPU 확인:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Resolved config 확인:

```bash
/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python \
  scripts/train.py --config configs/train_v19_medium.yaml --print-config
```

최소 확인 항목:

- `ckpt_path: null` for a new run
- AdamW와 의도한 LR
- BF16 mixed precision
- gradient clipping 1.0
- 5-epoch warm-up과 `val_ce_loss` plateau scheduler
- `devices=1`, `strategy=auto`
- task sampling과 covariance relation config

## 7. 테스트

현재 환경의 전체 test 명령:

```bash
PYTHONPATH=/NHNHOME/kimds/ICF:/usr/local/lib/python3.12/dist-packages \
/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/python -m pytest -q
```

구조 변경 후 최소 보존 조건:

- query label 미사용
- context-only fitting과 normalization
- instance/context-bag permutation invariance와 label equivariance
- common bag shift invariance
- token 수와 outer-batch shape 유지
- logits, loss, gradients와 parameters finite
- global norm clipping과 optimizer update 확인
- residual scale 0에서 기존 경로와 동일
- v18 checkpoint load 거부

기존 smoke script가 다른 precision/config를 사용하면 현재 medium BF16 검증으로 간주하지 않는다.

## 8. 새 학습 실행 형식

실제 다음 실행 여부와 config는 먼저 `current_status.md`에서 확인한다. 새 run이 승인됐을 때의 형식은 다음과 같다.

```bash
cd /NHNHOME/kimds/ICF

ICF_FOREGROUND=1 \
ICF_RUN_TIME=<NEW_UNIQUE_RUN_ID> \
CUDA_DEVICES=0 \
NPROC_PER_NODE=1 \
TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun \
NETRC=/NHNHOME/kimds/.netrc \
scripts/launch_interactive_training.sh \
  <RUN_NAME> \
  <CONFIG_PATH>
```

새 architecture 실험은 기존 checkpoint를 resume하지 않는다. 로그와 checkpoint는 launcher가 만드는 persistent 경로를 사용한다. W&B credential이 없으면 `WANDB_MODE=offline`을 사용한다.

## 9. 학습 관찰

첫 validation과 warm-up 종료 후 다음을 확인한다.

- train/validation CE와 전체 accuracy, balanced accuracy, AUROC
- task별 CE와 AUROC
- branch별 logit std
- covariance relation diagnostic
- gradient norm, learning rate와 optimizer update
- NaN/Inf, OOM, eigendecomposition 또는 shape 오류

5-epoch warm-up 중 결과만으로 plateau를 판정하지 않는다. Checkpoint 선택 기준은 사전에 정한 `val_ce_loss`이며 task AUROC를 보고 사후 변경하지 않는다.

## 10. 문서 갱신 책임

작업이 끝나면 `current_status.md`에 다음만 갱신한다.

- 실행 중/완료 상태
- run URL과 artifact 경로
- best checkpoint와 선택 기준
- 핵심 결과와 판정
- 다음 작업

완료된 상세 실험 기록이 길어지면 `history/`에 별도 문서로 이동하고 `current_status.md`에는 링크와 결론만 남긴다.
