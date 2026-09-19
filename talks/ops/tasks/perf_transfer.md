평가 실행에서 **GPU가 노는 시간**을 줄여라. 수치는 1비트도 바뀌면 안 된다.

## 실행 환경 (먼저 확인 — 2026-09-19 정비)

이 작업은 **GPU가 없는 Slurm 로그인 노드 `nexgem`**에서 로컬 모델에 위임된다(`nvidia-smi` 없음).
따라서:

- 코드 편집·CPU 지문·테스트는 여기서 그대로 하면 된다.
- **GPU 계측(전송↔연산 분리, 초/fold)은 `sbatch`/`srun`으로 GPU 노드에 제출해야 한다.**
  로그인 노드에서 직접 `--device cuda`를 부르면 실패한다.
- 로그는 `slurm_outputs/YYYY-MM-DD/HHMM/`에 절대경로로 남기고, 제출 전 `mkdir -p` 한다.
- 노드는 `--nodelist`로 지정한다(`--gres=gpu:<type>`은 이 클러스터에서 동작하지 않는다).
  A5000은 `gnode1`\~`gnode4`, A6000은 `gnode5`.
- **`talks/reports/numeric_baseline.json`은 쓰지 마라.** 그 기준선은 NEXGEM(B200)에서
  생성돼 이 기계에서는 해시가 다르다(비트 비교 불가). 아래 합격 조건 1을 참고하라.

## 관측된 사실 (소집자가 실측)

`scripts/analysis/dump_branch_margins.py` 실행 중 측정한 값이다.

- 프로세스 CPU 사용률 **590~761%** (코어 6~7개)
- 같은 시각 GPU 가동률 샘플: `42 44 50 67 0 0 0 0 0 0` (0.5초 간격 10회)
- fold 하나에 **20.0초**가 걸리는데 그중 GPU가 일하는 시간은 절반이 안 된다

즉 계산이 무거운 게 아니라 **GPU 밖에서 시간을 쓰고 있다.**

## 짚이는 원인 (확인하고 시작하라 — 틀렸을 수 있다)

`scripts/evaluate_pure.py`와 `scripts/analysis/dump_branch_margins.py`는 슬라이드 특징을
CPU 텐서로 `bags` 딕셔너리에 올려 두고, fold마다 이렇게 부른다.

```python
clf.branch_margins([bags[s] for s in ctx_ids], ctx_lab, [bags[s] for s in test_ids])
```

fold 50개가 **대부분 같은 슬라이드를 공유**하는데 매번 호스트에서 디바이스로 다시 보내는
것으로 보인다. 슬라이드 하나가 `패치 수 × 1536` float32다.

**이 가설을 먼저 확인하라.** `torch.cuda.synchronize()`와 `time.perf_counter()`로
(a) 전송 시간 (b) 실제 연산 시간을 fold 몇 개에 대해 따로 재서 보고하라. 전송이 병목이
아니면 거기서 멈추고 무엇이 병목인지 보고하라. **추정으로 고치지 마라.**

## 고쳐도 되는 범위

전송이 병목으로 확인되면:

1. **호출부에서** 과제 시작 시 `bags`를 한 번 GPU로 올리고 fold 루프는 그것을 재사용한다.
   대상 파일은 `scripts/analysis/dump_branch_margins.py`와 `scripts/evaluate_pure.py`다.
2. GPU 메모리가 부족할 수 있다. **반드시 예외를 잡아 CPU 경로로 되돌아가라.**
   Slurm GPU 노드 메모리는 A5000 24 GB / A6000 48 GB다. 슬라이드 전체를 상주시키면
   넘칠 수 있으니 OOM이 나면 상주를 포기하고 CPU 경로로 복귀하라.
   OOM으로 실행이 죽으면 그것은 개선이 아니라 퇴행이다.
3. 그 밖에 명백히 낭비인 것(같은 값을 fold마다 다시 계산하는 것 등)이 보이면 보고하라.
   고치는 것은 1번까지만이다.

## 절대 하지 말 것

- **`src/models/` 안의 수치 계산을 바꾸지 마라.** 연산 순서, dtype, 누적 방식, 어떤 것도.
  device 이동만으로 해결되지 않으면 고치지 말고 보고하라.
- 근사, 저정밀도(fp16/bf16/TF32) 도입, 알고리즘 교체 금지. 같은 계산을 더 적게 기다리는
  것만 허용한다.
- 배치 크기나 fold 수를 바꾸지 마라.

## 합격 조건 (전부 통과해야 한다)

```bash
# 1-a. 변경 전 기준선을 "이 기계에서" 새로 만든다 (기존 NEXGEM 기준선은 비교 불가)
.venv/bin/python scripts/analysis/numeric_fingerprint.py \
    --write /home/kimds/ICF/scratch/perf_transfer_baseline.json
# ... 변경 ...
# 1-b. 같은 기계·같은 항목으로 대조한다. 모든 항목이 "동일"이어야 한다.
.venv/bin/python scripts/analysis/numeric_fingerprint.py \
    --check /home/kimds/ICF/scratch/perf_transfer_baseline.json

# 2. 테스트가 통과해야 한다 (CPU로 이 노드에서 가능)
.venv/bin/python -m pytest tests/ -q

# 3. 실제로 빨라졌는지 같은 과제로 전후를 재라 — GPU 노드에 제출해서
#    (예시; 실제 옵션은 dump_branch_margins.py --help로 확인)
mkdir -p slurm_outputs/$(date +%F)/$(date +%H%M)
srun --nodes=1 --nodelist=gnode5 --time=00:30:00 \
  .venv/bin/python scripts/analysis/dump_branch_margins.py \
    --config configs/baseline/v121_active.yaml \
    --tasks cptac_pda/SMAD4_mutation --folds 10 --out /home/kimds/ICF/scratch/perf_after.pt
```

3번은 `cptac_pda/SMAD4_mutation` 한 과제, fold 10개로 전후를 재고 **초/fold를 보고하라.**
빨라지지 않았으면 변경을 되돌리고 그렇게 보고하라. **느려졌는데 병합되는 것이 가장 나쁘다.**

지문을 `--write`한 기계와 `--check`하는 기계·디바이스가 같아야 한다. GPU 전송 경로를 바꿨다면
GPU 노드에서 전후 지문을 따로 재라(로그인 노드 CPU 해시만으로는 GPU 경로 변경을 못 잡는다).

보고에는 다음을 넣어라: 측정한 전송 시간과 연산 시간, 변경 전후 초/fold, 지문 항목과 일치 여부
(어느 파일과 대조했는지 명시), 테스트 결과.
