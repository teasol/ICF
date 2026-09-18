평가 실행에서 **GPU가 노는 시간**을 줄여라. 수치는 1비트도 바뀌면 안 된다.

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
   이 기계의 GPU 4는 deepseek 서버가 168 GB를 이미 쓰고 있어 여유가 15 GB 안팎이다.
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
# 1. 수치가 같아야 한다 — 세 해시가 전부 "동일" 이어야 한다
.venv/bin/python scripts/analysis/numeric_fingerprint.py --check talks/reports/numeric_baseline.json

# 2. 테스트가 통과해야 한다
.venv/bin/python -m pytest tests/ -q

# 3. 실제로 빨라졌는지 같은 과제로 전후를 재라
```

3번은 `cptac_pda/SMAD4_mutation` 한 과제, fold 10개로 전후를 재고 **초/fold를 보고하라.**
빨라지지 않았으면 변경을 되돌리고 그렇게 보고하라. **느려졌는데 병합되는 것이 가장 나쁘다.**

보고에는 다음을 넣어라: 측정한 전송 시간과 연산 시간, 변경 전후 초/fold, 지문 3개의
일치 여부, 테스트 결과.
