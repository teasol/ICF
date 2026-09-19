# RU-99 — fold 적합 디바이스 상주화의 수치 동등성·비용 실측 (2026-09-19)

- 유형: `confirmatory` · 판정: **지지 · 채택 · 종료**
- 실행: Slurm `gnode5`(RTX A6000, 드라이버 595.91.07), job `156679`
- 원문: 대장 `docs/history/research_units_all.json`의 RU-99, 로그
  `slurm_outputs/2026-09-19/1608/ru99-156679.{out,err}`, 제출 스크립트
  `scripts/slurm/ru99_device_residency_ab.sbatch`

## 배경

`talks/reports/2026-09-18_foldfit_cost.md`가 fold 적합 1회를 `20~22`초로 재고 "절반 이상이
GPU 밖"이라고 지목했다. `scripts/evaluate_pure.py`는 bag 특징을 CPU에 둔 채 fold마다
`TrainingFreeClassifier`에 넘기고, `extract_bag_descriptor`가 호출마다
`bag.to(basis.device)`로 재전송한다. fold 50개는 같은 슬라이드를 공유하므로 이 전송이
반복된다.

## 개입 (호출부만)

`scripts/evaluate_pure.py`에서 bag을 과제 시작 시 GPU에 상주시키고, OOM이면 호스트 상주로
되돌린다. `--cpu-bags`로 옛 동작을 재현한다. **`src/models/`는 건드리지 않았다**
(`src_changed=0`).

## 결과 (SMAD4, fold 10, 같은 코드로 전후 대조)

| 항목 | 변경 전 (`--cpu-bags`) | 변경 후 (상주) |
|:---|---:|---:|
| fold-mean AUROC | `0.441699` | `0.441699` |
| 초/10 fold | `128` | `71` |
| 초/fold | `12.8` | `7.1` |

- `Δ fold-mean = +0.00000000`, `max|Δfold AUROC| = 0.000e+00` — **10 fold 전부 비트 동일.**
- 시간 **1.80배 감소**.

## 비용 외삽

- 350 fold 1 arm ≈ `0.69` GPU-hour (기존 `1.94`).
- **700 fold 2-arm paired ≈ `1.38` GPU-hour** — `PROJECT.md`의 `4` GPU-h 예산 안에 들어온다.
- 기준선 arm을 재사용하면 후보 하나당 `0.69` GPU-h.

## 한계

- 과제 하나(SMAD4) fold 10에서만 전후 비교했다. 나머지 6과제·전체 fold는 외삽이다.
- 7-branch는 측정하지 않았다.
- 수치 동일성은 같은 기계·같은 디바이스 쌍에서만 주장한다.
- 적합 분산은 이 RU의 대상이 아니다(현행 파이프라인은 closed-form 결정론).

[작성자: opencode / Platform Agent / deepseek-v4.1-flash (effort: 미확인) · 2026-09-19 16:50 KST]
