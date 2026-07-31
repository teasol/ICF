# v23-A0 50-epoch session handoff (archived 2026-07-31)

Moved verbatim from `docs/current_status.md` §9 on 2026-07-31. Superseded: the v23-A0 50-epoch continuation finished (best epoch 43, see `current_status.md` §3). v23/v24 evaluation is pending in the new §9.

---

## 9. 2026-07-31 세션 핸드오프 — v23-A0 50-epoch 재개

### 이번 세션에서 확정한 것

- 원격 `main`을 v22 기준선 commit `5c31a4f`까지 fast-forward하고 push했습니다.
  v23 작업은 `codex/v23-bag-mean`에서 계속합니다.
- v23-A0는 bag당 40 structured token을 exact mean 1개로 줄이지만,
  전체 peak VRAM은 raw-cell CUDA generation과 bag encoder가 지배합니다.
  동일 worst-case Medium shape에서 v22와 v23의 batch 8 peak allocation은
  모두 약 `64.7 GiB`였습니다.
- CUDA generation/prefetch와 BF16 forward/backward를 포함한 worst-case
  (`100 bags × 1,000 cells`) batch 경계:

| Episode batch | Peak allocated | Peak reserved | 판정 |
|---:|---:|---:|---|
| 8 | 64.7 GiB | 80.2 GiB | 안전 |
| 12 | 97.1 GiB | 121.5 GiB | 통과 |
| 16 | 130.5 GiB | 160.4 GiB | 3-step 안정 통과 |
| 20 | 163.0 GiB | 180.3 GiB | allocator OOM 경고, 사용 금지 |
| 24 | OOM | OOM | 불가 |

메모리상 batch 16까지 가능하지만 batch를 바꾸면 epoch당 optimizer step이
`512→256`으로 줄고 기존 곡선과 최적화 조건이 달라집니다. 순수한 추가
수렴을 보기 위해 사용자의 결정대로 batch 8을 유지했습니다.

### 실행 중인 50-epoch continuation

- Config commit `4d784dc`: `episode_batch_size=8`,
  `shape_group_size=8`, `max_epochs=50`.
- Status commit `9d375a9`: resume PID/로그/체크포인트 경로 기록.
- Resume source:
  `checkpoints/20260731_155635/v23_medium_bag_mean/last.ckpt`
  (epoch 19 model/optimizer/scheduler/global-step 전체 복원).
- PID `2671747` 생존, GPU worker PID `2671863`, 확인 시 GPU 사용량
  `133,506 MiB`.
- stdout가 아니라 artifact로 확인: `version_1/metrics.csv`와 `last.ckpt`가
  `2026-07-31 17:15:17 KST`에 갱신됐고 epoch 25 validation까지 완료했습니다.
- 현재 continuation best는 epoch 21
  `val_ce_loss=0.5933271`, `val_auroc=0.7368156`입니다. 최초 20-epoch
  best `0.5933738`보다 CE가 `0.0000467` 낮아 사실상 동률입니다.
- Epoch 20~25 validation CE:
  `0.59614, 0.59333, 0.59559, 0.59519, 0.59413, 0.59609`.
- Run log:
  `logs/20260731_v23_bag_mean_50e_resume/v23_medium_bag_mean_50e.out`
- Metrics: `logs/v23_medium_bag_mean/version_1/metrics.csv`
- Checkpoints:
  `checkpoints/20260731_v23_bag_mean_50e_resume/v23_medium_bag_mean_50e/`

### 다음 Action

1. epoch 49까지 완료되는지 artifact mtime와 `last.ckpt`로 확인합니다.
2. 50-epoch 전체에서 minimum `val_ce_loss` checkpoint를 선택합니다.
3. 해당 checkpoint를 동일 pool-400, 1,000 episodes, context
   `40/80/160/300`에서 v22와 paired 평가합니다.
4. `scripts/compare_predictions.py`로 episode-cluster paired delta와 CI를
   계산합니다.
5. overall `+0.03` 또는 target task `+0.05`가 없으면 exact mean을
   폐기하고 typed learned bag encoder(T5-A)로 이동합니다.
6. 합성 Medium+Hard 후보가 확정되기 전까지 ICI는 실행하지 않습니다.
