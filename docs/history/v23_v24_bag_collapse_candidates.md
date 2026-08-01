# v23-A0 / v24-A0 / v24-B0 완료 기록 (archived 2026-08-02)

Moved verbatim from `docs/current_status.md` §3 on 2026-08-02. Superseded:
these three candidates were all discarded when v24-B1(residual + bottleneck
projection) was confirmed as v24 on 2026-08-01 without the originally
planned 4-way paired comparison — see `current_status.md` §3 "🏁 최종 결정
(2026-08-01)". Kept here only for the exact config/checkpoint/param-count
provenance of each discarded variant.

---

### ✅ v23-A0 exact bag-mean: 50-epoch 완료 (2026-07-31)

각 bag의 `1 global + 36 slot-statistic + 3 tail = 40` structured token을
exact arithmetic mean 1개로 압축했습니다. Context와 query 모두 같은
mean을 사용하고, class-memory에는 `bags × 40`이 아니라 bag당 1개가
입력됩니다. 기존 v22 동작은 기본값 `false`로 보존되며 활성화한
checkpoint는 `architecture_version=23`입니다.

| 항목 | 값 |
|---|---|
| Branch / implementation | `codex/v23-bag-mean`, commit `8edd7c1` |
| Config | `configs/train_v23_medium_bag_mean.yaml` |
| Run | `20260731_155635`, scratch Medium 20 epochs / 10,240 steps |
| Completion | launcher가 `training completed successfully` 기록; PID 종료 |
| Best checkpoint | `checkpoints/20260731_155635/v23_medium_bag_mean/epoch=019-val_ce_loss=0.5934.ckpt` |
| Last checkpoint | `checkpoints/20260731_155635/v23_medium_bag_mean/last.ckpt` |
| Best `val_ce_loss` | **0.5933738** @ epoch 19 |
| Epoch 19 val AUROC | 0.7322822 (104-episode training val; 최종 판정용 아님) |
| Highest run val AUROC | 0.7379618 @ epoch 8 (동일한 작은 104-episode val) |
| Logs / metrics | `logs/20260731_155635/v23_medium_bag_mean.out`; `logs/v23_medium_bag_mean/version_0/metrics.csv` |
| Verification | 전체 unittest **123개 통과** (`696.503s`) |

20-epoch 경계에서 validation CE와 total loss가 모두 run best였고,
epoch 16→19 CE가 `0.59550→0.59452→0.59369→0.59337`로 연속 개선되어
추가 수렴 여지를 확인합니다. `last.ckpt`의 model/optimizer/scheduler/global
step을 복원하여 총 50 epochs까지 연장했습니다.

| 50-epoch 재개 항목 | 값 |
|---|---|
| Config commit | `4d784dc` (`episode_batch_size=8`, `shape_group_size=8`, `max_epochs=50`) |
| Resume source | `checkpoints/20260731_155635/v23_medium_bag_mean/last.ckpt` |
| Active epoch range | epoch 20~49 (추가 30 epochs) |
| PID | `2671747` (종료) |
| **완료 상태** | **`max_epochs=50` 도달, 정상 종료** |
| **50-epoch best** | **epoch 43 `val_ce_loss=0.5912154`, `val_auroc=0.7383`** |
| Best checkpoint | `checkpoints/20260731_v23_bag_mean_50e_resume/v23_medium_bag_mean_50e/epoch=043-val_ce_loss=0.5912.ckpt` |
| Training log | `logs/20260731_v23_bag_mean_50e_resume/v23_medium_bag_mean_50e.out` |
| Metrics | `logs/v23_medium_bag_mean/version_1/metrics.csv` |

50-epoch 연장으로 훈련 val CE가 `0.59337 → 0.59122`로 개선됐고
v22 공식 best `0.5946`보다 `0.0034` 낮습니다. 104-episode val AUROC는
분산이 커서 승패 판정에 쓰지 않습니다.

### ✅ v24-A0 learned bag projection: 50-epoch 완료 (2026-07-31)

exact mean(v23) 대신 **learned linear projection**으로 bag을 1토큰으로
압축합니다. Slot을 12→1로 줄여 bag당 `1 global + 3 slot-statistic + 3 tail
= 7` 토큰을 만들고, concat(`7×512=3584`) → `Linear(3584, 512)` → bag당
512-d 1토큰을 생성합니다. 활성화 checkpoint는 `architecture_version=24`.

| 항목 | 값 |
|---|---|
| Branch / implementation | `codex/v23-bag-mean`, commit `26b2b27` |
| Config | `configs/train_v24_medium_bag_proj.yaml` (`project_structured_tokens: true`, slot 1 / density slot 1, `max_epochs=50`) |
| Run | `20260731_182755`, scratch Medium 50 epochs |
| **완료 상태** | **`max_epochs=50` 도달, 정상 종료** |
| **Best `val_ce_loss`** | **0.5976237** @ epoch 45 (마지막 epoch 49: 0.59819) |
| Best val AUROC | 0.7339473 (104-episode training val; 최종 판정용 아님) |
| Best checkpoint | `checkpoints/20260731_182755/v24_medium_bag_proj/epoch=045-val_ce_loss=0.5976.ckpt` |
| Model size | 8.4M trainable params (v22 6.57M + bag_token_projection ≈1.8M) |
| Training log | `logs/20260731_182755/v24_medium_bag_proj.out` |
| Metrics | `logs/v24_medium_bag_proj/version_0/metrics.csv` |
| Verification | 신규 테스트 5개 포함 `test_base_model` + `test_model_interface` **76개 통과** (`553.453s`) |

> [!NOTE]
> 훈련 val CE `0.5976`은 v22(0.5946)와 v23-A0(0.5912)보다 높습니다. slot을
> 1개로 줄인 정보 손실 영향으로 보입니다.

### ✅ v24-B0 per-token bottleneck projection: 50-epoch 완료 (2026-07-31)

v24-A0가 slot 1개로 정보를 잃는 문제를 해결하기 위한 variant. **12 slot 유지**
(40 tokens) + 토큰별 전용 `Linear(512→64)` 40개 적용 → concat(40×64=2560) →
`Linear(2560→512)` → bag당 512-d 1토큰. 직결 40×512→512(~10.5M) 대신
병목으로 파라미터를 ~2.62M로 제한.

| 항목 | 값 |
|---|---|
| Branch / implementation | `codex/v23-bag-mean`, commit `b2fb9d0` |
| Config | `configs/train_v24_medium_bag_proj_bottleneck.yaml` (`project_structured_tokens: true`, `projection_bottleneck_dim: 64`, 12 slot, `max_epochs=50`) |
| Run | `20260731_201252`, scratch Medium 50 epochs |
| **완료 상태** | **`max_epochs=50` 도달, 정상 종료** (PID `3033073` 종료) |
| **Best `val_ce_loss`** | **0.5923204** @ epoch 46 (v22 baseline 0.5946 대비 -0.0023, v24-A0 0.5976 대비 -0.0053 개선) |
| Model size | 9.2M trainable params (v22 6.57M + 병목 projection ≈2.62M) |
| Best checkpoint | `checkpoints/20260731_201252/v24_medium_bag_proj_bottleneck/epoch=046-val_ce_loss=0.5923.ckpt` |
| Training log | `logs/20260731_201252/v24_medium_bag_proj_bottleneck.out` |
| Checkpoints | `checkpoints/20260731_201252/v24_medium_bag_proj_bottleneck/` |
| Verification | 신규 테스트 4개 포함 `test_base_model` + `test_model_interface` **80개 통과** (`578.291s`) |

이 세 후보 모두 원래는 v22(`predictions/v22_medium_baseline_pool400_curve/`)와
1,000 pool-400 episode paired 비교로 판정할 계획이었으나, 사용자가 4종
비교 없이 v24-B1을 train CE만으로 직접 확정하면서 이 평가는 실행되지
않았습니다.
