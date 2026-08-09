# v40 CV-only 파생 config (아카이브, 2026-08-09)

CV-only 도입기의 진단·폐기 노선. 본선은
[`../../train_v40_cv_only_1536.yaml`](../../train_v40_cv_only_1536.yaml) (er_status 0.6989).

| config | 무엇이었나 | 결말 |
|---|---|---|
| `train_v40_cv_only_skip_1536.yaml` | skip 구현 등가성 확인용 (같은 seed로 나란히 학습) | val_ce ep0-4 완전 일치 확인 후 역할 종료 |
| `train_v40_cv_only_e4_1536.yaml` | `episode_batch_size: 4` | **폐기** — skip끼리 비교 시 E=1 60s vs E=4 86s (§68-5) |
| `train_v40_cv_only_ladder_1536.yaml` | epoch 사다리 진단(`callbacks: save_all`) | 50 epoch 필요 확인 후 역할 종료 (§69-6) |

`tests/test_vram_guard.py`가 e4 config를 참조하므로 삭제하지 말 것
(CV-only의 VRAM 추정이 `activation_layers=1`로 보정되는지 검사).
