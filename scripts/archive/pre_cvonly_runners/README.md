# 일회성 러너 (아카이브, 2026-08-09)

특정 arm 세트 전용으로 만든 스크립트다. 대상 config가
`configs/archive/{v35_v39_pre_cvonly,v40_cvonly_variants}/`로 함께 이동했으므로 그대로는
동작하지 않는다(경로 갱신 필요).

| script | 대상 |
|---|---|
| `eval_v36_v37_arms.sh` | v36 Q1 / v37 4개 arm |
| `eval_v38_ridge_ablation.sh` | v38 ridge ablation 4개 arm |
| `eval_v39_arms.sh` | v39 3개 arm |
| `eval_v40_arms.sh` | v40 CV-only |
| `queue_v38_wave2.sh`, `queue_v39_wave2.sh` | 2-wave 학습 큐 |
| `queue_v30_poolz.sh` | v30 시절 |

**현행 평가 러너는 [`../../eval_seal_tasks.sh`](../../eval_seal_tasks.sh)** — SEAL 10개 task를
임의 ckpt/config로 채점한다. 새 arm은 그것을 쓸 것.

⚠️ `queue_*.sh`의 대기 로직에는 §66-5의 함정 2건에 대한 대응이 들어 있다(launcher wrapper가
torchrun child보다 먼저 종료, pgrep 패턴 자기 매칭). 새 큐를 만들 때 참고할 것.
