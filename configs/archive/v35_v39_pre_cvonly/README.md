# v35~v39 entry point configs (아카이브, 2026-08-09)

CV-only 이전의 **6-분기 아키텍처 계보**다. §68에서 6개 evidence 분기 중 4개를 삭제해도
성능이 동일함이 확인되어(fold-paired −0.0005) 이 계보 전체가 폐기됐다.

| config | 무엇이었나 | 결말 |
|---|---|---|
| `train_v35_phase0_largebag_1536.yaml` | 데이터 단독 arm (large bag) | §60 완주, v34 대비 개선 미미 |
| `train_v36_q1_{baseline,structured}_1536.yaml` | 40→1 압축 해제 | **기각** Δ −0.0024 (§65) |
| `train_v37_{baseline,context_adaptive}_1536.yaml` | 압축 가중치를 에피소드가 결정 | **기각** Δ −0.0001 (§65) |
| `train_v38_ridge_ablation_*.yaml` | ridge 3종 개별 제거 | G-2 무기여, CV-1 제거 불가 (§66) |
| `train_v39_*.yaml` | 위 + 수치 안정화 | 안정화가 역효과 −0.0317 (§67) |

⚠️ v36/v37이 겨냥한 Q-5 population attention은 **상수를 뱉고 있었다**(AUROC 0.5000,
std 0.0000, §68-1). 두 arm이 Δ≈0으로 끝난 것이 그것으로 설명된다.

`tests/test_context_adaptive_aggregation.py`가 v36 baseline config를 참조하므로 삭제하지 말 것.
현행 노선은 [`../../train_v41_cvonly_K128_1536.yaml`](../../train_v41_cvonly_K128_1536.yaml).
