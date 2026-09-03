# Current Status

- **Last Updated**: 2026-09-03 20:35 (KST)
- **Status**: CLEAN
- **Host / Node**: gnode3 (5x NVIDIA RTX A5000, 24GB VRAM)
- **Environment**: uv venv `.venv` | Python 3.12.11 (PyTorch 2.14.0+cu130, Lightning 2.6.5)
- **Active Session / Job**: None
- **Read First**: [#initial-baseline-migration](#initial-baseline-migration)

### Immediate Next Command
```bash
bash scripts/run_tests.sh
```

---

## 2026-09-03 — Subsampling 배제 순수 Context LOO Stacking 실측 및 LOO 영구 폐기 판정 (§212)

### Completed
- **Subsampling 배제 순수 Context LOO Stacking 50-Fold 전수 실측 완료 (§212)**:
  - 사용자 지침("subsampling 없이 LOO만 추가해서 LOO를 살릴지 말지 결정하자")에 따라, 깨끗한 풀백($S=1, f=1.0$) 환경에서 Context LOO 가중치 기반 앙상블 공식 50-Fold 측정.
  - **실측 결과**: v120 Active Baseline (`0.6265`) 대비 **`0.6125`로 -1.40%p 대폭 하락**.
  - **7개 중 5개 과제 침몰**: ARID1A (-2.72%p), KRAS (-2.44%p), SMAD4 (-3.00%p), KEAP1 (-2.07%p), Prog (-1.77%p).
  - **원인 규명**: Context LOO 점수와 실제 Test AUROC 간 스피어만 순위 상관계수가 **음수 ($\rho = -0.2679$)**로 측정됨. 단순 선형 모델(BM)이 Context 노이즈를 암기하여 가짜 높은 LOO를 얻고, 정작 테스트셋에서 강력하게 일반화되는 비선형 브랜치(DS)의 가중치를 깎아내려 앙상블이 붕괴됨.
- **최종 결정 (Definitive Decision)**:
  - **LOO는 영구 폐기(Discard)**.
  - 슬라이드 단위 극단치를 절사하는 **Trimmed Mean Voting (`0.6265`)**을 공식 앙상블 표준으로 확정.

### Active Research Queue (다음 연구 가설)
- **[Exp 1] Salience-Guided Selective Subsampling (국소 변이 앵커 보존)**:
  - **가설**: 살리언스 상위 패치를 앵커로 100% 보존하고 기질 패치만 서브샘플링하여, KRAS/KEAP1/PBRM1 하락을 원천 방어하면서 ARID1A(+7.2%p)와 Grade(+2.0%p)의 이점을 동시 획득.

### Code Reality vs Documentation Delta
- `scripts/test_pathobench.py`: `context_loo` aggregation 구현 및 실측 완료.
- `scripts/run_v120_clean_loo_experiments.sh`: 5-GPU 50-fold 오케스트레이터 완비.
- `docs/history/archive.md`: §212 기록 완료.

### Blockers & Tech Debt
- None. 131개 전 테스트 통과.

_by Antigravity on gnode3 at 2026-09-03 20:35:00_



