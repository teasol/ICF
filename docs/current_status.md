# Current Status

- **Last Updated**: 2026-09-03 14:30 (KST)
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

## 2026-09-03 — 8대 단독 브랜치 전수 실측 & 계산 시간 프로파일링 (§209)

### Completed
- **8대 단독 브랜치 Primary 7 50-Fold 전수 실측 재현 완료 (§209)**:
  - 단독 성능 순위: **1위 DS (`0.6265`)**, **2위 QA (`0.6209`)**, **3위 CT (`0.6147`)**, 4위 BM (`0.6089`), 5위 BD (`0.6071`), 6위 CV (`0.6004`), 7위 SW (`0.5976`), 8위 DE (`0.5953`).
- **과거 요약 문서상의 왜곡 및 착시 팩트체크 완료**:
  - 과거 요약표에 기재되었던 *"CT 단독 SEAL 10 `0.7197`"*은 전사 오류였으며, 실제 실측치는 **`0.6882`**로 확인됨.
  - CT는 단독 1위가 아니며(`0.6147`), 기질 노이즈를 제거한 **DS(`0.6265`)**와 상하위 분위수를 본 **QA(`0.6209`)**가 단독 성능 챔피언임.
- **다중 브랜치 앙상블(v120)의 수학적 상호보완 메커니즘 규명**:
  - `SMAD4 변이`: CT(0.4283), BM(0.4491), BD(0.4327), QA(0.4503), DS(0.4465) 등 대다수 브랜치가 역방향 예측으로 무너지나, **오직 `CV Alone (0.5483)`만이 유일하게 양(+)의 신호를 정상 방어**함.
  - `ARID1A 변이`: 반대로 CV가 0.4308로 무너지는 영역을 DS(0.5471), CT(0.5360), QA(0.5307)가 보완함.
- **계산 시간 계측 및 속도 최적화 패치 적용**:
  - **CT Alone**: 매 Fold마다 256개 K-Means++ 동적 클러스터링 및 소프트 할당이 불가피하여 **Fold당 10~15초 / 태스크당 약 8~12분** 소요 (캐시 공유 불가).
  - **비-CT 브랜치**: `bag_stats_cache` ($n, \boldsymbol{\mu}, X^T X$) 활용으로 **Fold당 ~3초 / 태스크당 약 2분 30초**로 CT 대비 2.5~4배 고속.
  - `scripts/test_pathobench.py`: `ct_weight == 0.0` 시 불필요한 K-Means를 즉시 건너뛰도록 패치하여 비-CT 평가 시간을 **태스크당 40초 이상 즉각 단축**.

### Active Research Queue (다음 연구 가설)
- **[Exp 1] CT 단독 브랜치 고도화 (SMAD4 사각지대 타격)**:
  - **가설**: K256 계층적 트리 토크나이저의 다중 스케일화, `cattopk` (Mean + Top-K) 풀링, 토큰별 국소 분산 모멘트 결합을 통해 CT의 고질적 약점인 `SMAD4 (0.4283)` 사각지대를 극복하고 단독 모델 완성도 제고.
- **[Exp 2] In-Episode Adaptive Dynamic Stacking (동적 앙상블)**:
  - **가설**: 각 브랜치(CV의 2차 모멘트 vs DS/QA의 1차 분위수)가 특정 암종에서 상호 보완적인 만큼, 단순 고정 가중치 Voting 대신 Context 슬라이드 내 Leave-One-Out (LOO) 오차 기반 동적 신뢰도 가중치 부여.

### Code Reality vs Documentation Delta
- **CT Standalone Metric**: 과거 요약표의 `SEAL 10 0.7197`은 허위/오기록이며 실측치는 `0.6882`임. Primary 7 단독 1위는 CT(`0.6147`)가 아닌 DS(`0.6265`)임.
- **Branch Evaluation Speed**: CT 제외 전 브랜치는 `bag_stats_cache` 및 메모리 사전 적재로 인해 50-Fold가 약 2분 30초 내에 완주됨.

### Blockers & Tech Debt
- None. 모든 단위 테스트 통과 및 5-GPU 병렬 인프라 100% 가동 확인.

### Immediate Next Steps
- 사용자 피드백에 따라 [Exp 1] CT 풀링/다중스케일 고도화 또는 [Exp 2] In-Episode 동적 앙상블 착수.

_by Antigravity on gnode3 at 2026-09-03 14:30:00_

