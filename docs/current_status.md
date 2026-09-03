# Current Status

- **Last Updated**: 2026-09-03 17:25 (KST)
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

## 2026-09-03 — MIL Sub-bag Augmentation (Method 1 vs Method 2) 실측 (§210)

### Completed
- **8대 단독 브랜치 Primary 7 50-Fold 전수 실측 재현 완료 (§209)**.
- **MIL Sub-bag Data Augmentation 50-Fold 전수 실측 완료 (§210)**:
  - **Method 1 (Context 가상 표본 증강)** vs **Method 2 (Query Test-Time Augmentation / TTA)** 전수 비교.
  - **발견 1 (초대형 호재)**: 광범위 형태학적 변이 과제인 `ARID1A`에서 **0.5471 $\to$ 0.6179 (+7.1%p 폭등)**, `Histologic Grade`에서 **0.6823 $\to$ 0.7024 (0.70 벽 돌파)** 달성.
  - **발견 2 (병리학적 한계 규명)**: 2~5% 면적의 국소 변이 세포에 의존하는 `KRAS` (0.7295 $\to$ 0.6395), `KEAP1`, `PBRM1`은 무작위 균일 샘플링 시 변이 패치가 누락되는 False-Negative Sub-bag 현상으로 양성 신호가 희석됨.
  - **도출된 정밀 해결책**: 균일 무작위 샘플링 대신, DS 살리언스 상위 10~20% 패치는 100% 보존(Anchor)하고 기질 배경 패치만 무작위 드롭하는 **Salience-Guided Subsampling** 가설 도출.

### Active Research Queue (다음 연구 가설)
- **[Exp 1] Salience-Guided Selective Subsampling (국소 변이 보존 증강)**:
  - **가설**: 살리언스 상위 패치(KRAS/KEAP1 변이 클론)를 앵커로 보존한 채 기질 패치만 서브샘플링하여, KRAS 하락 없이 ARID1A(+7%p)와 Grade(+2%p)의 이점만 취함.
- **[Exp 2] In-Episode Adaptive Dynamic Stacking (동적 앙상블)**:
  - **가설**: 각 브랜치가 특정 암종에서 상호 보완적인 만큼, Context 슬라이드 내 Leave-One-Out (LOO) 오차 기반 동적 신뢰도 가중치 부여.

### Code Reality vs Documentation Delta
- **Sub-bag Augmentation Architecture**: `scripts/test_pathobench.py`에 `ICF_DS_AUG_MODE` (context / query / none), `ICF_DS_AUG_S`, `ICF_DS_AUG_FRACTION` 완비.
- **Primal Ridge Solver**: `src/models/common/solvers.py`에 $N > D$일 때 $32 \times 32$ Primal 공간에서 초고속 엄밀 Cholesky를 수행하도록 최적화 완료.

### Blockers & Tech Debt
- None. 131개 전 테스트 통과.

### Immediate Next Steps
- 사용자 피드백에 따라 [Exp 1] Salience-Guided Subsampling 또는 v120 앙상블 적용 진행.

_by Antigravity on gnode3 at 2026-09-03 17:25:00_

