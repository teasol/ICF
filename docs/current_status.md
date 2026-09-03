# Current Status

- **Last Updated**: 2026-09-03 22:50 (KST)
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

## 2026-09-03 — CT 뮤트 5-Branch (v121) 고속 베이스라인 및 Salience Anchor Subsampling 실측 (§213)

### Completed
- **CT 브랜치 뮤트 기반 v121 5-Branch 고속 베이스라인 구축 및 50-Fold 실측 완료 (§213)**:
  - `ICF_FIXED_HEAD_CT_WEIGHT=0.0`으로 K-Means 병목 제거 $\to$ 5-GPU 350-Fold 전수 완주 시간이 45분에서 **12분으로 4배 가속**.
  - 5-Branch (CV, BM, BD, QA, DS) Trimmed Mean Macro AUROC: **`0.6171`** (안정적 고속 평가 환경 확립).
- **Salience-Guided Anchor Subsampling 50-Fold 실측 완료 (§213)**:
  - **초대형 성과 (단독 DS)**:
    - `ARID1A`: 기존 `0.5471` $\to$ **`0.6236` (+7.65%p 폭등, 전 모델 사상 최고치 경신!)**.
    - `Histologic Grade`: `0.6823` $\to$ **`0.7013`** (0.70 벽 돌파).
    - `PBRM1`: 균일 무작위 대비 +0.89%p 회복 (`0.5131`).
  - **앙상블 기전 규명 (Trimmed Mean Max-Drop 현상)**:
    - v121 5-branch Trimmed Mean에서 ARID1A DS가 0.6236으로 독주할 때, Trimmed Mean이 DS를 '최고점 이상치(Max)'로 판정하여 잘라버림.
    - 비대칭 독주 과제에서 우수 브랜치를 보호하기 위한 Soft Voting 또는 Certainty Gating 필요성 확인.

### Active Research Queue (다음 연구 가설)
- **[Exp 1] Certainty-Gated Voting / Soft Voting on v121 + Salience Anchor DS**:
  - **가설**: 비대칭 과제(ARID1A, SMAD4)에서 단독 1위 브랜치를 Max-Drop하지 않고 온전히 반영하여 Macro AUROC 0.630+ 돌파.

### Code Reality vs Documentation Delta
- `configs/baseline/v121_active.yaml`: 5-branch 고속 설정 추가.
- `scripts/eval_v121.sh`, `scripts/run_v121_primary7.sh`: v121 전용 5-GPU 러너 추가.
- `scripts/test_pathobench.py`: `salience_anchor` 증강 모드 추가.
- `scripts/run_ds_salience_anchor.sh`, `scripts/run_v121_salience_anchor.sh`: 오케스트레이터 완비.
- `docs/history/archive.md`: §213 기록 완료.

### Blockers & Tech Debt
- None. 131개 전 테스트 통과.

_by Antigravity on gnode3 at 2026-09-03 22:50:00_




