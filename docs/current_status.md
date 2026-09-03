# Current Status

- **Last Updated**: 2026-09-03 23:08 (KST)
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

## 2026-09-03 — Adaptive Trimmed & Hard Gated Voting 공식화 및 350-Fold 전수 실측 (§214)

### Completed
- **Adaptive Trimmed & Hard Gated Voting 공식 등록 및 350-Fold 전수 실측 완료 (§214)**:
  - Trimmed Mean의 Max 절사 함정을 극복하기 위해 두 가지 투표 메커니즘을 파이프라인 정식 옵션으로 등록:
    1. **`adaptive_trimmed`**: 높은 확신도($|p - 0.5|$)를 가진 우수 브랜치를 절사에서 보호.
       - **실측 결과**: Primary 7 Macro **`0.6204` (+0.31%p 상승, 전체 1위)**.
       - **KRAS 변이**: `0.7004` $\to$ **`0.7226` (+2.22%p 폭등)**.
       - **SMAD4 변이**: `0.4420` $\to$ **`0.4710` (+2.90%p 폭등)**.
       - **KEAP1 변이**: `0.6042` $\to$ **`0.6170` (+1.28%p 폭등)**.
    2. **`hard_gated`**: $|p - 0.5| < 0.05$ 무기력 브랜치 투표권 박탈.
       - **ARID1A 변이**: `0.5530` $\to$ **`0.5752` (+2.22%p)**.
       - **SMAD4 변이**: `0.4420` $\to$ **`0.4904` (+4.84%p 폭등)**.
- **아키텍처 및 테스트 완비**:
  - `src/models/config.py`: `VALID_AGGREGATIONS`에 `"adaptive_trimmed"`, `"hard_gated"` 정식 추가.
  - `src/models/aggregations/voting.py`, `src/models/training_free.py`, `scripts/test_pathobench.py` 구현 완료.
  - `tests/test_gated_and_adaptive_trimmed.py`: 라벨 반전 대칭성 검증 완료. 133개 전 테스트 통과.

### Active Research Queue (다음 연구 가설)
- **[Exp 1] v121 + Adaptive Trimmed 앙상블 전수 적용**:
  - `configs/baseline/v121_active.yaml`의 기본 집계 방식을 `adaptive_trimmed`로 승격하여 공식 표준 베이스라인 점수 향상.

### Code Reality vs Documentation Delta
- `src/models/config.py`: `adaptive_trimmed`, `hard_gated` 옵션 추가.
- `src/models/aggregations/voting.py`: 2개 aggregation 함수 구현.
- `src/models/training_free.py`: _solve_heads 배선 완료.
- `scripts/test_pathobench.py`: aggregation 지원 추가.
- `tests/test_gated_and_adaptive_trimmed.py`: 신규 테스트 추가.
- `docs/history/archive.md`: §214 기록 완료.

### Blockers & Tech Debt
- None. 133개 전 테스트 통과.

_by Antigravity on gnode3 at 2026-09-03 23:08:00_





