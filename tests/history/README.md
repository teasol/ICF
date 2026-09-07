# Archived test suites

These files preserve regression and research contracts for retired architectures,
superseded diagnostics, and historical experimental ablation arms (v1~v100+).
They deliberately use the `legacy_*.py` filename pattern, so the default discovery
command does not execute them:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The compact core suite (98 tests) is the required CI and development gate for active
v121 Training-Free (5-branch CV/BM/BD/QA/DS Trimmed Mean) work. Archived tests are historical
references and can be run individually on demand when inspecting or modifying preserved code paths:

```bash
python -m unittest tests/history/legacy_set_transformer_ridge.py
python -m unittest tests/history/legacy_v30_core_contracts.py
```

## 2026-09-07 재편 이관 내역
폐쇄 축 및 구시대 학습 파라미터/일회성 프로브 테스트 9건(60 tests)을 이관:
- `legacy_scheduler.py`: SGD LR 스케줄러 (학습 파라미터 0개 전환, `CA-14`)
- `legacy_set_transformer_ridge_contracts.py`: Era 4 학습형 SetTransformer 모델 (`CA-14`)
- `legacy_v30_core_contracts.py`: Era 2 (v30) 학습 파이프라인 및 PyTorch Lightning 인터페이스
- `legacy_ru81_probe.py`: 종료된 RU-81 전용 일회성 진단 프로브
- `legacy_in_episode_loo.py`: In-Episode Context LOO 가중 (`CA-07` 폐쇄)
- `legacy_de_branch.py` / `legacy_sw_branch.py`: 기각된 DE/SW 브랜치 (`CA-10`, `P2-SW`)
- `legacy_dd_adaptive_rank.py`: Adaptive Rank DD (`CA-02`)
- `legacy_ct_readout_contracts.py`: CT readout 단독 계약 (공식 비교 기준선 제외, `CA-03`)
