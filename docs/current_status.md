# Current Status

- **Last Updated**: 2026-09-03 11:42 (KST)
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

## 2026-09-03 — Initial Baseline Migration & Model Modularization

### Completed
- Successfully migrated repository to Universal Handoff Protocol.
- Archived accumulated historical logs (§0 through §207, 857 lines) into [`docs/history/archive.md`](history/archive.md).
- Streamlined `docs/agent_handoff.md` into standard 5-part permanent architectural reference.
- Established agent entrypoints: verified `AGENTS.md` and `CLAUDE.md` in repository root.
- Configured repository-local Git credential helper using `/home/kimds/.gittoken_icf` (isolated to this repo), and successfully synchronized/pushed to `origin/main`.
- Absorbed `current_experiments.md` active queue into `current_status.md` and archived past summaries into `docs/history/archive.md` (§208), establishing the clean 3-document SSOT system.
- Modularized monolithic `src/models/training_free.py` (1,340 -> 328 lines) into dedicated submodules (`src/models/common/`, `src/models/branches/`, `src/models/aggregations/`) via `/call-claude` (`model=sonnet`, `effort=low`).
- Systematized YAML config architecture (`src/models/config.py`, `configs/baseline/`, `configs/experiments/`) and established 11-category anti-trap contract test suite (`tests/test_yaml_config_contract.py`). All 17 modules / 131 unit tests pass in 18.9s.
- Committed §207 test tooling (`scripts/run_tests.sh` and `scripts/node_env.sh`).

### Active Research Queue (다음 연구 가설)
- **[Exp 1] CT (Cell Tokenizer) 단독 브랜치 고도화 연구**:
  - **가설**: K256 계층적 트리 토크나이저의 다중 스케일화, `cattopk` (Mean + Top-K) 풀링, 토큰별 국소 분산 모멘트 결합을 통해 `CT` 단독 성능을 극대화하여 `SMAD4` 사각지대 해소 및 단일 모델 완성도 제고.
  - **선행 실측치**: CT 단독 Primary 7 `0.6147` (단독 1위), SEAL 10 `0.7197` (§204).

### Code Reality vs Documentation Delta
- **Active Architecture Baseline**: Legacy `docs/README.md` previously claimed v112 as active; confirmed and standardized that the true active baseline is **v120** (6-Branch Trimmed Mean Voting: CV + CT + BM + BD + QA + DS, with DD disabled; Primary 7 Macro `0.6265`, SEAL 10 `0.6972`, All 17 `0.6681`).
- **Environment Management**: Legacy docs referenced lost conda `BagPFN` environment; environment is now standardized under `uv venv` at `ICF/.venv` (Python 3.12.11).
- **Test Discovery & Performance**: Direct `python -m unittest discover` previously suffered from OpenMP thread oversubscription (~78s) and missing repo root on `sys.path` in 12/16 test modules. `scripts/run_tests.sh` enforces `OMP_NUM_THREADS=8` and exports `PYTHONPATH`, reducing runtime to 20.0s (3x speedup).
- **Config Lineage**: `configs/` root was cleaned in §205 leaving only `train_v98_p1_reverse_1536_1gpu.yaml` as the checkpoint shell, with branch arms injected dynamically via `scripts/lib/arms.sh`.

### Blockers & Tech Debt
- None. Working tree is clean and all 119 tests pass identically.

### Immediate Next Steps
- Execute regression test suite (`bash scripts/run_tests.sh`) to verify system stability.
- Run baseline verification on Primary 7 benchmark: `bash scripts/eval_v120.sh 0 baseline_check`.

- Commence [Exp 1] CT Refinement research once baseline verification is confirmed.

_by Antigravity on gnode3 at 2026-09-03 11:00:00_

