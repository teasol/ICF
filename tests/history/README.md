# Archived test suites

These files preserve regression and research contracts for retired architectures,
superseded diagnostics, and historical experimental ablation arms (v1~v100+).
They deliberately use the `legacy_*.py` filename pattern, so the default discovery
command does not execute them:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The compact core suite (86 tests) is the required CI and development gate for active
v114 Training-Free work. Archived tests are historical references and can be run
individually on demand when inspecting or modifying preserved code paths:

```bash
python -m unittest tests/history/legacy_set_transformer_ridge.py
```
