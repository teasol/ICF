# One-off analysis probes

These are **standalone analysis scripts, not tests**. They lived in `tests/` under
`test_*.py` names until 2026-09-02 (docs §206), which meant
`unittest discover -s tests -p "test_*.py"` — the command
[`docs/agent_handoff.md`](../../docs/agent_handoff.md) §1 tells every new session to
run — picked them up as if they were regression tests. None of them imports
`unittest` or defines a `TestCase`, so three consequences followed:

- `context_weighting.py`, `drop2_furthest_detail.py` and `furthest_trimming.py`
  had no `if __name__ == "__main__"` guard, so **discovery executed their module
  bodies** — an experiment ran just to collect the suite.
- `drop2_furthest_detail.py` and `furthest_trimming.py` load
  `predictions/pathobench_cptac_lscc_ARID1A_mutation_ds_w1_primary7_official50_bf16.pt`,
  an artifact of one specific sweep. When it is absent the import raises, and the
  suite reported **2 permanent errors** that had nothing to do with the code.
- The real contract suite's size was unclear: 24 `test_*.py` files, but only 16
  held actual tests.

Each script is a record of one experiment recorded in
[`docs/current_status.md`](../../docs/current_status.md) — the KRR readout (§199),
the direct likelihood ratio (§200), the trimming-aggregation sweep (§201), the
in-context Fisher subspace (§202), DE/SW branch development (§203). They are kept
for reproducibility, not as gates.

## Running one

Run from the repo root, with the project environment loaded:

```bash
. scripts/node_env.sh
$PYTHON scripts/analysis/<script>.py
```

Several of them read saved per-fold logits out of `predictions/`, so they only
work after the sweep that produced those files. Check the script's `TASKS` /
path constants before running it.
