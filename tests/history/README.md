# Archived test suites

These files preserve regression and research contracts for retired architectures and
superseded diagnostics. They deliberately use the `legacy_*.py` filename pattern, so
the default command does not execute them:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

The compact default suite is the required gate for active v30/v33 work. Archived tests
are historical references and are not a required CI gate. Run an individual archived
module explicitly only when modifying its preserved code path.
