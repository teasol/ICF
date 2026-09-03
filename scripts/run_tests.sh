#!/usr/bin/env bash
# Regression suite runner (docs SS207).
#
# Two things this fixes over calling unittest directly:
#
# 1. THREAD OVERSUBSCRIPTION -- the actual bottleneck. The suite builds episodes
#    as CPU tensors, so the whole 6-branch pipeline runs through CPU BLAS. Its
#    hot spot is `within_slide_basis`: an eigh of a 1536x1536 float64 scatter,
#    ~30% of total suite time. On a many-core node OpenMP fans that tiny problem
#    across every core and burns the time in spin/sync, not arithmetic.
#    Measured on gnode-a5000 (52 cores, 5x RTX A5000), 119 tests:
#
#      OMP_NUM_THREADS=52 (default)  wall 78.6s   CPU 2721s
#      OMP_NUM_THREADS=16            wall 27.4s   CPU  253s
#      OMP_NUM_THREADS=8             wall 26.0s   CPU  140s   <- default here
#      OMP_NUM_THREADS=4             wall 38.0s   CPU   99s
#
#    3.0x wall and 19x less CPU burn, with all 119 tests passing identically at
#    every setting -- thread count changes no test outcome.
#
# 2. IMPORT PATH -- tests/ is not a package and only 4 of the 16 modules put the
#    repo root on sys.path themselves. Discovery imports alphabetically, so when
#    the repo root is not already on sys.path (any cwd but the root, or a runner
#    script living elsewhere) exactly the modules sorting before
#    test_config_numeric_types -- test_bd_branch and test_bm_branch, 14 tests --
#    die on `No module named 'src'`. unittest does report those as errors, but
#    the summary still reads a plausible "Ran 105 tests", which is easy to skim
#    past. Exporting PYTHONPATH makes the suite cwd-independent.
#
# Usage: bash scripts/run_tests.sh [pattern]     # default: test_*.py
#   ICF_TEST_THREADS=N   override the thread cap
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"

THREADS="${ICF_TEST_THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PATTERN="${1:-test_*.py}"
echo "run_tests: $PYTHON | threads=$THREADS | pattern=$PATTERN"
exec "$PYTHON" -m unittest discover -s tests -p "$PATTERN"
