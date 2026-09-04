#!/usr/bin/env bash
# Run a screening/eval orchestrator to completion and print ONE compact summary.
#
# Why this exists: waiting on a GPU run by polling from the agent side costs a tool
# call (and tokens) per check. This blocks in a single call instead -- launch it once
# with run_in_background, and the completion notification carries the whole result.
#
# Usage:
#   bash scripts/run_and_wait.sh <runner.sh> <tag> [analysis command...]
# Example:
#   bash scripts/run_and_wait.sh scripts/run_v121_shape_screen.sh v121_shape_screen \
#        "PYTHONPATH=$PWD .venv/bin/python scripts/analysis/branch_screen.py \
#         --tag v121_shape_screen --candidate m_sh"
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"

RUNNER="${1:?usage: run_and_wait.sh <runner.sh> <tag> [analysis cmd]}"
TAG="${2:?usage: run_and_wait.sh <runner.sh> <tag> [analysis cmd]}"
shift 2
ANALYSIS="${*:-}"
LOG="logs/${TAG}_orchestrator.log"
EXPECTED=7   # Primary 7

started=$(date +%s)
echo "=== ${TAG}: launching ${RUNNER} at $(date '+%F %T') ==="
bash "$RUNNER" "$TAG" > "$LOG" 2>&1
rc=$?
elapsed=$(( $(date +%s) - started ))

n=$(ls predictions/*_"${TAG}"_official50_bf16.pt 2>/dev/null | wc -l)
echo "=== ${TAG}: runner rc=${rc}, ${n}/${EXPECTED} tasks, ${elapsed}s elapsed ==="

# Contamination check: SCREEN_ONLY runs must leave fold-mean AUROC untouched.
echo "--- fold-mean AUROC per task (compare against the 5-branch baseline) ---"
grep -h "=== END" "$LOG" 2>/dev/null | sed 's/.*END *//' | awk '{print "  "$0}'

if [ "$rc" -ne 0 ] || [ "$n" -lt "$EXPECTED" ]; then
    echo "!!! incomplete -- last 15 log lines:"
    tail -15 "$LOG"
    exit 1
fi

if [ -n "$ANALYSIS" ]; then
    echo "--- analysis ---"
    eval "$ANALYSIS"
fi
echo "=== ${TAG}: done ==="
