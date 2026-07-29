#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <stage> [stage ...]" >&2
    exit 2
fi

for stage in "$@"; do
    echo "[$(date --iso-8601=seconds)] Starting stage ${stage}."
    WANDB_MODE="${WANDB_MODE:-offline}" \
        scripts/run_learnability_seed_queue.sh "${stage}" 42 43 44
    echo "[$(date --iso-8601=seconds)] Completed stage ${stage}."
done

echo "[$(date --iso-8601=seconds)] Stage queue completed: $*."
