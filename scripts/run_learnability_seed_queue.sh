#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <stage> <seed> [seed ...]" >&2
    exit 2
fi

stage="$1"
shift

for seed in "$@"; do
    echo "[$(date --iso-8601=seconds)] Starting ${stage} seed ${seed}."
    if ICF_FOREGROUND=1 \
       SEED="${seed}" \
       WANDB_MODE="${WANDB_MODE:-offline}" \
           scripts/run_learnability_ladder.sh "${stage}"; then
        echo "[$(date --iso-8601=seconds)] Completed ${stage} seed ${seed}."
    else
        echo "[$(date --iso-8601=seconds)] Stage ${stage} seed ${seed} failed." >&2
    fi
done

echo "[$(date --iso-8601=seconds)] Queue completed: ${stage} seeds $*."
