#!/usr/bin/env bash
set -uo pipefail

run_seed() {
    local stage="$1"
    local seed="$2"
    echo "[$(date --iso-8601=seconds)] Starting ${stage} seed ${seed}."
    if ICF_FOREGROUND=1 SEED="${seed}" WANDB_MODE="${WANDB_MODE:-offline}" \
        scripts/run_learnability_ladder.sh "${stage}"; then
        echo "[$(date --iso-8601=seconds)] Completed ${stage} seed ${seed}."
        return 0
    fi
    echo "[$(date --iso-8601=seconds)] Failed ${stage} seed ${seed}."
    return 1
}

run_replacement() {
    local stage="$1"
    local seed="$2"
    while ! run_seed "${stage}" "${seed}"; do
        seed=$((seed + 1))
        echo "[$(date --iso-8601=seconds)] Replacing with ${stage} seed ${seed}."
    done
    LAST_SUCCESSFUL_SEED="${seed}"
}

# D0 seeds 42 and 43 completed previously; seed 44 failed and is replaced.
run_replacement D0 45
echo "[$(date --iso-8601=seconds)] D0 replacement completed with seed ${LAST_SUCCESSFUL_SEED}."

for stage in D1 D2 D3 D4; do
    next_replacement=45
    for seed in 42 43 44; do
        if run_seed "${stage}" "${seed}"; then
            continue
        fi
        run_replacement "${stage}" "${next_replacement}"
        next_replacement=$((LAST_SUCCESSFUL_SEED + 1))
    done
    echo "[$(date --iso-8601=seconds)] Completed stage ${stage}."
done

echo "[$(date --iso-8601=seconds)] Remaining D queue completed."
