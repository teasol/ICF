#!/usr/bin/env bash

# Full ICI evaluation protocol: every available seed partition x 5 folds.
#
# The v21 work reported architecture comparisons from a single SEED42 5-fold
# run, and the retrieval investigation later showed those differences were
# indistinguishable from noise. Four of the five seed partitions on disk were
# never touched. This sweeps all of them so a result comes with a measured
# seed-to-seed spread instead of an implicit assumption that one partition
# speaks for the cohort.
#
# Folds inside a seed run concurrently (they fit on one B200 together, as the
# v21 5-fold runs showed); seeds run sequentially so the GPU is not
# oversubscribed 25 ways.
#
# Usage:
#   scripts/launch_ici_protocol.sh                      # scratch, all seeds
#   PRETRAINED_CKPT=/abs/path.ckpt scripts/launch_ici_protocol.sh
#   SEEDS="42 1234" scripts/launch_ici_protocol.sh      # subset
#   CONFIG=configs/train_v22_ici_scratch.yaml scripts/launch_ici_protocol.sh

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Seed partitions present under data/ICI_CVOnly_scConcept_512/SEED*.
SEEDS="${SEEDS:-42 1234 2026 271828 314159}"
FOLDS="${FOLDS:-0 1 2 3 4}"
CONFIG="${CONFIG:-configs/train_v22_ici_finetune.yaml}"
TAG="${TAG:-v22_ici}"
# Empty means train from scratch. Any pre-v22 checkpoint is rejected by
# ModelInterface.on_load_checkpoint, so this must point at a v22 pretrain.
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 2
fi

echo "=== ICI protocol sweep ==="
echo "config : ${CONFIG}"
echo "seeds  : ${SEEDS}"
echo "folds  : ${FOLDS}"
if [[ -n "${PRETRAINED_CKPT}" ]]; then
    echo "init   : fine-tune from ${PRETRAINED_CKPT}"
else
    echo "init   : from scratch"
fi
echo

MANIFEST="${PROJECT_ROOT}/logs/${TAG}_sweep_manifest.tsv"
mkdir -p "${PROJECT_ROOT}/logs"
printf 'seed\tfold\trun_kind\tcheckpoint_dir\n' >"${MANIFEST}"

for SEED in ${SEEDS}; do
    echo "--- seed ${SEED} ---"
    PIDS=()
    for FOLD in ${FOLDS}; do
        RUN_KIND="${TAG}_s${SEED}_f${FOLD}"
        RUN_TIME="$(date +%Y%m%d_%H%M%S)"
        SEED="${SEED}" \
        CV="${FOLD}" \
        CUDA_DEVICES=0 \
        NPROC_PER_NODE=1 \
        TORCHRUN_BIN=/home/aibio_3/miniconda3/envs/BagPFN/bin/torchrun \
        NETRC=/NHNHOME/BASE/kimds/.netrc \
        CKPT_PATH="${PRETRAINED_CKPT}" \
        ICF_RUN_TIME="${RUN_TIME}" \
        scripts/launch_interactive_training.sh "${RUN_KIND}" "${CONFIG}"
        printf '%s\t%s\t%s\t%s\n' \
            "${SEED}" "${FOLD}" "${RUN_KIND}" \
            "checkpoints/${RUN_TIME}/${RUN_KIND}" >>"${MANIFEST}"
        PIDS+=("${PROJECT_ROOT}/logs/${RUN_TIME}/${RUN_KIND}.pid")
        sleep 2
    done

    # Wait for this seed's folds before starting the next seed.
    echo "waiting for seed ${SEED} folds to finish..."
    while true; do
        running=0
        for pidfile in "${PIDS[@]}"; do
            [[ -f "${pidfile}" ]] || continue
            pid="$(cat "${pidfile}")"
            if kill -0 "${pid}" 2>/dev/null; then
                running=$((running + 1))
            fi
        done
        [[ ${running} -eq 0 ]] && break
        sleep 30
    done
    echo "seed ${SEED} done."
done

echo
echo "=== sweep complete ==="
echo "Run manifest: ${MANIFEST}"
echo
echo "Next: evaluate each seed, then aggregate."
echo "  python scripts/test.py --checkpoints <5 fold ckpts for a seed> \\"
echo "      --config ${CONFIG} --precision bf16-mixed --validation-only \\"
echo "      --output predictions/${TAG}_seed<SEED>.pt"
echo "  python scripts/evaluate_protocol.py --predictions predictions/${TAG}_seed*.pt"
