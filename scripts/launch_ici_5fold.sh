#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Optional v22 pretrain checkpoint to fine-tune from; empty means train from
# scratch. Every pre-v22 checkpoint (including the v21 Phase 5 large-context
# one) is rejected by ModelInterface.on_load_checkpoint, because the v22 bump
# removed the retrieval layer -- a new pretrain run is required before this
# can point anywhere.
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"

echo "=== Launching ICI 5-Fold CV Fine-Tuning (v22, full context) ==="
if [[ -n "${PRETRAINED_CKPT}" ]]; then
    echo "Fine-tuning from: ${PRETRAINED_CKPT}"
else
    echo "No PRETRAINED_CKPT set -- training from scratch."
fi

for FOLD in {0..4}; do
    RUN_KIND="v22_ici_finetune_f${FOLD}"
    CONFIG="configs/train_v22_ici_finetune_fold${FOLD}.yaml"

    echo "Launching Fold ${FOLD} fine-tuning..."
    CUDA_DEVICES=0 \
    NPROC_PER_NODE=1 \
    TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun \
    NETRC=/NHNHOME/kimds/.netrc \
    CKPT_PATH="${PRETRAINED_CKPT}" \
    scripts/launch_interactive_training.sh \
      "${RUN_KIND}" \
      "${CONFIG}"
    # launch_interactive_training.sh omits --ckpt-path when CKPT_PATH is empty.
    sleep 2
done

echo "=== All 5 folds launched in detached background sessions ==="
