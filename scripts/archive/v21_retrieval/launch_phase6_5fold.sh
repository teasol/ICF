#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PRETRAINED_CKPT="${PROJECT_ROOT}/checkpoints/20260728_144957/v21_large_context_pretrain/epoch=014-val_ce_loss=0.5940.ckpt"

echo "=== Launching Phase 6: ICI 5-Fold CV Fine-Tuning from Signal-Aware Retrieval Pretrain ==="

for FOLD in {0..4}; do
    RUN_KIND="v21_ici_finetune_phase6_f${FOLD}"
    CONFIG="configs/train_v21_ici_finetune_fold${FOLD}.yaml"

    echo "Launching Fold ${FOLD} fine-tuning..."
    CUDA_DEVICES=0 \
    NPROC_PER_NODE=1 \
    TORCHRUN_BIN=/NHNHOME/kimds/miniconda3/envs/BagPFN/bin/torchrun \
    NETRC=/NHNHOME/kimds/.netrc \
    CKPT_PATH="${PRETRAINED_CKPT}" \
    scripts/launch_interactive_training.sh \
      "${RUN_KIND}" \
      "${CONFIG}"
    sleep 2
done

echo "=== All 5 folds launched in detached background sessions ==="
