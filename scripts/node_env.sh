#!/usr/bin/env bash
# Node-specific settings, resolved in ONE place (docs SS164).
#
# ICF runs on several GPU servers that share Lustre storage under different mount
# names, and the conda env, the free GPUs and the checkpoint location all differ
# per node. Every runner sources this so that moving servers is one file, not a
# grep across scripts.
#
# Everything is overridable from the environment; nothing here overwrites a value
# that is already set. To pin a node permanently, export these in your shell or
# write them into scripts/node_env.local.sh (git-ignored, sourced at the end).
#
#   ICF_PYTHON     interpreter that has torch + lightning
#   ICF_DATA_ROOT  parent of official/ and features/
#   ICF_CKPT       v98 checkpoint used as the shell for the training-free configs
#   ICF_CONFIG     model config that checkpoint was built with
#   NGPU           how many GPUs this run may use
#   GPU_OFFSET     index of the first GPU it may use
#
# ⚠️ NGPU/GPU_OFFSET are a COURTESY setting, not a capability one. On the node this
# was written for, GPUs 4-7 carried another user's training, so the default is
# 0-3. On a node you have to yourself, export NGPU to the real count.

# ---- interpreter ----------------------------------------------------------
if [ -z "${ICF_PYTHON:-}" ]; then
  for candidate in \
    "${PYTHON_BIN:-}" \
    /home/aibio_3/miniconda3/envs/BagPFN/bin/python \
    "$HOME/miniconda3/envs/BagPFN/bin/python" \
    "$HOME/miniforge3/envs/BagPFN/bin/python" \
    "$(command -v python3 || true)"
  do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    # lightning is the import that fails on a bare python3 and silently drops 14
    # test modules while still printing a confident "Ran 158 tests" (SS141-3).
    if "$candidate" -c "import torch, lightning" >/dev/null 2>&1; then
      ICF_PYTHON="$candidate"; break
    fi
  done
fi
if [ -z "${ICF_PYTHON:-}" ]; then
  echo "node_env: no interpreter with torch+lightning found. Set ICF_PYTHON." >&2
  return 1 2>/dev/null || exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-$ICF_PYTHON}"

# ---- data -----------------------------------------------------------------
if [ -z "${ICF_DATA_ROOT:-}" ]; then
  for candidate in \
    /NHNHOME/BASE/kimds/Data/PathoBench \
    /lustre/BASE/kimds/Data/PathoBench \
    "$HOME/Data/PathoBench"
  do
    [ -d "$candidate/official" ] && { ICF_DATA_ROOT="$candidate"; break; }
  done
fi
if [ -z "${ICF_DATA_ROOT:-}" ] || [ ! -d "$ICF_DATA_ROOT/official" ]; then
  echo "node_env: PathoBench not found. Set ICF_DATA_ROOT (needs official/ and features/)." >&2
  return 1 2>/dev/null || exit 1
fi
OFFICIAL="$ICF_DATA_ROOT/official"
FEATURES="$ICF_DATA_ROOT/features"

# ---- checkpoint -----------------------------------------------------------
# Only a shell: v106+ overrides the projection and the head, so no learned value
# from it reaches the margin and any v98 seed gives the same number (SS152).
if [ -z "${ICF_CKPT:-}" ]; then
  ICF_CKPT="$(ls checkpoints/*/v98_p1_reverse_seed42/periodic-epoch=049*.ckpt 2>/dev/null | head -1)"
fi
ICF_CONFIG="${ICF_CONFIG:-configs/train_v98_p1_reverse_1536_1gpu.yaml}"

# ---- GPUs -----------------------------------------------------------------
if [ -z "${NGPU:-}" ]; then
  detected="$(nvidia-smi -L 2>/dev/null | wc -l)"
  NGPU="$([ "${detected:-0}" -gt 0 ] && echo 4 || echo 1)"
fi
GPU_OFFSET="${GPU_OFFSET:-0}"

[ -f "$(dirname "${BASH_SOURCE[0]}")/node_env.local.sh" ] && \
  . "$(dirname "${BASH_SOURCE[0]}")/node_env.local.sh"

export ICF_PYTHON PYTHON_BIN ICF_DATA_ROOT OFFICIAL FEATURES ICF_CKPT ICF_CONFIG NGPU GPU_OFFSET
