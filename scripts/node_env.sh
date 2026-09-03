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
#                  (default: the project's uv venv at ICF/.venv, created with
#                   `uv venv --python 3.12 .venv` and populated from
#                   requirements.txt -- see docs SS206)
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
# ICF_ROOT is derived from this file's own location, so the venv is found no
# matter which directory a runner is invoked from -- and no matter where the
# repo is mounted. The conda candidates below are kept only as a fallback for
# nodes that still carry a BagPFN env; the project environment is the uv venv.
ICF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${ICF_PYTHON:-}" ]; then
  for candidate in \
    "${PYTHON_BIN:-}" \
    "$ICF_ROOT/.venv/bin/python" \
    /home/aibio_3/miniconda3/envs/BagPFN/bin/python \
    "$HOME/miniconda3/envs/BagPFN/bin/python" \
    "$HOME/miniforge3/envs/BagPFN/bin/python" \
    "$(command -v python3 || true)"
  do
    [ -n "$candidate" ] && [ -x "$candidate" ] || continue
    # lightning is the import that fails on a bare python3 and silently drops 14
    # test modules while still printing a confident "Ran 158 tests" (SS141-3).
    # This node's /usr/bin/python3 is exactly that trap: it HAS torch but not
    # lightning, so both names have to be checked.
    #
    # find_spec, not `import`: resolving the two names costs 0.03s where
    # importing them costs 11.1s (lightning alone is ~7s), and node_env.sh is
    # sourced by every runner -- twice per eval, since eval_seal_tasks.sh
    # sources it again. It was 10.6s of a 37s test-suite run (SS207).
    # The trade-off: find_spec proves a module is installed and importable by
    # name, not that executing it succeeds. A module present but broken at
    # import time now gets selected and fails loudly with a real traceback at
    # first use, instead of being skipped in favour of the next candidate --
    # which, on this node, would fail the same way one step later.
    if "$candidate" -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('torch') and u.find_spec('lightning') else 1)" >/dev/null 2>&1; then
      ICF_PYTHON="$candidate"; break
    fi
  done
fi
if [ -z "${ICF_PYTHON:-}" ]; then
  echo "node_env: no interpreter with torch+lightning found." >&2
  echo "  Create the project env:  cd $ICF_ROOT && uv venv --python 3.12 .venv" >&2
  echo "                           uv pip install -r requirements.txt" >&2
  echo "  Or point ICF_PYTHON at an existing interpreter." >&2
  return 1 2>/dev/null || exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-$ICF_PYTHON}"

# ---- data -----------------------------------------------------------------
if [ -z "${ICF_DATA_ROOT:-}" ]; then
  for candidate in \
    /NHNHOME/BASE/kimds/Data/PathoBench \
    /lustre/BASE/kimds/Data/PathoBench \
    "$HOME/Data/PathoBench" \
    "$HOME/ICF/data/repro_labels_folds"
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

PY="${PYTHON_BIN:-$ICF_PYTHON}"
PYTHON="${PYTHON_BIN:-$ICF_PYTHON}"
export ICF_ROOT ICF_PYTHON PYTHON_BIN PY PYTHON ICF_DATA_ROOT OFFICIAL FEATURES ICF_CKPT ICF_CONFIG NGPU GPU_OFFSET
