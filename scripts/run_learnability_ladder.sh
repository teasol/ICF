#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <A|B|C|C0|C1|C2|C3|C4|C5|C4-N|C4-D|D0|D1|D2|D3|D4>" >&2
    exit 2
fi

requested_stage="$(printf '%s' "$1" | tr '[:upper:]-' '[:lower:]_')"
case "${requested_stage}" in
    a|b|c|c0|c1|c2|c3|c4|c5|c4_n|c4_d|d0|d1|d2|d3|d4) stage="${requested_stage}" ;;
    *) echo "Unknown learnability stage: $1" >&2; exit 2 ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

bagpfn_env="${BAGPFN_CONDA_ENV:-/home/kimds/miniconda3/envs/BagPFN}"
bagpfn_torchrun="${bagpfn_env}/bin/torchrun"
if [[ ! -x "${bagpfn_torchrun}" ]]; then
    echo "BagPFN torchrun executable not found: ${bagpfn_torchrun}" >&2
    echo "Set BAGPFN_CONDA_ENV to the BagPFN Conda environment path." >&2
    exit 2
fi

display_stage="${stage^^}"
display_stage="${display_stage//_/-}"
seed="${SEED:-42}"
run_kind="learnability_${stage}_seed${seed}"
config="configs/train_learnability_${stage}.yaml"
checkpoint_dir="${project_root}/checkpoints/learnability_ladder/${display_stage}/seed_${seed}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}" \
NPROC_PER_NODE="${NPROC_PER_NODE:-4}" \
ICF_CHECKPOINT_DIR="${checkpoint_dir}" \
SEED="${seed}" \
CKPT_PATH= \
TORCHRUN_BIN="${TORCHRUN_BIN:-${bagpfn_torchrun}}" \
    scripts/launch_interactive_training.sh "${run_kind}" "${config}"
