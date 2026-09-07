#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Pin the approved v121 configuration, excluding inherited experimental knobs.
for key in ${!ICF_@}; do unset "$key"; done
source scripts/node_env.sh || exit $?
source scripts/lib/arms.sh
icf_arm_v121
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" scripts/analysis/ru81_launch.py "${1:?tag required}"
