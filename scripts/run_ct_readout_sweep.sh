#!/usr/bin/env bash
# CT readout comparison on the OFFICIAL path (docs SS148).
#
# Same tokens, same 16-d abundance (shared `ct_abundance`), only step 6-7 varies:
#   ct_extreme    q1 - q0, today's v107 readout. Runs through the UNSET path, so
#                 it doubles as proof the hook is a no-op.
#   ct_prototype  class prototypes over all 16 dims, squared-distance difference
#   ct_ridge      class-balanced ridge over all 16 dims, logit1 - logit0
#
# The alternatives are calibrated to the extreme margin's CONTEXT mean and RMS
# before the head, so the fixed 0.286 CT weight keeps its meaning and this
# compares readout quality rather than CT's magnitude. `nocal` variants show what
# the uncalibrated magnitude would have done, as the control for that choice.
#
# Deterministic: no seeds (SS139 measured seed std 0.00000).
#
# Usage: bash scripts/run_ct_readout_sweep.sh <out_dir>
#   TASKSET=heldout  ARMS="ct_extreme ct_ridge"  to narrow
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
PY="${PYTHON_BIN:-/home/aibio_3/miniconda3/envs/BagPFN/bin/python}"
OFFICIAL=/NHNHOME/BASE/kimds/Data/PathoBench/official
FEATURES=/NHNHOME/BASE/kimds/Data/PathoBench/features
CKPT=$(ls checkpoints/20260815_113422/v98_p1_reverse_seed42/periodic-epoch=049*.ckpt)
CONFIG=configs/train_v98_p1_reverse_1536_1gpu.yaml
NGPU=${NGPU:-8}

OUT="$1"; shift
mkdir -p "$OUT"
if [ "${TASKSET:-seal}" = "heldout" ]; then
TASKS=(
  cptac_lscc/ARID1A_mutation cptac_lscc/Histologic_Grade cptac_lscc/KEAP1_mutation
  cptac_luad/KRAS_mutation cptac_pda/SMAD4_mutation
  ucla_lung/progression_regression cptac_ccrcc/PBRM1_mutation
)
else
TASKS=(
  bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status
  cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation
  cptac_luad/EGFR_mutation cptac_luad/STK11_mutation cptac_luad/TP53_mutation
  cptac_ccrcc/BAP1_mutation cptac_ccrcc/VHL_mutation
)
fi
ARMS="${ARMS:-ct_extreme ct_prototype ct_ridge ct_prototype_nocal ct_ridge_nocal}"

jobs=()
for arm in $ARMS; do for t in "${TASKS[@]}"; do jobs+=("$arm|$t"); done; done
echo "${#jobs[@]} jobs over $NGPU GPUs"

run_job() {
  local gpu="$1" arm="$2" task="$3" d
  local name="${task//\//_}"
  local log="$OUT/${arm}_${name}.log"
  local vars=(ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1 ICF_SKETCH_DIM=256)
  case "$arm" in
    ct_extreme)          ;;
    ct_prototype)        vars+=(ICF_CT_READOUT=prototype) ;;
    ct_ridge)            vars+=(ICF_CT_READOUT=ridge) ;;
    ct_prototype_nocal)  vars+=(ICF_CT_READOUT=prototype ICF_CT_CALIBRATE=0) ;;
    ct_ridge_nocal)      vars+=(ICF_CT_READOUT=ridge ICF_CT_CALIBRATE=0) ;;
    # SS149: distances in the leading k PCA directions instead of raw 1536-d,
    # reusing the basis the CV branch already built for the fold.
    # SS150: PCA and readout together. SS148 varied the readout at raw 1536 and
    # SS149 varied the dimension at the extreme readout -- one knob each, so the
    # COMBINATION was never run. If relieving concentration puts information into
    # the abundance, the readout only becomes binding after that.
    pca*_ridge)          d="${arm#pca}"; d="${d%_ridge}"
                         vars+=(ICF_CT_PCA_DIM="$d" ICF_CT_READOUT=ridge) ;;
    pca*_proto)          d="${arm#pca}"; d="${d%_proto}"
                         vars+=(ICF_CT_PCA_DIM="$d" ICF_CT_READOUT=prototype) ;;
    pca*)                vars+=(ICF_CT_PCA_DIM="${arm#pca}") ;;
    # SS151: CT weight sweep on top of the (pca32, ridge) configuration. 0.286 was
    # fitted for the OLD readout, so a better CT margin may want more of the head.
    w*)                  vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_FIXED_HEAD_CT_WEIGHT="${arm#w}") ;;
  esac
  CUDA_VISIBLE_DEVICES="$gpu" env "${vars[@]}" "$PY" scripts/test_pathobench.py \
    --checkpoint "$CKPT" --config "$CONFIG" \
    --official-folds "$OFFICIAL/$task" --features "$FEATURES" \
    --input-dim 1536 --precision bf16-mixed \
    --output "$OUT/${arm}_${name}.pt" > "$log" 2>&1
  echo "$arm $name  $(grep -ao 'fold-mean AUROC: [0-9.]*' "$log" | tail -1 || echo FAILED)"
}

i=0
for job in "${jobs[@]}"; do
  arm="${job%%|*}"; t="${job##*|}"
  run_job "$((i % NGPU))" "$arm" "$t" &
  i=$((i + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do sleep 5; done
done
wait
echo "SWEEP DONE"
