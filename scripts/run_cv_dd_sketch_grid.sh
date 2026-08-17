#!/usr/bin/env bash
# Attribute SS142's K effect to a BRANCH: sweep K_cv x K_dd (docs SS145).
#
# SS142 moved K from 128 to 256 and gained +0.0081, but K enters two branches at
# once, so that number is not yet attributable:
#   CV  descriptor is triu(B^T C_bag B) -- K sets its length (9,792 -> 34,432)
#   DD  rebuilds those K x K matrices from the same triangle, then takes a rank-1
#       dispersion direction through two eigh's
#   CT  selects on raw cells -- basis-free, K cannot reach it
#
# ⚠️ STRUCTURAL CONSTRAINT: K_dd <= K_cv. DD does not project cells itself, it
# reads a sub-block of CV's triangle (`_covariance_matrices_from_triangle`), so
# "DD with more directions than CV" is not a configuration this architecture can
# express -- it is not a limitation of the hook. The grid is lower-triangular and
# the decomposition is therefore sequential:
#     total      = (256,256) - (128,128)
#     CV alone   = (256,128) - (128,128)
#     DD given CV= (256,256) - (256,128)
#
# Slicing is EXACT: PCA eigenvectors are sorted by descending eigenvalue, so the
# top-left k x k block of B^T C B at K equals B^T C B built at k. No second eigh,
# no approximation. Verified: ICF_SKETCH_DIM_DD=256 at K_cv=256 reproduces the
# unset run bit for bit (0.8069 on brca TP53, 5 folds).
#
# Usage: bash scripts/run_cv_dd_sketch_grid.sh <out_dir> <Kcv:Kdd>...
#   e.g. bash scripts/run_cv_dd_sketch_grid.sh out 256:256 256:128 128:128
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
# TASKSET=heldout: the 7 non-SEAL rows, never used to choose anything (SS142-4).
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

jobs=()
for cell in "$@"; do for t in "${TASKS[@]}"; do jobs+=("$cell|$t"); done; done
echo "${#jobs[@]} jobs over $NGPU GPUs"

run_job() {
  local gpu="$1" kcv="$2" kdd="$3" task="$4"
  local name="${task//\//_}"
  local log="$OUT/cv${kcv}_dd${kdd}_${name}.log"
  # K_cv=128 with no DD override is the v106 arm and runs through the UNSET path.
  local vars=(ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1)
  [ "$kcv" != "128" ] && vars+=(ICF_SKETCH_DIM="$kcv")
  [ "$kdd" != "$kcv" ] && vars+=(ICF_SKETCH_DIM_DD="$kdd")
  CUDA_VISIBLE_DEVICES="$gpu" env "${vars[@]}" "$PY" scripts/test_pathobench.py \
    --checkpoint "$CKPT" --config "$CONFIG" \
    --official-folds "$OFFICIAL/$task" --features "$FEATURES" \
    --input-dim 1536 --precision bf16-mixed \
    --output "$OUT/cv${kcv}_dd${kdd}_${name}.pt" > "$log" 2>&1
  echo "cv=$kcv dd=$kdd $name  $(grep -ao 'fold-mean AUROC: [0-9.]*' "$log" | tail -1 || echo FAILED)"
}

i=0
for job in "${jobs[@]}"; do
  cell="${job%%|*}"; t="${job##*|}"
  run_job "$((i % NGPU))" "${cell%%:*}" "${cell##*:}" "$t" &
  i=$((i + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do sleep 5; done
done
wait
echo "GRID DONE"
