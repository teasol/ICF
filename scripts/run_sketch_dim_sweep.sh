#!/usr/bin/env bash
# K (covariance sketch dim) sweep on the OFFICIAL SEAL path (docs SS142).
#
# Only runnable because v106 is training-free: K used to be the output width of
# the learned projection P, so moving it meant retraining. With per-episode PCA
# and a constant head, K is an evaluation-time knob and `_covariance_projection`
# is the only checkpoint tensor that depends on it (dropped by ICF_SKETCH_DIM).
#
# lambda is NOT swept alongside K. Standardisation makes the dual Gram grow with
# the descriptor length (measured 3.519x from K=128 to K=256,
# `scripts/diagnose_sketch_dim_scale.py`), which would normally bundle a second
# knob into the comparison -- but lambda in [0.01, 3.519] gives a BIT-IDENTICAL
# fold-mean at both K, i.e. the ridge is inert in that whole range and only
# starts biting near 1e4. So K at fixed lambda=1.0 is a genuine single-knob arm.
#
# Deterministic: no seed repetition (SS139 measured seed std 0.00000).
#
# Usage: bash scripts/run_sketch_dim_sweep.sh <out_dir> <K>...
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
# TASKSET=seal   the 10 in_seal=yes rows -- the judging set
# TASKSET=heldout the other 7 rows of seal_univ2_baseline_17tasks.csv. They have
#                 no published SEAL number, which is why they are not used for
#                 judging -- and exactly why they are clean HELD-OUT data for
#                 confirming a K picked on the SEAL 10 (SS131-5: replication in an
#                 independent group outranks one t).
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
for k in "$@"; do for t in "${TASKS[@]}"; do jobs+=("$k|$t"); done; done
echo "${#jobs[@]} jobs over $NGPU GPUs"

run_job() {
  local gpu="$1" k="$2" task="$3"
  local name="${task//\//_}"
  local log="$OUT/K${k}_${name}.log"
  # K=128 is the v106 baseline: run it through the UNSET path so the arm also
  # proves the new env hooks are a no-op when absent.
  local sketch=(); [ "$k" != "128" ] && sketch=(ICF_SKETCH_DIM="$k")
  CUDA_VISIBLE_DEVICES="$gpu" env ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1 \
    "${sketch[@]}" "$PY" scripts/test_pathobench.py \
    --checkpoint "$CKPT" --config "$CONFIG" \
    --official-folds "$OFFICIAL/$task" --features "$FEATURES" \
    --input-dim 1536 --precision bf16-mixed \
    --output "$OUT/K${k}_${name}.pt" > "$log" 2>&1
  echo "K=$k $name  $(grep -ao 'fold-mean AUROC: [0-9.]*' "$log" | tail -1 || echo FAILED)"
}

i=0
for job in "${jobs[@]}"; do
  k="${job%%|*}"; t="${job##*|}"
  run_job "$((i % NGPU))" "$k" "$t" &
  i=$((i + 1))
  # Keep at most NGPU in flight so each job owns a card.
  while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do sleep 5; done
done
wait
echo "SWEEP DONE"
