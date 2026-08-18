#!/usr/bin/env bash
# Does DD benefit from more than one dispersion direction? (docs SS146)
#
# The t-gate this arm was built for is REFUTED as a selector
# (`diagnose_dd_rank_tstat.py`): post-selection inflation puts |t| between 2.3
# and 9.8 for every rank on every fold, and |t| RISES with rank because whitening
# shrinks the within-class scatter of low-|lambda| directions. A |t| gate would
# preferentially admit the directions the operator ranked least discriminative.
#
# So the gate is switched OFF here (ICF_DD_RANK_TSTAT=0) and r is swept as a
# plain fixed knob, which is the question underneath: does rank > 1 help at all?
# One knob per arm (SS127-2).
#
# ⚠️ Distances are SUMMED over directions because that sum is the Gaussian
# discriminant, so `d1 - d0` stays a log-likelihood ratio -- but its magnitude
# then grows with r while the fixed head's coefficients were derived at r=1.
# The `scale8` arm divides by the count kept, as the control for that confound.
#
# Usage: bash scripts/run_dd_rank_sweep.sh <out_dir>
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
PY="${PYTHON_BIN:-/home/aibio_3/miniconda3/envs/BagPFN/bin/python}"
OFFICIAL=/NHNHOME/BASE/kimds/Data/PathoBench/official
FEATURES=/NHNHOME/BASE/kimds/Data/PathoBench/features
CKPT=$(ls checkpoints/20260815_113422/v98_p1_reverse_seed42/periodic-epoch=049*.ckpt)
CONFIG=configs/train_v98_p1_reverse_1536_1gpu.yaml
# GPUs 4-7 host other users' jobs on this node -- default to 0-3.
NGPU=${NGPU:-4}
GPU_OFFSET=${GPU_OFFSET:-0}

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
ARMS="${ARMS:-r1 r2 r4 r8 r16 scale8}"

jobs=()
for arm in $ARMS; do for t in "${TASKS[@]}"; do jobs+=("$arm|$t"); done; done
echo "${#jobs[@]} jobs over $NGPU GPUs"

run_job() {
  local gpu="$1" arm="$2" task="$3"
  local name="${task//\//_}"
  local log="$OUT/${arm}_${name}.log"
  # r1 goes through the UNSET path, so it doubles as proof the hook is a no-op.
  local vars=(ICF_COVARIANCE_BASIS=pca_within ICF_FIXED_HEAD=1 ICF_SKETCH_DIM=256)
  case "$arm" in
    r1)     ;;
    scale8) vars+=(ICF_DD_RANK_MAX=8 ICF_DD_RANK_TSTAT=0 ICF_DD_RANK_SCALE=1) ;;
    # SS147: |lambda| picks the LARGEST dispersion gap, |t| the most CONSISTENT
    # one, and SS146-2 measured that they disagree. `lamt` takes one of each
    # (r=2, so `r2` is its scale-matched control); `tonly` swaps |lambda|'s pick
    # for |t|'s (r=1, so `r1` is its control). The |t| argmax is drawn from
    # |lambda|-ranks 1..15 -- the 2nd..16th directions -- so it can never
    # collapse onto rank 0.
    lamt)   vars+=(ICF_DD_SELECT=lambda_plus_t ICF_DD_TSTAT_RANGE=1:16) ;;
    tonly)  vars+=(ICF_DD_SELECT=tstat ICF_DD_TSTAT_RANGE=1:16) ;;
    r*)     vars+=(ICF_DD_RANK_MAX="${arm#r}" ICF_DD_RANK_TSTAT=0) ;;
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
  run_job "$(((i % NGPU) + GPU_OFFSET))" "$arm" "$t" &
  i=$((i + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$NGPU" ]; do sleep 5; done
done
wait
echo "SWEEP DONE"
