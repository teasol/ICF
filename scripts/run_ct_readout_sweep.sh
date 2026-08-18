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
. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"   # docs SS164: node paths in one place
PY="$PYTHON_BIN"


CKPT="$ICF_CKPT"
CONFIG="$ICF_CONFIG"

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
  local gpu="$1" arm="$2" task="$3" d rest blocks suffix
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
    # SS153: DD weight, swept DOWNWARD from 0.343 on top of v108. dd0 ablates DD.
    v108)                vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge) ;;
    # SS158 candidate: v108 + k-means tokens at weight 0.7 + off-diagonal-only CV.
    # The two live in DIFFERENT branches, but SS150 showed knobs interact, so the
    # combination is measured rather than assumed additive.
    # SS162: CV on the correlation matrix's off-diagonal, on top of v110.
    v110)                vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag) ;;
    # SS165: fit normalisation/tokens from a reproducible random 512-cell subset
    # per bag, then average assignments over EVERY cell.  This separates codebook
    # construction from abundance precision; the suffix is the sampling seed.
    r512all_s*)          rest="${arm##*_s}"
                         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag ICF_CT_CELLS=512
                                ICF_CT_ABUNDANCE_CELLS=all ICF_CT_SAMPLING=random
                                ICF_CT_SAMPLING_SEED="$rest") ;;
    # SS167: retain v110's 64-cell dictionary budget but replace deterministic
    # even spacing with seeded random sampling; abundance still uses every cell.
    r64all_s*)           rest="${arm##*_s}"
                         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag ICF_CT_CELLS=64
                                ICF_CT_ABUNDANCE_CELLS=all ICF_CT_SAMPLING=random
                                ICF_CT_SAMPLING_SEED="$rest") ;;
    # SS166: paired isolation of the abundance population.  Keep the random
    # 512-cell codebook fixed and cap assignment averaging at the same 512 cells.
    r512a512_s*)         rest="${arm##*_s}"
                         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag ICF_CT_CELLS=512
                                ICF_CT_ABUNDANCE_CELLS=512 ICF_CT_SAMPLING=random
                                ICF_CT_SAMPLING_SEED="$rest") ;;
    # SS163: CT ridge lambda sensitivity. CT's ridge is only 32-dimensional now, and
    # SS156-5 showed lambda comes alive at low dimension (+0.012 at 256 dims), so
    # lambda=1 was never checked where it might matter most.
    ctlam*)              vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag
                                ICF_CT_RIDGE_LAMBDA="${arm#ctlam}") ;;
    # SS163-3: same lambda sweep at 16 tokens (v109). If lambda=1 were suboptimal
    # there but optimal at 32, SS160's token comparison would have been unfair.
    t16lam*)             vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=16
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag
                                ICF_CT_RIDGE_LAMBDA="${arm#t16lam}") ;;
    ctolam*)             vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_CV_BLOCKS=offdiag
                                ICF_CT_RIDGE_LAMBDA="${arm#ctolam}"
                                ICF_FIXED_HEAD_CV_WEIGHT=0 ICF_FIXED_HEAD_DD_WEIGHT=0) ;;
    v110_ctonly)         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_CV_BLOCKS=offdiag
                                ICF_FIXED_HEAD_CV_WEIGHT=0 ICF_FIXED_HEAD_DD_WEIGHT=0) ;;
    corr)                vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag ICF_CV_CORR=1) ;;
    corr_cvonly)         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_CV_BLOCKS=offdiag ICF_CV_CORR=1
                                ICF_FIXED_HEAD_DD_WEIGHT=0 ICF_FIXED_HEAD_CT_WEIGHT=0) ;;
    v110_cvonly)         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_CT_TOKENS=32
                                ICF_CT_KMEANS=30 ICF_CV_BLOCKS=offdiag
                                ICF_FIXED_HEAD_DD_WEIGHT=0 ICF_FIXED_HEAD_CT_WEIGHT=0) ;;
    v109)                vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag) ;;
    # SS159: v109 with CT's 64-cell cap removed (every cell). CV/DD untouched.
    v109_fullcell)       vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag ICF_CT_CELLS=all) ;;
    # SS160: token count (= cluster count = CT feature count) at FULL cells.
    # SS160-4: token count at v109's 64-cell setting -- is the token gain conditional
    # on full cells, or does it stand alone? (SS150-4: check the diagonal cell.)
    c64T*)               vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag ICF_CT_TOKENS="${arm#c64T}") ;;
    fcT*)                rest="${arm#fcT}"
                         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag ICF_CT_CELLS=all
                                ICF_CT_TOKENS="${rest%%_*}")
                         case "$rest" in *_lam*) vars+=(ICF_CT_RIDGE_LAMBDA="${rest#*_lam}") ;; esac ;;
    v109_cells*)         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7
                                ICF_CV_BLOCKS=offdiag
                                ICF_CT_CELLS="${arm#v109_cells}") ;;
    # SS159-4: does the cell count matter BEFORE the SS157 k-means fix? Closing the
    # axis only at kmeans=30 would repeat SS148's mistake of closing "readout" using
    # measurements taken at raw 1536 dims (SS150-4).
    fps64)               vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_FIXED_HEAD_CT_WEIGHT=0.7 ICF_CV_BLOCKS=offdiag
                                ICF_CT_KMEANS=0) ;;
    fpsall)              vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_FIXED_HEAD_CT_WEIGHT=0.7 ICF_CV_BLOCKS=offdiag
                                ICF_CT_KMEANS=0 ICF_CT_CELLS=all) ;;
    v109_ctonly)         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CT_KMEANS=30 ICF_FIXED_HEAD_CT_WEIGHT=0.7) ;;
    v109_cvonly)         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CV_BLOCKS=offdiag) ;;
    # SS154: restore the LLR's log-determinant term, log(sigma_0^2/sigma_1^2).
    llr)                 vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_DD_LLR=1) ;;
    # SS155: rank DD by the ratio (D0-D1)/(D0+D1+eps) instead of the difference.
    rel)                 vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_DD_RELATIVE=1) ;;
    # SS157: k-means (Lloyd) refinement of the farthest-point tokens, on v108.
    # SS157-5: k-means tokens x CT weight. SS151 swept the weight on FPS tokens,
    # which the effective-token count says were collapsed onto ~2 of 16 -- so that
    # sweep says nothing about the weight a WORKING CT branch wants.
    km*w*)               rest="${arm#km}"
                         vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CT_KMEANS="${rest%%w*}"
                                ICF_FIXED_HEAD_CT_WEIGHT="${rest#*w}") ;;
    kme*)                vars+=(ICF_CT_PCA_DIM=32 ICF_CT_KMEANS="${arm#kme}") ;;
    km*)                 vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_CT_KMEANS="${arm#km}") ;;
    rel_nocal)           vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge ICF_DD_RELATIVE=1
                                ICF_DD_RELATIVE_CALIBRATE=0) ;;
    # SS156: CV descriptor decomposition, all on top of v108.
    #   cvo_*  = CV-ONLY: DD and CT weights set to 0, so the margin is 1.442*(cv1-cv0)
    #            and AUROC (scale-free) reads the CV branch alone.
    #   *_par  = ICF_CV_BLOCK_NORM=parent, the normalisation control.
    #   *_lam* = ridge lambda, checking SS142-2's inertness at the new dimensions.
    cvo_*|cv_*)
        rest="${arm#cv}"; rest="${rest#o}"; rest="${rest#_}"
        blocks="${rest%%_*}"; suffix="${rest#"$blocks"}"
        vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
               ICF_CV_BLOCKS="${blocks//./+}")
        case "$arm" in cvo_*) vars+=(ICF_FIXED_HEAD_DD_WEIGHT=0 ICF_FIXED_HEAD_CT_WEIGHT=0) ;; esac
        case "$suffix" in
          _par)   vars+=(ICF_CV_BLOCK_NORM=parent) ;;
          _lam*)  vars+=(ICF_RIDGE_LAMBDA="${suffix#_lam}") ;;
        esac ;;
    dd*)                 vars+=(ICF_CT_PCA_DIM=32 ICF_CT_READOUT=ridge
                                ICF_FIXED_HEAD_DD_WEIGHT="${arm#dd}") ;;
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
