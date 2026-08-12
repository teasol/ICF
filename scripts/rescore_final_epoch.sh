#!/usr/bin/env bash
# Re-score every 10-task verdict at its run's FINAL epoch (SS104 rule).
#
# The audit in SS105 found 27 of 36 historical verdicts were scored on a
# validation-best checkpoint from the middle of training, not the final epoch.
# Worst cases: ridge calibration at epoch 27, mlpbank2048/4096 at 28/27, and
# latent4/8 at 27/37 -- so the mlpbank "peak at M=1024" shape and the latent
# sweep non-monotonicity may be training-length artifacts rather than capacity
# effects. This re-scores each arm from its epoch-49 checkpoint (epoch 199 for
# the 200-epoch combo run) so the whole table shares one selection rule.
#
# Not re-scorable, and why:
#   large-ragged warm-start -- the run early-stopped at epoch 34, no epoch 49 exists
#   v41/v42/v43/v44/v45 (CV-only) -- pre-prune checkpoints do not load in this
#     tree (SS73); they need the 8caa96c worktree at /NHNHOME/BASE/kimds/ICF_pre_prune
#   v43_notanh / v44_lowT -- no artifacts at all, not in logs/ or predictions/
#
# Usage: bash scripts/rescore_final_epoch.sh [gpu-count]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NGPU="${1:-4}"
ARCH=configs/archive/v69_v76_relation
TASKS=(bc_therapy/er_status bc_therapy/grade bc_therapy/her2_status
       cptac_brca/PIK3CA_mutation cptac_brca/TP53_mutation
       cptac_luad/EGFR_mutation cptac_luad/STK11_mutation cptac_luad/TP53_mutation
       cptac_ccrcc/BAP1_mutation cptac_ccrcc/VHL_mutation)

# tag | checkpoint glob | config
JOBS=(
"v76_hard_ridge_calibration_ep49|checkpoints/20260812_v76_hard_ridge_calibration/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_ridge_calibration_1536.yaml"
"v78_dd_projection_ep49|checkpoints/20260812_v78_dd_projection/periodic-epoch=049-*.ckpt|configs/train_v78_dd_projection_1536.yaml"
"v78_dd_projection_unweighted_ep49|checkpoints/20260812_v78_dd_projection_unweighted/periodic-epoch=049-*.ckpt|configs/train_v78_dd_projection_unweighted_1536.yaml"
"v79_dual_projection_ep49|checkpoints/20260812_v79_dual_projection/periodic-epoch=049-*.ckpt|configs/train_v79_dual_projection_1536.yaml"
"v76_hard_latent2_ep49|checkpoints/20260812_v76_hard_latent_sweep/latent2/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_latent2_1536.yaml"
"v76_hard_latent4_ep49|checkpoints/20260812_v76_hard_latent_sweep/latent4/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_latent4_1536.yaml"
"v76_hard_latent8_ep49|checkpoints/20260812_v76_hard_latent_sweep/latent8/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_latent8_1536.yaml"
"v76_hard_latent16_ep49|checkpoints/20260812_v76_hard_latent_sweep/latent16/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_latent16_1536.yaml"
"v76_hard_mlpbank128_ep49|checkpoints/20260812_v76_hard_mlpbank_sweep/mlpbank128/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_mlpbank128_1536.yaml"
"v76_hard_mlpbank512_ep49|checkpoints/20260812_v76_hard_mlpbank_sweep/mlpbank512/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_mlpbank512_1536.yaml"
"v76_hard_mlpbank1024_ep49|checkpoints/20260812_v76_hard_mlpbank_sweep/mlpbank1024/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_mlpbank1024_1536.yaml"
"v76_hard_mlpbank2048_ep49|checkpoints/20260812_v76_hard_mlpbank_sweep/mlpbank2048/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_mlpbank2048_1536.yaml"
"v76_hard_mlpbank4096_ep49|checkpoints/20260812_v76_hard_mlpbank_sweep/mlpbank4096/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_mlpbank4096_1536.yaml"
"v76_hard_mixed50_mlpbank1024_ep49|checkpoints/20260812_v76_hard_mixed_linear_mlpbank1024/periodic-epoch=049-*.ckpt|$ARCH/train_v76_hard_mixed_linear_mlpbank1024_1536.yaml"
"v76_axis_noise_ep49|checkpoints/20260812_v76_axis_sweep/noise/periodic-epoch=049-*.ckpt|$ARCH/train_v76_axis_noise_medium_1536.yaml"
"v76_axis_rare_ep49|checkpoints/20260812_v76_axis_sweep/rare/periodic-epoch=049-*.ckpt|$ARCH/train_v76_axis_rare_medium_1536.yaml"
"v76_axis_classsep_ep49|checkpoints/20260812_v76_axis_sweep/classsep/periodic-epoch=049-*.ckpt|$ARCH/train_v76_axis_classsep_medium_1536.yaml"
"v76_axis_response_ep49|checkpoints/20260812_v76_axis_sweep/response/periodic-epoch=049-*.ckpt|$ARCH/train_v76_axis_response_medium_1536.yaml"
"v76_classsep_mild_ep49|checkpoints/20260812_v76_classsep_sweep/mild/periodic-epoch=049-*.ckpt|$ARCH/train_v76_classsep_mild_1536.yaml"
"v76_classsep_veryhard_ep49|checkpoints/20260812_v76_classsep_sweep/veryhard/periodic-epoch=049-*.ckpt|$ARCH/train_v76_classsep_veryhard_1536.yaml"
"v76_combo_classsep_rare_noise_8gpu_ep199|checkpoints/20260812_v76_combo/classsep_rare_noise_8gpu/periodic-epoch=199-*.ckpt|$ARCH/train_v76_combo_classsep_rare_noise_8gpu_200ep_1536.yaml"
"v69_mlp_ep49|checkpoints/20260811_152806/v69_cv_dd_mlp_1pop_linear/periodic-epoch=049-*.ckpt|$ARCH/train_v69_cv_dd_mlp_1pop_linear_1536.yaml"
"v71_cv_mlp_ep49|checkpoints/20260811_155038/v71_cv_mlp_1pop_linear/periodic-epoch=049-*.ckpt|$ARCH/train_v71_cv_mlp_1pop_linear_1536.yaml"
"v72_mlp1_ep49|checkpoints/20260811_161405/v72_cv_dd_mlp_1pop_mlp1/periodic-epoch=049-*.ckpt|$ARCH/train_v72_cv_dd_mlp_1pop_mlp1_1536.yaml"
"v73_magnitude_ep49|checkpoints/20260811_170107/v73_cv_dd_magnitude_mlp_1pop_linear/periodic-epoch=049-*.ckpt|$ARCH/train_v73_cv_dd_magnitude_mlp_1pop_linear_1536.yaml"
"v75_cv2_ep49|checkpoints/20260811_180756/v75_cv_cv2_dd_ct_mlp_1pop_linear/periodic-epoch=049-*.ckpt|$ARCH/train_v75_cv_cv2_dd_ct_mlp_1pop_linear_1536.yaml"
"v76_learnable_p_ep49|checkpoints/20260811_200356/v76_cv_learnable_p_dd_ct_mlp_1pop_linear/periodic-epoch=049-*.ckpt|$ARCH/train_v76_cv_learnable_p_dd_ct_mlp_1pop_linear_1536.yaml"
)

OUT=logs/20260812_rescore_ep49
mkdir -p "$OUT"

# Resolve every path up front: a missing checkpoint or config must be a loud
# skip in the manifest, not a silent hole in the results table.
QUEUE=()
: >"$OUT/manifest.txt"
for job in "${JOBS[@]}"; do
  IFS='|' read -r tag glob cfg <<<"$job"
  ck=$(ls $glob 2>/dev/null | head -1)
  if [[ -z "$ck" ]]; then echo "SKIP $tag: no checkpoint matching $glob" | tee -a "$OUT/manifest.txt"; continue; fi
  if [[ ! -f "$cfg" ]]; then echo "SKIP $tag: no config $cfg" | tee -a "$OUT/manifest.txt"; continue; fi
  echo "$tag|$ck|$cfg" >>"$OUT/manifest.txt"
  QUEUE+=("$tag|$ck|$cfg")
done
echo "queued ${#QUEUE[@]} arms across $NGPU GPUs" | tee -a "$OUT/manifest.txt"

worker() {  # $1 = gpu index
  local gpu="$1" i=$1
  while (( i < ${#QUEUE[@]} )); do
    IFS='|' read -r tag ck cfg <<<"${QUEUE[$i]}"
    echo "=== [gpu $gpu] START $tag $(date +%H:%M:%S)"
    bash scripts/eval_seal_tasks.sh "$gpu" "$ck" "$cfg" "$tag" "${TASKS[@]}" \
      >"$OUT/${tag}.out" 2>&1
    local n
    n=$(ls predictions/*_"${tag}"_official50_bf16.pt 2>/dev/null | wc -l)
    echo "=== [gpu $gpu] END   $tag $(date +%H:%M:%S) predictions=$n/10"
    (( i += NGPU ))
  done
}

for ((g=0; g<NGPU; g++)); do worker "$g" & done
wait
echo "=== ALL DONE $(date +%H:%M:%S)"
