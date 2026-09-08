#!/usr/bin/env bash
# RU-87: bf16 vs fp32 precision comparison, Arm A (5-branch baseline, v121) and
# Arm B (Arm A + SHJ, screen-only so the ensemble output is untouched -- same
# convention as scripts/run_v121_sh_variants.sh).
#
# Why this script exists instead of reusing scripts/eval_seal_tasks.sh: that
# wrapper hardcodes `--precision bf16-mixed` (line 35) and has no override, so
# it cannot produce the fp32 arm this RU needs. Rather than touch the shared
# wrapper (used by every other v1* runner), this script duplicates its
# task loop with --precision as a parameter. Precedent: an earlier ARID1A-only
# fp32 check (predictions/pathobench_cptac_lscc_ARID1A_mutation_shj_fp32_check_*)
# was produced the same way, by calling scripts/test_pathobench.py directly.
#
# m_shj is forced to float32 internally regardless of this flag
# (src/models/branches/shj.py, autocast(enabled=False) -- docs/history/archive.md
# SS222), so this script's bf16-vs-fp32 contrast measures the OTHER branches
# (CV/BM/BD/QA/DS) and any bf16 exposure in the rest of the pipeline, not SHJ's
# own math.
#
# Usage: bash scripts/run_ru87_precision.sh <bf16-mixed|32-true> <tag> [gpu_list]
#   gpu_list: space-separated CUDA device indices to use, default "1 2 3 4 6 7"
#             (GPUs 0 and 5 on nexgem-s1 are occupied by another job -- verified
#             via nvidia-smi 2026-09-08, STP-Bench PIDs at 96-100% util).
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$PROJECT_ROOT"
. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
. "$(dirname "${BASH_SOURCE[0]}")/lib/arms.sh"

PRECISION="${1:?usage: run_ru87_precision.sh <bf16-mixed|32-true> <tag> [gpu_list]}"
TAG="${2:?usage: run_ru87_precision.sh <bf16-mixed|32-true> <tag> [gpu_list]}"
shift 2
GPU_LIST=("$@")
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
  GPU_LIST=(1 2 3 4 6 7)
fi
NG="${#GPU_LIST[@]}"

CONFIG="configs/archive/v94_v102_cell_value/train_v98_p1_reverse_1536_1gpu.yaml"

# ---- v121 5-branch config (verbatim from scripts/eval_v121.sh) ------------
icf_arm_v121

# ---- SHJ screen-only, same knobs as scripts/run_v121_sh_variants.sh -------
export ICF_FIXED_HEAD_SH_WEIGHT=1.0
export ICF_FIXED_HEAD_BS_WEIGHT=0.0     # BS rejected in SS218
export ICF_SH_DIM="${ICF_SH_DIM:-32}"
export ICF_SH_WIDE="${ICF_SH_WIDE:-256}"
export ICF_SHAPE_SCREEN_ONLY=1

TASKS=(cptac_lscc/ARID1A_mutation cptac_lscc/Histologic_Grade cptac_lscc/KEAP1_mutation \
       cptac_luad/KRAS_mutation cptac_pda/SMAD4_mutation ucla_lung/progression_regression \
       cptac_ccrcc/PBRM1_mutation)

mkdir -p logs/official50 predictions
echo ">>> RU-87 | precision=${PRECISION} tag=${TAG} gpus=(${GPU_LIST[*]}) $(date '+%F %T')"

overall_rc=0
declare -a pids=()
declare -a pid_task=()

launch_one() {
  local task="$1" gpu="$2"
  local name="${task//\//_}"
  local out="predictions/pathobench_${name}_${TAG}_official50_bf16.pt"
  local log="logs/official50/${name}_${TAG}.log"
  echo "=== LAUNCH ${task} gpu=${gpu} precision=${PRECISION} $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/test_pathobench.py \
    --config "$CONFIG" \
    --official-folds "$OFFICIAL/$task" --features "$FEATURES" \
    --input-dim 1536 --precision "$PRECISION" --output "$out" > "$log" 2>&1 &
  pids+=($!)
  pid_task+=("$task")
}

idx=0
n_running=0
for task in "${TASKS[@]}"; do
  gpu="${GPU_LIST[$((idx % NG))]}"
  launch_one "$task" "$gpu"
  idx=$((idx + 1))
  n_running=$((n_running + 1))
  if [ "$n_running" -ge "$NG" ]; then
    # Fill exactly NG slots, then drain before starting more (avoids two tasks
    # sharing one GPU concurrently -- fp32 uses more VRAM than bf16, kill ①).
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || overall_rc=$?
    done
    pids=(); pid_task=(); n_running=0
  fi
done
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || overall_rc=$?
done

echo "--- per-task result (precision=${PRECISION}, tag=${TAG}) ---"
for task in "${TASKS[@]}"; do
  name="${task//\//_}"
  log="logs/official50/${name}_${TAG}.log"
  res=$(grep -aoE "fold-mean AUROC: [0-9.]+ ± [0-9.]+   pooled AUROC: [0-9.]+" "$log" 2>/dev/null | tail -1)
  echo "=== END   ${task} ${res:-$(tail -2 "$log" 2>/dev/null)}"
done
echo ">>> RU-87 | precision=${PRECISION} tag=${TAG} DONE rc=${overall_rc} $(date '+%F %T')"
exit "$overall_rc"
