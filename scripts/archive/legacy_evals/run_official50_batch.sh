#!/usr/bin/env bash
# Run the official Patho-Bench 50-fold eval batch (17 tasks) for v34-1536.
#
# Protocol (per docs §53): official k=all.tsv folds, official cohort/labels,
# all-context, raw 1536-d, zero-shot in-context. Output = SEAL-comparable
# 50-fold macro-AUC. Per-fold checkpoints let an interrupted run resume; the
# runner also auto-reduces workers 10 -> 6 -> 4 -> 2 on OOM (identical results).
#
# Resume-friendly: a task whose output file already exists is skipped, so
# re-running this script only processes the remaining tasks.
#
# Usage:  bash scripts/run_official50_batch.sh [--workers-max 10]
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# The interpreter comes from node_env.sh (the project uv venv at ICF/.venv);
# the old hardcoded conda default died with the BagPFN env.
. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"
PY="$PYTHON"
CHECKPOINT="checkpoints/20260806_215800/v34_phase0_largectx_1536/epoch=048-val_ce_loss=0.4419.ckpt"
CONFIG="configs/archive/v34_largectx/train_v34_phase0_largectx_1536.yaml"
FEATURES="/NHNHOME/BASE/kimds/Data/PathoBench/features"
OFFICIAL="/NHNHOME/BASE/kimds/Data/PathoBench/official"
OUTDIR="predictions"
LOG="logs/official50/batch_resume.log"
WORKERS_MAX="${1:-10}"

mkdir -p "$OUTDIR" "$(dirname "$LOG")" /tmp/pathobench_official_workers

# "source/task  display_name"  (17 tasks; display name -> output file suffix)
TASKS=(
  "bc_therapy/er_status             bc_therapy_er_status"
  "bc_therapy/grade                 bc_therapy_grade"
  "bc_therapy/her2_status           bc_therapy_her2_status"
  "cptac_brca/PIK3CA_mutation       cptac_brca_PIK3CA_mutation"
  "cptac_brca/TP53_mutation         cptac_brca_TP53_mutation"
  "cptac_lscc/ARID1A_mutation       cptac_lscc_ARID1A_mutation"
  "cptac_lscc/Histologic_Grade      cptac_lscc_Histologic_Grade"
  "cptac_lscc/KEAP1_mutation        cptac_lscc_KEAP1_mutation"
  "cptac_luad/EGFR_mutation         cptac_luad_EGFR_mutation"
  "cptac_luad/KRAS_mutation         cptac_luad_KRAS_mutation"
  "cptac_luad/STK11_mutation        cptac_luad_STK11_mutation"
  "cptac_luad/TP53_mutation         cptac_luad_TP53_mutation"
  "cptac_pda/SMAD4_mutation         cptac_pda_SMAD4_mutation"
  "ucla_lung/progression_regression ucla_lung_progression_regression"
  "cptac_ccrcc/BAP1_mutation        cptac_ccrcc_BAP1_mutation"
  "cptac_ccrcc/PBRM1_mutation       cptac_ccrcc_PBRM1_mutation"
  "cptac_ccrcc/VHL_mutation         cptac_ccrcc_VHL_mutation"
)

for entry in "${TASKS[@]}"; do
  read -r src_task name <<< "$entry"
  task_dir="$OFFICIAL/$src_task"
  out="$OUTDIR/pathobench_${name}_v34_1536_official50.pt"

  if [[ -f "$out" ]]; then
    echo "=== SKIP ${name} (output exists) $(date) ===" | tee -a "$LOG"
    continue
  fi
  echo "=== START ${name} $(date) ===" | tee -a "$LOG"

  ok=0
  for w in "$WORKERS_MAX" 6 4 2; do
    echo "  [${name}] try workers=$w $(date)" | tee -a "$LOG"
    if "$PY" scripts/run_official_folds_parallel.py \
        --checkpoint "$CHECKPOINT" \
        --config "$CONFIG" \
        --task-dir "$task_dir" \
        --features "$FEATURES" \
        --workers "$w" \
        --output "$out" >> "$LOG" 2>&1; then
      echo "  [${name}] OK workers=$w" | tee -a "$LOG"
      ok=1
      break
    fi
    echo "  [${name}] FAIL workers=$w rc=$? -> retry" | tee -a "$LOG"
  done
  echo "=== END ${name} ok=$ok $(date) ===" | tee -a "$LOG"
  if [[ "$ok" -eq 0 ]]; then
    echo "  !! ${name} FAILED at all worker counts -- aborting batch" | tee -a "$LOG"
    exit 1
  fi
done

echo "ALL DONE $(date)" | tee -a "$LOG"
