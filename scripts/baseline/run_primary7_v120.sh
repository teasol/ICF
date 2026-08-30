#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

show_usage() {
  cat << 'USAGE_EOF'
Usage: bash scripts/baseline/run_primary7_v120.sh [options]

Options:
  --tag <str>        Run tag (default: baseline_v120_primary7)
  --gpus <csv>       Comma-separated GPU indices (default: 0,1,2,3)
  --manifest <path>  Manifest output path (default: results/baseline/manifest.json)
  --dry-run          Build and write manifest, print per-GPU command lines, exit 0
  --preflight        Run environment preflight checks and exit
  --help             Show this help message and exit 0
USAGE_EOF
}

# Handle --help early for fast response
for arg in "$@"; do
  if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
    show_usage
    exit 0
  fi
done

. scripts/node_env.sh || true

TAG="baseline_v120_primary7"
GPUS_ARG="0,1,2,3"
MANIFEST="results/baseline/manifest.json"
DRY_RUN=0
PREFLIGHT=0

PRIMARY_TASKS=(
  "cptac_lscc/ARID1A_mutation"
  "cptac_lscc/Histologic_Grade"
  "cptac_lscc/KEAP1_mutation"
  "cptac_luad/KRAS_mutation"
  "cptac_pda/SMAD4_mutation"
  "ucla_lung/progression_regression"
  "cptac_ccrcc/PBRM1_mutation"
)

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag)
      if [ "$#" -lt 2 ]; then
        echo "Error: --tag requires an argument" >&2
        show_usage >&2
        exit 2
      fi
      TAG="$2"
      shift 2
      ;;
    --gpus)
      if [ "$#" -lt 2 ]; then
        echo "Error: --gpus requires an argument" >&2
        show_usage >&2
        exit 2
      fi
      GPUS_ARG="$2"
      shift 2
      ;;
    --manifest)
      if [ "$#" -lt 2 ]; then
        echo "Error: --manifest requires an argument" >&2
        show_usage >&2
        exit 2
      fi
      MANIFEST="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --preflight)
      PREFLIGHT=1
      shift
      ;;
    --help|-h)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_usage >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a raw_gpus <<< "$GPUS_ARG"
GPU_ARRAY=()
for g in "${raw_gpus[@]}"; do
  g_trimmed="$(echo "$g" | xargs)"
  if [ -n "$g_trimmed" ]; then
    GPU_ARRAY+=("$g_trimmed")
  fi
done

if [ ${#GPU_ARRAY[@]} -eq 0 ]; then
  echo "Error: --gpus must contain at least one GPU index" >&2
  exit 2
fi

if [ "$PREFLIGHT" -eq 1 ]; then
  all_pass=1

  # 1. "$ICF_PYTHON" -c "import torch, lightning" succeeds
  if "${ICF_PYTHON:-python3}" -c "import torch, lightning" >/dev/null 2>&1; then
    echo "PASS: python torch+lightning (${ICF_PYTHON:-python3})"
  else
    echo "FAIL: python torch+lightning (${ICF_PYTHON:-python3})"
    all_pass=0
  fi

  # 2. "$OFFICIAL" directory exists and contains a subdirectory for each of the 5 distinct dataset prefixes
  prefixes=(cptac_lscc cptac_luad cptac_pda ucla_lung cptac_ccrcc)
  missing_dirs=()
  if [ -n "${OFFICIAL:-}" ] && [ -d "$OFFICIAL" ]; then
    for p in "${prefixes[@]}"; do
      if [ ! -d "$OFFICIAL/$p" ]; then
        missing_dirs+=("$p")
      fi
    done
  else
    missing_dirs=("${prefixes[@]}")
  fi

  if [ ${#missing_dirs[@]} -eq 0 ]; then
    echo "PASS: OFFICIAL directories found for all 5 dataset prefixes ($OFFICIAL)"
  else
    echo "FAIL: OFFICIAL missing prefixes: ${missing_dirs[*]} in ($OFFICIAL)"
    all_pass=0
  fi

  # 3. "$FEATURES" directory exists and is non-empty
  if [ -n "${FEATURES:-}" ] && [ -d "$FEATURES" ] && [ -n "$(ls -A "$FEATURES" 2>/dev/null)" ]; then
    echo "PASS: FEATURES directory exists and is non-empty ($FEATURES)"
  else
    echo "FAIL: FEATURES directory missing or empty (${FEATURES:-})"
    all_pass=0
  fi

  # 4. nvidia-smi lists at least as many GPUs as --gpus requests
  detected_gpus="$(nvidia-smi -L 2>/dev/null | wc -l)"
  req_gpus="${#GPU_ARRAY[@]}"
  if [ "${detected_gpus:-0}" -ge "$req_gpus" ]; then
    echo "PASS: nvidia-smi detected $detected_gpus GPUs (requested $req_gpus)"
  else
    echo "FAIL: nvidia-smi detected $detected_gpus GPUs (requested $req_gpus)"
    all_pass=0
  fi

  if [ "$all_pass" -eq 1 ]; then
    exit 0
  else
    exit 1
  fi
fi

# Write manifest before launching any GPU work
mkdir -p "$(dirname "$MANIFEST")"

"${ICF_PYTHON:-python3}" - "$TAG" "$MANIFEST" "$GPUS_ARG" "${PRIMARY_TASKS[@]}" << 'PY'
import sys
import json

tag = sys.argv[1]
manifest_path = sys.argv[2]
gpus_raw = sys.argv[3].split(",")
tasks_raw = sys.argv[4:]

gpus = []
for g in gpus_raw:
    g_clean = g.strip()
    if g_clean:
        if g_clean.isdigit():
            gpus.append(int(g_clean))
        else:
            gpus.append(g_clean)

task_entries = []
num_gpus = len(gpus)
for i, task in enumerate(tasks_raw):
    gpu = gpus[i % num_gpus]
    task_clean = task.replace("/", "_")
    log = f"logs/official50/{task_clean}_{tag}.log"
    pred = f"predictions/pathobench_{task_clean}_{tag}_official50_bf16.pt"
    task_entries.append({
        "task": task,
        "gpu": gpu,
        "log": log,
        "predictions": pred
    })

manifest = {
    "tag": tag,
    "arm": "v120",
    "gpus": gpus,
    "tasks": task_entries
}

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
PY

num_gpus="${#GPU_ARRAY[@]}"

if [ "$DRY_RUN" -eq 1 ]; then
  for (( g_idx=0; g_idx<num_gpus; g_idx++ )); do
    gpu="${GPU_ARRAY[$g_idx]}"
    gpu_tasks=()
    for (( t_idx=0; t_idx<${#PRIMARY_TASKS[@]}; t_idx++ )); do
      if [ $(( t_idx % num_gpus )) -eq "$g_idx" ]; then
        gpu_tasks+=("${PRIMARY_TASKS[$t_idx]}")
      fi
    done
    if [ ${#gpu_tasks[@]} -gt 0 ]; then
      echo "bash scripts/eval_v120.sh $gpu $TAG ${gpu_tasks[*]}"
    fi
  done
  exit 0
fi

# Run GPU jobs in background
mkdir -p logs/baseline

pids=()
for (( g_idx=0; g_idx<num_gpus; g_idx++ )); do
  gpu="${GPU_ARRAY[$g_idx]}"
  gpu_tasks=()
  for (( t_idx=0; t_idx<${#PRIMARY_TASKS[@]}; t_idx++ )); do
    if [ $(( t_idx % num_gpus )) -eq "$g_idx" ]; then
      gpu_tasks+=("${PRIMARY_TASKS[$t_idx]}")
    fi
  done
  if [ ${#gpu_tasks[@]} -gt 0 ]; then
    bash scripts/eval_v120.sh "$gpu" "$TAG" "${gpu_tasks[@]}" > "logs/baseline/primary7_gpu${gpu}.log" 2>&1 &
    pids+=($!)
  fi
done

overall_rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=$?
    if [ "$overall_rc" -eq 0 ]; then
      overall_rc=$rc
    fi
  fi
done

exit "$overall_rc"
