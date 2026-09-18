#!/usr/bin/env python3
"""Run SEAL 10-Task Evaluation with evaluate_pure.py across GPUs.

Evaluates the active 7-branch configuration (configs/baseline/v121_7branch_active.yaml)
across all 10 SEAL hold-out tasks, comparing against:
  1. Historical v120 (5-branch) baseline on the same SEAL tasks
  2. Published SEAL supervised benchmarks (ABMIL and MeanMIL) from docs/seal_univ2_baseline_17tasks.csv
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEAL_TASKS = [
    "bc_therapy/er_status",
    "bc_therapy/grade",
    "bc_therapy/her2_status",
    "cptac_brca/PIK3CA_mutation",
    "cptac_brca/TP53_mutation",
    "cptac_luad/EGFR_mutation",
    "cptac_luad/STK11_mutation",
    "cptac_luad/TP53_mutation",
    "cptac_ccrcc/BAP1_mutation",
    "cptac_ccrcc/VHL_mutation",
]

CONFIG = PROJECT_ROOT / "configs/baseline/v121_7branch_active.yaml"
FEATURES = PROJECT_ROOT / "data/repro_labels_folds/features"
OFFICIAL = PROJECT_ROOT / "data/repro_labels_folds/official"
LOG_DIR = PROJECT_ROOT / "logs/seal10_v121_7branch"
PRED_DIR = PROJECT_ROOT / "predictions/seal10_pure"


def load_seal_published_benchmarks() -> dict[str, dict[str, float]]:
    csv_path = PROJECT_ROOT / "docs/seal_univ2_baseline_17tasks.csv"
    benchmarks = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("in_seal", "").lower() == "yes":
                task_key = f"{row['pathobench_dataset']}/{row['pathobench_task']}"
                benchmarks[task_key] = {
                    "abmil": float(row["abmil_mean"]),
                    "abmil_std": float(row["abmil_std"]),
                    "meanmil": float(row["meanmil_mean"]),
                    "meanmil_std": float(row["meanmil_std"]),
                }
    return benchmarks


def load_v120_aurocs() -> dict[str, float]:
    v120_aurocs = {}
    for task in SEAL_TASKS:
        task_name = task.replace("/", "_")
        p = PROJECT_ROOT / f"predictions/pathobench_{task_name}_v120_seal10_official50_bf16.pt"
        if p.exists():
            data = torch.load(p, map_location="cpu")
            v120_aurocs[task] = float(data["fold_auroc_mean"])
    return v120_aurocs


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SEAL 10-Task Evaluation (v121 7-Branch Active Baseline)")
    print(f"Config: {CONFIG}")
    print(f"Tasks: {len(SEAL_TASKS)}")
    print("=" * 70)

    # Detect available GPUs (prefer GPUs 0..7)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    # Use up to 8 GPUs
    gpu_pool = [i for i in range(min(num_gpus, 8))]
    print(f"Available GPUs in pool: {gpu_pool}")

    published = load_seal_published_benchmarks()
    v120_bench = load_v120_aurocs()

    # Worker queue execution
    start_time = time.time()
    queue = list(SEAL_TASKS)
    running: dict[int, tuple[str, subprocess.Popen, any, Path, Path]] = {}
    results = {}

    while queue or running:
        # Launch tasks on free GPUs
        free_gpus = [g for g in gpu_pool if g not in running]
        while queue and free_gpus:
            gpu_id = free_gpus.pop(0)
            task = queue.pop(0)
            task_name = task.replace("/", "_")
            task_dir = OFFICIAL / task
            out_file = PRED_DIR / f"pure_v121_7branch_{task_name}.pt"
            log_file = LOG_DIR / f"{task_name}.log"

            cmd = [
                str(PROJECT_ROOT / ".venv/bin/python"),
                str(PROJECT_ROOT / "scripts/evaluate_pure.py"),
                "--config", str(CONFIG),
                "--features", str(FEATURES),
                "--official-folds", str(task_dir),
                "--official-nfolds", "50",
                "--device", f"cuda:{gpu_id}",
                "--output", str(out_file),
            ]

            print(f"  [GPU {gpu_id}] Launching {task} -> {log_file.name}")
            fh = open(log_file, "w")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            running[gpu_id] = (task, p, fh, log_file, out_file)

        # Check running processes
        time.sleep(1.0)
        done_gpus = []
        for gpu_id, (task, p, fh, log_file, out_file) in list(running.items()):
            ret = p.poll()
            if ret is not None:
                fh.close()
                elapsed = time.time() - start_time
                done_gpus.append(gpu_id)
                print(f"  [GPU {gpu_id}] Finished {task} (rc={ret}, elapsed={elapsed:.1f}s)")

                # Parse log
                log_file.read_text()
                fold_mean = None
                fold_std = None
                if out_file.exists():
                    data = torch.load(out_file, map_location="cpu")
                    aurocs = [float(x) for x in data.get("fold_aurocs", [])]
                    fold_mean = float(data.get("fold_auroc_mean", 0.0))
                    import numpy as np
                    fold_std = float(np.std(aurocs)) if aurocs else 0.0
                else:
                    fold_mean = None
                    fold_std = None

                results[task] = {
                    "task": task,
                    "return_code": ret,
                    "fold_auroc_mean": fold_mean,
                    "fold_auroc_std": fold_std,
                    "pooled_auroc": None,
                    "v120_auroc": v120_bench.get(task),
                    "abmil": published.get(task, {}).get("abmil"),
                    "meanmil": published.get(task, {}).get("meanmil"),
                }

        for g in done_gpus:
            del running[g]

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"SEAL 10-Task Evaluation Finished in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min)")
    print("=" * 80)

    # Print summary table
    print("\n| Task | v121 (7-branch) | v120 (5-branch) | Δ(v121-v120) | SEAL ABMIL | Δ(v121-ABMIL) | SEAL MeanMIL | Δ(v121-MeanMIL) | Status |")
    print("|:---|---:|---:|---:|---:|---:|---:|---:|:---:|")

    v121_means, v120_means, abmil_means, meanmil_means = [], [], [], []

    for task in SEAL_TASKS:
        r = results[task]
        v121_val = r["fold_auroc_mean"]
        v120_val = r["v120_auroc"]
        abmil_val = r["abmil"]
        meanmil_val = r["meanmil"]

        v121_str = f"{v121_val:.4f} ± {r['fold_auroc_std']:.4f}" if v121_val is not None else "FAIL"
        v120_str = f"{v120_val:.4f}" if v120_val is not None else "N/A"
        delta_v120 = f"{v121_val - v120_val:+.4f}" if (v121_val and v120_val) else "—"
        abmil_str = f"{abmil_val:.4f}" if abmil_val is not None else "—"
        delta_abmil = f"{v121_val - abmil_val:+.4f}" if (v121_val and abmil_val) else "—"
        meanmil_str = f"{meanmil_val:.4f}" if meanmil_val is not None else "—"
        delta_meanmil = f"{v121_val - meanmil_val:+.4f}" if (v121_val and meanmil_val) else "—"
        status = "OK" if r["return_code"] == 0 else "FAIL"

        print(f"| `{task}` | {v121_str} | {v120_str} | {delta_v120} | {abmil_str} | {delta_abmil} | {meanmil_str} | {delta_meanmil} | {status} |")

        if v121_val is not None: v121_means.append(v121_val)
        if v120_val is not None: v120_means.append(v120_val)
        if abmil_val is not None: abmil_means.append(abmil_val)
        if meanmil_val is not None: meanmil_means.append(meanmil_val)

    macro_v121 = sum(v121_means) / len(v121_means) if v121_means else None
    macro_v120 = sum(v120_means) / len(v120_means) if v120_means else None
    macro_abmil = sum(abmil_means) / len(abmil_means) if abmil_means else None
    macro_meanmil = sum(meanmil_means) / len(meanmil_means) if meanmil_means else None

    print(f"| **Macro Average** | **{macro_v121:.4f}** | **{macro_v120:.4f}** | **{macro_v121 - macro_v120:+.4f}** | **{macro_abmil:.4f}** | **{macro_v121 - macro_abmil:+.4f}** | **{macro_meanmil:.4f}** | **{macro_v121 - macro_meanmil:+.4f}** | — |")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_elapsed,
        "config": str(CONFIG),
        "macro_v121_7branch": macro_v121,
        "macro_v120_5branch": macro_v120,
        "delta_v121_vs_v120": (macro_v121 - macro_v120) if (macro_v121 and macro_v120) else None,
        "macro_seal_abmil": macro_abmil,
        "delta_v121_vs_abmil": (macro_v121 - macro_abmil) if (macro_v121 and macro_abmil) else None,
        "macro_seal_meanmil": macro_meanmil,
        "delta_v121_vs_meanmil": (macro_v121 - macro_meanmil) if (macro_v121 and macro_meanmil) else None,
        "task_results": results,
    }

    out_json = PROJECT_ROOT / "docs/history/ru92_seal10_v121_7branch_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary JSON to {out_json}")


if __name__ == "__main__":
    main()
