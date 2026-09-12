#!/usr/bin/env python3
"""Run Primary 7 50-fold parity suite with evaluate_pure.py across GPUs (RFC Phase A-2).

Launches one evaluate_pure.py process per task on separate GPUs (0..6),
collects fold-mean AUROCs, max|Δp|, mean|Δp|, and computes macro-AUROC.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TASKS = [
    "cptac_lscc/ARID1A_mutation",
    "cptac_lscc/Histologic_Grade",
    "cptac_lscc/KEAP1_mutation",
    "cptac_luad/KRAS_mutation",
    "cptac_pda/SMAD4_mutation",
    "ucla_lung/progression_regression",
    "cptac_ccrcc/PBRM1_mutation",
]

CONFIG = PROJECT_ROOT / "configs/baseline/v121_7branch_active.yaml"
FEATURES = PROJECT_ROOT / "data/repro_labels_folds/features"
LOG_DIR = PROJECT_ROOT / "logs/parity_primary7"
PRED_DIR = PROJECT_ROOT / "predictions/parity_pure"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Primary 7 Parity Suite across 7 GPUs ===")
    print(f"Config: {CONFIG}")
    print(f"Tasks: {len(TASKS)}")

    procs = []
    start_time = time.time()

    for gpu_id, task in enumerate(TASKS):
        task_name = task.replace("/", "_")
        golden_file = PROJECT_ROOT / f"predictions/pathobench_{task_name}_ru90_shape_triple_official50_bf16.pt"
        task_dir = PROJECT_ROOT / f"data/repro_labels_folds/official/{task}"
        out_file = PRED_DIR / f"pure_{task_name}.pt"
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
            "--compare-golden", str(golden_file),
        ]

        print(f"  [GPU {gpu_id}] Launching {task} -> {log_file.name}")
        fh = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((task, gpu_id, p, fh, log_file, golden_file))

    print(f"\nAll 7 tasks launched. Waiting for completion...")

    results = []
    for task, gpu_id, p, fh, log_file, golden_file in procs:
        rc = p.wait()
        fh.close()
        elapsed = time.time() - start_time
        print(f"  [GPU {gpu_id}] Finished {task} (rc={rc}, elapsed={elapsed:.1f}s)")

        # Parse log file for results
        log_text = log_file.read_text()
        pure_auroc = None
        gold_auroc = None
        max_dp = None
        mean_dp = None

        for line in log_text.splitlines():
            if line.startswith("fold-mean AUROC:"):
                parts = line.split()
                pure_auroc = float(parts[2])
            if "[compare-golden]" in line and "pure AUROC" in line:
                m = re.search(r"max\|Δp\| = ([\d.e+-]+), mean\|Δp\| = ([\d.e+-]+) over (\d+) slide predictions\. pure AUROC = ([\d.]+) vs gold = ([\d.]+) \(Δ = ([+-][\d.]+)\)", line)
                if m:
                    max_dp = float(m.group(1))
                    mean_dp = float(m.group(2))
                    pure_auroc = float(m.group(4))
                    gold_auroc = float(m.group(5))

        results.append({
            "task": task,
            "pure_auroc": pure_auroc,
            "gold_auroc": gold_auroc,
            "delta_auroc": (pure_auroc - gold_auroc) if (pure_auroc is not None and gold_auroc is not None) else None,
            "max_delta_p": max_dp,
            "mean_delta_p": mean_dp,
            "return_code": rc,
        })

    total_time = time.time() - start_time
    print(f"\n=== Primary 7 Parity Results (Total Time: {total_time:.1f}s = {total_time/3600:.3f} GPU-h equivalent) ===")
    print(f"| Task | Pure AUROC | Golden AUROC | ΔAUROC | max|Δp| | mean|Δp| | Status |")
    print(f"|:---|---:|---:|---:|---:|---:|:---:|")

    pure_aurocs = []
    gold_aurocs = []
    for r in results:
        status = "PASS" if r["return_code"] == 0 else "FAIL"
        pure_str = f"{r['pure_auroc']:.4f}" if r["pure_auroc"] is not None else "N/A"
        gold_str = f"{r['gold_auroc']:.4f}" if r["gold_auroc"] is not None else "N/A"
        delta_str = f"{r['delta_auroc']:+.4f}" if r["delta_auroc"] is not None else "N/A"
        max_dp_str = f"{r['max_delta_p']:.3e}" if r["max_delta_p"] is not None else "N/A"
        mean_dp_str = f"{r['mean_delta_p']:.3e}" if r["mean_delta_p"] is not None else "N/A"
        print(f"| `{r['task']}` | {pure_str} | {gold_str} | {delta_str} | {max_dp_str} | {mean_dp_str} | {status} |")
        if r["pure_auroc"] is not None:
            pure_aurocs.append(r["pure_auroc"])
        if r["gold_auroc"] is not None:
            gold_aurocs.append(r["gold_auroc"])

    if pure_aurocs and gold_aurocs:
        macro_pure = sum(pure_aurocs) / len(pure_aurocs)
        macro_gold = sum(gold_aurocs) / len(gold_aurocs)
        macro_delta = macro_pure - macro_gold
        print(f"| **Macro Average** | **{macro_pure:.4f}** | **{macro_gold:.4f}** | **{macro_delta:+.4f}** | — | — | — |")

    # Save summary json
    summary_path = PROJECT_ROOT / "docs/history/ru91_primary7_parity_results.json"
    summary_path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_time,
        "results": results,
        "macro_pure": sum(pure_aurocs) / len(pure_aurocs) if pure_aurocs else None,
        "macro_gold": sum(gold_aurocs) / len(gold_aurocs) if gold_aurocs else None,
    }, indent=2))
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
