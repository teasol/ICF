#!/usr/bin/env python3
"""
Parse evaluation logs into a stable metrics.json with tolerance gate.
Stdlib-only: no third-party imports.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

RE_FOLD_MEAN = re.compile(
    r"fold-mean\s+AUROC:\s*([0-9\.\-eE+]+)\s*[±\u00b1]\s*([0-9\.\-eE+]+)\s+pooled\s+AUROC:\s*([0-9\.\-eE+]+)"
)
RE_PER_FOLD = re.compile(r"per-fold\s+AUROC:\s*(.+)$")


def parse_log_file(log_path: str, task_name: str) -> Dict[str, Any]:
    """
    Parse a single evaluation log file for a task.
    Extracts fold_mean_auroc, fold_std, pooled_auroc, per_fold_auroc, n_folds.
    Fails loudly (sys.exit(1)) if the log is missing, malformed, or has != 50 folds.
    """
    if not os.path.isfile(log_path):
        sys.stderr.write(f"Error: Log file not found for task '{task_name}': {log_path}\n")
        sys.exit(1)

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        sys.stderr.write(f"Error reading log file for task '{task_name}' ({log_path}): {e}\n")
        sys.exit(1)

    last_fm = None
    last_pf = None

    for line in lines:
        m_fm = RE_FOLD_MEAN.search(line)
        if m_fm:
            last_fm = m_fm
        m_pf = RE_PER_FOLD.search(line)
        if m_pf:
            last_pf = m_pf

    if last_fm is None:
        sys.stderr.write(
            f"Error: No 'fold-mean AUROC:' line found for task '{task_name}' in log: {log_path}\n"
        )
        sys.exit(1)

    if last_pf is None:
        sys.stderr.write(
            f"Error: No 'per-fold AUROC:' line found for task '{task_name}' in log: {log_path}\n"
        )
        sys.exit(1)

    try:
        fold_mean_raw = float(last_fm.group(1))
        fold_std_raw = float(last_fm.group(2))
        pooled_raw = float(last_fm.group(3))
    except ValueError as e:
        sys.stderr.write(
            f"Error parsing fold-mean / pooled AUROC floats for task '{task_name}' in log {log_path}: {e}\n"
        )
        sys.exit(1)

    per_fold_tokens = last_pf.group(1).strip().split()
    try:
        per_fold_floats = [float(x) for x in per_fold_tokens]
    except ValueError as e:
        sys.stderr.write(
            f"Error parsing per-fold AUROC values for task '{task_name}' in log {log_path}: {e}\n"
        )
        sys.exit(1)

    n_folds = len(per_fold_floats)
    if n_folds != 50:
        sys.stderr.write(
            f"Error: Expected 50 folds for task '{task_name}' in log {log_path}, found {n_folds}\n"
        )
        sys.exit(1)

    return {
        "fold_mean_auroc": round(fold_mean_raw, 6),
        "fold_std": round(fold_std_raw, 6),
        "pooled_auroc": round(pooled_raw, 6),
        "n_folds": int(n_folds),
        "per_fold_auroc": [round(x, 6) for x in per_fold_floats],
    }


def extract_metrics(
    manifest_path: str,
    reference: float = 0.6265,
    tolerance: float = 0.005,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Load manifest, parse all 7 task logs, compute Primary 7 macro AUROC and tolerance check.
    """
    if not os.path.isfile(manifest_path):
        sys.stderr.write(f"Error: Manifest file not found: {manifest_path}\n")
        sys.exit(1)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Error reading manifest JSON ({manifest_path}): {e}\n")
        sys.exit(1)

    tasks_list = manifest.get("tasks", [])
    if not isinstance(tasks_list, list) or len(tasks_list) != 7:
        actual_count = len(tasks_list) if isinstance(tasks_list, list) else 0
        sys.stderr.write(
            f"Error: Manifest '{manifest_path}' must list exactly 7 tasks, found {actual_count}\n"
        )
        sys.exit(1)

    tasks_dict = {}
    ordered_task_names = []

    for task_entry in tasks_list:
        task_name = task_entry.get("task")
        log_path = task_entry.get("log")
        if not task_name:
            sys.stderr.write(f"Error: Manifest entry missing 'task' field in {manifest_path}\n")
            sys.exit(1)
        if not log_path:
            sys.stderr.write(f"Error: Manifest entry for task '{task_name}' missing 'log' field in {manifest_path}\n")
            sys.exit(1)

        parsed = parse_log_file(log_path, task_name)
        tasks_dict[task_name] = {
            "fold_mean_auroc": parsed["fold_mean_auroc"],
            "fold_std": parsed["fold_std"],
            "log": log_path,
            "n_folds": parsed["n_folds"],
            "per_fold_auroc": parsed["per_fold_auroc"],
            "pooled_auroc": parsed["pooled_auroc"],
        }
        ordered_task_names.append(task_name)

    # Arithmetic — follow this order exactly so output is reproducible:
    # 1. round each task's fold_mean_auroc to 6 decimals (done in parse_log_file)
    # 2. macro_fold_mean_auroc = round(sum(those rounded values) / 7, 6)
    # 3. abs_delta_vs_reference = round(abs(macro_fold_mean_auroc - reference), 6)
    # 4. within_tolerance = 1 if abs_delta_vs_reference <= tolerance else 0
    rounded_means = [tasks_dict[t]["fold_mean_auroc"] for t in ordered_task_names]
    macro_fold_mean_auroc = round(sum(rounded_means) / 7.0, 6)
    abs_delta_vs_reference = round(abs(macro_fold_mean_auroc - reference), 6)
    within_tolerance = 1 if abs_delta_vs_reference <= tolerance else 0

    output_data = {
        "arm": manifest.get("arm", ""),
        "primary7": {
            "abs_delta_vs_reference": abs_delta_vs_reference,
            "macro_fold_mean_auroc": macro_fold_mean_auroc,
            "n_tasks": 7,
            "reference_macro_fold_mean_auroc": round(reference, 6) if isinstance(reference, float) else reference,
            "tolerance": round(tolerance, 6) if isinstance(tolerance, float) else tolerance,
            "within_tolerance": int(within_tolerance),
        },
        "tag": manifest.get("tag", ""),
        "tasks": tasks_dict,
    }

    return output_data, ordered_task_names


def main():
    parser = argparse.ArgumentParser(
        description="Parse evaluation logs into a stable metrics.json with tolerance gate."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file")
    parser.add_argument("--out", required=True, help="Path to output metrics JSON file")
    parser.add_argument(
        "--reference",
        type=float,
        default=0.6265,
        help="Reference macro fold-mean AUROC (default: 0.6265)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.005,
        help="Tolerance for macro fold-mean AUROC (default: 0.005)",
    )

    args = parser.parse_args()

    metrics_data, ordered_tasks = extract_metrics(
        manifest_path=args.manifest,
        reference=args.reference,
        tolerance=args.tolerance,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, sort_keys=True, indent=2)
        f.write("\n")

    # Print plain-text table to stdout
    tasks_dict = metrics_data["tasks"]
    for task_name in ordered_tasks:
        print(f"{task_name} {tasks_dict[task_name]['fold_mean_auroc']}")

    p7 = metrics_data["primary7"]
    print(
        f"MACRO {p7['macro_fold_mean_auroc']} "
        f"DELTA {p7['abs_delta_vs_reference']} "
        f"TOLERANCE {args.tolerance} "
        f"WITHIN {p7['within_tolerance']}"
    )


if __name__ == "__main__":
    main()
