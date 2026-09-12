#!/usr/bin/env python3
"""RU-91-B: Legacy test_pathobench.py 32-true parity execution across 7 GPUs.

Runs the legacy oracle under --precision 32-true with ICF_SHAPE_SCREEN_ONLY=0
(live 7-branch arm) on all Primary 7 tasks (50 folds each), and compares
predictions 1:1 against pure runner outputs (predictions/parity_pure/pure_*.pt).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

TASKS = [
    "cptac_lscc/ARID1A_mutation",
    "cptac_lscc/Histologic_Grade",
    "cptac_lscc/KEAP1_mutation",
    "cptac_luad/KRAS_mutation",
    "cptac_pda/SMAD4_mutation",
    "ucla_lung/progression_regression",
    "cptac_ccrcc/PBRM1_mutation",
]

CONFIG = PROJECT_ROOT / "configs/archive/v94_v102_cell_value/train_v98_p1_reverse_1536_1gpu.yaml"
FEATURES = PROJECT_ROOT / "data/repro_labels_folds/features"
LOG_DIR = PROJECT_ROOT / "logs/ru91b_legacy_32true"
PRED_DIR = PROJECT_ROOT / "predictions/ru91b_legacy_32true"
PURE_DIR = PROJECT_ROOT / "predictions/parity_pure"


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("=== RU-91-B: Starting Legacy 32-true Parity Run across 7 GPUs ===")
    start_time = time.time()

    env_base = os.environ.copy()
    # Exactly match icf_arm_v121 + shape branches
    env_base["ICF_COVARIANCE_BASIS"] = "pca_within"
    env_base["ICF_FIXED_HEAD"] = "1"
    env_base["ICF_SKETCH_DIM"] = "256"
    env_base["ICF_CT_PCA_DIM"] = "32"
    env_base["ICF_CT_READOUT"] = "ridge"
    env_base["ICF_CT_CELLS"] = "0.125"
    env_base["ICF_CT_CELLS_SCALE"] = "own"
    env_base["ICF_CT_CELLS_MIN"] = "64"
    env_base["ICF_CT_ABUNDANCE_CELLS"] = "match"
    env_base["ICF_CT_SAMPLING"] = "random"
    env_base["ICF_CT_SAMPLING_SEED"] = "0"
    env_base["ICF_CT_TOKENS"] = "256"
    env_base["ICF_CT_TOKENIZER"] = "kmeans_plusplus"
    env_base["ICF_CT_KMEANS_MAX_ITER"] = "8"
    env_base["ICF_CT_DISTANCE_KERNEL"] = "gemm"
    env_base["ICF_CV_BLOCKS"] = "offdiag"
    env_base["ICF_CV_BLOCK_NORM"] = "blockwise"
    env_base["ICF_CV_CORR"] = "0"
    env_base["ICF_FIXED_HEAD_CV_WEIGHT"] = "1.0"
    env_base["ICF_FIXED_HEAD_DD_WEIGHT"] = "0.0"
    env_base["ICF_FIXED_HEAD_CT_WEIGHT"] = "0.0"
    env_base["ICF_FIXED_HEAD_BM_WEIGHT"] = "1.0"
    env_base["ICF_BM_DIM"] = "32"
    env_base["ICF_BM_LAMBDA"] = "1.0"
    env_base["ICF_FIXED_HEAD_BD_WEIGHT"] = "1.0"
    env_base["ICF_BD_DIM"] = "256"
    env_base["ICF_BD_METRIC"] = "entropy"
    env_base["ICF_BD_READOUT"] = "ordered_typicality"
    env_base["ICF_BD_SEPARATION_FLOOR"] = "1.0"
    env_base["ICF_FIXED_HEAD_QA_WEIGHT"] = "1.0"
    env_base["ICF_QA_DIM"] = "32"
    env_base["ICF_QA_LAMBDA"] = "1.0"
    env_base["ICF_FIXED_HEAD_DS_WEIGHT"] = "1.0"
    env_base["ICF_DS_DIM"] = "32"
    env_base["ICF_DS_LAMBDA"] = "1.0"
    env_base["ICF_DS_TEMPERATURE"] = "1.0"
    env_base["ICF_DS_TOKENS"] = "256"
    env_base["ICF_AGGREGATION"] = "trimmed_mean"
    
    # 7-branch live activation
    env_base["ICF_SHAPE_SCREEN_ONLY"] = "0"
    env_base["ICF_FIXED_HEAD_SH_WEIGHT"] = "1.0"
    env_base["ICF_FIXED_HEAD_SJ_WEIGHT"] = "1.0"
    env_base["ICF_FIXED_HEAD_BS_WEIGHT"] = "0.0"
    env_base["ICF_BS_DIM"] = "256"
    env_base["ICF_BS_LAMBDA"] = "1.0"
    env_base["ICF_SH_DIM"] = "32"
    env_base["ICF_SH_WIDE"] = "256"
    env_base["ICF_SH_LAMBDA"] = "1.0"
    env_base["ICF_SH_VARIANTS"] = "sh,sj"

    procs = []
    for gpu_id, task in enumerate(TASKS):
        task_name = task.replace("/", "_")
        task_dir = PROJECT_ROOT / f"data/repro_labels_folds/official/{task}"
        out_file = PRED_DIR / f"legacy_32true_{task_name}.pt"
        log_file = LOG_DIR / f"{task_name}.log"

        cmd = [
            str(PROJECT_ROOT / ".venv/bin/python"),
            str(PROJECT_ROOT / "scripts/test_pathobench.py"),
            "--config", str(CONFIG),
            "--features", str(FEATURES),
            "--official-folds", str(task_dir),
            "--official-nfolds", "50",
            "--input-dim", "1536",
            "--precision", "32-true",
            "--output", str(out_file),
        ]

        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"  [GPU {gpu_id}] Launching legacy 32-true on {task} -> {log_file.name}")
        fh = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        procs.append((task, gpu_id, p, fh, log_file, out_file))

    print("\nAll 7 tasks launched. Waiting for completion...")
    for task, gpu_id, p, fh, log_file, out_file in procs:
        rc = p.wait()
        fh.close()
        elapsed = time.time() - start_time
        print(f"  [GPU {gpu_id}] Finished {task} (rc={rc}, elapsed={elapsed:.1f}s)")

    print("\n=== RU-91-B Comparison: Legacy 32-true vs Pure Runner ===")
    from src.utils.metrics import auroc

    results = []
    pure_aurocs = []
    legacy_aurocs = []

    for task, gpu_id, _, _, log_file, out_file in procs:
        task_name = task.replace("/", "_")
        pure_file = PURE_DIR / f"pure_{task_name}.pt"

        if not out_file.exists() or not pure_file.exists():
            print(f"Missing output for {task}")
            continue

        legacy_data = torch.load(out_file, map_location="cpu", weights_only=False)
        pure_data = torch.load(pure_file, map_location="cpu", weights_only=False)

        legacy_by_fold = dict(zip(legacy_data["fold_indices"], legacy_data["per_fold"]))
        pure_by_fold = dict(zip(pure_data["fold_indices"], pure_data["per_fold"]))

        max_dp = 0.0
        sum_dp = 0.0
        n_slides = 0
        l_aurocs = []
        p_aurocs = []

        for k in pure_data["fold_indices"]:
            if k not in legacy_by_fold:
                continue
            l_entry = legacy_by_fold[k]
            p_entry = pure_by_fold[k]

            p_l = torch.tensor(l_entry["probability"]) if isinstance(l_entry["probability"], list) else l_entry["probability"]
            p_p = p_entry["probability"]

            l_aurocs.append(float(auroc(p_l, l_entry["label"])))
            p_aurocs.append(float(auroc(p_p, p_entry["label"])))

            l_map = dict(zip(l_entry["slide_id"], p_l.tolist()))
            for sid, p in zip(p_entry["slide_id"], p_p.tolist()):
                if sid in l_map:
                    diff = abs(p - l_map[sid])
                    max_dp = max(max_dp, diff)
                    sum_dp += diff
                    n_slides += 1

        mean_dp = sum_dp / max(n_slides, 1)
        l_mean_auroc = sum(l_aurocs) / max(len(l_aurocs), 1)
        p_mean_auroc = sum(p_aurocs) / max(len(p_aurocs), 1)
        d_auroc = p_mean_auroc - l_mean_auroc

        results.append({
            "task": task,
            "pure_auroc": p_mean_auroc,
            "legacy_32true_auroc": l_mean_auroc,
            "delta_auroc": d_auroc,
            "max_delta_p": max_dp,
            "mean_delta_p": mean_dp,
            "n_slides": n_slides,
        })
        pure_aurocs.append(p_mean_auroc)
        legacy_aurocs.append(l_mean_auroc)

    print("| Task | Pure AUROC | Legacy 32-true AUROC | ΔAUROC | max|Δp| | mean|Δp| | N slides |")
    print("|:---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| `{r['task']}` | {r['pure_auroc']:.4f} | {r['legacy_32true_auroc']:.4f} | {r['delta_auroc']:+.4f} | {r['max_delta_p']:.3e} | {r['mean_delta_p']:.3e} | {r['n_slides']} |")

    macro_pure = sum(pure_aurocs) / len(pure_aurocs)
    macro_legacy = sum(legacy_aurocs) / len(legacy_aurocs)
    macro_delta = macro_pure - macro_legacy
    print(f"| **Macro Average** | **{macro_pure:.4f}** | **{macro_legacy:.4f}** | **{macro_delta:+.4f}** | — | — | **{sum(r['n_slides'] for r in results)}** |")

    summary_file = PROJECT_ROOT / "docs/history/ru91b_legacy_32true_parity_results.json"
    summary_file.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": time.time() - start_time,
        "macro_pure": macro_pure,
        "macro_legacy_32true": macro_legacy,
        "macro_delta": macro_delta,
        "results": results,
    }, indent=2))
    print(f"\nSaved RU-91-B summary to {summary_file}")


if __name__ == "__main__":
    main()
