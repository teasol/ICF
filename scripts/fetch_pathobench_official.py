"""Fetch official Patho-Bench fold-based splits (k=all.tsv + config.yaml) from HuggingFace.

Downloads, for each source/task used by this project, the official split files:
  {out_dir}/{source}/{task}/k=all.tsv      # case_id, slide_id, <task_col>, fold_0..fold_49 (train/val/test)
  {out_dir}/{source}/{task}/config.yaml    # task_col, label_dict, sample_col, task_type, metrics

Mirrors the layout that ``patho_bench.SplitFactory.from_local`` expects. The
official split is *case-based* (sample_col=case_id) and *fold-based*: the label
column is ``config.yaml.task_col``, and train/val/test assignment is per-fold in
the ``fold_*`` columns of ``k=all.tsv`` (choose a fold, e.g. fold_0).

Usage:
    python scripts/fetch_pathobench_official.py [--out-dir ...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Sources/tasks used by this project (from official available_splits.yaml).
SOURCES_TASKS: dict[str, list[str]] = {
    "bc_therapy": ["er_status", "grade", "her2_status", "residual_cancer_burden"],
    "bracs": ["slidelevel_coarse", "slidelevel_fine"],
    "cptac_brca": ["Immune_class", "PIK3CA_mutation", "TP53_mutation"],
    "cptac_ccrcc": ["BAP1_mutation", "Immune_class", "OS", "PBRM1_mutation", "VHL_mutation"],
    "cptac_lscc": ["ARID1A_mutation", "Histologic_Grade", "Immune_class", "KEAP1_mutation"],
    "cptac_luad": ["EGFR_mutation", "Immune_class", "KRAS_mutation", "OS", "STK11_mutation", "TP53_mutation"],
    "cptac_pda": ["Immune_class", "OS", "SMAD4_mutation"],
    "herroi": ["response"],
    "mbc_": ["OS", "Recist"],
    "ucla_lung": ["progression_regression"],
}

BASE_URL = "https://huggingface.co/datasets/MahmoodLab/Patho-Bench/resolve/main"


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sL", "-o", str(dest), url],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/NHNHOME/kimds/Data/PathoBench/official"),
    )
    parser.add_argument("--source", default=None, help="Only this source (default: all).")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    total = 0
    for source, tasks in sorted(SOURCES_TASKS.items()):
        if args.source and source != args.source:
            continue
        for task in tasks:
            split_url = f"{BASE_URL}/{source}/{task}/k=all.tsv"
            cfg_url = f"{BASE_URL}/{source}/{task}/config.yaml"
            fetch(split_url, out_dir / source / task / "k=all.tsv")
            fetch(cfg_url, out_dir / source / task / "config.yaml")
            total += 1
            print(f"  ok {source}/{task}", flush=True)
    print(f"Done: {total} tasks -> {out_dir}")


if __name__ == "__main__":
    main()
