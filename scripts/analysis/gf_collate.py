#!/usr/bin/env python3
"""Collate every GF exploration run into one table.

Two families of artefact are read:

  background GMM summaries  -- how many components are cohort-pure, and how many
                               components each BACKGROUND cohort spreads over
  standalone eval summaries -- per-task fold-mean AUROC and how many components
                               each PRIMARY 7 cohort (unseen by the GMM) spreads
                               over

The second block is the one that decides whether the cohort-standardisation +
PCA-whitening intervention worked: the criterion fixed before the results were
seen is Primary 7 effective components >= 4 at M=8.

Nothing here compares GF against the official 7-branch basis -- D-045 keeps GF
out of the adoption gates and the promotion review.

Usage:
    .venv/bin/python scripts/analysis/gf_collate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BACKGROUND_ROOT = Path("/NHNHOME/BASE/kimds/Data/PathoBench/gf_background")
PRED_ROOT = Path(__file__).resolve().parents[2] / "predictions/gf_standalone"

RAW_DIRS = {8: "fixed_gmm", 16: "fixed_gmm_m16", 32: "fixed_gmm_m32"}


def background_rows() -> list[dict]:
    rows = []
    for m, sub in RAW_DIRS.items():
        path = BACKGROUND_ROOT / sub / "summary.json"
        if not path.exists():
            continue
        blob = json.loads(path.read_text())
        for policy in ("raw", "balanced"):
            occ = blob.get(policy, {}).get("component_occupancy")
            if occ is None:
                continue
            rows.append({
                "basis": f"raw1536 M={m} {policy}",
                "n_components": m,
                "val_loglik": blob[policy]["val_loglik"]["overall"],
            })
    for path in sorted((BACKGROUND_ROOT / "fixed_gmm_whitened").glob("summary_d*.json")):
        blob = json.loads(path.read_text())
        rows.append({
            "basis": f"whiten D={blob['n_out']} M={blob['n_components']}",
            "n_components": blob["n_components"],
            "pure_components": blob["components_above_90pct_purity"],
            "effective": blob["effective_components"],
            "val_loglik": blob["val_loglik"]["overall"],
            "converged": blob["converged"],
        })
    return rows


def eval_rows(pred_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(pred_dir.glob("gf_*_summary.json")):
        blob = json.loads(path.read_text())
        rows.append(blob)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, default=PRED_ROOT)
    args = ap.parse_args()

    print("=== background GMM: component allocation on the BACKGROUND cohorts ===")
    for row in background_rows():
        if "pure_components" in row:
            eff = " ".join(f"{k.split('-')[0]} {v:.2f}" for k, v in row["effective"].items())
            print(f"  {row['basis']:24s} pure>90% {row['pure_components']}/{row['n_components']}"
                  f"  eff[{eff}]  valLL {row['val_loglik']:9.2f}")
        else:
            print(f"  {row['basis']:24s} valLL {row['val_loglik']:9.2f}")

    print("\n=== standalone Primary 7 (exploration record, gates skipped per D-045) ===")
    runs = eval_rows(args.pred_dir)
    if not runs:
        print("  (no evaluation summaries yet)")
        return

    tasks = [t["task"] for t in runs[0]["tasks"]]
    header = f"  {'run':18s} {'FVdim':>7s} {'macro':>7s} " + " ".join(f"{t.split('/')[-1][:9]:>9s}" for t in tasks)
    print(header)
    for blob in runs:
        cells = " ".join(f"{t['fold_mean_auroc']:9.4f}" for t in blob["tasks"])
        print(f"  {blob['tag']:18s} {blob['fv_dim']:7d} {blob['macro_auroc']:7.4f} {cells}")

    print("\n=== effective components on PRIMARY 7 cohorts (criterion: >= 4 at M=8) ===")
    print(header)
    for blob in runs:
        cells = " ".join(f"{t['effective_components']:9.2f}" for t in blob["tasks"])
        print(f"  {blob['tag']:18s} {blob['fv_dim']:7d} {'':>7s} {cells}")


if __name__ == "__main__":
    main()
