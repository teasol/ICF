#!/usr/bin/env python3
"""Paired per-fold macro differences between GF exploration runs.

The collated table reports one macro number per run, and those numbers sit
within about 0.03 of one another across every configuration tried. A gap that
small cannot be read off point estimates: the runs share the same fixed 50
folds, so the question is whether the per-fold differences are distinguishable
from zero, not whether one mean is larger.

This is the same paired construction PROJECT.md SS4 fixes for candidate review --
per-fold macro (mean over the 7 tasks within a fold), then the paired
difference against a reference run -- applied here to an exploration record.
It decides nothing: D-045 keeps GF out of the adoption gates.

Usage:
    .venv/bin/python scripts/analysis/gf_paired_delta.py --ref m8_raw
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

PRED_ROOT = Path(__file__).resolve().parents[2] / "predictions/gf_standalone"


def per_fold_macro(path: Path) -> tuple[list[float], list[str]]:
    """Return one macro AUROC per fold, averaged over the tasks present."""
    blob = torch.load(path, weights_only=False)
    tasks = [r["task"] for r in blob["results"]]
    curves = [r["per_fold_auroc"] for r in blob["results"]]
    n = min(len(c) for c in curves)
    return [sum(c[i] for c in curves) / len(curves) for i in range(n)], tasks


def paired_ci(a: list[float], b: list[float]) -> tuple[float, float, float, int]:
    """Mean paired difference (a - b) with a 95% t interval."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    # t_{0.975, 49} = 2.0096; the fold count is fixed at 50 by the manifest.
    half = 2.0096 * se
    return mean, mean - half, mean + half, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="m8_raw", help="reference run tag")
    ap.add_argument("--pred-dir", type=Path, default=PRED_ROOT)
    args = ap.parse_args()

    runs = {}
    for f in sorted(args.pred_dir.glob("gf_*_per_fold.pt")):
        tag = f.name[len("gf_"):-len("_per_fold.pt")]
        runs[tag] = per_fold_macro(f)[0]

    if args.ref not in runs:
        raise SystemExit(f"reference '{args.ref}' not found; have {sorted(runs)}")
    ref = runs[args.ref]

    print(f"paired per-fold macro Δ vs '{args.ref}'  (50 folds, 95% t interval)")
    print(f"{'run':<20} {'macro':>7} {'Δ':>9} {'95% CI':>22}  판별")
    rows = []
    for tag, curve in runs.items():
        macro = sum(curve) / len(curve)
        if tag == args.ref:
            rows.append((tag, macro, 0.0, 0.0, 0.0, "기준"))
            continue
        mean, lo, hi, _ = paired_ci(curve, ref)
        verdict = "0 포함 (판별 불가)" if lo <= 0 <= hi else ("우위" if lo > 0 else "열위")
        rows.append((tag, macro, mean, lo, hi, verdict))
    for tag, macro, mean, lo, hi, verdict in sorted(rows, key=lambda r: -r[1]):
        ci = "—" if verdict == "기준" else f"[{lo:+.4f}, {hi:+.4f}]"
        d = "—" if verdict == "기준" else f"{mean:+.4f}"
        print(f"{tag:<20} {macro:.4f} {d:>9} {ci:>22}  {verdict}")


if __name__ == "__main__":
    main()
