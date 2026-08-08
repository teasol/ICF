#!/usr/bin/env python
"""Summarize P0-slots probe outputs with fold-paired statistics.

The probe (`scripts/probe_slot_headroom.py`) evaluates every configuration on
the SAME official folds, so comparisons must be fold-paired rather than a
difference of two independent means. Reports paired mean deltas with a paired
bootstrap CI over folds.

    PY=/home/aibio_3/miniconda3/envs/BagPFN/bin/python
    $PY scripts/summarize_slot_headroom.py predictions/probe_slots_*.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def paired_bootstrap(
    a: list[float], b: list[float], n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean delta a-b, lo, hi) from a fold-paired bootstrap."""
    delta = torch.tensor(a) - torch.tensor(b)
    generator = torch.Generator().manual_seed(seed)
    n = delta.numel()
    index = torch.randint(0, n, (n_boot, n), generator=generator)
    means = delta[index].mean(dim=1)
    lo, hi = torch.quantile(means, torch.tensor([0.025, 0.975])).tolist()
    return float(delta.mean()), lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.paths:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        summary = blob["summary"]
        print(f"\n{'=' * 78}")
        print(f"{blob['task']}   ({blob['n_slides']} slides)")
        print("=" * 78)
        print(
            f"{'config':<26}{'tokens':>7}{'dim':>9}{'macro':>9}{'std':>8}{'pooled':>9}"
        )
        for key, row in summary.items():
            print(
                f"{key:<26}{row['tokens_per_bag']:>7}{row['feature_dim']:>9}"
                f"{row['macro']:>9.4f}{row['std']:>8.4f}{row['pooled']:>9.4f}"
            )

        def folds(key: str) -> list[float] | None:
            return summary[key]["fold_aurocs"] if key in summary else None

        comparisons: list[tuple[str, str, str]] = []
        if folds("slots12/all") and folds("slots12/projected"):
            comparisons.append(
                ("Q1  all vs projected (both @12 slots)", "slots12/all", "slots12/projected")
            )
        for n in (8, 24, 48):
            if folds(f"slots{n}/all") and folds("slots12/all"):
                comparisons.append(
                    (f"Q2  all@{n} vs all@12", f"slots{n}/all", "slots12/all")
                )

        if comparisons:
            print(f"\n{'fold-paired comparison':<40}{'delta':>9}{'95% CI':>22}")
            for label, key_a, key_b in comparisons:
                a, b = summary[key_a]["fold_aurocs"], summary[key_b]["fold_aurocs"]
                if len(a) != len(b):
                    print(f"{label:<40}  (fold counts differ, skipped)")
                    continue
                mean, lo, hi = paired_bootstrap(a, b)
                flag = "" if lo <= 0.0 <= hi else "  *"
                print(
                    f"{label:<40}{mean:>+9.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]{flag}"
                )
            print("  * = paired 95% CI excludes zero")


if __name__ == "__main__":
    main()
