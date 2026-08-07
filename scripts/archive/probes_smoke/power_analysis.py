"""How large an AUROC difference can the ICI cohort actually resolve?

Every architecture comparison so far was reported as a bare point estimate,
and the v21 retrieval investigation ended by discovering that differences of
0.04 AUROC were indistinguishable from noise at n=87. This script answers the
question that should be asked *before* running an experiment: given the cohort
size, what effect is worth trying to detect?

Method: simulate paired classifiers via a latent-normal model. For a target
AUROC `a`, positive scores are drawn from N(d, 1) and negatives from N(0, 1)
with d = sqrt(2) * Phi^-1(a). Two models are correlated by `rho` to mimic the
fact that competing architectures trained on the same data agree far more
often than chance -- ignoring that correlation would badly understate power
for a paired comparison. Each simulated pair is then run through the same
paired bootstrap used by `compare_predictions.py`, and power is the fraction
of trials the test calls the better model at the stated confidence.

Cost scales as trials * bootstrap * n log n. The ICI defaults finish in a few
minutes; a synthetic-scale cohort (~1,700 predictions) is roughly an order of
magnitude slower, so lower --trials/--bootstrap or expect a long run. For
synthetic runs the cheaper and more direct answer is usually the measured
interval from `evaluate_synthetic.py` rather than a simulation.

Usage:
    python scripts/power_analysis.py
    python scripts/power_analysis.py --n-positive 37 --n-negative 50
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import auroc_rows  # noqa: E402


def separation_for_auroc(value: float) -> float:
    """Latent-normal mean separation d such that P(pos > neg) == value."""
    return math.sqrt(2.0) * torch.special.ndtri(torch.tensor(value)).item()


def simulate_power(
    baseline_auroc: float,
    delta: float,
    n_positive: int,
    n_negative: int,
    rho: float,
    trials: int,
    bootstrap: int,
    confidence: float,
    generator: torch.Generator,
) -> float:
    labels = torch.cat(
        [torch.ones(n_positive, dtype=torch.long), torch.zeros(n_negative, dtype=torch.long)]
    )
    total = n_positive + n_negative
    d_weak = separation_for_auroc(baseline_auroc)
    d_strong = separation_for_auroc(min(baseline_auroc + delta, 0.999))
    shift = torch.cat([torch.ones(n_positive), torch.zeros(n_negative)])

    detected = 0
    for _ in range(trials):
        shared = torch.randn(total, generator=generator)
        weak = math.sqrt(rho) * shared + math.sqrt(1 - rho) * torch.randn(
            total, generator=generator
        )
        strong = math.sqrt(rho) * shared + math.sqrt(1 - rho) * torch.randn(
            total, generator=generator
        )
        weak = weak + d_weak * shift
        strong = strong + d_strong * shift

        index = torch.randint(0, total, (bootstrap, total), generator=generator)
        resampled_labels = labels[index]
        keep = (resampled_labels.sum(dim=1) > 0) & (resampled_labels.sum(dim=1) < total)
        if keep.sum() == 0:
            continue
        index = index[keep]
        resampled_labels = labels[index]
        strong_auroc = auroc_rows(strong[index], resampled_labels)
        weak_auroc = auroc_rows(weak[index], resampled_labels)
        usable = int(keep.sum())
        wins = int((strong_auroc > weak_auroc).sum())
        if wins / usable >= confidence:
            detected += 1
    return detected / trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-positive", type=int, default=37, help="ICI: 37 responders")
    parser.add_argument("--n-negative", type=int, default=50, help="ICI: 50 non-responders")
    parser.add_argument("--baseline-auroc", type=float, default=0.55)
    parser.add_argument(
        "--rho",
        type=float,
        default=0.7,
        help="Score correlation between the two compared models.",
    )
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Paired-bootstrap win rate required to call a winner.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10, 0.15, 0.20],
        help="True AUROC gains to evaluate power for.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional name for the cohort being analysed, shown in the header.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    total = args.n_positive + args.n_negative

    label = f"{args.label}: " if args.label else ""
    print(
        f"{label}n={total} ({args.n_positive} positive / {args.n_negative} negative), "
        f"baseline AUROC {args.baseline_auroc}, model correlation rho={args.rho}"
    )
    print(f"Calling a winner at paired-bootstrap win rate >= {args.confidence}\n")
    print(f"{'true AUROC gain':>16} | {'power':>7}")
    print("-" * 27)
    for delta in args.deltas:
        power = simulate_power(
            args.baseline_auroc,
            delta,
            args.n_positive,
            args.n_negative,
            args.rho,
            args.trials,
            args.bootstrap,
            args.confidence,
            generator,
        )
        print(f"{delta:>16.2f} | {power:>6.0%}")

    print(
        "\nPower is the chance of correctly calling the better model. Below ~80%"
        "\na negative result means nothing -- the experiment could not have"
        "\nshown a difference even if one existed."
    )


if __name__ == "__main__":
    main()
