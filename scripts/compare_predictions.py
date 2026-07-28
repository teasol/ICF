"""Compare saved 5-fold prediction files with bootstrap confidence intervals.

Point estimates alone are misleading on the ICI cohort: with n=87 (37
positive) the AUROC standard error is large enough that runs differing by
0.04 AUROC are statistically indistinguishable. This reports a 95% CI per run
and a paired bootstrap win-rate per pair, so a comparison states whether the
data can actually separate two runs rather than just which number is bigger.

Usage:
    python scripts/compare_predictions.py \
        predictions/ici_predictions_v21_retrieved_5fold.pt \
        predictions/ici_predictions_v21_phase6c_5fold.pt
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch


def auroc(probability: torch.Tensor, target: torch.Tensor) -> float:
    positive_scores = probability[target == 1]
    negative_scores = probability[target == 0]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return float("nan")
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    return (
        ((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean().item()
    )


def load(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, weights_only=False)
    aggregate = payload["validation_aggregate"]
    probability = aggregate["probabilities"][:, 1].float()
    target = aggregate["target"].long().flatten()
    return probability, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    runs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for path in args.predictions:
        runs[path.stem] = load(path)

    print(f"AUROC with {args.bootstrap}-sample bootstrap 95% CI\n")
    for name, (probability, target) in runs.items():
        count = target.numel()
        values = []
        for _ in range(args.bootstrap):
            index = torch.randint(0, count, (count,))
            resampled = target[index]
            if resampled.sum() in (0, count):
                continue
            values.append(auroc(probability[index], resampled))
        spread = torch.tensor(values)
        print(
            f"  {name}: {auroc(probability, target):.4f} "
            f"[{spread.quantile(0.025):.3f}, {spread.quantile(0.975):.3f}] "
            f"(n={count}, positive={int((target == 1).sum())})"
        )

    if len(runs) < 2:
        return

    print("\nPaired bootstrap P(row beats column) -- 0.5 means indistinguishable\n")
    for left, right in itertools.combinations(runs, 2):
        left_probability, left_target = runs[left]
        right_probability, right_target = runs[right]
        if not torch.equal(left_target, right_target):
            print(f"  {left} vs {right}: SKIPPED (different target order)")
            continue
        count = left_target.numel()
        wins = total = 0
        for _ in range(args.bootstrap):
            index = torch.randint(0, count, (count,))
            resampled = left_target[index]
            if resampled.sum() in (0, count):
                continue
            total += 1
            if auroc(left_probability[index], resampled) > auroc(
                right_probability[index], resampled
            ):
                wins += 1
        print(f"  {left} vs {right}: {wins / max(1, total):.2f}")


if __name__ == "__main__":
    main()
