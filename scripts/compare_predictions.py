"""Compare saved 5-fold prediction files with bootstrap confidence intervals.

Point estimates alone are misleading on the ICI cohort: with n=87 (37
positive) the AUROC standard error is large enough that runs differing by
0.04 AUROC are statistically indistinguishable. This reports a 95% CI per run
and a paired bootstrap win-rate per pair, so a comparison states whether the
data can actually separate two runs rather than just which number is bigger.

Usage:
    python scripts/compare_predictions.py \
        predictions/ici_predictions_run_a.pt \
        predictions/ici_predictions_run_b.pt
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import auroc, cluster_members, resample_index  # noqa: E402


def load(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    payload = torch.load(path, weights_only=False)
    aggregate = payload["validation_aggregate"]
    probability = aggregate["probabilities"][:, 1].float()
    target = aggregate["target"].long().flatten()
    # Synthetic runs carry an episode id per query. Queries within an episode
    # share a context set and generative parameters, so they must be resampled
    # as a block -- treating them as independent would report an interval far
    # narrower than the data supports. ICI runs have one prediction per donor
    # and no episode key, so each row is its own cluster.
    episode = aggregate.get("episode")
    if episode is not None:
        episode = episode.long().flatten()
    return probability, target, episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    runs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
    for path in args.predictions:
        runs[path.stem] = load(path)

    print(f"AUROC with {args.bootstrap}-sample bootstrap 95% CI\n")
    for name, (probability, target, episode) in runs.items():
        count = target.numel()
        members = cluster_members(episode)
        generator = torch.Generator().manual_seed(args.seed)
        values = []
        for _ in range(args.bootstrap):
            index = resample_index(count, members, generator)
            resampled = target[index]
            if resampled.sum() == 0 or resampled.sum() == resampled.numel():
                continue
            values.append(auroc(probability[index], resampled))
        spread = torch.tensor(values)
        unit = (
            f"{len(members)} episodes" if members is not None else f"n={count}"
        )
        print(
            f"  {name}: {auroc(probability, target):.4f} "
            f"[{spread.quantile(0.025):.3f}, {spread.quantile(0.975):.3f}] "
            f"({unit}, {count} predictions, positive={int((target == 1).sum())})"
        )
    if any(episode is not None for _, _, episode in runs.values()):
        print(
            "\n  Runs reporting episodes use a cluster bootstrap: whole episodes are"
            "\n  resampled, because queries inside one episode share a context set."
        )

    if len(runs) < 2:
        return

    print("\nPaired bootstrap P(row beats column) -- 0.5 means indistinguishable\n")
    for left, right in itertools.combinations(runs, 2):
        left_probability, left_target, left_episode = runs[left]
        right_probability, right_target, right_episode = runs[right]
        if not torch.equal(left_target, right_target):
            print(f"  {left} vs {right}: SKIPPED (different target order)")
            continue
        if (left_episode is None) != (right_episode is None) or (
            left_episode is not None
            and right_episode is not None
            and not torch.equal(left_episode, right_episode)
        ):
            print(f"  {left} vs {right}: SKIPPED (different episode grouping)")
            continue
        count = left_target.numel()
        members = cluster_members(left_episode)
        generator = torch.Generator().manual_seed(args.seed)
        wins = total = 0
        for _ in range(args.bootstrap):
            index = resample_index(count, members, generator)
            resampled = left_target[index]
            if resampled.sum() == 0 or resampled.sum() == resampled.numel():
                continue
            total += 1
            if auroc(left_probability[index], resampled) > auroc(
                right_probability[index], resampled
            ):
                wins += 1
        print(f"  {left} vs {right}: {wins / max(1, total):.2f}")


if __name__ == "__main__":
    main()
