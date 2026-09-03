"""Canonical ICI evaluation: multi-seed CV, confidence intervals, external cohort.

The v21 retrieval investigation concluded that every architecture comparison
made on this project was statistically indistinguishable from noise: single
SEED42 5-fold CV on 87 donors, reported as a bare point estimate. Four of the
five available seed partitions were never used, and the external GSE285888
cohort was never evaluated. This script is the replacement protocol.

It reports three distinct quantities, which answer different questions:

  * per-seed AUROC        -- one 5-fold CV over all 87 donors, per partition
  * across-seed mean +/- SD -- how much the result moves when only the
    partition and training run change. Large SD means a single-seed number is
    not reproducible, independent of cohort size.
  * pooled bootstrap CI   -- resampling the 87 donors. This estimates cohort
    sampling error and, crucially, does NOT shrink as seeds are added: every
    seed reuses the same 87 people. Only a larger or additional cohort moves
    it.

The external cohort is the one genuinely independent read, so it is reported
separately rather than pooled into the CV numbers.

Usage:
    python scripts/evaluate_protocol.py --run-root checkpoints/20260729_xxxx
    python scripts/evaluate_protocol.py --predictions predictions/run_seed*.pt
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

from src.utils.metrics import auroc, bootstrap_auroc_interval, log_loss  # noqa: E402

DEFAULT_SEEDS = (42, 1234, 2026, 271828, 314159)


def load_aggregate(path: Path) -> tuple[torch.Tensor, torch.Tensor, list]:
    payload = torch.load(path, weights_only=False)
    aggregate = payload["validation_aggregate"]
    return (
        aggregate["probabilities"][:, 1].float(),
        aggregate["target"].long().flatten(),
        list(aggregate.get("donor_id", [])),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One prediction file per seed, as written by scripts/test.py.",
    )
    parser.add_argument(
        "--external",
        type=Path,
        default=None,
        help="Optional prediction file holding external-cohort results.",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed_auroc: list[float] = []
    per_seed_log_loss: list[float] = []
    donor_scores: dict[str, list[float]] = {}
    donor_target: dict[str, int] = {}

    print("Per-seed 5-fold cross-validation\n")
    print(f"{'prediction file':<52} {'AUROC':>7} {'LogLoss':>9} {'n':>5}")
    print("-" * 76)
    for path in args.predictions:
        probability, target, donors = load_aggregate(path)
        value = auroc(probability, target)
        loss = log_loss(probability, target)
        per_seed_auroc.append(value)
        per_seed_log_loss.append(loss)
        print(f"{path.name:<52} {value:>7.4f} {loss:>9.4f} {target.numel():>5}")
        for i, donor in enumerate(donors):
            donor_scores.setdefault(str(donor), []).append(float(probability[i]))
            donor_target[str(donor)] = int(target[i])

    auroc_tensor = torch.tensor(per_seed_auroc)
    loss_tensor = torch.tensor(per_seed_log_loss)
    seeds = len(per_seed_auroc)
    print("-" * 76)
    if seeds > 1:
        auroc_sd = auroc_tensor.std(unbiased=True).item()
        print(
            f"{'across-seed mean +/- SD':<52} "
            f"{auroc_tensor.mean():>7.4f} {loss_tensor.mean():>9.4f}"
        )
        print(
            f"{'':<52} {'+/-' + format(auroc_sd, '.4f'):>7} "
            f"{'+/-' + format(loss_tensor.std(unbiased=True).item(), '.4f'):>9}"
        )
        print(
            f"\nSeed-to-seed AUROC range: "
            f"{auroc_tensor.min():.4f} .. {auroc_tensor.max():.4f} "
            f"(spread {auroc_tensor.max() - auroc_tensor.min():.4f})"
        )
        # Standard error of the across-seed mean. This shrinks with more seeds
        # but only bounds partition/training noise -- never cohort size.
        print(
            f"Standard error of the mean over {seeds} seeds: "
            f"{auroc_sd / math.sqrt(seeds):.4f}"
        )
    else:
        print("Only one seed supplied -- across-seed variability is unmeasured.")

    if donor_scores:
        names = sorted(donor_scores)
        averaged = torch.tensor([sum(donor_scores[n]) / len(donor_scores[n]) for n in names])
        targets = torch.tensor([donor_target[n] for n in names], dtype=torch.long)
        low, high = bootstrap_auroc_interval(
            averaged, targets, samples=args.bootstrap, seed=args.seed
        )
        print(
            f"\nSeed-averaged per-donor prediction ({targets.numel()} donors): "
            f"AUROC {auroc(averaged, targets):.4f}  95% CI [{low:.3f}, {high:.3f}]"
        )
        print(
            "This CI reflects sampling error over the cohort itself. Adding seeds"
            "\ndoes not narrow it -- only more donors do."
        )

    if args.external is not None:
        probability, target, _ = load_aggregate(args.external)
        low, high = bootstrap_auroc_interval(
            probability, target, samples=args.bootstrap, seed=args.seed
        )
        print(
            f"\nExternal cohort ({target.numel()} donors, "
            f"{int((target == 1).sum())} positive): "
            f"AUROC {auroc(probability, target):.4f}  95% CI [{low:.3f}, {high:.3f}]  "
            f"LogLoss {log_loss(probability, target):.4f}"
        )
        print("The only genuinely independent read; never pool it with CV results.")


if __name__ == "__main__":
    main()
