"""E7 re-test: supervised component-selection upper bound (no training).

Re-runs the E7 gate from docs/current_status.md SS16 /
docs/history/architecture_v28_analysis_ceiling_and_gates.md SS6.1:

For the same v24 population-slot split, how well can a CELL-LABEL-supervised
Fisher discriminant select responsive cells, measured held-out? This is the
principled upper bound for Path B (learnable differentiable split with
oracle-mask auxiliary supervision): if even supervised cell labels cannot
select responsive cells with purity >= 0.50, Path B is closed.

Method (faithful to the documented E7):
  * per bag, split the cells in half (fit / held), 4 independent random
    splits, averaged
  * fit a Fisher LDA direction on the fit half using responsive labels
        d = (mean_pos - mean_neg) / variance
  * score the held-out half -> held-out per-cell AUROC
  * purity@k = precision of the top-k held-out cells by the Fisher score, for
    the fractions the model actually keeps (tail/rare: 1..20%)
  * aggregates over episodes, per task and ALL, bag-level bootstrap 95% CI

Gate (SS16): purity >= 0.50 -> Path B proceed; < 0.30 -> Path B closed;
in between -> inconclusive. Previous E7: ALL purity 0.3351 (inconclusive),
covariance alone 0.2726 (below the discard bar).

No model is loaded -- this is pure cell geometry on the val stream.

Usage:
    python scripts/diagnose_component_selection_bound.py \
        --config configs/train_v24_medium_bag_proj_residual.yaml \
        --episodes 1000
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import RESPONSE_TASK_NAMES  # noqa: E402
from src.utils.metrics import auroc  # noqa: E402
from src.utils.utils import build_datamodule, merge_train_config  # noqa: E402

FRACTIONS = (0.01, 0.05, 0.10, 0.15, 0.20)


def bag_bootstrap(values: list[float], samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    v = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, v.numel(), (samples, v.numel()), generator=generator)
    means = v[index].mean(dim=1)
    return float(means.quantile(0.025)), float(means.quantile(0.975))


def held_out_fisher(
    cells: torch.Tensor,
    responsive: torch.Tensor,
    splits: int,
) -> tuple[float, dict[float, float]] | None:
    """Per-bag held-out Fisher LDA, `splits` independent half-splits averaged.

    Returns (mean held-out AUROC, mean purity@k per fraction) or None if the
    bag is unusable (too few cells of either class in a fit half).
    """
    n = cells.shape[0]
    aurocs: list[float] = []
    purity: dict[float, list[float]] = defaultdict(list)
    for _ in range(splits):
        half = torch.zeros(n, dtype=torch.bool, device=cells.device)
        half[torch.randperm(n, device=cells.device)[: n // 2]] = True
        fit_pos = responsive & half
        fit_neg = (~responsive) & half
        if int(fit_pos.sum()) < 2 or int(fit_neg.sum()) < 2:
            continue
        delta = cells[fit_pos].mean(dim=0) - cells[fit_neg].mean(dim=0)
        direction = delta / cells[half].var(dim=0, unbiased=False).clamp_min(1e-6)
        held = ~half
        held_resp = responsive[held]
        if not (bool(held_resp.any()) and bool((~held_resp).any())):
            continue
        score = cells[held] @ direction
        value = auroc(score.cpu(), held_resp.long().cpu())
        if value == value:
            aurocs.append(value)
        order = score.argsort(descending=True)
        n_held = int(held.sum())
        for fraction in FRACTIONS:
            keep = max(1, int(round(fraction * n_held)))
            hit = int(held_resp[order[:keep]].sum())
            purity[fraction].append(hit / keep)
    if not aurocs:
        return None
    return (
        sum(aurocs) / len(aurocs),
        {f: sum(purity[f]) / len(purity[f]) for f in purity},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/train_v24_medium_bag_proj_residual.yaml"),
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    config["data"].setdefault("val_dataset_kwargs", {})["episodes_per_epoch"] = args.episodes
    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    val_dataset = datamodule.val_dataset

    # task_key -> {bag_auroc: [...], purity: {fraction: [...]}, base: [...]}
    stats: dict[str, dict] = defaultdict(
        lambda: {"auroc": [], "purity": {f: [] for f in FRACTIONS}, "base": []}
    )
    evaluated = 0
    with torch.no_grad():
        for episode_index in range(args.episodes):
            episode = val_dataset.diagnostic_episode(episode_index)
            mask = episode.responsive_instance_mask
            if mask is None:
                continue
            key = (
                episode.response_task
                if episode.response_task in RESPONSE_TASK_NAMES
                else "ALL"
            )
            x = episode.x
            for bag, bag_mask in zip(x.unbind(0), mask.unbind(0)):
                responsive = bag_mask.bool()
                if not (bool(responsive.any()) and bool((~responsive).any())):
                    continue
                result = held_out_fisher(bag, responsive, args.splits)
                if result is None:
                    continue
                auroc_mean, purity = result
                stats[key]["auroc"].append(auroc_mean)
                stats[key]["base"].append(float(responsive.float().mean().item()))
                for fraction, value in purity.items():
                    stats[key]["purity"][fraction].append(value)
            evaluated += 1
            if (episode_index + 1) % 200 == 0:
                print(f"  ... {episode_index + 1}/{args.episodes} episodes", flush=True)

    # ALL = pooled over every bag regardless of task.
    all_stats = {"auroc": [], "purity": {f: [] for f in FRACTIONS}, "base": []}
    for key in list(stats.keys()):
        if key == "ALL":
            continue
        all_stats["auroc"].extend(stats[key]["auroc"])
        all_stats["base"].extend(stats[key]["base"])
        for f in FRACTIONS:
            all_stats["purity"][f].extend(stats[key]["purity"][f])
    stats["ALL"] = all_stats

    order = [k for k in ("ALL", *RESPONSE_TASK_NAMES) if k in stats]
    print(f"\nE7 supervised component-selection bound — {evaluated} episodes, "
          f"{args.splits} splits/bag, seed {args.seed}\n")
    print(f"{'task':<13} {'bags':>7} {'base':>7} {'heldAUROC':>9} "
          + "".join(f"{f'p@{int(f*100)}%':>8}" for f in FRACTIONS))
    print("-" * (13 + 7 + 7 + 9 + 8 * len(FRACTIONS)))

    rows = []
    for key in order:
        s = stats[key]
        n_bags = len(s["auroc"])
        if n_bags == 0:
            continue
        base = sum(s["base"]) / n_bags
        auroc_mean = sum(s["auroc"]) / n_bags
        print(
            f"{key:<13} {n_bags:>7,} {base:>7.3f} {auroc_mean:>9.4f} "
            + "".join(
                f"{sum(s['purity'][f]) / n_bags:>8.3f}" for f in FRACTIONS
            )
        )
        row = {
            "task": key,
            "bags": n_bags,
            "base_rate": base,
            "held_out_auroc": auroc_mean,
        }
        for f in FRACTIONS:
            values = s["purity"][f]
            mean = sum(values) / len(values)
            low, high = bag_bootstrap(values, args.bootstrap, args.seed)
            row[f"purity@{int(f*100)}"] = mean
            row[f"purity@{int(f*100)}_ci"] = f"[{low:.3f}, {high:.3f}]"
            print(f"    purity@{int(f*100):>2}% CI: [{low:.3f}, {high:.3f}] "
                  f"(base {base:.3f}, enrichment {mean / base:.2f}x)")
        rows.append(row)

    # Gate read on the primary metric (ALL purity, the documented E7 figure).
    all_p = stats["ALL"]["purity"]
    primary = 0.10
    mean_p = sum(all_p[primary]) / len(all_p[primary])
    print(f"\nGate (ALL purity@10%): {mean_p:.4f} -> "
          f"{'PROCEED (>=0.50)' if mean_p >= 0.50 else 'DISCARD (<0.30)' if mean_p < 0.30 else 'INCONCLUSIVE (0.30-0.50)'}")

    if args.output is None:
        args.output = Path("logs/e7_retest_20260803.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
