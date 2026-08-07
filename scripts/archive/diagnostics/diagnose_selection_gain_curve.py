"""T1-C step 1: what would better cell selection actually buy?

T1-A/T1-B established that every selection score in the model is at chance for
finding responsive cells, that a held-out supervised probe reaches ~0.70, and
that perfect (oracle) selection lifts covariance AUROC from 0.5704 to 0.8931.
What none of that says is how much of the 0.57 -> 0.89 span a *partially*
correct selection recovers. If the curve is flat until selection is nearly
perfect, building a new selection mechanism is not worth it.

This prices the change without training anything, two ways:

  mixing curve   -- select exactly as many cells as there are responsive ones,
                    but draw a fraction p of them from the true responsive set
                    and the rest at random. p at the base rate reproduces
                    random selection; p = 1 reproduces the oracle. The curve
                    between them is the gain curve.
  realistic point -- select by the held-out Fisher LDA score, the best stand-in
                    available for what a learned mechanism might achieve, and
                    measure the covariance AUROC that results.

Both are scored through the model's own relation scorer (prototype_cosine, the
strongest relation per T1-0) on the model's own covariance sketch, so only the
cell selection varies.

Usage:
    python scripts/diagnose_selection_gain_curve.py \\
        --config configs/train_v22_medium.yaml --val-episodes 400
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_oracle_slot_alignment import query_index  # noqa: E402
from src.utils.metrics import auroc  # noqa: E402
from src.utils.utils import build_datamodule, build_model, merge_train_config  # noqa: E402

PURITIES = (0.11, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def episode_bootstrap(values, samples=2000, seed=0):
    if not values:
        return float("nan"), float("nan")
    v = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, v.numel(), (samples, v.numel()), generator=generator)
    means = v[index].mean(dim=1)
    return float(means.quantile(0.025)), float(means.quantile(0.975))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_v22_medium.yaml")
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_train_config(args.config)
    L.seed_everything(int(config.get("seed", 42)), workers=True)
    if args.val_episodes is not None:
        config["data"].setdefault("val_dataset_kwargs", {})["episodes_per_epoch"] = args.val_episodes

    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    model = build_model(config).model.to("cuda").eval()
    aggregator, clf = model.aggregator, model.meta_classifier
    clf.covariance_relation_mode = "prototype_cosine"
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    curve = defaultdict(list)
    realistic: list[float] = []
    realistic_precision: list[float] = []
    all_cell: list[float] = []

    def sketch_from(selection: list[torch.Tensor], bags) -> torch.Tensor:
        out = []
        for bag, keep in zip(bags, selection):
            chosen = bag[keep]
            out.append(aggregator._covariance_sketch(chosen - chosen.mean(dim=0, keepdim=True)))
        return torch.stack(out)

    def score(values, y, context, query) -> float:
        logits, _ = clf._covariance_relation_scores(values[context], y[context], values[query])
        margin = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits
        return auroc(margin.float().cpu(), y[query].cpu())

    with torch.no_grad():
        for index in range(len(datamodule.val_dataset)):
            episode = datamodule.val_dataset.diagnostic_episode(index)
            if episode.response_task != "covariance" or episode.responsive_instance_mask is None:
                continue
            x, y, mask = episode.x, episode.y, episode.responsive_instance_mask
            bags = list(x.unbind(0)) if isinstance(x, torch.Tensor) else list(x)
            query = query_index(y)
            context = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
            context[query] = False
            if not (y[context] == 0).any() or not (y[context] == 1).any():
                continue

            responsive = [m.bool() for m in mask]
            if any(r.all() or not r.any() for r in responsive):
                continue

            # all cells, i.e. what the model effectively does today
            value = score(
                torch.stack([
                    aggregator._covariance_sketch(b - b.mean(dim=0, keepdim=True)) for b in bags
                ]), y, context, query,
            )
            if value == value:
                all_cell.append(value)

            for purity in PURITIES:
                selection = []
                for bag, keep in zip(bags, responsive):
                    positive = torch.nonzero(keep, as_tuple=False).flatten()
                    negative = torch.nonzero(~keep, as_tuple=False).flatten()
                    budget = positive.numel()
                    take_positive = min(positive.numel(), max(1, int(round(purity * budget))))
                    take_negative = min(negative.numel(), budget - take_positive)
                    picked = [
                        positive[torch.randperm(positive.numel(), generator=generator, device=positive.device)[:take_positive]]
                    ]
                    if take_negative > 0:
                        picked.append(
                            negative[torch.randperm(negative.numel(), generator=generator, device=negative.device)[:take_negative]]
                        )
                    selection.append(torch.cat(picked))
                value = score(sketch_from(selection, bags), y, context, query)
                if value == value:
                    curve[purity].append(value)

            # realistic: rank cells by a held-out Fisher LDA and keep the top k
            selection = []
            hits = 0
            budget_total = 0
            for bag, keep in zip(bags, responsive):
                f = bag.float()
                half = torch.zeros(f.shape[0], dtype=torch.bool, device=f.device)
                half[torch.randperm(f.shape[0], generator=generator, device=f.device)[: f.shape[0] // 2]] = True
                fit_positive, fit_negative = keep & half, (~keep) & half
                budget = int(keep.sum())
                if int(fit_positive.sum()) < 2 or int(fit_negative.sum()) < 2:
                    selection.append(torch.nonzero(keep, as_tuple=False).flatten())
                    hits += budget
                    budget_total += budget
                    continue
                delta = f[fit_positive].mean(dim=0) - f[fit_negative].mean(dim=0)
                direction = delta / f[half].var(dim=0, unbiased=False).clamp_min(1e-6)
                order = (f @ direction).argsort(descending=True)[:budget]
                selection.append(order)
                hits += int(keep[order].sum())
                budget_total += budget
            value = score(sketch_from(selection, bags), y, context, query)
            if value == value:
                realistic.append(value)
                realistic_precision.append(hits / max(1, budget_total))

    if not all_cell:
        print("No usable covariance episodes found.")
        return

    def summarise(label, values):
        mean = sum(values) / len(values)
        low, high = episode_bootstrap(values, args.bootstrap)
        return f"  {label:<34} {mean:>7.4f}  [{low:.3f}, {high:.3f}]  ({len(values)} eps)"

    print(f"Covariance AUROC vs cell-selection purity ({len(all_cell)} episodes)\n")
    print(summarise("all cells (what we do today)", all_cell))
    print()
    print(f"  {'selection purity':<34} {'AUROC':>7}  {'95% CI':<16}")
    print("  " + "-" * 66)
    for purity in PURITIES:
        if curve[purity]:
            print(summarise(f"purity = {purity:.2f}", curve[purity]))
    print()
    if realistic:
        precision = sum(realistic_precision) / len(realistic_precision)
        print(summarise(f"held-out LDA selection (prec {precision:.2f})", realistic))
    print(
        "\nRead the slope near the low-purity end. If AUROC barely moves until purity\n"
        "is high, a mechanism that only reaches moderate purity buys almost nothing\n"
        "and Tier 1 should be dropped."
    )


if __name__ == "__main__":
    main()
