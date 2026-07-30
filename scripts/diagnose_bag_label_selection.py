"""T1-C step 2: can bag labels alone find the responsive cells? (Tier 1 go/no-go)

T1-C step 1 showed the gain curve is linear, so partial cell selection pays:
at purity 0.40 the covariance branch goes 0.5571 -> 0.6638. But that purity came
from a Fisher discriminant fitted on cell-level responsive labels, which the
model never has. It only ever sees bag-level R/NR.

So the remaining question, and the gate on all of Tier 1, is whether a direction
learned from bag labels alone reaches useful purity. Fitting is done on context
bags and scoring on query bags, and cell-level labels are used only to score the
result, never to fit.

Two rules, because the effect they have to detect is not the same:

  bag_label_lda   mean difference between cells of R bags and cells of NR bags,
                  whitened. This is the obvious rule, and for composition/state
                  effects (which shift cells) it should work. For covariance
                  episodes the responsive component changes DISPERSION rather
                  than position, so a mean difference may be near zero by
                  construction -- worth confirming rather than assuming.

  bag_label_csp   per-bag-centred cells, weighted by log variance ratio between
                  R and NR bags per dimension, scored as weighted squared
                  deviation. This targets dispersion directly, which is the
                  effect covariance episodes actually carry. Diagonal rather
                  than a full generalised eigenproblem, for stability at 512
                  dimensions.

Decision rule set in advance (current_status.md Tier 1):
  purity >= 0.30  -> the architecture change is justified
  purity <= 0.15  -> Tier 1 ends here

Usage:
    python scripts/diagnose_bag_label_selection.py \\
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
from src.utils.utils import build_datamodule, merge_train_config  # noqa: E402


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
    parser.add_argument("--task", default="covariance")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_train_config(args.config)
    L.seed_everything(int(config.get("seed", 42)), workers=True)
    if args.val_episodes is not None:
        config["data"].setdefault("val_dataset_kwargs", {})["episodes_per_epoch"] = args.val_episodes

    datamodule = build_datamodule(config)
    datamodule.setup("fit")

    ranking = defaultdict(list)
    purity = defaultdict(list)
    base_rates: list[float] = []

    with torch.no_grad():
        for index in range(len(datamodule.val_dataset)):
            episode = datamodule.val_dataset.diagnostic_episode(index)
            if episode.response_task != args.task or episode.responsive_instance_mask is None:
                continue
            x, y, mask = episode.x, episode.y, episode.responsive_instance_mask
            bags = list(x.unbind(0)) if isinstance(x, torch.Tensor) else list(x)
            query = query_index(y)
            context = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
            context[query] = False

            context_positions = torch.nonzero(context, as_tuple=False).flatten().tolist()
            responder = [p for p in context_positions if int(y[p]) == 1]
            non_responder = [p for p in context_positions if int(y[p]) == 0]
            if not responder or not non_responder:
                continue

            # Fit on context bags only. Bag labels are the sole supervision.
            raw_r = torch.cat([bags[p].float() for p in responder])
            raw_n = torch.cat([bags[p].float() for p in non_responder])
            pooled_variance = torch.cat([raw_r, raw_n]).var(dim=0, unbiased=False).clamp_min(1e-6)
            lda_direction = (raw_r.mean(dim=0) - raw_n.mean(dim=0)) / pooled_variance

            centred_r = torch.cat([bags[p].float() - bags[p].float().mean(dim=0) for p in responder])
            centred_n = torch.cat([bags[p].float() - bags[p].float().mean(dim=0) for p in non_responder])
            variance_r = centred_r.var(dim=0, unbiased=False).clamp_min(1e-6)
            variance_n = centred_n.var(dim=0, unbiased=False).clamp_min(1e-6)
            csp_weight = torch.log(variance_r / variance_n)

            for position in query.tolist():
                bag = bags[position].float()
                responsive = mask[position].bool()
                if responsive.all() or not responsive.any():
                    continue
                base_rates.append(float(responsive.float().mean()))
                centred = bag - bag.mean(dim=0)
                scores = {
                    "bag_label_lda": bag @ lda_direction,
                    "bag_label_csp": (centred.square() * csp_weight).sum(dim=-1),
                }
                target = responsive.long().cpu()
                budget = int(responsive.sum())
                for name, score in scores.items():
                    value = auroc(score.cpu(), target)
                    if value == value:
                        ranking[name].append(value)
                    top = score.argsort(descending=True)[:budget]
                    purity[name].append(int(responsive[top].sum()) / budget)

    if not base_rates:
        print(f"No usable '{args.task}' episodes found.")
        return

    base = sum(base_rates) / len(base_rates)
    print(
        f"Task '{args.task}': {len(base_rates)} query bags, "
        f"responsive = {base:.1%} of a bag\n"
        "Fitted on context bags with bag labels only; cell labels used only to score.\n"
    )
    print(f"  {'rule':<18} {'AUROC':>7}  {'95% CI':<16} {'purity@k':>9}")
    print("  " + "-" * 56)
    for name in ("bag_label_lda", "bag_label_csp"):
        if not ranking[name]:
            continue
        mean = sum(ranking[name]) / len(ranking[name])
        low, high = episode_bootstrap(ranking[name], args.bootstrap)
        pure = sum(purity[name]) / len(purity[name])
        print(f"  {name:<18} {mean:>7.4f}  [{low:.3f}, {high:.3f}] {pure:>9.3f}")
    print(f"  {'(random)':<18} {0.5:>7.4f}  {'':16} {base:>9.3f}")

    best = max(
        (sum(purity[n]) / len(purity[n]) for n in purity if purity[n]), default=0.0
    )
    print(f"\n  Best bag-label purity: {best:.3f}  (random {base:.3f})")
    if best >= 0.30:
        print("  >= 0.30 -> architecture change is justified.")
    elif best <= 0.15:
        print("  <= 0.15 -> Tier 1 ends here; move to Tier 2/3.")
    else:
        print("  between 0.15 and 0.30 -> marginal; weigh cost against the gain curve.")


if __name__ == "__main__":
    main()
