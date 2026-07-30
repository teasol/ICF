"""T1-1: locate where the covariance branch loses signal.

Written to test three causes of an apparent 0.61-vs-0.89 gap, and it ended up
refuting the premise as well as two of the three hypotheses. Kept because the
measurements it makes are the ones that settle the question.

What it reports:

  (a) the learned fusion gates, read straight out of the checkpoint, so
      "the branch is switched off" is a matter of fact rather than inference;
  (b) each relation mode scored on the all-cell covariance sketch -- the view
      the model actually has -- with every other weight held fixed;
  (c) the model's end-to-end AUROC on the same episodes, for reference.

Findings on the v22 baseline (see current_status.md §3):

  gates are open           covariance_residual 0.295 x ridge 2.390 = 0.705
  every relation mode ties learned_head .513 / prototype_cosine .507 /
                           multiscale_rbf .508 / standardized_distance .515
  full model               0.620, above all of them

So neither fusion dilution nor the choice of relation head is the bottleneck.
All-cell covariance simply does not carry much signal (ceiling 0.5704), and
the 0.8931 figure comes from a descriptor that selects cells using
episode.responsive_instance_mask -- diagnostic-only ground truth the model
never receives. The real problem is identifying which ~12% of cells respond.

Usage:
    python scripts/diagnose_covariance_utilisation.py \
        --checkpoint checkpoints/<ts>/v22_medium_fixed/<best>.ckpt \
        --config configs/train_v22_medium.yaml --val-episodes 1000
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

RELATION_MODES = ("learned_head", "prototype_cosine", "multiscale_rbf", "standardized_distance")


def episode_bootstrap(values, weights, samples=2000, seed=0):
    """Interval on a query-weighted mean, resampling whole episodes."""
    if not values:
        return float("nan"), float("nan")
    v = torch.tensor(values, dtype=torch.float64)
    w = torch.tensor(weights, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    index = torch.randint(0, v.numel(), (samples, v.numel()), generator=generator)
    means = (v[index] * w[index]).sum(dim=1) / w[index].sum(dim=1)
    return float(means.quantile(0.025)), float(means.quantile(0.975))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_v22_medium.yaml")
    parser.add_argument("--val-episodes", type=int, default=None)
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
    interface = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    interface.on_load_checkpoint(checkpoint)
    interface.load_state_dict(checkpoint["state_dict"])
    model = interface.model.to("cuda").eval()
    clf = model.meta_classifier

    # ---- (a) what did the gates actually learn? -------------------------------
    print("=" * 78)
    print("(a) Learned fusion gates — are they open or collapsed?")
    print("=" * 78)
    covariance_residual = torch.sigmoid(clf.covariance_residual_logit).item()
    ridge_scale = clf.covariance_ridge_log_scale.exp().clamp(0.1, 100.0).item()
    population = clf._floored_residual_scale(
        clf.population_residual_logit, clf.minimum_population_residual_scale
    ).item()
    tail = clf._floored_residual_scale(
        clf.tail_residual_logit, clf.minimum_tail_residual_scale
    ).item()
    fusion = torch.sigmoid(clf.fusion_residual_logit).item()
    relation_residual = float(clf.covariance_relation_residual_scale)

    print(f"  {'gate':<34} {'value':>9}   {'note'}")
    print("  " + "-" * 74)
    print(f"  {'covariance_residual_scale':<34} {covariance_residual:>9.4f}   sigmoid, gates the ridge term")
    print(f"  {'covariance_ridge_scale':<34} {ridge_scale:>9.4f}   exp, clamped [0.1, 100]")
    print(f"  {'covariance_relation_residual_scale':<34} {relation_residual:>9.4f}   FIXED by config (CSP head)")
    print(f"  {'population_scale':<34} {population:>9.4f}   for reference")
    print(f"  {'tail_scale':<34} {tail:>9.4f}   for reference")
    print(f"  {'fusion_scale':<34} {fusion:>9.4f}   for reference")
    print(
        f"\n  Effective multiplier on the covariance ridge term: "
        f"{covariance_residual * ridge_scale:.4f}"
    )

    # ---- (b)/(c) re-score the branch under each relation mode ------------------
    print("\n" + "=" * 78)
    print("(b) Covariance relation modes on the all-cell sketch the model really sees")
    print("    (trained weights, same episodes — only the relation changes)")
    print("    Reference: all-cell ceiling 0.5704 [0.550, 0.592]; oracle-cell 0.8931")
    print("=" * 78)

    per_episode = defaultdict(list)
    per_weight = defaultdict(list)
    full_model = []
    full_weight = []
    original_mode = clf.covariance_relation_mode

    with torch.no_grad():
        for index in range(len(datamodule.val_dataset)):
            episode = datamodule.val_dataset.diagnostic_episode(index)
            if episode.response_task != "covariance":
                continue
            x, y = episode.x, episode.y
            query = query_index(y)
            context = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
            context[query] = False

            # Use the flat sketch, matching the ceiling diagnostic. The
            # aggregator's covariance_matrix is [bags, d, d]; the relation
            # scorer reads a leading 3rd dim as the episode axis and would
            # mis-broadcast it. Cells are NOT oracle-filtered here -- this is
            # the all-cell view the model actually has.
            values = torch.stack([
                model.aggregator._covariance_sketch(bag - bag.mean(dim=0, keepdim=True))
                for bag in x
            ])
            for mode in RELATION_MODES:
                clf.covariance_relation_mode = mode
                logits, _ = clf._covariance_relation_scores(
                    values[context], y[context], values[query]
                )
                score = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits
                value = auroc(score.float().cpu(), y[query].cpu())
                if value == value:  # skip NaN (single-class query set)
                    per_episode[mode].append(value)
                    per_weight[mode].append(int(query.numel()))
            clf.covariance_relation_mode = original_mode

            # the model's actual end-to-end prediction on the same episode
            logits = model(x, y, mask_index=query)
            score = (logits[:, 1] - logits[:, 0]).float().cpu()
            value = auroc(score, y[query].cpu())
            if value == value:
                full_model.append(value)
                full_weight.append(int(query.numel()))

    def summarise(name, values, weights):
        if not values:
            print(f"  {name:<24} (no valid episodes)")
            return
        v = torch.tensor(values, dtype=torch.float64)
        w = torch.tensor(weights, dtype=torch.float64)
        mean = float((v * w).sum() / w.sum())
        low, high = episode_bootstrap(values, weights, args.bootstrap)
        print(f"  {name:<24} {mean:>7.4f}  [{low:.3f}, {high:.3f}]  ({len(values)} eps)")

    print(f"  {'relation mode':<24} {'AUROC':>7}  {'95% CI':<16} episodes")
    print("  " + "-" * 62)
    for mode in RELATION_MODES:
        summarise(mode + ("  (current)" if mode == original_mode else ""), per_episode[mode], per_weight[mode])
    print()
    summarise("FULL MODEL (end-to-end)", full_model, full_weight)

    print("\n" + "=" * 78)
    print("Reading this")
    print("=" * 78)
    print(
        "  Gates near zero would mean the branch is switched off. Measured 0.705\n"
        "  on the v22 baseline, so it is not.\n"
        "  A parameter-free relation clearly beating learned_head would implicate\n"
        "  the head. Measured: all four tie near 0.51, so it does not.\n"
        "  Every mode landing near chance while the oracle-cell descriptor reaches\n"
        "  0.89 means the loss is in cell SELECTION, not in the relation. That is\n"
        "  the current reading -- see Tier 1 in current_status.md."
    )


if __name__ == "__main__":
    main()
