"""T1-A: does the sparse-evidence machinery actually find the responsive cells?

T1-0/T1-1 narrowed the covariance shortfall to one thing: knowing which cells
respond is worth 0.57 -> 0.89 on covariance episodes, and responsive cells are
only ~12% of a bag. Every selection mechanism in the model ranks cells by some
score and keeps a top fraction, but nothing has ever checked whether that
ranking corresponds to responsiveness.

This scores each candidate ranking against `episode.responsive_instance_mask`
(diagnostic-only ground truth) and asks two questions:

  * ranking quality -- AUROC of the score for predicting "this cell responds".
    0.5 means the criterion carries no information about responsiveness at all,
    in which case no choice of cutoff can help.
  * operating point  -- precision and recall at the fractions the model really
    uses, against the base rate a random pick would achieve.

Scores compared:
  novelty            1 - max cosine similarity to the slot anchors. This is
                     what the aggregator's tail tokens actually select on.
  outlier_distance   L2 distance from the bag centroid, the criterion the
                     architecture docs describe for Top-1% sparse evidence.
  studentized        the same distance after per-bag z-scoring, i.e. measured
                     on the representation the classifier consumes.
  class_memory       max learned similarity to the class-memory tokens, the
                     score the meta-classifier's rare-evidence path pools over.
                     This one is learned rather than geometric, so it needs a
                     trained checkpoint (--checkpoint); skipped without one.
  lda_probe          CHEATING upper bound: Fisher discriminant fitted on this
                     bag using the responsive labels themselves, then scored on
                     the same cells. No selection rule can beat it. If it is
                     near 0.5, responsive cells are not separable from cell
                     features at all and the oracle-cell ceiling is unreachable
                     in principle rather than merely unreached.

Usage:
    python scripts/diagnose_cell_selection.py \\
        --config configs/archive/v22/train_v22_medium.yaml --val-episodes 400
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_oracle_slot_alignment import query_index  # noqa: E402
from src.utils.metrics import auroc  # noqa: E402
from src.utils.utils import build_datamodule, build_model, merge_train_config  # noqa: E402

# The fractions the model actually keeps: aggregator tail_fractions and the
# meta-classifier's rare_evidence_fractions.
FRACTIONS = (0.01, 0.05, 0.10, 0.15, 0.20)


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
    parser.add_argument("--config", type=Path, default=ROOT / "configs/archive/v22/train_v22_medium.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Trained checkpoint; enables the learned class_memory score.")
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--task", default="covariance", help="Response task to analyse.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--probe", action="store_true",
                        help="Add the cheating LDA upper bound on separability.")
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
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        interface.on_load_checkpoint(checkpoint)
        interface.load_state_dict(checkpoint["state_dict"])
    model = interface.model.to("cuda").eval()
    aggregator = model.aggregator
    clf = model.meta_classifier

    ranking = defaultdict(list)                      # score -> per-episode AUROC
    precision = defaultdict(lambda: defaultdict(list))
    recall = defaultdict(lambda: defaultdict(list))
    base_rates: list[float] = []

    with torch.no_grad():
        for index in range(len(datamodule.val_dataset)):
            episode = datamodule.val_dataset.diagnostic_episode(index)
            if episode.response_task != args.task or episode.responsive_instance_mask is None:
                continue
            x, y, mask = episode.x, episode.y, episode.responsive_instance_mask
            bags = list(x.unbind(0)) if isinstance(x, torch.Tensor) else list(x)
            query = query_index(y)
            context = torch.ones(len(bags), dtype=torch.bool, device=bags[0].device)
            context[query] = False
            anchors = aggregator._context_anchors(bags, context)

            # The learned class-memory score is only defined for query bags:
            # the memories are built from labelled context, so scoring context
            # cells with them would leak. Restrict every score to the query
            # bags so all four are compared on identical cells.
            memories = None
            if args.checkpoint is not None:
                representation = aggregator(x, context_mask=context)
                context_representation = {
                    name: tokens[context] for name, tokens in representation.items()
                }
                memories = clf._class_memories(context_representation, y[context])
            scored = query.tolist() if args.checkpoint is not None else range(len(bags))

            episode_auroc = defaultdict(list)
            episode_precision = defaultdict(lambda: defaultdict(list))
            episode_recall = defaultdict(lambda: defaultdict(list))
            for bag_position in scored:
                bag, bag_mask = bags[bag_position], mask[bag_position]
                responsive = bag_mask.bool()
                if responsive.all() or not responsive.any():
                    continue
                base_rates.append(float(responsive.float().mean()))

                centred = bag - bag.mean(dim=0, keepdim=True)
                scores = {
                    "novelty": 1.0
                    - (F.normalize(bag.float(), dim=-1) @ anchors.float().T).max(dim=-1).values,
                    "outlier_distance": centred.float().norm(dim=-1),
                    "studentized": (
                        centred.float()
                        / centred.float().std(dim=0, keepdim=True).clamp_min(1e-6)
                    ).norm(dim=-1),
                }
                if memories is not None:
                    # Reproduce the rare-evidence path exactly: studentized bag
                    # view -> instance projection -> similarity to the class
                    # memories, which is the score its top-k pools over.
                    view = aggregator._bag_view(bag)[0].unsqueeze(0)
                    encoded = clf.instance_input_projection(clf.instance_input_norm(view))
                    similarity = torch.einsum(
                        "qnd,cmd->qcnm",
                        F.normalize(encoded.float(), dim=-1),
                        F.normalize(memories.float(), dim=-1),
                    )
                    scores["class_memory"] = similarity.amax(dim=(1, 3)).squeeze(0)
                if args.probe:
                    # Fisher LDA on the responsive labels. lda_probe fits and
                    # scores the same cells, so with 512 dims it overfits and is
                    # only a loose bound; lda_heldout fits on half the cells and
                    # scores the other half, which is the honest number.
                    f = bag.float()
                    variance = f.var(dim=0, unbiased=False).clamp_min(1e-6)
                    delta = f[responsive].mean(dim=0) - f[~responsive].mean(dim=0)
                    scores["lda_probe"] = f @ (delta / variance)

                    half = torch.zeros(f.shape[0], dtype=torch.bool, device=f.device)
                    half[torch.randperm(f.shape[0], device=f.device)[: f.shape[0] // 2]] = True
                    fit_pos = responsive & half
                    fit_neg = (~responsive) & half
                    if int(fit_pos.sum()) >= 2 and int(fit_neg.sum()) >= 2:
                        delta_fit = f[fit_pos].mean(dim=0) - f[fit_neg].mean(dim=0)
                        direction = delta_fit / f[half].var(dim=0, unbiased=False).clamp_min(1e-6)
                        held = ~half
                        if responsive[held].any() and (~responsive[held]).any():
                            value = auroc((f[held] @ direction).cpu(), responsive[held].long().cpu())
                            if value == value:
                                episode_auroc["lda_heldout"].append(value)
                target = responsive.long().cpu()
                for name, score in scores.items():
                    value = auroc(score.cpu(), target)
                    if value == value:
                        episode_auroc[name].append(value)
                    order = score.argsort(descending=True)
                    total_responsive = int(responsive.sum())
                    for fraction in FRACTIONS:
                        keep = max(1, int(round(fraction * bag.shape[0])))
                        hit = int(responsive[order[:keep]].sum())
                        episode_precision[name][fraction].append(hit / keep)
                        episode_recall[name][fraction].append(hit / total_responsive)

            for name, values in episode_auroc.items():
                if values:
                    ranking[name].append(sum(values) / len(values))
            for name in episode_precision:
                for fraction, values in episode_precision[name].items():
                    precision[name][fraction].append(sum(values) / len(values))
                for fraction, values in episode_recall[name].items():
                    recall[name][fraction].append(sum(values) / len(values))

    if not base_rates:
        print(f"No usable '{args.task}' episodes found.")
        return

    base = sum(base_rates) / len(base_rates)
    episodes = len(next(iter(ranking.values())))
    print(f"Task '{args.task}': {episodes} episodes, responsive cells = {base:.1%} of a bag\n")

    print("Can the score identify responsive cells at all?")
    print(f"  {'score':<20} {'AUROC':>7}  {'95% CI':<16}")
    print("  " + "-" * 46)
    for name, values in sorted(ranking.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        mean = sum(values) / len(values)
        low, high = episode_bootstrap(values, args.bootstrap)
        print(f"  {name:<20} {mean:>7.4f}  [{low:.3f}, {high:.3f}]")
    print("  0.5 would mean the ranking is unrelated to responsiveness.")

    print(f"\nPrecision at each kept fraction (random pick = {base:.3f})")
    header = "  " + f"{'score':<20}" + "".join(f"{f'{int(f*100)}%':>9}" for f in FRACTIONS)
    print(header)
    print("  " + "-" * (20 + 9 * len(FRACTIONS)))
    for name in precision:
        row = "".join(
            f"{sum(precision[name][f]) / len(precision[name][f]):>9.3f}" for f in FRACTIONS
        )
        print(f"  {name:<20}{row}")

    print("\nRecall at each kept fraction (share of responsive cells captured)")
    print(header)
    print("  " + "-" * (20 + 9 * len(FRACTIONS)))
    for name in recall:
        row = "".join(f"{sum(recall[name][f]) / len(recall[name][f]):>9.3f}" for f in FRACTIONS)
        print(f"  {name:<20}{row}")

    print(
        "\nPrecision at or below the base rate means the selection is no better than\n"
        "picking cells at random, and the cutoff is not the thing to tune."
    )


if __name__ == "__main__":
    main()
