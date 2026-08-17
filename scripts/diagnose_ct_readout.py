"""CT-only comparison of the three readouts, plus what the abundance contains
(docs SS148).

The full-model sweep (`run_ct_readout_sweep.sh`) answers "does the final macro
move". It cannot answer WHY, because CT enters the margin at weight 0.286
alongside CV and DD -- a readout could be much better and still be invisible
because CV already carries the same information. So this scores the CT margin
ALONE, and prints the diagnostics that separate the three possible bottlenecks:

  * `separation`  per-token |(mean_0 - mean_1) / SE| over the 16 dims. If only one
    or two tokens carry anything, `extreme` is already reading everything there is
    and the readout is NOT the bottleneck.
  * `ridge concentration` share of sum|w| in the top 1 and 2 coefficients, plus a
    participation ratio (effective number of tokens used). Spread coefficients mean
    the extra dims are doing real work.
  * `overlap` whether the two tokens `extreme` picks are the ridge's two largest
    |coefficient| tokens. If they always coincide, prototype/ridge cannot beat
    extreme by much and any difference is weighting, not token choice.

⚠️ Deviations from `eval_seal_tasks.sh`, applied IDENTICALLY to every arm: bags
are capped at `max_cells` once at load time rather than re-subsampled per query
(docs SS138-2), and the whole task is held in memory. Absolute numbers are
therefore not comparable to SEAL macro; the between-readout gaps are.

Usage:
    python scripts/diagnose_ct_readout.py [--tasks ...] [--heldout]
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_full_basis import ALL_TASKS, auroc, index_h5, load_task  # noqa: E402
from src.models.ct_readout import (  # noqa: E402
    CTReadoutConfig,
    calibrate,
    ct_abundance,
    discriminative_score,
    readout_extreme,
    readout_prototype,
    readout_ridge,
    ridge_coefficients,
)

FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
HELDOUT = [
    "cptac_lscc/ARID1A_mutation", "cptac_lscc/Histologic_Grade",
    "cptac_lscc/KEAP1_mutation", "cptac_luad/KRAS_mutation",
    "cptac_pda/SMAD4_mutation", "ucla_lung/progression_regression",
    "cptac_ccrcc/PBRM1_mutation",
]
READOUTS = {"extreme": readout_extreme, "prototype": readout_prototype,
            "ridge": readout_ridge}


def balanced_bce(labels: torch.Tensor, margins: torch.Tensor) -> float:
    """Class-weighted BCE on sigmoid(margin).

    ⚠️ Unlike AUROC this is scale-dependent, so it is only meaningful on the
    CALIBRATED margins -- otherwise it scores how large a readout's output is.
    """
    labels = labels.float()
    probability = torch.sigmoid(margins.double()).clamp(1e-7, 1 - 1e-7)
    loss = -(labels * probability.log() + (1 - labels) * (1 - probability).log())
    parts = [float(loss[labels == c].mean()) for c in (0, 1) if int((labels == c).sum())]
    return sum(parts) / len(parts)


def participation_ratio(weights: torch.Tensor) -> float:
    """Effective number of tokens carrying the ridge margin: (sum w^2)^2/sum w^4.

    1.0 means a single token does everything; 16.0 means all are used equally.
    """
    squared = weights.double().square()
    return float(squared.sum().square() / squared.square().sum().clamp_min(1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tasks = args.tasks or (HELDOUT if args.heldout else ALL_TASKS)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = CTReadoutConfig()
    generator = torch.Generator().manual_seed(0)
    h5_index = index_h5(FEATURES)
    results = {name: {} for name in READOUTS}
    losses = {name: {} for name in READOUTS}
    diagnostics = []

    for task in tasks:
        bags, labels, membership, folds = load_task(task, h5_index, device, 8192, generator)
        per_fold = {name: [] for name in READOUTS}
        per_fold_loss = {name: [] for name in READOUTS}
        for fold in folds:
            train, test = membership[fold]["train"], membership[fold]["test"]
            if not train or not test:
                continue
            y = torch.tensor([labels[s] for s in train], device=device)
            if len(set(y.tolist())) < 2:
                continue
            query_y = torch.tensor([labels[s] for s in test], device=device)
            context_bags = [bags[s] for s in train]
            query_bags = [bags[s] for s in test]

            abundance = ct_abundance(context_bags, query_bags, config)
            reference = readout_extreme(abundance, y, config)
            for name, readout in READOUTS.items():
                margins = readout(abundance, y, config)
                if name != "extreme":
                    margins = calibrate(margins, reference, config)
                value = auroc(query_y.cpu(), margins.query.float().cpu())
                if value is not None:
                    per_fold[name].append(value)
                    per_fold_loss[name].append(
                        balanced_bce(query_y.cpu(), margins.query.float().cpu())
                    )

            score = discriminative_score(abundance, y, config).abs()
            beta, _, _, _ = ridge_coefficients(abundance, y, config)
            weights = (beta[:, 1] - beta[:, 0]).abs()
            order = weights.argsort(descending=True)
            extreme_tokens = {int(score.argmax()), int((-score).argmax())}
            raw_score = discriminative_score(abundance, y, config)
            extreme_tokens = {int(raw_score.argmax()), int(raw_score.argmin())}
            diagnostics.append({
                "task": task,
                "sep_max": float(score.max()),
                "sep_median": float(score.median()),
                "sep_over_2": int((score > 2.0).sum()),
                "top1": float(weights[order[0]] / weights.sum().clamp_min(1e-12)),
                "top2": float(weights[order[:2]].sum() / weights.sum().clamp_min(1e-12)),
                "effective": participation_ratio(weights),
                "overlap": len(extreme_tokens & {int(order[0]), int(order[1])}),
            })
        for name in READOUTS:
            if per_fold[name]:
                results[name][task] = statistics.mean(per_fold[name])
                losses[name][task] = statistics.mean(per_fold_loss[name])
        print(f"{task:34s} " + "  ".join(
            f"{n}={results[n].get(task, float('nan')):.4f}" for n in READOUTS
        ), flush=True)
        del bags
        torch.cuda.empty_cache()

    print("\n=== CT-ONLY AUROC (CT margin alone, no CV, no DD) ===")
    print(f"{'task':34s}" + "".join(f"{n:>12}" for n in READOUTS))
    for task in results["extreme"]:
        print(f"{task:34s}" + "".join(f"{results[n][task]:>12.4f}" for n in READOUTS))
    print("-" * (34 + 12 * 3))
    macro = {n: statistics.mean(results[n].values()) for n in READOUTS}
    print(f"{'MACRO':34s}" + "".join(f"{macro[n]:>12.4f}" for n in READOUTS))
    print(f"{'Δ vs extreme':34s}" + "".join(
        f"{macro[n] - macro['extreme']:>+12.4f}" for n in READOUTS))
    for name in ("prototype", "ridge"):
        wins = sum(results[name][t] > results["extreme"][t] for t in results["extreme"])
        print(f"  {name} beats extreme on {wins}/{len(results['extreme'])} tasks")

    print("\n=== balanced BCE (on CALIBRATED margins; lower is better) ===")
    loss_macro = {n: statistics.mean(losses[n].values()) for n in READOUTS}
    print("  " + "  ".join(f"{n}={loss_macro[n]:.4f}" for n in READOUTS))

    print("\n=== abundance diagnostics, mean over all folds ===")
    def average(key):
        return statistics.mean(d[key] for d in diagnostics)
    print(f"  per-token |(m0-m1)/SE|: max {average('sep_max'):.2f}, "
          f"median {average('sep_median'):.2f}, tokens with |t|>2: "
          f"{average('sep_over_2'):.1f} / {config.num_tokens}")
    print(f"  ridge |w| concentration: top-1 {average('top1'):.3f}, "
          f"top-2 {average('top2'):.3f}, effective tokens "
          f"{average('effective'):.2f} / {config.num_tokens}")
    print(f"  extreme's 2 tokens among ridge's top 2: {average('overlap'):.2f} / 2")


if __name__ == "__main__":
    main()
