"""Is CT's distance metric actually suffering from concentration? (docs SS149)

SS148 traced CT's weakness upstream of the readout and named three suspects:
farthest-point token selection, the 64-cell sample, and squared-Euclidean distance
in 1,536 dimensions. This tests the third one, and it tests the PREMISE before
testing the fix.

The premise: as dimension grows, pairwise distances concentrate -- their spread
shrinks relative to their mean -- so every cell sits at nearly the same distance
from every token, the softmax over -distance/T flattens toward uniform, and the
per-bag abundance stops varying between bags. That would produce exactly SS148-4's
symptom (per-token discriminative |t| with a median of 1.31).

Three things are measured per fold, at the raw 1,536 dims and at each PCA
dimension, all on the SAME cells and the SAME pipeline:

  contrast     (max - min) / mean of each cell's distances to the 16 tokens.
               This is the quantity concentration destroys. Near 0 means the
               softmax cannot discriminate.
  entropy      Shannon entropy of the abundance rows, in nats, against ln(16) =
               2.77 for a uniform assignment. Directly says how flat the soft
               assignment has become.
  |t| median   the SS148-4 statistic, so the diagnostic connects to the symptom.

⚠️ Deviations from `eval_seal_tasks.sh`, applied identically at every dimension:
bags are capped once at load (docs SS138-2). Between-dimension gaps are the result;
absolute AUROC is not comparable to SEAL macro.

Usage: python scripts/diagnose_ct_pca_distance.py [--heldout]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_full_basis import ALL_TASKS, auroc, index_h5, load_task  # noqa: E402
from src.models.ct_readout import (  # noqa: E402
    CTReadoutConfig,
    ct_abundance,
    discriminative_score,
    prepare_cells,
    readout_extreme,
)
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig  # noqa: E402

FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
TASKS = ["cptac_brca/TP53_mutation", "bc_therapy/er_status", "cptac_luad/STK11_mutation"]
HELDOUT = [
    "cptac_lscc/ARID1A_mutation", "cptac_lscc/Histologic_Grade",
    "cptac_lscc/KEAP1_mutation", "cptac_luad/KRAS_mutation",
    "cptac_pda/SMAD4_mutation", "ucla_lung/progression_regression",
    "cptac_ccrcc/PBRM1_mutation",
]
DIMS = [None, 4, 8, 16, 32, 64, 128, 256]
BASIS_DIM = 256


def concentration(cells, tokens):
    """Two measures of how distinguishable a cell's token distances are.

    ⚠️ `cells` must be the CELLS, not the tokens. Passing the tokens makes every
    row contain its own zero self-distance, which turns `contrast` into max/mean
    and fabricates a trend -- the bug that produced SS149's first contrast table.

    contrast  (max - min) / mean of one cell's distances to the 16 tokens, then
              averaged over cells. This is what the softmax over -distance/T has
              to work with; concentration drives it toward 0.
    rel_std   std / mean over ALL cell-token distances. The textbook statement of
              distance concentration is that this shrinks as dimension grows.
    """
    distance = (cells[:, None, :] - tokens[None]).square().mean(-1)
    spread = distance.max(dim=1).values - distance.min(dim=1).values
    contrast = float((spread / distance.mean(dim=1).clamp_min(1e-12)).mean())
    rel_std = float(distance.std(unbiased=False) / distance.mean().clamp_min(1e-12))
    return contrast, rel_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--seal", action="store_true", help="all 10 SEAL tasks")
    parser.add_argument("--folds", type=int, default=5, help="0 = every fold")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.tasks is None:
        args.tasks = HELDOUT if args.heldout else (ALL_TASKS if args.seal else TASKS)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    reference = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=BASIS_DIM))
    generator = torch.Generator().manual_seed(0)
    h5_index = index_h5(FEATURES)
    stats = {dim: {"contrast": [], "relstd": [], "entropy": [], "tmed": [], "auroc": []}
             for dim in DIMS}
    per_task = {dim: {} for dim in DIMS}

    for task in args.tasks:
        bags, labels, membership, folds = load_task(task, h5_index, device, 8192, generator)
        task_auroc = {dim: [] for dim in DIMS}
        for fold in (folds if args.folds == 0 else folds[: args.folds]):
            train, test = membership[fold]["train"], membership[fold]["test"]
            if not train or not test:
                continue
            y = torch.tensor([labels[s] for s in train], device=device)
            if len(set(y.tolist())) < 2:
                continue
            query_y = torch.tensor([labels[s] for s in test], device=device)
            context_bags = [bags[s] for s in train]
            query_bags = [bags[s] for s in test]
            basis = reference.within_slide_basis(context_bags)
            for dim in DIMS:
                config = CTReadoutConfig(pca_dim=dim)
                abundance = ct_abundance(
                    context_bags, query_bags, config, None if dim is None else basis
                )
                margins = readout_extreme(abundance, y, config)
                rows = abundance.context.clamp_min(1e-12)
                entropy = float(-(rows * rows.log()).sum(dim=-1).mean())
                # The SAME cells the tokens were chosen from, via the shared helper.
                cells, _ = prepare_cells(
                    context_bags, query_bags, config, None if dim is None else basis
                )
                contrast, rel_std = concentration(torch.cat(cells, 0), abundance.tokens)
                stats[dim]["contrast"].append(contrast)
                stats[dim]["relstd"].append(rel_std)
                stats[dim]["entropy"].append(entropy)
                stats[dim]["tmed"].append(
                    float(discriminative_score(abundance, y, config).abs().median())
                )
                value = auroc(query_y.cpu(), margins.query.float().cpu())
                if value is not None:
                    stats[dim]["auroc"].append(value)
                    task_auroc[dim].append(value)
        for dim in DIMS:
            values = task_auroc[dim]
            if values:
                per_task[dim][task] = statistics.mean(values)
        print(f"{task:34s} " + "  ".join(
            f"{('raw' if d is None else d)}={per_task[d].get(task, float('nan')):.4f}"
            for d in DIMS), flush=True)
        del bags
        torch.cuda.empty_cache()

    print(f"\n{len(args.tasks)} tasks. ln(16) = {math.log(16):.3f} nats "
          f"= a perfectly uniform (useless) assignment.\n")
    print(f"{'PCA dim':>10} {'contrast':>10} {'rel_std':>9} {'entropy':>9} "
          f"{'|t| median':>11} {'CT-only AUROC':>14}")
    for dim in DIMS:
        row = stats[dim]
        label = "1536 (raw)" if dim is None else str(dim)
        print(f"{label:>10} {statistics.mean(row['contrast']):>10.4f} "
              f"{statistics.mean(row['relstd']):>9.4f} "
              f"{statistics.mean(row['entropy']):>9.4f} "
              f"{statistics.mean(row['tmed']):>11.2f} "
              f"{statistics.mean(row['auroc']):>14.4f}")
    print("\n=== CT-only AUROC per task ===")
    print(f"{'task':34s}" + "".join(f"{('raw' if d is None else d):>9}" for d in DIMS))
    for task in per_task[None]:
        print(f"{task:34s}" + "".join(f"{per_task[d][task]:>9.4f}" for d in DIMS))
    print("-" * (34 + 9 * len(DIMS)))
    macro = {d: statistics.mean(per_task[d].values()) for d in DIMS}
    print(f"{'MACRO':34s}" + "".join(f"{macro[d]:>9.4f}" for d in DIMS))
    print(f"{'Δ vs raw':34s}" + "".join(f"{macro[d]-macro[None]:>+9.4f}" for d in DIMS))

    print("\ncontrast = (max-min)/mean of a CELL's 16 token distances (higher = more "
          "distinguishable).\nrel_std  = std/mean over all cell-token distances; "
          "concentration shrinks this as dimension grows.\nentropy near ln(16) means "
          "the softmax assigns every cell to every token equally.")


if __name__ == "__main__":
    main()
