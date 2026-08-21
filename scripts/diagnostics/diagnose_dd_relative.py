"""Are there queries far from BOTH DD prototypes? (docs SS155)

The relative margin (D0 - D1) / (D0 + D1 + eps) only differs from the plain
difference when D0 + D1 varies a lot between queries -- if every query sits at a
similar total distance, dividing by it is a constant rescale and changes no
ranking. So measure the premise before running the arm.

Reported per fold, then averaged:
  total spread   p90/p10 of (D0 + D1) across the fold's queries. 1.0 means the
                 normaliser is constant and the transform is a no-op for ranking.
  far & flat     fraction of queries in the top tertile of (D0 + D1) whose
                 |D0 - D1| is nonetheless in the bottom half -- exactly the case
                 the proposal is meant to suppress.
  rank agreement Spearman-style agreement between ranking by difference and by
                 ratio. Below 1.0 means the two orderings genuinely differ.

⚠️ Uses the diagnostic loader (bags capped once at load, SS138-2), so absolute
AUROC is not comparable to SEAL macro. The comparison here is internal.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_full_basis import ALL_TASKS, auroc, index_h5, load_task  # noqa: E402
from src.models.dd_adaptive_rank import (  # noqa: E402
    AdaptiveRankConfig,
    adaptive_dd_distance_features,
    relative_margin,
)
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig  # noqa: E402

FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
HELDOUT = [
    "cptac_lscc/ARID1A_mutation", "cptac_lscc/Histologic_Grade",
    "cptac_lscc/KEAP1_mutation", "cptac_luad/KRAS_mutation",
    "cptac_pda/SMAD4_mutation", "ucla_lung/progression_regression",
    "cptac_ccrcc/PBRM1_mutation",
]
SKETCH = 256


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 3:
        return 1.0
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra @ rb) / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tasks = HELDOUT if args.heldout else ALL_TASKS
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    reference = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
    config = AdaptiveRankConfig()
    generator = torch.Generator().manual_seed(0)
    h5_index = index_h5(FEATURES)

    print(f"{'task':34s}{'total p90/p10':>14}{'far&flat':>10}{'rank agree':>12}"
          f"{'AUROC diff':>12}{'AUROC ratio':>13}")
    rows = []
    for task in tasks:
        bags, labels, membership, folds = load_task(task, h5_index, device, 8192, generator)
        spreads, farflat, agree, a_diff, a_ratio = [], [], [], [], []
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
            basis = reference.within_slide_basis(context_bags)
            triangle = torch.triu_indices(SKETCH, SKETCH, device=device)
            cov = lambda bs: torch.stack([
                ((b.float() - b.float().mean(0, keepdim=True)) @ basis).T
                @ ((b.float() - b.float().mean(0, keepdim=True)) @ basis) / b.shape[0]
                for b in bs
            ])
            del triangle
            distances, _, _ = adaptive_dd_distance_features(
                cov(context_bags), y, cov(query_bags), config
            )
            total = distances[:, 0] + distances[:, 1]
            difference = distances[:, 0] - distances[:, 1]
            ratio = relative_margin(distances, config.eps)
            if total.numel() >= 5:
                spreads.append(float(
                    total.quantile(0.9) / total.quantile(0.1).clamp_min(1e-12)))
                high = total >= total.quantile(2 / 3)
                flat = difference.abs() <= difference.abs().median()
                farflat.append(float((high & flat).float().mean()))
            agree.append(spearman(difference, ratio))
            for store, values in ((a_diff, difference), (a_ratio, ratio)):
                value = auroc(query_y.cpu(), values.float().cpu())
                if value is not None:
                    store.append(value)
        rows.append((task, statistics.mean(spreads), statistics.mean(farflat),
                     statistics.mean(agree), statistics.mean(a_diff),
                     statistics.mean(a_ratio)))
        print(f"{rows[-1][0]:34s}{rows[-1][1]:>14.2f}{rows[-1][2]:>10.3f}"
              f"{rows[-1][3]:>12.4f}{rows[-1][4]:>12.4f}{rows[-1][5]:>13.4f}", flush=True)
        del bags
        torch.cuda.empty_cache()

    print("-" * 95)
    print(f"{'MEAN':34s}" + "".join(
        f"{statistics.mean(r[i] for r in rows):>{w}.{p}f}"
        for i, (w, p) in ((1, (14, 2)), (2, (10, 3)), (3, (12, 4)),
                          (4, (12, 4)), (5, (13, 4)))))
    wins = sum(1 for r in rows if r[5] > r[4])
    print(f"\nDD-only AUROC: ratio beats difference on {wins}/{len(rows)} tasks")


if __name__ == "__main__":
    main()
