"""What |t| do DD's candidate directions actually reach? (docs SS146)

The gate in `dd_adaptive_rank` needs a threshold, and the first attempt at 2.5 --
a value that looks strict on a t-table -- let ALL EIGHT candidates through on
every fold. That is the post-selection inflation the module's docstring warns
about, measured: the directions are eigenvectors of an operator built to
maximise class dispersion difference on exactly the bags the t is computed from,
so |t| is not on a null scale at all.

So before sweeping the threshold, look at the distribution. For real folds this
prints |t| per rank, which is what a usable threshold has to separate. It also
prints the eigenvalue magnitudes, to see whether |lambda| ordering and |t|
ordering even agree -- if they did, the gate would be redundant with rank_max.
"""

from __future__ import annotations

import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_full_basis import index_h5, load_task  # noqa: E402
from src.models.dd_adaptive_rank import (  # noqa: E402
    AdaptiveRankConfig,
    _welch_t,
    dispersion_directions,
)
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig  # noqa: E402

FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
TASKS = ["cptac_brca/TP53_mutation", "bc_therapy/er_status", "cptac_ccrcc/VHL_mutation"]
SKETCH = 256
RANKS = 8


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
    config = AdaptiveRankConfig()
    generator = torch.Generator().manual_seed(0)
    per_rank = {rank: [] for rank in range(RANKS)}
    agreements = []

    for task in TASKS:
        bags, labels, membership, folds = load_task(
            task, index_h5(FEATURES), device, 4096, generator
        )
        print(f"\n=== {task} ===")
        print(f"{'fold':>6} " + " ".join(f"{'|t| r'+str(r):>9}" for r in range(RANKS)))
        for fold in folds[:5]:
            train = membership[fold]["train"]
            if not train:
                continue
            context_bags = [bags[s] for s in train]
            y = torch.tensor([labels[s] for s in train], device=device)
            basis = model.within_slide_basis(context_bags)
            triangle = torch.triu_indices(SKETCH, SKETCH, device=device)
            covariance = torch.stack([
                ((bag.float() - bag.float().mean(0, keepdim=True)) @ basis).T
                @ ((bag.float() - bag.float().mean(0, keepdim=True)) @ basis)
                / bag.shape[0]
                for bag in context_bags
            ])
            del triangle
            directions, eigenvalues = dispersion_directions(covariance, y, config)
            statistics_by_rank = []
            for rank in range(RANKS):
                direction = directions[:, rank]
                scalar = torch.einsum(
                    "d,bdk,k->b", direction, covariance, direction
                ).clamp_min(config.eps).log()
                value = float(_welch_t(scalar[y == 0], scalar[y == 1], config.eps).abs())
                statistics_by_rank.append(value)
                per_rank[rank].append(value)
            print(f"{fold:>6} " + " ".join(f"{v:>9.2f}" for v in statistics_by_rank))
            # Does |lambda| order agree with |t| order? If yes the gate adds nothing.
            best_by_t = int(max(range(RANKS), key=lambda r: statistics_by_rank[r]))
            agreements.append(best_by_t == 0)
        del bags
        torch.cuda.empty_cache()

    print("\n=== |t| by rank, pooled over folds ===")
    print(f"{'rank':>6} {'median':>9} {'min':>9} {'max':>9} {'>2.5':>7} {'>10':>6} {'>20':>6}")
    for rank in range(RANKS):
        values = per_rank[rank]
        print(f"{rank:>6} {statistics.median(values):>9.2f} {min(values):>9.2f} "
              f"{max(values):>9.2f} {sum(v > 2.5 for v in values):>5}/{len(values)}"
              f" {sum(v > 10 for v in values):>4}/{len(values)}"
              f" {sum(v > 20 for v in values):>4}/{len(values)}")
    print(f"\nrank 0 is also the |t|-argmax on {sum(agreements)}/{len(agreements)} folds "
          "— if this were 100%, |lambda| ordering already encodes the gate.")


if __name__ == "__main__":
    main()
