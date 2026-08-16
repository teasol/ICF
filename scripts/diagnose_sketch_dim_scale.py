"""How much does the ridge penalty drift when K changes? (docs SS142)

Standardisation scales each descriptor block to unit RMS, so a bag's squared
norm tracks the descriptor LENGTH, and the dual Gram grows with it while a fixed
`ridge_lambda = 1.0` does not. Comparing K at fixed lambda therefore moves two
knobs at once -- number of principal directions AND effective regularisation --
which SS127-2 forbids.

This measures the drift instead of assuming it: for one real fold it builds the
context descriptors at each K and reports the mean diagonal of the dual Gram
`design @ design.T`. The ratio against K=128 is the lambda that keeps the
penalty-to-signal ratio fixed, i.e. the control arm's setting.

The naive prediction is `(K(K+1)/2 + 1536) / (8256 + 1536)`, but the blocks are
standardised separately and the covariance triangle's entries are not
independent, so the measured number is what the control arm should use.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_full_basis import index_h5, load_task  # noqa: E402
from src.models.training_free import (  # noqa: E402
    TrainingFreeClassifier,
    TrainingFreeConfig,
    _standardise_blocks,
)

FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
TASK = "cptac_brca/TP53_mutation"
DIMS = (64, 128, 192, 256, 384)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(0)
    bags, labels, membership, folds = load_task(
        TASK, index_h5(FEATURES), device, 4096, generator
    )
    fold = folds[0]
    train = membership[fold]["train"]
    context_bags = [bags[s] for s in train]
    y = torch.tensor([labels[s] for s in train], device=device)
    print(f"{TASK}  {fold}: {len(context_bags)} context bags, "
          f"{sum(b.shape[0] for b in context_bags)} cells\n")

    print(f"{'K':>5} {'desc dim':>10} {'gram diag':>12} {'ratio vs 128':>13} "
          f"{'naive dim ratio':>16}")
    reference = None
    for sketch_dim in DIMS:
        model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=sketch_dim))
        basis = model.within_slide_basis(context_bags)
        triangle = torch.triu_indices(sketch_dim, sketch_dim, device=device)
        descriptors = torch.stack(
            [model._descriptor(bag, basis, triangle) for bag in context_bags]
        )
        split = [sketch_dim * (sketch_dim + 1) // 2, basis.shape[0]]
        standardised, _ = _standardise_blocks(
            descriptors.float(), descriptors.float(), split
        )
        counts = torch.bincount(y.long(), minlength=2)
        weight = counts.float().reciprocal()[y.long()]
        total = weight.sum().clamp_min(1e-12)
        centre = (weight[:, None] * standardised).sum(0, keepdim=True) / total
        design = (standardised - centre) * weight.sqrt()[:, None]
        diagonal = float((design @ design.T).diagonal().mean())
        if sketch_dim == 128:
            reference = diagonal
        print(f"{sketch_dim:>5} {sum(split):>10} {diagonal:>12.4g}", end="")
        if reference is not None and sketch_dim != 128:
            print(f" {diagonal / reference:>13.3f}", end="")
        elif sketch_dim == 128:
            print(f" {1.0:>13.3f}", end="")
        else:
            print(f" {'-':>13}", end="")
        print(f" {sum(split) / (8256 + 1536):>16.3f}")

    print(
        "\nThe 'ratio vs 128' column is the lambda for the dimension-matched "
        "control arm:\n  ICF_RIDGE_LAMBDA=<ratio> isolates 'more directions' "
        "from 'weaker ridge'."
    )


if __name__ == "__main__":
    main()
