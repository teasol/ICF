"""Does the correlation descriptor change the CV margin's SCALE? (docs SS162)

SS162 measured CV-only slightly UP (+0.0017, 11/17) while the full model went DOWN
(-0.0024, 8/17). AUROC is scale-free, so CV-only cannot see a magnitude change --
but the fixed head applies a constant 1.442 to `cv1 - cv0`, so it very much can.
SS148 calibrated CT's alternative margins to the reference's context RMS for exactly
this reason; the correlation arm did not get the same treatment.

So before concluding anything about correlation-vs-covariance, measure whether the
two CV margins even live on the same scale. Reported per fold, context bags only:

    rms      RMS of (cv1 - cv0) over CONTEXT bags, under each descriptor
    ratio    corr / cov -- 1.0 means the full-model comparison was already fair

⚠️ Diagnostic loader (bags capped once at load, SS138-2). The ratio is what matters
here, and it is computed on identical bags either way.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_full_basis import ALL_TASKS, index_h5, load_task  # noqa: E402
from src.models.training_free import (  # noqa: E402
    TrainingFreeClassifier,
    TrainingFreeConfig,
    _standardise,
)

FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
SKETCH = 256


def cv_margin(descriptor_context, descriptor_query, labels, ridge_lambda=1.0, scale=2.0):
    """The CV branch's margin, mirroring `_cv_logits` with a single block."""
    context, query = _standardise(descriptor_context.float(), descriptor_query.float())
    targets = torch.nn.functional.one_hot(labels.long(), 2).float()
    counts = torch.bincount(labels.long(), minlength=2)
    weight = counts.float().reciprocal()[labels.long()]
    total = weight.sum().clamp_min(1e-12)
    feature_mean = (weight[:, None] * context).sum(0, keepdim=True) / total
    target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
    root = weight.sqrt()[:, None]
    design = (context - feature_mean) * root
    centred = (targets - target_mean) * root
    gram = design @ design.T
    gram = gram + ridge_lambda * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    dual = torch.linalg.solve(gram, centred)
    coefficients = design.T @ dual
    intercept = target_mean - feature_mean @ coefficients
    logits = (query @ coefficients + intercept) * scale
    return logits[:, 1] - logits[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*", default=ALL_TASKS[:5])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
    generator = torch.Generator().manual_seed(0)
    h5_index = index_h5(FEATURES)
    ratios, cov_rms, corr_rms = [], [], []

    print(f"{'task':34s}{'cov rms':>10}{'corr rms':>11}{'ratio':>9}")
    for task in args.tasks:
        bags, labels, membership, folds = load_task(task, h5_index, device, 8192, generator)
        task_ratio = []
        for fold in folds[: args.folds]:
            train = membership[fold]["train"]
            if not train:
                continue
            y = torch.tensor([labels[s] for s in train], device=device)
            if len(set(y.tolist())) < 2:
                continue
            context_bags = [bags[s] for s in train]
            basis = model.within_slide_basis(context_bags)
            triangle = torch.triu_indices(SKETCH, SKETCH, device=device)
            descriptors = torch.stack(
                [model._descriptor(bag, basis, triangle) for bag in context_bags]
            )
            covariance = descriptors[:, : triangle.shape[1]]
            off = triangle[0] != triangle[1]
            diagonal_slot = torch.where(~off)[0]
            variances = covariance.index_select(-1, diagonal_slot).clamp_min(1e-12)
            denominator = (
                variances.index_select(-1, triangle[0]) * variances.index_select(-1, triangle[1])
            ).sqrt().clamp_min(1e-12)
            correlation = covariance / denominator
            # Score the CONTEXT bags as their own queries, which is what the head's
            # constant is implicitly calibrated against.
            a = cv_margin(covariance[:, off], covariance[:, off], y)
            b = cv_margin(correlation[:, off], correlation[:, off], y)
            first, second = float(a.square().mean().sqrt()), float(b.square().mean().sqrt())
            cov_rms.append(first)
            corr_rms.append(second)
            task_ratio.append(second / max(first, 1e-12))
        if task_ratio:
            ratios.extend(task_ratio)
            print(f"{task:34s}{statistics.mean(cov_rms[-len(task_ratio):]):>10.3f}"
                  f"{statistics.mean(corr_rms[-len(task_ratio):]):>11.3f}"
                  f"{statistics.mean(task_ratio):>9.3f}", flush=True)
        del bags
        torch.cuda.empty_cache()

    print("-" * 64)
    print(f"{'MEAN':34s}{statistics.mean(cov_rms):>10.3f}"
          f"{statistics.mean(corr_rms):>11.3f}{statistics.mean(ratios):>9.3f}")
    print(f"\nratio 1.0에서 멀수록 SS162의 full-model 비교가 CV의 크기를 비교한 것이 된다.")


if __name__ == "__main__":
    main()
