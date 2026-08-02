"""Normalization-ceiling probe: does the fixed cell normalization limit the ridge ceiling?

Hypothesis under test (user, 2026-08-03): the pattern we should remove
(donor/background structure) is a CROSS-FEATURE pattern, but the fixed
per-feature centering in `_bag_view` only removes each dimension's mean.
So "learning the pattern to remove" (rather than assuming it is per-feature)
could raise the information ceiling -- currently ~0.70 (F-series closed-form
ridge on v24 slot stats).

No model is loaded. For each val episode we transform the raw cells under
several normalizations, compute per-bag sufficient statistics, and fit a
class-balanced closed-form ridge per episode (context -> query), measuring the
AUROC ceiling. All normalizations are compared at the SAME feature
construction, so the *ordering* is the signal.

Normalizations:
  raw          cells as generated (the generator already L2-normalizes them)
  centered     per-feature centering only (subtract each dim's mean over cells)
  current      per-feature centering + per-cell L2 norm  == `_bag_view` input
  zscore       per-feature centering + per-feature std scaling
  whiten_ctx   per-feature centering + whitening transform fit on CONTEXT cells
               only (learned from the episode's labeled context = "learn the
               pattern to remove", applied to all cells; no query leakage)

Features (per bag, under each normalization):
  mean          per-feature mean [512]
  mean_var      mean + per-feature variance [1024]
  mean_var_cov  mean + variance + 64-d covariance sketch (32-d random
                projection: diag + 32 off-diag) [1088]

Reference (F-series): closed-form ridge on v24 slot stats = 0.700 overall;
simple global stats (bag_global K=1) = 0.630. This probe uses simpler global
stats, so absolute values are lower; the normalization ordering is the point.

Usage:
    python scripts/diagnose_normalization_ceiling.py --episodes 1000
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_oracle_slot_alignment import query_index  # noqa: E402
from src.utils.metrics import auroc, bootstrap_auroc_interval  # noqa: E402
from src.utils.utils import build_datamodule, merge_train_config  # noqa: E402

PROJECTION_DIM = 32
RIDGE_LAMBDA = 1.0
RANDOM_PROJECTION_SEED = 1234


def covariance_projection(device: torch.device, dim: int = 512) -> torch.Tensor:
    generator = torch.Generator().manual_seed(RANDOM_PROJECTION_SEED)
    projection = torch.randn(dim, PROJECTION_DIM, generator=generator)
    return (projection / projection.norm(dim=0, keepdim=True)).to(device)


def whiten_from_context(context_cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a whitening transform on context cells; returns (mean, W) with
    t = (x - mean) @ W  decorrelating the episode's cell distribution."""
    mean = context_cells.float().mean(dim=0)
    centered = context_cells.float() - mean
    cov = centered.T @ centered / max(context_cells.shape[0], 1)
    cov = cov + 1e-3 * cov.diag().mean().clamp_min(1e-12) * torch.eye(
        cov.shape[0], device=cov.device, dtype=cov.dtype
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = eigenvalues.clamp_min(1e-6)
    scale = torch.diag(eigenvalues.rsqrt())
    return mean, (eigenvectors @ scale @ eigenvectors.T)


def bag_features(cells: torch.Tensor, projection: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-bag sufficient statistics, batched over the bag axis.

    cells: [bags, cells, dim]. Returns per-bag mean/variance/covariance-sketch
    descriptors shaped [bags, feature_dim].
    """
    mean = cells.float().mean(dim=1)                      # [B, 512]
    variance = cells.float().var(dim=1, unbiased=False)   # [B, 512]
    centered = cells.float() - mean.unsqueeze(1)
    projected = centered @ projection                     # [B, cells, 32]
    sketch_cov = torch.einsum(
        "bni,bnj->bij", projected, projected
    ) / max(projected.shape[1], 1)
    triangle = torch.triu_indices(PROJECTION_DIM, PROJECTION_DIM, offset=1)
    sketch = torch.cat(
        (
            sketch_cov.diagonal(dim1=-2, dim2=-1),
            sketch_cov[..., triangle[0][:32], triangle[1][:32]],
        ),
        dim=-1,
    )
    return {
        "mean": mean,
        "mean_var": torch.cat((mean, variance), dim=-1),
        "mean_var_cov": torch.cat((mean, variance, sketch), dim=-1),
    }


def ridge_logits(
    design: torch.Tensor,
    labels: torch.Tensor,
    query_design: torch.Tensor,
    lam: float = RIDGE_LAMBDA,
) -> torch.Tensor:
    """Class-balanced closed-form ridge (0 learned params), per-episode."""
    design = design.float()
    query_design = query_design.float()
    num_classes = 2
    center = design.mean(dim=0, keepdim=True)
    context = design - center
    query = query_design - center
    rms = context.square().mean().sqrt().clamp_min(1e-6)
    context = context / rms
    query = query / rms
    class_counts = torch.bincount(labels.long(), minlength=num_classes).float()
    targets = torch.nn.functional.one_hot(labels.long(), num_classes).float()
    sample_weight = class_counts.reciprocal()[labels.long()]
    total_weight = sample_weight.sum().clamp_min(1e-12)
    feature_mean = (sample_weight.unsqueeze(-1) * context).sum(dim=0) / total_weight
    target_mean = (sample_weight.unsqueeze(-1) * targets).sum(dim=0) / total_weight
    centered_context = context - feature_mean
    centered_targets = targets - target_mean
    root_weight = sample_weight.sqrt().unsqueeze(-1)
    weighted_design = centered_context * root_weight
    weighted_targets = centered_targets * root_weight
    gram = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_targets
    system = gram + lam * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(system, rhs)
    intercept = target_mean - feature_mean @ coefficients
    return query @ coefficients + intercept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/train_v24_medium_bag_proj_residual.yaml"),
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("logs/normalization_ceiling_20260803.csv"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    config["data"].setdefault("val_dataset_kwargs", {})["episodes_per_epoch"] = args.episodes
    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    val_dataset = datamodule.val_dataset

    device = torch.device(args.device)
    projection = covariance_projection(device)

    NORMALIZATIONS = ("raw", "centered", "current", "zscore", "whiten_ctx")
    FEATURES = ("mean", "mean_var", "mean_var_cov")

    # scores[(norm, feature, task)] -> list of per-episode query scores
    collected: dict[tuple[str, str, str], list[torch.Tensor]] = defaultdict(list)
    targets_all: dict[tuple[str, str, str], list[torch.Tensor]] = defaultdict(list)
    groups_all: dict[tuple[str, str, str], list[torch.Tensor]] = defaultdict(list)

    evaluated = 0
    with torch.no_grad():
        for dataset_index in range(args.episodes):
            episode = val_dataset.diagnostic_episode(dataset_index)
            x = episode.x.to(device)
            y = episode.y.to(device).long()
            query = query_index(y)
            context = torch.ones(y.numel(), dtype=torch.bool, device=device)
            context[query] = False
            task = episode.response_task if episode.response_task in (
                "composition", "state", "covariance", "interaction", "combined"
            ) else "unknown"

            # Build each normalization's transformed cells.
            bag_mean = x.float().mean(dim=1, keepdim=True)
            transformed: dict[str, torch.Tensor] = {}
            raw = x.float()
            centered = raw - bag_mean
            current = torch.nn.functional.normalize(centered, dim=-1)
            zscore = centered / centered.float().std(dim=1, keepdim=True).clamp_min(1e-6)
            transformed["raw"] = raw
            transformed["centered"] = centered
            transformed["current"] = current
            transformed["zscore"] = zscore
            w_mean, W = whiten_from_context(x[context].reshape(-1, x.shape[-1]))
            transformed["whiten_ctx"] = (raw - w_mean.unsqueeze(0)) @ W

            for norm, cells in transformed.items():
                features = bag_features(cells, projection)
                for feature_name, descriptor in features.items():
                    key = (norm, feature_name, task)
                    logits = ridge_logits(
                        descriptor[context], y[context], descriptor[query]
                    )
                    score = (logits[:, 1] - logits[:, 0]).cpu()
                    collected[key].append(score)
                    targets_all[key].append(y[query].cpu())
                    groups_all[key].append(
                        torch.full((query.numel(),), evaluated, dtype=torch.long)
                    )
            evaluated += 1
            if (dataset_index + 1) % 200 == 0:
                print(f"  ... {dataset_index + 1}/{args.episodes}", flush=True)

    if evaluated == 0:
        raise RuntimeError("No episodes processed.")

    rows: list[dict] = []
    for norm in NORMALIZATIONS:
        for feature_name in FEATURES:
            # Aggregate over tasks for the overall row.
            task_names = ("composition", "state", "covariance", "interaction", "combined")
            all_scores = torch.cat(
                [s for t in task_names for s in collected[(norm, feature_name, t)]]
            )
            all_targets = torch.cat(
                [s for t in task_names for s in targets_all[(norm, feature_name, t)]]
            )
            all_groups = torch.cat(
                [s for t in task_names for s in groups_all[(norm, feature_name, t)]]
            )
            point = auroc(all_scores, all_targets)
            low, high = bootstrap_auroc_interval(
                all_scores, all_targets, groups=all_groups,
                samples=args.bootstrap, seed=args.seed,
            )
            rows.append({
                "normalization": norm, "features": feature_name, "task": "ALL",
                "queries": all_targets.numel(), "auroc": point,
                "auroc_ci_low": low, "auroc_ci_high": high,
            })
            for task in ("composition", "state", "covariance", "interaction", "combined"):
                key = (norm, feature_name, task)
                scores = torch.cat(collected[key])
                targets = torch.cat(targets_all[key])
                groups = torch.cat(groups_all[key])
                rows.append({
                    "normalization": norm, "features": feature_name, "task": task,
                    "queries": targets.numel(), "auroc": auroc(scores, targets),
                    "auroc_ci_low": float("nan"), "auroc_ci_high": float("nan"),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nNormalization ceiling probe — {evaluated} episodes, λ={RIDGE_LAMBDA}, "
          f"ridge is closed-form (0 learned params)")
    print("Reference: F-series ridge on v24 slot stats = 0.700; bag_global = 0.630.\n")
    print(f"{'normalization':<12} {'features':<12} {'ALL AUROC':>9} {'95% CI':>16}")
    print("-" * 52)
    for row in rows:
        if row["task"] != "ALL":
            continue
        interval = f"[{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}]"
        print(f"{row['normalization']:<12} {row['features']:<12} "
              f"{row['auroc']:>9.4f} {interval:>16}")

    print("\nPer-task (mean_var features):")
    print(f"{'normalization':<12} {'composition':>11} {'state':>7} "
          f"{'covariance':>11} {'interaction':>12} {'combined':>9}")
    for norm in NORMALIZATIONS:
        values = []
        for task in ("composition", "state", "covariance", "interaction", "combined"):
            key = (norm, "mean_var", task)
            scores = torch.cat(collected[key])
            targets = torch.cat(targets_all[key])
            values.append(auroc(scores, targets))
        print(f"{norm:<12} " + "".join(f"{v:>11.4f}" for v in values))
    print(f"\nsaved={args.output}")


if __name__ == "__main__":
    main()
