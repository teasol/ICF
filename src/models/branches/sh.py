"""SH branch: per-dimension skewness and excess kurtosis of the projection.

Every Location-family branch (CV, BM, QA, DS) describes where a slide's tokens
sit. SH instead standardises the within-slide projection per dimension and reads
the marginal moment shapes — skewness and excess kurtosis — so it is location-
and scale-invariant by construction and cannot restate the mean (§218). SJ is
its multivariate sibling: SJ whitens first and reads the radius distribution,
SH reads each dimension's moments (§219: max |r| = 0.418 across the branch
screen; gate ②b pass on 44/50 folds).

The implementation was moved verbatim from the screening closure inside
scripts/test_pathobench.py (`sh_all`), which remains the caller for the other
screen-only SH variants (shs/shk/sh2/shr/shr2).
"""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge

_EPS = 1e-6


def sh_slide_features(
    bag: torch.Tensor,
    basis: torch.Tensor,
    dim: int,
    wide: int | None = None,
) -> torch.Tensor:
    """Per-dimension [skewness, excess kurtosis] of one slide's projection.

    Tokens are projected ONCE onto the first `wide` basis columns (default: the
    full `basis`), the moments are read per dimension, and only the first `dim`
    of each moment is kept — the same contract the screening path used.
    """
    # Same fp32 contract as SJ. The eval pipeline runs under bf16 autocast; the
    # standardisation divides by a std that bf16 would perturb, and the fourth
    # power then amplifies the error. Forcing float32 keeps the branch
    # reproducible and independent of the surrounding autocast state.
    with torch.autocast(device_type=bag.device.type, enabled=False):
        values = bag.to(basis.device, dtype=torch.float32)
        width = basis.shape[1] if wide is None else min(int(wide), basis.shape[1])
        width = max(width, int(dim))
        proj = values @ basis[:, :width].to(dtype=torch.float32)

        mu = proj.mean(dim=0, keepdim=True)
        sd = proj.std(dim=0, keepdim=True).clamp_min(_EPS)
        z = (proj - mu) / sd
        skew = z.pow(3).mean(dim=0)                    # [width]
        kurt = z.pow(4).mean(dim=0) - 3.0              # [width]

    return torch.cat([skew[:dim], kurt[:dim]])


def sh_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension moment descriptors through a class-balanced kernel ridge."""
    wide = min(getattr(config, "sh_wide", 256), basis.shape[1])
    dim = min(getattr(config, "sh_dim", 32), wide)

    def stack(bags):
        feats = torch.stack([sh_slide_features(b, basis, dim, wide) for b in bags])
        return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return solve_kernel_ridge(
        stack(context_bags), context_labels, stack(query_bags),
        kernel=getattr(config, "krr_kernel", "linear"),
        gamma=getattr(config, "krr_gamma", None),
        degree=getattr(config, "krr_degree", 2),
        coef0=getattr(config, "krr_coef0", 1.0),
        reg_lambda=getattr(config, "sh_lambda", 1.0),
        return_loo=return_loo,
    )
