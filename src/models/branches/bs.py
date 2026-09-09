"""BS branch: log total variance of the projected token cloud.

Every Location-family branch (CV, BM, QA, DS) describes where a slide's tokens
sit; BD normalises the eigenvalue spectrum (p = eig / eig.sum()), discarding
total variance. BS is exactly that discarded quantity: the log of the trace of
the projected scatter, i.e. pure scale, so it cannot restate BD's (scale-free)
entropy nor any Location-family branch's mean (§218).

The implementation was moved verbatim from the screening closure inside
scripts/test_pathobench.py (`bs_feat`), which remains the caller: both the
`bag_stats_cache` (scatter) fast path and the plain from-bag path it used are
preserved here so the two callers cannot drift apart.
"""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge

_EPS = 1e-6


def bs_slide_features(
    bag: torch.Tensor,
    basis: torch.Tensor,
    dim: int,
    bag_stats: tuple | None = None,
) -> torch.Tensor:
    """log(clamp_min(tr(B^T S B) / n, eps)) of one slide's projection.

    `bag_stats`, when given, is the `(n, mean, scatter)` cache tuple
    `scripts/test_pathobench.py`'s `BagStatsCache` keeps per bag id: `scatter`
    is the centred sum-of-outer-products, so `tr(B^T scatter B) / n` equals the
    same quantity the from-bag path computes directly. Both branches must stay
    numerically identical to the screening closure they were moved from.
    """
    device = basis.device
    b_basis = basis[:, :dim].to(dtype=torch.float32)
    if bag_stats is not None:
        n_i, _, scatter = bag_stats
        tr = ((scatter.to(device).float() @ b_basis) * b_basis).sum() / float(n_i)
    else:
        v = bag.to(device).float()
        pr = (v - v.mean(dim=0, keepdim=True)) @ b_basis
        tr = pr.square().sum() / float(v.shape[0])
    return tr.clamp_min(_EPS).log().reshape(1)


def bs_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Log total projected variance through a class-balanced kernel ridge."""
    dim = min(getattr(config, "bs_dim", 256), basis.shape[1])

    def stack(bags):
        feats = torch.stack([bs_slide_features(b, basis, dim) for b in bags])
        return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return solve_kernel_ridge(
        stack(context_bags), context_labels, stack(query_bags),
        kernel=getattr(config, "krr_kernel", "linear"),
        gamma=getattr(config, "krr_gamma", None),
        degree=getattr(config, "krr_degree", 2),
        coef0=getattr(config, "krr_coef0", 1.0),
        reg_lambda=getattr(config, "bs_lambda", 1.0),
        return_loo=return_loo,
    )
