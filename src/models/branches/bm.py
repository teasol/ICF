"""BM branch: projected bag-mean in leading subspace with class-balanced ridge."""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge


def bm_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Projected bag-mean in leading subspace with class-balanced kernel ridge."""
    dim = min(config.bm_dim, basis.shape[1])
    bm_basis = basis[:, :dim].to(dtype=torch.float32)
    ctx_means = torch.stack([b.float().mean(dim=0).to(bm_basis.device) for b in context_bags]) @ bm_basis
    qry_means = torch.stack([b.float().mean(dim=0).to(bm_basis.device) for b in query_bags]) @ bm_basis

    kernel = getattr(config, "bm_kernel", getattr(config, "krr_kernel", "linear"))
    gamma = getattr(config, "krr_gamma", None)
    degree = getattr(config, "krr_degree", 2)
    coef0 = getattr(config, "krr_coef0", 1.0)
    return solve_kernel_ridge(
        ctx_means, context_labels, qry_means,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        reg_lambda=config.bm_lambda,
        return_loo=return_loo,
    )
