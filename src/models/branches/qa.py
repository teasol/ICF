"""QA branch: quantile / extremum statistics of projected cells with class-balanced ridge."""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge


def qa_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Quantile & Extremum Evidence (QA): multi-quantile features of projected cells with class-balanced kernel ridge."""
    dim = min(config.qa_dim, basis.shape[1])
    qa_basis = basis[:, :dim].to(dtype=torch.float32)
    quantiles = torch.tensor(config.qa_quantiles, device=qa_basis.device, dtype=torch.float32)

    def extract_quantiles(bag: torch.Tensor) -> torch.Tensor:
        z = bag.float().to(qa_basis.device) @ qa_basis
        q = torch.quantile(z, quantiles, dim=0)  # (n_quantiles, dim)
        return q.flatten()

    ctx_feats = torch.stack([extract_quantiles(b) for b in context_bags])
    qry_feats = torch.stack([extract_quantiles(b) for b in query_bags])

    kernel = getattr(config, "qa_kernel", getattr(config, "krr_kernel", "linear"))
    gamma = getattr(config, "krr_gamma", None)
    degree = getattr(config, "krr_degree", 2)
    coef0 = getattr(config, "krr_coef0", 1.0)
    return solve_kernel_ridge(
        ctx_feats, context_labels, qry_feats,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        reg_lambda=config.qa_lambda,
        return_loo=return_loo,
    )
