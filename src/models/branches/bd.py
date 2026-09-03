"""BD branch: bag dispersion / spectral entropy of projected covariance."""

from __future__ import annotations

import torch

from src.models.common.solvers import solve_ridge
from src.models.dd_adaptive_rank import ordered_typicality_margin


def bd_features(
    config,
    context_cov: torch.Tensor,
    labels: torch.Tensor,
    query_cov: torch.Tensor,
) -> torch.Tensor:
    """Bag Dispersion (BD): spectral entropy or log-trace of projected covariance."""
    dim = min(config.bd_dim, context_cov.shape[-1])
    ctx_sub = context_cov[:, :dim, :dim]
    qry_sub = query_cov[:, :dim, :dim]

    if config.bd_metric == "entropy":
        ctx_eig = torch.linalg.eigvalsh(ctx_sub.float()).clamp_min(config.bd_eps)
        qry_eig = torch.linalg.eigvalsh(qry_sub.float()).clamp_min(config.bd_eps)
        ctx_p = ctx_eig / ctx_eig.sum(dim=-1, keepdim=True).clamp_min(config.bd_eps)
        qry_p = qry_eig / qry_eig.sum(dim=-1, keepdim=True).clamp_min(config.bd_eps)

        ctx_v = -(ctx_p * torch.log(ctx_p.clamp_min(config.bd_eps))).sum(dim=-1)
        qry_v = -(qry_p * torch.log(qry_p.clamp_min(config.bd_eps))).sum(dim=-1)
        if dim > 1:
            log_dim = torch.log(torch.tensor(float(dim), device=ctx_v.device, dtype=ctx_v.dtype))
            ctx_v = ctx_v / log_dim
            qry_v = qry_v / log_dim
    elif config.bd_metric == "trace":
        ctx_trace = ctx_sub.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(config.bd_eps)
        qry_trace = qry_sub.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(config.bd_eps)
        ctx_v = ctx_trace.log()
        qry_v = qry_trace.log()
    else:
        raise ValueError(f"Unknown bd_metric: {config.bd_metric!r}")

    labels = labels.long()
    prototypes = torch.stack([ctx_v[labels == c].mean() for c in range(2)])
    dispersions = torch.stack([
        (ctx_v[labels == c] - prototypes[c]).square().mean().clamp_min(config.bd_eps)
        for c in range(2)
    ])


    if config.bd_readout == "ordered_typicality":
        margin = ordered_typicality_margin(
            qry_v,
            prototypes,
            dispersions,
            config.bd_eps,
            config.bd_separation_floor,
        )
        return margin
    elif config.bd_readout == "ridge":
        centre = ctx_v.mean()
        scale = (ctx_v - centre).square().mean().sqrt().clamp_min(config.bd_eps)
        std_ctx = ((ctx_v - centre) / scale).unsqueeze(-1)
        std_qry = ((qry_v - centre) / scale).unsqueeze(-1)

        targets = torch.nn.functional.one_hot(labels, 2).float()
        counts = torch.bincount(labels, minlength=2)
        if bool((counts == 0).any()):
            raise ValueError("Every class must occur in the context set.")
        weight = counts.float().reciprocal()[labels]
        total = weight.sum().clamp_min(1e-12)
        feat_mean = (weight[:, None] * std_ctx).sum(0, keepdim=True) / total
        tgt_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
        root = weight.sqrt()[:, None]

        design = (std_ctx - feat_mean) * root
        centred_targets = (targets - tgt_mean) * root

        dual = solve_ridge(design @ design.T, centred_targets, config.bd_lambda)
        coefficients = design.T @ dual
        intercept = tgt_mean - feat_mean @ coefficients
        logits = std_qry @ coefficients + intercept
        return logits[:, 1] - logits[:, 0]
    else:
        raise ValueError(f"Unknown bd_readout: {config.bd_readout!r}")
