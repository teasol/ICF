"""FC (Focus-Contrast) branch prototype: multiplicity of salience focus as features.

DS picks one focus temperature tau and returns a single denoised mean. FC keeps
the intensity axis instead of choosing: it computes the denoised mean at several
taus and stacks the differences between consecutive intensities. The result is a
feature block, read out by the same class-balanced ridge DS uses.

Not wired into TrainingFreeConfig / voting yet -- it takes explicit parameters so
a probe can screen it (gate 1) without touching the fixed-branch contracts.

Design: talks/reports/2026-09-20_fc_design_v0.md
"""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge


def _soft_assignments(proj_bags: Sequence[torch.Tensor], centroids: torch.Tensor):
    abundances, assignments = [], []
    for p in proj_bags:
        p_norm = torch.nn.functional.normalize(p, dim=-1)
        sim = p_norm @ centroids.T
        soft_p = torch.nn.functional.softmax(sim * 5.0, dim=-1)
        abundances.append(soft_p.mean(dim=0))
        assignments.append(soft_p)
    return torch.stack(abundances), assignments


def _denoised_mean(proj_bags, assignments, s_abs, tau: float):
    feats = []
    for p, soft_p in zip(proj_bags, assignments):
        u = soft_p @ s_abs
        u_std = u.std().clamp_min(1e-6)
        w = torch.nn.functional.softmax(tau * (u - u.mean()) / u_std, dim=0)
        feats.append((w.unsqueeze(-1) * p).sum(dim=0))
    return torch.stack(feats)


def fc_features(
    dim: int,
    taus: Sequence[float],
    lambda_: float,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Focus-contrast margins. Same salience machinery as DS, intensity marginalised."""
    ds_basis = basis[:, :dim].to(dtype=torch.float32)
    device = ds_basis.device
    labels = context_labels.long().to(device)

    ctx_proj = [b.float().to(device) @ ds_basis for b in context_bags]
    qry_proj = [b.float().to(device) @ ds_basis for b in query_bags]

    sampled = []
    for p in ctx_proj:
        if p.shape[0] > 0:
            idx = torch.linspace(0, p.shape[0] - 1, min(p.shape[0], 64), device=device).long()
            sampled.append(p[idx])
    all_cells = torch.cat(sampled, dim=0) if sampled else torch.zeros(1, dim, device=device)
    K = min(256, all_cells.shape[0])
    stride = all_cells.shape[0] / K
    centroids = all_cells[(torch.arange(K, device=device) * stride).long()]
    centroids = torch.nn.functional.normalize(centroids, dim=-1)

    ctx_abundances, ctx_assign = _soft_assignments(ctx_proj, centroids)
    qry_abundances, qry_assign = _soft_assignments(qry_proj, centroids)

    eps = 1e-5
    mask1, mask0 = (labels == 1), (labels == 0)
    a1 = ctx_abundances[mask1].mean(dim=0) if mask1.any() else ctx_abundances.mean(dim=0)
    a0 = ctx_abundances[mask0].mean(dim=0) if mask0.any() else ctx_abundances.mean(dim=0)
    s_abs = torch.log((a1 + eps) / (a0 + eps)).abs()

    def features(proj, assign):
        zs = [_denoised_mean(proj, assign, s_abs, t) for t in taus]
        blocks = [zs[0]] + [zs[i] - zs[i - 1] for i in range(1, len(zs))]
        return torch.cat(blocks, dim=-1)

    ctx_feats = features(ctx_proj, ctx_assign)
    qry_feats = features(qry_proj, qry_assign)
    return solve_kernel_ridge(
        ctx_feats, context_labels, qry_feats,
        kernel="linear", gamma=None, degree=2, coef0=1.0,
        reg_lambda=lambda_, return_loo=return_loo,
    )
