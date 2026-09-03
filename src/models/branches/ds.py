"""DS branch: in-context salience denoising with class-balanced ridge."""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge


def ds_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """In-Context Salience Denoising (DS): class-contrastive cluster salience weighting for noise patch suppression."""
    dim = min(config.ds_dim, basis.shape[1])
    ds_basis = basis[:, :dim].to(dtype=torch.float32)
    device = ds_basis.device
    labels = context_labels.long().to(device)

    # 1. Project all context and query bags to PCA subspace
    ctx_proj = [b.float().to(device) @ ds_basis for b in context_bags]
    qry_proj = [b.float().to(device) @ ds_basis for b in query_bags]

    # 2. Select K cluster centroids from sampled context cells
    sampled_cells = []
    for p in ctx_proj:
        n_cells = p.shape[0]
        if n_cells > 0:
            idx = torch.linspace(0, n_cells - 1, min(n_cells, 64), device=device).long()
            sampled_cells.append(p[idx])
    all_cells = torch.cat(sampled_cells, dim=0) if sampled_cells else torch.zeros(1, dim, device=device)

    K = min(config.ds_tokens, all_cells.shape[0])
    if all_cells.shape[0] > K:
        stride = all_cells.shape[0] / K
        centroids = all_cells[(torch.arange(K, device=device) * stride).long()]
    else:
        centroids = all_cells

    centroids = torch.nn.functional.normalize(centroids, dim=-1)

    # 3. Soft cluster assignments and slide abundances
    def get_assignments(proj_bags):
        abundances = []
        patch_assignments = []
        for p in proj_bags:
            p_norm = torch.nn.functional.normalize(p, dim=-1)
            sim = p_norm @ centroids.T  # (N_i, K)
            soft_p = torch.nn.functional.softmax(sim * 5.0, dim=-1)  # (N_i, K)
            a = soft_p.mean(dim=0)  # (K,)
            abundances.append(a)
            patch_assignments.append(soft_p)
        return torch.stack(abundances), patch_assignments

    ctx_abundances, ctx_assignments = get_assignments(ctx_proj)
    qry_abundances, qry_assignments = get_assignments(qry_proj)

    # 4. In-context Class Salience Log-Odds
    eps = 1e-5
    mask1 = (labels == 1)
    mask0 = (labels == 0)
    a1 = ctx_abundances[mask1].mean(dim=0) if mask1.any() else ctx_abundances.mean(dim=0)
    a0 = ctx_abundances[mask0].mean(dim=0) if mask0.any() else ctx_abundances.mean(dim=0)

    s = torch.log((a1 + eps) / (a0 + eps))  # (K,)
    s_abs = s.abs()  # Salience magnitude

    # 5. Denoised bag mean extraction
    def extract_denoised_mean(proj_bags, assignments):
        feats = []
        temp = config.ds_temperature
        for p, soft_p in zip(proj_bags, assignments):
            u = soft_p @ s_abs  # (N_i,)
            u_std = u.std().clamp_min(1e-6)
            w = torch.nn.functional.softmax(temp * (u - u.mean()) / u_std, dim=0)  # (N_i,)
            z_denoised = (w.unsqueeze(-1) * p).sum(dim=0)  # (dim,)
            feats.append(z_denoised)
        return torch.stack(feats)

    ctx_feats = extract_denoised_mean(ctx_proj, ctx_assignments)
    qry_feats = extract_denoised_mean(qry_proj, qry_assignments)

    # 6. Class-balanced kernel ridge
    kernel = getattr(config, "ds_kernel", getattr(config, "krr_kernel", "linear"))
    gamma = getattr(config, "krr_gamma", None)
    degree = getattr(config, "krr_degree", 2)
    coef0 = getattr(config, "krr_coef0", 1.0)
    return solve_kernel_ridge(
        ctx_feats, context_labels, qry_feats,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        reg_lambda=config.ds_lambda,
        return_loo=return_loo,
    )
