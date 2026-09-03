"""DE branch: in-subspace dual extreme instance MIL."""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge


def de_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """In-Subspace Dual Extreme Instance MIL (DE):
    Projects patches to 32D within-slide PCA space, scores patches along class
    centroid contrast direction, extracts Top-K and Bottom-K extreme patches,
    and solves Class-balanced Dual Ridge on the difference descriptor.
    """
    dim = min(config.de_dim, basis.shape[1])
    de_basis = basis[:, :dim].to(dtype=torch.float32)
    device = de_basis.device
    labels = context_labels.long().to(device)

    # 1. Project all context and query bags to 32D PCA space
    ctx_proj = [b.float().to(device) @ de_basis for b in context_bags]
    qry_proj = [b.float().to(device) @ de_basis for b in query_bags]

    # 2. Compute Class 0 and Class 1 centroids from bag means
    ctx_means = torch.stack([p.mean(dim=0) for p in ctx_proj])
    if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
        zero_q = torch.zeros(len(query_bags), device=device)
        return (zero_q, torch.zeros(len(context_bags), device=device)) if return_loo else zero_q

    mu0 = ctx_means[labels == 0].mean(dim=0)
    mu1 = ctx_means[labels == 1].mean(dim=0)
    w_contrast = mu1 - mu0
    w_norm = w_contrast.norm().clamp_min(1e-12)
    w_dir = w_contrast / w_norm

    # 3. Extract Dual Extreme difference descriptor per bag
    def extract_de_vector(p_bag):
        n_c = p_bag.shape[0]
        if n_c == 0:
            return torch.zeros(dim + 1, device=device)
        scores = p_bag @ w_dir  # (N_i,)
        k = max(config.de_topk_min, min(config.de_topk_max, int(n_c * config.de_topk_fraction)))
        k = min(k, n_c)

        topk_vals, topk_idx = torch.topk(scores, k=k, largest=True)
        botk_vals, botk_idx = torch.topk(scores, k=k, largest=False)

        z_plus = p_bag[topk_idx].mean(dim=0)
        z_minus = p_bag[botk_idx].mean(dim=0)

        delta_z = z_plus - z_minus  # (dim,)
        score_diff = 0.5 * (topk_vals.mean() + botk_vals.mean())  # (1,)

        return torch.cat([delta_z, score_diff.unsqueeze(0)], dim=-1)

    ctx_feats = torch.stack([extract_de_vector(p) for p in ctx_proj])
    qry_feats = torch.stack([extract_de_vector(p) for p in qry_proj])

    # 4. Class-balanced linear dual ridge solve
    return solve_kernel_ridge(
        ctx_feats, labels, qry_feats,
        kernel="linear",
        reg_lambda=config.de_lambda,
        return_loo=return_loo,
    )
