"""LR branch: direct in-context patch likelihood ratio + Top-K MIL extreme pooling."""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge


def lr_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
) -> torch.Tensor:
    """Direct In-Context Patch Likelihood Ratio + Top-K MIL Extreme Pooling with class-balanced ridge."""
    dim = min(config.lr_dim, basis.shape[1])
    lr_basis = basis[:, :dim].to(dtype=torch.float32)
    device = lr_basis.device
    labels = context_labels.long().to(device)

    # 1. Project all context and query bags to PCA subspace
    ctx_proj = [b.float().to(device) @ lr_basis for b in context_bags]
    qry_proj = [b.float().to(device) @ lr_basis for b in query_bags]

    # 2. Build Class 0 and Class 1 patch memory banks
    bank_0, bank_1 = [], []
    for p, y in zip(ctx_proj, labels):
        n_c = p.shape[0]
        if n_c == 0:
            continue
        n_sample = min(n_c, config.lr_patches_per_ctx)
        idx = torch.linspace(0, n_c - 1, n_sample, device=device).long()
        sampled = p[idx]
        if y == 1:
            bank_1.append(sampled)
        else:
            bank_0.append(sampled)

    if not bank_0 or not bank_1:
        return torch.zeros(len(query_bags), device=device)

    P0 = torch.cat(bank_0, dim=0)
    P1 = torch.cat(bank_1, dim=0)

    P0_norm = torch.nn.functional.normalize(P0, dim=-1)
    P1_norm = torch.nn.functional.normalize(P1, dim=-1)
    tau = config.lr_tau

    # 3. Compute Patch-level log-odds likelihood ratio and extract Top-K extreme instance features
    def get_slide_lr_features(proj_bags):
        feats = []
        for bag in proj_bags:
            n_c = bag.shape[0]
            if n_c == 0:
                feats.append(torch.zeros(dim + 1, device=device))
                continue
            bag_norm = torch.nn.functional.normalize(bag, dim=-1)
            sim1 = bag_norm @ P1_norm.T  # (N_i, |P1|)
            sim0 = bag_norm @ P0_norm.T  # (N_i, |P0|)

            score1 = torch.logsumexp(sim1 * tau, dim=-1) - torch.log(torch.tensor(P1_norm.shape[0], dtype=torch.float32, device=device))
            score0 = torch.logsumexp(sim0 * tau, dim=-1) - torch.log(torch.tensor(P0_norm.shape[0], dtype=torch.float32, device=device))

            lr = score1 - score0  # (N_i,)

            k = max(config.lr_topk_min, min(config.lr_topk_max, int(n_c * config.lr_topk_fraction)))
            k = min(k, n_c)

            topk_vals, topk_idx = torch.topk(lr, k=k, largest=True)
            botk_vals, botk_idx = torch.topk(lr, k=k, largest=False)

            z_plus = bag[topk_idx].mean(dim=0)
            z_minus = bag[botk_idx].mean(dim=0)

            delta_z = z_plus - z_minus
            e_scalar = 0.5 * (topk_vals.mean() + botk_vals.mean())

            v_i = torch.cat([delta_z, e_scalar.unsqueeze(0)], dim=-1)
            feats.append(v_i)
        return torch.stack(feats)

    ctx_feats = get_slide_lr_features(ctx_proj)
    qry_feats = get_slide_lr_features(qry_proj)

    # 4. Class-balanced linear dual ridge solve
    return solve_kernel_ridge(
        ctx_feats, labels, qry_feats,
        kernel="linear",
        reg_lambda=config.lr_lambda,
    )
