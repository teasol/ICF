"""SW branch: sliced Wasserstein distribution matching."""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge


def sw_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sliced Wasserstein Distribution Matching (SW):
    Projects patches to 32D PCA space, projects onto L deterministic orthogonal
    spherical slice directions, computes M 1D quantiles per slice, and solves
    Class-balanced Dual Ridge on the flattened (L * M) Sliced Wasserstein embedding.
    """
    dim = min(config.sw_dim, basis.shape[1])
    sw_basis = basis[:, :dim].to(dtype=torch.float32)
    device = sw_basis.device
    labels = context_labels.long().to(device)

    # 1. Deterministic orthogonal slice directions on S^{dim-1}
    num_slices = config.sw_num_slices
    g = torch.Generator(device="cpu").manual_seed(42)
    rand_dirs = torch.randn(dim, num_slices, generator=g, dtype=torch.float32)
    q_dirs, _ = torch.linalg.qr(rand_dirs)
    slice_dirs = q_dirs.to(device=device)  # (dim, num_slices)

    # 2. Quantile levels
    num_quantiles = config.sw_num_quantiles
    q_levels = torch.linspace(0.5 / num_quantiles, 1.0 - 0.5 / num_quantiles, num_quantiles, device=device)

    # 3. Project each bag onto slices and extract 1D quantiles
    def extract_sw_profile(bag):
        proj = bag.float().to(device) @ sw_basis  # (N_i, dim)
        n_c = proj.shape[0]
        if n_c == 0:
            return torch.zeros(num_slices * num_quantiles, device=device)

        slices = proj @ slice_dirs  # (N_i, num_slices)
        sorted_slices, _ = torch.sort(slices, dim=0)

        indices = (q_levels * (n_c - 1)).clamp(0, n_c - 1)
        low_idx = indices.floor().long()
        high_idx = indices.ceil().long()
        weights = (indices - low_idx.float())[:, None]

        low_vals = sorted_slices[low_idx, :]   # (num_quantiles, num_slices)
        high_vals = sorted_slices[high_idx, :] # (num_quantiles, num_slices)
        quantiles = (1.0 - weights) * low_vals + weights * high_vals

        return quantiles.flatten()

    ctx_feats = torch.stack([extract_sw_profile(b) for b in context_bags])
    qry_feats = torch.stack([extract_sw_profile(b) for b in query_bags])

    # 4. Class-balanced linear dual ridge solve
    return solve_kernel_ridge(
        ctx_feats, labels, qry_feats,
        kernel="linear",
        reg_lambda=config.sw_lambda,
        return_loo=return_loo,
    )
