"""SHJ branch: joint (multivariate) shape of the whitened token cloud.

Every Location-family branch (CV, BM, QA, DS) describes where a slide's tokens
sit; BD describes the spread of the eigenvalue spectrum. SHJ describes the shape
of the radius distribution after whitening each slide by its OWN mean and
covariance, so it is location- and scale-invariant by construction and cannot
restate the mean (§217-§219). Admitted in §220 as a task specialist: on ARID1A it
reaches 0.6363 against DS's 0.6193, above chance on 48/50 folds, with |r| <= 0.132
against every existing branch there.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge

SHJ_FEATURE_DIM = 8
_EPS = 1e-6


def _quantile(sorted_values: torch.Tensor, frac: float) -> torch.Tensor:
    """Linear-interpolated quantile of an already-sorted 1-D tensor.

    torch.quantile caps its input size, and slide bags routinely exceed it, so the
    quantiles are read off a sort instead.
    """
    n = sorted_values.shape[0]
    pos = frac * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def shj_slide_features(bag: torch.Tensor, basis: torch.Tensor, dim: int) -> torch.Tensor:
    """Eight shape descriptors of one slide's whitened-radius distribution.

    Shared by the training-free pipeline and scripts/test_pathobench.py so the two
    cannot drift apart.
    """
    # Autocast must be off here. The eval pipeline runs under bf16 autocast, which
    # would compute the projection in bf16 (~1e-3 relative error); whitening then
    # amplifies that by roughly two orders of magnitude, moving the margin by ~10%
    # and making the feature depend on the matmul's shape. Forcing float32 makes
    # the branch reproducible and independent of the surrounding autocast state.
    with torch.autocast(device_type=bag.device.type, enabled=False):
        values = bag.to(basis.device, dtype=torch.float32)
        proj = values @ basis[:, :dim].to(dtype=torch.float32)
        centred = proj - proj.mean(dim=0, keepdim=True)

        cov = (centred.T @ centred) / float(centred.shape[0])
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov.double())
        whitened = (centred.double() @ eigvecs) * eigvals.clamp_min(1e-8).rsqrt()
        radius = whitened.norm(dim=1).float()

    ordered = torch.sort(radius).values
    median = _quantile(ordered, 0.50).clamp_min(_EPS)
    iqr = (_quantile(ordered, 0.75) - _quantile(ordered, 0.25)).clamp_min(_EPS)
    standard = (radius - radius.mean()) / radius.std().clamp_min(_EPS)

    return torch.stack([
        standard.pow(3).mean(),                                  # skewness
        standard.pow(4).mean() - 3.0,                            # excess kurtosis
        (_quantile(ordered, 0.75) + _quantile(ordered, 0.25)
         - 2.0 * _quantile(ordered, 0.50)) / iqr,                # Bowley skewness
        (_quantile(ordered, 0.875) - _quantile(ordered, 0.625)
         + _quantile(ordered, 0.375) - _quantile(ordered, 0.125)) / iqr,  # Moors tail weight
        _quantile(ordered, 0.10) / median,
        _quantile(ordered, 0.90) / median,
        _quantile(ordered, 0.99) / median,
        iqr / median,
    ])


def shj_features(
    config,
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    basis: torch.Tensor,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Whitened-radius shape descriptors through a class-balanced kernel ridge."""
    dim = min(config.shj_dim, basis.shape[1])
    basis = basis.to(dtype=torch.float32)

    def stack(bags):
        feats = torch.stack([shj_slide_features(b, basis, dim) for b in bags])
        return torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return solve_kernel_ridge(
        stack(context_bags), context_labels, stack(query_bags),
        kernel=getattr(config, "shj_kernel", getattr(config, "krr_kernel", "linear")),
        gamma=getattr(config, "krr_gamma", None),
        degree=getattr(config, "krr_degree", 2),
        coef0=getattr(config, "krr_coef0", 1.0),
        reg_lambda=config.shj_lambda,
        return_loo=return_loo,
    )
