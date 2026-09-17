"""GF branch: improved Fisher Vector over a FIXED external background GMM.

Every other branch in this repository builds its basis from the fold context
(within-slide / within-context PCA), so nothing outside the fold enters the
active path. GF is deliberately different: it reads each slide through a
diagonal-covariance GMM fitted ONCE, offline, on background cohorts
(COMET / MUT-HET-RCC / PANDA -- disjoint from Primary 7 and SEAL 10) by
scripts/data/build_gf_background_gmm.py. At evaluation time the GMM is a
constant; no parameter here is fitted by gradient descent and no label is used
to build the descriptor.

Because that fixed external basis sits outside the convention the other
branches follow, GF is run STANDALONE (user decision, 2026-09-16): the
adoption gates (PROJECT.md SS5) and the promotion review (SS4) are skipped, so
nothing produced through this module may be compared against the official
7-branch basis or used to claim adoption.

Descriptor (Perronnin & Dance improved Fisher Vector):

    G_mu[k]    = 1/(N sqrt(pi_k))   * sum_n gamma_nk (x_n - mu_k) / sigma_k
    G_sigma[k] = 1/(N sqrt(2 pi_k)) * sum_n gamma_nk [ (x_n - mu_k)^2 / sigma_k^2 - 1 ]

followed by signed power normalisation sign(z)|z|^alpha (alpha = 0.5 by
default) and global L2 normalisation. The concatenated descriptor has
2 * M * D entries.

The accumulation never materialises an (N, M, D) tensor: the three sufficient
statistics below are obtained with two (M, N) x (N, D) matmuls, so peak memory
is O(N*M + M*D) and a 50k-patch slide at M=32 costs a few hundred MB.

    S0[k]   = sum_n gamma_nk
    S1[k,:] = sum_n gamma_nk x_n
    S2[k,:] = sum_n gamma_nk x_n^2
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from src.models.common.solvers import solve_kernel_ridge

_EPS = 1e-12
_VAR_FLOOR = 1e-8

# Patch-chunk size for the responsibility pass. Bounds the (chunk, M) log-prob
# buffer; every statistic below is additive over chunks, so this changes memory
# only, not the result.
_CHUNK = 65536


@dataclass
class BackgroundGMM:
    """A fixed diagonal-covariance GMM: weights (M,), means (M, D), variances (M, D)."""

    weights: torch.Tensor
    means: torch.Tensor
    variances: torch.Tensor

    @property
    def n_components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    @property
    def feature_dim(self) -> int:
        """Length of the Fisher Vector this GMM produces (mean + variance blocks)."""
        return 2 * self.n_components * self.n_features

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "BackgroundGMM":
        return BackgroundGMM(
            weights=self.weights.to(device=device, dtype=dtype),
            means=self.means.to(device=device, dtype=dtype),
            variances=self.variances.to(device=device, dtype=dtype).clamp_min(_VAR_FLOOR),
        )

    @classmethod
    def from_sklearn(cls, model) -> "BackgroundGMM":
        """Wrap a fitted sklearn GaussianMixture with covariance_type='diag'."""
        covariance_type = getattr(model, "covariance_type", None)
        if covariance_type != "diag":
            raise ValueError(f"GF expects covariance_type='diag', got {covariance_type!r}")
        return cls(
            weights=torch.as_tensor(model.weights_, dtype=torch.float32),
            means=torch.as_tensor(model.means_, dtype=torch.float32),
            variances=torch.as_tensor(model.covariances_, dtype=torch.float32).clamp_min(_VAR_FLOOR),
        )

    @classmethod
    def from_pickle(cls, path: str | Path) -> "BackgroundGMM":
        """Load a `*_model.pkl` written by scripts/data/build_gf_background_gmm.py.

        Unpickling needs scikit-learn importable; only the three parameter arrays
        are kept afterwards, so nothing else of the estimator survives.
        """
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        return cls.from_sklearn(model)


def gf_responsibilities(bag: torch.Tensor, gmm: BackgroundGMM) -> torch.Tensor:
    """Posterior responsibilities gamma (N, M) of every patch under the fixed GMM.

    Uses the expanded quadratic form so the (N, M) log-prob buffer is the only
    allocation that scales with N.
    """
    values = bag.to(device=gmm.means.device, dtype=torch.float32)
    inv_var = gmm.variances.reciprocal()                                   # (M, D)
    # -0.5 * [ D log(2pi) + sum_d log var_kd + sum_d mu_kd^2 / var_kd ] + log pi_k
    const = (
        gmm.weights.clamp_min(_EPS).log()
        - 0.5 * (
            gmm.n_features * math.log(2.0 * math.pi)
            + gmm.variances.log().sum(dim=1)
            + (gmm.means.square() * inv_var).sum(dim=1)
        )
    )                                                                       # (M,)
    log_prob = (
        -0.5 * (values.square() @ inv_var.T)
        + values @ (gmm.means * inv_var).T
        + const
    )                                                                       # (N, M)
    responsibilities = torch.softmax(log_prob, dim=1)
    if not bool(torch.isfinite(responsibilities).all()):
        raise FloatingPointError("GF responsibilities contain NaN or Inf")
    return responsibilities


def _sufficient_statistics(
    bag: torch.Tensor, gmm: BackgroundGMM
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Accumulate (S0, S1, S2, N) over patch chunks."""
    device = gmm.means.device
    n_components, n_features = gmm.n_components, gmm.n_features
    s0 = torch.zeros(n_components, device=device, dtype=torch.float32)
    s1 = torch.zeros(n_components, n_features, device=device, dtype=torch.float32)
    s2 = torch.zeros(n_components, n_features, device=device, dtype=torch.float32)

    total = 0
    for start in range(0, bag.shape[0], _CHUNK):
        chunk = bag[start:start + _CHUNK].to(device=device, dtype=torch.float32)
        resp = gf_responsibilities(chunk, gmm)                              # (n, M)
        s0 += resp.sum(dim=0)
        s1 += resp.T @ chunk
        s2 += resp.T @ chunk.square()
        total += chunk.shape[0]
    return s0, s1, s2, total


def gf_slide_features(
    bag: torch.Tensor,
    gmm: BackgroundGMM,
    alpha: float = 0.5,
    include_variance: bool = True,
    return_occupancy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Improved Fisher Vector of one slide's tokens against the fixed GMM.

    Returns a 1-D tensor of length 2*M*D (M*D when `include_variance` is False),
    power-normalised and L2-normalised. With `return_occupancy` the slide's mean
    responsibility vector (M,) is returned alongside it -- it falls out of the
    same accumulation, so the diagnostic costs nothing extra.
    """
    # Same fp32 contract as SH/SJ: the eval pipeline may run under bf16 autocast,
    # and the responsibilities here go through an exp of a 1536-term quadratic
    # form that bf16 would perturb well beyond the differences this branch reads.
    with torch.autocast(device_type=gmm.means.device.type, enabled=False):
        s0, s1, s2, total = _sufficient_statistics(bag, gmm)
        if total == 0:
            empty = torch.zeros(
                gmm.feature_dim if include_variance else gmm.feature_dim // 2,
                device=gmm.means.device, dtype=torch.float32,
            )
            return (empty, s0) if return_occupancy else empty

        means, variances, weights = gmm.means, gmm.variances, gmm.weights
        sigma = variances.sqrt()
        norm = float(total)
        sqrt_pi = weights.clamp_min(_EPS).sqrt()

        # G_mu[k] = (S1 - S0 mu) / (N sqrt(pi) sigma)
        g_mu = (s1 - s0[:, None] * means) / (norm * sqrt_pi[:, None] * sigma)

        blocks = [g_mu]
        if include_variance:
            # G_sigma[k] = (S2 - 2 mu S1 + S0 mu^2 - S0 sigma^2) / (N sqrt(2 pi) sigma^2)
            numerator = (
                s2
                - 2.0 * means * s1
                + s0[:, None] * means.square()
                - s0[:, None] * variances
            )
            g_sigma = numerator / (norm * math.sqrt(2.0) * sqrt_pi[:, None] * variances)
            blocks.append(g_sigma)

        vector = torch.cat([b.reshape(-1) for b in blocks])
        if not bool(torch.isfinite(vector).all()):
            raise FloatingPointError("GF Fisher Vector contains NaN or Inf before normalisation")

        # Signed power normalisation, then global L2.
        if alpha != 1.0:
            vector = vector.sign() * vector.abs().pow(alpha)
        vector = vector / vector.norm().clamp_min(_EPS)
        return (vector, s0 / norm) if return_occupancy else vector


def gf_features(
    context_bags: Sequence[torch.Tensor],
    context_labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    gmm: BackgroundGMM,
    alpha: float = 0.5,
    include_variance: bool = True,
    reg_lambda: float = 1.0,
    kernel: str = "linear",
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fisher Vector descriptors through the same class-balanced kernel ridge the
    other branches use, so the readout stays closed-form and label-free until the
    ridge solve itself."""

    def stack(bags: Sequence[torch.Tensor]) -> torch.Tensor:
        feats = torch.stack([gf_slide_features(b, gmm, alpha, include_variance) for b in bags])
        if not bool(torch.isfinite(feats).all()):
            raise FloatingPointError("GF descriptor matrix contains NaN or Inf")
        return feats

    return solve_kernel_ridge(
        stack(context_bags), context_labels, stack(query_bags),
        kernel=kernel,
        reg_lambda=reg_lambda,
        return_loo=return_loo,
    )


def gf_margins_from_descriptors(
    context_feats: torch.Tensor,
    context_labels: torch.Tensor,
    query_feats: torch.Tensor,
    reg_lambda: float = 1.0,
    kernel: str = "linear",
) -> torch.Tensor:
    """Same readout as `gf_features` for descriptors that were cached per slide.

    The fixed GMM does not depend on the fold, so a slide's Fisher Vector is
    identical in every fold it appears in; the standalone runner computes each
    one once and calls this.
    """
    return solve_kernel_ridge(
        context_feats, context_labels, query_feats,
        kernel=kernel, reg_lambda=reg_lambda,
    )


__all__ = [
    "BackgroundGMM",
    "gf_features",
    "gf_margins_from_descriptors",
    "gf_responsibilities",
    "gf_slide_features",
]
