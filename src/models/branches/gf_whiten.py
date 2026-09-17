"""Cohort standardisation + fixed PCA-whitening in front of the GF Fisher Vector.

Why this exists. Fitting the background GMM directly on raw UNI2 embeddings makes
it partition by COHORT, not by tissue: measured on the background validation
sample, the number of components whose responsibility mass is >90% a single
dataset was 7/8 at M=8, 15/16 at M=16 and 31/32 at M=32 -- the ratio is pinned at
(M-1)/M, so adding components only subdivides each cohort's own region. The
consequence downstream is that an unseen cohort lands almost entirely on one
component (Primary 7 effective component counts 1.03-2.53 across M=8/16/32), and
the Fisher Vector degenerates towards a single-Gaussian moment descriptor.

Between-cohort variation (scanner, stain, site, preprocessing) simply dominates
within-tissue variation in the raw space, and EM explains the largest variance
first. This module removes that dominance in two steps, both label-free:

  1. Cohort standardisation -- each cohort's patches are centred and scaled by
     that cohort's own per-dimension mean and std. Deliberately NOT per slide:
     centring a slide would drive its Fisher mean-gradient G_mu to exactly zero
     and delete half the descriptor. Every Primary 7 task lives in a single
     cohort, so context and query slides receive the same affine map.
  2. PCA-whitening -- a fixed projection onto the leading `n_out` directions of
     the cohort-standardised background sample, divided by sqrt(eigenvalue) so no
     direction keeps a variance advantage. This is the standard Fisher Vector
     recipe and it also shrinks the descriptor to 2*M*n_out.

The PCA basis is fitted ONCE on background cohorts only (COMET / MUT-HET-RCC /
PANDA) and is a constant at evaluation time. The cohort statistics of an
evaluation cohort are computed from that cohort's own unlabelled patches -- see
`accumulate_moments`; this is label-free but transductive over the cohort, and
is recorded as such rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

_EPS = 1e-6


@dataclass
class CohortStats:
    """Per-dimension mean and std of one cohort's patch embeddings."""

    mean: torch.Tensor
    std: torch.Tensor
    n_patches: int

    def to(self, device: torch.device | str) -> "CohortStats":
        return CohortStats(
            mean=self.mean.to(device=device, dtype=torch.float32),
            std=self.std.to(device=device, dtype=torch.float32),
            n_patches=self.n_patches,
        )


def accumulate_moments(
    bag: torch.Tensor,
    total: torch.Tensor,
    total_sq: torch.Tensor,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Fold one slide into running (sum, sum of squares, count) accumulators.

    Kept separate from the descriptor pass so a caller can gather a cohort's
    statistics over whichever slide subset it is allowed to see.
    """
    values = bag.to(device=total.device, dtype=torch.float64)
    return total + values.sum(dim=0), total_sq + values.square().sum(dim=0), count + values.shape[0]


def finalize_moments(
    total: torch.Tensor, total_sq: torch.Tensor, count: int
) -> CohortStats:
    mean = total / max(count, 1)
    var = (total_sq / max(count, 1)) - mean.square()
    return CohortStats(
        mean=mean.float(),
        std=var.clamp_min(0.0).sqrt().clamp_min(_EPS).float(),
        n_patches=count,
    )


@dataclass
class GFWhitening:
    """Fixed PCA-whitening fitted on the cohort-standardised background sample.

    `components` already carries the 1/sqrt(eigenvalue) scaling, so `transform`
    is one subtract and one matmul.
    """

    components: torch.Tensor          # (n_out, n_in), whitened rows
    background_mean: torch.Tensor     # (n_in,) mean after cohort standardisation
    explained_variance: torch.Tensor  # (n_out,) eigenvalues, largest first

    @property
    def n_in(self) -> int:
        return int(self.components.shape[1])

    @property
    def n_out(self) -> int:
        return int(self.components.shape[0])

    def to(self, device: torch.device | str) -> "GFWhitening":
        return GFWhitening(
            components=self.components.to(device=device, dtype=torch.float32),
            background_mean=self.background_mean.to(device=device, dtype=torch.float32),
            explained_variance=self.explained_variance.to(device=device, dtype=torch.float32),
        )

    @classmethod
    def fit(
        cls, features: torch.Tensor, n_out: int, device: torch.device | str | None = None
    ) -> "GFWhitening":
        """Eigendecomposition of the sample covariance, kept in float64.

        `features` must already be cohort-standardised. The covariance is
        accumulated in chunks on `device` (default: wherever `features` lives),
        so a 3M x 1536 CPU sample can be reduced on a GPU without ever sitting
        there in full float64.
        """
        n_in = features.shape[1]
        device = torch.device(device) if device is not None else features.device
        total = torch.zeros(n_in, dtype=torch.float64, device=device)
        gram = torch.zeros(n_in, n_in, dtype=torch.float64, device=device)
        count = 0
        for start in range(0, features.shape[0], 262144):
            chunk = features[start:start + 262144].to(device=device, dtype=torch.float64)
            total += chunk.sum(dim=0)
            gram += chunk.T @ chunk
            count += chunk.shape[0]

        mean = total / max(count, 1)
        cov = gram / max(count, 1) - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        order = torch.argsort(eigvals, descending=True)[:n_out]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        components = (eigvecs * eigvals.clamp_min(1e-10).rsqrt()).T
        return cls(
            components=components.float(),
            background_mean=mean.float(),
            explained_variance=eigvals.float(),
        )

    def transform(self, bag: torch.Tensor, stats: CohortStats | None = None) -> torch.Tensor:
        """Cohort-standardise (when `stats` is given), then project and whiten."""
        values = bag.to(device=self.components.device, dtype=torch.float32)
        if stats is not None:
            values = (values - stats.mean) / stats.std
        return (values - self.background_mean) @ self.components.T

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        """Persist the fitted basis.

        The projection is part of the fixed external basis: an evaluation run
        that does not load the exact same `components` is measuring a different
        descriptor, so the basis is stored with the background cohort statistics
        it was fitted under and with whatever provenance the caller passes in.
        """
        torch.save(
            {
                "components": self.components.cpu(),
                "background_mean": self.background_mean.cpu(),
                "explained_variance": self.explained_variance.cpu(),
                "metadata": metadata or {},
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GFWhitening":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        return cls(
            components=blob["components"],
            background_mean=blob["background_mean"],
            explained_variance=blob["explained_variance"],
        )

    @staticmethod
    def load_metadata(path: str | Path) -> dict:
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        return blob.get("metadata", {})


def cohort_standardise(
    features: torch.Tensor, labels: torch.Tensor, n_cohorts: int
) -> tuple[torch.Tensor, list[CohortStats]]:
    """Standardise each cohort's rows by that cohort's own mean/std, in place.

    Returns the same tensor (modified) plus the per-cohort statistics, so the
    background statistics can be recorded next to the fitted basis.
    """
    stats: list[CohortStats] = []
    for cohort in range(n_cohorts):
        mask = labels == cohort
        if not bool(mask.any()):
            stats.append(CohortStats(
                mean=torch.zeros(features.shape[1]),
                std=torch.ones(features.shape[1]),
                n_patches=0,
            ))
            continue
        block = features[mask]
        mean = block.mean(dim=0)
        std = block.std(dim=0).clamp_min(_EPS)
        features[mask] = (block - mean) / std
        stats.append(CohortStats(mean=mean, std=std, n_patches=int(mask.sum())))
    return features, stats


__all__ = [
    "CohortStats",
    "GFWhitening",
    "accumulate_moments",
    "cohort_standardise",
    "finalize_moments",
]
