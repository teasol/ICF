"""Adaptive-rank DD: take more than one dispersion direction, but only the ones
whose class separation survives a test (docs SS146).

WHAT DD DOES AT RANK 1. `_dd_direction` whitens the pooled sketched covariance,
forms `operator = W (C1bar - C0bar) W`, and takes the eigenvector of largest
|lambda|. That single direction u induces one scalar per bag,

    s_b = u^T C_b u        (a variance along u),

and `_dd_distance_features` models log(s_b) as class-conditional 1-D Gaussians:
standardise on the context, take per-class prototypes and dispersions, and emit
`d[q, c] = (f_q - mu_c)^2 / sigma_c^2`. The fixed head consumes only `d1 - d0`
(SS137-3), which is exactly a log-likelihood ratio.

WHY RANK 1 IS A CHOICE, NOT A LAW. `operator` has K eigenvalues; rank 1 keeps
one. SS145 showed DD saturates in K by 128, which says the top direction stops
improving — not that the others are empty.

THE EXTENSION. Keep r directions, sum the diagonal-Mahalanobis terms:

    d[q, c] = sum_j (f_qj - mu_cj)^2 / sigma_cj^2

Summing (not averaging) is the principled choice because that sum IS the
Gaussian discriminant, so `d1 - d0` stays a log-likelihood ratio. ⚠️ Its
magnitude therefore grows with the number of directions that genuinely separate,
which shifts DD's weight against CV in the fixed head. `scale_by_rank=True`
divides by the number kept, as a control for that confound rather than as the
default.

⚠️⚠️ THE TEST IS NOT A P-VALUE. Directions are selected to maximise class
dispersion difference on the very bags the t-statistic is then computed from, so
this is post-selection inference and |t| is inflated. It still ORDERS candidates
usefully, which is all the gate needs, but the threshold must be calibrated by
sweeping it against held-out tasks — never read off a t-table. Rank 1 is always
kept, so the arm degrades to today's behaviour rather than to nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AdaptiveRankConfig:
    rank_max: int = 1
    # |t| a direction beyond the first must reach. 0.0 keeps every candidate up
    # to rank_max; float("inf") keeps only rank 1 (today's behaviour).
    t_threshold: float = 2.5
    scale_by_rank: bool = False
    shrinkage: float = 0.25
    eps: float = 1e-6


def _welch_t(first: torch.Tensor, second: torch.Tensor, eps: float) -> torch.Tensor:
    """Two-sample t on unequal variances, for log(s_b) split by class.

    log is what makes this reasonable: s_b is a variance, so it is positive and
    right-skewed, and its log is far closer to the normality the test assumes.
    """
    if first.numel() < 2 or second.numel() < 2:
        return torch.zeros((), device=first.device, dtype=first.dtype)
    standard_error = (
        first.var(unbiased=True) / first.numel()
        + second.var(unbiased=True) / second.numel()
    ).clamp_min(eps).sqrt()
    return (second.mean() - first.mean()) / standard_error


def dispersion_directions(context_covariance, context_labels, config):
    """Candidate directions, ordered by descending |lambda|, plus the whitener.

    Mirrors `_dd_direction` exactly for the top one; `eigh` is never
    differentiated here (SS100), which training-free makes free to honour.
    """
    labels = context_labels.long()
    sketch_dim = context_covariance.shape[-1]
    with torch.no_grad():
        means = []
        for class_index in range(2):
            members = context_covariance[labels == class_index]
            if members.numel() == 0:
                raise ValueError("Every class must occur in the context set.")
            means.append(members.mean(dim=0))
        delta = means[1] - means[0]
        pooled = context_covariance.mean(dim=0)
        trace_scale = pooled.diagonal().mean().clamp_min(config.eps)
        identity = torch.eye(sketch_dim, device=pooled.device, dtype=pooled.dtype)
        shrunk = (1.0 - config.shrinkage) * pooled + config.shrinkage * trace_scale * identity
        values, vectors = torch.linalg.eigh(shrunk)
        whitening = (vectors * values.clamp_min(config.eps).rsqrt().unsqueeze(0)) @ vectors.T
        operator = whitening @ delta @ whitening
        eigenvalues, eigenvectors = torch.linalg.eigh(operator)
        order = eigenvalues.abs().argsort(descending=True)
        return whitening @ eigenvectors[:, order], eigenvalues[order]


def adaptive_dd_distance_features(
    context_covariance, context_labels, query_covariance, config
):
    """`_dd_distance_features` with an adaptive number of directions.

    Returns `(distances, separation, kept)` where `distances` is [queries, 2] as
    the rank-1 version is, so the fixed 12-feature head is untouched. `kept` is
    the number of directions used, for logging how often the gate fires.
    """
    labels = context_labels.long()
    directions, _ = dispersion_directions(context_covariance, context_labels, config)

    context_features, query_features, kept = [], [], 0
    for rank in range(min(config.rank_max, directions.shape[-1])):
        direction = directions[:, rank]
        context_scalar = torch.einsum(
            "d,bdk,k->b", direction, context_covariance, direction
        ).clamp_min(config.eps).log()
        query_scalar = torch.einsum(
            "d,qdk,k->q", direction, query_covariance, direction
        ).clamp_min(config.eps).log()
        if rank > 0:
            statistic = _welch_t(
                context_scalar[labels == 0], context_scalar[labels == 1], config.eps
            )
            if float(statistic.abs()) < config.t_threshold:
                continue
        # Context-only centring and scalar RMS, per direction, exactly as the
        # rank-1 path does. Each direction's log-variance has its own scale.
        centre = context_scalar.mean()
        scale = (context_scalar - centre).square().mean().sqrt().clamp_min(config.eps)
        context_features.append((context_scalar - centre) / scale)
        query_features.append((query_scalar - centre) / scale)
        kept += 1

    context_feature = torch.stack(context_features, dim=-1)
    query_feature = torch.stack(query_features, dim=-1)
    prototypes = torch.stack(
        [context_feature[labels == class_index].mean(dim=0) for class_index in range(2)]
    )
    dispersions = torch.stack(
        [
            (context_feature[labels == class_index] - prototypes[class_index])
            .square().mean(dim=0).clamp_min(config.eps)
            for class_index in range(2)
        ]
    )
    squared = (query_feature[:, None, :] - prototypes[None, :, :]).square()
    distances = (squared / dispersions[None, :, :]).sum(dim=-1)
    if config.scale_by_rank:
        distances = distances / kept
    # SEP carries weight 0 in the fixed head by label antisymmetry (SS137-3), so
    # its normalisation cannot affect the margin; kept comparable to rank 1.
    separation = (prototypes[1] - prototypes[0]).square().sum().sqrt()
    return distances, separation, kept
