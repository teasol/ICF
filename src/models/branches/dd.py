"""DD branch: rank-1 dispersion direction from sketched covariances."""

from __future__ import annotations

import torch

from src.models.dd_adaptive_rank import ordered_typicality_margin


def dd_features(config, context_cov: torch.Tensor, labels: torch.Tensor, query_cov: torch.Tensor):
    context_cov = context_cov.float()
    query_cov = query_cov.float()
    labels = labels.long()
    means = torch.stack([context_cov[labels == c].mean(dim=0) for c in range(2)])
    delta = means[1] - means[0]
    pooled = context_cov.mean(dim=0)
    trace_scale = pooled.diagonal().mean().clamp_min(config.dd_eps)
    identity = torch.eye(config.sketch_dim, device=pooled.device, dtype=pooled.dtype)
    shrunk = (1.0 - config.dd_shrinkage) * pooled + config.dd_shrinkage * trace_scale * identity
    # eigh here is never differentiated -- docs SS100 / `_dd_direction`: the
    # backward carries 1/(lambda_i - lambda_j) and the direction is a hard
    # argmax. Training-free, so the constraint is free to honour.
    values, vectors = torch.linalg.eigh(shrunk)
    whitening = (vectors * values.clamp_min(config.dd_eps).rsqrt()[None, :]) @ vectors.T
    operator = whitening @ delta @ whitening
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    direction = whitening @ eigenvectors[:, eigenvalues.abs().argmax()]

    def log_variance(covariances):
        return torch.einsum("d,bdk,k->b", direction, covariances, direction).clamp_min(
            config.dd_eps
        ).log()

    context_feature = log_variance(context_cov)
    centre = context_feature.mean()
    scale = (context_feature - centre).square().mean().sqrt().clamp_min(config.dd_eps)
    context_feature = (context_feature - centre) / scale
    query_feature = (log_variance(query_cov) - centre) / scale
    prototypes = torch.stack([context_feature[labels == c].mean() for c in range(2)])
    dispersions = torch.stack([
        (context_feature[labels == c] - prototypes[c]).square().mean().clamp_min(config.dd_eps)
        for c in range(2)
    ])
    distances = (
        (query_feature[:, None] - prototypes[None, :]).square()
        / dispersions[None, :]
    )
    if config.dd_readout == "distance":
        # Logits, not distances: -d_c is the class-c score, so a large
        # distance is evidence AGAINST that class and the head weighs the
        # difference (dd1 - dd0) positively, like CV and CT (SS183).
        return -distances
    if config.dd_readout == "ordered_typicality":
        margin = ordered_typicality_margin(
            query_feature,
            prototypes,
            dispersions,
            config.dd_eps,
            config.dd_separation_floor,
        )
        # Logits, like CT: the pair is (-margin/2, +margin/2) so its
        # difference IS the class-1-positive margin, consumed by the head
        # at a positive weight (SS183).
        return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)
    raise ValueError(
        'dd_readout must be "distance" or "ordered_typicality", '
        f"got {config.dd_readout!r}"
    )
