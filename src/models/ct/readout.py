"""Readout heads and margin predictors over CT abundance vectors."""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn.functional as F

from src.models.ct.abundance import ct_abundance
from src.models.ct.config import CTAbundance, CTMargins, CTReadoutConfig
from src.models.ct.tokenizers import _fp32_matmul


def discriminative_score(abundance: CTAbundance, labels: torch.Tensor, config: CTReadoutConfig):
    """Per-token (mean_0 - mean_1) / SE, the statistic step 6 ranks tokens by."""
    labels = labels.long()
    means, variances = [], []
    for class_index in range(2):
        members = abundance.context[labels == class_index]
        if members.numel() == 0:
            raise ValueError("Every class must occur in the context set.")
        means.append(members.mean(dim=0))
        variances.append((members - means[-1]).square().mean(dim=0))
    standard_error = (
        variances[0] / (labels == 0).sum().clamp_min(1)
        + variances[1] / (labels == 1).sum().clamp_min(1)
    ).sqrt().clamp_min(config.eps)
    return (means[0] - means[1]) / standard_error


def _standardise(abundance: CTAbundance, config: CTReadoutConfig):
    """Per-token centring and RMS from CONTEXT bags only."""
    centre = abundance.context.mean(dim=0)
    spread = (abundance.context - centre).square().mean(dim=0).sqrt().clamp_min(config.eps)
    return (abundance.context - centre) / spread, (abundance.query - centre) / spread


def readout_extreme(abundance: CTAbundance, labels: torch.Tensor, config: CTReadoutConfig) -> CTMargins:
    """Today's readout, kept bit-identical: two tokens, q1 - q0."""
    score = discriminative_score(abundance, labels, config)
    token0, token1 = score.argmax(), score.argmin()
    context, query = _standardise(abundance, config)
    separation = 0.5 * (score[token0].abs() + score[token1].abs())
    return CTMargins(
        context[:, token1] - context[:, token0],
        query[:, token1] - query[:, token0],
        separation,
    )


def readout_prototype(abundance: CTAbundance, labels: torch.Tensor, config: CTReadoutConfig) -> CTMargins:
    """All 16 dims, class prototypes, squared-distance difference."""
    labels = labels.long()
    context, query = _standardise(abundance, config)
    prototypes = torch.stack([context[labels == c].mean(dim=0) for c in range(2)])

    def margin(features):
        to_zero = (features - prototypes[0]).square().sum(dim=-1)
        to_one = (features - prototypes[1]).square().sum(dim=-1)
        return to_zero - to_one

    separation = (prototypes[1] - prototypes[0]).square().sum().sqrt()
    return CTMargins(margin(context), margin(query), separation)


def ridge_coefficients(abundance: CTAbundance, labels: torch.Tensor, config: CTReadoutConfig):
    """Class-balanced ridge in the PRIMAL: 16 dims, so a 16x16 solve is simplest."""
    labels = labels.long()
    context, query = _standardise(abundance, config)
    targets = torch.nn.functional.one_hot(labels, 2).float()
    counts = torch.bincount(labels, minlength=2)
    if bool((counts == 0).any()):
        raise ValueError("Every class must occur in the context set.")
    weight = counts.float().reciprocal()[labels]
    total = weight.sum().clamp_min(config.eps)
    feature_mean = (weight[:, None] * context).sum(0, keepdim=True) / total
    target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
    root = weight.sqrt()[:, None]
    design = (context - feature_mean) * root
    centred_targets = (targets - target_mean) * root
    dimension = context.shape[-1]
    identity = torch.eye(dimension, device=context.device, dtype=context.dtype)
    gram = design.T @ design + config.ridge_lambda * identity
    beta = torch.linalg.solve(gram, design.T @ centred_targets)
    intercept = target_mean - feature_mean @ beta
    return beta, intercept, context, query


def readout_ridge(abundance: CTAbundance, labels: torch.Tensor, config: CTReadoutConfig) -> CTMargins:
    """All 16 dims through a class-balanced ridge; margin = logit1 - logit0."""
    beta, intercept, context, query = ridge_coefficients(abundance, labels, config)

    def margin(features):
        logits = features @ beta + intercept
        return logits[:, 1] - logits[:, 0]

    separation = (beta[:, 1] - beta[:, 0]).abs().sum()
    return CTMargins(margin(context), margin(query), separation, beta)


def _kernel_matrix(left: torch.Tensor, right: torch.Tensor, config: CTReadoutConfig):
    """Kernel Gram matrix between rows of left and right."""
    if config.kernel == "linear":
        return left @ right.T
    squared = _fp32_matmul(left, right.T)
    if config.kernel == "rbf":
        sqnorm_left = (left * left).sum(dim=1, keepdim=True)
        sqnorm_right = (right * right).sum(dim=1, keepdim=True)
        distances = sqnorm_left - 2.0 * squared + sqnorm_right.T
        dims = left.shape[-1]
        gamma = config.kernel_gamma if config.kernel_gamma is not None else 1.0 / dims
        return torch.exp(-gamma * distances)
    if config.kernel == "poly":
        dims = left.shape[-1]
        gamma = config.kernel_gamma if config.kernel_gamma is not None else 1.0 / dims
        return (gamma * squared + config.kernel_coef0).pow(config.kernel_degree)
    raise ValueError(
        f"kernel must be 'linear', 'rbf', or 'poly', got {config.kernel!r}"
    )


def readout_kernel_ridge(abundance: CTAbundance, labels: torch.Tensor, config: CTReadoutConfig) -> CTMargins:
    """Class-balanced kernel ridge in the DUAL: an n x n solve."""
    labels = labels.long()
    context, query = _standardise(abundance, config)
    targets = torch.nn.functional.one_hot(labels, 2).float()
    counts = torch.bincount(labels, minlength=2)
    if bool((counts == 0).any()):
        raise ValueError("Every class must occur in the context set.")
    weight = counts.float().reciprocal()[labels]
    total = weight.sum().clamp_min(config.eps)
    root = weight.sqrt()

    k_context = _kernel_matrix(context, context, config)
    m_context = (weight[None, :] @ k_context).squeeze(0) / total
    mu2 = (weight[None, :] @ k_context @ weight[:, None]).squeeze() / (total * total)
    target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
    centred_targets = (targets - target_mean) * root[:, None]
    gram = root[:, None] * (
        k_context - m_context[:, None] - m_context[None, :] + mu2
    ) * root[None, :]
    dimension = gram.shape[0]
    identity = torch.eye(dimension, device=context.device, dtype=context.dtype)
    gram = gram + config.ridge_lambda * identity
    dual = torch.linalg.solve(gram, centred_targets)
    alpha = root[:, None] * dual
    intercept = target_mean - (m_context @ alpha)[None, :]

    def margin(features, k_features):
        m_features = (weight[None, :] @ k_features.T).squeeze(0) / total
        logits = k_features @ alpha - m_features[:, None] * alpha.sum(0, keepdim=True) + intercept
        return logits[:, 1] - logits[:, 0]

    k_query = _kernel_matrix(query, context, config)
    separation = (alpha[:, 1] - alpha[:, 0]).abs().sum()
    return CTMargins(margin(context, k_context), margin(query, k_query), separation, alpha)


def calibrate(alternative: CTMargins, reference: CTMargins, config: CTReadoutConfig) -> CTMargins:
    """Rescale and shift `alternative` to match `reference`."""
    scale = reference.context.std(unbiased=False).clamp_min(config.eps) / (
        alternative.context.std(unbiased=False).clamp_min(config.eps)
    )
    shift = reference.context.mean() - alternative.context.mean() * scale
    return CTMargins(
        alternative.context * scale + shift,
        alternative.query * scale + shift,
        alternative.separation,
        alternative.coefficients,
    )


def ct_margins(
    context_bags: Sequence[torch.Tensor],
    labels: torch.Tensor,
    query_bags: Sequence[torch.Tensor],
    config: CTReadoutConfig | None = None,
    mode: str = "extreme",
    pca_basis: torch.Tensor | None = None,
) -> tuple[CTMargins, CTAbundance]:
    """Two-token or all-token abundance readout over context and query bags."""
    if config is None:
        config = CTReadoutConfig()
    abundance = ct_abundance(context_bags, query_bags, config, pca_basis=pca_basis)
    if mode == "extreme":
        return readout_extreme(abundance, labels, config), abundance
    if mode == "prototype":
        return readout_prototype(abundance, labels, config), abundance
    if mode == "ridge":
        return readout_ridge(abundance, labels, config), abundance
    if mode in ("kernel_ridge", "kernel-ridge"):
        return readout_kernel_ridge(abundance, labels, config), abundance
    if mode == "calibrated":
        ref = readout_extreme(abundance, labels, config)
        alt = readout_ridge(abundance, labels, config)
        return calibrate(alt, ref, config), abundance
    raise ValueError(
        f"mode must be 'extreme', 'prototype', 'ridge', 'kernel_ridge', or 'calibrated', got {mode!r}"
    )
