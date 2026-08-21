"""Episode-level discrimination and diagnostics metrics."""

from __future__ import annotations

import torch


def binary_query_diagnostics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Episode-level baselines and discrimination metrics for binary queries."""
    if logits.ndim != 2 or logits.shape[-1] != 2:
        return {}
    targets = targets.long()
    positive_fraction = targets.float().mean()
    majority_accuracy = torch.maximum(positive_fraction, 1.0 - positive_fraction)
    eps = torch.finfo(logits.float().dtype).eps
    prior = positive_fraction.clamp(eps, 1.0 - eps)
    empirical_prior_ce = -(
        positive_fraction * prior.log()
        + (1.0 - positive_fraction) * (1.0 - prior).log()
    )
    predictions = logits.argmax(dim=-1)
    positive = targets == 1
    negative = ~positive
    positive_recall = (predictions[positive] == 1).float().mean()
    negative_recall = (predictions[negative] == 0).float().mean()
    both_classes = positive.any() & negative.any()
    balanced_accuracy = (positive_recall + negative_recall) / 2
    scores = (logits[:, 1] - logits[:, 0]).float()
    pairwise = scores[positive][:, None] - scores[negative][None, :]
    auroc = (pairwise.gt(0).float() + 0.5 * pairwise.eq(0).float()).mean()
    zero = logits.float().sum() * 0
    return {
        "query_positive_fraction": positive_fraction,
        "majority_accuracy": majority_accuracy,
        "empirical_prior_ce": empirical_prior_ce,
        "positive_recall": torch.where(positive.any(), positive_recall, zero),
        "negative_recall": torch.where(negative.any(), negative_recall, zero),
        "balanced_accuracy": torch.where(both_classes, balanced_accuracy, zero),
        "auroc": torch.where(both_classes, auroc, zero),
        "positive_recall_valid": positive.any().float(),
        "negative_recall_valid": negative.any().float(),
        "binary_ranking_metrics_valid": both_classes.float(),
    }
