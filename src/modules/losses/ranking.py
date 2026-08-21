"""Pairwise ranking loss functions for episode learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Rank positive queries above negative queries within an episode."""
    if logits.ndim != 2 or logits.shape[-1] != 2:
        raise ValueError("Pairwise ranking currently requires binary logits.")
    scores = logits[:, 1] - logits[:, 0]
    positive = scores[targets == 1]
    negative = scores[targets == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return logits.sum() * 0.0
    margins = positive[:, None] - negative[None, :]
    return F.softplus(-margins).mean()
