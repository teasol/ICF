"""Shared evaluation statistics.

One implementation used by every evaluation entry point (`scripts/test.py`,
`evaluate_protocol.py`, `evaluate_synthetic.py`, `compare_predictions.py`,
`power_analysis.py`) so a metric cannot drift between them.

AUROC uses the rank (Mann-Whitney U) identity instead of counting positive vs
negative pairs. The pairwise form is O(n_pos * n_neg) per evaluation, which is
fine for the 87-donor ICI cohort but makes a bootstrap sweep over the ~1,600
synthetic validation predictions take hours. Ranking is O(n log n) and agrees
exactly, ties included, via average ranks.
"""
from __future__ import annotations

import torch


def auroc_rows(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """AUROC per row. Returns NaN for rows that are single-class."""
    squeeze = scores.ndim == 1
    if squeeze:
        scores = scores.unsqueeze(0)
        labels = labels.unsqueeze(0)
    scores = scores.to(torch.float64)
    count = scores.shape[-1]

    order = scores.argsort(dim=-1, stable=True)
    sorted_scores = scores.gather(-1, order)
    sorted_labels = labels.gather(-1, order).to(scores.dtype)

    positions = torch.arange(count, dtype=scores.dtype, device=scores.device)
    ranks = positions.expand_as(sorted_scores) + 1.0

    # Average the ranks inside each run of equal scores, so ties contribute
    # 0.5 exactly as the pairwise definition does. Done with cumulative sums
    # rather than a Python loop to stay fast at bootstrap scale.
    is_new = torch.ones_like(sorted_scores, dtype=torch.bool)
    is_new[..., 1:] = sorted_scores[..., 1:] != sorted_scores[..., :-1]
    group = is_new.cumsum(dim=-1) - 1
    group_count = torch.zeros_like(sorted_scores).scatter_add_(
        -1, group, torch.ones_like(sorted_scores)
    )
    group_rank_sum = torch.zeros_like(sorted_scores).scatter_add_(-1, group, ranks)
    ranks = (group_rank_sum / group_count.clamp_min(1)).gather(-1, group)

    n_positive = sorted_labels.sum(dim=-1)
    n_negative = count - n_positive
    rank_sum = (ranks * sorted_labels).sum(dim=-1)
    value = (rank_sum - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    ).clamp_min(1)
    value = torch.where(
        (n_positive > 0) & (n_negative > 0),
        value,
        torch.full_like(value, float("nan")),
    )
    return value.squeeze(0) if squeeze else value


def auroc(probability: torch.Tensor, target: torch.Tensor) -> float:
    return float(auroc_rows(probability.flatten(), target.flatten()).item())


def log_loss(probability: torch.Tensor, target: torch.Tensor) -> float:
    probability = probability.to(torch.float64)
    eps = torch.finfo(probability.dtype).eps
    clipped = probability.clamp(eps, 1.0 - eps)
    return float(
        -(
            target.to(clipped.dtype) * clipped.log()
            + (1.0 - target.to(clipped.dtype)) * (1.0 - clipped).log()
        )
        .mean()
        .item()
    )


def cluster_members(groups: torch.Tensor | None) -> list[torch.Tensor] | None:
    """Row indices per group, or None when every row is its own group."""
    if groups is None:
        return None
    return [
        torch.nonzero(groups == value, as_tuple=False).flatten()
        for value in torch.unique(groups).tolist()
    ]


def resample_index(
    count: int,
    members: list[torch.Tensor] | None,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one bootstrap resample, by group when groups are supplied."""
    if members is None:
        return torch.randint(0, count, (count,), generator=generator)
    picks = torch.randint(0, len(members), (len(members),), generator=generator)
    return torch.cat([members[p] for p in picks.tolist()])


def bootstrap_auroc_interval(
    probability: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor | None = None,
    samples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI for AUROC.

    Pass `groups` (e.g. one id per synthetic episode) to resample whole groups
    instead of individual rows. Correlated rows carry less information than
    their raw count, so ignoring the grouping reports an interval narrower
    than the data supports.
    """
    positive = int((target == 1).sum())
    if positive == 0 or positive == target.numel():
        return float("nan"), float("nan")

    generator = torch.Generator().manual_seed(seed)
    members = cluster_members(groups)
    count = target.numel()
    values: list[torch.Tensor] = []
    # Batch the resamples so AUROC is evaluated on a matrix rather than in a
    # Python loop; grouped draws can vary in length, so those go in chunks of
    # equal size.
    batch = 256
    pending: list[torch.Tensor] = []
    for _ in range(samples):
        index = resample_index(count, members, generator)
        if pending and index.numel() != pending[0].numel():
            values.append(_auroc_batch(probability, target, pending))
            pending = []
        pending.append(index)
        if len(pending) >= batch:
            values.append(_auroc_batch(probability, target, pending))
            pending = []
    if pending:
        values.append(_auroc_batch(probability, target, pending))

    spread = torch.cat(values)
    spread = spread[~torch.isnan(spread)]
    if spread.numel() == 0:
        return float("nan"), float("nan")
    return float(spread.quantile(0.025)), float(spread.quantile(0.975))


def _auroc_batch(
    probability: torch.Tensor,
    target: torch.Tensor,
    indices: list[torch.Tensor],
) -> torch.Tensor:
    index = torch.stack(indices)
    return auroc_rows(probability[index], target[index])
