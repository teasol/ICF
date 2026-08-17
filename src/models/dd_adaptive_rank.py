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
    # SS147. How directions are CHOSEN, as opposed to how many pass a threshold.
    # The two criteria mean different things and SS146-2 measured that they
    # disagree (rank 0 is the |t| argmax on 1 of 15 folds):
    #
    #   |lambda|  the class dispersion gap is LARGE   (a mean difference)
    #   |t|       the class dispersion gap is CONSISTENT
    #             (that same difference divided by within-class scatter)
    #
    #   "eigenvalue"     top `rank_max` by |lambda| -- SS146's behaviour
    #   "lambda_plus_t"  rank 0 by |lambda|, PLUS the |t| argmax drawn from
    #                    `tstat_range`. r=2, two complementary criteria.
    #   "tstat"          the |t| argmax from `tstat_range` alone, r=1. Isolates
    #                    whether |t| picks a BETTER single direction than
    #                    |lambda| does, which "lambda_plus_t" cannot tell you.
    selection: str = "eigenvalue"
    # Half-open |lambda|-rank window the |t| argmax is drawn from, 0-indexed.
    # (1, 16) = the 2nd through 16th directions, excluding rank 0 so the |t| pick
    # cannot collapse onto the |lambda| pick.
    tstat_range: tuple[int, int] = (1, 16)


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


def _log_scalars(covariance, direction, eps):
    """log(s_b) where s_b = u^T C_b u -- the one number a direction induces."""
    return torch.einsum("d,bdk,k->b", direction, covariance, direction).clamp_min(eps).log()


def tstat_by_rank(context_covariance, context_labels, directions, config):
    """|t| of log(s_b) split by class, for every rank in `tstat_range`.

    ⚠️ Post-selection: these directions were chosen on these bags, so |t| is
    inflated and is NOT a p-value (SS146-2). Used here only to ARGMAX over
    candidates, which needs an ordering rather than a calibrated scale.
    """
    labels = context_labels.long()
    low, high = config.tstat_range
    high = min(high, directions.shape[-1])
    ranks, statistics = [], []
    for rank in range(low, high):
        scalar = _log_scalars(context_covariance, directions[:, rank], config.eps)
        ranks.append(rank)
        statistics.append(
            _welch_t(scalar[labels == 0], scalar[labels == 1], config.eps).abs()
        )
    return ranks, statistics


def select_ranks(context_covariance, context_labels, directions, config):
    """Which |lambda|-ranks to use, per `config.selection`."""
    if config.selection == "eigenvalue":
        # Threshold mode: rank 0 always, later ranks must pass |t| (SS146).
        labels = context_labels.long()
        chosen = [0]
        for rank in range(1, min(config.rank_max, directions.shape[-1])):
            scalar = _log_scalars(context_covariance, directions[:, rank], config.eps)
            statistic = _welch_t(scalar[labels == 0], scalar[labels == 1], config.eps)
            if float(statistic.abs()) >= config.t_threshold:
                chosen.append(rank)
        return chosen
    ranks, statistics = tstat_by_rank(
        context_covariance, context_labels, directions, config
    )
    if not ranks:
        return [0]
    best = ranks[int(torch.stack(statistics).argmax())]
    if config.selection == "tstat":
        return [best]
    if config.selection == "lambda_plus_t":
        return [0, best]
    raise ValueError(f"unknown selection {config.selection!r}")


def class_dispersions(context_covariance, context_labels, config):
    """sigma_c^2 of DD's standardised log-variance feature, per class (docs SS154).

    `_dd_distance_features` divides by these and never returns them, and they cannot
    be read back off its output: d_c already contains the division, so averaging
    d_c over class c gives exactly 1 by construction. Rather than re-deriving the
    feature by hand in a caller -- the mistake SS149-7 came from -- it is computed
    here, on the same code path `adaptive_dd_distance_features(rank_max=1)` uses to
    reproduce the lineage bit for bit.

    Needed for the log-determinant term of the Gaussian LLR:
        log p(f|1) - log p(f|0) = 1/2 (d0 - d1) + 1/2 log(sigma_0^2 / sigma_1^2)
    """
    labels = context_labels.long()
    directions, _ = dispersion_directions(context_covariance, context_labels, config)
    scalar = _log_scalars(context_covariance, directions[:, 0], config.eps)
    centre = scalar.mean()
    scale = (scalar - centre).square().mean().sqrt().clamp_min(config.eps)
    feature = (scalar - centre) / scale
    prototypes = torch.stack([feature[labels == c].mean() for c in range(2)])
    return torch.stack([
        (feature[labels == c] - prototypes[c]).square().mean().clamp_min(config.eps)
        for c in range(2)
    ])


def adaptive_dd_distance_features(
    context_covariance, context_labels, query_covariance, config
):
    """`_dd_distance_features` with an adaptive set of directions.

    Returns `(distances, separation, kept)` where `distances` is [queries, 2] as
    the rank-1 version is, so the fixed 12-feature head is untouched. `kept` is
    how many directions were used, so a null result can be told apart from a
    selector that never fired.
    """
    labels = context_labels.long()
    directions, _ = dispersion_directions(context_covariance, context_labels, config)
    chosen = select_ranks(context_covariance, context_labels, directions, config)

    context_features, query_features = [], []
    for rank in chosen:
        direction = directions[:, rank]
        context_scalar = _log_scalars(context_covariance, direction, config.eps)
        query_scalar = _log_scalars(query_covariance, direction, config.eps)
        # Context-only centring and scalar RMS, per direction, exactly as the
        # rank-1 path does. Each direction's log-variance has its own scale.
        centre = context_scalar.mean()
        scale = (context_scalar - centre).square().mean().sqrt().clamp_min(config.eps)
        context_features.append((context_scalar - centre) / scale)
        query_features.append((query_scalar - centre) / scale)
    kept = len(chosen)

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
