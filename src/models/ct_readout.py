"""CT readouts: is the bottleneck the tokens, or throwing away 14 of 16 dims?

WHAT CT DOES TODAY (docs SS140 step 5, `_ct_features` in both the lineage and
`training_free.py`). Per episode, with no labels until the last step:

  1. sample <= 64 cells per bag, evenly spaced (deterministic)
  2. standardise the 1,536 coordinates on CONTEXT cells only
  3. farthest-point sample 16 tokens from the pooled context cells
  4. soft-assign every cell to the tokens (softmax of -distance / temperature)
  5. average per bag -> a 16-d ABUNDANCE vector
  6. score each token by (mean_0 - mean_1) / standard_error
  7. keep the argmax and argmin token only, and emit q1 - q0

Step 7 builds a 16-dimensional representation and then reads two coordinates off
it. This module keeps steps 1-5 EXACTLY as they are -- one shared
`ct_abundance()` so no arm can accidentally differ in the representation -- and
varies only step 6-7:

  extreme    q1 - q0, today's readout. The baseline, unchanged.
  prototype  class prototypes over all 16 standardised dims; margin is the
             squared-distance difference, positive toward class 1.
  ridge      class-balanced ridge on all 16 dims; margin is logit1 - logit0.

WHY BOTH prototype AND ridge. They fail differently, which is the diagnostic.
`prototype` is a fixed isotropic geometry: it can only find a class difference
that lies along the line between the two centroids. `ridge` fits a signed
combination and so can use tokens that are individually uninformative but jointly
discriminative -- at the cost of estimating 16 coefficients from ~50-200 context
bags. If only ridge wins, the signal needs signed mixing; if only prototype wins,
ridge is overfitting the context.

⚠️ SCALE. The three margins have different natural scales, so feeding them to the
fixed head raw would compare CT's MAGNITUDE, not its quality. `calibrate()` maps
an alternative margin onto the extreme margin's CONTEXT mean and centred RMS, so
the head sees the same distribution it was calibrated against and the 0.286 CT
weight keeps its meaning. Query statistics are never used.

⚠️ LABEL ANTISYMMETRY holds for all three, and calibration preserves it: under a
class swap `score` negates so argmax/argmin exchange (extreme), the prototypes
exchange (prototype), the one-hot targets exchange (ridge). The context mean of
each margin therefore also negates, so the affine calibration commutes with the
swap. `tests/test_ct_readout.py` pins this -- it is what lets the fixed head's
three constants stay valid (SS137-3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch

MODES = ("extreme", "prototype", "ridge")


@dataclass(frozen=True)
class CTReadoutConfig:
    num_tokens: int = 16
    cells_per_bag: int = 64
    temperature: float = 0.5
    eps: float = 1e-6
    ridge_lambda: float = 1.0


class CTAbundance(NamedTuple):
    tokens: torch.Tensor          # [tokens, 1536] standardised cell coordinates
    context: torch.Tensor         # [context bags, tokens]
    query: torch.Tensor           # [query bags, tokens]


class CTMargins(NamedTuple):
    context: torch.Tensor         # [context bags] signed, positive favours class 1
    query: torch.Tensor           # [query bags]
    separation: torch.Tensor      # scalar, class-swap INVARIANT (head weight 0)


def sample_cells(bag: torch.Tensor, config: CTReadoutConfig) -> torch.Tensor:
    """Evenly spaced, never random -- the whole pipeline stays deterministic."""
    values = bag.float()
    if values.shape[0] == 0:
        raise ValueError("Every bag must contain at least one cell.")
    if values.shape[0] <= config.cells_per_bag:
        return values
    index = torch.linspace(
        0, values.shape[0] - 1, config.cells_per_bag, device=values.device
    ).round().long()
    return values.index_select(0, index)


def ct_abundance(
    context_bags: Sequence[torch.Tensor],
    query_bags: Sequence[torch.Tensor],
    config: CTReadoutConfig,
) -> CTAbundance:
    """Steps 1-5. Identical for every readout, and label-free by construction."""
    context = [sample_cells(bag, config) for bag in context_bags]
    query = [sample_cells(bag, config) for bag in query_bags]
    pooled = torch.cat(context, dim=0)
    centre = pooled.mean(dim=0, keepdim=True)
    scale = (pooled - centre).square().mean(dim=0, keepdim=True).sqrt().clamp_min(config.eps)
    context = [(bag - centre) / scale for bag in context]
    query = [(bag - centre) / scale for bag in query]
    pooled = torch.cat(context, dim=0)

    count = min(config.num_tokens, pooled.shape[0])
    first = (pooled - pooled.mean(dim=0, keepdim=True)).square().mean(dim=1).argmin()
    selected = [first]
    nearest = (pooled - pooled[first]).square().mean(dim=1)
    for _ in range(1, count):
        index = nearest.argmax()
        selected.append(index)
        nearest = torch.minimum(nearest, (pooled - pooled[index]).square().mean(dim=1))
    tokens = pooled[torch.stack(selected)]

    def abundance(bags):
        return torch.stack([
            (-(bag[:, None, :] - tokens[None]).square().mean(-1) / config.temperature)
            .softmax(dim=-1).mean(dim=0)
            for bag in bags
        ])

    return CTAbundance(tokens, abundance(context), abundance(query))


def discriminative_score(abundance: CTAbundance, labels: torch.Tensor, config):
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


def _standardise(abundance: CTAbundance, config):
    """Per-token centring and RMS from CONTEXT bags only."""
    centre = abundance.context.mean(dim=0)
    spread = (abundance.context - centre).square().mean(dim=0).sqrt().clamp_min(config.eps)
    return (abundance.context - centre) / spread, (abundance.query - centre) / spread


def readout_extreme(abundance, labels, config) -> CTMargins:
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


def readout_prototype(abundance, labels, config) -> CTMargins:
    """All 16 dims, class prototypes, squared-distance difference.

    margin = |a - p0|^2 - |a - p1|^2, so positive means closer to class 1. Note
    this is affine in `a` (the quadratic terms cancel), i.e. a nearest-centroid
    classifier with isotropic geometry -- it cannot weight tokens unequally,
    which is exactly what `readout_ridge` is here to test.
    """
    labels = labels.long()
    context, query = _standardise(abundance, config)
    prototypes = torch.stack([context[labels == c].mean(dim=0) for c in range(2)])

    def margin(features):
        to_zero = (features - prototypes[0]).square().sum(dim=-1)
        to_one = (features - prototypes[1]).square().sum(dim=-1)
        return to_zero - to_one

    separation = (prototypes[1] - prototypes[0]).square().sum().sqrt()
    return CTMargins(margin(context), margin(query), separation)


def ridge_coefficients(abundance, labels, config):
    """Class-balanced ridge in the PRIMAL: 16 dims, so a 16x16 solve is simplest.

    Same recipe as the CV branch (context-only standardisation, class-balanced
    weights, weighted centring for the intercept), so a difference against CV is
    attributable to the descriptor rather than the readout.
    """
    labels = labels.long()
    context, query = _standardise(abundance, config)
    targets = torch.nn.functional.one_hot(labels, 2).float()
    counts = torch.bincount(labels, minlength=2)
    if bool((counts == 0).any()):
        raise ValueError("Every class must occur in the context set.")
    # Class-balanced: without this the ridge tracks prevalence, and real tasks
    # run 0.178 to 0.780 positive (docs SS115-2).
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


def readout_ridge(abundance, labels, config) -> CTMargins:
    """All 16 dims through a class-balanced ridge; margin = logit1 - logit0."""
    beta, intercept, context, query = ridge_coefficients(abundance, labels, config)

    def margin(features):
        logits = features @ beta + intercept
        return logits[:, 1] - logits[:, 0]

    # Signed weight the margin actually applies to each token.
    separation = (beta[:, 1] - beta[:, 0]).abs().sum()
    return CTMargins(margin(context), margin(query), separation)


READOUTS = {
    "extreme": readout_extreme,
    "prototype": readout_prototype,
    "ridge": readout_ridge,
}


def calibrate(alternative: CTMargins, reference: CTMargins, config) -> CTMargins:
    """Put `alternative` on `reference`'s CONTEXT mean and centred RMS.

    Without this the full-model comparison measures how BIG a readout's output is
    rather than how good it is, since the head applies a fixed 0.286. Only
    context statistics are read, and the map is affine with a positive scale, so
    it changes neither the ranking within an arm nor label antisymmetry (the
    context mean negates under a class swap along with the margin itself).
    """
    centre = alternative.context.mean()
    spread = (alternative.context - centre).square().mean().sqrt().clamp_min(config.eps)
    target_centre = reference.context.mean()
    target_spread = (
        (reference.context - target_centre).square().mean().sqrt().clamp_min(config.eps)
    )
    factor = target_spread / spread

    def apply(values):
        return (values - centre) * factor + target_centre

    return CTMargins(apply(alternative.context), apply(alternative.query),
                     alternative.separation)


def ct_margins(context_bags, labels, query_bags, config, mode="extreme",
               calibrated=True):
    """One entry point: abundance once, then the requested readout.

    `calibrated` only affects the non-extreme modes; `extreme` IS the reference
    and is always returned untouched, so the baseline cannot move.
    """
    if mode not in READOUTS:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    abundance = ct_abundance(context_bags, query_bags, config)
    margins = READOUTS[mode](abundance, labels, config)
    if mode != "extreme" and calibrated:
        margins = calibrate(margins, readout_extreme(abundance, labels, config), config)
    return margins, abundance
