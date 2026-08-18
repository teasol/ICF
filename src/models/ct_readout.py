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
import math
from typing import Literal, NamedTuple, Sequence

import torch

MODES = ("extreme", "prototype", "ridge")


def _fp32_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """GEMM without TF32 so the alternative distance is as close to fp32 as possible."""
    if not left.is_cuda:
        return left @ right
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return left @ right
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@dataclass(frozen=True)
class CTReadoutConfig:
    num_tokens: int = 16
    # SS159. `None` = use EVERY cell in the bag. 64 is v109's value and stays the
    # default so it remains the reproduction path.
    #
    # 64 evenly spaced cells is 1.6% of an 8,192-cell slide, and the abundance is an
    # average over that sample -- so its sampling error can exceed the class
    # difference it is meant to carry. This is the last of SS148-5's three suspects
    # (token generation was SS157, the distance metric SS149).
    cells_per_bag: int | None = 64
    # SS165. The cells that FIT the coordinate normalisation and k-means tokens
    # need not be the cells whose assignments are averaged into each bag's
    # abundance.  "match" preserves the historical coupled path bit-for-bit;
    # None uses every cell for abundance while keeping `cells_per_bag` for the
    # context-only dictionary.  This isolates estimation precision from the
    # cell-count weighting of the pooled k-means objective.
    abundance_cells_per_bag: int | None | Literal["match"] = "match"
    # SS165. "random" draws a reproducible per-bag subset instead of assuming
    # that storage order makes evenly spaced indices representative.  The seed is
    # mixed with the bag index, so bags do not all receive the same index pattern.
    sampling: Literal["even", "random"] = "even"
    sampling_seed: int = 0
    # SS168. The historical broadcast kernel is the reproduction path.  "gemm"
    # computes the same token-dependent squared distance with
    # ||c||^2 - 2 x.c; ||x||^2 is omitted because it is constant across tokens
    # for both argmin and softmax. This avoids materialising [cells,tokens,dims].
    distance_kernel: Literal["broadcast", "gemm"] = "broadcast"
    # SS168. Hierarchical PCA-initialised 2-means reads every context cell in
    # O(N D log K), avoiding the 30 x O(N K D) Lloyd sweep at large K.
    tokenizer: Literal["fps_lloyd", "hierarchical_2means"] = "fps_lloyd"
    bisect_iterations: int = 2
    bisect_power_iterations: int = 3
    # "segment" is bit-reproducible but sorts ~N cells at every tree update.
    # "atomic" uses fast CUDA index_add; it guarantees non-empty leaves but can
    # differ slightly between runs because floating-point atomic order is free.
    tree_reduction: Literal["segment", "atomic"] = "segment"
    temperature: float = 0.5
    eps: float = 1e-6
    ridge_lambda: float = 1.0
    # SS149. Measure cell-token distances in a PCA subspace instead of raw 1536-d.
    # None = today's behaviour (v107), unchanged.
    #
    # Squared Euclidean distance concentrates as dimension grows: the spread of
    # pairwise distances shrinks relative to their mean, so every cell ends up
    # roughly equidistant from every token and the softmax abundance flattens
    # toward uniform. SS148-4 measured the symptom -- per-token discriminative
    # |t| with a MEDIAN of 1.31 -- without being able to name the cause.
    #
    # The basis is supplied by the caller, not recomputed here: under v107 the
    # within-slide PCA of the fold's context cells already exists for the CV
    # branch, so CT can reuse it for free and is guaranteed to see the same
    # subspace. `pca_dim` slices its leading columns, which is exact because PCA
    # eigenvectors are ordered by descending eigenvalue (same argument as SS145).
    pca_dim: int | None = None
    # "standardise" rescales each retained component to unit context RMS, keeping
    # the per-coordinate convention the 1536-d path already uses. "raw" leaves the
    # eigenvalue scaling in place, so the top components dominate the distance.
    pca_scaling: str = "standardise"
    # SS157. Lloyd (k-means) iterations refining the farthest-point tokens.
    # 0 = farthest-point only, i.e. v108 unchanged.
    #
    # Farthest-point sampling maximises spread, so it puts tokens in LOW-density
    # regions -- outlier cells. Every ordinary cell is then far from every token and
    # the soft assignment flattens, which is SS148-4's symptom (per-token |t| median
    # 1.31). k-means puts centroids at density modes instead, which is what "cell
    # token = a cell population" was supposed to mean.
    #
    # Initialised FROM the farthest-point tokens rather than at random, for three
    # reasons: determinism is a v107/v108 invariant (seed std 0.00000), 0 iterations
    # reproduces today's behaviour bit for bit, and the iteration count then becomes
    # ONE knob interpolating between coverage (FPS) and density (k-means).
    #
    # ⚠️ The two failure modes are opposite. FPS over-represents rare/outlier cells;
    # k-means over-represents the dominant population and can spend every centroid
    # on stroma, losing rare-but-informative ones. Partial refinement sits between,
    # which is why the count is swept rather than set to convergence.
    kmeans_iterations: int = 0


class CTAbundance(NamedTuple):
    tokens: torch.Tensor          # [tokens, 1536] standardised cell coordinates
    context: torch.Tensor         # [context bags, tokens]
    query: torch.Tensor           # [query bags, tokens]


class CTMargins(NamedTuple):
    context: torch.Tensor         # [context bags] signed, positive favours class 1
    query: torch.Tensor           # [query bags]
    separation: torch.Tensor      # scalar, class-swap INVARIANT (head weight 0)


_CONFIG_CELL_LIMIT = object()


def sample_cells(
    bag: torch.Tensor,
    config: CTReadoutConfig,
    cells_per_bag: int | None | object = _CONFIG_CELL_LIMIT,
    sampling_seed: int | None = None,
) -> torch.Tensor:
    """Select a capped subset using the configured reproducible policy.

    `cells_per_bag=None` keeps every cell (docs SS159); the bag is already capped
    upstream by the encoder's `max_cells`, so this is not unbounded.
    """
    values = bag.float()
    if values.shape[0] == 0:
        raise ValueError("Every bag must contain at least one cell.")
    limit = config.cells_per_bag if cells_per_bag is _CONFIG_CELL_LIMIT else cells_per_bag
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("cells_per_bag must be a positive integer or None.")
    if limit is None or values.shape[0] <= limit:
        return values
    if config.sampling == "even":
        index = torch.linspace(
            0, values.shape[0] - 1, limit, device=values.device
        ).round().long()
    elif config.sampling == "random":
        generator = torch.Generator(device=values.device)
        generator.manual_seed(
            config.sampling_seed if sampling_seed is None else sampling_seed
        )
        # Sorting restores storage order AFTER drawing the subset.  This keeps
        # accumulation order stable and makes sampling the only changed variable.
        index = torch.randperm(
            values.shape[0], generator=generator, device=values.device
        )[:limit].sort().values
    else:
        raise ValueError(
            f"sampling must be 'even' or 'random', got {config.sampling!r}"
        )
    return values.index_select(0, index)


def prepare_cells(
    context_bags: Sequence[torch.Tensor],
    query_bags: Sequence[torch.Tensor],
    config: CTReadoutConfig,
    pca_basis: torch.Tensor | None = None,
    *,
    cells_per_bag: int | None | object = _CONFIG_CELL_LIMIT,
    normalisation: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_normalisation: bool = False,
):
    """Steps 1-2: sample cells, optionally project, then standardise on context.

    Returned so diagnostics can measure distances among the SAME cells the tokens
    are chosen from. Recomputing this in a diagnostic instead is how SS149's first
    contrast table came to measure token-to-token distance by mistake.
    """
    context = [
        sample_cells(
            bag, config, cells_per_bag, sampling_seed=config.sampling_seed + index
        )
        for index, bag in enumerate(context_bags)
    ]
    query = [
        sample_cells(
            bag,
            config,
            cells_per_bag,
            sampling_seed=config.sampling_seed + 1_000_000_007 + index,
        )
        for index, bag in enumerate(query_bags)
    ]
    projected = pca_basis is not None and config.pca_dim is not None
    if projected:
        # Project the RAW cells: the basis was built in unstandardised UNI2 space,
        # so that is the space it is orthonormal in. Per-coordinate standardisation
        # then happens below, on the retained components instead of on all 1,536.
        basis = pca_basis[:, : config.pca_dim].to(context[0].dtype)
        context = [bag @ basis for bag in context]
        query = [bag @ basis for bag in query]
    if normalisation is None:
        pooled = torch.cat(context, dim=0)
        centre = pooled.mean(dim=0, keepdim=True)
        if config.pca_scaling == "standardise" or not projected:
            scale = (pooled - centre).square().mean(dim=0, keepdim=True).sqrt()
        elif config.pca_scaling == "raw":
            # One shared scale: centres the cloud without flattening the eigenvalue
            # spectrum, so leading components keep their larger share of the distance.
            scale = (pooled - centre).square().mean().sqrt().reshape(1, 1)
        else:
            raise ValueError(
                f"pca_scaling must be 'standardise' or 'raw', got {config.pca_scaling!r}"
            )
        scale = scale.clamp_min(config.eps)
    else:
        centre, scale = normalisation
    result = ([(bag - centre) / scale for bag in context],
              [(bag - centre) / scale for bag in query])
    if return_normalisation:
        return (*result, (centre, scale))
    return result


def farthest_point_tokens(pooled: torch.Tensor, config: CTReadoutConfig) -> torch.Tensor:
    """Step 3. Deterministic, and labels are deliberately absent."""
    count = min(config.num_tokens, pooled.shape[0])
    first = (pooled - pooled.mean(dim=0, keepdim=True)).square().mean(dim=1).argmin()
    selected = [first]
    if config.distance_kernel == "gemm":
        pooled_norm = pooled.square().mean(dim=1)

        def distance_to(index):
            token = pooled[index]
            return (pooled_norm + token.square().mean()
                    - (2.0 / pooled.shape[1]) * _fp32_matmul(pooled, token))
    elif config.distance_kernel == "broadcast":
        def distance_to(index):
            return (pooled - pooled[index]).square().mean(dim=1)
    else:
        raise ValueError(
            "distance_kernel must be 'broadcast' or 'gemm', "
            f"got {config.distance_kernel!r}"
        )
    nearest = distance_to(first)
    for _ in range(1, count):
        index = nearest.argmax()
        selected.append(index)
        nearest = torch.minimum(nearest, distance_to(index))
    return pooled[torch.stack(selected)]


def hierarchical_2means_tokens(
    pooled: torch.Tensor, config: CTReadoutConfig
) -> torch.Tensor:
    """Deterministic full-cell PCA/2-means tree with exactly K non-empty leaves.

    Requested SS168 token counts are powers of two, so every leaf is split once
    per level. A short power iteration estimates each node's principal direction;
    two deterministic 2-means updates then follow. If a natural split would leave
    too few cells to complete the remaining tree, a stable median projection split
    supplies the cardinality guarantee without inventing an empty centroid.
    """
    target = min(config.num_tokens, pooled.shape[0])
    if target < 1 or target & (target - 1):
        raise ValueError(
            "hierarchical_2means requires a positive power-of-two token count "
            "not exceeding the number of pooled cells."
        )
    if config.bisect_iterations < 0 or config.bisect_power_iterations < 1:
        raise ValueError("bisect iterations must be non-negative and power iterations positive.")

    levels = int(math.log2(target))
    dimension = pooled.shape[1]
    initial = torch.full(
        (dimension,), 1.0 / math.sqrt(dimension),
        device=pooled.device, dtype=pooled.dtype,
    )
    labels = torch.zeros(pooled.shape[0], device=pooled.device, dtype=torch.long)

    def group_means(group_labels, groups):
        counts_long = torch.bincount(group_labels, minlength=groups)
        if config.tree_reduction == "atomic":
            sums = torch.zeros(
                groups, dimension, device=pooled.device, dtype=pooled.dtype
            ).index_add_(0, group_labels, pooled)
            means = sums / counts_long.to(pooled.dtype).clamp_min(1.0)[:, None]
            return means, counts_long.to(pooled.dtype), None
        if config.tree_reduction != "segment":
            raise ValueError(
                "tree_reduction must be 'segment' or 'atomic', "
                f"got {config.tree_reduction!r}"
            )
        order = torch.argsort(group_labels, stable=True)
        means = torch.segment_reduce(
            pooled.index_select(0, order), reduce="mean", lengths=counts_long
        )
        return means, counts_long.to(pooled.dtype), order

    def enforce_capacity(right, projection, parent_labels, counts, minimum_child):
        groups = counts.shape[0]
        right_counts = torch.bincount(
            parent_labels[right], minlength=groups
        )
        bad = ((right_counts < minimum_child)
               | ((counts.long() - right_counts) < minimum_child)).nonzero().flatten()
        # Continuous UNI2 features almost never enter this path. It exists to
        # guarantee K even for identical/pathological cells; only bad parents pay
        # for a stable sort and host synchronisation.
        for parent in bad.tolist():
            members = (parent_labels == parent).nonzero().flatten()
            order = torch.argsort(projection.index_select(0, members), stable=True)
            cut = members.shape[0] // 2
            right[members] = False
            right[members.index_select(0, order[cut:])] = True
        return right

    for level in range(levels):
        groups = 1 << level
        minimum_child = 1 << (levels - level - 1)
        means, counts, parent_order = group_means(labels, groups)
        centred = pooled - means.index_select(0, labels)
        directions = initial.expand(groups, -1).clone()
        for _ in range(config.bisect_power_iterations):
            projection = (
                centred * directions.index_select(0, labels)
            ).sum(dim=1)
            weighted = centred * projection[:, None]
            if config.tree_reduction == "atomic":
                covariance_times_direction = torch.zeros_like(directions).index_add_(
                    0, labels, weighted
                )
            else:
                covariance_times_direction = torch.segment_reduce(
                    weighted.index_select(0, parent_order),
                    reduce="sum", lengths=counts.long(),
                )
            norms = covariance_times_direction.square().sum(dim=1).sqrt()
            usable = norms > config.eps
            directions = torch.where(
                usable[:, None],
                covariance_times_direction / norms.clamp_min(config.eps)[:, None],
                directions,
            )

        projection = (
            centred * directions.index_select(0, labels)
        ).sum(dim=1)
        if config.tree_reduction == "atomic":
            projected_energy = torch.zeros(
                groups, device=pooled.device, dtype=pooled.dtype
            ).index_add_(0, labels, projection.square())
        else:
            projected_energy = torch.segment_reduce(
                projection.square().index_select(0, parent_order),
                reduce="sum", lengths=counts.long(),
            )
        scale = (projected_energy / counts.clamp_min(1.0)).sqrt().clamp_min(config.eps)
        left_centres = means - scale[:, None] * directions
        right_centres = means + scale[:, None] * directions
        right = projection > 0

        for _ in range(config.bisect_iterations):
            right = enforce_capacity(right, projection, labels, counts, minimum_child)
            child_labels = labels * 2 + right.long()
            child_centres, _, _ = group_means(child_labels, groups * 2)
            left_centres = child_centres.index_select(0, labels * 2)
            right_centres = child_centres.index_select(0, labels * 2 + 1)
            left_distance = (pooled - left_centres).square().mean(dim=1)
            right_distance = (pooled - right_centres).square().mean(dim=1)
            right = right_distance < left_distance

        right = enforce_capacity(right, projection, labels, counts, minimum_child)
        labels = labels * 2 + right.long()

    tokens, _, _ = group_means(labels, target)
    return tokens


# docs SS160. Element budget for the cell-to-token distance block. The 3-D
# broadcast allocates [chunk, tokens, dims], and at 128 tokens over a full-cell
# context (~1.6M cells) the unchunked form asked for 24.4 GiB and OOM'd on a shared
# GPU. Chunking over CELLS keeps every element's arithmetic identical -- it only
# splits the allocation -- so nothing that fit before can change value. 2^27
# elements is ~537 MB in fp32, and it leaves v109 (64 cells/bag, ~13k pooled cells,
# 16 tokens) inside a SINGLE chunk, hence bit-identical.
_DISTANCE_ELEMENT_BUDGET = 1 << 27


def _distance_rows(tokens: torch.Tensor, distance_kernel: str) -> int:
    elements_per_row = tokens.shape[0]
    if distance_kernel == "broadcast":
        elements_per_row *= tokens.shape[1]
    elif distance_kernel != "gemm":
        raise ValueError(
            "distance_kernel must be 'broadcast' or 'gemm', "
            f"got {distance_kernel!r}"
        )
    return max(1, _DISTANCE_ELEMENT_BUDGET // max(1, elements_per_row))


def _token_distance(pooled: torch.Tensor, tokens: torch.Tensor,
                    distance_kernel: str) -> torch.Tensor:
    """Token-dependent mean squared distance; sufficient for argmin/softmax."""
    if distance_kernel == "broadcast":
        return (pooled[:, None, :] - tokens[None]).square().mean(-1)
    if distance_kernel == "gemm":
        distance = _fp32_matmul(pooled, tokens.T)
        distance.mul_(-2.0 / pooled.shape[1])
        distance.add_(tokens.square().mean(dim=1).unsqueeze(0))
        return distance
    raise ValueError(
        "distance_kernel must be 'broadcast' or 'gemm', "
        f"got {distance_kernel!r}"
    )


def _assign(pooled: torch.Tensor, tokens: torch.Tensor,
            distance_kernel: str = "broadcast") -> torch.Tensor:
    """Nearest-token index per cell, chunked over cells (docs SS160)."""
    rows = _distance_rows(tokens, distance_kernel)
    if rows >= pooled.shape[0]:
        return _token_distance(pooled, tokens, distance_kernel).argmin(dim=1)
    parts = [
        _token_distance(pooled[start:start + rows], tokens, distance_kernel).argmin(dim=1)
        for start in range(0, pooled.shape[0], rows)
    ]
    return torch.cat(parts)


def lloyd_refine(pooled: torch.Tensor, tokens: torch.Tensor, iterations: int,
                 distance_kernel: str = "broadcast"):
    """Move tokens to their cluster means, `iterations` times (docs SS157).

    Deterministic throughout: assignment is a hard argmin and the update is a plain
    mean, so nothing here depends on a seed. A cluster that loses every member keeps
    its previous position -- the standard fix, and the only one that stays
    deterministic. Returns the tokens and the final cluster sizes, since how BALANCED
    the clusters are is the diagnostic that separates "k-means helped" from
    "k-means collapsed onto the dominant population".
    """
    counts = None
    for _ in range(iterations):
        assignment = _assign(pooled, tokens, distance_kernel)
        sums = torch.zeros_like(tokens).index_add_(0, assignment, pooled)
        counts = torch.zeros(
            tokens.shape[0], device=pooled.device, dtype=pooled.dtype
        ).index_add_(0, assignment, torch.ones_like(assignment, dtype=pooled.dtype))
        occupied = counts > 0
        tokens = torch.where(
            occupied[:, None], sums / counts.clamp_min(1.0)[:, None], tokens
        )
    if counts is None:
        assignment = _assign(pooled, tokens, distance_kernel)
        counts = torch.zeros(
            tokens.shape[0], device=pooled.device, dtype=pooled.dtype
        ).index_add_(0, assignment, torch.ones_like(assignment, dtype=pooled.dtype))
    return tokens, counts


def ct_abundance(
    context_bags: Sequence[torch.Tensor],
    query_bags: Sequence[torch.Tensor],
    config: CTReadoutConfig,
    pca_basis: torch.Tensor | None = None,
) -> CTAbundance:
    """Steps 1-5. Identical for every readout, and label-free by construction.

    With `pca_basis` and `config.pca_dim` set, cells are projected into that
    subspace BEFORE tokens are chosen, so both the farthest-point selection and
    the soft assignment measure distance in the reduced space (docs SS149). The
    basis must come from CONTEXT cells only; nothing here checks that, because the
    caller owns it -- v107 passes the within-slide PCA the CV branch already built.
    """
    token_context, token_query, normalisation = prepare_cells(
        context_bags, query_bags, config, pca_basis, return_normalisation=True
    )
    pooled = torch.cat(token_context, dim=0)
    if config.tokenizer == "fps_lloyd":
        tokens = farthest_point_tokens(pooled, config)
        if config.kmeans_iterations > 0:
            tokens, _ = lloyd_refine(
                pooled, tokens, config.kmeans_iterations, config.distance_kernel
            )
    elif config.tokenizer == "hierarchical_2means":
        tokens = hierarchical_2means_tokens(pooled, config)
    else:
        raise ValueError(
            "tokenizer must be 'fps_lloyd' or 'hierarchical_2means', "
            f"got {config.tokenizer!r}"
        )

    if config.abundance_cells_per_bag == "match":
        context, query = token_context, token_query
    else:
        abundance_limit = config.abundance_cells_per_bag
        if abundance_limit is not None and (
            not isinstance(abundance_limit, int) or abundance_limit < 1
        ):
            raise ValueError(
                "abundance_cells_per_bag must be 'match', a positive integer, or None."
            )
        context, query = prepare_cells(
            context_bags,
            query_bags,
            config,
            pca_basis,
            cells_per_bag=abundance_limit,
            normalisation=normalisation,
        )

    def abundance(bags):
        # Chunked for the same reason as `_assign`, and identically exact: the mean
        # over cells is accumulated as a weighted sum of per-chunk means.
        rows = _distance_rows(tokens, config.distance_kernel)
        outputs = []
        for bag in bags:
            if rows >= bag.shape[0]:
                outputs.append(
                    (-_token_distance(bag, tokens, config.distance_kernel)
                     / config.temperature).softmax(dim=-1).mean(dim=0)
                )
                continue
            total = torch.zeros(tokens.shape[0], device=bag.device, dtype=bag.dtype)
            for start in range(0, bag.shape[0], rows):
                block = bag[start:start + rows]
                total = total + (
                    -_token_distance(block, tokens, config.distance_kernel)
                    / config.temperature
                ).softmax(dim=-1).sum(dim=0)
            outputs.append(total / bag.shape[0])
        return torch.stack(outputs)

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
               calibrated=True, pca_basis=None):
    """One entry point: abundance once, then the requested readout.

    `calibrated` only affects the non-extreme modes; `extreme` IS the reference
    and is always returned untouched, so the baseline cannot move.
    """
    if mode not in READOUTS:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    abundance = ct_abundance(context_bags, query_bags, config, pca_basis)
    margins = READOUTS[mode](abundance, labels, config)
    if mode != "extreme" and calibrated:
        margins = calibrate(margins, readout_extreme(abundance, labels, config), config)
    return margins, abundance
