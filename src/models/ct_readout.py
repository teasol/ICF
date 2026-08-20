"""CT readouts: is the bottleneck the tokens, or throwing away 14 of 16 dims?

WHAT THE SHARED CT PIPELINE DOES (docs SS140 step 5, `_ct_features` in both the
lineage and `training_free.py`). Per episode, with no labels until the last step:

  1. sample cells per bag (int cap, all cells, or a fraction of bag / episode size)
  2. standardise the 1,536 coordinates on CONTEXT cells only
  3. build configured tokens (fps+Lloyd, k-means++ + Lloyd, hierarchical 2-means, ...)
  4. soft-assign configured abundance cells to the tokens
  5. average per bag -> a K-dimensional ABUNDANCE vector
  6. score each token by (mean_0 - mean_1) / standard_error
  7. keep the argmax and argmin token only, and emit q1 - q0

The historical extreme readout builds a K-dimensional representation and then
reads two coordinates off it. This module routes every representation arm through
one shared `ct_abundance()` and varies only step 6-7 when comparing readouts:

  extreme    q1 - q0, today's readout. The baseline, unchanged.
  prototype  class prototypes over all 16 standardised dims; margin is the
             squared-distance difference, positive toward class 1.
  ridge      class-balanced ridge on all 16 dims; margin is logit1 - logit0.
  kernel_ridge  class-balanced KERNEL ridge (linear/rbf/poly) on all dims,
             solved in the dual; `kernel="linear"` reproduces `ridge`.

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

MODES = ("extreme", "prototype", "ridge", "kernel_ridge")


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
    # Size-aware alternative to a fixed cap. None keeps the historical
    # integer/`all` path bit-identical. A value in (0, 1] draws
    # round(fraction * reference) cells, clamped to [cells_min, n_i].
    # `cells_scale="own"` uses this bag's length (a 5k-cell LUAD slide
    # contributes more than a 2k-cell therapy slide). `"median"` uses the
    # CONTEXT-only median length so the episode's typical size sets one
    # shared budget; query bags never enter that statistic.
    cells_fraction: float | None = None
    cells_min: int = 1
    cells_scale: Literal["own", "median"] = "own"
    # SS165. The cells that FIT the coordinate normalisation and k-means tokens
    # need not be the cells whose assignments are averaged into each bag's
    # abundance.  "match" preserves the historical coupled path bit-for-bit;
    # None uses every cell for abundance while keeping `cells_per_bag` for the
    # context-only dictionary.  This isolates estimation precision from the
    # cell-count weighting of the pooled k-means objective. A float in (0, 1]
    # is the same size-aware policy as `cells_fraction`, applied only to the
    # abundance average.
    abundance_cells_per_bag: int | float | None | Literal["match"] = "match"
    # SS189. Pooling of the per-cell token assignment into the per-bag abundance
    # vector. "mean" is the historical default (bit-identical); "max" keeps, per
    # token, the single most similar cell; "topk" averages the most similar
    # `abundance_topk_fraction` of cells (floor `abundance_topk_min`), so it
    # interpolates between "max" (k=1) and "mean" (k=all). Max/topk are
    # non-linear in the cells, so they carry information the mean abundance
    # discards -- at the cost of outlier sensitivity.
    # "mean+topk" CONCATENATES the mean and top-k vectors into a 2K-dimensional
    # descriptor instead of replacing one with the other, so a ridge / kernel
    # ridge readout can use both jointly. This keeps the diagnostic 0-param: the
    # readout coefficients are still fit per episode from the context only.
    abundance_pooling: str = "mean"
    abundance_topk_fraction: float = 0.1
    abundance_topk_min: int = 1
    # Random is the active default: evenly spaced indices are biased whenever
    # storage order carries slide/location/batch structure. The fixed seed is
    # mixed with the bag index, so evaluation remains reproducible while bags do
    # not all receive the same index pattern. "even" remains available only for
    # replaying historical v107-v110 results.
    sampling: Literal["even", "random"] = "random"
    sampling_seed: int = 0
    # SS168. The historical broadcast kernel is the reproduction path.  "gemm"
    # computes the same token-dependent squared distance with
    # ||c||^2 - 2 x.c; ||x||^2 is omitted because it is constant across tokens
    # for both argmin and softmax. This avoids materialising [cells,tokens,dims].
    distance_kernel: Literal["broadcast", "gemm", "cosine"] = "broadcast"
    # SS168. Hierarchical PCA-initialised 2-means reads every context cell in
    # O(N D log K), avoiding the 30 x O(N K D) Lloyd sweep at large K.
    tokenizer: Literal[
        "fps_lloyd", "kmeans_plusplus", "spherical_kmeans",
        "hierarchical_2means", "hdbscan", "dbscan"
    ] = "fps_lloyd"
    bisect_iterations: int = 2
    bisect_power_iterations: int = 3
    # "segment" is bit-reproducible but sorts ~N cells at every tree update.
    # "atomic" uses fast CUDA index_add; it guarantees non-empty leaves but can
    # differ slightly between runs because floating-point atomic order is free.
    tree_reduction: Literal["segment", "atomic"] = "segment"
    # SS169. GPU HDBSCAN chooses K from the context-cell density hierarchy.
    # The relative floor makes the density scale comparable across folds while
    # the absolute floor avoids tiny, unstable cell groups on small contexts.
    hdbscan_min_cluster_size: int = 256
    hdbscan_min_cluster_fraction: float = 0.001
    hdbscan_min_samples: int = 32
    hdbscan_cluster_selection_method: Literal["eom", "leaf"] = "leaf"
    hdbscan_build_algo: Literal["brute_force", "nn_descent"] = "nn_descent"
    hdbscan_allow_single_cluster: bool = False
    # SS170. DBSCAN has no K, but needs a density radius. None chooses eps from
    # the knee of the context-only min_samples-neighbour distance curve.
    dbscan_eps: float | None = None
    dbscan_min_samples: int = 16
    temperature: float = 0.5
    eps: float = 1e-6
    ridge_lambda: float = 1.0
    # SS188 (G0). Kernel for `mode="kernel_ridge"`: tests whether the
    # abundance -> label relation has non-linear structure the linear ridge
    # misses. "linear" reproduces the primal ridge exactly (control); "rbf" and
    # "poly" (inhomogeneous) are the non-linear candidates. Kernels are evaluated
    # on the CONTEXT-standardised abundance, class-balanced like the primal
    # ridge, and solved in the dual (n x n, n = context bags).
    kernel: str = "rbf"
    kernel_gamma: float | None = None   # None -> 1 / dims (scikit-learn heuristic)
    kernel_degree: int = 2
    kernel_coef0: float = 1.0
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
    # Active efficient tokenizer. k-means++ avoids FPS's outlier-biased starting
    # points; convergence usually happens before this eight-pass ceiling.
    kmeans_max_iterations: int = 8
    kmeans_tolerance: float = 1e-4
    kmeans_seed: int = 0


class CTAbundance(NamedTuple):
    tokens: torch.Tensor          # [tokens, 1536] standardised cell coordinates
    context: torch.Tensor         # [context bags, tokens]
    query: torch.Tensor           # [query bags, tokens]


class CTMargins(NamedTuple):
    context: torch.Tensor         # [context bags] signed, positive favours class 1
    query: torch.Tensor           # [query bags]
    separation: torch.Tensor      # scalar, class-swap INVARIANT (head weight 0)


_CONFIG_CELL_LIMIT = object()


def parse_cell_budget(
    raw: str,
) -> tuple[int | None, float | None, str | None]:
    """Parse an `ICF_CT_CELLS` / abundance token.

    Returns `(cells_per_bag, cells_fraction, cells_scale)`. Historical values
    stay exact: ``all`` -> ``(None, None, None)``, ``64`` -> ``(64, None, None)``.
    Size-aware forms: ``0.125``, ``frac:0.125``, ``own:0.125``, ``median:0.125``.
    """
    text = raw.strip().lower()
    if text == "all":
        return None, None, None
    scale = None
    payload = text
    for prefix, named_scale in (
        ("fraction:", None),
        ("frac:", None),
        ("own:", "own"),
        ("median:", "median"),
        ("scale:", "own"),
    ):
        if text.startswith(prefix):
            payload = text[len(prefix):]
            scale = named_scale
            break
    if payload == "" or payload == "match":
        raise ValueError(f"unrecognised cell budget {raw!r}")
    if any(marker in payload for marker in (".", "e", "E")):
        try:
            fraction = float(payload)
        except ValueError as error:
            raise ValueError(f"unrecognised cell budget {raw!r}") from error
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"cell fraction must be in (0, 1], got {raw!r}"
            )
        return None, fraction, scale
    try:
        count = int(payload)
    except ValueError as error:
        raise ValueError(f"unrecognised cell budget {raw!r}") from error
    if count < 1:
        raise ValueError("cells_per_bag must be a positive integer or None.")
    return count, None, None


def typical_bag_size(bags: Sequence[torch.Tensor]) -> float:
    """Median cell count. Caller must pass CONTEXT bags only."""
    if not bags:
        raise ValueError("Need at least one context bag to set the sampling scale.")
    lengths = sorted(int(bag.shape[0]) for bag in bags)
    middle = len(lengths) // 2
    if len(lengths) % 2:
        return float(lengths[middle])
    return 0.5 * (lengths[middle - 1] + lengths[middle])


def _uses_fraction(
    config: CTReadoutConfig,
    cells_per_bag: int | float | None | object,
) -> bool:
    if cells_per_bag is _CONFIG_CELL_LIMIT:
        return config.cells_fraction is not None
    return isinstance(cells_per_bag, float)


def resolve_cells_per_bag(
    bag_size: int,
    config: CTReadoutConfig,
    cells_per_bag: int | float | None | object = _CONFIG_CELL_LIMIT,
    typical_size: float | None = None,
) -> int | None:
    """How many cells to keep from a bag of `bag_size`. None keeps every cell."""
    if bag_size < 1:
        raise ValueError("Every bag must contain at least one cell.")
    if config.cells_min < 1:
        raise ValueError("cells_min must be a positive integer.")
    if cells_per_bag is _CONFIG_CELL_LIMIT:
        limit: int | float | None = (
            config.cells_fraction
            if config.cells_fraction is not None
            else config.cells_per_bag
        )
        scale = config.cells_scale
    else:
        limit = cells_per_bag
        scale = config.cells_scale
    if isinstance(limit, float):
        if not 0.0 < limit <= 1.0:
            raise ValueError("cells_fraction must be in (0, 1].")
        if scale == "median":
            if typical_size is None:
                raise ValueError(
                    "median sampling needs a context-only typical bag size."
                )
            reference = float(typical_size)
        elif scale == "own":
            reference = float(bag_size)
        else:
            raise ValueError(
                f"cells_scale must be 'own' or 'median', got {scale!r}"
            )
        target = int(math.floor(limit * reference + 0.5))
        return min(bag_size, max(config.cells_min, target))
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("cells_per_bag must be a positive integer or None.")
    return limit


def _project_sampled_bag(bag: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project one sampled bag onto `basis`, matching the basis device.

    Official eval keeps raw 1536-d tiles on CPU. The sampled subset is
    small (a fraction of the bag, or already reduced) so uploading it for
    the GEMM is cheap, and the reduced cells stay next to the tokenizer.
    """
    if bag.device == basis.device and bag.dtype == basis.dtype:
        return bag @ basis
    return bag.to(device=basis.device, dtype=basis.dtype) @ basis


def sample_cells(
    bag: torch.Tensor,
    config: CTReadoutConfig,
    cells_per_bag: int | float | None | object = _CONFIG_CELL_LIMIT,
    sampling_seed: int | None = None,
    typical_size: float | None = None,
) -> torch.Tensor:
    """Select a capped subset using the configured reproducible policy.

    `cells_per_bag=None` keeps every cell (docs SS159); the bag is already capped
    upstream by the encoder's `max_cells`, so this is not unbounded. A fraction
    (config `cells_fraction`, or a float override) scales with bag / episode size
    instead of repeating a global 512-style cap.
    """
    values = bag.float()
    if values.shape[0] == 0:
        raise ValueError("Every bag must contain at least one cell.")
    limit = resolve_cells_per_bag(
        values.shape[0], config, cells_per_bag, typical_size
    )
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
    cells_per_bag: int | float | None | object = _CONFIG_CELL_LIMIT,
    normalisation: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_normalisation: bool = False,
):
    """Steps 1-2: sample cells, optionally project, then standardise on context.

    Returned so diagnostics can measure distances among the SAME cells the tokens
    are chosen from. Recomputing this in a diagnostic instead is how SS149's first
    contrast table came to measure token-to-token distance by mistake.

    Size-aware sampling (`cells_fraction` / a float override) may use the
    CONTEXT-only median bag length. Query bags never enter that statistic.
    """
    typical_size = None
    if _uses_fraction(config, cells_per_bag) and config.cells_scale == "median":
        typical_size = typical_bag_size(context_bags)
    context = [
        sample_cells(
            bag,
            config,
            cells_per_bag,
            sampling_seed=config.sampling_seed + index,
            typical_size=typical_size,
        )
        for index, bag in enumerate(context_bags)
    ]
    query = [
        sample_cells(
            bag,
            config,
            cells_per_bag,
            sampling_seed=config.sampling_seed + 1_000_000_007 + index,
            typical_size=typical_size,
        )
        for index, bag in enumerate(query_bags)
    ]
    projected = pca_basis is not None and config.pca_dim is not None
    if projected:
        # Project the RAW cells: the basis was built in unstandardised UNI2 space,
        # so that is the space it is orthonormal in. Per-coordinate standardisation
        # then happens below, on the retained components instead of on all 1,536.
        # Official eval keeps the 1536-d tiles on CPU; only this sampled bag
        # is uploaded, the GEMM runs next to the basis, and the reduced cells
        # stay on that device for the tokenizer.
        basis = pca_basis[:, : config.pca_dim].to(dtype=context[0].dtype)
        context = [_project_sampled_bag(bag, basis) for bag in context]
        query = [_project_sampled_bag(bag, basis) for bag in query]
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


def kmeans_plusplus_tokens(
    pooled: torch.Tensor,
    config: CTReadoutConfig,
    distance_kernel: str | None = None,
) -> torch.Tensor:
    """Seed K centroids with reproducible D-squared sampling.

    Unlike FPS, D-squared sampling still favours uncovered regions without
    deterministically spending every early centroid on the most extreme outlier.
    A tiny fallback mass makes degenerate clouds sample uniformly among the
    remaining cells, guaranteeing K distinct source indices without a host sync.
    """
    count = min(config.num_tokens, pooled.shape[0])
    if count < 1:
        raise ValueError("k-means++ requires at least one pooled cell.")
    generator = torch.Generator(device=pooled.device)
    generator.manual_seed(config.kmeans_seed)
    first = torch.randint(
        pooled.shape[0], (1,), generator=generator, device=pooled.device
    ).squeeze(0)
    selected = [first]
    chosen = torch.zeros(pooled.shape[0], device=pooled.device, dtype=torch.bool)
    chosen[first] = True
    kernel = config.distance_kernel if distance_kernel is None else distance_kernel
    nearest = _token_distance(
        pooled, pooled[first].unsqueeze(0), kernel
    ).squeeze(1).clamp_min_(0)

    for _ in range(1, count):
        weights = nearest.masked_fill(chosen, 0)
        # Keeping selection on-device matters: reading `weights.sum()` in Python
        # would force one CUDA synchronisation per centroid. The epsilon mass is
        # negligible for a non-degenerate D-squared distribution and becomes a
        # reproducible uniform fallback when all remaining distances are zero.
        weights = weights + (~chosen).to(weights.dtype) * config.eps
        index = torch.multinomial(
            weights, 1, replacement=False, generator=generator
        ).squeeze(0)
        selected.append(index)
        chosen[index] = True
        distance = _token_distance(
            pooled, pooled[index].unsqueeze(0), kernel
        ).squeeze(1).clamp_min_(0)
        nearest = torch.minimum(nearest, distance)
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


def hdbscan_tokens(pooled: torch.Tensor, config: CTReadoutConfig) -> torch.Tensor:
    """Fit GPU HDBSCAN on every context cell and return its stable centroids.

    HDBSCAN determines the number of clusters from its condensed density tree;
    ``num_tokens`` is deliberately ignored. Cluster membership probabilities
    weight the centroids, so boundary points contribute less. Noise is excluded
    from fitting the centroids but still participates in the later full-cell
    soft-abundance calculation against those centroids.
    """
    if not pooled.is_cuda:
        raise RuntimeError("The full-cell HDBSCAN tokenizer requires a CUDA tensor.")
    if config.hdbscan_min_cluster_size < 2:
        raise ValueError("hdbscan_min_cluster_size must be at least 2.")
    if not 0.0 <= config.hdbscan_min_cluster_fraction <= 1.0:
        raise ValueError("hdbscan_min_cluster_fraction must be in [0, 1].")
    if config.hdbscan_min_samples < 1:
        raise ValueError("hdbscan_min_samples must be positive.")

    try:
        import cupy as cp  # noqa: PLC0415
        from cuml.cluster import HDBSCAN  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "HDBSCAN tokenizer needs RAPIDS; install requirements-hdbscan.txt "
            "into the BagPFN environment."
        ) from error

    cells = pooled.shape[0]
    relative_floor = math.ceil(config.hdbscan_min_cluster_fraction * cells)
    min_cluster_size = min(
        cells, max(config.hdbscan_min_cluster_size, relative_floor)
    )
    min_samples = min(config.hdbscan_min_samples, min_cluster_size)
    if config.hdbscan_build_algo == "nn_descent" and min_samples >= 64:
        raise ValueError(
            "NN-descent uses graph degree 64, so hdbscan_min_samples must be < 64."
        )

    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=config.hdbscan_cluster_selection_method,
        allow_single_cluster=config.hdbscan_allow_single_cluster,
        build_algo=config.hdbscan_build_algo,
        build_kwds=(
            {
                "nnd_graph_degree": 64,
                "nnd_intermediate_graph_degree": 128,
                "nnd_max_iterations": 20,
                "nnd_termination_threshold": 1e-4,
            }
            if config.hdbscan_build_algo == "nn_descent" else None
        ),
        output_type="cupy",
        prediction_data=False,
        gen_min_span_tree=False,
    )
    values = cp.from_dlpack(pooled.detach().contiguous())
    labels = torch.from_dlpack(model.fit_predict(values)).clone()
    probabilities = torch.from_dlpack(model.probabilities_).to(pooled.dtype).clone()
    valid = labels >= 0
    noise_fraction = (~valid).float().mean().item()

    if not bool(valid.any()):
        print(
            f"ICF_CT_HDBSCAN cells={cells} min_cluster_size={min_cluster_size} "
            "clusters=1 noise=1.000000 fallback=global_mean",
            flush=True,
        )
        return pooled.mean(dim=0, keepdim=True)

    cluster_ids, inverse = torch.unique(labels[valid], sorted=True, return_inverse=True)
    weights = probabilities[valid].clamp_min(config.eps)
    weighted_sums = torch.zeros(
        cluster_ids.numel(), pooled.shape[1], device=pooled.device, dtype=pooled.dtype
    ).index_add_(0, inverse, pooled[valid] * weights[:, None])
    weight_sums = torch.zeros(
        cluster_ids.numel(), device=pooled.device, dtype=pooled.dtype
    ).index_add_(0, inverse, weights)
    tokens = weighted_sums / weight_sums.clamp_min(config.eps)[:, None]
    print(
        f"ICF_CT_HDBSCAN cells={cells} min_cluster_size={min_cluster_size} "
        f"min_samples={min_samples} clusters={tokens.shape[0]} "
        f"noise={noise_fraction:.6f} build={config.hdbscan_build_algo} "
        f"selection={config.hdbscan_cluster_selection_method}",
        flush=True,
    )
    return tokens


def dbscan_tokens(pooled: torch.Tensor, config: CTReadoutConfig) -> torch.Tensor:
    """Fit GPU DBSCAN with label-free adaptive eps and return hard centroids."""
    if not pooled.is_cuda:
        raise RuntimeError("The DBSCAN tokenizer requires a CUDA tensor.")
    if config.dbscan_min_samples < 2:
        raise ValueError("dbscan_min_samples must be at least 2.")
    if config.dbscan_eps is not None and config.dbscan_eps <= 0:
        raise ValueError("dbscan_eps must be positive or None for adaptive eps.")

    try:
        import cupy as cp  # noqa: PLC0415
        from cuml.cluster import DBSCAN  # noqa: PLC0415
        from cuml.neighbors import NearestNeighbors  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "DBSCAN tokenizer needs RAPIDS; install requirements-hdbscan.txt "
            "into the BagPFN environment."
        ) from error

    cells = pooled.shape[0]
    min_samples = min(config.dbscan_min_samples, cells)
    values = cp.from_dlpack(pooled.detach().contiguous())
    knee_quantile = None
    if config.dbscan_eps is None:
        distances, _ = NearestNeighbors(
            n_neighbors=min_samples, metric="euclidean", output_type="cupy"
        ).fit(values).kneighbors(values)
        curve = cp.sort(distances[:, -1])
        spread = curve[-1] - curve[0]
        if float(spread.item()) <= config.eps:
            eps = max(float(curve[-1].item()), config.eps)
            knee_index = cells - 1
        else:
            y = (curve - curve[0]) / spread
            x = cp.linspace(0.0, 1.0, cells, dtype=curve.dtype)
            # For the conventional ascending convex k-distance plot, the elbow
            # maximises the vertical gap between the endpoint chord and curve.
            knee_index = int(cp.argmax(x - y).item())
            eps = max(float(curve[knee_index].item()), config.eps)
        knee_quantile = knee_index / max(1, cells - 1)
    else:
        eps = config.dbscan_eps

    labels = torch.from_dlpack(DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="brute",
        calc_core_sample_indices=False,
        output_type="cupy",
    ).fit_predict(values)).clone()
    valid = labels >= 0
    noise_fraction = (~valid).float().mean().item()
    eps_source = (
        f"knee_quantile={knee_quantile:.6f}" if knee_quantile is not None
        else "eps_source=fixed"
    )
    if not bool(valid.any()):
        print(
            f"ICF_CT_DBSCAN cells={cells} eps={eps:.6g} {eps_source} "
            "clusters=1 noise=1.000000 fallback=global_mean",
            flush=True,
        )
        return pooled.mean(dim=0, keepdim=True)

    cluster_ids, inverse = torch.unique(labels[valid], sorted=True, return_inverse=True)
    sums = torch.zeros(
        cluster_ids.numel(), pooled.shape[1], device=pooled.device, dtype=pooled.dtype
    ).index_add_(0, inverse, pooled[valid])
    counts = torch.bincount(inverse, minlength=cluster_ids.numel()).to(pooled.dtype)
    tokens = sums / counts.clamp_min(1.0)[:, None]
    print(
        f"ICF_CT_DBSCAN cells={cells} eps={eps:.6g} {eps_source} "
        f"min_samples={min_samples} clusters={tokens.shape[0]} "
        f"noise={noise_fraction:.6f}",
        flush=True,
    )
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
    elif distance_kernel not in ("gemm", "cosine"):
        raise ValueError(
            "distance_kernel must be 'broadcast', 'gemm', or 'cosine', "
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
    if distance_kernel == "cosine":
        # Spherical k-means normalises both operands before reaching this path.
        # Clamp only round-off excursions so distance remains non-negative.
        return (1.0 - _fp32_matmul(pooled, tokens.T)).clamp_min_(0)
    raise ValueError(
        "distance_kernel must be 'broadcast', 'gemm', or 'cosine', "
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


def _assigned_error(
    pooled: torch.Tensor,
    tokens: torch.Tensor,
    assignment: torch.Tensor,
    distance_kernel: str,
) -> torch.Tensor:
    """Squared error to each cell's assigned token without a full N x K matrix."""
    rows = _distance_rows(tokens, distance_kernel)
    outputs = []
    for start in range(0, pooled.shape[0], rows):
        stop = min(start + rows, pooled.shape[0])
        distances = _token_distance(pooled[start:stop], tokens, distance_kernel)
        outputs.append(distances.gather(1, assignment[start:stop, None]).squeeze(1))
    return torch.cat(outputs)


def lloyd_refine(
    pooled: torch.Tensor,
    tokens: torch.Tensor,
    iterations: int,
    distance_kernel: str = "broadcast",
    *,
    tolerance: float = 0.0,
    recover_empty: bool = False,
    normalise_centroids: bool = False,
):
    """Move tokens to their cluster means, `iterations` times (docs SS157).

    Deterministic after initialisation. The efficient active path restores empty
    clusters from the highest-error cells and stops when the largest centroid RMS
    movement falls below `tolerance`. Historical callers retain the old behaviour
    through the defaults (`recover_empty=False`, `tolerance=0`).
    """
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    counts = None
    for _ in range(iterations):
        assignment = _assign(pooled, tokens, distance_kernel)
        sums = torch.zeros_like(tokens).index_add_(0, assignment, pooled)
        counts = torch.zeros(
            tokens.shape[0], device=pooled.device, dtype=pooled.dtype
        ).index_add_(0, assignment, torch.ones_like(assignment, dtype=pooled.dtype))
        occupied = counts > 0
        updated = torch.where(
            occupied[:, None], sums / counts.clamp_min(1.0)[:, None], tokens
        )
        if recover_empty and not bool(occupied.all()):
            errors = _assigned_error(pooled, tokens, assignment, distance_kernel)
            empty = (~occupied).nonzero().flatten()
            donor = counts.argmax()
            donor_cells = (assignment == donor).nonzero().flatten()
            donor_order = torch.argsort(
                errors.index_select(0, donor_cells), descending=True, stable=True
            )
            candidates = donor_cells.index_select(0, donor_order)
            # Pathological K ~= N can leave the largest cluster too small. Keep
            # the normal policy exact, with a deterministic global fallback.
            if candidates.numel() < empty.numel():
                candidates = torch.argsort(errors, descending=True, stable=True)
            updated[empty] = pooled[candidates[: empty.numel()]]
        if normalise_centroids:
            norms = updated.square().sum(dim=1, keepdim=True).sqrt()
            updated = updated / norms.clamp_min(1e-12)
        movement = (updated - tokens).square().mean(dim=1).sqrt().max()
        tokens = updated
        if tolerance > 0 and float(movement) <= tolerance:
            break
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
    spherical = config.tokenizer == "spherical_kmeans"
    distance_kernel = "cosine" if spherical else config.distance_kernel
    if spherical:
        def unit_normalise(bag):
            norms = bag.square().sum(dim=1, keepdim=True).sqrt()
            return bag / norms.clamp_min(config.eps)

        token_context = [unit_normalise(bag) for bag in token_context]
        token_query = [unit_normalise(bag) for bag in token_query]
    pooled = torch.cat(token_context, dim=0)
    if config.tokenizer == "fps_lloyd":
        tokens = farthest_point_tokens(pooled, config)
        if config.kmeans_iterations > 0:
            tokens, _ = lloyd_refine(
                pooled, tokens, config.kmeans_iterations, config.distance_kernel
            )
    elif config.tokenizer == "kmeans_plusplus":
        tokens = kmeans_plusplus_tokens(pooled, config)
        tokens, _ = lloyd_refine(
            pooled,
            tokens,
            config.kmeans_max_iterations,
            config.distance_kernel,
            tolerance=config.kmeans_tolerance,
            recover_empty=True,
        )
    elif config.tokenizer == "spherical_kmeans":
        tokens = kmeans_plusplus_tokens(pooled, config, distance_kernel)
        tokens, _ = lloyd_refine(
            pooled,
            tokens,
            config.kmeans_max_iterations,
            distance_kernel,
            tolerance=config.kmeans_tolerance,
            recover_empty=True,
            normalise_centroids=True,
        )
    elif config.tokenizer == "hierarchical_2means":
        tokens = hierarchical_2means_tokens(pooled, config)
    elif config.tokenizer == "hdbscan":
        tokens = hdbscan_tokens(pooled, config)
    elif config.tokenizer == "dbscan":
        tokens = dbscan_tokens(pooled, config)
    else:
        raise ValueError(
            "tokenizer must be 'fps_lloyd', 'kmeans_plusplus', 'spherical_kmeans', "
            "'hierarchical_2means', 'hdbscan', or 'dbscan', "
            f"got {config.tokenizer!r}"
        )

    if config.abundance_cells_per_bag == "match":
        context, query = token_context, token_query
    else:
        abundance_limit = config.abundance_cells_per_bag
        if abundance_limit is not None and not (
            isinstance(abundance_limit, int) and abundance_limit >= 1
            or isinstance(abundance_limit, float) and 0.0 < abundance_limit <= 1.0
        ):
            raise ValueError(
                "abundance_cells_per_bag must be 'match', a positive integer, "
                "a fraction in (0, 1], or None."
            )
        context, query = prepare_cells(
            context_bags,
            query_bags,
            config,
            pca_basis,
            cells_per_bag=abundance_limit,
            normalisation=normalisation,
        )
        if spherical:
            context = [unit_normalise(bag) for bag in context]
            query = [unit_normalise(bag) for bag in query]

    def abundance(bags):
        # Chunked for the same reason as `_assign`, and identically exact: the mean
        # over cells is accumulated as a weighted sum of per-chunk means. `max` /
        # `topk` materialise the per-cell assignment (gemm distance is
        # [cells, tokens], so even a 35k-cell LUAD bag is ~35 MB, transient).
        rows = _distance_rows(tokens, distance_kernel)
        pooling = config.abundance_pooling
        outputs = []

        def assignment(bag):
            # [n_cells, K] softmax token-assignment, chunked.
            if rows >= bag.shape[0]:
                return (
                    -_token_distance(bag, tokens, distance_kernel)
                    / config.temperature
                ).softmax(dim=-1)
            return torch.cat(
                [
                    (
                        -_token_distance(bag[start:start + rows], tokens, distance_kernel)
                        / config.temperature
                    ).softmax(dim=-1)
                    for start in range(0, bag.shape[0], rows)
                ],
                dim=0,
            )

        for bag in bags:
            if pooling == "mean":
                if rows >= bag.shape[0]:
                    outputs.append(
                        (-_token_distance(bag, tokens, distance_kernel)
                         / config.temperature).softmax(dim=-1).mean(dim=0)
                    )
                    continue
                total = torch.zeros(tokens.shape[0], device=bag.device, dtype=bag.dtype)
                for start in range(0, bag.shape[0], rows):
                    block = bag[start:start + rows]
                    total = total + (
                        -_token_distance(block, tokens, distance_kernel)
                        / config.temperature
                    ).softmax(dim=-1).sum(dim=0)
                outputs.append(total / bag.shape[0])
            elif pooling == "max":
                outputs.append(assignment(bag).amax(dim=0))
            elif pooling == "topk":
                count = int(round(config.abundance_topk_fraction * bag.shape[0]))
                count = min(max(count, config.abundance_topk_min), bag.shape[0])
                # Per-token top-k: each token averages its k most similar cells.
                # The mean of the k largest values is tie-invariant, so this stays
                # deterministic regardless of torch.topk's tie-breaking.
                top = torch.topk(assignment(bag).T, count, dim=1).values  # [K, k]
                outputs.append(top.mean(dim=1))
            elif pooling == "mean+topk":
                # Keep the historical mean AND append the top-k per-token average
                # as K extra coordinates ([mean; top] -> 2K). Both share the same
                # soft assignment, so no extra distance computation; only the
                # materialised [n_cells, K] assignment (already the cost of topk).
                assign = assignment(bag)  # [n_cells, K]
                count = int(round(config.abundance_topk_fraction * bag.shape[0]))
                count = min(max(count, config.abundance_topk_min), bag.shape[0])
                top = torch.topk(assign.T, count, dim=1).values  # [K, k]
                outputs.append(torch.cat([assign.mean(dim=0), top.mean(dim=1)], dim=0))
            else:
                raise ValueError(
                    f"abundance_pooling must be 'mean', 'max', 'topk', or "
                    f"'mean+topk', got {pooling!r}"
                )
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


def _kernel_matrix(left, right, config):
    """Kernel Gram matrix between rows of `left` and `right`.

    Operates on the CONTEXT-standardised abundance (`_standardise` output), so
    `kernel="linear"` is exactly the inner product the primal ridge solves.
    `kernel_gamma=None` defaults to 1 / dims (scikit-learn's heuristic), which
    keeps the RBF exponent O(1) on unit-RMS coordinates.
    """
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


def readout_kernel_ridge(abundance, labels, config) -> CTMargins:
    """Class-balanced kernel ridge in the DUAL: an n x n solve (n = context bags).

    Same class balance and CONTEXT-only standardisation as `readout_ridge`, so
    `kernel="linear"` reproduces the primal ridge to numerical precision. The
    non-linear kernels (rbf/poly) are the G0 diagnostic -- they answer whether
    the abundance->label relation has curvature the linear ridge cannot express.

    Weighted centring happens in kernel (feature) space, so the linear kernel
    recovers the primal's weighted centring plus intercept exactly, and label
    antisymmetry (one-hot targets exchange under a class swap) holds for every
    kernel: `alpha` columns swap, so the margin negates.
    """
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
    # Weighted feature-space mean: m(x_j) = sum_i w_i k(x_i, x_j) / W.
    m_context = (weight[None, :] @ k_context).squeeze(0) / total          # [n]
    mu2 = (weight[None, :] @ k_context @ weight[:, None]).squeeze() / (total * total)
    target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total  # [1, 2]
    centred_targets = (targets - target_mean) * root[:, None]              # [n, 2]
    gram = root[:, None] * (
        k_context - m_context[:, None] - m_context[None, :] + mu2
    ) * root[None, :] + config.ridge_lambda * torch.eye(
        k_context.shape[0], device=k_context.device, dtype=k_context.dtype
    )
    alpha = torch.linalg.solve(gram, centred_targets)                      # [n, 2]

    def values(kernel_to_context):
        # kernel_to_context: [n, t], k(context_j, target_i).
        m_target = (weight[None, :] @ kernel_to_context).squeeze(0) / total  # [t]
        cross = kernel_to_context - m_context[:, None] - m_target[None, :] + mu2  # [n, t]
        out = (alpha * root[:, None]).T @ cross + target_mean.T            # [2, t]
        out = out.T                                                        # [t, 2]
        return out[:, 1] - out[:, 0]

    context_margin = values(k_context)
    query_margin = values(_kernel_matrix(context, query, config))
    # Class-swap invariant magnitude (the head's SEP slot carries weight 0).
    separation = (alpha[:, 1] - alpha[:, 0]).abs().sum()
    return CTMargins(context_margin, query_margin, separation)


READOUTS = {
    "extreme": readout_extreme,
    "prototype": readout_prototype,
    "ridge": readout_ridge,
    "kernel_ridge": readout_kernel_ridge,
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
