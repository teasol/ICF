"""CT branch: deterministic cell-type tokenizers, abundance, and readout (SS148).

Consolidated from src/models/ct/ subsystem into a single branch file.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, NamedTuple, Sequence

import torch
import torch.nn.functional as F


# ==============================================================================
# 1. Configuration & Data Structures
# ==============================================================================
_CONFIG_CELL_LIMIT = object()
MODES = ("extreme", "prototype", "ridge", "kernel_ridge")


@dataclass(frozen=True)
class CTReadoutConfig:
    num_tokens: int = 16
    cells_per_bag: int | None = 64
    cells_fraction: float | None = None
    cells_min: int = 1
    cells_scale: Literal["own", "median"] = "own"
    abundance_cells_per_bag: int | float | None | Literal["match"] = "match"
    abundance_pooling: str = "mean"
    abundance_topk_fraction: float = 0.1
    abundance_topk_min: int = 1
    sampling: Literal["even", "random"] = "random"
    sampling_seed: int = 0
    distance_kernel: Literal["broadcast", "gemm", "cosine"] = "gemm"
    temperature: float = 0.5
    eps: float = 1e-6
    pca_dim: int | None = None
    pca_scaling: Literal["standardise", "raw"] = "standardise"
    readout: str = "ridge"
    kmeans_iterations: int = 0
    kmeans_max_iterations: int = 8
    kmeans_tolerance: float = 1e-4
    kmeans_seed: int = 0
    tokenizer: Literal[
        "fps_lloyd", "kmeans_plusplus", "spherical_kmeans", "hierarchical_2means",
        "hdbscan", "dbscan",
    ] = "fps_lloyd"
    bisect_iterations: int = 2
    bisect_power_iterations: int = 3
    tree_reduction: Literal["segment", "atomic"] = "segment"
    hdbscan_min_cluster_size: int = 256
    hdbscan_min_cluster_fraction: float = 0.001
    hdbscan_min_samples: int = 32
    hdbscan_cluster_selection_method: Literal["eom", "leaf"] = "leaf"
    hdbscan_build_algo: Literal["nn_descent", "kd_tree", "brute"] = "nn_descent"
    hdbscan_allow_single_cluster: bool = False
    dbscan_eps: float | None = None
    dbscan_min_samples: int = 16
    ridge_lambda: float = 1.0
    kernel: Literal["linear", "rbf", "poly"] = "linear"
    kernel_gamma: float | None = None
    kernel_degree: int = 3
    kernel_coef0: float = 1.0


class CTAbundance(NamedTuple):
    context: torch.Tensor
    query: torch.Tensor
    tokens: torch.Tensor


class CTMargins(NamedTuple):
    context: torch.Tensor
    query: torch.Tensor
    separation: torch.Tensor
    coefficients: torch.Tensor | None = None


# ==============================================================================
# 2. Tokenizers and Centroid Algorithms
# ==============================================================================

def _fp32_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Run GEMM in FP32 without changing tensor dtype."""
    if not left.is_cuda or left.dtype != torch.bfloat16:
        return left @ right
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        return (left.float() @ right.float()).to(left.dtype)


def farthest_point_tokens(pooled: torch.Tensor, config: CTReadoutConfig) -> torch.Tensor:
    """Deterministic farthest-point selection over pooled cells."""
    count = min(config.num_tokens, pooled.shape[0])
    first = (pooled - pooled.mean(dim=0, keepdim=True)).square().mean(dim=1).argmin()
    selected = [first]
    if config.distance_kernel == "gemm":
        pooled_norm = pooled.square().mean(dim=1)

        def distance_to(index):
            token = pooled[index]
            return (
                pooled_norm
                + token.square().mean()
                - (2.0 / pooled.shape[1]) * _fp32_matmul(pooled, token)
            )
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
    """Seed K centroids with reproducible D-squared sampling."""
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
    """Deterministic full-cell PCA/2-means tree with exactly K leaves."""
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
    """Fit GPU HDBSCAN on every context cell and return its stable centroids."""
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
    """Token-dependent mean squared distance."""
    if distance_kernel == "broadcast":
        return (pooled[:, None, :] - tokens[None]).square().mean(-1)
    if distance_kernel == "gemm":
        distance = _fp32_matmul(pooled, tokens.T)
        distance.mul_(-2.0 / pooled.shape[1])
        distance.add_(tokens.square().mean(dim=1).unsqueeze(0))
        return distance
    if distance_kernel == "cosine":
        return (1.0 - _fp32_matmul(pooled, tokens.T)).clamp_min_(0)
    raise ValueError(
        "distance_kernel must be 'broadcast', 'gemm', or 'cosine', "
        f"got {distance_kernel!r}"
    )


def _assign(pooled: torch.Tensor, tokens: torch.Tensor,
            distance_kernel: str = "broadcast") -> torch.Tensor:
    """Nearest-token index per cell."""
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
    """Squared error to assigned token."""
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
    """Move tokens to their cluster means `iterations` times."""
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


# ==============================================================================
# 3. Cell Sampling & Abundance
# ==============================================================================

def parse_cell_budget(spec: str | int | float | None) -> tuple[int | None, float | None, str | None]:
    """Parse a `--cells-per-bag` argument into an integer limit or fraction."""
    if spec is None or spec == "all":
        return None, None, None
    if isinstance(spec, float):
        if not 0.0 < spec <= 1.0:
            raise ValueError("cells_fraction must be in (0, 1].")
        return None, float(spec), None
    if isinstance(spec, str):
        text = spec.strip()
        scale = None
        if text.startswith(("own:", "median:")):
            scale, text = text.split(":", 1)
        elif text.startswith("frac:"):
            text = text[len("frac:") :]
        if "." in text:
            fraction = float(text)
            if not 0.0 < fraction <= 1.0:
                raise ValueError("cells_fraction must be in (0, 1].")
            return None, fraction, scale
        if text == "all":
            return None, None, scale
        count = int(text)
        if count < 1:
            raise ValueError("cells_per_bag must be a positive integer or None.")
        return count, None, scale
    count = int(spec)
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
    """Project one sampled bag onto `basis`, matching the basis device."""
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
    """Select a capped subset using the configured reproducible policy."""
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
    """Steps 1-2: sample cells, optionally project, then standardise on context."""
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
        basis = pca_basis[:, : config.pca_dim].to(dtype=context[0].dtype)
        context = [_project_sampled_bag(bag, basis) for bag in context]
        query = [_project_sampled_bag(bag, basis) for bag in query]
    if normalisation is None:
        pooled = torch.cat(context, dim=0)
        centre = pooled.mean(dim=0, keepdim=True)
        if config.pca_scaling == "standardise" or not projected:
            scale = (pooled - centre).square().mean(dim=0, keepdim=True).sqrt()
        elif config.pca_scaling == "raw":
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


def _bag_abundance(
    bag: torch.Tensor,
    tokens: torch.Tensor,
    config: CTReadoutConfig,
    distance_kernel: str,
) -> torch.Tensor:
    """Step 4 & 5. Soft-assign a bag's cells to tokens."""
    distances = _token_distance(bag, tokens, distance_kernel)
    similarities = -distances / config.temperature
    weights = F.softmax(similarities, dim=1)

    if config.abundance_pooling == "mean":
        return weights.mean(dim=0)
    if config.abundance_pooling == "max":
        return weights.max(dim=0).values
    if config.abundance_pooling == "topk":
        k = max(
            config.abundance_topk_min,
            int(math.floor(config.abundance_topk_fraction * bag.shape[0] + 0.5)),
        )
        k = min(bag.shape[0], k)
        return torch.topk(weights, k, dim=0, largest=True).values.mean(dim=0)
    if config.abundance_pooling in ("mean+topk", "cattopk"):
        k = max(
            config.abundance_topk_min,
            int(math.floor(config.abundance_topk_fraction * bag.shape[0] + 0.5)),
        )
        k = min(bag.shape[0], k)
        topk = torch.topk(weights, k, dim=0, largest=True).values.mean(dim=0)
        return torch.cat((weights.mean(dim=0), topk), dim=-1)
    raise ValueError(
        f"abundance_pooling must be 'mean', 'max', 'topk', or 'mean+topk', got {config.abundance_pooling!r}"
    )


def ct_abundance(
    context_bags: Sequence[torch.Tensor],
    query_bags: Sequence[torch.Tensor],
    config: CTReadoutConfig,
    pca_basis: torch.Tensor | None = None,
    basis: torch.Tensor | None = None,
) -> CTAbundance:
    """Steps 1-5. Identical for every readout, and label-free by construction."""
    effective_pca_basis = pca_basis if pca_basis is not None else basis
    token_context, token_query, normalisation = prepare_cells(
        context_bags, query_bags, config, effective_pca_basis, return_normalisation=True
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
            effective_pca_basis,
            cells_per_bag=abundance_limit,
            normalisation=normalisation,
        )
        if spherical:
            context = [unit_normalise(bag) for bag in context]
            query = [unit_normalise(bag) for bag in query]

    rows = _distance_rows(tokens, distance_kernel)
    if rows >= max(b.shape[0] for b in context + query):
        context_abundance = torch.stack(
            [_bag_abundance(bag, tokens, config, distance_kernel) for bag in context]
        )
        query_abundance = torch.stack(
            [_bag_abundance(bag, tokens, config, distance_kernel) for bag in query]
        )
    else:
        def chunked_abundance(bag):
            weights = []
            for start in range(0, bag.shape[0], rows):
                chunk = bag[start : start + rows]
                dist = _token_distance(chunk, tokens, distance_kernel)
                weights.append(F.softmax(-dist / config.temperature, dim=1))
            all_weights = torch.cat(weights, dim=0)
            if config.abundance_pooling == "mean":
                return all_weights.mean(dim=0)
            if config.abundance_pooling == "max":
                return all_weights.max(dim=0).values
            if config.abundance_pooling == "topk":
                k = max(
                    config.abundance_topk_min,
                    int(math.floor(config.abundance_topk_fraction * bag.shape[0] + 0.5)),
                )
                k = min(bag.shape[0], k)
                return torch.topk(all_weights, k, dim=0, largest=True).values.mean(dim=0)
            if config.abundance_pooling in ("mean+topk", "cattopk"):
                k = max(
                    config.abundance_topk_min,
                    int(math.floor(config.abundance_topk_fraction * bag.shape[0] + 0.5)),
                )
                k = min(bag.shape[0], k)
                topk = torch.topk(all_weights, k, dim=0, largest=True).values.mean(dim=0)
                return torch.cat((all_weights.mean(dim=0), topk), dim=-1)
            raise ValueError(f"Unknown abundance_pooling: {config.abundance_pooling!r}")

        context_abundance = torch.stack([chunked_abundance(bag) for bag in context])
        query_abundance = torch.stack([chunked_abundance(bag) for bag in query])

    return CTAbundance(context_abundance, query_abundance, tokens)


# ==============================================================================
# 4. Readouts & Scoring
# ==============================================================================

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
    calibrated: bool = False,
    pca_basis: torch.Tensor | None = None,
) -> tuple[CTMargins, CTAbundance]:
    """Two-token or all-token abundance readout over context and query bags."""
    if config is None:
        config = CTReadoutConfig()
    abundance = ct_abundance(context_bags, query_bags, config, pca_basis=pca_basis)
    if mode == "extreme":
        margins = readout_extreme(abundance, labels, config)
    elif mode == "prototype":
        margins = readout_prototype(abundance, labels, config)
    elif mode == "ridge":
        margins = readout_ridge(abundance, labels, config)
    elif mode in ("kernel_ridge", "kernel-ridge"):
        margins = readout_kernel_ridge(abundance, labels, config)
    elif mode == "calibrated":
        ref = readout_extreme(abundance, labels, config)
        alt = readout_ridge(abundance, labels, config)
        return calibrate(alt, ref, config), abundance
    else:
        raise ValueError(
            f"mode must be 'extreme', 'prototype', 'ridge', 'kernel_ridge', or 'calibrated', got {mode!r}"
        )
    if calibrated and mode != "extreme":
        ref = readout_extreme(abundance, labels, config)
        margins = calibrate(margins, ref, config)
    return margins, abundance


# ==============================================================================
# 5. Branch Entry Point
# ==============================================================================
def ct_features(config, context_bags, labels, query_bags, basis=None):
    """Two-token abundance readout, delegated to `ct_readout` (docs SS148).

    Steps 1-5 (sample, standardise, farthest-point tokens, soft assign, per-bag
    average) live in `ct_readout.ct_abundance` so that the readout experiments
    cannot accidentally differ from this path in the REPRESENTATION -- only in
    step 6-7. `mode="extreme"` is today's behaviour and stays the default, so
    v107's output is unchanged; `tests/test_training_free.py` pins that against
    the lineage and `tests/test_ct_readout.py` pins the refactor itself.
    """
    margins, _ = ct_margins(
        context_bags, labels, query_bags,
        CTReadoutConfig(
            num_tokens=config.ct_num_tokens,
            cells_per_bag=config.ct_cells_per_bag,
            abundance_cells_per_bag=config.ct_abundance_cells_per_bag,
            cells_fraction=config.ct_cells_fraction,
            cells_min=config.ct_cells_min,
            cells_scale=config.ct_cells_scale,
            sampling=config.ct_sampling,
            sampling_seed=config.ct_sampling_seed,
            distance_kernel=config.ct_distance_kernel,
            tokenizer=config.ct_tokenizer,
            bisect_iterations=config.ct_bisect_iterations,
            bisect_power_iterations=config.ct_bisect_power_iterations,
            tree_reduction=config.ct_tree_reduction,
            hdbscan_min_cluster_size=config.ct_hdbscan_min_cluster_size,
            hdbscan_min_cluster_fraction=config.ct_hdbscan_min_cluster_fraction,
            hdbscan_min_samples=config.ct_hdbscan_min_samples,
            hdbscan_cluster_selection_method=(
                config.ct_hdbscan_cluster_selection_method
            ),
            hdbscan_build_algo=config.ct_hdbscan_build_algo,
            hdbscan_allow_single_cluster=config.ct_hdbscan_allow_single_cluster,
            dbscan_eps=config.ct_dbscan_eps,
            dbscan_min_samples=config.ct_dbscan_min_samples,
            temperature=config.ct_temperature,
            eps=config.ct_eps,
            pca_dim=config.ct_pca_dim,
            kmeans_iterations=config.ct_kmeans_iterations,
            kmeans_max_iterations=config.ct_kmeans_max_iterations,
            kmeans_tolerance=config.ct_kmeans_tolerance,
            kmeans_seed=config.ct_kmeans_seed,
        ),
        mode=config.ct_readout,
        # The SAME within-slide basis the CV branch uses, sliced to
        # `ct_pca_dim`. Reusing it costs no extra eigh -- and is also why the
        # gain is capped, since CT then lives inside a subspace CV already
        # covers (SS149-4).
        pca_basis=basis,
    )
    # The head consumes (q0, q1) and weighs q1 - q0, so hand back a pair whose
    # difference IS the margin. For "extreme" this returns exactly the two
    # standardised token abundances it always did.
    return -0.5 * margins.query, 0.5 * margins.query


__all__ = [
    "CTAbundance",
    "CTMargins",
    "CTReadoutConfig",
    "calibrate",
    "ct_abundance",
    "ct_features",
    "ct_margins",
    "dbscan_tokens",
    "discriminative_score",
    "farthest_point_tokens",
    "hdbscan_tokens",
    "hierarchical_2means_tokens",
    "kmeans_plusplus_tokens",
    "lloyd_refine",
    "parse_cell_budget",
    "prepare_cells",
    "readout_extreme",
    "readout_kernel_ridge",
    "readout_prototype",
    "readout_ridge",
    "resolve_cells_per_bag",
    "ridge_coefficients",
    "sample_cells",
    "typical_bag_size",
]
