"""Tokenizers and centroid algorithms for cell summarization."""

from __future__ import annotations

import math
import torch
from src.models.ct.config import CTReadoutConfig


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
