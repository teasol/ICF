"""Cell sampling, projection, and abundance representation."""

from __future__ import annotations

import math
from typing import Sequence
import torch
import torch.nn.functional as F

from src.models.ct.config import _CONFIG_CELL_LIMIT, CTAbundance, CTReadoutConfig
from src.models.ct.tokenizers import (
    _distance_rows,
    _fp32_matmul,
    _token_distance,
    dbscan_tokens,
    farthest_point_tokens,
    hdbscan_tokens,
    hierarchical_2means_tokens,
    kmeans_plusplus_tokens,
    lloyd_refine,
)


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
