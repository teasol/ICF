"""CPU-resident bags with one-bag / one-chunk GPU uploads.

Official eval used to `.to(device)` every slide up front. On LUAD that is
~21 GiB of fp32 tiles, so the next `bag.double()` for within-slide PCA
(one 35k x 1536 block, ~400 MiB) OOMs. The tiles do not need to live on
the GPU: PCA and the covariance descriptor are both sums of per-bag
Gram matrices, and CT only needs the sampled / projected cells.

This module keeps the raw bags on CPU and streams one chunk at a time
onto `device` for the GEMM. Per-bag sufficient stats `(n, mean, S)` are
fold-independent, so a caller can cache them and assemble each fold's
scatter as a sum over context bags (test bags never enter).
"""

from __future__ import annotations

from typing import Iterable, Literal, Mapping, MutableMapping, Sequence

import torch

# fp64 GEMM of 16384 x 1536 is ~192 MiB — small enough that one LUAD bag
# (max ~35k cells) never sits on the GPU as a full double copy.
DEFAULT_STREAM_CHUNK = 2**14

BagStats = tuple[int, torch.Tensor, torch.Tensor]
BagStatsCache = MutableMapping[int, BagStats]
BasisMode = Literal["pca", "pca_within"]


def _as_device(device: torch.device | str | None) -> torch.device | None:
    if device is None:
        return None
    return torch.device(device)


def bag_within_stats(
    bag: torch.Tensor,
    device: torch.device | str | None = None,
    chunk: int = DEFAULT_STREAM_CHUNK,
    cache: BagStatsCache | None = None,
) -> BagStats:
    """Within-bag mean and centred scatter, streamed in float64.

    Returns ``(n, mean[D], S[D, D])`` where ``S = (X - mean)^T (X - mean)``.
    ``mean`` and ``S`` land on ``device`` (the bag's device when omitted).
    Cache entries are stored on CPU and keyed by ``id(bag)``, which is
    stable across official folds that reuse the same tensor objects.
    """
    if bag.ndim != 2:
        raise ValueError(f"bag must be [cells, dim], got {tuple(bag.shape)}")
    if chunk < 1:
        raise ValueError("chunk must be a positive integer.")
    n, dim = bag.shape
    if n < 1:
        raise ValueError("Every bag must contain at least one cell.")
    if cache is not None and id(bag) in cache:
        count, mean, scatter = cache[id(bag)]
        target = _as_device(device) or bag.device
        return count, mean.to(target), scatter.to(target)

    target = _as_device(device) or bag.device
    summation = torch.zeros(dim, dtype=torch.float64, device=target)
    for start in range(0, n, chunk):
        block = bag[start : start + chunk].to(target, non_blocking=True).double()
        summation += block.sum(dim=0)
    mean = summation / n
    scatter = torch.zeros(dim, dim, dtype=torch.float64, device=target)
    for start in range(0, n, chunk):
        block = bag[start : start + chunk].to(target, non_blocking=True).double()
        centered = block - mean
        scatter += centered.T @ centered
    if cache is not None:
        cache[id(bag)] = (n, mean.cpu(), scatter.cpu())
    return n, mean, scatter


def assemble_scatter(
    stats: Sequence[BagStats],
    mode: BasisMode,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, int]:
    """Fold scatter from per-bag sufficient stats. Test bags must be omitted."""
    if not stats:
        raise ValueError("Need at least one context bag to assemble a scatter.")
    if mode not in ("pca", "pca_within"):
        raise ValueError(f"mode must be 'pca' or 'pca_within', got {mode!r}")
    target = _as_device(device) or stats[0][2].device
    dim = stats[0][1].shape[0]
    total = 0
    scatter = torch.zeros(dim, dim, dtype=torch.float64, device=target)
    weighted_mean = torch.zeros(dim, dtype=torch.float64, device=target)
    means: list[tuple[int, torch.Tensor]] = []
    for count, mean, bag_scatter in stats:
        total += count
        scatter += bag_scatter.to(target)
        if mode == "pca":
            mean_dev = mean.to(target).double()
            weighted_mean += count * mean_dev
            means.append((count, mean_dev))
    if total < 1:
        raise ValueError("Context bags are empty.")
    if mode == "pca":
        global_mean = weighted_mean / total
        for count, mean_dev in means:
            delta = mean_dev - global_mean
            scatter += count * torch.outer(delta, delta)
    return scatter, total


def accumulate_scatter(
    bags: Sequence[torch.Tensor],
    mode: BasisMode,
    device: torch.device | str | None = None,
    chunk: int = DEFAULT_STREAM_CHUNK,
    cache: BagStatsCache | None = None,
) -> tuple[torch.Tensor, int]:
    """Stream each bag, then assemble the fold scatter."""
    if not bags:
        raise ValueError("Need at least one context bag to accumulate a scatter.")
    target = _as_device(device) or bags[0].device
    stats = [bag_within_stats(bag, target, chunk, cache) for bag in bags]
    return assemble_scatter(stats, mode, target)


def covariance_basis_from_scatter(
    scatter: torch.Tensor,
    total: int,
    sketch_dim: int,
) -> torch.Tensor:
    """Top-``sketch_dim`` eigenvectors of ``scatter / total``, descending."""
    if total < 1:
        raise ValueError("Need at least one cell to form a covariance basis.")
    if not 1 <= sketch_dim <= scatter.shape[0]:
        raise ValueError(
            f"sketch_dim must be in [1, {scatter.shape[0]}], got {sketch_dim}."
        )
    _, vectors = torch.linalg.eigh(scatter / total)
    return vectors[:, -sketch_dim:].flip(-1).float()


def covariance_basis_from_bags(
    bags: Sequence[torch.Tensor],
    mode: BasisMode,
    sketch_dim: int,
    device: torch.device | str | None = None,
    chunk: int = DEFAULT_STREAM_CHUNK,
    cache: BagStatsCache | None = None,
) -> torch.Tensor:
    """Within- or pooled-PCA basis of CONTEXT bags, streamed onto ``device``."""
    scatter, total = accumulate_scatter(bags, mode, device, chunk, cache)
    return covariance_basis_from_scatter(scatter, total, sketch_dim)


def cpu_bag_mapping(
    bags: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Keep a slide -> bag dict on CPU. No-op for tensors already there."""
    return {
        slide_id: bag if bag.device.type == "cpu" else bag.detach().cpu()
        for slide_id, bag in bags.items()
    }


def project_bags_to_cpu(
    bags: Mapping[str, torch.Tensor],
    mean: torch.Tensor,
    components: torch.Tensor,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Project one bag at a time on ``device`` and return CPU features."""
    target = torch.device(device)
    mean_dev = mean.to(target)
    components_dev = components.to(target)
    projected: dict[str, torch.Tensor] = {}
    for slide_id, bag in bags.items():
        gpu_bag = bag.to(target, non_blocking=True)
        projected[slide_id] = ((gpu_bag - mean_dev) @ components_dev).cpu()
        del gpu_bag
    return projected


def stream_descriptor(compute, bag: torch.Tensor, device: torch.device | str):
    """Run ``compute(bag[None])`` on ``device`` and free the uploaded bag.

    ``compute`` is the model's ``_descriptors``. The bag stays on CPU; only
    this one [1, n, d] view is uploaded. The descriptor itself is small
    (the K x K triangle plus the bag mean) and stays on ``device``.
    """
    target = torch.device(device)
    if bag.device == target:
        return compute(bag.unsqueeze(0))
    gpu_bag = bag.to(target, non_blocking=True)
    try:
        return compute(gpu_bag.unsqueeze(0))
    finally:
        del gpu_bag


def move_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Copy ``tensor`` onto ``reference``'s device/dtype when they differ."""
    if tensor.device == reference.device and tensor.dtype == reference.dtype:
        return tensor
    return tensor.to(device=reference.device, dtype=reference.dtype)


def bags_are_cpu(bags: Iterable[torch.Tensor]) -> bool:
    return all(bag.device.type == "cpu" for bag in bags)


def stream_lineage_forward(model, bags, labels, query_index, device):
    """`CovarianceMeanDDCTMLPModel.forward` with one-bag GPU residency.

    Descriptors are the only large GEMM; they are computed from a single
    uploaded bag and then discarded. Raw tiles stay on CPU and are handed
    to CT/PA as CPU tensors. Numerically this is the lineage forward:
    `_descriptors` and `_relation_logits` are the same methods.
    """
    target = torch.device(device)
    descriptors = torch.cat(
        [stream_descriptor(model._descriptors, bag, target) for bag in bags]
    )
    if labels.shape[0] != descriptors.shape[0]:
        raise ValueError("Every bag needs exactly one label.")
    index = query_index.long()
    is_context = model._context_split(len(bags), index, descriptors.device)
    context_bags = [bags[i] for i in is_context.nonzero().flatten().tolist()]
    query_bags = [bags[i] for i in index.tolist()]
    return model._relation_logits(
        descriptors[is_context],
        labels[is_context],
        descriptors[index],
        context_bags,
        query_bags,
    )
