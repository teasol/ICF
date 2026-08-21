"""Configuration and data structures for CT readout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple
import torch

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
