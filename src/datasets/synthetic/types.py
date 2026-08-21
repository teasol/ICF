"""Types and task definitions for synthetic episode generation."""

from __future__ import annotations

from typing import NamedTuple
import torch

RESPONSE_TASK_NAMES: tuple[str, ...] = (
    "linear",
    "sparse",
    "interaction",
    "polynomial",
    "cluster_abundance",
    "variance_shift",
    "subtype_shift",
    "rare_cell_type",
    "multimodal_abundance",
    "geometric_mean",
    "latent_factor",
    "bimodal_response",
    "correlated_pairs",
    "dispersion_shift",
)

_CATEGORICAL_TASK_NAMES: tuple[str, ...] = (
    "cluster_abundance",
    "variance_shift",
    "subtype_shift",
    "rare_cell_type",
    "multimodal_abundance",
    "geometric_mean",
    "bimodal_response",
    "correlated_pairs",
    "dispersion_shift",
)

_NUMERIC_TASK_NAMES: tuple[str, ...] = (
    "linear",
    "sparse",
    "interaction",
    "polynomial",
    "latent_factor",
)


class SyntheticEpisode(NamedTuple):
    instances: torch.Tensor
    labels: torch.Tensor
    mask_indices: torch.Tensor
    query_indices: torch.Tensor
    task_name: str
    oracle_abundance: torch.Tensor | None = None
    cell_mask: torch.Tensor | None = None
    bag_lengths: torch.Tensor | None = None
