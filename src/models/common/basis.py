"""Within-slide PCA basis and per-bag descriptor extraction."""

from __future__ import annotations

from typing import Sequence

import torch

from src.models.stream_eval import covariance_basis_from_bags


def within_slide_basis(
    context_bags: Sequence[torch.Tensor],
    sketch_dim: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Top-K eigenvectors of the WITHIN-slide pooled covariance.

    Accumulated bag by bag, one chunk at a time: concatenating every
    context cell first is what SS62-3 identified as an eval OOM driver
    (~12 GB for a full-tile episode), and `bag.double()` of a GPU-resident
    LUAD slide is what tips a 22 GiB card after the cohort is already up.
    """
    target_device = device if device is not None else context_bags[0].device
    return covariance_basis_from_bags(
        context_bags,
        "pca_within",
        sketch_dim,
        target_device,
    )


def extract_bag_descriptor(bag: torch.Tensor, basis: torch.Tensor, triangle) -> torch.Tensor:
    values = bag.to(basis.device).float()
    mean = values.mean(dim=0, keepdim=True)
    projected = (values - mean) @ basis
    covariance = (projected.T @ projected) / values.shape[0]
    return torch.cat((covariance[triangle[0], triangle[1]], mean.squeeze(0)))


def to_matrices(descriptors: torch.Tensor, triangle, sketch_dim: int) -> torch.Tensor:
    flat = descriptors[..., : triangle.shape[1]]
    matrices = flat.new_zeros(flat.shape[0], sketch_dim, sketch_dim)
    matrices[..., triangle[0], triangle[1]] = flat
    matrices[..., triangle[1], triangle[0]] = flat
    return matrices
