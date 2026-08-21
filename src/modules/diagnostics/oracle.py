"""Oracle abundance ridge fitting and diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from src.modules.diagnostics.metrics import binary_query_diagnostics


@torch.no_grad()
def fit_oracle_abundance_logits(
    abundance: torch.Tensor,
    labels: torch.Tensor,
    mask_index: torch.Tensor,
    ridge_lambda: float = 1e-3,
) -> torch.Tensor:
    """Fit a detached 1-D ridge classifier using labelled context only."""
    abundance = abundance.detach().float().flatten()
    labels = labels.detach().long().flatten()
    mask_index = mask_index.detach().long().flatten()
    context_mask = torch.ones_like(labels, dtype=torch.bool)
    context_mask[mask_index] = False
    context_abundance = abundance[context_mask]
    context_labels = labels[context_mask]
    if context_abundance.numel() < 2 or torch.unique(context_labels).numel() < 2:
        raise ValueError("Oracle ridge fitting requires both classes in context.")

    center = context_abundance.mean()
    scale = context_abundance.std(unbiased=False).clamp_min(1e-6)
    context_feature = (context_abundance - center) / scale
    query_feature = (abundance[mask_index] - center) / scale
    design = torch.stack((context_feature, torch.ones_like(context_feature)), dim=1)
    target = context_labels.float().mul(2).sub(1)
    penalty = torch.diag(
        torch.tensor([ridge_lambda, 0.0], device=design.device, dtype=design.dtype)
    )
    with torch.autocast(device_type=abundance.device.type, enabled=False):
        coefficients = torch.linalg.solve(
            design.float().T @ design.float() + penalty.float(),
            design.float().T @ target.float(),
        )
    score = query_feature * coefficients[0] + coefficients[1]
    return torch.stack((-0.5 * score, 0.5 * score), dim=-1).detach()


@torch.no_grad()
def oracle_abundance_diagnostics(
    abundance: torch.Tensor,
    labels: torch.Tensor,
    mask_index: torch.Tensor,
    model_auroc: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute performance gap between model and detached oracle ridge."""
    oracle_logits = fit_oracle_abundance_logits(abundance, labels, mask_index)
    query_labels = labels.detach().long()[mask_index]
    diagnostics = binary_query_diagnostics(oracle_logits, query_labels)
    class0 = abundance.detach().float()[mask_index][query_labels == 0]
    class1 = abundance.detach().float()[mask_index][query_labels == 1]
    if class0.numel() and class1.numel():
        pooled_variance = (
            class0.var(unbiased=False) + class1.var(unbiased=False)
        ) / 2
        snr = (class1.mean() - class0.mean()).abs() / torch.sqrt(
            pooled_variance + 1e-8
        )
    else:
        snr = abundance.detach().float().sum() * 0
    oracle_auroc = diagnostics["auroc"]
    return {
        "oracle_abundance_accuracy": (oracle_logits.argmax(dim=-1) == query_labels)
        .float()
        .mean(),
        "oracle_abundance_balanced_accuracy": diagnostics["balanced_accuracy"],
        "oracle_abundance_auroc": oracle_auroc,
        "oracle_abundance_ce": F.cross_entropy(oracle_logits, query_labels),
        "oracle_abundance_snr": snr,
        "oracle_model_auroc_gap": oracle_auroc - model_auroc.detach().float(),
    }
