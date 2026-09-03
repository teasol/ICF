"""CV branch: class-balanced ridge over context descriptors."""

from __future__ import annotations

import torch

from src.models.common.solvers import solve_ridge, standardise_blocks


def cv_logits(config, context: torch.Tensor, labels: torch.Tensor, query: torch.Tensor, split, return_loo: bool = False):
    context, query = standardise_blocks(context.float(), query.float(), split)
    labels = labels.long()
    targets = torch.nn.functional.one_hot(labels, 2).float()
    counts = torch.bincount(labels, minlength=2)
    weight = counts.float().reciprocal()[labels]
    total = weight.sum().clamp_min(1e-12)
    feature_mean = (weight[:, None] * context).sum(0, keepdim=True) / total
    target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
    root = weight.sqrt()[:, None]

    design = (context - feature_mean) * root
    centred_targets = (targets - target_mean) * root

    gram = design @ design.T
    dual = solve_ridge(gram, centred_targets, config.ridge_lambda)
    coefficients = design.T @ dual
    intercept = target_mean - feature_mean @ coefficients
    logits = (query @ coefficients + intercept) * config.ridge_scale

    if return_loo:
        size = gram.shape[0]
        identity = torch.eye(size, device=gram.device, dtype=torch.float32)
        inv_gram = solve_ridge(gram, identity, config.ridge_lambda)
        H = gram @ inv_gram
        h_diag = H.diagonal().clamp(0.0, 0.99)

        ctx_logits = (context @ coefficients + intercept) * config.ridge_scale
        ctx_m = ctx_logits[:, 1] - ctx_logits[:, 0]
        y_diff = torch.where(labels == 1, 1.0, -1.0).to(ctx_m.dtype).to(ctx_m.device)
        loo_m = (ctx_m - h_diag * y_diff) / (1.0 - h_diag).clamp_min(1e-4)
        return logits, loo_m

    return logits
