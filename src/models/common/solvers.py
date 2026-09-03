"""Shared ridge/kernel-ridge solvers and standardisation helpers."""

from __future__ import annotations

import torch


def solve_ridge(gram: torch.Tensor, targets: torch.Tensor, penalty: float) -> torch.Tensor:
    """Solve (gram + penalty*I) x = targets, adding jitter only if it fails."""
    orig_dtype = targets.dtype
    gram_f32 = gram.float()
    targets_f32 = targets.to(dtype=torch.float32, device=gram.device)
    size = gram_f32.shape[-1]
    identity = torch.eye(size, device=gram.device, dtype=torch.float32)
    jitter = 0.0
    for _ in range(6):
        try:
            factor = torch.linalg.cholesky(gram_f32 + (penalty + jitter) * identity)
            sol = torch.cholesky_solve(targets_f32, factor)
            return sol.to(dtype=orig_dtype)
        except RuntimeError:
            jitter = max(jitter * 10.0, 1e-6 * float(gram_f32.diagonal().abs().mean()) + 1e-12)
    sol = torch.linalg.lstsq(gram_f32 + (penalty + jitter) * identity, targets_f32).solution
    return sol.to(dtype=orig_dtype)


def kernel_matrix(left: torch.Tensor, right: torch.Tensor, kernel: str = "rbf", gamma: float | None = None, degree: int = 2, coef0: float = 1.0) -> torch.Tensor:

    """Compute kernel Gram matrix between left and right."""
    if kernel == "linear":
        return left @ right.T
    left_f = left.float()
    right_f = right.float()
    squared = left_f @ right_f.T
    dims = left.shape[-1]
    if gamma is None:
        gamma = 1.0 / dims
    if kernel == "rbf":
        sq_left = (left_f * left_f).sum(dim=1, keepdim=True)
        sq_right = (right_f * right_f).sum(dim=1, keepdim=True)
        distances = (sq_left - 2.0 * squared + sq_right.T).clamp_min(0.0)
        return torch.exp(-gamma * distances)
    elif kernel == "poly":
        return (gamma * squared + coef0).pow(degree)
    elif kernel == "cosine":
        l_norm = torch.nn.functional.normalize(left_f, dim=-1)
        r_norm = torch.nn.functional.normalize(right_f, dim=-1)
        return l_norm @ r_norm.T
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")


def fast_context_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Fast Mann-Whitney U AUROC on context set (tensor-native, no sklearn/scipy)."""
    labels = labels.long()
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
        return 0.5
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ordered = scores[order]
    index = 0
    n = len(ordered)
    while index < n:
        end = index
        while end + 1 < n and ordered[end + 1] == ordered[index]:
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    sum_pos_ranks = ranks[labels == 1].sum().item()
    return float((sum_pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg))


def solve_kernel_ridge(
    ctx_feats: torch.Tensor,
    ctx_labels: torch.Tensor,
    qry_feats: torch.Tensor,
    kernel: str = "linear",
    gamma: float | None = None,
    degree: int = 2,
    coef0: float = 1.0,
    reg_lambda: float = 1.0,
    return_logits: bool = False,
    return_loo: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Class-balanced centered Kernel Ridge Regression (n x n dual solve) with exact label antisymmetry."""
    if kernel == "linear":
        centre = ctx_feats.mean(dim=0, keepdim=True)
        scale = (ctx_feats - centre).square().mean(dim=0).sqrt().clamp_min(1e-6)
        ctx_std = (ctx_feats - centre) / scale
        qry_std = (qry_feats - centre) / scale

        labels = ctx_labels.long()
        targets = torch.nn.functional.one_hot(labels, 2).float()
        counts = torch.bincount(labels, minlength=2)
        if bool((counts == 0).any()):
            raise ValueError("Every class must occur in the context set.")
        weight = counts.float().reciprocal()[labels]
        total = weight.sum().clamp_min(1e-12)
        feature_mean = (weight[:, None] * ctx_std).sum(0, keepdim=True) / total
        target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
        root = weight.sqrt()[:, None]

        design = (ctx_std - feature_mean) * root
        centred_targets = (targets - target_mean) * root

        if not return_loo and design.shape[0] > design.shape[1]:
            # Primal solve: D x D instead of N x N (exact Woodbury equivalence)
            gram_primal = design.T @ design
            coefficients = solve_ridge(gram_primal, design.T @ centred_targets, reg_lambda)
        else:
            gram = design @ design.T
            dual = solve_ridge(gram, centred_targets, reg_lambda)
            coefficients = design.T @ dual
        intercept = target_mean - feature_mean @ coefficients
        logits = qry_std @ coefficients + intercept
        qry_margin = logits if return_logits else (logits[:, 1] - logits[:, 0])

        if return_loo:
            size = gram.shape[0]
            identity = torch.eye(size, device=gram.device, dtype=torch.float32)
            inv_gram = solve_ridge(gram, identity, reg_lambda)
            H = gram @ inv_gram
            h_diag = H.diagonal().clamp(0.0, 0.99)

            ctx_logits = ctx_std @ coefficients + intercept
            ctx_m = ctx_logits[:, 1] - ctx_logits[:, 0]
            y_diff = torch.where(labels == 1, 1.0, -1.0).to(ctx_m.dtype).to(ctx_m.device)
            loo_m = (ctx_m - h_diag * y_diff) / (1.0 - h_diag).clamp_min(1e-4)
            return qry_margin, loo_m

        return qry_margin


    labels = ctx_labels.long()
    device = ctx_feats.device

    # 1. Per-feature standardisation using context statistics
    centre = ctx_feats.mean(dim=0, keepdim=True)
    scale = (ctx_feats - centre).square().mean(dim=0).sqrt().clamp_min(1e-6)
    ctx_std = (ctx_feats - centre) / scale
    qry_std = (qry_feats - centre) / scale

    # 2. Kernel Gram matrices
    k_ctx = kernel_matrix(ctx_std, ctx_std, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    k_qry = kernel_matrix(qry_std, ctx_std, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

    # 3. Class-balanced centered targets and centered Gram matrix
    targets = torch.nn.functional.one_hot(labels, 2).float()
    counts = torch.bincount(labels, minlength=2)
    if bool((counts == 0).any()):
        raise ValueError("Every class must occur in the context set.")
    weight = counts.float().reciprocal()[labels]
    total = weight.sum().clamp_min(1e-12)
    root = weight.sqrt()

    m_ctx = (weight[None, :] @ k_ctx).squeeze(0) / total
    mu2 = (weight[None, :] @ k_ctx @ weight[:, None]).squeeze() / (total * total)
    target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
    centred_targets = (targets - target_mean) * root[:, None]

    gram = root[:, None] * (k_ctx - m_ctx[:, None] - m_ctx[None, :] + mu2) * root[None, :]
    dimension = gram.shape[0]
    identity = torch.eye(dimension, device=device, dtype=torch.float32)

    # 4. Robust solve
    jitter = 0.0
    dual = None
    for _ in range(6):
        try:
            factor = torch.linalg.cholesky(gram + (reg_lambda + jitter) * identity)
            dual = torch.cholesky_solve(centred_targets, factor)
            break
        except RuntimeError:
            jitter = max(jitter * 10.0, 1e-6 * float(gram.diagonal().abs().mean()) + 1e-12)
    if dual is None:
        dual = torch.linalg.lstsq(gram + (reg_lambda + jitter) * identity, centred_targets).solution

    alpha = root[:, None] * dual
    intercept = target_mean - (m_ctx @ alpha)[None, :]

    m_qry = (weight[None, :] @ k_qry.T).squeeze(0) / total
    logits = k_qry @ alpha - m_qry[:, None] * alpha.sum(0, keepdim=True) + intercept
    return logits if return_logits else (logits[:, 1] - logits[:, 0])


def standardise(context: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Context-only centring and one scalar RMS scale, applied to both sides."""
    centre = context.mean(dim=0, keepdim=True)
    context = context - centre
    query = query - centre
    rms = context.square().mean().sqrt().clamp_min(1e-6)
    return context / rms, query / rms


def standardise_blocks(context, query, split):
    """Standardise the covariance and mean halves SEPARATELY.

    Not cosmetic. The covariance triangle has ~33k entries and the bag mean 1,536,
    and their natural scales differ by orders of magnitude; one shared RMS would
    let whichever block is larger dominate the ridge. The lineage does the same
    (`CovarianceMeanRidgeModel._normalize_descriptors` calls `_normalize_block`
    once per block), and skipping it was the one discrepancy that made the first
    version of this file disagree with the lineage by ~2%.
    """
    context_covariance, context_mean = context.split(split, dim=-1)
    query_covariance, query_mean = query.split(split, dim=-1)
    context_covariance, query_covariance = standardise(context_covariance, query_covariance)
    if context_mean.shape[-1] == 0:
        # v109's off-diagonal descriptor has no mean block; standardising an empty
        # tensor would return NaN from the mean of nothing.
        return context_covariance, query_covariance
    context_mean, query_mean = standardise(context_mean, query_mean)
    return (
        torch.cat((context_covariance, context_mean), dim=-1),
        torch.cat((query_covariance, query_mean), dim=-1),
    )


# Backward-compatible underscored aliases.
_solve_ridge = solve_ridge
_kernel_matrix = kernel_matrix
_fast_context_auroc = fast_context_auroc
_solve_kernel_ridge = solve_kernel_ridge
_standardise = standardise
_standardise_blocks = standardise_blocks
