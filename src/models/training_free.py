"""v107: the training-free in-context classifier (docs SS139, SS140, SS142).

This is a standalone reimplementation of the configuration adopted in SS139 —
within-slide PCA projection plus a constant head — written from scratch rather
than assembled from the relation lineage. That lineage
(`set_transformer_ridge.py`, ~2,400 lines) carries a Set-Transformer encoder,
learnable-P variants, PA/dual-projection/gradient-weight arms and their
checkpoint-compatibility machinery, none of which this configuration uses. What
remains is small enough to read in one sitting, and being able to read it is the
point: with no parameters there is nothing to inspect but the code.

WHAT IT COMPUTES, for one episode of labelled context bags and unlabelled query
bags of UNI2 tiles (each bag is [cells, 1536]):

  1. BASIS      Pool the context cells' covariance with each bag centred on its
                OWN mean, take the top-K eigenvectors. Centring per bag rather
                than globally drops the between-slide term, which SS123-4 measured
                at ICC 31.6% and SS139-4 showed is worth +0.0020 — a third of the
                basis would otherwise be spent on directions that only tell
                slides apart (staining, scanner, patient).
  2. DESCRIPTOR Each bag becomes [upper triangle of B^T C B, bag mean].
  3. CV         Class-balanced ridge over context descriptors, solved in the dual
                (bags number ~200 against a descriptor of ~34k), giving two logits
                per query.
  4. DD         A rank-1 dispersion direction from the sketched covariances, then
                LOGITS per class, like CV and CT: the class-1-positive
                ordered-coordinate x nearest-class typicality evidence (SS183),
                or the negated normalised distances for the historical
                ``distance`` readout. The head weighs the logit difference
                (DD1-DD0) positively.
  5. CT         Deterministic farthest-point tokens over context cells (labels
                never enter the selection), each bag summarised by its soft
                abundance over those tokens, and the two most class-separating
                tokens read off.
  6. HEAD       margin = 1.442*(CV1-CV0) + 1.0*(DD1-DD0) + 0.7*(q1-q0)
                logits = (-margin/2, +margin/2)

WHY THE HEAD IS CONSTANT (SS137-3, confirmed on the official path in SS138-4 at a
cost of -0.0003). Swapping the two classes must flip the margin's sign. Under
that swap CV0<->CV1, D0<->D1 and q0<->q1 exchange while the three separation
features are INVARIANT, so label antisymmetry forces their weights and the bias
to zero and forces each pair to be equal and opposite. The difference features
are linear combinations of their pair and add nothing to a linear head. One
number per branch survives, and the eight trained heads agreed on those numbers
to two decimals (std 0.027 / 0.008 / 0.012).

⚠️ NOT a drop-in replacement for the lineage model: no `forward(instances,
labels, query_index)` signature, no checkpoint compatibility, no training path.
It is for evaluation of the training-free configuration only.

⚠️ It is NOT established as better than the trained v98 — seed-paired it is
-0.0037 (t = -1.08, unresolved). SS139-2 records the adoption as a deliberate
trade: a little macro for no training, no seed repetition, and a deterministic
result.

`tests/test_training_free.py` pins this against the patched lineage path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from src.models.ct_readout import CTReadoutConfig, ct_margins
from src.models.dd_adaptive_rank import ordered_typicality_margin
from src.models.stream_eval import covariance_basis_from_bags


def _solve_ridge(gram: torch.Tensor, targets: torch.Tensor, penalty: float) -> torch.Tensor:
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


def _kernel_matrix(left: torch.Tensor, right: torch.Tensor, kernel: str = "rbf", gamma: float | None = None, degree: int = 2, coef0: float = 1.0) -> torch.Tensor:

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


def _fast_context_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
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


def _solve_kernel_ridge(
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

        gram = design @ design.T
        dual = _solve_ridge(gram, centred_targets, reg_lambda)
        coefficients = design.T @ dual
        intercept = target_mean - feature_mean @ coefficients
        logits = qry_std @ coefficients + intercept
        qry_margin = logits if return_logits else (logits[:, 1] - logits[:, 0])

        if return_loo:
            size = gram.shape[0]
            identity = torch.eye(size, device=gram.device, dtype=torch.float32)
            inv_gram = _solve_ridge(gram, identity, reg_lambda)
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
    k_ctx = _kernel_matrix(ctx_std, ctx_std, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    k_qry = _kernel_matrix(qry_std, ctx_std, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

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



def _standardise(context: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Context-only centring and one scalar RMS scale, applied to both sides."""
    centre = context.mean(dim=0, keepdim=True)
    context = context - centre
    query = query - centre
    rms = context.square().mean().sqrt().clamp_min(1e-6)
    return context / rms, query / rms


def _standardise_blocks(context, query, split):
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
    context_covariance, query_covariance = _standardise(context_covariance, query_covariance)
    if context_mean.shape[-1] == 0:
        # v109's off-diagonal descriptor has no mean block; standardising an empty
        # tensor would return NaN from the mean of nothing.
        return context_covariance, query_covariance
    context_mean, query_mean = _standardise(context_mean, query_mean)
    return (
        torch.cat((context_covariance, context_mean), dim=-1),
        torch.cat((query_covariance, query_mean), dim=-1),
    )


@dataclass(frozen=True)
class TrainingFreeConfig:
    # SS142: K was locked to 128 while P was a learned 1536xK matrix. Under PCA it
    # is a free evaluation knob, and sweeping it found an inverted U peaking on a
    # 256-512 plateau. 256 is the promoted value: +0.0076 over 128 across all 17
    # tasks (t=3.01, 12/17), replicated on the 7 held out from the choice, at no
    # runtime cost. K=384 scored higher on the SEAL 10 but did not replicate.
    sketch_dim: int = 256
    ridge_lambda: float = 1.0
    ridge_scale: float = 2.0
    dd_shrinkage: float = 0.25
    dd_eps: float = 1e-6
    # SS183 (v112, user decision). ``ordered_typicality`` replaces ``distance``:
    # bounded ordered-coordinate x nearest-class typicality evidence, kappa=1.
    # Official 17-task macro: SEAL 0.70432 (-0.00021 vs v111), held-out 0.60181
    # (+0.00372), all-17 0.66211 (+0.00141). 7/17 tasks improve (BAP1 +0.0198,
    # KEAP1 +0.0144, Histologic Grade +0.0152 lead); SEAL is flat, held-out is
    # the clear driver. ``distance`` remains available for historical repro
    # (see scripts/eval_v111.sh).
    dd_readout: str = "ordered_typicality"
    dd_separation_floor: float = 1.0
    # SS181 (v111, user decision). Promote the best fully deterministic CT arm:
    # full-cell hierarchical PCA/2-means at PCA32/K256. It scores below v110 but
    # removes both storage-order selection bias and sampling-seed dependence.
    ct_num_tokens: int = 256
    ct_cells_per_bag: int | None = None
    ct_abundance_cells_per_bag: int | None | str | float = None
    # Size-aware sampling (v113). None keeps the v111/v112 full-cell path.
    # A value in (0, 1] draws round(fraction * bag_or_median) cells.
    ct_cells_fraction: float | None = None
    ct_cells_min: int = 1
    ct_cells_scale: str = "own"
    # Inactive when every cell is used; retained only as a valid explicit policy.
    ct_sampling: str = "even"
    ct_sampling_seed: int = 0
    ct_distance_kernel: str = "gemm"
    ct_tokenizer: str = "hierarchical_2means"
    ct_bisect_iterations: int = 2
    ct_bisect_power_iterations: int = 3
    ct_tree_reduction: str = "segment"
    ct_hdbscan_min_cluster_size: int = 256
    ct_hdbscan_min_cluster_fraction: float = 0.001
    ct_hdbscan_min_samples: int = 32
    ct_hdbscan_cluster_selection_method: str = "leaf"
    ct_hdbscan_build_algo: str = "nn_descent"
    ct_hdbscan_allow_single_cluster: bool = False
    ct_dbscan_eps: float | None = None
    ct_dbscan_min_samples: int = 16
    ct_temperature: float = 0.5
    ct_eps: float = 1e-6
    # SS137-3: CV : DD : CT. DD now emits LOGITS with a class-1-positive
    # difference (dd1 - dd0), exactly like CV and CT, so the weight is positive.
    # The ``distance`` readout is negated into logits internally (a large
    # distance is evidence AGAINST that class). -0.343 was fitted against the old
    # distance magnitude; SS183 (v112) removes that unjustified scaling and takes
    # the bounded ordered_typicality margin at weight 1.
    weight_cv: float = 1.442
    weight_dd: float = 1.0
    # SS157-5: 0.286 was fitted against the collapsed FPS tokens. With k-means
    # tokens the branch earns more weight -- sign agreement HOLDS at 11/17 for
    # 0.5-0.7 where on FPS tokens it fell to 7/17. 0.5/0.7/1.0 sit within 0.002 of
    # each other, so 0.7 is "best mean and best agreement", not a sharp optimum.
    weight_ct: float = 0.7
    # SS152 (v108, user decision). CT reads all 16 abundance dims through a
    # class-balanced ridge, and measures cell-token distance in the leading 32 PCA
    # directions instead of raw 1,536. The two are promoted TOGETHER because
    # neither works alone: singles are +0.0019 and +0.0008 on 17 tasks, the
    # combination is +0.0037 and is the only CT variant positive in both task
    # groups (SS150-2). v107's values were "extreme" and None.
    ct_readout: str = "ridge"
    ct_pca_dim: int | None = 32
    # Historical FPS/Lloyd and k-means++ controls; inactive under v111's tree.
    ct_kmeans_iterations: int = 0
    ct_kmeans_max_iterations: int = 8
    ct_kmeans_tolerance: float = 1e-4
    ct_kmeans_seed: int = 0
    # Which blocks of the descriptor the CV RIDGE sees (SS156). "offdiag" drops the
    # 256 diagonal entries AND the 1,536 raw bag mean: the mean adds nothing
    # (+0.0019 to remove) and the diagonal is actively harmful (+0.0052 to remove,
    # 13/17). ⚠️ DD still receives the FULL triangle -- it rebuilds its K x K
    # matrices from it, so masking globally would break DD, not just narrow CV.
    cv_blocks: str = "offdiag"
    weight_ct: float = 1.0
    dd_readout: str = "ordered_typicality"
    dd_separation_floor: float = 1.0
    weight_dd: float = 0.0  # v117/v118: DD removed (ablation win +0.0072)
    weight_cv: float = 1.0
    # BM branch (v115): projected bag-mean in leading subspace with class-balanced ridge.
    bm_dim: int = 32
    bm_lambda: float = 1.0
    weight_bm: float = 1.0
    # BD branch (v116): overall bag dispersion / spectral entropy of projected covariance with ordered-typicality or ridge.
    bd_dim: int = 256
    bd_metric: str = "entropy"
    bd_lambda: float = 1.0
    bd_separation_floor: float = 1.0
    bd_eps: float = 1e-6
    bd_readout: str = "ordered_typicality"
    weight_bd: float = 1.0
    # Head aggregation (v118): "soft_voting" (probability average via sigmoid) | "linear" (v117 linear sum)
    aggregation: str = "soft_voting"
    # QA branch (v119): quantile / extremum statistics of projected cells with class-balanced ridge.
    qa_dim: int = 32
    qa_quantiles: tuple[float, ...] = (0.05, 0.10, 0.90, 0.95)
    qa_lambda: float = 1.0
    weight_qa: float = 0.0  # Default 0.0 for v118 baseline, set to 1.0 for QA arm
    # DS branch: in-context salience denoising (cluster log-odds salience patch weighting) with class-balanced ridge.
    ds_dim: int = 32
    ds_lambda: float = 1.0
    ds_temperature: float = 1.0
    ds_tokens: int = 256
    weight_ds: float = 0.0
    # LR branch: direct in-context patch likelihood ratio + Top-K MIL Extreme Pooling with class-balanced ridge.
    lr_dim: int = 32
    lr_lambda: float = 1.0
    lr_tau: float = 5.0
    lr_topk_fraction: float = 0.05
    lr_topk_min: int = 4
    lr_topk_max: int = 64
    lr_patches_per_ctx: int = 64
    weight_lr: float = 0.0
    # DE branch: In-Subspace Dual Extreme Instance MIL (Top-K/Bottom-K difference within 32D PCA)
    de_dim: int = 32
    de_topk_fraction: float = 0.05
    de_topk_min: int = 4
    de_topk_max: int = 64
    de_lambda: float = 1.0
    weight_de: float = 0.0
    # SW branch: Sliced Wasserstein Distribution Matching (sorted 1D quantile projections)
    sw_dim: int = 32
    sw_num_slices: int = 32
    sw_num_quantiles: int = 32
    sw_lambda: float = 1.0
    weight_sw: float = 0.0
    # Non-linear Kernel Ridge Regression (KRR) options:
    krr_kernel: str = "linear"  # "linear" | "rbf" | "poly" | "cosine"
    krr_gamma: float | None = None  # None = 1 / dim
    krr_degree: int = 2
    krr_coef0: float = 1.0
    # In-Episode LOO Stacking / Reliability Weighting options:
    loo_gamma: float = 2.0
    loo_floor: float = 0.50











class TrainingFreeClassifier:
    """Zero-parameter in-context binary classifier. Nothing here is learned."""

    def __init__(self, config: TrainingFreeConfig | None = None) -> None:
        self.config = config or TrainingFreeConfig()

    # ---- 1. basis ---------------------------------------------------------
    def within_slide_basis(
        self,
        context_bags: Sequence[torch.Tensor],
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
            self.config.sketch_dim,
            target_device,
        )

    # ---- 2. descriptors ---------------------------------------------------
    def _descriptor(self, bag: torch.Tensor, basis: torch.Tensor, triangle) -> torch.Tensor:
        values = bag.to(basis.device).float()
        mean = values.mean(dim=0, keepdim=True)
        projected = (values - mean) @ basis
        covariance = (projected.T @ projected) / values.shape[0]
        return torch.cat((covariance[triangle[0], triangle[1]], mean.squeeze(0)))

    # ---- 3. CV ------------------------------------------------------------
    def _cv_logits(self, context: torch.Tensor, labels: torch.Tensor, query: torch.Tensor, split, return_loo: bool = False):
        context, query = _standardise_blocks(context.float(), query.float(), split)
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
        dual = _solve_ridge(gram, centred_targets, self.config.ridge_lambda)
        coefficients = design.T @ dual
        intercept = target_mean - feature_mean @ coefficients
        logits = (query @ coefficients + intercept) * self.config.ridge_scale

        if return_loo:
            size = gram.shape[0]
            identity = torch.eye(size, device=gram.device, dtype=torch.float32)
            inv_gram = _solve_ridge(gram, identity, self.config.ridge_lambda)
            H = gram @ inv_gram
            h_diag = H.diagonal().clamp(0.0, 0.99)

            ctx_logits = (context @ coefficients + intercept) * self.config.ridge_scale
            ctx_m = ctx_logits[:, 1] - ctx_logits[:, 0]
            y_diff = torch.where(labels == 1, 1.0, -1.0).to(ctx_m.dtype).to(ctx_m.device)
            loo_m = (ctx_m - h_diag * y_diff) / (1.0 - h_diag).clamp_min(1e-4)
            return logits, loo_m

        return logits




    # ---- 4. DD ------------------------------------------------------------
    def _dd_features(self, context_cov: torch.Tensor, labels: torch.Tensor, query_cov: torch.Tensor):
        config = self.config
        context_cov = context_cov.float()
        query_cov = query_cov.float()
        labels = labels.long()
        means = torch.stack([context_cov[labels == c].mean(dim=0) for c in range(2)])
        delta = means[1] - means[0]
        pooled = context_cov.mean(dim=0)
        trace_scale = pooled.diagonal().mean().clamp_min(config.dd_eps)
        identity = torch.eye(config.sketch_dim, device=pooled.device, dtype=pooled.dtype)
        shrunk = (1.0 - config.dd_shrinkage) * pooled + config.dd_shrinkage * trace_scale * identity
        # eigh here is never differentiated -- docs SS100 / `_dd_direction`: the
        # backward carries 1/(lambda_i - lambda_j) and the direction is a hard
        # argmax. Training-free, so the constraint is free to honour.
        values, vectors = torch.linalg.eigh(shrunk)
        whitening = (vectors * values.clamp_min(config.dd_eps).rsqrt()[None, :]) @ vectors.T
        operator = whitening @ delta @ whitening
        eigenvalues, eigenvectors = torch.linalg.eigh(operator)
        direction = whitening @ eigenvectors[:, eigenvalues.abs().argmax()]

        def log_variance(covariances):
            return torch.einsum("d,bdk,k->b", direction, covariances, direction).clamp_min(
                config.dd_eps
            ).log()

        context_feature = log_variance(context_cov)
        centre = context_feature.mean()
        scale = (context_feature - centre).square().mean().sqrt().clamp_min(config.dd_eps)
        context_feature = (context_feature - centre) / scale
        query_feature = (log_variance(query_cov) - centre) / scale
        prototypes = torch.stack([context_feature[labels == c].mean() for c in range(2)])
        dispersions = torch.stack([
            (context_feature[labels == c] - prototypes[c]).square().mean().clamp_min(config.dd_eps)
            for c in range(2)
        ])
        distances = (
            (query_feature[:, None] - prototypes[None, :]).square()
            / dispersions[None, :]
        )
        if config.dd_readout == "distance":
            # Logits, not distances: -d_c is the class-c score, so a large
            # distance is evidence AGAINST that class and the head weighs the
            # difference (dd1 - dd0) positively, like CV and CT (SS183).
            return -distances
        if config.dd_readout == "ordered_typicality":
            margin = ordered_typicality_margin(
                query_feature,
                prototypes,
                dispersions,
                config.dd_eps,
                config.dd_separation_floor,
            )
            # Logits, like CT: the pair is (-margin/2, +margin/2) so its
            # difference IS the class-1-positive margin, consumed by the head
            # at a positive weight (SS183).
            return torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)
        raise ValueError(
            'dd_readout must be "distance" or "ordered_typicality", '
            f"got {config.dd_readout!r}"
        )

    # ---- 5. CT ------------------------------------------------------------
    def _ct_features(self, context_bags, labels, query_bags, basis=None):
        """Two-token abundance readout, delegated to `ct_readout` (docs SS148).

        Steps 1-5 (sample, standardise, farthest-point tokens, soft assign, per-bag
        average) live in `ct_readout.ct_abundance` so that the readout experiments
        cannot accidentally differ from this path in the REPRESENTATION -- only in
        step 6-7. `mode="extreme"` is today's behaviour and stays the default, so
        v107's output is unchanged; `tests/test_training_free.py` pins that against
        the lineage and `tests/test_ct_readout.py` pins the refactor itself.
        """
        config = self.config
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

    # ---- 6. head ----------------------------------------------------------
    def margins(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Signed score per query bag. Positive favours class 1."""
        with torch.cuda.amp.autocast(enabled=False):
            config = self.config
            basis = self.within_slide_basis(context_bags, device=context_labels.device)
            triangle = torch.triu_indices(config.sketch_dim, config.sketch_dim, device=basis.device)
            context = torch.stack([self._descriptor(b, basis, triangle) for b in context_bags])
            query = torch.stack([self._descriptor(b, basis, triangle) for b in query_bags])

            covariance_dim = triangle.shape[1]
            # CV sees only the blocks `cv_blocks` names; DD below always gets the full
            # triangle out of `context`/`query` (SS156-1).
            if config.cv_blocks == "cov+mean":
                cv_context, cv_query = context, query
                split = (covariance_dim, context.shape[-1] - covariance_dim)
            elif config.cv_blocks == "offdiag":
                keep = triangle[0] != triangle[1]
                cv_context = context[..., :covariance_dim][..., keep]
                cv_query = query[..., :covariance_dim][..., keep]
                # One surviving block, so the whole descriptor is standardised together.
                split = (cv_context.shape[-1], 0)
            else:
                raise ValueError(
                    f'cv_blocks must be "offdiag" or "cov+mean", got {config.cv_blocks!r}'
                )
            return_loo = config.aggregation.startswith("context_loo")

            if return_loo:
                cv_res = self._cv_logits(cv_context, context_labels, cv_query, split, return_loo=True)
                cv, loo_cv = cv_res[0], cv_res[1]
            else:
                cv = self._cv_logits(cv_context, context_labels, cv_query, split)
                loo_cv = None
            m_cv = cv[:, 1] - cv[:, 0]


            def to_matrices(descriptors):
                flat = descriptors[..., : triangle.shape[1]]
                matrices = flat.new_zeros(flat.shape[0], config.sketch_dim, config.sketch_dim)
                matrices[..., triangle[0], triangle[1]] = flat
                matrices[..., triangle[1], triangle[0]] = flat
                return matrices

            if config.weight_dd != 0.0:
                dd = self._dd_features(to_matrices(context), context_labels, to_matrices(query))
                m_dd = dd[:, 1] - dd[:, 0]
                if return_loo:
                    ctx_dd = self._dd_features(to_matrices(context), context_labels, to_matrices(context))
                    loo_dd = ctx_dd[:, 1] - ctx_dd[:, 0]
                else:
                    loo_dd = None
            else:
                m_dd = None
                loo_dd = None

            if config.weight_ct != 0.0:
                q0, q1 = self._ct_features(context_bags, context_labels, query_bags, basis)
                m_ct = q1 - q0
                if return_loo:
                    cq0, cq1 = self._ct_features(context_bags, context_labels, context_bags, basis)
                    loo_ct = cq1 - cq0
                else:
                    loo_ct = None
            else:
                m_ct = None
                loo_ct = None

            if config.weight_bm != 0.0:
                bm_res = self._bm_features(context_bags, context_labels, query_bags, basis, return_loo=return_loo)
                m_bm, loo_bm = (bm_res[0], bm_res[1]) if return_loo else (bm_res, None)
            else:
                m_bm, loo_bm = None, None

            if config.weight_bd != 0.0:
                m_bd = self._bd_features(to_matrices(context), context_labels, to_matrices(query))
                if return_loo:
                    loo_bd = self._bd_features(to_matrices(context), context_labels, to_matrices(context))
                else:
                    loo_bd = None
            else:
                m_bd, loo_bd = None, None

            if config.weight_qa != 0.0:
                qa_res = self._qa_features(context_bags, context_labels, query_bags, basis, return_loo=return_loo)
                m_qa, loo_qa = (qa_res[0], qa_res[1]) if return_loo else (qa_res, None)
            else:
                m_qa, loo_qa = None, None

            if config.weight_ds != 0.0:
                ds_res = self._ds_features(context_bags, context_labels, query_bags, basis, return_loo=return_loo)
                m_ds, loo_ds = (ds_res[0], ds_res[1]) if return_loo else (ds_res, None)
            else:
                m_ds, loo_ds = None, None

            m_lr = self._lr_features(context_bags, context_labels, query_bags, basis) if config.weight_lr != 0.0 else None

            if config.weight_de != 0.0:
                de_res = self._de_features(context_bags, context_labels, query_bags, basis, return_loo=return_loo)
                m_de, loo_de = (de_res[0], de_res[1]) if return_loo else (de_res, None)
            else:
                m_de, loo_de = None, None

            if config.weight_sw != 0.0:
                sw_res = self._sw_features(context_bags, context_labels, query_bags, basis, return_loo=return_loo)
                m_sw, loo_sw = (sw_res[0], sw_res[1]) if return_loo else (sw_res, None)
            else:
                m_sw, loo_sw = None, None

            if config.aggregation == "linear":
                total_margin = config.weight_cv * m_cv
                if m_dd is not None:
                    total_margin = total_margin + config.weight_dd * m_dd
                if m_ct is not None:
                    total_margin = total_margin + config.weight_ct * m_ct
                if m_bm is not None:
                    total_margin = total_margin + config.weight_bm * m_bm
                if m_bd is not None:
                    total_margin = total_margin + config.weight_bd * m_bd
                if m_qa is not None:
                    total_margin = total_margin + config.weight_qa * m_qa
                if m_ds is not None:
                    total_margin = total_margin + config.weight_ds * m_ds
                if m_lr is not None:
                    total_margin = total_margin + config.weight_lr * m_lr
                if m_de is not None:
                    total_margin = total_margin + config.weight_de * m_de
                if m_sw is not None:
                    total_margin = total_margin + config.weight_sw * m_sw
                return total_margin
            elif config.aggregation == "soft_voting":
                active_pairs = []
                if config.weight_cv != 0.0:
                    active_pairs.append((config.weight_cv, m_cv))
                if m_dd is not None and config.weight_dd != 0.0:
                    active_pairs.append((config.weight_dd, m_dd))
                if m_ct is not None and config.weight_ct != 0.0:
                    active_pairs.append((config.weight_ct, m_ct))
                if m_bm is not None and config.weight_bm != 0.0:
                    active_pairs.append((config.weight_bm, m_bm))
                if m_bd is not None and config.weight_bd != 0.0:
                    active_pairs.append((config.weight_bd, m_bd))
                if m_qa is not None and config.weight_qa != 0.0:
                    active_pairs.append((config.weight_qa, m_qa))
                if m_ds is not None and config.weight_ds != 0.0:
                    active_pairs.append((config.weight_ds, m_ds))
                if m_lr is not None and config.weight_lr != 0.0:
                    active_pairs.append((config.weight_lr, m_lr))
                if m_de is not None and config.weight_de != 0.0:
                    active_pairs.append((config.weight_de, m_de))
                if m_sw is not None and config.weight_sw != 0.0:
                    active_pairs.append((config.weight_sw, m_sw))

                if not active_pairs:
                    return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

                total_weight = sum(w for w, _ in active_pairs)
                avg_prob = sum(w * torch.sigmoid(m) for w, m in active_pairs) / total_weight
                clamped = avg_prob.clamp(1e-7, 1.0 - 1e-7)
                return torch.log(clamped / (1.0 - clamped))
            elif config.aggregation.startswith("context_loo"):
                branch_pool = []
                if config.weight_cv != 0.0:
                    branch_pool.append((m_cv, loo_cv))
                if m_dd is not None and config.weight_dd != 0.0:
                    branch_pool.append((m_dd, loo_dd))
                if m_ct is not None and config.weight_ct != 0.0:
                    branch_pool.append((m_ct, loo_ct))
                if m_bm is not None and config.weight_bm != 0.0:
                    branch_pool.append((m_bm, loo_bm))
                if m_bd is not None and config.weight_bd != 0.0:
                    branch_pool.append((m_bd, loo_bd))
                if m_qa is not None and config.weight_qa != 0.0:
                    branch_pool.append((m_qa, loo_qa))
                if m_ds is not None and config.weight_ds != 0.0:
                    branch_pool.append((m_ds, loo_ds))
                if m_de is not None and config.weight_de != 0.0:
                    branch_pool.append((m_de, loo_de))
                if m_sw is not None and config.weight_sw != 0.0:
                    branch_pool.append((m_sw, loo_sw))

                if not branch_pool:
                    return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)

                r_list = []
                for q_m, l_m in branch_pool:
                    if l_m is not None:
                        r = _fast_context_auroc(l_m, context_labels)
                    else:
                        r = 0.50
                    r_list.append(r)

                gamma = getattr(config, "loo_gamma", 2.0)
                floor = getattr(config, "loo_floor", 0.50)
                q_list = [max(0.0, r - floor) ** gamma for r in r_list]
                sum_q = sum(q_list)
                if sum_q > 0:
                    weights = [q / sum_q for q in q_list]
                else:
                    weights = [1.0 / len(branch_pool)] * len(branch_pool)

                avg_prob = sum(w * torch.sigmoid(q_m) for w, (q_m, _) in zip(weights, branch_pool))
                clamped = avg_prob.clamp(1e-7, 1.0 - 1e-7)
                return torch.log(clamped / (1.0 - clamped))
            elif config.aggregation == "trimmed_mean":

                active_probs = []
                if config.weight_cv != 0.0:
                    active_probs.append(torch.sigmoid(m_cv))
                if m_dd is not None and config.weight_dd != 0.0:
                    active_probs.append(torch.sigmoid(m_dd))
                if m_ct is not None and config.weight_ct != 0.0:
                    active_probs.append(torch.sigmoid(m_ct))
                if m_bm is not None and config.weight_bm != 0.0:
                    active_probs.append(torch.sigmoid(m_bm))
                if m_bd is not None and config.weight_bd != 0.0:
                    active_probs.append(torch.sigmoid(m_bd))
                if m_qa is not None and config.weight_qa != 0.0:
                    active_probs.append(torch.sigmoid(m_qa))
                if m_ds is not None and config.weight_ds != 0.0:
                    active_probs.append(torch.sigmoid(m_ds))
                if m_lr is not None and config.weight_lr != 0.0:
                    active_probs.append(torch.sigmoid(m_lr))
                if m_de is not None and config.weight_de != 0.0:
                    active_probs.append(torch.sigmoid(m_de))
                if m_sw is not None and config.weight_sw != 0.0:
                    active_probs.append(torch.sigmoid(m_sw))

                if not active_probs:
                    return torch.zeros(cv.shape[0], device=cv.device, dtype=cv.dtype)


                if len(active_probs) >= 3:
                    stacked = torch.stack(active_probs, dim=-1)
                    sum_p = torch.sum(stacked, dim=-1)
                    min_p = torch.min(stacked, dim=-1).values
                    max_p = torch.max(stacked, dim=-1).values
                    trimmed_avg = (sum_p - min_p - max_p) / float(len(active_probs) - 2)
                    clamped = trimmed_avg.clamp(1e-7, 1.0 - 1e-7)
                    return torch.log(clamped / (1.0 - clamped))
                else:
                    stacked = torch.stack(active_probs, dim=-1)
                    avg_p = torch.mean(stacked, dim=-1)
                    clamped = avg_p.clamp(1e-7, 1.0 - 1e-7)
                    return torch.log(clamped / (1.0 - clamped))
            else:
                raise ValueError(f'aggregation must be "soft_voting", "trimmed_mean", or "linear", got {config.aggregation!r}')







    def _bd_features(
        self,
        context_cov: torch.Tensor,
        labels: torch.Tensor,
        query_cov: torch.Tensor,
    ) -> torch.Tensor:
        """Bag Dispersion (BD): spectral entropy or log-trace of projected covariance."""
        config = self.config
        dim = min(config.bd_dim, context_cov.shape[-1])
        ctx_sub = context_cov[:, :dim, :dim]
        qry_sub = query_cov[:, :dim, :dim]

        if config.bd_metric == "entropy":
            ctx_eig = torch.linalg.eigvalsh(ctx_sub.float()).clamp_min(config.bd_eps)
            qry_eig = torch.linalg.eigvalsh(qry_sub.float()).clamp_min(config.bd_eps)
            ctx_p = ctx_eig / ctx_eig.sum(dim=-1, keepdim=True).clamp_min(config.bd_eps)
            qry_p = qry_eig / qry_eig.sum(dim=-1, keepdim=True).clamp_min(config.bd_eps)

            ctx_v = -(ctx_p * torch.log(ctx_p.clamp_min(config.bd_eps))).sum(dim=-1)
            qry_v = -(qry_p * torch.log(qry_p.clamp_min(config.bd_eps))).sum(dim=-1)
            if dim > 1:
                log_dim = torch.log(torch.tensor(float(dim), device=ctx_v.device, dtype=ctx_v.dtype))
                ctx_v = ctx_v / log_dim
                qry_v = qry_v / log_dim
        elif config.bd_metric == "trace":
            ctx_trace = ctx_sub.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(config.bd_eps)
            qry_trace = qry_sub.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(config.bd_eps)
            ctx_v = ctx_trace.log()
            qry_v = qry_trace.log()
        else:
            raise ValueError(f"Unknown bd_metric: {config.bd_metric!r}")

        labels = labels.long()
        prototypes = torch.stack([ctx_v[labels == c].mean() for c in range(2)])
        dispersions = torch.stack([
            (ctx_v[labels == c] - prototypes[c]).square().mean().clamp_min(config.bd_eps)
            for c in range(2)
        ])


        if config.bd_readout == "ordered_typicality":
            margin = ordered_typicality_margin(
                qry_v,
                prototypes,
                dispersions,
                config.bd_eps,
                config.bd_separation_floor,
            )
            return margin
        elif config.bd_readout == "ridge":
            centre = ctx_v.mean()
            scale = (ctx_v - centre).square().mean().sqrt().clamp_min(config.bd_eps)
            std_ctx = ((ctx_v - centre) / scale).unsqueeze(-1)
            std_qry = ((qry_v - centre) / scale).unsqueeze(-1)

            targets = torch.nn.functional.one_hot(labels, 2).float()
            counts = torch.bincount(labels, minlength=2)
            if bool((counts == 0).any()):
                raise ValueError("Every class must occur in the context set.")
            weight = counts.float().reciprocal()[labels]
            total = weight.sum().clamp_min(1e-12)
            feat_mean = (weight[:, None] * std_ctx).sum(0, keepdim=True) / total
            tgt_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
            root = weight.sqrt()[:, None]

            design = (std_ctx - feat_mean) * root
            centred_targets = (targets - tgt_mean) * root

            dual = _solve_ridge(design @ design.T, centred_targets, config.bd_lambda)
            coefficients = design.T @ dual
            intercept = tgt_mean - feat_mean @ coefficients
            logits = std_qry @ coefficients + intercept
            return logits[:, 1] - logits[:, 0]
        else:
            raise ValueError(f"Unknown bd_readout: {config.bd_readout!r}")

    def _bm_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Projected bag-mean in leading subspace with class-balanced kernel ridge."""
        config = self.config
        dim = min(config.bm_dim, basis.shape[1])
        bm_basis = basis[:, :dim].to(dtype=torch.float32)
        ctx_means = torch.stack([b.float().mean(dim=0).to(bm_basis.device) for b in context_bags]) @ bm_basis
        qry_means = torch.stack([b.float().mean(dim=0).to(bm_basis.device) for b in query_bags]) @ bm_basis

        kernel = getattr(config, "bm_kernel", getattr(config, "krr_kernel", "linear"))
        gamma = getattr(config, "krr_gamma", None)
        degree = getattr(config, "krr_degree", 2)
        coef0 = getattr(config, "krr_coef0", 1.0)
        return _solve_kernel_ridge(
            ctx_means, context_labels, qry_means,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            reg_lambda=config.bm_lambda,
            return_loo=return_loo,
        )

    def _qa_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Quantile & Extremum Evidence (QA): multi-quantile features of projected cells with class-balanced kernel ridge."""
        config = self.config
        dim = min(config.qa_dim, basis.shape[1])
        qa_basis = basis[:, :dim].to(dtype=torch.float32)
        quantiles = torch.tensor(config.qa_quantiles, device=qa_basis.device, dtype=torch.float32)

        def extract_quantiles(bag: torch.Tensor) -> torch.Tensor:
            z = bag.float().to(qa_basis.device) @ qa_basis
            q = torch.quantile(z, quantiles, dim=0)  # (n_quantiles, dim)
            return q.flatten()

        ctx_feats = torch.stack([extract_quantiles(b) for b in context_bags])
        qry_feats = torch.stack([extract_quantiles(b) for b in query_bags])

        kernel = getattr(config, "qa_kernel", getattr(config, "krr_kernel", "linear"))
        gamma = getattr(config, "krr_gamma", None)
        degree = getattr(config, "krr_degree", 2)
        coef0 = getattr(config, "krr_coef0", 1.0)
        return _solve_kernel_ridge(
            ctx_feats, context_labels, qry_feats,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            reg_lambda=config.qa_lambda,
            return_loo=return_loo,
        )

    def _ds_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """In-Context Salience Denoising (DS): class-contrastive cluster salience weighting for noise patch suppression."""
        config = self.config
        dim = min(config.ds_dim, basis.shape[1])
        ds_basis = basis[:, :dim].to(dtype=torch.float32)
        device = ds_basis.device
        labels = context_labels.long().to(device)

        # 1. Project all context and query bags to PCA subspace
        ctx_proj = [b.float().to(device) @ ds_basis for b in context_bags]
        qry_proj = [b.float().to(device) @ ds_basis for b in query_bags]

        # 2. Select K cluster centroids from sampled context cells
        sampled_cells = []
        for p in ctx_proj:
            n_cells = p.shape[0]
            if n_cells > 0:
                idx = torch.linspace(0, n_cells - 1, min(n_cells, 64), device=device).long()
                sampled_cells.append(p[idx])
        all_cells = torch.cat(sampled_cells, dim=0) if sampled_cells else torch.zeros(1, dim, device=device)

        K = min(config.ds_tokens, all_cells.shape[0])
        if all_cells.shape[0] > K:
            stride = all_cells.shape[0] / K
            centroids = all_cells[(torch.arange(K, device=device) * stride).long()]
        else:
            centroids = all_cells

        centroids = torch.nn.functional.normalize(centroids, dim=-1)

        # 3. Soft cluster assignments and slide abundances
        def get_assignments(proj_bags):
            abundances = []
            patch_assignments = []
            for p in proj_bags:
                p_norm = torch.nn.functional.normalize(p, dim=-1)
                sim = p_norm @ centroids.T  # (N_i, K)
                soft_p = torch.nn.functional.softmax(sim * 5.0, dim=-1)  # (N_i, K)
                a = soft_p.mean(dim=0)  # (K,)
                abundances.append(a)
                patch_assignments.append(soft_p)
            return torch.stack(abundances), patch_assignments

        ctx_abundances, ctx_assignments = get_assignments(ctx_proj)
        qry_abundances, qry_assignments = get_assignments(qry_proj)

        # 4. In-context Class Salience Log-Odds
        eps = 1e-5
        mask1 = (labels == 1)
        mask0 = (labels == 0)
        a1 = ctx_abundances[mask1].mean(dim=0) if mask1.any() else ctx_abundances.mean(dim=0)
        a0 = ctx_abundances[mask0].mean(dim=0) if mask0.any() else ctx_abundances.mean(dim=0)

        s = torch.log((a1 + eps) / (a0 + eps))  # (K,)
        s_abs = s.abs()  # Salience magnitude

        # 5. Denoised bag mean extraction
        def extract_denoised_mean(proj_bags, assignments):
            feats = []
            temp = config.ds_temperature
            for p, soft_p in zip(proj_bags, assignments):
                u = soft_p @ s_abs  # (N_i,)
                u_std = u.std().clamp_min(1e-6)
                w = torch.nn.functional.softmax(temp * (u - u.mean()) / u_std, dim=0)  # (N_i,)
                z_denoised = (w.unsqueeze(-1) * p).sum(dim=0)  # (dim,)
                feats.append(z_denoised)
            return torch.stack(feats)

        ctx_feats = extract_denoised_mean(ctx_proj, ctx_assignments)
        qry_feats = extract_denoised_mean(qry_proj, qry_assignments)

        # 6. Class-balanced kernel ridge
        kernel = getattr(config, "ds_kernel", getattr(config, "krr_kernel", "linear"))
        gamma = getattr(config, "krr_gamma", None)
        degree = getattr(config, "krr_degree", 2)
        coef0 = getattr(config, "krr_coef0", 1.0)
        return _solve_kernel_ridge(
            ctx_feats, context_labels, qry_feats,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            reg_lambda=config.ds_lambda,
            return_loo=return_loo,
        )

    def _lr_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
    ) -> torch.Tensor:
        """Direct In-Context Patch Likelihood Ratio + Top-K MIL Extreme Pooling with class-balanced ridge."""
        config = self.config
        dim = min(config.lr_dim, basis.shape[1])
        lr_basis = basis[:, :dim].to(dtype=torch.float32)
        device = lr_basis.device
        labels = context_labels.long().to(device)

        # 1. Project all context and query bags to PCA subspace
        ctx_proj = [b.float().to(device) @ lr_basis for b in context_bags]
        qry_proj = [b.float().to(device) @ lr_basis for b in query_bags]

        # 2. Build Class 0 and Class 1 patch memory banks
        bank_0, bank_1 = [], []
        for p, y in zip(ctx_proj, labels):
            n_c = p.shape[0]
            if n_c == 0:
                continue
            n_sample = min(n_c, config.lr_patches_per_ctx)
            idx = torch.linspace(0, n_c - 1, n_sample, device=device).long()
            sampled = p[idx]
            if y == 1:
                bank_1.append(sampled)
            else:
                bank_0.append(sampled)

        if not bank_0 or not bank_1:
            return torch.zeros(len(query_bags), device=device)

        P0 = torch.cat(bank_0, dim=0)
        P1 = torch.cat(bank_1, dim=0)

        P0_norm = torch.nn.functional.normalize(P0, dim=-1)
        P1_norm = torch.nn.functional.normalize(P1, dim=-1)
        tau = config.lr_tau

        # 3. Compute Patch-level log-odds likelihood ratio and extract Top-K extreme instance features
        def get_slide_lr_features(proj_bags):
            feats = []
            for bag in proj_bags:
                n_c = bag.shape[0]
                if n_c == 0:
                    feats.append(torch.zeros(dim + 1, device=device))
                    continue
                bag_norm = torch.nn.functional.normalize(bag, dim=-1)
                sim1 = bag_norm @ P1_norm.T  # (N_i, |P1|)
                sim0 = bag_norm @ P0_norm.T  # (N_i, |P0|)

                score1 = torch.logsumexp(sim1 * tau, dim=-1) - torch.log(torch.tensor(P1_norm.shape[0], dtype=torch.float32, device=device))
                score0 = torch.logsumexp(sim0 * tau, dim=-1) - torch.log(torch.tensor(P0_norm.shape[0], dtype=torch.float32, device=device))

                lr = score1 - score0  # (N_i,)

                k = max(config.lr_topk_min, min(config.lr_topk_max, int(n_c * config.lr_topk_fraction)))
                k = min(k, n_c)

                topk_vals, topk_idx = torch.topk(lr, k=k, largest=True)
                botk_vals, botk_idx = torch.topk(lr, k=k, largest=False)

                z_plus = bag[topk_idx].mean(dim=0)
                z_minus = bag[botk_idx].mean(dim=0)

                delta_z = z_plus - z_minus
                e_scalar = 0.5 * (topk_vals.mean() + botk_vals.mean())

                v_i = torch.cat([delta_z, e_scalar.unsqueeze(0)], dim=-1)
                feats.append(v_i)
            return torch.stack(feats)

        ctx_feats = get_slide_lr_features(ctx_proj)
        qry_feats = get_slide_lr_features(qry_proj)

        # 4. Class-balanced linear dual ridge solve
        return _solve_kernel_ridge(
            ctx_feats, labels, qry_feats,
            kernel="linear",
            reg_lambda=config.lr_lambda,
        )

    def _de_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """In-Subspace Dual Extreme Instance MIL (DE):
        Projects patches to 32D within-slide PCA space, scores patches along class
        centroid contrast direction, extracts Top-K and Bottom-K extreme patches,
        and solves Class-balanced Dual Ridge on the difference descriptor.
        """
        config = self.config
        dim = min(config.de_dim, basis.shape[1])
        de_basis = basis[:, :dim].to(dtype=torch.float32)
        device = de_basis.device
        labels = context_labels.long().to(device)

        # 1. Project all context and query bags to 32D PCA space
        ctx_proj = [b.float().to(device) @ de_basis for b in context_bags]
        qry_proj = [b.float().to(device) @ de_basis for b in query_bags]

        # 2. Compute Class 0 and Class 1 centroids from bag means
        ctx_means = torch.stack([p.mean(dim=0) for p in ctx_proj])
        if (labels == 0).sum() == 0 or (labels == 1).sum() == 0:
            zero_q = torch.zeros(len(query_bags), device=device)
            return (zero_q, torch.zeros(len(context_bags), device=device)) if return_loo else zero_q

        mu0 = ctx_means[labels == 0].mean(dim=0)
        mu1 = ctx_means[labels == 1].mean(dim=0)
        w_contrast = mu1 - mu0
        w_norm = w_contrast.norm().clamp_min(1e-12)
        w_dir = w_contrast / w_norm

        # 3. Extract Dual Extreme difference descriptor per bag
        def extract_de_vector(p_bag):
            n_c = p_bag.shape[0]
            if n_c == 0:
                return torch.zeros(dim + 1, device=device)
            scores = p_bag @ w_dir  # (N_i,)
            k = max(config.de_topk_min, min(config.de_topk_max, int(n_c * config.de_topk_fraction)))
            k = min(k, n_c)

            topk_vals, topk_idx = torch.topk(scores, k=k, largest=True)
            botk_vals, botk_idx = torch.topk(scores, k=k, largest=False)

            z_plus = p_bag[topk_idx].mean(dim=0)
            z_minus = p_bag[botk_idx].mean(dim=0)

            delta_z = z_plus - z_minus  # (dim,)
            score_diff = 0.5 * (topk_vals.mean() + botk_vals.mean())  # (1,)

            return torch.cat([delta_z, score_diff.unsqueeze(0)], dim=-1)

        ctx_feats = torch.stack([extract_de_vector(p) for p in ctx_proj])
        qry_feats = torch.stack([extract_de_vector(p) for p in qry_proj])

        # 4. Class-balanced linear dual ridge solve
        return _solve_kernel_ridge(
            ctx_feats, labels, qry_feats,
            kernel="linear",
            reg_lambda=config.de_lambda,
            return_loo=return_loo,
        )

    def _sw_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Sliced Wasserstein Distribution Matching (SW):
        Projects patches to 32D PCA space, projects onto L deterministic orthogonal
        spherical slice directions, computes M 1D quantiles per slice, and solves
        Class-balanced Dual Ridge on the flattened (L * M) Sliced Wasserstein embedding.
        """
        config = self.config
        dim = min(config.sw_dim, basis.shape[1])
        sw_basis = basis[:, :dim].to(dtype=torch.float32)
        device = sw_basis.device
        labels = context_labels.long().to(device)

        # 1. Deterministic orthogonal slice directions on S^{dim-1}
        num_slices = config.sw_num_slices
        g = torch.Generator(device="cpu").manual_seed(42)
        rand_dirs = torch.randn(dim, num_slices, generator=g, dtype=torch.float32)
        q_dirs, _ = torch.linalg.qr(rand_dirs)
        slice_dirs = q_dirs.to(device=device)  # (dim, num_slices)

        # 2. Quantile levels
        num_quantiles = config.sw_num_quantiles
        q_levels = torch.linspace(0.5 / num_quantiles, 1.0 - 0.5 / num_quantiles, num_quantiles, device=device)

        # 3. Project each bag onto slices and extract 1D quantiles
        def extract_sw_profile(bag):
            proj = bag.float().to(device) @ sw_basis  # (N_i, dim)
            n_c = proj.shape[0]
            if n_c == 0:
                return torch.zeros(num_slices * num_quantiles, device=device)

            slices = proj @ slice_dirs  # (N_i, num_slices)
            sorted_slices, _ = torch.sort(slices, dim=0)

            indices = (q_levels * (n_c - 1)).clamp(0, n_c - 1)
            low_idx = indices.floor().long()
            high_idx = indices.ceil().long()
            weights = (indices - low_idx.float())[:, None]

            low_vals = sorted_slices[low_idx, :]   # (num_quantiles, num_slices)
            high_vals = sorted_slices[high_idx, :] # (num_quantiles, num_slices)
            quantiles = (1.0 - weights) * low_vals + weights * high_vals

            return quantiles.flatten()

        ctx_feats = torch.stack([extract_sw_profile(b) for b in context_bags])
        qry_feats = torch.stack([extract_sw_profile(b) for b in query_bags])

        # 4. Class-balanced linear dual ridge solve
        return _solve_kernel_ridge(
            ctx_feats, labels, qry_feats,
            kernel="linear",
            reg_lambda=config.sw_lambda,
            return_loo=return_loo,
        )


    def predict_proba(self, context_bags, context_labels, query_bags) -> torch.Tensor:
        """P(class 1) per query bag."""
        return torch.sigmoid(self.margins(context_bags, context_labels, query_bags))



