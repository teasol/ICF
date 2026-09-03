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

from src.models.common.solvers import (
    solve_ridge as _solve_ridge,
    kernel_matrix as _kernel_matrix,
    fast_context_auroc as _fast_context_auroc,
    solve_kernel_ridge as _solve_kernel_ridge,
    standardise as _standardise,
    standardise_blocks as _standardise_blocks,
)
from src.models.common.basis import (
    within_slide_basis as _within_slide_basis_fn,
    extract_bag_descriptor,
    to_matrices as _to_matrices_fn,
)
from src.models.branches.cv import cv_logits
from src.models.branches.dd import dd_features
from src.models.branches.ct import ct_features
from src.models.branches.bm import bm_features
from src.models.branches.bd import bd_features
from src.models.branches.qa import qa_features
from src.models.branches.ds import ds_features
from src.models.branches.experimental.lr import lr_features
from src.models.branches.experimental.de import de_features
from src.models.branches.experimental.sw import sw_features
from src.models.aggregations.voting import (
    linear_aggregation,
    soft_voting,
    trimmed_mean as trimmed_mean_aggregation,
    context_loo_stacking,
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
        return _within_slide_basis_fn(context_bags, self.config.sketch_dim, device)

    # ---- 2. descriptors ---------------------------------------------------
    def _descriptor(self, bag: torch.Tensor, basis: torch.Tensor, triangle) -> torch.Tensor:
        return extract_bag_descriptor(bag, basis, triangle)

    # ---- 3. CV ------------------------------------------------------------
    def _cv_logits(self, context: torch.Tensor, labels: torch.Tensor, query: torch.Tensor, split, return_loo: bool = False):
        return cv_logits(self.config, context, labels, query, split, return_loo=return_loo)

    # ---- 4. DD ------------------------------------------------------------
    def _dd_features(self, context_cov: torch.Tensor, labels: torch.Tensor, query_cov: torch.Tensor):
        return dd_features(self.config, context_cov, labels, query_cov)

    # ---- 5. CT ------------------------------------------------------------
    def _ct_features(self, context_bags, labels, query_bags, basis=None):
        return ct_features(self.config, context_bags, labels, query_bags, basis)

    def _bd_features(
        self,
        context_cov: torch.Tensor,
        labels: torch.Tensor,
        query_cov: torch.Tensor,
    ) -> torch.Tensor:
        return bd_features(self.config, context_cov, labels, query_cov)

    def _bm_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return bm_features(self.config, context_bags, context_labels, query_bags, basis, return_loo=return_loo)

    def _qa_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return qa_features(self.config, context_bags, context_labels, query_bags, basis, return_loo=return_loo)

    def _ds_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return ds_features(self.config, context_bags, context_labels, query_bags, basis, return_loo=return_loo)

    def _lr_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
    ) -> torch.Tensor:
        return lr_features(self.config, context_bags, context_labels, query_bags, basis)

    def _de_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return de_features(self.config, context_bags, context_labels, query_bags, basis, return_loo=return_loo)

    def _sw_features(
        self,
        context_bags: Sequence[torch.Tensor],
        context_labels: torch.Tensor,
        query_bags: Sequence[torch.Tensor],
        basis: torch.Tensor,
        return_loo: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return sw_features(self.config, context_bags, context_labels, query_bags, basis, return_loo=return_loo)

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
                return _to_matrices_fn(descriptors, triangle, config.sketch_dim)

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
                return linear_aggregation(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw)
            elif config.aggregation == "soft_voting":
                return soft_voting(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw)
            elif config.aggregation.startswith("context_loo"):
                return context_loo_stacking(
                    config, cv, context_labels,
                    m_cv, loo_cv,
                    m_dd, loo_dd,
                    m_ct, loo_ct,
                    m_bm, loo_bm,
                    m_bd, loo_bd,
                    m_qa, loo_qa,
                    m_ds, loo_ds,
                    m_de, loo_de,
                    m_sw, loo_sw,
                )
            elif config.aggregation == "trimmed_mean":
                return trimmed_mean_aggregation(config, cv, m_cv, m_dd, m_ct, m_bm, m_bd, m_qa, m_ds, m_lr, m_de, m_sw)
            else:
                raise ValueError(f'aggregation must be "soft_voting", "trimmed_mean", or "linear", got {config.aggregation!r}')

    def predict_proba(self, context_bags, context_labels, query_bags) -> torch.Tensor:
        """P(class 1) per query bag."""
        return torch.sigmoid(self.margins(context_bags, context_labels, query_bags))
