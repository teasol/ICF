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
from src.models.config import TrainingFreeConfig


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
