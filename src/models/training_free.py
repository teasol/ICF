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
                normalised squared DISTANCES from each query to the two class
                prototypes. Distances, not logits — which is why the head weighs
                them negatively.
  5. CT         Deterministic farthest-point tokens over context cells (labels
                never enter the selection), each bag summarised by its soft
                abundance over those tokens, and the two most class-separating
                tokens read off.
  6. HEAD       margin = 1.442*(CV1-CV0) - 0.343*(D1-D0) + 0.286*(q1-q0)
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


def _solve_ridge(gram: torch.Tensor, targets: torch.Tensor, penalty: float) -> torch.Tensor:
    """Solve (gram + penalty*I) x = targets, adding jitter only if it fails.

    Mirrors `solve_ridge_system`'s contract from the lineage: Cholesky first,
    escalating jitter on failure. The dual Gram matrix is context-bags square and
    can be near-singular when two slides are nearly identical.
    """
    size = gram.shape[-1]
    identity = torch.eye(size, device=gram.device, dtype=gram.dtype)
    jitter = 0.0
    for _ in range(6):
        try:
            factor = torch.linalg.cholesky(gram + (penalty + jitter) * identity)
            return torch.cholesky_solve(targets, factor)
        except RuntimeError:
            jitter = max(jitter * 10.0, 1e-6 * float(gram.diagonal().abs().mean()) + 1e-12)
    return torch.linalg.lstsq(gram + (penalty + jitter) * identity, targets).solution


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
    ct_abundance_cells_per_bag: int | None | str = None
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
    # SS137-3: CV : DD : CT. Sign of the DD term is required by DD returning
    # distances rather than logits (the ``distance`` readout); for
    # ``ordered_typicality`` the same negative sign cancels against the pseudo
    # -pair encoding (SS183) so this stays the coefficient on class-1-positive
    # margin. -0.343 was fitted against the old distance magnitude; SS183 (v112)
    # removes that unjustified scaling and takes the bounded margin at |weight|=1.
    weight_cv: float = 1.442
    weight_dd: float = -1.0
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



class TrainingFreeClassifier:
    """Zero-parameter in-context binary classifier. Nothing here is learned."""

    def __init__(self, config: TrainingFreeConfig | None = None) -> None:
        self.config = config or TrainingFreeConfig()

    # ---- 1. basis ---------------------------------------------------------
    def within_slide_basis(self, context_bags: Sequence[torch.Tensor]) -> torch.Tensor:
        """Top-K eigenvectors of the WITHIN-slide pooled covariance.

        Accumulated bag by bag: concatenating every context cell first is what
        SS62-3 identified as an eval OOM driver (~12 GB for a full-tile episode).
        """
        dim = context_bags[0].shape[-1]
        device = context_bags[0].device
        scatter = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        total = 0
        for bag in context_bags:
            values = bag.double()
            centred = values - values.mean(dim=0, keepdim=True)
            scatter += centred.T @ centred
            total += values.shape[0]
        _, vectors = torch.linalg.eigh(scatter / max(total, 1))
        return vectors[:, -self.config.sketch_dim:].flip(-1).float()

    # ---- 2. descriptors ---------------------------------------------------
    def _descriptor(self, bag: torch.Tensor, basis: torch.Tensor, triangle) -> torch.Tensor:
        values = bag.float()
        mean = values.mean(dim=0, keepdim=True)
        projected = (values - mean) @ basis
        covariance = (projected.T @ projected) / values.shape[0]
        return torch.cat((covariance[triangle[0], triangle[1]], mean.squeeze(0)))

    # ---- 3. CV ------------------------------------------------------------
    def _cv_logits(self, context: torch.Tensor, labels: torch.Tensor, query: torch.Tensor, split):
        context, query = _standardise_blocks(context.float(), query.float(), split)
        targets = torch.nn.functional.one_hot(labels.long(), 2).float()
        counts = torch.bincount(labels.long(), minlength=2)
        if bool((counts == 0).any()):
            raise ValueError("Every class must occur in the context set.")
        # Class-balanced: without this the ridge tracks prevalence, and real
        # tasks run 0.178 to 0.780 positive (docs SS115-2).
        weight = counts.float().reciprocal()[labels.long()]
        total = weight.sum().clamp_min(1e-12)
        feature_mean = (weight[:, None] * context).sum(0, keepdim=True) / total
        target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
        root = weight.sqrt()[:, None]
        design = (context - feature_mean) * root
        centred_targets = (targets - target_mean) * root
        dual = _solve_ridge(design @ design.T, centred_targets, self.config.ridge_lambda)
        coefficients = design.T @ dual
        intercept = target_mean - feature_mean @ coefficients
        return (query @ coefficients + intercept) * self.config.ridge_scale

    # ---- 4. DD ------------------------------------------------------------
    def _dd_features(self, context_cov: torch.Tensor, labels: torch.Tensor, query_cov: torch.Tensor):
        config = self.config
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
            # DISTANCES: small means close to that class, hence the head's
            # negative weight. This is the promoted v111 behaviour.
            return distances
        if config.dd_readout == "ordered_typicality":
            margin = ordered_typicality_margin(
                query_feature,
                prototypes,
                dispersions,
                config.dd_eps,
                config.dd_separation_floor,
            )
            # Keep the fixed head and its historical negative DD coefficient
            # unchanged: it consumes d1-d0, so this symmetric pseudo-pair makes
            # -weight*(d1-d0) equal +weight*margin.
            return torch.stack((0.5 * margin, -0.5 * margin), dim=-1)
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
        config = self.config
        basis = self.within_slide_basis(context_bags)
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
        cv = self._cv_logits(cv_context, context_labels, cv_query, split)

        def to_matrices(descriptors):
            flat = descriptors[..., : triangle.shape[1]]
            matrices = flat.new_zeros(flat.shape[0], config.sketch_dim, config.sketch_dim)
            matrices[..., triangle[0], triangle[1]] = flat
            matrices[..., triangle[1], triangle[0]] = flat
            return matrices

        dd = self._dd_features(to_matrices(context), context_labels, to_matrices(query))
        q0, q1 = self._ct_features(context_bags, context_labels, query_bags, basis)
        return (
            config.weight_cv * (cv[:, 1] - cv[:, 0])
            + config.weight_dd * (dd[:, 1] - dd[:, 0])
            + config.weight_ct * (q1 - q0)
        )

    def predict_proba(self, context_bags, context_labels, query_bags) -> torch.Tensor:
        """P(class 1) per query bag."""
        return torch.sigmoid(self.margins(context_bags, context_labels, query_bags))
