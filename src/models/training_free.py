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
    ct_num_tokens: int = 16
    ct_cells_per_bag: int = 64
    ct_temperature: float = 0.5
    ct_eps: float = 1e-6
    # SS137-3: CV : DD : CT. Sign of the DD term is required by DD returning
    # distances rather than logits.
    weight_cv: float = 1.442
    weight_dd: float = -0.343
    weight_ct: float = 0.286
    # SS148 diagnostic only. "extreme" is v107; "prototype"/"ridge" read all 16
    # abundance dims instead of two. Changing this changes the model.
    ct_readout: str = "extreme"



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
        # DISTANCES: small means close to that class, hence the head's negative weight.
        return (query_feature[:, None] - prototypes[None, :]).square() / dispersions[None, :]

    # ---- 5. CT ------------------------------------------------------------
    def _ct_features(self, context_bags, labels, query_bags):
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
                temperature=config.ct_temperature,
                eps=config.ct_eps,
            ),
            mode=config.ct_readout,
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
        split = (covariance_dim, context.shape[-1] - covariance_dim)
        cv = self._cv_logits(context, context_labels, query, split)

        def to_matrices(descriptors):
            flat = descriptors[..., : triangle.shape[1]]
            matrices = flat.new_zeros(flat.shape[0], config.sketch_dim, config.sketch_dim)
            matrices[..., triangle[0], triangle[1]] = flat
            matrices[..., triangle[1], triangle[0]] = flat
            return matrices

        dd = self._dd_features(to_matrices(context), context_labels, to_matrices(query))
        q0, q1 = self._ct_features(context_bags, context_labels, query_bags)
        return (
            config.weight_cv * (cv[:, 1] - cv[:, 0])
            + config.weight_dd * (dd[:, 1] - dd[:, 0])
            + config.weight_ct * (q1 - q0)
        )

    def predict_proba(self, context_bags, context_labels, query_bags) -> torch.Tensor:
        """P(class 1) per query bag."""
        return torch.sigmoid(self.margins(context_bags, context_labels, query_bags))
