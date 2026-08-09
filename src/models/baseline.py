"""CV-only single-cell meta-classifier: covariance sketch + closed-form ridge.

Scope note (docs SS73): this file used to carry six evidence branches. SS68
measured that deleting four of them changed the 50-fold result by -0.0005
(CI [-0.0037, +0.0024]), and a per-parameter gradient audit found 229 of
43,198,660 parameters ever received a gradient. Those branches were then
config-disabled for several arms and finally DELETED here -- git holds them at
`8caa96c` if they are ever needed again.

What remains:
  aggregator  -- centre, project through a fixed basis, form the covariance
                 sketch and the CV-2 covariance matrix. No learned parameters.
  CV-1        -- class-balanced dual ridge on the sketch, re-solved per episode.
  CV-2        -- label-chosen subspace -> log-variance -> prototype distance ->
                 a small MLP. The only learned module of consequence.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import os

import torch
from torch import nn
import torch.nn.functional as F


def solve_ridge_system(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    ridge_lambda: torch.Tensor,
) -> torch.Tensor:
    """Solve a positive-definite ridge system with adaptive FP32 jitter.

    Was `RidgeResidualMetaClassifier._solve_ridge_system`; that class carried
    the deleted global-shape branch, but CV-1 still needs this solver, so it
    moved to module scope unchanged.
    """
    if not torch.isfinite(gram).all() or not torch.isfinite(rhs).all():
        raise RuntimeError("The ridge system contains NaN or Inf values.")
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    if gram.ndim == 3:
        identity = identity.expand(gram.shape[0], -1, -1)
    system = gram + ridge_lambda.float() * identity
    if not torch.isfinite(system).all():
        raise RuntimeError("The ridge system contains NaN or Inf values.")

    diagonal_scale = (
        gram.diagonal(dim1=-2, dim2=-1)
        .abs()
        .mean(dim=-1, keepdim=True)
        .clamp_min(1.0)
    )
    jitter = diagonal_scale * 1e-6
    # A Cholesky factorization can succeed for a nearly singular system
    # while its backward pass still produces non-finite gradients. Always
    # include the adaptive jitter, including on the first attempt, so a
    # successful forward solve also has a numerically stable backward.
    for _ in range(6):
        candidate = system + jitter.unsqueeze(-1) * identity
        factor, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
        if bool((info == 0).all()):
            coefficients = torch.cholesky_solve(rhs, factor)
            if torch.isfinite(coefficients).all():
                return coefficients
        jitter = jitter * 10.0
    raise RuntimeError(
        "The ridge system remained non-finite or non-positive-definite "
        "after adaptive jitter."
    )


class EpisodePopulationAggregator(nn.Module):
    """Align variable-length bags to context-derived population slots.

    Anchors are selected from context cells only with permutation-invariant
    farthest-point sampling. Every context and query bag is then summarized in
    that shared episode coordinate system by population abundance, state shift,
    dispersion, and count-adaptive novelty tails. Labels are never used here.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_slots: int = 8,
        state_dim: int = 32,
        context_samples_per_bag: int = 32,
        assignment_temperature: float = 0.1,
        tail_fractions: Sequence[float] = (0.01, 0.05, 0.15),
        min_tail_instances: int = 1,
    ) -> None:
        super().__init__()
        if min(input_dim, num_slots, state_dim, context_samples_per_bag) < 1:
            raise ValueError("Population aggregator dimensions must be positive.")
        if assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive.")
        fractions = tuple(float(fraction) for fraction in tail_fractions)
        if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
            raise ValueError("tail_fractions must contain values in (0, 1].")
        if min_tail_instances < 1:
            raise ValueError("min_tail_instances must be positive.")

        self.input_dim = int(input_dim)
        self.num_slots = int(num_slots)
        self.state_dim = int(state_dim)
        self.context_samples_per_bag = int(context_samples_per_bag)
        self.assignment_temperature = float(assignment_temperature)
        self.tail_fractions = fractions
        self.min_tail_instances = int(min_tail_instances)
        self.state_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        self.tail_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim),
        )
        population_dim = num_slots * (state_dim + 2) + len(fractions) * state_dim
        self.population_projection = nn.Sequential(
            nn.LayerNorm(population_dim),
            nn.Linear(population_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )
        # Begin from the exact all-cell mean. The population path is introduced
        # smoothly as the final projection learns away from zero.
        nn.init.zeros_(self.population_projection[-1].weight)
        nn.init.zeros_(self.population_projection[-1].bias)
        self.residual_logit_scale = nn.Parameter(torch.tensor(-2.0))

    def _normalize_bags(
        self,
        instances: torch.Tensor | Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if isinstance(instances, torch.Tensor):
            if instances.ndim != 3:
                raise ValueError("Dense instances must be [bags, instances, features].")
            bags = list(instances.unbind(0))
        else:
            bags = list(instances)
        if not bags:
            raise ValueError("An episode must contain at least one bag.")
        if any(
            bag.ndim != 2 or bag.shape[0] == 0 or bag.shape[1] != self.input_dim
            for bag in bags
        ):
            raise ValueError(
                f"Every bag must contain [instances, {self.input_dim}] values."
            )
        return bags

    def _population_candidates(self, bag: torch.Tensor) -> torch.Tensor:
        """Select an order-invariant range of central-to-tail cells."""
        if bag.shape[0] <= self.context_samples_per_bag:
            return bag
        normalized = F.normalize(bag.float(), dim=-1)
        center = F.normalize(normalized.mean(dim=0, keepdim=True), dim=-1)
        centrality = (normalized * center).sum(dim=-1)
        order = torch.argsort(centrality)
        positions = (
            torch.linspace(
                0,
                bag.shape[0] - 1,
                self.context_samples_per_bag,
                device=bag.device,
            )
            .round()
            .long()
        )
        return bag[order[positions]]

    def _context_anchors(
        self,
        bags: list[torch.Tensor],
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        candidates = torch.cat(
            [
                self._population_candidates(bag)
                for bag, is_context in zip(bags, context_mask.tolist())
                if is_context
            ],
            dim=0,
        )
        if candidates.shape[0] < self.num_slots:
            raise ValueError(
                "Context does not contain enough cells for population slots."
            )
        normalized = F.normalize(candidates.float(), dim=-1)
        center = F.normalize(normalized.mean(dim=0, keepdim=True), dim=-1)
        first = torch.argmin((normalized * center).sum(dim=-1))
        selected = [first]
        max_similarity = normalized @ normalized[first]
        for _ in range(1, self.num_slots):
            next_index = torch.argmin(max_similarity)
            selected.append(next_index)
            similarity = normalized @ normalized[next_index]
            max_similarity = torch.maximum(max_similarity, similarity)
        return normalized[torch.stack(selected)].to(candidates.dtype)

    def forward(
        self,
        instances: torch.Tensor | Sequence[torch.Tensor],
        context_mask: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bags = self._normalize_bags(instances)
        context_mask = torch.as_tensor(
            context_mask,
            device=bags[0].device,
            dtype=torch.bool,
        ).flatten()
        if context_mask.numel() != len(bags) or not torch.any(context_mask):
            raise ValueError("context_mask must identify at least one context bag.")
        anchors = self._context_anchors(bags, context_mask)

        tokens: list[torch.Tensor] = []
        proportions: list[torch.Tensor] = []
        selected_counts: list[list[int]] = []
        for bag in bags:
            raw_mean = bag.mean(dim=0)
            normalized = F.normalize(bag.float(), dim=-1)
            similarity = normalized @ anchors.float().T
            assignment = torch.softmax(
                similarity / self.assignment_temperature,
                dim=-1,
            ).to(bag.dtype)
            mass = assignment.sum(dim=0).clamp_min(1e-6)
            proportion = mass / bag.shape[0]
            slot_mean = (assignment.T @ bag) / mass.unsqueeze(-1)
            state = self.state_projection(slot_mean - anchors)
            dispersion = (assignment * (1.0 - similarity).to(assignment.dtype)).sum(
                dim=0
            ) / mass

            nearest_similarity, nearest_slot = similarity.max(dim=-1)
            novelty = 1.0 - nearest_similarity
            tail_summaries: list[torch.Tensor] = []
            bag_selected_counts: list[int] = []
            for fraction in self.tail_fractions:
                count = min(
                    bag.shape[0],
                    max(
                        self.min_tail_instances,
                        int(math.ceil(fraction * bag.shape[0])),
                    ),
                )
                index = novelty.topk(count).indices
                deviation = bag[index] - anchors[nearest_slot[index]]
                tail_summaries.append(self.tail_projection(deviation).mean(dim=0))
                bag_selected_counts.append(count)

            population_features = torch.cat(
                (
                    proportion,
                    state.flatten(),
                    dispersion,
                    *tail_summaries,
                ),
                dim=0,
            )
            residual = self.population_projection(population_features)
            tokens.append(
                raw_mean + torch.sigmoid(self.residual_logit_scale) * residual
            )
            proportions.append(proportion)
            selected_counts.append(bag_selected_counts)

        result = torch.stack(tokens)
        if not return_auxiliary:
            return result
        return result, {
            "population_anchors": anchors,
            "population_proportions": torch.stack(proportions),
            "instance_counts": torch.tensor(
                [len(bag) for bag in bags], device=result.device
            ),
            "tail_counts": torch.tensor(selected_counts, device=result.device),
            "residual_scale": torch.sigmoid(self.residual_logit_scale),
        }


class StructuredEpisodePopulationAggregator(EpisodePopulationAggregator):
    """Hybrid density/rare population coordinates with multi-statistic slots.

    Density anchors are refined by deterministic soft k-means, while the
    remaining anchors cover high-residual context cells.  Every aligned slot
    keeps separate center, spread, and within-population rare-state tokens so
    that higher-order changes are not compressed back into a single mean.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_slots: int = 12,
        num_density_slots: int | None = None,
        context_samples_per_bag: int = 32,
        assignment_temperature: float = 0.1,
        density_refinement_steps: int = 4,
        density_temperature: float = 0.15,
        slot_rare_fraction: float = 0.05,
        tail_fractions: Sequence[float] = (0.01, 0.05, 0.15),
        absolute_tail_ks: Sequence[int] = (),
        ccts_lambdas: Sequence[float] = (),
        ccts_tau: float = 0.5,
        min_tail_instances: int = 1,
        bag_centered_representation: bool = True,
        bag_centered_l2_normalize: bool = True,
        bag_representation: str = "legacy",
        global_summary: str = "centered_spread",
        use_raw_mean_branch: bool = False,
        raw_stat_tokens: Sequence[str] = (),
        covariance_sketch_dim: int | None = None,
        covariance_matrix_dim: int | None = 32,
        covariance_slopes: "tuple[float, float] | None" = None,
        covariance_mode: str = "covariance",
        covariance_shrinkage: float = 0.0,
        include_cls_token: bool = False,
        cls_token_heads: int = 4,
        slot_latent_dim: int | None = None,
        slot_query_latent_dim: int | None = None,
        slot_affinity_dim: int | None = None,
        stream_eval_bags: bool = True,
    ) -> None:

        # Deliberately initialize nn.Module directly: anchor construction is
        # inherited, while the v13 compressed projections are not retained.
        nn.Module.__init__(self)
        if covariance_sketch_dim is None:
            covariance_sketch_dim = min(32, input_dim)
        if min(input_dim, num_slots, context_samples_per_bag) < 1:
            raise ValueError("Structured aggregator dimensions must be positive.")
        if not 1 <= covariance_sketch_dim <= input_dim:
            raise ValueError("covariance_sketch_dim must be in [1, input_dim].")
        covariance_modes = {
            "covariance",
            "correlation",
            "log_correlation",
            "covariance_log_correlation",
        }
        if covariance_mode not in covariance_modes:
            raise ValueError(f"Unsupported covariance_mode: {covariance_mode}")
        if not 0.0 <= covariance_shrinkage < 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1).")
        if num_density_slots is None:
            num_density_slots = min(8, max(1, round(2 * num_slots / 3)))
        if not 1 <= num_density_slots <= num_slots:
            raise ValueError("num_density_slots must be in [1, num_slots].")
        if assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive.")
        if density_refinement_steps < 1 or density_temperature <= 0:
            raise ValueError("Density refinement settings must be positive.")
        if not 0 < slot_rare_fraction <= 1:
            raise ValueError("slot_rare_fraction must be in (0, 1].")
        fractions = tuple(float(fraction) for fraction in tail_fractions)
        if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
            raise ValueError("tail_fractions must contain values in (0, 1].")
        abs_ks = tuple(int(k) for k in absolute_tail_ks)
        if any(k < 1 for k in abs_ks):
            raise ValueError("absolute_tail_ks must contain positive integers.")
        if min_tail_instances < 1:
            raise ValueError("min_tail_instances must be positive.")
        self.input_dim = int(input_dim)
        self.num_slots = int(num_slots)
        self.num_density_slots = int(num_density_slots)
        self.context_samples_per_bag = int(context_samples_per_bag)
        self.assignment_temperature = float(assignment_temperature)
        self.density_refinement_steps = int(density_refinement_steps)
        self.density_temperature = float(density_temperature)
        self.slot_rare_fraction = float(slot_rare_fraction)
        self.tail_fractions = fractions
        self.absolute_tail_ks = abs_ks
        self.ccts_lambdas = tuple(float(lamb) for lamb in ccts_lambdas)
        self.ccts_tau = float(ccts_tau)
        if self.ccts_lambdas:
            self.ccts_score_head = nn.Linear(self.input_dim, 1)
            self.ccts_metadata_projection = nn.Linear(5, self.input_dim)

        # MLA-style low-rank slot affinity (config-gated probe). When
        # slot_latent_dim is set, the cell-to-slot affinity is computed in a
        # low-rank latent space (query d_cq, key d_c) via absorbed query
        # weights (W_UQ^T W_UK), never expanding to the affinity dim. None
        # keeps the legacy full-dim dot product. Extra parameters are created
        # ONLY when enabled so existing checkpoints load unchanged.
        self.slot_latent_dim = (
            int(slot_latent_dim) if slot_latent_dim is not None else None
        )
        self.slot_query_latent_dim = (
            int(slot_query_latent_dim)
            if slot_query_latent_dim is not None
            else (self.slot_latent_dim if self.slot_latent_dim is not None else None)
        )
        self.slot_affinity_dim = (
            int(slot_affinity_dim)
            if slot_affinity_dim is not None
            else (
                max(self.slot_latent_dim, self.slot_query_latent_dim)
                if self.slot_latent_dim is not None
                else None
            )
        )
        if self.slot_latent_dim is not None:
            if not 1 <= self.slot_latent_dim <= self.input_dim:
                raise ValueError("slot_latent_dim must be in [1, input_dim].")
            if self.slot_query_latent_dim < 1 or self.slot_affinity_dim < 1:
                raise ValueError(
                    "slot_query_latent_dim / slot_affinity_dim must be positive."
                )
            self.slot_w_dq = nn.Linear(
                self.input_dim, self.slot_query_latent_dim, bias=False
            )
            self.slot_w_dkv = nn.Linear(
                self.input_dim, self.slot_latent_dim, bias=False
            )
            self.slot_w_uq = nn.Linear(
                self.slot_query_latent_dim, self.slot_affinity_dim, bias=False
            )
            self.slot_w_uk = nn.Linear(
                self.slot_latent_dim, self.slot_affinity_dim, bias=False
            )

        valid_bag_representations = {"legacy", "poolz", "poolz_l2"}
        if bag_representation not in valid_bag_representations:
            raise ValueError(
                f"bag_representation must be one of {sorted(valid_bag_representations)}, "
                f"got {bag_representation!r}."
            )
        if bag_representation == "legacy":
            # Historical flag pair keeps deciding the view (default path).
            if bag_centered_representation:
                if global_summary != "centered_spread" or use_raw_mean_branch:
                    raise ValueError(
                        "Centered v19 mode requires a centered spread summary "
                        "and use_raw_mean_branch=False."
                    )
            elif global_summary != "raw_mean" or not use_raw_mean_branch:
                raise ValueError(
                    "Raw-mean diagnostic mode requires global_summary=raw_mean "
                    "and use_raw_mean_branch=True."
                )
        else:
            # v30 B1: replace per-bag centering with context-pool standardization.
            # The summary stays the per-bag centered spread and `centered_delta`
            # is still returned per bag, so the covariance/spread branches are
            # untouched -- only `classification_instances` changes.
            if not bag_centered_representation or use_raw_mean_branch:
                raise ValueError(
                    f"bag_representation={bag_representation!r} requires "
                    "bag_centered_representation=True and use_raw_mean_branch=False "
                    "(it replaces the centering step, not the summary/covariance path)."
                )
            if global_summary != "centered_spread":
                raise ValueError(
                    f"bag_representation={bag_representation!r} requires "
                    "global_summary='centered_spread'."
                )
        self.min_tail_instances = int(min_tail_instances)
        self.slot_statistic_count = 3
        self.bag_centered_representation = bool(bag_centered_representation)
        self.bag_centered_l2_normalize = bool(bag_centered_l2_normalize)
        self.bag_representation = str(bag_representation)
        self.global_summary = str(global_summary)
        self.use_raw_mean_branch = bool(use_raw_mean_branch)
        valid_raw_stats = {"mean", "variance", "skewness", "kurtosis"}
        unknown_stats = set(raw_stat_tokens) - valid_raw_stats
        if unknown_stats:
            raise ValueError(
                f"Unknown raw_stat_tokens: {sorted(unknown_stats)} "
                f"(valid: {sorted(valid_raw_stats)})."
            )
        self.raw_stat_tokens = tuple(raw_stat_tokens)
        self.covariance_sketch_dim = int(covariance_sketch_dim)
        # How many of P's columns CV-2 sees (docs SS69-7). Historically hardcoded
        # to 32 via the `_projected_covariance_matrix` default, so CV-2 saw only
        # the first 32 of 64 columns and stayed at 32 no matter how large K got.
        # None ties it to K, which is what the K-sweep arms want.
        self.covariance_matrix_dim = (
            self.covariance_sketch_dim
            if covariance_matrix_dim is None
            else int(covariance_matrix_dim)
        )
        # CV-only (docs SS68): emit ONLY the covariance tensors and skip the
        # slot pipeline, tails, metadata, anchors and global summary entirely.
        # Measured: 18.73 ms -> 0.40 ms floor on 8 bags x 4000 cells, i.e. ~98%
        # of the old CV-only forward was computed and discarded.
        # The dead keys are ABSENT, not zero-filled, so a stray consumer raises
        # KeyError at its own line instead of silently reading a zero.
        self.covariance_mode = str(covariance_mode)
        self.covariance_shrinkage = float(covariance_shrinkage)
        # Eval-only bag-at-a-time streaming (docs v35 §3.3). Numerically exact --
        # the same `_bag_view`/`_covariance_sketch` on the same inputs, only
        # computed later and not retained -- so it is safe on by default. Set
        # False (or export BAGPFN_DISABLE_BAG_STREAMING=1, which is how the
        # equality claim is A/B'd against a real checkpoint) to force the eager
        # path.
        self.stream_eval_bags = (
            bool(stream_eval_bags)
            and os.environ.get("BAGPFN_DISABLE_BAG_STREAMING") != "1"
        )
        self.slot_covariance_descriptor = "correlation"
        self.emit_covariance_matrix = False
        self.include_cls_token = bool(include_cls_token)
        if self.include_cls_token:
            self.cls_token_pooling = ClassTokenPooling(
                token_dim=self.input_dim, num_heads=int(cls_token_heads)
            )
        candidate_index = torch.arange(
            1, self.context_samples_per_bag + 1, dtype=torch.float32
        )[:, None]
        feature_index = torch.arange(1, self.input_dim + 1, dtype=torch.float32)[
            None, :
        ]
        candidate_directions = torch.sin(
            0.017 * candidate_index * feature_index
        ) + torch.cos(0.013 * (candidate_index + 1) * feature_index)
        self.register_buffer(
            "_candidate_directions",
            F.normalize(candidate_directions, dim=-1, eps=1e-6),
            persistent=False,
        )
        covariance_index = torch.arange(
            1, self.covariance_sketch_dim + 1, dtype=torch.float32
        )[None, :]
        # Frequency-ladder slopes (docs SS69). Default reproduces the historical
        # hardcoded (0.019, 0.011) exactly. `a` is the ladder spacing: column k
        # oscillates along the channel axis at a*k rad/step, so a*K is the total
        # bandwidth used. Sweeping K with `a` FIXED changes bandwidth too, which
        # is what hid the dimension effect (SS69-4); setting a = 0.85*pi/K holds
        # bandwidth constant so K alone varies. 0.85 leaves a guard band -- at
        # a*K = pi exactly, sin(pi*d) = 0 for integer d and that column's sin
        # term vanishes identically.
        slope_a, slope_b = (0.019, 0.011) if covariance_slopes is None else (
            float(covariance_slopes[0]), float(covariance_slopes[1])
        )
        self.covariance_slopes = (slope_a, slope_b)
        covariance_directions = torch.sin(
            slope_a * feature_index.T * covariance_index
        ) + torch.cos(slope_b * (feature_index.T + 1) * covariance_index)
        covariance_basis = torch.linalg.qr(
            covariance_directions, mode="reduced"
        ).Q
        triangle = torch.triu_indices(
            self.covariance_sketch_dim, self.covariance_sketch_dim
        )
        self.register_buffer("_covariance_projection", covariance_basis, persistent=False)
        self.register_buffer("_covariance_triangle", triangle, persistent=False)
        self.center_slot_encoder = self._make_slot_encoder(input_dim)
        self.spread_slot_encoder = self._make_slot_encoder(input_dim)
        self.rare_slot_encoder = self._make_slot_encoder(input_dim)
        for encoder in (
            self.center_slot_encoder,
            self.spread_slot_encoder,
            self.rare_slot_encoder,
        ):
            nn.init.zeros_(encoder[-1].weight)
            nn.init.zeros_(encoder[-1].bias)
        self.slot_residual_logit = nn.Parameter(torch.tensor(-1.1))
        self.shared_tail_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

    @staticmethod
    def _make_slot_encoder(input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(2 * input_dim + 2),
            nn.Linear(2 * input_dim + 2, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

    def _context_pool_stats(
        self, bags: Sequence[torch.Tensor], context_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-feature mean/std over every cell of every CONTEXT bag.

        Cell-count weighted (bags differ in size by up to 1000x on real data, so
        averaging per-bag means would misweight them). Context-only, matching the
        leak-free convention already used by `_ridge_logits` and
        `_normalize_covariance_relation`.
        """
        selected = [
            bag.reshape(-1, bag.shape[-1])
            for bag, keep in zip(bags, context_mask.flatten().tolist())
            if keep
        ]
        if not selected:
            raise ValueError("context_mask must identify at least one context bag.")
        # Streaming (bag-by-bag, float64) accumulation. The old `torch.cat` of
        # every context cell materialized a [total_cells, dim] copy -- ~12 GB for
        # a full-tile PathoBench episode (270 slides x ~7.5k tiles x 1536) -- and
        # was a main driver of the eval OOM (docs v35 §3). Two passes (mean, then
        # centered variance) keep this numerically equal to
        # `_context_pool_stats_batched`, which also centers before squaring.
        # unbiased=False (divide by cells, no Bessel correction) so the training
        # (batched) and evaluation (list) paths agree; with the correction they
        # would differ by sqrt(cells/(cells-1)), a silent ~0.25% scale mismatch.
        dim = selected[0].shape[-1]
        device = selected[0].device
        total = 0
        sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
        for flat in selected:
            sum_x += flat.sum(dim=0, dtype=torch.float64)
            total += flat.shape[0]
        mean64 = sum_x / total
        mean32 = mean64.float()
        sum_sq = torch.zeros(dim, dtype=torch.float64, device=device)
        for flat in selected:
            sum_sq += (
                (flat.float() - mean32).square().sum(dim=0, dtype=torch.float64)
            )
        std = (sum_sq / total).clamp_min(0.0).sqrt().float().clamp_min(1e-6)
        return mean32, std

    @staticmethod
    def _context_pool_stats_batched(
        x: torch.Tensor,
        is_context: torch.Tensor,
        cell_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched twin of `_context_pool_stats`.

        `x` is [episodes, bags, instances, dim] and `is_context` is
        [episodes, bags]; returns per-episode mean/std shaped [episodes, dim].
        When `cell_mask` ([episodes, bags, instances]) is given (padded ragged
        batches), only cells that are both context and unpadded contribute.
        """
        episodes, num_bags, num_instances, dim = x.shape
        values = x.float()
        mask = is_context[..., None, None].expand(
            episodes, num_bags, num_instances, 1
        )
        if cell_mask is not None:
            if cell_mask.shape != (episodes, num_bags, num_instances):
                raise ValueError(
                    "cell_mask must be [episodes, bags, instances]."
                )
            mask = mask & cell_mask[..., None]
        mask = mask.to(values.dtype)
        cells = mask.sum(dim=(1, 2)).clamp_min(1.0)
        mean = (values * mask).sum(dim=(1, 2)) / cells
        variance = (
            ((values - mean[:, None, None, :]).square() * mask).sum(dim=(1, 2)) / cells
        )
        std = variance.clamp_min(1e-12).sqrt().clamp_min(1e-6)
        return mean, std

    def _bag_view(
        self,
        bag: torch.Tensor,
        pool_mean: torch.Tensor | None = None,
        pool_std: torch.Tensor | None = None,
        cell_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return classification instances, summary, and centered deltas.

        `pool_mean`/`pool_std` are required only when `bag_representation` is a
        pool-standardized mode; they must broadcast against `bag`'s feature axis.
        `cell_mask` ([..., instances]) marks the real cells of a padded bag;
        masked (padded) cells are excluded from the mean/spread statistics.
        """
        if cell_mask is None:
            bag_mean = bag.mean(dim=-2, keepdim=True)
            centered_delta = bag - bag_mean
            global_spread = torch.sqrt(
                centered_delta.float().square().mean(dim=-2) + 1e-6
            )
        else:
            if (
                cell_mask.shape[:-1] != bag.shape[:-2]
                or cell_mask.shape[-1] != bag.shape[-2]
            ):
                raise ValueError(
                    "cell_mask must match the bag's instance axis."
                )
            count = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            masked = bag.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            bag_mean = masked.sum(dim=-2, keepdim=True) / count.unsqueeze(-1)
            centered_delta = masked - bag_mean
            # Padded cells must not pollute downstream covariance/statistics.
            centered_delta = centered_delta.masked_fill(
                ~cell_mask.unsqueeze(-1), 0.0
            )
            global_spread = torch.sqrt(
                (centered_delta.float().square() * cell_mask.unsqueeze(-1)).sum(
                    dim=-2
                )
                / count
                + 1e-6
            )
        if self.bag_representation in ("poolz", "poolz_l2"):
            if pool_mean is None or pool_std is None:
                raise ValueError(
                    f"bag_representation={self.bag_representation!r} needs context-pool "
                    "statistics; call _bag_view via forward_episode_batch, "
                    "StructuredEpisodePopulationAggregator.forward, or BaseModel.forward "
                    "(or pass pool_mean/pool_std explicitly)."
                )
            standardized = (bag.float() - pool_mean) / pool_std
            if self.bag_representation == "poolz_l2":
                standardized = F.normalize(standardized, dim=-1, eps=1e-6)
            return (
                standardized.to(bag.dtype),
                global_spread.to(bag.dtype),
                centered_delta,
            )
        if self.bag_centered_representation:
            if self.bag_centered_l2_normalize:
                classification_instances = F.normalize(
                    centered_delta.float(), dim=-1, eps=1e-6
                ).to(bag.dtype)
            else:
                # Keep the deviation magnitude: the normalization-ceiling probe
                # (docs/current_status.md SS19) found per-cell L2 normalization
                # discards magnitude information worth ~0.05 AUROC to a
                # sufficient-stat ridge. Downstream slot assignment and the
                # token encoders re-normalize internally, so this only exposes
                # real magnitudes to the slot/tail statistics.
                classification_instances = centered_delta
            summary = global_spread.to(bag.dtype)
        else:
            classification_instances = bag
            summary = bag_mean.squeeze(-2)
        return classification_instances, summary, centered_delta

    def _covariance_sketch(
        self,
        centered_delta: torch.Tensor,
        count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a configurable translation-invariant second-moment feature.

        `count` (per-bag valid cell counts, [num_bags]) overrides the instance
        axis length for normalization so padded cells do not shrink the
        covariance of short bags.
        """
        values = centered_delta.float()
        projected = values @ self._covariance_projection.float()
        covariance = projected.transpose(-1, -2) @ projected
        if count is None:
            covariance = covariance / projected.shape[-2]
        else:
            covariance = covariance / count.reshape(
                projected.shape[:-2] + (1, 1)
            )
        row, column = self._covariance_triangle
        raw_feature = covariance[..., row, column]
        if self.covariance_mode == "covariance":
            return raw_feature.to(centered_delta.dtype)

        diagonal = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
        inverse_scale = diagonal.rsqrt()
        correlation = (
            covariance * inverse_scale.unsqueeze(-1) * inverse_scale.unsqueeze(-2)
        )
        if self.covariance_shrinkage:
            identity = torch.eye(
                self.covariance_sketch_dim,
                device=correlation.device,
                dtype=correlation.dtype,
            )
            correlation = (
                (1.0 - self.covariance_shrinkage) * correlation
                + self.covariance_shrinkage * identity
            )
        if self.covariance_mode == "correlation":
            return correlation[..., row, column].to(centered_delta.dtype)

        eigenvalues, eigenvectors = torch.linalg.eigh(correlation.float())
        log_values = eigenvalues.clamp_min(1e-4).log()
        log_correlation = (eigenvectors * log_values.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
        log_feature = log_correlation[..., row, column]
        if self.covariance_mode == "log_correlation":
            return log_feature.to(centered_delta.dtype)
        return torch.cat((raw_feature, log_feature), dim=-1).to(centered_delta.dtype)

    def _projected_covariance_matrix(
        self, centered_delta: torch.Tensor, dimension: int | None = None
    ) -> torch.Tensor:
        if dimension is None:
            dimension = self.covariance_matrix_dim
        dimension = min(int(dimension), self.covariance_sketch_dim)
        projected = centered_delta.float() @ self._covariance_projection[:, :dimension].float()
        return torch.einsum("...ni,...nj->...ij", projected, projected) / projected.shape[-2]



    def _population_candidates(
        self,
        bag: torch.Tensor,
        cell_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build stable, order-invariant soft candidates in centered space.

        `cell_mask` ([..., instances]) excludes padded cells from the soft
        candidate pool.
        """
        normalized = F.normalize(bag.float(), dim=-1, eps=1e-6)
        if cell_mask is None:
            candidate_count = min(self.context_samples_per_bag, bag.shape[0])
        else:
            candidate_count = min(
                self.context_samples_per_bag, int(cell_mask.sum().item())
            )
        directions = self._candidate_directions[:candidate_count].float()
        scores = normalized @ directions.T
        if cell_mask is not None:
            scores = scores.masked_fill(~cell_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores.mul(10.0), dim=0)
        candidates = weights.T @ normalized
        return F.normalize(candidates, dim=-1, eps=1e-6).to(bag.dtype)

    def _population_candidates_batched(
        self,
        bags: list[torch.Tensor],
    ) -> torch.Tensor:
        """Batched ``_population_candidates`` over a list of bags.

        Pads the bags to a common cell count and runs ONE masked softmax over
        [C, max_cells, k] instead of C separate (cells, k) softmaxes, removing
        the per-bag kernel-launch overhead (this was the dominant cost in large-
        context episodes). Per-bag result is EXACT when every bag has at least
        ``context_samples_per_bag`` cells (the large-context regime); otherwise
        it falls back to the per-bag loop (candidate_count varies per bag).

        Returns the same [n_candidates, dim] tensor as concatenating
        ``_population_candidates`` over the bags, in bag-major order.
        """
        k = min(self.context_samples_per_bag, min(bag.shape[0] for bag in bags))
        if k < self.context_samples_per_bag:
            return torch.cat(
                [self._population_candidates(bag) for bag in bags], dim=0
            )
        max_cells = max(bag.shape[0] for bag in bags)
        lengths = torch.as_tensor(
            [bag.shape[0] for bag in bags], device=bags[0].device
        )
        padded = torch.stack(
            [
                F.pad(bag.float(), (0, 0, 0, max_cells - bag.shape[0]))
                for bag in bags
            ]
        )  # [C, max_cells, dim]
        cell_mask = (
            torch.arange(max_cells, device=padded.device)[None, :] < lengths[:, None]
        )  # [C, max_cells]
        normalized = (
            padded if self._instances_are_unit else F.normalize(padded, dim=-1, eps=1e-6)
        )
        directions = self._candidate_directions[:k].float()  # [k, dim]
        scores = torch.einsum("cnd,kd->cnk", normalized, directions)
        scores = scores.masked_fill(~cell_mask.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores.mul(10.0), dim=1)  # [C, max_cells, k]
        candidates = torch.einsum("cnk,cnd->ckd", weights, normalized)  # [C, k, dim]
        candidates = F.normalize(candidates, dim=-1, eps=1e-6).to(bags[0].dtype)
        return candidates.reshape(-1, candidates.shape[-1])

    def _context_anchors(
        self,
        bags: list[torch.Tensor],
        context_mask: torch.Tensor,
        cell_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context_indices = [
            b_index
            for b_index, (_, is_context) in enumerate(
                zip(bags, context_mask.tolist())
            )
            if is_context
        ]
        context_bags = [bags[b_index] for b_index in context_indices]
        if cell_masks is None and self.training:
            # Batched path is a training speedup (single masked softmax over
            # [C, max_cells, k]); it materializes a [C, max_cells, dim] padded
            # tensor, which OOMs at eval when context bags are huge (e.g.
            # PathoBench full-tile slides). Eval always uses the per-bag loop.
            candidates = self._population_candidates_batched(context_bags)
        else:
            candidates = torch.cat(
                [
                    self._population_candidates(
                        context_bags[i],
                        cell_masks[context_indices[i]]
                        if cell_masks is not None
                        else None,
                    )
                    for i in range(len(context_bags))
                ],
                dim=0,
            )
        return self._select_anchors(candidates)

    def _select_anchors(self, candidates: torch.Tensor) -> torch.Tensor:
        """Pick population anchors from an already-built candidate pool.

        Split out of `_context_anchors` so the streaming path (docs v35 §3.3)
        can build the per-bag candidates one bag at a time and still land on
        BIT-IDENTICAL anchors: `_population_candidates` returns exactly
        `context_samples_per_bag` candidates per bag regardless of the bag's
        cell count, so the pool this sees is the same tensor either way.
        """
        if candidates.shape[0] < self.num_slots:
            raise ValueError(
                "Context does not contain enough cells for population slots."
            )
        normalized = F.normalize(candidates.float(), dim=-1)
        center = F.normalize(normalized.mean(dim=0, keepdim=True), dim=-1)
        centrality = (normalized * center).sum(dim=-1)

        # Centrality quantiles provide deterministic, order-invariant initial
        # seeds spanning dense context states without starting from outliers.
        order = torch.argsort(centrality, descending=True)
        density_limit = max(
            self.num_density_slots,
            int(math.ceil(0.85 * order.numel())),
        )
        positions = (
            torch.linspace(
                0,
                density_limit - 1,
                self.num_density_slots,
                device=order.device,
            )
            .round()
            .long()
        )
        density = normalized[order[positions]]
        for _ in range(self.density_refinement_steps):
            similarity = normalized @ density.T
            assignment = torch.softmax(
                similarity / self.density_temperature,
                dim=-1,
            )
            mass = assignment.sum(dim=0).clamp_min(1e-6)
            density = F.normalize(
                assignment.T @ normalized / mass.unsqueeze(-1),
                dim=-1,
            )

        rare_count = self.num_slots - self.num_density_slots
        if rare_count == 0:
            return density.to(candidates.dtype)
        density_similarity = normalized @ density.T
        density_residual = 1.0 - density_similarity.max(dim=-1).values
        selected: list[torch.Tensor] = []
        available = torch.ones(
            normalized.shape[0], dtype=torch.bool, device=normalized.device
        )
        diversity = torch.ones_like(density_residual)
        for _ in range(rare_count):
            score = density_residual * diversity
            score = score.masked_fill(~available, float("-inf"))
            index = score.argmax()
            selected.append(normalized[index])
            available[index] = False
            similarity = normalized @ normalized[index]
            diversity = torch.minimum(diversity, (1.0 - similarity).clamp_min(0.0))
        rare = torch.stack(selected)
        return torch.cat((density, rare), dim=0).to(candidates.dtype)



    @property
    def _instances_are_unit(self) -> bool:
        """True when ``_bag_view`` already L2-normalizes each cell, so the dense
        path / candidate pooling can reuse the vectors instead of re-normalizing
        (a full per-cell pass). poolz_l2 always normalizes; the legacy view
        normalizes when centered + l2.
        """
        return self.bag_representation == "poolz_l2" or (
            self.bag_centered_representation and self.bag_centered_l2_normalize
        )

    def _slot_similarity(
        self, cells: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        """Cell-to-slot affinity (pre-temperature), full-dim or MLA low-rank.

        ``slot_latent_dim is None``: legacy full-dim dot product.

        Otherwise: MLA-style low-rank affinity. Cells (query) are compressed to
        ``d_cq`` (W_DQ) and anchors (keys) to ``d_c`` (W_DKV); the affinity is
        computed in the KV latent space with the absorbed query weight
        ``W_UQ^T W_UK`` (d_cq x d_c), never expanding to the affinity dim.

        cells: [..., num_cells, input_dim]; anchors: [..., num_slots, input_dim].
        Returns [..., num_cells, num_slots].
        """
        if self.slot_latent_dim is None:
            if cells.ndim == 3:
                return torch.einsum("bnd,bsd->bns", cells, anchors)
            return cells @ anchors.transpose(-1, -2)
        c_q = self.slot_w_dq(cells)  # [..., num_cells, d_cq]
        c_kv = self.slot_w_dkv(anchors)  # [..., num_slots, d_c]
        w_abs = self.slot_w_uq.weight.T @ self.slot_w_uk.weight  # [d_cq, d_c]
        q_latent = torch.einsum("...cd,dn->...cn", c_q, w_abs)  # [..., num_cells, d_c]
        return torch.einsum("...cd,...sd->...cs", q_latent, c_kv)

    def _forward_dense(
        self,
        instances: torch.Tensor,
        anchors: torch.Tensor,
        return_auxiliary: bool,
        global_summary: torch.Tensor | None = None,
        covariance_sketch: torch.Tensor | None = None,
        centered_delta: torch.Tensor | None = None,
        raw_stats: dict[str, torch.Tensor] | None = None,
        cell_mask: torch.Tensor | None = None,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
        """Vectorized equivalent of the per-bag path for synthetic episodes.

        `cell_mask` ([bags, instances], bool) marks the real cells of padded
        ragged (B2b) bags; padded cells are excluded from slot assignment,
        tails, rare states, and covariance normalization. ``ccts_lambdas`` is
        not supported on the padded path (v30 configs do not enable it).
        """
        # `instances` here is ALREADY pool-standardised by the caller, while
        # `centered_delta` is raw-centred and passed in separately. Deriving
        # it from `instances` instead is exactly the bug that broke
        # dense/ragged agreement by 2.4e-2 on the first attempt -- honour the
        # argument, and fall back to the same formula the full path uses.
        if centered_delta is None:
            if cell_mask is None:
                centered_delta = instances - instances.mean(dim=-2, keepdim=True)
                counts = None
            else:
                counts = cell_mask.sum(dim=-1).clamp_min(1)
                masked = instances.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
                centered_delta = masked - (
                    masked.sum(dim=-2, keepdim=True)
                    / counts.unsqueeze(-1).unsqueeze(-1)
                )
                centered_delta = centered_delta.masked_fill(
                    ~cell_mask.unsqueeze(-1), 0.0
                )
        else:
            counts = (
                cell_mask.sum(dim=-1).clamp_min(1)
                if cell_mask is not None
                else None
            )
        if covariance_sketch is None:
            covariance_sketch = self._covariance_sketch(
                centered_delta, count=counts
            )
        representation = {
            "covariance_sketch": covariance_sketch,
            "covariance_matrix": (
                self._projected_covariance_matrix(centered_delta)
                if self.emit_covariance_matrix
                else centered_delta.new_zeros((instances.shape[0], 1, 1))
            ),
        }
        return (representation, {}) if return_auxiliary else representation
    def forward(
        self,
        instances: torch.Tensor | Sequence[torch.Tensor],
        context_mask: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
        raw_bags = self._normalize_bags(instances)
        # context_mask is validated BEFORE _bag_view because pool-standardized
        # representations need context-only statistics to build the view.
        context_mask = torch.as_tensor(
            context_mask,
            device=raw_bags[0].device,
            dtype=torch.bool,
        ).flatten()
        if context_mask.numel() != len(raw_bags) or not torch.any(context_mask):
            raise ValueError("context_mask must identify at least one context bag.")
        if self.bag_representation in ("poolz", "poolz_l2"):
            pool_mean, pool_std = self._context_pool_stats(raw_bags, context_mask)
        else:
            pool_mean = pool_std = None
        # Same derivation as the full path (_bag_view with the episode's
        # pool statistics), just without the slot pipeline that follows.
        sketches, matrices = [], []
        for raw_bag in raw_bags:
            _, _, centered_delta = self._bag_view(raw_bag, pool_mean, pool_std)
            sketches.append(self._covariance_sketch(centered_delta))
            matrices.append(
                self._projected_covariance_matrix(centered_delta)
                if self.emit_covariance_matrix
                else centered_delta.new_zeros((1, 1))
            )
        representation = {
            "covariance_sketch": torch.stack(sketches),
            "covariance_matrix": torch.stack(matrices),
        }
        return (representation, {}) if return_auxiliary else representation


class StructuredPopulationMetaClassifier(nn.Module):
    """Distribution-aware, label-equivariant class-memory meta classifier."""

    def __init__(
        self,
        token_dim: int = 512,
        covariance_ridge_logit_scale: float = 2.0,
        covariance_residual_scale: float = 0.25,
        covariance_relation: dict[str, object] | None = None,
        meta_enable_covariance_ridge: bool = True,
        num_classes: int = 2,
    ) -> None:
        """CV-only meta classifier: CV-1 (dual ridge) + CV-2 (subspace head).

        The pre-prune signature took 45 arguments configuring six evidence
        branches. Five of those branches are gone (docs SS73), and with them
        every argument that only fed them -- including `meta_covariance_only`
        itself, which no longer selects anything because there is nothing left
        to select against.
        """
        super().__init__()
        if covariance_ridge_logit_scale <= 0:
            raise ValueError("covariance_ridge_logit_scale must be positive.")
        if not 0 < covariance_residual_scale < 1:
            raise ValueError("covariance_residual_scale must be in (0, 1).")
        self.token_dim = int(token_dim)
        self.num_classes = int(num_classes)
        # Ridge ablation (docs SS65/SS66): zeroing CV-1 leaves only CV-2. Kept
        # because SS66 measured that removing it collapses training -- the flag
        # is how that gets re-measured, not a leftover.
        self.force_covariance_ridge_zero = not bool(meta_enable_covariance_ridge)
        relation_config = dict(covariance_relation or {})
        self.covariance_relation_enabled = bool(relation_config.get("enabled", False))
        self.covariance_relation_mode = str(
            relation_config.get("mode", "prototype_cosine")
        )
        self.covariance_relation_granularity = str(
            relation_config.get("granularity", "bag")
        )
        self.covariance_relation_slot_routing = str(
            relation_config.get("slot_routing", "reliability_mean")
        )
        self.covariance_relation_routing_temperature = float(
            relation_config.get("routing_temperature", 0.1)
        )
        self.covariance_relation_subspace_rank = int(
            relation_config.get("subspace_rank", 1)
        )
        self.covariance_relation_subspace_whiten = bool(
            relation_config.get("subspace_whiten", True)
        )
        self.covariance_relation_subspace_shrinkage = float(
            relation_config.get("subspace_shrinkage", 0.1)
        )
        self.covariance_relation_diagnostic_only = bool(
            relation_config.get("diagnostic_only", True)
        )
        self.covariance_relation_residual_scale = float(
            relation_config.get("residual_scale", 0.02)
        )
        self.covariance_relation_eps = float(relation_config.get("eps", 1e-6))
        # Margin activation (docs SS72). "tanh" is the historical default from
        # `5869535`, which replaced a query-axis RMS with a raw tanh so that
        # 1-query eval == multi-query == training. That invariance comes from
        # `margin` being per-query, NOT from tanh itself, so "identity" keeps it.
        #
        # ⚠️ Measured (seed 42, K=128): tanh saturates PER EPISODE, not globally.
        # On the trained v41_K128 checkpoint |tanh(margin)| > 0.9999 in 40% of
        # episodes, where the local slope 1 - tanh² is ~2e-4; mean |margin| is
        # 0.88. The head still trains -- 229 parameters carry a non-zero gradient
        # at init AND at epoch 49 -- but it learns only from the unsaturated
        # minority. "identity" removes the ceiling, at the cost of unbounded
        # logits, hence the temperature below.
        #
        # An earlier note here claimed the head received gradient 0.0 and that
        # only 3 parameters trained. That was WRONG and is retracted; the direct
        # per-parameter gradient measurement above supersedes it.
        margin_activation = str(
            relation_config.get("margin_activation", "tanh")
        ).lower()
        if margin_activation not in ("tanh", "identity"):
            raise ValueError(
                "covariance_relation.margin_activation must be 'tanh' or "
                f"'identity', got {margin_activation!r}."
            )
        self.covariance_relation_margin_activation = margin_activation
        # Learnable temperature for the identity path (docs SS72). Only created
        # in that mode: adding a parameter unconditionally would change the
        # state_dict and break strict loading of every existing checkpoint.
        # Init from `margin_temperature` so the arm STARTS where the tanh version
        # sat, then learns its own scale.
        # ⚠️ Match the margin's BODY, not its tail. v43 used T=150 to align
        # |max| with CV-1, but |max| tracks the tail (raw margin max 12-25) while
        # the body is mean 1.5-3.3 -- so mean |CV-2 margin| came out 0.0101 vs
        # the tanh version's 0.594, i.e. CV-2 effectively started switched off.
        # The parameter is learnable so it recovers (150 -> 53.4 in 8 epochs),
        # but it spends ~30 epochs doing so. v44 uses T=2.0. See those configs.
        # Dividing by a learned scalar keeps the query-count invariance
        # that `5869535` established -- unlike the query-axis RMS it replaced,
        # this does not couple queries.
        if margin_activation == "identity":
            self.covariance_relation_log_temperature = nn.Parameter(
                torch.tensor(
                    math.log(float(relation_config.get("margin_temperature", 1.0)))
                )
            )
        self.covariance_relation_kernel_scales = tuple(
            float(value)
            for value in relation_config.get("kernel_scales", (0.5, 1.0, 2.0))
        )
        valid_relation_modes = {
            "prototype_cosine",
            "standardized_distance",
            "multiscale_rbf",
            "gated_distance",
            "learned_head",
        }
        if self.covariance_relation_granularity not in {"bag", "slot", "subspace"}:
            raise ValueError("covariance relation granularity must be bag, slot, or subspace.")
        if self.covariance_relation_slot_routing not in {
            "reliability_mean", "context_top1", "context_top3", "context_softmax"
        }:
            raise ValueError("unsupported covariance relation slot routing.")
        if self.covariance_relation_routing_temperature <= 0:
            raise ValueError("covariance relation routing_temperature must be positive.")
        if self.covariance_relation_mode not in valid_relation_modes:
            raise ValueError(
                "covariance relation mode must be one of "
                f"{sorted(valid_relation_modes)}."
            )
        if self.covariance_relation_eps <= 0:
            raise ValueError("covariance relation eps must be positive.")
        if not 0 <= self.covariance_relation_residual_scale < 1:
            raise ValueError("covariance relation residual_scale must be in [0, 1).")
        if (
            not self.covariance_relation_kernel_scales
            or any(value <= 0 for value in self.covariance_relation_kernel_scales)
        ):
            raise ValueError("covariance relation kernel_scales must be positive.")
        if self.covariance_relation_mode == "gated_distance":
            self.covariance_relation_gate_a = nn.Parameter(torch.tensor(5.0))
            self.covariance_relation_gate_b = nn.Parameter(torch.tensor(0.5))
        elif self.covariance_relation_mode == "learned_head":
            self.covariance_relation_head = nn.Sequential(
                nn.Linear(4, 32),
                nn.GELU(),
                nn.Linear(32, self.num_classes),
            )
        self.covariance_ridge_log_lambda = nn.Parameter(
            torch.tensor(math.log(1.0))
        )
        self.covariance_ridge_log_scale = nn.Parameter(
            torch.tensor(math.log(float(covariance_ridge_logit_scale)))
        )
        covariance_logit = math.log(
            covariance_residual_scale / (1.0 - covariance_residual_scale)
        )
        self.covariance_residual_logit = nn.Parameter(torch.tensor(covariance_logit))

    def _normalize_covariance_relation(
        self,
        context_covariance: torch.Tensor,
        query_covariance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize descriptors using labelled context statistics only."""
        context32 = context_covariance.float()
        query32 = query_covariance.float()
        center = context32.mean(dim=-2, keepdim=True)
        scale = torch.sqrt(
            (context32 - center).square().mean(dim=(-2, -1), keepdim=True)
            + self.covariance_relation_eps
        )
        return (context32 - center) / scale, (query32 - center) / scale

    def _covariance_relation_scores(
        self,
        context_covariance: torch.Tensor,
        context_labels: torch.Tensor,
        query_covariance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare query descriptors with labelled context distributions."""
        context_z, query_z = self._normalize_covariance_relation(
            context_covariance, query_covariance
        )
        batched = context_z.ndim == 3
        if not batched:
            context_z = context_z.unsqueeze(0)
            query_z = query_z.unsqueeze(0)
            context_labels = context_labels.unsqueeze(0)
        class_masks = [
            (context_labels == class_index).to(context_z.dtype)
            for class_index in range(self.num_classes)
        ]
        if any(torch.any(mask.sum(dim=-1) == 0) for mask in class_masks):
            raise ValueError(
                "Covariance relation diagnostics require every class in context."
            )
        prototypes = torch.stack(
            [
                torch.einsum("ec,ecd->ed", mask, context_z)
                / mask.sum(dim=-1, keepdim=True)
                for mask in class_masks
            ],
            dim=1,
        )
        separation = (
            prototypes[:, 1] - prototypes[:, 0]
        ).square().mean(dim=-1).sqrt()

        if self.covariance_relation_mode == "prototype_cosine":
            class_scores = torch.einsum(
                "eqd,ecd->eqc",
                F.normalize(query_z, dim=-1, eps=self.covariance_relation_eps),
                F.normalize(prototypes, dim=-1, eps=self.covariance_relation_eps),
            )
        elif self.covariance_relation_mode in {"standardized_distance", "gated_distance"}:
            differences = context_z.unsqueeze(2) - prototypes.unsqueeze(1)
            dispersions = torch.stack(
                [
                    (
                        differences[:, :, class_index].square().mean(dim=-1)
                        * class_masks[class_index]
                    ).sum(dim=-1)
                    / class_masks[class_index].sum(dim=-1)
                    for class_index in range(self.num_classes)
                ],
                dim=-1,
            ).clamp_min(self.covariance_relation_eps)
            distances = (
                query_z.unsqueeze(2) - prototypes.unsqueeze(1)
            ).square().mean(dim=-1)
            class_scores = -distances / dispersions.unsqueeze(1)
            if self.covariance_relation_mode == "gated_distance":
                gate = torch.sigmoid(
                    self.covariance_relation_gate_a * (separation.unsqueeze(1) - self.covariance_relation_gate_b)
                )
                class_scores = gate.unsqueeze(-1) * class_scores
        elif self.covariance_relation_mode == "learned_head":
            differences = context_z.unsqueeze(2) - prototypes.unsqueeze(1)
            dispersions = torch.stack(
                [
                    (
                        differences[:, :, class_index].square().mean(dim=-1)
                        * class_masks[class_index]
                    ).sum(dim=-1)
                    / class_masks[class_index].sum(dim=-1)
                    for class_index in range(self.num_classes)
                ],
                dim=-1,
            ).clamp_min(self.covariance_relation_eps)
            distances = (
                query_z.unsqueeze(2) - prototypes.unsqueeze(1)
            ).square().mean(dim=-1)
            d0 = distances[:, :, 0] / dispersions[:, 0].unsqueeze(1)
            d1 = distances[:, :, 1] / dispersions[:, 1].unsqueeze(1)
            delta_d = d0 - d1
            sep_feat = separation.unsqueeze(1).expand_as(d0)
            feats = torch.stack([d0, d1, delta_d, sep_feat], dim=-1)
            class_scores = self.covariance_relation_head(feats)
        else:
            distances = (
                query_z.unsqueeze(2) - context_z.unsqueeze(1)
            ).square().mean(dim=-1)
            pairwise = (
                context_z.unsqueeze(2) - context_z.unsqueeze(1)
            ).square().mean(dim=-1)
            context_count = context_z.shape[1]
            upper = torch.triu_indices(
                context_count, context_count, offset=1, device=context_z.device
            )
            base_temperature = pairwise[:, upper[0], upper[1]].median(dim=-1).values
            base_temperature = base_temperature.detach().clamp_min(
                self.covariance_relation_eps
            )
            scale_scores = []
            for kernel_scale in self.covariance_relation_kernel_scales:
                kernel = torch.exp(
                    -distances
                    / (base_temperature[:, None, None] * kernel_scale)
                )
                class_kernel_scores = torch.stack(
                    [
                        torch.einsum("eqc,ec->eq", kernel, mask)
                        / mask.sum(dim=-1).unsqueeze(-1)
                        for mask in class_masks
                    ],
                    dim=-1,
                )
                scale_scores.append(
                    class_kernel_scores.clamp_min(
                        self.covariance_relation_eps
                    ).log()
                )
            class_scores = torch.stack(scale_scores).mean(dim=0)

        margin = class_scores[..., 1] - class_scores[..., 0]
        # Query-count-invariant margin: raw tanh (no batch normalization). The
        # old margin_rms averaged over the QUERY axis, coupling batched queries
        # and degenerating to tanh(+-1) at 1-query eval while training used
        # 5-12 queries/episode. Margins are already context-dispersion-scaled
        # (d0/d1 = distance/dispersion), so tanh(margin) is per-query and
        # 1-query == multi-query == training.
        if self.covariance_relation_margin_activation == "identity":
            bounded_margin = margin / self.covariance_relation_log_temperature.exp().clamp(
                1e-3, 1e4
            )
        else:
            bounded_margin = torch.tanh(margin)
        logits = torch.stack((-0.5 * bounded_margin, 0.5 * bounded_margin), dim=-1)
        if not batched:
            return logits.squeeze(0), separation.squeeze(0)
        return logits, separation

    def _covariance_subspace_features(
        self,
        context_covariance: torch.Tensor,
        context_labels: torch.Tensor,
        query_covariance: torch.Tensor,
        rank: int,
        whiten: bool,
        shrinkage: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract a context-labelled low-rank covariance subspace."""
        if context_covariance.ndim == 4 and query_covariance.ndim == 4:
            results = [
                self._covariance_subspace_features(
                    context_covariance[index], context_labels[index],
                    query_covariance[index], rank, whiten, shrinkage,
                )
                for index in range(context_covariance.shape[0])
            ]
            return tuple(torch.stack(values) for values in zip(*results))
        if context_covariance.ndim != 3 or query_covariance.ndim != 3:
            raise ValueError(
                "covariance tensors must be [bags, dim, dim] or batched."
            )
        if context_covariance.shape[1:] != query_covariance.shape[1:]:
            raise ValueError("context and query covariance dimensions must match.")
        dimension = context_covariance.shape[-1]
        if not 1 <= rank <= dimension:
            raise ValueError("rank must be in [1, covariance dimension].")
        class_means = []
        for class_index in range(self.num_classes):
            members = context_covariance[context_labels == class_index].float()
            if members.numel() == 0:
                raise ValueError("Every class must occur in context covariance.")
            class_means.append(members.mean(dim=0))
        delta = class_means[1] - class_means[0]
        if whiten:
            pooled = context_covariance.float().mean(dim=0)
            trace_scale = pooled.diagonal().mean().clamp_min(
                self.covariance_relation_eps
            )
            pooled = (1.0 - shrinkage) * pooled + shrinkage * trace_scale * torch.eye(
                dimension, device=pooled.device, dtype=pooled.dtype
            )
            values, vectors = torch.linalg.eigh(pooled)
            safe_values = values.clamp_min(1e-5)
            whitening = (vectors * safe_values.rsqrt().unsqueeze(0)) @ vectors.T
            operator = whitening @ delta @ whitening
            if torch.isnan(operator).any():
                whitening = torch.eye(
                    dimension, device=delta.device, dtype=delta.dtype
                )
                operator = delta
        else:
            whitening = torch.eye(
                dimension, device=delta.device, dtype=delta.dtype
            )
            operator = delta
        eigenvalues, eigenvectors = torch.linalg.eigh(operator)
        selected = eigenvalues.abs().topk(rank).indices
        filters = whitening @ eigenvectors[:, selected]
        context_variance = torch.einsum(
            "di,bdk,ki->bi", filters, context_covariance.float(), filters
        ).clamp_min(self.covariance_relation_eps)
        query_variance = torch.einsum(
            "di,bdk,ki->bi", filters, query_covariance.float(), filters
        ).clamp_min(self.covariance_relation_eps)
        return context_variance.log(), query_variance.log(), eigenvalues[selected]


    def _covariance_only_forward(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query: dict[str, torch.Tensor],
        batched: bool,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """CV-only evidence path (docs SS68).

        Computes ONLY

            final = cov_res_scale*CV-1 + cov_rel_res_scale*CV-2

        and never touches the structural tokens, class memories, population
        attention, or rare branch -- none of which the aggregator even produces
        in this mode. `tests/test_ridge_ablation.py::TestCovarianceOnly` pins
        this against a full-branch model carrying the same weights, so if this
        implementation ever drifts from the branch it replaces, that test fails
        rather than a run silently training a different model (the SS62-7
        duplicate-drift failure mode).
        """
        ridge = (
            self._abundance_ridge_logits_batched
            if batched
            else self._abundance_ridge_logits
        )
        covariance_ridge_logits = ridge(
            context["covariance_sketch"],
            context_labels,
            query["covariance_sketch"],
            ridge_lambda=self.covariance_ridge_log_lambda,
            dual=True,
        )
        if self.force_covariance_ridge_zero:
            covariance_ridge_logits = covariance_ridge_logits.new_zeros(
                covariance_ridge_logits.shape
            )
        covariance_ridge_scale = self.covariance_ridge_log_scale.exp().clamp(
            0.1, 100.0
        )
        covariance_logits = covariance_ridge_scale * covariance_ridge_logits
        covariance_residual_scale = torch.sigmoid(self.covariance_residual_logit)
        logits = covariance_residual_scale * covariance_logits

        covariance_relation_logits = covariance_logits.new_zeros(
            covariance_logits.shape
        )
        covariance_relation_class_separation = covariance_logits.new_zeros(
            covariance_logits.shape[0] if batched else ()
        )
        if self.covariance_relation_enabled:
            if self.covariance_relation_granularity == "subspace":
                with torch.autocast(
                    device_type=context["covariance_matrix"].device.type,
                    enabled=False,
                ):
                    context_feature, query_feature, selected_eigenvalues = (
                        self._covariance_subspace_features(
                            context["covariance_matrix"].float(), context_labels,
                            query["covariance_matrix"].float(),
                            rank=self.covariance_relation_subspace_rank,
                            whiten=self.covariance_relation_subspace_whiten,
                            shrinkage=self.covariance_relation_subspace_shrinkage,
                        )
                    )
                covariance_relation_logits, covariance_relation_class_separation = (
                    self._covariance_relation_scores(
                        context_feature, context_labels, query_feature
                    )
                )
            else:
                raise ValueError(
                    "Unsupported covariance_relation_granularity "
                    f"{self.covariance_relation_granularity!r}."
                )
            if not self.covariance_relation_diagnostic_only:
                logits = logits + (
                    self.covariance_relation_residual_scale
                    * covariance_relation_logits
                )
        if not return_auxiliary:
            return logits
        # In the dense path `training_step` slices every auxiliary entry per
        # episode (`value[episode]`), so scalars must carry an episode axis --
        # exactly what the full batched path does with `.expand(episodes)`.
        # Returning 0-dim scalars here is what made E=4 fail with
        # "invalid index of a 0-dim tensor".
        if batched:
            episodes = covariance_logits.shape[0]
            expand = lambda t: (
                t.expand(episodes) if t.dim() == 0 else t
            )
            relation_scale = torch.full(
                (episodes,),
                float(self.covariance_relation_residual_scale),
                device=covariance_logits.device,
                dtype=torch.float32,
            )
        else:
            expand = lambda t: t
            relation_scale = self.covariance_relation_residual_scale
        return logits, {
            "covariance_logits": covariance_logits,
            "covariance_ridge_logits": covariance_ridge_logits,
            "covariance_ridge_scale": expand(covariance_ridge_scale),
            "covariance_residual_scale": expand(covariance_residual_scale),
            # Mirrors the full batched path: a tensor with an episode axis,
            # not a bool, so `training_step`'s per-episode slice works. Keeping
            # it preserves the CV-2 diagnostics (auroc, class separation) --
            # which matter more here than anywhere, CV-2 being one of the two
            # branches this arm keeps.
            "covariance_relation_enabled": (
                torch.tensor(
                    self.covariance_relation_enabled,
                    device=covariance_logits.device,
                ).expand(covariance_logits.shape[0])
                if batched
                else self.covariance_relation_enabled
            ),
            "covariance_relation_logits": covariance_relation_logits,
            "covariance_relation_class_separation": (
                covariance_relation_class_separation
            ),
            "covariance_relation_residual_scale": relation_scale,
        }

    def _validate_representation(
        self,
        representation: dict[str, torch.Tensor],
        name: str,
    ) -> None:
        """The representation contract is exact: no missing keys, no extra ones.

        Dead keys are ABSENT rather than present-and-zero (docs SS68). Absent
        means a stray consumer raises KeyError at the offending line instead of
        silently averaging zeros into a live branch -- which is how three
        consumers were caught when CV-only was introduced. If a new consumer
        hits KeyError here, guard the branch; do not zero-fill.
        """
        expected_keys = {"covariance_sketch", "covariance_matrix"}
        missing = expected_keys - set(representation)
        if missing:
            raise ValueError(
                f"{name} covariance-only representation is missing "
                f"{sorted(missing)}."
            )
        extra = set(representation) - expected_keys
        if extra:
            raise ValueError(
                f"{name} covariance-only representation carries "
                f"{sorted(extra)}; CV-only must not compute them."
            )

    def _abundance_ridge_logits(
        self,
        context_metadata: torch.Tensor,
        context_labels: torch.Tensor,
        query_metadata: torch.Tensor,
        ridge_lambda: torch.Tensor | None = None,
        dual: bool = False,
    ) -> torch.Tensor:
        """Class-balanced ridge on identity-aligned slot statistics."""
        output_dtype = query_metadata.dtype
        context = context_metadata.float().flatten(start_dim=1)
        query = query_metadata.float().flatten(start_dim=1)
        center = context.mean(dim=0, keepdim=True)
        context = context - center
        query = query - center
        rms = context.square().mean().sqrt().clamp_min(1e-6)
        context = context / rms
        query = query / rms
        targets = F.one_hot(
            context_labels.long(), num_classes=self.num_classes
        ).float()
        class_counts = torch.bincount(
            context_labels.long(), minlength=self.num_classes
        )
        sample_weight = class_counts.float().reciprocal()[context_labels.long()]
        total_weight = sample_weight.sum().clamp_min(1e-12)
        feature_mean = (sample_weight.unsqueeze(-1) * context).sum(
            dim=0, keepdim=True
        ) / total_weight
        target_mean = (sample_weight.unsqueeze(-1) * targets).sum(
            dim=0, keepdim=True
        ) / total_weight
        centered_context = context - feature_mean
        centered_targets = targets - target_mean
        root_weight = sample_weight.sqrt().unsqueeze(-1)
        weighted_design = centered_context * root_weight
        weighted_targets = centered_targets * root_weight
        if ridge_lambda is None:
            ridge_lambda = self.abundance_ridge_log_lambda
        ridge_lambda = ridge_lambda.exp().clamp(1e-4, 1e4)
        with torch.autocast(device_type=context.device.type, enabled=False):
            design32 = weighted_design.float()
            targets32 = weighted_targets.float()
            if dual:
                dual_coefficients = solve_ridge_system(
                    design32 @ design32.T, targets32, ridge_lambda.float()
                )
                coefficients = design32.T @ dual_coefficients
            else:
                coefficients = solve_ridge_system(
                    design32.T @ design32,
                    design32.T @ targets32,
                    ridge_lambda.float(),
                )
            intercept = target_mean.float() - feature_mean.float() @ coefficients
            logits = query.float() @ coefficients + intercept
        if not torch.isfinite(logits).all():
            raise RuntimeError("The abundance ridge logits contain NaN or Inf.")
        return logits.to(output_dtype)

    def _abundance_ridge_logits_batched(
        self,
        context_metadata: torch.Tensor,
        context_labels: torch.Tensor,
        query_metadata: torch.Tensor,
        ridge_lambda: torch.Tensor | None = None,
        dual: bool = False,
    ) -> torch.Tensor:
        output_dtype = query_metadata.dtype
        context = context_metadata.float().flatten(start_dim= 2 )
        query = query_metadata.float().flatten(start_dim= 2 )
        center = context.mean(dim=1, keepdim=True)
        context = context - center
        query = query - center
        rms = context.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        context = context / rms
        query = query / rms
        targets = F.one_hot(
            context_labels.long(), num_classes=self.num_classes
        ).float()
        class_counts = F.one_hot(
            context_labels.long(), num_classes=self.num_classes
        ).sum(dim=1)
        sample_weight = class_counts.float().reciprocal().gather(
            1, context_labels.long()
        )
        total_weight = sample_weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
        feature_mean = (sample_weight.unsqueeze(-1) * context).sum(
            dim=1, keepdim=True
        ) / total_weight.unsqueeze(-1)
        target_mean = (sample_weight.unsqueeze(-1) * targets).sum(
            dim=1, keepdim=True
        ) / total_weight.unsqueeze(-1)
        centered_context = context - feature_mean
        centered_targets = targets - target_mean
        root_weight = sample_weight.sqrt().unsqueeze(-1)
        weighted_design = centered_context * root_weight
        weighted_targets = centered_targets * root_weight
        if ridge_lambda is None:
            ridge_lambda = self.abundance_ridge_log_lambda
        ridge_lambda = ridge_lambda.exp().clamp(1e-4, 1e4)
        with torch.autocast(device_type=context.device.type, enabled=False):
            design32 = weighted_design.float()
            targets32 = weighted_targets.float()
            if dual:
                dual_coefficients = solve_ridge_system(
                    design32 @ design32.transpose(1, 2),
                    targets32,
                    ridge_lambda.float(),
                )
                coefficients = design32.transpose(1, 2) @ dual_coefficients
            else:
                coefficients = solve_ridge_system(
                    design32.transpose(1, 2) @ design32,
                    design32.transpose(1, 2) @ targets32,
                    ridge_lambda.float(),
                )
            intercept = target_mean.float() - feature_mean.float() @ coefficients
            logits = query.float() @ coefficients + intercept
        if not torch.isfinite(logits).all():
            raise RuntimeError(
                "The batched abundance ridge logits contain NaN or Inf."
            )
        return logits.to(output_dtype)

    def forward_batched(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query: dict[str, torch.Tensor],
        query_instances: torch.Tensor,
        query_cell_mask: torch.Tensor | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # NOTE: no `_validate_representation` here, unlike the ragged `forward`.
        # That asymmetry predates the prune and is preserved deliberately --
        # adding the check would be a behaviour change smuggled into a deletion.
        return self._covariance_only_forward(
            context, context_labels, query,
            batched=True, return_auxiliary=return_auxiliary,
        )

    def forward(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query: dict[str, torch.Tensor],
        query_instances: Sequence[torch.Tensor],
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_representation(context, "context")
        self._validate_representation(query, "query")
        return self._covariance_only_forward(
            context, context_labels, query,
            batched=False, return_auxiliary=return_auxiliary,
        )


class BaseModel(nn.Module):
    """Compose hybrid population aggregation with class-memory meta learning."""

    architecture_version = 22

    def __init__(
        self,
        input_dim: int = 512,
        aggregator_num_slots: int = 12,
        aggregator_num_density_slots: int = 8,
        aggregator_context_samples_per_bag: int = 32,
        aggregator_assignment_temperature: float = 0.1,
        aggregator_density_refinement_steps: int = 4,
        aggregator_density_temperature: float = 0.15,
        aggregator_slot_rare_fraction: float = 0.05,
        aggregator_tail_fractions: Sequence[float] = (0.01, 0.05, 0.15),
        aggregator_absolute_tail_ks: Sequence[int] = (),
        aggregator_ccts_lambdas: Sequence[float] = (),
        aggregator_ccts_tau: float = 0.5,
        aggregator_min_tail_instances: int = 1,
        bag_centered_representation: bool = True,
        bag_centered_l2_normalize: bool = True,
        bag_representation: str = "legacy",
        global_summary: str = "centered_spread",
        use_raw_mean_branch: bool = False,
        raw_stat_tokens: Sequence[str] = (),
        aggregator_covariance_sketch_dim: int | None = None,
        aggregator_covariance_matrix_dim: int | None = 32,
        aggregator_covariance_slopes: "Sequence[float] | None" = None,
        aggregator_covariance_mode: str = "covariance",
        aggregator_covariance_shrinkage: float = 0.0,
        meta_enable_covariance_ridge: bool = True,
        meta_covariance_ridge_logit_scale: float = 2.0,
        meta_covariance_residual_scale: float = 0.25,
        mean_pool_structured_tokens: bool = False,
        project_structured_tokens: bool = False,
        typed_bag_preserving_branch: bool = False,
        covariance_relation: dict[str, object] | None = None,
        cls_token_pooling: bool = False,
        cls_token_heads: int = 4,
        aggregator_slot_latent_dim: int | None = None,
        aggregator_slot_query_latent_dim: int | None = None,
        aggregator_slot_affinity_dim: int | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.aggregator = StructuredEpisodePopulationAggregator(
            input_dim=self.input_dim,
            num_slots=aggregator_num_slots,
            num_density_slots=aggregator_num_density_slots,
            context_samples_per_bag=aggregator_context_samples_per_bag,
            assignment_temperature=aggregator_assignment_temperature,
            density_refinement_steps=aggregator_density_refinement_steps,
            density_temperature=aggregator_density_temperature,
            slot_rare_fraction=aggregator_slot_rare_fraction,
            tail_fractions=aggregator_tail_fractions,
            absolute_tail_ks=aggregator_absolute_tail_ks,
            ccts_lambdas=aggregator_ccts_lambdas,
            ccts_tau=aggregator_ccts_tau,
            min_tail_instances=aggregator_min_tail_instances,
            bag_centered_representation=bag_centered_representation,
            bag_centered_l2_normalize=bag_centered_l2_normalize,
            bag_representation=bag_representation,
            global_summary=global_summary,
            use_raw_mean_branch=use_raw_mean_branch,
            raw_stat_tokens=raw_stat_tokens,
            covariance_sketch_dim=aggregator_covariance_sketch_dim,
            covariance_matrix_dim=aggregator_covariance_matrix_dim,
            covariance_slopes=aggregator_covariance_slopes,
            covariance_mode=aggregator_covariance_mode,
            covariance_shrinkage=aggregator_covariance_shrinkage,
            include_cls_token=cls_token_pooling,
            cls_token_heads=cls_token_heads,
            slot_latent_dim=aggregator_slot_latent_dim,
            slot_query_latent_dim=aggregator_slot_query_latent_dim,
            slot_affinity_dim=aggregator_slot_affinity_dim,
        )
        relation_config = dict(covariance_relation or {})
        self.aggregator.slot_covariance_descriptor = str(
            relation_config.get("descriptor", "correlation")
        )
        self.aggregator.emit_covariance_matrix = bool(
            relation_config.get("enabled", False)
            and relation_config.get("granularity") == "subspace"
        )
        # Bag = 1 global summary + num_slots * 3 (center/spread/rare) slot
        # statistics + len(tail_fractions) + len(absolute_tail_ks) + len(ccts_lambdas) tail tokens
        # (+ 1 cls-pooled token, see cls_token_pooling). For the v24 learned
        # projection these are concatenated and linearly mapped to one token.
        structured_tokens_per_bag = (
            1
            + 3 * int(aggregator_num_slots)
            + len(tuple(aggregator_tail_fractions))
            + len(tuple(aggregator_absolute_tail_ks))
            + len(tuple(aggregator_ccts_lambdas))
            + (1 if cls_token_pooling else 0)
            + len(tuple(raw_stat_tokens))
        )


        self.meta_classifier = StructuredPopulationMetaClassifier(
            token_dim=self.input_dim,
            covariance_ridge_logit_scale=meta_covariance_ridge_logit_scale,
            covariance_residual_scale=meta_covariance_residual_scale,
            covariance_relation=covariance_relation,
            meta_enable_covariance_ridge=meta_enable_covariance_ridge,
            num_classes=self.num_classes,
        )
        # v26 = cls_token_pooling (CLS cross-attention over raw cells, see
        # configs/train_v26_medium_cls_token_pool.yaml). This is the design
        # that ended up using the v26 slot -- the earlier EC-MoE proposal
        # that first claimed that number was rejected before implementation
        # (docs/history/architecture_v26_proposal.md) after the E2
        # oracle-gating check found zero headroom in episode-conditional
        # fusion (docs/architecture_v28_proposal.md SS6.1).
        self.architecture_version = (
            26 if cls_token_pooling
            else 25 if typed_bag_preserving_branch
            else 24 if project_structured_tokens
            else 23 if mean_pool_structured_tokens
            else 22
        )
        self.register_buffer(
            "_architecture_version",
            torch.tensor(self.architecture_version, dtype=torch.long),
            persistent=True,
        )

    @staticmethod
    def _normalize_mask_index(
        mask_index: torch.Tensor | Sequence[int] | int,
        num_bags: int,
        device: torch.device,
    ) -> torch.Tensor:
        index = torch.as_tensor(mask_index, device=device, dtype=torch.long).flatten()
        if index.numel() == 0:
            raise ValueError("At least one query bag must be masked.")
        if torch.any((index < 0) | (index >= num_bags)):
            raise IndexError("mask_index contains an out-of-range bag index.")
        if torch.unique(index).numel() != index.numel():
            raise ValueError("mask_index cannot contain duplicate bag indices.")
        return index

    def forward_episode_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask_index: torch.Tensor,
        return_auxiliary: bool = False,
        cell_mask: torch.Tensor | None = None,
        bag_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run dense equal-shape episodes through one batched aggregator.

        `cell_mask` ([episodes, bags, instances], bool) and `bag_mask`
        ([episodes, bags], bool) mark the real cells/bags of padded ragged
        (B2b) batches; padded entries are excluded from every statistic. The
        cell-level aggregator stays fully batched; the (cheap, bag-token-level)
        meta-classifier runs per episode because padded episodes have different
        context sizes.
        """
        if x.ndim != 4:
            raise ValueError(
                "Batched x must be [episodes, bags, instances, input_dim]."
            )
        episodes, num_bags, num_instances, input_dim = x.shape
        if input_dim != self.input_dim or y.shape != (episodes, num_bags):
            raise ValueError("Batched x/y shapes are incompatible.")
        if mask_index.ndim != 2 or mask_index.shape[0] != episodes:
            raise ValueError("Batched mask_index must be [episodes, queries].")
        if torch.any((mask_index < 0) | (mask_index >= num_bags)):
            raise IndexError("mask_index contains an out-of-range bag index.")
        if cell_mask is not None and cell_mask.shape != x.shape[:3]:
            raise ValueError("cell_mask must be [episodes, bags, instances].")
        if bag_mask is not None and bag_mask.shape != (episodes, num_bags):
            raise ValueError("bag_mask must be [episodes, bags].")

        is_context = torch.ones(episodes, num_bags, dtype=torch.bool, device=x.device)
        is_context.scatter_(1, mask_index.long(), False)
        if bag_mask is not None:
            is_context = is_context & bag_mask
        flat_x = x.reshape(episodes * num_bags, num_instances, input_dim)
        flat_cell_mask = (
            cell_mask.reshape(episodes * num_bags, num_instances)
            if cell_mask is not None
            else None
        )
        flat_valid_count = (
            flat_cell_mask.sum(dim=-1).clamp_min(1)
            if flat_cell_mask is not None
            else None
        )
        # Downstream needs only the covariance sketch and matrix (docs SS68).
        # `centered_delta` -- the raw per-bag centring -- does not depend on the
        # context-pool statistics, so this skips the pool stats, the poolz_l2
        # standardisation of every cell, the per-episode anchors (top-k over
        # cells) and the raw-stat tokens outright. That skip is what makes the
        # training forward 5.9x faster and the peak VRAM 3.4x smaller.
        if flat_cell_mask is None:
            centered_delta = flat_x - flat_x.mean(dim=-2, keepdim=True)
        else:
            masked = flat_x.masked_fill(~flat_cell_mask.unsqueeze(-1), 0.0)
            centered_delta = masked - (
                masked.sum(dim=-2, keepdim=True)
                / flat_valid_count.unsqueeze(-1).unsqueeze(-1)
            )
            centered_delta = centered_delta.masked_fill(
                ~flat_cell_mask.unsqueeze(-1), 0.0
            )
        covariance_sketch = self.aggregator._covariance_sketch(
            centered_delta, count=flat_valid_count
        )
        # Passed through unstandardised: the only consumer left,
        # `_covariance_sketch`, already has its input. `classification_x` is the
        # same tensor viewed per episode -- a reshape, not a recomputation --
        # and feeds `query_instances` below.
        classification_flat = flat_x
        classification_x = x
        global_summary = None
        raw_stats = None
        per_bag_anchors = None
        flat_representation = self.aggregator._forward_dense(
            classification_flat,
            per_bag_anchors,
            return_auxiliary=False,
            global_summary=global_summary,
            covariance_sketch=covariance_sketch,
            centered_delta=centered_delta,
            raw_stats=raw_stats,
            cell_mask=flat_cell_mask,
        )
        representation = {
            name: tokens.reshape(episodes, num_bags, *tokens.shape[1:])
            for name, tokens in flat_representation.items()
        }

        if cell_mask is None and bag_mask is None:
            # Uniform-context dense path (arm B): batch the meta-classifier too.
            context_count = num_bags - mask_index.shape[1]
            context_index = torch.nonzero(is_context, as_tuple=False)[:, 1].reshape(
                episodes, context_count
            )

            def gather_bags(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
                view_shape = index.shape + (1,) * (tokens.ndim - 2)
                expanded = index.reshape(view_shape).expand(
                    index.shape + tokens.shape[2:]
                )
                return tokens.gather(1, expanded)

            context = {
                name: gather_bags(tokens, context_index)
                for name, tokens in representation.items()
            }
            query = {
                name: gather_bags(tokens, mask_index.long())
                for name, tokens in representation.items()
            }
            context_labels = y.gather(1, context_index)
            query_instances = gather_bags(classification_x, mask_index.long())
            return self.meta_classifier.forward_batched(
                context=context,
                context_labels=context_labels,
                query=query,
                query_instances=query_instances,
                return_auxiliary=return_auxiliary,
            )

        # Padded ragged path: context sizes vary per episode, so the
        # meta-classifier (over ~100 bag tokens) runs per episode while the
        # expensive cell-level aggregator above stayed batched. Padded query
        # cells are masked so the rare-evidence branch ignores them.
        logits_list: list[torch.Tensor] = []
        auxiliary_list: list[dict[str, torch.Tensor]] = []
        for episode in range(episodes):
            context_index = torch.nonzero(
                is_context[episode], as_tuple=False
            ).flatten()
            context = {
                name: tokens[episode][context_index].unsqueeze(0)
                for name, tokens in representation.items()
            }
            query = {
                name: tokens[episode][mask_index[episode]].unsqueeze(0)
                for name, tokens in representation.items()
            }
            context_labels = y[episode][context_index].unsqueeze(0)
            query_instances = (
                classification_x[episode][mask_index[episode]].unsqueeze(0)
            )
            query_cell_mask = (
                cell_mask[episode][mask_index[episode]].unsqueeze(0)
                if cell_mask is not None
                else None
            )
            result = self.meta_classifier.forward_batched(
                context=context,
                context_labels=context_labels,
                query=query,
                query_instances=query_instances,
                query_cell_mask=query_cell_mask,
                return_auxiliary=return_auxiliary,
            )
            if return_auxiliary:
                logits_list.append(result[0])
                auxiliary_list.append(result[1])
            else:
                logits_list.append(result)
        logits = torch.cat(logits_list, dim=0)
        if not return_auxiliary:
            return logits
        combined = {
            key: torch.cat([aux[key] for aux in auxiliary_list], dim=0)
            for key in auxiliary_list[0]
        }
        return logits, combined

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        y: torch.Tensor,
        mask_index: torch.Tensor | Sequence[int] | int,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(x, torch.Tensor) and x.ndim == 4:
            # 4D Batched Forward Mode
            E, N, num_instances, feature_dim = x.shape
            device = x.device
            res_logits, res_aux = [], []
            for b in range(E):
                mask_b = mask_index[b] if isinstance(mask_index, torch.Tensor) and mask_index.ndim >= 1 else mask_index
                res = self.forward(
                    x[b], y[b], mask_index=mask_b, return_auxiliary=return_auxiliary
                )
                if return_auxiliary:
                    res_logits.append(res[0])
                    res_aux.append(res[1])
                else:
                    res_logits.append(res)

            logits_cat = torch.cat(res_logits, dim=0)
            if return_auxiliary:
                aux_combined = {
                    k: torch.stack([aux[k] for aux in res_aux if k in aux])
                    for k in res_aux[0].keys()
                } if res_aux else {}
                return logits_cat, aux_combined
            return logits_cat
        if isinstance(x, torch.Tensor):
            if x.ndim != 3:
                raise ValueError(
                    "Dense x must have shape [bags, instances, input_dim]."
                )
            num_bags = x.shape[0]
        else:
            num_bags = len(x)
        if y.ndim != 1 or y.shape[0] != num_bags:
            raise ValueError("y must have shape [bags].")
        if num_bags < self.num_classes + 1:
            raise ValueError("An episode needs context bags plus at least one query.")

        query_index = self._normalize_mask_index(
            mask_index, num_bags=num_bags, device=y.device
        )
        normalized_bags = self.aggregator._normalize_bags(x)
        # is_context is built BEFORE the view so pool-standardized representations
        # can use the same context-only statistics the aggregator will use below;
        # otherwise this duplicate _bag_view would feed the rare-evidence/MIL
        # branches differently-normalized cells than the slot branch.
        is_context = torch.ones(num_bags, dtype=torch.bool, device=y.device)
        is_context[query_index] = False
        if self.aggregator.bag_representation in ("poolz", "poolz_l2"):
            pool_mean, pool_std = self.aggregator._context_pool_stats(
                normalized_bags, is_context
            )
        else:
            pool_mean = pool_std = None
        # Only the QUERY bags' views are needed here (they feed the rare-evidence
        # / MIL branches). Building the view for every bag materialized a second
        # full [total_cells, dim] copy on top of the aggregator's own -- ~12 GB
        # per full-tile PathoBench episode, for data that was then indexed down
        # to a single query slide (docs v35 §3). Restricting it to the queried
        # bags is exact: `_bag_view` is per-bag and the pool statistics it uses
        # are unchanged.
        query_instances = [
            self.aggregator._bag_view(normalized_bags[index], pool_mean, pool_std)[0]
            for index in query_index.detach().cpu().tolist()
        ]
        if isinstance(x, torch.Tensor):
            query_instances = torch.stack(query_instances)
        if return_auxiliary:
            representation, aggregator_auxiliary = self.aggregator(
                x,
                context_mask=is_context,
                return_auxiliary=True,
            )
        else:
            representation = self.aggregator(x, context_mask=is_context)
            aggregator_auxiliary = None
        context_representation = {
            name: tokens[is_context] for name, tokens in representation.items()
        }
        query_representation = {
            name: tokens[query_index] for name, tokens in representation.items()
        }
        result = self.meta_classifier(
            context=context_representation,
            context_labels=y[is_context],
            query=query_representation,
            query_instances=query_instances,
            return_auxiliary=return_auxiliary,
        )
        if not return_auxiliary:
            return result
        logits, auxiliary = result
        # No bag/slot/tail tokens here: they are never built, and reporting
        # them as zeros would hand callers a plausible-looking tensor for
        # something that was never computed (docs SS68).
        return logits, {
            "context_mask": is_context,
            "aggregator": aggregator_auxiliary,
            **auxiliary,
        }
