"""Mean bag aggregation with a label-equivariant set meta-classifier."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    """Represent each unordered instance bag by its exact valid-instance mean."""

    def __init__(self, input_dim: int = 512) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        self.input_dim = int(input_dim)

    def forward(
        self,
        instances: torch.Tensor,
        instance_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if instances.ndim < 2:
            raise ValueError(
                "instances must have at least [..., num_instances, input_dim]."
            )
        if instances.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected instance dimension {self.input_dim}, "
                f"got {instances.shape[-1]}."
            )
        if instances.shape[-2] == 0:
            raise ValueError("A bag must contain at least one instance.")
        if instance_mask is None:
            return instances.mean(dim=-2)

        expected_shape = instances.shape[:-1]
        if instance_mask.shape != expected_shape:
            raise ValueError(
                "instance_mask must match instances without the feature axis: "
                f"expected {expected_shape}, got {instance_mask.shape}."
            )
        valid = instance_mask.to(device=instances.device, dtype=torch.bool)
        counts = valid.sum(dim=-1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("Every bag must contain at least one valid instance.")
        weights = valid.to(dtype=instances.dtype).unsqueeze(-1)
        return (instances * weights).sum(dim=-2) / counts.to(instances.dtype)


class MeanResidualAggregator(nn.Module):
    """Exact bag mean plus count-adaptive residuals from unusual instances.

    Each residual head selects a fraction of a bag rather than a fixed number
    of instances. A 1% head therefore uses one instance for a 100-cell bag and
    ten instances for a 1000-cell bag. The base token remains the exact raw
    instance mean; learned residuals can only add information to that path.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        tail_fractions: Sequence[float] = (0.01, 0.05, 0.15),
        min_tail_instances: int = 1,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive.")
        fractions = tuple(float(fraction) for fraction in tail_fractions)
        if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
            raise ValueError("tail_fractions must contain values in (0, 1].")
        if min_tail_instances < 1:
            raise ValueError("min_tail_instances must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.tail_fractions = fractions
        self.min_tail_instances = int(min_tail_instances)
        self.instance_norm = nn.LayerNorm(input_dim)
        self.instance_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tail_scorer = nn.Linear(hidden_dim, len(fractions))
        self.residual_projection = nn.Sequential(
            nn.Linear(hidden_dim * len(fractions), input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )
        self.residual_logit_scale = nn.Parameter(torch.tensor(-2.0))

    def _normalize_bags(
        self,
        instances: torch.Tensor | Sequence[torch.Tensor],
        instance_mask: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], bool]:
        single_bag = isinstance(instances, torch.Tensor) and instances.ndim == 2
        if isinstance(instances, torch.Tensor):
            if instances.ndim == 2:
                instances = instances.unsqueeze(0)
            if instances.ndim != 3:
                raise ValueError(
                    "instances must be [bags, instances, features], one "
                    "[instances, features] bag, or a sequence of bags."
                )
            if instances.shape[-1] != self.input_dim:
                raise ValueError(f"Expected instance dimension {self.input_dim}.")
            if instance_mask is not None:
                if instance_mask.shape != instances.shape[:2]:
                    raise ValueError("instance_mask must have shape [bags, instances].")
                valid = instance_mask.to(device=instances.device, dtype=torch.bool)
                bags = [bag[mask] for bag, mask in zip(instances, valid)]
            else:
                bags = list(instances.unbind(0))
        else:
            if instance_mask is not None:
                raise ValueError("instance_mask is only supported for dense tensors.")
            bags = list(instances)
        if not bags or any(bag.ndim != 2 for bag in bags):
            raise ValueError("Every bag must be a non-empty rank-2 tensor.")
        if any(bag.shape[0] == 0 or bag.shape[1] != self.input_dim for bag in bags):
            raise ValueError(
                f"Every bag must contain [instances, {self.input_dim}] values."
            )
        return bags, single_bag

    def forward(
        self,
        instances: torch.Tensor | Sequence[torch.Tensor],
        instance_mask: torch.Tensor | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bags, single_bag = self._normalize_bags(instances, instance_mask)
        counts = [bag.shape[0] for bag in bags]
        concatenated = torch.cat(bags, dim=0)
        encoded = self.instance_projection(self.instance_norm(concatenated))
        encoded_bags = encoded.split(counts, dim=0)

        tokens: list[torch.Tensor] = []
        selected_counts: list[list[int]] = []
        for raw_bag, encoded_bag in zip(bags, encoded_bags):
            raw_mean = raw_bag.mean(dim=0)
            encoded_deviation = encoded_bag - encoded_bag.mean(dim=0, keepdim=True)
            scores = self.tail_scorer(encoded_deviation)
            pooled_heads: list[torch.Tensor] = []
            bag_selected_counts: list[int] = []
            for head, fraction in enumerate(self.tail_fractions):
                count = min(
                    raw_bag.shape[0],
                    max(
                        self.min_tail_instances,
                        int(math.ceil(fraction * raw_bag.shape[0])),
                    ),
                )
                selected_score, selected_index = scores[:, head].topk(count)
                selected_value = encoded_deviation[selected_index]
                weights = torch.softmax(selected_score.float(), dim=0).to(
                    selected_value.dtype
                )
                pooled_heads.append((weights.unsqueeze(-1) * selected_value).sum(dim=0))
                bag_selected_counts.append(count)
            residual = self.residual_projection(torch.cat(pooled_heads, dim=-1))
            residual_scale = torch.sigmoid(self.residual_logit_scale)
            tokens.append(raw_mean + residual_scale * residual)
            selected_counts.append(bag_selected_counts)

        result = torch.stack(tokens)
        if single_bag:
            result = result.squeeze(0)
        if not return_auxiliary:
            return result
        auxiliary = {
            "instance_counts": torch.tensor(counts, device=result.device),
            "tail_counts": torch.tensor(selected_counts, device=result.device),
            "residual_scale": torch.sigmoid(self.residual_logit_scale),
        }
        return result, auxiliary


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


class ClassTokenPooling(nn.Module):
    """Learned cross-attention pooling over every raw cell in a bag.

    A single learned CLS query attends over all N cell instances (key/value)
    and the attended result becomes one additional structured token. Unlike
    the population-slot tokens, which are built from a fixed, per-episode
    k-means-style partition (`_context_anchors`), this aggregation has no
    dependence on that partition and is trained end-to-end against the
    classification loss -- it can in principle learn to weight whichever
    cells the loss rewards, rather than being confined to the anchors'
    population coordinate system.

    Cross-attention (CLS as query, cells as key/value) rather than full
    self-attention among cells: self-attention over up to ~1,500 raw cells
    per bag costs O(N^2) per bag, roughly two orders of magnitude more than
    the O(N * num_slots) the rest of the aggregator spends per bag. A single
    query token attending over the cells costs O(N) and captures the same
    "one learned summary of the whole bag" intent without that blowup.
    """

    def __init__(self, token_dim: int = 512, num_heads: int = 4) -> None:
        super().__init__()
        if token_dim < 1 or num_heads < 1:
            raise ValueError("token_dim and num_heads must be positive.")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads.")
        self.token_dim = int(token_dim)
        self.cls_seed = nn.Parameter(torch.randn(1, self.token_dim) / math.sqrt(self.token_dim))
        self.input_norm = nn.LayerNorm(self.token_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.token_dim, num_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(self.token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.token_dim, 2 * self.token_dim),
            nn.GELU(),
            nn.Linear(2 * self.token_dim, self.token_dim),
        )
        self.output_norm = nn.LayerNorm(self.token_dim)

    def forward(
        self, instances: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """instances: [bags, cells, token_dim] (uniform cell count per call). Returns [bags, token_dim]."""
        if instances.ndim != 3 or instances.shape[-1] != self.token_dim:
            raise ValueError("instances must be [bags, cells, token_dim].")
        bags = instances.shape[0]
        query = self.cls_seed.unsqueeze(0).expand(bags, -1, -1).to(instances.dtype)
        normed = self.input_norm(instances)
        key_padding_mask = None if cell_mask is None else ~cell_mask
        attended, _ = self.cross_attention(
            query,
            normed,
            normed,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )
        token = self.attention_norm(query + attended)
        token = self.output_norm(token + self.ffn(token))
        return token.squeeze(1)


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
        covariance_mode: str = "covariance",
        covariance_shrinkage: float = 0.0,
        include_cls_token: bool = False,
        cls_token_heads: int = 4,
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
        self.covariance_mode = str(covariance_mode)
        self.covariance_shrinkage = float(covariance_shrinkage)
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
        covariance_directions = torch.sin(
            0.019 * feature_index.T * covariance_index
        ) + torch.cos(0.011 * (feature_index.T + 1) * covariance_index)
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
            bag.reshape(-1, bag.shape[-1]).float()
            for bag, keep in zip(bags, context_mask.flatten().tolist())
            if keep
        ]
        if not selected:
            raise ValueError("context_mask must identify at least one context bag.")
        pool = torch.cat(selected, dim=0)
        mean = pool.mean(dim=0)
        # unbiased=False so this matches `_context_pool_stats_batched` exactly.
        # With Bessel's correction the training (batched) and evaluation (list)
        # paths would disagree by a factor sqrt(cells/(cells-1)), i.e. a silent
        # ~0.25% scale mismatch between train and real-data inference.
        std = pool.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean, std

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
        self, centered_delta: torch.Tensor, dimension: int = 32
    ) -> torch.Tensor:
        dimension = min(int(dimension), self.covariance_sketch_dim)
        projected = centered_delta.float() @ self._covariance_projection[:, :dimension].float()
        return torch.einsum("...ni,...nj->...ij", projected, projected) / projected.shape[-2]

    def _slot_covariance_sketch(
        self,
        assignment: torch.Tensor,
        centered_delta: torch.Tensor,
        count: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure local covariance inside aligned population slots.

        `count` (per-bag valid cell counts) fixes the RMS normalization so
        padded cells do not distort short bags.
        """
        measurement_dim = min(16, self.covariance_sketch_dim)
        values = centered_delta.float()
        if count is None:
            bag_rms = (
                values.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
            )
        else:
            # Normalize by valid cells AND the feature dim to match `.mean()`.
            denom = count.reshape(values.shape[:-2] + (1, 1)) * values.shape[-1]
            bag_rms = (
                values.square().sum(dim=(-2, -1), keepdim=True) / denom
            ).sqrt().clamp_min(1e-6)
        projection = self._covariance_projection[:, :measurement_dim].float()
        projected = (values / bag_rms) @ projection
        weights = assignment.float()
        mass = weights.sum(dim=-2).clamp_min(1e-6)
        means = torch.einsum("...ns,...nd->...sd", weights, projected)
        means = means / mass.unsqueeze(-1)
        differences = projected.unsqueeze(-2) - means.unsqueeze(-3)
        covariance = torch.einsum(
            "...ns,...nsi,...nsj->...sij", weights, differences, differences
        ) / mass.unsqueeze(-1).unsqueeze(-1)
        diagonal = covariance.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
        inverse_scale = diagonal.rsqrt()
        correlation = covariance * inverse_scale.unsqueeze(-1) * inverse_scale.unsqueeze(-2)
        row, column = torch.triu_indices(
            measurement_dim, measurement_dim, offset=1, device=values.device
        )
        if self.slot_covariance_descriptor == "correlation":
            descriptor = torch.cat(
                (diagonal.log(), correlation[..., row, column]), dim=-1
            )
        elif self.slot_covariance_descriptor == "spectral":
            eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(1e-6)
            trace = eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            normalized = eigenvalues / trace
            log_shape = eigenvalues.log()
            log_shape = log_shape - log_shape.mean(dim=-1, keepdim=True)
            entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(
                dim=-1, keepdim=True
            )
            top_fraction = normalized[..., -1:]
            anisotropy = eigenvalues[..., -1:] / eigenvalues.mean(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            effective_rank = entropy.exp() / measurement_dim
            descriptor = torch.cat(
                (normalized, log_shape, top_fraction, anisotropy, entropy, effective_rank),
                dim=-1,
            )
        else:
            raise ValueError(
                "slot covariance descriptor must be correlation or spectral."
            )
        reliability = mass / (mass + 5.0)
        return descriptor.to(centered_delta.dtype), reliability.to(centered_delta.dtype)


    def _local_geometry_sketch(
        self,
        instances: torch.Tensor,
        neighbor_counts: Sequence[int] = (4, 8, 16),
    ) -> dict[str, torch.Tensor]:
        """Summarize order-invariant local geometry without component slots."""
        if instances.ndim != 3:
            raise ValueError("instances must be [bags, instances, features].")
        counts = tuple(sorted({int(value) for value in neighbor_counts}))
        if len(counts) < 2 or counts[0] < 2 or instances.shape[1] < 3:
            raise ValueError(
                "neighbor counts require at least two settings and three instances."
            )
        effective_counts = tuple(
            min(value, instances.shape[1] - 1) for value in counts
        )
        projection_dim = min(16, self.covariance_sketch_dim)
        projection = self._covariance_projection[:, :projection_dim].float()
        quantiles = torch.tensor(
            (0.10, 0.25, 0.50, 0.75, 0.90),
            device=instances.device,
            dtype=torch.float32,
        )
        distance_features = []
        anisotropy_features = []
        for bag in instances:
            projected = F.normalize(
                bag.float() @ projection, dim=-1, eps=1e-6
            )
            similarity = projected @ projected.T
            similarity.fill_diagonal_(float("-inf"))
            nearest = similarity.topk(effective_counts[-1], dim=-1).indices
            neighbor_values = projected[nearest]
            center_values = projected.unsqueeze(1)
            squared_distance = (neighbor_values - center_values).square().sum(dim=-1)
            bag_distance = []
            for count in effective_counts:
                local = squared_distance[:, :count]
                local_mean = local.mean(dim=-1)
                local_std = local.std(dim=-1, unbiased=False)
                bag_distance.extend(
                    (torch.quantile(local_mean, quantiles),
                     torch.quantile(local_std, quantiles))
                )
            distance_features.append(torch.cat(bag_distance))

            local_difference = neighbor_values[:, : effective_counts[1]] - center_values
            local_covariance = torch.einsum(
                "nki,nkj->nij", local_difference, local_difference
            ) / effective_counts[1]
            eigenvalues = torch.linalg.eigvalsh(local_covariance).clamp_min(1e-8)
            normalized = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            top_fraction = normalized[:, -1]
            entropy = -(normalized * normalized.clamp_min(1e-8).log()).sum(dim=-1)
            effective_rank = entropy.exp() / projection_dim
            anisotropy_features.append(
                torch.cat(
                    (
                        torch.quantile(top_fraction, quantiles),
                        torch.quantile(effective_rank, quantiles),
                    )
                )
            )
        distance = torch.stack(distance_features).to(instances.dtype)
        anisotropy = torch.stack(anisotropy_features).to(instances.dtype)
        return {
            "distance": distance,
            "anisotropy": anisotropy,
            "combined": torch.cat((distance, anisotropy), dim=-1),
        }


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

    def _context_anchors(
        self,
        bags: list[torch.Tensor],
        context_mask: torch.Tensor,
        cell_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        candidates = torch.cat(
            [
                self._population_candidates(
                    bag,
                    None if cell_masks is None else cell_masks[b_index],
                )
                for b_index, (bag, is_context) in enumerate(
                    zip(bags, context_mask.tolist())
                )
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


    def _context_spherical_kmeans_anchors(
        self,
        bags: list[torch.Tensor],
        context_mask: torch.Tensor,
        num_slots: int,
        refinement_steps: int = 12,
    ) -> torch.Tensor:
        """Cluster context-wide population candidates into component anchors."""
        if num_slots < 1:
            raise ValueError("num_slots must be positive.")
        candidates = torch.cat(
            [
                self._population_candidates(bag)
                for bag, is_context in zip(bags, context_mask.tolist())
                if is_context
            ],
            dim=0,
        )
        if candidates.shape[0] < num_slots:
            raise ValueError("Context does not contain enough anchor candidates.")
        normalized = F.normalize(candidates.float(), dim=-1, eps=1e-6)
        global_center = F.normalize(normalized.mean(dim=0, keepdim=True), dim=-1)
        first = (normalized * global_center).sum(dim=-1).argmax()
        selected = [first]
        nearest_similarity = normalized @ normalized[first]
        for _ in range(1, num_slots):
            index = nearest_similarity.argmin()
            selected.append(index)
            nearest_similarity = torch.maximum(
                nearest_similarity, normalized @ normalized[index]
            )
        anchors = normalized[torch.stack(selected)]
        for _ in range(refinement_steps):
            similarity = normalized @ anchors.T
            assignment = similarity.argmax(dim=-1)
            updated = []
            for slot_index in range(num_slots):
                members = normalized[assignment == slot_index]
                if members.numel() == 0:
                    updated.append(anchors[slot_index])
                else:
                    updated.append(
                        F.normalize(members.mean(dim=0), dim=-1, eps=1e-6)
                    )
            new_anchors = torch.stack(updated)
            if torch.allclose(new_anchors, anchors, atol=1e-5, rtol=1e-5):
                anchors = new_anchors
                break
            anchors = new_anchors
        return anchors.to(candidates.dtype)


    def _raw_stat_tokens(
        self, raw: torch.Tensor, cell_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Per-feature bag statistics computed from the RAW cells (before the
        centered _bag_view transform). Each is a [..., 512] tensor keyed by
        `stat_<name>`.

        * mean: L2-normalized -- the raw bag mean is scale-dependent (Musk
          descriptors span hundreds), so normalize to the unit sphere to match
          the per-cell input distribution at inference. Preserves the mean
          DIRECTION/profile, which is the informative part.
        * variance: raw per-feature variance. NOTE: overlaps with the existing
          `global_spread` summary (sqrt(variance)); provided for flexibility
          but redundant by default.
        * skewness / kurtosis: standardized 3rd/4th central moments, which are
          scale-free by construction, so they are passed through raw and carry
          genuinely new shape information the architecture does not otherwise
          see (the existing summary/spread/covariance/tails cover 1st/2nd
          moments and order statistics only).
        """
        names = set(self.raw_stat_tokens)
        out: dict[str, torch.Tensor] = {}
        if not names:
            return out
        if cell_mask is None:
            mean = raw.mean(dim=-2)
            centered = raw - raw.mean(dim=-2, keepdim=True)
            count = None
        else:
            count = cell_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            masked = raw.masked_fill(~cell_mask.unsqueeze(-1), 0.0)
            mean = masked.sum(dim=-2) / count
            centered = masked - mean.unsqueeze(-2)
        if "mean" in names:
            out["stat_mean"] = F.normalize(mean.float(), dim=-1).to(raw.dtype)
        if "variance" in names:
            denominator = count if count is not None else None
            variance = centered.square()
            if denominator is None:
                variance = variance.mean(dim=-2)
            else:
                variance = variance.sum(dim=-2) / denominator
            out["stat_variance"] = variance.to(raw.dtype)
        if "skewness" in names or "kurtosis" in names:
            std_square = centered.square()
            if count is None:
                std_square = std_square.mean(dim=-2)
            else:
                std_square = std_square.sum(dim=-2) / count
            std = torch.sqrt(std_square + 1e-6)
            if "skewness" in names:
                cube = centered**3
                if count is None:
                    cube = cube.mean(dim=-2)
                else:
                    cube = cube.sum(dim=-2) / count
                out["stat_skewness"] = (cube / (std**3 + 1e-6)).to(raw.dtype)
            if "kurtosis" in names:
                fourth = centered**4
                if count is None:
                    fourth = fourth.mean(dim=-2)
                else:
                    fourth = fourth.sum(dim=-2) / count
                out["stat_kurtosis"] = (fourth / (std**4 + 1e-6)).to(raw.dtype)
        return out

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
        num_bags, num_instances, _ = instances.shape
        if cell_mask is not None:
            if cell_mask.shape != (num_bags, num_instances):
                raise ValueError("cell_mask must be [bags, instances].")
            valid = cell_mask
            valid_count = cell_mask.sum(dim=-1).clamp_min(1)
            max_valid = int(valid_count.max().item())
        else:
            valid = None
            valid_count = None
            max_valid = num_instances
        if cell_mask is not None and self.ccts_lambdas:
            raise NotImplementedError(
                "ccts_lambdas is not supported on padded (ragged-batched) "
                "episodes; v30 configs do not enable it."
            )
        normalized = F.normalize(instances.float(), dim=-1)
        if anchors.ndim == 2:
            expanded_anchors = anchors.unsqueeze(0).expand(num_bags, -1, -1)
        elif anchors.ndim == 3 and anchors.shape[0] == num_bags:
            expanded_anchors = anchors
        else:
            raise ValueError("Anchors must be [slots, dim] or [bags, slots, dim].")
        similarity = torch.einsum("bnd,bsd->bns", normalized, expanded_anchors.float())
        assignment = torch.softmax(
            similarity / self.assignment_temperature,
            dim=-1,
        ).to(instances.dtype)
        if valid is not None:
            # Padded cells must not claim any slot mass. Masking similarity
            # with -inf is unsafe here: softmax over all -inf slots is NaN.
            assignment = assignment * valid.unsqueeze(-1)
        mass = assignment.sum(dim=1).clamp_min(1e-6)
        if valid_count is None:
            proportion = mass / num_instances
        else:
            proportion = mass / valid_count.unsqueeze(-1)
        slot_mean = torch.einsum(
            "bns,bnd->bsd", assignment, instances
        ) / mass.unsqueeze(-1)
        difference = instances[:, :, None, :] - slot_mean[:, None, :, :]
        slot_std = torch.sqrt(
            (
                assignment.float().transpose(1, 2).unsqueeze(-1)
                * difference.float().square().transpose(1, 2)
            ).sum(dim=2)
            / mass.float().unsqueeze(-1)
            + 1e-6
        ).to(instances.dtype)
        dispersion = (assignment * (1.0 - similarity).to(assignment.dtype)).sum(
            dim=1
        ) / mass
        metadata = torch.stack((proportion.log(), dispersion), dim=-1)
        if centered_delta is None:
            if valid is None:
                centered_delta = instances - instances.mean(dim=-2, keepdim=True)
            else:
                masked = instances.masked_fill(~valid.unsqueeze(-1), 0.0)
                centered_delta = masked - (
                    masked.sum(dim=-2, keepdim=True)
                    / valid_count.unsqueeze(-1).unsqueeze(-1)
                )
        slot_covariance_sketch, slot_covariance_reliability = (
            self._slot_covariance_sketch(
                assignment, centered_delta, count=valid_count
            )
        )

        slot_distance = difference.float().square().mean(dim=-1)
        rare_score = assignment.float() * slot_distance
        if valid is not None:
            rare_score = rare_score.masked_fill(
                ~valid.unsqueeze(-1), float("-inf")
            )
        if valid_count is None:
            rare_count = min(
                num_instances,
                max(1, int(math.ceil(self.slot_rare_fraction * num_instances))),
            )
            values, index = rare_score.transpose(1, 2).topk(rare_count, dim=-1)
            weights = torch.softmax(values, dim=-1).to(instances.dtype)
        else:
            # Per-bag rare count (matches the per-bag list path), realized with
            # a uniform topk over the max count plus a keep mask.
            rare_count_bag = torch.minimum(
                torch.clamp(
                    torch.ceil(self.slot_rare_fraction * valid_count).long(),
                    min=1,
                ),
                valid_count.long(),
            )
            k_rare = min(num_instances, int(rare_count_bag.max().item()))
            values, index = rare_score.transpose(1, 2).topk(k_rare, dim=-1)
            keep = torch.arange(
                k_rare, device=instances.device
            ) < rare_count_bag.unsqueeze(-1).unsqueeze(-1)
            # Softmax only over kept cells so extra/padded cells do not shift
            # the kept weights (matches the per-bag list path).
            masked_values = values.masked_fill(~keep, float("-inf"))
            weights = torch.softmax(masked_values, dim=-1).to(instances.dtype)
        batch_index = torch.arange(num_bags, device=instances.device)[:, None, None]
        selected = instances[batch_index, index]
        rare_state = (weights.unsqueeze(-1) * selected).sum(dim=2)

        center_features = torch.cat(
            (expanded_anchors, slot_mean - expanded_anchors, metadata), dim=-1
        )
        spread_features = torch.cat((expanded_anchors, slot_std, metadata), dim=-1)
        rare_features = torch.cat(
            (expanded_anchors, rare_state - expanded_anchors, metadata), dim=-1
        )
        residual_scale = torch.sigmoid(self.slot_residual_logit)
        center_token = slot_mean + residual_scale * self.center_slot_encoder(
            center_features
        )
        spread_token = slot_std + residual_scale * self.spread_slot_encoder(
            spread_features
        )
        rare_token = rare_state + residual_scale * self.rare_slot_encoder(rare_features)
        slot_tokens = torch.stack((center_token, spread_token, rare_token), dim=2)

        nearest_similarity, nearest_slot = similarity.max(dim=-1)
        novelty = 1.0 - nearest_similarity
        if valid is not None:
            novelty = novelty.masked_fill(~valid, float("-inf"))
        tail_tokens: list[torch.Tensor] = []
        selected_counts: list[torch.Tensor] = []
        for fraction in self.tail_fractions:
            if valid_count is None:
                count = min(
                    num_instances,
                    max(
                        self.min_tail_instances,
                        int(math.ceil(fraction * num_instances)),
                    ),
                )
                index = novelty.topk(count, dim=1).indices
                selected_instances = instances.gather(
                    1, index.unsqueeze(-1).expand(-1, -1, instances.shape[-1])
                )
                selected_slots = nearest_slot.gather(1, index)
                selected_anchors = expanded_anchors.gather(
                    1, selected_slots.unsqueeze(-1).expand(-1, -1, anchors.shape[-1])
                )
                deviation = selected_instances - selected_anchors
                with torch.autocast(device_type=instances.device.type, enabled=False):
                    encoded_tail = self.shared_tail_encoder(deviation.float())
                    # Arithmetic mean over the top-fraction cells; the per-bag
                    # list path uses the exact same `.mean(dim=0)`.
                    tail_tokens.append(encoded_tail.mean(dim=1))
                selected_counts.append(torch.tensor(count, device=anchors.device))
            else:
                # Per-bag tail count (matches the per-bag list path), realized
                # with a uniform topk over the max count plus a keep mask.
                count_bag = torch.minimum(
                    torch.clamp(
                        torch.ceil(fraction * valid_count).long(),
                        min=self.min_tail_instances,
                    ),
                    valid_count.long(),
                )
                k = min(num_instances, int(count_bag.max().item()))
                index = novelty.topk(k, dim=1).indices
                selected_instances = instances.gather(
                    1, index.unsqueeze(-1).expand(-1, -1, instances.shape[-1])
                )
                selected_slots = nearest_slot.gather(1, index)
                selected_anchors = expanded_anchors.gather(
                    1, selected_slots.unsqueeze(-1).expand(-1, -1, anchors.shape[-1])
                )
                deviation = selected_instances - selected_anchors
                with torch.autocast(device_type=instances.device.type, enabled=False):
                    encoded_tail = self.shared_tail_encoder(deviation.float())
                    # Arithmetic mean over the per-bag kept cells (matches the
                    # per-bag list path's `.mean(dim=0)`); extra/padded cells
                    # are zeroed so they do not contribute.
                    keep = (
                        torch.arange(k, device=instances.device)
                        < count_bag.unsqueeze(-1)
                    )
                    kept = encoded_tail.masked_fill(
                        ~keep.unsqueeze(-1), 0.0
                    ).sum(dim=1)
                    tail_tokens.append(kept / count_bag.unsqueeze(-1))
                selected_counts.append(count_bag.to(anchors.device))

        for abs_k in self.absolute_tail_ks:
            if valid_count is None:
                count = min(num_instances, max(1, abs_k))
                index = novelty.topk(count, dim=1).indices
                selected_instances = instances.gather(
                    1, index.unsqueeze(-1).expand(-1, -1, instances.shape[-1])
                )
                selected_slots = nearest_slot.gather(1, index)
                selected_anchors = expanded_anchors.gather(
                    1, selected_slots.unsqueeze(-1).expand(-1, -1, anchors.shape[-1])
                )
                deviation = selected_instances - selected_anchors
                with torch.autocast(device_type=instances.device.type, enabled=False):
                    encoded_tail = self.shared_tail_encoder(deviation.float())
                    lse_weights = torch.softmax(encoded_tail * 2.0, dim=1)
                    tail_tokens.append((lse_weights * encoded_tail).sum(dim=1))
                selected_counts.append(torch.tensor(count, device=anchors.device))
            else:
                count_bag = torch.minimum(
                    torch.full_like(valid_count.long(), abs_k).clamp_min(1),
                    valid_count.long(),
                )
                k = min(num_instances, int(count_bag.max().item()))
                index = novelty.topk(k, dim=1).indices
                selected_instances = instances.gather(
                    1, index.unsqueeze(-1).expand(-1, -1, instances.shape[-1])
                )
                selected_slots = nearest_slot.gather(1, index)
                selected_anchors = expanded_anchors.gather(
                    1, selected_slots.unsqueeze(-1).expand(-1, -1, anchors.shape[-1])
                )
                deviation = selected_instances - selected_anchors
                with torch.autocast(device_type=instances.device.type, enabled=False):
                    encoded_tail = self.shared_tail_encoder(deviation.float())
                    # Softmax only over kept cells so extra/padded cells do not
                    # shift the kept weights (matches the per-bag list path).
                    keep = (
                        torch.arange(k, device=instances.device)
                        < count_bag.unsqueeze(-1)
                    )
                    tail_logits = encoded_tail.masked_fill(
                        ~keep.unsqueeze(-1), float("-inf")
                    )
                    lse_weights = torch.softmax(tail_logits * 2.0, dim=1)
                    tail_tokens.append((lse_weights * encoded_tail).sum(dim=1))
                selected_counts.append(count_bag.to(anchors.device))

        if self.ccts_lambdas:
            batch_index = torch.arange(num_bags, device=instances.device)[:, None]
            instance_anchors = expanded_anchors[batch_index, nearest_slot]
            all_deviations = instances - instance_anchors
            with torch.autocast(device_type=instances.device.type, enabled=False):
                all_encoded = self.shared_tail_encoder(all_deviations.float())
                raw_score = novelty + torch.sigmoid(self.ccts_score_head(all_encoded).squeeze(-1))
                sorted_scores = raw_score.detach().flatten().sort().values
                ranks = torch.searchsorted(sorted_scores, raw_score.flatten()).reshape(raw_score.shape)
                soft_p = (1.0 + (sorted_scores.numel() - ranks).float()) / (1.0 + float(sorted_scores.numel()))
                soft_p = soft_p.clamp(1e-6, 1.0)
                e_b = -torch.log(soft_p) - math.log(max(1, num_instances))


                for lamb in self.ccts_lambdas:
                    gate = torch.sigmoid((math.log(lamb) - torch.log(float(num_instances) * soft_p + 1e-6)) / self.ccts_tau)
                    gate_weight = gate.unsqueeze(-1)
                    pooled_tail = (gate_weight * all_encoded).sum(dim=1) / (gate_weight.sum(dim=1).clamp_min(1e-6))
                    meta = torch.stack(
                        (
                            torch.log1p(gate.sum(dim=1)),
                            e_b.max(dim=1).values,
                            gate.sum(dim=1) / float(num_instances),
                            torch.full((num_bags,), math.log(float(num_instances)), device=instances.device),
                            torch.full((num_bags,), math.log1p(float(sorted_scores.numel())), device=instances.device),

                        ),
                        dim=-1,
                    )
                    ccts_token = pooled_tail + self.ccts_metadata_projection(meta)
                    tail_tokens.append(ccts_token.to(instances.dtype))





        if global_summary is None:
            _, global_summary, _ = self._bag_view(instances, cell_mask=valid)
        if covariance_sketch is None:
            covariance_sketch = self._covariance_sketch(
                centered_delta, count=valid_count
            )
        representation = {
            "global_summary": global_summary,
            "slots": slot_tokens,
            "tails": torch.stack(tail_tokens, dim=1),
            "slot_metadata": metadata,
            "covariance_sketch": covariance_sketch,
            "slot_covariance_sketch": slot_covariance_sketch,
            "slot_covariance_reliability": slot_covariance_reliability,
            "covariance_matrix": (
                self._projected_covariance_matrix(centered_delta)
                if self.emit_covariance_matrix
                else centered_delta.new_zeros((num_bags, 1, 1))
            ),
            "cls_token": (
                self.cls_token_pooling(instances, cell_mask=valid)
                if self.include_cls_token
                else global_summary.new_zeros((num_bags, self.input_dim))
            ),
        }
        if self.raw_stat_tokens:
            if raw_stats is None:
                raise ValueError(
                    "raw_stat_tokens requires precomputed raw_stats "
                    "(callers must pass _raw_stat_tokens of the raw cells)."
                )
            for stat_name in self.raw_stat_tokens:
                representation[f"stat_{stat_name}"] = raw_stats[f"stat_{stat_name}"]
        if not return_auxiliary:
            return representation
        return representation, {
            "population_anchors": anchors,
            "num_density_slots": torch.tensor(
                self.num_density_slots, device=anchors.device
            ),
            "population_proportions": proportion,
            "population_dispersions": dispersion,
            "population_slot_means": slot_mean,
            "instance_counts": (
                valid_count
                if valid_count is not None
                else torch.full((num_bags,), num_instances, device=anchors.device)
            ),
            "tail_counts": torch.stack(selected_counts, dim=1),
            "slot_residual_scale": residual_scale,
        }

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
        prepared = [self._bag_view(bag, pool_mean, pool_std) for bag in raw_bags]
        bags = [item[0] for item in prepared]
        global_summaries = [item[1] for item in prepared]
        centered_deltas = [item[2] for item in prepared]
        covariance_sketches = [self._covariance_sketch(delta) for delta in centered_deltas]
        anchors = self._context_anchors(bags, context_mask)
        if self.raw_stat_tokens:
            if isinstance(instances, torch.Tensor):
                # Dense path: all bags share one shape, stack is safe.
                raw_stats = self._raw_stat_tokens(torch.stack(raw_bags))
            else:
                # List path: bags can have different instance counts (e.g. real
                # Musk). _raw_stat_tokens on a 2D bag returns [512]; stack over
                # bags -> [num_bags, 512], matching the per-bag token shape.
                per_bag_stats = [
                    self._raw_stat_tokens(bag) for bag in raw_bags
                ]
                raw_stats = {
                    name: torch.stack([stats[name] for stats in per_bag_stats])
                    for name in per_bag_stats[0]
                }
        else:
            raw_stats = None
        if isinstance(instances, torch.Tensor):
            result = self._forward_dense(
                torch.stack(bags),
                anchors,
                return_auxiliary,
                global_summary=torch.stack(global_summaries),
                covariance_sketch=torch.stack(covariance_sketches),
                centered_delta=torch.stack(centered_deltas),
                raw_stats=raw_stats,
            )
            if return_auxiliary:
                representation, auxiliary = result
                return representation, auxiliary
            return result

        ccts_per_bag_tokens: list[list[torch.Tensor]] = [[] for _ in range(len(bags))]
        if self.ccts_lambdas:
            all_raw_scores: list[torch.Tensor] = []
            all_encoded_list: list[torch.Tensor] = []
            for bag in bags:
                normalized = F.normalize(bag.float(), dim=-1)
                similarity = normalized @ anchors.float().T
                nearest_similarity, nearest_slot = similarity.max(dim=-1)
                novelty = 1.0 - nearest_similarity
                instance_anchors = anchors[nearest_slot]
                all_deviations = bag - instance_anchors
                with torch.autocast(device_type=bag.device.type, enabled=False):
                    all_encoded = self.shared_tail_encoder(all_deviations.float())
                    raw_score = novelty + torch.sigmoid(self.ccts_score_head(all_encoded).squeeze(-1))
                all_raw_scores.append(raw_score)
                all_encoded_list.append(all_encoded)

            context_mask_bool = context_mask.to(torch.bool)
            context_scores = [
                score for score, is_ctx in zip(all_raw_scores, context_mask_bool) if is_ctx
            ]
            if context_scores:
                all_context_scores = torch.cat(context_scores, dim=0)
            else:
                all_context_scores = torch.cat(all_raw_scores, dim=0)

            sorted_scores = all_context_scores.detach().sort().values

            for b_idx, (bag, raw_score, all_encoded) in enumerate(zip(bags, all_raw_scores, all_encoded_list)):
                n_inst = bag.shape[0]
                ranks = torch.searchsorted(sorted_scores, raw_score)
                soft_p = (1.0 + (sorted_scores.numel() - ranks).float()) / (1.0 + float(sorted_scores.numel()))
                soft_p = soft_p.clamp(1e-6, 1.0)
                e_b = -torch.log(soft_p) - math.log(max(1, n_inst))

                for lamb in self.ccts_lambdas:
                    gate = torch.sigmoid((math.log(lamb) - torch.log(float(n_inst) * soft_p + 1e-6)) / self.ccts_tau)
                    gate_weight = gate.unsqueeze(-1)
                    pooled_tail = (gate_weight * all_encoded).sum(dim=0) / (gate_weight.sum(dim=0).clamp_min(1e-6))
                    meta = torch.stack(
                        (
                            torch.log1p(gate.sum()),
                            e_b.max(),
                            gate.sum() / float(n_inst),
                            torch.tensor(math.log(float(n_inst)), device=bag.device),
                            torch.tensor(math.log1p(float(sorted_scores.numel())), device=bag.device),
                        ),
                        dim=-1,
                    )
                    ccts_token = pooled_tail + self.ccts_metadata_projection(meta)
                    ccts_per_bag_tokens[b_idx].append(ccts_token.to(bag.dtype))

        slot_tokens: list[torch.Tensor] = []
        tail_tokens: list[torch.Tensor] = []
        slot_covariance_sketches: list[torch.Tensor] = []
        slot_covariance_reliabilities: list[torch.Tensor] = []
        proportions: list[torch.Tensor] = []
        dispersions: list[torch.Tensor] = []
        slot_means: list[torch.Tensor] = []
        selected_counts: list[list[int]] = []
        cls_tokens: list[torch.Tensor] = []
        for b_idx, (bag, centered_delta) in enumerate(zip(bags, centered_deltas)):
            cls_tokens.append(
                self.cls_token_pooling(bag.unsqueeze(0)).squeeze(0)
                if self.include_cls_token
                else bag.new_zeros(self.input_dim)
            )
            normalized = F.normalize(bag.float(), dim=-1)
            similarity = normalized @ anchors.float().T
            assignment = torch.softmax(
                similarity / self.assignment_temperature,
                dim=-1,
            ).to(bag.dtype)
            mass = assignment.sum(dim=0).clamp_min(1e-6)
            proportion = mass / bag.shape[0]
            slot_mean = (assignment.T @ bag) / mass.unsqueeze(-1)
            difference = bag[:, None, :] - slot_mean[None, :, :]
            slot_std = torch.sqrt(
                (
                    assignment.float().T.unsqueeze(-1)
                    * difference.float().square().transpose(0, 1)
                ).sum(dim=1)
                / mass.float().unsqueeze(-1)
                + 1e-6
            ).to(bag.dtype)
            dispersion = (assignment * (1.0 - similarity).to(assignment.dtype)).sum(
                dim=0
            ) / mass
            metadata = torch.stack((proportion.log(), dispersion), dim=-1)
            slot_covariance_sketch, slot_covariance_reliability = (
                self._slot_covariance_sketch(
                    assignment.unsqueeze(0), centered_delta.unsqueeze(0)
                )
            )
            slot_covariance_sketches.append(slot_covariance_sketch.squeeze(0))
            slot_covariance_reliabilities.append(
                slot_covariance_reliability.squeeze(0)
            )

            rare_states: list[torch.Tensor] = []
            rare_count = min(
                bag.shape[0],
                max(1, int(math.ceil(self.slot_rare_fraction * bag.shape[0]))),
            )
            slot_distance = difference.float().square().mean(dim=-1)
            for slot_index in range(self.num_slots):
                rare_score = (
                    assignment[:, slot_index].float() * slot_distance[:, slot_index]
                )
                values, index = rare_score.topk(rare_count)
                weights = torch.softmax(values, dim=0).to(bag.dtype)
                rare_states.append((weights.unsqueeze(-1) * bag[index]).sum(dim=0))
            rare_state = torch.stack(rare_states)

            center_features = torch.cat(
                (anchors, slot_mean - anchors, metadata), dim=-1
            )
            spread_features = torch.cat((anchors, slot_std, metadata), dim=-1)
            rare_features = torch.cat((anchors, rare_state - anchors, metadata), dim=-1)
            residual_scale = torch.sigmoid(self.slot_residual_logit)
            center_token = slot_mean + residual_scale * self.center_slot_encoder(
                center_features
            )
            spread_token = slot_std + residual_scale * self.spread_slot_encoder(
                spread_features
            )
            rare_token = rare_state + residual_scale * self.rare_slot_encoder(
                rare_features
            )
            slot_tokens.append(
                torch.stack((center_token, spread_token, rare_token), dim=1)
            )

            nearest_similarity, nearest_slot = similarity.max(dim=-1)
            novelty = 1.0 - nearest_similarity
            bag_tail_tokens: list[torch.Tensor] = []
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
                # Tail class prototypes can be almost identical, which makes
                # their centered cosine path sensitive to FP16 loss scaling.
                # Keep the complete tail encoder/gradient path in FP32.
                with torch.autocast(
                    device_type=bag.device.type,
                    enabled=False,
                ):
                    bag_tail_tokens.append(
                        self.shared_tail_encoder(deviation.float()).mean(dim=0)
                    )
                bag_selected_counts.append(count)

            for abs_k in self.absolute_tail_ks:
                count = min(bag.shape[0], max(1, abs_k))
                index = novelty.topk(count).indices
                deviation = bag[index] - anchors[nearest_slot[index]]
                with torch.autocast(device_type=bag.device.type, enabled=False):
                    encoded_tail = self.shared_tail_encoder(deviation.float())
                    if encoded_tail.ndim == 1:
                        encoded_tail = encoded_tail.unsqueeze(0)
                    lse_weights = torch.softmax(encoded_tail * 2.0, dim=0)
                    bag_tail_tokens.append((lse_weights * encoded_tail).sum(dim=0))
                bag_selected_counts.append(count)

            if self.ccts_lambdas:
                bag_tail_tokens.extend(ccts_per_bag_tokens[b_idx])

            tail_tokens.append(torch.stack(bag_tail_tokens))
            proportions.append(proportion)
            dispersions.append(dispersion)
            slot_means.append(slot_mean)
            selected_counts.append(bag_selected_counts)


        representation = {
            "global_summary": torch.stack(global_summaries),
            "slots": torch.stack(slot_tokens),
            "tails": torch.stack(tail_tokens),
            "slot_metadata": torch.stack(
                [
                    torch.stack((proportion.log(), dispersion), dim=-1)
                    for proportion, dispersion in zip(proportions, dispersions)
                ]
            ),
            "covariance_sketch": torch.stack(covariance_sketches),
            "slot_covariance_sketch": torch.stack(slot_covariance_sketches),
            "slot_covariance_reliability": torch.stack(
                slot_covariance_reliabilities
            ),
            "covariance_matrix": (
                torch.stack([
                    self._projected_covariance_matrix(delta)
                    for delta in centered_deltas
                ])
                if self.emit_covariance_matrix
                else centered_deltas[0].new_zeros((len(centered_deltas), 1, 1))
            ),
            "cls_token": torch.stack(cls_tokens),
        }
        if self.raw_stat_tokens:
            for stat_name in self.raw_stat_tokens:
                representation[f"stat_{stat_name}"] = raw_stats[
                    f"stat_{stat_name}"
                ].to(centered_deltas[0].dtype)
        if not return_auxiliary:
            return representation
        return representation, {
            "population_anchors": anchors,
            "num_density_slots": torch.tensor(
                self.num_density_slots, device=anchors.device
            ),
            "population_proportions": torch.stack(proportions),
            "population_dispersions": torch.stack(dispersions),
            "population_slot_means": torch.stack(slot_means),
            "instance_counts": torch.tensor(
                [len(bag) for bag in bags], device=anchors.device
            ),
            "tail_counts": torch.tensor(selected_counts, device=anchors.device),
            "slot_residual_scale": torch.sigmoid(self.slot_residual_logit),
            "centered_delta_mean": torch.stack(
                [delta.float().mean(dim=0) for delta in centered_deltas]
            ),
            "global_summary": torch.stack(global_summaries),
        }


class SetCrossAttentionMetaClassifier(nn.Module):
    """Classify queries by directly attending to every labelled context token.

    Context tokens are split into class sets only for routing.  The set encoder,
    cross-attention, and relation scorer are shared across all classes.  Thus a
    label permutation only reorders which context set is scored in each output
    column and cannot change the underlying prediction rule.
    """

    def __init__(
        self,
        token_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_set_layers: int = 2,
        relation_hidden_dim: int = 512,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if min(token_dim, hidden_dim, relation_hidden_dim) <= 0:
            raise ValueError("All feature dimensions must be positive.")
        if num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if num_set_layers < 0:
            raise ValueError("num_set_layers cannot be negative.")
        if num_classes < 2:
            raise ValueError("num_classes must be at least two.")

        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.input_norm = nn.LayerNorm(token_dim)
        self.input_projection = nn.Linear(token_dim, hidden_dim)

        if num_set_layers:
            set_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.set_encoder: nn.Module = nn.TransformerEncoder(
                set_layer,
                num_layers=num_set_layers,
                enable_nested_tensor=False,
            )
        else:
            self.set_encoder = nn.Identity()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_dim)
        self.relation_scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 1),
        )

    def _validate_inputs(
        self,
        context_tokens: torch.Tensor,
        context_labels: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if context_tokens.ndim != 2 or context_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                f"context_tokens must have shape [context, {self.token_dim}]."
            )
        if query_tokens.ndim != 2 or query_tokens.shape[-1] != self.token_dim:
            raise ValueError(f"query_tokens must have shape [query, {self.token_dim}].")
        if (
            context_labels.ndim != 1
            or context_labels.shape[0] != context_tokens.shape[0]
        ):
            raise ValueError("context_labels must have shape [context].")
        if context_tokens.shape[0] == 0 or query_tokens.shape[0] == 0:
            raise ValueError("Context and query sets must both be non-empty.")
        if torch.any((context_labels < 0) | (context_labels >= self.num_classes)):
            raise ValueError(f"Context labels must be in [0, {self.num_classes - 1}].")
        counts = torch.bincount(context_labels.long(), minlength=self.num_classes)
        if torch.any(counts == 0):
            missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(
                "Every class must occur in the context set; "
                f"missing classes: {missing}."
            )
        return counts

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_labels: torch.Tensor,
        query_tokens: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        class_counts = self._validate_inputs(
            context_tokens, context_labels, query_tokens
        )
        encoded_context = self.input_projection(self.input_norm(context_tokens))
        encoded_query = self.input_projection(self.input_norm(query_tokens))

        class_logits: list[torch.Tensor] = []
        class_attention_entropy: list[torch.Tensor] = []
        for class_index in range(self.num_classes):
            class_context = encoded_context[context_labels == class_index].unsqueeze(0)
            class_context = self.set_encoder(class_context)
            attended, weights = self.cross_attention(
                encoded_query.unsqueeze(0),
                class_context,
                class_context,
                need_weights=True,
                average_attn_weights=True,
            )
            attended = attended.squeeze(0)
            # Preserve a stable class-set mean path while cross-attention learns
            # query-specific deviations from that global summary.
            class_mean = class_context.mean(dim=1).expand_as(attended)
            class_summary = self.cross_attention_norm(attended + class_mean)
            relation = torch.cat(
                (
                    encoded_query,
                    class_summary,
                    encoded_query - class_summary,
                    encoded_query * class_summary,
                ),
                dim=-1,
            )
            class_logits.append(self.relation_scorer(relation).squeeze(-1))
            probability = weights.squeeze(0).float().clamp_min(1e-12)
            class_attention_entropy.append(
                -(probability * probability.log()).sum(dim=-1)
            )

        logits = torch.stack(class_logits, dim=-1)
        if not return_auxiliary:
            return logits
        return logits, {
            "context_class_counts": class_counts,
            "cross_attention_entropy": torch.stack(class_attention_entropy, dim=-1),
        }

    def forward_batched(
        self,
        context_tokens: torch.Tensor,
        context_labels: torch.Tensor,
        query_tokens: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Batched equivalent of forward for equal-size episode sets."""
        if context_tokens.ndim != 3 or query_tokens.ndim != 3:
            raise ValueError("Batched context/query tokens must have three dimensions.")
        encoded_context = self.input_projection(self.input_norm(context_tokens))
        encoded_query = self.input_projection(self.input_norm(query_tokens))
        counts = F.one_hot(context_labels.long(), num_classes=self.num_classes).sum(
            dim=1
        )
        class_logits = []
        class_entropies = []
        for class_index in range(self.num_classes):
            valid = context_labels == class_index
            class_context = self.set_encoder(
                encoded_context,
                src_key_padding_mask=~valid,
            )
            attended, weights = self.cross_attention(
                encoded_query,
                class_context,
                class_context,
                key_padding_mask=~valid,
                need_weights=True,
                average_attn_weights=True,
            )
            denominator = valid.sum(dim=1, keepdim=True).clamp_min(1)
            class_mean = (class_context * valid.unsqueeze(-1)).sum(dim=1) / denominator
            class_summary = self.cross_attention_norm(
                attended + class_mean.unsqueeze(1)
            )
            relation = torch.cat(
                (
                    encoded_query,
                    class_summary,
                    encoded_query - class_summary,
                    encoded_query * class_summary,
                ),
                dim=-1,
            )
            class_logits.append(self.relation_scorer(relation).squeeze(-1))
            probability = weights.float().clamp_min(1e-12)
            class_entropies.append(-(probability * probability.log()).sum(dim=-1))
        logits = torch.stack(class_logits, dim=-1)
        if not return_auxiliary:
            return logits
        return logits, {
            "context_class_counts": counts,
            "cross_attention_entropy": torch.stack(class_entropies, dim=-1),
        }


class RidgeResidualMetaClassifier(SetCrossAttentionMetaClassifier):
    """Class-balanced ridge prediction with a bounded attention residual.

    Ridge supplies an explicit, label-equivariant episode-level decision rule.
    Set/cross-attention is retained, but it only learns a gated correction to
    that stable base instead of having to invent the full classification rule.
    """

    def __init__(
        self,
        token_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_set_layers: int = 1,
        relation_hidden_dim: int = 256,
        ridge_dim: int = 64,
        ridge_lambda: float = 1.0,
        ridge_logit_scale: float = 5.0,
        attention_residual_scale: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_set_layers=num_set_layers,
            relation_hidden_dim=relation_hidden_dim,
            num_classes=num_classes,
        )
        if ridge_dim < 1 or ridge_lambda <= 0 or ridge_logit_scale <= 0:
            raise ValueError("Ridge parameters must be positive.")
        if not 0 < attention_residual_scale < 1:
            raise ValueError("attention_residual_scale must be in (0, 1).")
        self.ridge_dim = int(ridge_dim)
        self.ridge_projection = nn.Sequential(
            nn.Linear(token_dim, ridge_dim),
            nn.GELU(),
            nn.Linear(ridge_dim, ridge_dim),
        )
        self.ridge_log_lambda = nn.Parameter(torch.tensor(math.log(ridge_lambda)))
        self.ridge_log_scale = nn.Parameter(torch.tensor(math.log(ridge_logit_scale)))
        residual_logit = math.log(
            attention_residual_scale / (1.0 - attention_residual_scale)
        )
        self.attention_residual_logit = nn.Parameter(torch.tensor(residual_logit))

    @staticmethod
    def _solve_ridge_system(
        gram: torch.Tensor,
        rhs: torch.Tensor,
        ridge_lambda: torch.Tensor,
    ) -> torch.Tensor:
        """Solve a positive-definite ridge system with adaptive FP32 jitter."""
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

    def _ridge_logits(
        self,
        context_tokens: torch.Tensor,
        context_labels: torch.Tensor,
        query_tokens: torch.Tensor,
        class_counts: torch.Tensor,
    ) -> torch.Tensor:
        # Center and globally scale each episode before the learned projection.
        # A scalar scale preserves the geometry between feature dimensions.
        output_dtype = query_tokens.dtype
        context_tokens = context_tokens.float()
        query_tokens = query_tokens.float()
        center = context_tokens.mean(dim=0, keepdim=True)
        context = context_tokens - center
        query = query_tokens - center
        rms = context.square().mean().sqrt().clamp_min(1e-6)
        with torch.autocast(device_type=context_tokens.device.type, enabled=False):
            context = self.ridge_projection(context / rms)
            query = self.ridge_projection(query / rms)

        # The solve is kept in fp32 under AMP. Class weights give both context
        # classes equal total mass even when donor counts are imbalanced.
        context32 = context.float()
        query32 = query.float()
        targets = F.one_hot(context_labels.long(), num_classes=self.num_classes).float()
        sample_weight = class_counts.float().reciprocal()[context_labels.long()]
        ridge_lambda = self.ridge_log_lambda.exp().clamp(1e-4, 1e4)
        with torch.autocast(device_type=context_tokens.device.type, enabled=False):
            # Eliminate the unregularized intercept by weighted centering. This
            # is algebraically equivalent to solving the augmented system, but
            # leaves a strictly positive-definite feature block. Forming one
            # joint system with an unregularized bias made rare CUDA episodes
            # singular and could produce non-finite gradients in solve backward.
            total_weight = sample_weight.sum().clamp_min(1e-12)
            feature_mean = (sample_weight.unsqueeze(-1) * context32).sum(
                dim=0, keepdim=True
            ) / total_weight
            target_mean = (sample_weight.unsqueeze(-1) * targets).sum(
                dim=0, keepdim=True
            ) / total_weight
            centered_context = context32 - feature_mean
            centered_targets = targets - target_mean
            root_weight = sample_weight.sqrt().unsqueeze(-1)
            weighted_design = centered_context * root_weight
            weighted_targets = centered_targets * root_weight
            gram = weighted_design.T @ weighted_design
            rhs = weighted_design.T @ weighted_targets
            coefficients = self._solve_ridge_system(gram, rhs, ridge_lambda)
            intercept = target_mean - feature_mean @ coefficients
            logits = query32 @ coefficients + intercept
            if not torch.isfinite(logits).all():
                raise RuntimeError("The ridge logits contain NaN or Inf values.")
        return logits.to(output_dtype)

    def _ridge_logits_batched(
        self,
        context_tokens: torch.Tensor,
        context_labels: torch.Tensor,
        query_tokens: torch.Tensor,
        class_counts: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = query_tokens.dtype
        context_tokens = context_tokens.float()
        query_tokens = query_tokens.float()
        center = context_tokens.mean(dim=1, keepdim=True)
        context = context_tokens - center
        query = query_tokens - center
        rms = context.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        with torch.autocast(device_type=context_tokens.device.type, enabled=False):
            context = self.ridge_projection(context / rms)
            query = self.ridge_projection(query / rms)
        context32 = context.float()
        query32 = query.float()
        targets = F.one_hot(context_labels.long(), num_classes=self.num_classes).float()
        sample_weight = (
            class_counts.float().reciprocal().gather(1, context_labels.long())
        )
        ridge_lambda = self.ridge_log_lambda.exp().clamp(1e-4, 1e4)
        with torch.autocast(device_type=context_tokens.device.type, enabled=False):
            total_weight = sample_weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
            feature_mean = (sample_weight.unsqueeze(-1) * context32).sum(
                dim=1, keepdim=True
            ) / total_weight.unsqueeze(-1)
            target_mean = (sample_weight.unsqueeze(-1) * targets).sum(
                dim=1, keepdim=True
            ) / total_weight.unsqueeze(-1)
            centered_context = context32 - feature_mean
            centered_targets = targets - target_mean
            root_weight = sample_weight.sqrt().unsqueeze(-1)
            weighted_design = centered_context * root_weight
            weighted_targets = centered_targets * root_weight
            gram = weighted_design.transpose(1, 2) @ weighted_design
            rhs = weighted_design.transpose(1, 2) @ weighted_targets
            coefficients = self._solve_ridge_system(gram, rhs, ridge_lambda)
            intercept = target_mean - feature_mean @ coefficients
            logits = query32 @ coefficients + intercept
            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    "The batched ridge logits contain NaN or Inf values."
                )
        return logits.to(output_dtype)

    def forward_batched(
        self,
        context_tokens: torch.Tensor,
        context_labels: torch.Tensor,
        query_tokens: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        class_counts = F.one_hot(
            context_labels.long(), num_classes=self.num_classes
        ).sum(dim=1)
        ridge_logits = self._ridge_logits_batched(
            context_tokens, context_labels, query_tokens, class_counts
        )
        attention_logits, attention_auxiliary = super().forward_batched(
            context_tokens,
            context_labels,
            query_tokens,
            return_auxiliary=True,
        )
        ridge_scale = self.ridge_log_scale.exp().clamp(0.1, 100.0)
        residual_scale = torch.sigmoid(self.attention_residual_logit)
        logits = ridge_scale * ridge_logits + residual_scale * attention_logits
        if not return_auxiliary:
            return logits
        episodes = context_tokens.shape[0]
        return logits, {
            **attention_auxiliary,
            "ridge_logits": ridge_logits,
            "attention_logits": attention_logits,
            "ridge_lambda": self.ridge_log_lambda.exp().expand(episodes),
            "ridge_scale": ridge_scale.expand(episodes),
            "attention_residual_scale": residual_scale.expand(episodes),
        }

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_labels: torch.Tensor,
        query_tokens: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        class_counts = self._validate_inputs(
            context_tokens, context_labels, query_tokens
        )
        ridge_logits = self._ridge_logits(
            context_tokens, context_labels, query_tokens, class_counts
        )
        attention_logits, attention_auxiliary = super().forward(
            context_tokens,
            context_labels,
            query_tokens,
            return_auxiliary=True,
        )
        ridge_scale = self.ridge_log_scale.exp().clamp(0.1, 100.0)
        residual_scale = torch.sigmoid(self.attention_residual_logit)
        logits = ridge_scale * ridge_logits + residual_scale * attention_logits
        if not return_auxiliary:
            return logits
        return logits, {
            **attention_auxiliary,
            "ridge_logits": ridge_logits,
            "attention_logits": attention_logits,
            "ridge_lambda": self.ridge_log_lambda.exp(),
            "ridge_scale": ridge_scale,
            "attention_residual_scale": residual_scale,
        }


class StructuredPopulationMetaClassifier(nn.Module):
    """Distribution-aware, label-equivariant class-memory meta classifier."""

    def __init__(
        self,
        token_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_set_layers: int = 1,
        relation_hidden_dim: int = 256,
        ridge_dim: int = 64,
        ridge_lambda: float = 1.0,
        ridge_logit_scale: float = 5.0,
        attention_residual_scale: float = 0.1,
        population_residual_scale: float = 0.25,
        tail_residual_scale: float = 0.10,
        minimum_population_residual_scale: float = 0.10,
        minimum_tail_residual_scale: float = 0.05,
        routing_temperature: float = 0.5,
        class_memory_tokens: int = 8,
        rare_evidence_fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
        ccer_temperatures: Sequence[float] = (),
        ccer_residual_scale: float = 0.10,
        ccer_presence_temperature: float = 0.5,
        ccer_v2_topks: Sequence[int] = (),
        ccer_v2_route_floor: float = 0.30,
        ccer_v2_residual_scale: float = 0.10,
        dr_ccer_enabled: bool = False,
        dr_ccer_abs_topks: Sequence[int] = (1, 4, 16),
        dr_ccer_frac_topks: Sequence[float] = (0.01, 0.05),
        dr_ccer_donor_topk: int = 3,
        dr_ccer_use_bottom_tail: bool = True,
        dr_ccer_expert_hidden: int = 128,
        dr_ccer_route_floor: float = 0.05,
        dr_ccer_max_donors: int = 32,
        fusion_residual_scale: float = 0.10,
        covariance_ridge_logit_scale: float = 2.0,
        covariance_residual_scale: float = 0.25,
        covariance_relation: dict[str, object] | None = None,
        mean_pool_structured_tokens: bool = False,
        project_structured_tokens: bool = False,
        structured_tokens_per_bag: int | None = None,
        projection_bottleneck_dim: int | None = None,
        projection_residual_mean: bool = False,
        typed_bag_preserving_branch: bool = False,
        typed_bag_bottleneck_dim: int | None = None,
        typed_bag_num_slots: int | None = None,
        typed_bag_num_tail_fractions: int | None = None,
        typed_bag_residual_scale: float = 0.02,
        include_cls_token: bool = False,
        raw_stat_tokens: Sequence[str] = (),
        use_instance_attention_mil: bool = False,
        mil_hidden_dim: int | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if not 0 < population_residual_scale < 1:
            raise ValueError("population_residual_scale must be in (0, 1).")
        if not 0 < tail_residual_scale < 1:
            raise ValueError("tail_residual_scale must be in (0, 1).")
        if not 0 <= minimum_population_residual_scale < population_residual_scale:
            raise ValueError(
                "minimum_population_residual_scale must be non-negative and "
                "smaller than population_residual_scale."
            )
        if not 0 <= minimum_tail_residual_scale < tail_residual_scale:
            raise ValueError(
                "minimum_tail_residual_scale must be non-negative and smaller "
                "than tail_residual_scale."
            )
        if routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive.")
        if class_memory_tokens < 1:
            raise ValueError("class_memory_tokens must be positive.")
        rare_fractions = tuple(float(value) for value in rare_evidence_fractions)
        if not rare_fractions or any(not 0 < value <= 1 for value in rare_fractions):
            raise ValueError("rare_evidence_fractions must contain values in (0, 1].")
        ccer_temperatures = tuple(float(value) for value in ccer_temperatures)
        if any(value <= 0 for value in ccer_temperatures):
            raise ValueError("ccer_temperatures must contain positive values.")
        if len(set(ccer_temperatures)) != len(ccer_temperatures):
            raise ValueError("ccer_temperatures must not contain duplicates.")
        if not 0 < ccer_residual_scale < 1:
            raise ValueError("ccer_residual_scale must be in (0, 1).")
        if ccer_presence_temperature <= 0:
            raise ValueError("ccer_presence_temperature must be positive.")
        ccer_v2_topks = tuple(int(value) for value in ccer_v2_topks)
        if any(value < 1 for value in ccer_v2_topks):
            raise ValueError("ccer_v2_topks must contain positive integers.")
        if len(set(ccer_v2_topks)) != len(ccer_v2_topks):
            raise ValueError("ccer_v2_topks must not contain duplicates.")
        if ccer_v2_topks and 1 not in ccer_v2_topks:
            raise ValueError("ccer_v2_topks must include 1 for the sparse route.")
        if not 0 <= ccer_v2_route_floor < 1:
            raise ValueError("ccer_v2_route_floor must be in [0, 1).")
        if not 0 < ccer_v2_residual_scale < 1:
            raise ValueError("ccer_v2_residual_scale must be in (0, 1).")
        dr_ccer_abs_topks = tuple(int(value) for value in dr_ccer_abs_topks)
        if any(value < 1 for value in dr_ccer_abs_topks):
            raise ValueError("dr_ccer_abs_topks must contain positive integers.")
        if len(set(dr_ccer_abs_topks)) != len(dr_ccer_abs_topks):
            raise ValueError("dr_ccer_abs_topks must not contain duplicates.")
        dr_ccer_frac_topks = tuple(float(value) for value in dr_ccer_frac_topks)
        if any(not 0 < value <= 1 for value in dr_ccer_frac_topks):
            raise ValueError("dr_ccer_frac_topks must contain values in (0, 1].")
        if dr_ccer_donor_topk < 1:
            raise ValueError("dr_ccer_donor_topk must be positive.")
        if not 0 <= dr_ccer_route_floor < 1:
            raise ValueError("dr_ccer_route_floor must be in [0, 1).")
        if dr_ccer_max_donors < 2:
            raise ValueError("dr_ccer_max_donors must be at least 2.")
        if dr_ccer_enabled and not dr_ccer_abs_topks and not dr_ccer_frac_topks:
            raise ValueError("DR-CCER requires at least one scan route.")
        if not 0 < fusion_residual_scale < 1:
            raise ValueError("fusion_residual_scale must be in (0, 1).")
        if covariance_ridge_logit_scale <= 0:
            raise ValueError("covariance_ridge_logit_scale must be positive.")
        if not 0 < covariance_residual_scale < 1:
            raise ValueError("covariance_residual_scale must be in (0, 1).")
        if not 0 < typed_bag_residual_scale < 1:
            raise ValueError("typed_bag_residual_scale must be in (0, 1).")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.routing_temperature = float(routing_temperature)
        self.class_memory_tokens = int(class_memory_tokens)
        self.rare_evidence_fractions = rare_fractions
        self.ccer_temperatures = ccer_temperatures
        self.ccer_presence_temperature = float(ccer_presence_temperature)
        self.ccer_v2_topks = ccer_v2_topks
        self.ccer_v2_route_floor = float(ccer_v2_route_floor)
        self.dr_ccer_enabled = bool(dr_ccer_enabled)
        self.dr_ccer_abs_topks = dr_ccer_abs_topks
        self.dr_ccer_frac_topks = dr_ccer_frac_topks
        self.dr_ccer_donor_topk = int(dr_ccer_donor_topk)
        self.dr_ccer_use_bottom_tail = bool(dr_ccer_use_bottom_tail)
        self.dr_ccer_route_floor = float(dr_ccer_route_floor)
        self.dr_ccer_max_donors = int(dr_ccer_max_donors)
        self.mean_pool_structured_tokens = bool(mean_pool_structured_tokens)
        self.project_structured_tokens = bool(project_structured_tokens)
        if self.mean_pool_structured_tokens and self.project_structured_tokens:
            raise ValueError(
                "mean_pool_structured_tokens and project_structured_tokens "
                "are mutually exclusive."
            )
        self.include_cls_token = bool(include_cls_token)
        self.raw_stat_tokens = tuple(raw_stat_tokens)
        self.use_instance_attention_mil = bool(use_instance_attention_mil)
        if self.include_cls_token and typed_bag_preserving_branch:
            raise NotImplementedError(
                "include_cls_token with typed_bag_preserving_branch is not "
                "supported: the typed-bag token-type/tail-fraction embeddings "
                "are sized from typed_bag_num_slots/typed_bag_num_tail_fractions "
                "only and do not have a role for the extra cls token position."
            )
        self.structured_tokens_per_bag = (
            None if structured_tokens_per_bag is None else int(structured_tokens_per_bag)
        )
        self.projection_bottleneck_dim = (
            None
            if projection_bottleneck_dim is None
            else int(projection_bottleneck_dim)
        )
        self.projection_residual_mean = bool(projection_residual_mean)
        if self.project_structured_tokens:
            if self.structured_tokens_per_bag is None or self.structured_tokens_per_bag < 1:
                raise ValueError(
                    "project_structured_tokens requires structured_tokens_per_bag >= 1."
                )
            if self.projection_bottleneck_dim is None:
                in_dim = self.structured_tokens_per_bag * self.token_dim
                if self.projection_residual_mean:
                    in_dim += self.token_dim
                self.bag_token_projection = nn.Linear(
                    in_dim,
                    self.token_dim,
                )
            else:
                if self.projection_bottleneck_dim < 1:
                    raise ValueError("projection_bottleneck_dim must be positive.")
                # v24-B0: one position-specific Linear(token_dim -> bottleneck)
                # per structured token, then one Linear(tokens * bottleneck ->
                # token_dim). Keeps all slots while shrinking the 40*512 input.
                self.bag_token_bottlenecks = nn.ModuleList(
                    [
                        nn.Linear(self.token_dim, self.projection_bottleneck_dim)
                        for _ in range(self.structured_tokens_per_bag)
                    ]
                )
                in_dim = self.structured_tokens_per_bag * self.projection_bottleneck_dim
                if self.projection_residual_mean:
                    in_dim += self.token_dim
                self.bag_token_projection = nn.Linear(
                    in_dim,
                    self.token_dim,
                )
        self.typed_bag_preserving_branch = bool(typed_bag_preserving_branch)
        if self.typed_bag_preserving_branch:
            if self.structured_tokens_per_bag is None or self.structured_tokens_per_bag < 1:
                raise ValueError(
                    "typed_bag_preserving_branch requires structured_tokens_per_bag >= 1."
                )
            if typed_bag_num_slots is None or typed_bag_num_tail_fractions is None:
                raise ValueError(
                    "typed_bag_preserving_branch requires typed_bag_num_slots and "
                    "typed_bag_num_tail_fractions."
                )
            num_slots = int(typed_bag_num_slots)
            num_tails = int(typed_bag_num_tail_fractions)
            if 1 + 3 * num_slots + num_tails != self.structured_tokens_per_bag:
                raise ValueError(
                    "typed_bag_num_slots/typed_bag_num_tail_fractions do not match "
                    "structured_tokens_per_bag."
                )
            # T5-A (typed, bag-preserving structured context branch): fixed
            # token layout is [global] + [slot_0 center/spread/rare, ...,
            # slot_{num_slots-1} center/spread/rare] + [tail_0, ..., tail_{k-1}]
            # (see StructuredEpisodePopulationAggregator.forward and
            # _all_structured_tokens). token_type: 0=global, 1=center,
            # 2=spread, 3=rare, 4=tail. tail_fraction sentinel = 0 for tokens
            # without a tail fraction (tail tokens use 1-indexed position).
            #
            # Deliberately NOT embedding slot index (unlike the T5-A design
            # doc's original proposal): slots come from per-episode
            # spherical k-means over context cells
            # (_context_spherical_kmeans_anchors), so slot index is an
            # arbitrary/exchangeable cluster id with no stable meaning across
            # bags or episodes -- a global embedding keyed on it has no
            # consistent target to learn (unlike token_type/tail_fraction,
            # which name a fixed role/threshold every bag shares). v24's
            # per-position bottleneck Linears already differentiate the 40
            # positions more expressively than an additive slot-index
            # embedding could, so nothing is lost by leaving it out.
            token_type_ids = [0]
            tail_fraction_ids = [0]
            for _slot in range(num_slots):
                token_type_ids.extend((1, 2, 3))
                tail_fraction_ids.extend((0, 0, 0))
            for tail in range(num_tails):
                token_type_ids.append(4)
                tail_fraction_ids.append(tail + 1)
            self.register_buffer(
                "typed_bag_token_type_ids",
                torch.tensor(token_type_ids, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "typed_bag_tail_fraction_ids",
                torch.tensor(tail_fraction_ids, dtype=torch.long),
                persistent=False,
            )
            self.typed_bag_token_type_embedding = nn.Embedding(5, self.token_dim)
            self.typed_bag_tail_fraction_embedding = nn.Embedding(
                num_tails + 1, self.token_dim
            )
            self.typed_bag_bottleneck_dim = (
                None
                if typed_bag_bottleneck_dim is None
                else int(typed_bag_bottleneck_dim)
            )
            if self.typed_bag_bottleneck_dim is None:
                in_dim = self.structured_tokens_per_bag * self.token_dim + self.token_dim
                self.typed_bag_token_projection = nn.Linear(in_dim, self.token_dim)
            else:
                if self.typed_bag_bottleneck_dim < 1:
                    raise ValueError("typed_bag_bottleneck_dim must be positive.")
                self.typed_bag_token_bottlenecks = nn.ModuleList(
                    [
                        nn.Linear(self.token_dim, self.typed_bag_bottleneck_dim)
                        for _ in range(self.structured_tokens_per_bag)
                    ]
                )
                in_dim = (
                    self.structured_tokens_per_bag * self.typed_bag_bottleneck_dim
                    + self.token_dim
                )
                self.typed_bag_token_projection = nn.Linear(in_dim, self.token_dim)
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
        self.minimum_population_residual_scale = float(
            minimum_population_residual_scale
        )
        self.minimum_tail_residual_scale = float(minimum_tail_residual_scale)
        self.global_shape_classifier = RidgeResidualMetaClassifier(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_set_layers=num_set_layers,
            relation_hidden_dim=relation_hidden_dim,
            ridge_dim=ridge_dim,
            ridge_lambda=ridge_lambda,
            ridge_logit_scale=ridge_logit_scale,
            attention_residual_scale=attention_residual_scale,
            num_classes=num_classes,
        )
        if self.typed_bag_preserving_branch:
            self.typed_bag_classifier = RidgeResidualMetaClassifier(
                token_dim=token_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_set_layers=num_set_layers,
                relation_hidden_dim=relation_hidden_dim,
                ridge_dim=ridge_dim,
                ridge_lambda=ridge_lambda,
                ridge_logit_scale=ridge_logit_scale,
                attention_residual_scale=attention_residual_scale,
                num_classes=num_classes,
            )
            typed_bag_logit = math.log(
                typed_bag_residual_scale / (1.0 - typed_bag_residual_scale)
            )
            self.typed_bag_residual_logit = nn.Parameter(torch.tensor(typed_bag_logit))
        self.abundance_ridge_log_lambda = nn.Parameter(
            torch.tensor(math.log(ridge_lambda))
        )
        self.abundance_ridge_log_scale = nn.Parameter(
            torch.tensor(math.log(ridge_logit_scale))
        )
        self.covariance_ridge_log_lambda = nn.Parameter(
            torch.tensor(math.log(ridge_lambda))
        )
        self.covariance_ridge_log_scale = nn.Parameter(
            torch.tensor(math.log(covariance_ridge_logit_scale))
        )
        covariance_logit = math.log(
            covariance_residual_scale / (1.0 - covariance_residual_scale)
        )
        self.covariance_residual_logit = nn.Parameter(torch.tensor(covariance_logit))
        population_attention_logit = math.log(
            attention_residual_scale / (1.0 - attention_residual_scale)
        )
        self.population_attention_residual_logit = nn.Parameter(
            torch.tensor(population_attention_logit)
        )
        self.memory_input_norm = nn.LayerNorm(token_dim)
        self.memory_input_projection = nn.Linear(token_dim, hidden_dim)
        self.memory_seeds = nn.Parameter(
            torch.randn(class_memory_tokens, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.memory_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.memory_norm = nn.LayerNorm(hidden_dim)
        if num_set_layers:
            memory_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.memory_encoder: nn.Module = nn.TransformerEncoder(
                memory_layer,
                num_layers=num_set_layers,
                enable_nested_tensor=False,
            )
        else:
            self.memory_encoder = nn.Identity()

        self.slot_input_norm = nn.LayerNorm(token_dim)
        self.slot_input_projection = nn.Linear(token_dim, hidden_dim)
        self.population_cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.slot_relation_scorer = self._make_relation_scorer(
            hidden_dim, relation_hidden_dim
        )
        self.slot_importance = self._make_importance_scorer(
            token_dim, relation_hidden_dim
        )

        self.instance_input_norm = nn.LayerNorm(token_dim)
        self.instance_input_projection = nn.Linear(token_dim, hidden_dim)
        self.rare_similarity_log_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.rare_evidence_head = nn.Sequential(
            nn.LayerNorm(len(rare_fractions)),
            nn.Linear(len(rare_fractions), relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 1),
        )
        if self.ccer_temperatures:
            # The same temperature weights are shared across classes, preserving
            # label equivariance.  A support-class separation statistic adjusts
            # the sparse/dense mixture per episode without looking at query labels.
            self.ccer_route_logits = nn.Parameter(
                torch.zeros(len(self.ccer_temperatures))
            )
            self.ccer_support_router = nn.Linear(1, len(self.ccer_temperatures))
            nn.init.zeros_(self.ccer_support_router.weight)
            nn.init.zeros_(self.ccer_support_router.bias)
            self.ccer_null_threshold = nn.Parameter(torch.tensor(1.0))
            ccer_logit = math.log(
                ccer_residual_scale / (1.0 - ccer_residual_scale)
            )
            self.ccer_residual_logit = nn.Parameter(torch.tensor(ccer_logit))
        if self.ccer_v2_topks:
            # CCER-v2 deliberately owns its complete instance/prototype path.
            # Gradients from this branch cannot weaken the v30 rare branch.
            self.ccer_v2_instance_encoder = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.ccer_v2_prototype_encoder = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.ccer_v2_similarity_log_scale = nn.Parameter(
                torch.tensor(math.log(5.0))
            )
            route_count = len(self.ccer_v2_topks) + 1  # + dense mean route
            router_feature_dim = route_count + 2  # route gaps + support sep + log n
            self.ccer_v2_router = nn.Sequential(
                nn.LayerNorm(router_feature_dim),
                nn.Linear(router_feature_dim, relation_hidden_dim),
                nn.GELU(),
                nn.Linear(relation_hidden_dim, route_count),
            )
            self.ccer_v2_output_head = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.ccer_v2_output_head.weight)
            ccer_v2_logit = math.log(
                ccer_v2_residual_scale / (1.0 - ccer_v2_residual_scale)
            )
            self.ccer_v2_residual_logit = nn.Parameter(
                torch.tensor(ccer_v2_logit)
            )
        if self.dr_ccer_enabled:
            # DR-CCER owns a complete donor-resolved evidence path.  The expert
            # is trained with its own loss (CE + ranking + donor consistency)
            # while v30 is frozen, then gated into v30 by a reliability router.
            self.dr_ccer_donor_encoder = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, dr_ccer_expert_hidden),
                nn.GELU(),
                nn.Linear(dr_ccer_expert_hidden, dr_ccer_expert_hidden),
            )
            self.dr_ccer_instance_encoder = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, dr_ccer_expert_hidden),
                nn.GELU(),
                nn.Linear(dr_ccer_expert_hidden, dr_ccer_expert_hidden),
            )
            self.dr_ccer_similarity_log_scale = nn.Parameter(
                torch.tensor(math.log(3.0))
            )
            self.dr_ccer_expert_log_scale = nn.Parameter(torch.tensor(0.0))
            self.dr_ccer_route_count = (
                len(self.dr_ccer_abs_topks)
                + len(self.dr_ccer_frac_topks)
                + 1  # dense mean
                + (1 if self.dr_ccer_use_bottom_tail else 0)
                + 1  # agreement mean
            )
            route_feature_dim = self.dr_ccer_route_count + 3  # spread + disp + log n
            self.dr_ccer_route_router = nn.Sequential(
                nn.LayerNorm(route_feature_dim),
                nn.Linear(route_feature_dim, dr_ccer_expert_hidden // 2),
                nn.GELU(),
                nn.Linear(dr_ccer_expert_hidden // 2, self.dr_ccer_route_count),
            )
            gate_feature_dim = 6  # |routed|, spread, disp, log n, sep, mean|margin|
            self.dr_ccer_reliability_router = nn.Sequential(
                nn.LayerNorm(gate_feature_dim),
                nn.Linear(gate_feature_dim, 16),
                nn.GELU(),
                nn.Linear(16, 1),
            )
            # Closed gate at init: g = sigmoid(-4) ~= 0.018, so the exact v30
            # prediction is preserved until the expert earns replacement.
            nn.init.zeros_(self.dr_ccer_reliability_router[-1].weight)
            nn.init.constant_(self.dr_ccer_reliability_router[-1].bias, -4.0)
            self.dr_ccer_output_head = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.dr_ccer_output_head.weight)
        self.fusion_scorer = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, relation_hidden_dim),
            nn.GELU(),
            nn.Linear(relation_hidden_dim, 1),
        )
        fusion_logit = math.log(fusion_residual_scale / (1.0 - fusion_residual_scale))
        self.fusion_residual_logit = nn.Parameter(torch.tensor(fusion_logit))
        # A sigmoid gate with a positive floor prevents either specialized path
        # from becoming permanently disconnected from the final prediction.
        self.population_residual_logit = nn.Parameter(
            torch.tensor(
                self._residual_scale_to_logit(
                    population_residual_scale,
                    self.minimum_population_residual_scale,
                )
            )
        )
        self.tail_residual_logit = nn.Parameter(
            torch.tensor(
                self._residual_scale_to_logit(
                    tail_residual_scale,
                    self.minimum_tail_residual_scale,
                )
            )
        )
        if self.use_instance_attention_mil:
            mil_hidden = int(mil_hidden_dim if mil_hidden_dim is not None else hidden_dim)
            if mil_hidden < 1:
                raise ValueError("mil_hidden_dim must be positive.")
            self.mil_hidden_dim = mil_hidden
            self.mil_instance_encoder = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, mil_hidden),
                nn.GELU(),
                nn.Linear(mil_hidden, mil_hidden),
                nn.GELU(),
            )
            self.mil_relevance_mlp = nn.Sequential(
                nn.LayerNorm(2 * mil_hidden),
                nn.Linear(2 * mil_hidden, mil_hidden),
                nn.GELU(),
                nn.Linear(mil_hidden, 1),
            )
            self.mil_score_head = nn.Sequential(
                nn.LayerNorm(2 * mil_hidden),
                nn.Linear(2 * mil_hidden, mil_hidden),
                nn.GELU(),
                nn.Linear(mil_hidden, 1),
            )
            self.mil_attention_log_scale = nn.Parameter(
                torch.tensor(math.log(5.0))
            )
            self.mil_residual_logit = nn.Parameter(
                torch.tensor(
                    self._residual_scale_to_logit(0.10, self.minimum_tail_residual_scale)
                )
            )

    @staticmethod
    def _residual_scale_to_logit(scale: float, minimum: float) -> float:
        unit_scale = (scale - minimum) / (1.0 - minimum)
        return math.log(unit_scale / (1.0 - unit_scale))

    @staticmethod
    def _floored_residual_scale(logit: torch.Tensor, minimum: float) -> torch.Tensor:
        return minimum + (1.0 - minimum) * torch.sigmoid(logit)

    @staticmethod
    def _make_relation_scorer(token_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(4 * token_dim),
            nn.Linear(4 * token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _make_importance_scorer(token_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

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
        margin_rms = margin.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(
            self.covariance_relation_eps
        )
        bounded_margin = torch.tanh(margin / margin_rms)
        logits = torch.stack((-0.5 * bounded_margin, 0.5 * bounded_margin), dim=-1)
        if not batched:
            return logits.squeeze(0), separation.squeeze(0)
        return logits, separation

    def _slot_covariance_relation_scores(
        self,
        context_covariance: torch.Tensor,
        context_reliability: torch.Tensor,
        context_labels: torch.Tensor,
        query_covariance: torch.Tensor,
        query_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare slot-local covariance relative to context class structure."""
        batched = context_covariance.ndim == 4
        if not batched:
            context_covariance = context_covariance.unsqueeze(0)
            context_reliability = context_reliability.unsqueeze(0)
            context_labels = context_labels.unsqueeze(0)
            query_covariance = query_covariance.unsqueeze(0)
            query_reliability = query_reliability.unsqueeze(0)
        context32 = context_covariance.float()
        query32 = query_covariance.float()
        reliability = context_reliability.float().clamp_min(self.covariance_relation_eps)
        normalization_weight = reliability / reliability.sum(dim=1, keepdim=True)
        center = (normalization_weight.unsqueeze(-1) * context32).sum(dim=1, keepdim=True)
        scale = torch.sqrt(
            (
                normalization_weight.unsqueeze(-1)
                * (context32 - center).square()
            ).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            + self.covariance_relation_eps
        )
        context_z = (context32 - center) / scale
        query_z = (query32 - center) / scale
        class_weights = []
        prototypes = []
        for class_index in range(self.num_classes):
            mask = (context_labels == class_index).float().unsqueeze(-1)
            weights = mask * reliability
            if torch.any(weights.sum(dim=1) == 0):
                raise ValueError(
                    "Slot covariance relation requires every class in context."
                )
            weights = weights / weights.sum(dim=1, keepdim=True)
            class_weights.append(weights)
            prototypes.append((weights.unsqueeze(-1) * context_z).sum(dim=1))
        prototypes = torch.stack(prototypes, dim=1)
        separation_per_slot = (
            prototypes[:, 1] - prototypes[:, 0]
        ).square().mean(dim=-1).sqrt()

        if self.covariance_relation_mode == "prototype_cosine":
            class_scores = torch.einsum(
                "eqsd,ecsd->eqcs",
                F.normalize(query_z, dim=-1, eps=self.covariance_relation_eps),
                F.normalize(prototypes, dim=-1, eps=self.covariance_relation_eps),
            )
        elif self.covariance_relation_mode == "standardized_distance":
            differences = context_z.unsqueeze(2) - prototypes.unsqueeze(1)
            dispersions = torch.stack(
                [
                    (
                        class_weights[class_index].unsqueeze(-1)
                        * differences[:, :, class_index].square()
                    ).sum(dim=1).mean(dim=-1)
                    for class_index in range(self.num_classes)
                ],
                dim=1,
            ).clamp_min(self.covariance_relation_eps)
            distances = (
                query_z.unsqueeze(2) - prototypes.unsqueeze(1)
            ).square().mean(dim=-1)
            class_scores = -distances / dispersions.unsqueeze(1)
        else:
            distances = (
                query_z.unsqueeze(2) - context_z.unsqueeze(1)
            ).square().mean(dim=-1)
            pairwise = (
                context_z.unsqueeze(2) - context_z.unsqueeze(1)
            ).square().mean(dim=-1)
            count = context_z.shape[1]
            upper = torch.triu_indices(count, count, offset=1, device=context_z.device)
            temperature = pairwise[:, upper[0], upper[1]].median(dim=1).values
            temperature = temperature.detach().clamp_min(self.covariance_relation_eps)
            scale_scores = []
            for kernel_scale in self.covariance_relation_kernel_scales:
                kernel = torch.exp(
                    -distances / (temperature[:, None, None, :] * kernel_scale)
                )
                scores = torch.stack(
                    [
                        torch.einsum("eqcs,ecs->eqs", kernel, weights)
                        for weights in class_weights
                    ],
                    dim=2,
                )
                scale_scores.append(scores.clamp_min(self.covariance_relation_eps).log())
            class_scores = torch.stack(scale_scores).mean(dim=0)

        slot_margin = class_scores[:, :, 1] - class_scores[:, :, 0]
        margin_rms = slot_margin.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        slot_margin = torch.tanh(
            slot_margin / margin_rms.clamp_min(self.covariance_relation_eps)
        )
        query_weights = query_reliability.float().clamp_min(0)
        if self.covariance_relation_slot_routing == "context_top1":
            routing = torch.zeros_like(separation_per_slot)
            routing.scatter_(1, separation_per_slot.argmax(dim=-1, keepdim=True), 1.0)
            query_weights = query_weights * routing.unsqueeze(1)
        elif self.covariance_relation_slot_routing == "context_top3":
            count = min(3, separation_per_slot.shape[-1])
            routing = torch.zeros_like(separation_per_slot)
            routing.scatter_(1, separation_per_slot.topk(count, dim=-1).indices, 1.0)
            query_weights = query_weights * routing.unsqueeze(1)
        elif self.covariance_relation_slot_routing == "context_softmax":
            routing = torch.softmax(
                separation_per_slot / self.covariance_relation_routing_temperature,
                dim=-1,
            )
            query_weights = query_weights * routing.unsqueeze(1)
        query_weights = query_weights / query_weights.sum(dim=-1, keepdim=True).clamp_min(
            self.covariance_relation_eps
        )
        margin = (query_weights * slot_margin).sum(dim=-1)
        logits = torch.stack((-0.5 * margin, 0.5 * margin), dim=-1)
        mean_reliability = reliability.mean(dim=1)
        separation = (
            mean_reliability * separation_per_slot
        ).sum(dim=-1) / mean_reliability.sum(dim=-1).clamp_min(
            self.covariance_relation_eps
        )
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


    def _validate_representation(
        self,
        representation: dict[str, torch.Tensor],
        name: str,
    ) -> None:
        expected_keys = {
            "global_summary", "slots", "tails", "slot_metadata",
            "covariance_sketch", "slot_covariance_sketch",
            "slot_covariance_reliability", "covariance_matrix", "cls_token",
        }
        for stat_name in self.raw_stat_tokens:
            expected_keys.add(f"stat_{stat_name}")
        if set(representation) != expected_keys:
            raise ValueError(f"{name} has invalid structured representation keys.")
        global_summary = representation["global_summary"]
        slots = representation["slots"]
        tails = representation["tails"]
        metadata = representation["slot_metadata"]
        covariance_sketch = representation["covariance_sketch"]
        cls_token = representation["cls_token"]
        if global_summary.ndim != 2 or global_summary.shape[-1] != self.token_dim:
            raise ValueError(f"{name} global-summary tokens have an invalid shape.")
        if (
            slots.ndim != 4
            or slots.shape[0] != global_summary.shape[0]
            or slots.shape[-1] != self.token_dim
        ):
            raise ValueError(f"{name} slot tokens have an invalid shape.")
        if (
            tails.ndim != 3
            or tails.shape[0] != global_summary.shape[0]
            or tails.shape[-1] != self.token_dim
        ):
            raise ValueError(f"{name} tail tokens have an invalid shape.")
        if metadata.shape != slots.shape[:2] + (2,):
            raise ValueError(f"{name} slot metadata have an invalid shape.")
        if covariance_sketch.ndim != 2 or covariance_sketch.shape[0] != global_summary.shape[0]:
            raise ValueError(f"{name} covariance sketches have an invalid shape.")
        if (
            cls_token.ndim != 2
            or cls_token.shape[0] != global_summary.shape[0]
            or cls_token.shape[-1] != self.token_dim
        ):
            raise ValueError(f"{name} cls token has an invalid shape.")

    @staticmethod
    def _flatten_slot_tokens(representation: dict[str, torch.Tensor]) -> torch.Tensor:
        slots = representation["slots"]
        return slots.reshape(slots.shape[0], -1, slots.shape[-1])

    def _all_structured_tokens(
        self, representation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        parts = [
            representation["global_summary"].unsqueeze(1),
            self._flatten_slot_tokens(representation),
            representation["tails"],
        ]
        if self.include_cls_token:
            parts.append(representation["cls_token"].unsqueeze(1))
        for stat_name in self.raw_stat_tokens:
            parts.append(representation[f"stat_{stat_name}"].unsqueeze(1))
        return torch.cat(parts, dim=1)

    def _projected_bag_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reduce each bag's stacked structured tokens to one projected token.

        v24-A0 concatenates the per-bag structured tokens along the feature axis
        (`num_tokens * token_dim`) and applies one learned linear projection.
        v24-B0 first applies a position-specific Linear(token_dim -> bottleneck)
        to every token, then concatenates (`num_tokens * bottleneck`) and applies
        the projection to 512.
        """
        if tokens.shape[-2] != self.structured_tokens_per_bag:
            raise ValueError(
                "Structured token count does not match bag projection input: "
                f"expected {self.structured_tokens_per_bag}, got {tokens.shape[-2]}."
            )
        if self.projection_bottleneck_dim is None:
            flat = tokens.reshape(*tokens.shape[:-2], -1)
            if self.projection_residual_mean:
                mean_token = tokens.mean(dim=-2)
                flat = torch.cat([flat, mean_token], dim=-1)
            return self.bag_token_projection(flat)
        compressed = torch.stack(
            [
                self.bag_token_bottlenecks[index](tokens[..., index, :])
                for index in range(self.structured_tokens_per_bag)
            ],
            dim=-2,
        )
        flat = compressed.reshape(*compressed.shape[:-2], -1)
        if self.projection_residual_mean:
            mean_token = tokens.mean(dim=-2)
            flat = torch.cat([flat, mean_token], dim=-1)
        return self.bag_token_projection(flat)

    def _typed_bag_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """T5-A typed, bag-preserving branch: add learned token-type and
        tail-fraction identity embeddings to a bag's structured tokens, then
        reduce to one embedding per bag with an independent copy of the
        v24-B1 bottleneck + residual-mean projection. Unlike
        `_projected_bag_tokens`, this feeds a separate `typed_bag_classifier`
        and never replaces the tokens consumed by
        `_class_memories`/`_population_tokens`. No slot-index embedding is
        added: slots are per-episode k-means cluster ids with no stable
        cross-bag identity (see the constructor comment).
        """
        if tokens.shape[-2] != self.structured_tokens_per_bag:
            raise ValueError(
                "Structured token count does not match typed bag projection "
                f"input: expected {self.structured_tokens_per_bag}, got "
                f"{tokens.shape[-2]}."
            )
        typed_tokens = (
            tokens
            + self.typed_bag_token_type_embedding(self.typed_bag_token_type_ids)
            + self.typed_bag_tail_fraction_embedding(self.typed_bag_tail_fraction_ids)
        )
        mean_token = typed_tokens.mean(dim=-2)
        if self.typed_bag_bottleneck_dim is None:
            flat = typed_tokens.reshape(*typed_tokens.shape[:-2], -1)
        else:
            compressed = torch.stack(
                [
                    self.typed_bag_token_bottlenecks[index](typed_tokens[..., index, :])
                    for index in range(self.structured_tokens_per_bag)
                ],
                dim=-2,
            )
            flat = compressed.reshape(*compressed.shape[:-2], -1)
        flat = torch.cat([flat, mean_token], dim=-1)
        return self.typed_bag_token_projection(flat)

    def _population_tokens(
        self, representation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return either all structured tokens, one exact mean token, or one
        learned projection token per bag."""
        if self.project_structured_tokens:
            tokens = self._all_structured_tokens(representation)
            return self._projected_bag_tokens(tokens).unsqueeze(-2)
        if self.mean_pool_structured_tokens:
            return self._all_structured_tokens(representation).mean(
                dim=-2, keepdim=True
            )
        return self._flatten_slot_tokens(representation)

    def _class_memories(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        context_tokens = self._all_structured_tokens(context)
        if self.project_structured_tokens:
            context_tokens = self._projected_bag_tokens(context_tokens).unsqueeze(1)
        elif self.mean_pool_structured_tokens:
            context_tokens = context_tokens.mean(dim=1, keepdim=True)
        memories: list[torch.Tensor] = []
        for class_index in range(self.num_classes):
            class_tokens = context_tokens[context_labels == class_index].reshape(
                -1, self.token_dim
            )
            encoded = self.memory_input_projection(
                self.memory_input_norm(class_tokens)
            ).unsqueeze(0)
            seeds = self.memory_seeds.unsqueeze(0)
            attended, _ = self.memory_cross_attention(
                seeds, encoded, encoded, need_weights=False
            )
            memory = self.memory_norm(seeds + attended)
            memories.append(self.memory_encoder(memory).squeeze(0))
        return torch.stack(memories)

    def _population_memory_logits(
        self,
        query: dict[str, torch.Tensor],
        class_memories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_tokens = self._population_tokens(query)
        encoded_query = self.slot_input_projection(self.slot_input_norm(query_tokens))
        importance_logits = self.slot_importance(query_tokens).squeeze(-1)
        token_weights = F.softmax(
            importance_logits.float() / self.routing_temperature,
            dim=-1,
        ).to(query_tokens.dtype)

        class_logits: list[torch.Tensor] = []
        for class_index in range(self.num_classes):
            memory = (
                class_memories[class_index]
                .unsqueeze(0)
                .expand(encoded_query.shape[0], -1, -1)
            )
            attended, _ = self.population_cross_attention(
                encoded_query, memory, memory, need_weights=False
            )
            relation_features = torch.cat(
                (
                    encoded_query,
                    attended,
                    encoded_query - attended,
                    encoded_query * attended,
                ),
                dim=-1,
            )
            relation = self.slot_relation_scorer(relation_features).squeeze(-1)
            class_logits.append((relation * token_weights).sum(dim=-1))
        return torch.stack(class_logits, dim=-1), token_weights

    def _rare_instance_logits(
        self,
        query_instances: torch.Tensor | Sequence[torch.Tensor],
        class_memories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        class_memory32 = F.normalize(class_memories.float(), dim=-1)
        similarity_scale = self.rare_similarity_log_scale.exp().clamp(0.1, 50.0)
        if isinstance(query_instances, torch.Tensor):
            if query_instances.ndim != 3 or query_instances.shape[-1] != self.token_dim:
                raise ValueError(
                    "Dense query instances must be [queries, instances, token_dim]."
                )
            encoded = self.instance_input_projection(
                self.instance_input_norm(query_instances)
            )
            encoded32 = F.normalize(encoded.float(), dim=-1)
            similarities = similarity_scale.float() * torch.einsum(
                "qnd,cmd->qcnm", encoded32, class_memory32
            )
            evidence = torch.logsumexp(similarities, dim=-1) - math.log(
                self.class_memory_tokens
            )
            fraction_scores = []
            counts = []
            for fraction in self.rare_evidence_fractions:
                count = min(
                    query_instances.shape[1],
                    max(1, int(math.ceil(fraction * query_instances.shape[1]))),
                )
                fraction_scores.append(evidence.topk(count, dim=-1).values.mean(dim=-1))
                counts.append(count)
            stacked_scores = torch.stack(fraction_scores, dim=-1)
            logits = self.rare_evidence_head(
                stacked_scores.to(query_instances.dtype)
            ).squeeze(-1)
            return (
                logits,
                stacked_scores,
                torch.tensor(counts, device=class_memories.device).expand(
                    query_instances.shape[0], -1
                ),
            )
        query_logits: list[torch.Tensor] = []
        query_fraction_scores: list[torch.Tensor] = []
        query_counts: list[list[int]] = []
        for instances in query_instances:
            if instances.ndim != 2 or instances.shape[-1] != self.token_dim:
                raise ValueError(
                    f"Every query instance bag must be [instances, {self.token_dim}]."
                )
            encoded = self.instance_input_projection(
                self.instance_input_norm(instances)
            )
            encoded32 = F.normalize(encoded.float(), dim=-1)
            class_scores: list[torch.Tensor] = []
            fraction_scores_by_class: list[torch.Tensor] = []
            counts: list[int] = []
            for class_index in range(self.num_classes):
                similarities = similarity_scale.float() * (
                    encoded32 @ class_memory32[class_index].T
                )
                evidence = torch.logsumexp(similarities, dim=-1) - math.log(
                    self.class_memory_tokens
                )
                pooled: list[torch.Tensor] = []
                for fraction in self.rare_evidence_fractions:
                    count = min(
                        instances.shape[0],
                        max(1, int(math.ceil(fraction * instances.shape[0]))),
                    )
                    pooled.append(evidence.topk(count).values.mean())
                    if class_index == 0:
                        counts.append(count)
                fraction_scores = torch.stack(pooled)
                fraction_scores_by_class.append(fraction_scores)
                class_scores.append(
                    self.rare_evidence_head(
                        fraction_scores.to(instances.dtype)
                    ).squeeze(-1)
                )
            query_logits.append(torch.stack(class_scores))
            query_fraction_scores.append(torch.stack(fraction_scores_by_class))
            query_counts.append(counts)
        return (
            torch.stack(query_logits),
            torch.stack(query_fraction_scores),
            torch.tensor(query_counts, device=class_memories.device),
        )

    def _ccer_pool_evidence(
        self,
        evidence: torch.Tensor,
        class_memories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool class-conditional instance evidence without hard rank selection.

        `evidence` is either [queries, classes, instances] or
        [episodes, queries, classes, instances].  Cardinality-normalized
        LogMeanExp is exactly invariant to uniform duplication of all instances;
        temperatures interpolate between sparse/max-like and dense/mean-like
        evidence.  A class-neutral presence gate supplies the explicit null path.
        """
        num_instances = evidence.shape[-1]
        route_scores = torch.stack(
            [
                temperature
                * (
                    torch.logsumexp(evidence / temperature, dim=-1)
                    - math.log(float(num_instances))
                )
                for temperature in self.ccer_temperatures
            ],
            dim=-1,
        )
        # Remove evidence common to every class. Generic anomalies must not
        # become class evidence merely because all support memories match poorly.
        centered_routes = route_scores - route_scores.mean(dim=-2, keepdim=True)

        class_prototypes = class_memories.float().mean(dim=-2)
        support_separation = class_prototypes.var(
            dim=-2, unbiased=False
        ).mean(dim=-1).sqrt()
        route_logits = self.ccer_route_logits + self.ccer_support_router(
            torch.log1p(support_separation).unsqueeze(-1)
        )
        route_weights = torch.softmax(route_logits.float(), dim=-1).to(
            centered_routes.dtype
        )
        if evidence.ndim == 4:
            routed = (
                centered_routes * route_weights[:, None, None, :]
            ).sum(dim=-1)
        else:
            routed = (centered_routes * route_weights).sum(dim=-1)

        centered_instance_evidence = evidence - evidence.mean(
            dim=-2, keepdim=True
        )
        discriminative_strength = centered_instance_evidence.abs().amax(
            dim=(-2, -1)
        )
        presence = torch.sigmoid(
            (discriminative_strength - self.ccer_null_threshold)
            / self.ccer_presence_temperature
        ).to(routed.dtype)
        return presence.unsqueeze(-1) * routed, route_scores, presence, route_weights

    def _ccer_instance_logits_batched(
        self,
        query_instances: torch.Tensor,
        class_memories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.instance_input_projection(
            self.instance_input_norm(query_instances)
        )
        encoded32 = F.normalize(encoded.float(), dim=-1)
        memory32 = F.normalize(class_memories.float(), dim=-1)
        scale = self.rare_similarity_log_scale.exp().clamp(0.1, 50.0)
        similarities = scale.float() * torch.einsum(
            "eqnd,ecmd->eqcnm", encoded32, memory32
        )
        evidence = torch.logsumexp(similarities, dim=-1) - math.log(
            self.class_memory_tokens
        )
        return self._ccer_pool_evidence(evidence, class_memories)

    def _ccer_instance_logits(
        self,
        query_instances: torch.Tensor | Sequence[torch.Tensor],
        class_memories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(query_instances, torch.Tensor):
            if query_instances.ndim != 3:
                raise ValueError(
                    "Dense CCER query instances must be [queries, instances, token_dim]."
                )
            encoded = self.instance_input_projection(
                self.instance_input_norm(query_instances)
            )
            encoded32 = F.normalize(encoded.float(), dim=-1)
            memory32 = F.normalize(class_memories.float(), dim=-1)
            scale = self.rare_similarity_log_scale.exp().clamp(0.1, 50.0)
            similarities = scale.float() * torch.einsum(
                "qnd,cmd->qcnm", encoded32, memory32
            )
            evidence = torch.logsumexp(similarities, dim=-1) - math.log(
                self.class_memory_tokens
            )
            return self._ccer_pool_evidence(evidence, class_memories)

        memory32 = F.normalize(class_memories.float(), dim=-1)
        scale = self.rare_similarity_log_scale.exp().clamp(0.1, 50.0)
        pooled = []
        routes = []
        presences = []
        route_weights = None
        for instances in query_instances:
            encoded = self.instance_input_projection(self.instance_input_norm(instances))
            encoded32 = F.normalize(encoded.float(), dim=-1)
            similarities = scale.float() * torch.einsum(
                "nd,cmd->cnm", encoded32, memory32
            )
            evidence = torch.logsumexp(similarities, dim=-1) - math.log(
                self.class_memory_tokens
            )
            bag_logits, bag_routes, bag_presence, route_weights = (
                self._ccer_pool_evidence(evidence.unsqueeze(0), class_memories)
            )
            pooled.append(bag_logits.squeeze(0))
            routes.append(bag_routes.squeeze(0))
            presences.append(bag_presence.squeeze(0))
        if route_weights is None:
            raise ValueError("CCER requires at least one query bag.")
        return (
            torch.stack(pooled),
            torch.stack(routes),
            torch.stack(presences),
            route_weights,
        )

    def _ccer_v2_class_slot_prototypes(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Build class prototypes from unprojected aligned slot-center tokens."""
        slots = context["slots"]
        if slots.ndim == 5:
            # [episodes, context bags, slots, statistics, token_dim]
            centers = slots[:, :, :, 0, :]
            class_mask = F.one_hot(
                context_labels.long(), num_classes=self.num_classes
            ).to(centers.dtype)
            counts = class_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).unsqueeze(-1)
            return torch.einsum("ebc,ebsd->ecsd", class_mask, centers) / counts
        if slots.ndim != 4:
            raise ValueError(
                "CCER-v2 context slots must be [bags, slots, statistics, token_dim] "
                "or their batched equivalent."
            )
        centers = slots[:, :, 0, :]
        prototypes = []
        for class_index in range(self.num_classes):
            selected = centers[context_labels == class_index]
            if selected.shape[0] == 0:
                raise ValueError("CCER-v2 requires every class in the context set.")
            prototypes.append(selected.mean(dim=0))
        return torch.stack(prototypes)

    def _ccer_v2_pool_evidence(
        self,
        evidence: torch.Tensor,
        encoded_prototypes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route class-centered evidence through Top-K and dense paths.

        Evidence is [Q,C,N] or [E,Q,C,N].  Top-1 preserves a fixed causal
        instance when lower-scoring background is appended; Top-K and mean
        provide micro-population and dense alternatives.
        """
        centered = evidence - evidence.mean(dim=-2, keepdim=True)
        route_scores = []
        for requested_k in self.ccer_v2_topks:
            count = min(requested_k, centered.shape[-1])
            route_scores.append(
                centered.topk(count, dim=-1).values.mean(dim=-1)
            )
        route_scores.append(centered.mean(dim=-1))
        routes = torch.stack(route_scores, dim=-1)

        route_gaps = routes.amax(dim=-2) - routes.amin(dim=-2)
        prototype_centers = encoded_prototypes.float().mean(dim=-2)
        support_separation = prototype_centers.var(
            dim=-2, unbiased=False
        ).mean(dim=-1).sqrt()
        log_n = math.log(float(centered.shape[-1]))
        if evidence.ndim == 4:
            support_feature = support_separation[:, None, None].expand(
                -1, routes.shape[1], 1
            )
            cardinality_feature = routes.new_full(
                (routes.shape[0], routes.shape[1], 1), log_n
            )
        else:
            support_feature = support_separation.reshape(1, 1).expand(
                routes.shape[0], 1
            )
            cardinality_feature = routes.new_full((routes.shape[0], 1), log_n)
        router_features = torch.cat(
            (route_gaps, support_feature.to(routes.dtype), cardinality_feature),
            dim=-1,
        )
        route_logits = self.ccer_v2_router(router_features)
        soft_weights = torch.softmax(route_logits.float(), dim=-1).to(routes.dtype)
        route_count = routes.shape[-1]
        route_weights = (
            (1.0 - self.ccer_v2_route_floor) * soft_weights
            + self.ccer_v2_route_floor / route_count
        )
        routed = (routes * route_weights.unsqueeze(-2)).sum(dim=-1)
        logits = self.ccer_v2_output_head(routed.unsqueeze(-1)).squeeze(-1)
        return logits, routes, route_weights, router_features

    def _ccer_v2_logits_batched(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query_instances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prototypes = self._ccer_v2_class_slot_prototypes(context, context_labels)
        encoded_prototypes = F.normalize(
            self.ccer_v2_prototype_encoder(prototypes).float(), dim=-1
        )
        encoded_instances = F.normalize(
            self.ccer_v2_instance_encoder(query_instances).float(), dim=-1
        )
        scale = self.ccer_v2_similarity_log_scale.exp().clamp(0.1, 50.0)
        similarities = scale.float() * torch.einsum(
            "eqnd,ecsd->eqcns", encoded_instances, encoded_prototypes
        )
        evidence = torch.logsumexp(similarities, dim=-1) - math.log(
            prototypes.shape[-2]
        )
        return self._ccer_v2_pool_evidence(evidence, encoded_prototypes)

    def _ccer_v2_logits(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query_instances: torch.Tensor | Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prototypes = self._ccer_v2_class_slot_prototypes(context, context_labels)
        encoded_prototypes = F.normalize(
            self.ccer_v2_prototype_encoder(prototypes).float(), dim=-1
        )
        scale = self.ccer_v2_similarity_log_scale.exp().clamp(0.1, 50.0)
        if isinstance(query_instances, torch.Tensor):
            encoded_instances = F.normalize(
                self.ccer_v2_instance_encoder(query_instances).float(), dim=-1
            )
            similarities = scale.float() * torch.einsum(
                "qnd,csd->qcns", encoded_instances, encoded_prototypes
            )
            evidence = torch.logsumexp(similarities, dim=-1) - math.log(
                prototypes.shape[-2]
            )
            return self._ccer_v2_pool_evidence(evidence, encoded_prototypes)

        logits = []
        routes = []
        weights = []
        features = []
        for instances in query_instances:
            encoded_instances = F.normalize(
                self.ccer_v2_instance_encoder(instances).float(), dim=-1
            )
            similarities = scale.float() * torch.einsum(
                "nd,csd->cns", encoded_instances, encoded_prototypes
            )
            evidence = torch.logsumexp(similarities, dim=-1) - math.log(
                prototypes.shape[-2]
            )
            result = self._ccer_v2_pool_evidence(
                evidence.unsqueeze(0), encoded_prototypes
            )
            logits.append(result[0].squeeze(0))
            routes.append(result[1].squeeze(0))
            weights.append(result[2].squeeze(0))
            features.append(result[3].squeeze(0))
        return (
            torch.stack(logits),
            torch.stack(routes),
            torch.stack(weights),
            torch.stack(features),
        )

    def _dr_ccer_scan_routes(
        self,
        cell_margin: torch.Tensor,
        agree_margin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Absolute, fractional, dense, bottom-tail, and agreement scan routes.

        Routes are means over the cell axis. The returned counts let the caller
        mask duplicate ``k`` values that collapse for small bags (e.g. a 1%
        fraction and Top-1 both selecting the same single cell).
        """
        route_values: list[torch.Tensor] = []
        route_counts: list[int] = []
        n = cell_margin.shape[-1]
        for k in self.dr_ccer_abs_topks:
            count = min(n, int(k))
            route_values.append(cell_margin.topk(count, dim=-1).values.mean(dim=-1))
            route_counts.append(count)
        for frac in self.dr_ccer_frac_topks:
            count = min(n, max(1, int(math.ceil(frac * n))))
            route_values.append(cell_margin.topk(count, dim=-1).values.mean(dim=-1))
            route_counts.append(count)
        route_values.append(cell_margin.mean(dim=-1))
        route_counts.append(n)
        if self.dr_ccer_use_bottom_tail:
            count = min(n, 4)
            route_values.append(
                cell_margin.topk(count, dim=-1, largest=False).values.mean(dim=-1)
            )
            route_counts.append(count)
        route_values.append(agree_margin.mean(dim=-1))
        route_counts.append(n)
        return (
            torch.stack(route_values, dim=-1),
            torch.as_tensor(route_counts, device=cell_margin.device),
        )

    def _dr_ccer_logits(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query_instances: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Donor-resolved evidence expert plus a reliability-gated margin mixture.

        Works on both the batched (leading episode dim) and single-episode
        paths via ``...`` einsum. The mixture operates on the class margin, so
        at ``g ~ 0`` the exact v30 prediction (up to the softmax-shift that CE
        is invariant to) is preserved. ``base_logits`` are the v30 logits
        computed before this branch.
        """
        slots = context["slots"]
        if slots.ndim == 5:
            centers = slots[:, :, :, 0, :].float()  # [E,B,S,D]
        elif slots.ndim == 4:
            centers = slots[:, :, 0, :].float()  # [B,S,D]
        else:
            raise ValueError("DR-CCER context slots must be 4D or 5D.")
        query_instances = query_instances.float()
        labels = context_labels.long()

        # ---- per-donor class prototypes (donor axis retained) ----
        encoded_centers = F.normalize(
            self.dr_ccer_donor_encoder(centers), dim=-1
        )
        donor_proto = F.normalize(
            torch.logsumexp(encoded_centers, dim=-2), dim=-1
        )  # [..., B, H]

        # ---- per-cell similarity to every donor ----
        encoded_instances = F.normalize(
            self.dr_ccer_instance_encoder(query_instances), dim=-1
        )  # [..., Q, N, H]
        scale = self.dr_ccer_similarity_log_scale.exp().clamp(0.05, 20.0)
        donor_sim = scale * torch.einsum(
            "...qnh, ...bh -> ...qnb", encoded_instances, donor_proto
        )  # [..., Q, N, B]

        # ---- per-class masked donor statistics ----
        class_valid = F.one_hot(labels, num_classes=self.num_classes).to(
            donor_sim.dtype
        )  # [..., B, C]
        valid_count = class_valid.sum(dim=-2).clamp_min(1.0)  # [..., C]
        # Expand to [..., 1, 1, B, C] so it broadcasts with [..., Q, N, B, C].
        class_valid_exp = class_valid.view(
            *class_valid.shape[:-2], 1, 1, *class_valid.shape[-2:]
        )
        valid = class_valid_exp == 1
        ev = donor_sim.unsqueeze(-1).expand(
            donor_sim.shape + (self.num_classes,)
        )
        # Materialize the broadcast mask to the full evidence shape so every
        # downstream boolean reduction is unambiguous.
        valid_b = valid.expand(ev.shape)
        ev = torch.where(
            valid_b, ev, torch.full_like(ev, -1e6)
        )  # [..., Q, N, B, C], invalid donors at the floor

        donor_k = min(
            self.dr_ccer_donor_topk, max(1, int(valid_count.max().item()))
        )
        topk_values = ev.topk(donor_k, dim=-2).values  # [..., Q, N, k, C]
        valid_topk = topk_values > -1e5
        topk_mean = (
            (topk_values * valid_topk.to(topk_values.dtype)).sum(dim=-2)
            / valid_topk.sum(dim=-2).clamp_min(1.0)
        )  # [..., Q, N, C] -- repeatable class evidence
        max_ev = ev.max(dim=-2).values  # [..., Q, N, C] -- upper envelope
        # Invalid donors sit at -1e6 (< 0), so `ev > 0` already selects only
        # valid donors for the agreement statistic -- no mask broadcast needed.
        # valid_count is [..., C]; reshape to [..., 1, 1, C] to divide [..., Q, N, C].
        per_cell = valid_count.unsqueeze(-2).unsqueeze(-2)
        agreement = (
            (ev > 0.0).to(donor_sim.dtype).sum(dim=-2) / per_cell
        )  # [..., Q, N, C]
        squared_dev = torch.where(
            valid_b,
            (ev - topk_mean.unsqueeze(-2)) ** 2,
            torch.zeros_like(ev),
        )
        dispersion = torch.sqrt(
            squared_dev.sum(dim=-2) / per_cell.clamp_min(1.0)
        )  # [..., Q, N, C]

        def contrast(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[..., 1] - tensor[..., 0]

        cell_margin = contrast(topk_mean)  # [..., Q, N]
        agree_margin = contrast(agreement)  # [..., Q, N]
        disp_feat = dispersion[..., 0] + dispersion[..., 1]  # [..., Q, N]

        # Support-derived null standardization: divide the margin by the donor
        # dispersion so a noisy support set does not inflate raw margins.
        null_scale = disp_feat.mean(dim=-1, keepdim=True).clamp_min(1e-3)
        cell_margin_n = cell_margin / null_scale

        routes, route_counts = self._dr_ccer_scan_routes(
            cell_margin_n, agree_margin
        )  # [..., Q, R], [R]

        # Mask duplicate-route counts (small bags collapse several routes).
        unique_route = torch.ones_like(routes, dtype=torch.bool)
        seen: dict[int, int] = {}
        for route_index, count in enumerate(route_counts.tolist()):
            if count in seen:
                unique_route[..., route_index] = False
            else:
                seen[count] = route_index

        log_n = math.log(float(cell_margin.shape[-1]))
        log_n = torch.full(
            routes.shape[:-1], log_n, device=routes.device, dtype=routes.dtype
        )  # [..., Q]
        route_spread = routes.amax(dim=-1) - routes.amin(dim=-1)  # [..., Q]
        mean_disp = disp_feat.mean(dim=-1)  # [..., Q]
        # Support separation (per episode, broadcast over queries).
        class_mean_proto = torch.einsum(
            "...bh, ...bc -> ...ch", donor_proto, class_valid
        ) / valid_count.unsqueeze(-1)  # [..., C, H]
        support_sep = (
            class_mean_proto[..., 1, :] - class_mean_proto[..., 0, :]
        ).norm(dim=-1)  # [..., ]
        support_sep_q = support_sep.unsqueeze(-1).expand(
            support_sep.shape + (routes.shape[-2],)
        )  # [..., Q]

        route_features = torch.cat(
            (
                routes,
                route_spread.unsqueeze(-1),
                mean_disp.unsqueeze(-1),
                log_n.unsqueeze(-1),
            ),
            dim=-1,
        )
        route_logits = self.dr_ccer_route_router(route_features)
        route_logits = torch.where(
            unique_route,
            route_logits,
            torch.full_like(route_logits, -1e6),
        )
        soft_weights = torch.softmax(route_logits.float(), dim=-1).to(routes.dtype)
        # Distribute the route floor over unmasked routes only, so duplicate
        # (masked) routes never receive floor mass.
        unmasked = unique_route.to(routes.dtype)
        unmasked_count = unmasked.sum(dim=-1, keepdim=True).clamp_min(1.0)
        floor_mass = unmasked * (self.dr_ccer_route_floor / unmasked_count)
        route_weights = (
            (1.0 - self.dr_ccer_route_floor) * soft_weights + floor_mass
        )
        routed = (routes * route_weights).sum(dim=-1)  # [..., Q]

        # ---- expert margin ----
        head_scale = self.dr_ccer_expert_log_scale.exp().clamp(1e-2, 10.0)
        expert_margin = head_scale * self.dr_ccer_output_head(
            routed.unsqueeze(-1)
        ).squeeze(-1)  # [..., Q]

        # ---- reliability gate ----
        mean_abs_margin = cell_margin_n.abs().mean(dim=-1)  # [..., Q]
        gate_features = torch.stack(
            (
                routed.abs(),
                route_spread,
                mean_disp,
                log_n,
                support_sep_q,
                mean_abs_margin,
            ),
            dim=-1,
        )
        gate_logit = self.dr_ccer_reliability_router(gate_features).squeeze(-1)
        gate = torch.sigmoid(gate_logit)  # [..., Q]

        # ---- standardized convex mixture on the margin ----
        base_margin = base_logits[..., 1] - base_logits[..., 0]  # [..., Q]
        with torch.no_grad():
            mu_b = base_margin.mean(dim=-1)
            sd_b = base_margin.std(dim=-1, unbiased=False).clamp_min(1e-4)
            mu_e = expert_margin.mean(dim=-1)
            sd_e = expert_margin.std(dim=-1, unbiased=False).clamp_min(1e-4)
        expert_scaled = (
            (expert_margin - mu_e.unsqueeze(-1))
            / sd_e.unsqueeze(-1)
            * sd_b.unsqueeze(-1)
            + mu_b.unsqueeze(-1)
        )
        final_margin = (1.0 - gate) * base_margin + gate * expert_scaled
        final_logits = torch.stack(
            (-final_margin / 2.0, final_margin / 2.0), dim=-1
        )

        auxiliary = {
            "dr_ccer_expert_logits": torch.stack(
                (-expert_margin / 2.0, expert_margin / 2.0), dim=-1
            ),
            "dr_ccer_expert_margin": expert_margin,
            "dr_ccer_gate": gate,
            "dr_ccer_routed": routed,
            "dr_ccer_routes": routes,
            "dr_ccer_route_weights": route_weights,
            "dr_ccer_agree": agree_margin.mean(dim=-1),
            "dr_ccer_disp": mean_disp,
            "dr_ccer_log_n": log_n,
            "dr_ccer_support_sep": support_sep_q,
        }
        return final_logits, auxiliary

    def _fuse_evidence(
        self,
        global_shape_logits: torch.Tensor,
        population_logits: torch.Tensor,
        rare_logits: torch.Tensor,
        population_scale: torch.Tensor,
        rare_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        evidence = torch.stack(
            (global_shape_logits, population_logits, rare_logits), dim=-1
        )
        pair_products = torch.stack(
            (
                evidence[..., 0] * evidence[..., 1],
                evidence[..., 0] * evidence[..., 2],
                evidence[..., 1] * evidence[..., 2],
            ),
            dim=-1,
        )
        pair_differences = torch.stack(
            (
                (evidence[..., 0] - evidence[..., 1]).abs(),
                (evidence[..., 0] - evidence[..., 2]).abs(),
                (evidence[..., 1] - evidence[..., 2]).abs(),
            ),
            dim=-1,
        )
        interaction_features = torch.cat(
            (evidence, pair_products, pair_differences), dim=-1
        )
        interaction = self.fusion_scorer(interaction_features).squeeze(-1)
        fusion_scale = torch.sigmoid(self.fusion_residual_logit)
        logits = (
            global_shape_logits
            + population_scale * population_logits
            + rare_scale * rare_logits
            + fusion_scale * interaction
        )
        return logits, fusion_scale

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
                dual_coefficients = RidgeResidualMetaClassifier._solve_ridge_system(
                    design32 @ design32.T, targets32, ridge_lambda.float()
                )
                coefficients = design32.T @ dual_coefficients
            else:
                coefficients = RidgeResidualMetaClassifier._solve_ridge_system(
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
                dual_coefficients = RidgeResidualMetaClassifier._solve_ridge_system(
                    design32 @ design32.transpose(1, 2),
                    targets32,
                    ridge_lambda.float(),
                )
                coefficients = design32.transpose(1, 2) @ dual_coefficients
            else:
                coefficients = RidgeResidualMetaClassifier._solve_ridge_system(
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

    def _all_structured_tokens_batched(
        self, representation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Batched (`[episodes, bags, tokens, token_dim]`) equivalent of
        `_all_structured_tokens`. Kept as one shared helper because the 4D
        batched path (`forward_batched`, `_class_memories_batched`) duplicates
        this token-stacking logic in several places; drifting one copy and
        not the others is exactly how the cls token was first missed here."""
        slots = representation["slots"]
        flat_slots = slots.reshape(slots.shape[0], slots.shape[1], -1, slots.shape[-1])
        parts = [
            representation["global_summary"].unsqueeze(2),
            flat_slots,
            representation["tails"],
        ]
        if self.include_cls_token:
            parts.append(representation["cls_token"].unsqueeze(2))
        for stat_name in self.raw_stat_tokens:
            parts.append(representation[f"stat_{stat_name}"].unsqueeze(2))
        return torch.cat(parts, dim=2)

    def _class_memories_batched(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        context_tokens = self._all_structured_tokens_batched(context)
        if self.project_structured_tokens:
            context_tokens = self._projected_bag_tokens(context_tokens).unsqueeze(2)
        elif self.mean_pool_structured_tokens:
            context_tokens = context_tokens.mean(dim=2, keepdim=True)
        episodes, context_count, tokens_per_bag, _ = context_tokens.shape
        flat_tokens = context_tokens.reshape(
            episodes, context_count * tokens_per_bag, self.token_dim
        )
        encoded = self.memory_input_projection(self.memory_input_norm(flat_tokens))
        memories = []
        for class_index in range(self.num_classes):
            valid_bags = context_labels == class_index
            valid_tokens = (
                valid_bags.unsqueeze(-1)
                .expand(-1, -1, tokens_per_bag)
                .reshape(episodes, -1)
            )
            seeds = self.memory_seeds.unsqueeze(0).expand(episodes, -1, -1)
            attended, _ = self.memory_cross_attention(
                seeds,
                encoded,
                encoded,
                key_padding_mask=~valid_tokens,
                need_weights=False,
            )
            memory = self.memory_norm(seeds + attended)
            memories.append(self.memory_encoder(memory))
        return torch.stack(memories, dim=1)

    def _population_memory_logits_batched(
        self,
        query: dict[str, torch.Tensor],
        class_memories: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_slots = query["slots"]
        if self.project_structured_tokens:
            all_tokens = self._all_structured_tokens_batched(query)
            query_tokens = self._projected_bag_tokens(all_tokens).unsqueeze(2)
        elif self.mean_pool_structured_tokens:
            all_tokens = self._all_structured_tokens_batched(query)
            query_tokens = all_tokens.mean(dim=2, keepdim=True)
        else:
            query_tokens = raw_slots.reshape(
                raw_slots.shape[0], raw_slots.shape[1], -1, raw_slots.shape[-1]
            )
        episodes, queries, slots, _ = query_tokens.shape
        encoded_query = self.slot_input_projection(self.slot_input_norm(query_tokens))
        importance_logits = self.slot_importance(query_tokens).squeeze(-1)
        token_weights = F.softmax(
            importance_logits.float() / self.routing_temperature,
            dim=-1,
        ).to(query_tokens.dtype)
        flat_query = encoded_query.reshape(episodes * queries, slots, -1)
        class_logits = []
        for class_index in range(self.num_classes):
            memory = class_memories[:, class_index].repeat_interleave(queries, dim=0)
            attended, _ = self.population_cross_attention(
                flat_query, memory, memory, need_weights=False
            )
            attended = attended.reshape(episodes, queries, slots, -1)
            relation_features = torch.cat(
                (
                    encoded_query,
                    attended,
                    encoded_query - attended,
                    encoded_query * attended,
                ),
                dim=-1,
            )
            relation = self.slot_relation_scorer(relation_features).squeeze(-1)
            class_logits.append((relation * token_weights).sum(dim=-1))
        return torch.stack(class_logits, dim=-1), token_weights

    def _rare_instance_logits_batched(
        self,
        query_instances: torch.Tensor,
        class_memories: torch.Tensor,
        query_cell_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.instance_input_projection(
            self.instance_input_norm(query_instances)
        )
        encoded32 = F.normalize(encoded.float(), dim=-1)
        memory32 = F.normalize(class_memories.float(), dim=-1)
        scale = self.rare_similarity_log_scale.exp().clamp(0.1, 50.0)
        similarities = scale.float() * torch.einsum(
            "eqnd,ecmd->eqcnm", encoded32, memory32
        )
        evidence = torch.logsumexp(similarities, dim=-1) - math.log(
            self.class_memory_tokens
        )
        if query_cell_mask is not None:
            if query_cell_mask.shape != query_instances.shape[:3]:
                raise ValueError(
                    "query_cell_mask must be [episodes, queries, instances]."
                )
            # evidence is [E, Q, C, n]; broadcast the cell mask over classes.
            cell_mask_expanded = query_cell_mask.unsqueeze(-2).expand_as(evidence)
            evidence = evidence.masked_fill(~cell_mask_expanded, float("-inf"))
            valid_query = query_cell_mask.sum(dim=-1).clamp_min(1)  # [E, Q]
        else:
            cell_mask_expanded = None
            valid_query = None
        fraction_scores = []
        counts: list[torch.Tensor] = []
        for fraction in self.rare_evidence_fractions:
            if valid_query is None:
                count = min(
                    query_instances.shape[2],
                    max(1, int(math.ceil(fraction * query_instances.shape[2]))),
                )
                values, index = evidence.topk(count, dim=-1)
                fraction_scores.append(values.mean(dim=-1))
                counts.append(
                    torch.full(
                        query_instances.shape[:2],
                        count,
                        device=query_instances.device,
                        dtype=torch.long,
                    )
                )
            else:
                # Per-query count (matches the list path's top-fraction mean),
                # realized with a uniform topk over the max count plus a mask.
                count_query = torch.minimum(
                    torch.clamp(
                        torch.ceil(fraction * valid_query).long(), min=1
                    ),
                    valid_query.long(),
                )
                k = min(query_instances.shape[2], int(count_query.max().item()))
                values, index = evidence.topk(k, dim=-1)
                keep = (
                    torch.arange(k, device=evidence.device)
                    < count_query.unsqueeze(-1).unsqueeze(-1)
                )
                masked = values.masked_fill(~keep, 0.0)
                # [E, Q, 1] denominator so [E, Q, C] / [E, Q, 1] -> [E, Q, C].
                denom = count_query.unsqueeze(-1)
                fraction_scores.append(masked.sum(dim=-1) / denom)
                counts.append(count_query)
        stacked_scores = torch.stack(fraction_scores, dim=-1)
        logits = self.rare_evidence_head(
            stacked_scores.to(query_instances.dtype)
        ).squeeze(-1)
        rare_counts = torch.stack(counts, dim=-1).expand(
            query_instances.shape[0],
            query_instances.shape[1],
            len(self.rare_evidence_fractions),
        )
        return logits, stacked_scores, rare_counts

    def _instance_attention_mil_logits_batched(
        self,
        query_instances: torch.Tensor,
        class_memories: torch.Tensor,
    ) -> torch.Tensor:
        """Nonlinear, task-adaptive instance-attention MIL bag scoring (4D).

        query_instances [E, Q, n, d], class_memories [E, C, T, h] -> [E, Q, C].

        For each class c, relevance is a nonlinear MLP over (instance embedding,
        class-memory context) -- not a plain cosine -- so which instances matter
        is learned per task. Soft-attention pooling gives the bag embedding and
        the max-attention instance embedding gives the MIL "any-positive" bias:
        a single strongly matching instance is enough to raise that class logit.
        """
        episodes, queries, num_instances, _ = query_instances.shape
        hidden = self.mil_hidden_dim
        num_classes = class_memories.shape[1]
        h = self.mil_instance_encoder(query_instances)          # [E,Q,n,H]
        m = class_memories.mean(dim=-2)                         # [E,C,H]
        h_e = h.unsqueeze(-2)                                   # [E,Q,n,1,H]
        m_e = m.unsqueeze(1).unsqueeze(2)                       # [E,1,1,C,H]
        pair = torch.cat(
            (
                h_e.expand(episodes, queries, num_instances, num_classes, hidden),
                m_e.expand(episodes, queries, num_instances, num_classes, hidden),
            ),
            dim=-1,
        )                                                       # [E,Q,n,C,2H]
        relevance = self.mil_relevance_mlp(pair).squeeze(-1)    # [E,Q,n,C]
        scale = self.mil_attention_log_scale.exp().clamp(0.1, 50.0)
        attention = F.softmax(
            scale.float() * relevance.float(), dim=-2
        ).to(h.dtype)                                           # over n -> [E,Q,n,C]
        z_soft = torch.einsum("eqnc,eqnh->eqch", attention, h)  # [E,Q,C,H]
        max_index = attention.max(dim=-2).indices               # [E,Q,C]
        z_max = torch.gather(
            h,
            -2,
            max_index.unsqueeze(-1).expand(
                episodes, queries, num_classes, hidden
            ),
        )                                                       # [E,Q,C,H]
        z = torch.cat((z_soft, z_max), dim=-1)                  # [E,Q,C,2H]
        return self.mil_score_head(z).squeeze(-1)               # [E,Q,C]

    def _instance_attention_mil_logits(
        self,
        query_instances: Sequence[torch.Tensor],
        class_memories: torch.Tensor,
    ) -> torch.Tensor:
        """Per-bag list-path equivalent (variable instance counts)."""
        hidden = self.mil_hidden_dim
        m = class_memories.mean(dim=-2)                         # [C,H]
        num_classes = m.shape[0]
        per_bag: list[torch.Tensor] = []
        for bag in query_instances:
            h = self.mil_instance_encoder(bag)                  # [n,H]
            num_instances = h.shape[0]
            h_e = h.unsqueeze(1)                                # [n,1,H]
            m_e = m.unsqueeze(0)                                # [1,C,H]
            pair = torch.cat(
                (
                    h_e.expand(num_instances, num_classes, hidden),
                    m_e.expand(num_instances, num_classes, hidden),
                ),
                dim=-1,
            )                                                   # [n,C,2H]
            relevance = self.mil_relevance_mlp(pair).squeeze(-1)  # [n,C]
            scale = self.mil_attention_log_scale.exp().clamp(0.1, 50.0)
            attention = F.softmax(
                scale.float() * relevance.float(), dim=0
            ).to(h.dtype)                                       # over n -> [n,C]
            z_soft = torch.einsum("nc,nh->ch", attention, h)    # [C,H]
            max_index = attention.max(dim=0).indices            # [C]
            z_max = h[max_index]                                # [C,H]
            z = torch.cat((z_soft, z_max), dim=-1)              # [C,2H]
            per_bag.append(self.mil_score_head(z).squeeze(-1))  # [C]
        return torch.stack(per_bag)                             # [Q,C]

    def forward_batched(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query: dict[str, torch.Tensor],
        query_instances: torch.Tensor,
        query_cell_mask: torch.Tensor | None = None,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context_global_tokens = context["global_summary"]
        query_global_tokens = query["global_summary"]
        if self.project_structured_tokens:
            context_global_tokens = self._projected_bag_tokens(
                self._all_structured_tokens_batched(context)
            )
            query_global_tokens = self._projected_bag_tokens(
                self._all_structured_tokens_batched(query)
            )
        elif self.mean_pool_structured_tokens:
            context_global_tokens = self._all_structured_tokens_batched(
                context
            ).mean(dim=2)
            query_global_tokens = self._all_structured_tokens_batched(
                query
            ).mean(dim=2)
        global_shape_logits, global_shape_auxiliary = (
            self.global_shape_classifier.forward_batched(
                context_global_tokens,
                context_labels,
                query_global_tokens,
                return_auxiliary=True,
            )
        )
        if self.typed_bag_preserving_branch:
            # include_cls_token and typed_bag_preserving_branch are mutually
            # exclusive (enforced in __init__), so the helper's cls-token
            # branch is always a no-op here.
            context_typed_bag_tokens = self._typed_bag_tokens(
                self._all_structured_tokens_batched(context)
            )
            query_typed_bag_tokens = self._typed_bag_tokens(
                self._all_structured_tokens_batched(query)
            )
            typed_bag_logits, typed_bag_auxiliary = (
                self.typed_bag_classifier.forward_batched(
                    context_typed_bag_tokens,
                    context_labels,
                    query_typed_bag_tokens,
                    return_auxiliary=True,
                )
            )
            typed_bag_residual_scale = torch.sigmoid(self.typed_bag_residual_logit)
        class_memories = self._class_memories_batched(context, context_labels)
        population_attention_logits, population_weights = (
            self._population_memory_logits_batched(query, class_memories)
        )
        abundance_ridge_logits = self._abundance_ridge_logits_batched(
            context["slot_metadata"],
            context_labels,
            query["slot_metadata"],
        )
        abundance_ridge_scale = self.abundance_ridge_log_scale.exp().clamp(
            0.1, 100.0
        )
        covariance_ridge_logits = self._abundance_ridge_logits_batched(
            context["covariance_sketch"],
            context_labels,
            query["covariance_sketch"],
            ridge_lambda=self.covariance_ridge_log_lambda,
            dual=True,
        )
        covariance_ridge_scale = self.covariance_ridge_log_scale.exp().clamp(
            0.1, 100.0
        )
        covariance_logits = covariance_ridge_scale * covariance_ridge_logits
        covariance_residual_scale = torch.sigmoid(self.covariance_residual_logit)
        if self.covariance_relation_enabled:
            if self.covariance_relation_granularity == "slot":
                covariance_relation_logits, covariance_relation_class_separation = (
                    self._slot_covariance_relation_scores(
                        context["slot_covariance_sketch"],
                        context["slot_covariance_reliability"], context_labels,
                        query["slot_covariance_sketch"],
                        query["slot_covariance_reliability"],
                    )
                )
            elif self.covariance_relation_granularity == "subspace":
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
                covariance_relation_logits, covariance_relation_class_separation = (
                    self._covariance_relation_scores(
                        context["covariance_sketch"],
                        context_labels,
                        query["covariance_sketch"],
                    )
                )
        else:
            covariance_relation_logits = covariance_logits.new_zeros(
                covariance_logits.shape
            )
            covariance_relation_class_separation = covariance_logits.new_zeros(
                covariance_logits.shape[0]
            )
        population_attention_scale = torch.sigmoid(
            self.population_attention_residual_logit
        )
        population_logits = (
            abundance_ridge_scale * abundance_ridge_logits
            + population_attention_scale * population_attention_logits
        )
        tail_logits, rare_fraction_scores, rare_counts = (
            self._rare_instance_logits_batched(
                query_instances,
                class_memories,
                query_cell_mask=query_cell_mask,
            )
        )
        if self.ccer_temperatures:
            ccer_logits, ccer_route_scores, ccer_presence, ccer_route_weights = (
                self._ccer_instance_logits_batched(query_instances, class_memories)
            )
            ccer_residual_scale = torch.sigmoid(self.ccer_residual_logit)
        if self.ccer_v2_topks:
            (
                ccer_v2_logits,
                ccer_v2_route_scores,
                ccer_v2_route_weights,
                ccer_v2_router_features,
            ) = self._ccer_v2_logits_batched(
                context, context_labels, query_instances
            )
            ccer_v2_residual_scale = torch.sigmoid(
                self.ccer_v2_residual_logit
            )
        population_scale = self._floored_residual_scale(
            self.population_residual_logit,
            self.minimum_population_residual_scale,
        )
        tail_scale = self._floored_residual_scale(
            self.tail_residual_logit,
            self.minimum_tail_residual_scale,
        )
        logits, fusion_scale = self._fuse_evidence(
            global_shape_logits,
            population_logits,
            tail_logits,
            population_scale,
            tail_scale,
        )
        logits = logits + covariance_residual_scale * covariance_logits
        if self.ccer_temperatures:
            logits = logits + ccer_residual_scale * ccer_logits
        if self.ccer_v2_topks:
            logits = logits + ccer_v2_residual_scale * ccer_v2_logits
        if self.covariance_relation_enabled and not self.covariance_relation_diagnostic_only:
            logits = logits + (
                self.covariance_relation_residual_scale * covariance_relation_logits
            )
        if self.typed_bag_preserving_branch:
            logits = logits + typed_bag_residual_scale * typed_bag_logits
        if self.use_instance_attention_mil:
            mil_logits = self._instance_attention_mil_logits_batched(
                query_instances, class_memories
            )
            logits = logits + torch.sigmoid(self.mil_residual_logit) * mil_logits
        if self.dr_ccer_enabled:
            # The reliability-gated mixture replaces additive residual fusion:
            # final = (1-g)*v30 + g*expert with a closed gate at init.
            logits, dr_auxiliary = self._dr_ccer_logits(
                context, context_labels, query_instances, base_logits=logits
            )
        if not return_auxiliary:
            return logits
        episodes = context_labels.shape[0]
        return logits, {
            **global_shape_auxiliary,
            "global_shape_logits": global_shape_logits,
            "population_logits": population_logits,
            "abundance_ridge_logits": abundance_ridge_logits,
            "covariance_logits": covariance_logits,
            "covariance_ridge_logits": covariance_ridge_logits,
            "covariance_ridge_scale": covariance_ridge_scale.expand(episodes),
            "covariance_residual_scale": covariance_residual_scale.expand(episodes),
            "covariance_relation_enabled": torch.tensor(
                self.covariance_relation_enabled, device=logits.device
            ).expand(episodes),
            "covariance_relation_logits": covariance_relation_logits,
            "covariance_relation_class_separation": (
                covariance_relation_class_separation
            ),
            "covariance_relation_residual_scale": torch.full(
                (episodes,),
                self.covariance_relation_residual_scale,
                device=logits.device,
                dtype=logits.dtype,
            ),
            "abundance_ridge_scale": abundance_ridge_scale.expand(episodes),
            "population_attention_logits": population_attention_logits,
            "population_attention_residual_scale": (
                population_attention_scale.expand(episodes)
            ),
            "tail_logits": tail_logits,
            "population_slot_weights": population_weights,
            "tail_weights": torch.softmax(rare_fraction_scores, dim=-1),
            "rare_fraction_scores": rare_fraction_scores,
            "rare_counts": rare_counts,
            "class_memories": class_memories,
            "population_residual_scale": population_scale.expand(episodes),
            "tail_residual_scale": tail_scale.expand(episodes),
            "fusion_residual_scale": fusion_scale.expand(episodes),
            **(
                {
                    "ccer_logits": ccer_logits,
                    "ccer_route_scores": ccer_route_scores,
                    "ccer_presence": ccer_presence,
                    "ccer_route_weights": ccer_route_weights,
                    "ccer_residual_scale": ccer_residual_scale.expand(episodes),
                }
                if self.ccer_temperatures
                else {}
            ),
            **(
                {
                    "ccer_v2_logits": ccer_v2_logits,
                    "ccer_v2_route_scores": ccer_v2_route_scores,
                    "ccer_v2_route_weights": ccer_v2_route_weights,
                    "ccer_v2_router_features": ccer_v2_router_features,
                    "ccer_v2_residual_scale": ccer_v2_residual_scale.expand(
                        episodes
                    ),
                }
                if self.ccer_v2_topks
                else {}
            ),
            **(
                {
                    "typed_bag_logits": typed_bag_logits,
                    "typed_bag_residual_scale": typed_bag_residual_scale.expand(
                        episodes
                    ),
                }
                if self.typed_bag_preserving_branch
                else {}
            ),
            **(dr_auxiliary if self.dr_ccer_enabled else {}),
        }

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
        if context_labels.shape != (context["global_summary"].shape[0],):
            raise ValueError("context_labels must have shape [context].")
        if torch.any((context_labels < 0) | (context_labels >= self.num_classes)):
            raise ValueError(f"Context labels must be in [0, {self.num_classes - 1}].")
        class_counts = torch.bincount(context_labels.long(), minlength=self.num_classes)
        if torch.any(class_counts == 0):
            raise ValueError("Every class must occur in the context set.")
        if context["slots"].shape[1:3] != query["slots"].shape[1:3]:
            raise ValueError("Context and query slot counts must match.")
        if context["tails"].shape[1] != query["tails"].shape[1]:
            raise ValueError("Context and query tail counts must match.")

        context_global_tokens = context["global_summary"]
        query_global_tokens = query["global_summary"]
        if self.project_structured_tokens:
            context_global_tokens = self._projected_bag_tokens(
                self._all_structured_tokens(context)
            )
            query_global_tokens = self._projected_bag_tokens(
                self._all_structured_tokens(query)
            )
        elif self.mean_pool_structured_tokens:
            context_global_tokens = self._all_structured_tokens(context).mean(dim=1)
            query_global_tokens = self._all_structured_tokens(query).mean(dim=1)
        global_shape_logits, global_shape_auxiliary = self.global_shape_classifier(
            context_global_tokens,
            context_labels,
            query_global_tokens,
            return_auxiliary=True,
        )
        if self.typed_bag_preserving_branch:
            context_typed_bag_tokens = self._typed_bag_tokens(
                self._all_structured_tokens(context)
            )
            query_typed_bag_tokens = self._typed_bag_tokens(
                self._all_structured_tokens(query)
            )
            typed_bag_logits, typed_bag_auxiliary = self.typed_bag_classifier(
                context_typed_bag_tokens,
                context_labels,
                query_typed_bag_tokens,
                return_auxiliary=True,
            )
            typed_bag_residual_scale = torch.sigmoid(self.typed_bag_residual_logit)
        class_memories = self._class_memories(context, context_labels)
        population_attention_logits, population_weights = (
            self._population_memory_logits(query, class_memories)
        )
        abundance_ridge_logits = self._abundance_ridge_logits(
            context["slot_metadata"],
            context_labels,
            query["slot_metadata"],
        )
        abundance_ridge_scale = self.abundance_ridge_log_scale.exp().clamp(
            0.1, 100.0
        )
        covariance_ridge_logits = self._abundance_ridge_logits(
            context["covariance_sketch"],
            context_labels,
            query["covariance_sketch"],
            ridge_lambda=self.covariance_ridge_log_lambda,
            dual=True,
        )
        covariance_ridge_scale = self.covariance_ridge_log_scale.exp().clamp(
            0.1, 100.0
        )
        covariance_logits = covariance_ridge_scale * covariance_ridge_logits
        covariance_residual_scale = torch.sigmoid(self.covariance_residual_logit)
        if self.covariance_relation_enabled:
            if self.covariance_relation_granularity == "slot":
                covariance_relation_logits, covariance_relation_class_separation = (
                    self._slot_covariance_relation_scores(
                        context["slot_covariance_sketch"],
                        context["slot_covariance_reliability"], context_labels,
                        query["slot_covariance_sketch"],
                        query["slot_covariance_reliability"],
                    )
                )
            elif self.covariance_relation_granularity == "subspace":
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
                covariance_relation_logits, covariance_relation_class_separation = (
                    self._covariance_relation_scores(
                        context["covariance_sketch"],
                        context_labels,
                        query["covariance_sketch"],
                    )
                )
        else:
            covariance_relation_logits = covariance_logits.new_zeros(
                covariance_logits.shape
            )
            covariance_relation_class_separation = covariance_logits.new_zeros(())
        population_attention_scale = torch.sigmoid(
            self.population_attention_residual_logit
        )
        population_logits = (
            abundance_ridge_scale * abundance_ridge_logits
            + population_attention_scale * population_attention_logits
        )
        tail_logits, rare_fraction_scores, rare_counts = self._rare_instance_logits(
            query_instances, class_memories
        )
        if self.ccer_temperatures:
            ccer_logits, ccer_route_scores, ccer_presence, ccer_route_weights = (
                self._ccer_instance_logits(query_instances, class_memories)
            )
            ccer_residual_scale = torch.sigmoid(self.ccer_residual_logit)
        if self.ccer_v2_topks:
            (
                ccer_v2_logits,
                ccer_v2_route_scores,
                ccer_v2_route_weights,
                ccer_v2_router_features,
            ) = self._ccer_v2_logits(
                context, context_labels, query_instances
            )
            ccer_v2_residual_scale = torch.sigmoid(
                self.ccer_v2_residual_logit
            )
        population_scale = self._floored_residual_scale(
            self.population_residual_logit,
            self.minimum_population_residual_scale,
        )
        tail_scale = self._floored_residual_scale(
            self.tail_residual_logit,
            self.minimum_tail_residual_scale,
        )
        logits, fusion_scale = self._fuse_evidence(
            global_shape_logits,
            population_logits,
            tail_logits,
            population_scale,
            tail_scale,
        )
        logits = logits + covariance_residual_scale * covariance_logits
        if self.ccer_temperatures:
            logits = logits + ccer_residual_scale * ccer_logits
        if self.ccer_v2_topks:
            logits = logits + ccer_v2_residual_scale * ccer_v2_logits
        if self.covariance_relation_enabled and not self.covariance_relation_diagnostic_only:
            logits = logits + (
                self.covariance_relation_residual_scale * covariance_relation_logits
            )
        if self.typed_bag_preserving_branch:
            logits = logits + typed_bag_residual_scale * typed_bag_logits
        if self.use_instance_attention_mil:
            mil_logits = self._instance_attention_mil_logits(
                query_instances, class_memories
            )
            logits = logits + torch.sigmoid(self.mil_residual_logit) * mil_logits
        if self.dr_ccer_enabled:
            logits, dr_auxiliary = self._dr_ccer_logits(
                context, context_labels, query_instances, base_logits=logits
            )
        if not return_auxiliary:
            return logits
        return logits, {
            **global_shape_auxiliary,
            "global_shape_logits": global_shape_logits,
            "population_logits": population_logits,
            "abundance_ridge_logits": abundance_ridge_logits,
            "covariance_logits": covariance_logits,
            "covariance_ridge_logits": covariance_ridge_logits,
            "covariance_ridge_scale": covariance_ridge_scale,
            "covariance_residual_scale": covariance_residual_scale,
            "covariance_relation_enabled": torch.tensor(
                self.covariance_relation_enabled, device=logits.device
            ),
            "covariance_relation_logits": covariance_relation_logits,
            "covariance_relation_class_separation": (
                covariance_relation_class_separation
            ),
            "covariance_relation_residual_scale": torch.as_tensor(
                self.covariance_relation_residual_scale,
                device=logits.device,
                dtype=logits.dtype,
            ),
            "abundance_ridge_scale": abundance_ridge_scale,
            "population_attention_logits": population_attention_logits,
            "population_attention_residual_scale": population_attention_scale,
            "tail_logits": tail_logits,
            "population_slot_weights": population_weights,
            "tail_weights": torch.softmax(rare_fraction_scores, dim=-1),
            "rare_fraction_scores": rare_fraction_scores,
            "rare_counts": rare_counts,
            "class_memories": class_memories,
            "population_residual_scale": population_scale,
            "tail_residual_scale": tail_scale,
            "fusion_residual_scale": fusion_scale,
            **(
                {
                    "ccer_logits": ccer_logits,
                    "ccer_route_scores": ccer_route_scores,
                    "ccer_presence": ccer_presence,
                    "ccer_route_weights": ccer_route_weights,
                    "ccer_residual_scale": ccer_residual_scale,
                }
                if self.ccer_temperatures
                else {}
            ),
            **(
                {
                    "ccer_v2_logits": ccer_v2_logits,
                    "ccer_v2_route_scores": ccer_v2_route_scores,
                    "ccer_v2_route_weights": ccer_v2_route_weights,
                    "ccer_v2_router_features": ccer_v2_router_features,
                    "ccer_v2_residual_scale": ccer_v2_residual_scale,
                }
                if self.ccer_v2_topks
                else {}
            ),
            **(
                {
                    "typed_bag_logits": typed_bag_logits,
                    "typed_bag_residual_scale": typed_bag_residual_scale,
                }
                if self.typed_bag_preserving_branch
                else {}
            ),
            **(dr_auxiliary if self.dr_ccer_enabled else {}),
        }


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
        use_instance_attention_mil: bool = False,
        mil_hidden_dim: int | None = None,
        aggregator_covariance_sketch_dim: int | None = None,
        aggregator_covariance_mode: str = "covariance",
        aggregator_covariance_shrinkage: float = 0.0,
        meta_hidden_dim: int = 256,
        meta_num_heads: int = 8,
        meta_num_set_layers: int = 1,
        meta_relation_hidden_dim: int = 256,
        meta_ridge_dim: int = 64,
        meta_ridge_lambda: float = 1.0,
        meta_ridge_logit_scale: float = 5.0,
        meta_attention_residual_scale: float = 0.1,
        meta_population_residual_scale: float = 0.25,
        meta_tail_residual_scale: float = 0.10,
        meta_minimum_population_residual_scale: float = 0.10,
        meta_minimum_tail_residual_scale: float = 0.05,
        meta_routing_temperature: float = 0.5,
        meta_class_memory_tokens: int = 8,
        meta_rare_evidence_fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
        meta_ccer_temperatures: Sequence[float] = (),
        meta_ccer_residual_scale: float = 0.10,
        meta_ccer_presence_temperature: float = 0.5,
        meta_ccer_v2_topks: Sequence[int] = (),
        meta_ccer_v2_route_floor: float = 0.30,
        meta_ccer_v2_residual_scale: float = 0.10,
        meta_dr_ccer_enabled: bool = False,
        meta_dr_ccer_abs_topks: Sequence[int] = (1, 4, 16),
        meta_dr_ccer_frac_topks: Sequence[float] = (0.01, 0.05),
        meta_dr_ccer_donor_topk: int = 3,
        meta_dr_ccer_use_bottom_tail: bool = True,
        meta_dr_ccer_expert_hidden: int = 128,
        meta_dr_ccer_route_floor: float = 0.05,
        meta_dr_ccer_max_donors: int = 32,
        meta_fusion_residual_scale: float = 0.10,
        meta_covariance_ridge_logit_scale: float = 2.0,
        meta_covariance_residual_scale: float = 0.25,
        mean_pool_structured_tokens: bool = False,
        project_structured_tokens: bool = False,
        projection_bottleneck_dim: int | None = None,
        projection_residual_mean: bool = False,
        typed_bag_preserving_branch: bool = False,
        typed_bag_bottleneck_dim: int | None = None,
        meta_typed_bag_residual_scale: float = 0.02,
        covariance_relation: dict[str, object] | None = None,
        cls_token_pooling: bool = False,
        cls_token_heads: int = 4,
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
            covariance_mode=aggregator_covariance_mode,
            covariance_shrinkage=aggregator_covariance_shrinkage,
            include_cls_token=cls_token_pooling,
            cls_token_heads=cls_token_heads,
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
            hidden_dim=meta_hidden_dim,
            num_heads=meta_num_heads,
            num_set_layers=meta_num_set_layers,
            relation_hidden_dim=meta_relation_hidden_dim,
            ridge_dim=meta_ridge_dim,
            ridge_lambda=meta_ridge_lambda,
            ridge_logit_scale=meta_ridge_logit_scale,
            attention_residual_scale=meta_attention_residual_scale,
            population_residual_scale=meta_population_residual_scale,
            tail_residual_scale=meta_tail_residual_scale,
            minimum_population_residual_scale=(meta_minimum_population_residual_scale),
            minimum_tail_residual_scale=meta_minimum_tail_residual_scale,
            routing_temperature=meta_routing_temperature,
            class_memory_tokens=meta_class_memory_tokens,
            rare_evidence_fractions=meta_rare_evidence_fractions,
            ccer_temperatures=meta_ccer_temperatures,
            ccer_residual_scale=meta_ccer_residual_scale,
            ccer_presence_temperature=meta_ccer_presence_temperature,
            ccer_v2_topks=meta_ccer_v2_topks,
            ccer_v2_route_floor=meta_ccer_v2_route_floor,
            ccer_v2_residual_scale=meta_ccer_v2_residual_scale,
            dr_ccer_enabled=meta_dr_ccer_enabled,
            dr_ccer_abs_topks=meta_dr_ccer_abs_topks,
            dr_ccer_frac_topks=meta_dr_ccer_frac_topks,
            dr_ccer_donor_topk=meta_dr_ccer_donor_topk,
            dr_ccer_use_bottom_tail=meta_dr_ccer_use_bottom_tail,
            dr_ccer_expert_hidden=meta_dr_ccer_expert_hidden,
            dr_ccer_route_floor=meta_dr_ccer_route_floor,
            dr_ccer_max_donors=meta_dr_ccer_max_donors,
            fusion_residual_scale=meta_fusion_residual_scale,
            covariance_ridge_logit_scale=meta_covariance_ridge_logit_scale,
            covariance_residual_scale=meta_covariance_residual_scale,
            covariance_relation=covariance_relation,
            mean_pool_structured_tokens=mean_pool_structured_tokens,
            project_structured_tokens=project_structured_tokens,
            structured_tokens_per_bag=structured_tokens_per_bag,
            projection_bottleneck_dim=projection_bottleneck_dim,
            projection_residual_mean=projection_residual_mean,
            typed_bag_preserving_branch=typed_bag_preserving_branch,
            typed_bag_bottleneck_dim=typed_bag_bottleneck_dim,
            typed_bag_num_slots=aggregator_num_slots,
            typed_bag_num_tail_fractions=len(tuple(aggregator_tail_fractions)),
            typed_bag_residual_scale=meta_typed_bag_residual_scale,
            include_cls_token=cls_token_pooling,
            raw_stat_tokens=raw_stat_tokens,
            use_instance_attention_mil=use_instance_attention_mil,
            mil_hidden_dim=mil_hidden_dim,
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
            32 if meta_dr_ccer_enabled
            else 31 if tuple(meta_ccer_v2_topks)
            else 26 if cls_token_pooling
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

    def apply_dr_ccer_stage(self, stage: str) -> None:
        """Freeze parameters outside the DR-CCER scope for a staged training run.

        Stage A trains only the donor-resolved expert (v30 and the reliability
        gate frozen). Stage B trains only the reliability gate (expert frozen).
        ``None``/"both" leaves every DR-CCER parameter trainable and freezes
        nothing beyond what the optimizer selects.
        """
        if stage not in ("A", "B", "both", None):
            raise ValueError(f"Unknown dr_ccer_stage: {stage}")
        if not self.meta_classifier.dr_ccer_enabled:
            return
        for name, parameter in self.named_parameters():
            is_dr = name.startswith("meta_classifier.dr_ccer_")
            is_gate = "dr_ccer_reliability_router" in name
            if not is_dr:
                parameter.requires_grad = False
                continue
            if stage == "A" and is_gate:
                parameter.requires_grad = False
            elif stage == "B" and not is_gate:
                parameter.requires_grad = False

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
        if self.aggregator.bag_representation in ("poolz", "poolz_l2"):
            # Per-episode context-pool stats, broadcast back over that episode's bags.
            episode_mean, episode_std = self.aggregator._context_pool_stats_batched(
                x, is_context, cell_mask=cell_mask
            )
            pool_mean = (
                episode_mean[:, None, :]
                .expand(episodes, num_bags, input_dim)
                .reshape(episodes * num_bags, 1, input_dim)
            )
            pool_std = (
                episode_std[:, None, :]
                .expand(episodes, num_bags, input_dim)
                .reshape(episodes * num_bags, 1, input_dim)
            )
        else:
            pool_mean = pool_std = None
        classification_flat, global_summary, centered_delta = self.aggregator._bag_view(
            flat_x, pool_mean, pool_std, cell_mask=flat_cell_mask
        )
        covariance_sketch = self.aggregator._covariance_sketch(
            centered_delta, count=flat_valid_count
        )
        if self.aggregator.raw_stat_tokens:
            raw_stats = self.aggregator._raw_stat_tokens(
                flat_x, cell_mask=flat_cell_mask
            )
        else:
            raw_stats = None
        classification_x = classification_flat.reshape_as(x)
        anchors = torch.stack(
            [
                self.aggregator._context_anchors(
                    list(classification_x[episode].unbind(0)),
                    is_context[episode],
                    (
                        list(cell_mask[episode].unbind(0))
                        if cell_mask is not None
                        else None
                    ),
                )
                for episode in range(episodes)
            ]
        )
        per_bag_anchors = (
            anchors[:, None]
            .expand(-1, num_bags, -1, -1)
            .reshape(episodes * num_bags, anchors.shape[1], anchors.shape[2])
        )
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
        classification_bags = [
            self.aggregator._bag_view(bag, pool_mean, pool_std)[0]
            for bag in normalized_bags
        ]
        query_instances = [
            classification_bags[index] for index in query_index.detach().cpu().tolist()
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
        return logits, {
            "bag_tokens": representation["global_summary"],
            "slot_tokens": representation["slots"],
            "tail_tokens": representation["tails"],
            "slot_metadata": representation["slot_metadata"],
            "context_mask": is_context,
            "aggregator": aggregator_auxiliary,
            **auxiliary,
        }
