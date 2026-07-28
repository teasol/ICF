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
        min_tail_instances: int = 1,
        bag_centered_representation: bool = True,
        global_summary: str = "centered_spread",
        use_raw_mean_branch: bool = False,
        covariance_sketch_dim: int | None = None,
        covariance_mode: str = "covariance",
        covariance_shrinkage: float = 0.0,
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
        self.min_tail_instances = int(min_tail_instances)
        self.slot_statistic_count = 3
        self.bag_centered_representation = bool(bag_centered_representation)
        self.global_summary = str(global_summary)
        self.use_raw_mean_branch = bool(use_raw_mean_branch)
        self.covariance_sketch_dim = int(covariance_sketch_dim)
        self.covariance_mode = str(covariance_mode)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.slot_covariance_descriptor = "correlation"
        self.emit_covariance_matrix = False
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

    def _bag_view(
        self, bag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return classification instances, summary, and centered deltas."""
        bag_mean = bag.mean(dim=-2, keepdim=True)
        centered_delta = bag - bag_mean
        global_spread = torch.sqrt(
            centered_delta.float().square().mean(dim=-2) + 1e-6
        )
        if self.bag_centered_representation:
            classification_instances = F.normalize(
                centered_delta.float(), dim=-1, eps=1e-6
            ).to(bag.dtype)
            summary = global_spread.to(bag.dtype)
        else:
            classification_instances = bag
            summary = bag_mean.squeeze(-2)
        return classification_instances, summary, centered_delta

    def _covariance_sketch(self, centered_delta: torch.Tensor) -> torch.Tensor:
        """Return a configurable translation-invariant second-moment feature."""
        values = centered_delta.float()
        projected = values @ self._covariance_projection.float()
        covariance = projected.transpose(-1, -2) @ projected
        covariance = covariance / projected.shape[-2]
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure local covariance inside aligned population slots."""
        measurement_dim = min(16, self.covariance_sketch_dim)
        values = centered_delta.float()
        bag_rms = values.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
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


    def _population_candidates(self, bag: torch.Tensor) -> torch.Tensor:
        """Build stable, order-invariant soft candidates in centered space."""
        normalized = F.normalize(bag.float(), dim=-1, eps=1e-6)
        candidate_count = min(self.context_samples_per_bag, bag.shape[0])
        directions = self._candidate_directions[:candidate_count].float()
        scores = normalized @ directions.T
        weights = torch.softmax(scores.mul(10.0), dim=0)
        candidates = weights.T @ normalized
        return F.normalize(candidates, dim=-1, eps=1e-6).to(bag.dtype)

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


    def _forward_dense(
        self,
        instances: torch.Tensor,
        anchors: torch.Tensor,
        return_auxiliary: bool,
        global_summary: torch.Tensor | None = None,
        covariance_sketch: torch.Tensor | None = None,
        centered_delta: torch.Tensor | None = None,
    ) -> (
        dict[str, torch.Tensor]
        | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
    ):
        """Vectorized equivalent of the per-bag path for synthetic episodes."""
        num_bags, num_instances, _ = instances.shape
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
        mass = assignment.sum(dim=1).clamp_min(1e-6)
        proportion = mass / num_instances
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
            centered_delta = instances - instances.mean(dim=-2, keepdim=True)
        slot_covariance_sketch, slot_covariance_reliability = (
            self._slot_covariance_sketch(assignment, centered_delta)
        )

        rare_count = min(
            num_instances,
            max(1, int(math.ceil(self.slot_rare_fraction * num_instances))),
        )
        slot_distance = difference.float().square().mean(dim=-1)
        rare_score = assignment.float() * slot_distance
        values, index = rare_score.transpose(1, 2).topk(rare_count, dim=-1)
        weights = torch.softmax(values, dim=-1).to(instances.dtype)
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
        tail_tokens: list[torch.Tensor] = []
        selected_counts: list[int] = []
        for fraction in self.tail_fractions:
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
                lse_weights = torch.softmax(encoded_tail * 2.0, dim=1)
                tail_tokens.append((lse_weights * encoded_tail).sum(dim=1))
            selected_counts.append(count)



        if global_summary is None:
            _, global_summary, _ = self._bag_view(instances)
        if covariance_sketch is None:
            covariance_sketch = self._covariance_sketch(instances - instances.mean(dim=-2, keepdim=True))
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
        }
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
            "instance_counts": torch.full(
                (num_bags,), num_instances, device=anchors.device
            ),
            "tail_counts": torch.tensor(selected_counts, device=anchors.device).expand(
                num_bags, -1
            ),
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
        prepared = [self._bag_view(bag) for bag in raw_bags]
        bags = [item[0] for item in prepared]
        global_summaries = [item[1] for item in prepared]
        centered_deltas = [item[2] for item in prepared]
        covariance_sketches = [self._covariance_sketch(delta) for delta in centered_deltas]
        context_mask = torch.as_tensor(
            context_mask,
            device=bags[0].device,
            dtype=torch.bool,
        ).flatten()
        if context_mask.numel() != len(bags) or not torch.any(context_mask):
            raise ValueError("context_mask must identify at least one context bag.")
        anchors = self._context_anchors(bags, context_mask)
        if isinstance(instances, torch.Tensor):
            result = self._forward_dense(
                torch.stack(bags),
                anchors,
                return_auxiliary,
                global_summary=torch.stack(global_summaries),
                covariance_sketch=torch.stack(covariance_sketches),
                centered_delta=torch.stack(centered_deltas),
            )
            if return_auxiliary:
                representation, auxiliary = result
                return representation, auxiliary
            return result

        slot_tokens: list[torch.Tensor] = []
        tail_tokens: list[torch.Tensor] = []
        slot_covariance_sketches: list[torch.Tensor] = []
        slot_covariance_reliabilities: list[torch.Tensor] = []
        proportions: list[torch.Tensor] = []
        dispersions: list[torch.Tensor] = []
        slot_means: list[torch.Tensor] = []
        selected_counts: list[list[int]] = []
        for bag, centered_delta in zip(bags, centered_deltas):
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
        }
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
        for attempt in range(6):
            candidate = system
            if attempt:
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
        fusion_residual_scale: float = 0.10,
        covariance_ridge_logit_scale: float = 2.0,
        covariance_residual_scale: float = 0.25,
        covariance_relation: dict[str, object] | None = None,
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
        if not 0 < fusion_residual_scale < 1:
            raise ValueError("fusion_residual_scale must be in (0, 1).")
        if covariance_ridge_logit_scale <= 0:
            raise ValueError("covariance_ridge_logit_scale must be positive.")
        if not 0 < covariance_residual_scale < 1:
            raise ValueError("covariance_residual_scale must be in (0, 1).")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.routing_temperature = float(routing_temperature)
        self.class_memory_tokens = int(class_memory_tokens)
        self.rare_evidence_fractions = rare_fractions
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
            "slot_covariance_reliability", "covariance_matrix",
        }
        if set(representation) != expected_keys:
            raise ValueError(f"{name} has invalid structured representation keys.")
        global_summary = representation["global_summary"]
        slots = representation["slots"]
        tails = representation["tails"]
        metadata = representation["slot_metadata"]
        covariance_sketch = representation["covariance_sketch"]
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

    @staticmethod
    def _flatten_slot_tokens(representation: dict[str, torch.Tensor]) -> torch.Tensor:
        slots = representation["slots"]
        return slots.reshape(slots.shape[0], -1, slots.shape[-1])

    def _all_structured_tokens(
        self, representation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.cat(
            (
                representation["global_summary"].unsqueeze(1),
                self._flatten_slot_tokens(representation),
                representation["tails"],
            ),
            dim=1,
        )

    def _class_memories(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        context_tokens = self._all_structured_tokens(context)
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
        query_tokens = self._flatten_slot_tokens(query)
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

    def _class_memories_batched(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        slots = context["slots"]
        flat_slots = slots.reshape(slots.shape[0], slots.shape[1], -1, slots.shape[-1])
        context_tokens = torch.cat(
            (context["global_summary"].unsqueeze(2), flat_slots, context["tails"]),
            dim=2,
        )
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
        fraction_scores = []
        counts = []
        for fraction in self.rare_evidence_fractions:
            count = min(
                query_instances.shape[2],
                max(1, int(math.ceil(fraction * query_instances.shape[2]))),
            )
            fraction_scores.append(evidence.topk(count, dim=-1).values.mean(dim=-1))
            counts.append(count)
        stacked_scores = torch.stack(fraction_scores, dim=-1)
        logits = self.rare_evidence_head(
            stacked_scores.to(query_instances.dtype)
        ).squeeze(-1)
        rare_counts = torch.tensor(counts, device=query_instances.device).expand(
            query_instances.shape[0], query_instances.shape[1], -1
        )
        return logits, stacked_scores, rare_counts

    def forward_batched(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query: dict[str, torch.Tensor],
        query_instances: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        global_shape_logits, global_shape_auxiliary = (
            self.global_shape_classifier.forward_batched(
                context["global_summary"],
                context_labels,
                query["global_summary"],
                return_auxiliary=True,
            )
        )
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
            self._rare_instance_logits_batched(query_instances, class_memories)
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
        if self.covariance_relation_enabled and not self.covariance_relation_diagnostic_only:
            logits = logits + (
                self.covariance_relation_residual_scale * covariance_relation_logits
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

        global_shape_logits, global_shape_auxiliary = self.global_shape_classifier(
            context["global_summary"],
            context_labels,
            query["global_summary"],
            return_auxiliary=True,
        )
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
        if self.covariance_relation_enabled and not self.covariance_relation_diagnostic_only:
            logits = logits + (
                self.covariance_relation_residual_scale * covariance_relation_logits
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
        }


class BaseModel(nn.Module):
    """Compose hybrid population aggregation with class-memory meta learning."""

    architecture_version = 21

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
        aggregator_min_tail_instances: int = 1,
        bag_centered_representation: bool = True,
        global_summary: str = "centered_spread",
        use_raw_mean_branch: bool = False,
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
        meta_fusion_residual_scale: float = 0.10,
        meta_covariance_ridge_logit_scale: float = 2.0,
        meta_covariance_residual_scale: float = 0.25,
        covariance_relation: dict[str, object] | None = None,
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
            min_tail_instances=aggregator_min_tail_instances,
            bag_centered_representation=bag_centered_representation,
            global_summary=global_summary,
            use_raw_mean_branch=use_raw_mean_branch,
            covariance_sketch_dim=aggregator_covariance_sketch_dim,
            covariance_mode=aggregator_covariance_mode,
            covariance_shrinkage=aggregator_covariance_shrinkage,
        )
        relation_config = dict(covariance_relation or {})
        self.aggregator.slot_covariance_descriptor = str(
            relation_config.get("descriptor", "correlation")
        )
        self.aggregator.emit_covariance_matrix = bool(
            relation_config.get("enabled", False)
            and relation_config.get("granularity") == "subspace"
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
            fusion_residual_scale=meta_fusion_residual_scale,
            covariance_ridge_logit_scale=meta_covariance_ridge_logit_scale,
            covariance_residual_scale=meta_covariance_residual_scale,
            covariance_relation=covariance_relation,
            num_classes=self.num_classes,
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
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run dense equal-shape episodes through one batched aggregator."""
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

        is_context = torch.ones(episodes, num_bags, dtype=torch.bool, device=x.device)
        is_context.scatter_(1, mask_index.long(), False)
        flat_x = x.reshape(episodes * num_bags, num_instances, input_dim)
        classification_flat, global_summary, centered_delta = self.aggregator._bag_view(flat_x)
        covariance_sketch = self.aggregator._covariance_sketch(centered_delta)
        classification_x = classification_flat.reshape_as(x)
        anchors = torch.stack(
            [
                self.aggregator._context_anchors(
                    list(classification_x[episode].unbind(0)), is_context[episode]
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
        )
        representation = {
            name: tokens.reshape(episodes, num_bags, *tokens.shape[1:])
            for name, tokens in flat_representation.items()
        }

        context_count = num_bags - mask_index.shape[1]
        context_index = torch.nonzero(is_context, as_tuple=False)[:, 1].reshape(
            episodes, context_count
        )

        def gather_bags(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            view_shape = index.shape + (1,) * (tokens.ndim - 2)
            expanded = index.reshape(view_shape).expand(index.shape + tokens.shape[2:])
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

    def extract_bag_features(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        context_mask: torch.Tensor | None = None,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """Extract internal 40-dim Bag Representation (global summary & tail evidence)."""
        if isinstance(x, torch.Tensor):
            num_bags = x.shape[0]
            device = x.device
        else:
            num_bags = len(x)
            device = x[0].device

        if chunk_size > 0 and num_bags > chunk_size:
            feature_chunks: list[torch.Tensor] = []
            for start_idx in range(0, num_bags, chunk_size):
                end_idx = min(start_idx + chunk_size, num_bags)
                chunk_x = x[start_idx:end_idx] if isinstance(x, torch.Tensor) else x[start_idx:end_idx]
                chunk_mask = context_mask[start_idx:end_idx] if context_mask is not None else None
                feat = self.extract_bag_features(chunk_x, context_mask=chunk_mask, chunk_size=0)
                feature_chunks.append(feat)
            return torch.cat(feature_chunks, dim=0)

        if context_mask is None:
            context_mask = torch.ones(num_bags, dtype=torch.bool, device=device)
        representation = self.aggregator(x, context_mask=context_mask)
        global_tokens = representation["global_summary"]
        tail_tokens = representation["tails"].mean(dim=-2) if "tails" in representation and representation["tails"].ndim == 3 else global_tokens
        if global_tokens.ndim == 2 and tail_tokens.ndim == 2 and global_tokens.shape == tail_tokens.shape:
            return torch.cat([global_tokens, tail_tokens], dim=-1)
        return global_tokens

    def retrieve_context_indices(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        y: torch.Tensor,
        mask_index: torch.Tensor | Sequence[int] | int,
        retrieval_k: int = 24,
        chunk_size: int = 32,
    ) -> tuple[torch.Tensor | Sequence[torch.Tensor], torch.Tensor, torch.Tensor]:
        if retrieval_k <= 0:
            query_index = self._normalize_mask_index(mask_index, num_bags=len(y) if y.ndim == 1 else y.shape[1], device=y.device)
            return x, y, query_index

        if isinstance(x, torch.Tensor) and x.ndim == 4:
            # 4D Batched Episode Input: [E, N, instances, features]
            E, N, num_instances, feature_dim = x.shape
            device = x.device
            
            # Flatten to [E*N, instances, feature_dim] to compute 40-dim features efficiently
            x_flat = x.view(E * N, num_instances, feature_dim)
            bag_features_flat = self.extract_bag_features(x_flat, chunk_size=chunk_size)
            bag_features = bag_features_flat.view(E, N, -1)

            res_x, res_y, res_masks = [], [], []
            for b in range(E):
                x_b, y_b = x[b], y[b]
                mask_b = mask_index[b] if isinstance(mask_index, torch.Tensor) and mask_index.ndim >= 1 else mask_index
                q_idx_b = self._normalize_mask_index(mask_b, num_bags=N, device=device)
                
                is_ctx_b = torch.ones(N, dtype=torch.bool, device=device)
                is_ctx_b[q_idx_b] = False
                ctx_indices_b = torch.nonzero(is_ctx_b, as_tuple=False).flatten()

                feat_b = bag_features[b]
                q_summary_b = feat_b[q_idx_b].mean(dim=0, keepdim=True)
                ctx_features_b = feat_b[ctx_indices_b]
                ctx_y_b = y_b[ctx_indices_b]

                sims_b = F.cosine_similarity(q_summary_b, ctx_features_b, dim=-1)
                k_per_class = max(1, retrieval_k // 2)
                selected_context_idx: list[int] = []
                observed_classes = torch.unique(ctx_y_b, sorted=True)
                for class_idx in observed_classes:
                    class_mask = (ctx_y_b == class_idx)
                    class_idxs_in_ctx = torch.nonzero(class_mask, as_tuple=False).flatten()
                    if class_idxs_in_ctx.numel() == 0:
                        continue
                    class_sims = sims_b[class_idxs_in_ctx]
                    k_for_class = min(k_per_class, class_idxs_in_ctx.numel())
                    top_k_local = torch.topk(class_sims, k=k_for_class).indices
                    selected_context_idx.extend(ctx_indices_b[class_idxs_in_ctx[top_k_local]].cpu().tolist())

                if len(selected_context_idx) < retrieval_k and ctx_indices_b.numel() > len(selected_context_idx):
                    remaining = [idx.item() for idx in ctx_indices_b if idx.item() not in selected_context_idx]
                    needed = retrieval_k - len(selected_context_idx)
                    selected_context_idx.extend(remaining[:needed])

                selected_context_tensor = torch.tensor(selected_context_idx, dtype=torch.long, device=device)
                final_x_b = torch.cat([x_b[selected_context_tensor], x_b[q_idx_b]], dim=0)
                final_y_b = torch.cat([y_b[selected_context_tensor], y_b[q_idx_b]], dim=0)
                mask_index_b = torch.tensor([len(selected_context_idx)], dtype=torch.long, device=device)

                res_x.append(final_x_b)
                res_y.append(final_y_b)
                res_masks.append(mask_index_b)

            return torch.stack(res_x), torch.stack(res_y), torch.stack(res_masks)

        num_bags = len(y)
        query_index = self._normalize_mask_index(mask_index, num_bags=num_bags, device=y.device)
        is_context = torch.ones(num_bags, dtype=torch.bool, device=y.device)
        is_context[query_index] = False
        context_indices = torch.nonzero(is_context, as_tuple=False).flatten()

        bag_features = self.extract_bag_features(x, chunk_size=chunk_size)
        query_summary = bag_features[query_index].mean(dim=0, keepdim=True)
        context_features = bag_features[context_indices]
        context_y = y[context_indices]

        sims = F.cosine_similarity(query_summary, context_features, dim=-1)

        k_per_class = max(1, retrieval_k // 2)
        selected_context_idx: list[int] = []
        observed_classes = torch.unique(context_y, sorted=True)
        for class_idx in observed_classes:
            class_mask = (context_y == class_idx)
            class_idxs_in_context = torch.nonzero(class_mask, as_tuple=False).flatten()
            if class_idxs_in_context.numel() == 0:
                continue
            class_sims = sims[class_idxs_in_context]
            k_for_class = min(k_per_class, class_idxs_in_context.numel())
            top_k_local = torch.topk(class_sims, k=k_for_class).indices
            selected_context_idx.extend(context_indices[class_idxs_in_context[top_k_local]].cpu().tolist())

        if len(selected_context_idx) < retrieval_k and context_indices.numel() > len(selected_context_idx):
            remaining = [idx.item() for idx in context_indices if idx.item() not in selected_context_idx]
            needed = retrieval_k - len(selected_context_idx)
            selected_context_idx.extend(remaining[:needed])

        selected_context_tensor = torch.tensor(selected_context_idx, dtype=torch.long, device=y.device)

        if isinstance(x, torch.Tensor):
            final_x = torch.cat([x[selected_context_tensor], x[query_index]], dim=0)
        else:
            final_x = [x[i.item()] for i in selected_context_tensor] + [x[i.item()] for i in query_index]

        final_y = torch.cat([y[selected_context_tensor], y[query_index]], dim=0)
        final_mask = torch.arange(
            len(selected_context_idx),
            len(selected_context_idx) + len(query_index),
            dtype=torch.long,
            device=y.device,
        )
        return final_x, final_y, final_mask

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        y: torch.Tensor,
        mask_index: torch.Tensor | Sequence[int] | int,
        return_auxiliary: bool = False,
        retrieval_k: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if retrieval_k > 0:
            x, y, mask_index = self.retrieve_context_indices(
                x, y, mask_index=mask_index, retrieval_k=retrieval_k
            )

        if isinstance(x, torch.Tensor) and x.ndim == 4:
            # 4D Batched Forward Mode
            E, N, num_instances, feature_dim = x.shape
            device = x.device
            res_logits, res_aux = [], []
            for b in range(E):
                mask_b = mask_index[b] if isinstance(mask_index, torch.Tensor) and mask_index.ndim >= 1 else mask_index
                res = self.forward(
                    x[b], y[b], mask_index=mask_b, return_auxiliary=return_auxiliary, retrieval_k=0
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
        classification_bags = [
            self.aggregator._bag_view(bag)[0] for bag in normalized_bags
        ]
        query_instances = [
            classification_bags[index] for index in query_index.detach().cpu().tolist()
        ]
        if isinstance(x, torch.Tensor):
            query_instances = torch.stack(query_instances)
        is_context = torch.ones(num_bags, dtype=torch.bool, device=y.device)
        is_context[query_index] = False
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
