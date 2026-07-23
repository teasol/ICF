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
        population_dim = (
            num_slots * (state_dim + 2)
            + len(fractions) * state_dim
        )
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
            bag.ndim != 2
            or bag.shape[0] == 0
            or bag.shape[1] != self.input_dim
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
        positions = torch.linspace(
            0,
            bag.shape[0] - 1,
            self.context_samples_per_bag,
            device=bag.device,
        ).round().long()
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
            raise ValueError("Context does not contain enough cells for population slots.")
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
            dispersion = (
                assignment * (1.0 - similarity).to(assignment.dtype)
            ).sum(dim=0) / mass

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
    ) -> None:
        # Deliberately initialize nn.Module directly: anchor construction is
        # inherited, while the v13 compressed projections are not retained.
        nn.Module.__init__(self)
        if min(input_dim, num_slots, context_samples_per_bag) < 1:
            raise ValueError("Structured aggregator dimensions must be positive.")
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
        self.min_tail_instances = int(min_tail_instances)
        self.slot_statistic_count = 3
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
            raise ValueError("Context does not contain enough cells for population slots.")
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
        positions = torch.linspace(
            0,
            density_limit - 1,
            self.num_density_slots,
            device=order.device,
        ).round().long()
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
            diversity = torch.minimum(
                diversity, (1.0 - similarity).clamp_min(0.0)
            )
        rare = torch.stack(selected)
        return torch.cat((density, rare), dim=0).to(candidates.dtype)

    def _forward_dense(
        self,
        instances: torch.Tensor,
        anchors: torch.Tensor,
        return_auxiliary: bool,
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
        dispersion = (
            assignment * (1.0 - similarity).to(assignment.dtype)
        ).sum(dim=1) / mass
        metadata = torch.stack((proportion.log(), dispersion), dim=-1)

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
        spread_features = torch.cat(
            (expanded_anchors, slot_std, metadata), dim=-1
        )
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
        rare_token = rare_state + residual_scale * self.rare_slot_encoder(
            rare_features
        )
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
                tail_tokens.append(
                    self.shared_tail_encoder(deviation.float()).mean(dim=1)
                )
            selected_counts.append(count)

        representation = {
            "mean": instances.mean(dim=1),
            "slots": slot_tokens,
            "tails": torch.stack(tail_tokens, dim=1),
            "slot_metadata": metadata,
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
            "tail_counts": torch.tensor(
                selected_counts, device=anchors.device
            ).expand(num_bags, -1),
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
        bags = self._normalize_bags(instances)
        context_mask = torch.as_tensor(
            context_mask,
            device=bags[0].device,
            dtype=torch.bool,
        ).flatten()
        if context_mask.numel() != len(bags) or not torch.any(context_mask):
            raise ValueError("context_mask must identify at least one context bag.")
        anchors = self._context_anchors(bags, context_mask)
        if isinstance(instances, torch.Tensor):
            return self._forward_dense(instances, anchors, return_auxiliary)

        mean_tokens: list[torch.Tensor] = []
        slot_tokens: list[torch.Tensor] = []
        tail_tokens: list[torch.Tensor] = []
        proportions: list[torch.Tensor] = []
        dispersions: list[torch.Tensor] = []
        slot_means: list[torch.Tensor] = []
        selected_counts: list[list[int]] = []
        for bag in bags:
            mean_tokens.append(bag.mean(dim=0))
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
            dispersion = (
                assignment * (1.0 - similarity).to(assignment.dtype)
            ).sum(dim=0) / mass
            metadata = torch.stack((proportion.log(), dispersion), dim=-1)

            rare_states: list[torch.Tensor] = []
            rare_count = min(
                bag.shape[0],
                max(1, int(math.ceil(self.slot_rare_fraction * bag.shape[0]))),
            )
            slot_distance = difference.float().square().mean(dim=-1)
            for slot_index in range(self.num_slots):
                rare_score = assignment[:, slot_index].float() * slot_distance[:, slot_index]
                values, index = rare_score.topk(rare_count)
                weights = torch.softmax(values, dim=0).to(bag.dtype)
                rare_states.append((weights.unsqueeze(-1) * bag[index]).sum(dim=0))
            rare_state = torch.stack(rare_states)

            center_features = torch.cat(
                (anchors, slot_mean - anchors, metadata), dim=-1
            )
            spread_features = torch.cat(
                (anchors, slot_std, metadata), dim=-1
            )
            rare_features = torch.cat(
                (anchors, rare_state - anchors, metadata), dim=-1
            )
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
            "mean": torch.stack(mean_tokens),
            "slots": torch.stack(slot_tokens),
            "tails": torch.stack(tail_tokens),
            "slot_metadata": torch.stack(
                [
                    torch.stack((proportion.log(), dispersion), dim=-1)
                    for proportion, dispersion in zip(proportions, dispersions)
                ]
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
            raise ValueError(
                f"query_tokens must have shape [query, {self.token_dim}]."
            )
        if context_labels.ndim != 1 or context_labels.shape[0] != context_tokens.shape[0]:
            raise ValueError("context_labels must have shape [context].")
        if context_tokens.shape[0] == 0 or query_tokens.shape[0] == 0:
            raise ValueError("Context and query sets must both be non-empty.")
        if torch.any((context_labels < 0) | (context_labels >= self.num_classes)):
            raise ValueError(
                f"Context labels must be in [0, {self.num_classes - 1}]."
            )
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
            "cross_attention_entropy": torch.stack(
                class_attention_entropy, dim=-1
            ),
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
        counts = F.one_hot(
            context_labels.long(), num_classes=self.num_classes
        ).sum(dim=1)
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
            class_mean = (
                class_context * valid.unsqueeze(-1)
            ).sum(dim=1) / denominator
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
            class_entropies.append(
                -(probability * probability.log()).sum(dim=-1)
            )
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
        if (
            ridge_dim < 1
            or ridge_lambda <= 0
            or ridge_logit_scale <= 0
        ):
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
        identity = torch.eye(
            gram.shape[-1], device=gram.device, dtype=gram.dtype
        )
        if gram.ndim == 3:
            identity = identity.expand(gram.shape[0], -1, -1)
        system = gram + ridge_lambda.float() * identity
        if not torch.isfinite(system).all():
            raise RuntimeError("The ridge system contains NaN or Inf values.")

        diagonal_scale = gram.diagonal(dim1=-2, dim2=-1).abs().mean(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
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
        targets = F.one_hot(
            context_labels.long(), num_classes=self.num_classes
        ).float()
        sample_weight = class_counts.float().reciprocal()[context_labels.long()]
        ridge_lambda = self.ridge_log_lambda.exp().clamp(1e-4, 1e4)
        with torch.autocast(device_type=context_tokens.device.type, enabled=False):
            # Eliminate the unregularized intercept by weighted centering. This
            # is algebraically equivalent to solving the augmented system, but
            # leaves a strictly positive-definite feature block. Forming one
            # joint system with an unregularized bias made rare CUDA episodes
            # singular and could produce non-finite gradients in solve backward.
            total_weight = sample_weight.sum().clamp_min(1e-12)
            feature_mean = (
                sample_weight.unsqueeze(-1) * context32
            ).sum(dim=0, keepdim=True) / total_weight
            target_mean = (
                sample_weight.unsqueeze(-1) * targets
            ).sum(dim=0, keepdim=True) / total_weight
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
        targets = F.one_hot(
            context_labels.long(), num_classes=self.num_classes
        ).float()
        sample_weight = class_counts.float().reciprocal().gather(
            1, context_labels.long()
        )
        ridge_lambda = self.ridge_log_lambda.exp().clamp(1e-4, 1e4)
        with torch.autocast(device_type=context_tokens.device.type, enabled=False):
            total_weight = sample_weight.sum(dim=1, keepdim=True).clamp_min(1e-12)
            feature_mean = (
                sample_weight.unsqueeze(-1) * context32
            ).sum(dim=1, keepdim=True) / total_weight.unsqueeze(-1)
            target_mean = (
                sample_weight.unsqueeze(-1) * targets
            ).sum(dim=1, keepdim=True) / total_weight.unsqueeze(-1)
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
        logits = (
            ridge_scale * ridge_logits
            + residual_scale * attention_logits
        )
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
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.routing_temperature = float(routing_temperature)
        self.class_memory_tokens = int(class_memory_tokens)
        self.rare_evidence_fractions = rare_fractions
        self.minimum_population_residual_scale = float(
            minimum_population_residual_scale
        )
        self.minimum_tail_residual_scale = float(minimum_tail_residual_scale)
        self.mean_classifier = RidgeResidualMetaClassifier(
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
        fusion_logit = math.log(
            fusion_residual_scale / (1.0 - fusion_residual_scale)
        )
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

    def _validate_representation(
        self,
        representation: dict[str, torch.Tensor],
        name: str,
    ) -> None:
        if set(representation) != {"mean", "slots", "tails", "slot_metadata"}:
            raise ValueError(
                f"{name} must contain mean, slots, tails, and slot_metadata."
            )
        mean = representation["mean"]
        slots = representation["slots"]
        tails = representation["tails"]
        metadata = representation["slot_metadata"]
        if mean.ndim != 2 or mean.shape[-1] != self.token_dim:
            raise ValueError(f"{name} mean tokens have an invalid shape.")
        if slots.ndim != 4 or slots.shape[0] != mean.shape[0] or slots.shape[-1] != self.token_dim:
            raise ValueError(f"{name} slot tokens have an invalid shape.")
        if tails.ndim != 3 or tails.shape[0] != mean.shape[0] or tails.shape[-1] != self.token_dim:
            raise ValueError(f"{name} tail tokens have an invalid shape.")
        if metadata.shape != slots.shape[:2] + (2,):
            raise ValueError(f"{name} slot metadata have an invalid shape.")

    @staticmethod
    def _flatten_slot_tokens(representation: dict[str, torch.Tensor]) -> torch.Tensor:
        slots = representation["slots"]
        return slots.reshape(slots.shape[0], -1, slots.shape[-1])

    def _all_structured_tokens(
        self, representation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return torch.cat(
            (
                representation["mean"].unsqueeze(1),
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
        encoded_query = self.slot_input_projection(
            self.slot_input_norm(query_tokens)
        )
        importance_logits = self.slot_importance(query_tokens).squeeze(-1)
        token_weights = F.softmax(
            importance_logits.float() / self.routing_temperature,
            dim=-1,
        ).to(query_tokens.dtype)

        class_logits: list[torch.Tensor] = []
        for class_index in range(self.num_classes):
            memory = class_memories[class_index].unsqueeze(0).expand(
                encoded_query.shape[0], -1, -1
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
                fraction_scores.append(
                    evidence.topk(count, dim=-1).values.mean(dim=-1)
                )
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
        mean_logits: torch.Tensor,
        population_logits: torch.Tensor,
        rare_logits: torch.Tensor,
        population_scale: torch.Tensor,
        rare_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        evidence = torch.stack(
            (mean_logits, population_logits, rare_logits), dim=-1
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
            mean_logits
            + population_scale * population_logits
            + rare_scale * rare_logits
            + fusion_scale * interaction
        )
        return logits, fusion_scale

    def _class_memories_batched(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
    ) -> torch.Tensor:
        slots = context["slots"]
        flat_slots = slots.reshape(
            slots.shape[0], slots.shape[1], -1, slots.shape[-1]
        )
        context_tokens = torch.cat(
            (
                context["mean"].unsqueeze(2), flat_slots, context["tails"]
            ),
            dim=2,
        )
        episodes, context_count, tokens_per_bag, _ = context_tokens.shape
        flat_tokens = context_tokens.reshape(
            episodes, context_count * tokens_per_bag, self.token_dim
        )
        encoded = self.memory_input_projection(
            self.memory_input_norm(flat_tokens)
        )
        memories = []
        for class_index in range(self.num_classes):
            valid_bags = context_labels == class_index
            valid_tokens = valid_bags.unsqueeze(-1).expand(
                -1, -1, tokens_per_bag
            ).reshape(episodes, -1)
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
        encoded_query = self.slot_input_projection(
            self.slot_input_norm(query_tokens)
        )
        importance_logits = self.slot_importance(query_tokens).squeeze(-1)
        token_weights = F.softmax(
            importance_logits.float() / self.routing_temperature,
            dim=-1,
        ).to(query_tokens.dtype)
        flat_query = encoded_query.reshape(episodes * queries, slots, -1)
        class_logits = []
        for class_index in range(self.num_classes):
            memory = class_memories[:, class_index].repeat_interleave(
                queries, dim=0
            )
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
            relation = self.slot_relation_scorer(
                relation_features
            ).squeeze(-1)
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
            fraction_scores.append(
                evidence.topk(count, dim=-1).values.mean(dim=-1)
            )
            counts.append(count)
        stacked_scores = torch.stack(fraction_scores, dim=-1)
        logits = self.rare_evidence_head(
            stacked_scores.to(query_instances.dtype)
        ).squeeze(-1)
        rare_counts = torch.tensor(
            counts, device=query_instances.device
        ).expand(query_instances.shape[0], query_instances.shape[1], -1)
        return logits, stacked_scores, rare_counts

    def forward_batched(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        query: dict[str, torch.Tensor],
        query_instances: torch.Tensor,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean_logits, mean_auxiliary = self.mean_classifier.forward_batched(
            context["mean"],
            context_labels,
            query["mean"],
            return_auxiliary=True,
        )
        class_memories = self._class_memories_batched(context, context_labels)
        population_logits, population_weights = (
            self._population_memory_logits_batched(query, class_memories)
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
            mean_logits,
            population_logits,
            tail_logits,
            population_scale,
            tail_scale,
        )
        if not return_auxiliary:
            return logits
        episodes = context_labels.shape[0]
        return logits, {
            **mean_auxiliary,
            "mean_logits": mean_logits,
            "population_logits": population_logits,
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
        if context_labels.shape != (context["mean"].shape[0],):
            raise ValueError("context_labels must have shape [context].")
        if torch.any((context_labels < 0) | (context_labels >= self.num_classes)):
            raise ValueError(
                f"Context labels must be in [0, {self.num_classes - 1}]."
            )
        class_counts = torch.bincount(
            context_labels.long(), minlength=self.num_classes
        )
        if torch.any(class_counts == 0):
            raise ValueError("Every class must occur in the context set.")
        if context["slots"].shape[1:3] != query["slots"].shape[1:3]:
            raise ValueError("Context and query slot counts must match.")
        if context["tails"].shape[1] != query["tails"].shape[1]:
            raise ValueError("Context and query tail counts must match.")

        mean_logits, mean_auxiliary = self.mean_classifier(
            context["mean"], context_labels, query["mean"], return_auxiliary=True
        )
        class_memories = self._class_memories(context, context_labels)
        population_logits, population_weights = self._population_memory_logits(
            query, class_memories
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
            mean_logits,
            population_logits,
            tail_logits,
            population_scale,
            tail_scale,
        )
        if not return_auxiliary:
            return logits
        return logits, {
            **mean_auxiliary,
            "mean_logits": mean_logits,
            "population_logits": population_logits,
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

    architecture_version = 18

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
            minimum_population_residual_scale=(
                meta_minimum_population_residual_scale
            ),
            minimum_tail_residual_scale=meta_minimum_tail_residual_scale,
            routing_temperature=meta_routing_temperature,
            class_memory_tokens=meta_class_memory_tokens,
            rare_evidence_fractions=meta_rare_evidence_fractions,
            fusion_residual_scale=meta_fusion_residual_scale,
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
            raise ValueError("Batched x must be [episodes, bags, instances, input_dim].")
        episodes, num_bags, num_instances, input_dim = x.shape
        if input_dim != self.input_dim or y.shape != (episodes, num_bags):
            raise ValueError("Batched x/y shapes are incompatible.")
        if mask_index.ndim != 2 or mask_index.shape[0] != episodes:
            raise ValueError("Batched mask_index must be [episodes, queries].")
        if torch.any((mask_index < 0) | (mask_index >= num_bags)):
            raise IndexError("mask_index contains an out-of-range bag index.")

        is_context = torch.ones(
            episodes, num_bags, dtype=torch.bool, device=x.device
        )
        is_context.scatter_(1, mask_index.long(), False)
        anchors = torch.stack([
            self.aggregator._context_anchors(
                list(x[episode].unbind(0)), is_context[episode]
            )
            for episode in range(episodes)
        ])
        per_bag_anchors = anchors[:, None].expand(
            -1, num_bags, -1, -1
        ).reshape(episodes * num_bags, anchors.shape[1], anchors.shape[2])
        flat_representation = self.aggregator._forward_dense(
            x.reshape(episodes * num_bags, num_instances, input_dim),
            per_bag_anchors,
            return_auxiliary=False,
        )
        representation = {
            name: tokens.reshape(episodes, num_bags, *tokens.shape[1:])
            for name, tokens in flat_representation.items()
        }

        context_count = num_bags - mask_index.shape[1]
        context_index = torch.nonzero(
            is_context, as_tuple=False
        )[:, 1].reshape(episodes, context_count)

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
        query_instances = gather_bags(x, mask_index.long())
        return self.meta_classifier.forward_batched(
            context=context,
            context_labels=context_labels,
            query=query,
            query_instances=query_instances,
            return_auxiliary=return_auxiliary,
        )

    def forward(
        self,
        x: torch.Tensor | Sequence[torch.Tensor],
        y: torch.Tensor,
        mask_index: torch.Tensor | Sequence[int] | int,
        return_auxiliary: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(x, torch.Tensor):
            if x.ndim != 3:
                raise ValueError("Dense x must have shape [bags, instances, input_dim].")
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
        if isinstance(normalized_bags, torch.Tensor):
            query_instances: torch.Tensor | list[torch.Tensor] = normalized_bags[
                query_index
            ]
        else:
            query_instances = [
                normalized_bags[index] for index in query_index.detach().cpu().tolist()
            ]
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
            "bag_tokens": representation["mean"],
            "slot_tokens": representation["slots"],
            "tail_tokens": representation["tails"],
            "slot_metadata": representation["slot_metadata"],
            "context_mask": is_context,
            "aggregator": aggregator_auxiliary,
            **auxiliary,
        }
