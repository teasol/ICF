"""Mean bag aggregation with a label-equivariant set meta-classifier."""

from __future__ import annotations

from collections.abc import Sequence
import math
import os

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
        covariance_only: bool = False,
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
        # CV-only (docs SS68): emit ONLY the covariance tensors and skip the
        # slot pipeline, tails, metadata, anchors and global summary entirely.
        # Measured: 18.73 ms -> 0.40 ms floor on 8 bags x 4000 cells, i.e. ~98%
        # of the old CV-only forward was computed and discarded.
        # The dead keys are ABSENT, not zero-filled, so a stray consumer raises
        # KeyError at its own line instead of silently reading a zero.
        self.covariance_only = bool(covariance_only)
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
        if self.covariance_only:
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
        normalized = (
            instances.float()
            if self._instances_are_unit
            else F.normalize(instances.float(), dim=-1)
        )
        if anchors.ndim == 2:
            expanded_anchors = anchors.unsqueeze(0).expand(num_bags, -1, -1)
        elif anchors.ndim == 3 and anchors.shape[0] == num_bags:
            expanded_anchors = anchors
        else:
            raise ValueError("Anchors must be [slots, dim] or [bags, slots, dim].")
        similarity = self._slot_similarity(normalized, expanded_anchors.float())
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
        if self.slot_latent_dim is not None:
            # Exact variance reformulation (Var = E[X^2] - E[X]^2): the
            # [cells, slots, dim] difference tensor is never materialized.
            # slot_std[s,d]^2 = (Sum_c assignment[c,s] x[c,d]^2)/mass[s]
            #                  - slot_mean[s,d]^2.
            x_sq = instances.float().square()
            second_moment = (
                torch.einsum("bns,bnd->bsd", assignment.float(), x_sq)
                / mass.float().unsqueeze(-1)
            )
            slot_std = torch.sqrt(
                (second_moment - slot_mean.float().square()).clamp_min(0.0) + 1e-6
            ).to(instances.dtype)
        else:
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

        if self.slot_latent_dim is not None:
            # Expanded squared-distance (no [cells, slots, dim] tensor):
            # dist^2 = E_d[x^2] - 2 (x . m)/dim + E_d[m^2].
            x_mean_sq = instances.float().square().mean(dim=-1)  # [B, cells]
            m_mean_sq = slot_mean.float().square().mean(dim=-1)  # [B, slots]
            x_dot_m = (
                torch.einsum("bnd,bsd->bns", instances.float(), slot_mean.float())
                / instances.shape[-1]
            )  # [B, cells, slots]
            slot_distance = (
                x_mean_sq.unsqueeze(-1) - 2.0 * x_dot_m + m_mean_sq.unsqueeze(-2)
            ).to(instances.dtype)
        else:
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
                # Broadcast the shared count to per-bag shape so the later
                # `torch.stack(..., dim=1)` matches the cell-mask branch (which
                # appends [num_bags] count_bag tensors).
                selected_counts.append(
                    torch.full((num_bags,), count, device=anchors.device, dtype=torch.long)
                )
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
                # Broadcast the shared count to per-bag shape (see tail_fractions).
                selected_counts.append(
                    torch.full((num_bags,), count, device=anchors.device, dtype=torch.long)
                )
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
        # Streaming (docs v35 §3.3): hold ONE bag's view at a time instead of
        # every bag's. The eager `prepared` list below keeps the standardized
        # view AND the centered delta for all bags alive simultaneously -- for a
        # full-tile PathoBench episode (270 slides x ~7.5k tiles x 1536) that is
        # ~25 GB and is what capped eval at 2 workers/GPU.
        #
        # Pass 1 builds only the per-bag soft candidates (32 per bag, regardless
        # of bag size) so `_select_anchors` receives the identical pool; pass 2
        # recomputes one bag's view inside the token loop. This is EXACT -- the
        # same `_bag_view` on the same inputs -- at the cost of one extra
        # `_bag_view` per context bag.
        #
        # Kept eager for: the dense/tensor path (needs stacked tensors for
        # `_forward_dense`), `ccts_lambdas` (its scores span all bags), and
        # training (which uses the batched `_population_candidates_batched`
        # anchor path and equal-shape episodes where the copies are bounded).
        stream_bags = (
            self.stream_eval_bags
            and not isinstance(instances, torch.Tensor)
            and not self.ccts_lambdas
            and not self.training
        )
        if stream_bags:
            anchors = self._select_anchors(
                torch.cat(
                    [
                        self._population_candidates(
                            self._bag_view(raw_bags[index], pool_mean, pool_std)[0]
                        )
                        for index in range(len(raw_bags))
                        if bool(context_mask[index])
                    ],
                    dim=0,
                )
            )
            prepared = bags = centered_deltas = None
            global_summaries = covariance_sketches = None
        else:
            prepared = [self._bag_view(bag, pool_mean, pool_std) for bag in raw_bags]
            bags = [item[0] for item in prepared]
            global_summaries = [item[1] for item in prepared]
            centered_deltas = [item[2] for item in prepared]
            covariance_sketches = [
                self._covariance_sketch(delta) for delta in centered_deltas
            ]
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

        ccts_per_bag_tokens: list[list[torch.Tensor]] = [
            [] for _ in range(len(raw_bags))
        ]
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
        covariance_matrices: list[torch.Tensor] = []
        centered_delta_means: list[torch.Tensor] = []
        if stream_bags:
            # Pass 2: these are filled per bag below instead of up front.
            global_summaries = []
            covariance_sketches = []
        if self.covariance_only:
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
        for b_idx in range(len(raw_bags)):
            if stream_bags:
                bag, bag_summary, centered_delta = self._bag_view(
                    raw_bags[b_idx], pool_mean, pool_std
                )
                global_summaries.append(bag_summary)
                covariance_sketches.append(self._covariance_sketch(centered_delta))
            else:
                bag = bags[b_idx]
                centered_delta = centered_deltas[b_idx]
            covariance_matrices.append(
                self._projected_covariance_matrix(centered_delta)
                if self.emit_covariance_matrix
                else centered_delta.new_zeros((1, 1))
            )
            if return_auxiliary:
                centered_delta_means.append(centered_delta.float().mean(dim=0))
            cls_tokens.append(
                self.cls_token_pooling(bag.unsqueeze(0)).squeeze(0)
                if self.include_cls_token
                else bag.new_zeros(self.input_dim)
            )
            normalized = F.normalize(bag.float(), dim=-1)
            similarity = self._slot_similarity(normalized, anchors.float())
            assignment = torch.softmax(
                similarity / self.assignment_temperature,
                dim=-1,
            ).to(bag.dtype)
            mass = assignment.sum(dim=0).clamp_min(1e-6)
            proportion = mass / bag.shape[0]
            slot_mean = (assignment.T @ bag) / mass.unsqueeze(-1)
            if self.slot_latent_dim is not None:
                # Exact variance reformulation (no [cells, slots, dim] tensor).
                x_sq = bag.float().square()
                second_moment = (
                    (assignment.float().T @ x_sq) / mass.float().unsqueeze(-1)
                )
                slot_std = torch.sqrt(
                    (second_moment - slot_mean.float().square()).clamp_min(0.0)
                    + 1e-6
                ).to(bag.dtype)
            else:
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
            if self.slot_latent_dim is not None:
                # Expanded squared-distance (no [cells, slots, dim] tensor).
                x_mean_sq = bag.float().square().mean(dim=-1)
                m_mean_sq = slot_mean.float().square().mean(dim=-1)
                x_dot_m = (bag.float() @ slot_mean.float().T) / bag.shape[-1]
                slot_distance = (
                    x_mean_sq.unsqueeze(-1) - 2.0 * x_dot_m + m_mean_sq.unsqueeze(0)
                ).to(bag.dtype)
            else:
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
            # Accumulated inside the token loop so the streaming path never holds
            # every bag's centered delta at once (docs v35 §3.3).
            "covariance_matrix": torch.stack(covariance_matrices),
            "cls_token": torch.stack(cls_tokens),
        }
        if self.raw_stat_tokens:
            for stat_name in self.raw_stat_tokens:
                representation[f"stat_{stat_name}"] = raw_stats[
                    f"stat_{stat_name}"
                ].to(covariance_matrices[0].dtype)
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
                [len(bag) for bag in raw_bags], device=anchors.device
            ),
            "tail_counts": torch.tensor(selected_counts, device=anchors.device),
            "slot_residual_scale": torch.sigmoid(self.slot_residual_logit),
            # Accumulated in the token loop (docs v35 §3.3) so the streaming path
            # does not have to keep every bag's centered delta alive for this
            # diagnostic-only tensor.
            "centered_delta_mean": torch.stack(centered_delta_means),
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
        enable_ridge: bool = True,
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
        # Ridge ablation (docs SS65). enable_ridge=False forces ridge_logits = 0
        # so only the attention residual survives in this branch. Shape-preserving
        # and parameter-preserving (ridge_projection/log_lambda/log_scale stay in
        # the state dict) so checkpoints load strict either way; the ridge solve
        # is skipped outright, so those parameters receive no gradient and stay
        # at their initial values. Evaluate such a checkpoint with the SAME flag.
        self.force_ridge_logits_zero = not bool(enable_ridge)

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
        if self.force_ridge_logits_zero:
            return query_tokens.new_zeros(
                (*query_tokens.shape[:-1], self.num_classes)
            )
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
        if self.force_ridge_logits_zero:
            return query_tokens.new_zeros(
                (*query_tokens.shape[:-1], self.num_classes)
            )
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
        meta_enable_rare_evidence: bool = True,
        meta_enable_global_ridge: bool = True,
        meta_enable_abundance_ridge: bool = True,
        meta_enable_covariance_ridge: bool = True,
        meta_covariance_only: bool = False,
        population_token_mode: str = "projected",
        bag_aggregation: str = "projected",
        bag_aggregation_heads: int = 8,
        bag_aggregation_num_slots: int | None = None,
        bag_aggregation_num_density_slots: int | None = None,
        bag_aggregation_num_tail_fractions: int | None = None,
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
        if not 0 < typed_bag_residual_scale < 1:
            raise ValueError("typed_bag_residual_scale must be in (0, 1).")
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.routing_temperature = float(routing_temperature)
        self.class_memory_tokens = int(class_memory_tokens)
        self.rare_evidence_fractions = rare_fractions
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
        # P0-b gate (rev.2 §4.2): eval-time forcing of rare_logits = 0 to
        # measure the rare branch's contribution. Non-persistent; default
        # keeps exact existing behavior. meta_enable_rare_evidence=False is
        # the "rare-removed" arm (rev.2 step 5) -- numerically identical to
        # deletion (P0-b verified |Δpooled| 0.0009 < 0.003 on PIK3CA).
        self.force_rare_logits_zero = not meta_enable_rare_evidence
        # Ridge ablation (docs SS65). The three closed-form ridge solves are
        # independently removable to test whether they crowd out the learned
        # branches. Each flag zeroes ONLY its own ridge term, leaving that
        # branch's learned residual intact:
        #   global      (G-2) -> global_shape keeps its attention residual
        #   abundance   (P-2) -> population keeps population_attention
        #   covariance  (CV-1) -> the CV-2 relation branch is untouched
        # Parameter-preserving (strict ckpt load both ways), no new parameters.
        self.force_abundance_ridge_zero = not bool(meta_enable_abundance_ridge)
        self.force_covariance_ridge_zero = not bool(meta_enable_covariance_ridge)
        # CV-only arm (docs SS68): keep ONLY the two covariance evidence terms.
        # Applied in _fuse_evidence, which both forward paths call.
        self.covariance_only = bool(meta_covariance_only)
        if self.covariance_only and self.force_covariance_ridge_zero:
            raise ValueError(
                "meta_covariance_only with meta_enable_covariance_ridge=False "
                "leaves only CV-2, which would make the arm's name misleading. "
                "Disable one of the two."
            )
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
        # v36 Q1 (docs/architecture_v36_q1_structured_population_proposal.md):
        # which bag tokens the POPULATION branch consumes. `projected` (default)
        # is the historical behaviour -- `_projected_bag_tokens` collapses the
        # structured tokens to one, which makes the routing softmax in
        # `_population_memory_logits` a length-1 axis (weights identically 1.0),
        # i.e. the ABMIL-style selection mechanism is present but inert.
        # `structured` feeds the full token set instead, restoring it.
        # Only the population branch changes: `global_shape_classifier` and
        # `_class_memories` keep the projected token (single-factor arm).
        # No parameter is added or resized -- every parameter in that branch is
        # sized by token_dim/hidden_dim, never by the token count -- so
        # checkpoints load strictly in either mode.
        if population_token_mode not in ("projected", "structured"):
            raise ValueError(
                "population_token_mode must be 'projected' or 'structured', "
                f"got {population_token_mode!r}."
            )
        self.population_token_mode = str(population_token_mode)
        # v37 (docs/architecture_v37_context_adaptive_aggregation_proposal.md):
        # how the T structured tokens collapse to one bag token.
        #   projected        -- `_projected_bag_tokens`, a FIXED linear map learned
        #                       once and then frozen w.r.t. the episode.
        #   context_adaptive -- the collapse is a convex combination whose weights
        #                       `w` (one per token position) are produced from THIS
        #                       episode's context bags at inference time.
        # The weights are built in D space but applied to the original I-dim
        # tokens, so the bag token keeps token_dim and every downstream consumer
        # (global_shape/class memories/population) is untouched.
        if bag_aggregation not in ("projected", "context_adaptive"):
            raise ValueError(
                "bag_aggregation must be 'projected' or 'context_adaptive', "
                f"got {bag_aggregation!r}."
            )
        self.bag_aggregation = str(bag_aggregation)
        if self.bag_aggregation == "context_adaptive":
            if self.structured_tokens_per_bag is None:
                raise ValueError(
                    "bag_aggregation='context_adaptive' needs "
                    "structured_tokens_per_bag."
                )
            if (
                bag_aggregation_num_slots is None
                or bag_aggregation_num_density_slots is None
                or bag_aggregation_num_tail_fractions is None
            ):
                raise ValueError(
                    "bag_aggregation='context_adaptive' needs the aggregator's "
                    "token layout (num_slots / num_density_slots / "
                    "num_tail_fractions) to build the type embedding."
                )
            n_tokens = self.structured_tokens_per_bag
            n_slots = int(bag_aggregation_num_slots)
            n_density = int(bag_aggregation_num_density_slots)
            n_tails = int(bag_aggregation_num_tail_fractions)
            self.bag_agg_input_projection = nn.Linear(token_dim, hidden_dim)
            # Type embedding == positional encoding over the FIXED token layout
            # [global, slot_i x (center, spread, rare), tails]. Compositional on
            # purpose: slots share parameters (a slot INDEX has no stable meaning
            # across episodes, but its TYPE does -- same call `_typed_bag_tokens`
            # (v25) made when it added type/tail-fraction ids and no slot index).
            self.bag_agg_token_type = nn.Embedding(6, hidden_dim)
            self.bag_agg_slot_class = nn.Embedding(3, hidden_dim)
            self.bag_agg_tail_fraction = nn.Embedding(1 + n_tails, hidden_dim)
            type_ids, slot_class_ids, tail_ids = self._build_bag_token_type_ids(
                n_tokens, n_slots, n_density, n_tails
            )
            self.register_buffer("_bag_agg_type_ids", type_ids, persistent=False)
            self.register_buffer(
                "_bag_agg_slot_class_ids", slot_class_ids, persistent=False
            )
            self.register_buffer("_bag_agg_tail_ids", tail_ids, persistent=False)
            # One SHARED encoder over the bag axis; the type embedding is what
            # tells it which position it is looking at. Forty separate encoders
            # would cost 31.6M parameters instead of 0.79M.
            self.bag_agg_encoder = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(bag_aggregation_heads),
                dim_feedforward=4 * hidden_dim,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            # Symmetric (permutation-invariant) pooling over bags: PMA with one
            # learned seed, mirroring `memory_seeds` in `_class_memories`.
            self.bag_agg_seed = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
            self.bag_agg_pool = nn.MultiheadAttention(
                hidden_dim, int(bag_aggregation_heads), dropout=0.0, batch_first=True
            )
            self.bag_agg_norm = nn.LayerNorm(hidden_dim)
            weight_head = nn.Linear(hidden_dim, n_tokens)
            # zero-init -> w starts exactly uniform -> the bag token starts as the
            # plain mean of the T tokens, an interpretable, non-random baseline.
            nn.init.zeros_(weight_head.weight)
            nn.init.zeros_(weight_head.bias)
            self.bag_agg_weight_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                weight_head,
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
            # spherical k-means over context cells, so slot index is an
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
        # Query-count-invariant margin: the relation scorers use raw tanh(margin)
        # (no query-axis batch RMS and no learned scale), so each query's output
        # depends only on its own margin and 1-query == multi-query == training.
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
            enable_ridge=meta_enable_global_ridge,
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
        # Query-count-invariant margin: raw tanh (no batch normalization). The
        # old margin_rms averaged over the QUERY axis, coupling batched queries
        # and degenerating to tanh(+-1) at 1-query eval while training used
        # 5-12 queries/episode. Margins are already context-dispersion-scaled
        # (d0/d1 = distance/dispersion), so tanh(margin) is per-query and
        # 1-query == multi-query == training.
        bounded_margin = torch.tanh(margin)
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
        # Query-count-invariant: raw tanh, NOT a (query, slot) batch RMS or a
        # learned scale (see _covariance_relation_scores) -- each (query, slot)
        # margin depends only on itself, so 1-query == multi-query == training.
        slot_margin = torch.tanh(slot_margin)
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
        if getattr(self, "covariance_only", False):
            # CV-only (docs SS68) has a SMALLER but equally strict contract: the
            # aggregator does not compute the structural tokens at all, so the
            # dead keys are ABSENT rather than present-and-zero. Absent means a
            # stray consumer raises KeyError at the offending line instead of
            # silently averaging zeros into a live branch.
            expected_keys = {"covariance_sketch", "covariance_matrix"}
            if (
                self.covariance_relation_enabled
                and self.covariance_relation_granularity == "slot"
            ):
                expected_keys |= {
                    "slot_covariance_sketch",
                    "slot_covariance_reliability",
                }
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
            return
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

    @staticmethod
    def _build_bag_token_type_ids(
        n_tokens: int,
        num_slots: int,
        num_density_slots: int,
        num_tails: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map each structured-token position to its (type, slot class, tail id).

        Layout is fixed by `_all_structured_tokens`:
            0                       global_summary
            1 + 3i, 2 + 3i, 3 + 3i  slot i -> (center, spread, rare)
            then                    tails (one per tail fraction)
            then                    optional cls / raw-stat tokens -> "other"
        """
        type_ids = torch.full((n_tokens,), 5, dtype=torch.long)   # 5 = other
        slot_class_ids = torch.zeros(n_tokens, dtype=torch.long)  # 0 = not a slot
        tail_ids = torch.zeros(n_tokens, dtype=torch.long)        # 0 = not a tail
        type_ids[0] = 0                                           # global
        for slot in range(num_slots):
            base = 1 + 3 * slot
            if base + 2 >= n_tokens:
                break
            type_ids[base] = 1                                    # slot center
            type_ids[base + 1] = 2                                # slot spread
            type_ids[base + 2] = 3                                # slot rare
            # density vs rare slot is stable across episodes; the slot INDEX is not.
            slot_class = 1 if slot < num_density_slots else 2
            slot_class_ids[base : base + 3] = slot_class
        tail_start = 1 + 3 * num_slots
        for tail in range(num_tails):
            position = tail_start + tail
            if position >= n_tokens:
                break
            type_ids[position] = 4                                # tail
            tail_ids[position] = tail + 1
        return type_ids, slot_class_ids, tail_ids

    def _bag_agg_type_embedding(self) -> torch.Tensor:
        """[T, D] additive position identity for the aggregation encoder."""
        return (
            self.bag_agg_token_type(self._bag_agg_type_ids)
            + self.bag_agg_slot_class(self._bag_agg_slot_class_ids)
            + self.bag_agg_tail_fraction(self._bag_agg_tail_ids)
        )

    def _context_aggregation_weights(
        self, context_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Convex aggregation weights built from the CONTEXT bags only (v37).

        `context_tokens` is [C, T, I] (ragged path) or [E, C, T, I] (4D path);
        the return is [T] or [E, T]. The query never enters -- callers must pass
        the context-only token tensor, exactly like `_context_pool_stats`.

        Permutation invariance over bags is structural: the encoder gets no
        positional encoding along the bag axis and the pooling is a single-seed
        cross-attention (symmetric).
        """
        batched = context_tokens.ndim == 4
        if not batched:
            context_tokens = context_tokens.unsqueeze(0)
        episodes, n_bags, n_tokens, _ = context_tokens.shape
        if n_tokens != self.structured_tokens_per_bag:
            raise ValueError(
                "context_adaptive aggregation expected "
                f"{self.structured_tokens_per_bag} tokens, got {n_tokens}."
            )
        hidden = self.bag_agg_input_projection(context_tokens)      # [E,C,T,D]
        hidden = hidden + self._bag_agg_type_embedding()            # broadcast [T,D]
        # one sequence per (episode, token position); the sequence IS the bag axis
        sequences = hidden.permute(0, 2, 1, 3).reshape(
            episodes * n_tokens, n_bags, -1
        )                                                            # [E*T, C, D]
        encoded = self.bag_agg_encoder(sequences)                    # [E*T, C, D]
        seed = self.bag_agg_seed.unsqueeze(0).expand(
            encoded.shape[0], -1, -1
        )                                                            # [E*T, 1, D]
        pooled, _ = self.bag_agg_pool(seed, encoded, encoded, need_weights=False)
        pooled = self.bag_agg_norm(pooled.squeeze(-2))               # [E*T, D]
        weight_maker_token = pooled.reshape(episodes, n_tokens, -1)  # [E,T,D]
        logits = self.bag_agg_weight_mlp(weight_maker_token.mean(dim=1))  # [E,T]
        weights = torch.softmax(logits.float(), dim=-1).to(context_tokens.dtype)
        return weights if batched else weights.squeeze(0)

    def _projected_bag_tokens(
        self,
        tokens: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        if self.bag_aggregation == "context_adaptive":
            # v37: convex combination with episode-derived weights. The weights
            # are made in D space but applied HERE, on the original token_dim
            # tokens, so the bag token keeps its dimension and no downstream
            # module changes.
            if weights is None:
                raise ValueError(
                    "bag_aggregation='context_adaptive' requires aggregation "
                    "weights; call `_context_aggregation_weights` on the context "
                    "tokens and pass the result down."
                )
            if weights.ndim == 1:                    # [T] -> [1, T, 1]
                shaped = weights.reshape(1, -1, 1)
            elif weights.ndim == 2 and tokens.ndim == 4:   # [E, T] -> [E, 1, T, 1]
                shaped = weights.reshape(weights.shape[0], 1, -1, 1)
            else:
                raise ValueError(
                    f"Unsupported weight shape {tuple(weights.shape)} for tokens "
                    f"{tuple(tokens.shape)}."
                )
            return (tokens * shaped.to(tokens.dtype)).sum(dim=-2)
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
        self,
        representation: dict[str, torch.Tensor],
        aggregation_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return either all structured tokens, one exact mean token, or one
        learned projection token per bag."""
        if self.population_token_mode == "structured":
            # v36 Q1: bypass the 40->1 collapse for this branch only. NOTE this
            # is NOT the legacy `else` path below, which returns the slot tokens
            # alone (3 * num_slots) and drops global_summary and the tails.
            return self._all_structured_tokens(representation)
        if self.project_structured_tokens:
            tokens = self._all_structured_tokens(representation)
            return self._projected_bag_tokens(
                tokens, aggregation_weights
            ).unsqueeze(-2)
        if self.mean_pool_structured_tokens:
            return self._all_structured_tokens(representation).mean(
                dim=-2, keepdim=True
            )
        return self._flatten_slot_tokens(representation)

    def _class_memories(
        self,
        context: dict[str, torch.Tensor],
        context_labels: torch.Tensor,
        aggregation_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_tokens = self._all_structured_tokens(context)
        if self.project_structured_tokens:
            context_tokens = self._projected_bag_tokens(
                context_tokens, aggregation_weights
            ).unsqueeze(1)
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
        aggregation_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_tokens = self._population_tokens(query, aggregation_weights)
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
        if getattr(self, "force_rare_logits_zero", False):
            rare_logits = rare_logits.new_zeros(rare_logits.shape)
        if getattr(self, "covariance_only", False):
            # CV-only arm (docs SS68). Branch diagnostic: CV-1 alone scores
            # 0.9052 vs the full model's 0.9199, while Q-5 population attention
            # emits a CONSTANT (AUROC 0.5000, std 0.0000) and rare is ~0. This
            # drops every non-covariance evidence term -- including the ungated
            # `global_shape` base and the fusion interaction, which is built
            # from exactly those dead/weak terms -- leaving
            #     final = cov_res_scale*CV-1 + cov_rel_res_scale*CV-2
            # Both callers (dense and ragged) route through here, so one guard
            # covers both paths.
            #
            # NOTE (docs SS68): this still COMPUTES the discarded machinery.
            # Measured 18.73 ms vs a 0.40 ms floor on 8 bags x 4000 cells, so
            # ~98% of the forward is wasted. A real skip must derive
            # `centered_delta` from `_bag_view` (pool-standardised under
            # bag_representation=poolz_l2), NOT from a per-bag mean -- a first
            # attempt at the skip used the per-bag mean and broke dense/ragged
            # agreement by 2.4e-2.
            zeros = global_shape_logits.new_zeros(global_shape_logits.shape)
            return zeros, torch.sigmoid(self.fusion_residual_logit)
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
        aggregation_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_tokens = self._all_structured_tokens_batched(context)
        if self.project_structured_tokens:
            context_tokens = self._projected_bag_tokens(
                context_tokens, aggregation_weights
            ).unsqueeze(2)
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
        aggregation_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_slots = query["slots"]
        if self.population_token_mode == "structured":
            # v36 Q1 -- must mirror `_population_tokens`. This block duplicates
            # the token-selection logic on purpose (4D path); the file comment on
            # `_all_structured_tokens_batched` warns that drifting one copy and
            # not the other is exactly how the cls token was first missed.
            query_tokens = self._all_structured_tokens_batched(query)
        elif self.project_structured_tokens:
            all_tokens = self._all_structured_tokens_batched(query)
            query_tokens = self._projected_bag_tokens(
                all_tokens, aggregation_weights
            ).unsqueeze(2)
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
        if self.covariance_only:
            return self._covariance_only_forward(
                context, context_labels, query,
                batched=True, return_auxiliary=return_auxiliary,
            )
        context_global_tokens = context["global_summary"]
        query_global_tokens = query["global_summary"]
        # v37 -- mirrors the ragged path above; see the comment there.
        aggregation_weights = None
        if self.bag_aggregation == "context_adaptive":
            aggregation_weights = self._context_aggregation_weights(
                self._all_structured_tokens_batched(context)
            )
        if self.project_structured_tokens:
            context_global_tokens = self._projected_bag_tokens(
                self._all_structured_tokens_batched(context), aggregation_weights
            )
            query_global_tokens = self._projected_bag_tokens(
                self._all_structured_tokens_batched(query), aggregation_weights
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
        class_memories = self._class_memories_batched(
            context, context_labels, aggregation_weights
        )
        population_attention_logits, population_weights = (
            self._population_memory_logits_batched(
                query, class_memories, aggregation_weights
            )
        )
        abundance_ridge_logits = self._abundance_ridge_logits_batched(
            context["slot_metadata"],
            context_labels,
            query["slot_metadata"],
        )
        if self.force_abundance_ridge_zero:
            abundance_ridge_logits = abundance_ridge_logits.new_zeros(
                abundance_ridge_logits.shape
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
        if self.force_covariance_ridge_zero:
            covariance_ridge_logits = covariance_ridge_logits.new_zeros(
                covariance_ridge_logits.shape
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
        if self.typed_bag_preserving_branch:
            logits = logits + typed_bag_residual_scale * typed_bag_logits
        if self.use_instance_attention_mil:
            mil_logits = self._instance_attention_mil_logits_batched(
                query_instances, class_memories
            )
            logits = logits + torch.sigmoid(self.mil_residual_logit) * mil_logits
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
                    "typed_bag_logits": typed_bag_logits,
                    "typed_bag_residual_scale": typed_bag_residual_scale.expand(
                        episodes
                    ),
                }
                if self.typed_bag_preserving_branch
                else {}
            ),
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
        if self.covariance_only:
            return self._covariance_only_forward(
                context, context_labels, query,
                batched=False, return_auxiliary=return_auxiliary,
            )
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
        # v37: one set of aggregation weights per episode, derived from the
        # CONTEXT bags only, then shared by every consumer (global shape, class
        # memories, population). Computing it here -- rather than inside each
        # consumer -- is what keeps the query out of the weights.
        aggregation_weights = None
        if self.bag_aggregation == "context_adaptive":
            aggregation_weights = self._context_aggregation_weights(
                self._all_structured_tokens(context)
            )
        if self.project_structured_tokens:
            context_global_tokens = self._projected_bag_tokens(
                self._all_structured_tokens(context), aggregation_weights
            )
            query_global_tokens = self._projected_bag_tokens(
                self._all_structured_tokens(query), aggregation_weights
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
        class_memories = self._class_memories(
            context, context_labels, aggregation_weights
        )
        population_attention_logits, population_weights = (
            self._population_memory_logits(
                query, class_memories, aggregation_weights
            )
        )
        abundance_ridge_logits = self._abundance_ridge_logits(
            context["slot_metadata"],
            context_labels,
            query["slot_metadata"],
        )
        if self.force_abundance_ridge_zero:
            abundance_ridge_logits = abundance_ridge_logits.new_zeros(
                abundance_ridge_logits.shape
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
        if self.force_covariance_ridge_zero:
            covariance_ridge_logits = covariance_ridge_logits.new_zeros(
                covariance_ridge_logits.shape
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
        if self.typed_bag_preserving_branch:
            logits = logits + typed_bag_residual_scale * typed_bag_logits
        if self.use_instance_attention_mil:
            mil_logits = self._instance_attention_mil_logits(
                query_instances, class_memories
            )
            logits = logits + torch.sigmoid(self.mil_residual_logit) * mil_logits
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
                    "typed_bag_logits": typed_bag_logits,
                    "typed_bag_residual_scale": typed_bag_residual_scale,
                }
                if self.typed_bag_preserving_branch
                else {}
            ),
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
        meta_enable_rare_evidence: bool = True,
        meta_enable_global_ridge: bool = True,
        meta_enable_abundance_ridge: bool = True,
        meta_enable_covariance_ridge: bool = True,
        meta_covariance_only: bool = False,
        meta_population_token_mode: str = "projected",
        meta_bag_aggregation: str = "projected",
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
        aggregator_slot_latent_dim: int | None = None,
        aggregator_slot_query_latent_dim: int | None = None,
        aggregator_slot_affinity_dim: int | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.aggregator = StructuredEpisodePopulationAggregator(
            covariance_only=meta_covariance_only,
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
            meta_enable_rare_evidence=meta_enable_rare_evidence,
            meta_enable_global_ridge=meta_enable_global_ridge,
            meta_enable_abundance_ridge=meta_enable_abundance_ridge,
            meta_enable_covariance_ridge=meta_enable_covariance_ridge,
            meta_covariance_only=meta_covariance_only,
            population_token_mode=meta_population_token_mode,
            bag_aggregation=meta_bag_aggregation,
            bag_aggregation_num_slots=self.aggregator.num_slots,
            bag_aggregation_num_density_slots=self.aggregator.num_density_slots,
            bag_aggregation_num_tail_fractions=len(self.aggregator.tail_fractions),
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
        if self.meta_classifier.covariance_only:
            # CV-only (docs SS68): downstream needs only the covariance sketch
            # and matrix. `centered_delta` -- the raw per-bag centring that
            # `_bag_view` returns as its THIRD value -- does not depend on the
            # context-pool statistics, so this skips the pool stats, the
            # poolz_l2 standardisation of every cell, the per-episode anchors
            # (top-k over cells) and the raw-stat tokens outright.
            # `classification_flat` is passed through unstandardised because the
            # only consumer left, `_covariance_sketch`, already has its input.
            # Equivalence with the full-branch model is pinned by
            # tests/test_ridge_ablation.py::TestCovarianceOnly.
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
            classification_flat = flat_x
            # Only consumer left is `query_instances` for the rare branch, which
            # the CV-only meta-classifier returns before ever touching. A
            # reshape, not a recomputation.
            classification_x = x
            global_summary = None
            raw_stats = None
            per_bag_anchors = None
        elif self.aggregator.bag_representation in ("poolz", "poolz_l2"):
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
        if not self.meta_classifier.covariance_only:
            classification_flat, global_summary, centered_delta = (
                self.aggregator._bag_view(
                    flat_x, pool_mean, pool_std, cell_mask=flat_cell_mask
                )
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
        structural = {}
        if not self.meta_classifier.covariance_only:
            # CV-only never builds these -- reporting them as zeros would hand
            # callers a plausible-looking tensor for something that was never
            # computed (docs SS68).
            structural = {
                "bag_tokens": representation["global_summary"],
                "slot_tokens": representation["slots"],
                "tail_tokens": representation["tails"],
                "slot_metadata": representation["slot_metadata"],
            }
        return logits, {
            **structural,
            "context_mask": is_context,
            "aggregator": aggregator_auxiliary,
            **auxiliary,
        }
