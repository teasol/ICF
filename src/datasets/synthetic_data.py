from __future__ import annotations
from typing import Any
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_MANIFOLD_DECOMPOSITION_LOCK = Lock()


@dataclass(frozen=True)
class SyntheticEpisode:
    """One binary classification episode made of bags of manifold samples."""

    x: torch.Tensor
    y: torch.Tensor
    response_score: torch.Tensor | None = None
    response_task: str | None = None
    response_fraction: torch.Tensor | None = None
    response_dispersion_factor: torch.Tensor | None = None
    responsive_component_index: int | None = None
    effect_cell_fraction: torch.Tensor | None = None
    oracle_response_abundance: torch.Tensor | None = None
    effect_scale_multiplier: float = 1.0
    rare_response: bool = False
    oracle_population_features: torch.Tensor | None = None

    @property
    def flipped_y(self) -> torch.Tensor:
        """Return the opposite label assignment for the same bags."""
        return 1 - self.y


class SyntheticManifoldGenerator:
    """Generate binary episodes from low-dimensional distribution mixtures.

    Every episode samples a new latent mixture and a new manifold mapping. In
    continuous-response episodes, both classes use the same components; a
    donor-specific signed response score changes the response component's
    abundance and state, and the score sign defines the binary label.
    """

    def __init__(
        self,
        num_bags: int | tuple[int, int] = 50,
        num_cells: int | tuple[int, int] = 1000,
        latent_dim: int = 8,
        output_dim: int = 512,
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 3,
        class_separation: tuple[float, float] = (0.5, 2.0),
        latent_scale: tuple[float, float] = (0.5, 1.5),
        shared_component_probability: float = 0.5,
        shared_component_fraction: float | tuple[float, float] = (0.3, 0.7),
        num_shared_components: int | tuple[int, int] = 1,
        donor_shift_scale: float = 0.25,
        donor_component_shift_scale: float = 0.1,
        donor_mixture_logit_scale: float = 0.5,
        shared_component_base_logit_scale: float = 0.0,
        donor_shared_component_logit_scale: float = 0.0,
        continuous_response_probability: float = 0.0,
        response_score_scale: float = 1.0,
        response_score_min_margin: float = 0.05,
        response_mixture_effect_scale: float = 0.5,
        response_state_effect_scale: float | tuple[float, float] = (0.1, 0.5),
        response_task_probabilities: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 1.0),
        response_covariance_effect_scale: float | tuple[float, float] = (0.0, 0.0),
        rare_response_probability: float = 0.0,
        rare_response_fraction: float | tuple[float, float] = (0.01, 0.08),
        observation_noise: float = 0.01,
        normalize_output: bool = False,
        output_norm_eps: float = 1e-8,
        manifold_mode: str = "nonlinear",
        manifold_seed: int = 0,
        manifold_max_condition_number: float = 3.0,
        balanced: bool = True,
    ) -> None:
        if isinstance(num_bags, int):
            num_bags = (num_bags, num_bags)
        if (
            len(num_bags) != 2
            or not all(isinstance(value, int) for value in num_bags)
            or num_bags[0] < 2
            or num_bags[0] > num_bags[1]
        ):
            raise ValueError(
                "num_bags must be an integer of at least two or an ordered "
                "[min_bags, max_bags] integer range."
            )
        if isinstance(num_cells, int):
            num_cells = (num_cells, num_cells)
        if (
            len(num_cells) != 2
            or not all(isinstance(value, int) for value in num_cells)
            or num_cells[0] < 1
            or num_cells[0] > num_cells[1]
        ):
            raise ValueError(
                "num_cells must be a positive integer or an ordered "
                "[min_cells, max_cells] integer range."
            )
        if latent_dim < 1 or output_dim < 1 or mlp_hidden_dim < 1:
            raise ValueError("All feature dimensions must be positive.")
        if mlp_num_layers < 1:
            raise ValueError("mlp_num_layers must be at least 1.")
        self._validate_range("class_separation", class_separation, positive=False)
        self._validate_range("latent_scale", latent_scale, positive=True)
        if isinstance(shared_component_fraction, (int, float)):
            shared_component_fraction = (
                float(shared_component_fraction),
                float(shared_component_fraction),
            )
        self._validate_range(
            "shared_component_fraction", shared_component_fraction, positive=False
        )
        if shared_component_fraction[1] > 1:
            raise ValueError("shared_component_fraction values cannot exceed 1.")
        if not 0 <= shared_component_probability <= 1:
            raise ValueError("shared_component_probability must be in [0, 1].")
        if isinstance(num_shared_components, int):
            num_shared_components = (
                num_shared_components,
                num_shared_components,
            )
        shared_count_low, shared_count_high = num_shared_components
        if shared_count_low < 1 or shared_count_low > shared_count_high:
            raise ValueError(
                "num_shared_components must be a positive integer or an ordered "
                "positive integer range."
            )
        if min(
            donor_shift_scale,
            donor_component_shift_scale,
            donor_mixture_logit_scale,
            shared_component_base_logit_scale,
            donor_shared_component_logit_scale,
            response_score_scale,
            response_score_min_margin,
            response_mixture_effect_scale,
        ) < 0:
            raise ValueError("Donor and response scales must be non-negative.")
        if not 0 <= continuous_response_probability <= 1:
            raise ValueError("continuous_response_probability must be in [0, 1].")
        if isinstance(response_state_effect_scale, (int, float)):
            response_state_effect_scale = (
                float(response_state_effect_scale),
                float(response_state_effect_scale),
            )
        self._validate_range(
            "response_state_effect_scale",
            response_state_effect_scale,
            positive=False,
        )
        if len(response_task_probabilities) not in (3, 4, 5):
            raise ValueError(
                "response_task_probabilities must contain weights for "
                "composition-only, state-only, covariance-only, interaction, "
                "and combined tasks. Legacy three/four-value weights are also "
                "accepted."
            )
        if len(response_task_probabilities) == 3:
            composition_weight, state_weight, combined_weight = (
                response_task_probabilities
            )
            response_task_probabilities = (
                composition_weight,
                state_weight,
                0.0,
                0.0,
                combined_weight,
            )
        elif len(response_task_probabilities) == 4:
            composition_weight, state_weight, covariance_weight, combined_weight = (
                response_task_probabilities
            )
            response_task_probabilities = (
                composition_weight,
                state_weight,
                covariance_weight,
                0.0,
                combined_weight,
            )
        if any(weight < 0 for weight in response_task_probabilities):
            raise ValueError("response_task_probabilities cannot be negative.")
        response_task_weight_sum = float(sum(response_task_probabilities))
        if response_task_weight_sum <= 0:
            raise ValueError("At least one response task must have positive weight.")
        if response_task_probabilities[3] > 0 and shared_count_high < 2:
            raise ValueError(
                "Interaction tasks require num_shared_components to allow at "
                "least two background components."
            )
        if isinstance(response_covariance_effect_scale, (int, float)):
            response_covariance_effect_scale = (
                float(response_covariance_effect_scale),
                float(response_covariance_effect_scale),
            )
        self._validate_range(
            "response_covariance_effect_scale",
            response_covariance_effect_scale,
            positive=False,
        )
        if not 0 <= rare_response_probability <= 1:
            raise ValueError("rare_response_probability must be in [0, 1].")
        if isinstance(rare_response_fraction, (int, float)):
            rare_response_fraction = (
                float(rare_response_fraction),
                float(rare_response_fraction),
            )
        self._validate_range(
            "rare_response_fraction", rare_response_fraction, positive=True
        )
        if rare_response_fraction[1] >= 1:
            raise ValueError("rare_response_fraction values must be smaller than 1.")
        if observation_noise < 0:
            raise ValueError("observation_noise must be non-negative.")
        if output_norm_eps <= 0:
            raise ValueError("output_norm_eps must be positive.")
        valid_manifold_modes = {
            "nonlinear", "shared_nonlinear", "orthogonal", "bounded_linear"
        }
        if manifold_mode not in valid_manifold_modes:
            raise ValueError(
                f"manifold_mode must be one of {sorted(valid_manifold_modes)}."
            )
        if manifold_max_condition_number < 1:
            raise ValueError("manifold_max_condition_number must be at least 1.")

        self.num_bags = tuple(num_bags)
        self.num_cells = tuple(num_cells)
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.class_separation = class_separation
        self.latent_scale = latent_scale
        self.shared_component_probability = shared_component_probability
        self.shared_component_fraction = shared_component_fraction
        self.num_shared_components = num_shared_components
        self.donor_shift_scale = donor_shift_scale
        self.donor_component_shift_scale = donor_component_shift_scale
        self.donor_mixture_logit_scale = donor_mixture_logit_scale
        self.shared_component_base_logit_scale = shared_component_base_logit_scale
        self.donor_shared_component_logit_scale = donor_shared_component_logit_scale
        self.continuous_response_probability = continuous_response_probability
        self.response_score_scale = response_score_scale
        self.response_score_min_margin = response_score_min_margin
        self.response_mixture_effect_scale = response_mixture_effect_scale
        self.response_state_effect_scale = response_state_effect_scale
        self.response_task_probabilities = tuple(
            float(weight) / response_task_weight_sum
            for weight in response_task_probabilities
        )
        self.response_covariance_effect_scale = response_covariance_effect_scale
        self.rare_response_probability = rare_response_probability
        self.rare_response_fraction = rare_response_fraction
        self.observation_noise = observation_noise
        self.normalize_output = bool(normalize_output)
        self.output_norm_eps = float(output_norm_eps)
        self.manifold_mode = manifold_mode
        self.manifold_seed = int(manifold_seed)
        self.manifold_max_condition_number = float(manifold_max_condition_number)
        self.balanced = balanced

    def sample_episode(
        self,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        effect_scale_multiplier: float = 1.0,
        num_bags: int | None = None,
        num_cells: int | None = None,
    ) -> SyntheticEpisode:
        """Sample one episode and randomly permute its bag order."""
        if effect_scale_multiplier < 0:
            raise ValueError("effect_scale_multiplier must be non-negative.")
        device = torch.device(device)
        if num_bags is None:
            num_bags = self.sample_num_bags(generator, device)
        elif not self.num_bags[0] <= num_bags <= self.num_bags[1]:
            raise ValueError(
                f"num_bags must be within the configured range {self.num_bags}."
            )
        if num_cells is None:
            num_cells = self.sample_num_cells(generator, device)
        elif not self.num_cells[0] <= num_cells <= self.num_cells[1]:
            raise ValueError(
                f"num_cells must be within the configured range {self.num_cells}."
            )
        y = self._sample_labels(num_bags, generator, device)
        response_score = self._sample_response_score(y, generator, device)
        use_shared_component = bool(
            torch.rand((), device=device, generator=generator).item()
            < self.shared_component_probability
        )
        use_continuous_response = use_shared_component and bool(
            torch.rand((), device=device, generator=generator).item()
            < self.continuous_response_probability
        )
        response_task = (
            self._sample_response_task(generator, device)
            if use_continuous_response
            else None
        )
        rare_response = use_continuous_response and response_task != "interaction" and bool(
            torch.rand((), device=device, generator=generator).item()
            < self.rare_response_probability
        )
        response_fraction = None
        response_dispersion_factor = None
        if use_shared_component:
            shared_component_range = self.num_shared_components
            if response_task == "interaction":
                shared_component_range = (
                    max(2, shared_component_range[0]),
                    shared_component_range[1],
                )
            num_shared = self._sample_integer(shared_component_range, generator, device)
            num_distributions = num_shared + (1 if use_continuous_response else 2)
        else:
            num_shared = 0
            num_distributions = 2
        class_mean, class_scale = self._sample_distributions(
            num_distributions, generator, device
        )
        responsive_component_index = None
        if response_task == "interaction":
            responsive_component_index = int(
                torch.randint(
                    0,
                    num_shared,
                    (),
                    device=device,
                    generator=generator,
                ).item()
            )

        # A latent point is one cell. Every cell in a bag is sampled from the
        # one episode-level distribution selected by that bag's label.
        z = torch.randn(
            num_bags,
            num_cells,
            self.latent_dim,
            device=device,
            generator=generator,
        )
        if use_shared_component:
            # Categorical episodes contain A_1...A_k plus label-specific X/Y.
            # Continuous episodes instead use one common response component
            # whose abundance and state vary with each donor's signed score.
            if rare_response:
                base_response_fraction = self._sample_uniform(
                    self.rare_response_fraction, generator, device
                )
                shared_fraction = 1.0 - base_response_fraction
            else:
                shared_fraction = self._sample_uniform(
                    self.shared_component_fraction, generator, device
                )
            shared_fraction = torch.full(
                (num_bags, 1), shared_fraction, device=device
            )
            if self.donor_mixture_logit_scale > 0:
                base_logit = torch.logit(shared_fraction.clamp(1e-4, 1 - 1e-4))
                mixture_noise = torch.randn(
                    num_bags,
                    1,
                    device=device,
                    generator=generator,
                )
                base_logit = base_logit + self.donor_mixture_logit_scale * mixture_noise
            else:
                base_logit = torch.logit(shared_fraction.clamp(1e-4, 1 - 1e-4))
            if response_task in ("composition", "combined"):
                # Positive scores continuously increase the response-related
                # component by decreasing the total shared fraction.
                base_logit = base_logit - (
                    self.response_mixture_effect_scale
                    * effect_scale_multiplier
                    * response_score.unsqueeze(1)
                )
            shared_fraction = torch.sigmoid(base_logit)
            if use_continuous_response:
                response_fraction = 1.0 - shared_fraction
            shared_mask = torch.rand(
                num_bags,
                num_cells,
                device=device,
                generator=generator,
            ) < shared_fraction
            # One episode defines a shared cohort composition, while each donor
            # has its own immune-population mixture around that background.
            shared_logits = torch.zeros(num_shared, device=device)
            if self.shared_component_base_logit_scale > 0:
                shared_logits = shared_logits + (
                    self.shared_component_base_logit_scale
                    * torch.randn(
                        num_shared,
                        device=device,
                        generator=generator,
                    )
                )
            shared_logits = shared_logits.unsqueeze(0).expand(num_bags, -1)
            if self.donor_shared_component_logit_scale > 0:
                shared_logits = shared_logits + (
                    self.donor_shared_component_logit_scale
                    * torch.randn(
                        num_bags,
                        num_shared,
                        device=device,
                        generator=generator,
                    )
                )
            shared_probabilities = torch.softmax(shared_logits, dim=-1)
            shared_index = torch.multinomial(
                shared_probabilities,
                num_samples=num_cells,
                replacement=True,
                generator=generator,
            )
            if use_continuous_response:
                specific_index = torch.full(
                    (num_bags, num_cells),
                    num_shared,
                    dtype=torch.long,
                    device=device,
                )
            else:
                specific_index = (num_shared + y).unsqueeze(1).expand(
                    -1, num_cells
                )
            component_index = torch.where(
                shared_mask, shared_index, specific_index
            )
        else:
            component_index = y.unsqueeze(1).expand(-1, num_cells)
        z = z * class_scale[component_index]
        z = z + class_mean[component_index]

        effect_mask = None
        effect_cell_fraction = None
        effect_component_index = None
        if use_continuous_response:
            effect_component_index = (
                responsive_component_index
                if responsive_component_index is not None
                else num_shared
            )
            effect_mask = (component_index == effect_component_index).unsqueeze(-1)
            effect_cell_fraction = effect_mask.squeeze(-1).float().mean(dim=1)

        if response_task in ("covariance", "interaction", "combined"):
            covariance_effect_scale = self._sample_uniform(
                self.response_covariance_effect_scale, generator, device
            )
            covariance_effect_scale *= effect_scale_multiplier
            covariance_direction = torch.randn(
                self.latent_dim, device=device, generator=generator
            )
            covariance_direction = covariance_direction / (
                covariance_direction.norm().clamp_min(1e-8)
            )
            response_center = class_mean[effect_component_index].view(
                1, 1, self.latent_dim
            )
            centered_response = z - response_center
            projected_response = (
                centered_response * covariance_direction.view(1, 1, -1)
            ).sum(dim=-1, keepdim=True)
            response_log_scale = (
                response_score.view(num_bags, 1, 1)
                * covariance_effect_scale
            ).clamp(-1.5, 1.5)
            response_dispersion_factor = response_log_scale.exp().flatten()
            z = z + (
                effect_mask
                * projected_response
                * torch.expm1(response_log_scale)
                * covariance_direction.view(1, 1, -1)
            )

        if response_task in ("state", "interaction", "combined"):
            effect_scale = self._sample_uniform(
                self.response_state_effect_scale, generator, device
            )
            effect_direction = torch.randn(
                self.latent_dim, device=device, generator=generator
            )
            effect_direction = effect_direction / effect_direction.norm().clamp_min(1e-8)
            response_shift = (
                response_score.view(num_bags, 1, 1)
                * effect_scale
                * effect_scale_multiplier
                * effect_direction.view(1, 1, self.latent_dim)
            )
            z = z + effect_mask * response_shift

        # Same-label donors share episode-level prototypes, but each donor has
        # independent nuisance and component-specific biological variation.
        if self.donor_shift_scale > 0:
            donor_shift = torch.randn(
                num_bags,
                1,
                self.latent_dim,
                device=device,
                generator=generator,
            )
            z = z + self.donor_shift_scale * donor_shift
        if self.donor_component_shift_scale > 0:
            component_shift = torch.randn(
                num_bags,
                num_distributions,
                self.latent_dim,
                device=device,
                generator=generator,
            )
            bag_index = torch.arange(num_bags, device=device).unsqueeze(1)
            z = z + self.donor_component_shift_scale * component_shift[
                bag_index, component_index
            ]

        x = self._map_episode_manifold(z, generator, device)
        if self.observation_noise > 0:
            noise = torch.randn(
                x.shape, dtype=x.dtype, device=device, generator=generator
            )
            x = x + self.observation_noise * noise
        if self.normalize_output:
            x = F.normalize(x, dim=-1, eps=self.output_norm_eps)

        # Diagnostic-only features use the generator's true responsive
        # population membership. They are never returned by the training
        # Dataset, but let us verify that abundance/state/dispersion remain
        # identifiable after the random manifold mapping.
        oracle_population_features = None
        if effect_mask is not None:
            population_weight = effect_mask.to(x.dtype)
            population_count = population_weight.sum(dim=1).clamp_min(1.0)
            population_fraction = population_count / num_cells
            population_mean = (x * population_weight).sum(dim=1) / population_count
            centered = (x - population_mean.unsqueeze(1)) * population_weight
            population_variance = centered.square().sum(dim=1) / population_count
            fraction_logit = torch.logit(
                population_fraction.clamp(1.0 / num_cells, 1.0 - 1.0 / num_cells)
            )
            oracle_population_features = torch.cat(
                (fraction_logit, population_mean, population_variance), dim=-1
            )

        permutation = torch.randperm(
            num_bags, device=device, generator=generator
        )
        return SyntheticEpisode(
            x=x[permutation],
            y=y[permutation],
            response_score=response_score[permutation],
            response_task=response_task,
            response_fraction=(
                response_fraction[permutation].squeeze(1)
                if response_fraction is not None
                else None
            ),
            response_dispersion_factor=(
                response_dispersion_factor[permutation]
                if response_dispersion_factor is not None
                else None
            ),
            responsive_component_index=responsive_component_index,
            effect_cell_fraction=(
                effect_cell_fraction[permutation]
                if effect_cell_fraction is not None
                else None
            ),
            oracle_response_abundance=(
                effect_cell_fraction[permutation].detach()
                if effect_cell_fraction is not None
                else None
            ),
            effect_scale_multiplier=effect_scale_multiplier,
            rare_response=rare_response,
            oracle_population_features=(
                oracle_population_features[permutation]
                if oracle_population_features is not None
                else None
            ),
        )

    def sample_num_cells(
        self,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> int:
        """Sample one episode-wide cell count from the configured range."""
        return self._sample_integer(
            self.num_cells,
            generator,
            torch.device(device),
        )

    def sample_num_bags(
        self,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> int:
        """Sample one episode-wide bag count from the configured range."""
        return self._sample_integer(
            self.num_bags,
            generator,
            torch.device(device),
        )

    def _sample_response_task(
        self,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> str:
        probabilities = torch.tensor(
            self.response_task_probabilities,
            dtype=torch.float32,
            device=device,
        )
        task_index = int(
            torch.multinomial(
                probabilities,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).item()
        )
        return (
            "composition",
            "state",
            "covariance",
            "interaction",
            "combined",
        )[task_index]

    def _sample_response_score(
        self,
        y: torch.Tensor,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        magnitude = self.response_score_min_margin + self.response_score_scale * torch.abs(
            torch.randn(y.numel(), device=device, generator=generator)
        )
        sign = y.to(dtype=torch.float32).mul(2).sub(1)
        return sign * magnitude

    def _sample_labels(
        self,
        num_bags: int,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        if self.balanced:
            num_one = num_bags // 2
            return torch.cat(
                (
                    torch.zeros(
                        num_bags - num_one,
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.ones(num_one, dtype=torch.long, device=device),
                )
            )

        # Independent labels remove the episode-level class-count variable:
        # context label counts contain no information about a masked target.
        return torch.randint(
            0,
            2,
            (num_bags,),
            dtype=torch.long,
            device=device,
            generator=generator,
        )

    def _sample_distributions(
        self,
        num_distributions: int,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        directions = torch.randn(
            num_distributions,
            self.latent_dim,
            device=device,
            generator=generator,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        separation = self._sample_uniform(self.class_separation, generator, device)

        center = torch.randn(self.latent_dim, device=device, generator=generator)
        center = center * 0.5
        class_mean = center.unsqueeze(0) + separation * directions

        scale_low, scale_high = self.latent_scale
        class_scale = torch.empty(
            num_distributions, self.latent_dim, device=device
        ).uniform_(
            scale_low,
            scale_high,
            generator=generator,
        )
        return class_mean, class_scale

    def _sample_mlp(
        self,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        dimensions = [self.latent_dim]
        dimensions.extend([self.mlp_hidden_dim] * (self.mlp_num_layers - 1))
        dimensions.append(self.output_dim)

        weights: list[torch.Tensor] = []
        biases: list[torch.Tensor] = []
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
            bound = math.sqrt(6.0 / (input_dim + output_dim))
            weight = torch.empty(output_dim, input_dim, device=device).uniform_(
                -bound,
                bound,
                generator=generator,
            )
            bias = torch.empty(output_dim, device=device).uniform_(
                -1.0 / math.sqrt(input_dim),
                1.0 / math.sqrt(input_dim),
                generator=generator,
            )
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    def _manifold_generator(self, device: torch.device) -> torch.Generator:
        """Return a reproducible generator without advancing the episode RNG."""
        return torch.Generator(device=device).manual_seed(self.manifold_seed)

    def _map_episode_manifold(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None,
        device: torch.device,
    ) -> torch.Tensor:
        mode = self.manifold_mode
        parameter_generator = (
            self._manifold_generator(device) if mode == "shared_nonlinear" else generator
        )
        if mode in ("nonlinear", "shared_nonlinear"):
            weights, biases = self._sample_mlp(parameter_generator, device)
            return self._map_to_manifold(z, weights, biases)

        # A tall matrix with orthonormal columns is an isometric embedding from
        # latent space into output space. A square latent-space rotation makes
        # every episode use a different orientation without changing distances.
        if self.output_dim < self.latent_dim:
            raise ValueError(
                f"{mode} requires output_dim >= latent_dim for an isometric basis."
            )
        # CUDA linalg lazy initialization is not thread-safe when outer-batch
        # prefetch creates several episode manifolds concurrently.
        with _MANIFOLD_DECOMPOSITION_LOCK:
            basis, _ = torch.linalg.qr(
                torch.randn(
                    self.output_dim,
                    self.latent_dim,
                    device=device,
                    generator=parameter_generator,
                ),
                mode="reduced",
            )
            rotation, _ = torch.linalg.qr(
                torch.randn(
                    self.latent_dim,
                    self.latent_dim,
                    device=device,
                    generator=parameter_generator,
                )
            )
        singular_values = torch.ones(self.latent_dim, device=device)
        if mode == "bounded_linear":
            singular_values = torch.empty(self.latent_dim, device=device).uniform_(
                1.0,
                self.manifold_max_condition_number,
                generator=parameter_generator,
            )
            # Make the configured upper bound exact and avoid scale drift.
            singular_values[0] = 1.0
            if self.latent_dim > 1:
                singular_values[-1] = self.manifold_max_condition_number
        weight = basis @ torch.diag(singular_values) @ rotation.T
        return F.linear(z, weight)

    @staticmethod
    def _map_to_manifold(
        z: torch.Tensor,
        weights: list[torch.Tensor],
        biases: list[torch.Tensor],
    ) -> torch.Tensor:
        x = z
        for layer_index, (weight, bias) in enumerate(zip(weights, biases)):
            x = F.linear(x, weight, bias)
            if layer_index < len(weights) - 1:
                x = F.gelu(x)
        return x

    @staticmethod
    def _sample_uniform(
        value_range: tuple[float, float],
        generator: torch.Generator | None,
        device: torch.device,
    ) -> float:
        low, high = value_range
        return float(
            torch.empty((), device=device)
            .uniform_(low, high, generator=generator)
            .item()
        )

    @staticmethod
    def _sample_integer(
        value_range: tuple[int, int],
        generator: torch.Generator | None,
        device: torch.device,
    ) -> int:
        low, high = value_range
        if low == high:
            return low
        return int(
            torch.randint(
                low, high + 1, (), device=device, generator=generator
            ).item()
        )

    @staticmethod
    def _validate_range(
        name: str,
        value_range: tuple[float, float],
        positive: bool,
    ) -> None:
        low, high = value_range
        if low > high or (positive and low <= 0) or (not positive and low < 0):
            qualifier = "positive " if positive else "non-negative "
            raise ValueError(f"{name} must be an ordered {qualifier}range.")


class SyntheticEpisodeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map-style dataset that generates a fresh episode for every item access."""

    def __init__(
        self,
        episodes_per_epoch: int = 1000,
        seed: int | None = None,
        fixed_episode_count: int | None = None,
        generation_device: str = "cpu",
        shape_group_size: int = 1,
        difficulty_curriculum_episodes: int = 0,
        effect_scale_start: float | tuple[float, float] = (1.0, 1.0),
        effect_scale_end: float | tuple[float, float] = (1.0, 1.0),
        return_oracle_diagnostics: bool = False,
        **generator_kwargs: Any,
    ) -> None:
        if episodes_per_epoch < 1:
            raise ValueError("episodes_per_epoch must be positive.")
        if fixed_episode_count is not None and not 1 <= fixed_episode_count <= episodes_per_epoch:
            raise ValueError(
                "fixed_episode_count must be in [1, episodes_per_epoch]."
            )
        if fixed_episode_count is not None and seed is None:
            raise ValueError("fixed_episode_count requires a fixed dataset seed.")
        if difficulty_curriculum_episodes < 0:
            raise ValueError("difficulty_curriculum_episodes cannot be negative.")
        if shape_group_size < 1:
            raise ValueError("shape_group_size must be positive.")
        effect_scale_start = self._as_non_negative_range(
            "effect_scale_start", effect_scale_start
        )
        effect_scale_end = self._as_non_negative_range(
            "effect_scale_end", effect_scale_end
        )
        self.episodes_per_epoch = episodes_per_epoch
        self.seed = seed
        self.fixed_episode_count = fixed_episode_count
        self.generation_device = generation_device
        self.difficulty_curriculum_episodes = difficulty_curriculum_episodes
        self.effect_scale_start = effect_scale_start
        self.shape_group_size = int(shape_group_size)
        self.effect_scale_end = effect_scale_end
        self.return_oracle_diagnostics = bool(return_oracle_diagnostics)
        self._sample_count = 0
        self.episode_generator = SyntheticManifoldGenerator(**generator_kwargs)

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def set_curriculum_epoch(self, epoch: int, samples_per_rank: int) -> None:
        """Restore the per-rank stream position, including after a resume."""
        if epoch < 0 or samples_per_rank < 1:
            raise ValueError(
                "epoch must be non-negative and samples_per_rank must be positive."
            )
        self._sample_count = epoch * samples_per_rank

    def _generation_device(self) -> torch.device:
        """Resolve bare ``cuda`` to this DDP rank's current local device.

        CUDA's current device is thread-local. Nested generation workers would
        otherwise interpret ``torch.device("cuda")`` as device zero, causing
        different DDP ranks to generate into the same GPU.
        """
        device = torch.device(self.generation_device)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device

    def __getitems__(
        self, indices: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if (
            len(indices) <= 1
            or self.seed is not None
            or self._generation_device().type != "cuda"
        ):
            return [self[index] for index in indices]
        start = self._sample_count
        self._sample_count += len(indices)
        device = self._generation_device()
        shape_seed = torch.initial_seed() + start // self.shape_group_size
        shape_generator = torch.Generator(device=device).manual_seed(shape_seed)
        num_bags = self.episode_generator.sample_num_bags(
            shape_generator, device=device
        )
        num_cells = self.episode_generator.sample_num_cells(
            shape_generator, device=device
        )

        def generate(offset: int, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                sample = self._generate_at(
                    index,
                    start + offset,
                    num_bags=num_bags,
                    num_cells=num_cells,
                    device=device,
                )
            stream.synchronize()
            return sample

        with ThreadPoolExecutor(max_workers=len(indices)) as executor:
            futures = [
                executor.submit(generate, offset, index)
                for offset, index in enumerate(indices)
            ]
            return [future.result() for future in futures]

    def _generate_at(
        self,
        index: int,
        sample_count: int,
        num_bags: int,
        num_cells: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rank = int(os.environ.get("RANK", "0"))
        sample_seed = torch.initial_seed() + rank * 1_000_003 + sample_count
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        effect_scale_multiplier = self._sample_effect_scale(
            generator,
            device,
            sample_count,
            final_difficulty=False,
        )
        episode = self.episode_generator.sample_episode(
            generator,
            device=device,
            effect_scale_multiplier=effect_scale_multiplier,
            num_bags=num_bags,
            num_cells=num_cells,
        )
        return self._format_episode(episode)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self._generation_device()
        sample_count = self._sample_count
        episode_index = (
            index % self.fixed_episode_count
            if self.fixed_episode_count is not None
            else index
        )
        if self.seed is None:
            # Every DDP rank starts from the same Lightning seed. Give each rank
            # an independent, non-repeating episode stream without changing the
            # global RNG used by the model.
            rank = int(os.environ.get("RANK", "0"))
            sample_seed = torch.initial_seed() + rank * 1_000_003 + self._sample_count
            self._sample_count += 1
        else:
            # Validation/test episodes remain fixed and reproducible by index.
            sample_seed = self.seed + episode_index
        # Keep the variable tensor shape synchronized across training ranks at
        # each local step. This avoids making faster ranks wait for a rank that
        # happened to draw a much larger episode, while the episode contents
        # remain rank-specific through sample_seed above.
        shape_seed = (
            torch.initial_seed() + sample_count // self.shape_group_size
            if self.seed is None
            else self.seed + episode_index
        )
        shape_generator = torch.Generator(device=device).manual_seed(shape_seed)
        num_bags = self.episode_generator.sample_num_bags(
            shape_generator,
            device=device,
        )
        num_cells = self.episode_generator.sample_num_cells(
            shape_generator,
            device=device,
        )
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        effect_scale_multiplier = self._sample_effect_scale(
            generator,
            device,
            sample_count,
            final_difficulty=self.seed is not None,
        )
        episode = self.episode_generator.sample_episode(
            generator,
            device=device,
            effect_scale_multiplier=effect_scale_multiplier,
            num_bags=num_bags,
            num_cells=num_cells,
        )
        return self._format_episode(episode)

    def _format_episode(self, episode: SyntheticEpisode) -> tuple[torch.Tensor, ...]:
        if not self.return_oracle_diagnostics:
            return episode.x, episode.y
        abundance = episode.oracle_response_abundance
        if abundance is None:
            raise RuntimeError(
                "Oracle abundance diagnostics require a responsive component."
            )
        return episode.x, episode.y, abundance.detach()

    def _sample_effect_scale(
        self,
        generator: torch.Generator,
        device: torch.device,
        sample_count: int,
        final_difficulty: bool,
    ) -> float:
        if final_difficulty or self.difficulty_curriculum_episodes == 0:
            progress = 1.0
        else:
            progress = min(
                sample_count / self.difficulty_curriculum_episodes,
                1.0,
            )
        low = self.effect_scale_start[0] + progress * (
            self.effect_scale_end[0] - self.effect_scale_start[0]
        )
        high = self.effect_scale_start[1] + progress * (
            self.effect_scale_end[1] - self.effect_scale_start[1]
        )
        return float(
            torch.empty((), device=device)
            .uniform_(low, high, generator=generator)
            .item()
        )

    @staticmethod
    def _as_non_negative_range(
        name: str,
        value_range: float | tuple[float, float],
    ) -> tuple[float, float]:
        if isinstance(value_range, (int, float)):
            value_range = (float(value_range), float(value_range))
        low, high = value_range
        if low < 0 or low > high:
            raise ValueError(f"{name} must be an ordered non-negative range.")
        return float(low), float(high)
