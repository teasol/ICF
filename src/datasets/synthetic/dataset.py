from __future__ import annotations
import math
from typing import Any, Sequence
import torch
from torch.utils.data import Dataset
from src.datasets.synthetic.types import SyntheticEpisode, RESPONSE_TASK_NAMES
from src.datasets.synthetic.generator import SyntheticManifoldGenerator
class SyntheticEpisodeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map-style dataset that generates a fresh episode for every item access."""

    def __init__(
        self,
        episodes_per_epoch: int = 1000,
        seed: int | None = None,
        fixed_episode_count: int | None = None,
        generation_device: str = "cpu",
        shape_group_size: int = 1,
        parallel_cuda_generation: bool = True,
        difficulty_curriculum_episodes: int = 0,
        effect_scale_start: float | tuple[float, float] = (1.0, 1.0),
        effect_scale_end: float | tuple[float, float] = (1.0, 1.0),
        return_oracle_diagnostics: bool = False,
        return_task_metadata: bool = False,
        training_context_sizes: Sequence[int] | None = None,
        training_context_jitter: int = 0,
        training_query_range: Sequence[int] = (1, 1),
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
        if training_context_jitter < 0:
            raise ValueError("training_context_jitter cannot be negative.")
        if len(training_query_range) != 2:
            raise ValueError("training_query_range must be [min_queries, max_queries].")
        min_queries, max_queries = map(int, training_query_range)
        if not 1 <= min_queries <= max_queries:
            raise ValueError("training_query_range must be an ordered positive range.")
        context_sizes = (
            None
            if training_context_sizes is None
            else tuple(int(size) for size in training_context_sizes)
        )
        if context_sizes is not None and (
            not context_sizes
            or len(set(context_sizes)) != len(context_sizes)
            or any(size - training_context_jitter < 2 for size in context_sizes)
        ):
            raise ValueError(
                "training_context_sizes must be unique and remain at least two "
                "after applying training_context_jitter."
            )
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
        self.parallel_cuda_generation = bool(parallel_cuda_generation)
        self.effect_scale_end = effect_scale_end
        self.return_oracle_diagnostics = bool(return_oracle_diagnostics)
        self.return_task_metadata = bool(return_task_metadata)
        self.training_context_sizes = context_sizes
        self.training_context_jitter = int(training_context_jitter)
        self.training_query_range = (min_queries, max_queries)
        self._sample_count = 0
        self.episode_generator = SyntheticManifoldGenerator(**generator_kwargs)
        if context_sizes is not None:
            required_min = min(context_sizes) - training_context_jitter + min_queries
            required_max = max(context_sizes) + training_context_jitter + max_queries
            configured_min, configured_max = self.episode_generator.num_bags
            if configured_min > required_min or configured_max < required_max:
                raise ValueError(
                    "num_bags must cover every configured context/query combination: "
                    f"required [{required_min}, {required_max}], configured "
                    f"[{configured_min}, {configured_max}]."
                )

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
            or not self.parallel_cuda_generation
        ):
            return [self[index] for index in indices]
        start = self._sample_count
        self._sample_count += len(indices)
        device = self._generation_device()
        shape_seed = torch.initial_seed() + start // self.shape_group_size
        shape_generator = torch.Generator(device=device).manual_seed(shape_seed)
        num_bags = self._sample_num_bags(shape_generator, device)
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
        num_bags = self._sample_num_bags(shape_generator, device)
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

    def diagnostic_episode(self, index: int) -> SyntheticEpisode:
        """Return a fixed episode with oracle fields for diagnostic scripts only."""
        if self.seed is None:
            raise ValueError("diagnostic_episode requires a fixed dataset seed.")
        device = self._generation_device()
        episode_index = (
            index % self.fixed_episode_count
            if self.fixed_episode_count is not None
            else index
        )
        sample_seed = self.seed + episode_index
        shape_generator = torch.Generator(device=device).manual_seed(sample_seed)
        num_bags = self.episode_generator.sample_num_bags(shape_generator, device=device)
        num_cells = self.episode_generator.sample_num_cells(shape_generator, device=device)
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        effect_scale_multiplier = self._sample_effect_scale(
            generator, device, 0, final_difficulty=True
        )
        return self.episode_generator.sample_episode(
            generator, device=device,
            effect_scale_multiplier=effect_scale_multiplier,
            num_bags=num_bags, num_cells=num_cells,
        )

    def _sample_num_bags(
        self,
        generator: torch.Generator,
        device: torch.device,
    ) -> int:
        """Sample a shape, optionally from a context-centred training mixture."""
        if self.training_context_sizes is None:
            return self.episode_generator.sample_num_bags(generator, device=device)
        center_index = int(
            torch.randint(
                len(self.training_context_sizes),
                (),
                generator=generator,
                device=device,
            ).item()
        )
        center = self.training_context_sizes[center_index]
        jitter = int(
            torch.randint(
                -self.training_context_jitter,
                self.training_context_jitter + 1,
                (),
                generator=generator,
                device=device,
            ).item()
        )
        min_queries, max_queries = self.training_query_range
        queries = int(
            torch.randint(
                min_queries,
                max_queries + 1,
                (),
                generator=generator,
                device=device,
            ).item()
        )
        return center + jitter + queries

    def _format_episode(self, episode: SyntheticEpisode) -> tuple[torch.Tensor, ...]:
        fields: list[torch.Tensor] = [episode.x, episode.y]
        if self.return_oracle_diagnostics:
            abundance = episode.oracle_response_abundance
            if abundance is None:
                raise RuntimeError(
                    "Oracle abundance diagnostics require a responsive component."
                )
            fields.append(abundance.detach())
        if self.return_task_metadata:
            if episode.response_task not in RESPONSE_TASK_NAMES:
                raise RuntimeError("Task diagnostics require a known response task.")
            fields.append(
                torch.tensor(
                    RESPONSE_TASK_NAMES.index(episode.response_task),
                    dtype=torch.long,
                    device=episode.y.device,
                )
            )
        return tuple(fields)

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
