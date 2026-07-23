import os
import unittest
from unittest.mock import patch

import torch

from src.datasets.synthetic_data import (
    SyntheticEpisodeDataset,
    SyntheticManifoldGenerator,
)
from src.modules.data_interface import collate_synthetic_evaluation_episode


def build_generator(
    num_cells: int | tuple[int, int],
    num_bags: int | tuple[int, int] = 6,
) -> SyntheticManifoldGenerator:
    return SyntheticManifoldGenerator(
        num_bags=num_bags,
        num_cells=num_cells,
        latent_dim=4,
        output_dim=8,
        mlp_hidden_dim=8,
        mlp_num_layers=2,
    )


class VariableCellCountTest(unittest.TestCase):
    def test_minimum_problem_is_mean_separable(self) -> None:
        generator = SyntheticManifoldGenerator(
            num_bags=24,
            num_cells=100,
            latent_dim=8,
            output_dim=64,
            mlp_hidden_dim=32,
            mlp_num_layers=1,
            class_separation=(3.0, 5.0),
            latent_scale=(0.5, 0.8),
            shared_component_probability=0.0,
            donor_shift_scale=0.05,
            donor_component_shift_scale=0.0,
            observation_noise=0.005,
            normalize_output=True,
            balanced=True,
        )
        accuracies = []
        for seed in range(8):
            episode = generator.sample_episode(torch.Generator().manual_seed(seed))
            bag_mean = episode.x.mean(dim=1)
            prototypes = torch.stack(
                [bag_mean[episode.y == label].mean(dim=0) for label in (0, 1)]
            )
            logits = torch.nn.functional.normalize(bag_mean, dim=-1) @ (
                torch.nn.functional.normalize(prototypes, dim=-1).T
            )
            accuracies.append((logits.argmax(dim=-1) == episode.y).float().mean())
        self.assertGreater(torch.stack(accuracies).mean(), 0.95)

    def test_evaluation_split_keeps_queries_out_of_context(self) -> None:
        x = torch.randn(10, 4, 8)
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        collated_x, collated_y, query_index = (
            collate_synthetic_evaluation_episode([(x, y)])
        )
        self.assertEqual(query_index.numel(), 2)
        context_mask = torch.ones(10, dtype=torch.bool)
        context_mask[query_index] = False
        self.assertEqual(torch.unique(collated_y[context_mask]).tolist(), [0, 1])
        torch.testing.assert_close(collated_x, x)

    def test_oracle_metadata_is_opt_in_detached_and_aligned(self) -> None:
        kwargs = dict(
            episodes_per_epoch=1,
            seed=41,
            generation_device="cpu",
            num_bags=64,
            num_cells=256,
            latent_dim=4,
            output_dim=8,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
            shared_component_probability=1.0,
            continuous_response_probability=1.0,
            response_task_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
            response_score_min_margin=0.5,
            response_mixture_effect_scale=2.0,
            balanced=True,
        )
        plain = SyntheticEpisodeDataset(**kwargs)[0]
        diagnostic = SyntheticEpisodeDataset(
            **kwargs, return_oracle_diagnostics=True
        )[0]
        self.assertEqual(len(plain), 2)
        self.assertEqual(len(diagnostic), 3)
        x, y, abundance = diagnostic
        torch.testing.assert_close(x, plain[0])
        torch.testing.assert_close(y, plain[1])
        self.assertEqual(abundance.shape, y.shape)
        self.assertFalse(abundance.requires_grad)
        self.assertGreater(abundance[y == 1].mean(), abundance[y == 0].mean())

        collated = collate_synthetic_evaluation_episode([diagnostic])
        self.assertEqual(len(collated), 4)
        torch.testing.assert_close(collated[3], abundance)

    def test_disabled_curriculum_uses_validation_difficulty(self) -> None:
        dataset = SyntheticEpisodeDataset(
            episodes_per_epoch=1,
            seed=None,
            difficulty_curriculum_episodes=0,
            effect_scale_start=(0.8, 1.4),
            effect_scale_end=(0.25, 0.25),
            num_bags=6,
            num_cells=10,
            latent_dim=4,
            output_dim=8,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
        )
        scale = dataset._sample_effect_scale(
            torch.Generator().manual_seed(1),
            torch.device("cpu"),
            sample_count=0,
            final_difficulty=False,
        )
        self.assertEqual(scale, 0.25)

    def test_output_can_be_unit_normalized(self) -> None:
        generator = SyntheticManifoldGenerator(
            num_bags=6,
            num_cells=20,
            latent_dim=4,
            output_dim=16,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
            normalize_output=True,
        )
        episode = generator.sample_episode(torch.Generator().manual_seed(3))
        norms = episode.x.norm(dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_combined_response_task_is_reachable(self) -> None:
        generator = SyntheticManifoldGenerator(
            num_bags=8,
            num_cells=20,
            latent_dim=4,
            output_dim=8,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
            shared_component_probability=1.0,
            continuous_response_probability=1.0,
            response_task_probabilities=(0.0, 0.0, 0.0, 0.0, 1.0),
        )
        episode = generator.sample_episode(torch.Generator().manual_seed(7))
        self.assertEqual(episode.response_task, "combined")
        self.assertEqual(
            episode.oracle_population_features.shape,
            (8, 1 + 2 * generator.output_dim),
        )
        self.assertTrue(torch.isfinite(episode.oracle_population_features).all())

    def test_donor_shared_component_mixtures_are_reproducible(self) -> None:
        generator = SyntheticManifoldGenerator(
            num_bags=8,
            num_cells=100,
            latent_dim=4,
            output_dim=8,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
            shared_component_probability=1.0,
            num_shared_components=4,
            shared_component_fraction=1.0,
            donor_shift_scale=0.0,
            donor_component_shift_scale=0.0,
            shared_component_base_logit_scale=0.5,
            donor_shared_component_logit_scale=1.0,
        )
        first = generator.sample_episode(torch.Generator().manual_seed(11))
        second = generator.sample_episode(torch.Generator().manual_seed(11))
        torch.testing.assert_close(first.x, second.x)
        self.assertGreater(first.x.mean(dim=1).var(dim=0).mean().item(), 0.0)

    def test_bag_count_varies_between_episodes(self) -> None:
        generator = build_generator(10, num_bags=(6, 10))
        observed = {
            generator.sample_episode(
                torch.Generator().manual_seed(seed)
            ).x.shape[0]
            for seed in range(16)
        }

        self.assertGreater(len(observed), 1)
        self.assertTrue(all(6 <= count <= 10 for count in observed))

    def test_cell_count_varies_between_episodes(self) -> None:
        generator = build_generator((8, 12))
        observed = {
            generator.sample_episode(
                torch.Generator().manual_seed(seed)
            ).x.shape[1]
            for seed in range(16)
        }

        self.assertGreater(len(observed), 1)
        self.assertTrue(all(8 <= count <= 12 for count in observed))

    def test_cell_count_is_reproducible_from_seed(self) -> None:
        generator = build_generator((8, 12))
        first = generator.sample_episode(torch.Generator().manual_seed(17))
        second = generator.sample_episode(torch.Generator().manual_seed(17))

        self.assertEqual(first.x.shape, second.x.shape)
        torch.testing.assert_close(first.x, second.x)
        torch.testing.assert_close(first.y, second.y)

    def test_fixed_integer_cell_count_remains_supported(self) -> None:
        generator = build_generator(10)
        episode = generator.sample_episode(torch.Generator().manual_seed(3))

        self.assertEqual(episode.x.shape, (6, 10, 8))

    def test_training_ranks_draw_the_same_shape_but_different_content(self) -> None:
        kwargs = {
            "episodes_per_epoch": 2,
            "seed": None,
            "generation_device": "cpu",
            "num_bags": 6,
            "num_cells": (8, 12),
            "latent_dim": 4,
            "output_dim": 8,
            "mlp_hidden_dim": 8,
            "mlp_num_layers": 2,
        }
        torch.manual_seed(23)
        rank_0_dataset = SyntheticEpisodeDataset(**kwargs)
        with patch.dict(os.environ, {"RANK": "0"}):
            rank_0_episode = rank_0_dataset[0]

        torch.manual_seed(23)
        rank_1_dataset = SyntheticEpisodeDataset(**kwargs)
        with patch.dict(os.environ, {"RANK": "1"}):
            rank_1_episode = rank_1_dataset[0]

        self.assertEqual(rank_0_episode[0].shape, rank_1_episode[0].shape)
        self.assertFalse(torch.equal(rank_0_episode[0], rank_1_episode[0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_outer_batch_generation_matches_sequential_stream(self) -> None:
        kwargs = {
            "episodes_per_epoch": 8,
            "seed": None,
            "generation_device": "cuda",
            "shape_group_size": 8,
            "num_bags": (6, 8),
            "num_cells": (8, 12),
            "latent_dim": 4,
            "output_dim": 8,
            "mlp_hidden_dim": 8,
            "mlp_num_layers": 2,
        }
        torch.manual_seed(31)
        sequential_dataset = SyntheticEpisodeDataset(**kwargs)
        sequential = [sequential_dataset[index] for index in range(8)]

        torch.manual_seed(31)
        batched_dataset = SyntheticEpisodeDataset(**kwargs)
        batched = batched_dataset.__getitems__(list(range(8)))

        self.assertEqual(batched_dataset._sample_count, 8)
        self.assertEqual(
            {tuple(x.shape) for x, _ in batched},
            {tuple(batched[0][0].shape)},
        )
        for sequential_episode, batched_episode in zip(sequential, batched):
            torch.testing.assert_close(
                sequential_episode[0], batched_episode[0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                sequential_episode[1], batched_episode[1], rtol=0, atol=0
            )

if __name__ == "__main__":
    unittest.main()
