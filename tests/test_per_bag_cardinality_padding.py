from __future__ import annotations

import unittest

import torch

from src.datasets.synthetic_data import (
    SyntheticEpisodeDataset,
    SyntheticManifoldGenerator,
)
from src.modules.data_interface import (
    collate_synthetic_evaluation_episode,
    collate_synthetic_training_episode,
)


def _dataset() -> SyntheticEpisodeDataset:
    return SyntheticEpisodeDataset(
        episodes_per_epoch=2,
        seed=17,
        per_bag_cardinality=True,
        num_bags=(8, 8),
        num_cells=(1, 32),
        num_cells_log_uniform=True,
        num_cells_log_uniform_power=1.5,
        latent_dim=4,
        output_dim=8,
        mlp_hidden_dim=8,
        mlp_num_layers=2,
        shared_component_probability=1.0,
        num_shared_components=(2, 4),
        continuous_response_probability=1.0,
        response_task_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
    )


class TestPerBagCardinalityPadding(unittest.TestCase):
    def test_one_episode_draws_independent_bag_cardinalities(self) -> None:
        bags, labels = _dataset()[0][:2]
        lengths = [bag.shape[0] for bag in bags]

        self.assertEqual(len(bags), labels.numel())
        self.assertGreater(len(set(lengths)), 1)
        self.assertTrue(all(1 <= length <= 32 for length in lengths))

    def test_single_ragged_episode_is_zero_padded_with_a_mask(self) -> None:
        sample = _dataset()[0]
        padded, labels, cell_mask, bag_mask = collate_synthetic_training_episode(
            [sample]
        )

        self.assertEqual(padded.shape[:2], (1, len(sample[0])))
        self.assertEqual(padded.shape[2], max(bag.shape[0] for bag in sample[0]))
        self.assertEqual(labels.shape, bag_mask.shape)
        self.assertEqual(labels.shape, (1, len(sample[0])))
        self.assertEqual(cell_mask.shape, padded.shape[:3])
        self.assertTrue(bag_mask.all())

        for bag_index, bag in enumerate(sample[0]):
            count = bag.shape[0]
            torch.testing.assert_close(padded[0, bag_index, :count], bag)
            self.assertTrue(cell_mask[0, bag_index, :count].all())
            self.assertFalse(cell_mask[0, bag_index, count:].any())
            self.assertEqual(
                torch.count_nonzero(padded[0, bag_index, count:]).item(), 0
            )

    def test_preserve_ragged_returns_the_single_episode_without_padding(self) -> None:
        sample = _dataset()[0]
        actual = collate_synthetic_training_episode(
            [sample], preserve_ragged=True
        )
        self.assertIs(actual, sample)

    def test_preserve_ragged_rejects_episode_batching(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode_batch_size=1"):
            collate_synthetic_training_episode(
                [_dataset()[0], _dataset()[1]], preserve_ragged=True
            )

    def test_cap_is_applied_before_dense_episode_generation(self) -> None:
        generator = SyntheticManifoldGenerator(
            num_bags=2, num_cells=(256, 8192), num_cells_log_uniform=True,
            num_cells_log_uniform_power=2.0, per_bag_cardinality=True,
            per_bag_max_cells=4096, latent_dim=2, output_dim=4,
            mlp_hidden_dim=4,
        )
        dense_lengths = []
        original = generator._map_episode_manifold

        def record_dense_length(z, *args):
            dense_lengths.append(z.shape[1])
            return original(z, *args)

        generator._map_episode_manifold = record_dense_length
        episode = generator.sample_episode(
            generator=torch.Generator().manual_seed(9),
            num_bags=2, num_cells_per_bag=[8192, 256],
        )
        self.assertEqual(dense_lengths, [4096])
        self.assertEqual(sorted(bag.shape[0] for bag in episode.x), [256, 4096])

    def test_long_bag_is_randomly_truncated_to_4096_before_padding(self) -> None:
        long_bag = torch.arange(5000, dtype=torch.float32).unsqueeze(1)
        sample = ([long_bag, long_bag + 10_000], torch.tensor([0, 1]))

        torch.manual_seed(123)
        first = collate_synthetic_training_episode([sample])
        torch.manual_seed(123)
        second = collate_synthetic_training_episode([sample])

        padded, _, cell_mask, _ = first
        self.assertEqual(padded.shape, (1, 2, 4096, 1))
        self.assertTrue(cell_mask.all())
        torch.testing.assert_close(padded, second[0])
        self.assertEqual(torch.unique(padded[0, 0]).numel(), 4096)
        self.assertFalse(
            torch.equal(padded[0, 0, :, 0], torch.arange(4096).float())
        )

    def test_evaluation_cap_is_deterministic_and_remains_ragged(self) -> None:
        long_bag = torch.arange(5000, dtype=torch.float32).unsqueeze(1)
        sample = (
            [long_bag + offset for offset in (0, 10_000, 20_000, 30_000)],
            torch.tensor([0, 1, 0, 1]),
        )

        first = collate_synthetic_evaluation_episode([sample])
        second = collate_synthetic_evaluation_episode([sample])

        self.assertIsInstance(first[0], list)
        self.assertTrue(all(bag.shape[0] == 4096 for bag in first[0]))
        for first_bag, second_bag in zip(first[0], second[0]):
            torch.testing.assert_close(first_bag, second_bag)
        torch.testing.assert_close(first[2], second[2])

if __name__ == "__main__":
    unittest.main()
