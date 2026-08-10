from __future__ import annotations

import unittest

import torch

from src.datasets.synthetic_data import SyntheticEpisodeDataset
from src.modules.data_interface import collate_synthetic_training_episode


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


if __name__ == "__main__":
    unittest.main()
