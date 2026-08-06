"""Padded ragged (B2b) batching contracts.

The padded batch path (collator -> forward_episode_batch with cell/bag masks)
must produce exactly the same predictions as the per-episode ragged list path,
and the collator must pad variable bag/cell counts into a usable 4-tuple.
"""

from __future__ import annotations

import unittest

import torch

from src.datasets.synthetic_data import SyntheticManifoldGenerator
from src.models.baseline import BaseModel
from src.modules.data_interface import collate_synthetic_training_episode


def build_v30_small() -> BaseModel:
    """Small CPU model with the confirmed v30 architecture/config contracts."""
    torch.manual_seed(1234)
    return BaseModel(
        input_dim=8,
        meta_hidden_dim=16,
        meta_num_heads=4,
        meta_num_set_layers=1,
        meta_relation_hidden_dim=16,
        meta_ridge_dim=4,
        aggregator_num_slots=4,
        aggregator_num_density_slots=3,
        project_structured_tokens=True,
        projection_bottleneck_dim=4,
        projection_residual_mean=True,
        bag_representation="poolz_l2",
        num_classes=2,
    ).eval()


def make_generator() -> SyntheticManifoldGenerator:
    return SyntheticManifoldGenerator(
        latent_dim=4,
        output_dim=8,
        mlp_hidden_dim=16,
        mlp_num_layers=2,
        num_bags=(6, 10),
        num_cells=(2, 16),
        num_cells_log_uniform=True,
        per_bag_cardinality=True,
        normalize_output=True,
        shared_component_probability=0.5,
        continuous_response_probability=0.0,
        observation_noise=0.005,
    )


def sample_episode(generator: SyntheticManifoldGenerator, seed: int, num_bags: int):
    generator_ = generator
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    return generator_.sample_episode(g, device="cpu", num_bags=num_bags)


def query_split(y: torch.Tensor) -> torch.Tensor:
    """Queries that keep at least one context bag per class."""
    num_classes = 2
    protected = [
        int(torch.nonzero(y == c, as_tuple=False).flatten()[0])
        for c in range(num_classes)
    ]
    candidates = [i for i in range(y.numel()) if i not in protected]
    return torch.tensor(candidates[:2], dtype=torch.long)


class CollateRaggedBatchTest(unittest.TestCase):
    def test_single_episode_passthrough(self):
        generator = make_generator()
        episode = sample_episode(generator, 1, 8)
        result = collate_synthetic_training_episode([(episode.x, episode.y)])
        self.assertIs(result[0], episode.x)  # returned as-is, not padded

    def test_pads_variable_bags_and_cells(self):
        generator = make_generator()
        episodes = [sample_episode(generator, 10 + i, nb) for i, nb in enumerate([6, 10, 8])]
        samples = [(ep.x, ep.y) for ep in episodes]
        x, y, cell_mask, bag_mask = collate_synthetic_training_episode(samples)

        self.assertEqual(x.shape, (3, 10, 16, 8))  # max bags, max cells, dim
        self.assertEqual(y.shape, (3, 10))
        self.assertEqual(cell_mask.shape, (3, 10, 16))
        self.assertEqual(bag_mask.shape, (3, 10))

        # Real bags/cells marked, padded entries zeroed.
        for episode_index, episode in enumerate(episodes):
            n_bags = len(episode.x)
            self.assertTrue(bag_mask[episode_index, :n_bags].all())
            self.assertFalse(bag_mask[episode_index, n_bags:].any())
            for bag_index, bag in enumerate(episode.x):
                count = bag.shape[0]
                self.assertTrue(cell_mask[episode_index, bag_index, :count].all())
                self.assertFalse(cell_mask[episode_index, bag_index, count:].any())
                torch.testing.assert_close(
                    x[episode_index, bag_index, :count], bag
                )
        # Padded bag labels are the -1 sentinel (never sampled as context).
        for episode_index, n_bags in enumerate([len(ep.x) for ep in episodes]):
            self.assertTrue((y[episode_index, n_bags:] == -1).all())


class PaddedBatchEqualsListPathTest(unittest.TestCase):
    def test_padded_batch_matches_per_episode_list_path(self):
        torch.manual_seed(0)
        model = build_v30_small()
        generator = make_generator()
        episodes = [sample_episode(generator, 20 + i, nb) for i, nb in enumerate([6, 10, 8])]

        # Reference: per-episode ragged list path (slow per-bag loop).
        reference = []
        for episode in episodes:
            mask_index = query_split(episode.y)
            with torch.no_grad():
                logits = model(episode.x, episode.y, mask_index)
            reference.append(logits)

        # Padded batch path: collator -> forward_episode_batch with masks.
        x, y, cell_mask, bag_mask = collate_synthetic_training_episode(
            [(ep.x, ep.y) for ep in episodes]
        )
        mask_index = torch.stack([query_split(ep.y) for ep in episodes])
        with torch.no_grad():
            logits, auxiliary = model.forward_episode_batch(
                x,
                y,
                mask_index,
                return_auxiliary=True,
                cell_mask=cell_mask,
                bag_mask=bag_mask,
            )

        self.assertEqual(logits.shape, (3, 2, 2))
        for episode_index in range(3):
            torch.testing.assert_close(
                logits[episode_index],
                reference[episode_index],
                atol=1e-4,
                rtol=1e-4,
            )
        # Auxiliary keys survive the per-episode combination.
        self.assertIn("tail_logits", auxiliary)
        self.assertEqual(auxiliary["tail_logits"].shape[0], 3)


if __name__ == "__main__":
    unittest.main()
