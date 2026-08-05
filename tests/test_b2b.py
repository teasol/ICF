"""B2b (per-bag cardinality) contract tests.

These cover the v33 Phase 0 data-control surface: per-bag cell counts drawn
log-uniformly inside an episode, the ragged list-of-bags output, collation, and
the v30 model's behaviour on ragged training episodes. Kept CPU-only and fast
so they stay in the default `test_*.py` discovery.
"""

from __future__ import annotations

import unittest

import torch

from src.datasets.synthetic_data import (
    SyntheticEpisodeDataset,
    SyntheticManifoldGenerator,
)
from src.models.baseline import BaseModel
from src.modules.data_interface import (
    collate_synthetic_evaluation_episode,
    collate_synthetic_training_episode,
)
from src.modules.model_interface import ModelInterface


def build_b2b_generator(
    *,
    task_probabilities: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2),
    per_bag: bool = True,
) -> SyntheticManifoldGenerator:
    return SyntheticManifoldGenerator(
        num_bags=(6, 30),
        num_cells=(1, 64),
        num_cells_log_uniform=True,
        per_bag_cardinality=per_bag,
        latent_dim=4,
        output_dim=8,
        mlp_hidden_dim=8,
        mlp_num_layers=2,
        shared_component_probability=1.0,
        num_shared_components=(2, 4),
        continuous_response_probability=1.0,
        response_task_probabilities=task_probabilities,
    )


def build_v30_small() -> BaseModel:
    torch.manual_seed(1234)
    model = BaseModel(
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
    )
    return model.eval()


class B2bGeneratorContractTest(unittest.TestCase):
    def test_episode_is_ragged_with_valid_per_bag_sizes(self) -> None:
        generator = build_b2b_generator()
        episode = generator.sample_episode(
            torch.Generator().manual_seed(7)
        )
        self.assertIsInstance(episode.x, list)
        self.assertEqual(len(episode.x), episode.y.shape[0])
        for bag in episode.x:
            self.assertEqual(bag.ndim, 2)
            self.assertEqual(bag.shape[-1], 8)
            self.assertGreaterEqual(bag.shape[0], 1)
            self.assertLessEqual(bag.shape[0], 64)

    def test_episode_mixes_bag_sizes(self) -> None:
        generator = build_b2b_generator()
        episode = generator.sample_episode(
            torch.Generator().manual_seed(11)
        )
        sizes = [bag.shape[0] for bag in episode.x]
        self.assertGreater(len(set(sizes)), 1)

    def test_is_seed_reproducible(self) -> None:
        generator = build_b2b_generator()
        first = generator.sample_episode(torch.Generator().manual_seed(42))
        second = generator.sample_episode(torch.Generator().manual_seed(42))
        self.assertTrue(
            all(
                torch.equal(a, b)
                for a, b in zip(first.x, second.x)
            )
        )
        torch.testing.assert_close(first.y, second.y)

    def test_explicit_per_bag_sizes_are_honoured(self) -> None:
        generator = build_b2b_generator()
        requested = [1, 3, 8, 16, 32, 64]
        episode = generator.sample_episode(
            torch.Generator().manual_seed(5),
            num_bags=len(requested),
            num_cells_per_bag=requested,
        )
        # sample_episode randomly permutes bag order; sizes are preserved as a
        # multiset.
        self.assertEqual(
            sorted(bag.shape[0] for bag in episode.x), sorted(requested)
        )

    def test_dense_path_is_unchanged_when_disabled(self) -> None:
        """per_bag_cardinality=False keeps the dense tensor output contract."""
        dense = build_b2b_generator(per_bag=False)
        episode = dense.sample_episode(
            torch.Generator().manual_seed(3),
            num_bags=6,
            num_cells=8,
        )
        self.assertIsInstance(episode.x, torch.Tensor)
        self.assertEqual(tuple(episode.x.shape), (6, 8, 8))
        self.assertEqual(tuple(episode.y.shape), (6,))

    def test_positive_sparse_bags_retain_effect_cells(self) -> None:
        generator = build_b2b_generator(
            task_probabilities=(0, 0, 0, 0, 0, 1.0)
        )
        seen = False
        for seed in range(30):
            episode = generator.sample_episode(
                torch.Generator().manual_seed(seed)
            )
            if (episode.y == 1).any():
                positive_fraction = episode.effect_cell_fraction[episode.y == 1]
                self.assertGreaterEqual(float(positive_fraction.min()), 0.0)
                # Every positive bag must keep at least one shifted cell.
                self.assertGreater(float(positive_fraction.max()), 0.0)
                self.assertGreater(float(positive_fraction.min()), 0.0)
                seen = True
                break
        self.assertTrue(seen)


class B2bPipelineContractTest(unittest.TestCase):
    def test_dataset_returns_ragged_and_both_collators_work(self) -> None:
        dataset = SyntheticEpisodeDataset(
            episodes_per_epoch=8,
            seed=3,
            fixed_episode_count=8,
            per_bag_cardinality=True,
            num_bags=(6, 30),
            num_cells=(1, 64),
            num_cells_log_uniform=True,
            latent_dim=4,
            output_dim=8,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
            shared_component_probability=1.0,
            num_shared_components=(2, 4),
            continuous_response_probability=1.0,
            response_task_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
        )
        sample = dataset[0]
        x, y = sample[:2]
        self.assertIsInstance(x, list)
        self.assertEqual(len(x), y.shape[0])

        # Single-episode batch passes through unchanged (episode_batch_size=1).
        batch = collate_synthetic_training_episode([sample])
        self.assertIs(batch, sample)

        # Evaluation collator must build a valid context/query split on ragged x.
        eval_batch = collate_synthetic_evaluation_episode([sample])
        eval_x, eval_y, query = eval_batch[:3]
        self.assertIsInstance(eval_x, list)
        self.assertEqual(eval_x, x)
        self.assertEqual(len(query), query.numel())
        context = torch.ones(len(eval_y), dtype=torch.bool)
        context[query] = False
        self.assertEqual(
            torch.unique(eval_y[context], sorted=True).tolist(), [0, 1]
        )

    def test_collate_rejects_multi_episode_ragged_batch(self) -> None:
        dataset = SyntheticEpisodeDataset(
            episodes_per_epoch=4,
            seed=3,
            fixed_episode_count=4,
            per_bag_cardinality=True,
            num_bags=(6, 20),
            num_cells=(1, 32),
            num_cells_log_uniform=True,
            latent_dim=4,
            output_dim=8,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
            shared_component_probability=1.0,
            num_shared_components=(2, 4),
            continuous_response_probability=1.0,
            response_task_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
        )
        with self.assertRaisesRegex(ValueError, "episode_batch_size=1"):
            collate_synthetic_training_episode([dataset[0], dataset[1]])

    def test_v30_model_forward_is_finite_on_ragged_episode(self) -> None:
        generator = build_b2b_generator()
        episode = generator.sample_episode(
            torch.Generator().manual_seed(9),
            num_bags=10,
            num_cells_per_bag=[1, 2, 4, 8, 16, 8, 4, 2, 3, 5],
        )
        model = build_v30_small()
        query = torch.tensor([8, 9])
        logits, auxiliary = model(
            episode.x, episode.y, query, return_auxiliary=True
        )
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(
            sorted(auxiliary["aggregator"]["instance_counts"].tolist()),
            sorted([1, 2, 4, 8, 16, 8, 4, 2, 3, 5]),
        )

    def test_training_episode_loss_handles_ragged_input(self) -> None:
        generator = build_b2b_generator()
        episode = generator.sample_episode(
            torch.Generator().manual_seed(13),
            num_bags=12,
            num_cells_per_bag=[1, 2, 4, 8, 16, 32, 8, 4, 2, 3, 5, 7],
        )
        interface = ModelInterface(
            model_src="src.models.baseline.BaseModel",
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
            training_targets_per_episode=2,
        )
        interface.eval()
        mask = interface._sample_training_queries(episode.y)
        loss, terms = interface._episode_losses(
            episode.x, episode.y, mask
        )
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("ce_loss", terms)


if __name__ == "__main__":
    unittest.main()
