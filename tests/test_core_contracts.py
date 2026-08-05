from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from src.datasets.base_data import ICIDataset
from src.datasets.synthetic_data import SyntheticManifoldGenerator
from src.models.baseline import BaseModel
from src.modules.data_interface import collate_synthetic_evaluation_episode
from src.modules.model_interface import ModelInterface
from src.utils.metrics import auroc, bootstrap_auroc_interval, log_loss


def build_v30_small(*, train: bool = False) -> BaseModel:
    """Small CPU model with the confirmed v30 architecture/config contracts."""
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
    return model.train() if train else model.eval()


def small_episode() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    x = torch.randn(10, 13, 8) * 2.0 + 0.5
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return x, y, torch.tensor([8, 9])


class V30ModelContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = build_v30_small()
        self.x, self.y, self.query = small_episode()

    def test_forward_backward_is_finite(self) -> None:
        model = build_v30_small(train=True)
        logits = model(self.x, self.y, self.query)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        F.cross_entropy(logits, self.y[self.query]).backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_query_labels_are_never_read(self) -> None:
        expected = self.model(self.x, self.y, self.query)
        changed = self.y.clone()
        changed[self.query] = 1 - changed[self.query]
        torch.testing.assert_close(expected, self.model(self.x, changed, self.query))

    def test_label_swap_is_equivariant(self) -> None:
        expected = self.model(self.x, self.y, self.query)
        actual = self.model(self.x, 1 - self.y, self.query)
        torch.testing.assert_close(expected, actual.flip(-1), atol=3e-5, rtol=3e-5)

    def test_cell_and_context_order_are_invariant(self) -> None:
        expected = self.model(self.x, self.y, self.query)
        cells = self.x[:, torch.randperm(self.x.shape[1])]
        torch.testing.assert_close(
            expected, self.model(cells, self.y, self.query), atol=3e-5, rtol=3e-5
        )
        permutation = torch.cat((torch.randperm(8), torch.tensor([8, 9])))
        torch.testing.assert_close(
            expected,
            self.model(self.x[permutation], self.y[permutation], self.query),
            atol=3e-5,
            rtol=3e-5,
        )

    def test_ragged_bags_are_supported(self) -> None:
        bags = [torch.randn(4 + index, 8) for index in range(10)]
        logits, auxiliary = self.model(
            bags, self.y, self.query, return_auxiliary=True
        )
        self.assertEqual(logits.shape, (2, 2))
        torch.testing.assert_close(
            auxiliary["aggregator"]["instance_counts"], torch.arange(4, 14)
        )

    def test_pool_statistics_ignore_query_bags(self) -> None:
        bags = list(self.x.unbind(0))
        context = torch.ones(10, dtype=torch.bool)
        context[self.query] = False
        before = self.model.aggregator._context_pool_stats(bags, context)
        changed = [bag.clone() for bag in bags]
        for index in self.query.tolist():
            changed[index] = changed[index] * 100.0 + 50.0
        after = self.model.aggregator._context_pool_stats(changed, context)
        torch.testing.assert_close(before[0], after[0])
        torch.testing.assert_close(before[1], after[1])

    def test_single_instance_poolz_bag_is_not_zero(self) -> None:
        bag = torch.randn(1, 8) + 1.0
        view = self.model.aggregator._bag_view(
            bag, torch.zeros(8), torch.ones(8)
        )[0]
        self.assertGreater(float(view.abs().max()), 0.0)

    def test_batched_matches_sequential_and_validates_indices(self) -> None:
        batch_x = torch.stack((self.x, self.x + 0.1))
        batch_y = torch.stack((self.y, self.y.roll(2)))
        batch_query = self.query.expand(2, -1)
        expected = torch.stack(
            [
                self.model(x, y, query)
                for x, y, query in zip(batch_x, batch_y, batch_query)
            ]
        )
        actual = self.model.forward_episode_batch(batch_x, batch_y, batch_query)
        torch.testing.assert_close(expected, actual, atol=3e-5, rtol=3e-5)
        invalid = batch_query.clone()
        invalid[0, 0] = 99
        with self.assertRaises(IndexError):
            self.model.forward_episode_batch(batch_x, batch_y, invalid)


class TrainingInterfaceContractTest(unittest.TestCase):
    def test_pairwise_ranking_prefers_correct_order(self) -> None:
        targets = torch.tensor([0, 1])
        correct = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        wrong = correct.flip(0)
        self.assertLess(
            ModelInterface._pairwise_ranking_loss(correct, targets),
            ModelInterface._pairwise_ranking_loss(wrong, targets),
        )

    def test_v30_checkpoint_marker_is_enforced(self) -> None:
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
        )
        interface.on_load_checkpoint(
            {"state_dict": {"model._architecture_version": torch.tensor(24)}}
        )
        with self.assertRaisesRegex(RuntimeError, "Expected v24, found 22"):
            interface.on_load_checkpoint(
                {"state_dict": {"model._architecture_version": torch.tensor(22)}}
            )


class SyntheticDataContractTest(unittest.TestCase):
    def test_evaluation_split_preserves_both_context_classes(self) -> None:
        x = torch.randn(10, 4, 8)
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        collated_x, collated_y, query = collate_synthetic_evaluation_episode([(x, y)])
        context = torch.ones(10, dtype=torch.bool)
        context[query] = False
        self.assertEqual(torch.unique(collated_y[context]).tolist(), [0, 1])
        torch.testing.assert_close(collated_x, x)

    def test_log_uniform_cell_count_is_seed_reproducible(self) -> None:
        generator = SyntheticManifoldGenerator(
            num_bags=8,
            num_cells=(1, 1024),
            num_cells_log_uniform=True,
            latent_dim=4,
            output_dim=8,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
        )
        first_rng = torch.Generator().manual_seed(42)
        second_rng = torch.Generator().manual_seed(42)
        first = [generator.sample_num_cells(first_rng, torch.device("cpu")) for _ in range(8)]
        second = [generator.sample_num_cells(second_rng, torch.device("cpu")) for _ in range(8)]
        self.assertEqual(first, second)
        self.assertGreater(len(set(first)), 1)

    def test_sparse_response_task_is_reachable(self) -> None:
        generator = SyntheticManifoldGenerator(
            num_bags=8,
            num_cells=16,
            latent_dim=4,
            output_dim=8,
            mlp_hidden_dim=8,
            mlp_num_layers=2,
            shared_component_probability=1.0,
            continuous_response_probability=1.0,
            response_task_probabilities=(0, 0, 0, 0, 0, 1),
            balanced=True,
        )
        episode = generator.sample_episode(torch.Generator().manual_seed(9))
        self.assertEqual(episode.response_task, "any_positive_sparse")
        self.assertTrue(torch.isfinite(episode.x).all())
        self.assertEqual(torch.unique(episode.y).tolist(), [0, 1])


class MetricsContractTest(unittest.TestCase):
    def test_auroc_and_log_loss_direction(self) -> None:
        target = torch.tensor([1, 1, 0, 0])
        good = torch.tensor([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(auroc(good, target), 1.0)
        self.assertLess(log_loss(good, target), log_loss(1.0 - good, target))

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        probability = torch.tensor([0.9, 0.7, 0.4, 0.2, 0.8, 0.3])
        target = torch.tensor([1, 1, 0, 0, 1, 0])
        episode = torch.tensor([0, 0, 1, 1, 2, 2])
        first = bootstrap_auroc_interval(
            probability, target, groups=episode, samples=100, seed=7
        )
        second = bootstrap_auroc_interval(
            probability, target, groups=episode, samples=100, seed=7
        )
        self.assertEqual(first, second)


class ICIDatasetContractTest(unittest.TestCase):
    def test_all_cell_mean_uses_every_donor_cell(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fold = Path(directory) / "SEED42" / "CV0"
            fold.mkdir(parents=True)
            torch.save(
                torch.tensor([[1.0, 3.0], [3.0, 5.0], [100.0, 200.0]]),
                fold / "val_hvg.pt",
            )
            pd.DataFrame(
                {
                    "donor_id": ["a", "a", "b"],
                    "Response": ["NR", "NR", "R"],
                }
            ).to_csv(fold / "val_donor_info.csv", index=False)
            dataset = ICIDataset(
                cv=0,
                state="val",
                root_dir=directory,
                seed=42,
                target_cells=1,
                all_cell_mean=True,
            )
            torch.testing.assert_close(dataset[0][0], torch.tensor([[2.0, 4.0]]))
            torch.testing.assert_close(
                dataset[1][0], torch.tensor([[100.0, 200.0]])
            )


if __name__ == "__main__":
    unittest.main()
