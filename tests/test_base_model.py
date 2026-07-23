import unittest

import torch
import torch.nn.functional as F

from src.models.baseline import (
    BaseModel,
    EpisodePopulationAggregator,
    MeanAggregator,
    MeanResidualAggregator,
    RidgeResidualMetaClassifier,
    SetCrossAttentionMetaClassifier,
    StructuredEpisodePopulationAggregator,
    StructuredPopulationMetaClassifier,
)


def build_small_model() -> BaseModel:
    return BaseModel(
        input_dim=8,
        meta_hidden_dim=16,
        meta_num_heads=4,
        meta_num_set_layers=1,
        meta_relation_hidden_dim=16,
        num_classes=2,
    )


class MeanAggregatorTest(unittest.TestCase):
    def test_returns_exact_bag_mean(self) -> None:
        aggregator = MeanAggregator(input_dim=3)
        x = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
                [[-1.0, 0.0, 1.0], [1.0, 2.0, 3.0]],
            ]
        )
        torch.testing.assert_close(aggregator(x), x.mean(dim=1))

    def test_is_invariant_to_instance_order(self) -> None:
        torch.manual_seed(1)
        aggregator = MeanAggregator(input_dim=8)
        x = torch.randn(5, 11, 8)
        permutation = torch.randperm(x.shape[1])
        torch.testing.assert_close(aggregator(x), aggregator(x[:, permutation]))

    def test_supports_valid_instance_mask(self) -> None:
        aggregator = MeanAggregator(input_dim=2)
        x = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])
        expected = torch.tensor([[2.0, 4.0]])
        torch.testing.assert_close(aggregator(x, mask), expected)


class MeanResidualAggregatorTest(unittest.TestCase):
    def test_zero_residual_preserves_exact_mean_base(self) -> None:
        aggregator = MeanResidualAggregator(
            input_dim=3,
            hidden_dim=4,
            tail_fractions=(0.25, 0.5),
        )
        for parameter in aggregator.residual_projection.parameters():
            torch.nn.init.zeros_(parameter)
        x = torch.randn(2, 8, 3)
        torch.testing.assert_close(aggregator(x), x.mean(dim=1))

    def test_tail_counts_scale_with_instance_count(self) -> None:
        aggregator = MeanResidualAggregator(
            input_dim=3,
            hidden_dim=4,
            tail_fractions=(0.01, 0.05, 0.15),
        )
        bags = [torch.randn(100, 3), torch.randn(1000, 3)]
        _, auxiliary = aggregator(bags, return_auxiliary=True)
        torch.testing.assert_close(
            auxiliary["tail_counts"],
            torch.tensor([[1, 5, 15], [10, 50, 150]]),
        )

    def test_ragged_bags_are_instance_order_invariant(self) -> None:
        torch.manual_seed(9)
        aggregator = MeanResidualAggregator(input_dim=8, hidden_dim=12)
        bags = [torch.randn(17, 8), torch.randn(31, 8)]
        expected = aggregator(bags)
        permuted = [bag[torch.randperm(len(bag))] for bag in bags]
        torch.testing.assert_close(expected, aggregator(permuted))


class EpisodePopulationAggregatorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(21)
        self.aggregator = EpisodePopulationAggregator(
            input_dim=8,
            num_slots=3,
            state_dim=4,
            context_samples_per_bag=7,
            tail_fractions=(0.1, 0.25),
        ).eval()
        self.bags = [torch.randn(20 + index, 8) for index in range(5)]
        self.context_mask = torch.tensor([True, True, True, False, False])

    def test_initial_token_preserves_exact_mean_base(self) -> None:
        expected = torch.stack([bag.mean(dim=0) for bag in self.bags])
        torch.testing.assert_close(
            self.aggregator(self.bags, self.context_mask), expected
        )

    def test_instance_permutation_does_not_change_population_tokens(self) -> None:
        expected = self.aggregator(self.bags, self.context_mask)
        permuted = [bag[torch.randperm(len(bag))] for bag in self.bags]
        actual = self.aggregator(permuted, self.context_mask)
        torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)

    def test_query_cells_do_not_change_context_population_anchors(self) -> None:
        _, expected = self.aggregator(
            self.bags, self.context_mask, return_auxiliary=True
        )
        changed = [*self.bags[:3], torch.randn(23, 8), torch.randn(24, 8)]
        _, actual = self.aggregator(
            changed, self.context_mask, return_auxiliary=True
        )
        torch.testing.assert_close(
            expected["population_anchors"], actual["population_anchors"]
        )

    def test_tail_counts_are_count_adaptive(self) -> None:
        _, auxiliary = self.aggregator(
            self.bags, self.context_mask, return_auxiliary=True
        )
        torch.testing.assert_close(
            auxiliary["tail_counts"][0], torch.tensor([2, 5])
        )


class StructuredEpisodePopulationAggregatorTest(unittest.TestCase):
    def test_preserves_full_state_slots_and_adaptive_tails(self) -> None:
        torch.manual_seed(22)
        aggregator = StructuredEpisodePopulationAggregator(
            input_dim=8,
            num_slots=4,
            context_samples_per_bag=6,
            tail_fractions=(0.1, 0.25),
        ).eval()
        bags = [torch.randn(20 + index, 8) for index in range(5)]
        context_mask = torch.tensor([True, True, True, False, False])
        representation, auxiliary = aggregator(
            bags, context_mask, return_auxiliary=True
        )
        self.assertEqual(representation["mean"].shape, (5, 8))
        self.assertEqual(representation["slots"].shape, (5, 4, 3, 8))
        self.assertEqual(representation["tails"].shape, (5, 2, 8))
        self.assertEqual(representation["slot_metadata"].shape, (5, 4, 2))
        torch.testing.assert_close(
            auxiliary["tail_counts"][0], torch.tensor([2, 5])
        )
        self.assertEqual(auxiliary["num_density_slots"].item(), 3)

    def test_is_invariant_to_instance_order(self) -> None:
        torch.manual_seed(23)
        aggregator = StructuredEpisodePopulationAggregator(
            input_dim=8, num_slots=4, context_samples_per_bag=6
        ).eval()
        bags = [torch.randn(20 + index, 8) for index in range(5)]
        context_mask = torch.tensor([True, True, True, False, False])
        expected = aggregator(bags, context_mask)
        actual = aggregator(
            [bag[torch.randperm(len(bag))] for bag in bags], context_mask
        )
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], atol=2e-6, rtol=2e-6)


class SetCrossAttentionMetaClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2)
        self.classifier = SetCrossAttentionMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            num_classes=2,
        ).eval()
        self.context = torch.randn(8, 8)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        self.query = torch.randn(3, 8)

    def test_context_order_does_not_change_logits(self) -> None:
        logits = self.classifier(self.context, self.labels, self.query)
        permutation = torch.randperm(self.context.shape[0])
        permuted = self.classifier(
            self.context[permutation], self.labels[permutation], self.query
        )
        torch.testing.assert_close(logits, permuted)

    def test_label_swap_only_swaps_output_columns(self) -> None:
        logits = self.classifier(self.context, self.labels, self.query)
        swapped = self.classifier(self.context, 1 - self.labels, self.query)
        torch.testing.assert_close(logits, swapped.flip(-1))

    def test_query_batching_does_not_change_each_prediction(self) -> None:
        together = self.classifier(self.context, self.labels, self.query)
        separately = torch.cat(
            [
                self.classifier(self.context, self.labels, query[None])
                for query in self.query
            ],
            dim=0,
        )
        torch.testing.assert_close(together, separately)

    def test_requires_every_context_class(self) -> None:
        with self.assertRaisesRegex(ValueError, "Every class"):
            self.classifier(
                self.context,
                torch.zeros_like(self.labels),
                self.query,
            )


class RidgeResidualMetaClassifierTest(unittest.TestCase):
    def test_label_swap_only_swaps_output_columns(self) -> None:
        torch.manual_seed(12)
        classifier = RidgeResidualMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
        ).eval()
        context = torch.randn(10, 8)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        query = torch.randn(3, 8)
        logits = classifier(context, labels, query)
        swapped = classifier(context, 1 - labels, query)
        torch.testing.assert_close(logits, swapped.flip(-1))

    def test_degenerate_context_has_finite_forward_and_backward(self) -> None:
        torch.manual_seed(13)
        classifier = RidgeResidualMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=64,
        )
        context = torch.ones(10, 8, requires_grad=True)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        query = torch.ones(3, 8, requires_grad=True)
        loss = classifier(context, labels, query).square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in classifier.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


class StructuredPopulationMetaClassifierTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(24)
        self.classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
        ).eval()
        self.context = {
            "mean": torch.randn(8, 8),
            "slots": torch.randn(8, 4, 3, 8),
            "tails": torch.randn(8, 3, 8),
            "slot_metadata": torch.randn(8, 4, 2),
        }
        self.query = {
            "mean": torch.randn(3, 8),
            "slots": torch.randn(3, 4, 3, 8),
            "tails": torch.randn(3, 3, 8),
            "slot_metadata": torch.randn(3, 4, 2),
        }
        self.query_instances = [torch.randn(13 + index, 8) for index in range(3)]
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    def test_label_swap_only_swaps_output_columns(self) -> None:
        logits = self.classifier(
            self.context, self.labels, self.query, self.query_instances
        )
        swapped = self.classifier(
            self.context, 1 - self.labels, self.query, self.query_instances
        )
        torch.testing.assert_close(logits, swapped.flip(-1))

    def test_simultaneous_slot_permutation_does_not_change_logits(self) -> None:
        expected = self.classifier(
            self.context, self.labels, self.query, self.query_instances
        )
        permutation = torch.randperm(4)
        context = {
            **self.context,
            "slots": self.context["slots"][:, permutation],
            "slot_metadata": self.context["slot_metadata"][:, permutation],
        }
        query = {
            **self.query,
            "slots": self.query["slots"][:, permutation],
            "slot_metadata": self.query["slot_metadata"][:, permutation],
        }
        actual = self.classifier(
            context, self.labels, query, self.query_instances
        )
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)

    def test_residual_gates_cannot_disconnect_specialized_paths(self) -> None:
        with torch.no_grad():
            self.classifier.population_residual_logit.fill_(-100.0)
            self.classifier.tail_residual_logit.fill_(-100.0)
        _, auxiliary = self.classifier(
            self.context,
            self.labels,
            self.query,
            self.query_instances,
            return_auxiliary=True,
        )
        torch.testing.assert_close(
            auxiliary["population_residual_scale"], torch.tensor(0.10)
        )
        torch.testing.assert_close(
            auxiliary["tail_residual_scale"], torch.tensor(0.05)
        )

    def test_rare_instance_evidence_has_finite_gradients(self) -> None:
        classifier = self.classifier.train()
        shared_tail = torch.randn(1, 3, 8)
        context = {
            **self.context,
            "tails": (
                shared_tail.expand(8, -1, -1)
                + 1e-7 * torch.randn(8, 3, 8)
            ).requires_grad_(),
        }
        query_instances = [
            bag.detach().clone().requires_grad_() for bag in self.query_instances
        ]
        logits = classifier(
            context, self.labels, self.query, query_instances
        )
        F.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
        self.assertTrue(torch.isfinite(context["tails"].grad).all())
        self.assertTrue(
            all(torch.isfinite(bag.grad).all() for bag in query_instances)
        )


class BaseModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.model = build_small_model().eval()
        self.x = torch.randn(10, 13, 8)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.mask_index = torch.tensor([8, 9])

    def test_target_labels_are_never_read(self) -> None:
        logits = self.model(self.x, self.y, self.mask_index)
        changed_y = self.y.clone()
        changed_y[self.mask_index] = 1 - changed_y[self.mask_index]
        changed = self.model(self.x, changed_y, self.mask_index)
        torch.testing.assert_close(logits, changed)

    def test_outer_episode_batch_matches_sequential_forward(self) -> None:
        batch_x = torch.stack((self.x, self.x + 0.1))
        batch_y = torch.stack((self.y, self.y.roll(2)))
        batch_mask = self.mask_index.expand(2, -1)
        expected = torch.stack([
            self.model(x, y, mask)
            for x, y, mask in zip(batch_x, batch_y, batch_mask)
        ])
        actual = self.model.forward_episode_batch(
            batch_x, batch_y, batch_mask
        )
        torch.testing.assert_close(expected, actual, atol=3e-5, rtol=3e-5)

    def test_all_cell_evidence_is_instance_order_invariant(self) -> None:
        expected = self.model(self.x, self.y, self.mask_index)
        permuted = self.x.clone()
        for bag_index in range(permuted.shape[0]):
            permuted[bag_index] = permuted[
                bag_index, torch.randperm(permuted.shape[1])
            ]
        actual = self.model(permuted, self.y, self.mask_index)
        torch.testing.assert_close(expected, actual, atol=3e-6, rtol=3e-6)

    def test_context_bag_order_does_not_change_prediction(self) -> None:
        expected = self.model(self.x, self.y, self.mask_index)
        context_permutation = torch.randperm(8)
        permutation = torch.cat((context_permutation, torch.tensor([8, 9])))
        actual = self.model(
            self.x[permutation], self.y[permutation], self.mask_index
        )
        torch.testing.assert_close(expected, actual, atol=3e-6, rtol=3e-6)

    def test_variable_length_bags_are_supported(self) -> None:
        ragged_x = [torch.randn(7 + index, 8) for index in range(10)]
        logits, auxiliary = self.model(
            ragged_x,
            self.y,
            self.mask_index,
            return_auxiliary=True,
        )
        self.assertEqual(logits.shape, (2, 2))
        torch.testing.assert_close(
            auxiliary["aggregator"]["instance_counts"],
            torch.arange(7, 17),
        )

    def test_label_swap_equivariance_is_exact(self) -> None:
        logits = self.model(self.x, self.y, self.mask_index)
        swapped = self.model(self.x, 1 - self.y, self.mask_index)
        torch.testing.assert_close(logits, swapped.flip(-1))

    def test_classification_gradient_reaches_shared_scorer(self) -> None:
        model = build_small_model().train()
        logits = model(self.x, self.y, self.mask_index)
        loss = F.cross_entropy(logits, self.y[self.mask_index])
        loss.backward()
        gradient = model.meta_classifier.slot_relation_scorer[1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm(), 0)

    def test_auxiliary_output_exposes_only_new_architecture_state(self) -> None:
        logits, auxiliary = self.model(
            self.x, self.y, self.mask_index, return_auxiliary=True
        )
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(auxiliary["bag_tokens"].shape, (10, 8))
        self.assertEqual(auxiliary["slot_tokens"].shape, (10, 12, 3, 8))
        self.assertEqual(auxiliary["tail_tokens"].shape, (10, 3, 8))
        self.assertEqual(auxiliary["slot_metadata"].shape, (10, 12, 2))
        self.assertEqual(auxiliary["population_slot_weights"].shape, (2, 36))
        self.assertTrue((auxiliary["population_slot_weights"] > 0).all())
        torch.testing.assert_close(
            auxiliary["population_slot_weights"].sum(dim=-1),
            torch.ones(2),
        )
        self.assertEqual(auxiliary["rare_counts"].shape, (2, 4))
        torch.testing.assert_close(
            auxiliary["rare_counts"],
            torch.tensor([[1, 1, 2, 3], [1, 1, 2, 3]]),
        )
        self.assertEqual(auxiliary["class_memories"].shape, (2, 8, 16))
        self.assertGreaterEqual(
            auxiliary["population_residual_scale"].item(), 0.10
        )
        self.assertGreaterEqual(auxiliary["tail_residual_scale"].item(), 0.05)
        self.assertEqual(auxiliary["cross_attention_entropy"].shape, (2, 2))
        torch.testing.assert_close(
            auxiliary["context_class_counts"], torch.tensor([4, 4])
        )


if __name__ == "__main__":
    unittest.main()
