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

    def test_centered_view_is_translation_invariant(self) -> None:
        torch.manual_seed(25)
        aggregator = StructuredEpisodePopulationAggregator(
            input_dim=8, num_slots=4, context_samples_per_bag=6
        ).eval()
        bag = torch.randn(5, 23, 8)
        shifted = bag + torch.randn(5, 1, 8)
        centered, spread, delta = aggregator._bag_view(bag)
        shifted_centered, shifted_spread, shifted_delta = aggregator._bag_view(shifted)
        torch.testing.assert_close(
            delta.float().mean(dim=1), torch.zeros(5, 8), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(centered, shifted_centered, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(spread, shifted_spread, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(delta, shifted_delta, atol=2e-6, rtol=2e-6)

    def test_covariance_sketch_is_shift_and_instance_order_invariant(self) -> None:
        torch.manual_seed(29)
        aggregator = StructuredEpisodePopulationAggregator(
            input_dim=8, num_slots=4, context_samples_per_bag=6
        ).eval()
        bags = torch.randn(5, 23, 8)
        delta = aggregator._bag_view(bags)[2]
        shifted_delta = aggregator._bag_view(bags + torch.randn(5, 1, 8))[2]
        expected = aggregator._covariance_sketch(delta)
        shifted = aggregator._covariance_sketch(shifted_delta)
        permuted = aggregator._covariance_sketch(delta[:, torch.randperm(23)])
        self.assertEqual(expected.shape, (5, 36))
        torch.testing.assert_close(expected, shifted, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(expected, permuted, atol=2e-6, rtol=2e-6)


    def test_covariance_modes_are_shift_invariant_and_finite(self) -> None:
        torch.manual_seed(30)
        bags = torch.randn(4, 29, 8)
        shifts = torch.randn(4, 1, 8)
        for mode, expected_width in (
            ("correlation", 36),
            ("log_correlation", 36),
            ("covariance_log_correlation", 72),
        ):
            aggregator = StructuredEpisodePopulationAggregator(
                input_dim=8,
                num_slots=4,
                context_samples_per_bag=6,
                covariance_mode=mode,
                covariance_shrinkage=0.1,
            ).eval()
            expected = aggregator._covariance_sketch(aggregator._bag_view(bags)[2])
            actual = aggregator._covariance_sketch(
                aggregator._bag_view(bags + shifts)[2]
            )
            self.assertEqual(expected.shape, (4, expected_width))
            self.assertTrue(torch.isfinite(expected).all())
            torch.testing.assert_close(expected, actual, atol=5e-5, rtol=5e-5)

    def test_invalid_centering_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Centered v19 mode"):
            StructuredEpisodePopulationAggregator(
                input_dim=8,
                bag_centered_representation=True,
                global_summary="raw_mean",
                use_raw_mean_branch=False,
            )

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
        _, actual = self.aggregator(changed, self.context_mask, return_auxiliary=True)
        torch.testing.assert_close(
            expected["population_anchors"], actual["population_anchors"]
        )

    def test_tail_counts_are_count_adaptive(self) -> None:
        _, auxiliary = self.aggregator(
            self.bags, self.context_mask, return_auxiliary=True
        )
        torch.testing.assert_close(auxiliary["tail_counts"][0], torch.tensor([2, 5]))


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
        self.assertEqual(representation["global_summary"].shape, (5, 8))
        self.assertEqual(representation["slots"].shape, (5, 4, 3, 8))
        self.assertEqual(representation["tails"].shape, (5, 2, 8))
        self.assertEqual(representation["slot_metadata"].shape, (5, 4, 2))
        torch.testing.assert_close(auxiliary["tail_counts"][0], torch.tensor([2, 5]))
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
            torch.testing.assert_close(
                expected[name], actual[name], atol=1e-5, rtol=1e-5
            )


    def test_local_geometry_is_shift_and_instance_order_invariant(self) -> None:
        torch.manual_seed(230)
        aggregator = StructuredEpisodePopulationAggregator(
            input_dim=8, num_slots=4, context_samples_per_bag=8,
            covariance_sketch_dim=8,
        ).eval()
        raw = torch.randn(2, 24, 8)
        centered, _, _ = aggregator._bag_view(raw)
        expected = aggregator._local_geometry_sketch(
            centered, neighbor_counts=(4, 8)
        )
        shift = torch.randn(2, 1, 8)
        shifted, _, _ = aggregator._bag_view(raw + shift)
        permutation = torch.randperm(raw.shape[1])
        actual = aggregator._local_geometry_sketch(
            shifted[:, permutation], neighbor_counts=(4, 8)
        )
        for name in expected:
            self.assertTrue(torch.isfinite(actual[name]).all())
            torch.testing.assert_close(expected[name], actual[name], atol=2e-5, rtol=2e-5)

    def test_spherical_kmeans_anchors_are_context_order_invariant(self) -> None:
        torch.manual_seed(231)
        aggregator = StructuredEpisodePopulationAggregator(
            input_dim=8, num_slots=4, context_samples_per_bag=8
        ).eval()
        bags = [torch.randn(24 + index, 8) for index in range(4)]
        context_mask = torch.ones(4, dtype=torch.bool)
        expected = aggregator._context_spherical_kmeans_anchors(
            bags, context_mask, num_slots=4
        )
        permutation = torch.tensor([2, 0, 3, 1])
        actual = aggregator._context_spherical_kmeans_anchors(
            [bags[index][torch.randperm(len(bags[index]))] for index in permutation],
            context_mask, num_slots=4,
        )
        torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)

    def test_slot_spectral_descriptor_is_rotation_invariant(self) -> None:
        aggregator = StructuredEpisodePopulationAggregator(
            input_dim=8, num_slots=4, num_density_slots=3,
            context_samples_per_bag=8, covariance_sketch_dim=8,
        )
        aggregator.slot_covariance_descriptor = "spectral"
        delta = torch.randn(2, 31, 8)
        assignment = torch.softmax(torch.randn(2, 31, 4), dim=-1)
        orthogonal, _ = torch.linalg.qr(torch.randn(8, 8))
        expected, reliability = aggregator._slot_covariance_sketch(assignment, delta)
        actual, rotated_reliability = aggregator._slot_covariance_sketch(
            assignment, delta @ orthogonal
        )
        torch.testing.assert_close(expected, actual, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(reliability, rotated_reliability)


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
            "global_summary": torch.randn(8, 8),
            "slots": torch.randn(8, 4, 3, 8),
            "tails": torch.randn(8, 3, 8),
            "slot_metadata": torch.randn(8, 4, 2),
            "covariance_sketch": torch.randn(8, 36),
            "slot_covariance_sketch": torch.randn(8, 4, 12),
            "slot_covariance_reliability": torch.rand(8, 4).add(0.1),
            "covariance_matrix": torch.eye(6).repeat(8, 1, 1),
        }
        self.query = {
            "global_summary": torch.randn(3, 8),
            "slots": torch.randn(3, 4, 3, 8),
            "tails": torch.randn(3, 3, 8),
            "slot_metadata": torch.randn(3, 4, 2),
            "covariance_sketch": torch.randn(3, 36),
            "slot_covariance_sketch": torch.randn(3, 4, 12),
            "slot_covariance_reliability": torch.rand(3, 4).add(0.1),
            "covariance_matrix": torch.eye(6).repeat(3, 1, 1),
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

    def test_mean_pooling_collapses_all_structured_tokens_per_bag(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            mean_pool_structured_tokens=True,
        ).eval()
        expected = classifier._all_structured_tokens(self.query).mean(
            dim=1, keepdim=True
        )
        actual = classifier._population_tokens(self.query)
        self.assertEqual(actual.shape, (3, 1, 8))
        torch.testing.assert_close(actual, expected)

    def test_mean_pooling_preserves_one_context_item_per_bag(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            mean_pool_structured_tokens=True,
        ).eval()
        captured_shape: list[tuple[int, ...]] = []

        def capture_shape(_module, inputs):
            captured_shape.append(tuple(inputs[0].shape))

        handle = classifier.memory_input_norm.register_forward_pre_hook(capture_shape)
        try:
            classifier._class_memories(self.context, self.labels)
        finally:
            handle.remove()
        # Eight context bags become eight items; without mean pooling this is
        # 8 * (1 + 4 * 3 + 3) = 128 items.
        self.assertEqual(captured_shape, [(4, 8), (4, 8)])

    def test_batched_mean_pooling_preserves_one_context_item_per_bag(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            mean_pool_structured_tokens=True,
        ).eval()
        context = {
            name: value.unsqueeze(0).expand(2, *value.shape)
            for name, value in self.context.items()
        }
        labels = self.labels.unsqueeze(0).expand(2, -1)
        captured_shape: list[tuple[int, ...]] = []

        def capture_shape(_module, inputs):
            captured_shape.append(tuple(inputs[0].shape))

        handle = classifier.memory_input_norm.register_forward_pre_hook(capture_shape)
        try:
            memories = classifier._class_memories_batched(context, labels)
        finally:
            handle.remove()
        self.assertEqual(captured_shape, [(2, 8, 8)])
        self.assertEqual(memories.shape, (2, 2, 8, 16))

    def test_projection_collapses_all_structured_tokens_per_bag(self) -> None:
        # 4 slots -> 1 global + 4*3 slot statistics + 3 tails = 16 tokens.
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            project_structured_tokens=True,
            structured_tokens_per_bag=16,
        ).eval()
        tokens = classifier._all_structured_tokens(self.query)
        expected = classifier.bag_token_projection(
            tokens.reshape(3, -1)
        ).unsqueeze(1)
        actual = classifier._population_tokens(self.query)
        self.assertEqual(actual.shape, (3, 1, 8))
        torch.testing.assert_close(actual, expected)

    def test_projection_preserves_one_context_item_per_bag(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            project_structured_tokens=True,
            structured_tokens_per_bag=16,
        ).eval()
        captured_shape: list[tuple[int, ...]] = []

        def capture_shape(_module, inputs):
            captured_shape.append(tuple(inputs[0].shape))

        handle = classifier.memory_input_norm.register_forward_pre_hook(capture_shape)
        try:
            classifier._class_memories(self.context, self.labels)
        finally:
            handle.remove()
        # Eight context bags become eight items; without pooling this is
        # 8 * (1 + 4 * 3 + 3) = 128 items.
        self.assertEqual(captured_shape, [(4, 8), (4, 8)])

    def test_batched_projection_preserves_one_context_item_per_bag(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            project_structured_tokens=True,
            structured_tokens_per_bag=16,
        ).eval()
        context = {
            name: value.unsqueeze(0).expand(2, *value.shape)
            for name, value in self.context.items()
        }
        labels = self.labels.unsqueeze(0).expand(2, -1)
        captured_shape: list[tuple[int, ...]] = []

        def capture_shape(_module, inputs):
            captured_shape.append(tuple(inputs[0].shape))

        handle = classifier.memory_input_norm.register_forward_pre_hook(capture_shape)
        try:
            memories = classifier._class_memories_batched(context, labels)
        finally:
            handle.remove()
        self.assertEqual(captured_shape, [(2, 8, 8)])
        self.assertEqual(memories.shape, (2, 2, 8, 16))

    def test_bottleneck_projection_collapses_all_structured_tokens_per_bag(self) -> None:
        # 4 slots -> 1 global + 4*3 slot statistics + 3 tails = 16 tokens.
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            project_structured_tokens=True,
            structured_tokens_per_bag=16,
            projection_bottleneck_dim=4,
        ).eval()
        tokens = classifier._all_structured_tokens(self.query)
        compressed = torch.stack(
            [
                classifier.bag_token_bottlenecks[index](tokens[..., index, :])
                for index in range(16)
            ],
            dim=-2,
        )
        expected = classifier.bag_token_projection(
            compressed.reshape(3, -1)
        ).unsqueeze(1)
        actual = classifier._population_tokens(self.query)
        self.assertEqual(actual.shape, (3, 1, 8))
        torch.testing.assert_close(actual, expected)

    def test_bottleneck_projection_preserves_one_context_item_per_bag(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            project_structured_tokens=True,
            structured_tokens_per_bag=16,
            projection_bottleneck_dim=4,
        ).eval()
        captured_shape: list[tuple[int, ...]] = []

        def capture_shape(_module, inputs):
            captured_shape.append(tuple(inputs[0].shape))

        handle = classifier.memory_input_norm.register_forward_pre_hook(capture_shape)
        try:
            classifier._class_memories(self.context, self.labels)
        finally:
            handle.remove()
        self.assertEqual(captured_shape, [(4, 8), (4, 8)])

    def test_bottleneck_projection_uses_per_token_bottlenecks(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            project_structured_tokens=True,
            structured_tokens_per_bag=16,
            projection_bottleneck_dim=4,
        ).eval()
        self.assertEqual(len(classifier.bag_token_bottlenecks), 16)
        for layer in classifier.bag_token_bottlenecks:
            self.assertEqual(layer.in_features, 8)
            self.assertEqual(layer.out_features, 4)
        self.assertEqual(
            classifier.bag_token_projection.in_features, 16 * 4
        )
        self.assertEqual(classifier.bag_token_projection.out_features, 8)

    def test_bottleneck_projection_with_residual_mean(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            project_structured_tokens=True,
            structured_tokens_per_bag=16,
            projection_bottleneck_dim=4,
            projection_residual_mean=True,
        ).eval()
        self.assertEqual(len(classifier.bag_token_bottlenecks), 16)
        self.assertEqual(
            classifier.bag_token_projection.in_features, 16 * 4 + 8
        )
        self.assertEqual(classifier.bag_token_projection.out_features, 8)
        dummy_tokens = torch.randn(2, 5, 16, 8)
        proj = classifier._projected_bag_tokens(dummy_tokens)
        self.assertEqual(tuple(proj.shape), (2, 5, 8))

    def _make_typed_bag_classifier(
        self, bottleneck_dim: int | None = None
    ) -> StructuredPopulationMetaClassifier:
        return StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            structured_tokens_per_bag=16,
            typed_bag_preserving_branch=True,
            typed_bag_num_slots=4,
            typed_bag_num_tail_fractions=3,
            typed_bag_bottleneck_dim=bottleneck_dim,
        ).eval()

    def test_typed_bag_token_identity_ids_match_layout(self) -> None:
        # index 0 = global; 1-12 = 4 slots x (center, spread, rare); 13-15 = tails.
        classifier = self._make_typed_bag_classifier()
        self.assertEqual(
            classifier.typed_bag_token_type_ids.tolist(),
            [0] + [1, 2, 3] * 4 + [4, 4, 4],
        )
        self.assertEqual(
            classifier.typed_bag_tail_fraction_ids.tolist(),
            [0] + [0] * 12 + [1, 2, 3],
        )
        self.assertFalse(hasattr(classifier, "typed_bag_slot_index_ids"))
        self.assertFalse(hasattr(classifier, "typed_bag_slot_index_embedding"))

    def test_typed_bag_tokens_add_identity_embeddings(self) -> None:
        classifier = self._make_typed_bag_classifier()
        tokens = classifier._all_structured_tokens(self.query)
        typed_tokens = tokens + classifier.typed_bag_token_type_embedding(
            classifier.typed_bag_token_type_ids
        ) + classifier.typed_bag_tail_fraction_embedding(
            classifier.typed_bag_tail_fraction_ids
        )
        expected = classifier.typed_bag_token_projection(
            torch.cat([typed_tokens.reshape(3, -1), typed_tokens.mean(dim=-2)], dim=-1)
        )
        actual = classifier._typed_bag_tokens(tokens)
        self.assertEqual(actual.shape, (3, 8))
        torch.testing.assert_close(actual, expected)

    def test_typed_bag_bottleneck_uses_per_token_bottlenecks(self) -> None:
        classifier = self._make_typed_bag_classifier(bottleneck_dim=4)
        self.assertEqual(len(classifier.typed_bag_token_bottlenecks), 16)
        for layer in classifier.typed_bag_token_bottlenecks:
            self.assertEqual(layer.in_features, 8)
            self.assertEqual(layer.out_features, 4)
        self.assertEqual(
            classifier.typed_bag_token_projection.in_features, 16 * 4 + 8
        )
        self.assertEqual(classifier.typed_bag_token_projection.out_features, 8)

    def test_typed_bag_branch_does_not_change_class_memory_path(self) -> None:
        classifier = StructuredPopulationMetaClassifier(
            token_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_set_layers=1,
            relation_hidden_dim=16,
            ridge_dim=4,
            project_structured_tokens=True,
            structured_tokens_per_bag=16,
            typed_bag_preserving_branch=True,
            typed_bag_num_slots=4,
            typed_bag_num_tail_fractions=3,
        ).eval()
        tokens = classifier._all_structured_tokens(self.query)
        expected = classifier.bag_token_projection(tokens.reshape(3, -1)).unsqueeze(1)
        actual = classifier._population_tokens(self.query)
        self.assertEqual(actual.shape, (3, 1, 8))
        torch.testing.assert_close(actual, expected)

    def test_typed_bag_preserving_branch_requires_slot_and_tail_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "typed_bag_num_slots"):
            StructuredPopulationMetaClassifier(
                token_dim=8,
                hidden_dim=16,
                num_heads=4,
                num_set_layers=1,
                relation_hidden_dim=16,
                ridge_dim=4,
                structured_tokens_per_bag=16,
                typed_bag_preserving_branch=True,
            )

    def test_simultaneous_slot_permutation_does_not_change_logits(self) -> None:
        expected = self.classifier(
            self.context, self.labels, self.query, self.query_instances
        )
        permutation = torch.randperm(4)
        context = {
            **self.context,
            "slots": self.context["slots"][:, permutation],
            "slot_metadata": self.context["slot_metadata"][:, permutation],
            "slot_covariance_sketch": self.context["slot_covariance_sketch"][:, permutation],
            "slot_covariance_reliability": self.context[
                "slot_covariance_reliability"
            ][:, permutation],
        }
        query = {
            **self.query,
            "slots": self.query["slots"][:, permutation],
            "slot_metadata": self.query["slot_metadata"][:, permutation],
            "slot_covariance_sketch": self.query["slot_covariance_sketch"][:, permutation],
            "slot_covariance_reliability": self.query[
                "slot_covariance_reliability"
            ][:, permutation],
        }
        actual = self.classifier(context, self.labels, query, self.query_instances)
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)

    def test_abundance_ridge_is_slot_permutation_equivariant(self) -> None:
        expected = self.classifier._abundance_ridge_logits(
            self.context["slot_metadata"], self.labels, self.query["slot_metadata"]
        )
        permutation = torch.randperm(4)
        actual = self.classifier._abundance_ridge_logits(
            self.context["slot_metadata"][:, permutation],
            self.labels,
            self.query["slot_metadata"][:, permutation],
        )
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)

    def test_dual_ridge_matches_primal_ridge(self) -> None:
        primal = self.classifier._abundance_ridge_logits(
            self.context["covariance_sketch"], self.labels, self.query["covariance_sketch"]
        )
        dual = self.classifier._abundance_ridge_logits(
            self.context["covariance_sketch"],
            self.labels,
            self.query["covariance_sketch"],
            dual=True,
        )
        torch.testing.assert_close(primal, dual, atol=2e-5, rtol=2e-5)

    def test_abundance_ridge_responds_with_finite_gradients(self) -> None:
        context = self.context["slot_metadata"].clone().requires_grad_()
        query = self.query["slot_metadata"].clone().requires_grad_()
        logits = self.classifier._abundance_ridge_logits(
            context, self.labels, query
        )
        changed = query.detach().clone()
        changed[:, 0, 0] += 1.0
        changed_logits = self.classifier._abundance_ridge_logits(
            context.detach(), self.labels, changed
        )
        self.assertFalse(torch.allclose(logits.detach(), changed_logits))
        logits.square().mean().backward()
        self.assertTrue(torch.isfinite(context.grad).all())
        self.assertTrue(torch.isfinite(query.grad).all())

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
        torch.testing.assert_close(auxiliary["tail_residual_scale"], torch.tensor(0.05))

    def test_rare_instance_evidence_has_finite_gradients(self) -> None:
        classifier = self.classifier.train()
        shared_tail = torch.randn(1, 3, 8)
        context = {
            **self.context,
            "tails": (
                shared_tail.expand(8, -1, -1) + 1e-7 * torch.randn(8, 3, 8)
            ).requires_grad_(),
        }
        query_instances = [
            bag.detach().clone().requires_grad_() for bag in self.query_instances
        ]
        logits = classifier(context, self.labels, self.query, query_instances)
        F.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
        self.assertTrue(torch.isfinite(context["tails"].grad).all())
        self.assertTrue(all(torch.isfinite(bag.grad).all() for bag in query_instances))


    def test_covariance_relation_candidates_are_offset_and_scale_invariant(self) -> None:
        offset = torch.randn(1, self.context["covariance_sketch"].shape[-1])
        for mode in ("prototype_cosine", "standardized_distance", "multiscale_rbf"):
            self.classifier.covariance_relation_mode = mode
            expected, _ = self.classifier._covariance_relation_scores(
                self.context["covariance_sketch"], self.labels,
                self.query["covariance_sketch"],
            )
            actual, _ = self.classifier._covariance_relation_scores(
                3.5 * (self.context["covariance_sketch"] + offset), self.labels,
                3.5 * (self.query["covariance_sketch"] + offset),
            )
            torch.testing.assert_close(expected, actual, atol=2e-5, rtol=2e-5)

    def test_covariance_relation_label_swap_flips_logits(self) -> None:
        for mode in ("prototype_cosine", "standardized_distance", "multiscale_rbf"):
            self.classifier.covariance_relation_mode = mode
            expected, _ = self.classifier._covariance_relation_scores(
                self.context["covariance_sketch"], self.labels,
                self.query["covariance_sketch"],
            )
            swapped, _ = self.classifier._covariance_relation_scores(
                self.context["covariance_sketch"], 1 - self.labels,
                self.query["covariance_sketch"],
            )
            torch.testing.assert_close(expected, swapped.flip(-1), atol=1e-6, rtol=1e-6)

    def test_covariance_relation_outer_batch_matches_single_episode(self) -> None:
        for mode in ("prototype_cosine", "standardized_distance", "multiscale_rbf"):
            self.classifier.covariance_relation_mode = mode
            expected, expected_separation = self.classifier._covariance_relation_scores(
                self.context["covariance_sketch"], self.labels,
                self.query["covariance_sketch"],
            )
            actual, actual_separation = self.classifier._covariance_relation_scores(
                self.context["covariance_sketch"].repeat(2, 1, 1),
                self.labels.repeat(2, 1),
                self.query["covariance_sketch"].repeat(2, 1, 1),
            )
            torch.testing.assert_close(actual[0], expected)
            torch.testing.assert_close(actual[1], expected)
            torch.testing.assert_close(actual_separation[0], expected_separation)

    def test_covariance_relation_requires_both_context_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "every class"):
            self.classifier._covariance_relation_scores(
                self.context["covariance_sketch"], torch.zeros_like(self.labels),
                self.query["covariance_sketch"],
            )

    def test_covariance_relation_has_finite_input_gradients(self) -> None:
        context = self.context["covariance_sketch"].clone().requires_grad_()
        query = self.query["covariance_sketch"].clone().requires_grad_()
        for mode in ("prototype_cosine", "standardized_distance", "multiscale_rbf"):
            self.classifier.covariance_relation_mode = mode
            logits, _ = self.classifier._covariance_relation_scores(
                context, self.labels, query
            )
            self.assertTrue(torch.isfinite(logits).all())
            gradients = torch.autograd.grad(
                logits.square().mean(), (context, query), retain_graph=True
            )
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_slot_covariance_relation_is_slot_permutation_invariant(self) -> None:
        context = self.context["slot_covariance_sketch"]
        context_reliability = self.context["slot_covariance_reliability"]
        query = self.query["slot_covariance_sketch"]
        query_reliability = self.query["slot_covariance_reliability"]
        for mode in ("prototype_cosine", "standardized_distance", "multiscale_rbf"):
            self.classifier.covariance_relation_mode = mode
            for routing in ("reliability_mean", "context_top1", "context_top3", "context_softmax"):
                self.classifier.covariance_relation_slot_routing = routing
                expected, _ = self.classifier._slot_covariance_relation_scores(
                    context, context_reliability, self.labels, query, query_reliability
                )
                permutation = torch.randperm(context.shape[1])
                actual, _ = self.classifier._slot_covariance_relation_scores(
                    context[:, permutation], context_reliability[:, permutation],
                    self.labels, query[:, permutation], query_reliability[:, permutation]
                )
                torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)

    def test_slot_covariance_relation_label_swap_flips_logits(self) -> None:
        expected, _ = self.classifier._slot_covariance_relation_scores(
            self.context["slot_covariance_sketch"],
            self.context["slot_covariance_reliability"], self.labels,
            self.query["slot_covariance_sketch"],
            self.query["slot_covariance_reliability"],
        )
        swapped, _ = self.classifier._slot_covariance_relation_scores(
            self.context["slot_covariance_sketch"],
            self.context["slot_covariance_reliability"], 1 - self.labels,
            self.query["slot_covariance_sketch"],
            self.query["slot_covariance_reliability"],
        )
        torch.testing.assert_close(expected, swapped.flip(-1), atol=1e-6, rtol=1e-6)

    def test_covariance_subspace_uses_context_only_and_is_finite(self) -> None:
        torch.manual_seed(241)
        matrices = torch.randn(11, 6, 6)
        covariance = matrices @ matrices.transpose(-1, -2) / 6
        context = covariance[:8].requires_grad_()
        query = covariance[8:].requires_grad_()
        for whiten in (False, True):
            context_feature, query_feature, eigenvalues = (
                self.classifier._covariance_subspace_features(
                    context, self.labels, query, rank=2, whiten=whiten
                )
            )
            self.assertEqual(query_feature.shape, (3, 2))
            self.assertTrue(torch.isfinite(context_feature).all())
            self.assertTrue(torch.isfinite(query_feature).all())
            self.assertTrue(torch.isfinite(eigenvalues).all())
            gradients = torch.autograd.grad(
                query_feature.square().mean(), (context, query), retain_graph=True
            )
            self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_zero_relation_residual_preserves_existing_logits(self) -> None:
        self.classifier.covariance_relation_enabled = False
        expected = self.classifier(
            self.context, self.labels, self.query, self.query_instances
        )
        self.classifier.covariance_relation_enabled = True
        self.classifier.covariance_relation_diagnostic_only = False
        self.classifier.covariance_relation_residual_scale = 0.0
        actual = self.classifier(
            self.context, self.labels, self.query, self.query_instances
        )
        torch.testing.assert_close(expected, actual, atol=0.0, rtol=0.0)


class BaseModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.model = build_small_model().eval()
        self.x = torch.randn(10, 13, 8)
        self.y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.mask_index = torch.tensor([8, 9])

    def test_architecture_version_is_22(self) -> None:
        self.assertEqual(self.model.architecture_version, 22)
        self.assertEqual(self.model._architecture_version.item(), 22)

    def test_mean_pooling_model_is_v23_and_runs_end_to_end(self) -> None:
        model = BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            mean_pool_structured_tokens=True,
            num_classes=2,
        ).train()
        logits, auxiliary = model(
            self.x, self.y, self.mask_index, return_auxiliary=True
        )
        self.assertEqual(model.architecture_version, 23)
        self.assertEqual(model._architecture_version.item(), 23)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(auxiliary["population_slot_weights"].shape, (2, 1))
        torch.testing.assert_close(
            auxiliary["population_slot_weights"], torch.ones(2, 1)
        )
        F.cross_entropy(logits, self.y[self.mask_index]).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_projection_model_is_v24_and_runs_end_to_end(self) -> None:
        model = BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            aggregator_num_slots=1,
            aggregator_num_density_slots=1,
            project_structured_tokens=True,
            num_classes=2,
        ).train()
        logits, auxiliary = model(
            self.x, self.y, self.mask_index, return_auxiliary=True
        )
        self.assertEqual(model.architecture_version, 24)
        self.assertEqual(model._architecture_version.item(), 24)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(auxiliary["population_slot_weights"].shape, (2, 1))
        torch.testing.assert_close(
            auxiliary["population_slot_weights"], torch.ones(2, 1)
        )
        F.cross_entropy(logits, self.y[self.mask_index]).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_bottleneck_projection_model_is_v24_and_runs_end_to_end(self) -> None:
        model = BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            aggregator_num_slots=4,
            aggregator_num_density_slots=3,
            project_structured_tokens=True,
            projection_bottleneck_dim=4,
            num_classes=2,
        ).train()
        logits, auxiliary = model(
            self.x, self.y, self.mask_index, return_auxiliary=True
        )
        self.assertEqual(model.architecture_version, 24)
        self.assertEqual(model._architecture_version.item(), 24)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(auxiliary["population_slot_weights"].shape, (2, 1))
        torch.testing.assert_close(
            auxiliary["population_slot_weights"], torch.ones(2, 1)
        )
        F.cross_entropy(logits, self.y[self.mask_index]).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))

    def test_typed_bag_preserving_model_is_v25_and_runs_end_to_end(self) -> None:
        model = BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            aggregator_num_slots=4,
            aggregator_num_density_slots=3,
            project_structured_tokens=True,
            projection_bottleneck_dim=4,
            projection_residual_mean=True,
            typed_bag_preserving_branch=True,
            typed_bag_bottleneck_dim=4,
            num_classes=2,
        ).train()
        logits, auxiliary = model(
            self.x, self.y, self.mask_index, return_auxiliary=True
        )
        self.assertEqual(model.architecture_version, 25)
        self.assertEqual(model._architecture_version.item(), 25)
        self.assertEqual(logits.shape, (2, 2))
        # Small-initialized residual: at init the typed-bag branch should
        # contribute far less than the established global/population paths.
        self.assertLess(
            auxiliary["typed_bag_residual_scale"].item(), 0.05
        )
        F.cross_entropy(logits, self.y[self.mask_index]).backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        typed_bag_grad_params = [
            model.meta_classifier.typed_bag_token_type_embedding.weight,
            model.meta_classifier.typed_bag_tail_fraction_embedding.weight,
            model.meta_classifier.typed_bag_token_projection.weight,
        ]
        self.assertTrue(
            all(parameter.grad is not None for parameter in typed_bag_grad_params)
        )
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in typed_bag_grad_params
            )
        )

    def test_typed_bag_preserving_branch_defaults_off_and_stays_v24(self) -> None:
        model = BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            aggregator_num_slots=4,
            aggregator_num_density_slots=3,
            project_structured_tokens=True,
            projection_bottleneck_dim=4,
            projection_residual_mean=True,
            num_classes=2,
        ).eval()
        self.assertEqual(model.architecture_version, 24)
        self.assertFalse(hasattr(model.meta_classifier, "typed_bag_classifier"))

    def test_final_logits_are_invariant_to_per_bag_shift(self) -> None:
        shift = torch.randn(self.x.shape[0], 1, self.x.shape[-1])
        expected = self.model(self.x, self.y, self.mask_index)
        actual = self.model(self.x + shift, self.y, self.mask_index)
        torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)

    def test_all_branch_logits_are_invariant_to_per_bag_shift(self) -> None:
        shift = torch.randn(self.x.shape[0], 1, self.x.shape[-1])
        _, expected = self.model(self.x, self.y, self.mask_index, return_auxiliary=True)
        _, actual = self.model(
            self.x + shift, self.y, self.mask_index, return_auxiliary=True
        )
        for name in (
            "global_shape_logits",
            "population_logits",
            "tail_logits",
            "class_memories",
            "slot_tokens",
            "tail_tokens",
        ):
            torch.testing.assert_close(
                expected[name], actual[name], atol=1e-5, rtol=1e-4
            )

    def test_ragged_final_logits_are_invariant_to_per_bag_shift(self) -> None:
        bags = [torch.randn(9 + index, 8) for index in range(10)]
        shifts = [torch.randn(1, 8) for _ in bags]
        expected = self.model(bags, self.y, self.mask_index)
        actual = self.model(
            [bag + shift for bag, shift in zip(bags, shifts)],
            self.y,
            self.mask_index,
        )
        torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)

    def test_outer_batch_is_invariant_to_per_bag_shift(self) -> None:
        batch_x = torch.stack((self.x, self.x.roll(1, dims=0)))
        batch_y = torch.stack((self.y, self.y.roll(2)))
        batch_mask = self.mask_index.expand(2, -1)
        shifts = torch.randn(2, 10, 1, 8)
        expected = self.model.forward_episode_batch(batch_x, batch_y, batch_mask)
        actual = self.model.forward_episode_batch(batch_x + shifts, batch_y, batch_mask)
        torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4)

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
        expected = torch.stack(
            [self.model(x, y, mask) for x, y, mask in zip(batch_x, batch_y, batch_mask)]
        )
        actual = self.model.forward_episode_batch(batch_x, batch_y, batch_mask)
        torch.testing.assert_close(expected, actual, atol=3e-5, rtol=3e-5)

    def test_all_cell_evidence_is_instance_order_invariant(self) -> None:
        expected = self.model(self.x, self.y, self.mask_index)
        permuted = self.x.clone()
        for bag_index in range(permuted.shape[0]):
            permuted[bag_index] = permuted[bag_index, torch.randperm(permuted.shape[1])]
        actual = self.model(permuted, self.y, self.mask_index)
        torch.testing.assert_close(expected, actual, atol=3e-5, rtol=3e-5)

    def test_context_bag_order_does_not_change_prediction(self) -> None:
        expected = self.model(self.x, self.y, self.mask_index)
        context_permutation = torch.randperm(8)
        permutation = torch.cat((context_permutation, torch.tensor([8, 9])))
        actual = self.model(self.x[permutation], self.y[permutation], self.mask_index)
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
        self.assertGreaterEqual(auxiliary["population_residual_scale"].item(), 0.10)
        self.assertGreaterEqual(auxiliary["tail_residual_scale"].item(), 0.05)
        self.assertEqual(auxiliary["cross_attention_entropy"].shape, (2, 2))
        torch.testing.assert_close(
            auxiliary["context_class_counts"], torch.tensor([4, 4])
        )

    def test_gated_distance_relation_mode(self) -> None:
        model = BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            num_classes=2,
            covariance_relation={
                "enabled": True,
                "mode": "gated_distance",
                "granularity": "subspace",
                "subspace_rank": 1,
            },
        ).eval()
        logits = model(self.x, self.y, self.mask_index)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_learned_head_relation_mode(self) -> None:
        model = BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            num_classes=2,
            covariance_relation={
                "enabled": True,
                "mode": "learned_head",
                "granularity": "subspace",
                "subspace_rank": 1,
            },
        ).eval()
        logits = model(self.x, self.y, self.mask_index)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
