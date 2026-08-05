import unittest

import torch

from src.modules.model_interface import ModelInterface


class PairwiseRankingLossTest(unittest.TestCase):
    def test_prefers_correct_positive_negative_order(self) -> None:
        targets = torch.tensor([0, 1])
        correctly_ranked = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        incorrectly_ranked = correctly_ranked.flip(0)
        correct_loss = ModelInterface._pairwise_ranking_loss(correctly_ranked, targets)
        incorrect_loss = ModelInterface._pairwise_ranking_loss(
            incorrectly_ranked, targets
        )
        self.assertLess(correct_loss, incorrect_loss)

    def test_is_label_permutation_equivariant(self) -> None:
        logits = torch.tensor([[1.0, -0.5], [-0.2, 0.7], [0.1, 0.2]])
        targets = torch.tensor([0, 1, 1])
        expected = ModelInterface._pairwise_ranking_loss(logits, targets)
        actual = ModelInterface._pairwise_ranking_loss(logits.flip(-1), 1 - targets)
        torch.testing.assert_close(expected, actual)

    def test_single_class_query_batch_has_zero_ranking_term(self) -> None:
        logits = torch.randn(3, 2, requires_grad=True)
        loss = ModelInterface._pairwise_ranking_loss(
            logits, torch.ones(3, dtype=torch.long)
        )
        torch.testing.assert_close(loss, torch.tensor(0.0))
        loss.backward()
        self.assertIsNotNone(logits.grad)


class RoutingBalanceLossTest(unittest.TestCase):
    def test_uniform_episode_usage_has_zero_penalty(self) -> None:
        weights = torch.full((4, 3), 1.0 / 3.0)
        loss = ModelInterface._routing_balance_loss(weights)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)

    def test_collapsed_episode_usage_is_penalized(self) -> None:
        weights = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
        loss = ModelInterface._routing_balance_loss(weights)
        torch.testing.assert_close(loss, torch.tensor(3.0).log())


class TrainingContextSamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context_sizes = (40, 80, 120, 160, 180, 240, 300)
        self.interface = ModelInterface(
            model_src="src.models.baseline.BaseModel",
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ridge_dim=4,
            training_targets_per_episode=(5, 12),
            training_context_sizes=self.context_sizes,
            training_context_jitter=5,
        )

    def test_queries_leave_context_inside_configured_bucket(self) -> None:
        torch.manual_seed(3)
        for total_bags in (40, 52, 85, 127, 172, 191, 247, 317):
            y = torch.arange(total_bags) % 2
            query = self.interface._sample_training_queries(y)
            context = total_bags - query.numel()
            self.assertTrue(
                any(abs(context - center) <= 5 for center in self.context_sizes)
            )

    def test_impossible_total_bag_count_is_rejected(self) -> None:
        y = torch.arange(70) % 2
        with self.assertRaisesRegex(ValueError, "cannot produce"):
            self.interface._sample_training_queries(y)


class FinalObjectiveTest(unittest.TestCase):
    def test_path_auxiliary_losses_are_absent(self) -> None:
        interface = ModelInterface(
            model_src="src.models.baseline.BaseModel",
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ridge_dim=4,
            ranking_loss_weight=0.1,
            routing_balance_weight=0.01,
        )
        x = torch.randn(10, 13, 8)
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        _, terms = interface._episode_losses(x, y, torch.tensor([8, 9]))
        self.assertFalse(any(name.endswith("aux_loss") for name in terms))
        self.assertIn("ce_loss", terms)
        self.assertIn("routing_balance_loss", terms)


class ArchitectureCheckpointCompatibilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.interface = ModelInterface(
            model_src="src.models.baseline.BaseModel",
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ridge_dim=4,
        )

    def test_v21_checkpoint_is_rejected(self) -> None:
        # v22 removed the retrieval layer, so every v21 checkpoint -- including
        # the Phase 5 large-context pretrain one -- must fail loudly rather
        # than silently load into a structurally different model.
        checkpoint = {"state_dict": {"model._architecture_version": torch.tensor(21)}}
        with self.assertRaisesRegex(RuntimeError, "Expected v22, found 21"):
            self.interface.on_load_checkpoint(checkpoint)

    def test_versionless_checkpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Expected v22, found missing"):
            self.interface.on_load_checkpoint({"state_dict": {}})

    def test_v22_checkpoint_is_accepted(self) -> None:
        checkpoint = {"state_dict": {"model._architecture_version": torch.tensor(22)}}
        self.interface.on_load_checkpoint(checkpoint)

    def test_v24_model_rejects_v22_checkpoint(self) -> None:
        interface = ModelInterface(
            model_src="src.models.baseline.BaseModel",
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ridge_dim=4,
            aggregator_num_slots=1,
            aggregator_num_density_slots=1,
            project_structured_tokens=True,
        )
        checkpoint = {"state_dict": {"model._architecture_version": torch.tensor(22)}}
        with self.assertRaisesRegex(RuntimeError, "Expected v24, found 22"):
            interface.on_load_checkpoint(checkpoint)

    def test_v25_model_rejects_v24_checkpoint(self) -> None:
        # T5-A adds a new typed-bag-preserving branch (new embeddings + a
        # second RidgeResidualMetaClassifier) on top of v24's state_dict, so
        # a v24 checkpoint is missing weights v25 requires -- it must be
        # rejected rather than partially loaded.
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
            typed_bag_preserving_branch=True,
            typed_bag_bottleneck_dim=4,
        )
        checkpoint = {"state_dict": {"model._architecture_version": torch.tensor(24)}}
        with self.assertRaisesRegex(RuntimeError, "Expected v25, found 24"):
            interface.on_load_checkpoint(checkpoint)

    def test_v26_model_rejects_v24_checkpoint(self) -> None:
        # cls_token_pooling adds a new ClassTokenPooling module (learned CLS
        # cross-attention over raw cells) whose weights a v24 checkpoint does
        # not have, so it must be rejected rather than partially loaded.
        interface = ModelInterface(
            model_src="src.models.baseline.BaseModel",
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            meta_ridge_dim=4,
            aggregator_num_slots=1,
            aggregator_num_density_slots=1,
            project_structured_tokens=True,
            cls_token_pooling=True,
            cls_token_heads=4,
        )
        checkpoint = {"state_dict": {"model._architecture_version": torch.tensor(24)}}
        with self.assertRaisesRegex(RuntimeError, "Expected v26, found 24"):
            interface.on_load_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
