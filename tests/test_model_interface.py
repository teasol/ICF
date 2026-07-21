import unittest

import torch

from src.modules.model_interface import ModelInterface


class PairwiseRankingLossTest(unittest.TestCase):
    def test_prefers_correct_positive_negative_order(self) -> None:
        targets = torch.tensor([0, 1])
        correctly_ranked = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        incorrectly_ranked = correctly_ranked.flip(0)
        correct_loss = ModelInterface._pairwise_ranking_loss(
            correctly_ranked, targets
        )
        incorrect_loss = ModelInterface._pairwise_ranking_loss(
            incorrectly_ranked, targets
        )
        self.assertLess(correct_loss, incorrect_loss)

    def test_is_label_permutation_equivariant(self) -> None:
        logits = torch.tensor([[1.0, -0.5], [-0.2, 0.7], [0.1, 0.2]])
        targets = torch.tensor([0, 1, 1])
        expected = ModelInterface._pairwise_ranking_loss(logits, targets)
        actual = ModelInterface._pairwise_ranking_loss(
            logits.flip(-1), 1 - targets
        )
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


if __name__ == "__main__":
    unittest.main()
