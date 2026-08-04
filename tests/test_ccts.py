import unittest
import torch
from src.models.baseline import BaseModel

class CCTSTest(unittest.TestCase):
    """Unit tests for Cardinality-Calibrated Tail Scan (CCTS) implementation."""

    def test_ccts_token_generation_and_gradient(self) -> None:
        torch.manual_seed(42)
        model = BaseModel(
            input_dim=8,
            meta_hidden_dim=16,
            meta_num_heads=4,
            meta_num_set_layers=1,
            meta_relation_hidden_dim=16,
            aggregator_ccts_lambdas=(0.25, 1.0, 4.0),
            aggregator_ccts_tau=0.5,
            num_classes=2,
            bag_representation="poolz_l2",
        ).train()

        self.assertEqual(model.aggregator.ccts_lambdas, (0.25, 1.0, 4.0))

        episodes, bags, instances, dim = 2, 5, 8, 8
        x = torch.randn(episodes, bags, instances, dim, requires_grad=True)
        y = torch.tensor([[0, 1, 0, 1, 0]] * episodes)
        mask_index = torch.zeros(episodes, 1, dtype=torch.long)

        logits = model.forward_episode_batch(x, y, mask_index)
        logits = logits[0] if isinstance(logits, tuple) else logits

        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(logits.shape, (episodes, 1, 2))

        loss = logits.float().square().mean()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

if __name__ == "__main__":
    unittest.main()
