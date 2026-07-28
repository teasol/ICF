"""Unit tests for Architecture v21 Large Context Signal-Aware Pre-training."""

import unittest
import torch

from src.models.baseline import BaseModel
from src.modules.data_interface import SignalAwarePretrainEpisodeCollator


class TestLargeContextPretraining(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel(
            input_dim=512,
            meta_hidden_dim=256,
            num_classes=2,
        )
        self.model.eval()

    def test_chunked_extract_bag_features(self):
        # 96 bags with 100 cells of 512 dimensions
        num_bags = 96
        num_cells = 100
        input_dim = 512
        x = torch.randn(num_bags, num_cells, input_dim)

        # Standard extraction
        features_dense = self.model.extract_bag_features(x, chunk_size=0)
        self.assertEqual(features_dense.shape[0], num_bags)
        self.assertEqual(features_dense.shape[1], 40)

        # Chunked extraction (chunk_size = 32)
        features_chunked = self.model.extract_bag_features(x, chunk_size=32)
        self.assertEqual(features_chunked.shape[0], num_bags)
        self.assertEqual(features_chunked.shape[1], 40)

        # Numerical equivalence
        diff = (features_dense - features_chunked).abs().max()
        self.assertLess(diff.item(), 1e-4)

    def test_large_candidate_pool_retrieval(self):
        num_bags = 96
        num_cells = 100
        input_dim = 512
        x = torch.randn(num_bags, num_cells, input_dim)
        y = torch.randint(0, 2, (num_bags,))
        mask_index = num_bags - 1  # query bag index

        retrieval_k = 24
        final_x, final_y, final_mask = self.model.retrieve_context_indices(
            x=x, y=y, mask_index=mask_index, retrieval_k=retrieval_k, chunk_size=32
        )

        # 24 context bags + 1 query bag = 25 bags total
        self.assertEqual(final_x.shape[0], retrieval_k + 1)
        self.assertEqual(final_y.shape[0], retrieval_k + 1)
        self.assertEqual(final_mask.item(), retrieval_k)

        # Test forward pass with retrieved subset
        logits = self.model.forward(x, y, mask_index=mask_index, retrieval_k=retrieval_k)
        self.assertEqual(logits.shape, (1, 2))

    def test_collator_formatting(self):
        collator = SignalAwarePretrainEpisodeCollator(retrieval_k=24)
        samples = [
            (
                torch.randn(60, 50, 512),
                torch.randint(0, 2, (60,)),
                {},
            )
        ]
        # Test collator call
        formatted = collator(samples)
        self.assertEqual(len(formatted), 3)
        x_out, y_out, mask_out = formatted
        self.assertEqual(x_out.shape[0], 60)
        self.assertEqual(y_out.shape[0], 60)
        self.assertEqual(mask_out.item(), 59)


if __name__ == "__main__":
    unittest.main()
