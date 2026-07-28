import sys
from pathlib import Path
import unittest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import BaseModel


class TestFeatureRetrieval(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel(
            input_dim=512,
            aggregator_num_slots=12,
            aggregator_num_density_slots=8,
            bag_centered_representation=True,
            num_classes=2,
        )
        self.model.eval()

    def test_extract_bag_features(self):
        # 30 bags, 100 cells, 512 dim
        x = torch.randn(30, 100, 512)
        features = self.model.extract_bag_features(x)
        self.assertEqual(features.ndim, 2)
        self.assertEqual(features.shape[0], 30)

    def test_retrieve_context_indices(self):
        x = torch.randn(30, 100, 512)
        y = torch.randint(0, 2, (30,))
        # Ensure at least 12 of each class
        y[:15] = 0
        y[15:] = 1
        mask_index = 29  # 1 query at index 29

        ret_x, ret_y, ret_mask = self.model.retrieve_context_indices(
            x, y, mask_index=mask_index, retrieval_k=24
        )
        # Should return 24 context bags + 1 query bag = 25 bags total
        self.assertEqual(ret_x.shape[0], 25)
        self.assertEqual(ret_y.shape[0], 25)
        self.assertEqual(ret_mask.item(), 24)

    def test_forward_with_retrieval(self):
        x = torch.randn(30, 100, 512)
        y = torch.randint(0, 2, (30,))
        y[:15] = 0
        y[15:] = 1
        mask_index = 29

        logits = self.model(x, y, mask_index=mask_index, retrieval_k=24)
        self.assertEqual(logits.ndim, 2)
        self.assertEqual(logits.shape[0], 1)  # 1 query logit
        self.assertEqual(logits.shape[1], 2)  # 2 classes


if __name__ == "__main__":
    unittest.main()
