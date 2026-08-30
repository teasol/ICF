import unittest
import torch
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig

class SWBranchTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.n_ctx = 20
        self.n_qry = 6
        self.ctx_bags = [torch.randn(torch.randint(50, 150, (1,)).item(), self.dim) for _ in range(self.n_ctx)]
        self.ctx_labels = torch.tensor([i % 2 for i in range(self.n_ctx)], dtype=torch.long)
        self.qry_bags = [torch.randn(torch.randint(50, 150, (1,)).item(), self.dim) for _ in range(self.n_qry)]

    def test_exact_label_antisymmetry(self):
        config = TrainingFreeConfig(
            sketch_dim=32,
            weight_cv=0.0, weight_ct=0.0, weight_bm=0.0, weight_bd=0.0,
            weight_qa=0.0, weight_ds=0.0, weight_lr=0.0, weight_de=0.0,
            weight_sw=1.0, sw_dim=32, sw_num_slices=16, sw_num_quantiles=16,
            aggregation="linear",
        )
        clf = TrainingFreeClassifier(config)
        m_orig = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        m_flip = clf.margins(self.ctx_bags, 1 - self.ctx_labels, self.qry_bags)

        diff = (m_orig + m_flip).abs().max().item()
        print(f"SW Branch Max Antisymmetry Error: {diff:.8e}")
        self.assertLess(diff, 1e-5, f"SW branch failed label antisymmetry: {diff}")

    def test_soft_voting_antisymmetry(self):
        config = TrainingFreeConfig(
            sketch_dim=32,
            weight_cv=0.0, weight_ct=0.0, weight_bm=0.0, weight_bd=0.0,
            weight_qa=0.0, weight_ds=0.0, weight_lr=0.0, weight_de=0.0,
            weight_sw=1.0, sw_dim=32, sw_num_slices=16, sw_num_quantiles=16,
            aggregation="soft_voting",
        )
        clf = TrainingFreeClassifier(config)
        p_orig = clf.predict_proba(self.ctx_bags, self.ctx_labels, self.qry_bags)
        p_flip = clf.predict_proba(self.ctx_bags, 1 - self.ctx_labels, self.qry_bags)

        diff = (p_orig + p_flip - 1.0).abs().max().item()
        print(f"SW Branch Soft Voting Symmetry Error: {diff:.8e}")
        self.assertLess(diff, 1e-5, f"SW soft voting symmetry failed: {diff}")


if __name__ == "__main__":
    unittest.main()
