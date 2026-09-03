import unittest
import torch
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


class TestGatedAndAdaptiveTrimmed(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.n_ctx = 30
        self.n_qry = 15
        self.ctx_bags = [torch.randn(torch.randint(40, 80, (1,)).item(), self.dim) for _ in range(self.n_ctx)]
        self.ctx_labels = torch.tensor([i % 2 for i in range(self.n_ctx)], dtype=torch.long)
        self.qry_bags = [torch.randn(torch.randint(40, 80, (1,)).item(), self.dim) for _ in range(self.n_qry)]

    def test_hard_gated_forward_and_antisymmetry(self):
        cfg = TrainingFreeConfig(
            sketch_dim=32,
            weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0, weight_ds=1.0,
            weight_ct=0.0, weight_dd=0.0,
            aggregation="hard_gated",
            gated_tau=0.05,
        )
        clf = TrainingFreeClassifier(cfg)
        m_orig = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertEqual(m_orig.shape, (self.n_qry,))

        inv_labels = 1 - self.ctx_labels
        m_inv = clf.margins(self.ctx_bags, inv_labels, self.qry_bags)
        sym_error = (m_orig + m_inv).abs().max().item()
        self.assertLess(sym_error, 1e-4, f"Hard Gated antisymmetry broken: {sym_error}")

    def test_adaptive_trimmed_forward_and_antisymmetry(self):
        cfg = TrainingFreeConfig(
            sketch_dim=32,
            weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0, weight_ds=1.0,
            weight_ct=0.0, weight_dd=0.0,
            aggregation="adaptive_trimmed",
            adaptive_tau=0.08,
            adaptive_ratio=1.5,
        )
        clf = TrainingFreeClassifier(cfg)
        m_orig = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertEqual(m_orig.shape, (self.n_qry,))

        inv_labels = 1 - self.ctx_labels
        m_inv = clf.margins(self.ctx_bags, inv_labels, self.qry_bags)
        sym_error = (m_orig + m_inv).abs().max().item()
        self.assertLess(sym_error, 1e-4, f"Adaptive Trimmed antisymmetry broken: {sym_error}")
