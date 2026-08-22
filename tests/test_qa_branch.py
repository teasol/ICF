"""Unit tests for QA (Quantile & Extremum Evidence) Branch."""

import unittest
import torch

from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


class TestQABranch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d = 1536
        self.k = 32
        self.basis = torch.randn(self.d, self.k)
        # 6 context bags: 3 class 0, 3 class 1
        self.context_bags = [torch.randn(50, self.d) for _ in range(6)]
        self.context_labels = torch.tensor([0, 1, 0, 1, 0, 1])
        # 4 query bags
        self.query_bags = [torch.randn(40, self.d) for _ in range(4)]

    def test_default_config_includes_qa(self):
        cfg = TrainingFreeConfig()
        self.assertTrue(hasattr(cfg, "qa_dim"))
        self.assertTrue(hasattr(cfg, "qa_quantiles"))
        self.assertTrue(hasattr(cfg, "qa_lambda"))
        self.assertTrue(hasattr(cfg, "weight_qa"))
        self.assertEqual(cfg.qa_dim, 32)
        self.assertEqual(cfg.qa_quantiles, (0.05, 0.10, 0.90, 0.95))
        self.assertEqual(cfg.qa_lambda, 1.0)
        self.assertEqual(cfg.weight_qa, 0.0)

    def test_qa_features_shape_and_finite(self):
        clf = TrainingFreeClassifier(TrainingFreeConfig(weight_qa=1.0))
        m_qa = clf._qa_features(
            self.context_bags, self.context_labels, self.query_bags, self.basis
        )
        self.assertEqual(m_qa.shape, (len(self.query_bags),))
        self.assertTrue(torch.isfinite(m_qa).all())

    def test_qa_label_antisymmetry(self):
        """Swapping context labels 0 <-> 1 must negate QA margins."""
        clf = TrainingFreeClassifier(TrainingFreeConfig(weight_qa=1.0))
        m_fwd = clf._qa_features(
            self.context_bags, self.context_labels, self.query_bags, self.basis
        )
        flipped_labels = 1 - self.context_labels
        m_rev = clf._qa_features(
            self.context_bags, flipped_labels, self.query_bags, self.basis
        )
        self.assertTrue(
            torch.allclose(m_fwd, -m_rev, atol=1e-5),
            f"Expected m_fwd == -m_rev, got m_fwd={m_fwd}, m_rev={m_rev}",
        )

    def test_qa_determinism(self):
        """Two calls with identical inputs must produce exact identical outputs."""
        clf = TrainingFreeClassifier(TrainingFreeConfig(weight_qa=1.0))
        m1 = clf._qa_features(
            self.context_bags, self.context_labels, self.query_bags, self.basis
        )
        m2 = clf._qa_features(
            self.context_bags, self.context_labels, self.query_bags, self.basis
        )
        self.assertTrue(torch.equal(m1, m2))

    def test_qa_query_isolation_no_leakage(self):
        """Evaluating query bag 0 alone must equal evaluating it in a batch."""
        clf = TrainingFreeClassifier(TrainingFreeConfig(weight_qa=1.0))
        m_batch = clf._qa_features(
            self.context_bags, self.context_labels, self.query_bags, self.basis
        )
        m_single = clf._qa_features(
            self.context_bags, self.context_labels, [self.query_bags[0]], self.basis
        )
        self.assertTrue(
            torch.allclose(m_batch[0:1], m_single, atol=1e-5),
            f"Query leakage detected: batch={m_batch[0]}, single={m_single[0]}",
        )

    def test_qa_soft_voting_integration(self):
        """Soft voting with weight_qa=1.0 must include QA branch in probability averaging."""
        cfg_no_qa = TrainingFreeConfig(
            weight_cv=1.0, weight_ct=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=0.0,
            aggregation="soft_voting"
        )
        cfg_with_qa = TrainingFreeConfig(
            weight_cv=1.0, weight_ct=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0,
            aggregation="soft_voting"
        )
        clf_no_qa = TrainingFreeClassifier(cfg_no_qa)
        clf_with_qa = TrainingFreeClassifier(cfg_with_qa)

        p_no = clf_no_qa.predict_proba(self.context_bags, self.context_labels, self.query_bags)
        p_with = clf_with_qa.predict_proba(self.context_bags, self.context_labels, self.query_bags)

        self.assertEqual(p_no.shape, (len(self.query_bags),))
        self.assertEqual(p_with.shape, (len(self.query_bags),))
        self.assertTrue(torch.isfinite(p_with).all())
        self.assertTrue(((p_with >= 0.0) & (p_with <= 1.0)).all())
        # With QA branch active, probabilities should differ from 4-branch baseline
        self.assertFalse(torch.allclose(p_no, p_with))

    def test_extreme_quantile_sensitivity(self):
        """Adding extreme outlier cells to Class 1 bags must positively shift QA margin for outlier query."""
        torch.manual_seed(123)
        # Create normal background for all bags
        ctx_bags = [torch.randn(100, self.d) for _ in range(6)]
        ctx_lbls = torch.tensor([0, 1, 0, 1, 0, 1])

        # Inject 3% high positive spike along basis direction 0 into Class 1 context bags
        spike_direction = self.basis[:, 0]
        for idx in [1, 3, 5]:
            ctx_bags[idx][:3] += 10.0 * spike_direction

        # Query 0: normal background, Query 1: has 3% high positive spike
        q_normal = torch.randn(100, self.d)
        q_spiked = torch.randn(100, self.d)
        q_spiked[:3] += 10.0 * spike_direction

        clf = TrainingFreeClassifier(TrainingFreeConfig(weight_qa=1.0))
        m = clf._qa_features(ctx_bags, ctx_lbls, [q_normal, q_spiked], self.basis)

        self.assertGreater(
            m[1].item(), m[0].item(),
            f"QA branch failed to detect extreme rare spike: spiked={m[1].item()}, normal={m[0].item()}"
        )


if __name__ == "__main__":
    unittest.main()
