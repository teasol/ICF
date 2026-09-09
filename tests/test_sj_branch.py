import unittest
import torch

from src.models.branches.sj import (
    SJ_FEATURE_DIM,
    SHJ_FEATURE_DIM,
    sj_slide_features,
    shj_slide_features,
)
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


class TestSjBranch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dim = 64
        self.n_ctx, self.n_qry = 30, 12
        self.ctx_bags = [torch.randn(torch.randint(60, 120, (1,)).item(), self.dim) for _ in range(self.n_ctx)]
        self.ctx_labels = torch.tensor([i % 2 for i in range(self.n_ctx)], dtype=torch.long)
        self.qry_bags = [torch.randn(torch.randint(60, 120, (1,)).item(), self.dim) for _ in range(self.n_qry)]

    def _config(self, **kw):
        base = dict(sketch_dim=32, weight_cv=1.0, weight_bm=1.0, weight_bd=1.0,
                    weight_qa=1.0, weight_ds=1.0, weight_ct=0.0, weight_dd=0.0,
                    aggregation="trimmed_mean")
        base.update(kw)
        return TrainingFreeConfig(**base)

    def test_features_are_location_and_scale_invariant(self):
        """SJ's orthogonality rests on this: shifting or rescaling a slide's
        tokens must not move its features, or SJ would restate the mean."""
        basis = torch.linalg.qr(torch.randn(self.dim, 32))[0]
        bag = self.ctx_bags[0]
        base = sj_slide_features(bag, basis, 16)
        self.assertEqual(base.shape, (SJ_FEATURE_DIM,))
        self.assertEqual(SJ_FEATURE_DIM, SHJ_FEATURE_DIM)
        shifted = sj_slide_features(bag + 3.7, basis, 16)
        scaled = sj_slide_features(bag * 2.5, basis, 16)
        self.assertLess((base - shifted).abs().max().item(), 1e-3)
        self.assertLess((base - scaled).abs().max().item(), 1e-3)

        # Backward compatibility alias test
        shj_base = shj_slide_features(bag, basis, 16)
        self.assertLess((base - shj_base).abs().max().item(), 1e-7)

    def test_sj_forward_and_antisymmetry(self):
        clf = TrainingFreeClassifier(self._config(weight_sj=1.0, sj_dim=16))
        m = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertEqual(m.shape, (self.n_qry,))
        self.assertTrue(torch.isfinite(m).all())
        inv = clf.margins(self.ctx_bags, 1 - self.ctx_labels, self.qry_bags)
        self.assertLess((m + inv).abs().max().item(), 1e-4)

    def test_backward_compatible_shj_config(self):
        """Passing weight_shj instead of weight_sj should yield identical margins."""
        cfg_new = self._config(weight_sj=1.0, sj_dim=16)
        cfg_old = self._config(weight_shj=1.0, shj_dim=16)
        m_new = TrainingFreeClassifier(cfg_new).margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        m_old = TrainingFreeClassifier(cfg_old).margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertLess((m_new - m_old).abs().max().item(), 1e-6)

    def test_zero_weight_leaves_ensemble_untouched(self):
        """weight_sj defaults to 0.0, so existing configs must be unaffected."""
        off = TrainingFreeClassifier(self._config()).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        explicit = TrainingFreeClassifier(self._config(weight_sj=0.0, sj_dim=16)).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertLess((off - explicit).abs().max().item(), 1e-6)

    def test_sj_changes_the_ensemble_when_enabled(self):
        off = TrainingFreeClassifier(self._config()).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        on = TrainingFreeClassifier(self._config(weight_sj=1.0, sj_dim=16)).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertGreater((off - on).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
