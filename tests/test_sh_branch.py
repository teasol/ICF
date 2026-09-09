import unittest

import torch

from src.models.branches.sh import sh_slide_features
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


class TestShBranch(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
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
        """SH's orthogonality rests on this: standardisation must absorb any
        shift or rescaling of a slide's tokens, or SH would restate the mean."""
        basis = torch.linalg.qr(torch.randn(self.dim, 32))[0]
        bag = self.ctx_bags[0]
        base = sh_slide_features(bag, basis, 16)
        self.assertEqual(base.shape, (32,))  # 2 * dim: skew half + kurt half
        shifted = sh_slide_features(bag + 3.7, basis, 16)
        scaled = sh_slide_features(bag * 2.5, basis, 16)
        self.assertLess((base - shifted).abs().max().item(), 1e-3)
        self.assertLess((base - scaled).abs().max().item(), 1e-3)

    def test_matches_the_screening_formula(self):
        """sh_slide_features must stay bit-identical to the §219 screening
        derivation: moments read at the wide projection, first `dim` kept."""
        basis = torch.linalg.qr(torch.randn(self.dim, 32))[0]
        bag = self.ctx_bags[0]
        wide = 24
        dim = 16
        pr = bag.float() @ basis[:, :wide].float()
        sd = pr.std(dim=0, keepdim=True).clamp_min(1e-6)
        z = (pr - pr.mean(dim=0, keepdim=True)) / sd
        expected = torch.cat([z.pow(3).mean(dim=0)[:dim], (z.pow(4).mean(dim=0) - 3.0)[:dim]])
        actual = sh_slide_features(bag, basis, dim, wide)
        self.assertLess((expected - actual).abs().max().item(), 1e-5)

    def test_sh_forward_and_antisymmetry(self):
        clf = TrainingFreeClassifier(self._config(weight_sh=1.0, sh_dim=16))
        m = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertEqual(m.shape, (self.n_qry,))
        self.assertTrue(torch.isfinite(m).all())
        inv = clf.margins(self.ctx_bags, 1 - self.ctx_labels, self.qry_bags)
        self.assertLess((m + inv).abs().max().item(), 1e-4)

    def test_zero_weight_leaves_ensemble_untouched(self):
        """weight_sh defaults to 0.0, so existing configs must be unaffected."""
        off = TrainingFreeClassifier(self._config()).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        explicit = TrainingFreeClassifier(self._config(weight_sh=0.0, sh_dim=16)).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertLess((off - explicit).abs().max().item(), 1e-6)

    def test_sh_changes_the_ensemble_when_enabled(self):
        off = TrainingFreeClassifier(self._config()).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        on = TrainingFreeClassifier(self._config(weight_sh=1.0, sh_dim=16)).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertGreater((off - on).abs().max().item(), 1e-6)

    def test_default_config_has_no_sh_fields_set(self):
        cfg = TrainingFreeConfig()
        self.assertEqual(cfg.weight_sh, 0.0)
        self.assertEqual(cfg.sh_dim, 32)
        self.assertEqual(cfg.sh_wide, 256)
        self.assertEqual(cfg.sh_lambda, 1.0)


class TestContextLooIncludesShapeBranches(unittest.TestCase):
    """Regression pin: context_loo_stacking used to drop SJ (and had no SH) even
    when their weights were set — the margins were computed but never entered the
    branch pool. A shape branch whose context-LOO AUROC beats the floor must
    receive pooling weight and move the ensemble."""

    def setUp(self):
        torch.manual_seed(5)
        self.n = 10
        self.labels = torch.tensor([i % 2 for i in range(self.n)], dtype=torch.long)
        # Context-LOO margins that rank the labels perfectly: the shape branch
        # earns the maximum pooling weight under any loo_gamma/loo_floor.
        self.perfect = (self.labels.float() * 2.0 - 1.0) * 3.0

    def _stack(self, config_kw=None, **extra):
        from src.models.aggregations.voting import context_loo_stacking
        config = TrainingFreeConfig(
            weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0,
            weight_ds=1.0, weight_ct=0.0, weight_dd=0.0,
            aggregation="context_loo_stacking", **(config_kw or {}))
        cv = torch.randn(self.n)
        m_cv = torch.randn(self.n)
        loo_cv = torch.randn(self.n)
        m_bm = torch.randn(self.n)
        m_bd = torch.randn(self.n)
        m_qa = torch.randn(self.n)
        m_ds = torch.randn(self.n)
        kwargs = dict(
            cv=cv, context_labels=self.labels,
            m_cv=m_cv, loo_cv=loo_cv,
            m_dd=None, loo_dd=None,
            m_ct=None, loo_ct=None,
            m_bm=m_bm, loo_bm=torch.randn(self.n),
            m_bd=m_bd, loo_bd=torch.randn(self.n),
            m_qa=m_qa, loo_qa=torch.randn(self.n),
            m_ds=m_ds, loo_ds=torch.randn(self.n),
            m_de=None, loo_de=None,
            m_sw=None, loo_sw=None,
        )
        return context_loo_stacking(config, **kwargs)

    def test_sj_enters_the_context_loo_pool(self):
        off = self._stack()
        on = self._stack(config_kw=dict(weight_sj=1.0, sj_dim=16),
                         m_sj=self.perfect, loo_sj=self.perfect)
        self.assertGreater((off - on).abs().max().item(), 1e-6)

    def test_sh_enters_the_context_loo_pool(self):
        off = self._stack()
        on = self._stack(config_kw=dict(weight_sh=1.0, sh_dim=16),
                         m_sh=self.perfect, loo_sh=self.perfect)
        self.assertGreater((off - on).abs().max().item(), 1e-6)


if __name__ == "__main__":
    unittest.main()
