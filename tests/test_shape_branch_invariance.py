"""Shape-family branches (BS/SH/SJ): docstring invariance claims vs. actual code.

Checks the claims in src/models/branches/{bs,sh,sj}.py's module docstrings
numerically (RU: BS/SH/SJ implementation-soundness review):

  * BS ("pure scale (log total variance)"): translating a bag must not move
    the BS feature, and scaling a bag by `s` must shift it by exactly
    `log(s**2)`.
  * SH ("location- AND scale-invariant by construction"): per-dimension
    skew/kurtosis of the z-scored projection must not move under translation
    or isotropic scaling.
  * SJ ("location- and scale-invariant, multivariate"): the whitened-radius
    shape descriptors must not move under translation or isotropic scaling.

All three are computed in float32 (see the branches' own autocast-disabling
comments), so the tolerance here is float32-precision (~1e-5), not exact.

Also pins the N=1 degenerate-bag NaN behaviour flagged during the review
(A3): SH and SJ divide by a std/radius-std with zero degrees of freedom at
N=1, producing NaN slide features; the branch-level `*_features` wrappers
already absorb this via `torch.nan_to_num(..., nan=0.0)` before the ridge
solve, so the pinned behaviour is "slide feature is 0 downstream", not "no
NaN ever appears in `*_slide_features`". This is documented, not changed --
see the task report for whether this is judged sufficient.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.branches.bs import bs_features, bs_slide_features  # noqa: E402
from src.models.branches.sh import sh_features, sh_slide_features  # noqa: E402
from src.models.branches.sj import sj_features, sj_slide_features  # noqa: E402

_TOL = 1e-4  # float32 headroom; SJ's eigh path is the noisiest of the three.


class _InvarianceBase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.dim = 64
        self.basis = torch.linalg.qr(torch.randn(self.dim, self.dim))[0]
        self.bag = torch.randn(200, self.dim) * 3.0 + 5.0
        self.shift = torch.randn(self.dim) * 2.0
        self.scale = 2.7


class TestBsInvariance(_InvarianceBase):
    def test_translation_invariant(self):
        f0 = bs_slide_features(self.bag, self.basis, 8)
        f1 = bs_slide_features(self.bag + self.shift, self.basis, 8)
        self.assertLess((f0 - f1).abs().max().item(), _TOL)

    def test_scaling_shifts_by_log_s_squared(self):
        f0 = bs_slide_features(self.bag, self.basis, 8)
        f1 = bs_slide_features(self.bag * self.scale, self.basis, 8)
        expected_shift = torch.log(torch.tensor(self.scale ** 2))
        self.assertLess(((f1 - f0) - expected_shift).abs().max().item(), _TOL)


class TestShInvariance(_InvarianceBase):
    def test_translation_invariant(self):
        f0 = sh_slide_features(self.bag, self.basis, 8, wide=16)
        f1 = sh_slide_features(self.bag + self.shift, self.basis, 8, wide=16)
        self.assertLess((f0 - f1).abs().max().item(), _TOL)

    def test_scaling_invariant(self):
        f0 = sh_slide_features(self.bag, self.basis, 8, wide=16)
        f1 = sh_slide_features(self.bag * self.scale, self.basis, 8, wide=16)
        self.assertLess((f0 - f1).abs().max().item(), _TOL)


class TestSjInvariance(_InvarianceBase):
    def test_translation_invariant(self):
        f0 = sj_slide_features(self.bag, self.basis, 8)
        f1 = sj_slide_features(self.bag + self.shift, self.basis, 8)
        self.assertLess((f0 - f1).abs().max().item(), _TOL)

    def test_scaling_invariant(self):
        f0 = sj_slide_features(self.bag, self.basis, 8)
        f1 = sj_slide_features(self.bag * self.scale, self.basis, 8)
        self.assertLess((f0 - f1).abs().max().item(), _TOL)


class TestDegenerateBagNanHandling(unittest.TestCase):
    """A3: N=1 slides produce NaN inside *_slide_features (std has 0 dof), but
    the *_features wrapper's stack()+nan_to_num absorbs it into a 0 feature
    before the ridge solve. This pins that absorption, not the NaN itself."""

    def setUp(self):
        torch.manual_seed(3)
        self.dim = 32
        self.basis = torch.linalg.qr(torch.randn(self.dim, self.dim))[0]

    def test_sh_slide_features_n1_is_nan(self):
        bag = torch.randn(1, self.dim)
        feat = sh_slide_features(bag, self.basis, 8, wide=16)
        self.assertTrue(torch.isnan(feat).any())

    def test_sj_slide_features_n1_is_nan(self):
        bag = torch.randn(1, self.dim)
        feat = sj_slide_features(bag, self.basis, 8)
        self.assertTrue(torch.isnan(feat).any())

    def test_bs_slide_features_n1_is_finite(self):
        # BS reads a trace, not a std ratio, so N=1 stays finite (variance of
        # a single centred point is exactly 0, clamped by _EPS before log()).
        bag = torch.randn(1, self.dim)
        feat = bs_slide_features(bag, self.basis, 8)
        self.assertFalse(torch.isnan(feat).any())
        self.assertFalse(torch.isinf(feat).any())

    def _config(self, **overrides):
        class _Cfg:
            pass
        cfg = _Cfg()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def _mixed_bags(self):
        # One degenerate N=1 bag mixed with ordinary bags, both in context and
        # query, so the *_features wrapper's stack() must survive a NaN row.
        torch.manual_seed(5)
        normal = lambda n: torch.randn(n, self.dim)
        context_bags = [normal(50), normal(1), normal(80)]
        context_labels = torch.tensor([0, 1, 0])
        query_bags = [normal(1), normal(60)]
        return context_bags, context_labels, query_bags

    def test_sh_features_absorbs_n1_bag_without_nan(self):
        context_bags, context_labels, query_bags = self._mixed_bags()
        cfg = self._config(sh_wide=16, sh_dim=8, sh_lambda=1.0)
        out = sh_features(cfg, context_bags, context_labels, query_bags, self.basis)
        self.assertFalse(torch.isnan(out).any())

    def test_sj_features_absorbs_n1_bag_without_nan(self):
        context_bags, context_labels, query_bags = self._mixed_bags()
        cfg = self._config(sj_dim=8, sj_lambda=1.0)
        out = sj_features(cfg, context_bags, context_labels, query_bags, self.basis)
        self.assertFalse(torch.isnan(out).any())

    def test_bs_features_handles_n1_bag_without_nan(self):
        context_bags, context_labels, query_bags = self._mixed_bags()
        cfg = self._config(bs_dim=8, bs_lambda=1.0)
        out = bs_features(cfg, context_bags, context_labels, query_bags, self.basis)
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
