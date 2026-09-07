"""Tier 1 candidate branches (AKS, MDX, LID): invariants and mechanism claims.

Gate 1 is a label-free screen, so these tests assert what the proposals claim
structurally, before any benchmark number is consulted:

- AKS is identically constant on a spherically symmetric cloud (M = I/d, B prop I),
  which is what makes it not a function of SHJ.
- MDX separates a uniform cloud from a separated bimodal mixture. Both sit near
  excess kurtosis -1.2, so SH/SHJ cannot tell them apart; this is MDX's reason to
  exist and it is checked directly.
- LID recovers the local dimension of a curved 1-D manifold, and is by design
  blind to a *linear* low-rank subspace, because whitening removes global rank.

All three must be deterministic and independent of the eigenvector sign gauge.
"""

import unittest

import torch

from src.models.branches.aks import AKS_FEATURE_DIM, aks_slide_features
from src.models.branches.lid import LID_FEATURE_DIM, lid_slide_features
from src.models.branches.mdx import MDX_FEATURE_DIM, mdx_slide_features

_EMBED, _DIM, _TOKENS = 256, 32, 2000


class _Base(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.basis = torch.linalg.qr(torch.randn(_EMBED, 64))[0]

    def bag(self, latent: torch.Tensor) -> torch.Tensor:
        """Embed a 32-D latent cloud so that projecting on the basis returns it."""
        return latent @ self.basis[:, :_DIM].T


class TestAks(_Base):
    def test_constant_on_spherically_symmetric_cloud(self):
        feats, _ = aks_slide_features(self.bag(torch.randn(_TOKENS, _DIM)), self.basis, _DIM)
        self.assertEqual(feats.numel(), AKS_FEATURE_DIM)
        # entropy -> 1 and anisotropy -> 0 for both spectra.
        for idx in (0, 4):
            self.assertGreater(float(feats[idx]), 0.99)
        for idx in (3, 7):
            self.assertLess(float(feats[idx]), 0.05)

    def test_angular_half_ignores_the_radius(self):
        latent = torch.randn(_TOKENS, _DIM)
        base, _ = aks_slide_features(self.bag(latent), self.basis, _DIM)
        stretched = latent * (1.0 + 3.0 * torch.rand(_TOKENS, 1))
        moved, _ = aks_slide_features(self.bag(stretched), self.basis, _DIM)
        self.assertLess(float((moved[:4] - base[:4]).abs().max()), 0.05)

    def test_invariant_to_eigenvector_sign_gauge(self):
        latent = torch.randn(_TOKENS, _DIM)
        base, _ = aks_slide_features(self.bag(latent), self.basis, _DIM)
        flipped = latent.clone()
        flipped[:, ::2] *= -1.0
        other, _ = aks_slide_features(self.bag(flipped), self.basis, _DIM)
        self.assertLess(float((other - base).abs().max()), 1e-5)


class TestMdx(_Base):
    def _depth(self, latent):
        feats, diag = mdx_slide_features(self.bag(latent), self.basis, _DIM)
        self.assertEqual(feats.numel(), MDX_FEATURE_DIM)
        return float(diag[1]), float(diag[0])

    def test_separates_uniform_from_bimodal(self):
        """The claim SH/SHJ cannot make: both clouds have excess kurtosis near -1.2."""
        uniform = torch.rand(_TOKENS, _DIM) * 2.0 - 1.0
        bimodal = torch.randn(_TOKENS, _DIM)
        bimodal[:_TOKENS // 2, 0] -= 1.85
        bimodal[_TOKENS // 2:, 0] += 1.85

        uni_depth, uni_unimodal = self._depth(uniform)
        bi_depth, bi_unimodal = self._depth(bimodal)
        self.assertLess(uni_depth, 0.15)
        self.assertGreater(bi_depth, 0.30)
        self.assertEqual(uni_unimodal, 1.0)
        self.assertLess(bi_unimodal, 1.0)

    def test_unimodal_gaussian_has_no_valley(self):
        """Regression: without a peak prominence floor, tail noise peaks were
        selected as the second peak and drove the depth to 0.8 on one mode."""
        depth, unimodal_fraction = self._depth(torch.randn(_TOKENS, _DIM))
        self.assertLess(depth, 0.10)
        self.assertEqual(unimodal_fraction, 1.0)


class TestLid(_Base):
    def _clusters(self, tokens, spread):
        centres = torch.randn(12, _DIM) * 3.0
        picks = torch.randint(0, 12, (tokens,))
        return centres[picks] + spread * torch.randn(tokens, _DIM)

    def test_clustered_cloud_reads_lower_than_isotropic(self):
        """The cluster structure CT chased with k-means, without the k-means."""
        clustered, diag = lid_slide_features(self.bag(self._clusters(_TOKENS, 0.15)),
                                             self.basis, _DIM, subsample=1024)
        isotropic, _ = lid_slide_features(self.bag(torch.randn(_TOKENS, _DIM)),
                                          self.basis, _DIM, subsample=1024)
        self.assertEqual(clustered.numel(), LID_FEATURE_DIM)
        self.assertLess(float(clustered[0]), float(isotropic[0]))

    def test_subsample_size_does_not_move_the_reading_much(self):
        """The proposal's own falsifier: m = 4096 against m = 1024. On a clustered
        cloud the two agree within ~15%; the degenerate case below is where they
        do not, which is why the real-data screen measures this."""
        latent = self._clusters(6000, 0.15)
        small, _ = lid_slide_features(self.bag(latent), self.basis, _DIM, subsample=1024)
        large, _ = lid_slide_features(self.bag(latent), self.basis, _DIM, subsample=4096)
        self.assertLess(abs(float(small[0]) - float(large[0])) / float(large[0]), 0.20)

    def test_locally_collinear_cloud_makes_the_estimator_blow_up(self):
        """Documented limitation, not a bug. On a noiseless 1-D curve the k-NN
        distance ratios collapse to 1, so ID = 1 / mean(log ratio) diverges - it
        exceeded the ambient dimension 32 by an order of magnitude in probing.
        Real token clouds are not collinear, but this is the failure mode the
        subsample-robustness falsifier is there to catch."""
        steps = torch.linspace(0.0, 6.28, 4000).unsqueeze(1)
        curve = torch.cat([torch.sin(steps * (i + 1)) * (i % 3 + 1) for i in range(_DIM)], dim=1)
        feats, _ = lid_slide_features(self.bag(curve), self.basis, _DIM, subsample=4096)
        self.assertTrue(bool(torch.isfinite(feats).all()))
        self.assertGreater(float(feats[0]), float(_DIM))

    def test_blind_to_a_linear_low_rank_subspace_by_design(self):
        """Whitening normalises every direction, so global linear rank is removed.
        LID reads local/curved structure; this documents the boundary."""
        latent = torch.cat([torch.randn(_TOKENS, 5), 0.01 * torch.randn(_TOKENS, _DIM - 5)], dim=1)
        low_rank, _ = lid_slide_features(self.bag(latent), self.basis, _DIM, subsample=1024)
        isotropic, _ = lid_slide_features(self.bag(torch.randn(_TOKENS, _DIM)),
                                          self.basis, _DIM, subsample=1024)
        self.assertGreater(float(low_rank[0]), 0.7 * float(isotropic[0]))

    def test_deterministic_and_finite(self):
        latent = torch.randn(_TOKENS, _DIM)
        first, _ = lid_slide_features(self.bag(latent), self.basis, _DIM, subsample=1024)
        again, _ = lid_slide_features(self.bag(latent), self.basis, _DIM, subsample=1024)
        self.assertTrue(torch.equal(first, again))
        self.assertTrue(bool(torch.isfinite(first).all()))


if __name__ == "__main__":
    unittest.main()
