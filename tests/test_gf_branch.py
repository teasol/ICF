"""Unit tests for src/models/branches/gf.py (GF Fisher Vector branch).

unittest style so `bash scripts/run_tests.sh` (unittest discover) collects them.
Everything here is synthetic and CPU-only; no GMM pickle and no PathoBench data
is touched.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.branches import gf  # noqa: E402
from src.models.branches.gf import (  # noqa: E402
    BackgroundGMM, gf_margins_from_descriptors, gf_responsibilities, gf_slide_features,
)


def unit_gmm(n_components: int, n_features: int, seed: int = 0) -> BackgroundGMM:
    g = torch.Generator().manual_seed(seed)
    return BackgroundGMM(
        weights=torch.full((n_components,), 1.0 / n_components),
        means=torch.randn(n_components, n_features, generator=g),
        variances=torch.rand(n_components, n_features, generator=g) + 0.5,
    )


class TestResponsibilities(unittest.TestCase):
    def test_rows_sum_to_one(self):
        gmm = unit_gmm(4, 6)
        bag = torch.randn(64, 6)
        resp = gf_responsibilities(bag, gmm)
        self.assertEqual(resp.shape, (64, 4))
        torch.testing.assert_close(resp.sum(dim=1), torch.ones(64), atol=1e-5, rtol=0)

    def test_matches_explicit_gaussian_density(self):
        """Compare against a direct per-component density evaluation."""
        gmm = unit_gmm(3, 5, seed=1)
        bag = torch.randn(17, 5)
        resp = gf_responsibilities(bag, gmm)

        dens = torch.empty(17, 3)
        for k in range(3):
            mu, var = gmm.means[k], gmm.variances[k]
            log_p = -0.5 * (
                5 * math.log(2 * math.pi) + var.log().sum()
                + ((bag - mu).square() / var).sum(dim=1)
            )
            dens[:, k] = gmm.weights[k].log() + log_p
        expected = torch.softmax(dens, dim=1)
        torch.testing.assert_close(resp, expected, atol=1e-5, rtol=1e-4)

    def test_far_component_gets_no_mass(self):
        gmm = BackgroundGMM(
            weights=torch.tensor([0.5, 0.5]),
            means=torch.stack([torch.zeros(4), torch.full((4,), 500.0)]),
            variances=torch.ones(2, 4),
        )
        resp = gf_responsibilities(torch.randn(32, 4), gmm)
        self.assertLess(float(resp[:, 1].max()), 1e-12)

    def test_nonfinite_input_fails_loudly(self):
        gmm = unit_gmm(2, 4)
        bag = torch.zeros(3, 4)
        bag[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "responsibilities"):
            gf_responsibilities(bag, gmm)


class TestFisherVector(unittest.TestCase):
    def test_shape_and_l2_norm(self):
        gmm = unit_gmm(8, 16)
        vector = gf_slide_features(torch.randn(256, 16), gmm)
        self.assertEqual(vector.shape, (2 * 8 * 16,))
        self.assertAlmostEqual(float(vector.norm()), 1.0, places=5)

    def test_mean_only_drops_variance_half(self):
        gmm = unit_gmm(8, 16)
        bag = torch.randn(256, 16)
        full = gf_slide_features(bag, gmm)
        mean_only = gf_slide_features(bag, gmm, include_variance=False)
        self.assertEqual(mean_only.shape, (8 * 16,))
        # Same mean-gradient direction; only the normalisation constant differs.
        cos = torch.nn.functional.cosine_similarity(
            full[:8 * 16], mean_only, dim=0
        )
        self.assertGreater(float(cos), 0.999)

    def test_single_component_matches_closed_form(self):
        """M=1, mu=0, sigma=1: G_mu = mean(x), G_sigma = (mean(x^2) - 1)/sqrt(2)."""
        n_features = 5
        gmm = BackgroundGMM(
            weights=torch.ones(1),
            means=torch.zeros(1, n_features),
            variances=torch.ones(1, n_features),
        )
        bag = torch.randn(400, n_features)
        vector = gf_slide_features(bag, gmm, alpha=1.0)

        g_mu = bag.mean(dim=0)
        g_sigma = (bag.square().mean(dim=0) - 1.0) / math.sqrt(2.0)
        expected = torch.cat([g_mu, g_sigma])
        expected = expected / expected.norm()
        torch.testing.assert_close(vector, expected, atol=1e-5, rtol=1e-4)

    def test_unoccupied_component_block_is_zero(self):
        """A component no patch reaches contributes exactly zero to the descriptor.

        This is the property that makes the background GMM's component allocation
        matter: a cohort that lands on one component uses only that component's
        block of the Fisher Vector.
        """
        n_features = 4
        gmm = BackgroundGMM(
            weights=torch.tensor([0.5, 0.5]),
            means=torch.stack([torch.zeros(n_features), torch.full((n_features,), 500.0)]),
            variances=torch.ones(2, n_features),
        )
        vector = gf_slide_features(torch.randn(128, n_features), gmm)
        mean_block_far = vector[n_features:2 * n_features]
        var_block_far = vector[3 * n_features:4 * n_features]
        self.assertLess(float(mean_block_far.abs().max()), 1e-6)
        self.assertLess(float(var_block_far.abs().max()), 1e-6)

    def test_power_normalisation_is_signed(self):
        gmm = unit_gmm(4, 8)
        bag = torch.randn(128, 8)
        alpha_one = gf_slide_features(bag, gmm, alpha=1.0)
        alpha_half = gf_slide_features(bag, gmm, alpha=0.5)
        # Signed power keeps every sign and only compresses magnitudes.
        nonzero = alpha_one.abs() > 1e-8
        torch.testing.assert_close(
            alpha_half[nonzero].sign(), alpha_one[nonzero].sign(), atol=0, rtol=0
        )

    def test_occupancy_is_a_distribution(self):
        gmm = unit_gmm(6, 8)
        _, occupancy = gf_slide_features(torch.randn(200, 8), gmm, return_occupancy=True)
        self.assertEqual(occupancy.shape, (6,))
        self.assertAlmostEqual(float(occupancy.sum()), 1.0, places=5)

    def test_chunking_does_not_change_the_result(self):
        gmm = unit_gmm(4, 8)
        bag = torch.randn(500, 8)
        full = gf_slide_features(bag, gmm)
        original = gf._CHUNK
        try:
            gf._CHUNK = 37
            chunked = gf_slide_features(bag, gmm)
        finally:
            gf._CHUNK = original
        torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-4)

    def test_empty_bag_returns_zeros(self):
        gmm = unit_gmm(3, 4)
        vector = gf_slide_features(torch.zeros(0, 4), gmm)
        self.assertEqual(vector.shape, (2 * 3 * 4,))
        self.assertEqual(float(vector.abs().max()), 0.0)


class TestReadout(unittest.TestCase):
    def test_margins_are_finite_and_separate_a_planted_signal(self):
        torch.manual_seed(0)
        gmm = unit_gmm(4, 8)
        # Class 1 bags are shifted, so the Fisher Vector should carry the signal.
        ctx_bags = [torch.randn(120, 8) + (2.0 if i % 2 else 0.0) for i in range(24)]
        qry_bags = [torch.randn(120, 8) + (2.0 if i % 2 else 0.0) for i in range(12)]
        ctx_labels = torch.tensor([i % 2 for i in range(24)], dtype=torch.long)
        qry_labels = torch.tensor([i % 2 for i in range(12)], dtype=torch.long)

        ctx = torch.stack([gf_slide_features(b, gmm) for b in ctx_bags])
        qry = torch.stack([gf_slide_features(b, gmm) for b in qry_bags])
        margin = gf_margins_from_descriptors(ctx, ctx_labels, qry)

        self.assertEqual(margin.shape, (12,))
        self.assertTrue(bool(torch.isfinite(margin).all()))
        self.assertGreater(
            float(margin[qry_labels == 1].mean() - margin[qry_labels == 0].mean()), 0.0
        )

    def test_label_flip_negates_margin(self):
        torch.manual_seed(1)
        context = torch.randn(20, 12)
        query = torch.randn(7, 12)
        labels = torch.tensor([0, 1] * 10, dtype=torch.long)
        margin = gf_margins_from_descriptors(context, labels, query)
        flipped = gf_margins_from_descriptors(context, 1 - labels, query)
        torch.testing.assert_close(flipped, -margin, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
