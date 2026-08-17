"""Adaptive-rank DD must reduce to today's DD at rank 1 (docs SS146).

The arm is only interpretable if `rank_max=1` is bit-for-bit the existing
`_dd_distance_features`. If it is not, every "rank 2 gains X" number silently
mixes the rank change with an implementation difference.
"""

import unittest

import torch

from src.models.dd_adaptive_rank import (
    AdaptiveRankConfig,
    _welch_t,
    adaptive_dd_distance_features,
    dispersion_directions,
)
from src.models.set_transformer_ridge import CovarianceMeanLearnablePDDCTMLPModel

SKETCH = 12


def lineage_model():
    torch.manual_seed(0)
    return CovarianceMeanLearnablePDDCTMLPModel(
        input_dim=48, token_dim=16, num_heads=1, num_layers=1, feedforward_dim=16,
        num_summary_tokens=1, max_cells=4096, dropout=0.0, covariance_sketch_dim=SKETCH,
        ridge_lambda=1.0, ridge_logit_scale=2.0, num_classes=2,
        ct_num_tokens=16, ct_cells_per_bag=64, ct_temperature=0.5, ct_eps=1e-6,
        ct_head_hidden_dims=[], dd_shrinkage=0.25, dd_eps=1e-6,
    ).eval()


def covariances(seed=0, context=14, query=5):
    """Symmetric positive-definite sketched covariances, as DD receives them."""
    generator = torch.Generator().manual_seed(seed)
    def batch(count):
        factors = torch.randn(count, SKETCH, SKETCH * 2, generator=generator)
        return factors @ factors.transpose(-1, -2) / (SKETCH * 2)
    labels = torch.tensor([i % 2 for i in range(context)])
    return batch(context), labels, batch(query)


class Rank1EquivalenceTest(unittest.TestCase):
    def test_matches_the_lineage_at_rank_1(self):
        model = lineage_model()
        for seed in range(4):
            context, labels, query = covariances(seed)
            expected, expected_separation = model._dd_distance_features(
                context, labels, query
            )
            actual, _, kept = adaptive_dd_distance_features(
                context, labels, query, AdaptiveRankConfig(rank_max=1)
            )
            self.assertEqual(kept, 1)
            self.assertTrue(
                torch.allclose(actual, expected, atol=1e-5, rtol=1e-5),
                f"seed {seed}: {actual[:2].tolist()} vs {expected[:2].tolist()}",
            )
            del expected_separation

    def test_top_direction_matches_dd_direction(self):
        model = lineage_model()
        context, labels, _ = covariances(1)
        directions, eigenvalues = dispersion_directions(
            context, labels, AdaptiveRankConfig()
        )
        expected = model._dd_direction(context, labels)
        # eigh fixes sign arbitrarily; the induced scalar s_b = u^T C u does not
        # depend on it, so agreement up to sign is the real contract.
        self.assertTrue(
            torch.allclose(directions[:, 0], expected, atol=1e-5)
            or torch.allclose(directions[:, 0], -expected, atol=1e-5)
        )
        self.assertTrue(
            torch.all(eigenvalues.abs()[:-1] >= eigenvalues.abs()[1:] - 1e-6)
        )

    def test_infinite_threshold_pins_rank_1(self):
        context, labels, query = covariances(2)
        one, _, kept_one = adaptive_dd_distance_features(
            context, labels, query, AdaptiveRankConfig(rank_max=1)
        )
        many, _, kept_many = adaptive_dd_distance_features(
            context, labels, query,
            AdaptiveRankConfig(rank_max=SKETCH, t_threshold=float("inf")),
        )
        self.assertEqual((kept_one, kept_many), (1, 1))
        self.assertTrue(torch.equal(one, many))


class GateTest(unittest.TestCase):
    def test_zero_threshold_keeps_every_candidate(self):
        context, labels, query = covariances(3)
        _, _, kept = adaptive_dd_distance_features(
            context, labels, query,
            AdaptiveRankConfig(rank_max=5, t_threshold=0.0),
        )
        self.assertEqual(kept, 5)

    def test_threshold_is_monotone_in_how_many_it_keeps(self):
        context, labels, query = covariances(4, context=40)
        counts = [
            adaptive_dd_distance_features(
                context, labels, query,
                AdaptiveRankConfig(rank_max=8, t_threshold=threshold),
            )[2]
            for threshold in (0.0, 1.0, 2.0, 3.0, 100.0)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True))
        self.assertEqual(counts[-1], 1)

    def test_distances_stay_two_wide_so_the_fixed_head_is_untouched(self):
        context, labels, query = covariances(5, query=7)
        distances, _, _ = adaptive_dd_distance_features(
            context, labels, query,
            AdaptiveRankConfig(rank_max=6, t_threshold=0.0),
        )
        self.assertEqual(distances.shape, (7, 2))

    def test_scale_by_rank_divides_out_the_direction_count(self):
        context, labels, query = covariances(6)
        raw, _, kept = adaptive_dd_distance_features(
            context, labels, query, AdaptiveRankConfig(rank_max=4, t_threshold=0.0)
        )
        scaled, _, _ = adaptive_dd_distance_features(
            context, labels, query,
            AdaptiveRankConfig(rank_max=4, t_threshold=0.0, scale_by_rank=True),
        )
        self.assertTrue(torch.allclose(scaled, raw / kept))


class WelchTest(unittest.TestCase):
    def test_separated_groups_beat_identical_ones(self):
        generator = torch.Generator().manual_seed(0)
        base = torch.randn(60, generator=generator)
        other = torch.randn(60, generator=generator)
        null = float(_welch_t(base, other, 1e-6).abs())
        shifted = float(_welch_t(base, other + 3.0, 1e-6).abs())
        self.assertLess(null, 2.5)
        self.assertGreater(shifted, 5.0)

    def test_degenerate_group_returns_zero_rather_than_nan(self):
        single = torch.tensor([1.0])
        self.assertEqual(float(_welch_t(single, torch.randn(10), 1e-6)), 0.0)


if __name__ == "__main__":
    unittest.main()
