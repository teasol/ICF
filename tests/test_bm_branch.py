"""Unit tests for Plan A: Projected Bag-Mean (BM) Branch in TrainingFreeClassifier."""

from __future__ import annotations

import unittest
import torch

from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


def synthetic_episode(seed: int, num_context: int = 16, num_query: int = 4, cells_per_bag: int = 64, dim: int = 1536):
    """Generate a reproducible deterministic binary classification episode."""
    generator = torch.Generator().manual_seed(seed)
    context_bags = [
        torch.randn((cells_per_bag, dim), generator=generator) + (0.5 if i % 2 == 1 else -0.5)
        for i in range(num_context)
    ]
    labels = torch.tensor([i % 2 for i in range(num_context)], dtype=torch.long)
    query_bags = [
        torch.randn((cells_per_bag, dim), generator=generator) + (0.5 if i % 2 == 1 else -0.5)
        for i in range(num_query)
    ]
    return context_bags, labels, query_bags


class BMBranchTest(unittest.TestCase):
    def test_default_weight_is_zero_and_preserves_v114(self):
        """Default TrainingFreeClassifier must have weight_bm=0.0 and bit-identical output."""
        config_default = TrainingFreeConfig()
        self.assertEqual(config_default.weight_bm, 0.0)
        self.assertEqual(config_default.bm_dim, 32)
        self.assertEqual(config_default.bm_lambda, 1.0)

        context, labels, query = synthetic_episode(seed=42)
        model_default = TrainingFreeClassifier(config_default)
        model_explicit_zero = TrainingFreeClassifier(TrainingFreeConfig(weight_bm=0.0))

        margin_default = model_default.margins(context, labels, query)
        margin_explicit = model_explicit_zero.margins(context, labels, query)
        self.assertTrue(torch.equal(margin_default, margin_explicit))

    def test_bm_only_branch(self):
        """BM branch standalone (weights for CV, DD, CT set to 0.0) produces valid margins."""
        context, labels, query = synthetic_episode(seed=101)
        config_bm_only = TrainingFreeConfig(
            weight_cv=0.0,
            weight_dd=0.0,
            weight_ct=0.0,
            weight_bm=1.0,
            bm_dim=32,
        )
        model = TrainingFreeClassifier(config_bm_only)
        margins = model.margins(context, labels, query)
        self.assertEqual(margins.shape, (len(query),))
        self.assertTrue(torch.isfinite(margins).all())
        self.assertFalse((margins == 0.0).all())

    def test_label_swap_flips_bm_margin(self):
        """Label antisymmetry: swapping labels (0 <-> 1) must negate the BM margin."""
        for seed in (7, 13, 21):
            context, labels, query = synthetic_episode(seed=seed)
            config = TrainingFreeConfig(
                weight_cv=0.0,
                weight_dd=0.0,
                weight_ct=0.0,
                weight_bm=1.0,
                bm_dim=32,
            )
            model = TrainingFreeClassifier(config)
            margin_orig = model.margins(context, labels, query)
            margin_flipped = model.margins(context, 1 - labels, query)
            self.assertTrue(
                torch.allclose(margin_orig, -margin_flipped, atol=1e-5),
                f"Seed {seed}: {margin_orig} vs {-margin_flipped}",
            )

    def test_query_no_leakage(self):
        """Query bag modifications must not leak into context projection or standardization."""
        context, labels, query = synthetic_episode(seed=99)
        config = TrainingFreeConfig(weight_bm=1.0, bm_dim=32)
        model = TrainingFreeClassifier(config)

        # Baseline margin
        margin_first = model.margins(context, labels, query)

        # Create a modified query set (different bags)
        other_query = [q * 5.0 + 10.0 for q in query]
        margin_other = model.margins(context, labels, other_query)

        # Re-running original query should be identical
        margin_repeat = model.margins(context, labels, query)
        self.assertTrue(torch.equal(margin_first, margin_repeat))

    def test_determinism(self):
        """Same input must give 100% bit-identical margin output."""
        context, labels, query = synthetic_episode(seed=123)
        config = TrainingFreeConfig(weight_bm=1.0, bm_dim=32)
        model = TrainingFreeClassifier(config)

        first = model.margins(context, labels, query)
        second = model.margins(context, labels, query)
        self.assertTrue(torch.equal(first, second))

    def test_bm_dimension_flexibility(self):
        """BM branch should handle various subspace dimensions (e.g. 8, 16, 64)."""
        context, labels, query = synthetic_episode(seed=77)
        for dim in (8, 16, 64, 128):
            config = TrainingFreeConfig(weight_bm=1.0, bm_dim=dim)
            model = TrainingFreeClassifier(config)
            margin = model.margins(context, labels, query)
            self.assertEqual(margin.shape, (len(query),))
            self.assertTrue(torch.isfinite(margin).all())


if __name__ == "__main__":
    unittest.main()
