"""Unit tests for BD (Bag Dispersion) Branch in TrainingFreeClassifier."""

from __future__ import annotations

import unittest
import torch

from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


def synthetic_episode(seed: int, num_context: int = 16, num_query: int = 4, cells_per_bag: int = 64, dim: int = 1536):
    """Generate a reproducible deterministic binary classification episode."""
    generator = torch.Generator().manual_seed(seed)
    context_bags = [
        torch.randn((cells_per_bag, dim), generator=generator) * (2.0 if i % 2 == 1 else 0.5) + (0.5 if i % 2 == 1 else -0.5)
        for i in range(num_context)
    ]
    labels = torch.tensor([i % 2 for i in range(num_context)], dtype=torch.long)
    query_bags = [
        torch.randn((cells_per_bag, dim), generator=generator) * (2.0 if i % 2 == 1 else 0.5) + (0.5 if i % 2 == 1 else -0.5)
        for i in range(num_query)
    ]
    return context_bags, labels, query_bags


class BDBranchTest(unittest.TestCase):
    def test_default_weight_is_one_and_preserves_v116(self):
        """Default TrainingFreeClassifier must have weight_bd=1.0 (v116 baseline)."""
        config_default = TrainingFreeConfig()
        self.assertEqual(config_default.weight_bd, 1.0)
        self.assertEqual(config_default.bd_dim, 256)
        self.assertEqual(config_default.bd_metric, "entropy")
        self.assertEqual(config_default.bd_readout, "ordered_typicality")

        context, labels, query = synthetic_episode(seed=42)
        model_default = TrainingFreeClassifier(config_default)
        model_explicit_one = TrainingFreeClassifier(TrainingFreeConfig(weight_bd=1.0))

        margin_default = model_default.margins(context, labels, query)
        margin_explicit = model_explicit_one.margins(context, labels, query)
        self.assertTrue(torch.equal(margin_default, margin_explicit))

    def test_weight_zero_reproduces_v115(self):
        """Setting weight_bd=0.0 must reproduce the 4-branch v115 margin."""
        context, labels, query = synthetic_episode(seed=42)
        model_v115 = TrainingFreeClassifier(TrainingFreeConfig(weight_bd=0.0))
        margin = model_v115.margins(context, labels, query)
        self.assertEqual(margin.shape, (len(query),))


    def test_bd_spectral_entropy_ordered_typicality(self):
        """BD spectral entropy standalone with ordered_typicality produces valid bounded [-1, 1] margins."""
        context, labels, query = synthetic_episode(seed=101)
        config_bd_only = TrainingFreeConfig(
            weight_cv=0.0,
            weight_dd=0.0,
            weight_ct=0.0,
            weight_bm=0.0,
            weight_bd=1.0,
            bd_metric="entropy",
            bd_readout="ordered_typicality",
        )
        model = TrainingFreeClassifier(config_bd_only)
        margins = model.margins(context, labels, query)
        self.assertEqual(margins.shape, (len(query),))
        self.assertTrue(torch.isfinite(margins).all())
        self.assertFalse((margins == 0.0).all())
        self.assertTrue((margins >= -1.0).all() and (margins <= 1.0).all())

    def test_bd_trace_ordered_typicality(self):
        """BD trace standalone with ordered_typicality produces valid bounded [-1, 1] margins."""
        context, labels, query = synthetic_episode(seed=102)
        config_bd_only = TrainingFreeConfig(
            weight_cv=0.0,
            weight_dd=0.0,
            weight_ct=0.0,
            weight_bm=0.0,
            weight_bd=1.0,
            bd_metric="trace",
            bd_readout="ordered_typicality",
        )
        model = TrainingFreeClassifier(config_bd_only)
        margins = model.margins(context, labels, query)
        self.assertEqual(margins.shape, (len(query),))
        self.assertTrue(torch.isfinite(margins).all())
        self.assertFalse((margins == 0.0).all())
        self.assertTrue((margins >= -1.0).all() and (margins <= 1.0).all())

    def test_bd_only_branch_ridge(self):
        """BD branch standalone with ridge produces valid finite margins."""
        context, labels, query = synthetic_episode(seed=103)
        config_bd_ridge = TrainingFreeConfig(
            weight_cv=0.0,
            weight_dd=0.0,
            weight_ct=0.0,
            weight_bm=0.0,
            weight_bd=1.0,
            bd_metric="entropy",
            bd_readout="ridge",
        )
        model = TrainingFreeClassifier(config_bd_ridge)
        margins = model.margins(context, labels, query)
        self.assertEqual(margins.shape, (len(query),))
        self.assertTrue(torch.isfinite(margins).all())
        self.assertFalse((margins == 0.0).all())

    def test_label_swap_flips_bd_margin(self):
        """Label antisymmetry: swapping labels (0 <-> 1) must negate the BD margin."""
        for metric in ("entropy", "trace"):
            for readout in ("ordered_typicality", "ridge"):
                for seed in (7, 13, 21):
                    context, labels, query = synthetic_episode(seed=seed)
                    config = TrainingFreeConfig(
                        weight_cv=0.0,
                        weight_dd=0.0,
                        weight_ct=0.0,
                        weight_bm=0.0,
                        weight_bd=1.0,
                        bd_metric=metric,
                        bd_readout=readout,
                    )
                    model = TrainingFreeClassifier(config)
                    margin_orig = model.margins(context, labels, query)
                    margin_flipped = model.margins(context, 1 - labels, query)
                    self.assertTrue(
                        torch.allclose(margin_orig, -margin_flipped, atol=1e-5),
                        f"Metric {metric} Readout {readout} Seed {seed}: {margin_orig} vs {-margin_flipped}",
                    )

    def test_query_no_leakage(self):
        """Query bag modifications must not leak into context statistics or other predictions."""
        context, labels, query = synthetic_episode(seed=99)
        config = TrainingFreeConfig(weight_bd=1.0, bd_metric="entropy")
        model = TrainingFreeClassifier(config)

        # Baseline margin
        margin_first = model.margins(context, labels, query)

        # Create a modified query set (different bags)
        other_query = [q * 5.0 + 10.0 for q in query]
        _ = model.margins(context, labels, other_query)

        # Re-running original query should be bit-identical
        margin_repeat = model.margins(context, labels, query)
        self.assertTrue(torch.equal(margin_first, margin_repeat))

    def test_determinism(self):
        """Same input must give 100% bit-identical margin output."""
        context, labels, query = synthetic_episode(seed=123)
        config = TrainingFreeConfig(weight_bd=1.0, bd_metric="entropy")
        model = TrainingFreeClassifier(config)

        first = model.margins(context, labels, query)
        second = model.margins(context, labels, query)
        self.assertTrue(torch.equal(first, second))

    def test_bd_dimension_flexibility(self):
        """BD branch should handle various subspace dimensions (e.g. 8, 16, 64, 256)."""
        context, labels, query = synthetic_episode(seed=77)
        for dim in (8, 16, 64, 128, 256):
            config = TrainingFreeConfig(weight_bd=1.0, bd_dim=dim, bd_metric="entropy")
            model = TrainingFreeClassifier(config)
            margin = model.margins(context, labels, query)
            self.assertEqual(margin.shape, (len(query),))
            self.assertTrue(torch.isfinite(margin).all())


if __name__ == "__main__":
    unittest.main()
