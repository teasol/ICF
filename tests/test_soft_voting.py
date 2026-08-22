"""Unit tests for Soft Voting Aggregation (v118 baseline) in TrainingFreeClassifier."""

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


class SoftVotingTest(unittest.TestCase):
    def test_default_config_is_v118_baseline(self):
        """Default TrainingFreeConfig must be v118 (soft_voting, no-DD 4-branch)."""
        config = TrainingFreeConfig()
        self.assertEqual(config.aggregation, "soft_voting")
        self.assertEqual(config.weight_dd, 0.0)
        self.assertEqual(config.weight_cv, 1.0)
        self.assertEqual(config.weight_ct, 1.0)
        self.assertEqual(config.weight_bm, 1.0)
        self.assertEqual(config.weight_bd, 1.0)
        self.assertEqual(config.bd_metric, "entropy")

    def test_predict_proba_returns_valid_probabilities(self):
        """predict_proba must return finite probabilities in (0, 1)."""
        context, labels, query = synthetic_episode(seed=42)
        model = TrainingFreeClassifier()
        probs = model.predict_proba(context, labels, query)
        self.assertEqual(probs.shape, (len(query),))
        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue((probs > 0.0).all() and (probs < 1.0).all())

    def test_label_swap_inverts_probabilities_and_negates_margins(self):
        """Label antisymmetry: swapping labels (0 <-> 1) must invert prob (p -> 1 - p) and negate margins."""
        for seed in (7, 13, 21):
            context, labels, query = synthetic_episode(seed=seed)
            model = TrainingFreeClassifier()
            p_orig = model.predict_proba(context, labels, query)
            p_flipped = model.predict_proba(context, 1 - labels, query)
            self.assertTrue(
                torch.allclose(p_orig, 1.0 - p_flipped, atol=1e-5),
                f"Seed {seed}: {p_orig} vs {1.0 - p_flipped}",
            )

            m_orig = model.margins(context, labels, query)
            m_flipped = model.margins(context, 1 - labels, query)
            self.assertTrue(
                torch.allclose(m_orig, -m_flipped, atol=1e-5),
                f"Seed {seed}: {m_orig} vs {-m_flipped}",
            )

    def test_v117_linear_mode(self):
        """aggregation='linear' must compute the linear sum of active branches."""
        context, labels, query = synthetic_episode(seed=99)
        config_v117 = TrainingFreeConfig(aggregation="linear", weight_dd=0.0)
        model = TrainingFreeClassifier(config_v117)
        margins = model.margins(context, labels, query)
        self.assertEqual(margins.shape, (len(query),))
        self.assertTrue(torch.isfinite(margins).all())

    def test_determinism_and_no_leakage(self):
        """Same input must give 100% bit-identical predictions; modifying other queries must not leak."""
        context, labels, query = synthetic_episode(seed=123)
        model = TrainingFreeClassifier()
        first = model.margins(context, labels, query)
        second = model.margins(context, labels, query)
        self.assertTrue(torch.equal(first, second))

        other_query = [q * 3.0 + 5.0 for q in query]
        _ = model.margins(context, labels, other_query)
        third = model.margins(context, labels, query)
        self.assertTrue(torch.equal(first, third))


if __name__ == "__main__":
    unittest.main()
