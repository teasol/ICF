"""Tests for the evaluation-protocol statistics.

These guard the machinery that exists specifically to stop the project from
repeating the v21 mistake of reporting bare point estimates: a whole line of
architecture work was pursued on 0.04-AUROC differences that later turned out
to be indistinguishable from noise at n=87.
"""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_protocol import auroc as protocol_auroc  # noqa: E402
from scripts.evaluate_protocol import bootstrap_interval, log_loss  # noqa: E402
from scripts.test import auroc_confidence_interval, binary_metrics  # noqa: E402


def separable(n_positive: int, n_negative: int, gap: float) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    positive = torch.rand(n_positive, generator=generator) * 0.4 + gap
    negative = torch.rand(n_negative, generator=generator) * 0.4
    probability = torch.cat([positive, negative]).clamp(0.001, 0.999)
    target = torch.cat(
        [torch.ones(n_positive, dtype=torch.long), torch.zeros(n_negative, dtype=torch.long)]
    )
    return probability, target


class TestAurocPointEstimate(unittest.TestCase):
    def test_perfect_and_inverted_separation(self) -> None:
        probability = torch.tensor([0.9, 0.8, 0.2, 0.1])
        target = torch.tensor([1, 1, 0, 0])
        self.assertAlmostEqual(protocol_auroc(probability, target), 1.0, places=6)
        self.assertAlmostEqual(protocol_auroc(1.0 - probability, target), 0.0, places=6)

    def test_ties_score_one_half(self) -> None:
        probability = torch.tensor([0.5, 0.5, 0.5, 0.5])
        target = torch.tensor([1, 1, 0, 0])
        self.assertAlmostEqual(protocol_auroc(probability, target), 0.5, places=6)

    def test_single_class_is_nan(self) -> None:
        probability = torch.tensor([0.3, 0.7])
        self.assertTrue(torch.isnan(torch.tensor(protocol_auroc(probability, torch.tensor([1, 1])))))


class TestConfidenceInterval(unittest.TestCase):
    def test_interval_brackets_the_point_estimate(self) -> None:
        probability, target = separable(37, 50, gap=0.25)
        low, high = auroc_confidence_interval(probability, target, samples=400)
        point = protocol_auroc(probability, target)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_smaller_cohort_gives_a_wider_interval(self) -> None:
        """The core claim justifying the protocol: n drives resolvability."""
        small_probability, small_target = separable(8, 10, gap=0.25)
        large_probability, large_target = separable(370, 500, gap=0.25)
        small_low, small_high = auroc_confidence_interval(
            small_probability, small_target, samples=400
        )
        large_low, large_high = auroc_confidence_interval(
            large_probability, large_target, samples=400
        )
        self.assertGreater(small_high - small_low, large_high - large_low)

    def test_ici_sized_cohort_cannot_resolve_small_gaps(self) -> None:
        """At n=87 the interval must be far wider than the gaps v21 chased."""
        probability, target = separable(37, 50, gap=0.12)
        low, high = auroc_confidence_interval(probability, target, samples=800)
        self.assertGreater(high - low, 0.04)

    def test_single_class_interval_is_nan(self) -> None:
        probability = torch.tensor([0.3, 0.7, 0.5])
        low, high = auroc_confidence_interval(probability, torch.tensor([1, 1, 1]), samples=50)
        self.assertTrue(torch.isnan(torch.tensor(low)))
        self.assertTrue(torch.isnan(torch.tensor(high)))

    def test_bootstrap_interval_is_deterministic_for_a_fixed_generator(self) -> None:
        probability, target = separable(37, 50, gap=0.2)
        first = bootstrap_interval(
            probability, target, 300, torch.Generator().manual_seed(7)
        )
        second = bootstrap_interval(
            probability, target, 300, torch.Generator().manual_seed(7)
        )
        self.assertEqual(first, second)


class TestLogLoss(unittest.TestCase):
    def test_confident_and_correct_beats_confident_and_wrong(self) -> None:
        target = torch.tensor([1, 1, 0, 0])
        good = torch.tensor([0.95, 0.9, 0.05, 0.1])
        bad = 1.0 - good
        self.assertLess(log_loss(good, target), log_loss(bad, target))

    def test_saturated_probabilities_stay_finite(self) -> None:
        target = torch.tensor([1, 0])
        value = log_loss(torch.tensor([1.0, 0.0]), target)
        self.assertTrue(torch.isfinite(torch.tensor(value)))


class TestBinaryMetricsReportsInterval(unittest.TestCase):
    def test_metrics_carry_a_confidence_interval(self) -> None:
        probability, target = separable(37, 50, gap=0.2)
        metrics = binary_metrics(
            {
                "target": target,
                "prediction": (probability > 0.5).long(),
                "probabilities": torch.stack([1.0 - probability, probability], dim=1),
            }
        )
        self.assertIn("auroc_ci_low", metrics)
        self.assertIn("auroc_ci_high", metrics)
        self.assertLessEqual(metrics["auroc_ci_low"], metrics["auroc"])
        self.assertGreaterEqual(metrics["auroc_ci_high"], metrics["auroc"])


if __name__ == "__main__":
    unittest.main()
