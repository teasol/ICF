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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test import binary_metrics  # noqa: E402
from src.utils.metrics import (  # noqa: E402
    auroc as protocol_auroc,
    bootstrap_auroc_interval,
    cluster_members,
    log_loss,
    resample_index,
)


def auroc_confidence_interval(probability, target, samples=2000, seed=0):
    return bootstrap_auroc_interval(probability, target, samples=samples, seed=seed)


def cluster_bootstrap_interval(probability, target, episode, samples=2000, seed=0):
    return bootstrap_auroc_interval(
        probability, target, groups=episode, samples=samples, seed=seed
    )


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
        first = bootstrap_auroc_interval(probability, target, samples=300, seed=7)
        second = bootstrap_auroc_interval(probability, target, samples=300, seed=7)
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


def clustered(
    episodes: int, per_episode: int, offset_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Queries grouped into episodes that each carry their own difficulty."""
    generator = torch.Generator().manual_seed(0)
    offsets = torch.randn(episodes, generator=generator) * offset_scale
    probability, target, episode = [], [], []
    for index in range(episodes):
        labels = torch.randint(0, 2, (per_episode,), generator=generator)
        scores = torch.sigmoid(
            0.8 * labels.float()
            + offsets[index]
            + 0.3 * torch.randn(per_episode, generator=generator)
        )
        probability.append(scores)
        target.append(labels)
        episode.append(torch.full((per_episode,), index, dtype=torch.long))
    return torch.cat(probability), torch.cat(target).long(), torch.cat(episode)


class TestClusterBootstrap(unittest.TestCase):
    def test_clustering_widens_the_interval(self) -> None:
        """Correlated queries carry less information than their raw count."""
        probability, target, episode = clustered(100, 16, offset_scale=1.2)
        clustered_low, clustered_high = cluster_bootstrap_interval(
            probability, target, episode, samples=400, seed=1
        )
        # Treating every query as its own cluster is the naive per-query bootstrap.
        naive_low, naive_high = cluster_bootstrap_interval(
            probability, target, torch.arange(target.numel()), samples=400, seed=1
        )
        self.assertGreater(clustered_high - clustered_low, naive_high - naive_low)

    def test_interval_brackets_the_point_estimate(self) -> None:
        probability, target, episode = clustered(60, 10, offset_scale=0.8)
        low, high = cluster_bootstrap_interval(
            probability, target, episode, samples=400, seed=2
        )
        point = protocol_auroc(probability, target)
        self.assertLessEqual(low, point)
        self.assertGreaterEqual(high, point)

    def test_resample_keeps_episodes_intact(self) -> None:
        episode = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        members = cluster_members(episode)
        self.assertEqual(len(members), 3)
        index = resample_index(episode.numel(), members, torch.Generator().manual_seed(3))
        # A cluster bootstrap draws whole episodes, so every episode present in
        # the resample must appear with all of its members.
        self.assertEqual(index.numel(), episode.numel())
        picked = episode[index]
        for value in torch.unique(picked).tolist():
            self.assertEqual(int((picked == value).sum()) % 3, 0)

    def test_no_episode_key_falls_back_to_row_resampling(self) -> None:
        self.assertIsNone(cluster_members(None))
        index = resample_index(20, None, torch.Generator().manual_seed(4))
        self.assertEqual(index.numel(), 20)
        self.assertTrue(bool(((index >= 0) & (index < 20)).all()))


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
