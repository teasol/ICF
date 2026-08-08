"""Pin the per-fold context-representation caching contract (SS64).

`evaluate_trial`'s cached path builds one episode representation per fold and
slices it per query, instead of re-encoding every context bag once per query.
That is exact only because the aggregator's episode-level state is derived from
the CONTEXT bags alone:

  * `_context_pool_stats(bags, context_mask)` -- context cells only
  * `_context_anchors(bags, context_mask)`    -- context bags only
  * `_bag_view` / slot / tail statistics      -- per bag

This test pins those properties on a small synthetic episode, so a future change
that lets a query bag leak into the pool statistics or the anchors (which would
silently invalidate every cached official-fold number) fails here rather than in
a 50-fold run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline import StructuredEpisodePopulationAggregator  # noqa: E402


def _aggregator(input_dim: int = 32) -> StructuredEpisodePopulationAggregator:
    return StructuredEpisodePopulationAggregator(
        input_dim=input_dim,
        num_slots=6,
        num_density_slots=4,
        covariance_sketch_dim=8,
        context_samples_per_bag=8,
        bag_representation="poolz_l2",
    ).eval()


class TestContextCacheEquivalence(unittest.TestCase):
    """A query bag's representation must not depend on its co-queries."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.input_dim = 32
        self.aggregator = _aggregator(self.input_dim)
        self.n_context = 5
        self.bags = [
            torch.randn(40 + 7 * i, self.input_dim) for i in range(self.n_context + 3)
        ]

    def _mask(self, n_bags: int) -> torch.Tensor:
        mask = torch.zeros(n_bags, dtype=torch.bool)
        mask[: self.n_context] = True
        return mask

    def test_joint_pass_matches_per_query_pass(self) -> None:
        """One pass over context+all queries == one pass per query."""
        with torch.no_grad():
            joint = self.aggregator(self.bags, context_mask=self._mask(len(self.bags)))
            for offset in range(len(self.bags) - self.n_context):
                position = self.n_context + offset
                subset = self.bags[: self.n_context] + [self.bags[position]]
                single = self.aggregator(subset, context_mask=self._mask(len(subset)))
                for name, tokens in joint.items():
                    with self.subTest(query=offset, key=name):
                        delta = (
                            tokens[position].float() - single[name][-1].float()
                        ).abs().max()
                        self.assertEqual(
                            float(delta),
                            0.0,
                            f"{name} for query {offset} changed with co-queries "
                            f"(||delta||inf={float(delta):.3e}); the cached "
                            "eval path would no longer be exact.",
                        )

    def test_context_representation_is_query_independent(self) -> None:
        """Context tokens must be identical whichever query shares the pass."""
        with torch.no_grad():
            first = self.aggregator(
                self.bags[: self.n_context + 1],
                context_mask=self._mask(self.n_context + 1),
            )
            second = self.aggregator(
                self.bags[: self.n_context] + [self.bags[-1]],
                context_mask=self._mask(self.n_context + 1),
            )
        for name, tokens in first.items():
            with self.subTest(key=name):
                delta = (
                    tokens[: self.n_context].float()
                    - second[name][: self.n_context].float()
                ).abs().max()
                self.assertEqual(float(delta), 0.0)

    def test_pool_stats_ignore_query_cells(self) -> None:
        """Replacing the query bag must not move the context pool statistics."""
        bags_a = self.bags[: self.n_context] + [self.bags[-1]]
        bags_b = self.bags[: self.n_context] + [self.bags[-1] * 100.0 + 5.0]
        mask = self._mask(self.n_context + 1)
        with torch.no_grad():
            mean_a, std_a = self.aggregator._context_pool_stats(
                self.aggregator._normalize_bags(bags_a), mask
            )
            mean_b, std_b = self.aggregator._context_pool_stats(
                self.aggregator._normalize_bags(bags_b), mask
            )
        self.assertEqual(float((mean_a - mean_b).abs().max()), 0.0)
        self.assertEqual(float((std_a - std_b).abs().max()), 0.0)

    def test_anchors_ignore_query_cells(self) -> None:
        """Anchors are selected from context candidates only."""
        bags_a = self.bags[: self.n_context] + [self.bags[-1]]
        bags_b = self.bags[: self.n_context] + [self.bags[-1] * 100.0 + 5.0]
        mask = self._mask(self.n_context + 1)
        with torch.no_grad():
            anchors_a = self.aggregator._context_anchors(
                self.aggregator._normalize_bags(bags_a), mask
            )
            anchors_b = self.aggregator._context_anchors(
                self.aggregator._normalize_bags(bags_b), mask
            )
        self.assertEqual(float((anchors_a - anchors_b).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
