"""MLA-style low-rank slot affinity (config-gated aggregator probe).

`StructuredEpisodePopulationAggregator(slot_latent_dim=...)` computes the
cell-to-slot affinity in a low-rank latent space (MLA-style absorbed query)
instead of the full-dim dot product. Disabled (None) must be byte-identical to
the legacy full-dim affinity and add no parameters, so existing checkpoints
load unchanged.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import StructuredEpisodePopulationAggregator  # noqa: E402


def _make_aggregator(**overrides) -> StructuredEpisodePopulationAggregator:
    kwargs = dict(
        input_dim=64,
        num_slots=4,
        num_density_slots=2,
        tail_fractions=(0.05,),
        absolute_tail_ks=(),
    )
    kwargs.update(overrides)
    return StructuredEpisodePopulationAggregator(**kwargs)


class TestSlotMla(unittest.TestCase):
    def test_disabled_adds_no_latent_params(self) -> None:
        agg = _make_aggregator()
        self.assertIsNone(agg.slot_latent_dim)
        self.assertFalse(hasattr(agg, "slot_w_dq"))
        self.assertFalse(hasattr(agg, "slot_w_dkv"))

    def test_disabled_affinity_equals_full_dim_dot(self) -> None:
        agg = _make_aggregator()
        torch.manual_seed(0)
        cells = torch.randn(3, 20, 64)
        anchors = torch.randn(3, 4, 64)
        got = agg._slot_similarity(cells, anchors)
        expected = torch.einsum("bnd,bsd->bns", cells, anchors)
        torch.testing.assert_close(got, expected)
        # single-bag (list-path) shape
        got_single = agg._slot_similarity(torch.randn(20, 64), torch.randn(4, 64))
        self.assertEqual(tuple(got_single.shape), (20, 4))

    def test_enabled_creates_params_and_returns_correct_shapes(self) -> None:
        agg = _make_aggregator(slot_latent_dim=8)
        self.assertEqual(agg.slot_latent_dim, 8)
        self.assertEqual(agg.slot_query_latent_dim, 8)
        self.assertEqual(agg.slot_affinity_dim, 8)
        self.assertTrue(hasattr(agg, "slot_w_dq"))
        self.assertTrue(hasattr(agg, "slot_w_uk"))
        torch.manual_seed(0)
        cells = torch.randn(3, 20, 64)
        anchors = torch.randn(3, 4, 64)
        sim = agg._slot_similarity(cells, anchors)
        self.assertEqual(tuple(sim.shape), (3, 20, 4))
        self.assertTrue(torch.isfinite(sim).all())
        sim_single = agg._slot_similarity(torch.randn(20, 64), torch.randn(4, 64))
        self.assertEqual(tuple(sim_single.shape), (20, 4))

    def test_enabled_forward_is_differentiable(self) -> None:
        agg = _make_aggregator(slot_latent_dim=8)
        cells = torch.randn(2, 20, 64, requires_grad=True)
        anchors = torch.randn(2, 4, 64)
        sim = agg._slot_similarity(cells, anchors)
        loss = sim.sum()
        loss.backward()
        self.assertIsNotNone(cells.grad)
        self.assertTrue(torch.isfinite(cells.grad).all())


if __name__ == "__main__":
    unittest.main()
