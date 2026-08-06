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

    def test_variance_trick_matches_direct_formula(self) -> None:
        # The low-rank slot path computes slot_std / slot_distance with the
        # exact Var = E[X^2] - E[X]^2 reformulation (no [cells, slots, dim]
        # difference tensor). It must match the direct formula to float32.
        torch.manual_seed(0)
        b, cells, slots, dim = 3, 50, 4, 64
        assignment = torch.rand(b, cells, slots).softmax(dim=-1)
        instances = torch.nn.functional.normalize(torch.randn(b, cells, dim), dim=-1)
        mass = assignment.sum(dim=1).clamp_min(1e-6)
        slot_mean = torch.einsum("bns,bnd->bsd", assignment, instances) / mass.unsqueeze(-1)

        difference = instances[:, :, None, :] - slot_mean[:, None, :, :]
        std_direct = torch.sqrt(
            (
                assignment.float().transpose(1, 2).unsqueeze(-1)
                * difference.float().square().transpose(1, 2)
            ).sum(dim=2)
            / mass.float().unsqueeze(-1)
            + 1e-6
        )
        dist_direct = difference.float().square().mean(dim=-1)

        x_sq = instances.float().square()
        second_moment = (
            torch.einsum("bns,bnd->bsd", assignment.float(), x_sq)
            / mass.float().unsqueeze(-1)
        )
        std_trick = torch.sqrt(
            (second_moment - slot_mean.float().square()).clamp_min(0.0) + 1e-6
        )
        x_mean_sq = instances.float().square().mean(dim=-1)
        m_mean_sq = slot_mean.float().square().mean(dim=-1)
        x_dot_m = torch.einsum("bnd,bsd->bns", instances.float(), slot_mean.float()) / dim
        dist_trick = x_mean_sq.unsqueeze(-1) - 2.0 * x_dot_m + m_mean_sq.unsqueeze(-2)

        torch.testing.assert_close(std_trick, std_direct, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(dist_trick, dist_direct, atol=1e-5, rtol=1e-4)

    def test_population_candidates_batched_matches_loop(self) -> None:
        # Batched anchor-candidate pooling must equal the per-bag loop (the
        # batched path when all bags >= context_samples_per_bag cells, and the
        # fallback path when a bag is smaller).
        normalize = torch.nn.functional.normalize
        agg = _make_aggregator(context_samples_per_bag=32)
        torch.manual_seed(0)
        bags = [normalize(torch.randn(n, 64)) for n in (40, 50, 33)]
        batched = agg._population_candidates_batched(bags)
        loop = torch.cat([agg._population_candidates(b) for b in bags], dim=0)
        self.assertEqual(tuple(batched.shape), tuple(loop.shape))
        torch.testing.assert_close(batched, loop, atol=1e-5, rtol=1e-4)
        # fallback path with a small bag (< context_samples_per_bag cells)
        bags2 = [normalize(torch.randn(n, 64)) for n in (40, 10, 33)]
        b2 = agg._population_candidates_batched(bags2)
        l2 = torch.cat([agg._population_candidates(b) for b in bags2], dim=0)
        self.assertEqual(tuple(b2.shape), tuple(l2.shape))
        torch.testing.assert_close(b2, l2)


if __name__ == "__main__":
    unittest.main()
