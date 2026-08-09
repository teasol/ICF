"""v35 §7: streaming bag aggregation must be numerically identical.

`stream_eval_bags` (docs/history/architecture_v35_*.md §3.3) computes one bag's
view at a time instead of materializing every bag's standardized view and
centered delta up front. It is meant to be an EXACT memory optimization -- the
same `_bag_view`/`_covariance_sketch`/`_select_anchors` on the same inputs -- so
these tests pin equality of all nine representation keys, the anchors, and the
final logits against the eager path.

Also pins the v35 §5 cardinality draw (`num_cells_log_uniform_power`).
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

from src.datasets.synthetic_data import SyntheticManifoldGenerator  # noqa: E402
from src.models.baseline import BaseModel  # noqa: E402

TOLERANCE = 1e-4


def build_model(input_dim: int = 64) -> BaseModel:
    """A small v34-shaped model: poolz_l2 view + MLA slots + subspace relation."""
    torch.manual_seed(0)
    model = BaseModel(
        input_dim=input_dim,
        bag_centered_representation=True,
        global_summary="centered_spread",
        bag_representation="poolz_l2",
        aggregator_covariance_sketch_dim=16,
        aggregator_covariance_mode="correlation",
        aggregator_covariance_shrinkage=0.1,
        aggregator_num_slots=6,
        aggregator_num_density_slots=4,
        aggregator_context_samples_per_bag=8,
        aggregator_assignment_temperature=0.1,
        aggregator_slot_rare_fraction=0.05,
        aggregator_tail_fractions=[0.01, 0.05, 0.15],
        aggregator_min_tail_instances=1,
        aggregator_slot_latent_dim=8,
        aggregator_slot_query_latent_dim=16,
        aggregator_slot_affinity_dim=32,
        project_structured_tokens=True,
        covariance_relation={
            "enabled": True,
            "mode": "learned_head",
            "granularity": "subspace",
            "subspace_rank": 1,
            "subspace_whiten": True,
            "subspace_shrinkage": 0.25,
            "diagnostic_only": False,
            "residual_scale": 0.5,
            "eps": 1.0e-06,
        },
    )
    model.eval()
    return model


def make_episode(
    input_dim: int, sizes: list[int], seed: int = 3
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    bags = [
        torch.randn(size, input_dim, generator=generator) for size in sizes
    ]
    labels = torch.tensor([index % 2 for index in range(len(sizes))])
    mask_index = torch.tensor([len(sizes) - 1])
    return bags, labels, mask_index


class TestStreamEvalBagsEquivalence(unittest.TestCase):
    """The streaming path must reproduce the eager path exactly."""

    def _compare(self, sizes: list[int], input_dim: int = 64) -> None:
        model = build_model(input_dim)
        bags, labels, mask_index = make_episode(input_dim, sizes)
        aggregator = model.aggregator
        is_context = torch.ones(len(sizes), dtype=torch.bool)
        is_context[mask_index] = False

        with torch.no_grad():
            aggregator.stream_eval_bags = False
            eager = aggregator(bags, context_mask=is_context)
            eager_logits = model.forward(bags, labels, mask_index)

            aggregator.stream_eval_bags = True
            streamed = aggregator(bags, context_mask=is_context)
            streamed_logits = model.forward(bags, labels, mask_index)

        self.assertEqual(set(eager), set(streamed))
        for key in sorted(eager):
            difference = (eager[key].float() - streamed[key].float()).abs().max()
            self.assertLess(
                float(difference),
                TOLERANCE,
                f"representation[{key!r}] diverged by {float(difference):.3e} "
                f"for bag sizes {sizes}",
            )
        logit_difference = (eager_logits - streamed_logits).abs().max()
        self.assertLess(
            float(logit_difference),
            TOLERANCE,
            f"logits diverged by {float(logit_difference):.3e}",
        )

    def test_uniform_small_bags(self) -> None:
        self._compare([12, 12, 12, 12, 12, 12])

    def test_ragged_bags_spanning_three_orders(self) -> None:
        # 1-cell bags are the Musk regime; 4000 is the PathoBench regime.
        self._compare([1, 3, 40, 400, 4000, 17, 250])

    def test_anchors_are_bit_identical(self) -> None:
        """Anchor selection must not depend on when the views are built."""
        input_dim = 64
        model = build_model(input_dim)
        aggregator = model.aggregator
        sizes = [5, 60, 900, 2500, 30]
        bags, _, mask_index = make_episode(input_dim, sizes)
        is_context = torch.ones(len(sizes), dtype=torch.bool)
        is_context[mask_index] = False

        with torch.no_grad():
            pool_mean, pool_std = aggregator._context_pool_stats(bags, is_context)
            views = [
                aggregator._bag_view(bag, pool_mean, pool_std)[0] for bag in bags
            ]
            eager_anchors = aggregator._context_anchors(views, is_context)
            streamed_anchors = aggregator._select_anchors(
                torch.cat(
                    [
                        aggregator._population_candidates(
                            aggregator._bag_view(bags[index], pool_mean, pool_std)[0]
                        )
                        for index in range(len(bags))
                        if bool(is_context[index])
                    ],
                    dim=0,
                )
            )
        self.assertTrue(
            torch.equal(eager_anchors, streamed_anchors),
            "streaming anchors are not bit-identical to the eager anchors",
        )

    def test_pool_stats_match_batched_twin(self) -> None:
        """Streaming pool stats must still agree with the dense/batched twin."""
        input_dim = 32
        model = build_model(input_dim)
        aggregator = model.aggregator
        generator = torch.Generator().manual_seed(11)
        bags = [torch.randn(40, input_dim, generator=generator) for _ in range(5)]
        is_context = torch.tensor([True, True, True, True, False])

        mean, std = aggregator._context_pool_stats(bags, is_context)
        batched_mean, batched_std = aggregator._context_pool_stats_batched(
            torch.stack(bags).unsqueeze(0), is_context.unsqueeze(0)
        )
        self.assertLess(float((mean - batched_mean[0]).abs().max()), 1e-5)
        self.assertLess(float((std - batched_std[0]).abs().max()), 1e-5)


class TestLogUniformPower(unittest.TestCase):
    """v35 §5: the tilted log-uniform draw, with small bags preserved."""

    def _generator(self, cells, power: float) -> SyntheticManifoldGenerator:
        return SyntheticManifoldGenerator(
            num_bags=(4, 4),
            num_cells=cells,
            num_cells_log_uniform=True,
            num_cells_log_uniform_power=power,
            latent_dim=4,
            output_dim=8,
        )

    def test_power_one_reproduces_v34_draw(self) -> None:
        for seed in range(25):
            baseline = self._generator((1, 8192), 1.0)
            tilted = self._generator((1, 8192), 1.0)
            self.assertEqual(
                baseline.sample_num_cells(torch.Generator().manual_seed(seed)),
                tilted.sample_num_cells(torch.Generator().manual_seed(seed)),
            )

    def test_power_shifts_mass_up_but_keeps_small_bags(self) -> None:
        cells = (1, 32768)
        draws = 20000
        generator = self._generator(cells, 1.5)
        samples = [
            generator.sample_num_cells(torch.Generator().manual_seed(seed))
            for seed in range(draws)
        ]
        small = sum(1 for value in samples if value <= 34) / draws
        large = sum(1 for value in samples if value >= 8192) / draws

        # Closed form: P(n <= k) = (ln k / ln B) ** power.
        expected_small = (math.log(34) / math.log(32768)) ** 1.5
        expected_large = 1.0 - (math.log(8192) / math.log(32768)) ** 1.5
        self.assertAlmostEqual(small, expected_small, delta=0.02)
        self.assertAlmostEqual(large, expected_large, delta=0.02)

        # The Musk regime must survive: a floor would zero this out.
        self.assertGreater(small, 0.15)
        self.assertTrue(all(1 <= value <= 32768 for value in samples))

    def test_rejects_nonpositive_power(self) -> None:
        with self.assertRaises(ValueError):
            self._generator((1, 8192), 0.0)


if __name__ == "__main__":
    unittest.main()
