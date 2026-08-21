"""Contract and property tests for the CT readout branch (v108-v114 active).

Historical variants and rejected sweeps (HDBSCAN, DBSCAN, Kernel-Ridge,
Top-k pooling, KMeans Lloyd) have been preserved in
`tests/history/legacy_ct_readout.py`.
"""

from __future__ import annotations

import unittest
import torch

from src.models.ct_readout import (
    CTReadoutConfig,
    ct_abundance,
    ct_margins,
    hierarchical_2means_tokens,
    parse_cell_budget,
    resolve_cells_per_bag,
    sample_cells,
)

DIM = 32
CONFIG = CTReadoutConfig(
    num_tokens=16,
    tokenizer="hierarchical_2means",
    pca_dim=16,
)


def _make_episode(seed=0, context=6, query=2, cells=30):
    torch.manual_seed(seed)
    context_bags = [torch.randn(cells, DIM) for _ in range(context)]
    labels = torch.tensor([0, 1] * (context // 2))
    query_bags = [torch.randn(cells, DIM) for _ in range(query)]
    return context_bags, labels, query_bags


class ShapeAndDeterminismTest(unittest.TestCase):
    def test_shapes(self):
        context_bags, labels, query_bags = _make_episode(3, context=16, query=6)
        for mode in ("extreme", "prototype", "ridge"):
            margins, abundance = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            self.assertEqual(margins.context.shape, (16,), mode)
            self.assertEqual(margins.query.shape, (6,), mode)
            self.assertEqual(margins.separation.shape, (), mode)
            self.assertEqual(abundance.context.shape, (16, CONFIG.num_tokens), mode)
            self.assertEqual(abundance.query.shape, (6, CONFIG.num_tokens), mode)

    def test_deterministic(self):
        context_bags, labels, query_bags = _make_episode(4)
        for mode in ("extreme", "prototype", "ridge"):
            first, _ = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            second, _ = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            self.assertTrue(torch.equal(first.query, second.query), mode)

    def test_abundance_rows_are_simplex(self):
        context_bags, _, query_bags = _make_episode(5)
        abundance = ct_abundance(context_bags, query_bags, CONFIG)
        for side in (abundance.context, abundance.query):
            self.assertTrue(torch.allclose(side.sum(dim=-1), torch.ones(side.shape[0]), atol=1e-5))
            self.assertTrue(bool((side >= 0).all()))


class AntisymmetryTest(unittest.TestCase):
    def test_class_swap_negates_every_readout(self):
        context_bags, labels, query_bags = _make_episode(8)
        for mode in ("extreme", "prototype", "ridge"):
            a, _ = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            flipped, _ = ct_margins(context_bags, 1 - labels, query_bags, CONFIG, mode)
            self.assertTrue(torch.allclose(a.query, -flipped.query, atol=1e-4), mode)


class HierarchicalTwoMeansTest(unittest.TestCase):
    def test_tree_generates_exact_power_of_two_tokens(self):
        torch.manual_seed(42)
        pooled = torch.randn(100, 32)
        config = CTReadoutConfig(num_tokens=16)
        tokens = hierarchical_2means_tokens(pooled, config)
        self.assertEqual(tokens.shape, (16, 32))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_deterministic_token_generation(self):
        torch.manual_seed(42)
        pooled = torch.randn(100, 32)
        config = CTReadoutConfig(num_tokens=8)
        tokens1 = hierarchical_2means_tokens(pooled, config)
        tokens2 = hierarchical_2means_tokens(pooled, config)
        torch.testing.assert_close(tokens1, tokens2)


class SizeAwareSamplingTest(unittest.TestCase):
    def test_parse_cell_budget(self):
        self.assertEqual(parse_cell_budget("all"), (None, None, None))
        self.assertEqual(parse_cell_budget("64"), (64, None, None))
        self.assertEqual(parse_cell_budget("0.125"), (None, 0.125, None))
        self.assertEqual(parse_cell_budget("own:0.1"), (None, 0.1, "own"))
        self.assertEqual(parse_cell_budget("median:0.25"), (None, 0.25, "median"))

    def test_larger_bags_draw_more_cells_under_the_same_fraction(self):
        small = torch.arange(80 * DIM, dtype=torch.float32).reshape(80, DIM)
        large = torch.arange(400 * DIM, dtype=torch.float32).reshape(400, DIM)
        config = CTReadoutConfig(
            cells_per_bag=None, cells_fraction=0.25, sampling="random", sampling_seed=4,
        )
        self.assertEqual(sample_cells(small, config).shape[0], 20)
        self.assertEqual(sample_cells(large, config).shape[0], 100)

    def test_resolve_cells_per_bag_with_fraction(self):
        config = CTReadoutConfig(cells_fraction=0.1, cells_min=10, cells_scale="own")
        self.assertEqual(resolve_cells_per_bag(100, config), 10)
        self.assertEqual(resolve_cells_per_bag(400, config), 40)
        self.assertEqual(resolve_cells_per_bag(50, config), 10)  # floor at 10


if __name__ == "__main__":
    unittest.main()
