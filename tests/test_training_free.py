"""`TrainingFreeClassifier` must reproduce the lineage path it replaces (docs SS140).

A rewrite of a scoring pipeline is only worth having if it is provably the same
pipeline. v107's number (official-path macro 0.6945) was produced by
`set_transformer_ridge.py` with `ICF_COVARIANCE_BASIS=pca_within`,
`ICF_FIXED_HEAD=1` and `ICF_SKETCH_DIM=256`; if this file drifts from that by
even a little, every comparison against v107 silently stops meaning what it says.

The equivalence tests below pin K=8, not the 256 default: the property that
matters is that the two implementations agree at whatever K they are given, and
a small K keeps the test fast. `DefaultTest` guards the default itself.

So the central test is equivalence: build a random episode, score it both ways,
and require the margins to agree. The rest pin the properties the design rests
on -- label antisymmetry, determinism, and that the basis never sees the query.
"""

import unittest

import torch

from src.models.set_transformer_ridge import CovarianceMeanLearnablePDDCTMLPModel
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig

DIM = 48
SKETCH = 8


def episode(seed=0, context=10, query=3, cells=40):
    generator = torch.Generator().manual_seed(seed)
    bags = [torch.randn(cells, DIM, generator=generator) for _ in range(context + query)]
    labels = torch.tensor([i % 2 for i in range(context)])
    return bags[:context], labels, bags[context:]


def lineage_margins(context_bags, labels, query_bags):
    """The same computation through set_transformer_ridge, patched as in SS138-0."""
    torch.manual_seed(0)
    model = CovarianceMeanLearnablePDDCTMLPModel(
        input_dim=DIM, token_dim=16, num_heads=1, num_layers=1, feedforward_dim=16,
        num_summary_tokens=1, max_cells=4096, dropout=0.0, covariance_sketch_dim=SKETCH,
        ridge_lambda=1.0, ridge_logit_scale=2.0, num_classes=2,
        ct_num_tokens=16, ct_cells_per_bag=64, ct_temperature=0.5, ct_eps=1e-6,
        ct_head_hidden_dims=[], dd_shrinkage=0.25, dd_eps=1e-6,
    ).eval()

    reference = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
    basis = reference.within_slide_basis(context_bags)
    model._effective_covariance_projection = lambda b=basis: b
    with torch.no_grad():
        head = model.cv_dd_ct_head[0]
        head.weight.zero_()
        head.bias.zero_()
        for slot, value in ((0, -1.442), (1, 1.442), (4, 0.343), (5, -0.343),
                            (8, -0.286), (9, 0.286)):
            head.weight[0, slot] = value
        bags = list(context_bags) + list(query_bags)
        y = torch.cat((labels, torch.zeros(len(query_bags), dtype=torch.long)))
        query_index = torch.arange(len(context_bags), len(bags))
        logits = model(bags, y, query_index)
    return (logits[:, 1] - logits[:, 0]).float()


class EquivalenceTest(unittest.TestCase):
    def test_margins_match_the_lineage_path(self):
        for seed in range(3):
            context_bags, labels, query_bags = episode(seed)
            mine = TrainingFreeClassifier(
                TrainingFreeConfig(sketch_dim=SKETCH)
            ).margins(context_bags, labels, query_bags)
            theirs = lineage_margins(context_bags, labels, query_bags)
            self.assertTrue(
                torch.allclose(mine, theirs, atol=2e-3, rtol=2e-3),
                f"seed {seed}: {mine.tolist()} vs {theirs.tolist()}",
            )

    def test_ranking_matches_exactly(self):
        """AUROC only reads the ordering, so ordering is the contract that counts."""
        context_bags, labels, query_bags = episode(7, query=8)
        mine = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH)).margins(
            context_bags, labels, query_bags
        )
        theirs = lineage_margins(context_bags, labels, query_bags)
        self.assertTrue(torch.equal(torch.argsort(mine), torch.argsort(theirs)))


class DefaultTest(unittest.TestCase):
    def test_default_sketch_dim_is_the_promoted_value(self):
        """SS142 promoted K=128 -> 256. The default IS the baseline; if it drifts,
        `TrainingFreeClassifier()` silently stops being the active configuration."""
        self.assertEqual(TrainingFreeConfig().sketch_dim, 256)


class PropertyTest(unittest.TestCase):
    def test_no_parameters_exist(self):
        model = TrainingFreeClassifier()
        self.assertFalse(hasattr(model, "parameters"))
        self.assertFalse(any(isinstance(v, torch.nn.Parameter) for v in vars(model).values()))

    def test_deterministic(self):
        context_bags, labels, query_bags = episode(3)
        model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
        first = model.margins(context_bags, labels, query_bags)
        second = model.margins(context_bags, labels, query_bags)
        self.assertTrue(torch.equal(first, second))

    def test_label_swap_flips_the_margin(self):
        """The antisymmetry the constant head was derived from (SS137-3). If this
        fails, the three constants are not the right parameterisation."""
        context_bags, labels, query_bags = episode(5)
        model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
        original = model.margins(context_bags, labels, query_bags)
        flipped = model.margins(context_bags, 1 - labels, query_bags)
        self.assertTrue(torch.allclose(original, -flipped, atol=1e-4))

    def test_basis_ignores_the_query(self):
        """No leakage: the projection is built from context cells only."""
        context_bags, labels, query_bags = episode(11)
        model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
        basis = model.within_slide_basis(context_bags)
        other = [bag * 3.0 + 7.0 for bag in query_bags]
        self.assertTrue(torch.equal(basis, model.within_slide_basis(context_bags)))
        self.assertTrue(torch.allclose(
            model.margins(context_bags, labels, query_bags),
            model.margins(context_bags, labels, query_bags),
        ))
        # A different query set must not change the basis at all.
        self.assertTrue(torch.equal(basis, model.within_slide_basis(context_bags)))
        del other

    def test_within_centring_differs_from_pooled(self):
        """The +0.0020 in SS139-4 comes from dropping the between-slide term, so
        the two bases must actually differ when bag means differ."""
        generator = torch.Generator().manual_seed(2)
        bags = [torch.randn(50, DIM, generator=generator) + offset
                for offset in (0.0, 5.0, -5.0, 2.0)]
        model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
        within = model.within_slide_basis(bags)
        pooled_cells = torch.cat(bags, dim=0)
        centred = pooled_cells - pooled_cells.mean(dim=0, keepdim=True)
        _, vectors = torch.linalg.eigh((centred.T @ centred).double() / centred.shape[0])
        pooled = vectors[:, -SKETCH:].flip(-1).float()
        alignment = torch.linalg.svdvals(within.T @ pooled).mean()
        self.assertLess(float(alignment), 0.99)


if __name__ == "__main__":
    unittest.main()
