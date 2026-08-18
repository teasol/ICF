"""`TrainingFreeClassifier` must reproduce the lineage path it replaces (docs SS140).

A rewrite of a scoring pipeline is only worth having if it is provably the same
pipeline. The lineage path is reachable with `ICF_COVARIANCE_BASIS=pca_within`,
`ICF_FIXED_HEAD=1` and `ICF_SKETCH_DIM=256`; if this file drifts from that by
even a little, every comparison against it silently stops meaning what it says.

⚠️ The equivalence tests pin the **v107** CT configuration explicitly
(`ct_readout="extreme"`, `ct_pca_dim=None`), because that is what the lineage
`_ct_features` computes. v108 changed the DEFAULTS (SS152), and the lineage has no
ridge/PCA CT to compare against -- so equivalence is a statement about a
configuration, not about the default. `DefaultTest` guards the default separately,
and the property tests below run on the default so they cover the live model.

They also pin K=8 rather than 256: what matters is that the two implementations
agree at whatever K they are given, and a small K keeps the test fast.

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
# The configuration the lineage `_ct_features` implements -- v108's defaults differ.
V107 = TrainingFreeConfig(
    sketch_dim=SKETCH, ct_readout="extreme", ct_pca_dim=None,
    ct_kmeans_iterations=0, cv_blocks="cov+mean", weight_ct=0.286, ct_num_tokens=16,
)


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

    reference = TrainingFreeClassifier(V107)
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
            mine = TrainingFreeClassifier(V107).margins(
                context_bags, labels, query_bags
            )
            theirs = lineage_margins(context_bags, labels, query_bags)
            self.assertTrue(
                torch.allclose(mine, theirs, atol=2e-3, rtol=2e-3),
                f"seed {seed}: {mine.tolist()} vs {theirs.tolist()}",
            )

    def test_ranking_matches_exactly(self):
        """AUROC only reads the ordering, so ordering is the contract that counts."""
        context_bags, labels, query_bags = episode(7, query=8)
        mine = TrainingFreeClassifier(V107).margins(
            context_bags, labels, query_bags
        )
        theirs = lineage_margins(context_bags, labels, query_bags)
        self.assertTrue(torch.equal(torch.argsort(mine), torch.argsort(theirs)))


class DefaultTest(unittest.TestCase):
    def test_defaults_are_the_promoted_values(self):
        """The default IS the baseline; if it drifts, `TrainingFreeClassifier()`
        silently stops being the active configuration. SS142 promoted K=128 -> 256;
        SS152 promoted CT to the ridge readout inside a 32-d PCA subspace; SS158
        added k-means tokens at weight 0.7 and off-diagonal-only CV; SS161 took the
        cluster count 16 -> 32 while KEEPING the 64-cell sample."""
        config = TrainingFreeConfig()
        self.assertEqual(config.sketch_dim, 256)
        self.assertEqual(config.ct_readout, "ridge")
        self.assertEqual(config.ct_pca_dim, 32)
        self.assertEqual(config.ct_kmeans_iterations, 30)
        self.assertEqual(config.ct_num_tokens, 32)
        self.assertEqual(config.ct_cells_per_bag, 64)
        self.assertEqual(config.ct_abundance_cells_per_bag, "match")
        self.assertEqual(config.ct_sampling, "even")
        self.assertEqual(config.ct_sampling_seed, 0)
        self.assertEqual(config.ct_distance_kernel, "broadcast")
        self.assertEqual(config.ct_tokenizer, "fps_lloyd")
        self.assertEqual(config.ct_bisect_iterations, 2)
        self.assertEqual(config.ct_bisect_power_iterations, 3)
        self.assertEqual(config.ct_tree_reduction, "segment")
        self.assertEqual(config.cv_blocks, "offdiag")
        self.assertEqual(config.weight_ct, 0.7)


class V109Test(unittest.TestCase):
    """SS158. CV must see only the off-diagonal entries while DD still gets the FULL
    triangle -- masking globally would break DD rather than narrow CV (SS156-1)."""

    def test_cv_blocks_changes_the_margin(self):
        context_bags, labels, query_bags = episode(20)
        wide = TrainingFreeClassifier(
            TrainingFreeConfig(sketch_dim=SKETCH, cv_blocks="cov+mean")
        ).margins(context_bags, labels, query_bags)
        narrow = TrainingFreeClassifier(
            TrainingFreeConfig(sketch_dim=SKETCH, cv_blocks="offdiag")
        ).margins(context_bags, labels, query_bags)
        self.assertFalse(torch.allclose(wide, narrow, atol=1e-4))

    def test_dd_is_unaffected_by_the_cv_mask(self):
        """Same episode, CV weight 0: the margin is then DD+CT only and must not
        move when cv_blocks changes."""
        context_bags, labels, query_bags = episode(21)
        margins = []
        for blocks in ("cov+mean", "offdiag"):
            model = TrainingFreeClassifier(TrainingFreeConfig(
                sketch_dim=SKETCH, cv_blocks=blocks, weight_cv=0.0
            ))
            margins.append(model.margins(context_bags, labels, query_bags))
        self.assertTrue(torch.allclose(margins[0], margins[1], atol=1e-6))

    def test_offdiag_descriptor_has_no_diagonal_and_no_mean(self):
        model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
        expected = SKETCH * (SKETCH + 1) // 2 - SKETCH
        triangle = torch.triu_indices(SKETCH, SKETCH)
        self.assertEqual(int((triangle[0] != triangle[1]).sum()), expected)
        del model

    def test_unknown_cv_blocks_is_rejected(self):
        context_bags, labels, query_bags = episode(22)
        with self.assertRaisesRegex(ValueError, "cv_blocks"):
            TrainingFreeClassifier(
                TrainingFreeConfig(sketch_dim=SKETCH, cv_blocks="nope")
            ).margins(context_bags, labels, query_bags)


class PropertyTest(unittest.TestCase):
    def test_no_parameters_exist(self):
        model = TrainingFreeClassifier()
        self.assertFalse(hasattr(model, "parameters"))
        self.assertFalse(any(isinstance(v, torch.nn.Parameter) for v in vars(model).values()))

    def test_deterministic(self):
        context_bags, labels, query_bags = episode(3)
        model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=SKETCH))
        first = model.margins(context_bags, labels, query_bags)  # v108 defaults
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
