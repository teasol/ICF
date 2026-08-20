"""CT readouts (docs SS148): the baseline must not move, and the alternatives
must be honest.

`ct_readout` factors steps 1-5 of CT out of two places (`training_free.py` and
the lineage) so the readout experiment varies ONLY step 6-7. That refactor is the
risk: if `extreme` drifts from the lineage by any amount, every "prototype gains
X" number silently mixes a readout change with a representation change. So the
first test compares against `_ct_features` directly.

The rest pin what the alternatives must satisfy to be usable at all: no query
statistics anywhere, determinism, and label antisymmetry -- the property the fixed
head's three constants are derived from (SS137-3).
"""

import unittest

import torch

from src.models.ct_readout import (
    CTReadoutConfig,
    calibrate,
    ct_abundance,
    ct_margins,
    discriminative_score,
    readout_extreme,
    farthest_point_tokens,
    hierarchical_2means_tokens,
    hdbscan_tokens,
    dbscan_tokens,
    kmeans_plusplus_tokens,
    lloyd_refine,
    parse_cell_budget,
    prepare_cells,
    readout_prototype,
    readout_ridge,
    readout_kernel_ridge,
    resolve_cells_per_bag,
    ridge_coefficients,
    sample_cells,
    typical_bag_size,
)
from src.models.set_transformer_ridge import CovarianceMeanLearnablePDDCTMLPModel

DIM = 40
# Historical lineage equivalence intentionally keeps the old even sampler. The
# active CTReadoutConfig default is tested separately below.
CONFIG = CTReadoutConfig(
    num_tokens=8, cells_per_bag=24, temperature=0.5, eps=1e-6, sampling="even"
)


def episode(seed=0, context=16, query=6, cells=30):
    generator = torch.Generator().manual_seed(seed)
    bags = [torch.randn(cells, DIM, generator=generator) for _ in range(context + query)]
    # A real class signal, so the alternatives are not being asked to fit noise.
    labels = torch.tensor([i % 2 for i in range(context)])
    for index, label in enumerate(labels.tolist()):
        if label == 1:
            bags[index] = bags[index] + 0.4
    return bags[:context], labels, bags[context:]


def lineage_model():
    torch.manual_seed(0)
    return CovarianceMeanLearnablePDDCTMLPModel(
        input_dim=DIM, token_dim=16, num_heads=1, num_layers=1, feedforward_dim=16,
        num_summary_tokens=1, max_cells=4096, dropout=0.0, covariance_sketch_dim=8,
        ridge_lambda=1.0, ridge_logit_scale=2.0, num_classes=2,
        ct_num_tokens=CONFIG.num_tokens, ct_cells_per_bag=CONFIG.cells_per_bag,
        ct_temperature=CONFIG.temperature, ct_eps=CONFIG.eps,
        ct_head_hidden_dims=[], dd_shrinkage=0.25, dd_eps=1e-6,
    ).eval()


class BaselineUnchangedTest(unittest.TestCase):
    def test_active_default_uses_random_sampling(self):
        self.assertEqual(CTReadoutConfig().sampling, "random")

    def test_extreme_matches_the_lineage_ct_features(self):
        """The refactor must not move v107 by a float."""
        model = lineage_model()
        for seed in range(4):
            context_bags, labels, query_bags = episode(seed)
            with torch.no_grad():
                q0, q1, separation = model._ct_features(context_bags, labels, query_bags)
            margins, _ = ct_margins(context_bags, labels, query_bags, CONFIG, "extreme")
            self.assertTrue(
                torch.allclose(margins.query, q1 - q0, atol=1e-6, rtol=1e-6),
                f"seed {seed}: {margins.query.tolist()} vs {(q1 - q0).tolist()}",
            )
            self.assertTrue(torch.allclose(margins.separation, separation, atol=1e-6))

    def test_extreme_is_never_calibrated(self):
        """`extreme` IS the reference, so `calibrated` must not touch it."""
        context_bags, labels, query_bags = episode(1)
        on, _ = ct_margins(context_bags, labels, query_bags, CONFIG, "extreme", True)
        off, _ = ct_margins(context_bags, labels, query_bags, CONFIG, "extreme", False)
        self.assertTrue(torch.equal(on.query, off.query))

    def test_score_matches_the_lineage_token_ranking(self):
        model = lineage_model()
        context_bags, labels, query_bags = episode(2)
        abundance = ct_abundance(context_bags, query_bags, CONFIG)
        score = discriminative_score(abundance, labels, CONFIG)
        with torch.no_grad():
            q0, q1, _ = model._ct_features(context_bags, labels, query_bags)
        _, standardised_query = None, None
        # Indirect check: the two selected coordinates must reproduce q0/q1.
        _, query = (abundance.context, abundance.query)
        centre = abundance.context.mean(dim=0)
        spread = (abundance.context - centre).square().mean(dim=0).sqrt().clamp_min(CONFIG.eps)
        standardised = (query - centre) / spread
        self.assertTrue(torch.allclose(standardised[:, score.argmax()], q0, atol=1e-6))
        self.assertTrue(torch.allclose(standardised[:, score.argmin()], q1, atol=1e-6))
        del standardised_query


class ShapeAndDeterminismTest(unittest.TestCase):
    def test_shapes(self):
        context_bags, labels, query_bags = episode(3, context=16, query=6)
        for mode in ("extreme", "prototype", "ridge"):
            margins, abundance = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            self.assertEqual(margins.context.shape, (16,), mode)
            self.assertEqual(margins.query.shape, (6,), mode)
            self.assertEqual(margins.separation.shape, (), mode)
            self.assertEqual(abundance.context.shape, (16, CONFIG.num_tokens), mode)
            self.assertEqual(abundance.query.shape, (6, CONFIG.num_tokens), mode)

    def test_deterministic(self):
        context_bags, labels, query_bags = episode(4)
        for mode in ("extreme", "prototype", "ridge"):
            first, _ = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            second, _ = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            self.assertTrue(torch.equal(first.query, second.query), mode)

    def test_abundance_rows_are_simplex(self):
        context_bags, _, query_bags = episode(5)
        abundance = ct_abundance(context_bags, query_bags, CONFIG)
        for side in (abundance.context, abundance.query):
            self.assertTrue(torch.allclose(side.sum(dim=-1), torch.ones(side.shape[0]), atol=1e-5))
            self.assertTrue(bool((side >= 0).all()))


class ContextOnlyTest(unittest.TestCase):
    def test_query_bags_never_change_the_context_side(self):
        """No leakage: standardisation, tokens, prototypes and ridge are context-only."""
        context_bags, labels, query_bags = episode(6)
        other = [bag * 5.0 + 11.0 for bag in query_bags]
        for mode in ("extreme", "prototype", "ridge"):
            a, first = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            b, second = ct_margins(context_bags, labels, other, CONFIG, mode)
            self.assertTrue(torch.equal(first.tokens, second.tokens), mode)
            self.assertTrue(torch.equal(first.context, second.context), mode)
            self.assertTrue(torch.equal(a.context, b.context), mode)

    def test_adding_query_bags_does_not_move_existing_query_scores(self):
        context_bags, labels, query_bags = episode(7, query=4)
        few, _ = ct_margins(context_bags, labels, query_bags, CONFIG, "ridge")
        more, _ = ct_margins(
            context_bags, labels, list(query_bags) + [query_bags[0] * 3.0],
            CONFIG, "ridge",
        )
        self.assertTrue(torch.allclose(few.query, more.query[:4], atol=1e-6))


class AntisymmetryTest(unittest.TestCase):
    """SS137-3: the fixed head's constants exist because a class swap negates the
    margin. A readout that breaks this cannot be dropped into that head."""

    def test_class_swap_negates_every_readout(self):
        context_bags, labels, query_bags = episode(8, context=20)
        for mode in ("extreme", "prototype", "ridge"):
            original, _ = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            flipped, _ = ct_margins(context_bags, 1 - labels, query_bags, CONFIG, mode)
            self.assertTrue(
                torch.allclose(original.query, -flipped.query, atol=1e-4),
                f"{mode}: {original.query.tolist()} vs {(-flipped.query).tolist()}",
            )

    def test_calibration_preserves_antisymmetry(self):
        context_bags, labels, query_bags = episode(9, context=20)
        for mode in ("prototype", "ridge"):
            original, _ = ct_margins(context_bags, labels, query_bags, CONFIG, mode, True)
            flipped, _ = ct_margins(context_bags, 1 - labels, query_bags, CONFIG, mode, True)
            self.assertTrue(
                torch.allclose(original.query, -flipped.query, atol=1e-4), mode
            )


class CalibrationTest(unittest.TestCase):
    def test_context_rms_and_mean_match_the_reference(self):
        context_bags, labels, query_bags = episode(10, context=20)
        abundance = ct_abundance(context_bags, query_bags, CONFIG)
        reference = readout_extreme(abundance, labels, CONFIG)
        for readout in (readout_prototype, readout_ridge):
            calibrated = calibrate(readout(abundance, labels, CONFIG), reference, CONFIG)
            self.assertAlmostEqual(
                float(calibrated.context.mean()), float(reference.context.mean()), places=4
            )
            self.assertAlmostEqual(
                float(calibrated.context.std(unbiased=False)),
                float(reference.context.std(unbiased=False)),
                places=4,
            )

    def test_calibration_is_monotone_so_ct_only_auroc_is_unaffected(self):
        context_bags, labels, query_bags = episode(11, context=20)
        abundance = ct_abundance(context_bags, query_bags, CONFIG)
        reference = readout_extreme(abundance, labels, CONFIG)
        raw = readout_ridge(abundance, labels, CONFIG)
        calibrated = calibrate(raw, reference, CONFIG)
        self.assertTrue(torch.equal(torch.argsort(raw.query), torch.argsort(calibrated.query)))


class RidgeTest(unittest.TestCase):
    def test_coefficients_are_antisymmetric_across_the_two_classes(self):
        context_bags, labels, query_bags = episode(12, context=20)
        abundance = ct_abundance(context_bags, query_bags, CONFIG)
        beta, _, _, _ = ridge_coefficients(abundance, labels, CONFIG)
        # One-hot targets sum to 1, so the two columns are mirror images.
        self.assertTrue(torch.allclose(beta[:, 0], -beta[:, 1], atol=1e-5))

    def test_class_balancing_removes_prevalence_from_the_centring(self):
        """The weighted centre must be the midpoint of the two class means.

        Testing this by duplicating context bags would NOT isolate the weighting:
        extra bags also move the coordinate standardisation and the farthest-point
        token set, so the representation changes too. So assert the invariant the
        weights exist for, on a fixed abundance and a deliberately skewed context.
        """
        context_bags, _, query_bags = episode(13, context=18)
        labels = torch.tensor([1] * 13 + [0] * 5)   # 72% positive
        abundance = ct_abundance(context_bags, query_bags, CONFIG)
        _, _, context, _ = ridge_coefficients(abundance, labels, CONFIG)
        counts = torch.bincount(labels, minlength=2)
        weight = counts.float().reciprocal()[labels]
        weighted_centre = (weight[:, None] * context).sum(0) / weight.sum()
        midpoint = 0.5 * (
            context[labels == 0].mean(dim=0) + context[labels == 1].mean(dim=0)
        )
        self.assertTrue(torch.allclose(weighted_centre, midpoint, atol=1e-5))
        # An unweighted centre would sit near the majority class instead.
        self.assertFalse(torch.allclose(context.mean(dim=0), midpoint, atol=1e-3))

    def test_unknown_mode_is_rejected(self):
        context_bags, labels, query_bags = episode(14)
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            ct_margins(context_bags, labels, query_bags, CONFIG, "nope")


class KernelRidgeTest(unittest.TestCase):
    """SS188 (G0): kernel ridge must reproduce the primal ridge under the linear
    kernel, stay deterministic, and keep label antisymmetry -- the invariants the
    fixed head's constants depend on."""

    def kernel_config(self, kernel):
        return CTReadoutConfig(
            num_tokens=8, cells_per_bag=24, temperature=0.5, eps=1e-6,
            sampling="even", kernel=kernel,
        )

    def test_linear_kernel_reproduces_the_primal_ridge(self):
        context_bags, labels, query_bags = episode(20, context=20)
        abundance = ct_abundance(context_bags, query_bags, CONFIG)
        ridge = readout_ridge(abundance, labels, CONFIG)
        kernel = readout_kernel_ridge(abundance, labels, self.kernel_config("linear"))
        self.assertTrue(torch.allclose(ridge.context, kernel.context, atol=1e-4))
        self.assertTrue(torch.allclose(ridge.query, kernel.query, atol=1e-4))

    def test_nonlinear_kernels_are_antisymmetric(self):
        context_bags, labels, query_bags = episode(21, context=20)
        for name in ("rbf", "poly"):
            config = self.kernel_config(name)
            abundance = ct_abundance(context_bags, query_bags, config)
            original = readout_kernel_ridge(abundance, labels, config)
            flipped = readout_kernel_ridge(abundance, 1 - labels, config)
            self.assertTrue(
                torch.allclose(original.query, -flipped.query, atol=1e-4), name
            )

    def test_nonlinear_kernels_are_deterministic(self):
        context_bags, labels, query_bags = episode(22, context=20)
        for name in ("rbf", "poly"):
            config = self.kernel_config(name)
            abundance = ct_abundance(context_bags, query_bags, config)
            first = readout_kernel_ridge(abundance, labels, config)
            second = readout_kernel_ridge(abundance, labels, config)
            self.assertTrue(torch.equal(first.query, second.query), name)

    def test_query_bags_never_change_the_context_side(self):
        context_bags, labels, query_bags = episode(23, context=16)
        other = [bag * 5.0 + 11.0 for bag in query_bags]
        for name in ("rbf", "poly"):
            config = self.kernel_config(name)
            a, first = ct_margins(context_bags, labels, query_bags, config, "kernel_ridge")
            b, second = ct_margins(context_bags, labels, other, config, "kernel_ridge")
            self.assertTrue(torch.equal(first.context, second.context), name)
            self.assertTrue(torch.equal(a.context, b.context), name)


class AbundancePoolingTest(unittest.TestCase):
    """SS189: max/topk pooling of the per-cell assignment adds cell-dimension
    non-linearity. fraction=1.0 must recover the mean; k=1 must recover max."""

    def pooling_config(self, pooling, fraction=0.1, minimum=1):
        return CTReadoutConfig(
            num_tokens=8, cells_per_bag=24, temperature=0.5, eps=1e-6,
            sampling="even", abundance_pooling=pooling,
            abundance_topk_fraction=fraction, abundance_topk_min=minimum,
        )

    def test_topk_fraction_one_recovers_the_mean(self):
        context_bags, _, query_bags = episode(30, context=12, query=4)
        mean = ct_abundance(context_bags, query_bags, CONFIG)
        topk = ct_abundance(
            context_bags, query_bags, self.pooling_config("topk", fraction=1.0)
        )
        self.assertTrue(torch.allclose(mean.context, topk.context, atol=1e-5))
        self.assertTrue(torch.allclose(mean.query, topk.query, atol=1e-5))

    def test_max_matches_topk_with_one_cell(self):
        context_bags, _, query_bags = episode(31, context=12, query=4)
        topk_one = ct_abundance(
            context_bags, query_bags,
            self.pooling_config("topk", fraction=0.0, minimum=1),
        )
        maxed = ct_abundance(context_bags, query_bags, self.pooling_config("max"))
        self.assertTrue(torch.allclose(maxed.context, topk_one.context, atol=1e-5))
        self.assertTrue(torch.allclose(maxed.query, topk_one.query, atol=1e-5))

    def test_topk_is_deterministic(self):
        context_bags, _, query_bags = episode(32, context=12, query=4)
        config = self.pooling_config("topk", fraction=0.3)
        first = ct_abundance(context_bags, query_bags, config)
        second = ct_abundance(context_bags, query_bags, config)
        self.assertTrue(torch.equal(first.context, second.context))
        self.assertTrue(torch.equal(first.query, second.query))

    def test_mean_plus_topk_concatenates_both(self):
        """SS189-2. "mean+topk" appends the top-k vector, it does not replace the
        mean: the first K coordinates equal the mean abundance and the last K
        equal the top-k abundance, so the ridge can use both jointly."""
        context_bags, _, query_bags = episode(34, context=12, query=4)
        mean = ct_abundance(context_bags, query_bags, CONFIG)
        topk = ct_abundance(
            context_bags, query_bags, self.pooling_config("topk", fraction=0.3)
        )
        both = ct_abundance(
            context_bags, query_bags, self.pooling_config("mean+topk", fraction=0.3)
        )
        k = CONFIG.num_tokens
        self.assertEqual(both.context.shape[-1], 2 * k)
        self.assertEqual(both.query.shape[-1], 2 * k)
        for left, right in ((both.context, mean.context), (both.query, mean.query)):
            self.assertTrue(torch.allclose(left[:, :k], right, atol=1e-5))
        for left, right in ((both.context, topk.context), (both.query, topk.query)):
            self.assertTrue(torch.allclose(left[:, k:], right, atol=1e-5))

    def test_unknown_pooling_is_rejected(self):
        context_bags, _, query_bags = episode(33, context=12, query=4)
        with self.assertRaisesRegex(ValueError, "abundance_pooling must be"):
            ct_abundance(
                context_bags, query_bags, self.pooling_config("nope")
            )


class PrepareCellsTest(unittest.TestCase):
    """SS149-7. The diagnostic that measured concentration re-derived the cells by
    hand and passed the TOKENS in by mistake. `prepare_cells` is now the one
    definition, so pin that `ct_abundance` really uses it."""

    def test_tokens_are_drawn_from_prepare_cells_output(self):
        context_bags, _, query_bags = episode(20)
        for pca_dim, basis in ((None, None), (5, torch.linalg.qr(
                torch.randn(DIM, 8, generator=torch.Generator().manual_seed(0)))[0])):
            config = CTReadoutConfig(
                num_tokens=CONFIG.num_tokens, cells_per_bag=CONFIG.cells_per_bag,
                temperature=CONFIG.temperature, eps=CONFIG.eps, pca_dim=pca_dim,
            )
            abundance = ct_abundance(context_bags, query_bags, config, basis)
            context, _ = prepare_cells(context_bags, query_bags, config, basis)
            pooled = torch.cat(context, dim=0)
            self.assertEqual(abundance.tokens.shape[-1], pooled.shape[-1])
            # Every token must BE one of the prepared cells, not a derived point.
            for token in abundance.tokens:
                distance = (pooled - token).square().sum(dim=-1)
                self.assertLess(float(distance.min()), 1e-8)

    def test_projection_reduces_the_working_dimension(self):
        context_bags, _, query_bags = episode(21)
        basis = torch.linalg.qr(
            torch.randn(DIM, 8, generator=torch.Generator().manual_seed(1)))[0]
        config = CTReadoutConfig(pca_dim=5, num_tokens=CONFIG.num_tokens,
                                 cells_per_bag=CONFIG.cells_per_bag)
        context, query = prepare_cells(context_bags, query_bags, config, basis)
        self.assertEqual(context[0].shape[-1], 5)
        self.assertEqual(query[0].shape[-1], 5)

    def test_raw_scaling_keeps_component_variance_unequal(self):
        """`pca_scaling='raw'` must NOT flatten the spectrum, unlike 'standardise'."""
        context_bags, _, query_bags = episode(22)
        basis = torch.linalg.qr(
            torch.randn(DIM, 8, generator=torch.Generator().manual_seed(2)))[0]
        common = dict(pca_dim=6, num_tokens=CONFIG.num_tokens,
                      cells_per_bag=CONFIG.cells_per_bag)
        standardised, _ = prepare_cells(
            context_bags, query_bags, CTReadoutConfig(**common), basis)
        raw, _ = prepare_cells(
            context_bags, query_bags,
            CTReadoutConfig(**common, pca_scaling="raw"), basis)
        spread = lambda cells: torch.cat(cells, 0).var(dim=0)
        self.assertLess(float(spread(standardised).std()), 1e-3)
        self.assertGreater(float(spread(raw).std()), 1e-3)

    def test_unknown_scaling_is_rejected(self):
        context_bags, _, query_bags = episode(23)
        basis = torch.eye(DIM)[:, :4]
        with self.assertRaisesRegex(ValueError, "pca_scaling"):
            prepare_cells(context_bags, query_bags,
                          CTReadoutConfig(pca_dim=4, pca_scaling="nope"), basis)


class KMeansTest(unittest.TestCase):
    """SS157. Lloyd refinement of the farthest-point tokens. `iterations=0` must be
    today's behaviour exactly, or every "k-means gains X" mixes two changes."""

    def _config(self, iterations, pca_dim=None):
        return CTReadoutConfig(
            num_tokens=CONFIG.num_tokens, cells_per_bag=CONFIG.cells_per_bag,
            temperature=CONFIG.temperature, eps=CONFIG.eps,
            pca_dim=pca_dim, kmeans_iterations=iterations,
            sampling=CONFIG.sampling,
        )

    def test_zero_iterations_is_farthest_point_exactly(self):
        context_bags, labels, query_bags = episode(30)
        for mode in ("extreme", "ridge"):
            a, _ = ct_margins(context_bags, labels, query_bags, self._config(0), mode)
            b, _ = ct_margins(context_bags, labels, query_bags, CONFIG, mode)
            self.assertTrue(torch.equal(a.query, b.query), mode)

    def test_kmeans_plusplus_is_reproducible_and_seeded(self):
        pooled = torch.randn(200, DIM, generator=torch.Generator().manual_seed(51))
        first = kmeans_plusplus_tokens(
            pooled, CTReadoutConfig(num_tokens=12, kmeans_seed=7)
        )
        repeated = kmeans_plusplus_tokens(
            pooled, CTReadoutConfig(num_tokens=12, kmeans_seed=7)
        )
        other = kmeans_plusplus_tokens(
            pooled, CTReadoutConfig(num_tokens=12, kmeans_seed=8)
        )
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(first.shape, (12, DIM))

    def test_kmeans_plusplus_handles_a_degenerate_cloud(self):
        pooled = torch.ones(20, 4)
        tokens = kmeans_plusplus_tokens(
            pooled, CTReadoutConfig(num_tokens=8, kmeans_seed=3)
        )
        self.assertEqual(tokens.shape, (8, 4))
        self.assertTrue(torch.equal(tokens, torch.ones_like(tokens)))

    def test_lloyd_early_stopping_matches_one_update(self):
        pooled = torch.randn(80, 5, generator=torch.Generator().manual_seed(52))
        initial = pooled[:6].clone()
        one, _ = lloyd_refine(pooled, initial, 1)
        stopped, _ = lloyd_refine(pooled, initial, 8, tolerance=1e9)
        self.assertTrue(torch.equal(one, stopped))

    def test_lloyd_recovers_empty_cluster_from_high_error_cell(self):
        pooled = torch.tensor([[0.0], [0.1], [4.0], [9.0], [10.0]])
        initial = torch.tensor([[0.0], [0.0], [10.0]])
        kept, _ = lloyd_refine(pooled, initial, 1)
        recovered, _ = lloyd_refine(
            pooled, initial, 1, recover_empty=True
        )
        self.assertTrue(torch.equal(kept[1], initial[1]))
        self.assertFalse(torch.equal(recovered[1], initial[1]))
        self.assertTrue(bool((pooled == recovered[1]).all(dim=1).any()))

    def test_spherical_kmeans_is_reproducible_and_unit_norm(self):
        context_bags, _, query_bags = episode(53)
        config = CTReadoutConfig(
            num_tokens=8,
            cells_per_bag=24,
            sampling="random",
            tokenizer="spherical_kmeans",
            kmeans_seed=7,
            kmeans_max_iterations=5,
        )
        first = ct_abundance(context_bags, query_bags, config)
        repeated = ct_abundance(context_bags, query_bags, config)
        self.assertTrue(torch.equal(first.tokens, repeated.tokens))
        self.assertTrue(torch.equal(first.context, repeated.context))
        self.assertTrue(torch.allclose(
            first.tokens.norm(dim=1), torch.ones(config.num_tokens), atol=1e-6
        ))

    def test_spherical_lloyd_normalises_centroids(self):
        pooled = torch.randn(80, 5, generator=torch.Generator().manual_seed(54))
        pooled = pooled / pooled.norm(dim=1, keepdim=True)
        tokens, _ = lloyd_refine(
            pooled,
            pooled[:6].clone(),
            3,
            "cosine",
            normalise_centroids=True,
        )
        self.assertTrue(torch.allclose(
            tokens.norm(dim=1), torch.ones(tokens.shape[0]), atol=1e-6
        ))

    def test_refinement_moves_the_tokens(self):
        context_bags, _, query_bags = episode(31)
        before = ct_abundance(context_bags, query_bags, self._config(0)).tokens
        after = ct_abundance(context_bags, query_bags, self._config(5)).tokens
        self.assertFalse(torch.allclose(before, after, atol=1e-6))
        self.assertEqual(before.shape, after.shape)

    def test_tokens_stop_being_actual_cells(self):
        """FPS tokens ARE cells; centroids are averages, so they need not be."""
        context_bags, _, query_bags = episode(32)
        context, _ = prepare_cells(context_bags, query_bags, self._config(5), None)
        pooled = torch.cat(context, dim=0)
        tokens = ct_abundance(context_bags, query_bags, self._config(5)).tokens
        distances = [
            float((pooled - token).square().sum(dim=-1).min()) for token in tokens
        ]
        self.assertGreater(max(distances), 1e-6)

    def test_deterministic_across_calls(self):
        context_bags, labels, query_bags = episode(33)
        for iterations in (1, 5, 20):
            a, _ = ct_margins(context_bags, labels, query_bags,
                              self._config(iterations), "ridge")
            b, _ = ct_margins(context_bags, labels, query_bags,
                              self._config(iterations), "ridge")
            self.assertTrue(torch.equal(a.query, b.query), str(iterations))

    def test_query_cells_never_move_the_tokens(self):
        context_bags, _, query_bags = episode(34)
        config = self._config(8)
        first = ct_abundance(context_bags, query_bags, config).tokens
        other = [bag * 4.0 - 9.0 for bag in query_bags]
        second = ct_abundance(context_bags, other, config).tokens
        self.assertTrue(torch.equal(first, second))

    def test_converges_so_extra_iterations_stop_changing_anything(self):
        context_bags, _, query_bags = episode(35)
        near = ct_abundance(context_bags, query_bags, self._config(60)).tokens
        further = ct_abundance(context_bags, query_bags, self._config(80)).tokens
        self.assertTrue(torch.allclose(near, further, atol=1e-5))

    def test_empty_cluster_keeps_its_previous_position(self):
        """Two tight blobs and many tokens forces empties; must not produce NaN."""
        generator = torch.Generator().manual_seed(0)
        bags = [torch.randn(20, DIM, generator=generator) * 0.01 + offset
                for offset in (0.0, 10.0)] * 4
        config = CTReadoutConfig(num_tokens=12, cells_per_bag=20, temperature=0.5,
                                 eps=1e-6, kmeans_iterations=10)
        tokens = ct_abundance(bags, bags[:2], config).tokens
        self.assertTrue(bool(torch.isfinite(tokens).all()))

    def test_cluster_sizes_are_reported_and_sum_to_the_cell_count(self):
        context_bags, _, query_bags = episode(36)
        config = self._config(5)
        context, _ = prepare_cells(context_bags, query_bags, config, None)
        pooled = torch.cat(context, dim=0)
        tokens = farthest_point_tokens(pooled, config)
        _, counts = lloyd_refine(pooled, tokens, config.kmeans_iterations)
        self.assertEqual(int(counts.sum()), pooled.shape[0])


class FullCellTest(unittest.TestCase):
    """SS159. `cells_per_bag=None` uses every cell. The invariants that matter are
    that 64 still reproduces v109 and that the query still cannot reach the tokens
    or the normalisation statistics."""

    def _config(self, cells, **kw):
        return CTReadoutConfig(
            num_tokens=CONFIG.num_tokens, cells_per_bag=cells,
            temperature=CONFIG.temperature, eps=CONFIG.eps, **kw
        )

    def test_none_keeps_every_cell(self):
        context_bags, _, query_bags = episode(40, cells=37)
        context, query = prepare_cells(context_bags, query_bags, self._config(None), None)
        self.assertTrue(all(bag.shape[0] == 37 for bag in context))
        self.assertTrue(all(bag.shape[0] == 37 for bag in query))

    def test_a_cap_below_the_bag_size_still_subsamples(self):
        context_bags, _, query_bags = episode(41, cells=37)
        context, _ = prepare_cells(context_bags, query_bags, self._config(12), None)
        self.assertTrue(all(bag.shape[0] == 12 for bag in context))

    def test_cap_at_or_above_the_bag_size_equals_none(self):
        context_bags, labels, query_bags = episode(42, cells=30)
        wide, _ = ct_margins(context_bags, labels, query_bags, self._config(30), "ridge")
        every, _ = ct_margins(context_bags, labels, query_bags, self._config(None), "ridge")
        self.assertTrue(torch.equal(wide.query, every.query))

    def test_full_cells_change_the_margin(self):
        context_bags, labels, query_bags = episode(43, cells=50)
        few, _ = ct_margins(context_bags, labels, query_bags,
                            self._config(8, kmeans_iterations=5), "ridge")
        every, _ = ct_margins(context_bags, labels, query_bags,
                              self._config(None, kmeans_iterations=5), "ridge")
        self.assertFalse(torch.allclose(few.query, every.query, atol=1e-4))

    def test_abundance_all_keeps_the_capped_dictionary_exact(self):
        """SS165: only the cells averaged into abundance may change."""
        context_bags, _, query_bags = episode(46, cells=50)
        reference = ct_abundance(
            context_bags, query_bags,
            self._config(12, kmeans_iterations=5),
        )
        split = ct_abundance(
            context_bags, query_bags,
            CTReadoutConfig(
                num_tokens=CONFIG.num_tokens,
                cells_per_bag=12,
                abundance_cells_per_bag=None,
                temperature=CONFIG.temperature,
                eps=CONFIG.eps,
                kmeans_iterations=5,
            ),
        )
        self.assertTrue(torch.equal(reference.tokens, split.tokens))
        self.assertFalse(torch.allclose(reference.context, split.context, atol=1e-5))
        self.assertFalse(torch.allclose(reference.query, split.query, atol=1e-5))

    def test_match_is_bit_identical_to_the_legacy_default(self):
        context_bags, _, query_bags = episode(47, cells=50)
        implicit = ct_abundance(
            context_bags, query_bags,
            self._config(12, kmeans_iterations=5),
        )
        explicit = ct_abundance(
            context_bags, query_bags,
            CTReadoutConfig(
                num_tokens=CONFIG.num_tokens,
                cells_per_bag=12,
                abundance_cells_per_bag="match",
                temperature=CONFIG.temperature,
                eps=CONFIG.eps,
                kmeans_iterations=5,
            ),
        )
        for left, right in zip(implicit, explicit):
            self.assertTrue(torch.equal(left, right))

    def test_random_sampling_is_reproducible_and_seeded(self):
        bag = torch.arange(200 * DIM, dtype=torch.float32).reshape(200, DIM)
        first = sample_cells(
            bag,
            CTReadoutConfig(cells_per_bag=32, sampling="random", sampling_seed=7),
        )
        repeated = sample_cells(
            bag,
            CTReadoutConfig(cells_per_bag=32, sampling="random", sampling_seed=7),
        )
        other = sample_cells(
            bag,
            CTReadoutConfig(cells_per_bag=32, sampling="random", sampling_seed=8),
        )
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(first.shape, (32, DIM))

    def test_random_bags_do_not_share_one_index_pattern(self):
        bag = torch.arange(200 * DIM, dtype=torch.float32).reshape(200, DIM)
        context, _ = prepare_cells(
            [bag, bag],
            [bag],
            CTReadoutConfig(cells_per_bag=32, sampling="random", sampling_seed=9),
        )
        self.assertFalse(torch.equal(context[0], context[1]))

    def test_integer_and_none_ignore_unused_fraction_knobs(self):
        bag = torch.arange(80 * DIM, dtype=torch.float32).reshape(80, DIM)
        historical = CTReadoutConfig(cells_per_bag=32, sampling="random", sampling_seed=3)
        unused = CTReadoutConfig(
            cells_per_bag=32, sampling="random", sampling_seed=3,
            cells_min=7, cells_scale="median",
        )
        self.assertTrue(torch.equal(sample_cells(bag, historical), sample_cells(bag, unused)))
        self.assertTrue(torch.equal(
            sample_cells(bag, CTReadoutConfig(cells_per_bag=None, cells_min=7)),
            bag.float(),
        ))

    def test_query_cannot_reach_tokens_or_statistics_at_full_cells(self):
        context_bags, labels, query_bags = episode(44, cells=45)
        config = self._config(None, kmeans_iterations=8)
        first = ct_abundance(context_bags, query_bags, config)
        other = [bag * 7.0 - 3.0 for bag in query_bags]
        second = ct_abundance(context_bags, other, config)
        self.assertTrue(torch.equal(first.tokens, second.tokens))
        self.assertTrue(torch.equal(first.context, second.context))

    def test_deterministic_and_antisymmetric_at_full_cells(self):
        context_bags, labels, query_bags = episode(45, context=20, cells=45)
        config = self._config(None, kmeans_iterations=10)
        a, _ = ct_margins(context_bags, labels, query_bags, config, "ridge")
        b, _ = ct_margins(context_bags, labels, query_bags, config, "ridge")
        flipped, _ = ct_margins(context_bags, 1 - labels, query_bags, config, "ridge")
        self.assertTrue(torch.equal(a.query, b.query))
        self.assertTrue(torch.allclose(a.query, -flipped.query, atol=1e-4))


class SizeAwareSamplingTest(unittest.TestCase):
    """A fixed 512-style cap ignored bag size. The fraction path must not."""

    def test_parse_keeps_historical_tokens_and_reads_fractions(self):
        self.assertEqual(parse_cell_budget("all"), (None, None, None))
        self.assertEqual(parse_cell_budget("64"), (64, None, None))
        self.assertEqual(parse_cell_budget("0.125"), (None, 0.125, None))
        self.assertEqual(parse_cell_budget("frac:0.2"), (None, 0.2, None))
        self.assertEqual(parse_cell_budget("own:0.1"), (None, 0.1, "own"))
        self.assertEqual(parse_cell_budget("median:0.25"), (None, 0.25, "median"))
        with self.assertRaises(ValueError):
            parse_cell_budget("1.5")
        with self.assertRaises(ValueError):
            parse_cell_budget("0")

    def test_larger_bags_draw_more_cells_under_the_same_fraction(self):
        small = torch.arange(80 * DIM, dtype=torch.float32).reshape(80, DIM)
        large = torch.arange(400 * DIM, dtype=torch.float32).reshape(400, DIM)
        config = CTReadoutConfig(
            cells_per_bag=None, cells_fraction=0.25, sampling="random", sampling_seed=4,
        )
        self.assertEqual(sample_cells(small, config).shape[0], 20)
        self.assertEqual(sample_cells(large, config).shape[0], 100)

    def test_fraction_sampling_is_reproducible_and_seeded(self):
        bag = torch.arange(200 * DIM, dtype=torch.float32).reshape(200, DIM)
        config = CTReadoutConfig(
            cells_per_bag=None, cells_fraction=0.2, sampling="random", sampling_seed=7,
        )
        first = sample_cells(bag, config)
        repeated = sample_cells(bag, config)
        other = sample_cells(
            bag,
            CTReadoutConfig(
                cells_per_bag=None, cells_fraction=0.2, sampling="random", sampling_seed=8,
            ),
        )
        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(first.shape, (40, DIM))

    def test_query_bags_do_not_set_the_median_budget(self):
        context = [
            torch.arange(i * 40 * DIM, (i + 1) * 40 * DIM, dtype=torch.float32).reshape(40, DIM)
            for i in range(4)
        ]
        huge_query = [torch.randn(4000, DIM)]
        config = CTReadoutConfig(
            cells_per_bag=None,
            cells_fraction=0.5,
            cells_scale="median",
            sampling="random",
            sampling_seed=5,
        )
        sampled_context, sampled_query = prepare_cells(context, huge_query, config, None)
        self.assertEqual(typical_bag_size(context), 40.0)
        self.assertTrue(all(bag.shape[0] == 20 for bag in sampled_context))
        self.assertEqual(sampled_query[0].shape[0], 20)
        other_query = [torch.randn(9, DIM)]
        again_context, again_query = prepare_cells(context, other_query, config, None)
        self.assertTrue(all(
            torch.equal(left, right) for left, right in zip(sampled_context, again_context)
        ))
        self.assertEqual(again_query[0].shape[0], 9)

    def test_median_budget_is_shared_across_unequal_bags(self):
        bags = [
            torch.randn(10, DIM),
            torch.randn(80, DIM),
            torch.randn(40, DIM),
        ]
        config = CTReadoutConfig(
            cells_per_bag=None,
            cells_fraction=0.5,
            cells_scale="median",
            sampling="even",
        )
        sampled, _ = prepare_cells(bags, [torch.randn(100, DIM)], config, None)
        # Median context n is 40, so the shared budget is 20. The 10-cell bag
        # cannot exceed its own length.
        self.assertEqual([bag.shape[0] for bag in sampled], [10, 20, 20])

    def test_own_fraction_clamps_to_the_bag_and_the_floor(self):
        config = CTReadoutConfig(
            cells_per_bag=None, cells_fraction=0.5, cells_min=8, cells_scale="own",
        )
        self.assertEqual(resolve_cells_per_bag(10, config), 8)
        self.assertEqual(resolve_cells_per_bag(6, config), 6)
        self.assertEqual(resolve_cells_per_bag(40, config), 20)

    def test_kmeans_plusplus_fraction_path_is_deterministic_and_antisymmetric(self):
        context_bags, labels, query_bags = episode(61, context=20, cells=48)
        config = CTReadoutConfig(
            num_tokens=8,
            cells_per_bag=None,
            cells_fraction=0.5,
            sampling="random",
            sampling_seed=2,
            tokenizer="kmeans_plusplus",
            kmeans_seed=3,
            kmeans_max_iterations=4,
        )
        a, _ = ct_margins(context_bags, labels, query_bags, config, "ridge")
        b, _ = ct_margins(context_bags, labels, query_bags, config, "ridge")
        flipped, _ = ct_margins(context_bags, 1 - labels, query_bags, config, "ridge")
        self.assertTrue(torch.equal(a.query, b.query))
        self.assertTrue(torch.allclose(a.query, -flipped.query, atol=1e-4))


class ChunkingTest(unittest.TestCase):
    """SS160. Chunking the cell-to-token distance must be EXACT, not approximate --
    v109 is a promoted baseline and cannot shift because of a memory fix."""

    def test_chunked_assignment_matches_unchunked(self):
        import src.models.ct_readout as module
        generator = torch.Generator().manual_seed(0)
        pooled = torch.randn(5000, 16, generator=generator)
        tokens = torch.randn(24, 16, generator=generator)
        saved = module._DISTANCE_ELEMENT_BUDGET
        try:
            reference = module._assign(pooled, tokens)          # single chunk
            module._DISTANCE_ELEMENT_BUDGET = 24 * 16 * 7       # forces many chunks
            chunked = module._assign(pooled, tokens)
        finally:
            module._DISTANCE_ELEMENT_BUDGET = saved
        self.assertTrue(torch.equal(reference, chunked))

    def test_gemm_distance_matches_broadcast(self):
        import src.models.ct_readout as module
        torch.manual_seed(168)
        pooled = torch.randn(91, 13)
        tokens = torch.randn(17, 13)
        broadcast = module._token_distance(pooled, tokens, "broadcast")
        gemm = module._token_distance(pooled, tokens, "gemm")
        # GEMM intentionally omits ||x||^2, which is constant across tokens.
        restored = gemm + pooled.square().mean(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(broadcast, restored, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.equal(
            module._assign(pooled, tokens, "broadcast"),
            module._assign(pooled, tokens, "gemm"),
        ))

    def test_gemm_abundance_matches_broadcast(self):
        context_bags, _, query_bags = episode(168, cells=37)
        common = dict(num_tokens=12, cells_per_bag=None, kmeans_iterations=3)
        broadcast = ct_abundance(
            context_bags, query_bags,
            CTReadoutConfig(**common, distance_kernel="broadcast"),
        )
        gemm = ct_abundance(
            context_bags, query_bags,
            CTReadoutConfig(**common, distance_kernel="gemm"),
        )
        self.assertTrue(torch.allclose(broadcast.context, gemm.context, atol=2e-5))
        self.assertTrue(torch.allclose(broadcast.query, gemm.query, atol=2e-5))

    def test_chunked_abundance_matches_unchunked(self):
        import src.models.ct_readout as module
        context_bags, _, query_bags = episode(50, cells=400)
        config = CTReadoutConfig(num_tokens=12, cells_per_bag=None,
                                 temperature=0.5, eps=1e-6, kmeans_iterations=3)
        saved = module._DISTANCE_ELEMENT_BUDGET
        try:
            reference = ct_abundance(context_bags, query_bags, config)
            module._DISTANCE_ELEMENT_BUDGET = 12 * DIM * 5
            chunked = ct_abundance(context_bags, query_bags, config)
        finally:
            module._DISTANCE_ELEMENT_BUDGET = saved
        self.assertTrue(torch.equal(reference.tokens, chunked.tokens))
        self.assertTrue(torch.allclose(reference.context, chunked.context, atol=1e-6))
        self.assertTrue(torch.allclose(reference.query, chunked.query, atol=1e-6))

    def test_v109_setting_stays_in_one_chunk(self):
        """64 cells/bag x 16 tokens must never chunk, so v109 is bit-identical."""
        import src.models.ct_readout as module
        rows = module._DISTANCE_ELEMENT_BUDGET // (16 * 32)
        self.assertGreater(rows, 200 * 64)


class HierarchicalTwoMeansTest(unittest.TestCase):
    def test_exact_power_of_two_count_and_determinism(self):
        generator = torch.Generator().manual_seed(168)
        pooled = torch.randn(1024, 12, generator=generator)
        config = CTReadoutConfig(
            num_tokens=64, tokenizer="hierarchical_2means",
            bisect_iterations=2, bisect_power_iterations=3,
        )
        first = hierarchical_2means_tokens(pooled, config)
        second = hierarchical_2means_tokens(pooled, config)
        self.assertEqual(first.shape, (64, 12))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

    def test_identical_points_use_nonempty_fallback(self):
        pooled = torch.ones(128, 7)
        config = CTReadoutConfig(
            num_tokens=32, tokenizer="hierarchical_2means",
            bisect_iterations=2, bisect_power_iterations=2,
        )
        tokens = hierarchical_2means_tokens(pooled, config)
        self.assertEqual(tokens.shape, (32, 7))
        self.assertTrue(torch.equal(tokens, torch.ones_like(tokens)))

    def test_atomic_reduction_still_guarantees_exact_count(self):
        pooled = torch.randn(4096, 9, generator=torch.Generator().manual_seed(169))
        tokens = hierarchical_2means_tokens(
            pooled,
            CTReadoutConfig(
                num_tokens=128, tokenizer="hierarchical_2means",
                tree_reduction="atomic",
            ),
        )
        self.assertEqual(tokens.shape, (128, 9))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_non_power_of_two_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            hierarchical_2means_tokens(
                torch.randn(100, 5),
                CTReadoutConfig(num_tokens=24, tokenizer="hierarchical_2means"),
            )


@unittest.skipUnless(torch.cuda.is_available(), "GPU HDBSCAN requires CUDA")
class HDBSCANTest(unittest.TestCase):
    def test_two_obvious_blobs_choose_two_clusters_without_fixed_k(self):
        try:
            import cuml  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("RAPIDS cuML is not installed")
        generator = torch.Generator(device="cuda").manual_seed(169)
        pooled = torch.cat([
            torch.randn(512, 8, generator=generator, device="cuda") * 0.1 - 3,
            torch.randn(512, 8, generator=generator, device="cuda") * 0.1 + 3,
        ])
        config = CTReadoutConfig(
            tokenizer="hdbscan",
            num_tokens=777,
            hdbscan_min_cluster_size=64,
            hdbscan_min_cluster_fraction=0.0,
            hdbscan_min_samples=16,
        )
        tokens = hdbscan_tokens(pooled, config)
        self.assertEqual(tokens.shape, (2, 8))
        self.assertTrue(torch.allclose(
            tokens.sort(dim=0).values.mean(dim=1),
            torch.tensor([-3.0, 3.0], device="cuda"), atol=0.1,
        ))


@unittest.skipUnless(torch.cuda.is_available(), "GPU DBSCAN requires CUDA")
class DBSCANTest(unittest.TestCase):
    def test_adaptive_eps_finds_blobs_and_ignores_fixed_token_count(self):
        try:
            import cuml  # noqa: F401, PLC0415
        except ImportError:
            self.skipTest("RAPIDS cuML is not installed")
        generator = torch.Generator(device="cuda").manual_seed(170)
        pooled = torch.cat([
            torch.randn(512, 8, generator=generator, device="cuda") * 0.1 - 3,
            torch.randn(512, 8, generator=generator, device="cuda") * 0.1 + 3,
        ])
        tokens = dbscan_tokens(
            pooled,
            CTReadoutConfig(
                tokenizer="dbscan", num_tokens=999,
                dbscan_eps=None, dbscan_min_samples=16,
            ),
        )
        self.assertEqual(tokens.shape, (2, 8))
        self.assertTrue(torch.allclose(
            tokens.sort(dim=0).values.mean(dim=1),
            torch.tensor([-3.0, 3.0], device="cuda"), atol=0.1,
        ))

if __name__ == "__main__":
    unittest.main()
