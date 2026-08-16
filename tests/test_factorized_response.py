"""Factorized / XOR response and the orthogonal manifold (generator knobs, live).

These were module-level pytest functions with no `TestCase`, so
`python -m unittest discover` collected the file and ran NOTHING from it -- a
silent zero, quieter and worse than the import error in the sibling bank test.
Rewritten as `unittest.TestCase`; the assertions are unchanged.
"""

import unittest

import torch

from src.datasets.synthetic_data import SyntheticManifoldGenerator


def _kwargs(**overrides):
    kwargs = dict(
        num_bags=32,
        num_cells=64,
        latent_dim=6,
        output_dim=12,
        mlp_hidden_dim=10,
        shared_component_probability=1.0,
        continuous_response_probability=1.0,
        num_shared_components=4,
        response_task_probabilities=(0.0, 1.0, 0.0, 0.0, 0.0),
        balanced=False,
    )
    kwargs.update(overrides)
    return kwargs


def _episode(seed, **kwargs):
    generator = torch.Generator().manual_seed(seed)
    return SyntheticManifoldGenerator(**_kwargs(**kwargs)).sample_episode(generator)


class DefaultEquivalenceTest(unittest.TestCase):
    def test_explicit_scalar_one_population_preserves_default_episode(self):
        """Spelling out the defaults must not perturb the RNG stream."""
        implicit = _episode(123)
        explicit = _episode(
            123,
            response_dim=1,
            responsive_population_count=1,
            label_rule="single",
            random_causal_factors=False,
            separate_nuisance_rng=False,
        )
        self.assertTrue(torch.equal(implicit.y, explicit.y))
        self.assertTrue(torch.equal(implicit.response_score, explicit.response_score))
        self.assertTrue(torch.equal(implicit.x, explicit.x))


class XorLabelTest(unittest.TestCase):
    def test_xor_label_is_exact_and_each_factor_is_marginally_balanced(self):
        episode = _episode(456, response_dim=2, responsive_population_count=2,
                           label_rule="xor")
        bits = episode.response_score.gt(0).long()
        expected = torch.logical_xor(bits[:, 0].bool(), bits[:, 1].bool()).long()
        self.assertTrue(torch.equal(episode.y, expected))
        for factor in range(2):
            agreement = float((bits[:, factor] == episode.y).float().mean())
            self.assertGreater(agreement, 0.25)
            self.assertLess(agreement, 0.75)

    def test_full_arm_exposes_eight_factors_four_populations_two_causal(self):
        episode = _episode(
            789,
            response_dim=8,
            responsive_population_count=4,
            label_rule="xor",
            random_causal_factors=True,
            separate_nuisance_rng=True,
        )
        self.assertEqual(episode.response_score.shape, (32, 8))
        self.assertEqual(episode.causal_factor_indices.shape, (2,))
        self.assertEqual(episode.causal_factor_indices.unique().numel(), 2)
        self.assertEqual(episode.responsive_population_factors.shape, (4, 2))
        bits = episode.response_score.gt(0)
        causal = episode.causal_factor_indices
        expected = torch.logical_xor(bits[:, causal[0]], bits[:, causal[1]]).long()
        self.assertTrue(torch.equal(episode.y, expected))


class NuisanceSeedTest(unittest.TestCase):
    def test_separate_nuisance_seed_is_independent_of_response_configuration(self):
        scalar = _episode(321, separate_nuisance_rng=True)
        xor = _episode(
            321,
            response_dim=8,
            responsive_population_count=4,
            label_rule="xor",
            random_causal_factors=True,
            separate_nuisance_rng=True,
        )
        self.assertIsNotNone(scalar.nuisance_seed)
        self.assertEqual(scalar.nuisance_seed, xor.nuisance_seed)


class OrthogonalManifoldTest(unittest.TestCase):
    def test_orthogonal_linear_manifold_preserves_latent_geometry(self):
        generator = SyntheticManifoldGenerator(
            num_bags=4, num_cells=8, latent_dim=16, output_dim=1536,
            manifold_mode="orthogonal",
        )
        z = torch.randn(2, 7, 16, generator=torch.Generator().manual_seed(11))
        mapped = generator._map_episode_manifold(
            z, torch.Generator().manual_seed(12), torch.device("cpu")
        )
        self.assertEqual(mapped.shape, (2, 7, 1536))
        torch.testing.assert_close(
            torch.cdist(z[0], z[0]), torch.cdist(mapped[0], mapped[0]),
            rtol=1e-5, atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
