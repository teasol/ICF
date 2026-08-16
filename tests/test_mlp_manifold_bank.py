"""`manifold_mode="mlp_bank"` and its mixed variant (generator knobs, live).

Written in pytest style originally, which meant `python -m unittest discover`
could not import it at all -- the suite reported one hard error and none of
these four checks ever ran. Rewritten as `unittest.TestCase` so the repo needs
no pytest dependency for two files.
"""

import unittest

import torch

from src.datasets.synthetic_data import SyntheticManifoldGenerator

COMMON = dict(
    num_bags=4,
    num_cells=8,
    latent_dim=4,
    output_dim=12,
    mlp_hidden_dim=7,
    mlp_num_layers=3,
    manifold_seed=31,
    manifold_bank_size=8,
)


def _generator(bank_size: int = 8) -> SyntheticManifoldGenerator:
    return SyntheticManifoldGenerator(
        **{**COMMON, "manifold_bank_size": bank_size}, manifold_mode="mlp_bank"
    )


def _latents() -> torch.Tensor:
    return torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(2))


def _map(manifold: SyntheticManifoldGenerator, z: torch.Tensor, seed: int) -> torch.Tensor:
    return manifold._map_episode_manifold(
        z, torch.Generator().manual_seed(seed), torch.device("cpu")
    )


class MlpBankTest(unittest.TestCase):
    def test_mlp_bank_replays_the_same_member(self):
        manifold, z = _generator(), _latents()
        torch.testing.assert_close(_map(manifold, z, 19), _map(manifold, z, 19))

    def test_mlp_bank_uses_multiple_fixed_members(self):
        manifold, z = _generator(), _latents()
        outputs = [_map(manifold, z, seed) for seed in range(16)]
        self.assertTrue(
            any(not torch.equal(outputs[0], output) for output in outputs[1:])
        )

    def test_mlp_bank_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "manifold_bank_size"):
            _generator(bank_size=0)


class MixedModeTest(unittest.TestCase):
    def test_probability_endpoints_match_each_branch(self):
        z = _latents()
        linear = _map(
            SyntheticManifoldGenerator(
                **COMMON,
                manifold_mode="mixed_linear_mlp_bank",
                manifold_linear_probability=1.0,
            ),
            z,
            5,
        )
        nonlinear = _map(
            SyntheticManifoldGenerator(
                **COMMON,
                manifold_mode="mixed_linear_mlp_bank",
                manifold_linear_probability=0.0,
            ),
            z,
            5,
        )
        self.assertEqual(linear.shape, (2, 5, 12))
        self.assertEqual(nonlinear.shape, (2, 5, 12))
        self.assertFalse(torch.equal(linear, nonlinear))


if __name__ == "__main__":
    unittest.main()
