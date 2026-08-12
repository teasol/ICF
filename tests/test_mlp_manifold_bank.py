import pytest
import torch

from src.datasets.synthetic_data import SyntheticManifoldGenerator


def _generator(bank_size: int = 8) -> SyntheticManifoldGenerator:
    return SyntheticManifoldGenerator(
        num_bags=4,
        num_cells=8,
        latent_dim=4,
        output_dim=12,
        mlp_hidden_dim=7,
        mlp_num_layers=3,
        manifold_mode="mlp_bank",
        manifold_seed=31,
        manifold_bank_size=bank_size,
    )


def test_mlp_bank_replays_the_same_member() -> None:
    manifold = _generator()
    z = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(2))
    first = manifold._map_episode_manifold(
        z, torch.Generator().manual_seed(19), torch.device("cpu")
    )
    replay = manifold._map_episode_manifold(
        z, torch.Generator().manual_seed(19), torch.device("cpu")
    )
    torch.testing.assert_close(first, replay)


def test_mlp_bank_uses_multiple_fixed_members() -> None:
    manifold = _generator()
    z = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(2))
    outputs = [
        manifold._map_episode_manifold(
            z, torch.Generator().manual_seed(seed), torch.device("cpu")
        )
        for seed in range(16)
    ]
    assert any(not torch.equal(outputs[0], output) for output in outputs[1:])


def test_mlp_bank_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="manifold_bank_size"):
        _generator(bank_size=0)


def test_mixed_mode_probability_endpoints_match_each_branch() -> None:
    z = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(2))
    common = dict(
        num_bags=4, num_cells=8, latent_dim=4, output_dim=12,
        mlp_hidden_dim=7, mlp_num_layers=3, manifold_seed=31,
        manifold_bank_size=8,
    )
    mixed_linear = SyntheticManifoldGenerator(
        **common, manifold_mode="mixed_linear_mlp_bank",
        manifold_linear_probability=1.0,
    )
    mixed_mlp = SyntheticManifoldGenerator(
        **common, manifold_mode="mixed_linear_mlp_bank",
        manifold_linear_probability=0.0,
    )
    linear = mixed_linear._map_episode_manifold(
        z, torch.Generator().manual_seed(5), torch.device("cpu")
    )
    nonlinear = mixed_mlp._map_episode_manifold(
        z, torch.Generator().manual_seed(5), torch.device("cpu")
    )
    assert linear.shape == nonlinear.shape == (2, 5, 12)
    assert not torch.equal(linear, nonlinear)
