"""Pin the v36 Q1 `population_token_mode` contract.

`projected` (default) collapses a bag's structured tokens to one BEFORE the
population branch, which makes `_population_memory_logits`' routing softmax a
length-1 axis -- `population_slot_weights` is identically 1.0, so the ABMIL-style
selection mechanism is inert. `structured` feeds the full token set instead.

The dense (training) and ragged (eval) paths duplicate the token-selection
logic, so `test_dense_and_ragged_paths_agree` is the guard that a future change
touching only one copy fails here rather than silently training the old
behaviour (see the `_all_structured_tokens_batched` comment in baseline.py).
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline import BaseModel  # noqa: E402
from src.utils.utils import merge_train_config  # noqa: E402

INPUT_DIM = 96


def _build(mode: str) -> BaseModel:
    """Build the v34/v35 architecture at a small input dim, in `mode`."""
    config = merge_train_config(
        REPO_ROOT / "configs" / "train_v34_phase0_largectx_1536.yaml"
    )
    kwargs = dict(config["model"])
    kwargs.update(config["model_kwargs"])
    kwargs.pop("model_src", None)
    kwargs["input_dim"] = INPUT_DIM
    kwargs["aggregator_covariance_sketch_dim"] = 16
    kwargs["aggregator_slot_affinity_dim"] = INPUT_DIM
    kwargs["meta_population_token_mode"] = mode
    accepted = inspect.signature(BaseModel.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    torch.manual_seed(0)
    return BaseModel(**kwargs).eval()


def _episode(n_bags: int = 8, n_cells: int = 120):
    torch.manual_seed(1)
    x = torch.randn(n_bags, n_cells, INPUT_DIM)
    y = torch.tensor([0, 1] * (n_bags // 2))
    return x, y


class TestPopulationTokenMode(unittest.TestCase):
    def test_projected_routing_is_degenerate(self) -> None:
        """Current behaviour, pinned: the routing softmax has nothing to route."""
        model = _build("projected")
        x, y = _episode()
        with torch.no_grad():
            _, auxiliary = model(x, y, torch.tensor([2, 5]), return_auxiliary=True)
        weights = auxiliary["population_slot_weights"]
        self.assertEqual(weights.shape[-1], 1)
        self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))

    def test_structured_routing_is_non_degenerate(self) -> None:
        """`structured` restores a real distribution over the token set."""
        model = _build("structured")
        expected = model.meta_classifier.structured_tokens_per_bag
        x, y = _episode()
        with torch.no_grad():
            _, auxiliary = model(x, y, torch.tensor([2, 5]), return_auxiliary=True)
        weights = auxiliary["population_slot_weights"].float()
        self.assertEqual(weights.shape[-1], expected)
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(len(weights))))
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
        self.assertTrue(
            bool((entropy > 0).all()),
            "routing collapsed to a single token at init -- that would rebuild "
            "the 40->1 bottleneck through another route.",
        )

    def test_structured_is_not_the_legacy_slot_only_path(self) -> None:
        """`structured` = 1 global + 3*num_slots + tails, NOT slot tokens alone."""
        model = _build("structured")
        aggregator = model.aggregator
        expected = 1 + 3 * aggregator.num_slots + len(aggregator.tail_fractions)
        self.assertEqual(model.meta_classifier.structured_tokens_per_bag, expected)
        self.assertNotEqual(expected, 3 * aggregator.num_slots)

    def test_dense_and_ragged_paths_agree(self) -> None:
        """The 4D (training) and ragged (eval) paths must implement one rule."""
        for mode in ("projected", "structured"):
            with self.subTest(mode=mode):
                model = _build(mode)
                x, y = _episode()
                mask = torch.tensor([2, 5])
                with torch.no_grad():
                    ragged = model(list(x.unbind(0)), y, mask)
                    dense = model.forward_episode_batch(
                        x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0)
                    )
                delta = (ragged.float() - dense[0].float()).abs().max()
                self.assertLess(
                    float(delta),
                    1e-4,
                    f"dense/ragged disagree in {mode!r} mode "
                    f"(||delta||inf={float(delta):.3e}); the token-selection "
                    "logic is duplicated across both paths and one copy drifted.",
                )

    def test_mode_adds_no_parameters_and_keeps_checkpoints_loadable(self) -> None:
        """Shape-preserving: a `projected` checkpoint loads into `structured`."""
        projected = _build("projected")
        structured = _build("structured")
        keys_p = {k: tuple(v.shape) for k, v in projected.state_dict().items()}
        keys_s = {k: tuple(v.shape) for k, v in structured.state_dict().items()}
        self.assertEqual(keys_p, keys_s)
        self.assertEqual(
            sum(p.numel() for p in projected.parameters()),
            sum(p.numel() for p in structured.parameters()),
        )
        structured.load_state_dict(projected.state_dict(), strict=True)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build("all")


if __name__ == "__main__":
    unittest.main()
