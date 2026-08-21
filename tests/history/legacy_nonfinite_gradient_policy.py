"""Pin the non-finite gradient policy (docs SS67).

Default is `raise` -- the historical fail-fast that caught the SS66 P-2 ablation
blowing up at epoch 13. `zero` keeps training by dropping the poisoned entries,
which is what the stabilised ablation arms need.

Why this exists as a separate lever from `gradient_clip_val`: the guard runs in
`on_before_optimizer_step`, which Lightning calls BEFORE clipping, so the raise
pre-empts clipping entirely. And clipping cannot repair a NaN regardless -- one
non-finite entry makes the total norm non-finite, and the clip coefficient then
poisons every other gradient. Clipping bounds the finite-but-large gradients
that precede a blow-up; zeroing removes the non-finite ones. They are
complementary, not alternatives.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.model_interface import ModelInterface  # noqa: E402


class _Tiny(torch.nn.Module):
    architecture_version = 34

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)


def _interface(**overrides) -> ModelInterface:
    return ModelInterface(
        model_src=f"{__name__}._Tiny",
        **overrides,
    )


class TestNonfiniteGradientPolicy(unittest.TestCase):
    def test_default_policy_is_raise(self) -> None:
        """Unchanged fail-fast for every existing config."""
        interface = _interface()
        self.assertEqual(interface._nonfinite_gradient_policy, "raise")
        for parameter in interface.parameters():
            parameter.grad = torch.full_like(parameter, float("nan"))
        with self.assertRaises(RuntimeError):
            interface._raise_if_nonfinite_gradients("test")

    def test_zero_policy_replaces_and_counts(self) -> None:
        """`zero` drops nan/inf, keeps finite values, and counts the event."""
        interface = _interface(nonfinite_gradient_policy="zero")
        weight = interface.model.linear.weight
        bias = interface.model.linear.bias
        weight.grad = torch.tensor([[float("nan"), 1.0, float("inf"), -2.0],
                                    [float("-inf"), 3.0, 0.5, 0.25]])
        bias.grad = torch.tensor([1.5, -0.5])

        interface._raise_if_nonfinite_gradients("test")

        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertEqual(interface._nonfinite_gradient_steps, 1)
        # nan/inf -> 0, every finite entry untouched.
        torch.testing.assert_close(
            weight.grad,
            torch.tensor([[0.0, 1.0, 0.0, -2.0], [0.0, 3.0, 0.5, 0.25]]),
        )
        torch.testing.assert_close(bias.grad, torch.tensor([1.5, -0.5]))

    def test_zero_policy_is_a_noop_on_clean_gradients(self) -> None:
        """A healthy step must not be counted or altered."""
        interface = _interface(nonfinite_gradient_policy="zero")
        for parameter in interface.parameters():
            parameter.grad = torch.ones_like(parameter)
        interface._raise_if_nonfinite_gradients("test")
        self.assertEqual(interface._nonfinite_gradient_steps, 0)
        for parameter in interface.parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))

    def test_invalid_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _interface(nonfinite_gradient_policy="skip")

    def test_policy_is_not_forwarded_to_the_model_constructor(self) -> None:
        """Trainer-side knobs must not leak into the model's kwargs."""
        interface = _interface(nonfinite_gradient_policy="zero")
        self.assertIsInstance(interface.model, _Tiny)


if __name__ == "__main__":
    unittest.main()
