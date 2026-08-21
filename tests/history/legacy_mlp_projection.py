"""Contract tests for the MLP cell projection (docs SS134, v105).

`CovarianceMeanMLPProjectionDDCTMLPModel` replaces the single learnable matrix P
with `act(x @ W1) @ QR(W2).Q`. Two properties have to hold for the arm to mean
what it claims:

1. **The QR pin survives.** The whole reason this is not simply "lineage B again"
   (rejected in SS79) is that the OUTPUT projection is still orthonormalised, so
   covariance magnitudes stay on the scale that `ridge_lambda` and
   `covariance_slopes` were calibrated for (SS70). If that pin were lost the arm
   would be confounded with a recalibration and any result would be
   uninterpretable. `test_output_projection_stays_orthonormal` pins it.
2. **It is a strict superset of the linear model.** With `identity` activation the
   composition is linear, so the model can still express exactly what P expressed.
   `test_identity_activation_is_linear` pins that, which is what makes a loss
   attributable to the nonlinearity rather than to lost capacity.

Gradient reach is pinned for the same reason as SS100 and
tests/test_population_attention.py: `nonfinite_gradient_policy: zero` makes a
broken backward path silent, so a dead projection would train quietly and look
like a clean null.
"""

import unittest

import torch
from torch import nn

from src.models.set_transformer_ridge import (
    CovarianceMeanLearnablePDDCTMLPModel,
    CovarianceMeanMLPProjectionDDCTMLPModel,
)

BASE_KWARGS = dict(
    input_dim=64,
    token_dim=16,
    num_heads=1,
    num_layers=1,
    feedforward_dim=16,
    num_summary_tokens=1,
    max_cells=128,
    dropout=0.0,
    covariance_sketch_dim=8,
    ridge_lambda=1.0,
    ridge_logit_scale=2.0,
    num_classes=2,
)


def build(**overrides):
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    torch.manual_seed(0)
    return CovarianceMeanMLPProjectionDDCTMLPModel(**kwargs)


def episode(generator_seed=7):
    generator = torch.Generator().manual_seed(generator_seed)
    num_bags, num_cells, dim = 8, 40, BASE_KWARGS["input_dim"]
    bags = torch.randn(num_bags, num_cells, dim, generator=generator)
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])
    query_index = torch.tensor([6, 7])
    return bags, labels, query_index


class MLPProjectionShapeTest(unittest.TestCase):
    def test_architecture_version_is_58(self):
        self.assertEqual(int(build()._architecture_version.item()), 58)

    def test_shapes_and_that_old_p_is_replaced(self):
        model = build(projection_hidden_dim=32)
        self.assertEqual(tuple(model.projection_input.weight.shape), (32, 64))
        # The sketch matrix is now hidden x K, not input_dim x K.
        self.assertEqual(tuple(model._covariance_projection.shape), (32, 8))

    def test_rejects_bad_knobs(self):
        with self.assertRaises(ValueError):  # hidden < covariance_sketch_dim
            build(projection_hidden_dim=4)
        with self.assertRaises(ValueError):
            build(projection_activation="tanh")

    def test_checkpoint_does_not_silently_half_load_from_v98(self):
        """A v98 checkpoint carries P as [input_dim, K] and must not load into
        the [hidden, K] sketch matrix."""
        torch.manual_seed(0)
        source = CovarianceMeanLearnablePDDCTMLPModel(**BASE_KWARGS)
        with self.assertRaises(RuntimeError):
            build(projection_hidden_dim=32).load_state_dict(
                source.state_dict(), strict=True
            )


class MLPProjectionContractTest(unittest.TestCase):
    def test_output_projection_stays_orthonormal(self):
        """The SS70 calibration pin: QR(W2).Q must have orthonormal columns even
        after the raw parameter is perturbed."""
        model = build(projection_hidden_dim=32)
        with torch.no_grad():
            model._covariance_projection.mul_(7.3).add_(0.5)
        projection = model._effective_covariance_projection()
        gram = projection.T @ projection
        self.assertTrue(torch.allclose(gram, torch.eye(8), atol=1e-5))

    def test_identity_activation_is_linear(self):
        """With `identity` the model is a composition of two linear maps, so the
        descriptor must equal the one from the composed matrix. This is what makes
        the arm a superset of the linear model rather than a different one."""
        model = build(projection_hidden_dim=32, projection_activation="identity")
        cells = torch.randn(3, 40, BASE_KWARGS["input_dim"])
        centered = cells - cells.mean(dim=-2, keepdim=True)
        composed = model.projection_input.weight.T @ model._effective_covariance_projection()
        self.assertTrue(
            torch.allclose(model._project_cells(centered), centered @ composed, atol=1e-5)
        )

    def test_forward_is_finite_and_antisymmetric(self):
        bags, labels, query_index = episode()
        model = build(projection_hidden_dim=32)
        model.eval()
        with torch.no_grad():
            logits = model(bags, labels, query_index)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.allclose(logits.sum(dim=-1), torch.zeros(2), atol=1e-5))

    def test_gradient_reaches_both_projection_layers(self):
        """SS100's contract. `nonfinite_gradient_policy: zero` hides a broken
        backward path, so a dead layer would train silently."""
        bags, labels, query_index = episode()
        model = build(projection_hidden_dim=32)
        model(bags, labels, query_index).square().sum().backward()
        for name, parameter in [
            ("projection_input.weight", model.projection_input.weight),
            ("_covariance_projection", model._covariance_projection),
        ]:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
