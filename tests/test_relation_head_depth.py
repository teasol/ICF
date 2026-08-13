"""Contract tests for the configurable CV/DD/CT relation head (docs SS108).

The head used to be a hard-coded `Linear(12,h) -> GELU -> Linear(h,1)`. It is now
built from `ct_head_hidden_dims`, so `[]` is a linear probe and `[32,32]` is a
deeper MLP. The point of these tests is that adding the knob did not disturb the
default: v82 and every earlier relation checkpoint must keep loading strict, with
byte-identical initial weights.
"""

import unittest

import torch
from torch import nn

from src.models.set_transformer_ridge import (
    CovarianceMeanDDCTMLPModel,
    CovarianceMeanLearnablePDDCTMLPModel,
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
    return CovarianceMeanLearnablePDDCTMLPModel(**kwargs)


class RelationHeadShapeTest(unittest.TestCase):
    def test_default_head_is_unchanged(self):
        """No kwarg -> the historical Linear/GELU/Linear at keys 0 and 2."""
        model = build()
        head = model.cv_dd_ct_head
        self.assertEqual(len(head), 3)
        self.assertIsInstance(head[0], nn.Linear)
        self.assertIsInstance(head[1], nn.GELU)
        self.assertIsInstance(head[2], nn.Linear)
        self.assertEqual(head[0].in_features, 12)
        self.assertEqual(head[0].out_features, 32)
        self.assertEqual(head[2].out_features, 1)
        self.assertEqual(
            sorted(name for name, _ in head.named_parameters()),
            ["0.bias", "0.weight", "2.bias", "2.weight"],
        )

    def test_default_init_is_byte_identical_to_the_old_hard_coded_head(self):
        """The builder must draw exactly what the old literal head drew.

        This is what protects checkpoint reproducibility. Comparing two models
        built by the current code would be a tautology, so the reference here is
        the literal `Linear(12,h) -> GELU -> Linear(h,1)` the head used to be,
        constructed from the same RNG state: everything before the head is
        identical, so patching only the builder isolates the head's draws.
        """
        def legacy_head(hidden_dims):
            width = hidden_dims[0]
            return nn.Sequential(
                nn.Linear(12, width), nn.GELU(), nn.Linear(width, 1)
            )

        model = build()

        # Fetch through __dict__: plain attribute access unwraps the
        # staticmethod, and restoring the bare function would rebind it as an
        # instance method and break every later test in this module.
        original = CovarianceMeanDDCTMLPModel.__dict__["_build_relation_head"]
        CovarianceMeanDDCTMLPModel._build_relation_head = staticmethod(legacy_head)
        try:
            legacy_model = build()
        finally:
            CovarianceMeanDDCTMLPModel._build_relation_head = original

        legacy_params = dict(legacy_model.cv_dd_ct_head.named_parameters())
        new_params = dict(model.cv_dd_ct_head.named_parameters())
        self.assertEqual(sorted(new_params), sorted(legacy_params))
        for name, parameter in new_params.items():
            self.assertTrue(torch.equal(parameter, legacy_params[name]), name)

    def test_explicit_single_hidden_layer_matches_default(self):
        default = build()
        explicit = build(ct_head_hidden_dims=[32])
        self.assertEqual(
            [type(m) for m in default.cv_dd_ct_head],
            [type(m) for m in explicit.cv_dd_ct_head],
        )
        for (name, a), (_, b) in zip(
            default.cv_dd_ct_head.named_parameters(),
            explicit.cv_dd_ct_head.named_parameters(),
        ):
            self.assertTrue(torch.equal(a, b), name)

    def test_linear_head_has_no_activation(self):
        model = build(ct_head_hidden_dims=[])
        head = model.cv_dd_ct_head
        self.assertEqual(len(head), 1)
        self.assertIsInstance(head[0], nn.Linear)
        self.assertEqual(head[0].in_features, 12)
        self.assertEqual(head[0].out_features, 1)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 13)

    def test_deep_head_stacks_hidden_layers(self):
        model = build(ct_head_hidden_dims=[32, 32])
        head = model.cv_dd_ct_head
        self.assertEqual(
            [type(m) for m in head],
            [nn.Linear, nn.GELU, nn.Linear, nn.GELU, nn.Linear],
        )
        self.assertEqual(head[0].in_features, 12)
        self.assertEqual(head[2].in_features, 32)
        self.assertEqual(head[4].out_features, 1)
        expected = (12 * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
        self.assertEqual(sum(p.numel() for p in head.parameters()), expected)

    def test_trainable_parameter_counts_at_baseline_dims(self):
        """P (1536x128 = 196,608) + head. The default total is the documented 197,057."""
        def trainable(**overrides):
            kwargs = dict(BASE_KWARGS)
            kwargs.update(input_dim=1536, covariance_sketch_dim=128)
            kwargs.update(overrides)
            torch.manual_seed(0)
            model = CovarianceMeanLearnablePDDCTMLPModel(**kwargs)
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertEqual(trainable(), 197_057)
        self.assertEqual(trainable(ct_head_hidden_dims=[]), 196_608 + 13)
        self.assertEqual(
            trainable(ct_head_hidden_dims=[32, 32]), 196_608 + 449 + 1056
        )

    def test_head_parameter_counts_at_test_dims(self):
        def head_params(**overrides):
            return sum(
                p.numel() for p in build(**overrides).cv_dd_ct_head.parameters()
            )

        self.assertEqual(head_params(), 449)
        self.assertEqual(head_params(ct_head_hidden_dims=[]), 13)
        self.assertEqual(head_params(ct_head_hidden_dims=[32, 32]), 449 + 1056)

    def test_rejects_bad_widths(self):
        with self.assertRaises(ValueError):
            build(ct_head_hidden_dims=[32, 0])
        with self.assertRaises(ValueError):
            build(ct_head_hidden_dims=[-1])
        # A bare int is a plausible typo for the list form and would silently
        # iterate as a scalar; reject it loudly instead.
        with self.assertRaises(ValueError):
            build(ct_head_hidden_dims=32)


class RelationHeadCheckpointTest(unittest.TestCase):
    def test_default_state_dict_loads_strict(self):
        source = build()
        target = build()
        target.load_state_dict(source.state_dict(), strict=True)

    def test_shape_change_fails_loudly_not_silently(self):
        """A checkpoint from one head shape must not half-load into another."""
        source = build()
        for widths in ([], [32, 32]):
            target = build(ct_head_hidden_dims=widths)
            with self.assertRaises(RuntimeError):
                target.load_state_dict(source.state_dict(), strict=True)

    def test_base_class_gets_the_knob_too(self):
        """The head lives on the v74 base, so every relation arm inherits it."""
        torch.manual_seed(0)
        model = CovarianceMeanDDCTMLPModel(**BASE_KWARGS, ct_head_hidden_dims=[])
        self.assertEqual(len(model.cv_dd_ct_head), 1)


class RelationHeadForwardTest(unittest.TestCase):
    def _episode(self):
        """8 bags in one tensor; the last two are the queries.

        `forward(instances, labels, query_index)` takes every bag together and
        splits context vs query by index -- there is no separate query tensor.
        """
        generator = torch.Generator().manual_seed(7)
        bags = torch.randn(8, 40, BASE_KWARGS["input_dim"], generator=generator)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])
        query_index = torch.tensor([6, 7])
        return bags, labels, query_index

    def test_forward_and_symmetry_hold_for_every_head_shape(self):
        bags, labels, query_index = self._episode()
        for widths in ([], [32], [32, 32]):
            with self.subTest(widths=widths):
                model = build(ct_head_hidden_dims=widths)
                model.eval()
                with torch.no_grad():
                    logits = model(bags, labels, query_index)
                self.assertEqual(logits.shape, (2, 2))
                self.assertTrue(torch.isfinite(logits).all())
                # logits are built as [-margin/2, +margin/2]
                self.assertTrue(
                    torch.allclose(
                        logits.sum(dim=-1), torch.zeros(2), atol=1e-5
                    )
                )

    def test_gradient_reaches_p_and_head_for_every_shape(self):
        """SS100's contract: assert P's gradient is finite and nonzero.

        `nonfinite_gradient_policy: zero` makes a broken backward path silent, so
        a new head shape has to prove the gradient still arrives.
        """
        bags, labels, query_index = self._episode()
        for widths in ([], [32], [32, 32]):
            with self.subTest(widths=widths):
                model = build(ct_head_hidden_dims=widths)
                logits = model(bags, labels, query_index)
                logits.square().sum().backward()
                projection_grad = model._covariance_projection.grad
                self.assertIsNotNone(projection_grad)
                self.assertTrue(torch.isfinite(projection_grad).all())
                self.assertGreater(float(projection_grad.abs().max()), 0.0)
                for name, parameter in model.cv_dd_ct_head.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
