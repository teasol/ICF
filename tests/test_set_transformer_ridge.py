"""Contracts for the learned-token + ridge branch (docs SS75).

This model exists to test the one axis SS69 never reached: a bag descriptor
that is LEARNED, with the gradient arriving through the closed-form ridge. The
properties below are what make such a descriptor trustworthy, and each one is
cheap to break silently:

  cells are a set                  no dependence on cell order
  bags are a set                   no dependence on bag order
  the query's label is unseen      changing it must change nothing
  classes are names                swapping them flips the logits
  the encoder actually trains      gradient survives the ridge solve

The last one is the risk this branch is built around: SS66 recorded gradients
diverging once the covariance ridge was disturbed, and here every encoder
gradient has to pass through a Cholesky solve.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.set_transformer_ridge import (  # noqa: E402
    CovarianceSetTransformerRidgeModel,
    CovarianceMeanCVMLPModel,
    CovarianceMeanDDMLPModel,
    CovarianceMeanDDCTMLPModel,
    CovarianceMeanLearnablePDDCTMLPModel,
    PopulationTokenResidualModel,
    CovarianceMeanCV2DDCTMLPModel,
    CovarianceMeanDDMagnitudeMLPModel,
    CovarianceMeanDDRidgeModel,
    CovarianceMeanRidgeModel,
    CovarianceOnlyRidgeModel,
    STCVLPRidgeModel,
    SetTransformerRidgeModel,
)

INPUT_DIM = 24


def _model(seed: int = 0, **overrides) -> SetTransformerRidgeModel:
    torch.manual_seed(seed)
    kwargs = dict(
        input_dim=INPUT_DIM,
        token_dim=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=48,
        num_summary_tokens=4,
        max_cells=64,
    )
    kwargs.update(overrides)
    return SetTransformerRidgeModel(**kwargs).eval()


def _episode(bags: int = 8, cells: int = 20):
    torch.manual_seed(5)
    x = torch.randn(bags, cells, INPUT_DIM)
    y = torch.tensor([0, 1] * (bags // 2))
    return x, y, torch.tensor([bags - 2, bags - 1])


class SetTransformerRidgeTest(unittest.TestCase):
    def test_forward_shape_and_finiteness(self) -> None:
        model = _model()
        x, y, query = _episode()
        logits = model(x, y, query)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_cell_order_does_not_matter(self) -> None:
        model = _model()
        x, y, query = _episode()
        expected = model(x, y, query)
        shuffled = x[:, torch.randperm(x.shape[1])]
        torch.testing.assert_close(
            expected, model(shuffled, y, query), atol=1e-4, rtol=1e-4,
            msg="cells within a bag are a set; the encoder read their order",
        )

    def test_bag_order_does_not_matter(self) -> None:
        model = _model()
        x, y, query = _episode()
        expected = model(x, y, query)
        # Permute context bags only, leaving the two query bags in place.
        permutation = torch.cat((torch.randperm(6), torch.tensor([6, 7])))
        torch.testing.assert_close(
            expected,
            model(x[permutation], y[permutation], query),
            atol=1e-4,
            rtol=1e-4,
            msg="context bags are a set; the ridge read their order",
        )

    def test_query_labels_are_never_read(self) -> None:
        model = _model()
        x, y, query = _episode()
        expected = model(x, y, query)
        changed = y.clone()
        changed[query] = 1 - changed[query]
        torch.testing.assert_close(expected, model(x, changed, query))

    def test_label_swap_is_equivariant(self) -> None:
        model = _model()
        x, y, query = _episode()
        expected = model(x, y, query)
        torch.testing.assert_close(
            expected, model(x, 1 - y, query).flip(-1), atol=1e-4, rtol=1e-4,
            msg="renaming the classes must only flip the logits",
        )

    def test_gradient_reaches_the_encoder_through_the_ridge(self) -> None:
        """The premise of the branch: a descriptor optimised for its readout."""
        model = _model().train()
        x, y, query = _episode()
        logits = model(x, y, query)
        torch.nn.functional.cross_entropy(logits.float(), y[query]).backward()
        encoder_parameters = dict(model.encoder.named_parameters())
        self.assertTrue(encoder_parameters)
        live = [
            name
            for name, parameter in encoder_parameters.items()
            if parameter.grad is not None and float(parameter.grad.abs().max()) > 0
        ]
        self.assertEqual(
            len(live),
            len(encoder_parameters),
            "some encoder parameters got no gradient through the ridge solve: "
            f"{sorted(set(encoder_parameters) - set(live))}",
        )
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in encoder_parameters.values()
            ),
            "non-finite gradient through the Cholesky solve",
        )

    def test_ragged_and_dense_agree(self) -> None:
        model = _model()
        x, y, query = _episode()
        torch.testing.assert_close(
            model(x, y, query),
            model(list(x.unbind(0)), y, query),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_batched_matches_sequential(self) -> None:
        model = _model()
        x, y, query = _episode()
        batch_x = torch.stack((x, x + 0.1))
        batch_y = torch.stack((y, y.roll(2)))
        batch_query = query.expand(2, -1)
        expected = torch.stack(
            [
                model(one_x, one_y, one_query)
                for one_x, one_y, one_query in zip(batch_x, batch_y, batch_query)
            ]
        )
        actual = model.forward_episode_batch(batch_x, batch_y, batch_query)
        torch.testing.assert_close(expected, actual, atol=1e-4, rtol=1e-4)

    def test_padded_cells_are_ignored(self) -> None:
        """A padded ragged batch must equal the unpadded episode it encodes."""
        model = _model()
        x, y, query = _episode(cells=12)
        padded = torch.cat((x, torch.randn(x.shape[0], 7, INPUT_DIM) * 50), dim=1)
        cell_mask = torch.zeros(
            1, x.shape[0], padded.shape[1], dtype=torch.bool
        )
        cell_mask[:, :, : x.shape[1]] = True
        with torch.no_grad():
            reference = model.forward_episode_batch(
                x.unsqueeze(0), y.unsqueeze(0), query.unsqueeze(0)
            )
            masked = model.forward_episode_batch(
                padded.unsqueeze(0),
                y.unsqueeze(0),
                query.unsqueeze(0),
                cell_mask=cell_mask,
            )
        torch.testing.assert_close(
            reference, masked, atol=1e-4, rtol=1e-4,
            msg="padding leaked into the bag token despite the cell mask",
        )

    def test_missing_class_in_context_is_rejected(self) -> None:
        model = _model()
        x, _, query = _episode()
        single_class = torch.zeros(x.shape[0], dtype=torch.long)
        with self.assertRaises(ValueError):
            model(x, single_class, query)

    def test_checkpoint_version_marker_is_present(self) -> None:
        state = _model().state_dict()
        self.assertIn("_architecture_version", state)
        self.assertEqual(
            int(state["_architecture_version"]),
            SetTransformerRidgeModel.architecture_version,
        )

    def test_descriptor_is_all_summary_tokens_flattened(self) -> None:
        """The ridge must see S x token_dim, not one pooled vector.

        Collapsing a bag to a single token was the previous design's mistake:
        it put a 256-number bottleneck against the covariance sketch's 8,256.
        """
        model = _model(num_summary_tokens=4, token_dim=32)
        x, _, _ = _episode()
        descriptors = model._descriptors(x)
        self.assertEqual(descriptors.shape, (x.shape[0], 4 * 32))
        self.assertEqual(model.descriptor_dim, 4 * 32)

    def test_cells_above_the_cap_are_subsampled(self) -> None:
        model = _model(max_cells=16)
        torch.manual_seed(11)
        cells = torch.randn(3, 400, INPUT_DIM)
        kept, _ = model.encoder._subsample(cells, None)
        self.assertEqual(kept.shape[1], 16)

    def test_cells_below_the_cap_are_untouched(self) -> None:
        """The cap must bind on the tail only, and exactly, not approximately."""
        model = _model(max_cells=64)
        cells = torch.randn(3, 64, INPUT_DIM)
        kept, _ = model.encoder._subsample(cells, None)
        self.assertTrue(torch.equal(kept, cells))

    def test_summary_tokens_attend_to_each_other(self) -> None:
        """They are prepended to the sequence, not pooled independently.

        If each summary token only read the cells, perturbing one token's
        parameters could not change another's output. It can.
        """
        model = _model(num_summary_tokens=4)
        x, _, _ = _episode()
        torch.manual_seed(13)
        with torch.no_grad():
            before = model.encoder(x)
            # A NON-UNIFORM perturbation. Adding a constant to every dimension
            # would be removed by the pre-norm LayerNorm, so that version of
            # this test passed for the wrong reason (measured 1.8e-6).
            model.encoder.summary_tokens[0] += torch.randn_like(
                model.encoder.summary_tokens[0]
            )
            after = model.encoder(x)
        moved = (before[:, 1:] - after[:, 1:]).abs().max()
        self.assertGreater(
            float(moved),
            1e-4,
            "changing one summary token left the others unchanged; they are "
            "not attending to each other",
        )


class CovarianceSetTransformerRidgeTest(unittest.TestCase):
    def _hybrid(self, **overrides) -> CovarianceSetTransformerRidgeModel:
        torch.manual_seed(0)
        kwargs = dict(
            input_dim=INPUT_DIM, token_dim=32, num_heads=4, num_layers=2,
            feedforward_dim=48, num_summary_tokens=4, max_cells=64,
            covariance_sketch_dim=8, covariance_slopes=(0.07, 0.05),
        )
        kwargs.update(overrides)
        return CovarianceSetTransformerRidgeModel(**kwargs).eval()

    def test_centered_summary_is_invariant_to_per_bag_translation(self) -> None:
        model = self._hybrid(center_cells=True)
        x, _, _ = _episode()
        shift = torch.randn(x.shape[0], 1, x.shape[-1]) * 7
        torch.testing.assert_close(
            model.encoder(x), model.encoder(x + shift), atol=2e-5, rtol=2e-5
        )

    def test_centered_summary_ignores_padded_values(self) -> None:
        model = self._hybrid(center_cells=True)
        x, _, _ = _episode(cells=12)
        padded = torch.cat((x, torch.randn(x.shape[0], 7, INPUT_DIM) * 50), dim=1)
        mask = torch.zeros(x.shape[0], padded.shape[1], dtype=torch.bool)
        mask[:, : x.shape[1]] = True
        torch.testing.assert_close(
            model.encoder(x), model.encoder(padded, cell_mask=mask),
            atol=1e-4, rtol=1e-4,
        )

    def test_descriptor_dimensions_are_balanced_and_concatenated(self) -> None:
        model = self._hybrid()
        x, _, _ = _episode()
        descriptor = model._descriptors(x)
        self.assertEqual(model.summary_descriptor_dim, 4 * 32)
        self.assertEqual(model.covariance_descriptor_dim, 8 * 9 // 2)
        self.assertEqual(model.mean_descriptor_dim, INPUT_DIM)
        self.assertEqual(descriptor.shape, (x.shape[0], 128 + 36 + INPUT_DIM))

    def test_covariance_descriptor_is_exact_cv1_upper_triangle(self) -> None:
        model = self._hybrid()
        x, _, _ = _episode(bags=8, cells=12)
        centered = x.float() - x.float().mean(dim=-2, keepdim=True)
        projected = centered @ model._covariance_projection.float()
        covariance = projected.transpose(-1, -2) @ projected / x.shape[-2]
        row, column = model._covariance_triangle
        torch.testing.assert_close(
            model._covariance_descriptors(x).float(), covariance[..., row, column]
        )

    def test_each_block_is_normalized_independently(self) -> None:
        model = self._hybrid()
        context = torch.randn(7, model.descriptor_dim)
        query = torch.randn(2, model.descriptor_dim)
        normalized, _ = model._normalize_descriptors(context, query)
        summary, covariance, mean = normalized.split(
            (model.summary_descriptor_dim, model.covariance_descriptor_dim,
             model.mean_descriptor_dim), dim=-1
        )
        self.assertAlmostEqual(float(summary.square().mean()), 1.0, places=5)
        self.assertAlmostEqual(float(covariance.square().mean()), 1.0, places=5)
        self.assertAlmostEqual(float(mean.square().mean()), 1.0, places=5)

    def test_hybrid_gradient_reaches_encoder(self) -> None:
        model = self._hybrid().train()
        x, y, query = _episode()
        torch.nn.functional.cross_entropy(model(x, y, query), y[query]).backward()
        self.assertTrue(all(p.grad is not None for p in model.encoder.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.encoder.parameters()))



class MeanTokenSetTransformerTest(unittest.TestCase):
    def test_one_summary_plus_mean_is_1024_at_full_width(self) -> None:
        model = SetTransformerRidgeModel(
            input_dim=INPUT_DIM, token_dim=512, num_heads=8, num_layers=1,
            feedforward_dim=128, num_summary_tokens=1, max_cells=64,
            center_cells=True, include_mean_token=True,
        ).eval()
        x, _, _ = _episode(cells=12)
        descriptor = model._descriptors(x)
        self.assertEqual(model.descriptor_dim, 1024)
        self.assertEqual(descriptor.shape, (x.shape[0], 1024))

    def test_mean_token_ignores_padding_and_cell_order(self) -> None:
        model = _model(
            num_summary_tokens=1, center_cells=True, include_mean_token=True
        )
        x, _, _ = _episode(cells=12)
        reference = model.encoder(x)
        permutation = torch.randperm(x.shape[1])
        torch.testing.assert_close(
            reference, model.encoder(x[:, permutation]), atol=1e-4, rtol=1e-4
        )
        padded = torch.cat((x, torch.randn(x.shape[0], 7, INPUT_DIM) * 50), dim=1)
        mask = torch.zeros(x.shape[0], padded.shape[1], dtype=torch.bool)
        mask[:, : x.shape[1]] = True
        torch.testing.assert_close(
            reference, model.encoder(padded, cell_mask=mask),
            atol=1e-4, rtol=1e-4,
        )

if __name__ == "__main__":
    unittest.main()


class STCVLPRidgeTest(unittest.TestCase):
    def _model(self) -> STCVLPRidgeModel:
        torch.manual_seed(0)
        return STCVLPRidgeModel(
            input_dim=INPUT_DIM, token_dim=32, num_heads=4, num_layers=2,
            feedforward_dim=48, num_summary_tokens=4, max_cells=64,
            covariance_sketch_dim=8, covariance_slopes=(0.07, 0.05),
            center_cells=True,
        )

    def test_three_branches_are_concatenated(self) -> None:
        model = self._model().eval()
        x, _, _ = _episode()
        descriptor = model._descriptors(x)
        self.assertEqual(model.summary_descriptor_dim, 128)
        self.assertEqual(model.covariance_descriptor_dim, 36)
        self.assertEqual(model.mean_descriptor_dim, INPUT_DIM)
        self.assertEqual(model.lp_descriptor_dim, 36)
        self.assertEqual(descriptor.shape, (x.shape[0], 200 + INPUT_DIM))

    def test_lp_starts_equal_to_fixed_cv(self) -> None:
        model = self._model().eval()
        x, _, _ = _episode()
        torch.testing.assert_close(
            model._lp_descriptors(x), model._covariance_descriptors(x),
            atol=2e-5, rtol=2e-5,
        )

    def test_lp_projection_is_learned_through_ridge(self) -> None:
        model = self._model().train()
        x, y, query = _episode()
        torch.nn.functional.cross_entropy(model(x, y, query), y[query]).backward()
        self.assertIsNotNone(model.lp_projection.grad)
        self.assertTrue(torch.isfinite(model.lp_projection.grad).all())
        self.assertGreater(float(model.lp_projection.grad.abs().max()), 0.0)

    def test_all_three_blocks_are_normalized_independently(self) -> None:
        model = self._model().eval()
        context = torch.randn(7, model.descriptor_dim)
        query = torch.randn(2, model.descriptor_dim)
        normalized, _ = model._normalize_descriptors(context, query)
        blocks = normalized.split(
            (model.summary_descriptor_dim, model.covariance_descriptor_dim,
             model.mean_descriptor_dim, model.lp_descriptor_dim), dim=-1
        )
        for block in blocks:
            self.assertAlmostEqual(float(block.square().mean()), 1.0, places=5)

class CovarianceMeanAblationTest(unittest.TestCase):
    def _kwargs(self):
        return dict(
            input_dim=INPUT_DIM, token_dim=32, num_heads=4, num_layers=1,
            feedforward_dim=48, num_summary_tokens=2, max_cells=64,
            covariance_sketch_dim=8, covariance_slopes=(0.07, 0.05),
        )

    def test_cv_only_descriptor_has_no_st_tokens(self):
        model = CovarianceOnlyRidgeModel(**self._kwargs()).eval()
        x, _, _ = _episode(cells=12)
        self.assertEqual(model._descriptors(x).shape, (x.shape[0], 36))

    def test_mean_is_computed_before_bag_centering(self):
        model = CovarianceMeanRidgeModel(**self._kwargs()).eval()
        x, _, _ = _episode(cells=12)
        shift = torch.randn(x.shape[0], 1, INPUT_DIM)
        descriptor = model._descriptors(x)
        shifted = model._descriptors(x + shift)
        torch.testing.assert_close(
            shifted[:, :36], descriptor[:, :36], atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(
            shifted[:, 36:] - descriptor[:, 36:], shift.squeeze(1),
            atol=2e-5, rtol=2e-5,
        )

    def test_mean_ignores_padding(self):
        model = CovarianceMeanRidgeModel(**self._kwargs()).eval()
        x, _, _ = _episode(cells=12)
        padded = torch.cat((x, torch.randn(x.shape[0], 7, INPUT_DIM) * 50), dim=1)
        mask = torch.zeros(x.shape[0], padded.shape[1], dtype=torch.bool)
        mask[:, :x.shape[1]] = True
        torch.testing.assert_close(
            model._descriptors(x), model._descriptors(padded, cell_mask=mask),
            atol=1e-4, rtol=1e-4,
        )

    def test_cv_and_mean_blocks_are_normalized_independently(self):
        model = CovarianceMeanRidgeModel(**self._kwargs()).eval()
        context = torch.randn(7, model.descriptor_dim)
        query = torch.randn(2, model.descriptor_dim)
        normalized, _ = model._normalize_descriptors(context, query)
        covariance, mean = normalized.split((36, INPUT_DIM), dim=-1)
        self.assertAlmostEqual(float(covariance.square().mean()), 1.0, places=5)
        self.assertAlmostEqual(float(mean.square().mean()), 1.0, places=5)

    def test_dd_ensemble_returns_normalized_log_probabilities(self):
        model = CovarianceMeanDDRidgeModel(**self._kwargs()).eval()
        x, y, query = _episode(cells=12)
        output = model(x, y, query)
        probabilities = output.exp()
        torch.testing.assert_close(
            probabilities.sum(dim=-1), torch.ones(query.numel()),
            atol=1e-5, rtol=1e-5,
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_dd_ensemble_respects_label_swap(self):
        model = CovarianceMeanDDRidgeModel(**self._kwargs()).eval()
        x, y, query = _episode(cells=12)
        forward = model(x, y, query)
        swapped = model(x, 1 - y, query)
        torch.testing.assert_close(forward, swapped.flip(-1), atol=2e-4, rtol=2e-4)

    def test_cv_dd_mlp_is_the_only_trainable_module(self):
        model = CovarianceMeanDDMLPModel(**self._kwargs()).train()
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("cv_dd_head.") for name in trainable))
        self.assertEqual(sum(value.numel() for value in model.parameters() if value.requires_grad), 321)

        x, y, query = _episode(cells=12)
        logits = model(x, y, query)
        self.assertEqual(logits.shape, (query.numel(), 2))
        torch.nn.functional.cross_entropy(logits, y[query]).backward()
        for name, value in model.named_parameters():
            if value.requires_grad:
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)

    def test_cv_dd_ct_mlp_contract_and_label_swap(self):
        model = CovarianceMeanDDCTMLPModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
        ).train()
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("cv_dd_ct_head.") for name in trainable))
        self.assertEqual(
            sum(value.numel() for value in model.parameters() if value.requires_grad),
            449,
        )
        x, y, query = _episode(cells=12)
        logits = model(x, y, query)
        self.assertEqual(logits.shape, (query.numel(), 2))
        self.assertTrue(torch.isfinite(logits).all())
        torch.nn.functional.cross_entropy(logits, y[query]).backward()
        self.assertTrue(all(
            value.grad is not None and torch.isfinite(value.grad).all()
            for value in model.cv_dd_ct_head.parameters()
        ))

        context = [x[i] for i in range(6)]
        query_bags = [x[6], x[7]]
        q0, q1, separation = model._ct_features(
            context, y[:6], query_bags
        )
        swapped0, swapped1, swapped_separation = model._ct_features(
            context, 1 - y[:6], query_bags
        )
        torch.testing.assert_close(q0, swapped1)
        torch.testing.assert_close(q1, swapped0)
        torch.testing.assert_close(separation, swapped_separation)

    def test_cv_cv2_dd_ct_mlp_trainable_contract(self):
        model = CovarianceMeanCV2DDCTMLPModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
        ).train()
        trainable = [
            name for name, value in model.named_parameters() if value.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(
            name.startswith(("cv2_relation_head.", "cv_cv2_dd_ct_head."))
            for name in trainable
        ))
        self.assertEqual(
            sum(value.numel() for value in model.parameters() if value.requires_grad),
            803,
        )
        x, y, query = _episode(cells=12)
        logits = model(x, y, query)
        self.assertEqual(logits.shape, (query.numel(), 2))
        self.assertTrue(torch.isfinite(logits).all())
        torch.nn.functional.cross_entropy(logits, y[query]).backward()
        for name, value in model.named_parameters():
            if value.requires_grad:
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)

        descriptors = model._descriptors(x)
        context_mask = model._context_split(x.shape[0], query, x.device)
        with torch.no_grad():
            features, separation = model._cv2_relation_features(
                model._covariance_matrices_from_triangle(descriptors[context_mask]),
                y[context_mask],
                model._covariance_matrices_from_triangle(descriptors[query]),
            )
        self.assertEqual(features.shape, (query.numel(), 4))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.isfinite(separation))

    def test_learnable_p_dd_ct_contract_and_gradient(self):
        model = CovarianceMeanLearnablePDDCTMLPModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
        ).train()
        trainable = {
            name: value for name, value in model.named_parameters()
            if value.requires_grad
        }
        self.assertEqual(
            set(trainable),
            {
                "_covariance_projection",
                "cv_dd_ct_head.0.weight", "cv_dd_ct_head.0.bias",
                "cv_dd_ct_head.2.weight", "cv_dd_ct_head.2.bias",
            },
        )
        self.assertEqual(sum(value.numel() for value in trainable.values()), 641)
        projection = model._effective_covariance_projection()
        torch.testing.assert_close(
            projection.T @ projection,
            torch.eye(model.covariance_sketch_dim),
            atol=2e-5, rtol=2e-5,
        )
        x, y, query = _episode(cells=12)
        loss = torch.nn.functional.cross_entropy(model(x, y, query), y[query])
        loss.backward()
        gradient = model._covariance_projection.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().max()), 0.0)

    def test_learnable_p_can_train_ridge_calibration(self):
        model = CovarianceMeanLearnablePDDCTMLPModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8,
            train_ridge_calibration=True,
        ).train()
        trainable = {
            name: value for name, value in model.named_parameters()
            if value.requires_grad
        }
        self.assertEqual(
            set(trainable),
            {
                "_covariance_projection", "ridge_log_lambda", "ridge_log_scale",
                "cv_dd_ct_head.0.weight", "cv_dd_ct_head.0.bias",
                "cv_dd_ct_head.2.weight", "cv_dd_ct_head.2.bias",
            },
        )
        x, y, query = _episode(cells=12)
        loss = torch.nn.functional.cross_entropy(model(x, y, query), y[query])
        loss.backward()
        for name in ("ridge_log_lambda", "ridge_log_scale"):
            gradient = trainable[name].grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs()), 0.0)

    def _learnable_p(self, seed, **extra):
        torch.manual_seed(seed)
        return CovarianceMeanLearnablePDDCTMLPModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8, **extra
        ).train()

    def test_dd_projection_defaults_off_and_is_forward_identical(self):
        """v78 widens the backward graph only -- forward values must not move."""
        control = self._learnable_p(31)
        arm = self._learnable_p(31, train_dd_projection=True)
        self.assertFalse(control.train_dd_projection)
        self.assertTrue(arm.train_dd_projection)
        # v74 has no learnable projection, so it must stay opted out.
        self.assertFalse(
            CovarianceMeanDDCTMLPModel(
                **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
            ).train_dd_projection
        )
        x, y, query = _episode(cells=12)
        torch.testing.assert_close(control(x, y, query), arm(x, y, query))

    def test_dd_direction_never_enters_the_autograd_graph(self):
        """The eigh/argmax block must stay out of backward in every arm.

        Both eigh backwards carry 1/(lambda_i - lambda_j) and the argmax
        selection is discontinuous, so v78 holds the direction constant and only
        differentiates the quadratic form that consumes it.
        """
        for flag in (False, True):
            with self.subTest(train_dd_projection=flag):
                model = self._learnable_p(31, train_dd_projection=flag)
                x, y, _ = _episode(cells=12)
                triangle = model._covariance_descriptors(x).float()
                triangle.requires_grad_(True)
                covariance = model._covariance_matrices_from_triangle(triangle)
                self.assertTrue(covariance.requires_grad)
                direction = model._dd_direction(covariance, y)
                self.assertFalse(direction.requires_grad)
                self.assertIsNone(direction.grad_fn)

    def test_dd_projection_actually_reaches_p_and_changes_its_gradient(self):
        """Guard against a silent null.

        `nonfinite_gradient_policy: zero` would let a broken DD path train to
        completion while contributing nothing, so assert the gradient is finite,
        non-zero, AND different from the control's -- otherwise a Delta of ~0 in
        the arm cannot be read as the hypothesis failing.
        """
        x, y, query = _episode(cells=12)
        gradients = {}
        for flag in (False, True):
            model = self._learnable_p(31, train_dd_projection=flag)
            loss = torch.nn.functional.cross_entropy(model(x, y, query), y[query])
            loss.backward()
            gradient = model._covariance_projection.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().max()), 0.0)
            gradients[flag] = gradient.clone()
        self.assertFalse(
            torch.allclose(gradients[False], gradients[True]),
            "train_dd_projection did not change P's gradient -- the DD path is "
            "not actually reaching the projection",
        )

    def test_dd_projection_gradient_weight_scales_only_the_dd_path(self):
        """The weight must be exact identity forward and linear in backward.

        Unweighted, DD reaches P with ~52x the CV path's gradient magnitude at
        1536-d/K=128, so this knob is what keeps v78 a controlled change.
        """
        x, y, query = _episode(cells=12)
        control = self._learnable_p(31)
        control_logits = control(x, y, query)
        torch.nn.functional.cross_entropy(control_logits, y[query]).backward()
        cv_only = control._covariance_projection.grad.clone()

        contributions = {}
        for weight in (0.25, 0.5):
            model = self._learnable_p(
                31, train_dd_projection=True, dd_projection_gradient_weight=weight
            )
            logits = model(x, y, query)
            torch.testing.assert_close(logits, control_logits)
            torch.nn.functional.cross_entropy(logits, y[query]).backward()
            contributions[weight] = (
                model._covariance_projection.grad - cv_only
            ).clone()
        torch.testing.assert_close(
            contributions[0.5], contributions[0.25] * 2.0, atol=1e-6, rtol=1e-4
        )
        # Zero weight must reduce to the control exactly.
        zero = self._learnable_p(
            31, train_dd_projection=True, dd_projection_gradient_weight=0.0
        )
        torch.nn.functional.cross_entropy(zero(x, y, query), y[query]).backward()
        torch.testing.assert_close(zero._covariance_projection.grad, cv_only)
        with self.assertRaises(ValueError):
            self._learnable_p(31, dd_projection_gradient_weight=-1.0)

    def test_dd_projection_adds_no_parameters_and_loads_strict_both_ways(self):
        control = self._learnable_p(31)
        arm = self._learnable_p(31, train_dd_projection=True)
        for model in (control, arm):
            self.assertEqual(
                sum(v.numel() for v in model.parameters() if v.requires_grad), 641
            )
        self.assertEqual(arm.architecture_version, control.architecture_version)
        arm.load_state_dict(control.state_dict(), strict=True)
        control.load_state_dict(arm.state_dict(), strict=True)

    def test_learnable_p_starts_from_v74_outputs(self):
        torch.manual_seed(17)
        baseline = CovarianceMeanDDCTMLPModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
        ).eval()
        torch.manual_seed(17)
        learnable = CovarianceMeanLearnablePDDCTMLPModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
        ).eval()
        x, y, query = _episode(cells=12)
        torch.testing.assert_close(
            learnable(x, y, query), baseline(x, y, query),
            atol=2e-5, rtol=2e-5,
        )

    def test_population_residual_trainable_contract_and_zero_init(self):
        torch.manual_seed(19)
        baseline = CovarianceMeanLearnablePDDCTMLPModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
        ).eval()
        torch.manual_seed(19)
        model = PopulationTokenResidualModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
        ).train()
        trainable = {
            name: value for name, value in model.named_parameters()
            if value.requires_grad
        }
        self.assertEqual(
            set(trainable),
            {
                "population_residual_gate",
                "population_relation_head.0.weight",
                "population_relation_head.0.bias",
                "population_relation_head.2.weight",
                "population_relation_head.2.bias",
            },
        )
        self.assertEqual(sum(value.numel() for value in trainable.values()), 162)
        x, y, query = _episode(cells=12)
        torch.testing.assert_close(model(x, y, query), baseline(x, y, query))
        loss = torch.nn.functional.cross_entropy(model(x, y, query), y[query])
        loss.backward()
        self.assertIsNotNone(model.population_residual_gate.grad)
        self.assertTrue(torch.isfinite(model.population_residual_gate.grad))

    def test_population_residual_is_label_antisymmetric(self):
        model = PopulationTokenResidualModel(
            **self._kwargs(), ct_num_tokens=4, ct_cells_per_bag=8
        ).eval()
        x, y, query = _episode(cells=12)
        context = [x[i] for i in range(6)]
        query_bags = [x[i] for i in query.tolist()]
        margin = model._population_residual_margin(context, y[:6], query_bags)
        swapped = model._population_residual_margin(
            context, 1 - y[:6], query_bags
        )
        torch.testing.assert_close(margin, -swapped)

    def test_cv_dd_magnitude_mlp_is_the_only_trainable_module(self):
        model = CovarianceMeanDDMagnitudeMLPModel(**self._kwargs()).train()
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("cv_dd_magnitude_head.") for name in trainable)
        )
        self.assertEqual(
            sum(value.numel() for value in model.parameters() if value.requires_grad),
            449,
        )

        x, y, query = _episode(cells=12)
        logits = model(x, y, query)
        self.assertEqual(logits.shape, (query.numel(), 2))
        self.assertTrue(torch.isfinite(logits).all())
        torch.nn.functional.cross_entropy(logits, y[query]).backward()
        for name, value in model.named_parameters():
            if value.requires_grad:
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)

        context_mean = torch.randn(6, INPUT_DIM)
        query_mean = torch.randn(2, INPUT_DIM)
        distances, separation = model._magnitude_distance_features(
            context_mean, torch.tensor([0, 0, 0, 1, 1, 1]), query_mean
        )
        self.assertEqual(distances.shape, (2, 2))
        self.assertTrue(torch.isfinite(distances).all())
        self.assertTrue(torch.isfinite(separation))

    def test_cv_mlp_ablation_is_the_only_trainable_module(self):
        model = CovarianceMeanCVMLPModel(**self._kwargs()).train()
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("cv_head.") for name in trainable))
        self.assertEqual(sum(value.numel() for value in model.parameters() if value.requires_grad), 193)

        x, y, query = _episode(cells=12)
        logits = model(x, y, query)
        torch.nn.functional.cross_entropy(logits, y[query]).backward()
        self.assertEqual(logits.shape, (query.numel(), 2))
        for name, value in model.named_parameters():
            if value.requires_grad:
                self.assertIsNotNone(value.grad, name)
                self.assertTrue(torch.isfinite(value.grad).all(), name)
