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

from src.models.set_transformer_ridge import SetTransformerRidgeModel  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
