"""Core contracts for the learned-token + ridge branch (docs SS75).

Historical ablation tests covering specific experimental arms (v74-v80,
dual projection, population residual, etc.) have been archived to
`tests/history/legacy_set_transformer_ridge.py`.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.set_transformer_ridge import (  # noqa: E402
    CovarianceSetTransformerRidgeModel,
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
            expected,
            model(shuffled, y, query),
            atol=1e-4,
            rtol=1e-4,
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
            msg="context bags are a set; permutation changed the prediction",
        )

    def test_query_labels_are_never_read(self) -> None:
        model = _model()
        x, y, query = _episode()
        expected = model(x, y, query)
        y_poisoned = y.clone()
        y_poisoned[query] = 1 - y_poisoned[query]
        torch.testing.assert_close(
            expected,
            model(x, y_poisoned, query),
            msg="changing query labels changed predictions -- query leaked",
        )

    def test_label_swap_is_equivariant(self) -> None:
        model = _model()
        x, y, query = _episode()
        forward = model(x, y, query)
        swapped = model(x, 1 - y, query)
        # Swapping class names must swap the two output logits.
        torch.testing.assert_close(forward, swapped.flip(-1), atol=1e-4, rtol=1e-4)

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
        )

    def test_cells_above_the_cap_are_subsampled(self) -> None:
        model = _model(max_cells=8)
        x, y, query = _episode(cells=20)
        # Random subsampling uses PyTorch's RNG; check it runs and produces finite logits.
        torch.manual_seed(123)
        logits = model(x, y, query)
        self.assertEqual(logits.shape, (2, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_gradient_reaches_the_encoder_through_the_ridge(self) -> None:
        model = _model(num_layers=1).train()
        x, y, query = _episode()
        logits = model(x, y, query)
        loss = torch.nn.functional.cross_entropy(logits, y[query])
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"no gradient for {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"non-finite gradient for {name}")


if __name__ == "__main__":
    unittest.main()
