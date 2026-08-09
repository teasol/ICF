"""An equal-shape training episode must take the vectorised forward.

At `episode_batch_size: 1` the training step used to call the ragged forward --
a Python loop over bags, written for evaluation, where bag lengths differ. On
equal-shape synthetic episodes that is hundreds of kernel launches for a result
the dense path produces in one pass: measured 30.48 ms vs 8.50 ms per
fwd+bwd+step on 64 bags x 19k cells, agreeing to 3.3e-07.

Two things are pinned here, because the regression would be silent otherwise:
the loss must not change, and the dense entry point must actually be the one
that runs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline import BaseModel  # noqa: E402
from src.modules.model_interface import ModelInterface  # noqa: E402
from src.utils.utils import merge_train_config  # noqa: E402

INPUT_DIM = 32


def _interface() -> ModelInterface:
    config = merge_train_config(
        REPO_ROOT / "configs" / "train_v41_cvonly_K128_1536.yaml"
    )
    hparams = {**config["model"], **config.get("model_kwargs", {})}
    for key, value in (config.get("model_overrides") or {}).items():
        hparams[key] = value
    hparams["input_dim"] = INPUT_DIM
    hparams["aggregator_covariance_sketch_dim"] = 8
    hparams["aggregator_covariance_matrix_dim"] = 8
    torch.manual_seed(0)
    return ModelInterface(**hparams).eval()


def _episode():
    torch.manual_seed(3)
    x = torch.randn(8, 40, INPUT_DIM)
    y = torch.tensor([0, 1] * 4)
    return x, y, torch.tensor([6, 7])


class TrainingPathTest(unittest.TestCase):
    def test_single_episode_goes_through_forward_episode_batch(self) -> None:
        interface = _interface()
        x, y, mask_index = _episode()
        with patch.object(
            BaseModel,
            "forward_episode_batch",
            side_effect=BaseModel.forward_episode_batch,
            autospec=True,
        ) as dense:
            interface._episode_losses(x, y, mask_index)
        self.assertEqual(
            dense.call_count,
            1,
            "a single equal-shape episode did not take the dense path; the "
            "ragged forward is the per-bag Python loop meant for evaluation.",
        )

    def test_dense_and_ragged_losses_agree(self) -> None:
        interface = _interface()
        x, y, mask_index = _episode()
        with torch.no_grad():
            dense_loss, dense_terms = interface._episode_losses(x, y, mask_index)
            ragged_logits, ragged_auxiliary = interface.model(
                x, y, mask_index, return_auxiliary=True
            )
            ragged_loss, ragged_terms = interface._losses_from_output(
                ragged_logits, ragged_auxiliary, y[mask_index]
            )
        torch.testing.assert_close(
            dense_loss, ragged_loss, atol=1e-4, rtol=1e-4,
            msg="routing training through the dense path changed the loss",
        )
        self.assertEqual(set(dense_terms), set(ragged_terms))
        for name in dense_terms:
            torch.testing.assert_close(
                dense_terms[name].float(),
                ragged_terms[name].float(),
                atol=1e-4,
                rtol=1e-4,
                msg=f"logged term '{name}' changed with the dense path",
            )

    def test_ragged_episode_still_uses_the_ragged_forward(self) -> None:
        """A list of unequal bags has no dense equivalent."""
        interface = _interface()
        torch.manual_seed(4)
        bags = [torch.randn(6 + index, INPUT_DIM) for index in range(8)]
        y = torch.tensor([0, 1] * 4)
        mask_index = torch.tensor([6, 7])
        with patch.object(
            BaseModel,
            "forward_episode_batch",
            side_effect=BaseModel.forward_episode_batch,
            autospec=True,
        ) as dense:
            interface._episode_losses(bags, y, mask_index)
        self.assertEqual(dense.call_count, 0)


if __name__ == "__main__":
    unittest.main()
