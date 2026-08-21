"""CV-2's `paired_head`: the subspace reaches the head, and labels are symmetric.

Two structural defects of `learned_head` motivated this mode (docs SS74):

  the head only ever saw 4 pooled scalars. `.square().mean(dim=-1)` collapses
  the rank axis BEFORE the MLP. Raising `subspace_rank` does still move the
  output -- different eigenvectors get picked -- but the head cannot tell WHICH
  dimension carried the signal, so a discriminative direction is averaged
  against uninformative ones. That is the shape of the v42 null result
  (rank 2 +0.0004, rank 4 -0.0008 on the SEAL 10).

  label symmetry was not enforced. Renaming the classes must flip the margin
  and change nothing else. `learned_head` broke that by 4.4e-2 on a swapped
  episode while CV-1's ridge was exact to 0.0; the old contract test missed it
  because it builds a model with `covariance_relation` disabled.

These tests pin both properties, plus the checkpoint compatibility that comes
from sharing the head across dimensions.
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

INPUT_DIM = 32


def _build(mode: str = "paired_head", rank: int = 2, seed: int = 0) -> BaseModel:
    config = merge_train_config(
        REPO_ROOT / "configs" / "archive" / "v40_v45_cvonly" / "train_v41_cvonly_K128_1536.yaml"
    )
    kwargs = {**config["model"], **config["model_kwargs"]}
    for key, value in (config.get("model_overrides") or {}).items():
        kwargs[key] = value
    kwargs.pop("model_src", None)
    kwargs["input_dim"] = INPUT_DIM
    kwargs["aggregator_covariance_sketch_dim"] = 8
    kwargs["aggregator_covariance_matrix_dim"] = 8
    relation = dict(kwargs.get("covariance_relation") or {})
    relation.update(
        {"enabled": True, "mode": mode, "granularity": "subspace",
         "subspace_rank": rank, "diagnostic_only": False}
    )
    kwargs["covariance_relation"] = relation
    accepted = inspect.signature(BaseModel.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    torch.manual_seed(seed)
    return BaseModel(**kwargs).eval()


def _episode(bags: int = 10, cells: int = 48):
    torch.manual_seed(3)
    x = torch.randn(bags, cells, INPUT_DIM)
    y = torch.tensor([0, 1] * (bags // 2))
    return x, y, torch.tensor([bags - 2, bags - 1])


def _relation_logits(model: BaseModel, x, y, query) -> torch.Tensor:
    with torch.no_grad():
        _, auxiliary = model(x, y, query, return_auxiliary=True)
    return auxiliary["covariance_relation_logits"].float()


class PairedHeadTest(unittest.TestCase):
    def test_label_swap_flips_the_margin_exactly(self) -> None:
        model = _build()
        x, y, query = _episode()
        forward = _relation_logits(model, x, y, query)
        swapped = _relation_logits(model, x, 1 - y, query)
        torch.testing.assert_close(
            forward,
            swapped.flip(-1),
            atol=1e-6,
            rtol=1e-6,
            msg="paired_head must be exactly antisymmetric under a label swap; "
            "the construction h(e0,e1,s) - h(e1,e0,s) guarantees it, so a "
            "failure here means the pairing was broken.",
        )

    def test_learned_head_is_the_asymmetry_this_replaces(self) -> None:
        """Documents the defect, so the comparison is not folklore."""
        model = _build(mode="learned_head")
        x, y, query = _episode()
        forward = _relation_logits(model, x, y, query)
        swapped = _relation_logits(model, x, 1 - y, query)
        self.assertGreater(
            float((forward - swapped.flip(-1)).abs().max()),
            1e-3,
            "learned_head became label-symmetric; if that is now true by "
            "design, this test and the motivation for paired_head are stale.",
        )

    def _head_input_widths(self, mode: str) -> dict[int, int]:
        """How many rows the head is asked to score, per subspace rank.

        This is the actual difference between the two modes. Both respond to
        `subspace_rank` in their output -- a larger rank picks different
        eigenvectors either way -- but only `paired_head` lets the rank reach
        the head as separate inputs instead of one pooled average.
        """
        widths: dict[int, int] = {}
        for rank in (1, 3):
            model = _build(mode=mode, rank=rank)
            seen: list[int] = []
            handle = model.meta_classifier.covariance_relation_head.register_forward_pre_hook(
                lambda _module, inputs, seen=seen: seen.append(inputs[0].shape[-2])
            )
            try:
                _relation_logits(model, *_episode())
            finally:
                handle.remove()
            widths[rank] = seen[0]
        return widths

    def test_the_head_sees_one_row_per_subspace_dimension(self) -> None:
        widths = self._head_input_widths("paired_head")
        self.assertEqual(
            widths,
            {1: 1, 3: 3},
            "paired_head must score each subspace dimension separately; "
            f"got {widths}",
        )

    def test_learned_head_sees_the_same_width_at_every_rank(self) -> None:
        """The defect this mode replaces, pinned so it stays documented."""
        widths = self._head_input_widths("learned_head")
        self.assertEqual(
            widths[1],
            widths[3],
            "learned_head stopped pooling the rank axis; the motivation for "
            f"paired_head would then be stale (got {widths})",
        )

    def test_head_shapes_are_independent_of_rank(self) -> None:
        """Shared across dimensions, so a rank sweep keeps checkpoints valid."""
        shapes = {
            rank: {
                name: tuple(value.shape)
                for name, value in _build(rank=rank).state_dict().items()
            }
            for rank in (1, 2, 4)
        }
        self.assertEqual(shapes[1], shapes[2])
        self.assertEqual(shapes[1], shapes[4])
        _build(rank=4).load_state_dict(_build(rank=1).state_dict(), strict=True)

    def test_dense_and_ragged_paths_agree(self) -> None:
        model = _build()
        x, y, query = _episode()
        with torch.no_grad():
            ragged = model(list(x.unbind(0)), y, query).float()
            dense = model.forward_episode_batch(
                x.unsqueeze(0), y.unsqueeze(0), query.unsqueeze(0)
            )[0].float()
        torch.testing.assert_close(ragged, dense, atol=1e-4, rtol=1e-4)

    def test_head_receives_gradient(self) -> None:
        model = _build().train()
        x, y, query = _episode()
        logits = model(x, y, query)
        torch.nn.functional.cross_entropy(logits.float(), y[query]).backward()
        head = model.meta_classifier.covariance_relation_head
        # The output bias is the one exception, and it is structural: an
        # additive constant is identical in h(e0,e1,s) and h(e1,e0,s), so it
        # cancels in the difference. It cannot receive gradient and cannot
        # affect the margin -- that is the price of exact antisymmetry.
        output_bias = "2.bias"
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, f"head.{name} got no gradient")
            magnitude = float(parameter.grad.abs().max())
            if name == output_bias:
                self.assertEqual(
                    magnitude, 0.0, "the output bias must cancel in the pairing"
                )
                continue
            self.assertGreater(
                magnitude, 0.0, f"head.{name} received an all-zero gradient"
            )

    def test_multiclass_is_rejected_rather_than_silently_wrong(self) -> None:
        with self.assertRaises(ValueError):
            _build_multiclass()


def _build_multiclass() -> BaseModel:
    model = _build.__wrapped__ if hasattr(_build, "__wrapped__") else None
    config = merge_train_config(
        REPO_ROOT / "configs" / "archive" / "v40_v45_cvonly" / "train_v41_cvonly_K128_1536.yaml"
    )
    kwargs = {**config["model"], **config["model_kwargs"]}
    for key, value in (config.get("model_overrides") or {}).items():
        kwargs[key] = value
    kwargs.pop("model_src", None)
    kwargs["input_dim"] = INPUT_DIM
    kwargs["num_classes"] = 3
    relation = dict(kwargs.get("covariance_relation") or {})
    relation.update({"enabled": True, "mode": "paired_head"})
    kwargs["covariance_relation"] = relation
    accepted = inspect.signature(BaseModel.__init__).parameters
    return BaseModel(**{k: v for k, v in kwargs.items() if k in accepted})


if __name__ == "__main__":
    unittest.main()
