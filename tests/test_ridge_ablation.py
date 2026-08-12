"""Pin the CV-1 ridge-ablation contract and the two-term output shape.

Before the prune (docs SS73) this file covered three closed-form ridge solves --
G-2 global, P-2 abundance, CV-1 covariance -- and the CV-only arm that switched
the other five branches off. Those branches are gone from the source, so the
tests that exercised them are gone too; git holds them at `8caa96c`.

What still has meaning, and why:

  CV-1 ablation      SS66 measured that removing the covariance ridge collapses
                     training, twice, with and without numerical stabilisation.
                     That is the one ridge flag left, and it must stay wired.
  dense == ragged    Every site is duplicated across the training (dense) and
                     eval (ragged) paths. This is the guard that a change
                     touching one copy fails here instead of silently training
                     one behaviour and evaluating another (SS62-7).
  two-term output    final == cov_res*CV-1 + cov_rel_res*CV-2 and nothing else.

Numerical equivalence with the pre-prune model is pinned separately, by
`tests/test_cvonly_golden.py`.
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


def _build(**overrides) -> BaseModel:
    config = merge_train_config(
        REPO_ROOT / "configs" / "archive" / "v40_v45_cvonly" / "train_v41_cvonly_K128_1536.yaml"
    )
    kwargs = dict(config["model"])
    kwargs.update(config["model_kwargs"])
    kwargs.pop("model_src", None)
    kwargs["input_dim"] = INPUT_DIM
    kwargs["aggregator_covariance_sketch_dim"] = 16
    kwargs.update(overrides)
    accepted = inspect.signature(BaseModel.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    torch.manual_seed(0)
    return BaseModel(**kwargs).eval()


def _episode(n_bags: int = 8, n_cells: int = 120):
    torch.manual_seed(1)
    x = torch.randn(n_bags, n_cells, INPUT_DIM)
    y = torch.tensor([0, 1] * (n_bags // 2))
    return x, y, torch.tensor([2, 5])


class TestCovarianceRidgeAblation(unittest.TestCase):
    def test_default_keeps_the_ridge_on(self) -> None:
        self.assertFalse(_build().meta_classifier.force_covariance_ridge_zero)

    def test_disabled_ridge_term_is_exactly_zero(self) -> None:
        """The ablated term is zero, not merely small."""
        x, y, mask = _episode()
        model = _build(meta_enable_covariance_ridge=False)
        with torch.no_grad():
            _, auxiliary = model(x, y, mask, return_auxiliary=True)
        term = auxiliary["covariance_ridge_logits"]
        self.assertTrue(
            bool((term == 0).all()),
            "covariance_ridge_logits is not identically zero with "
            "meta_enable_covariance_ridge=False.",
        )

    def test_the_flag_changes_the_output(self) -> None:
        """A flag that removes the dominant branch must move the logits."""
        x, y, mask = _episode()
        with torch.no_grad():
            on = _build()(x, y, mask).float()
            off = _build(meta_enable_covariance_ridge=False)(x, y, mask).float()
        self.assertGreater(
            float((on - off).abs().max()),
            1e-5,
            "meta_enable_covariance_ridge=False did not change the logits -- "
            "the ablation is not wired to the branch it claims to remove.",
        )

    def test_ablation_adds_no_parameters_and_keeps_checkpoints_loadable(self) -> None:
        """Shape-preserving both ways, so ckpts stay strict-loadable."""
        full = _build()
        ablated = _build(meta_enable_covariance_ridge=False)
        self.assertEqual(
            {k: tuple(v.shape) for k, v in full.state_dict().items()},
            {k: tuple(v.shape) for k, v in ablated.state_dict().items()},
        )
        ablated.load_state_dict(full.state_dict(), strict=True)


class TestCovarianceOnlyOutput(unittest.TestCase):
    def test_final_logits_equal_the_two_covariance_terms(self) -> None:
        """final == cov_res*CV-1 + cov_rel_res*CV-2, nothing else."""
        model = _build()
        x, y, mask = _episode()
        with torch.no_grad():
            logits, auxiliary = model(x, y, mask, return_auxiliary=True)
        meta = model.meta_classifier
        expected = (
            torch.sigmoid(meta.covariance_residual_logit)
            * auxiliary["covariance_logits"]
            + meta.covariance_relation_residual_scale
            * auxiliary["covariance_relation_logits"]
        )
        delta = (logits.float() - expected.float()).abs().max()
        self.assertLess(
            float(delta),
            1e-4,
            "output is not exactly the two covariance terms "
            f"(||delta||inf={float(delta):.3e}) -- some other term leaks in.",
        )

    def test_dense_and_ragged_paths_agree(self) -> None:
        model = _build()
        x, y, mask = _episode()
        with torch.no_grad():
            ragged = model(list(x.unbind(0)), y, mask)
            dense = model.forward_episode_batch(
                x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0)
            )
        delta = (ragged.float() - dense[0].float()).abs().max()
        self.assertLess(float(delta), 1e-4, f"dense/ragged disagree ({delta:.3e})")

    def test_representation_carries_only_the_two_covariance_keys(self) -> None:
        """Dead keys are ABSENT, not zero-filled (docs SS68).

        A stray consumer then raises KeyError at its own line instead of
        silently averaging zeros into a live branch.
        """
        model = _build()
        x, _, _ = _episode()
        representation = model.aggregator(
            x, torch.ones(x.shape[0], dtype=torch.bool)
        )
        self.assertEqual(
            set(representation), {"covariance_sketch", "covariance_matrix"}
        )

    def test_validation_rejects_extra_keys(self) -> None:
        model = _build()
        x, _, _ = _episode()
        representation = model.aggregator(
            x, torch.ones(x.shape[0], dtype=torch.bool)
        )
        with self.assertRaises(ValueError):
            model.meta_classifier._validate_representation(
                {**representation, "slots": torch.zeros(1)}, "context"
            )
        with self.assertRaises(ValueError):
            model.meta_classifier._validate_representation(
                {"covariance_sketch": representation["covariance_sketch"]},
                "context",
            )


if __name__ == "__main__":
    unittest.main()
