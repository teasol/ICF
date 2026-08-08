"""Pin the ridge-ablation contract (docs SS65).

Three independent closed-form ridge solves feed the final logits:

    G-2  global ridge      -> global_shape_logits (residual: set/cross attention)
    P-2  abundance ridge   -> population_logits   (residual: population attention)
    CV-1 covariance ridge  -> covariance_logits   (CV-2 relation branch is separate)

The hypothesis under test is that these closed-form solves crowd out the learned
branches. Each flag removes exactly one of them, leaving that branch's learned
residual in place, so an arm isolates one ridge rather than one whole branch.

Every ridge site is duplicated across the dense (training) and ragged (eval)
code paths -- `test_dense_and_ragged_paths_agree` is the guard that a change
touching only one copy fails here instead of silently training one behaviour and
evaluating another (the exact accident SS62-7 called out for v36 Q1).
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

FLAGS = (
    "meta_enable_global_ridge",
    "meta_enable_abundance_ridge",
    "meta_enable_covariance_ridge",
)

# flag -> the auxiliary key that must be identically zero when it is off
ZEROED = {
    "meta_enable_abundance_ridge": "abundance_ridge_logits",
    "meta_enable_covariance_ridge": "covariance_ridge_logits",
}


def _build(**overrides) -> BaseModel:
    """Build the v34/v35 architecture at a small input dim."""
    config = merge_train_config(
        REPO_ROOT / "configs" / "train_v34_phase0_largectx_1536.yaml"
    )
    kwargs = dict(config["model"])
    kwargs.update(config["model_kwargs"])
    kwargs.pop("model_src", None)
    kwargs["input_dim"] = INPUT_DIM
    kwargs["aggregator_covariance_sketch_dim"] = 16
    kwargs["aggregator_slot_affinity_dim"] = INPUT_DIM
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


class TestRidgeAblation(unittest.TestCase):
    def test_default_keeps_every_ridge_on(self) -> None:
        """All three default to on -- no silent behaviour change for old configs."""
        model = _build()
        meta = model.meta_classifier
        self.assertFalse(meta.global_shape_classifier.force_ridge_logits_zero)
        self.assertFalse(meta.force_abundance_ridge_zero)
        self.assertFalse(meta.force_covariance_ridge_zero)

    def test_default_matches_a_model_built_without_the_flags(self) -> None:
        """Passing the flags explicitly as True is a no-op on the logits."""
        x, y, mask = _episode()
        with torch.no_grad():
            base = _build()(x, y, mask)
            explicit = _build(**{f: True for f in FLAGS})(x, y, mask)
        self.assertTrue(torch.equal(base, explicit))

    def test_each_flag_changes_the_logits(self) -> None:
        """A disabled ridge must actually move the output, or the arm is a no-op."""
        x, y, mask = _episode()
        with torch.no_grad():
            reference = _build()(x, y, mask).float()
        for flag in FLAGS:
            with self.subTest(flag=flag):
                with torch.no_grad():
                    ablated = _build(**{flag: False})(x, y, mask).float()
                delta = (reference - ablated).abs().max()
                self.assertGreater(
                    float(delta),
                    1e-5,
                    f"{flag}=False did not change the logits -- the ablation is "
                    "not wired to the branch it claims to remove.",
                )

    def test_disabled_ridge_terms_are_exactly_zero(self) -> None:
        """The ablated term is zero, not merely small."""
        x, y, mask = _episode()
        for flag, key in ZEROED.items():
            with self.subTest(flag=flag):
                model = _build(**{flag: False})
                with torch.no_grad():
                    _, auxiliary = model(x, y, mask, return_auxiliary=True)
                term = auxiliary[key]
                self.assertTrue(
                    bool((term == 0).all()),
                    f"{key} is not identically zero with {flag}=False.",
                )

    def test_disabled_global_ridge_leaves_only_the_attention_residual(self) -> None:
        """G-2 off must not take the learned G-3 residual down with it."""
        model = _build(meta_enable_global_ridge=False)
        classifier = model.meta_classifier.global_shape_classifier
        context = torch.randn(6, model.meta_classifier.token_dim)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        query = torch.randn(3, model.meta_classifier.token_dim)
        with torch.no_grad():
            logits, auxiliary = classifier(context, labels, query, return_auxiliary=True)
        self.assertTrue(bool((auxiliary["ridge_logits"] == 0).all()))
        self.assertGreater(
            float(auxiliary["attention_logits"].abs().max()),
            0.0,
            "the attention residual vanished too -- this ablates the whole "
            "branch instead of just the ridge.",
        )
        self.assertTrue(torch.isfinite(logits).all())

    def test_dense_and_ragged_paths_agree(self) -> None:
        """Both duplicated implementations must ablate identically."""
        x, y, mask = _episode()
        for flag in FLAGS:
            with self.subTest(flag=flag):
                model = _build(**{flag: False})
                with torch.no_grad():
                    ragged = model(list(x.unbind(0)), y, mask)
                    dense = model.forward_episode_batch(
                        x.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0)
                    )
                delta = (ragged.float() - dense[0].float()).abs().max()
                self.assertLess(
                    float(delta),
                    1e-4,
                    f"dense/ragged disagree with {flag}=False "
                    f"(||delta||inf={float(delta):.3e}); the ridge guard is "
                    "duplicated across both paths and one copy was missed.",
                )

    def test_ablation_adds_no_parameters_and_keeps_checkpoints_loadable(self) -> None:
        """Shape-preserving both ways, so ckpts stay strict-loadable."""
        full = _build()
        shapes = {k: tuple(v.shape) for k, v in full.state_dict().items()}
        for flag in FLAGS:
            with self.subTest(flag=flag):
                ablated = _build(**{flag: False})
                self.assertEqual(
                    shapes,
                    {k: tuple(v.shape) for k, v in ablated.state_dict().items()},
                )
                self.assertEqual(
                    sum(p.numel() for p in full.parameters()),
                    sum(p.numel() for p in ablated.parameters()),
                )
                ablated.load_state_dict(full.state_dict(), strict=True)

    def test_ablated_ridge_parameters_get_no_gradient(self) -> None:
        """The documented cost of the ablation: those parameters stay at init.

        This is why an ablated checkpoint must be evaluated with the same flag
        off -- re-enabling the ridge would inject a never-trained branch.
        """
        x, y, mask = _episode()
        model = _build(meta_enable_global_ridge=False).train()
        logits = model(x, y, mask)
        logits.float().square().mean().backward()
        ridge = model.meta_classifier.global_shape_classifier
        for name, parameter in ridge.ridge_projection.named_parameters():
            self.assertIsNone(
                parameter.grad,
                f"ridge_projection.{name} received a gradient with the global "
                "ridge disabled -- the solve was not actually skipped.",
            )


if __name__ == "__main__":
    unittest.main()
