"""Pin the sketch-geometry knobs added for the K sweep (docs SS69).

Two things were hardcoded and are now configurable:
  * the frequency-ladder slopes (0.019, 0.011)
  * how many of P's columns CV-2 consumes (32, via a default argument)

Both default to the historical values, so every existing config and checkpoint
is unaffected -- that is what `test_defaults_reproduce_history` guards.
"""
from __future__ import annotations
import inspect, math, sys, unittest
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.models.baseline import BaseModel  # noqa: E402
from src.utils.utils import merge_train_config  # noqa: E402

INPUT_DIM = 96


def _build(**over):
    cfg = merge_train_config(REPO_ROOT / "configs" / "train_v34_phase0_largectx_1536.yaml")
    kw = {**cfg["model"], **cfg["model_kwargs"]}
    kw.pop("model_src", None)
    kw["input_dim"] = INPUT_DIM
    kw["aggregator_covariance_sketch_dim"] = 16
    kw["aggregator_slot_affinity_dim"] = INPUT_DIM
    kw.update(over)
    acc = inspect.signature(BaseModel.__init__).parameters
    torch.manual_seed(0)
    return BaseModel(**{k: v for k, v in kw.items() if k in acc})


class TestSketchKnobs(unittest.TestCase):
    def test_defaults_reproduce_history(self) -> None:
        """Untouched configs must keep the hardcoded slopes and the 32-column CV-2."""
        agg = _build().aggregator
        self.assertEqual(agg.covariance_slopes, (0.019, 0.011))
        self.assertEqual(agg.covariance_matrix_dim, 32)

    def test_matrix_dim_none_ties_cv2_to_K(self) -> None:
        agg = _build(aggregator_covariance_matrix_dim=None).aggregator
        self.assertEqual(agg.covariance_matrix_dim, agg.covariance_sketch_dim)
        x = torch.randn(3, 64, INPUT_DIM)
        centered = x - x.mean(dim=-2, keepdim=True)
        self.assertEqual(
            agg._projected_covariance_matrix(centered).shape[-1],
            agg.covariance_sketch_dim,
        )

    def test_matrix_dim_default_caps_cv2_at_32(self) -> None:
        """The historical behaviour: CV-2 stays at 32 however large K gets."""
        agg = _build(aggregator_covariance_sketch_dim=64).aggregator
        x = torch.randn(3, 64, INPUT_DIM)
        centered = x - x.mean(dim=-2, keepdim=True)
        self.assertEqual(agg._projected_covariance_matrix(centered).shape[-1], 32)

    def test_slopes_change_the_basis(self) -> None:
        base = _build().aggregator._covariance_projection
        other = _build(
            aggregator_covariance_slopes=[0.85 * math.pi / 16, 0.733 * 0.85 * math.pi / 16]
        ).aggregator._covariance_projection
        self.assertEqual(base.shape, other.shape)
        overlap = float(torch.linalg.svdvals(base.T.double() @ other.double()).mean())
        self.assertLess(overlap, 0.95, "slopes did not actually move the subspace")

    def test_basis_stays_orthonormal(self) -> None:
        for slopes in (None, [0.85 * math.pi / 16, 0.733 * 0.85 * math.pi / 16]):
            with self.subTest(slopes=slopes):
                P = _build(aggregator_covariance_slopes=slopes).aggregator._covariance_projection
                gram = P.T.double() @ P.double()
                delta = (gram - torch.eye(gram.shape[0], dtype=gram.dtype)).abs().max()
                # The basis is stored in float32, so QR orthonormality holds to
                # ~1e-7, not to double precision. Measured 2.7e-7.
                self.assertLess(float(delta), 1e-5)


if __name__ == "__main__":
    unittest.main()


class TestMarginActivation(unittest.TestCase):
    """Pin the CV-2 margin activation (docs SS72).

    `tanh` (the historical default from `5869535`) SATURATES at init on the
    learned_head margin -- relation logits are exactly +-0.5, so
    d(tanh)/dx = 1 - tanh^2 = 0 and the head receives gradient 0.0, not None.
    In CV-only that leaves the whole 43.20M-parameter model with exactly THREE
    parameters that train, all of them scales. `identity` unblocks it.
    """

    def _grad_count(self, **relation):
        cfg = merge_train_config(
            REPO_ROOT / "configs" / "train_v34_phase0_largectx_1536.yaml"
        )
        kw = {**cfg["model"], **cfg["model_kwargs"]}
        kw.pop("model_src", None)
        kw["input_dim"] = INPUT_DIM
        kw["aggregator_covariance_sketch_dim"] = 16
        kw["aggregator_slot_affinity_dim"] = INPUT_DIM
        kw["meta_covariance_only"] = True
        kw["covariance_relation"] = {**kw["covariance_relation"], **relation}
        acc = inspect.signature(BaseModel.__init__).parameters
        torch.manual_seed(0)
        model = BaseModel(**{k: v for k, v in kw.items() if k in acc}).train()
        x = torch.randn(1, 8, 120, INPUT_DIM)
        y = torch.tensor([[0, 1] * 4])
        q = torch.tensor([[2, 5]])
        out, auxiliary = model.forward_episode_batch(
            x, y, q, return_auxiliary=True
        )
        out.float().square().mean().backward()
        live = sum(
            p.numel()
            for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        # The CV-2 term alone -- `out` also carries CV-1, so it is not bounded.
        return live, float(auxiliary["covariance_relation_logits"].abs().max())

    def test_tanh_bounds_the_output_and_identity_does_not(self) -> None:
        """The mechanism. Saturation itself is SCALE-dependent -- it was measured
        on the real config (1536-d, K=128) where relation logits sit at exactly
        +-0.5; this toy scale (96-d, K=16) does not necessarily saturate, so the
        test pins the bound rather than the gradient count."""
        _, tanh_max = self._grad_count()
        _, ident_max = self._grad_count(
            margin_activation="identity", margin_temperature=1.0
        )
        self.assertLessEqual(tanh_max, 0.5 + 1e-4)
        self.assertGreater(ident_max, tanh_max)

    def test_identity_gives_at_least_as_much_gradient_as_tanh(self) -> None:
        """tanh can only shrink gradient (|d tanh/dx| <= 1), never grow it."""
        tanh_live, _ = self._grad_count()
        ident_live, _ = self._grad_count(margin_activation="identity")
        self.assertGreaterEqual(ident_live, tanh_live)
        self.assertGreater(ident_live, 100)

    def test_temperature_scales_the_margin(self) -> None:
        """Larger temperature must shrink the CV-2 contribution."""
        _, small = self._grad_count(
            margin_activation="identity", margin_temperature=1.0
        )
        _, large = self._grad_count(
            margin_activation="identity", margin_temperature=100.0
        )
        self.assertLess(large, small)

    def test_temperature_parameter_only_exists_in_identity_mode(self) -> None:
        """Creating it unconditionally would break strict loads of old ckpts."""
        cfg = merge_train_config(
            REPO_ROOT / "configs" / "train_v34_phase0_largectx_1536.yaml"
        )
        kw = {**cfg["model"], **cfg["model_kwargs"]}
        kw.pop("model_src", None)
        kw["input_dim"] = INPUT_DIM
        kw["aggregator_covariance_sketch_dim"] = 16
        kw["aggregator_slot_affinity_dim"] = INPUT_DIM
        acc = inspect.signature(BaseModel.__init__).parameters
        names = lambda m: [n for n in m.state_dict() if "log_temperature" in n]
        torch.manual_seed(0)
        base = BaseModel(**{k: v for k, v in kw.items() if k in acc})
        kw["covariance_relation"] = {
            **kw["covariance_relation"], "margin_activation": "identity",
        }
        torch.manual_seed(0)
        ident = BaseModel(**{k: v for k, v in kw.items() if k in acc})
        self.assertEqual(names(base), [])
        self.assertEqual(len(names(ident)), 1)

    def test_invalid_activation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._grad_count(margin_activation="relu")
