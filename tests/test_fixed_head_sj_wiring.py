"""RU-89: `ICF_FIXED_HEAD_SJ_WEIGHT` must be a no-op at its default (0.0).

`scripts/test_pathobench.py`'s fixed-head shape fusion (SS221-ish, next to the
BS/SH pair) only carried BS and SH into the ensemble logits; SJ's margin was
computed (`sh_variant_margins["sj"]`) but never added. This wires SJ into the
SAME fusion tuple as BS/SH, guarded by `ICF_FIXED_HEAD_SJ_WEIGHT` (default
"0.0"). Same shape as `tests/test_sh_branch.py::test_zero_weight_leaves_ensemble_untouched`:
weight 0.0 (default OR explicit) must reproduce the pre-SJ-wiring output
bit-for-bit, since the fusion loop only touches `logits` when both the weight
is non-zero AND the margin is not None.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.set_transformer_ridge import CovarianceMeanLearnablePDDCTMLPModel  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "test_pathobench_module", REPO_ROOT / "scripts" / "test_pathobench.py"
)
_pathobench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_pathobench)

DIM = 48
SKETCH = 8


class _Wrapper:
    """Mimics the `.model` attribute evaluate_trial expects (LightningModule wrapper)."""

    def __init__(self, model):
        self.model = model


def _build_model():
    torch.manual_seed(0)
    return CovarianceMeanLearnablePDDCTMLPModel(
        input_dim=DIM, token_dim=16, num_heads=1, num_layers=1, feedforward_dim=16,
        num_summary_tokens=1, max_cells=4096, dropout=0.0, covariance_sketch_dim=SKETCH,
        ridge_lambda=1.0, ridge_logit_scale=2.0, num_classes=2,
        ct_num_tokens=16, ct_cells_per_bag=64, ct_temperature=0.5, ct_eps=1e-6,
        ct_head_hidden_dims=[], dd_shrinkage=0.25, dd_eps=1e-6,
    ).eval()


def _episode(seed=0, n_context=6, n_query=2, cells=30):
    generator = torch.Generator().manual_seed(seed)
    ids = [f"s{i}" for i in range(n_context + n_query)]
    bags = {
        s: torch.randn(cells, DIM, generator=generator) for s in ids
    }
    labels = {s: i % 2 for i, s in enumerate(ids)}
    train_ids = ids[:n_context]
    test_ids = ids[n_context:]
    train_y = {s: labels[s] for s in train_ids}
    test_y = {s: labels[s] for s in test_ids}
    return bags, train_ids, test_ids, train_y, test_y


class SjFixedHeadNoOpTest(unittest.TestCase):
    """Pins that wiring SJ into the shape-fusion tuple changes nothing at weight 0."""

    def setUp(self):
        self._env_backup = dict(os.environ)
        # Enter the fixed-head generic-model path and the shape-fusion branch:
        # BS enabled (nonzero) so the fusion loop this change touches actually
        # runs; SCREEN_ONLY=0 so the fusion loop (not just the screening
        # computation) executes.
        os.environ["ICF_FIXED_HEAD"] = "1"
        os.environ["ICF_FIXED_HEAD_BS_WEIGHT"] = "1.0"
        # `sh_variant_margins` (and hence the "sj" margin) is only populated
        # inside the `if sh_weight != 0.0:` block, so SH_WEIGHT must be
        # non-zero for the SJ margin to exist at all -- it is identical
        # across the "off" and "on" runs below, so it cancels in the diff.
        os.environ["ICF_FIXED_HEAD_SH_WEIGHT"] = "1.0"
        os.environ["ICF_SHAPE_SCREEN_ONLY"] = "0"
        # None of the voting aggregations (soft_voting/hard_gated/trimmed_mean/
        # adaptive_trimmed) read BS/SH/SJ margins -- only the `else:
        # torch.softmax(logits...)` fallback does, which fires only when every
        # voting weight is zero. Zero CV/CT/DD (their fixed-head defaults are
        # non-zero) so the fusion this change touches actually reaches `scores`.
        os.environ["ICF_FIXED_HEAD_CV_WEIGHT"] = "0.0"
        os.environ["ICF_FIXED_HEAD_CT_WEIGHT"] = "0.0"
        os.environ["ICF_FIXED_HEAD_DD_WEIGHT"] = "0.0"
        for key in (
            "ICF_FIXED_HEAD_BM_WEIGHT",
            "ICF_FIXED_HEAD_BD_WEIGHT", "ICF_FIXED_HEAD_QA_WEIGHT",
            "ICF_FIXED_HEAD_DS_WEIGHT", "ICF_FIXED_HEAD_LR_WEIGHT",
            "ICF_FIXED_HEAD_DE_WEIGHT", "ICF_FIXED_HEAD_SW_WEIGHT",
            "ICF_FIXED_HEAD_RM_WEIGHT", "ICF_FIXED_HEAD_SJ_WEIGHT",
        ):
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def _run(self):
        bags, train_ids, test_ids, train_y, test_y = _episode()
        model = _build_model()
        wrapper = _Wrapper(model)
        with torch.no_grad():
            result = _pathobench.evaluate_trial(
                model=wrapper,
                projected=bags,
                train_ids=train_ids,
                test_ids=test_ids,
                train_y=train_y,
                test_y=test_y,
                context_mode="all",
                context_per_class=3,
                max_tiles=None,
                seed=0,
                device=torch.device("cpu"),
                precision="32-true",
            )
        return result["probability"]

    def test_default_sj_weight_is_a_noop(self):
        """`ICF_FIXED_HEAD_SJ_WEIGHT` unset (default "0.0") must match it being
        explicitly set to "0.0" -- the tuple entry this change adds is inert
        at weight 0 regardless of whether the margin was ever computed."""
        off = self._run()
        os.environ["ICF_FIXED_HEAD_SJ_WEIGHT"] = "0.0"
        explicit = self._run()
        self.assertTrue(torch.equal(off, explicit))

    def test_nonzero_sj_weight_moves_the_ensemble(self):
        """Sanity check that the wiring is actually live (not a silent no-op
        at every weight): a large SJ weight must change at least one logit."""
        off = self._run()
        os.environ["ICF_FIXED_HEAD_SJ_WEIGHT"] = "5.0"
        on = self._run()
        self.assertFalse(torch.equal(off, on))


if __name__ == "__main__":
    unittest.main()
