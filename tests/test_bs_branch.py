"""BS branch: log total projected variance of a slide (S1-S4 wiring, this RU).

Covers the completion conditions of the BS/SH/SJ wiring task:
  (a) bit-invariance pin: weight_bs=weight_sh=weight_sj=0 must reproduce the
      6 aggregation functions' pre-BS output bit-for-bit.
  (b) each of the 5 voting aggregations actually moves when BS/SH/SJ is
      enabled (5 aggregations x 3 branches).
  (c) context_loo_stacking does not silently drop BS/SH/SJ (D-039 regression).
  (d) scripts/test_pathobench.py's aggregation and src/models/aggregations/
      voting.py agree on the same (weight, margin) inputs.
  (e) bs_slide_features' bag_stats_cache path matches the plain from-bag path.
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

from src.models.branches.bs import bs_features, bs_slide_features  # noqa: E402
from src.models.config import TrainingFreeConfig  # noqa: E402
from src.models.training_free import TrainingFreeClassifier  # noqa: E402
from src.models.aggregations.voting import (  # noqa: E402
    linear_aggregation,
    soft_voting,
    trimmed_mean,
    hard_gated,
    adaptive_trimmed,
    context_loo_stacking,
)

_SPEC = importlib.util.spec_from_file_location(
    "test_pathobench_module_bs", REPO_ROOT / "scripts" / "test_pathobench.py"
)
_pathobench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_pathobench)

from src.models.set_transformer_ridge import CovarianceMeanLearnablePDDCTMLPModel  # noqa: E402


# ---- (e) bs_slide_features: cache path vs plain path ------------------------

class TestBsSlideFeatures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.dim = 48
        self.bag = torch.randn(90, self.dim)
        self.basis = torch.linalg.qr(torch.randn(self.dim, 32))[0]

    def test_shape(self):
        feat = bs_slide_features(self.bag, self.basis, 16)
        self.assertEqual(feat.shape, (1,))

    def test_cache_path_matches_plain_path(self):
        """The bag_stats_cache (scatter) fast path must be numerically
        identical to computing straight off the bag (S1: both branches
        preserved, must agree bit-for-bit up to fp64/fp32 rounding)."""
        dim = 16
        plain = bs_slide_features(self.bag, self.basis, dim)

        n = self.bag.shape[0]
        mean = self.bag.mean(dim=0)
        centered = (self.bag - mean).double()
        scatter = centered.T @ centered
        cached = bs_slide_features(self.bag, self.basis, dim, bag_stats=(n, mean, scatter))

        self.assertLess((plain - cached).abs().max().item(), 1e-4)

    def test_scale_changes_the_feature(self):
        """BS is pure scale (log total variance) -- unlike SH/SJ it must NOT be
        scale-invariant, or it would restate BD's normalised entropy instead of
        occupying the axis BD discards."""
        base = bs_slide_features(self.bag, self.basis, 16)
        scaled = bs_slide_features(self.bag * 2.0, self.basis, 16)
        self.assertGreater((base - scaled).abs().max().item(), 0.1)


# ---- training-free pipeline: weight_bs wiring --------------------------------

class TestBsBranchEnsemble(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.dim = 64
        self.n_ctx, self.n_qry = 30, 12
        self.ctx_bags = [torch.randn(torch.randint(60, 120, (1,)).item(), self.dim) for _ in range(self.n_ctx)]
        self.ctx_labels = torch.tensor([i % 2 for i in range(self.n_ctx)], dtype=torch.long)
        self.qry_bags = [torch.randn(torch.randint(60, 120, (1,)).item(), self.dim) for _ in range(self.n_qry)]

    def _config(self, **kw):
        base = dict(sketch_dim=32, weight_cv=1.0, weight_bm=1.0, weight_bd=1.0,
                    weight_qa=1.0, weight_ds=1.0, weight_ct=0.0, weight_dd=0.0,
                    aggregation="trimmed_mean")
        base.update(kw)
        return TrainingFreeConfig(**base)

    def test_default_config_has_no_bs_fields_set(self):
        cfg = TrainingFreeConfig()
        self.assertEqual(cfg.weight_bs, 0.0)
        self.assertEqual(cfg.bs_dim, 256)
        self.assertEqual(cfg.bs_lambda, 1.0)

    def test_zero_weight_leaves_ensemble_untouched(self):
        off = TrainingFreeClassifier(self._config()).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        explicit = TrainingFreeClassifier(self._config(weight_bs=0.0, bs_dim=32)).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertLess((off - explicit).abs().max().item(), 1e-6)

    def test_bs_changes_the_ensemble_when_enabled(self):
        off = TrainingFreeClassifier(self._config()).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        on = TrainingFreeClassifier(self._config(weight_bs=1.0, bs_dim=32)).margins(
            self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertGreater((off - on).abs().max().item(), 1e-6)

    def test_bs_forward_and_antisymmetry(self):
        clf = TrainingFreeClassifier(self._config(weight_bs=1.0, bs_dim=32))
        m = clf.margins(self.ctx_bags, self.ctx_labels, self.qry_bags)
        self.assertEqual(m.shape, (self.n_qry,))
        self.assertTrue(torch.isfinite(m).all())
        inv = clf.margins(self.ctx_bags, 1 - self.ctx_labels, self.qry_bags)
        self.assertLess((m + inv).abs().max().item(), 1e-4)


# ---- (a) bit-invariance pin: all 6 aggregations, weight_bs/sh/sj == 0 --------

class TestAggregationBitInvariancePin(unittest.TestCase):
    """Hardcoded reference values computed (seed=123, n=6) with the pre-BS
    call signature (no m_bs argument at all). Passing m_bs explicitly at
    weight_bs=0.0 -- default or explicit -- must reproduce them exactly."""

    def setUp(self):
        torch.manual_seed(123)
        n = 6
        self.cv = torch.randn(n)
        self.m_cv = torch.randn(n)
        self.m_dd = torch.randn(n)
        self.m_ct = torch.randn(n)
        self.m_bm = torch.randn(n)
        self.m_bd = torch.randn(n)
        self.m_qa = torch.randn(n)
        self.m_ds = torch.randn(n)
        self.m_lr = torch.randn(n)
        self.m_de = torch.randn(n)
        self.m_sw = torch.randn(n)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1])
        self.loo_cv = torch.randn(n)
        self.loo_bm = torch.randn(n)
        self.loo_bd = torch.randn(n)
        self.loo_qa = torch.randn(n)
        self.loo_ds = torch.randn(n)
        self.loo_de = torch.randn(n)
        self.loo_sw = torch.randn(n)
        self.cfg = TrainingFreeConfig(
            weight_cv=1.0, weight_dd=1.0, weight_ct=1.0, weight_bm=1.0,
            weight_bd=1.0, weight_qa=1.0, weight_ds=1.0, weight_lr=1.0,
            weight_de=1.0, weight_sw=1.0, aggregation="soft_voting",
        )
        # A non-None, non-zero-weight BS margin: it must have ZERO effect
        # below, since weight_bs stays at its default 0.0. A dummy margin
        # that WOULD move the ensemble if it leaked through is the point.
        self.decoy_bs = torch.randn(n) * 100.0

        self.expected = {
            "linear": [1.0509684085845947, -5.537031173706055, 3.252016305923462,
                       1.9871536493301392, 3.417811870574951, -1.2693045139312744],
            "soft_voting": [0.07352205365896225, -0.39189133048057556, 0.31428390741348267,
                            0.1545349657535553, 0.3202900290489197, -0.11098966002464294],
            "trimmed_mean": [0.019831359386444092, -0.3939433991909027, 0.39520499110221863,
                             0.15881791710853577, 0.36477476358413696, -0.1096535325050354],
            "hard_gated": [0.07028209418058395, -0.39189133048057556, 0.35312989354133606,
                          0.18392784893512726, 0.4109418988227844, -0.10978403687477112],
            "adaptive_trimmed": [0.07352205365896225, -0.39189133048057556, 0.31428390741348267,
                                0.1545349657535553, 0.36477503180503845, -0.11098966002464294],
            "context_loo": [-0.10718153417110443, -0.7079964876174927, 0.24563433229923248,
                            -0.14570969343185425, 0.5855027437210083, 0.3569691479206085],
        }

    def _args(self):
        return (self.cfg, self.cv, self.m_cv, self.m_dd, self.m_ct, self.m_bm, self.m_bd,
                self.m_qa, self.m_ds, self.m_lr, self.m_de, self.m_sw)

    def _assert_pinned(self, name, actual):
        expected = torch.tensor(self.expected[name])
        self.assertLess((actual - expected).abs().max().item(), 1e-5,
                         f"{name} drifted from its pre-BS reference value")

    def test_linear_aggregation_pin(self):
        self._assert_pinned("linear", linear_aggregation(*self._args()))
        self._assert_pinned("linear", linear_aggregation(*self._args(), m_bs=self.decoy_bs))
        self._assert_pinned("linear", linear_aggregation(*self._args(), m_bs=self.decoy_bs, m_sh=self.decoy_bs, m_sj=self.decoy_bs))

    def test_soft_voting_pin(self):
        self._assert_pinned("soft_voting", soft_voting(*self._args()))
        self._assert_pinned("soft_voting", soft_voting(*self._args(), m_bs=self.decoy_bs))

    def test_trimmed_mean_pin(self):
        self._assert_pinned("trimmed_mean", trimmed_mean(*self._args()))
        self._assert_pinned("trimmed_mean", trimmed_mean(*self._args(), m_bs=self.decoy_bs))

    def test_hard_gated_pin(self):
        self._assert_pinned("hard_gated", hard_gated(*self._args()))
        self._assert_pinned("hard_gated", hard_gated(*self._args(), m_bs=self.decoy_bs))

    def test_adaptive_trimmed_pin(self):
        self._assert_pinned("adaptive_trimmed", adaptive_trimmed(*self._args()))
        self._assert_pinned("adaptive_trimmed", adaptive_trimmed(*self._args(), m_bs=self.decoy_bs))

    def test_context_loo_stacking_pin(self):
        out = context_loo_stacking(
            self.cfg, self.cv, self.labels,
            self.m_cv, self.loo_cv, self.m_dd, None, self.m_ct, None,
            self.m_bm, self.loo_bm, self.m_bd, self.loo_bd, self.m_qa, self.loo_qa,
            self.m_ds, self.loo_ds, self.m_de, self.loo_de, self.m_sw, self.loo_sw,
        )
        self._assert_pinned("context_loo", out)
        out_decoy = context_loo_stacking(
            self.cfg, self.cv, self.labels,
            self.m_cv, self.loo_cv, self.m_dd, None, self.m_ct, None,
            self.m_bm, self.loo_bm, self.m_bd, self.loo_bd, self.m_qa, self.loo_qa,
            self.m_ds, self.loo_ds, self.m_de, self.loo_de, self.m_sw, self.loo_sw,
            m_bs=self.decoy_bs, loo_bs=self.decoy_bs,
        )
        self._assert_pinned("context_loo", out_decoy)


# ---- (b) each aggregation actually picks up BS/SH/SJ when weighted ----------

class TestEachAggregationPicksUpShapeBranches(unittest.TestCase):
    """5 aggregations x 3 branches: a large weight on BS, SH, or SJ (in
    isolation) must move every aggregation's output relative to weight 0."""

    def setUp(self):
        torch.manual_seed(19)
        n = 8
        self.cfg = TrainingFreeConfig(
            weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0,
            weight_ds=1.0, weight_ct=0.0, weight_dd=0.0, aggregation="soft_voting",
        )
        self.cv = torch.randn(n)
        self.m_cv = torch.randn(n)
        self.m_bm = torch.randn(n)
        self.m_bd = torch.randn(n)
        self.m_qa = torch.randn(n)
        self.m_ds = torch.randn(n)
        self.perfect = torch.randn(n) * 3.0

    def _base_kwargs(self):
        return dict(m_dd=None, m_ct=None, m_bm=self.m_bm, m_bd=self.m_bd,
                    m_qa=self.m_qa, m_ds=self.m_ds, m_lr=None, m_de=None, m_sw=None)

    def _run(self, fn, **shape_kwargs):
        return fn(self.cfg, self.cv, self.m_cv, **self._base_kwargs(), **shape_kwargs)

    def _check_branch_moves_every_aggregation(self, weight_field, margin_field):
        cfg_off = self.cfg
        cfg_on = TrainingFreeConfig(**{**self.cfg.to_dict(), weight_field: 5.0})
        for fn in (linear_aggregation, soft_voting, trimmed_mean, hard_gated, adaptive_trimmed):
            off = fn(cfg_off, self.cv, self.m_cv, **self._base_kwargs())
            on = fn(cfg_on, self.cv, self.m_cv, **self._base_kwargs(), **{margin_field: self.perfect})
            self.assertGreater(
                (off - on).abs().max().item(), 1e-6,
                f"{fn.__name__} did not move when {weight_field}=5.0 with a non-trivial margin",
            )

    def test_bs_moves_every_aggregation(self):
        self._check_branch_moves_every_aggregation("weight_bs", "m_bs")

    def test_sh_moves_every_aggregation(self):
        self._check_branch_moves_every_aggregation("weight_sh", "m_sh")

    def test_sj_moves_every_aggregation(self):
        self._check_branch_moves_every_aggregation("weight_sj", "m_sj")


# ---- (c) context_loo_stacking must not drop BS -------------------------------

class TestContextLooIncludesBs(unittest.TestCase):
    """Regression pin, extending tests/test_sh_branch.py's SJ/SH pin to BS
    (D-039's failure mode: a shape branch computed but never entering the
    branch pool)."""

    def setUp(self):
        torch.manual_seed(5)
        self.n = 10
        self.labels = torch.tensor([i % 2 for i in range(self.n)], dtype=torch.long)
        self.perfect = (self.labels.float() * 2.0 - 1.0) * 3.0

    def _stack(self, config_kw=None, **extra):
        config = TrainingFreeConfig(
            weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0,
            weight_ds=1.0, weight_ct=0.0, weight_dd=0.0,
            aggregation="context_loo_stacking", **(config_kw or {}))
        cv = torch.randn(self.n)
        m_cv = torch.randn(self.n)
        loo_cv = torch.randn(self.n)
        m_bm = torch.randn(self.n)
        m_bd = torch.randn(self.n)
        m_qa = torch.randn(self.n)
        m_ds = torch.randn(self.n)
        kwargs = dict(
            cv=cv, context_labels=self.labels,
            m_cv=m_cv, loo_cv=loo_cv,
            m_dd=None, loo_dd=None,
            m_ct=None, loo_ct=None,
            m_bm=m_bm, loo_bm=torch.randn(self.n),
            m_bd=m_bd, loo_bd=torch.randn(self.n),
            m_qa=m_qa, loo_qa=torch.randn(self.n),
            m_ds=m_ds, loo_ds=torch.randn(self.n),
            m_de=None, loo_de=None,
            m_sw=None, loo_sw=None,
        )
        kwargs.update(extra)
        return context_loo_stacking(config, **kwargs)

    def test_bs_enters_the_context_loo_pool(self):
        off = self._stack()
        on = self._stack(config_kw=dict(weight_bs=1.0, bs_dim=16),
                         m_bs=self.perfect, loo_bs=self.perfect)
        self.assertGreater((off - on).abs().max().item(), 1e-6)


# ---- (d) scripts/test_pathobench.py and voting.py must agree ----------------

class _Wrapper:
    def __init__(self, model):
        self.model = model


class TestPathobenchAndVotingAgree(unittest.TestCase):
    """The fixed-head path (scripts/test_pathobench.py) and the training-free
    pipeline path (src/models/aggregations/voting.py) must compute the SAME
    aggregation given the SAME (weight, margin) inputs -- this drift is the
    root cause this RU fixes, so it is pinned directly."""

    DIM = 48
    SKETCH = 8

    def setUp(self):
        self._env_backup = dict(os.environ)
        os.environ["ICF_FIXED_HEAD"] = "1"
        os.environ["ICF_SHAPE_SCREEN_ONLY"] = "0"
        os.environ["ICF_AGGREGATION"] = "soft_voting"
        os.environ["ICF_FIXED_HEAD_CV_WEIGHT"] = "0.0"
        os.environ["ICF_FIXED_HEAD_CT_WEIGHT"] = "0.0"
        os.environ["ICF_FIXED_HEAD_DD_WEIGHT"] = "0.0"
        os.environ["ICF_FIXED_HEAD_BM_WEIGHT"] = "1.0"
        os.environ["ICF_FIXED_HEAD_BD_WEIGHT"] = "1.0"
        os.environ["ICF_FIXED_HEAD_QA_WEIGHT"] = "1.0"
        os.environ["ICF_FIXED_HEAD_DS_WEIGHT"] = "0.0"
        os.environ["ICF_FIXED_HEAD_BS_WEIGHT"] = "1.0"
        os.environ["ICF_FIXED_HEAD_SH_WEIGHT"] = "1.0"
        os.environ["ICF_FIXED_HEAD_SJ_WEIGHT"] = "1.0"
        for key in ("ICF_FIXED_HEAD_LR_WEIGHT", "ICF_FIXED_HEAD_DE_WEIGHT",
                    "ICF_FIXED_HEAD_SW_WEIGHT", "ICF_FIXED_HEAD_RM_WEIGHT"):
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env_backup)

    def _build_model(self):
        torch.manual_seed(0)
        return CovarianceMeanLearnablePDDCTMLPModel(
            input_dim=self.DIM, token_dim=16, num_heads=1, num_layers=1, feedforward_dim=16,
            num_summary_tokens=1, max_cells=4096, dropout=0.0, covariance_sketch_dim=self.SKETCH,
            ridge_lambda=1.0, ridge_logit_scale=2.0, num_classes=2,
            ct_num_tokens=16, ct_cells_per_bag=64, ct_temperature=0.5, ct_eps=1e-6,
            ct_head_hidden_dims=[], dd_shrinkage=0.25, dd_eps=1e-6,
        ).eval()

    def _episode(self, seed=0, n_context=6, n_query=2, cells=30):
        generator = torch.Generator().manual_seed(seed)
        ids = [f"s{i}" for i in range(n_context + n_query)]
        bags = {s: torch.randn(cells, self.DIM, generator=generator) for s in ids}
        labels = {s: i % 2 for i, s in enumerate(ids)}
        train_ids = ids[:n_context]
        test_ids = ids[n_context:]
        train_y = {s: labels[s] for s in train_ids}
        test_y = {s: labels[s] for s in test_ids}
        return bags, train_ids, test_ids, train_y, test_y

    def test_soft_voting_matches_voting_py_on_the_same_margins(self):
        bags, train_ids, test_ids, train_y, test_y = self._episode()
        model = _Wrapper(self._build_model())
        with torch.no_grad():
            result = _pathobench.evaluate_trial(
                model=model, projected=bags, train_ids=train_ids, test_ids=test_ids,
                train_y=train_y, test_y=test_y, context_mode="all", context_per_class=3,
                max_tiles=None, seed=0, device=torch.device("cpu"), precision="32-true",
            )

        for key in ("m_bm", "m_bd", "m_qa", "m_sh", "m_bs", "m_sj"):
            self.assertIsNotNone(result[key], f"{key} missing from evaluate_trial's result")

        cfg = TrainingFreeConfig(
            weight_cv=0.0, weight_ct=0.0, weight_dd=0.0,
            weight_bm=1.0, weight_bd=1.0, weight_qa=1.0, weight_ds=0.0,
            weight_sh=1.0, weight_bs=1.0, weight_sj=1.0, aggregation="soft_voting",
        )
        zeros = torch.zeros_like(result["m_bm"])
        expected_margin = soft_voting(
            cfg, zeros, zeros,
            m_dd=None, m_ct=None, m_bm=result["m_bm"], m_bd=result["m_bd"],
            m_qa=result["m_qa"], m_ds=None, m_lr=None, m_de=None, m_sw=None,
            m_sh=result["m_sh"], m_bs=result["m_bs"], m_sj=result["m_sj"],
        )
        expected_probability = torch.sigmoid(expected_margin)
        self.assertLess(
            (expected_probability - result["probability"]).abs().max().item(), 1e-5,
            "scripts/test_pathobench.py's soft_voting pool drifted from "
            "src/models/aggregations/voting.py's soft_voting given identical margins",
        )


if __name__ == "__main__":
    unittest.main()
