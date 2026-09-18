"""Per-branch margins must match what aggregation actually consumed."""

import unittest

import torch

from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


def _config(**kw):
    base = dict(sketch_dim=32, weight_cv=1.0, weight_bm=1.0, weight_bd=1.0,
                weight_qa=1.0, weight_ds=1.0, weight_ct=0.0, weight_dd=0.0,
                aggregation="trimmed_mean")
    base.update(kw)
    return TrainingFreeConfig(**base)


class TestBranchMargins(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.ctx = [torch.randn(60, 64) for _ in range(20)]
        self.lab = torch.tensor([i % 2 for i in range(20)], dtype=torch.long)
        self.qry = [torch.randn(60, 64) for _ in range(4)]

    def test_zero_weight_branches_are_absent_not_zero(self):
        clf = TrainingFreeClassifier(_config())
        got = clf.branch_margins(self.ctx, self.lab, self.qry)
        self.assertEqual(set(got), {"cv", "bm", "bd", "qa", "ds"})
        self.assertNotIn("ct", got)
        self.assertNotIn("sh", got)

    def test_each_branch_has_one_score_per_query(self):
        clf = TrainingFreeClassifier(_config())
        for name, m in clf.branch_margins(self.ctx, self.lab, self.qry).items():
            self.assertEqual(m.shape[0], len(self.qry), name)

    def test_capture_does_not_change_the_aggregate(self):
        """The diagnostic must not perturb the thing it diagnoses."""
        clf = TrainingFreeClassifier(_config())
        before = clf.margins(self.ctx, self.lab, self.qry)
        clf.branch_margins(self.ctx, self.lab, self.qry)
        after = clf.margins(self.ctx, self.lab, self.qry)
        self.assertLess((before - after).abs().max().item(), 1e-6)

    def test_capture_is_released_after_the_call(self):
        clf = TrainingFreeClassifier(_config())
        clf.branch_margins(self.ctx, self.lab, self.qry)
        self.assertIsNone(clf._branch_capture)

    def test_seven_branch_config_yields_seven(self):
        clf = TrainingFreeClassifier(_config(weight_sh=1.0, weight_sj=1.0))
        got = clf.branch_margins(self.ctx, self.lab, self.qry)
        self.assertEqual(len(got), 7, sorted(got))


if __name__ == "__main__":
    unittest.main()
