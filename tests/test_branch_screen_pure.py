"""Gate-1 pure-runner screen: loader and correlation logic, on synthetic margins.

unittest style (scripts/run_tests.sh uses unittest discover).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.analysis.branch_screen_pure import max_abs_corr, load_pure


class TestBranchScreenPure(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _dump(self, task: str, branches: dict):
        per_fold = []
        for k in range(3):
            rec = {"fold": k}
            for b, scale in branches.items():
                n = 8
                base = torch.linspace(-1.0, 1.0, n) * float(scale)
                rec.setdefault("branch_margins", {})[b] = base
            rec["label"] = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
            per_fold.append(rec)
        torch.save({"task": task, "per_fold": per_fold}, self.dir / f"{task}.pt")

    def test_load_builds_margin_keys(self):
        self._dump("t1", {"cv": 0.1, "sj": 0.2})
        data = load_pure(self.dir)
        self.assertIn("m_cv", data["t1"][0])
        self.assertIn("m_sj", data["t1"][0])
        self.assertEqual(len(data["t1"]), 3)

    def test_identical_candidate_is_correlated(self):
        # A candidate column identical to a reference column -> |r| ~ 1 -> rejected.
        self._dump("t1", {"cv": 0.1, "bm": 0.2, "sj": 0.1})
        data = load_pure(self.dir)
        worst, per_task = max_abs_corr(data, ["m_cv", "m_bm"], "m_sj")
        self.assertGreater(worst, 0.99)
        self.assertIn("t1", per_task)


if __name__ == "__main__":
    unittest.main()
