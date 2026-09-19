"""The official runner must record provenance, or its results are unattributable.

D-050: stored predictions recorded only task_dir/folds/AUROCs, so no result
could be attributed to a branch list, weights, code hash or data version. The
public checks in the council also flagged this. This test pins the schema so a
later edit cannot quietly drop a field again.

unittest style on purpose: `scripts/run_tests.sh` runs `unittest discover`, so
pytest-style functions are not part of the regression suite.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.evaluate_pure import build_provenance
from src.models.config import TrainingFreeConfig


def _config() -> TrainingFreeConfig:
    return TrainingFreeConfig(
        sketch_dim=32, aggregation="trimmed_mean",
        weight_cv=1.0, weight_bm=1.0, weight_bd=1.0, weight_qa=1.0,
        weight_ds=1.0, weight_sh=1.0, weight_sj=1.0,
        weight_ct=0.0, weight_dd=0.0,
    )


class TestEvaluatePureProvenance(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _task(self) -> Path:
        task = self.tmp / "official" / "cptac_pda" / "SMAD4_mutation"
        task.mkdir(parents=True)
        (task / "k=all.tsv").write_text("slide\tfold_0\ns1\ttest\n", encoding="utf-8")
        (task / "config.yaml").write_text("name: tiny\n", encoding="utf-8")
        return task

    def _prov(self, task, cfg_name="x.yaml"):
        return build_provenance(_config(), self.tmp / cfg_name, task,
                                self.tmp / "features", torch.device("cpu"))

    def test_has_attribution_fields(self):
        task = self._task()
        (self.tmp / "v121.yaml").write_text(
            "aggregation: trimmed_mean\nsketch_dim: 32\n"
            "branches:\n"
            "  cv: {weight: 1.0}\n  bm: {weight: 1.0}\n  bd: {weight: 1.0}\n"
            "  qa: {weight: 1.0}\n  ds: {weight: 1.0}\n  sh: {weight: 1.0}\n"
            "  sj: {weight: 1.0}\n  ct: {weight: 0.0}\n  dd: {weight: 0.0}\n",
            encoding="utf-8")
        prov = self._prov(task, "v121.yaml")
        for key in ("branch_list", "weights", "aggregation", "sketch_dim",
                    "config_sha256", "code_hash", "git_commit", "data_version",
                    "manifest_hash", "env"):
            self.assertIn(key, prov, f"provenance에 {key} 가 없다")
        # The declared branches are 7; the legacy shj alias is NOT a branch and
        # must not appear even though TrainingFreeConfig mirrors weight_sj into it.
        self.assertEqual(prov["branch_list"],
                         ["bd", "bm", "cv", "ds", "qa", "sh", "sj"])
        self.assertNotIn("shj", prov["branch_list"])
        self.assertNotIn("ct", prov["branch_list"])
        self.assertTrue(prov["config_sha256"])
        self.assertTrue(prov["manifest_hash"])
        self.assertEqual(prov["data_version"]["task_dir"], str(task))

    def test_manifest_hash_tracks_fold_content(self):
        task = self._task()
        before = self._prov(task)["manifest_hash"]
        (task / "k=all.tsv").write_text(
            "slide\tfold_0\ns2\ttest\n", encoding="utf-8")
        after = self._prov(task)["manifest_hash"]
        self.assertNotEqual(before, after)

    def test_scanning_keys_are_json_serialisable(self):
        prov = self._prov(self._task())
        blob = json.loads(json.dumps(prov))  # no tensors, no Path objects
        self.assertTrue(
            {"branch_list", "weights", "code_hash", "git_commit",
             "data_version", "manifest_hash", "env"} <= set(blob))


if __name__ == "__main__":
    unittest.main()
