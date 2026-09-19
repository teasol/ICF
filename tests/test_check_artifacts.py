"""L1 attribution check tests (contamination-check redesign).

unittest style: scripts/run_tests.sh runs `unittest discover`.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.analysis.check_artifacts import check_file, expected_for
from scripts.evaluate_pure import build_provenance
from src.models.config import TrainingFreeConfig


def _write_config(path: Path, weights: dict[str, float]) -> None:
    lines = ["aggregation: trimmed_mean", "sketch_dim: 32", "branches:"]
    for name, w in weights.items():
        lines.append(f"  {name}: {{weight: {w}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCheckArtifacts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.task = self.tmp / "official" / "SMAD4_mutation"
        self.task.mkdir(parents=True)
        (self.task / "k=all.tsv").write_text("slide\tfold_0\ns1\ttest\n", encoding="utf-8")
        (self.task / "config.yaml").write_text("name: tiny\n", encoding="utf-8")
        self.cfg5 = self.tmp / "five.yaml"
        _write_config(self.cfg5, {"cv": 1.0, "bm": 1.0, "bd": 1.0, "qa": 1.0, "ds": 1.0,
                                  "ct": 0.0, "dd": 0.0})
        self.cfg7 = self.tmp / "seven.yaml"
        _write_config(self.cfg7, {"cv": 1.0, "bm": 1.0, "bd": 1.0, "qa": 1.0, "ds": 1.0,
                                  "sh": 1.0, "sj": 1.0, "ct": 0.0, "dd": 0.0})

    def tearDown(self):
        self._tmp.cleanup()

    def _artifact(self, cfg_path: Path, name: str = "a.pt") -> Path:
        cfg = TrainingFreeConfig.from_yaml(cfg_path)
        prov = build_provenance(cfg, cfg_path, self.task, self.tmp / "feat",
                                torch.device("cpu"))
        p = self.tmp / name
        torch.save({"per_fold": [], "provenance": prov}, p)
        return p

    def test_matching_artifact_passes(self):
        expected = [expected_for(self.cfg5)]
        ok, why = check_file(self._artifact(self.cfg5), expected)
        self.assertTrue(ok, why)

    def test_missing_provenance_fails(self):
        p = self.tmp / "noprov.pt"
        torch.save({"per_fold": []}, p)
        ok, why = check_file(p, [expected_for(self.cfg5)])
        self.assertFalse(ok)
        self.assertIn("provenance", why)

    def test_wrong_config_is_detected(self):
        # A 7-branch artifact checked against the 5-branch expectation must fail:
        # this is the D-053 silent-substitution signature.
        expected = [expected_for(self.cfg5)]
        ok, why = check_file(self._artifact(self.cfg7), expected)
        self.assertFalse(ok)
        self.assertIn("config_sha256", why)

    def test_both_configs_accepted_together(self):
        expected = [expected_for(self.cfg5), expected_for(self.cfg7)]
        for cfg in (self.cfg5, self.cfg7):
            ok, why = check_file(self._artifact(cfg), expected)
            self.assertTrue(ok, why)

    def test_branch_list_mismatch_fails(self):
        # Provenance with the right hash but a tampered branch_list.
        p = self._artifact(self.cfg5)
        d = torch.load(p, weights_only=False)
        d["provenance"]["branch_list"] = ["cv", "bm", "bd", "qa", "ds", "xx"]
        torch.save(d, p)
        ok, why = check_file(p, [expected_for(self.cfg5)])
        self.assertFalse(ok)
        self.assertIn("branch_list", why)


if __name__ == "__main__":
    unittest.main()
