#!/usr/bin/env python3
"""
Unit tests for src/baseline/extract_metrics.py.
Stdlib-only: no third-party imports.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baseline.extract_metrics import extract_metrics, parse_log_file


class TestMetricsExtractor(unittest.TestCase):
    def setUp(self):
        self.repo_root = REPO_ROOT
        self.fixture_log_path = self.repo_root / "tests" / "baseline" / "fixtures" / "sample_task_log.txt"
        self.primary_tasks = [
            "cptac_lscc/ARID1A_mutation",
            "cptac_lscc/Histologic_Grade",
            "cptac_lscc/KEAP1_mutation",
            "cptac_luad/KRAS_mutation",
            "cptac_pda/SMAD4_mutation",
            "ucla_lung/progression_regression",
            "cptac_ccrcc/PBRM1_mutation",
        ]

    def _create_mock_log(self, path, mean=0.63, std=0.09, pooled=0.63, n_folds=50, corrupt_summary=False, prefix_noise=True):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            if prefix_noise:
                f.write("Training epoch 01/10: loss=0.456\n")
                f.write("Validation fold 01: AUROC=0.710\n")
            if corrupt_summary:
                f.write("CUDA out of memory. Tried to allocate 2.00 GiB\n")
                return
            folds = " ".join(f"{mean:.4f}" for _ in range(n_folds))
            f.write(f"=== PathoBench official {n_folds}-fold — mock/task — 300 slides (folds 1..{n_folds}) ===\n")
            f.write(f"per-fold AUROC: {folds}\n")
            f.write(f"fold-mean AUROC: {mean:.4f} ± {std:.4f}   pooled AUROC: {pooled:.4f}\n")
            f.write("Saved official-fold predictions to predictions/mock.pt\n")

    def _create_manifest(self, manifest_path, task_means, tag="test_tag", arm="v120"):
        m_path = Path(manifest_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for task, mean in zip(self.primary_tasks, task_means):
            name = task.replace("/", "_")
            log_path = m_path.parent / "logs" / f"{name}.log"
            self._create_mock_log(log_path, mean=mean)
            entries.append({
                "task": task,
                "gpu": 0,
                "log": str(log_path),
                "predictions": f"predictions/pathobench_{name}_{tag}_official50_bf16.pt",
            })
        manifest_data = {
            "tag": tag,
            "arm": arm,
            "gpus": [0],
            "tasks": entries,
        }
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        return m_path

    def test_01_parse_sample_task_log_fixture(self):
        """1. Parsing sample_task_log.txt yields fold-mean, std, pooled and 50 per-fold values."""
        self.assertTrue(self.fixture_log_path.is_file(), f"Fixture missing: {self.fixture_log_path}")
        parsed = parse_log_file(str(self.fixture_log_path), "cptac_luad/KRAS_mutation")

        self.assertEqual(parsed["fold_mean_auroc"], 0.734)
        self.assertEqual(parsed["fold_std"], 0.0944)
        self.assertEqual(parsed["pooled_auroc"], 0.727)
        self.assertEqual(parsed["n_folds"], 50)
        self.assertEqual(len(parsed["per_fold_auroc"]), 50)
        # Check specific values from fixture
        self.assertAlmostEqual(parsed["per_fold_auroc"][0], 0.7786, places=4)
        self.assertAlmostEqual(parsed["per_fold_auroc"][1], 0.6949, places=4)
        self.assertAlmostEqual(parsed["per_fold_auroc"][-1], 0.5044, places=4)

    def test_02_golden_pass_7task_manifest(self):
        """2. 7-task manifest with means 0.60..0.66 yields macro 0.63, delta 0.0035, within 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.json"
            out_file = Path(tmpdir) / "metrics.json"
            means = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66]
            self._create_manifest(manifest_file, means)

            cmd = [
                sys.executable,
                str(self.repo_root / "src" / "baseline" / "extract_metrics.py"),
                "--manifest", str(manifest_file),
                "--out", str(out_file),
                "--reference", "0.6265",
                "--tolerance", "0.005",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Process failed: {res.stderr}")
            self.assertTrue(out_file.is_file())

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            p7 = data["primary7"]
            self.assertEqual(p7["macro_fold_mean_auroc"], 0.63)
            self.assertEqual(p7["abs_delta_vs_reference"], 0.0035)
            self.assertEqual(p7["within_tolerance"], 1)
            self.assertIs(type(p7["within_tolerance"]), int)
            self.assertEqual(p7["n_tasks"], 7)
            self.assertEqual(data["tasks"]["cptac_pda/SMAD4_mutation"]["fold_mean_auroc"], 0.64)
            self.assertEqual(data["tasks"]["cptac_ccrcc/PBRM1_mutation"]["n_folds"], 50)

            # Check stdout format
            self.assertIn("cptac_lscc/ARID1A_mutation 0.6", res.stdout)
            self.assertIn("MACRO 0.63 DELTA 0.0035 TOLERANCE 0.005 WITHIN 1", res.stdout)

    def test_03_out_of_tolerance_7task_manifest(self):
        """3. 7-task manifest with all fold-mean 0.70 produces macro 0.7, delta 0.0735, within 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.json"
            out_file = Path(tmpdir) / "metrics.json"
            means = [0.70] * 7
            self._create_manifest(manifest_file, means)

            cmd = [
                sys.executable,
                str(self.repo_root / "src" / "baseline" / "extract_metrics.py"),
                "--manifest", str(manifest_file),
                "--out", str(out_file),
                "--reference", "0.6265",
                "--tolerance", "0.005",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Process failed: {res.stderr}")

            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            p7 = data["primary7"]
            self.assertEqual(p7["macro_fold_mean_auroc"], 0.7)
            self.assertEqual(p7["abs_delta_vs_reference"], 0.0735)
            self.assertEqual(p7["within_tolerance"], 0)
            self.assertIs(type(p7["within_tolerance"]), int)

    def test_04_missing_log_file_fails(self):
        """4. A manifest naming a missing log exits non-zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.json"
            out_file = Path(tmpdir) / "metrics.json"
            means = [0.63] * 7
            self._create_manifest(manifest_file, means)

            # Delete one log file
            log_to_delete = Path(tmpdir) / "logs" / "cptac_luad_KRAS_mutation.log"
            if log_to_delete.exists():
                log_to_delete.unlink()

            cmd = [
                sys.executable,
                str(self.repo_root / "src" / "baseline" / "extract_metrics.py"),
                "--manifest", str(manifest_file),
                "--out", str(out_file),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("cptac_luad/KRAS_mutation", res.stderr)

    def test_05_missing_fold_mean_line_fails(self):
        """5. A log with no 'fold-mean AUROC:' line exits non-zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.json"
            out_file = Path(tmpdir) / "metrics.json"
            means = [0.63] * 7
            self._create_manifest(manifest_file, means)

            # Overwrite one log with crashed/OOM output (no fold-mean line)
            crashed_log = Path(tmpdir) / "logs" / "cptac_pda_SMAD4_mutation.log"
            self._create_mock_log(crashed_log, corrupt_summary=True)

            cmd = [
                sys.executable,
                str(self.repo_root / "src" / "baseline" / "extract_metrics.py"),
                "--manifest", str(manifest_file),
                "--out", str(out_file),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("cptac_pda/SMAD4_mutation", res.stderr)

    def test_06_wrong_fold_count_fails(self):
        """6. A log whose per-fold line has 49 values exits non-zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.json"
            out_file = Path(tmpdir) / "metrics.json"
            means = [0.63] * 7
            self._create_manifest(manifest_file, means)

            # Overwrite one log with 49 folds
            bad_folds_log = Path(tmpdir) / "logs" / "ucla_lung_progression_regression.log"
            self._create_mock_log(bad_folds_log, n_folds=49)

            cmd = [
                sys.executable,
                str(self.repo_root / "src" / "baseline" / "extract_metrics.py"),
                "--manifest", str(manifest_file),
                "--out", str(out_file),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("ucla_lung/progression_regression", res.stderr)

    def test_07_deterministic_output(self):
        """7. Running the extractor twice on the same input produces byte-identical output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest.json"
            out1 = Path(tmpdir) / "metrics_1.json"
            out2 = Path(tmpdir) / "metrics_2.json"
            means = [0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66]
            self._create_manifest(manifest_file, means)

            cmd1 = [
                sys.executable,
                str(self.repo_root / "src" / "baseline" / "extract_metrics.py"),
                "--manifest", str(manifest_file),
                "--out", str(out1),
            ]
            cmd2 = [
                sys.executable,
                str(self.repo_root / "src" / "baseline" / "extract_metrics.py"),
                "--manifest", str(manifest_file),
                "--out", str(out2),
            ]
            subprocess.run(cmd1, check=True)
            subprocess.run(cmd2, check=True)

            bytes1 = out1.read_bytes()
            bytes2 = out2.read_bytes()
            self.assertEqual(bytes1, bytes2)

    def test_08_invalid_task_count_in_manifest_fails(self):
        """Manifest listing != 7 tasks exits non-zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "manifest_6.json"
            out_file = Path(tmpdir) / "metrics.json"
            # 6 tasks
            entries = []
            for task in self.primary_tasks[:6]:
                log_path = Path(tmpdir) / f"{task.replace('/', '_')}.log"
                self._create_mock_log(log_path)
                entries.append({"task": task, "log": str(log_path)})
            with open(manifest_file, "w") as f:
                json.dump({"tag": "t", "arm": "v120", "tasks": entries}, f)

            cmd = [
                sys.executable,
                str(self.repo_root / "src" / "baseline" / "extract_metrics.py"),
                "--manifest", str(manifest_file),
                "--out", str(out_file),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("must list exactly 7 tasks", res.stderr)

    def test_09_multiple_summary_blocks_takes_last(self):
        """If log contains multiple summary blocks (e.g. restart), takes the last match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "restart.log"
            with open(log_file, "w", encoding="utf-8") as f:
                # First run (stale / crashed)
                folds1 = " ".join("0.5000" for _ in range(50))
                f.write(f"per-fold AUROC: {folds1}\n")
                f.write("fold-mean AUROC: 0.5000 ± 0.0500   pooled AUROC: 0.5000\n")
                f.write("Restarting evaluation...\n")
                # Second run (final)
                folds2 = " ".join("0.7500" for _ in range(50))
                f.write(f"per-fold AUROC: {folds2}\n")
                f.write("fold-mean AUROC: 0.7500 ± 0.0800   pooled AUROC: 0.7400\n")

            parsed = parse_log_file(str(log_file), "mock_task")
            self.assertEqual(parsed["fold_mean_auroc"], 0.75)
            self.assertEqual(parsed["fold_std"], 0.08)
            self.assertEqual(parsed["pooled_auroc"], 0.74)


if __name__ == "__main__":
    unittest.main()
