"""--check가 비교할 항목이 없는데도 성공을 보고하지 않는지 확인한다."""

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class FingerprintKeyMismatch(unittest.TestCase):
    """--check가 비교할 항목이 없는데도 0을 돌려주면 안 된다.

    7b53973이 키에 '@device'를 붙였는데 기준선 파일은 옛 이름 그대로여서
    교집합이 비었고, --check는 아무것도 비교하지 않은 채 성공을 보고했다.
    """

    def test_missing_baseline_keys_fail_loudly(self):
        import json as _json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            stale = Path(d) / "stale_baseline.json"
            stale.write_text(_json.dumps({"5branch": {"sha256": "deadbeefdeadbeef",
                                                      "margins": [], "branches": {}}}),
                             encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(REPO / "scripts/analysis/numeric_fingerprint.py"),
                 "--check", str(stale), "--cpu-only"],
                capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("5branch", proc.stdout)


if __name__ == "__main__":
    unittest.main()
