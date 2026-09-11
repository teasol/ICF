import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INBOX_SCRIPT = REPO_ROOT / "scripts" / "inbox.py"


class TestInboxCLI(unittest.TestCase):
    def test_inbox_script_exists_and_runnable(self):
        self.assertTrue(INBOX_SCRIPT.is_file(), "scripts/inbox.py must exist")

    def test_inbox_list_returns_zero(self):
        res = subprocess.run(
            [sys.executable, str(INBOX_SCRIPT), "list", "orca"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(res.returncode, 0, f"inbox list orca failed: {res.stderr}")

    def test_inbox_check_empty_returns_one(self):
        res = subprocess.run(
            [sys.executable, str(INBOX_SCRIPT), "check", "owl"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(res.returncode, 1, "inbox check owl on empty inbox should exit 1")


if __name__ == "__main__":
    unittest.main()
