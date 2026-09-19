"""Queue handling in the supervisor, without starting any process.

Only the pure parts are covered: dispatch ordering and the malformed-item path.
The launch path itself spawns detached processes and is exercised by running
the loop, not by the suite.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.ops.supervisor as sup  # noqa: E402


class TestQueueOrder(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self._saved = (sup.QUEUE, sup.DONE)
        sup.QUEUE, sup.DONE = root / "queue", root / "done"
        sup.QUEUE.mkdir(); sup.DONE.mkdir()

    def tearDown(self):
        sup.QUEUE, sup.DONE = self._saved
        self._tmp.cleanup()

    def _write(self, name, spec):
        (sup.QUEUE / name).write_text(json.dumps(spec), encoding="utf-8")

    def test_empty_queue_yields_nothing(self):
        self.assertIsNone(sup.next_item())

    def test_filename_order_is_priority(self):
        # Callers control priority by naming, so 01 must precede 02 regardless
        # of creation order.
        self._write("02_second.json", {"kind": "shell", "cmd": "true"})
        self._write("01_first.json", {"kind": "shell", "cmd": "true"})
        self.assertEqual(sup.next_item().name, "01_first.json")

    def test_unknown_kind_is_retired_not_retried(self):
        # A malformed item left in the queue would be re-dispatched every tick.
        self._write("01_bad.json", {"kind": "nope", "label": "bad"})
        rec, handle = sup.launch(sup.next_item())
        self.assertIsNone(handle)
        self.assertIn("unknown kind", rec["error"])
        self.assertIsNone(sup.next_item())
        self.assertTrue((sup.DONE / "01_bad.json").is_file())


class TestResourceConflicts(unittest.TestCase):
    def test_login_cpu_and_remote_llm_coexist(self):
        active = [{"resource": "login_cpu", "worktree": None}]
        self.assertFalse(sup.conflicts("remote_llm", None, active))
        self.assertFalse(sup.conflicts("login_cpu", None, active))

    def test_exclusive_gpu_excludes_slurm_and_itself(self):
        active = [{"resource": "slurm", "worktree": None}]
        self.assertTrue(sup.conflicts("exclusive_gpu", None, active))
        active = [{"resource": "exclusive_gpu", "worktree": None}]
        self.assertTrue(sup.conflicts("exclusive_gpu", None, active))

    def test_same_worktree_never_runs_twice(self):
        active = [{"resource": "remote_llm", "worktree": "chore-x"}]
        self.assertTrue(sup.conflicts("remote_llm", "chore-x", active))
        self.assertFalse(sup.conflicts("remote_llm", "chore-y", active))


class TestRecover(unittest.TestCase):
    def test_dead_running_task_becomes_failed(self):
        store = {"t": {"name": "t", "state": "running", "pid": 2 ** 31 - 1}}
        with TemporaryDirectory() as d:
            notes = sup.recover(store, Path(d) / "state.json")
        self.assertEqual(store["t"]["state"], "failed")
        self.assertTrue(notes)

    def test_live_running_task_is_adopted_not_retried(self):
        store = {"t": {"name": "t", "state": "running", "pid": os.getpid()}}
        sup.recover(store)
        self.assertEqual(store["t"]["state"], "running")


if __name__ == "__main__":
    unittest.main()
