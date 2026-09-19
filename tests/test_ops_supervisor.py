"""Queue handling in the supervisor, without starting any process.

Only the pure parts are covered: dispatch ordering and the malformed-item path.
The launch path itself spawns detached processes and is exercised by running
the loop, not by the suite.
"""

import json
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


class TestDelegatedTasks(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self._saved = (sup.QUEUE, sup.DONE, sup.TASKS, sup._git_lines)
        sup.QUEUE, sup.DONE, sup.TASKS = root / "queue", root / "done", root / "tasks"
        sup.QUEUE.mkdir(); sup.DONE.mkdir(); sup.TASKS.mkdir()

    def tearDown(self):
        sup.QUEUE, sup.DONE, sup.TASKS, sup._git_lines = self._saved
        self._tmp.cleanup()

    def test_merged_tasks_are_not_queued_but_pending_tasks_are(self):
        for name in ("docs_links", "perf_transfer"):
            (sup.TASKS / f"{name}.md").write_text("task", encoding="utf-8")
        sup._git_lines = lambda *args: (
            ["main"] if args[0] == "for-each-ref"
            else ["Merge branch 'chore/docs_links-0918-2055'"])

        self.assertEqual(sup.refill_delegated_tasks(), ["perf_transfer"])
        card = json.loads((sup.QUEUE / "40_task_perf_transfer.json").read_text())
        self.assertEqual(card["label"], "task:perf_transfer")
        self.assertIn("perf_transfer.md", card["cmd"])

    def test_done_marker_prevents_automatic_retry(self):
        (sup.TASKS / "perf_transfer.md").write_text("task", encoding="utf-8")
        (sup.DONE / "40_task_perf_transfer.json").write_text("{}", encoding="utf-8")
        sup._git_lines = lambda *args: []
        self.assertEqual(sup.refill_delegated_tasks(), [])

    def test_task_state_prefers_durable_completion(self):
        refs = ["chore/queue_monitor-0919-0855"]
        subjects = ["Merge branch 'chore/queue_monitor-0919-0855'"]
        self.assertEqual(sup.task_state("queue_monitor", refs, subjects), "completed")


if __name__ == "__main__":
    unittest.main()
