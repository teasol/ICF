"""Task cards enter the queue exactly once, and merged ones never do.

Measured defect this pins down. The monitor inferred "started" from a
`chore/<name>-*` branch, so a merged task whose branch had been deleted read as
미착수 forever. `perf_transfer` is the one card in this repository with no merge
commit; every other one has one. The tests use synthetic names so they do not
depend on the repository's current git history.
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ops import task_queue, task_state  # noqa: E402


class TestSyncTasks(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tasks = Path(self._tmp.name) / "tasks"
        self.queue = Path(self._tmp.name) / "queue"
        self.tasks.mkdir()
        self.queue.mkdir()
        for name in ("landed", "waiting"):
            (self.tasks / f"{name}.md").write_text("x", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_merged_task_is_not_enqueued(self):
        store = {}
        added = task_queue.sync_tasks(store, self.tasks, self.queue,
                                      merged={"landed"})
        self.assertEqual(added, ["waiting"])
        self.assertEqual(task_state.state_of(store, "landed"), "completed")
        self.assertFalse((self.queue / "task_landed.json").exists())
        self.assertTrue((self.queue / "task_waiting.json").is_file())

    def test_unfinished_task_is_enqueued_exactly_once(self):
        store = {}
        task_queue.sync_tasks(store, self.tasks, self.queue, merged=set())
        task_queue.sync_tasks(store, self.tasks, self.queue, merged=set())
        task_queue.sync_tasks(store, self.tasks, self.queue, merged=set())
        cards = list(self.queue.glob("task_waiting.json"))
        self.assertEqual(len(cards), 1)
        self.assertEqual(task_state.state_of(store, "waiting"), "pending")

    def test_running_task_is_not_requeued(self):
        # After a supervisor restart, a `running` record must not be re-fired.
        store = {}
        task_queue.sync_tasks(store, self.tasks, self.queue, merged=set())
        (self.queue / "task_waiting.json").unlink()
        task_state.transition(store, "waiting", "running", pid=999)
        added = task_queue.sync_tasks(store, self.tasks, self.queue, merged=set())
        self.assertEqual(added, [])
        self.assertFalse((self.queue / "task_waiting.json").exists())

    def test_retry_after_failure_requeues(self):
        store = {}
        task_queue.sync_tasks(store, self.tasks, self.queue, merged=set())
        (self.queue / "task_waiting.json").unlink()
        task_state.transition(store, "waiting", "running")
        task_state.transition(store, "waiting", "failed", reason="rc=1")
        task_queue.retry(store, "waiting", self.tasks, self.queue)
        self.assertEqual(task_state.state_of(store, "waiting"), "pending")
        self.assertTrue((self.queue / "task_waiting.json").is_file())

    def test_merge_detection_is_message_based(self):
        log = ("abc123 chore(queue_monitor): opencode 자동 편집 (미검토)\n"
               "def456 docs: something else\n")
        self.assertEqual(task_queue.merged_task_names(log), {"queue_monitor"})


if __name__ == "__main__":
    unittest.main()
