"""Delegated-task lifecycle: queue -> running -> completed / failed.

These are the transitions the old supervisor had no record of at all. A task's
card moved to done/ whether the command worked or not, and `started` was read
from a git branch. The tests below drive the store and the wrapper that writes
its terminal state, in a temporary directory, without starting a supervisor.
"""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ops import run_task, task_queue, task_state  # noqa: E402


class TestTransitions(unittest.TestCase):
    def test_queue_then_running_then_completed(self):
        with TemporaryDirectory() as d:
            tasks, queue = Path(d) / "tasks", Path(d) / "queue"
            tasks.mkdir()
            (tasks / "t1.md").write_text("x", encoding="utf-8")
            store = {}

            task_queue.sync_tasks(store, tasks, queue, merged=set())
            self.assertEqual(task_state.state_of(store, "t1"), "pending")
            self.assertTrue((queue / "task_t1.json").is_file())

            # The supervisor persists `running` before spawning the wrapper;
            # the wrapper then writes the terminal state to the same file.
            task_state.transition(store, "t1", "running", pid=1)
            task_state.save(Path(d) / "state.json", store)
            spec = json.loads((queue / "task_t1.json").read_text(encoding="utf-8"))
            spec["cmd"] = "true"
            rc = run_task.execute(spec, Path(d) / "state.json", cwd=d)
            store = task_state.load(Path(d) / "state.json")

            self.assertEqual(rc, 0)
            self.assertEqual(task_state.state_of(store, "t1"), "completed")
            self.assertEqual([h["to"] for h in store["t1"]["history"]],
                             ["pending", "running", "completed"])

    def test_queue_then_running_then_failed(self):
        with TemporaryDirectory() as d:
            tasks, queue = Path(d) / "tasks", Path(d) / "queue"
            tasks.mkdir()
            (tasks / "t1.md").write_text("x", encoding="utf-8")
            store = {}
            task_queue.sync_tasks(store, tasks, queue, merged=set())
            task_state.transition(store, "t1", "running", pid=1)
            task_state.save(Path(d) / "state.json", store)
            spec = json.loads((queue / "task_t1.json").read_text(encoding="utf-8"))
            spec["cmd"] = "false"

            rc = run_task.execute(spec, Path(d) / "state.json", cwd=d)
            store = task_state.load(Path(d) / "state.json")
            self.assertNotEqual(rc, 0)
            self.assertEqual(task_state.state_of(store, "t1"), "failed")

    def test_terminal_state_is_not_overwritten(self):
        # A late writer must not be able to resurrect a finished task.
        store = {}
        task_state.transition(store, "t1", "running")
        task_state.transition(store, "t1", "completed")
        with self.assertRaises(task_state.InvalidTransition):
            task_state.transition(store, "t1", "running")
        self.assertEqual(task_state.state_of(store, "t1"), "completed")

    def test_failed_needs_explicit_retry(self):
        store = {}
        task_state.transition(store, "t1", "running")
        task_state.transition(store, "t1", "failed", reason="rc=1")
        self.assertEqual(task_state.state_of(store, "t1"), "failed")
        # Not a legal edge by itself: auto-retry is what the store forbids.
        with self.assertRaises(task_state.InvalidTransition):
            task_state.transition(store, "t1", "running")
        task_state.transition(store, "t1", "pending", reason="explicit retry")
        self.assertEqual(task_state.state_of(store, "t1"), "pending")

    def test_store_roundtrip(self):
        with TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            store = {}
            task_state.transition(store, "t1", "pending", resource="remote_llm")
            task_state.save(path, store)
            self.assertEqual(task_state.load(path)["t1"]["resource"], "remote_llm")


if __name__ == "__main__":
    unittest.main()
