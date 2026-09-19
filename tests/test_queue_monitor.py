"""큐 모니터의 순수 부분만 확인한다. 실제 서버에는 붙지 않는다.

1. vLLM Prometheus 텍스트 샘플에서 running/waiting/토큰을 뽑는 파서.
2. talks/ops/ 하위에 아무 파일도 없을 때 예외 없이 "기록 없음" 상태가 되는지.
"""

import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ops import queue_monitor as qm  # noqa: E402

SAMPLE = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="deepseek-v4.1-flash"} 2.0
vllm:num_requests_waiting{engine="0",model_name="deepseek-v4.1-flash"} 5.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="deepseek-v4.1-flash",reason="capacity"} 5.0
vllm:gpu_cache_usage_perc{engine="0",model_name="deepseek-v4.1-flash"} 0.5
vllm:generation_tokens_total{engine="0",model_name="deepseek-v4.1-flash"} 1.000648e+06
vllm:prompt_tokens_total{engine="0",model_name="deepseek-v4.1-flash"} 12345.0
"""


class TestParsePrometheus(unittest.TestCase):
    def test_counts_and_tokens(self):
        m = qm.parse_prometheus(SAMPLE)
        self.assertEqual(m["vllm:num_requests_running"], 2.0)
        self.assertEqual(m["vllm:num_requests_waiting"], 5.0)
        # 이름이 접두사처럼 겹쳐도 by_reason과 섞이지 않아야 한다.
        self.assertEqual(m["vllm:num_requests_waiting_by_reason"], 5.0)
        self.assertAlmostEqual(m["vllm:generation_tokens_total"], 1_000_648.0)
        self.assertAlmostEqual(m["vllm:prompt_tokens_total"], 12345.0)
        self.assertAlmostEqual(m["vllm:gpu_cache_usage_perc"], 0.5)

    def test_label_sets_are_summed(self):
        text = (
            'vllm:num_requests_running{engine="0"} 1.0\n'
            'vllm:num_requests_running{engine="1"} 3.0\n'
        )
        self.assertEqual(qm.parse_prometheus(text)["vllm:num_requests_running"], 4.0)

    def test_junk_lines_are_skipped(self):
        self.assertEqual(qm.parse_prometheus("# only a comment\n\nnot a metric\n"), {})
        self.assertEqual(qm.parse_prometheus('m{a="b"}notanumber\n'), {})


class TestEmptyOpsDir(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        root = Path(self._tmp.name)
        self._saved = (qm.RECURRING, qm.TASKS, qm.TICKS, qm.FAILURES,
                       qm.HEARTBEAT, qm.STORE)
        qm.RECURRING = root / "recurring"
        qm.TASKS = root / "tasks"
        qm.TICKS = root / "ticks.jsonl"
        qm.FAILURES = root / "failures.jsonl"
        qm.HEARTBEAT = root / "heartbeat"
        qm.STORE = root / "task_state.json"

    def tearDown(self):
        (qm.RECURRING, qm.TASKS, qm.TICKS, qm.FAILURES,
         qm.HEARTBEAT, qm.STORE) = self._saved
        self._tmp.cleanup()

    def test_no_files_yields_empty_records_without_raising(self):
        state = qm.build_state(metrics={"reachable": False, "error": "test"},
                               now=datetime(2026, 9, 19, 9, 0, 0, tzinfo=qm.KST))
        self.assertEqual(state["recurring"], [])
        self.assertEqual(state["tasks"], [])
        self.assertEqual(state["ticks"], [])
        self.assertEqual(state["failures"], [])
        self.assertEqual(state["heartbeat"]["present"], False)
        self.assertFalse(state["server"]["reachable"])
        # 클라이언트는 빈 목록을 이 문구로 그린다.
        self.assertIn("기록 없음", qm.PAGE)

    def test_recurring_remaining_minutes_from_ticks(self):
        root = self._tmp.name
        Path(root, "recurring").mkdir()
        Path(root, "ticks.jsonl").write_text(
            json.dumps({"t": "2026-09-19 08:00:00",
                        "dispatch": [{"launch": "lit-index"}]}, ensure_ascii=False)
            + "\n" + "{ this line is truncated",
            encoding="utf-8")
        (qm.RECURRING / "30_lit.json").write_text(
            json.dumps({"kind": "routine", "label": "lit-index",
                        "cmd": "true", "every_minutes": 30, "parallel": True}),
            encoding="utf-8")
        now = datetime(2026, 9, 19, 8, 10, 0, tzinfo=qm.KST)
        item = qm.read_recurring(now)[0]
        self.assertEqual(item["last_run"], "2026-09-19 08:00:00")
        self.assertAlmostEqual(item["remaining_minutes"], 20.0)
        self.assertEqual(item["status"], "다음 20분")

        overdue = qm.read_recurring(now + timedelta(minutes=25))[0]
        self.assertEqual(overdue["status"], "지연 5분")


class TestTaskStates(unittest.TestCase):
    """Task status comes from the store, not from a deletable branch name."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._saved_tasks, self._saved_store = qm.TASKS, qm.STORE
        qm.TASKS = Path(self._tmp.name) / "tasks"
        qm.TASKS.mkdir()
        qm.STORE = Path(self._tmp.name) / "task_state.json"
        for name in ("merged", "running", "broke", "fresh"):
            (qm.TASKS / f"{name}.md").write_text("x", encoding="utf-8")
        qm.STORE.write_text(json.dumps({
            "merged": {"name": "merged", "state": "completed", "branch": "chore/merged"},
            "running": {"name": "running", "state": "running", "resource": "remote_llm"},
            "broke": {"name": "broke", "state": "failed", "reason": "rc=1"},
        }), encoding="utf-8")

    def tearDown(self):
        qm.TASKS, qm.STORE = self._saved_tasks, self._saved_store
        self._tmp.cleanup()

    def test_state_survives_branch_deletion(self):
        # `merged` has no branch in this fixture at all, yet reads completed.
        by_name = {t["name"]: t for t in qm.read_tasks()}
        self.assertEqual(by_name["merged"]["state"], "completed")
        self.assertEqual(by_name["running"]["state"], "running")
        self.assertEqual(by_name["broke"]["state"], "failed")
        self.assertEqual(by_name["broke"]["error"], "rc=1")

    def test_task_with_no_record_reads_unstarted(self):
        by_name = {t["name"]: t for t in qm.read_tasks()}
        self.assertEqual(by_name["fresh"]["state"], "미착수")


class TestStarvation(unittest.TestCase):
    def test_idle_server_with_no_work_is_starved(self):
        metrics = {"reachable": True, "running": 0.0, "waiting": 0.0}
        st = qm.starvation(metrics, tasks=[], recurring=[])
        self.assertEqual(st["state"], "work-starved")

    def test_idle_server_with_pending_task_is_not_starved(self):
        metrics = {"reachable": True, "running": 0.0, "waiting": 0.0}
        tasks = [{"name": "t", "state": "pending", "resource": "remote_llm"}]
        self.assertNotEqual(qm.starvation(metrics, tasks, [])["state"], "work-starved")

    def test_unreachable_server_is_unknown_not_zero(self):
        st = qm.starvation({"reachable": False}, tasks=[], recurring=[])
        self.assertEqual(st["state"], "모름")

    def test_busy_server_is_not_starved(self):
        metrics = {"reachable": True, "running": 2.0, "waiting": 5.0}
        self.assertNotEqual(qm.starvation(metrics, [], [])["state"], "work-starved")


if __name__ == "__main__":
    unittest.main()
