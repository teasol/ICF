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

from scripts.ops import gpu_telemetry  # noqa: E402
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
                       qm.HEARTBEAT, qm.STORE, qm.TOKEN_STORE, qm.GPU_CACHE)
        qm.RECURRING = root / "recurring"
        qm.TASKS = root / "tasks"
        qm.TICKS = root / "ticks.jsonl"
        qm.FAILURES = root / "failures.jsonl"
        qm.HEARTBEAT = root / "heartbeat"
        qm.STORE = root / "task_state.json"
        qm.TOKEN_STORE = root / "token_baseline.json"
        qm.GPU_CACHE = root / "gpu_telemetry_cache.json"

    def tearDown(self):
        (qm.RECURRING, qm.TASKS, qm.TICKS, qm.FAILURES, qm.HEARTBEAT,
         qm.STORE, qm.TOKEN_STORE, qm.GPU_CACHE) = self._saved
        self._tmp.cleanup()

    def test_no_files_yields_empty_records_without_raising(self):
        state = qm.build_state(metrics={"reachable": False, "error": "test"},
                               now=datetime(2026, 9, 19, 9, 0, 0, tzinfo=qm.KST))
        self.assertEqual(state["recurring"], [])
        self.assertEqual(state["work"], [])
        self.assertEqual(state["completed"], [])
        self.assertEqual(state["ticks"], [])
        self.assertEqual(state["failures"], [])
        self.assertEqual(state["heartbeat"]["present"], False)
        self.assertFalse(state["server"]["reachable"])
        self.assertFalse(state["gpus"]["available"])
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


class TestWorkAndCompleted(unittest.TestCase):
    """할 일 화면은 실제 running/pending만, 완료는 별도 섹션이다."""

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._saved_tasks, self._saved_store = qm.TASKS, qm.STORE
        qm.TASKS = Path(self._tmp.name) / "tasks"
        qm.TASKS.mkdir()
        qm.STORE = Path(self._tmp.name) / "task_state.json"
        (qm.TASKS / "running.md").write_text(
            "# 전송 병목 확인\n\nfold마다 같은 슬라이드를 다시 올리는지 재라.\n",
            encoding="utf-8")
        for name in ("pending", "done", "broke"):
            (qm.TASKS / f"{name}.md").write_text("# " + name + "\n\n요약\n",
                                                   encoding="utf-8")
        qm.STORE.write_text(json.dumps({
            "running": {"name": "running", "state": "running", "resource": "remote_llm",
                        "history": [{"t": "2026-09-19 08:00:00", "from": "pending",
                                     "to": "running"}]},
            "pending": {"name": "pending", "state": "pending", "history": [
                {"t": "2026-09-19 08:05:00", "from": None, "to": "pending"}]},
            "done": {"name": "done", "state": "completed", "history": [
                {"t": "2026-09-19 07:00:00", "from": "running", "to": "completed"}]},
            "broke": {"name": "broke", "state": "failed", "reason": "rc=1"},
        }), encoding="utf-8")

    def tearDown(self):
        qm.TASKS, qm.STORE = self._saved_tasks, self._saved_store
        self._tmp.cleanup()

    def test_work_has_only_running_and_pending(self):
        work = qm.read_work()
        names = [w["name"] for w in work]
        self.assertEqual(names, ["running", "pending"])
        self.assertNotIn("done", names)
        self.assertNotIn("broke", names)

    def test_completed_is_separate_and_has_no_running(self):
        done = qm.read_completed()
        self.assertEqual([d["name"] for d in done], ["done"])
        self.assertEqual(done[0]["since"], "2026-09-19 07:00:00")

    def test_work_item_carries_title_summary_state_and_since(self):
        running = qm.read_work()[0]
        self.assertEqual(running["title"], "전송 병목 확인")
        self.assertIn("fold마다", running["summary"])
        self.assertEqual(running["state"], "running")
        self.assertEqual(running["since"], "2026-09-19 08:00:00")

    def test_work_and_completed_never_overlap(self):
        work = {w["name"] for w in qm.read_work()}
        done = {d["name"] for d in qm.read_completed()}
        self.assertEqual(work & done, set())


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


class TestTodayTokens(unittest.TestCase):
    """자정 rollover, counter reset, M 단위 표기 — 순수 함수."""

    def test_new_day_starts_partial(self):
        now = datetime(2026, 9, 19, 9, 0, 0, tzinfo=qm.KST)
        display, rec = qm.advance_today({}, 5_000_000.0, now)
        self.assertEqual(display["value"], 0.0)
        self.assertTrue(display["partial"])          # 자정 이후 첫 관측
        self.assertEqual(rec["observed_since"], "2026-09-19 09:00:00")
        self.assertEqual(rec["last_counter"], 5_000_000.0)

    def test_partial_is_false_only_when_observed_from_midnight(self):
        display, _ = qm.advance_today(
            {}, 0.0, datetime(2026, 9, 19, 0, 0, 0, tzinfo=qm.KST))
        self.assertFalse(display["partial"])

    def test_same_day_accumulates_delta(self):
        now = datetime(2026, 9, 19, 12, 0, 0, tzinfo=qm.KST)
        rec = {"date": "2026-09-19", "accumulated": 200_000.0,
               "last_counter": 1_000_000.0,
               "observed_since": "2026-09-19 09:00:00", "resets": 0}
        display, new = qm.advance_today(rec, 1_500_000.0, now)
        self.assertEqual(display["value"], 700_000.0)
        self.assertEqual(new["last_counter"], 1_500_000.0)
        self.assertFalse(display["reset"])

    def test_rollover_discards_yesterday(self):
        now = datetime(2026, 9, 20, 1, 0, 0, tzinfo=qm.KST)
        rec = {"date": "2026-09-19", "accumulated": 9_000_000.0,
               "last_counter": 20_000_000.0,
               "observed_since": "2026-09-19 09:00:00", "resets": 0}
        display, new = qm.advance_today(rec, 20_500_000.0, now)
        self.assertEqual(display["value"], 0.0)
        self.assertEqual(new["date"], "2026-09-20")
        self.assertEqual(new["accumulated"], 0.0)
        self.assertTrue(display["partial"])

    def test_counter_reset_is_detected_not_negative(self):
        now = datetime(2026, 9, 19, 12, 0, 0, tzinfo=qm.KST)
        rec = {"date": "2026-09-19", "accumulated": 100_000.0,
               "last_counter": 2_000_000.0,
               "observed_since": "2026-09-19 09:00:00", "resets": 0}
        display, new = qm.advance_today(rec, 150_000.0, now)
        self.assertGreaterEqual(display["value"], 0.0)
        self.assertEqual(display["value"], 250_000.0)   # acc + post-reset counter
        self.assertTrue(display["reset"])
        self.assertEqual(new["resets"], 1)

    def test_unknown_counter_is_unknown_not_zero(self):
        display, _ = qm.advance_today({}, None,
                                      datetime(2026, 9, 19, 9, 0, 0, tzinfo=qm.KST))
        self.assertIsNone(display["value"])
        self.assertEqual(display["text"], "모름")

    def test_m_units_formatting(self):
        self.assertEqual(qm.format_tokens_m(1_234_567.0), "1.23 M")
        self.assertEqual(qm.format_tokens_m(0.0), "0.00 M")
        self.assertEqual(qm.format_tokens_m(None), "모름")


class TestNoCumulativeTokens(unittest.TestCase):
    """누적 prompt/전체 누적 generation은 화면과 API 양쪽에서 사라져야 한다."""

    FORBIDDEN = ("generation_tokens_total", "prompt_tokens_total")

    def _payload(self):
        m = qm.parse_prometheus(SAMPLE)
        today = {"value": 700_000.0, "text": "0.70 M", "partial": True,
                 "observed_since": "2026-09-19 09:00:00", "reset": False}
        return qm.build_server_payload(m, "http://x/metrics", 1.5, today)

    def test_api_payload_has_no_cumulative_fields(self):
        payload = self._payload()
        for key in self.FORBIDDEN:
            self.assertNotIn(key, payload)
        self.assertEqual(payload["generation_tokens_today"], 700_000.0)
        self.assertEqual(payload["generation_tokens_today_text"], "0.70 M")
        self.assertTrue(payload["generation_tokens_today_partial"])

    def test_unreachable_payload_has_no_cumulative_fields(self):
        payload = qm._unreachable_server("test")
        for key in self.FORBIDDEN:
            self.assertNotIn(key, payload)

    def test_page_does_not_render_cumulative_cards(self):
        self.assertNotIn("누적 프롬프트", qm.PAGE)
        self.assertNotIn("누적 생성 토큰", qm.PAGE)
        self.assertIn("오늘 생성 토큰", qm.PAGE)


class TestGpuTelemetry(unittest.TestCase):
    """GPU telemetry는 순수 파싱과 캐시 표시를 표본으로 검증한다."""

    FULL = ("4, 242.69, 1000.00, 0\n"
            "5, 240.58, 1000.00, 0\n"
            "6, 239.67, 1000.00, 0\n"
            "7, 243.10, 1000.00, 0\n")
    PARTIAL = ("4, 242.69, 1000.00, 0\n"
               "5, 240.58, 1000.00, 12\n")
    NOW = datetime(2026, 9, 19, 9, 0, 0, tzinfo=qm.KST)

    def test_parse_full_keeps_real_zero(self):
        parsed = gpu_telemetry.parse_nvidia_smi(self.FULL)
        self.assertEqual(set(parsed["gpus"]), {4, 5, 6, 7})
        self.assertEqual(parsed["gpus"][4]["utilization"], 0.0)  # 0은 모름이 아니다
        self.assertEqual(parsed["gpus"][4]["power_draw"], 242.69)

    def test_parse_missing_values_become_none(self):
        parsed = gpu_telemetry.parse_nvidia_smi("4, [N/A], 1000.00, 0\n")
        self.assertIsNone(parsed["gpus"][4]["power_draw"])

    def test_normal_state_does_not_claim_idle_from_one_zero_sample(self):
        cache = {"epoch": self.NOW.timestamp(), "collected_at": "2026-09-19 09:00:00",
                 "mode": "ssh", "source": "ssh:nhn", "ok": True, "error": None,
                 "gpus": {str(k): v for k, v in
                          gpu_telemetry.parse_nvidia_smi(self.FULL)["gpus"].items()}}
        state = gpu_telemetry.telemetry_state(cache, self.NOW, None, 15)
        self.assertTrue(state["available"])
        # 0% 표본 하나만으로는 유휴가 아니다.
        self.assertEqual([g["status"] for g in state["gpus"]], ["모름"] * 4)
        self.assertEqual(state["gpus"][0]["power_limit"], 1000.0)

    def test_partial_missing_is_unknown_not_zero(self):
        cache = {"epoch": self.NOW.timestamp(), "collected_at": "2026-09-19 09:00:00",
                 "mode": "ssh", "source": "ssh:nhn", "ok": True, "error": None,
                 "gpus": {str(k): v for k, v in
                          gpu_telemetry.parse_nvidia_smi(self.PARTIAL)["gpus"].items()}}
        state = gpu_telemetry.telemetry_state(cache, self.NOW, None, 15)
        self.assertTrue(state["available"])
        by_index = {g["index"]: g for g in state["gpus"]}
        self.assertEqual(by_index[4]["status"], "모름")   # 0% 한 표본은 유휴 아님
        self.assertEqual(by_index[5]["utilization"], 12.0)
        self.assertEqual(by_index[5]["status"], "사용 중")
        self.assertEqual(by_index[6]["status"], "모름")
        self.assertIsNone(by_index[6]["power_draw"])
        self.assertIsNone(by_index[7]["power_draw"])

    def test_unreachable_is_unknown_not_zero(self):
        state = gpu_telemetry.telemetry_state(
            {"epoch": self.NOW.timestamp(), "ok": False, "error": "ssh: timeout",
             "gpus": {}, "collected_at": "2026-09-19 09:00:00"}, self.NOW, None, 15)
        self.assertFalse(state["available"])
        self.assertIn("timeout", state["error"])
        for g in state["gpus"]:
            self.assertIsNone(g["power_draw"])
            self.assertEqual(g["status"], "모름")

    def test_no_cache_is_unavailable(self):
        state = gpu_telemetry.telemetry_state({}, self.NOW, None, 15)
        self.assertFalse(state["available"])
        self.assertTrue(all(g["status"] == "모름" for g in state["gpus"]))

    def test_build_state_never_collects(self):
        # 표시 경로는 캐시만 읽는다. collect가 불리면 예외가 나도록 바꾼다.
        def boom(*a, **k):
            raise AssertionError("요청 경로에서 GPU를 수집하면 안 된다")
        with TemporaryDirectory() as d:
            saved_cache = qm.GPU_CACHE
            qm.GPU_CACHE = Path(d) / "gpu.json"
            (qm.GPU_CACHE).write_text(json.dumps(
                {"epoch": self.NOW.timestamp(), "ok": True,
                 "collected_at": "2026-09-19 09:00:00", "source": "ssh:nhn",
                 "gpus": {"4": {"power_draw": 1.0, "power_limit": 2.0,
                                "utilization": 3.0}}}), encoding="utf-8")
            original = gpu_telemetry.collect
            gpu_telemetry.collect = boom
            try:
                state = qm.build_state(metrics={"reachable": False, "error": "x"},
                                       now=self.NOW)
                self.assertTrue(state["gpus"]["available"])
            finally:
                gpu_telemetry.collect = original
                qm.GPU_CACHE = saved_cache


class TestGpuActivityStatus(unittest.TestCase):
    """GPU 상태는 값 존재가 아니라 연속 0% 구간으로 정한다 (순수 함수)."""

    def test_zero_shorter_than_idle_seconds_is_not_idle(self):
        self.assertNotEqual(
            gpu_telemetry.gpu_activity_status(0.0, 100.0, 109.0), "유휴")

    def test_zero_for_idle_seconds_is_idle(self):
        self.assertEqual(
            gpu_telemetry.gpu_activity_status(0.0, 100.0, 110.0), "유휴")

    def test_positive_utilization_is_busy(self):
        self.assertEqual(
            gpu_telemetry.gpu_activity_status(12.0, None, 200.0), "사용 중")
        self.assertEqual(
            gpu_telemetry.gpu_activity_status(0.5, 100.0, 200.0), "사용 중")

    def test_single_zero_sample_has_no_zero_since_and_is_unknown(self):
        self.assertEqual(
            gpu_telemetry.gpu_activity_status(0.0, None, 200.0), "모름")

    def test_missing_utilization_is_unknown_not_zero(self):
        self.assertEqual(
            gpu_telemetry.gpu_activity_status(None, 100.0, 200.0), "모름")

    def test_stale_is_unknown_even_with_long_zero_run(self):
        self.assertEqual(
            gpu_telemetry.gpu_activity_status(0.0, 0.0, 999.0, stale=True), "모름")

    def test_browser_render_time_does_not_create_idle(self):
        # 수집 시각(collected)이 0% 시작과 같으면, 렌더 시각이 아무리 흘러도 유휴가 아니다.
        self.assertEqual(
            gpu_telemetry.gpu_activity_status(0.0, 500.0, 500.0), "모름")


class TestCarryZeroSince(unittest.TestCase):
    """수집 공백·새로고침이 유휴 시간을 거짓으로 늘리지 않아야 한다."""

    NOW = datetime(2026, 9, 19, 9, 0, 0, tzinfo=qm.KST)

    def _record(self, util):
        return {"gpus": {"4": {"power_draw": 100.0, "power_limit": 1000.0,
                               "utilization": util}}}

    def test_contiguous_zero_keeps_start(self):
        prev = {"epoch": self.NOW.timestamp(),
                "gpus": {"4": {"utilization": 0.0, "zero_since": 1.0}}}
        rec = gpu_telemetry.carry_zero_since(
            prev, self._record(0.0), self.NOW, 15)
        self.assertEqual(rec["gpus"]["4"]["zero_since"], 1.0)

    def test_stale_previous_sample_restarts_zero_run(self):
        prev = {"epoch": self.NOW.timestamp() - 3600,
                "gpus": {"4": {"utilization": 0.0, "zero_since": 1.0}}}
        rec = gpu_telemetry.carry_zero_since(
            prev, self._record(0.0), self.NOW, 15)
        self.assertAlmostEqual(rec["gpus"]["4"]["zero_since"], self.NOW.timestamp())

    def test_busy_clears_zero_since(self):
        prev = {"epoch": self.NOW.timestamp(),
                "gpus": {"4": {"utilization": 0.0, "zero_since": 1.0}}}
        rec = gpu_telemetry.carry_zero_since(
            prev, self._record(20.0), self.NOW, 15)
        self.assertIsNone(rec["gpus"]["4"]["zero_since"])


class TestGpuNotMistakenForIdle(unittest.TestCase):
    """누락·stale·실패는 유휴로 오인되지 않는다."""

    NOW = datetime(2026, 9, 19, 9, 0, 0, tzinfo=qm.KST)

    def test_missing_row_is_unknown(self):
        state = gpu_telemetry.telemetry_state({}, self.NOW, [4], 15)
        self.assertEqual(state["gpus"][0]["status"], "모름")

    def test_stale_cache_is_unknown_not_idle(self):
        cache = {"epoch": self.NOW.timestamp() - 3600, "ok": True,
                 "collected_at": "2026-09-19 08:00:00", "source": "ssh:nhn",
                 "gpus": {"4": {"utilization": 0.0, "power_draw": 100.0,
                                "power_limit": 1000.0, "zero_since": 1.0}}}
        state = gpu_telemetry.telemetry_state(cache, self.NOW, [4], 15)
        self.assertTrue(state["stale"])
        self.assertEqual(state["gpus"][0]["status"], "모름")
        self.assertNotEqual(state["gpus"][0]["status"], "유휴")

    def test_failed_collection_is_unknown_not_idle(self):
        state = gpu_telemetry.telemetry_state(
            {"epoch": self.NOW.timestamp(), "ok": False, "error": "ssh: timeout",
             "gpus": {}, "collected_at": "2026-09-19 09:00:00"}, self.NOW, [4], 15)
        self.assertFalse(state["available"])
        self.assertEqual(state["gpus"][0]["status"], "모름")


class TestPageLayout(unittest.TestCase):
    """HTML 문자열 수준에서 제거 대상과 표시 위치를 확인한다."""

    def test_removed_phrases_are_gone(self):
        for phrase in ("KV 캐시", "전력 한도",
                       "자정~관측 시작 구간 미포함", "관측 시작",
                       "서버 재시작 이후만 집계", "원격 NHN NEXGEM",
                       "캐시 표시", "요청마다 SSH 안 함",
                       "nvidia_gpu_exporter", "출처",
                       "공급 상태"):
            self.assertNotIn(phrase, qm.PAGE, phrase)

    def test_endpoint_url_is_not_rendered(self):
        self.assertNotIn("sv.url", qm.PAGE)

    def test_page_reference_time_remains_current_time(self):
        self.assertIn("'기준 시각 ' + s.now + ' KST'", qm.PAGE)
        self.assertNotIn("s.day_start", qm.PAGE)

    def test_running_and_waiting_moved_to_work_section(self):
        server_part = qm.PAGE.split("const gel = document.getElementById('gpus')")[0]
        self.assertNotIn("처리 중", server_part)
        self.assertNotIn("대기 중", server_part)
        work_part = qm.PAGE.split('id="workmeta"')[1]
        self.assertIn("처리 중", work_part)
        self.assertIn("대기 중", work_part)

    def test_top_cards_are_only_rate_and_today(self):
        server_part = qm.PAGE.split("const gel = document.getElementById('gpus')")[0]
        self.assertIn("초당 생성", server_part)
        self.assertIn("오늘 생성 토큰", server_part)
        self.assertNotIn("KV 캐시", server_part)

    def test_work_starved_is_a_title_badge_not_a_block(self):
        self.assertIn('id="badge"', qm.PAGE)
        self.assertIn("badge warn", qm.PAGE)
        self.assertNotIn('id="starvation"', qm.PAGE)
        self.assertNotIn('<div class="starved">', qm.PAGE)

    def test_gpu_section_has_three_columns_only(self):
        self.assertIn("['GPU','전력 사용','가동률','상태']", qm.PAGE)
        self.assertNotIn("전력 한도", qm.PAGE)

    def test_gpu_failure_keeps_table_without_explanation(self):
        self.assertNotIn("GPU 텔레메트리 도달 불가 ·", qm.PAGE)
        self.assertIn("도달 불가", qm.PAGE)


if __name__ == "__main__":
    unittest.main()
