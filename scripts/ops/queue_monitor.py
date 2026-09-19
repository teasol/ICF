#!/usr/bin/env python3
"""딥시크 서버가 처리 중인 큐와 프로젝트가 시킬 일을 한 페이지에서 본다.

    .venv/bin/python scripts/ops/queue_monitor.py --host 0.0.0.0 --port 8899

읽는 출처 (셋 다 없을 수 있고, 없으면 "기록 없음"이다):

  - vLLM /metrics          `scripts/ops/llm_env.py`의 엔드포인트에서 호스트·포트만
                           떼어 `/metrics`를 붙여 Prometheus 텍스트를 받는다.
                           지금 처리 중·대기 중 요청, KV 캐시 사용률, 누적 토큰과
                           직전 폴링 대비 초당 생성 토큰을 보여준다.
  - talks/ops/recurring/*.json  주기 잡무 명세, 자원 종류, 다음 실행까지 남은 분.
  - talks/ops/tasks/*.md + talks/ops/task_state.json
                               위임 작업의 수명주기(`pending/running/completed/failed`).
                               브랜치 존재가 아니라 영속 상태 기록이 정본이다.
  - talks/ops/ticks.jsonl       최근 실행 이력(마지막 20줄).
  - talks/ops/failures.jsonl    실패 이력(마지막 10줄, 붉게).
  - talks/ops/heartbeat         감독자 심박(수정 시각). 5분 넘으면 정지 의심.

서버에 닿지 않으면 숫자를 0으로 채우지 않고 "서버 도달 불가"로 표시한다.
주소는 코드에 적지 않는다 -- `llm_env`가 정한 것을 그대로 쓴다(D-053).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import llm_env  # noqa: E402
import task_queue  # noqa: E402
import task_state  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPS = PROJECT_ROOT / "talks" / "ops"
RECURRING = OPS / "recurring"
TASKS = OPS / "tasks"
QUEUE = OPS / "queue"
STORE = OPS / "task_state.json"
TICKS = OPS / "ticks.jsonl"
FAILURES = OPS / "failures.jsonl"
HEARTBEAT = OPS / "heartbeat"
KST = timezone(timedelta(hours=9))

#: 심박이 이 시간(초)보다 오래됐으면 감독자를 정지로 본다.
HEARTBEAT_STALE_SECONDS = 300
TICK_TAIL = 20
FAILURE_TAIL = 10
FETCH_TIMEOUT = 3.0

#: 직전 폴링의 (monotonic 시각, 누적 생성 토큰). 초당 토큰을 내는 데만 쓴다.
_LAST_TOKENS: dict[str, tuple[float, float]] = {}
_LAST_TOKENS_LOCK = threading.Lock()


# --------------------------------------------------------------------------
# vLLM Prometheus
# --------------------------------------------------------------------------

def metrics_urls() -> list[str]:
    """llm_env가 정한 채팅 엔드포인트에서 `/metrics` 주소만 만든다."""
    urls = []
    for endpoint in llm_env.endpoints():
        parsed = urlparse(endpoint)
        if parsed.scheme and parsed.netloc:
            urls.append(f"{parsed.scheme}://{parsed.netloc}/metrics")
        else:
            urls.append(endpoint.rstrip("/") + "/metrics")
    return urls


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus 텍스트에서 지표별 값을 뽑는다.

    라벨이 여럿이면(엔진·모델별) 합산한다. `vllm:num_requests_waiting`과
    `..._by_reason`은 이름이 다르므로 정확히 일치하는 것만 센다. HELP/TYPE
    주석, 값 없는 줄, 숫자로 안 읽히는 줄은 조용히 건너뛴다.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name_end = len(line)
        for stop in ("{", " "):
            pos = line.find(stop)
            if 0 <= pos < name_end:
                name_end = pos
        name = line[:name_end]
        if not name:
            continue
        rest = line[name_end:]
        if rest.startswith("{"):
            close = rest.find("}")
            if close < 0:
                continue
            rest = rest[close + 1:]
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + value
    return out


def _read_metric(metrics: dict[str, float], name: str) -> float | None:
    return metrics.get(name)


def server_metrics() -> dict[str, Any]:
    """살아 있는 첫 엔드포인트의 /metrics를 읽어 상태로 만든다.

    닿지 않으면 수치를 0으로 채우지 않는다 -- reachable=False와 이유만 남긴다.
    """
    last_error = "엔드포인트 없음"
    for url in metrics_urls():
        try:
            req = urllib.request.Request(url, headers={"Accept": "text/plain"})
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - 도달 불가는 상태이지 예외가 아니다
            last_error = f"{url}: {exc}"
            continue

        m = parse_prometheus(text)
        gen = _read_metric(m, "vllm:generation_tokens_total")
        per_sec = None
        if gen is not None:
            now = time.monotonic()
            with _LAST_TOKENS_LOCK:
                prev = _LAST_TOKENS.get(url)
                _LAST_TOKENS[url] = (now, gen)
            if prev is not None:
                dt = now - prev[0]
                if dt > 0 and gen >= prev[1]:
                    per_sec = (gen - prev[1]) / dt
        return {
            "reachable": True,
            "url": url,
            "error": None,
            "running": _read_metric(m, "vllm:num_requests_running"),
            "waiting": _read_metric(m, "vllm:num_requests_waiting"),
            "waiting_capacity": _read_metric(
                m, "vllm:num_requests_waiting_by_reason"),
            "kv_cache_perc": _read_metric(m, "vllm:gpu_cache_usage_perc"),
            "generation_tokens_total": gen,
            "prompt_tokens_total": _read_metric(m, "vllm:prompt_tokens_total"),
            "generation_tokens_per_sec": per_sec,
        }
    return {"reachable": False, "url": None, "error": last_error,
            "running": None, "waiting": None, "waiting_capacity": None,
            "kv_cache_perc": None, "generation_tokens_total": None,
            "prompt_tokens_total": None, "generation_tokens_per_sec": None}


# --------------------------------------------------------------------------
# 파일에서 읽는 부분
# --------------------------------------------------------------------------

def _parse_kst(stamp: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(stamp.strip(), fmt).replace(tzinfo=KST)
        except (ValueError, AttributeError):
            continue
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """깨진 줄(잘린 마지막 줄 포함)은 버리고 나머지만 돌려준다."""
    if not path.exists():
        return []
    records = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return records


def last_runs(ticks: list[dict[str, Any]]) -> dict[str, datetime]:
    """tick의 dispatch 기록에서 label별 마지막 발사 시각을 모은다."""
    latest: dict[str, datetime] = {}
    for tick in ticks:
        when = _parse_kst(str(tick.get("t", "")))
        if when is None:
            continue
        for rec in tick.get("dispatch") or []:
            label = rec.get("launch")
            if not label:
                continue
            if label not in latest or when > latest[label]:
                latest[label] = when
    return latest


def read_recurring(now: datetime) -> list[dict[str, Any]]:
    runs = last_runs(_read_jsonl(TICKS))
    items = []
    for path in sorted(RECURRING.glob("*.json")):
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        label = spec.get("label", path.stem)
        period = float(spec.get("every_minutes") or 0)
        last = runs.get(label)
        remaining = None
        status = "미실행"
        if last is not None and period > 0:
            remaining = period - (now - last).total_seconds() / 60.0
            if remaining <= 0:
                status = f"지연 {int(-remaining)}분"
            else:
                status = f"다음 {int(remaining)}분"
        items.append({
            "label": label,
            "cmd": spec.get("cmd"),
            "every_minutes": spec.get("every_minutes"),
            "parallel": bool(spec.get("parallel")),
            "resource": task_queue.resource_of(spec),
            "last_run": last.strftime("%Y-%m-%d %H:%M:%S") if last else None,
            "remaining_minutes": remaining,
            "status": status,
        })
    return items


def read_tasks(store: dict | None = None) -> list[dict[str, Any]]:
    """One row per task card, its lifecycle state taken from the persistent store.

    The old version inferred `started` from a `chore/<name>-*` branch. Merging a
    task and deleting the branch made a finished task read `미착수` again, and a
    task that never ran was indistinguishable from one that ran and failed. The
    store now carries `pending / running / completed / failed` plus the last
    error, so the table says what actually happened.
    """
    store = store if store is not None else task_state.load(STORE)
    items = []
    for path in sorted(TASKS.glob("*.md")):
        name = path.stem
        rec = store.get(name) or {}
        state = rec.get("state") or "미착수"
        items.append({
            "name": name,
            "state": state,
            "resource": rec.get("resource") or task_queue.TASK_RESOURCE,
            "branch": rec.get("branch"),
            "commit": rec.get("commit"),
            "rc": rec.get("rc"),
            "error": rec.get("reason") or rec.get("error"),
            "updated": rec.get("updated"),
        })
    return items


def read_heartbeat(now: datetime) -> dict[str, Any]:
    if not HEARTBEAT.exists():
        return {"present": False, "mtime": None, "age_seconds": None,
                "alive": False}
    mtime = datetime.fromtimestamp(HEARTBEAT.stat().st_mtime, tz=KST)
    age = (now - mtime).total_seconds()
    return {"present": True, "mtime": mtime.strftime("%Y-%m-%d %H:%M:%S"),
            "age_seconds": age, "alive": age <= HEARTBEAT_STALE_SECONDS}


def read_ticks() -> list[dict[str, Any]]:
    recent = _read_jsonl(TICKS)[-TICK_TAIL:]
    slim = []
    for tick in recent:
        launches = [r.get("launch") for r in (tick.get("dispatch") or [])
                    if r.get("launch")]
        active = tick.get("active_resources") or {}
        slim.append({
            "t": tick.get("t"),
            "busy": bool(tick.get("busy")),
            "queue": tick.get("queue"),
            "resources": ", ".join(f"{k}:{v}" for k, v in active.items()) or "–",
            "starved": bool(tick.get("work_starved")),
            "launches": launches,
        })
    return slim


def read_failures() -> list[dict[str, Any]]:
    recent = _read_jsonl(FAILURES)[-FAILURE_TAIL:]
    return [{"t": f.get("t"), "label": f.get("label"),
             "rc": f.get("rc"), "log": f.get("log")} for f in recent]


def starvation(metrics: dict[str, Any], tasks: list[dict[str, Any]],
               recurring: list[dict[str, Any]]) -> dict[str, Any]:
    """Whether the local server is idle for lack of work.

    The distinction the old monitor could not make: a recurring CPU test every
    few minutes kept the node looking "busy" while no request ever reached the
    model. Starvation is claimed only when the server is reachable and idle,
    nothing remote_llm is running or queued, and no task is pending.
    """
    if not metrics.get("reachable"):
        return {"state": "모름", "reason": "서버 도달 불가 — 유휴 여부를 판단할 수 없다"}
    running = metrics.get("running")
    waiting = metrics.get("waiting")
    if running is None or waiting is None:
        return {"state": "모름", "reason": "요청 수 지표 없음"}
    remote_running = [t["name"] for t in tasks if t["state"] == "running"]
    pending = [t["name"] for t in tasks if t["state"] == "pending"]
    remote_due = [r["label"] for r in recurring
                  if r.get("resource") == "remote_llm" and r.get("remaining_minutes") is not None
                  and r["remaining_minutes"] <= 0]
    idle = running == 0 and waiting == 0
    starved = idle and not remote_running and not pending and not remote_due
    if starved:
        reason = "서버가 놀고 있는데 remote_llm 작업도 대기 작업도 없다 — 오케스트레이터 공급 필요"
    elif not idle:
        reason = f"서버가 일하고 있다 (running={running:g}, waiting={waiting:g})"
    elif remote_due:
        reason = "곧 실행될 주기 잡무가 있다: " + ", ".join(remote_due)
    elif pending:
        reason = "미착수 작업이 큐를 기다린다: " + ", ".join(pending)
    else:
        reason = "remote_llm 작업이 실행 중이다: " + ", ".join(remote_running)
    return {"state": "work-starved" if starved else "정상", "reason": reason,
            "running": running, "waiting": waiting}


def build_state(metrics: dict[str, Any] | None = None,
                now: datetime | None = None) -> dict[str, Any]:
    """페이지가 그릴 모든 값. metrics를 주면 서버 폴링을 건너뛴다(시험용)."""
    now = now or datetime.now(KST)
    if metrics is None:
        metrics = server_metrics()
    failures = read_failures()
    tasks = read_tasks()
    recurring = read_recurring(now)
    return {
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "server": metrics,
        "recurring": recurring,
        "tasks": tasks,
        "ticks": read_ticks(),
        "failures": failures,
        "failures_total": len(_read_jsonl(FAILURES)),
        "heartbeat": read_heartbeat(now),
        "starvation": starvation(metrics, tasks, recurring),
    }


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>딥시크 큐 모니터</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
         background:#11151c; color:#d7dde5; margin:0; padding:20px; }
  h1 { font-size:18px; margin:0 0 4px; }
  h2 { font-size:14px; margin:24px 0 8px; color:#8fb4de; }
  .now { color:#6b7686; font-size:12px; margin-bottom:12px; }
  .grid { display:flex; flex-wrap:wrap; gap:12px; }
  .card { background:#1a2029; border:1px solid #2a323e; border-radius:8px;
          padding:12px 16px; min-width:120px; }
  .card .k { font-size:11px; color:#8892a0; }
  .card .v { font-size:24px; font-weight:600; margin-top:4px; }
  .down { background:#5c1a1a; border:1px solid #a33; color:#ffd7d7;
          padding:14px 18px; border-radius:8px; font-size:22px; font-weight:700; }
  .down .sub { font-size:12px; font-weight:400; color:#f0b8b8; margin-top:6px; }
  .starved { background:#4a3a12; border:1px solid #c9a227; color:#ffe9a8;
             padding:12px 16px; border-radius:8px; margin-top:10px; font-weight:700; }
  .starved .sub { font-size:12px; font-weight:400; color:#e8d59a; margin-top:6px; }
  .pill.running { background:#1d2f4a; color:#9cc4ff; }
  .pill.completed { background:#1d3a24; color:#8fe0a0; }
  .pill.failed { background:#3a1212; color:#ffb3b3; }
  .pill.pending { background:#3a341d; color:#e0d08f; }
  .pill.unknown { background:#2a323e; color:#aab4c0; }
  table { border-collapse:collapse; width:100%%; font-size:12px; }
  th, td { text-align:left; padding:4px 10px 4px 0; border-bottom:1px solid #232a35; }
  th { color:#8892a0; font-weight:500; }
  .muted { color:#6b7686; }
  .stale { color:#ff6b6b; font-weight:700; }
  .ok { color:#7ec96f; }
  .fail { background:#3a1212; color:#ffb3b3; padding:2px 8px; border-radius:4px; }
  tr.fail td { background:#3a1212; color:#ffb3b3; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; }
  .pill.started { background:#1d3a24; color:#8fe0a0; }
  .pill.todo { background:#3a341d; color:#e0d08f; }
  code { color:#c8d3e0; }
</style>
</head>
<body>
<h1>딥시크 큐 모니터</h1>
<div class="now" id="now">불러오는 중…</div>
<div id="server"></div>
<div id="starvation"></div>

<h2>프로젝트가 딥시크에 시킬 것 — 주기 잡무</h2>
<div id="recurring"></div>

<h2>위임 작업 (talks/ops/tasks)</h2>
<div id="tasks"></div>

<h2>방금 끝난 것 — 최근 실행 (ticks.jsonl 마지막 20줄)</h2>
<div id="ticks"></div>

<h2>실패 (failures.jsonl 마지막 10줄)</h2>
<div id="failures"></div>

<h2>감독자 심박 (heartbeat)</h2>
<div id="heartbeat"></div>

<script>
const EMPTY = '기록 없음';
function esc(s){return String(s==null?'':s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function num(x, d){return (x===null||x===undefined||isNaN(x))?'모름':Number(x).toFixed(d);}
function render(s){
  document.getElementById('now').textContent = '기준 시각 ' + s.now + ' KST';

  const sv = s.server, el = document.getElementById('server');
  if(!sv.reachable){
    el.innerHTML = '<div class="down">서버 도달 불가'
      + '<div class="sub">' + esc(sv.error || '') + '</div></div>';
  } else {
    const rate = (sv.generation_tokens_per_sec===null)?
        '<span class="muted">측정 전</span>' : num(sv.generation_tokens_per_sec,1)+' tok/s';
    el.innerHTML = '<div class="grid">'
      + card('처리 중', num(sv.running,0))
      + card('대기 중', num(sv.waiting,0))
      + card('KV 캐시', (sv.kv_cache_perc===null)?'–':num(sv.kv_cache_perc*100,1)+'%')
      + card('초당 생성', rate)
      + card('누적 생성 토큰', (sv.generation_tokens_total===null)?'–':Math.round(sv.generation_tokens_total).toLocaleString())
      + card('누적 프롬프트 토큰', (sv.prompt_tokens_total===null)?'–':Math.round(sv.prompt_tokens_total).toLocaleString())
      + '</div><div class="muted" style="margin-top:8px">'
      + esc(sv.url) + '</div>';
  }

  const st = s.starvation || {state:'모름', reason:''}, stel = document.getElementById('starvation');
  if(st.state === 'work-starved'){
    stel.innerHTML = '<div class="starved">⚠ work-starved'
      + '<div class="sub">' + esc(st.reason) + '</div></div>';
  } else {
    stel.innerHTML = '<div class="muted" style="margin-top:8px">'
      + '공급 상태: ' + esc(st.state) + ' · ' + esc(st.reason) + '</div>';
  }

  document.getElementById('recurring').innerHTML = table(
    ['잡무','자원','주기(분)','마지막 실행','상태','명령'],
    s.recurring.map(r=>[esc(r.label), '<code>'+esc(r.resource)+'</code>', esc(r.every_minutes),
      esc(r.last_run||'미실행'), esc(r.status), '<code>'+esc(r.cmd)+'</code>']));

  function statePill(state){
    const cls = (state==='running'||state==='completed'||state==='failed'||state==='pending')
      ? state : 'unknown';
    return '<span class="pill '+cls+'">'+esc(state)+'</span>';
  }
  document.getElementById('tasks').innerHTML = table(
    ['작업','상태','자원','브랜치','최근 오류'],
    s.tasks.map(t=>[esc(t.name), statePill(t.state), '<code>'+esc(t.resource)+'</code>',
      t.branch?('<code>'+esc(t.branch)+'</code>'):'<span class="muted">없음</span>',
      t.error?('<span class="fail">'+esc(t.error)+'</span>'):'<span class="muted">–</span>']));

  document.getElementById('ticks').innerHTML = table(
    ['시각','가동','큐','자원별 실행','공급','발사'],
    s.ticks.map(t=>[esc(t.t), t.busy?'<span class="ok">예</span>':'<span class="muted">아니오</span>',
      esc(t.queue), esc(t.resources),
      t.starved?'<span class="stale">work-starved</span>':'<span class="muted">정상</span>',
      esc((t.launches||[]).join(', ')||'–')]));

  const f = document.getElementById('failures');
  if(!s.failures.length){ f.innerHTML = '<span class="muted">'+EMPTY+'</span>'; }
  else {
    f.innerHTML = '<table><tr><th>시각</th><th>잡무</th><th>rc</th><th>로그</th></tr>'
      + s.failures.map(x=>'<tr class="fail"><td>'+esc(x.t)+'</td><td>'+esc(x.label)
        +'</td><td>'+esc(x.rc)+'</td><td>'+esc(x.log)+'</td></tr>').join('')
      + '</table>';
  }

  const hb = s.heartbeat, h = document.getElementById('heartbeat');
  if(!hb.present){ h.innerHTML = '<span class="muted">'+EMPTY+'</span>'; }
  else if(hb.alive){ h.innerHTML = '<span class="ok">살아 있음</span> · '+esc(hb.mtime)
      +' <span class="muted">('+Math.round(hb.age_seconds)+'초 전)</span>'; }
  else { h.innerHTML = '<span class="stale">감독자 정지 의심</span> · '+esc(hb.mtime)
      +' <span class="muted">('+Math.round(hb.age_seconds/60)+'분 전)</span>'; }
}
function card(k,v){return '<div class="card"><div class="k">'+k+'</div><div class="v">'+v+'</div></div>';}
function table(head, rows){
  if(!rows.length) return '<span class="muted">'+EMPTY+'</span>';
  let h = '<table><tr>' + head.map(x=>'<th>'+x+'</th>').join('') + '</tr>';
  for(const r of rows){ h += '<tr>' + r.map(c=>'<td>'+c+'</td>').join('') + '</tr>'; }
  return h + '</table>';
}
async function tick(){
  try{
    const r = await fetch('/api/state', {cache:'no-store'});
    render(await r.json());
  }catch(e){
    document.getElementById('server').innerHTML =
      '<div class="down">API 도달 불가<div class="sub">'+esc(e)+'</div></div>';
  }
}
tick();
setInterval(tick, 2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler의 이름
        path = self.path.split("?", 1)[0]
        if path == "/api/state":
            body = json.dumps(build_state(), ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        elif path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def log_message(self, fmt: str, *args: Any) -> None:
        # 2초마다 들어오는 정상 요청으로 로그를 채우지 않는다.
        if not self.command or self.path.startswith("/api/state"):
            return
        super().log_message(fmt, *args)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"큐 모니터: http://{args.host}:{args.port}/  "
          f"(모델 {llm_env.describe()})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
