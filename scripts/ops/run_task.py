#!/usr/bin/env python3
"""Run one queued delegated command and record how it ended.

The supervisor used to spawn the command itself and infer the outcome from a
`Popen` return code held only in memory. A restart in the middle of a chore
therefore lost the outcome entirely: the task stayed `running` forever, and the
only way to clear it was to edit state by hand. Moving the command into a child
process whose last act is writing the terminal state means the record survives
the supervisor.

This file is that child. On exit:

  rc == 0   task -> completed
  rc != 0   task -> failed, with the return code and log tail kept

Recurring chores have no task record and are handled by the supervisor's own
reaper; `--store` is simply unused for them.

    .venv/bin/python scripts/ops/run_task.py --item talks/ops/done/task_x.json \
        --store talks/ops/task_state.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_state  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tail(path: Path | None, limit: int = 400) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


def execute(spec: dict, store_path: str | Path, cwd: Path | str = PROJECT_ROOT,
            log_path: Path | str | None = None) -> int:
    """Run `spec['cmd']`, record the terminal state, return the return code.

    Exposed rather than buried in `main` so the transition can be tested
    without a supervisor: a temp store, `cmd=true`, and one assertion.
    """
    cwd = Path(cwd)
    cmd = spec.get("cmd")
    if not cmd:
        _finish(spec, store_path, 2, "spec 에 cmd 가 없다", log_path)
        return 2
    rc = subprocess.run(cmd, shell=True, cwd=str(cwd)).returncode
    if spec.get("kind") == "delegate" or spec.get("task"):
        _finish(spec, store_path, rc, None, log_path)
    return rc


def _finish(spec: dict, store_path: str | Path, rc: int,
            error: str | None, log_path: Path | str | None) -> None:
    name = spec.get("label") or spec.get("task") or spec.get("name")
    if not name:
        return
    store = task_state.load(store_path)
    fields = {"rc": rc, "log": str(log_path) if log_path else spec.get("log")}
    if rc == 0:
        task_state.transition(store, name, "completed", **fields)
    else:
        reason = error or f"rc={rc}"
        fields["error"] = error or _tail(Path(log_path) if log_path else None)
        task_state.transition(store, name, "failed", reason=reason, **fields)
    task_state.save(store_path, store)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--item", required=True, help="큐 카드 JSON 경로")
    ap.add_argument("--store", default=str(task_state.DEFAULT_STORE))
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)

    try:
        spec = json.loads(Path(args.item).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"큐 카드를 읽을 수 없다: {exc}", file=sys.stderr)
        return 2
    return execute(spec, args.store, PROJECT_ROOT, args.log)


if __name__ == "__main__":
    sys.exit(main())
