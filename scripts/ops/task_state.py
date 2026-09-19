#!/usr/bin/env python3
"""One-shot delegated task lifecycle, kept in a store that outlives branches.

Measured defect this replaces. `talks/ops/tasks/*.md` had no path into the
supervisor queue at all, and the only "started" signal the monitor had was the
existence of a `chore/<name>-*` branch. Once a task merged and its branch was
deleted, the task read as `미착수` forever -- merge made the record worse, not
better. A failed run was indistinguishable from a successful one because the
card moved to done/ on both paths.

The store below is the authority. It is a single JSON file; each task records
one of four states and its full transition history. Completion is therefore a
durable record rather than a branch name that a cleanup can erase.

    pending -> running -> completed
                       -> failed
    failed  -> pending            (explicit `retry` only; never automatic)

Why `failed` cannot return to `running` directly: an automatic re-dispatch of a
failing task is the "broken item fires every tick" failure the queue already
paid for once. Retry is a deliberate act by the orchestrator and goes through
`retry`, which is what the store records.

Every transition is validated. An illegal one raises rather than silently
overwriting a terminal state, so a wrapper that races the supervisor cannot
turn `completed` back into `running`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = PROJECT_ROOT / "talks" / "ops" / "task_state.json"
KST = timezone(timedelta(hours=9))

STATES = ("pending", "running", "completed", "failed")

#: Allowed edges. Terminal states have no outgoing edge except the explicit
#: `failed -> pending` retry, which is intentional. `pending -> completed`
#: exists because merge evidence can surface a task this store had already
#: recorded pending; the work landed outside the queue.
ALLOWED: dict[str, set[str]] = {
    "pending": {"running", "completed", "failed"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": {"pending"},
}

#: A task with no record yet may only enter these states.
INITIAL_STATES = {"pending", "running"}


class InvalidTransition(ValueError):
    """Raised instead of silently overwriting a task's terminal state."""


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def load(path: Path | str = DEFAULT_STORE) -> dict:
    """Read the store. A missing or corrupt file is an empty store, not a crash."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save(path: Path | str, store: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def state_of(store: dict, name: str) -> str | None:
    rec = store.get(name) or {}
    return rec.get("state")


def transition(store: dict, name: str, new_state: str, *,
               reason: str | None = None, **fields) -> dict:
    """Move `name` to `new_state`, validating the edge and appending history.

    `fields` are written onto the record (e.g. branch, commit, pid, error).
    They never decide the transition, so a late writer must use `update_fields`
    rather than call this again.
    """
    if new_state not in STATES:
        raise InvalidTransition(f"unknown state {new_state!r}")
    rec = store.setdefault(name, {"name": name, "history": []})
    old = rec.get("state")
    if old is None:
        if new_state not in INITIAL_STATES:
            raise InvalidTransition(
                f"{name}: 새 기록은 {new_state!r} 로 시작할 수 없다")
    elif new_state not in ALLOWED.get(old, set()):
        raise InvalidTransition(f"{name}: {old} -> {new_state} 는 허용되지 않는다")

    rec["state"] = new_state
    rec["updated"] = now_kst()
    if reason:
        rec["reason"] = reason
    rec.update(fields)
    rec.setdefault("history", []).append(
        {"t": rec["updated"], "from": old, "to": new_state})
    return rec


def update_fields(store: dict, name: str, **fields) -> dict:
    """Attach metadata without touching the state or its history.

    Needed because the pid is known only after the process is spawned, by which
    time a fast task may already have completed. Recording the pid must not
    resurrect a completed task.
    """
    rec = store.setdefault(name, {"name": name, "history": []})
    rec.update(fields)
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("show", help="print the store as JSON")
    args = ap.parse_args(argv)

    store = load(args.store)
    if args.action == "show":
        print(json.dumps(store, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
