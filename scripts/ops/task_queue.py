#!/usr/bin/env python3
"""Wire `talks/ops/tasks/*.md` into the supervisor queue, exactly once.

Measured defect this replaces. A task card was visible in the monitor and
nowhere else: the supervisor only ever read `talks/ops/queue/*.json` and
`talks/ops/recurring/*.json`, and nothing wrote a queue entry from a task card.
Writing the card was therefore indistinguishable from not writing it.

Two facts decide whether a task may enter the queue.

  already merged  A commit in the main history names the task
                  (`chore(<name>)`, the message routine_opencode.sh writes).
                  That is durable evidence the work landed, so the task is
                  recorded completed and never enqueued. This is what the old
                  branch-existence check got wrong: a merged task whose branch
                  was deleted read as `미착수`.
  not yet done    No merge evidence and no terminal store state. The store
                  records it `pending` and one queue card is written.

Calling `sync` twice writes one card, not two -- the queue file itself is the
idempotence guard, and the store keeps a `running` task from being re-queued
after a supervisor restart.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_state  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPS = PROJECT_ROOT / "talks" / "ops"
TASKS = OPS / "tasks"
QUEUE = OPS / "queue"
DEFAULT_STORE = task_state.DEFAULT_STORE

#: The four resource classes the scheduler reasons about. They are not a
#: priority order; `login_cpu` and `remote_llm` may run together, while
#: `exclusive_gpu` demands the node to itself.
RESOURCES = ("remote_llm", "slurm", "login_cpu", "exclusive_gpu")

#: Delegate-task commands call the local model, so they occupy remote_llm.
TASK_RESOURCE = "remote_llm"

#: Tokens that identify a command's resource when the spec does not say.
_LLM_TOKENS = ("opencode", "seat_thinking", "lit_digest",
               "routine_open_questions", "supervisor.py")
_SLURM_TOKENS = ("sbatch", "srun", "squeue")
_GPU_TOKENS = ("evaluate_pure", "dump_branch_margins", "train", "council.py")

MERGE_RE = re.compile(r"chore\(([A-Za-z0-9_.-]+)\)")


def classify(cmd: str) -> str:
    """Best-effort resource class for a command string."""
    low = (cmd or "").lower()
    if any(t in low for t in _SLURM_TOKENS):
        return "slurm"
    if any(t in low for t in _GPU_TOKENS):
        return "exclusive_gpu"
    if any(t in low for t in _LLM_TOKENS):
        return "remote_llm"
    return "login_cpu"


def resource_of(spec: dict) -> str:
    """A spec may declare `resource`; otherwise infer it from its command."""
    declared = spec.get("resource")
    if declared in RESOURCES:
        return declared
    return classify(spec.get("cmd", ""))


def merged_task_names(log_text: str) -> set[str]:
    """Task names named by `chore(<name>)` commits in a `git log` dump."""
    return {m.group(1) for m in MERGE_RE.finditer(log_text or "")}


def git_log(root: Path = PROJECT_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "log", "--format=%s", "main"],
            capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return out.stdout
    except Exception:  # noqa: BLE001 - git absent means unknown, not done
        pass
    return ""


def task_spec(name: str, task_path: Path) -> dict:
    return {
        "kind": "delegate",
        "label": name,
        "task": str(task_path),
        "resource": TASK_RESOURCE,
        "worktree": f"chore-{name}",
        "cmd": f"bash scripts/ops/routine_opencode.sh {task_path}",
    }


def enqueue(store: dict, name: str, queue_dir: Path,
            task_path: Path | None = None) -> Path:
    """Write the one queue card for `name`; it must not already exist."""
    qpath = Path(queue_dir) / f"task_{name}.json"
    if qpath.exists():
        return qpath
    path = task_path or (TASKS / f"{name}.md")
    qpath.parent.mkdir(parents=True, exist_ok=True)
    qpath.write_text(json.dumps(task_spec(name, path), ensure_ascii=False, indent=1)
                     + "\n", encoding="utf-8")
    return qpath


def sync_tasks(store: dict, tasks_dir: Path | str = TASKS,
               queue_dir: Path | str = QUEUE, merged: set[str] | None = None) -> list[str]:
    """Reconcile task cards into the queue. Returns names newly enqueued."""
    tasks_dir = Path(tasks_dir)
    queue_dir = Path(queue_dir)
    if merged is None:
        merged = merged_task_names(git_log())
    enqueued: list[str] = []
    for path in sorted(tasks_dir.glob("*.md")):
        name = path.stem
        state = task_state.state_of(store, name)
        if state in ("running", "completed", "failed"):
            continue  # in flight or terminal: never auto-enqueue
        if name in merged:
            # A task merged before this store existed has no record at all.
            # Route it through pending so the history is a legal edge chain
            # rather than starting a task life in a terminal state.
            if state is None:
                task_state.transition(store, name, "pending",
                                      reason="병합 커밋 발견 — 완료로 기록")
            task_state.transition(store, name, "completed",
                                  reason="main 이력에 병합 커밋이 있다",
                                  evidence="git-log:chore(%s)" % name)
            continue
        if state is None:
            task_state.transition(store, name, "pending", reason="미착수")
        if not (queue_dir / f"task_{name}.json").exists():
            enqueue(store, name, queue_dir, path)
            enqueued.append(name)
    return enqueued


def retry(store: dict, name: str, tasks_dir: Path | str = TASKS,
          queue_dir: Path | str = QUEUE) -> Path:
    """Explicit retry: failed (or pending) -> pending, then one queue card."""
    state = task_state.state_of(store, name)
    if state not in ("failed", "pending"):
        raise ValueError(f"{name}: state={state!r} 는 재시도 대상이 아니다")
    if state == "failed":
        task_state.transition(store, name, "pending", reason="explicit retry")
    return enqueue(store, name, Path(queue_dir), Path(tasks_dir) / f"{name}.md")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default=str(DEFAULT_STORE))
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--queue", default=str(QUEUE))
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("sync")
    r = sub.add_parser("retry")
    r.add_argument("name")
    args = ap.parse_args(argv)

    store = task_state.load(args.store)
    if args.action == "sync":
        added = sync_tasks(store, args.tasks, args.queue)
        task_state.save(args.store, store)
        print("큐 추가: " + (", ".join(added) if added else "없음"))
    elif args.action == "retry":
        retry(store, args.name, args.tasks, args.queue)
        task_state.save(args.store, store)
        print(f"재시도 큐: {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
