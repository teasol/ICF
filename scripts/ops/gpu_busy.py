"""Which of our processes are actually holding the experiment GPU.

WORK_PATTERNS listed experiment scripts by name, which means every new script
had to be added to it or the node looked idle while an experiment ran -- and an
idle-looking node gets dispatched over. branch_redundancy.py was the first to
fall through.

Asking the driver removes the enumeration. A process counts as ours when it
holds memory on the experiment GPU and its command line is inside this
repository; the vLLM server sits on the same GPU and is excluded by name, since
it is a service rather than an experiment.

  python scripts/ops/gpu_busy.py [--gpu 4]   -> prints count and names, exit 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_MARKERS = ("vllm", "sglang", "tensorrt")


def _in_repo(pid: str, cmd: str) -> bool:
    """Command lines are often relative, so the repo is identified by cwd."""
    if str(PROJECT_ROOT) in cmd:
        return True
    try:
        return Path(f"/proc/{pid}/cwd").resolve() == PROJECT_ROOT
    except OSError:
        return False


def _cmdline(pid: str) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace")
    except OSError:
        return ""


def experiment_pids(gpu: int) -> list[tuple[str, str]]:
    try:
        uuid = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(gpu)],
            capture_output=True, text=True, timeout=30).stdout.strip()
        rows = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []

    found = []
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 2 or parts[1] != uuid:
            continue
        cmd = _cmdline(parts[0])
        if not cmd or not _in_repo(parts[0], cmd):
            continue
        if any(m in cmd.lower() for m in SERVICE_MARKERS):
            continue
        # The vLLM workers rename themselves (VLLM_PROCESS_NAME_PREFIX), so a
        # bare "python" with this cwd is the server, not an experiment. Require
        # a named script under scripts/ -- that is what we launch.
        script = next((tok for tok in cmd.split()
                       if tok.endswith(".py") and "scripts/" in tok), None)
        if script is None or "scripts/ops/" in script or "council.py" in script:
            continue
        found.append((parts[0], Path(script).name))
    return found


def repo_experiment_procs() -> list[tuple[str, str]]:
    """Our python processes running an experiment script, GPU-allocated or not.

    The driver list only shows a process once it has allocated device memory,
    and these scripts spend their first minutes reading features off disk. A
    process that is about to take the GPU must not read as idle.

    ops/ and council scripts are excluded: those are the chores and rounds,
    counted separately, and treating them as experiments would make the node
    look busy whenever the supervisor ticks.
    """
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        cmd = _cmdline(proc.name)
        if not cmd or not _in_repo(proc.name, cmd):
            continue
        script = next((tok for tok in cmd.split()
                       if tok.endswith(".py") and "scripts/" in tok), None)
        if script is None:
            continue
        # ops/ is the supervisor and chores, council.py is a round -- both are
        # counted elsewhere. Excluding self is not paranoia: a checker whose own
        # command line matches its own pattern has killed this shell three times.
        if "scripts/ops/" in script or "council.py" in script:
            continue
        if proc.name == str(os.getpid()):
            continue
        found.append((proc.name, Path(script).name))
    return found


def busy(gpu: int) -> list[tuple[str, str]]:
    seen, out = set(), []
    for pid, name in experiment_pids(gpu) + repo_experiment_procs():
        if pid not in seen:
            seen.add(pid)
            out.append((pid, name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=4)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    found = busy(args.gpu)
    names = sorted({n for _, n in found})
    if not args.quiet:
        print(f"{len(names)}건 · " + ", ".join(names) if names else "0건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
