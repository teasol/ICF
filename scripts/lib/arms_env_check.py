#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ARMS = ["v114", "v115", "v116", "v117", "v118", "v119", "v120"]
TASK = "cptac_lscc/ARID1A_mutation"


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def git_show(repo, base, path):
    proc = subprocess.run(
        ["git", "-C", repo, "show", base + ":" + path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "git show failed for %s: %s" % (path, proc.stderr.decode("utf-8", "replace"))
        )
    return proc.stdout


def materialize_original(repo, base, arm):
    rel = "scripts/.parity_orig_eval_%s.sh" % arm
    full = os.path.join(repo, rel)
    content = git_show(repo, base, "scripts/eval_%s.sh" % arm)
    with open(full, "wb") as fh:
        fh.write(content)
    os.chmod(full, 0o755)
    return rel


def run_capture(repo, script_rel, arm, capture_out, env):
    env = dict(env)
    env["CAPTURE_OUT"] = capture_out
    subprocess.run(
        ["bash", script_rel, "0", "parity_%s" % arm, TASK],
        cwd=repo, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def sort_file(path):
    env = {
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }
    proc = subprocess.run(["sort", path], env=env, stdout=subprocess.PIPE)
    return proc.stdout


def argv_count(data):
    return sum(1 for line in data.split(b"\n") if line.startswith(b"ARGV "))


def first_diff_line(orig_bytes, new_bytes):
    left = orig_bytes.split(b"\n")
    right = new_bytes.split(b"\n")
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else b""
        b = right[i] if i < len(right) else b""
        if a != b:
            return (a if a else b).decode("utf-8", "replace")
    return None


def usage_ok(repo, script_rel, env):
    proc = subprocess.run(
        ["bash", script_rel],
        cwd=repo, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return proc.returncode != 0 and b"usage:" in proc.stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo = repo_root()
    stub = os.path.join(repo, "scripts", "lib", "env_capture_stub")

    capture_env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "TERM": "dumb",
        "ICF_PYTHON": stub,
        "PYTHON_BIN": stub,
    }
    usage_env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
    }

    tmpdir = tempfile.mkdtemp(prefix="arms_env_check_")
    materialized = []
    arms = {}
    all_identical = 1
    usage_ok_all = 1
    try:
        for arm in ARMS:
            orig_rel = materialize_original(repo, args.base, arm)
            materialized.append(orig_rel)

            orig_cap = os.path.join(tmpdir, "orig_%s.txt" % arm)
            new_cap = os.path.join(tmpdir, "new_%s.txt" % arm)
            open(orig_cap, "w").close()
            open(new_cap, "w").close()

            run_capture(repo, orig_rel, arm, orig_cap, capture_env)
            run_capture(repo, "scripts/eval_%s.sh" % arm, arm, new_cap, capture_env)

            orig_sorted = sort_file(orig_cap)
            new_sorted = sort_file(new_cap)
            orig_count = argv_count(orig_sorted)
            new_count = argv_count(new_sorted)

            identical = 1
            diff = None
            if orig_count < 1 or new_count < 1 or orig_sorted != new_sorted:
                identical = 0
                diff = first_diff_line(orig_sorted, new_sorted)

            if not usage_ok(repo, "scripts/eval_%s.sh" % arm, usage_env):
                usage_ok_all = 0

            arms[arm] = {"identical": identical, "diff": diff}
            if identical == 0:
                all_identical = 0
    finally:
        for rel in materialized:
            try:
                os.remove(os.path.join(repo, rel))
            except OSError:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)

    result = {
        "parity": {
            "base": args.base,
            "all_identical": all_identical,
            "usage_ok": usage_ok_all,
            "arms": arms,
        }
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")

    if all_identical == 1 and usage_ok_all == 1:
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
