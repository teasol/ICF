"""Run official-fold PathoBench eval for ONE task across N parallel workers, then merge.

Splits a task's official folds (from ``{task_dir}/k=all.tsv``) into N contiguous
chunks, launches one ``test_pathobench.py --official-folds`` subprocess per chunk
(optionally pinned to distinct GPUs via ``CUDA_VISIBLE_DEVICES``), waits for all,
then merges the per-worker results into a single output file:

  fold_aurocs / fold_auroc_mean±std / auroc_pooled / per_fold (all folds)

Designed for the SEAL-comparable official 50-fold protocol (17 tasks).

Usage:
    python scripts/run_official_folds_parallel.py \
        --checkpoint checkpoints/.../epoch=048-val_ce_loss=0.4419.ckpt \
        --config configs/train_v34_phase0_largectx_1536.yaml \
        --task-dir /NHNHOME/BASE/kimds/Data/PathoBench/official/bc_therapy/er_status \
        --features /NHNHOME/BASE/kimds/Data/PathoBench/features \
        --workers 5 --output predictions/pathobench_bc_therapy_er_status_official50.pt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import auroc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True,
                        help="Official task dir with k=all.tsv + config.yaml.")
    parser.add_argument("--features", type=Path,
                        default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/features"))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--nfolds-total", type=int, default=None,
                        help="Only evaluate the first K official folds (scaling "
                        "probe). Default: all folds.")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated CUDA device ids to round-robin across workers.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tmp-dir", type=Path,
                        default=Path("/tmp/pathobench_official_workers"))
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--context-max-tiles", type=int, default=None)
    parser.add_argument("--keep-tmp", action="store_true")
    return parser.parse_args()


def count_folds(task_dir: Path) -> int:
    with (task_dir / "k=all.tsv").open() as fh:
        header = fh.readline().split("\t")
    return sum(1 for c in header if c.startswith("fold_"))


def main() -> None:
    args = parse_args()
    task_dir = args.task_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    tmp_dir = args.tmp_dir.expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]

    total_folds = count_folds(task_dir)
    if args.nfolds_total is not None:
        total_folds = min(args.nfolds_total, total_folds)
    n = args.workers
    chunk = math.ceil(total_folds / n)
    ranges = [(s, min(s + chunk, total_folds)) for s in range(0, total_folds, chunk)]

    procs = []
    start_t = time.time()
    for i, (s, e) in enumerate(ranges):
        worker_out = tmp_dir / f"worker_{i}.pt"
        worker_log = tmp_dir / f"worker_{i}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "test_pathobench.py"),
            "--checkpoint", str(args.checkpoint.expanduser().resolve()),
            "--config", str(args.config.expanduser().resolve()),
            "--official-folds", str(task_dir),
            "--official-fold-start", str(s),
            "--official-nfolds", str(e - s),
            "--features", str(args.features.expanduser().resolve()),
            "--official-ckpt", str(tmp_dir / f"{task_dir.name}_official_folds.ckpt"),
            "--output", str(worker_out),
            "--seed", str(args.seed),
        ]
        if args.max_tiles is not None:
            cmd += ["--max-tiles", str(args.max_tiles)]
        if args.context_max_tiles is not None:
            cmd += ["--context-max-tiles", str(args.context_max_tiles)]
        with worker_log.open("w") as lf:
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((i, s, e, p, worker_out, worker_log))
        print(f"  worker {i}: folds {s + 1}..{e} (n={e - s}) gpu={gpus[i % len(gpus)]} "
              f"pid={p.pid}", flush=True)

    for i, s, e, p, worker_out, worker_log in procs:
        p.wait()
        if p.returncode != 0:
            print(f"  worker {i} FAILED (exit {p.returncode}); tail of log:")
            lines = worker_log.read_text().splitlines()
            print("\n".join(lines[-20:]))
            # Kill any still-running sibling workers so they release the GPU:
            # an OOM'd worker leaves its siblings alive holding device memory,
            # which would make a lower-worker retry OOM instantly (cascade).
            for _, _, _, pj, _, _ in procs:
                if pj.poll() is None:
                    try:
                        pj.kill()
                    except Exception:
                        pass
            for _, _, _, pj, _, _ in procs:
                try:
                    pj.wait(timeout=10)
                except Exception:
                    pass
            sys.exit(1)
        print(f"  worker {i} done (folds {s + 1}..{e}) in {time.time() - start_t:.0f}s", flush=True)

    # Merge per-worker results.
    fold_aurocs: dict[int, float] = {}
    prob_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    per_fold: list[dict] = []
    for i, s, e, p, worker_out, worker_log in procs:
        state = torch.load(worker_out, map_location="cpu", weights_only=False)
        for k, a in zip(state["fold_indices"], state["fold_aurocs"]):
            fold_aurocs[k] = a
        for rec in state["per_fold"]:
            per_fold.append({**rec, "fold": None})
        prob_parts.append(state["per_fold"] and torch.cat([r["probability"] for r in state["per_fold"]]))
        target_parts.append(state["per_fold"] and torch.cat([r["label"] for r in state["per_fold"]]))
        if not args.keep_tmp:
            worker_out.unlink(missing_ok=True)
            worker_log.unlink(missing_ok=True)

    keys = sorted(fold_aurocs)
    aurocs = [fold_aurocs[k] for k in keys]
    mean = sum(aurocs) / len(aurocs)
    std = (sum((x - mean) ** 2 for x in aurocs) / len(aurocs)) ** 0.5
    pooled = auroc(torch.cat(prob_parts), torch.cat(target_parts))

    print(f"\n=== Merged official {len(aurocs)}-fold — {task_dir.parent.name}/{task_dir.name} ===")
    print(f"per-fold AUROC: {' '.join(f'{x:.4f}' for x in aurocs)}")
    print(f"fold-mean AUROC: {mean:.4f} ± {std:.4f}   pooled AUROC: {pooled:.4f}")
    print(f"workers={n} total_time={time.time() - start_t:.0f}s")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task_dir.name,
            "official_folds": len(aurocs),
            "fold_indices": keys,
            "fold_aurocs": aurocs,
            "fold_auroc_mean": mean,
            "fold_auroc_std": std,
            "auroc_pooled": float(pooled),
            "per_fold": per_fold,
            "workers": n,
        },
        output,
    )
    print(f"Saved merged official-fold predictions to {output}")


if __name__ == "__main__":
    main()
