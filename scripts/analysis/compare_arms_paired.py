#!/usr/bin/env python
"""Fold-paired Delta and bootstrap CI between two arms on the official SEAL tasks.

Why this exists (2026-08-12, user decision): arm judgments were being made by
subtracting two point estimates of the 10-task macro. The per-fold spread inside
one task is an order of magnitude larger than the effects being judged --
er_status is `fold-mean 0.7023 +/- 0.0903` while the deltas in docs SS98 range
0.0012..0.0118. Every arm is scored on the SAME case-disjoint official folds, so
differencing per fold first removes the fold-difficulty term and leaves only
Var(d_f).

The statistical unit here is the FOLD: d_f = auroc_arm(f) - auroc_baseline(f).
Because d_f is already a difference, resampling folds of d_f is automatically the
paired bootstrap -- one resample index, applied to both arms by construction.

Reads the predictions that `test_pathobench.py` already saved, so this needs no
GPU and no re-evaluation:
    predictions/pathobench_{task}_{tag}_official50_bf16.pt

Usage:
    python scripts/compare_arms_paired.py --baseline <TAG> --arm <TAG> [--arm <TAG> ...]

Pairing is verified, not assumed: fold count, per-fold slide_id order and
per-fold labels must match between arms. A mismatch is a hard error rather than a
silent fall back to an unpaired comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import auroc_rows

SEAL_TASKS = [
    "bc_therapy_er_status",
    "bc_therapy_grade",
    "bc_therapy_her2_status",
    "cptac_brca_PIK3CA_mutation",
    "cptac_brca_TP53_mutation",
    "cptac_luad_EGFR_mutation",
    "cptac_luad_STK11_mutation",
    "cptac_luad_TP53_mutation",
    "cptac_ccrcc_BAP1_mutation",
    "cptac_ccrcc_VHL_mutation",
]


class PairingError(RuntimeError):
    """Raised when two arms cannot be compared fold-by-fold."""


def prediction_path(root: Path, task: str, tag: str) -> Path:
    return root / "predictions" / f"pathobench_{task}_{tag}_official50_bf16.pt"


def load_arm(root: Path, task: str, tag: str) -> dict:
    path = prediction_path(root, task, tag)
    if not path.exists():
        raise PairingError(f"missing predictions: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def fold_aurocs(record: dict) -> torch.Tensor:
    """Recompute per-fold AUROC with one implementation for both arms.

    The files carry `fold_aurocs`, but recomputing means the two arms cannot
    differ because of how their numbers were produced. The stored values are
    cross-checked below.
    """
    values = []
    for fold in record["per_fold"]:
        values.append(auroc_rows(fold["probability"].flatten(), fold["label"].flatten()))
    return torch.stack(values).to(torch.float64)


def verify_pairing(task: str, baseline: dict, arm: dict) -> None:
    left, right = baseline["per_fold"], arm["per_fold"]
    if len(left) != len(right):
        raise PairingError(f"{task}: fold count {len(left)} vs {len(right)}")
    if baseline["fold_indices"] != arm["fold_indices"]:
        raise PairingError(f"{task}: fold_indices differ")
    for index, (lhs, rhs) in enumerate(zip(left, right), start=1):
        if list(lhs["slide_id"]) != list(rhs["slide_id"]):
            raise PairingError(f"{task}: fold {index} slide_id order differs")
        if not torch.equal(lhs["label"], rhs["label"]):
            raise PairingError(f"{task}: fold {index} labels differ")


def check_recompute(task: str, tag: str, record: dict, recomputed: torch.Tensor) -> None:
    stored = torch.tensor(record["fold_aurocs"], dtype=torch.float64)
    gap = (stored - recomputed).abs().max().item()
    if gap > 1e-6:
        print(
            f"  ! {task} [{tag}]: recomputed AUROC differs from stored by {gap:.2e}",
            file=sys.stderr,
        )


def bootstrap_means(
    deltas: torch.Tensor, draws: int, generator: torch.Generator
) -> torch.Tensor:
    """Resample folds with replacement; return the mean delta of each replicate."""
    count = deltas.numel()
    index = torch.randint(0, count, (draws, count), generator=generator)
    return deltas[index].mean(dim=1)


def percentile_interval(samples: torch.Tensor, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    quantiles = torch.tensor([tail, 1.0 - tail], dtype=samples.dtype)
    low, high = torch.quantile(samples, quantiles).tolist()
    return low, high


def compare(
    root: Path,
    baseline_tag: str,
    arm_tag: str,
    tasks: list[str],
    draws: int,
    seed: int,
    level: float,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    per_task = []
    replicates = []

    for task in tasks:
        baseline = load_arm(root, task, baseline_tag)
        arm = load_arm(root, task, arm_tag)
        verify_pairing(task, baseline, arm)

        baseline_folds = fold_aurocs(baseline)
        arm_folds = fold_aurocs(arm)
        check_recompute(task, baseline_tag, baseline, baseline_folds)
        check_recompute(task, arm_tag, arm, arm_folds)

        deltas = arm_folds - baseline_folds
        samples = bootstrap_means(deltas, draws, generator)
        low, high = percentile_interval(samples, level)
        per_task.append(
            {
                "task": task,
                "baseline": baseline_folds.mean().item(),
                "arm": arm_folds.mean().item(),
                "delta": deltas.mean().item(),
                "low": low,
                "high": high,
                "wins": int((deltas > 0).sum().item()),
                "folds": deltas.numel(),
            }
        )
        replicates.append(samples)

    macro_samples = torch.stack(replicates).mean(dim=0)
    macro_low, macro_high = percentile_interval(macro_samples, level)
    macro = {
        "baseline": sum(row["baseline"] for row in per_task) / len(per_task),
        "arm": sum(row["arm"] for row in per_task) / len(per_task),
        "delta": sum(row["delta"] for row in per_task) / len(per_task),
        "low": macro_low,
        "high": macro_high,
        "tasks_up": sum(1 for row in per_task if row["delta"] > 0),
        "tasks": len(per_task),
    }
    return {"per_task": per_task, "macro": macro}


def report(baseline_tag: str, arm_tag: str, result: dict, level: float) -> None:
    percent = int(round(level * 100))
    print(f"\n=== {arm_tag}  -  {baseline_tag} ===")
    print(f"fold-paired delta, {percent}% percentile CI over resampled folds\n")
    header = f"{'task':32s} {'base':>7s} {'arm':>7s} {'delta':>8s}  {'CI':>20s}  won"
    print(header)
    print("-" * len(header))
    for row in result["per_task"]:
        interval = f"[{row['low']:+.4f}, {row['high']:+.4f}]"
        print(
            f"{row['task']:32s} {row['baseline']:7.4f} {row['arm']:7.4f} "
            f"{row['delta']:+8.4f}  {interval:>20s}  {row['wins']:2d}/{row['folds']}"
        )
    macro = result["macro"]
    print("-" * len(header))
    interval = f"[{macro['low']:+.4f}, {macro['high']:+.4f}]"
    print(
        f"{'MACRO':32s} {macro['baseline']:7.4f} {macro['arm']:7.4f} "
        f"{macro['delta']:+8.4f}  {interval:>20s}  "
        f"{macro['tasks_up']:2d}/{macro['tasks']} tasks"
    )
    excludes_zero = macro["low"] > 0.0 or macro["high"] < 0.0
    verdict = "CI excludes 0" if excludes_zero else "CI includes 0 -- indistinguishable"
    print(f"\n  macro verdict: {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="control arm tag")
    parser.add_argument("--arm", required=True, action="append", help="arm tag (repeatable)")
    parser.add_argument("--tasks", nargs="*", default=SEAL_TASKS)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    failures = 0
    for arm_tag in args.arm:
        try:
            result = compare(
                args.root,
                args.baseline,
                arm_tag,
                list(args.tasks),
                args.bootstrap,
                args.seed,
                args.level,
            )
        except PairingError as error:
            print(f"\n=== {arm_tag}  -  {args.baseline} ===\n  FAILED: {error}")
            failures += 1
            continue
        report(args.baseline, arm_tag, result, args.level)

    print(
        "\nCaveats: the 50 folds overlap in slides, so treating folds as "
        "independent draws likely understates the variance. This captures "
        "fold noise only -- not training-seed (one checkpoint per arm) and not "
        "task-selection noise (10 fixed tasks)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
