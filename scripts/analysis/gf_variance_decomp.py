#!/usr/bin/env python3
"""Decompose GF per-fold AUROC variance into fold, seed and residual components.

Round C-20260917-5 fixed a pre-condition on the promotion procedure: the number
of refits R may not be calibrated from an observed spread until that spread is
shown to be fit variance rather than fold variance or leakage. This script does
that decomposition on the artefacts already written by run_gf_standalone.py.

For a configuration evaluated under several GMM initialisation seeds, the same
task/fold cell is measured once per seed, giving a fold x seed table per task.
A two-way random-effects decomposition of that table separates:

    fold      how much folds differ from one another (shared across seeds)
    seed      how much a whole refit shifts every fold together
    residual  fold x seed interaction plus measurement noise

A large seed component is the thing that invalidates single-fit comparisons; a
large residual means refits reorder folds rather than shifting them, which is a
different failure and is not fixed by averaging more folds.

Usage:
    .venv/bin/python scripts/analysis/gf_variance_decomp.py --config m8_raw
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import torch

PRED_ROOT = Path(__file__).resolve().parents[2] / "predictions/gf_standalone"


def load_curves(pred_dir: Path, config: str) -> dict[str, dict[str, list[float]]]:
    """Return {seed_label: {task: per-fold AUROC}} for every refit of `config`."""
    out: dict[str, dict[str, list[float]]] = {}
    for f in sorted(pred_dir.glob("gf_*_per_fold.pt")):
        tag = f.name[len("gf_"):-len("_per_fold.pt")]
        if tag == config:
            label = "seed42"
        elif tag.startswith(f"{config}_seed"):
            label = tag[len(config) + 1:]
        else:
            continue
        blob = torch.load(f, weights_only=False)
        out[label] = {r["task"]: list(r["per_fold_auroc"]) for r in blob["results"]}
    return out


def decompose(table: list[list[float]]) -> tuple[float, float, float]:
    """Two-way random-effects variance components for a [fold][seed] table."""
    n_fold, n_seed = len(table), len(table[0])
    grand = sum(sum(row) for row in table) / (n_fold * n_seed)
    fold_means = [sum(row) / n_seed for row in table]
    seed_means = [sum(table[i][j] for i in range(n_fold)) / n_fold for j in range(n_seed)]

    ms_fold = n_seed * sum((m - grand) ** 2 for m in fold_means) / (n_fold - 1)
    ms_seed = n_fold * sum((m - grand) ** 2 for m in seed_means) / (n_seed - 1)
    ss_res = sum(
        (table[i][j] - fold_means[i] - seed_means[j] + grand) ** 2
        for i in range(n_fold) for j in range(n_seed)
    )
    ms_res = ss_res / ((n_fold - 1) * (n_seed - 1))

    # Negative estimates are clipped to zero: a variance component cannot be
    # negative, and the unbiased estimator can go below zero when the true
    # component is near zero.
    var_res = ms_res
    var_fold = max((ms_fold - ms_res) / n_seed, 0.0)
    var_seed = max((ms_seed - ms_res) / n_fold, 0.0)
    return var_fold, var_seed, var_res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="e.g. m8_raw, m32_raw, whiten_d256_m8")
    ap.add_argument("--pred-dir", type=Path, default=PRED_ROOT)
    args = ap.parse_args()

    curves = load_curves(args.pred_dir, args.config)
    if len(curves) < 2:
        raise SystemExit(f"'{args.config}': 재적합이 {len(curves)}개뿐이라 분해할 수 없다")
    seeds = sorted(curves)
    tasks = sorted(set.intersection(*(set(c) for c in curves.values())))
    print(f"config={args.config}  refits={seeds}  tasks={len(tasks)}")

    # Macro level: average the tasks within a fold, then decompose fold x seed.
    n_fold = min(len(curves[s][t]) for s in seeds for t in tasks)
    macro = [[st.mean(curves[s][t][i] for t in tasks) for s in seeds] for i in range(n_fold)]
    vf, vs, vr = decompose(macro)
    tot = vf + vs + vr
    print(f"\nmacro 수준 분산분해 (fold={n_fold}, seed={len(seeds)})")
    print(f"  fold     {vf:.6f}  ({100*vf/tot:5.1f}%)  sd={vf**0.5:.4f}")
    print(f"  seed     {vs:.6f}  ({100*vs/tot:5.1f}%)  sd={vs**0.5:.4f}   <- 단일 적합 비교가 놓치는 성분")
    print(f"  잔차     {vr:.6f}  ({100*vr/tot:5.1f}%)  sd={vr**0.5:.4f}   <- fold×seed 상호작용")
    print(f"\n  적합 평균의 표준오차 (seed {len(seeds)}회): {(vs/len(seeds))**0.5:.4f}")
    print(f"  승격 기준 +0.0030 대비 seed sd: {vs**0.5/0.003:.1f}배")


if __name__ == "__main__":
    main()
