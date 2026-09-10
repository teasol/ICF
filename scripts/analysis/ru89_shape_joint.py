"""RU-89: offline re-aggregation of the 7-branch (5 + SH + SJ) ensemble.

Loads the saved 50-fold margins for tag `v121_sh_variants` (which stores both
`m_sh` and `m_sj` per §225/§220), and re-aggregates -- purely offline, no GPU,
identical to `branch_screen.py`'s `trimmed_mean(stack([sigmoid(f[x]) for x in
BRANCHES + [cand]]))` -- four configurations: BASE (5-branch), +SH, +SJ, and
+SH+SJ (7-branch).

Reports paired per-fold deltas against BASE and a task-clustered 95% t
interval (df=6, one degree of freedom per PRIMARY7 task, NOT per fold) for
each variant's mean delta, plus an additivity gap for +SH+SJ vs (+SH)+(+SJ).

This script computes numbers only. It does not render a support/refute
verdict; that judgment belongs to Main (Orca) per docs/agent_handoff.md.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from scripts.analysis.branch_diagnostics import (
    BRANCHES_V121_5 as BRANCHES, PRIMARY7, auroc, short, trimmed_mean,
)

try:
    from scipy import stats as _scipy_stats
    _T_CRIT_DF6 = float(_scipy_stats.t.ppf(0.975, df=6))
    _T_CRIT_SRC = "scipy.stats.t.ppf(0.975, df=6)"
except ImportError:
    _T_CRIT_DF6 = 2.446949
    _T_CRIT_SRC = "scipy unavailable -> hardcoded constant 2.446949 (t, df=6, 95%)"


def load(tag: str) -> dict:
    out = {}
    for t in PRIMARY7:
        path = f"predictions/pathobench_{t}_{tag}_official50_bf16.pt"
        folds = torch.load(path, map_location="cpu", weights_only=False)["per_fold"]
        for f in folds:
            if "m_sj" not in f and "m_shj" in f:
                f["m_sj"] = f["m_shj"]
        out[t] = folds
    return out


def per_fold_auroc(folds: list, subset: tuple[str, ...]) -> list[float]:
    return [
        auroc(trimmed_mean(torch.stack([torch.sigmoid(f[b]) for b in subset], dim=0)), f["label"])
        for f in folds
    ]


def task_cluster_ci(task_means: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (mean, se, ci_lo, ci_hi) using a t interval with df = n_tasks - 1."""
    mean = float(np.mean(task_means))
    se = float(np.std(task_means, ddof=1) / math.sqrt(len(task_means)))
    lo = mean - _T_CRIT_DF6 * se
    hi = mean + _T_CRIT_DF6 * se
    return mean, se, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v121_sh_variants")
    args = ap.parse_args()

    data = load(args.tag)

    variants = {
        "BASE": tuple(BRANCHES),
        "+SH": tuple(BRANCHES) + ("m_sh",),
        "+SJ": tuple(BRANCHES) + ("m_sj",),
        "+SH+SJ": tuple(BRANCHES) + ("m_sh", "m_sj"),
    }

    # per_fold[variant][task] = list[float] (50 folds)
    per_fold = {v: {} for v in variants}
    for v, subset in variants.items():
        for t in PRIMARY7:
            per_fold[v][t] = per_fold_auroc(data[t], subset)

    # ---- table: per-task AUROC for each variant + deltas ----
    print(f"\n[RU-89] tag={args.tag}  t_crit source: {_T_CRIT_SRC}  (t_crit={_T_CRIT_DF6:.6f})")
    print(f"\n{'task':<24}{'BASE':>8}{'+SH':>8}{'+SJ':>8}{'+SH+SJ':>8}"
          f"{'d_SH':>8}{'d_SJ':>8}{'d_SHSJ':>8}")

    task_mean_base = {}
    task_mean_variant = {v: {} for v in variants if v != "BASE"}
    task_mean_delta = {v: {} for v in variants if v != "BASE"}

    for t in PRIMARY7:
        base_fold = per_fold["BASE"][t]
        row = {"BASE": float(np.mean(base_fold))}
        deltas_row = {}
        for v in ("+SH", "+SJ", "+SH+SJ"):
            vf = per_fold[v][t]
            row[v] = float(np.mean(vf))
            d = [b - a for b, a in zip(vf, base_fold)]  # paired per fold
            deltas_row[v] = float(np.mean(d))
            task_mean_variant[v][t] = row[v]
            task_mean_delta[v][t] = deltas_row[v]
        task_mean_base[t] = row["BASE"]
        print(f"{short(t):<24}{row['BASE']:>8.4f}{row['+SH']:>8.4f}{row['+SJ']:>8.4f}"
              f"{row['+SH+SJ']:>8.4f}{deltas_row['+SH']:>+8.4f}{deltas_row['+SJ']:>+8.4f}"
              f"{deltas_row['+SH+SJ']:>+8.4f}")

    macro_base = float(np.mean(list(task_mean_base.values())))
    macro_row = {v: float(np.mean(list(task_mean_variant[v].values()))) for v in task_mean_variant}
    macro_delta = {v: macro_row[v] - macro_base for v in task_mean_variant}
    print(f"{'MACRO':<24}{macro_base:>8.4f}{macro_row['+SH']:>8.4f}{macro_row['+SJ']:>8.4f}"
          f"{macro_row['+SH+SJ']:>8.4f}{macro_delta['+SH']:>+8.4f}{macro_delta['+SJ']:>+8.4f}"
          f"{macro_delta['+SH+SJ']:>+8.4f}")

    # ---- task-clustered 95% t CI (df=6) for each variant's mean delta ----
    print(f"\n[RU-89] task-clustered 95% t CI (df=6, n_tasks=7) on mean per-fold delta vs BASE")
    print(f"{'variant':<10}{'mean_delta':>12}{'se':>10}{'ci_lo':>10}{'ci_hi':>10}{'sign agr.':>11}")
    ci_result = {}
    for v in ("+SH", "+SJ", "+SH+SJ"):
        tmeans = np.array([task_mean_delta[v][t] for t in PRIMARY7])
        mean, se, lo, hi = task_cluster_ci(tmeans)
        ci_result[v] = (mean, se, lo, hi)
        wins = sum(1 for t in PRIMARY7 if task_mean_delta[v][t] > 0)
        print(f"{v:<10}{mean:>+12.4f}{se:>10.4f}{lo:>+10.4f}{hi:>+10.4f}{f'{wins}/7':>11}")

    # ---- additivity gap ----
    d_sh = ci_result["+SH"][0]
    d_sj = ci_result["+SJ"][0]
    d_shsj = ci_result["+SH+SJ"][0]
    gap = d_shsj - (d_sh + d_sj)
    print(f"\n[RU-89] additivity gap (task-mean basis, secondary indicator only)")
    print(f"  gap = delta(+SH+SJ) - [delta(+SH) + delta(+SJ)] "
          f"= {d_shsj:+.4f} - [{d_sh:+.4f} + {d_sj:+.4f}] = {gap:+.4f}")

    # ---- worsened tasks, listed in full ----
    print(f"\n[RU-89] tasks worsened vs BASE (task-mean delta <= 0), listed in full")
    for v in ("+SH", "+SJ", "+SH+SJ"):
        worsened = [short(t) for t in PRIMARY7 if task_mean_delta[v][t] <= 0]
        print(f"  {v}: {worsened if worsened else '(none)'}")


if __name__ == "__main__":
    main()
