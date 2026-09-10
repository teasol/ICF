"""RU-90: numeric layers for wiring-equivalence and single-generation SH/SJ re-measurement.

Three layers, run offline from saved `predictions/pathobench_{TASK}_{TAG}_official50_bf16.pt`
files (no GPU work in this script itself):

  Layer 1 (--phase1): measurement integrity of a single screen-only tag.
    - Primary 7 macro AUROC of the BASE 5-branch arm (stored `probability`) vs the
      fixed anchor 0.6171.
    - Contamination check: cptac_pda_SMAD4_mutation vs 0.4421, cptac_ccrcc_PBRM1_mutation
      vs 0.5553.
    - BASE arm re-aggregation check: stored `probability` vs offline
      trimmed_mean(stack(sigmoid(BRANCHES))) recomputed from the saved margins,
      max|diff| over every fold/task, plus per-fold AUROC 6-decimal match.
    - Margin presence: m_sh, m_sj (falling back to m_shj), m_bs -- counted per task,
      to surface RU-90 kill condition (5).

  Layer 3 (chained into the same --phase1 run): RU-89-style re-aggregation of
    BASE / +SH / +SJ / +SH+SJ (BS is NOT an arm -- only its presence is checked),
    paired per-fold deltas vs BASE, task-clustered 95% t interval (df=6), sign
    agreement, worsened tasks listed in full, additivity gap, comparison against
    the RU-89 anchors (+SH +0.26%p, +SJ +0.32%p, +SH+SJ +0.60%p), and the SH-SJ
    inter-margin correlation (max|r| across tasks).

  Layer 2 (--phase3): live-path equivalence for a single task, given 3 live tags
    `{prefix}_sh`, `{prefix}_sj`, `{prefix}_shsj` (each: SH-only, SJ-only, and
    SH+SJ in the live pool respectively) plus a --ref-tag screen-only run (Phase 1)
    supplying the margins to re-aggregate offline. For each arm:
      1. max|diff| of live-stored margins vs ref-tag margins, per branch
         (m_cv, m_bm, m_bd, m_qa, m_ds, m_sh, m_sj), same fold.
      2. max|diff| of live `probability` vs offline re-aggregation of the
         ref-tag's margins under that arm's branch subset.
      3. max|diff| of the per-fold AUROC of those two probability series.

This script computes numbers only. It does not render a support/refute verdict or
any pass/fail label against the 1e-6 / 1e-3 thresholds in RU-90's criteria --
that judgment belongs to Main (Orca) per docs/agent_handoff.md.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from scripts.analysis.branch_diagnostics import (
    BRANCHES_V121_5 as BRANCHES, PRIMARY7, auroc, short, trimmed_mean,
)
from scripts.analysis.ru89_shape_joint import task_cluster_ci, _T_CRIT_DF6, _T_CRIT_SRC

ANCHOR_MACRO = 0.6171
ANCHOR_SMAD4 = 0.4421
ANCHOR_PBRM1 = 0.5553
RU89_ANCHOR = {"+SH": 0.26, "+SJ": 0.32, "+SH+SJ": 0.60}  # %p

VARIANTS = {
    "BASE": tuple(BRANCHES),
    "+SH": tuple(BRANCHES) + ("m_sh",),
    "+SJ": tuple(BRANCHES) + ("m_sj",),
    "+SH+SJ": tuple(BRANCHES) + ("m_sh", "m_sj"),
}


def _path(task: str, tag: str) -> str:
    return f"predictions/pathobench_{task}_{tag}_official50_bf16.pt"


def load_raw(task: str, tag: str) -> dict:
    path = _path(task, tag)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        sys.exit(f"[RU-90] missing file: tag={tag!r} task={task!r} -> {path}")


def load_folds(tag: str, tasks: list[str]) -> dict:
    """Loads per_fold lists for `tasks` under `tag`, applying the m_sj<-m_shj fallback."""
    out = {}
    for t in tasks:
        folds = load_raw(t, tag)["per_fold"]
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


def per_fold_prob(folds: list, subset: tuple[str, ...]) -> list[torch.Tensor]:
    return [
        trimmed_mean(torch.stack([torch.sigmoid(f[b]) for b in subset], dim=0))
        for f in folds
    ]


def _match(actual: float, anchor: float, decimals: int = 4) -> str:
    return "MATCH" if round(actual, decimals) == round(anchor, decimals) else "MISMATCH"


# ---------------------------------------------------------------------------
# Layer 1 + Layer 3
# ---------------------------------------------------------------------------

def run_phase1(tag: str) -> None:
    data = load_folds(tag, PRIMARY7)

    # ---- 1. Primary 7 macro (BASE = stored probability) ----
    task_mean_base = {}
    for t in PRIMARY7:
        folds = data[t]
        scores = [auroc(f["probability"], f["label"]) for f in folds]
        task_mean_base[t] = float(np.mean(scores))
    macro_base = float(np.mean(list(task_mean_base.values())))

    print(f"\n[RU-90 Layer 1] tag={tag}  measurement integrity")
    print(f"\n(1) Primary 7 macro AUROC (BASE = stored `probability`)")
    print(f"{'task':<24}{'auroc':>8}")
    for t in PRIMARY7:
        print(f"{short(t):<24}{task_mean_base[t]:>8.4f}")
    print(f"{'MACRO':<24}{macro_base:>8.4f}   anchor={ANCHOR_MACRO:.4f}  diff={macro_base - ANCHOR_MACRO:+.4f}")

    # ---- 2. contamination check ----
    smad4 = task_mean_base["cptac_pda_SMAD4_mutation"]
    pbrm1 = task_mean_base["cptac_ccrcc_PBRM1_mutation"]
    print(f"\n(2) contamination check")
    print(f"{'task':<24}{'auroc':>8}{'anchor':>8}{'result':>10}")
    print(f"{'SMAD4_mutation':<24}{smad4:>8.4f}{ANCHOR_SMAD4:>8.4f}{_match(smad4, ANCHOR_SMAD4):>10}")
    print(f"{'PBRM1_mutation':<24}{pbrm1:>8.4f}{ANCHOR_PBRM1:>8.4f}{_match(pbrm1, ANCHOR_PBRM1):>10}")

    # ---- 3. BASE arm offline re-aggregation check ----
    print(f"\n(3) BASE arm: stored `probability` vs offline trimmed_mean(sigmoid(BRANCHES))")
    print(f"{'task':<24}{'max|diff|prob':>16}{'auroc match(6dp)':>18}")
    global_max_diff = 0.0
    all_auroc_match = True
    for t in PRIMARY7:
        folds = data[t]
        # Folds hold different slide counts, so compare fold by fold rather
        # than stacking into one tensor.
        recomputed = per_fold_prob(folds, tuple(BRANCHES))
        diff = max(
            float((f["probability"] - r).abs().max().item())
            for f, r in zip(folds, recomputed)
        )
        global_max_diff = max(global_max_diff, diff)
        stored_auroc = [round(auroc(f["probability"], f["label"]), 6) for f in folds]
        recomp_auroc = [round(a, 6) for a in per_fold_auroc(folds, tuple(BRANCHES))]
        match = "MATCH" if stored_auroc == recomp_auroc else "MISMATCH"
        if match == "MISMATCH":
            all_auroc_match = False
        print(f"{short(t):<24}{diff:>16.3e}{match:>18}")
    print(f"{'GLOBAL MAX':<24}{global_max_diff:>16.3e}{('MATCH' if all_auroc_match else 'MISMATCH'):>18}")

    # ---- 4. margin presence ----
    print(f"\n(4) margin presence (RU-90 kill condition (5))")
    print(f"{'task':<24}{'m_sh':>8}{'m_sj':>8}{'m_bs':>8}{'n_folds':>9}")
    for t in PRIMARY7:
        folds = data[t]
        n = len(folds)
        n_sh = sum(1 for f in folds if f.get("m_sh") is not None)
        n_sj = sum(1 for f in folds if f.get("m_sj") is not None)
        n_bs = sum(1 for f in folds if f.get("m_bs") is not None)
        print(f"{short(t):<24}{n_sh:>8}{n_sj:>8}{n_bs:>8}{n:>9}")

    # ---- Layer 3: RU-89-style re-aggregation ----
    run_layer3(data, tag)


def run_layer3(data: dict, tag: str) -> None:
    per_fold = {v: {} for v in VARIANTS}
    for v, subset in VARIANTS.items():
        for t in PRIMARY7:
            per_fold[v][t] = per_fold_auroc(data[t], subset)

    print(f"\n[RU-90 Layer 3] tag={tag}  RU-89-style re-aggregation "
          f"(BASE/+SH/+SJ/+SH+SJ; BS is not an arm)  t_crit(df=6)={_T_CRIT_DF6:.6f} ({_T_CRIT_SRC})")
    print(f"\n{'task':<24}{'BASE':>8}{'+SH':>8}{'+SJ':>8}{'+SH+SJ':>8}"
          f"{'d_SH':>8}{'d_SJ':>8}{'d_SHSJ':>8}")

    task_mean_base = {}
    task_mean_variant = {v: {} for v in VARIANTS if v != "BASE"}
    task_mean_delta = {v: {} for v in VARIANTS if v != "BASE"}

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

    print(f"\n[RU-90 Layer 3] task-clustered 95% t CI (df=6, n_tasks=7) on mean per-fold delta vs BASE")
    print(f"{'variant':<10}{'mean_delta':>12}{'se':>10}{'ci_lo':>10}{'ci_hi':>10}{'sign agr.':>11}"
          f"{'vs RU-89':>12}{'|diff|%p':>10}")
    ci_result = {}
    for v in ("+SH", "+SJ", "+SH+SJ"):
        tmeans = np.array([task_mean_delta[v][t] for t in PRIMARY7])
        mean, se, lo, hi = task_cluster_ci(tmeans)
        ci_result[v] = (mean, se, lo, hi)
        wins = sum(1 for t in PRIMARY7 if task_mean_delta[v][t] > 0)
        anchor_pp = RU89_ANCHOR[v]
        diff_pp = abs(mean * 100 - anchor_pp)
        print(f"{v:<10}{mean:>+12.4f}{se:>10.4f}{lo:>+10.4f}{hi:>+10.4f}{f'{wins}/7':>11}"
              f"{f'{anchor_pp:+.2f}%p':>12}{diff_pp:>10.2f}")

    d_sh = ci_result["+SH"][0]
    d_sj = ci_result["+SJ"][0]
    d_shsj = ci_result["+SH+SJ"][0]
    gap = d_shsj - (d_sh + d_sj)
    print(f"\n[RU-90 Layer 3] additivity gap (task-mean basis, secondary indicator only)")
    print(f"  gap = delta(+SH+SJ) - [delta(+SH) + delta(+SJ)] "
          f"= {d_shsj:+.4f} - [{d_sh:+.4f} + {d_sj:+.4f}] = {gap:+.4f}")

    print(f"\n[RU-90 Layer 3] tasks worsened vs BASE (task-mean delta <= 0), listed in full")
    for v in ("+SH", "+SJ", "+SH+SJ"):
        worsened = [short(t) for t in PRIMARY7 if task_mean_delta[v][t] <= 0]
        print(f"  {v}: {worsened if worsened else '(none)'}")

    # ---- SH-SJ inter-margin correlation (max|r| across tasks, gate (1) reference) ----
    print(f"\n[RU-90 Layer 3] SH-SJ inter-margin correlation (mean within-fold Pearson r), gate (1) reference")
    print(f"{'task':<24}{'mean_r':>10}")
    corrs = []
    for t in PRIMARY7:
        folds = data[t]
        rs = []
        for f in folds:
            sh = f["m_sh"].float().numpy()
            sj = f["m_sj"].float().numpy()
            if sh.std() < 1e-12 or sj.std() < 1e-12:
                continue
            r = float(np.corrcoef(sh, sj)[0, 1])
            rs.append(r)
        mean_r = float(np.mean(rs)) if rs else float("nan")
        corrs.append(mean_r)
        print(f"{short(t):<24}{mean_r:>10.3f}")
    max_abs_r = float(np.nanmax(np.abs(corrs)))
    print(f"{'max|r| over tasks':<24}{max_abs_r:>10.3f}")


# ---------------------------------------------------------------------------
# Layer 2 (--phase3)
# ---------------------------------------------------------------------------

LIVE_ARM_SUFFIX = {
    "sh": ("+SH", tuple(BRANCHES) + ("m_sh",)),
    "sj": ("+SJ", tuple(BRANCHES) + ("m_sj",)),
    "shsj": ("+SH+SJ", tuple(BRANCHES) + ("m_sh", "m_sj")),
}
LIVE_CHECK_BRANCHES = ["m_cv", "m_bm", "m_bd", "m_qa", "m_ds", "m_sh", "m_sj"]


def run_phase3(tag_prefix: str, task: str, ref_tag: str) -> None:
    ref_folds = load_folds(ref_tag, [task])[task]

    print(f"\n[RU-90 Layer 2] task={short(task)}  ref_tag={ref_tag}  live prefix={tag_prefix}")

    for suffix, (arm_name, subset) in LIVE_ARM_SUFFIX.items():
        live_tag = f"{tag_prefix}_{suffix}"
        live_folds = load_folds(live_tag, [task])[task]
        if len(live_folds) != len(ref_folds):
            sys.exit(
                f"[RU-90] fold-count mismatch: live_tag={live_tag!r} has {len(live_folds)} "
                f"folds, ref_tag={ref_tag!r} has {len(ref_folds)}"
            )

        print(f"\n--- arm {arm_name}  (live_tag={live_tag})")

        # (1) margin equivalence, per branch
        print(f"  (1) live margin vs ref margin, max|diff| per branch")
        print(f"      {'branch':<8}{'max|diff|':>14}")
        for b in LIVE_CHECK_BRANCHES:
            diffs = []
            for lf, rf in zip(live_folds, ref_folds):
                lv = lf.get(b)
                rv = rf.get(b)
                if lv is None or rv is None:
                    continue
                diffs.append(float((lv - rv).abs().max().item()))
            if diffs:
                print(f"      {b:<8}{max(diffs):>14.3e}")
            else:
                print(f"      {b:<8}{'N/A (missing)':>14}")

        # (2) probability equivalence: live `probability` vs offline re-agg of ref margins
        live_prob = torch.stack([f["probability"] for f in live_folds])
        ref_recomp_prob = torch.stack(per_fold_prob(ref_folds, subset))
        prob_diff = float((live_prob - ref_recomp_prob).abs().max().item())
        print(f"  (2) live `probability` vs offline re-agg(ref margins, {arm_name}): "
              f"max|diff| = {prob_diff:.3e}")

        # (3) per-fold AUROC equivalence
        live_auroc = np.array([auroc(f["probability"], f["label"]) for f in live_folds])
        ref_recomp_auroc = np.array(per_fold_auroc(ref_folds, subset))
        auroc_diff = float(np.max(np.abs(live_auroc - ref_recomp_auroc)))
        print(f"  (3) per-fold AUROC(live) vs per-fold AUROC(offline re-agg): "
              f"max|diff| = {auroc_diff:.3e}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase1", action="store_true", help="Layer 1 + Layer 3 (single screen-only tag)")
    mode.add_argument("--phase3", action="store_true", help="Layer 2 (live-path equivalence, single task)")
    ap.add_argument("--tag", default="ru90_shape_triple", help="[--phase1] screen-only tag")
    ap.add_argument("--tag-prefix", default="ru90_live", help="[--phase3] live tag prefix")
    ap.add_argument("--task", default=None, help="[--phase3] single task name")
    ap.add_argument("--ref-tag", default=None, help="[--phase3] Phase 1 screen-only tag to compare against")
    args = ap.parse_args()

    if args.phase1:
        run_phase1(args.tag)
    else:
        if args.task is None or args.ref_tag is None:
            ap.error("--phase3 requires --task and --ref-tag")
        run_phase3(args.tag_prefix, args.task, args.ref_tag)


if __name__ == "__main__":
    main()
