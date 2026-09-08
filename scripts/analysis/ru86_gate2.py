"""RU-86: gate 2b fold-reproducibility screen for MDX on the Primary 7 tasks.

Card: docs/ru/RU-86.json. Criteria, kill conditions and budget are fixed there
and are NOT re-derived here — this script only applies them mechanically to
stored predictions.

Step 1 finding (recorded here, not re-measured): the RU-85 stage-B run
(scripts/run_ru85_tier1_screen.sh) already stored, for every Primary-7 task and
every one of its 50 official folds, the per-query-slide MDX margin (`m_mdx`)
and label inside `predictions/pathobench_<task>_ru85_stageB_official50_bf16.pt`
(key `per_fold`). No GPU recomputation is required — this is a CPU-only
reaggregation of files that already exist on disk.

Primary path (1st, confirmatory): task `Grade` (cptac_lscc/Histologic_Grade).
  - count fold-level MDX-alone AUROC > 0.5 across the 50 official folds -> k
  - one-sided binomial test, p0 = 0.5, H1: rate > 0.5
  - pass: k >= 40 AND p < 0.01
  - undetermined: > 10 folds have undefined AUROC (single-class fold); do not
    recompute the test on the reduced denominator when that happens
  - execution-invalid: contamination check (max |fold AUROC diff| vs
    v121_baseline, ensemble level) is nonzero

Secondary path (2nd, exploratory, declared before results per the card): same
per-fold MDX AUROC distribution for the other 6 tasks, plus a within-task vs
between-task variance comparison. This is recorded for the "no information"
vs "task-dependent" distinction only and must NOT be used to re-rank or
re-select the confirmatory task.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.metrics import auroc  # noqa: E402

# task key -> (predictions file stem, display name); order fixed by
# scripts/eval_v121.sh default task list (= Primary 7).
TASKS = [
    ("cptac_lscc_ARID1A_mutation", "ARID1A"),
    ("cptac_lscc_Histologic_Grade", "Grade"),
    ("cptac_lscc_KEAP1_mutation", "KEAP1"),
    ("cptac_luad_KRAS_mutation", "KRAS"),
    ("cptac_pda_SMAD4_mutation", "SMAD4"),
    ("ucla_lung_progression_regression", "Progression"),
    ("cptac_ccrcc_PBRM1_mutation", "PBRM1"),
]

STAGE_TAG = "ru85_stageB"
BASELINE_TAG = "v121_baseline"
N_FOLDS = 50
PRIMARY_TASK = "Grade"


def load(stem: str, tag: str) -> dict:
    path = ROOT / "predictions" / f"pathobench_{stem}_{tag}_official50_bf16.pt"
    if not path.exists():
        raise SystemExit(f"missing predictions file: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def per_fold_mdx_auroc(blob: dict) -> list[float]:
    """One MDX-alone AUROC per fold; NaN where the fold is single-class."""
    out = []
    for fold in blob["per_fold"]:
        label = fold["label"].float()
        margin = fold["m_mdx"]
        out.append(auroc(margin, label))
    return out


def summarize_task(stem: str, name: str) -> dict:
    stage_blob = load(stem, STAGE_TAG)
    base_blob = load(stem, BASELINE_TAG)
    assert len(stage_blob["per_fold"]) == N_FOLDS, (
        f"{name}: expected {N_FOLDS} folds, got {len(stage_blob['per_fold'])}"
    )

    fold_auroc = per_fold_mdx_auroc(stage_blob)
    valid = [v for v in fold_auroc if not math.isnan(v)]
    n_valid = len(valid)
    n_invalid = N_FOLDS - n_valid
    k = sum(1 for v in fold_auroc if (not math.isnan(v)) and v > 0.5)

    # contamination check: ensemble-level fold AUROC must match baseline
    # exactly (MDX runs SCREEN_ONLY=1, i.e. must not enter the ensemble).
    stage_ens = np.asarray(stage_blob["fold_aurocs"], dtype=np.float64)
    base_ens = np.asarray(base_blob["fold_aurocs"], dtype=np.float64)
    max_ens_diff = float(np.max(np.abs(stage_ens - base_ens)))

    quartiles = (
        list(np.percentile(valid, [25, 50, 75])) if n_valid > 0 else [float("nan")] * 3
    )
    var_valid = float(np.var(valid, ddof=1)) if n_valid > 1 else float("nan")

    result = {
        "task": name,
        "stem": stem,
        "n_folds": N_FOLDS,
        "n_valid_folds": n_valid,
        "n_invalid_folds": n_invalid,
        "fold_auroc": fold_auroc,
        "k_gt_0.5": k,
        "median": quartiles[1],
        "q1": quartiles[0],
        "q3": quartiles[2],
        "variance_ddof1": var_valid,
        "max_ensemble_fold_diff_vs_baseline": max_ens_diff,
    }
    return result


def binomial_verdict(k: int, n_invalid: int) -> dict:
    """Apply the card's fixed 1st-path (confirmatory) rule to one task's k."""
    if n_invalid > 10:
        return {
            "verdict": "판별_불가",
            "reason": f"invalid folds = {n_invalid} > 10 (undefined AUROC); "
            "not recomputed on the reduced denominator per card criteria",
            "p_value": None,
        }
    test = binomtest(k, n=N_FOLDS, p=0.5, alternative="greater")
    p_value = float(test.pvalue)
    passed = (k >= 40) and (p_value < 0.01)
    return {
        "verdict": "지지" if passed else "반박",
        "k": k,
        "n": N_FOLDS,
        "p_value": p_value,
        "pass_threshold": "k>=40 and p<0.01",
    }


def main() -> None:
    per_task = {}
    for stem, name in TASKS:
        per_task[name] = summarize_task(stem, name)

    max_ens_diff_all = max(t["max_ensemble_fold_diff_vs_baseline"] for t in per_task.values())
    contamination_invalid = max_ens_diff_all > 0.0

    primary = per_task[PRIMARY_TASK]
    kill_2 = primary["n_valid_folds"] < 40  # gate ②: valid < 40 => k>=40 impossible

    if contamination_invalid:
        primary_verdict = {
            "verdict": "실행_무효",
            "reason": f"contamination check failed: max |fold AUROC diff| = {max_ens_diff_all:.3e} > 0",
        }
    elif kill_2:
        primary_verdict = {
            "verdict": "판별_불가_kill②",
            "reason": f"{PRIMARY_TASK}: valid folds = {primary['n_valid_folds']} < 40, "
            "k>=40 is arithmetically impossible",
        }
    else:
        primary_verdict = binomial_verdict(primary["k_gt_0.5"], primary["n_invalid_folds"])

    # secondary (exploratory only, per card: never used to re-rank/re-select)
    secondary = {}
    for name, t in per_task.items():
        secondary[name] = binomial_verdict(t["k_gt_0.5"], t["n_invalid_folds"])

    task_medians = np.array([per_task[name]["median"] for _, name in TASKS], dtype=np.float64)
    within_task_variances = np.array(
        [per_task[name]["variance_ddof1"] for _, name in TASKS], dtype=np.float64
    )
    pooled_within_variance = float(np.nanmean(within_task_variances))
    between_task_variance = float(np.var(task_medians, ddof=1))

    report = {
        "ru": "RU-86",
        "primary_task": PRIMARY_TASK,
        "n_folds_per_task": N_FOLDS,
        "contamination_check": {
            "max_ensemble_fold_diff_vs_baseline": max_ens_diff_all,
            "invalid": contamination_invalid,
        },
        "primary_confirmatory": {
            "task": PRIMARY_TASK,
            "n_valid_folds": primary["n_valid_folds"],
            "n_invalid_folds": primary["n_invalid_folds"],
            "k_gt_0.5": primary["k_gt_0.5"],
            "verdict": primary_verdict,
        },
        "per_task": {
            name: {
                "n_valid_folds": per_task[name]["n_valid_folds"],
                "n_invalid_folds": per_task[name]["n_invalid_folds"],
                "k_gt_0.5": per_task[name]["k_gt_0.5"],
                "median": per_task[name]["median"],
                "q1": per_task[name]["q1"],
                "q3": per_task[name]["q3"],
                "variance_ddof1": per_task[name]["variance_ddof1"],
                "max_ensemble_fold_diff_vs_baseline": per_task[name][
                    "max_ensemble_fold_diff_vs_baseline"
                ],
            }
            for _, name in TASKS
        },
        "secondary_exploratory": {
            "note": "NOT used to re-rank or re-select the confirmatory task (card §criteria warning)",
            "per_task_binomial": secondary,
            "task_medians_gt0.5_count": int((task_medians > 0.5).sum()),
            "pooled_within_task_variance": pooled_within_variance,
            "between_task_variance_of_medians": between_task_variance,
            "between_over_within_ratio": (
                between_task_variance / pooled_within_variance
                if pooled_within_variance and pooled_within_variance > 0
                else None
            ),
        },
        "raw_fold_auroc": {name: per_task[name]["fold_auroc"] for _, name in TASKS},
    }

    out_dir = ROOT / "predictions" / "ru86_gate2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=True))

    print(f"contamination max_ens_diff = {max_ens_diff_all:.3e}  invalid={contamination_invalid}")
    print(
        f"[1st/confirmatory] {PRIMARY_TASK}: valid={primary['n_valid_folds']} "
        f"invalid={primary['n_invalid_folds']} k={primary['k_gt_0.5']}/{N_FOLDS} "
        f"-> {primary_verdict}"
    )
    print("\n[2nd/exploratory] per-task k / median / IQR:")
    for _, name in TASKS:
        t = per_task[name]
        print(
            f"  {name:12s} valid={t['n_valid_folds']:2d} invalid={t['n_invalid_folds']:2d} "
            f"k={t['k_gt_0.5']:2d}/{N_FOLDS} median={t['median']:.4f} "
            f"IQR=[{t['q1']:.4f},{t['q3']:.4f}] var={t['variance_ddof1']:.5f} "
            f"binom_p={secondary[name].get('p_value')}"
        )
    print(
        f"\nbetween-task variance (of medians) = {between_task_variance:.5f}  "
        f"pooled within-task variance = {pooled_within_variance:.5f}  "
        f"ratio = {report['secondary_exploratory']['between_over_within_ratio']}"
    )
    print(f"\nwritten -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
