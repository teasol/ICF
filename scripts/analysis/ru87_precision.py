"""RU-87: does bf16 numerical error survive into the PAIRED per-fold Delta?

The card (docs/ru/RU-87.json) fixed the estimand and the interpretation bands
BEFORE the run. This script only computes; it does not judge and it does not
touch the card.

What the GPU sweep produced
---------------------------
scripts/run_ru87_precision.sh ran the v121 5-branch config twice, once with
`--precision bf16-mixed` (tag `ru87_bf16`) and once with `--precision 32-true`
(tag `ru87_fp32`), with `ICF_SHAPE_SCREEN_ONLY=1`. Screen-only means the SHJ
margin is recorded but never folded into the logits
(scripts/test_pathobench.py:1293), so the saved `probability` is Arm A and
nothing else. Arm B therefore does not exist as a saved ensemble and has to be
rebuilt offline from the recorded `m_shj`.

Arm definitions (PROJECT.md 3.2 Trimmed Mean convention, implemented in
src/models/aggregations/voting.py:128 `trimmed_mean`)
  Arm A = trimmed_mean over sigmoid of {m_cv, m_bm, m_bd, m_qa, m_ds}   (5)
  Arm B = trimmed_mean over sigmoid of {m_cv, m_bm, m_bd, m_qa, m_ds, m_shj} (6)
where trimmed_mean = drop exactly one min and one max across the branch axis,
average the rest, clamp to [1e-7, 1-1e-7]. AUROC is rank-based, so the final
logit transform in voting.py is order-preserving and omitted here.

Arm A is not asserted, it is verified two ways (see `verify_arm_a`):
  1. the offline 5-branch reconstruction must match the stored `probability`
     to float32 rounding, and
  2. the stored `probability` of tag `ru87_bf16` must be bit-identical to the
     official `v121_baseline` predictions.

Kill (3) of the card -- fold construction must match between the fp32 and the
bf16 arm -- is checked as a hard error, not a warning: fold_indices, per-fold
slide_id order and per-fold labels.

Quantities (card `criteria`)
  D_abs   = |AUROC_fp32 - AUROC_bf16|, per arm, per fold
  D_delta = |(B_fp32 - A_fp32) - (B_bf16 - A_bf16)|, same fold
reported as max, 95th percentile, and the per-task fold-mean difference. The
judged quantity is the maximum over tasks of the D_delta per-task fold-mean
difference; D_abs is reported alongside but is not the judged quantity.

Usage:
    python scripts/analysis/ru87_precision.py
    python scripts/analysis/ru87_precision.py --out predictions/ru87_precision/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.metrics import auroc_rows  # noqa: E402

# Primary 7, in the order scripts/run_ru87_precision.sh launched them.
TASKS = [
    "cptac_lscc_ARID1A_mutation",
    "cptac_lscc_Histologic_Grade",
    "cptac_lscc_KEAP1_mutation",
    "cptac_luad_KRAS_mutation",
    "cptac_pda_SMAD4_mutation",
    "ucla_lung_progression_regression",
    "cptac_ccrcc_PBRM1_mutation",
]

ARM_A_BRANCHES = ["m_cv", "m_bm", "m_bd", "m_qa", "m_ds"]
ARM_B_BRANCHES = ARM_A_BRANCHES + ["m_shj"]

BF16_TAG = "ru87_bf16"
FP32_TAG = "ru87_fp32"
OFFICIAL_TAG = "v121_baseline"


class PairingError(RuntimeError):
    """The two precision arms cannot be compared fold by fold."""


def prediction_path(task: str, tag: str) -> Path:
    return ROOT / "predictions" / f"pathobench_{task}_{tag}_official50_bf16.pt"


def load(task: str, tag: str) -> dict:
    path = prediction_path(task, tag)
    if not path.exists():
        raise PairingError(f"missing predictions: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def trimmed_mean(margins: list[torch.Tensor]) -> torch.Tensor:
    """PROJECT.md 3.2 aggregation: sigmoid, drop one min and one max, average.

    Computed in float64 so the aggregation itself does not add precision noise
    on top of the bf16-vs-fp32 effect being measured.
    """
    stacked = torch.stack([torch.sigmoid(m.to(torch.float64)) for m in margins], dim=-1)
    count = stacked.shape[-1]
    if count < 3:
        raise ValueError(f"trimmed mean needs >= 3 branches, got {count}")
    total = stacked.sum(dim=-1)
    lo = stacked.min(dim=-1).values
    hi = stacked.max(dim=-1).values
    return ((total - lo - hi) / float(count - 2)).clamp(1e-7, 1.0 - 1e-7)


def arm_fold_aurocs(record: dict, branches: list[str]) -> torch.Tensor:
    """Per-fold AUROC of one offline-reaggregated arm."""
    values = []
    for fold in record["per_fold"]:
        missing = [b for b in branches if fold.get(b) is None]
        if missing:
            raise PairingError(f"fold is missing branch margins: {missing}")
        score = trimmed_mean([fold[b] for b in branches])
        values.append(auroc_rows(score.flatten(), fold["label"].flatten()))
    return torch.stack(values).to(torch.float64)


def verify_arm_a(task: str, record: dict, tag: str, tol: float = 1e-6) -> dict:
    """Confirm the saved ensemble output really is Arm A, not Arm A + SHJ."""
    gaps = []
    for fold in record["per_fold"]:
        rebuilt = trimmed_mean([fold[b] for b in ARM_A_BRANCHES])
        gaps.append((rebuilt - fold["probability"].to(torch.float64)).abs().max().item())
    worst = max(gaps)
    if worst > tol:
        raise PairingError(
            f"{task} [{tag}]: saved probability is not the 5-branch Arm A "
            f"(max |rebuilt - stored| = {worst:.3e} > {tol:.0e})"
        )
    return {"task": task, "tag": tag, "max_abs_gap_vs_stored_probability": worst}


def verify_matches_official(task: str) -> dict:
    """The bf16 arm must reproduce the official v121_baseline run bit-for-bit."""
    mine = load(task, BF16_TAG)
    official = load(task, OFFICIAL_TAG)
    if mine["fold_indices"] != official["fold_indices"]:
        raise PairingError(f"{task}: ru87_bf16 fold_indices differ from {OFFICIAL_TAG}")
    worst = 0.0
    for lhs, rhs in zip(mine["per_fold"], official["per_fold"]):
        worst = max(worst, (lhs["probability"] - rhs["probability"]).abs().max().item())
    return {
        "task": task,
        "max_abs_probability_gap_vs_v121_baseline": worst,
        "fold_auroc_mean_ru87_bf16": mine["fold_auroc_mean"],
        "fold_auroc_mean_v121_baseline": official["fold_auroc_mean"],
        "bit_identical": worst == 0.0,
    }


def verify_fold_construction(task: str, bf16: dict, fp32: dict) -> dict:
    """Card kill (3): mismatched fold construction makes the RU 실행 무효."""
    if len(bf16["per_fold"]) != len(fp32["per_fold"]):
        raise PairingError(
            f"{task}: fold count {len(bf16['per_fold'])} vs {len(fp32['per_fold'])}"
        )
    if bf16["fold_indices"] != fp32["fold_indices"]:
        raise PairingError(f"{task}: fold_indices differ between bf16 and fp32")
    if bf16.get("official_folds") != fp32.get("official_folds"):
        raise PairingError(f"{task}: official_folds differ between bf16 and fp32")
    if bf16.get("n_slides") != fp32.get("n_slides"):
        raise PairingError(f"{task}: n_slides differ between bf16 and fp32")
    for index, (lhs, rhs) in enumerate(zip(bf16["per_fold"], fp32["per_fold"]), start=1):
        if list(lhs["slide_id"]) != list(rhs["slide_id"]):
            raise PairingError(f"{task}: fold {index} slide_id order differs")
        if not torch.equal(lhs["label"], rhs["label"]):
            raise PairingError(f"{task}: fold {index} labels differ")
        if not torch.equal(lhs["context_label"], rhs["context_label"]):
            raise PairingError(f"{task}: fold {index} context_label differs")
    return {
        "task": task,
        "n_folds": len(bf16["per_fold"]),
        "fold_indices_match": True,
        "slide_id_order_match": True,
        "labels_match": True,
        "context_label_match": True,
        "n_slides": bf16.get("n_slides"),
    }


def stats(values: torch.Tensor) -> dict:
    """max / 95th percentile / mean of a distribution of absolute deviations."""
    return {
        "max": values.max().item(),
        "p95": torch.quantile(values, 0.95).item(),
        "mean": values.mean().item(),
        "n": int(values.numel()),
    }


def analyse(tasks: list[str]) -> dict:
    per_task = []
    pooled = {"abs_a": [], "abs_b": [], "delta": []}
    checks = {"arm_a_identity": [], "vs_official": [], "fold_construction": []}

    for task in tasks:
        bf16 = load(task, BF16_TAG)
        fp32 = load(task, FP32_TAG)

        checks["fold_construction"].append(verify_fold_construction(task, bf16, fp32))
        checks["arm_a_identity"].append(verify_arm_a(task, bf16, BF16_TAG))
        checks["arm_a_identity"].append(verify_arm_a(task, fp32, FP32_TAG))
        checks["vs_official"].append(verify_matches_official(task))

        a_bf16 = arm_fold_aurocs(bf16, ARM_A_BRANCHES)
        a_fp32 = arm_fold_aurocs(fp32, ARM_A_BRANCHES)
        b_bf16 = arm_fold_aurocs(bf16, ARM_B_BRANCHES)
        b_fp32 = arm_fold_aurocs(fp32, ARM_B_BRANCHES)

        abs_a = (a_fp32 - a_bf16).abs()
        abs_b = (b_fp32 - b_bf16).abs()
        delta = ((b_fp32 - a_fp32) - (b_bf16 - a_bf16)).abs()

        pooled["abs_a"].append(abs_a)
        pooled["abs_b"].append(abs_b)
        pooled["delta"].append(delta)

        per_task.append(
            {
                "task": task,
                "n_folds": int(a_bf16.numel()),
                "fold_mean_auroc": {
                    "A_bf16": a_bf16.mean().item(),
                    "A_fp32": a_fp32.mean().item(),
                    "B_bf16": b_bf16.mean().item(),
                    "B_fp32": b_fp32.mean().item(),
                },
                # "fold-mean difference": difference of the two fold-means, i.e.
                # the mean of the signed per-fold differences, then |.|.
                "fold_mean_diff": {
                    "D_abs_A": abs(a_fp32.mean().item() - a_bf16.mean().item()),
                    "D_abs_B": abs(b_fp32.mean().item() - b_bf16.mean().item()),
                    "D_delta": abs(
                        (b_fp32 - a_fp32).mean().item() - (b_bf16 - a_bf16).mean().item()
                    ),
                },
                # per-fold absolute deviations, summarised
                "per_fold": {
                    "D_abs_A": stats(abs_a),
                    "D_abs_B": stats(abs_b),
                    "D_delta": stats(delta),
                },
            }
        )

    def worst(key: str, sub: str) -> dict:
        rows = [(row["fold_mean_diff"][sub], row["task"]) for row in per_task]
        value, task = max(rows)
        return {"value_auroc": value, "value_pp": value * 100.0, "task": task}

    summary = {
        "ru": "RU-87",
        "arms": {
            "A": {
                "definition": "trimmed_mean over sigmoid of " + ", ".join(ARM_A_BRANCHES),
                "source": "saved ensemble output (probability) == offline rebuild",
            },
            "B": {
                "definition": "trimmed_mean over sigmoid of " + ", ".join(ARM_B_BRANCHES),
                "source": "offline re-aggregation of the recorded m_shj margin; "
                "the run was ICF_SHAPE_SCREEN_ONLY=1 so no saved ensemble contains SHJ",
            },
            "aggregation": "PROJECT.md 3.2 Trimmed Mean: drop 1 min + 1 max, "
            "average the rest (float64 offline)",
        },
        "tags": {"bf16": BF16_TAG, "fp32": FP32_TAG, "official_reference": OFFICIAL_TAG},
        "checks": checks,
        "per_task": per_task,
        "pooled_per_fold": {
            "D_abs_A": stats(torch.cat(pooled["abs_a"])),
            "D_abs_B": stats(torch.cat(pooled["abs_b"])),
            "D_delta": stats(torch.cat(pooled["delta"])),
        },
        "judged_quantity": {
            "name": "max over tasks of the per-task fold-mean D_delta",
            **worst("fold_mean_diff", "D_delta"),
            "bands_from_card": {
                "cancels": "< 0.072 pp",
                "partial": "0.072 pp <= x < 0.3 pp",
                "persists": ">= 0.3 pp",
            },
        },
        "reported_not_judged": {
            "max_fold_mean_D_abs_A": worst("fold_mean_diff", "D_abs_A"),
            "max_fold_mean_D_abs_B": worst("fold_mean_diff", "D_abs_B"),
        },
    }

    value_pp = summary["judged_quantity"]["value_pp"]
    if value_pp < 0.072:
        band = "cancels"
    elif value_pp < 0.3:
        band = "partial"
    else:
        band = "persists"
    summary["judged_quantity"]["band"] = band
    return summary


def report(summary: dict) -> None:
    print(f"arm A = {summary['arms']['A']['definition']}")
    print(f"arm B = {summary['arms']['B']['definition']}")
    print(f"aggregation: {summary['arms']['aggregation']}\n")

    identical = all(row["bit_identical"] for row in summary["checks"]["vs_official"])
    print(f"fold construction match (kill 3): OK for all "
          f"{len(summary['checks']['fold_construction'])} tasks")
    print(f"ru87_bf16 == v121_baseline bit-for-bit: {identical}\n")

    header = (f"{'task':34s} {'A_bf16':>7s} {'A_fp32':>7s} {'B_bf16':>7s} {'B_fp32':>7s}"
              f" {'|dA|pp':>7s} {'|dB|pp':>7s} {'Ddelta pp':>10s}")
    print(header)
    print("-" * len(header))
    for row in summary["per_task"]:
        m = row["fold_mean_auroc"]
        d = row["fold_mean_diff"]
        print(f"{row['task']:34s} {m['A_bf16']:7.4f} {m['A_fp32']:7.4f} "
              f"{m['B_bf16']:7.4f} {m['B_fp32']:7.4f} "
              f"{d['D_abs_A'] * 100:7.4f} {d['D_abs_B'] * 100:7.4f} "
              f"{d['D_delta'] * 100:10.4f}")
    print("-" * len(header))

    print("\npooled per-fold absolute deviation (350 folds, AUROC units):")
    for key, row in summary["pooled_per_fold"].items():
        print(f"  {key:9s} max={row['max']:.6f}  p95={row['p95']:.6f}  "
              f"mean={row['mean']:.6f}  n={row['n']}")

    judged = summary["judged_quantity"]
    print(f"\njudged quantity ({judged['name']}):")
    print(f"  {judged['value_pp']:.4f} pp   (task: {judged['task']})")
    print(f"  band: {judged['band']}  "
          f"[cancels < 0.072 pp <= partial < 0.3 pp <= persists]")
    for key, row in summary["reported_not_judged"].items():
        print(f"  reported-not-judged {key}: {row['value_pp']:.4f} pp ({row['task']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(ROOT / "predictions/ru87_precision/summary.json"),
        help="where to write the machine-readable summary",
    )
    args = parser.parse_args()

    summary = analyse(TASKS)
    report(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
