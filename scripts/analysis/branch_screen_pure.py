"""Gate-1 admission screen on the PURE runner's margins (PROJECT.md SS5).

`scripts/analysis/branch_screen.py` reads legacy tags
(predictions/pathobench_*_m_*_official50_bf16.pt), which the pure runner no longer
produces. This screen reads the pure margins dump (predictions/margins_7b format:
per_fold[i]["branch_margins"][branch]) so gate-1 works without the dead SCREEN_ONLY
path: the pure runner is a standalone process and cannot contaminate the ensemble.

Rule (unchanged, SS5): reject if max |r| > 0.6 before any performance is consulted;
reject if rank efficiency drops. Performance is printed afterwards for the record.

  python scripts/analysis/branch_screen_pure.py \
      --margins predictions/margins_7b --candidate sj \
      --reference cv,bm,bd,qa,ds,sh
"""

from __future__ import annotations

import argparse
import glob
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.branch_diagnostics import (  # noqa: E402
    effective_rank, corr_matrix, trimmed_mean, auroc,
)

REJECT_ABOVE = 0.6
DEFAULT_REF = ("cv", "bm", "bd", "qa", "ds", "sh", "sj")


def load_pure(margins: Path) -> dict:
    out = {}
    for f in sorted(glob.glob(str(margins / "*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        task = d["task"]
        folds = []
        for pf in d["per_fold"]:
            rec = {"label": pf["label"]}
            for b, v in pf["branch_margins"].items():
                rec[f"m_{b}"] = v.float()
            folds.append(rec)
        out[task] = folds
    return out


def max_abs_corr(data: dict, ref: list[str], cand: str) -> tuple[float, dict]:
    """Worst |correlation| of the candidate against the reference, per task."""
    worst, per_task = 0.0, {}
    for t, folds in data.items():
        row = corr_matrix(folds, ref + [cand])[-1, :-1]
        v = float(np.abs(row).max())
        per_task[t] = v
        worst = max(worst, v)
    return worst, per_task


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margins", type=Path, required=True)
    ap.add_argument("--candidate", required=True, help="branch name, e.g. sj")
    ap.add_argument("--reference", default=",".join(DEFAULT_REF))
    args = ap.parse_args()

    data = load_pure(args.margins)
    cand = f"m_{args.candidate}"
    ref = [f"m_{b.strip()}" for b in args.reference.split(",") if b.strip()]
    ref = [r for r in ref if r != cand]
    print(f"candidate={cand}  reference={ref}")

    print(f"\nSTEP 1 - correlation (reject if |r| > {REJECT_ABOVE})")
    worst, per_task = max_abs_corr(data, ref, cand)
    for t, v in per_task.items():
        print(f"{t:<34} max|r|={v:>7.3f}")
    print(f"max |r| = {worst:.3f} -> {'REJECT' if worst > REJECT_ABOVE else 'ADMIT'}")

    n0 = len(ref)
    r0 = np.mean([effective_rank(corr_matrix(f, ref)) for f in data.values()])
    r1 = np.mean([effective_rank(corr_matrix(f, ref + [cand])) for f in data.values()])
    print(f"\nSTEP 2 - effective rank {r0:.2f}/{n0} -> {r1:.2f}/{n0 + 1} | "
          f"efficiency {100*r0/n0:.0f}% -> {100*r1/(n0+1):.0f}%"
          + ("  <- DROPS" if (100*r1/(n0+1)) < (100*r0/n0) else "  <- holds"))

    if worst > REJECT_ABOVE:
        print("\nScreen failed; performance is not consulted.")
        return 0

    print("\nSTEP 3 - performance (post-screen record)")
    base_all, new_all, wins = [], [], 0
    for t, folds in data.items():
        solo = st.mean(float(auroc(torch.sigmoid(f[cand]), f["label"])) for f in folds)
        b = st.mean(float(auroc(trimmed_mean(torch.stack([torch.sigmoid(f[x]) for x in ref], 0)),
                                f["label"])) for f in folds)
        n = st.mean(float(auroc(trimmed_mean(torch.stack([torch.sigmoid(f[x]) for x in ref + [cand]], 0)),
                                f["label"])) for f in folds)
        base_all.append(b); new_all.append(n); wins += int(n > b)
        print(f"{t:<34} solo={solo:.4f}  base={b:.4f}  +cand={n:.4f}  Δ={n - b:+.4f}")
    bm, nm = st.mean(base_all), st.mean(new_all)
    print(f"{'MACRO':<34} base={bm:.4f}  +cand={nm:.4f}  Δ={nm - bm:+.4f}  sign {wins}/7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
