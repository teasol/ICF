"""Trim-rate diagnostic: which branch is the trimmed min/max, per task? (C-20260919-3)

The trimmed_mean drops one min and one max probability per query slide. If SJ is
the least informative branch on the tasks where the 7-branch worsens, it should
be the trimmed min more often there. This is derivable from the stored per-branch
margins (no trim mask was saved), so it is GPU 0.

  python scripts/analysis/trim_rate_diagnostic.py --margins predictions/margins_7b
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BRANCHES = ("cv", "bm", "bd", "qa", "ds", "sh", "sj")
WORSEN = {"Histologic_Grade", "progression_regression", "PBRM1_mutation"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margins", type=Path, required=True)
    args = ap.parse_args()

    per_task = {}
    for f in sorted(glob.glob(str(args.margins / "*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        mn, mx, n = Counter(), Counter(), 0
        for pf in d["per_fold"]:
            probs = torch.stack([torch.sigmoid(pf["branch_margins"][b].float())
                                 for b in BRANCHES], dim=-1)  # [n_slide, 7]
            idx_min = probs.argmin(dim=-1)
            idx_max = probs.argmax(dim=-1)
            for i in idx_min.tolist():
                mn[BRANCHES[i]] += 1
            for i in idx_max.tolist():
                mx[BRANCHES[i]] += 1
            n += probs.shape[0]
        per_task[d["task"]] = (mn, mx, n)

    hdr = "".join(f"{b:>7}" for b in BRANCHES)
    print(f"{'task':<34}{hdr}  (trimmed-MIN rate %)")
    for t, (mn, mx, n) in per_task.items():
        tag = "W" if t.split("/")[-1] in WORSEN else " "
        print(f"{t:<34}" + "".join(f"{100*mn[b]/n:>7.1f}" for b in BRANCHES) + f"  {tag}")
    print(f"\n{'task':<34}{hdr}  (trimmed-MAX rate %)")
    for t, (mn, mx, n) in per_task.items():
        tag = "W" if t.split("/")[-1] in WORSEN else " "
        print(f"{t:<34}" + "".join(f"{100*mx[b]/n:>7.1f}" for b in BRANCHES) + f"  {tag}")

    w = [v for t, v in per_task.items() if t.split("/")[-1] in WORSEN]
    o = [v for t, v in per_task.items() if t.split("/")[-1] not in WORSEN]
    print("\n평균 trimmed-MIN rate (%) — 악화 vs 나머지")
    for b in BRANCHES:
        ra = sum(100*v[0][b]/v[2] for v in w) / len(w)
        rb = sum(100*v[0][b]/v[2] for v in o) / len(o)
        print(f"  {b:>4}: 악화 {ra:5.1f}  나머지 {rb:5.1f}  차 {ra-rb:+5.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
