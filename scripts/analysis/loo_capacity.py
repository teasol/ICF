"""Does context-LOO reliability work for LOW-capacity branches?

§212 retired In-Episode Context LOO after finding the context LOO score
anti-correlated with test AUROC (rho = -0.27), and attributed it to capacity:
a branch with many features fits context noise smoothly, inflating its LOO, while
a conservative branch looks worse on context and generalises better.

If that diagnosis is right, the effect should scale with feature dimension, and
the lowest-capacity branches should be spared. That is testable and falsifiable:
SHJ carries 8 features, DS and BM 32, SH 64, QA 128. This script measures, per
branch, the LOO inflation and the fold-level rank correlation between the context
LOO score and the query AUROC actually achieved.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from scripts.analysis.branch_diagnostics import PRIMARY7, auroc, short

# feature dimensionality entering each branch's ridge
DIMS = {"bm": 32, "qa": 128, "ds": 32, "sh": 64, "shj": 8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--branches", default="shj,ds,bm,sh,qa")
    args = ap.parse_args()

    names = [b.strip() for b in args.branches.split(",") if b.strip()]
    data = {t: torch.load(f"predictions/pathobench_{t}_{args.tag}_official50_bf16.pt",
                          map_location="cpu", weights_only=False)["per_fold"]
            for t in PRIMARY7}

    print(f"{'branch':>7}{'dim':>6}{'ctxLOO':>9}{'query':>8}{'inflation':>11}"
          f"{'rho(ctx,qry)':>14}{'p':>10}")
    rows = []
    for b in sorted(names, key=lambda x: DIMS.get(x, 0)):
        ctx_all, qry_all = [], []
        for t in PRIMARY7:
            for f in data[t]:
                loo, cl = f.get(f"loo_{b}"), f.get("context_label")
                m = f.get(f"m_{b}")
                if loo is None or cl is None or m is None:
                    continue
                ctx_all.append(auroc(torch.sigmoid(loo.float()), cl.long()))
                qry_all.append(auroc(torch.sigmoid(m.float()), f["label"]))
        if not ctx_all:
            print(f"{b.upper():>7}{DIMS.get(b, 0):>6}   (no LOO recorded)")
            continue
        ctx, qry = np.array(ctx_all), np.array(qry_all)
        rho, pv = spearmanr(ctx, qry)
        rows.append((b, DIMS.get(b, 0), ctx.mean(), qry.mean(), rho))
        print(f"{b.upper():>7}{DIMS.get(b, 0):>6}{ctx.mean():>9.4f}{qry.mean():>8.4f}"
              f"{ctx.mean() - qry.mean():>+11.4f}{rho:>+14.3f}{pv:>10.1e}")

    if len(rows) >= 3:
        dims = np.array([r[1] for r in rows], dtype=float)
        infl = np.array([r[2] - r[3] for r in rows])
        rhos = np.array([r[4] for r in rows])
        print(f"\ncapacity hypothesis (§212 diagnosis) across {len(rows)} branches:")
        print(f"  rho(feature dim, LOO inflation) = {spearmanr(dims, infl)[0]:+.3f} "
              "-- positive means bigger branches inflate more, as §212 claimed")
        print(f"  rho(feature dim, rho(ctx,qry))  = {spearmanr(dims, rhos)[0]:+.3f} "
              "-- negative means low-capacity branches keep a usable LOO signal")
        usable = [r[0].upper() for r in rows if r[4] > 0.2]
        print(f"  branches with rho(ctx,qry) > 0.2 (LOO plausibly usable): "
              f"{usable if usable else 'none'}")


if __name__ == "__main__":
    main()
