"""Fold-level significance of a candidate branch, with no cross-branch comparison.

Replaces oracle_shift.py. That script judged a candidate by whether it beat the
best incumbent branch on some task, which is unusable as a criterion (§224): the
comparison picks a branch using the same data it scores on, so it is biased upward
by the winner's curse (+0.0063 by split-half, +0.0144 on ARID1A where branches sit
close together), and it is not a ceiling either -- the ensemble exceeds the best
single branch on Prog (0.7892 against 0.7779).

This asks only about the candidate itself: on its strongest task, is it above
chance on enough of the 50 folds to be reproducible? Gate 2b requires >= 40/50
(binomial p < 0.01). Nothing here depends on how other branches score, so there is
no selection bias to correct.
"""

from __future__ import annotations

import argparse
from math import comb

import numpy as np
import torch

from scripts.analysis.branch_diagnostics import PRIMARY7, auroc, short

FOLD_BAR = 40


def binomial_tail(k: int, n: int) -> float:
    """P(X >= k) under X ~ Binomial(n, 0.5)."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--candidates", required=True, help="comma-separated margin keys")
    args = ap.parse_args()

    data = {t: torch.load(f"predictions/pathobench_{t}_{args.tag}_official50_bf16.pt",
                          map_location="cpu", weights_only=False)["per_fold"]
            for t in PRIMARY7}

    for cand in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        name = cand[2:].upper()
        print(f"\n{name}: per-task standalone AUROC and fold-level reproducibility")
        print(f"{'task':<24}{'AUROC':>8}{'>0.5 folds':>12}{'binom p':>11}")
        best = None
        for t in PRIMARY7:
            if data[t][0].get(cand) is None:
                raise SystemExit(f"{cand} missing for {t}")
            scores = np.array([auroc(torch.sigmoid(f[cand].float()), f["label"]) for f in data[t]])
            k, n = int((scores > 0.5).sum()), len(scores)
            p = binomial_tail(k, n)
            print(f"{short(t):<24}{scores.mean():>8.4f}{f'{k}/{n}':>12}{p:>11.1e}")
            if best is None or k > best[1]:
                best = (t, k, n, scores.mean(), p)
        t, k, n, mean, p = best
        verdict = "PASS" if k >= FOLD_BAR else "FAIL"
        print(f"strongest task {short(t)}: {k}/{n} folds above chance (p = {p:.1e}), "
              f"AUROC {mean:.4f}  ->  gate 2b {verdict} (bar is {FOLD_BAR}/{n})")


if __name__ == "__main__":
    main()
