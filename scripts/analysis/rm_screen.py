"""§218 screening for the RM (Residual Bag-Mean) branch.

Applies the admission rule recorded in docs/agent_handoff.md: a candidate branch
is screened on its correlation with the existing branches BEFORE any performance
number is consulted, and is rejected at |r| > 0.6 regardless of how well it scores.
Performance is reported afterwards, for the record only.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from scripts.analysis.branch_diagnostics import (
    BRANCHES, PRIMARY7, auroc, corr_matrix, effective_rank, short, trimmed_mean,
)

REJECT_ABOVE = 0.6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v121_rm_screen")
    ap.add_argument("--candidate", default="m_rm")
    args = ap.parse_args()

    data = {}
    for t in PRIMARY7:
        folds = torch.load(f"predictions/pathobench_{t}_{args.tag}_official50_bf16.pt",
                           map_location="cpu", weights_only=False)["per_fold"]
        if folds[0].get(args.candidate) is None:
            raise SystemExit(f"{args.candidate} missing for {t} - was the screening run enabled?")
        data[t] = folds

    cand = args.candidate
    names = [b[2:].upper() for b in BRANCHES]
    tag_c = cand[2:].upper()

    # ---- STEP 1: correlation screen (no labels used) ----
    print(f"STEP 1 - correlation screen for {tag_c} (reject if |r| > {REJECT_ABOVE})")
    print(f"{'task':<24}" + "".join(f"{n:>8}" for n in names) + f"{'max|r|':>9}")
    worst = 0.0
    for t in PRIMARY7:
        c = corr_matrix(data[t], BRANCHES + [cand])
        row = c[-1, :-1]
        worst = max(worst, float(np.abs(row).max()))
        print(f"{short(t):<24}" + "".join(f"{v:>8.3f}" for v in row) + f"{np.abs(row).max():>9.3f}")
    verdict = "REJECT" if worst > REJECT_ABOVE else "ADMIT"
    print(f"\nmax |r| across all tasks/branches = {worst:.3f}  ->  {verdict}")

    # ---- STEP 2: effective rank contribution (no labels used) ----
    r5 = np.mean([effective_rank(corr_matrix(data[t], BRANCHES)) for t in PRIMARY7])
    r6 = np.mean([effective_rank(corr_matrix(data[t], BRANCHES + [cand])) for t in PRIMARY7])
    print(f"\nSTEP 2 - effective rank: {r5:.2f}/5 -> {r6:.2f}/6 "
          f"({r6 - r5:+.2f} independent signals added)")

    if verdict == "REJECT":
        print("\nScreen failed; performance is not consulted.")
        return

    # ---- STEP 3: performance, reported only after the screen passes ----
    print(f"\nSTEP 3 - performance (post-screen record)")
    print(f"{'task':<24}{tag_c+' alone':>12}{'5-branch':>10}{'+'+tag_c:>10}{'delta':>9}")
    base_all, new_all, wins = [], [], 0
    for t in PRIMARY7:
        solo = np.mean([auroc(torch.sigmoid(f[cand]), f["label"]) for f in data[t]])
        b = np.mean([auroc(trimmed_mean(torch.stack([torch.sigmoid(f[x]) for x in BRANCHES], 0)),
                           f["label"]) for f in data[t]])
        n = np.mean([auroc(trimmed_mean(torch.stack([torch.sigmoid(f[x]) for x in BRANCHES + [cand]], 0)),
                           f["label"]) for f in data[t]])
        base_all.append(b); new_all.append(n); wins += int(n > b)
        print(f"{short(t):<24}{solo:>12.4f}{b:>10.4f}{n:>10.4f}{n - b:>+9.4f}")
    bm, nm = float(np.mean(base_all)), float(np.mean(new_all))
    print(f"{'MACRO':<24}{'':>12}{bm:>10.4f}{nm:>10.4f}{nm - bm:>+9.4f}")
    print(f"sign agreement: {wins}/7 (promotion bar is >=5/7)")


if __name__ == "__main__":
    main()
