"""Admission screen for candidate branches (§217 RM, §218 BS/SH).

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

# Adopted but not promoted into the default ensemble: SH (§218), SHJ (§220).
# Gate 1 screens candidates against these too - they are real branches - while
# BRANCHES stays the official 5-branch comparison basis (invariant 3).
ADOPTED = ["m_sh", "m_shj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v121_rm_screen")
    ap.add_argument("--candidate", default="m_rm",
                    help="comma-separated candidate margin keys, e.g. m_bs,m_sh")
    ap.add_argument("--adopted", default=",".join(ADOPTED),
                    help="comma-separated already-adopted branches to screen against in "
                         "addition to BRANCHES (gate 1 must see SH/SHJ). Pass '' to disable.")
    args = ap.parse_args()

    cands = [c.strip() for c in args.candidate.split(",") if c.strip()]
    wanted = [a.strip() for a in args.adopted.split(",") if a.strip()]
    data = {}
    for t in PRIMARY7:
        folds = torch.load(f"predictions/pathobench_{t}_{args.tag}_official50_bf16.pt",
                           map_location="cpu", weights_only=False)["per_fold"]
        for c in cands:
            if folds[0].get(c) is None:
                raise SystemExit(f"{c} missing for {t} - was the screening run enabled?")
        data[t] = folds

    # Adopted-but-unpromoted branches (SH §218, SHJ §220) are part of the gate-1
    # reference set but NOT of BRANCHES, which stays the official 5-branch
    # comparison basis (agent_handoff.md invariant 3). Report presence explicitly:
    # a silently dropped reference would let a candidate pass a screen it never faced.
    adopted = [a for a in wanted if all(data[t][0].get(a) is not None for t in PRIMARY7)]
    missing = [a for a in wanted if a not in adopted]
    print(f"gate-1 reference = BRANCHES({len(BRANCHES)}) + adopted{adopted}")
    if missing:
        print(f"  WARNING: {missing} absent from tag '{args.tag}' - NOT screened against. "
              f"Gate 1 is incomplete for these; re-run with a tag that emits them.")

    admitted = [screen_one(data, c, cands, adopted) for c in cands]
    admitted = [c for c in admitted if c]
    if len(cands) > 1 and len(admitted) > 1:
        print(f"\nJOINT: admitted candidates {admitted} together")
        r_base = np.mean([effective_rank(corr_matrix(data[t], BRANCHES)) for t in PRIMARY7])
        r_joint = np.mean([effective_rank(corr_matrix(data[t], BRANCHES + admitted)) for t in PRIMARY7])
        n = len(BRANCHES)
        print(f"  effective rank {r_base:.2f}/{n} -> {r_joint:.2f}/{n + len(admitted)} "
              f"| efficiency {100 * r_base / n:.0f}% -> {100 * r_joint / (n + len(admitted)):.0f}%")


def screen_one(data: dict, cand: str, siblings: list[str],
               adopted: list[str] | None = None) -> str | None:
    names = [b[2:].upper() for b in BRANCHES]
    tag_c = cand[2:].upper()
    others = [c for c in siblings if c != cand] + [a for a in (adopted or []) if a != cand]
    ref = BRANCHES + others
    ref_names = names + [c[2:].upper() for c in others]

    # ---- STEP 1: correlation screen (no labels used) ----
    print(f"\n{'=' * 72}\nSTEP 1 - correlation screen for {tag_c} (reject if |r| > {REJECT_ABOVE})")
    print(f"{'task':<24}" + "".join(f"{n:>8}" for n in ref_names) + f"{'max|r|':>9}")
    worst = 0.0
    for t in PRIMARY7:
        c = corr_matrix(data[t], ref + [cand])
        row = c[-1, :-1]
        worst = max(worst, float(np.abs(row).max()))
        print(f"{short(t):<24}" + "".join(f"{v:>8.3f}" for v in row) + f"{np.abs(row).max():>9.3f}")
    verdict = "REJECT" if worst > REJECT_ABOVE else "ADMIT"
    print(f"\nmax |r| across all tasks/branches = {worst:.3f}  ->  {verdict}")

    # ---- STEP 2: effective rank contribution (no labels used) ----
    n0 = len(BRANCHES)
    r5 = np.mean([effective_rank(corr_matrix(data[t], BRANCHES)) for t in PRIMARY7])
    r6 = np.mean([effective_rank(corr_matrix(data[t], BRANCHES + [cand])) for t in PRIMARY7])
    eff0, eff1 = 100 * r5 / n0, 100 * r6 / (n0 + 1)
    print(f"\nSTEP 2 - effective rank: {r5:.2f}/{n0} -> {r6:.2f}/{n0 + 1} "
          f"({r6 - r5:+.2f} signals) | efficiency {eff0:.0f}% -> {eff1:.0f}%"
          + ("  <- DROPS" if eff1 < eff0 else "  <- holds"))

    if verdict == "REJECT":
        print("Screen failed; performance is not consulted.")
        return None

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
    return cand


if __name__ == "__main__":
    main()
