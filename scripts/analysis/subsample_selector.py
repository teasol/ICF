"""Choose between subsampled and full DS from draw stability alone (§222).

Salience-anchor subsampling is a high-variance technique: on DS standalone it wins
big on ARID1A (+7.65%p) and Grade (+1.90%p) and loses on the other five, so always-full
scores 0.6265, always-subsampled 0.6059, and a perfect per-task choice 0.6401.

Context LOO cannot make that choice (§212, §221): context performance does not
predict query performance, because the two are separated by distribution shift.
This selector uses no labels at all. Subsampling averages S views of a slide; if the
views agree, the background it discarded was redundant, and if they disagree, the
signal lives in rare patches that individual draws miss -- exactly when subsampling
hurts. §213 saw the same thing from the other side: KRAS, described there as a local
mutation, lost the most (-9.67%p).

Stability is measured as an intraclass correlation over the per-draw margins,

    ICC = var_between_slides / (var_between_slides + var_within_draws),

the share of margin variance that is real between-slide signal rather than draw
noise. The rule is pre-declared as ICC > 0.5 -- the conventional "moderate
reliability" cut, and the same 0.50 floor §212 used -- so nothing is fitted to the
seven evaluation tasks.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from scripts.analysis.branch_diagnostics import PRIMARY7, auroc, short

ICC_THRESHOLD = 0.50  # pre-declared, not tuned


def draw_icc(draws: torch.Tensor) -> float:
    """draws: [S, n_query] margins, one per subsample draw."""
    x = draws.float()
    between = x.mean(dim=0).var(unbiased=False).item()
    within = x.var(dim=0, unbiased=False).mean().item()
    total = between + within
    return between / total if total > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    rows = []
    for t in PRIMARY7:
        folds = torch.load(f"predictions/pathobench_{t}_{args.tag}_official50_bf16.pt",
                           map_location="cpu", weights_only=False)["per_fold"]
        sub, full, iccs, sel = [], [], [], []
        for f in folds:
            if f.get("m_ds_full") is None or f.get("m_ds_draws") is None:
                raise SystemExit(f"{t}: comparison arm missing - run with ICF_DS_SUBSAMPLE_COMPARE=1")
            s = auroc(torch.sigmoid(f["m_ds"].float()), f["label"])
            u = auroc(torch.sigmoid(f["m_ds_full"].float()), f["label"])
            icc = draw_icc(f["m_ds_draws"])
            sub.append(s); full.append(u); iccs.append(icc)
            sel.append(s if icc > ICC_THRESHOLD else u)
        rows.append((t, float(np.mean(full)), float(np.mean(sub)),
                     float(np.mean(iccs)), float(np.mean(sel)),
                     sum(1 for i in iccs if i > ICC_THRESHOLD)))

    print(f"{'task':<24}{'full':>8}{'sub':>8}{'oracle':>8}{'ICC':>8}{'sub-folds':>11}{'selected':>10}")
    for t, u, s, icc, sl, n in rows:
        print(f"{short(t):<24}{u:>8.4f}{s:>8.4f}{max(u, s):>8.4f}{icc:>8.3f}{f'{n}/50':>11}{sl:>10.4f}")

    full_m = float(np.mean([r[1] for r in rows]))
    sub_m = float(np.mean([r[2] for r in rows]))
    orc_m = float(np.mean([max(r[1], r[2]) for r in rows]))
    sel_m = float(np.mean([r[4] for r in rows]))
    print(f"\n{'always full':<28}{full_m:.4f}")
    print(f"{'always subsampled':<28}{sub_m:.4f}")
    print(f"{'per-fold ICC>0.5 selector':<28}{sel_m:.4f}  ({sel_m - full_m:+.4f} vs full)")
    print(f"{'oracle (perfect per task)':<28}{orc_m:.4f}")
    denom = orc_m - full_m
    if denom > 0:
        print(f"\nrecovery of the available headroom: {100 * (sel_m - full_m) / denom:.0f}%")
    agree = sum(1 for r in rows if (r[2] > r[1]) == (r[3] > ICC_THRESHOLD))
    print(f"task-level sign agreement between ICC and the right choice: {agree}/7")


if __name__ == "__main__":
    main()
