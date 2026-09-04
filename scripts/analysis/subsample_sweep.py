"""How strong must salience-anchor subsampling be before it does anything? (§225)

§223 found the shipped setting (anchor 0.15, background-keep 0.7 -- 74.5% of
patches retained) inert: macro moves +0.03%p against the full bag, the largest
per-task move is 0.74%p, and the five draws correlate at 0.9999. That could mean
subsampling is useless, or simply that removing a quarter of the background is not
a perturbation. This distinguishes the two by sweeping the background-keep
fraction down to anchors-only.

This measures effect SIZE, not which setting scores best. The pre-declared reading:
if no strength moves macro by at least the benchmark's ~1%p resolution floor, the
axis closes; a setting that merely scores highest is not thereby promoted.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import torch

from scripts.analysis.branch_diagnostics import PRIMARY7, auroc, short

ANCHOR = 0.15
FLOOR = 0.01  # Primary 7 resolution floor, §214-V


def icc(draws: torch.Tensor) -> float:
    x = draws.float()
    between = x.mean(dim=0).var(unbiased=False).item()
    within = x.var(dim=0, unbiased=False).mean().item()
    return between / (between + within) if between + within > 0 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    data = {t: torch.load(f"predictions/pathobench_{t}_{args.tag}_official50_bf16.pt",
                          map_location="cpu", weights_only=False)["per_fold"]
            for t in PRIMARY7}
    # "m_ds_full" also starts with m_ds_f, so match the digits explicitly.
    keys = sorted([k for k in data[PRIMARY7[0]][0]
                   if re.fullmatch(r"m_ds_f\d{3}", k)], reverse=True)
    if not keys:
        raise SystemExit("no sweep arms found - run with ICF_DS_SUBSAMPLE_SWEEP set")

    full = {t: float(np.mean([auroc(torch.sigmoid(f["m_ds_full"].float()), f["label"])
                              for f in data[t]])) for t in PRIMARY7}
    full_macro = float(np.mean(list(full.values())))

    print(f"full-bag DS macro = {full_macro:.4f}\n")
    print(f"{'keep':>6}{'retained':>10}" + "".join(f"{short(t)[:9]:>10}" for t in PRIMARY7)
          + f"{'macro':>9}{'delta':>9}{'ICC':>8}")
    rows = []
    for k in keys:
        frac = int(k.split("f")[-1]) / 100.0
        per, iccs = [], []
        for t in PRIMARY7:
            per.append(float(np.mean([auroc(torch.sigmoid(f[k].float()), f["label"])
                                      for f in data[t]])))
            d = data[t][0].get(f"draws_ds_f{int(round(frac*100)):03d}")
            if d is not None:
                iccs.append(float(np.mean([icc(f[f"draws_ds_f{int(round(frac*100)):03d}"])
                                           for f in data[t]])))
        macro = float(np.mean(per))
        rows.append((frac, macro, macro - full_macro, per))
        print(f"{frac:>6.2f}{ANCHOR + (1-ANCHOR)*frac:>9.1%}"
              + "".join(f"{v:>10.4f}" for v in per)
              + f"{macro:>9.4f}{macro - full_macro:>+9.4f}"
              + (f"{np.mean(iccs):>8.4f}" if iccs else f"{'-':>8}"))

    print(f"{'full':>6}{1.0:>9.1%}" + "".join(f"{full[t]:>10.4f}" for t in PRIMARY7)
          + f"{full_macro:>9.4f}{0.0:>+9.4f}{'-':>8}")

    biggest = max(rows, key=lambda r: abs(r[2]))
    per_task = max(abs(v - full[t]) for _, _, _, per in rows for t, v in zip(PRIMARY7, per))
    print(f"\nlargest macro move: {biggest[2]:+.4f} at keep={biggest[0]:.2f}")
    print(f"largest per-task move across all arms: {per_task:.4f}")
    print(f"resolution floor: {FLOOR:.4f}")
    print("VERDICT: " + ("some strength clears the floor - subsampling has a real effect"
                         if abs(biggest[2]) >= FLOOR else
                         "no strength clears the floor - close the axis"))


if __name__ == "__main__":
    main()
