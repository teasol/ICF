"""Does a candidate branch move the oracle ceiling?

The Primary 7 macro cannot resolve differences under ~1%p (§214-V), and the fold
count and hold-out are fixed, so macro is a poor research signal. The oracle
ceiling -- the mean over tasks of the best single-branch AUROC -- asks a sharper
question: does this candidate carry information no existing branch has? It moved
for §213's salience anchor and did not move for §218's SH, where macro
distinguished neither.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from scripts.analysis.branch_diagnostics import BRANCHES, PRIMARY7, auroc, short


def solo(folds: list, key: str) -> float:
    return float(np.mean([auroc(torch.sigmoid(f[key]), f["label"]) for f in folds]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--candidates", required=True, help="comma-separated margin keys")
    args = ap.parse_args()

    cands = [c.strip() for c in args.candidates.split(",") if c.strip()]
    data = {t: torch.load(f"predictions/pathobench_{t}_{args.tag}_official50_bf16.pt",
                          map_location="cpu", weights_only=False)["per_fold"]
            for t in PRIMARY7}

    base_oracle, rows = [], []
    for t in PRIMARY7:
        base = {b: solo(data[t], b) for b in BRANCHES}
        best_b, best_v = max(base.items(), key=lambda kv: kv[1])
        base_oracle.append(best_v)
        cand_v = {c: (solo(data[t], c) if data[t][0].get(c) is not None else float("nan"))
                  for c in cands}
        rows.append((t, best_b[2:].upper(), best_v, cand_v))

    names = [c[2:].upper() for c in cands]
    print(f"{'task':<24}{'incumbent':>10}{'best':>8}" + "".join(f"{n:>9}" for n in names))
    for t, bb, bv, cv in rows:
        marks = "".join(
            f"{cv[c]:>8.4f}" + ("*" if cv[c] > bv else " ") for c in cands
        )
        print(f"{short(t):<24}{bb:>10}{bv:>8.4f}{marks}")

    print(f"\n{'oracle ceiling':<24}{'':>10}{np.mean(base_oracle):>8.4f}")
    print("(* = candidate beats the incumbent best branch on that task)\n")

    any_move = False
    for c, n in zip(cands, names):
        new_oracle = np.mean([max(bv, cv[c]) for _, _, bv, cv in rows])
        wins = sum(1 for _, _, bv, cv in rows if cv[c] > bv)
        above = sum(1 for _, _, _, cv in rows if cv[c] > 0.5)
        moved = new_oracle > np.mean(base_oracle) + 1e-6
        any_move |= moved
        print(f"{n:<6} solo>0.5 on {above}/7 | beats incumbent on {wins}/7 | "
              f"oracle {np.mean(base_oracle):.4f} -> {new_oracle:.4f} "
              f"({new_oracle - np.mean(base_oracle):+.4f}) {'MOVES' if moved else 'no move'}")

    joint = np.mean([max([bv] + [cv[c] for c in cands if not np.isnan(cv[c])])
                     for _, _, bv, cv in rows])
    print(f"\nall candidates together: oracle -> {joint:.4f} "
          f"({joint - np.mean(base_oracle):+.4f})")
    print("\nPRE-DECLARED STOPPING RULE (§219): if no variant moves the oracle "
          "ceiling, the SH axis is closed." if not any_move else
          "\nAt least one variant moves the ceiling; proceed to the two admission gates.")


if __name__ == "__main__":
    main()
