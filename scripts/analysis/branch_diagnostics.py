"""Branch-level diagnostics from saved 50-fold prediction files.

Computes, without any GPU work, from `predictions/pathobench_<task>_<tag>_official50_bf16.pt`:
  1. per-branch standalone AUROC per task,
  2. the oracle ceiling (best single branch per task) and its gap to Trimmed Mean,
  3. optionally (--ablate) every branch-subset Trimmed Mean with sign agreement.

Sign agreement (n/7 tasks improved over the reference) is always reported alongside
macro AUROC, per the Reporting Integrity Contract in docs/agent_handoff.md.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

PRIMARY7 = [
    "cptac_lscc_ARID1A_mutation",
    "cptac_lscc_Histologic_Grade",
    "cptac_lscc_KEAP1_mutation",
    "cptac_luad_KRAS_mutation",
    "cptac_pda_SMAD4_mutation",
    "ucla_lung_progression_regression",
    "cptac_ccrcc_PBRM1_mutation",
]
BRANCHES = ["m_cv", "m_bm", "m_bd", "m_qa", "m_ds"]  # CT excluded: official 5-branch basis


def auroc(score: torch.Tensor, target: torch.Tensor) -> float:
    idx = torch.argsort(score, descending=True)
    t = target[idx].float()
    n_pos, n_neg = (t == 1).sum(), (t == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = torch.cat([torch.tensor([0.0]), (t == 1).float().cumsum(0) / n_pos])
    fpr = torch.cat([torch.tensor([0.0]), (t == 0).float().cumsum(0) / n_neg])
    return float(torch.trapz(tpr, fpr).item())


def trimmed_mean(probs: torch.Tensor) -> torch.Tensor:
    """probs: [B, N]. Drops one min and one max per slide when B >= 3."""
    if probs.shape[0] < 3:
        return probs.mean(dim=0)
    sorted_p, _ = torch.sort(probs, dim=0)
    return sorted_p[1:-1].mean(dim=0)


def load(tag: str) -> dict:
    out = {}
    for t in PRIMARY7:
        path = f"predictions/pathobench_{t}_{tag}_official50_bf16.pt"
        out[t] = torch.load(path, map_location="cpu", weights_only=False)["per_fold"]
    return out


def subset_macro(data: dict, subset: tuple[str, ...]) -> tuple[float, list[float]]:
    per_task = []
    for t in PRIMARY7:
        scores = [
            auroc(trimmed_mean(torch.stack([torch.sigmoid(f[b]) for b in subset], dim=0)), f["label"])
            for f in data[t]
        ]
        per_task.append(float(np.mean(scores)))
    return float(np.mean(per_task)), per_task


def short(task: str) -> str:
    return task.replace("cptac_", "").replace("ucla_lung_", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v121_baseline")
    ap.add_argument("--ablate", action="store_true", help="also sweep all branch subsets")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    data = load(args.tag)
    base_macro, base_per_task = subset_macro(data, tuple(BRANCHES))

    # --- per-branch standalone + oracle ceiling ---
    print(f"\n[{args.tag}] per-branch standalone AUROC (50-fold mean)")
    header = f"{'task':<24}" + "".join(f"{b[2:].upper():>8}" for b in BRANCHES)
    print(header + f"{'ORACLE':>9}{'TRIM':>8}{'gap':>9}")
    oracles = []
    for t, trim_score in zip(PRIMARY7, base_per_task):
        means = {}
        for b in BRANCHES:
            means[b] = float(np.mean([auroc(torch.sigmoid(f[b]), f["label"]) for f in data[t]]))
        best = max(means.values())
        oracles.append(best)
        row = f"{short(t):<24}" + "".join(f"{means[b]:>8.4f}" for b in BRANCHES)
        print(row + f"{best:>9.4f}{trim_score:>8.4f}{best - trim_score:>+9.4f}")
    oracle_macro = float(np.mean(oracles))
    print(f"{'MACRO':<24}{'':>40}{oracle_macro:>9.4f}{base_macro:>8.4f}{oracle_macro - base_macro:>+9.4f}")
    print(f"\noracle gap = {oracle_macro - base_macro:+.4f} "
          f"(recoverable headroom if the best branch per task were selectable)")

    if not args.ablate:
        return

    # --- branch subset ablation ---
    results = []
    for r in range(2, len(BRANCHES) + 1):
        for sub in itertools.combinations(BRANCHES, r):
            macro, per_task = subset_macro(data, sub)
            wins = sum(1 for a, b in zip(per_task, base_per_task) if a > b)
            results.append(("+".join(b[2:].upper() for b in sub), macro, wins))
    results.sort(key=lambda x: -x[1])

    print(f"\n[{args.tag}] branch-subset ablation (Trimmed Mean), reference = 5-branch {base_macro:.4f}")
    print(f"{'subset':<22}{'macro':>8}{'delta':>9}{'sign agr.':>11}")
    for name, macro, wins in results[: args.top]:
        print(f"{name:<22}{macro:>8.4f}{macro - base_macro:>+9.4f}{f'{wins}/7':>11}")
    promotable = [r for r in results if r[2] >= 5]
    print(f"\nsubsets meeting the >=5/7 promotion bar: {len(promotable)}"
          + (f" -> {[r[0] for r in promotable]}" if promotable else " (none)"))


if __name__ == "__main__":
    main()
