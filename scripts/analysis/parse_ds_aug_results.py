"""Parse and compare Context Sub-bag Data Augmentation results on DS branch."""

import glob
from pathlib import Path
import torch
import numpy as np

def compute_auroc(score, target):
    indices = torch.argsort(score, descending=True)
    t = target[indices].float()
    n_pos = (t == 1).sum()
    n_neg = (t == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = (t == 1).float().cumsum(0)
    fps = (t == 0).float().cumsum(0)
    tpr = torch.cat([torch.tensor([0.0]), tps / n_pos])
    fpr = torch.cat([torch.tensor([0.0]), fps / n_neg])
    return float(torch.trapz(tpr, fpr).item())

primary7 = [
    "cptac_lscc_ARID1A_mutation",
    "cptac_lscc_Histologic_Grade",
    "cptac_lscc_KEAP1_mutation",
    "cptac_luad_KRAS_mutation",
    "cptac_pda_SMAD4_mutation",
    "ucla_lung_progression_regression",
    "cptac_ccrcc_PBRM1_mutation",
]

arms = {
    "Baseline (S=1, f=1.0)": "ds_w1_primary7",
    "Case A (S=5, f=0.7)": "ds_aug_s5_f07",
    "Case B (S=10, f=0.5)": "ds_aug_s10_f05",
    "Query TTA (S=5, f=0.7)": "ds_query_tta_s5_f07",
}

results = {arm: {} for arm in arms}

for arm, tag in arms.items():
    for t in primary7:
        files = glob.glob(f"predictions/pathobench_{t}_{tag}*.pt")
        if files:
            d = torch.load(files[0], map_location="cpu", weights_only=False)
            res = d.get("per_fold", d.get("results", []))
            scores = []
            for fold in res:
                if fold is None:
                    continue
                prob = fold.get("probability")
                lbl = fold.get("label")
                if prob is not None and lbl is not None and len(lbl) >= 2 and lbl.unique().numel() >= 2:
                    scores.append(compute_auroc(prob, lbl))
            if scores:
                results[arm][t] = np.mean(scores)

print("\n" + "=" * 95)
print(f"{'Task':32s} | " + " | ".join([f"{a:13s}" for a in arms]))
print("-" * 95)

for t in primary7:
    short_t = t.replace("cptac_", "").replace("ucla_lung_", "")
    row = f"{short_t:32s} | "
    base_val = results["Baseline (S=1, f=1.0)"].get(t, float("nan"))
    for arm in arms:
        val = results[arm].get(t, float("nan"))
        if arm == "Baseline (S=1, f=1.0)" or np.isnan(val) or np.isnan(base_val):
            row += f"{val:13.4f} | "
        else:
            diff = val - base_val
            sign = "+" if diff >= 0 else ""
            row += f"{val:6.4f} ({sign}{diff:.3f}) | "
    print(row)

print("=" * 95)
macro_row = f"{'MACRO MEAN (Primary 7)':32s} | "
base_macro = np.mean([results["Baseline (S=1, f=1.0)"][t] for t in primary7 if t in results["Baseline (S=1, f=1.0)"]])
for arm in arms:
    vals = [results[arm][t] for t in primary7 if t in results[arm]]
    if vals:
        m = np.mean(vals)
        if arm == "Baseline (S=1, f=1.0)":
            macro_row += f"{m:13.4f} | "
        else:
            diff = m - base_macro
            sign = "+" if diff >= 0 else ""
            macro_row += f"{m:6.4f} ({sign}{diff:.3f}) | "
    else:
        macro_row += f"{'N/A':13s} | "
print(macro_row)
print("=" * 95 + "\n")
