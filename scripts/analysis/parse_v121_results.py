import glob
import sys
from pathlib import Path
import torch
import numpy as np

# `scripts/analysis/` is two levels below the repo root, so the repo root has
# to go on sys.path before `src` imports resolve (matches drop2_furthest_detail.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import auroc as compute_auroc  # noqa: E402

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
    "v120 Baseline (6-br)": "v120_active",
    "v121 Baseline (5-br, CT=0)": "v121_baseline",
    "DS Standalone Full": "ds_w1_primary7",
    "DS Salience Anchor": "ds_salience_anchor_s5_f07_a15",
    "v121 + DS Salience": "v121_salience_anchor_s5_f07_a15",
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

print("\n" + "=" * 125)
header_str = "| " + "Task".ljust(32) + " | " + " | ".join([a.ljust(16) for a in arms]) + " |"
print(header_str)
print("|" + "-" * 34 + "|" + "|".join(["-" * 18 for _ in arms]) + "|")

for t in primary7:
    task_short = t.replace("cptac_", "").replace("ucla_lung_", "")
    row = "| " + task_short.ljust(32) + " | "
    vals = []
    for a in arms:
        v = results[a].get(t)
        vals.append(f"{v:.4f}".ljust(16) if v is not None else "N/A".ljust(16))
    row += " | ".join(vals) + " |"
    print(row)

print("|" + "=" * 34 + "|" + "|".join(["=" * 18 for _ in arms]) + "|")
row = "| " + "MACRO MEAN (Primary 7)".ljust(32) + " | "
vals = []
for a in arms:
    task_vals = [results[a].get(t) for t in primary7 if results[a].get(t) is not None]
    if len(task_vals) == len(primary7):
        vals.append(f"{np.mean(task_vals):.4f}".ljust(16))
    else:
        vals.append("N/A".ljust(16))
row += " | ".join(vals) + " |"
print(row)
print("=" * 125 + "\n")
