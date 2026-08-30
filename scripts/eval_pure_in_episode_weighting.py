import torch
import numpy as np
from pathlib import Path

TASKS = [
    ("cptac_lscc/ARID1A_mutation", "ARID1A"),
    ("cptac_lscc/Histologic_Grade", "Grade"),
    ("cptac_lscc/KEAP1_mutation", "KEAP1"),
    ("cptac_luad/KRAS_mutation", "KRAS"),
    ("cptac_pda/SMAD4_mutation", "SMAD4"),
    ("ucla_lung/progression_regression", "Progression"),
    ("cptac_ccrcc/PBRM1_mutation", "PBRM1"),
]

V120_BASELINE = {
    "ARID1A": 0.5471,
    "Grade": 0.6823,
    "KEAP1": 0.6129,
    "KRAS": 0.7295,
    "SMAD4": 0.4465,
    "Progression": 0.7986,
    "PBRM1": 0.5685,
    "Macro": 0.6265,
}

def compute_auroc(scores: torch.Tensor, labels: torch.Tensor):
    labels = labels.long()
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ordered = scores[order]
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1] == ordered[index]:
            end += 1
        ranks[order[index:end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))

def main():
    tag = "ensemble_8branch_primary7"
    branch_keys = ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds", "m_de", "m_sw"]
    B = len(branch_keys)

    task_data = {}
    slide_lookup = {} # task -> {slide_id: (labels, branch_margins)}
    for task_path, name in TASKS:
        task_clean = task_path.replace("/", "_")
        ckpt = Path(f"predictions/pathobench_{task_clean}_{tag}_official50_bf16.pt")
        if not ckpt.exists():
            print(f"Missing {ckpt}")
            return
        data = torch.load(ckpt, map_location="cpu")
        task_data[name] = data["per_fold"]

        # Build slide lookup for each task
        sdict = {}
        for f in data["per_fold"]:
            sids = f["slide_id"]
            labs = f["label"]
            for i, sid in enumerate(sids):
                if sid not in sdict:
                    sdict[sid] = {
                        "label": labs[i].item(),
                        "margins": {b: f[b][i].item() for b in branch_keys}
                    }
        slide_lookup[name] = sdict

    print("=" * 110)
    print("Testing Pure In-Episode Context Weighting (Zero Leakage, Context-Only Calibration)")
    print("=" * 110)

    for gamma in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]:
        task_scores = []
        for _, name in TASKS:
            folds = task_data[name]
            sdict = slide_lookup[name]
            f_aurocs = []

            for k in range(len(folds)):
                test_f = folds[k]
                test_sids = set(test_f["slide_id"])

                # Pure context slides for fold k: all slides in cohort NOT in test_sids
                ctx_sids = [sid for sid in sdict if sid not in test_sids]

                # Compute Context AUROC R_b on ONLY the context slides of fold k
                ctx_y = torch.tensor([sdict[sid]["label"] for sid in ctx_sids], dtype=torch.long)
                R = []
                for b in branch_keys:
                    ctx_m = torch.tensor([sdict[sid]["margins"][b] for sid in ctx_sids], dtype=torch.float32)
                    r_b = compute_auroc(ctx_m, ctx_y)
                    R.append(r_b if r_b is not None else 0.50)

                # Reliability weights
                q = [max(0.0, r - 0.50) ** gamma for r in R]
                sum_q = sum(q)
                if sum_q > 0:
                    w = [v / sum_q for v in q]
                else:
                    w = [1.0 / B] * B

                # Apply weights to test query slides
                test_M = torch.stack([torch.sigmoid(test_f[b].float()) for b in branch_keys], dim=-1)
                test_pred = test_M @ torch.tensor(w, dtype=torch.float32)
                fa = compute_auroc(test_pred, test_f["label"])
                if fa is not None:
                    f_aurocs.append(fa)

            mean_a = sum(f_aurocs) / len(f_aurocs) if f_aurocs else 0.0
            task_scores.append(mean_a)

        macro = sum(task_scores) / len(task_scores)
        diff = macro - V120_BASELINE["Macro"]
        task_str = " | ".join([f"{s:5.4f}" for s in task_scores])
        print(f"In-Episode gamma={gamma:<4} | Macro: {macro:6.4f} ({diff:+6.4f}) | {task_str}")

if __name__ == "__main__":
    main()
