import sys
import torch
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
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/eval_combinations.py <tag>")
        sys.exit(1)
    tag = sys.argv[1]

    branch_keys = ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds", "m_de", "m_sw"]
    task_data = {}

    for task_path, name in TASKS:
        task_clean = task_path.replace("/", "_")
        ckpt = Path(f"predictions/pathobench_{task_clean}_{tag}_official50_bf16.pt")
        if not ckpt.exists():
            print(f"Missing {ckpt}")
            continue
        data = torch.load(ckpt, map_location="cpu")
        task_data[name] = data["per_fold"]

    if len(task_data) != len(TASKS):
        print(f"Loaded {len(task_data)}/{len(TASKS)} tasks.")
        return

    print("=" * 100)
    print(f"Comprehensive Multi-Branch & Ensemble Evaluation for TAG: {tag}")
    print("=" * 100)

    # 1. Standalone AUROCs of each branch
    print("\n--- 1. Standalone Branch Performance (50-fold Mean AUROC) ---")
    header = f"{'Branch':<12} | " + " | ".join([f"{name[:6]:>6}" for _, name in TASKS]) + " | Macro"
    print(header)
    print("-" * len(header))

    branch_macros = {}
    for b in branch_keys:
        b_scores = []
        for _, name in TASKS:
            folds = task_data[name]
            f_aurocs = []
            for f in folds:
                m = f.get(b)
                y = f["label"]
                if m is not None and isinstance(m, torch.Tensor):
                    val = compute_auroc(m.float(), y)
                    if val is not None:
                        f_aurocs.append(val)
            if f_aurocs:
                mean_a = sum(f_aurocs) / len(f_aurocs)
                b_scores.append(mean_a)
            else:
                b_scores.append(0.0)
        macro = sum(b_scores) / len(b_scores) if b_scores else 0.0
        branch_macros[b] = macro
        row = f"{b:<12} | " + " | ".join([f"{s:6.4f}" for s in b_scores]) + f" | {macro:6.4f}"
        print(row)

    # 2. Ensemble Comparisons
    print("\n--- 2. Ensemble Strategy Comparisons ---")
    ensembles = {
        "v120_6branch_trimmed" : ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds"],
        "7branch_de_trimmed"   : ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds", "m_de"],
        "7branch_sw_trimmed"   : ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds", "m_sw"],
        "8branch_all_trimmed"  : ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds", "m_de", "m_sw"],
        "8branch_all_mean"     : ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds", "m_de", "m_sw"],
    }

    ens_header = f"{'Ensemble Configuration':<26} | " + " | ".join([f"{name[:6]:>6}" for _, name in TASKS]) + " | Macro  | Δ vs v120"
    print(ens_header)
    print("-" * len(ens_header))

    for ens_name, b_list in ensembles.items():
        ens_scores = []
        is_mean_only = ens_name.endswith("_mean")
        for _, name in TASKS:
            folds = task_data[name]
            f_aurocs = []
            for f in folds:
                probs = []
                for b in b_list:
                    m = f.get(b)
                    if m is not None and isinstance(m, torch.Tensor):
                        probs.append(torch.sigmoid(m.float()))
                if len(probs) >= 3 and not is_mean_only:
                    stacked = torch.stack(probs, dim=-1)
                    sum_p = torch.sum(stacked, dim=-1)
                    min_p = torch.min(stacked, dim=-1).values
                    max_p = torch.max(stacked, dim=-1).values
                    final_p = (sum_p - min_p - max_p) / (len(probs) - 2)
                elif probs:
                    final_p = torch.stack(probs, dim=-1).mean(dim=-1)
                else:
                    continue
                val = compute_auroc(final_p, f["label"])
                if val is not None:
                    f_aurocs.append(val)
            mean_a = sum(f_aurocs) / len(f_aurocs) if f_aurocs else 0.0
            ens_scores.append(mean_a)

        macro = sum(ens_scores) / len(ens_scores) if ens_scores else 0.0
        diff = macro - V120_BASELINE["Macro"]
        diff_str = f"{diff:+6.4f}"
        row = f"{ens_name:<26} | " + " | ".join([f"{s:6.4f}" for s in ens_scores]) + f" | {macro:6.4f} | {diff_str}"
        print(row)
    print("=" * len(ens_header))

if __name__ == "__main__":
    main()
