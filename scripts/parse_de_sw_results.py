import sys
import torch
from pathlib import Path

def compute_auroc(scores: torch.Tensor, labels: torch.Tensor):
    """Mann-Whitney U with tie handling."""
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


TASKS = [
    "cptac_lscc/ARID1A_mutation",
    "cptac_lscc/Histologic_Grade",
    "cptac_lscc/KEAP1_mutation",
    "cptac_luad/KRAS_mutation",
    "cptac_pda/SMAD4_mutation",
    "ucla_lung/progression_regression",
    "cptac_ccrcc/PBRM1_mutation",
]

TASK_NAMES = [
    "ARID1A",
    "Grade",
    "KEAP1",
    "KRAS",
    "SMAD4",
    "Progression",
    "PBRM1",
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/parse_de_sw_results.py <tag>")
        sys.exit(1)

    tag = sys.argv[1]
    scores = {}
    print("=" * 80)
    print(f"Evaluation Results for TAG: {tag}")
    print("=" * 80)
    print(f"{'Task':<20} | {'AUROC':<10} | {'v120 Base':<10} | {'Diff (Δ)':<10}")
    print("-" * 80)

    for task_path, name in zip(TASKS, TASK_NAMES):
        task_clean = task_path.replace("/", "_")
        ckpt = Path(f"predictions/pathobench_{task_clean}_{tag}_official50_bf16.pt")
        if not ckpt.exists():
            print(f"{name:<20} | {'MISSING':<10} | {V120_BASELINE[name]:<10.4f} | {'N/A':<10}")
            continue
        data = torch.load(ckpt, map_location="cpu")
        macro = float(data.get("fold_auroc_mean", 0.0))
        scores[name] = macro
        base = V120_BASELINE[name]
        diff = macro - base
        diff_str = f"{diff:+.4f}"
        print(f"{name:<20} | {macro:<10.4f} | {base:<10.4f} | {diff_str:<10}")

    print("=" * 80)
    if len(scores) == len(TASKS):
        total_macro = sum(scores.values()) / len(scores)
        diff_macro = total_macro - V120_BASELINE["Macro"]
        print(f"{'Primary 7 Macro':<20} | {total_macro:<10.4f} | {V120_BASELINE['Macro']:<10.4f} | {diff_macro:+.4f}")
        print("=" * 80)
    else:
        print(f"Completed {len(scores)}/{len(TASKS)} tasks.")

if __name__ == "__main__":
    main()
