import sys
import itertools
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

V120_BASELINE = 0.6265

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
    branch_names = ["CV", "CT", "BM", "BD", "QA", "DS", "DE", "SW"]
    branch_keys = ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds", "m_de", "m_sw"]

    task_data = {}
    for task_path, name in TASKS:
        task_clean = task_path.replace("/", "_")
        ckpt = Path(f"predictions/pathobench_{task_clean}_{tag}_official50_bf16.pt")
        if not ckpt.exists():
            print(f"Missing {ckpt}")
            return
        data = torch.load(ckpt, map_location="cpu")
        task_data[name] = data["per_fold"]

    # Pre-extract probabilities for each fold of each task
    # task_probs[name][fold_idx][branch_idx] = Tensor(n_query)
    task_probs = {name: [] for _, name in TASKS}
    task_labels = {name: [] for _, name in TASKS}

    for _, name in TASKS:
        folds = task_data[name]
        for f in folds:
            b_p = []
            for k in branch_keys:
                m = f.get(k)
                if m is not None and isinstance(m, torch.Tensor):
                    b_p.append(torch.sigmoid(m.float()))
                else:
                    b_p.append(None)
            task_probs[name].append(b_p)
            task_labels[name].append(f["label"])

    results = []

    # Search all combinations of size >= 3 with Trimmed Mean
    for r in range(3, 9):
        for combo in itertools.combinations(range(8), r):
            combo_names = "+".join([branch_names[i] for i in combo])
            task_aurocs = []
            for _, name in TASKS:
                f_aurocs = []
                for b_p, label in zip(task_probs[name], task_labels[name]):
                    probs = [b_p[i] for i in combo if b_p[i] is not None]
                    if len(probs) < 3:
                        continue
                    stacked = torch.stack(probs, dim=-1)
                    sum_p = torch.sum(stacked, dim=-1)
                    min_p = torch.min(stacked, dim=-1).values
                    max_p = torch.max(stacked, dim=-1).values
                    final_p = (sum_p - min_p - max_p) / (len(probs) - 2)
                    fa = compute_auroc(final_p, label)
                    if fa is not None:
                        f_aurocs.append(fa)
                if f_aurocs:
                    task_aurocs.append(sum(f_aurocs) / len(f_aurocs))
            if len(task_aurocs) == len(TASKS):
                macro = sum(task_aurocs) / len(task_aurocs)
                results.append((macro, combo_names, task_aurocs))

    results.sort(key=lambda x: x[0], reverse=True)

    print("=" * 110)
    print("Top 15 Subset Combinations with Trimmed Mean Voting across Primary 7 Tasks")
    print("=" * 110)
    header = f"{'Rank':<4} | {'Macro':<7} | {'Δ vs v120':<9} | " + " | ".join([f"{name[:5]:>5}" for _, name in TASKS]) + " | Configuration"
    print(header)
    print("-" * len(header))

    for rank, (macro, combo_names, task_aurocs) in enumerate(results[:20], 1):
        diff = macro - V120_BASELINE
        scores_str = " | ".join([f"{s:5.4f}" for s in task_aurocs])
        print(f"{rank:<4} | {macro:7.4f} | {diff:+9.4f} | {scores_str} | {combo_names}")

    print("=" * len(header))

if __name__ == "__main__":
    main()
