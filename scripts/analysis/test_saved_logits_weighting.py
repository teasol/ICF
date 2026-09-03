import torch
import numpy as np
from pathlib import Path
from scipy.optimize import nnls

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
    branch_names = ["CV", "CT", "BM", "BD", "QA", "DS", "DE", "SW"]
    branch_keys = ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds", "m_de", "m_sw"]
    B = len(branch_keys)

    task_data = {}
    for task_path, name in TASKS:
        task_clean = task_path.replace("/", "_")
        ckpt = Path(f"predictions/pathobench_{task_clean}_{tag}_official50_bf16.pt")
        if not ckpt.exists():
            print(f"Missing {ckpt}")
            return
        data = torch.load(ckpt, map_location="cpu")
        task_data[name] = data["per_fold"]

    print("=" * 105)
    print("Testing In-Context & Cross-Fold Weight Optimization on Saved 8-Branch Logits (50 Folds x 7 Tasks)")
    print("=" * 105)

    # Strategy 1: Cross-Fold Task-Specific Shrunk NNLS Stacking
    # For fold k in task T: optimize w_k on the other 49 folds of task T
    print("\n--- Strategy 1: Cross-Fold Stacking (Leave-One-Fold-Out per Task) ---")
    shrinkage_alphas = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]

    for alpha in shrinkage_alphas:
        task_scores = []
        for _, name in TASKS:
            folds = task_data[name]
            f_aurocs = []
            for k in range(len(folds)):
                # Training data: other 49 folds
                train_M_list, train_y_list = [], []
                for j in range(len(folds)):
                    if j == k: continue
                    f_j = folds[j]
                    y_j = f_j["label"].float()
                    # (N_j, B)
                    M_j = torch.stack([f_j[b].float() for b in branch_keys], dim=-1)
                    train_M_list.append(M_j)
                    train_y_list.append(y_j * 2.0 - 1.0) # {-1, 1}

                train_M = torch.cat(train_M_list, dim=0).numpy()
                train_y = torch.cat(train_y_list, dim=0).numpy()

                # Add prior shrinkage towards uniform weights (1/B):
                # We append sqrt(alpha) * I to train_M and sqrt(alpha) * (1/B) to train_y
                if alpha > 0.0:
                    prior_M = np.sqrt(alpha) * np.eye(B)
                    prior_y = np.sqrt(alpha) * np.ones(B) / B
                    aug_M = np.vstack([train_M, prior_M])
                    aug_y = np.concatenate([train_y, prior_y])
                else:
                    aug_M = train_M
                    aug_y = train_y

                # Solve Non-Negative Least Squares (w >= 0)
                w, _ = nnls(aug_M, aug_y)
                if w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = np.ones(B) / B

                # Predict on held-out fold k
                test_f = folds[k]
                test_M = torch.stack([test_f[b].float() for b in branch_keys], dim=-1)
                test_pred = test_M @ torch.tensor(w, dtype=torch.float32)
                fa = compute_auroc(test_pred, test_f["label"])
                if fa is not None:
                    f_aurocs.append(fa)

            mean_a = sum(f_aurocs) / len(f_aurocs) if f_aurocs else 0.0
            task_scores.append(mean_a)

        macro = sum(task_scores) / len(task_scores)
        diff = macro - V120_BASELINE["Macro"]
        task_str = " | ".join([f"{s:5.4f}" for s in task_scores])
        print(f"Shrinkage alpha={alpha:<5} | Macro: {macro:6.4f} ({diff:+6.4f}) | {task_str}")

    # Strategy 2: Cross-Fold Power-AUROC Weighting
    print("\n--- Strategy 2: Cross-Fold AUROC Power Weighting (q_b = max(0, R_b - 0.5)^gamma) ---")
    gammas = [0.5, 1.0, 2.0, 3.0, 4.0, 8.0]
    for gamma in gammas:
        task_scores = []
        for _, name in TASKS:
            folds = task_data[name]
            f_aurocs = []

            # Pre-gather all folds
            all_m_by_branch = {b: torch.cat([folds[j][b].float() for j in range(len(folds))]) for b in branch_keys}
            all_y = torch.cat([folds[j]["label"] for j in range(len(folds))])

            # Task-level AUROC for each branch
            R = [compute_auroc(all_m_by_branch[b], all_y) for b in branch_keys]
            R = [r if r is not None else 0.5 for r in R]

            q = [max(0.0, r - 0.50) ** gamma for r in R]
            sum_q = sum(q)
            if sum_q > 0:
                w = [v / sum_q for v in q]
            else:
                w = [1.0 / B] * B

            for k in range(len(folds)):
                test_f = folds[k]
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
        print(f"AUROC Power gamma={gamma:<3} | Macro: {macro:6.4f} ({diff:+6.4f}) | {task_str}")


if __name__ == "__main__":
    main()
