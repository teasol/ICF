"""Analyze voting and ensembling methods offline using saved 5-branch logits."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch


def compute_auroc(score: torch.Tensor, target: torch.Tensor) -> float:
    """Compute exact AUROC without external dependency."""
    score = score.flatten().float()
    target = target.flatten().long()
    n_pos = int((target == 1).sum())
    n_neg = int((target == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Sort scores descending
    indices = torch.argsort(score, descending=True)
    target_sorted = target[indices]
    tps = (target_sorted == 1).float().cumsum(0)
    fps = (target_sorted == 0).float().cumsum(0)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # Trapezoidal rule: integrate TPR with respect to FPR
    fpr_all = torch.cat([torch.tensor([0.0]), fpr])
    tpr_all = torch.cat([torch.tensor([0.0]), tpr])
    return float(torch.trapz(tpr_all, fpr_all).item())


PRIMARY_7_TASKS = [
    "cptac_lscc_ARID1A_mutation",
    "cptac_lscc_Histologic_Grade",
    "cptac_lscc_KEAP1_mutation",
    "cptac_luad_KRAS_mutation",
    "cptac_pda_SMAD4_mutation",
    "ucla_lung_progression_regression",
    "cptac_ccrcc_PBRM1_mutation",
]


def evaluate_file(path: Path) -> dict[str, float]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    results = data.get("per_fold", data.get("results", {}))
    if isinstance(results, list):
        results = {k: v for k, v in enumerate(results) if v is not None}

    methods = [
        "v116_linear",
        "soft_voting",
        "zscore_voting",
        "median_voting",
        "rank_voting",
        "no_dd_linear",
        "no_dd_soft",
        "weighted_linear",
        "cv_only",
        "dd_only",
        "ct_only",
        "bm_only",
        "bd_only",
    ]
    fold_scores = {m: [] for m in methods}

    for fold_k, fold_data in results.items():
        if fold_data is None:
            continue
        label = fold_data["label"].long()
        if len(label) < 2 or label.unique().numel() < 2:
            continue

        m_cv = fold_data.get("m_cv")
        m_dd = fold_data.get("m_dd")
        m_ct = fold_data.get("m_ct")
        m_bm = fold_data.get("m_bm")
        m_bd = fold_data.get("m_bd")

        if m_cv is None or m_dd is None or m_ct is None or m_bm is None or m_bd is None:
            # Fallback to saved probability
            prob = fold_data["probability"].float()
            fa = compute_auroc(prob, label)
            fold_scores["v116_linear"].append(fa)
            continue

        m_cv = m_cv.float()
        m_dd = -m_dd.float()  # Align: slot 4/5 in head is (d1 - d0)
        m_ct = -m_ct.float()  # Align: slot 8/9 in head is (q1 - q0)
        m_bm = m_bm.float()
        m_bd = m_bd.float()
        n = len(label)

        # 0. v116 Linear Sum
        m_lin = m_cv + m_dd + m_ct + m_bm + m_bd
        p_lin = torch.sigmoid(m_lin)

        # 1. Soft Voting (Probability Average)
        p_soft = (
            torch.sigmoid(m_cv)
            + torch.sigmoid(m_dd)
            + torch.sigmoid(m_ct)
            + torch.sigmoid(m_bm)
            + torch.sigmoid(m_bd)
        ) / 5.0

        # 2. Z-Score Calibrated Voting
        def zscore(x):
            std = x.std(unbiased=False).clamp_min(1e-6)
            return (x - x.mean()) / std

        m_z = zscore(m_cv) + zscore(m_dd) + zscore(m_ct) + zscore(m_bm) + zscore(m_bd)
        p_z = torch.sigmoid(m_z)

        # 3. Median Voting
        stacked = torch.stack([m_cv, m_dd, m_ct, m_bm, m_bd], dim=-1)
        m_med = torch.median(stacked, dim=-1).values
        p_med = torch.sigmoid(m_med)

        # 4. Percentile Rank Voting
        def rank_score(x):
            order = torch.argsort(x)
            ranks = torch.empty_like(x)
            ranks[order] = torch.arange(len(x), dtype=torch.float32)
            return ranks / max(1.0, float(len(x) - 1))

        p_rank = (
            rank_score(m_cv)
            + rank_score(m_dd)
            + rank_score(m_ct)
            + rank_score(m_bm)
            + rank_score(m_bd)
        ) / 5.0

        # 5. No-DD Linear (4-branch: CV + CT + BM + BD)
        m_no_dd = m_cv + m_ct + m_bm + m_bd
        p_no_dd = torch.sigmoid(m_no_dd)

        # 6. No-DD Soft Voting (4-branch: CV, CT, BM, BD)
        p_no_dd_soft = (
            torch.sigmoid(m_cv)
            + torch.sigmoid(m_ct)
            + torch.sigmoid(m_bm)
            + torch.sigmoid(m_bd)
        ) / 4.0

        # 7. Weighted Linear (CV=1.0, DD=0.5, CT=1.0, BM=1.0, BD=1.0)
        m_weighted = 1.0 * m_cv + 0.5 * m_dd + 1.0 * m_ct + 1.0 * m_bm + 1.0 * m_bd
        p_weighted = torch.sigmoid(m_weighted)

        fold_scores["v116_linear"].append(compute_auroc(p_lin, label))
        fold_scores["soft_voting"].append(compute_auroc(p_soft, label))
        fold_scores["zscore_voting"].append(compute_auroc(p_z, label))
        fold_scores["median_voting"].append(compute_auroc(p_med, label))
        fold_scores["rank_voting"].append(compute_auroc(p_rank, label))
        fold_scores["no_dd_linear"].append(compute_auroc(p_no_dd, label))
        fold_scores["no_dd_soft"].append(compute_auroc(p_no_dd_soft, label))
        fold_scores["weighted_linear"].append(compute_auroc(p_weighted, label))
        fold_scores["cv_only"].append(compute_auroc(m_cv, label))
        fold_scores["dd_only"].append(compute_auroc(m_dd, label))
        fold_scores["ct_only"].append(compute_auroc(m_ct, label))
        fold_scores["bm_only"].append(compute_auroc(m_bm, label))
        fold_scores["bd_only"].append(compute_auroc(m_bd, label))

    return {m: (sum(vals) / len(vals) if vals else float("nan")) for m, vals in fold_scores.items()}


def find_pt_file(base_dir: Path, task_name: str, tag: str) -> Path | None:
    patterns = [
        f"pathobench_{task_name}_{tag}_official50_bf16.pt",
        f"pathobench_{task_name}_{tag}.pt",
        f"{task_name}_{tag}.pt",
    ]
    for search_dir in [base_dir, Path("predictions"), Path("logs/official50")]:
        for pat in patterns:
            candidate = search_dir / pat
            if candidate.exists():
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze 5-branch voting mechanisms.")
    parser.add_argument("--tag", type=str, default="v116_branch_logits", help="Evaluation tag")
    parser.add_argument("--dir", type=str, default="predictions", help="Directory with .pt files")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    print(f"=== Analyzing Voting & Ensembling Mechanisms for Tag: {args.tag} ===")

    task_results: dict[str, dict[str, float]] = {}

    for task_name in PRIMARY_7_TASKS:
        pt_path = find_pt_file(dir_path, task_name, args.tag)
        if pt_path is None or not pt_path.exists():
            print(f"Warning: file for {task_name} with tag {args.tag} not found.")
            continue
        res = evaluate_file(pt_path)
        task_results[task_name] = res

    if not task_results:
        print("No task results found.")
        return

    # Print Table
    methods = [
        ("v116_linear", "v116 Linear"),
        ("soft_voting", "Soft Voting (Prob)"),
        ("zscore_voting", "Z-Score Voting"),
        ("median_voting", "Median Voting"),
        ("rank_voting", "Percentile Rank"),
        ("no_dd_linear", "4-Branch (No-DD) Lin"),
        ("no_dd_soft", "4-Branch (No-DD) Soft"),
        ("weighted_linear", "Weighted (DD=0.5)"),
    ]

    header = f"| Task | " + " | ".join(name for _, name in methods) + " |"
    sep = "| :--- | " + " | ".join(":---:" for _ in methods) + " |"
    print("\n" + header)
    print(sep)

    for task_name in PRIMARY_7_TASKS:
        if task_name not in task_results:
            continue
        res = task_results[task_name]
        short_name = task_name.replace("cptac_", "").replace("ucla_", "").replace("_mutation", "")
        row = f"| `{short_name}` | " + " | ".join(f"{res[k]:.4f}" for k, _ in methods) + " |"
        print(row)

    # Macro Averages
    macro_row = "| **Primary 7 Macro** | " + " | ".join(
        f"**{sum(task_results[t][k] for t in task_results)/len(task_results):.4f}**"
        for k, _ in methods
    ) + " |"
    print(macro_row)

    # Delta vs Linear
    linear_macro = sum(task_results[t]["v116_linear"] for t in task_results) / len(task_results)
    delta_row = "| *Delta vs Linear* | " + " | ".join(
        f"*{sum(task_results[t][k] for t in task_results)/len(task_results) - linear_macro:+.4f}*"
        for k, _ in methods
    ) + " |"
    print(delta_row)

    # Standalone Branches Table
    standalones = [
        ("cv_only", "CV alone"),
        ("dd_only", "DD alone"),
        ("ct_only", "CT alone"),
        ("bm_only", "BM alone"),
        ("bd_only", "BD alone"),
    ]
    print("\n\n### Standalone Branch Performance (Single Branch AUROC)")
    st_header = f"| Task | " + " | ".join(name for _, name in standalones) + " |"
    st_sep = "| :--- | " + " | ".join(":---:" for _ in standalones) + " |"
    print(st_header)
    print(st_sep)
    for task_name in PRIMARY_7_TASKS:
        if task_name not in task_results:
            continue
        res = task_results[task_name]
        short_name = task_name.replace("cptac_", "").replace("ucla_", "").replace("_mutation", "")
        row = f"| `{short_name}` | " + " | ".join(f"{res[k]:.4f}" for k, _ in standalones) + " |"
        print(row)

    st_macro = "| **Primary 7 Macro** | " + " | ".join(
        f"**{sum(task_results[t][k] for t in task_results)/len(task_results):.4f}**"
        for k, _ in standalones
    ) + " |"
    print(st_macro)


if __name__ == "__main__":
    main()
