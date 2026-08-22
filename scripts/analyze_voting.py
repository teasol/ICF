"""Analyze voting and ensembling methods offline using saved branch logits."""

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

SEAL_10_TASKS = [
    "bc_therapy_er_status",
    "bc_therapy_grade",
    "bc_therapy_her2_status",
    "cptac_brca_PIK3CA_mutation",
    "cptac_brca_TP53_mutation",
    "cptac_luad_EGFR_mutation",
    "cptac_luad_STK11_mutation",
    "cptac_luad_TP53_mutation",
    "cptac_ccrcc_BAP1_mutation",
    "cptac_ccrcc_VHL_mutation",
]


def evaluate_file(path: Path) -> dict[str, float]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    results = data.get("per_fold", data.get("results", {}))
    if isinstance(results, list):
        results = {k: v for k, v in enumerate(results) if v is not None}

    methods = [
        "v119_soft",
        "v118_soft",
        "v117_linear",
        "v116_linear",
        "v116_soft",
        "median_voting",
        "rank_voting",
        "zscore_voting",
        "cv_only",
        "dd_only",
        "ct_only",
        "bm_only",
        "bd_only",
        "qa_only",
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
        m_qa = fold_data.get("m_qa")

        if m_cv is None or m_ct is None or m_bm is None or m_bd is None:
            # Fallback to saved probability
            prob = fold_data["probability"].float()
            fa = compute_auroc(prob, label)
            fold_scores["v118_soft"].append(fa)
            continue

        m_cv = m_cv.float()
        m_dd = m_dd.float() if m_dd is not None else torch.zeros_like(m_cv)
        m_ct = m_ct.float()
        m_bm = m_bm.float()
        m_bd = m_bd.float()
        m_qa = m_qa.float() if m_qa is not None else None
        n = len(label)

        # 1. v119 Soft Voting (5-branch: CV + CT + BM + BD + QA)
        if m_qa is not None:
            p_v119 = (
                torch.sigmoid(m_cv)
                + torch.sigmoid(m_ct)
                + torch.sigmoid(m_bm)
                + torch.sigmoid(m_bd)
                + torch.sigmoid(m_qa)
            ) / 5.0
            fold_scores["v119_soft"].append(compute_auroc(p_v119, label))
            fold_scores["qa_only"].append(compute_auroc(m_qa, label))

        # 2. v118 Soft Voting (Active Baseline: 4-branch CV + CT + BM + BD)
        p_v118 = (
            torch.sigmoid(m_cv)
            + torch.sigmoid(m_ct)
            + torch.sigmoid(m_bm)
            + torch.sigmoid(m_bd)
        ) / 4.0

        # 3. v117 Linear Sum (4-branch: CV + CT + BM + BD)
        m_v117 = m_cv + m_ct + m_bm + m_bd
        p_v117 = torch.sigmoid(m_v117)

        # 4. v116 Linear Sum (5-branch: CV + DD + CT + BM + BD)
        m_v116 = m_cv + m_dd + m_ct + m_bm + m_bd
        p_v116 = torch.sigmoid(m_v116)

        # 5. v116 Soft Voting (5-branch)
        p_v116_soft = (
            torch.sigmoid(m_cv)
            + torch.sigmoid(m_dd)
            + torch.sigmoid(m_ct)
            + torch.sigmoid(m_bm)
            + torch.sigmoid(m_bd)
        ) / 5.0

        # 6. Median Voting (5-branch)
        stacked = torch.stack([m_cv, m_dd, m_ct, m_bm, m_bd], dim=-1)
        m_med = torch.median(stacked, dim=-1).values
        p_med = torch.sigmoid(m_med)

        # 7. Percentile Rank Voting (4-branch)
        def rank_score(x):
            order = torch.argsort(x)
            ranks = torch.empty_like(x)
            ranks[order] = torch.arange(len(x), dtype=torch.float32)
            return ranks / max(1.0, float(len(x) - 1))

        p_rank = (
            rank_score(m_cv)
            + rank_score(m_ct)
            + rank_score(m_bm)
            + rank_score(m_bd)
        ) / 4.0

        # 8. Z-Score Voting (4-branch)
        def zscore(x):
            std = x.std(unbiased=False).clamp_min(1e-6)
            return (x - x.mean()) / std

        m_z = zscore(m_cv) + zscore(m_ct) + zscore(m_bm) + zscore(m_bd)
        p_z = torch.sigmoid(m_z)

        fold_scores["v118_soft"].append(compute_auroc(p_v118, label))
        fold_scores["v117_linear"].append(compute_auroc(p_v117, label))
        fold_scores["v116_linear"].append(compute_auroc(p_v116, label))
        fold_scores["v116_soft"].append(compute_auroc(p_v116_soft, label))
        fold_scores["median_voting"].append(compute_auroc(p_med, label))
        fold_scores["rank_voting"].append(compute_auroc(p_rank, label))
        fold_scores["zscore_voting"].append(compute_auroc(p_z, label))
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
    parser.add_argument("--tag", type=str, default="qa_w1_primary7", help="Evaluation tag")
    parser.add_argument("--suite", type=str, default="primary7", choices=["primary7", "seal10", "all"], help="Benchmark suite")
    parser.add_argument("--dir", type=str, default="predictions", help="Directory with .pt files")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    print(f"=== Analyzing Voting & Ensembling Mechanisms for Tag: {args.tag} (Suite: {args.suite}) ===")

    task_list = (
        PRIMARY_7_TASKS if args.suite == "primary7"
        else SEAL_10_TASKS if args.suite == "seal10"
        else PRIMARY_7_TASKS + SEAL_10_TASKS
    )

    task_results: dict[str, dict[str, float]] = {}

    for task_name in task_list:
        pt_path = find_pt_file(dir_path, task_name, args.tag)
        if pt_path is None or not pt_path.exists():
            print(f"Warning: file for {task_name} with tag {args.tag} not found.")
            continue
        res = evaluate_file(pt_path)
        task_results[task_name] = res

    if not task_results:
        print("No task results found.")
        return

    # Check if QA is available
    has_qa = any(not torch.isnan(torch.tensor(task_results[t]["qa_only"])) for t in task_results if "qa_only" in task_results[t])

    # Print Table
    methods = [
        ("v119_soft", "v119 Soft (5B: +QA)"),
        ("v118_soft", "v118 Soft (4B Base)"),
        ("v117_linear", "v117 Linear (4B)"),
        ("v116_linear", "v116 Linear (5B)"),
        ("median_voting", "Median Voting"),
        ("rank_voting", "Percentile Rank"),
    ]
    if not has_qa:
        methods = [m for m in methods if m[0] != "v119_soft"]

    header = f"| Task | " + " | ".join(name for _, name in methods) + " |"
    sep = "| :--- | " + " | ".join(":---:" for _ in methods) + " |"
    print("\n" + header)
    print(sep)

    for task_name in task_list:
        if task_name not in task_results:
            continue
        res = task_results[task_name]
        short_name = task_name.replace("cptac_", "").replace("ucla_", "").replace("_mutation", "")
        row = f"| `{short_name}` | " + " | ".join(f"{res[k]:.4f}" for k, _ in methods) + " |"
        print(row)

    # Macro Averages
    macro_row = f"| **{args.suite.upper()} Macro** | " + " | ".join(
        f"**{sum(task_results[t][k] for t in task_results)/len(task_results):.4f}**"
        for k, _ in methods
    ) + " |"
    print(macro_row)

    # Standalone Branches Table
    standalones = [
        ("cv_only", "CV alone"),
        ("ct_only", "CT alone"),
        ("bm_only", "BM alone"),
        ("bd_only", "BD alone"),
        ("qa_only", "QA alone"),
    ]
    if not has_qa:
        standalones = [s for s in standalones if s[0] != "qa_only"]

    print(f"\n\n### Standalone Branch Performance ({args.suite.upper()} Single Branch AUROC)")
    st_header = f"| Task | " + " | ".join(name for _, name in standalones) + " |"
    st_sep = "| :--- | " + " | ".join(":---:" for _ in standalones) + " |"
    print(st_header)
    print(st_sep)
    for task_name in task_list:
        if task_name not in task_results:
            continue
        res = task_results[task_name]
        short_name = task_name.replace("cptac_", "").replace("ucla_", "").replace("_mutation", "")
        row = f"| `{short_name}` | " + " | ".join(f"{res[k]:.4f}" for k, _ in standalones) + " |"
        print(row)

    st_macro = f"| **{args.suite.upper()} Macro** | " + " | ".join(
        f"**{sum(task_results[t][k] for t in task_results)/len(task_results):.4f}**"
        for k, _ in standalones
    ) + " |"
    print(st_macro)


if __name__ == "__main__":
    main()
