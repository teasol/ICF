"""Diagnostic analysis script for BagPFN v19 best checkpoint.

Evaluates branch-level logit relationships, covariance ridge vs CSP relation collision,
context-only class separation margin, and offline gating candidates across 5 synthetic tasks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

from src.modules.model_interface import ModelInterface
from src.modules.data_interface import DataInterface
from src.datasets.synthetic_data import RESPONSE_TASK_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose v19 branch logits.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/20260725_v19_medium_csp_rank1_100e/v19_medium_csp_rank1/epoch=052-val_ce_loss=0.5966.ckpt",
        help="Path to checkpoint file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_v19_medium.yaml",
        help="Path to training config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run diagnostics on.",
    )
    return parser.parse_args()


def compute_binary_auroc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute AUROC for binary classification logits [N, 2]."""
    scores = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    y_true = targets.cpu().numpy()
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    auc = np.mean(pos_scores[:, None] > neg_scores[None, :]) + 0.5 * np.mean(pos_scores[:, None] == neg_scores[None, :])
    return float(auc)


def compute_ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Cross-Entropy loss for binary classification logits [N, 2]."""
    return float(F.cross_entropy(logits, targets).item())


from src.utils.utils import merge_train_config, build_datamodule

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from: {ckpt_path}")
    model_module = ModelInterface.load_from_checkpoint(
        str(ckpt_path),
        map_location=device,
    )
    model_module.to(device)
    model_module.eval()

    # Load resolved config to get complete dataset kwargs
    config = merge_train_config(Path(args.config))
    config["data"].setdefault("dataset_kwargs", {})["response_task_probabilities"] = [0.20, 0.20, 0.20, 0.20, 0.20]
    config["data"]["dataset_kwargs"]["return_task_metadata"] = True

    data_module = build_datamodule(config)
    data_module.setup(stage=None)
    val_loader = data_module.val_dataloader()

    print(f"Evaluating diagnostic metrics over {len(val_loader)} validation batches...")

    all_targets = []
    all_task_indices = []
    all_logits = []
    
    branch_logits = {
        "global_shape": [],
        "population": [],
        "tail": [],
        "covariance_ridge": [],
        "covariance_relation": [],
    }
    separations = []

    with torch.no_grad():
        for batch in val_loader:
            x, y, mask_index, oracle_abundance, task_index = (
                model_module._unpack_evaluation_batch(batch, "val")
            )
            x = x.to(device) if isinstance(x, torch.Tensor) else x
            y = y.to(device) if isinstance(y, torch.Tensor) else y
            mask_index = mask_index.to(device)
            task_index = task_index.to(device)

            logits, aux = model_module.model(x, y, mask_index, return_auxiliary=True)
            query_targets = y[mask_index]

            all_targets.append(query_targets.cpu())
            num_queries = mask_index.numel()
            if task_index.ndim == 0 or task_index.numel() == 1:
                t_expanded = task_index.reshape(-1).repeat(num_queries)
            else:
                t_expanded = task_index.reshape(-1)
            all_task_indices.append(t_expanded.cpu())
            all_logits.append(logits.cpu())

            for name in branch_logits.keys():
                key = f"{name}_logits"
                if key in aux:
                    branch_logits[name].append(aux[key].cpu())

            if "covariance_relation_class_separation" in aux:
                sep = aux["covariance_relation_class_separation"].cpu().reshape(-1)
                if sep.numel() == 1:
                    sep = sep.repeat(num_queries)
                elif sep.numel() != num_queries and num_queries % sep.numel() == 0:
                    sep = sep.repeat_interleave(num_queries // sep.numel())
                separations.append(sep)

    targets = torch.cat(all_targets, dim=0)
    task_indices = torch.cat(all_task_indices, dim=0)
    logits = torch.cat(all_logits, dim=0)

    for name in branch_logits.keys():
        branch_logits[name] = torch.cat(branch_logits[name], dim=0)
    
    separations = torch.cat(separations, dim=0) if separations else torch.zeros(logits.shape[0])

    print("\n" + "="*80)
    print(f"{'OVERALL & PER-TASK PERFORMANCE DIAGNOSTIC':^80}")
    print("="*80)

    print(f"\nTotal Validation Queries: {len(targets)}")
    
    print("\n--- 1. Final Model & Branch AUROCs ---")
    header = f"{'Task':<15} | {'Count':<6} | {'Final':<7} | {'GS':<6} | {'Pop':<6} | {'Tail':<6} | {'CovRidge':<9} | {'CovRel':<7}"
    print(header)
    print("-" * len(header))

    overall_auc = compute_binary_auroc(logits, targets)
    
    gs_auc = compute_binary_auroc(branch_logits["global_shape"], targets)
    pop_auc = compute_binary_auroc(branch_logits["population"], targets)
    tail_auc = compute_binary_auroc(branch_logits["tail"], targets)
    cr_auc = compute_binary_auroc(branch_logits["covariance_ridge"], targets)
    crel_auc = compute_binary_auroc(branch_logits["covariance_relation"], targets)

    print(f"{'OVERALL':<15} | {len(targets):<6} | {overall_auc:.4f}  | {gs_auc:.4f} | {pop_auc:.4f} | {tail_auc:.4f} | {cr_auc:.7f}  | {crel_auc:.4f}")

    for t_idx, t_name in enumerate(RESPONSE_TASK_NAMES):
        mask = (task_indices == t_idx)
        if mask.sum() == 0:
            continue
        t_targets = targets[mask]
        t_logits = logits[mask]
        
        t_final = compute_binary_auroc(t_logits, t_targets)
        t_gs = compute_binary_auroc(branch_logits["global_shape"][mask], t_targets)
        t_pop = compute_binary_auroc(branch_logits["population"][mask], t_targets)
        t_tail = compute_binary_auroc(branch_logits["tail"][mask], t_targets)
        t_cr = compute_binary_auroc(branch_logits["covariance_ridge"][mask], t_targets)
        t_crel = compute_binary_auroc(branch_logits["covariance_relation"][mask], t_targets)
        
        print(f"{t_name:<15} | {mask.sum().item():<6} | {t_final:.4f}  | {t_gs:.4f} | {t_pop:.4f} | {t_tail:.4f} | {t_cr:.7f}  | {t_crel:.4f}")

    print("\n--- 2. Covariance Ridge vs Covariance Relation Logit Correlation ---")
    cr_diff = (branch_logits["covariance_ridge"][:, 1] - branch_logits["covariance_ridge"][:, 0]).numpy()
    crel_diff = (branch_logits["covariance_relation"][:, 1] - branch_logits["covariance_relation"][:, 0]).numpy()
    
    corr_overall = float(np.corrcoef(cr_diff, crel_diff)[0, 1])
    print(f"Overall Logit Correlation (Covariance Ridge vs CSP Relation): {corr_overall:.4f}")

    for t_idx, t_name in enumerate(RESPONSE_TASK_NAMES):
        mask = (task_indices == t_idx).numpy()
        if mask.sum() > 1:
            t_corr = float(np.corrcoef(cr_diff[mask], crel_diff[mask])[0, 1])
            print(f"  {t_name:<15}: Correlation = {t_corr:.4f}")

    print("\n--- 3. Context-Only Class Separation Margin Analysis ---")
    sep_np = separations.numpy()
    print(f"Separation Margin Mean: {sep_np.mean():.4f}, Std: {sep_np.std():.4f}, Min: {sep_np.min():.4f}, Max: {sep_np.max():.4f}")

    print("\n--- 4. Offline Simulation: Context Margin Gated CSP Relation ---")
    res_scale = 0.50
    base_logits = logits - res_scale * branch_logits["covariance_relation"]
    
    for threshold in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for alpha in [1.0, 2.0, 5.0]:
            gate = 1.0 / (1.0 + np.exp(-alpha * (sep_np[:, None] - threshold)))
            gated_logits = base_logits + res_scale * torch.from_numpy(gate).float() * branch_logits["covariance_relation"]
            
            g_auc = compute_binary_auroc(gated_logits, targets)
            g_ce = compute_ce_loss(gated_logits, targets)
            
            state_mask = (task_indices == 1)
            cov_mask = (task_indices == 2)
            comp_mask = (task_indices == 0)
            
            state_auc = compute_binary_auroc(gated_logits[state_mask], targets[state_mask])
            cov_auc = compute_binary_auroc(gated_logits[cov_mask], targets[cov_mask])
            comp_auc = compute_binary_auroc(gated_logits[comp_mask], targets[comp_mask])
            
            if g_auc >= overall_auc - 0.001 and (state_auc > 0.630 or cov_auc > 0.608):
                print(f"Gate(th={threshold:<4}, a={alpha:<3}) -> Overall AUROC: {g_auc:.4f} (CE: {g_ce:.4f}) | Comp: {comp_auc:.4f} | State: {state_auc:.4f} | Cov: {cov_auc:.4f}")

    print("\nDiagnostic evaluation completed.")

if __name__ == "__main__":
    main()
