"""Evaluation runner for 16-Branch Dual Pool (8 Full + 8 Sub) with Context LOO Top-3 Selection.

Evaluates 8 core branches in both Full-bag and Sub-bag (S=5, frac=0.7) regimes:
CV, CT, BM, BD, QA, DS, DE, SW (total 16 candidates).
In each fold, selects the Top-3 candidates based on exact Context Leave-One-Out (LOO) AUROC,
and aggregates their predictions via Soft Voting.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.models.common.solvers import fast_context_auroc, solve_kernel_ridge
from src.models.branches.cv import cv_logits
from src.models.branches.bd import bd_features
from src.models.ct.readout import ct_margins


def parse_args():
    parser = argparse.ArgumentParser(description="16-Branch Dual Pool with Context LOO Top-3 Selection")
    parser.add_argument("--official-folds", type=str, required=True, help="Path to official folds dir")
    parser.add_argument("--features", type=str, required=True, help="Path to features h5")
    parser.add_argument("--output", type=str, required=True, help="Path to save output .pt")
    parser.add_argument("--official-nfolds", type=int, default=50, help="Number of folds to evaluate")
    parser.add_argument("--sub-s", type=int, default=5, help="Number of sub-bags per slide")
    parser.add_argument("--sub-frac", type=float, default=0.7, help="Fraction of patches per sub-bag")
    parser.add_argument("--top-k", type=int, default=3, help="Number of branches to select via LOO")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_h5_bags(h5_path: str, slide_ids: list[str]) -> list[torch.Tensor]:
    import h5py
    bags = []
    with h5py.File(h5_path, "r") as f:
        for sid in slide_ids:
            if sid in f:
                feat = torch.from_numpy(f[sid][:]).float()
            elif sid in f.get("features", {}):
                feat = torch.from_numpy(f["features"][sid][:]).float()
            else:
                raise KeyError(f"Slide {sid} not found in {h5_path}")
            bags.append(feat)
    return bags


def compute_within_slide_pca(bags: list[torch.Tensor], dim: int = 256, device="cuda") -> torch.Tensor:
    total_cov = torch.zeros(bags[0].shape[-1], bags[0].shape[-1], dtype=torch.float64, device=device)
    total_n = 0
    for b in bags:
        bf = b.to(device).double()
        mu = bf.mean(dim=0, keepdim=True)
        centered = bf - mu
        total_cov += centered.T @ centered
        total_n += bf.shape[0]
    total_cov /= max(1, total_n)
    eigvals, eigvecs = torch.linalg.eigh(total_cov)
    basis = eigvecs[:, -dim:].float()
    return basis


def get_sub_bags(bag: torch.Tensor, S: int, frac: float, base_seed: int) -> list[torch.Tensor]:
    n = bag.shape[0]
    k = max(16, int(n * frac))
    if n <= 16:
        return [bag] * S
    sub_bags = []
    for s in range(S):
        g = torch.Generator(device="cpu").manual_seed(base_seed + s * 100)
        perm = torch.randperm(n, generator=g)[:k]
        sub_bags.append(bag[perm])
    return sub_bags


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running 16-Branch Dual Pool with Top-{args.top_k} Context LOO Selection ===")
    print(f"Folds: {args.official_folds} | Sub: S={args.sub_s}, frac={args.sub_frac}")

    # Load official folds from k=all.tsv + config.yaml
    import csv
    import yaml
    folds_dir = Path(args.official_folds)
    tsv = folds_dir / "k=all.tsv"
    cfg = folds_dir / "config.yaml"
    if not (tsv.exists() and cfg.exists()):
        raise FileNotFoundError(f"official task dir needs k=all.tsv+config.yaml: {folds_dir}")
    task_col = yaml.safe_load(cfg.read_text())["task_col"]

    header = tsv.read_text().split("\n")[0].split("\t")
    fold_cols = [c for c in header if c.startswith("fold_")]
    with tsv.open() as fh:
        records = list(csv.DictReader(fh, delimiter="\t"))
    all_slide_ids = [str(r["slide_id"]).strip() for r in records]
    labels_raw = {sid: int(float(r[task_col])) for sid, r in zip(all_slide_ids, records)}
    n_classes = len(set(labels_raw.values()))
    if n_classes > 2:
        labels_raw = {s: int(labels_raw[s] != 0) for s in labels_raw}

    n_folds = min(len(fold_cols), args.official_nfolds)

    from scripts.test_pathobench import index_h5_files, load_slide_features
    h5_index = index_h5_files(Path(args.features))
    all_slide_ids = [s for s in all_slide_ids if s in h5_index]
    print(f"Preloading {len(all_slide_ids)} slide features into CPU RAM from {args.features}...")
    t0 = time.time()
    all_bags_dict = {sid: load_slide_features(sid, h5_index) for sid in all_slide_ids}
    print(f"Preloaded {len(all_bags_dict)} slides in {time.time() - t0:.2f}s")

    per_fold_aurocs = []
    results_record = []
    selection_counts = {}

    for fold_idx in range(n_folds):
        fold_col = fold_cols[fold_idx]
        ctx_sids = [sid for sid, r in zip(all_slide_ids, records) if r[fold_col] != "test"]
        qry_sids = [sid for sid, r in zip(all_slide_ids, records) if r[fold_col] == "test"]

        ctx_labels = torch.tensor([labels_raw[s] for s in ctx_sids], device=device, dtype=torch.long)
        qry_labels = torch.tensor([labels_raw[s] for s in qry_sids], device=device, dtype=torch.long)

        ctx_bags = [all_bags_dict[sid] for sid in ctx_sids]
        qry_bags = [all_bags_dict[sid] for sid in qry_sids]

        # 1. Within-slide PCA basis (256D) from context slides
        basis = compute_within_slide_pca(ctx_bags, dim=256, device=device)
        basis32 = basis[:, :32]

        # Project bags to 32D and 256D
        ctx_p32 = [(b.to(device) @ basis32) for b in ctx_bags]
        qry_p32 = [(b.to(device) @ basis32) for b in qry_bags]
        ctx_p256 = [(b.to(device) @ basis) for b in ctx_bags]
        qry_p256 = [(b.to(device) @ basis) for b in qry_bags]

        # Generate sub-bag projections (S=5, frac=0.7)
        def make_subs(bags_proj, base_offset):
            subs = []
            for i, p in enumerate(bags_proj):
                subs.append(get_sub_bags(p, args.sub_s, args.sub_frac, args.seed + fold_idx * 10000 + i * 100 + base_offset))
            return subs

        ctx_sub32 = make_subs(ctx_p32, 0)
        qry_sub32 = make_subs(qry_p32, 5000)
        ctx_sub256 = make_subs(ctx_p256, 10000)
        qry_sub256 = make_subs(qry_p256, 15000)

        candidates = []

        # =========================================================================
        # 1. BM (Bag-Mean Ridge, 32D)
        # =========================================================================
        # Full:
        bm_ctx_f = torch.stack([p.mean(dim=0) for p in ctx_p32])
        bm_qry_f = torch.stack([p.mean(dim=0) for p in qry_p32])
        m, loo = solve_kernel_ridge(bm_ctx_f, ctx_labels, bm_qry_f, reg_lambda=1.0, return_loo=True)
        candidates.append(("BM_Full", fast_context_auroc(loo, ctx_labels), m))

        # Sub:
        bm_ctx_s = torch.stack([torch.stack([sb.mean(dim=0) for sb in subs]).mean(dim=0) for subs in ctx_sub32])
        bm_qry_s = torch.stack([torch.stack([sb.mean(dim=0) for sb in subs]).mean(dim=0) for subs in qry_sub32])
        m, loo = solve_kernel_ridge(bm_ctx_s, ctx_labels, bm_qry_s, reg_lambda=1.0, return_loo=True)
        candidates.append(("BM_Sub", fast_context_auroc(loo, ctx_labels), m))

        # =========================================================================
        # 2. QA (4-Quantiles [0.05, 0.10, 0.90, 0.95], 128D)
        # =========================================================================
        q_levels = torch.tensor([0.05, 0.10, 0.90, 0.95], device=device)
        def extract_qa(p):
            n = p.shape[0]
            if n <= 1:
                return p.repeat(4, 1).flatten()
            indices = (q_levels * (n - 1)).clamp(0, n - 1)
            low = indices.floor().long()
            high = indices.ceil().long()
            w = (indices - low.float())[:, None]
            sorted_p, _ = torch.sort(p, dim=0)
            q = (1.0 - w) * sorted_p[low, :] + w * sorted_p[high, :]
            return q.flatten()

        qa_ctx_f = torch.stack([extract_qa(p) for p in ctx_p32])
        qa_qry_f = torch.stack([extract_qa(p) for p in qry_p32])
        m, loo = solve_kernel_ridge(qa_ctx_f, ctx_labels, qa_qry_f, reg_lambda=1.0, return_loo=True)
        candidates.append(("QA_Full", fast_context_auroc(loo, ctx_labels), m))

        qa_ctx_s = torch.stack([torch.stack([extract_qa(sb) for sb in subs]).mean(dim=0) for subs in ctx_sub32])
        qa_qry_s = torch.stack([torch.stack([extract_qa(sb) for sb in subs]).mean(dim=0) for subs in qry_sub32])
        m, loo = solve_kernel_ridge(qa_ctx_s, ctx_labels, qa_qry_s, reg_lambda=1.0, return_loo=True)
        candidates.append(("QA_Sub", fast_context_auroc(loo, ctx_labels), m))

        # =========================================================================
        # 3. DS (Denoised Salience Bag-Mean, 32D)
        # =========================================================================
        # Sample centroids from ctx_p32
        sampled_cells = []
        for p in ctx_p32:
            nc = p.shape[0]
            if nc > 0:
                idx = torch.linspace(0, nc - 1, min(nc, 64), device=device).long()
                sampled_cells.append(p[idx])
        all_cells = torch.cat(sampled_cells, dim=0)
        K_ds = min(256, all_cells.shape[0])
        stride = all_cells.shape[0] / K_ds
        centroids = torch.nn.functional.normalize(all_cells[(torch.arange(K_ds, device=device) * stride).long()], dim=-1)

        def get_ds_soft(p):
            sim = torch.nn.functional.normalize(p, dim=-1) @ centroids.T
            soft_p = torch.nn.functional.softmax(sim * 5.0, dim=-1)
            return soft_p, soft_p.mean(dim=0)

        ctx_soft_f = [get_ds_soft(p) for p in ctx_p32]
        qry_soft_f = [get_ds_soft(p) for p in qry_p32]
        ctx_ab_f = torch.stack([x[1] for x in ctx_soft_f])

        eps = 1e-5
        m1 = (ctx_labels == 1)
        m0 = (ctx_labels == 0)
        a1 = ctx_ab_f[m1].mean(dim=0) if m1.any() else ctx_ab_f.mean(dim=0)
        a0 = ctx_ab_f[m0].mean(dim=0) if m0.any() else ctx_ab_f.mean(dim=0)
        s_ds = torch.log((a1 + eps) / (a0 + eps)).abs()

        def extract_ds_mean(p, soft_p):
            u = soft_p @ s_ds
            u_std = u.std().clamp_min(1e-6)
            w = torch.nn.functional.softmax(2.0 * (u - u.mean()) / u_std, dim=0)
            return (w.unsqueeze(-1) * p).sum(dim=0)

        ds_ctx_f = torch.stack([extract_ds_mean(p, s[0]) for p, s in zip(ctx_p32, ctx_soft_f)])
        ds_qry_f = torch.stack([extract_ds_mean(p, s[0]) for p, s in zip(qry_p32, qry_soft_f)])
        m, loo = solve_kernel_ridge(ds_ctx_f, ctx_labels, ds_qry_f, reg_lambda=1.0, return_loo=True)
        candidates.append(("DS_Full", fast_context_auroc(loo, ctx_labels), m))

        # DS Sub:
        def ds_sub_feature(subs):
            feats = []
            for sb in subs:
                sp, _ = get_ds_soft(sb)
                feats.append(extract_ds_mean(sb, sp))
            return torch.stack(feats).mean(dim=0)

        ds_ctx_s = torch.stack([ds_sub_feature(subs) for subs in ctx_sub32])
        ds_qry_s = torch.stack([ds_sub_feature(subs) for subs in qry_sub32])
        m, loo = solve_kernel_ridge(ds_ctx_s, ctx_labels, ds_qry_s, reg_lambda=1.0, return_loo=True)
        candidates.append(("DS_Sub", fast_context_auroc(loo, ctx_labels), m))

        # =========================================================================
        # 4. DE (Dual Extreme In-Subspace, 33D)
        # =========================================================================
        def extract_de(p):
            nc = p.shape[0]
            if nc <= 1:
                return torch.zeros(p.shape[-1] + 1, device=device)
            scores = p[:, 0]
            k_val = min(max(1, int(nc * 0.1)), nc)
            top_vals, top_idx = torch.topk(scores, k=k_val, largest=True)
            bot_vals, bot_idx = torch.topk(scores, k=k_val, largest=False)
            delta_z = p[top_idx].mean(dim=0) - p[bot_idx].mean(dim=0)
            diff = 0.5 * (top_vals.mean() + bot_vals.mean())
            return torch.cat([delta_z, diff.unsqueeze(0)], dim=-1)

        de_ctx_f = torch.stack([extract_de(p) for p in ctx_p32])
        de_qry_f = torch.stack([extract_de(p) for p in qry_p32])
        m, loo = solve_kernel_ridge(de_ctx_f, ctx_labels, de_qry_f, reg_lambda=1.0, return_loo=True)
        candidates.append(("DE_Full", fast_context_auroc(loo, ctx_labels), m))

        de_ctx_s = torch.stack([torch.stack([extract_de(sb) for sb in subs]).mean(dim=0) for subs in ctx_sub32])
        de_qry_s = torch.stack([torch.stack([extract_de(sb) for sb in subs]).mean(dim=0) for subs in qry_sub32])
        m, loo = solve_kernel_ridge(de_ctx_s, ctx_labels, de_qry_s, reg_lambda=1.0, return_loo=True)
        candidates.append(("DE_Sub", fast_context_auroc(loo, ctx_labels), m))

        # =========================================================================
        # 5. SW (Sliced Wasserstein, 128D)
        # =========================================================================
        g_sw = torch.Generator(device="cpu").manual_seed(42)
        dirs_rand = torch.randn(32, 32, generator=g_sw, dtype=torch.float32)
        q_dirs, _ = torch.linalg.qr(dirs_rand)
        slice_dirs = q_dirs.to(device)
        sw_q_levels = torch.linspace(0.5 / 4, 1.0 - 0.5 / 4, 4, device=device)

        def extract_sw(p):
            nc = p.shape[0]
            if nc == 0:
                return torch.zeros(32 * 4, device=device)
            slices = p @ slice_dirs
            sorted_slices, _ = torch.sort(slices, dim=0)
            idx = (sw_q_levels * (nc - 1)).clamp(0, nc - 1)
            low_idx = idx.floor().long()
            high_idx = idx.ceil().long()
            w = (idx - low_idx.float())[:, None]
            q = (1.0 - w) * sorted_slices[low_idx, :] + w * sorted_slices[high_idx, :]
            return q.flatten()

        sw_ctx_f = torch.stack([extract_sw(p) for p in ctx_p32])
        sw_qry_f = torch.stack([extract_sw(p) for p in qry_p32])
        m, loo = solve_kernel_ridge(sw_ctx_f, ctx_labels, sw_qry_f, reg_lambda=1.0, return_loo=True)
        candidates.append(("SW_Full", fast_context_auroc(loo, ctx_labels), m))

        sw_ctx_s = torch.stack([torch.stack([extract_sw(sb) for sb in subs]).mean(dim=0) for subs in ctx_sub32])
        sw_qry_s = torch.stack([torch.stack([extract_sw(sb) for sb in subs]).mean(dim=0) for subs in qry_sub32])
        m, loo = solve_kernel_ridge(sw_ctx_s, ctx_labels, sw_qry_s, reg_lambda=1.0, return_loo=True)
        candidates.append(("SW_Sub", fast_context_auroc(loo, ctx_labels), m))

        # =========================================================================
        # 6. BD (Bag Dispersion Spectral Entropy, 1D)
        # =========================================================================
        def extract_bd_entropy(p256):
            cov = (p256.T @ p256) / max(1, p256.shape[0])
            eigs = torch.linalg.eigvalsh(cov).clamp_min(1e-6)
            prob = eigs / eigs.sum()
            ent = -(prob * torch.log(prob)).sum() / np.log(256.0)
            return ent.unsqueeze(0)

        bd_ctx_f = torch.stack([extract_bd_entropy(p) for p in ctx_p256])
        bd_qry_f = torch.stack([extract_bd_entropy(p) for p in qry_p256])
        m, loo = solve_kernel_ridge(bd_ctx_f, ctx_labels, bd_qry_f, reg_lambda=1.0, return_loo=True)
        candidates.append(("BD_Full", fast_context_auroc(loo, ctx_labels), m))

        bd_ctx_s = torch.stack([torch.stack([extract_bd_entropy(sb) for sb in subs]).mean(dim=0) for subs in ctx_sub256])
        bd_qry_s = torch.stack([torch.stack([extract_bd_entropy(sb) for sb in subs]).mean(dim=0) for subs in qry_sub256])
        m, loo = solve_kernel_ridge(bd_ctx_s, ctx_labels, bd_qry_s, reg_lambda=1.0, return_loo=True)
        candidates.append(("BD_Sub", fast_context_auroc(loo, ctx_labels), m))

        # =========================================================================
        # 7. CV (Off-diagonal Covariance Ridge, 32,640D)
        # =========================================================================
        tri_idx = torch.triu_indices(256, 256, offset=1)
        def extract_cv_offdiag(p256):
            cov = (p256.T @ p256) / max(1, p256.shape[0])
            return cov[tri_idx[0], tri_idx[1]]

        cv_ctx_f = torch.stack([extract_cv_offdiag(p) for p in ctx_p256])
        cv_qry_f = torch.stack([extract_cv_offdiag(p) for p in qry_p256])
        cv_lambda = 1.0 * (cv_ctx_f.shape[-1] / 32.0)
        m, loo = solve_kernel_ridge(cv_ctx_f, ctx_labels, cv_qry_f, reg_lambda=cv_lambda, return_loo=True)
        candidates.append(("CV_Full", fast_context_auroc(loo, ctx_labels), m))

        cv_ctx_s = torch.stack([torch.stack([extract_cv_offdiag(sb) for sb in subs]).mean(dim=0) for subs in ctx_sub256])
        cv_qry_s = torch.stack([torch.stack([extract_cv_offdiag(sb) for sb in subs]).mean(dim=0) for subs in qry_sub256])
        m, loo = solve_kernel_ridge(cv_ctx_s, ctx_labels, cv_qry_s, reg_lambda=cv_lambda, return_loo=True)
        candidates.append(("CV_Sub", fast_context_auroc(loo, ctx_labels), m))

        # =========================================================================
        # 8. CT (Cell-Type Abundance Soft K-Means, 256D)
        # =========================================================================
        # Sample context cells for K-Means tokens
        ct_sampled = []
        for p in ctx_p32:
            nc = p.shape[0]
            if nc > 0:
                idx = torch.linspace(0, nc - 1, min(nc, 64), device=device).long()
                ct_sampled.append(p[idx])
        ct_all = torch.cat(ct_sampled, dim=0)
        K_ct = min(256, ct_all.shape[0])
        stride_ct = ct_all.shape[0] / K_ct
        ct_tokens = torch.nn.functional.normalize(ct_all[(torch.arange(K_ct, device=device) * stride_ct).long()], dim=-1)

        def extract_ct_ab(p):
            sim = torch.nn.functional.normalize(p, dim=-1) @ ct_tokens.T
            soft = torch.nn.functional.softmax(sim * 5.0, dim=-1)
            return soft.mean(dim=0)

        ct_ctx_f = torch.stack([extract_ct_ab(p) for p in ctx_p32])
        ct_qry_f = torch.stack([extract_ct_ab(p) for p in qry_p32])
        m, loo = solve_kernel_ridge(ct_ctx_f, ctx_labels, ct_qry_f, reg_lambda=1.0, return_loo=True)
        candidates.append(("CT_Full", fast_context_auroc(loo, ctx_labels), m))

        ct_ctx_s = torch.stack([torch.stack([extract_ct_ab(sb) for sb in subs]).mean(dim=0) for subs in ctx_sub32])
        ct_qry_s = torch.stack([torch.stack([extract_ct_ab(sb) for sb in subs]).mean(dim=0) for subs in qry_sub32])
        m, loo = solve_kernel_ridge(ct_ctx_s, ctx_labels, ct_qry_s, reg_lambda=1.0, return_loo=True)
        candidates.append(("CT_Sub", fast_context_auroc(loo, ctx_labels), m))

        # =========================================================================
        # Dynamic Diverse Top-K Selection based on Context LOO AUROC
        # =========================================================================
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_k_branches = []
        seen_families = set()
        for bname, b_score, q_m in candidates:
            fam = bname.split("_")[0]
            if fam not in seen_families:
                seen_families.add(fam)
                top_k_branches.append((bname, b_score, q_m))
                if len(top_k_branches) == args.top_k:
                    break

        for bname, b_score, _ in top_k_branches:
            selection_counts[bname] = selection_counts.get(bname, 0) + 1

        top_names_str = ", ".join([f"{b[0]} ({b[1]:.3f})" for b in top_k_branches])

        # Soft Voting among Top-K branches
        top_probs = [torch.sigmoid(b[2]) for b in top_k_branches]
        final_prob = torch.stack(top_probs, dim=-1).mean(dim=-1)

        # Compute query fold AUROC
        fold_auroc = fast_context_auroc(final_prob, qry_labels)
        per_fold_aurocs.append(fold_auroc)

        print(f"  fold {fold_idx + 1}/{n_folds}: AUROC {fold_auroc:.4f}  |  Top-{args.top_k}: {top_names_str}")

        results_record.append({
            "probability": final_prob.cpu(),
            "label": qry_labels.cpu(),
            "top_branches": [b[0] for b in top_k_branches],
            "top_scores": [b[1] for b in top_k_branches],
        })

    mean_auroc = np.mean(per_fold_aurocs)
    std_auroc = np.std(per_fold_aurocs)
    print(f"\n=== {n_folds}-Fold Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f} ===")
    print("\nBranch Selection Frequency across folds:")
    for bname, count in sorted(selection_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {bname:10s}: {count:2d}/{n_folds} ({count/n_folds*100:.1f}%)")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fold_mean_auroc": mean_auroc,
        "fold_std_auroc": std_auroc,
        "per_fold": results_record,
        "selection_counts": selection_counts,
    }, args.output)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
