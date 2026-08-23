import torch
import numpy as np
from src.utils.metrics import auroc

primary7 = [
    "cptac_lscc_ARID1A_mutation",
    "cptac_lscc_Histologic_Grade",
    "cptac_lscc_KEAP1_mutation",
    "cptac_luad_KRAS_mutation",
    "cptac_pda_SMAD4_mutation",
    "ucla_lung_progression_regression",
    "cptac_ccrcc_PBRM1_mutation",
]

seal10 = [
    "cptac_brca_PIK3CA_mutation",
    "cptac_brca_TP53_mutation",
    "cptac_ccrcc_BAP1_mutation",
    "cptac_ccrcc_VHL_mutation",
    "cptac_luad_EGFR_mutation",
    "cptac_luad_STK11_mutation",
    "cptac_luad_TP53_mutation",
    "bc_therapy_er_status",
    "bc_therapy_grade",
    "bc_therapy_her2_status",
]

branches = ["m_cv", "m_ct", "m_bm", "m_bd", "m_qa", "m_ds"]

pts_p7 = [torch.load(f"predictions/pathobench_{t}_ds_w1_primary7_official50_bf16.pt", map_location="cpu") for t in primary7]
pts_s10 = [torch.load(f"predictions/pathobench_{t}_v120_seal10_official50_bf16.pt", map_location="cpu") for t in seal10]

methods = [
    "standard_mean",
    "trimmed_mean (1 min, 1 max)",
    "drop_1_furthest_prob",
    "drop_2_furthest_prob",
    "drop_3_furthest_prob",
    "drop_1_furthest_margin",
    "drop_2_furthest_margin",
    "drop_3_furthest_margin",
    "median_prob",
]

def eval_method(pts, method):
    task_scores = []
    for pt in pts:
        fold_aucs = []
        for fold in pt["per_fold"]:
            y = fold["label"]
            m_list = [fold[b] for b in branches]
            if isinstance(m_list[0], list):
                m_list = [torch.tensor(m) for m in m_list]
            m_stack = torch.stack(m_list, dim=-1)
            p_stack = torch.sigmoid(m_stack)
            
            if method == "standard_mean":
                p = torch.mean(p_stack, dim=-1)
            elif method == "trimmed_mean (1 min, 1 max)":
                sum_p = torch.sum(p_stack, dim=-1)
                min_p = torch.min(p_stack, dim=-1).values
                max_p = torch.max(p_stack, dim=-1).values
                p = (sum_p - min_p - max_p) / 4.0
            elif method == "drop_1_furthest_prob":
                med = torch.median(p_stack, dim=-1, keepdim=True).values
                dev = (p_stack - med).abs()
                max_idx = torch.topk(dev, k=1, dim=-1).indices
                mask = torch.ones_like(p_stack)
                mask.scatter_(-1, max_idx, 0.0)
                p = (p_stack * mask).sum(dim=-1) / 5.0
            elif method == "drop_2_furthest_prob":
                med = torch.median(p_stack, dim=-1, keepdim=True).values
                dev = (p_stack - med).abs()
                max_idx = torch.topk(dev, k=2, dim=-1).indices
                mask = torch.ones_like(p_stack)
                mask.scatter_(-1, max_idx, 0.0)
                p = (p_stack * mask).sum(dim=-1) / 4.0
            elif method == "drop_3_furthest_prob":
                med = torch.median(p_stack, dim=-1, keepdim=True).values
                dev = (p_stack - med).abs()
                max_idx = torch.topk(dev, k=3, dim=-1).indices
                mask = torch.ones_like(p_stack)
                mask.scatter_(-1, max_idx, 0.0)
                p = (p_stack * mask).sum(dim=-1) / 3.0
            elif method == "drop_1_furthest_margin":
                med = torch.median(m_stack, dim=-1, keepdim=True).values
                dev = (m_stack - med).abs()
                max_idx = torch.topk(dev, k=1, dim=-1).indices
                mask = torch.ones_like(m_stack)
                mask.scatter_(-1, max_idx, 0.0)
                avg_m = (m_stack * mask).sum(dim=-1) / 5.0
                p = torch.sigmoid(avg_m)
            elif method == "drop_2_furthest_margin":
                med = torch.median(m_stack, dim=-1, keepdim=True).values
                dev = (m_stack - med).abs()
                max_idx = torch.topk(dev, k=2, dim=-1).indices
                mask = torch.ones_like(m_stack)
                mask.scatter_(-1, max_idx, 0.0)
                avg_m = (m_stack * mask).sum(dim=-1) / 4.0
                p = torch.sigmoid(avg_m)
            elif method == "drop_3_furthest_margin":
                med = torch.median(m_stack, dim=-1, keepdim=True).values
                dev = (m_stack - med).abs()
                max_idx = torch.topk(dev, k=3, dim=-1).indices
                mask = torch.ones_like(m_stack)
                mask.scatter_(-1, max_idx, 0.0)
                avg_m = (m_stack * mask).sum(dim=-1) / 3.0
                p = torch.sigmoid(avg_m)
            elif method == "median_prob":
                p = torch.median(p_stack, dim=-1).values
            
            fold_aucs.append(auroc(p, y))
        task_scores.append(np.mean(fold_aucs))
    return task_scores

print("="*90)
print(f"{'Aggregation Method':35s} | {'Primary 7':10s} | {'SEAL 10':10s} | {'Total 17 (All)':14s}")
print("="*90)

for m in methods:
    s_p7 = eval_method(pts_p7, m)
    s_s10 = eval_method(pts_s10, m)
    s_all = s_p7 + s_s10
    print(f"{m:35s} | {np.mean(s_p7):10.4f} | {np.mean(s_s10):10.4f} | {np.mean(s_all):14.4f}")
print("="*90)
