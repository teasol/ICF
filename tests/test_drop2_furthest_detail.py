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

all_tasks = primary7 + seal10
all_pts = pts_p7 + pts_s10

print("="*80)
print(f"{'Task':32s} | {'Trimmed(1min,1max)':20s} | {'Drop2Furthest':15s}")
print("="*80)

trim_all, furth_all = [], []

for t, pt in zip(all_tasks, all_pts):
    short = t.replace("cptac_", "").replace("_mutation", "").replace("_regression", "")
    trim_aucs, furth_aucs = [], []
    for fold in pt["per_fold"]:
        y = fold["label"]
        m_list = [fold[b] for b in branches]
        if isinstance(m_list[0], list):
            m_list = [torch.tensor(m) for m in m_list]
        p_stack = torch.sigmoid(torch.stack(m_list, dim=-1))
        
        # Trimmed (1 min, 1 max)
        sum_p = torch.sum(p_stack, dim=-1)
        min_p = torch.min(p_stack, dim=-1).values
        max_p = torch.max(p_stack, dim=-1).values
        p_trim = (sum_p - min_p - max_p) / 4.0
        trim_aucs.append(auroc(p_trim, y))
        
        # Drop 2 furthest from median
        med = torch.median(p_stack, dim=-1, keepdim=True).values
        dev = (p_stack - med).abs()
        top2 = torch.topk(dev, k=2, dim=-1).indices
        mask = torch.ones_like(p_stack)
        mask.scatter_(-1, top2, 0.0)
        p_furth = (p_stack * mask).sum(dim=-1) / 4.0
        furth_aucs.append(auroc(p_furth, y))
        
    m_trim = np.mean(trim_aucs)
    m_furth = np.mean(furth_aucs)
    trim_all.append(m_trim)
    furth_all.append(m_furth)
    print(f"{short:32s} | {m_trim:20.4f} | {m_furth:15.4f}")

print("="*80)
print(f"{'Primary 7 Macro':32s} | {np.mean(trim_all[:7]):20.4f} | {np.mean(furth_all[:7]):15.4f}")
print(f"{'SEAL 10 Macro':32s} | {np.mean(trim_all[7:]):20.4f} | {np.mean(furth_all[7:]):15.4f}")
print(f"{'Total 17 Macro':32s} | {np.mean(trim_all):20.4f} | {np.mean(furth_all):15.4f}")
print("="*80)
