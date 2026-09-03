import glob
import torch
import numpy as np

def compute_auroc(score, target):
    indices = torch.argsort(score, descending=True)
    t = target[indices].float()
    n_pos = (t == 1).sum()
    n_neg = (t == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = (t == 1).float().cumsum(0)
    fps = (t == 0).float().cumsum(0)
    tpr = torch.cat([torch.tensor([0.0]), tps / n_pos])
    fpr = torch.cat([torch.tensor([0.0]), fps / n_neg])
    return float(torch.trapz(tpr, fpr).item())

primary7 = [
    "cptac_lscc_ARID1A_mutation",
    "cptac_lscc_Histologic_Grade",
    "cptac_lscc_KEAP1_mutation",
    "cptac_luad_KRAS_mutation",
    "cptac_pda_SMAD4_mutation",
    "ucla_lung_progression_regression",
    "cptac_ccrcc_PBRM1_mutation",
]

tag = "v121_salience_anchor_s5_f07_a15"
branches = ["m_cv", "m_bm", "m_bd", "m_qa", "m_ds"]

# Load all 50 folds for all 7 tasks
data = {}
for t in primary7:
    fname = f"predictions/pathobench_{t}_{tag}_official50_bf16.pt"
    d = torch.load(fname, map_location="cpu", weights_only=False)
    data[t] = d["per_fold"]

def eval_method(agg_fn):
    task_aurocs = {}
    for t in primary7:
        folds = data[t]
        f_scores = []
        for fold in folds:
            lbl = fold["label"]
            # stack probabilities [B, N]
            probs = torch.stack([torch.sigmoid(fold[b]) for b in branches], dim=0)
            ens_p = agg_fn(probs)
            f_scores.append(compute_auroc(ens_p, lbl))
        task_aurocs[t] = float(np.mean(f_scores))
    return task_aurocs

methods = {}

# 1. Baseline Trimmed Mean (drops 1 min, 1 max)
def trimmed_mean(probs):
    sorted_p, _ = torch.sort(probs, dim=0)
    return sorted_p[1:-1].mean(dim=0)
methods["Trimmed Mean"] = trimmed_mean

# 2. Simple Soft Voting (Mean of all 5)
def soft_voting(probs):
    return probs.mean(dim=0)
methods["Soft Voting (Mean)"] = soft_voting

# 3. Certainty Weighted (gamma = 1.0)
def cert_weight_g1(probs):
    w = (probs - 0.5).abs().clamp_min(1e-4)
    return (w * probs).sum(dim=0) / w.sum(dim=0)
methods["Certainty (g=1.0)"] = cert_weight_g1

# 4. Certainty Weighted (gamma = 2.0)
def cert_weight_g2(probs):
    w = ((probs - 0.5).abs() ** 2).clamp_min(1e-4)
    return (w * probs).sum(dim=0) / w.sum(dim=0)
methods["Certainty (g=2.0)"] = cert_weight_g2

# 5. Hard Gated Voting (tau = 0.05)
def hard_gated_t05(probs):
    c = (probs - 0.5).abs()
    mask = (c >= 0.05)
    weights = mask.float()
    has_active = (weights.sum(dim=0) > 0)
    default_w = torch.ones_like(weights)
    final_w = torch.where(has_active.unsqueeze(0), weights, default_w)
    return (final_w * probs).sum(dim=0) / final_w.sum(dim=0)
methods["Hard Gated (t=0.05)"] = hard_gated_t05

# 6. Hard Gated Voting (tau = 0.10)
def hard_gated_t10(probs):
    c = (probs - 0.5).abs()
    mask = (c >= 0.10)
    weights = mask.float()
    has_active = (weights.sum(dim=0) > 0)
    default_w = torch.ones_like(weights)
    final_w = torch.where(has_active.unsqueeze(0), weights, default_w)
    return (final_w * probs).sum(dim=0) / final_w.sum(dim=0)
methods["Hard Gated (t=0.10)"] = hard_gated_t10

# 7. Entropy-weighted Voting
def entropy_weighted(probs):
    eps = 1e-6
    p_clamped = probs.clamp(eps, 1.0 - eps)
    ent = -(p_clamped * torch.log2(p_clamped) + (1.0 - p_clamped) * torch.log2(1.0 - p_clamped))
    w = (1.0 - ent).clamp_min(1e-4)
    return (w * probs).sum(dim=0) / w.sum(dim=0)
methods["Entropy Weighted"] = entropy_weighted

print("Computing 50-fold official results for all voting methods...")
results = {m: eval_method(fn) for m, fn in methods.items()}

col_names = [m.ljust(18) for m in methods]
print("\n" + "=" * 165)
header = "| " + "Task".ljust(26) + " | " + " | ".join(col_names) + " |"
print(header)
print("|" + "-" * 28 + "|" + "|".join(["-" * 20 for _ in methods]) + "|")

for t in primary7:
    task_short = t.replace("cptac_", "").replace("ucla_lung_", "")
    row = "| " + task_short.ljust(26) + " | "
    vals = [f"{results[m][t]:.4f}".ljust(18) for m in methods]
    row += " | ".join(vals) + " |"
    print(row)

print("|" + "=" * 28 + "|" + "|".join(["=" * 20 for _ in methods]) + "|")
row = "| " + "MACRO MEAN (Primary 7)".ljust(26) + " | "
vals = [f"{np.mean([results[m][t] for t in primary7]):.4f}".ljust(18) for m in methods]
row += " | ".join(vals) + " |"
print(row)
print("=" * 165 + "\n")
