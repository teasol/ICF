import os
import torch
import numpy as np
from pathlib import Path
from torcheval.metrics.functional import binary_auroc as auroc_eval

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

print("Testing Concept: In-Context Meta-Optimization of Branch Weights...")
