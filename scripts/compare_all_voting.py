import torch, pathlib
from scripts.analyze_voting import compute_auroc, PRIMARY_7_TASKS, find_pt_file

tag = 'qa_w1_primary7'
dir_path = pathlib.Path('predictions')

data_by_task = {}
for t in PRIMARY_7_TASKS:
    p = find_pt_file(dir_path, t, tag)
    data = torch.load(p, map_location='cpu', weights_only=False)
    data_by_task[t] = data.get('per_fold', data.get('results', {}))

def rank_norm(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(x)
    ranks[order] = torch.arange(len(x), dtype=torch.float32)
    return ranks / max(1.0, float(len(x) - 1))

def zscore_norm(x):
    std = x.std(unbiased=False).clamp_min(1e-6)
    return (x - x.mean()) / std

methods = {
    '1. Soft Voting (Prob Mean)': lambda b_list: sum(torch.sigmoid(b) for b in b_list) / len(b_list),
    '2. Linear Logit Sum': lambda b_list: sum(b_list),
    '3. Median Logit Voting': lambda b_list: torch.median(torch.stack(b_list, dim=-1), dim=-1).values,
    '4. Median Prob Voting': lambda b_list: torch.median(torch.stack([torch.sigmoid(b) for b in b_list], dim=-1), dim=-1).values,
    '5. Trimmed Mean Prob (drop min/max)': lambda b_list: (torch.sum(torch.stack([torch.sigmoid(b) for b in b_list], dim=-1), dim=-1) - torch.min(torch.stack([torch.sigmoid(b) for b in b_list], dim=-1), dim=-1).values - torch.max(torch.stack([torch.sigmoid(b) for b in b_list], dim=-1), dim=-1).values) / max(1, len(b_list) - 2),
    '6. Percentile Rank Mean': lambda b_list: sum(rank_norm(b) for b in b_list) / len(b_list),
    '7. Z-Score Logit Sum': lambda b_list: sum(zscore_norm(b) for b in b_list),
    '8. Geometric Mean Prob': lambda b_list: torch.prod(torch.stack([torch.sigmoid(b).clamp_min(1e-7) for b in b_list], dim=-1), dim=-1) ** (1.0 / len(b_list)),
    '9. Harmonic Mean Prob': lambda b_list: len(b_list) / sum(1.0 / torch.sigmoid(b).clamp_min(1e-7) for b in b_list),
}

configs = {
    '4-Branch Base (CV, CT, BM, BD)': ['cv', 'ct', 'bm', 'bd'],
    '5-Branch with QA (CV, CT, BM, BD, QA)': ['cv', 'ct', 'bm', 'bd', 'qa'],
    '4-Branch No-CV (CT, BM, BD, QA)': ['ct', 'bm', 'bd', 'qa'],
}

for cfg_name, branches in configs.items():
    print(f'=== {cfg_name} ===')
    for m_name, fn in methods.items():
        task_scores = []
        for t in PRIMARY_7_TASKS:
            folds = data_by_task[t]
            f_scores = []
            for f_k, f_data in (folds.items() if isinstance(folds, dict) else enumerate(folds)):
                if f_data is None: continue
                label = f_data['label'].long()
                branch_tensors = [f_data[f'm_{b}'].float() for b in branches]
                p = fn(branch_tensors)
                f_scores.append(compute_auroc(p, label))
            task_scores.append(sum(f_scores)/len(f_scores))
        macro = sum(task_scores)/len(task_scores)
        print(f'{m_name:38s} : Macro AUROC = {macro:.5f}')
    print()
