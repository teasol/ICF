"""CV branch only: learned P vs per-episode PCA, no training (docs SS136).

What this isolates. The CV branch alone is already a classifier: bag descriptors
go into a closed-form ridge solved per episode, and `CovarianceMeanRidgeModel.
_ridge_logits` returns a 2-class logit without touching the relation head, DD or
CT. So the projection can be judged on the thing it actually feeds, with nothing
else diluting it.

Why PCA is the right control. P is a learned 1536->K orthonormal basis, and v104
showed learning it is worth +0.0126 (SS133). PCA of the episode's own context
cells is the obvious training-free alternative: also orthonormal, also
K-dimensional, but data-derived per episode instead of learned once. If PCA
matches P, then 196,608 trained parameters are buying something a per-episode
eigendecomposition gives for free.

The trick that makes this cheap. For any basis B,

    projected = centered @ B
    cov       = projected^T projected / n = B^T (centered^T centered / n) B = B^T C B

so each slide's 1536x1536 centered covariance C and mean mu can be computed ONCE
and then reused for every basis and every fold. Tiles are read once per task
instead of once per (fold x basis x seed).

The pooled context covariance that PCA diagonalises is assembled exactly from
those per-slide pieces, including the between-slide term:

    C_pool = sum_i n_i (C_i + (mu_i - mu)(mu_i - mu)^T) / sum_i n_i

⚠️ Deviations from the official eval path, applied IDENTICALLY to both arms so
the comparison stays fair: all tiles are used (the model's `max_cells` random
subsample is skipped, which also makes this deterministic), and the head/DD/CT
are absent by construction. Absolute numbers here are therefore NOT comparable to
SEAL macro from `eval_seal_tasks.sh`; only the P-vs-PCA difference is.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import h5py
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OFFICIAL = "/NHNHOME/BASE/kimds/Data/PathoBench/official"
FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
TASKS = [
    "bc_therapy/er_status", "bc_therapy/grade", "bc_therapy/her2_status",
    "cptac_brca/PIK3CA_mutation", "cptac_brca/TP53_mutation",
    "cptac_luad/EGFR_mutation", "cptac_luad/STK11_mutation", "cptac_luad/TP53_mutation",
    "cptac_ccrcc/BAP1_mutation", "cptac_ccrcc/VHL_mutation",
]


def auroc(labels: torch.Tensor, scores: torch.Tensor):
    """Mann-Whitney U with tie handling. None when a fold is single-class."""
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
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def index_h5(root):
    mapping = {}
    for directory in sorted(os.listdir(root)):
        path = os.path.join(root, directory)
        if os.path.isdir(path):
            for h5_path in glob.glob(os.path.join(path, "*.h5")):
                mapping.setdefault(os.path.basename(h5_path)[:-3], h5_path)
    return mapping


def slide_statistics(task, h5_index, device):
    """Per-slide (count, mean, centered covariance) plus labels and folds."""
    import csv

    label_column = task.split("/")[1]
    rows = list(csv.DictReader(open(f"{OFFICIAL}/{task}/k=all.tsv"), delimiter="\t"))
    stats, labels = {}, {}
    for row in rows:
        slide = row["slide_id"]
        if slide in stats or slide not in h5_index:
            continue
        with h5py.File(h5_index[slide], "r") as handle:
            features = torch.as_tensor(handle["features"][:], dtype=torch.float32).to(device)
        mean = features.mean(dim=0)
        centered = features - mean
        stats[slide] = (features.shape[0], mean, (centered.T @ centered) / features.shape[0])
        labels[slide] = int(row[label_column])
        del features, centered
    folds = [c for c in rows[0] if c.startswith("fold_")]
    membership = {f: {"train": [], "test": []} for f in folds}
    for row in rows:
        if row["slide_id"] not in stats:
            continue
        for f in folds:
            if row[f] in membership[f]:
                membership[f][row[f]].append(row["slide_id"])
    return stats, labels, membership, folds


def pooled_covariance(slides, stats):
    """Exact pooled covariance over every cell of every listed slide."""
    total = sum(stats[s][0] for s in slides)
    dim = stats[slides[0]][1].numel()
    device = stats[slides[0]][1].device
    grand = torch.zeros(dim, device=device, dtype=torch.float64)
    for s in slides:
        count, mean, _ = stats[s]
        grand += count * mean.double()
    grand /= total
    pooled = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    for s in slides:
        count, mean, covariance = stats[s]
        shift = (mean.double() - grand)
        pooled += count * (covariance.double() + torch.outer(shift, shift))
    return pooled / total


def descriptors(slides, stats, basis, triangle):
    """[covariance triangle, bag mean] for each slide under `basis`."""
    rows = []
    for s in slides:
        _, mean, covariance = stats[s]
        sketch = basis.T @ covariance @ basis
        rows.append(torch.cat((sketch[triangle[0], triangle[1]], mean)))
    return torch.stack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--config", default="configs/archive/v94_v102_cell_value/train_v98_p1_reverse_1536_1gpu.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import importlib
    from src.models.set_transformer_ridge import CovarianceMeanRidgeModel

    config = yaml.safe_load(open(args.config))
    model_kwargs = dict(config["model"])
    module_path, class_name = model_kwargs.pop("model_src").rsplit(".", 1)
    device = torch.device(args.device)
    model = getattr(importlib.import_module(module_path), class_name)(**model_kwargs).to(device)
    model.eval()
    triangle = model._covariance_triangle
    sketch_dim = model.covariance_sketch_dim

    learned = []
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        raw = state["model._covariance_projection"].float().to(device)
        learned.append((os.path.basename(os.path.dirname(path)), torch.linalg.qr(raw, mode="reduced").Q))

    h5_index = index_h5(FEATURES)
    results = defaultdict(lambda: defaultdict(list))
    for task in TASKS:
        stats, labels, membership, folds = slide_statistics(task, h5_index, device)
        for fold in folds:
            train, test = membership[fold]["train"], membership[fold]["test"]
            if not train or not test:
                continue
            y_context = torch.tensor([labels[s] for s in train], device=device)
            y_query = torch.tensor([labels[s] for s in test])
            if len(set(y_context.tolist())) < 2:
                continue
            bases = {}
            eigenvalues, eigenvectors = torch.linalg.eigh(pooled_covariance(train, stats))
            bases["PCA"] = eigenvectors[:, -sketch_dim:].flip(-1).float()
            for name, basis in learned:
                bases[name] = basis
            for name, basis in bases.items():
                context = descriptors(train, stats, basis, triangle)
                query = descriptors(test, stats, basis, triangle)
                with torch.no_grad():
                    logits = CovarianceMeanRidgeModel._ridge_logits(
                        model, context, y_context, query
                    )
                score = (logits[:, 1] - logits[:, 0]).float().cpu()
                value = auroc(y_query, score)
                if value is not None:
                    results[name][task].append(value)
        done = {n: sum(results[n][task]) / max(1, len(results[n][task])) for n in results}
        print(f"{task:34s} " + "  ".join(f"{n}={v:.4f}" for n, v in done.items()), flush=True)
        del stats
        torch.cuda.empty_cache()

    print("\n=== CV-branch-only fold-mean AUROC, macro over 10 tasks ===")
    for name in results:
        per_task = [sum(v) / len(v) for v in results[name].values()]
        print(f"  {name:34s} {sum(per_task)/len(per_task):.4f}")


if __name__ == "__main__":
    main()
