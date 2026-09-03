"""FULL model, trained head, projection swapped to per-episode PCA (docs SS136).

The point the CV-only run missed. SS136's first pass compared P against PCA using
only the CV ridge, on the worry that a rotated basis would break the trained
head. That worry was wrong: the head reads twelve RELATION features -- ridge
logits, differences and separations -- not raw descriptor coordinates, so it is
indifferent to which orthonormal basis produced them. The same trained head can
therefore be fed logits computed under either basis and compared directly.

So this runs the real model end to end, one checkpoint at a time, twice per fold:

    P    -- `_effective_covariance_projection()` as trained
    PCA  -- top-K eigenvectors of the pooled covariance of that fold's CONTEXT
            cells, recomputed per fold, no training, no test data touched

Everything else is identical: same head, same DD, same CT, same bags, same folds.
CT is basis-independent by construction (it selects on raw cells), so any
difference is attributable to the projection alone.

⚠️ Deviations from `eval_seal_tasks.sh`, applied IDENTICALLY to both arms: bags
are capped at `max_cells` once, deterministically, at load time rather than being
re-subsampled per query, and the whole task is held in memory. Absolute numbers
are therefore not directly comparable to SEAL macro; the P-vs-PCA gap is.
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib
import os
import sys
from collections import defaultdict

import h5py
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OFFICIAL = "/NHNHOME/BASE/kimds/Data/PathoBench/official"
FEATURES = "/NHNHOME/BASE/kimds/Data/PathoBench/features"
ALL_TASKS = [
    "bc_therapy/er_status", "bc_therapy/grade", "bc_therapy/her2_status",
    "cptac_brca/PIK3CA_mutation", "cptac_brca/TP53_mutation",
    "cptac_luad/EGFR_mutation", "cptac_luad/STK11_mutation", "cptac_luad/TP53_mutation",
    "cptac_ccrcc/BAP1_mutation", "cptac_ccrcc/VHL_mutation",
]


def auroc(labels, scores):
    labels = labels.long()
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ordered, index = scores[order], 0
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


def load_task(task, h5_index, device, max_cells, generator):
    label_column = task.split("/")[1]
    rows = list(csv.DictReader(open(f"{OFFICIAL}/{task}/k=all.tsv"), delimiter="\t"))
    bags, labels = {}, {}
    for row in rows:
        slide = row["slide_id"]
        if slide in bags or slide not in h5_index:
            continue
        with h5py.File(h5_index[slide], "r") as handle:
            features = torch.as_tensor(handle["features"][:], dtype=torch.float32)
        if features.shape[0] > max_cells:
            keep = torch.randperm(features.shape[0], generator=generator)[:max_cells]
            features = features.index_select(0, keep)
        bags[slide] = features.to(device)
        labels[slide] = int(row[label_column])
    folds = [c for c in rows[0] if c.startswith("fold_")]
    membership = {f: {"train": [], "test": []} for f in folds}
    for row in rows:
        if row["slide_id"] not in bags:
            continue
        for f in folds:
            if row[f] in membership[f]:
                membership[f][row[f]].append(row["slide_id"])
    return bags, labels, membership, folds


def pca_basis(context_bags, sketch_dim):
    """Top-`sketch_dim` eigenvectors of the pooled context-cell covariance.

    Streamed bag by bag: materialising every context cell at once is what SS62-3
    identified as an eval OOM driver (~12 GB for a full-tile episode).
    """
    dim = context_bags[0].shape[-1]
    device = context_bags[0].device
    total = 0
    summation = torch.zeros(dim, dtype=torch.float64, device=device)
    for bag in context_bags:
        summation += bag.double().sum(dim=0)
        total += bag.shape[0]
    mean = summation / total
    scatter = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    for bag in context_bags:
        centered = bag.double() - mean
        scatter += centered.T @ centered
    _, eigenvectors = torch.linalg.eigh(scatter / total)
    return eigenvectors[:, -sketch_dim:].flip(-1).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/archive/v94_v102_cell_value/train_v98_p1_reverse_1536_1gpu.yaml")
    parser.add_argument("--tasks", nargs="*", default=ALL_TASKS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    model_kwargs = dict(config["model"])
    # `max_cells` lives on the encoder, not the top-level model, so read the
    # config value rather than the module attribute.
    max_cells = int(model_kwargs.get("max_cells", 8192))
    module_path, class_name = model_kwargs.pop("model_src").rsplit(".", 1)
    device = torch.device(args.device)
    model = getattr(importlib.import_module(module_path), class_name)(**model_kwargs)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state.items()}, strict=True)
    model = model.to(device).eval()

    trained_projection = model._effective_covariance_projection
    trained_head = model.cv_dd_ct_head[0].weight.detach().clone()
    trained_bias = model.cv_dd_ct_head[0].bias.detach().clone()

    # Fixed head (docs SS137). The final logits are (-margin/2, +margin/2), so
    # swapping the two classes must flip the margin's sign. Under that swap
    # CV0<->CV1, D0<->D1 and q0<->q1 exchange while the three SEP_* features are
    # INVARIANT -- so label antisymmetry forces w(SEP) = 0 and bias = 0, and
    # forces each pair's weights to be equal and opposite. The difference
    # features (CV1-CV0 etc.) are linear combinations of their pair and add
    # nothing to a LINEAR head. What is left is one number per branch.
    #
    # Decomposing the eight trained heads that way gives, with std across seeds
    # of 0.027 / 0.008 / 0.012:
    #     margin = 1.442*(CV1-CV0) - 0.343*(D1-D0) + 0.286*(q1-q0)
    # i.e. CV : DD : CT = 1 : -0.238 : +0.199, identical to two decimals on every
    # seed. AUROC is scale-free so only the ratios matter.
    #
    # DD's negative sign is required, not a bug: `_dd_distance_features` returns
    # squared normalised DISTANCES to each class prototype, so a large D1 is
    # evidence AGAINST class 1, the opposite of the CV/CT logits.
    fixed = torch.zeros_like(trained_head)
    for slot, coefficient in ((0, -1.442), (1, 1.442), (4, 0.343), (5, -0.343),
                              (8, -0.286), (9, 0.286)):
        fixed[0, slot] = coefficient
    generator = torch.Generator().manual_seed(args.seed)
    h5_index = index_h5(FEATURES)
    results = defaultdict(lambda: defaultdict(list))

    for task in args.tasks:
        bags, labels, membership, folds = load_task(
            task, h5_index, device, max_cells, generator
        )
        for fold in folds:
            train, test = membership[fold]["train"], membership[fold]["test"]
            if not train or not test:
                continue
            y = torch.tensor([labels[s] for s in train] + [labels[s] for s in test],
                             dtype=torch.long, device=device)
            if len(set(y[:len(train)].tolist())) < 2:
                continue
            episode = [bags[s] for s in train] + [bags[s] for s in test]
            query_index = torch.arange(len(train), len(episode), device=device)
            basis = pca_basis([bags[s] for s in train], model.covariance_sketch_dim)
            for name in ("P+head", "PCA+head", "P+fixed", "PCA+fixed"):
                if name.startswith("PCA"):
                    model._effective_covariance_projection = lambda b=basis: b
                else:
                    model._effective_covariance_projection = trained_projection
                with torch.no_grad():
                    if name.endswith("fixed"):
                        model.cv_dd_ct_head[0].weight.copy_(fixed)
                        model.cv_dd_ct_head[0].bias.zero_()
                    else:
                        model.cv_dd_ct_head[0].weight.copy_(trained_head)
                        model.cv_dd_ct_head[0].bias.copy_(trained_bias)
                    logits = model(episode, y, query_index)
                score = (logits[:, 1] - logits[:, 0]).float().cpu()
                value = auroc(y[len(train):].cpu(), score)
                if value is not None:
                    results[name][task].append(value)
            model._effective_covariance_projection = trained_projection
            with torch.no_grad():
                model.cv_dd_ct_head[0].weight.copy_(trained_head)
                model.cv_dd_ct_head[0].bias.copy_(trained_bias)
        line = "  ".join(
            f"{n}={sum(results[n][task])/max(1,len(results[n][task])):.4f}"
            for n in ("P+head", "PCA+head", "P+fixed", "PCA+fixed")
        )
        print(f"{task:34s} {line}", flush=True)
        del bags
        torch.cuda.empty_cache()

    print(f"\n=== {os.path.basename(os.path.dirname(args.checkpoint))} — full model, macro over {len(args.tasks)} tasks ===")
    for name in ("P+head", "PCA+head", "P+fixed", "PCA+fixed"):
        per_task = [sum(v) / len(v) for v in results[name].values()]
        print(f"  {name:10s} {sum(per_task)/len(per_task):.4f}")


if __name__ == "__main__":
    main()
