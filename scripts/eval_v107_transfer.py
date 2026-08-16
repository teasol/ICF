"""v107 on the two non-PathoBench benchmarks: ICI and UCI Musk (docs SS144).

Both are transfer tests — nothing here was trained on either cohort, and with
v107 nothing was trained at all. That removes the two bridges the old scripts
needed and that always muddied these numbers:

  * `scripts/test_musk.py` had to pad Musk's 166-d descriptors up to the trained
    model's 1536-d input (zero-pad or tile), a "crude OOD bridge" in its own
    words. `TrainingFreeClassifier` has no input dim — the basis is eigenvectors
    of whatever arrives — so Musk runs at its native 166.
  * ICI went through `launch_ici_protocol.sh`, which FINE-TUNES per fold
    (5 seeds x 5 folds = 25 trainings). Here the same 25 splits are scored with
    zero fitting, so the fold spread is the data's, not the optimiser's.

⚠️ K is capped at the input dim. v107's K=256 is defined at 1536-d UNI2; Musk
has 166 dims, so K=166 is the ceiling and the covariance sketch degenerates to
the full covariance. Reported explicitly rather than silently clipped.

⚠️ Musk bags are TINY — median 12 instances, minimum 1. A bag of n cells has a
centred covariance of rank <= n-1, so most Musk descriptors are severely
rank-deficient and a 1-cell bag contributes an exactly zero covariance block.
The CV branch is built for ~4,000-cell slides; treat Musk as a stress test of
the failure mode, not as a benchmark v107 was designed for.

Usage:
    python scripts/eval_v107_transfer.py musk [--sketch-dims 8 16 ...]
    python scripts/eval_v107_transfer.py ici  [--sketch-dims 64 128 256]
"""

from __future__ import annotations

import argparse
import os
import pickle
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig  # noqa: E402

MUSK_PKL = "/NHNHOME/BASE/kimds/Data/Musk/musk.pkl"
ICI_ROOT = "data/ICI_CVOnly_scConcept_512"


def auroc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    """Rank AUROC with ties averaged (same definition as the PathoBench path)."""
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


def score(context_bags, context_labels, query_bags, sketch_dim, device):
    dim = context_bags[0].shape[-1]
    effective = min(sketch_dim, dim)
    model = TrainingFreeClassifier(TrainingFreeConfig(sketch_dim=effective))
    return model.margins(
        [b.to(device) for b in context_bags],
        context_labels.to(device),
        [b.to(device) for b in query_bags],
    ).cpu(), effective


# --------------------------------------------------------------------------- musk
def load_musk():
    records = pickle.load(open(MUSK_PKL, "rb"))
    bags = [torch.as_tensor(r["X"], dtype=torch.float32) for r in records]
    labels = torch.tensor([int(r["y"]) for r in records], dtype=torch.long)
    return bags, labels


def run_musk(args, device):
    bags, labels = load_musk()
    sizes = [b.shape[0] for b in bags]
    print(f"Musk: {len(bags)} bags, {bags[0].shape[1]}-d, "
          f"{min(sizes)}..{max(sizes)} instances (median {int(statistics.median(sizes))}), "
          f"{int(labels.sum())} positive\n")
    print("Leave-one-out: each bag is queried against the other 101 as labelled context.\n")
    print(f"{'K asked':>9} {'K used':>8} {'AUROC':>9} {'95% CI (bootstrap)':>22}")
    for sketch_dim in args.sketch_dims:
        scores, effective = torch.empty(len(bags)), None
        for held_out in range(len(bags)):
            keep = [i for i in range(len(bags)) if i != held_out]
            margin, effective = score(
                [bags[i] for i in keep], labels[keep], [bags[held_out]],
                sketch_dim, device,
            )
            scores[held_out] = margin[0]
        # n=102 is small enough that a point estimate invites over-reading; the
        # trained baseline in `test_musk.py` reports a CI, so this must too.
        generator = torch.Generator().manual_seed(0)
        boot = []
        for _ in range(2000):
            pick = torch.randint(len(bags), (len(bags),), generator=generator)
            value = auroc(labels[pick], scores[pick])
            if value is not None:
                boot.append(value)
        boot = torch.tensor(sorted(boot))
        low, high = boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]
        print(f"{sketch_dim:>9} {effective:>8} {auroc(labels, scores):>9.4f}"
              f"{f'[{low:.3f}, {high:.3f}]':>22}")


# --------------------------------------------------------------------------- ici
def load_ici_fold(seed, fold, device):
    import pandas as pd

    base = f"{ICI_ROOT}/{seed}/CV{fold}"
    out = []
    for split in ("train", "val"):
        cells = torch.load(f"{base}/{split}_hvg.pt", map_location="cpu", weights_only=False)
        info = pd.read_csv(f"{base}/{split}_donor_info.csv")
        donors, bags, labels = list(dict.fromkeys(info["donor_id"])), [], []
        donor_index = {d: i for i, d in enumerate(donors)}
        which = torch.tensor([donor_index[d] for d in info["donor_id"]])
        response = info.groupby("donor_id")["Response"].first()
        for i, donor in enumerate(donors):
            bags.append(cells[which == i].to(device))
            labels.append(1 if response[donor] == "R" else 0)
        out.append((bags, torch.tensor(labels, dtype=torch.long)))
    return out[0], out[1]


def run_ici(args, device):
    seeds = args.seeds or sorted(os.listdir(ICI_ROOT))
    print(f"ICI: {len(seeds)} seed partitions x 5 folds, donors are bags, "
          f"label = Response (R vs NR).")
    print("Context = that fold's train donors, query = its val donors. No fitting.\n")
    results = {k: {} for k in args.sketch_dims}
    for sketch_dim in args.sketch_dims:
        print(f"--- K = {sketch_dim}")
        for seed in seeds:
            values = []
            for fold in range(5):
                (train_bags, train_y), (val_bags, val_y) = load_ici_fold(seed, fold, device)
                margins, effective = score(train_bags, train_y, val_bags, sketch_dim, device)
                value = auroc(val_y, margins)
                if value is not None:
                    values.append(value)
                del train_bags, val_bags
                torch.cuda.empty_cache()
            results[sketch_dim][seed] = values
            print(f"  {seed:12s} " + " ".join(f"{v:.4f}" for v in values)
                  + f"   mean {statistics.mean(values):.4f}")
        allv = [v for vs in results[sketch_dim].values() for v in vs]
        per_seed = [statistics.mean(vs) for vs in results[sketch_dim].values()]
        print(f"  {'POOLED':12s} fold-mean {statistics.mean(allv):.4f} "
              f"(fold std {statistics.stdev(allv):.4f}, n={len(allv)})   "
              f"seed-mean std {statistics.stdev(per_seed):.4f}\n")
    if len(args.sketch_dims) > 1:
        print(f"{'K':>6} {'fold-mean':>11} {'seed std':>10}")
        for k in args.sketch_dims:
            allv = [v for vs in results[k].values() for v in vs]
            per_seed = [statistics.mean(vs) for vs in results[k].values()]
            print(f"{k:>6} {statistics.mean(allv):>11.4f} {statistics.stdev(per_seed):>10.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("musk", "ici"))
    parser.add_argument("--sketch-dims", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.sketch_dims is None:
        args.sketch_dims = [8, 16, 32, 64, 128, 166] if args.dataset == "musk" else [64, 128, 256]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    (run_musk if args.dataset == "musk" else run_ici)(args, device)


if __name__ == "__main__":
    main()
