"""Musk transfer diagnosis: bag cardinality, not feature semantics.

Motivation (2026-08-04). `docs/history/musk095_architecture_proposal.md` attributes
the Musk zero-shot ceiling (AUROC 0.822) to the input representation discarding the
bag mean, and proposes a raw-bag-mean channel (P2) plus a learned 166->512 read
bridge (P1). This script measures a competing explanation that the proposal never
tests: BagPFN is trained on bags of 500-1000 cells (`configs/data/medium.yaml`
`num_cells: [500, 1000]`, one shared count per episode), while Musk2 bags hold a
median of 12 conformers, 29/102 hold <= 4, and 2 hold exactly 1. Per-bag centering
in `StructuredEpisodePopulationAggregator._bag_view` leaves an n-1 dimensional
shadow of a 166-dimensional space, so small bags arrive at the model as noise.

Four independent reports, none of which requires training:

  cardinality  Bag-size distribution, and whether size alone leaks the label.
  stratified   AUROC of committed `predictions/musk_*.pt` split by query-bag size.
  decompose    Pairwise (Mann-Whitney) concordance split by size, which bounds how
               much AUROC a cardinality fix can recover.
  ceiling      LOO closed-form ridge on bag sufficient statistics under several
               instance normalizations, including a context-pooled z-score that
               `scripts/diagnose_normalization_ceiling.py` does not implement.
               Answers whether small-bag signal is absent or merely destroyed.

The ceiling probe is the Musk counterpart of `diagnose_normalization_ceiling.py`
(which is hard-wired to dense synthetic `[bags, cells, dim]` episodes and cannot
represent variable-length bags). It loads no checkpoint and trains nothing.

Usage:
    python scripts/diagnose_musk_cardinality.py --report all
    python scripts/diagnose_musk_cardinality.py --report ceiling --bootstrap 2000
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import auroc as auroc_metric  # noqa: E402
from src.utils.metrics import bootstrap_auroc_interval  # noqa: E402

DEFAULT_DATA = Path("/NHNHOME/kimds/Data/Musk/musk.pkl")
DEFAULT_PREDICTIONS = (
    "musk_v24_zero_shot.pt",
    "musk_v24_musklike_easy.pt",
    "musk_v24_musklike_easy_rawstats.pt",
    "musk_v24_musklike_easy_mil.pt",
)
# Instance normalizations. "center_l2" is what the model actually receives under
# the v24 default (`_bag_view` with bag_centered_l2_normalize=True); "raw" is what
# `test_musk.py --preprocess raw` supplies. The pool* variants standardize each
# feature by statistics pooled over every instance of every OTHER bag, which keeps
# between-bag mean differences while staying free of per-dataset units.
NORMALIZATIONS = (
    "raw",
    "l2",
    "center",
    "center_l2",
    "poolz",
    "poolz_l2",
    "poolz_shrink",
)


def load_musk(path: Path) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    ids = [str(r["bag_id"]) for r in records]
    bags = [np.asarray(r["X"], dtype=np.float64) for r in records]
    labels = np.asarray([int(r["y"]) for r in records], dtype=np.int64)
    return ids, bags, labels


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return auroc_metric(
        torch.as_tensor(scores, dtype=torch.float64), torch.as_tensor(labels)
    )


def ci(labels: np.ndarray, scores: np.ndarray, samples: int) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return bootstrap_auroc_interval(
        torch.as_tensor(scores, dtype=torch.float64),
        torch.as_tensor(labels),
        samples=samples,
    )


def cell(labels: np.ndarray, scores: np.ndarray, samples: int) -> str:
    if labels.size < 6 or len(np.unique(labels)) < 2:
        return f"{'n/a':>18s}"
    low, high = ci(labels, scores, samples)
    return f"{auroc(labels, scores):.3f} [{low:.2f},{high:.2f}]"


def report_cardinality(sizes: np.ndarray, labels: np.ndarray, samples: int) -> None:
    print("== bag cardinality ==")
    print(
        f"  bags={sizes.size}  positive={int(labels.sum())}  negative={int((labels == 0).sum())}"
    )
    percentiles = np.percentile(sizes, [10, 25, 50, 75, 90]).round(1)
    print(
        f"  instances/bag: min={sizes.min()} p10..p90={percentiles} "
        f"mean={sizes.mean():.1f} max={sizes.max()} total={sizes.sum()}"
    )
    for bound in (1, 2, 4, 10, 20, 50):
        mask = sizes <= bound
        print(
            f"    n <= {bound:3d}: {mask.sum():3d} bags ({mask.mean() * 100:4.0f}%), "
            f"{int(labels[mask].sum())} positive"
        )
    print(
        "  degenerate under _bag_view: n=1 -> centered_delta==0 -> F.normalize(0)==0 "
        f"(all-zero bag): {(sizes == 1).sum()} bags; "
        f"n=2 -> two antipodal unit vectors: {(sizes == 2).sum()} bags"
    )
    print("\n== is bag size alone predictive? (label-leak check) ==")
    for name, score in (("size", sizes.astype(np.float64)), ("-size", -sizes.astype(np.float64))):
        low, high = ci(labels, score, samples)
        print(f"  AUROC({name:5s}) = {auroc(labels, score):.4f}  [{low:.3f}, {high:.3f}]")


def report_stratified(
    ids: list[str], sizes: np.ndarray, prediction_dir: Path, names: tuple[str, ...], samples: int
) -> None:
    size_of = dict(zip(ids, sizes))
    print("== committed Musk predictions, AUROC stratified by query-bag cardinality ==")
    header = (
        f"  {'prediction':42s} {'ALL':>18s} {'n<=4':>18s} {'5..10':>18s} {'11..34':>18s} {'n>34':>18s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name in names:
        path = prediction_dir / name
        if not path.exists():
            print(f"  {name:42s} (missing)")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        bag_ids = payload["bag_id"]
        labels = payload["label"].numpy()
        scores = payload["probability"].numpy().astype(np.float64)
        n = np.asarray([size_of[b] for b in bag_ids])
        bands = (
            np.ones(n.size, dtype=bool),
            n <= 4,
            (n > 4) & (n <= 10),
            (n > 10) & (n <= 34),
            n > 34,
        )
        cells = " ".join(f"{cell(labels[m], scores[m], samples):>18s}" for m in bands)
        print(f"  {name:42s} {cells}")
        correlation = np.corrcoef(scores, np.log(n))[0, 1]
        print(f"  {'':42s} pearson(prob, log n) = {correlation:+.3f}")


def report_decompose(
    ids: list[str], sizes: np.ndarray, prediction_dir: Path, name: str, bound: int
) -> None:
    size_of = dict(zip(ids, sizes))
    payload = torch.load(prediction_dir / name, map_location="cpu", weights_only=False)
    labels = payload["label"].numpy()
    scores = payload["probability"].numpy().astype(np.float64)
    n = np.asarray([size_of[b] for b in payload["bag_id"]])
    small = n <= bound
    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    total = positive.size * negative.size
    print(f"== pairwise concordance decomposition ({name}, small = n <= {bound}) ==")
    print(f"  AUROC = P(score_pos > score_neg); {total} positive-negative pairs")

    def concordance(rows: np.ndarray, columns: np.ndarray) -> float:
        left = scores[rows][:, None]
        right = scores[columns][None, :]
        return float(((left > right).sum() + 0.5 * (left == right).sum()) / (rows.size * columns.size))

    blocks = {}
    for prow, rows in (("pos small", positive[small[positive]]), ("pos big", positive[~small[positive]])):
        for pcol, columns in (("neg small", negative[small[negative]]), ("neg big", negative[~small[negative]])):
            if rows.size == 0 or columns.size == 0:
                continue
            weight = rows.size * columns.size / total
            value = concordance(rows, columns)
            blocks[(prow, pcol)] = (value, weight)
            print(
                f"    {prow:9s} vs {pcol:9s}: {rows.size * columns.size:5d} pairs "
                f"({weight * 100:4.1f}%)  concordance={value:.3f}"
            )
    print(f"  weighted total = overall AUROC = {sum(v * w for v, w in blocks.values()):.4f}")
    clean = blocks.get(("pos big", "neg big"))
    if clean is not None:
        print(f"  concordance among big-vs-big pairs only: {clean[0]:.4f}")
        print("  projected overall AUROC if small-bag pairs reached:")
        for target in (0.75, 0.85, clean[0]):
            projected = sum(
                (value if key == ("pos big", "neg big") else target) * weight
                for key, (value, weight) in blocks.items()
            )
            print(f"    {target:.3f} -> {projected:.3f}")


def bag_statistics(bag: np.ndarray, mode: str, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    def unit(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-6)

    if mode == "poolz_shrink":
        # Cardinality-aware shrinkage toward the pooled prior, plus log n so a
        # downstream reader can calibrate on sample size. lambda -> 1 as n grows,
        # so this is a no-op in the large-bag regime the model trains on.
        standardized = (bag - mean) / std
        weight = bag.shape[0] / (bag.shape[0] + 8.0)
        return np.concatenate(
            [
                weight * standardized.mean(axis=0),
                weight * standardized.var(axis=0) + (1.0 - weight),
                [np.log(bag.shape[0])],
            ]
        )
    if mode == "raw":
        view = bag
    elif mode == "l2":
        view = unit(bag)
    elif mode == "center":
        view = bag - bag.mean(axis=0, keepdims=True)
    elif mode == "center_l2":
        view = unit(bag - bag.mean(axis=0, keepdims=True))
    elif mode == "poolz":
        view = (bag - mean) / std
    elif mode == "poolz_l2":
        view = unit((bag - mean) / std)
    else:
        raise ValueError(f"Unsupported normalization: {mode}")
    return np.concatenate([view.mean(axis=0), view.var(axis=0)])


def ridge_loo(
    bags: list[np.ndarray],
    labels: np.ndarray,
    mode: str,
    penalty: float,
    design_norm: str = "feature",
) -> np.ndarray:
    """Leave-one-bag-out ridge. Pool statistics come from training bags only.

    `design_norm` controls how the bag-descriptor design matrix is scaled before
    the ridge solve, and it is NOT a free choice -- it decides whether a diagonal
    input rescaling is even observable:

      "feature"  per-column standardization. Absorbs any per-feature affine map,
                 so `poolz` becomes mathematically identical to `raw`.
      "scalar"   context centering + one scalar RMS, preserving inter-feature
                 geometry. This is what `diagnose_normalization_ceiling.py` does
                 (see its `ridge_logits`), so use it for apples-to-apples
                 comparison against the synthetic ceiling table.
    """
    count = len(bags)
    scores = np.zeros(count)
    targets = labels * 2.0 - 1.0
    needs_pool = mode.startswith("poolz")
    if not needs_pool:
        zero = np.zeros(bags[0].shape[1])
        one = np.ones(bags[0].shape[1])
        features = np.stack([bag_statistics(b, mode, zero, one) for b in bags])
    for index in range(count):
        train = np.arange(count) != index
        if needs_pool:
            pool = np.concatenate([bags[j] for j in range(count) if j != index], axis=0)
            mean = pool.mean(axis=0)
            std = np.maximum(pool.std(axis=0), 1e-6)
            matrix = np.stack([bag_statistics(bags[j], mode, mean, std) for j in range(count)])
        else:
            matrix = features
        location = matrix[train].mean(axis=0)
        if design_norm == "feature":
            scale = np.maximum(matrix[train].std(axis=0), 1e-8)
        elif design_norm == "scalar":
            centered = matrix[train] - location
            scale = max(float(np.sqrt((centered**2).mean())), 1e-8)
        else:
            raise ValueError(f"Unsupported design_norm: {design_norm}")
        standardized = (matrix - location) / scale
        design = standardized[train]
        gram = design.T @ design + penalty * np.eye(design.shape[1])
        weights = np.linalg.solve(gram, design.T @ targets[train])
        scores[index] = standardized[index] @ weights
    return scores


def report_ceiling(
    bags: list[np.ndarray],
    labels: np.ndarray,
    sizes: np.ndarray,
    penalties: tuple[float, ...],
    samples: int,
    bound: int,
    design_norm: str = "feature",
) -> None:
    print("== LOO ridge ceiling on bag [mean, var], by instance normalization ==")
    print("  (linear probe: zero-padding 166->512 is irrelevant here by construction)")
    print(f"  design_norm={design_norm}" + ("  -- poolz is absorbed and equals raw" if design_norm == "feature" else "  -- matches diagnose_normalization_ceiling.py"))
    print(f"  {'normalization':14s}" + "".join(f"{'lam=' + str(p):>11s}" for p in penalties))
    cached: dict[str, dict[float, np.ndarray]] = {}
    for mode in NORMALIZATIONS:
        cached[mode] = {p: ridge_loo(bags, labels, mode, p, design_norm) for p in penalties}
        row = "".join(f"{auroc(labels, cached[mode][p]):11.4f}" for p in penalties)
        print(f"  {mode:14s}{row}")
    print(f"\n  best penalty per normalization, stratified by query-bag cardinality:")
    print(
        f"  {'normalization':14s} {'ALL':>18s} {'n<=' + str(bound):>18s} {'n>' + str(bound):>18s}  lam"
    )
    for mode in NORMALIZATIONS:
        best = max(penalties, key=lambda p: auroc(labels, cached[mode][p]))
        scores = cached[mode][best]
        small = sizes <= bound
        print(
            f"  {mode:14s} {cell(labels, scores, samples):>18s} "
            f"{cell(labels[small], scores[small], samples):>18s} "
            f"{cell(labels[~small], scores[~small], samples):>18s}  {best:g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", type=Path, default=Path("predictions"))
    parser.add_argument(
        "--report",
        choices=("all", "cardinality", "stratified", "decompose", "ceiling"),
        default="all",
    )
    parser.add_argument("--reference", default="musk_v24_musklike_easy.pt")
    parser.add_argument("--small-bound", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--penalties", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument(
        "--design-norm",
        choices=("feature", "scalar"),
        default="feature",
        help="Design-matrix scaling for the ceiling probe; 'scalar' matches "
        "diagnose_normalization_ceiling.py and makes poolz observable.",
    )
    args = parser.parse_args()

    ids, bags, labels = load_musk(args.data)
    sizes = np.asarray([b.shape[0] for b in bags])

    if args.report in ("all", "cardinality"):
        report_cardinality(sizes, labels, args.bootstrap)
        print()
    if args.report in ("all", "stratified"):
        report_stratified(ids, sizes, args.predictions, DEFAULT_PREDICTIONS, args.bootstrap)
        print()
    if args.report in ("all", "decompose"):
        report_decompose(ids, sizes, args.predictions, args.reference, args.small_bound)
        print()
    if args.report in ("all", "ceiling"):
        report_ceiling(
            bags,
            labels,
            sizes,
            tuple(args.penalties),
            args.bootstrap,
            args.small_bound,
            args.design_norm,
        )


if __name__ == "__main__":
    main()
