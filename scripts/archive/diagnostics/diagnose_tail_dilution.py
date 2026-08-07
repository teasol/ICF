"""Diagnose Tail Token Evidence Dilution on Musk Large Bags (n > 34).

Measures whether fractional tail pooling (ceil(fraction * n)) dilutes single-conformer
active signals on large bags (n > 34), comparing Top-1 absolute tail instance vs.
fractional (1%, 5%, 15%) tail pooling across cardinality bands.

Usage:
    python scripts/diagnose_tail_dilution.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.metrics import auroc as auroc_metric

DEFAULT_DATA = Path("/NHNHOME/BASE/kimds/Data/Musk/musk.pkl")


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
    return float(
        auroc_metric(
            torch.as_tensor(scores, dtype=torch.float64),
            torch.as_tensor(labels, dtype=torch.int64),
        )
    )


def extract_tail_features(
    bags: list[np.ndarray], mode: str = "top_1"
) -> np.ndarray:
    """Extract bag representations using different tail pooling strategies under poolz_l2."""
    num_bags = len(bags)
    num_features = bags[0].shape[1]
    features = np.zeros((num_bags, num_features), dtype=np.float64)

    # Global context pool statistics across all bags (102 bags, 101 LOO is practically identical)
    all_instances = np.concatenate(bags, axis=0)
    pool_mean = all_instances.mean(axis=0, keepdims=True)
    pool_std = np.clip(all_instances.std(axis=0, keepdims=True), 1e-6, None)

    ctx_center = (all_instances - pool_mean) / pool_std
    ctx_center_norm = ctx_center / np.clip(
        np.linalg.norm(ctx_center, axis=1, keepdims=True), 1e-6, None
    )
    global_center = ctx_center_norm.mean(axis=0, keepdims=True)
    global_center = global_center / np.clip(
        np.linalg.norm(global_center, axis=1, keepdims=True), 1e-6, None
    )

    for i in range(num_bags):
        # Standardize query bag and L2 normalize
        z = (bags[i] - pool_mean) / pool_std
        norms = np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-6, None)
        z_norm = z / norms

        # Novelty = 1 - cosine similarity to global context center
        similarity = (z_norm * global_center).sum(axis=1)
        novelty = 1.0 - similarity
        n_cells = len(z_norm)


        if mode == "top_1":
            idx = np.argmax(novelty)
            features[i] = z_norm[idx]
        elif mode == "top_2_avg":
            k = min(n_cells, 2)
            indices = np.argsort(novelty)[-k:]
            features[i] = z_norm[indices].mean(axis=0)
        elif mode == "top_3_avg":
            k = min(n_cells, 3)
            indices = np.argsort(novelty)[-k:]
            features[i] = z_norm[indices].mean(axis=0)
        elif mode.startswith("frac_"):
            frac = float(mode.split("_")[1])
            k = min(n_cells, max(1, int(np.ceil(frac * n_cells))))
            indices = np.argsort(novelty)[-k:]
            # Softmax / LSE weighting over top-k
            scores = novelty[indices]
            weights = np.exp(scores * 2.0)
            weights = weights / weights.sum()
            features[i] = (z_norm[indices] * weights[:, None]).sum(axis=0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return features


def loo_ridge_auroc(
    X: np.ndarray, y: np.ndarray, alpha: float = 100.0
) -> np.ndarray:
    """Fast closed-form LOO Ridge classification probabilities."""
    num_samples, num_features = X.shape
    # Standardize X
    mean = X.mean(axis=0, keepdims=True)
    std = np.clip(X.std(axis=0, keepdims=True), 1e-6, None)
    X_std = (X - mean) / std

    y_centered = y.astype(np.float64) - y.mean()

    # Closed-form hat matrix computation
    cov = X_std.T @ X_std + alpha * np.eye(num_features)
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ (X_std.T @ y_centered)

    # Predictions and leverage values h_ii
    preds = X_std @ w
    H_diag = np.sum((X_std @ inv_cov) * X_std, axis=1)

    # Exact LOO predictions
    loo_preds = (preds - H_diag * y_centered) / np.clip(1.0 - H_diag, 1e-6, None)
    scores = loo_preds + y.mean()
    probs = 1.0 / (1.0 + np.exp(-scores))
    return probs


def main() -> None:
    ids, bags, labels = load_musk(DEFAULT_DATA)
    sizes = np.array([len(b) for b in bags])

    bands = {
        "ALL": np.ones(len(sizes), dtype=bool),
        "n <= 4": sizes <= 4,
        "5..10": (sizes > 4) & (sizes <= 10),
        "11..34": (sizes > 10) & (sizes <= 34),
        "n > 34": sizes > 34,
    }

    modes = ["top_1", "top_2_avg", "top_3_avg", "frac_0.01", "frac_0.05", "frac_0.15"]

    print("=" * 80)
    print("  Musk Evidence Dilution Probe (Top-1 vs Fractional Tail Pooling)")
    print("=" * 80)
    header = f"  {'Pooling Mode':20s}" + "".join(f"{b:>14s}" for b in bands.keys())
    print(header)
    print("  " + "-" * (len(header) - 2))

    for mode in modes:
        X = extract_tail_features(bags, mode=mode)
        probs = loo_ridge_auroc(X, labels, alpha=100.0)

        results = []
        for band_name, mask in bands.items():
            score = auroc(labels[mask], probs[mask])
            results.append(f"{score:14.3f}")

        print(f"  {mode:20s}" + "".join(results))

    print("=" * 80)


if __name__ == "__main__":
    main()
