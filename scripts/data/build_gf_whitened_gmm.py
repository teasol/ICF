#!/usr/bin/env python3
"""Background GMM on cohort-standardised, PCA-whitened UNI2 patches (GF option 1+2).

Motivation and the failure this addresses are in src/models/branches/gf_whiten.py:
fitted on raw embeddings, the background GMM splits by cohort rather than by
tissue ((M-1)/M of its components are >90% one dataset at every M tried), and an
unseen cohort then collapses onto a single component.

Pipeline, all label-free:

    X_balanced (3M x 1536, 1M per cohort)
      -> per-cohort standardisation            (removes the cohort offset)
      -> PCA-whitening to n_out dimensions     (removes the variance advantage)
      -> GaussianMixture(M, covariance_type='diag')

Everything needed to reproduce the descriptor at evaluation time is written to
--out-dir:

    whitening_d{n_out}.pt    components (already 1/sqrt(eigval)-scaled), the
                             background mean, eigenvalues, background cohort
                             statistics and provenance metadata
    gmm_d{n_out}_model.pkl   the fitted mixture
    summary_d{n_out}.json    convergence, timings, validation log-likelihood,
                             per-cohort component occupancy and purity

The evaluation runner MUST load the same whitening_d{n_out}.pt; a different
projection is a different descriptor.

Usage:
    .venv/bin/python scripts/data/build_gf_whitened_gmm.py --n-out 512 --device cuda:5
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.branches.gf_whiten import GFWhitening, cohort_standardise  # noqa: E402

DATASETS = ("COMET", "MUT-HET-RCC", "PANDA")
DEFAULT_SAMPLE_DIR = Path("/NHNHOME/BASE/kimds/Data/PathoBench/gf_background/fixed_gmm")
DEFAULT_OUT_DIR = Path("/NHNHOME/BASE/kimds/Data/PathoBench/gf_background/fixed_gmm_whitened")


def transform_in_chunks(
    features: torch.Tensor, whitening: GFWhitening, device: torch.device, chunk: int = 262144
) -> np.ndarray:
    """Project a CPU sample through the whitening on `device`, back to CPU float32."""
    out = torch.empty(features.shape[0], whitening.n_out, dtype=torch.float32)
    for start in range(0, features.shape[0], chunk):
        block = features[start:start + chunk].to(device)
        out[start:start + chunk] = whitening.transform(block).cpu()
    return out.numpy()


def occupancy_report(model, X_val: np.ndarray, val_labels: np.ndarray) -> dict:
    """Per-cohort mean responsibility, effective component count and purity."""
    resp = model.predict_proba(X_val)
    per_dataset = np.stack([resp[val_labels == i].mean(0) for i in range(len(DATASETS))])
    effective = {}
    for i, name in enumerate(DATASETS):
        p = per_dataset[i]
        p = p[p > 1e-12]
        effective[name] = float(np.exp(-(p * np.log(p)).sum()))
    purity = per_dataset / per_dataset.sum(0, keepdims=True)
    return {
        "mean_responsibility": {d: per_dataset[i].tolist() for i, d in enumerate(DATASETS)},
        "effective_components": effective,
        "component_dominant_dataset_share": purity.max(0).tolist(),
        "components_above_90pct_purity": int((purity.max(0) > 0.9).sum()),
        "val_loglik": {
            "overall": float(model.score(X_val)),
            **{d: float(model.score(X_val[val_labels == i])) for i, d in enumerate(DATASETS)},
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-out", type=int, required=True, help="PCA-whitened dimension")
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR,
                   help="directory holding X_balanced.npz and val.npz")
    p.add_argument("--sample", default="X_balanced", choices=["X_balanced", "X_raw"])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default="cuda:5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    from sklearn.mixture import GaussianMixture

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    whitening_path = args.out_dir / f"whitening_d{args.n_out}.pt"
    model_path = args.out_dir / f"gmm_d{args.n_out}_model.pkl"
    summary_path = args.out_dir / f"summary_d{args.n_out}.json"

    print(f"=== GF whitened background GMM | n_out={args.n_out} M={args.n_components} ===",
          flush=True)

    t0 = time.monotonic()
    blob = np.load(args.sample_dir / f"{args.sample}.npz")
    features = torch.from_numpy(np.ascontiguousarray(blob["features"]))
    labels = torch.from_numpy(blob["labels"].astype(np.int64))
    val_blob = np.load(args.sample_dir / "val.npz")
    val_features = torch.from_numpy(np.ascontiguousarray(val_blob["features"]))
    val_labels = val_blob["labels"].astype(np.int64)
    print(f"  loaded {tuple(features.shape)} train, {tuple(val_features.shape)} val "
          f"in {time.monotonic() - t0:.1f}s", flush=True)

    # Step 1 -- cohort standardisation. The statistics are kept: they document the
    # frame the PCA basis was fitted in.
    t1 = time.monotonic()
    features, cohort_stats = cohort_standardise(features, labels, len(DATASETS))
    for i, name in enumerate(DATASETS):
        vs = val_labels == i
        if vs.any():
            val_features[torch.from_numpy(vs)] = (
                val_features[torch.from_numpy(vs)] - cohort_stats[i].mean
            ) / cohort_stats[i].std
    print(f"  cohort standardisation {time.monotonic() - t1:.1f}s "
          f"({', '.join(f'{d}:{s.n_patches:,}' for d, s in zip(DATASETS, cohort_stats))})",
          flush=True)

    # Step 2 -- PCA-whitening fitted on the background sample only.
    t2 = time.monotonic()
    if whitening_path.exists() and not args.force:
        whitening = GFWhitening.load(whitening_path)
        print(f"  reusing {whitening_path}", flush=True)
    else:
        whitening = GFWhitening.fit(features, args.n_out, device=device)
        whitening.save(whitening_path, metadata={
            "sample": args.sample,
            "sample_dir": str(args.sample_dir),
            "n_in": whitening.n_in,
            "n_out": whitening.n_out,
            "datasets": list(DATASETS),
            "cohort_stats": {
                name: {
                    "mean": stats.mean.tolist(),
                    "std": stats.std.tolist(),
                    "n_patches": stats.n_patches,
                }
                for name, stats in zip(DATASETS, cohort_stats)
            },
            "built_by": "scripts/data/build_gf_whitened_gmm.py",
        })
        print(f"  PCA-whitening fitted and saved -> {whitening_path}", flush=True)
    whitening = whitening.to(device)
    total_variance = float(whitening.explained_variance.sum())
    print(f"  eigenvalues: top {whitening.explained_variance[0]:.4f} .. "
          f"tail {whitening.explained_variance[-1]:.6f}  "
          f"({time.monotonic() - t2:.1f}s)", flush=True)

    t3 = time.monotonic()
    X = transform_in_chunks(features, whitening, device)
    X_val = transform_in_chunks(val_features, whitening, device)
    del features, val_features
    print(f"  projected to {X.shape} / {X_val.shape} in {time.monotonic() - t3:.1f}s", flush=True)

    # Step 3 -- the mixture itself.
    if model_path.exists() and not args.force:
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        fit_seconds = None
        print(f"  reusing {model_path}", flush=True)
    else:
        t4 = time.monotonic()
        model = GaussianMixture(
            n_components=args.n_components, covariance_type="diag",
            random_state=args.seed, max_iter=args.max_iter, tol=args.tol,
            n_init=1, init_params="k-means++",
        ).fit(X)
        fit_seconds = time.monotonic() - t4
        tmp = model_path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as handle:
            pickle.dump(model, handle)
        tmp.rename(model_path)
        print(f"  GMM converged={model.converged_} n_iter={model.n_iter_} "
              f"in {fit_seconds:.1f}s -> {model_path}", flush=True)

    report = occupancy_report(model, X_val, val_labels)
    summary = {
        "n_out": args.n_out,
        "n_components": args.n_components,
        "sample": args.sample,
        "seed": args.seed,
        "converged": bool(model.converged_),
        "n_iter": int(model.n_iter_),
        "fit_seconds": fit_seconds,
        "whitening_path": str(whitening_path),
        "model_path": str(model_path),
        "explained_variance_sum": total_variance,
        "component_weights": model.weights_.tolist(),
        **report,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"  components >90% single-dataset purity: "
          f"{report['components_above_90pct_purity']}/{args.n_components}", flush=True)
    print(f"  effective components per cohort: "
          f"{', '.join(f'{k} {v:.2f}' for k, v in report['effective_components'].items())}",
          flush=True)
    print(f"  wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
