#!/usr/bin/env python3
"""Standalone GF (GMM Fisher Vector) branch evaluation on the Primary 7 tasks.

GF is NOT screened against the other branches and is NOT aggregated with them.
By user decision (2026-09-16) the adoption gates (docs/PROJECT.md SS5) and the
promotion review (SS4) are skipped for this candidate, so this runner answers
one question only: what does the GF branch do on its own, per task and per
fold, when its basis is a background GMM fitted offline on disjoint cohorts?

Nothing here may be compared against the official 7-branch basis (macro 0.6227)
or used to claim adoption -- it is an exploration record. SEAL 10 is never
touched.

Per slide the runner computes one Fisher Vector against the fixed GMM (the GMM
does not depend on the fold, so each descriptor is computed once and reused
across all 50 folds), then reads out a margin through the same class-balanced
kernel ridge the other branches use. It also reports, per task, the mean
responsibility over every patch -- the effective number of components tells you
how much of the Fisher Vector a cohort actually occupies.

Usage:
    .venv/bin/python scripts/run_gf_standalone.py \
        --gmm /NHNHOME/BASE/kimds/Data/PathoBench/gf_background/fixed_gmm/gmm_balanced_model.pkl \
        --device cuda:4 --tag m8_balanced
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_pure import (  # noqa: E402
    index_h5_files, load_official_folds, load_slide_features,
)
from src.models.branches.gf import (  # noqa: E402
    BackgroundGMM, gf_margins_from_descriptors, gf_slide_features,
)
from src.models.branches.gf_whiten import (  # noqa: E402
    GFWhitening, accumulate_moments, finalize_moments,
)
from src.utils.metrics import auroc  # noqa: E402

PRIMARY7 = [
    "cptac_lscc/ARID1A_mutation",
    "cptac_lscc/Histologic_Grade",
    "cptac_lscc/KEAP1_mutation",
    "cptac_luad/KRAS_mutation",
    "cptac_pda/SMAD4_mutation",
    "ucla_lung/progression_regression",
    "cptac_ccrcc/PBRM1_mutation",
]

DEFAULT_DATA_ROOT = Path("/NHNHOME/BASE/kimds/Data/PathoBench")


def effective_components(occupancy: torch.Tensor) -> float:
    """exp(entropy) of a mean responsibility vector: how many components a cohort
    actually spreads over (1.0 = everything collapses onto a single component)."""
    p = occupancy[occupancy > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def evaluate_task(
    task: str,
    gmm: BackgroundGMM,
    args: argparse.Namespace,
    device: torch.device,
    whitening: GFWhitening | None = None,
) -> dict:
    task_dir = args.official_root / task
    records, slide_ids, labels, fold_cols = load_official_folds(task_dir)

    h5_index = index_h5_files(args.features_root)
    slide_ids = [s for s in slide_ids if s in h5_index]
    missing = len(records) - len(slide_ids)
    if missing:
        print(f"  WARNING: dropping {missing} slides with no feature file", flush=True)
    records_by_sid = {str(r["slide_id"]).strip(): r for r in records}

    # Cohort standardisation needs this cohort's own patch statistics. They are
    # gathered over every slide of the task in a separate pass: label-free, but
    # transductive over the cohort -- recorded in the summary as such.
    cohort_stats = None
    if whitening is not None:
        t_stats = time.monotonic()
        total = torch.zeros(whitening.n_in, dtype=torch.float64, device=device)
        total_sq = torch.zeros(whitening.n_in, dtype=torch.float64, device=device)
        count = 0
        for sid in slide_ids:
            total, total_sq, count = accumulate_moments(
                load_slide_features(sid, h5_index).to(device), total, total_sq, count
            )
        cohort_stats = finalize_moments(total, total_sq, count).to(device)
        print(f"  cohort statistics over {count:,} patches "
              f"({time.monotonic() - t_stats:.1f}s)", flush=True)

    # One Fisher Vector per slide, computed once for all folds.
    t0 = time.monotonic()
    descriptors: dict[str, torch.Tensor] = {}
    occupancy_sum = torch.zeros(gmm.n_components, device=device, dtype=torch.float64)
    patch_total = 0
    for i, sid in enumerate(slide_ids):
        bag = load_slide_features(sid, h5_index).to(device)
        if whitening is not None:
            bag = whitening.transform(bag, cohort_stats)
        vector, occupancy = gf_slide_features(
            bag, gmm, alpha=args.alpha,
            include_variance=not args.mean_only, return_occupancy=True,
        )
        if not bool(torch.isfinite(vector).all()) or not bool(torch.isfinite(occupancy).all()):
            raise FloatingPointError(f"non-finite GF output for slide {sid}")
        responsibility_error = abs(float(occupancy.sum()) - 1.0)
        if responsibility_error > 1e-4:
            raise FloatingPointError(
                f"GF responsibility sum error {responsibility_error:.3e} for slide {sid}"
            )
        descriptors[sid] = vector
        occupancy_sum += occupancy.double() * bag.shape[0]
        patch_total += bag.shape[0]
        if (i + 1) % 100 == 0:
            print(f"  descriptors {i + 1}/{len(slide_ids)}", flush=True)
    descriptor_seconds = time.monotonic() - t0

    occupancy = (occupancy_sum / max(patch_total, 1)).float()
    print(f"  {len(slide_ids)} slides, {patch_total:,} patches, "
          f"FV dim {descriptors[slide_ids[0]].numel()}, {descriptor_seconds:.1f}s", flush=True)
    print(f"  mean responsibility: "
          f"{' '.join(f'{v:.4f}' for v in occupancy.tolist())}", flush=True)
    print(f"  effective components: {effective_components(occupancy):.2f}"
          f" / {gmm.n_components}", flush=True)

    total_folds = len(fold_cols)
    n_folds = args.nfolds or total_folds
    per_fold: list[dict] = []
    for k in range(min(n_folds, total_folds)):
        fc = fold_cols[k]
        test_ids = [s for s in slide_ids if records_by_sid[s][fc].strip() == "test"]
        context_ids = [s for s in slide_ids if records_by_sid[s][fc].strip() != "test"]
        if len(test_ids) < 2:
            print(f"  fold {k + 1}/{total_folds}: skip (only {len(test_ids)} test slides)")
            continue
        target = torch.tensor([labels[s] for s in test_ids], dtype=torch.long)
        if target.unique().numel() < 2:
            print(f"  fold {k + 1}/{total_folds}: skip (single-class query set)")
            continue
        context_labels = torch.tensor(
            [labels[s] for s in context_ids], dtype=torch.long, device=device
        )
        if context_labels.unique().numel() < 2:
            print(f"  fold {k + 1}/{total_folds}: skip (single-class context set)")
            continue

        with torch.no_grad():
            margin = gf_margins_from_descriptors(
                torch.stack([descriptors[s] for s in context_ids]),
                context_labels,
                torch.stack([descriptors[s] for s in test_ids]),
                reg_lambda=args.reg_lambda,
            )
        margin = margin.detach().to("cpu", dtype=torch.float32)
        if not bool(torch.isfinite(margin).all()):
            raise FloatingPointError(f"non-finite GF margin in {task} fold {k}")
        fold_auroc = float(auroc(torch.sigmoid(margin), target))
        per_fold.append({
            "fold": k, "slide_id": test_ids, "label": target,
            "margin": margin, "auroc": fold_auroc,
        })
        print(f"  fold {k + 1}/{total_folds}: AUROC {fold_auroc:.4f}  n_query {len(margin)}",
              flush=True)

    aurocs = [e["auroc"] for e in per_fold]
    fold_mean = sum(aurocs) / max(len(aurocs), 1)
    above_half = sum(1 for a in aurocs if a > 0.5)
    print(f"  == {task}: fold-mean AUROC {fold_mean:.4f}  "
          f"({above_half}/{len(aurocs)} folds > 0.5)", flush=True)

    return {
        "task": task,
        "n_slides": len(slide_ids),
        "n_patches": patch_total,
        "fold_mean_auroc": fold_mean,
        "n_folds_scored": len(aurocs),
        "folds_above_half": above_half,
        "per_fold_auroc": aurocs,
        "mean_responsibility": occupancy.tolist(),
        "effective_components": effective_components(occupancy),
        "descriptor_seconds": descriptor_seconds,
        "cohort_stats_patches": None if cohort_stats is None else cohort_stats.n_patches,
        "_per_fold": per_fold,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gmm", type=Path, required=True,
                   help="path to a *_model.pkl from build_gf_background_gmm.py")
    p.add_argument("--tag", required=True, help="output tag, e.g. m8_balanced")
    p.add_argument("--device", default="cuda:4")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--features-root", type=Path, default=None)
    p.add_argument("--official-root", type=Path, default=None)
    p.add_argument("--tasks", default=",".join(PRIMARY7))
    p.add_argument("--nfolds", type=int, default=50)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="signed power-normalisation exponent (0.5 = improved FV)")
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--mean-only", action="store_true",
                   help="drop the variance-gradient half of the Fisher Vector")
    p.add_argument("--whitening", type=Path, default=None,
                   help="whitening_d*.pt from build_gf_whitened_gmm.py. MUST be the same "
                        "basis the GMM was fitted on -- a different projection is a "
                        "different descriptor. Enables per-cohort standardisation too.")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "predictions/gf_standalone")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.features_root = args.features_root or (args.data_root / "features")
    args.official_root = args.official_root or (args.data_root / "official")
    device = torch.device(args.device)

    gmm = BackgroundGMM.from_pickle(args.gmm).to(device)

    whitening = None
    if args.whitening is not None:
        whitening = GFWhitening.load(args.whitening).to(device)
        if whitening.n_out != gmm.n_features:
            raise ValueError(
                f"whitening produces {whitening.n_out} dims but the GMM expects "
                f"{gmm.n_features}: this is not the basis the GMM was fitted on"
            )
        print(f"  whitening {args.whitening} ({whitening.n_in} -> {whitening.n_out}), "
              f"per-cohort standardisation ON", flush=True)

    print(f"GF standalone | GMM {args.gmm}", flush=True)
    print(f"  M={gmm.n_components} D={gmm.n_features} FV dim={gmm.feature_dim} "
          f"alpha={args.alpha} lambda={args.reg_lambda} device={args.device}", flush=True)
    print("  gates and promotion review are SKIPPED by user decision; "
          "results are an exploration record only.", flush=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = []
    for task in tasks:
        print(f"\n=== {task} ===", flush=True)
        results.append(evaluate_task(task, gmm, args, device, whitening))

    macro = sum(r["fold_mean_auroc"] for r in results) / max(len(results), 1)
    print(f"\n=== GF standalone macro over {len(results)} tasks: {macro:.4f} ===", flush=True)
    for r in results:
        print(f"  {r['task']:45s} {r['fold_mean_auroc']:.4f}  "
              f"({r['folds_above_half']}/{r['n_folds_scored']} folds > 0.5, "
              f"eff.components {r['effective_components']:.2f})", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"tag": args.tag, "gmm": str(args.gmm), "results": results},
        args.out_dir / f"gf_{args.tag}_per_fold.pt",
    )
    summary = {
        "tag": args.tag,
        "gmm": str(args.gmm),
        "whitening": None if args.whitening is None else str(args.whitening),
        "whitening_dim": None if whitening is None else whitening.n_out,
        "cohort_standardised": whitening is not None,
        "n_components": gmm.n_components,
        "fv_dim": gmm.feature_dim,
        "alpha": args.alpha,
        "reg_lambda": args.reg_lambda,
        "mean_only": args.mean_only,
        "macro_auroc": macro,
        "tasks": [{k: v for k, v in r.items() if k != "_per_fold"} for r in results],
    }
    summary_path = args.out_dir / f"gf_{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
