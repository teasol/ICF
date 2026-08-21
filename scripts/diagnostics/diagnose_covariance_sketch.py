#!/usr/bin/env python
"""Diagnose the covariance sketch's projection basis (docs SS69).

CV-only training showed the two covariance terms carry essentially all of this
model's discrimination (CV-1 alone 0.9052 vs the full model's 0.9199, SS68), so
the sketch's construction is now the thing worth improving. Three questions,
answered WITHOUT training:

  1. FREQUENCY LADDER. The basis is deterministic sinusoidal:
         directions[d,k] = sin(0.019*d*k) + cos(0.011*(d+1)*k)   d=1..1536, k=1..64
     then orthonormalised by QR. The two slopes are hardcoded. With d up to 1536
     and k up to 64 the argument reaches ~1867 rad (~297 cycles), so the high-kd
     corner is heavily aliased. If the PRE-QR matrix is near rank-deficient, the
     later Q columns are numerical noise rather than structure -- QR always
     returns orthonormal columns, so the conditioning must be read BEFORE it.

  2. SKETCH DIMENSION. 64 -> 64*65/2 = 2080 features. Cost grows as d^2.

  3. BASIS CHOICE. The basis is deterministic, NOT random -- so it carries no
     Johnson-Lindenstrauss distance-preservation guarantee. This compares it
     against a random Gaussian basis (which does) and against PCA on the actual
     cells (a data-adaptive but label-free upper bound for a learnable
     projection).

Stage 3 probes with the SAME class-balanced dual ridge that CV-1 uses, on the
official er_status 50 folds, so the number it reports is directly comparable to
the CV-1 branch AUROC rather than a proxy.

Usage:
  python scripts/diagnose_covariance_sketch.py [--stages 123] [--dims 16,32,64,128]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from src.utils.metrics import auroc as _auroc  # noqa: E402

FEATURE_DIM = 1536


def sinusoidal_basis(input_dim: int, sketch_dim: int, slopes=(0.019, 0.011)):
    """The basis actually used, plus its pre-QR matrix (baseline.py:687-700)."""
    a, b = slopes
    d = torch.arange(1, input_dim + 1, dtype=torch.float64)[:, None]
    k = torch.arange(1, sketch_dim + 1, dtype=torch.float64)[None, :]
    raw = torch.sin(a * d * k) + torch.cos(b * (d + 1) * k)
    return torch.linalg.qr(raw, mode="reduced").Q, raw


def gaussian_basis(input_dim: int, sketch_dim: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(input_dim, sketch_dim, generator=g, dtype=torch.float64)
    return torch.linalg.qr(raw, mode="reduced").Q, raw


def stage1(dims) -> None:
    print("=" * 72)
    print("STAGE 1 — 주파수 사다리와 사영 기저의 조건수")
    print("=" * 72)
    print("QR 이후 P는 정의상 정규직교(모든 특이값 1)이므로 QR **이전** 행렬을 본다.")
    print("pre-QR이 rank-deficient면 뒤쪽 Q 열은 구조가 아니라 반올림 잡음이다.\n")
    print(f"{'sketch_dim':>10} {'pre-QR rank':>12} {'cond':>12} "
          f"{'sv_last/sv_1':>13}   {'basis':<10}")
    for dim in dims:
        for name, fn in (("sinusoid", sinusoidal_basis), ("gaussian", gaussian_basis)):
            _, raw = fn(FEATURE_DIM, dim)
            sv = torch.linalg.svdvals(raw)
            rank = int((sv > sv[0] * 1e-10).sum())
            print(f"{dim:>10} {rank:>12} {sv[0] / sv[-1]:>12.3e} "
                  f"{sv[-1] / sv[0]:>13.3e}   {name:<10}")
    print("\n앨리어싱 점검 — sin(0.019*d*k) 인자의 최대 라디안:")
    for dim in dims:
        arg = 0.019 * FEATURE_DIM * dim
        print(f"  sketch_dim={dim:>4}: {arg:>9.1f} rad = {arg / (2 * 3.14159265):>7.1f} cycles")


def _load_cells(task_dir: Path, features: Path, max_slides: int, max_cells: int):
    from test_pathobench import index_h5_files, load_slide_features
    import csv as _csv
    import yaml

    tsv = task_dir / "k=all.tsv"
    task_col = yaml.safe_load((task_dir / "config.yaml").read_text())["task_col"]
    with tsv.open() as fh:
        records = list(_csv.DictReader(fh, delimiter="\t"))
    h5 = index_h5_files(features)
    fold_cols = [c for c in records[0] if c.startswith("fold_")]
    slides, labels, folds = [], [], []
    for r in records[:max_slides]:
        sid = str(r["slide_id"]).strip()
        if sid not in h5:
            continue
        x = load_slide_features(sid, h5)
        if x.shape[0] > max_cells:
            x = x[torch.randperm(x.shape[0])[:max_cells]]
        slides.append(x)
        labels.append(int(float(r[task_col])))
        folds.append({c: r[c] for c in fold_cols})
    return slides, torch.tensor(labels), folds, fold_cols


def _sketch(cells, basis, shrinkage=0.1):
    """Mirror of aggregator._covariance_sketch with covariance_mode=correlation."""
    centered = (cells - cells.mean(dim=0, keepdim=True)).double()
    projected = centered @ basis
    cov = projected.T @ projected / projected.shape[0]
    diag = cov.diagonal().clamp_min(1e-6)
    inv = diag.rsqrt()
    corr = cov * inv[None, :] * inv[:, None]
    if shrinkage:
        corr = (1 - shrinkage) * corr + shrinkage * torch.eye(
            corr.shape[0], dtype=corr.dtype
        )
    row, col = torch.triu_indices(corr.shape[0], corr.shape[0])
    return corr[row, col]


def _dual_ridge_auroc(sketches, labels, folds, fold_cols, lam=1.0):
    """Class-balanced dual ridge = exactly what CV-1 does, scored per fold."""
    aurocs = []
    X = torch.stack(sketches).double()
    X = X - X.mean(dim=0, keepdim=True)
    X = X / X.square().mean().sqrt().clamp_min(1e-6)
    y = labels.double()
    for col in fold_cols:
        test = torch.tensor([f[col].strip() == "test" for f in folds])
        if test.sum() < 2 or len(set(y[test].tolist())) < 2:
            continue
        ctx = ~test
        Xc, yc = X[ctx], y[ctx]
        counts = torch.bincount(yc.long(), minlength=2).double().clamp_min(1)
        w = counts.reciprocal()[yc.long()]
        wsum = w.sum().clamp_min(1e-12)
        fmean = (w[:, None] * Xc).sum(0, keepdim=True) / wsum
        tmean = (w * yc).sum() / wsum
        Xc_c, yc_c = Xc - fmean, yc - tmean
        rw = w.sqrt()[:, None]
        design, target = Xc_c * rw, yc_c[:, None] * rw
        gram = design @ design.T
        alpha = torch.linalg.solve(
            gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype), target
        )
        scores = (X[test] - fmean) @ (design.T @ alpha) + tmean
        aurocs.append(_auroc(scores.flatten().float(), y[test].float()))
    t = torch.tensor(aurocs)
    return float(t.mean()), float(t.std()), len(aurocs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="123")
    ap.add_argument("--dims", default="16,32,48,64,96,128")
    ap.add_argument("--task-dir", type=Path,
                    default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/official/bc_therapy/er_status"))
    ap.add_argument("--features", type=Path,
                    default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/features"))
    ap.add_argument("--max-slides", type=int, default=166)
    ap.add_argument("--max-cells", type=int, default=3000)
    args = ap.parse_args()
    dims = [int(d) for d in args.dims.split(",")]

    if "1" in args.stages:
        stage1(dims)

    if "2" not in args.stages and "3" not in args.stages:
        return

    print("\n" + "=" * 72)
    print("STAGE 2/3 — 실데이터(er_status)에서 정보 보존율 + 차원·기저 스윕")
    print("=" * 72)
    torch.manual_seed(0)
    slides, labels, folds, fold_cols = _load_cells(
        args.task_dir, args.features, args.max_slides, args.max_cells
    )
    print(f"slides={len(slides)}  labels={labels.sum().item()}/{len(labels)} positive  "
          f"folds={len(fold_cols)}  (cells capped at {args.max_cells})\n")

    if "2" in args.stages:
        pooled = torch.cat([s[: min(200, s.shape[0])] for s in slides]).double()
        pooled = pooled - pooled.mean(0, keepdim=True)
        total = pooled.square().sum()
        print("정보 보존율  ||X P||_F^2 / ||X||_F^2   (클수록 분산을 많이 담음)")
        print(f"{'dim':>6} {'sinusoid':>10} {'gaussian':>10} {'PCA':>10}")
        u, s, v = torch.linalg.svd(pooled, full_matrices=False)
        for dim in dims:
            row = [dim]
            for fn in (sinusoidal_basis, gaussian_basis):
                P, _ = fn(FEATURE_DIM, dim)
                row.append(float((pooled @ P).square().sum() / total))
            row.append(float(s[:dim].square().sum() / total))
            print(f"{row[0]:>6} {row[1]:>10.4f} {row[2]:>10.4f} {row[3]:>10.4f}")
        pca_basis = {d: v[:d].T.contiguous() for d in dims}
    else:
        pca_basis = {}

    if "3" in args.stages:
        print("\nCV-1과 동일한 class-balanced dual ridge로 sketch만 바꿔 50-fold 채점")
        print(f"{'dim':>6} {'feats':>7} {'sinusoid':>16} {'gaussian':>16} {'PCA':>16}")
        for dim in dims:
            cells = []
            out = [f"{dim:>6}", f"{dim * (dim + 1) // 2:>7}"]
            for name, fn in (("sinusoid", sinusoidal_basis), ("gaussian", gaussian_basis)):
                P, _ = fn(FEATURE_DIM, dim)
                sk = [_sketch(s, P) for s in slides]
                m, sd, n = _dual_ridge_auroc(sk, labels, folds, fold_cols)
                out.append(f"{m:.4f}±{sd:.3f}")
            if dim in pca_basis:
                sk = [_sketch(s, pca_basis[dim].double()) for s in slides]
                m, sd, n = _dual_ridge_auroc(sk, labels, folds, fold_cols)
                out.append(f"{m:.4f}±{sd:.3f}")
            print("  ".join(out))
        print("\n참고: v40 CV-only(학습된 전체 모델)의 er_status 50-fold와 직접 비교할 것.")


if __name__ == "__main__":
    main()
