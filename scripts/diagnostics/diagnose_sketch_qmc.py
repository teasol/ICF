#!/usr/bin/env python
"""Is the sinusoidal basis a low-discrepancy (QMC) point set? (docs SS69)

Hypothesis: an aliased sinusoidal ladder sin(a*k*d), d=1,2,..., with a*k not a
rational multiple of 2*pi is a Weyl equidistributed sequence -- i.e. a
low-discrepancy quasi-random set. QMC theory says such sets cover space MORE
uniformly than i.i.d. draws (discrepancy O(log^s N / N) vs O(N^-1/2)), which
would explain the measured ordering
    low-frequency sinusoid  <  i.i.d. gaussian  <  aliased sinusoid
        0.6614                    0.6801                 0.7632
If an EXPLICIT low-discrepancy basis (scrambled Sobol) also beats i.i.d.
gaussian by a similar margin, the QMC explanation is supported. If it does not,
the sinusoidal grid has some other property and the hypothesis fails.

All three families are orthonormalised by the same QR and scored by the same
class-balanced dual ridge CV-1 uses, on the official er_status 50 folds.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from diagnose_covariance_sketch import (  # noqa: E402
    _load_cells, _sketch, _dual_ridge_auroc, sinusoidal_basis, gaussian_basis, FEATURE_DIM,
)

def sobol_basis(input_dim, sketch_dim, seed=0):
    eng = torch.quasirandom.SobolEngine(dimension=sketch_dim, scramble=True, seed=seed)
    u = eng.draw(input_dim).double().clamp(1e-6, 1 - 1e-6)
    z = torch.special.ndtri(u)          # uniform -> normal, keeps the QMC structure
    return torch.linalg.qr(z, mode="reduced").Q, z

ap = argparse.ArgumentParser()
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--max-cells", type=int, default=2000)
ap.add_argument("--task-dir", type=Path, default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/official/bc_therapy/er_status"))
ap.add_argument("--features", type=Path, default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/features"))
a = ap.parse_args()

torch.manual_seed(0)
slides, labels, folds, fold_cols = _load_cells(a.task_dir, a.features, 166, a.max_cells)
print(f"slides={len(slides)} folds={len(fold_cols)} dim={a.dim} cells<={a.max_cells}\n")

def score(P):
    return _dual_ridge_auroc([_sketch(s, P) for s in slides], labels, folds, fold_cols)[0]

def summarise(name, vals):
    t = torch.tensor(vals)
    print(f"{name:<26} {t.mean():>8.4f} {t.std():>7.4f} {t.min():>8.4f} {t.max():>8.4f}   "
          + " ".join(f"{v:.4f}" for v in vals))

print(f"{'family':<26} {'mean':>8} {'std':>7} {'min':>8} {'max':>8}   개별값")
summarise("i.i.d. gaussian", [score(gaussian_basis(FEATURE_DIM, a.dim, seed=s)[0]) for s in range(a.seeds)])
summarise("Sobol (QMC, scrambled)", [score(sobol_basis(FEATURE_DIM, a.dim, seed=s)[0]) for s in range(a.seeds)])

# aliased-region sinusoidal ladders: a*dim must exceed pi (=> a > 0.049 at dim 64)
SLOPES = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 2.5]
vals = [score(sinusoidal_basis(FEATURE_DIM, a.dim, slopes=(s, s * 0.733))[0]) for s in SLOPES]
summarise("sinusoid, aliased a", vals)
print(f"{'':<26} {'':>8} {'':>7} {'':>8} {'':>8}   a = " + " ".join(f"{s:.2f} " for s in SLOPES))

LOW = [0.005, 0.010, 0.019, 0.030]
vals = [score(sinusoidal_basis(FEATURE_DIM, a.dim, slopes=(s, s * 0.579))[0]) for s in LOW]
summarise("sinusoid, non-aliased a", vals)
print(f"{'':<26} {'':>8} {'':>7} {'':>8} {'':>8}   a = " + " ".join(f"{s:.3f}" for s in LOW))
print("\n(현행은 a=0.019, b=0.011 -> 비앨리어싱 계열)")
