#!/usr/bin/env python
"""Is the sinusoidal ladder special, or is any fixed subspace equivalent? (SS69)

Two questions in one grid, both scored with the SAME class-balanced dual ridge
CV-1 uses, on the official er_status 50 folds:

  * SEED VARIANCE -- N random Gaussian bases per dim. The single-seed dim=16
    result (0.7562) needs a spread before it can be believed.
  * LADDER SPACING -- the basis is sin(a*d*k) + cos(b*(d+1)*k) with a=0.019,
    b=0.011 hardcoded. Varying (a, b) over orders of magnitude tests directly
    whether the ladder's spacing carries any information.
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

ap = argparse.ArgumentParser()
ap.add_argument("--dims", default="16,32,64")
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--max-cells", type=int, default=2000)
ap.add_argument("--task-dir", type=Path, default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/official/bc_therapy/er_status"))
ap.add_argument("--features", type=Path, default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/features"))
a = ap.parse_args()
dims = [int(d) for d in a.dims.split(",")]

torch.manual_seed(0)
slides, labels, folds, fold_cols = _load_cells(a.task_dir, a.features, 166, a.max_cells)
print(f"slides={len(slides)} folds={len(fold_cols)} cells<= {a.max_cells}\n")

def score(P):
    sk = [_sketch(s, P) for s in slides]
    return _dual_ridge_auroc(sk, labels, folds, fold_cols)[0]

print("A) 랜덤 가우시안 기저 — seed 분산")
print(f"{'dim':>5} {'mean':>8} {'std':>7} {'min':>8} {'max':>8}   개별 seed")
for d in dims:
    vals = [score(gaussian_basis(FEATURE_DIM, d, seed=s)[0]) for s in range(a.seeds)]
    t = torch.tensor(vals)
    print(f"{d:>5} {t.mean():>8.4f} {t.std():>7.4f} {t.min():>8.4f} {t.max():>8.4f}   "
          + " ".join(f"{v:.4f}" for v in vals))

print("\nB) 사인 사다리 간격 (a, b) 변형 — 현행은 (0.019, 0.011)")
LADDERS = [(0.019, 0.011), (0.0019, 0.0011), (0.19, 0.11), (0.019, 0.019),
           (0.5, 0.3), (1.5, 1.1), (3.0, 2.5)]
print(f"{'dim':>5} " + " ".join(f"{str(l):>16}" for l in LADDERS))
for d in dims:
    row = [f"{d:>5}"]
    for slopes in LADDERS:
        P, _ = sinusoidal_basis(FEATURE_DIM, d, slopes=slopes)
        row.append(f"{score(P):>16.4f}")
    print(" ".join(row))

print("\nC) 나이퀴스트 점검 — 열 k의 각주파수 a*k (rad/sample), pi=3.1416 초과 시 앨리어싱")
for slopes in [(0.019, 0.011), (0.19, 0.11), (3.0, 2.5)]:
    for d in (16, 64):
        w = slopes[0] * d
        print(f"  a={slopes[0]:<6} dim={d:<4} 최고열 각주파수={w:>7.3f} rad/sample "
              f"({'앨리어싱' if w > 3.14159 else f'{2*3.14159/w:.1f} samples/cycle'})")
