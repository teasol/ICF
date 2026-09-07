"""RU-85 stage A: the label-free kill conditions for AKS, MDX and LID.

Every threshold here was declared in the RU-85 card before the run, taken from the
proposals themselves. No labels are read and no performance number is computed -
that is what makes this a gate-1 screen rather than a candidate comparison.

Diagnostics layout, as recorded by scripts/test_pathobench.py:
  d_aks = [angular spectrum entropy H_q, fourth-moment entropy H_p, log N]
  d_mdx = [unimodal fraction at b=0.25, max valley depth, degenerate eig-ratio frac]
  d_lid = [TwoNN ID (median), |ID_20 - ID_5|, duplicate-token rate]
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def pooled(folds, key):
    """Stack one recorded tensor across every fold's query slides."""
    parts = [f[key] for f in folds if f.get(key) is not None]
    if not parts:
        return None
    return torch.cat([p.float() for p in parts]).numpy()


def correlate(a, b):
    if a is None or b is None or a.size < 3:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ru85_stageA")
    args = ap.parse_args()

    files = sorted(glob.glob(str(ROOT / f"predictions/*_{args.tag}_official50_bf16.pt")))
    if not files:
        raise SystemExit(f"no predictions for tag {args.tag}")

    report = {"tag": args.tag, "tasks": [], "candidates": {}}
    folds = []
    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        report["tasks"].append({"file": Path(path).name, "n_folds": len(blob["per_fold"]),
                                "fold_auroc_mean": blob["fold_auroc_mean"]})
        folds.extend(blob["per_fold"])
    print(f"tag={args.tag}  files={len(files)}  folds={len(folds)}")

    d_aks, d_mdx, d_lid = (pooled(folds, k) for k in ("d_aks", "d_mdx", "d_lid"))
    log_n = d_aks[:, 2] if d_aks is not None else None
    n_slides = 0 if d_aks is None else d_aks.shape[0]
    print(f"query slide-fold pairs = {n_slides}\n")

    # ---- AKS ----------------------------------------------------------------
    if d_aks is not None:
        h_q = d_aks[:, 0]
        frac_flat = float((h_q > 0.99).mean())
        sd = float(h_q.std(ddof=1))
        r_split = correlate(pooled(folds, "m_akd"), pooled(folds, "m_akf"))
        dead = frac_flat >= 0.90 and sd < 0.01
        report["candidates"]["aks"] = {
            "H_q_mean": float(h_q.mean()), "H_q_sd": sd,
            "H_q_fraction_above_0.99": frac_flat,
            "H_p_mean": float(d_aks[:, 1].mean()),
            "r_angular_vs_fourth": r_split,
            "verdict": "dead" if dead else "alive",
            "redundant_8d": bool(abs(r_split) > 0.9),
        }
        print("AKS  isotropy kill: H_q > 0.99 in >=90% of slides AND sd < 0.01")
        print(f"     H_q mean={h_q.mean():.4f} sd={sd:.4f} frac>0.99={frac_flat:.3f}"
              f"  ->  {'DEAD' if dead else 'ALIVE'}")
        print(f"     |r|(m_akd, m_akf) = {abs(r_split):.3f}"
              f"  ->  {'8D redundant, fall back to 4D' if abs(r_split) > 0.9 else 'both halves carry signal'}\n")

    # ---- MDX ----------------------------------------------------------------
    if d_mdx is not None:
        unimodal = float(d_mdx[:, 0].mean())
        depth_sd = float(d_mdx[:, 1].std(ddof=1))
        r_grid = correlate(pooled(folds, "m_mdx"), pooled(folds, "m_mdx129"))
        degenerate = float(d_mdx[:, 2].mean())
        dead_uni = unimodal >= 0.95
        dead_depth = depth_sd < 0.02
        dead_grid = abs(r_grid) < 0.9
        report["candidates"]["mdx"] = {
            "unimodal_pair_fraction": unimodal, "max_depth_sd": depth_sd,
            "max_depth_mean": float(d_mdx[:, 1].mean()),
            "r_grid_257_vs_129": r_grid,
            "degenerate_eigratio_fraction": degenerate,
            "verdict": "dead" if (dead_uni or dead_depth or dead_grid) else "alive",
            "dead_by": [n for n, f in (("unimodal", dead_uni), ("depth_sd", dead_depth),
                                        ("grid_dependence", dead_grid)) if f],
        }
        print("MDX  kills: unimodal pairs >=95%  OR  depth sd < 0.02  OR  |r| vs G=129 < 0.9")
        print(f"     unimodal fraction={unimodal:.3f}  depth sd={depth_sd:.4f} "
              f"(mean {d_mdx[:, 1].mean():.4f})  |r| grid={abs(r_grid):.3f}")
        print(f"     degenerate eigenvalue-ratio fraction={degenerate:.4f} "
              f"(proposal degrades to 4 directions above 0.05)")
        print(f"     ->  {report['candidates']['mdx']['verdict'].upper()}"
              f"{' by ' + ', '.join(report['candidates']['mdx']['dead_by']) if report['candidates']['mdx']['dead_by'] else ''}\n")

    # ---- LID ----------------------------------------------------------------
    if d_lid is not None:
        id_sd = float(d_lid[:, 0].std(ddof=1))
        flat_scale = float((d_lid[:, 1] < 0.3).mean())
        r_sub = correlate(pooled(folds, "m_lid"), pooled(folds, "m_lid1024"))
        r_size_margin = correlate(pooled(folds, "m_lid"), log_n)
        r_size_id = correlate(d_lid[:, 0], log_n)
        dup = float(d_lid[:, 2].mean())
        dead_sd = id_sd < 0.2
        dead_scale = flat_scale >= 0.95
        dead_sub = abs(r_sub) < 0.8
        report["candidates"]["lid"] = {
            "id_twonn_mean": float(d_lid[:, 0].mean()), "id_twonn_sd": id_sd,
            "fraction_flat_multiscale": flat_scale,
            "r_subsample_4096_vs_1024": r_sub,
            "r_margin_vs_logN": r_size_margin, "r_id_vs_logN": r_size_id,
            "duplicate_token_rate": dup,
            "verdict": "dead" if (dead_sd or dead_scale or dead_sub) else "alive",
            "dead_by": [n for n, f in (("id_sd", dead_sd), ("flat_multiscale", dead_scale),
                                        ("subsample", dead_sub)) if f],
            "size_proxy_flag": bool(abs(r_size_margin) > 0.5),
        }
        print("LID  kills: ID sd < 0.2  OR  |ID20-ID5| < 0.3 in >=95%  OR  |r| m=4096 vs 1024 < 0.8")
        print(f"     ID_TwoNN mean={d_lid[:, 0].mean():.3f} sd={id_sd:.3f}  "
              f"flat multiscale frac={flat_scale:.3f}  |r| subsample={abs(r_sub):.3f}")
        print(f"     size proxying: |r|(m_lid, log N)={abs(r_size_margin):.3f}, "
              f"|r|(ID, log N)={abs(r_size_id):.3f}  (flag above 0.5)")
        print(f"     duplicate-token rate={dup:.5f}")
        print(f"     ->  {report['candidates']['lid']['verdict'].upper()}"
              f"{' by ' + ', '.join(report['candidates']['lid']['dead_by']) if report['candidates']['lid']['dead_by'] else ''}\n")

    survivors = [k for k, v in report["candidates"].items() if v["verdict"] == "alive"]
    report["survivors"] = survivors
    print(f"stage A survivors -> {survivors or 'none'}")

    out = ROOT / "predictions" / f"{args.tag}_gate1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
