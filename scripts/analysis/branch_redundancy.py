"""Branch correlation and effective rank on real Primary 7 margins.

The figure in circulation -- effective rank 3.52 of 7 -- decides whether adding
branches is a dead end or whether we simply have not built a branch that
measures something different. Several rounds asked to re-derive it and could
not: only the aggregated probability ever left the classifier. branch_margins()
changed that; this reads it.

Label-free by construction. Correlations are computed between branch margins on
the query slides, with no label ever touched, so this is a gate-1 diagnostic and
its output cannot be an argument for promotion.

  .venv/bin/python scripts/analysis/branch_redundancy.py \
      --config configs/baseline/v121_7branch_active.yaml --gpu 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pure import index_h5_files, load_official_folds, load_slide_features  # noqa: E402
from src.models.config import TrainingFreeConfig  # noqa: E402
from src.models.training_free import TrainingFreeClassifier  # noqa: E402

TASKS = ["cptac_lscc/ARID1A_mutation", "cptac_lscc/Histologic_Grade",
         "cptac_lscc/KEAP1_mutation", "cptac_luad/KRAS_mutation",
         "cptac_pda/SMAD4_mutation", "ucla_lung/progression_regression",
         "cptac_ccrcc/PBRM1_mutation"]


def effective_rank(corr: torch.Tensor) -> float:
    """exp of the entropy of the normalised eigenvalue spectrum.

    Reported alongside the spectrum itself: a single number hides whether the
    redundancy sits in one pair of branches or is spread across all of them,
    and that distinction is exactly what the open question asks about.
    """
    eig = torch.linalg.eigvalsh(corr).clamp(min=1e-12)
    p = eig / eig.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def run_task(cfg: TrainingFreeConfig, task: str, folds: int, device) -> dict | None:
    task_dir = OFFICIAL / task
    records, slide_ids, labels, fold_cols = load_official_folds(task_dir)
    h5_index = index_h5_files(FEATURES)
    slide_ids = [s for s in slide_ids if s in h5_index]
    bags = {s: load_slide_features(s, h5_index) for s in slide_ids}
    by_sid = {str(r["slide_id"]).strip(): r for r in records}
    clf = TrainingFreeClassifier(cfg)

    per_branch: dict[str, list[torch.Tensor]] = {}
    used = 0
    for k in range(min(folds, len(fold_cols))):
        fc = fold_cols[k]
        test_ids = [s for s in slide_ids if by_sid[s][fc].strip() == "test"]
        ctx_ids = [s for s in slide_ids if by_sid[s][fc].strip() != "test"]
        if len(test_ids) < 2:
            continue
        ctx_lab = torch.tensor([labels[s] for s in ctx_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            m = clf.branch_margins([bags[s] for s in ctx_ids], ctx_lab,
                                   [bags[s] for s in test_ids])
        for name, v in m.items():
            per_branch.setdefault(name, []).append(v.float())
        used += 1
        print(f"  fold {k + 1}: {used}", flush=True)

    if used == 0:
        return None
    names = sorted(per_branch)
    # Standardise within fold before pooling: fold-level scale differences would
    # otherwise show up as correlation that is not branch agreement.
    cols = []
    for name in names:
        chunks = []
        for v in per_branch[name]:
            s = v.std()
            chunks.append((v - v.mean()) / s if float(s) > 1e-8 else v - v.mean())
        cols.append(torch.cat(chunks))
    M = torch.stack(cols)
    corr = torch.corrcoef(M)
    off = corr - torch.eye(len(names))
    return {"task": task, "folds": used, "n_query": int(M.shape[1]),
            "branches": names,
            "corr": [[round(float(x), 3) for x in row] for row in corr],
            "max_abs_offdiag": round(float(off.abs().max()), 3),
            "mean_abs_offdiag": round(
                float(off.abs().sum() / (len(names) * (len(names) - 1))), 3),
            "effective_rank": round(effective_rank(corr), 3),
            "eigenvalues": [round(float(x), 3)
                            for x in torch.linalg.eigvalsh(corr).flip(0)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline/v121_7branch_active.yaml")
    ap.add_argument("--folds", type=int, default=50)
    ap.add_argument("--out", default="talks/reports/branch_redundancy.json")
    args = ap.parse_args()

    raw = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    cfg = TrainingFreeConfig.from_yaml(raw) if hasattr(TrainingFreeConfig, "from_yaml") \
        else TrainingFreeConfig(**raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"config {args.config} · device {device} · folds {args.folds}", flush=True)

    results = []
    for task in TASKS:
        print(f"=== {task}", flush=True)
        r = run_task(cfg, task, args.folds, device)
        if r:
            results.append(r)
            print(f"  eff.rank {r['effective_rank']} / {len(r['branches'])} · "
                  f"max|r| {r['max_abs_offdiag']} · mean|r| {r['mean_abs_offdiag']}",
                  flush=True)
    (ROOT / args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    if results:
        er = [r["effective_rank"] for r in results]
        print(f"\n과제 {len(results)}개 · eff.rank {min(er):.2f}~{max(er):.2f} "
              f"(평균 {sum(er) / len(er):.2f})")
    print(f"기록: {args.out}")
    return 0


# Paths come from scripts/node_env.sh, the same place every runner reads them.
OFFICIAL = Path(os.environ.get("OFFICIAL", "/NHNHOME/BASE/kimds/Data/PathoBench/official"))
FEATURES = Path(os.environ.get("FEATURES", "/NHNHOME/BASE/kimds/Data/PathoBench/features"))

if __name__ == "__main__":
    raise SystemExit(main())
