"""Dump CONTEXT self-margins and QUERY margins per branch, for the TGW transfer test.

RU-103 needs both: the context-only signal (a branch's AUROC on the context set
itself, using context labels -- no query labels) and the query outcome it is
supposed to predict. dump_branch_margins.py stores only query margins.

  python scripts/analysis/dump_context_query_margins.py \
      --config configs/baseline/v121_7branch_active.yaml --out predictions/ctxq_7b
"""

from __future__ import annotations

import argparse
import hashlib
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

OFFICIAL = Path(os.environ.get("OFFICIAL", "data/repro_labels_folds/official"))
FEATURES = Path(os.environ.get("FEATURES", "data/repro_labels_folds/features"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=50)
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    cfg = TrainingFreeConfig.from_yaml(cfg_path)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = TrainingFreeClassifier(cfg)
    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()] or TASKS

    for task in tasks:
        records, slide_ids, labels, fold_cols = load_official_folds(OFFICIAL / task)
        h5 = index_h5_files(FEATURES)
        slide_ids = [s for s in slide_ids if s in h5]
        bags = {s: load_slide_features(s, h5) for s in slide_ids}
        if device.type == "cuda":
            try:
                bags = {s: v.to(device, non_blocking=True) for s, v in bags.items()}
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        by_sid = {str(r["slide_id"]).strip(): r for r in records}

        per_fold = []
        for k in range(min(args.folds, len(fold_cols))):
            fc = fold_cols[k]
            q_ids = [s for s in slide_ids if by_sid[s][fc].strip() == "test"]
            c_ids = [s for s in slide_ids if by_sid[s][fc].strip() != "test"]
            if len(q_ids) < 2:
                continue
            c_lab = torch.tensor([labels[s] for s in c_ids], dtype=torch.long, device=device)
            q_lab = torch.tensor([labels[s] for s in q_ids], dtype=torch.long)
            with torch.no_grad():
                m_ctx = clf.branch_margins([bags[s] for s in c_ids], c_lab, [bags[s] for s in c_ids])
                m_qry = clf.branch_margins([bags[s] for s in c_ids], c_lab, [bags[s] for s in q_ids])
            per_fold.append({
                "fold": k,
                "context_label": c_lab.cpu(),
                "context_margins": {kk: v.cpu() for kk, v in m_ctx.items()},
                "query_label": q_lab,
                "query_margins": {kk: v.cpu() for kk, v in m_qry.items()},
            })
            print(f"  {task} fold {k + 1}: {len(per_fold)}", flush=True)

        name = task.replace("/", "_")
        torch.save({"task": task, "per_fold": per_fold,
                    "provenance": {
                        "requested_config": args.config,
                        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16],
                        "code_sha256": None, "folds_requested": args.folds,
                    }}, out / f"{name}.pt")
        print(f"= {task}: fold {len(per_fold)}개 저장", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
