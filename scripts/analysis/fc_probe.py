"""FC probe: compute Focus-Contrast margins and merge with the official margins.

Produces a margins dump in the predictions/margins_7b format, with one extra
branch key "fc", so `scripts/analysis/branch_screen_pure.py` can screen FC against
the official branches (gate 1) before any integration into config/voting.

  python scripts/analysis/fc_probe.py \
      --config configs/baseline/v121_7branch_active.yaml \
      --official predictions/margins_7b --out predictions/fc_gate \
      --taus 0.5,1.0,2.0,4.0 --dim 32 --lambda 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_pure import index_h5_files, load_official_folds, load_slide_features  # noqa: E402
from src.models.branches.fc import fc_features  # noqa: E402
from src.models.config import TrainingFreeConfig  # noqa: E402
from src.models.training_free import TrainingFreeClassifier  # noqa: E402

OFFICIAL = Path(os.environ.get("OFFICIAL", "data/repro_labels_folds/official"))
FEATURES = Path(os.environ.get("FEATURES", "data/repro_labels_folds/features"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--taus", default="0.5,1.0,2.0,4.0")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",") if t.strip()]
    cfg = TrainingFreeConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = TrainingFreeClassifier(cfg)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tasks = [s.strip() for s in args.tasks.split(",") if s.strip()]
    if not tasks:
        tasks = [torch.load(p, weights_only=False, map_location="cpu")["task"]
                 for p in sorted((ROOT / args.official).glob("*.pt"))]
    for task in tasks:
        off_path = ROOT / args.official / f"{task.replace('/', '_')}.pt"
        off = torch.load(off_path, weights_only=False, map_location="cpu")
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
        for pf in off["per_fold"]:
            k = pf["fold"]
            fc_key = fold_cols[k]
            q_ids = [s for s in slide_ids if by_sid[s][fc_key].strip() == "test"]
            c_ids = [s for s in slide_ids if by_sid[s][fc_key].strip() != "test"]
            if len(q_ids) < 2:
                continue
            q_lab = torch.tensor([labels[s] for s in q_ids], dtype=torch.long)
            if q_lab.shape != pf["label"].shape or not torch.equal(q_lab, pf["label"]):
                raise SystemExit(f"fold {k} label mismatch vs official margins: {task}")
            c_lab = torch.tensor([labels[s] for s in c_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                basis = clf.within_slide_basis([bags[s] for s in c_ids], device=device)
                m_fc = fc_features(args.dim, taus, args.lambda_,
                                   [bags[s] for s in c_ids], c_lab,
                                   [bags[s] for s in q_ids], basis)
            bm = dict(pf["branch_margins"])
            bm["fc"] = m_fc.detach().to("cpu").float()
            per_fold.append({"fold": k, "slide_id": pf["slide_id"],
                             "label": pf["label"], "branch_margins": bm})
            print(f"  {task} fold {k + 1} ok", flush=True)

        torch.save({"task": task, "per_fold": per_fold,
                    "provenance": {"config_sha256": None, "fc": {"taus": taus,
                                   "dim": args.dim, "lambda": args.lambda_}}},
                   out / f"{task.replace('/', '_')}.pt")
        print(f"= {task}: {len(per_fold)} folds", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
