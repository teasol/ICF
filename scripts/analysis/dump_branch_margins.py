"""Save per-branch margins for Primary 7, once, so aggregation candidates are free.

Every aggregation candidate -- trimmed mean, soft voting, hard gated, adaptive
trimmed, any weighting -- is a function of the per-branch margins and nothing
else. Re-running the model for each of them pays 1.94 GPU-hour to recompute
identical margins. Paying once and keeping them makes every later candidate
cost zero fold fits, which is the only way a promotion round fits the budget at
all (D-054).

Provenance is written beside the margins: requested config path and hash, the
branch list actually computed, the weights, and the code hash. D-053 happened
because artefacts could not be attributed to a configuration.

  python scripts/analysis/dump_branch_margins.py \
      --config configs/baseline/v121_7branch_active.yaml --out predictions/margins_7b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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

OFFICIAL = Path(os.environ.get("OFFICIAL", "/NHNHOME/BASE/kimds/Data/PathoBench/official"))
FEATURES = Path(os.environ.get("FEATURES", "/NHNHOME/BASE/kimds/Data/PathoBench/features"))


def code_hash() -> str:
    """Hash of the files that decide the margins, so a later diff is visible."""
    h = hashlib.sha256()
    for rel in sorted(["src/models/training_free.py", "src/models/config.py"]):
        h.update((ROOT / rel).read_bytes())
    for f in sorted((ROOT / "src/models/branches").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=50)
    args = ap.parse_args()

    cfg_path = ROOT / args.config
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg = TrainingFreeConfig.from_yaml(raw)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    provenance = {
        "requested_config": args.config,
        "config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16],
        "weights": {k: getattr(cfg, k) for k in dir(cfg) if k.startswith("weight_")},
        "aggregation": cfg.aggregation,
        "sketch_dim": cfg.sketch_dim,
        "code_sha256": code_hash(),
        "git": git_rev(),
        "folds_requested": args.folds,
    }
    print(f"config {args.config} · {provenance['config_sha256']} · "
          f"code {provenance['code_sha256']} · git {provenance['git']}", flush=True)

    clf = TrainingFreeClassifier(cfg)
    for task in TASKS:
        task_dir = OFFICIAL / task
        records, slide_ids, labels, fold_cols = load_official_folds(task_dir)
        h5 = index_h5_files(FEATURES)
        slide_ids = [s for s in slide_ids if s in h5]
        bags = {s: load_slide_features(s, h5) for s in slide_ids}
        by_sid = {str(r["slide_id"]).strip(): r for r in records}

        per_fold = []
        for k in range(min(args.folds, len(fold_cols))):
            fc = fold_cols[k]
            test_ids = [s for s in slide_ids if by_sid[s][fc].strip() == "test"]
            ctx_ids = [s for s in slide_ids if by_sid[s][fc].strip() != "test"]
            if len(test_ids) < 2:
                continue
            y = torch.tensor([labels[s] for s in test_ids], dtype=torch.long)
            if y.unique().numel() < 2:
                continue
            ctx_lab = torch.tensor([labels[s] for s in ctx_ids], dtype=torch.long,
                                   device=device)
            with torch.no_grad():
                m = clf.branch_margins([bags[s] for s in ctx_ids], ctx_lab,
                                       [bags[s] for s in test_ids])
            per_fold.append({"fold": k, "slide_id": test_ids, "label": y,
                             "branch_margins": {kk: v.cpu() for kk, v in m.items()}})
            print(f"  {task} fold {k + 1}: {len(per_fold)}", flush=True)

        name = task.replace("/", "_")
        torch.save({"task": task, "per_fold": per_fold, "provenance": provenance},
                   out_dir / f"{name}.pt")
        print(f"= {task}: fold {len(per_fold)}개 저장", flush=True)

    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"기록: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
