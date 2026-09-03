import os
import sys
import yaml
import csv
import time
import argparse
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stream_eval import cpu_bag_mapping
from scripts.test_pathobench import index_h5_files, load_slide_features
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig
from src.utils.metrics import auroc


TASKS = [
    ("cptac_lscc/ARID1A_mutation", "ARID1A"),
    ("cptac_lscc/Histologic_Grade", "Grade"),
    ("cptac_lscc/KEAP1_mutation", "KEAP1"),
    ("cptac_luad/KRAS_mutation", "KRAS"),
    ("cptac_pda/SMAD4_mutation", "SMAD4"),
    ("ucla_lung/progression_regression", "Progression"),
    ("cptac_ccrcc/PBRM1_mutation", "PBRM1"),
]

V120_BASELINE = {
    "ARID1A": 0.5471,
    "Grade": 0.6823,
    "KEAP1": 0.6129,
    "KRAS": 0.7295,
    "SMAD4": 0.4465,
    "Progression": 0.7986,
    "PBRM1": 0.5685,
    "Macro": 0.6265,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task-idx", type=int, default=None)
    parser.add_argument("--nfolds", type=int, default=50)
    parser.add_argument("--aggregation", type=str, default="context_loo_power")
    parser.add_argument("--loo-gamma", type=float, default=2.0)
    parser.add_argument("--loo-floor", type=float, default=0.50)
    return parser.parse_args()

def evaluate_task(task_path, task_name, args, h5_index, official_root):
    task_dir = official_root / task_path
    tsv = task_dir / "k=all.tsv"
    cfg = task_dir / "config.yaml"
    if not (tsv.exists() and cfg.exists()):
        print(f"Missing {tsv} or {cfg}")
        return None

    task_col = yaml.safe_load(cfg.read_text())["task_col"]
    header = tsv.read_text().split("\n")[0].split("\t")
    fold_cols = [c for c in header if c.startswith("fold_")]
    with tsv.open() as fh:
        records = list(csv.DictReader(fh, delimiter="\t"))
    slide_ids = [str(r["slide_id"]).strip() for r in records]
    labels_raw = {sid: int(float(r[task_col])) for sid, r in zip(slide_ids, records)}
    n_classes = len(set(labels_raw.values()))
    if n_classes > 2:
        labels_raw = {s: int(labels_raw[s] != 0) for s in labels_raw}

    slide_ids = [s for s in slide_ids if s in h5_index]
    bags = {sid: load_slide_features(sid, h5_index) for sid in slide_ids}
    projected = cpu_bag_mapping(bags)

    config = TrainingFreeConfig(
        sketch_dim=32,
        weight_cv=1.0, weight_ct=1.0, weight_bm=1.0, weight_bd=1.0,
        weight_qa=1.0, weight_ds=1.0, weight_de=1.0, weight_sw=1.0,
        aggregation=args.aggregation,
        loo_gamma=args.loo_gamma,
        loo_floor=args.loo_floor,
    )
    clf = TrainingFreeClassifier(config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    fold_aurocs = []
    t0 = time.time()
    for k in range(min(args.nfolds, len(fold_cols))):
        fcol = fold_cols[k]
        context_ids = [r["slide_id"] for r in records if r[fcol] in ("train", "val") and r["slide_id"] in projected]
        test_ids = [r["slide_id"] for r in records if r[fcol] == "test" and r["slide_id"] in projected]

        # Keep bags on CPU to avoid allocating 20GB of raw 1536D tensors on GPU
        ctx_bags = [projected[sid] for sid in context_ids]
        ctx_labels = torch.tensor([labels_raw[sid] for sid in context_ids], dtype=torch.long, device=device)
        qry_bags = [projected[sid] for sid in test_ids]
        test_labels = torch.tensor([labels_raw[sid] for sid in test_ids], dtype=torch.long)


        with torch.no_grad():
            margins = clf.margins(ctx_bags, ctx_labels, qry_bags)
            probs = torch.sigmoid(margins).cpu()

        fa = auroc(probs, test_labels)
        if fa is not None and not np.isnan(fa):
            fold_aurocs.append(fa)
        print(f"  Fold {k+1:2d}/{min(args.nfolds, len(fold_cols))}: AUROC = {fa:6.4f} (running mean: {np.mean(fold_aurocs):6.4f})", flush=True)

    t_elapsed = time.time() - t0
    mean_a = float(np.mean(fold_aurocs)) if fold_aurocs else 0.0
    diff = mean_a - V120_BASELINE.get(task_name, 0.0)
    print(f"[{task_name:<11}] 50-Fold In-Episode LOO AUROC: {mean_a:6.4f} (v120: {V120_BASELINE.get(task_name, 0.0):6.4f}, diff: {diff:+6.4f}) in {t_elapsed:.1f}s", flush=True)
    return mean_a


def main():
    args = parse_args()
    data_root = Path(os.environ.get("ICF_DATA_ROOT", "/lustre/BASE/kimds/Data/PathoBench"))
    official_root = data_root / "official"
    features_root = data_root / "features"
    h5_index = index_h5_files(features_root)


    print("=" * 80)
    print(f"Running In-Episode Context LOO Benchmark ({args.aggregation}, gamma={args.loo_gamma}, floor={args.loo_floor}) on {args.device}")
    print("=" * 80)

    if args.task_idx is not None:
        task_path, task_name = TASKS[args.task_idx]
        evaluate_task(task_path, task_name, args, h5_index, official_root)
    else:
        scores = []
        for task_path, task_name in TASKS:
            s = evaluate_task(task_path, task_name, args, h5_index, official_root)
            if s is not None:
                scores.append(s)
        if scores:
            macro = float(np.mean(scores))
            print("=" * 80)
            print(f"Primary 7 Macro AUROC: {macro:6.4f} (v120: {V120_BASELINE['Macro']:6.4f}, diff: {macro - V120_BASELINE['Macro']:+6.4f})")
            print("=" * 80)

if __name__ == "__main__":
    main()
