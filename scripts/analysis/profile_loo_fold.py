import os
import sys
import time
import torch
import yaml

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stream_eval import cpu_bag_mapping
from scripts.test_pathobench import index_h5_files, load_slide_features
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig


def main():
    data_root = Path(os.environ.get("ICF_DATA_ROOT", "/lustre/BASE/kimds/Data/PathoBench"))
    official_root = data_root / "official"
    features_root = data_root / "features"
    h5_index = index_h5_files(features_root)


    task_dir = official_root / "cptac_lscc/ARID1A_mutation"
    tsv = task_dir / "k=all.tsv"
    cfg = task_dir / "config.yaml"
    task_col = yaml.safe_load(cfg.read_text())["task_col"]
    with tsv.open() as fh:
        records = list(csv.DictReader(fh, delimiter="\t"))
    slide_ids = [str(r["slide_id"]).strip() for r in records if str(r["slide_id"]).strip() in h5_index]
    labels_raw = {sid: int(float(r[task_col])) for sid, r in zip(slide_ids, records) if sid in h5_index}

    bags = {sid: load_slide_features(sid, h5_index) for sid in slide_ids}
    projected = cpu_bag_mapping(bags)

    device = torch.device("cuda:0")
    fcol = "fold_0"
    context_ids = [r["slide_id"] for r in records if r[fcol] in ("train", "val") and r["slide_id"] in projected]
    test_ids = [r["slide_id"] for r in records if r[fcol] == "test" and r["slide_id"] in projected]

    ctx_bags = [projected[sid] for sid in context_ids]
    ctx_labels = torch.tensor([labels_raw[sid] for sid in context_ids], dtype=torch.long, device=device)
    qry_bags = [projected[sid] for sid in test_ids]

    config = TrainingFreeConfig(
        sketch_dim=32,
        weight_cv=1.0, weight_ct=1.0, weight_bm=1.0, weight_bd=1.0,
        weight_qa=1.0, weight_ds=1.0, weight_de=1.0, weight_sw=1.0,
        aggregation="context_loo_power",
        loo_gamma=2.0,
        loo_floor=0.50,
    )
    clf = TrainingFreeClassifier(config)

    print("Running 1 fold with detailed profiling...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        m = clf.margins(ctx_bags, ctx_labels, qry_bags)
    print(f"Total Fold Time: {time.time() - t0:.2f}s", flush=True)

if __name__ == "__main__":
    main()
