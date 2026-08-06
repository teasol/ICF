"""Prepare per-task PathoBench data: 1536->512 PCA (train-only) + cached .pt.

For each task CSV's official 8:2 train/test split, load each slide's h5 tile
features, fit a 1536->512 PCA on the TRAIN tiles only (on GPU), project both
splits through it, and save ``{out_dir}/{task}_train.pt`` and
``{out_dir}/{task}_test.pt``. Eval runs (``scripts/test_pathobench.py``) then
load these cached 512-d bags directly instead of re-reading h5 and re-fitting
PCA.

Saved format (torch):
    {"slide_id": list[str], "bag": list[Tensor [n, 512]], "label": Tensor [n]}

Usage:
    python scripts/prepare_pathobench.py \
        --csv /NHNHOME/kimds/Data/PathoBench/csv/cptac_luad_tp53.csv \
        --features /NHNHOME/kimds/Data/PathoBench/features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as L
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.test_pathobench import (  # noqa: E402
    MODEL_INPUT_DIM,
    fit_pca,
    index_h5_files,
    load_slide_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Task label CSV.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/NHNHOME/kimds/Data/PathoBench/features"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pathobench",
    )
    parser.add_argument("--max-tiles", type=int, default=1024)
    parser.add_argument("--pca-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)

    table = pd.read_csv(args.csv)
    if not {"slide_id", "label", "split"}.issubset(table.columns):
        raise ValueError(
            f"CSV must have slide_id/label/split columns: {list(table.columns)}"
        )
    table = table[table["split"].isin(("train", "test"))]
    raw_labels = table["label"].astype(int)
    if raw_labels.nunique() > 2:
        # Same binarization as the eval: class 0 vs the rest.
        table = table.assign(label=(raw_labels != 0).astype(int))
        print(f"[binarized] {raw_labels.nunique()} classes -> 0 vs rest")
    train_table = table[table["split"] == "train"]
    test_table = table[table["split"] == "test"]

    h5_index = index_h5_files(args.features)
    missing = sorted(
        slide
        for slide in set(train_table["slide_id"]) | set(test_table["slide_id"])
        if slide not in h5_index
    )
    if missing:
        print(f"WARNING: dropping {len(missing)} slides with no feature file "
              f"(e.g. {missing[:3]})")
        table = table[table["slide_id"].isin(h5_index)]
        train_table = table[table["split"] == "train"]
        test_table = table[table["split"] == "test"]

    slide_ids = sorted(set(train_table["slide_id"]) | set(test_table["slide_id"]))
    bags: dict[str, torch.Tensor] = {}
    for index, slide_id in enumerate(slide_ids):
        bags[slide_id] = load_slide_features(
            slide_id, h5_index, args.max_tiles, generator
        )
        if (index + 1) % 100 == 0 or index + 1 == len(slide_ids):
            print(f"  loaded {index + 1}/{len(slide_ids)} slides", flush=True)

    # PCA fit on TRAIN tiles only (no test leakage), on the GPU.
    train_ids = train_table["slide_id"].tolist()
    train_tiles = torch.cat([bags[s] for s in train_ids], dim=0)
    if train_tiles.shape[0] > args.pca_samples:
        p = torch.randperm(train_tiles.shape[0], generator=generator)
        train_tiles = train_tiles[p[: args.pca_samples]]
    pca_mean, pca_components = fit_pca(
        train_tiles, MODEL_INPUT_DIM, generator, device
    )
    print(f"PCA fit on {train_tiles.shape[0]} TRAIN tiles -> {MODEL_INPUT_DIM}-d")

    mean_cuda = pca_mean.to(device)
    components_cuda = pca_components.to(device)

    def save_split(split_table: pd.DataFrame, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "slide_id": [],
            "bag": [],
            "label": [],
        }
        for _, row in split_table.iterrows():
            slide_id = str(row["slide_id"])
            bag = (bags[slide_id].to(device) - mean_cuda) @ components_cuda
            state["slide_id"].append(slide_id)
            state["bag"].append(bag.cpu())
            state["label"].append(int(row["label"]))
        torch.save(state, out_path)
        print(f"  saved {len(state['slide_id'])} slides -> {out_path}")

    args.out_dir = args.out_dir.expanduser().resolve()
    save_split(train_table, args.out_dir / f"{args.csv.stem}_train.pt")
    save_split(test_table, args.out_dir / f"{args.csv.stem}_test.pt")
    print(f"Done. Eval can now load these cached 512-d files.")


if __name__ == "__main__":
    main()
