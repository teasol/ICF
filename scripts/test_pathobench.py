"""Zero-shot PathoBench meta-test of a BagPFN checkpoint.

PathoBench (whole-slide histopathology MIL): each slide is a bag of tile
features extracted by a foundation model (1536-d, stored as per-slide .h5), and
each task CSV maps ``slide_id`` -> label with train/test splits. BagPFN is a
bag-level in-context meta-classifier, so for every test slide we build an
episode of labeled context slides (sampled from the train split, tiles
subsampled) plus the held-out test slide as the masked query, mirroring the
training objective.

Zero-shot bridges (no retraining):
  * 1536 -> 512 input: PCA fit on the train-split tile features (torch SVD).
  * multi-class tasks are binarized (default: class 0 vs the rest) because the
    v30 model is a binary in-context classifier.

Usage:
    python scripts/test_pathobench.py \
        --checkpoint checkpoints/20260806_145050/v33_phase0_armC_ddp8_batch2/epoch=088-val_ce_loss=0.5282.ckpt \
        --config configs/train_v33_phase0_armC_ddp8_batch2.yaml \
        --csv /NHNHOME/kimds/Data/PathoBench/csv/bracs_coarse.csv \
        --features /NHNHOME/kimds/Data/PathoBench/features \
        --output predictions/pathobench_bracs_coarse.pt
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

from src.utils.metrics import auroc, log_loss  # noqa: E402
from src.utils.utils import build_model, merge_train_config  # noqa: E402

MODEL_INPUT_DIM = 512
FEATURE_DIM = 1536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/train_v33_phase0_armC_ddp8_batch2.yaml",
    )
    parser.add_argument("--csv", type=Path, required=True, help="Task label CSV.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/NHNHOME/kimds/Data/PathoBench/features"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pathobench",
        help="Directory with preprocessed {task}_train.pt / {task}_test.pt "
        "(from scripts/prepare_pathobench.py); used when present.",
    )
    parser.add_argument("--context-per-class", type=int, default=6)
    parser.add_argument(
        "--context-mode",
        choices=("sample", "all"),
        default="sample",
        help="sample = context-per-class train slides (default); "
        "all = every train slide is context (all tiles).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_slide_features(
    slide_id: str,
    h5_index: dict[str, Path],
) -> torch.Tensor:
    """Load one slide's FULL tile features (no subsampling)."""
    import h5py

    path = h5_index.get(slide_id)
    if path is None:
        raise FileNotFoundError(f"No feature file for slide {slide_id}")
    with h5py.File(path, "r") as handle:
        features = torch.as_tensor(handle["features"][:], dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Slide {slide_id} has unexpected features shape {tuple(features.shape)}"
        )
    return features


def index_h5_files(features_root: Path) -> dict[str, Path]:
    """Map slide_id -> h5 path by scanning each dataset directory once."""
    index: dict[str, Path] = {}
    for dataset_dir in sorted(features_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for path in dataset_dir.glob("*.h5"):
            index.setdefault(path.stem, path)
    return index


def fit_pca(
    features: torch.Tensor,
    out_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA over ALL given tile features, chunked so the full data fits on GPU.

    Two exact passes over the whole tile set (mean, then centered covariance,
    both accumulated in float64), then eigendecomposition of the D x D
    covariance matrix. Returns (mean [1, dim], components [dim, out_dim]) on
    CPU.
    """
    n_total, dim = features.shape
    chunk = 2**16  # 65536 tiles per GPU block (~400MB float32 at D=1536)
    mean = torch.zeros(dim, device=device, dtype=torch.float64)
    for start in range(0, n_total, chunk):
        block = features[start : start + chunk].to(device).double()
        mean += block.sum(dim=0)
    mean /= n_total
    covariance = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    for start in range(0, n_total, chunk):
        centered = features[start : start + chunk].to(device).double() - mean
        covariance += centered.t() @ centered
    covariance /= n_total
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -out_dim:]  # top out_dim eigenvectors
    return (
        mean.float().cpu().unsqueeze(0),
        components.float().cpu().contiguous(),
    )


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    table = pd.read_csv(args.csv)
    if not {"slide_id", "label", "split"}.issubset(table.columns):
        raise ValueError(f"CSV must have slide_id/label/split columns: {list(table.columns)}")
    table = table[table["split"].isin(("train", "test"))]
    # Slide ids are string h5 stems; some CSVs have numeric ids (e.g.
    # BC_Therapy / CPTAC-CCRCC) that pandas reads as int64.
    table["slide_id"] = table["slide_id"].astype(str)
    train_table = table[table["split"] == "train"]
    test_table = table[table["split"] == "test"]
    if len(train_table) < 2 or len(test_table) < 1:
        raise ValueError("CSV needs at least 2 train and 1 test slides.")

    labels = table["label"].astype(int)
    if labels.nunique() > 2:
        # Binarize: class 0 vs the rest (v30 model is binary).
        table = table.assign(label=(labels != 0).astype(int))
        train_table = table[table["split"] == "train"]
        test_table = table[table["split"] == "test"]
        print(f"[binarized] {labels.nunique()} classes -> 0 vs rest "
              f"(n_pos {int((table['label'] == 1).sum())})")

    generator = torch.Generator().manual_seed(args.seed)

    # Load preprocessed 512-d {task}_train.pt / {task}_test.pt when available
    # (produced by scripts/prepare_pathobench.py); otherwise fall back to
    # reading h5 + fitting a train-only PCA on the GPU.
    data_dir = args.data_dir.expanduser().resolve()
    cached_train = data_dir / f"{args.csv.stem}_train.pt"
    cached_test = data_dir / f"{args.csv.stem}_test.pt"
    use_cache = cached_train.exists() and cached_test.exists()
    projected: dict[str, torch.Tensor] = {}
    if use_cache:
        train_state = torch.load(
            cached_train, map_location="cpu", weights_only=False
        )
        test_state = torch.load(cached_test, map_location="cpu", weights_only=False)
        train_ids = list(train_state["slide_id"])
        test_ids = list(test_state["slide_id"])
        for slide_id, bag in zip(train_state["slide_id"], train_state["bag"]):
            projected[slide_id] = bag.to(device)
        for slide_id, bag in zip(test_state["slide_id"], test_state["bag"]):
            projected[slide_id] = bag.to(device)
        print(f"Loaded preprocessed {args.csv.name}: {len(train_ids)} train / "
              f"{len(test_ids)} test slides (512-d, {cached_train.name})")
    else:
        # Index h5 files once, then load all needed slide features (subsampled).
        # Some CSV slide ids have no extracted feature file (feature-extraction
        # failures); drop them with a warning instead of crashing.
        h5_index = index_h5_files(args.features)
        all_ids = set(train_table["slide_id"]) | set(test_table["slide_id"])
        missing = sorted(slide for slide in all_ids if slide not in h5_index)
        if missing:
            print(f"WARNING: dropping {len(missing)} slides with no feature file "
                  f"(e.g. {missing[:3]})")
            table = table[table["slide_id"].isin(h5_index)]
            train_table = table[table["split"] == "train"]
            test_table = table[table["split"] == "test"]
            if len(train_table) < 2 or len(test_table) < 1:
                raise ValueError(
                    "Not enough slides with feature files after dropping."
                )
        slide_ids = sorted(
            set(train_table["slide_id"]) | set(test_table["slide_id"])
        )
        bags: dict[str, torch.Tensor] = {}
        for index, slide_id in enumerate(slide_ids):
            bags[slide_id] = load_slide_features(slide_id, h5_index)
            if (index + 1) % 100 == 0 or index + 1 == len(slide_ids):
                print(f"  loaded {index + 1}/{len(slide_ids)} slides", flush=True)
        train_ids = train_table["slide_id"].tolist()
        test_ids = test_table["slide_id"].tolist()
        print(f"PathoBench task {args.csv.name}: {len(train_ids)} train / "
              f"{len(test_ids)} test slides")

        # PCA bridge 1536 -> 512 fit on ALL train tiles (GPU, chunked).
        train_tiles = torch.cat([bags[s] for s in train_ids], dim=0)
        pca_mean, pca_components = fit_pca(
            train_tiles, MODEL_INPUT_DIM, device
        )
        print(f"PCA fit on all {train_tiles.shape[0]} train tiles (GPU)")

    train_labels = torch.tensor(train_table["label"].tolist(), dtype=torch.long)
    test_labels = torch.tensor(test_table["label"].tolist(), dtype=torch.long)
    train_y = {sid: int(train_table.loc[train_table["slide_id"] == sid, "label"].iloc[0]) for sid in train_ids}
    test_y = {sid: int(test_table.loc[test_table["slide_id"] == sid, "label"].iloc[0]) for sid in test_ids}

    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    if not use_cache:
        # Project every slide once on the GPU (CPU projection is ~1000x slower)
        # and cache it; per-episode CPU projection was the eval's main bottleneck.
        pca_mean_cuda = pca_mean.to(device)
        pca_components_cuda = pca_components.to(device)
        projected = {
            slide_id: (bag.to(device) - pca_mean_cuda) @ pca_components_cuda
            for slide_id, bag in bags.items()
        }
        print(f"Projected {len(projected)} slides to {MODEL_INPUT_DIM}-d on {device}")
    print(f"Model: arch v{model.model.architecture_version}, checkpoint {args.checkpoint.name}")
    if args.context_mode == "all":
        print(f"Context: ALL {len(train_ids)} train slides (all tiles)")
    else:
        print(f"Context: {args.context_per_class} slides per class (all tiles)")

    probabilities: list[float] = []
    queried_ids: list[str] = []
    nan_count = 0

    context_summary = f"{args.context_per_class} per class"
    with torch.no_grad():
        for query_index, query_id in enumerate(test_ids):
            if args.context_mode == "all":
                context_ids = list(train_ids)
                context_summary = (
                    f"all {len(context_ids)} train slides (all tiles)"
                )
                context_bags = [projected[slide] for slide in context_ids]
            else:
                context_ids = []
                for class_index in (0, 1):
                    pool = [s for s in train_ids if train_y[s] == class_index]
                    if not pool:
                        raise ValueError(
                            f"No train slides of class {class_index}."
                        )
                    permutation = torch.randperm(len(pool), generator=generator)[
                        : min(args.context_per_class, len(pool))
                    ].tolist()
                    context_ids.extend(pool[index] for index in permutation)
                context_bags = [projected[slide] for slide in context_ids]
            query_bag = projected[query_id]
            # Pad the episode into one dense batch and use the vectorized
            # batched path (the per-bag list path was CPU-bound, leaving the
            # GPU idle).
            episode_bags = [*context_bags, query_bag]
            n_bags = len(episode_bags)
            max_cells = max(bag.shape[0] for bag in episode_bags)
            dim = episode_bags[0].shape[-1]
            padded_x = episode_bags[0].new_zeros((1, n_bags, max_cells, dim))
            cell_mask = episode_bags[0].new_zeros(
                (1, n_bags, max_cells), dtype=torch.bool
            )
            for bag_index, bag in enumerate(episode_bags):
                count = bag.shape[0]
                cell_mask[0, bag_index, :count] = True
                padded_x[0, bag_index, :count] = bag
            bag_mask = torch.ones(1, n_bags, dtype=torch.bool, device=device)
            episode_y = torch.tensor(
                [[train_y[s] for s in context_ids] + [test_y[query_id]]],
                dtype=torch.long,
                device=device,
            )
            mask_index = torch.tensor([[len(context_ids)]], device=device)
            logits = model.model.forward_episode_batch(
                padded_x,
                episode_y,
                mask_index,
                return_auxiliary=False,
                cell_mask=cell_mask,
                bag_mask=bag_mask,
            )
            probability = float(
                torch.softmax(logits.float(), dim=-1)[0, 0, 1].item()
            )
            if probability != probability:
                nan_count += 1
            probabilities.append(probability)
            queried_ids.append(query_id)
            if (query_index + 1) % 20 == 0 or query_index + 1 == len(test_ids):
                print(f"  ... {query_index + 1}/{len(test_ids)} episodes", flush=True)

    probability = torch.tensor(probabilities)
    target = torch.tensor([test_y[s] for s in queried_ids], dtype=torch.long)
    if nan_count:
        print(f"WARNING: {nan_count}/{len(queried_ids)} predictions were NaN (dropped).")
    valid = torch.isfinite(probability)
    probability = probability[valid]
    target = target[valid]
    queried_ids = [s for s, v in zip(queried_ids, valid.tolist()) if v]
    if int(valid.sum()) < 2 or target.unique().numel() < 2:
        print("Not enough valid predictions / both classes to compute AUROC.")
        return

    low, high = float("nan"), float("nan")
    predicted = (probability > 0.5).long()
    accuracy = float((predicted == target).float().mean().item())
    sensitivity = float((predicted[target == 1] == 1).float().mean().item())
    specificity = float((predicted[target == 0] == 0).float().mean().item())

    print(f"\n=== PathoBench zero-shot — {args.csv.name} — {int(valid.sum())} test slides ===")
    print(f"AUROC             {auroc(probability, target):.4f}")
    print(f"Accuracy          {accuracy:.4f}")
    print(f"Balanced accuracy {(0.5 * (sensitivity + specificity)):.4f} "
          f"(sens {sensitivity:.3f} / spec {specificity:.3f})")
    print(f"Log loss          {log_loss(probability, target):.4f}")

    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "slide_id": queried_ids,
                "label": target,
                "probability": probability,
                "prediction": predicted,
                "metrics": {
                    "auroc": auroc(probability, target),
                    "accuracy": accuracy,
                    "balanced_accuracy": 0.5 * (sensitivity + specificity),
                    "log_loss": log_loss(probability, target),
                    "task": args.csv.name,
                },
            },
            args.output,
        )
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()
