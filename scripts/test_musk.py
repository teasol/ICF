"""Zero-shot meta-test of a BagPFN checkpoint on the UCI Musk (Musk2) benchmark.

Musk is a canonical multiple-instance (MIL) benchmark: each bag is a molecule,
each instance is one conformer's 166 chemical descriptors, and a bag is
positive iff ANY conformer is a musk molecule. BagPFN is a bag-level in-context
meta-classifier, so we run it as a leave-one-out episode sweep: for each
held-out molecule, the other 101 molecules (with labels) form the labeled
context and the held-out molecule is the masked query. This mirrors the model's
training objective (predict a masked bag from labeled context bags).

Preprocessing / caveats:
  * Musk conformers are 166-dim; each instance is mapped to the model's input
    dim (config input_dim, e.g. 512 for v30 or 1536 for v34-1536). Default is
    zero-padding (first 166 dims filled, rest zero -- a crude OOD bridge);
    `--pad-mode tile` repeats the 166-d descriptor floor(input_dim/166) times
    and zero-pads only the leftover, so most input dims carry real signal.
    synthetic-trained weights carry no semantics for these chemical
    descriptors, so treat this as a distribution-shift baseline, not a tuned
    model. A learned 166->512 projection trained on Musk itself would be a
    proper follow-up.
  * `_bag_view` (v24 default) centers by bag mean and L2-normalizes each
    instance, which absorbs the raw descriptor scale (range -471..625).
  * Bags have variable instance counts (1..1044); the model accepts a
    variable-length bag sequence.

Usage:
    python scripts/test_musk.py \
        --data /NHNHOME/kimds/Data/Musk/musk.pkl \
        --checkpoint checkpoints/20260731_220100/v24_medium_bag_proj_residual/epoch=041-val_ce_loss=0.5903.ckpt \
        --config configs/train_v24_medium_bag_proj_residual.yaml
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import lightning as L
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import auroc, bootstrap_auroc_interval, log_loss  # noqa: E402
from src.utils.utils import build_model, merge_train_config  # noqa: E402

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/20260731_220100/v24_medium_bag_proj_residual/"
    / "epoch=041-val_ce_loss=0.5903.ckpt"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/train_v24_medium_bag_proj_residual.yaml"
MODEL_INPUT_DIM = 512
MUSK_FEATURE_DIM = 166


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/NHNHOME/kimds/Data/Musk/musk.pkl"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pad-mode",
        choices=("zero", "tile"),
        default="zero",
        help="How to map 166-d Musk descriptors into the model input. "
        "zero = fill the first 166 dims, zero the rest (default, crude bridge). "
        "tile = repeat the descriptor floor(input_dim/166) times and zero-pad "
        "only the remainder, so most input dims carry real descriptor signal.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Model input dim. Defaults to config['model']['input_dim'] "
        "(512 for v30; 1536 for v34-1536). Musk 166-d descriptors are "
        "zero-padded to this dim.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--preprocess",
        choices=("bag_view", "raw", "centered_no_l2"),
        default="bag_view",
        help="How to map 166-d Musk descriptors into the model input. "
        "bag_view     = model default (per-bag centering + per-cell L2). "
        "               NOTE: this DISCARDS the bag mean, which is Musk's most "
        "               informative signal (ridge ceiling 0.83 -> 0.55). "
        "raw          = keep raw descriptors; summary = bag mean. Zero-shot "
        "               config override (bag_centered_representation=False) "
        "               that preserves the bag-mean signal. "
        "centered_no_l2 = center only, keep magnitude "
        "               (bag_centered_l2_normalize=False).",
    )
    return parser.parse_args()


def load_musk(
    path: Path,
    input_dim: int,
    pad_mode: str = "zero",
) -> tuple[list[str], list[torch.Tensor], torch.Tensor]:
    """Load musk.pkl into (bag_ids, padded_instance_tensors, labels).

    ``pad_mode`` maps the 166-d descriptors into ``input_dim``:
      * ``zero``: fill the first 166 dims, zero the rest.
      * ``tile``: repeat the descriptor floor(input_dim / 166) times, then
        zero-pad only the leftover remainder.
    """
    with path.open("rb") as handle:
        records = pickle.load(handle)
    n_repeat = input_dim // MUSK_FEATURE_DIM
    remainder = input_dim % MUSK_FEATURE_DIM
    bag_ids: list[str] = []
    padded: list[torch.Tensor] = []
    labels: list[int] = []
    for record in records:
        bag_ids.append(str(record["bag_id"]))
        x = torch.as_tensor(record["X"], dtype=torch.float32)  # [n, 166]
        if x.ndim != 2 or x.shape[1] != MUSK_FEATURE_DIM:
            raise ValueError(f"Unexpected Musk instance shape: {tuple(x.shape)}")
        if pad_mode == "tile":
            blocks = [x] * n_repeat
            if remainder:
                blocks.append(
                    torch.zeros(x.shape[0], remainder, dtype=torch.float32)
                )
            padded_bag = torch.cat(blocks, dim=1)
        else:
            padded_bag = torch.zeros(x.shape[0], input_dim, dtype=torch.float32)
            padded_bag[:, :MUSK_FEATURE_DIM] = x
        padded.append(padded_bag)
        labels.append(int(record["y"]))
    return bag_ids, padded, torch.tensor(labels, dtype=torch.long)


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    input_dim = (
        args.input_dim
        if args.input_dim is not None
        else int(config["model"].get("input_dim", 512))
    )
    bag_ids, bags, labels = load_musk(
        args.data.expanduser().resolve(), input_dim, args.pad_mode
    )
    n_bags = len(bags)
    if n_bags < 3:
        raise ValueError("Musk needs at least 3 bags for a leave-one-out episode.")
    num_positive = int((labels == 1).sum().item())
    print(
        f"Musk: {n_bags} bags ({num_positive} positive / {n_bags - num_positive} "
        f"negative), instances per bag {min(b.shape[0] for b in bags)}.."
        f"{max(b.shape[0] for b in bags)}, padded 166 -> {input_dim}-d "
        f"(mode={args.pad_mode})"
    )

    # Zero-shot preprocessing overrides (no retraining). These are model __init__
    # params (not in state_dict), so overriding them at inference only changes how
    # Musk descriptors are mapped into the input; the weights are untouched.
    if args.preprocess == "raw":
        config["model"]["bag_centered_representation"] = False
        config["model"]["global_summary"] = "raw_mean"
        config["model"]["use_raw_mean_branch"] = True
    elif args.preprocess == "centered_no_l2":
        config["model"]["bag_centered_l2_normalize"] = False
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    print(f"Model: arch v{model.model.architecture_version}, checkpoint {args.checkpoint}")

    probabilities: list[float] = []
    nan_count = 0
    with torch.no_grad():
        for query in range(n_bags):
            episode_x = [bag.to(device) for bag in bags]
            episode_y = labels.to(device)
            logits = model.model(episode_x, episode_y, torch.tensor([query], device=device))
            probability = float(torch.softmax(logits.float(), dim=-1)[0, 1].item())
            if probability != probability:
                nan_count += 1
            probabilities.append(probability)
            if (query + 1) % 20 == 0:
                print(f"  ... {query + 1}/{n_bags} leave-one-out episodes", flush=True)

    probability = torch.tensor(probabilities)
    target = labels
    if nan_count:
        print(f"WARNING: {nan_count}/{n_bags} predictions were NaN (dropped).")

    valid = torch.isfinite(probability)
    probability = probability[valid]
    target = target[valid]
    low, high = bootstrap_auroc_interval(probability, target, samples=2000, seed=args.seed)
    predicted = (probability > 0.5).long()
    accuracy = float((predicted == target).float().mean().item())
    sensitivity = float((predicted[target == 1] == 1).float().mean().item())
    specificity = float((predicted[target == 0] == 0).float().mean().item())

    print(f"\n=== Zero-shot Musk (Musk2) meta-test — {int(valid.sum())} bags ===")
    print(f"AUROC             {auroc(probability, target):.4f}  "
          f"[{low:.3f}, {high:.3f}]")
    print(f"Accuracy          {accuracy:.4f}")
    print(f"Balanced accuracy {(0.5 * (sensitivity + specificity)):.4f} "
          f"(sens {sensitivity:.3f} / spec {specificity:.3f})")
    print(f"Log loss          {log_loss(probability, target):.4f}")
    print(f"predicted_pos     {int((predicted == 1).sum())}/{int(valid.sum())}")

    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bag_id": bag_ids,
                "label": target,
                "probability": probability,
                "prediction": predicted,
                "metrics": {
                    "auroc": auroc(probability, target),
                    "auroc_ci_low": low,
                    "auroc_ci_high": high,
                    "accuracy": accuracy,
                    "balanced_accuracy": 0.5 * (sensitivity + specificity),
                    "log_loss": log_loss(probability, target),
                },
            },
            args.output,
        )
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()
