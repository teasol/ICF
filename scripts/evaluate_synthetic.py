"""Evaluate a checkpoint on the synthetic validation set with honest intervals.

Synthetic data is now the decision metric: architectures are compared here,
and the ICI cohort is reserved for a single final test. That only works if the
synthetic numbers carry the same statistical discipline the ICI protocol
gained -- previously the synthetic side logged bare scalars to W&B with no
interval and no saved predictions, so two runs could not be compared properly
at all.

Interval method: queries inside one episode share a context set and the same
generative parameters, so they are *not* independent. Resampling individual
queries would treat ~1,600 correlated predictions as ~1,600 independent ones
and report an interval far too narrow. This uses a cluster bootstrap that
resamples whole episodes, which is the honest unit of replication.

Predictions are saved with their episode index and task label so
`compare_predictions.py` can run a paired cluster bootstrap between two runs.

Usage:
    python scripts/evaluate_synthetic.py \
        --checkpoint checkpoints/<ts>/v22_medium/last.ckpt \
        --config configs/archive/v22/train_v22_medium.yaml \
        --output predictions/synthetic_v22_medium.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import lightning as L
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import RESPONSE_TASK_NAMES  # noqa: E402
from src.utils.metrics import auroc, bootstrap_auroc_interval, log_loss  # noqa: E402
from src.utils.utils import build_datamodule, build_model, merge_train_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--val-episodes", type=int, default=None,
                        help="Override val_dataset_kwargs.episodes_per_epoch.")
    parser.add_argument(
        "--effect-scale", type=float, default=None,
        help="Set all three response effect scales to this value (T3-1). Per-task\n"
             "AUROCs are otherwise confounded: the generator drives composition at\n"
             "1.40, state at 0.45-1.00 and covariance at 0.30-0.80, so the usual\n"
             "per-task ranking partly reflects the data rather than the model.")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument(
        "--split",
        default="val",
        choices=("val", "test"),
        help="Synthetic split to evaluate. Keep architecture selection on val.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    if args.val_episodes is not None:
        config["data"].setdefault("val_dataset_kwargs", {})["episodes_per_epoch"] = args.val_episodes
    if args.effect_scale is not None:
        # Equal numbers are not equal difficulty -- these scales act on different
        # mechanisms -- so read the sweep as sensitivity per task, and compare
        # tasks only at a matched scale.
        for section in ("dataset_kwargs", "val_dataset_kwargs"):
            kwargs = config["data"].setdefault(section, {})
            kwargs["response_mixture_effect_scale"] = args.effect_scale
            kwargs["response_state_effect_scale"] = args.effect_scale
            kwargs["response_covariance_effect_scale"] = args.effect_scale
    datamodule = build_datamodule(config)
    model = build_model(config)

    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device(
        "cuda" if (args.accelerator != "cpu" and torch.cuda.is_available()) else "cpu"
    )
    model.to(device)

    datamodule.setup("fit" if args.split == "val" else "test")
    loader = (
        datamodule.val_dataloader() if args.split == "val" else datamodule.test_dataloader()
    )

    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    episodes: list[torch.Tensor] = []
    tasks: list[torch.Tensor] = []

    autocast = torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    with torch.no_grad(), autocast:
        for episode_index, batch in enumerate(loader):
            x, y, mask_index, _, task_index = model._unpack_evaluation_batch(batch, "val")
            x = [t.to(device) for t in x] if isinstance(x, list) else x.to(device)
            y = y.to(device)
            mask_index = mask_index.to(device)
            logits = model.model(x, y, mask_index)
            probability = torch.softmax(logits.float(), dim=-1)[:, 1].cpu()
            probabilities.append(probability)
            targets.append(y[mask_index].cpu())
            episodes.append(torch.full((probability.numel(),), episode_index, dtype=torch.long))
            tasks.append(
                torch.full(
                    (probability.numel(),),
                    -1 if task_index is None else int(task_index.item()),
                    dtype=torch.long,
                )
            )

    probability = torch.cat(probabilities)
    target = torch.cat(targets).long()
    episode = torch.cat(episodes)
    task = torch.cat(tasks)

    episode_count = int(torch.unique(episode).numel())
    overall = auroc(probability, target)
    low, high = bootstrap_auroc_interval(
        probability, target, groups=episode, samples=args.bootstrap, seed=args.seed
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(
        f"Synthetic {args.split}: {episode_count} episodes, {target.numel()} queries "
        f"({int((target == 1).sum())} positive)\n"
    )
    print(f"AUROC     {overall:.4f}  95% CI [{low:.3f}, {high:.3f}]  (cluster bootstrap over episodes)")
    print(f"Log loss  {log_loss(probability, target):.4f}")

    if int((task >= 0).sum()) > 0:
        print(f"\n{'task':<14} {'AUROC':>7} {'queries':>8} {'episodes':>9}")
        print("-" * 42)
        for index, name in enumerate(RESPONSE_TASK_NAMES):
            selected = task == index
            if selected.sum() == 0:
                continue
            print(
                f"{name:<14} {auroc(probability[selected], target[selected]):>7.4f} "
                f"{int(selected.sum()):>8} "
                f"{int(torch.unique(episode[selected]).numel()):>9}"
            )
        print(
            "\nPer-task AUROCs rest on far fewer episodes than the overall number,"
            "\nso treat them as diagnostic hints rather than decision criteria."
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "seed": args.seed,
            # Same key layout as scripts/test.py so compare_predictions.py can
            # read synthetic and ICI runs through one code path.
            "validation_aggregate": {
                "probabilities": torch.stack([1.0 - probability, probability], dim=1),
                "target": target,
                "prediction": (probability > 0.5).long(),
                "episode": episode,
                "task": task,
                "metrics": {
                    "auroc": overall,
                    "auroc_ci_low": low,
                    "auroc_ci_high": high,
                    "log_loss": log_loss(probability, target),
                    "num_samples": int(target.numel()),
                    "num_episodes": episode_count,
                },
            },
        },
        output,
    )
    print(f"\nSaved predictions to {output}")


if __name__ == "__main__":
    main()
