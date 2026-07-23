from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.fabric.plugins.environments import LightningEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import build_datamodule, build_model, merge_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run five-fold ICI validation followed by external inference."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        help="One pretrained baseline checkpoint reused for all five folds.",
    )
    checkpoint_group.add_argument(
        "--checkpoints",
        type=Path,
        nargs=5,
        metavar=("FOLD0", "FOLD1", "FOLD2", "FOLD3", "FOLD4"),
        help="Five Lightning checkpoints in fold 0 through fold 4 order.",
    )
    checkpoint_group.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Directory containing fold0/last.ckpt through fold4/last.ckpt.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "test.yaml",
        help="ICI training config used to construct the model and data module.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "predictions" / "ici_predictions.pt",
        help="Output path; it must be inside the ICF project.",
    )
    parser.add_argument(
        "--accelerator",
        default="auto",
        help="Lightning accelerator, for example auto, gpu, or cpu.",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Run five-fold CV validation without requiring the external cohort.",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        help="Lightning inference precision (default: 16-mixed for RTX 2080 Ti).",
    )
    return parser.parse_args()


def ensure_project_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"Output must be inside {PROJECT_ROOT}: {resolved}") from error
    return resolved


def concatenate_predictions(outputs: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    if not outputs:
        raise RuntimeError("The ICI test dataloader returned no prediction batches.")
    result: dict[str, Any] = {}
    for key in ("target", "logits", "probabilities", "prediction"):
        result[key] = torch.cat([output[key].detach().cpu() for output in outputs])
    return result


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint is not None:
        checkpoints = [args.checkpoint] * 5
    elif args.checkpoints is not None:
        checkpoints = args.checkpoints
    else:
        root = args.checkpoint_root.expanduser().resolve()
        checkpoints = [root / f"fold{fold}" / "last.ckpt" for fold in range(5)]

    resolved = [checkpoint.expanduser().resolve() for checkpoint in checkpoints]
    missing = [checkpoint for checkpoint in resolved if not checkpoint.is_file()]
    if missing:
        missing_text = "\n".join(str(checkpoint) for checkpoint in missing)
        raise FileNotFoundError(f"Checkpoint files not found:\n{missing_text}")
    return resolved


def attach_donor_ids(
    result: dict[str, Any],
    dataset: Any,
) -> None:
    donor_ids = list(getattr(dataset, "unique_donors", []))
    if donor_ids and len(donor_ids) != len(result["prediction"]):
        raise RuntimeError(
            "The number of donor IDs does not match the number of predictions."
        )
    result["donor_id"] = donor_ids


def accuracy(result: dict[str, Any]) -> float:
    return (result["prediction"] == result["target"]).float().mean().item()


def binary_metrics(result: dict[str, Any]) -> dict[str, float | int]:
    """Compute robust binary metrics without an optional sklearn dependency."""
    target = result["target"].long().flatten()
    prediction = result["prediction"].long().flatten()
    probability = result["probabilities"][:, 1].float().flatten()
    if target.numel() == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set.")

    positive = target == 1
    negative = target == 0
    num_positive = int(positive.sum().item())
    num_negative = int(negative.sum().item())
    if num_positive == 0 or num_negative == 0:
        auroc = float("nan")
        balanced_accuracy = float("nan")
    else:
        positive_scores = probability[positive]
        negative_scores = probability[negative]
        comparisons = positive_scores[:, None] - negative_scores[None, :]
        auroc = float(
            ((comparisons > 0).float() + 0.5 * (comparisons == 0).float())
            .mean()
            .item()
        )
        sensitivity = (prediction[positive] == 1).float().mean()
        specificity = (prediction[negative] == 0).float().mean()
        balanced_accuracy = float((0.5 * (sensitivity + specificity)).item())

    eps = torch.finfo(probability.dtype).eps
    clipped_probability = probability.clamp(eps, 1.0 - eps)
    log_loss = float(
        -(
            target.float() * clipped_probability.log()
            + (1.0 - target.float()) * (1.0 - clipped_probability).log()
        )
        .mean()
        .item()
    )
    return {
        "accuracy": accuracy(result),
        "balanced_accuracy": balanced_accuracy,
        "auroc": auroc,
        "log_loss": log_loss,
        "num_samples": int(target.numel()),
        "num_positive": num_positive,
        "num_predicted_positive": int((prediction == 1).sum().item()),
        "probability_mean": float(probability.mean().item()),
        "probability_std": float(probability.std(unbiased=False).item()),
        "probability_min": float(probability.min().item()),
        "probability_max": float(probability.max().item()),
    }


def format_metrics(metrics: dict[str, float | int]) -> str:
    return (
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"AUROC={metrics['auroc']:.4f}, "
        f"log_loss={metrics['log_loss']:.4f}, "
        f"predicted_positive={metrics['num_predicted_positive']}/"
        f"{metrics['num_samples']}, "
        f"p1_std={metrics['probability_std']:.6f}, "
        f"p1_range=[{metrics['probability_min']:.4f}, "
        f"{metrics['probability_max']:.4f}]"
    )


def main() -> None:
    args = parse_args()
    checkpoints = resolve_checkpoints(args)
    output_path = ensure_project_path(args.output)
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        plugins=[LightningEnvironment()],
        logger=False,
        enable_checkpointing=False,
        inference_mode=True,
    )

    validation_results: list[dict[str, Any]] = []
    external_results: list[dict[str, Any]] = []
    for fold, checkpoint in enumerate(checkpoints):
        config = merge_train_config(args.config.expanduser().resolve())
        config["seed"] = args.seed
        dataset_kwargs = config["data"].setdefault("dataset_kwargs", {})
        dataset_kwargs["seed"] = args.seed
        dataset_kwargs["cv"] = fold

        datamodule = build_datamodule(config)
        model = build_model(config)
        datamodule.setup("fit")
        validation_outputs = trainer.predict(
            model=model,
            dataloaders=datamodule.val_dataloader(),
            ckpt_path=str(checkpoint),
            return_predictions=True,
        )
        validation_result = concatenate_predictions(validation_outputs)
        attach_donor_ids(validation_result, datamodule.val_dataset)
        validation_result["checkpoint"] = str(checkpoint)
        validation_result["fold"] = fold
        validation_result["metrics"] = binary_metrics(validation_result)
        validation_results.append(validation_result)
        print(
            f"Fold {fold} validation: "
            f"{format_metrics(validation_result['metrics'])}"
        )
        if args.validation_only:
            continue

        # Keep this fold's training dataset as context for external inference.
        datamodule.setup("test")
        external_outputs = trainer.predict(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=None,
            return_predictions=True,
        )
        external_result = concatenate_predictions(external_outputs)
        attach_donor_ids(external_result, datamodule.test_dataset)
        external_result["checkpoint"] = str(checkpoint)
        external_result["fold"] = fold
        external_result["metrics"] = binary_metrics(external_result)
        external_results.append(external_result)
        print(
            f"Fold {fold} external: "
            f"{format_metrics(external_result['metrics'])}"
        )

    if args.validation_only:
        validation_aggregate = concatenate_predictions(validation_results)
        validation_aggregate["donor_id"] = [
            donor_id
            for fold_result in validation_results
            for donor_id in fold_result["donor_id"]
        ]
        validation_aggregate["metrics"] = binary_metrics(validation_aggregate)
        result = {
            "seed": args.seed,
            "validation": validation_results,
            "validation_aggregate": validation_aggregate,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
        print(
            "Validation aggregate: "
            f"{format_metrics(validation_aggregate['metrics'])}"
        )
        print(f"Saved five-fold validation predictions to {output_path}")
        return

    external_donor_ids = external_results[0]["donor_id"]
    external_targets = external_results[0]["target"]
    for fold_result in external_results[1:]:
        if fold_result["donor_id"] != external_donor_ids:
            raise RuntimeError("External donor order differs between folds.")
        if not torch.equal(fold_result["target"], external_targets):
            raise RuntimeError("External targets differ between folds.")

    ensemble_probabilities = torch.stack(
        [result["probabilities"] for result in external_results]
    ).mean(dim=0)
    ensemble_result = {
        "donor_id": external_donor_ids,
        "target": external_targets,
        "probabilities": ensemble_probabilities,
        "prediction": ensemble_probabilities.argmax(dim=-1),
    }
    ensemble_result["metrics"] = binary_metrics(ensemble_result)
    validation_aggregate = concatenate_predictions(validation_results)
    validation_aggregate["donor_id"] = [
        donor_id
        for fold_result in validation_results
        for donor_id in fold_result["donor_id"]
    ]
    validation_aggregate["metrics"] = binary_metrics(validation_aggregate)
    result = {
        "seed": args.seed,
        "validation": validation_results,
        "validation_aggregate": validation_aggregate,
        "external_by_fold": external_results,
        "external_ensemble": ensemble_result,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(
        "Validation aggregate: "
        f"{format_metrics(validation_aggregate['metrics'])}"
    )
    print(
        "External ensemble: "
        f"{format_metrics(ensemble_result['metrics'])}"
    )
    print(f"Saved five-fold validation and external predictions to {output_path}")


if __name__ == "__main__":
    main()
