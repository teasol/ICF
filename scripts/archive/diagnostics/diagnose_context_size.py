"""Inference-only context-size curve on fixed synthetic episodes and queries.

Each episode supplies the same 10 positive and 10 negative query bags at every
context size. Contexts are class-balanced and nested: the 10-bag context is a
subset of 20, which is a subset of 40, and so on. This isolates the amount of
labelled within-episode evidence without retraining or changing query difficulty.

Sizes up to 80 match the training episode range. A 160-bag context is an
intentional out-of-distribution ceiling diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_state_upper_bound import load_model  # noqa: E402
from src.datasets.synthetic_data import RESPONSE_TASK_NAMES  # noqa: E402
from src.utils.metrics import auroc, bootstrap_auroc_interval, log_loss  # noqa: E402
from src.utils.utils import build_datamodule, merge_train_config  # noqa: E402

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints/20260731_035538/v22_hard_baseline/"
    / "epoch=044-val_ce_loss=0.6839.ckpt"
)


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(","))
    if not sizes or any(size < 2 or size % 2 for size in sizes):
        raise argparse.ArgumentTypeError("context sizes must be positive even integers")
    if tuple(sorted(set(sizes))) != sizes:
        raise argparse.ArgumentTypeError("context sizes must be unique and increasing")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_v22_hard_realworld.yaml")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--context-sizes", type=parse_sizes, default=parse_sizes("10,20,40,80,160"))
    parser.add_argument("--queries-per-class", type=int, default=10)
    parser.add_argument("--pool-bags", type=int, default=220)
    parser.add_argument("--val-episodes", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "predictions/v22_hard_context_curve")
    parser.add_argument("--summary", type=Path, default=ROOT / "logs/v22_hard_context_curve_1000ep.csv")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def split_indices(
    labels: torch.Tensor,
    queries_per_class: int,
    max_context_per_class: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None:
    """Choose fixed balanced queries and nested per-class context streams."""
    members = tuple(
        torch.nonzero(labels == class_index, as_tuple=False).flatten()
        for class_index in (0, 1)
    )
    required = queries_per_class + max_context_per_class
    if any(index.numel() < required for index in members):
        return None
    query = torch.cat(tuple(index[:queries_per_class] for index in members)).sort().values
    context = tuple(
        index[queries_per_class : queries_per_class + max_context_per_class]
        for index in members
    )
    return query, context


def main() -> None:
    args = parse_args()
    if args.queries_per_class < 1 or args.val_episodes < 1:
        raise ValueError("query and episode counts must be positive")
    max_context_per_class = max(args.context_sizes) // 2
    minimum_bags = 2 * (args.queries_per_class + max_context_per_class)
    if args.pool_bags < minimum_bags:
        raise ValueError(f"--pool-bags must be at least {minimum_bags}")

    config = merge_train_config(args.config.expanduser().resolve())
    seed = int(config.get("seed", 42))
    validation = config["data"].setdefault("val_dataset_kwargs", {})
    validation["episodes_per_epoch"] = args.val_episodes
    validation["num_bags"] = args.pool_bags
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    model = load_model(config, args.checkpoint, device)

    probabilities: dict[int, list[torch.Tensor]] = defaultdict(list)
    targets: dict[int, list[torch.Tensor]] = defaultdict(list)
    groups: dict[int, list[torch.Tensor]] = defaultdict(list)
    tasks: dict[int, list[torch.Tensor]] = defaultdict(list)
    used_episodes = 0
    skipped_episodes = 0

    with torch.no_grad():
        for dataset_index in range(len(datamodule.val_dataset)):
            episode = datamodule.val_dataset.diagnostic_episode(dataset_index)
            chosen = split_indices(
                episode.y,
                queries_per_class=args.queries_per_class,
                max_context_per_class=max_context_per_class,
            )
            if chosen is None:
                skipped_episodes += 1
                continue
            query_index, context_by_class = chosen
            task_index = RESPONSE_TASK_NAMES.index(episode.response_task)
            query_target: torch.Tensor | None = None

            for context_size in args.context_sizes:
                per_class = context_size // 2
                context_index = torch.cat(
                    (context_by_class[0][:per_class], context_by_class[1][:per_class])
                )
                selected = torch.cat((context_index, query_index)).sort().values
                is_query = torch.isin(selected, query_index)
                local_query = torch.nonzero(is_query, as_tuple=False).flatten().to(device)
                x = episode.x[selected].to(device)
                y = episode.y[selected].to(device).long()
                logits = model(x, y, local_query).float()
                probability = torch.softmax(logits, dim=-1)[:, 1].cpu()
                current_target = y[local_query].cpu()
                if query_target is None:
                    query_target = current_target
                elif not torch.equal(query_target, current_target):
                    raise RuntimeError("query order changed across context sizes")
                probabilities[context_size].append(probability)
                targets[context_size].append(current_target)
                groups[context_size].append(
                    torch.full((current_target.numel(),), used_episodes, dtype=torch.long)
                )
                tasks[context_size].append(
                    torch.full((current_target.numel(),), task_index, dtype=torch.long)
                )
            used_episodes += 1

    if used_episodes == 0:
        raise RuntimeError("No episode contained enough bags per class")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for context_size in args.context_sizes:
        probability = torch.cat(probabilities[context_size])
        target = torch.cat(targets[context_size]).long()
        group = torch.cat(groups[context_size])
        task = torch.cat(tasks[context_size])
        point = auroc(probability, target)
        low, high = bootstrap_auroc_interval(
            probability, target, groups=group, samples=args.bootstrap, seed=seed
        )
        scopes = (("all", torch.ones_like(target, dtype=torch.bool)),) + tuple(
            (name, task == index) for index, name in enumerate(RESPONSE_TASK_NAMES)
        )
        for scope, selected in scopes:
            if not bool(selected.any()):
                continue
            scope_point = auroc(probability[selected], target[selected])
            scope_low, scope_high = bootstrap_auroc_interval(
                probability[selected],
                target[selected],
                groups=group[selected],
                samples=args.bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "context_bags": context_size,
                    "scope": scope,
                    "episodes": int(torch.unique(group[selected]).numel()),
                    "queries": int(selected.sum()),
                    "auroc": scope_point,
                    "auroc_ci_low": scope_low,
                    "auroc_ci_high": scope_high,
                    "log_loss": log_loss(probability[selected], target[selected]),
                }
            )
        output = args.output_dir / f"context_{context_size}.pt"
        torch.save(
            {
                "checkpoint": str(args.checkpoint),
                "context_bags": context_size,
                "validation_aggregate": {
                    "probabilities": torch.stack((1.0 - probability, probability), dim=1),
                    "target": target,
                    "prediction": (probability > 0.5).long(),
                    "episode": group,
                    "task": task,
                    "metrics": {
                        "auroc": point,
                        "auroc_ci_low": low,
                        "auroc_ci_high": high,
                        "log_loss": log_loss(probability, target),
                        "num_samples": target.numel(),
                        "num_episodes": used_episodes,
                    },
                },
            },
            output,
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"used episodes={used_episodes}, skipped={skipped_episodes}, "
        f"queries/size={2 * args.queries_per_class * used_episodes}"
    )
    print(f"{'context':>8} {'AUROC':>7} {'95% CI':>16} {'logloss':>8}")
    print("-" * 44)
    for row in rows:
        if row["scope"] != "all":
            continue
        interval = f"[{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}]"
        print(
            f"{row['context_bags']:>8} {row['auroc']:>7.4f} "
            f"{interval:>16} {row['log_loss']:>8.4f}"
        )
    print(f"summary={args.summary}")
    print(f"predictions={args.output_dir}")


if __name__ == "__main__":
    main()
