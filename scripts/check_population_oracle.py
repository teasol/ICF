"""Check whether true synthetic population summaries predict episode labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import SyntheticManifoldGenerator
from src.utils.utils import merge_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/train_medium.yaml")
    parser.add_argument("--episodes", type=int, default=104)
    parser.add_argument("--seed", type=int, default=50042)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--minimum-auroc", type=float, default=0.70)
    return parser.parse_args()


def oracle_logits(
    features: torch.Tensor,
    labels: torch.Tensor,
    query_index: torch.Tensor,
    ridge_lambda: float = 10.0,
) -> torch.Tensor:
    context_mask = torch.ones(len(labels), dtype=torch.bool, device=labels.device)
    context_mask[query_index] = False
    context = features[context_mask].float()
    query = features[query_index].float()
    context_labels = labels[context_mask].long()
    center = context.mean(dim=0, keepdim=True)
    scale = context.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-3)
    context = (context - center) / scale
    query = (query - center) / scale
    context = torch.cat((context, torch.ones(len(context), 1, device=context.device)), dim=-1)
    query = torch.cat((query, torch.ones(len(query), 1, device=query.device)), dim=-1)
    targets = F.one_hot(context_labels, num_classes=2).float()
    counts = torch.bincount(context_labels, minlength=2).float()
    weights = counts.reciprocal()[context_labels].sqrt().unsqueeze(-1)
    design = context * weights
    targets = targets * weights
    # Dual ridge solve scales with context donors rather than 1025 features.
    kernel = design @ design.T
    alpha = torch.linalg.solve(
        kernel + ridge_lambda * torch.eye(len(kernel), device=kernel.device),
        targets,
    )
    return query @ design.T @ alpha


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float(
        ((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean()
    )


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive.")
    config = merge_train_config(args.config.expanduser().resolve())
    generator_kwargs = dict(config["data"]["dataset_kwargs"])
    for key in (
        "episodes_per_epoch",
        "seed",
        "generation_device",
        "difficulty_curriculum_episodes",
        "effect_scale_start",
        "effect_scale_end",
    ):
        generator_kwargs.pop(key, None)
    generator = SyntheticManifoldGenerator(**generator_kwargs)
    device = torch.device(args.device)
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    task_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    task_labels: dict[str, list[torch.Tensor]] = defaultdict(list)
    for index in range(args.episodes):
        random = torch.Generator(device=device).manual_seed(args.seed + index)
        episode = generator.sample_episode(random, device=device)
        features = episode.oracle_population_features
        if features is None or episode.response_task is None:
            raise RuntimeError("Oracle features require continuous-response episodes.")
        protected = []
        for class_index in (0, 1):
            protected.append(torch.nonzero(episode.y == class_index)[0])
        can_query = torch.ones(len(episode.y), dtype=torch.bool, device=device)
        can_query[torch.stack(protected)] = False
        candidates = torch.nonzero(can_query).flatten()
        num_queries = min(20, max(1, (len(episode.y) + 4) // 5), len(candidates))
        query_index = candidates[:num_queries]
        logits = oracle_logits(features, episode.y, query_index)
        scores = (logits[:, 1] - logits[:, 0]).cpu()
        labels = episode.y[query_index].cpu()
        all_scores.append(scores)
        all_labels.append(labels)
        task_scores[episode.response_task].append(scores)
        task_labels[episode.response_task].append(labels)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    aggregate = auroc(scores, labels)
    print(f"Oracle aggregate: AUROC={aggregate:.4f}, queries={len(labels)}")
    for task in sorted(task_scores):
        task_auc = auroc(torch.cat(task_scores[task]), torch.cat(task_labels[task]))
        print(f"  {task}: AUROC={task_auc:.4f}, episodes={len(task_scores[task])}")
    if aggregate < args.minimum_auroc:
        raise RuntimeError(
            f"Oracle AUROC {aggregate:.4f} is below {args.minimum_auroc:.4f}."
        )


if __name__ == "__main__":
    main()
