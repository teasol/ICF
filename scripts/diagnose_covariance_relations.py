from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import RESPONSE_TASK_NAMES
from src.utils.utils import build_datamodule, build_model, merge_train_config

EXPERIMENTS = {
    "E0": ("correlation", "reliability_mean"),
    "E2": ("correlation", "context_top1"),
    "E4": ("correlation", "context_top3"),
    "E5": ("correlation", "context_softmax"),
}
METRICS = (
    "covariance_relation_auroc",
    "covariance_relation_balanced_accuracy",
    "covariance_relation_ce",
    "covariance_relation_logit_std",
    "covariance_relation_class_separation",
    "slot_top1_stability",
    "slot_top3_jaccard",
    "slot_softmax_cosine",
)


def episode_metrics(logits: torch.Tensor, targets: torch.Tensor, separation: torch.Tensor):
    targets = targets.long()
    predictions = logits.argmax(dim=-1)
    positive = targets == 1
    negative = targets == 0
    both = bool(positive.any() and negative.any())
    if both:
        positive_recall = (predictions[positive] == 1).float().mean()
        negative_recall = (predictions[negative] == 0).float().mean()
        balanced_accuracy = (positive_recall + negative_recall) / 2
        scores = (logits[:, 1] - logits[:, 0]).float()
        pairwise = scores[positive][:, None] - scores[negative][None, :]
        auroc = (pairwise.gt(0).float() + 0.5 * pairwise.eq(0).float()).mean()
    else:
        balanced_accuracy = logits.new_zeros(())
        auroc = logits.new_zeros(())
    return {
        "covariance_relation_auroc": float(auroc),
        "covariance_relation_balanced_accuracy": float(balanced_accuracy),
        "covariance_relation_ce": float(F.cross_entropy(logits, targets)),
        "covariance_relation_logit_std": float(logits.float().std(unbiased=False)),
        "covariance_relation_class_separation": float(separation.float().mean()),
    }, both


def context_routing_stability(
    classifier, covariance: torch.Tensor, reliability: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    halves = []
    for parity in (0, 1):
        selected = []
        for class_index in range(2):
            members = torch.nonzero(labels == class_index, as_tuple=False).flatten()
            chosen = members[parity::2]
            if chosen.numel() == 0:
                chosen = members[:1]
            selected.append(chosen)
        index = torch.cat(selected).sort().values
        context = covariance[index].float()
        weights = reliability[index].float().clamp_min(classifier.covariance_relation_eps)
        subset_labels = labels[index]
        normalization = weights / weights.sum(dim=0, keepdim=True)
        center = (normalization.unsqueeze(-1) * context).sum(dim=0, keepdim=True)
        scale = torch.sqrt(
            (normalization.unsqueeze(-1) * (context - center).square())
            .sum(dim=0, keepdim=True).mean(dim=-1, keepdim=True)
            + classifier.covariance_relation_eps
        )
        normalized = (context - center) / scale
        prototypes = []
        for class_index in range(2):
            class_weight = (subset_labels == class_index).float().unsqueeze(-1) * weights
            class_weight = class_weight / class_weight.sum(dim=0, keepdim=True)
            prototypes.append((class_weight.unsqueeze(-1) * normalized).sum(dim=0))
        separation = (prototypes[1] - prototypes[0]).square().mean(dim=-1).sqrt()
        halves.append(separation)
    first, second = halves
    top1 = float(first.argmax() == second.argmax())
    count = min(3, first.numel())
    set1 = set(first.topk(count).indices.tolist())
    set2 = set(second.topk(count).indices.tolist())
    jaccard = len(set1 & set2) / len(set1 | set2)
    soft1 = torch.softmax(first / classifier.covariance_relation_routing_temperature, dim=-1)
    soft2 = torch.softmax(second / classifier.covariance_relation_routing_temperature, dim=-1)
    cosine = float(F.cosine_similarity(soft1, soft2, dim=0))
    return {
        "slot_top1_stability": top1,
        "slot_top3_jaccard": jaccard,
        "slot_softmax_cosine": cosine,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/train_covariance_relation_r1.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "logs/covariance_relation_diagnostics.csv",
    )
    args = parser.parse_args()
    config = merge_train_config(args.config)
    seed = int(config.get("seed", 42))
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    interface = build_model(config).to("cuda").eval()
    classifier = interface.model.meta_classifier
    aggregator = interface.model.aggregator

    sums = defaultdict(lambda: defaultdict(float))
    query_counts = defaultdict(int)
    valid_query_counts = defaultdict(int)
    episode_counts = defaultdict(int)

    with torch.no_grad():
        for batch in datamodule.val_dataloader():
            x, y, mask_index = batch[:3]
            x = x.to("cuda")
            y = y.to("cuda")
            mask_index = mask_index.to("cuda").long().flatten()
            task_index = None
            for metadata in batch[3:]:
                value = torch.as_tensor(metadata)
                if value.numel() == 1 and not value.is_floating_point():
                    task_index = int(value.item())
            task_name = (
                RESPONSE_TASK_NAMES[task_index]
                if task_index is not None
                else "unknown"
            )
            context_mask = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
            context_mask[mask_index] = False
            classification_x, _, centered_delta = aggregator._bag_view(x)
            anchors = aggregator._context_anchors(
                list(classification_x.unbind(0)), context_mask
            )
            similarity = torch.einsum(
                "bnd,sd->bns",
                F.normalize(classification_x.float(), dim=-1),
                anchors.float(),
            )
            assignment = torch.softmax(
                similarity / aggregator.assignment_temperature, dim=-1
            ).to(classification_x.dtype)
            descriptors = {}
            for descriptor in ("correlation", "spectral"):
                aggregator.slot_covariance_descriptor = descriptor
                descriptors[descriptor] = aggregator._slot_covariance_sketch(
                    assignment, centered_delta
                )
            context_labels = y[context_mask]
            targets = y[mask_index]
            correlation_covariance, correlation_reliability = descriptors["correlation"]
            stability = context_routing_stability(
                classifier, correlation_covariance[context_mask],
                correlation_reliability[context_mask], context_labels,
            )

            for mode, (descriptor, routing) in EXPERIMENTS.items():
                slot_covariance, slot_reliability = descriptors[descriptor]
                classifier.covariance_relation_mode = "prototype_cosine"
                classifier.covariance_relation_slot_routing = routing
                logits, separation = classifier._slot_covariance_relation_scores(
                    slot_covariance[context_mask],
                    slot_reliability[context_mask], context_labels,
                    slot_covariance[mask_index], slot_reliability[mask_index],
                )
                metrics, valid = episode_metrics(logits, targets, separation)
                metrics.update(stability)
                for group in ("all", task_name):
                    key = (mode, group)
                    for name, value in metrics.items():
                        if name in (
                            "covariance_relation_auroc",
                            "covariance_relation_balanced_accuracy",
                        ):
                            if valid:
                                sums[key][name] += value * targets.numel()
                        else:
                            sums[key][name] += value * targets.numel()
                    query_counts[key] += targets.numel()
                    if valid:
                        valid_query_counts[key] += targets.numel()
                    episode_counts[key] += 1

    rows = []
    for mode in EXPERIMENTS:
        for group in ("all", *RESPONSE_TASK_NAMES):
            key = (mode, group)
            if query_counts[key] == 0:
                continue
            row = {
                "mode": mode,
                "task": group,
                "episodes": episode_counts[key],
                "queries": query_counts[key],
                "valid_ranking_queries": valid_query_counts[key],
            }
            for name in METRICS:
                denominator = (
                    valid_query_counts[key]
                    if name in (
                        "covariance_relation_auroc",
                        "covariance_relation_balanced_accuracy",
                    )
                    else query_counts[key]
                )
                row[f"val/{name}"] = sums[key][name] / max(1, denominator)
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = [row for row in rows if row["task"] in ("all", "covariance")]
    print(json.dumps(summary, indent=2))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
