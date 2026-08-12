"""Measure observable and oracle upper bounds for synthetic state episodes.

Every descriptor is explicitly labelled by information access:

* model_input: tensors that the current classifier actually receives;
* observable: statistics computable from input cells but not directly exposed to
  the current classifier;
* oracle: diagnostic-only membership or latent generator state.

A class-balanced ridge probe is fitted from context bags inside each episode.
Query labels are used only for evaluation. AUROC intervals resample complete
episodes because queries from one episode are correlated.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import lightning as L
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_oracle_slot_alignment import query_index  # noqa: E402
from src.utils.metrics import auroc, bootstrap_auroc_interval, log_loss  # noqa: E402
from src.utils.utils import build_datamodule, build_model, merge_train_config  # noqa: E402

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints/20260729_160643/v22_medium_fixed/"
    / "epoch=013-val_ce_loss=0.5946.ckpt"
)

ACCESS = {
    "current_model": "model",
    "model_global_summary": "model_input",
    "model_slot_center_tokens": "model_input",
    "model_global_plus_slot_centers": "model_input",
    "observable_raw_mean": "observable",
    "observable_raw_mean_plus_spread": "observable",
    "observable_centered_direction_mean": "observable",
    "oracle_responsive_mean": "oracle_mask",
    "oracle_population_features": "oracle_mask",
    "oracle_response_score": "oracle_latent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/archive/v22/train_v22_medium.yaml")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/v22_state_upper_bound.csv")
    parser.add_argument("--val-episodes", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(config: dict[str, Any], checkpoint_path: Path, device: torch.device):
    interface = build_model(config)
    checkpoint = torch.load(checkpoint_path.expanduser().resolve(), map_location="cpu")
    interface.on_load_checkpoint(checkpoint)
    interface.load_state_dict(checkpoint["state_dict"])
    return interface.model.to(device).eval()


def responsive_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(x.dtype).unsqueeze(-1)
    count = weight.sum(dim=1).clamp_min(1.0)
    return (x * weight).sum(dim=1) / count


def main() -> None:
    args = parse_args()
    if args.val_episodes < 1:
        raise ValueError("--val-episodes must be positive.")
    config = merge_train_config(args.config.expanduser().resolve())
    seed = int(config.get("seed", 42))
    config["data"].setdefault("val_dataset_kwargs", {})[
        "episodes_per_epoch"
    ] = args.val_episodes
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    model = load_model(config, args.checkpoint, device)
    aggregator = model.aggregator
    classifier = model.meta_classifier

    scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    probabilities: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets: dict[str, list[torch.Tensor]] = defaultdict(list)
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    feature_dims: dict[str, int] = {}
    state_episode = 0

    with torch.no_grad():
        for dataset_index in range(len(datamodule.val_dataset)):
            episode = datamodule.val_dataset.diagnostic_episode(dataset_index)
            if episode.response_task != "state":
                continue
            if episode.responsive_instance_mask is None or episode.response_score is None:
                continue

            x = episode.x.to(device)
            y = episode.y.to(device).long()
            query = query_index(y)
            context = torch.ones(y.numel(), dtype=torch.bool, device=device)
            context[query] = False

            representation, _ = aggregator(x, context_mask=context, return_auxiliary=True)
            _, global_summary, _ = aggregator._bag_view(x)
            classification_x, _, _ = aggregator._bag_view(x)
            slot_centers = representation["slots"][:, :, 0, :].flatten(start_dim=1)
            raw_mean = x.float().mean(dim=1)
            responsive = responsive_mean(
                x.float(), episode.responsive_instance_mask.to(device)
            )

            descriptors = {
                "model_global_summary": representation["global_summary"].float(),
                "model_slot_center_tokens": slot_centers.float(),
                "model_global_plus_slot_centers": torch.cat(
                    (representation["global_summary"].float(), slot_centers.float()), dim=-1
                ),
                "observable_raw_mean": raw_mean,
                "observable_raw_mean_plus_spread": torch.cat(
                    (raw_mean, global_summary.float()), dim=-1
                ),
                "observable_centered_direction_mean": classification_x.float().mean(dim=1),
                "oracle_responsive_mean": responsive,
                "oracle_population_features": episode.oracle_population_features.to(device).float(),
                "oracle_response_score": episode.response_score.to(device).float().unsqueeze(-1),
            }

            candidate_logits = {"current_model": model(x, y, query).float()}
            for name, descriptor in descriptors.items():
                feature_dims[name] = descriptor.shape[-1]
                candidate_logits[name] = classifier._abundance_ridge_logits(
                    descriptor[context], y[context], descriptor[query], dual=True
                ).float()
            feature_dims["current_model"] = 0

            for name, logits in candidate_logits.items():
                probability = torch.softmax(logits, dim=-1)[:, 1].cpu()
                scores[name].append((logits[:, 1] - logits[:, 0]).cpu())
                probabilities[name].append(probability)
                targets[name].append(y[query].cpu())
                groups[name].append(
                    torch.full((query.numel(),), state_episode, dtype=torch.long)
                )
            state_episode += 1

    if state_episode == 0:
        raise RuntimeError("No state episodes were generated.")

    rows = []
    for name in ACCESS:
        candidate_scores = torch.cat(scores[name])
        candidate_probabilities = torch.cat(probabilities[name])
        candidate_targets = torch.cat(targets[name])
        candidate_groups = torch.cat(groups[name])
        point = auroc(candidate_scores, candidate_targets)
        low, high = bootstrap_auroc_interval(
            candidate_scores,
            candidate_targets,
            groups=candidate_groups,
            samples=args.bootstrap,
            seed=seed,
        )
        rows.append(
            {
                "candidate": name,
                "access": ACCESS[name],
                "feature_dim": feature_dims[name],
                "episodes": state_episode,
                "queries": candidate_targets.numel(),
                "auroc": point,
                "auroc_ci_low": low,
                "auroc_ci_high": high,
                "log_loss": log_loss(candidate_probabilities, candidate_targets),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    rows.sort(key=lambda row: -row["auroc"])
    print(f"\nstate episodes={state_episode}, queries={rows[0]['queries']}")
    print(f"{'access':<13} {'candidate':<38} {'AUROC':>7} {'95% CI':>16} {'dim':>6}")
    print("-" * 88)
    for row in rows:
        interval = f"[{row['auroc_ci_low']:.3f}, {row['auroc_ci_high']:.3f}]"
        print(
            f"{row['access']:<13} {row['candidate']:<38} "
            f"{row['auroc']:>7.4f} {interval:>16} {row['feature_dim']:>6}"
        )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
