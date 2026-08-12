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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.synthetic_data import RESPONSE_TASK_NAMES
from src.utils.utils import build_datamodule, build_model, merge_train_config

METRICS = (
    "oracle_best_slot_purity",
    "oracle_best_slot_capture",
    "oracle_fragmentation_entropy",
    "oracle_context_modal_slot_fraction",
    "oracle_query_modal_slot_agreement",
    "oracle_hard_best_slot_purity",
    "oracle_hard_best_slot_capture",
    "oracle_hard_fragmentation_entropy",
    "oracle_hard_query_modal_slot_agreement",
)


def query_index(y: torch.Tensor) -> torch.Tensor:
    protected = []
    for class_index in torch.unique(y, sorted=True):
        protected.append(torch.nonzero(y == class_index, as_tuple=False).flatten()[0])
    can_query = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
    can_query[torch.stack(protected)] = False
    candidates = torch.nonzero(can_query, as_tuple=False).flatten()
    requested = max(1, min(20, (y.numel() + 4) // 5))
    return candidates[: min(requested, candidates.numel())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/archive/v18_v19/train_covariance_relation_e0.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/oracle_slot_alignment.csv")
    args = parser.parse_args()
    config = merge_train_config(args.config)
    L.seed_everything(int(config.get("seed", 42)), workers=True)
    datamodule = build_datamodule(config)
    datamodule.setup("fit")
    model = build_model(config).model.to("cuda").eval()
    dataset = datamodule.val_dataset
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    episodes = defaultdict(int)

    with torch.no_grad():
        for index in range(len(dataset)):
            episode = dataset.diagnostic_episode(index)
            mask = episode.responsive_instance_mask
            if mask is None:
                continue
            x, y = episode.x, episode.y
            queries = query_index(y)
            context = torch.ones(y.numel(), dtype=torch.bool, device=y.device)
            context[queries] = False
            classification_x, _, _ = model.aggregator._bag_view(x)
            anchors = model.aggregator._context_anchors(
                list(classification_x.unbind(0)), context
            )
            similarity = torch.einsum(
                "bnd,sd->bns", F.normalize(classification_x.float(), dim=-1),
                anchors.float(),
            )
            assignment = torch.softmax(
                similarity / model.aggregator.assignment_temperature, dim=-1
            )
            responsive = mask.float().unsqueeze(-1)
            responsive_mass = (assignment * responsive).sum(dim=1)
            slot_mass = assignment.sum(dim=1).clamp_min(1e-8)
            purity = responsive_mass / slot_mass
            responsive_total = responsive_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            capture_distribution = responsive_mass / responsive_total
            best_slot = responsive_mass.argmax(dim=-1)
            best_purity = purity.gather(1, best_slot.unsqueeze(1)).squeeze(1)
            best_capture = capture_distribution.max(dim=-1).values
            entropy = -(
                capture_distribution.clamp_min(1e-12)
                * capture_distribution.clamp_min(1e-12).log()
            ).sum(dim=-1) / torch.log(
                torch.tensor(assignment.shape[-1], device=x.device, dtype=torch.float32)
            )
            context_slots = best_slot[context]
            modal_slot = torch.bincount(
                context_slots, minlength=assignment.shape[-1]
            ).argmax()
            context_modal_fraction = (context_slots == modal_slot).float().mean()
            query_modal_agreement = (best_slot[queries] == modal_slot).float().mean()
            hard_assignment = F.one_hot(
                similarity.argmax(dim=-1), num_classes=assignment.shape[-1]
            ).float()
            hard_responsive_mass = (hard_assignment * responsive).sum(dim=1)
            hard_slot_mass = hard_assignment.sum(dim=1).clamp_min(1e-8)
            hard_purity = hard_responsive_mass / hard_slot_mass
            hard_total = hard_responsive_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            hard_distribution = hard_responsive_mass / hard_total
            hard_best_slot = hard_responsive_mass.argmax(dim=-1)
            hard_best_purity = hard_purity.gather(
                1, hard_best_slot.unsqueeze(1)
            ).squeeze(1)
            hard_best_capture = hard_distribution.max(dim=-1).values
            hard_entropy = -(
                hard_distribution.clamp_min(1e-12)
                * hard_distribution.clamp_min(1e-12).log()
            ).sum(dim=-1) / torch.log(
                torch.tensor(assignment.shape[-1], device=x.device, dtype=torch.float32)
            )
            hard_context_slots = hard_best_slot[context]
            hard_modal_slot = torch.bincount(
                hard_context_slots, minlength=assignment.shape[-1]
            ).argmax()
            hard_query_agreement = (
                hard_best_slot[queries] == hard_modal_slot
            ).float().mean()
            values = {
                "oracle_best_slot_purity": best_purity.mean(),
                "oracle_best_slot_capture": best_capture.mean(),
                "oracle_fragmentation_entropy": entropy.mean(),
                "oracle_context_modal_slot_fraction": context_modal_fraction,
                "oracle_query_modal_slot_agreement": query_modal_agreement,
                "oracle_hard_best_slot_purity": hard_best_purity.mean(),
                "oracle_hard_best_slot_capture": hard_best_capture.mean(),
                "oracle_hard_fragmentation_entropy": hard_entropy.mean(),
                "oracle_hard_query_modal_slot_agreement": hard_query_agreement,
            }
            task = episode.response_task or "unknown"
            for group in ("all", task):
                for name, value in values.items():
                    sums[group][name] += float(value) * y.numel()
                counts[group] += y.numel()
                episodes[group] += 1

    rows = []
    for group in ("all", *RESPONSE_TASK_NAMES):
        if counts[group] == 0:
            continue
        row = {"task": group, "episodes": episodes[group], "bags": counts[group]}
        row.update({name: sums[group][name] / counts[group] for name in METRICS})
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"saved={args.output}")

if __name__ == "__main__":
    main()
