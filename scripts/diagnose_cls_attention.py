"""Diagnose whether v26's learned CLS cross-attention selects responsive cells.

Question
--------
v26 (CLS-token pooling) finished scratch training at best val_ce_loss 0.5908,
essentially tied with the v24 baseline (0.5903). Before spending ~2 h on a
K-CLS-token + self-attention variant, we want to know *why* the single CLS
token did not move the needle. Two readings:

  (a) capacity-limited  -- one global 512-d readout is too coarse; K learned
      queries with self-attention (Perceiver-style) could extract more.
  (b) access-limited    -- the learned readout cannot identify responsive
      cells from the observed manifold at all, consistent with the closed
      cell-selection paths (T1-A/T1-C: ~0.50 AUROC, purity 0.128-0.23) and
      the split-quality audits. If so, K tokens will not help either.

This probe distinguishes (a) from (b) directly: it measures whether the
trained CLS query's cross-attention weight over cells is higher for cells the
generator actually marked responsive (`responsive_instance_mask`).

Method
------
No source code is modified. We reuse the trained v26 checkpoint and the val
episode stream (seed 50042, `diagnostic_episode`), reproduce *exactly* the
tensor the aggregator feeds to `ClassTokenPooling` in `_forward_dense`
(`_normalize_bags` + `_bag_view` + stack), then run the module's own
cross-attention with `need_weights=True` (per-head weights, averaged and
best-head reported).

Metrics (cell level, 0.5 = random; cf. T1-A cell-selection ~0.50, oracle ~0.93)
  - overall AUROC of averaged attention as a responsive-cell detector
  - per-head AUROC (specialization check: does any single head select?)
  - attention mass on responsive vs background cells (lift over base rate)
  - per-task breakdown

Usage:
    python scripts/diagnose_cls_attention.py \
        --checkpoint checkpoints/20260802_225848/v26_medium_cls_token_pool/last.ckpt \
        --config configs/train_v26_medium_cls_token_pool.yaml \
        --episodes 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning as L
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import RESPONSE_TASK_NAMES  # noqa: E402
from src.utils.metrics import auroc  # noqa: E402
from src.utils.utils import build_datamodule, build_model, merge_train_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="v26 checkpoint. Default: the v26 medium cls-token-pool last.ckpt.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Number of val episodes (seed 50042) to audit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    datamodule = build_datamodule(config)
    model = build_model(config)

    if args.checkpoint is None:
        args.checkpoint = (
            Path("checkpoints/20260802_225848/v26_medium_cls_token_pool/last.ckpt")
        )
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    device = torch.device(
        "cuda" if (args.accelerator != "cpu" and torch.cuda.is_available()) else "cpu"
    )
    model.to(device)

    datamodule.setup("fit")
    val_dataset = datamodule.val_dataset
    aggregator = model.model.aggregator
    cls_pooling = getattr(aggregator, "cls_token_pooling", None)
    if cls_pooling is None:
        raise RuntimeError(
            "Checkpoint has no cls_token_pooling -- is it a v26 model "
            "(cls_token_pooling: true)?"
        )

    # Accumulators.
    avg_scores: list[torch.Tensor] = []
    head_scores: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    task_ids: list[torch.Tensor] = []
    rows: list[dict[str, float | int | str]] = []

    evaluated = 0
    skipped = 0
    with torch.no_grad():
        for episode_index in range(args.episodes):
            episode = val_dataset.diagnostic_episode(episode_index)
            mask = episode.responsive_instance_mask
            if mask is None:
                skipped += 1
                continue
            task = (
                RESPONSE_TASK_NAMES.index(episode.response_task)
                if episode.response_task in RESPONSE_TASK_NAMES
                else -1
            )
            x = episode.x.to(device)
            # Reproduce the exact tensor `_forward_dense` feeds to the CLS pooler.
            raw_bags = aggregator._normalize_bags(x)
            prepared = [aggregator._bag_view(bag) for bag in raw_bags]
            instances = torch.stack([item[0] for item in prepared]).to(device)

            autocast = torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            )
            with autocast:
                num_bags = instances.shape[0]
                query = cls_pooling.cls_seed.unsqueeze(0).expand(num_bags, -1, -1)
                query = query.to(instances.dtype)
                normed = cls_pooling.input_norm(instances)
                _, weights = cls_pooling.cross_attention(
                    query, normed, normed,
                    need_weights=True,
                    average_attn_weights=False,
                )
            # weights: [num_bags, heads, 1, cells]
            attn_heads = weights.squeeze(2).float()  # [bags, heads, cells]
            attn_avg = attn_heads.mean(dim=1)        # [bags, cells]
            mask_f = mask.to(device).float()         # [bags, cells]

            avg_scores.append(attn_avg.flatten())
            head_scores.append(attn_heads.permute(1, 0, 2).reshape(
                attn_heads.shape[1], -1
            ))
            labels.append(mask_f.flatten())
            task_ids.append(
                torch.full((mask_f.numel(),), task, dtype=torch.long)
            )

            # Per-episode diagnostics.
            resp_mass = attn_avg[mask_f.bool()].sum().item()
            bg_mass = attn_avg[~mask_f.bool()].sum().item()
            n_resp = int(mask_f.sum().item())
            n_cells = int(mask_f.numel())
            per_bag = auroc_rows_or_nan(attn_avg, mask_f)
            rows.append(
                {
                    "episode": episode_index,
                    "task": RESPONSE_TASK_NAMES[task] if task >= 0 else "unknown",
                    "num_bags": num_bags,
                    "num_cells": n_cells,
                    "n_responsive": n_resp,
                    "responsive_fraction": n_resp / n_cells if n_cells else float("nan"),
                    "attn_mass_on_responsive": resp_mass,
                    "attn_mass_on_background": bg_mass,
                    "per_bag_auroc_mean": per_bag,
                }
            )
            evaluated += 1
            if (episode_index + 1) % 200 == 0:
                print(f"  ... {episode_index + 1}/{args.episodes} episodes", flush=True)

    if evaluated == 0:
        raise RuntimeError("No evaluated episodes with a responsive_instance_mask.")

    scores = torch.cat(avg_scores)
    labels_all = torch.cat(labels)
    tasks_all = torch.cat(task_ids)
    head_all = torch.cat(head_scores, dim=1)  # [heads, total_cells]

    overall_auroc = auroc(scores, labels_all)
    n_resp = int(labels_all.sum().item())
    n_cells = int(labels_all.numel())
    resp_mass = scores[labels_all.bool()].sum().item()
    total_mass = scores.sum().item()  # attention sums to 1 per bag
    resp_frac = n_resp / n_cells
    resp_mass_share = resp_mass / total_mass if total_mass > 0 else float("nan")

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Config:     {args.config}")
    print(
        f"Audited {evaluated} episodes (skipped {skipped} with no responsive mask), "
        f"{n_cells:,} cells, {n_resp:,} responsive "
        f"(base rate {resp_frac:.4f})\n"
    )
    print("=== CLS cross-attention as a responsive-cell detector (cell level) ===")
    print(f"Overall AUROC (averaged heads)   {overall_auroc:.4f}   (0.5 = random; "
          f"T1-A cell-selection was ~0.50)")
    head_aurocs = [
        auroc(head_all[h], labels_all) for h in range(head_all.shape[0])
    ]
    print(f"Per-head AUROC                   "
          f"{'  '.join(f'h{h}:{v:.4f}' for h, v in enumerate(head_aurocs))}")
    print(f"Best head AUROC                  {max(head_aurocs):.4f}")
    print(
        f"Attention mass share on resp.    {resp_mass_share:.4f} "
        f"(cell base rate {resp_frac:.4f}; uniform == base rate)"
    )
    lift = resp_mass_share / resp_frac if resp_frac > 0 else float("nan")
    print(f"  lift over base rate            {lift:.3f}x")

    if int((tasks_all >= 0).sum()) > 0:
        print(f"\n{'task':<14} {'AUROC':>7} {'cells':>12} {'resp_frac':>10}")
        print("-" * 48)
        for index, name in enumerate(RESPONSE_TASK_NAMES):
            selected = tasks_all == index
            if selected.sum() == 0:
                continue
            sel_resp = labels_all[selected].sum().item()
            print(
                f"{name:<14} "
                f"{auroc(scores[selected], labels_all[selected]):>7.4f} "
                f"{int(selected.sum()):>12,} "
                f"{sel_resp / int(selected.sum()):>10.4f}"
            )

    # Save per-episode rows for the record.
    output_dir = Path("logs/20260802_225848")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "v26_cls_attention_probe.csv"
    import csv

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved per-episode diagnostics to {output}")


def auroc_rows_or_nan(
    attn: torch.Tensor, mask: torch.Tensor
) -> float:
    """Mean per-bag AUROC, NaN if no bag has both classes."""
    per_bag = []
    for bag_attn, bag_mask in zip(attn, mask):
        if bool(bag_mask.sum()) > 0 and bool((~bag_mask.bool()).sum()) > 0:
            per_bag.append(auroc(bag_attn, bag_mask))
    if not per_bag:
        return float("nan")
    values = [v for v in per_bag if v == v]  # drop NaN
    if not values:
        return float("nan")
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
