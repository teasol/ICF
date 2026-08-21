#!/usr/bin/env python
"""Diagnose the population routing distribution (v36 Q1 P0 gate, proposal SS6-2).

`_population_memory_logits` weights each bag token by
`softmax(slot_importance(token) / routing_temperature)`. Under the default
`projected` mode that softmax sits on a length-1 axis, so the weights are
identically 1.0 and the branch cannot select anything. Under `structured` it
spans the full token set -- and the risk is the opposite failure: if it collapses
onto one token, the 40->1 bottleneck is simply rebuilt through another route.

This reports the distribution, not an AUROC. Running a `structured` checkpoint-
less model (i.e. weights trained at T=1) produces meaningless logits; only the
weight distribution is informative before training.

    PY=/home/aibio_3/miniconda3/envs/BagPFN/bin/python
    $PY scripts/diagnose_population_routing.py \
      --config configs/archive/v35_v39_pre_cvonly/train_v36_q1_structured_1536.yaml \
      --checkpoint checkpoints/.../epoch=048-....ckpt \
      --official-folds /NHNHOME/BASE/kimds/Data/PathoBench/official/bc_therapy/er_status \
      --folds 2 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_pathobench import (  # noqa: E402
    index_h5_files,
    load_slide_features,
)
from src.utils.utils import (  # noqa: E402
    add_eval_precision_argument,
    build_model,
    eval_autocast,
    merge_train_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--official-folds", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/features"),
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    add_eval_precision_argument(parser)
    return parser.parse_args()


def token_type_labels(aggregator) -> list[str]:
    """`_all_structured_tokens` order: global, slots (center/spread/rare), tails."""
    labels = ["global"]
    for slot in range(aggregator.num_slots):
        kind = "density" if slot < aggregator.num_density_slots else "rare-slot"
        labels += [f"{kind}:center", f"{kind}:spread", f"{kind}:rare"]
    labels += [f"tail:{fraction}" for fraction in aggregator.tail_fractions]
    return labels


def main() -> None:
    import yaml

    args = parse_args()
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")

    task_dir = args.official_folds.expanduser().resolve()
    task_col = yaml.safe_load((task_dir / "config.yaml").read_text())["task_col"]
    records = list(csv.DictReader((task_dir / "k=all.tsv").open(), delimiter="\t"))
    fold_columns = [c for c in records[0] if c.startswith("fold_")][: args.folds]

    h5_index = index_h5_files(args.features)
    slide_ids = [
        str(r["slide_id"]).strip()
        for r in records
        if str(r["slide_id"]).strip() in h5_index
    ]
    row_of = {str(r["slide_id"]).strip(): r for r in records}
    labels = {s: int(float(row_of[s][task_col])) for s in slide_ids}
    if len(set(labels.values())) > 2:
        labels = {s: int(labels[s] != 0) for s in labels}

    config = merge_train_config(args.config.expanduser().resolve())
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    base = model.model
    aggregator = base.aggregator
    mode = base.meta_classifier.population_token_mode
    temperature = base.meta_classifier.routing_temperature
    print(
        f"mode={mode} | tokens/bag={base.meta_classifier.structured_tokens_per_bag} "
        f"| routing_temperature={temperature} | ckpt={args.checkpoint.name}"
    )

    bags = {s: load_slide_features(s, h5_index).to(device) for s in slide_ids}
    types = token_type_labels(aggregator)
    all_weights: list[torch.Tensor] = []

    with torch.no_grad(), eval_autocast(device, args.precision):
        for column in fold_columns:
            test_ids = [s for s in slide_ids if row_of[s][column].strip() == "test"]
            context_ids = [s for s in slide_ids if row_of[s][column].strip() != "test"]
            episode = [bags[s] for s in context_ids] + [bags[s] for s in test_ids]
            n_context = len(context_ids)
            is_context = torch.zeros(len(episode), dtype=torch.bool, device=device)
            is_context[:n_context] = True
            episode_y = torch.tensor(
                [labels[s] for s in context_ids] + [labels[s] for s in test_ids],
                dtype=torch.long,
                device=device,
            )
            normalized = aggregator._normalize_bags(episode)
            if aggregator.bag_representation in ("poolz", "poolz_l2"):
                pool_mean, pool_std = aggregator._context_pool_stats(
                    normalized, is_context
                )
            else:
                pool_mean = pool_std = None
            representation = aggregator(episode, context_mask=is_context)
            context_representation = {
                name: tokens[is_context] for name, tokens in representation.items()
            }
            for offset in range(len(test_ids)):
                position = n_context + offset
                _, auxiliary = base.meta_classifier(
                    context=context_representation,
                    context_labels=episode_y[is_context],
                    query={
                        name: tokens[position : position + 1]
                        for name, tokens in representation.items()
                    },
                    query_instances=[
                        aggregator._bag_view(
                            normalized[position], pool_mean, pool_std
                        )[0]
                    ],
                    return_auxiliary=True,
                )
                all_weights.append(
                    auxiliary["population_slot_weights"].float().cpu()
                )

    weights = torch.cat(all_weights, dim=0)
    n_tokens = weights.shape[-1]
    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
    uniform = math.log(n_tokens)
    print(f"\nqueries={weights.shape[0]}  tokens={n_tokens}")
    print(
        f"entropy: mean {entropy.mean():.4f} / min {entropy.min():.4f} / "
        f"max {entropy.max():.4f}   (uniform ln {n_tokens} = {uniform:.4f})"
    )
    print(
        f"effective tokens exp(H): mean {entropy.mean().exp():.2f} of {n_tokens}"
        f"   -> {100 * float(entropy.mean()) / uniform:.1f}% of uniform entropy"
    )
    print(
        f"max weight per query: mean {weights.max(dim=-1).values.mean():.4f} / "
        f"worst {weights.max(dim=-1).values.max():.4f}   (uniform = {1 / n_tokens:.4f})"
    )
    if n_tokens == len(types):
        mean_usage = weights.mean(dim=0)
        order = mean_usage.argsort(descending=True)
        print("\ntop-8 tokens by mean routing weight:")
        for rank in order[:8].tolist():
            print(f"  {types[rank]:<22} idx {rank:>3}  {mean_usage[rank]:.4f}")
        print("bottom-3:")
        for rank in order[-3:].tolist():
            print(f"  {types[rank]:<22} idx {rank:>3}  {mean_usage[rank]:.4f}")
    collapsed = float(entropy.mean()) < 0.1 * uniform
    print(
        f"\nVERDICT: {'COLLAPSED -- rebuilds the bottleneck' if collapsed else 'non-degenerate'}"
    )


if __name__ == "__main__":
    main()
