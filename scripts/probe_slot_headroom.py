#!/usr/bin/env python
"""P0-slots: free slot-resolution headroom probe (no training).

Motivation (docs current_status.md SS62, architecture_v36 critique)
------------------------------------------------------------------
The v34/v35 meta-classifier compresses a bag's 40 structured tokens
(1 global_summary + num_slots x 3 + 3 tails) into ONE token with a fixed,
label-independent linear map (`_projected_bag_tokens`) BEFORE any task
information enters. `_population_memory_logits` then runs its routing softmax
over that single token, so `population_slot_weights` is identically 1.0 --
the slot-level selection mechanism is present but inert.

Two questions follow, and both can be answered WITHOUT training because the
aggregator has **zero** parameters whose shape depends on `num_slots`
(verified: 29 aggregator tensors are shape-identical at num_slots 12 vs 24;
the only shape-mismatched tensor in the whole model is
`meta_classifier.bag_token_projection.weight`):

  Q1  Does the full structured-token set carry label information that the
      projected token throws away?          -> compare `all` vs `projected`
  Q2  Does finer slot resolution carry more label information?
                                            -> sweep `--num-slots`

Method
------
For each official fold, run ONLY the aggregator once over the whole cohort
with `context_mask` marking the fold's context slides, then fit a ridge probe
on the context bags' tokens and score the held-out query bags.

This is exact with respect to the deployed per-query protocol: pool statistics
(`_context_pool_stats`) and slot anchors (`_context_anchors`) are computed from
the CONTEXT bags only, and `_bag_view`/slot statistics are per-bag, so a query
bag's representation does not depend on which other queries share the pass.
(The one documented cross-query coupling, `_covariance_relation_scores`, lives
in the meta-classifier, which this probe does not use.)  It is also ~50x
cheaper than the deployed eval: one aggregator pass per FOLD instead of one
per QUERY.

Caveat to report with any result: the slot encoders were trained at
`aggregator_num_slots=12`, so running them at 24/48 is off-distribution for
those weights. The sweep is a directional lower bound on the headroom of a
retrained model, not a guarantee.

Example
-------
    PY=/home/aibio_3/miniconda3/envs/BagPFN/bin/python
    $PY scripts/probe_slot_headroom.py \
      --config configs/train_v34_phase0_largectx_1536.yaml \
      --checkpoint checkpoints/20260807_224559/v35_largebag/epoch=048-val_ce_loss=0.3469.ckpt \
      --official-folds /NHNHOME/BASE/kimds/Data/PathoBench/official/cptac_brca/PIK3CA_mutation \
      --num-slots 8 12 24 48 --device cuda:0 \
      --output predictions/probe_slots_brca_pik3ca.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.test_pathobench import (  # noqa: E402
    FEATURE_DIM,
    index_h5_files,
    load_slide_features,
)
from src.utils.metrics import auroc  # noqa: E402
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
    parser.add_argument(
        "--official-folds",
        type=Path,
        required=True,
        help="Official task dir containing k=all.tsv + config.yaml",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/features"),
    )
    parser.add_argument(
        "--num-slots",
        type=int,
        nargs="+",
        default=[8, 12, 24, 48],
        help="Slot counts to sweep. Density slots default to round(2/3 * n).",
    )
    parser.add_argument(
        "--density-slots",
        type=int,
        nargs="+",
        default=None,
        help="Explicit density-slot count per --num-slots entry. WITHOUT this "
        "the aggregator default caps density slots at 8 (baseline.py:524), "
        "which would turn a slot increase into a rare-slot-only increase.",
    )
    parser.add_argument("--folds", type=int, default=None, help="Limit fold count")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        help="Ridge lambda grid; picked per fold by inner CV on context only.",
    )
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    add_eval_precision_argument(parser)
    return parser.parse_args()


# --------------------------------------------------------------------------
# ridge probe
# --------------------------------------------------------------------------


def _stratified_indices(
    labels: torch.Tensor, n_splits: int, seed: int
) -> list[torch.Tensor]:
    """Stratified split of sample indices into `n_splits` folds."""
    generator = torch.Generator().manual_seed(seed)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for class_index in labels.unique().tolist():
        members = (labels == class_index).nonzero(as_tuple=True)[0]
        permutation = members[torch.randperm(members.numel(), generator=generator)]
        for position, index in enumerate(permutation.tolist()):
            buckets[position % n_splits].append(index)
    return [torch.tensor(sorted(b), dtype=torch.long) for b in buckets if b]


def _solve(gram: torch.Tensor, targets: torch.Tensor, lam: float) -> torch.Tensor:
    n = gram.shape[0]
    eye = torch.eye(n, dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram + lam * eye, targets)


def ridge_probe(
    context_features: torch.Tensor,
    context_labels: torch.Tensor,
    query_features: torch.Tensor,
    lambdas: list[float],
    inner_folds: int,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Dual-form ridge probe with inner-CV lambda selection.

    Standardization uses CONTEXT statistics only. Returns query scores and the
    selected lambda.
    """
    mean = context_features.mean(dim=0, keepdim=True)
    std = context_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_context = ((context_features - mean) / std).double()
    x_query = ((query_features - mean) / std).double()

    gram = x_context @ x_context.T
    cross = x_query @ x_context.T
    targets = (context_labels.double() * 2.0) - 1.0

    best_lambda, best_score = lambdas[0], -1.0
    splits = _stratified_indices(context_labels, inner_folds, seed)
    if len(splits) >= 2:
        for lam in lambdas:
            scores: list[float] = []
            for held_out in splits:
                mask = torch.ones(
                    x_context.shape[0], dtype=torch.bool, device=gram.device
                )
                mask[held_out.to(gram.device)] = False
                train_index = mask.nonzero(as_tuple=True)[0]
                valid_index = (~mask).nonzero(as_tuple=True)[0]
                valid_labels = context_labels[valid_index.cpu()]
                if valid_labels.unique().numel() < 2:
                    continue
                alpha = _solve(
                    gram[train_index][:, train_index],
                    targets[train_index],
                    lam,
                )
                prediction = gram[valid_index][:, train_index] @ alpha
                scores.append(
                    float(auroc(prediction.float().cpu(), valid_labels))
                )
            if scores:
                mean_score = sum(scores) / len(scores)
                if mean_score > best_score:
                    best_score, best_lambda = mean_score, lam

    alpha = _solve(gram, targets, best_lambda)
    return (cross @ alpha).float().cpu(), best_lambda


# --------------------------------------------------------------------------
# model / features
# --------------------------------------------------------------------------


def build_probe_model(
    config: dict,
    checkpoint_path: Path,
    num_slots: int,
    density_slots: int,
    device: torch.device,
):
    """Build a model at `num_slots` and load the checkpoint's aggregator.

    The meta-classifier's `bag_token_projection` is shape-dependent on the
    token count, so at num_slots != the trained value it stays randomly
    initialized. That is fine: this probe only consumes the aggregator (plus
    `_all_structured_tokens`, which is a pure reshape/concat).
    """
    probe_config = {**config, "model": {**config["model"]}}
    probe_config["model"]["aggregator_num_slots"] = num_slots
    probe_config["model"]["aggregator_num_density_slots"] = density_slots
    model = build_model(probe_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    state = checkpoint["state_dict"]
    own_state = model.state_dict()
    loadable = {
        key: value
        for key, value in state.items()
        if key in own_state and own_state[key].shape == value.shape
    }
    skipped = sorted(set(state) - set(loadable))
    model.load_state_dict(loadable, strict=False)

    aggregator_prefix = "model.aggregator."
    aggregator_keys = [k for k in own_state if k.startswith(aggregator_prefix)]
    aggregator_missing = [k for k in aggregator_keys if k not in loadable]
    if aggregator_missing:
        raise RuntimeError(
            "Aggregator weights failed to load at num_slots="
            f"{num_slots}: {aggregator_missing[:5]}"
        )
    model.eval().to(device)
    return model, len(aggregator_keys), skipped


@torch.no_grad()
def fold_features(
    model,
    bags: list[torch.Tensor],
    context_mask: torch.Tensor,
    projection_trained: bool,
    precision: str = "bf16-mixed",
) -> dict[str, torch.Tensor]:
    """Run the aggregator once and return per-bag feature matrices.

    `projection_trained` is False whenever `num_slots` differs from the
    checkpoint's, because `bag_token_projection`/`bag_token_bottlenecks` are
    sized by the token count and stay randomly initialized. The variant is then
    reported as `projected_random` -- a random-projection control, NOT the
    deployed bag token.
    """
    base = model.model
    with eval_autocast(context_mask.device, precision):
        representation = base.aggregator(bags, context_mask=context_mask)
        tokens = base.meta_classifier._all_structured_tokens(representation)
    features = {"all": tokens.reshape(tokens.shape[0], -1).float().cpu()}
    if base.meta_classifier.project_structured_tokens:
        projected = base.meta_classifier._projected_bag_tokens(tokens)
        name = "projected" if projection_trained else "projected_random"
        features[name] = projected.float().cpu()
    return features


def main() -> None:
    import yaml

    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    task_dir = args.official_folds.expanduser().resolve()
    tsv, task_config = task_dir / "k=all.tsv", task_dir / "config.yaml"
    if not (tsv.exists() and task_config.exists()):
        raise FileNotFoundError(f"need k=all.tsv + config.yaml in {task_dir}")
    task_col = yaml.safe_load(task_config.read_text())["task_col"]

    header = tsv.read_text().split("\n")[0].split("\t")
    fold_cols = [c for c in header if c.startswith("fold_")]
    with tsv.open() as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    slide_ids = [str(r["slide_id"]).strip() for r in records]
    labels_raw = {sid: int(float(r[task_col])) for sid, r in zip(slide_ids, records)}
    if len(set(labels_raw.values())) > 2:
        labels_raw = {s: int(labels_raw[s] != 0) for s in labels_raw}

    h5_index = index_h5_files(args.features)
    kept = [s for s in slide_ids if s in h5_index]
    if len(kept) != len(slide_ids):
        print(f"WARNING: dropping {len(slide_ids) - len(kept)} slides with no h5")
    row_of = {sid: i for i, sid in enumerate(slide_ids)}
    slide_ids = kept

    print(f"Loading {len(slide_ids)} slides ({FEATURE_DIM}-d) ...", flush=True)
    bags = [load_slide_features(sid, h5_index).to(device) for sid in slide_ids]
    total_tiles = sum(b.shape[0] for b in bags)
    print(f"  {total_tiles} tiles total, {total_tiles / len(bags):.0f} per slide")

    config = merge_train_config(args.config.expanduser().resolve())
    checkpoint_path = args.checkpoint.expanduser().resolve()

    if args.density_slots is not None:
        if len(args.density_slots) != len(args.num_slots):
            raise ValueError("--density-slots must match --num-slots length")
        density_list = args.density_slots
    else:
        density_list = [max(1, round(2 * n / 3)) for n in args.num_slots]

    fold_scope = list(range(len(fold_cols)))
    if args.folds is not None:
        fold_scope = fold_scope[: args.folds]

    summary: dict[str, dict] = {}
    for num_slots, density_slots in zip(args.num_slots, density_list):
        model, n_aggregator, skipped = build_probe_model(
            config, checkpoint_path, num_slots, density_slots, device
        )
        tokens_per_bag = model.model.meta_classifier.structured_tokens_per_bag
        print(
            f"\n=== num_slots={num_slots} (density {density_slots} / rare "
            f"{num_slots - density_slots}) | tokens/bag {tokens_per_bag} | "
            f"aggregator tensors loaded {n_aggregator} | "
            f"ckpt tensors skipped {len(skipped)} ===",
            flush=True,
        )

        per_variant: dict[str, dict[str, list]] = {}
        variant_dims: dict[str, int] = {}
        started = time.time()
        for position, k in enumerate(fold_scope):
            column = fold_cols[k]
            is_test = [
                records[row_of[s]][column].strip() == "test" for s in slide_ids
            ]
            test_index = [i for i, flag in enumerate(is_test) if flag]
            context_index = [i for i, flag in enumerate(is_test) if not flag]
            labels = torch.tensor(
                [labels_raw[s] for s in slide_ids], dtype=torch.long
            )
            if len(test_index) < 2 or labels[test_index].unique().numel() < 2:
                continue
            if labels[context_index].unique().numel() < 2:
                continue

            context_mask = torch.zeros(len(bags), dtype=torch.bool, device=device)
            context_mask[torch.tensor(context_index, device=device)] = True
            features = fold_features(
                model,
                bags,
                context_mask,
                projection_trained=not skipped,
                precision=args.precision,
            )

            for name, matrix in features.items():
                variant_dims[name] = int(matrix.shape[1])
                scores, lam = ridge_probe(
                    matrix[context_index],
                    labels[context_index],
                    matrix[test_index],
                    args.lambdas,
                    args.inner_folds,
                    args.seed + k,
                )
                bucket = per_variant.setdefault(
                    name, {"auroc": [], "scores": [], "labels": [], "lambda": []}
                )
                bucket["auroc"].append(float(auroc(scores, labels[test_index])))
                bucket["scores"].append(scores)
                bucket["labels"].append(labels[test_index])
                bucket["lambda"].append(lam)

            if (position + 1) % 5 == 0 or position + 1 == len(fold_scope):
                elapsed = time.time() - started
                done = ", ".join(
                    f"{name} {sum(b['auroc']) / len(b['auroc']):.4f}"
                    for name, b in per_variant.items()
                )
                print(
                    f"  fold {position + 1}/{len(fold_scope)} "
                    f"({elapsed:.0f}s) running mean: {done}",
                    flush=True,
                )

        for name, bucket in per_variant.items():
            values = bucket["auroc"]
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
            pooled = float(
                auroc(torch.cat(bucket["scores"]), torch.cat(bucket["labels"]))
            )
            key = f"slots{num_slots}/{name}"
            summary[key] = {
                "num_slots": num_slots,
                "density_slots": density_slots,
                "variant": name,
                "tokens_per_bag": tokens_per_bag,
                "feature_dim": variant_dims[name],
                "n_folds": len(values),
                "fold_aurocs": values,
                "macro": mean,
                "std": std,
                "pooled": pooled,
                "lambdas": bucket["lambda"],
            }
            print(
                f"  [{key}] macro {mean:.4f} +- {std:.4f}  pooled {pooled:.4f} "
                f"({len(values)} folds)",
                flush=True,
            )
        del model
        torch.cuda.empty_cache()

    print(f"\n=== P0-slots summary — {task_dir.parent.name}/{task_dir.name} ===")
    print(f"{'config':<22}{'tokens':>8}{'dim':>9}{'macro':>10}{'std':>9}{'pooled':>10}")
    for key, row in summary.items():
        print(
            f"{key:<22}{row['tokens_per_bag']:>8}{row['feature_dim']:>9}"
            f"{row['macro']:>10.4f}{row['std']:>9.4f}{row['pooled']:>10.4f}"
        )

    if args.output is not None:
        out = args.output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": f"{task_dir.parent.name}/{task_dir.name}",
                "checkpoint": str(checkpoint_path),
                "n_slides": len(slide_ids),
                "summary": summary,
            },
            out,
        )
        print(f"\nSaved probe summary to {out}")


if __name__ == "__main__":
    main()
