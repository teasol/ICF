#!/usr/bin/env python
"""Which branch actually carries the discrimination? (docs SS68)

Motivation: across v36-v39 every arm lands in the same ~0.007 AUROC band, and
val AUROC moves by <=0.026 over 50 epochs while CE falls by 0.08-0.13. That
pattern says training is fitting logit SCALE (CE is scale-sensitive, AUROC is
not) rather than adding evidence. This script tests that directly on a trained
checkpoint, without any training:

  1. SCALE AUDIT     -- the learned fusion gates. A gate pinned at its floor
                        means that branch is switched off regardless of what it
                        computes.
  2. BRANCH AUROC    -- AUROC of each individual logit term against the labels.
                        A closed-form term scoring ~the full model while the
                        learned terms score ~0.5 is the decisive evidence.
  3. CONTRIBUTION    -- each term's scaled std, i.e. how much it actually moves
                        the final logit.

Usage:
  python scripts/diagnose_branch_contributions.py \
      --checkpoint <ckpt> --config <the arm's own training config> \
      [--episodes 200]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.synthetic_data import SyntheticManifoldGenerator  # noqa: E402
from src.modules.model_interface import ModelInterface  # noqa: E402
from src.utils.metrics import auroc as _auroc  # noqa: E402
from src.utils.utils import merge_train_config  # noqa: E402

# (auxiliary key, is the branch trained end-to-end?, human label)
TERMS = [
    ("global_shape_logits", "mixed", "G  global_shape (ridge + attn residual)"),
    ("abundance_ridge_logits", "closed", "P-2  abundance ridge"),
    ("population_attention_logits", "learned", "Q-5  population attention"),
    ("population_logits", "mixed", "P    population (P-2 + Q-5)"),
    ("covariance_ridge_logits", "closed", "CV-1 covariance ridge"),
    ("covariance_relation_logits", "learned", "CV-2 covariance relation"),
    ("tail_logits", "learned", "R    rare/tail"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_train_config(args.config)
    kwargs = {**config["model"], **config["model_kwargs"]}

    interface = ModelInterface.load_from_checkpoint(
        args.checkpoint, map_location="cpu", strict=True
    )
    model = interface.model.to(args.device).eval()

    # The data group carries loader-level keys too; keep only generator kwargs.
    import inspect as _inspect

    accepted = _inspect.signature(SyntheticManifoldGenerator.__init__).parameters
    data_kwargs = {
        key: value
        for key, value in dict(config["data"].get("dataset_kwargs", {})).items()
        if key in accepted
    }
    generator = SyntheticManifoldGenerator(**data_kwargs)

    print(f"checkpoint : {args.checkpoint}")
    print(f"config     : {args.config}")
    print(f"episodes   : {args.episodes}\n")

    # ---- 1. scale audit -------------------------------------------------
    meta = model.meta_classifier
    print("=== 1. learned fusion gates (a gate at its floor = branch off) ===")
    gates = {
        "attention_residual_scale (G-3)": torch.sigmoid(
            meta.global_shape_classifier.attention_residual_logit
        ),
        "population_attention_scale (Q-5)": torch.sigmoid(
            meta.population_attention_residual_logit
        ),
        "population_scale (P)": meta._floored_residual_scale(
            meta.population_residual_logit, meta.minimum_population_residual_scale
        ),
        "tail_scale (R)": meta._floored_residual_scale(
            meta.tail_residual_logit, meta.minimum_tail_residual_scale
        ),
        "fusion_scale (F-2)": torch.sigmoid(meta.fusion_residual_logit),
        "covariance_residual_scale (CV-1)": torch.sigmoid(
            meta.covariance_residual_logit
        ),
        "ridge_scale (G-2)": meta.global_shape_classifier.ridge_log_scale.exp().clamp(
            0.1, 100.0
        ),
        "abundance_ridge_scale (P-2)": meta.abundance_ridge_log_scale.exp().clamp(
            0.1, 100.0
        ),
        "covariance_ridge_scale (CV-1)": meta.covariance_ridge_log_scale.exp().clamp(
            0.1, 100.0
        ),
    }
    for name, value in gates.items():
        print(f"  {name:<36} {float(value):.4f}")
    floors = (
        f"  (floors: population {meta.minimum_population_residual_scale}, "
        f"tail {meta.minimum_tail_residual_scale})"
    )
    print(floors)

    # ---- 2/3. per-branch AUROC and contribution -------------------------
    collected: dict[str, list[float]] = {key: [] for key, _, _ in TERMS}
    collected["FINAL"] = []
    labels: list[float] = []
    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    for _ in range(args.episodes):
        episode = generator.sample_episode(generator=rng)
        x = episode.x.to(args.device)
        y = episode.y.to(args.device)
        n_bags = x.shape[0]
        query_index = torch.tensor([n_bags - 1], device=args.device)
        with torch.no_grad(), torch.autocast(
            device_type=args.device, dtype=torch.bfloat16
        ):
            logits, auxiliary = model(x, y, query_index, return_auxiliary=True)
        margin = lambda t: float((t[..., 1] - t[..., 0]).float().reshape(-1)[0])
        collected["FINAL"].append(margin(logits))
        for key, _, _ in TERMS:
            value = auxiliary.get(key)
            collected[key].append(margin(value) if value is not None else float("nan"))
        labels.append(float(y[query_index].item()))

    y_true = np.array(labels)
    print(f"\n=== 2. per-branch AUROC (n={len(y_true)}, "
          f"positives={int(y_true.sum())}) ===")
    print(f"  {'branch':<42} {'kind':>8} {'AUROC':>8} {'|scaled std|':>13}")

    def auroc(values: np.ndarray) -> float:
        finite = np.isfinite(values)
        if finite.sum() < 10 or len(set(y_true[finite])) < 2:
            return float("nan")
        return _auroc(
            torch.tensor(values[finite], dtype=torch.float64),
            torch.tensor(y_true[finite], dtype=torch.float64),
        )

    final = np.array(collected["FINAL"])
    print(f"  {'FINAL (model output)':<42} {'--':>8} {auroc(final):>8.4f} "
          f"{np.nanstd(final):>13.4f}")
    for key, kind, label in TERMS:
        values = np.array(collected[key])
        if not np.isfinite(values).any():
            continue
        print(f"  {label:<42} {kind:>8} {auroc(values):>8.4f} "
              f"{np.nanstd(values):>13.4f}")

    print(
        "\nReading: if the closed-form terms score near FINAL while the learned "
        "terms sit at ~0.5, the trained parameters are not producing the "
        "discrimination -- consistent with AUROC being flat across 50 epochs "
        "while CE falls."
    )


if __name__ == "__main__":
    main()
