"""Pure PyTorch WSI in-context evaluator (RFC 2026-09-12 Phase A-1).

Zero-parameter evaluation of `TrainingFreeClassifier` against the PathoBench
official k=all.tsv fold protocol, with NO dependency on the deep-learning
legacy stack this RFC is strangling out: `src.datasets`, `src.modules`,
`src.models.baseline`, `src.models.set_transformer_ridge`, `lightning`, or
`scripts/registry.py`-style dynamic model construction. This script imports
only:
  - `src.models.config.TrainingFreeConfig`  (YAML config parser)
  - `src.models.training_free.TrainingFreeClassifier`  (0-parameter classifier)
  - `src.models.aggregations.voting.trimmed_mean`  (used only by --compare-golden
    to reconstruct the golden aggregated probability from stored margins --
    the SAME function `TrainingFreeClassifier` calls internally, so there is
    exactly one aggregation implementation, not two)
  - `src.utils.metrics.auroc`  (the single official AUROC estimator)
  - stdlib + torch + h5py + yaml (no Lightning, no registry, no dataset modules)

H5 feature loading and official-fold parsing below are self-contained
reimplementations of the same logic `scripts/test_pathobench.py` uses --
NOT imported from it, because importing that module pulls in its top-level
`import lightning as L` and would violate the import-independence gate
(RFC gate 3: `python -c "import scripts.evaluate_pure"` must succeed with
zero legacy imports).

Precision contract (RFC item 3, "sh/sj의 fp32 강제 규격 준수"): H5 features are
loaded as float32 and `TrainingFreeClassifier.margins()` runs its ENTIRE
computation under `torch.cuda.amp.autocast(enabled=False)` (see
src/models/training_free.py) -- there is no bf16/half autocast path in this
runner for SH/SJ (or any branch) to be pulled into. `src/models/branches/sh.py`
and `sj.py` additionally force an internal float32 cast of their own
projection regardless of caller dtype, so the fp32 contract holds even if a
caller ever fed half-precision bags.

Usage:
    python scripts/evaluate_pure.py \\
        --config configs/baseline/v121_7branch_active.yaml \\
        --features /path/to/PathoBench/features \\
        --official-folds data/repro_labels_folds/official/cptac_pda/SMAD4_mutation \\
        --output predictions/pure_smad4.pt \\
        --compare-golden predictions/pathobench_cptac_pda_SMAD4_mutation_ru90_shape_triple_official50_bf16.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import TrainingFreeConfig  # noqa: E402
from src.models.training_free import TrainingFreeClassifier  # noqa: E402
from src.models.aggregations.voting import trimmed_mean  # noqa: E402
from src.utils.metrics import auroc  # noqa: E402

FEATURE_DIM = 1536


# ---- Pure H5 feature loading (self-contained, no legacy imports) -----------

def index_h5_files(features_root: Path) -> dict[str, Path]:
    """Map slide_id -> h5 path by scanning each dataset directory once."""
    index: dict[str, Path] = {}
    for dataset_dir in sorted(features_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for path in dataset_dir.glob("*.h5"):
            index.setdefault(path.stem, path)
    return index


def load_slide_features(slide_id: str, h5_index: dict[str, Path]) -> torch.Tensor:
    """Load one slide's FULL tile features (no subsampling), forced float32."""
    import h5py

    path = h5_index.get(slide_id)
    if path is None:
        raise FileNotFoundError(f"No feature file for slide {slide_id}")
    with h5py.File(path, "r") as handle:
        features = torch.as_tensor(handle["features"][:], dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Slide {slide_id} has unexpected features shape {tuple(features.shape)}"
        )
    return features


# ---- Official fold protocol (k=all.tsv + config.yaml) ----------------------

def load_official_folds(task_dir: Path) -> tuple[list[dict], list[str], dict[str, int], list[str]]:
    """Parse {task_dir}/k=all.tsv + config.yaml into (records, slide_ids,
    binary labels, fold_<k> column names), same on-disk contract PathoBench's
    official 50-fold protocol uses."""
    tsv = task_dir / "k=all.tsv"
    cfg = task_dir / "config.yaml"
    if not (tsv.exists() and cfg.exists()):
        raise FileNotFoundError(f"official task dir needs k=all.tsv+config.yaml: {task_dir}")
    task_col = yaml.safe_load(cfg.read_text())["task_col"]

    header = tsv.read_text().split("\n")[0].split("\t")
    fold_cols = [c for c in header if c.startswith("fold_")]
    with tsv.open() as fh:
        records = list(csv.DictReader(fh, delimiter="\t"))
    slide_ids = [str(r["slide_id"]).strip() for r in records]
    labels = {sid: int(float(r[task_col])) for sid, r in zip(slide_ids, records)}
    n_classes = len(set(labels.values()))
    if n_classes > 2:
        print(f"[binarized] {n_classes} classes -> 0 vs rest")
        labels = {s: int(labels[s] != 0) for s in labels}
    return records, slide_ids, labels, fold_cols


# ---- Evaluation loop ---------------------------------------------------------

def evaluate_task(
    config: TrainingFreeConfig,
    task_dir: Path,
    features_root: Path,
    device: torch.device,
    n_folds: int | None = None,
    fold_start: int = 0,
) -> list[dict]:
    """Run the official k-fold protocol with TrainingFreeClassifier. Returns a
    list of {"fold": k, "slide_id": [...], "label": Tensor, "probability": Tensor}."""
    records, slide_ids, labels, fold_cols = load_official_folds(task_dir)

    h5_index = index_h5_files(features_root)
    slide_ids = [s for s in slide_ids if s in h5_index]
    missing = len(records) - len(slide_ids)
    if missing:
        print(f"WARNING: dropping {missing} slides with no feature file")
    bags = {sid: load_slide_features(sid, h5_index) for sid in slide_ids}
    index = {sid: i for i, sid in enumerate(slide_ids)}
    print(f"Loaded {len(bags)} slides, {len(fold_cols)} official folds "
          f"({fold_cols[0]}..{fold_cols[-1]}), raw {FEATURE_DIM}-d")

    classifier = TrainingFreeClassifier(config)

    total_folds = len(fold_cols)
    n_folds = n_folds or (total_folds - fold_start)
    scope = range(fold_start, min(fold_start + n_folds, total_folds))

    per_fold: list[dict] = []
    for k in scope:
        fc = fold_cols[k]
        test_ids = [s for s in slide_ids if records[index[s]][fc].strip() == "test"]
        context_ids = [s for s in slide_ids if records[index[s]][fc].strip() != "test"]
        if len(test_ids) < 2:
            print(f"  fold {k + 1}/{total_folds}: skip (only {len(test_ids)} test slides)")
            continue

        context_bags = [bags[s].to(device) for s in context_ids]
        query_bags = [bags[s].to(device) for s in test_ids]
        context_labels = torch.tensor(
            [labels[s] for s in context_ids], dtype=torch.long, device=device
        )
        target = torch.tensor([labels[s] for s in test_ids], dtype=torch.long)

        if target.unique().numel() < 2:
            print(f"  fold {k + 1}/{total_folds}: skip (single-class query set)")
            continue

        with torch.no_grad():
            probability = classifier.predict_proba(context_bags, context_labels, query_bags)
        probability = probability.detach().to("cpu", dtype=torch.float32)

        fold_auroc = auroc(probability, target)
        per_fold.append({
            "fold": k, "slide_id": test_ids, "label": target, "probability": probability,
        })
        print(f"  fold {k + 1}/{total_folds}: AUROC {fold_auroc:.4f}  n_query {len(probability)}",
              flush=True)

    return per_fold


# ---- Golden comparison (--compare-golden) -----------------------------------

def reconstruct_golden_probability(entry: dict, config: TrainingFreeConfig) -> torch.Tensor:
    """Recompute the official trimmed_mean probability from ONE golden per-fold
    entry's stored per-branch margins (m_cv, m_bm, m_bd, m_qa, m_ds, m_sh, m_sj),
    using the SAME `trimmed_mean` this script's own classifier calls. The golden
    .pt's own `probability` field is NOT used directly: it was recorded under
    `ICF_SHAPE_SCREEN_ONLY=1` (5-branch pool only, see
    scripts/run_ru90_shape_triple.sh), while the official 7-branch number
    (`docs/PROJECT.md` SS3.2, macro 0.6227) is an OFFLINE RE-AGGREGATION of
    these same stored margins with SH/SJ folded into the pool -- exactly what
    this function does."""
    m_cv = entry["m_cv"]
    zeros = torch.zeros_like(m_cv)
    logit = trimmed_mean(
        config, zeros, m_cv,
        m_dd=None, m_ct=None,
        m_bm=entry["m_bm"], m_bd=entry["m_bd"], m_qa=entry["m_qa"], m_ds=entry["m_ds"],
        m_lr=None, m_de=None, m_sw=None,
        m_sh=entry["m_sh"], m_sj=entry["m_sj"],
    )
    return torch.sigmoid(logit)


def compare_with_golden(
    golden_path: Path,
    pure_results: list[dict],
    config: TrainingFreeConfig,
) -> tuple[float, int]:
    """max|Δp| between this script's per-slide probabilities and the golden
    reference's per-slide probabilities (reconstructed per
    `reconstruct_golden_probability`), matched by (fold index, slide_id)."""
    if not golden_path.exists():
        raise FileNotFoundError(f"golden reference not found: {golden_path}")
    golden = torch.load(golden_path, map_location="cpu", weights_only=False)
    golden_by_fold = dict(zip(golden["fold_indices"], golden["per_fold"]))

    max_abs_diff = 0.0
    n_compared = 0
    for entry in pure_results:
        g = golden_by_fold.get(entry["fold"])
        if g is None:
            continue
        golden_probability = reconstruct_golden_probability(g, config)
        golden_map = dict(zip(g["slide_id"], golden_probability.tolist()))
        for sid, p in zip(entry["slide_id"], entry["probability"].tolist()):
            if sid not in golden_map:
                continue
            max_abs_diff = max(max_abs_diff, abs(p - golden_map[sid]))
            n_compared += 1
    return max_abs_diff, n_compared


# ---- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                         default=PROJECT_ROOT / "configs/baseline/v121_7branch_active.yaml")
    parser.add_argument("--features", type=Path,
                         default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/features"))
    parser.add_argument("--official-folds", type=Path, required=True,
                         help="Task dir with k=all.tsv + config.yaml")
    parser.add_argument("--official-nfolds", type=int, default=None)
    parser.add_argument("--official-fold-start", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--compare-golden", type=Path, default=None,
                         help="predictions/*.pt to diff against (reconstructed via "
                              "trimmed_mean over its stored per-branch margins)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    config = TrainingFreeConfig.from_yaml(args.config)
    print(f"config: aggregation={config.aggregation} "
          f"weights(cv={config.weight_cv},bm={config.weight_bm},bd={config.weight_bd},"
          f"qa={config.weight_qa},ds={config.weight_ds},sh={config.weight_sh},"
          f"sj={config.weight_sj},ct={config.weight_ct},dd={config.weight_dd})")

    per_fold = evaluate_task(
        config, args.official_folds, args.features, device,
        n_folds=args.official_nfolds, fold_start=args.official_fold_start,
    )
    if not per_fold:
        raise SystemExit("no folds evaluated -- check --features / --official-folds")

    fold_aurocs = [float(auroc(e["probability"], e["label"])) for e in per_fold]
    mean = sum(fold_aurocs) / len(fold_aurocs)
    print(f"\nper-fold AUROC: {' '.join(f'{x:.4f}' for x in fold_aurocs)}")
    print(f"fold-mean AUROC: {mean:.4f}   n_folds={len(per_fold)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "task_dir": str(args.official_folds),
            "fold_indices": [e["fold"] for e in per_fold],
            "fold_aurocs": fold_aurocs,
            "fold_auroc_mean": mean,
            "per_fold": per_fold,
        }, args.output)
        print(f"Saved pure-runner predictions to {args.output}")

    if args.compare_golden is not None:
        max_delta, n_compared = compare_with_golden(args.compare_golden, per_fold, config)
        print(f"\n[compare-golden] {args.compare_golden.name}: "
              f"max|Δp| = {max_delta:.3e} over {n_compared} slide predictions")


if __name__ == "__main__":
    main()
