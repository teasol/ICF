"""Zero-shot PathoBench meta-test of a BagPFN checkpoint.

PathoBench (whole-slide histopathology MIL): each slide is a bag of tile
features extracted by a foundation model (1536-d, stored as per-slide .h5), and
each task CSV maps ``slide_id`` -> label with train/test splits. BagPFN is a
bag-level in-context meta-classifier, so for every test slide we build an
episode of labeled context slides (sampled from the train split, tiles
subsampled) plus the held-out test slide as the masked query, mirroring the
training objective.

Zero-shot bridges (no retraining):
  * 1536 -> 512 input: PCA fit on the train-split tile features (torch SVD).
  * multi-class tasks are binarized (default: class 0 vs the rest) because the
    v30 model is a binary in-context classifier.

All-context (every train slide as context, full tiles) is the default;
sample-context is deprecated. `--max-tiles` randomly subsamples each bag
(context and query) to a per-bag cap, and `--trials` repeats inference with
different seeds to average over that subsampling randomness.

Usage:
    python scripts/test_pathobench.py \
        --checkpoint checkpoints/20260806_145050/v33_phase0_armC_ddp8_batch2/epoch=125-val_ce_loss=0.5142.ckpt \
        --config configs/archive/v33/train_v33_phase0_armC_ddp8_batch2.yaml \
        --csv /NHNHOME/BASE/kimds/Data/PathoBench/csv/cptac_luad_tp53.csv \
        --features /NHNHOME/BASE/kimds/Data/PathoBench/features \
        --output predictions/pathobench_cptac_luad_tp53.pt

    # tile-limit sweep (random subsample, repeated): e.g. 2000 tiles, 5 trials
    python scripts/test_pathobench.py --checkpoint <ckpt> \
        --csv .../cptac_luad_tp53.csv --max-tiles 2000 --trials 5 \
        --output predictions/..._mt2000.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import lightning as L
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.metrics import auroc, log_loss  # noqa: E402
from src.utils.utils import (  # noqa: E402
    add_eval_precision_argument,
    build_model,
    eval_autocast,
    merge_train_config,
)

MODEL_INPUT_DIM = 512
FEATURE_DIM = 1536

# docs SS146: how many DD directions the adaptive gate kept, per episode. An
# arm that reports only AUROC cannot distinguish "rank 2 does not help" from
# "the gate never fired", so the firing rate is part of the result.
DD_RANKS_KEPT: list[int] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/archive/v33/train_v33_phase0_armC_ddp8_batch2.yaml",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Task label CSV (not needed with --official-folds).",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/NHNHOME/BASE/kimds/Data/PathoBench/features"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pathobench",
        help="Directory with preprocessed {task}_train.pt / {task}_test.pt "
        "(from scripts/prepare_pathobench.py); used when present.",
    )
    parser.add_argument("--context-per-class", type=int, default=6)
    parser.add_argument(
        "--context-mode",
        choices=("sample", "all"),
        default="all",
        help="all = every train slide is context, full tiles (default; "
        "sample-context deprecated); sample = context-per-class train slides.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Per-bag tile cap: randomly subsample each bag (context and "
        "query) to at most this many tiles. None = use all tiles.",
    )
    parser.add_argument(
        "--context-max-tiles",
        type=int,
        default=None,
        help="Per-CONTEXT-bag tile cap only; the query bag stays at --max-tiles "
        "(None = full tiles). Isolates context size from query size. "
        "Defaults to --max-tiles when unset.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Model input dim. Defaults to config['model']['input_dim'] (512). "
        "When it equals FEATURE_DIM (1536), the raw 1536-d features are used "
        "directly (no PCA bridge); the 512-d preprocessed cache is skipped.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of independent inference runs (seed base + trial). "
        ">1 to average over random tile subsampling (tile-limit sweep).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Stratified K-fold cross-validation over ALL slides (train+test "
        "union): each fold's held-out slides are queried with the other folds "
        "as context (all-context). Reports per-fold and pooled AUROC. Default "
        "None = use the CSV train/test split as-is.",
    )
    parser.add_argument(
        "--official-folds",
        type=Path,
        default=None,
        help="Path to an official Patho-Bench task dir containing k=all.tsv + "
        "config.yaml. Evaluates with the OFFICIAL fold protocol (e.g. 50 folds): "
        "for each official fold k, slides with fold_k=='test' are queried using "
        "all other slides (train+val) as all-context. Reports per-fold AUROC, "
        "fold mean+std and pooled AUROC (the protocol SEAL's macro-AUC baselines "
        "follow). Requires raw features (input_dim == FEATURE_DIM).",
    )
    parser.add_argument(
        "--official-nfolds",
        type=int,
        default=None,
        help="Number of official folds to evaluate from --official-fold-start "
        "(quick checks / parallel fold splitting). Default: all.",
    )
    parser.add_argument(
        "--official-fold-start",
        type=int,
        default=0,
        help="First official fold index to evaluate (for parallel fold "
        "splitting across worker processes).",
    )
    parser.add_argument(
        "--official-ckpt",
        type=Path,
        default=None,
        help="Per-fold incremental checkpoint path for --official-folds. Each "
        "completed fold is saved immediately; on re-run, already-done folds "
        "are skipped. Folds are static/deterministic, so results never change. "
        "Useful for resume after interruption.",
    )
    parser.add_argument(
        "--batch-queries",
        action="store_true",
        help="Mask every query slide of a fold in ONE ragged forward call "
        "instead of one call per query (context re-encoded once per fold "
        "instead of once per query -- faster). NOT bit-identical to the "
        "default: batching queries together measurably shifts each query's "
        "own logits (verified up to ~0.02 logit / ~0.005 probability from "
        "an unrelated co-batched query alone). Treat as a distinct,  "
        "unvalidated protocol; default False matches every previously "
        "reported number.",
    )
    parser.add_argument(
        "--rare-logits-zero",
        action="store_true",
        help="P0-b gate (rev.2 §4.2): force rare_logits = 0 during eval (no "
        "parameter change) to measure the rare branch's contribution to the "
        "final logits. Default False = exact existing behavior.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-cache-context",
        dest="cache_context",
        action="store_false",
        help="Disable per-fold context-representation caching (docs SS64). "
        "Caching is exact for all-context full-tile folds and is what makes "
        "the official protocol ~25x cheaper; this flag exists for A/B checks.",
    )
    parser.set_defaults(cache_context=True)
    add_eval_precision_argument(parser)
    return parser.parse_args()


def load_slide_features(
    slide_id: str,
    h5_index: dict[str, Path],
) -> torch.Tensor:
    """Load one slide's FULL tile features (no subsampling)."""
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


def index_h5_files(features_root: Path) -> dict[str, Path]:
    """Map slide_id -> h5 path by scanning each dataset directory once."""
    index: dict[str, Path] = {}
    for dataset_dir in sorted(features_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for path in dataset_dir.glob("*.h5"):
            index.setdefault(path.stem, path)
    return index


def fit_pca(
    features: torch.Tensor,
    out_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA over ALL given tile features, chunked so the full data fits on GPU.

    Two exact passes over the whole tile set (mean, then centered covariance,
    both accumulated in float64), then eigendecomposition of the D x D
    covariance matrix. Returns (mean [1, dim], components [dim, out_dim]) on
    CPU.
    """
    n_total, dim = features.shape
    chunk = 2**16  # 65536 tiles per GPU block (~400MB float32 at D=1536)
    mean = torch.zeros(dim, device=device, dtype=torch.float64)
    for start in range(0, n_total, chunk):
        block = features[start : start + chunk].to(device).double()
        mean += block.sum(dim=0)
    mean /= n_total
    covariance = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    for start in range(0, n_total, chunk):
        centered = features[start : start + chunk].to(device).double() - mean
        covariance += centered.t() @ centered
    covariance /= n_total
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    components = eigenvectors[:, -out_dim:]  # top out_dim eigenvectors
    return (
        mean.float().cpu().unsqueeze(0),
        components.float().cpu().contiguous(),
    )


def stratified_folds(
    slide_ids: list[str],
    labels: list[int],
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    """Stratified (per-class round-robin) K-fold split of slide ids."""
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {}
    for slide_id, label in zip(slide_ids, labels):
        by_label.setdefault(int(label), []).append(slide_id)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for members in by_label.values():
        rng.shuffle(members)
        for index, slide_id in enumerate(members):
            folds[index % n_folds].append(slide_id)
    return folds


def evaluate_trial(
    *,
    model,
    projected: dict[str, torch.Tensor],
    train_ids: list[str],
    test_ids: list[str],
    train_y: dict[str, int],
    test_y: dict[str, int],
    context_mode: str,
    context_per_class: int,
    max_tiles: int | None,
    context_max_tiles: int | None = None,
    seed: int,
    device: torch.device,
    batch_queries: bool = False,
    precision: str = "bf16-mixed",
    cache_context: bool = True,
) -> dict:
    """Run one all-context (or sample-context) inference pass.

    Every query slide in ``test_ids`` shares the exact same context set.
    Default (``batch_queries=False``, matches every previously reported
    number): one ragged ``forward()`` call PER query, context re-encoded
    each time.

    ``batch_queries=True`` masks all of ``test_ids`` as queries in a SINGLE
    ragged forward call, so context encoding happens once per fold instead
    of once per query -- ``model.model.forward`` already accepts an
    arbitrary number of query indices in this ragged (list-of-bags) path
    (see ``_normalize_mask_index``), and training already masks multiple
    query bags per episode (``training_targets_per_episode``).

    WARNING: this is NOT bit-identical to the default. Verified against the
    v34-1536 checkpoint and isolated in a minimal repro: the coupling is
    ``StructuredPopulationMetaClassifier._covariance_relation_scores``
    (src/models/baseline.py), which normalizes each query's margin by the
    batch-wide margin RMS along the QUERY axis:
    ``margin_rms = margin.square().mean(dim=-1, keepdim=True).sqrt()`` then
    ``bounded_margin = tanh(margin / margin_rms)``. With one query
    ``margin_rms == |margin|`` so this is the fixed tanh(+-1) scale; with
    multiple queries the shared RMS couples them (order-independent but
    content-dependent, matching the observed shift). This branch is active
    in the v34 config (``covariance_relation.enabled=true``), so batching
    otherwise-independent query bags shifts each query's own logits by
    ~0.01-0.05 probability purely from the other queries' presence/content.
    Treat ``batch_queries=True`` as a distinct, unvalidated protocol -- do
    not use it for reported numbers without first quantifying the AUROC
    impact across a full task and getting sign-off.

    Uses the ragged per-episode path (list of bags) so each bag is processed
    individually — full-tile all-context episodes (up to tens of thousands of
    cells per slide) fit in memory, whereas the padded dense batched path
    OOMs ([bags, max_cells, slots, dim] explodes). ``max_tiles`` randomly
    subsamples every bag (context and query) using the trial's own generator;
    ``context_max_tiles``, when set, caps ONLY the context bags while the
    query bag stays at ``max_tiles`` (None = full tiles).
    """
    generator = torch.Generator().manual_seed(seed)

    def cap(features: torch.Tensor, limit: int | None) -> torch.Tensor:
        if limit is None or features.shape[0] <= limit:
            return features
        perm = torch.randperm(features.shape[0], generator=generator)
        return features[perm[:limit]]

    context_limit = (
        context_max_tiles if context_max_tiles is not None else max_tiles
    )

    def subsample_bag(features: torch.Tensor) -> torch.Tensor:
        return cap(features, max_tiles)

    def subsample_context_bag(features: torch.Tensor) -> torch.Tensor:
        return cap(features, context_limit)

    def sample_context_ids() -> list[str]:
        if context_mode == "all":
            return list(train_ids)
        context_ids: list[str] = []
        for class_index in (0, 1):
            pool = [s for s in train_ids if train_y[s] == class_index]
            if not pool:
                raise ValueError(f"No train slides of class {class_index}.")
            permutation = torch.randperm(len(pool), generator=generator)[
                : min(context_per_class, len(pool))
            ].tolist()
            context_ids.extend(pool[index] for index in permutation)
        return context_ids

    probabilities: list[float] = []
    queried_ids: list[str] = []
    nan_count = 0

    autocast = eval_autocast(device, precision)

    # ---- context-representation caching (docs SS62-3 / SS64) --------------
    # Every query in an all-context fold sees the SAME context set, and the
    # aggregator's only episode-level state -- `_context_pool_stats` and
    # `_context_anchors` -- is computed from the context bags alone. So the
    # whole episode's representation can be built ONCE per fold and sliced per
    # query, instead of re-encoding all context bags for every query.
    # Measured bit-identical (||delta||_inf = 0.000e+00, SS62-3).
    #
    # Guards, both required for exactness:
    #   * context_mode == "all": otherwise each query draws its own context.
    #   * context_limit is None: with --max-tiles/--context-max-tiles the shared
    #     `generator` advances per query, so each query's context is a DIFFERENT
    #     random subsample. Caching would silently freeze one draw.
    # `batch_queries` keeps its own (unvalidated) single-call path.
    use_cache = (
        cache_context
        and not batch_queries
        and context_mode == "all"
        and context_limit is None
    )
    # The cached path reaches into BaseModel's internals (`aggregator`,
    # `meta_classifier`) to hoist the context work out of the per-query loop.
    # A model built differently -- the learned bag-token branch (docs SS75) has
    # no aggregator at all -- cannot take it. Those models get a generic path
    # that calls the public forward once per fold with EVERY query at once,
    # which needs no knowledge of what is inside.
    # ICF_FORCE_GENERIC_EVAL=1 puts a BaseModel down the generic path too. That
    # is how the path gets validated: a model with a known score must reproduce
    # it here, otherwise a low number on a NEW model cannot be told apart from a
    # bug in this code.
    generic_model = not hasattr(model.model, "aggregator") or os.environ.get(
        "ICF_FORCE_GENERIC_EVAL"
    ) == "1"
    if generic_model:
        context_ids = sample_context_ids()
        episode_bags = [
            *(subsample_context_bag(projected[s]) for s in context_ids),
            *(subsample_bag(projected[s]) for s in test_ids),
        ]
        n_context = len(context_ids)
        episode_y = torch.tensor(
            [train_y[s] for s in context_ids] + [test_y[s] for s in test_ids],
            dtype=torch.long,
            device=device,
        )
        query_index = torch.arange(
            n_context, n_context + len(test_ids), device=device
        )
        # docs SS138. Two env-var overrides, following the ICF_FORCE_GENERIC_EVAL
        # idiom above, so the OFFICIAL path can score the training-free variants
        # from SS136/SS137 without a second evaluation implementation:
        #
        #   ICF_COVARIANCE_BASIS=pca | pca_within
        #       replace the learned projection P with the top-K eigenvectors of
        #       THIS fold's CONTEXT cells (context-only, so no test leakage).
        #       `pca` pools around a global mean and therefore includes the
        #       between-slide term; `pca_within` centres each bag on its own mean
        #       and drops it. See the block below.
        #   ICF_FIXED_HEAD=1          replace the trained 12->1 head with the
        #       label-antisymmetric constants
        #       margin = 1.442*(CV1-CV0) - 0.343*(D1-D0) + 0.286*(q1-q0).
        #
        # Both are restored after the fold so one process can score several folds.
        basis_mode = os.environ.get("ICF_COVARIANCE_BASIS", "trained")
        use_fixed_head = os.environ.get("ICF_FIXED_HEAD") == "1"
        inner = model.model
        saved_projection = getattr(inner, "_effective_covariance_projection", None)
        saved_head = None
        if basis_mode in ("pca", "pca_within"):
            # `pca`        pools every context cell around one GLOBAL mean, so the
            #              covariance carries both within-slide variation and the
            #              between-slide mean differences. SS123-4 measured the
            #              latter at ICC 31.6%, i.e. roughly a third of this
            #              basis is spent on directions that merely tell slides
            #              apart -- staining, scanner, patient -- which is
            #              nuisance unless the task rides on it.
            # `pca_within` centres each bag on its OWN mean before accumulating,
            #              which drops the between-slide term exactly and leaves
            #              sum_i n_i * C_i / sum_i n_i. SS138-5's hypothesis for why
            #              pooled PCA loses on the full-tile path is that better
            #              per-slide mean estimates sharpen the between-slide term
            #              and pull the basis further toward it.
            with torch.no_grad():
                dim = episode_bags[0].shape[-1]
                total = 0
                scatter = torch.zeros(dim, dim, dtype=torch.float64, device=device)
                if basis_mode == "pca":
                    summation = torch.zeros(dim, dtype=torch.float64, device=device)
                    for bag in episode_bags[:n_context]:
                        summation += bag.double().sum(dim=0)
                        total += bag.shape[0]
                    pooled_mean = summation / total
                    for bag in episode_bags[:n_context]:
                        centered = bag.double() - pooled_mean
                        scatter += centered.T @ centered
                else:
                    for bag in episode_bags[:n_context]:
                        values = bag.double()
                        centered = values - values.mean(dim=0, keepdim=True)
                        scatter += centered.T @ centered
                        total += values.shape[0]
                _, vectors = torch.linalg.eigh(scatter / total)
                pca = vectors[:, -inner.covariance_sketch_dim:].flip(-1).float()
            inner._effective_covariance_projection = lambda b=pca: b
        # docs SS145. `ICF_SKETCH_DIM_DD` gives the DD branch its OWN K, so the K
        # effect measured in SS142 can be attributed to a branch instead of to
        # "the model". K enters two places and SS142 moved both at once:
        #
        #   CV  the descriptor is triu(B^T C_bag B), so K sets its length
        #   DD  reconstructs those same K x K matrices from that triangle
        #       (`_covariance_matrices_from_triangle`) and takes a rank-1
        #       dispersion direction through two `eigh`s
        #   CT  selects on RAW cells -- basis-free, so K cannot touch it
        #
        # Decoupling is EXACT, not an approximation: the PCA basis is eigenvectors
        # sorted by descending eigenvalue, so the top-left k x k block of
        # B^T C B computed at K equals B^T C B computed at k. Slicing is therefore
        # the same thing as rebuilding at the smaller K, with no second eigh.
        #
        # `covariance_sketch_dim` has to move with the slice because `_dd_direction`
        # builds its shrinkage identity from it; it is restored immediately.
        dd_sketch_dim = os.environ.get("ICF_SKETCH_DIM_DD")
        saved_dd_features = None
        if dd_sketch_dim is not None:
            dd_sketch_dim = int(dd_sketch_dim)
            if dd_sketch_dim > inner.covariance_sketch_dim:
                raise ValueError(
                    f"ICF_SKETCH_DIM_DD={dd_sketch_dim} exceeds the CV sketch dim "
                    f"{inner.covariance_sketch_dim}; DD reads a SUB-block of the CV "
                    "triangle and cannot see directions CV never computed."
                )
            saved_dd_features = inner._dd_distance_features

            def dd_with_own_sketch_dim(
                context_covariance, context_labels, query_covariance,
                _original=saved_dd_features, _k=dd_sketch_dim,
            ):
                saved_dim = inner.covariance_sketch_dim
                inner.covariance_sketch_dim = _k
                try:
                    return _original(
                        context_covariance[..., :_k, :_k],
                        context_labels,
                        query_covariance[..., :_k, :_k],
                    )
                finally:
                    inner.covariance_sketch_dim = saved_dim

            inner._dd_distance_features = dd_with_own_sketch_dim
        # docs SS146. Adaptive-rank DD: keep r > 1 dispersion directions, but only
        # those whose class separation in log(u^T C_b u) passes a Welch t on the
        # CONTEXT bags. Rank 1 is always kept, so the arm falls back to today's
        # behaviour rather than to nothing.
        #   ICF_DD_RANK_MAX=r        candidates considered (1 = unchanged)
        #   ICF_DD_RANK_TSTAT=x      |t| a direction past the first must reach
        #                            (0 = keep all r, inf = keep only rank 1)
        #   ICF_DD_RANK_SCALE=1      divide distances by the count kept (control
        #                            for DD's magnitude growing with r)
        #   ICF_DD_SELECT=lambda_plus_t|tstat   how directions are CHOSEN (SS147).
        #                            |lambda| = the dispersion gap is LARGE,
        #                            |t| = it is CONSISTENT. SS146-2 measured that
        #                            the two disagree, so they are complementary.
        #   ICF_DD_TSTAT_RANGE=1:16  |lambda|-rank window the |t| argmax comes from
        rank_max = int(os.environ.get("ICF_DD_RANK_MAX", "1"))
        selection = os.environ.get("ICF_DD_SELECT", "eigenvalue")
        saved_rank_features = None
        if rank_max > 1 or selection != "eigenvalue":
            from src.models.dd_adaptive_rank import (  # noqa: PLC0415
                AdaptiveRankConfig,
                adaptive_dd_distance_features,
            )

            if saved_dd_features is not None:
                raise ValueError(
                    "ICF_DD_RANK_MAX and ICF_SKETCH_DIM_DD both replace "
                    "_dd_distance_features; run them one at a time (SS127-2)."
                )
            saved_rank_features = inner._dd_distance_features
            low, high = os.environ.get("ICF_DD_TSTAT_RANGE", "1:16").split(":")
            rank_config = AdaptiveRankConfig(
                rank_max=rank_max,
                t_threshold=float(os.environ.get("ICF_DD_RANK_TSTAT", "2.5")),
                scale_by_rank=os.environ.get("ICF_DD_RANK_SCALE") == "1",
                shrinkage=float(inner.dd_shrinkage),
                eps=float(inner.dd_eps),
                selection=selection,
                tstat_range=(int(low), int(high)),
            )
            print(f"ICF_DD_SELECT={selection} range=({low},{high}) rank_max={rank_max}",
                  flush=True)

            def dd_with_adaptive_rank(
                context_covariance, context_labels, query_covariance,
                _config=rank_config,
            ):
                distances, separation, kept = adaptive_dd_distance_features(
                    context_covariance, context_labels, query_covariance, _config
                )
                DD_RANKS_KEPT.append(kept)
                return distances, separation

            inner._dd_distance_features = dd_with_adaptive_rank
        # docs SS148. `ICF_CT_READOUT=prototype|ridge` replaces CT's step 6-7 only.
        # Steps 1-5 (cells -> tokens -> 16-d abundance) come from the SHARED
        # `ct_abundance`, so an arm cannot differ in the representation -- which is
        # the whole point: this separates "the tokens are weak" from "reading two
        # of sixteen coordinates loses information".
        #
        # The alternative margin is calibrated to the extreme margin's CONTEXT mean
        # and RMS before entering the head, so the fixed 0.286 weight keeps its
        # meaning and the comparison is of readout QUALITY, not CT's magnitude.
        # `extreme` is passed through untouched, so the baseline cannot move.
        #   ICF_CT_PCA_DIM=k       measure cell-token distances in the leading k
        #                          PCA directions instead of raw 1536-d (SS149).
        #                          The basis is the one the CV branch already built
        #                          for this fold, so no extra eigh and CT is
        #                          guaranteed to see the same subspace.
        #   ICF_CT_PCA_SCALING     standardise (default) | raw
        ct_readout = os.environ.get("ICF_CT_READOUT", "extreme")
        ct_pca_dim = os.environ.get("ICF_CT_PCA_DIM")
        saved_ct_features = None
        if ct_readout != "extreme" or ct_pca_dim is not None:
            from src.models.ct_readout import CTReadoutConfig, ct_margins  # noqa: PLC0415

            readout_config = CTReadoutConfig(
                num_tokens=int(inner.ct_num_tokens),
                cells_per_bag=int(inner.ct_cells_per_bag),
                temperature=float(inner.ct_temperature),
                eps=float(inner.ct_eps),
                ridge_lambda=float(os.environ.get("ICF_CT_RIDGE_LAMBDA", "1.0")),
                pca_dim=None if ct_pca_dim is None else int(ct_pca_dim),
                pca_scaling=os.environ.get("ICF_CT_PCA_SCALING", "standardise"),
            )
            if ct_pca_dim is not None and basis_mode not in ("pca", "pca_within"):
                raise ValueError(
                    "ICF_CT_PCA_DIM reuses the CV branch's PCA basis, so it needs "
                    "ICF_COVARIANCE_BASIS=pca or pca_within."
                )
            saved_ct_features = inner._ct_features
            calibrated = os.environ.get("ICF_CT_CALIBRATE", "1") == "1"

            def ct_with_readout(
                context_bags_, context_labels_, query_bags_,
                _mode=ct_readout, _config=readout_config, _calibrated=calibrated,
            ):
                margins, _ = ct_margins(
                    context_bags_, context_labels_, query_bags_, _config,
                    mode=_mode, calibrated=_calibrated,
                    pca_basis=(
                        None if _config.pca_dim is None
                        else inner._effective_covariance_projection()
                    ),
                )
                # The head weighs q1 - q0, so split the margin symmetrically.
                return -0.5 * margins.query, 0.5 * margins.query, margins.separation

            inner._ct_features = ct_with_readout
            print(f"ICF_CT_READOUT={ct_readout} calibrated={calibrated} "
                  f"lambda={readout_config.ridge_lambda} "
                  f"pca_dim={readout_config.pca_dim} "
                  f"scaling={readout_config.pca_scaling}", flush=True)
        if use_fixed_head:
            head = inner.cv_dd_ct_head[0]
            saved_head = (head.weight.detach().clone(), head.bias.detach().clone())
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
                for slot, value in ((0, -1.442), (1, 1.442), (4, 0.343),
                                    (5, -0.343), (8, -0.286), (9, 0.286)):
                    head.weight[0, slot] = value
        try:
            with torch.no_grad(), autocast:
                logits = model.model(episode_bags, episode_y, query_index)
        finally:
            if basis_mode in ("pca", "pca_within") and saved_projection is not None:
                inner._effective_covariance_projection = saved_projection
            if saved_head is not None:
                with torch.no_grad():
                    inner.cv_dd_ct_head[0].weight.copy_(saved_head[0])
                    inner.cv_dd_ct_head[0].bias.copy_(saved_head[1])
            if saved_dd_features is not None:
                inner._dd_distance_features = saved_dd_features
            if saved_rank_features is not None:
                inner._dd_distance_features = saved_rank_features
            if saved_ct_features is not None:
                inner._ct_features = saved_ct_features
        scores = torch.softmax(logits.float(), dim=-1)[:, 1]
        nan_count = int(torch.isnan(scores).sum())
        probabilities = [float(value) for value in scores]
        queried_ids = list(test_ids)
    elif use_cache:
        context_ids = sample_context_ids()
        episode_bags = [
            *(subsample_context_bag(projected[s]) for s in context_ids),
            *(subsample_bag(projected[s]) for s in test_ids),
        ]
        n_context = len(context_ids)
        episode_y = torch.tensor(
            [train_y[s] for s in context_ids] + [test_y[s] for s in test_ids],
            dtype=torch.long,
            device=device,
        )
        is_context = torch.zeros(len(episode_bags), dtype=torch.bool, device=device)
        is_context[:n_context] = True

        base = model.model
        aggregator = base.aggregator
        with torch.no_grad(), autocast:
            # One pass for the whole fold. Mirrors BaseModel.forward's ragged
            # path, hoisting everything that does not depend on the query.
            normalized_bags = aggregator._normalize_bags(episode_bags)
            if aggregator.bag_representation in ("poolz", "poolz_l2"):
                pool_mean, pool_std = aggregator._context_pool_stats(
                    normalized_bags, is_context
                )
            else:
                pool_mean = pool_std = None
            representation = aggregator(episode_bags, context_mask=is_context)
            context_representation = {
                name: tokens[is_context] for name, tokens in representation.items()
            }
            context_labels = episode_y[is_context]

            for query_index, query_id in enumerate(test_ids):
                position = n_context + query_index
                query_representation = {
                    name: tokens[position : position + 1]
                    for name, tokens in representation.items()
                }
                # Per-query `_bag_view` is one bag and uses context-only pool
                # statistics, so it is unchanged by the hoist.
                query_instances = [
                    aggregator._bag_view(
                        normalized_bags[position], pool_mean, pool_std
                    )[0]
                ]
                # One meta-classifier call PER QUERY keeps the single-query
                # `_covariance_relation_scores` margin behaviour that
                # --batch-queries breaks.
                logits = base.meta_classifier(
                    context=context_representation,
                    context_labels=context_labels,
                    query=query_representation,
                    query_instances=query_instances,
                )
                probability = float(
                    torch.softmax(logits.float(), dim=-1)[0, 1].item()
                )
                if probability != probability:
                    nan_count += 1
                probabilities.append(probability)
                queried_ids.append(query_id)
                if (query_index + 1) % 20 == 0 or query_index + 1 == len(test_ids):
                    print(
                        f"  ... {query_index + 1}/{len(test_ids)} queries (cached)",
                        flush=True,
                    )
    elif batch_queries:
        # Single shared context (built/subsampled once), all queries masked
        # together in one forward call. See WARNING in the docstring above.
        context_ids = sample_context_ids()
        context_bags = [subsample_context_bag(projected[s]) for s in context_ids]
        query_bags = [subsample_bag(projected[s]) for s in test_ids]
        episode_bags = [*context_bags, *query_bags]
        episode_y = torch.tensor(
            [train_y[s] for s in context_ids] + [test_y[s] for s in test_ids],
            dtype=torch.long,
            device=device,
        )
        n_context = len(context_ids)
        mask_index = torch.arange(
            n_context, n_context + len(test_ids), device=device
        )
        with torch.no_grad(), autocast:
            logits = model.model.forward(episode_bags, episode_y, mask_index)
        probability_t = torch.softmax(logits.float(), dim=-1)[:, 1].cpu()
        probabilities = probability_t.tolist()
        queried_ids = list(test_ids)
        nan_count = int(sum(1 for p in probabilities if p != p))
    else:
        with torch.no_grad(), autocast:
            for query_index, query_id in enumerate(test_ids):
                context_ids = sample_context_ids()
                context_bags = [
                    subsample_context_bag(projected[s]) for s in context_ids
                ]
                query_bag = subsample_bag(projected[query_id])
                episode_bags = [*context_bags, query_bag]
                episode_y = torch.tensor(
                    [train_y[s] for s in context_ids] + [test_y[query_id]],
                    dtype=torch.long,
                    device=device,
                )
                mask_index = torch.tensor([len(context_ids)], device=device)
                logits = model.model.forward(episode_bags, episode_y, mask_index)
                probability = float(
                    torch.softmax(logits.float(), dim=-1)[0, 1].item()
                )
                if probability != probability:
                    nan_count += 1
                probabilities.append(probability)
                queried_ids.append(query_id)
                if (query_index + 1) % 20 == 0 or query_index + 1 == len(test_ids):
                    print(
                        f"  ... {query_index + 1}/{len(test_ids)} episodes",
                        flush=True,
                    )

    probability = torch.tensor(probabilities)
    target = torch.tensor([test_y[s] for s in queried_ids], dtype=torch.long)
    valid = torch.isfinite(probability)
    return {
        "probability": probability[valid],
        "target": target[valid],
        "queried_ids": [
            s for s, v in zip(queried_ids, valid.tolist()) if v
        ],
        "nan_count": nan_count,
    }


def evaluate_cv(
    *,
    model,
    fold_states: list[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Run the K-fold CV loop over pre-split fold files and report/save."""
    y_dict = {
        sid: int(label)
        for state in fold_states
        for sid, label in zip(state["slide_id"], state["label"])
    }
    all_ids = [sid for state in fold_states for sid in state["slide_id"]]
    print(f"CV: {len(fold_states)}-fold over {len(all_ids)} slides "
          f"({', '.join(str(len(state['slide_id'])) for state in fold_states)} "
          f"per fold), raw {FEATURE_DIM}-d, {args.context_mode}-context, all tiles")
    # Move every slide's features to the device once; fold k queries its own
    # slides with the other folds as context.
    projected = {
        sid: bag.to(device)
        for state in fold_states
        for sid, bag in zip(state["slide_id"], state["bag"])
    }
    fold_records: list[dict | None] = []
    pooled_prob: list[torch.Tensor] = []
    pooled_target: list[torch.Tensor] = []
    for fold_index, fold_state in enumerate(fold_states):
        fold_ids = fold_state["slide_id"]
        context_ids = [
            sid
            for other_index, other in enumerate(fold_states)
            if other_index != fold_index
            for sid in other["slide_id"]
        ]
        print(f"\n--- CV fold {fold_index + 1}/{len(fold_states)} "
              f"({len(fold_ids)} query, {len(context_ids)} context) ---")
        result = evaluate_trial(
            model=model,
            projected=projected,
            train_ids=context_ids,
            test_ids=fold_ids,
            train_y=y_dict,
            test_y=y_dict,
            context_mode=args.context_mode,
            context_per_class=args.context_per_class,
            max_tiles=args.max_tiles,
            context_max_tiles=args.context_max_tiles,
            seed=args.seed + fold_index,
            device=device,
            batch_queries=args.batch_queries,
            precision=args.precision,
            cache_context=args.cache_context,
        )
        if result["nan_count"]:
            print(f"WARNING: {result['nan_count']}/{len(result['queried_ids'])} "
                  f"predictions were NaN (dropped).")
        probability = result["probability"]
        target = result["target"]
        if len(probability) < 2 or target.unique().numel() < 2:
            print("Fold has insufficient valid predictions / both classes "
                  "to compute AUROC.")
            fold_records.append(None)
            continue
        predicted = (probability > 0.5).long()
        fold_auroc = auroc(probability, target)
        fold_acc = float((predicted == target).float().mean().item())
        sensitivity = float((predicted[target == 1] == 1).float().mean().item())
        specificity = float((predicted[target == 0] == 0).float().mean().item())
        print(f"  fold {fold_index + 1}: AUROC {fold_auroc:.4f}  "
              f"Acc {fold_acc:.4f}  "
              f"BAcc {0.5 * (sensitivity + specificity):.4f}  "
              f"n_query {len(probability)}")
        fold_records.append(
            {
                "slide_id": result["queried_ids"],
                "label": target,
                "probability": probability,
                "prediction": predicted,
                "auroc": fold_auroc,
            }
        )
        pooled_prob.append(probability)
        pooled_target.append(target)

    valid_folds = [r for r in fold_records if r is not None]
    if not valid_folds:
        return
    fold_aurocs = [r["auroc"] for r in valid_folds]
    fold_auroc_mean = sum(fold_aurocs) / len(fold_aurocs)
    pooled_prob_t = torch.cat(pooled_prob)
    pooled_target_t = torch.cat(pooled_target)
    pooled_auroc = auroc(pooled_prob_t, pooled_target_t)
    print(f"\n=== PathoBench {len(fold_states)}-fold CV — {args.csv.name} — "
          f"{len(all_ids)} slides ===")
    print("per-fold AUROC: " + " ".join(f"{a:.4f}" for a in fold_aurocs))
    print(f"fold-mean AUROC: {fold_auroc_mean:.4f}  "
          f"pooled AUROC: {pooled_auroc:.4f}")

    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": args.csv.name,
                "cv_folds": len(fold_states),
                "per_fold": [
                    {
                        "slide_id": r["slide_id"],
                        "label": r["label"],
                        "probability": r["probability"],
                        "prediction": r["prediction"],
                        "auroc": r["auroc"],
                    }
                    for r in valid_folds
                ],
                "aggregate": {
                    "fold_aurocs": fold_aurocs,
                    "fold_auroc_mean": fold_auroc_mean,
                    "auroc_pooled": pooled_auroc,
                    "n_slides": len(all_ids),
                },
            },
            args.output,
        )
        print(f"\nSaved CV predictions to {args.output}")


def evaluate_official_folds(
    *,
    model,
    task_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Run the official Patho-Bench k=all.tsv fold protocol (e.g. 50-fold).

    Reads ``{task_dir}/k=all.tsv`` (case_id, slide_id, <task_col>, fold_0..)
    and ``config.yaml`` (task_col). For each official fold k, slides with
    ``fold_k == 'test'`` are queried using all other slides (train+val) as
    in-context context (all-context, full tiles). Reports per-fold AUROC,
    fold mean+std, and pooled AUROC -- matching how SEAL reports macro-AUC
    over the official folds.
    """
    import yaml

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
    labels_raw = {sid: int(float(r[task_col])) for sid, r in zip(slide_ids, records)}
    n_classes = len(set(labels_raw.values()))
    if n_classes > 2:
        print(f"[binarized] {n_classes} classes -> 0 vs rest")
        labels_raw = {s: int(labels_raw[s] != 0) for s in labels_raw}

    h5_index = index_h5_files(args.features)
    slide_ids = [s for s in slide_ids if s in h5_index]
    missing = len(records) - len(slide_ids)
    if missing:
        print(f"WARNING: dropping {missing} slides with no feature file")
    bags = {sid: load_slide_features(sid, h5_index) for sid in slide_ids}
    projected = {sid: bag.to(device) for sid, bag in bags.items()}
    print(f"Loaded {len(projected)} slides, {len(fold_cols)} official folds "
          f"({fold_cols[0]}..{fold_cols[-1]}), raw {FEATURE_DIM}-d")

    start = args.official_fold_start
    total_folds = len(fold_cols)
    n_folds = args.official_nfolds or (total_folds - start)
    end = start + n_folds
    scope = list(range(start, min(end, total_folds)))

    ckpt_path = args.official_ckpt.expanduser().resolve() if args.official_ckpt else None
    results: dict[int, dict] = {}
    if ckpt_path and ckpt_path.exists():
        try:
            results = torch.load(ckpt_path, map_location="cpu", weights_only=False)["results"]
            print(f"Resuming official-fold checkpoint {ckpt_path.name}: "
                  f"{len(results)} folds already done")
        except Exception:
            results = {}
            print("WARNING: could not read checkpoint (will recompute)")

    index = {sid: i for i, sid in enumerate(slide_ids)}
    for k in scope:
        if k in results:
            print(f"  fold {k + 1}/{total_folds}: already done (skip)")
            continue
        fc = fold_cols[k]
        test_ids = [s for s in slide_ids if records[index[s]][fc].strip() == "test"]
        context_ids = [s for s in slide_ids if records[index[s]][fc].strip() != "test"]
        if len(test_ids) < 2:
            print(f"  fold {k + 1}/{total_folds}: skip (only {len(test_ids)} test slides)")
            continue
        result = evaluate_trial(
            model=model,
            projected=projected,
            train_ids=context_ids,
            test_ids=test_ids,
            train_y=labels_raw,
            test_y=labels_raw,
            context_mode="all",
            context_per_class=args.context_per_class,
            max_tiles=args.max_tiles,
            context_max_tiles=args.context_max_tiles,
            seed=args.seed + k,
            device=device,
            batch_queries=args.batch_queries,
            precision=args.precision,
            cache_context=args.cache_context,
        )
        probability = result["probability"]
        target = result["target"]
        if len(probability) < 2 or target.unique().numel() < 2:
            print(f"  fold {k + 1}/{total_folds}: insufficient valid predictions/classes")
            continue
        fa = auroc(probability, target)
        results[k] = {
            "slide_id": result["queried_ids"],
            "label": target,
            "probability": probability,
        }
        print(f"  fold {k + 1}/{total_folds}: AUROC {fa:.4f}  n_query {len(probability)}", flush=True)
        if ckpt_path:
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"task": tsv.parent.name, "results": results}, ckpt_path)

    done_scope = [k for k in scope if k in results]
    if not done_scope:
        print("No folds recorded.")
        return
    fold_aurocs = [float(auroc(results[k]["probability"], results[k]["label"])) for k in done_scope]
    pooled = auroc(
        torch.cat([results[k]["probability"] for k in done_scope]),
        torch.cat([results[k]["label"] for k in done_scope]),
    )
    mean = sum(fold_aurocs) / len(fold_aurocs)
    std = (sum((x - mean) ** 2 for x in fold_aurocs) / len(fold_aurocs)) ** 0.5
    print(f"\n=== PathoBench official {len(done_scope)}-fold — {tsv.parent.parent.name}/"
          f"{tsv.parent.name} — {len(slide_ids)} slides (folds {start + 1}..{end}) ===")
    print(f"per-fold AUROC: {' '.join(f'{x:.4f}' for x in fold_aurocs)}")
    print(f"fold-mean AUROC: {mean:.4f} ± {std:.4f}   pooled AUROC: {pooled:.4f}")
    if DD_RANKS_KEPT:
        # docs SS146: without this, "rank 2 did not help" is indistinguishable
        # from "the threshold never let a second direction through".
        counts = sorted(set(DD_RANKS_KEPT))
        histogram = "  ".join(
            f"r={r}:{DD_RANKS_KEPT.count(r)}" for r in counts
        )
        print(f"DD ranks kept: mean {sum(DD_RANKS_KEPT)/len(DD_RANKS_KEPT):.2f}   {histogram}")

    if args.output is not None:
        out = args.output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task": tsv.parent.name,
                "official_folds": len(done_scope),
                "fold_start": start,
                "fold_indices": done_scope,
                "fold_aurocs": fold_aurocs,
                "fold_auroc_mean": mean,
                "fold_auroc_std": std,
                "auroc_pooled": float(pooled),
                "per_fold": [results[k] for k in done_scope],
                "n_slides": len(slide_ids),
            },
            out,
        )
        print(f"Saved official-fold predictions to {out}")


def apply_sketch_dim_override(config: dict) -> int | None:
    """`ICF_SKETCH_DIM` — change K, the covariance sketch dimension (docs SS142).

    K used to be locked to the checkpoint: P was a learned 1536xK matrix, so
    moving K meant retraining. Under v106 the projection is per-episode PCA and
    the head is three constants, so K became a free evaluation-time knob and
    `_covariance_projection` is the ONLY parameter whose shape depends on it --
    verified against the v98 checkpoint, where the other 24 tensors are all
    encoder/head weights of fixed shape. That single tensor is exactly the one
    `ICF_COVARIANCE_BASIS=pca_within` overrides, so dropping it costs nothing.

    ⚠️ Only meaningful together with `ICF_COVARIANCE_BASIS=pca|pca_within`. With
    the trained basis there is no 1536xK matrix to supply and the load will fail.
    """
    override = os.environ.get("ICF_SKETCH_DIM")
    if override is None:
        return None
    sketch_dim = int(override)
    previous = config["model"].get("covariance_sketch_dim")
    config["model"]["covariance_sketch_dim"] = sketch_dim
    print(f"ICF_SKETCH_DIM: covariance_sketch_dim {previous} -> {sketch_dim}", flush=True)
    if os.environ.get("ICF_COVARIANCE_BASIS") not in ("pca", "pca_within"):
        raise ValueError(
            "ICF_SKETCH_DIM requires ICF_COVARIANCE_BASIS=pca or pca_within; "
            "the trained projection has no K-agnostic form."
        )
    return sketch_dim


def load_state_dict_for_sketch_dim(model, state_dict: dict) -> None:
    """`load_state_dict`, dropping ONLY the K-shaped projection when K moved.

    Anything else that fails to match is a real error and is re-raised: a silent
    `strict=False` here would let an unrelated architecture drift load as zeros
    and quietly change the number being reported.
    """
    if os.environ.get("ICF_SKETCH_DIM") is None:
        model.load_state_dict(state_dict)
        return
    key = "model._covariance_projection"
    expected = model.state_dict()[key].shape
    if state_dict[key].shape != expected:
        print(
            f"ICF_SKETCH_DIM: dropping {key} {tuple(state_dict[key].shape)} "
            f"(model wants {tuple(expected)}; PCA supplies it per episode)",
            flush=True,
        )
        state_dict = {k: v for k, v in state_dict.items() if k != key}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if set(missing) - {key} or unexpected:
        raise RuntimeError(
            f"unexpected state_dict mismatch beyond {key}: "
            f"missing={sorted(set(missing) - {key})} unexpected={sorted(unexpected)}"
        )


def apply_ridge_lambda_override(model) -> None:
    """`ICF_RIDGE_LAMBDA` — set the ridge penalty (docs SS142).

    Needed as a CONTROL for the K sweep, not as a tuning knob. Standardisation
    scales each descriptor block to unit RMS, so a bag's squared norm grows with
    the descriptor length: K=128 gives 8,256+1,536 entries and K=256 gives
    32,896+1,536. The dual Gram therefore grows with K while a fixed lambda=1.0
    does not, and raising K silently weakens the ridge. Comparing K at fixed
    lambda bundles two knobs, which SS127-2 forbids.

    lambda was never trained: `ridge_log_lambda` sits at exp(0)=1.0 and
    `ridge_log_scale` at 2.0 on all eight v98 seeds, i.e. still at init.
    """
    override = os.environ.get("ICF_RIDGE_LAMBDA")
    if override is None:
        return
    value = float(override)
    import math

    with torch.no_grad():
        model.model.ridge_log_lambda.fill_(math.log(value))
    print(f"ICF_RIDGE_LAMBDA: ridge lambda -> {value:.4f}", flush=True)


def main() -> None:
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values)

    args = parse_args()
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    apply_sketch_dim_override(config)
    input_dim = (
        args.input_dim
        if args.input_dim is not None
        else int(config["model"].get("input_dim", 512))
    )
    pca_needed = input_dim != FEATURE_DIM

    if args.official_folds is not None:
        if pca_needed:
            raise ValueError(
                "--official-folds requires raw features (model input_dim == "
                f"FEATURE_DIM {FEATURE_DIM}); PCA-per-fold is not supported."
            )
        model = build_model(config)
        checkpoint = torch.load(
            args.checkpoint.expanduser().resolve(), map_location="cpu"
        )
        model.on_load_checkpoint(checkpoint)
        load_state_dict_for_sketch_dim(model, checkpoint["state_dict"])
        apply_ridge_lambda_override(model)
        if args.rare_logits_zero:
            model.model.meta_classifier.force_rare_logits_zero = True
        model.eval()
        model.to(device)
        print(f"Model: arch v{model.model.architecture_version}, "
              f"checkpoint {args.checkpoint.name}")
        evaluate_official_folds(
            model=model, task_dir=args.official_folds, args=args, device=device
        )
        return

    if args.csv is None:
        raise ValueError("--csv is required unless --official-folds is given")

    table = pd.read_csv(args.csv)
    if not {"slide_id", "label", "split"}.issubset(table.columns):
        raise ValueError(f"CSV must have slide_id/label/split columns: {list(table.columns)}")
    table = table[table["split"].isin(("train", "test"))]
    # Slide ids are string h5 stems; some CSVs have numeric ids (e.g.
    # BC_Therapy / CPTAC-CCRCC) that pandas reads as int64.
    table["slide_id"] = table["slide_id"].astype(str)
    train_table = table[table["split"] == "train"]
    test_table = table[table["split"] == "test"]
    if len(train_table) < 2 or len(test_table) < 1:
        raise ValueError("CSV needs at least 2 train and 1 test slides.")

    labels = table["label"].astype(int)
    if labels.nunique() > 2:
        # Binarize: class 0 vs the rest (v30 model is binary).
        table = table.assign(label=(labels != 0).astype(int))
        train_table = table[table["split"] == "train"]
        test_table = table[table["split"] == "test"]
        print(f"[binarized] {labels.nunique()} classes -> 0 vs rest "
              f"(n_pos {int((table['label'] == 1).sum())})")

    # Resolve the model input dim first: a 512-d model reads the cached
    # 1536->512 PCA features; a 1536-d model uses the raw 1536-d features
    # directly (no PCA bridge).
    config = merge_train_config(args.config.expanduser().resolve())
    config["seed"] = args.seed
    apply_sketch_dim_override(config)
    input_dim = (
        args.input_dim
        if args.input_dim is not None
        else int(config["model"].get("input_dim", 512))
    )
    pca_needed = input_dim != FEATURE_DIM

    # CV mode: if the per-fold .pt files already exist, they are the data
    # source -- skip reading h5/cache entirely so re-evaluations (any
    # checkpoint) start instantly and reuse the exact same fold split.
    fold_dir = args.data_dir.expanduser().resolve()
    cv_fold_paths = (
        [
            fold_dir / f"{args.csv.stem}_cvfold{fold_index}.pt"
            for fold_index in range(args.cv_folds)
        ]
        if args.cv_folds is not None and args.cv_folds > 1
        else []
    )
    if cv_fold_paths and all(path.exists() for path in cv_fold_paths):
        if pca_needed:
            raise ValueError(
                "--cv-folds requires raw features (model input_dim == "
                f"FEATURE_DIM {FEATURE_DIM}); PCA-per-fold CV is not supported."
            )
        print(f"Loaded {len(cv_fold_paths)} CV fold files from {fold_dir} "
              f"(skipping h5/cache)")
        fold_states = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in cv_fold_paths
        ]
        model = build_model(config)
        checkpoint = torch.load(
            args.checkpoint.expanduser().resolve(), map_location="cpu"
        )
        model.on_load_checkpoint(checkpoint)
        load_state_dict_for_sketch_dim(model, checkpoint["state_dict"])
        apply_ridge_lambda_override(model)
        if args.rare_logits_zero:
            model.model.meta_classifier.force_rare_logits_zero = True
        model.eval()
        model.to(device)
        print(f"Model: arch v{model.model.architecture_version}, "
              f"checkpoint {args.checkpoint.name}")
        evaluate_cv(model=model, fold_states=fold_states, args=args, device=device)
        return

    # Load preprocessed 512-d {task}_train.pt / {task}_test.pt when available
    # (produced by scripts/prepare_pathobench.py); otherwise fall back to
    # reading h5 + (if pca_needed) fitting a train-only PCA on the GPU.
    data_dir = args.data_dir.expanduser().resolve()
    cached_train = data_dir / f"{args.csv.stem}_train.pt"
    cached_test = data_dir / f"{args.csv.stem}_test.pt"
    # The preprocessed cache holds 512-d PCA features; use it only when its
    # stored dim equals the model input dim. A 1536-d model must read raw
    # 1536-d h5 directly (no PCA, no 512-d cache) -- otherwise feeding the
    # 512-d cache to a 1536-d model would silently break.
    use_cache = cached_train.exists() and cached_test.exists()
    if use_cache:
        probe = torch.load(cached_train, map_location="cpu", weights_only=False)
        cache_dim = int(probe["bag"][0].shape[-1])
        use_cache = cache_dim == input_dim
        if not use_cache:
            print(
                f"NOTE: preprocessed cache is {cache_dim}-d but model input_dim="
                f"{input_dim} -> reading raw {FEATURE_DIM}-d h5 instead."
            )
    projected: dict[str, torch.Tensor] = {}
    if use_cache:
        train_state = torch.load(
            cached_train, map_location="cpu", weights_only=False
        )
        test_state = torch.load(cached_test, map_location="cpu", weights_only=False)
        train_ids = list(train_state["slide_id"])
        test_ids = list(test_state["slide_id"])
        for slide_id, bag in zip(train_state["slide_id"], train_state["bag"]):
            projected[slide_id] = bag.to(device)
        for slide_id, bag in zip(test_state["slide_id"], test_state["bag"]):
            projected[slide_id] = bag.to(device)
        print(f"Loaded preprocessed {args.csv.name}: {len(train_ids)} train / "
              f"{len(test_ids)} test slides (512-d, {cached_train.name})")
    else:
        # Index h5 files once, then load all needed slide features (subsampled).
        # Some CSV slide ids have no extracted feature file (feature-extraction
        # failures); drop them with a warning instead of crashing.
        h5_index = index_h5_files(args.features)
        all_ids = set(train_table["slide_id"]) | set(test_table["slide_id"])
        missing = sorted(slide for slide in all_ids if slide not in h5_index)
        if missing:
            print(f"WARNING: dropping {len(missing)} slides with no feature file "
                  f"(e.g. {missing[:3]})")
            table = table[table["slide_id"].isin(h5_index)]
            train_table = table[table["split"] == "train"]
            test_table = table[table["split"] == "test"]
            if len(train_table) < 2 or len(test_table) < 1:
                raise ValueError(
                    "Not enough slides with feature files after dropping."
                )
        slide_ids = sorted(
            set(train_table["slide_id"]) | set(test_table["slide_id"])
        )
        bags: dict[str, torch.Tensor] = {}
        for index, slide_id in enumerate(slide_ids):
            bags[slide_id] = load_slide_features(slide_id, h5_index)
            if (index + 1) % 100 == 0 or index + 1 == len(slide_ids):
                print(f"  loaded {index + 1}/{len(slide_ids)} slides", flush=True)
        train_ids = train_table["slide_id"].tolist()
        test_ids = test_table["slide_id"].tolist()
        print(f"PathoBench task {args.csv.name}: {len(train_ids)} train / "
              f"{len(test_ids)} test slides")

        # PCA bridge fit on ALL train tiles (GPU, chunked) only when the model
        # input dim is smaller than the raw feature dim.
        if pca_needed:
            train_tiles = torch.cat([bags[s] for s in train_ids], dim=0)
            pca_mean, pca_components = fit_pca(
                train_tiles, input_dim, device
            )
            print(f"PCA fit on all {train_tiles.shape[0]} train tiles -> {input_dim}-d (GPU)")
        else:
            print(f"Using raw {FEATURE_DIM}-d features (no PCA, model input_dim={input_dim})")

    train_labels = torch.tensor(train_table["label"].tolist(), dtype=torch.long)
    test_labels = torch.tensor(test_table["label"].tolist(), dtype=torch.long)
    train_y = {sid: int(train_table.loc[train_table["slide_id"] == sid, "label"].iloc[0]) for sid in train_ids}
    test_y = {sid: int(test_table.loc[test_table["slide_id"] == sid, "label"].iloc[0]) for sid in test_ids}

    model = build_model(config)
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    if args.rare_logits_zero:
        model.model.meta_classifier.force_rare_logits_zero = True
    model.eval()
    model.to(device)
    if not use_cache:
        if pca_needed:
            # Project every slide once on the GPU (CPU projection is ~1000x
            # slower); per-episode CPU projection was the eval's main bottleneck.
            pca_mean_cuda = pca_mean.to(device)
            pca_components_cuda = pca_components.to(device)
            projected = {
                slide_id: (bag.to(device) - pca_mean_cuda) @ pca_components_cuda
                for slide_id, bag in bags.items()
            }
            print(f"Projected {len(projected)} slides to {input_dim}-d on {device}")
        else:
            projected = {
                slide_id: bag.to(device) for slide_id, bag in bags.items()
            }
            print(f"Moved {len(projected)} slides to {device} as raw {FEATURE_DIM}-d (no PCA)")
    print(f"Model: arch v{model.model.architecture_version}, checkpoint {args.checkpoint.name}")

    if args.cv_folds is not None and args.cv_folds > 1:
        if pca_needed:
            raise ValueError(
                "--cv-folds requires raw features (model input_dim == "
                f"FEATURE_DIM {FEATURE_DIM}); PCA-per-fold CV is not supported."
            )
        # Build the per-fold .pt files once from the loaded raw features;
        # later runs (any checkpoint) skip h5 entirely and load them.
        if use_cache:
            raise ValueError(
                "CV fold files must be built from raw h5 features, but the "
                "512-d cache was used. Remove the cache or the fold files."
            )
        all_ids = train_ids + test_ids
        all_labels = [train_y[s] for s in train_ids] + [test_y[s] for s in test_ids]
        folds = stratified_folds(all_ids, all_labels, args.cv_folds, args.seed)
        fold_dir = args.data_dir.expanduser().resolve()
        fold_paths = [
            fold_dir / f"{args.csv.stem}_cvfold{fold_index}.pt"
            for fold_index in range(args.cv_folds)
        ]
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_states = []
        for fold_index, fold_ids in enumerate(folds):
            state = {
                "slide_id": fold_ids,
                "bag": [bags[s] for s in fold_ids],
                "label": [int(all_labels[all_ids.index(s)]) for s in fold_ids],
            }
            torch.save(state, fold_paths[fold_index])
            fold_states.append(state)
        print(f"Created {args.cv_folds} CV fold files in {fold_dir}: "
              f"{args.csv.stem}_cvfold{{0..{args.cv_folds - 1}}}.pt "
              f"(raw {FEATURE_DIM}-d)")
        evaluate_cv(model=model, fold_states=fold_states, args=args, device=device)
        return

    context_cap = (
        args.context_max_tiles if args.context_max_tiles is not None else args.max_tiles
    )
    if args.context_mode == "all":
        print(f"Context: ALL {len(train_ids)} train slides"
              + (f", capped at {context_cap} tiles/context-bag (query unlimited)"
                 if args.context_max_tiles else
                 (f", max {args.max_tiles} tiles/bag (context+query)" if args.max_tiles
                  else ", all tiles")))
    else:
        print(f"Context: {args.context_per_class} slides per class"
              + (f", capped at {context_cap} tiles/context-bag (query unlimited)"
                 if args.context_max_tiles else
                 (f", max {args.max_tiles} tiles/bag (context+query)" if args.max_tiles
                  else ", all tiles")))
    print(f"Trials: {args.trials} (seeds {args.seed}..{args.seed + args.trials - 1})")

    trial_results: list[dict | None] = []
    for trial in range(args.trials):
        trial_seed = args.seed + trial
        if args.trials > 1:
            print(f"\n--- trial {trial + 1}/{args.trials} (seed {trial_seed}) ---")
        result = evaluate_trial(
            model=model,
            projected=projected,
            train_ids=train_ids,
            test_ids=test_ids,
            train_y=train_y,
            test_y=test_y,
            context_mode=args.context_mode,
            context_per_class=args.context_per_class,
            max_tiles=args.max_tiles,
            context_max_tiles=args.context_max_tiles,
            seed=trial_seed,
            device=device,
            batch_queries=args.batch_queries,
            precision=args.precision,
            cache_context=args.cache_context,
        )
        if result["nan_count"]:
            print(f"WARNING: {result['nan_count']}/{len(result['queried_ids'])} "
                  f"predictions were NaN (dropped).")
        probability = result["probability"]
        target = result["target"]
        if len(probability) < 2 or target.unique().numel() < 2:
            print("Not enough valid predictions / both classes to compute AUROC.")
            trial_results.append(None)
            continue
        predicted = (probability > 0.5).long()
        accuracy = float((predicted == target).float().mean().item())
        sensitivity = float((predicted[target == 1] == 1).float().mean().item())
        specificity = float((predicted[target == 0] == 0).float().mean().item())
        metrics = {
            "auroc": auroc(probability, target),
            "accuracy": accuracy,
            "balanced_accuracy": 0.5 * (sensitivity + specificity),
            "log_loss": log_loss(probability, target),
            "task": args.csv.name,
        }
        print(f"  trial {trial + 1}: AUROC {metrics['auroc']:.4f}  "
              f"Acc {metrics['accuracy']:.4f}  "
              f"BAcc {metrics['balanced_accuracy']:.4f}  "
              f"LL {metrics['log_loss']:.4f}")
        trial_results.append(
            {
                "slide_id": result["queried_ids"],
                "label": target,
                "probability": probability,
                "prediction": predicted,
                "metrics": metrics,
            }
        )

    valid_results = [r for r in trial_results if r is not None]
    if not valid_results:
        return

    aurocs = [r["metrics"]["auroc"] for r in valid_results]
    accs = [r["metrics"]["accuracy"] for r in valid_results]
    baccs = [r["metrics"]["balanced_accuracy"] for r in valid_results]
    ll = [r["metrics"]["log_loss"] for r in valid_results]

    print(f"\n=== PathoBench zero-shot — {args.csv.name} — "
          f"{int(valid_results[0]['label'].numel())} test slides "
          f"({args.context_mode}-context, max_tiles={args.max_tiles}, "
          f"trials={len(aurocs)}) ===")
    if len(aurocs) > 1:
        print(f"AUROC   mean {_mean(aurocs):.4f}  min {min(aurocs):.4f}  "
              f"max {max(aurocs):.4f}")
        print(f"Acc     mean {_mean(accs):.4f}  min {min(accs):.4f}  "
              f"max {max(accs):.4f}")
        print(f"BAcc    mean {_mean(baccs):.4f}  min {min(baccs):.4f}  "
              f"max {max(baccs):.4f}")
        print(f"LogL    mean {_mean(ll):.4f}")
    else:
        print(f"AUROC             {aurocs[0]:.4f}")
        print(f"Accuracy          {accs[0]:.4f}")
        print(f"Balanced accuracy {baccs[0]:.4f}")
        print(f"Log loss          {ll[0]:.4f}")

    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        last = valid_results[-1]
        torch.save(
            {
                "slide_id": last["slide_id"],
                "label": last["label"],
                "probability": last["probability"],
                "prediction": last["prediction"],
                "metrics": last["metrics"],
                "trial_aurocs": aurocs,
                "trial_accuracies": accs,
                "trial_balanced_accuracies": baccs,
                "aggregate": {
                    "auroc_mean": _mean(aurocs),
                    "auroc_min": min(aurocs),
                    "auroc_max": max(aurocs),
                    "accuracy_mean": _mean(accs),
                    "balanced_accuracy_mean": _mean(baccs),
                    "log_loss_mean": _mean(ll),
                    "max_tiles": args.max_tiles,
                    "trials": len(aurocs),
                    "task": args.csv.name,
                },
            },
            args.output,
        )
        print(f"\nSaved predictions to {args.output}")


if __name__ == "__main__":
    main()
