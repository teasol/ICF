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
import math
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

from src.models.stream_eval import (  # noqa: E402
    BagStatsCache,
    covariance_basis_from_bags,
    cpu_bag_mapping,
    fisher_basis_from_bags,
    project_bags_to_cpu,
    stream_lineage_forward,
)

from src.models.training_free import _solve_kernel_ridge, _fast_context_auroc  # noqa: E402
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

# docs SS154: the per-episode log(sigma_0^2/sigma_1^2) the LLR term adds. Recorded
# so "no change" can be told apart from "the term was zero".
DD_LLR_OFFSETS: list[float] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint. Optional: when omitted, a fresh model instance "
        "is built and used uninitialized -- valid for the training-free "
        "configuration (v106+), which overrides every learned value (docs SS183).",
    )
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
    bag_stats_cache: BagStatsCache | None = None,
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
        if basis_mode in ("pca", "pca_within", "fisher", "fisher_within"):
            with torch.no_grad():
                if basis_mode in ("fisher", "fisher_within"):
                    shrinkage = float(os.environ.get("ICF_FISHER_SHRINKAGE", "0.1"))
                    pca = fisher_basis_from_bags(
                        episode_bags[:n_context],
                        episode_y[:n_context],
                        inner.covariance_sketch_dim,
                        device,
                        cache=bag_stats_cache,
                        shrinkage=shrinkage,
                    )
                else:
                    pca = covariance_basis_from_bags(
                        episode_bags[:n_context],
                        basis_mode,
                        inner.covariance_sketch_dim,
                        device,
                        cache=bag_stats_cache,
                    )
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
        # docs SS154. `ICF_DD_LLR=1` completes DD's log-likelihood ratio.
        #
        # `_dd_distance_features` returns d_c = (f - mu_c)^2 / sigma_c^2 and the head
        # forms 0.343*(d0 - d1). For two univariate Gaussians the actual LLR is
        #
        #   log p(f|1) - log p(f|0) = 1/2 (d0 - d1) + 1/2 log(sigma_0^2 / sigma_1^2)
        #
        # so today's DD is the ratio with its LOG-DETERMINANT TERM DROPPED. Adding
        # log(sigma_c^2) to d_c restores it exactly, since
        # (d0 + log s0) - (d1 + log s1) = (d0 - d1) + log(s0/s1). Class swap negates
        # it (s0 <-> s1), so the fixed head stays valid.
        #
        # ⚠️ sigma_c comes from the CONTEXT only, so the added term is CONSTANT across
        # every query in a fold. Per-fold AUROC reads only the ranking within a fold
        # and therefore cannot move; what moves is pooled AUROC (which mixes folds
        # carrying different constants) and anything calibration-based.
        dd_llr = os.environ.get("ICF_DD_LLR") == "1"
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
        dd_selection = os.environ.get("ICF_DD_SELECT", "eigenvalue")
        saved_rank_features = None
        if rank_max > 1 or dd_selection != "eigenvalue":
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
                selection=dd_selection,
                tstat_range=(int(low), int(high)),
            )
            print(f"ICF_DD_SELECT={dd_selection} range=({low},{high}) rank_max={rank_max}",
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
        ct_kmeans = os.environ.get("ICF_CT_KMEANS")
        ct_cells = os.environ.get("ICF_CT_CELLS")
        ct_abundance_cells = os.environ.get("ICF_CT_ABUNDANCE_CELLS")
        ct_sampling = os.environ.get("ICF_CT_SAMPLING")
        ct_sampling_seed = os.environ.get("ICF_CT_SAMPLING_SEED")
        ct_distance_kernel = os.environ.get("ICF_CT_DISTANCE_KERNEL")
        ct_tokenizer = os.environ.get("ICF_CT_TOKENIZER")
        ct_tokens = os.environ.get("ICF_CT_TOKENS")
        ct_kernel = os.environ.get("ICF_CT_KERNEL")
        ct_abundance_pooling = os.environ.get("ICF_CT_ABUNDANCE_POOLING")
        saved_ct_features = None
        if (ct_readout != "extreme" or ct_pca_dim is not None
                or ct_kmeans is not None or ct_cells is not None
                or ct_abundance_cells is not None or ct_sampling is not None
                or ct_sampling_seed is not None or ct_distance_kernel is not None
                or ct_tokenizer is not None or ct_tokens is not None
                or ct_kernel is not None or ct_abundance_pooling is not None):
            from src.models.ct_readout import (  # noqa: PLC0415
                CTReadoutConfig,
                ct_margins,
                parse_cell_budget,
            )

            if ct_cells is None:
                cells_per_bag = inner.ct_cells_per_bag
                cells_fraction = None
                cells_scale = os.environ.get("ICF_CT_CELLS_SCALE", "own")
            else:
                cells_per_bag, cells_fraction, parsed_scale = parse_cell_budget(
                    ct_cells
                )
                cells_scale = parsed_scale or os.environ.get(
                    "ICF_CT_CELLS_SCALE", "own"
                )
            if ct_abundance_cells is None or ct_abundance_cells == "match":
                abundance_cells_per_bag = "match"
            elif ct_abundance_cells == "all":
                abundance_cells_per_bag = None
            else:
                abundance_cap, abundance_fraction, _ = parse_cell_budget(
                    ct_abundance_cells
                )
                abundance_cells_per_bag = (
                    abundance_fraction
                    if abundance_fraction is not None
                    else abundance_cap
                )

            readout_config = CTReadoutConfig(
                # SS160: ICF_CT_TOKENS sweeps the number of clusters/features.
                num_tokens=int(os.environ.get("ICF_CT_TOKENS", inner.ct_num_tokens)),
                # SS159: ICF_CT_CELLS=all uses every cell; a number caps as before.
                # A fraction (0.125, frac:0.125, own:0.125, median:0.125) scales
                # with bag / episode size instead of repeating a global 512 cap.
                cells_per_bag=cells_per_bag,
                cells_fraction=cells_fraction,
                cells_min=int(os.environ.get("ICF_CT_CELLS_MIN", "1")),
                cells_scale=cells_scale,
                # SS165: keep dictionary/statistics at `ICF_CT_CELLS` but use a
                # separate cap for the per-bag abundance average. "all" supports
                # the random-512 dictionary / full-abundance diagnostic.
                abundance_cells_per_bag=abundance_cells_per_bag,
                abundance_pooling=os.environ.get("ICF_CT_ABUNDANCE_POOLING", "mean"),
                abundance_topk_fraction=float(
                    os.environ.get("ICF_CT_ABUNDANCE_TOPK_FRACTION", "0.1")
                ),
                abundance_topk_min=int(
                    os.environ.get("ICF_CT_ABUNDANCE_TOPK_MIN", "1")
                ),
                # Random is the active default; even spacing is retained behind
                # ICF_CT_SAMPLING=even for historical v110 reproduction.
                sampling=os.environ.get("ICF_CT_SAMPLING", "random"),
                sampling_seed=int(os.environ.get("ICF_CT_SAMPLING_SEED", "0")),
                distance_kernel=os.environ.get("ICF_CT_DISTANCE_KERNEL", "broadcast"),
                tokenizer=os.environ.get("ICF_CT_TOKENIZER", "kmeans_plusplus"),
                bisect_iterations=int(os.environ.get("ICF_CT_BISECT_ITERS", "2")),
                bisect_power_iterations=int(
                    os.environ.get("ICF_CT_BISECT_POWER_ITERS", "3")
                ),
                tree_reduction=os.environ.get("ICF_CT_TREE_REDUCTION", "segment"),
                hdbscan_min_cluster_size=int(
                    os.environ.get("ICF_CT_HDBSCAN_MIN_CLUSTER_SIZE", "256")
                ),
                hdbscan_min_cluster_fraction=float(
                    os.environ.get("ICF_CT_HDBSCAN_MIN_CLUSTER_FRACTION", "0.001")
                ),
                hdbscan_min_samples=int(
                    os.environ.get("ICF_CT_HDBSCAN_MIN_SAMPLES", "32")
                ),
                hdbscan_cluster_selection_method=os.environ.get(
                    "ICF_CT_HDBSCAN_SELECTION", "leaf"
                ),
                hdbscan_build_algo=os.environ.get(
                    "ICF_CT_HDBSCAN_BUILD_ALGO", "nn_descent"
                ),
                hdbscan_allow_single_cluster=(
                    os.environ.get("ICF_CT_HDBSCAN_ALLOW_SINGLE", "0") == "1"
                ),
                dbscan_eps=(
                    None if os.environ.get("ICF_CT_DBSCAN_EPS") is None
                    else float(os.environ["ICF_CT_DBSCAN_EPS"])
                ),
                dbscan_min_samples=int(
                    os.environ.get("ICF_CT_DBSCAN_MIN_SAMPLES", "16")
                ),
                temperature=float(inner.ct_temperature),
                eps=float(inner.ct_eps),
                ridge_lambda=float(os.environ.get("ICF_CT_RIDGE_LAMBDA", "1.0")),
                kernel=os.environ.get("ICF_CT_KERNEL", "rbf"),
                kernel_gamma=(
                    None if os.environ.get("ICF_CT_KERNEL_GAMMA") is None
                    else float(os.environ["ICF_CT_KERNEL_GAMMA"])
                ),
                kernel_degree=int(os.environ.get("ICF_CT_KERNEL_DEGREE", "2")),
                kernel_coef0=float(os.environ.get("ICF_CT_KERNEL_COEF0", "1.0")),
                pca_dim=None if ct_pca_dim is None else int(ct_pca_dim),
                pca_scaling=os.environ.get("ICF_CT_PCA_SCALING", "standardise"),
                # SS157: Lloyd iterations refining the farthest-point tokens.
                kmeans_iterations=int(os.environ.get("ICF_CT_KMEANS", "0")),
                kmeans_max_iterations=int(
                    os.environ.get("ICF_CT_KMEANS_MAX_ITER", "8")
                ),
                kmeans_tolerance=float(
                    os.environ.get("ICF_CT_KMEANS_TOL", "1e-4")
                ),
                kmeans_seed=int(os.environ.get("ICF_CT_KMEANS_SEED", "0")),
            )
            if ct_pca_dim is not None and basis_mode not in ("pca", "pca_within", "fisher", "fisher_within"):
                raise ValueError(
                    "ICF_CT_PCA_DIM reuses the CV branch's PCA basis, so it needs "
                    "ICF_COVARIANCE_BASIS=pca, pca_within, or fisher."
                )

            saved_ct_features = inner._ct_features
            calibrated = os.environ.get("ICF_CT_CALIBRATE", "1") == "1"

            def ct_with_readout(
                context_bags_, context_labels_, query_bags_,
                _mode=ct_readout, _config=readout_config, _calibrated=calibrated,
            ):
                if float(os.environ.get("ICF_FIXED_HEAD_CT_WEIGHT", "0.286")) == 0.0:
                    nq = len(query_bags_)
                    dev = context_labels_.device
                    zero = torch.zeros(nq, device=dev)
                    return zero, zero, torch.tensor(0.0, device=dev)
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
                  f"scaling={readout_config.pca_scaling} "
                  f"tokenizer={readout_config.tokenizer} "
                  f"kmeans={readout_config.kmeans_iterations} "
                  f"kmeans_max={readout_config.kmeans_max_iterations} "
                  f"kmeans_tol={readout_config.kmeans_tolerance} "
                  f"kmeans_seed={readout_config.kmeans_seed} "
                  f"cells={readout_config.cells_per_bag} "
                  f"cells_fraction={readout_config.cells_fraction} "
                  f"cells_min={readout_config.cells_min} "
                  f"cells_scale={readout_config.cells_scale} "
                  f"abundance_cells={readout_config.abundance_cells_per_bag} "
                  f"sampling={readout_config.sampling} "
                  f"sampling_seed={readout_config.sampling_seed} "
                  f"distance_kernel={readout_config.distance_kernel} "
                  f"tree_reduction={readout_config.tree_reduction} "
                  f"hdbscan_min_cluster_size={readout_config.hdbscan_min_cluster_size} "
                  f"hdbscan_min_cluster_fraction="
                  f"{readout_config.hdbscan_min_cluster_fraction} "
                  f"hdbscan_min_samples={readout_config.hdbscan_min_samples} "
                  f"dbscan_eps={readout_config.dbscan_eps} "
                  f"dbscan_min_samples={readout_config.dbscan_min_samples} "
                  f"kernel={readout_config.kernel} "
                  f"kernel_gamma={readout_config.kernel_gamma} "
                  f"kernel_degree={readout_config.kernel_degree} "
                  f"kernel_coef0={readout_config.kernel_coef0} "
                  f"abundance_pooling={readout_config.abundance_pooling} "
                  f"abundance_topk_fraction={readout_config.abundance_topk_fraction} "
                  f"abundance_topk_min={readout_config.abundance_topk_min} "
                  f"tokens={readout_config.num_tokens}", flush=True)
        # docs SS155. `ICF_DD_RELATIVE=1` ranks by (D0-D1)/(D0+D1+eps) instead of
        # (D0-D1), i.e. by the RATIO rather than the difference, which suppresses a
        # query that is far from BOTH prototypes yet has a large raw gap.
        # `ICF_DD_RELATIVE_CALIBRATE=0` turns off the rescale, as the control: the
        # relative margin lives in (-1, 1) while the difference is unbounded, so
        # feeding it raw to a fixed 0.343 compares DD's magnitude, not its shape.
        # docs SS156. `ICF_CV_BLOCKS` restricts what the CV RIDGE sees, to decompose
        # where CV's 34,432-d descriptor [vech(B^T C_bag B), xbar_b] earns its keep.
        #
        #   cov+mean (default) | cov | mean | diag | offdiag | diag+mean
        #
        # ⚠️ DD is deliberately UNTOUCHED. It rebuilds its K x K matrices from the
        # CV descriptor's triangle (`_covariance_matrices_from_triangle`) using the
        # raw `context` that `_relation_logits` holds, so masking here cannot reach
        # it -- which is the whole point: masking the descriptor globally would break
        # DD on the mean-only arm and confound the two branches. CT reads raw cells
        # and is independent either way.
        #
        # The masking rides inside `_normalize_descriptors` because `_ridge_logits`
        # looks that up on the INSTANCE (while `_ridge_logits` itself is called by
        # class from `_relation_logits`, so it cannot be patched per-instance, SS145).
        # It may change the descriptor width, and the ridge simply uses what it gets.
        #
        #   ICF_CV_BLOCK_NORM=blockwise (default) each surviving block gets its own
        #       context RMS -- what the pipeline would do for a descriptor of that
        #       shape. `parent` normalises the FULL covariance and mean blocks first
        #       and then selects columns, so diag/offdiag differ from `cov` in CONTENT
        #       ONLY. Running both separates content from normalisation.
        # docs SS162. `ICF_CV_CORR=1` gives the CV ridge the CORRELATION matrix's
        # off-diagonal instead of the covariance's:
        #
        #     corr[i,j] = C[i,j] / sqrt(C[i,i] * C[j,j]),   C = B^T C_bag B
        #
        # SS156 removed the diagonal ENTRIES, but the diagonal still sets the SCALE of
        # every off-diagonal: C[i,j] grows with sqrt(lambda_i lambda_j), so leading
        # PCA pairs are automatically larger and an unweighted ridge reads that size
        # as importance. Dividing it out is the completion of SS156's logic, and it is
        # the smallest possible change -- one transform on the raw descriptor.
        #
        # ⚠️ Per BAG: each bag is normalised by its OWN variance profile, so a bag's
        # overall spread along each direction stops contributing. If that spread was
        # signal (tumour heterogeneity, say) rather than nuisance, this loses it --
        # which is exactly the question.
        # ⚠️ DD is untouched; it reads the raw triangle from `context` (SS156-1).
        cv_blocks = os.environ.get("ICF_CV_BLOCKS")
        cv_corr = os.environ.get("ICF_CV_CORR") == "1"
        saved_normalize = None
        if cv_blocks is not None or cv_corr:
            cv_blocks = cv_blocks or "cov+mean"
            covariance_dim = int(inner.covariance_descriptor_dim)
            mean_dim = int(inner.mean_descriptor_dim)
            row, column = inner._covariance_triangle
            is_diagonal = (row == column)
            device = row.device
            cov_index = torch.arange(covariance_dim, device=device)
            groups = {
                "cov": [cov_index],
                "mean": [torch.arange(covariance_dim, covariance_dim + mean_dim, device=device)],
                "diag": [cov_index[is_diagonal]],
                "offdiag": [cov_index[~is_diagonal]],
            }
            selection = []
            for name in cv_blocks.split("+"):
                if name not in groups:
                    raise ValueError(
                        f"ICF_CV_BLOCKS parts must be in {sorted(groups)}, got {name!r}"
                    )
                selection.extend(groups[name])
            block_norm = os.environ.get("ICF_CV_BLOCK_NORM", "blockwise")
            if block_norm not in ("blockwise", "parent"):
                raise ValueError("ICF_CV_BLOCK_NORM must be blockwise or parent")
            saved_normalize = inner._normalize_descriptors
            # Triangle slot of C[i,i], in increasing i -- `triu_indices` emits the
            # diagonal in that order, so this indexes straight by direction.
            diagonal_slot = torch.where(is_diagonal)[0]

            def to_correlation(descriptor):
                triangle_block = descriptor[..., :covariance_dim]
                variances = triangle_block.index_select(-1, diagonal_slot).clamp_min(1e-12)
                scale = (variances.index_select(-1, row) * variances.index_select(-1, column)).sqrt()
                return torch.cat(
                    (triangle_block / scale.clamp_min(1e-12), descriptor[..., covariance_dim:]),
                    dim=-1,
                )

            def masked_normalize(context, query, _sel=selection, _norm=block_norm,
                                 _corr=cv_corr):
                if _corr:
                    context, query = to_correlation(context), to_correlation(query)
                if _norm == "parent":
                    # Normalise the two ORIGINAL blocks, then select columns, so a
                    # sub-block keeps the scale it had inside the full descriptor.
                    context, query = saved_normalize(context, query)
                    index = torch.cat(_sel)
                    return context.index_select(-1, index), query.index_select(-1, index)
                parts_c, parts_q = [], []
                for index in _sel:
                    piece_c, piece_q = inner._normalize_block(
                        context.index_select(-1, index), query.index_select(-1, index)
                    )
                    parts_c.append(piece_c)
                    parts_q.append(piece_q)
                return torch.cat(parts_c, dim=-1), torch.cat(parts_q, dim=-1)

            inner._normalize_descriptors = masked_normalize
            print(f"ICF_CV_BLOCKS={cv_blocks} norm={block_norm} corr={cv_corr} "
                  f"dims={sum(int(i.numel()) for i in selection)}", flush=True)
        dd_relative = os.environ.get("ICF_DD_RELATIVE") == "1"
        dd_ordered_typicality = os.environ.get("ICF_DD_ORDERED_TYPICALITY") == "1"
        saved_ordered_features = None
        if dd_ordered_typicality:
            if dd_relative or dd_llr or rank_max != 1 or dd_selection != "eigenvalue":
                raise ValueError(
                    "ICF_DD_ORDERED_TYPICALITY cannot be combined with another DD arm"
                )
            saved_ordered_features = inner._dd_distance_features
            separation_floor = float(os.environ.get("ICF_DD_SEPARATION_FLOOR", "1.0"))

            def dd_with_ordered_typicality(
                context_covariance, context_labels, query_covariance,
                _floor=separation_floor,
            ):
                return inner._dd_ordered_typicality_features(
                    context_covariance, context_labels, query_covariance, _floor
                )

            inner._dd_distance_features = dd_with_ordered_typicality
            print(
                f"ICF_DD_ORDERED_TYPICALITY=1 separation_floor={separation_floor}",
                flush=True,
            )
        saved_relative_features = None
        if dd_relative:
            from src.models.dd_adaptive_rank import relative_margin  # noqa: PLC0415

            saved_relative_features = inner._dd_distance_features
            relative_calibrate = os.environ.get("ICF_DD_RELATIVE_CALIBRATE", "1") == "1"

            def dd_with_relative_margin(
                context_covariance, context_labels, query_covariance,
                _original=saved_relative_features, _calibrate=relative_calibrate,
            ):
                distances, separation = _original(
                    context_covariance, context_labels, query_covariance
                )
                margin = relative_margin(distances, inner.dd_eps)
                if _calibrate:
                    # Context-only: score the CONTEXT bags as if they were queries
                    # and match the reference difference's centre and RMS there.
                    context_distances, _ = _original(
                        context_covariance, context_labels, context_covariance
                    )
                    reference = context_distances[:, 0] - context_distances[:, 1]
                    own = relative_margin(context_distances, inner.dd_eps)
                    centre, target = own.mean(), reference.mean()
                    spread = (own - centre).square().mean().sqrt().clamp_min(inner.dd_eps)
                    target_spread = (
                        (reference - target).square().mean().sqrt().clamp_min(inner.dd_eps)
                    )
                    margin = (margin - centre) * (target_spread / spread) + target
                # The head weighs (d0 - d1), so split the margin symmetrically.
                pair = torch.stack((0.5 * margin, -0.5 * margin), dim=-1)
                return pair, separation

            inner._dd_distance_features = dd_with_relative_margin
        saved_llr_features = None
        if dd_llr:
            saved_llr_features = inner._dd_distance_features

            def dd_with_log_determinant(
                context_covariance, context_labels, query_covariance,
                _original=saved_llr_features,
            ):
                distances, separation = _original(
                    context_covariance, context_labels, query_covariance
                )
                # sigma_c^2 cannot be read back off `distances` -- d_c already
                # divides by it, so averaging over class c returns exactly 1. It
                # comes from `class_dispersions`, which sits on the code path that
                # is pinned equal to the lineage (SS146 tests).
                from src.models.dd_adaptive_rank import (  # noqa: PLC0415
                    AdaptiveRankConfig, class_dispersions,
                )

                offset = class_dispersions(
                    context_covariance, context_labels,
                    AdaptiveRankConfig(shrinkage=float(inner.dd_shrinkage),
                                       eps=float(inner.dd_eps)),
                ).log()
                DD_LLR_OFFSETS.append(float(offset[0] - offset[1]))
                return distances + offset[None, :], separation

            inner._dd_distance_features = dd_with_log_determinant
        if use_fixed_head:
            # docs SS151. `ICF_FIXED_HEAD_CT_WEIGHT` overrides the CT coefficient.
            # 0.286 came from decomposing the eight trained heads (SS137-3), i.e. it
            # was fitted against the OLD two-token readout at raw 1536 dims. A
            # better CT margin has no reason to want the same weight, and CT's
            # weight is only a fifth of CV's 1.442, so the branch is structurally
            # limited in how far it can move the macro at 0.286.
            # Antisymmetry is untouched: the pair stays equal and opposite.
            # `ICF_FIXED_HEAD_DD_WEIGHT` likewise (docs SS153). 0.343 is a MAGNITUDE
            # fitted against the old squared-DISTANCE readout, so a large D1 is
            # evidence AGAINST class 1 and the head weighs (d0 - d1). The
            # ordered_typicality readout (SS182-3) emits LOGITS like CV/CT, so the
            # head weighs (d1 - d0) -- see the slot selection below. 0 ablates DD
            # entirely, which is the question worth asking now that SS145-147
            # closed every way of improving it -- K saturates by 128, r=1 is the
            # peak, and both the |t| gate and the |t| selector lose to |lambda|.
            # `ICF_FIXED_HEAD_CV_WEIGHT` completes the set, so any single branch can
            # be isolated by zeroing the other two (docs SS163). CV is the dominant
            # branch at 1.442, so zeroing it is the only way to see CT or DD alone.
            cv_weight = float(os.environ.get("ICF_FIXED_HEAD_CV_WEIGHT", "1.442"))
            ct_weight = float(os.environ.get("ICF_FIXED_HEAD_CT_WEIGHT", "0.286"))
            dd_weight = float(os.environ.get("ICF_FIXED_HEAD_DD_WEIGHT", "0.343"))
            head = inner.cv_dd_ct_head[0]
            saved_head = (head.weight.detach().clone(), head.bias.detach().clone())
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
                if dd_ordered_typicality:
                    dd_slots = ((4, -dd_weight), (5, dd_weight))
                else:
                    dd_slots = ((4, dd_weight), (5, -dd_weight))
                for slot, value in ((0, -cv_weight), (1, cv_weight), *dd_slots,
                                    (8, -ct_weight), (9, ct_weight)):
                    head.weight[0, slot] = value
            if (cv_weight, dd_weight, ct_weight) != (1.442, 0.343, 0.286):
                print(f"fixed head: cv={cv_weight} dd={dd_weight} ct={ct_weight}", flush=True)
        try:
            with torch.no_grad(), autocast:
                logits = stream_lineage_forward(
                    model.model, episode_bags, episode_y, query_index, device
                )
                do_context_loo = os.environ.get("ICF_AGGREGATION", "").startswith("context_loo")
                bm_weight = float(os.environ.get("ICF_FIXED_HEAD_BM_WEIGHT", "0.0"))
                if bm_weight != 0.0:
                    bm_dim = int(os.environ.get("ICF_BM_DIM", "32"))
                    bm_lambda = float(os.environ.get("ICF_BM_LAMBDA", "1.0"))
                    basis = inner._effective_covariance_projection()
                    dim = min(bm_dim, basis.shape[1])
                    bm_basis = basis[:, :dim].float()

                    def get_bag_mean(b):
                        if bag_stats_cache is not None and id(b) in bag_stats_cache:
                            stats = bag_stats_cache[id(b)]
                            return (stats.mean if hasattr(stats, 'mean') else stats[1]).to(device).float()
                        return b.float().mean(dim=0).to(device)

                    ctx_means = torch.stack([get_bag_mean(b) for b in episode_bags[:n_context]]) @ bm_basis
                    qry_means = torch.stack([get_bag_mean(episode_bags[i]) for i in query_index.tolist()]) @ bm_basis

                    krr_kernel = os.environ.get("ICF_KRR_KERNEL", "linear")
                    krr_gamma = float(os.environ["ICF_KRR_GAMMA"]) if "ICF_KRR_GAMMA" in os.environ else None
                    krr_degree = int(os.environ.get("ICF_KRR_DEGREE", "2"))
                    krr_coef0 = float(os.environ.get("ICF_KRR_COEF0", "1.0"))

                    labels = episode_y[:n_context].long().to(device)
                    bm_res = _solve_kernel_ridge(
                        ctx_means, labels, qry_means,
                        kernel=os.environ.get("ICF_BM_KERNEL", krr_kernel),
                        gamma=krr_gamma,
                        degree=krr_degree,
                        coef0=krr_coef0,
                        reg_lambda=bm_lambda,
                        return_loo=do_context_loo,
                    )
                    bm_margin, loo_bm = (bm_res[0], bm_res[1]) if do_context_loo else (bm_res, None)

                    logits = logits.clone()
                    logits[:, 0] -= 0.5 * bm_weight * bm_margin
                    logits[:, 1] += 0.5 * bm_weight * bm_margin

                # RM (Residual Bag-Mean) -- §218 screening branch.
                # BM/QA/DS all read basis[:, :32]; RM reads the DISCARDED tail
                # basis[:, rm_start:rm_start+rm_dim] of the same K=256 PCA basis,
                # so it is orthogonal to them by construction of the basis.
                rm_weight = float(os.environ.get("ICF_FIXED_HEAD_RM_WEIGHT", "0.0"))
                if rm_weight != 0.0:
                    rm_start = int(os.environ.get("ICF_RM_START", "32"))
                    rm_dim = int(os.environ.get("ICF_RM_DIM", "224"))
                    rm_lambda = float(os.environ.get("ICF_RM_LAMBDA", "1.0"))
                    basis = inner._effective_covariance_projection()
                    lo = min(rm_start, basis.shape[1])
                    hi = min(lo + rm_dim, basis.shape[1])
                    if hi > lo:
                        rm_basis = basis[:, lo:hi].float()

                        def get_bag_mean_rm(b):
                            if bag_stats_cache is not None and id(b) in bag_stats_cache:
                                stats = bag_stats_cache[id(b)]
                                return (stats.mean if hasattr(stats, 'mean') else stats[1]).to(device).float()
                            return b.float().mean(dim=0).to(device)

                        ctx_rm = torch.stack([get_bag_mean_rm(b) for b in episode_bags[:n_context]]) @ rm_basis
                        qry_rm = torch.stack([get_bag_mean_rm(episode_bags[i]) for i in query_index.tolist()]) @ rm_basis

                        rm_res = _solve_kernel_ridge(
                            ctx_rm, episode_y[:n_context].long().to(device), qry_rm,
                            kernel=os.environ.get("ICF_RM_KERNEL", os.environ.get("ICF_KRR_KERNEL", "linear")),
                            gamma=float(os.environ["ICF_KRR_GAMMA"]) if "ICF_KRR_GAMMA" in os.environ else None,
                            degree=int(os.environ.get("ICF_KRR_DEGREE", "2")),
                            coef0=float(os.environ.get("ICF_KRR_COEF0", "1.0")),
                            reg_lambda=rm_lambda,
                            return_loo=do_context_loo,
                        )
                        rm_margin, loo_rm = (rm_res[0], rm_res[1]) if do_context_loo else (rm_res, None)

                        # ICF_RM_SCREEN_ONLY=1 records the margin without letting it
                        # touch the ensemble, so correlation screening stays honest.
                        if os.environ.get("ICF_RM_SCREEN_ONLY", "0") != "1":
                            logits = logits.clone()
                            logits[:, 0] -= 0.5 * rm_weight * rm_margin
                            logits[:, 1] += 0.5 * rm_weight * rm_margin

                # ---- §218 shape-family screening candidates -------------------
                # BD's entropy path normalises the eigenvalues (p = eig / eig.sum()),
                # discarding total variance; and every Location-family branch is a
                # projection or reweighting of the slide MEAN. These two candidates
                # occupy what is left:
                #   BS  = log total variance      -> pure scale (what BD normalises away)
                #   SH  = per-dim skew + kurtosis -> location- AND scale-invariant by
                #         construction, so it cannot restate BM/QA/DS or BS.
                bs_weight = float(os.environ.get("ICF_FIXED_HEAD_BS_WEIGHT", "0.0"))
                sh_weight = float(os.environ.get("ICF_FIXED_HEAD_SH_WEIGHT", "0.0"))
                if bs_weight != 0.0 or sh_weight != 0.0:
                    shp_eps = 1e-6
                    basis = inner._effective_covariance_projection()
                    labels_shp = episode_y[:n_context].long().to(device)
                    shp_idx = list(range(n_context)) + query_index.tolist()

                    if bs_weight != 0.0:
                        bs_dim = min(int(os.environ.get("ICF_BS_DIM", "256")), basis.shape[1])
                        bs_basis = basis[:, :bs_dim].float()

                        def bs_feat(b):
                            if bag_stats_cache is not None and id(b) in bag_stats_cache:
                                n_i, _, scatter = bag_stats_cache[id(b)]
                                tr = ((scatter.to(device).float() @ bs_basis) * bs_basis).sum() / float(n_i)
                            else:
                                v = b.to(device).float()
                                pr = (v - v.mean(dim=0, keepdim=True)) @ bs_basis
                                tr = pr.square().sum() / float(v.shape[0])
                            return tr.clamp_min(shp_eps).log().reshape(1)

                        f_bs = torch.stack([bs_feat(episode_bags[i]) for i in shp_idx])
                        bs_margin = _solve_kernel_ridge(
                            f_bs[:n_context], labels_shp, f_bs[n_context:],
                            kernel="linear", gamma=None, degree=2, coef0=1.0,
                            reg_lambda=float(os.environ.get("ICF_BS_LAMBDA", "1.0")),
                            return_loo=False,
                        )

                    if sh_weight != 0.0:
                        # §219 SH variants. Tokens are projected ONCE at the widest
                        # dim; every variant is a cheap derivation of that projection.
                        sh_wide = min(int(os.environ.get("ICF_SH_WIDE", "256")), basis.shape[1])
                        sh_narrow = min(int(os.environ.get("ICF_SH_DIM", "32")), sh_wide)
                        sh_lam = float(os.environ.get("ICF_SH_LAMBDA", "1.0"))
                        wide_basis = basis[:, :sh_wide].float()

                        def _q(sorted_p, frac):
                            n = sorted_p.shape[0]
                            pos = frac * (n - 1)
                            lo = int(math.floor(pos)); hi = min(lo + 1, n - 1)
                            w = pos - lo
                            return sorted_p[lo] * (1.0 - w) + sorted_p[hi] * w

                        def sh_all(b):
                            v = b.to(device).float()
                            pr = v @ wide_basis                      # [N, wide]
                            mu = pr.mean(dim=0, keepdim=True)
                            sd = pr.std(dim=0, keepdim=True).clamp_min(shp_eps)
                            z = (pr - mu) / sd
                            skew = z.pow(3).mean(dim=0)              # [wide]
                            kurt = z.pow(4).mean(dim=0) - 3.0        # [wide]

                            sp, _ = torch.sort(pr, dim=0)
                            q125, q250, q375 = _q(sp, .125), _q(sp, .25), _q(sp, .375)
                            q500 = _q(sp, .50)
                            q625, q750, q875 = _q(sp, .625), _q(sp, .75), _q(sp, .875)
                            iqr = (q750 - q250).clamp_min(shp_eps)
                            bowley = (q750 + q250 - 2.0 * q500) / iqr          # robust skew
                            moors = (q875 - q625 + q375 - q125) / iqr          # robust tail weight

                            # Joint shape: whiten tokens in the slide's OWN top-k basis,
                            # then describe the radius distribution. Location- and
                            # scale-invariant, and multivariate rather than marginal.
                            zc = pr[:, :sh_narrow] - mu[:, :sh_narrow]
                            cov = (zc.T @ zc) / float(zc.shape[0])
                            cov = 0.5 * (cov + cov.T)
                            ev, evec = torch.linalg.eigh(cov.double())
                            inv = (ev.clamp_min(1e-8).rsqrt())
                            wz = (zc.double() @ evec) * inv
                            r = wz.norm(dim=1).float()
                            rs, _ = torch.sort(r)
                            rmed = _q(rs, .50).clamp_min(shp_eps)
                            rz = (r - r.mean()) / r.std().clamp_min(shp_eps)
                            r_iqr = (_q(rs, .75) - _q(rs, .25)).clamp_min(shp_eps)
                            joint = torch.stack([
                                rz.pow(3).mean(), rz.pow(4).mean() - 3.0,
                                (_q(rs, .75) + _q(rs, .25) - 2.0 * _q(rs, .50)) / r_iqr,
                                (_q(rs, .875) - _q(rs, .625) + _q(rs, .375) - _q(rs, .125)) / r_iqr,
                                _q(rs, .10) / rmed, _q(rs, .90) / rmed, _q(rs, .99) / rmed,
                                r_iqr / rmed,
                            ])
                            n_ = sh_narrow
                            return {
                                "sh":   torch.cat([skew[:n_], kurt[:n_]]),
                                "shs":  skew[:n_],
                                "shk":  kurt[:n_],
                                "sh2":  torch.cat([skew, kurt]),
                                "shr":  torch.cat([bowley[:n_], moors[:n_]]),
                                "shr2": torch.cat([bowley, moors]),
                                "shj":  joint,
                            }

                        _feats = [sh_all(episode_bags[i]) for i in shp_idx]
                        _keys = ("sh", "shs", "shk", "sh2", "shr", "shr2", "shj")
                        _want = os.environ.get("ICF_SH_VARIANTS", ",".join(_keys)).split(",")
                        _out = {}
                        for _k in _keys:
                            if _k not in _want:
                                continue
                            _F = torch.stack([d[_k] for d in _feats])
                            _F = torch.nan_to_num(_F, nan=0.0, posinf=0.0, neginf=0.0)
                            _out[_k] = _solve_kernel_ridge(
                                _F[:n_context], labels_shp, _F[n_context:],
                                kernel="linear", gamma=None, degree=2, coef0=1.0,
                                reg_lambda=sh_lam, return_loo=False,
                            )
                        sh_margin = _out.get("sh")
                        sh_variant_margins = _out

                    if os.environ.get("ICF_SHAPE_SCREEN_ONLY", "1") != "1":
                        logits = logits.clone()
                        for _w, _m in ((bs_weight, locals().get("bs_margin")),
                                       (sh_weight, locals().get("sh_margin"))):
                            if _w != 0.0 and _m is not None:
                                logits[:, 0] -= 0.5 * _w * _m
                                logits[:, 1] += 0.5 * _w * _m

                bd_weight = float(os.environ.get("ICF_FIXED_HEAD_BD_WEIGHT", "0.0"))
                if bd_weight != 0.0:
                    bd_dim = int(os.environ.get("ICF_BD_DIM", "256"))
                    bd_metric = os.environ.get("ICF_BD_METRIC", "entropy")
                    bd_readout = os.environ.get("ICF_BD_READOUT", "ordered_typicality")
                    bd_separation_floor = float(os.environ.get("ICF_BD_SEPARATION_FLOOR", "1.0"))
                    bd_lambda = float(os.environ.get("ICF_BD_LAMBDA", "1.0"))
                    bd_eps = 1e-6
                    basis = inner._effective_covariance_projection()
                    dim = min(bd_dim, basis.shape[1])
                    bd_basis = basis[:, :dim].float()

                    def get_bag_feature(b):
                        if bag_stats_cache is not None and id(b) in bag_stats_cache:
                            stats = bag_stats_cache[id(b)]
                            n, _, scatter = stats
                            sc_dev = scatter.to(device).float()
                            if bd_metric == "trace":
                                tr = ((sc_dev @ bd_basis) * bd_basis).sum() / float(n)
                                return tr.clamp_min(bd_eps).log()
                            else:
                                S_proj = (bd_basis.T @ sc_dev @ bd_basis) / float(n)
                                S_proj = 0.5 * (S_proj + S_proj.T)
                        else:
                            vals = b.to(device).float()
                            centered = vals - vals.mean(dim=0, keepdim=True)
                            proj = centered @ bd_basis
                            if bd_metric == "trace":
                                tr = (proj.square()).sum() / float(vals.shape[0])
                                return tr.clamp_min(bd_eps).log()
                            else:
                                S_proj = (proj.T @ proj) / float(vals.shape[0])

                        if bd_metric == "entropy":
                            eigvals = torch.linalg.eigvalsh(S_proj.float()).clamp_min(bd_eps)
                            p = eigvals / eigvals.sum().clamp_min(bd_eps)
                            ent = -(p * torch.log(p.clamp_min(bd_eps))).sum()
                            if dim > 1:
                                ent = ent / math.log(dim)
                            return ent
                        else:
                            raise ValueError(f"Unknown bd_metric: {bd_metric!r}")

                    ctx_v = torch.stack([get_bag_feature(b) for b in episode_bags[:n_context]])
                    qry_v = torch.stack([get_bag_feature(episode_bags[i]) for i in query_index.tolist()])

                    labels = episode_y[:n_context].long().to(device)
                    if bd_readout == "ordered_typicality":
                        prototypes = torch.stack([ctx_v[labels == c].mean() for c in range(2)])
                        dispersions = torch.stack([
                            (ctx_v[labels == c] - prototypes[c]).square().mean().clamp_min(bd_eps)
                            for c in range(2)
                        ])
                        from src.models.dd_adaptive_rank import ordered_typicality_margin  # noqa: PLC0415
                        bd_margin = ordered_typicality_margin(
                            qry_v,
                            prototypes,
                            dispersions,
                            bd_eps,
                            bd_separation_floor,
                        )
                    elif bd_readout == "ridge":
                        centre = ctx_v.mean()
                        scale = (ctx_v - centre).square().mean().sqrt().clamp_min(bd_eps)
                        std_ctx = ((ctx_v - centre) / scale).unsqueeze(-1)
                        std_qry = ((qry_v - centre) / scale).unsqueeze(-1)

                        targets = torch.nn.functional.one_hot(labels, 2).float()
                        counts = torch.bincount(labels, minlength=2)
                        weight = counts.float().reciprocal()[labels]
                        total = weight.sum().clamp_min(1e-12)
                        feature_mean = (weight[:, None] * std_ctx).sum(0, keepdim=True) / total
                        target_mean = (weight[:, None] * targets).sum(0, keepdim=True) / total
                        root = weight.sqrt()[:, None]

                        design = (std_ctx - feature_mean) * root
                        centred_targets = (targets - target_mean) * root

                        gram_f32 = (design @ design.T).float()
                        targets_f32 = centred_targets.float()
                        size = gram_f32.shape[-1]
                        identity = torch.eye(size, device=device, dtype=torch.float32)
                        jitter = 0.0
                        dual = None
                        for _ in range(6):
                            try:
                                factor = torch.linalg.cholesky(gram_f32 + (bd_lambda + jitter) * identity)
                                dual = torch.cholesky_solve(targets_f32, factor)
                                break
                            except RuntimeError:
                                jitter = max(jitter * 10.0, 1e-6 * float(gram_f32.diagonal().abs().mean()) + 1e-12)
                        if dual is None:
                            dual = torch.linalg.lstsq(gram_f32 + (bd_lambda + jitter) * identity, targets_f32).solution

                        coefficients = design.T @ dual
                        intercept = target_mean - feature_mean @ coefficients
                        bd_logits = std_qry @ coefficients + intercept
                        bd_margin = bd_logits[:, 1] - bd_logits[:, 0]
                    else:
                        raise ValueError(f"Unknown bd_readout: {bd_readout!r}")

                    logits = logits.clone()
                    logits[:, 0] -= 0.5 * bd_weight * bd_margin
                    logits[:, 1] += 0.5 * bd_weight * bd_margin

                qa_weight = float(os.environ.get("ICF_FIXED_HEAD_QA_WEIGHT", "0.0"))
                if qa_weight != 0.0:
                    qa_dim = int(os.environ.get("ICF_QA_DIM", "32"))
                    qa_lambda = float(os.environ.get("ICF_QA_LAMBDA", "1.0"))
                    qa_quantiles = torch.tensor([0.05, 0.10, 0.90, 0.95], device=device, dtype=torch.float32)
                    basis = inner._effective_covariance_projection()
                    dim = min(qa_dim, basis.shape[1])
                    qa_basis = basis[:, :dim].float()

                    def extract_bag_quantiles(b):
                        z = (b.float().to(device) @ qa_basis).float()
                        q = torch.quantile(z, qa_quantiles, dim=0)
                        return q.flatten()

                    ctx_qas = torch.stack([extract_bag_quantiles(b) for b in episode_bags[:n_context]])
                    qry_qas = torch.stack([extract_bag_quantiles(episode_bags[i]) for i in query_index.tolist()])

                    labels = episode_y[:n_context].long().to(device)
                    krr_kernel = os.environ.get("ICF_KRR_KERNEL", "linear")
                    krr_gamma = float(os.environ["ICF_KRR_GAMMA"]) if "ICF_KRR_GAMMA" in os.environ else None
                    krr_degree = int(os.environ.get("ICF_KRR_DEGREE", "2"))
                    krr_coef0 = float(os.environ.get("ICF_KRR_COEF0", "1.0"))
                    qa_res = _solve_kernel_ridge(
                        ctx_qas, labels, qry_qas,
                        kernel=os.environ.get("ICF_QA_KERNEL", krr_kernel),
                        gamma=krr_gamma,
                        degree=krr_degree,
                        coef0=krr_coef0,
                        reg_lambda=qa_lambda,
                        return_loo=do_context_loo,
                    )
                    qa_margin, loo_qa = (qa_res[0], qa_res[1]) if do_context_loo else (qa_res, None)

                    logits = logits.clone()
                    logits[:, 0] -= 0.5 * qa_weight * qa_margin
                    logits[:, 1] += 0.5 * qa_weight * qa_margin

                ds_weight = float(os.environ.get("ICF_FIXED_HEAD_DS_WEIGHT", "0.0"))
                if ds_weight != 0.0:
                    ds_dim = int(os.environ.get("ICF_DS_DIM", "32"))
                    ds_lambda = float(os.environ.get("ICF_DS_LAMBDA", "1.0"))
                    ds_temperature = float(os.environ.get("ICF_DS_TEMPERATURE", "1.0"))
                    ds_tokens = int(os.environ.get("ICF_DS_TOKENS", "256"))
                    basis = inner._effective_covariance_projection()
                    dim = min(ds_dim, basis.shape[1])
                    ds_basis = basis[:, :dim].float()

                    # 1. Project all context and query bags
                    ctx_proj = [(b.float().to(device) @ ds_basis).float() for b in episode_bags[:n_context]]
                    qry_proj = [(episode_bags[i].float().to(device) @ ds_basis).float() for i in query_index.tolist()]

                    # 2. Select K centroids from context cells
                    sampled_cells = []
                    for p in ctx_proj:
                        n_c = p.shape[0]
                        if n_c > 0:
                            idx = torch.linspace(0, n_c - 1, min(n_c, 64), device=device).long()
                            sampled_cells.append(p[idx])
                    all_c = torch.cat(sampled_cells, dim=0) if sampled_cells else torch.zeros(1, dim, device=device)
                    K = min(ds_tokens, all_c.shape[0])
                    if all_c.shape[0] > K:
                        stride = all_c.shape[0] / K
                        centroids = all_c[(torch.arange(K, device=device) * stride).long()]
                    else:
                        centroids = all_c
                    centroids = torch.nn.functional.normalize(centroids, dim=-1)

                    # 3. Soft cluster assignments
                    def get_ds_assignments(proj_bags):
                        abundances = []
                        patch_assignments = []
                        for p in proj_bags:
                            p_norm = torch.nn.functional.normalize(p, dim=-1)
                            sim = p_norm @ centroids.T
                            soft_p = torch.nn.functional.softmax(sim * 5.0, dim=-1)
                            a = soft_p.mean(dim=0)
                            abundances.append(a)
                            patch_assignments.append(soft_p)
                        return torch.stack(abundances), patch_assignments

                    # Sub-bag Data Augmentation:
                    # mode: "none" (baseline), "context" (Method 1), "query" (Method 2)
                    # Sub-bag Data Augmentation:
                    # mode: "none" (baseline), "context" (Method 1), "query" (Method 2), "auto_loo" (In-Episode LOO Dual Selection)
                    ds_aug_mode = os.environ.get("ICF_DS_AUG_MODE", "none").lower()
                    ds_aug_s = int(os.environ.get("ICF_DS_AUG_S", "5"))
                    ds_aug_fraction = float(os.environ.get("ICF_DS_AUG_FRACTION", "0.7"))
                    ds_aug_seed = int(os.environ.get("ICF_DS_AUG_SEED", "42"))

                    labels = episode_y[:n_context].long().to(device)
                    krr_kernel = os.environ.get("ICF_KRR_KERNEL", "linear")
                    krr_gamma = float(os.environ["ICF_KRR_GAMMA"]) if "ICF_KRR_GAMMA" in os.environ else None
                    krr_degree = int(os.environ.get("ICF_KRR_DEGREE", "2"))
                    krr_coef0 = float(os.environ.get("ICF_KRR_COEF0", "1.0"))

                    if ds_aug_mode == "auto_loo":
                        # 1. Full Bag representation
                        ctx_ab_f, ctx_as_f = get_ds_assignments(ctx_proj)
                        qry_ab_f, qry_as_f = get_ds_assignments(qry_proj)
                        eps = 1e-5
                        m1 = (labels == 1)
                        m0 = (labels == 0)
                        a1 = ctx_ab_f[m1].mean(dim=0) if m1.any() else ctx_ab_f.mean(dim=0)
                        a0 = ctx_ab_f[m0].mean(dim=0) if m0.any() else ctx_ab_f.mean(dim=0)
                        s_f = torch.log((a1 + eps) / (a0 + eps)).abs()

                        def ext_mean(proj_bags, assignments, s_weights):
                            feats = []
                            for p, soft_p in zip(proj_bags, assignments):
                                u = soft_p @ s_weights
                                u_std = u.std().clamp_min(1e-6)
                                w = torch.nn.functional.softmax(ds_temperature * (u - u.mean()) / u_std, dim=0)
                                feats.append((w.unsqueeze(-1) * p).sum(dim=0))
                            return torch.stack(feats)

                        ctx_ds_f = ext_mean(ctx_proj, ctx_as_f, s_f)
                        qry_ds_f = ext_mean(qry_proj, qry_as_f, s_f)
                        m_f, loo_f = _solve_kernel_ridge(
                            ctx_ds_f, labels, qry_ds_f,
                            kernel=os.environ.get("ICF_DS_KERNEL", krr_kernel),
                            gamma=krr_gamma, degree=krr_degree, coef0=krr_coef0,
                            reg_lambda=ds_lambda, return_loo=True,
                        )
                        sc_f = _fast_context_auroc(loo_f, labels)

                        # 2. Sub-bag representation (S=5, frac=0.7)
                        base_seed_val = ds_aug_seed + (seed if "seed" in locals() and isinstance(seed, int) else 0) * 10000
                        def make_sub_features(bags):
                            sub_feats = []
                            for i, b in enumerate(bags):
                                n_b = b.shape[0]
                                k_sub = max(1, int(n_b * ds_aug_fraction))
                                b_subs = []
                                for s_idx in range(ds_aug_s):
                                    gen = torch.Generator(device="cpu").manual_seed(base_seed_val + i * 100 + s_idx)
                                    perm = torch.randperm(n_b, generator=gen)[:k_sub]
                                    b_subs.append(b[perm])
                                _, sub_as = get_ds_assignments(b_subs)
                                sub_feats.append(ext_mean(b_subs, sub_as, s_f).mean(dim=0))
                            return torch.stack(sub_feats)

                        ctx_ds_s = make_sub_features(ctx_proj)
                        qry_ds_s = make_sub_features(qry_proj)
                        m_s, loo_s = _solve_kernel_ridge(
                            ctx_ds_s, labels, qry_ds_s,
                            kernel=os.environ.get("ICF_DS_KERNEL", krr_kernel),
                            gamma=krr_gamma, degree=krr_degree, coef0=krr_coef0,
                            reg_lambda=ds_lambda, return_loo=True,
                        )
                        sc_s = _fast_context_auroc(loo_s, labels)

                        if sc_s >= sc_f:
                            ds_margin = m_s
                            print(f"  [DS LOO Auto-Switch] Selected Sub ({sc_s:.3f} >= {sc_f:.3f})", flush=True)
                        else:
                            ds_margin = m_f
                            print(f"  [DS LOO Auto-Switch] Selected Full ({sc_f:.3f} > {sc_s:.3f})", flush=True)

                    else:
                        # 1. Context Augmentation (Method 1)
                        if ds_aug_mode == "context" and ds_aug_s > 1 and ds_aug_fraction < 1.0:
                            ctx_proj_aug = []
                            ctx_labels_aug = []
                            base_seed_val = ds_aug_seed + (seed if "seed" in locals() and isinstance(seed, int) else 0) * 10000
                            for i, p in enumerate(ctx_proj):
                                n_p = p.shape[0]
                                k_sub = max(1, int(n_p * ds_aug_fraction))
                                lbl_i = labels[i]
                                for s_idx in range(ds_aug_s):
                                    gen = torch.Generator(device="cpu").manual_seed(base_seed_val + i * 100 + s_idx)
                                    perm = torch.randperm(n_p, generator=gen)[:k_sub]
                                    ctx_proj_aug.append(p[perm])
                                    ctx_labels_aug.append(lbl_i)
                            ctx_proj_eval = ctx_proj_aug
                            labels_eval = torch.stack(ctx_labels_aug)
                        else:
                            ctx_proj_eval = ctx_proj
                            labels_eval = labels

                        # 2. Query TTA Augmentation (Method 2)
                        if ds_aug_mode == "query" and ds_aug_s > 1 and ds_aug_fraction < 1.0:
                            qry_proj_eval = []
                            qry_slide_indices = []
                            base_seed_val = ds_aug_seed + (seed if "seed" in locals() and isinstance(seed, int) else 0) * 10000 + 500000
                            for j, q in enumerate(qry_proj):
                                n_q = q.shape[0]
                                k_sub = max(1, int(n_q * ds_aug_fraction))
                                # Anchor view: full bag
                                qry_proj_eval.append(q)
                                qry_slide_indices.append(j)
                                # Sub-bag views:
                                for s_idx in range(ds_aug_s):
                                    gen = torch.Generator(device="cpu").manual_seed(base_seed_val + j * 100 + s_idx)
                                    perm = torch.randperm(n_q, generator=gen)[:k_sub]
                                    qry_proj_eval.append(q[perm])
                                    qry_slide_indices.append(j)
                            qry_slide_indices = torch.tensor(qry_slide_indices, device=device)
                        else:
                            qry_proj_eval = qry_proj
                            qry_slide_indices = None

                        ctx_abundances, ctx_assignments = get_ds_assignments(ctx_proj_eval)
                        qry_abundances, qry_assignments = get_ds_assignments(qry_proj_eval)

                        # 4. Salience log-odds
                        eps = 1e-5
                        mask1 = (labels_eval == 1)
                        mask0 = (labels_eval == 0)
                        a1 = ctx_abundances[mask1].mean(dim=0) if mask1.any() else ctx_abundances.mean(dim=0)
                        a0 = ctx_abundances[mask0].mean(dim=0) if mask0.any() else ctx_abundances.mean(dim=0)
                        s = torch.log((a1 + eps) / (a0 + eps))
                        s_abs = s.abs()

                        # 5. Denoised bag mean
                        def extract_denoised_mean(proj_bags, assignments):
                            feats = []
                            for p, soft_p in zip(proj_bags, assignments):
                                u = soft_p @ s_abs
                                u_std = u.std().clamp_min(1e-6)
                                w = torch.nn.functional.softmax(ds_temperature * (u - u.mean()) / u_std, dim=0)
                                z_denoised = (w.unsqueeze(-1) * p).sum(dim=0)
                                feats.append(z_denoised)
                            return torch.stack(feats)

                        ds_anchor_fraction = float(os.environ.get("ICF_DS_ANCHOR_FRACTION", "0.15"))
                        if ds_aug_mode == "salience_anchor" and ds_aug_s > 1 and ds_aug_fraction < 1.0:
                            base_seed_val = ds_aug_seed + (seed if "seed" in locals() and isinstance(seed, int) else 0) * 10000

                            def extract_anchor_sub_mean(proj_bags, assignments, seed_offset):
                                feats = []
                                for i, (p, soft_p) in enumerate(zip(proj_bags, assignments)):
                                    n_p = p.shape[0]
                                    k_anchor = max(1, min(n_p, int(n_p * ds_anchor_fraction)))
                                    k_bg = n_p - k_anchor
                                    u = soft_p @ s_abs
                                    anchor_idx = torch.topk(u, k=k_anchor, largest=True).indices
                                    if k_bg > 0:
                                        bg_idx = torch.topk(u, k=k_bg, largest=False).indices
                                        k_bg_sub = max(1, int(k_bg * ds_aug_fraction))
                                    else:
                                        bg_idx = None
                                        k_bg_sub = 0

                                    s_feats = []
                                    for s_idx in range(ds_aug_s):
                                        if k_bg_sub > 0:
                                            gen = torch.Generator(device="cpu").manual_seed(seed_offset + i * 100 + s_idx)
                                            perm = torch.randperm(k_bg, generator=gen)[:k_bg_sub]
                                            sub_idx = torch.cat([anchor_idx, bg_idx[perm]])
                                        else:
                                            sub_idx = anchor_idx
                                        sub_p = p[sub_idx]
                                        p_norm = torch.nn.functional.normalize(sub_p, dim=-1)
                                        sub_soft = torch.nn.functional.softmax((p_norm @ centroids.T) * 5.0, dim=-1)
                                        sub_u = sub_soft @ s_abs
                                        u_std = sub_u.std().clamp_min(1e-6)
                                        w = torch.nn.functional.softmax(ds_temperature * (sub_u - sub_u.mean()) / u_std, dim=0)
                                        s_feats.append((w.unsqueeze(-1) * sub_p).sum(dim=0))
                                    feats.append(torch.stack(s_feats).mean(dim=0))
                                return torch.stack(feats)

                            ctx_ds = extract_anchor_sub_mean(ctx_proj_eval, ctx_assignments, base_seed_val)
                            qry_ds = extract_anchor_sub_mean(qry_proj_eval, qry_assignments, base_seed_val + 500000)
                        else:
                            ctx_ds = extract_denoised_mean(ctx_proj_eval, ctx_assignments)
                            qry_ds = extract_denoised_mean(qry_proj_eval, qry_assignments)

                        # 6. Class-balanced kernel ridge
                        raw_ds_res = _solve_kernel_ridge(
                            ctx_ds, labels_eval, qry_ds,
                            kernel=os.environ.get("ICF_DS_KERNEL", krr_kernel),
                            gamma=krr_gamma,
                            degree=krr_degree,
                            coef0=krr_coef0,
                            reg_lambda=ds_lambda,
                            return_loo=do_context_loo,
                        )
                        raw_ds_margin, loo_ds = (raw_ds_res[0], raw_ds_res[1]) if do_context_loo else (raw_ds_res, None)

                        # If query TTA was used, aggregate multiple sub-bag margins per query slide
                        if qry_slide_indices is not None:
                            n_orig_qry = len(qry_proj)
                            raw_probs = torch.sigmoid(raw_ds_margin)
                            agg_probs = torch.zeros(n_orig_qry, device=device, dtype=raw_probs.dtype)
                            counts = torch.zeros(n_orig_qry, device=device, dtype=raw_probs.dtype)
                            agg_probs.scatter_add_(0, qry_slide_indices, raw_probs)
                            counts.scatter_add_(0, qry_slide_indices, torch.ones_like(raw_probs))
                            mean_probs = agg_probs / counts.clamp_min(1.0)
                            ds_margin = torch.logit(mean_probs.clamp(1e-6, 1.0 - 1e-6))
                        else:
                            ds_margin = raw_ds_margin

                    logits = logits.clone()
                    logits[:, 0] -= 0.5 * ds_weight * ds_margin
                    logits[:, 1] += 0.5 * ds_weight * ds_margin

                lr_weight = float(os.environ.get("ICF_FIXED_HEAD_LR_WEIGHT", "0.0"))
                if lr_weight != 0.0:
                    lr_dim = int(os.environ.get("ICF_LR_DIM", "32"))
                    lr_lambda = float(os.environ.get("ICF_LR_LAMBDA", "1.0"))
                    lr_tau = float(os.environ.get("ICF_LR_TAU", "5.0"))
                    lr_topk_fraction = float(os.environ.get("ICF_LR_TOPK_FRACTION", "0.05"))
                    lr_topk_min = int(os.environ.get("ICF_LR_TOPK_MIN", "4"))
                    lr_topk_max = int(os.environ.get("ICF_LR_TOPK_MAX", "64"))
                    lr_patches_per_ctx = int(os.environ.get("ICF_LR_PATCHES_PER_CTX", "64"))
                    basis = inner._effective_covariance_projection()
                    dim = min(lr_dim, basis.shape[1])
                    lr_basis = basis[:, :dim].float()

                    # 1. Project bags
                    ctx_proj = [(b.float().to(device) @ lr_basis).float() for b in episode_bags[:n_context]]
                    qry_proj = [(episode_bags[i].float().to(device) @ lr_basis).float() for i in query_index.tolist()]
                    labels = episode_y[:n_context].long().to(device)

                    # 2. Build Class 0 and Class 1 patch memory banks
                    bank_0, bank_1 = [], []
                    for p, y in zip(ctx_proj, labels):
                        n_c = p.shape[0]
                        if n_c == 0:
                            continue
                        n_sample = min(n_c, lr_patches_per_ctx)
                        idx = torch.linspace(0, n_c - 1, n_sample, device=device).long()
                        sampled = p[idx]
                        if y == 1:
                            bank_1.append(sampled)
                        else:
                            bank_0.append(sampled)

                    if bank_0 and bank_1:
                        P0 = torch.cat(bank_0, dim=0)
                        P1 = torch.cat(bank_1, dim=0)
                        P0_norm = torch.nn.functional.normalize(P0, dim=-1)
                        P1_norm = torch.nn.functional.normalize(P1, dim=-1)

                        def get_slide_lr_features(proj_bags):
                            feats = []
                            for bag in proj_bags:
                                n_c = bag.shape[0]
                                if n_c == 0:
                                    feats.append(torch.zeros(dim + 1, device=device))
                                    continue
                                bag_norm = torch.nn.functional.normalize(bag, dim=-1)
                                sim1 = bag_norm @ P1_norm.T
                                sim0 = bag_norm @ P0_norm.T

                                score1 = torch.logsumexp(sim1 * lr_tau, dim=-1) - torch.log(torch.tensor(P1_norm.shape[0], dtype=torch.float32, device=device))
                                score0 = torch.logsumexp(sim0 * lr_tau, dim=-1) - torch.log(torch.tensor(P0_norm.shape[0], dtype=torch.float32, device=device))
                                lr = score1 - score0

                                k = max(lr_topk_min, min(lr_topk_max, int(n_c * lr_topk_fraction)))
                                k = min(k, n_c)

                                topk_vals, topk_idx = torch.topk(lr, k=k, largest=True)
                                botk_vals, botk_idx = torch.topk(lr, k=k, largest=False)

                                z_plus = bag[topk_idx].mean(dim=0)
                                z_minus = bag[botk_idx].mean(dim=0)

                                delta_z = z_plus - z_minus
                                e_scalar = 0.5 * (topk_vals.mean() + botk_vals.mean())

                                v_i = torch.cat([delta_z, e_scalar.unsqueeze(0)], dim=-1)
                                feats.append(v_i)
                            return torch.stack(feats)

                        ctx_lr = get_slide_lr_features(ctx_proj)
                        qry_lr = get_slide_lr_features(qry_proj)

                        lr_margin = _solve_kernel_ridge(
                            ctx_lr, labels, qry_lr,
                            kernel="linear",
                            reg_lambda=lr_lambda,
                        )

                        logits = logits.clone()
                        logits[:, 0] -= 0.5 * lr_weight * lr_margin
                        logits[:, 1] += 0.5 * lr_weight * lr_margin
                    else:
                        lr_margin = torch.zeros(len(query_index), device=device)
                else:
                    lr_margin = torch.zeros(len(query_index), device=device)

                de_weight = float(os.environ.get("ICF_FIXED_HEAD_DE_WEIGHT", "0.0"))
                if de_weight != 0.0:
                    de_dim = int(os.environ.get("ICF_DE_DIM", "32"))
                    de_lambda = float(os.environ.get("ICF_DE_LAMBDA", "1.0"))
                    de_topk_fraction = float(os.environ.get("ICF_DE_TOPK_FRACTION", "0.05"))
                    de_topk_min = int(os.environ.get("ICF_DE_TOPK_MIN", "4"))
                    de_topk_max = int(os.environ.get("ICF_DE_TOPK_MAX", "64"))
                    basis = inner._effective_covariance_projection()
                    dim = min(de_dim, basis.shape[1])
                    de_basis = basis[:, :dim].float()

                    ctx_proj = [(b.float().to(device) @ de_basis).float() for b in episode_bags[:n_context]]
                    qry_proj = [(episode_bags[i].float().to(device) @ de_basis).float() for i in query_index.tolist()]
                    labels = episode_y[:n_context].long().to(device)

                    ctx_means = torch.stack([p.mean(dim=0) for p in ctx_proj])
                    if (labels == 0).sum() > 0 and (labels == 1).sum() > 0:
                        mu0 = ctx_means[labels == 0].mean(dim=0)
                        mu1 = ctx_means[labels == 1].mean(dim=0)
                        w_contrast = mu1 - mu0
                        w_dir = w_contrast / w_contrast.norm().clamp_min(1e-12)

                        def extract_de_vector(p_bag):
                            n_c = p_bag.shape[0]
                            if n_c == 0:
                                return torch.zeros(dim + 1, device=device)
                            scores = p_bag @ w_dir
                            k = max(de_topk_min, min(de_topk_max, int(n_c * de_topk_fraction)))
                            k = min(k, n_c)
                            topk_vals, topk_idx = torch.topk(scores, k=k, largest=True)
                            botk_vals, botk_idx = torch.topk(scores, k=k, largest=False)
                            z_plus = p_bag[topk_idx].mean(dim=0)
                            z_minus = p_bag[botk_idx].mean(dim=0)
                            delta_z = z_plus - z_minus
                            score_diff = 0.5 * (topk_vals.mean() + botk_vals.mean())
                            return torch.cat([delta_z, score_diff.unsqueeze(0)], dim=-1)

                        ctx_feats = torch.stack([extract_de_vector(p) for p in ctx_proj])
                        qry_feats = torch.stack([extract_de_vector(p) for p in qry_proj])
                        de_margin = _solve_kernel_ridge(
                            ctx_feats, labels, qry_feats,
                            kernel="linear",
                            reg_lambda=de_lambda,
                        )
                        logits = logits.clone()
                        logits[:, 0] -= 0.5 * de_weight * de_margin
                        logits[:, 1] += 0.5 * de_weight * de_margin
                    else:
                        de_margin = torch.zeros(len(query_index), device=device)
                else:
                    de_margin = torch.zeros(len(query_index), device=device)

                sw_weight = float(os.environ.get("ICF_FIXED_HEAD_SW_WEIGHT", "0.0"))
                if sw_weight != 0.0:
                    sw_dim = int(os.environ.get("ICF_SW_DIM", "32"))
                    sw_lambda = float(os.environ.get("ICF_SW_LAMBDA", "1.0"))
                    sw_num_slices = int(os.environ.get("ICF_SW_NUM_SLICES", "32"))
                    sw_num_quantiles = int(os.environ.get("ICF_SW_NUM_QUANTILES", "32"))
                    basis = inner._effective_covariance_projection()
                    dim = min(sw_dim, basis.shape[1])
                    sw_basis = basis[:, :dim].float()

                    g = torch.Generator(device="cpu").manual_seed(42)
                    rand_dirs = torch.randn(dim, sw_num_slices, generator=g, dtype=torch.float32)
                    q_dirs, _ = torch.linalg.qr(rand_dirs)
                    slice_dirs = q_dirs.to(device=device)

                    q_levels = torch.linspace(0.5 / sw_num_quantiles, 1.0 - 0.5 / sw_num_quantiles, sw_num_quantiles, device=device)
                    labels = episode_y[:n_context].long().to(device)

                    def extract_sw_profile(bag):
                        proj = bag.float().to(device) @ sw_basis
                        n_c = proj.shape[0]
                        if n_c == 0:
                            return torch.zeros(sw_num_slices * sw_num_quantiles, device=device)
                        slices = proj @ slice_dirs
                        sorted_slices, _ = torch.sort(slices, dim=0)
                        indices = (q_levels * (n_c - 1)).clamp(0, n_c - 1)
                        low_idx = indices.floor().long()
                        high_idx = indices.ceil().long()
                        weights = (indices - low_idx.float())[:, None]
                        low_vals = sorted_slices[low_idx, :]
                        high_vals = sorted_slices[high_idx, :]
                        quantiles = (1.0 - weights) * low_vals + weights * high_vals
                        return quantiles.flatten()

                    ctx_feats = torch.stack([extract_sw_profile(b) for b in episode_bags[:n_context]])
                    qry_feats = torch.stack([extract_sw_profile(episode_bags[i]) for i in query_index.tolist()])
                    sw_margin = _solve_kernel_ridge(
                        ctx_feats, labels, qry_feats,
                        kernel="linear",
                        reg_lambda=sw_lambda,
                    )
                    logits = logits.clone()
                    logits[:, 0] -= 0.5 * sw_weight * sw_margin
                    logits[:, 1] += 0.5 * sw_weight * sw_margin
                else:
                    sw_margin = torch.zeros(len(query_index), device=device)

        finally:
            if basis_mode in ("pca", "pca_within", "fisher", "fisher_within") and saved_projection is not None:
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
            if saved_llr_features is not None:
                inner._dd_distance_features = saved_llr_features
            if saved_relative_features is not None:
                inner._dd_distance_features = saved_relative_features
            if saved_ordered_features is not None:
                inner._dd_distance_features = saved_ordered_features
            if saved_normalize is not None:
                inner._normalize_descriptors = saved_normalize
        aggregation = os.environ.get("ICF_AGGREGATION", "soft_voting")
        branch_margins = getattr(inner, "_last_branch_margins", {})
        m_cv = branch_margins.get("cv", torch.zeros(len(test_ids))).to(device)
        m_dd = (-branch_margins.get("dd", torch.zeros(len(test_ids)))).to(device)
        m_ct = (-branch_margins.get("ct", torch.zeros(len(test_ids)))).to(device)
        m_bm = bm_margin.detach() if ("bm_margin" in locals() and bm_margin is not None) else torch.zeros(len(test_ids), device=device)
        m_rm = rm_margin.detach() if ("rm_margin" in locals() and rm_margin is not None) else None
        m_bs = bs_margin.detach() if ("bs_margin" in locals() and bs_margin is not None) else None
        m_sh = sh_margin.detach() if ("sh_margin" in locals() and sh_margin is not None) else None
        m_bd = bd_margin.detach() if ("bd_margin" in locals() and bd_margin is not None) else torch.zeros(len(test_ids), device=device)
        m_qa = qa_margin.detach() if ("qa_margin" in locals() and qa_margin is not None) else torch.zeros(len(test_ids), device=device)
        m_ds = ds_margin.detach() if ("ds_margin" in locals() and ds_margin is not None) else torch.zeros(len(test_ids), device=device)
        m_lr = lr_margin.detach() if ("lr_margin" in locals() and lr_margin is not None) else torch.zeros(len(test_ids), device=device)
        m_de = de_margin.detach() if ("de_margin" in locals() and de_margin is not None) else torch.zeros(len(test_ids), device=device)
        m_sw = sw_margin.detach() if ("sw_margin" in locals() and sw_margin is not None) else torch.zeros(len(test_ids), device=device)

        if aggregation == "trimmed_mean":
            active_probs = []
            if cv_weight != 0.0:
                active_probs.append(torch.sigmoid(m_cv.float()))
            if dd_weight != 0.0:
                active_probs.append(torch.sigmoid(m_dd.float()))
            if ct_weight != 0.0:
                active_probs.append(torch.sigmoid(m_ct.float()))
            if bm_weight != 0.0:
                active_probs.append(torch.sigmoid(m_bm.float()))
            if bd_weight != 0.0:
                active_probs.append(torch.sigmoid(m_bd.float()))
            if qa_weight != 0.0:
                active_probs.append(torch.sigmoid(m_qa.float()))
            if ds_weight != 0.0:
                active_probs.append(torch.sigmoid(m_ds.float()))
            if lr_weight != 0.0:
                active_probs.append(torch.sigmoid(m_lr.float()))
            if de_weight != 0.0:
                active_probs.append(torch.sigmoid(m_de.float()))
            if sw_weight != 0.0:
                active_probs.append(torch.sigmoid(m_sw.float()))

            if len(active_probs) >= 3:
                stacked = torch.stack(active_probs, dim=-1)
                sum_p = torch.sum(stacked, dim=-1)
                min_p = torch.min(stacked, dim=-1).values
                max_p = torch.max(stacked, dim=-1).values
                scores = (sum_p - min_p - max_p) / (len(active_probs) - 2)
            elif active_probs:
                scores = sum(active_probs) / len(active_probs)
            else:
                scores = torch.softmax(logits.float(), dim=-1)[:, 1]
        elif aggregation == "soft_voting":
            active_pairs = []
            if cv_weight != 0.0:
                active_pairs.append((cv_weight, m_cv))
            if dd_weight != 0.0:
                active_pairs.append((dd_weight, m_dd))
            if ct_weight != 0.0:
                active_pairs.append((ct_weight, m_ct))
            if bm_weight != 0.0:
                active_pairs.append((bm_weight, m_bm))
            if bd_weight != 0.0:
                active_pairs.append((bd_weight, m_bd))
            if qa_weight != 0.0:
                active_pairs.append((qa_weight, m_qa))
            if ds_weight != 0.0:
                active_pairs.append((ds_weight, m_ds))
            if lr_weight != 0.0:
                active_pairs.append((lr_weight, m_lr))
            if de_weight != 0.0:
                active_pairs.append((de_weight, m_de))
            if sw_weight != 0.0:
                active_pairs.append((sw_weight, m_sw))

            if active_pairs:
                total_weight = sum(w for w, _ in active_pairs)
                scores = sum(w * torch.sigmoid(m.float()) for w, m in active_pairs) / total_weight
            else:
                scores = torch.softmax(logits.float(), dim=-1)[:, 1]
        elif aggregation == "hard_gated":
            active_probs = []
            if cv_weight != 0.0:
                active_probs.append(torch.sigmoid(m_cv.float()))
            if dd_weight != 0.0:
                active_probs.append(torch.sigmoid(m_dd.float()))
            if ct_weight != 0.0:
                active_probs.append(torch.sigmoid(m_ct.float()))
            if bm_weight != 0.0:
                active_probs.append(torch.sigmoid(m_bm.float()))
            if bd_weight != 0.0:
                active_probs.append(torch.sigmoid(m_bd.float()))
            if qa_weight != 0.0:
                active_probs.append(torch.sigmoid(m_qa.float()))
            if ds_weight != 0.0:
                active_probs.append(torch.sigmoid(m_ds.float()))
            if lr_weight != 0.0:
                active_probs.append(torch.sigmoid(m_lr.float()))
            if de_weight != 0.0:
                active_probs.append(torch.sigmoid(m_de.float()))
            if sw_weight != 0.0:
                active_probs.append(torch.sigmoid(m_sw.float()))

            if active_probs:
                stacked = torch.stack(active_probs, dim=-1)  # [N, B]
                tau = float(os.environ.get("ICF_GATED_TAU", "0.05"))
                c = (stacked - 0.5).abs()
                mask = (c >= tau).float()
                has_active = (mask.sum(dim=-1, keepdim=True) > 0)
                weights = torch.where(has_active, mask, torch.ones_like(mask))
                scores = (weights * stacked).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
            else:
                scores = torch.softmax(logits.float(), dim=-1)[:, 1]
        elif aggregation == "adaptive_trimmed":
            active_probs = []
            if cv_weight != 0.0:
                active_probs.append(torch.sigmoid(m_cv.float()))
            if dd_weight != 0.0:
                active_probs.append(torch.sigmoid(m_dd.float()))
            if ct_weight != 0.0:
                active_probs.append(torch.sigmoid(m_ct.float()))
            if bm_weight != 0.0:
                active_probs.append(torch.sigmoid(m_bm.float()))
            if bd_weight != 0.0:
                active_probs.append(torch.sigmoid(m_bd.float()))
            if qa_weight != 0.0:
                active_probs.append(torch.sigmoid(m_qa.float()))
            if ds_weight != 0.0:
                active_probs.append(torch.sigmoid(m_ds.float()))
            if lr_weight != 0.0:
                active_probs.append(torch.sigmoid(m_lr.float()))
            if de_weight != 0.0:
                active_probs.append(torch.sigmoid(m_de.float()))
            if sw_weight != 0.0:
                active_probs.append(torch.sigmoid(m_sw.float()))

            if active_probs:
                stacked = torch.stack(active_probs, dim=-1)  # [N, B]
                B = stacked.shape[-1]
                if B < 3:
                    scores = stacked.mean(dim=-1)
                else:
                    sorted_p, _ = torch.sort(stacked, dim=-1)
                    c = (stacked - 0.5).abs()
                    c_med = torch.median(c, dim=-1).values
                    min_p = sorted_p[:, 0]
                    max_p = sorted_p[:, -1]
                    c_min = (min_p - 0.5).abs()
                    c_max = (max_p - 0.5).abs()

                    tau = float(os.environ.get("ICF_ADAPTIVE_TAU", "0.08"))
                    ratio = float(os.environ.get("ICF_ADAPTIVE_RATIO", "1.5"))

                    drop_min = (c_min <= ratio * c_med) | (c_min <= tau)
                    drop_max = (c_max <= ratio * c_med) | (c_max <= tau)

                    sum_all = sorted_p.sum(dim=-1)
                    count_all = torch.full_like(sum_all, float(B))
                    sum_trimmed = sum_all - torch.where(drop_min, min_p, torch.zeros_like(min_p)) - torch.where(drop_max, max_p, torch.zeros_like(max_p))
                    count_trimmed = count_all - drop_min.float() - drop_max.float()
                    scores = sum_trimmed / count_trimmed.clamp_min(1.0)
            else:
                scores = torch.softmax(logits.float(), dim=-1)[:, 1]
        elif aggregation.startswith("context_loo"):
            branch_pool = []
            context_labels = episode_y[:n_context].long().to(device)
            if cv_weight != 0.0:
                branch_pool.append(("cv", cv_weight, m_cv, None))
            if dd_weight != 0.0:
                branch_pool.append(("dd", dd_weight, m_dd, None))
            if ct_weight != 0.0:
                branch_pool.append(("ct", ct_weight, m_ct, None))
            if bm_weight != 0.0:
                branch_pool.append(("bm", bm_weight, m_bm, loo_bm if "loo_bm" in locals() else None))
            if bd_weight != 0.0:
                branch_pool.append(("bd", bd_weight, m_bd, loo_bd if "loo_bd" in locals() else None))
            if qa_weight != 0.0:
                branch_pool.append(("qa", qa_weight, m_qa, loo_qa if "loo_qa" in locals() else None))
            if ds_weight != 0.0:
                branch_pool.append(("ds", ds_weight, m_ds, loo_ds if "loo_ds" in locals() else None))
            if lr_weight != 0.0:
                branch_pool.append(("lr", lr_weight, m_lr, None))
            if de_weight != 0.0:
                branch_pool.append(("de", de_weight, m_de, loo_de if "loo_de" in locals() else None))
            if sw_weight != 0.0:
                branch_pool.append(("sw", sw_weight, m_sw, loo_sw if "loo_sw" in locals() else None))

            gamma = float(os.environ.get("ICF_LOO_GAMMA", "2.0"))
            floor = float(os.environ.get("ICF_LOO_FLOOR", "0.50"))

            r_list = []
            for name, w_init, q_m, l_m in branch_pool:
                if l_m is not None and len(context_labels.unique()) >= 2:
                    r = _fast_context_auroc(l_m, context_labels)
                else:
                    r = 0.50
                r_list.append(r)

            q_list = [max(0.0, r - floor) ** gamma for r in r_list]
            sum_q = sum(q_list)
            if sum_q > 0:
                weights = [q / sum_q for q in q_list]
            else:
                weights = [1.0 / len(branch_pool)] * len(branch_pool)

            scores = sum(w * torch.sigmoid(q_m.float()) for w, (_, _, q_m, _) in zip(weights, branch_pool))
        else:
            scores = torch.softmax(logits.float(), dim=-1)[:, 1]

        nan_count = int(torch.isnan(scores).sum())
        probabilities = [float(value) for value in scores]
        queried_ids = list(test_ids)
        m_cv = m_cv.cpu()
        m_dd = m_dd.cpu()
        m_ct = m_ct.cpu()
        m_bm = m_bm.cpu()
        if "m_rm" in locals() and isinstance(m_rm, torch.Tensor):
            m_rm = m_rm.cpu()
        if isinstance(m_bs, torch.Tensor):
            m_bs = m_bs.cpu()
        if isinstance(m_sh, torch.Tensor):
            m_sh = m_sh.cpu()
        m_bd = m_bd.cpu()
        m_qa = m_qa.cpu()
        m_ds = m_ds.cpu()
        m_lr = m_lr.cpu()
        m_de = m_de.cpu()
        m_sw = m_sw.cpu()






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
        "m_cv": m_cv[valid] if ("m_cv" in locals() and isinstance(m_cv, torch.Tensor)) else None,
        "m_dd": m_dd[valid] if ("m_dd" in locals() and isinstance(m_dd, torch.Tensor)) else None,
        "m_ct": m_ct[valid] if ("m_ct" in locals() and isinstance(m_ct, torch.Tensor)) else None,
        "m_bm": m_bm[valid] if ("m_bm" in locals() and isinstance(m_bm, torch.Tensor)) else None,
        "m_rm": m_rm[valid] if ("m_rm" in locals() and isinstance(m_rm, torch.Tensor)) else None,
        "m_bs": m_bs[valid] if ("m_bs" in locals() and isinstance(m_bs, torch.Tensor)) else None,
        "m_sh": m_sh[valid] if ("m_sh" in locals() and isinstance(m_sh, torch.Tensor)) else None,
        **{
            f"m_{_k}": (_v.detach().cpu()[valid] if isinstance(_v, torch.Tensor) else None)
            for _k, _v in (sh_variant_margins.items() if "sh_variant_margins" in locals() else [])
            if _k != "sh"
        },
        "m_bd": m_bd[valid] if ("m_bd" in locals() and isinstance(m_bd, torch.Tensor)) else None,
        "m_qa": m_qa[valid] if ("m_qa" in locals() and isinstance(m_qa, torch.Tensor)) else None,
        "m_ds": m_ds[valid] if ("m_ds" in locals() and isinstance(m_ds, torch.Tensor)) else None,
        "m_lr": m_lr[valid] if ("m_lr" in locals() and isinstance(m_lr, torch.Tensor)) else None,
        "m_de": m_de[valid] if ("m_de" in locals() and isinstance(m_de, torch.Tensor)) else None,
        "m_sw": m_sw[valid] if ("m_sw" in locals() and isinstance(m_sw, torch.Tensor)) else None,
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
    # Keep every slide on CPU. Fold k queries its own slides with the
    # other folds as context; evaluate_trial streams one bag at a time.
    projected = cpu_bag_mapping(
        {
            sid: bag
            for state in fold_states
            for sid, bag in zip(state["slide_id"], state["bag"])
        }
    )
    bag_stats_cache: BagStatsCache = {}
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
            bag_stats_cache=bag_stats_cache,
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
    projected = cpu_bag_mapping(bags)
    bag_stats_cache: BagStatsCache = {}
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
            bag_stats_cache=bag_stats_cache,
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
            "m_cv": result.get("m_cv"),
            "m_dd": result.get("m_dd"),
            "m_ct": result.get("m_ct"),
            "m_bm": result.get("m_bm"),
            "m_rm": result.get("m_rm"),
            "m_bs": result.get("m_bs"),
            "m_sh": result.get("m_sh"),
            **{f"m_{_k}": result.get(f"m_{_k}")
               for _k in ("shs", "shk", "sh2", "shr", "shr2", "shj")},
            "m_bd": result.get("m_bd"),
            "m_qa": result.get("m_qa"),
            "m_ds": result.get("m_ds"),
            "m_lr": result.get("m_lr"),
            "m_de": result.get("m_de"),
            "m_sw": result.get("m_sw"),
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
    if DD_LLR_OFFSETS:
        values = torch.tensor(DD_LLR_OFFSETS)
        print(f"DD LLR log(s0/s1): mean {values.mean():+.4f}  "
              f"sd {values.std():.4f}  |max| {values.abs().max():.4f}  "
              f"n {len(DD_LLR_OFFSETS)}")
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
    if os.environ.get("ICF_COVARIANCE_BASIS") not in ("pca", "pca_within", "fisher", "fisher_within"):
        raise ValueError(
            "ICF_SKETCH_DIM requires ICF_COVARIANCE_BASIS=pca, pca_within, or fisher; "
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


def load_checkpoint_or_fresh(model, checkpoint_path) -> None:
    """Load ``--checkpoint`` weights, or skip loading for a fresh instance.

    The training-free configuration (v106+, docs SS183) overrides the projection
    with per-episode PCA and the head with fixed constants, and the ridge
    calibration parameters stay at init -- so no learned value from a checkpoint
    reaches the margin. A fresh instance (random init) is therefore equivalent,
    and ``--checkpoint`` becomes optional.
    """
    if checkpoint_path is None:
        print("Model: fresh instance (no --checkpoint) -- training-free eval",
              flush=True)
        return
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location="cpu"
    )
    model.on_load_checkpoint(checkpoint)
    load_state_dict_for_sketch_dim(model, checkpoint["state_dict"])
    print(f"Model: loaded checkpoint {checkpoint_path.name}", flush=True)


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
        load_checkpoint_or_fresh(model, args.checkpoint)
        apply_ridge_lambda_override(model)
        if args.rare_logits_zero:
            model.model.meta_classifier.force_rare_logits_zero = True
        model.eval()
        model.to(device)
        print(f"Model: arch v{model.model.architecture_version}")
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
        load_checkpoint_or_fresh(model, args.checkpoint)
        apply_ridge_lambda_override(model)
        if args.rare_logits_zero:
            model.model.meta_classifier.force_rare_logits_zero = True
        model.eval()
        model.to(device)
        print(f"Model: arch v{model.model.architecture_version}")
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
            projected[slide_id] = bag.cpu() if bag.device.type != "cpu" else bag
        for slide_id, bag in zip(test_state["slide_id"], test_state["bag"]):
            projected[slide_id] = bag.cpu() if bag.device.type != "cpu" else bag
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
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for --csv evaluation.")
    checkpoint = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu")
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])
    if args.rare_logits_zero:
        model.model.meta_classifier.force_rare_logits_zero = True
    model.eval()
    model.to(device)
    if not use_cache:
        if pca_needed:
            # Project one bag at a time on the GPU, then park the result on
            # CPU. Holding every projected slide on device was the eval's
            # residency cost; the GEMM itself still needs the GPU.
            projected = project_bags_to_cpu(bags, pca_mean, pca_components, device)
            print(f"Projected {len(projected)} slides to {input_dim}-d via {device} (CPU-resident)")
        else:
            projected = cpu_bag_mapping(bags)
            print(f"Keeping {len(projected)} slides on CPU as raw {FEATURE_DIM}-d (no PCA)")
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
