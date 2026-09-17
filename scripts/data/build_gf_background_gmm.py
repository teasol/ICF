"""GF 배경 GMM 표본 구성 비교 파일럿: Raw vs Balanced 표본화, checkpoint 가능한 EM 적합.

연구 질문(docs/current_status.md "GF 배경 GMM의 표본 구성 비교 실험"): 전체 patch 풀에서
자연 비율로 뽑은 GMM(Raw)과 dataset별 동일 수로 뽑은 GMM(Balanced)이 실제로 다른 분포를
학습하는가. UNI2 1536차원 feature와 diagonal covariance를 쓰며 기본 진단값은 M=8이다.
후속 GF 단독 탐색은 같은 표본을 재사용해 ``--n-components``로 M=16·32도 적합한다.
배경 GMM 적합에는 라벨/Primary 7/SEAL 10을 쓰지 않는다.

파이프라인은 순서대로 재개 가능한 단계로 나뉜다:
    index -> sample (raw/balanced/val 각각) -> fit (raw/balanced 각각) -> summary
각 단계 산출물은 --out-dir 아래 파일로 존재하면 재계산하지 않고 재사용한다(--force로 강제 재계산).
GMM 적합은 sklearn `warm_start`로 한 번에 한 EM 반복만 수행하고 매 반복마다 모델 상태를
디스크에 저장하므로, 실행이 도중에 죽어도(예: 시간 제한) 마지막 완료된 반복부터 재개한다.

Usage (직접 실행, Slurm 없음):
    python scripts/data/build_gf_background_gmm.py --stage all

Smoke test (파일 몇 개, 수만 patch):
    python scripts/data/build_gf_background_gmm.py --stage all --smoke \
        --out-dir /tmp/gf_gmm_smoke
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

DATASETS = ("COMET", "MUT-HET-RCC", "PANDA")
FEATURE_DIM = 1536

DEFAULT_DATA_ROOT = Path("/NHNHOME/BASE/kimds/Data/PathoBench/gf_background")
DEFAULT_OUT_ROOT = Path("/NHNHOME/BASE/kimds/Data/PathoBench/gf_background/fixed_gmm")


# ---------------------------------------------------------------------------
# Stage 0: index
# ---------------------------------------------------------------------------


@dataclass
class FileEntry:
    dataset: str
    path: str
    n_patches: int


def build_index(data_root: Path, max_files_per_dataset: int | None = None) -> list[FileEntry]:
    entries: list[FileEntry] = []
    for dataset in DATASETS:
        h5_dir = data_root / dataset / "features_uni_v2"
        files = sorted(h5_dir.glob("*.h5"))
        if max_files_per_dataset is not None:
            files = files[:max_files_per_dataset]
        for path in files:
            with h5py.File(path, "r") as f:
                n = int(f["features"].shape[0])
            entries.append(FileEntry(dataset=dataset, path=str(path), n_patches=n))
    return entries


def load_or_build_index(out_dir: Path, data_root: Path, force: bool, smoke: bool) -> list[FileEntry]:
    index_path = out_dir / "index.json"
    if index_path.exists() and not force:
        raw = json.loads(index_path.read_text())
        return [FileEntry(**r) for r in raw]
    max_files = 2 if smoke else None
    entries = build_index(data_root, max_files_per_dataset=max_files)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps([asdict(e) for e in entries], indent=2))
    return entries


# ---------------------------------------------------------------------------
# Stage 1: sampling
# ---------------------------------------------------------------------------


def _cumulative_offsets(entries: list[FileEntry]) -> np.ndarray:
    counts = np.array([e.n_patches for e in entries], dtype=np.int64)
    return np.concatenate([[0], np.cumsum(counts)])


def _global_indices_to_file_groups(
    global_idx: np.ndarray, entries: list[FileEntry], offsets: np.ndarray
) -> dict[int, np.ndarray]:
    """Map sorted global patch indices to {entry_index: local_row_indices (sorted)}."""
    global_idx = np.sort(global_idx)
    file_of_idx = np.searchsorted(offsets, global_idx, side="right") - 1
    groups: dict[int, np.ndarray] = {}
    for entry_i in np.unique(file_of_idx):
        mask = file_of_idx == entry_i
        local = global_idx[mask] - offsets[entry_i]
        groups[int(entry_i)] = local
    return groups


def _read_rows(entries: list[FileEntry], groups: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Read (features, dataset_label_ids) for the given per-file row groups, in file order."""
    feats_chunks = []
    labels_chunks = []
    dataset_to_id = {d: i for i, d in enumerate(DATASETS)}
    for entry_i in sorted(groups):
        entry = entries[entry_i]
        rows = groups[entry_i]
        with h5py.File(entry.path, "r") as f:
            feats_chunks.append(f["features"][rows, :])
        labels_chunks.append(np.full(len(rows), dataset_to_id[entry.dataset], dtype=np.int8))
    if not feats_chunks:
        return (
            np.zeros((0, FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int8),
        )
    return np.concatenate(feats_chunks, axis=0), np.concatenate(labels_chunks, axis=0)


def sample_validation(
    entries: list[FileEntry], seed: int, val_total: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw a validation pool proportional to natural dataset ratio, disjoint from later draws.

    Returns (global_indices used, features, dataset_label_ids).
    """
    offsets = _cumulative_offsets(entries)
    total = int(offsets[-1])
    val_total = min(val_total, total)
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(total, size=val_total, replace=False)
    groups = _global_indices_to_file_groups(val_idx, entries, offsets)
    feats, labels = _read_rows(entries, groups)
    return np.sort(val_idx), feats, labels


def sample_raw(
    entries: list[FileEntry], seed: int, target: int, excluded_global_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Natural-ratio sample from the full pool, excluding validation indices."""
    offsets = _cumulative_offsets(entries)
    total = int(offsets[-1])
    pool = np.setdiff1d(np.arange(total), excluded_global_idx, assume_unique=False)
    target = min(target, len(pool))
    rng = np.random.default_rng(seed)
    chosen_pos = rng.choice(len(pool), size=target, replace=False)
    chosen = pool[chosen_pos]
    groups = _global_indices_to_file_groups(chosen, entries, offsets)
    return _read_rows(entries, groups)


def sample_balanced(
    entries: list[FileEntry], seed: int, per_dataset: int, excluded_global_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count-per-dataset sample, excluding validation indices."""
    offsets = _cumulative_offsets(entries)
    rng = np.random.default_rng(seed)
    feats_chunks = []
    labels_chunks = []
    dataset_to_id = {d: i for i, d in enumerate(DATASETS)}
    for dataset in DATASETS:
        ds_entry_ids = [i for i, e in enumerate(entries) if e.dataset == dataset]
        if not ds_entry_ids:
            continue
        lo = offsets[ds_entry_ids[0]]
        hi = offsets[ds_entry_ids[-1] + 1]
        ds_pool = np.arange(lo, hi)
        excluded_in_ds = excluded_global_idx[
            (excluded_global_idx >= lo) & (excluded_global_idx < hi)
        ]
        ds_pool = np.setdiff1d(ds_pool, excluded_in_ds, assume_unique=False)
        n = min(per_dataset, len(ds_pool))
        chosen_pos = rng.choice(len(ds_pool), size=n, replace=False)
        chosen = ds_pool[chosen_pos]
        groups = _global_indices_to_file_groups(chosen, entries, offsets)
        f, _ = _read_rows(entries, groups)
        feats_chunks.append(f)
        labels_chunks.append(np.full(len(f), dataset_to_id[dataset], dtype=np.int8))
    return np.concatenate(feats_chunks, axis=0), np.concatenate(labels_chunks, axis=0)


def _npz_path(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.npz"


def load_or_sample(
    out_dir: Path,
    name: str,
    force: bool,
    sampler,
) -> tuple[np.ndarray, np.ndarray]:
    path = _npz_path(out_dir, name)
    if path.exists() and not force:
        data = np.load(path)
        return data["features"], data["labels"]
    feats, labels = sampler()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as fh:
        np.savez(fh, features=feats, labels=labels)
    tmp_path.rename(path)
    return feats, labels


# ---------------------------------------------------------------------------
# Stage 2: checkpointed GMM fitting
# ---------------------------------------------------------------------------


@dataclass
class GmmCheckpoint:
    n_iter_done: int
    converged: bool
    lower_bound: float
    elapsed_seconds: float


def _checkpoint_paths(out_dir: Path, name: str) -> tuple[Path, Path]:
    return out_dir / f"{name}_model.pkl", out_dir / f"{name}_ckpt.json"


def fit_gmm_checkpointed(
    X: np.ndarray,
    out_dir: Path,
    name: str,
    n_components: int,
    seed: int,
    max_iter: int,
    tol: float,
    time_limit_seconds: float | None,
    force: bool,
):
    """Fit a diagonal-covariance GaussianMixture one EM step at a time, checkpointing
    the full model + iteration count + wall time after every step so a killed job
    can resume from the last completed iteration instead of losing progress."""
    from sklearn.mixture import GaussianMixture

    model_path, ckpt_path = _checkpoint_paths(out_dir, name)
    start_iter = 0
    prior_elapsed = 0.0
    model: GaussianMixture | None = None

    ckpt: GmmCheckpoint | None = None
    if model_path.exists() and ckpt_path.exists() and not force:
        with model_path.open("rb") as fh:
            model = pickle.load(fh)
        ckpt = GmmCheckpoint(**json.loads(ckpt_path.read_text()))
        start_iter = ckpt.n_iter_done
        prior_elapsed = ckpt.elapsed_seconds
        if ckpt.converged or start_iter >= max_iter:
            return model, ckpt

    if model is None:
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=seed,
            warm_start=True,
            max_iter=1,
            tol=tol,
            n_init=1,
            init_params="k-means++",
        )

    t0 = time.monotonic()
    converged = False
    prev_lower_bound = -np.inf
    n_iter_done = start_iter
    for step in range(start_iter, max_iter):
        model.max_iter = 1
        model.fit(X)
        n_iter_done = step + 1
        lower_bound = float(model.lower_bound_)
        elapsed = prior_elapsed + (time.monotonic() - t0)
        if lower_bound - prev_lower_bound < tol and step > start_iter:
            converged = True
        prev_lower_bound = lower_bound

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_model_path = model_path.with_suffix(".pkl.tmp")
        with tmp_model_path.open("wb") as fh:
            pickle.dump(model, fh)
        tmp_model_path.rename(model_path)
        ckpt = GmmCheckpoint(
            n_iter_done=n_iter_done,
            converged=converged,
            lower_bound=lower_bound,
            elapsed_seconds=elapsed,
        )
        tmp_ckpt_path = ckpt_path.with_suffix(".json.tmp")
        tmp_ckpt_path.write_text(json.dumps(asdict(ckpt)))
        tmp_ckpt_path.rename(ckpt_path)

        if converged:
            break
        if time_limit_seconds is not None and elapsed > time_limit_seconds:
            break

    return model, ckpt


# ---------------------------------------------------------------------------
# Stage 3: measurement / summary
# ---------------------------------------------------------------------------


def per_dataset_val_loglik(model, X_val: np.ndarray, val_labels: np.ndarray) -> dict:
    result = {"overall": float(model.score(X_val))}
    for i, dataset in enumerate(DATASETS):
        mask = val_labels == i
        if mask.sum() == 0:
            result[dataset] = None
            continue
        result[dataset] = float(model.score(X_val[mask]))
    return result


def dataset_component_occupancy(model, X_val: np.ndarray, val_labels: np.ndarray) -> dict:
    resp = model.predict_proba(X_val)
    result = {}
    for i, dataset in enumerate(DATASETS):
        mask = val_labels == i
        if mask.sum() == 0:
            result[dataset] = None
            continue
        result[dataset] = resp[mask].mean(axis=0).tolist()
    return result


def align_components(model_a, model_b) -> dict:
    """Greedy nearest-mean matching between two models' components, then report
    weight/mean/diag-covariance differences under that alignment."""
    from scipy.optimize import linear_sum_assignment

    means_a, means_b = model_a.means_, model_b.means_
    dist = np.linalg.norm(means_a[:, None, :] - means_b[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(dist)

    weight_diff = (model_a.weights_[row_ind] - model_b.weights_[col_ind]).tolist()
    mean_l2_diff = np.linalg.norm(
        means_a[row_ind] - means_b[col_ind], axis=-1
    ).tolist()
    cov_l2_diff = np.linalg.norm(
        model_a.covariances_[row_ind] - model_b.covariances_[col_ind], axis=-1
    ).tolist()
    return {
        "raw_component": row_ind.tolist(),
        "balanced_component": col_ind.tolist(),
        "weight_diff": weight_diff,
        "mean_l2_diff": mean_l2_diff,
        "diag_cov_l2_diff": cov_l2_diff,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--raw-target", type=int, default=3_000_000)
    p.add_argument("--balanced-per-dataset", type=int, default=1_000_000)
    p.add_argument("--val-total", type=int, default=30_000)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--time-limit-seconds", type=float, default=None)
    p.add_argument(
        "--stage",
        choices=["index", "sample", "fit_raw", "fit_balanced", "summary", "all"],
        default="all",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Restrict to 2 files/dataset and small sample targets for smoke testing.",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> dict:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        raw_target = min(args.raw_target, 20_000)
        balanced_per_dataset = min(args.balanced_per_dataset, 6_000)
        val_total = min(args.val_total, 3_000)
        max_iter = min(args.max_iter, 5)
    else:
        raw_target = args.raw_target
        balanced_per_dataset = args.balanced_per_dataset
        val_total = args.val_total
        max_iter = args.max_iter

    entries = load_or_build_index(out_dir, args.data_root, args.force, args.smoke)

    if args.stage == "index":
        return {"n_files": len(entries), "n_patches": sum(e.n_patches for e in entries)}

    # val_idx is cheap (RNG only, no I/O) so it is always recomputed to exclude
    # validation rows from the raw/balanced draws below, even when val.npz is cached.
    val_idx_full = sample_validation(entries, args.seed, val_total)[0]

    X_val, val_labels = load_or_sample(
        out_dir,
        "val",
        args.force,
        lambda: sample_validation(entries, args.seed, val_total)[1:],
    )

    X_raw, raw_labels = load_or_sample(
        out_dir,
        "X_raw",
        args.force,
        lambda: sample_raw(entries, args.seed, raw_target, val_idx_full),
    )
    X_balanced, balanced_labels = load_or_sample(
        out_dir,
        "X_balanced",
        args.force,
        lambda: sample_balanced(entries, args.seed, balanced_per_dataset, val_idx_full),
    )

    if args.stage == "sample":
        return {
            "n_val": len(X_val),
            "n_raw": len(X_raw),
            "n_balanced": len(X_balanced),
        }

    result: dict = {}
    if args.stage in ("fit_raw", "all"):
        model_raw, ckpt_raw = fit_gmm_checkpointed(
            X_raw, out_dir, "gmm_raw", args.n_components, args.seed,
            max_iter, args.tol, args.time_limit_seconds, args.force,
        )
        result["raw_ckpt"] = asdict(ckpt_raw)
    if args.stage == "fit_raw":
        return result

    if args.stage in ("fit_balanced", "all"):
        model_balanced, ckpt_balanced = fit_gmm_checkpointed(
            X_balanced, out_dir, "gmm_balanced", args.n_components, args.seed,
            max_iter, args.tol, args.time_limit_seconds, args.force,
        )
        result["balanced_ckpt"] = asdict(ckpt_balanced)
    if args.stage == "fit_balanced":
        return result

    if args.stage in ("summary", "all"):
        model_raw_path, _ = _checkpoint_paths(out_dir, "gmm_raw")
        model_balanced_path, _ = _checkpoint_paths(out_dir, "gmm_balanced")
        with model_raw_path.open("rb") as fh:
            model_raw = pickle.load(fh)
        with model_balanced_path.open("rb") as fh:
            model_balanced = pickle.load(fh)

        summary = {
            "n_components": args.n_components,
            "seed": args.seed,
            "n_val": len(X_val),
            "n_raw": len(X_raw),
            "n_balanced": len(X_balanced),
            "raw": {
                "val_loglik": per_dataset_val_loglik(model_raw, X_val, val_labels),
                "component_weights": model_raw.weights_.tolist(),
                "component_occupancy": dataset_component_occupancy(model_raw, X_val, val_labels),
            },
            "balanced": {
                "val_loglik": per_dataset_val_loglik(model_balanced, X_val, val_labels),
                "component_weights": model_balanced.weights_.tolist(),
                "component_occupancy": dataset_component_occupancy(model_balanced, X_val, val_labels),
            },
            "alignment": align_components(model_raw, model_balanced),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        result["summary_path"] = str(out_dir / "summary.json")
        result["summary"] = summary
    return result


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
