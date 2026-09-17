"""Unit tests for scripts/data/build_gf_background_gmm.py.

Uses tiny synthetic h5 fixtures (a few files, hundreds of patches) so the
suite runs in seconds -- it does not touch the real gf_background dataset.
Covers: index building, disjointness of raw/balanced samples from the
validation pool, determinism under a fixed seed, and EM checkpoint/resume.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.data.build_gf_background_gmm import (
    DATASETS,
    build_index,
    fit_gmm_checkpointed,
    sample_balanced,
    sample_raw,
    sample_validation,
)


def _write_h5(path: Path, n: int, dim: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=rng.normal(size=(n, dim)).astype(np.float32))
        f.create_dataset("coords", data=rng.integers(0, 1000, size=(n, 2)).astype(np.int64))


@pytest.fixture()
def fake_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "gf_background"
    sizes = {"COMET": [40, 30], "MUT-HET-RCC": [120, 80, 60], "PANDA": [200]}
    for dataset, file_sizes in sizes.items():
        h5_dir = root / dataset / "features_uni_v2"
        h5_dir.mkdir(parents=True)
        for i, n in enumerate(file_sizes):
            _write_h5(h5_dir / f"slide_{i}.h5", n, dim=16, seed=hash((dataset, i)) % 2**31)
    return root


def test_build_index_matches_file_sizes(fake_data_root: Path):
    entries = build_index(fake_data_root)
    assert {e.dataset for e in entries} == set(DATASETS)
    by_dataset = {}
    for e in entries:
        by_dataset.setdefault(e.dataset, 0)
        by_dataset[e.dataset] += e.n_patches
    assert by_dataset == {"COMET": 70, "MUT-HET-RCC": 260, "PANDA": 200}


def test_validation_disjoint_from_raw_and_balanced(fake_data_root: Path):
    entries = build_index(fake_data_root)
    val_idx, X_val, val_labels = sample_validation(entries, seed=0, val_total=60)
    assert len(X_val) == 60
    assert X_val.shape[1] == 16

    X_raw, raw_labels = sample_raw(entries, seed=1, target=200, excluded_global_idx=val_idx)
    X_bal, bal_labels = sample_balanced(
        entries, seed=2, per_dataset=50, excluded_global_idx=val_idx
    )

    # Disjointness is checked at the feature-row level via a marker trick:
    # every validation row is unique (drawn without replacement from the full
    # pool) and excluded from the complement pool the other two draws sample
    # from, so no validation row's exact feature vector should reappear.
    val_set = {tuple(row) for row in X_val}
    raw_set = {tuple(row) for row in X_raw}
    bal_set = {tuple(row) for row in X_bal}
    assert val_set.isdisjoint(raw_set)
    assert val_set.isdisjoint(bal_set)


def test_balanced_sample_is_equal_per_dataset(fake_data_root: Path):
    entries = build_index(fake_data_root)
    val_idx, _, _ = sample_validation(entries, seed=0, val_total=10)
    X_bal, bal_labels = sample_balanced(entries, seed=3, per_dataset=30, excluded_global_idx=val_idx)
    counts = {DATASETS[i]: int((bal_labels == i).sum()) for i in range(len(DATASETS))}
    assert counts == {"COMET": 30, "MUT-HET-RCC": 30, "PANDA": 30}


def test_sampling_is_deterministic_given_seed(fake_data_root: Path):
    entries = build_index(fake_data_root)
    val_idx_a, X_val_a, _ = sample_validation(entries, seed=42, val_total=50)
    val_idx_b, X_val_b, _ = sample_validation(entries, seed=42, val_total=50)
    assert np.array_equal(val_idx_a, val_idx_b)
    assert np.array_equal(X_val_a, X_val_b)

    X_raw_a, _ = sample_raw(entries, seed=42, target=100, excluded_global_idx=val_idx_a)
    X_raw_b, _ = sample_raw(entries, seed=42, target=100, excluded_global_idx=val_idx_b)
    assert np.array_equal(X_raw_a, X_raw_b)


def test_raw_natural_ratio_close_to_pool_proportions(fake_data_root: Path):
    entries = build_index(fake_data_root)
    val_idx, _, _ = sample_validation(entries, seed=0, val_total=10)
    X_raw, raw_labels = sample_raw(entries, seed=4, target=400, excluded_global_idx=val_idx)
    counts = {DATASETS[i]: int((raw_labels == i).sum()) for i in range(len(DATASETS))}
    total_pool = {"COMET": 70, "MUT-HET-RCC": 260, "PANDA": 200}
    total = sum(total_pool.values())
    for dataset in DATASETS:
        expected_frac = total_pool[dataset] / total
        actual_frac = counts[dataset] / len(X_raw)
        assert abs(actual_frac - expected_frac) < 0.1


def test_gmm_checkpoint_resume_continues_iteration_count(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8)).astype(np.float32)

    model1, ckpt1 = fit_gmm_checkpointed(
        X, tmp_path, "gmm", n_components=3, seed=0,
        max_iter=2, tol=1e-3, time_limit_seconds=None, force=False,
    )
    assert ckpt1.n_iter_done == 2
    assert (tmp_path / "gmm_model.pkl").exists()
    assert (tmp_path / "gmm_ckpt.json").exists()

    model2, ckpt2 = fit_gmm_checkpointed(
        X, tmp_path, "gmm", n_components=3, seed=0,
        max_iter=4, tol=1e-3, time_limit_seconds=None, force=False,
    )
    assert ckpt2.n_iter_done == 4
    assert ckpt2.elapsed_seconds >= ckpt1.elapsed_seconds


def test_gmm_checkpoint_force_restarts_from_zero(tmp_path: Path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8)).astype(np.float32)

    _, ckpt1 = fit_gmm_checkpointed(
        X, tmp_path, "gmm", n_components=3, seed=0,
        max_iter=3, tol=1e-3, time_limit_seconds=None, force=False,
    )
    assert ckpt1.n_iter_done == 3

    _, ckpt2 = fit_gmm_checkpointed(
        X, tmp_path, "gmm", n_components=3, seed=0,
        max_iter=2, tol=1e-3, time_limit_seconds=None, force=True,
    )
    assert ckpt2.n_iter_done == 2
