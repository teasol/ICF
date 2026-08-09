"""VRAM safety-guard contracts (worst-case estimate + fail-fast validator).

The estimator is a pure function and the validator's CUDA branch is exercised
with mocks so the compact suite never needs a GPU. The bounds mirror the
v33 Phase 0 / v30 configs (B2b ragged: num_bags up to 100, per-bag
n_b ~ LogUniform[1,1024], input_dim 512).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.utils.utils import (
    estimate_training_vram_bytes,
    validate_vram_budget,
)

ARM_C_NUM_BAGS_MAX = 100
ARM_C_MAX_CELLS = 1024
ARM_C_INPUT_DIM = 512
ARM_C_PARAM_COUNT = 9_500_000


class EstimateTrainingVramBytesTest(unittest.TestCase):
    def test_positive_and_covers_model_optimizer_footprint(self):
        estimate = estimate_training_vram_bytes(
            num_bags_max=ARM_C_NUM_BAGS_MAX,
            max_cells_per_bag=ARM_C_MAX_CELLS,
            input_dim=ARM_C_INPUT_DIM,
            param_count=ARM_C_PARAM_COUNT,
        )
        self.assertGreater(estimate, 0)
        # Must at least cover weights + two Adam moments + gradients.
        self.assertGreater(estimate, ARM_C_PARAM_COUNT * 14)

    def test_monotonic_in_bags_and_cells(self):
        base = estimate_training_vram_bytes(
            num_bags_max=10, max_cells_per_bag=10, input_dim=8, param_count=1_000
        )
        more_bags = estimate_training_vram_bytes(
            num_bags_max=20, max_cells_per_bag=10, input_dim=8, param_count=1_000
        )
        more_cells = estimate_training_vram_bytes(
            num_bags_max=10, max_cells_per_bag=20, input_dim=8, param_count=1_000
        )
        self.assertGreater(more_bags, base)
        self.assertGreater(more_cells, base)

    def test_scales_with_episode_batch_size(self):
        """Peak memory tracks TOTAL cells per step, so batch must count.

        v34-1536 (batch 4 x 100 bags x 8192 cells) and v35 (batch 1 x 100 x
        32768) reach the same 3.28M cells and the same measured peak band; a
        bound that ignored batch would call a 4x batch increase free.
        """
        single = estimate_training_vram_bytes(
            num_bags_max=100,
            max_cells_per_bag=8192,
            input_dim=1536,
            param_count=ARM_C_PARAM_COUNT,
            episode_batch_size=1,
        )
        quadruple = estimate_training_vram_bytes(
            num_bags_max=100,
            max_cells_per_bag=8192,
            input_dim=1536,
            param_count=ARM_C_PARAM_COUNT,
            episode_batch_size=4,
        )
        self.assertGreater(quadruple, single)
        # Equal total cells (batch x bags x cells) must give equal estimates.
        traded = estimate_training_vram_bytes(
            num_bags_max=100,
            max_cells_per_bag=32768,
            input_dim=1536,
            param_count=ARM_C_PARAM_COUNT,
            episode_batch_size=1,
        )
        self.assertEqual(quadruple, traded)

    def test_multiplier_stays_above_measured_peaks(self):
        """The bound must exceed the two peaks measured on a B200.

        v34-1536: 112 GB at 3.28M cells x 1536-d. v35: 122.4 GB
        (`max_memory_allocated` after forward+backward+step, 100 x 32768).
        """
        estimate = estimate_training_vram_bytes(
            num_bags_max=100,
            max_cells_per_bag=32768,
            input_dim=1536,
            param_count=41_670_000,
            episode_batch_size=1,
        )
        self.assertGreater(estimate, 122.4e9)
        # ...but must not be so loose that it stops rejecting oversized configs.
        self.assertLess(estimate, 0.9 * 192e9)

    def test_arm_c_worst_case_fits_a6000_with_large_margin(self):
        estimate = estimate_training_vram_bytes(
            num_bags_max=ARM_C_NUM_BAGS_MAX,
            max_cells_per_bag=ARM_C_MAX_CELLS,
            input_dim=ARM_C_INPUT_DIM,
            param_count=ARM_C_PARAM_COUNT,
        )
        # A6000 = 48 GiB. The deliberately conservative bound must stay far
        # below it; the true footprint is much smaller still.
        self.assertLess(estimate, 24 * (1 << 30))


class _FakeDeviceProperties(SimpleNamespace):
    name = "Fake GPU"
    total_memory = 48 * (1 << 30)


def _arm_c_config(**dataset_overrides: int) -> dict:
    dataset_kwargs = {
        "num_bags": [60, 100],
        "num_cells": [1, 1024],
        **dataset_overrides,
    }
    return {
        "data": {"dataset_kwargs": dataset_kwargs},
        "model": {"input_dim": ARM_C_INPUT_DIM},
        "trainer": {},
    }


class ValidateVramBudgetTest(unittest.TestCase):
    def test_passes_with_large_margin_on_cuda(self):
        model = nn.Linear(8, 8)
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.current_device", return_value=0
        ), patch(
            "torch.cuda.get_device_properties",
            return_value=_FakeDeviceProperties(),
        ):
            validate_vram_budget(_arm_c_config(), model, verbose=False)  # no raise

    def test_raises_when_worst_case_exceeds_hard_limit(self):
        model = nn.Linear(8, 8)
        # Absurd per-bag size -> the estimate must trip the 0.9 hard limit.
        config = _arm_c_config(num_cells=[1, 100_000_000])
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.current_device", return_value=0
        ), patch(
            "torch.cuda.get_device_properties",
            return_value=_FakeDeviceProperties(),
        ):
            with self.assertRaises(RuntimeError):
                validate_vram_budget(config, model, verbose=False)

    def test_skips_without_cuda(self):
        model = nn.Linear(8, 8)
        with patch("torch.cuda.is_available", return_value=False):
            validate_vram_budget(_arm_c_config(), model)  # must not raise


if __name__ == "__main__":
    unittest.main()


class TestCovarianceOnlyBudget(unittest.TestCase):
    """The estimator must know CV-only skips the per-cell activation chain.

    The 6-layer default is calibrated on full-branch v34/v35. Applying it to
    CV-only over-estimates by ~3.4x, which rejected an E=4 configuration that
    measured 58 GiB (60 bags) / ~97 GiB (100 bags worst case) -- comfortably
    inside the budget. See docs SS68 for the measured table.
    """

    def test_covariance_only_estimate_is_about_a_third_of_full(self) -> None:
        common = dict(
            num_bags_max=100,
            max_cells_per_bag=16384,
            input_dim=1536,
            param_count=44_000_000,
            episode_batch_size=4,
        )
        full = estimate_training_vram_bytes(**common, activation_layers=6)
        cv = estimate_training_vram_bytes(**common, activation_layers=1)
        ratio = cv / full
        self.assertLess(ratio, 0.40)
        self.assertGreater(ratio, 0.20)

    def test_guard_reads_the_flag_from_model_kwargs(self) -> None:
        """A CV-only config must not be rejected for a cost it does not pay."""
        from pathlib import Path

        from src.utils.utils import merge_train_config

        config = merge_train_config(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "archive"
            / "v40_cvonly_variants"
            / "train_v40_cv_only_e4_1536.yaml"
        )
        self.assertTrue(config["model_kwargs"]["meta_covariance_only"])
        dataset_kwargs = config["data"]["dataset_kwargs"]
        estimate = estimate_training_vram_bytes(
            num_bags_max=max(dataset_kwargs["num_bags"]),
            max_cells_per_bag=max(dataset_kwargs["num_cells"]),
            input_dim=config["model"]["input_dim"],
            param_count=44_000_000,
            episode_batch_size=config["data"]["episode_batch_size"],
            activation_layers=1,
        )
        # 183 GiB device, 90% hard limit.
        self.assertLess(estimate / (183 * 2**30), 0.90)
