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
