"""Pin the bf16-mixed numerical-safety contract (agent_handoff SS3.4).

The covariance sketch inverts a correlation feature and the ridge branches
solve gram systems; fp16 coefficients overflow to NaN there. The contract has
been stated in `docs/agent_handoff.md` SS3.4 since 2026-08-04 but was not
enforced anywhere -- `configs/trainer/default.yaml` set no precision at all
between SS56 (v34 group defaults) and 2026-08-08, so the v34/v35 entry points
silently resolved to Lightning's 32-true default. This test makes the contract
executable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.utils import merge_train_config  # noqa: E402

REQUIRED_PRECISION = "bf16-mixed"


class TestPrecisionContract(unittest.TestCase):
    """Every active training entry point must resolve to bf16-mixed."""

    def _active_train_configs(self) -> list[Path]:
        return sorted((REPO_ROOT / "configs").glob("train_*.yaml"))

    def test_active_train_configs_exist(self) -> None:
        self.assertTrue(
            self._active_train_configs(),
            "configs/ root has no train_*.yaml entry point to check.",
        )

    def test_active_train_configs_use_bf16_mixed(self) -> None:
        for path in self._active_train_configs():
            with self.subTest(config=path.name):
                config = merge_train_config(path)
                precision = config.get("trainer", {}).get("precision")
                self.assertEqual(
                    precision,
                    REQUIRED_PRECISION,
                    f"{path.name} resolves to precision={precision!r}; "
                    f"{REQUIRED_PRECISION!r} is required (agent_handoff SS3.4). "
                    "fp16 overflows the covariance/ridge solves to NaN, and an "
                    "unset precision falls back to Lightning's 32-true.",
                )

    def test_default_trainer_group_pins_precision(self) -> None:
        """The group default is the single source of truth (SS56)."""
        import yaml

        group = yaml.safe_load(
            (REPO_ROOT / "configs" / "trainer" / "default.yaml").read_text()
        )
        self.assertEqual(group.get("precision"), REQUIRED_PRECISION)


if __name__ == "__main__":
    unittest.main()
