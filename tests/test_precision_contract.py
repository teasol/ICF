"""Pin the bf16-mixed numerical-safety contract (agent_handoff SS3.4).

The covariance sketch inverts a correlation feature and the ridge branches
solve gram systems; fp16 coefficients overflow to NaN there. The contract has
been stated in `docs/agent_handoff.md` SS3.4 since 2026-08-04 but was not
enforced anywhere -- `configs/trainer/default.yaml` set no precision at all
between SS56 (v34 group defaults) and 2026-08-08, so the v34/v35 entry points
silently resolved to Lightning's 32-true default. This test makes the contract
executable.

SCOPE (SS63, extended to evaluation on 2026-08-08 by user decision):

* **Training.** Every active `configs/train_*.yaml` entry point AND every
  selectable `configs/trainer/*.yaml` group must resolve to bf16-mixed, so the
  contract cannot be dodged by picking a different trainer group.
* **Evaluation.** Every inference script must default to bf16-mixed via the
  shared `eval_autocast` / `add_eval_precision_argument` helpers. Before this,
  `evaluate_synthetic.py` ran eval under bf16 autocast while
  `test_pathobench.py` / `test_musk.py` ran fp32 and `test.py` defaulted to
  `16-mixed` (fp16 -- the exact overflow path SS3.4 forbids), so the same
  checkpoint scored differently depending on which script was called.
  Consequence, accepted by the user: **every official 50-fold AUROC recorded
  before 2026-08-08 was produced under fp32 and is now reference-only.**
* **`configs/archive/` is NOT covered.** Those are historical reproducibility
  records for dead architectures; rewriting their precision would falsify what
  was actually run.
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

    def test_eval_helper_defaults_to_bf16_mixed(self) -> None:
        """Evaluation shares one precision definition and it defaults to bf16."""
        from src.utils.utils import DEFAULT_EVAL_PRECISION, eval_autocast

        self.assertEqual(DEFAULT_EVAL_PRECISION, REQUIRED_PRECISION)
        # fp16 must be refused outright, not silently accepted.
        with self.assertRaises(ValueError):
            eval_autocast("cuda", "16-mixed")
        # 32-true stays available for reproducing fp32-era numbers.
        eval_autocast("cuda", "32-true")

    def test_eval_scripts_use_the_shared_precision_helper(self) -> None:
        """No inference script may hardcode its own precision.

        `scripts/test.py` defaulted to `16-mixed` (fp16) until 2026-08-08 while
        `test_pathobench.py` ran fp32 and `evaluate_synthetic.py` ran bf16, so
        the same checkpoint scored three different ways.
        """
        scripts = REPO_ROOT / "scripts"
        for name in ("test_pathobench.py", "test_musk.py", "probe_slot_headroom.py"):
            with self.subTest(script=name):
                source = (scripts / name).read_text()
                self.assertIn("add_eval_precision_argument", source)
                self.assertIn("eval_autocast", source)
        for name in ("test.py", "run_official_folds_parallel.py"):
            with self.subTest(script=name):
                source = (scripts / name).read_text()
                self.assertIn(f'"{REQUIRED_PRECISION}"', source)
                self.assertNotIn('default="16-mixed"', source)

    def test_every_selectable_trainer_group_uses_bf16_mixed(self) -> None:
        """No trainer group may opt out -- selecting one must not dodge SS3.4.

        `configs/trainer/ddp5.yaml` and `ddp8.yaml` carried `16-mixed` (fp16)
        until 2026-08-08; that is exactly the overflow path SS3.4 forbids.
        """
        import yaml

        groups = sorted((REPO_ROOT / "configs" / "trainer").glob("*.yaml"))
        self.assertTrue(groups, "configs/trainer/ has no group to check.")
        for path in groups:
            with self.subTest(group=path.name):
                precision = yaml.safe_load(path.read_text()).get("precision")
                self.assertEqual(
                    precision,
                    REQUIRED_PRECISION,
                    f"trainer group {path.name} sets precision={precision!r}; "
                    f"{REQUIRED_PRECISION!r} is required (agent_handoff SS3.4).",
                )


if __name__ == "__main__":
    unittest.main()
