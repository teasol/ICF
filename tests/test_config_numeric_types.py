"""Numeric config values must arrive as numbers, not strings.

YAML 1.1 only reads scientific notation as a float when it has a decimal point
AND a signed exponent: `2.0e-5` parses, `2e-05` becomes the STRING "2e-05".
That string travels all the way into the optimizer before failing, as
`TypeError: '<=' not supported between instances of 'float' and 'str'` from
inside AdamW -- far from the config that caused it.

Printing the value does not reveal this (a string prints identically), so the
check has to be on the type.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.utils import merge_train_config  # noqa: E402

# (config group, key) pairs whose values must be numeric wherever they appear.
NUMERIC_KEYS = {
    "lr", "weight_decay", "eps", "min_lr", "factor", "threshold",
    "warmup_start_factor", "gradient_clip_val",
}


def _numeric_offenders(node, path: str = "") -> list[str]:
    offenders: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            where = f"{path}.{key}" if path else str(key)
            if key in NUMERIC_KEYS and isinstance(value, str):
                offenders.append(f"{where} = {value!r}")
            offenders.extend(_numeric_offenders(value, where))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            offenders.extend(_numeric_offenders(value, f"{path}[{index}]"))
    return offenders


class ConfigNumericTypeTest(unittest.TestCase):
    def test_every_training_config_parses_numbers_as_numbers(self) -> None:
        configs = sorted((REPO_ROOT / "configs").glob("train_*.yaml"))
        self.assertTrue(configs, "no training configs found")
        for path in configs:
            with self.subTest(config=path.name):
                offenders = _numeric_offenders(merge_train_config(path))
                self.assertEqual(
                    offenders,
                    [],
                    f"{path.name}: numeric keys parsed as strings: {offenders}. "
                    "YAML needs `2.0e-5`, not `2e-05`.",
                )


if __name__ == "__main__":
    unittest.main()
