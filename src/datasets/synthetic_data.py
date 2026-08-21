"""Backward-compatibility facade for synthetic data generator and dataset.

The implementation has been decomposed into the modular `src.datasets.synthetic` package:
  - `src.datasets.synthetic.types`: SyntheticEpisode, RESPONSE_TASK_NAMES
  - `src.datasets.synthetic.generator`: SyntheticManifoldGenerator
  - `src.datasets.synthetic.dataset`: SyntheticEpisodeDataset
"""

from src.datasets.synthetic import (
    RESPONSE_TASK_NAMES,
    SyntheticEpisode,
    SyntheticEpisodeDataset,
    SyntheticManifoldGenerator,
    _CATEGORICAL_TASK_NAMES,
    _NUMERIC_TASK_NAMES,
)

__all__ = [
    "RESPONSE_TASK_NAMES",
    "SyntheticEpisode",
    "SyntheticEpisodeDataset",
    "SyntheticManifoldGenerator",
    "_CATEGORICAL_TASK_NAMES",
    "_NUMERIC_TASK_NAMES",
]
