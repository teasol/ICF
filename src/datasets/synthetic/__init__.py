"""Synthetic data generation package."""

from src.datasets.synthetic.dataset import SyntheticEpisodeDataset
from src.datasets.synthetic.generator import SyntheticManifoldGenerator
from src.datasets.synthetic.types import (
    RESPONSE_TASK_NAMES,
    SyntheticEpisode,
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
