"""ICF models package."""

from src.models.base import BaseInContextClassifier, InContextClassifierProtocol
from src.models.registry import build_model, get_model_class, register_model
from src.models.training_free import TrainingFreeClassifier, TrainingFreeConfig

__all__ = [
    "BaseInContextClassifier",
    "InContextClassifierProtocol",
    "TrainingFreeClassifier",
    "TrainingFreeConfig",
    "build_model",
    "get_model_class",
    "register_model",
]
