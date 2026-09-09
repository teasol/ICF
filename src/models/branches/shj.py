"""SHJ branch (deprecated alias: see src.models.branches.sj).

Retained for backward compatibility. All implementations have moved to sj.py.
"""

from src.models.branches.sj import (  # noqa: F401
    SHJ_FEATURE_DIM,
    SJ_FEATURE_DIM,
    shj_features,
    shj_slide_features,
    sj_features,
    sj_slide_features,
)

__all__ = [
    "SHJ_FEATURE_DIM",
    "SJ_FEATURE_DIM",
    "shj_features",
    "shj_slide_features",
    "sj_features",
    "sj_slide_features",
]
