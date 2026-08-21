"""Backward-compatibility facade for CT readout.

The implementation has been decomposed into the modular `src.models.ct` package:
  - `src.models.ct.config`: Configuration dataclass and NamedTuples
  - `src.models.ct.tokenizers`: Centroid selection algorithms
  - `src.models.ct.abundance`: Cell sampling and soft assignment
  - `src.models.ct.readout`: Margin readouts and predictors
"""

from src.models.ct import (
    CTAbundance,
    CTMargins,
    CTReadoutConfig,
    calibrate,
    ct_abundance,
    ct_margins,
    dbscan_tokens,
    discriminative_score,
    farthest_point_tokens,
    hdbscan_tokens,
    hierarchical_2means_tokens,
    kmeans_plusplus_tokens,
    lloyd_refine,
    parse_cell_budget,
    prepare_cells,
    readout_extreme,
    readout_kernel_ridge,
    readout_prototype,
    readout_ridge,
    resolve_cells_per_bag,
    ridge_coefficients,
    sample_cells,
    typical_bag_size,
)

__all__ = [
    "CTAbundance",
    "CTMargins",
    "CTReadoutConfig",
    "calibrate",
    "ct_abundance",
    "ct_margins",
    "dbscan_tokens",
    "discriminative_score",
    "farthest_point_tokens",
    "hdbscan_tokens",
    "hierarchical_2means_tokens",
    "kmeans_plusplus_tokens",
    "lloyd_refine",
    "parse_cell_budget",
    "prepare_cells",
    "readout_extreme",
    "readout_kernel_ridge",
    "readout_prototype",
    "readout_ridge",
    "resolve_cells_per_bag",
    "ridge_coefficients",
    "sample_cells",
    "typical_bag_size",
]
