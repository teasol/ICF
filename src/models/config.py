"""Configuration schema and YAML loader/serializer for TrainingFreeClassifier.

Provides:
- `TrainingFreeConfig`: Clean, typed, frozen dataclass schema with zero duplicate fields.
- `from_yaml()`: Safe YAML loader with custom PyYAML scientific-notation resolver.
- `from_dict()`: Normalizes flat/nested dicts, strictly coerces types, validates domains.
- `to_yaml()` / `to_dict()`: Lossless bidirectional serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import difflib
from io import StringIO
from pathlib import Path
import re
from typing import Any, Sequence, get_args, get_origin, get_type_hints
import yaml


# ---- Custom YAML Loader with Scientific Notation Resolver --------------------

class _ScientificSafeLoader(yaml.SafeLoader):
    """SafeLoader that recognizes scientific notation (e.g. 1e-3, 2e-5) as float.

    YAML 1.1 requires a decimal point (1.0e-3) for float recognition; without it,
    expressions like '1e-3' or '2e-05' are erroneously parsed as strings.
    """
    pass


_SCIENTIFIC_REGEX = re.compile(
    r"""^(?:
     [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)

_ScientificSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _SCIENTIFIC_REGEX,
    list("-+0123456789."),
)


# ---- Valid Choices (Enums) ---------------------------------------------------

VALID_AGGREGATIONS = {"soft_voting", "trimmed_mean", "linear", "adaptive_trimmed", "hard_gated"}
VALID_CV_BLOCKS = {"offdiag", "cov+mean"}
VALID_BD_METRICS = {"entropy", "trace"}
VALID_BD_READOUTS = {"ordered_typicality", "ridge"}
VALID_CT_TOKENIZERS = {
    "hierarchical_2means", "kmeans_plusplus", "fps", "hdbscan", "dbscan",
}
VALID_CT_READOUTS = {"ridge", "prototype", "extreme"}
VALID_KRR_KERNELS = {"linear", "rbf", "poly", "cosine"}


# ---- Typed Configuration Dataclass -------------------------------------------

@dataclass(frozen=True)
class TrainingFreeConfig:
    """Zero-parameter in-context classifier configuration."""

    # 1. Basis & Projection
    sketch_dim: int = 256
    ridge_lambda: float = 1.0
    ridge_scale: float = 2.0

    # 2. CV Branch (Cross-Covariance)
    cv_blocks: str = "offdiag"
    weight_cv: float = 1.0

    # 3. CT Branch (Cell Tokenizer Abundance)
    weight_ct: float = 1.0
    ct_readout: str = "ridge"
    ct_pca_dim: int | None = 32
    ct_num_tokens: int = 256
    ct_cells_per_bag: int | None = None
    ct_abundance_cells_per_bag: int | None | str | float = None
    ct_cells_fraction: float | None = None
    ct_cells_min: int = 1
    ct_cells_scale: str = "own"
    ct_sampling: str = "even"
    ct_sampling_seed: int = 0
    ct_distance_kernel: str = "gemm"
    ct_tokenizer: str = "hierarchical_2means"
    ct_bisect_iterations: int = 2
    ct_bisect_power_iterations: int = 3
    ct_tree_reduction: str = "segment"
    ct_hdbscan_min_cluster_size: int = 256
    ct_hdbscan_min_cluster_fraction: float = 0.001
    ct_hdbscan_min_samples: int = 32
    ct_hdbscan_cluster_selection_method: str = "leaf"
    ct_hdbscan_build_algo: str = "nn_descent"
    ct_hdbscan_allow_single_cluster: bool = False
    ct_dbscan_eps: float | None = None
    ct_dbscan_min_samples: int = 16
    ct_temperature: float = 0.5
    ct_eps: float = 1e-6
    ct_kmeans_iterations: int = 0
    ct_kmeans_max_iterations: int = 8
    ct_kmeans_tolerance: float = 1e-4
    ct_kmeans_seed: int = 0

    # 4. BM Branch (Projected Bag-Mean)
    weight_bm: float = 1.0
    bm_dim: int = 32
    bm_lambda: float = 1.0

    # 5. BD Branch (Bag Dispersion / Spectral Entropy)
    weight_bd: float = 1.0
    bd_dim: int = 256
    bd_metric: str = "entropy"
    bd_lambda: float = 1.0
    bd_separation_floor: float = 1.0
    bd_eps: float = 1e-6
    bd_readout: str = "ordered_typicality"

    # 6. QA Branch (Quantile & Extremum Evidence)
    weight_qa: float = 0.0
    qa_dim: int = 32
    qa_quantiles: tuple[float, ...] = (0.05, 0.10, 0.90, 0.95)
    qa_lambda: float = 1.0

    # 7. DS Branch (In-Context Salience Denoised Bag-Mean)
    weight_ds: float = 0.0
    ds_dim: int = 32
    ds_lambda: float = 1.0
    ds_temperature: float = 1.0
    ds_tokens: int = 256

    # 8. DD Branch (Historical Data-Dependent Direction)
    weight_dd: float = 0.0
    dd_shrinkage: float = 0.25
    dd_eps: float = 1e-6
    dd_readout: str = "ordered_typicality"
    dd_separation_floor: float = 1.0

    # 9. Experimental Branches
    # DE (In-Subspace Dual Extreme Instance MIL)
    weight_de: float = 0.0
    de_dim: int = 32
    de_topk_fraction: float = 0.05
    de_topk_min: int = 4
    de_topk_max: int = 64
    de_lambda: float = 1.0

    # SW (Sliced Wasserstein Distribution Matching)
    weight_sw: float = 0.0
    sw_dim: int = 32
    sw_num_slices: int = 32
    sw_num_quantiles: int = 32
    sw_lambda: float = 1.0

    # LR (Direct Patch Likelihood Ratio + Top-K MIL)
    weight_lr: float = 0.0
    lr_dim: int = 32
    lr_lambda: float = 1.0
    lr_tau: float = 5.0
    lr_topk_fraction: float = 0.05
    lr_topk_min: int = 4
    lr_topk_max: int = 64
    lr_patches_per_ctx: int = 64

    # 10. Non-linear Kernel Ridge Regression (KRR) options
    krr_kernel: str = "linear"
    krr_gamma: float | None = None
    krr_degree: int = 2
    krr_coef0: float = 1.0

    # 11. Head Aggregation & Context LOO options
    aggregation: str = "soft_voting"
    loo_gamma: float = 2.0
    loo_floor: float = 0.50
    gated_tau: float = 0.05
    adaptive_tau: float = 0.08
    adaptive_ratio: float = 1.5

    @classmethod
    def from_yaml(cls, path_or_content: str | Path, strict: bool = True) -> TrainingFreeConfig:
        """Instantiate TrainingFreeConfig from a YAML file path or YAML string."""
        return from_yaml(path_or_content, strict=strict)

    @classmethod
    def from_dict(cls, d: dict[str, Any], strict: bool = True) -> TrainingFreeConfig:
        """Instantiate TrainingFreeConfig from a dictionary (flat or nested)."""
        return from_dict(d, strict=strict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary."""
        return to_dict(self)

    def to_yaml(self, path: Path | str | None = None) -> str:
        """Serialize configuration to a formatted YAML string, optionally writing to path."""
        return to_yaml(self, path=path)


# ---- Dictionary Normalization (Nested -> Flat) -------------------------------

_FIELD_NAMES = {f.name for f in fields(TrainingFreeConfig)}


def _flatten_dict(d: dict[str, Any], prefix: tuple[str, ...] = ()) -> dict[str, Any]:
    """Recursively flatten nested YAML dictionary into TrainingFreeConfig keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k in ("name", "description", "version"):
            continue  # Metadata fields
        current_path = prefix + (k,)
        if isinstance(v, dict):
            # Special handling for aggregation block: aggregation.method -> aggregation
            if current_path == ("aggregation",) and "method" in v:
                out["aggregation"] = v["method"]
                for subk, subv in v.items():
                    if subk != "method":
                        out[subk] = subv
                continue

            # Special handling for branches: branches.cv.weight -> weight_cv
            if len(current_path) == 2 and current_path[0] in ("branches", "experimental"):
                branch_name = current_path[1]
                for param_k, param_v in v.items():
                    if param_k == "weight":
                        out[f"weight_{branch_name}"] = param_v
                    elif param_k == "tokens" and branch_name == "ct":
                        out["ct_num_tokens"] = param_v
                    elif param_k == "method" and branch_name in ("bd", "ct"):
                        out[f"{branch_name}_metric" if branch_name == "bd" else f"{branch_name}_tokenizer"] = param_v
                    elif f"{branch_name}_{param_k}" in _FIELD_NAMES:
                        out[f"{branch_name}_{param_k}"] = param_v
                    elif param_k in _FIELD_NAMES:
                        out[param_k] = param_v
                    else:
                        out[f"{branch_name}_{param_k}"] = param_v
                continue

            # Basis block: basis.sketch_dim -> sketch_dim
            if current_path == ("basis",):
                for param_k, param_v in v.items():
                    out[param_k] = param_v
                continue

            # General nested traversal
            out.update(_flatten_dict(v, current_path))
        else:
            out[k] = v
    return out


# ---- Schema Reflection & Validation Helpers ----------------------------------

def _coerce_numeric_value(val: Any, target_type: type, field_name: str) -> Any:
    """Coerce numeric values handling scientific notation strings and float conversions."""
    if val is None:
        return None

    # Handle float fields
    if target_type is float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                raise ValueError(
                    f"Field '{field_name}' expects float, got unparseable string {val!r}"
                )
        raise TypeError(f"Field '{field_name}' expects float, got {type(val).__name__} ({val!r})")

    # Handle int fields
    if target_type is int:
        if isinstance(val, bool):
            raise TypeError(f"Field '{field_name}' expects int, got bool ({val!r})")
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            if val.is_integer():
                return int(val)
            raise ValueError(
                f"Field '{field_name}' expects int, got fractional float {val}"
            )
        if isinstance(val, str):
            try:
                f_val = float(val)
                if f_val.is_integer():
                    return int(f_val)
                raise ValueError(
                    f"Field '{field_name}' expects int, got fractional string {val!r}"
                )
            except ValueError:
                raise ValueError(
                    f"Field '{field_name}' expects int, got unparseable string {val!r}"
                )
        raise TypeError(f"Field '{field_name}' expects int, got {type(val).__name__} ({val!r})")

    # Handle string fields (prevent bool 'on'/'off' from becoming boolean)
    if target_type is str:
        if isinstance(val, bool):
            raise TypeError(
                f"Field '{field_name}' expects string, but received boolean {val!r}. "
                "Quote YAML strings like 'off' or 'yes' to prevent boolean parsing."
            )
        return str(val)

    # Handle bool fields
    if target_type is bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            if val in (0, 1):
                return bool(val)
        if isinstance(val, str):
            lower = val.lower().strip()
            if lower in ("true", "1", "yes"):
                return True
            if lower in ("false", "0", "no"):
                return False
        raise TypeError(f"Field '{field_name}' expects bool, got {type(val).__name__} ({val!r})")

    return val


def _coerce_field_value(val: Any, field_type: Any, field_name: str) -> Any:
    """Recursively coerce value to declared dataclass field type."""
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is tuple:
        if not isinstance(val, (list, tuple)):
            raise TypeError(
                f"Field '{field_name}' expects sequence, got {type(val).__name__} ({val!r})"
            )
        elem_type = args[0] if args else float
        return tuple(_coerce_numeric_value(x, elem_type, f"{field_name}[]") for x in val)

    is_union = origin is not None and getattr(origin, "__name__", "") in ("Union", "UnionType")
    if is_union:
        if val is None:
            if type(None) in args:
                return None
            raise ValueError(f"Field '{field_name}' does not accept None")

        # Try matching types in order: int, float, str, bool
        for cand in (int, float, str, bool):
            if cand in args:
                try:
                    return _coerce_numeric_value(val, cand, field_name)
                except (ValueError, TypeError):
                    continue
        return val

    return _coerce_numeric_value(val, field_type, field_name)


# ---- Validation Functions ----------------------------------------------------

def _validate_config_domain(cfg: TrainingFreeConfig) -> None:
    """Validate numerical bounds, enum options, and mathematical constraints."""
    # 1. Enums
    if not (cfg.aggregation in VALID_AGGREGATIONS or cfg.aggregation.startswith("context_loo")):
        raise ValueError(
            f"Invalid aggregation '{cfg.aggregation}'. Allowed: {sorted(VALID_AGGREGATIONS)} or 'context_loo*'"
        )
    if cfg.cv_blocks not in VALID_CV_BLOCKS:
        raise ValueError(f"Invalid cv_blocks '{cfg.cv_blocks}'. Allowed: {sorted(VALID_CV_BLOCKS)}")
    if cfg.bd_metric not in VALID_BD_METRICS:
        raise ValueError(f"Invalid bd_metric '{cfg.bd_metric}'. Allowed: {sorted(VALID_BD_METRICS)}")
    if cfg.bd_readout not in VALID_BD_READOUTS:
        raise ValueError(f"Invalid bd_readout '{cfg.bd_readout}'. Allowed: {sorted(VALID_BD_READOUTS)}")
    if cfg.ct_tokenizer not in VALID_CT_TOKENIZERS:
        raise ValueError(f"Invalid ct_tokenizer '{cfg.ct_tokenizer}'. Allowed: {sorted(VALID_CT_TOKENIZERS)}")
    if cfg.ct_readout not in VALID_CT_READOUTS:
        raise ValueError(f"Invalid ct_readout '{cfg.ct_readout}'. Allowed: {sorted(VALID_CT_READOUTS)}")
    if cfg.krr_kernel not in VALID_KRR_KERNELS:
        raise ValueError(f"Invalid krr_kernel '{cfg.krr_kernel}'. Allowed: {sorted(VALID_KRR_KERNELS)}")

    # 2. Numeric bounds
    if cfg.sketch_dim <= 0:
        raise ValueError(f"sketch_dim must be positive, got {cfg.sketch_dim}")
    if cfg.bm_dim <= 0:
        raise ValueError(f"bm_dim must be positive, got {cfg.bm_dim}")
    if cfg.bd_dim <= 0:
        raise ValueError(f"bd_dim must be positive, got {cfg.bd_dim}")
    if cfg.qa_dim <= 0:
        raise ValueError(f"qa_dim must be positive, got {cfg.qa_dim}")
    if cfg.ds_dim <= 0:
        raise ValueError(f"ds_dim must be positive, got {cfg.ds_dim}")

    # Weights >= 0
    for w_name in ("weight_cv", "weight_ct", "weight_bm", "weight_bd", "weight_qa", "weight_ds", "weight_dd"):
        w = getattr(cfg, w_name)
        if w < 0.0:
            raise ValueError(f"{w_name} must be non-negative, got {w}")

    # Quantiles in (0, 1) and sorted
    if cfg.qa_quantiles:
        prev = 0.0
        for q in cfg.qa_quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError(f"qa_quantiles elements must be in (0, 1), got {q}")
            if q <= prev:
                raise ValueError(f"qa_quantiles must be strictly ascending, got {cfg.qa_quantiles}")
            prev = q


# ---- Public Parsing & Serialization API --------------------------------------

def from_dict(d: dict[str, Any], strict: bool = True) -> TrainingFreeConfig:
    """Create a validated TrainingFreeConfig from a flat or nested dictionary.

    Args:
        d: Raw configuration dictionary.
        strict: If True, raises ValueError for unknown/typo keys.
    """
    flat = _flatten_dict(d)
    type_hints = get_type_hints(TrainingFreeConfig)

    # Check for unknown keys
    if strict:
        unknown = set(flat.keys()) - _FIELD_NAMES
        if unknown:
            first = sorted(unknown)[0]
            matches = difflib.get_close_matches(first, _FIELD_NAMES, n=1)
            suggestion = f". Did you mean '{matches[0]}'?" if matches else ""
            raise ValueError(f"Unknown config key '{first}'{suggestion}")

    coerced: dict[str, Any] = {}
    for k, v in flat.items():
        if k in _FIELD_NAMES:
            coerced[k] = _coerce_field_value(v, type_hints[k], k)

    cfg = TrainingFreeConfig(**coerced)
    _validate_config_domain(cfg)
    return cfg


def from_yaml(path_or_content: str | Path, strict: bool = True) -> TrainingFreeConfig:
    """Load TrainingFreeConfig from a YAML file path or YAML content string."""
    if isinstance(path_or_content, Path):
        content = path_or_content.read_text(encoding="utf-8")
    elif isinstance(path_or_content, str) and "\n" not in path_or_content and Path(path_or_content).exists():
        content = Path(path_or_content).read_text(encoding="utf-8")
    else:
        content = str(path_or_content)

    data = yaml.load(content, Loader=_ScientificSafeLoader)
    if not isinstance(data, dict):
        raise ValueError(f"YAML content must evaluate to a mapping, got {type(data).__name__}")
    return from_dict(data, strict=strict)


def to_dict(config: TrainingFreeConfig) -> dict[str, Any]:
    """Convert TrainingFreeConfig to a clean dictionary."""
    return asdict(config)


def to_yaml(config: TrainingFreeConfig, path: Path | str | None = None) -> str:
    """Serialize TrainingFreeConfig to structured YAML format."""
    d = asdict(config)
    structured = {
        "name": "training_free_config",
        "basis": {
            "sketch_dim": d["sketch_dim"],
            "ridge_lambda": d["ridge_lambda"],
            "ridge_scale": d["ridge_scale"],
        },
        "branches": {
            "cv": {
                "weight": d["weight_cv"],
                "blocks": d["cv_blocks"],
            },
            "bm": {
                "weight": d["weight_bm"],
                "dim": d["bm_dim"],
                "lambda": d["bm_lambda"],
            },
            "bd": {
                "weight": d["weight_bd"],
                "dim": d["bd_dim"],
                "metric": d["bd_metric"],
                "readout": d["bd_readout"],
                "lambda": d["bd_lambda"],
                "separation_floor": d["bd_separation_floor"],
                "eps": d["bd_eps"],
            },
            "qa": {
                "weight": d["weight_qa"],
                "dim": d["qa_dim"],
                "quantiles": list(d["qa_quantiles"]),
                "lambda": d["qa_lambda"],
            },
            "ds": {
                "weight": d["weight_ds"],
                "dim": d["ds_dim"],
                "tokens": d["ds_tokens"],
                "temperature": d["ds_temperature"],
                "lambda": d["ds_lambda"],
            },
            "ct": {
                "weight": d["weight_ct"],
                "pca_dim": d["ct_pca_dim"],
                "tokens": d["ct_num_tokens"],
                "tokenizer": d["ct_tokenizer"],
                "readout": d["ct_readout"],
                "temperature": d["ct_temperature"],
                "eps": d["ct_eps"],
            },
            "dd": {
                "weight": d["weight_dd"],
                "shrinkage": d["dd_shrinkage"],
                "eps": d["dd_eps"],
                "readout": d["dd_readout"],
                "separation_floor": d["dd_separation_floor"],
            },
        },
        "aggregation": {
            "method": d["aggregation"],
            "loo_gamma": d["loo_gamma"],
            "loo_floor": d["loo_floor"],
            "gated_tau": d["gated_tau"],
            "adaptive_tau": d["adaptive_tau"],
            "adaptive_ratio": d["adaptive_ratio"],
        },
    }

    yaml_str = yaml.dump(structured, sort_keys=False, default_flow_style=False)
    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml_str, encoding="utf-8")
    return yaml_str
