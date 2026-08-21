"""Model registry and factory for ICF bag classifiers."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, TypeVar
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(*names: str) -> Callable[[type[M]], type[M]]:
    """Decorator to register a model class under one or more alias names."""

    def decorator(cls: type[M]) -> type[M]:
        for name in names:
            key = name.lower().strip()
            _MODEL_REGISTRY[key] = cls
            # Also register raw name
            _MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_model_class(name_or_path: str) -> type[nn.Module]:
    """Resolve a model class from registry name or full module path."""
    if not isinstance(name_or_path, str) or not name_or_path.strip():
        raise ValueError(f"Invalid model name or path: {name_or_path!r}")

    # Check exact registry lookup
    if name_or_path in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[name_or_path]

    # Check lower-case lookup
    lower = name_or_path.lower().strip()
    if lower in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[lower]

    # Fallback to dynamic import for full module path
    if "." in name_or_path:
        module_name, class_name = name_or_path.rsplit(".", 1)
        module = import_module(module_name)
        cls = getattr(module, class_name)
        if not issubclass(cls, nn.Module):
            raise TypeError(f"Imported class {cls} is not a subclass of nn.Module")
        # Cache in registry
        _MODEL_REGISTRY[name_or_path] = cls
        _MODEL_REGISTRY[lower] = cls
        return cls

    available = sorted(set(_MODEL_REGISTRY.keys()))
    raise KeyError(
        f"Model {name_or_path!r} not found in registry. "
        f"Available models: {available[:10]}..."
    )


def list_registered_models() -> list[str]:
    """Return a sorted list of all unique registered model names."""
    return sorted(set(_MODEL_REGISTRY.keys()))


def build_model(
    model_src_or_config: str | dict[str, Any], *args: Any, **kwargs: Any
) -> nn.Module:
    """Factory to instantiate a model from registry name or config dict."""
    if isinstance(model_src_or_config, dict):
        config = dict(model_src_or_config)
        model_src = config.pop("model_src", None)
        if model_src is None:
            raise ValueError("Config dict must contain 'model_src' key.")
        # Merge remaining config entries with kwargs (kwargs takes precedence)
        merged_kwargs = {**config, **kwargs}
        model_cls = get_model_class(model_src)
        return model_cls(*args, **merged_kwargs)

    model_cls = get_model_class(model_src_or_config)
    return model_cls(*args, **kwargs)
