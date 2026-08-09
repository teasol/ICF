from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.modules.data_interface import DataInterface
from src.modules.model_interface import ModelInterface


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AlwaysSaveLastModelCheckpoint(ModelCheckpoint):
    """Keep ``last.ckpt`` current even when the epoch misses the top-k.

    Lightning 2.5 only calls ``_save_last_checkpoint`` after a top-k checkpoint
    was saved in the same hook.  With a monitored top-k callback this left
    ``last.ckpt`` pointing at the best epoch throughout a long plateau, which
    made interruption recovery silently lose many epochs.
    """

    def _save_last_if_due(self, trainer: L.Trainer) -> None:
        if not self.save_last or self._last_global_step_saved == trainer.global_step:
            return
        if self._every_n_epochs < 1:
            return
        if (trainer.current_epoch + 1) % self._every_n_epochs != 0:
            return
        self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        super().on_validation_end(trainer, pl_module)
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and not self._should_save_on_train_epoch_end(trainer)
        ):
            self._save_last_if_due(trainer)

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._should_save_on_train_epoch_end(trainer)
        ):
            self._save_last_if_due(trainer)


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TIRANOS model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train.yaml",
        help="Path to the training config yaml.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the merged config and exit.",
    )
    parser.add_argument("--seed", type=int, help="Override the training and dataset seed.")
    parser.add_argument("--cv", type=int, help="Override the dataset cross-validation fold.")
    parser.add_argument("--run-name", help="Override the logger run name.")
    parser.add_argument("--run-group", help="Override the logger run group.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Override the checkpoint output directory.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        help="Resume model, optimizer, scheduler, and loop state from a checkpoint.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Load model weights only; optimizer, scheduler, and loop state start fresh.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config_group(group: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return load_yaml(PROJECT_ROOT / "configs" / group / f"{value}.yaml")
    raise TypeError(f"{group} config must be a name or mapping, got {type(value).__name__}.")


def merge_train_config(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _load_train_config(config_path, ())
    dataset_config = resolve_config_group("dataset", config.get("dataset"))

    merged = dict(config)
    merged["data"] = deep_merge(
        dataset_config,
        resolve_config_group("data", config.get("data")),
    )
    merged["data"] = deep_merge(merged["data"], config.get("data_overrides", {}))
    merged.pop("data_overrides", None)
    merged["model"] = resolve_config_group("model", config.get("model"))
    merged["model"] = deep_merge(merged["model"], config.get("model_overrides", {}))
    merged.pop("model_overrides", None)
    merged["optimizer"] = resolve_config_group("optimizer", config.get("optimizer"))
    merged["scheduler"] = resolve_config_group("scheduler", config.get("scheduler"))
    merged["trainer"] = resolve_config_group("trainer", config.get("trainer"))
    merged["trainer"] = resolve_config_group("trainer", config.get("trainer"))
    merged["trainer"] = deep_merge(
        merged["trainer"], config.get("trainer_overrides", {})
    )
    merged.pop("trainer_overrides", None)
    merged["logger"] = resolve_config_group("logger", config.get("logger"))
    merged["logger"] = deep_merge(
        merged["logger"], config.get("logger_overrides", {})
    )
    merged.pop("logger_overrides", None)
    merged["callbacks"] = resolve_config_group("callbacks", config.get("callbacks"))
    return merged


def _load_train_config(
    config_path: Path,
    stack: tuple[Path, ...],
) -> dict[str, Any]:
    """Load an optional train-config base before resolving config groups."""
    if config_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Circular base_config chain: {cycle}")
    config = load_yaml(config_path)
    base_value = config.pop("base_config", None)
    if base_value is None:
        return config
    base_path = Path(base_value)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = _load_train_config(base_path.resolve(), (*stack, config_path))
    return deep_merge(base, config)


def build_datamodule(config: dict[str, Any]) -> DataInterface:
    data_config: dict[str, Any] = config.get("data", {})
    return DataInterface(**data_config)


def build_model(config: dict[str, Any]) -> ModelInterface:
    model_config: dict[str, Any] = config.get("model", {})
    optimizer_config: dict[str, Any] = config.get("optimizer", {})
    scheduler_config: dict[str, Any] = config.get("scheduler", {})
    model_kwargs: dict[str, Any] = deep_merge(
        config.get("model_kwargs", {}),
        model_config.get("kwargs", {}),
    )
    model_kwargs = deep_merge(
        {key: value for key, value in model_config.items() if key != "kwargs"},
        model_kwargs,
    )
    model_kwargs = deep_merge(model_kwargs, optimizer_config)
    model_kwargs = deep_merge(model_kwargs, scheduler_config)
    return ModelInterface(**model_kwargs)


DEFAULT_EVAL_PRECISION = "bf16-mixed"
SUPPORTED_EVAL_PRECISIONS = ("bf16-mixed", "32-true")


def eval_autocast(
    device: torch.device | str,
    precision: str = DEFAULT_EVAL_PRECISION,
):
    """Autocast context for an inference pass (agent_handoff SS3.4).

    The bf16-mixed contract covers evaluation as well as training (user
    decision, 2026-08-08). `evaluate_synthetic.py` already ran eval under bf16
    autocast while `test_pathobench.py` / `test_musk.py` ran fp32, so the same
    checkpoint was scored under two different precisions depending on which
    script you called; this helper is the single definition.

    ``32-true`` stays available as an escape hatch for reproducing the fp32-era
    numbers (every official 50-fold AUROC recorded before 2026-08-08). fp16 is
    rejected outright: fp16 coefficients overflow the covariance-sketch inverse
    and the ridge solves to NaN, which is the whole reason for SS3.4.

    CPU autocast is skipped -- bf16 matmul on CPU is emulated and would make
    unit tests both slow and needlessly lossy.
    """
    if precision not in SUPPORTED_EVAL_PRECISIONS:
        raise ValueError(
            f"Unsupported eval precision {precision!r}; expected one of "
            f"{SUPPORTED_EVAL_PRECISIONS}. fp16 ('16-mixed') is forbidden by "
            "the numerical-safety contract (docs/agent_handoff.md SS3.4)."
        )
    device = torch.device(device)
    if precision == "32-true" or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def add_eval_precision_argument(parser: argparse.ArgumentParser) -> None:
    """Register the shared `--precision` flag on an evaluation script."""
    parser.add_argument(
        "--precision",
        type=str,
        default=DEFAULT_EVAL_PRECISION,
        choices=list(SUPPORTED_EVAL_PRECISIONS),
        help=(
            "Inference precision (agent_handoff SS3.4). Default bf16-mixed is "
            "the enforced contract; 32-true reproduces the fp32-era numbers "
            "reported before 2026-08-08."
        ),
    )


def estimate_training_vram_bytes(
    *,
    num_bags_max: int,
    max_cells_per_bag: int,
    input_dim: int,
    param_count: int,
    episode_batch_size: int = 1,
    activation_layers: int = 6,
    backward_factor: float = 1.0,
    context_overhead_bytes: int = 1 << 30,
) -> int:
    """Conservative worst-case resident bytes for a single training step.

    Covers, for the biggest possible optimizer step:
      * bf16 weights + two fp32 Adam moments + fp32 gradients;
      * the dense episode buffer at
        ``episode_batch_size x num_bags_max x max_cells``;
      * the per-cell activation chain (input tensor plus ``activation_layers``
        layer tensors) scaled by a backward-retention factor;
      * a fixed CUDA/context overhead.

    ``episode_batch_size`` matters because peak memory scales with the TOTAL
    cells in a step, not per episode: v34-1536 (batch 4 x 100 bags x 8192
    cells) and v35 (batch 1 x 100 x 32768) both reach 3.28M cells and both peak
    around 112-122 GiB. Omitting it made the bound blind to the batch dimension
    entirely, so a 4x batch increase looked free.

    The multiplier is calibrated against measured peaks rather than guessed:
    ``(1 + activation_layers) * backward_factor = 7`` per input-tensor byte,
    versus measured 6.0x for v34-1536 (112 GiB peak / 18.8 GiB of cells) and
    6.5x for v35 (122.4 GiB, `torch.cuda.max_memory_allocated` after
    forward+backward+step at exactly 100 x 32768). That leaves ~1.25x headroom
    and still blocks genuinely oversized configs (batch 4 x 32768 cells would
    estimate ~600 GiB). The previous default of ``backward_factor=3.0`` implied
    21x, which over-estimated by ~3.4x and -- combined with the missing batch
    term -- happened to roughly cancel out for v34 while rejecting v35.
    """
    weights_bytes = param_count * 2
    adam_bytes = param_count * 8
    gradient_bytes = param_count * 4
    batch = max(1, int(episode_batch_size))
    dense_buffer_bytes = (
        batch * num_bags_max * max_cells_per_bag * input_dim * 4
    )
    worst_cells = batch * num_bags_max * max_cells_per_bag
    activation_bytes = (
        worst_cells
        * input_dim
        * 4
        * (1 + int(activation_layers))
        * float(backward_factor)
    )
    return int(
        weights_bytes
        + adam_bytes
        + gradient_bytes
        + dense_buffer_bytes
        + activation_bytes
        + context_overhead_bytes
    )


def validate_vram_budget(
    config: dict[str, Any],
    model: Any,
    *,
    verbose: bool = True,
) -> None:
    """Fail fast if a worst-case training step could OOM the visible GPU.

    Uses conservative worst-case bounds (max bags, max cells) so the check is
    robust against a future config that scales up ``num_bags``/``num_cells``.
    Only active when CUDA is available; CPU runs (tests, smoke) are unaffected.
    Warning/hard-error thresholds are tunable via ``trainer.vram_warn_fraction``
    and ``trainer.vram_error_fraction`` (defaults 0.6 / 0.9).
    """
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    total = properties.total_memory

    dataset_kwargs = config.get("data", {}).get("dataset_kwargs", {})
    num_bags = dataset_kwargs.get("num_bags", (60, 100))
    num_bags_max = (
        max(num_bags) if isinstance(num_bags, (list, tuple)) else int(num_bags)
    )
    num_cells = dataset_kwargs.get("num_cells", (500, 1000))
    num_cells_max = (
        max(num_cells) if isinstance(num_cells, (list, tuple)) else int(num_cells)
    )
    input_dim = int(config.get("model", {}).get("input_dim", 512))
    param_count = sum(parameter.numel() for parameter in model.parameters())
    episode_batch_size = int(
        config.get("data", {}).get("episode_batch_size", 1) or 1
    )

    # How many per-cell activation tensors the model actually keeps. The
    # 6-layer default was calibrated on the full-branch v34/v35 model, which no
    # longer exists (SS73); applying it to a model that keeps one per-cell
    # transform over-estimates by ~3.4x and rejects configurations that fit.
    #
    # The MODEL declares this, rather than the guard inferring it from a config
    # flag. The flag it used to read (`meta_covariance_only`) was deleted along
    # with the branches it selected, at which point the guard silently fell back
    # to 6 for every model -- a config key is the wrong place to record a fact
    # about an architecture.
    #
    # Measured peaks (60 bags x 16384 cells, forward+backward):
    #     ep_batch      1            2           4
    #     CV-only     14,720 MiB   29,272     58,312
    #     full        50,527 MiB  100,789        OOM
    # ratio 14720/50527 = 0.291; (1+1)/(1+6) = 0.286 -- activation_layers=1
    # reproduces the measurement rather than being fitted to it. The learned
    # bag-token model (SS75) measures 21.20 GiB at 100 x 16384, which the same
    # activation_layers=1 covers at 29.15 GiB.
    activation_layers = int(getattr(model, "vram_activation_layers", 6))
    estimate = estimate_training_vram_bytes(
        num_bags_max=num_bags_max,
        max_cells_per_bag=num_cells_max,
        input_dim=input_dim,
        param_count=param_count,
        episode_batch_size=episode_batch_size,
        activation_layers=activation_layers,
    )
    fraction = estimate / total

    trainer_config = config.get("trainer", {})
    warn_fraction = float(trainer_config.get("vram_warn_fraction", 0.6))
    error_fraction = float(trainer_config.get("vram_error_fraction", 0.9))
    if fraction > error_fraction:
        raise RuntimeError(
            f"[vram] worst-case step estimate {estimate / 1e9:.1f} GiB "
            f"({fraction:.0%}) exceeds the hard limit "
            f"{error_fraction:.0%} of {properties.name} "
            f"({total / 1e9:.0f} GiB). Reduce num_bags/num_cells "
            "(or episode batch) before running."
        )
    if verbose:
        print(
            f"[vram] worst-case estimate {estimate / 1e9:.2f} GiB "
            f"({fraction:.1%} of {properties.name} {total / 1e9:.0f} GiB)"
            + (" -- OK" if fraction < warn_fraction else " -- caution")
        )


def initialize_model_weights(
    model: ModelInterface, checkpoint_path: str | Path
) -> tuple[list[str], list[str]]:
    """Load compatible weights without invoking Lightning resume semantics.

    Architecture-version metadata is intentionally skipped so an additive,
    zero-output branch can warm-start from its immediate predecessor.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint has no state dictionary: {path}")
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key != "model._architecture_version"
    }
    incompatible = model.load_state_dict(compatible_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    allowed_missing = {
        key
        for key in missing
        if key == "model._architecture_version"
    }
    disallowed_missing = sorted(set(missing) - allowed_missing)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Weight-only initialization found incompatible keys: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    return missing, unexpected


def build_logger(config: dict[str, Any]):
    logger_config: dict[str, Any] = config.get("logger", {})
    logger_name: str | None = logger_config.get("name", "tensorboard")
    save_dir: str = logger_config.get("save_dir", "logs")
    experiment_name: str = logger_config.get("experiment_name", "tiranos")

    if logger_name == "csv":
        return CSVLogger(save_dir=save_dir, name=experiment_name)
    if logger_name in ("tensorboard", "tb"):
        return TensorBoardLogger(save_dir=save_dir, name=experiment_name)
    if logger_name in ("wandb", "weights_and_biases"):
        from lightning.pytorch.loggers import WandbLogger

        run_name: str = logger_config.get("run_name") or (
            f"{logger_config.get('run_name_prefix', 'v1')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        wandb_kwargs: dict[str, Any] = {
            key: value
            for key, value in logger_config.items()
            if key not in (
                "name",
                "save_dir",
                "experiment_name",
                "run_name",
                "run_name_prefix",
            )
        }
        return WandbLogger(save_dir=save_dir, name=run_name, **wandb_kwargs)
    if logger_name in ("none", None):
        return False
    raise ValueError(f"Unsupported logger: {logger_name}")


def build_callbacks(config: dict[str, Any]) -> list[Any]:
    callbacks_config: dict[str, Any] = config.get("callbacks", {})
    callbacks: list[Any] = []

    checkpoint_config: dict[str, Any] = callbacks_config.get("checkpoint", {})
    if checkpoint_config.get("enabled", True):
        callbacks.append(
            AlwaysSaveLastModelCheckpoint(
                dirpath=checkpoint_config.get("dirpath", "checkpoints"),
                filename=checkpoint_config.get("filename", "{epoch:03d}-{val_loss:.4f}"),
                monitor=checkpoint_config.get("monitor", "val_loss"),
                mode=checkpoint_config.get("mode", "min"),
                save_top_k=checkpoint_config.get("save_top_k", 3),
                save_last=checkpoint_config.get("save_last", True),
            )
        )

    lr_monitor_config: dict[str, Any] = callbacks_config.get("lr_monitor", {})
    if lr_monitor_config.get("enabled", True):
        callbacks.append(
            LearningRateMonitor(logging_interval=lr_monitor_config.get("logging_interval", "epoch"))
        )

    return callbacks


def build_trainer(config: dict[str, Any]) -> L.Trainer:
    trainer_kwargs: dict[str, Any] = config.get("trainer", {})
    trainer_kwargs.setdefault("max_epochs", 1)
    trainer_kwargs.setdefault("accelerator", "auto")
    trainer_kwargs.setdefault("devices", "auto")

    return L.Trainer(
        **trainer_kwargs,
        logger=build_logger(config),
        callbacks=build_callbacks(config),
    )
