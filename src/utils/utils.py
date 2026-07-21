from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
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
    config = load_yaml(config_path)
    dataset_config = resolve_config_group("dataset", config.get("dataset"))

    merged = dict(config)
    merged["data"] = deep_merge(
        dataset_config,
        resolve_config_group("data", config.get("data")),
    )
    merged["model"] = resolve_config_group("model", config.get("model"))
    merged["optimizer"] = resolve_config_group("optimizer", config.get("optimizer"))
    merged["scheduler"] = resolve_config_group("scheduler", config.get("scheduler"))
    merged["trainer"] = resolve_config_group("trainer", config.get("trainer"))
    merged["logger"] = resolve_config_group("logger", config.get("logger"))
    merged["callbacks"] = resolve_config_group("callbacks", config.get("callbacks"))
    return merged


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
